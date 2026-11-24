// Ufunc hook: the call pointer inside each of NumPy's ufunc objects, and the
// lane behind it. See docs/ufunc-lane-design.md.
//
// `np.add`, `np.exp` and the rest are objects of type np.ufunc, each holding
// the function CPython calls them through (the vectorcall protocol), at the
// offset `PyUFunc_Type` publishes as `tp_vectorcall_offset`. `a + b`, `a > 0`,
// `-a` and `a += b` call the same objects through the same pointer, from
// ndarray's number slots. The hook points every such object at `call_lane`.
//
// A call whose operands are plain contiguous arrays of one shape, a matrix and
// a row of it, or an array and a number, takes the lane: the binding
// allocates the output, or takes the one it was given, and calls the inner
// loop NumPy itself would have called. Which loop that is, the binding asks
// NumPy once per ufunc and operand kinds, through the interface NumPy offers
// other compilers for this (`ufunc._resolve_dtypes_and_context` and
// `ufunc._get_strided_loop`), and keeps the answer in a table. The loop and
// its arguments are NumPy's, so the result is NumPy's. Floating-point
// exceptions the loop raises go to NumPy's own handler, which warns, raises
// or does nothing as np.errstate says. Every other call goes to NumPy's
// pointer with its arguments untouched.
#include "binding.h"

// The arguments of a vectorcall: the callable, the positional arguments
// followed by the keyword values, the positional count with a flag bit, and
// the keyword names.
#define HK_VECTORCALL_PARAMETERS \
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames

// Most elements a call's result may have to take the lane. Above it the call
// path the lane skips, about 150 ns, is under 5% of every call measured
// (docs/ufunc-lane-design.md).
#define LANE_MAX_ELEMENTS 16384
// NumPy releases the interpreter lock for a loop over more elements than this
// (NPY_BEGIN_THREADS_THRESHOLDED in numpy/ndarraytypes.h); the lane does the
// same at the same count.
#define LANE_THREADS_ELEMENTS 500
// NumPy's default buffer size, in elements (NPY_BUFSIZE; np.getbufsize()).
// For an array and a row of it, NumPy's iterator copies the row, repeated,
// into a buffer and calls the loop once per chunk of whole rows that fits in
// it. The lane does the same, and takes such a call only when NumPy makes it
// in one chunk.
#define LANE_BUFFER_ELEMENTS 8192
// Entries of the table of loops: every ufunc and operand-kind combination a
// program uses, with room to spare. A combination that finds neither its
// entry nor a free one within LANE_PROBES of its slot takes the place of the
// entry in its slot, which is asked for again when next needed.
#define LANE_TABLE_SIZE 2048
#define LANE_PROBES 8
// Name of the capsule `ufunc._resolve_dtypes_and_context` returns. The name
// carries the layout of what it points to (lane_call_info), which NumPy may
// change in a later release; a capsule of another name leaves the hook off.
#define LANE_CAPSULE "numpy_1.24_ufunc_call_info"
// Bytes of the buffer a number is converted into: one element of the widest
// type the lane takes.
#define LANE_SCALAR_BYTES 8

// Operand kinds beyond an array's type number. A NumPy number of type t is
// KIND_NUMPY_SCALAR + t; a Python float and int have a kind each. An array's
// kind is its type number, below KIND_NUMPY_SCALAR.
#define KIND_NUMPY_SCALAR 32
#define KIND_PYTHON_FLOAT 64
#define KIND_PYTHON_INT 65
// The kind of the second operand of a one-operand ufunc.
#define KIND_NONE 66
// Not an operand the lane takes.
#define KIND_DECLINED (-1)

// What `ufunc._resolve_dtypes_and_context` returns in its capsule, and
// `ufunc._get_strided_loop` fills in, as NumPy documents it for the capsule
// named LANE_CAPSULE (the docstring of `np.ufunc._get_strided_loop`).
typedef struct {
    // The inner loop.
    PyArrayMethod_StridedLoop *strided_loop;
    // Passed to the loop as its first argument.
    PyArrayMethod_Context *context;
    // Passed to the loop as its last argument.
    NpyAuxData *auxdata;
    // Whether the loop needs the interpreter lock and the Python API.
    npy_bool requires_pyapi;
    // Whether the loop never raises floating-point exceptions, so the flags
    // need not be tested after it.
    npy_bool no_floatingpoint_errors;
} lane_call_info;

// What a table entry says about its combination.
typedef enum {
    // A free entry.
    LANE_FREE = 0,
    // NumPy is being asked; a call that meets it, from inside NumPy's
    // answer, goes to NumPy.
    LANE_RESOLVING,
    // The lane serves the combination with the loop below.
    LANE_READY,
    // The lane does not serve it: NumPy's loop needs a cast of an array, the
    // Python API, or an output type the lane does not take.
    LANE_DECLINED,
} lane_state;

// One ufunc and operand-kind combination and what NumPy answered for it.
typedef struct {
    // The ufunc object. Kept alive by `hooked`.
    PyObject *ufunc;
    // Kinds of the first and second operand.
    int kind0, kind1;
    lane_state state;
    // Type number a number among the operands is converted to before the loop.
    int scalar_type;
    // The output's type. A reference is held.
    PyArray_Descr *out_descr;
    // The capsule NumPy returned, which owns the loop's data. A reference is held.
    PyObject *capsule;
    // Inside the capsule: the loop and its data.
    lane_call_info *info;
    // The ufunc's legacy loop for the resolved types, as an index into its
    // `functions` and the function it held when NumPy was asked, or -1 for a
    // loop that is not a legacy one. NumPy's loop for these types wraps that
    // function, and PyUFunc_ReplaceLoopBySignature can replace it at any
    // time, after which NumPy calls the new one: an entry whose function has
    // changed is asked for again.
    int legacy_index;
    PyUFuncGenericFunction legacy_function;
} lane_entry;

static lane_entry lane_table[LANE_TABLE_SIZE];

// The ufunc objects whose call pointer is the binding's, while the hook is
// applied: a list that holds a reference to each.
static PyObject *hooked;
// NumPy's own call function, the same for every ufunc NumPy creates
// (`ufunc_generic_vectorcall`), kept while the hook is applied.
static vectorcallfunc lane_original;
// The ufuncs for which NumPy 2.5 deprecates an output passed positionally,
// and warns: `np.maximum(a, b, out)`. A call of one of them with a positional
// output goes to NumPy, so that it warns. Borrowed from `hooked`.
#define LANE_POSITIONAL_DEPRECATED 2
static const char *const POSITIONAL_DEPRECATED_NAMES[LANE_POSITIONAL_DEPRECATED] = {"maximum", "minimum"};
static PyObject *positional_deprecated[LANE_POSITIONAL_DEPRECATED];

// ---------------------------------------------------------------- operands

// Element type numbers the lane takes: bool, every integer type and the two
// float types. float16, complex, long double, strings, dates and objects go
// to NumPy.
static inline bool lane_type(int type_num) {
    return type_num == NPY_BOOL || (type_num >= NPY_BYTE && type_num <= NPY_ULONGLONG) || type_num == NPY_FLOAT ||
           type_num == NPY_DOUBLE;
}

// The type number of a NumPy number of exactly one of the types the lane
// takes, or -1 for anything else.
static inline int numpy_scalar_type(PyObject *obj) {
    PyTypeObject *type = Py_TYPE(obj);
    if (type == &PyDoubleArrType_Type) return NPY_DOUBLE;
    if (type == &PyFloatArrType_Type) return NPY_FLOAT;
    if (type == &PyLongArrType_Type) return NPY_LONG;
    if (type == &PyBoolArrType_Type) return NPY_BOOL;
    if (type == &PyIntArrType_Type) return NPY_INT;
    if (type == &PyLongLongArrType_Type) return NPY_LONGLONG;
    if (type == &PyULongArrType_Type) return NPY_ULONG;
    if (type == &PyULongLongArrType_Type) return NPY_ULONGLONG;
    if (type == &PyUIntArrType_Type) return NPY_UINT;
    if (type == &PyShortArrType_Type) return NPY_SHORT;
    if (type == &PyUShortArrType_Type) return NPY_USHORT;
    if (type == &PyByteArrType_Type) return NPY_BYTE;
    if (type == &PyUByteArrType_Type) return NPY_UBYTE;
    return -1;
}

// True for an exact ndarray the lane can read or write directly: with at
// least one dimension, C-contiguous, aligned, in native byte order, of a
// type the lane takes and with at most LANE_MAX_ELEMENTS elements.
static inline bool lane_array_fits(PyArrayObject *array) {
    const int needed = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
    if ((PyArray_FLAGS(array) & needed) != needed || PyArray_NDIM(array) < 1) return false;
    const npy_intp n = PyArray_SIZE(array);
    if (n < 1 || n > LANE_MAX_ELEMENTS) return false;
    return lane_type(PyArray_TYPE(array)) && PyArray_ISNBO(PyArray_DESCR(array)->byteorder);
}

// The same for any object: false for anything but an exact ndarray.
static inline bool lane_array(PyObject *obj) {
    return Py_TYPE(obj) == hk_ndarray_type && lane_array_fits((PyArrayObject *)obj);
}

// The kind of one input, or KIND_DECLINED. The checks go from the commonest
// operand to the rarest, and an array is settled by its own checks alone, so
// that a call the lane declines costs as little as it can.
static inline int operand_kind(PyObject *obj) {
    PyTypeObject *type = Py_TYPE(obj);
    if (type == hk_ndarray_type) {
        return lane_array_fits((PyArrayObject *)obj) ? PyArray_TYPE((PyArrayObject *)obj) : KIND_DECLINED;
    }
    // Exact types only: bool is an int subclass and np.float64 a float subclass.
    if (type == &PyFloat_Type) return KIND_PYTHON_FLOAT;
    if (type == &PyLong_Type) return KIND_PYTHON_INT;
    // An ndarray subclass, a masked array among them, is no NumPy number.
    if (PyType_IsSubtype(type, hk_ndarray_type)) return KIND_DECLINED;
    const int type_num = numpy_scalar_type(obj);
    return type_num < 0 ? KIND_DECLINED : KIND_NUMPY_SCALAR + type_num;
}

static inline bool is_array_kind(int kind) {
    return kind >= 0 && kind < KIND_NUMPY_SCALAR;
}

// Store a Python int in `buffer` as the type `type_num`, if it is exactly
// representable there. Integers must be in range, where NumPy would raise
// OverflowError otherwise. A float type takes magnitudes up to 2^53 (float64)
// or 2^24 (float32), every one of which converts exactly, so no question of
// how NumPy rounds a larger one arises. Returns false, with no exception set,
// for anything else.
static bool python_int_as(PyObject *obj, int type_num, char *buffer) {
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0 || (v == -1 && PyErr_Occurred())) {
        PyErr_Clear();
        return false;
    }
    switch (type_num) {
    case NPY_DOUBLE:
        if (v > (1LL << DBL_MANT_DIG) || v < -(1LL << DBL_MANT_DIG)) return false;
        *(double *)buffer = (double)v;
        return true;
    case NPY_FLOAT:
        if (v > (1LL << FLT_MANT_DIG) || v < -(1LL << FLT_MANT_DIG)) return false;
        *(float *)buffer = (float)v;
        return true;
    case NPY_BYTE:
        if (v < INT8_MIN || v > INT8_MAX) return false;
        *(int8_t *)buffer = (int8_t)v;
        return true;
    case NPY_UBYTE:
        if (v < 0 || v > UINT8_MAX) return false;
        *(uint8_t *)buffer = (uint8_t)v;
        return true;
    case NPY_SHORT:
        if (v < INT16_MIN || v > INT16_MAX) return false;
        *(int16_t *)buffer = (int16_t)v;
        return true;
    case NPY_USHORT:
        if (v < 0 || v > UINT16_MAX) return false;
        *(uint16_t *)buffer = (uint16_t)v;
        return true;
    case NPY_INT:
        if (v < INT32_MIN || v > INT32_MAX) return false;
        *(int32_t *)buffer = (int32_t)v;
        return true;
    case NPY_UINT:
        if (v < 0 || v > (long long)UINT32_MAX) return false;
        *(uint32_t *)buffer = (uint32_t)v;
        return true;
    case NPY_LONG:
    case NPY_LONGLONG:
        // Both are 64 bits on every platform the package builds for.
        *(int64_t *)buffer = (int64_t)v;
        return true;
    case NPY_ULONG:
    case NPY_ULONGLONG:
        if (v < 0) return false;
        *(uint64_t *)buffer = (uint64_t)v;
        return true;
    default:
        return false;
    }
}

// Store a number operand in `buffer` as the loop's type `type_num`. A Python
// float goes to float64 as it is, and to float32 rounded as C rounds it,
// which is what NumPy's conversion does, unless a finite value would become
// an infinity, where NumPy warns. A NumPy number is taken only by a loop of
// its own type, so its bits are copied. Returns false for anything the lane
// leaves to NumPy.
static bool number_as(PyObject *obj, int kind, int type_num, char *buffer) {
    if (kind == KIND_PYTHON_FLOAT) {
        const double v = PyFloat_AS_DOUBLE(obj);
        if (type_num == NPY_DOUBLE) {
            *(double *)buffer = v;
            return true;
        }
        if (type_num == NPY_FLOAT) {
            const float f = (float)v;
            if (isinf(f) && !isinf(v)) return false;
            *(float *)buffer = f;
            return true;
        }
        return false;
    }
    if (kind == KIND_PYTHON_INT) return python_int_as(obj, type_num, buffer);
    // A NumPy number, whose loop is of its own type: its value is copied as
    // it is.
    PyArray_ScalarAsCtype(obj, buffer);
    return true;
}

// --------------------------------------------------------------- the table

// The first slot to look in for a combination.
static inline size_t lane_slot(PyObject *ufunc, int kind0, int kind1) {
    // Objects are at least 16-byte aligned, so the low bits carry nothing.
    uintptr_t h = ((uintptr_t)ufunc >> 4) * 0x9E3779B97F4A7C15ull;
    h ^= (uintptr_t)(kind0 * 131 + kind1) * 0xBF58476D1CE4E5B9ull;
    return (size_t)(h >> 20) & (LANE_TABLE_SIZE - 1);
}

// Release what one entry holds and mark it free.
static void lane_release(lane_entry *entry) {
    Py_CLEAR(entry->out_descr);
    Py_CLEAR(entry->capsule);
    memset(entry, 0, sizeof *entry);
}

// Find the entry of a combination, or a free entry for it. When its slots are
// all taken by other combinations, the entry in its first slot is released
// and given to it, unless NumPy is being asked about that one, when there is
// none (NULL). A call that has copied an entry's loop holds the capsule, so
// releasing the entry does not free a loop that is running.
static lane_entry *lane_find(PyObject *ufunc, int kind0, int kind1) {
    const size_t first = lane_slot(ufunc, kind0, kind1);
    size_t slot = first;
    for (int probe = 0; probe < LANE_PROBES; probe++, slot = (slot + 1) & (LANE_TABLE_SIZE - 1)) {
        lane_entry *entry = &lane_table[slot];
        if (entry->state == LANE_FREE) return entry;
        if (entry->ufunc == ufunc && entry->kind0 == kind0 && entry->kind1 == kind1) return entry;
    }
    lane_entry *evicted = &lane_table[first];
    if (evicted->state == LANE_RESOLVING) return NULL;
    lane_release(evicted);
    return evicted;
}

// Empty the table, releasing what it holds.
static void lane_clear(void) {
    for (size_t i = 0; i < LANE_TABLE_SIZE; i++) lane_release(&lane_table[i]);
}

// The first argument of `_resolve_dtypes_and_context` for one operand: the
// array's or the NumPy number's dtype, or the Python type of a Python number,
// which NumPy takes as a number of no fixed type (NEP 50). A new reference.
static PyObject *resolution_type(int kind) {
    if (kind == KIND_PYTHON_FLOAT) return Py_NewRef((PyObject *)&PyFloat_Type);
    if (kind == KIND_PYTHON_INT) return Py_NewRef((PyObject *)&PyLong_Type);
    const int type_num = is_array_kind(kind) ? kind : kind - KIND_NUMPY_SCALAR;
    return (PyObject *)PyArray_DescrFromType(type_num);
}

// The element size of one input as NumPy passes it to the loop: an array's
// element size, and zero for a number, which NumPy broadcasts.
static npy_intp resolution_stride(int kind) {
    if (!is_array_kind(kind)) return 0;
    PyArray_Descr *descr = PyArray_DescrFromType(kind);
    const npy_intp size = descr->elsize;
    Py_DECREF(descr);
    return size;
}

// Whether NumPy's resolved types let the lane serve the combination. An array
// must not need a cast, nor a NumPy number; a Python number must go to a type
// the lane converts to; the output must be of a type the lane takes. On
// success `entry->scalar_type` is the type a number is converted to.
static bool resolved_fits(PyObject *types, int nin, const int *kinds, lane_entry *entry) {
    entry->scalar_type = -1;
    for (int i = 0; i < nin; i++) {
        PyArray_Descr *descr = (PyArray_Descr *)PyTuple_GET_ITEM(types, i);
        if (!PyArray_DescrCheck((PyObject *)descr) || !lane_type(descr->type_num) ||
            !PyArray_ISNBO(descr->byteorder)) return false;
        if (kinds[i] == KIND_PYTHON_FLOAT || kinds[i] == KIND_PYTHON_INT) {
            if (descr->type_num == NPY_BOOL) return false;
            entry->scalar_type = descr->type_num;
        } else {
            const int own = is_array_kind(kinds[i]) ? kinds[i] : kinds[i] - KIND_NUMPY_SCALAR;
            if (descr->type_num != own) return false;
            if (!is_array_kind(kinds[i])) entry->scalar_type = own;
        }
    }
    PyArray_Descr *out = (PyArray_Descr *)PyTuple_GET_ITEM(types, nin);
    return PyArray_DescrCheck((PyObject *)out) && lane_type(out->type_num) && PyArray_ISNBO(out->byteorder);
}

// Find the legacy loop NumPy wraps for the resolved types: the first entry of
// the ufunc's type table whose types are those, which is the one NumPy's
// legacy wrapper and PyUFunc_ReplaceLoopBySignature both take. Sets
// `legacy_index` to -1 when there is none.
static void note_legacy_loop(PyUFuncObject *ufunc, PyObject *dtypes, lane_entry *entry) {
    entry->legacy_index = -1;
    entry->legacy_function = NULL;
    const int nargs = ufunc->nin + ufunc->nout;
    if (ufunc->functions == NULL || ufunc->types == NULL || PyTuple_GET_SIZE(dtypes) != nargs) return;
    for (int i = 0; i < ufunc->ntypes; i++) {
        bool same = true;
        for (int k = 0; k < nargs && same; k++) {
            same = ufunc->types[i * nargs + k] == ((PyArray_Descr *)PyTuple_GET_ITEM(dtypes, k))->type_num;
        }
        if (same) {
            entry->legacy_index = i;
            entry->legacy_function = ufunc->functions[i];
            return;
        }
    }
}

// Ask NumPy for its loop for one combination and fill the entry. Leaves the
// entry LANE_READY or LANE_DECLINED, never with an exception set.
static void lane_resolve(PyObject *ufunc, int nin, const int *kinds, lane_entry *entry) {
    entry->ufunc = ufunc;
    entry->kind0 = kinds[0];
    entry->kind1 = nin == 2 ? kinds[1] : KIND_NONE;
    entry->state = LANE_RESOLVING;
    PyObject *types = PyTuple_New(nin + 1);
    PyObject *strides = PyTuple_New(nin + 1);
    PyObject *resolved = NULL, *filled = NULL, *keywords = NULL;
    if (types == NULL || strides == NULL) goto declined;
    for (int i = 0; i < nin; i++) {
        PyObject *type = resolution_type(kinds[i]);
        PyObject *stride = PyLong_FromSsize_t(resolution_stride(kinds[i]));
        if (type == NULL || stride == NULL) {
            Py_XDECREF(type);
            Py_XDECREF(stride);
            goto declined;
        }
        PyTuple_SET_ITEM(types, i, type);
        PyTuple_SET_ITEM(strides, i, stride);
    }
    PyTuple_SET_ITEM(types, nin, Py_NewRef(Py_None));
    resolved = PyObject_CallMethod(ufunc, "_resolve_dtypes_and_context", "(O)", types);
    if (resolved == NULL || !PyTuple_Check(resolved) || PyTuple_GET_SIZE(resolved) != 2) goto declined;
    PyObject *dtypes = PyTuple_GET_ITEM(resolved, 0);
    PyObject *capsule = PyTuple_GET_ITEM(resolved, 1);
    if (!PyTuple_Check(dtypes) || PyTuple_GET_SIZE(dtypes) != nin + 1 || !PyCapsule_IsValid(capsule, LANE_CAPSULE) ||
        !resolved_fits(dtypes, nin, kinds, entry)) goto declined;
    PyArray_Descr *out = (PyArray_Descr *)PyTuple_GET_ITEM(dtypes, nin);
    // The output's stride is its element size, as for an array input.
    PyObject *out_stride = PyLong_FromSsize_t(out->elsize);
    if (out_stride == NULL) goto declined;
    PyTuple_SET_ITEM(strides, nin, out_stride);
    keywords = Py_BuildValue("{s:O}", "fixed_strides", strides);
    PyObject *method = keywords ? PyObject_GetAttrString(ufunc, "_get_strided_loop") : NULL;
    if (method != NULL) {
        PyObject *call_args = PyTuple_Pack(1, capsule);
        filled = call_args ? PyObject_Call(method, call_args, keywords) : NULL;
        Py_XDECREF(call_args);
        Py_DECREF(method);
    }
    if (filled == NULL) goto declined;
    lane_call_info *info = PyCapsule_GetPointer(capsule, LANE_CAPSULE);
    if (info == NULL || info->strided_loop == NULL || info->requires_pyapi) goto declined;
    note_legacy_loop((PyUFuncObject *)ufunc, dtypes, entry);
    entry->info = info;
    entry->capsule = Py_NewRef(capsule);
    entry->out_descr = (PyArray_Descr *)Py_NewRef((PyObject *)out);
    entry->state = LANE_READY;
    goto done;
declined:
    PyErr_Clear();
    entry->state = LANE_DECLINED;
done:
    Py_XDECREF(types);
    Py_XDECREF(strides);
    Py_XDECREF(resolved);
    Py_XDECREF(filled);
    Py_XDECREF(keywords);
}

// ---------------------------------------------------------------- the lane

// The floating-point exceptions NumPy reports that are raised now, as NumPy's
// NPY_FPE_* bits, read from the status register directly (see
// hk_clear_float_flags for why not the C library). The flags are left as
// they are.
static inline int lane_float_errors(void) {
#if defined(__aarch64__)
    uint64_t fpsr;
    __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr) : : "memory");
    // FPSR: IOC (invalid) bit 0, DZC (divide by zero) bit 1, OFC (overflow)
    // bit 2, UFC (underflow) bit 3.
    return ((fpsr & 0x1) ? NPY_FPE_INVALID : 0) | ((fpsr & 0x2) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((fpsr & 0x4) ? NPY_FPE_OVERFLOW : 0) | ((fpsr & 0x8) ? NPY_FPE_UNDERFLOW : 0);
#elif defined(__x86_64__)
    const unsigned int csr = _mm_getcsr();
    // MXCSR: IE (invalid) bit 0, ZE (divide by zero) bit 2, OE (overflow)
    // bit 3, UE (underflow) bit 4.
    return ((csr & 0x01) ? NPY_FPE_INVALID : 0) | ((csr & 0x04) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((csr & 0x08) ? NPY_FPE_OVERFLOW : 0) | ((csr & 0x10) ? NPY_FPE_UNDERFLOW : 0);
#else
    const int raised = fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
    return ((raised & FE_INVALID) ? NPY_FPE_INVALID : 0) | ((raised & FE_DIVBYZERO) ? NPY_FPE_DIVIDEBYZERO : 0) |
           ((raised & FE_OVERFLOW) ? NPY_FPE_OVERFLOW : 0) | ((raised & FE_UNDERFLOW) ? NPY_FPE_UNDERFLOW : 0);
#endif
}

// The output a call names, positionally or as `out=`: NULL when it names none
// (or None), the output when it is one the lane can write, and `declined`
// otherwise. `nargs` and `kwnames` are the call's.
static PyObject *given_output(PyObject *callable, PyObject *const *args, Py_ssize_t nargs, int nin,
                              PyObject *kwnames, bool *declined) {
    *declined = false;
    PyObject *out = NULL;
    if (nargs == nin + 1) {
        for (int i = 0; i < LANE_POSITIONAL_DEPRECATED; i++) {
            if (callable == positional_deprecated[i]) {
                *declined = true;
                return NULL;
            }
        }
        out = args[nin];
    }
    if (kwnames != NULL) {
        // Only `out=` is taken, and not together with a positional output.
        if (PyTuple_GET_SIZE(kwnames) != 1 || nargs != nin ||
            !hk_text_is(PyTuple_GET_ITEM(kwnames, 0), "out")) {
            *declined = true;
            return NULL;
        }
        out = args[nargs];
        if (PyTuple_CheckExact(out)) {
            if (PyTuple_GET_SIZE(out) != 1) {
                *declined = true;
                return NULL;
            }
            out = PyTuple_GET_ITEM(out, 0);
        }
    }
    if (out == NULL || out == Py_None) return NULL;
    if (!lane_array(out) || !PyArray_ISWRITEABLE((PyArrayObject *)out)) *declined = true;
    return out;
}

// True when an input array may be read while the output is written: the two
// are the same memory, element for element, or they do not overlap.
static inline bool may_share(PyArrayObject *input, PyArrayObject *out) {
    const char *in_start = PyArray_BYTES(input), *out_start = PyArray_BYTES(out);
    const npy_intp in_bytes = PyArray_NBYTES(input), out_bytes = PyArray_NBYTES(out);
    if (in_start == out_start && in_bytes == out_bytes) return true;
    return in_start + in_bytes <= out_start || out_start + out_bytes <= in_start;
}

// The call function of every hooked ufunc.
static PyObject *call_lane(HK_VECTORCALL_PARAMETERS) {
    // Every ufunc call in the process comes through here, most of them to
    // be declined: see operand_kind.
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    const int nin = ufunc->nin;
    if (nargs != nin && nargs != nin + 1) goto pass;
    PyObject *given = NULL;
    if (nargs != nin || kwnames != NULL) {
        bool declined;
        given = given_output(callable, args, nargs, nin, kwnames, &declined);
        if (declined) goto pass;
    }

    // The inputs, and the array that gives the result its shape.
    int kinds[2] = {KIND_NONE, KIND_NONE};
    PyArrayObject *shape = NULL;
    for (int i = 0; i < nin; i++) {
        kinds[i] = operand_kind(args[i]);
        if (kinds[i] == KIND_DECLINED) goto pass;
        if (is_array_kind(kinds[i]) &&
            (shape == NULL || PyArray_NDIM((PyArrayObject *)args[i]) > PyArray_NDIM(shape))) {
            shape = (PyArrayObject *)args[i];
        }
    }
    if (shape == NULL) goto pass;
    const int ndim = PyArray_NDIM(shape);
    const npy_intp *dims = PyArray_DIMS(shape);
    const npy_intp total = PyArray_SIZE(shape);

    // Each array is of the result's shape, or a row: the trailing part of it.
    // NumPy copies a row, repeated, into a buffer of whole rows: one loop call
    // when the result fits in it, which is the only case the lane takes. Its
    // iterator drops dimensions of one, so a row of one element is a stride
    // of zero over the column instead, and is left to NumPy too.
    int row_input = -1;
    for (int i = 0; i < nin; i++) {
        if (!is_array_kind(kinds[i])) continue;
        PyArrayObject *array = (PyArrayObject *)args[i];
        const int own = PyArray_NDIM(array);
        if (memcmp(PyArray_DIMS(array), dims + (ndim - own), own * sizeof(npy_intp)) != 0) goto pass;
        if (own == ndim) continue;
        const npy_intp length = PyArray_SIZE(array);
        if (length < 2 || total > (LANE_BUFFER_ELEMENTS / length) * length) goto pass;
        row_input = i;
    }

    lane_entry *entry = lane_find(callable, kinds[0], kinds[1]);
    if (entry == NULL || entry->state == LANE_RESOLVING) goto pass;
    // A legacy loop replaced since NumPy was asked: ask again.
    if (entry->state == LANE_READY && entry->legacy_index >= 0 &&
        ufunc->functions[entry->legacy_index] != entry->legacy_function) {
        lane_release(entry);
    }
    if (entry->state == LANE_FREE) lane_resolve(callable, nin, kinds, entry);
    if (entry->state != LANE_READY) goto pass;

    PyArrayObject *out = NULL;
    if (given != NULL) {
        out = (PyArrayObject *)given;
        if (PyArray_NDIM(out) != ndim || memcmp(PyArray_DIMS(out), dims, ndim * sizeof(npy_intp)) != 0 ||
            PyArray_TYPE(out) != entry->out_descr->type_num) goto pass;
        for (int i = 0; i < nin; i++) {
            if (is_array_kind(kinds[i]) && !may_share((PyArrayObject *)args[i], out)) goto pass;
        }
    }

    // The loop's arguments: each array's data and element size, a number
    // converted into a buffer with a stride of zero, as NumPy passes them.
    char scalar[2][LANE_SCALAR_BYTES];
    char *data[3];
    npy_intp strides[3];
    for (int i = 0; i < nin; i++) {
        if (is_array_kind(kinds[i])) {
            data[i] = PyArray_BYTES((PyArrayObject *)args[i]);
            strides[i] = PyArray_ITEMSIZE((PyArrayObject *)args[i]);
        } else {
            if (!number_as(args[i], kinds[i], entry->scalar_type, scalar[i])) goto pass;
            data[i] = scalar[i];
            strides[i] = 0;
        }
    }

    // A row, repeated to the result's length, as NumPy's buffer holds it:
    // on the stack when it fits there.
    char stack_rows[HK_STACK_SCRATCH_BYTES];
    char *rows = NULL;
    if (row_input >= 0) {
        PyArrayObject *row = (PyArrayObject *)args[row_input];
        const npy_intp row_bytes = PyArray_NBYTES(row);
        const npy_intp bytes = total * PyArray_ITEMSIZE(row);
        rows = bytes <= (npy_intp)sizeof stack_rows ? stack_rows : PyMem_Malloc(bytes);
        if (rows == NULL) return PyErr_NoMemory();
        for (npy_intp offset = 0; offset < bytes; offset += row_bytes) memcpy(rows + offset, data[row_input], row_bytes);
        data[row_input] = rows;
    }

    if (out == NULL) {
        Py_INCREF(entry->out_descr);
        out = (PyArrayObject *)PyArray_NewFromDescr(hk_ndarray_type, entry->out_descr, ndim, (npy_intp *)dims, NULL,
                                                    NULL, 0, NULL);
        if (out == NULL) {
            if (rows != stack_rows) PyMem_Free(rows);
            return NULL;
        }
    } else {
        Py_INCREF(out);
    }
    data[nin] = PyArray_BYTES(out);
    strides[nin] = entry->out_descr->elsize;

    // Copied out of the entry, which another thread may clear while the
    // interpreter lock is released; the capsule keeps the loop alive.
    lane_call_info *info = entry->info;
    PyObject *capsule = Py_NewRef(entry->capsule);
    const bool test_flags = !info->no_floatingpoint_errors;
    if (test_flags) hk_clear_reported_float_flags();
    npy_intp count = total;
    PyThreadState *released = total > LANE_THREADS_ELEMENTS ? PyEval_SaveThread() : NULL;
    const int status = info->strided_loop(info->context, data, &count, strides, info->auxdata);
    if (released != NULL) PyEval_RestoreThread(released);
    Py_DECREF(capsule);
    if (rows != stack_rows) PyMem_Free(rows);

    if (status < 0 || PyErr_Occurred()) {
        // NumPy raises the loop's error and returns nothing; the output
        // keeps what the loop wrote, as NumPy's would.
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "a ufunc loop failed without an error");
        Py_DECREF(out);
        return NULL;
    }
    if (test_flags) {
        const int raised = lane_float_errors();
        if (raised != 0) {
            hk_clear_reported_float_flags();
            // NumPy's own handler: warns, raises, calls or ignores as
            // np.errstate says, with the ufunc's name in the message.
            if (PyUFunc_GiveFloatingpointErrors(ufunc->name ? ufunc->name : "<unnamed>", raised) < 0) {
                Py_DECREF(out);
                return NULL;
            }
        }
    }
    hk_take(HK_UFUNC_LANE);
    return (PyObject *)out;
pass:
    hk_pass(HK_OP_UFUNC);
    return lane_original(callable, args, nargsf, kwnames);
}

// ---------------------------------------------------------------- the hook

// Whether a ufunc is one the lane serves: one or two inputs, one output, no
// core dimensions, and NumPy's own call function. Another call function
// means someone else has changed the ufunc, and it is left alone.
static bool lane_ufunc(PyObject *obj, vectorcallfunc original) {
    if (Py_TYPE(obj) != &PyUFunc_Type) return false;
    PyUFuncObject *ufunc = (PyUFuncObject *)obj;
    return (ufunc->nin == 1 || ufunc->nin == 2) && ufunc->nout == 1 && !ufunc->core_enabled &&
           ufunc->vectorcall == original;
}

// Whether this NumPy's `_resolve_dtypes_and_context` answers with a capsule
// of the name the lane reads. Returns 0, or -1 with an exception set.
static int check_interface(PyObject *add) {
    PyObject *f8 = (PyObject *)PyArray_DescrFromType(NPY_DOUBLE);
    PyObject *types = f8 ? PyTuple_Pack(3, f8, f8, Py_None) : NULL;
    PyObject *answer = types ? PyObject_CallMethod(add, "_resolve_dtypes_and_context", "(O)", types) : NULL;
    Py_XDECREF(f8);
    Py_XDECREF(types);
    if (answer == NULL) return -1;
    const bool named = PyTuple_Check(answer) && PyTuple_GET_SIZE(answer) == 2 &&
                       PyCapsule_IsValid(PyTuple_GET_ITEM(answer, 1), LANE_CAPSULE);
    Py_DECREF(answer);
    if (!named) {
        PyErr_SetString(PyExc_RuntimeError, "this NumPy's ufunc call information is not a " LANE_CAPSULE " capsule");
        return -1;
    }
    return 0;
}

// Point every ufunc of the `numpy` module at the lane. See hk_hook.apply.
static int ufunc_apply(int index) {
    (void)index;
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return -1;
    // The module's own dictionary, not dir() and getattr(): NumPy imports a
    // submodule on first access to its name, and nothing here should.
    PyObject *namespace = PyModule_GetDict(numpy);
    PyObject *add = namespace ? PyDict_GetItemString(namespace, "add") : NULL;
    if (add == NULL || Py_TYPE(add) != &PyUFunc_Type || ((PyUFuncObject *)add)->vectorcall == NULL) {
        PyErr_SetString(PyExc_TypeError, "numpy.add is not a ufunc with a call function");
        Py_DECREF(numpy);
        return -1;
    }
    // `np.add`'s call function is the one every ufunc NumPy makes starts with.
    const vectorcallfunc original = ((PyUFuncObject *)add)->vectorcall;
    PyObject *list = NULL;
    if (check_interface(add) != 0) goto failed;
    list = PyList_New(0);
    if (list == NULL) goto failed;
    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (PyDict_Next(namespace, &position, &key, &value)) {
        if (!lane_ufunc(value, original)) continue;
        const int present = PySequence_Contains(list, value);
        if (present < 0 || (present == 0 && PyList_Append(list, value) != 0)) goto failed;
    }
    lane_clear();
    lane_original = original;
    for (int i = 0; i < LANE_POSITIONAL_DEPRECATED; i++) {
        positional_deprecated[i] = PyDict_GetItemString(namespace, POSITIONAL_DEPRECATED_NAMES[i]);
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); i++) {
        ((PyUFuncObject *)PyList_GET_ITEM(list, i))->vectorcall = call_lane;
    }
    hooked = list;
    Py_DECREF(numpy);
    return 0;
failed:
    Py_XDECREF(list);
    Py_DECREF(numpy);
    return -1;
}

// Put NumPy's call function back in every hooked ufunc. See hk_hook.remove.
// Returns 0 if any of them no longer held the lane, which is then left as it is.
static int ufunc_remove(int index) {
    (void)index;
    if (hooked == NULL) return 0;
    int restored = 1;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(hooked); i++) {
        PyUFuncObject *ufunc = (PyUFuncObject *)PyList_GET_ITEM(hooked, i);
        if (ufunc->vectorcall == call_lane) {
            ufunc->vectorcall = lane_original;
        } else {
            restored = 0;
        }
    }
    Py_CLEAR(hooked);
    memset(positional_deprecated, 0, sizeof positional_deprecated);
    lane_clear();
    return restored;
}

hk_hook hk_ufunc_hooks[] = {
    {"ufunc.call", ufunc_apply, ufunc_remove, 0, false},
    {NULL, NULL, NULL, 0, false},
};
