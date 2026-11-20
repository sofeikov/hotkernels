// Method hook: the C function pointer in a type's method table.
//
// Each method of a built-in type is described by a PyMethodDef in a table
// inside NumPy, and every way of calling the method, including a bound method
// saved earlier, reads the function pointer from that entry at the time of the
// call. Changing the pointer hooks the method and leaves `np.ndarray.astype`
// the same object. NumPy's own functions of the same name call the method, so
// `np.nonzero(x)` and `np.cumsum(x)` arrive here too. `np.sum(x)` does not:
// it has its own function hook. `np.argmax(x)` and `np.clip(x, lo, hi)` have
// one as well, and arrive here only when it passed them on.
//
// `np.any(m, axis=1)` does not arrive here either: for a plain ndarray it calls
// `np.logical_or.reduce` itself, and it has a function hook of its own.
//
// Most of the hooked methods are ndarray's. Two are the ufunc type's:
// `accumulate` and `reduceat`, which every ufunc object shares, so
// `np.maximum.accumulate(x)` and `np.add.accumulate(x)` reach the same entry
// and the wrapper tells them apart by `self`.
//
// A method is called in one of several conventions, recorded in the entry's
// flags. A wrapper must have the shape the flags say, so a hook is applied
// only when the flags are the ones its wrapper was written for.
#include "binding.h"

// The methods that are hooked, as indices into `methods`.
enum { M_ASTYPE, M_SUM, M_NONZERO, M_CUMSUM, M_SEARCHSORTED, M_ARGMAX, M_ARGMIN, M_CLIP, M_ACCUMULATE, M_REDUCEAT,
       M_ANY, M_ROUND, M_ARGSORT, M_COUNT };

// The type whose method table holds a hooked method. The ufunc type is reached
// through NumPy's table of pointers, which is filled only when the module is
// imported, so a method names its type by this and not by a pointer.
typedef enum { OWNER_NDARRAY, OWNER_UFUNC } hk_owner;

// A method called with an argument array and keyword names
// (METH_FASTCALL | METH_KEYWORDS). CPython names this type differently
// across versions.
typedef PyObject *(*hk_fast_keywords)(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                      PyObject *kwnames);

// One hooked method.
typedef struct {
    // Method name on its type.
    const char *name;
    // The type whose method it is.
    hk_owner owner;
    // Fetches what the wrapper compares its arguments with, before the hook is
    // first applied. Returns 0, or -1 with an exception set. NULL when the
    // wrapper needs nothing.
    int (*prepare)(void);
    // Calling convention the wrapper implements, as PyMethodDef.ml_flags.
    int flags;
    // The binding's function, stored as the generic pointer type.
    PyCFunction wrapper;
    // NumPy's table entry, set while the hook is applied.
    PyMethodDef *definition;
    // NumPy's own function, kept while the hook is applied.
    PyCFunction original;
} hk_method;

static hk_method methods[M_COUNT];

// The names of cumsum's parameters, in order.
static const char *const CUMSUM_PARAMETERS[] = {"axis", "dtype", "out"};
#define CUMSUM_PARAMETER_COUNT 3

// Floating-point exceptions NumPy can turn into a warning or an error.
#define HK_REPORTED_EXCEPTIONS (FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO | FE_UNDERFLOW)

// ------------------------------------------------------------------ astype

// The type `np.float32` and its dtype object, which NumPy keeps one of.
// Fetched when the hook is applied.
static PyObject *float32_type;
static PyObject *float32_descr;

// Fetch the two float32 objects. See hk_method.prepare.
static int astype_prepare(void) {
    if (float32_descr != NULL) return 0;
    float32_type = (PyObject *)&PyFloatArrType_Type;
    float32_descr = (PyObject *)PyArray_DescrFromType(NPY_FLOAT);
    return float32_descr == NULL ? -1 : 0;
}

// True when `dtype`, the argument of astype, names native float32 in one of
// the usual spellings: the type `np.float32`, its dtype object, or the
// strings "float32" and "f4". The two objects are compared by identity,
// which is the cheapest test and comes first: astype is called with a type
// object far more often than with anything else.
static bool names_float32(PyObject *dtype) {
    if (dtype == float32_type || dtype == float32_descr) return true;
    if (PyUnicode_CheckExact(dtype)) {
        return PyUnicode_CompareWithASCIIString(dtype, "float32") == 0 ||
               PyUnicode_CompareWithASCIIString(dtype, "f4") == 0;
    }
    return false;
}

// Convert a contiguous uint8 array to float32. The result has the shape of
// the input.
static HK_NOINLINE PyObject *try_astype(PyObject *self) {
    PyArrayObject *x = (PyArrayObject *)self;
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_FLOAT);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(x);
    HK_BEGIN_NOGIL(n)
    cast_u8_f32(PyArray_DATA(x), PyArray_DATA((PyArrayObject *)result), (size_t)n);
    HK_END_NOGIL
    hk_take(HK_CAST_U8_F32);
    return result;
}

// x.astype(np.float32) for a contiguous uint8 array, with every other
// argument at its default.
static PyObject *method_astype(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (hk_is_flat(self, NPY_UINT8) && nargs == 1 && kwnames == NULL && names_float32(args[0])) {
        PyObject *result = try_astype(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_ASTYPE);
    return ((hk_fast_keywords)(void (*)(void))methods[M_ASTYPE].original)(self, args, nargs, kwnames);
}

// --------------------------------------------------------------------- sum

// x.sum() with no arguments.
static PyObject *method_sum(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (nargs == 0 && kwnames == NULL) {
        PyObject *result = hk_try_sum(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_SUM);
    return ((hk_fast_keywords)(void (*)(void))methods[M_SUM].original)(self, args, nargs, kwnames);
}

// ----------------------------------------------------------------- nonzero

// mask.nonzero() for a contiguous one-dimensional boolean array: a tuple
// holding one int64 array of the positions that are True. Shared with the
// function hook for np.where(mask).
//
// The kernel's output is scratch: it needs room for n indices and may write
// anywhere in it, whatever it reports. So it runs on a buffer of the
// binding's own, and the reported indices are copied into an array of
// exactly that size.
HK_NOINLINE PyObject *hk_try_nonzero(PyObject *self) {
    if (!hk_is_flat_1d(self, NPY_BOOL)) return NULL;
    PyArrayObject *mask = (PyArrayObject *)self;
    const npy_intp n = PyArray_SIZE(mask);
    const size_t bytes = (size_t)n * sizeof(int64_t);
    int64_t on_stack[HK_STACK_SCRATCH_BYTES / sizeof(int64_t)];
    int64_t *scratch = bytes <= sizeof on_stack ? on_stack : malloc(bytes);
    if (scratch == NULL) return NULL;

    npy_intp count = -1;
    {
        HK_BEGIN_NOGIL(n)
        if (hk_mask_is_clean(PyArray_DATA(mask), (size_t)n)) {
            count = (npy_intp)nonzero_bool(PyArray_DATA(mask), scratch, (size_t)n);
        }
        HK_END_NOGIL
    }
    PyObject *indices = count >= 0 ? PyArray_SimpleNew(1, &count, NPY_INTP) : NULL;
    PyObject *result = NULL;
    if (indices != NULL) {
        memcpy(PyArray_DATA((PyArrayObject *)indices), scratch, (size_t)count * sizeof(int64_t));
        result = PyTuple_Pack(1, indices);
        Py_DECREF(indices);
    }
    if (scratch != on_stack) free(scratch);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    hk_take(HK_NONZERO_BOOL);
    return result;
}

// Called with a tuple of positional arguments (METH_VARARGS).
static PyObject *method_nonzero(PyObject *self, PyObject *args) {
    if (PyTuple_GET_SIZE(args) == 0) {
        PyObject *result = hk_try_nonzero(self);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_NONZERO);
    return methods[M_NONZERO].original(self, args);
}

// ------------------------------------------------------------------ cumsum

// True when a cumsum call leaves `axis`, `dtype` and `out` at None, however
// it spells that: `x.cumsum()`, `x.cumsum(None)` or, as np.cumsum(x) does,
// `x.cumsum(axis=None, dtype=None, out=None)`.
static HK_NOINLINE bool cumsum_arguments_are_default(PyObject *args, PyObject *kwargs) {
    const Py_ssize_t positional = PyTuple_GET_SIZE(args);
    if (positional > CUMSUM_PARAMETER_COUNT) return false;
    for (Py_ssize_t i = 0; i < positional; i++) {
        if (PyTuple_GET_ITEM(args, i) != Py_None) return false;
    }
    if (kwargs == NULL) return true;
    PyObject *key, *value;
    Py_ssize_t position = 0;
    while (PyDict_Next(kwargs, &position, &key, &value)) {
        if (value != Py_None) return false;
        // Only a parameter that was not also given by position.
        Py_ssize_t i = positional;
        while (i < CUMSUM_PARAMETER_COUNT && PyUnicode_CompareWithASCIIString(key, CUMSUM_PARAMETERS[i]) != 0) i++;
        if (i == CUMSUM_PARAMETER_COUNT) return false;
    }
    return true;
}

// The kernel for `x.cumsum()` on this array: int64 and bool always, float32
// and float64 when install(inexact=True) was given. Returns HK_KERNEL_COUNT when
// there is none. This is tested before the arguments are looked at: most
// calls are for another element type, and rejecting those must cost almost
// nothing.
static hk_kernel cumsum_kernel(PyObject *self) {
    if (hk_is_flat(self, NPY_INT64)) return HK_CUMSUM_I64;
    if (hk_is_flat(self, NPY_BOOL)) return HK_CUMSUM_BOOL;
    if (hk_inexact && hk_is_flat(self, NPY_FLOAT)) return HK_CUMSUM_F32;
    if (hk_inexact && hk_is_flat(self, NPY_DOUBLE)) return HK_CUMSUM_F64;
    return HK_KERNEL_COUNT;
}

// Run a cumsum kernel. The result is one-dimensional, as NumPy's is when no
// axis is given.
//
// The float kernels regroup the additions and can raise floating-point
// exceptions NumPy's sequential sum does not, or the other way round (issue
// 9). NumPy turns those into warnings. So the flags are cleared before the
// kernel, and if any that NumPy reports is set afterwards the kernel's output
// is dropped and NumPy computes the result and issues its own warnings.
//
// A boolean array gives a running count of its True elements, as np.intp:
// NumPy's result type for it. The kernel adds the mask's bytes, and NumPy
// counts a byte other than 0 or 1 once, so the bytes are checked first and
// such a mask is NumPy's.
static HK_NOINLINE PyObject *try_cumsum(PyObject *self, hk_kernel kernel) {
    PyArrayObject *x = (PyArrayObject *)self;
    npy_intp n = PyArray_SIZE(x);
    if (n < 1) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &n, kernel == HK_CUMSUM_BOOL ? NPY_INTP : PyArray_TYPE(x));
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    void *out = PyArray_DATA((PyArrayObject *)result);
    // True when the kernel's output is not NumPy's and the call is passed on.
    bool raised = false;
    {
        HK_BEGIN_NOGIL(n)
        if (kernel == HK_CUMSUM_I64) {
            cumsum_i64(PyArray_DATA(x), out, (size_t)n);
        } else if (kernel == HK_CUMSUM_BOOL) {
            raised = !hk_mask_is_clean(PyArray_DATA(x), (size_t)n);
            if (!raised) cumsum_bool(PyArray_DATA(x), out, (size_t)n);
        } else {
            feclearexcept(FE_ALL_EXCEPT);
            if (kernel == HK_CUMSUM_F32) cumsum_f32(PyArray_DATA(x), out, (size_t)n);
            else cumsum_f64(PyArray_DATA(x), out, (size_t)n);
            raised = fetestexcept(HK_REPORTED_EXCEPTIONS) != 0;
            feclearexcept(FE_ALL_EXCEPT);
        }
        HK_END_NOGIL
    }
    if (raised) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(kernel);
    return result;
}

// x.cumsum(axis=0) on a matrix, with np.add's row loop. Below.
static PyObject *try_cumsum_rows(PyObject *self, PyObject *args, PyObject *kwargs, bool *answered);

// Called with a tuple and a dictionary (METH_VARARGS | METH_KEYWORDS). A
// float64 array of two or more dimensions called with arguments may be a
// cumsum along the first axis, which the row loop takes here rather than at
// ufunc.accumulate, so that the call is not counted as passed on here first.
static PyObject *method_cumsum(PyObject *self, PyObject *args, PyObject *kwargs) {
    if ((PyTuple_GET_SIZE(args) != 0 || kwargs != NULL) && hk_is_flat(self, NPY_DOUBLE) &&
        PyArray_NDIM((PyArrayObject *)self) >= 2) {
        bool answered;
        PyObject *result = try_cumsum_rows(self, args, kwargs, &answered);
        if (answered) return result;
    }
    const hk_kernel kernel = cumsum_kernel(self);
    if (kernel != HK_KERNEL_COUNT && cumsum_arguments_are_default(args, kwargs)) {
        PyObject *result = try_cumsum(self, kernel);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_CUMSUM);
    return ((PyCFunctionWithKeywords)(void (*)(void))methods[M_CUMSUM].original)(self, args, kwargs);
}

// ------------------------------------------------------------ searchsorted

// Choose the kernel for a search of `a`, which is float64 or int64, from the
// `side` argument: absent or 'left' gives the left kernel of that element
// type, 'right' the right one. Returns HK_KERNEL_COUNT for anything else,
// including a subclass of str, which is left for NumPy to judge.
static hk_kernel search_kernel(PyArrayObject *a, PyObject *side) {
    const bool integers = PyArray_TYPE(a) == NPY_INT64;
    if (side == NULL || (PyUnicode_CheckExact(side) && hk_text_is(side, "left"))) {
        return integers ? HK_SEARCHSORTED_I64 : HK_SEARCHSORTED_F64;
    }
    if (PyUnicode_CheckExact(side) && hk_text_is(side, "right")) {
        return integers ? HK_SEARCHSORTED_RIGHT_I64 : HK_SEARCHSORTED_RIGHT_F64;
    }
    return HK_KERNEL_COUNT;
}

// Run one of the four search kernels, which have one signature but for the
// element type: `a` holds n sorted elements and `v` the needles, both float64
// for the float64 kernels and both int64 for the int64 ones; `out` gets one
// position per needle.
static void run_search(hk_kernel kernel, const void *a, const void *v, int64_t *out, size_t n, size_t needles) {
    switch (kernel) {
    case HK_SEARCHSORTED_F64: searchsorted_f64(a, v, out, n, needles); break;
    case HK_SEARCHSORTED_RIGHT_F64: searchsorted_right_f64(a, v, out, n, needles); break;
    case HK_SEARCHSORTED_I64: searchsorted_i64(a, v, out, n, needles); break;
    default: searchsorted_right_i64(a, v, out, n, needles); break;
    }
}

// Run the search for one number, as `a.searchsorted(2.5)` or, on an int64
// array, `a.searchsorted(np.int64(7))`: NumPy returns the position as an
// np.intp scalar. Returns NULL with no exception set if `needle` is not a
// number the kernel's element type holds as NumPy would convert it: for
// float64 a Python float, or a Python int a double holds exactly; for int64
// an np.int64, or a Python int that fits int64.
static PyObject *searchsorted_scalar(PyArrayObject *a, PyObject *needle, hk_kernel kernel) {
    // The needle's eight bytes, as the element type of `a`.
    union {
        double f64;
        int64_t i64;
    } value;
    if (PyArray_TYPE(a) == NPY_INT64) {
        if (!hk_as_int64(needle, &value.i64)) return NULL;
    } else if (!hk_as_exact_double(needle, &value.f64)) {
        return NULL;
    }
    npy_intp position;
    run_search(kernel, PyArray_DATA(a), &value, (int64_t *)&position, (size_t)PyArray_SIZE(a), 1);
    return hk_scalar(NPY_INTP, &position);
}

// a.searchsorted(v) for a one-dimensional contiguous float64 or int64 array
// `a` and needles of the same element type: a contiguous array of any shape,
// whose shape the result takes, or one number (see searchsorted_scalar).
// `side` must be absent, 'left' or 'right'; each side has its own kernel.
// There must be no `sorter`. Needles of another type are NumPy's: it converts
// them and `a` to a common type by its own rules.
// np.searchsorted(a, v) calls the method as
// `a.searchsorted(v, side='left', sorter=None)`, and np.digitize calls
// np.searchsorted: with side='right' by default, with side='left' for
// right=True, and on a reversed view, which is not contiguous and is passed
// on, when its bins decrease.
//
// An array that is not sorted is not detected. NumPy does not detect it
// either; both give a meaningless result, not the same one, and the kernel
// stays inside its buffers whatever the data.
static HK_NOINLINE PyObject *try_searchsorted(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                              PyObject *kwnames) {
    enum { V, SIDE, SORTER, PARAMETERS };
    static const char *const names[PARAMETERS] = {"v", "side", "sorter"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[V] == NULL || !hk_is_absent(bound[SORTER])) return NULL;
    PyArrayObject *a = (PyArrayObject *)self;
    const hk_kernel kernel = search_kernel(a, bound[SIDE]);
    if (kernel == HK_KERNEL_COUNT) return NULL;

    PyObject *result;
    if (hk_is_flat(bound[V], PyArray_TYPE(a))) {
        PyArrayObject *v = (PyArrayObject *)bound[V];
        result = PyArray_SimpleNew(PyArray_NDIM(v), PyArray_DIMS(v), NPY_INTP);
        if (result == NULL) {
            PyErr_Clear();
            return NULL;
        }
        const npy_intp needles = PyArray_SIZE(v);
        HK_BEGIN_NOGIL(needles)
        run_search(kernel, PyArray_DATA(a), PyArray_DATA(v), PyArray_DATA((PyArrayObject *)result),
                   (size_t)PyArray_SIZE(a), (size_t)needles);
        HK_END_NOGIL
    } else {
        result = searchsorted_scalar(a, bound[V], kernel);
        if (result == NULL) return NULL;
    }
    hk_take(kernel);
    return result;
}

// The array's type is tested before the arguments are looked at: most calls
// are for another element type, and rejecting those must cost almost nothing.
static PyObject *method_searchsorted(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (hk_is_flat_1d(self, NPY_DOUBLE) || hk_is_flat_1d(self, NPY_INT64)) {
        PyObject *result = try_searchsorted(self, args, nargs, kwnames);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_SEARCHSORTED);
    return ((hk_fast_keywords)(void (*)(void))methods[M_SEARCHSORTED].original)(self, args, nargs, kwnames);
}

// ------------------------------------------------------ argmax and argmin

// True when every argument of an argmax or argmin call is None: `x.argmax()`,
// `x.argmax(None)`, or, as np.argmax(x, axis=None) calls the method,
// `x.argmax(axis=None, out=None)`. `keepdims` is left to NumPy.
static HK_NOINLINE bool argext_arguments_are_default(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { AXIS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"axis", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return false;
    return hk_is_absent(bound[AXIS]) && hk_is_absent(bound[OUT]);
}

// True when the arguments of an argmax or argmin call ask for the last axis
// of a matrix: `axis` the Python int 1 or -1, by position or by keyword, and
// `out` absent or None, as np.argmax(m, axis=1) passes it to the method.
static HK_NOINLINE bool argext_arguments_are_rows(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { AXIS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"axis", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return false;
    return bound[AXIS] != NULL && hk_is_last_axis(bound[AXIS]) && hk_is_absent(bound[OUT]);
}

// Hand an argmax or argmin call of a float32 or float64 array on to NumPy and
// count it once, whatever NumPy does with it. NumPy may call the type slot
// once per row (`axis=`); the slot hook records in hk_argext_mark what it did
// instead of counting each row. The call is then counted under the kernel the
// slot took, or as left to NumPy when it took none, which includes a call
// NumPy raises on. The mark in place before is put back, so that a call made
// while another is waiting, which NumPy does not do, is not lost.
static HK_NOINLINE PyObject *argext_counted(PyObject *self, PyObject *const *args, Py_ssize_t nargs,
                                            PyObject *kwnames, int method, hk_operation operation) {
    const hk_argext_call outer = hk_argext_mark;
    hk_argext_mark = (hk_argext_call){.active = true, .taken = false};
    PyObject *result = ((hk_fast_keywords)(void (*)(void))methods[method].original)(self, args, nargs, kwnames);
    const hk_argext_call call = hk_argext_mark;
    hk_argext_mark = outer;
    if (call.taken) {
        hk_take(call.slot_kernel);
    } else {
        hk_pass(operation);
    }
    return result;
}

// x.argmax() and x.argmin() on a contiguous float32 or float64 array: the
// kernel on the whole array, as the function hook calls it for np.argmax(x).
// The type slot behind the method leaves runs under 128 elements to NumPy,
// because it has to take the interpreter lock; a method already holds it, so
// the kernel is called here at every size. m.argmax(axis=1) on a C-ordered
// float32 or float64 matrix is taken too (hk_try_argext_rows). Every other
// call of a float32 or float64 array, of any layout or subclass, is passed on
// and counted once (argext_counted); a call of any other type has no kernel
// behind it and goes on uncounted, as before. The array's type is tested
// before the arguments, as for clip below.
static PyObject *method_argext(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                               int method, bool want_max) {
    const int type_num = PyArray_TYPE((PyArrayObject *)self);
    if (type_num != NPY_FLOAT && type_num != NPY_DOUBLE) {
        return ((hk_fast_keywords)(void (*)(void))methods[method].original)(self, args, nargs, kwnames);
    }
    const bool float_array = hk_is_flat(self, type_num);
    if (float_array && ((nargs == 0 && kwnames == NULL) || argext_arguments_are_default(args, nargs, kwnames))) {
        PyObject *result = hk_try_argext(self, want_max);
        if (result != NULL) return result;
    } else if (float_array && PyArray_NDIM((PyArrayObject *)self) == 2 &&
               argext_arguments_are_rows(args, nargs, kwnames)) {
        PyObject *result = hk_try_argext_rows(self, want_max);
        if (result != NULL) return result;
    }
    return argext_counted(self, args, nargs, kwnames, method, want_max ? HK_OP_ARGMAX : HK_OP_ARGMIN);
}

static PyObject *method_argmax(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    return method_argext(self, args, nargs, kwnames, M_ARGMAX, true);
}

static PyObject *method_argmin(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    return method_argext(self, args, nargs, kwnames, M_ARGMIN, false);
}

// -------------------------------------------------------------------- clip

// x.clip(min, max) for a contiguous float32 or float64 array, or a short
// int64 one, the bounds by
// position or as `min=` and `max=`, and `out` absent or None, or for float32
// and float64 the array itself: the same direct call as np.clip's function
// hook makes (see hk_try_clip). The method passes its arguments to NumPy's
// `_methods._clip`, whose parameters these are.
static HK_NOINLINE PyObject *try_clip(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { MIN, MAX, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"min", "max", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    // A bound left out or None is the commonest call left to NumPy, and the
    // cheapest to recognise.
    if (hk_is_absent(bound[MIN]) || hk_is_absent(bound[MAX])) return NULL;
    return hk_try_clip(self, bound[MIN], bound[MAX], bound[OUT]);
}

// The array's type is tested first: most calls are for another element type,
// and rejecting those must cost almost nothing. A call passed on reaches the
// clip loop, whose hook counts it, so it is not counted here; nor is an int64
// call passed on, which no hook counted before the binding had a loop for it.
static PyObject *method_clip(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (hk_is_flat(self, NPY_FLOAT) || hk_is_flat(self, NPY_DOUBLE) || hk_is_flat(self, NPY_INT64)) {
        PyObject *result = try_clip(self, args, nargs, kwnames);
        if (result != NULL) return result;
    }
    return ((hk_fast_keywords)(void (*)(void))methods[M_CLIP].original)(self, args, nargs, kwnames);
}

// -------------------------------------------------------------- accumulate

// The ufunc objects `np.maximum` and `np.fmax`, the two whose accumulation of
// one dimension has a kernel; `np.fmax` is also the one whose reduceat has one. NumPy keeps one
// of each, so a call is recognised by comparing `self` with them. Fetched
// when the first of the two hooks is applied, and kept.
static PyObject *maximum_ufunc;
static PyObject *fmax_ufunc;

// np.add and np.multiply, which have row loops (below).
static PyObject *add_ufunc;
static PyObject *multiply_ufunc;

// Fetch np.maximum, np.fmax, np.add and np.multiply. See hk_method.prepare.
// Also the preparation of method.cumsum, whose row loop is np.add's.
static int accumulate_prepare(void) {
    if (multiply_ufunc != NULL) return 0;
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return -1;
    PyObject *maximum = PyObject_GetAttrString(numpy, "maximum");
    PyObject *fmax = maximum != NULL ? PyObject_GetAttrString(numpy, "fmax") : NULL;
    PyObject *add = fmax != NULL ? PyObject_GetAttrString(numpy, "add") : NULL;
    PyObject *multiply = add != NULL ? PyObject_GetAttrString(numpy, "multiply") : NULL;
    Py_DECREF(numpy);
    if (multiply == NULL) {
        Py_XDECREF(maximum);
        Py_XDECREF(fmax);
        Py_XDECREF(add);
        return -1;
    }
    maximum_ufunc = maximum;
    fmax_ufunc = fmax;
    add_ufunc = add;
    multiply_ufunc = multiply;
    return 0;
}

// True when the `axis` argument of an accumulation names the only axis of a
// one-dimensional array as NumPy's default does: absent, or the Python int 0
// or -1, which NumPy turns into 0. None, a tuple and NumPy integers are
// NumPy's to interpret.
static bool accumulate_axis_is_first(PyObject *axis) {
    if (axis == NULL) return true;
    if (!PyLong_CheckExact(axis)) return false;
    int overflow = 0;
    const long value = PyLong_AsLongAndOverflow(axis, &overflow);
    return overflow == 0 && (value == 0 || value == -1);
}

// `ufunc` is np.maximum or np.fmax: the kernel for its accumulation of
// `array`, or HK_KERNEL_COUNT when there is none. int64 takes cummax_i64
// under either ufunc: for integers the two are one function, and NumPy returns
// int64 for both. float64 takes cumfmax_f64 under np.fmax only. np.maximum
// on floats propagates NaN, which neither kernel does.
static hk_kernel accumulate_kernel(PyObject *ufunc, PyObject *array) {
    if (hk_is_flat_1d(array, NPY_INT64)) return HK_CUMMAX_I64;
    if (ufunc == fmax_ufunc && hk_is_flat_1d(array, NPY_DOUBLE)) return HK_CUMFMAX_F64;
    return HK_KERNEL_COUNT;
}

// True when np.fmax.accumulate of the `n` doubles at `values` is left to
// NumPy: the array holds a signalling NaN, a negative zero, or NaNs of more
// than one bit pattern.
//
// cumfmax_f64 ignores every NaN; NumPy ignores a quiet one and answers a
// signalling one with a quiet NaN, through the C library's fmax. The other two
// are where NumPy's own answer depends on its build: which zero a running
// maximum of zeros of both signs is, and which NaN an output is while every
// element so far is NaN. NumPy's x86-64 and ARM64 builds give a different sign
// for the zero (seen in CI on Apple Silicon), and the kernel reproduces one
// build. Both conditions are broader than needed, since a negative zero
// matters only where a zero is the running maximum, and are kept simple.
//
// One pass with no branch. A NaN is the one value that differs from itself:
// `nan` is all ones for one. `set` and `clear` gather the bits set and clear
// in some NaN, so the NaNs share one pattern exactly when no bit is in both,
// and one is signalling when the quiet bit is among the clear ones. A
// negative zero is found by comparing the bits with its pattern. The NaN
// compare is the quiet kind: it raises the invalid-operation flag for a
// signalling NaN only, whose call is NumPy's. The bits are copied out with
// memcpy, which the compiler turns into a plain load. Compiled for the
// kernels' level, as hk_has_signalling_nan is, for the 64-bit compare;
// elsewhere the same loop is compiled for what the platform has. With AVX2
// it is two compares and five logical instructions per four elements: from
// Python at 4,096 elements the call takes 1.17 us, where it took 0.94 with
// the read for signalling NaNs alone (Intel Core i5-14600KF). Finding a zero
// with a floating-point compare and testing its sign bit took two
// instructions more per four elements, and the call 1.31 us.
//
// On ARM64 the same loop takes about 0.35 ns per element on the Apple M1 Pro,
// three times the 0.11 to 0.14 of the cumfmax_f64 kernel it guards, and the
// version below reads the array once more cheaply first.
#if defined(__aarch64__)

// ARM64: the same answer, and the same floating-point flags, a block of 32
// elements at a time (HK_GUARD_BLOCK). The block's top halfwords
// (hk_top_halves) say whether it holds a NaN or an infinity, all ones in the
// exponent, and whether it holds a top halfword of 0x8000, which a negative
// zero has (and a negative number too small to matter, which only costs a
// second look). Only then is the block read again for the NaN bit patterns,
// or for negative zeros, into masks across the whole array as on x86-64: the
// NaNs share one pattern when no bit is both set in one and clear in one.
// The NaN compare is the quiet kind and runs on every block with a NaN, so it
// raises the invalid-operation flag exactly when the x86-64 loop would: for a
// signalling NaN. The last block ends at n and overlaps the one before it,
// which reads some elements twice into masks that only gather. Fewer than 32
// elements are read by the loop of the x86-64 version.
//
// About 0.074 ns per element with no NaN, where the loop took 0.350, and
// 0.24 with 5% of the elements NaN (Apple M1 Pro, first-level cache).
static bool fmax_accumulate_left_to_numpy(const double *values, size_t n) {
    // The quiet bit of a NaN, the top bit of the fraction.
    const uint64_t quiet = 0x0008000000000000ULL;
    // The sign bit alone: the pattern of -0.0.
    const uint64_t sign = 0x8000000000000000ULL;
    if (n < HK_GUARD_BLOCK) {
        uint64_t set = 0, clear = 0, negative_zero = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t bits;
            memcpy(&bits, &values[i], sizeof bits);
            const uint64_t nan = values[i] != values[i] ? ~0ULL : 0;
            set |= bits & nan;
            clear |= ~bits & nan;
            negative_zero |= bits == sign ? ~0ULL : 0;
        }
        return (clear & quiet) != 0 || (set & clear) != 0 || negative_zero != 0;
    }
    // A top halfword shifted left by one, which drops the sign, is at least
    // this exactly for a NaN or an infinity.
    const uint16_t exponent_all_ones = 0xffe0;
    // The top halfword of -0.0.
    const uint16x8_t sign_half = vdupq_n_u16(0x8000);
    const uint64x2_t sign_bits = vdupq_n_u64(sign);
    // The bits set in some NaN; the bits set in every NaN (all ones while
    // there is none); all ones in a lane that has seen -0.0.
    uint64x2_t set = vdupq_n_u64(0), kept = vdupq_n_u64(~0ULL), negative_zero = vdupq_n_u64(0);
    for (size_t i = 0;; i += HK_GUARD_BLOCK) {
        if (i > n - HK_GUARD_BLOCK) i = n - HK_GUARD_BLOCK;
        const double *x = values + i;
        uint16x8_t top[4];
        hk_top_halves(x, top);
        // The largest top halfword without its sign, and the smallest top
        // halfword with its sign flipped, which is 0 for 0x8000.
        const uint16x8_t high = vmaxq_u16(vmaxq_u16(vshlq_n_u16(top[0], 1), vshlq_n_u16(top[1], 1)),
                                          vmaxq_u16(vshlq_n_u16(top[2], 1), vshlq_n_u16(top[3], 1)));
        const uint16x8_t flipped = vminq_u16(vminq_u16(veorq_u16(top[0], sign_half), veorq_u16(top[1], sign_half)),
                                             vminq_u16(veorq_u16(top[2], sign_half), veorq_u16(top[3], sign_half)));
        if (vmaxvq_u16(high) >= exponent_all_ones) {
            for (size_t k = 0; k < HK_GUARD_BLOCK; k += 2) {
                const float64x2_t v = vld1q_f64(x + k);
                const uint64x2_t bits = vreinterpretq_u64_f64(v);
                // All ones for a number, zero for a NaN.
                const uint64x2_t number = vceqq_f64(v, v);
                set = vorrq_u64(set, vbicq_u64(bits, number));
                kept = vandq_u64(kept, vorrq_u64(bits, number));
            }
        }
        if (vminvq_u16(flipped) == 0) {
            for (size_t k = 0; k < HK_GUARD_BLOCK; k += 2) {
                const uint64x2_t bits = vreinterpretq_u64_f64(vld1q_f64(x + k));
                negative_zero = vorrq_u64(negative_zero, vceqq_u64(bits, sign_bits));
            }
        }
        if (i == n - HK_GUARD_BLOCK) break;
    }
    const uint64_t set_bits = vgetq_lane_u64(set, 0) | vgetq_lane_u64(set, 1);
    const uint64_t clear_bits = ~(vgetq_lane_u64(kept, 0) & vgetq_lane_u64(kept, 1));
    const uint64_t zero_seen = vgetq_lane_u64(negative_zero, 0) | vgetq_lane_u64(negative_zero, 1);
    return (clear_bits & quiet) != 0 || (set_bits & clear_bits) != 0 || zero_seen != 0;
}

#else
static HK_TARGET_KERNEL_LEVEL bool fmax_accumulate_left_to_numpy(const double *values, size_t n) {
    // The quiet bit of a NaN, the top bit of the fraction.
    const uint64_t quiet = 0x0008000000000000ULL;
    // The sign bit alone: the pattern of -0.0.
    const uint64_t sign = 0x8000000000000000ULL;
    // 64-bit masks, as wide as an element: with bools the compiler leaves the
    // loop scalar.
    uint64_t set = 0, clear = 0, negative_zero = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        const uint64_t nan = values[i] != values[i] ? ~0ULL : 0;
        set |= bits & nan;
        clear |= ~bits & nan;
        negative_zero |= bits == sign ? ~0ULL : 0;
    }
    return (clear & quiet) != 0 || (set & clear) != 0 || negative_zero != 0;
}
#endif

// np.maximum.accumulate(x) or np.fmax.accumulate(x) on a one-dimensional
// contiguous array with at least one element, `axis` the first, and `dtype`
// and `out` absent or None. The parameters are NumPy's, which takes all four
// by position or by keyword. The result is a new array of the input's type
// and length, so it cannot overlap the input, which both kernels require.
//
// A float64 array is read first (fmax_accumulate_left_to_numpy), and one
// that holds a signalling NaN, a negative zero, or NaNs of several patterns
// is NumPy's. The kernel's vmaxpd sets the invalid-operation flag for every
// NaN operand, quiet ones included, and NumPy's own loop leaves no flag
// behind, so the flags are cleared after it (hk_clear_float_flags), as after
// the clip kernels.
static HK_NOINLINE PyObject *try_accumulate(PyObject *ufunc, PyObject *const *args, Py_ssize_t nargs,
                                            PyObject *kwnames) {
    enum { ARRAY, AXIS, DTYPE, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"array", "axis", "dtype", "out"};
    // The array is nearly always the first positional argument, and its type
    // decides most calls; it is tested before the arguments are matched.
    if (nargs >= 1 && accumulate_kernel(ufunc, args[0]) == HK_KERNEL_COUNT) return NULL;
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[ARRAY] == NULL) return NULL;
    const hk_kernel kernel = accumulate_kernel(ufunc, bound[ARRAY]);
    if (kernel == HK_KERNEL_COUNT || !accumulate_axis_is_first(bound[AXIS]) || !hk_is_absent(bound[DTYPE]) ||
        !hk_is_absent(bound[OUT])) {
        return NULL;
    }
    PyArrayObject *x = (PyArrayObject *)bound[ARRAY];
    npy_intp n = PyArray_SIZE(x);
    if (n < 1) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &n, PyArray_TYPE(x));
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    void *out = PyArray_DATA((PyArrayObject *)result);
    // True when the float64 input is NumPy's and the call is passed on.
    bool left = false;
    {
        HK_BEGIN_NOGIL(n)
        if (kernel == HK_CUMMAX_I64) {
            cummax_i64(PyArray_DATA(x), out, (size_t)n);
        } else {
            left = fmax_accumulate_left_to_numpy(PyArray_DATA(x), (size_t)n);
            if (!left) {
                cumfmax_f64(PyArray_DATA(x), out, (size_t)n);
                hk_clear_float_flags();
            }
        }
        HK_END_NOGIL
    }
    if (left) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(kernel);
    return result;
}

// ------------------------------------- accumulation along the first axis
//
// np.add, np.multiply, np.maximum and np.fmax accumulated along axis 0 of a
// C-ordered float64 array of two or more dimensions, and x.cumsum(axis=0),
// a row at a time. See docs/accumulate-rows-design.md.
//
// NumPy hands its inner loop one column at a time, a row apart between
// elements, and computes out[i] = op(out[i-1], x[i]) down the column. The
// loops below compute the same operation on the same two operands for every
// element, along the rows, which is about six times faster on a matrix of
// thousands of columns. Only where an operation is not a function of its
// operands' values can the result differ:
//
// - add and multiply: which NaN results from two NaNs of different bits, and
//   the floating-point flags NumPy reports. The flags are read after each
//   block of rows, and each row says whether it made a NaN; at the first
//   block that did either the rest of the call is handed to NumPy
//   (rows_numpy_rest).
// - maximum and fmax: the sign of a zero maximum of zeros of both signs, and
//   which NaN wins among NaNs of different bits. The array is read first by
//   fmax_accumulate_left_to_numpy, and one that holds a negative zero, a
//   signalling NaN or NaNs of several patterns is NumPy's.

// The four operations, in the order of ROW_FUNCTIONS and ROW_KERNELS.
typedef enum { ROWS_ADD, ROWS_MULTIPLY, ROWS_MAXIMUM, ROWS_FMAX, ROWS_OPERATION_COUNT } rows_operation;

// Bytes of input taken a block at a time: as many whole rows as fit, and one
// row when a row is longer. In place, a block is computed into a scratch
// buffer, which stays in the first-level cache, and then copied over the
// input; for add and multiply the flags are read once per block.
#define ROWS_BLOCK_BYTES 16384

// One row of an accumulation: out[j] = op(prev[j], x[j]) for the `cols`
// columns, where `prev` is the output's previous row. No two of the pointers
// overlap, which lets the compiler make vector code of each loop. Returns all
// ones when add or multiply made a NaN in the row, and otherwise 0; maximum
// and fmax return 0. Compiled for the kernels' level.
typedef uint64_t (*row_function)(const double *restrict prev, const double *restrict x, double *restrict out,
                                 size_t cols);

// The masks are 64-bit, as wide as an element: with a bool the compiler
// leaves the loop scalar on x86-64 (see fmax_accumulate_left_to_numpy).
static HK_TARGET_KERNEL_LEVEL uint64_t add_row(const double *restrict prev, const double *restrict x,
                                               double *restrict out, size_t cols) {
    uint64_t nan = 0;
    for (size_t j = 0; j < cols; j++) {
        const double sum = prev[j] + x[j];
        out[j] = sum;
        nan |= sum != sum ? ~0ULL : 0;
    }
    return nan;
}

static HK_TARGET_KERNEL_LEVEL uint64_t multiply_row(const double *restrict prev, const double *restrict x,
                                                    double *restrict out, size_t cols) {
    uint64_t nan = 0;
    for (size_t j = 0; j < cols; j++) {
        const double product = prev[j] * x[j];
        out[j] = product;
        nan |= product != product ? ~0ULL : 0;
    }
    return nan;
}

// np.maximum propagates a NaN, and np.fmax skips one: the other operand, or
// the NaN when both are. With the array guarded, every NaN in it has one bit
// pattern and is quiet, and no zero is negative, so any operation that
// propagates (or skips) a NaN and takes the larger number gives NumPy's bits.
// The array is read by the guard before these run, which is about a quarter
// of the call (0.55 ms against 0.42 for the product on a 253 x 4,789 matrix,
// Apple M1 Pro); the loops themselves written with ARM64's fmax and fmaxnm
// instructions took the same time as this plain C.
static HK_TARGET_KERNEL_LEVEL uint64_t maximum_row(const double *restrict prev, const double *restrict x,
                                                   double *restrict out, size_t cols) {
    for (size_t j = 0; j < cols; j++) {
        const double a = prev[j], b = x[j];
        out[j] = a != a ? a : (b != b ? b : (b > a ? b : a));
    }
    return 0;
}

static HK_TARGET_KERNEL_LEVEL uint64_t fmax_row(const double *restrict prev, const double *restrict x,
                                                double *restrict out, size_t cols) {
    for (size_t j = 0; j < cols; j++) {
        const double a = prev[j], b = x[j];
        out[j] = b != b ? a : (a != a ? b : (b > a ? b : a));
    }
    return 0;
}

static const row_function ROW_FUNCTIONS[ROWS_OPERATION_COUNT] = {add_row, multiply_row, maximum_row, fmax_row};

// The counter of each operation's loop.
static const hk_kernel ROW_KERNELS[ROWS_OPERATION_COUNT] = {
    HK_CUMSUM_ROWS_F64, HK_CUMPROD_ROWS_F64, HK_CUMMAX_ROWS_F64, HK_CUMFMAX_ROWS_F64,
};

// Accumulate the `rows` rows of `cols` doubles at `x` into `out`, which is
// either `x` itself or shares no byte with it, `block_rows` rows at a time.
// In place, each block is computed into `scratch`, which holds a block, and
// then copied over the input's rows: computing straight into the rows the
// input was just read from took 2.4 to 9 ms on a 253 x 4,789 product where
// this takes 0.5 to 0.8 (Apple M1 Pro, a loop in C outside the binding), and
// copying the input's block aside first and computing in place was as slow.
// `scratch` is not used otherwise.
//
// Returns `rows` when every row is done. For add and multiply, returns
// instead the first row of the first block that made a NaN or raised a flag
// NumPy reports, with `out` from that row on holding the input's rows as
// they were: rows before it hold the accumulation. The flags are left clear
// either way, as NumPy's loops leave them. Touches no Python object, and runs
// with the interpreter lock released.
static size_t accumulate_rows(rows_operation op, const double *x, double *out, double *scratch, size_t rows,
                              size_t cols, size_t block_rows) {
    const bool in_place = x == out;
    const bool checked = op == ROWS_ADD || op == ROWS_MULTIPLY;
    const row_function row = ROW_FUNCTIONS[op];
    const size_t row_bytes = cols * sizeof(double);
    if (!in_place) memcpy(out, x, row_bytes);
    hk_clear_float_flags();
    for (size_t b = 1; b < rows; b += block_rows) {
        const size_t count = rows - b < block_rows ? rows - b : block_rows;
        // Where the block's rows are computed: the output, or the scratch
        // buffer in place.
        double *target = in_place ? scratch : out + b * cols;
        uint64_t nan = 0;
        for (size_t i = 0; i < count; i++) {
            const double *prev = i == 0 ? out + (b - 1) * cols : target + (i - 1) * cols;
            nan |= row(prev, x + (b + i) * cols, target + i * cols, cols);
        }
        // The rows were computed by calls the compiler cannot see into, so
        // the flags are read after all of their arithmetic.
        if (checked && (nan != 0 || hk_float_flags_reported())) {
            if (!in_place) memcpy(out + b * cols, x + b * cols, (rows - b) * row_bytes);
            hk_clear_float_flags();
            return b;
        }
        if (in_place) memcpy(out + b * cols, scratch, count * row_bytes);
    }
    hk_clear_float_flags();
    return rows;
}

// True when `axis` names the first axis of an array of `dimensions`
// dimensions as a Python int: 0 or -dimensions. Absent, it is the first axis
// when `absent_is_first`, as for accumulate, and not for cumsum, whose default
// flattens. None, a tuple and NumPy integers are NumPy's to interpret.
static bool rows_axis_is_first(PyObject *axis, int dimensions, bool absent_is_first) {
    if (axis == NULL) return absent_is_first;
    if (!PyLong_CheckExact(axis)) return false;
    int overflow = 0;
    const long value = PyLong_AsLongAndOverflow(axis, &overflow);
    return overflow == 0 && (value == 0 || value == -dimensions);
}

// True when `out`, given for the result of accumulating `x`, can be written
// by the loop: a plain float64 ndarray in native byte order, C-contiguous and
// writable, of `x`'s shape, and either `x`'s memory exactly or sharing no
// byte with it. A partial overlap is NumPy's.
static bool rows_output_fits(PyArrayObject *x, PyObject *out) {
    if (!hk_is_flat(out, NPY_DOUBLE)) return false;
    PyArrayObject *o = (PyArrayObject *)out;
    if (!PyArray_ISWRITEABLE(o) || !hk_same_shape(x, o)) return false;
    const size_t bytes = (size_t)PyArray_NBYTES(x);
    return PyArray_DATA(o) == PyArray_DATA(x) || hk_disjoint(PyArray_DATA(o), bytes, PyArray_DATA(x), bytes);
}

// NumPy's own `ufunc.accumulate`: the function the hook replaced while it is
// applied, so that the call never comes back to the binding, and the method
// as NumPy has it otherwise (when only method.cumsum is applied). Returns a
// new reference, or NULL with NumPy's exception set.
static PyObject *numpy_accumulate(PyObject *ufunc, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (methods[M_ACCUMULATE].definition != NULL) {
        return ((hk_fast_keywords)(void (*)(void))methods[M_ACCUMULATE].original)(ufunc, args, nargs, kwnames);
    }
    PyObject *method = PyObject_GetAttrString(ufunc, "accumulate");
    if (method == NULL) return NULL;
    PyObject *result = PyObject_Vectorcall(method, args, (size_t)nargs, kwnames);
    Py_DECREF(method);
    return result;
}

// Hand the rows of `out` from `stop` on to NumPy, where accumulate_rows
// stopped: `ufunc.accumulate(out[stop-1:], 0, out=out[stop-1:])`. Row
// stop - 1 holds the accumulation up to it and the rows after it the input,
// so NumPy computes what it would have for the whole call, and warns or
// raises as np.errstate says for the operations that raised a flag; none did
// before row `stop`. Returns 0, or -1 with NumPy's exception set.
static HK_NOINLINE int rows_numpy_rest(PyObject *ufunc, PyArrayObject *out, npy_intp stop) {
    PyObject *view = PySequence_GetSlice((PyObject *)out, stop - 1, PyArray_DIM(out, 0));
    PyObject *axis = PyLong_FromLong(0);
    PyObject *kwnames = Py_BuildValue("(s)", "out");
    PyObject *result = NULL;
    if (view != NULL && axis != NULL && kwnames != NULL) {
        PyObject *const args[] = {view, axis, view};
        result = numpy_accumulate(ufunc, args, 2, kwnames);
    }
    Py_XDECREF(view);
    Py_XDECREF(axis);
    Py_XDECREF(kwnames);
    if (result == NULL) return -1;
    Py_DECREF(result);
    return 0;
}

// The accumulation of `array` by `op` along its first axis, given the call's
// `axis`, `dtype` and `out` (NULL when not given), when the call is one the
// loop takes: `array` a plain C-ordered float64 ndarray of two or more
// dimensions with at least one row and one column, `axis` the first,
// `dtype` absent or None, and `out` absent, None, or an array
// rows_output_fits. `ufunc` is the ufunc of `op`, and `operation` the
// counter a call handed to NumPy part way is counted under.
//
// Sets *answered to true when the call was answered here: the return value is
// then the result (a new array, or `out` with a new reference), or NULL with
// NumPy's exception set when NumPy computed part of it and raised. Otherwise
// returns NULL with no exception set and nothing written, and the call is
// NumPy's.
static HK_NOINLINE PyObject *try_rows(rows_operation op, PyObject *array, PyObject *axis, bool absent_is_first,
                                      PyObject *dtype, PyObject *out_argument, PyObject *ufunc,
                                      hk_operation operation, bool *answered) {
    *answered = false;
    if (!hk_is_flat(array, NPY_DOUBLE)) return NULL;
    PyArrayObject *x = (PyArrayObject *)array;
    const int dimensions = PyArray_NDIM(x);
    if (dimensions < 2 || !rows_axis_is_first(axis, dimensions, absent_is_first) || !hk_is_absent(dtype)) {
        return NULL;
    }
    const npy_intp rows = PyArray_DIM(x, 0), n = PyArray_SIZE(x);
    if (rows < 1 || n < 1) return NULL;
    const size_t cols = (size_t)(n / rows);
    PyArrayObject *out;
    if (hk_is_absent(out_argument)) {
        out = (PyArrayObject *)PyArray_SimpleNew(dimensions, PyArray_DIMS(x), NPY_DOUBLE);
        if (out == NULL) {
            PyErr_Clear();
            return NULL;
        }
    } else {
        if (!rows_output_fits(x, out_argument)) return NULL;
        out = (PyArrayObject *)out_argument;
        Py_INCREF(out);
    }
    const bool in_place = PyArray_DATA(out) == PyArray_DATA(x);
    const size_t row_bytes = cols * sizeof(double);
    const size_t block_rows = row_bytes >= ROWS_BLOCK_BYTES ? 1 : ROWS_BLOCK_BYTES / row_bytes;
    double on_stack[HK_STACK_SCRATCH_BYTES / sizeof(double)];
    double *scratch = NULL;
    if (in_place) {
        const size_t scratch_bytes = block_rows * row_bytes;
        scratch = scratch_bytes <= sizeof on_stack ? on_stack : malloc(scratch_bytes);
        if (scratch == NULL) {
            Py_DECREF(out);
            return NULL;
        }
    }
    // True when the guard of maximum and fmax leaves the array to NumPy.
    bool left = false;
    size_t stop = (size_t)rows;
    {
        HK_BEGIN_NOGIL(n)
        if (op == ROWS_MAXIMUM || op == ROWS_FMAX) left = fmax_accumulate_left_to_numpy(PyArray_DATA(x), (size_t)n);
        if (!left) {
            stop = accumulate_rows(op, PyArray_DATA(x), PyArray_DATA(out), scratch, (size_t)rows, cols, block_rows);
        }
        HK_END_NOGIL
    }
    if (scratch != NULL && scratch != on_stack) free(scratch);
    if (left) {
        Py_DECREF(out);
        return NULL;
    }
    *answered = true;
    if (stop == (size_t)rows) {
        hk_take(ROW_KERNELS[op]);
        return (PyObject *)out;
    }
    hk_pass(operation);
    if (rows_numpy_rest(ufunc, out, (npy_intp)stop) != 0) {
        Py_DECREF(out);
        return NULL;
    }
    return (PyObject *)out;
}

// The operation of a ufunc that has a row loop, or ROWS_OPERATION_COUNT.
static rows_operation rows_operation_of(PyObject *ufunc) {
    if (ufunc == add_ufunc) return ROWS_ADD;
    if (ufunc == multiply_ufunc) return ROWS_MULTIPLY;
    if (ufunc == maximum_ufunc) return ROWS_MAXIMUM;
    if (ufunc == fmax_ufunc) return ROWS_FMAX;
    return ROWS_OPERATION_COUNT;
}

// `ufunc.accumulate(array, axis, dtype, out)` along the first axis, for the
// four ufuncs with a row loop: try_rows on the call's arguments, which NumPy
// takes by position or by keyword. See try_rows for *answered and the result.
static HK_NOINLINE PyObject *try_accumulate_rows(PyObject *ufunc, PyObject *const *args, Py_ssize_t nargs,
                                                 PyObject *kwnames, bool *answered) {
    enum { ARRAY, AXIS, DTYPE, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"array", "axis", "dtype", "out"};
    *answered = false;
    // The array is nearly always the first positional argument, and its type
    // and shape decide most calls; it is tested before the arguments are
    // matched.
    if (nargs >= 1 && !(hk_is_flat(args[0], NPY_DOUBLE) && PyArray_NDIM((PyArrayObject *)args[0]) >= 2)) {
        return NULL;
    }
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[ARRAY] == NULL) return NULL;
    return try_rows(rows_operation_of(ufunc), bound[ARRAY], bound[AXIS], true, bound[DTYPE], bound[OUT], ufunc,
                    HK_OP_ACCUMULATE, answered);
}

// Every ufunc's accumulate arrives here. np.maximum and np.fmax have kernels
// for one dimension and the row loops for more; a call to either that is
// not taken is counted. np.add and np.multiply have the row loops; a call to
// either that is not taken goes on uncounted, since nearly all of them are
// accumulations of one dimension that nothing serves, and np.cumsum counts
// its own at its method. Any other ufunc goes straight on after four pointer
// comparisons, uncounted.
static PyObject *method_accumulate(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    bool answered;
    if (self == maximum_ufunc || self == fmax_ufunc) {
        PyObject *result = try_accumulate(self, args, nargs, kwnames);
        if (result != NULL) return result;
        result = try_accumulate_rows(self, args, nargs, kwnames, &answered);
        if (answered) return result;
        hk_pass(HK_OP_ACCUMULATE);
    } else if (self == add_ufunc || self == multiply_ufunc) {
        PyObject *result = try_accumulate_rows(self, args, nargs, kwnames, &answered);
        if (answered) return result;
    }
    return ((hk_fast_keywords)(void (*)(void))methods[M_ACCUMULATE].original)(self, args, nargs, kwnames);
}

// x.cumsum(axis, dtype, out) along the first axis of a matrix: try_rows for
// np.add, the arguments bound by name from the tuple and dictionary the
// method is called with. `axis` must be given: without it cumsum flattens.
// See try_rows for *answered and the result.
static HK_NOINLINE PyObject *try_cumsum_rows(PyObject *self, PyObject *args, PyObject *kwargs, bool *answered) {
    *answered = false;
    const Py_ssize_t positional = PyTuple_GET_SIZE(args);
    if (positional > CUMSUM_PARAMETER_COUNT) return NULL;
    PyObject *bound[CUMSUM_PARAMETER_COUNT] = {NULL, NULL, NULL};
    for (Py_ssize_t i = 0; i < positional; i++) bound[i] = PyTuple_GET_ITEM(args, i);
    if (kwargs != NULL) {
        PyObject *key, *value;
        Py_ssize_t position = 0;
        while (PyDict_Next(kwargs, &position, &key, &value)) {
            Py_ssize_t i = 0;
            while (i < CUMSUM_PARAMETER_COUNT && !(PyUnicode_CheckExact(key) && hk_text_is(key, CUMSUM_PARAMETERS[i]))) {
                i++;
            }
            if (i == CUMSUM_PARAMETER_COUNT || bound[i] != NULL) return NULL;
            bound[i] = value;
        }
    }
    return try_rows(ROWS_ADD, self, bound[0], false, bound[1], bound[2], add_ufunc, HK_OP_CUMSUM, answered);
}

// ---------------------------------------------------------------- reduceat

// Elements of the array read for signalling NaNs at a time, ahead of the runs
// that use them: 16,384 bytes, so that the kernel then reads them from the
// first-level cache. Checking the whole array first read an array beyond the
// caches twice, and np.fmax.reduceat on runs of 1,024 elements of a 262,144
// element array was 0.6x of NumPy (Intel Core i5-14600KF).
#define REDUCEAT_CHECK_BLOCK 2048

// Read `x` for signalling NaNs from `*checked`, up to which it has been read,
// to at least `upto`, which is above it: a block of REDUCEAT_CHECK_BLOCK
// elements, or more if `upto` is further, and never past `n`. Returns false
// when a signalling NaN was found, and advances `*checked` otherwise.
static bool check_ahead(const double *x, uint64_t n, uint64_t upto, uint64_t *checked) {
    uint64_t to = upto - *checked < REDUCEAT_CHECK_BLOCK ? *checked + REDUCEAT_CHECK_BLOCK : upto;
    if (to > n) to = n;
    if (hk_has_signalling_nan(x + *checked, (size_t)(to - *checked))) return false;
    *checked = to;
    return true;
}

// Every run of np.fmax.reduceat(x, indices), into `out`, one element per
// index. NumPy's rule for run i: it ends at indices[i + 1], or at n for the
// last run; if it starts before it ends, its result is the reduction of
// x[indices[i]:end], and otherwise the single element x[indices[i]]. A run of
// one element is that element too, so the kernel, which needs at least one,
// is called only for runs of two or more, and a single element is copied
// with its bits.
//
// Each index is read once, into a register, and tested there against [0, n)
// before it is used: as unsigned numbers, so that a negative index fails the
// same test. Another thread may change the indices while this runs, and a
// second read could give an index that was never tested.
//
// `x` is read for signalling NaNs a block at a time, just ahead of the
// elements a run uses (check_ahead): everything below the furthest element
// any run has used. The last run ends at n, so the whole array is read by
// the end. A signalling NaN anywhere sends the call to NumPy: NumPy's fmax
// answers one with a quiet NaN, where the kernel would skip it.
//
// Returns false at the first index outside [0, n), whose call is NumPy's (it
// raises IndexError), and when `x` holds a signalling NaN. Inside the loop
// the only call besides the kernel is the check, once per block of
// elements. The loop runs once per run of perhaps 16 elements, where the
// kernel takes about 1.5 ns.
static bool fmax_runs(const double *x, uint64_t n, const npy_intp *indices, size_t m, double *out) {
    uint64_t checked = 0;
    uint64_t start = (uint64_t)indices[0];
    if (start >= n) return false;
    for (size_t i = 0; i + 1 < m; i++) {
        const uint64_t end = (uint64_t)indices[i + 1];
        if (end >= n) return false;
        // One past the last element the run reads.
        const uint64_t last = end > start + 1 ? end : start + 1;
        if (last > checked && !check_ahead(x, n, last, &checked)) return false;
        out[i] = end > start + 1 ? fmax_reduce_f64(x + start, (size_t)(end - start)) : x[start];
        start = end;
    }
    if (n > checked && !check_ahead(x, n, n, &checked)) return false;
    out[m - 1] = n > start + 1 ? fmax_reduce_f64(x + start, (size_t)(n - start)) : x[start];
    return true;
}

// True when any of the `n` doubles at `values` is a negative zero, the one
// pattern that is the sign bit alone. Compiled for the kernels' level, as
// hk_has_signalling_nan is, for a 64-bit compare.
#if defined(__aarch64__)

// Elements per step of the ARM64 version: eight vectors of two.
#define NEGATIVE_ZERO_STEP 16

// ARM64: the same test with eight independent masks, a compare and an OR per
// two elements, where clang's loop narrows each compare first. The last step
// ends at n and overlaps the one before it; fewer than 16 elements are tested
// one at a time. About 0.078 ns per element, where the plain loop took 0.117
// (Apple M1 Pro, first-level cache).
static bool has_negative_zero(const double *values, size_t n) {
    // The pattern of -0.0.
    const uint64_t negative_zero = 0x8000000000000000ULL;
    if (n < NEGATIVE_ZERO_STEP) {
        uint64_t seen = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t bits;
            memcpy(&bits, &values[i], sizeof bits);
            seen |= bits == negative_zero ? ~0ULL : 0;
        }
        return seen != 0;
    }
    const uint64x2_t pattern = vdupq_n_u64(negative_zero);
    uint64x2_t seen[NEGATIVE_ZERO_STEP / 2];
    for (size_t k = 0; k < NEGATIVE_ZERO_STEP / 2; k++) seen[k] = vdupq_n_u64(0);
    for (size_t i = 0;; i += NEGATIVE_ZERO_STEP) {
        if (i > n - NEGATIVE_ZERO_STEP) i = n - NEGATIVE_ZERO_STEP;
        for (size_t k = 0; k < NEGATIVE_ZERO_STEP / 2; k++) {
            const uint64x2_t bits = vreinterpretq_u64_f64(vld1q_f64(values + i + 2 * k));
            seen[k] = vorrq_u64(seen[k], vceqq_u64(bits, pattern));
        }
        if (i == n - NEGATIVE_ZERO_STEP) break;
    }
    uint64x2_t any = seen[0];
    for (size_t k = 1; k < NEGATIVE_ZERO_STEP / 2; k++) any = vorrq_u64(any, seen[k]);
    return (vgetq_lane_u64(any, 0) | vgetq_lane_u64(any, 1)) != 0;
}

#else
static HK_TARGET_KERNEL_LEVEL bool has_negative_zero(const double *values, size_t n) {
    // The pattern of -0.0.
    const uint64_t negative_zero = 0x8000000000000000ULL;
    // A 64-bit mask, as wide as an element: with a bool the compiler leaves
    // the loop scalar.
    uint64_t seen = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        seen |= bits == negative_zero ? ~0ULL : 0;
    }
    return seen != 0;
}
#endif

// True when the NaNs among the `n` doubles at `values` do not all have the
// same bit pattern. A NaN is found as in hk_has_signalling_nan; `set` gathers
// the bits set in some NaN and `clear` the bits clear in some NaN, and the
// NaNs all have one pattern exactly when no bit is in both.
static HK_TARGET_KERNEL_LEVEL bool nans_are_mixed(const double *values, size_t n) {
    // Clears the sign bit.
    const uint64_t magnitude = 0x7fffffffffffffffULL;
    // The pattern of infinity, which the magnitude of every NaN is above.
    const int64_t infinity = 0x7ff0000000000000LL;
    uint64_t set = 0, clear = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        const uint64_t nan = (int64_t)(bits & magnitude) > infinity ? ~0ULL : 0;
        set |= bits & nan;
        clear |= ~bits & nan;
    }
    return (set & clear) != 0;
}

// True when the results of the runs of `x`, `out[0..m)`, hold one whose bits
// NumPy does not fix apart from the lane structure of its vector loop: a
// zero, when `x` holds a negative zero, or a NaN, when the NaNs of `x` are
// not all of one pattern. For a zero maximum with zeros of both signs
// present, or a run that is all NaN with different NaNs, NumPy's answer
// depends on which lane of its loop each element falls in, and NumPy picks
// its AVX2 or its AVX-512 loop when it runs, which have four and eight
// lanes, and its ARM64 loop a chain of fmaxnm. Everywhere else every correct
// maximum is the same number. The kernel reproduces the AVX2 loop; the
// binding does not depend on it, and leaves such calls to NumPy. The spec's
// domain is this condition, one run at a time, so the judge checks the
// kernel only where every loop agrees.
//
// The results are read first, which is short: one per run. `x` is read again
// only when a result is a zero or a NaN. The comparisons raise no flag for a
// quiet NaN, and `out` holds no other kind.
static bool fmax_runs_ambiguous(const double *x, size_t n, const double *out, size_t m) {
    bool zero = false, nan = false;
    for (size_t i = 0; i < m; i++) {
        zero |= out[i] == 0.0;
        nan |= out[i] != out[i];
    }
    return (zero && has_negative_zero(x, n)) || (nan && nans_are_mixed(x, n));
}

// np.fmax.reduceat(x, indices) on a one-dimensional contiguous float64 array
// with at least one element, `indices` a one-dimensional contiguous array of
// np.intp (int64) with at least one element, `axis` the first, and `dtype`
// and `out` absent or None. The parameters are NumPy's, which takes all five
// by position or by keyword. Indices of another type, and lists, are NumPy's
// to convert. The result is a new float64 array with one element per index.
//
// Three guards, all of which send the whole call to NumPy: a signalling NaN
// anywhere in `x`, looked for as the runs go (fmax_runs); and, after the
// runs, a zero result when `x` holds a negative zero, and a NaN result when
// its NaNs are of more than one pattern (fmax_runs_ambiguous). The kernel's vmaxpd sets the
// invalid-operation flag for every NaN operand, quiet ones included, and
// NumPy's own loop leaves no flag behind, so the flags are cleared after the
// kernel (hk_clear_float_flags), as after the clip kernels.
static HK_NOINLINE PyObject *try_reduceat(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { ARRAY, INDICES, AXIS, DTYPE, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"array", "indices", "axis", "dtype", "out"};
    // The array is nearly always the first positional argument, and its type
    // decides most calls; it is tested before the arguments are matched.
    if (nargs >= 1 && !hk_is_flat_1d(args[0], NPY_DOUBLE)) return NULL;
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[ARRAY] == NULL || bound[INDICES] == NULL) return NULL;
    if (!hk_is_flat_1d(bound[ARRAY], NPY_DOUBLE) || !hk_is_flat_1d(bound[INDICES], NPY_INTP) ||
        !accumulate_axis_is_first(bound[AXIS]) || !hk_is_absent(bound[DTYPE]) || !hk_is_absent(bound[OUT])) {
        return NULL;
    }
    PyArrayObject *x = (PyArrayObject *)bound[ARRAY], *indices = (PyArrayObject *)bound[INDICES];
    const npy_intp n = PyArray_SIZE(x);
    npy_intp m = PyArray_SIZE(indices);
    if (n < 1 || m < 1) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &m, NPY_DOUBLE);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const double *data = PyArray_DATA(x);
    double *out = PyArray_DATA((PyArrayObject *)result);
    // True when the kernel's results stand and the call is taken.
    bool taken = false;
    {
        HK_BEGIN_NOGIL(n)
        taken = fmax_runs(data, (uint64_t)n, PyArray_DATA(indices), (size_t)m, out);
        hk_clear_float_flags();
        taken = taken && !fmax_runs_ambiguous(data, (size_t)n, out, (size_t)m);
        HK_END_NOGIL
    }
    if (!taken) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_FMAX_REDUCE_F64);
    return result;
}

// Every ufunc's reduceat arrives here. Any ufunc but np.fmax goes straight on
// after one pointer comparison, uncounted: no kernel serves it. A call to
// np.fmax that is not taken is counted.
static PyObject *method_reduceat(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (self == fmax_ufunc) {
        PyObject *result = try_reduceat(args, nargs, kwnames);
        if (result != NULL) return result;
        hk_pass(HK_OP_REDUCEAT);
    }
    return ((hk_fast_keywords)(void (*)(void))methods[M_REDUCEAT].original)(self, args, nargs, kwnames);
}

// --------------------------------------------------------------------- any

// Bytes of the matrix that are checked and then reduced at a time: a whole
// number of rows, and one row when a row is longer. Each block is read twice,
// so it is a third of the smallest first-level data cache the kernels run on
// (48 KiB on the x86-64 performance cores), as for np.sum on a mask.
#define ANY_ROWS_BLOCK 16384

// out[r] = whether row r of the `rows` rows of `cols` bytes at `m` holds a
// non-zero byte, block by block: each block's bytes are checked for values
// other than 0 and 1 (hk_mask_is_clean) just before the kernel reads them
// from the first-level cache. Returns false at the first block with such a
// byte, having written some of `out`; the caller then drops it.
//
// Compiled for the kernels' level, so that the check inlined here is AVX2
// code: the kernel takes about 0.013 ns per element on rows of 22, and the
// check compiled for any x86-64 would cost about as much again (see
// hk_has_signalling_nan for the rule this follows). No Python object is
// touched here; it runs with the interpreter lock released.
static HK_TARGET_KERNEL_LEVEL bool any_rows_blocks(const uint8_t *m, bool *out, size_t rows, size_t cols) {
    // A matrix that fits in one block is one block, without the division,
    // which is a few nanoseconds of a call on a small matrix.
    const size_t block_rows = rows * cols <= ANY_ROWS_BLOCK ? rows
                              : cols < ANY_ROWS_BLOCK      ? ANY_ROWS_BLOCK / cols
                                                           : 1;
    for (size_t done = 0; done < rows; done += block_rows) {
        const size_t count = rows - done < block_rows ? rows - done : block_rows;
        const uint8_t *block = m + done * cols;
        if (!hk_mask_is_clean(block, count * cols)) return false;
        // The kernel writes exactly `count` results: n is a whole number of
        // rows, so there is no partial row for it to skip.
        any_rows_bool((const bool *)block, out + done, count * cols, cols);
    }
    return true;
}

// m.any(axis=1) for a two-dimensional boolean array, by either spelling: the
// caller has tested that `mask` is a plain ndarray of bool with two
// dimensions (hk_is_bool_matrix) and that `axis` was 1 or -1 with no other
// argument. Shared with the function hook for np.any(m, axis=1).
//
// The result is a new one-dimensional boolean array of one element per row,
// as NumPy's is. The kernel needs C order and at least one column; with zero
// columns NumPy's answer is all False and the call is NumPy's. Zero rows are
// taken: the kernel writes nothing. The new array cannot overlap the mask,
// which the kernel requires.
//
// Every byte is checked to be 0 or 1 before the kernel reads it. NumPy takes
// any non-zero byte as True, and on one of the kernel's paths (fewer than 32
// elements, rows of other than four) a byte above 1 changes the results of
// the rows beside it; a mask that holds one is NumPy's.
//
// The mask is read twice, by the check and by the kernel, and the lock is
// released for large ones all the same, as for the other mask kernels: a
// thread that changes the mask between the two reads can at worst make the
// result of a row wrong. The kernel stays inside its buffers whatever the
// bytes.
HK_NOINLINE PyObject *hk_try_any_rows(PyObject *mask) {
    PyArrayObject *m = (PyArrayObject *)mask;
    if (!PyArray_IS_C_CONTIGUOUS(m)) return NULL;
    npy_intp rows = PyArray_DIM(m, 0);
    const npy_intp cols = PyArray_DIM(m, 1);
    if (cols < 1) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &rows, NPY_BOOL);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    bool clean;
    {
        HK_BEGIN_NOGIL(rows * cols)
        clean = any_rows_blocks(PyArray_DATA(m), PyArray_DATA((PyArrayObject *)result), (size_t)rows,
                                (size_t)cols);
        HK_END_NOGIL
    }
    if (!clean) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_ANY_ROWS_BOOL);
    return result;
}

// The work of method_any once the array is a boolean matrix: the arguments
// must be `axis` alone, by position or by keyword, the Python int 1 or -1.
// The method passes its arguments to NumPy's `_methods._any`, whose
// parameters are axis, dtype, out, keepdims and where; any of the others,
// even at its default, leaves the call to NumPy.
static HK_NOINLINE PyObject *try_any(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    const Py_ssize_t keywords = kwnames == NULL ? 0 : PyTuple_GET_SIZE(kwnames);
    if (nargs + keywords != 1) return NULL;
    if (keywords == 1 && !hk_text_is(PyTuple_GET_ITEM(kwnames, 0), "axis")) return NULL;
    // A keyword's value follows the positional arguments, so either way the
    // one argument is args[0].
    if (!hk_is_last_axis(args[0])) return NULL;
    return hk_try_any_rows(self);
}

// m.any(axis=1). `m.any()` with no argument is one of the commonest NumPy
// calls there is and has no kernel: it goes on at the first test, uncounted.
// So does any call on an array that is not a plain boolean one of two
// dimensions. A call on such an array with arguments that is not taken is
// counted.
static PyObject *method_any(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if ((nargs != 0 || kwnames != NULL) && hk_is_bool_matrix(self)) {
        PyObject *result = try_any(self, args, nargs, kwnames);
        if (result != NULL) return result;
        hk_pass(HK_OP_ANY);
    }
    return ((hk_fast_keywords)(void (*)(void))methods[M_ANY].original)(self, args, nargs, kwnames);
}

// ------------------------------------------------------------------- round
//
// No kernel serves np.round: on the short arrays it is called on, nearly all
// of a call is NumPy's call path (about 1.1 us at 32 elements, for a loop of
// a few nanoseconds), so the method hook, and the function hooks of np.round
// and np.around, run a loop of the binding's own, up to ROUND_F64_MAX_N
// elements. Above it NumPy's three passes in vector code are the better
// choice, and the call goes on to NumPy.
//
// NumPy's PyArray_Round computes, for a float array and decimals d >= 1,
// rint(x * 10^d) / 10^d in three ufunc calls: np.multiply, np.rint and
// np.true_divide, each correctly rounded in the current rounding mode. For
// d = 0 it is np.rint alone. The loop does the same operations in the same
// order on each element, so the result is the same bits, the sign of a zero
// included: -0.004 rounded to two decimals is -0.0 in both. 10^d is a double
// that holds the power exactly for every d taken, as NumPy's is.
//
// NumPy reports the flags those operations raise: overflow in the multiply,
// and invalid for a signalling NaN. (Underflow cannot arise: a subnormal
// number times a power of ten is exact while it stays subnormal, and a whole
// number divided by one is not tiny.) So the flags are cleared before the
// loop and read after it; if one NumPy reports is set, the result is dropped
// and NumPy computes the call and warns or raises as np.errstate says. The
// flags are left clear either way, as NumPy's own loops leave them.

// Most elements of a float64 array the binding's loop rounds: the largest
// power of two up to 4,096 at which the hooked call was at least 1.5x faster
// than NumPy's. Measured from Python on the Intel Core i5-14600KF,
// `np.round(x, 2)` was 7.4x faster at 16 elements and 2.2x at 4,096, and
// `np.round(x)` 3.5x and 1.7x; at 8,192 `np.round(x)` was 1.1x, NumPy's rint
// alone being as fast as the loop there.
#define ROUND_F64_MAX_N 4096

// The powers of ten np.round multiplies by, for 0 to 15 decimals: every one
// is a double that holds it exactly. Negative decimals divide first and are
// NumPy's.
static const double ROUND_POWERS[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
};
#define ROUND_MAX_DECIMALS ((long)(sizeof ROUND_POWERS / sizeof ROUND_POWERS[0]) - 1)

// Read np.round's `decimals`: absent is 0, and a Python int from 0 to
// ROUND_MAX_DECIMALS is itself. Returns false for anything else: a negative
// or larger int, a bool, a NumPy integer or a float, which NumPy converts,
// rejects or treats by rules of its own.
static bool round_decimals(PyObject *decimals, long *value) {
    if (decimals == NULL) {
        *value = 0;
        return true;
    }
    if (!PyLong_CheckExact(decimals)) return false;
    int overflow = 0;
    *value = PyLong_AsLongAndOverflow(decimals, &overflow);
    return overflow == 0 && *value >= 0 && *value <= ROUND_MAX_DECIMALS;
}

// out[i] = rint(x[i] * scale) / scale for the `n` elements at `x`, and
// rint(x[i]) when `scale` is 1, which is what multiplying and dividing by 1
// would give. Compiled for the kernels' level, where rint is one vroundpd
// per four elements; for any x86-64 it is a call into the C library per
// element, since SSE2 has no rounding instruction.
static HK_TARGET_KERNEL_LEVEL void round_loop(const double *restrict x, double *restrict out, size_t n,
                                              double scale) {
    if (scale == 1.0) {
        for (size_t i = 0; i < n; i++) out[i] = rint(x[i]);
    } else {
        for (size_t i = 0; i < n; i++) out[i] = rint(x[i] * scale) / scale;
    }
}

HK_NOINLINE PyObject *hk_try_round(PyObject *array, PyObject *decimals) {
    PyArrayObject *x = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(x);
    long places;
    // NumPy returns an empty array without computing anything.
    if (n < 1 || n > ROUND_F64_MAX_N || !round_decimals(decimals, &places)) return NULL;
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    hk_clear_float_flags();
    round_loop(PyArray_DATA(x), PyArray_DATA((PyArrayObject *)result), (size_t)n, ROUND_POWERS[places]);
    const bool raised = hk_float_flags_reported();
    hk_clear_float_flags();
    if (raised) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_ROUND_F64);
    return result;
}

// The work of method_round once the array is contiguous float64: the
// arguments are NumPy's, `decimals` and `out`, by position or by keyword, and
// `out` must be absent or None. Returns NULL with no exception set for a call
// that is NumPy's, including one NumPy would reject.
static HK_NOINLINE PyObject *try_round(PyObject *self, PyObject *args, PyObject *kwargs) {
    enum { DECIMALS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"decimals", "out"};
    const Py_ssize_t positional = PyTuple_GET_SIZE(args);
    if (positional > PARAMETERS) return NULL;
    PyObject *bound[PARAMETERS] = {NULL, NULL};
    for (Py_ssize_t i = 0; i < positional; i++) bound[i] = PyTuple_GET_ITEM(args, i);
    if (kwargs != NULL) {
        PyObject *key, *value;
        Py_ssize_t position = 0;
        while (PyDict_Next(kwargs, &position, &key, &value)) {
            Py_ssize_t i = 0;
            while (i < PARAMETERS && !(PyUnicode_CheckExact(key) && hk_text_is(key, names[i]))) i++;
            if (i == PARAMETERS || bound[i] != NULL) return NULL;
            bound[i] = value;
        }
    }
    if (!hk_is_absent(bound[OUT])) return NULL;
    return hk_try_round(self, bound[DECIMALS]);
}

// x.round(decimals) on a contiguous float64 array. np.round(x, decimals) and
// np.around call the method as x.round(decimals=d, out=o) when their own
// hooks pass a call on, and NumPy converts a list or a scalar to an array and
// calls its method; all of them that are not taken are counted here, once.
// Called with a tuple and a dictionary (METH_VARARGS | METH_KEYWORDS).
static PyObject *method_round(PyObject *self, PyObject *args, PyObject *kwargs) {
    if (hk_is_flat(self, NPY_DOUBLE)) {
        PyObject *result = try_round(self, args, kwargs);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_ROUND);
    return ((PyCFunctionWithKeywords)(void (*)(void))methods[M_ROUND].original)(self, args, kwargs);
}

// ------------------------------------------------------------------ argsort

// Below this many elements a stable argsort is left to NumPy. Its timsort
// sorts short arrays by insertion, and from Python (Apple M1 Pro) the kernel
// was level with it at 128 elements (0.95x to 1.30x by element type), 1.2x to
// 1.6x faster at 192 and 1.6x to 1.95x at 256; at 64 and 96 it took up to
// 1.6 times as long on keys that differ in all 64 bits.
#define ARGSORT_MIN_N 256

// The argsort kernels are defined below this many elements (their header
// comments say why).
#define ARGSORT_MAX_N ((npy_intp)1 << 26)

// Words of memory a call needs per element: a copy of the keys, and the
// kernel's output of three words, of which the first is the result.
#define ARGSORT_WORDS_PER_ELEMENT 4

// NumPy sorts stably with timsort, which takes time in proportion to the
// disorder of its input and is far quicker than the kernel on input that is
// nearly in order: a million sorted float64 take it 1 ms where the kernel
// takes 7. Merging is what it is slow at: two sorted halves of a million
// already take it as long as the kernel takes on random elements. So the
// call is looked at before the kernel gets it (argsort_order_*), in terms of
// falls: places where an element is less than the one before it.
//
// - No fall at all: the array is sorted and the answer is 0, 1, 2, ..., which
//   the hook writes itself.
// - At most ARGSORT_FALL_LIMIT falls: sorted but for a few elements. NumPy's.
// - One run, from one fall to the next, that leaves out fewer than one
//   element in ARGSORT_RUN_REMAINDER: a sorted array with a few new elements
//   appended, for example. NumPy's. Such a run starts in the first
//   n / ARGSORT_RUN_REMAINDER elements, so only the run that begins at the
//   last fall in that stretch is measured.
//
// The same three, mirrored, for an array in strictly descending order, which
// timsort reverses: no element that is not a fall gives n-1, ..., 1, 0.
//
// On shuffled data the count passes the limit within a few dozen elements
// and the run that is measured ends at once, so the look costs a few hundred
// comparisons. On ordered data it is one pass over the array a block at a
// time, which the compiler turns into vector compares.
#define ARGSORT_FALL_LIMIT 8
#define ARGSORT_RUN_REMAINDER 32

// Elements compared at a time when looking for the next fall.
#define ARGSORT_SCAN_BLOCK 64

// What a look at the order of an array found.
typedef enum {
    // Not in any order worth leaving to NumPy: the kernel's.
    ARGSORT_SHUFFLED,
    // No fall at all: the answer is known.
    ARGSORT_IN_ORDER,
    // Nearly in order: NumPy's.
    ARGSORT_NEARLY_IN_ORDER,
} argsort_order;

// NumPy's order of two elements: true when `a` sorts before `b`. For floats
// a NaN sorts after every number and is not before another NaN.
#define ARGSORT_LESS_INT(a, b) ((a) < (b))
#define ARGSORT_LESS_FLOAT(a, b) (((a) < (b)) | (((b) != (b)) & ((a) == (a))))

// Defines `next_function`, the first index from `k` up to `n` whose element
// breaks the run after the one before it (or `n`), and `order_function`, the
// look described above. `BREAKS(a, b)` says whether `b`, coming right after
// `a`, breaks a run: `b` sorts before `a` for ascending runs, and does not
// for strictly descending ones. `KEEPS(a, b)` is one comparison that is true
// only when `b` does not break the run; a block is passed over when it holds
// for every pair in it. For floats it is false whenever a NaN is involved,
// so a block with a NaN gets the exact test: on the Apple M1 Pro a million
// sorted float64 are passed over in 0.16 ms this way and in 0.25 ms by the
// exact test alone.
#define ARGSORT_ORDER_TEST(next_function, order_function, c_type, BREAKS, KEEPS)                \
    static npy_intp next_function(const c_type *x, npy_intp k, npy_intp n) {                    \
        while (k + ARGSORT_SCAN_BLOCK <= n) {                                                   \
            unsigned kept = 1;                                                                  \
            for (int j = 0; j < ARGSORT_SCAN_BLOCK; j++) kept &= (unsigned)KEEPS(x[k + j - 1], x[k + j]); \
            if (!kept) {                                                                        \
                unsigned any = 0;                                                               \
                for (int j = 0; j < ARGSORT_SCAN_BLOCK; j++) any |= (unsigned)BREAKS(x[k + j - 1], x[k + j]); \
                if (any) break;                                                                 \
            }                                                                                   \
            k += ARGSORT_SCAN_BLOCK;                                                            \
        }                                                                                       \
        while (k < n && !BREAKS(x[k - 1], x[k])) k++;                                           \
        return k;                                                                               \
    }                                                                                           \
    static argsort_order order_function(const c_type *x, npy_intp n) {                          \
        const npy_intp enough = n - n / ARGSORT_RUN_REMAINDER, head = n - enough;               \
        npy_intp falls = 0, start = 0, longest = 0;                                             \
        for (;;) {                                                                              \
            const npy_intp fall = next_function(x, start + 1, n);                               \
            if (fall - start > longest) longest = fall - start;                                 \
            if (fall == n) return falls == 0 ? ARGSORT_IN_ORDER : ARGSORT_NEARLY_IN_ORDER;      \
            start = fall;                                                                       \
            if (++falls > ARGSORT_FALL_LIMIT) break;                                            \
        }                                                                                       \
        if (longest >= enough) return ARGSORT_NEARLY_IN_ORDER;                                  \
        if (start > head) return ARGSORT_SHUFFLED;                                              \
        /* The last fall at or before `head`, found from `head` downwards. */                   \
        npy_intp last = head;                                                                   \
        while (last > start && !BREAKS(x[last - 1], x[last])) last--;                           \
        const npy_intp end = next_function(x, head + 1, n);                                     \
        return end - last >= enough ? ARGSORT_NEARLY_IN_ORDER : ARGSORT_SHUFFLED;               \
    }

#define ARGSORT_FALLS_INT(a, b) ARGSORT_LESS_INT(b, a)
#define ARGSORT_FALLS_FLOAT(a, b) ARGSORT_LESS_FLOAT(b, a)
#define ARGSORT_RISES_INT(a, b) (!ARGSORT_LESS_INT(b, a))
#define ARGSORT_RISES_FLOAT(a, b) (!ARGSORT_LESS_FLOAT(b, a))
#define ARGSORT_NOT_BELOW(a, b) ((b) >= (a))
#define ARGSORT_BELOW(a, b) ((b) < (a))

ARGSORT_ORDER_TEST(argsort_next_fall_f64, argsort_ascending_f64, double, ARGSORT_FALLS_FLOAT, ARGSORT_NOT_BELOW)
ARGSORT_ORDER_TEST(argsort_next_rise_f64, argsort_descending_f64, double, ARGSORT_RISES_FLOAT, ARGSORT_BELOW)
ARGSORT_ORDER_TEST(argsort_next_fall_i64, argsort_ascending_i64, int64_t, ARGSORT_FALLS_INT, ARGSORT_NOT_BELOW)
ARGSORT_ORDER_TEST(argsort_next_rise_i64, argsort_descending_i64, int64_t, ARGSORT_RISES_INT, ARGSORT_BELOW)
ARGSORT_ORDER_TEST(argsort_next_fall_u64, argsort_ascending_u64, uint64_t, ARGSORT_FALLS_INT, ARGSORT_NOT_BELOW)
ARGSORT_ORDER_TEST(argsort_next_rise_u64, argsort_descending_u64, uint64_t, ARGSORT_RISES_INT, ARGSORT_BELOW)

// The element types with an argsort kernel.
typedef enum { ARGSORT_F64, ARGSORT_I64, ARGSORT_U64, ARGSORT_NONE } argsort_keys;

// Which kernel sorts the elements of `self`: float64, or a 64-bit integer
// type. C's `long` and `long long` are two NumPy types and both are 64 bits
// on the platforms the kernels are built for. Everything else, datetimes
// with their NaT included, is ARGSORT_NONE.
static inline argsort_keys argsort_key_type(PyArrayObject *self) {
    switch (PyArray_TYPE(self)) {
    case NPY_DOUBLE: return ARGSORT_F64;
#if NPY_SIZEOF_LONG == 8
    case NPY_LONG: return ARGSORT_I64;
    case NPY_ULONG: return ARGSORT_U64;
#endif
#if NPY_SIZEOF_LONGLONG == 8
    case NPY_LONGLONG: return ARGSORT_I64;
    case NPY_ULONGLONG: return ARGSORT_U64;
#endif
    default: return ARGSORT_NONE;
    }
}

// True when the arguments of an argsort call ask for a stable sort along the
// only axis of a one-dimensional array and for nothing else:
//
// - `axis` absent, or the Python int -1 or 0;
// - `order` absent or None;
// - `kind` the string 'stable' or 'mergesort', NumPy's two names for its
//   stable sort, with `stable` absent or None; or `kind` absent or None and
//   `stable` True;
// - `descending`, which NumPy 2.4 added, absent.
//
// NumPy reads only the first letter of `kind`, so 'Stable' and 'm' are
// stable sorts too; they are rare spellings and are left to it, as is every
// combination it rejects (`kind` and `stable` together).
static HK_NOINLINE bool argsort_arguments_are_stable(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { AXIS, KIND, ORDER, STABLE, DESCENDING, PARAMETERS };
    static const char *const names[PARAMETERS] = {"axis", "kind", "order", "stable", "descending"};
    // `stable` and `descending` are keyword-only.
    enum { POSITIONAL = 3 };
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, POSITIONAL, bound)) return false;
    if (bound[DESCENDING] != NULL || !hk_is_absent(bound[ORDER])) return false;
    if (bound[AXIS] != NULL) {
        if (!PyLong_CheckExact(bound[AXIS])) return false;
        int overflow = 0;
        const long axis = PyLong_AsLongAndOverflow(bound[AXIS], &overflow);
        if (overflow != 0 || (axis != -1 && axis != 0)) return false;
    }
    PyObject *kind = bound[KIND], *stable = bound[STABLE];
    if (hk_is_absent(kind)) return stable == Py_True;
    if (!hk_is_absent(stable) || !PyUnicode_CheckExact(kind)) return false;
    return hk_text_is(kind, "stable") || hk_text_is(kind, "mergesort");
}

// What the look at an array's order means for the call.
typedef enum {
    // The kernel sorts it.
    ARGSORT_SORT,
    // In ascending order: the answer is 0, 1, 2, ...
    ARGSORT_ASCENDING,
    // In strictly descending order: the answer is n-1, ..., 1, 0.
    ARGSORT_DESCENDING,
    // Nearly in order one way or the other: NumPy's.
    ARGSORT_LEAVE,
} argsort_plan;

// Look at the order of the `n` contiguous elements at `data`; see
// ARGSORT_FALL_LIMIT. An array is tested for descending order only when it
// is not in ascending order, so one whose elements are all equal is ascending.
static argsort_plan argsort_look(argsort_keys keys, const void *data, npy_intp n) {
    const argsort_order ascending = keys == ARGSORT_F64   ? argsort_ascending_f64(data, n)
                                    : keys == ARGSORT_I64 ? argsort_ascending_i64(data, n)
                                                          : argsort_ascending_u64(data, n);
    if (ascending == ARGSORT_IN_ORDER) return ARGSORT_ASCENDING;
    if (ascending == ARGSORT_NEARLY_IN_ORDER) return ARGSORT_LEAVE;
    const argsort_order descending = keys == ARGSORT_F64   ? argsort_descending_f64(data, n)
                                     : keys == ARGSORT_I64 ? argsort_descending_i64(data, n)
                                                           : argsort_descending_u64(data, n);
    if (descending == ARGSORT_IN_ORDER) return ARGSORT_DESCENDING;
    return descending == ARGSORT_NEARLY_IN_ORDER ? ARGSORT_LEAVE : ARGSORT_SORT;
}

// The memory of one call: ARGSORT_WORDS_PER_ELEMENT words per element, or
// NULL when it cannot be had.
static uint64_t *argsort_memory(npy_intp n) {
    return malloc((size_t)n * ARGSORT_WORDS_PER_ELEMENT * sizeof(uint64_t));
}

// Copy the `n` elements of a one-dimensional array of 8-byte elements, which
// lie `stride` bytes apart from `data`, to `copy`. An element may sit at an
// address that is not a multiple of 8, so each is moved as bytes.
static void argsort_copy(uint64_t *copy, const char *data, npy_intp stride, npy_intp n) {
    if (stride == (npy_intp)sizeof(uint64_t)) {
        memcpy(copy, data, (size_t)n * sizeof(uint64_t));
        return;
    }
    for (npy_intp i = 0; i < n; i++) memcpy(&copy[i], data + i * stride, sizeof(uint64_t));
}

// x.argsort(kind='stable') for a one-dimensional float64, int64 or uint64
// array: the argsort kernels. np.argsort(x, kind=...) calls the method, as do
// np.unique(x, return_index=True) and scikit-learn's ranking metrics.
//
// The kernel reads its input three times and sizes its buckets from the first
// reading, so it must not see the array change in between: it would write
// outside its buffers. It is therefore given a copy, made here with the
// interpreter lock held, and sorts the copy with the lock released. The copy
// costs about a fiftieth of the sort.
//
// Because the kernel sorts a copy, the array need not be contiguous: a view
// with a step, or a reversed one, is copied element by element. NumPy copies
// such an array before it sorts it too. scikit-learn's metrics sort
// `np.flip(scores)`, a reversed view, so this is the call that matters there.
// A contiguous array is looked at for order where it lies, before any memory
// is asked for; any other is copied first and the copy looked at.
//
// The kernel's output is scratch: three words per element, of which the first
// n are the result. It is a buffer of the binding's own, and the result is
// copied out of it.
static HK_NOINLINE PyObject *try_argsort(PyObject *self, argsort_keys keys, PyObject *const *args, Py_ssize_t nargs,
                                         PyObject *kwnames) {
    PyArrayObject *x = (PyArrayObject *)self;
    npy_intp n = PyArray_SIZE(x);
    if (n < ARGSORT_MIN_N || n >= ARGSORT_MAX_N) return NULL;
    if (!argsort_arguments_are_stable(args, nargs, kwnames)) return NULL;
    const char *data = PyArray_BYTES(x);
    const npy_intp stride = PyArray_STRIDES(x)[0];
    const bool in_place = stride == (npy_intp)sizeof(uint64_t) && PyArray_ISALIGNED(x);
    const hk_kernel kernel = keys == ARGSORT_F64 ? HK_ARGSORT_F64 : keys == ARGSORT_I64 ? HK_ARGSORT_I64 : HK_ARGSORT_U64;

    uint64_t *memory = NULL;
    if (!in_place) {
        memory = argsort_memory(n);
        if (memory == NULL) return NULL;
        argsort_copy(memory, data, stride, n);
    }
    const argsort_plan plan = argsort_look(keys, in_place ? (const void *)data : (const void *)memory, n);
    PyObject *result = plan == ARGSORT_LEAVE ? NULL : PyArray_SimpleNew(1, &n, NPY_INTP);
    if (result == NULL) {
        // Left to NumPy, or no memory for the result, which is NumPy's to report.
        PyErr_Clear();
        free(memory);
        return NULL;
    }
    npy_intp *indices = PyArray_DATA((PyArrayObject *)result);
    if (plan == ARGSORT_ASCENDING) {
        for (npy_intp i = 0; i < n; i++) indices[i] = i;
    } else if (plan == ARGSORT_DESCENDING) {
        // No two elements are equal, so the reversed order is the only sorted one.
        for (npy_intp i = 0; i < n; i++) indices[i] = n - 1 - i;
    } else {
        if (memory == NULL) {
            memory = argsort_memory(n);
            if (memory == NULL) {
                Py_DECREF(result);
                return NULL;
            }
            argsort_copy(memory, data, stride, n);
        }
        const uint64_t *copy = memory;
        int64_t *out = (int64_t *)(memory + n);
        HK_BEGIN_NOGIL(n)
        if (keys == ARGSORT_F64) argsort_f64((const double *)copy, out, (size_t)n);
        else if (keys == ARGSORT_I64) argsort_i64((const int64_t *)copy, out, (size_t)n);
        else argsort_u64(copy, out, (size_t)n);
        memcpy(indices, out, (size_t)n * sizeof(int64_t));
        HK_END_NOGIL
    }
    free(memory);
    hk_take(kernel);
    return result;
}

// The array's type is tested before the arguments are looked at. A call on
// an element type without a kernel goes on uncounted; one on a type that has
// a kernel and is not taken (the default kind, an axis of a matrix, a short
// or nearly sorted array) is counted.
static PyObject *method_argsort(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    if (hk_is_plain_array(self)) {
        PyArrayObject *x = (PyArrayObject *)self;
        const argsort_keys keys = argsort_key_type(x);
        if (keys != ARGSORT_NONE) {
            if (PyArray_NDIM(x) == 1 && PyArray_ISNOTSWAPPED(x)) {
                PyObject *result = try_argsort(self, keys, args, nargs, kwnames);
                if (result != NULL) return result;
            }
            hk_pass(HK_OP_ARGSORT);
        }
    }
    return ((hk_fast_keywords)(void (*)(void))methods[M_ARGSORT].original)(self, args, nargs, kwnames);
}

// ----------------------------------------------------------------- the hook

// Shorthand for one entry of `methods`: the method `name` of type `owner`,
// called in convention `flags`, served by `wrapper` after `prepare`.
#define HK_METHOD(name, owner, prepare, flags, wrapper) \
    {name, owner, prepare, flags, (PyCFunction)(void (*)(void))(wrapper), NULL, NULL}

static hk_method methods[M_COUNT] = {
    [M_ASTYPE] = HK_METHOD("astype", OWNER_NDARRAY, astype_prepare, METH_FASTCALL | METH_KEYWORDS, method_astype),
    [M_SUM] = HK_METHOD("sum", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS, method_sum),
    [M_NONZERO] = HK_METHOD("nonzero", OWNER_NDARRAY, NULL, METH_VARARGS, method_nonzero),
    [M_CUMSUM] = HK_METHOD("cumsum", OWNER_NDARRAY, accumulate_prepare, METH_VARARGS | METH_KEYWORDS, method_cumsum),
    [M_SEARCHSORTED] = HK_METHOD("searchsorted", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS,
                                 method_searchsorted),
    [M_ARGMAX] = HK_METHOD("argmax", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS, method_argmax),
    [M_ARGMIN] = HK_METHOD("argmin", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS, method_argmin),
    [M_CLIP] = HK_METHOD("clip", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS, method_clip),
    [M_ACCUMULATE] = HK_METHOD("accumulate", OWNER_UFUNC, accumulate_prepare, METH_FASTCALL | METH_KEYWORDS,
                               method_accumulate),
    [M_REDUCEAT] = HK_METHOD("reduceat", OWNER_UFUNC, accumulate_prepare, METH_FASTCALL | METH_KEYWORDS,
                             method_reduceat),
    [M_ANY] = HK_METHOD("any", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS, method_any),
    [M_ROUND] = HK_METHOD("round", OWNER_NDARRAY, NULL, METH_VARARGS | METH_KEYWORDS, method_round),
    [M_ARGSORT] = HK_METHOD("argsort", OWNER_NDARRAY, NULL, METH_FASTCALL | METH_KEYWORDS, method_argsort),
};

// The type object of `owner`, and its name for messages.
static PyTypeObject *owner_type(hk_owner owner) {
    return owner == OWNER_UFUNC ? &PyUFunc_Type : &PyArray_Type;
}

static const char *owner_name(hk_owner owner) {
    return owner == OWNER_UFUNC ? "ufunc" : "ndarray";
}

// Point the method at the binding. See hk_hook.apply.
static int method_apply(int index) {
    hk_method *method = &methods[index];
    if (method->prepare != NULL && method->prepare() != 0) return -1;
    const char *owner = owner_name(method->owner);
    PyObject *descriptor = PyObject_GetAttrString((PyObject *)owner_type(method->owner), method->name);
    if (descriptor == NULL) return -1;
    if (!Py_IS_TYPE(descriptor, &PyMethodDescr_Type)) {
        PyErr_Format(PyExc_TypeError, "%s.%s is not a built-in method", owner, method->name);
        Py_DECREF(descriptor);
        return -1;
    }
    // The table entry belongs to NumPy and outlives the descriptor object.
    PyMethodDef *definition = ((PyMethodDescrObject *)descriptor)->d_method;
    Py_DECREF(descriptor);
    if (definition->ml_flags != method->flags) {
        PyErr_Format(PyExc_TypeError, "%s.%s is called with convention %d, and the hook was written for %d",
                     owner, method->name, definition->ml_flags, method->flags);
        return -1;
    }
    method->definition = definition;
    method->original = definition->ml_meth;
    definition->ml_meth = method->wrapper;
    return 0;
}

// Put NumPy's function back. See hk_hook.remove.
static int method_remove(int index) {
    PyMethodDef *definition = methods[index].definition;
    if (definition == NULL || definition->ml_meth != methods[index].wrapper) return 0;
    definition->ml_meth = methods[index].original;
    methods[index].definition = NULL;
    return 1;
}

hk_hook hk_method_hooks[] = {
    {"method.astype", method_apply, method_remove, M_ASTYPE, false},
    {"method.sum", method_apply, method_remove, M_SUM, false},
    {"method.nonzero", method_apply, method_remove, M_NONZERO, false},
    {"method.cumsum", method_apply, method_remove, M_CUMSUM, false},
    {"method.searchsorted", method_apply, method_remove, M_SEARCHSORTED, false},
    {"method.argmax", method_apply, method_remove, M_ARGMAX, false},
    {"method.argmin", method_apply, method_remove, M_ARGMIN, false},
    {"method.clip", method_apply, method_remove, M_CLIP, false},
    {"ufunc.accumulate", method_apply, method_remove, M_ACCUMULATE, false},
    {"ufunc.reduceat", method_apply, method_remove, M_REDUCEAT, false},
    {"method.any", method_apply, method_remove, M_ANY, false},
    {"method.round", method_apply, method_remove, M_ROUND, false},
    {"method.argsort", method_apply, method_remove, M_ARGSORT, false},
    {NULL, NULL, NULL, 0, false},
};
