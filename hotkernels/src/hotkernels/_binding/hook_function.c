// Function hook: the call pointer inside a NumPy function object.
//
// `np.argmax`, `np.histogram` and the rest are objects of one NumPy type.
// CPython calls such an object through a function pointer stored inside it
// (the vectorcall protocol), at an offset the object's type publishes as
// `tp_vectorcall_offset`. Changing that one pointer leaves `np.argmax` the
// same object: names imported before install(), NumPy's own internal callers
// and code that uses the function as a dictionary key all keep working, and
// all reach the hook.
//
// A hooked call arrives with the caller's arguments exactly as written. The
// wrapper takes a kernel only when they already are what the kernel needs.
// Otherwise it calls the original pointer with the same arguments, so NumPy
// does its own conversions and raises its own errors.
#include "binding.h"

// The arguments of a vectorcall: the callable, the positional arguments
// followed by the keyword values, the positional count with a flag bit, and
// the keyword names.
#define HK_VECTORCALL_PARAMETERS \
    PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames

// The functions that are hooked, as indices into `functions`.
enum {
    F_ARGMAX, F_ARGMIN, F_COUNT_NONZERO, F_SUM, F_HISTOGRAM, F_BINCOUNT, F_ISIN, F_INTERP,
    F_COMPRESS, F_EXTRACT, F_PUTMASK, F_COPYTO, F_WHERE, F_CLIP, F_ANY, F_ROUND, F_AROUND, F_MEDIAN,
    F_UNIQUE,
    F_COUNT
};

// One hooked function.
typedef struct {
    // Attribute of the `numpy` module that holds the function object.
    const char *attribute;
    // The binding's call function.
    vectorcallfunc wrapper;
    // Reads what the wrapper needs to know about this NumPy's function when
    // the hook is applied, or NULL. Takes the function object; raises nothing.
    void (*prepare)(PyObject *object);
    // The function object, kept alive while the hook is applied.
    PyObject *object;
    // NumPy's own call function, kept while the hook is applied.
    vectorcallfunc original;
} hk_function;

static hk_function functions[F_COUNT];

// `np.histogram`'s default number of bins.
#define HISTOGRAM_DEFAULT_BINS 10
// Number of counters `bincount_u8` writes: one per byte value.
#define BYTE_VALUES 256
// Counters `bincount_i16` and `bincount_i64` are always given, whatever the
// element count: 512 KB. At that size a call on 1,000 elements takes about
// as long with the kernel as NumPy's does (6.5 against 6.9 us on the Intel
// Core i5-14600KF), both being mostly the counters. Above it a call is taken
// only while it has no more counters than elements; see bincount_codes.
#define BINCOUNT_FREE_BINS 65536
// Bytes of a boolean array that `np.sum` checks and then sums at a time. Each
// block is read twice, so it is a third of the smallest first-level data
// cache the kernels run on (48 KiB on the x86-64 performance cores).
#define SUM_BOOL_BLOCK 16384
// Number of test values `isin_i64` reads. Fewer are padded by repeating one.
#define ISIN_TEST_VALUES 8
// Most sample points per query for which `np.interp` takes the kernel. The
// sample points are read once for NaN before the kernel sees them, at about
// 0.13 ns each, and NumPy spends at least 30 ns on a query. At 64 sample
// points per query the pass costs under a third of what NumPy would have
// spent; with fewer queries than that it would be most of the call.
#define INTERP_MAX_POINTS_PER_QUERY 64
// Elements of the buffer np.where spreads a scalar over, and so of each call
// of the kernel when one side is a scalar: the largest buffer kept on the
// stack.
#define WHERE_BLOCK ((npy_intp)(HK_STACK_SCRATCH_BYTES / sizeof(double)))

// numpy.linspace and the float64 type descriptor, fetched on first use.
static PyObject *numpy_linspace;
static PyObject *linspace_keywords;

// ------------------------------------------------------------ small helpers

PyObject *hk_scalar(int type_num, void *value) {
    PyArray_Descr *descr = PyArray_DescrFromType(type_num);
    PyObject *scalar = PyArray_Scalar(value, descr, NULL);
    Py_DECREF(descr);
    if (scalar == NULL) PyErr_Clear();
    return scalar;
}

bool hk_bind_arguments(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                       const char *const *names, Py_ssize_t count,
                       Py_ssize_t max_positional, PyObject **bound) {
    if (nargs > max_positional) return false;
    for (Py_ssize_t i = 0; i < count; i++) bound[i] = i < nargs ? args[i] : NULL;
    const Py_ssize_t keywords = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t k = 0; k < keywords; k++) {
        PyObject *name = PyTuple_GET_ITEM(kwnames, k);
        Py_ssize_t i = 0;
        while (i < count && !(names[i] && hk_text_is(name, names[i]))) i++;
        if (i == count || bound[i] != NULL) return false;
        bound[i] = args[nargs + k];
    }
    return true;
}

// ------------------------------------------------------ argmax and argmin

PyObject *hk_try_argext(PyObject *array, bool want_max) {
    hk_kernel kernel;
    npy_intp index;
    if (hk_is_flat(array, NPY_FLOAT)) kernel = want_max ? HK_ARGMAX_F32 : HK_ARGMIN_F32;
    else if (hk_is_flat(array, NPY_DOUBLE)) kernel = want_max ? HK_ARGMAX_F64 : HK_ARGMIN_F64;
    else return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(a);
    // NumPy raises on an empty array; the kernels are never called with n = 0.
    if (n < 1) return NULL;
    // The interpreter lock stays held: the kernels read the array twice and
    // are not safe against a writer (issue 9).
    switch (kernel) {
        case HK_ARGMAX_F32: index = (npy_intp)argmax_f32(PyArray_DATA(a), (size_t)n); break;
        case HK_ARGMIN_F32: index = (npy_intp)argmin_f32(PyArray_DATA(a), (size_t)n); break;
        case HK_ARGMAX_F64: index = (npy_intp)argmax_f64(PyArray_DATA(a), (size_t)n); break;
        default: index = (npy_intp)argmin_f64(PyArray_DATA(a), (size_t)n); break;
    }
    PyObject *result = hk_scalar(NPY_INTP, &index);
    if (result != NULL) hk_take(kernel);
    return result;
}

// ------------------------------------------ argmax and argmin along rows

// Narrowest row, in elements, given to the operation's kernel once per row.
// Narrower rows go to the binding's own loop, one row per vector lane. The
// kernels take about 1.6 ns a row from four float64 or eight float32
// elements, and read fewer than that one at a time, at 5 to 12 ns a row,
// where NumPy takes 7 to 13. The loop takes 0.4 ns a row on rows of two and
// grows by about 0.25 ns (float64) or 0.18 ns (float32) per column: it meets
// the kernel at six float64 and nine float32 columns. Apple M1 Pro, from
// Python, docs/argmax-rows-design.md. The same values are used on x86-64,
// where they have not been timed.
#define ARGEXT_ROWS_KERNEL_COLS_F64 6
#define ARGEXT_ROWS_KERNEL_COLS_F32 9

// Lanes of the binding's loop: 32 bytes of rows, one row per lane.
#define ARGEXT_ROWS_LANES_F64 4
#define ARGEXT_ROWS_LANES_F32 8

// 32-byte vectors of the compiler's own vector types. clang and gcc compile
// them to two NEON registers on ARM64 and to one AVX2 register inside a
// function compiled for the kernels' level on x86-64. A comparison gives a
// vector of signed integers of the element's width, all ones where it holds;
// it is cast to the integer type below because the compilers name that
// type differently (long or long long).
typedef double hk_v4d __attribute__((vector_size(32)));
typedef int64_t hk_v4l __attribute__((vector_size(32)));
typedef float hk_v8f __attribute__((vector_size(32)));
typedef int32_t hk_v8i __attribute__((vector_size(32)));

// Element `j` of each of the rows that start `cols` elements apart at `x`.
#define ARGEXT_ROWS_GATHER4(x, cols, j) {(x)[j], (x)[(cols) + (j)], (x)[2 * (cols) + (j)], (x)[3 * (cols) + (j)]}
#define ARGEXT_ROWS_GATHER8(x, cols, j)                                                            \
    {(x)[j], (x)[(cols) + (j)], (x)[2 * (cols) + (j)], (x)[3 * (cols) + (j)], (x)[4 * (cols) + (j)], \
     (x)[5 * (cols) + (j)], (x)[6 * (cols) + (j)], (x)[7 * (cols) + (j)]}

// One loop of the binding: out[r] = NumPy's argmax (or argmin) of row r of
// the `rows` rows of `cols` elements at `m`, cols >= 1.
//
// Each lane keeps the best element so far and its column, and takes column j
// when the best is a number and `!(x[j] NOT_BETTER best)`: for argmax, with
// NOT_BETTER `<=`, when x[j] is larger or is a NaN. A NaN is then never
// replaced, since every comparison with it is false, so the first NaN wins;
// an equal element never replaces an earlier one, so the first of a tie wins,
// and -0.0 and 0.0 are equal. These are NumPy's rules and naive.c's. Rows
// after the last whole group of lanes take the same rule one at a time.
//
// The loop reads each element once and touches no Python object, so it runs
// with the interpreter lock released for a large matrix.
#define ARGEXT_ROWS_LOOP(name, c_type, v_type, i_type, lane_int, LANES, GATHER, NOT_BETTER)            \
    static HK_TARGET_KERNEL_LEVEL void name(const c_type *m, size_t rows, size_t cols, npy_intp *out) { \
        size_t r = 0;                                                                                 \
        for (; r + LANES <= rows; r += LANES) {                                                       \
            const c_type *x = m + r * cols;                                                           \
            v_type best = GATHER(x, cols, 0);                                                         \
            i_type index = {0};                                                                       \
            for (size_t j = 1; j < cols; j++) {                                                       \
                const v_type v = GATHER(x, cols, j);                                                  \
                const i_type take = (i_type)(best == best) & ~(i_type)(v NOT_BETTER best);            \
                best = (v_type)(((i_type)v & take) | ((i_type)best & ~take));                         \
                index = (index & ~take) | (((i_type){0} + (lane_int)j) & take);                       \
            }                                                                                         \
            for (size_t k = 0; k < LANES; k++) out[r + k] = (npy_intp)index[k];                       \
        }                                                                                             \
        for (; r < rows; r++) {                                                                       \
            const c_type *x = m + r * cols;                                                           \
            c_type best = x[0];                                                                       \
            npy_intp index = 0;                                                                       \
            for (size_t j = 1; j < cols; j++) {                                                       \
                const bool take = best == best && !(x[j] NOT_BETTER best);                            \
                best = take ? x[j] : best;                                                            \
                index = take ? (npy_intp)j : index;                                                   \
            }                                                                                         \
            out[r] = index;                                                                           \
        }                                                                                             \
    }

ARGEXT_ROWS_LOOP(argmax_rows_f64, double, hk_v4d, hk_v4l, int64_t, ARGEXT_ROWS_LANES_F64, ARGEXT_ROWS_GATHER4, <=)
ARGEXT_ROWS_LOOP(argmin_rows_f64, double, hk_v4d, hk_v4l, int64_t, ARGEXT_ROWS_LANES_F64, ARGEXT_ROWS_GATHER4, >=)
ARGEXT_ROWS_LOOP(argmax_rows_f32, float, hk_v8f, hk_v8i, int32_t, ARGEXT_ROWS_LANES_F32, ARGEXT_ROWS_GATHER8, <=)
ARGEXT_ROWS_LOOP(argmin_rows_f32, float, hk_v8f, hk_v8i, int32_t, ARGEXT_ROWS_LANES_F32, ARGEXT_ROWS_GATHER8, >=)

// The kernel once per row: out[r] = kernel(row r). With the interpreter lock
// held, as for the kernels everywhere in the function and method hooks: they
// read a row twice (issue 9).
static HK_NOINLINE void argext_rows_kernel(hk_kernel kernel, const void *m, size_t rows, size_t cols,
                                           npy_intp *out) {
    switch (kernel) {
        case HK_ARGMAX_F32:
            for (size_t r = 0; r < rows; r++) out[r] = (npy_intp)argmax_f32((const float *)m + r * cols, cols);
            break;
        case HK_ARGMIN_F32:
            for (size_t r = 0; r < rows; r++) out[r] = (npy_intp)argmin_f32((const float *)m + r * cols, cols);
            break;
        case HK_ARGMAX_F64:
            for (size_t r = 0; r < rows; r++) out[r] = (npy_intp)argmax_f64((const double *)m + r * cols, cols);
            break;
        default:
            for (size_t r = 0; r < rows; r++) out[r] = (npy_intp)argmin_f64((const double *)m + r * cols, cols);
            break;
    }
}

// The binding's own loop on every row, with the lock released for a large matrix.
static HK_NOINLINE void argext_rows_loop(hk_kernel loop, const void *m, size_t rows, size_t cols, npy_intp *out) {
    HK_BEGIN_NOGIL(rows * cols)
    switch (loop) {
        case HK_ARGMAX_ROWS_F32: argmax_rows_f32(m, rows, cols, out); break;
        case HK_ARGMIN_ROWS_F32: argmin_rows_f32(m, rows, cols, out); break;
        case HK_ARGMAX_ROWS_F64: argmax_rows_f64(m, rows, cols, out); break;
        default: argmin_rows_f64(m, rows, cols, out); break;
    }
    HK_END_NOGIL
}

PyObject *hk_try_argext_rows(PyObject *array, bool want_max) {
    if (!hk_is_plain_array(array)) return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const int type_num = PyArray_TYPE(a);
    if (PyArray_NDIM(a) != 2 || !(type_num == NPY_FLOAT || type_num == NPY_DOUBLE) || !PyArray_ISNOTSWAPPED(a) ||
        !PyArray_IS_C_CONTIGUOUS(a)) {
        return NULL;
    }
    npy_intp rows = PyArray_DIM(a, 0);
    const npy_intp cols = PyArray_DIM(a, 1);
    // NumPy raises on rows of no element.
    if (cols < 1) return NULL;
    const bool f32 = type_num == NPY_FLOAT;
    hk_kernel kernel = f32 ? (want_max ? HK_ARGMAX_F32 : HK_ARGMIN_F32) : (want_max ? HK_ARGMAX_F64 : HK_ARGMIN_F64);
    const hk_kernel loop = f32 ? (want_max ? HK_ARGMAX_ROWS_F32 : HK_ARGMIN_ROWS_F32)
                               : (want_max ? HK_ARGMAX_ROWS_F64 : HK_ARGMIN_ROWS_F64);
    const npy_intp kernel_cols = f32 ? ARGEXT_ROWS_KERNEL_COLS_F32 : ARGEXT_ROWS_KERNEL_COLS_F64;
    PyObject *result = PyArray_SimpleNew(1, &rows, NPY_INTP);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    npy_intp *out = PyArray_DATA((PyArrayObject *)result);
    // A matrix of no rows is the loop's, which writes nothing: the kernels
    // are never called for nothing.
    if (cols >= kernel_cols && rows > 0) {
        argext_rows_kernel(kernel, PyArray_DATA(a), (size_t)rows, (size_t)cols, out);
    } else {
        argext_rows_loop(loop, PyArray_DATA(a), (size_t)rows, (size_t)cols, out);
        kernel = loop;
    }
    hk_take(kernel);
    return result;
}

// True when `array` is an ndarray, or an instance of a subclass, of float32
// or float64. Every Python-level argmax of such an array that the function
// hook passes on reaches the method hook, which counts it there.
static bool reaches_method_count(PyObject *array) {
    if (!PyArray_Check(array)) return false;
    const int type_num = PyArray_TYPE((PyArrayObject *)array);
    return type_num == NPY_FLOAT || type_num == NPY_DOUBLE;
}

// The rows form of np.argmax and np.argmin: the array, the axis as the
// Python int 1 or -1, and `out` absent or None, by position or by keyword.
// The parameters are a, axis, out and, by keyword only, keepdims, which is
// NumPy's.
static HK_NOINLINE PyObject *try_argext_rows(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                                             bool want_max) {
    enum { A, AXIS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"a", "axis", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[A] == NULL || bound[AXIS] == NULL || !hk_is_last_axis(bound[AXIS]) || !hk_is_absent(bound[OUT])) {
        return NULL;
    }
    return hk_try_argext_rows(bound[A], want_max);
}

// np.argmax(a) and np.argmin(a) with nothing but the array: the kernel is
// called directly, which skips NumPy's Python wrapper. That wrapper is most
// of the cost of a call on a small array. So is np.argmax(m, axis=1) on a
// matrix (hk_try_argext_rows). A call passed on is counted here only when it
// is not of a float32 or float64 array, whose call the method hook counts.
static PyObject *call_argext(HK_VECTORCALL_PARAMETERS, int function, bool want_max, hk_operation operation) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (nargs == 1 && kwnames == NULL) {
        PyObject *result = hk_try_argext(args[0], want_max);
        if (result != NULL) return result;
    } else if (nargs >= 1 && hk_is_plain_array(args[0]) && PyArray_NDIM((PyArrayObject *)args[0]) == 2) {
        PyObject *result = try_argext_rows(args, nargs, kwnames, want_max);
        if (result != NULL) return result;
    }
    if (!(nargs >= 1 && reaches_method_count(args[0]))) hk_pass(operation);
    return functions[function].original(callable, args, nargsf, kwnames);
}

static PyObject *call_argmax(HK_VECTORCALL_PARAMETERS) {
    return call_argext(callable, args, nargsf, kwnames, F_ARGMAX, true, HK_OP_ARGMAX);
}

static PyObject *call_argmin(HK_VECTORCALL_PARAMETERS) {
    return call_argext(callable, args, nargsf, kwnames, F_ARGMIN, false, HK_OP_ARGMIN);
}

// ----------------------------------------------------------- count_nonzero

// np.count_nonzero(a) for a contiguous int64 or float64 array. NumPy returns
// the count as an np.intp scalar when no axis is given.
static HK_NOINLINE PyObject *try_count_nonzero(PyObject *array) {
    hk_kernel kernel;
    if (hk_is_flat(array, NPY_INT64)) kernel = HK_COUNT_NONZERO_I64;
    else if (hk_is_flat(array, NPY_DOUBLE)) kernel = HK_COUNT_NONZERO_F64;
    else return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(a);
    npy_intp count;
    {
        HK_BEGIN_NOGIL(n)
        count = (npy_intp)(kernel == HK_COUNT_NONZERO_I64 ? count_nonzero_i64(PyArray_DATA(a), (size_t)n)
                                                          : count_nonzero_f64(PyArray_DATA(a), (size_t)n));
        HK_END_NOGIL
    }
    PyObject *result = hk_scalar(NPY_INTP, &count);
    if (result != NULL) hk_take(kernel);
    return result;
}

static PyObject *call_count_nonzero(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 1 && kwnames == NULL) {
        PyObject *result = try_count_nonzero(args[0]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_COUNT_NONZERO);
    return functions[F_COUNT_NONZERO].original(callable, args, nargsf, kwnames);
}

// --------------------------------------------------------------------- sum

// The number of True elements of a contiguous boolean array, as the np.intp
// NumPy's sum returns for one: the sum_u8 kernel on the mask's bytes. A byte
// other than 0 or 1 counts once in NumPy and by its value in the kernel, so
// the bytes are checked first and such a mask is NumPy's. Checking and
// summing go block by block, so that the kernel reads each block from the
// first-level cache and not from wherever the whole array lies.
static HK_NOINLINE PyObject *try_sum_bool(PyArrayObject *mask) {
    const uint8_t *bytes = PyArray_DATA(mask);
    const npy_intp n = PyArray_SIZE(mask);
    npy_intp total = 0;
    bool clean = true;
    {
        HK_BEGIN_NOGIL(n)
        for (npy_intp done = 0; done < n && clean; done += SUM_BOOL_BLOCK) {
            const size_t count = (size_t)(n - done < SUM_BOOL_BLOCK ? n - done : SUM_BOOL_BLOCK);
            clean = hk_mask_is_clean(bytes + done, count);
            if (clean) total += (npy_intp)sum_u8(bytes + done, count);
        }
        HK_END_NOGIL
    }
    if (!clean) return NULL;
    PyObject *result = hk_scalar(NPY_INTP, &total);
    if (result != NULL) hk_take(HK_SUM_U8);
    return result;
}

// The sum of a contiguous, aligned, one-dimensional float32 array, as the
// np.float32 NumPy returns: the sum_f32 kernel, which makes NumPy's additions in
// NumPy's order (its pairwise summation) and so gives its bits. An empty
// array sums to +0.0, as in NumPy. A sum that is not finite is NumPy's to
// give, with its warning and the bits of its NaN, and the call is passed on.
// The flags are left as NumPy leaves them (hk_clear_reported_float_flags).
static HK_NOINLINE PyObject *try_sum_f32(PyArrayObject *a) {
    const npy_intp n = PyArray_SIZE(a);
    float total;
    {
        HK_BEGIN_NOGIL(n)
        total = sum_f32(PyArray_DATA(a), (size_t)n);
        HK_END_NOGIL
    }
    hk_clear_reported_float_flags();
    if (!isfinite(total)) return NULL;
    PyObject *result = hk_scalar(NPY_FLOAT, &total);
    if (result != NULL) hk_take(HK_SUM_F32);
    return result;
}

// The sum of a contiguous uint8 array, as the np.uint64 NumPy returns, or of
// a contiguous boolean array, as the np.intp it returns for that, and of a
// contiguous, aligned, one-dimensional float32 array.
PyObject *hk_try_sum(PyObject *array) {
    if (hk_is_flat(array, NPY_BOOL)) return try_sum_bool((PyArrayObject *)array);
    // An array whose elements are not aligned to their size is copied by
    // NumPy in blocks of its buffer size, 8,192 elements, whose sums it adds
    // one after another: another order. It stays with NumPy.
    if (hk_is_flat_1d(array, NPY_FLOAT) && PyArray_ISALIGNED((PyArrayObject *)array)) {
        return try_sum_f32((PyArrayObject *)array);
    }
    if (!hk_is_flat(array, NPY_UINT8)) return NULL;
    PyArrayObject *a = (PyArrayObject *)array;
    const npy_intp n = PyArray_SIZE(a);
    uint64_t total;
    {
        HK_BEGIN_NOGIL(n)
        total = sum_u8(PyArray_DATA(a), (size_t)n);
        HK_END_NOGIL
    }
    PyObject *result = hk_scalar(NPY_UINT64, &total);
    if (result != NULL) hk_take(HK_SUM_U8);
    return result;
}

static PyObject *call_sum(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 1 && kwnames == NULL) {
        PyObject *result = hk_try_sum(args[0]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_SUM);
    return functions[F_SUM].original(callable, args, nargsf, kwnames);
}

// --------------------------------------------------------------- histogram

// Fetch numpy.linspace and build the keywords it is called with. Returns
// false with no exception set if either fails.
static bool prepare_linspace(void) {
    if (numpy_linspace != NULL) return true;
    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *linspace = numpy ? PyObject_GetAttrString(numpy, "linspace") : NULL;
    PyObject *keywords = PyDict_New();
    PyObject *float64 = (PyObject *)PyArray_DescrFromType(NPY_DOUBLE);
    Py_XDECREF(numpy);
    if (linspace == NULL || keywords == NULL || PyDict_SetItemString(keywords, "dtype", float64) != 0) {
        PyErr_Clear();
        Py_XDECREF(linspace);
        Py_XDECREF(keywords);
        Py_DECREF(float64);
        return false;
    }
    Py_DECREF(float64);
    numpy_linspace = linspace;
    linspace_keywords = keywords;
    return true;
}

// Work out the two outer bin edges of a histogram call, as new references to
// the objects NumPy itself would use: the caller's `range` values, or the
// array's minimum and maximum when no range was given. `lo` and `hi` receive
// them as doubles. Returns false, with no exception set, when the range is
// one NumPy treats specially: not a tuple of two plain numbers, not finite,
// empty, reversed, or wider than a double can hold. NumPy then widens it,
// warns or raises, and the call is its own.
static bool histogram_range(PyObject *array, PyObject *range, PyObject **first, PyObject **last,
                            double *lo, double *hi) {
    *first = *last = NULL;
    if (hk_is_absent(range)) {
        if (PyArray_SIZE((PyArrayObject *)array) == 0) return false;
        *first = PyObject_CallMethod(array, "min", NULL);
        *last = PyObject_CallMethod(array, "max", NULL);
        if (*first == NULL || *last == NULL) goto fail;
        *lo = PyFloat_AsDouble(*first);
        *hi = PyFloat_AsDouble(*last);
        if (PyErr_Occurred()) goto fail;
    } else {
        if (!PyTuple_CheckExact(range) || PyTuple_GET_SIZE(range) != 2) return false;
        if (!hk_as_exact_double(PyTuple_GET_ITEM(range, 0), lo) || !hk_as_exact_double(PyTuple_GET_ITEM(range, 1), hi)) return false;
        *first = Py_NewRef(PyTuple_GET_ITEM(range, 0));
        *last = Py_NewRef(PyTuple_GET_ITEM(range, 1));
    }
    // A width that overflows makes NumPy's own arithmetic warn. Such a call
    // is left to NumPy before anything here has computed with the range, so
    // that each warning is issued once.
    if (isfinite(*lo) && isfinite(*hi) && *lo < *hi && isfinite(*hi - *lo)) return true;
fail:
    PyErr_Clear();
    Py_CLEAR(*first);
    Py_CLEAR(*last);
    return false;
}

// Build the bin edges exactly as np.histogram does, with the same call to
// np.linspace, so they are identical by construction. Returns NULL with no
// exception set if that fails or if the edges are not strictly increasing:
// NumPy raises "Too many bins for data range" then, and the kernel would
// return counts (issue 9).
static PyObject *histogram_edges(PyObject *first, PyObject *last, Py_ssize_t bins) {
    if (!prepare_linspace()) return NULL;
    PyObject *positional = Py_BuildValue("(OOn)", first, last, bins + 1);
    if (positional == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyObject *edges = PyObject_Call(numpy_linspace, positional, linspace_keywords);
    Py_DECREF(positional);
    if (edges == NULL) {
        PyErr_Clear();
        return NULL;
    }
    if (!hk_is_flat_1d(edges, NPY_DOUBLE) || PyArray_SIZE((PyArrayObject *)edges) != bins + 1) {
        Py_DECREF(edges);
        return NULL;
    }
    const double *e = PyArray_DATA((PyArrayObject *)edges);
    for (Py_ssize_t i = 0; i < bins; i++) {
        if (!(e[i] < e[i + 1])) {
            Py_DECREF(edges);
            return NULL;
        }
    }
    return edges;
}

// np.histogram(a, bins, range) for a contiguous float64 array, an integer
// number of bins, and no density or weights. Returns the pair NumPy returns:
// the counts as int64 and the bin edges as float64.
static HK_NOINLINE PyObject *try_histogram(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, BINS, RANGE, DENSITY, WEIGHTS, PARAMETERS };
    static const char *const names[PARAMETERS] = {"a", "bins", "range", "density", "weights"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[A] == NULL || !hk_is_flat(bound[A], NPY_DOUBLE)) return NULL;
    if (!hk_is_absent(bound[DENSITY]) || !hk_is_absent(bound[WEIGHTS])) return NULL;

    Py_ssize_t bins = HISTOGRAM_DEFAULT_BINS;
    if (bound[BINS] != NULL) {
        if (!PyLong_CheckExact(bound[BINS])) return NULL;
        bins = PyLong_AsSsize_t(bound[BINS]);
        if (bins == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NULL;
        }
    }
    if (bins < 1) return NULL;

    PyObject *first, *last;
    double lo, hi;
    if (!histogram_range(bound[A], bound[RANGE], &first, &last, &lo, &hi)) return NULL;
    PyObject *edges = histogram_edges(first, last, bins);
    Py_DECREF(first);
    Py_DECREF(last);
    if (edges == NULL) return NULL;

    npy_intp shape = bins;
    PyObject *counts = PyArray_SimpleNew(1, &shape, NPY_INTP);
    PyObject *result = counts ? PyTuple_Pack(2, counts, edges) : NULL;
    Py_DECREF(edges);
    Py_XDECREF(counts);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)bound[A];
    const npy_intp n = PyArray_SIZE(a);
    {
        HK_BEGIN_NOGIL(n)
        histogram_f64(PyArray_DATA(a), PyArray_DATA((PyArrayObject *)counts), (size_t)n, (size_t)bins, lo, hi);
        HK_END_NOGIL
    }
    hk_take(HK_HISTOGRAM_F64);
    return result;
}

static PyObject *call_histogram(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_histogram(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_HISTOGRAM);
    return functions[F_HISTOGRAM].original(callable, args, nargsf, kwnames);
}

// ---------------------------------------------------------------- bincount

// The range scans of np.bincount keep BINCOUNT_SCAN_VECTORS running maxima
// in 32-byte vectors, of these types. They are written with the compiler's
// vector types because, written as loops over elements, the compiler kept one
// running maximum for int64: a chain of a compare and a blend through every
// four elements, about 0.3 ns per element on the Intel Core i5-14600KF, as
// long as the kernel itself takes. Several independent maxima leave the loop
// bound by its loads.
typedef uint16_t bincount_u16x16 __attribute__((vector_size(32)));
typedef uint64_t bincount_u64x4 __attribute__((vector_size(32)));
#define BINCOUNT_SCAN_VECTORS 4
// Bytes of one vector, and of the array one step of a scan reads.
#define BINCOUNT_SCAN_VECTOR_BYTES 32
#define BINCOUNT_SCAN_STEP_BYTES (BINCOUNT_SCAN_VECTORS * BINCOUNT_SCAN_VECTOR_BYTES)

#if defined(__aarch64__)

// ARM64 versions of the two scans below, with the same answers. Written for
// the 32-byte vectors of x86-64, the int16 scan took about 0.023 ns per
// element on the Apple M1 Pro and the int64 scan 0.090, both short of the
// three 16-byte loads per cycle the core can make. These keep eight running
// maxima of NEON's 16 bytes. The last step ends at n and overlaps the one
// before it, which a maximum does not mind.

// Running maxima of the ARM64 scans, and the bytes one step reads: 128.
#define BINCOUNT_NEON_VECTORS 8
#define BINCOUNT_NEON_STEP_BYTES (BINCOUNT_NEON_VECTORS * 16)

// As the x86-64 bincount_top_i16 below: the largest of the `n` int16
// elements at `x`, each read as unsigned. A step is 64 elements, an unsigned
// maximum (umax) per eight; fewer than 64 are read one at a time. About
// 0.014 ns per element in the first-level cache, where the x86-64 code took
// 0.023 (Apple M1 Pro).
static uint64_t bincount_top_i16(const int16_t *x, size_t n) {
    const size_t step = BINCOUNT_NEON_STEP_BYTES / sizeof(int16_t);
    uint16_t largest = 0;
    if (n < step) {
        for (size_t i = 0; i < n; i++) largest = (uint16_t)x[i] > largest ? (uint16_t)x[i] : largest;
        return largest;
    }
    uint16x8_t top[BINCOUNT_NEON_VECTORS];
    for (size_t k = 0; k < BINCOUNT_NEON_VECTORS; k++) top[k] = vdupq_n_u16(0);
    for (size_t i = 0;; i += step) {
        if (i > n - step) i = n - step;
        for (size_t k = 0; k < BINCOUNT_NEON_VECTORS; k++) {
            top[k] = vmaxq_u16(top[k], vreinterpretq_u16_s16(vld1q_s16(x + i + 8 * k)));
        }
        if (i == n - step) break;
    }
    for (size_t k = 1; k < BINCOUNT_NEON_VECTORS; k++) top[0] = vmaxq_u16(top[0], top[k]);
    return vmaxvq_u16(top[0]);
}

// As the x86-64 bincount_top_i64 below: the largest of the `n` int64
// elements at `x`, each read as unsigned. NEON has no 64-bit maximum, and a
// compare and a select per two elements left the loop at 0.090 ns per
// element. So the elements are taken as pairs of 32-bit words, an unsigned
// maximum (umax) per four words: when the high words' maximum is 0, every
// element is below 2^32 and the largest is the low words' maximum. Otherwise,
// a negative element or one of 2^32 or more, which np.bincount rejects or
// meets rarely, the elements are read again one at a time. A step is 16
// elements; fewer are read one at a time. About 0.059 ns per element in the
// first-level cache (Apple M1 Pro).
static uint64_t bincount_top_i64(const int64_t *x, size_t n) {
    const size_t step = BINCOUNT_NEON_STEP_BYTES / sizeof(int64_t);
    if (n >= step) {
        uint32x4_t top[BINCOUNT_NEON_VECTORS];
        for (size_t k = 0; k < BINCOUNT_NEON_VECTORS; k++) top[k] = vdupq_n_u32(0);
        for (size_t i = 0;; i += step) {
            if (i > n - step) i = n - step;
            for (size_t k = 0; k < BINCOUNT_NEON_VECTORS; k++) {
                top[k] = vmaxq_u32(top[k], vreinterpretq_u32_s64(vld1q_s64(x + i + 2 * k)));
            }
            if (i == n - step) break;
        }
        for (size_t k = 1; k < BINCOUNT_NEON_VECTORS; k++) top[0] = vmaxq_u32(top[0], top[k]);
        // Little-endian: lanes 0 and 2 are the low words, 1 and 3 the high.
        if ((vgetq_lane_u32(top[0], 1) | vgetq_lane_u32(top[0], 3)) == 0) {
            const uint32_t even = vgetq_lane_u32(top[0], 0), odd = vgetq_lane_u32(top[0], 2);
            return even > odd ? even : odd;
        }
    }
    uint64_t largest = 0;
    for (size_t i = 0; i < n; i++) largest = (uint64_t)x[i] > largest ? (uint64_t)x[i] : largest;
    return largest;
}

#else
// The largest of the `n` int16 elements at `x`, each read as unsigned. A
// negative element reads as 0x8000 or more, so the one maximum is both the
// largest element, when none is negative, and the test for a negative one.
// No branch on the data. The vectors are loaded with memcpy, which the
// compiler turns into unaligned loads, and the maximum is a compare and a
// select, which it turns into vpmaxuw.
static HK_TARGET_KERNEL_LEVEL uint64_t bincount_top_i16(const int16_t *x, size_t n) {
    const size_t step = BINCOUNT_SCAN_STEP_BYTES / sizeof(int16_t);
    bincount_u16x16 top[BINCOUNT_SCAN_VECTORS] = {{0}};
    size_t i = 0;
    for (; i + step <= n; i += step) {
        for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
            bincount_u16x16 v;
            memcpy(&v, (const char *)(x + i) + k * BINCOUNT_SCAN_VECTOR_BYTES, sizeof v);
            const bincount_u16x16 more = (bincount_u16x16)(v > top[k]);
            top[k] = (v & more) | (top[k] & ~more);
        }
    }
    uint16_t largest = 0;
    for (; i < n; i++) largest = (uint16_t)x[i] > largest ? (uint16_t)x[i] : largest;
    for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
        for (size_t j = 0; j < sizeof top[k] / sizeof top[k][0]; j++) {
            largest = top[k][j] > largest ? top[k][j] : largest;
        }
    }
    return largest;
}

// As bincount_top_i16 for int64 elements: a negative one reads as 2^63 or
// more. AVX2 has no 64-bit maximum: the compiler makes a signed compare of
// the elements with their top bit flipped, and a blend.
static HK_TARGET_KERNEL_LEVEL uint64_t bincount_top_i64(const int64_t *x, size_t n) {
    const size_t step = BINCOUNT_SCAN_STEP_BYTES / sizeof(int64_t);
    bincount_u64x4 top[BINCOUNT_SCAN_VECTORS] = {{0}};
    size_t i = 0;
    for (; i + step <= n; i += step) {
        for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
            bincount_u64x4 v;
            memcpy(&v, (const char *)(x + i) + k * BINCOUNT_SCAN_VECTOR_BYTES, sizeof v);
            const bincount_u64x4 more = (bincount_u64x4)(v > top[k]);
            top[k] = (v & more) | (top[k] & ~more);
        }
    }
    uint64_t largest = 0;
    for (; i < n; i++) largest = (uint64_t)x[i] > largest ? (uint64_t)x[i] : largest;
    for (size_t k = 0; k < BINCOUNT_SCAN_VECTORS; k++) {
        for (size_t j = 0; j < sizeof top[k] / sizeof top[k][0]; j++) {
            largest = top[k][j] > largest ? top[k][j] : largest;
        }
    }
    return largest;
}
#endif

// np.bincount(x, minlength=m) for a contiguous one-dimensional int16 or
// int64 array of at least one element: the bincount_i16 and bincount_i64
// kernels. `minimum` is m, at least 0.
//
// NumPy sizes its result by the largest element, plus one, or by m if that is
// more, and raises ValueError for a negative element. So the array is read
// once for its largest element first, as an unsigned number, which a negative
// element makes too large for its type's positive range: that call is
// NumPy's, and it raises. Otherwise every element lies in [0, bins), where
// the kernel's counts and NumPy's agree, and the kernel zeroes the counters
// itself. The read cannot be done a block at a time ahead of the kernel, as
// the guards of np.fmax.reduceat and m.any(axis=1) are: the kernel needs the
// number of counters before it reads its first element.
//
// NumPy takes its result from calloc, whose large blocks are pages the system
// zeroes only when they are first touched; the kernel stores every counter.
// A call with far more counters than elements, such as np.bincount([10**9]),
// costs NumPy almost nothing and would have the kernel write 8 GB, so it is
// left to NumPy. So is a largest element whose length overflows, and a
// result that cannot be allocated: NumPy raises its own error.
//
// The array is read twice, by the scan and by the kernel, and the lock is
// released for both all the same: the kernel ignores elements outside
// [0, bins), so a thread that writes the array between the two reads can at
// worst change the counts, never make the kernel write outside them.
static HK_NOINLINE PyObject *bincount_codes(PyArrayObject *x, npy_intp minimum) {
    const bool narrow = PyArray_TYPE(x) == NPY_INT16;
    const npy_intp n = PyArray_SIZE(x);
    // NumPy's answer for no elements is m zeros; the kernel has no largest
    // element to size them by, and the call is NumPy's.
    if (n == 0) return NULL;
    uint64_t top;
    {
        HK_BEGIN_NOGIL(n)
        top = narrow ? bincount_top_i16(PyArray_DATA(x), (size_t)n) : bincount_top_i64(PyArray_DATA(x), (size_t)n);
        HK_END_NOGIL
    }
    // The sign bit of the element type: set in `top` when some element is
    // negative. For int64 it also keeps `top + 1` below 2^63, but the
    // largest int64, whose length NumPy cannot hold, is still excluded.
    const uint64_t sign = narrow ? (uint64_t)1 << 15 : (uint64_t)1 << 63;
    if (top >= sign || top >= (uint64_t)NPY_MAX_INTP) return NULL;
    npy_intp bins = (npy_intp)top + 1;
    if (bins < minimum) bins = minimum;
    if (bins > BINCOUNT_FREE_BINS && bins > n) return NULL;
    PyObject *result = PyArray_SimpleNew(1, &bins, NPY_INTP);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    int64_t *counts = PyArray_DATA((PyArrayObject *)result);
    {
        HK_BEGIN_NOGIL(n)
        if (narrow) bincount_i16(PyArray_DATA(x), counts, (size_t)n, (size_t)bins);
        else bincount_i64(PyArray_DATA(x), counts, (size_t)n, (size_t)bins);
        HK_END_NOGIL
    }
    hk_take(narrow ? HK_BINCOUNT_I16 : HK_BINCOUNT_I64);
    return result;
}

// np.bincount(x, minlength=m) for a contiguous one-dimensional uint8, int16
// or int64 array without weights. For uint8 the kernel always fills 256
// counters. NumPy's result has one entry more than the largest value present,
// or `minlength` entries if that is more, so the counters are cut or padded
// to that length. int16 and int64 go to bincount_codes.
static HK_NOINLINE PyObject *try_bincount(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { X, WEIGHTS, MINLENGTH, PARAMETERS };
    // `x` can only be given by position.
    static const char *const names[PARAMETERS] = {NULL, "weights", "minlength"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[X] == NULL || !hk_is_absent(bound[WEIGHTS])) return NULL;
    // The element types with a kernel. Every other one, bool and int32
    // included, which NumPy converts to intp, stays with NumPy.
    const bool bytes = hk_is_flat_1d(bound[X], NPY_UINT8);
    if (!bytes && !hk_is_flat_1d(bound[X], NPY_INT16) && !hk_is_flat_1d(bound[X], NPY_INT64)) return NULL;

    Py_ssize_t minimum = 0;
    if (bound[MINLENGTH] != NULL) {
        if (!PyLong_CheckExact(bound[MINLENGTH])) return NULL;
        minimum = PyLong_AsSsize_t(bound[MINLENGTH]);
        if (minimum == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return NULL;
        }
        if (minimum < 0) return NULL;
    }

    PyArrayObject *x = (PyArrayObject *)bound[X];
    if (!bytes) return bincount_codes(x, minimum);
    const npy_intp n = PyArray_SIZE(x);
    int64_t counters[BYTE_VALUES];
    {
        HK_BEGIN_NOGIL(n)
        bincount_u8(PyArray_DATA(x), counters, (size_t)n);
        HK_END_NOGIL
    }
    npy_intp used = BYTE_VALUES;
    while (used > 0 && counters[used - 1] == 0) used--;
    npy_intp length = used > minimum ? used : minimum;
    PyObject *result = PyArray_ZEROS(1, &length, NPY_INTP, 0);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *)result), counters, (size_t)used * sizeof counters[0]);
    hk_take(HK_BINCOUNT_U8);
    return result;
}

static PyObject *call_bincount(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_bincount(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_BINCOUNT);
    return functions[F_BINCOUNT].original(callable, args, nargsf, kwnames);
}

// -------------------------------------------------------------------- isin

// Read up to eight int64 test values from a contiguous one-dimensional int64
// array, or from a list or tuple of Python ints that fit int64, which is what
// NumPy would convert such a list to. The kernel reads exactly eight; a
// shorter set is padded by repeating its first value, which does not change
// membership. Returns false for anything else.
static bool isin_test_values(PyObject *test, int64_t values[ISIN_TEST_VALUES]) {
    Py_ssize_t count;
    if (hk_is_flat_1d(test, NPY_INT64)) {
        count = PyArray_SIZE((PyArrayObject *)test);
        if (count < 1 || count > ISIN_TEST_VALUES) return false;
        memcpy(values, PyArray_DATA((PyArrayObject *)test), (size_t)count * sizeof values[0]);
    } else if (PyList_CheckExact(test) || PyTuple_CheckExact(test)) {
        count = PySequence_Fast_GET_SIZE(test);
        if (count < 1 || count > ISIN_TEST_VALUES) return false;
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(test, i);
            int overflow = 0;
            if (!PyLong_CheckExact(item)) return false;
            values[i] = PyLong_AsLongLongAndOverflow(item, &overflow);
            if (overflow != 0) return false;
        }
    } else {
        return false;
    }
    for (Py_ssize_t i = count; i < ISIN_TEST_VALUES; i++) values[i] = values[0];
    return true;
}

// np.isin(element, test_elements) for a contiguous int64 array and at most
// eight int64 test values, with every other argument left at its default.
static HK_NOINLINE PyObject *try_isin(PyObject *element, PyObject *test) {
    int64_t values[ISIN_TEST_VALUES];
    if (!hk_is_flat(element, NPY_INT64) || !isin_test_values(test, values)) return NULL;
    PyArrayObject *x = (PyArrayObject *)element;
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_BOOL);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const npy_intp n = PyArray_SIZE(x);
    {
        HK_BEGIN_NOGIL(n)
        isin_i64(PyArray_DATA(x), values, PyArray_DATA((PyArrayObject *)result), (size_t)n);
        HK_END_NOGIL
    }
    hk_take(HK_ISIN_I64);
    return result;
}

static PyObject *call_isin(HK_VECTORCALL_PARAMETERS) {
    if (PyVectorcall_NARGS(nargsf) == 2 && kwnames == NULL) {
        PyObject *result = try_isin(args[0], args[1]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_ISIN);
    return functions[F_ISIN].original(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------------------ interp

// np.interp(x, xp, fp) for a contiguous float64 array of queries and two
// one-dimensional contiguous float64 arrays of sample points, with `left`,
// `right` and `period` left alone. The result has the shape of `x`.
//
// The kernel requires sample points without a NaN. With one at xp[0] it reads
// before the start of xp and fp, and with one anywhere it returns something
// other than NumPy, whose own answer then depends on which elements its
// search happens to probe (issue 9). So xp is read once first, and a NaN
// sends the call to NumPy. That pass is only worth making when there are
// enough queries to pay for it; see INTERP_MAX_POINTS_PER_QUERY.
//
// Sample points that are not increasing are not detected. NumPy does not
// detect them either; both give a meaningless result, not the same one, and
// the kernel stays inside its buffers whatever the data.
static HK_NOINLINE PyObject *try_interp(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { X, XP, FP, LEFT, RIGHT, PERIOD, PARAMETERS };
    static const char *const names[PARAMETERS] = {"x", "xp", "fp", "left", "right", "period"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[X] == NULL || bound[XP] == NULL || bound[FP] == NULL) return NULL;
    if (!hk_is_absent(bound[LEFT]) || !hk_is_absent(bound[RIGHT]) || !hk_is_absent(bound[PERIOD])) return NULL;
    if (!hk_is_flat(bound[X], NPY_DOUBLE) || !hk_is_flat_1d(bound[XP], NPY_DOUBLE) ||
        !hk_is_flat_1d(bound[FP], NPY_DOUBLE)) return NULL;

    PyArrayObject *x = (PyArrayObject *)bound[X];
    PyArrayObject *xp = (PyArrayObject *)bound[XP];
    const npy_intp queries = PyArray_SIZE(x);
    const npy_intp points = PyArray_SIZE(xp);
    // NumPy raises on empty sample points and on lengths that differ.
    if (points < 1 || points != PyArray_SIZE((PyArrayObject *)bound[FP])) return NULL;
    if (points / INTERP_MAX_POINTS_PER_QUERY > queries) return NULL;

    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_DOUBLE);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    bool clean;
    {
        HK_BEGIN_NOGIL(queries)
        clean = !hk_has_nan(PyArray_DATA(xp), (size_t)points);
        if (clean) {
            interp_f64(PyArray_DATA(x), PyArray_DATA(xp), PyArray_DATA((PyArrayObject *)bound[FP]),
                       PyArray_DATA((PyArrayObject *)result), (size_t)points, (size_t)queries);
        }
        HK_END_NOGIL
    }
    if (!clean) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_INTERP_F64);
    return result;
}

static PyObject *call_interp(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_interp(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_INTERP);
    return functions[F_INTERP].original(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------ the mask operations
//
// The function forms of x[mask], x[mask] = v and mask.nonzero(). Each takes
// the kernel under the conditions of the indexing or method hook that serves
// the same kernel, and under whatever more NumPy's own function requires.

// np.compress(condition, a) for a one-dimensional boolean condition and a
// contiguous float32 or float64 array of any shape and the same size, with
// no `axis` and no `out`. Without an axis NumPy flattens `a` in C order, so
// the result is the flat array's selected elements. NumPy raises for a
// condition of more than one dimension, and cuts or rejects one of another
// length; both go to NumPy. With `out=` the result would be written into the
// caller's buffer, and a scratch kernel must only ever see a buffer of the
// binding's own: that call is NumPy's too.
static HK_NOINLINE PyObject *try_compress(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { CONDITION, A, AXIS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"condition", "a", "axis", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[CONDITION] == NULL || bound[A] == NULL) return NULL;
    if (!hk_is_absent(bound[AXIS]) || !hk_is_absent(bound[OUT])) return NULL;
    if (!hk_is_plain_array(bound[CONDITION]) || PyArray_NDIM((PyArrayObject *)bound[CONDITION]) != 1) return NULL;
    return hk_try_compress(bound[A], bound[CONDITION]);
}

static PyObject *call_compress(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_compress(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_COMPRESS);
    return functions[F_COMPRESS].original(callable, args, nargsf, kwnames);
}

// np.extract(condition, arr) for a boolean condition and a contiguous float32
// or float64 array of the same size, each of any shape. NumPy flattens both
// in C order and takes the elements where the flat condition is True.
static HK_NOINLINE PyObject *try_extract(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { CONDITION, ARR, PARAMETERS };
    static const char *const names[PARAMETERS] = {"condition", "arr"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[CONDITION] == NULL || bound[ARR] == NULL) return NULL;
    return hk_try_compress(bound[ARR], bound[CONDITION]);
}

// A call left to NumPy runs np.nonzero on the condition, which the method
// hook may serve; that is counted there, under nonzero.
static PyObject *call_extract(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_extract(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_COMPRESS);
    return functions[F_EXTRACT].original(callable, args, nargsf, kwnames);
}

// np.putmask(a, mask, values) for a contiguous writable float64 `a`, a
// contiguous boolean mask of the same size, each of any shape, and a Python
// float or int as the value. NumPy reads both arrays in C order and requires
// only their sizes to match. `a` can only be given by position.
static HK_NOINLINE bool try_putmask(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, MASK, VALUES, PARAMETERS };
    static const char *const names[PARAMETERS] = {NULL, "mask", "values"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return false;
    if (bound[A] == NULL || bound[MASK] == NULL || bound[VALUES] == NULL) return false;
    return hk_try_maskfill(bound[A], bound[MASK], bound[VALUES]);
}

// NumPy's putmask returns None.
static PyObject *call_putmask(HK_VECTORCALL_PARAMETERS) {
    if (try_putmask(args, PyVectorcall_NARGS(nargsf), kwnames)) Py_RETURN_NONE;
    hk_pass(HK_OP_MASKFILL);
    return functions[F_PUTMASK].original(callable, args, nargsf, kwnames);
}

// np.copyto(dst, src, where=mask) for a contiguous writable float64 `dst`, a
// Python float or int as `src`, and a contiguous boolean mask of the same
// shape, with no `casting`. NumPy broadcasts the mask against `dst`, so a
// mask of another shape, even of the same size, is NumPy's to broadcast or
// refuse. Sets *masked when the call has a `where` argument at all: only
// then is it a masked assignment, and only then is a call left to NumPy
// counted.
static HK_NOINLINE bool try_copyto(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames, bool *masked) {
    enum { DST, SRC, CASTING, WHERE, PARAMETERS };
    static const char *const names[PARAMETERS] = {"dst", "src", "casting", "where"};
    PyObject *bound[PARAMETERS];
    *masked = false;
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return false;
    *masked = bound[WHERE] != NULL;
    if (bound[DST] == NULL || bound[SRC] == NULL || bound[WHERE] == NULL || bound[CASTING] != NULL) return false;
    if (!hk_is_plain_array(bound[DST]) || !hk_is_plain_array(bound[WHERE])) return false;
    if (!hk_same_shape((PyArrayObject *)bound[DST], (PyArrayObject *)bound[WHERE])) return false;
    return hk_try_maskfill(bound[DST], bound[WHERE], bound[SRC]);
}

// NumPy's copyto returns None. A copy without `where` is not a masked
// assignment and is passed on without being counted.
static PyObject *call_copyto(HK_VECTORCALL_PARAMETERS) {
    bool masked;
    if (try_copyto(args, PyVectorcall_NARGS(nargsf), kwnames, &masked)) Py_RETURN_NONE;
    if (masked) hk_pass(HK_OP_MASKFILL);
    return functions[F_COPYTO].original(callable, args, nargsf, kwnames);
}

// -------------------------------------------------------------------- clip
//
// `np.clip(x, lo, hi)` runs a Python wrapper, then `x.clip`, then NumPy's
// `_methods._clip`, then the `clip` ufunc, whose inner loop the loop hook
// serves. On a small array that path is nearly the whole call. The function
// and method hooks call the kernel directly when the call needs nothing the
// ufunc machinery does: no output buffer, no `dtype`, `casting` or `where`,
// and bounds that leave the result in the array's element type.
//
// Which element type NumPy gives the result follows NEP 50. A Python float
// or int is a weak scalar and takes the array's type: a float32 array clipped
// to Python floats stays float32. A NumPy scalar is not: clipped to
// np.float64 bounds, a float32 array gives float64. So a bound is accepted
// when it is a Python float, a Python int, or a NumPy scalar of exactly the
// array's type, and anything else goes to NumPy.

// The smallest normal float32 and the largest finite one, as doubles. NumPy
// converts a Python float bound for a float32 array with a C cast, which
// raises the overflow flag beyond the largest (NumPy warns) and the underflow
// flag below the smallest normal. Such a bound is left to NumPy, and the
// conversion here never raises a flag NumPy reports.
#define CLIP_F32_SMALLEST_NORMAL ((double)FLT_MIN)
#define CLIP_F32_LARGEST ((double)FLT_MAX)

// True when np.clip accepts the keywords `min` and `max`, which NumPy 2.1
// added. In NumPy 2.0 they are passed on to the ufunc, which rejects them.
// Set when the hook is applied.
static bool clip_takes_min_max;

// Convert a bound for a float64 array: a Python float, a Python int a double
// holds exactly, or an np.float64. Returns false for anything else.
static bool clip_bound_f64(PyObject *bound, double *value) {
    if (Py_IS_TYPE(bound, &PyDoubleArrType_Type)) {
        *value = PyArrayScalar_VAL(bound, Double);
        return true;
    }
    return hk_as_exact_double(bound, value);
}

// Convert a bound for a float32 array as NumPy does: an np.float32 as it is,
// a Python float rounded to the nearest float32, a Python int only when a
// float32 holds it exactly. Returns false for anything else, and for a
// Python float that is a NaN (a NaN bound is NumPy's either way) or whose
// conversion would overflow or underflow. The conversion can raise only the
// inexact flag, which NumPy never reports.
static bool clip_bound_f32(PyObject *bound, float *value) {
    if (Py_IS_TYPE(bound, &PyFloatArrType_Type)) {
        *value = PyArrayScalar_VAL(bound, Float);
        return true;
    }
    double d;
    if (PyFloat_CheckExact(bound)) {
        d = PyFloat_AS_DOUBLE(bound);
        // The comparison macros, unlike the operators, raise no flag for a NaN.
        const double magnitude = fabs(d);
        const bool in_range = magnitude == 0.0 || isinf(d) ||
                              (isgreaterequal(magnitude, CLIP_F32_SMALLEST_NORMAL) &&
                               islessequal(magnitude, CLIP_F32_LARGEST));
        if (!in_range) return false;
        *value = (float)d;
        return true;
    }
    if (!PyLong_CheckExact(bound) || !hk_as_exact_double(bound, &d)) return false;
    *value = (float)d;
    return (double)*value == d;
}

// Bytes of the array an in-place clip copies out and clips back at a time,
// and the size of the stack buffer that holds the copy. Measured on the
// Apple M1 Pro from Python: 2 KB and 4 KB are level, 8 KB is 5% to 10%
// slower, and from 16 KB a million float64 elements take 1.2 ms or more
// against NumPy's 0.66 ms, the C library's copy of that size apparently
// leaving the cache. See docs/clip-in-place-design.md.
#define CLIP_IN_PLACE_RUN_BYTES 4096

void hk_clip_in_place(hk_kernel kernel, void *data, size_t n, double lo, double hi) {
    // Doubles, so that the buffer is aligned for either element type.
    double buffer[CLIP_IN_PLACE_RUN_BYTES / sizeof(double)];
    const size_t element = kernel == HK_CLIP_F32 ? sizeof(float) : sizeof(double);
    const size_t run = CLIP_IN_PLACE_RUN_BYTES / element;
    char *at = data;
    for (size_t done = 0; done < n; done += run) {
        const size_t count = n - done < run ? n - done : run;
        // The run is copied out, and the kernel clips the copy back over it:
        // each element is clipped once, from its own value, and the kernel
        // never sees overlapping buffers.
        memcpy(buffer, at, count * element);
        if (kernel == HK_CLIP_F32) clip_f32((const float *)buffer, (float *)at, count, (float)lo, (float)hi);
        else clip_f64(buffer, (double *)at, count, lo, hi);
        at += count * element;
    }
}

// Run a clip kernel on the whole of `x`, into a new array of its shape, or
// into `x` itself when `in_place` is true. The kernels leave the
// invalid-operation flag set on NaN data; NumPy's own loop ends by clearing
// the flags, and so does this, as the loop hook does.
static PyObject *run_clip(PyArrayObject *x, hk_kernel kernel, double lo, double hi, bool in_place) {
    PyObject *result;
    if (in_place) {
        result = Py_NewRef((PyObject *)x);
    } else {
        result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), PyArray_TYPE(x));
        if (result == NULL) {
            PyErr_Clear();
            return NULL;
        }
    }
    const npy_intp n = PyArray_SIZE(x);
    void *out = PyArray_DATA((PyArrayObject *)result);
    {
        HK_BEGIN_NOGIL(n)
        if (in_place) hk_clip_in_place(kernel, out, (size_t)n, lo, hi);
        else if (kernel == HK_CLIP_F32) clip_f32(PyArray_DATA(x), out, (size_t)n, (float)lo, (float)hi);
        else clip_f64(PyArray_DATA(x), out, (size_t)n, lo, hi);
        feclearexcept(FE_ALL_EXCEPT);
        HK_END_NOGIL
    }
    hk_take(kernel);
    return result;
}

// ---- int64, on short arrays
//
// No kernel serves int64, and NumPy's clip loop for it is quick: what a
// short call costs is NumPy's call path, about 1.7 us at 32 elements, against
// a few nanoseconds for the loop. So the function and method hooks run a
// plain C loop of their own, up to CLIP_I64_MAX_N elements; above it NumPy's
// own loop, in vector code, is the better one and the call goes on to NumPy.
// Such a call was never counted, as no hook served an int64 clip, and it is
// still not counted.
//
// NumPy keeps the result int64 for a bound that is a Python int, a weak
// scalar under NEP 50, and for an np.int64. A Python int outside the range of
// int64 is NumPy's: it raises for some and accepts others. Any other bound,
// a Python float, a bool or a NumPy scalar of another type, is NumPy's too:
// most change the result's type.

// Most elements of an int64 array the binding's loop clips. Measured from
// Python on the Intel Core i5-14600KF, `np.clip(x, -100, 100)` was 16x
// faster than NumPy at 16 elements, 7.1x at 1,024 and 3.5x at 4,096, the
// largest size this limit was chosen among. The loop stays ahead further
// (2.0x at 16,384, 1.5x at 32,768, level from 262,144), but those sizes are
// not short arrays: they are left to NumPy, and the loop never needs to
// release the interpreter lock, which the binding does from HK_NOGIL_MIN_N.
#define CLIP_I64_MAX_N 4096

// out[i] = min(max(x[i], lo), hi) for the `n` elements at `x`, as NumPy's
// clip loop computes it for integers. Compiled for the kernels' level, where
// the compiler makes it a compare and a blend per four elements; for any
// x86-64 it would leave it a scalar loop, since SSE2 has no 64-bit compare.
static HK_TARGET_KERNEL_LEVEL void clip_i64_loop(const int64_t *restrict x, int64_t *restrict out, size_t n,
                                                 int64_t lo, int64_t hi) {
    for (size_t i = 0; i < n; i++) {
        const int64_t raised = x[i] < lo ? lo : x[i];
        out[i] = raised > hi ? hi : raised;
    }
}

// np.clip(x, lo, hi) or x.clip(lo, hi) on a contiguous int64 array: the
// caller has tested the array's type. Returns NULL with no exception set for
// a call that is NumPy's. The lock stays held: the array is short.
static HK_NOINLINE PyObject *try_clip_i64(PyArrayObject *x, PyObject *lo, PyObject *hi) {
    const npy_intp n = PyArray_SIZE(x);
    int64_t lo64, hi64;
    if (n > CLIP_I64_MAX_N || !hk_as_int64(lo, &lo64) || !hk_as_int64(hi, &hi64) || lo64 > hi64) return NULL;
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(x), PyArray_DIMS(x), NPY_INT64);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    clip_i64_loop(PyArray_DATA(x), PyArray_DATA((PyArrayObject *)result), (size_t)n, lo64, hi64);
    hk_take(HK_CLIP_I64);
    return result;
}

// The bounds are compared with islessequal, which raises no flag for a NaN:
// a NaN bound fails the comparison, as does lo > hi, and both are NumPy's.
// The float32 bounds are compared after their conversion, as the loop sees
// them.
//
// An `out` that is the array itself is clipped in place, through a buffer of
// the binding's (hk_clip_in_place): the kernels are wrong when their output
// is their input (issue 7). NumPy then returns that same array. A read-only
// array is NumPy's, which raises, and so is an in-place int64 clip.
HK_NOINLINE PyObject *hk_try_clip(PyObject *array, PyObject *lo, PyObject *hi, PyObject *out) {
    if (!hk_is_plain_array(array)) return NULL;
    PyArrayObject *x = (PyArrayObject *)array;
    const bool in_place = !hk_is_absent(out);
    if (in_place && (out != array || !PyArray_ISWRITEABLE(x))) return NULL;
    // NumPy returns an empty array without calling the loop; the kernels are
    // never called with n = 0.
    if (PyArray_SIZE(x) < 1) return NULL;
    if (hk_is_flat(array, NPY_INT64)) return in_place ? NULL : try_clip_i64(x, lo, hi);
    if (hk_is_flat(array, NPY_FLOAT)) {
        float lo32, hi32;
        if (!clip_bound_f32(lo, &lo32) || !clip_bound_f32(hi, &hi32) || !islessequal(lo32, hi32)) return NULL;
        return run_clip(x, HK_CLIP_F32, lo32, hi32, in_place);
    }
    if (hk_is_flat(array, NPY_DOUBLE)) {
        double lo64, hi64;
        if (!clip_bound_f64(lo, &lo64) || !clip_bound_f64(hi, &hi64) || !islessequal(lo64, hi64)) return NULL;
        return run_clip(x, HK_CLIP_F64, lo64, hi64, in_place);
    }
    return NULL;
}

// np.clip(a, a_min, a_max) for a contiguous float32 or float64 array, or a
// short int64 one, with
// the bounds given by position, as `a_min=` and `a_max=`, or, where NumPy
// has them, as `min=` and `max=`; `out` absent, None, or for float32 and
// float64 the array itself; nothing else. A bound left out, None, or both
// spellings at once are NumPy's: it raises, or calls np.maximum or
// np.minimum instead.
static HK_NOINLINE PyObject *try_clip(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, A_MIN, A_MAX, OUT, MIN, MAX, PARAMETERS };
    // `min` and `max` can only be given by keyword.
    static const char *const names[PARAMETERS] = {"a", "a_min", "a_max", "out", "min", "max"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, OUT + 1, bound)) return NULL;
    if (bound[A] == NULL) return NULL;
    PyObject *lo = bound[A_MIN], *hi = bound[A_MAX];
    if (bound[MIN] != NULL || bound[MAX] != NULL) {
        if (!clip_takes_min_max || lo != NULL || hi != NULL) return NULL;
        lo = bound[MIN];
        hi = bound[MAX];
    }
    // A bound left out or None is the commonest call left to NumPy, and the
    // cheapest to recognise.
    if (hk_is_absent(lo) || hk_is_absent(hi)) return NULL;
    return hk_try_clip(bound[A], lo, hi, bound[OUT]);
}

// A call left to NumPy reaches `x.clip` and then the clip loop, whose hooks
// count it, so it is not counted here.
static PyObject *call_clip(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_clip(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    return functions[F_CLIP].original(callable, args, nargsf, kwnames);
}

// Find out whether this NumPy's np.clip takes `min` and `max`: they are
// keyword-only parameters of the Python function behind the function object,
// which NumPy keeps as its `_implementation`. If that cannot be read, the
// keywords are left to NumPy.
static void prepare_clip(PyObject *object) {
    clip_takes_min_max = false;
    PyObject *implementation = PyObject_GetAttrString(object, "_implementation");
    PyObject *defaults = implementation ? PyObject_GetAttrString(implementation, "__kwdefaults__") : NULL;
    if (defaults != NULL && PyDict_Check(defaults)) {
        clip_takes_min_max = PyDict_GetItemString(defaults, "min") != NULL &&
                             PyDict_GetItemString(defaults, "max") != NULL;
    }
    Py_XDECREF(defaults);
    Py_XDECREF(implementation);
    PyErr_Clear();
}

// ------------------------------------------------------------------- where
//
// np.where has two operations behind one name. With the condition alone it
// gives the positions that are True, as np.nonzero does: the nonzero_bool
// kernel, and for a matrix the nonzero_2d_bool kernel, counted under
// nonzero. With `x` and `y` it selects, element by element, from `x` where
// the condition is True and from `y` where it is not: the where_f64 kernel,
// counted under where. NumPy accepts no keywords for either, and a call with
// keywords or another number of arguments is passed on without being
// counted.
//
// The selection is taken only when it needs no broadcasting and no type
// conversion: the condition and the arrays have one shape, and the arrays
// one element type. The kernel only moves 64-bit elements, so it serves
// float64 and int64 alike, and a NaN keeps its payload. Which type NumPy
// gives the result follows NEP 50: beside a float64 array a Python float, a
// Python int or an np.float64 leaves it float64, so such a scalar is taken
// on either side, converted as a float64 clip bound is (clip_bound_f64: an
// int only when a double holds it exactly). Beside an int64 array a float
// would change the result's type, and an int may not fit; both are NumPy's.

// Run where_f64 into a new array of the condition's shape and of NumPy type
// `type_num`. `a` and `b` are the data of the arrays selected from, of the
// condition's size; either may be NULL, and that side is then the scalar
// `fill` at every position. The result is the binding's own buffer and cannot
// overlap an input, which the kernel requires.
//
// The kernel reads n elements from each side, so a scalar is spread over a
// buffer on the stack, WHERE_BLOCK elements long, and the kernel runs block
// by block with that buffer as the scalar's side. A buffer of n elements
// would be written in full and read back from wherever it lies; at a million
// elements that made the call no faster than NumPy's. The blocks give up the
// x86-64 kernel's stores past the caches, which it starts from millions of
// elements.
//
// Both kernels take any non-zero byte of the condition as True, as NumPy
// does, but the spec promises only 0 and 1; the condition is checked first,
// as for the other mask kernels, and one with another byte is NumPy's. Returns NULL with no exception set when the call
// goes to NumPy after all.
static HK_NOINLINE PyObject *run_where(PyArrayObject *condition, const void *a, const void *b, double fill,
                                       int type_num) {
    const npy_intp n = PyArray_SIZE(condition);
    PyObject *result = PyArray_SimpleNew(PyArray_NDIM(condition), PyArray_DIMS(condition), type_num);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    const bool *c = PyArray_DATA(condition);
    double *out = PyArray_DATA((PyArrayObject *)result);
    bool clean = true;
    {
        // The kernel reads each input once and writes only the new array, which
        // no other thread can see yet.
        HK_BEGIN_NOGIL(n)
        if (a != NULL && b != NULL) {
            clean = hk_mask_is_clean((const uint8_t *)c, (size_t)n);
            if (clean) where_f64(c, a, b, out, (size_t)n);
        } else {
            double spread[WHERE_BLOCK];
            const npy_intp filled = n < WHERE_BLOCK ? n : WHERE_BLOCK;
            for (npy_intp i = 0; i < filled; i++) spread[i] = fill;
            const double *array = a != NULL ? a : b;
            // Each block's condition is checked just before the kernel reads it,
            // from the first-level cache. A bad byte drops the whole result.
            for (npy_intp done = 0; done < n; done += WHERE_BLOCK) {
                const size_t count = (size_t)(n - done < WHERE_BLOCK ? n - done : WHERE_BLOCK);
                clean = hk_mask_is_clean((const uint8_t *)c + done, count);
                if (!clean) break;
                if (a != NULL) where_f64(c + done, array + done, spread, out + done, count);
                else where_f64(c + done, spread, array + done, out + done, count);
            }
        }
        HK_END_NOGIL
    }
    if (!clean) {
        Py_DECREF(result);
        return NULL;
    }
    hk_take(HK_WHERE_F64);
    return result;
}

// np.where(condition, x, y) for a contiguous boolean condition with at least
// one dimension and, of exactly its shape, two contiguous float64 arrays, two
// contiguous int64 arrays, or one contiguous float64 array and a scalar that
// keeps the result float64 (see above). An empty condition is taken: the
// kernel writes nothing and the result is NumPy's empty array of that shape.
static HK_NOINLINE PyObject *try_where(PyObject *condition, PyObject *x, PyObject *y) {
    if (!hk_is_flat(condition, NPY_BOOL)) return NULL;
    PyArrayObject *c = (PyArrayObject *)condition;
    double fill = 0.0;
    if (hk_is_plain_array(x) && hk_is_plain_array(y)) {
        const int type_num = PyArray_TYPE((PyArrayObject *)x);
        if (type_num != NPY_DOUBLE && type_num != NPY_INT64) return NULL;
        if (!hk_is_flat(x, type_num) || !hk_is_flat(y, type_num)) return NULL;
        if (!hk_same_shape(c, (PyArrayObject *)x) || !hk_same_shape(c, (PyArrayObject *)y)) return NULL;
        return run_where(c, PyArray_DATA((PyArrayObject *)x), PyArray_DATA((PyArrayObject *)y), fill, type_num);
    }
    if (hk_is_flat(x, NPY_DOUBLE) && hk_same_shape(c, (PyArrayObject *)x) && clip_bound_f64(y, &fill)) {
        return run_where(c, PyArray_DATA((PyArrayObject *)x), NULL, fill, NPY_DOUBLE);
    }
    if (hk_is_flat(y, NPY_DOUBLE) && hk_same_shape(c, (PyArrayObject *)y) && clip_bound_f64(x, &fill)) {
        return run_where(c, NULL, PyArray_DATA((PyArrayObject *)y), fill, NPY_DOUBLE);
    }
    return NULL;
}

// np.where(condition, x, y), and np.where(condition) with nothing but a
// contiguous boolean array of one dimension or two: the positions that are
// True, as np.nonzero gives them. np.where calls NumPy's C code for nonzero directly,
// so the method hook does not see that call.
static PyObject *call_where(HK_VECTORCALL_PARAMETERS) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (kwnames == NULL && nargs == 3) {
        PyObject *result = try_where(args[0], args[1], args[2]);
        if (result != NULL) return result;
        hk_pass(HK_OP_WHERE);
    } else if (kwnames == NULL && nargs == 1) {
        PyObject *result = hk_try_nonzero(args[0]);
        if (result != NULL) return result;
        hk_pass(HK_OP_NONZERO);
    }
    return functions[F_WHERE].original(callable, args, nargsf, kwnames);
}

// --------------------------------------------------------------------- any

// np.any(m, axis=1) and np.any(m, axis=0) for a two-dimensional boolean
// array: the array and the axis, the Python int 1 or -1 for the rows, 0 or
// -2 for the columns, by position or the axis as `axis=`, and nothing else.
// For a plain ndarray np.any calls np.logical_or.reduce itself and never
// reaches the method, so it needs this hook as well; the work is the method
// hook's (hk_try_any_rows, hk_try_any_cols). The parameters are a, axis,
// out, keepdims and, by keyword only, where.
static HK_NOINLINE PyObject *try_any(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    const Py_ssize_t keywords = kwnames == NULL ? 0 : PyTuple_GET_SIZE(kwnames);
    if (nargs < 1 || nargs + keywords != 2) return NULL;
    if (keywords == 1 && !hk_text_is(PyTuple_GET_ITEM(kwnames, 0), "axis")) return NULL;
    // A keyword's value follows the positional arguments, so either way the
    // axis is args[1].
    if (hk_is_last_axis(args[1])) return hk_try_any_rows(args[0]);
    if (hk_is_first_axis(args[1])) return hk_try_any_cols(args[0]);
    return NULL;
}

// `np.any(m)` with the array alone is one of the commonest NumPy calls and
// has no kernel: it goes on at the first test, uncounted, as does any call
// whose first argument is not a plain boolean array of two dimensions. A call
// on such an array with more arguments that is not taken is counted.
static PyObject *call_any(HK_VECTORCALL_PARAMETERS) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if ((nargs > 1 || kwnames != NULL) && nargs >= 1 && hk_is_bool_matrix(args[0])) {
        PyObject *result = try_any(args, nargs, kwnames);
        if (result != NULL) return result;
        hk_pass(HK_OP_ANY);
    }
    return functions[F_ANY].original(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------------------- round

// np.round(a, decimals, out) and np.around, which in NumPy 2 is a second
// function object with the same parameters and the same body: a contiguous
// float64 array, `decimals` absent or a Python int from 0 to 15, `out`
// absent or None, by position or by keyword. The work is the method hook's
// (hk_try_round).
static HK_NOINLINE PyObject *try_round_function(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { A, DECIMALS, OUT, PARAMETERS };
    static const char *const names[PARAMETERS] = {"a", "decimals", "out"};
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, PARAMETERS, bound)) return NULL;
    if (bound[A] == NULL || !hk_is_flat(bound[A], NPY_DOUBLE) || !hk_is_absent(bound[OUT])) return NULL;
    return hk_try_round(bound[A], bound[DECIMALS]);
}

// A call left to NumPy calls the array's `round` method, whose hook counts
// it, so it is not counted here.
static PyObject *call_round_with(HK_VECTORCALL_PARAMETERS, int function) {
    PyObject *result = try_round_function(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    return functions[function].original(callable, args, nargsf, kwnames);
}

static PyObject *call_round(HK_VECTORCALL_PARAMETERS) {
    return call_round_with(callable, args, nargsf, kwnames, F_ROUND);
}

static PyObject *call_around(HK_VECTORCALL_PARAMETERS) {
    return call_round_with(callable, args, nargsf, kwnames, F_AROUND);
}

// ------------------------------------------------------------------ median
//
// np.median is a Python function: it partitions a copy of the array around
// its middle with np.partition, and takes np.mean of the middle element, or
// of the two middle elements for an even count. On the short arrays it is
// called on that is nearly all of the call: about 4.8 us at 32 elements, for
// a selection of a few dozen nanoseconds. So the function hook answers
// np.median(x) on a short float64 array with a loop of the binding's own:
// copy, sort, take the middle. No kernel serves it.
//
// NumPy's mean of the middle elements is their sum, from np.add.reduce,
// divided by their count. For one element that is the element; for two it
// is a + b, then / 2. The loop computes the same, so the result is the same
// bits, with three exceptions, each sent to NumPy:
//
// - A NaN anywhere: NumPy's answer and its handling of it are its own.
// - A flag NumPy would report from the sum or the division: a sum that
//   overflows (NumPy warns), infinities of both signs (invalid, NumPy warns),
//   or a division that underflows (NumPy is silent by default, and raises
//   under np.errstate(under='raise')). Each is recognised from the numbers
//   (median_raises) rather than from the status flags: the compiler may move
//   inline arithmetic across a read of MXCSR, where it does not move a call
//   that writes memory, which is how np.round's loop is fenced.
// - A zero median when the array holds a negative zero. Which zeros end up
//   in the middle depends on where NumPy's partition leaves equal elements,
//   and np.add.reduce of a single -0.0 gives +0.0 in NumPy 2.5; neither is
//   worth reproducing.
//
// Infinities of one sign are numbers here: their sum raises no flag.
//
// Up to MEDIAN_F64_SORT_MAX_N elements the middle elements are found by
// sorting a copy on the stack. Above it they are selected, exactly, by their
// bits: see median_select and docs/median-large-design.md.

// Most elements of a float64 array whose median is found by sorting a copy,
// and the length of the buffer on the stack they are copied to: 512 bytes.
// The sort takes time in the square of the count; the selection takes about
// 0.6 us however short the array, for its counters. Measured from Python on
// the Apple M1 Pro on normally distributed values, the call took 0.17 us by
// the sort at 32 elements, 0.46 at 64, 1.4 at 96 and 2.2 at 128, and 0.6 by
// the selection at 129. (When the sort served up to 128 elements, on the
// Intel Core i5-14600KF np.median(x) was 47x faster than NumPy at 16
// elements and 13x at 64.)
#define MEDIAN_F64_SORT_MAX_N 64

// Most elements of a float64 array whose median the binding takes: the
// selection's counters are 32 bits wide. A larger array is NumPy's.
#define MEDIAN_F64_MAX_N ((npy_intp)UINT32_MAX)

// The selection orders doubles by a key: their bits, with every bit flipped
// for a negative number and only the sign bit set for a positive one. As
// unsigned integers the keys are in the order of the numbers, -inf lowest
// and +inf highest, with -0.0 just below +0.0. A signed compare of 64-bit
// lanes is one instruction on both architectures and an unsigned one is not
// on x86-64, so the range of a buffer of keys is found on the keys with the
// top bit flipped, whose order as signed integers is the same.
#define MEDIAN_KEY_TOP ((uint64_t)1 << 63)

// Width of a digit of the selection, in bits: each pass counts the elements
// by the next MEDIAN_DIGIT_BITS bits of their keys below the bits all of
// them share. Measured on the Apple M1 Pro from 10,000 to 1,048,576
// elements, eleven bits were as fast as twelve or better on most data, and
// sixteen, whose counters leave the first-level cache, slower on most.
#define MEDIAN_DIGIT_BITS 11
#define MEDIAN_DIGITS ((size_t)1 << MEDIAN_DIGIT_BITS)
// Sets of counters a count over many elements uses, element i of a block of
// MEDIAN_BLOCK going to set i. Most of the elements of real data fall in a
// few digits, and of data in order in one digit at a time; with a single
// set each increment would wait for the one before it to the same counter.
// Eight sets of 2,048 counters are 64 KB, half the first-level data cache of
// the M1 Pro's performance cores. They are added together afterwards.
#define MEDIAN_WAYS 8
// Fewest elements counted into MEDIAN_WAYS sets; fewer go into one, where
// clearing 64 KB would cost more than the sets save. At 10,000 elements in
// ascending order eight sets took 7 us where one took 21.
#define MEDIAN_WAYS_MIN_N 4096
// Elements the loops over a set read at a time, as vectors: the keys and
// digits of a block are computed in vectors, and the block is then counted,
// or tested and gathered, one element at a time.
#define MEDIAN_BLOCK 8
// A digit is dense when it holds more than one element in
// 2^MEDIAN_DENSE_SHIFT of the set it is gathered from. A dense digit is
// gathered without a branch on each element, which would be mispredicted
// often; a sparse one by testing each block first and looking at its
// elements only when one of them is in the digit. A sixty-fourth was the
// best of a sixteenth, a thirty-second and a sixty-fourth on an
// astronomical image, whose middle digit holds about 4% of its pixels.
#define MEDIAN_DENSE_SHIFT 6
// Most candidates that are sorted by insertion rather than counted again.
#define MEDIAN_SORT_CANDIDATES 64

// The loops read vectors of the compiler's vector types: 16 bytes on ARM64,
// NEON's width, and 32 on x86-64, AVX2's, where the functions that use them
// are compiled for the kernels' instruction level (HK_TARGET_KERNEL_LEVEL).
#if defined(__aarch64__)
#define MEDIAN_VECTOR_BYTES 16
#else
#define MEDIAN_VECTOR_BYTES 32
#endif
typedef int64_t median_i64xv __attribute__((vector_size(MEDIAN_VECTOR_BYTES)));
typedef uint64_t median_u64xv __attribute__((vector_size(MEDIAN_VECTOR_BYTES)));
// Elements of one vector.
#define MEDIAN_LANES (MEDIAN_VECTOR_BYTES / sizeof(uint64_t))
// Vectors of running minima and maxima the range scans keep, so that the
// scan is bound by its loads rather than by the latency of one chain.
#define MEDIAN_SCAN_VECTORS 4
#define MEDIAN_SCAN_STEP (MEDIAN_SCAN_VECTORS * MEDIAN_LANES)

// The unsigned key of the double whose bits are `bits`.
static inline uint64_t median_key(uint64_t bits) {
    return bits ^ ((uint64_t)((int64_t)bits >> 63) | MEDIAN_KEY_TOP);
}

// The unsigned keys of the doubles whose bits are the lanes of `bits`.
static inline HK_TARGET_KERNEL_LEVEL median_u64xv median_vector_key(median_u64xv bits) {
    return bits ^ ((median_u64xv)((median_i64xv)bits >> 63) | MEDIAN_KEY_TOP);
}

// The double whose unsigned key is `key`.
static inline double median_value(uint64_t key) {
    const uint64_t bits = key & MEDIAN_KEY_TOP ? key ^ MEDIAN_KEY_TOP : ~key;
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

// The smallest and largest key of a set of elements, unsigned.
typedef struct {
    // The smallest key.
    uint64_t low;
    // The largest key.
    uint64_t high;
} median_range;

// The signed keys of the doubles whose bits are the lanes of `bits`: all but
// the sign bit flipped for a negative number. In the order of the numbers as
// signed integers, the NaNs outside the infinities.
static inline HK_TARGET_KERNEL_LEVEL median_i64xv median_signed_keys(median_i64xv bits) {
    return bits ^ ((bits >> 63) & (int64_t)~MEDIAN_KEY_TOP);
}

// The signed keys of +inf and -inf. A signed key above the first or below the
// second is a NaN's.
#define MEDIAN_SIGNED_KEY_PLUS_INF ((int64_t)0x7ff0000000000000)
#define MEDIAN_SIGNED_KEY_MINUS_INF ((int64_t)0x800fffffffffffffULL)

// The smallest and largest key of the `n` doubles, n >= 1, whose bits are at
// `x`, into `range`, and into `in_order` whether they are in order, either
// no element below the one before it or none above it, in the order of the
// keys; false, with neither set, when one of them is a NaN. Integer operations only, so
// no floating-point flag is raised, by a signalling NaN either.
//
// Each step of MEDIAN_SCAN_STEP elements also compares every element with
// the one before it, until a fall and a rise have both been seen; on data in
// no order that is within the first step, and the rest of the scan is the
// minimum and maximum alone. The last step ends at n and overlaps the one
// before it, which none of this minds.
static HK_TARGET_KERNEL_LEVEL bool median_scan(const uint64_t *x, size_t n, median_range *range, bool *in_order) {
    int64_t first;
    memcpy(&first, x, sizeof first);
    first ^= (first >> 63) & (int64_t)~MEDIAN_KEY_TOP;
    int64_t lo = first, hi = first;
    bool falls = false, rises = false;
    if (n > MEDIAN_SCAN_STEP) {
        median_i64xv low[MEDIAN_SCAN_VECTORS], high[MEDIAN_SCAN_VECTORS];
        median_i64xv fell = {0}, rose = {0};
        for (size_t k = 0; k < MEDIAN_SCAN_VECTORS; k++) low[k] = high[k] = (median_i64xv){0} + first;
        // Steps start at 1, so that every element from the second on is
        // compared with the one before it.
        for (size_t i = 1;; i += MEDIAN_SCAN_STEP) {
            if (i > n - MEDIAN_SCAN_STEP) i = n - MEDIAN_SCAN_STEP;
            const bool ordering = !(falls && rises);
            for (size_t k = 0; k < MEDIAN_SCAN_VECTORS; k++) {
                median_i64xv v;
                memcpy(&v, x + i + k * MEDIAN_LANES, sizeof v);
                v = median_signed_keys(v);
                if (ordering) {
                    median_i64xv before;
                    memcpy(&before, x + i + k * MEDIAN_LANES - 1, sizeof before);
                    before = median_signed_keys(before);
                    fell |= (median_i64xv)(v < before);
                    rose |= (median_i64xv)(v > before);
                }
                const median_i64xv below = (median_i64xv)(v < low[k]), above = (median_i64xv)(v > high[k]);
                low[k] = (v & below) | (low[k] & ~below);
                high[k] = (v & above) | (high[k] & ~above);
            }
            if (ordering) {
                for (size_t j = 0; j < MEDIAN_LANES; j++) {
                    falls |= fell[j] != 0;
                    rises |= rose[j] != 0;
                }
            }
            if (i == n - MEDIAN_SCAN_STEP) break;
        }
        for (size_t k = 0; k < MEDIAN_SCAN_VECTORS; k++) {
            for (size_t j = 0; j < MEDIAN_LANES; j++) {
                lo = low[k][j] < lo ? low[k][j] : lo;
                hi = high[k][j] > hi ? high[k][j] : hi;
            }
        }
    } else {
        int64_t before = first;
        for (size_t i = 1; i < n; i++) {
            int64_t v;
            memcpy(&v, x + i, sizeof v);
            v ^= (v >> 63) & (int64_t)~MEDIAN_KEY_TOP;
            falls |= v < before;
            rises |= v > before;
            before = v;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
    }
    if (lo < MEDIAN_SIGNED_KEY_MINUS_INF || hi > MEDIAN_SIGNED_KEY_PLUS_INF) return false;
    // The unsigned form: the signed one with its top bit flipped.
    range->low = (uint64_t)lo ^ MEDIAN_KEY_TOP;
    range->high = (uint64_t)hi ^ MEDIAN_KEY_TOP;
    *in_order = !falls || !rises;
    return true;
}

// The smallest and largest of the `n` unsigned keys at `x`, n >= 1, into
// `range`, by signed compares of the keys with their top bit flipped. The
// last step ends at n and overlaps the one before it, which a minimum and a
// maximum do not mind.
static HK_TARGET_KERNEL_LEVEL void median_key_range(const uint64_t *x, size_t n, median_range *range) {
    const int64_t flip = (int64_t)MEDIAN_KEY_TOP;
    int64_t lo = (int64_t)x[0] ^ flip, hi = lo;
    if (n >= MEDIAN_SCAN_STEP) {
        median_i64xv low[MEDIAN_SCAN_VECTORS], high[MEDIAN_SCAN_VECTORS];
        for (size_t k = 0; k < MEDIAN_SCAN_VECTORS; k++) low[k] = high[k] = (median_i64xv){0} + lo;
        for (size_t i = 0;; i += MEDIAN_SCAN_STEP) {
            if (i > n - MEDIAN_SCAN_STEP) i = n - MEDIAN_SCAN_STEP;
            for (size_t k = 0; k < MEDIAN_SCAN_VECTORS; k++) {
                median_i64xv v;
                memcpy(&v, x + i + k * MEDIAN_LANES, sizeof v);
                v ^= flip;
                const median_i64xv below = (median_i64xv)(v < low[k]), above = (median_i64xv)(v > high[k]);
                low[k] = (v & below) | (low[k] & ~below);
                high[k] = (v & above) | (high[k] & ~above);
            }
            if (i == n - MEDIAN_SCAN_STEP) break;
        }
        for (size_t k = 0; k < MEDIAN_SCAN_VECTORS; k++) {
            for (size_t j = 0; j < MEDIAN_LANES; j++) {
                lo = low[k][j] < lo ? low[k][j] : lo;
                hi = high[k][j] > hi ? high[k][j] : hi;
            }
        }
    } else {
        for (size_t i = 1; i < n; i++) {
            const int64_t v = (int64_t)x[i] ^ flip;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
    }
    range->low = (uint64_t)(lo ^ flip);
    range->high = (uint64_t)(hi ^ flip);
}

// The elements one pass of the selection reads: the array itself, whose
// elements are bits to be turned into keys, or the binding's buffer of the
// candidates, which holds keys.
typedef struct {
    // The elements. The array's need not be aligned, so they are read with
    // memcpy, which the compiler turns into plain loads.
    const uint64_t *elements;
    // How many.
    size_t count;
    // Whether they are the array's bits rather than keys.
    bool raw;
} median_source;

// Key `i` of `source`.
static inline uint64_t median_source_key(median_source source, size_t i) {
    uint64_t value;
    memcpy(&value, source.elements + i, sizeof value);
    return source.raw ? median_key(value) : value;
}

// The keys of the MEDIAN_LANES elements of `source` from `i`.
static inline HK_TARGET_KERNEL_LEVEL median_u64xv median_source_keys(median_source source, size_t i) {
    median_u64xv value;
    memcpy(&value, source.elements + i, sizeof value);
    return source.raw ? median_vector_key(value) : value;
}

// The digit of `key`: its MEDIAN_DIGIT_BITS bits from bit `shift` up.
static inline size_t median_digit(uint64_t key, unsigned shift) {
    return (size_t)(key >> shift) & (MEDIAN_DIGITS - 1);
}

// Count the elements of `source` by their digit at `shift` into the first
// MEDIAN_DIGITS of `counts`, which holds MEDIAN_WAYS * MEDIAN_DIGITS: in
// MEDIAN_WAYS sets added together afterwards when there are at least
// MEDIAN_WAYS_MIN_N elements, else in one. Every element is counted once,
// whatever its value, so the counts add up to source.count.
static HK_TARGET_KERNEL_LEVEL void median_count(median_source source, unsigned shift, uint32_t *counts) {
    const size_t n = source.count;
    const bool ways = n >= MEDIAN_WAYS_MIN_N;
    memset(counts, 0, (ways ? MEDIAN_WAYS : 1) * MEDIAN_DIGITS * sizeof counts[0]);
    size_t i = 0;
    for (; i + MEDIAN_BLOCK <= n; i += MEDIAN_BLOCK) {
        uint64_t digits[MEDIAN_BLOCK];
        for (size_t k = 0; k < MEDIAN_BLOCK; k += MEDIAN_LANES) {
            const median_u64xv digit = (median_source_keys(source, i + k) >> shift) & (MEDIAN_DIGITS - 1);
            memcpy(digits + k, &digit, sizeof digit);
        }
        if (ways) {
            for (size_t k = 0; k < MEDIAN_BLOCK; k++) counts[k * MEDIAN_DIGITS + digits[k]]++;
        } else {
            for (size_t k = 0; k < MEDIAN_BLOCK; k++) counts[digits[k]]++;
        }
    }
    for (; i < n; i++) counts[median_digit(median_source_key(source, i), shift)]++;
    if (!ways) return;
    for (size_t w = 1; w < MEDIAN_WAYS; w++) {
        for (size_t d = 0; d < MEDIAN_DIGITS; d++) counts[d] += counts[w * MEDIAN_DIGITS + d];
    }
}

// The largest key of `source` whose digit at `shift` is `low_digit` and the
// smallest whose digit is `high_digit`, into keys[0] and keys[1]: the two
// middle elements when they lie in different digits. False when either
// digit holds no element, which happens only when the array changed while it
// was read.
static HK_TARGET_KERNEL_LEVEL bool median_edges(median_source source, unsigned shift, size_t low_digit,
                                                size_t high_digit, uint64_t keys[2]) {
    uint64_t largest = 0, smallest = UINT64_MAX;
    bool low_seen = false, high_seen = false;
    for (size_t i = 0; i < source.count; i++) {
        const uint64_t key = median_source_key(source, i);
        const size_t digit = median_digit(key, shift);
        if (digit == low_digit) {
            low_seen = true;
            largest = key > largest ? key : largest;
        } else if (digit == high_digit) {
            high_seen = true;
            smallest = key < smallest ? key : smallest;
        }
    }
    keys[0] = largest;
    keys[1] = smallest;
    return low_seen && high_seen;
}

// Copy the keys of the elements of `source` whose digit at `shift` is
// `digit`, of which the count said there are `capacity`, to `out`, which
// holds capacity + MEDIAN_BLOCK. `out` may be `source.elements` itself: no
// key is written before the element at its place has been read. Returns how
// many there were, or capacity + 1 when there were more, which happens only
// when the array changed while it was read; nothing is then written past
// out[capacity + MEDIAN_BLOCK - 1].
//
// A dense digit (see MEDIAN_DENSE_SHIFT) is gathered without a branch: each
// key is stored at the next free place, which advances only when it is in
// the digit, and the count is checked once a block, so a block may store up
// to MEDIAN_BLOCK - 1 keys past the capacity before the check. A sparse one
// is gathered by testing a block's digits in vectors first.
static HK_TARGET_KERNEL_LEVEL size_t median_gather(median_source source, unsigned shift, size_t digit, uint64_t *out,
                                                   size_t capacity) {
    const size_t n = source.count;
    const bool dense = capacity > n >> MEDIAN_DENSE_SHIFT;
    size_t kept = 0, i = 0;
    for (; i + MEDIAN_BLOCK <= n; i += MEDIAN_BLOCK) {
        if (dense) {
            for (size_t k = 0; k < MEDIAN_BLOCK; k++) {
                const uint64_t key = median_source_key(source, i + k);
                out[kept] = key;
                kept += median_digit(key, shift) == digit;
            }
            if (kept > capacity) return capacity + 1;
            continue;
        }
        median_u64xv hits = {0};
        for (size_t k = 0; k < MEDIAN_BLOCK; k += MEDIAN_LANES) {
            hits |= (median_u64xv)(((median_source_keys(source, i + k) >> shift) & (MEDIAN_DIGITS - 1)) == digit);
        }
        uint64_t any = 0;
        for (size_t j = 0; j < MEDIAN_LANES; j++) any |= hits[j];
        if (!any) continue;
        for (size_t k = 0; k < MEDIAN_BLOCK; k++) {
            const uint64_t key = median_source_key(source, i + k);
            if (median_digit(key, shift) != digit) continue;
            if (kept == capacity) return capacity + 1;
            out[kept++] = key;
        }
    }
    for (; i < n; i++) {
        const uint64_t key = median_source_key(source, i);
        if (median_digit(key, shift) != digit) continue;
        if (kept == capacity) return capacity + 1;
        out[kept++] = key;
    }
    return kept;
}

// Sort the `n` keys at `v` into ascending order by insertion.
static void median_sort_keys(uint64_t *v, size_t n) {
    for (size_t i = 1; i < n; i++) {
        const uint64_t value = v[i];
        size_t j = i;
        for (; j > 0 && v[j - 1] > value; j--) v[j] = v[j - 1];
        v[j] = value;
    }
}

// What a selection found.
typedef enum {
    // The two middle keys.
    MEDIAN_SELECTED,
    // A NaN: the call is NumPy's.
    MEDIAN_HAS_NAN,
    // No memory for the counters or the candidates: the call is NumPy's.
    MEDIAN_NO_MEMORY,
    // The array changed while it was read: the call is NumPy's.
    MEDIAN_CHANGED,
} median_outcome;

// Find the keys of the elements of rank `ranks[0]` and `ranks[1]`, counted
// from 0 in ascending order, among the `n` doubles whose bits are at `x`,
// into `keys`: for an even count the two middle elements, for an odd one the
// middle element twice. Exact: it works on the bits alone, with integer
// operations.
//
// The first read of the array finds the range of the keys and whether the
// array is already in order; one in order, ascending or descending, has its
// middle elements at known places, where NumPy's partition is also quick.
// Otherwise it is a radix selection that looks only at the bits that vary.
// The range of the keys gives the bits all of them share; the elements are counted by
// the next MEDIAN_DIGIT_BITS bits below those, which says which digit holds
// each rank. If both ranks fall in one digit, the elements with that digit
// are copied to a buffer, and the selection goes on among them, whose keys
// share MEDIAN_DIGIT_BITS more bits; the array itself is read three times
// (range, count, copy). If the ranks fall in two digits, the lower is the
// largest key in its digit and the upper the smallest in its, which one
// more pass finds. A set whose keys are all equal is its own answer, and one
// of at most MEDIAN_SORT_CANDIDATES elements is sorted. Each count leaves
// MEDIAN_DIGIT_BITS fewer bits that vary, so there are at most six.
//
// The array may change while it is read if another thread writes to it,
// with the interpreter lock released or not. Every write is then still
// inside the binding's buffers, and a count that disagrees with what is
// copied gives MEDIAN_CHANGED. Memory: MEDIAN_WAYS * MEDIAN_DIGITS counters,
// 64 KB, and a buffer as long as the digit that holds the middle, at most
// eight bytes an element.
static median_outcome median_select(const uint64_t *x, size_t n, size_t ranks[2], uint64_t keys[2]) {
    median_range range;
    bool in_order;
    if (!median_scan(x, n, &range, &in_order)) return MEDIAN_HAS_NAN;
    if (in_order) {
        // The two middle places hold the two middle elements: in ascending
        // order the one of rank (n-1)/2 first, in descending order the one
        // of rank n/2, which is the other for an even count and the same for
        // an odd one.
        uint64_t first, second;
        memcpy(&first, x + ranks[0], sizeof first);
        memcpy(&second, x + ranks[1], sizeof second);
        first = median_key(first);
        second = median_key(second);
        keys[0] = first < second ? first : second;
        keys[1] = first < second ? second : first;
        return MEDIAN_SELECTED;
    }
    uint32_t *counts = malloc(MEDIAN_WAYS * MEDIAN_DIGITS * sizeof(uint32_t));
    if (counts == NULL) return MEDIAN_NO_MEMORY;
    uint64_t *candidates = NULL;
    median_source source = {x, n, true};
    median_outcome outcome = MEDIAN_SELECTED;
    for (;;) {
        if (range.low == range.high) {
            keys[0] = keys[1] = range.low;
            break;
        }
        if (!source.raw && source.count <= MEDIAN_SORT_CANDIDATES) {
            median_sort_keys(candidates, source.count);
            keys[0] = candidates[ranks[0]];
            keys[1] = candidates[ranks[1]];
            break;
        }
        // The bits from `varying` up are the same in every key. The digit is
        // the MEDIAN_DIGIT_BITS bits just below them, or the lowest bits.
        const unsigned varying = 64 - (unsigned)__builtin_clzll(range.low ^ range.high);
        const unsigned shift = varying > MEDIAN_DIGIT_BITS ? varying - MEDIAN_DIGIT_BITS : 0;
        median_count(source, shift, counts);
        // The digits that hold the two ranks, and the elements below the first.
        // The counts add up to more than either rank, so the walks end inside
        // the counters.
        size_t low_digit = 0, below = 0;
        while (below + counts[low_digit] <= ranks[0]) below += counts[low_digit++];
        size_t high_digit = low_digit, below_high = below;
        while (below_high + counts[high_digit] <= ranks[1]) below_high += counts[high_digit++];
        if (high_digit != low_digit) {
            if (!median_edges(source, shift, low_digit, high_digit, keys)) outcome = MEDIAN_CHANGED;
            break;
        }
        const size_t kept = counts[low_digit];
        if (source.raw) {
            // Later rounds gather in place, from fewer elements.
            candidates = malloc((kept + MEDIAN_BLOCK) * sizeof candidates[0]);
            if (candidates == NULL) {
                outcome = MEDIAN_NO_MEMORY;
                break;
            }
        }
        if (median_gather(source, shift, low_digit, candidates, kept) != kept) {
            outcome = MEDIAN_CHANGED;
            break;
        }
        source = (median_source){candidates, kept, false};
        median_key_range(candidates, kept, &range);
        ranks[0] -= below;
        ranks[1] -= below;
    }
    free(candidates);
    free(counts);
    return outcome;
}

// Sort the `n` doubles at `v`, none of them NaN, into ascending order: an
// insertion sort, which on a few dozen elements is quicker than anything
// that needs more bookkeeping. Comparisons of numbers raise no flag NumPy
// reports.
static void median_sort(double *v, size_t n) {
    for (size_t i = 1; i < n; i++) {
        const double value = v[i];
        size_t j = i;
        for (; j > 0 && v[j - 1] > value; j--) v[j] = v[j - 1];
        v[j] = value;
    }
}

// True when NumPy's mean of the middle elements raises a flag it reports.
// `sum` is the sum of the two middle elements and `median` that divided by
// two; for an odd count nothing is computed and nothing can be raised. The
// sum overflows exactly when it is infinite and neither element is, and is
// invalid exactly when it is a NaN, the elements being numbers. The division
// underflows when its result is tiny and not exact; it is exact when twice
// the result, which doubling computes exactly at that size, is the sum. A
// result of exactly the smallest normal number counts as tiny, which takes
// no position on whether the processor detects tininess before rounding or
// after: such a call is NumPy's either way.
static bool median_raises(double low, double high, double sum, double median) {
    if (isnan(sum) || (isinf(sum) && !isinf(low) && !isinf(high))) return true;
    return fabs(median) <= DBL_MIN && median * 2.0 != sum;
}

// True when one of the `n` doubles at `v` is a negative zero.
static bool median_has_negative_zero(const double *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (v[i] == 0.0 && signbit(v[i])) return true;
    }
    return false;
}

// The two middle elements of the `n` doubles at `data`, 1 <= n <=
// MEDIAN_F64_SORT_MAX_N, into `middle`, by sorting a copy on the stack; for
// an odd count the middle element twice. False when one is a NaN. A
// signalling NaN raises the invalid flag in the test; the call is then
// NumPy's, whose ufuncs clear the flags before they compute. The lock stays
// held, as the array is short.
static bool median_middle_by_sort(const double *data, size_t n, double middle[2]) {
    if (hk_has_nan(data, n)) return false;
    double sorted[MEDIAN_F64_SORT_MAX_N];
    memcpy(sorted, data, n * sizeof sorted[0]);
    median_sort(sorted, n);
    middle[0] = sorted[(n - 1) / 2];
    middle[1] = sorted[n / 2];
    return true;
}

// The two middle elements of the `n` doubles at `data`, n >
// MEDIAN_F64_SORT_MAX_N, into `middle`, by median_select; for an odd count
// the middle element twice. False for a call that is NumPy's. The array is
// not copied first, and the interpreter lock is released from
// HK_NOGIL_MIN_N elements: median_select stays inside its own buffers
// whatever another thread writes to the array meanwhile.
static bool median_middle_by_select(const double *data, size_t n, double middle[2]) {
    size_t ranks[2] = {(n - 1) / 2, n / 2};
    uint64_t keys[2];
    median_outcome outcome;
    HK_BEGIN_NOGIL(n)
    outcome = median_select((const uint64_t *)data, n, ranks, keys);
    HK_END_NOGIL
    if (outcome != MEDIAN_SELECTED) return false;
    middle[0] = median_value(keys[0]);
    middle[1] = median_value(keys[1]);
    return true;
}

// np.median(x) for a contiguous float64 array: the caller has tested the
// type. Returns the np.float64 NumPy returns, or NULL with no exception set
// for a call that is NumPy's.
static HK_NOINLINE PyObject *try_median(PyArrayObject *x) {
    const npy_intp n = PyArray_SIZE(x);
    // NumPy warns and answers NaN for an empty array.
    if (n < 1 || n > MEDIAN_F64_MAX_N) return NULL;
    const double *data = PyArray_DATA(x);
    double middle[2];
    const bool found = n <= MEDIAN_F64_SORT_MAX_N ? median_middle_by_sort(data, (size_t)n, middle)
                                                  : median_middle_by_select(data, (size_t)n, middle);
    if (!found) return NULL;
    double median = middle[0];
    if (n % 2 == 0) {
        // The two operations of NumPy's mean, in its order.
        const double sum = middle[0] + middle[1];
        median = sum / 2.0;
        if (median_raises(middle[0], middle[1], sum, median)) return NULL;
        // The sum may have raised the precision flag, which NumPy's own loops
        // leave clear. The test above depends on the arithmetic, so this
        // cannot run before it. A call passed on leaves the flags to NumPy,
        // whose ufuncs clear them before they compute.
        hk_clear_float_flags();
    }
    // The array holds no NaN, so these comparisons raise no flag.
    if (median == 0.0 && median_has_negative_zero(data, (size_t)n)) return NULL;
    PyObject *result = hk_scalar(NPY_DOUBLE, (void *)&median);
    if (result != NULL) hk_take(HK_MEDIAN_F64);
    return result;
}

// True when the arguments of np.median after the array ask for the median of
// all its elements and for nothing else: none, or `axis` as None, by
// position or by keyword. NumPy flattens the array either way. Every other
// argument, and `a=` by keyword, is NumPy's even at its default. Astropy's
// biweight statistics and median absolute deviation call
// np.median(data, axis=None).
static inline bool median_arguments_are_plain(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    const Py_ssize_t keywords = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    if (nargs < 1 || nargs + keywords > 2) return false;
    if (nargs + keywords == 1) return true;
    return args[1] == Py_None && (keywords == 0 || hk_text_is(PyTuple_GET_ITEM(kwnames, 0), "axis"));
}

// np.median(x) and np.median(x, axis=None). The array's type is tested
// first, so that a call left to NumPy costs that test, at most a look at the
// arguments, and a count. Every call not taken is counted.
static PyObject *call_median(HK_VECTORCALL_PARAMETERS) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (nargs >= 1 && hk_is_flat(args[0], NPY_DOUBLE) && median_arguments_are_plain(args, nargs, kwnames)) {
        PyObject *result = try_median((PyArrayObject *)args[0]);
        if (result != NULL) return result;
    }
    hk_pass(HK_OP_MEDIAN);
    return functions[F_MEDIAN].original(callable, args, nargsf, kwnames);
}

// ------------------------------------------------------------------ unique

// The widest range of values np.unique's kernel is always given, whatever the
// element count: 256 values, whose table of int64 is the 2 KB a hook keeps on
// the stack. Above it a range is taken only while it is no wider than the
// array is long, so that the table costs at most about one more pass over as
// many bytes as the array has. See docs/unique-design.md.
#define UNIQUE_FREE_SPAN ((npy_intp)(HK_STACK_SCRATCH_BYTES / sizeof(int64_t)))

// The range scan of np.unique keeps UNIQUE_SCAN_VECTORS running minima and
// as many maxima, in vectors of the compiler's vector types: 16 bytes, NEON's
// width, on ARM64 and 32, AVX2's, on x86-64. Comparing 64-bit lanes and
// selecting is one cmgt and one bsl per two elements on ARM64, and one
// vpcmpgtq and one vpblendvb per four on x86-64. Several independent chains
// leave the loop bound by its loads rather than by the latency of one.
#if defined(__aarch64__)
#define UNIQUE_SCAN_VECTOR_BYTES 16
#else
#define UNIQUE_SCAN_VECTOR_BYTES 32
#endif
#define UNIQUE_SCAN_VECTORS 4
typedef int64_t unique_i64xv __attribute__((vector_size(UNIQUE_SCAN_VECTOR_BYTES)));
// Elements of one vector, and of the array one step of the scan reads.
#define UNIQUE_SCAN_LANES (UNIQUE_SCAN_VECTOR_BYTES / sizeof(int64_t))
#define UNIQUE_SCAN_STEP (UNIQUE_SCAN_VECTORS * UNIQUE_SCAN_LANES)

// The smallest and the largest of the `n` int64 elements at `x`, n >= 1,
// into `smallest` and `largest`. No branch on the data. The last step ends at
// n and overlaps the one before it, which a minimum and a maximum do not
// mind; fewer elements than a step are read one at a time. The vectors are
// loaded with memcpy, which the compiler turns into unaligned loads, and a
// lane is chosen with a mask, which it turns into a select (bsl, vpblendvb).
static HK_TARGET_KERNEL_LEVEL void unique_range(const int64_t *x, size_t n, int64_t *smallest, int64_t *largest) {
    int64_t lo = x[0], hi = x[0];
    if (n >= UNIQUE_SCAN_STEP) {
        unique_i64xv low[UNIQUE_SCAN_VECTORS], high[UNIQUE_SCAN_VECTORS];
        for (size_t k = 0; k < UNIQUE_SCAN_VECTORS; k++) {
            memcpy(&low[k], x + k * UNIQUE_SCAN_LANES, sizeof low[k]);
            high[k] = low[k];
        }
        for (size_t i = 0;; i += UNIQUE_SCAN_STEP) {
            if (i > n - UNIQUE_SCAN_STEP) i = n - UNIQUE_SCAN_STEP;
            for (size_t k = 0; k < UNIQUE_SCAN_VECTORS; k++) {
                unique_i64xv v;
                memcpy(&v, x + i + k * UNIQUE_SCAN_LANES, sizeof v);
                const unique_i64xv below = (unique_i64xv)(v < low[k]), above = (unique_i64xv)(v > high[k]);
                low[k] = (v & below) | (low[k] & ~below);
                high[k] = (v & above) | (high[k] & ~above);
            }
            if (i == n - UNIQUE_SCAN_STEP) break;
        }
        for (size_t k = 0; k < UNIQUE_SCAN_VECTORS; k++) {
            for (size_t j = 0; j < UNIQUE_SCAN_LANES; j++) {
                lo = low[k][j] < lo ? low[k][j] : lo;
                hi = high[k][j] > hi ? high[k][j] : hi;
            }
        }
    } else {
        for (size_t i = 1; i < n; i++) {
            lo = x[i] < lo ? x[i] : lo;
            hi = x[i] > hi ? x[i] : hi;
        }
    }
    *smallest = lo;
    *largest = hi;
}

// np.unique(x) for an int64 array `x`, of any shape and at least one element,
// C-contiguous: the unique_i64 kernel. Returns a new reference, or NULL with
// no exception set for a call that is NumPy's.
//
// For an integer array every NumPy release from 2.3 answers this call with a
// new one-dimensional array of the array's element type holding each
// distinct element once, in ascending order: the hash path of _unique1d
// sorts its result, and the sort path finds it in order. The array is read
// once for its smallest and largest element. When the range between them is
// narrow (UNIQUE_FREE_SPAN), the kernel marks the values present in a buffer
// of one int64 per value of the range and writes the distinct values to its
// front; they are copied into the result. A wide range is NumPy's, which then
// costs it one read of the array more.
//
// The lock is released for the scan and the kernel from HK_NOGIL_MIN_N
// elements. The kernel ignores elements outside the range it is given, so a
// thread that writes the array between the two reads can change the values
// found, never make the kernel write outside its buffer.
static HK_NOINLINE PyObject *unique_values(PyArrayObject *x) {
    const npy_intp n = PyArray_SIZE(x);
    const int64_t *data = PyArray_DATA(x);
    int64_t smallest, largest;
    {
        HK_BEGIN_NOGIL(n)
        unique_range(data, (size_t)n, &smallest, &largest);
        HK_END_NOGIL
    }
    // The width of the range less one, in unsigned arithmetic, where it
    // cannot overflow: at most 2^64 - 1.
    const uint64_t width = (uint64_t)largest - (uint64_t)smallest;
    const npy_intp limit = n > UNIQUE_FREE_SPAN ? n : UNIQUE_FREE_SPAN;
    if (width >= (uint64_t)limit) return NULL;
    const npy_intp span = (npy_intp)width + 1;

    int64_t on_stack[UNIQUE_FREE_SPAN];
    int64_t *table = span <= UNIQUE_FREE_SPAN ? on_stack : malloc((size_t)span * sizeof(int64_t));
    if (table == NULL) return NULL;
    npy_intp count;
    {
        HK_BEGIN_NOGIL(n)
        count = (npy_intp)unique_i64(data, table, (size_t)n, smallest, (size_t)span);
        HK_END_NOGIL
    }
    PyObject *result = PyArray_SimpleNew(1, &count, PyArray_TYPE(x));
    if (result != NULL) memcpy(PyArray_DATA((PyArrayObject *)result), table, (size_t)count * sizeof(int64_t));
    if (table != on_stack) free(table);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    hk_take(HK_UNIQUE_I64);
    return result;
}

// True when a flag of np.unique was not given or is False itself. Anything
// else, 0 and None included, is NumPy's to interpret.
static inline bool unique_flag_off(PyObject *flag) {
    return flag == NULL || flag == Py_False;
}

// np.unique(ar) for an int64 ndarray `ar` of at least one dimension and one
// element, with return_index, return_inverse and return_counts absent or
// False, axis absent or None, equal_nan absent or a bool (it means nothing
// for integers), and sorted absent or True. The parameters are those of NumPy
// 2.3 to 2.5: five by position, equal_nan and sorted by keyword only.
// sorted=False is NumPy's: its order is that of NumPy's hash set.
//
// An array of any layout is taken: one that is not C-contiguous is copied in
// C order first, as NumPy's own flatten() copies every array.
static HK_NOINLINE PyObject *try_unique(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    enum { AR, RETURN_INDEX, RETURN_INVERSE, RETURN_COUNTS, AXIS, EQUAL_NAN, SORTED, PARAMETERS };
    static const char *const names[PARAMETERS] = {
        "ar", "return_index", "return_inverse", "return_counts", "axis", "equal_nan", "sorted",
    };
    // Parameters that may be given by position: all but the two keyword-only ones.
    const Py_ssize_t positional = EQUAL_NAN;
    PyObject *bound[PARAMETERS];
    if (!hk_bind_arguments(args, nargs, kwnames, names, PARAMETERS, positional, bound)) return NULL;
    PyObject *array = bound[AR];
    if (array == NULL || !hk_is_plain_array(array)) return NULL;
    PyArrayObject *x = (PyArrayObject *)array;
    if (PyArray_TYPE(x) != NPY_INT64 || !PyArray_ISNOTSWAPPED(x) || PyArray_NDIM(x) < 1 || PyArray_SIZE(x) < 1) {
        return NULL;
    }
    if (!unique_flag_off(bound[RETURN_INDEX]) || !unique_flag_off(bound[RETURN_INVERSE]) ||
        !unique_flag_off(bound[RETURN_COUNTS]) || !hk_is_absent(bound[AXIS])) {
        return NULL;
    }
    if (bound[EQUAL_NAN] != NULL && !PyBool_Check(bound[EQUAL_NAN])) return NULL;
    if (bound[SORTED] != NULL && bound[SORTED] != Py_True) return NULL;
    if (PyArray_IS_C_CONTIGUOUS(x)) return unique_values(x);
    PyObject *copy = PyArray_NewCopy(x, NPY_CORDER);
    if (copy == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyObject *result = unique_values((PyArrayObject *)copy);
    Py_DECREF(copy);
    return result;
}

// np.unique: the kernel when the call meets its conditions; otherwise counted
// and passed to NumPy unchanged.
static PyObject *call_unique(HK_VECTORCALL_PARAMETERS) {
    PyObject *result = try_unique(args, PyVectorcall_NARGS(nargsf), kwnames);
    if (result != NULL) return result;
    hk_pass(HK_OP_UNIQUE);
    return functions[F_UNIQUE].original(callable, args, nargsf, kwnames);
}

// ----------------------------------------------------------------- the hook

static hk_function functions[F_COUNT] = {
    [F_ARGMAX] = {"argmax", call_argmax, NULL, NULL, NULL},
    [F_ARGMIN] = {"argmin", call_argmin, NULL, NULL, NULL},
    [F_COUNT_NONZERO] = {"count_nonzero", call_count_nonzero, NULL, NULL, NULL},
    [F_SUM] = {"sum", call_sum, NULL, NULL, NULL},
    [F_HISTOGRAM] = {"histogram", call_histogram, NULL, NULL, NULL},
    [F_BINCOUNT] = {"bincount", call_bincount, NULL, NULL, NULL},
    [F_ISIN] = {"isin", call_isin, NULL, NULL, NULL},
    [F_INTERP] = {"interp", call_interp, NULL, NULL, NULL},
    [F_COMPRESS] = {"compress", call_compress, NULL, NULL, NULL},
    [F_EXTRACT] = {"extract", call_extract, NULL, NULL, NULL},
    [F_PUTMASK] = {"putmask", call_putmask, NULL, NULL, NULL},
    [F_COPYTO] = {"copyto", call_copyto, NULL, NULL, NULL},
    [F_WHERE] = {"where", call_where, NULL, NULL, NULL},
    [F_CLIP] = {"clip", call_clip, prepare_clip, NULL, NULL},
    [F_ANY] = {"any", call_any, NULL, NULL, NULL},
    [F_ROUND] = {"round", call_round, NULL, NULL, NULL},
    [F_AROUND] = {"around", call_around, NULL, NULL, NULL},
    [F_MEDIAN] = {"median", call_median, NULL, NULL, NULL},
    [F_UNIQUE] = {"unique", call_unique, NULL, NULL, NULL},
};

// Return the address of the call pointer inside a callable object, or NULL
// with an exception set if its type does not use the vectorcall protocol.
static vectorcallfunc *call_pointer(PyObject *object) {
    PyTypeObject *type = Py_TYPE(object);
    if (!PyType_HasFeature(type, Py_TPFLAGS_HAVE_VECTORCALL) || type->tp_vectorcall_offset <= 0) {
        PyErr_Format(PyExc_TypeError, "%s objects have no call pointer to hook", type->tp_name);
        return NULL;
    }
    return (vectorcallfunc *)((char *)object + type->tp_vectorcall_offset);
}

// Point the function at the binding. See hk_hook.apply.
static int function_apply(int index) {
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) return -1;
    PyObject *object = PyObject_GetAttrString(numpy, functions[index].attribute);
    Py_DECREF(numpy);
    if (object == NULL) return -1;
    vectorcallfunc *pointer = call_pointer(object);
    if (pointer == NULL || *pointer == NULL) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "the function has no call pointer set");
        Py_DECREF(object);
        return -1;
    }
    if (functions[index].prepare != NULL) functions[index].prepare(object);
    functions[index].object = object;
    functions[index].original = *pointer;
    *pointer = functions[index].wrapper;
    return 0;
}

// Put NumPy's call pointer back. See hk_hook.remove.
static int function_remove(int index) {
    PyObject *object = functions[index].object;
    if (object == NULL) return 0;
    vectorcallfunc *pointer = call_pointer(object);
    if (pointer == NULL) {
        PyErr_Clear();
        return 0;
    }
    if (*pointer != functions[index].wrapper) return 0;
    *pointer = functions[index].original;
    functions[index].object = NULL;
    Py_DECREF(object);
    return 1;
}

hk_hook hk_function_hooks[] = {
    {"function.argmax", function_apply, function_remove, F_ARGMAX, false},
    {"function.argmin", function_apply, function_remove, F_ARGMIN, false},
    {"function.count_nonzero", function_apply, function_remove, F_COUNT_NONZERO, false},
    {"function.sum", function_apply, function_remove, F_SUM, false},
    {"function.histogram", function_apply, function_remove, F_HISTOGRAM, false},
    {"function.bincount", function_apply, function_remove, F_BINCOUNT, false},
    {"function.isin", function_apply, function_remove, F_ISIN, false},
    {"function.interp", function_apply, function_remove, F_INTERP, false},
    {"function.compress", function_apply, function_remove, F_COMPRESS, false},
    {"function.extract", function_apply, function_remove, F_EXTRACT, false},
    {"function.putmask", function_apply, function_remove, F_PUTMASK, false},
    {"function.copyto", function_apply, function_remove, F_COPYTO, false},
    {"function.where", function_apply, function_remove, F_WHERE, false},
    {"function.clip", function_apply, function_remove, F_CLIP, false},
    {"function.any", function_apply, function_remove, F_ANY, false},
    {"function.round", function_apply, function_remove, F_ROUND, false},
    {"function.around", function_apply, function_remove, F_AROUND, false},
    {"function.median", function_apply, function_remove, F_MEDIAN, false},
    {"function.unique", function_apply, function_remove, F_UNIQUE, false},
    {NULL, NULL, NULL, 0, false},
};
