// Indexing hook: the two function pointers behind `a[i]` and `a[i] = v`.
//
// CPython runs `a[i]` by calling a pointer in the type of `a`
// (tp_as_mapping->mp_subscript), and `a[i] = v` by calling its neighbour
// (mp_ass_subscript). For ndarray both point into NumPy. They are the only
// way to reach `x[mask]` and `x[mask] = v`, which are neither functions nor
// methods.
//
// This hook sits on every index operation on every plain ndarray in the
// process, so the common case, an index that is not a boolean array, must
// leave after one comparison. Subclasses of ndarray defined in Python do not
// come through here: CPython gives them their own pointers.
#include "binding.h"

// NumPy's own functions, kept while the hooks are applied.
static binaryfunc original_get;
static objobjargproc original_set;

// True when `index` is a plain boolean ndarray. Tested first, on every index
// operation: for an integer or a slice it fails on the first comparison.
static inline bool is_mask(PyObject *index) {
    return hk_is_plain_array(index) && PyArray_TYPE((PyArrayObject *)index) == NPY_BOOL;
}

// The compress kernels on a buffer of the binding's own. See binding.h.
// The kernels move elements by their bits and never compute with one, so
// each serves every element type of its width: int64 elements go to the
// float64 kernel and int32 elements to the float32 one, counted as their
// calls; int8 and bool to compress_u8, uint16 to compress_i16.
// As with nonzero, the kernel's output is scratch, so it runs on a buffer of
// n elements and the selected elements are copied into an array of exactly
// their number. Shared with the function hooks for np.compress and
// np.extract.
HK_NOINLINE PyObject *hk_try_compress(PyObject *array, PyObject *selector) {
    hk_kernel kernel;
    int type_num;
    if (hk_is_flat(array, NPY_DOUBLE)) kernel = HK_COMPRESS_F64, type_num = NPY_DOUBLE;
    else if (hk_is_flat(array, NPY_FLOAT)) kernel = HK_COMPRESS_F32, type_num = NPY_FLOAT;
    else if (hk_is_flat(array, NPY_INT64)) kernel = HK_COMPRESS_F64, type_num = NPY_INT64;
    else if (hk_is_flat(array, NPY_INT32)) kernel = HK_COMPRESS_F32, type_num = NPY_INT32;
    else if (hk_is_flat(array, NPY_UINT8)) kernel = HK_COMPRESS_U8, type_num = NPY_UINT8;
    else if (hk_is_flat(array, NPY_INT8)) kernel = HK_COMPRESS_U8, type_num = NPY_INT8;
    else if (hk_is_flat(array, NPY_BOOL)) kernel = HK_COMPRESS_U8, type_num = NPY_BOOL;
    else if (hk_is_flat(array, NPY_INT16)) kernel = HK_COMPRESS_I16, type_num = NPY_INT16;
    else if (hk_is_flat(array, NPY_UINT16)) kernel = HK_COMPRESS_I16, type_num = NPY_UINT16;
    else return NULL;
    if (!hk_is_flat(selector, NPY_BOOL)) return NULL;
    PyArrayObject *x = (PyArrayObject *)array, *mask = (PyArrayObject *)selector;
    const npy_intp n = PyArray_SIZE(x);
    if (n != PyArray_SIZE(mask)) return NULL;

    const size_t element = (size_t)PyArray_ITEMSIZE(x);
    const size_t bytes = (size_t)n * element;
    double on_stack[HK_STACK_SCRATCH_BYTES / sizeof(double)];
    void *scratch = bytes <= sizeof on_stack ? (void *)on_stack : malloc(bytes);
    if (scratch == NULL) return NULL;

    npy_intp count = -1;
    {
        HK_BEGIN_NOGIL(n)
        if (hk_mask_is_clean(PyArray_DATA(mask), (size_t)n)) {
            const bool *selected = PyArray_DATA(mask);
            switch (kernel) {
            case HK_COMPRESS_F64: count = (npy_intp)compress_f64(PyArray_DATA(x), selected, scratch, (size_t)n); break;
            case HK_COMPRESS_F32: count = (npy_intp)compress_f32(PyArray_DATA(x), selected, scratch, (size_t)n); break;
            case HK_COMPRESS_I16: count = (npy_intp)compress_i16(PyArray_DATA(x), selected, scratch, (size_t)n); break;
            default: count = (npy_intp)compress_u8(PyArray_DATA(x), selected, scratch, (size_t)n); break;
            }
        }
        HK_END_NOGIL
    }
    PyObject *result = count >= 0 ? PyArray_SimpleNew(1, &count, type_num) : NULL;
    if (result != NULL) memcpy(PyArray_DATA((PyArrayObject *)result), scratch, (size_t)count * element);
    if (scratch != (void *)on_stack) free(scratch);
    if (result == NULL) {
        PyErr_Clear();
        return NULL;
    }
    hk_take(kernel);
    return result;
}

// x[mask] for a boolean mask of the shape of `x`, of any number of
// dimensions: NumPy then reads both in C order and returns the selected
// elements as one dimension, which is what the kernel does with two
// contiguous buffers. A mask of fewer dimensions selects rows, and one of
// another shape is an error or a different selection; both are NumPy's.
static inline PyObject *try_compress(PyObject *self, PyObject *index) {
    if (!hk_same_shape((PyArrayObject *)self, (PyArrayObject *)index)) return NULL;
    return hk_try_compress(self, index);
}

// `a[mask]` with a boolean array as the index: take the kernel if the call
// meets its conditions, and count it and pass it on if not.
static HK_NOINLINE PyObject *get_by_mask(PyObject *self, PyObject *index) {
    if (hk_is_plain_array(self)) {
        PyObject *result = try_compress(self, index);
        if (result != NULL) return result;
        hk_pass(HK_OP_COMPRESS);
    }
    return original_get(self, index);
}

// The hooked `a[i]`. Anything but a boolean array index leaves at once.
static PyObject *index_get(PyObject *self, PyObject *index) {
    if (is_mask(index)) return get_by_mask(self, index);
    return original_get(self, index);
}

// The maskfill kernels. See binding.h. Shared with the function hooks for
// np.putmask and np.copyto.
//
// maskfill_f64 rewrites elements the mask does not select with their own
// bits. The interpreter lock stays held throughout, so no other Python thread
// can write one of them in between and have its write undone. maskfill_f32
// does not rewrite them on either architecture; it keeps the lock as well, so
// that the four element types behave alike.
//
// The kernels move bits and never compute with them, so int64 goes to the
// float64 kernel and int32 to the float32 one, each with the value's bits.
HK_NOINLINE bool hk_try_maskfill(PyObject *array, PyObject *selector, PyObject *value) {
    // The value's bits, as the kernel for x's type takes them.
    union {
        double f64;
        int64_t i64;
        float f32;
        int32_t i32;
    } fill;
    hk_kernel kernel;
    size_t element;
    if (hk_is_flat(array, NPY_DOUBLE)) {
        if (!hk_as_double(value, &fill.f64)) return false;
        kernel = HK_MASKFILL_F64, element = sizeof(double);
    } else if (hk_is_flat(array, NPY_FLOAT)) {
        if (!hk_as_float32(value, &fill.f32)) return false;
        kernel = HK_MASKFILL_F32, element = sizeof(float);
    } else if (hk_is_flat(array, NPY_INT64)) {
        if (!hk_as_int64(value, &fill.i64)) return false;
        kernel = HK_MASKFILL_F64, element = sizeof(int64_t);
    } else if (hk_is_flat(array, NPY_INT32)) {
        if (!hk_as_int32(value, &fill.i32)) return false;
        kernel = HK_MASKFILL_F32, element = sizeof(int32_t);
    } else {
        return false;
    }
    if (!hk_is_flat(selector, NPY_BOOL)) return false;
    PyArrayObject *x = (PyArrayObject *)array, *mask = (PyArrayObject *)selector;
    const npy_intp n = PyArray_SIZE(x);
    if (n != PyArray_SIZE(mask) || !PyArray_ISWRITEABLE(x)) return false;
    // A boolean view of x's own memory would be read while it is written.
    if (!hk_disjoint(PyArray_DATA(x), (size_t)n * element, PyArray_DATA(mask), (size_t)n)) return false;
    if (!hk_mask_is_clean(PyArray_DATA(mask), (size_t)n)) return false;
    if (kernel == HK_MASKFILL_F64) maskfill_f64(PyArray_DATA(x), PyArray_DATA(mask), (size_t)n, fill.f64);
    else maskfill_f32(PyArray_DATA(x), PyArray_DATA(mask), (size_t)n, fill.f32);
    hk_take(kernel);
    return true;
}

// x[mask] = v for a boolean mask of the shape of `x`, of any number of
// dimensions. As for x[mask] above, any other mask is NumPy's.
static inline bool try_maskfill(PyObject *self, PyObject *index, PyObject *value) {
    if (!hk_same_shape((PyArrayObject *)self, (PyArrayObject *)index)) return false;
    return hk_try_maskfill(self, index, value);
}

// `a[mask] = v` with a boolean array as the index: take the kernel if the
// call meets its conditions, and count it and pass it on if not.
static HK_NOINLINE int set_by_mask(PyObject *self, PyObject *index, PyObject *value) {
    if (hk_is_plain_array(self)) {
        if (try_maskfill(self, index, value)) return 0;
        hk_pass(HK_OP_MASKFILL);
    }
    return original_set(self, index, value);
}

// The hooked `a[i] = v`. A NULL value is `del a[i]`, which is NumPy's to
// refuse. Anything but a boolean array index leaves at once.
static int index_set(PyObject *self, PyObject *index, PyObject *value) {
    if (value != NULL && is_mask(index)) return set_by_mask(self, index, value);
    return original_set(self, index, value);
}

// Point `a[i]` (index 0) or `a[i] = v` (index 1) at the binding. See
// hk_hook.apply.
//
// CPython reads these two pointers from the type on every index operation
// and caches nothing derived from them, so nothing has to be invalidated.
// PyType_Modified is deliberately not called: it would discard the type's
// cached attribute lookups, and CPython stops caching for a type altogether
// once it has been modified a thousand times.
static int index_apply(int index) {
    PyMappingMethods *mapping = PyArray_Type.tp_as_mapping;
    if (mapping == NULL || mapping->mp_subscript == NULL || mapping->mp_ass_subscript == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "ndarray has no indexing functions to hook");
        return -1;
    }
    if (index == 0) {
        original_get = mapping->mp_subscript;
        mapping->mp_subscript = index_get;
    } else {
        original_set = mapping->mp_ass_subscript;
        mapping->mp_ass_subscript = index_set;
    }
    return 0;
}

// Put NumPy's function back. See hk_hook.remove.
static int index_remove(int index) {
    PyMappingMethods *mapping = PyArray_Type.tp_as_mapping;
    if (index == 0) {
        if (mapping->mp_subscript != index_get) return 0;
        mapping->mp_subscript = original_get;
    } else {
        if (mapping->mp_ass_subscript != index_set) return 0;
        mapping->mp_ass_subscript = original_set;
    }
    return 1;
}

hk_hook hk_index_hooks[] = {
    {"index.get", index_apply, index_remove, 0, false},
    {"index.set", index_apply, index_remove, 1, false},
    {NULL, NULL, NULL, 0, false},
};
