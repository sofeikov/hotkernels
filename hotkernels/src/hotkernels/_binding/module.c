// The extension module `hotkernels._binding`: the hook table and the
// counters, exposed to Python. The policy lives in Python (which hooks to
// apply, the self-test after each, the report); this file only switches
// pointers and counts.
#define HK_DEFINE_NUMPY_API
#include "binding.h"

_Atomic uint64_t hk_taken[HK_KERNEL_COUNT];
_Atomic uint64_t hk_passed[HK_OPERATION_COUNT];
bool hk_inexact = false;
_Thread_local hk_argext_call hk_argext_mark;
PyTypeObject *hk_ndarray_type;

// Kernel names, in the order of hk_kernel.
static const char *const HK_KERNEL_NAMES[HK_KERNEL_COUNT] = {
    "argmax_f32", "argmax_f64", "argmin_f32", "argmin_f64",
    "clip_f32", "clip_f64",
    "compress_f32", "compress_f64", "maskfill_f64", "nonzero_bool",
    "cumsum_i64", "cumsum_f32", "cumsum_f64", "cumsum_bool",
    "cast_u8_f32", "sum_u8", "count_nonzero_i64", "count_nonzero_f64",
    "histogram_f64", "bincount_u8", "isin_i64",
    "searchsorted_f64", "searchsorted_right_f64", "interp_f64",
    "where_f64", "cummax_i64", "cumfmax_f64", "fmax_reduce_f64", "any_rows_bool",
    "bincount_i16", "bincount_i64", "unique_i64",
    "argsort_f64", "argsort_i64", "argsort_u64",
    "searchsorted_i64", "searchsorted_right_i64", "any_cols_bool", "sum_f32",
    // Served through bottleneck's functions.
    "nanmean_cols_f64", "nanstd_cols_f64", "nanmin_cols_f64", "nanmax_cols_f64",
    // Built into the binding.
    "clip_i64", "round_f64", "median_f64",
    "cumsum_rows_f64", "cumprod_rows_f64", "cummax_rows_f64", "cumfmax_rows_f64",
    "argmax_rows_f32", "argmax_rows_f64", "argmin_rows_f32", "argmin_rows_f64",
    "add_inplace_f64", "subtract_inplace_f64",
};

// Operation names, in the order of hk_operation.
static const char *const HK_OPERATION_NAMES[HK_OPERATION_COUNT] = {
    "argmax", "argmin", "clip", "compress", "maskfill",
    "nonzero", "cumsum", "astype", "sum", "count_nonzero",
    "histogram", "bincount", "isin", "searchsorted", "interp",
    "where", "accumulate", "reduceat", "any", "round", "median",
    "unique", "argsort",
    "bn.nanmean", "bn.nanstd", "bn.nanmin", "bn.nanmax",
};

// Every table of hooks, in the order hooks are listed to Python: deepest first.
static hk_hook *const HK_HOOK_TABLES[] = {
    hk_slot_hooks, hk_loop_hooks, hk_function_hooks, hk_method_hooks, hk_index_hooks, hk_bottleneck_hooks,
};
#define HK_HOOK_TABLE_COUNT (sizeof HK_HOOK_TABLES / sizeof HK_HOOK_TABLES[0])

// Find a hook by name. Returns NULL with KeyError set if there is none.
static hk_hook *find_hook(PyObject *name) {
    for (size_t t = 0; t < HK_HOOK_TABLE_COUNT; t++) {
        for (hk_hook *hook = HK_HOOK_TABLES[t]; hook->name != NULL; hook++) {
            if (PyUnicode_CompareWithASCIIString(name, hook->name) == 0) return hook;
        }
    }
    PyErr_SetObject(PyExc_KeyError, name);
    return NULL;
}

// hooks() -> tuple of every hook name, deepest hooks first.
static PyObject *binding_hooks(PyObject *module, PyObject *unused) {
    PyObject *names = PyList_New(0);
    if (names == NULL) return NULL;
    for (size_t t = 0; t < HK_HOOK_TABLE_COUNT; t++) {
        for (hk_hook *hook = HK_HOOK_TABLES[t]; hook->name != NULL; hook++) {
            PyObject *name = PyUnicode_FromString(hook->name);
            if (name == NULL || PyList_Append(names, name) != 0) {
                Py_XDECREF(name);
                Py_DECREF(names);
                return NULL;
            }
            Py_DECREF(name);
        }
    }
    PyObject *result = PyList_AsTuple(names);
    Py_DECREF(names);
    return result;
}

// apply(name): point NumPy at the binding for one hook. Raises if NumPy is
// not shaped as the hook expects; nothing is changed then. Applying a hook
// that is already applied does nothing.
static PyObject *binding_apply(PyObject *module, PyObject *name) {
    hk_hook *hook = find_hook(name);
    if (hook == NULL) return NULL;
    if (!hook->applied) {
        if (hook->apply(hook->index) != 0) return NULL;
        hook->applied = true;
    }
    Py_RETURN_NONE;
}

// remove(name) -> bool: put NumPy's own pointer back. False when the pointer
// no longer held the binding's function, in which case it was left alone.
// Removing a hook that is not applied returns True.
static PyObject *binding_remove(PyObject *module, PyObject *name) {
    hk_hook *hook = find_hook(name);
    if (hook == NULL) return NULL;
    if (!hook->applied) Py_RETURN_TRUE;
    const int restored = hook->remove(hook->index);
    hook->applied = false;
    return PyBool_FromLong(restored);
}

// applied(name) -> bool: whether a hook is in place.
static PyObject *binding_applied(PyObject *module, PyObject *name) {
    hk_hook *hook = find_hook(name);
    if (hook == NULL) return NULL;
    return PyBool_FromLong(hook->applied);
}

// Build a dict of name -> count from an array of counters.
static PyObject *counter_dict(const char *const *names, _Atomic uint64_t *counters, int count) {
    PyObject *result = PyDict_New();
    if (result == NULL) return NULL;
    for (int i = 0; i < count; i++) {
        PyObject *value = PyLong_FromUnsignedLongLong(atomic_load_explicit(&counters[i], memory_order_relaxed));
        if (value == NULL || PyDict_SetItemString(result, names[i], value) != 0) {
            Py_XDECREF(value);
            Py_DECREF(result);
            return NULL;
        }
        Py_DECREF(value);
    }
    return result;
}

// counters() -> (taken, passed): calls each kernel took, by kernel name, and
// calls each operation's hooks passed on to NumPy, by operation name.
static PyObject *binding_counters(PyObject *module, PyObject *unused) {
    PyObject *taken = counter_dict(HK_KERNEL_NAMES, hk_taken, HK_KERNEL_COUNT);
    PyObject *passed = counter_dict(HK_OPERATION_NAMES, hk_passed, HK_OPERATION_COUNT);
    PyObject *result = taken && passed ? PyTuple_Pack(2, taken, passed) : NULL;
    Py_XDECREF(taken);
    Py_XDECREF(passed);
    return result;
}

// reset_counters(): set every counter to zero.
static PyObject *binding_reset_counters(PyObject *module, PyObject *unused) {
    for (int i = 0; i < HK_KERNEL_COUNT; i++) atomic_store_explicit(&hk_taken[i], 0, memory_order_relaxed);
    for (int i = 0; i < HK_OPERATION_COUNT; i++) atomic_store_explicit(&hk_passed[i], 0, memory_order_relaxed);
    Py_RETURN_NONE;
}

// forget(kernel, calls): take `calls` off a kernel's counter, which then stays
// at zero or above. For the self-test of a hook applied after install() has
// reset the counters, so that its probe's calls are not reported as the
// program's.
static PyObject *binding_forget(PyObject *module, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "forget() takes a kernel name and a count");
        return NULL;
    }
    const unsigned long long calls = PyLong_AsUnsignedLongLong(args[1]);
    if (PyErr_Occurred()) return NULL;
    for (int i = 0; i < HK_KERNEL_COUNT; i++) {
        if (PyUnicode_CompareWithASCIIString(args[0], HK_KERNEL_NAMES[i]) == 0) {
            const uint64_t now = atomic_load_explicit(&hk_taken[i], memory_order_relaxed);
            atomic_store_explicit(&hk_taken[i], now > calls ? now - calls : 0, memory_order_relaxed);
            Py_RETURN_NONE;
        }
    }
    PyErr_SetObject(PyExc_KeyError, args[0]);
    return NULL;
}

// set_inexact(flag): allow or forbid kernels that are not bit-identical to NumPy.
static PyObject *binding_set_inexact(PyObject *module, PyObject *flag) {
    const int truth = PyObject_IsTrue(flag);
    if (truth < 0) return NULL;
    hk_inexact = truth != 0;
    Py_RETURN_NONE;
}

static PyMethodDef binding_methods[] = {
    {"hooks", binding_hooks, METH_NOARGS, "Return the name of every hook, deepest first."},
    {"apply", binding_apply, METH_O, "Point NumPy at the binding for one hook."},
    {"remove", binding_remove, METH_O, "Put NumPy's own pointer back for one hook."},
    {"applied", binding_applied, METH_O, "Return whether a hook is in place."},
    {"counters", binding_counters, METH_NOARGS, "Return (calls taken per kernel, calls passed on per operation)."},
    {"reset_counters", binding_reset_counters, METH_NOARGS, "Set every counter to zero."},
    {"forget", (PyCFunction)(void (*)(void))binding_forget, METH_FASTCALL, "Take calls off a kernel's counter."},
    {"set_inexact", binding_set_inexact, METH_O, "Allow or forbid kernels that are not bit-identical to NumPy."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef binding_module = {
    PyModuleDef_HEAD_INIT, "_binding", "Pointer switching and counters for hotkernels.", -1, binding_methods,
    NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__binding(void) {
    import_array();
    import_umath();
    hk_ndarray_type = &PyArray_Type;
    return PyModule_Create(&binding_module);
}
