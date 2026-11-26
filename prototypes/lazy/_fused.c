/* Calls the lazy layer's fused kernels on NumPy arrays.
 *
 * A fused kernel (prototypes/lazy/kernels/) is a leaf function over
 * contiguous runs of float64: `f(v0, ..., out, n, s0, ...)`. This module
 * stands between it and an expression on arrays of any shape. It checks that
 * the arrays are ones the kernel can read, walks the outer dimensions, and
 * calls the kernel once per contiguous run along the innermost ones. It
 * decides nothing about what is fused: `fuse.py` does, and hands over a
 * kernel's place in the table with the operands in the kernel's order.
 *
 * Two entry points:
 *
 *   elementwise(kernel, operands, out) -> flags, or DECLINED
 *   reduce(kernel, operands)           -> (value, flags), or DECLINED
 *
 * An operand is a float64 array, or a Python float for a number. `flags`
 * are the floating-point exceptions the kernel raised, as the FE_* bits of
 * <fenv.h>: the caller compares them with the `np.errstate` the operations
 * were written under, and computes the expression with NumPy's own calls
 * when NumPy would have warned or raised. DECLINED means the arrays are
 * laid out in a way this module does not walk; the caller does the same.
 *
 * The table of kernels is generated from their specs by `build_fused.py`,
 * which also assembles them into this module.
 */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <fenv.h>
#include <stdint.h>
#include <string.h>
#include <numpy/arrayobject.h>

/* The most dimensions an expression may have here. More are declined. */
#define MOST_DIMENSIONS 8
/* A kernel takes its pointers and its length in integer registers and its
 * numbers in floating-point ones. Both ARM64 and x86-64 pass eight of each
 * kind in registers or, for integers past the sixth on x86-64, on the stack
 * in order, so one call with eight of each reaches any kernel: the kernel
 * reads the ones it has and the rest are ignored. */
#define MOST_INTEGERS 8
#define MOST_NUMBERS 8
/* Bytes in a float64. */
#define ELEMENT 8
/* Elements a kernel computes in one step of its loop: it loads these from every array before it stores them. */
#define BLOCK 8
/* What the entry points return when they will not make the call. */
#define DECLINED (-1)
/* The exceptions a caller is told of. Inexact is not one: nearly every operation raises it. */
#define REPORTED (FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW)

typedef void (*elementwise_kernel)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                                   double, double, double, double, double, double, double, double);
typedef double (*reduction_kernel)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                                   double, double, double, double, double, double, double, double);

/* One row of the table: a kernel and how many arrays and numbers it reads. */
typedef struct {
    const char *name;
    void (*function)(void);
    int vectors;
    int scalars;
    int reduces; /* 1 for a kernel that returns one number, 0 for one that fills `out`. */
} fused_kernel;

#include "_fused_kernels.h" /* KERNELS and KERNEL_COUNT, generated. */

/* One operand as the walk sees it: where it is now, and how far it moves per step along each dimension. */
typedef struct {
    char *data;
    npy_intp strides[MOST_DIMENSIONS];
    int is_array; /* 0 for a number, whose value is in `number`. */
    double number;
} operand;

/* Reads an operand's layout against the shape walked.
 *
 * Args:
 *   object: A float64 array, or a Python float.
 *   dimensions, shape: The shape the expression runs over.
 *   vector: 1 for an operand the kernel reads along the innermost dimension, 0 for a number.
 *   read: Filled in.
 *
 * Returns 1 when the operand can be walked, 0 when the call is to be declined.
 * An array is taken when it is a plain `ndarray` of native aligned float64
 * that broadcasts to the shape: each of its dimensions is the shape's or 1.
 * Its strides must be positive and must not grow from one dimension to the
 * next: then NumPy's own result for it is laid out in C order, as `out` is.
 * A vector moves by one element along the innermost dimension; a number
 * that is an array has length 1 there.
 */
static int
read_operand(PyObject *object, int dimensions, const npy_intp *shape, int vector, operand *read)
{
    if (PyFloat_CheckExact(object)) {
        if (vector) {
            return 0;
        }
        read->is_array = 0;
        read->number = PyFloat_AS_DOUBLE(object);
        return 1;
    }
    if (!PyArray_CheckExact(object)) {
        return 0;
    }
    PyArrayObject *array = (PyArrayObject *)object;
    if (PyArray_TYPE(array) != NPY_DOUBLE || !PyArray_ISBEHAVED_RO(array)) {
        return 0;
    }
    int own = PyArray_NDIM(array);
    if (own > dimensions) {
        return 0;
    }
    read->is_array = 1;
    read->data = PyArray_BYTES(array);
    npy_intp previous = NPY_MAX_INTP;
    for (int axis = 0; axis < dimensions; axis++) {
        int theirs = axis - (dimensions - own);
        npy_intp length = theirs < 0 ? 1 : PyArray_DIM(array, theirs);
        if (length == 1) {
            read->strides[axis] = 0; /* Broadcast, or a dimension with nothing to step over. */
            continue;
        }
        if (length != shape[axis]) {
            return 0;
        }
        npy_intp stride = PyArray_STRIDE(array, theirs);
        if (stride <= 0 || stride > previous) {
            return 0;
        }
        previous = stride;
        read->strides[axis] = stride;
    }
    npy_intp inner = read->strides[dimensions - 1];
    if (vector) {
        return inner == ELEMENT || shape[dimensions - 1] == 1;
    }
    return inner == 0;
}

/* Says whether the last outer dimension can be walked as part of the inner run.
 *
 * It can when stepping once along it lands where the inner run ended: for a
 * vector, a stride of the run's length in bytes; for a number, no stride at
 * all, so it is the same number for the whole longer run.
 */
static int
joins(const operand *read, int vector, int axis, npy_intp run)
{
    if (!read->is_array) {
        return 1;
    }
    return read->strides[axis] == (vector ? run * ELEMENT : 0);
}

/* The first and the last byte an operand's walk touches, for an operand `read_operand` took: its strides are
 * positive or zero, so the walk starts at `data` and ends at the last element of the last step. */
static void
extent(const operand *read, int dimensions, const npy_intp *shape, const char **low, const char **high)
{
    npy_intp span = 0;
    for (int axis = 0; axis < dimensions; axis++) {
        span += read->strides[axis] * (shape[axis] - 1);
    }
    *low = read->data;
    *high = read->data + span + ELEMENT;
}

/* elementwise(kernel, operands, out) -> flags, or DECLINED.
 *
 * Args:
 *   kernel: The kernel's place in the table.
 *   operands: A tuple, the kernel's arrays and then its numbers, in the order of its arguments.
 *   out: A writable float64 array of the expression's shape. It is filled.
 *
 * `out` may be one of the operands exactly, element for element: `x -= s * y`
 * made where `x` is. A kernel loads the eight elements of a block from every
 * array before it stores the block, so a block may be written where it was
 * read. Only its last block is different: for a length that is no multiple
 * of eight it is a block that ends with the array and overlaps the one
 * before, which would then be read after it was written. So with such an
 * operand each run is made as two calls: the whole blocks, and then the
 * fewer than eight elements left, which every kernel computes one at a time.
 * An operand that overlaps `out` in any other way is declined.
 */
static PyObject *
fused_elementwise(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 3 || !PyLong_CheckExact(args[0]) || !PyTuple_CheckExact(args[1]) || !PyArray_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "elementwise(kernel, operands, out)");
        return NULL;
    }
    long index = PyLong_AsLong(args[0]);
    if (index < 0 || index >= KERNEL_COUNT || KERNELS[index].reduces) {
        PyErr_SetString(PyExc_ValueError, "no such elementwise kernel");
        return NULL;
    }
    const fused_kernel *kernel = &KERNELS[index];
    PyArrayObject *out = (PyArrayObject *)args[2];
    int dimensions = PyArray_NDIM(out);
    int vectors = kernel->vectors, scalars = kernel->scalars;
    if (PyTuple_GET_SIZE(args[1]) != vectors + scalars) {
        PyErr_SetString(PyExc_ValueError, "the kernel takes another number of operands");
        return NULL;
    }
    if (dimensions < 1 || dimensions > MOST_DIMENSIONS || vectors + 2 > MOST_INTEGERS || scalars > MOST_NUMBERS
        || PyArray_TYPE(out) != NPY_DOUBLE || !PyArray_ISBEHAVED(out)) {
        return PyLong_FromLong(DECLINED);
    }
    npy_intp shape[MOST_DIMENSIONS];
    for (int axis = 0; axis < dimensions; axis++) {
        shape[axis] = PyArray_DIM(out, axis);
        if (shape[axis] == 0) {
            return PyLong_FromLong(DECLINED);
        }
    }
    /* The operands, the kernel's arrays first, then `out`, then its numbers. */
    operand reads[MOST_INTEGERS + MOST_NUMBERS];
    for (int place = 0; place < vectors; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[1], place), dimensions, shape, 1, &reads[place])) {
            return PyLong_FromLong(DECLINED);
        }
    }
    operand *target = &reads[vectors];
    if (!read_operand((PyObject *)out, dimensions, shape, 1, target)) {
        return PyLong_FromLong(DECLINED);
    }
    operand *numbers = &reads[vectors + 1];
    for (int place = 0; place < scalars; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[1], vectors + place), dimensions, shape, 0, &numbers[place])) {
            return PyLong_FromLong(DECLINED);
        }
    }
    /* Where `out` lies against each array read: apart from it, or the very same elements. */
    int in_place = 0;
    {
        const char *out_low, *out_high;
        extent(target, dimensions, shape, &out_low, &out_high);
        for (int place = 0; place < vectors + 1 + scalars; place++) {
            const operand *read = &reads[place];
            if (place == vectors || !read->is_array) {
                continue;
            }
            const char *low, *high;
            extent(read, dimensions, shape, &low, &high);
            if (low >= out_high || out_low >= high) {
                continue;
            }
            if (place > vectors || read->data != target->data
                || memcmp(read->strides, target->strides, (size_t)dimensions * sizeof(npy_intp)) != 0) {
                return PyLong_FromLong(DECLINED);
            }
            in_place = 1;
        }
    }
    /* Lengthen the inner run by every outer dimension that every operand walks straight through. */
    int outer = dimensions - 1;
    npy_intp run = shape[outer];
    while (outer > 0) {
        int axis = outer - 1, joined = 1;
        if (shape[axis] != 1) {
            for (int place = 0; place <= vectors && joined; place++) {
                joined = joins(&reads[place], 1, axis, run);
            }
            for (int place = 0; place < scalars && joined; place++) {
                joined = joins(&numbers[place], 0, axis, run);
            }
        }
        if (!joined) {
            break;
        }
        run *= shape[axis];
        outer = axis;
    }
    elementwise_kernel call = (elementwise_kernel)kernel->function;
    uintptr_t integers[MOST_INTEGERS] = {0};
    double floats[MOST_NUMBERS] = {0};
    /* The lengths of the calls of one run: the run, or its whole blocks and then the elements left over. */
    npy_intp whole = in_place ? run - run % BLOCK : run;
    npy_intp lengths[2] = {whole, run - whole};
    npy_intp position[MOST_DIMENSIONS] = {0};
    feclearexcept(FE_ALL_EXCEPT);
    for (;;) {
        for (int place = 0; place < scalars; place++) {
            floats[place] = numbers[place].is_array ? *(const double *)numbers[place].data : numbers[place].number;
        }
        npy_intp done = 0;
        for (int part = 0; part < 2; part++) {
            if (lengths[part] == 0) {
                continue;
            }
            for (int place = 0; place <= vectors; place++) {
                /* An array of one element along the run is read at that element by every call. */
                npy_intp step = reads[place].strides[dimensions - 1];
                integers[place] = (uintptr_t)(reads[place].data + done * step);
            }
            integers[vectors + 1] = (uintptr_t)lengths[part];
            call(integers[0], integers[1], integers[2], integers[3], integers[4], integers[5], integers[6], integers[7],
                 floats[0], floats[1], floats[2], floats[3], floats[4], floats[5], floats[6], floats[7]);
            done += lengths[part];
        }
        /* Step the outer dimensions as an odometer: the last one first, carrying into the one before. */
        int axis = outer - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                for (int place = 0; place <= vectors + scalars; place++) {
                    if (reads[place].is_array) {
                        reads[place].data += reads[place].strides[axis];
                    }
                }
                break;
            }
            /* This dimension is done: go back to its start, and step the one before. */
            for (int place = 0; place <= vectors + scalars; place++) {
                if (reads[place].is_array) {
                    reads[place].data -= reads[place].strides[axis] * (shape[axis] - 1);
                }
            }
            position[axis] = 0;
        }
        if (axis < 0) {
            break;
        }
    }
    int raised = fetestexcept(REPORTED);
    if (raised) {
        feclearexcept(FE_ALL_EXCEPT); /* So that nothing that runs next finds them and takes them for its own. */
    }
    return PyLong_FromLong(raised);
}

/* reduce(kernel, operands) -> (value, flags), or DECLINED.
 *
 * Args:
 *   kernel: The kernel's place in the table.
 *   operands: A tuple, the kernel's arrays and then its numbers, in the order of its arguments.
 *
 * NumPy adds the elements of a C-contiguous array in the order they lie in
 * memory, whatever its shape, and a reduction kernel adds in that order. So
 * every array has to be C-contiguous and of one shape, and every number a
 * Python float; anything else is declined.
 */
static PyObject *
fused_reduce(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 2 || !PyLong_CheckExact(args[0]) || !PyTuple_CheckExact(args[1])) {
        PyErr_SetString(PyExc_TypeError, "reduce(kernel, operands)");
        return NULL;
    }
    long index = PyLong_AsLong(args[0]);
    if (index < 0 || index >= KERNEL_COUNT || !KERNELS[index].reduces) {
        PyErr_SetString(PyExc_ValueError, "no such reduction kernel");
        return NULL;
    }
    const fused_kernel *kernel = &KERNELS[index];
    int vectors = kernel->vectors, scalars = kernel->scalars;
    if (PyTuple_GET_SIZE(args[1]) != vectors + scalars) {
        PyErr_SetString(PyExc_ValueError, "the kernel takes another number of operands");
        return NULL;
    }
    if (vectors < 1 || vectors + 1 > MOST_INTEGERS || scalars > MOST_NUMBERS) {
        return PyLong_FromLong(DECLINED);
    }
    uintptr_t integers[MOST_INTEGERS] = {0};
    double floats[MOST_NUMBERS] = {0};
    PyArrayObject *first = NULL;
    for (int place = 0; place < vectors; place++) {
        PyObject *object = PyTuple_GET_ITEM(args[1], place);
        if (!PyArray_CheckExact(object)) {
            return PyLong_FromLong(DECLINED);
        }
        PyArrayObject *array = (PyArrayObject *)object;
        if (PyArray_TYPE(array) != NPY_DOUBLE || !PyArray_ISBEHAVED_RO(array) || !PyArray_IS_C_CONTIGUOUS(array)) {
            return PyLong_FromLong(DECLINED);
        }
        if (first == NULL) {
            first = array;
        } else if (!PyArray_SAMESHAPE(array, first)) {
            return PyLong_FromLong(DECLINED);
        }
        integers[place] = (uintptr_t)PyArray_BYTES(array);
    }
    for (int place = 0; place < scalars; place++) {
        PyObject *object = PyTuple_GET_ITEM(args[1], vectors + place);
        if (!PyFloat_CheckExact(object)) {
            return PyLong_FromLong(DECLINED);
        }
        floats[place] = PyFloat_AS_DOUBLE(object);
    }
    integers[vectors] = (uintptr_t)PyArray_SIZE(first);
    reduction_kernel call = (reduction_kernel)kernel->function;
    feclearexcept(FE_ALL_EXCEPT);
    double value = call(integers[0], integers[1], integers[2], integers[3], integers[4], integers[5], integers[6],
                        integers[7], floats[0], floats[1], floats[2], floats[3], floats[4], floats[5], floats[6],
                        floats[7]);
    int raised = fetestexcept(REPORTED);
    if (raised) {
        feclearexcept(FE_ALL_EXCEPT);
    }
    return Py_BuildValue("(di)", value, raised);
}

/* kernels() -> a tuple of (name, arrays, numbers, reduces), one per row of the table, in its order. */
static PyObject *
fused_kernels(PyObject *module, PyObject *unused)
{
    PyObject *rows = PyTuple_New(KERNEL_COUNT);
    if (rows == NULL) {
        return NULL;
    }
    for (int index = 0; index < KERNEL_COUNT; index++) {
        PyObject *row = Py_BuildValue("(siii)", KERNELS[index].name, KERNELS[index].vectors, KERNELS[index].scalars,
                                      KERNELS[index].reduces);
        if (row == NULL) {
            Py_DECREF(rows);
            return NULL;
        }
        PyTuple_SET_ITEM(rows, index, row);
    }
    return rows;
}

static PyMethodDef methods[] = {
    {"elementwise", (PyCFunction)(void (*)(void))fused_elementwise, METH_FASTCALL,
     "elementwise(kernel, operands, out) -> the exceptions raised, or DECLINED"},
    {"reduce", (PyCFunction)(void (*)(void))fused_reduce, METH_FASTCALL,
     "reduce(kernel, operands) -> (value, the exceptions raised), or DECLINED"},
    {"kernels", fused_kernels, METH_NOARGS, "kernels() -> ((name, arrays, numbers, reduces), ...)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "_fused", "Calls the lazy layer's fused kernels on NumPy arrays.", -1, methods,
};

PyMODINIT_FUNC
PyInit__fused(void)
{
    import_array();
    PyObject *module = PyModule_Create(&definition);
    if (module == NULL) {
        return NULL;
    }
    /* What `fuse.py` compares the answers with. */
    if (PyModule_AddIntConstant(module, "DECLINED", DECLINED) < 0
        || PyModule_AddIntConstant(module, "INVALID", FE_INVALID) < 0
        || PyModule_AddIntConstant(module, "DIVIDE", FE_DIVBYZERO) < 0
        || PyModule_AddIntConstant(module, "OVER", FE_OVERFLOW) < 0
        || PyModule_AddIntConstant(module, "UNDER", FE_UNDERFLOW) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
