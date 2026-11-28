/* Calls the lazy layer's fused kernels on NumPy arrays.
 *
 * A fused kernel (prototypes/lazy/kernels/) is a leaf function over
 * contiguous runs of float64: `f(v0, ..., out, n, s0, ...)`. This module
 * stands between it and an expression on arrays of any shape. It checks that
 * the arrays are ones the kernel can read, walks the outer dimensions, and
 * calls the kernel once per contiguous run along the innermost ones. Outer
 * and innermost are as the arrays lie in memory, not as their dimensions are
 * written: see `walk_order`. It
 * decides nothing about what is fused: `fuse.py` does, and hands over a
 * kernel's place in the table with the operands in the kernel's order.
 *
 * Three entry points:
 *
 *   program(steps, arrays, operands, out, block) -> flags, or DECLINED
 *   elementwise(kernel, operands, out)           -> flags, or DECLINED
 *   reduce(kernel, operands)                     -> (value, flags), or DECLINED
 *
 * `program` is what `fuse.py` calls for an expression: several kernels, each
 * a step, all made on one block of elements before any is made on the next,
 * so that what one step gives another is a buffer that stays in the cache.
 * `elementwise` is one kernel by itself on whole runs.
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
/* A program: the most steps, the most arrays and the most numbers it reads, and the most buffers it keeps. */
#define MOST_STEPS 512
#define MOST_OPERANDS 128
#define MOST_BUFFERS 64
/* The longest stretch of an array that a program repeats into a buffer of its own; see `fused_program`. */
#define LONGEST_PERIOD 8192
/* Where a program's last step writes: the program's result, not a buffer. */
#define RESULT (-1)
/* The shortest inner run that is worth copying arrays into row order for, and below which rows at one pitch are
 * walked straight through instead of one by one. */
#define SHORTEST_RUN 32
/* The shortest row a program of one step is run on by itself: on rows of one element, which a slice `x[:, :1]`
 * has, a call for each takes 3 to 9 times what NumPy's pass does, and on rows of two it is 1.3 times as fast. */
#define SHORTEST_CALL 2
/* The fewest arrays and numbers a kernel reads for a program of that one kernel to be run on rows shorter than
 * `SHORTEST_ROW`, a call for each. A kernel that reads fewer is one operation, or two at most, and NumPy's one
 * pass over rows of 2 to 15 elements is then 1.6 to 4.3 times as fast as a call for each row: `table *
 * column[:, None]` on 100,000 rows. A kernel of two operations on three arrays is 1.2 to 2 times NumPy's two. */
#define FEWEST_READ 3
/* The longest row such a program is walked straight through on with more than `MOST_BETWEEN` between the rows:
 * `wide[:, half:] * 0.1` on 100,000 rows with as much again between them is 1.4 to 1.5 times NumPy's pass for rows
 * of 2, 4 and 8 elements, and 0.64 for rows of 15, where NumPy's pass costs little more than the arithmetic. */
#define SLIGHT_ROW 8
/* The shortest row a program of several steps is run on by itself, every step a call for that row: two of the
 * kernels' blocks of eight. Timed on `(a + b) * c - d` with rows at two pitches, four million elements, against
 * NumPy's three passes: 1.06 to 1.21 times as fast for rows of 16 to 28, and 0.70 to 0.88 for rows of 4 to 12, where
 * leaving the expression to NumPy costs 0.96. */
#define SHORTEST_ROW 16
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
    /* 1 for an array a kernel cannot read where it lies: along the innermost dimension it steps over elements
     * (a column of a grid). A program gathers it into a block first; `elementwise` declines it. */
    int stepped;
} operand;

/* Finds the order in which the dimensions of `out` are walked: the order its elements lie in memory.
 *
 * Args:
 *   out: The array a call fills.
 *   dimensions: How many dimensions it has.
 *   order: Filled in: for each dimension of the walk, outermost first, which dimension of `out` it is.
 *   shape: Filled in: the length of each dimension of the walk.
 *   sorted: Set to 1 when the walk is in another order than the dimensions are written in, else to 0.
 *
 * Returns 1, or 0 when the call is to be declined.
 * NumPy's loops walk the operands of an operation in the order of memory: its iterator sorts the dimensions by
 * stride, largest outermost, and leaves two of one stride in the order they are written in. So does this, for
 * `out`: `np.mgrid[...].transpose(1, 2, 0) % 1` is an array of shape (m, n, 2) whose last dimension is the slowest
 * in memory, and it is walked as one run of 2 * m * n elements and not as m * n rows of two. Where the strides
 * already fall from each dimension to the next the order is the one written, whatever the strides are, and
 * `read_operand` has the rest of the rule. Where they do not, an empty array, a stride below zero and a dimension
 * of length one are declined. NumPy gives a dimension of length one no stride when it sorts, the place of such a
 * dimension in the order then depends on which of its loops an operation takes, and with it the strides of the
 * result: not something to get nearly right.
 */
static int
walk_order(PyArrayObject *out, int dimensions, int *order, npy_intp *shape, int *sorted)
{
    int rises = 0;
    npy_intp previous = NPY_MAX_INTP;
    for (int axis = 0; axis < dimensions; axis++) {
        order[axis] = axis;
        shape[axis] = PyArray_DIM(out, axis);
        if (shape[axis] == 0) {
            return 0;
        }
        if (shape[axis] != 1) {
            npy_intp stride = PyArray_STRIDE(out, axis);
            rises = rises || stride > previous;
            previous = stride;
        }
    }
    *sorted = rises;
    if (!rises) {
        return 1;
    }
    for (int axis = 0; axis < dimensions; axis++) {
        if (shape[axis] == 1 || PyArray_STRIDE(out, axis) <= 0) {
            return 0;
        }
    }
    /* An insertion sort, largest stride first. A dimension moves only past a smaller stride, never past an equal
     * one, so dimensions of one stride stay in the order they are written in. */
    for (int place = 1; place < dimensions; place++) {
        int axis = order[place], at = place;
        npy_intp stride = PyArray_STRIDE(out, axis);
        for (; at > 0 && PyArray_STRIDE(out, order[at - 1]) < stride; at--) {
            order[at] = order[at - 1];
        }
        order[at] = axis;
    }
    for (int axis = 0; axis < dimensions; axis++) {
        shape[axis] = PyArray_DIM(out, order[axis]);
    }
    return 1;
}

/* Reads an operand's layout against the shape walked.
 *
 * Args:
 *   object: A float64 array, or a Python float.
 *   dimensions, shape: The shape the expression runs over, in the order of the walk.
 *   order: For each dimension of the walk, which dimension of the expression it is; from `walk_order`.
 *   sorted: 1 when that is another order than the dimensions are written in.
 *   vector: 1 for an operand the kernel reads along the innermost dimension, 0 for a number.
 *   read: Filled in, its strides in the order of the walk.
 *
 * Returns 1 when the operand can be walked, 0 when the call is to be declined.
 * An array is taken when it is a plain `ndarray` of native aligned float64
 * that broadcasts to the shape: each of its dimensions is the shape's or 1.
 * Its strides must be positive and must not grow from one dimension of the
 * walk to the next: then NumPy's own result for it is laid out in the order
 * of the walk, as `out` is. A vector moves by one element along the innermost
 * dimension; a number that is an array has length 1 there.
 *
 * In a walk that is sorted the rule is narrower, because the layout NumPy
 * gives a result is then decided by every operation of the expression, of
 * which this module sees only the arrays. An array has every dimension of the
 * shape, or is one element. One that is the same along a dimension would
 * leave the order of that dimension to the other operand of its operation,
 * and `row + column` is in C order whatever a third array is. And where two
 * dimensions are walked in the other order than they are written in, the
 * outer one's stride is the larger and not merely as large: of two equal
 * strides NumPy takes the order written.
 */
static int
read_operand(PyObject *object, int dimensions, const npy_intp *shape, const int *order, int sorted, int vector,
             operand *read)
{
    if (PyFloat_CheckExact(object)) {
        if (vector) {
            return 0;
        }
        read->is_array = 0;
        read->stepped = 0;
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
    read->stepped = 0;
    read->data = PyArray_BYTES(array);
    npy_intp previous = NPY_MAX_INTP;
    int same = 0; /* The dimensions of the walk along which the array is one element. */
    for (int axis = 0; axis < dimensions; axis++) {
        int theirs = order[axis] - (dimensions - own);
        npy_intp length = theirs < 0 ? 1 : PyArray_DIM(array, theirs);
        if (length == 1) {
            read->strides[axis] = 0; /* Broadcast, or a dimension with nothing to step over. */
            same++;
            continue;
        }
        if (length != shape[axis]) {
            return 0;
        }
        npy_intp stride = PyArray_STRIDE(array, theirs);
        if (stride <= 0 || stride > previous) {
            return 0;
        }
        if (sorted && axis > 0 && stride == previous && order[axis - 1] > order[axis]) {
            return 0;
        }
        previous = stride;
        read->strides[axis] = stride;
    }
    if (sorted && same != 0 && same != dimensions) {
        return 0; /* No dimension of a sorted walk has length one, so this array is the same along one of them. */
    }
    npy_intp inner = read->strides[dimensions - 1];
    if (vector) {
        /* One value for the whole row is not an array that steps: it is one element over and over, which the walks
         * read from a buffer (`period`), and the walk that gathers fills in. */
        read->stepped = inner != ELEMENT && inner != 0 && shape[dimensions - 1] != 1;
        return 1;
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
    int order[MOST_DIMENSIONS], sorted;
    if (!walk_order(out, dimensions, order, shape, &sorted)) {
        return PyLong_FromLong(DECLINED);
    }
    /* The operands, the kernel's arrays first, then `out`, then its numbers. */
    operand reads[MOST_INTEGERS + MOST_NUMBERS];
    for (int place = 0; place < vectors; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[1], place), dimensions, shape, order, sorted, 1, &reads[place]) || reads[place].stepped) {
            return PyLong_FromLong(DECLINED);
        }
    }
    operand *target = &reads[vectors];
    if (!read_operand((PyObject *)out, dimensions, shape, order, sorted, 1, target) || target->stepped) {
        return PyLong_FromLong(DECLINED);
    }
    operand *numbers = &reads[vectors + 1];
    for (int place = 0; place < scalars; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[1], vectors + place), dimensions, shape, order, sorted, 0, &numbers[place])) {
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
                /* Within a run every array is walked an element at a time: that is what joined its dimensions. */
                integers[place] = (uintptr_t)(reads[place].data + done * ELEMENT);
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

/* Says in which order an array's dimensions lie in memory, for an array whose elements are one run.
 *
 * Args:
 *   array: A float64 array.
 *   order: Filled in: its dimensions, the one that steps by one element first.
 *
 * Returns how many dimensions it has, or -1 for an array that is not one run of elements side by side, has no
 * elements, or has a dimension of length one, which `walk_order` leaves out for the same reason.
 * `np.ones((3, 4)).T` is one run, its first dimension the quickest; so is a `transpose` of any contiguous array.
 */
static int
one_run(PyArrayObject *array, int *order)
{
    int dimensions = PyArray_NDIM(array);
    if (dimensions < 1 || dimensions > MOST_DIMENSIONS) {
        return -1;
    }
    char placed[MOST_DIMENSIONS] = {0};
    npy_intp expected = ELEMENT; /* The stride of the next dimension outwards: the bytes of everything inside it. */
    for (int place = 0; place < dimensions; place++) {
        int found = -1;
        for (int axis = dimensions - 1; axis >= 0 && found < 0; axis--) {
            if (!placed[axis] && PyArray_STRIDE(array, axis) == expected) {
                found = axis;
            }
        }
        if (found < 0 || PyArray_DIM(array, found) < 2) {
            return -1;
        }
        placed[found] = 1;
        order[place] = found;
        expected *= PyArray_DIM(array, found);
    }
    return dimensions;
}

/* reduce(kernel, operands) -> (value, flags), or DECLINED.
 *
 * Args:
 *   kernel: The kernel's place in the table.
 *   operands: A tuple, the kernel's arrays and then its numbers, in the order of its arguments.
 *
 * NumPy adds the elements of an array that lies in one run in the order they
 * lie in memory, whatever its shape and whichever of its dimensions is the
 * slowest, and a reduction kernel adds in that order. So every array has to
 * lie in one run and be of one shape, and every number a Python float;
 * anything else is declined. The arrays are all C-contiguous, or all left in
 * one other order by the calls that made them (`a.T`, see `one_run`): what
 * NumPy computes of them before it adds then lies in that order too.
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
    int in_c_order = 1;
    for (int place = 0; place < vectors; place++) {
        PyObject *object = PyTuple_GET_ITEM(args[1], place);
        if (!PyArray_CheckExact(object)) {
            return PyLong_FromLong(DECLINED);
        }
        PyArrayObject *array = (PyArrayObject *)object;
        if (PyArray_TYPE(array) != NPY_DOUBLE || !PyArray_ISBEHAVED_RO(array)) {
            return PyLong_FromLong(DECLINED);
        }
        in_c_order = in_c_order && PyArray_IS_C_CONTIGUOUS(array);
        if (first == NULL) {
            first = array;
        } else if (!PyArray_SAMESHAPE(array, first)) {
            return PyLong_FromLong(DECLINED);
        }
        integers[place] = (uintptr_t)PyArray_BYTES(array);
    }
    if (!in_c_order) {
        /* One of them is not C-contiguous: then each is one run in the same order of its dimensions, or the
         * elements one call reads side by side are not the elements of one place. */
        int order[MOST_DIMENSIONS], other[MOST_DIMENSIONS];
        int dimensions = one_run(first, order);
        if (dimensions < 0) {
            return PyLong_FromLong(DECLINED);
        }
        for (int place = 1; place < vectors; place++) {
            PyArrayObject *array = (PyArrayObject *)PyTuple_GET_ITEM(args[1], place);
            if (one_run(array, other) != dimensions || memcmp(order, other, (size_t)dimensions * sizeof(int)) != 0) {
                return PyLong_FromLong(DECLINED);
            }
        }
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

/* One step of a program: a kernel, where it reads and where it writes. */
typedef struct {
    const fused_kernel *kernel;
    /* Where each array the kernel reads is: an operand's place, or MOST_OPERANDS and up for a block's buffer. */
    int reads[MOST_INTEGERS];
    /* The buffer the result goes to, counted like `reads`; or RESULT for the program's own result. */
    int writes;
    /* Which of the program's numbers each number of the kernel is. */
    int numbers[MOST_NUMBERS];
} program_step;

/* Copies `count` elements that lie `step` bytes apart into a buffer: a row, or part of one. A step of no bytes is
 * one element for the whole row. */
static inline void
gather_run(double *to, const char *from, npy_intp count, npy_intp step)
{
    if (step == ELEMENT) {
        const double *source = (const double *)from;
        for (npy_intp each = 0; each < count; each++) {
            to[each] = source[each];
        }
    } else {
        for (npy_intp each = 0; each < count; each++) {
            to[each] = *(const double *)(from + each * step);
        }
    }
}

/* Where the rows of one array are while the shape is walked row by row: the row the walk is at, as a position
 * along each outer dimension, where it lies in memory, and at which element of the walk it starts. */
typedef struct {
    npy_intp position[MOST_DIMENSIONS];
    npy_intp start;
    int done;
} row_walk;

/* Moves a walk of rows to the next row: the position, and the place in memory of each of `count` arrays. A row is
 * the dimensions from `inner` inwards, `width` elements, and the dimensions outside it are stepped. */
static void
next_row(row_walk *walk, operand **arrays, char **rows, int count, int inner, npy_intp width, const npy_intp *shape)
{
    int axis = inner - 1;
    for (; axis >= 0; axis--) {
        if (++walk->position[axis] < shape[axis]) {
            for (int each = 0; each < count; each++) {
                rows[each] += arrays[each]->strides[axis];
            }
            break;
        }
        for (int each = 0; each < count; each++) {
            rows[each] -= arrays[each]->strides[axis] * (shape[axis] - 1);
        }
        walk->position[axis] = 0;
    }
    walk->start += width;
    walk->done = axis < 0;
}

/* The outermost dimension from which every one of `count` arrays walks at one step to the end of its innermost:
 * each dimension's stride is the length of the one inside it times that one's stride, so the elements of those
 * dimensions lie a fixed step apart, as a row's do. `tke[:, :, :, tau]` of a grid with three time levels is all
 * its elements, three apart. Returns the dimension; the innermost when no other joins it. */
static int
joined_from(operand **arrays, int count, int dimensions, const npy_intp *shape)
{
    int inner = dimensions - 1;
    for (; inner > 0; inner--) {
        for (int each = 0; each < count; each++) {
            if (arrays[each]->strides[inner - 1] != arrays[each]->strides[inner] * shape[inner]) {
                return inner;
            }
        }
    }
    return inner;
}

/* A program on arrays of which one steps over elements (a column of a table), which no kernel can read where it
 * lies.
 *
 * The shape is walked element by element in the order of its rows, a block at a time. An array that lies in that
 * order, one row after the other, is read where it is. Any other is gathered: the rows of the block are copied
 * into a buffer of the array's own, by a loop written here and not a call for each row. The result is written
 * where it belongs when it lies in the order of the rows, and otherwise into a buffer whose rows are copied out,
 * as it is when it is one of the arrays read, so that every row of a block is read before one is written.
 *
 * A row is as many of the innermost dimensions as every array gathered walks at one step (`joined_from`): an
 * array that steps the same way through all of them is copied by one loop over the lot. An array that is read in
 * two places of the program, `vel[s]` in both halves of a flux, is gathered once. Where a row is shorter than
 * `SHORTEST_RUN`, in the arrays gathered or in a result copied out, nothing is done: the copies of rows that
 * short cost what they save.
 *
 * A copy for each array that lies otherwise, within the cache, in place of a pass over memory for each step.
 * Returns 1 when the program was made, 0 when it was not and nothing was done, or -1 with an error set. */
static int
program_flat(const program_step *steps, Py_ssize_t step_count, operand *reads, int arrays, const double *floats,
             int dimensions, const npy_intp *shape, npy_intp block, int buffers, int in_place)
{
    operand *target = &reads[arrays];
    int last = dimensions - 1;
    npy_intp total = 1;
    for (int axis = 0; axis < dimensions; axis++) {
        total *= shape[axis];
    }
    /* Whether each array, and the result, lies in the order of the rows: a stride for each dimension that is the
     * elements of everything inside it. */
    int direct[MOST_OPERANDS + 1];
    operand *gathered[MOST_OPERANDS];
    int gathered_at[MOST_OPERANDS], gathered_count = 0;
    for (int place = 0; place <= arrays; place++) {
        npy_intp inside = ELEMENT;
        direct[place] = 1;
        for (int axis = last; axis >= 0; axis--) {
            direct[place] = direct[place] && (shape[axis] == 1 || reads[place].strides[axis] == inside);
            inside *= shape[axis];
        }
        if (place < arrays && !direct[place]) {
            /* The same memory, walked the same way, as an array met before: its buffer serves both. */
            int known = -1;
            for (int each = 0; each < gathered_count && known < 0; each++) {
                if (gathered[each]->data == reads[place].data
                    && memcmp(gathered[each]->strides, reads[place].strides, (size_t)dimensions * sizeof(npy_intp)) == 0) {
                    known = each;
                }
            }
            if (known >= 0) {
                gathered_at[place] = known;
            } else {
                gathered_at[place] = gathered_count;
                gathered[gathered_count++] = &reads[place];
            }
        }
    }
    int scattered = !direct[arrays] || in_place;
    /* The rows of each array gathered, and of the result where it is copied out. Each walks its own: a grid taken
     * at one time level is thousands of elements at one step, and one value for each row beside it is a row. */
    int inner_of[MOST_OPERANDS], inner_out = joined_from(&target, 1, dimensions, shape);
    npy_intp width_of[MOST_OPERANDS], width_out = 1;
    for (int axis = last; axis >= 0; axis--) {
        width_out *= axis >= inner_out ? shape[axis] : 1;
    }
    for (int each = 0; each < gathered_count; each++) {
        inner_of[each] = joined_from(&gathered[each], 1, dimensions, shape);
        width_of[each] = 1;
        for (int axis = last; axis >= inner_of[each]; axis--) {
            width_of[each] *= shape[axis];
        }
        /* A value for each row is filled in, which costs little for a row of any length. Elements copied from
         * where they lie are not worth it for rows of a few. */
        if (gathered[each]->strides[last] != 0 && width_of[each] < SHORTEST_RUN) {
            return 0;
        }
    }
    if (!direct[arrays] && width_out < SHORTEST_RUN) {
        return 0;
    }
    if (block > total) {
        block = total + (BLOCK - total % BLOCK) % BLOCK;
    }
    /* The steps' buffers, then the result's, then one for each array gathered: a block each. */
    double *memory = PyMem_Malloc((size_t)(buffers + 1 + gathered_count) * (size_t)block * sizeof(double));
    if (memory == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    double *result = memory + (size_t)buffers * (size_t)block;
    double *copies = result + block;
    char *rows[MOST_OPERANDS], *out_row = target->data;
    for (int each = 0; each < gathered_count; each++) {
        rows[each] = gathered[each]->data;
    }
    row_walk reading[MOST_OPERANDS], writing = {{0}, 0, 0};
    memset(reading, 0, sizeof reading);
    for (npy_intp start = 0; start < total; start += block) {
        npy_intp length = total - start < block ? total - start : block;
        for (int each = 0; each < gathered_count; each++) {
            row_walk *walk = &reading[each];
            npy_intp step = gathered[each]->strides[last], width = width_of[each];
            double *copy = copies + (size_t)each * (size_t)block;
            while (!walk->done && walk->start < start + length) {
                npy_intp from = walk->start > start ? walk->start : start;
                npy_intp upto = walk->start + width < start + length ? walk->start + width : start + length;
                gather_run(copy + (from - start), rows[each] + (from - walk->start) * step, upto - from, step);
                if (walk->start + width > start + length) {
                    break; /* The rest of this row is in the next block. */
                }
                next_row(walk, &gathered[each], &rows[each], 1, inner_of[each], width, shape);
            }
        }
        for (Py_ssize_t each = 0; each < step_count; each++) {
            const program_step *step = &steps[each];
            int vectors = step->kernel->vectors, scalars = step->kernel->scalars;
            uintptr_t integers[MOST_INTEGERS] = {0};
            double values[MOST_NUMBERS] = {0};
            for (int place = 0; place < vectors; place++) {
                int from = step->reads[place];
                if (from >= MOST_OPERANDS) {
                    integers[place] = (uintptr_t)(memory + (size_t)(from - MOST_OPERANDS) * (size_t)block);
                } else if (direct[from]) {
                    integers[place] = (uintptr_t)(reads[from].data + start * ELEMENT);
                } else {
                    integers[place] = (uintptr_t)(copies + (size_t)gathered_at[from] * (size_t)block);
                }
            }
            if (step->writes != RESULT) {
                integers[vectors] = (uintptr_t)(memory + (size_t)(step->writes - MOST_OPERANDS) * (size_t)block);
            } else {
                integers[vectors] = scattered ? (uintptr_t)result : (uintptr_t)(target->data + start * ELEMENT);
            }
            integers[vectors + 1] = (uintptr_t)length;
            for (int place = 0; place < scalars; place++) {
                values[place] = floats[step->numbers[place]];
            }
            ((elementwise_kernel)step->kernel->function)(
                integers[0], integers[1], integers[2], integers[3], integers[4], integers[5], integers[6], integers[7],
                values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
        }
        while (scattered && !writing.done && writing.start < start + length) {
            npy_intp from = writing.start > start ? writing.start : start;
            npy_intp upto = writing.start + width_out < start + length ? writing.start + width_out : start + length;
            npy_intp step = target->strides[last];
            char *to = out_row + (from - writing.start) * step;
            const double *source = result + (from - start);
            if (step == ELEMENT) {
                double *row = (double *)to;
                for (npy_intp each = 0; each < upto - from; each++) {
                    row[each] = source[each];
                }
            } else {
                for (npy_intp each = 0; each < upto - from; each++) {
                    *(double *)(to + each * step) = source[each];
                }
            }
            if (writing.start + width_out > start + length) {
                break;
            }
            next_row(&writing, &target, &out_row, 1, inner_out, width_out, shape);
        }
    }
    PyMem_Free(memory);
    return 1;
}

/* The most of a walk that may be elements between rows, as a divisor: a quarter. More than that and the rows are
 * not walked as one stretch, since the steps would mostly compute what nobody asked for, and a row of
 * `SHORTEST_ROW` elements or more is quicker as a program of its own. */
#define MOST_BETWEEN 4
/* For rows shorter than `SHORTEST_ROW`, how many times a row's length the rows may lie apart and still be walked
 * straight through. NumPy's own loop is called once for each such row, which costs it more than the row's
 * arithmetic does: `particles[:, :2] += particles[:, 2:] * dt` on 100,000 rows takes it 1,255 µs, and the same
 * arithmetic on 200,000 elements in one run 82 µs. So a walk that computes eight times what was asked for still
 * wins. Timed on 400,000 elements in rows of 2 to 15, every 1.25 to 8 times their length, a program of one kernel
 * is 1.3 to 2.9 times NumPy's passes and one of two kernels 1.05 to 3.2 times, but for one layout of the 98 that
 * is 0.93. Not timed farther apart. */
#define MOST_SPREAD 8

/* A program on rows too short to be worth a call each, where the rows lie at one pitch in every array: slices that
 * cut the innermost dimension, `x[2:-2, 2:-2, :-1]` of a grid 26 deep, are rows of 25 elements every 26.
 *
 * Such rows cannot be joined, since a row does not end where the next begins. But every array has the same
 * elements between its rows as it has rows, so the steps can walk each array straight through, from its first
 * element to its last, the elements between the rows included: element for element the arrays still correspond.
 * What the steps make of the elements between rows is never looked at. The last step writes a buffer of a block,
 * and only the rows are copied from it to where the result belongs, which may be laid out any way. So the arrays
 * are read where they lie, with no copy, at the cost of the arithmetic on the elements between rows.
 *
 * An array that is the same along a dimension walked this way (`dzt[None, None, :-1]`) is read from a buffer that
 * holds its stretch at the others' pitch, over and over. Dimensions are taken from the innermost outwards for as
 * long as the arrays agree; the ones left are stepped one position at a time, as everywhere.
 *
 * The elements between rows are other elements of the same arrays, so they are numbers like any, and are read
 * within the arrays' memory. Their arithmetic can raise a floating-point flag that no row raised; the result is
 * then dropped like any that raised one, and NumPy's calls made.
 *
 * How much may lie between rows is `MOST_BETWEEN` of the walk, and for rows of a few elements `MOST_SPREAD` times
 * the row: rows shorter than `SHORTEST_ROW`, or `SLIGHT_ROW` at most for a program of one kernel that reads fewer
 * than `FEWEST_READ`.
 *
 * Returns 1 when the program was made this way, 0 when it does not apply and nothing was done, -1 with an error. */
static int
program_spanned(const program_step *steps, Py_ssize_t step_count, operand *reads, int arrays, operand *numbers,
                int numbers_given, int dimensions, const npy_intp *shape, npy_intp block, int buffers)
{
    /* The longest row that may have more than `MOST_BETWEEN` between it and the next. */
    int slight = step_count == 1 && steps[0].kernel->vectors + steps[0].kernel->scalars < FEWEST_READ;
    npy_intp spread_below = slight ? SLIGHT_ROW + 1 : SHORTEST_ROW;
    operand *target = &reads[arrays];
    int last = dimensions - 1;
    npy_intp width = shape[last];
    if (dimensions < 2 || arrays == 0) {
        return 0;
    }
    for (int place = 0; place < arrays; place++) {
        if (reads[place].strides[last] != ELEMENT) {
            return 0;
        }
    }
    /* From the innermost dimension outwards: `pitch[axis]` is how many elements every array moves by for one
     * position along the dimension, and `extent` how many lie from the first to the last element of what is
     * walked so far. `period[place]` is that pitch for an array that is the same along a dimension, with
     * `own[place]` the elements of its own it has within one period. */
    npy_intp pitch[MOST_DIMENSIONS] = {0}, period[MOST_OPERANDS] = {0}, own[MOST_OPERANDS] = {0};
    npy_intp extent = width;
    int first = last;
    for (int axis = last - 1; axis >= 0; axis--) {
        if (shape[axis] == 1) {
            first = axis;
            continue;
        }
        npy_intp common = 0;
        int agreed = 1;
        for (int place = 0; place < arrays && agreed; place++) {
            npy_intp stride = reads[place].strides[axis];
            if (period[place]) {
                agreed = stride == 0; /* Repeats already: only if it is the same along this dimension too. */
            } else if (stride != 0) {
                agreed = stride % ELEMENT == 0 && (common == 0 || stride == common);
                common = stride;
            }
        }
        for (int place = 0; place < numbers_given && agreed; place++) {
            agreed = !numbers[place].is_array || numbers[place].strides[axis] == 0;
        }
        npy_intp step = common / ELEMENT;
        if (!agreed || common == 0 || step < extent) {
            break;
        }
        /* Too much between: for the rows themselves when they are short, by their length; else by the walk. */
        if (axis == last - 1 && width < spread_below ? step > width * MOST_SPREAD : (step - extent) * MOST_BETWEEN > step) {
            break;
        }
        int fits = 1;
        for (int place = 0; place < arrays; place++) {
            fits = fits && (period[place] || reads[place].strides[axis] != 0 || step <= LONGEST_PERIOD);
        }
        if (!fits) {
            break;
        }
        for (int place = 0; place < arrays; place++) {
            if (!period[place] && reads[place].strides[axis] == 0) {
                period[place] = step;
                own[place] = extent;
            }
        }
        pitch[axis] = step;
        extent += (shape[axis] - 1) * step;
        first = axis;
    }
    if (first == last || extent < SHORTEST_RUN) {
        return 0;
    }
    if (block > extent) {
        block = extent + (BLOCK - extent % BLOCK) % BLOCK;
    }
    /* The steps' buffers and the result's, a block each; then for each array that repeats, a block and a period. */
    size_t needed = (size_t)(buffers + 1) * (size_t)block;
    npy_intp tile_at[MOST_OPERANDS];
    for (int place = 0; place < arrays; place++) {
        tile_at[place] = (npy_intp)needed;
        if (period[place]) {
            needed += (size_t)(block + period[place]);
        }
    }
    double *memory = PyMem_Malloc(needed * sizeof(double));
    if (memory == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    double *result = memory + (size_t)buffers * (size_t)block;
    npy_intp out_step = target->strides[last];
    npy_intp position[MOST_DIMENSIONS] = {0};
    double floats[MOST_OPERANDS];
    for (;;) {
        for (int place = 0; place < numbers_given; place++) {
            floats[place] = numbers[place].is_array ? *(const double *)numbers[place].data : numbers[place].number;
        }
        for (int place = 0; place < arrays; place++) {
            if (!period[place]) {
                continue;
            }
            /* One period: the array's own elements, then one of them again for the elements between rows; and the
             * period over and over for a block and a period more, so that a block may start anywhere in one. */
            double *tile = memory + tile_at[place];
            npy_intp held = period[place], length = block + period[place];
            memcpy(tile, reads[place].data, (size_t)own[place] * sizeof(double));
            for (npy_intp each = own[place]; each < held; each++) {
                tile[each] = tile[0];
            }
            while (held < length) {
                npy_intp more = held < length - held ? held : length - held;
                memcpy(tile + held, tile, (size_t)more * sizeof(double));
                held += more;
            }
        }
        /* The rows of the result, in the order they lie: where the next one starts in the walk, and in memory. */
        npy_intp row[MOST_DIMENSIONS] = {0};
        npy_intp row_at = 0;
        char *row_out = target->data;
        int rows_left = 1;
        for (npy_intp start = 0; start < extent; start += block) {
            npy_intp length = extent - start < block ? extent - start : block;
            for (Py_ssize_t each = 0; each < step_count; each++) {
                const program_step *step = &steps[each];
                int vectors = step->kernel->vectors, scalars = step->kernel->scalars;
                uintptr_t integers[MOST_INTEGERS] = {0};
                double values[MOST_NUMBERS] = {0};
                for (int place = 0; place < vectors; place++) {
                    int from = step->reads[place];
                    if (from >= MOST_OPERANDS) {
                        integers[place] = (uintptr_t)(memory + (size_t)(from - MOST_OPERANDS) * (size_t)block);
                    } else if (period[from]) {
                        integers[place] = (uintptr_t)(memory + tile_at[from] + start % period[from]);
                    } else {
                        integers[place] = (uintptr_t)(reads[from].data + start * ELEMENT);
                    }
                }
                integers[vectors] = step->writes == RESULT ? (uintptr_t)result
                                                           : (uintptr_t)(memory + (size_t)(step->writes - MOST_OPERANDS) * (size_t)block);
                integers[vectors + 1] = (uintptr_t)length;
                for (int place = 0; place < scalars; place++) {
                    values[place] = floats[step->numbers[place]];
                }
                ((elementwise_kernel)step->kernel->function)(
                    integers[0], integers[1], integers[2], integers[3], integers[4], integers[5], integers[6], integers[7],
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
            }
            while (rows_left && row_at < start + length) {
                npy_intp from = row_at > start ? row_at : start;
                npy_intp upto = row_at + width < start + length ? row_at + width : start + length;
                if (out_step == ELEMENT && upto - from < SHORTEST_ROW) {
                    /* A few elements: a loop here, which a call to `memcpy` for each row would cost more than. */
                    double *to = (double *)(row_out + (from - row_at) * ELEMENT);
                    const double *made = result + (from - start);
                    for (npy_intp each = 0; each < upto - from; each++) {
                        to[each] = made[each];
                    }
                } else if (out_step == ELEMENT) {
                    memcpy(row_out + (from - row_at) * ELEMENT, result + (from - start), (size_t)(upto - from) * sizeof(double));
                } else {
                    for (npy_intp each = from; each < upto; each++) {
                        *(double *)(row_out + (each - row_at) * out_step) = result[each - start];
                    }
                }
                if (row_at + width > start + length) {
                    break; /* The rest of this row is in the next block. */
                }
                int axis = last - 1;
                for (; axis >= first; axis--) {
                    if (++row[axis] < shape[axis]) {
                        row_at += pitch[axis];
                        row_out += target->strides[axis];
                        break;
                    }
                    row_at -= (shape[axis] - 1) * pitch[axis];
                    row_out -= (shape[axis] - 1) * target->strides[axis];
                    row[axis] = 0;
                }
                rows_left = axis >= first;
            }
        }
        /* Step the dimensions that are not walked through, as an odometer. */
        int axis = first - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                for (int place = 0; place <= arrays + numbers_given; place++) {
                    if (reads[place].is_array) {
                        reads[place].data += reads[place].strides[axis];
                    }
                }
                break;
            }
            for (int place = 0; place <= arrays + numbers_given; place++) {
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
    PyMem_Free(memory);
    return 1;
}

/* The kernels that are one operation of arithmetic on two arrays, or on an array and a number: what `program_rows`
 * makes by a loop of its own. `left` and `right` say which side is the kernel's number. */
typedef struct {
    const char *name;
    char operation;
    int left, right;
} row_kernel;
static const row_kernel ROW_KERNELS[] = {
    {"fuse_add_v_v_f64", '+', 0, 0}, {"fuse_sub_v_v_f64", '-', 0, 0}, {"fuse_mul_v_v_f64", '*', 0, 0}, {"fuse_div_v_v_f64", '/', 0, 0},
    {"fuse_add_s_v_f64", '+', 1, 0}, {"fuse_mul_s_v_f64", '*', 1, 0}, {"fuse_sub_s_v_f64", '-', 1, 0}, {"fuse_div_s_v_f64", '/', 1, 0},
    {"fuse_sub_v_s_f64", '-', 0, 1}, {"fuse_div_v_s_f64", '/', 0, 1},
};

/* The entry of `ROW_KERNELS` for a kernel, or NULL when it is none of them. */
static const row_kernel *
row_kernel_of(const fused_kernel *kernel)
{
    for (size_t each = 0; each < sizeof ROW_KERNELS / sizeof ROW_KERNELS[0]; each++) {
        if (strcmp(kernel->name, ROW_KERNELS[each].name) == 0) {
            return &ROW_KERNELS[each];
        }
    }
    return NULL;
}

/* The rows along one dimension of one operation: `rows` of them, `width` elements each, of the arrays at `pa` and
 * `pb` or of one of them and a number, into the rows at `po`. From one row to the next each moves by its own count
 * of bytes: `sa`, `sb`, `so`, and `sx` for the number at `px`, which is nothing for a number that is the same for
 * every row. Each element is the processor's one instruction for the operation, which is what NumPy's loop and the
 * kernel make. */
static void
rows_made(const row_kernel *made, const char *pa, const char *pb, const char *px, char *po, npy_intp rows, npy_intp width,
          npy_intp sa, npy_intp sb, npy_intp sx, npy_intp so)
{
#define ROWS(expression) \
    for (npy_intp row = 0; row < rows; row++) { \
        const double *a = (const double *)(pa + row * sa), *b = (const double *)(pb + row * sb); \
        const double x = *(const double *)(px + row * sx); \
        double *out = (double *)(po + row * so); \
        for (npy_intp each = 0; each < width; each++) { \
            out[each] = expression; \
        } \
        (void)b; \
        (void)x; \
    }
    switch (made->operation) {
    case '+':
        if (made->left) { ROWS(x + a[each]) } else { ROWS(a[each] + b[each]) }
        break;
    case '*':
        if (made->left) { ROWS(x * a[each]) } else { ROWS(a[each] * b[each]) }
        break;
    case '-':
        if (made->left) { ROWS(x - a[each]) } else if (made->right) { ROWS(a[each] - x) } else { ROWS(a[each] - b[each]) }
        break;
    default:
        if (made->left) { ROWS(x / a[each]) } else if (made->right) { ROWS(a[each] / x) } else { ROWS(a[each] / b[each]) }
        break;
    }
#undef ROWS
}

/* A program that is one kernel of one operation, on rows too short for a call each. NumPy's loop is called once for
 * each row of arrays it cannot walk as one run, which costs it more than a short row's arithmetic: `table *
 * weights[:, None]` on 100,000 rows of four takes it 294 µs and `table[:, :2] += made` on rows of two 861 µs. A
 * call of the kernel for each row costs as much. So the operation is made here, by two loops with nothing in them
 * but the rows' arithmetic: 105 µs and 90 µs for those two, written out by themselves. The arrays may lie at any
 * pitch, a number may be another for each row, and the result may be one of the arrays, element for element.
 *
 * `made` is the kernel's entry and `step` the program's one step. The dimension outside the rows is walked by
 * `rows_made`, and the ones outside that are stepped here. Returns 0. */
static int
program_rows(const row_kernel *made, const program_step *step, operand *reads, int arrays, int numbers_given, int dimensions,
             const npy_intp *shape)
{
    operand *target = &reads[arrays], *numbers = &reads[arrays + 1];
    const operand *first = &reads[step->reads[0]];
    const operand *second = step->kernel->vectors > 1 ? &reads[step->reads[1]] : first;
    const operand *number = step->kernel->scalars ? &numbers[step->numbers[0]] : NULL;
    int last = dimensions - 1, along = dimensions - 2;
    /* One value for each row that the planner had read as an array (`weights[:, None]` beside rows this short) is
     * the operation's number here: the kernel of two arrays is made as its like of an array and a number. */
    row_kernel as_made = *made;
    if (step->kernel->vectors > 1 && first->strides[last] == 0) {
        as_made.left = 1;
        number = first;
        first = second;
    } else if (step->kernel->vectors > 1 && second->strides[last] == 0) {
        if (as_made.operation == '+' || as_made.operation == '*') {
            as_made.left = 1; /* Either order gives the same sum and the same product. */
        } else {
            as_made.right = 1;
        }
        number = second;
        second = first;
    }
    made = &as_made;
    double same = number != NULL && !number->is_array ? number->number : 0.0;
    npy_intp position[MOST_DIMENSIONS] = {0};
    for (;;) {
        const char *px = number != NULL && number->is_array ? number->data : (const char *)&same;
        npy_intp sx = number != NULL && number->is_array ? number->strides[along] : 0;
        rows_made(made, first->data, second->data, px, target->data, shape[along], shape[last], first->strides[along],
                  second->strides[along], sx, target->strides[along]);
        int axis = along - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                for (int place = 0; place <= arrays + numbers_given; place++) {
                    if (reads[place].is_array) {
                        reads[place].data += reads[place].strides[axis];
                    }
                }
                break;
            }
            for (int place = 0; place <= arrays + numbers_given; place++) {
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
    return 0;
}

/* The kernel `program_stored` makes by a loop of its own: a product of three arrays. */
#define STORED_KERNEL "fuse_product3_f64"

/* A program that is one product of three arrays, stored into a result that steps over elements, in rows shorter
 * than `SHORTEST_RUN` that the arrays read hold as runs: isoneutral mixing's `Ai_bx[2:-2, 2:-2, :-1, ip, kr] =
 * taper * sxb * maskW[2:-2, 2:-2, :-1]`, sixteen such lines on a result with two more axes after its grid.
 *
 * NumPy makes the two products a row at a time and stores the rows at the step: 1,537 µs for a million elements in
 * rows of 25, of which 731 µs the products. Rows that short are not worth gathering (`program_flat`), and a call
 * of the kernel for each row with the row stored afterwards was built and took 1,632 µs: the call costs what the
 * second pass did. Here each row is made into a buffer on the stack by a loop with nothing in it but the two
 * multiplications, in the kernel's order, `a * (b * c)`, and stored from there: 985 µs for that statement written
 * out by itself.
 *
 * `step` is the program's one step. The dimension outside the rows is walked here with the rows, and the ones
 * outside that are stepped. Returns 0. */
static int
program_stored(const program_step *step, operand *reads, int arrays, int numbers_given, int dimensions, const npy_intp *shape)
{
    operand *target = &reads[arrays];
    const operand *first = &reads[step->reads[0]], *second = &reads[step->reads[1]], *third = &reads[step->reads[2]];
    int last = dimensions - 1, along = dimensions - 2;
    npy_intp rows = shape[along], width = shape[last], spread = target->strides[last];
    npy_intp position[MOST_DIMENSIONS] = {0};
    double made[SHORTEST_RUN];
    for (;;) {
        for (npy_intp row = 0; row < rows; row++) {
            const double *a = (const double *)(first->data + row * first->strides[along]);
            const double *b = (const double *)(second->data + row * second->strides[along]);
            const double *c = (const double *)(third->data + row * third->strides[along]);
            char *out = target->data + row * target->strides[along];
            for (npy_intp each = 0; each < width; each++) {
                made[each] = a[each] * (b[each] * c[each]); /* The kernel's order, which is the expression's. */
            }
            for (npy_intp each = 0; each < width; each++) {
                *(double *)(out + each * spread) = made[each];
            }
        }
        int axis = along - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                for (int place = 0; place <= arrays + numbers_given; place++) {
                    if (reads[place].is_array) {
                        reads[place].data += reads[place].strides[axis];
                    }
                }
                break;
            }
            for (int place = 0; place <= arrays + numbers_given; place++) {
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
    return 0;
}

/* One step of a program with everything about it that does not change from run to run worked out: the kernel, and
 * for each array it reads and for its result the address of the pointer that says where that is now. An operand's
 * pointer moves as the walk steps; a buffer's stays. */
typedef struct {
    elementwise_kernel function;
    int vectors, scalars;
    char **from[MOST_INTEGERS];
    char **into;
    int numbers[MOST_NUMBERS];
} resolved_step;

/* A program whose every run is one block: rows shorter than a block, each row a call of every step. This is where
 * a slice that cuts the innermost dimension ends up when its arrays do not share a pitch, 25 elements to a row,
 * and what a step costs beside its kernel is then much of what the program costs. So everything that is the same
 * for every run is worked out once, and a run is the kernels' calls and the pointers moved on. Against the general
 * walk, on `(a + b) * c - d` over four million elements: 3.9 ms for 4.5 ms on rows of 32, 3.0 to 3.2 for 3.3 on rows
 * of 64, and the same from 128 up.
 *
 * `outer` is how many dimensions are stepped, `run` the elements of a run, `memory` the steps' buffers and `copies`
 * the buffers of the arrays that repeat, each a block. The result is never one of the arrays read: the caller has
 * the general walk for that. Returns 0, or -1 with an error set. */
static int
program_runs(const program_step *steps, Py_ssize_t step_count, operand *reads, int arrays, int numbers_given, const npy_intp *period,
             const int *repeating, double *memory, double *copies, npy_intp block, npy_intp run, int outer, const npy_intp *shape)
{
    operand *target = &reads[arrays], *numbers = &reads[arrays + 1];
    resolved_step *resolved = PyMem_Malloc((size_t)step_count * sizeof(resolved_step));
    if (resolved == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    /* The pointers that stay: a buffer for each step that writes one, then one for each array that repeats. */
    char *fixed[MOST_BUFFERS + MOST_OPERANDS];
    for (Py_ssize_t each = 0; each < step_count; each++) {
        const program_step *step = &steps[each];
        resolved_step *made = &resolved[each];
        made->function = (elementwise_kernel)step->kernel->function;
        made->vectors = step->kernel->vectors;
        made->scalars = step->kernel->scalars;
        for (int place = 0; place < made->vectors; place++) {
            int from = step->reads[place];
            if (from >= MOST_OPERANDS) {
                fixed[from - MOST_OPERANDS] = (char *)(memory + (size_t)(from - MOST_OPERANDS) * (size_t)block);
                made->from[place] = &fixed[from - MOST_OPERANDS];
            } else if (period[from]) {
                fixed[MOST_BUFFERS + repeating[from]] = (char *)(copies + (size_t)repeating[from] * (size_t)block);
                made->from[place] = &fixed[MOST_BUFFERS + repeating[from]];
            } else {
                made->from[place] = &reads[from].data;
            }
        }
        if (step->writes == RESULT) {
            made->into = &target->data;
        } else {
            fixed[step->writes - MOST_OPERANDS] = (char *)(memory + (size_t)(step->writes - MOST_OPERANDS) * (size_t)block);
            made->into = &fixed[step->writes - MOST_OPERANDS];
        }
        for (int place = 0; place < made->scalars; place++) {
            made->numbers[place] = step->numbers[place];
        }
    }
    double floats[MOST_OPERANDS];
    int moving = 0; /* Whether a number is an array, read again at every run. */
    for (int place = 0; place < numbers_given; place++) {
        floats[place] = numbers[place].is_array ? *(const double *)numbers[place].data : numbers[place].number;
        moving = moving || numbers[place].is_array;
    }
    const char *copied[MOST_OPERANDS] = {NULL}; /* Where each array that repeats was when its buffer was filled. */
    npy_intp position[MOST_DIMENSIONS] = {0};
    uintptr_t integers[MOST_INTEGERS] = {0};
    double values[MOST_NUMBERS] = {0};
    for (;;) {
        for (int place = 0; moving && place < numbers_given; place++) {
            if (numbers[place].is_array) {
                floats[place] = *(const double *)numbers[place].data;
            }
        }
        for (int place = 0; place < arrays; place++) {
            if (period[place] && reads[place].data != copied[place]) {
                /* The stretch as it is at this run, over and over for the length of a block. */
                double *copy = copies + (size_t)repeating[place] * (size_t)block;
                npy_intp held = period[place] < block ? period[place] : block;
                memcpy(copy, reads[place].data, (size_t)held * sizeof(double));
                while (held < block) {
                    npy_intp more = held < block - held ? held : block - held;
                    memcpy(copy + held, copy, (size_t)more * sizeof(double));
                    held += more;
                }
                copied[place] = reads[place].data;
            }
        }
        for (Py_ssize_t each = 0; each < step_count; each++) {
            const resolved_step *step = &resolved[each];
            int vectors = step->vectors;
            for (int place = 0; place < vectors; place++) {
                integers[place] = (uintptr_t)*step->from[place];
            }
            integers[vectors] = (uintptr_t)*step->into;
            integers[vectors + 1] = (uintptr_t)run;
            for (int place = 0; place < step->scalars; place++) {
                values[place] = floats[step->numbers[place]];
            }
            step->function(integers[0], integers[1], integers[2], integers[3], integers[4], integers[5], integers[6], integers[7],
                           values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
        }
        int axis = outer - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                for (int place = 0; place <= arrays + numbers_given; place++) {
                    if (reads[place].is_array) {
                        reads[place].data += reads[place].strides[axis];
                    }
                }
                break;
            }
            for (int place = 0; place <= arrays + numbers_given; place++) {
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
    PyMem_Free(resolved);
    return 0;
}

/* program(steps, arrays, operands, out, block) -> flags, or DECLINED.
 *
 * An expression computed block by block. Every operation of it is a step, a
 * kernel call; a step reads the expression's operands or what earlier steps
 * gave, and writes a buffer of one block, but for the last, which writes
 * `out`. All the steps are made on one block before any is made on the
 * next, so what passes from step to step never leaves the cache: only the
 * operands are read from memory and only the result written to it, however
 * many operations lie between.
 *
 * Args:
 *   steps: A tuple of steps, each a tuple of whole numbers: the kernel's place in the table, where the result
 *       goes, then where each of the kernel's arrays is read, then which number each of its numbers is. A place
 *       below `arrays` is an operand; `arrays` and up is a buffer, numbered from `arrays`; -1, for a result
 *       only, is `out`. A step reads only operands and buffers that an earlier step wrote, and never the
 *       buffer it writes.
 *   arrays: How many of the operands are arrays.
 *   operands: A tuple, the arrays and then the numbers.
 *   out: A writable float64 array of the expression's shape. It is filled. It may be one of the operands
 *       exactly, as for `elementwise`, and is declined where it overlaps one otherwise.
 *   block: The elements of a block: a multiple of eight.
 */
static PyObject *
fused_program(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 5 || !PyTuple_CheckExact(args[0]) || !PyLong_CheckExact(args[1]) || !PyTuple_CheckExact(args[2])
        || !PyArray_CheckExact(args[3]) || !PyLong_CheckExact(args[4])) {
        PyErr_SetString(PyExc_TypeError, "program(steps, arrays, operands, out, block)");
        return NULL;
    }
    Py_ssize_t step_count = PyTuple_GET_SIZE(args[0]);
    long arrays = PyLong_AsLong(args[1]);
    Py_ssize_t given = PyTuple_GET_SIZE(args[2]);
    npy_intp block = PyLong_AsSsize_t(args[4]);
    if (step_count < 1 || step_count > MOST_STEPS || arrays < 0 || arrays > MOST_OPERANDS || given - arrays < 0
        || given - arrays > MOST_OPERANDS || block < BLOCK || block % BLOCK) {
        PyErr_SetString(PyExc_ValueError, "program: too many steps or operands, or a block that is no multiple of eight");
        return NULL;
    }
    int numbers_given = (int)(given - arrays);
    program_step steps[MOST_STEPS];
    int buffers = 0;
    for (Py_ssize_t place = 0; place < step_count; place++) {
        PyObject *written = PyTuple_GET_ITEM(args[0], place);
        program_step *step = &steps[place];
        long index = PyTuple_CheckExact(written) && PyTuple_GET_SIZE(written) >= 2 ? PyLong_AsLong(PyTuple_GET_ITEM(written, 0)) : -1;
        if (index < 0 || index >= KERNEL_COUNT || KERNELS[index].reduces) {
            PyErr_SetString(PyExc_ValueError, "program: a step names no elementwise kernel");
            return NULL;
        }
        step->kernel = &KERNELS[index];
        int vectors = step->kernel->vectors, scalars = step->kernel->scalars;
        if (PyTuple_GET_SIZE(written) != 2 + vectors + scalars || vectors + 2 > MOST_INTEGERS || scalars > MOST_NUMBERS) {
            PyErr_SetString(PyExc_ValueError, "program: a step has another number of operands than its kernel");
            return NULL;
        }
        long writes = PyLong_AsLong(PyTuple_GET_ITEM(written, 1));
        int last = place == step_count - 1;
        if ((last && writes != RESULT) || (!last && (writes < arrays || writes >= arrays + MOST_BUFFERS))) {
            PyErr_SetString(PyExc_ValueError, "program: only the last step writes the result, and every other a buffer");
            return NULL;
        }
        step->writes = last ? RESULT : (int)(writes - arrays) + MOST_OPERANDS;
        if (!last && writes - arrays + 1 > buffers) {
            buffers = (int)(writes - arrays) + 1;
        }
        for (int each = 0; each < vectors; each++) {
            long reads = PyLong_AsLong(PyTuple_GET_ITEM(written, 2 + each));
            if (reads < 0 || reads >= arrays + buffers || reads == writes) {
                PyErr_SetString(PyExc_ValueError, "program: a step reads what is not there, or what it writes");
                return NULL;
            }
            step->reads[each] = reads < arrays ? (int)reads : (int)(reads - arrays) + MOST_OPERANDS;
        }
        for (int each = 0; each < scalars; each++) {
            long number = PyLong_AsLong(PyTuple_GET_ITEM(written, 2 + vectors + each));
            if (number < 0 || number >= numbers_given) {
                PyErr_SetString(PyExc_ValueError, "program: a step names a number that is not there");
                return NULL;
            }
            step->numbers[each] = (int)number;
        }
    }
    PyArrayObject *out = (PyArrayObject *)args[3];
    int dimensions = PyArray_NDIM(out);
    if (dimensions < 1 || dimensions > MOST_DIMENSIONS || PyArray_TYPE(out) != NPY_DOUBLE || !PyArray_ISBEHAVED(out)) {
        return PyLong_FromLong(DECLINED);
    }
    npy_intp shape[MOST_DIMENSIONS];
    int order[MOST_DIMENSIONS], sorted;
    if (!walk_order(out, dimensions, order, shape, &sorted)) {
        return PyLong_FromLong(DECLINED);
    }
    /* The operands as the walk sees them: the arrays, then `out`, then the numbers. */
    operand reads[2 * MOST_OPERANDS + 1];
    for (int place = 0; place < arrays; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[2], place), dimensions, shape, order, sorted, 1, &reads[place])) {
            return PyLong_FromLong(DECLINED);
        }
    }
    operand *target = &reads[arrays];
    if (!read_operand((PyObject *)out, dimensions, shape, order, sorted, 1, target)) {
        return PyLong_FromLong(DECLINED);
    }
    operand *numbers = &reads[arrays + 1];
    for (int place = 0; place < numbers_given; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[2], arrays + place), dimensions, shape, order, sorted, 0, &numbers[place])) {
            return PyLong_FromLong(DECLINED);
        }
    }
    int in_place = 0;
    {
        const char *out_low, *out_high;
        extent(target, dimensions, shape, &out_low, &out_high);
        for (int place = 0; place < arrays + 1 + numbers_given; place++) {
            const operand *read = &reads[place];
            if (place == arrays || !read->is_array) {
                continue;
            }
            const char *low, *high;
            extent(read, dimensions, shape, &low, &high);
            if (low >= out_high || out_low >= high) {
                continue;
            }
            if (place > arrays || read->data != target->data
                || memcmp(read->strides, target->strides, (size_t)dimensions * sizeof(npy_intp)) != 0) {
                return PyLong_FromLong(DECLINED);
            }
            in_place = 1;
        }
    }
    /* Lengthen the inner run by every outer dimension that every operand walks straight through. An array that
     * does not, because it is the same along that dimension (`dzt[None, None, :]` beside a grid), repeats: the run
     * is lengthened all the same, and the array is read from a buffer that holds its stretch over and over. Without
     * that a run would be as short as that array, and every step a call for a handful of elements. */
    int outer = dimensions - 1;
    npy_intp run = shape[outer];
    npy_intp period[MOST_OPERANDS] = {0}; /* For an array that repeats: the elements of its stretch. Else 0. */
    npy_intp longest = 0;
    for (int place = 0; place < arrays; place++) {
        if (reads[place].strides[outer] == 0 && shape[outer] != 1) {
            /* One value for each row (`dx[:, :, None]`): a stretch of one element. The rows it differs between do
             * not join, so each row is a run, and its value is put into the array's buffer when the run begins. */
            period[place] = 1;
            longest = 1;
        }
    }
    while (outer > 0) {
        int axis = outer - 1, joined = 1;
        npy_intp repeats[MOST_OPERANDS] = {0};
        if (shape[axis] != 1) {
            joined = joins(target, 1, axis, run);
            for (int place = 0; place < arrays && joined; place++) {
                if (period[place]) {
                    joined = reads[place].strides[axis] == 0; /* Repeats on: only if it is the same along this one too. */
                } else if (!joins(&reads[place], 1, axis, run)) {
                    if (reads[place].strides[axis] == 0 && run <= LONGEST_PERIOD) {
                        repeats[place] = run;
                    } else {
                        joined = 0;
                    }
                }
            }
            for (int place = 0; place < numbers_given && joined; place++) {
                joined = joins(&numbers[place], 0, axis, run);
            }
        }
        if (!joined) {
            break;
        }
        for (int place = 0; place < arrays; place++) {
            if (repeats[place]) {
                period[place] = repeats[place];
                longest = repeats[place] > longest ? repeats[place] : longest;
            }
        }
        run *= shape[axis];
        outer = axis;
    }
    int stepped = target->stepped;
    for (int place = 0; place < arrays; place++) {
        stepped = stepped || reads[place].stepped;
    }
    int by_rows = run < SHORTEST_ROW && outer > 0 && step_count == 1 && !target->stepped && row_kernel_of(steps[0].kernel) != NULL;
    if (by_rows) {
        /* Every array the kernel reads is a row where it lies, but that one of two may be a value for each row. */
        int values = 0;
        for (int place = 0; place < steps[0].kernel->vectors; place++) {
            npy_intp inner = reads[steps[0].reads[place]].strides[dimensions - 1];
            values += inner == 0;
            by_rows = by_rows && (inner == ELEMENT || inner == 0);
        }
        by_rows = by_rows && values < steps[0].kernel->vectors;
    }
    if (by_rows) {
        /* One operation of arithmetic on rows of a few elements: the module's own loop; see `program_rows`. */
        feclearexcept(FE_ALL_EXCEPT);
        program_rows(row_kernel_of(steps[0].kernel), &steps[0], reads, (int)arrays, numbers_given, dimensions, shape);
        int flags = fetestexcept(REPORTED);
        if (flags) {
            feclearexcept(FE_ALL_EXCEPT);
        }
        return PyLong_FromLong(flags);
    }
    int by_stored = step_count == 1 && target->stepped && !in_place && dimensions > 1 && outer == dimensions - 1
                    && run < SHORTEST_RUN && strcmp(steps[0].kernel->name, STORED_KERNEL) == 0;
    for (int place = 0; by_stored && place < steps[0].kernel->vectors; place++) {
        by_stored = reads[steps[0].reads[place]].strides[dimensions - 1] == ELEMENT;
    }
    if (by_stored) {
        /* A product of three arrays into short rows of a result that steps: the module's own loop; see
         * `program_stored`. */
        feclearexcept(FE_ALL_EXCEPT);
        program_stored(&steps[0], reads, (int)arrays, numbers_given, dimensions, shape);
        int flags = fetestexcept(REPORTED);
        if (flags) {
            feclearexcept(FE_ALL_EXCEPT);
        }
        return PyLong_FromLong(flags);
    }
    if (!stepped && run < SHORTEST_RUN) {
        /* Rows too short to be worth a call each. Where they lie at one pitch in every array they are walked
         * straight through, the elements between them included: see `program_spanned`. */
        feclearexcept(FE_ALL_EXCEPT);
        int spanned = program_spanned(steps, step_count, reads, (int)arrays, numbers, numbers_given, dimensions, shape, block, buffers);
        if (spanned < 0) {
            return NULL;
        }
        if (spanned) {
            int flags = fetestexcept(REPORTED);
            if (flags) {
                feclearexcept(FE_ALL_EXCEPT);
            }
            return PyLong_FromLong(flags);
        }
    }
    int slight = step_count == 1 && outer > 0 && steps[0].kernel->vectors + steps[0].kernel->scalars < FEWEST_READ;
    if (stepped || (run < SHORTEST_ROW && (step_count > 1 || slight)) || (run < SHORTEST_CALL && outer > 0)) {
        /* Arrays a kernel cannot read where they lie: the shape is walked in the order of its rows and the arrays
         * that lie otherwise are gathered; see `program_flat`, which leaves rows of a few elements to NumPy. Left
         * to NumPy as well are rows a kernel can read where they lie but that are shorter than `SHORTEST_ROW`, at
         * a pitch the arrays do not share: every step would be a call for a handful of elements. A program of one
         * step is a call for each row down to `SHORTEST_CALL` elements, and left to NumPy below; one whose kernel
         * reads fewer than `FEWEST_READ` is left to NumPy on every row shorter than `SHORTEST_ROW`. A number that
         * is not the same for every row cannot be gathered: a call has one value for each number. */
        if (!stepped) {
            return PyLong_FromLong(DECLINED);
        }
        for (int place = 0; place < numbers_given; place++) {
            if (!numbers[place].is_array) {
                continue;
            }
            for (int axis = 0; axis < dimensions; axis++) {
                if (numbers[place].strides[axis] != 0) {
                    return PyLong_FromLong(DECLINED);
                }
            }
        }
        double constants[MOST_OPERANDS];
        for (int place = 0; place < numbers_given; place++) {
            constants[place] = numbers[place].is_array ? *(const double *)numbers[place].data : numbers[place].number;
        }
        feclearexcept(FE_ALL_EXCEPT);
        int made = program_flat(steps, step_count, reads, (int)arrays, constants, dimensions, shape, block, buffers, in_place);
        if (made < 0) {
            return NULL;
        }
        if (!made) {
            return PyLong_FromLong(DECLINED);
        }
        int flags = fetestexcept(REPORTED);
        if (flags) {
            feclearexcept(FE_ALL_EXCEPT);
        }
        return PyLong_FromLong(flags);
    }
    if (block > run || (step_count == 1 && !longest)) {
        /* No buffer longer than a run, rounded up to whole steps. And a program of one step has nothing to keep
         * in the cache between steps: it is one call for the whole run, as a kernel is called by itself. */
        block = run + (BLOCK - run % BLOCK) % BLOCK;
    }
    if (longest) {
        /* A block holds each repeating stretch a whole number of times. Every stretch is a product of the shape's
         * last lengths, so each divides the longest. */
        block = block > longest ? block - block % longest : longest;
    }
    /* The buffers of the steps, then one for each array that repeats, each a block long. */
    int repeating[MOST_OPERANDS];
    int repeated = 0;
    for (int place = 0; place < arrays; place++) {
        repeating[place] = period[place] ? repeated++ : -1;
    }
    double *memory = buffers + repeated ? PyMem_Malloc((size_t)(buffers + repeated) * (size_t)block * sizeof(double)) : NULL;
    if (buffers + repeated && memory == NULL) {
        return PyErr_NoMemory();
    }
    double *copies = memory + (size_t)buffers * (size_t)block;
    double floats[MOST_OPERANDS];
    npy_intp position[MOST_DIMENSIONS] = {0};
    feclearexcept(FE_ALL_EXCEPT);
    if (!in_place && block >= run && outer > 0) {
        /* Every run is one block, and there is more than one: the lean walk. */
        int walked = program_runs(steps, step_count, reads, (int)arrays, numbers_given, period, repeating, memory, copies,
                                  block, run, outer, shape);
        PyMem_Free(memory);
        if (walked < 0) {
            return NULL;
        }
        int flags = fetestexcept(REPORTED);
        if (flags) {
            feclearexcept(FE_ALL_EXCEPT);
        }
        return PyLong_FromLong(flags);
    }
    for (;;) {
        for (int place = 0; place < numbers_given; place++) {
            floats[place] = numbers[place].is_array ? *(const double *)numbers[place].data : numbers[place].number;
        }
        for (int place = 0; place < arrays; place++) {
            if (period[place]) {
                /* The stretch as it is at this run, over and over for the length of a block. */
                double *copy = copies + (size_t)repeating[place] * (size_t)block;
                npy_intp held = period[place] < block ? period[place] : block;
                memcpy(copy, reads[place].data, (size_t)held * sizeof(double));
                while (held < block) {
                    npy_intp more = held < block - held ? held : block - held;
                    memcpy(copy + held, copy, (size_t)more * sizeof(double));
                    held += more;
                }
            }
        }
        for (npy_intp start = 0; start < run; start += block) {
            npy_intp length = run - start < block ? run - start : block;
            /* With the result written where an operand is, a block that is no multiple of eight is made as its
             * whole steps and then the elements left over, each a pass of every step; see `elementwise`. */
            npy_intp whole = in_place ? length - length % BLOCK : length;
            npy_intp lengths[2] = {whole, length - whole};
            npy_intp done = start;
            for (int part = 0; part < 2; part++) {
                if (lengths[part] == 0) {
                    continue;
                }
                for (Py_ssize_t each = 0; each < step_count; each++) {
                    const program_step *step = &steps[each];
                    int vectors = step->kernel->vectors, scalars = step->kernel->scalars;
                    uintptr_t integers[MOST_INTEGERS] = {0};
                    double values[MOST_NUMBERS] = {0};
                    for (int place = 0; place < vectors; place++) {
                        int from = step->reads[place];
                        if (from >= MOST_OPERANDS) {
                            integers[place] = (uintptr_t)(memory + (size_t)(from - MOST_OPERANDS) * (size_t)block);
                        } else if (period[from]) {
                            /* A block starts where every stretch starts, so the copy is read from its own start. */
                            integers[place] = (uintptr_t)(copies + (size_t)repeating[from] * (size_t)block + (done - start));
                        } else {
                            integers[place] = (uintptr_t)(reads[from].data + done * ELEMENT);
                        }
                    }
                    integers[vectors] = step->writes == RESULT ? (uintptr_t)(target->data + done * ELEMENT)
                                                               : (uintptr_t)(memory + (size_t)(step->writes - MOST_OPERANDS) * (size_t)block);
                    integers[vectors + 1] = (uintptr_t)lengths[part];
                    for (int place = 0; place < scalars; place++) {
                        values[place] = floats[step->numbers[place]];
                    }
                    ((elementwise_kernel)step->kernel->function)(
                        integers[0], integers[1], integers[2], integers[3], integers[4], integers[5], integers[6], integers[7],
                        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
                }
                done += lengths[part];
            }
        }
        /* Step the outer dimensions as an odometer: the last one first, carrying into the one before. */
        int axis = outer - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                for (int place = 0; place <= arrays + numbers_given; place++) {
                    if (reads[place].is_array) {
                        reads[place].data += reads[place].strides[axis];
                    }
                }
                break;
            }
            for (int place = 0; place <= arrays + numbers_given; place++) {
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
    PyMem_Free(memory);
    int raised = fetestexcept(REPORTED);
    if (raised) {
        feclearexcept(FE_ALL_EXCEPT);
    }
    return PyLong_FromLong(raised);
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

/* assign(part, values) -> True, or False when it was not done.
 *
 * `part[...] = values` for two float64 arrays of one shape and two dimensions or more, of which `values` lies in
 * one run and `part` need not: the rows of a result that was computed into memory of its own, copied to where they
 * belong. NumPy's assignment is a call of its copying loop for each row, 419 µs for 100,000 rows of two elements.
 * Nothing is done, and False returned, for arrays that are anything else or that may share memory. */
static PyObject *
fused_assign(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 2 || !PyArray_CheckExact(args[0]) || !PyArray_CheckExact(args[1])) {
        PyErr_SetString(PyExc_TypeError, "assign(part, values) takes two arrays");
        return NULL;
    }
    PyArrayObject *part = (PyArrayObject *)args[0], *values = (PyArrayObject *)args[1];
    int dimensions = PyArray_NDIM(part);
    if (dimensions < 2 || dimensions > MOST_DIMENSIONS || PyArray_NDIM(values) != dimensions || PyArray_TYPE(part) != NPY_DOUBLE
        || PyArray_TYPE(values) != NPY_DOUBLE || !PyArray_ISBEHAVED(part) || !PyArray_ISBEHAVED_RO(values)
        || !PyArray_IS_C_CONTIGUOUS(values) || !PyArray_SAMESHAPE(part, values) || PyArray_SIZE(part) == 0) {
        Py_RETURN_FALSE;
    }
    npy_intp shape[MOST_DIMENSIONS], position[MOST_DIMENSIONS] = {0};
    const npy_intp *strides = PyArray_STRIDES(part);
    const char *low = PyArray_BYTES(part), *high = PyArray_BYTES(part) + ELEMENT;
    for (int axis = 0; axis < dimensions; axis++) {
        shape[axis] = PyArray_DIM(part, axis);
        if (strides[axis] < 0) {
            Py_RETURN_FALSE;
        }
        high += strides[axis] * (shape[axis] - 1);
    }
    const char *from_low = PyArray_BYTES(values), *from_high = from_low + PyArray_NBYTES(values);
    if (strides[dimensions - 1] != ELEMENT || (low < from_high && from_low < high)) {
        Py_RETURN_FALSE; /* Rows that step, or arrays that may share memory: NumPy's assignment knows what to do. */
    }
    int last = dimensions - 1, along = dimensions - 2;
    npy_intp width = shape[last], rows = shape[along], pitch = strides[along];
    const double *source = (const double *)PyArray_BYTES(values);
    char *at = PyArray_BYTES(part);
    for (;;) {
        for (npy_intp row = 0; row < rows; row++) {
            double *to = (double *)(at + row * pitch);
            for (npy_intp each = 0; each < width; each++) {
                to[each] = source[each];
            }
            source += width;
        }
        int axis = along - 1;
        for (; axis >= 0; axis--) {
            if (++position[axis] < shape[axis]) {
                at += strides[axis];
                break;
            }
            at -= strides[axis] * (shape[axis] - 1);
            position[axis] = 0;
        }
        if (axis < 0) {
            break;
        }
    }
    Py_RETURN_TRUE;
}

static PyMethodDef methods[] = {
    {"assign", (PyCFunction)(void (*)(void))fused_assign, METH_FASTCALL,
     "assign(part, values) -> whether the rows of `values` were copied into `part`"},
    {"elementwise", (PyCFunction)(void (*)(void))fused_elementwise, METH_FASTCALL,
     "elementwise(kernel, operands, out) -> the exceptions raised, or DECLINED"},
    {"program", (PyCFunction)(void (*)(void))fused_program, METH_FASTCALL,
     "program(steps, arrays, operands, out, block) -> the exceptions raised, or DECLINED"},
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
