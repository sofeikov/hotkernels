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
/* The shortest inner run a program of several steps is made on. */
#define SHORTEST_RUN 32
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
        read->stepped = inner != ELEMENT && shape[dimensions - 1] != 1;
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
    for (int axis = 0; axis < dimensions; axis++) {
        shape[axis] = PyArray_DIM(out, axis);
        if (shape[axis] == 0) {
            return PyLong_FromLong(DECLINED);
        }
    }
    /* The operands, the kernel's arrays first, then `out`, then its numbers. */
    operand reads[MOST_INTEGERS + MOST_NUMBERS];
    for (int place = 0; place < vectors; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[1], place), dimensions, shape, 1, &reads[place]) || reads[place].stepped) {
            return PyLong_FromLong(DECLINED);
        }
    }
    operand *target = &reads[vectors];
    if (!read_operand((PyObject *)out, dimensions, shape, 1, target) || target->stepped) {
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

/* Moves a walk of rows to the next row: the position, and the place in memory of each of `count` arrays. */
static void
next_row(row_walk *walk, operand **arrays, char **rows, int count, int dimensions, const npy_intp *shape)
{
    int axis = dimensions - 2;
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
    walk->start += shape[dimensions - 1];
    walk->done = axis < 0;
}

/* A program on arrays that do not all lie the same way: rows too short to be worth a call each at more than one
 * pitch (a slice of a grid beside an array computed from one), or an array that steps over elements (a column of
 * a table).
 *
 * The shape is walked element by element in the order of its rows, a block at a time. An array that lies in that
 * order, one row after the other, is read where it is. Any other is gathered: the rows of the block are copied
 * into a buffer of the array's own, by a loop written here and not a call for each row, which is what made
 * gathering rows of 25 elements cost more than it saved. The result is written where it belongs when it lies in
 * the order of the rows, and otherwise into a buffer whose rows are copied out, as it is when it is one of the
 * arrays read, so that every row of a block is read before one is written.
 *
 * A copy for each array that lies otherwise, within the cache, in place of a pass over memory for each step.
 * Returns 0, or -1 with an error set. */
static int
program_flat(const program_step *steps, Py_ssize_t step_count, operand *reads, int arrays, const double *floats,
             int dimensions, const npy_intp *shape, npy_intp block, int buffers, int in_place)
{
    operand *target = &reads[arrays];
    int last = dimensions - 1;
    npy_intp width = shape[last], total = 1;
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
            gathered_at[place] = gathered_count;
            gathered[gathered_count++] = &reads[place];
        }
    }
    int scattered = !direct[arrays] || in_place;
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
    row_walk reading = {{0}, 0, 0}, writing = {{0}, 0, 0};
    for (npy_intp start = 0; start < total; start += block) {
        npy_intp length = total - start < block ? total - start : block;
        while (gathered_count && !reading.done && reading.start < start + length) {
            npy_intp from = reading.start > start ? reading.start : start;
            npy_intp upto = reading.start + width < start + length ? reading.start + width : start + length;
            for (int each = 0; each < gathered_count; each++) {
                npy_intp step = gathered[each]->strides[last];
                gather_run(copies + (size_t)each * (size_t)block + (from - start), rows[each] + (from - reading.start) * step,
                           upto - from, step);
            }
            if (reading.start + width > start + length) {
                break; /* The rest of this row is in the next block. */
            }
            next_row(&reading, gathered, rows, gathered_count, dimensions, shape);
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
            npy_intp upto = writing.start + width < start + length ? writing.start + width : start + length;
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
            if (writing.start + width > start + length) {
                break;
            }
            next_row(&writing, &target, &out_row, 1, dimensions, shape);
        }
    }
    PyMem_Free(memory);
    return 0;
}

/* The most of a walk that may be elements between rows, as a divisor: a quarter. More than that and the rows are
 * not walked as one stretch, since the steps would mostly compute what nobody asked for. */
#define MOST_BETWEEN 4

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
 * Returns 1 when the program was made this way, 0 when it does not apply and nothing was done, -1 with an error. */
static int
program_spanned(const program_step *steps, Py_ssize_t step_count, operand *reads, int arrays, operand *numbers,
                int numbers_given, int dimensions, const npy_intp *shape, npy_intp block, int buffers)
{
    operand *target = &reads[arrays];
    int last = dimensions - 1;
    npy_intp width = shape[last];
    if (dimensions < 2 || width < 2 || arrays == 0) {
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
        if (!agreed || common == 0 || step < extent || (step - extent) * MOST_BETWEEN > step) {
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
                if (out_step == ELEMENT) {
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
    for (int axis = 0; axis < dimensions; axis++) {
        shape[axis] = PyArray_DIM(out, axis);
        if (shape[axis] == 0) {
            return PyLong_FromLong(DECLINED);
        }
    }
    /* The operands as the walk sees them: the arrays, then `out`, then the numbers. */
    operand reads[2 * MOST_OPERANDS + 1];
    for (int place = 0; place < arrays; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[2], place), dimensions, shape, 1, &reads[place])) {
            return PyLong_FromLong(DECLINED);
        }
    }
    operand *target = &reads[arrays];
    if (!read_operand((PyObject *)out, dimensions, shape, 1, target)) {
        return PyLong_FromLong(DECLINED);
    }
    operand *numbers = &reads[arrays + 1];
    for (int place = 0; place < numbers_given; place++) {
        if (!read_operand(PyTuple_GET_ITEM(args[2], arrays + place), dimensions, shape, 0, &numbers[place])) {
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
    if (stepped || (run < SHORTEST_RUN && step_count > 1)) {
        /* Arrays a kernel cannot read where they lie, or runs so short that every step would be a call for a
         * handful of elements, at a pitch the arrays do not share: the shape is walked in the order of its rows
         * and the arrays that lie otherwise are gathered; see `program_flat`. Not rows of a few elements, though.
         * Gathering one is a copy for each array that lies otherwise, and for rows that short the copies cost what
         * NumPy's own passes do: on the Veros routines, whose innermost dimension is 26 long, isoneutral mixing
         * runs in 193 to 200 ms either way and the kinetic energy in 102 to 103 ms gathered against 97 to 100 ms
         * declined. Those are left to NumPy. A number that is not the same for every row cannot be gathered: a
         * call has one value for each number. */
        if (shape[dimensions - 1] < SHORTEST_RUN) {
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
        if (program_flat(steps, step_count, reads, (int)arrays, constants, dimensions, shape, block, buffers, in_place) < 0) {
            return NULL;
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

static PyMethodDef methods[] = {
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
