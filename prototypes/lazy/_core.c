/* The lazy layer's hot path in C: writing one node of the graph.
 *
 * This file adds nothing to what the layer does. Each function here is the
 * quick road of a function in array.py, graph.py or rules.py. A call met
 * before is answered from the layer's memories: the answer of `rules.plan`,
 * the line of the user's code, the error settings. A first call of the
 * common kinds (a ufunc, a reduction, a view function) is answered here as
 * well: the element type is NumPy's, asked as rules.py asks, and the shape
 * is by one of the three rules this file holds beside Python's, broadcasting,
 * the matrix product's and the axes of a reduction. What is found is kept in
 * the same memories, by the steps Python keeps it. Wherever this file is not
 * sure, it hands the call to the Python function, unchanged. The Python
 * functions are the definition; a node made here is, field for field, the
 * node they would have made.
 *
 * Why C at all: once nothing is asked twice, what is left of a node's time
 * in Python is the interpreter itself, about ten calls and five objects. Here
 * the same work is one call from the interpreter and the same five objects.
 *
 * The objects are the Python classes' own. `Node`, `LazyArray`, `_Cell` and
 * `_Step` are declared with `__slots__`, so each instance is a C structure
 * of object pointers at offsets the class's member descriptors give;
 * `setup()` reads the offsets, and an instance is made by allocating it and
 * storing its fields. No `__init__` runs, so each constructor below must
 * store what the Python `__init__` stores, and says where that is written.
 *
 * While the core is in use it also makes and frees the instances of `Node`,
 * `LazyArray` and `_Cell` itself: a freed one is kept and handed out again
 * (`Pool`), and the result of a call made at once is a lazy array that holds
 * NumPy's value and nothing else, with no node and no cell until something
 * asks for either (`lazy_holding`, `cell_of`).
 *
 * It is built by `python -m prototypes.lazy.build_core`, and the layer runs
 * without it.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <structmember.h> /* `PyMemberDef`, which Python.h itself brings only from 3.12 on. */
/* NumPy's own header, for the size and the element type of an array without a call into Python. */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
/* And the structure of a ufunc, for the number of inputs it takes. */
#include <numpy/ufuncobject.h>

/* Raised by one whenever this file and the Python it stands for change together, so that a module built from an
 * older source is not used: array.py asks for the number it was written against. */
#define CORE_VERSION 39

/* A field of an instance of a class with `__slots__`, by its offset. */
#define SLOT(object, offset) (*(PyObject **)((char *)(object) + (offset)))

/* The fields of `graph.Node`, in the order its dataclass declares them. */
enum { N_NUMBER, N_NAME, N_OPERATION, N_ARGS, N_KWARGS, N_SHAPE, N_DTYPE, N_ORIGIN, N_ERRSTATE, N_VALUE, N_FAILURE,
       N_KIND, N_STRICT, N_EXPOSED, N_SPEC, N_FIELDS };
static const char *const NODE_FIELDS[N_FIELDS] = {
    "number", "name", "operation", "args", "kwargs", "shape", "dtype", "origin", "errstate", "value", "failure",
    "kind", "strict", "exposed", "spec"};
/* The fields of `array.LazyArray`. */
enum { L_HELD, L_STEPS, L_SEEN, L_FIELDS };
static const char *const LAZY_FIELDS[L_FIELDS] = {"_held", "_steps", "_seen"};
/* The fields of `array._Cell`. */
enum { C_NODE, C_DOUBTFUL, C_DOUBTED, C_FIELDS };
static const char *const CELL_FIELDS[C_FIELDS] = {"node", "doubtful", "doubted"};
/* The fields of `array._Step`. */
enum { S_OPERATION, S_NAME, S_TAIL, S_KWARGS, S_SHAPE, S_DTYPE, S_ORIGIN, S_PLAIN, S_FIELDS };
static const char *const STEP_FIELDS[S_FIELDS] = {"operation", "name", "tail", "kwargs", "shape", "dtype", "origin", "plain"};

/* The fields of `array._Write` and of `array._Assign`. */
enum { W_STEPS, W_PERFORM, W_FIELDS };
static const char *const WRITE_FIELDS[W_FIELDS] = {"steps", "perform"};
enum { A_INDEX, A_FIELDS };
static const char *const ASSIGN_FIELDS[A_FIELDS] = {"index"};
/* The fields of `array._Select`. */
enum { X_TEMPLATE, X_WHOLE, X_FIELDS };
static const char *const SELECT_FIELDS[X_FIELDS] = {"template", "whole"};
static Py_ssize_t select_at[X_FIELDS];

/* The fields of `namespace.LazyCallable` and of `namespace.LazyNumPy`. */
enum { F_FUNCTION, F_STRICT, F_SOURCE, F_FIELDS };
static const char *const CALLABLE_FIELDS[F_FIELDS] = {"_function", "_strict", "_source"};
enum { M_MODULE, M_STRICT, M_ANSWERS, M_FIELDS };
static const char *const MODULE_FIELDS[M_FIELDS] = {"_module", "_strict", "_answers"};

static Py_ssize_t node_at[N_FIELDS], lazy_at[L_FIELDS], cell_at[C_FIELDS], step_at[S_FIELDS], write_at[W_FIELDS], assign_at[A_FIELDS],
    callable_at[F_FIELDS], module_at[M_FIELDS];

/* What `setup()` was given. Each is held for the life of the process. */
static PyTypeObject *Node, *Lazy, *Cell, *Step, *Ufunc, *Ndarray, *Write, *Assign, *StandInCall, *StandInModule;
static PyObject *fitting;            /* `array._fitting` */
static PyTypeObject *Select;         /* `array._Select` */
static PyTypeObject *Into;           /* `array._Into` */
static PyTypeObject *Through;        /* `array._AssignThrough`, with the fields of `_Select` */
static Py_ssize_t through_at[X_FIELDS];
static PyObject *assignable;         /* `array._assignable` */
static PyObject *types;              /* `rules._types` */
static PyObject *python_keep;        /* `rules.keep` */
static PyObject *python_spec_of;     /* `graph.spec_of` */
static PyTypeObject *Origin;         /* `graph.Origin` */
static Py_ssize_t origin_at[4];      /* Its fields: filename, lineno, module, registry. */
static PyObject *internal_files;     /* `graph._internal_files` */
/* The call the rules were last asked about here and said "not deferred" to: its operation, its arguments and its
 * keywords. The Python method that then takes the call asks `try_defer` first, and is told so at once instead of
 * the rules being asked again. Forgotten at the next call into this file, whatever it is. */
static PyObject *declined_operation, *declined_args, *declined_kwargs;
static PyObject *elementwise_kind;   /* `rules.ELEMENTWISE` */
static PyObject *matrix_kind;        /* `rules.MATRIX_PRODUCT` */
static PyObject *matmul;             /* `np.matmul` */
static PyObject *numpy_dtype;        /* `np.dtype` */
/* The identities of the `dtype` objects found to be NumPy's own for their type: `np.dtype(d.type) is d`. Such an
 * object lives as long as NumPy does, so what is noted of it stays true. */
static PyObject *own_dtypes;
static long long small_whole, large_whole; /* `rules.SMALL_WHOLE` and `rules.LARGE_WHOLE` */
static PyObject *view_functions;     /* `rules.VIEWS` */
static PyObject *reductions;         /* `rules.REDUCTIONS` */
static PyObject *positions;          /* `rules.POSITIONS` */
static PyObject *reduction_kind;     /* `rules.REDUCTION` */
static PyObject *numpy_ravel;        /* `np.ravel` */
static PyObject *numpy_empty;        /* `np.empty` */
static PyObject *no_size;            /* `rules.NO_SIZE` */
static PyObject *python_element_type; /* `rules._element_type` */
static PyObject *spec_numbers;       /* `graph._spec_numbers` */
static PyTypeObject *Signature;      /* `rules.Signature` */
static Py_ssize_t into_function_at;  /* Its one field, `function`. */
static PyObject *slots;              /* A tuple of `array._Slot(0)`, `_Slot(1)`, ...: the places of an index's pending arrays. */
static PyObject *write_kind;         /* `rules.WRITE` */
static PyObject *graph_names;        /* `graph.__dict__`: names a test or the layer rebinds are read from it each time. */
static PyObject *rules_names;        /* `rules.__dict__`, for the same reason. */
static PyObject *numbers;            /* `graph._numbers` */
static PyObject *pending;            /* `graph._pending` */
static PyObject *forget;             /* `graph._forget` */
static PyObject *counts;             /* `graph.COUNTS` */
static PyObject *origins;            /* `graph._origins` */
static PyObject *specs;              /* `graph._specs` */
static PyObject *plans;              /* `rules._plans` */
static PyObject *admissible;         /* `rules._admissible_specs` */
static PyObject *python_caller;      /* `graph.caller` */
static PyObject *python_errstate;    /* `graph.errstate` */
static PyObject *view_kind;          /* `rules.VIEW` */
static PyObject *sure_views;         /* `rules.SURE_VIEWS` */
static PyObject *fractional_scalars; /* `rules._FRACTIONAL_SCALARS` */
static PyObject *nonzeros;           /* `rules.NONZEROS` */
static PyObject *getitem;            /* `operator.getitem` */
static PyObject *count_nonzero;      /* `np.count_nonzero` */
static long long unknown_floor;      /* `graph.UNKNOWN_FLOOR`: a length at or above it is a stand-in. */

/* False while the layer is told to go the Python way everywhere. */
static int enabled = 0;
/* An operation that gives fewer elements than this, on operands that all have their values, is made at once instead
 * of being written as a node: `array.at_once_below`. 0 for never, which is how the layer starts. */
static Py_ssize_t small_limit = 0;
/* How many calls and writes were made at once that way, for the tools that measure a program. */
static long long made_at_once = 0, written_at_once = 0;
/* Calls that wrote into the memory of an operand nothing else read, since the module was loaded: `lent_count()`. */
static long long lent = 0;
/* The callables a fused kernel can make, and the element type it makes them on: `array.at_once_below` is given
 * both. A call of any size that is not one of these on that type is made at once when its operands have their
 * values: nothing above it could be computed together with it, so its node would only cost. NULL for none given,
 * where only small calls and views are made at once. */
static PyObject *fusible_calls = NULL, *fusible_dtype = NULL;
static PyObject *compute_small = NULL; /* `array._compute_for_small`: computes the node of a lazy array. */
static PyObject *at_once_kind;       /* `graph.AT_ONCE` */
static PyObject *handled_apart;      /* `array.HANDLED_APART`: the functions `__array_function__` has a way of its own for */
static PyObject *shaped_like;        /* `array.SHAPED_LIKE`: `np.zeros_like` and its kind */
static PyObject *fresh_functions;    /* `array.FRESH_FUNCTIONS`: functions whose one result never shares memory with an argument */
static PyObject *empty_string;       /* "", the kind of a node for an array that came from outside or was just made */
static PyObject *s_array;            /* "array", what a node with a value is called */
static PyObject *no_keywords;        /* The one empty dictionary the nodes made with a value have for their keywords */
static PyObject *s_transpose;        /* "transpose", the method behind `x.T` */
static PyObject *valued;             /* `array.VALUED`: what a lazy array that holds its value alone has in `_seen` */
static PyTypeObject *Generic;        /* `np.generic`, the type of every NumPy scalar */
static PyObject *in_place_ufuncs;    /* `array.IN_PLACE_UFUNCS`: the ufunc behind each in-place function of `operator` */
static PyObject *never_deferred;     /* `array.NEVER_DEFERRED`: functions with no rule, whose one result is a new array */
static PyObject *at_once_result;     /* `array._made_lazy`: what makes the result of a call made at once lazy */
static PyObject *out_names;          /* ("out",): the keyword names of a call that names its output and nothing else */
/* `dtype.kind` of the element types the layer defers, from `rules.DEFERRED_KINDS`, ended by a zero. */
#define MOST_KINDS 16
static char deferred_kinds[MOST_KINDS];
/* Returned by a quick road that does not apply: the caller then takes the Python road. Never seen from Python
 * except as the result of `try_defer`. */
static PyObject *Miss;
/* Constants made once. */
static PyObject *zero, *one, *empty_tuple;
static PyObject *s_node, *s_made, *s_deferred, *s_observer, *s_limit, *s_settings, *s_last, *s_registry, *s_name,
    *s_unknown, *s_shape, *s_dtype, *s_doubt, *s_call, *s_getitem, *s_basic_shape, *s_out, *s_dunder_name, *s_kind, *s_b,
    *s_setitem, *s_plan, *s_origin_limit, *s_resolve, *s_spec_limit, *s_plan_limit, *s_type_limit, *s_nin, *s_nout, *s_signature,
    *s_type, *s_fallback;

/* ---- Making and freeing the layer's objects. ---- */

/* Instances of one class that were freed and are kept for the next one made. A lazy array, its cell and its node
 * are made and dropped once per operation of the program, and what the interpreter does for an instance of a
 * class written in Python (find the memory, zero it, tell the garbage collector, and on the way out walk the
 * class's slots by their descriptors) costs more than a small NumPy call. Here an instance comes off this stack
 * and goes back to it. */
#define MOST_SPARE 128
typedef struct {
    PyTypeObject *type;          /* The class. */
    destructor before;           /* The class's own way of freeing an instance: put back when the core is put away. */
    traverseproc walked;         /* The class's own way of showing an instance to the collector, put back likewise. */
    PyObject *spare[MOST_SPARE]; /* Freed instances: memory of the class's size, with every slot empty. */
    int count;                   /* How many of them there are. */
} Pool;
static Pool lazy_pool, cell_pool, node_pool, step_pool;

/* A new instance with every slot empty, which the garbage collector does not know of yet: the caller fills the
 * slots and calls `PyObject_GC_Track`, unless the instance can be part of no cycle. */
static PyObject *
pool_new(Pool *pool)
{
    if (pool->count > 0) {
        PyObject *made = pool->spare[--pool->count];
        PyObject_Init(made, pool->type); /* One reference, and one more to the class, as for any instance. */
        return made;
    }
    PyObject *made = _PyObject_GC_New(pool->type);
    if (made != NULL) {
        memset((char *)made + sizeof(PyObject), 0, (size_t)pool->type->tp_basicsize - sizeof(PyObject));
    }
    return made;
}

/* What stands as `tp_dealloc` of the three classes while the core is in use. An instance of a class with
 * `__slots__` and no dictionary is its header and one object pointer per slot, which is what is cleared here; the
 * list of weak references is cleared first, by Python's own function, wherever the class keeps it. An instance of
 * a subclass arrives here from the subclass's own deallocator, which has done the same for its own slots and
 * frees the class itself: only the base's slots are cleared, and the memory goes back the subclass's way. */
static void
pooled_dealloc(PyObject *self)
{
    PyTypeObject *type = Py_TYPE(self);
    Pool *pool = type == Lazy ? &lazy_pool : type == Node ? &node_pool : type == Cell ? &cell_pool : type == Step ? &step_pool : NULL;
    PyTypeObject *base = pool != NULL ? type : PyType_IsSubtype(type, Lazy) ? Lazy : PyType_IsSubtype(type, Node) ? Node
        : PyType_IsSubtype(type, Cell) ? Cell : Step;
    PyObject_GC_UnTrack(self);
    /* For an instance of a subclass too: its deallocator leaves the weak references to a base that has them. */
    if (type->tp_weaklistoffset != 0 && *(PyObject **)((char *)self + type->tp_weaklistoffset) != NULL) {
        PyObject_ClearWeakRefs(self);
    }
    PyObject **slot = (PyObject **)((char *)self + sizeof(PyObject));
    PyObject **end = (PyObject **)((char *)self + base->tp_basicsize);
    for (; slot < end; slot++) {
        PyObject *held = *slot;
        if (held != NULL) {
            *slot = NULL;
            Py_DECREF(held);
        }
    }
    if (pool == NULL) {
        type->tp_free(self);
        return;
    }
    if (pool->count < MOST_SPARE) {
        pool->spare[pool->count++] = self;
    } else {
        PyObject_GC_Del(self);
    }
    Py_DECREF(type);
}

/* What stands as `tp_traverse` of the four classes while the core is in use: every slot, and the class, which an
 * instance of a class written in Python refers to. The classes' own walks their slots by the descriptors, once
 * per object the collector looks at, and a computation of a few thousand nodes is looked at often. An instance of
 * a subclass has more slots than the base, and goes the subclass's way, which ends here for the base's. */
static int
pooled_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyTypeObject *type = Py_TYPE(self);
    PyTypeObject *base = type == Lazy || type == Node || type == Cell || type == Step ? type
        : PyType_IsSubtype(type, Lazy) ? Lazy : PyType_IsSubtype(type, Node) ? Node : PyType_IsSubtype(type, Cell) ? Cell : Step;
    PyObject **slot = (PyObject **)((char *)self + sizeof(PyObject));
    PyObject **end = (PyObject **)((char *)self + base->tp_basicsize);
    PyObject **weak = base->tp_weaklistoffset > 0 ? (PyObject **)((char *)self + base->tp_weaklistoffset) : NULL;
    for (; slot < end; slot++) {
        if (slot != weak) {
            Py_VISIT(*slot);
        }
    }
    if (type == base) {
        Py_VISIT((PyObject *)type);
    }
    return 0;
}

/* Take the making and freeing of the four classes' instances, or give it back. */
static void
pools_in_use(int on)
{
    Pool *pools[4] = {&lazy_pool, &cell_pool, &node_pool, &step_pool};
    PyTypeObject *types[4] = {Lazy, Cell, Node, Step};
    for (int each = 0; each < 4; each++) {
        Pool *pool = pools[each];
        if (on && types[each]->tp_dealloc != pooled_dealloc) {
            if (pool->type != types[each]) {
                /* Another class than the one the spare instances are of: they go back to the allocator. */
                while (pool->count > 0) {
                    PyObject_GC_Del(pool->spare[--pool->count]);
                }
                pool->type = types[each];
            }
            pool->before = types[each]->tp_dealloc;
            pool->walked = types[each]->tp_traverse;
            types[each]->tp_dealloc = pooled_dealloc;
            types[each]->tp_traverse = pooled_traverse;
        } else if (!on && pool->type != NULL && pool->type->tp_dealloc == pooled_dealloc) {
            pool->type->tp_dealloc = pool->before;
            pool->type->tp_traverse = pool->walked;
        }
    }
}

/* `array._Cell(node)`, with its defaults. Returns a new reference. */
static PyObject *
new_cell(PyObject *node)
{
    PyObject *cell = pool_new(&cell_pool);
    if (cell == NULL) {
        return NULL;
    }
    SLOT(cell, cell_at[C_NODE]) = Py_NewRef(node);
    SLOT(cell, cell_at[C_DOUBTFUL]) = Py_NewRef(Py_None);
    SLOT(cell, cell_at[C_DOUBTED]) = Py_NewRef(Py_False);
    PyObject_GC_Track(cell);
    return cell;
}

/* `array.LazyArray(node)`: an array of its own, on a new cell. As `LazyArray.__init__` with no cell given, and
 * `_Cell.__init__` with its defaults. Steals nothing; returns a new reference. */
static PyObject *
new_lazy(PyObject *node)
{
    PyObject *cell = new_cell(node);
    if (cell == NULL) {
        return NULL;
    }
    PyObject *lazy = pool_new(&lazy_pool);
    if (lazy == NULL) {
        Py_DECREF(cell);
        return NULL;
    }
    SLOT(lazy, lazy_at[L_HELD]) = cell;
    SLOT(lazy, lazy_at[L_STEPS]) = Py_NewRef(empty_tuple);
    SLOT(lazy, lazy_at[L_SEEN]) = Py_NewRef(Py_None);
    PyObject_GC_Track(lazy);
    return lazy;
}

/* `array.LazyArray(node, cell, steps)`: a view. As `LazyArray.__init__` with a cell given. */
static PyObject *
new_view(PyObject *node, PyObject *cell, PyObject *steps)
{
    PyObject *seen = PyTuple_Pack(2, SLOT(cell, cell_at[C_NODE]), node);
    if (seen == NULL) {
        return NULL;
    }
    PyObject *lazy = pool_new(&lazy_pool);
    if (lazy == NULL) {
        Py_DECREF(seen);
        return NULL;
    }
    SLOT(lazy, lazy_at[L_HELD]) = Py_NewRef(cell);
    SLOT(lazy, lazy_at[L_STEPS]) = Py_NewRef(steps);
    SLOT(lazy, lazy_at[L_SEEN]) = seen;
    PyObject_GC_Track(lazy);
    return lazy;
}

/* `steps` with one more step at the end. A new reference. */
static PyObject *
steps_with(PyObject *steps, PyObject *step)
{
    Py_ssize_t count = PyTuple_GET_SIZE(steps);
    PyObject *all = PyTuple_New(count + 1);
    if (all == NULL) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyTuple_SET_ITEM(all, place, Py_NewRef(PyTuple_GET_ITEM(steps, place)));
    }
    PyTuple_SET_ITEM(all, count, Py_NewRef(step));
    return all;
}

/* The entries of `graph._origins` asked for last, by the code object and the instruction: a line of a loop asks
 * for its entry every turn, and finding it in the dictionary takes a key of two new numbers each time. An entry
 * here is the dictionary's own tuple, (code, origin, registry, module), and is checked as that one is. It is not
 * emptied with the dictionary: what it holds stays true, and it is bounded by its size. */
#define LINE_SLOTS 512
typedef struct {
    PyCodeObject *code; /* The code object, which `found` keeps alive. */
    int offset;         /* The instruction running in it. */
    PyObject *found;    /* The dictionary's entry, or NULL. */
} Line;
static Line lines[LINE_SLOTS];

/* Whether two parts of an index are the same whole number, or the same slice of whole numbers. */
static int
same_plain_part(PyObject *first, PyObject *second)
{
    if (first == second) {
        return PyLong_CheckExact(first) || first == Py_None || PySlice_Check(first);
    }
    if (PyLong_CheckExact(first) && PyLong_CheckExact(second)) {
        return PyObject_RichCompareBool(first, second, Py_EQ) == 1;
    }
    if (PySlice_Check(first) && PySlice_Check(second)) {
        PySliceObject *one = (PySliceObject *)first, *other = (PySliceObject *)second;
        PyObject *pairs[3][2] = {{one->start, other->start}, {one->stop, other->stop}, {one->step, other->step}};
        for (int each = 0; each < 3; each++) {
            PyObject *left = pairs[each][0], *right = pairs[each][1];
            if (left != right && !(PyLong_CheckExact(left) && PyLong_CheckExact(right)
                                   && PyObject_RichCompareBool(left, right, Py_EQ) == 1)) {
                return 0;
            }
            if (left == right && left != Py_None && !PyLong_CheckExact(left)) {
                return 0;
            }
        }
        return 1;
    }
    return 0;
}

/* Whether two indices are the same whole numbers and slices of whole numbers, in the same places: what
 * `array._same_index` says of such indices, and "no" for any other. */
static int
same_plain_index(PyObject *first, PyObject *second)
{
    if (!PyTuple_CheckExact(first) || !PyTuple_CheckExact(second)) {
        return !PyTuple_CheckExact(first) && !PyTuple_CheckExact(second) && same_plain_part(first, second);
    }
    Py_ssize_t count = PyTuple_GET_SIZE(first);
    if (count != PyTuple_GET_SIZE(second)) {
        return 0;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        if (!same_plain_part(PyTuple_GET_ITEM(first, place), PyTuple_GET_ITEM(second, place))) {
            return 0;
        }
    }
    return 1;
}

static PyObject *new_step(PyObject *operation, PyObject *name, PyObject *tail, PyObject *kwargs, PyObject *shape,
                          PyObject *dtype, PyObject *origin, int plain);
static PyObject *origin_of(PyFrameObject *frame);

/* The steps of a view taken at once while something observes the layer: those of the array viewed and one more,
 * for `operation` giving `value`, with `index` as its one argument after the array, or NULL for none. `plain` is
 * whether the argument is whole numbers and slices of them, as `array._is_plain_step` asks. Returns a new
 * reference to a tuple. */
static PyObject *
steps_now(PyObject *before, PyObject *operation, PyObject *name, PyObject *index, PyObject *value, int plain, PyFrameObject *frame)
{
    PyArrayObject *array = (PyArrayObject *)value;
    PyObject *origin = origin_of(frame);
    PyObject *tail = origin == NULL ? NULL : index == NULL ? Py_NewRef(empty_tuple) : PyTuple_Pack(1, index);
    PyObject *shape = tail ? PyTuple_New(PyArray_NDIM(array)) : NULL;
    for (int axis = 0; shape != NULL && axis < PyArray_NDIM(array); axis++) {
        PyObject *length = PyLong_FromSsize_t(PyArray_DIM(array, axis));
        if (length == NULL) {
            Py_CLEAR(shape);
            break;
        }
        PyTuple_SET_ITEM(shape, axis, length);
    }
    PyObject *kwargs = shape ? PyDict_New() : NULL;
    PyObject *step = kwargs ? new_step(operation, name, tail, kwargs, shape, (PyObject *)PyArray_DESCR(array), origin, plain) : NULL;
    PyObject *all = step ? steps_with(before, step) : NULL;
    Py_XDECREF(origin);
    Py_XDECREF(tail);
    Py_XDECREF(shape);
    Py_XDECREF(kwargs);
    Py_XDECREF(step);
    return all;
}

/* A lazy array that holds a value and nothing else: the form `LazyArray._cell` describes. `at_once` and `strict`
 * choose which of `array.VALUED` it keeps, which is the kind and the strictness its node will have. It refers to
 * an array or a NumPy scalar and to two constants, so it can be part of no cycle the garbage collector could
 * find, and is not made known to it; `cell_of` does that when the array gets its cell. Steals the reference to
 * `value`. Returns a new reference. */
static PyObject *
lazy_holding(PyObject *value, int at_once, int strict)
{
    PyObject *lazy = pool_new(&lazy_pool);
    if (lazy == NULL) {
        Py_DECREF(value);
        return NULL;
    }
    SLOT(lazy, lazy_at[L_HELD]) = value;
    SLOT(lazy, lazy_at[L_STEPS]) = Py_NewRef(empty_tuple);
    SLOT(lazy, lazy_at[L_SEEN]) = Py_NewRef(PyTuple_GET_ITEM(valued, (at_once ? 2 : 0) + (strict ? 1 : 0)));
    return lazy;
}

static PyObject *value_node(PyObject *value, PyObject *name, PyObject *shape, PyObject *dtype, PyObject *strict,
                            PyObject *spec, PyObject *origin, PyObject *kind);

/* The cell a lazy array looks into: the property `LazyArray._cell`. For an array that holds its value alone the
 * node and the cell are made now, as that property makes them. Borrowed, or NULL with an error set. */
static PyObject *
cell_of(PyObject *lazy)
{
    PyObject *held = SLOT(lazy, lazy_at[L_HELD]);
    if (held != NULL && Py_TYPE(held) == Cell) {
        return held;
    }
    PyObject *seen = SLOT(lazy, lazy_at[L_SEEN]);
    if (held == NULL || seen == NULL || !PyTuple_CheckExact(seen) || PyTuple_GET_SIZE(seen) != 2) {
        PyErr_SetString(PyExc_AttributeError, "a lazy array that was not set up has no cell");
        return NULL;
    }
    PyObject *shape = PyObject_GetAttr(held, s_shape);
    PyObject *dtype = shape ? PyObject_GetAttr(held, s_dtype) : NULL;
    /* `graph.leaf(value, kind=kind, strict=strict)`. */
    PyObject *node = dtype && PyTuple_CheckExact(shape)
        ? value_node(Py_NewRef(held), s_array, shape, dtype, PyTuple_GET_ITEM(seen, 1), zero, NULL, PyTuple_GET_ITEM(seen, 0)) : NULL;
    Py_XDECREF(shape);
    Py_XDECREF(dtype);
    PyObject *cell = node ? new_cell(node) : NULL;
    Py_XDECREF(node);
    if (cell == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "a value NumPy gave has a shape that is no tuple");
        }
        return NULL;
    }
    SLOT(lazy, lazy_at[L_HELD]) = cell;
    SLOT(lazy, lazy_at[L_SEEN]) = Py_NewRef(Py_None);
    Py_DECREF(held);
    Py_DECREF(seen);
    if (!PyObject_GC_IsTracked(lazy)) {
        PyObject_GC_Track(lazy);
    }
    return cell;
}

/* `array._Step(...)`, with every field given. */
static PyObject *
new_step(PyObject *operation, PyObject *name, PyObject *tail, PyObject *kwargs, PyObject *shape, PyObject *dtype,
         PyObject *origin, int plain)
{
    PyObject *step = pool_new(&step_pool);
    if (step == NULL) {
        return NULL;
    }
    SLOT(step, step_at[S_OPERATION]) = Py_NewRef(operation);
    SLOT(step, step_at[S_NAME]) = Py_NewRef(name);
    SLOT(step, step_at[S_TAIL]) = Py_NewRef(tail);
    SLOT(step, step_at[S_KWARGS]) = Py_NewRef(kwargs);
    SLOT(step, step_at[S_SHAPE]) = Py_NewRef(shape);
    SLOT(step, step_at[S_DTYPE]) = Py_NewRef(dtype);
    SLOT(step, step_at[S_ORIGIN]) = Py_NewRef(origin);
    SLOT(step, step_at[S_PLAIN]) = Py_NewRef(plain ? Py_True : Py_False);
    PyObject_GC_Track(step);
    return step;
}

/* The node a lazy array stands for now: `LazyArray._node`. For an array of its own that is its cell's node, read
 * here. For a view it is the property, which may make the view's nodes again. Returns a new reference. */
static PyObject *
node_of(PyObject *lazy)
{
    PyObject *steps = SLOT(lazy, lazy_at[L_STEPS]);
    if (steps != NULL && PyTuple_CheckExact(steps)) {
        PyObject *cell = cell_of(lazy);
        if (cell == NULL) {
            return NULL;
        }
        PyObject *node = SLOT(cell, cell_at[C_NODE]);
        if (node != NULL && PyTuple_GET_SIZE(steps) == 0) {
            return Py_NewRef(node);
        }
        /* A view whose cell still has the contents the view was last read of: the node it had then. */
        PyObject *seen = SLOT(lazy, lazy_at[L_SEEN]);
        if (node != NULL && seen != NULL && PyTuple_CheckExact(seen) && PyTuple_GET_SIZE(seen) == 2
            && PyTuple_GET_ITEM(seen, 0) == node) {
            return Py_NewRef(PyTuple_GET_ITEM(seen, 1));
        }
    }
    return PyObject_GetAttr(lazy, s_node);
}

/* Whether a value is one a lazy array can hold: an array or a NumPy scalar, or anything else with a `dtype`. */
static int
is_arraylike(PyObject *value)
{
    return Py_TYPE(value) == Ndarray || PyObject_TypeCheck(value, Generic) || PyObject_HasAttr(value, s_dtype);
}

/* How many elements a value has: an array's size, and 1 for a NumPy scalar. -1 for anything else. */
static Py_ssize_t
count_of(PyObject *value)
{
    if (PyArray_Check(value)) {
        return (Py_ssize_t)PyArray_SIZE((PyArrayObject *)value);
    }
    return PyObject_TypeCheck(value, Generic) ? 1 : -1;
}

/* Whether an exact `ndarray` is one the layer defers operations on: what `rules._admissible` says of it. */
static int
value_admissible(PyObject *value)
{
    PyArrayObject *array = (PyArrayObject *)value;
    return strchr(deferred_kinds, PyArray_DESCR(array)->kind) != NULL && PyArray_DESCR(array)->kind != 0
        && PyArray_SIZE(array) != 0;
}

static int has_value(PyObject *node);

/* The value a lazy array has now, without a node being made for an array that holds its value alone. A new
 * reference, with `*strict` set when the array is strict; `Miss` when it has none (it is pending, it failed, or
 * its memory was taken by a write); NULL with an error set. */
static PyObject *
value_of(PyObject *lazy, int *strict)
{
    PyObject *held = SLOT(lazy, lazy_at[L_HELD]);
    if (held != NULL && Py_TYPE(held) != Cell) {
        PyObject *seen = SLOT(lazy, lazy_at[L_SEEN]);
        if (seen == NULL || !PyTuple_CheckExact(seen) || PyTuple_GET_SIZE(seen) != 2) {
            return Miss;
        }
        *strict = *strict || PyTuple_GET_ITEM(seen, 1) == Py_True;
        return Py_NewRef(held);
    }
    PyObject *node = node_of(lazy);
    if (node == NULL) {
        return NULL;
    }
    if (Py_TYPE(node) != Node || !has_value(node)) {
        Py_DECREF(node);
        return Miss;
    }
    *strict = *strict || SLOT(node, node_at[N_STRICT]) == Py_True;
    PyObject *value = Py_NewRef(SLOT(node, node_at[N_VALUE]));
    Py_DECREF(node);
    return value;
}

/* ---- What `graph.deferred` looks up. ---- */

/* The line a frame is at, for a frame that is not remembered: what `graph.caller(frame)` does then. A frame of a
 * file known to be the user's gets its `Origin` made and remembered here, by the steps of that function; a frame of
 * the layer's or NumPy's, or of a file not looked at yet, is handed to it, which walks the stack. */
static PyObject *
new_origin(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    PyObject *names = NULL, *line = NULL, *identity = NULL, *offset = NULL, *key = NULL, *origin = NULL, *kept = NULL;
    PyObject *internal = PyDict_GetItemWithError(internal_files, code->co_filename);
    if (internal != Py_False) {
        Py_DECREF(code);
        return PyErr_Occurred() ? NULL : PyObject_CallOneArg(python_caller, (PyObject *)frame);
    }
    names = PyFrame_GetGlobals(frame);
    PyObject *module = PyDict_GetItemWithError(names, s_name);
    if (module == NULL) {
        module = s_unknown;
    }
    PyObject *registry = PyDict_GetItemWithError(names, s_registry);
    if (registry == NULL && !PyErr_Occurred()) {
        /* `names.setdefault("__warningregistry__", {})`. */
        PyObject *fresh = PyDict_New();
        registry = fresh ? PyDict_SetDefault(names, s_registry, fresh) : NULL;
        Py_XDECREF(fresh);
    }
    line = registry ? PyLong_FromLong(PyFrame_GetLineNumber(frame)) : NULL;
    identity = line ? PyLong_FromVoidPtr(code) : NULL;
    offset = identity ? PyLong_FromLong(PyFrame_GetLasti(frame)) : NULL;
    key = offset ? PyTuple_Pack(2, identity, offset) : NULL;
    origin = key ? Origin->tp_alloc(Origin, 0) : NULL;
    if (origin != NULL) {
        SLOT(origin, origin_at[0]) = Py_NewRef(code->co_filename);
        SLOT(origin, origin_at[1]) = Py_NewRef(line);
        SLOT(origin, origin_at[2]) = Py_NewRef(module);
        SLOT(origin, origin_at[3]) = Py_NewRef(registry);
        kept = PyTuple_Pack(4, (PyObject *)code, origin, registry, module);
        PyObject *limit = PyDict_GetItemWithError(graph_names, s_origin_limit);
        Py_ssize_t most = limit ? PyLong_AsSsize_t(limit) : 0;
        if (PyDict_GET_SIZE(origins) >= most) {
            PyDict_Clear(origins);
        }
        if (kept == NULL || PyErr_Occurred() || PyDict_SetItem(origins, key, kept) < 0) {
            Py_CLEAR(origin);
        }
    }
    Py_DECREF(code);
    Py_XDECREF(names);
    Py_XDECREF(line);
    Py_XDECREF(identity);
    Py_XDECREF(offset);
    Py_XDECREF(key);
    Py_XDECREF(kept);
    return origin;
}

/* The line of the user's code a frame is at: `graph.caller(frame)`. Found in the memory when the frame is one of
 * the user's and its module still has the name and the record of warnings it had, as `graph._remembered` checks;
 * otherwise the Python function is asked, which walks the stack and remembers. Returns a new reference. */
static PyObject *
origin_of(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    int offset_now = PyFrame_GetLasti(frame);
    Line *line = &lines[(((uintptr_t)code >> 4) ^ ((uintptr_t)offset_now * 31)) & (LINE_SLOTS - 1)];
    PyObject *known = line->code == code && line->offset == offset_now ? line->found : NULL; /* (code, origin, registry, module) */
    PyObject *found = known;
    if (found == NULL) {
        PyObject *identity = PyLong_FromVoidPtr(code); /* What `id(code)` gives. */
        PyObject *offset = PyLong_FromLong(offset_now);
        PyObject *key = (identity && offset) ? PyTuple_Pack(2, identity, offset) : NULL;
        Py_XDECREF(identity);
        Py_XDECREF(offset);
        if (key == NULL) {
            Py_DECREF(code);
            return NULL;
        }
        found = PyDict_GetItemWithError(origins, key); /* Borrowed. */
        Py_DECREF(key);
        if (found == NULL && PyErr_Occurred()) {
            Py_DECREF(code);
            return NULL;
        }
    }
    if (found != NULL && PyTuple_CheckExact(found) && PyTuple_GET_SIZE(found) == 4) {
        PyObject *names = PyFrame_GetGlobals(frame);
        PyObject *registry = PyDict_GetItemWithError(names, s_registry);
        PyObject *name = PyDict_GetItemWithError(names, s_name);
        if (name == NULL) {
            name = s_unknown;
        }
        int same = registry == PyTuple_GET_ITEM(found, 2) ? PyObject_RichCompareBool(name, PyTuple_GET_ITEM(found, 3), Py_EQ) : 0;
        Py_DECREF(names);
        if (same < 0 || PyErr_Occurred()) {
            Py_DECREF(code);
            return NULL;
        }
        if (same) {
            if (known == NULL) {
                /* Kept for the next time this instruction asks. The entry holds the code object, as the
                 * dictionary's does, so that no other code object can come to have its address. */
                Py_XSETREF(line->found, Py_NewRef(found));
                line->code = code;
                line->offset = offset_now;
            }
            PyObject *origin = Py_NewRef(PyTuple_GET_ITEM(found, 1));
            Py_DECREF(code);
            return origin;
        }
    }
    Py_DECREF(code);
    return new_origin(frame);
}

/* The `np.errstate` in effect: `graph.errstate()`. The remembered dictionary when NumPy's record of the settings
 * is the one it was made for; otherwise the Python function is asked. Returns a new reference. */
static PyObject *
errstate_now(void)
{
    PyObject *settings = PyDict_GetItemWithError(graph_names, s_settings);
    PyObject *last = PyDict_GetItemWithError(graph_names, s_last);
    if (settings != NULL && last != NULL && PyContextVar_CheckExact(settings) && PyTuple_CheckExact(last)
        && PyTuple_GET_SIZE(last) == 2) {
        PyObject *record = NULL;
        if (PyContextVar_Get(settings, NULL, &record) < 0) {
            return NULL;
        }
        /* Compared by address, which stays meaningful after the reference is given back: the context holds the record. */
        int same = record != NULL && record == PyTuple_GET_ITEM(last, 0);
        Py_XDECREF(record);
        if (same) {
            return Py_NewRef(PyTuple_GET_ITEM(last, 1));
        }
    }
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyObject_CallNoArgs(python_errstate);
}

/* Whether the pending set has reached `graph.PENDING_LIMIT`. 1, 0, or -1 with an error set. */
static int
at_the_limit(void)
{
    PyObject *limit = PyDict_GetItemWithError(graph_names, s_limit);
    if (limit == NULL) {
        return PyErr_Occurred() ? -1 : 1;
    }
    Py_ssize_t most = PyLong_AsSsize_t(limit);
    if (most == -1 && PyErr_Occurred()) {
        return -1;
    }
    return PySet_GET_SIZE(pending) >= most;
}

/* Whether a node has its value: computed, not failed, and the value not taken by a write made in place. */
static int
has_value(PyObject *node)
{
    return SLOT(node, node_at[N_OPERATION]) == Py_None && SLOT(node, node_at[N_FAILURE]) == Py_None
        && SLOT(node, node_at[N_VALUE]) != Py_None;
}

/* Whether an element type holds whole numbers or booleans: `dtype.kind in fuse.WHOLE_KINDS`, by NumPy's own
 * numbers for its types. One such number without dimensions is a number to a fused kernel beside float64. */
static int
whole_kind(PyObject *dtype)
{
    if (!PyArray_DescrCheck(dtype)) {
        return 0;
    }
    int number = ((PyArray_Descr *)dtype)->type_num;
    return PyTypeNum_ISINTEGER(number) || PyTypeNum_ISBOOL(number);
}

/* Whether a call can be made now instead of being written as a node: `small_limit` is set, every operand that is
 * a node has its value, no operand is a collection that might hold one, and the result is a view or has fewer than
 * `small_limit` elements. A view of an array that exists costs nothing to take. A small result is the other case:
 * NumPy makes such a call in less time than the layer needs to write its node, plan it and come back to it, and
 * there is nothing to fuse in it. A write is not asked about here; see `write_now`. */
static int
ready_now(PyObject *operation, PyObject *args, PyObject *kwargs, PyObject *shape, PyObject *dtype, PyObject *kind)
{
    if (small_limit == 0 || kind == write_kind || !PyTuple_CheckExact(args) || !PyTuple_CheckExact(shape)) {
        return 0;
    }
    if (kind != view_kind) {
        int small = 1;
        Py_ssize_t size = 1;
        for (Py_ssize_t axis = 0; small && axis < PyTuple_GET_SIZE(shape); axis++) {
            PyObject *length = PyTuple_GET_ITEM(shape, axis);
            if (!PyLong_CheckExact(length)) {
                small = 0; /* A length that is not known yet. */
                break;
            }
            Py_ssize_t count = PyLong_AsSsize_t(length);
            if (count < 0 || (count != 0 && size > small_limit / count)) {
                PyErr_Clear();
                small = 0;
                break;
            }
            size *= count;
        }
        if (!small || size >= small_limit) {
            /* A large result. It is still made now if no kernel could make this call: a comparison, a selection, a
             * matrix product, a call into a mathematics library, anything on another element type. What reads it
             * later reads an array either way. A call a kernel can make waits, so that what is written on top of
             * it can be computed with it. */
            if (fusible_calls == NULL) {
                return 0;
            }
            int same_type = dtype == fusible_dtype;
            for (Py_ssize_t place = 0; same_type && place < PyTuple_GET_SIZE(args); place++) {
                /* `abs(z)` of a complex array gives float64 and is no kernel's: every array read has to be of the type,
                 * or one whole number, which a kernel takes as it takes a number written in the program. */
                PyObject *arg = PyTuple_GET_ITEM(args, place);
                same_type = Py_TYPE(arg) != Node || SLOT(arg, node_at[N_DTYPE]) == fusible_dtype
                            || (PyTuple_CheckExact(SLOT(arg, node_at[N_SHAPE])) && PyTuple_GET_SIZE(SLOT(arg, node_at[N_SHAPE])) == 0
                                && whole_kind(SLOT(arg, node_at[N_DTYPE])));
            }
            if (same_type) {
                int waits = PySet_Contains(fusible_calls, operation);
                if (waits != 0) {
                    PyErr_Clear(); /* A callable that cannot be looked up is none of them, but is left to its node. */
                    return 0;
                }
            }
        }
    }
    for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(args); place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        if (Py_TYPE(arg) == Node) {
            if (!has_value(arg)) {
                return 0;
            }
        } else if (PyTuple_CheckExact(arg)) {
            /* An index or a shape. One that holds a node, or anything that might, is left to the node's own call. */
            for (Py_ssize_t part = 0; part < PyTuple_GET_SIZE(arg); part++) {
                PyObject *item = PyTuple_GET_ITEM(arg, part);
                if (Py_TYPE(item) == Node || Py_TYPE(item) == Lazy || PyTuple_Check(item) || PyList_Check(item)
                    || PyDict_Check(item)) {
                    return 0;
                }
            }
        } else if (Py_TYPE(arg) == Lazy || PyTuple_Check(arg) || PyList_Check(arg) || PyDict_Check(arg)) {
            return 0;
        }
    }
    if (kwargs != NULL) {
        Py_ssize_t position = 0;
        PyObject *value;
        while (PyDict_Next(kwargs, &position, NULL, &value)) {
            if (Py_TYPE(value) == Node || Py_TYPE(value) == Lazy || PyTuple_Check(value) || PyList_Check(value)
                || PyDict_Check(value)) {
                return 0;
            }
        }
    }
    return 1;
}

/* Make a call now, on its operands' values: what the node's own call would have been, made where the program
 * wrote it. NumPy's exception or warning, if any, is the program's there and then, as without the layer. Returns a
 * node that has the value (new reference), `Miss` when NumPy answered with something that is no array, or NULL. */
static PyObject *value_node(PyObject *value, PyObject *name, PyObject *shape, PyObject *dtype, PyObject *strict,
                            PyObject *spec, PyObject *origin, PyObject *kind);
static Py_ssize_t size_of(PyObject *shape);

static PyObject *
made_now(PyObject *operation, PyObject *name, PyObject *args, PyObject *kwargs, PyObject *shape, PyObject *dtype,
         PyObject *strict, PyObject *spec, PyObject *origin)
{
    Py_ssize_t count = PyTuple_GET_SIZE(args);
    PyObject *values = PyTuple_New(count);
    if (values == NULL) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        PyTuple_SET_ITEM(values, place, Py_NewRef(Py_TYPE(arg) == Node ? SLOT(arg, node_at[N_VALUE]) : arg));
    }
    PyObject *value = PyObject_Call(operation, values, kwargs != NULL && PyDict_GET_SIZE(kwargs) ? kwargs : NULL);
    Py_DECREF(values);
    if (value == NULL) {
        return NULL;
    }
    if (!is_arraylike(value)) {
        Py_DECREF(value);
        return Miss;
    }
    made_at_once++;
    if (size_of(shape) < 0) {
        /* A length that depended on values, such as how many elements a mask selects: the value has it now. */
        PyObject *settled = PyObject_GetAttr(value, s_shape);
        if (settled == NULL || !PyTuple_CheckExact(settled)) {
            Py_XDECREF(settled);
            Py_DECREF(value);
            return NULL;
        }
        PyObject *node = value_node(value, name, settled, dtype, strict, zero, origin, at_once_kind);
        Py_DECREF(settled);
        return node;
    }
    return value_node(value, name, shape, dtype, strict, spec, origin, at_once_kind);
}

/* A node that has its value: as `graph.leaf`, with the shape and the element type given, which are the value's.
 * Steals the reference to `value`, also when it fails. `origin` is the line for a view, whose step keeps it, and
 * NULL otherwise. Returns a new reference. */
static PyObject *
value_node(PyObject *value, PyObject *name, PyObject *shape, PyObject *dtype, PyObject *strict, PyObject *spec,
           PyObject *origin, PyObject *kind)
{
    PyObject *number = Py_TYPE(numbers)->tp_iternext(numbers);
    /* A node that has its value has no keywords and never gains any: every such node made here holds the same
     * empty dictionary, where one each would be an object made and freed for every call made at once. */
    PyObject *none = Py_NewRef(no_keywords);
    PyObject *node = number ? pool_new(&node_pool) : NULL;
    if (node == NULL) {
        Py_XDECREF(number);
        Py_DECREF(none);
        Py_DECREF(value);
        return NULL;
    }
    SLOT(node, node_at[N_NUMBER]) = number;
    SLOT(node, node_at[N_NAME]) = Py_NewRef(s_array);
    SLOT(node, node_at[N_OPERATION]) = Py_NewRef(Py_None);
    SLOT(node, node_at[N_ARGS]) = Py_NewRef(empty_tuple);
    SLOT(node, node_at[N_KWARGS]) = none;
    SLOT(node, node_at[N_SHAPE]) = Py_NewRef(shape);
    SLOT(node, node_at[N_DTYPE]) = Py_NewRef(dtype);
    SLOT(node, node_at[N_ORIGIN]) = Py_NewRef(origin != NULL ? origin : Py_None);
    SLOT(node, node_at[N_ERRSTATE]) = Py_NewRef(Py_None);
    SLOT(node, node_at[N_VALUE]) = value;
    SLOT(node, node_at[N_FAILURE]) = Py_NewRef(Py_None);
    SLOT(node, node_at[N_KIND]) = Py_NewRef(kind);
    SLOT(node, node_at[N_STRICT]) = Py_NewRef(strict);
    SLOT(node, node_at[N_EXPOSED]) = Py_NewRef(Py_False);
    SLOT(node, node_at[N_SPEC]) = Py_NewRef(spec);
    PyObject_GC_Track(node);
    PyObject *observer = PyDict_GetItemWithError(graph_names, s_observer);
    if (observer != NULL && observer != Py_None) {
        PyObject *heard = PyObject_CallMethodObjArgs(observer, s_made, node, name, NULL);
        if (heard == NULL) {
            Py_DECREF(node);
            return NULL;
        }
        Py_DECREF(heard);
    } else if (PyErr_Occurred()) {
        Py_DECREF(node);
        return NULL;
    }
    return node;
}

/* `graph.deferred(...)`: make a pending node and add it to the pending set, in the order the Python function
 * does: the number, the line, the error settings, the node, the set, the counter, the observer. `origin` may be
 * NULL or None for the line of `frame`. Returns a new reference. When `ready_now` says so the call is made instead,
 * and the node returned has its value. */
static PyObject *
deferred(PyObject *operation, PyObject *name, PyObject *args, PyObject *kwargs, PyObject *shape, PyObject *dtype,
         PyObject *kind, PyObject *strict, PyObject *origin, PyObject *spec, PyFrameObject *frame)
{
    PyObject *number = NULL, *line = NULL, *settings = NULL, *node = NULL, *reference = NULL;
    if (ready_now(operation, args, kwargs, shape, dtype, kind)) {
        /* A view keeps the line it was written at: its step is read again whenever the array is written into. */
        if (kind == view_kind) {
            line = (origin != NULL && origin != Py_None) ? Py_NewRef(origin) : origin_of(frame);
            if (line == NULL) {
                return NULL;
            }
        }
        node = made_now(operation, name, args, kwargs, shape, dtype, strict, spec, line);
        Py_CLEAR(line);
        if (node != Miss) {
            return node;
        }
        node = NULL; /* Not an array: written as a node after all, which makes the call again when it is computed. */
    }
    number = Py_TYPE(numbers)->tp_iternext(numbers);
    if (number == NULL) {
        goto fail;
    }
    line = (origin != NULL && origin != Py_None) ? Py_NewRef(origin) : origin_of(frame);
    if (line == NULL) {
        goto fail;
    }
    settings = errstate_now();
    if (settings == NULL) {
        goto fail;
    }
    node = pool_new(&node_pool);
    if (node == NULL) {
        goto fail;
    }
    /* As the dataclass's `__init__`, with `value` and `failure` None and `exposed` False. */
    SLOT(node, node_at[N_NUMBER]) = number;
    SLOT(node, node_at[N_ORIGIN]) = line;
    SLOT(node, node_at[N_ERRSTATE]) = settings;
    number = line = settings = NULL;
    SLOT(node, node_at[N_NAME]) = Py_NewRef(name);
    SLOT(node, node_at[N_OPERATION]) = Py_NewRef(operation);
    SLOT(node, node_at[N_ARGS]) = Py_NewRef(args);
    SLOT(node, node_at[N_KWARGS]) = Py_NewRef(kwargs);
    SLOT(node, node_at[N_SHAPE]) = Py_NewRef(shape);
    SLOT(node, node_at[N_DTYPE]) = Py_NewRef(dtype);
    SLOT(node, node_at[N_VALUE]) = Py_NewRef(Py_None);
    SLOT(node, node_at[N_FAILURE]) = Py_NewRef(Py_None);
    SLOT(node, node_at[N_KIND]) = Py_NewRef(kind);
    SLOT(node, node_at[N_STRICT]) = Py_NewRef(strict);
    SLOT(node, node_at[N_EXPOSED]) = Py_NewRef(Py_False);
    SLOT(node, node_at[N_SPEC]) = Py_NewRef(spec);
    PyObject_GC_Track(node);
    reference = PyWeakref_NewRef(node, forget);
    if (reference == NULL || PySet_Add(pending, reference) < 0) {
        goto fail;
    }
    Py_CLEAR(reference);
    PyObject *count = PyObject_GetAttr(counts, s_deferred);
    PyObject *more = count ? PyNumber_Add(count, one) : NULL;
    int counted = more ? PyObject_SetAttr(counts, s_deferred, more) : -1;
    Py_XDECREF(count);
    Py_XDECREF(more);
    if (counted < 0) {
        goto fail;
    }
    PyObject *observer = PyDict_GetItemWithError(graph_names, s_observer);
    if (observer != NULL && observer != Py_None) {
        PyObject *heard = PyObject_CallMethodObjArgs(observer, s_made, node, name, NULL);
        if (heard == NULL) {
            goto fail;
        }
        Py_DECREF(heard);
    } else if (PyErr_Occurred()) {
        goto fail;
    }
    return node;
fail:
    Py_XDECREF(number);
    Py_XDECREF(line);
    Py_XDECREF(settings);
    Py_XDECREF(reference);
    Py_XDECREF(node);
    return NULL;
}

/* ---- Keeping what a first call found, by the steps of the Python functions that keep it. Each memory has a bound,
 * read where Python reads it; a memory that is full is Python's to empty, so the Python function is called then. ---- */

/* Whether a memory has room for one more entry under the bound of that name in a module's namespace. */
static int
has_room(PyObject *memory, PyObject *names, PyObject *bound)
{
    PyObject *limit = PyDict_GetItemWithError(names, bound);
    Py_ssize_t most = limit ? PyLong_AsSsize_t(limit) : 0;
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    return PyDict_GET_SIZE(memory) < most;
}

/* The number that stands for a shape, every length of it known, and an element type: `graph.spec_of(shape, dtype)`.
 * A new reference. */
static PyObject *
spec_number(PyObject *shape, PyObject *dtype)
{
    PyObject *identity = PyLong_FromVoidPtr(dtype);
    PyObject *key = identity ? PyTuple_Pack(2, shape, identity) : NULL;
    Py_XDECREF(identity);
    if (key == NULL) {
        return NULL;
    }
    PyObject *number = NULL;
    PyObject *found = PyDict_GetItemWithError(specs, key); /* (number, dtype), borrowed. */
    if (found != NULL) {
        number = Py_NewRef(PyTuple_GET_ITEM(found, 0));
    } else if (!PyErr_Occurred()) {
        if (has_room(specs, graph_names, s_spec_limit)) {
            number = Py_TYPE(spec_numbers)->tp_iternext(spec_numbers);
            PyObject *kept = number ? PyTuple_Pack(2, number, dtype) : NULL;
            if (kept == NULL || PyDict_SetItem(specs, key, kept) < 0) {
                Py_CLEAR(number);
            }
            Py_XDECREF(kept);
        } else {
            number = PyObject_CallFunctionObjArgs(python_spec_of, shape, dtype, NULL);
        }
    }
    Py_DECREF(key);
    return number;
}

/* `rules.keep(key, shape, dtype, kind, types)`: keep the answer of a ufunc's first call under its key, and under
 * `types`, if that is given, the element type for the operands' types. A new reference to the signature kept. */
static PyObject *
keep_answer(PyObject *key, PyObject *shape, PyObject *dtype, PyObject *kind, PyObject *types_key)
{
    if (!has_room(plans, rules_names, s_plan_limit) || (types_key != NULL && !has_room(types, rules_names, s_type_limit))) {
        return PyObject_CallFunctionObjArgs(python_keep, key, shape, dtype, kind, types_key ? types_key : Py_None, NULL);
    }
    if (types_key != NULL) {
        PyObject *remembered = PyTuple_Pack(2, dtype, kind);
        int kept = remembered ? PyDict_SetItem(types, types_key, remembered) : -1;
        Py_XDECREF(remembered);
        if (kept < 0) {
            return NULL;
        }
    }
    /* `rules._remember`: the signature with the number of its shape and element type. */
    PyObject *number = spec_number(shape, dtype);
    PyObject *fields = number ? PyTuple_Pack(4, shape, dtype, kind, number) : NULL;
    /* `tuple.__new__(Signature, fields)`, which is what a named tuple's own `__new__` calls. */
    PyObject *given = fields ? PyTuple_Pack(1, fields) : NULL;
    PyObject *signature = given ? PyTuple_Type.tp_new(Signature, given, NULL) : NULL;
    Py_XDECREF(number);
    Py_XDECREF(fields);
    Py_XDECREF(given);
    if (signature != NULL && PyDict_SetItem(plans, key, signature) < 0) {
        Py_CLEAR(signature);
    }
    return signature;
}

/* ---- The key of `rules.plan`. Each function returns a new reference, `Miss` (borrowed) for a call whose key
 * this file does not write, or NULL with an error set. The tuples must equal the ones `rules._key` writes, since
 * the answers found by them were stored under those. ---- */

/* The stand-ins a call brings, in the order they are met: what `rules._key` keeps in `marks`. The placeholder of
 * the stand-in at place k is -k-1. A call with more than fit here is left to Python. */
#define MOST_MARKS 8
typedef struct {
    int count;
    PyObject *length[MOST_MARKS]; /* Borrowed: each is held by a shape of the call, or made here and kept in `made`. */
    PyObject *made[MOST_MARKS];   /* Owned: the stand-ins this file made for the masks of an index. */
    int made_count;
} Marks;

static void
marks_clear(Marks *marks)
{
    for (int each = 0; each < marks->made_count; each++) {
        Py_DECREF(marks->made[each]);
    }
    marks->count = marks->made_count = 0;
}

/* The place of a stand-in among a call's marks, added if it is new: `marks.setdefault(length, -len(marks) - 1)`.
 * -1 when there is no room, or with an error set. */
static int
mark_of(Marks *marks, PyObject *length)
{
    for (int each = 0; each < marks->count; each++) {
        int same = PyObject_RichCompareBool(marks->length[each], length, Py_EQ);
        if (same != 0) {
            return same < 0 ? -1 : each;
        }
    }
    if (marks->count == MOST_MARKS) {
        return -1;
    }
    marks->length[marks->count] = length;
    return marks->count++;
}

/* Whether a length is a stand-in: at or above `graph.UNKNOWN_FLOOR`. */
static int
is_stand_in(PyObject *length)
{
    int overflow = 0;
    long long value = PyLong_AsLongLongAndOverflow(length, &overflow);
    return overflow > 0 || (overflow == 0 && value >= unknown_floor);
}

/* The number of a node's shape and element type: `node.spec` if it has one. For a shape with a stand-in, and
 * with `marks`, what `rules._marked_part` writes: the number of the shape with a placeholder for each stand-in,
 * if that shape was met before. */
static PyObject *
spec_part(PyObject *node, Marks *marks)
{
    PyObject *spec = SLOT(node, node_at[N_SPEC]);
    if (spec == NULL || !PyLong_CheckExact(spec)) {
        return Miss;
    }
    int known = PyObject_IsTrue(spec);
    if (known != 0) {
        return known < 0 ? NULL : Py_NewRef(spec);
    }
    PyObject *shape = SLOT(node, node_at[N_SHAPE]);
    Py_ssize_t dimensions = PyTuple_GET_SIZE(shape);
    int unknown = 0;
    PyObject *written = PyTuple_New(dimensions);
    if (written == NULL) {
        return NULL;
    }
    for (Py_ssize_t axis = 0; axis < dimensions; axis++) {
        PyObject *length = PyTuple_GET_ITEM(shape, axis);
        if (!is_stand_in(length)) {
            PyTuple_SET_ITEM(written, axis, Py_NewRef(length));
            continue;
        }
        if (marks == NULL) {
            Py_DECREF(written);
            return Miss;
        }
        int place = mark_of(marks, length);
        PyObject *placeholder = place < 0 ? NULL : PyLong_FromLong(-place - 1);
        if (placeholder == NULL) {
            Py_DECREF(written);
            return PyErr_Occurred() ? NULL : Miss;
        }
        PyTuple_SET_ITEM(written, axis, placeholder);
        unknown = 1;
    }
    PyObject *identity = PyLong_FromVoidPtr(SLOT(node, node_at[N_DTYPE]));
    PyObject *key = identity ? PyTuple_Pack(2, written, identity) : NULL;
    Py_DECREF(written);
    Py_XDECREF(identity);
    if (key == NULL) {
        return NULL;
    }
    PyObject *found = PyDict_GetItemWithError(specs, key); /* (number, dtype), borrowed. */
    Py_DECREF(key);
    if (found == NULL && PyErr_Occurred()) {
        return NULL;
    }
    if (unknown) {
        /* Under placeholders: a shape not met before is left to Python. */
        return found ? Py_NewRef(PyTuple_GET_ITEM(found, 0)) : Miss;
    }
    /* A node that was not asked for its number yet, as a new leaf is: it gets the one its shape and type have,
     * as `graph.spec` would give it, and Python gives them one if they have none. */
    PyObject *number = found ? Py_NewRef(PyTuple_GET_ITEM(found, 0)) : spec_number(shape, SLOT(node, node_at[N_DTYPE]));
    if (number == NULL) {
        return NULL;
    }
    Py_SETREF(SLOT(node, node_at[N_SPEC]), Py_NewRef(number));
    return number;
}

/* What `rules._array_part` writes for an exact `ndarray`: the number of its shape and `dtype` object, negated. */
static PyObject *
ndarray_part(PyObject *array)
{
    PyObject *shape = PyObject_GetAttr(array, s_shape);
    PyObject *dtype = shape ? PyObject_GetAttr(array, s_dtype) : NULL;
    PyObject *identity = dtype ? PyLong_FromVoidPtr(dtype) : NULL;
    PyObject *key = identity ? PyTuple_Pack(2, shape, identity) : NULL;
    PyObject *number = NULL, *negated = NULL;
    if (key != NULL) {
        PyObject *found = PyDict_GetItemWithError(specs, key); /* (number, dtype), borrowed. */
        /* A shape and type not met before get their number from Python, as `rules._array_part` has it. */
        number = found ? Py_NewRef(PyTuple_GET_ITEM(found, 0)) : PyErr_Occurred() ? NULL : spec_number(shape, dtype);
        negated = number ? PyNumber_Negative(number) : NULL;
    }
    Py_XDECREF(shape);
    Py_XDECREF(dtype);
    Py_XDECREF(identity);
    Py_XDECREF(key);
    Py_XDECREF(number);
    return negated;
}

/* What `rules._setting_part` writes: the type with the value for a plain value, item by item for a tuple, and
 * the type `type` with a class. */
static PyObject *
setting_part(PyObject *value)
{
    if (PyLong_CheckExact(value) || PyBool_Check(value) || PyFloat_CheckExact(value) || PyComplex_CheckExact(value)
        || PyUnicode_CheckExact(value) || value == Py_None || value == Py_Ellipsis) {
        return PyTuple_Pack(2, (PyObject *)Py_TYPE(value), value);
    }
    if (PyTuple_CheckExact(value)) {
        Py_ssize_t count = PyTuple_GET_SIZE(value);
        PyObject *items = PyTuple_New(count);
        if (items == NULL) {
            return NULL;
        }
        for (Py_ssize_t place = 0; place < count; place++) {
            if (Py_EnterRecursiveCall(" in a tuple of settings")) {
                Py_DECREF(items);
                return NULL;
            }
            PyObject *part = setting_part(PyTuple_GET_ITEM(value, place));
            Py_LeaveRecursiveCall();
            if (part == NULL || part == Miss) {
                Py_DECREF(items);
                return part;
            }
            PyTuple_SET_ITEM(items, place, part);
        }
        PyObject *whole = PyTuple_Pack(2, (PyObject *)&PyTuple_Type, items);
        Py_DECREF(items);
        return whole;
    }
    if (Py_IS_TYPE(value, &PyType_Type)) {
        return PyTuple_Pack(2, (PyObject *)&PyType_Type, value);
    }
    return Miss;
}

/* What `rules.operand_part` writes for an operand of a ufunc. A whole NumPy scalar is left to Python. */
static PyObject *
operand_part(PyObject *operand, Marks *marks)
{
    PyTypeObject *kind = Py_TYPE(operand);
    if (kind == Node) {
        return spec_part(operand, marks);
    }
    if (kind == &PyFloat_Type || kind == &PyComplex_Type) {
        return Py_NewRef((PyObject *)kind);
    }
    if (kind == &PyLong_Type || kind == &PyBool_Type) {
        return PyTuple_Pack(2, (PyObject *)kind, operand);
    }
    if (kind == Ndarray) {
        return ndarray_part(operand);
    }
    int fractional = PySet_Contains(fractional_scalars, (PyObject *)kind);
    return fractional < 0 ? NULL : fractional ? Py_NewRef((PyObject *)kind) : Miss;
}

/* What `rules._index_key` writes for `x[index]`: the node, whether the index is one item, then the items. A
 * pending mask brings its own stand-in, the length it selects, as a mark. */
static PyObject *
index_key(PyObject *unwrapped, Marks *marks)
{
    if (PyTuple_GET_SIZE(unwrapped) != 2 || Py_TYPE(PyTuple_GET_ITEM(unwrapped, 0)) != Node) {
        return Miss;
    }
    PyObject *index = PyTuple_GET_ITEM(unwrapped, 1);
    int plain = !PyTuple_CheckExact(index);
    if (plain && PyTuple_Check(index)) {
        return Miss;
    }
    PyObject *single[1] = {index};
    PyObject **items = plain ? single : &PyTuple_GET_ITEM(index, 0);
    Py_ssize_t count = plain ? 1 : PyTuple_GET_SIZE(index);
    PyObject *key = PyTuple_New(3 + count);
    PyObject *part = key ? spec_part(PyTuple_GET_ITEM(unwrapped, 0), marks) : NULL;
    if (part == NULL || part == Miss) {
        Py_XDECREF(key);
        return part;
    }
    PyTuple_SET_ITEM(key, 0, Py_NewRef(getitem));
    PyTuple_SET_ITEM(key, 1, part);
    PyTuple_SET_ITEM(key, 2, Py_NewRef(plain ? Py_True : Py_False));
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *item = items[place];
        if (Py_TYPE(item) == Node) {
            part = spec_part(item, marks);
            if (part != NULL && part != Miss) {
                PyObject *kind = PyObject_GetAttr(SLOT(item, node_at[N_DTYPE]), s_kind);
                int is_mask = kind ? PyObject_RichCompareBool(kind, s_b, Py_EQ) : -1;
                Py_XDECREF(kind);
                if (is_mask > 0) {
                    /* `graph.unknown_length(item)`: the floor plus the mask's number. */
                    PyObject *floor = PyLong_FromLongLong(unknown_floor);
                    PyObject *length = floor ? PyNumber_Add(floor, SLOT(item, node_at[N_NUMBER])) : NULL;
                    Py_XDECREF(floor);
                    if (length == NULL || marks->made_count == MOST_MARKS) {
                        Py_XDECREF(length);
                        is_mask = PyErr_Occurred() ? -1 : -2;
                    } else {
                        marks->made[marks->made_count++] = length;
                        if (mark_of(marks, length) < 0) {
                            is_mask = PyErr_Occurred() ? -1 : -2;
                        }
                    }
                }
                if (is_mask < 0) {
                    Py_DECREF(part);
                    part = is_mask == -1 ? NULL : Miss;
                }
            }
        } else if (PySlice_Check(item)) {
            PySliceObject *slice = (PySliceObject *)item;
            PyObject *start = setting_part(slice->start);
            PyObject *stop = start && start != Miss ? setting_part(slice->stop) : start;
            PyObject *step = stop && stop != Miss ? setting_part(slice->step) : stop;
            if (step == NULL || step == Miss) {
                part = step;
                if (start != NULL && start != Miss) {
                    Py_DECREF(start);
                }
                if (stop != NULL && stop != Miss) {
                    Py_DECREF(stop);
                }
            } else {
                part = PyTuple_Pack(4, (PyObject *)&PySlice_Type, start, stop, step);
                Py_DECREF(start);
                Py_DECREF(stop);
                Py_DECREF(step);
            }
        } else {
            part = setting_part(item);
        }
        if (part == NULL || part == Miss) {
            Py_DECREF(key);
            return part;
        }
        PyTuple_SET_ITEM(key, 3 + place, part);
    }
    return key;
}

/* What `rules._key` writes for a call. Without `marks`, a call with a length that is not known, and an index,
 * are not written; with it they are, under placeholders. */
static PyObject *
key_of(PyObject *rule, PyObject *unwrapped, PyObject *kwargs, Marks *marks)
{
    Py_ssize_t count = PyTuple_GET_SIZE(unwrapped);
    Py_ssize_t named = kwargs ? PyDict_GET_SIZE(kwargs) : 0;
    int is_ufunc = Py_TYPE(rule) == Ufunc;
    if (is_ufunc && named) {
        return Miss;
    }
    if (!is_ufunc) {
        if (rule == getitem) {
            return marks ? index_key(unwrapped, marks) : Miss;
        }
        if (rule == count_nonzero) {
            return Miss;
        }
        int positions = PySet_Contains(nonzeros, rule);
        if (positions != 0) {
            if (positions < 0) {
                PyErr_Clear(); /* A callable without a hash is in no set of functions. */
            } else {
                return Miss;
            }
        }
    }
    PyObject *key = PyTuple_New(1 + count + 2 * named);
    if (key == NULL) {
        return NULL;
    }
    PyTuple_SET_ITEM(key, 0, Py_NewRef(rule));
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(unwrapped, place);
        PyObject *part;
        if (is_ufunc) {
            part = operand_part(arg, marks);
        } else if (Py_TYPE(arg) == Node) {
            part = spec_part(arg, marks);
        } else if (Py_TYPE(arg) == Ndarray) {
            part = ndarray_part(arg);
        } else {
            part = setting_part(arg);
        }
        if (part == NULL || part == Miss) {
            Py_DECREF(key);
            return part;
        }
        PyTuple_SET_ITEM(key, 1 + place, part);
    }
    Py_ssize_t position = 0, place = 1 + count;
    PyObject *name, *value;
    while (named && PyDict_Next(kwargs, &position, &name, &value)) {
        PyObject *part = setting_part(value);
        if (part == NULL || part == Miss) {
            Py_DECREF(key);
            return part;
        }
        PyTuple_SET_ITEM(key, place++, Py_NewRef(name));
        PyTuple_SET_ITEM(key, place++, part);
    }
    return key;
}

/* NumPy's broadcasting, for `count` shapes given as their items and their numbers of dimensions: the shapes are
 * lined up at their last axis; along each axis the lengths are one number, apart from those that are 1. A new
 * tuple; `Miss` for a length of zero, which is nothing to compute on, and for shapes that do not broadcast, which
 * NumPy has an error for. */
static PyObject *
broadcast(PyObject **const *items, const Py_ssize_t *sizes, Py_ssize_t count)
{
    Py_ssize_t dimensions = 0;
    for (Py_ssize_t place = 0; place < count; place++) {
        dimensions = sizes[place] > dimensions ? sizes[place] : dimensions;
    }
    PyObject *shape = PyTuple_New(dimensions);
    if (shape == NULL) {
        return NULL;
    }
    for (Py_ssize_t back = 1; back <= dimensions; back++) {
        PyObject *length = one;
        for (Py_ssize_t place = 0; place < count; place++) {
            if (back > sizes[place]) {
                continue;
            }
            PyObject *here = items[place][sizes[place] - back];
            Py_ssize_t value = PyLong_AsSsize_t(here);
            if (value == -1 && PyErr_Occurred()) {
                Py_DECREF(shape);
                return NULL;
            }
            if (value == 0) {
                Py_DECREF(shape);
                return Miss;
            }
            if (value == 1) {
                continue;
            }
            if (length == one) {
                length = here;
            } else {
                int equal = PyObject_RichCompareBool(length, here, Py_EQ);
                if (equal <= 0) {
                    Py_DECREF(shape);
                    return equal < 0 ? NULL : Miss;
                }
            }
        }
        PyTuple_SET_ITEM(shape, dimensions - back, Py_NewRef(length));
    }
    return shape;
}

/* Whether a `dtype` object is NumPy's own for its type, `np.dtype(d.type) is d`, as `rules._resolver_types` asks:
 * another byte order and metadata are other objects. 1, 0, or -1 with an error set. */
static int
is_own_dtype(PyObject *dtype)
{
    PyObject *identity = PyLong_FromVoidPtr(dtype);
    if (identity == NULL) {
        return -1;
    }
    int own = PyDict_Contains(own_dtypes, identity);
    if (own == 0) {
        PyObject *type = PyObject_GetAttr(dtype, s_type);
        PyObject *usual = type ? PyObject_CallOneArg(numpy_dtype, type) : NULL;
        Py_XDECREF(type);
        own = usual == NULL ? -1 : usual == dtype;
        Py_XDECREF(usual);
        if (own == 1 && PyDict_SetItem(own_dtypes, identity, Py_True) < 0) {
            own = -1;
        }
    }
    Py_DECREF(identity);
    return own;
}

/* The answer for the first call of a ufunc on its operands, where `rules._plan` would get the element type from
 * NumPy's resolver: an element-by-element call or a matrix product, on arrays of NumPy's own element types and
 * numbers with a fraction. The type is the one remembered for the operands' types (`rules._types`), or the
 * resolver's, asked here as `rules._resolved_type` asks it. The shape is by the two rules of a shape this file
 * holds, broadcasting and the matrix product's, each held to Python's by a test. `key` is the call's plain key,
 * under which Python then keeps the answer. A new reference; `Miss` wherever `rules._plan` would do anything
 * else: decline, ask by a call on arrays of one element, or let NumPy raise. */
static PyObject *
first_ufunc(PyObject *ufunc, PyObject *unwrapped, PyObject *key)
{
    Py_ssize_t count = PyTuple_GET_SIZE(unwrapped);
    PyObject *shapes[8] = {NULL}, *given[8] = {NULL};
    PyObject **items[8];
    Py_ssize_t sizes[8];
    PyObject *result = Miss, *shape = NULL, *dtype = NULL, *kind = NULL, *asked = NULL;
    int small = 1, fractional = 1; /* As in `rules._resolver_types`: every `int` small, every array with a fraction. */
    if (count == 0 || count > 8) {
        return Miss;
    }
    PyObject *written = PyTuple_New(1 + count);
    if (written == NULL) {
        return NULL;
    }
    PyTuple_SET_ITEM(written, 0, Py_NewRef(ufunc));
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(unwrapped, place);
        if (Py_TYPE(arg) == Node) {
            given[place] = Py_NewRef(SLOT(arg, node_at[N_DTYPE]));
            shapes[place] = Py_NewRef(SLOT(arg, node_at[N_SHAPE]));
        } else if (Py_TYPE(arg) == Ndarray) {
            given[place] = PyObject_GetAttr(arg, s_dtype);
            shapes[place] = PyObject_GetAttr(arg, s_shape);
        } else if (PyFloat_CheckExact(arg) || PyComplex_CheckExact(arg)) {
            given[place] = Py_NewRef((PyObject *)Py_TYPE(arg)); /* A Python number, which takes the array's type. */
            shapes[place] = Py_NewRef(empty_tuple);
        } else if (PyLong_CheckExact(arg)) {
            /* A Python `int`, within the bounds in which its value cannot make NumPy raise. */
            int overflow = 0;
            long long value = PyLong_AsLongLongAndOverflow(arg, &overflow);
            if (overflow || value < -large_whole || value > large_whole) {
                goto done;
            }
            small = small && value >= 0 && value <= small_whole;
            given[place] = Py_NewRef((PyObject *)&PyLong_Type);
            shapes[place] = Py_NewRef(empty_tuple);
        } else {
            /* A NumPy scalar with a fraction: by its `dtype`, as an array of none. Anything else has no entry. */
            int fractional = PySet_Contains(fractional_scalars, (PyObject *)Py_TYPE(arg));
            if (fractional <= 0) {
                if (fractional < 0) {
                    goto fail;
                }
                goto done;
            }
            given[place] = PyObject_GetAttr(arg, s_dtype);
            shapes[place] = Py_NewRef(empty_tuple);
        }
        if (given[place] == NULL || shapes[place] == NULL || !PyTuple_CheckExact(shapes[place])) {
            goto fail;
        }
        PyObject *part = PyType_Check(given[place]) ? Py_NewRef(given[place]) : PyLong_FromVoidPtr(given[place]);
        if (part == NULL) {
            goto fail;
        }

        PyTuple_SET_ITEM(written, 1 + place, part);
        items[place] = &PyTuple_GET_ITEM(shapes[place], 0);
        sizes[place] = PyTuple_GET_SIZE(shapes[place]);
    }
    if (!small) {
        /* An `int` that is not small is taken by its type only beside arrays with a fraction: every one of them,
         * those after it in the call too. */
        for (Py_ssize_t place = 0; fractional && place < count; place++) {
            if (PyType_Check(given[place])) {
                continue;
            }
            PyObject *letter = PyObject_GetAttr(given[place], s_kind);
            if (letter == NULL) {
                goto fail;
            }
            fractional = PyUnicode_CompareWithASCIIString(letter, "f") == 0 || PyUnicode_CompareWithASCIIString(letter, "c") == 0;
            Py_DECREF(letter);
        }
        if (!fractional) {
            goto done;
        }
    }
    PyObject *found = PyDict_GetItemWithError(types, written); /* (dtype, kind), borrowed. */
    if (found != NULL) {
        dtype = Py_NewRef(PyTuple_GET_ITEM(found, 0));
        kind = PyTuple_GET_ITEM(found, 1);
    } else {
        if (PyErr_Occurred()) {
            goto fail;
        }
        /* These types for the first time: what `rules._kind` and `rules._resolved_type` do for a ufunc. */
        PyObject *takes = PyObject_GetAttr(ufunc, s_nin), *gives = PyObject_GetAttr(ufunc, s_nout);
        PyObject *core = PyObject_GetAttr(ufunc, s_signature);
        int fits = takes && gives && core && PyLong_AsSsize_t(takes) == count;
        kind = !fits ? NULL : ufunc == matmul ? matrix_kind : (core == Py_None && PyLong_AsSsize_t(gives) == 1) ? elementwise_kind : NULL;
        Py_XDECREF(takes);
        Py_XDECREF(gives);
        Py_XDECREF(core);
        if (PyErr_Occurred()) {
            goto fail;
        }
        if (kind == NULL) {
            goto done;
        }
        asked = PyTuple_New(count + 1);
        if (asked == NULL) {
            goto fail;
        }
        PyTuple_SET_ITEM(asked, count, Py_NewRef(Py_None));
        for (Py_ssize_t place = 0; place < count; place++) {
            PyTuple_SET_ITEM(asked, place, Py_NewRef(given[place]));
        }
        for (Py_ssize_t place = 0; place < count; place++) {
            if (PyType_Check(given[place])) {
                continue;
            }
            /* An element type the layer defers (`rules.DEFERRED_KINDS`), and NumPy's own object for it. */
            PyObject *letter = PyObject_GetAttr(given[place], s_kind);
            int deferred_kind = letter && PyUnicode_CheckExact(letter) && PyUnicode_GET_LENGTH(letter) == 1
                                && strchr("biufc", (int)PyUnicode_READ_CHAR(letter, 0)) != NULL;
            Py_XDECREF(letter);
            int own = deferred_kind ? is_own_dtype(given[place]) : 0;
            if (PyErr_Occurred() || own < 0) {
                goto fail;
            }
            if (!own) {
                goto done;
            }
        }
        PyObject *resolved = PyObject_CallMethodOneArg(ufunc, s_resolve, asked);
        if (resolved == NULL) {
            PyErr_Clear(); /* NumPy has no loop for these types: its error comes from the call, in Python. */
            goto done;
        }
        dtype = PyTuple_Check(resolved) && PyTuple_GET_SIZE(resolved) == count + 1 ? Py_NewRef(PyTuple_GET_ITEM(resolved, count)) : NULL;
        Py_DECREF(resolved);
        if (dtype == NULL) {
            goto done;
        }
    }
    int product = PyObject_RichCompareBool(kind, matrix_kind, Py_EQ);
    if (product < 0) {
        goto fail;
    }
    if (!product) {
        shape = broadcast(items, sizes, count);
    } else if (count == 2 && sizes[0] > 0 && sizes[1] > 0) {
        /* `rules._matmul_shape`: an operand of one dimension is a row on the left and a column on the right, and
         * that dimension is dropped; the inner sizes agree; the leading dimensions broadcast. */
        Py_ssize_t left = sizes[0], right = sizes[1];
        PyObject *inner = right > 1 ? items[1][right - 2] : items[1][0];
        int agree = PyObject_RichCompareBool(items[0][left - 1], inner, Py_EQ);
        if (agree < 0) {
            goto fail;
        }
        PyObject **lead_items[2] = {items[0], items[1]};
        Py_ssize_t lead_sizes[2] = {left > 1 ? left - 2 : 0, right > 1 ? right - 2 : 0};
        PyObject *lead = agree ? broadcast(lead_items, lead_sizes, 2) : Miss;
        /* A length of zero anywhere is nothing to compute on: made at once. */
        int empty = 0;
        for (Py_ssize_t place = 0; place < 2 && lead != NULL && lead != Miss; place++) {
            for (Py_ssize_t axis = 0; axis < sizes[place]; axis++) {
                Py_ssize_t value = PyLong_AsSsize_t(items[place][axis]);
                empty = empty || value == 0;
            }
        }
        if (lead == NULL || PyErr_Occurred()) {
            Py_XDECREF(lead == Miss ? NULL : lead);
            goto fail;
        }
        if (lead == Miss || empty) {
            if (lead != Miss) {
                Py_DECREF(lead);
            }
            goto done;
        }
        PyObject *kept = PyTuple_New((left > 1) + (right > 1));
        if (kept != NULL) {
            if (left > 1) {
                PyTuple_SET_ITEM(kept, 0, Py_NewRef(items[0][left - 2]));
            }
            if (right > 1) {
                PyTuple_SET_ITEM(kept, left > 1, Py_NewRef(items[1][right - 1]));
            }
            shape = PySequence_Concat(lead, kept);
        }
        Py_DECREF(lead);
        Py_XDECREF(kept);
    } else {
        goto done;
    }
    if (shape == NULL) {
        goto fail;
    }
    if (shape == Miss) {
        shape = NULL;
        goto done;
    }
    result = keep_answer(key, shape, dtype, kind, asked ? written : NULL);
done:
    Py_XDECREF(written);
    Py_XDECREF(shape);
    Py_XDECREF(dtype);
    Py_XDECREF(asked);
    for (Py_ssize_t place = 0; place < count; place++) {
        Py_XDECREF(shapes[place]);
        Py_XDECREF(given[place]);
    }
    return result;
fail:
    result = NULL;
    goto done;
}

/* The answer for the first call of a view function or a reduction on one node, as `rules._plan` works it out.
 * For a view, NumPy is asked for the shape as `rules._view_shape` asks it, on an array of an element type of no
 * size. For a reduction the shape is the rule of `rules._reduced_shape`, and the element type is asked of
 * `rules._element_type`, which makes NumPy's call on an array of one element. `Miss` wherever `rules._plan` would
 * decline or let NumPy raise, and for whatever is written in a way this function does not read: an axis that is
 * not an `int` or a tuple of them, `keepdims` that is not a `bool`. */
static PyObject *
first_function(PyObject *rule, PyObject *unwrapped, PyObject *kwargs, PyObject *key)
{
    Py_ssize_t count = PyTuple_GET_SIZE(unwrapped), named = kwargs ? PyDict_GET_SIZE(kwargs) : 0;
    PyObject *node = count ? PyTuple_GET_ITEM(unwrapped, 0) : NULL;
    if (node == NULL || Py_TYPE(node) != Node) {
        return Miss;
    }
    int is_view = PySet_Contains(view_functions, rule);
    int reduces = is_view == 0 ? PySet_Contains(reductions, rule) : 0;
    int position = is_view == 0 && reduces == 0 ? PySet_Contains(positions, rule) : 0;
    if (is_view < 0 || reduces < 0 || position < 0) {
        PyErr_Clear(); /* A callable without a hash is in none of the sets. */
        return Miss;
    }
    if (!is_view && !reduces && !position) {
        return Miss;
    }
    /* `rules._admissible`: an element type the layer defers, and no length of zero. */
    PyObject *held = SLOT(node, node_at[N_SHAPE]), *dtype = SLOT(node, node_at[N_DTYPE]);
    Py_ssize_t dimensions = PyTuple_GET_SIZE(held);
    PyObject *letter = PyObject_GetAttr(dtype, s_kind);
    int deferred_kind = letter && PyUnicode_CheckExact(letter) && PyUnicode_GET_LENGTH(letter) == 1
                        && strchr("biufc", (int)PyUnicode_READ_CHAR(letter, 0)) != NULL;
    Py_XDECREF(letter);
    if (PyErr_Occurred()) {
        return NULL;
    }
    for (Py_ssize_t axis = 0; deferred_kind && axis < dimensions; axis++) {
        deferred_kind = PyLong_AsSsize_t(PyTuple_GET_ITEM(held, axis)) != 0;
    }
    if (!deferred_kind) {
        return Miss;
    }
    PyObject *shape = NULL, *result = Miss, *found = NULL;
    if (is_view) {
        /* `rules._kind`: the node first, and no node among the other arguments. */
        for (Py_ssize_t place = 1; place < count; place++) {
            if (Py_TYPE(PyTuple_GET_ITEM(unwrapped, place)) == Node) {
                return Miss;
            }
        }
        Py_ssize_t at = 0;
        PyObject *value;
        while (named && PyDict_Next(kwargs, &at, NULL, &value)) {
            if (Py_TYPE(value) == Node) {
                return Miss;
            }
        }
        if (rule == numpy_ravel) {
            return Miss; /* Its shape is arithmetic on the lengths, which Python does. */
        }
        PyObject *viewed = PyObject_CallFunctionObjArgs(numpy_empty, held, no_size, NULL);
        /* The call's arguments with the stand-in where the node is, in a tuple of their own. */
        PyObject *given = viewed ? PyTuple_New(count) : NULL;
        if (given != NULL) {
            PyTuple_SET_ITEM(given, 0, viewed);
            viewed = NULL;
            for (Py_ssize_t place = 1; place < count; place++) {
                PyTuple_SET_ITEM(given, place, Py_NewRef(PyTuple_GET_ITEM(unwrapped, place)));
            }
            PyObject *view = PyObject_Call(rule, given, named ? kwargs : NULL);
            shape = view ? PyObject_GetAttr(view, s_shape) : NULL;
            Py_XDECREF(view);
        }
        Py_XDECREF(viewed);
        Py_XDECREF(given);
        if (shape == NULL || !PyTuple_CheckExact(shape)) {
            PyErr_Clear(); /* NumPy refuses the axes or the shape: its error comes from Python's road. */
            Py_XDECREF(shape);
            return Miss;
        }
        result = keep_answer(key, shape, dtype, view_kind, NULL);
        Py_DECREF(shape);
        return result;
    }
    /* `rules._is_plain_reduction`: the node, an axis given at most once, no keyword but `axis` and `keepdims`. */
    if (count > 2 || (count == 2 && Py_TYPE(PyTuple_GET_ITEM(unwrapped, 1)) == Node)) {
        return Miss;
    }
    PyObject *axis = count == 2 ? PyTuple_GET_ITEM(unwrapped, 1) : NULL, *keepdims = Py_False;
    Py_ssize_t at = 0;
    PyObject *name, *value;
    while (named && PyDict_Next(kwargs, &at, &name, &value)) {
        if (PyUnicode_CompareWithASCIIString(name, "axis") == 0 && axis == NULL) {
            axis = value;
        } else if (PyUnicode_CompareWithASCIIString(name, "keepdims") == 0 && PyBool_Check(value)) {
            keepdims = value;
        } else {
            return Miss;
        }
    }
    /* `rules._reduced_shape`: without the reduced axes, or with them at length one. */
    char reduced[64] = {0};
    if (dimensions > 64) {
        return Miss;
    }
    if (axis == NULL || axis == Py_None) {
        memset(reduced, 1, (size_t)dimensions);
    } else {
        PyObject *single[1] = {axis};
        PyObject **axes = PyTuple_CheckExact(axis) ? &PyTuple_GET_ITEM(axis, 0) : single;
        Py_ssize_t given = PyTuple_CheckExact(axis) ? PyTuple_GET_SIZE(axis) : 1;
        if (position && PyTuple_CheckExact(axis)) {
            return Miss; /* A position takes one axis or none: `rules._names_one_axis`. */
        }
        for (Py_ssize_t each = 0; each < given; each++) {
            if (!PyLong_CheckExact(axes[each])) {
                return Miss;
            }
            Py_ssize_t which = PyLong_AsSsize_t(axes[each]);
            if (which == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return Miss;
            }
            which = which < 0 ? which + dimensions : which;
            if (which < 0 || which >= dimensions || reduced[which]) {
                return Miss; /* An axis the array lacks, or one named twice: NumPy's error to raise. */
            }
            reduced[which] = 1;
        }
    }
    PyObject *lengths = PyList_New(0);
    for (Py_ssize_t each = 0; lengths != NULL && each < dimensions; each++) {
        if (!reduced[each] || keepdims == Py_True) {
            if (PyList_Append(lengths, reduced[each] ? one : PyTuple_GET_ITEM(held, each)) < 0) {
                Py_CLEAR(lengths);
            }
        }
    }
    shape = lengths ? PyList_AsTuple(lengths) : NULL;
    Py_XDECREF(lengths);
    /* The element type is NumPy's answer to the call on an array of one element; its error is the caller's. */
    found = shape ? PyObject_CallFunctionObjArgs(python_element_type, reduction_kind, rule, unwrapped, kwargs, NULL) : NULL;
    result = found ? keep_answer(key, shape, found, reduction_kind, NULL) : NULL;
    Py_XDECREF(shape);
    Py_XDECREF(found);
    return result;
}

/* The remembered answer of `rules.plan` for a call: a new tuple (shape, dtype, kind, spec), as `rules.plan`
 * returns a signature. First by the plain key; then, as `rules._plan_marked`, by the key with placeholders, with
 * the call's own stand-ins put back into the shape. `Miss` (borrowed) when nothing is remembered. */
static PyObject *
remembered_plan(PyObject *rule, PyObject *unwrapped, PyObject *kwargs, PyObject **missed)
{
    PyObject *key = key_of(rule, unwrapped, kwargs, NULL);
    if (key == NULL) {
        return NULL;
    }
    if (key != Miss) {
        PyObject *kept = PyDict_GetItemWithError(plans, key);
        PyObject *answer = kept ? Py_NewRef(kept) : PyErr_Occurred() ? NULL : Miss;
        if (answer == Miss) {
            answer = Py_TYPE(rule) == Ufunc ? first_ufunc(rule, unwrapped, key) : first_function(rule, unwrapped, kwargs, key);
        }
        if (answer == Miss && missed != NULL) {
            *missed = key; /* The caller asks Python, and hands it the key so that it is not written twice. */
        } else {
            Py_DECREF(key);
        }
        return answer;
    }
    /* `rules._plan_marked` keeps no answer for a call whose arrays are too large for it to hold for every stand-in
     * (`rules._is_moderate`), so none is found for such a call here: that rule needs no second copy. */
    Marks marks = {0};
    PyObject *result = Miss;
    key = key_of(rule, unwrapped, kwargs, &marks);
    if (key == NULL || key == Miss) {
        result = key;
    } else {
        PyObject *kept = PyDict_GetItemWithError(plans, key);
        Py_DECREF(key);
        if (kept == NULL) {
            result = PyErr_Occurred() ? NULL : Miss;
        } else if (marks.count == 0) {
            result = Py_NewRef(kept);
        } else {
            PyObject *canonical = PyTuple_GET_ITEM(kept, 0);
            Py_ssize_t dimensions = PyTuple_GET_SIZE(canonical);
            PyObject *shape = PyTuple_New(dimensions);
            for (Py_ssize_t axis = 0; shape != NULL && axis < dimensions; axis++) {
                PyObject *length = PyTuple_GET_ITEM(canonical, axis);
                long place = PyLong_AsLong(length);
                if (place < 0 && -place - 1 < marks.count) {
                    length = marks.length[-place - 1]; /* The stand-in this call has where that placeholder is. */
                } else if (place < 0) {
                    Py_CLEAR(shape); /* A placeholder this call has no stand-in for: not expected; Python decides. */
                    PyErr_Clear();
                    break;
                }
                PyTuple_SET_ITEM(shape, axis, Py_NewRef(length));
            }
            result = shape ? PyTuple_Pack(4, shape, PyTuple_GET_ITEM(kept, 1), PyTuple_GET_ITEM(kept, 2), zero) : PyErr_Occurred() ? NULL : Miss;
            Py_XDECREF(shape);
        }
    }
    marks_clear(&marks);
    return result;
}

/* ---- `array._try_defer`, for a call whose answer is remembered. ---- */

/* Whether `array._lazies_in` would find nothing in a collection, as `array._holds_no_lazy` says at a glance. */
static int
holds_no_lazy(PyObject *collection)
{
    if (!PyTuple_Check(collection) && !PyList_Check(collection)) {
        return 0;
    }
    Py_ssize_t count = PySequence_Fast_GET_SIZE(collection);
    PyObject **items = PySequence_Fast_ITEMS(collection);
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *item = items[place];
        if (PyObject_TypeCheck(item, Lazy) || PyTuple_Check(item) || PyList_Check(item) || PyDict_Check(item)) {
            return 0;
        }
    }
    return 1;
}

/* Whether a view's shape follows from the shape viewed and whole numbers: `array._is_plain_step`. */
static int
is_plain_step(PyObject *unwrapped)
{
    PyObject *viewed = PyTuple_GET_ITEM(unwrapped, 0);
    PyObject *spec = Py_TYPE(viewed) == Node ? SLOT(viewed, node_at[N_SPEC]) : NULL;
    if (spec == NULL || PyObject_IsTrue(spec) != 1) {
        return 0; /* A shape with a length that is not known, or one that has no number yet: not vouched for. */
    }
    for (Py_ssize_t place = 1; place < PyTuple_GET_SIZE(unwrapped); place++) {
        PyObject *item = PyTuple_GET_ITEM(unwrapped, place);
        if (PyLong_CheckExact(item)) {
            continue;
        }
        if (!PyTuple_CheckExact(item)) {
            return 0;
        }
        for (Py_ssize_t inner = 0; inner < PyTuple_GET_SIZE(item); inner++) {
            if (!PyLong_CheckExact(PyTuple_GET_ITEM(item, inner))) {
                return 0;
            }
        }
    }
    return 1;
}

/* Add a call to the graph if its answer is remembered: the quick road of `array._try_defer`. `kwargs` is the
 * dictionary the node will keep. Returns a new lazy array, `Miss` (borrowed) when Python has to look, or NULL
 * with an error set. Nothing has changed when it returns `Miss`, but that a view among the arguments may have
 * had its nodes made again, which reading it does in Python too. */
static PyObject *lazy_for(PyObject *node);

static PyObject *
try_defer(PyObject *operation, PyObject *name, PyObject *args, PyObject *kwargs, PyObject *ruled_as, PyFrameObject *frame)
{
    if (!enabled || frame == NULL) {
        return Miss;
    }
    Py_ssize_t count = PyTuple_GET_SIZE(args);
    PyObject *result = Miss, *key = NULL, *node = NULL;
    int strict = 0;
    PyObject *unwrapped = PyTuple_New(count);
    if (unwrapped == NULL) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        if (Py_TYPE(arg) == Lazy) {
            PyObject *operand = node_of(arg);
            if (operand == NULL) {
                goto fail;
            }
            PyTuple_SET_ITEM(unwrapped, place, operand);
            if (Py_TYPE(operand) != Node) {
                goto done;
            }
            strict = strict || SLOT(operand, node_at[N_STRICT]) == Py_True;
            continue;
        }
        PyTuple_SET_ITEM(unwrapped, place, Py_NewRef(arg));
        if (PyObject_TypeCheck(arg, Lazy) || PyDict_Check(arg)
            || ((PyTuple_Check(arg) || PyList_Check(arg)) && !holds_no_lazy(arg))) {
            goto done; /* A lazy array inside a collection: Python searches for it. */
        }
    }
    {
        Py_ssize_t position = 0;
        PyObject *value;
        while (PyDict_Next(kwargs, &position, NULL, &value)) {
            if (PyObject_TypeCheck(value, Lazy) || PyDict_Check(value)
                || ((PyTuple_Check(value) || PyList_Check(value)) && !holds_no_lazy(value))) {
                goto done;
            }
        }
    }
    PyObject *signature = remembered_plan(ruled_as ? ruled_as : operation, unwrapped, kwargs, &key); /* (shape, dtype, kind, spec) */
    if (signature == NULL) {
        goto fail;
    }
    if (signature == Miss) {
        /* The first call of its kind: the rules are asked, in Python, where they are. `rules.plan` works the answer
         * out and keeps it, raises NumPy's error, or says None for a call that is made at once. The node is then
         * made here all the same, without the rest of Python's road. */
        PyObject *plan = PyDict_GetItemWithError(rules_names, s_plan); /* `rules.first` */
        signature = plan ? PyObject_CallFunctionObjArgs(plan, key ? key : Py_None, ruled_as ? ruled_as : operation, unwrapped, kwargs, NULL) : NULL;
        if (signature == NULL) {
            goto fail;
        }
        if (signature == Py_None || !PyTuple_Check(signature) || PyTuple_GET_SIZE(signature) != 4) {
            if (signature == Py_None) {
                Py_XSETREF(declined_operation, Py_NewRef(operation));
                Py_XSETREF(declined_args, Py_NewRef(args));
                Py_XSETREF(declined_kwargs, Py_NewRef(kwargs));
            }
            Py_DECREF(signature);
            goto done;
        }
    }
    int full = at_the_limit();
    if (full != 0) {
        Py_DECREF(signature);
        if (full < 0) {
            goto fail;
        }
        goto done; /* Python computes everything pending first. */
    }
    PyObject *shape = PyTuple_GET_ITEM(signature, 0), *dtype = PyTuple_GET_ITEM(signature, 1);
    PyObject *kind = PyTuple_GET_ITEM(signature, 2), *spec = PyTuple_GET_ITEM(signature, 3);
    node = deferred(operation, name, unwrapped, kwargs, shape, dtype, kind, strict ? Py_True : Py_False, NULL, spec, frame);
    if (node == NULL) {
        Py_DECREF(signature);
        goto fail;
    }
    int is_view = PyObject_RichCompareBool(kind, view_kind, Py_EQ);
    PyObject *base = count > 0 && Py_TYPE(PyTuple_GET_ITEM(args, 0)) == Lazy ? PyTuple_GET_ITEM(args, 0) : NULL;
    if (is_view < 0) {
        result = NULL;
    } else if (!is_view || base == NULL || (operation == getitem && PyTuple_GET_SIZE(shape) == 0)) {
        /* A new array. An index that leaves no dimension gives a number, which is a copy, not a view. */
        result = lazy_for(node);
    } else {
        int sure = operation == getitem ? 1 : PySet_Contains(sure_views, operation);
        if (sure < 0) {
            result = NULL;
        } else if (sure) {
            /* A view that always shares its argument's memory: the argument's cell, and one more step. */
            PyObject *tail = PyTuple_GetSlice(unwrapped, 1, count);
            PyObject *step = tail ? new_step(operation, name, tail, kwargs, shape, dtype, SLOT(node, node_at[N_ORIGIN]),
                                             PyDict_GET_SIZE(kwargs) == 0 && is_plain_step(unwrapped)) : NULL;
            PyObject *steps = step ? PyTuple_Pack(1, step) : NULL;
            PyObject *all = steps ? PySequence_Concat(SLOT(base, lazy_at[L_STEPS]), steps) : NULL;
            result = all ? new_view(node, cell_of(base), all) : NULL;
            Py_XDECREF(tail);
            Py_XDECREF(step);
            Py_XDECREF(steps);
            Py_XDECREF(all);
        } else {
            /* `np.reshape` and `np.ravel`: a view or a copy, which is not known yet. */
            result = new_lazy(node);
            if (result != NULL) {
                PyObject *made_cell = SLOT(result, lazy_at[L_HELD]);
                Py_SETREF(SLOT(made_cell, cell_at[C_DOUBTED]), Py_NewRef(Py_True));
                PyObject *noted = PyObject_CallMethodOneArg(cell_of(base), s_doubt, result);
                if (noted == NULL) {
                    Py_CLEAR(result);
                } else {
                    Py_DECREF(noted);
                }
            }
        }
    }
    Py_DECREF(signature);
done:
    Py_XDECREF(key);
    Py_XDECREF(node);
    Py_DECREF(unwrapped);
    return result;
fail:
    Py_XDECREF(key);
    Py_XDECREF(node);
    Py_DECREF(unwrapped);
    return NULL;
}

/* The shape of `shape[index]` for a tuple or a slice: `rules.basic_shape`, line for line. A new reference;
 * `Miss` (borrowed) wherever that function returns None; NULL with an error set. */
static PyObject *
basic_shape(PyObject *shape, PyObject *index)
{
    PyObject *single[1] = {index};
    PyObject **items = PyTuple_CheckExact(index) ? &PyTuple_GET_ITEM(index, 0) : single;
    Py_ssize_t count = PyTuple_CheckExact(index) ? PyTuple_GET_SIZE(index) : 1;
    Py_ssize_t dimensions = PyTuple_GET_SIZE(shape), axis = 0, taken = 0;
    int whole = 1;
    PyObject *lengths = PyList_New(0);
    if (lengths == NULL) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        if (items[place] != Py_None && items[place] != Py_Ellipsis) {
            taken++;
        }
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *item = items[place];
        if (PyLong_CheckExact(item) || PySlice_Check(item)) {
            if (axis >= dimensions) {
                goto unsure;
            }
            Py_ssize_t length = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, axis));
            if (length == -1 && PyErr_Occurred()) {
                goto fail;
            }
            if (PyLong_CheckExact(item)) {
                Py_ssize_t position = PyLong_AsSsize_t(item);
                if (position == -1 && PyErr_Occurred()) {
                    PyErr_Clear(); /* Too large for the machine: out of bounds. */
                    goto unsure;
                }
                if (position < -length || position >= length) {
                    goto unsure;
                }
            } else {
                PySliceObject *slice = (PySliceObject *)item;
                if ((slice->start != Py_None && !PyLong_CheckExact(slice->start))
                    || (slice->stop != Py_None && !PyLong_CheckExact(slice->stop))
                    || (slice->step != Py_None && !PyLong_CheckExact(slice->step))) {
                    goto unsure;
                }
                Py_ssize_t start, stop, step;
                if (PySlice_Unpack(item, &start, &stop, &step) < 0) {
                    PyErr_Clear(); /* A step of zero: NumPy's error to raise. */
                    goto unsure;
                }
                /* Python's own arithmetic for a slice, which is NumPy's for whole numbers. */
                PyObject *selected = PyLong_FromSsize_t(PySlice_AdjustIndices(length, &start, &stop, step));
                if (selected == NULL || PyList_Append(lengths, selected) < 0) {
                    Py_XDECREF(selected);
                    goto fail;
                }
                Py_DECREF(selected);
            }
            axis++;
        } else if (item == Py_None) {
            if (PyList_Append(lengths, one) < 0) {
                goto fail;
            }
        } else if (item == Py_Ellipsis) {
            if (!whole || taken > dimensions) {
                goto unsure;
            }
            whole = 0;
            /* `...` stands for the axes the other items do not take. */
            for (Py_ssize_t skipped = dimensions - taken; skipped > 0; skipped--, axis++) {
                if (PyList_Append(lengths, PyTuple_GET_ITEM(shape, axis)) < 0) {
                    goto fail;
                }
            }
        } else {
            goto unsure;
        }
    }
    for (; axis < dimensions; axis++) {
        if (PyList_Append(lengths, PyTuple_GET_ITEM(shape, axis)) < 0) {
            goto fail;
        }
    }
    PyObject *result = PyList_AsTuple(lengths);
    Py_DECREF(lengths);
    return result;
unsure:
    Py_DECREF(lengths);
    return Miss;
fail:
    Py_DECREF(lengths);
    return NULL;
}

/* `x[index]` for a pending array and an index of whole numbers and slices: the quick road of
 * `array._plain_item`. An array that has its values is left to Python, which reads one element on the spot. */
static PyObject *
plain_item(PyObject *target, PyObject *index, PyFrameObject *frame)
{
    if (!enabled || frame == NULL || !(PyLong_CheckExact(index) || PySlice_Check(index) || PyTuple_CheckExact(index))) {
        return Miss;
    }
    PyObject *result = Miss, *shape = NULL, *kwargs = NULL, *args = NULL, *made = NULL;
    PyObject *node = node_of(target);
    if (node == NULL) {
        return NULL;
    }
    if (Py_TYPE(node) != Node || SLOT(node, node_at[N_OPERATION]) == Py_None) {
        goto done;
    }
    PyObject *spec = spec_part(node, NULL);
    if (spec == NULL) {
        goto fail;
    }
    if (spec == Miss) {
        goto done;
    }
    PyObject *allowed = PyDict_GetItemWithError(admissible, spec);
    Py_DECREF(spec);
    if (allowed != Py_True) {
        if (PyErr_Occurred()) {
            goto fail;
        }
        goto done;
    }
    PyObject *held = SLOT(node, node_at[N_SHAPE]);
    if (PyLong_CheckExact(index)) {
        /* One whole number takes one row: as the first lines of `rules.basic_shape`. */
        if (PyTuple_GET_SIZE(held) == 0) {
            goto done;
        }
        Py_ssize_t length = PyLong_AsSsize_t(PyTuple_GET_ITEM(held, 0));
        Py_ssize_t position = PyLong_AsSsize_t(index);
        if ((length == -1 || position == -1) && PyErr_Occurred()) {
            PyErr_Clear(); /* Too large for the machine: out of bounds, which NumPy says in its own words. */
            goto done;
        }
        if (position < -length || position >= length) {
            goto done;
        }
        shape = PyTuple_GetSlice(held, 1, PyTuple_GET_SIZE(held));
    } else {
        shape = basic_shape(held, index);
        if (shape == Miss) {
            shape = NULL;
            goto done;
        }
    }
    if (shape == NULL) {
        goto fail;
    }
    int full = at_the_limit();
    if (full < 0) {
        goto fail;
    }
    if (full) {
        goto done;
    }
    kwargs = PyDict_New();
    args = kwargs ? PyTuple_Pack(2, node, index) : NULL;
    if (args == NULL) {
        goto fail;
    }
    PyObject *dtype = SLOT(node, node_at[N_DTYPE]);
    made = deferred(getitem, s_getitem, args, kwargs, shape, dtype, view_kind, SLOT(node, node_at[N_STRICT]), NULL, zero, frame);
    if (made == NULL) {
        goto fail;
    }
    if (PyTuple_GET_SIZE(shape) == 0) {
        result = new_lazy(made); /* A number, which is a copy, not a view. */
    } else {
        PyObject *tail = PyTuple_Pack(1, index);
        PyObject *step = tail ? new_step(getitem, s_getitem, tail, kwargs, shape, dtype, SLOT(made, node_at[N_ORIGIN]), 1) : NULL;
        PyObject *steps = step ? PyTuple_Pack(1, step) : NULL;
        PyObject *all = steps ? PySequence_Concat(SLOT(target, lazy_at[L_STEPS]), steps) : NULL;
        result = all ? new_view(made, cell_of(target), all) : NULL;
        Py_XDECREF(tail);
        Py_XDECREF(step);
        Py_XDECREF(steps);
        Py_XDECREF(all);
    }
done:
    Py_XDECREF(shape);
    Py_XDECREF(kwargs);
    Py_XDECREF(args);
    Py_XDECREF(made);
    Py_DECREF(node);
    return result;
fail:
    Py_XDECREF(shape);
    Py_XDECREF(kwargs);
    Py_XDECREF(args);
    Py_XDECREF(made);
    Py_DECREF(node);
    return NULL;
}

/* Take an index apart as `array._index_parts` does: `template` gets each item, with a slot where a pending array
 * is, and `with_nodes` the same with the array's node there. 1 when done; 0 for an index Python has to look at (an
 * array that has its values or is of another type than whole numbers and masks, a list, an `ndarray`); -1 with an
 * error set. */
static int
index_apart(PyObject **items, Py_ssize_t count, PyObject *template, PyObject *with_nodes, Py_ssize_t *found, int *strict,
            int *numbers)
{
    Py_ssize_t most = PyTuple_GET_SIZE(slots);
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *item = items[place];
        if (Py_TYPE(item) == Lazy) {
            PyObject *pending_node = node_of(item);
            if (pending_node == NULL) {
                return -1;
            }
            PyTuple_SET_ITEM(with_nodes, place, pending_node);
            if (Py_TYPE(pending_node) != Node || SLOT(pending_node, node_at[N_OPERATION]) == Py_None || *found == most) {
                return 0;
            }
            PyObject *kind = PyObject_GetAttr(SLOT(pending_node, node_at[N_DTYPE]), s_kind);
            if (kind == NULL) {
                return -1;
            }
            /* A mask, or whole numbers: `"biu"`. */
            int indexes = PyUnicode_CompareWithASCIIString(kind, "b") == 0 || PyUnicode_CompareWithASCIIString(kind, "i") == 0
                          || PyUnicode_CompareWithASCIIString(kind, "u") == 0;
            Py_DECREF(kind);
            if (!indexes) {
                return 0;
            }
            *strict = *strict || SLOT(pending_node, node_at[N_STRICT]) == Py_True;
            *numbers = *numbers && PyTuple_GET_SIZE(SLOT(pending_node, node_at[N_SHAPE])) == 0;
            PyTuple_SET_ITEM(template, place, Py_NewRef(PyTuple_GET_ITEM(slots, (*found)++)));
            continue;
        }
        int plain = PyLong_CheckExact(item) || item == Py_None || item == Py_Ellipsis;
        if (!plain && PySlice_Check(item)) {
            PySliceObject *slice = (PySliceObject *)item;
            plain = (slice->start == Py_None || PyLong_CheckExact(slice->start)) && (slice->stop == Py_None || PyLong_CheckExact(slice->stop))
                    && (slice->step == Py_None || PyLong_CheckExact(slice->step));
        }
        if (!plain) {
            return 0;
        }
        PyTuple_SET_ITEM(template, place, Py_NewRef(item));
        PyTuple_SET_ITEM(with_nodes, place, Py_NewRef(item));
    }
    return 1;
}

/* `x[index]` for an index that holds pending arrays, masks or positions, beside whole numbers and slices: the
 * quick road of `array._select`, with the index taken apart as `array._index_parts` takes it. An array in the
 * index that has its values, a list, an `ndarray` and anything else are left to Python. */
static PyObject *
select_item(PyObject *target, PyObject *index, PyFrameObject *frame)
{
    int whole = PyTuple_CheckExact(index);
    if (!enabled || frame == NULL || (!whole && PyTuple_Check(index))) {
        return Miss;
    }
    PyObject *single[1] = {index};
    PyObject **items = whole ? &PyTuple_GET_ITEM(index, 0) : single;
    Py_ssize_t count = whole ? PyTuple_GET_SIZE(index) : 1, found = 0;
    PyObject *result = Miss, *aim = NULL, *unwrapped = NULL, *kwargs = NULL, *signature = NULL, *run = NULL, *args = NULL,
             *node = NULL;
    int strict = 0, numbers = 1;
    PyObject *template = PyTuple_New(count), *with_nodes = PyTuple_New(count);
    if (template == NULL || with_nodes == NULL) {
        goto fail;
    }
    int apart = index_apart(items, count, template, with_nodes, &found, &strict, &numbers);
    if (apart < 0) {
        goto fail;
    }
    if (apart == 0) {
        goto done;
    }
    if (found == 0) {
        goto done;
    }
    aim = node_of(target);
    if (aim == NULL) {
        goto fail;
    }
    if (Py_TYPE(aim) != Node) {
        goto done;
    }
    strict = strict || SLOT(aim, node_at[N_STRICT]) == Py_True;
    unwrapped = PyTuple_Pack(2, aim, whole ? with_nodes : PyTuple_GET_ITEM(with_nodes, 0));
    kwargs = unwrapped ? PyDict_New() : NULL;
    if (kwargs == NULL) {
        goto fail;
    }
    signature = remembered_plan(getitem, unwrapped, kwargs, NULL);
    if (signature == NULL) {
        goto fail;
    }
    if (signature == Miss) {
        signature = NULL;
        goto done;
    }
    int full = at_the_limit();
    if (full < 0) {
        goto fail;
    }
    if (full) {
        goto done;
    }
    /* The node reads the array, then the index's pending arrays in the order of their slots. */
    args = PyTuple_New(1 + found);
    run = args ? Select->tp_alloc(Select, 0) : NULL;
    if (run == NULL) {
        goto fail;
    }
    PyTuple_SET_ITEM(args, 0, Py_NewRef(aim));
    for (Py_ssize_t place = 0, taken = 0; place < count; place++) {
        if (Py_TYPE(items[place]) == Lazy) {
            PyTuple_SET_ITEM(args, 1 + taken++, Py_NewRef(PyTuple_GET_ITEM(with_nodes, place)));
        }
    }
    SLOT(run, select_at[X_TEMPLATE]) = Py_NewRef(template);
    SLOT(run, select_at[X_WHOLE]) = Py_NewRef(whole ? Py_True : Py_False);
    PyObject *shape = PyTuple_GET_ITEM(signature, 0);
    node = deferred(run, s_getitem, args, kwargs, shape, PyTuple_GET_ITEM(signature, 1), PyTuple_GET_ITEM(signature, 2),
                    strict ? Py_True : Py_False, NULL, PyTuple_GET_ITEM(signature, 3), frame);
    if (node == NULL) {
        goto fail;
    }
    result = new_lazy(node);
    if (result != NULL && PyTuple_GET_SIZE(shape) != 0 && numbers) {
        /* Every array in the index is a pending one without dimensions: a view of a row if it is a number when
         * computed, as `rules.is_doubtful_view` says. Not known yet. */
        PyObject *made_cell = SLOT(result, lazy_at[L_HELD]);
        Py_SETREF(SLOT(made_cell, cell_at[C_DOUBTED]), Py_NewRef(Py_True));
        PyObject *noted = PyObject_CallMethodOneArg(cell_of(target), s_doubt, result);
        if (noted == NULL) {
            Py_CLEAR(result);
        } else {
            Py_DECREF(noted);
        }
    }
done:
    Py_XDECREF(template);
    Py_XDECREF(with_nodes);
    Py_XDECREF(aim);
    Py_XDECREF(unwrapped);
    Py_XDECREF(kwargs);
    Py_XDECREF(signature);
    Py_XDECREF(run);
    Py_XDECREF(args);
    Py_XDECREF(node);
    return result;
fail:
    result = NULL;
    goto done;
}

/* ---- Writes: `x += y` and `x[index] = v` as nodes. ---- */

/* Whether a view taken at once may be alive that the layer keeps no record of: `array._views_stand_alone`. Set
 * with the first limit under which calls are made at once, and not cleared when the limit is: see `writable`. */
static int views_alone = 0;

/* Whether a write into a lazy array can be a node, as far as this file can tell: `array._may_defer_write`, with
 * anything it would have to work out left to Python. Returns the cell's node (borrowed), or NULL for "ask Python".
 *
 * With `waits` the write is one that would be computed later. A view the core took at once holds NumPy's view
 * and nothing that leads back to the array, so a write that waits could not be shown to it: it would be computed
 * into a copy, since the view's array still refers to the memory, and the view would go on showing what the array
 * held. So while such views may exist, a write is not made to wait when the array's memory is another array's (it
 * is such a view itself), or when anything but the node refers to the array: NumPy's view refers to the array it
 * is a view of. The write is then made at once, in the memory, after everything pending, where every view sees it.
 * Without `waits` the question is only whether NumPy's write into the memory now is what the layer may make. It
 * is for an array the program was handed (`exposed`) and for one whose memory may be shared with another (the
 * cell's doubts) as for any other: with nothing pending, a write into the memory is what the eager program does,
 * and whoever else knows the memory sees it, as there. Those two matter to a write that waits, which may be
 * computed into a copy. Left to Python, as it was, such a write took 3.9 µs for NumPy's 0.2 µs, and a program that
 * updates the array it was given a row or an element at a time ran 4 to 25 times slower than eagerly. */
static int is_admissible(PyObject *node);

static PyObject *
writable_as(PyObject *target, int waits)
{
    PyObject *cell = cell_of(target);
    if (cell == NULL) {
        PyErr_Clear();
        return NULL;
    }
    PyObject *held = SLOT(cell, cell_at[C_NODE]);
    PyObject *doubtful = SLOT(cell, cell_at[C_DOUBTFUL]);
    if (Py_TYPE(held) != Node || SLOT(held, node_at[N_FAILURE]) != Py_None
        || PyTuple_GET_SIZE(SLOT(held, node_at[N_SHAPE])) == 0) {
        return NULL;
    }
    if (waits && (SLOT(held, node_at[N_EXPOSED]) != Py_False || SLOT(cell, cell_at[C_DOUBTED]) != Py_False || doubtful != Py_None)) {
        return NULL;
    }
    /* No length of zero and an element type the layer defers: what `rules._admissible` says, remembered. A node
     * made with its value has not been asked for its number yet, and is asked here. */
    if (is_admissible(held) != 1) {
        PyErr_Clear();
        return NULL;
    }
    if (waits && views_alone && has_value(held)) {
        PyObject *memory = SLOT(held, node_at[N_VALUE]);
        if (PyArray_Check(memory) && (PyArray_BASE((PyArrayObject *)memory) != NULL || Py_REFCNT(memory) > 1)) {
            return NULL;
        }
    }
    return held;
}

/* `writable_as` for a write that would wait: what every road that writes a node asks. */
static PyObject *
writable(PyObject *target)
{
    return writable_as(target, 1);
}

/* `array._defer_write(target, name, perform, operands)`: the cell gets a new node, the contents after the
 * write. `reads` is a tuple of what the write reads, with a node for each lazy array. 0, or -1 with an error set,
 * or 1 for "ask Python" (the pending set is full). */
/* Make a write now, into the memory the array has, when that is what the eager program does and nothing can tell
 * the difference: `small_limit` is set and no node at all is pending, so no operation written before the write is
 * still waiting to read what the array held. The array's node stays the cell's; its memory has the new contents,
 * and every view of it sees them, as with NumPy. `direct`, when given, is called with the part written and the one
 * value in place of `perform`: the function of the `operator` module behind `x += y`, so that no frame of the
 * layer's stands between NumPy and the program's line. Returns 1 when the write was made, 0 when it was not and is
 * to be written as a node, -1 with an error set. */
static int
write_now(PyObject *target, PyObject *held, PyObject *perform, PyObject *reads, PyObject *direct)
{
    if (small_limit == 0 || PySet_GET_SIZE(pending) != 0 || !has_value(held)
        || Py_TYPE(SLOT(held, node_at[N_VALUE])) != Ndarray) {
        return 0;
    }
    Py_ssize_t count = PyTuple_GET_SIZE(reads);
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *read = PyTuple_GET_ITEM(reads, place);
        if (Py_TYPE(read) == Node && !has_value(read)) {
            return 0;
        }
    }
    int done = -1;
    PyObject *view = Py_NewRef(SLOT(held, node_at[N_VALUE]));
    PyObject *steps = SLOT(target, lazy_at[L_STEPS]);
    PyObject *values = NULL, *made = NULL;
    for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(steps); place++) {
        /* `step.of(view)`: the view operation on the array, then its own arguments. */
        PyObject *step = PyTuple_GET_ITEM(steps, place);
        PyObject *first = PyTuple_Pack(1, view);
        PyObject *all = first ? PySequence_Concat(first, SLOT(step, step_at[S_TAIL])) : NULL;
        PyObject *next = all ? PyObject_Call(SLOT(step, step_at[S_OPERATION]), all, SLOT(step, step_at[S_KWARGS])) : NULL;
        Py_XDECREF(first);
        Py_XDECREF(all);
        if (next == NULL) {
            goto out;
        }
        Py_SETREF(view, next);
    }
    values = PyTuple_New(count);
    if (values == NULL) {
        goto out;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *read = PyTuple_GET_ITEM(reads, place);
        PyTuple_SET_ITEM(values, place, Py_NewRef(Py_TYPE(read) == Node ? SLOT(read, node_at[N_VALUE]) : read));
    }
    if (direct != NULL && count == 1) {
        made = PyObject_CallFunctionObjArgs(direct, view, PyTuple_GET_ITEM(values, 0), NULL);
    } else if (Py_TYPE(perform) == Assign && count == 1) {
        made = PyObject_SetItem(view, SLOT(perform, assign_at[A_INDEX]), PyTuple_GET_ITEM(values, 0)) < 0 ? NULL : Py_NewRef(Py_None);
    } else {
        PyObject *first = PyTuple_Pack(1, view);
        PyObject *all = first ? PySequence_Concat(first, values) : NULL;
        made = all ? PyObject_Call(perform, all, NULL) : NULL;
        Py_XDECREF(first);
        Py_XDECREF(all);
    }
    if (made != NULL) {
        done = 1;
        written_at_once++;
        for (Py_ssize_t place = 0; place < count; place++) {
            PyObject *read = PyTuple_GET_ITEM(reads, place);
            if (Py_TYPE(read) == Node && SLOT(read, node_at[N_STRICT]) == Py_True) {
                Py_SETREF(SLOT(held, node_at[N_STRICT]), Py_NewRef(Py_True));
            }
        }
    }
out:
    Py_XDECREF(made);
    Py_XDECREF(values);
    Py_DECREF(view);
    return done;
}

static int
defer_write(PyObject *target, PyObject *name, PyObject *perform, PyObject *reads, PyFrameObject *frame, PyObject *direct)
{
    int full = at_the_limit();
    if (full != 0) {
        return full;
    }
    PyObject *cell = cell_of(target);
    if (cell == NULL) {
        return -1;
    }
    PyObject *held = SLOT(cell, cell_at[C_NODE]);
    int now = write_now(target, held, perform, reads, direct);
    if (now != 0) {
        return now < 0 ? -1 : 0;
    }
    int strict = SLOT(held, node_at[N_STRICT]) == Py_True;
    for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(reads); place++) {
        PyObject *read = PyTuple_GET_ITEM(reads, place);
        strict = strict || (Py_TYPE(read) == Node && SLOT(read, node_at[N_STRICT]) == Py_True);
    }
    PyObject *run = Write->tp_alloc(Write, 0);
    PyObject *first = run ? PyTuple_Pack(1, held) : NULL;
    PyObject *args = first ? PySequence_Concat(first, reads) : NULL;
    Py_XDECREF(first);
    PyObject *kwargs = args ? PyDict_New() : NULL;
    PyObject *node = NULL;
    if (kwargs != NULL) {
        SLOT(run, write_at[W_STEPS]) = Py_NewRef(SLOT(target, lazy_at[L_STEPS]));
        SLOT(run, write_at[W_PERFORM]) = Py_NewRef(perform);
        node = deferred(run, name, args, kwargs, SLOT(held, node_at[N_SHAPE]), SLOT(held, node_at[N_DTYPE]), write_kind,
                        strict ? Py_True : Py_False, NULL, SLOT(held, node_at[N_SPEC]), frame);
    }
    Py_XDECREF(run);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    if (node == NULL) {
        return -1;
    }
    Py_SETREF(SLOT(cell, cell_at[C_NODE]), node);
    return 0;
}

/* `x += y` and its like, for a write that is known to fit: the quick road of the method `array._in_place` builds.
 * `apply` is the function of the `operator` module, `perform` what makes the write. Returns `target` (new
 * reference), `Miss`, or NULL. */
static PyObject *ufunc_now(PyObject *ufunc, PyObject *const *operands, Py_ssize_t count, int operator);
static PyObject *ufunc_on(PyObject *ufunc, PyObject *const *operands, Py_ssize_t count, int operator, int strict);
static PyObject *number_in_place(PyObject *ufunc, PyObject *target, PyObject *other);

/* The array a write made at once goes into: the memory of `target`'s cell with the target's steps taken of it.
 * `*held` is the cell's node, or NULL for an array that holds its value alone, which is nobody's but the layer's,
 * never in doubt and has no view: what `writable` asks of a node, but for the element type and the dimensions,
 * which are asked here. A new reference, `Miss` when the write is not one to make now, or NULL. */
static PyObject *
memory_now(PyObject *target, PyObject **held)
{
    *held = NULL;
    PyObject *alone = SLOT(target, lazy_at[L_HELD]);
    PyObject *view;
    if (alone != NULL && Py_TYPE(alone) != Cell) {
        if (Py_TYPE(alone) != Ndarray || PyArray_NDIM((PyArrayObject *)alone) == 0 || !value_admissible(alone)) {
            return Miss;
        }
        return Py_NewRef(alone);
    }
    *held = writable_as(target, 0);
    if (*held == NULL || !has_value(*held) || Py_TYPE(SLOT(*held, node_at[N_VALUE])) != Ndarray) {
        return Miss;
    }
    view = Py_NewRef(SLOT(*held, node_at[N_VALUE]));
    PyObject *steps = SLOT(target, lazy_at[L_STEPS]);
    for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(steps); place++) {
        /* `step.of(view)`: the view operation on the array, then its own arguments. */
        PyObject *step = PyTuple_GET_ITEM(steps, place);
        PyObject *first = PyTuple_Pack(1, view);
        PyObject *all = first ? PySequence_Concat(first, SLOT(step, step_at[S_TAIL])) : NULL;
        PyObject *next = all ? PyObject_Call(SLOT(step, step_at[S_OPERATION]), all, SLOT(step, step_at[S_KWARGS])) : NULL;
        Py_XDECREF(first);
        Py_XDECREF(all);
        if (next == NULL) {
            Py_DECREF(view);
            return NULL;
        }
        Py_SETREF(view, next);
    }
    return view;
}

/* Note that an array written into at once has been given something of a strict array, and is strict from now on.
 * `held` is what `memory_now` gave. 0, or -1 with an error set. */
static int
strict_now(PyObject *target, PyObject *held)
{
    PyObject *alone = SLOT(target, lazy_at[L_HELD]);
    if (held != NULL) {
        Py_SETREF(SLOT(held, node_at[N_STRICT]), Py_NewRef(Py_True));
    } else if (Py_TYPE(alone) != Cell) {
        /* Still holding its value alone: the same kind, and strict. */
        PyObject *seen = SLOT(target, lazy_at[L_SEEN]);
        int at_once = PyTuple_GET_ITEM(seen, 0) == at_once_kind;
        Py_SETREF(SLOT(target, lazy_at[L_SEEN]), Py_NewRef(PyTuple_GET_ITEM(valued, (at_once ? 2 : 0) + 1)));
    } else {
        /* The write gave the array its cell on the way. */
        Py_SETREF(SLOT(SLOT(alone, cell_at[C_NODE]), node_at[N_STRICT]), Py_NewRef(Py_True));
    }
    return 0;
}

/* `x += y` and its like made now, in the array's own memory, when nothing at all is pending (`small_limit`):
 * the function of the `operator` module on NumPy's arrays, which is what the eager program does at that line,
 * NumPy's error for an operand that does not fit included. For a number, which nothing is written into, it is
 * the ufunc's result, a new array, as `array._in_place` gives. Returns what the operator gives (a new reference:
 * the target, or the new number), `Miss` for what the other roads decide, or NULL. */
static PyObject *
in_place_now(PyObject *target, PyObject *other, PyObject *apply)
{
    if (small_limit == 0 || PySet_GET_SIZE(pending) != 0) {
        return Miss;
    }
    int strict = 0;
    PyObject *alone = SLOT(target, lazy_at[L_HELD]);
    if (alone != NULL && PyObject_TypeCheck(alone, Generic)) {
        /* A number that holds its value alone: `s += v` makes a new number. */
        PyObject *ufunc = PyDict_GetItemWithError(in_place_ufuncs, apply);
        if (ufunc == NULL) {
            return PyErr_Occurred() ? NULL : Miss;
        }
        /* `s += v` on a NumPy scalar is `s = s + v`, the scalar's own operator: 35 ns in the eager program. The
         * ufunc on two numbers, which this was, is a call of 600 ns. */
        PyObject *made = number_in_place(ufunc, target, other);
        if (made != Miss) {
            return made;
        }
        PyObject *pair[2] = {target, other};
        return ufunc_now(ufunc, pair, 2, 0);
    }
    PyObject *operand;
    if (Py_TYPE(other) == Lazy) {
        operand = value_of(other, &strict);
        if (operand == NULL || operand == Miss) {
            return operand;
        }
    } else if (PyFloat_CheckExact(other) || PyLong_CheckExact(other) || PyBool_Check(other) || PyComplex_CheckExact(other)
               || Py_TYPE(other) == Ndarray) {
        operand = Py_NewRef(other);
    } else {
        return Miss;
    }
    PyObject *held = NULL;
    PyObject *view = memory_now(target, &held);
    if (view == NULL || view == Miss) {
        Py_DECREF(operand);
        return view;
    }
    PyObject *made = PyObject_CallFunctionObjArgs(apply, view, operand, NULL);
    Py_DECREF(operand);
    if (made == NULL) {
        Py_DECREF(view);
        return NULL;
    }
    int same = made == view;
    Py_DECREF(view);
    Py_DECREF(made);
    if (!same) {
        /* Never for an array with dimensions, which is all that came this far. */
        PyErr_SetString(PyExc_RuntimeError, "an in-place operator on an array gave another object");
        return NULL;
    }
    written_at_once++;
    if (strict && strict_now(target, held) < 0) {
        return NULL;
    }
    return Py_NewRef(target);
}

static PyObject *
in_place(PyObject *target, PyObject *other, PyObject *apply, PyObject *name, PyObject *perform, PyFrameObject *frame)
{
    PyObject *result = in_place_now(target, other, apply), *read = NULL, *key = NULL;
    if (result != Miss) {
        return result;
    }
    PyObject *node = node_of(target);
    if (node == NULL) {
        return NULL;
    }
    if (Py_TYPE(node) != Node || PyTuple_GET_SIZE(SLOT(node, node_at[N_SHAPE])) == 0 || writable(target) == NULL) {
        goto done; /* A number is not written in place, and the rest Python decides. */
    }
    read = Py_TYPE(other) == Lazy ? node_of(other) : Py_NewRef(other);
    if (read == NULL) {
        goto fail;
    }
    /* What `array._fit_key` writes: what makes the write, the operand, the target. */
    PyObject *operand = operand_part(read, NULL);
    PyObject *aim = operand && operand != Miss ? operand_part(node, NULL) : operand;
    if (aim == NULL || aim == Miss) {
        if (operand != NULL && operand != Miss) {
            Py_DECREF(operand);
        }
        if (aim == NULL) {
            goto fail;
        }
        goto done;
    }
    key = PyTuple_Pack(3, apply, operand, aim);
    Py_DECREF(operand);
    Py_DECREF(aim);
    int fits = key ? PySet_Contains(fitting, key) : -1;
    if (fits < 0) {
        goto fail;
    }
    if (fits) {
        PyObject *reads = PyTuple_Pack(1, read);
        int written = reads ? defer_write(target, name, perform, reads, frame, apply) : -1;
        Py_XDECREF(reads);
        if (written < 0) {
            goto fail;
        }
        if (written == 0) {
            result = Py_NewRef(target);
        }
    }
done:
    Py_XDECREF(read);
    Py_XDECREF(key);
    Py_DECREF(node);
    return result;
fail:
    Py_XDECREF(read);
    Py_XDECREF(key);
    Py_DECREF(node);
    return NULL;
}

/* `x[index] = value` for an index of whole numbers and slices and a value known to fit: the quick road of
 * `LazyArray.__setitem__`, by what `array._plain_assignment_fits` vouches for. None (new reference), `Miss`, or NULL. */
static PyObject *
set_item(PyObject *target, PyObject *index, PyObject *value, PyFrameObject *frame)
{
    if (!(PyLong_CheckExact(index) || PySlice_Check(index) || PyTuple_CheckExact(index))) {
        return Miss;
    }
    if (Py_TYPE(value) == Lazy && SLOT(value, lazy_at[L_HELD]) == SLOT(target, lazy_at[L_HELD])) {
        return Miss; /* A view of the same array, which the road looked at before it came here: Python looks. */
    }
    PyObject *whole = writable(target);
    if (whole == NULL) {
        return Miss;
    }
    PyObject *held = SLOT(whole, node_at[N_SHAPE]);
    PyObject *steps = SLOT(target, lazy_at[L_STEPS]);
    for (Py_ssize_t each = 0; each < PyTuple_GET_SIZE(steps); each++) {
        PyObject *step = PyTuple_GET_ITEM(steps, each);
        if (SLOT(step, step_at[S_PLAIN]) != Py_True) {
            return Miss;
        }
        held = SLOT(step, step_at[S_SHAPE]);
    }
    PyObject *result = Miss, *read = NULL, *shape = NULL, *assign = NULL;
    PyObject *selected = basic_shape(held, index);
    if (selected == NULL || selected == Miss) {
        return selected;
    }
    /* `array._shape_of(value)`, for a lazy array, an `ndarray` and a Python number. */
    if (Py_TYPE(value) == Lazy) {
        read = node_of(value);
        shape = read && Py_TYPE(read) == Node ? Py_NewRef(SLOT(read, node_at[N_SHAPE])) : NULL;
    } else if (Py_TYPE(value) == Ndarray) {
        read = Py_NewRef(value);
        shape = PyObject_GetAttr(value, s_shape);
    } else if (PyLong_CheckExact(value) || PyBool_Check(value) || PyFloat_CheckExact(value) || PyComplex_CheckExact(value)) {
        read = Py_NewRef(value);
        shape = Py_NewRef(empty_tuple);
    }
    if (shape == NULL || !PyTuple_CheckExact(shape)) {
        if (PyErr_Occurred()) {
            goto fail;
        }
        goto done;
    }
    /* `rules.broadcasts_into(shape, selected)`. */
    Py_ssize_t ahead = PyTuple_GET_SIZE(selected) - PyTuple_GET_SIZE(shape);
    if (ahead < 0) {
        goto done;
    }
    for (Py_ssize_t axis = 0; axis < PyTuple_GET_SIZE(shape); axis++) {
        PyObject *length = PyTuple_GET_ITEM(shape, axis);
        int fits = PyObject_RichCompareBool(length, one, Py_EQ);
        if (fits == 0) {
            fits = PyObject_RichCompareBool(length, PyTuple_GET_ITEM(selected, ahead + axis), Py_EQ);
        }
        if (fits < 0) {
            goto fail;
        }
        if (!fits) {
            goto done;
        }
    }
    assign = Assign->tp_alloc(Assign, 0);
    if (assign == NULL) {
        goto fail;
    }
    SLOT(assign, assign_at[A_INDEX]) = Py_NewRef(index);
    PyObject *reads = PyTuple_Pack(1, read);
    int written = reads ? defer_write(target, s_setitem, assign, reads, frame, NULL) : -1;
    Py_XDECREF(reads);
    if (written < 0) {
        goto fail;
    }
    if (written == 0) {
        result = Py_NewRef(Py_None);
    }
done:
    Py_DECREF(selected);
    Py_XDECREF(read);
    Py_XDECREF(shape);
    Py_XDECREF(assign);
    return result;
fail:
    Py_DECREF(selected);
    Py_XDECREF(read);
    Py_XDECREF(shape);
    Py_XDECREF(assign);
    return NULL;
}

/* `x[index] = value` through an index that holds pending arrays, for an assignment remembered to fit: the quick
 * road of that branch of `LazyArray.__setitem__`, by the key `array._assignment_key` writes. None, `Miss`, or NULL. */
static PyObject *
set_through(PyObject *target, PyObject *index, PyObject *value, PyFrameObject *frame)
{
    int whole = PyTuple_CheckExact(index);
    PyObject *aim = (!whole && PyTuple_Check(index)) ? NULL : writable(target);
    if (aim == NULL || (Py_TYPE(value) == Lazy && SLOT(value, lazy_at[L_HELD]) == SLOT(target, lazy_at[L_HELD]))) {
        return Miss;
    }
    PyObject *held = Py_None, *steps = SLOT(target, lazy_at[L_STEPS]);
    for (Py_ssize_t each = 0; each < PyTuple_GET_SIZE(steps); each++) {
        PyObject *step = PyTuple_GET_ITEM(steps, each);
        if (SLOT(step, step_at[S_PLAIN]) != Py_True) {
            return Miss;
        }
        held = SLOT(step, step_at[S_SHAPE]);
    }
    PyObject *single[1] = {index};
    PyObject **items = whole ? &PyTuple_GET_ITEM(index, 0) : single;
    Py_ssize_t count = whole ? PyTuple_GET_SIZE(index) : 1, found = 0;
    int strict = 0, numbers = 1;
    Marks marks = {0};
    PyObject *result = Miss, *read = NULL, *shape = NULL, *unwrapped = NULL, *part = NULL, *written = NULL, *key = NULL,
             *assign = NULL, *reads = NULL;
    PyObject *template = PyTuple_New(count), *with_nodes = PyTuple_New(count);
    if (template == NULL || with_nodes == NULL) {
        goto fail;
    }
    int apart = index_apart(items, count, template, with_nodes, &found, &strict, &numbers);
    if (apart < 0) {
        goto fail;
    }
    if (apart == 0 || found == 0) {
        goto done;
    }
    /* `array._shape_of(value)`, for a lazy array, an `ndarray` and a Python number. */
    if (Py_TYPE(value) == Lazy) {
        read = node_of(value);
        shape = read && Py_TYPE(read) == Node ? Py_NewRef(SLOT(read, node_at[N_SHAPE])) : NULL;
    } else if (Py_TYPE(value) == Ndarray) {
        read = Py_NewRef(value);
        shape = PyObject_GetAttr(value, s_shape);
    } else if (PyLong_CheckExact(value) || PyBool_Check(value) || PyFloat_CheckExact(value) || PyComplex_CheckExact(value)) {
        read = Py_NewRef(value);
        shape = Py_NewRef(empty_tuple);
    }
    if (shape == NULL || !PyTuple_CheckExact(shape)) {
        if (PyErr_Occurred()) {
            goto fail;
        }
        goto done;
    }
    /* The key: the shape assigned into if the array is a view, the value's shape with a placeholder for a length
     * that is not known, and the index as a call's key writes it, on the cell's node. */
    unwrapped = PyTuple_Pack(2, aim, whole ? with_nodes : PyTuple_GET_ITEM(with_nodes, 0));
    part = unwrapped ? index_key(unwrapped, &marks) : NULL;
    if (part == NULL) {
        goto fail;
    }
    if (part == Miss) {
        part = NULL;
        goto done;
    }
    written = PyTuple_New(PyTuple_GET_SIZE(shape));
    for (Py_ssize_t axis = 0; written != NULL && axis < PyTuple_GET_SIZE(shape); axis++) {
        PyObject *length = PyTuple_GET_ITEM(shape, axis);
        if (is_stand_in(length)) {
            int place = mark_of(&marks, length);
            if (place < 0) {
                if (PyErr_Occurred()) {
                    goto fail;
                }
                goto done;
            }
            length = PyLong_FromLong(-place - 1);
            if (length == NULL) {
                goto fail;
            }
            PyTuple_SET_ITEM(written, axis, length);
        } else {
            PyTuple_SET_ITEM(written, axis, Py_NewRef(length));
        }
    }
    key = written ? PyTuple_Pack(3, held, written, part) : NULL;
    int fits = key ? PySet_Contains(assignable, key) : -1;
    if (fits < 0) {
        goto fail;
    }
    if (!fits) {
        goto done;
    }
    /* What the write reads: the value, then the index's pending arrays in the order of their slots. */
    reads = PyTuple_New(1 + found);
    assign = reads ? Through->tp_alloc(Through, 0) : NULL;
    if (assign == NULL) {
        goto fail;
    }
    PyTuple_SET_ITEM(reads, 0, Py_NewRef(read));
    for (Py_ssize_t place = 0, taken = 0; place < count; place++) {
        if (Py_TYPE(items[place]) == Lazy) {
            PyTuple_SET_ITEM(reads, 1 + taken++, Py_NewRef(PyTuple_GET_ITEM(with_nodes, place)));
        }
    }
    SLOT(assign, through_at[X_TEMPLATE]) = Py_NewRef(template);
    SLOT(assign, through_at[X_WHOLE]) = Py_NewRef(whole ? Py_True : Py_False);
    int done_write = defer_write(target, s_setitem, assign, reads, frame, NULL);
    if (done_write < 0) {
        goto fail;
    }
    if (done_write == 0) {
        result = Py_NewRef(Py_None);
    }
done:
    marks_clear(&marks);
    Py_XDECREF(template);
    Py_XDECREF(with_nodes);
    Py_XDECREF(read);
    Py_XDECREF(shape);
    Py_XDECREF(unwrapped);
    Py_XDECREF(part);
    Py_XDECREF(written);
    Py_XDECREF(key);
    Py_XDECREF(assign);
    Py_XDECREF(reads);
    return result;
fail:
    result = NULL;
    goto done;
}

/* A call that writes its result into `out=`, for a write that is known to fit: the quick road of
 * `array._write_out`. `function` is the ufunc or the NumPy function, `inputs` its `count` positional arguments.
 * Returns `target` (new reference), `Miss`, or NULL. */
static PyObject *
write_out(PyObject *function, PyObject *const *inputs, Py_ssize_t count, PyObject *target, PyFrameObject *frame)
{
    if (Py_TYPE(target) != Lazy || writable(target) == NULL) {
        return Miss;
    }
    PyObject *result = Miss, *aim = NULL, *into = NULL, *name = NULL;
    PyObject *reads = PyTuple_New(count), *key = PyTuple_New(count + 2);
    if (reads == NULL || key == NULL) {
        goto fail;
    }
    /* What `array._fit_key` writes: the function, each operand, the target; read in that order. */
    PyTuple_SET_ITEM(key, 0, Py_NewRef(function));
    for (Py_ssize_t place = 0; place <= count; place++) {
        PyObject *read = place == count ? node_of(target) : Py_TYPE(inputs[place]) == Lazy ? node_of(inputs[place]) : Py_NewRef(inputs[place]);
        if (read == NULL) {
            goto fail;
        }
        if (place == count) {
            aim = read;
        } else {
            PyTuple_SET_ITEM(reads, place, read);
        }
        PyObject *part = operand_part(read, NULL);
        if (part == NULL) {
            goto fail;
        }
        if (part == Miss) {
            goto done;
        }
        PyTuple_SET_ITEM(key, 1 + place, part);
    }
    int fits = PySet_Contains(fitting, key);
    if (fits < 0) {
        goto fail;
    }
    if (fits) {
        into = Into->tp_alloc(Into, 0);
        name = into ? PyObject_GetAttr(function, s_dunder_name) : NULL;
        if (name == NULL) {
            goto fail;
        }
        SLOT(into, into_function_at) = Py_NewRef(function);
        int written = defer_write(target, name, into, reads, frame, NULL);
        if (written < 0) {
            goto fail;
        }
        if (written == 0) {
            result = Py_NewRef(target);
        }
    }
done:
    Py_XDECREF(reads);
    Py_XDECREF(key);
    Py_XDECREF(aim);
    Py_XDECREF(into);
    Py_XDECREF(name);
    return result;
fail:
    result = NULL;
    goto done;
}

/* ---- `x[index]` and `x[index] = v` made at once (`small_limit`), on arrays that have their values. ---- */

/* Whether an index item is a whole number or a slice of whole numbers written in the program. */
static int
is_plain_part(PyObject *item)
{
    if (PyLong_CheckExact(item)) {
        return 1;
    }
    if (!PySlice_Check(item)) {
        return 0;
    }
    PySliceObject *slice = (PySliceObject *)item;
    return (slice->start == Py_None || PyLong_CheckExact(slice->start)) && (slice->stop == Py_None || PyLong_CheckExact(slice->stop))
        && (slice->step == Py_None || PyLong_CheckExact(slice->step));
}

/* One item of an index as NumPy can take it now: a whole number, a slice, `None`, `...` or an `ndarray` as it is,
 * and a lazy array that has its values as those values. `*basic` is cleared for an item that selects by an array,
 * which gives a copy and not a view, and `*plain` for one that is not a number or slice written in the program.
 * A new reference, or `Miss` for anything else: a pending array, a list, a nested tuple. */
static PyObject *
index_item_now(PyObject *item, int *basic, int *plain, int *strict)
{
    if (is_plain_part(item)) {
        return Py_NewRef(item);
    }
    *plain = 0;
    if (PySlice_Check(item) || item == Py_None || item == Py_Ellipsis || PyArray_IsScalar(item, Integer)) {
        /* A whole number of NumPy's, as `np.argmax` and `np.random.choice` give, indexes as Python's does. */
        return Py_NewRef(item);
    }
    if (Py_TYPE(item) == Ndarray) {
        *basic = 0;
        return Py_NewRef(item);
    }
    if (Py_TYPE(item) == Lazy) {
        PyObject *value = value_of(item, strict);
        if (value != NULL && value != Miss) {
            *basic = 0;
        }
        return value;
    }
    return Miss;
}

/* A whole index as NumPy can take it now, by `index_item_now` for it or for each item of a tuple. */
static PyObject *
index_now(PyObject *index, int *basic, int *plain, int *strict)
{
    if (!PyTuple_CheckExact(index)) {
        return index_item_now(index, basic, plain, strict);
    }
    Py_ssize_t count = PyTuple_GET_SIZE(index);
    PyObject *settled = PyTuple_New(count);
    if (settled == NULL) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *item = index_item_now(PyTuple_GET_ITEM(index, place), basic, plain, strict);
        if (item == NULL || item == Miss) {
            Py_DECREF(settled);
            return item;
        }
        PyTuple_SET_ITEM(settled, place, item);
    }
    return settled;
}

/* Whether a node's element type is one the layer defers and none of its lengths is zero: `rules.is_admissible`,
 * remembered. 1, 0, or -1 with an error set. */
static int
is_admissible(PyObject *node)
{
    PyObject *spec = spec_part(node, NULL);
    if (spec == NULL) {
        return -1;
    }
    if (spec == Miss) {
        return 0;
    }
    PyObject *allowed = PyDict_GetItemWithError(admissible, spec);
    Py_DECREF(spec);
    if (allowed == NULL && !PyErr_Occurred()) {
        /* Not asked about yet: the rules answer and remember, so this is once for each shape and type. */
        PyObject *ask = PyDict_GetItemString(rules_names, "is_admissible");
        PyObject *answer = ask ? PyObject_CallOneArg(ask, node) : NULL;
        if (answer == NULL) {
            return -1;
        }
        int said = PyObject_IsTrue(answer);
        Py_DECREF(answer);
        return said;
    }
    if (allowed != Py_True) {
        return PyErr_Occurred() ? -1 : 0;
    }
    return 1;
}

/* `x[index]` on an array that has its values, made now: NumPy's own indexing of the array, where the program wrote
 * it. A view is a lazy array on the same cell with one more step, as a deferred view is; a number or a selection is
 * an array of its own. Returns a new lazy array, `Miss` for what the other roads decide, or NULL. */
static PyObject *lazy_of_value(PyObject *value, PyObject *source, PyObject *kind, PyObject *strict);
static int is_observed(void);

static PyObject *
item_now(PyObject *target, PyObject *index, PyFrameObject *frame)
{
    if (small_limit == 0) {
        return Miss;
    }
    PyObject *result = Miss, *settled = NULL, *value = NULL, *shape = NULL, *made = NULL, *origin = NULL;
    int basic = 1, plain = 1, strict = 0;
    PyObject *memory = value_of(target, &strict);
    if (memory == NULL) {
        return NULL;
    }
    if (memory == Miss) {
        return Miss;
    }
    if (Py_TYPE(memory) != Ndarray || !value_admissible(memory)) {
        goto done;
    }
    settled = index_now(index, &basic, &plain, &strict);
    if (settled == NULL) {
        goto fail;
    }
    if (settled == Miss) {
        settled = NULL;
        goto done;
    }
    value = PyObject_GetItem(memory, settled);
    if (value == NULL) {
        goto fail;
    }
    if (!basic || Py_TYPE(value) != Ndarray) {
        /* One element, or a selection by an array: an array of its own. */
        if (!is_arraylike(value)) {
            goto done;
        }
        result = lazy_of_value(value, s_getitem, at_once_kind, strict ? Py_True : Py_False);
        value = NULL; /* Stolen. */
        goto done;
    }
    /* A view. It holds NumPy's view and nothing else, like any result of a call made at once: NumPy's view shows
     * what the memory holds whatever is written there, and `writable` sees to it that no write waits while it may
     * be alive. While something observes the layer it is a lazy array on the same cell with one more step, as a
     * deferred view is, so that a recording has it as a view. */
    int observed = is_observed();
    if (observed < 0) {
        goto fail;
    }
    if (!observed) {
        result = lazy_holding(value, 1, strict);
        value = NULL; /* Stolen. */
        goto done;
    }
    if (frame == NULL) {
        frame = PyEval_GetFrame();
        if (frame == NULL) {
            goto done;
        }
    }
    PyObject *cell = cell_of(target);
    PyObject *all = cell ? steps_now(SLOT(target, lazy_at[L_STEPS]), getitem, s_getitem, settled, value, plain, frame) : NULL;
    if (all == NULL) {
        goto fail;
    }
    PyObject *step = PyTuple_GET_ITEM(all, PyTuple_GET_SIZE(all) - 1);
    made = value_node(value, s_getitem, SLOT(step, step_at[S_SHAPE]), SLOT(step, step_at[S_DTYPE]),
                      strict ? Py_True : Py_False, zero, SLOT(step, step_at[S_ORIGIN]), at_once_kind);
    value = NULL; /* Stolen. */
    result = made ? new_view(made, cell, all) : NULL;
    Py_DECREF(all);
done:
    Py_XDECREF(settled);
    Py_XDECREF(value);
    Py_XDECREF(shape);
    Py_XDECREF(made);
    Py_XDECREF(origin);
    Py_DECREF(memory);
    return result;
fail:
    Py_XDECREF(settled);
    Py_XDECREF(value);
    Py_XDECREF(shape);
    Py_XDECREF(made);
    Py_XDECREF(origin);
    Py_DECREF(memory);
    return NULL;
}

/* `x.T` and `x.transpose()` of an array that has its values, made now: NumPy's own, and a lazy array on the same
 * cell with one more step, which is `operation`, `np.transpose`. Returns a new lazy array, `Miss` for
 * what the ordinary road decides, or NULL. */
static PyObject *
view_now(PyObject *target, PyObject *operation, PyObject *name, PyFrameObject *frame)
{
    if (small_limit == 0) {
        return Miss;
    }
    PyObject *result = Miss, *value = NULL, *shape = NULL, *made = NULL, *origin = NULL;
    int strict = 0;
    PyObject *memory = value_of(target, &strict);
    if (memory == NULL) {
        return NULL;
    }
    if (memory == Miss) {
        return Miss;
    }
    if (Py_TYPE(memory) != Ndarray || !value_admissible(memory)) {
        goto done;
    }
    /* `np.transpose(x)` with no axes is `x.transpose()`, which NumPy's function reaches through two more calls. */
    value = PyObject_CallMethodNoArgs(memory, s_transpose);
    if (value == NULL) {
        goto fail;
    }
    if (Py_TYPE(value) != Ndarray) {
        goto done;
    }
    int observed = is_observed();
    if (observed < 0) {
        goto fail;
    }
    if (!observed) {
        result = lazy_holding(value, 1, strict); /* As for `x[index]`: see `item_now`. */
        value = NULL; /* Stolen. */
        goto done;
    }
    PyObject *cell = cell_of(target);
    PyObject *all = cell ? steps_now(SLOT(target, lazy_at[L_STEPS]), operation, name, NULL, value, 1, frame) : NULL;
    if (all == NULL) {
        goto fail;
    }
    PyObject *step = PyTuple_GET_ITEM(all, PyTuple_GET_SIZE(all) - 1);
    made = value_node(value, name, SLOT(step, step_at[S_SHAPE]), SLOT(step, step_at[S_DTYPE]),
                      strict ? Py_True : Py_False, zero, SLOT(step, step_at[S_ORIGIN]), at_once_kind);
    value = NULL; /* Stolen. */
    result = made ? new_view(made, cell, all) : NULL;
    Py_DECREF(all);
done:
    Py_XDECREF(value);
    Py_XDECREF(shape);
    Py_XDECREF(made);
    Py_XDECREF(origin);
    Py_DECREF(memory);
    return result;
fail:
    Py_XDECREF(value);
    Py_XDECREF(shape);
    Py_XDECREF(made);
    Py_XDECREF(origin);
    Py_DECREF(memory);
    return NULL;
}

/* Whether `value` is `target[index]` as a view the layer made, for a plain index: `LazyArray._is_view_at`, with
 * the test of the cell that goes before it. `x[i] += v` takes the view, updates it, and assigns it back to where
 * it already is, which is nothing to do. 0 also for an index this file does not compare, which Python then does. */
static int
is_own_view_at(PyObject *target, PyObject *index, PyObject *value)
{
    if (Py_TYPE(value) != Lazy || SLOT(value, lazy_at[L_HELD]) != SLOT(target, lazy_at[L_HELD])
        || Py_TYPE(SLOT(target, lazy_at[L_HELD])) != Cell) {
        return 0;
    }
    PyObject *mine = SLOT(target, lazy_at[L_STEPS]), *theirs = SLOT(value, lazy_at[L_STEPS]);
    Py_ssize_t count = PyTuple_GET_SIZE(mine);
    if (PyTuple_GET_SIZE(theirs) != count + 1) {
        return 0;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        if (PyTuple_GET_ITEM(mine, place) != PyTuple_GET_ITEM(theirs, place)) {
            return 0;
        }
    }
    PyObject *last = PyTuple_GET_ITEM(theirs, count);
    PyObject *tail = SLOT(last, step_at[S_TAIL]);
    return SLOT(last, step_at[S_OPERATION]) == getitem && PyTuple_GET_SIZE(tail) == 1
        && same_plain_index(PyTuple_GET_ITEM(tail, 0), index);
}

/* `x[index] = value` made now, into the array's own memory, when nothing at all is pending: NumPy's own assignment,
 * where the program wrote it. Returns None (new reference), `Miss` for what the other roads decide, or NULL. */
static PyObject *
set_now(PyObject *target, PyObject *index, PyObject *value)
{
    if (small_limit == 0 || PySet_GET_SIZE(pending) != 0) {
        return Miss;
    }
    PyObject *held = NULL;
    PyObject *view = memory_now(target, &held);
    if (view == NULL || view == Miss) {
        return view;
    }
    PyObject *result = Miss, *settled = NULL, *assigned = NULL;
    int basic = 1, plain = 1, strict = 0;
    settled = index_now(index, &basic, &plain, &strict);
    if (settled == NULL) {
        goto fail;
    }
    if (settled == Miss) {
        settled = NULL;
        goto done;
    }
    if (Py_TYPE(value) == Lazy) {
        assigned = value_of(value, &strict);
        if (assigned == NULL) {
            goto fail;
        }
        if (assigned == Miss) {
            assigned = NULL;
            goto done;
        }
    } else if (PyTuple_Check(value) || PyList_Check(value) || PyDict_Check(value)) {
        goto done; /* May hold a lazy array: Python looks. */
    } else {
        assigned = Py_NewRef(value);
    }
    if (PyObject_SetItem(view, settled, assigned) < 0) {
        goto fail;
    }
    written_at_once++;
    if (strict && strict_now(target, held) < 0) {
        goto fail;
    }
    result = Py_NewRef(Py_None);
done:
    Py_XDECREF(settled);
    Py_XDECREF(assigned);
    Py_DECREF(view);
    return result;
fail:
    Py_XDECREF(settled);
    Py_XDECREF(assigned);
    Py_DECREF(view);
    return NULL;
}

/* The number of elements of a shape whose lengths are all known, or -1. */
static Py_ssize_t
size_of(PyObject *shape)
{
    Py_ssize_t size = 1;
    for (Py_ssize_t axis = 0; axis < PyTuple_GET_SIZE(shape); axis++) {
        PyObject *length = PyTuple_GET_ITEM(shape, axis);
        if (!PyLong_CheckExact(length)) {
            return -1;
        }
        Py_ssize_t count = PyLong_AsSsize_t(length);
        if (count < 0 || (count != 0 && size > PY_SSIZE_T_MAX / count)) {
            PyErr_Clear();
            return -1;
        }
        size *= count;
    }
    return size;
}

/* Whether something is listening to what the layer makes: `graph.observer`. 1, 0, or -1 with an error set. */
static int
is_observed(void)
{
    PyObject *observer = PyDict_GetItemWithError(graph_names, s_observer);
    if (observer == NULL) {
        return PyErr_Occurred() ? -1 : 0;
    }
    return observer != Py_None;
}

/* A lazy array for a value NumPy has just given: an `ndarray` or a NumPy scalar. Steals the reference to `value`.
 * `kind` is what its node says of itself, the kind of a call made at once or the empty string, and `source` what
 * an observer is told made it. While calls are made at once and nothing observes, the array holds the value alone
 * (`lazy_holding`); otherwise it is on a new node, there and then, so that a recording has its nodes in the order
 * they were made and a program run with every call deferred has the numbers it had. */
static PyObject *
lazy_of_value(PyObject *value, PyObject *source, PyObject *kind, PyObject *strict)
{
    if (small_limit != 0) {
        int observed = is_observed();
        if (observed < 0) {
            Py_DECREF(value);
            return NULL;
        }
        if (!observed) {
            return lazy_holding(value, kind == at_once_kind, strict == Py_True);
        }
    }
    PyObject *shape = PyObject_GetAttr(value, s_shape);
    PyObject *dtype = shape ? PyObject_GetAttr(value, s_dtype) : NULL;
    if (dtype == NULL || !PyTuple_CheckExact(shape)) {
        if (dtype != NULL) {
            PyErr_SetString(PyExc_TypeError, "a value NumPy gave has a shape that is no tuple");
        }
        Py_XDECREF(shape);
        Py_XDECREF(dtype);
        Py_DECREF(value);
        return NULL;
    }
    PyObject *node = value_node(value, source, shape, dtype, strict, zero, NULL, kind);
    Py_DECREF(shape);
    Py_DECREF(dtype);
    if (node == NULL) {
        return NULL;
    }
    PyObject *lazy = new_lazy(node);
    Py_DECREF(node);
    return lazy;
}

/* A lazy array for a node that has just been made: `new_lazy`, but that a node which came back from `deferred`
 * with its value, because the call was made at once, gives the value to an array that holds it alone. Nothing
 * else refers to such a node yet, so nothing can tell. Returns a new reference. */
static PyObject *
lazy_for(PyObject *node)
{
    if (small_limit != 0 && Py_REFCNT(node) == 1 && has_value(node) && SLOT(node, node_at[N_KIND]) == at_once_kind
        && SLOT(node, node_at[N_EXPOSED]) == Py_False && is_observed() == 0) {
        return lazy_holding(Py_NewRef(SLOT(node, node_at[N_VALUE])), 1, SLOT(node, node_at[N_STRICT]) == Py_True);
    }
    if (PyErr_Occurred()) {
        return NULL;
    }
    return new_lazy(node);
}

/* The value of a pending operand of a small call, computed now. `y[i, :] = a * out[i, :] + b * y[i - 1, :]` in a
 * loop, with `out` a large array still pending, is thousands of operations on rows of a thousand elements, and
 * every one of them waited as a node because the row of `out` it reads had no value: 36,000 nodes, a plan and a
 * program of kernels for each row, 170 ms where NumPy takes 37 ms. A call on small operands is never worth a node,
 * whatever its operands wait for. So a pending operand with fewer than `small_limit` elements is computed, with
 * what it reads, by the function the layer set (`small_reads`), and the call is made at once as on any value.
 * Returns the value (a new reference), `Miss` for an operand that is large, of a length not known, failed or
 * taken by a write, or NULL with the error computing raised, which is the program's at that line. */
static PyObject *
small_pending(PyObject *operand, int *strict)
{
    if (compute_small == NULL) {
        return Miss;
    }
    PyObject *node = node_of(operand);
    if (node == NULL) {
        return NULL;
    }
    int waits = Py_TYPE(node) == Node && SLOT(node, node_at[N_OPERATION]) != Py_None
        && SLOT(node, node_at[N_FAILURE]) == Py_None && PyTuple_CheckExact(SLOT(node, node_at[N_SHAPE]));
    Py_ssize_t size = waits ? size_of(SLOT(node, node_at[N_SHAPE])) : -1;
    Py_DECREF(node);
    if (size < 0 || size >= small_limit) {
        return Miss;
    }
    PyObject *done = PyObject_CallOneArg(compute_small, operand);
    if (done == NULL) {
        return NULL;
    }
    Py_DECREF(done);
    return value_of(operand, strict);
}

/* A ufunc called now on operands that have their values (`small_limit`): NumPy's own call, with no rule asked
 * and nothing made but the lazy array that holds the result. Taken when every operand is a lazy array with a value,
 * or a small one still pending, which is computed first (`small_pending`), or a plain `ndarray`, or a Python
 * number, one at least is an array, and either all have fewer than `small_limit` elements or the call is one no
 * kernel could make. `operator` says that the callable is what an operator's road calls, which is a ufunc or
 * `operator.pow`; a ufunc called by name is taken only with as many operands as it has inputs, since one more by
 * position is its output, which is a write (`np.add(a, b, c)`). `strict` says that the result is strict whatever
 * its operands are: the stand-in that made the call is, and a plain array it is given is strict then. Returns a
 * new lazy array, `Miss` for what the ordinary road decides, or NULL with NumPy's error, which is the program's
 * at that line.
 *
 * A plain `ndarray` is an operand like a lazy array that has its values: a library's own arrays are plain
 * whatever its `np` is, and `plain + lazy`, `np.sqrt(plain)` through the stand-in and `lazy * plain` went the
 * ordinary road for it, which made the plain array lazy first (a node, a cell and a lazy array for each) and
 * came back through NumPy's protocol: 0.5 to 1.3 µs a call over NumPy's own on 20,000 elements, for 0.1 µs
 * here. Not while something observes the layer: a recording has a node for each array that came in. */
static PyObject *
ufunc_on(PyObject *ufunc, PyObject *const *operands, Py_ssize_t count, int operator, int strict)
{
    if (small_limit == 0 || count < 1 || count > 3 || (!operator && Py_TYPE(ufunc) != Ufunc)) {
        return Miss;
    }
    if (!operator && ((PyUFuncObject *)ufunc)->nin != count) {
        return Miss;
    }
    PyObject *values[3], *held[3] = {NULL, NULL, NULL};
    PyObject *result = Miss;
    int arrays = 0;
    int large = 0, other_type = 0; /* Whether an operand has `small_limit` elements or more; is of another type. */
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *operand = operands[place];
        if (Py_TYPE(operand) == Lazy) {
            PyObject *value = value_of(operand, &strict);
            if (value == Miss) {
                value = small_pending(operand, &strict);
            }
            if (value == NULL) {
                result = NULL;
                goto done;
            }
            if (value == Miss) {
                goto done;
            }
            held[place] = value; /* Kept for the length of the call, whatever the call does to the lazy array. */
            Py_ssize_t size = count_of(value);
            if (size < 0) {
                goto done;
            }
            large = large || size >= small_limit;
            /* An array of another type, but for one whole number: a kernel takes that as a number (`leaf_kind`). */
            other_type = other_type || (Py_TYPE(value) == Ndarray && (PyObject *)PyArray_DESCR((PyArrayObject *)value) != fusible_dtype
                                        && !(PyArray_NDIM((PyArrayObject *)value) == 0
                                             && whole_kind((PyObject *)PyArray_DESCR((PyArrayObject *)value))));
            values[place] = value;
            arrays++;
        } else if (Py_TYPE(operand) == Ndarray) {
            int observed = is_observed();
            if (observed != 0) {
                result = observed < 0 ? NULL : Miss;
                goto done;
            }
            large = large || PyArray_SIZE((PyArrayObject *)operand) >= small_limit;
            other_type = other_type || (PyObject *)PyArray_DESCR((PyArrayObject *)operand) != fusible_dtype;
            values[place] = operand;
            arrays++;
        } else if (PyFloat_CheckExact(operand) || PyLong_CheckExact(operand) || PyBool_Check(operand)
                   || PyComplex_CheckExact(operand)) {
            values[place] = operand;
        } else if (PyObject_TypeCheck(operand, Generic)) {
            /* A NumPy scalar of the program's own, `np.exp(-alpha)` of a number: NumPy's call takes it as it is.
             * Left to the ordinary road, `k * row` with such a `k` took 650 to 900 ns where `0.5 * row` took 360. */
            values[place] = operand;
        } else {
            goto done;
        }
    }
    if (arrays == 0) {
        goto done;
    }
    if (large) {
        /* A large operand. The call is still made now if no kernel could make it, as `ready_now` has it for a
         * node: a comparison, a call into a mathematics library, anything on another element type. One a kernel
         * can make waits, so that what is written on top of it can be computed with it. */
        if (fusible_calls == NULL) {
            goto done;
        }
        if (!other_type) {
            int waits = PySet_Contains(fusible_calls, ufunc);
            if (waits != 0) {
                PyErr_Clear(); /* A callable that cannot be looked up is none of them, but is left to its node. */
                goto done;
            }
        }
    }
    PyObject *value = PyObject_Vectorcall(ufunc, values, (size_t)count, NULL);
    if (value == NULL) {
        result = NULL;
        goto done;
    }
    if (!is_arraylike(value)) {
        result = value; /* Not an array: whatever NumPy answered is the answer. */
        goto done;
    }
    made_at_once++;
    result = lazy_of_value(value, s_array, at_once_kind, strict ? Py_True : Py_False);
done:
    Py_XDECREF(held[0]);
    Py_XDECREF(held[1]);
    Py_XDECREF(held[2]);
    return result;
}

/* `ufunc_on` for operands whose strictness is all there is to say: every road but the stand-in's. */
static PyObject *
ufunc_now(PyObject *ufunc, PyObject *const *operands, Py_ssize_t count, int operator)
{
    return ufunc_on(ufunc, operands, count, operator, 0);
}

/* A ufunc called with `out=` made now, into the memory the output has, when nothing at all is pending
 * (`small_limit`): NumPy's own call on the values with the output's array as `out`, which is what the eager
 * program does at that line, whoever else knows the memory. `np.maximum(x, 0, out=x)`, `np.exp(x, out=x)`, and
 * `a += x` on a plain `a`, which NumPy makes as `np.add(a, x, out=(a,))` and hands to the lazy `x`. `target` is
 * a lazy array with dimensions, of an element type the layer defers, that has its values (`memory_now`), or a
 * plain `ndarray`; every operand a lazy array that has its values, a plain `ndarray` or a number. `strict` says
 * that what is written is strict whatever the operands are, as for `ufunc_on`.
 *
 * With nothing pending no operation written before the write still waits to read what the output held, so the
 * copy a deferred write is defined by has no reader, and the write into the memory is NumPy's own meaning, views
 * included: what `write_now` and `in_place_now` rest on. Those are reached for an in-place operator and for a
 * write the layer may defer and has seen to fit. This is for every other: an output the program was handed, or
 * one that is a plain array, went to `array._write` in Python, 4.5 to 6 µs a call over NumPy's own. A strict
 * value does not go into a plain array, and nothing is done while something observes the layer: both are
 * Python's to answer. Returns `target` (a new reference), `Miss`, or NULL with NumPy's error, which is the
 * program's at that line and leaves the output as it was. */
static PyObject *
ufunc_into_now(PyObject *ufunc, PyObject *const *operands, Py_ssize_t count, PyObject *target, int strict)
{
    int lazy_target = Py_TYPE(target) == Lazy;
    if (small_limit == 0 || count < 1 || count > 3 || Py_TYPE(ufunc) != Ufunc || PySet_GET_SIZE(pending) != 0
        || (!lazy_target && Py_TYPE(target) != Ndarray)) {
        return Miss;
    }
    if (((PyUFuncObject *)ufunc)->nin != count || ((PyUFuncObject *)ufunc)->nout != 1) {
        return Miss; /* Not a call the ufunc has, or one with two outputs: NumPy's to answer, by the ordinary road. */
    }
    int observed = is_observed();
    if (observed != 0) {
        return observed < 0 ? NULL : Miss;
    }
    PyObject *stack[4], *values[3] = {NULL, NULL, NULL};
    PyObject *result = Miss, *held = NULL, *view = NULL;
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *operand = operands[place];
        if (Py_TYPE(operand) == Lazy) {
            PyObject *value = value_of(operand, &strict);
            if (value == NULL) {
                result = NULL;
                goto done;
            }
            if (value == Miss) {
                goto done;
            }
            values[place] = value; /* Kept for the length of the call. */
            stack[place] = value;
        } else if (Py_TYPE(operand) == Ndarray || PyFloat_CheckExact(operand) || PyLong_CheckExact(operand)
                   || PyBool_Check(operand) || PyComplex_CheckExact(operand) || PyObject_TypeCheck(operand, Generic)) {
            stack[place] = operand;
        } else {
            goto done;
        }
    }
    if (strict && !lazy_target) {
        goto done; /* The values of a strict array would leave into a plain one: Python refuses. */
    }
    view = lazy_target ? memory_now(target, &held) : Py_NewRef(target);
    if (view == NULL || view == Miss) {
        result = view;
        view = NULL;
        goto done;
    }
    stack[count] = view;
    PyObject *made = PyObject_Vectorcall(ufunc, stack, (size_t)count, out_names);
    if (made == NULL) {
        result = NULL;
        goto done;
    }
    Py_DECREF(made);
    written_at_once++;
    result = strict && strict_now(target, held) < 0 ? NULL : Py_NewRef(target);
done:
    Py_XDECREF(view);
    Py_XDECREF(values[0]);
    Py_XDECREF(values[1]);
    Py_XDECREF(values[2]);
    return result;
}

/* A lazy array for an `ndarray` the program holds, as `array.wrap(value, strict=strict)` makes one: on a node that
 * has the array for its value and says that its memory is known outside the layer (`exposed`), so that no write
 * into it is made to wait. Not for a time when something observes the layer, which `graph.leaf` tells of the
 * node with what made it. Returns a new reference. */
static PyObject *
lazy_of_handed(PyObject *value, PyObject *strict)
{
    PyObject *shape = PyObject_GetAttr(value, s_shape);
    PyObject *dtype = shape ? PyObject_GetAttr(value, s_dtype) : NULL;
    PyObject *node = dtype && PyTuple_CheckExact(shape)
        ? value_node(Py_NewRef(value), s_array, shape, dtype, strict, zero, NULL, empty_string) : NULL;
    Py_XDECREF(shape);
    Py_XDECREF(dtype);
    if (node == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "an array has a shape that is no tuple");
        }
        return NULL;
    }
    Py_SETREF(SLOT(node, node_at[N_EXPOSED]), Py_NewRef(Py_True));
    PyObject *lazy = new_lazy(node);
    Py_DECREF(node);
    return lazy;
}

/* `graph.fallback(graph.AT_ONCE, call)`: count one call that is outside the deferred set and was made at once, by
 * the line of the program that wrote it, and log the line the first time. The function is Python's, read each
 * time since a test may rebind it; it finds the program's line as it does for the Python road, from the frame
 * that called into this file. 0, or -1 with an error set. */
static int
count_at_once(PyObject *call)
{
    PyObject *fallback = PyDict_GetItemWithError(graph_names, s_fallback);
    if (fallback == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_AttributeError, "graph.fallback is gone");
        }
        return -1;
    }
    PyObject *counted = PyObject_CallFunctionObjArgs(fallback, at_once_kind, call, NULL);
    if (counted == NULL) {
        return -1;
    }
    Py_DECREF(counted);
    return 0;
}

/* ---- A road: what stands on `LazyArray` where a method stood, takes the calls it can, and hands the others
 * to the method. ---- */

/* The kinds of road, by the method each stands for. */
enum { BINARY, UNARY, ARRAY_UFUNC, ARRAY_FUNCTION, REDUCTION, GETITEM, ATTRIBUTE, INPLACE, SETITEM, STANDIN_CALL,
       STANDIN_NAME, RESHAPE, TRANSPOSE, SWAPAXES, RAVEL, LEAVE, ROAD_KINDS };
static const char *const ROAD_NAMES[ROAD_KINDS] = {"binary", "unary", "array_ufunc", "array_function", "reduction", "getitem",
                                                   "attribute", "inplace", "setitem", "standin_call", "standin_name",
                                                   "reshape", "transpose", "swapaxes", "ravel", "leave"};
/* The most arguments, the array among them, a value is handed out with by the road of `array._leaving`. */
#define MOST_LEAVING 6

typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    int kind;            /* One of the kinds above. */
    PyObject *fallback;  /* The Python method, which takes every call this road does not. */
    PyObject *operation; /* The ufunc of an operator; the callable a reduction's node calls; for
                            `__array_function__`, the functions it answers by a road of their own. */
    PyObject *name;      /* What the node is called, for an operator and a reduction. */
    PyObject *ruled_as;  /* For a reduction, the NumPy function whose rule gives the shape and element type. For an
                            in-place operator, what makes the write into the copy. */
    int reflected;       /* For a binary operator: whether the lazy array is on the right. */
} Road;

/* Whether an argument is a value written in the program that holds no array: a number, a string, None, a type
 * or an element type, or a tuple of whole numbers such as an axis or a shape. */
static int
is_plain_argument(PyObject *arg)
{
    if (arg == Py_None || PyLong_CheckExact(arg) || PyFloat_CheckExact(arg) || PyBool_Check(arg) || PyUnicode_CheckExact(arg)
        || PyType_Check(arg) || PyArray_DescrCheck(arg)) {
        return 1;
    }
    if (!PyTuple_CheckExact(arg)) {
        return 0;
    }
    for (Py_ssize_t part = 0; part < PyTuple_GET_SIZE(arg); part++) {
        if (!PyLong_CheckExact(PyTuple_GET_ITEM(arg, part))) {
            return 0;
        }
    }
    return 1;
}

/* The most arguments, the array among them, a method called by `method_now` may have. */
#define MOST_METHOD_ARGUMENTS 8

/* `x.sum()`, `x.max(axis=0)`, `x.dot(y)` and their like on small arrays that have their values (`small_limit`):
 * NumPy's own method of the array, called now, with no rule asked. Only for the methods whose NumPy function is
 * listed as giving one new array that shares no memory with an argument (`fresh_functions`), on arrays of the
 * element types the layer defers and with no length of zero, and with every other argument a plain value: the
 * calls the rules would have had written as a node, whose node would have made this very call. `args[0]` is the
 * lazy array and `name` the method's name. Returns a new lazy array, `Miss` for what the ordinary road decides,
 * or NULL with NumPy's error, which is the program's at that line. */
static PyObject *
method_now(PyObject *name, PyObject *function, PyObject *const *args, Py_ssize_t count, PyObject *kwnames)
{
    Py_ssize_t named = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    if (small_limit == 0 || function == NULL || count + named > MOST_METHOD_ARGUMENTS) {
        return Miss;
    }
    int fresh = PySet_Contains(fresh_functions, function);
    if (fresh <= 0) {
        PyErr_Clear();
        return Miss;
    }
    for (Py_ssize_t place = 0; place < named; place++) {
        if (PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, place), s_out) == 0) {
            return Miss; /* A write. */
        }
    }
    PyObject *stack[MOST_METHOD_ARGUMENTS], *values[MOST_METHOD_ARGUMENTS];
    PyObject *result = Miss;
    Py_ssize_t held = 0;
    int strict = 0;
    for (Py_ssize_t place = 0; place < count + named; place++) {
        PyObject *arg = args[place];
        if (Py_TYPE(arg) != Lazy) {
            if (place == 0 || !is_plain_argument(arg)) {
                goto done;
            }
            stack[place] = arg;
            continue;
        }
        PyObject *value = value_of(arg, &strict);
        if (value == NULL) {
            result = NULL;
            goto done;
        }
        if (value == Miss) {
            goto done;
        }
        values[held++] = value;
        if (Py_TYPE(value) != Ndarray || !value_admissible(value) || PyArray_SIZE((PyArrayObject *)value) >= small_limit) {
            goto done;
        }
        stack[place] = value;
    }
    PyObject *made = PyObject_VectorcallMethod(name, stack, (size_t)count, kwnames);
    if (made == NULL) {
        result = NULL;
    } else if (!is_arraylike(made)) {
        Py_DECREF(made); /* No array: the ordinary road decides what to do with such a call. */
    } else {
        made_at_once++;
        result = lazy_of_value(made, s_array, at_once_kind, strict ? Py_True : Py_False);
    }
done:
    while (held > 0) {
        Py_DECREF(values[--held]);
    }
    return result;
}

/* Forget the call the rules last declined. */
static void
forget_declined(void)
{
    Py_CLEAR(declined_operation);
    Py_CLEAR(declined_args);
    Py_CLEAR(declined_kwargs);
}

/* Take one call of the method a road stands for. `args[0]` is the lazy array. */
static PyObject *
road_call(PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames)
{
    Road *road = (Road *)callable;
    Py_ssize_t count = PyVectorcall_NARGS(nargsf);
    Py_ssize_t named = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    PyObject *result = Miss, *operands = NULL, *kwargs = NULL, *name = NULL;
    /* The frame that made this call: C has none of its own, so it is the one the method's caller is in. */
    PyFrameObject *frame = PyEval_GetFrame();
    forget_declined();
    if (!enabled || frame == NULL || count < 1) {
        goto python;
    }
    if (road->kind == STANDIN_CALL) {
        /* `LazyCallable.__call__`, for the commonest call: lazy arrays and plain values by position, plain values
         * by keyword. Nothing has to be made lazy, so the arguments go to NumPy as they came. `operation` is
         * `namespace._made_lazy` and `name` the types of a plain value.
         *
         * While calls are made at once (`small_limit`) two more kinds of argument are taken, for the calls below
         * that are made here on values: a plain `ndarray` by position, which is its own value, and `out=` with a
         * lazy array or a plain one. Whatever is not made here goes to the Python method with them, which makes
         * the plain arrays lazy, as it did. */
        if (Py_TYPE(args[0]) != StandInCall) {
            goto python;
        }
        int owned = 1;
        /* Whether an argument is a shape or an axis: a tuple of plain values, which holds no array. */
        int with_tuple = 0;
        /* How many positional arguments are plain arrays, and what `out=` names when it names an array. */
        int plain = 0;
        PyObject *into = NULL;
        PyObject *function = SLOT(args[0], callable_at[F_FUNCTION]);
        for (Py_ssize_t place = 1; place < count + named; place++) {
            if (place < count && Py_TYPE(args[place]) == Lazy) {
                owned = 0;
            } else if (place < count && small_limit != 0 && Py_TYPE(args[place]) == Ndarray) {
                if (plain == 0 && is_observed() != 0) {
                    PyErr_Clear();
                    goto python; /* A recording has a node for each array that comes in. */
                }
                owned = 0;
                plain++;
            } else if (place >= count && small_limit != 0 && into == NULL
                       && (Py_TYPE(args[place]) == Lazy || Py_TYPE(args[place]) == Ndarray)
                       && PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, place - count), s_out) == 0) {
                owned = 0;
                into = args[place];
            } else if (small_limit != 0 && PyObject_TypeCheck(args[place], Generic)) {
                /* A NumPy scalar of the program's own, `np.finfo(t).eps`: a plain value, which holds no array. */
            } else {
                int is_plain = PySet_Contains(road->name, (PyObject *)Py_TYPE(args[place]));
                if (is_plain < 0) {
                    PyErr_Clear();
                    goto python;
                }
                if (is_plain == 0) {
                    if (!PyTuple_CheckExact(args[place])) {
                        goto python;
                    }
                    for (Py_ssize_t part = 0; part < PyTuple_GET_SIZE(args[place]); part++) {
                        if (PySet_Contains(road->name, (PyObject *)Py_TYPE(PyTuple_GET_ITEM(args[place], part))) <= 0) {
                            PyErr_Clear();
                            goto python;
                        }
                    }
                    with_tuple = 1;
                }
            }
        }
        if (into != NULL) {
            /* `np.maximum(x, 0, out=x)`, `np.exp(x, out=x)`: a ufunc that writes into an array, and names nothing
             * else. Made here when nothing is pending. The call's result is its output: the lazy array it was
             * given, or for a plain array a lazy array on it, which is what the Python method hands to NumPy as
             * `out` and gets back. */
            if (named == 1 && !with_tuple && count >= 2) {
                int strict = plain && SLOT(args[0], callable_at[F_STRICT]) == Py_True;
                PyObject *written = ufunc_into_now(function, args + 1, count - 1, into, strict);
                if (written == NULL) {
                    return NULL;
                }
                if (written != Miss) {
                    if (Py_TYPE(into) == Lazy) {
                        return written;
                    }
                    Py_DECREF(written);
                    return lazy_of_handed(into, SLOT(args[0], callable_at[F_STRICT]));
                }
            }
            goto python;
        }
        if (!owned && !named && !with_tuple) {
            /* A ufunc on small arrays that have their values: made here, without NumPy's way round through
             * `__array_ufunc__`. */
            PyObject *quick = ufunc_on(function, args + 1, count - 1, 0, plain && SLOT(args[0], callable_at[F_STRICT]) == Py_True);
            if (quick != Miss) {
                return quick;
            }
        }
        if (!owned && !plain && small_limit != 0 && count >= 2 && Py_TYPE(args[1]) == Lazy && Py_TYPE(function) != Ufunc) {
            int alike = PySet_Contains(shaped_like, function);
            if (alike < 0) {
                PyErr_Clear();
            } else if (alike) {
                /* `np.zeros_like(x)` of an array that has its values: NumPy's own, on the values. */
                int strict = SLOT(args[0], callable_at[F_STRICT]) == Py_True;
                int lone = 1;
                for (Py_ssize_t place = 2; place < count; place++) {
                    lone = lone && Py_TYPE(args[place]) != Lazy;
                }
                PyObject *like = lone && count + named <= 8 ? value_of(args[1], &strict) : Miss;
                if (like == NULL) {
                    return NULL;
                }
                if (like != Miss && Py_TYPE(like) == Ndarray) {
                    PyObject *stack[8];
                    stack[0] = like;
                    for (Py_ssize_t place = 2; place < count + named; place++) {
                        stack[place - 1] = args[place];
                    }
                    PyObject *made = PyObject_Vectorcall(function, stack, (size_t)(count - 1), kwnames);
                    Py_DECREF(like);
                    if (made == NULL) {
                        return NULL;
                    }
                    if (Py_TYPE(made) == Ndarray) {
                        return lazy_of_value(made, SLOT(args[0], callable_at[F_SOURCE]), empty_string,
                                             strict ? Py_True : Py_False);
                    }
                    PyObject *lazy = PyObject_CallFunctionObjArgs(road->operation, made, SLOT(args[0], callable_at[F_SOURCE]),
                                                                  strict ? Py_True : Py_False, Py_True, NULL);
                    Py_DECREF(made);
                    return lazy;
                }
                if (like != Miss) {
                    Py_DECREF(like);
                }
            }
        }
        if (!owned && small_limit != 0 && Py_TYPE(function) != Ufunc && count + named <= 9) {
            /* `np.dot(a, b)`, `np.sum(x)` and the like on small arrays that have their values: NumPy's own call,
             * there and then, with no rule asked. Only for the functions listed as giving one new array that
             * shares no memory with an argument, so the result is an array of its own like any call's. */
            int fresh = PySet_Contains(fresh_functions, function);
            /* Whether the call is made now whatever its size, as `ready_now` has it for a node: the planner has
             * said what a kernel can make, and this function is none of it. A matrix product of a batch. */
            int any_size = fresh > 0 && fusible_calls != NULL && PySet_Contains(fusible_calls, function) == 0;
            if (fresh < 0 || PyErr_Occurred()) {
                PyErr_Clear();
            } else if (fresh) {
                PyObject *stack[8], *values[8];
                Py_ssize_t held = 0;
                int ready = 1, strict = SLOT(args[0], callable_at[F_STRICT]) == Py_True;
                for (Py_ssize_t place = 1; ready && place < count + named; place++) {
                    PyObject *arg = args[place];
                    if (Py_TYPE(arg) != Lazy) {
                        /* A plain value, or a plain array, which is its own: the loop above saw to that. */
                        stack[place - 1] = arg;
                        ready = any_size || Py_TYPE(arg) != Ndarray || PyArray_SIZE((PyArrayObject *)arg) < small_limit;
                        continue;
                    }
                    PyObject *value = value_of(arg, &strict);
                    if (value == NULL) {
                        while (held > 0) {
                            Py_DECREF(values[--held]);
                        }
                        return NULL;
                    }
                    if (value == Miss) {
                        ready = 0;
                        break;
                    }
                    values[held++] = value;
                    Py_ssize_t size = count_of(value);
                    ready = size >= 0 && (any_size || size < small_limit);
                    stack[place - 1] = value;
                }
                PyObject *made = ready ? PyObject_Vectorcall(function, stack, (size_t)(count - 1), kwnames) : Miss;
                while (held > 0) {
                    Py_DECREF(values[--held]);
                }
                if (made == NULL) {
                    return NULL;
                }
                if (made != Miss) {
                    if (is_arraylike(made)) {
                        made_at_once++;
                        return lazy_of_value(made, s_array, at_once_kind, strict ? Py_True : Py_False);
                    }
                    PyObject *lazy = PyObject_CallFunctionObjArgs(road->operation, made, SLOT(args[0], callable_at[F_SOURCE]),
                                                                  strict ? Py_True : Py_False, Py_True, NULL);
                    Py_DECREF(made);
                    return lazy;
                }
            }
        }
        if (!owned && small_limit != 0 && Py_TYPE(function) != Ufunc && count + named <= 9) {
            /* `np.clip(x, lo, hi)`: a function the rules have no rule for, so that a call of it is never a node,
             * and whose one result is a new array (`never_deferred`). Python makes such a call at once on the
             * values, after computing what is pending among its arguments (`array._at_once`), and so does this,
             * for arguments that have their values, whatever their size: 0.6 µs over NumPy's call where the way
             * through NumPy's protocol and Python took 7.5. The call is outside the deferred set, and is counted
             * and logged as one, which is 0.4 µs of the 0.6. */
            int never = PySet_Contains(never_deferred, function);
            if (never < 0) {
                PyErr_Clear();
            } else if (never) {
                PyObject *stack[8], *values[8];
                Py_ssize_t held = 0;
                int ready = 1, strict = SLOT(args[0], callable_at[F_STRICT]) == Py_True;
                for (Py_ssize_t place = 1; ready && place < count + named; place++) {
                    PyObject *arg = args[place];
                    if (Py_TYPE(arg) != Lazy) {
                        stack[place - 1] = arg; /* A plain value, or a plain array, which is its own. */
                        continue;
                    }
                    PyObject *value = value_of(arg, &strict);
                    if (value == NULL) {
                        while (held > 0) {
                            Py_DECREF(values[--held]);
                        }
                        return NULL;
                    }
                    if (value == Miss) {
                        ready = 0;
                        break;
                    }
                    values[held++] = value;
                    stack[place - 1] = value;
                }
                PyObject *source = SLOT(args[0], callable_at[F_SOURCE]);
                PyObject *made = Miss;
                if (ready) {
                    made = count_at_once(source) < 0 ? NULL : PyObject_Vectorcall(function, stack, (size_t)(count - 1), kwnames);
                }
                while (held > 0) {
                    Py_DECREF(values[--held]);
                }
                if (made == NULL) {
                    return NULL;
                }
                if (made != Miss) {
                    if (is_arraylike(made)) {
                        made_at_once++;
                        return lazy_of_value(made, source, at_once_kind, strict ? Py_True : Py_False);
                    }
                    /* No array: `array._made_lazy` says what leaves then. */
                    PyObject *lazy = PyObject_CallFunctionObjArgs(at_once_result, made, source, strict ? Py_True : Py_False, NULL);
                    Py_DECREF(made);
                    return lazy;
                }
            }
        }
        if (plain) {
            goto python; /* Nothing above made the call: the plain arrays are made lazy, and NumPy is asked. */
        }
        if (!owned && Py_TYPE(function) != Ufunc && PyFunction_Check(function) == 0 && Py_TYPE(args[1]) == Lazy) {
            /* A function of NumPy's on a lazy array. NumPy would look through the arguments, find the lazy array
             * and call its `__array_function__`, which asks the rules. They are asked here, with the same
             * function and the same arguments; whatever they do not answer goes NumPy's way as before. */
            int apart = PySet_Contains(handled_apart, function);
            int writes = 0;
            for (Py_ssize_t place = 0; place < named; place++) {
                writes = writes || PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, place), s_out) == 0;
            }
            if (apart < 0) {
                PyErr_Clear();
            } else if (!apart && !writes) {
                PyObject *given = PyTuple_New(count - 1);
                PyObject *keywords = given ? PyDict_New() : NULL;
                PyObject *called = keywords ? PyObject_GetAttr(function, s_dunder_name) : NULL;
                PyObject *quick = Miss;
                if (called != NULL) {
                    for (Py_ssize_t place = 1; place < count; place++) {
                        PyTuple_SET_ITEM(given, place - 1, Py_NewRef(args[place]));
                    }
                    int kept = 0;
                    for (Py_ssize_t place = 0; place < named && kept == 0; place++) {
                        kept = PyDict_SetItem(keywords, PyTuple_GET_ITEM(kwnames, place), args[count + place]);
                    }
                    quick = kept < 0 ? NULL : try_defer(function, called, given, keywords, NULL, frame);
                } else {
                    PyErr_Clear();
                }
                Py_XDECREF(given);
                Py_XDECREF(keywords);
                Py_XDECREF(called);
                if (quick != Miss) {
                    return quick;
                }
                forget_declined();
            }
        }
        PyObject *made = PyObject_Vectorcall(function, args + 1, count - 1, kwnames);
        if (made == NULL || Py_TYPE(made) == Lazy) {
            return made; /* Deferred: the commonest answer. */
        }
        if (owned && Py_TYPE(made) == Ndarray) {
            /* An array made of no array, as `np.zeros(n)` is: a lazy array that has its values, whose memory
             * nothing outside the layer knows. `array.wrap(made, source, strict, owned=True)`. */
            return lazy_of_value(made, SLOT(args[0], callable_at[F_SOURCE]), empty_string, SLOT(args[0], callable_at[F_STRICT]));
        }
        PyObject *lazy = PyObject_CallFunctionObjArgs(road->operation, made, SLOT(args[0], callable_at[F_SOURCE]),
                                                      SLOT(args[0], callable_at[F_STRICT]), owned ? Py_True : Py_False, NULL);
        Py_DECREF(made);
        return lazy;
    }
    if (road->kind == STANDIN_NAME) {
        /* `LazyNumPy.__getattr__`, for a name answered before, while the module still has what it had. */
        if (named || count != 2 || Py_TYPE(args[0]) != StandInModule) {
            goto python;
        }
        PyObject *attribute = PyObject_GetAttr(SLOT(args[0], module_at[M_MODULE]), args[1]);
        if (attribute == NULL) {
            return NULL;
        }
        PyObject *answered = PyDict_GetItemWithError(SLOT(args[0], module_at[M_ANSWERS]), args[1]);
        int same = answered != NULL && PyTuple_CheckExact(answered) && PyTuple_GET_ITEM(answered, 0) == attribute;
        Py_DECREF(attribute);
        if (same) {
            return Py_NewRef(PyTuple_GET_ITEM(answered, 1));
        }
        PyErr_Clear();
        goto python;
    }
    if (Py_TYPE(args[0]) != Lazy) {
        goto python;
    }
    switch (road->kind) {
    case BINARY:
    case UNARY:
        if (named || count != (road->kind == BINARY ? 2 : 1)) {
            goto python;
        }
        if (small_limit != 0) {
            PyObject *pair[2] = {road->reflected ? args[count - 1] : args[0], road->reflected ? args[0] : args[count - 1]};
            result = ufunc_now(road->operation, pair, count, 1);
            if (result != Miss) {
                break;
            }
        }
        operands = road->kind == UNARY ? PyTuple_Pack(1, args[0])
                   : road->reflected ? PyTuple_Pack(2, args[1], args[0]) : PyTuple_Pack(2, args[0], args[1]);
        /* A call written with no keyword has the one empty dictionary every such node has: nothing adds to a
         * node's keywords, and one each would be an object made, freed and counted by the collector per node. */
        kwargs = Py_NewRef(no_keywords);
        if (operands == NULL || kwargs == NULL) {
            goto fail;
        }
        /* `ruled_as` is the ufunc whose rule the call has, for an operator that is not its ufunc alone: `x ** e`. */
        result = try_defer(road->operation, road->name, operands, kwargs,
                           road->ruled_as == Py_None ? NULL : road->ruled_as, frame);
        break;
    case ARRAY_UFUNC:
        /* (self, ufunc, method, *inputs). `out=` alone is a write; any other keyword sends it to Python. */
        if (count < 3 || !PyUnicode_CheckExact(args[2]) || PyUnicode_Compare(args[2], s_call) != 0) {
            goto python;
        }
        if (named == 1 && PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, 0), s_out) == 0 && PyTuple_CheckExact(args[count])
            && PyTuple_GET_SIZE(args[count]) == 1) {
            /* Made now when nothing is pending, whatever the output is; else a node, for a lazy output the layer
             * may defer a write into. `a += x` on a plain `a` comes here as `np.add(a, x, out=(a,))`, and what it
             * gives is `a`. */
            result = ufunc_into_now(args[1], args + 3, count - 3, PyTuple_GET_ITEM(args[count], 0), 0);
            if (result == Miss) {
                result = write_out(args[1], args + 3, count - 3, PyTuple_GET_ITEM(args[count], 0), frame);
            }
            break;
        }
        if (named) {
            goto python;
        }
        if (count == 5 && small_limit != 0) {
            /* `alpha * a[i, k]` with `alpha` a NumPy scalar of the program's own: the scalar's operator does not
             * know a lazy array and hands the call to its ufunc, which comes here. Both operands are numbers, so
             * the scalar's own operator on the two values is what the eager program calls at that line. */
            result = number_in_place(args[1], args[3], args[4]);
            if (result != Miss) {
                break;
            }
        }
        result = ufunc_now(args[1], args + 3, count - 3, 0);
        if (result != Miss) {
            break;
        }
        operands = PyTuple_New(count - 3);
        kwargs = Py_NewRef(no_keywords);
        name = PyObject_GetAttr(args[1], s_dunder_name);
        if (operands == NULL || kwargs == NULL || name == NULL) {
            goto fail;
        }
        for (Py_ssize_t place = 3; place < count; place++) {
            PyTuple_SET_ITEM(operands, place - 3, Py_NewRef(args[place]));
        }
        result = try_defer(args[1], name, operands, kwargs, NULL, frame);
        break;
    case ARRAY_FUNCTION: {
        /* (self, func, types, args, kwargs), as NumPy calls it. */
        if (named || count != 5 || !PyTuple_CheckExact(args[3]) || !PyDict_CheckExact(args[4])) {
            goto python;
        }
        int apart = PySet_Contains(road->operation, args[1]);
        if (apart < 0) {
            PyErr_Clear();
            goto python;
        }
        PyObject *out = PyDict_GetItemWithError(args[4], s_out);
        if (!apart && out != NULL && out != Py_None && PyDict_GET_SIZE(args[4]) == 1 && road->ruled_as != NULL
            && PySet_Contains(road->ruled_as, args[1]) == 1) {
            /* `np.clip(x, lo, hi, out=x)` and the roundings: a write. */
            result = write_out(args[1], &PyTuple_GET_ITEM(args[3], 0), PyTuple_GET_SIZE(args[3]), out, frame);
            break;
        }
        if (apart || (out != NULL && out != Py_None) || PyErr_Occurred()) {
            PyErr_Clear();
            goto python;
        }
        name = PyObject_GetAttr(args[1], s_dunder_name);
        if (name == NULL) {
            goto fail;
        }
        result = try_defer(args[1], name, args[3], args[4], NULL, frame);
        break;
    }
    case REDUCTION:
        /* (self, *args, **kwargs): the node calls the method on the value, with the arguments as written. */
        result = method_now(road->name, road->ruled_as, args, count, kwnames);
        if (result != Miss) {
            break;
        }
        operands = PyTuple_New(count);
        kwargs = named ? PyDict_New() : Py_NewRef(no_keywords);
        if (operands == NULL || kwargs == NULL) {
            goto fail;
        }
        for (Py_ssize_t place = 0; place < count; place++) {
            PyTuple_SET_ITEM(operands, place, Py_NewRef(args[place]));
        }
        for (Py_ssize_t place = 0; place < named; place++) {
            if (PyDict_SetItem(kwargs, PyTuple_GET_ITEM(kwnames, place), args[count + place]) < 0) {
                goto fail;
            }
        }
        result = try_defer(road->operation, road->name, operands, kwargs, road->ruled_as, frame);
        break;
    case GETITEM:
        if (named || count != 2) {
            goto python;
        }
        result = item_now(args[0], args[1], frame);
        if (result == Miss) {
            result = plain_item(args[0], args[1], frame);
        }
        if (result == Miss) {
            result = select_item(args[0], args[1], frame);
        }
        break;
    case RESHAPE:
    case TRANSPOSE:
    case SWAPAXES:
    case RAVEL: {
        /* The methods that are a call of one NumPy function on the array, with the arguments put as the Python
         * method puts them: `x.reshape(4, 3)` is `np.reshape(x, (4, 3))`, `x.transpose(1, 0)` is
         * `np.transpose(x, (1, 0))` and `x.transpose(axes)` is `np.transpose(x, axes)`. */
        Py_ssize_t given = count - 1;
        if (named || (road->kind == SWAPAXES && given != 2) || (road->kind == RAVEL && given != 0)) {
            goto python;
        }
        if (road->kind == TRANSPOSE && given == 0) {
            result = view_now(args[0], road->operation, road->name, frame);
            if (result != Miss) {
                break;
            }
        }
        if (road->kind == SWAPAXES) {
            operands = PyTuple_Pack(3, args[0], args[1], args[2]);
        } else if (given == 0 && road->kind != RESHAPE) {
            operands = PyTuple_Pack(1, args[0]);
        } else if (given == 1 && (road->kind == RESHAPE || !PyLong_Check(args[1]))) {
            operands = PyTuple_Pack(2, args[0], args[1]);
        } else {
            PyObject *rest = PyTuple_New(given);
            for (Py_ssize_t place = 0; rest != NULL && place < given; place++) {
                PyTuple_SET_ITEM(rest, place, Py_NewRef(args[1 + place]));
            }
            operands = rest ? PyTuple_Pack(2, args[0], rest) : NULL;
            Py_XDECREF(rest);
        }
        kwargs = Py_NewRef(no_keywords);
        if (operands == NULL || kwargs == NULL) {
            goto fail;
        }
        result = try_defer(road->operation, road->name, operands, kwargs, NULL, frame);
        break;
    }
    case INPLACE:
        if (named || count != 2) {
            goto python;
        }
        result = in_place(args[0], args[1], road->operation, road->name, road->ruled_as, frame);
        break;
    case LEAVE: {
        /* `float(x)`, `if x:`, `np.asarray(x)` and the other ways a value leaves without `gather()`, for an
         * array that has its value while nothing is pending: what `array._hand_out` lets through without a
         * word, where the method `array._leaving` built then only converts. `operation` is what converts, and
         * `ruled_as` is True when the array itself is handed out, whose memory is known outside from then on.
         * A strict array, a pending one and a failed one are Python's to answer. */
        if (small_limit == 0 || PySet_GET_SIZE(pending) != 0 || count + named > MOST_LEAVING) {
            goto python;
        }
        int strict = 0;
        PyObject *value = value_of(args[0], &strict);
        if (value == NULL) {
            return NULL;
        }
        if (value == Miss) {
            goto python;
        }
        if (strict) {
            Py_DECREF(value);
            goto python;
        }
        if (road->ruled_as == Py_True) {
            PyObject *cell = cell_of(args[0]); /* `LazyArray._expose`. */
            if (cell == NULL) {
                Py_DECREF(value);
                return NULL;
            }
            Py_SETREF(SLOT(SLOT(cell, cell_at[C_NODE]), node_at[N_EXPOSED]), Py_NewRef(Py_True));
        }
        PyObject *stack[MOST_LEAVING];
        stack[0] = value;
        for (Py_ssize_t place = 1; place < count + named; place++) {
            stack[place] = args[place];
        }
        result = PyObject_Vectorcall(road->operation, stack, (size_t)count, kwnames);
        Py_DECREF(value);
        return result;
    }
    case SETITEM:
        if (named || count != 3) {
            goto python;
        }
        if (is_own_view_at(args[0], args[1], args[2])) {
            result = Py_NewRef(Py_None);
            break;
        }
        result = set_now(args[0], args[1], args[2]);
        if (result == Miss) {
            result = set_item(args[0], args[1], args[2], frame);
        }
        if (result == Miss) {
            result = set_through(args[0], args[1], args[2], frame);
        }
        break;
    default:
        goto python;
    }
    Py_XDECREF(operands);
    Py_XDECREF(kwargs);
    Py_XDECREF(name);
    if (result != Miss) {
        return result;
    }
python:
    return PyObject_Vectorcall(road->fallback, args, nargsf, kwnames);
fail:
    Py_XDECREF(operands);
    Py_XDECREF(kwargs);
    Py_XDECREF(name);
    return NULL;
}

/* `Road(kind, fallback, operation=None, name=None, ruled_as=None, reflected=False)`. */
static int
road_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *names[] = {"kind", "fallback", "operation", "name", "ruled_as", "reflected", NULL};
    Road *road = (Road *)self;
    const char *kind;
    PyObject *fallback, *operation = Py_None, *name = Py_None, *ruled_as = NULL;
    int reflected = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sO|OOOp", names, &kind, &fallback, &operation, &name, &ruled_as, &reflected)) {
        return -1;
    }
    road->kind = -1;
    for (int each = 0; each < ROAD_KINDS; each++) {
        if (strcmp(kind, ROAD_NAMES[each]) == 0) {
            road->kind = each;
        }
    }
    if (road->kind < 0) {
        PyErr_Format(PyExc_ValueError, "no road of kind '%s'", kind);
        return -1;
    }
    if (ruled_as == Py_None) {
        ruled_as = NULL;
    }
    Py_XSETREF(road->fallback, Py_NewRef(fallback));
    Py_XSETREF(road->operation, Py_NewRef(operation));
    Py_XSETREF(road->name, Py_NewRef(name));
    Py_XSETREF(road->ruled_as, Py_XNewRef(ruled_as));
    road->reflected = reflected;
    road->vectorcall = road_call;
    return 0;
}

static void
road_dealloc(PyObject *self)
{
    Road *road = (Road *)self;
    Py_XDECREF(road->fallback);
    Py_XDECREF(road->operation);
    Py_XDECREF(road->name);
    Py_XDECREF(road->ruled_as);
    Py_TYPE(self)->tp_free(self);
}

/* `x.method`: the road bound to the array, as a function on a class is. On the class itself, the road. */
static PyObject *
road_get(PyObject *self, PyObject *instance, PyObject *owner)
{
    if (instance == NULL || instance == Py_None) {
        return Py_NewRef(self);
    }
    Road *road = (Road *)self;
    if (road->kind == ATTRIBUTE) {
        /* A property that is a call of one NumPy function on the array: `x.T` is `np.transpose(x)`. */
        PyObject *result = Miss;
        PyFrameObject *frame = PyEval_GetFrame();
        if (enabled && frame != NULL && Py_TYPE(instance) == Lazy) {
            result = view_now(instance, road->operation, road->name, frame);
            if (result != Miss) {
                return result;
            }
            PyObject *operands = PyTuple_Pack(1, instance);
            PyObject *kwargs = PyDict_New();
            result = operands && kwargs ? try_defer(road->operation, road->name, operands, kwargs, NULL, frame) : NULL;
            Py_XDECREF(operands);
            Py_XDECREF(kwargs);
        }
        return result == Miss ? PyObject_CallOneArg(road->fallback, instance) : result;
    }
    return PyMethod_New(self, instance);
}

static PyObject *
road_fallback(PyObject *self, void *closure)
{
    return Py_NewRef(((Road *)self)->fallback);
}

static PyGetSetDef road_getset[] = {
    {"fallback", road_fallback, NULL, "The Python method this road stands for.", NULL},
    {NULL},
};

static PyTypeObject RoadType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "prototypes.lazy._core.Road",
    .tp_doc = "Stands on LazyArray where a method stood: takes the calls whose answers are remembered, and hands the others to the method.",
    .tp_basicsize = sizeof(Road),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_METHOD_DESCRIPTOR,
    .tp_new = PyType_GenericNew,
    .tp_init = road_init,
    .tp_dealloc = road_dealloc,
    .tp_call = PyVectorcall_Call,
    .tp_vectorcall_offset = offsetof(Road, vectorcall),
    .tp_descr_get = road_get,
    .tp_getset = road_getset,
};

/* ---- The operators of `LazyArray` on small arrays, taken where the interpreter first looks. ----
 *
 * `a + b` reaches a road in three calls: the interpreter calls the class's slot for the operator, the slot looks
 * `__add__` up on the class and calls what it finds, which is the road. For a call made at once that is a third of
 * what the layer adds to NumPy's own time. While small calls are made at once, each slot below stands in the
 * class's table itself: it makes the call if it is one to make now, exactly as the road would have, and hands
 * every other call to the slot that stood there, which goes to the road. Putting a method back on the class
 * (`array.use_core(None)`) puts the interpreter's own slot back, as it does for any class. */

/* The binary operators that are one ufunc each, by the method that carries the road. */
enum { Q_ADD, Q_SUBTRACT, Q_MULTIPLY, Q_TRUE_DIVIDE, Q_FLOOR_DIVIDE, Q_REMAINDER, Q_MATRIX_MULTIPLY, Q_AND, Q_OR, Q_XOR,
       Q_BINARY };
static const char *const QUICK_BINARY[Q_BINARY] = {"__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__",
                                                   "__matmul__", "__and__", "__or__", "__xor__"};
static const size_t QUICK_BINARY_AT[Q_BINARY] = {
    offsetof(PyNumberMethods, nb_add), offsetof(PyNumberMethods, nb_subtract), offsetof(PyNumberMethods, nb_multiply),
    offsetof(PyNumberMethods, nb_true_divide), offsetof(PyNumberMethods, nb_floor_divide), offsetof(PyNumberMethods, nb_remainder),
    offsetof(PyNumberMethods, nb_matrix_multiply), offsetof(PyNumberMethods, nb_and), offsetof(PyNumberMethods, nb_or),
    offsetof(PyNumberMethods, nb_xor)};
static binaryfunc binary_before[Q_BINARY];
static PyObject *binary_calls[Q_BINARY];

/* The unary operators. */
enum { Q_NEGATIVE, Q_POSITIVE, Q_ABSOLUTE, Q_INVERT, Q_UNARY };
static const char *const QUICK_UNARY[Q_UNARY] = {"__neg__", "__pos__", "__abs__", "__invert__"};
static const size_t QUICK_UNARY_AT[Q_UNARY] = {offsetof(PyNumberMethods, nb_negative), offsetof(PyNumberMethods, nb_positive),
                                               offsetof(PyNumberMethods, nb_absolute), offsetof(PyNumberMethods, nb_invert)};
static unaryfunc unary_before[Q_UNARY];
static PyObject *unary_calls[Q_UNARY];

/* The in-place operators. */
enum { Q_IADD, Q_ISUBTRACT, Q_IMULTIPLY, Q_ITRUE_DIVIDE, Q_INPLACE };
static const char *const QUICK_INPLACE[Q_INPLACE] = {"__iadd__", "__isub__", "__imul__", "__itruediv__"};
static const size_t QUICK_INPLACE_AT[Q_INPLACE] = {
    offsetof(PyNumberMethods, nb_inplace_add), offsetof(PyNumberMethods, nb_inplace_subtract),
    offsetof(PyNumberMethods, nb_inplace_multiply), offsetof(PyNumberMethods, nb_inplace_true_divide)};
static binaryfunc inplace_before[Q_INPLACE];
static PyObject *inplace_calls[Q_INPLACE];

/* The comparisons, in the order of Python's own numbers for them: `Py_LT` to `Py_GE`. */
#define Q_COMPARISONS 6
static const char *const QUICK_COMPARISONS[Q_COMPARISONS] = {"__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__"};
static PyObject *comparison_calls[Q_COMPARISONS];
static richcmpfunc compare_before;
static ternaryfunc power_before;
static PyObject *power_call;
static binaryfunc subscript_before;
static objobjargproc assign_before;

/* Where a type keeps one slot of its numbers, or NULL for a type that has none. */
static void **
number_slot(PyTypeObject *type, size_t at)
{
    return type->tp_as_number != NULL ? (void **)((char *)type->tp_as_number + at) : NULL;
}

/* Call the slot that stood in the class's table for a binary operator, with that slot back in the table for the
 * length of the call. The interpreter's slot for an operator of two operands decides whose method to call, the
 * left operand's or the right one's reflected, by looking in each operand's class for itself: with another
 * function there it would find that neither has the method. The same holds for a subclass, which was given the
 * class's slots when it was made. `quick` is this file's function for the slot and `before` the one it replaced. */
static PyObject *
with_own_slot(size_t at, void *quick, void *before, PyObject *left, PyObject *right, PyObject *third)
{
    void **first = number_slot(Py_TYPE(left), at), **second = number_slot(Py_TYPE(right), at);
    int one = first != NULL && *first == quick, other = second != NULL && second != first && *second == quick;
    if (one) {
        *first = before;
    }
    if (other) {
        *second = before;
    }
    PyObject *result = third == NULL ? ((binaryfunc)before)(left, right) : ((ternaryfunc)before)(left, right, third);
    if (one && *first == before) {
        *first = quick;
    }
    if (other && *second == before) {
        *second = quick;
    }
    return result;
}

static const binaryfunc QUICK_BINARY_SLOTS[];

/* ---- An operator on numbers. `x.max() * i / n` in a loop is arithmetic on NumPy scalars: the eager program has
 * `np.float64` objects there, whose own operators make the arithmetic in 40 to 80 ns. A lazy array that holds
 * such a scalar gave it to the operator's ufunc, which is a call of 600 ns for one element. So where every
 * operand is a number, the scalar's own operator is called on the values: what the eager program calls, with its
 * result, its type and its warning's words. ---- */

/* The interpreter's function for each binary operator, by the same numbers as `QUICK_BINARY`. */
static const binaryfunc NUMBER_BINARY[Q_BINARY] = {PyNumber_Add, PyNumber_Subtract, PyNumber_Multiply, PyNumber_TrueDivide,
                                                   PyNumber_FloorDivide, PyNumber_Remainder, PyNumber_MatrixMultiply,
                                                   PyNumber_And, PyNumber_Or, PyNumber_Xor};
/* And for each unary operator, by the numbers of `QUICK_UNARY`. */
static const unaryfunc NUMBER_UNARY[Q_UNARY] = {PyNumber_Negative, PyNumber_Positive, PyNumber_Absolute, PyNumber_Invert};

/* The values of an operator's operands when every one is a number: a lazy array whose value is a NumPy scalar,
 * or a number written in the program, with one at least of the first kind. Returns 1 with `values`
 * filled, borrowed from `held`, whose references the caller drops; 0 when the operands are anything else, with
 * nothing to drop; -1 with an error. `*strict` is set when one of the arrays is strict. */
static int
numbers_of(PyObject *const *operands, Py_ssize_t count, PyObject **values, PyObject **held, int *strict)
{
    int lazies = 0;
    for (Py_ssize_t place = 0; place < count; place++) {
        held[place] = NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *operand = operands[place];
        if (Py_TYPE(operand) == Lazy) {
            /* Looked at before it is asked for: a scalar the array holds alone, or the value of its cell's node. */
            PyObject *peek = SLOT(operand, lazy_at[L_HELD]);
            if (peek != NULL && Py_TYPE(peek) == Cell) {
                PyObject *node = SLOT(peek, cell_at[C_NODE]);
                peek = node != NULL ? SLOT(node, node_at[N_VALUE]) : NULL;
            }
            if (peek == NULL || !PyObject_TypeCheck(peek, Generic)) {
                goto other;
            }
            PyObject *value = value_of(operand, strict);
            if (value == NULL) {
                for (Py_ssize_t each = 0; each < place; each++) {
                    Py_CLEAR(held[each]);
                }
                return -1;
            }
            if (value == Miss) {
                goto other;
            }
            if (!PyObject_TypeCheck(value, Generic)) {
                Py_DECREF(value);
                goto other;
            }
            held[place] = value;
            values[place] = value;
            lazies++;
        } else if (PyFloat_CheckExact(operand) || PyLong_CheckExact(operand) || PyBool_Check(operand)
                   || PyComplex_CheckExact(operand) || PyObject_TypeCheck(operand, Generic)) {
            values[place] = operand; /* A NumPy scalar of the program's own is a number like any written in it. */
        } else {
            goto other;
        }
    }
    if (lazies) {
        return 1;
    }
other:
    for (Py_ssize_t place = 0; place < count; place++) {
        Py_CLEAR(held[place]);
    }
    return 0;
}

/* What an operator on numbers gives, as the layer hands it on: a lazy array that holds the scalar NumPy answered
 * with, or whatever else it answered. Steals `value`, which may be NULL with the error set. */
static PyObject *
number_made(PyObject *value, int strict)
{
    if (value == NULL || !is_arraylike(value)) {
        return value;
    }
    made_at_once++;
    return lazy_of_value(value, s_array, at_once_kind, strict ? Py_True : Py_False);
}

/* An in-place operator on a number, for `in_place_now`, and an operator that reached the layer as a call of its
 * ufunc on two numbers, for the road of `__array_ufunc__`: the scalar's own operator on the two values, when the
 * operator is one of `QUICK_BINARY` and both operands are numbers. `ufunc` is the ufunc the operator stands for.
 * Returns the new number as a lazy array, `Miss` when the operands are anything else, or NULL. */
static PyObject *
number_in_place(PyObject *ufunc, PyObject *target, PyObject *other)
{
    for (int which = 0; which < Q_BINARY; which++) {
        if (binary_calls[which] != ufunc) {
            continue;
        }
        PyObject *pair[2] = {target, other}, *values[2], *held[2];
        int strict = 0;
        int numbers = numbers_of(pair, 2, values, held, &strict);
        if (numbers < 0) {
            return NULL;
        }
        if (!numbers) {
            return Miss;
        }
        PyObject *value = NUMBER_BINARY[which](values[0], values[1]);
        Py_XDECREF(held[0]);
        Py_XDECREF(held[1]);
        return number_made(value, strict);
    }
    return Miss;
}

static PyObject *
quick_binary(int which, PyObject *left, PyObject *right)
{
    if (enabled && small_limit != 0 && binary_calls[which] != NULL) {
        PyObject *pair[2] = {left, right}, *values[2], *held[2];
        int strict = 0;
        forget_declined();
        int numbers = numbers_of(pair, 2, values, held, &strict);
        if (numbers < 0) {
            return NULL;
        }
        if (numbers) {
            PyObject *value = NUMBER_BINARY[which](values[0], values[1]);
            Py_XDECREF(held[0]);
            Py_XDECREF(held[1]);
            return number_made(value, strict);
        }
        PyObject *result = ufunc_now(binary_calls[which], pair, 2, 1);
        if (result != Miss) {
            return result;
        }
    }
    return with_own_slot(QUICK_BINARY_AT[which], (void *)QUICK_BINARY_SLOTS[which], (void *)binary_before[which], left, right, NULL);
}

static PyObject *
quick_unary(int which, PyObject *operand)
{
    if (enabled && small_limit != 0 && unary_calls[which] != NULL) {
        PyObject *value, *held;
        int strict = 0;
        forget_declined();
        int numbers = numbers_of(&operand, 1, &value, &held, &strict);
        if (numbers < 0) {
            return NULL;
        }
        if (numbers) {
            PyObject *made = NUMBER_UNARY[which](value);
            Py_XDECREF(held);
            return number_made(made, strict);
        }
        PyObject *result = ufunc_now(unary_calls[which], &operand, 1, 1);
        if (result != Miss) {
            return result;
        }
    }
    return unary_before[which](operand);
}

static PyObject *
quick_inplace(int which, PyObject *target, PyObject *other)
{
    if (enabled && small_limit != 0 && Py_TYPE(target) == Lazy && inplace_calls[which] != NULL) {
        forget_declined();
        PyObject *result = in_place_now(target, other, inplace_calls[which]);
        if (result != Miss) {
            return result;
        }
    }
    return inplace_before[which](target, other);
}

/* One function per slot: the interpreter's table holds a function and nothing that says which operator it is for. */
#define QUICK_BINARY_SLOT(name, which) \
    static PyObject *name(PyObject *left, PyObject *right) { return quick_binary(which, left, right); }
#define QUICK_UNARY_SLOT(name, which) \
    static PyObject *name(PyObject *operand) { return quick_unary(which, operand); }
#define QUICK_INPLACE_SLOT(name, which) \
    static PyObject *name(PyObject *target, PyObject *other) { return quick_inplace(which, target, other); }
QUICK_BINARY_SLOT(quick_add, Q_ADD)
QUICK_BINARY_SLOT(quick_subtract, Q_SUBTRACT)
QUICK_BINARY_SLOT(quick_multiply, Q_MULTIPLY)
QUICK_BINARY_SLOT(quick_true_divide, Q_TRUE_DIVIDE)
QUICK_BINARY_SLOT(quick_floor_divide, Q_FLOOR_DIVIDE)
QUICK_BINARY_SLOT(quick_remainder, Q_REMAINDER)
QUICK_BINARY_SLOT(quick_matrix_multiply, Q_MATRIX_MULTIPLY)
QUICK_BINARY_SLOT(quick_and, Q_AND)
QUICK_BINARY_SLOT(quick_or, Q_OR)
QUICK_BINARY_SLOT(quick_xor, Q_XOR)
QUICK_UNARY_SLOT(quick_negative, Q_NEGATIVE)
QUICK_UNARY_SLOT(quick_positive, Q_POSITIVE)
QUICK_UNARY_SLOT(quick_absolute, Q_ABSOLUTE)
QUICK_UNARY_SLOT(quick_invert, Q_INVERT)
QUICK_INPLACE_SLOT(quick_iadd, Q_IADD)
QUICK_INPLACE_SLOT(quick_isubtract, Q_ISUBTRACT)
QUICK_INPLACE_SLOT(quick_imultiply, Q_IMULTIPLY)
QUICK_INPLACE_SLOT(quick_itrue_divide, Q_ITRUE_DIVIDE)
static const binaryfunc QUICK_BINARY_SLOTS[] = {quick_add, quick_subtract, quick_multiply, quick_true_divide,
                                                        quick_floor_divide, quick_remainder, quick_matrix_multiply,
                                                        quick_and, quick_or, quick_xor};
static const unaryfunc QUICK_UNARY_SLOTS[Q_UNARY] = {quick_negative, quick_positive, quick_absolute, quick_invert};
static const binaryfunc QUICK_INPLACE_SLOTS[Q_INPLACE] = {quick_iadd, quick_isubtract, quick_imultiply, quick_itrue_divide};

static PyObject *quick_power(PyObject *base, PyObject *exponent, PyObject *modulus);

/* `x ** e`. With a third operand it is Python's three-argument power, which is nothing of NumPy's. */
static PyObject *
quick_power(PyObject *base, PyObject *exponent, PyObject *modulus)
{
    if (enabled && small_limit != 0 && modulus == Py_None && power_call != NULL) {
        PyObject *pair[2] = {base, exponent}, *values[2], *held[2];
        int strict = 0;
        forget_declined();
        int numbers = numbers_of(pair, 2, values, held, &strict);
        if (numbers < 0) {
            return NULL;
        }
        if (numbers) {
            PyObject *value = PyNumber_Power(values[0], values[1], Py_None);
            Py_XDECREF(held[0]);
            Py_XDECREF(held[1]);
            return number_made(value, strict);
        }
        PyObject *result = ufunc_now(power_call, pair, 2, 1);
        if (result != Miss) {
            return result;
        }
    }
    return with_own_slot(offsetof(PyNumberMethods, nb_power), (void *)quick_power, (void *)power_before, base, exponent, modulus);
}

/* `x < y` and the other five. */
static PyObject *
quick_compare(PyObject *left, PyObject *right, int operation)
{
    if (enabled && small_limit != 0 && operation >= 0 && operation < Q_COMPARISONS && comparison_calls[operation] != NULL) {
        PyObject *pair[2] = {left, right}, *values[2], *held[2];
        int strict = 0;
        forget_declined();
        int numbers = numbers_of(pair, 2, values, held, &strict);
        if (numbers < 0) {
            return NULL;
        }
        if (numbers) {
            PyObject *value = PyObject_RichCompare(values[0], values[1], operation);
            Py_XDECREF(held[0]);
            Py_XDECREF(held[1]);
            return number_made(value, strict);
        }
        PyObject *result = ufunc_now(comparison_calls[operation], pair, 2, 1);
        if (result != Miss) {
            return result;
        }
    }
    return compare_before(left, right, operation);
}

/* `x[index]`. */
static PyObject *
quick_subscript(PyObject *target, PyObject *index)
{
    if (enabled && small_limit != 0 && Py_TYPE(target) == Lazy) {
        forget_declined();
        PyObject *result = item_now(target, index, NULL);
        if (result != Miss) {
            return result;
        }
    }
    return subscript_before(target, index);
}

/* `x[index] = value`; with no value it is `del x[index]`, which is the class's to refuse. */
static int
quick_assign(PyObject *target, PyObject *index, PyObject *value)
{
    if (value != NULL && enabled && small_limit != 0 && Py_TYPE(target) == Lazy) {
        forget_declined();
        if (is_own_view_at(target, index, value)) {
            return 0;
        }
        PyObject *result = set_now(target, index, value);
        if (result == NULL) {
            return -1;
        }
        if (result != Miss) {
            Py_DECREF(result);
            return 0;
        }
    }
    return assign_before(target, index, value);
}

/* What a road on the class calls for an operator: the `operation` of the `Road` the class has under a name. NULL,
 * with no error, when the class has something else there. Borrowed from the class, which keeps the road. */
static PyObject *
road_operation(PyObject *names, const char *name, int kind)
{
    PyObject *road = PyDict_GetItemString(names, name);
    if (road == NULL || Py_TYPE(road) != &RoadType || ((Road *)road)->kind != kind) {
        return NULL;
    }
    return ((Road *)road)->operation;
}

/* `quick_slots()`: stand in the slots of `LazyArray` for the operators that carry a road. Called by
 * `array.use_core` once the roads are on the class, since putting a method on a class makes the interpreter
 * write the class's slots again. */
static PyObject *
module_quick_slots(PyObject *module, PyObject *ignored)
{
    if (Lazy == NULL || Lazy->tp_as_number == NULL || Lazy->tp_as_mapping == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "quick_slots() comes after setup()");
        return NULL;
    }
#if PY_VERSION_HEX >= 0x030C0000
    PyObject *names = PyType_GetDict(Lazy);
#else
    PyObject *names = Py_XNewRef(Lazy->tp_dict); /* Before Python 3.12 a class's names are read from the class. */
#endif
    if (names == NULL) {
        return NULL;
    }
    char *numbers = (char *)Lazy->tp_as_number;
    for (int each = 0; each < Q_BINARY; each++) {
        PyObject *call = road_operation(names, QUICK_BINARY[each], BINARY);
        binaryfunc *slot = (binaryfunc *)(numbers + QUICK_BINARY_AT[each]);
        if (call != NULL && *slot != NULL && *slot != QUICK_BINARY_SLOTS[each]) {
            Py_XSETREF(binary_calls[each], Py_NewRef(call));
            binary_before[each] = *slot;
            *slot = QUICK_BINARY_SLOTS[each];
        }
    }
    for (int each = 0; each < Q_UNARY; each++) {
        PyObject *call = road_operation(names, QUICK_UNARY[each], UNARY);
        unaryfunc *slot = (unaryfunc *)(numbers + QUICK_UNARY_AT[each]);
        if (call != NULL && *slot != NULL && *slot != QUICK_UNARY_SLOTS[each]) {
            Py_XSETREF(unary_calls[each], Py_NewRef(call));
            unary_before[each] = *slot;
            *slot = QUICK_UNARY_SLOTS[each];
        }
    }
    for (int each = 0; each < Q_INPLACE; each++) {
        PyObject *call = road_operation(names, QUICK_INPLACE[each], INPLACE);
        binaryfunc *slot = (binaryfunc *)(numbers + QUICK_INPLACE_AT[each]);
        if (call != NULL && *slot != NULL && *slot != QUICK_INPLACE_SLOTS[each]) {
            Py_XSETREF(inplace_calls[each], Py_NewRef(call));
            inplace_before[each] = *slot;
            *slot = QUICK_INPLACE_SLOTS[each];
        }
    }
    PyObject *call = road_operation(names, "__pow__", BINARY);
    if (call != NULL && Lazy->tp_as_number->nb_power != NULL && Lazy->tp_as_number->nb_power != quick_power) {
        Py_XSETREF(power_call, Py_NewRef(call));
        power_before = Lazy->tp_as_number->nb_power;
        Lazy->tp_as_number->nb_power = quick_power;
    }
    int compared = 1;
    for (int each = 0; each < Q_COMPARISONS; each++) {
        call = road_operation(names, QUICK_COMPARISONS[each], BINARY);
        compared = compared && call != NULL;
        Py_XSETREF(comparison_calls[each], Py_XNewRef(call));
    }
    if (compared && Lazy->tp_richcompare != NULL && Lazy->tp_richcompare != quick_compare) {
        compare_before = Lazy->tp_richcompare;
        Lazy->tp_richcompare = quick_compare;
    }
    if (road_operation(names, "__getitem__", GETITEM) != NULL && Lazy->tp_as_mapping->mp_subscript != NULL
        && Lazy->tp_as_mapping->mp_subscript != quick_subscript) {
        subscript_before = Lazy->tp_as_mapping->mp_subscript;
        Lazy->tp_as_mapping->mp_subscript = quick_subscript;
    }
    if (road_operation(names, "__setitem__", SETITEM) != NULL && Lazy->tp_as_mapping->mp_ass_subscript != NULL
        && Lazy->tp_as_mapping->mp_ass_subscript != quick_assign) {
        assign_before = Lazy->tp_as_mapping->mp_ass_subscript;
        Lazy->tp_as_mapping->mp_ass_subscript = quick_assign;
    }
    Py_DECREF(names);
    PyType_Modified(Lazy);
    Py_RETURN_NONE;
}

/* ---- Reading the expressions off the nodes of a computation: `fuse.find`, in C. ----
 *
 * The functions of `fuse._Finder`, one for one and in its order, on the same objects: what is returned is the
 * `Expression`s, `Term`s and `Leaf`s the Python makes, for the Python that plans them. The Python is the
 * definition, and the tests of fusion run both. What differs is how the finder keeps its own notes: where the
 * Python has a set and two dictionaries by node number, this has one table by node, since a computation of a
 * few thousand nodes asks them a few times per node. */

/* What `find_setup()` was given: the classes of fuse.py, their slots, its tables and its names. */
static PyTypeObject *FExpression, *FTerm, *FLeaf, *FViewOf, *FWholeCall;
enum { E_ROOT, E_TERM, E_OPERANDS, E_RESULT, E_SHAPE, E_TARGET, E_SIGNATURE, E_NODES, E_ERRSTATES, E_FIELDS };
static const char *const EXPRESSION_FIELDS[E_FIELDS] = {"root", "term", "operands", "result", "shape", "target",
                                                        "signature", "nodes", "errstates"};
enum { T_OP, T_OPERANDS, T_NODE, T_FIELDS };
static const char *const TERM_FIELDS[T_FIELDS] = {"op", "operands", "node"};
enum { LEAF_KIND, LEAF_PLACE, LEAF_FIELDS };
static const char *const LEAF_FIELD_NAMES[LEAF_FIELDS] = {"kind", "place"};
enum { V_OLD, V_STEPS, V_FIELDS };
static const char *const VIEWOF_FIELDS[V_FIELDS] = {"old", "steps"};
enum { WC_UFUNC, WC_READS_OLD, WC_FIELDS };
static const char *const WHOLECALL_FIELDS[WC_FIELDS] = {"ufunc", "reads_old"};
static Py_ssize_t expression_at[E_FIELDS], term_at[T_FIELDS], leaf_at[LEAF_FIELDS], viewof_at[V_FIELDS], wholecall_at[WC_FIELDS];
static PyObject *f_operations;       /* `fuse.OPERATIONS` */
static PyObject *f_in_place;         /* `fuse.IN_PLACE` */
static PyObject *f_arithmetic;       /* `fuse.ARITHMETIC` */
static PyObject *f_unary;            /* `fuse.UNARY` */
static PyObject *f_masks;            /* `fuse.MASKS` */
static PyObject *f_reductions;       /* `fuse.REDUCTIONS` */
static PyObject *f_square_exponents; /* `fuse.SQUARE_EXPONENTS` */
static PyObject *f_square, *f_where, *f_vector, *f_scalar, *f_array, *f_store, *f_mul; /* Its names, "mul" among them */
static PyObject *f_float64, *f_boolean; /* `fuse.FLOAT64` and `fuse.BOOLEAN` */
static PyObject *f_power, *f_pow;    /* `np.power` and `operator.pow` */
static PyObject *f_is_basic_index;   /* `rules.is_basic_index` */
static long long f_exact_integer;    /* `fuse.EXACT_INTEGER` */
static PyObject *f_whole_scalars;    /* `fuse.WHOLE_SCALARS` */
static PyObject *f_scaled;           /* `fuse.SCALED`: `np.radians` and the others, by the constant each multiplies by */
static Py_ssize_t f_most_depth;      /* `fuse.MOST_DEPTH` */
static Py_ssize_t f_short_row, f_slight_row; /* `fuse.SHORT_ROW`, `fuse.SLIGHT_ROW` */
static PyObject *f_row_operations;           /* `fuse.ROW_OPERATIONS` */
static PyObject *s_isnative;
static PyObject *s_active, *s_caught, *s_found, *s_note, *s_issue; /* The names of `evaluate._Quiet` */

/* What the finder notes of one node the computation takes. */
typedef struct {
    PyObject *node;   /* The node, borrowed from the list of them; NULL for an empty place of the table. */
    int uses;         /* How many operands of the computation's nodes it is: `_Finder.uses`. */
    int unread;       /* How often it is kept, unread, as the old contents of an array: `_Finder.unread`. */
    char held;        /* Whether something outside the computation will want its value: `_Finder.held`. */
    char inside;      /* Whether it is inside an expression found so far: `_Finder.inside`. */
} Taken;

typedef struct {
    Taken *table;             /* The nodes the computation takes, by their addresses: `_Finder.taking`. */
    size_t mask;              /* The table's size less one; the size is a power of two. */
    Py_ssize_t fewest;        /* The fewest elements an expression's operations run over for it to be read. */
    Taken **journal;          /* The nodes put inside, in order: `_Finder.journal`. */
    Py_ssize_t journal_count; /* How many there are. */
    Py_ssize_t journal_room;  /* How many the journal has room for. */
    Py_ssize_t depth;         /* How many operations deep the operand being read is. */
} Finder;

/* Where a node is in the table, or where it would go. */
static Taken *
taken_place(Finder *finder, PyObject *node)
{
    size_t place = (((uintptr_t)node >> 4) * (size_t)0x9E3779B97F4A7C15ULL) & finder->mask;
    while (finder->table[place].node != NULL && finder->table[place].node != node) {
        place = (place + 1) & finder->mask;
    }
    return &finder->table[place];
}

/* What is noted of a node, or NULL for one the computation does not take. */
static Taken *
taken_of(Finder *finder, PyObject *node)
{
    Taken *taken = taken_place(finder, node);
    return taken->node == node ? taken : NULL;
}

/* `_Finder._put_inside`. 0, or -1 with an error set. */
static int
put_inside(Finder *finder, Taken *taken)
{
    if (finder->journal_count == finder->journal_room) {
        Py_ssize_t room = finder->journal_room ? finder->journal_room * 2 : 64;
        Taken **grown = PyMem_Realloc(finder->journal, (size_t)room * sizeof(Taken *));
        if (grown == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        finder->journal = grown;
        finder->journal_room = room;
    }
    taken->inside = 1;
    finder->journal[finder->journal_count++] = taken;
    return 0;
}

/* `_Finder.give_up`. */
static void
give_up(Finder *finder, Py_ssize_t mark)
{
    while (finder->journal_count > mark) {
        finder->journal[--finder->journal_count]->inside = 0;
    }
}

/* Whether two strings are the same text. The names and kinds compared here are constants of the layer's modules,
 * and nearly always the same object. */
static int
same_text(PyObject *first, PyObject *second)
{
    return first == second
        || (PyUnicode_CheckExact(first) && PyUnicode_CheckExact(second) && PyUnicode_Compare(first, second) == 0);
}

/* Whether an element type is `wanted`: `dtype == wanted`. */
static int
dtype_is(PyObject *dtype, PyObject *wanted)
{
    if (dtype == wanted) {
        return 1;
    }
    int same = PyObject_RichCompareBool(dtype, wanted, Py_EQ);
    if (same < 0) {
        PyErr_Clear();
        return 0;
    }
    return same;
}

/* `graph.is_known(shape)`. */
static int
shape_known(PyObject *shape)
{
    for (Py_ssize_t axis = 0; axis < PyTuple_GET_SIZE(shape); axis++) {
        long long length = PyLong_AsLongLong(PyTuple_GET_ITEM(shape, axis));
        if (length == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return 0; /* Too large for a length, as a stand-in a reshape doubled is. */
        }
        if (length >= unknown_floor) {
            return 0;
        }
    }
    return 1;
}

/* Whether a shape has `fewest` elements or more: `fuse._elements(shape) >= fewest`, without the product ever
 * being too large to hold. */
static int
has_elements(PyObject *shape, Py_ssize_t fewest)
{
    Py_ssize_t count = 1;
    int zero = 0;
    for (Py_ssize_t axis = 0; axis < PyTuple_GET_SIZE(shape); axis++) {
        long long length = PyLong_AsLongLong(PyTuple_GET_ITEM(shape, axis));
        if (length == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            length = PY_SSIZE_T_MAX;
        }
        if (length <= 0) {
            zero = 1;
        } else if (count > PY_SSIZE_T_MAX / length) {
            count = PY_SSIZE_T_MAX;
        } else {
            count *= (Py_ssize_t)length;
        }
    }
    return (zero ? 0 : count) >= fewest;
}

/* The length of a shape along its last axis; -1 for one too large to be a length. */
static long long
last_length(PyObject *shape)
{
    long long length = PyLong_AsLongLong(PyTuple_GET_ITEM(shape, PyTuple_GET_SIZE(shape) - 1));
    if (length == -1 && PyErr_Occurred()) {
        PyErr_Clear();
    }
    return length;
}

/* `fuse.call_of(node)`: 1 with the callable and its operands as written, which are `count` items of the node's
 * own arguments; 0 for None. */
static int
call_of(PyObject *node, PyObject **operation, PyObject ***items, Py_ssize_t *count)
{
    PyObject *made = SLOT(node, node_at[N_OPERATION]), *args = SLOT(node, node_at[N_ARGS]);
    PyObject *kind = SLOT(node, node_at[N_KIND]);
    if (!PyTuple_CheckExact(args)) {
        return 0;
    }
    PyObject **all = &PyTuple_GET_ITEM(args, 0);
    Py_ssize_t given = PyTuple_GET_SIZE(args);
    if (same_text(kind, elementwise_kind)) {
        PyObject *kwargs = SLOT(node, node_at[N_KWARGS]);
        if (!PyDict_Check(kwargs) || PyDict_GET_SIZE(kwargs) != 0) {
            return 0;
        }
        *operation = made;
        *items = all;
        *count = given;
        return 1;
    }
    if (!same_text(kind, write_kind)) {
        return 0;
    }
    if (Py_TYPE(made) == Write) {
        if (PyTuple_GET_SIZE(SLOT(made, write_at[W_STEPS])) != 0) {
            return 0;
        }
        PyObject *perform = SLOT(made, write_at[W_PERFORM]);
        if (Py_TYPE(perform) == Into) {
            /* `np.add(a, b, out=x)` leaves nothing of what `x` held: the operands after it are the call's. */
            PyObject *function = SLOT(perform, into_function_at);
            if (!PyObject_TypeCheck(function, Ufunc) || given < 1) {
                return 0;
            }
            *operation = function;
            *items = all + 1;
            *count = given - 1;
            return 1;
        }
        PyObject *ufunc = PyDict_GetItemWithError(f_in_place, SLOT(node, node_at[N_NAME]));
        if (ufunc == NULL) {
            PyErr_Clear();
            return 0;
        }
        *operation = ufunc;
        *items = all;
        *count = given;
        return 1;
    }
    if (Py_TYPE(made) == FWholeCall) {
        /* A write `rewrite_write` has already turned into its ufunc. */
        int reads_old = SLOT(made, wholecall_at[WC_READS_OLD]) == Py_True;
        if (!reads_old && given < 1) {
            return 0;
        }
        *operation = SLOT(made, wholecall_at[WC_UFUNC]);
        *items = reads_old ? all : all + 1;
        *count = reads_old ? given : given - 1;
        return 1;
    }
    return 0;
}

/* Whether an object is a NumPy scalar of a whole-number or boolean type: `type(arg) in fuse.WHOLE_SCALARS`. */
static int
whole_scalar(PyObject *arg)
{
    int found = PySet_Contains(f_whole_scalars, (PyObject *)Py_TYPE(arg));
    if (found < 0) {
        PyErr_Clear(); /* A type that cannot be looked up is none of them. */
        return 0;
    }
    return found;
}

/* `fuse.operation_of(node)`: 1 with the operation's name in a pattern (borrowed) and its operands; 0 for None;
 * -1 with an error set. The operands are borrowed from the node, except for a conversion of `fuse.SCALED`, which is
 * `mul` of its one operand and its constant: those two are written into `spare`, two places the caller keeps for
 * as long as it reads the operands, since reading them may read another node with another `spare`. */
static int
operation_of(PyObject *node, PyObject **name, PyObject ***items, Py_ssize_t *count, PyObject **spare)
{
    PyObject *operation;
    if (!call_of(node, &operation, items, count)) {
        return 0;
    }
    PyObject *found = PyDict_GetItemWithError(f_operations, operation);
    if (found != NULL) {
        *name = found;
        return 1;
    }
    if (PyErr_Occurred()) {
        return -1;
    }
    PyObject *factor = PyDict_GetItemWithError(f_scaled, operation);
    if (factor != NULL) {
        if (*count != 1) {
            return 0;
        }
        spare[0] = (*items)[0];
        spare[1] = factor; /* Kept alive by `fuse.SCALED`. */
        *items = spare;
        *count = 2;
        *name = f_mul;
        return 1;
    }
    if (PyErr_Occurred()) {
        return -1;
    }
    if ((operation == f_power || operation == f_pow) && *count >= 2) {
        PyObject *exponent = (*items)[1];
        if (PyLong_CheckExact(exponent) || PyFloat_CheckExact(exponent) || whole_scalar(exponent)) {
            int square = PySequence_Contains(f_square_exponents, exponent);
            if (square < 0) {
                return -1;
            }
            if (square) {
                *name = f_square;
                *count = 1;
                return 1;
            }
        }
    }
    return 0;
}

/* `fuse._leaf_kind(arg, shape)`: `fuse.VECTOR` or `fuse.SCALAR` (borrowed), or `Miss` for None. */
static PyObject *
leaf_kind(PyObject *arg, PyObject *shape)
{
    if (PyFloat_CheckExact(arg) || Py_TYPE(arg) == &PyDoubleArrType_Type) {
        return f_scalar;
    }
    if (PyLong_CheckExact(arg)) {
        int overflow = 0;
        long long value = PyLong_AsLongLongAndOverflow(arg, &overflow);
        return !overflow && value >= -f_exact_integer && value <= f_exact_integer ? f_scalar : Miss;
    }
    if (whole_scalar(arg)) {
        return f_scalar; /* `np.int64(3)`: the float64 NumPy's conversion gives, whatever its value. */
    }
    long long wanted = last_length(shape), own;
    if (Py_TYPE(arg) == Node) {
        PyObject *dtype = SLOT(arg, node_at[N_DTYPE]), *has = SLOT(arg, node_at[N_SHAPE]);
        if (dtype != f_float64) {
            /* Another object that may still be the type: `arg.dtype != FLOAT64 or not arg.dtype.isnative`. */
            PyObject *native = dtype_is(dtype, f_float64) ? PyObject_GetAttr(dtype, s_isnative) : NULL;
            int is_native = native != NULL && PyObject_IsTrue(native) == 1;
            Py_XDECREF(native);
            PyErr_Clear();
            if (!is_native) {
                /* One whole number is a number, as its NumPy scalar is; with a dimension it is another type's array. */
                return PyTuple_GET_SIZE(has) == 0 && whole_kind(dtype) ? f_scalar : Miss;
            }
        }
        if (PyTuple_GET_SIZE(has) == 0) {
            return f_scalar;
        }
        own = last_length(has);
        if (own == 1 && wanted != 1) {
            return f_scalar;
        }
        return own == wanted && shape_known(has) ? f_vector : Miss;
    }
    if (Py_TYPE(arg) == Ndarray) {
        PyArrayObject *array = (PyArrayObject *)arg;
        PyArray_Descr *descr = PyArray_DESCR(array);
        if ((PyObject *)descr != f_float64 && !(descr->type_num == NPY_DOUBLE && PyArray_ISNBO(descr->byteorder)
                                                && dtype_is((PyObject *)descr, f_float64))) {
            return PyArray_NDIM(array) == 0 && whole_kind((PyObject *)descr) ? f_scalar : Miss;
        }
        if (PyArray_NDIM(array) == 0) {
            return f_scalar;
        }
        own = (long long)PyArray_DIM(array, PyArray_NDIM(array) - 1);
        if (own == 1 && wanted != 1) {
            return f_scalar;
        }
        return own == wanted ? f_vector : Miss; /* An array's lengths are all known. */
    }
    return Miss;
}

/* `_Finder._is_inner(node, shape)`: what is noted of the node when it may be computed inside an expression of
 * that shape, and NULL when it may not. */
static Taken *
inner_of(Finder *finder, PyObject *node, PyObject *shape)
{
    if (SLOT(node, node_at[N_OPERATION]) == Py_None) {
        return NULL;
    }
    Taken *taken = taken_of(finder, node);
    if (taken == NULL || taken->uses != 1 || taken->held) {
        return NULL;
    }
    PyObject *has = SLOT(node, node_at[N_SHAPE]);
    if (has != shape) {
        int same = PyObject_RichCompareBool(has, shape, Py_EQ);
        if (same != 1) {
            PyErr_Clear();
            return NULL;
        }
    }
    return taken;
}

/* `Leaf(kind, place)`. */
static PyObject *
new_leaf(PyObject *kind, Py_ssize_t place)
{
    PyObject *number = PyLong_FromSsize_t(place);
    PyObject *leaf = number ? FLeaf->tp_alloc(FLeaf, 0) : NULL;
    if (leaf == NULL) {
        Py_XDECREF(number);
        return NULL;
    }
    SLOT(leaf, leaf_at[LEAF_KIND]) = Py_NewRef(kind);
    SLOT(leaf, leaf_at[LEAF_PLACE]) = number;
    return leaf;
}

/* `Term(op, operands, node)`. Steals the reference to `operands`. */
static PyObject *
new_term(PyObject *op, PyObject *operands, PyObject *node)
{
    PyObject *term = FTerm->tp_alloc(FTerm, 0);
    if (term == NULL) {
        Py_DECREF(operands);
        return NULL;
    }
    SLOT(term, term_at[T_OP]) = Py_NewRef(op);
    SLOT(term, term_at[T_OPERANDS]) = operands;
    SLOT(term, term_at[T_NODE]) = Py_NewRef(node);
    return term;
}

/* `ViewOf(old, steps)`. */
static PyObject *
new_viewof(PyObject *old, PyObject *steps)
{
    PyObject *view = FViewOf->tp_alloc(FViewOf, 0);
    if (view != NULL) {
        SLOT(view, viewof_at[V_OLD]) = Py_NewRef(old);
        SLOT(view, viewof_at[V_STEPS]) = Py_NewRef(steps);
    }
    return view;
}

/* `Expression(root, term, operands, result, shape, target)`. Steals the reference to `term`. */
static PyObject *
new_expression(PyObject *root, PyObject *term, PyObject *operands, PyObject *result, PyObject *shape, PyObject *target)
{
    PyObject *expression = FExpression->tp_alloc(FExpression, 0);
    if (expression == NULL) {
        Py_DECREF(term);
        return NULL;
    }
    SLOT(expression, expression_at[E_ROOT]) = Py_NewRef(root);
    SLOT(expression, expression_at[E_TERM]) = term;
    SLOT(expression, expression_at[E_OPERANDS]) = Py_NewRef(operands);
    SLOT(expression, expression_at[E_RESULT]) = Py_NewRef(result);
    SLOT(expression, expression_at[E_SHAPE]) = Py_NewRef(shape);
    SLOT(expression, expression_at[E_TARGET]) = Py_NewRef(target);
    SLOT(expression, expression_at[E_SIGNATURE]) = Py_NewRef(Py_None);
    SLOT(expression, expression_at[E_NODES]) = Py_NewRef(Py_None);
    SLOT(expression, expression_at[E_ERRSTATES]) = Py_NewRef(Py_None);
    return expression;
}

/* `fuse._signature(part)`: the shape of a tree as nested tuples. A new reference. */
static PyObject *
signature_of(PyObject *part)
{
    if (Py_TYPE(part) != FTerm) {
        return PyTuple_Pack(2, SLOT(part, leaf_at[LEAF_KIND]), SLOT(part, leaf_at[LEAF_PLACE]));
    }
    PyObject *operands = SLOT(part, term_at[T_OPERANDS]);
    Py_ssize_t count = PyTuple_GET_SIZE(operands);
    PyObject *written = PyTuple_New(count + 1);
    if (written == NULL) {
        return NULL;
    }
    PyTuple_SET_ITEM(written, 0, Py_NewRef(SLOT(part, term_at[T_OP])));
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *below = signature_of(PyTuple_GET_ITEM(operands, place));
        if (below == NULL) {
            Py_DECREF(written);
            return NULL;
        }
        PyTuple_SET_ITEM(written, place + 1, below);
    }
    return written;
}

/* Add the nodes of a tree's terms to a list. 0, or -1 with an error set. */
static int
nodes_of(PyObject *part, PyObject *nodes)
{
    if (Py_TYPE(part) != FTerm) {
        return 0;
    }
    if (PyList_Append(nodes, SLOT(part, term_at[T_NODE])) < 0) {
        return -1;
    }
    PyObject *operands = SLOT(part, term_at[T_OPERANDS]);
    for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(operands); place++) {
        if (nodes_of(PyTuple_GET_ITEM(operands, place), nodes) < 0) {
            return -1;
        }
    }
    return 0;
}

/* One node and its number, to sort by. */
typedef struct {
    long long number;
    PyObject *node;
} Numbered;

static int
by_number(const void *first, const void *second)
{
    long long one = ((const Numbered *)first)->number, other = ((const Numbered *)second)->number;
    return (one > other) - (one < other);
}

/* Write down, on an expression that is kept, what planning it reads off its tree every time: the tree's
 * signature, the nodes of its terms in ascending number, and the different error settings those nodes were
 * written under. `fuse._plan_expression` and `fuse._replayed` work the same things out where they are not
 * written. 0, or -1 with an error set. */
static int
note_tree(PyObject *expression)
{
    PyObject *term = SLOT(expression, expression_at[E_TERM]);
    PyObject *signature = signature_of(term);
    PyObject *nodes = signature ? PyList_New(0) : NULL;
    PyObject *errstates = NULL;
    Numbered *sorted = NULL;
    int done = -1;
    if (nodes == NULL || nodes_of(term, nodes) < 0) {
        goto out;
    }
    Py_ssize_t count = PyList_GET_SIZE(nodes);
    sorted = PyMem_Malloc((size_t)(count ? count : 1) * sizeof(Numbered));
    errstates = PyList_New(0);
    if (sorted == NULL || errstates == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *node = PyList_GET_ITEM(nodes, place);
        sorted[place].node = node;
        sorted[place].number = PyLong_AsLongLong(SLOT(node, node_at[N_NUMBER]));
        PyObject *settings = SLOT(node, node_at[N_ERRSTATE]);
        int known = 0;
        for (Py_ssize_t each = 0; !known && each < PyList_GET_SIZE(errstates); each++) {
            known = PyList_GET_ITEM(errstates, each) == settings;
        }
        if (!known && PyList_Append(errstates, settings) < 0) {
            goto out;
        }
    }
    if (PyErr_Occurred()) {
        goto out;
    }
    qsort(sorted, (size_t)count, sizeof(Numbered), by_number);
    for (Py_ssize_t place = 0; place < count; place++) {
        Py_INCREF(sorted[place].node);
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *before = PyList_GET_ITEM(nodes, place);
        PyList_SET_ITEM(nodes, place, sorted[place].node);
        Py_DECREF(before);
    }
    PyObject *settings = PyList_AsTuple(errstates);
    if (settings == NULL) {
        goto out;
    }
    Py_SETREF(SLOT(expression, expression_at[E_SIGNATURE]), Py_NewRef(signature));
    Py_SETREF(SLOT(expression, expression_at[E_NODES]), Py_NewRef(nodes));
    Py_SETREF(SLOT(expression, expression_at[E_ERRSTATES]), settings);
    done = 0;
out:
    PyMem_Free(sorted);
    Py_XDECREF(signature);
    Py_XDECREF(nodes);
    Py_XDECREF(errstates);
    return done;
}

static PyObject *read_operation(Finder *finder, PyObject *node, PyObject *name, PyObject **written, Py_ssize_t count,
                                PyObject *shape, PyObject *operands);

/* `_Finder._operand`: a term, a leaf, `Miss` for None, or NULL with an error set. */
static PyObject *
read_operand(Finder *finder, PyObject *arg, PyObject *shape, PyObject *operands, int mask)
{
    if (Py_TYPE(arg) == Node && finder->depth < f_most_depth) {
        Taken *taken = inner_of(finder, arg, shape);
        if (taken != NULL) {
            PyObject *name, **items, *spare[2];
            Py_ssize_t count;
            int read = operation_of(arg, &name, &items, &count, spare);
            if (read < 0) {
                return NULL;
            }
            if (read) {
                int is_mask = PySet_Contains(f_masks, name);
                if (is_mask < 0) {
                    return NULL;
                }
                if (is_mask == mask && dtype_is(SLOT(arg, node_at[N_DTYPE]), mask ? f_boolean : f_float64)) {
                    finder->depth++;
                    PyObject *term = read_operation(finder, arg, name, items, count, shape, operands);
                    finder->depth--;
                    if (term == NULL) {
                        return NULL;
                    }
                    if (term != Miss) {
                        if (put_inside(finder, taken) < 0) {
                            Py_DECREF(term);
                            return NULL;
                        }
                        return term;
                    }
                }
            }
        }
    }
    if (mask) {
        return Miss;
    }
    PyObject *kind = Py_TYPE(arg) == FViewOf ? f_vector : leaf_kind(arg, shape);
    if (kind == Miss) {
        return Miss;
    }
    if (kind == f_vector) {
        for (Py_ssize_t place = 0; place < PyList_GET_SIZE(operands); place++) {
            if (PyList_GET_ITEM(operands, place) == arg) {
                return new_leaf(kind, place);
            }
        }
    }
    if (PyList_Append(operands, arg) < 0) {
        return NULL;
    }
    return new_leaf(kind, PyList_GET_SIZE(operands) - 1);
}

/* `_Finder._operation`: a term, `Miss` for None with nothing added to `operands`, or NULL with an error set. */
static PyObject *
read_operation(Finder *finder, PyObject *node, PyObject *name, PyObject **written, Py_ssize_t count, PyObject *shape,
               PyObject *operands)
{
    Py_ssize_t mark = PyList_GET_SIZE(operands), inside = finder->journal_count;
    int is_where = same_text(name, f_where);
    PyObject *parts = PyTuple_New(count);
    if (parts == NULL) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *part = read_operand(finder, written[place], shape, operands, is_where && place == 0);
        if (part == NULL || part == Miss) {
            Py_DECREF(parts);
            if (part == NULL || PyList_SetSlice(operands, mark, PyList_GET_SIZE(operands), NULL) < 0) {
                return NULL;
            }
            give_up(finder, inside);
            return Miss;
        }
        PyTuple_SET_ITEM(parts, place, part);
    }
    if (count == 2 && same_text(name, f_mul)) {
        /* `x * x` of one operand read twice is one multiplication of a value by itself. */
        PyObject *first = PyTuple_GET_ITEM(parts, 0), *second = PyTuple_GET_ITEM(parts, 1);
        if (Py_TYPE(first) == FLeaf && Py_TYPE(second) == FLeaf
            && same_text(SLOT(first, leaf_at[LEAF_KIND]), SLOT(second, leaf_at[LEAF_KIND]))
            && PyObject_RichCompareBool(SLOT(first, leaf_at[LEAF_PLACE]), SLOT(second, leaf_at[LEAF_PLACE]), Py_EQ) == 1) {
            PyObject *one = PyTuple_Pack(1, first);
            Py_DECREF(parts);
            return one ? new_term(f_square, one, node) : NULL;
        }
    }
    return new_term(name, parts, node);
}

/* `_Finder._term`. */
static PyObject *
read_term(Finder *finder, PyObject *node, PyObject *shape, PyObject *operands, int inner)
{
    PyObject *name, **items, *spare[2];
    Py_ssize_t count;
    int read = operation_of(node, &name, &items, &count, spare);
    if (read <= 0) {
        return read < 0 ? NULL : Miss;
    }
    PyObject *term = read_operation(finder, node, name, items, count, shape, operands);
    if (term != NULL && term != Miss && inner) {
        Taken *taken = taken_of(finder, node);
        if (taken == NULL || put_inside(finder, taken) < 0) {
            Py_DECREF(term);
            if (taken == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "a node inside an expression is not one the computation takes");
            }
            return NULL;
        }
    }
    return term;
}

/* `_Finder._reduction`. */
static PyObject *
read_reduction(Finder *finder, PyObject *root, PyObject *operands)
{
    PyObject *name = SLOT(root, node_at[N_NAME]), *args = SLOT(root, node_at[N_ARGS]), *kwargs = SLOT(root, node_at[N_KWARGS]);
    int listed = PySet_Contains(f_reductions, name);
    if (listed < 0) {
        return NULL;
    }
    if (!listed || PyTuple_GET_SIZE(args) != 1 || !PyDict_Check(kwargs) || PyDict_GET_SIZE(kwargs) != 0
        || !dtype_is(SLOT(root, node_at[N_DTYPE]), f_float64)) {
        return Miss;
    }
    PyObject *reduced = PyTuple_GET_ITEM(args, 0);
    if (Py_TYPE(reduced) != Node) {
        return Miss;
    }
    PyObject *shape = SLOT(reduced, node_at[N_SHAPE]);
    if (PyTuple_GET_SIZE(shape) == 0 || inner_of(finder, reduced, shape) == NULL || !has_elements(shape, finder->fewest)) {
        return Miss;
    }
    PyObject *inner = read_term(finder, reduced, shape, operands, 1);
    if (inner == NULL || inner == Miss) {
        return inner;
    }
    PyObject *one = PyTuple_Pack(1, inner);
    Py_DECREF(inner);
    PyObject *term = one ? new_term(name, one, root) : NULL;
    return term ? new_expression(root, term, operands, name, shape, Py_None) : NULL;
}

/* The most operands a write is read with on the C stack; one of more has them in memory of its own. */
#define MOST_WRITTEN 16

/* `_Finder._store`. */
static PyObject *
read_store(Finder *finder, PyObject *root, PyObject *operands)
{
    PyObject *write = SLOT(root, node_at[N_OPERATION]), *args = SLOT(root, node_at[N_ARGS]);
    Py_ssize_t given = PyTuple_GET_SIZE(args);
    PyObject *old = given ? PyTuple_GET_ITEM(args, 0) : NULL;
    if (old == NULL || Py_TYPE(old) != Node || !dtype_is(SLOT(old, node_at[N_DTYPE]), f_float64)
        || !shape_known(SLOT(old, node_at[N_SHAPE]))) {
        return Miss;
    }
    PyObject *steps = SLOT(write, write_at[W_STEPS]), *perform = SLOT(write, write_at[W_PERFORM]);
    Py_ssize_t views = PyTuple_GET_SIZE(steps);
    if (!has_elements(SLOT(old, node_at[N_SHAPE]), finder->fewest)) {
        return Miss; /* The part is no larger than the array. */
    }
    PyObject *result = Miss, *part = NULL, *target = NULL, *term = NULL;
    if (Py_TYPE(perform) == Assign) {
        PyObject *index = SLOT(perform, assign_at[A_INDEX]);
        PyObject *basic = PyObject_CallOneArg(f_is_basic_index, index);
        int is_basic = basic ? PyObject_IsTrue(basic) : -1;
        Py_XDECREF(basic);
        if (is_basic <= 0) {
            return is_basic < 0 ? NULL : Miss;
        }
        PyObject *whole = views ? SLOT(PyTuple_GET_ITEM(steps, views - 1), step_at[S_SHAPE]) : SLOT(old, node_at[N_SHAPE]);
        part = basic_shape(whole, index);
        if (part == NULL || part == Miss) {
            return part;
        }
        PyObject *stored = given >= 2 ? PyTuple_GET_ITEM(args, 1) : NULL;
        if (PyTuple_GET_SIZE(part) == 0 || stored == NULL || Py_TYPE(stored) != Node || inner_of(finder, stored, part) == NULL) {
            goto done;
        }
        term = read_term(finder, stored, part, operands, 1);
        if (term == NULL) {
            goto fail;
        }
        if (term == Miss) {
            term = NULL;
            goto done;
        }
        PyObject *tail = PyTuple_Pack(1, index);
        PyObject *kwargs = tail ? PyDict_New() : NULL;
        PyObject *step = kwargs ? new_step(getitem, s_getitem, tail, kwargs, part, f_float64, SLOT(root, node_at[N_ORIGIN]), 0) : NULL;
        PyObject *all = step ? steps_with(steps, step) : NULL;
        target = all ? new_viewof(old, all) : NULL;
        Py_XDECREF(tail);
        Py_XDECREF(kwargs);
        Py_XDECREF(step);
        Py_XDECREF(all);
        if (target == NULL) {
            goto fail;
        }
        result = new_expression(root, term, operands, f_store, part, target);
        term = NULL; /* Stolen. */
        goto done;
    }
    if (views == 0) {
        return Miss;
    }
    PyObject *last = PyTuple_GET_ITEM(steps, views - 1);
    part = Py_NewRef(SLOT(last, step_at[S_SHAPE]));
    if (PyTuple_GET_SIZE(part) == 0 || !shape_known(part) || !dtype_is(SLOT(last, step_at[S_DTYPE]), f_float64)) {
        goto done;
    }
    target = new_viewof(old, steps);
    if (target == NULL) {
        goto fail;
    }
    PyObject *name = NULL, *kept[MOST_WRITTEN], **written = kept;
    Py_ssize_t count;
    if (given > MOST_WRITTEN) {
        written = PyMem_Malloc((size_t)given * sizeof(PyObject *));
        if (written == NULL) {
            PyErr_NoMemory();
            goto fail;
        }
    }
    if (Py_TYPE(perform) == Into) {
        /* A ufunc with `out=x[index]` stores the operation on its own operands. */
        PyObject *function = SLOT(perform, into_function_at);
        name = PyDict_GetItemWithError(f_arithmetic, function);
        if (name == NULL && !PyErr_Occurred()) {
            name = PyDict_GetItemWithError(f_unary, function);
        }
        count = given - 1;
        for (Py_ssize_t place = 0; place < count; place++) {
            written[place] = PyTuple_GET_ITEM(args, place + 1);
        }
    } else {
        /* `x[index] += e` stores the operation, which reads what the part held. */
        PyObject *ufunc = PyDict_GetItemWithError(f_in_place, SLOT(root, node_at[N_NAME]));
        name = ufunc ? PyDict_GetItemWithError(f_arithmetic, ufunc) : NULL;
        count = given;
        written[0] = target;
        for (Py_ssize_t place = 1; place < count; place++) {
            written[place] = PyTuple_GET_ITEM(args, place);
        }
    }
    if (name != NULL) {
        term = read_operation(finder, root, name, written, count, part, operands);
    }
    if (written != kept) {
        PyMem_Free(written);
    }
    if (name == NULL || term == NULL) {
        if (PyErr_Occurred()) {
            term = NULL;
            goto fail;
        }
        goto done;
    }
    if (term == Miss) {
        term = NULL;
        goto done;
    }
    result = new_expression(root, term, operands, f_store, part, target);
    term = NULL; /* Stolen. */
done:
    Py_XDECREF(part);
    Py_XDECREF(target);
    Py_XDECREF(term);
    return result;
fail:
    Py_XDECREF(part);
    Py_XDECREF(target);
    Py_XDECREF(term);
    return NULL;
}

/* `_Finder.expression`: an `Expression`, `Miss` for None, or NULL with an error set. */
static PyObject *
read_expression(Finder *finder, PyObject *root)
{
    PyObject *shape = SLOT(root, node_at[N_SHAPE]), *kind = SLOT(root, node_at[N_KIND]);
    if (!shape_known(shape)) {
        return Miss;
    }
    PyObject *operands = PyList_New(0);
    if (operands == NULL) {
        return NULL;
    }
    PyObject *result = Miss;
    PyObject *made = SLOT(root, node_at[N_OPERATION]);
    if (same_text(kind, reduction_kind)) {
        result = read_reduction(finder, root, operands);
    } else if (same_text(kind, write_kind) && Py_TYPE(made) == Write
               && (PyTuple_GET_SIZE(SLOT(made, write_at[W_STEPS])) != 0 || Py_TYPE(SLOT(made, write_at[W_PERFORM])) == Assign)) {
        result = read_store(finder, root, operands);
    } else if (PyTuple_GET_SIZE(shape) != 0 && dtype_is(SLOT(root, node_at[N_DTYPE]), f_float64)
               && has_elements(shape, finder->fewest)) {
        PyObject *term = read_term(finder, root, shape, operands, 0);
        result = term == NULL || term == Miss ? term : new_expression(root, term, operands, f_array, shape, Py_None);
    }
    Py_DECREF(operands);
    return result;
}

/* `fuse.count_operations(part)`. */
static Py_ssize_t
count_terms(PyObject *part)
{
    if (Py_TYPE(part) != FTerm) {
        return 0;
    }
    PyObject *operands = SLOT(part, term_at[T_OPERANDS]);
    Py_ssize_t count = 1;
    for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(operands); place++) {
        count += count_terms(PyTuple_GET_ITEM(operands, place));
    }
    return count;
}

/* `fuse.worth_alone`: whether one operation by itself is worth a kernel. It gives an array of two dimensions or
 * more whose innermost is shorter than `fuse.SHORT_ROW`, and an array it reads along that dimension is the same
 * along an outer one; or the operation is one of `fuse.ROW_OPERATIONS` and an operand is one value for each row or
 * does not lie in one run; or the innermost is `fuse.SLIGHT_ROW` at most and an operand does not lie in one run.
 * An operand with its value is asked how it lies, and of those still pending a view counts as cutting its rows. */
static int
worth_alone(PyObject *expression)
{
    PyObject *shape = SLOT(expression, expression_at[E_SHAPE]), *operands = SLOT(expression, expression_at[E_OPERANDS]);
    PyObject *term = SLOT(expression, expression_at[E_TERM]);
    if (!same_text(SLOT(expression, expression_at[E_RESULT]), f_array) || !PyTuple_CheckExact(shape)
        || PyTuple_GET_SIZE(shape) < 2 || !PyList_CheckExact(operands)) {
        return 0;
    }
    Py_ssize_t dimensions = PyTuple_GET_SIZE(shape);
    PyObject *last = PyTuple_GET_ITEM(shape, dimensions - 1);
    if (!PyLong_CheckExact(last)) {
        return 0;
    }
    Py_ssize_t length = PyLong_AsSsize_t(last);
    if (length == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return 0;
    }
    if (length <= 0 || length >= f_short_row) {
        return 0;
    }
    int by_rows = Py_TYPE(term) == FTerm ? PySequence_Contains(f_row_operations, SLOT(term, term_at[T_OP])) : 0;
    if (by_rows < 0) {
        PyErr_Clear();
        by_rows = 0;
    }
    for (Py_ssize_t place = 0; place < PyList_GET_SIZE(operands); place++) {
        PyObject *operand = PyList_GET_ITEM(operands, place);
        PyObject *value = NULL;
        Py_ssize_t laid = 0, along = 0;
        int same = 1, many = 0, pending_view = 0;
        if (Py_TYPE(operand) == Node) {
            PyObject *written = SLOT(operand, node_at[N_SHAPE]);
            if (written == NULL || !PyTuple_CheckExact(written) || PyTuple_GET_SIZE(written) == 0) {
                continue;
            }
            laid = PyTuple_GET_SIZE(written);
            same = laid == dimensions;
            for (Py_ssize_t axis = 0; axis < laid; axis++) {
                Py_ssize_t count = PyLong_AsSsize_t(PyTuple_GET_ITEM(written, axis));
                if (count == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                    count = -1;
                }
                many = many || count != 1;
                along = count;
                if (same && count != PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, axis))) {
                    same = 0;
                }
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
                continue;
            }
            if (SLOT(operand, node_at[N_OPERATION]) != Py_None) {
                PyObject *kind = SLOT(operand, node_at[N_KIND]);
                pending_view = kind == view_kind ? 1 : PyObject_RichCompareBool(kind, view_kind, Py_EQ);
                if (pending_view < 0) {
                    PyErr_Clear();
                    pending_view = 0;
                }
            } else {
                value = SLOT(operand, node_at[N_VALUE]);
            }
        } else if (Py_TYPE(operand) == Ndarray) {
            PyArrayObject *array = (PyArrayObject *)operand;
            laid = PyArray_NDIM(array);
            if (laid == 0) {
                continue;
            }
            same = laid == dimensions;
            for (Py_ssize_t axis = 0; axis < laid; axis++) {
                Py_ssize_t count = PyArray_DIM(array, (int)axis);
                many = many || count != 1;
                along = count;
                if (same && count != PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, axis))) {
                    same = 0;
                }
            }
            if (PyErr_Occurred()) {
                PyErr_Clear();
                continue;
            }
            value = operand;
        } else {
            continue; /* A number. */
        }
        if (along != length) {
            if (by_rows && along == 1 && many) {
                return 1; /* One value for each row. */
            }
            continue;
        }
        if (!same) {
            return 1;
        }
        if (!by_rows && length > f_slight_row) {
            continue;
        }
        if (pending_view) {
            return 1;
        }
        if (value != NULL && Py_TYPE(value) == Ndarray && !PyArray_IS_C_CONTIGUOUS((PyArrayObject *)value)) {
            return 1;
        }
    }
    return 0;
}

/* `find(todo, fewest, alone)`: `fuse.find`. Returns (the expressions, the numbers of the nodes inside them). */
static PyObject *
module_find(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 3 || !PyList_CheckExact(args[0]) || FExpression == NULL) {
        PyErr_SetString(PyExc_TypeError, "find(todo, fewest, alone) takes a list of nodes, after find_setup()");
        return NULL;
    }
    PyObject *todo = args[0];
    Py_ssize_t nodes = PyList_GET_SIZE(todo);
    Finder finder = {NULL, 0, PyLong_AsSsize_t(args[1]), NULL, 0, 0, 0};
    int alone = PyObject_IsTrue(args[2]);
    if (alone < 0 || (finder.fewest == -1 && PyErr_Occurred())) {
        return NULL;
    }
    size_t size = 16;
    while (size < (size_t)nodes * 2) {
        size *= 2;
    }
    finder.table = PyMem_Calloc(size, sizeof(Taken));
    if (finder.table == NULL) {
        return PyErr_NoMemory();
    }
    finder.mask = size - 1;
    PyObject *found = NULL, *inside = NULL, *result = NULL;
    for (Py_ssize_t place = 0; place < nodes; place++) {
        PyObject *node = PyList_GET_ITEM(todo, place);
        if (Py_TYPE(node) != Node) {
            PyErr_SetString(PyExc_TypeError, "find() takes nodes");
            goto done;
        }
        taken_place(&finder, node)->node = node;
    }
    /* `fuse.count_uses`. */
    for (Py_ssize_t place = 0; place < nodes; place++) {
        PyObject *node = PyList_GET_ITEM(todo, place);
        PyObject *read = SLOT(node, node_at[N_ARGS]), *made = SLOT(node, node_at[N_OPERATION]);
        if (!PyTuple_CheckExact(read)) {
            continue;
        }
        Py_ssize_t first = 0;
        if (Py_TYPE(made) == FWholeCall && SLOT(made, wholecall_at[WC_READS_OLD]) != Py_True && PyTuple_GET_SIZE(read) > 0) {
            Taken *old = taken_of(&finder, PyTuple_GET_ITEM(read, 0));
            if (old != NULL) {
                old->unread++;
            }
            first = 1;
        }
        for (Py_ssize_t each = first; each < PyTuple_GET_SIZE(read); each++) {
            PyObject *arg = PyTuple_GET_ITEM(read, each);
            Taken *taken = Py_TYPE(arg) == Node ? taken_of(&finder, arg) : NULL;
            if (taken != NULL) {
                taken->uses++;
            }
        }
    }
    /* `fuse.held_among`: a reference that the computation's nodes and the list do not account for. Nothing here
     * holds one of its own. */
    for (size_t place = 0; place < size; place++) {
        Taken *taken = &finder.table[place];
        if (taken->node != NULL) {
            taken->held = Py_REFCNT(taken->node) - 1 > (Py_ssize_t)taken->uses + taken->unread;
        }
    }
    found = PyList_New(0);
    if (found == NULL) {
        goto done;
    }
    for (Py_ssize_t place = 0; place < nodes; place++) {
        PyObject *node = PyList_GET_ITEM(todo, place);
        if (taken_of(&finder, node)->inside) {
            continue;
        }
        Py_ssize_t mark = finder.journal_count;
        PyObject *expression = read_expression(&finder, node);
        if (expression == NULL) {
            goto done;
        }
        int kept = 0;
        if (expression != Miss) {
            kept = alone || count_terms(SLOT(expression, expression_at[E_TERM])) > 1
                || same_text(SLOT(expression, expression_at[E_RESULT]), f_store) || worth_alone(expression);
            if (kept && (note_tree(expression) < 0 || PyList_Append(found, expression) < 0)) {
                Py_DECREF(expression);
                goto done;
            }
            Py_DECREF(expression);
        }
        if (!kept) {
            give_up(&finder, mark);
        }
    }
    inside = PySet_New(NULL);
    for (Py_ssize_t place = 0; inside != NULL && place < finder.journal_count; place++) {
        if (PySet_Add(inside, SLOT(finder.journal[place]->node, node_at[N_NUMBER])) < 0) {
            Py_CLEAR(inside);
        }
    }
    if (inside != NULL) {
        result = PyTuple_Pack(2, found, inside);
    }
done:
    PyMem_Free(finder.table);
    PyMem_Free(finder.journal);
    Py_XDECREF(found);
    Py_XDECREF(inside);
    return result;
}

static int offsets(PyTypeObject *type, const char *const *names, Py_ssize_t count, Py_ssize_t *at);
static int take(PyObject *given, const char *name, PyTypeObject *type, PyObject **into);

/* `find_setup(**given)`: be told the classes, tables and names of fuse.py. */
static PyObject *
module_find_setup(PyObject *module, PyObject *args, PyObject *given)
{
    if (given == NULL || PyTuple_GET_SIZE(args) != 0 || Node == NULL) {
        PyErr_SetString(PyExc_TypeError, "find_setup() takes keywords only, after setup()");
        return NULL;
    }
    PyObject *expression = NULL, *term = NULL, *leaf = NULL, *viewof = NULL, *wholecall = NULL;
    if (take(given, "Expression", &PyType_Type, &expression) || take(given, "Term", &PyType_Type, &term)
        || take(given, "Leaf", &PyType_Type, &leaf) || take(given, "ViewOf", &PyType_Type, &viewof)
        || take(given, "WholeCall", &PyType_Type, &wholecall)
        || offsets((PyTypeObject *)expression, EXPRESSION_FIELDS, E_FIELDS, expression_at)
        || offsets((PyTypeObject *)term, TERM_FIELDS, T_FIELDS, term_at)
        || offsets((PyTypeObject *)leaf, LEAF_FIELD_NAMES, LEAF_FIELDS, leaf_at)
        || offsets((PyTypeObject *)viewof, VIEWOF_FIELDS, V_FIELDS, viewof_at)
        || offsets((PyTypeObject *)wholecall, WHOLECALL_FIELDS, WC_FIELDS, wholecall_at)
        || take(given, "operations", &PyDict_Type, &f_operations) || take(given, "in_place", &PyDict_Type, &f_in_place)
        || take(given, "arithmetic", &PyDict_Type, &f_arithmetic) || take(given, "unary", &PyDict_Type, &f_unary)
        || take(given, "masks", &PyFrozenSet_Type, &f_masks) || take(given, "reductions", &PyFrozenSet_Type, &f_reductions)
        || take(given, "square_exponents", &PyTuple_Type, &f_square_exponents)
        || take(given, "square", &PyUnicode_Type, &f_square) || take(given, "where", &PyUnicode_Type, &f_where)
        || take(given, "vector", &PyUnicode_Type, &f_vector) || take(given, "scalar", &PyUnicode_Type, &f_scalar)
        || take(given, "array", &PyUnicode_Type, &f_array) || take(given, "store", &PyUnicode_Type, &f_store)
        || take(given, "mul", &PyUnicode_Type, &f_mul) || take(given, "float64", NULL, &f_float64)
        || take(given, "boolean", NULL, &f_boolean) || take(given, "power", NULL, &f_power) || take(given, "pow", NULL, &f_pow)
        || take(given, "is_basic_index", NULL, &f_is_basic_index)
        || take(given, "whole_scalars", &PyFrozenSet_Type, &f_whole_scalars)
        || take(given, "scaled", &PyDict_Type, &f_scaled)) {
        Py_XDECREF(expression);
        Py_XDECREF(term);
        Py_XDECREF(leaf);
        Py_XDECREF(viewof);
        Py_XDECREF(wholecall);
        return NULL;
    }
    PyObject *exact = PyDict_GetItemString(given, "exact_integer"), *depth = PyDict_GetItemString(given, "most_depth");
    PyObject *short_row = PyDict_GetItemString(given, "short_row"), *slight_row = PyDict_GetItemString(given, "slight_row");
    if (exact == NULL || depth == NULL || short_row == NULL || slight_row == NULL) {
        PyErr_SetString(PyExc_KeyError, "find_setup() was not given 'exact_integer', 'most_depth', 'short_row' and 'slight_row'");
        return NULL;
    }
    f_exact_integer = PyLong_AsLongLong(exact);
    f_most_depth = PyLong_AsSsize_t(depth);
    f_short_row = PyLong_AsSsize_t(short_row);
    f_slight_row = PyLong_AsSsize_t(slight_row);
    PyObject *row_operations = PyDict_GetItemString(given, "row_operations");
    if (row_operations == NULL || !PyTuple_CheckExact(row_operations)) {
        PyErr_SetString(PyExc_KeyError, "find_setup() was not given the tuple 'row_operations'");
        return NULL;
    }
    Py_XSETREF(f_row_operations, Py_NewRef(row_operations));
    if (PyErr_Occurred()) {
        return NULL;
    }
    Py_XSETREF(FTerm, (PyTypeObject *)term);
    Py_XSETREF(FLeaf, (PyTypeObject *)leaf);
    Py_XSETREF(FViewOf, (PyTypeObject *)viewof);
    Py_XSETREF(FWholeCall, (PyTypeObject *)wholecall);
    Py_XSETREF(FExpression, (PyTypeObject *)expression);
    Py_RETURN_NONE;
}

/* What stands as `tp_getattro` of `namespace.LazyNumPy` while the core is in use: `np.tanh` on the stand-in. A name
 * that `LazyNumPy.__getattr__` answered before, while the module still has under that name what it had then, is
 * answered here with the same object; that is the whole of what that method does for such a name, and the
 * interpreter reaches it only after it has looked for the name everywhere else. Every other name goes the
 * class's own way. */
static getattrofunc standin_getattr_before;

static PyObject *
standin_getattr(PyObject *self, PyObject *name)
{
    if (enabled && Py_TYPE(self) == StandInModule && PyUnicode_CheckExact(name)) {
        PyObject *module = SLOT(self, module_at[M_MODULE]), *answers = SLOT(self, module_at[M_ANSWERS]);
        if (module != NULL && answers != NULL && PyModule_CheckExact(module) && PyDict_CheckExact(answers)) {
            PyObject *answered = PyDict_GetItemWithError(answers, name);
            if (answered != NULL && PyTuple_CheckExact(answered) && PyTuple_GET_SIZE(answered) == 2) {
                PyObject *now = PyDict_GetItemWithError(PyModule_GetDict(module), name);
                if (now != NULL && now == PyTuple_GET_ITEM(answered, 0)) {
                    return Py_NewRef(PyTuple_GET_ITEM(answered, 1));
                }
            }
            if (PyErr_Occurred()) {
                return NULL;
            }
        }
    }
    return standin_getattr_before(self, name);
}

/* ---- The module. ---- */

/* The frame an argument names: the frame itself, or for None the one that called into this module. */
static PyFrameObject *
frame_given(PyObject *frame)
{
    if (frame == NULL || frame == Py_None) {
        return PyEval_GetFrame();
    }
    if (!PyFrame_Check(frame)) {
        PyErr_SetString(PyExc_TypeError, "frame must be a frame or None");
        return NULL;
    }
    return (PyFrameObject *)frame;
}

/* `_core.deferred(operation, name, args, kwargs, shape, dtype, kind, strict=False, origin=None, spec=0, frame=None)`:
 * `graph.deferred`, for every caller in Python. */
static PyObject *
module_deferred(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count < 7 || count > 11) {
        PyErr_SetString(PyExc_TypeError, "deferred() takes from 7 to 11 positional arguments");
        return NULL;
    }
    if (!PyTuple_CheckExact(args[2]) || !PyDict_CheckExact(args[3])) {
        PyErr_SetString(PyExc_TypeError, "deferred() takes a tuple of arguments and a dict of keywords");
        return NULL;
    }
    PyFrameObject *frame = frame_given(count > 10 ? args[10] : NULL);
    if (frame == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "deferred() needs a Python frame to find the line from");
        }
        return NULL;
    }
    PyObject *strict = count > 7 ? args[7] : Py_False;
    int is_strict = PyObject_IsTrue(strict);
    if (is_strict < 0) {
        return NULL;
    }
    return deferred(args[0], args[1], args[2], args[3], args[4], args[5], args[6], is_strict ? Py_True : Py_False,
                    count > 8 ? args[8] : NULL, count > 9 ? args[9] : zero, frame);
}

/* `_core.try_defer(operation, name, args, kwargs, ruled_as, frame)`: a lazy array, or `MISS`. */
static PyObject *
module_try_defer(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 6 || !PyTuple_CheckExact(args[2]) || !PyDict_CheckExact(args[3])) {
        return Py_NewRef(Miss); /* Whatever this is, Python knows what to say about it. */
    }
    PyFrameObject *frame = frame_given(args[5]);
    if (frame == NULL && PyErr_Occurred()) {
        return NULL;
    }
    if (declined_operation != NULL) {
        /* Was this very call, with these very objects, just asked about, and declined by the rules? */
        int same = declined_operation == args[0] && PyTuple_GET_SIZE(declined_args) == PyTuple_GET_SIZE(args[2])
                   && PyDict_GET_SIZE(declined_kwargs) == PyDict_GET_SIZE(args[3]);
        for (Py_ssize_t place = 0; same && place < PyTuple_GET_SIZE(args[2]); place++) {
            same = PyTuple_GET_ITEM(declined_args, place) == PyTuple_GET_ITEM(args[2], place);
        }
        Py_ssize_t position = 0;
        PyObject *name, *value;
        while (same && PyDict_Next(args[3], &position, &name, &value)) {
            same = PyDict_GetItemWithError(declined_kwargs, name) == value;
        }
        forget_declined();
        if (PyErr_Occurred()) {
            return NULL;
        }
        if (same) {
            Py_RETURN_NONE;
        }
    }
    PyObject *result = try_defer(args[0], args[1], args[2], args[3], args[4] == Py_None ? NULL : args[4], frame);
    if (result == Miss && declined_operation != NULL) {
        /* The rules were asked just now, for this call, and declined it. They are not to be asked again: working an
         * answer out can itself compute something, as comparing a slice that holds a lazy array does. */
        forget_declined();
        Py_RETURN_NONE;
    }
    return result == Miss ? Py_NewRef(Miss) : result;
}

/* `_core.plan(operation, args, kwargs)`: the remembered answer of `rules.plan` as a tuple (shape, dtype, kind,
 * spec), or `MISS`. `args` has a `Node` for each lazy operand, as `rules.plan` takes them. */
static PyObject *
module_plan(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (!enabled || count != 3 || !PyTuple_CheckExact(args[1]) || !PyDict_CheckExact(args[2])) {
        return Py_NewRef(Miss);
    }
    PyObject *result = remembered_plan(args[0], args[1], args[2], NULL);
    return result == Miss ? Py_NewRef(Miss) : result;
}

/* `_core.enable(on)`: take the quick roads, or hand every call to Python. Returns what it was. */
static PyObject *
module_enable(PyObject *module, PyObject *on)
{
    int now = PyObject_IsTrue(on);
    if (now < 0) {
        return NULL;
    }
    if (now && Node == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "setup() has not been called");
        return NULL;
    }
    int before = enabled;
    enabled = now;
    if (Node != NULL) {
        pools_in_use(now);
        if (now && StandInModule->tp_getattro != standin_getattr) {
            standin_getattr_before = StandInModule->tp_getattro;
            StandInModule->tp_getattro = standin_getattr;
            PyType_Modified(StandInModule);
        } else if (!now && StandInModule->tp_getattro == standin_getattr) {
            StandInModule->tp_getattro = standin_getattr_before;
            PyType_Modified(StandInModule);
        }
    }
    return PyBool_FromLong(before);
}

/* Read the offset of each named slot of a class into `at`. */
static int
offsets(PyTypeObject *type, const char *const *names, Py_ssize_t count, Py_ssize_t *at)
{
    for (Py_ssize_t each = 0; each < count; each++) {
        PyObject *descriptor = PyObject_GetAttrString((PyObject *)type, names[each]);
        if (descriptor == NULL) {
            return -1;
        }
        if (!Py_IS_TYPE(descriptor, &PyMemberDescr_Type)) {
            PyErr_Format(PyExc_TypeError, "%s.%s is not a slot", type->tp_name, names[each]);
            Py_DECREF(descriptor);
            return -1;
        }
        at[each] = ((PyMemberDescrObject *)descriptor)->d_member->offset;
        Py_DECREF(descriptor);
    }
    return 0;
}

/* Take one named item of what `setup()` was given, checking its type if one is asked for. */
static int
take(PyObject *given, const char *name, PyTypeObject *type, PyObject **into)
{
    PyObject *item = PyDict_GetItemString(given, name);
    if (item == NULL) {
        PyErr_Format(PyExc_KeyError, "setup() was not given '%s'", name);
        return -1;
    }
    if (type != NULL && !PyObject_TypeCheck(item, type)) {
        PyErr_Format(PyExc_TypeError, "setup(): '%s' must be a %s", name, type->tp_name);
        return -1;
    }
    Py_XSETREF(*into, Py_NewRef(item));
    return 0;
}

/* `_core.setup(**given)`: be told the layer's classes, memories and constants. Called once, by array.py. */
static PyObject *
module_setup(PyObject *module, PyObject *args, PyObject *given)
{
    if (given == NULL || PyTuple_GET_SIZE(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "setup() takes keywords only");
        return NULL;
    }
    PyObject *node = NULL, *lazy = NULL, *cell = NULL, *step = NULL, *ufunc = NULL, *ndarray = NULL, *write = NULL, *assign = NULL;
    PyObject *standin_call = NULL, *standin_module = NULL;
    if (take(given, "Node", &PyType_Type, &node) || take(given, "LazyArray", &PyType_Type, &lazy)
        || take(given, "Cell", &PyType_Type, &cell) || take(given, "Step", &PyType_Type, &step)
        || take(given, "ufunc", &PyType_Type, &ufunc) || take(given, "ndarray", &PyType_Type, &ndarray)
        || take(given, "graph_names", &PyDict_Type, &graph_names) || take(given, "rules_names", &PyDict_Type, &rules_names)
        || take(given, "numbers", NULL, &numbers) || take(given, "pending", &PySet_Type, &pending)
        || take(given, "forget", NULL, &forget) || take(given, "counts", NULL, &counts)
        || take(given, "origins", &PyDict_Type, &origins) || take(given, "specs", &PyDict_Type, &specs)
        || take(given, "plans", &PyDict_Type, &plans) || take(given, "admissible", &PyDict_Type, &admissible)
        || take(given, "caller", NULL, &python_caller) || take(given, "errstate", NULL, &python_errstate)
        || take(given, "view_kind", &PyUnicode_Type, &view_kind) || take(given, "sure_views", &PyFrozenSet_Type, &sure_views)
        || take(given, "fractional_scalars", &PyFrozenSet_Type, &fractional_scalars)
        || take(given, "nonzeros", &PyFrozenSet_Type, &nonzeros) || take(given, "getitem", NULL, &getitem)
        || take(given, "count_nonzero", NULL, &count_nonzero) || take(given, "Write", &PyType_Type, &write)
        || take(given, "Assign", &PyType_Type, &assign) || take(given, "fitting", &PySet_Type, &fitting)
        || take(given, "write_kind", &PyUnicode_Type, &write_kind) || take(given, "at_once", &PyUnicode_Type, &at_once_kind) || take(given, "handled_apart", NULL, &handled_apart)
        || take(given, "shaped_like", NULL, &shaped_like) || take(given, "fresh_functions", NULL, &fresh_functions) || take(given, "LazyCallable", &PyType_Type, &standin_call)
        || take(given, "LazyNumPy", &PyType_Type, &standin_module) || take(given, "slots", &PyTuple_Type, &slots)
        || take(given, "valued", &PyTuple_Type, &valued) || take(given, "in_place_ufuncs", &PyDict_Type, &in_place_ufuncs)
        || take(given, "never_deferred", &PyFrozenSet_Type, &never_deferred) || take(given, "at_once_result", NULL, &at_once_result)) {
        return NULL;
    }
    PyObject *generic = PyDict_GetItemString(given, "generic"), *kinds = PyDict_GetItemString(given, "deferred_kinds");
    if (generic == NULL || !PyType_Check(generic) || kinds == NULL || !PyUnicode_Check(kinds) || PyTuple_GET_SIZE(valued) != 4) {
        PyErr_SetString(PyExc_TypeError, "setup() takes the type 'generic', the string 'deferred_kinds' and four 'valued'");
        return NULL;
    }
    const char *written = PyUnicode_AsUTF8(kinds);
    if (written == NULL || strlen(written) >= MOST_KINDS) {
        PyErr_SetString(PyExc_ValueError, "setup(): 'deferred_kinds' is too long");
        return NULL;
    }
    strcpy(deferred_kinds, written);
    Py_XSETREF(Generic, (PyTypeObject *)Py_NewRef(generic));
    PyObject *floor = PyDict_GetItemString(given, "unknown_floor");
    if (floor == NULL) {
        PyErr_SetString(PyExc_KeyError, "setup() was not given 'unknown_floor'");
        return NULL;
    }
    unknown_floor = PyLong_AsLongLong(floor);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (Py_TYPE(numbers)->tp_iternext == NULL) {
        PyErr_SetString(PyExc_TypeError, "setup(): 'numbers' must be an iterator");
        return NULL;
    }
    if (offsets((PyTypeObject *)node, NODE_FIELDS, N_FIELDS, node_at) || offsets((PyTypeObject *)lazy, LAZY_FIELDS, L_FIELDS, lazy_at)
        || offsets((PyTypeObject *)cell, CELL_FIELDS, C_FIELDS, cell_at) || offsets((PyTypeObject *)step, STEP_FIELDS, S_FIELDS, step_at)
        || offsets((PyTypeObject *)write, WRITE_FIELDS, W_FIELDS, write_at) || offsets((PyTypeObject *)assign, ASSIGN_FIELDS, A_FIELDS, assign_at)
        || offsets((PyTypeObject *)standin_call, CALLABLE_FIELDS, F_FIELDS, callable_at)
        || offsets((PyTypeObject *)standin_module, MODULE_FIELDS, M_FIELDS, module_at)) {
        return NULL;
    }
    Node = (PyTypeObject *)node;
    Lazy = (PyTypeObject *)lazy;
    Cell = (PyTypeObject *)cell;
    Step = (PyTypeObject *)step;
    Ufunc = (PyTypeObject *)ufunc;
    Ndarray = (PyTypeObject *)ndarray;
    Write = (PyTypeObject *)write;
    Assign = (PyTypeObject *)assign;
    StandInCall = (PyTypeObject *)standin_call;
    StandInModule = (PyTypeObject *)standin_module;
    PyObject *select = PyDict_GetItemString(given, "Select");
    if (select == NULL || !PyType_Check(select) || offsets((PyTypeObject *)select, SELECT_FIELDS, X_FIELDS, select_at)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_KeyError, "setup() was not given the class 'Select'");
        }
        return NULL;
    }
    Select = (PyTypeObject *)Py_NewRef(select);
    PyObject *into = PyDict_GetItemString(given, "Into");
    static const char *const INTO_FIELDS[1] = {"function"};
    if (into == NULL || !PyType_Check(into) || offsets((PyTypeObject *)into, INTO_FIELDS, 1, &into_function_at)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_KeyError, "setup() was not given the class 'Into'");
        }
        return NULL;
    }
    Into = (PyTypeObject *)Py_NewRef(into);
    PyObject *origin = PyDict_GetItemString(given, "Origin");
    static const char *const ORIGIN_FIELDS[4] = {"filename", "lineno", "module", "registry"};
    if (origin == NULL || !PyType_Check(origin) || offsets((PyTypeObject *)origin, ORIGIN_FIELDS, 4, origin_at)
        || take(given, "internal_files", &PyDict_Type, &internal_files)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_KeyError, "setup() was not given the class 'Origin'");
        }
        return NULL;
    }
    Origin = (PyTypeObject *)Py_NewRef(origin);
    PyObject *signature = NULL;
    if (take(given, "spec_numbers", NULL, &spec_numbers) || take(given, "Signature", &PyType_Type, &signature)) {
        return NULL;
    }
    Signature = (PyTypeObject *)signature;
    if (Py_TYPE(spec_numbers)->tp_iternext == NULL || !PyType_IsSubtype(Signature, &PyTuple_Type)) {
        PyErr_SetString(PyExc_TypeError, "setup(): 'spec_numbers' must be an iterator and 'Signature' a kind of tuple");
        return NULL;
    }
    if (take(given, "views", &PyFrozenSet_Type, &view_functions) || take(given, "reductions", &PyFrozenSet_Type, &reductions)
        || take(given, "positions", &PyFrozenSet_Type, &positions) || take(given, "reduction_kind", &PyUnicode_Type, &reduction_kind)
        || take(given, "ravel", NULL, &numpy_ravel) || take(given, "empty", NULL, &numpy_empty) || take(given, "no_size", NULL, &no_size)
        || take(given, "element_type", NULL, &python_element_type)) {
        return NULL;
    }
    PyObject *few = PyDict_GetItemString(given, "small_whole"), *many = PyDict_GetItemString(given, "large_whole");
    if (few == NULL || many == NULL) {
        PyErr_SetString(PyExc_KeyError, "setup() was not given 'small_whole' and 'large_whole'");
        return NULL;
    }
    small_whole = PyLong_AsLongLong(few);
    large_whole = PyLong_AsLongLong(many);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (take(given, "dtype", NULL, &numpy_dtype) || take(given, "matmul", NULL, &matmul)
        || take(given, "matrix_product", &PyUnicode_Type, &matrix_kind)) {
        return NULL;
    }
    if (take(given, "types", &PyDict_Type, &types) || take(given, "keep", NULL, &python_keep) || take(given, "spec_of", NULL, &python_spec_of)
        || take(given, "elementwise", &PyUnicode_Type, &elementwise_kind)) {
        return NULL;
    }
    PyObject *through = PyDict_GetItemString(given, "AssignThrough");
    if (through == NULL || !PyType_Check(through) || offsets((PyTypeObject *)through, SELECT_FIELDS, X_FIELDS, through_at)
        || take(given, "assignable", &PySet_Type, &assignable)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_KeyError, "setup() was not given the class 'AssignThrough'");
        }
        return NULL;
    }
    Through = (PyTypeObject *)Py_NewRef(through);
    Py_RETURN_NONE;
}

/* `write_now(target, perform, reads)`: make a write at once if `write_now` above allows it, for the road of a
 * write that Python takes. True when it was made. */
static PyObject *
module_write_now(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 3 || Py_TYPE(args[0]) != Lazy || !PyTuple_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "write_now(target, perform, reads)");
        return NULL;
    }
    if (!enabled) {
        Py_RETURN_FALSE;
    }
    PyObject *cell = cell_of(args[0]);
    if (cell == NULL) {
        return NULL;
    }
    PyObject *held = SLOT(cell, cell_at[C_NODE]);
    if (Py_TYPE(held) != Node) {
        Py_RETURN_FALSE;
    }
    int now = write_now(args[0], held, args[1], args[2], NULL);
    if (now < 0) {
        return NULL;
    }
    return PyBool_FromLong(now);
}

/* `at_once_below(limit)`: set `small_limit`, returning what it was. */
static PyObject *
module_at_once_below(PyObject *module, PyObject *limit)
{
    Py_ssize_t wanted = PyLong_AsSsize_t(limit);
    if (wanted == -1 && PyErr_Occurred()) {
        return NULL;
    }
    if (wanted < 0) {
        PyErr_SetString(PyExc_ValueError, "the limit is a count of elements, 0 or more");
        return NULL;
    }
    Py_ssize_t before = small_limit;
    small_limit = wanted;
    if (wanted != 0) {
        views_alone = 1;
    }
    return PyLong_FromSsize_t(before);
}

/* `views_alone(on)`: say whether a view taken at once may be alive; see `writable_as`. `at_once_below` says yes by
 * itself. Saying no is for a caller that knows every lazy array made so far is gone, as a test does. */
static PyObject *
module_views_alone(PyObject *module, PyObject *on)
{
    int now = PyObject_IsTrue(on);
    if (now < 0) {
        return NULL;
    }
    views_alone = now;
    Py_RETURN_NONE;
}

/* `fusible(calls, dtype)`: the callables a kernel can make and the element type it makes them on, or None twice
 * for none; see `fusible_calls`. */
static PyObject *
module_fusible(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 2 || (args[0] != Py_None && !PyFrozenSet_CheckExact(args[0]))) {
        PyErr_SetString(PyExc_TypeError, "fusible(calls, dtype) takes a frozenset and an element type, or None twice");
        return NULL;
    }
    Py_XSETREF(fusible_calls, args[0] == Py_None ? NULL : Py_NewRef(args[0]));
    Py_XSETREF(fusible_dtype, args[0] == Py_None ? NULL : Py_NewRef(args[1]));
    Py_RETURN_NONE;
}

/* `at_once_counts()`: (calls made at once because they were small, writes made at once in place), since the module
 * was loaded. */
static PyObject *
module_at_once_counts(PyObject *module, PyObject *ignored)
{
    return Py_BuildValue("(LL)", made_at_once, written_at_once);
}

/* `small_reads(function)`: the function that computes the node of a lazy array, which `small_pending` calls; None
 * to have a small call wait for a pending operand as any other does. */
static PyObject *
module_small_reads(PyObject *module, PyObject *function)
{
    Py_XSETREF(compute_small, function == Py_None ? NULL : Py_NewRef(function));
    Py_RETURN_NONE;
}

/* `lent_count()`: calls that wrote their result into the memory of an operand nothing else read, since the module
 * was loaded (`lendable`). */
static PyObject *
module_lent_count(PyObject *module, PyObject *ignored)
{
    return PyLong_FromLongLong(lent);
}

/* `release_pending()`: give back the table of the pending set when nothing is pending. A set keeps the table of its
 * largest moment, and reading it walks the whole table (`graph.release_pending`). Finding the set empty and
 * clearing it are one step here, with no other thread of the interpreter between them: in Python they are two, and
 * a node written between them would be taken out of the set and never computed before a write. True when the set
 * was empty. */
static PyObject *
module_release_pending(PyObject *module, PyObject *ignored)
{
    if (PySet_GET_SIZE(pending) != 0) {
        Py_RETURN_FALSE;
    }
    if (PySet_Clear(pending) < 0) {
        return NULL;
    }
    Py_RETURN_TRUE;
}

/* The most operands a node is computed with on the C stack; one of more is left to Python. */
#define MOST_COMPUTED 8

/* The fewest bytes an operand must have for its memory to be lent to the call that reads it last: NumPy's own
 * threshold for reusing a temporary (`NPY_MIN_ELIDE_BYTES`, 256 KiB). `evaluate.LEND_BYTES` is the same. */
#define LEND_BYTES (256 * 1024)


/* Whether an array operand leaves the layout of a ufunc's result as NumPy gives it out of place: it has no
 * dimensions, or its elements lie in C order. With every array operand so, NumPy's result is in C order, as the
 * operand lent is. */
static int
lays_in_order(PyObject *value)
{
    return Py_TYPE(value) != Ndarray || PyArray_NDIM((PyArrayObject *)value) == 0
        || PyArray_IS_C_CONTIGUOUS((PyArrayObject *)value);
}

/* The operand of a plain node whose memory its call may write its result into, as NumPy reuses a temporary's;
 * `evaluate._lendable` is the definition, and this says the same from the counts the core sees. A borrowed
 * reference, or NULL for none. The node's call is a ufunc of one output with an operand for each input and no
 * keyword, element by element; every array operand lies in C order; and the operand lent is a computed node that
 * no lazy array the user can name stands for (`exposed`), that nothing but this call reads (the only references to
 * it are the places of this node's operands, since a pending node holds its operands and a lazy array its node),
 * whose value is an `ndarray` that owns its memory, can be written, has the result's shape and element type and at
 * least `LEND_BYTES`, and that nothing but the node refers to: no view of it is alive. */
static PyObject *
lendable(PyObject *node, PyObject *operation, PyObject *args, Py_ssize_t count)
{
    PyObject *kind = SLOT(node, node_at[N_KIND]);
    if (Py_TYPE(operation) != Ufunc || ((PyUFuncObject *)operation)->nout != 1 || ((PyUFuncObject *)operation)->nin != count
        || (kind != elementwise_kind && PyUnicode_Compare(kind, elementwise_kind) != 0)) {
        return NULL;
    }
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        if (!lays_in_order(Py_TYPE(arg) == Node ? SLOT(arg, node_at[N_VALUE]) : arg)) {
            return NULL;
        }
    }
    PyObject *shape = SLOT(node, node_at[N_SHAPE]), *dtype = SLOT(node, node_at[N_DTYPE]);
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        if (Py_TYPE(arg) != Node || SLOT(arg, node_at[N_EXPOSED]) != Py_False) {
            continue;
        }
        PyObject *value = SLOT(arg, node_at[N_VALUE]);
        if (Py_TYPE(value) != Ndarray || Py_REFCNT(value) != 1) {
            continue;
        }
        PyArrayObject *memory = (PyArrayObject *)value;
        Py_ssize_t places = 0;
        for (Py_ssize_t other = 0; other < count; other++) {
            places += PyTuple_GET_ITEM(args, other) == arg;
        }
        if (Py_REFCNT(arg) != places || PyArray_BASE(memory) != NULL || !PyArray_CHKFLAGS(memory, NPY_ARRAY_OWNDATA)
            || !PyArray_ISWRITEABLE(memory) || PyArray_NBYTES(memory) < LEND_BYTES
            || PyTuple_GET_SIZE(shape) != PyArray_NDIM(memory)) {
            continue;
        }
        int fits = 1;
        for (int axis = 0; fits && axis < PyArray_NDIM(memory); axis++) {
            PyObject *length = PyTuple_GET_ITEM(shape, axis);
            fits = PyLong_CheckExact(length) && PyLong_AsSsize_t(length) == PyArray_DIM(memory, axis);
        }
        if (!fits || ((PyObject *)PyArray_DESCR(memory) != dtype
                      && PyObject_RichCompareBool((PyObject *)PyArray_DESCR(memory), dtype, Py_EQ) != 1)) {
            PyErr_Clear();
            continue;
        }
        return arg;
    }
    return NULL;
}

/* Compute one pending node whose call is plain, as `evaluate._compute_one` does: the operation on its operands'
 * values, the value kept, the operands let go (`graph.settle`). A plain call is one that is no write, has no
 * keyword, gives a shape that is known, and reads only values that are there; the caller sees to the error
 * settings and to what is caught. 1 when the node was computed; 0, with nothing done, for any other node, which
 * Python computes; -1 with NumPy's error set, the node untouched. */
static int
plain_now(PyObject *node)
{
    if (Py_TYPE(node) != Node) {
        return 0;
    }
    PyObject *operation = SLOT(node, node_at[N_OPERATION]), *args = SLOT(node, node_at[N_ARGS]);
    PyObject *kwargs = SLOT(node, node_at[N_KWARGS]), *kind = SLOT(node, node_at[N_KIND]);
    if (operation == Py_None || !PyTuple_CheckExact(args) || PyTuple_GET_SIZE(args) > MOST_COMPUTED || !PyDict_CheckExact(kwargs)
        || PyDict_GET_SIZE(kwargs) != 0 || kind == write_kind || !PyUnicode_CheckExact(kind)
        || PyUnicode_Compare(kind, write_kind) == 0 || !PyTuple_CheckExact(SLOT(node, node_at[N_SHAPE]))
        || !shape_known(SLOT(node, node_at[N_SHAPE]))) {
        return 0;
    }
    PyObject *values[MOST_COMPUTED + 1];
    Py_ssize_t count = PyTuple_GET_SIZE(args);
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        if (Py_TYPE(arg) == Node && !has_value(arg)) {
            return 0; /* Failed, or its memory taken by a write: Python says what follows. */
        }
    }
    /* Asked before the values are taken, which are references to them. */
    PyObject *lender = lendable(node, operation, args, count);
    for (Py_ssize_t place = 0; place < count; place++) {
        PyObject *arg = PyTuple_GET_ITEM(args, place);
        values[place] = Py_NewRef(Py_TYPE(arg) == Node ? SLOT(arg, node_at[N_VALUE]) : arg);
    }
    if (lender != NULL) {
        values[count] = Py_NewRef(SLOT(lender, node_at[N_VALUE]));
    }
    Py_INCREF(operation);
    PyObject *value = PyObject_Vectorcall(operation, values, (size_t)count, lender != NULL ? out_names : NULL);
    Py_DECREF(operation);
    for (Py_ssize_t place = 0; place < count + (lender != NULL); place++) {
        Py_DECREF(values[place]);
    }
    if (value == NULL) {
        return -1;
    }
    if (lender != NULL) {
        /* The lender's memory is the node's now: as a write made in place does, the lender keeps no value. */
        Py_SETREF(SLOT(lender, node_at[N_VALUE]), Py_NewRef(Py_None));
        lent++;
    }
    /* `node.value = value`, then `graph.settle(node)`. */
    Py_SETREF(SLOT(node, node_at[N_VALUE]), value);
    Py_SETREF(SLOT(node, node_at[N_OPERATION]), Py_NewRef(Py_None));
    Py_SETREF(SLOT(node, node_at[N_ARGS]), Py_NewRef(empty_tuple));
    Py_SETREF(SLOT(node, node_at[N_KWARGS]), Py_NewRef(no_keywords));
    PyObject *reference = PyWeakref_NewRef(node, NULL);
    if (reference == NULL || PySet_Discard(pending, reference) < 0) {
        Py_XDECREF(reference);
        return -1;
    }
    Py_DECREF(reference);
    return 1;
}

/* `compute_plain(node)`: `plain_now` for `evaluate._compute_one`. True when the node was computed, False when
 * it is Python's to compute, NULL with NumPy's error. */
static PyObject *
module_compute_plain(PyObject *module, PyObject *node)
{
    int done = enabled ? plain_now(node) : 0;
    if (done < 0) {
        return NULL;
    }
    return PyBool_FromLong(done);
}

/* `run_nodes(todo, inner, groups, quiet, numbers, slow, failed, pending_only)`: the loop of `evaluate.compute` and
 * of `evaluate.compute_pending`, which takes the nodes of one computation in the order they were written.
 *
 * Two kinds of node are taken here, and they are most of a computation's: a node inside a planned group, which is
 * only noted, and a node whose call is plain and written under the error settings in effect, which `plain_now`
 * computes. Every other node goes to `slow`, the Python that takes one node of that loop: a group's own node, a
 * write, a node with keywords or other error settings, one whose operand failed. So the Python loop is the
 * definition, and this is its quick road node by node.
 *
 * `todo` is the list, newest first, which is emptied from its end; `inner` and `groups` are the plan's, or None;
 * `quiet` is the computation's `evaluate._Quiet`; `numbers` gains the number of each node taken here; `failed` is
 * called with a node and the error its plain call raised, to do what `_compute_one` and `_step` do with one;
 * `pending_only` is true for `compute_pending`, which passes over a node that has been computed meanwhile. */
static PyObject *
module_run_nodes(PyObject *module, PyObject *const *args, Py_ssize_t count)
{
    if (count != 8 || !PyList_CheckExact(args[0]) || !PyList_CheckExact(args[4])
        || (args[1] != Py_None && !PyAnySet_Check(args[1])) || (args[2] != Py_None && !PyDict_Check(args[2]))) {
        PyErr_SetString(PyExc_TypeError, "run_nodes(todo, inner, groups, quiet, numbers, slow, failed, pending_only)");
        return NULL;
    }
    PyObject *todo = args[0], *inner = args[1], *groups = args[2], *quiet = args[3], *numbers = args[4];
    PyObject *slow = args[5], *failed = args[6];
    int pending_only = PyObject_IsTrue(args[7]);
    PyObject *active = PyObject_GetAttr(quiet, s_active), *caught = PyObject_GetAttr(quiet, s_caught);
    PyObject *node = NULL;
    if (active == NULL || caught == NULL || pending_only < 0) {
        goto fail;
    }
    while (PyList_GET_SIZE(todo) > 0) {
        /* `todo.pop()`: the list lets go of the node, so an intermediate is freed with its last reader. */
        Py_ssize_t last = PyList_GET_SIZE(todo) - 1;
        node = Py_NewRef(PyList_GET_ITEM(todo, last));
        if (PyList_SetSlice(todo, last, last + 1, NULL) < 0) {
            goto fail;
        }
        int python = 1, look = 1;
        if (Py_TYPE(node) == Node) {
            PyObject *number = SLOT(node, node_at[N_NUMBER]);
            if (pending_only && SLOT(node, node_at[N_OPERATION]) == Py_None) {
                Py_CLEAR(node);
                continue;
            }
            /* `compute_pending` first fails a node one of whose operands failed: Python's to do. */
            int inherited = 0;
            PyObject *read = SLOT(node, node_at[N_ARGS]);
            for (Py_ssize_t place = 0; pending_only && PyTuple_CheckExact(read) && place < PyTuple_GET_SIZE(read); place++) {
                PyObject *arg = PyTuple_GET_ITEM(read, place);
                inherited = inherited || (Py_TYPE(arg) == Node && SLOT(arg, node_at[N_FAILURE]) != Py_None);
            }
            int inside = !inherited && inner != Py_None ? PySet_Contains(inner, number) : 0;
            int grouped = !inherited && inside == 0 && groups != Py_None ? PyDict_Contains(groups, number) : 0;
            if (inside < 0 || grouped < 0) {
                goto fail;
            }
            if (inside) {
                /* Inside a group: its group's own node gives what it would have. */
                if (PyList_Append(numbers, number) < 0) {
                    goto fail;
                }
                python = look = 0;
            } else if (!inherited && !grouped && enabled && SLOT(node, node_at[N_ERRSTATE]) == active) {
                PyObject *origin = Py_NewRef(SLOT(node, node_at[N_ORIGIN]));
                Py_ssize_t noted = PyList_GET_SIZE(numbers);
                int done = PyList_Append(numbers, number) < 0 ? -2 : plain_now(node);
                if (done == 0) {
                    /* Python's to compute, and to note. */
                    if (PyList_SetSlice(numbers, noted, PyList_GET_SIZE(numbers), NULL) < 0) {
                        Py_DECREF(origin);
                        goto fail;
                    }
                } else if (done == 1) {
                    python = 0;
                    look = 0;
                    if (PyList_CheckExact(caught) && PyList_GET_SIZE(caught) != 0) {
                        PyObject *heard = PyObject_CallMethodOneArg(quiet, s_note, origin);
                        if (heard == NULL) {
                            Py_DECREF(origin);
                            goto fail;
                        }
                        Py_DECREF(heard);
                        look = 1;
                    }
                } else {
                    Py_DECREF(origin);
                    if (done == -1 && PyErr_ExceptionMatches(PyExc_Exception)) {
                        /* What `_compute_one` and `_step` do with an error: a note of the line, the node failed. */
                        PyObject *kind, *error, *trace;
                        PyErr_Fetch(&kind, &error, &trace);
                        PyErr_NormalizeException(&kind, &error, &trace);
                        if (trace != NULL) {
                            PyException_SetTraceback(error, trace);
                        }
                        PyObject *told = PyObject_CallFunctionObjArgs(failed, node, error, NULL);
                        if (told == NULL) {
                            Py_XDECREF(kind);
                            Py_XDECREF(error);
                            Py_XDECREF(trace);
                            goto fail;
                        }
                        Py_DECREF(told);
                        PyErr_Restore(kind, error, trace);
                    }
                    goto fail;
                }
                Py_DECREF(origin);
            }
        }
        if (python) {
            PyObject *taken = PyObject_CallOneArg(slow, node);
            if (taken == NULL) {
                goto fail;
            }
            Py_DECREF(taken);
            /* Python may have put other error settings, and issued warnings, which starts another catching. */
            Py_SETREF(active, PyObject_GetAttr(quiet, s_active));
            Py_SETREF(caught, PyObject_GetAttr(quiet, s_caught));
            if (active == NULL || caught == NULL) {
                goto fail;
            }
        }
        Py_CLEAR(node);
        if (look) {
            /* `if quiet.found: quiet.issue()`. */
            PyObject *found = PyObject_GetAttr(quiet, s_found);
            int any = found ? PyObject_IsTrue(found) : -1;
            Py_XDECREF(found);
            if (any < 0) {
                goto fail;
            }
            if (any) {
                PyObject *issued = PyObject_CallMethodNoArgs(quiet, s_issue);
                if (issued == NULL) {
                    goto fail;
                }
                Py_DECREF(issued);
                Py_SETREF(caught, PyObject_GetAttr(quiet, s_caught));
                if (caught == NULL) {
                    goto fail;
                }
            }
        }
    }
    Py_DECREF(active);
    Py_DECREF(caught);
    Py_RETURN_NONE;
fail:
    Py_XDECREF(node);
    Py_XDECREF(active);
    Py_XDECREF(caught);
    return NULL;
}

/* One pending node and its number, for `needed`. */
typedef struct {
    long long number;
    PyObject *node;
} Wanted;

static int
newest_first(const void *first, const void *second)
{
    long long one = ((const Wanted *)first)->number, other = ((const Wanted *)second)->number;
    return (one < other) - (one > other);
}

/* `needed(roots)`: `evaluate._needed`. The pending nodes some roots depend on, themselves included, in descending
 * number, in a new list that is the only thing here to refer to them. Found with a stack, not by recursion. */
static PyObject *
module_needed(PyObject *module, PyObject *roots)
{
    if (!PyList_CheckExact(roots)) {
        PyErr_SetString(PyExc_TypeError, "needed(roots) takes a list of nodes");
        return NULL;
    }
    Py_ssize_t room = 64, depth = 0, count = 0, kept = 64;
    PyObject **stack = PyMem_Malloc((size_t)room * sizeof(PyObject *));
    Wanted *found = PyMem_Malloc((size_t)kept * sizeof(Wanted));
    PyObject *seen = PySet_New(NULL), *result = NULL;
    if (stack == NULL || found == NULL || seen == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    for (Py_ssize_t place = 0; place < PyList_GET_SIZE(roots); place++) {
        PyObject *root = PyList_GET_ITEM(roots, place);
        if (Py_TYPE(root) != Node) {
            PyErr_SetString(PyExc_TypeError, "needed(roots) takes a list of nodes");
            goto done;
        }
        if (SLOT(root, node_at[N_OPERATION]) == Py_None) {
            continue;
        }
        if (depth == room) {
            room *= 2;
            PyObject **grown = PyMem_Realloc(stack, (size_t)room * sizeof(PyObject *));
            if (grown == NULL) {
                PyErr_NoMemory();
                goto done;
            }
            stack = grown;
        }
        stack[depth++] = root;
    }
    while (depth > 0) {
        PyObject *node = stack[--depth];
        PyObject *number = SLOT(node, node_at[N_NUMBER]);
        int known = PySet_Contains(seen, number);
        if (known < 0) {
            goto done;
        }
        if (known) {
            continue;
        }
        if (PySet_Add(seen, number) < 0) {
            goto done;
        }
        if (count == kept) {
            kept *= 2;
            Wanted *grown = PyMem_Realloc(found, (size_t)kept * sizeof(Wanted));
            if (grown == NULL) {
                PyErr_NoMemory();
                goto done;
            }
            found = grown;
        }
        found[count].node = node;
        found[count].number = PyLong_AsLongLong(number);
        count++;
        PyObject *read = SLOT(node, node_at[N_ARGS]);
        if (!PyTuple_CheckExact(read)) {
            continue;
        }
        for (Py_ssize_t place = 0; place < PyTuple_GET_SIZE(read); place++) {
            PyObject *arg = PyTuple_GET_ITEM(read, place);
            if (Py_TYPE(arg) != Node || SLOT(arg, node_at[N_OPERATION]) == Py_None) {
                continue;
            }
            if (depth == room) {
                room *= 2;
                PyObject **grown = PyMem_Realloc(stack, (size_t)room * sizeof(PyObject *));
                if (grown == NULL) {
                    PyErr_NoMemory();
                    goto done;
                }
                stack = grown;
            }
            stack[depth++] = arg;
        }
    }
    if (PyErr_Occurred()) {
        goto done;
    }
    qsort(found, (size_t)count, sizeof(Wanted), newest_first);
    result = PyList_New(count);
    for (Py_ssize_t place = 0; result != NULL && place < count; place++) {
        PyList_SET_ITEM(result, place, Py_NewRef(found[place].node));
    }
done:
    PyMem_Free(stack);
    PyMem_Free(found);
    Py_XDECREF(seen);
    return result;
}

/* `track(lazy)`: make a lazy array known to the garbage collector, for one that held its value alone and was given
 * its cell by the Python property. */
static PyObject *
module_track(PyObject *module, PyObject *lazy)
{
    if (PyType_IS_GC(Py_TYPE(lazy)) && !PyObject_GC_IsTracked(lazy)) {
        PyObject_GC_Track(lazy);
    }
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"find", (PyCFunction)(void (*)(void))module_find, METH_FASTCALL,
     "fuse.find, in C: (the expressions of a computation's nodes, the numbers of the nodes inside them)."},
    {"find_setup", (PyCFunction)(void (*)(void))module_find_setup, METH_VARARGS | METH_KEYWORDS,
     "Be told the classes, tables and names of fuse.py, for find()."},
    {"views_alone", module_views_alone, METH_O,
     "Say whether a view taken at once, which the layer keeps no record of, may be alive."},
    {"quick_slots", module_quick_slots, METH_NOARGS,
     "Stand in the slots of LazyArray for the operators that carry a road, once the roads are on the class."},
    {"run_nodes", (PyCFunction)(void (*)(void))module_run_nodes, METH_FASTCALL,
     "The loop of evaluate.compute: take a computation's nodes in order, computing the plain ones here."},
    {"needed", module_needed, METH_O, "evaluate._needed: the pending nodes some roots depend on, newest first."},
    {"compute_plain", module_compute_plain, METH_O,
     "Compute one pending node whose call is plain, as evaluate._compute_one does. True when it was computed."},
    {"track", module_track, METH_O, "Make a lazy array that has just been given its cell known to the garbage collector."},
    {"at_once_below", module_at_once_below, METH_O,
     "Have a call that gives fewer elements than this, on operands that have their values, made at once. Returns the limit it replaces."},
    {"write_now", (PyCFunction)(void (*)(void))module_write_now, METH_FASTCALL,
     "Make a write at once, in place, when nothing is pending and at_once_below is set. True when it was made."},
    {"fusible", (PyCFunction)(void (*)(void))module_fusible, METH_FASTCALL,
     "Name the callables a kernel can make and their element type: a large call that is none of them is made at once."},
    {"at_once_counts", module_at_once_counts, METH_NOARGS, "(calls, writes) made at once since the module was loaded."},
    {"lent_count", module_lent_count, METH_NOARGS, "Calls that wrote into an operand's memory since the module was loaded."},
    {"small_reads", module_small_reads, METH_O,
     "Be told the function that computes a lazy array's node, for a small call that reads a pending one; None for none."},
    {"release_pending", module_release_pending, METH_NOARGS,
     "Give back the table of the pending set when nothing is pending. True when the set was empty."},
    {"setup", (PyCFunction)(void (*)(void))module_setup, METH_VARARGS | METH_KEYWORDS,
     "Be told the layer's classes, memories and constants."},
    {"enable", module_enable, METH_O, "Take the quick roads (True) or hand every call to Python (False). Returns what it was."},
    {"deferred", (PyCFunction)(void (*)(void))module_deferred, METH_FASTCALL, "graph.deferred, in C."},
    {"plan", (PyCFunction)(void (*)(void))module_plan, METH_FASTCALL,
     "The remembered answer of rules.plan as (shape, dtype, kind, spec), or MISS."},
    {"try_defer", (PyCFunction)(void (*)(void))module_try_defer, METH_FASTCALL,
     "The quick road of array._try_defer: a lazy array, or MISS when Python has to look."},
    {NULL},
};

static struct PyModuleDef definition = {
    PyModuleDef_HEAD_INIT, "prototypes.lazy._core",
    "The lazy layer's hot path: writing one node of the graph when every answer it needs is remembered.", -1, methods,
};

PyMODINIT_FUNC
PyInit__core(void)
{
    if (PyType_Ready(&RoadType) < 0) {
        return NULL;
    }
    import_array();
    PyObject *module = PyModule_Create(&definition);
    if (module == NULL) {
        return NULL;
    }
    Miss = PyObject_CallNoArgs((PyObject *)&PyBaseObject_Type);
    zero = PyLong_FromLong(0);
    one = PyLong_FromLong(1);
    empty_tuple = PyTuple_New(0);
    s_node = PyUnicode_InternFromString("_node");
    s_made = PyUnicode_InternFromString("made");
    s_deferred = PyUnicode_InternFromString("deferred");
    s_observer = PyUnicode_InternFromString("observer");
    s_limit = PyUnicode_InternFromString("PENDING_LIMIT");
    s_settings = PyUnicode_InternFromString("_error_settings");
    s_last = PyUnicode_InternFromString("_last_errstate");
    s_registry = PyUnicode_InternFromString("__warningregistry__");
    s_name = PyUnicode_InternFromString("__name__");
    s_dunder_name = s_name;
    s_unknown = PyUnicode_InternFromString("<unknown>");
    s_shape = PyUnicode_InternFromString("shape");
    s_dtype = PyUnicode_InternFromString("dtype");
    s_doubt = PyUnicode_InternFromString("doubt");
    s_call = PyUnicode_InternFromString("__call__");
    s_getitem = PyUnicode_InternFromString("getitem");
    s_basic_shape = PyUnicode_InternFromString("basic_shape");
    s_out = PyUnicode_InternFromString("out");
    s_kind = PyUnicode_InternFromString("kind");
    s_b = PyUnicode_InternFromString("b");
    s_setitem = PyUnicode_InternFromString("setitem");
    s_plan = PyUnicode_InternFromString("first");
    s_resolve = PyUnicode_InternFromString("resolve_dtypes");
    s_spec_limit = PyUnicode_InternFromString("SPEC_MEMORY_LIMIT");
    s_plan_limit = PyUnicode_InternFromString("PLAN_MEMORY_LIMIT");
    s_type_limit = PyUnicode_InternFromString("TYPE_MEMORY_LIMIT");
    s_nin = PyUnicode_InternFromString("nin");
    s_nout = PyUnicode_InternFromString("nout");
    s_signature = PyUnicode_InternFromString("signature");
    s_type = PyUnicode_InternFromString("type");
    own_dtypes = PyDict_New();
    s_origin_limit = PyUnicode_InternFromString("ORIGIN_MEMORY_LIMIT");
    s_array = PyUnicode_InternFromString("array");
    no_keywords = PyDict_New();
    empty_string = PyUnicode_InternFromString("");
    s_transpose = PyUnicode_InternFromString("transpose");
    s_isnative = PyUnicode_InternFromString("isnative");
    s_active = PyUnicode_InternFromString("active");
    s_caught = PyUnicode_InternFromString("caught");
    s_found = PyUnicode_InternFromString("found");
    s_note = PyUnicode_InternFromString("note");
    s_issue = PyUnicode_InternFromString("issue");
    s_fallback = PyUnicode_InternFromString("fallback");
    out_names = s_out ? PyTuple_Pack(1, s_out) : NULL;
    if (!s_fallback || !out_names) {
        Py_DECREF(module);
        return NULL;
    }
    if (!Miss || !zero || !one || !empty_tuple || !s_node || !s_made || !s_deferred || !s_observer || !s_limit || !s_settings
        || !s_last || !s_registry || !s_name || !s_unknown || !s_shape || !s_dtype || !s_doubt || !s_call || !s_getitem
        || !s_basic_shape || !s_out || !s_kind || !s_b || !s_setitem || !s_plan || !s_origin_limit || !s_resolve || !own_dtypes || !s_spec_limit || !s_plan_limit
        || !s_type_limit || !s_nin || !s_nout || !s_signature || !s_type || !s_array || !no_keywords || !empty_string || !s_transpose || !s_isnative || !s_active || !s_caught || !s_found
        || !s_note || !s_issue) {
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddObjectRef(module, "MISS", Miss) < 0 || PyModule_AddObjectRef(module, "Road", (PyObject *)&RoadType) < 0
        || PyModule_AddIntConstant(module, "VERSION", CORE_VERSION) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
