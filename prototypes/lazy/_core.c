/* The lazy layer's hot path in C: writing one node of the graph when every answer it needs is remembered.
 *
 * This file adds nothing to what the layer does. Each function here is the
 * quick road of a function in array.py, graph.py or rules.py, for the calls
 * where everything that function would look up is already in the layer's
 * memories: the answer of `rules.plan`, the line of the user's code, the
 * error settings. It reads those memories and never writes them. Wherever it
 * is not sure, it hands the call to the Python function, unchanged, which
 * does the work, fills the memory, and so makes the next such call one this
 * file can take. The Python functions are the definition; a node made here
 * is, field for field, the node they would have made.
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
 * It is built by `python -m prototypes.lazy.build_core`, and the layer runs
 * without it.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <structmember.h> /* `PyMemberDef`, which Python.h itself brings only from 3.12 on. */

/* Raised by one whenever this file and the Python it stands for change together, so that a module built from an
 * older source is not used: array.py asks for the number it was written against. */
#define CORE_VERSION 11

/* A field of an instance of a class with `__slots__`, by its offset. */
#define SLOT(object, offset) (*(PyObject **)((char *)(object) + (offset)))

/* The fields of `graph.Node`, in the order its dataclass declares them. */
enum { N_NUMBER, N_NAME, N_OPERATION, N_ARGS, N_KWARGS, N_SHAPE, N_DTYPE, N_ORIGIN, N_ERRSTATE, N_VALUE, N_FAILURE,
       N_KIND, N_STRICT, N_EXPOSED, N_SPEC, N_FIELDS };
static const char *const NODE_FIELDS[N_FIELDS] = {
    "number", "name", "operation", "args", "kwargs", "shape", "dtype", "origin", "errstate", "value", "failure",
    "kind", "strict", "exposed", "spec"};
/* The fields of `array.LazyArray`. */
enum { L_CELL, L_STEPS, L_SEEN, L_FIELDS };
static const char *const LAZY_FIELDS[L_FIELDS] = {"_cell", "_steps", "_seen"};
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
/* Returned by a quick road that does not apply: the caller then takes the Python road. Never seen from Python
 * except as the result of `try_defer`. */
static PyObject *Miss;
/* Constants made once. */
static PyObject *zero, *one, *empty_tuple;
static PyObject *s_node, *s_made, *s_deferred, *s_observer, *s_limit, *s_settings, *s_last, *s_registry, *s_name,
    *s_unknown, *s_shape, *s_dtype, *s_doubt, *s_call, *s_getitem, *s_basic_shape, *s_out, *s_dunder_name, *s_kind, *s_b,
    *s_setitem;

/* ---- Making the layer's objects. ---- */

/* `array.LazyArray(node)`: an array of its own, on a new cell. As `LazyArray.__init__` with no cell given, and
 * `_Cell.__init__` with its defaults. Steals nothing; returns a new reference. */
static PyObject *
new_lazy(PyObject *node)
{
    PyObject *cell = Cell->tp_alloc(Cell, 0);
    if (cell == NULL) {
        return NULL;
    }
    SLOT(cell, cell_at[C_NODE]) = Py_NewRef(node);
    SLOT(cell, cell_at[C_DOUBTFUL]) = Py_NewRef(Py_None);
    SLOT(cell, cell_at[C_DOUBTED]) = Py_NewRef(Py_False);
    PyObject *lazy = Lazy->tp_alloc(Lazy, 0);
    if (lazy == NULL) {
        Py_DECREF(cell);
        return NULL;
    }
    SLOT(lazy, lazy_at[L_CELL]) = cell;
    SLOT(lazy, lazy_at[L_STEPS]) = Py_NewRef(empty_tuple);
    SLOT(lazy, lazy_at[L_SEEN]) = Py_NewRef(Py_None);
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
    PyObject *lazy = Lazy->tp_alloc(Lazy, 0);
    if (lazy == NULL) {
        Py_DECREF(seen);
        return NULL;
    }
    SLOT(lazy, lazy_at[L_CELL]) = Py_NewRef(cell);
    SLOT(lazy, lazy_at[L_STEPS]) = Py_NewRef(steps);
    SLOT(lazy, lazy_at[L_SEEN]) = seen;
    return lazy;
}

/* `array._Step(...)`, with every field given. */
static PyObject *
new_step(PyObject *operation, PyObject *name, PyObject *tail, PyObject *kwargs, PyObject *shape, PyObject *dtype,
         PyObject *origin, int plain)
{
    PyObject *step = Step->tp_alloc(Step, 0);
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
    return step;
}

/* The node a lazy array stands for now: `LazyArray._node`. For an array of its own that is its cell's node, read
 * here. For a view it is the property, which may make the view's nodes again. Returns a new reference. */
static PyObject *
node_of(PyObject *lazy)
{
    PyObject *steps = SLOT(lazy, lazy_at[L_STEPS]);
    PyObject *cell = SLOT(lazy, lazy_at[L_CELL]);
    if (steps != NULL && cell != NULL && PyTuple_CheckExact(steps) && PyTuple_GET_SIZE(steps) == 0) {
        PyObject *node = SLOT(cell, cell_at[C_NODE]);
        if (node != NULL) {
            return Py_NewRef(node);
        }
    }
    return PyObject_GetAttr(lazy, s_node);
}

/* ---- What `graph.deferred` looks up. ---- */

/* The line of the user's code a frame is at: `graph.caller(frame)`. Found in the memory when the frame is one of
 * the user's and its module still has the name and the record of warnings it had, as `graph._remembered` checks;
 * otherwise the Python function is asked, which walks the stack and remembers. Returns a new reference. */
static PyObject *
origin_of(PyFrameObject *frame)
{
    PyCodeObject *code = PyFrame_GetCode(frame);
    PyObject *identity = PyLong_FromVoidPtr(code); /* What `id(code)` gives. */
    Py_DECREF(code);                              /* The frame keeps it alive. */
    PyObject *offset = PyLong_FromLong(PyFrame_GetLasti(frame));
    PyObject *key = (identity && offset) ? PyTuple_Pack(2, identity, offset) : NULL;
    Py_XDECREF(identity);
    Py_XDECREF(offset);
    if (key == NULL) {
        return NULL;
    }
    PyObject *found = PyDict_GetItemWithError(origins, key); /* (code, origin, registry, module), borrowed. */
    Py_DECREF(key);
    if (found == NULL && PyErr_Occurred()) {
        return NULL;
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
            return NULL;
        }
        if (same) {
            return Py_NewRef(PyTuple_GET_ITEM(found, 1));
        }
    }
    return PyObject_CallOneArg(python_caller, (PyObject *)frame);
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

/* `graph.deferred(...)`: make a pending node and add it to the pending set, in the order the Python function
 * does: the number, the line, the error settings, the node, the set, the counter, the observer. `origin` may be
 * NULL or None for the line of `frame`. Returns a new reference. */
static PyObject *
deferred(PyObject *operation, PyObject *name, PyObject *args, PyObject *kwargs, PyObject *shape, PyObject *dtype,
         PyObject *kind, PyObject *strict, PyObject *origin, PyObject *spec, PyFrameObject *frame)
{
    PyObject *number = NULL, *line = NULL, *settings = NULL, *node = NULL, *reference = NULL;
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
    node = Node->tp_alloc(Node, 0);
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
    if (found == NULL) {
        return PyErr_Occurred() ? NULL : Miss; /* A shape and type not met before: Python gives them a number. */
    }
    if (!unknown) {
        /* A node that was not asked for its number yet, as a new leaf is: it gets the one its shape and type
         * have, as `graph.spec` would give it. The number is a fact about the node, not an answer. */
        Py_SETREF(SLOT(node, node_at[N_SPEC]), Py_NewRef(PyTuple_GET_ITEM(found, 0)));
    }
    return Py_NewRef(PyTuple_GET_ITEM(found, 0));
}

/* What `rules._array_part` writes for an exact `ndarray`: the number of its shape and `dtype` object, negated. */
static PyObject *
ndarray_part(PyObject *array)
{
    PyObject *shape = PyObject_GetAttr(array, s_shape);
    PyObject *dtype = shape ? PyObject_GetAttr(array, s_dtype) : NULL;
    PyObject *identity = dtype ? PyLong_FromVoidPtr(dtype) : NULL;
    PyObject *key = identity ? PyTuple_Pack(2, shape, identity) : NULL;
    Py_XDECREF(shape);
    Py_XDECREF(dtype); /* The array keeps it alive, and so does the memory if it is there. */
    Py_XDECREF(identity);
    if (key == NULL) {
        return NULL;
    }
    PyObject *found = PyDict_GetItemWithError(specs, key); /* (number, dtype), borrowed. */
    Py_DECREF(key);
    if (found == NULL) {
        return PyErr_Occurred() ? NULL : Miss;
    }
    return PyNumber_Negative(PyTuple_GET_ITEM(found, 0));
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

/* The remembered answer of `rules.plan` for a call: a new tuple (shape, dtype, kind, spec), as `rules.plan`
 * returns a signature. First by the plain key; then, as `rules._plan_marked`, by the key with placeholders, with
 * the call's own stand-ins put back into the shape. `Miss` (borrowed) when nothing is remembered. */
static PyObject *
remembered_plan(PyObject *rule, PyObject *unwrapped, PyObject *kwargs)
{
    PyObject *key = key_of(rule, unwrapped, kwargs, NULL);
    if (key == NULL) {
        return NULL;
    }
    if (key != Miss) {
        PyObject *kept = PyDict_GetItemWithError(plans, key);
        Py_DECREF(key);
        return kept ? Py_NewRef(kept) : PyErr_Occurred() ? NULL : Miss;
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
    PyObject *signature = remembered_plan(ruled_as ? ruled_as : operation, unwrapped, kwargs); /* (shape, dtype, kind, spec) */
    if (signature == NULL) {
        goto fail;
    }
    if (signature == Miss) {
        goto done;
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
        result = new_lazy(node);
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
            result = all ? new_view(node, SLOT(base, lazy_at[L_CELL]), all) : NULL;
            Py_XDECREF(tail);
            Py_XDECREF(step);
            Py_XDECREF(steps);
            Py_XDECREF(all);
        } else {
            /* `np.reshape` and `np.ravel`: a view or a copy, which is not known yet. */
            result = new_lazy(node);
            if (result != NULL) {
                PyObject *made_cell = SLOT(result, lazy_at[L_CELL]);
                Py_SETREF(SLOT(made_cell, cell_at[C_DOUBTED]), Py_NewRef(Py_True));
                PyObject *noted = PyObject_CallMethodOneArg(SLOT(base, lazy_at[L_CELL]), s_doubt, result);
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
        result = all ? new_view(made, SLOT(target, lazy_at[L_CELL]), all) : NULL;
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
    signature = remembered_plan(getitem, unwrapped, kwargs);
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
        PyObject *made_cell = SLOT(result, lazy_at[L_CELL]);
        Py_SETREF(SLOT(made_cell, cell_at[C_DOUBTED]), Py_NewRef(Py_True));
        PyObject *noted = PyObject_CallMethodOneArg(SLOT(target, lazy_at[L_CELL]), s_doubt, result);
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

/* Whether a write into a lazy array can be a node, as far as this file can tell: `array._may_defer_write`, with
 * anything it would have to work out left to Python. Returns the cell's node (borrowed), or NULL for "ask Python". */
static PyObject *
writable(PyObject *target)
{
    PyObject *cell = SLOT(target, lazy_at[L_CELL]);
    PyObject *held = SLOT(cell, cell_at[C_NODE]);
    PyObject *doubtful = SLOT(cell, cell_at[C_DOUBTFUL]);
    if (Py_TYPE(held) != Node || SLOT(held, node_at[N_EXPOSED]) != Py_False || SLOT(held, node_at[N_FAILURE]) != Py_None
        || SLOT(cell, cell_at[C_DOUBTED]) != Py_False || doubtful != Py_None
        || PyTuple_GET_SIZE(SLOT(held, node_at[N_SHAPE])) == 0) {
        return NULL;
    }
    /* No length of zero and an element type the layer defers: what `rules._admissible` says, remembered. */
    PyObject *spec = SLOT(held, node_at[N_SPEC]);
    PyObject *allowed = spec ? PyDict_GetItemWithError(admissible, spec) : NULL;
    if (allowed != Py_True) {
        PyErr_Clear();
        return NULL;
    }
    return held;
}

/* `array._defer_write(target, name, perform, operands)`: the cell gets a new node, the contents after the
 * write. `reads` is a tuple of what the write reads, with a node for each lazy array. 0, or -1 with an error set,
 * or 1 for "ask Python" (the pending set is full). */
static int
defer_write(PyObject *target, PyObject *name, PyObject *perform, PyObject *reads, PyFrameObject *frame)
{
    int full = at_the_limit();
    if (full != 0) {
        return full;
    }
    PyObject *cell = SLOT(target, lazy_at[L_CELL]);
    PyObject *held = SLOT(cell, cell_at[C_NODE]);
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
static PyObject *
in_place(PyObject *target, PyObject *other, PyObject *apply, PyObject *name, PyObject *perform, PyFrameObject *frame)
{
    PyObject *result = Miss, *read = NULL, *key = NULL;
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
        int written = reads ? defer_write(target, name, perform, reads, frame) : -1;
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
    if (Py_TYPE(value) == Lazy && SLOT(value, lazy_at[L_CELL]) == SLOT(target, lazy_at[L_CELL])) {
        return Miss; /* Perhaps `x[i] += v` assigning a view back to where it is: Python looks. */
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
    int written = reads ? defer_write(target, s_setitem, assign, reads, frame) : -1;
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
    if (aim == NULL || (Py_TYPE(value) == Lazy && SLOT(value, lazy_at[L_CELL]) == SLOT(target, lazy_at[L_CELL]))) {
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
    int done_write = defer_write(target, s_setitem, assign, reads, frame);
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
        int written = defer_write(target, name, into, reads, frame);
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

/* ---- A road: what stands on `LazyArray` where a method stood, takes the calls it can, and hands the others
 * to the method. ---- */

/* The kinds of road, by the method each stands for. */
enum { BINARY, UNARY, ARRAY_UFUNC, ARRAY_FUNCTION, REDUCTION, GETITEM, ATTRIBUTE, INPLACE, SETITEM, STANDIN_CALL,
       STANDIN_NAME, RESHAPE, TRANSPOSE, SWAPAXES, RAVEL, ROAD_KINDS };
static const char *const ROAD_NAMES[ROAD_KINDS] = {"binary", "unary", "array_ufunc", "array_function", "reduction", "getitem",
                                                   "attribute", "inplace", "setitem", "standin_call", "standin_name",
                                                   "reshape", "transpose", "swapaxes", "ravel"};

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
    if (!enabled || frame == NULL || count < 1) {
        goto python;
    }
    if (road->kind == STANDIN_CALL) {
        /* `LazyCallable.__call__`, for the commonest call: lazy arrays and plain values by position, plain values
         * by keyword. Nothing has to be made lazy, so the arguments go to NumPy as they came. `operation` is
         * `namespace._made_lazy` and `name` the types of a plain value. */
        if (Py_TYPE(args[0]) != StandInCall) {
            goto python;
        }
        int owned = 1;
        for (Py_ssize_t place = 1; place < count + named; place++) {
            if (place < count && Py_TYPE(args[place]) == Lazy) {
                owned = 0;
            } else {
                int is_plain = PySet_Contains(road->name, (PyObject *)Py_TYPE(args[place]));
                if (is_plain <= 0) {
                    PyErr_Clear();
                    goto python;
                }
            }
        }
        PyObject *made = PyObject_Vectorcall(SLOT(args[0], callable_at[F_FUNCTION]), args + 1, count - 1, kwnames);
        if (made == NULL || Py_TYPE(made) == Lazy) {
            return made; /* Deferred: the commonest answer. */
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
        operands = road->kind == UNARY ? PyTuple_Pack(1, args[0])
                   : road->reflected ? PyTuple_Pack(2, args[1], args[0]) : PyTuple_Pack(2, args[0], args[1]);
        kwargs = PyDict_New();
        if (operands == NULL || kwargs == NULL) {
            goto fail;
        }
        result = try_defer(road->operation, road->name, operands, kwargs, NULL, frame);
        break;
    case ARRAY_UFUNC:
        /* (self, ufunc, method, *inputs). `out=` alone is a write; any other keyword sends it to Python. */
        if (count < 3 || !PyUnicode_CheckExact(args[2]) || PyUnicode_Compare(args[2], s_call) != 0) {
            goto python;
        }
        if (named == 1 && PyUnicode_Compare(PyTuple_GET_ITEM(kwnames, 0), s_out) == 0 && PyTuple_CheckExact(args[count])
            && PyTuple_GET_SIZE(args[count]) == 1) {
            result = write_out(args[1], args + 3, count - 3, PyTuple_GET_ITEM(args[count], 0), frame);
            break;
        }
        if (named) {
            goto python;
        }
        operands = PyTuple_New(count - 3);
        kwargs = PyDict_New();
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
        operands = PyTuple_New(count);
        kwargs = PyDict_New();
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
        result = plain_item(args[0], args[1], frame);
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
        kwargs = PyDict_New();
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
    case SETITEM:
        if (named || count != 3) {
            goto python;
        }
        result = set_item(args[0], args[1], args[2], frame);
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
    PyObject *result = try_defer(args[0], args[1], args[2], args[3], args[4] == Py_None ? NULL : args[4], frame);
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
    PyObject *result = remembered_plan(args[0], args[1], args[2]);
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
        || take(given, "write_kind", &PyUnicode_Type, &write_kind) || take(given, "LazyCallable", &PyType_Type, &standin_call)
        || take(given, "LazyNumPy", &PyType_Type, &standin_module) || take(given, "slots", &PyTuple_Type, &slots)) {
        return NULL;
    }
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

static PyMethodDef methods[] = {
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
    if (!Miss || !zero || !one || !empty_tuple || !s_node || !s_made || !s_deferred || !s_observer || !s_limit || !s_settings
        || !s_last || !s_registry || !s_name || !s_unknown || !s_shape || !s_dtype || !s_doubt || !s_call || !s_getitem
        || !s_basic_shape || !s_out || !s_kind || !s_b || !s_setitem) {
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
