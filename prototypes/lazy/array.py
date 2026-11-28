"""`LazyArray`: what the user holds. A handle on one node of the graph.

Every NumPy call that meets a lazy array arrives in this file, by one of
three roads: an operator or method of the class, `__array_ufunc__` (NumPy's
ufuncs, and an `ndarray`'s own operators), or `__array_function__` (NumPy's
functions). Each call then takes one of three paths:

- deferred: `rules.plan` accepts it, and a node joins the graph;
- made at once: NumPy makes the call on gathered operands, and the result is
  a lazy array that already has its values;
- a write: a node too, the array's contents after the write, when nothing
  outside the layer can know the array's memory; otherwise everything
  pending is computed first and NumPy writes in the memory.

`gather()` is the way to ask for values. Anything else that needs them
(`float(x)`, `if x > 0:`, `np.asarray(x)`, a library) gets them too: they are
computed on the spot and the log says so, once per line. An array made by a
strict stand-in raises `NotGathered` there instead.
"""
from __future__ import annotations

import operator
import sys
import warnings
import weakref
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable, Iterator

import numpy as np

from . import evaluate, explain, graph, record, rules
from .graph import Node, NotGathered, Trigger

try:
    # The quick road for writing a node, in C. The layer is the same without it; see `use_core`.
    from . import _core as _built_core
except ImportError:
    _built_core = None

# The version of `_core.c` this file was written against. A module built from another is not used.
CORE_VERSION = 38

# NumPy functions that answer from the shape alone, and the attribute that answers each.
SHAPE_QUESTIONS = {np.shape: "shape", np.ndim: "ndim", np.size: "size"}
# NumPy functions that make a new array from the shape and element type of
# their first argument and from nothing else of it. They need no value.
SHAPED_LIKE = frozenset({np.zeros_like, np.ones_like, np.empty_like, np.full_like})
# NumPy functions that write into their first argument.
WRITING_FUNCTIONS = frozenset({np.copyto, np.putmask, np.place, np.put, np.fill_diagonal})
# Methods of `ndarray` that write into the array they are called on.
WRITING_METHODS = frozenset({"sort", "fill", "put", "partition", "resize", "itemset", "setfield", "byteswap"})
# The one ufunc method that writes into its first argument.
WRITING_UFUNC_METHOD = "at"
# `ndarray.view(dtype, type)`: its two parameters by name, in the order they are given by position.
VIEW_PARAMETERS = ("dtype", "type")
# The call a view made at once is counted and reported as.
VIEW_CALL = "ndarray.view"
# The index that `x.view()` is deferred as: `x[...]`, written as the tuple `_plain_item` reads, so that the view is
# one whose shape the layer knows without asking NumPy and an assignment through it is known to fit the same way.
WHOLE_INDEX = (Ellipsis,)


# In-place ufunc calls, `np.clip` and the roundings take `out=`; these functions are deferred as writes when they do.
OUT_FUNCTIONS = frozenset({np.clip, np.round, np.around})
# Functions that give one new array or number, which never shares memory with an argument, whatever they are
# given. With small calls made at once (`at_once_below`) the C core calls one of these on small arrays that have
# their values without asking the rules: the result is an array of its own either way.
FRESH_FUNCTIONS = frozenset({np.dot, np.matmul, np.sum, np.mean, np.max, np.min, np.prod, np.std, np.var, np.argmax,
                             np.argmin, np.copy, np.cumsum, np.any, np.all, np.outer})
# Functions the rules have no rule for, so that a call of one is never a node, and which give one new array or
# number that shares no memory with an argument when they are called without `out=`. A call of one is made at
# once on gathered operands (`_at_once`) and counted as outside the deferred set. With small calls made at once
# (`at_once_below`) the C core makes it, on arrays of any size that have their values, where the stand-in is called.
NEVER_DEFERRED = frozenset({np.clip, np.average})
# What a recording calls an element read from an array that has its values, where it is an input of a later graph.
ELEMENT_SOURCE = "x[index]"
# The layout a write's copy of an array is given: the array's own, as far as NumPy can keep it. A write that
# NumPy makes in place leaves the array laid out as it was, and a later sum over it adds in that order.
KEPT_ORDER = "K"
# What the node of `x.shape = s` is called, and the length NumPy works out by itself in a shape.
SHAPE_ASSIGNMENT = "x.shape = s"
UNKNOWN_LENGTH = -1
# The types `_lazies_in` looks inside.
COLLECTIONS = (list, tuple, dict)
# NumPy functions `__array_function__` answers by a road of their own, before it tries to defer.
HANDLED_APART = frozenset(SHAPE_QUESTIONS) | SHAPED_LIKE | WRITING_FUNCTIONS | {np.nonzero}
# Element-by-element writes found to fit, remembered at most, before the memory is emptied.
FIT_MEMORY_LIMIT = 4096
# What a lazy array that holds its value alone keeps where a view keeps what it has seen: the kind and the
# strictness its node is to have, once something asks for the node. One tuple for each pair, made here once, so
# that such an array is one object and nothing else. In the order the C core reads them: by kind, then strictness.
VALUED = tuple((kind, strict) for kind in ("", graph.AT_ONCE) for strict in (False, True))
# The ufunc behind each in-place function of the `operator` module that `LazyArray` has a method for: what
# `s += v` is for a number, into which nothing is written.
IN_PLACE_UFUNCS = {operator.iadd: np.add, operator.isub: np.subtract, operator.imul: np.multiply,
                   operator.itruediv: np.true_divide}

# The element-by-element writes that `_fits_elementwise` found to fit, by what
# `_fit_key` writes down of each. Only those: one that does not fit is made at
# once, or raises, and is looked at again every time.
_fitting: set[tuple] = set()
# The keys hold numbers of shapes, which mean nothing once that memory is emptied.
graph.spec_memory_emptied.append(_fitting.clear)
# The assignments through an index of pending arrays that `_assignment_fits` found to fit, by what
# `_assignment_key` writes down of each. Only those, as for `_fitting`, and within the same bound.
_assignable: set[tuple] = set()
graph.spec_memory_emptied.append(_assignable.clear)


@dataclass(eq=False, slots=True)
class _Cell:
    """One array as NumPy knows it: memory that several lazy arrays may be looking at.

    A lazy array and every view made of it share a cell. A deferred write
    gives the cell a new node, and each of them reads the new contents.
    """

    # The node of the cell's contents now.
    node: Node
    # Lazy arrays made by `np.reshape` of this cell's, held weakly. NumPy
    # shares memory between them or does not, depending on a layout that is
    # not known before computing; while one is alive, writes are not deferred.
    # A list of references, not a set: a lazy array, like an `ndarray`, has no hash.
    # None until there is one: most cells never have any.
    doubtful: list[weakref.ref] | None = None
    # True for the cell of such an array.
    doubted: bool = False

    def doubt(self, made: "LazyArray") -> None:
        """Note an array that may share this cell's memory.

        Args:
            made: A lazy array made by `np.reshape` of this cell's, or by an index that may give a view of it.
        """
        if self.doubtful is None:
            self.doubtful = []
        self.doubtful.append(weakref.ref(made))

    def shared_in_doubt(self) -> bool:
        """Say whether an array that may share this cell's memory is still alive, forgetting those that are not."""
        if not self.doubtful:
            return False
        self.doubtful = [reference for reference in self.doubtful if reference() is not None]
        return bool(self.doubtful)


@dataclass(eq=False, slots=True)
class _Step:
    """One view operation on the way from a cell's contents to a lazy array that is a view of them.

    It is never changed once made: a view and the views made of it share their steps.
    """

    # The view function, or `operator.getitem`.
    operation: Callable[..., Any]
    # What a node of it is called.
    name: str
    # The arguments after the array: the index, the axes.
    tail: tuple
    # Keyword arguments as written.
    kwargs: dict
    # Shape of the view.
    shape: tuple[int, ...]
    # Element type of the view.
    dtype: np.dtype
    # The line that wrote the view.
    origin: graph.Origin
    # True when the view's shape follows from the shape viewed and from whole
    # numbers written in the program, so `shape` is right for any array of
    # the cell's shape. False for a step whose arguments hold anything that
    # is read again each time the view is taken, and for one made of a shape
    # with a length that was not known.
    plain: bool = False

    def of(self, array: Any) -> Any:
        """Take the view of an array.

        Args:
            array: A computed array.

        Returns:
            NumPy's view of it.
        """
        return self.operation(array, *self.tail, **self.kwargs)


@dataclass(eq=False, slots=True)
class _Write:
    """What the node of a deferred write calls: a copy of what the array held, with the write made into it.

    A class and not a function made inside `_defer_write`, so that the C core can make one too.
    """

    # The view operations from the cell's contents to the array written into.
    steps: tuple
    # Makes the write: called with the part of the copy the target is a view of, then the operands' values.
    perform: Callable[..., Any]

    def __call__(self, old: np.ndarray, *values: Any) -> np.ndarray:
        """Copy what the array held and make the write into the copy.

        Args:
            old: The cell's contents before the write.
            *values: The values of what the write reads.

        Returns:
            The cell's contents after the write.
        """
        new = old.copy(order=KEPT_ORDER)
        view = new
        for step in self.steps:
            view = step.of(view)
        self.perform(view, *values)
        return new

    def in_place(self, old: np.ndarray, *values: Any) -> np.ndarray:
        """Make the write into the memory the array has: what `evaluate` calls when nothing reads the old contents again.

        Args:
            old: The cell's contents before the write, in an array nothing else will read.
            *values: The values of what the write reads.

        Returns:
            The same array, written into.
        """
        view = old
        for step in self.steps:
            view = step.of(view)
        self.perform(view, *values)
        return old


@dataclass(eq=False, slots=True)
class _Assign:
    """Makes `view[index] = assigned` for an index that holds no lazy array: what `_Write` performs for a plain assignment."""

    # The index as written between the brackets.
    index: Any

    def __call__(self, view: np.ndarray, assigned: Any) -> None:
        """Assign into the part of the copy the target is a view of.

        Args:
            view: That part.
            assigned: The value assigned.
        """
        view[self.index] = assigned


@dataclass(eq=False, slots=True)
class _Into:
    """Makes a call with its result written into `out=`: what `_Write` performs for `np.add(a, b, out=x)`."""

    # A ufunc, or one of OUT_FUNCTIONS.
    function: Callable[..., Any]

    def __call__(self, view: np.ndarray, *values: Any) -> None:
        """Make the call with its result written into the part of the copy the target is a view of.

        Args:
            view: That part.
            *values: The call's positional arguments, with values for lazy arrays.
        """
        self.function(*values, out=view)


def _set_shape(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return a view of an array in another shape, or fail as `array.shape = shape` fails.

    Args:
        array: A computed array.
        shape: The new shape, every length known.

    Returns:
        A view of the same memory. Never a copy: where `np.reshape` would
        copy, assigning a shape raises, and so does this.
    """
    view = array.view()
    view.shape = shape
    return view


def _resolved_shape(shape: Any, size: int) -> tuple[int, ...]:
    """Work out the shape an assignment to `.shape` asks for.

    Args:
        shape: What was assigned: a whole number or a sequence of them, at most one of them -1.
        size: How many elements the array has.

    Returns:
        The shape with the -1 replaced by the length that makes the sizes agree.

    Raises:
        ValueError: With NumPy's words, when no such shape holds `size` elements.
    """
    wanted = tuple(operator.index(length) for length in shape) if isinstance(shape, (tuple, list)) \
        else (operator.index(shape),)
    known = 1
    for length in wanted:
        if length != UNKNOWN_LENGTH:
            known *= length
    if wanted.count(UNKNOWN_LENGTH) > 1:
        raise ValueError("can only specify one unknown dimension")
    if UNKNOWN_LENGTH in wanted:
        if known == 0 or size % known:
            raise ValueError(f"cannot reshape array of size {size} into shape {wanted}")
        wanted = tuple(size // known if length == UNKNOWN_LENGTH else length for length in wanted)
    elif known != size:
        raise ValueError(f"cannot reshape array of size {size} into shape {wanted}")
    return wanted


def wrap(value: Any, source: str = "array", strict: bool = False, owned: bool = False) -> "LazyArray":
    """Make a lazy array of something that already has values.

    Args:
        value: A lazy array, which is returned as it is; or an `ndarray` or
            NumPy scalar, which is held by reference; or anything
            `np.asarray` accepts.
        source: What made the array, for a recording: "array", or the name
            of the stand-in's function, such as "np.zeros".
        strict: Whether the new array refuses to give its values without
            `gather()`. A lazy array that is passed in keeps what it has.
        owned: True when nothing outside the layer can know the array's
            memory: the stand-in has just made it. False for an array the
            user handed in, into which a write has to happen at once.

    Returns:
        A lazy array on a computed node.
    """
    if isinstance(value, LazyArray):
        return value
    if not isinstance(value, (np.ndarray, np.generic)):
        value = np.asarray(value)
    return LazyArray(graph.leaf(value, source, strict=strict, exposed=not owned))


def gather(*arrays: Any) -> tuple:
    """Compute several lazy arrays in one walk of the graph.

    Args:
        *arrays: Lazy arrays; anything else is returned as it is. With no
            argument, everything pending is computed.

    Returns:
        The value of each, in order: what NumPy would have returned for it.
    """
    trigger = Trigger(graph.GATHER, "gather")
    if not arrays:
        evaluate.compute_pending(trigger)
    evaluate.compute([array._node for array in arrays if isinstance(array, LazyArray)], trigger)
    for array in arrays:
        if isinstance(array, LazyArray):
            array._expose()
    return tuple(array._node.value if isinstance(array, LazyArray) else array for array in arrays)


def _unwrapped(args: tuple) -> tuple:
    """Replace each lazy array among some arguments by its node.

    Args:
        args: Positional arguments as the user wrote them.

    Returns:
        The same arguments with `Node`s where lazy arrays were. Only the top
        level is looked at: a list holding a lazy array is left alone, and
        `rules.plan` then declines the call.
    """
    return tuple(arg._node if isinstance(arg, LazyArray) else arg for arg in args)


def _read(args: tuple) -> tuple[tuple, bool]:
    """Replace each lazy array among some arguments by its node, and say whether a strict one is involved.

    Args:
        args: Positional arguments as the user wrote them.

    Returns:
        (what `_unwrapped` gives, what `_strict` says of the same
        arguments), from one pass where the arguments hold no collection.
    """
    unwrapped = []
    strict = nested = False
    for arg in args:
        if isinstance(arg, LazyArray):
            arg = arg._node
            strict = strict or arg.strict
        elif isinstance(arg, COLLECTIONS) and not _holds_no_lazy(arg):
            nested = True
        unwrapped.append(arg)
    return tuple(unwrapped), strict or (nested and _strict(args))


def _holds_no_lazy(collection: Any) -> bool:
    """Say, at a glance, that a collection holds no lazy array: the shape of a reshape, a tuple of axes.

    Args:
        collection: A list, a tuple or a dict.

    Returns:
        True for a list or tuple none of whose items is a lazy array or
        another collection, in which `_lazies_in` would find nothing. False
        says nothing: the caller then lets `_lazies_in` look.
    """
    if isinstance(collection, dict):
        return False
    for item in collection:
        if isinstance(item, LazyArray) or isinstance(item, COLLECTIONS):
            return False
    return True


def _lazies_in(obj: Any) -> list["LazyArray"]:
    """Find every lazy array in an argument, looking inside lists, tuples and dicts.

    Args:
        obj: An argument, or a collection of arguments.

    Returns:
        The lazy arrays found, in order.
    """
    if isinstance(obj, LazyArray):
        return [obj]
    if isinstance(obj, dict):
        obj = tuple(obj.values())
    if isinstance(obj, (list, tuple)):
        return [lazy for item in obj for lazy in _lazies_in(item)]
    return []


def _valued(obj: Any) -> Any:
    """Replace every lazy array in an argument by its value, which must be computed.

    Args:
        obj: An argument, or a list, tuple or dict of arguments.

    Returns:
        The same structure with values where lazy arrays were.
    """
    if isinstance(obj, LazyArray):
        return obj._node.value
    if isinstance(obj, dict):
        return {key: _valued(item) for key, item in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_valued(item) for item in obj)
    return obj


def _strict(*arguments: Any) -> bool:
    """Say whether a call involves a strict lazy array.

    Args:
        *arguments: Arguments of the call, or collections of them.

    Returns:
        True when any lazy array among them is strict. What is computed from
        a strict array is strict.
    """
    return any(lazy._node.strict for lazy in _lazies_in(arguments))


def _hand_out(call: str, strict: bool, everything: bool, nodes: list[Node], refusal: str) -> None:
    """Let values leave without `gather()`: refuse for a strict array, otherwise compute and say so.

    A number, a truth value or a string asked of arrays that all have their
    values is given without a word: nothing is computed for it, and the log
    and the counters are the places where a graph was cut.

    Args:
        call: What needs the values, as the log and a recording name it.
        strict: Whether a strict array is involved.
        everything: True when an array itself is handed out, or an `ndarray`
            is about to be written. Everything pending is then computed
            first: the receiver may write into memory that a pending
            operation still reads, and once nothing is pending the program is
            in the state the eager program would be in. False when only a
            number, a truth value or a string leaves.
        nodes: The nodes whose values are needed.
        refusal: What `NotGathered` says for a strict array.

    Raises:
        NotGathered: For a strict array, whether or not it has its values.
        Exception: What computing raised.
    """
    if strict:
        raise NotGathered(refusal)
    if _at_once_limit and not graph._pending and not any(node.pending or node.failure is not None for node in nodes):
        # With small calls made at once (`at_once_below`) most arrays have their values when they are handed
        # out, and with nothing pending no graph is cut by it: nothing to compute, to count or to log.
        return
    if everything or any(node.pending for node in nodes):
        # A number, a truth value or a string read from arrays that have
        # their values computes nothing and cuts no graph: not counted, not
        # logged. A loop over the elements of such an array reads one per turn.
        graph.fallback(graph.NEEDED, call)
    trigger = Trigger(graph.NEEDED, call)
    if everything:
        evaluate.compute_pending(trigger)
    evaluate.compute(nodes, trigger)


def _try_defer(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict,
               ruled_as: Callable[..., Any] | None = None, frame: FrameType | None = None) -> "LazyArray | None":
    """Add a call to the graph if the rules allow it.

    Args:
        operation: The callable the node will call: a NumPy function or
            ufunc, or a method written on a lazy array.
        name: What to call the node.
        args: Positional arguments as the user wrote them.
        kwargs: Keyword arguments as the user wrote them.
        ruled_as: The NumPy function whose rule gives the shape and element
            type, when `operation` is a method and has no rule of its own:
            `np.sum` for `x.sum(...)`. None when `operation` has one.
        frame: The frame that called into the layer, when the caller has it; see `graph.deferred`.

    Returns:
        A pending lazy array, or None when the call has to be made at once.

    Raises:
        Exception: NumPy's own, when the shapes or types are wrong.
    """
    if _core is not None:
        # A lazy array; or None, when the core has just asked the rules about this very call and they said it is
        # not deferred; or MISS, when the call is one the core leaves to the lines below.
        made = _core.try_defer(operation, name, args, kwargs, ruled_as, frame)
        if made is not _core.MISS:
            return made
    # One pass over the arguments does the work of `_unwrapped` and, for
    # arguments that are not collections, of `_strict`.
    unwrapped = []
    strict = nested = False
    for arg in args:
        if isinstance(arg, LazyArray):
            # `_node`, without the call for an array that is no view.
            arg = arg._node if arg._steps else arg._cell.node
            strict = strict or arg.strict
        elif isinstance(arg, COLLECTIONS) and not _holds_no_lazy(arg):
            nested = True
        unwrapped.append(arg)
    unwrapped = tuple(unwrapped)
    for value in kwargs.values():
        if isinstance(value, LazyArray) or (isinstance(value, COLLECTIONS) and not _holds_no_lazy(value)):
            nested = True
    signature = rules.plan(ruled_as or operation, unwrapped, kwargs)
    if signature is None:
        return None
    if len(graph.NOT_COMPUTED) >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    if nested and not strict:
        strict = _strict(args, kwargs)  # A lazy array inside a collection or behind a keyword counts too.
    node = graph.deferred(operation, name, unwrapped, kwargs, signature.shape, signature.dtype, signature.kind,
                          strict, None, signature.spec, frame)
    if signature.kind != rules.VIEW:
        return LazyArray(node)
    base = args[0] if args and isinstance(args[0], LazyArray) else None
    if base is None or (operation is operator.getitem and signature.shape == ()):
        # A new array. An index that leaves no dimension gives a number, which is a copy, not a view.
        return LazyArray(node)
    if operation is operator.getitem or operation in rules.SURE_VIEWS:
        step = _Step(operation, name, unwrapped[1:], kwargs, signature.shape, signature.dtype, node.origin,
                     not kwargs and _is_plain_step(unwrapped))
        return LazyArray(node, base._cell, base._steps + (step,))
    # `np.reshape`: a view or a copy, which is not known yet.
    made = LazyArray(node)
    made._cell.doubted = True
    base._cell.doubt(made)
    return made


def _is_plain_step(unwrapped: tuple) -> bool:
    """Say whether a view's shape follows from the shape viewed and from whole numbers written in the program.

    Args:
        unwrapped: The view's arguments: the node viewed, then its axes or its index. A view with keywords is not asked about.

    Returns:
        True when the node's shape is known and every other argument is an
        `int`, or a tuple of them. See `_Step.plain`. A slice is not looked
        into: the indices that are plain in every part are made by
        `_plain_item`, which says so itself.
    """
    for item in unwrapped[1:]:
        if type(item) is not int and not (type(item) is tuple and all(type(part) is int for part in item)):
            return False
    return graph.spec(unwrapped[0]) != 0


def _part(operation: Callable[..., Any], place: int) -> Callable[..., Any]:
    """Build a function that makes a call and gives one item of the tuple it returns.

    Args:
        operation: A callable that returns a tuple, such as `np.nonzero`.
        place: Which item of the tuple.

    Returns:
        A function with the arguments of `operation`.
    """
    def call(*args, **kwargs):
        """Make the call and take the item."""
        return operation(*args, **kwargs)[place]
    return call


def _defer_nonzero(operation: Callable[..., Any], name: str, args: tuple,
                   ruled_as: Callable[..., Any] | None = None) -> "tuple[LazyArray, ...] | None":
    """Add `np.nonzero(a)` to the graph if the rules allow it: one node for each array of the tuple.

    A node has one value, and `np.nonzero` gives a tuple with an array per
    dimension of its operand. Each of those arrays is a node of its own that
    makes the call and keeps its item, so the first build calls `np.nonzero`
    once per dimension. They carry one mark for their length.

    Args:
        operation: `np.nonzero`, or the method written on a lazy array.
        name: What to call the nodes; each adds its place, as in "nonzero[1]".
        args: The one operand.
        ruled_as: `np.nonzero` when `operation` is the method. None when it is the function.

    Returns:
        A tuple of pending lazy arrays, or None when the call has to be made at once.
    """
    unwrapped = _unwrapped(args)
    signature = rules.plan(ruled_as or operation, unwrapped, {})
    if signature is None:
        return None
    if len(graph.NOT_COMPUTED) >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    return tuple(LazyArray(graph.deferred(_part(operation, place), f"{name}[{place}]", unwrapped, {}, signature.shape,
                                          signature.dtype, signature.kind, _strict(args)))
                 for place in range(len(unwrapped[0].shape)))


def _defer_or_call(operation: Callable[..., Any], name: str, call: str, args: tuple,
                   frame: FrameType | None = None) -> Any:
    """Defer a call if the rules allow it, and make it at once if they do not.

    Args:
        operation: The NumPy callable.
        name: What to call the node if the call is deferred.
        call: The call's name if it is made at once, for the counters and for messages.
        args: Positional arguments as the user wrote them.
        frame: The frame that called into the layer, when the caller has it; see `graph.deferred`.

    Returns:
        A pending lazy array, or the result made at once.
    """
    deferred = _try_defer(operation, name, args, {}, None, frame)
    # Not `deferred or ...`: a truth test of a lazy array would compute it.
    return deferred if deferred is not None else _at_once(operation, call, args, {})


def _view(function: Callable[..., Any], args: tuple, frame: FrameType) -> Any:
    """Make the call of a NumPy function that a method of a lazy array stands for.

    `x.T` is `np.transpose(x)`. NumPy would hand that call to
    `__array_function__`, which would do what this function does: the detour
    through NumPy is all that is left out.

    Args:
        function: A NumPy function that `__array_function__` answers by no road of its own, with no keyword.
        args: The lazy array, then the function's other positional arguments.
        frame: The frame that called the method.

    Returns:
        A pending lazy array, or the result made at once.
    """
    deferred = _try_defer(function, function.__name__, args, {}, None, frame)
    # Not `deferred or ...`: a truth test of a lazy array would compute it.
    return deferred if deferred is not None else _at_once(function, f"np.{function.__name__}", args, {})


def _view_request(args: tuple, kwargs: dict) -> tuple[bool, tuple, dict]:
    """Read the arguments of `x.view(...)` on a lazy array, with the lazy array's own type taken out.

    `np.empty(shape).view(type(x))` is how NumPy code keeps the kind of
    array it was given: a masked array stays masked. The type of a lazy
    array is `LazyArray`, which NumPy does not know: it would take the class
    for an element type and raise. To `view` the class means what it means
    in that line, the kind of array the receiver already is, so it is taken
    out: `x.view(type(x))` is `x.view()`, and `x.view(dtype, type(x))` is
    `x.view(dtype)`.

    NumPy takes a subclass of `ndarray` for the array type wherever it is
    given, as `dtype` too, and raises for a call that gives `type` beside
    it. The same holds here with `LazyArray` as an array type: any other
    call that names the class is handed to NumPy with `ndarray` in its
    place, and NumPy raises what it raises for `ndarray` there.

    Args:
        args: The positional arguments of the call, without the array.
        kwargs: Its keyword arguments.

    Returns:
        (whether the call is `x.view()` or names `ndarray` and nothing
        else, so that it asks for the same elements with no change of
        element type; the positional arguments to give NumPy if the call is
        made at once; the keyword arguments for the same).
    """
    named = args + tuple(kwargs.values())
    # By identity: `==` on an element type or an array would ask them to compare themselves with a class.
    if not any(value is LazyArray for value in named):
        whole = not named or (len(named) == 1 and named[0] is np.ndarray and set(kwargs) <= set(VIEW_PARAMETERS))
        return whole, args, kwargs
    given = dict(zip(VIEW_PARAMETERS, args))
    proper = len(args) <= len(VIEW_PARAMETERS) and not set(given) & set(kwargs) and set(kwargs) <= set(VIEW_PARAMETERS)
    given.update(kwargs)
    dtype_name, type_name = VIEW_PARAMETERS
    if proper and len(given) == 1:
        # `x.view(type(x))`, by position or under either name.
        return True, (), {}
    if proper and given[type_name] is LazyArray and not (isinstance(given[dtype_name], type)
                                                         and issubclass(given[dtype_name], (np.ndarray, LazyArray))):
        # `x.view(dtype, type(x))`: a change of element type, NumPy's.
        return False, (given[dtype_name],), {}
    # A call `view` does not have, or an array type beside `type`: NumPy says which, of `ndarray`.
    return (False, tuple(np.ndarray if arg is LazyArray else arg for arg in args),
            {name: np.ndarray if value is LazyArray else value for name, value in kwargs.items()})


def _is_viewed_by_index(target: "LazyArray") -> bool:
    """Say whether `x[...]` of a lazy array is what NumPy's `x.view()` would give, and is an index the layer defers.

    For an `ndarray` of one dimension or more the two are the same view.
    They differ for a number, whose view is a number where `x[...]` is an
    array without dimensions, and they may differ for a subclass of
    `ndarray`, which has methods of its own for both.

    Args:
        target: The array `view` was called on.

    Returns:
        True for an array of one dimension or more, no dimension of length
        zero and an element type the layer defers, that is pending or has
        an exact `ndarray` for its value. Nothing is computed to answer,
        and an array that holds its value alone is given no node for it.
    """
    held = target._held
    if type(held) is not _Cell:
        # The value alone, as the C core leaves the result of a call it made at once.
        return (type(held) is np.ndarray and held.ndim > 0 and held.dtype.kind in rules.DEFERRED_KINDS
                and 0 not in held.shape)
    node = target._node
    return node.shape != () and rules.is_admissible(node) and (node.pending or type(node.value) is np.ndarray)


def _at_once(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict) -> Any:
    """Make a call now, with NumPy, on gathered operands.

    With small calls made at once (`at_once_below`) the C core makes the
    calls of NEVER_DEFERRED itself where the stand-in is called, on
    arguments that have their values: the same count, the same call and the
    same lazy result as here.

    Args:
        operation: The NumPy callable.
        name: The call's name, for the counters and for messages.
        args: Positional arguments, possibly holding lazy arrays.
        kwargs: Keyword arguments, possibly holding lazy arrays.

    Returns:
        The result with every array in it made lazy, so that what is computed
        from a lazy array is a lazy array. A plain Python value (a bool, a
        list of numbers) is returned as it is.

    Raises:
        NotGathered: When NumPy answers with a plain Python value and a
            strict array is among the arguments.
    """
    graph.fallback(graph.AT_ONCE, name)
    strict = _strict(args, kwargs)
    lazies = _lazies_in((args, kwargs))
    evaluate.compute([lazy._node for lazy in lazies], Trigger(graph.AT_ONCE, name))
    result = operation(*_valued(args), **_valued(kwargs))
    return _made_lazy(result, name, strict, _shares_memory(result, lazies))


def _shares_memory(result: Any, lazies: list["LazyArray"]) -> bool:
    """Say whether what a call made at once returned may share memory with one of its lazy arguments.

    `x.ravel()`, `np.atleast_2d(x)` and `x.astype(t, copy=False)` can return
    a view of `x`, or `x` itself. NumPy then lets a write into either be
    seen through the other, so neither may have a write deferred.

    Args:
        result: What NumPy returned.
        lazies: The lazy arrays among the call's arguments, computed.

    Returns:
        True when an array of the result may overlap one of them. Those
        arguments are marked as exposed on the way.
    """
    arrays = [item for item in (result if isinstance(result, (list, tuple)) else [result]) if isinstance(item, np.ndarray)]
    shared = False
    for lazy in lazies:
        value = lazy._node.value
        if isinstance(value, np.ndarray) and any(np.may_share_memory(array, value) for array in arrays):
            lazy._expose()
            shared = True
    return shared


def _made_lazy(result: Any, name: str, strict: bool, exposed: bool = False) -> Any:
    """Turn the result of a call made at once into lazy arrays.

    Args:
        result: What NumPy returned.
        name: The call's name, for messages and the log.
        strict: Whether a strict array was among the call's arguments.
        exposed: Whether the result may share memory with an argument.

    Returns:
        A lazy array for an array or a NumPy scalar; the same kind of tuple
        or list for a tuple or list of those; None for None. Anything else
        (a Python bool, float, list of numbers) is values leaving without
        `gather()`: it is returned as it is, and the log says so.

    Raises:
        NotGathered: For such a value when a strict array was involved.
    """
    if isinstance(result, (np.ndarray, np.generic)):
        return LazyArray(graph.leaf(result, name, graph.AT_ONCE, strict, exposed))
    if result is None:
        return None
    if isinstance(result, (list, tuple)) and all(isinstance(item, (np.ndarray, np.generic)) for item in result):
        items = [LazyArray(graph.leaf(item, name, graph.AT_ONCE, strict, exposed)) for item in result]
        return type(result)(*items) if hasattr(result, "_fields") else type(result)(items)
    if strict:
        raise NotGathered(f"{name} gives a plain Python value ({type(result).__name__}) and its arguments are lazy: "
                          "call .gather() on them first")
    graph.fallback(graph.NEEDED, name)
    return result


def _write(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict, targets: tuple) -> None:
    """Make a call that writes into arrays that exist, after computing everything pending.

    With small calls made at once (`at_once_below`) the C core makes a ufunc
    call with one output itself when nothing is pending and every operand
    has its values, into a lazy array or a plain one (`ufunc_into_now`): the
    same call on the same values, with nothing to compute first and so
    nothing counted. Every other write is made here.

    Args:
        operation: The NumPy callable.
        name: The call's name, for the counters and for messages.
        args: Positional arguments, possibly holding lazy arrays.
        kwargs: Keyword arguments, possibly holding lazy arrays.
        targets: The arrays the call writes into.

    Raises:
        NotGathered: When a target is a plain `ndarray` and a strict array is
            among the arguments: lazy values would leave without `gather()`.
    """
    if all(isinstance(target, LazyArray) for target in targets):
        graph.fallback(graph.WRITE, name)
        evaluate.compute_pending(Trigger(graph.WRITE, name))
    else:
        # Lazy values are written into an `ndarray`. Something pending may
        # read that `ndarray`, so everything pending is computed before it changes.
        _hand_out(name, _strict(args, kwargs), True, [lazy._node for lazy in _lazies_in((args, kwargs))],
                  f"{name} would write the values of a lazy array into an ndarray: call .gather() on the lazy array first")
    operation(*_valued(args), **_valued(kwargs))


def _compute_for_small(lazy: "LazyArray") -> None:
    """Compute a pending array that a call on small operands reads, so that the call need not wait as a node.

    The C core calls this (`small_reads`) when small calls are made at
    once and an operand of one, itself small, has no value yet: a row of a
    large array that is still pending. The array is computed with what it
    reads, as `float(x)` computes one, and the call is then NumPy's at that
    line.

    Args:
        lazy: A lazy array whose node is pending.

    Raises:
        Exception: What computing the node raised.
    """
    evaluate.compute([lazy._node], Trigger(graph.NEEDED, "a call on small arrays"))


def _method(name: str) -> Callable[..., Any]:
    """Build a function that calls a method by name on its first argument.

    Args:
        name: The method's name.

    Returns:
        A function `f(value, *args, **kwargs)`. The method is looked up on the
        value, not on `ndarray`: the value of a reduction is a NumPy scalar,
        which has the same methods and is not an `ndarray`.
    """
    def call(value, *args, **kwargs):
        """Call the named method on the computed value."""
        return getattr(value, name)(*args, **kwargs)
    return call


def _leaving(what: str, convert: Callable[..., Any], everything: bool = False) -> Callable[..., Any]:
    """Build a method through which values leave without `gather()`.

    Args:
        what: What asks for the values, as messages, the log and a recording name it.
        convert: Turns the computed value, and any further arguments of the
            method, into what was asked for.
        everything: True when the array itself is handed out; see `_hand_out`.

    Returns:
        A method that computes the array, says so in the log, and returns
        what was asked for; and that raises `NotGathered` for a strict array.
    """
    def leave(self, *args, **kwargs):
        """Give what was asked for, computing first; refuse for a strict array."""
        _hand_out(what, self._node.strict, everything, [self._node],
                  f"{what} needs the values of a lazy array: call .gather() first")
        if everything:
            self._expose()
        return convert(self._node.value, *args, **kwargs)
    leave.road = ("leave", convert, what, everything)
    return leave


def _as_array(value: Any, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
    """Answer NumPy's `__array__`: an `ndarray` of a computed value.

    Args:
        value: The computed `ndarray` or NumPy scalar.
        dtype: The element type NumPy asks for, or None for the value's own.
        copy: True when NumPy asks for a copy; None or False when it does not.

    Returns:
        The array itself when no copy and no other type is asked for, as
        `gather()` returns it; otherwise what NumPy makes of it.
    """
    return np.array(value, dtype=dtype, copy=True) if copy else np.asarray(value, dtype=dtype)


def _binary(ufunc: np.ufunc, reflected: bool = False, written: Callable[..., Any] | None = None) -> Callable[..., Any]:
    """Build the method for a binary operator.

    Args:
        ufunc: The ufunc the operator stands for.
        reflected: True for the method Python calls when the lazy array is on the right.
        written: What the node calls, when that is not the ufunc: the function of the `operator` module for
            an operator that NumPy does not compute by its ufunc alone. `x ** 2` is `ndarray.__pow__`, which
            squares, and takes a reciprocal for -1 and a square root for 0.5; `np.power(x, 2)` runs the
            general loop, ten times slower, and for a complex root gives other bits. The node makes the
            call the program wrote. The ufunc still gives the rule: the shape and the element type are its.

    Returns:
        A method that defers the call, or makes it at once when it cannot be deferred.
    """
    name = ufunc.__name__
    call = f"np.{name}"
    operation, ruled_as = (ufunc, None) if written is None else (written, ufunc)

    def apply(self, other):
        """Defer the operator's call on the two operands, in the order written."""
        args = (other, self) if reflected else (self, other)
        deferred = _try_defer(operation, name, args, {}, ruled_as, sys._getframe(1))
        # Not `deferred or ...`: a truth test of a lazy array would compute it.
        return deferred if deferred is not None else _at_once(operation, call, args, {})
    apply.road = ("binary", operation, name, ruled_as, reflected)
    return apply


def _unary(ufunc: np.ufunc) -> Callable[..., Any]:
    """Build the method for a unary operator.

    Args:
        ufunc: The ufunc the operator stands for.

    Returns:
        A method that defers the ufunc on the array.
    """
    name = ufunc.__name__
    call = f"np.{name}"

    def apply(self):
        """Defer the operator's ufunc on the array."""
        deferred = _try_defer(ufunc, name, (self,), {}, None, sys._getframe(1))
        return deferred if deferred is not None else _at_once(ufunc, call, (self,), {})
    apply.road = ("unary", ufunc, name)
    return apply


@dataclass(frozen=True)
class _Slot:
    """The place, in an index, of a lazy array whose values come when the write is computed."""

    # Which of the write's lazy index arrays goes here.
    place: int


# Pending arrays in one index that the C core writes down itself; an index with more goes the Python way.
SLOTS_MADE = 8


@dataclass(eq=False, slots=True)
class _Select:
    """What the node of `x[index]` calls when the index holds arrays that were pending: the index with their values put in.

    A class and not a function made inside `_select`, so that the C core can make one too.
    """

    # The index's items, with a `_Slot` where a pending array's value goes.
    template: tuple
    # True when the index was written as a tuple, False for one item.
    whole: bool

    def __call__(self, base: Any, *found: Any) -> Any:
        """Index the computed array with the computed index.

        Args:
            base: The array's value.
            *found: The values of the index's pending arrays, in the order of their slots.

        Returns:
            What NumPy gives for the index.
        """
        parts = tuple(found[part.place] if isinstance(part, _Slot) else part for part in self.template)
        return base[parts if self.whole else parts[0]]


@dataclass(eq=False, slots=True)
class _AssignThrough:
    """Makes `view[index] = assigned` for an index that held pending arrays: what `_Write` performs for such an assignment."""

    # The index's items, with a `_Slot` where a pending array's value goes.
    template: tuple
    # True when the index was written as a tuple, False for one item.
    whole: bool

    def __call__(self, view: np.ndarray, assigned: Any, *found: Any) -> None:
        """Assign into the part of the copy the target is a view of, through the computed index.

        Args:
            view: That part.
            assigned: The value assigned.
            *found: The values of the index's pending arrays, in the order of their slots.
        """
        parts = tuple(found[part.place] if isinstance(part, _Slot) else part for part in self.template)
        view[parts if self.whole else parts[0]] = assigned


def _is_number(node: Node) -> bool:
    """Say whether NumPy gives a number, not an array, for a node without dimensions.

    Args:
        node: A node of shape ().

    Returns:
        True for the result of a ufunc, a reduction, a product, an index or
        a search for one value: NumPy returns a scalar for those, and `s += v` on a scalar makes a
        new scalar and changes nothing that exists. False for what may be an
        array without dimensions, such as `np.zeros(())`, which is written in place.
    """
    if not node.pending:
        return isinstance(node.value, np.generic)
    return (isinstance(node.operation, np.ufunc)
            or node.kind in (rules.REDUCTION, rules.MATRIX_PRODUCT, rules.INDEXING, rules.SEARCH)
            or (node.kind == rules.VIEW and node.operation is operator.getitem))


def _may_defer_write(target: "LazyArray") -> bool:
    """Say whether a write into a lazy array can be a node of the graph instead of happening now.

    Args:
        target: The array written into.

    Returns:
        True when nothing outside the layer can know the array's memory (so
        nobody can be watching it for the write), no array made by
        `np.reshape` may be sharing it, and it is an array the layer defers
        operations on at all. Otherwise the write is made at once, in the
        memory, after everything pending.
    """
    cell = target._cell
    node = cell.node
    # `rules.is_admissible`: no length of zero and an element type the layer defers, remembered by the node's number.
    if not (not node.exposed and node.failure is None and not cell.doubted and not cell.shared_in_doubt()
            and node.shape != () and rules.is_admissible(node)):
        return False
    return not (_views_stand_alone and node.operation is None and _seen_by_a_view(node))


class _Held:
    """Stands where a node stands when `_seen_by_a_view` is tried on an array nothing else refers to."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        """Hold a value as a node holds its own."""
        self.value = value


def _references(node: Any) -> int:
    """Count the references to a node's value, this function's own among them.

    Args:
        node: A computed node, or what stands in for one.

    Returns:
        `sys.getrefcount` of the value.
    """
    return sys.getrefcount(node.value)


# What `_references` reports for a value that its node alone refers to. Measured, not assumed, as in `evaluate`.
_VALUE_ALONE = _references(_Held(np.zeros(1)))
# Whether a view taken at once may be alive that the layer keeps no record of. The C core takes such views once
# `at_once_below` has been given a limit, and they outlive the limit, so this is not cleared with it.
_views_stand_alone = False


def _seen_by_a_view(node: Node) -> bool:
    """Say whether a view the layer keeps no record of may be looking at a computed node's memory.

    A view the C core took at once holds NumPy's view and nothing that
    leads back to the array it is a view of. A write that waits could not
    be shown to it: when the write is computed the view's array still
    refers to the memory, so the write goes to a copy, and the view goes on
    showing what the array held. NumPy's view refers to the array it is a
    view of, and CPython counts references exactly. So a write is made at
    once, where every view sees it, when the array's memory is another
    array's (it is such a view itself) or when anything but the node refers
    to the array. A view the layer does keep a record of, and an `ndarray`
    the program holds, count too; the write is then made at once where it
    could have waited, which costs time and changes no value.

    Args:
        node: A computed node.

    Returns:
        True when a write into the node's array may not wait.
    """
    memory = node.value
    if type(memory) is not np.ndarray:
        return False
    if memory.base is not None:
        return True
    del memory
    return _references(node) > _VALUE_ALONE


def _defer_write(target: "LazyArray", name: str, perform: Callable[..., Any], operands: tuple,
                 frame: FrameType | None = None) -> None:
    """Make a write into a lazy array a node: the array's cell gets new contents, and nothing is computed.

    The node's value is a copy of what the cell held with the write made
    into it. Operations written before the write read the node the cell had,
    so they have the old contents whenever they are computed; the array, its
    other names and its views read the new node from here on.

    Args:
        target: The array written into: a cell's own array, or a view of it.
        name: What to call the node.
        perform: Makes the write: called with the part of the copy the
            target is a view of, then the operands' values.
        operands: What the write reads: lazy arrays, `ndarray`s, numbers.
        frame: The frame that called into the layer, when the caller has it; see `graph.deferred`.
    """
    cell = target._cell
    run = _Write(target._steps, perform)
    # Read before the cell changes: an operand that is a view of this array is a view of what it held.
    read, strict = _read(operands)
    if _core is not None and _core.write_now(target, perform, read):
        return  # Made in the array's own memory, at once: see `at_once_below`.
    strict = cell.node.strict or strict
    if len(graph.NOT_COMPUTED) >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    held = cell.node
    cell.node = graph.deferred(run, name, (held,) + read, {}, held.shape, held.dtype, rules.WRITE, strict, None,
                               held.spec, frame)


def _shape_of(operand: Any) -> tuple[int, ...] | None:
    """Return the shape of something a write reads, or None for what the layer does not look into.

    Args:
        operand: A lazy array, an `ndarray`, a NumPy scalar, a Python number, or anything else.

    Returns:
        The shape; () for a number; None for a list or another object.
    """
    if isinstance(operand, LazyArray):
        return operand._node.shape  # Not `.shape`, which computes when a length is not known.
    if isinstance(operand, (np.ndarray, np.generic)):
        return tuple(operand.shape)
    return () if isinstance(operand, rules.PYTHON_NUMBERS) else None


def _probe_of(operand: Any) -> Any:
    """Return a one-element stand-in for something a write reads, to try the write's types on.

    Args:
        operand: A lazy array, an `ndarray`, a NumPy scalar or a Python number.

    Returns:
        Zeros of its element type and number of dimensions; a number as it is.
    """
    return rules.probe(operand._node if isinstance(operand, LazyArray) else operand)


def _types_fit(call: Callable[..., Any], target: "LazyArray", operands: tuple) -> None:
    """Try a write on one-element stand-ins, so that a write of the wrong types raises on its line.

    Args:
        call: The write, taking the target then the operands.
        target: The array written into.
        operands: What the write reads.

    Raises:
        Exception: NumPy's, for a result that cannot be cast into the target.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        call(rules.probe(target._node), *(_probe_of(operand) for operand in operands))


def _fit_key(written_by: Any, target: "LazyArray", operands: tuple) -> tuple | None:
    """Write down everything that decides whether an element-by-element write fits, or decline.

    Args:
        written_by: What makes the write: the in-place operator, or the function called with `out=`.
        target: The array written into.
        operands: What the write reads.

    Returns:
        A tuple that two writes share only if `_fits_elementwise` answers
        them alike: what makes the write, then each operand and the target
        as `rules.operand_part` writes them down, which is the shape and
        element type of an array and what counts of a number. None when one
        of them cannot be written down so.
    """
    parts = [written_by]
    for operand in (*operands, target):
        # The operands first and the target last, the order `_fits_elementwise` reads them in: reading
        # a view of an array that was written since makes the view's nodes again.
        part = rules.operand_part(operand._node if isinstance(operand, LazyArray) else operand)
        if not part:
            return None
        parts.append(part)
    return tuple(parts)


def _fits_elementwise(call: Callable[..., Any], target: "LazyArray", operands: tuple, written_by: Any = None) -> bool:
    """Say whether an element-by-element write into an array can be deferred, raising if its types are wrong.

    Args:
        call: The write, taking the target then the operands.
        target: The array written into.
        operands: What the write reads.
        written_by: What stands for `call` in the memory of writes that fit,
            when `call` is a function made for this one write: the NumPy
            function it calls. None when `call` stands for itself.

    Returns:
        True when every operand has a shape the layer knows and they
        broadcast to the target's shape exactly. False sends the write the
        other way, where NumPy makes it at once and raises its own message
        if the shapes are wrong.

    Raises:
        Exception: NumPy's, for a result that cannot be cast into the target.
    """
    key = _fit_key(written_by or call, target, operands)
    if key is not None and key in _fitting:
        return True
    shapes = [_shape_of(operand) for operand in operands]
    if any(shape is None for shape in shapes):
        return False
    try:
        if np.broadcast_shapes(target._node.shape, *shapes) != target._node.shape:
            return False
    except ValueError:
        return False
    _types_fit(call, target, operands)
    if key is not None:
        if len(_fitting) >= FIT_MEMORY_LIMIT:
            _fitting.clear()
        _fitting.add(key)
    return True


def _index_parts(index: Any) -> tuple[Callable[[tuple], Any], list["LazyArray"], list] | None:
    """Take an index apart into what is known now and the lazy arrays whose values come later.

    Args:
        index: What was written between the brackets.

    Returns:
        (a function that puts values in and gives the index back, the lazy
        arrays still pending, the index's items with a `_Slot` for each of
        them). A lazy array that has its values is replaced by them. A
        pending one may be a mask or an array of integers; what the integers
        are is checked against the bounds when the graph is computed. Which
        of those an index may hold together is for `rules.is_array_index`
        to say. None when the index holds a pending lazy array of another
        type, or a lazy array inside a list.
    """
    items = index if isinstance(index, tuple) else (index,)
    pending: list[LazyArray] = []
    template: list[Any] = []
    for item in items:
        if not isinstance(item, LazyArray):
            if _lazies_in(item):
                return None
            template.append(item)
        elif not item._node.pending:
            template.append(item._node.value)
        elif item.dtype.kind not in "biu":
            return None
        else:
            template.append(_Slot(len(pending)))
            pending.append(item)

    def fill(found: tuple) -> Any:
        """Put the pending arrays' values, or what stands in for them, in their places."""
        parts = tuple(found[part.place] if isinstance(part, _Slot) else part for part in template)
        return parts if isinstance(index, tuple) else parts[0]
    return fill, pending, template


def _assignment_fits(target: "LazyArray", parts: tuple, value: Any) -> bool:
    """Say whether `x[index] = value` can be deferred, raising NumPy's error if the index or the shapes are wrong.

    The assignment is tried on arrays of an element type of no size, which
    take no memory: NumPy checks the index against the bounds and the value
    against the shape selected, and raises its own message on the line. A
    pending array in the index is stood in for as `rules.index_stand_in`
    says: zeros for integers, which are in bounds, and a mask that selects nothing.

    Args:
        target: The array assigned into.
        parts: What `_index_parts` returned for the index.
        value: What is assigned.

    Returns:
        False for a value the layer does not know the shape of, and for an
        index of pending arrays that `rules.is_array_index` does not accept.
        Through a mask that is still pending, True for a number, and for an
        array made from a selection by the same mask (`x[mask] = f(y[mask])`),
        which has as many elements as the mask selects whatever that number
        is; False for any other array, which fits or not depending on the
        mask's values. Through pending positions whose number is not known,
        such as `np.flatnonzero(m)`, the same.

    Raises:
        Exception: NumPy's, for an index out of bounds or a value that does not fit.
    """
    fill, pending, _ = parts
    shape = _shape_of(value)
    if shape is None or not graph.is_known(target._cell.node.shape):
        return False
    nodes = tuple(lazy._node for lazy in pending)
    if nodes and not rules.is_array_index(fill(nodes)):
        return False
    if not graph.is_known(shape):
        if len(pending) != 1 or rules.index_mark(nodes[0]) not in shape:
            return False
        # The mask is stood in for by one that selects nothing; so is the value's length along it.
        shape = tuple(length if graph.is_known((length,)) else 0 for length in shape)
    elif shape != () and any(rules.index_mark(node) is not None for node in nodes):
        return False
    ghost = np.empty(target._cell.node.shape, rules.NO_SIZE)
    for step in target._steps:
        ghost = step.of(ghost)
    masks = tuple(rules.index_stand_in(lazy._node) for lazy in pending)
    ghost[fill(masks)] = np.broadcast_to(np.zeros((), rules.NO_SIZE), shape)
    return True


def _assignment_key(target: "LazyArray", parts: tuple, value: Any) -> tuple | None:
    """Write down everything that decides whether `x[index] = value` fits, for an index that holds pending arrays, or decline.

    `_assignment_fits` asks NumPy, on stand-ins without memory. Its answer
    depends on the shape assigned into, on the index as `rules._key` writes
    an index down (each pending array by its shape and element type, each
    other item by its value, a length that is not known by a placeholder),
    and on the value's shape, in which a length that is not known gets the
    placeholder of the mask it came from, or one of its own. Not on any
    array's values: every array in such an index is stood in for.

    Args:
        target: The array assigned into.
        parts: What `_index_parts` returned for the index.
        value: What is assigned.

    Returns:
        A tuple that two assignments share only if `_assignment_fits`
        answers them alike. None when the array is a view by a step that is
        not plain, when its shape or the value's is not one the layer knows,
        or when the index holds what `rules._key` does not write down: a
        list or an `ndarray`, whose values NumPy checks against the bounds.
    """
    fill, pending, _ = parts
    whole = target._cell.node
    if not (whole.spec or graph.spec(whole)):
        return None
    held = None
    for step in target._steps:
        if not step.plain:
            return None
        held = step.shape
    shape = _shape_of(value)
    if shape is None:
        return None
    marks: dict[int, int] = {}
    index = rules._key(operator.getitem, (whole, fill(tuple(lazy._node for lazy in pending))), {}, marks)
    if index is None:
        return None
    written = tuple(length if length < graph.UNKNOWN_FLOOR else marks.setdefault(length, -len(marks) - 1) for length in shape)
    return (held, written, index)


def _select(target: "LazyArray", parts: tuple, frame: FrameType | None = None) -> "LazyArray | None":
    """Defer `x[index]` for an index that holds lazy arrays still pending.

    Args:
        target: The array indexed.
        parts: What `_index_parts` returned for the index; at least one lazy array is pending.
        frame: The frame that called into the layer, when the caller has it; see `graph.deferred`.

    Returns:
        A pending lazy array whose node reads the array and the index's
        arrays, or None when the rules decline and the indexing has to be
        made at once. Selected by a pending mask, it has a length that is
        not known.

    Raises:
        IndexError: NumPy's, for a mask of the wrong shape or index arrays that do not broadcast.
    """
    fill, pending, template = parts
    nodes = tuple(lazy._node for lazy in pending)
    index = fill(nodes)
    signature = _core.plan(operator.getitem, (target._node, index), {}) if _core is not None else None
    if signature is None or signature is _core.MISS:
        signature = rules.plan(operator.getitem, (target._node, index), {})
        if signature is None:
            return None
    else:
        signature = rules.Signature(*signature)

    run = _Select(tuple(template), isinstance(index, tuple))
    if len(graph.NOT_COMPUTED) >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    # What `_strict(target, pending)` says, without its search: the lazy arrays are these and no others.
    strict = target._node.strict
    for lazy in pending:
        strict = strict or lazy._node.strict
    node = graph.deferred(run, "getitem", (target._node,) + nodes, {}, signature.shape, signature.dtype, signature.kind,
                          strict, None, signature.spec, frame)
    made = LazyArray(node)
    if signature.shape != () and rules.is_doubtful_view(index):
        # `x[np.argmax(w)]`: a view of a row if the index is a number when computed. Not known yet.
        made._cell.doubted = True
        target._cell.doubt(made)
    return made


def _element(target: "LazyArray", index: Any) -> "LazyArray | None":
    """Read one element of an array that has its values, without a node.

    `for i in positions:` and `flags[i]` in a Python loop ask for one
    element per turn. Deferred, each would be a node, and a graph of its own
    as soon as the element is used as an index or tested. Nothing is pending
    under an array that has its values, and NumPy's answer for one element is
    a number, which is a copy and not a view, so nothing is lost by reading it.

    Args:
        target: The array indexed.
        index: What was written between the brackets, with every lazy array in it replaced by its values.

    Returns:
        A lazy array that has its values, holding NumPy's scalar: strict if
        the array is. None when the array is pending or failed, when its
        element type is not one the layer defers, or when the index is
        anything but one whole number for each dimension. Those take the
        ordinary road.

    Raises:
        IndexError: NumPy's, for a position out of bounds.
    """
    node = target._node
    if node.pending or node.failure is not None or node.dtype.kind not in rules.DEFERRED_KINDS:
        return None
    items = index if isinstance(index, tuple) else (index,)
    if len(items) != len(node.shape):
        return None
    if not all(isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)) for item in items):
        return None
    return LazyArray(graph.leaf(node.value[index], ELEMENT_SOURCE, strict=node.strict))


def _plain_item(target: "LazyArray", index: Any, frame: FrameType) -> "LazyArray | None":
    """Defer `x[index]` for an index of whole numbers and slices of them, without asking NumPy for the shape.

    A loop reads `x[i]` with another `i` on every turn, so the answer for
    one index is of no use for the next, and NumPy's own answer costs more
    than the node. `rules.basic_shape` works the shape out. What this
    function returns is what `__getitem__` would have made the long way: it
    is that road, for the indices where every step of it is known beforehand.

    Args:
        target: The array indexed.
        index: An `int`, a slice or a tuple, as written between the brackets.
        frame: The frame that wrote the index.

    Returns:
        One element of an array that has its values, as `_element` gives
        it; or a pending lazy array, a view of `target` unless the index
        leaves no dimension. None when this road does not apply and
        `__getitem__` has to take the long one: an index
        `rules.basic_shape` is not sure of, a shape with a length that is
        not known, an array the layer does not defer operations on.

    Raises:
        IndexError: NumPy's, for a position out of bounds in an array that has its values.
    """
    element = _element(target, index)
    if element is not None:
        return element
    node = target._node
    if not (node.spec or graph.spec(node)) or not rules.is_admissible(node):
        return None
    shape = rules.basic_shape(node.shape, index)
    if shape is None:
        return None
    if len(graph.NOT_COMPUTED) >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    kwargs: dict = {}
    made = graph.deferred(operator.getitem, "getitem", (node, index), kwargs, shape, node.dtype, rules.VIEW, node.strict,
                          None, 0, frame)
    if shape == ():
        return LazyArray(made)  # A number, which is a copy, not a view.
    step = _Step(operator.getitem, "getitem", (index,), kwargs, shape, node.dtype, made.origin, True)
    return LazyArray(made, target._cell, target._steps + (step,))


def _plain_assignment_fits(target: "LazyArray", index: Any, value: Any) -> bool:
    """Say whether `x[index] = value` is known to fit without asking NumPy.

    Args:
        target: The array assigned into.
        index: What was written between the brackets.
        value: What is assigned.

    Returns:
        True when the index is one `rules.basic_shape` is sure of, every
        view between the cell and `target` is plain, and the value's shape
        broadcasts to the shape selected: the cases where `_assignment_fits`
        is certain to say yes. False for everything else, which says nothing
        about the assignment: `_assignment_fits` then asks NumPy, and
        NumPy raises its own error if there is one.
    """
    kind = type(index)
    if kind is not int and kind is not slice and kind is not tuple:
        return False
    whole = target._cell.node
    if not (whole.spec or graph.spec(whole)):
        return False
    held = whole.shape
    for step in target._steps:
        if not step.plain:
            return False
        held = step.shape
    selected = rules.basic_shape(held, index)
    if selected is None:
        return False
    shape = _shape_of(value)
    return shape is not None and rules.broadcasts_into(shape, selected)


def _same_index(first: Any, second: Any) -> bool:
    """Say whether two basic indices are the same.

    Args:
        first: One index.
        second: Another.

    Returns:
        True for equal integers, slices and tuples of them. False for
        anything that cannot be compared plainly, such as an index holding an array.
    """
    try:
        return type(first) is type(second) and bool(first == second)
    except (ValueError, TypeError):
        return False


# The innermost dimension below which an in-place operator made at once is made as an expression and an assignment
# (`_is_short_rows`): the one below which NumPy's call for each row costs more than the row's arithmetic.
SHORT_ROW = 16
# A quicker `part[...] = values` for two float64 arrays of one shape whose rows are a few elements, or None for
# NumPy's own: `fuse.enable()` sets it.
assign_rows: Callable[[np.ndarray, np.ndarray], None] | None = None


def _is_short_rows(node: Node, other: Any) -> bool:
    """Say whether an in-place operator about to be made at once is better made as an expression and an assignment.

    `x[:, :2] += made` on 100,000 rows takes NumPy 861 µs: its loop is
    called for each row of two elements. The sum as an expression of its
    own is a loop of the planner's module over the rows, and its rows are
    then assigned, each in about a fifth of that. A warning the sum raises
    is NumPy's for the same operation on the same line; an error leaves the
    array as it was.

    Args:
        node: The node of the array written into.
        other: What the operator's right-hand side is.

    Returns:
        True when a planner and its assignment are set, the array has its
        values, float64, in two dimensions or more with an innermost shorter
        than `SHORT_ROW` and not in one run, and is large enough for a call
        on it to wait; and `other` is a float64 lazy array of the same shape.
    """
    if evaluate.planner is None or assign_rows is None or not _at_once_limit:
        return False
    value = node.value
    if node.operation is not None or type(value) is not np.ndarray or value.dtype != np.float64:
        return False
    if value.ndim < 2 or value.shape[-1] >= SHORT_ROW or value.flags.c_contiguous or value.size < _at_once_limit:
        return False
    if type(other) is not LazyArray:
        return False
    waiting = other._node
    return waiting.shape == value.shape and waiting.dtype == np.float64


def _in_place(apply: Callable[[Any, Any], Any], symbol: str, ufunc: np.ufunc) -> Callable[..., Any]:
    """Build the method for an in-place operator, which is a write.

    Args:
        apply: The function of the `operator` module, such as `operator.iadd`.
        symbol: The operator as written, for the node, the counters and the log.
        ufunc: The ufunc the operator stands for.

    Returns:
        A method that makes the write a node of the graph when
        `_may_defer_write` allows it, and the array is returned unchanged in
        identity, so every name for it reads the new contents. For a number,
        which cannot be written, the method gives a new lazy array, as
        `s += v` gives a new scalar in NumPy. Otherwise everything pending
        is computed and NumPy applies the operator in the memory.
    """
    def perform(view: np.ndarray, other: Any) -> None:
        """Apply the operator in place to the part of the copy the target is a view of."""
        apply(view, other)

    def write(self, other):
        """Apply the in-place operator with NumPy's meaning."""
        node = self._node
        if node.shape == () and _is_number(node):
            return _defer_or_call(ufunc, ufunc.__name__, f"np.{ufunc.__name__}", (self, other), sys._getframe(1))
        if _may_defer_write(self) and _fits_elementwise(apply, self, (other,)):
            _defer_write(self, symbol, perform, (other,), sys._getframe(1))
            return self
        graph.fallback(graph.WRITE, symbol)
        trigger = Trigger(graph.WRITE, symbol)
        if _is_short_rows(node, other):
            # NumPy would make the operator a row at a time. The operator's own expression is made instead, which
            # a planner computes by a loop over the rows, and the result's rows are assigned.
            made = _defer_or_call(ufunc, ufunc.__name__, f"np.{ufunc.__name__}", (self, other), sys._getframe(1))
            evaluate.compute_pending(trigger)
            result = _valued(made)
            if type(result) is np.ndarray and result.shape == node.value.shape and result.dtype == node.value.dtype:
                assign_rows(node.value, result)
                return self
        evaluate.compute_pending(trigger)
        evaluate.compute([self._node], trigger)
        value = self._node.value
        result = apply(value, _valued(other))
        return self if result is value else LazyArray(graph.leaf(result, symbol, graph.AT_ONCE, self._node.strict))
    write.road = ("inplace", apply, symbol, perform)
    return write


def _write_out(function: Callable[..., Any], call: str, args: tuple, target: Any) -> bool:
    """Defer a call that writes its result into `out=`, when that is safe.

    Args:
        function: A ufunc, or one of OUT_FUNCTIONS.
        call: What to call the node.
        args: The call's positional arguments.
        target: What was given as `out`.

    Returns:
        True when the write was made a node. False when it has to be made at once.

    Raises:
        Exception: NumPy's, for a result that cannot be cast into `out`.
    """
    into = _Into(function)
    if not isinstance(target, LazyArray) or not _may_defer_write(target) or not _fits_elementwise(into, target, args, function):
        return False
    _defer_write(target, call, into, args)
    return True


def _reduction(function: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Build the method form of a reduction.

    Args:
        function: The NumPy function whose rule gives the shape and element type, such as `np.sum`.
        name: The method's name, such as "sum". `dot` and `copy` are built the same way.

    Returns:
        A method that defers the call as it was written: the node calls the
        same method on the computed value, with the arguments that were
        given, given the same way. `x.sum()` is not turned into
        `np.sum(x, axis=None, keepdims=False)`. The values would be the same,
        but code that looks at how a call is spelled would not treat it the
        same: the kernels of `hotkernels.install()` take `x.sum()` and leave a
        sum that names an axis or `keepdims` to NumPy.
    """
    method = _method(name)

    def reduce(self, *args, **kwargs):
        """Reduce the array, with the arguments `ndarray`'s method of this name takes."""
        operands = (self,) + args
        deferred = _try_defer(method, name, operands, kwargs, function, sys._getframe(1))
        return deferred if deferred is not None else _at_once(method, f"ndarray.{name}", operands, kwargs)
    reduce.road = ("reduction", method, name, function)
    return reduce


class LazyArray:
    """An array whose values may not have been computed yet.

    Its shape and element type are known from the start, but for one length
    of what a pending mask selects. It is not an `ndarray`: an `ndarray` owns
    memory that anything may read at any time, and a pending array has none.
    """

    __slots__ = ("_held", "_steps", "_seen", "__weakref__")
    # So that type(x) prints the name a user imports, not the file the class is written in.
    __module__ = "prototypes.lazy"
    # Arrays compare element by element, so they cannot be hashed. As for `ndarray`.
    __hash__ = None

    def __init__(self, node: Node, cell: _Cell | None = None, steps: tuple[_Step, ...] = ()) -> None:
        """Make a lazy array.

        Args:
            node: The node of the graph the array stands for now.
            cell: The cell the array is a view into, or None for an array of its own, which gets a new cell.
            steps: For a view, the view operations from the cell's contents to this array.
        """
        if cell is None:
            self._held = _Cell(node)
            self._steps = steps
            self._seen = None
        else:
            self._held = cell
            self._steps = steps
            # For a view: the cell's contents the node was made from, and the node. Nothing reads it for an array of its own.
            self._seen = (cell.node, node)

    @property
    def _cell(self) -> _Cell:
        """The cell the array looks into.

        `_held` is the cell, for every array this file makes. The C core
        makes one more form, for the result of a call it made at once: the
        array holds the value NumPy gave and nothing else, with one of
        `VALUED` in `_seen`. Such an array has no node and no cell, because
        most are read by the next small call and dropped, and the three
        objects cost more than the call. A view taken at once is such an
        array too, holding NumPy's view: see `_may_defer_write` for what
        that asks of a write. The first thing that asks for the
        cell or the node gets them made here, as `graph.leaf` makes a node
        for any array that has its values, and from then on the array is
        like every other.

        Returns:
            The cell, made now for an array that held its value alone.
        """
        held = self._held
        if type(held) is not _Cell:
            kind, strict = self._seen
            held = self._held = _Cell(graph.leaf(held, kind=kind, strict=strict))
            self._seen = None
            _track(self)
        return held

    @property
    def _node(self) -> Node:
        """The node the array stands for now.

        Returns:
            The cell's contents for an array of its own. For a view, the
            view of them: made again, as deferred nodes, when a write has
            given the cell new contents since the view was last read.
        """
        if not self._steps:
            return self._cell.node
        contents, node = self._seen
        if contents is not self._cell.node:
            node = self._cell.node
            for step in self._steps:
                node = graph.deferred(step.operation, step.name, (node,) + step.tail, step.kwargs, step.shape, step.dtype,
                                      rules.VIEW, node.strict, step.origin)
            self._seen = (self._cell.node, node)
        return node

    def _expose(self) -> None:
        """Note that the array's memory is known outside the layer from now on, so a write into it is made at once."""
        self._cell.node.exposed = True

    # ---- Known without computing.

    def _settled_shape(self) -> tuple[int, ...]:
        """Return the shape with every length known.

        Returns:
            The node's shape. When one of its lengths is how many elements a
            pending mask selects, the array is computed first, and the log
            says so: the length is a value.

        Raises:
            NotGathered: For such a shape, on a strict array.
        """
        node = self._node
        if not graph.is_known(node.shape):
            what = "The shape of what a mask selects"
            _hand_out(what, node.strict, False, [node], f"{what} needs the mask's values: call .gather() first")
        return node.shape

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape NumPy would give the array."""
        return self._settled_shape()

    @shape.setter
    def shape(self, shape: Any) -> None:
        """Give the array another shape where it is, as `x.shape = s` does to an `ndarray`.

        Nothing is computed and nothing is copied. The array becomes a view
        of its own contents in the new shape: other arrays that look at the
        same memory keep their shapes, and a write through either is seen by
        both, as with NumPy.

        Args:
            shape: The new shape: a whole number or a sequence of them, one of which may be -1.

        Raises:
            ValueError: With NumPy's words, when the shape does not hold the array's elements.
            AttributeError: NumPy's own, when the array is computed, if its
                memory cannot be read in the new shape without a copy.
        """
        size = int(np.prod(self._settled_shape(), dtype=np.int64))
        wanted = _resolved_shape(shape, size)
        node = self._node
        made = graph.deferred(_set_shape, SHAPE_ASSIGNMENT, (node, wanted), {}, wanted, node.dtype, rules.VIEW,
                              node.strict, None, 0, sys._getframe(1))
        step = _Step(_set_shape, SHAPE_ASSIGNMENT, (wanted,), {}, wanted, node.dtype, made.origin)
        self._steps = self._steps + (step,)
        self._seen = (self._cell.node, made)

    @property
    def dtype(self) -> np.dtype:
        """The element type NumPy would give the array: NumPy's own `dtype` object."""
        return self._node.dtype

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self._node.shape)

    @property
    def size(self) -> int:
        """The number of elements."""
        return int(np.prod(self._settled_shape(), dtype=np.intp))

    @property
    def itemsize(self) -> int:
        """The size of one element, in bytes."""
        return self._node.dtype.itemsize

    @property
    def nbytes(self) -> int:
        """The size of the values, in bytes, once computed."""
        return self.size * self.itemsize

    @property
    def T(self) -> "LazyArray":
        """The transpose, deferred."""
        return _view(np.transpose, (self,), sys._getframe(1))

    def __len__(self) -> int:
        """Return the length of the first dimension, as for an `ndarray`."""
        if not self._node.shape:
            raise TypeError("len() of unsized object")
        return self._settled_shape()[0]

    def __repr__(self) -> str:
        """Describe the array without computing it and without showing values."""
        return f"LazyArray({self.dtype}, shape={graph.shape_text(self._node.shape)}, {self._node.state})"

    def explain(self) -> str:
        """Return the pending graph of this array as text. Computes nothing."""
        return explain.render(self._node)

    def graph(self) -> record.Graph:
        """Return the pending graph of this array as an object. Computes nothing.

        Returns:
            Plain data: no array and no live node. It can be written as JSON
            or opened as a page in the browser.
        """
        return record.graph_of(self._node)

    # ---- The way to ask for values.

    def gather(self) -> Any:
        """Compute the array.

        Returns:
            What NumPy would have returned for it: an `ndarray`, or a NumPy
            scalar where NumPy gives one. It is the array the lazy array
            holds, not a copy, and the lazy array stays usable.

        Raises:
            Exception: What computing it raised, with a note naming the line
                that wrote the operation.
        """
        evaluate.compute([self._node], Trigger(graph.GATHER, "gather"))
        self._expose()
        return self._node.value

    # ---- Everything else that needs values: computed and logged, or refused for a strict array.

    __float__ = _leaving("float()", float)
    __int__ = _leaving("int()", int)
    __complex__ = _leaving("complex()", complex)
    __bool__ = _leaving("A truth test (if, while, and, or, not)", bool)
    __index__ = _leaving("Use as an index or in range()", operator.index)
    __contains__ = _leaving("The in operator", operator.contains)
    # np.asarray(x), ndarray[i] = x, and every library that asks NumPy for an
    # array. The receiver holds the array and may write: everything pending first.
    __array__ = _leaving("Conversion to an ndarray", _as_array, everything=True)

    def __iter__(self) -> Iterator["LazyArray"]:
        """Iterate over the first dimension. No value is needed: each item is a lazy view.

        Returns:
            An iterator of `self[0]`, `self[1]`, ..., each deferred. Over one
            dimension of an array that has its values, each is an element
            that has its value, as `_element` says.

        Raises:
            TypeError: For an array without dimensions, as NumPy raises.
        """
        if not self._node.shape:
            raise TypeError("iteration over a 0-d array")
        return (self[position] for position in range(self._settled_shape()[0]))

    def __format__(self, spec: str) -> str:
        """Format the description, or with a specification the value.

        Args:
            spec: The format specification. Empty for `f"{x}"`.

        Returns:
            The description, as `print(x)` shows it, for an empty
            specification. For one such as `.3f`, the value formatted with it.

        Raises:
            NotGathered: For a specification, on a strict array.
        """
        if not spec:
            return repr(self)
        what = f"Formatting with '{spec}'"
        _hand_out(what, self._node.strict, False, [self._node], f"{what} needs the values of a lazy array: call .gather() first")
        return format(self._node.value, spec)

    def __getattr__(self, name: str) -> Any:
        """Answer a method this class does not define with `ndarray`'s, made at once.

        Args:
            name: The attribute asked for.

        Returns:
            For a method of `ndarray`, a function that makes the call at once
            on the gathered array (after computing everything pending, if the
            method writes).

        Raises:
            AttributeError: For a name `ndarray` lacks, and for every
                double-underscore name. NumPy looks for `__array_interface__`
                and `__array_struct__` before it calls `__array__`; answering
                those would hand it the memory without the layer noticing.
            NotGathered: On a strict array, for an attribute that is not a
                method (`x.flat`, `x.real`, `x.data`): those are the values
                themselves. Otherwise the array is computed, with everything
                else pending, and the attribute of the result is returned.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        attribute = getattr(np.ndarray, name)
        if not callable(attribute):
            what = f"The attribute .{name}"
            _hand_out(what, self._node.strict, True, [self._node], f"{what} needs the values of a lazy array: call .gather() first")
            self._expose()
            return getattr(self._node.value, name)
        method, call = _method(name), f"ndarray.{name}"
        if name in WRITING_METHODS:
            return lambda *args, **kwargs: _write(method, call, (self,) + args, kwargs, (self,))
        return lambda *args, **kwargs: _at_once(method, call, (self,) + args, kwargs)

    # ---- NumPy's two protocols.

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs: Any, out: tuple | None = None, **kwargs: Any) -> Any:
        """Receive every ufunc call that involves a lazy array.

        `np.exp(x)`, `np.add(a, x)`, `np.add.reduce(x)` arrive here, and so do
        `a + x` and `a += x` for an `ndarray` `a`, whose operators call the ufunc.

        Args:
            ufunc: The ufunc.
            method: "__call__" for a plain call, or the ufunc method: "reduce", "accumulate", "outer", "at".
            *inputs: The ufunc's inputs as written.
            out: The output arrays, when the call names any.
            **kwargs: Any other keyword of the call.

        Returns:
            A pending lazy array for a call that is deferred; the output for
            a call that writes; otherwise the result made at once.
        """
        if method == "__call__" and out is None:
            deferred = _try_defer(ufunc, ufunc.__name__, inputs, kwargs, None, sys._getframe(1))
            return deferred if deferred is not None else _at_once(ufunc, f"np.{ufunc.__name__}", inputs, kwargs)
        call = f"np.{ufunc.__name__}" if method == "__call__" else f"np.{ufunc.__name__}.{method}"
        if out is not None:
            if method == "__call__" and len(out) == 1 and not kwargs and _write_out(ufunc, ufunc.__name__, inputs, out[0]):
                return out[0]
            _write(getattr(ufunc, method), call, inputs, {**kwargs, "out": out}, out)
            return out[0] if len(out) == 1 else out
        if method == WRITING_UFUNC_METHOD:
            return _write(getattr(ufunc, method), call, inputs, kwargs, inputs[:1])
        return _at_once(getattr(ufunc, method), call, inputs, kwargs)

    def __array_function__(self, func: Callable[..., Any], types: tuple, args: tuple, kwargs: dict) -> Any:
        """Receive every call of a NumPy function that involves a lazy array.

        `np.sum(x)`, `np.where(c, x, y)`, `np.transpose(x)`, `np.linalg.solve(m, x)`
        and `np.concatenate([a, x])` arrive here, with the function and its
        arguments as written.

        Args:
            func: The public NumPy function.
            types: The array types NumPy found among the arguments. Not used.
            args: The positional arguments as written.
            kwargs: The keyword arguments as written.

        Returns:
            An answer from the shape alone; a pending lazy array; the output
            of a call that writes; or the result made at once.
        """
        if func not in HANDLED_APART and (not kwargs or kwargs.get("out") is None):
            # The common case, which none of the roads below takes.
            deferred = _try_defer(func, func.__name__, args, kwargs, None, sys._getframe(1))
            return deferred if deferred is not None else _at_once(func, f"np.{func.__name__}", args, kwargs)
        call = f"np.{func.__name__}"
        if func in SHAPE_QUESTIONS and len(args) == 1 and not kwargs and isinstance(args[0], LazyArray):
            return getattr(args[0], SHAPE_QUESTIONS[func])
        target = kwargs.get("out")
        if target is not None:
            if func in OUT_FUNCTIONS and set(kwargs) == {"out"} and _write_out(func, func.__name__, args, target):
                return target
            _write(func, call, args, kwargs, target if isinstance(target, tuple) else (target,))
            return target
        if func in WRITING_FUNCTIONS:
            return _write(func, call, args, kwargs, args[:1])
        if (func in SHAPED_LIKE and isinstance(args[0], LazyArray) and graph.is_known(args[0]._node.shape)
                and not _lazies_in((args[1:], kwargs))):
            # Only the shape and element type of the argument are used, and
            # those are known: the new array is made without computing it.
            like = rules.ghost(args[0]._node)
            return LazyArray(graph.leaf(func(like, *args[1:], **kwargs), call, strict=args[0]._node.strict))
        if func is np.nonzero and not kwargs:
            parts = _defer_nonzero(func, func.__name__, args)
            return parts if parts is not None else _at_once(func, call, args, kwargs)
        deferred = _try_defer(func, func.__name__, args, kwargs)
        return deferred if deferred is not None else _at_once(func, call, args, kwargs)

    # ---- Indexing.

    def __getitem__(self, index: Any) -> "LazyArray":
        """Index the array.

        Args:
            index: What was written between the brackets.

        Returns:
            A deferred view for integers, slices, `None` and `...`, and a
            deferred selection for arrays of integers and for masks. A lazy
            array in the index that has its values is replaced by them. One
            still pending becomes an operand of the node: an array of
            integers is checked against the bounds when the graph is
            computed, and a mask gives a result whose length along the
            mask is not known until then. One element of an array that has
            its values is read on the spot, as `_element` says: a lazy array
            that has its values, and no node.
        """
        kind = type(index)
        if kind is int or kind is slice or kind is tuple:
            made = _plain_item(self, index, sys._getframe(1))
            # Not `made or ...`: a truth test of a lazy array would ask for its value.
            if made is not None:
                return made
        parts = _index_parts(index)
        if parts is None or not parts[1]:
            settled = index if parts is None else parts[0](())
            element = None if parts is None else _element(self, settled)
            # Not `element or ...`: a truth test of a lazy array would ask for its value.
            return element if element is not None else _defer_or_call(operator.getitem, "getitem", "x[index]", (self, settled))
        selected = _select(self, parts, sys._getframe(1))
        return selected if selected is not None else _at_once(operator.getitem, "x[index]", (self, index), {})

    def __setitem__(self, index: Any, value: Any) -> None:
        """Assign into the array: a write, with NumPy's meaning.

        Args:
            index: What was written between the brackets.
            value: What to assign. A lazy array's values are used.
        """
        if isinstance(value, LazyArray) and value._cell is self._cell and self._is_view_at(value, index):
            # `x[i] += v`: Python takes the view, updates it, and assigns it back to where it already is.
            return
        if _may_defer_write(self):
            if _plain_assignment_fits(self, index, value):
                _defer_write(self, "setitem", _Assign(index), (value,), sys._getframe(1))
                return
            parts = _index_parts(index)
            key = None if parts is None else _assignment_key(self, parts, value)
            if parts is not None and ((key is not None and key in _assignable) or _assignment_fits(self, parts, value)):
                if key is not None:
                    if len(_assignable) >= FIT_MEMORY_LIMIT:
                        _assignable.clear()
                    _assignable.add(key)
                assign = _AssignThrough(tuple(parts[2]), isinstance(index, tuple))
                _defer_write(self, "setitem", assign, (value, *parts[1]), sys._getframe(1))
                return
        _write(operator.setitem, "x[index] = v", (self, index, value), {}, (self,))

    def _is_view_at(self, other: "LazyArray", index: Any) -> bool:
        """Say whether another lazy array is this array's own view at an index.

        Args:
            other: A lazy array of the same cell.
            index: An index into this array.

        Returns:
            True when `other` is `self[index]` as a tracked view.
        """
        if len(other._steps) != len(self._steps) + 1 or any(a is not b for a, b in zip(other._steps, self._steps)):
            return False
        last = other._steps[-1]
        return last.operation is operator.getitem and _same_index(last.tail[0], index)

    # ---- Methods the graph knows. Any other method is `ndarray`'s, through `__getattr__`.

    dot = _reduction(np.dot, "dot")
    copy = _reduction(np.copy, "copy")
    sum = _reduction(np.sum, "sum")
    mean = _reduction(np.mean, "mean")
    max = _reduction(np.max, "max")
    min = _reduction(np.min, "min")
    prod = _reduction(np.prod, "prod")
    var = _reduction(np.var, "var")
    std = _reduction(np.std, "std")
    any = _reduction(np.any, "any")
    all = _reduction(np.all, "all")
    argmax = _reduction(np.argmax, "argmax")
    argmin = _reduction(np.argmin, "argmin")
    cumsum = _reduction(np.cumsum, "cumsum")
    cumprod = _reduction(np.cumprod, "cumprod")
    searchsorted = _reduction(np.searchsorted, "searchsorted")
    round = _reduction(np.round, "round")
    astype = _reduction(np.ndarray.astype, "astype")

    def view(self, *args: Any, **kwargs: Any) -> "LazyArray":
        """Give a new array on the same memory: `ndarray.view`, with a lazy array's own type understood.

        `x.view()`, `x.view(np.ndarray)` and `x.view(type(x))` ask for the
        same elements with no change of element type. On an array of one
        dimension or more that is the view `x[...]`, which NumPy lays out
        the same way, and it is deferred as that index is: the result
        shares the array's cell, so a write into either is seen through the
        other and can wait in the graph. `type(x)` is `LazyArray`; see
        `_view_request` for what it means here.

        Args:
            *args: `dtype` and `type`, when given by position.
            **kwargs: `dtype` and `type`, when given by keyword.

        Returns:
            A lazy array that is a view of this one, for the three calls
            above: pending, or NumPy's view where the index `x[...]` would
            be taken at once. For anything else the call is made at once, by
            NumPy's own method on the computed array: a change of element
            type, another array type (`np.matrix`, a masked array), an
            array without dimensions, whose view is a number when the value
            is one, an array whose value is of a subclass of `ndarray`, and
            an array the layer defers no operation on.

        Raises:
            Exception: NumPy's own, for arguments `ndarray.view` does not take.
        """
        whole, args, kwargs = _view_request(args, kwargs)
        if whole and _is_viewed_by_index(self):
            return self[WHOLE_INDEX]
        return _at_once(_method("view"), VIEW_CALL, (self,) + args, kwargs)

    def nonzero(self) -> "tuple[LazyArray, ...]":
        """Give the positions of the elements that are not zero, deferred.

        Returns:
            One pending lazy array per dimension, each of a length that is
            not known until it is computed. For an array without dimensions
            the call is made at once, and NumPy raises.
        """
        method = _method("nonzero")
        parts = _defer_nonzero(method, "nonzero", (self,), ruled_as=np.nonzero)
        return parts if parts is not None else _at_once(method, "ndarray.nonzero", (self,), {})

    def ravel(self, *args: Any, **kwargs: Any) -> "LazyArray":
        """Give the elements in one dimension, deferred.

        Args:
            *args: `order`, when given by position.
            **kwargs: `order`, when given by keyword.

        Returns:
            A pending lazy array in NumPy's default order. With an `order`,
            the call is made at once.
        """
        return np.ravel(self, *args, **kwargs)

    def reshape(self, *shape: Any) -> "LazyArray":
        """Give the array a new shape, deferred.

        Args:
            *shape: The new shape, as one tuple or as separate integers.

        Returns:
            A pending lazy array of that shape.
        """
        return _view(np.reshape, (self, shape[0] if len(shape) == 1 else shape), sys._getframe(1))

    def transpose(self, *axes: Any) -> "LazyArray":
        """Permute the axes, deferred.

        Args:
            *axes: The new order, as one tuple or as separate integers; none to reverse.

        Returns:
            A pending lazy array.
        """
        if not axes:
            return _view(np.transpose, (self,), sys._getframe(1))
        return _view(np.transpose, (self, axes[0] if len(axes) == 1 and not isinstance(axes[0], int) else axes), sys._getframe(1))

    def swapaxes(self, first: int, second: int) -> "LazyArray":
        """Exchange two axes, deferred.

        Args:
            first: One axis.
            second: The other.

        Returns:
            A pending lazy array.
        """
        return _view(np.swapaxes, (self, first, second), sys._getframe(1))

    # ---- Operators.

    __add__, __radd__ = _binary(np.add), _binary(np.add, reflected=True)
    __sub__, __rsub__ = _binary(np.subtract), _binary(np.subtract, reflected=True)
    __mul__, __rmul__ = _binary(np.multiply), _binary(np.multiply, reflected=True)
    __truediv__, __rtruediv__ = _binary(np.true_divide), _binary(np.true_divide, reflected=True)
    __floordiv__, __rfloordiv__ = _binary(np.floor_divide), _binary(np.floor_divide, reflected=True)
    __mod__, __rmod__ = _binary(np.remainder), _binary(np.remainder, reflected=True)
    __pow__ = _binary(np.power, written=operator.pow)
    __rpow__ = _binary(np.power, reflected=True, written=operator.pow)
    __matmul__, __rmatmul__ = _binary(np.matmul), _binary(np.matmul, reflected=True)
    __and__, __rand__ = _binary(np.bitwise_and), _binary(np.bitwise_and, reflected=True)
    __or__, __ror__ = _binary(np.bitwise_or), _binary(np.bitwise_or, reflected=True)
    __xor__, __rxor__ = _binary(np.bitwise_xor), _binary(np.bitwise_xor, reflected=True)
    __lt__, __le__ = _binary(np.less), _binary(np.less_equal)
    __gt__, __ge__ = _binary(np.greater), _binary(np.greater_equal)
    __eq__, __ne__ = _binary(np.equal), _binary(np.not_equal)
    __neg__, __pos__ = _unary(np.negative), _unary(np.positive)
    __abs__, __invert__ = _unary(np.absolute), _unary(np.invert)
    __iadd__ = _in_place(operator.iadd, "x += y", IN_PLACE_UFUNCS[operator.iadd])
    __isub__ = _in_place(operator.isub, "x -= y", IN_PLACE_UFUNCS[operator.isub])
    __imul__ = _in_place(operator.imul, "x *= y", IN_PLACE_UFUNCS[operator.imul])
    __itruediv__ = _in_place(operator.itruediv, "x /= y", IN_PLACE_UFUNCS[operator.itruediv])


# The roads the C core can stand on, beside the operators and reductions that say so where they are made.
LazyArray.__array_ufunc__.road = ("array_ufunc",)
LazyArray.__array_function__.road = ("array_function", HANDLED_APART, None, OUT_FUNCTIONS)
LazyArray.__getitem__.road = ("getitem",)
LazyArray.__setitem__.road = ("setitem",)
LazyArray.reshape.road = ("reshape", np.reshape, "reshape")
LazyArray.transpose.road = ("transpose", np.transpose, "transpose")
LazyArray.swapaxes.road = ("swapaxes", np.swapaxes, "swapaxes")
LazyArray.ravel.road = ("ravel", np.ravel, "ravel")

def _no_tracking(made: LazyArray) -> None:
    """Do nothing: what `_track` is while every lazy array is known to the garbage collector from the start.

    Args:
        made: A lazy array that has just been given its cell.
    """


# Called with a lazy array that held its value alone and has just been given its cell. The C core makes such an
# array unknown to the garbage collector, since it can be part of no cycle; with a cell it can, so the core's
# function makes it known. Python makes no such array, and without a core there is nothing to do.
_track: Callable[[LazyArray], None] = _no_tracking
# `graph.deferred` as Python has it, to put back when the C core is taken away.
_python_deferred = graph.deferred
# The C core in use, or None: every call then goes the Python way.
_core: Any = None
# The limit `at_once_below` was last given, for the roads Python takes; 0 for none.
_at_once_limit = 0
# The Python methods of `LazyArray` that a road of the core stands in for at present, by name.
_stood_in_for: dict[str, Callable[..., Any]] = {}
# The same for the classes of the stand-in for the module: (the class, the name, the method).
_stood_in_for_elsewhere: list[tuple[type, str, Callable[..., Any]]] = []


def _set_up(core: Any) -> None:
    """Tell the C core the layer's classes, memories and constants.

    Args:
        core: The module built from `_core.c`.
    """
    from . import namespace  # Imports this module, so not at the top.
    core.setup(Node=Node, LazyArray=LazyArray, Cell=_Cell, Step=_Step, ufunc=np.ufunc, ndarray=np.ndarray,
               graph_names=vars(graph), rules_names=vars(rules), numbers=graph._numbers, pending=graph._pending,
               forget=graph._forget, counts=graph.COUNTS, origins=graph._origins, specs=graph._specs, plans=rules._plans,
               admissible=rules._admissible_specs, caller=graph.caller, errstate=graph.errstate, view_kind=rules.VIEW,
               sure_views=rules.SURE_VIEWS, fractional_scalars=rules._FRACTIONAL_SCALARS, nonzeros=rules.NONZEROS,
               getitem=operator.getitem, count_nonzero=np.count_nonzero, unknown_floor=graph.UNKNOWN_FLOOR, Select=_Select,
               slots=tuple(_Slot(place) for place in range(SLOTS_MADE)), Write=_Write, Assign=_Assign, Into=_Into, AssignThrough=_AssignThrough,
               assignable=_assignable, types=rules._types, keep=rules.keep, spec_of=graph.spec_of, Origin=graph.Origin, dtype=np.dtype, views=rules.VIEWS, reductions=rules.REDUCTIONS, positions=rules.POSITIONS,
               reduction_kind=rules.REDUCTION, ravel=np.ravel, empty=np.empty, no_size=rules.NO_SIZE,
               element_type=rules._element_type, small_whole=rules.SMALL_WHOLE, large_whole=rules.LARGE_WHOLE,
               spec_numbers=graph._spec_numbers, Signature=rules.Signature,
               matmul=np.matmul, matrix_product=rules.MATRIX_PRODUCT,
               internal_files=graph._internal_files, elementwise=rules.ELEMENTWISE, fitting=_fitting,
               write_kind=rules.WRITE, LazyCallable=namespace.LazyCallable, LazyNumPy=namespace.LazyNumPy,
               at_once=graph.AT_ONCE, handled_apart=HANDLED_APART, shaped_like=SHAPED_LIKE,
               fresh_functions=FRESH_FUNCTIONS, never_deferred=NEVER_DEFERRED, at_once_result=_made_lazy, valued=VALUED, generic=np.generic, in_place_ufuncs=IN_PLACE_UFUNCS,
               deferred_kinds="".join(sorted(rules.DEFERRED_KINDS)))


def use_core(core: Any) -> None:
    """Take the C core for writing nodes, or put it away.

    With a core, each method of `LazyArray` that carries a `road` is
    replaced by a `Road` of the core, which takes the calls whose answers
    are remembered and hands every other to that method; `graph.deferred`
    is the core's; and `_try_defer` asks the core first. The Python
    functions stay the definition of what happens: the core is their quick
    road, and without it nothing but the time is different.

    Args:
        core: The module built from `_core.c`, or None for Python alone.
            A module of another version than this file was written against
            is put away like None.
    """
    global _core, _track
    from . import namespace  # Imports this module, so not at the top.
    for name, method in _stood_in_for.items():
        setattr(LazyArray, name, method)
    _stood_in_for.clear()
    for owner, name, method in _stood_in_for_elsewhere:
        setattr(owner, name, method)
    _stood_in_for_elsewhere.clear()
    graph.deferred = _python_deferred
    evaluate.compute_plain = evaluate.run_nodes = evaluate.needed = None
    graph.release = None
    if _core is not None:
        at_once_below(0)
        _core.enable(False)
    _core = None
    if core is None or getattr(core, "VERSION", None) != CORE_VERSION:
        return
    _set_up(core)
    core.enable(True)
    # Kept when the core is put away again: an array it made may still hold its value alone.
    _track = core.track
    for name, method in list(vars(LazyArray).items()):
        road = getattr(method, "road", None)
        if road is not None:
            _stood_in_for[name] = method
            setattr(LazyArray, name, core.Road(road[0], method, *road[1:]))
    for owner in (namespace.LazyCallable, namespace.LazyNumPy):
        for name, method in list(vars(owner).items()):
            road = getattr(method, "road", None)
            if road is not None:
                _stood_in_for_elsewhere.append((owner, name, method))
                setattr(owner, name, core.Road(road[0], method, *road[1:]))
    # `x.T` is a property, which carries no attribute of its own: `np.transpose(x)`, deferred.
    _stood_in_for["T"] = transposed = vars(LazyArray)["T"]
    LazyArray.T = core.Road("attribute", transposed.fget, np.transpose, "transpose")
    # Last, since putting a method on a class has the interpreter write the class's slots again: the core's own
    # slots for the operators, which take the calls made at once without the way through the method.
    core.quick_slots()
    graph.deferred = core.deferred
    evaluate.compute_plain, evaluate.run_nodes, evaluate.needed = core.compute_plain, core.run_nodes, core.needed
    graph.release = core.release_pending
    core.small_reads(_compute_for_small)
    _core = core


def at_once_below(limit: int, fusible: frozenset | None = None, dtype: np.dtype | None = None) -> int:
    """Have small calls made as the program writes them, instead of written as nodes; or stop doing so.

    With a limit, a call that gives fewer elements than it is made at
    once when every array it reads has its values, and so is any view of
    an array that has them. A write is made at once, into the array's own
    memory, when nothing at all is pending. In each case that is what the
    eager program does at that line, and no later operation can tell: the
    result is a lazy array like any other, and a larger operation on it is
    deferred as before.

    The layer starts with no limit, where every operation it can defer is a
    node. A limit is for running a program quickly: NumPy makes a call on a
    small array in less time than the layer needs to write the node, plan
    the computation and come back to it. Only the C core does this; without
    it the limit is not kept.

    With `fusible` given, a call of any size is made at once as well when
    no kernel could make it: its callable is not one of `fusible`, or its
    result is not of `dtype`. A comparison, a selection by a mask, a matrix
    product, `np.exp`, anything complex or float32. Nothing written on top
    of such a call could be computed together with it, so its node would
    cost its writing, its planning and its turn in the computing loop, and
    buy nothing. A call a kernel can make still waits, whatever its size.

    Args:
        limit: A count of elements, or 0 for no call made at once.
        fusible: The callables a fused kernel can make, as a node has them; None for no such rule.
        dtype: The element type the kernels make them on.

    Returns:
        The limit this one replaces; 0 without the C core.
    """
    global _at_once_limit, _views_stand_alone
    if _core is None:
        return 0
    _at_once_limit = limit
    if limit:
        _views_stand_alone = True
    _core.fusible(fusible if limit else None, dtype if limit else None)
    return _core.at_once_below(limit)


def views_stand_alone(alive: bool) -> None:
    """Say whether a view taken at once, which the layer keeps no record of, may be alive.

    `at_once_below` says yes by itself, and nothing says no: such a view
    outlives the limit it was taken under. This is for a caller that knows
    every lazy array made so far is gone, as a test does between two
    programs, and wants writes to wait again wherever they can.

    Args:
        alive: False to have `_may_defer_write` ask nothing about views.
    """
    global _views_stand_alone
    _views_stand_alone = alive
    if _core is not None:
        _core.views_alone(alive)


def at_once_counts() -> tuple[int, int]:
    """Return how many calls and how many writes were made at once under `at_once_below`, since the core was loaded."""
    return _core.at_once_counts() if _core is not None else (0, 0)


def core_in_use() -> bool:
    """Say whether nodes are being written by the C core."""
    return _core is not None
