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
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from . import evaluate, explain, graph, record, rules
from .graph import Node, NotGathered, Trigger

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


# In-place ufunc calls, `np.clip` and the roundings take `out=`; these functions are deferred as writes when they do.
OUT_FUNCTIONS = frozenset({np.clip, np.round, np.around})


@dataclass(eq=False)
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
    doubtful: list[weakref.ref] = field(default_factory=list)
    # True for the cell of such an array.
    doubted: bool = False

    def shared_in_doubt(self) -> bool:
        """Say whether an array that may share this cell's memory is still alive, forgetting those that are not."""
        self.doubtful = [reference for reference in self.doubtful if reference() is not None]
        return bool(self.doubtful)


@dataclass(frozen=True)
class _Step:
    """One view operation on the way from a cell's contents to a lazy array that is a view of them."""

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

    def of(self, array: Any) -> Any:
        """Take the view of an array.

        Args:
            array: A computed array.

        Returns:
            NumPy's view of it.
        """
        return self.operation(array, *self.tail, **self.kwargs)


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
        NotGathered: For a strict array.
        Exception: What computing raised.
    """
    if strict:
        raise NotGathered(refusal)
    graph.fallback(graph.NEEDED, call)
    trigger = Trigger(graph.NEEDED, call)
    if everything:
        evaluate.compute_pending(trigger)
    evaluate.compute(nodes, trigger)


def _try_defer(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict,
               ruled_as: Callable[..., Any] | None = None) -> "LazyArray | None":
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

    Returns:
        A pending lazy array, or None when the call has to be made at once.

    Raises:
        Exception: NumPy's own, when the shapes or types are wrong.
    """
    unwrapped = _unwrapped(args)
    signature = rules.plan(ruled_as or operation, unwrapped, kwargs)
    if signature is None:
        return None
    if graph.pending_count() >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    node = graph.deferred(operation, name, unwrapped, kwargs, signature.shape, signature.dtype, signature.kind,
                          _strict(args, kwargs))
    base = args[0] if args and isinstance(args[0], LazyArray) else None
    if signature.kind != rules.VIEW or base is None or (operation is operator.getitem and signature.shape == ()):
        # A new array. An index that leaves no dimension gives a number, which is a copy, not a view.
        return LazyArray(node)
    if operation is operator.getitem or operation in rules.SURE_VIEWS:
        step = _Step(operation, name, unwrapped[1:], kwargs, signature.shape, signature.dtype, node.origin)
        return LazyArray(node, base._cell, base._steps + (step,))
    # `np.reshape`: a view or a copy, which is not known yet.
    made = LazyArray(node)
    made._cell.doubted = True
    base._cell.doubtful.append(weakref.ref(made))
    return made


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
    if graph.pending_count() >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    return tuple(LazyArray(graph.deferred(_part(operation, place), f"{name}[{place}]", unwrapped, {}, signature.shape,
                                          signature.dtype, signature.kind, _strict(args)))
                 for place in range(len(unwrapped[0].shape)))


def _defer_or_call(operation: Callable[..., Any], name: str, call: str, args: tuple) -> Any:
    """Defer a call if the rules allow it, and make it at once if they do not.

    Args:
        operation: The NumPy callable.
        name: What to call the node if the call is deferred.
        call: The call's name if it is made at once, for the counters and for messages.
        args: Positional arguments as the user wrote them.

    Returns:
        A pending lazy array, or the result made at once.
    """
    deferred = _try_defer(operation, name, args, {})
    # Not `deferred or ...`: a truth test of a lazy array would compute it.
    return deferred if deferred is not None else _at_once(operation, call, args, {})


def _at_once(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict) -> Any:
    """Make a call now, with NumPy, on gathered operands.

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


def _binary(ufunc: np.ufunc, reflected: bool = False) -> Callable[..., Any]:
    """Build the method for a binary operator.

    Args:
        ufunc: The ufunc the operator stands for.
        reflected: True for the method Python calls when the lazy array is on the right.

    Returns:
        A method that defers the ufunc, or makes the call at once when it cannot be deferred.
    """
    def apply(self, other):
        """Defer the operator's ufunc on the two operands, in the order written."""
        return _defer_or_call(ufunc, ufunc.__name__, f"np.{ufunc.__name__}", (other, self) if reflected else (self, other))
    return apply


def _unary(ufunc: np.ufunc) -> Callable[..., Any]:
    """Build the method for a unary operator.

    Args:
        ufunc: The ufunc the operator stands for.

    Returns:
        A method that defers the ufunc on the array.
    """
    def apply(self):
        """Defer the operator's ufunc on the array."""
        return _defer_or_call(ufunc, ufunc.__name__, f"np.{ufunc.__name__}", (self,))
    return apply


@dataclass(frozen=True)
class _Slot:
    """The place, in an index, of a lazy array whose values come when the write is computed."""

    # Which of the write's lazy index arrays goes here.
    place: int


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
    return (not node.exposed and node.failure is None and not cell.doubted and not cell.shared_in_doubt()
            and node.shape != () and 0 not in node.shape and node.dtype.kind in rules.DEFERRED_KINDS)


def _defer_write(target: "LazyArray", name: str, perform: Callable[..., Any], operands: tuple) -> None:
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
    """
    cell, steps = target._cell, target._steps

    def run(old: np.ndarray, *values: Any) -> np.ndarray:
        """Copy what the array held and make the write into the copy."""
        new = old.copy()
        view = new
        for step in steps:
            view = step.of(view)
        perform(view, *values)
        return new

    # Read before the cell changes: an operand that is a view of this array is a view of what it held.
    read = _unwrapped(tuple(operands))
    strict = cell.node.strict or _strict(operands)
    if graph.pending_count() >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    cell.node = graph.deferred(run, name, (cell.node,) + read, {}, cell.node.shape, cell.node.dtype, rules.WRITE, strict)


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


def _fits_elementwise(call: Callable[..., Any], target: "LazyArray", operands: tuple) -> bool:
    """Say whether an element-by-element write into an array can be deferred, raising if its types are wrong.

    Args:
        call: The write, taking the target then the operands.
        target: The array written into.
        operands: What the write reads.

    Returns:
        True when every operand has a shape the layer knows and they
        broadcast to the target's shape exactly. False sends the write the
        other way, where NumPy makes it at once and raises its own message
        if the shapes are wrong.

    Raises:
        Exception: NumPy's, for a result that cannot be cast into the target.
    """
    shapes = [_shape_of(operand) for operand in operands]
    if any(shape is None for shape in shapes):
        return False
    try:
        if np.broadcast_shapes(target._node.shape, *shapes) != target._node.shape:
            return False
    except ValueError:
        return False
    _types_fit(call, target, operands)
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


def _select(target: "LazyArray", parts: tuple) -> "LazyArray | None":
    """Defer `x[index]` for an index that holds lazy arrays still pending.

    Args:
        target: The array indexed.
        parts: What `_index_parts` returned for the index; at least one lazy array is pending.

    Returns:
        A pending lazy array whose node reads the array and the index's
        arrays, or None when the rules decline and the indexing has to be
        made at once. Selected by a pending mask, it has a length that is
        not known.

    Raises:
        IndexError: NumPy's, for a mask of the wrong shape or index arrays that do not broadcast.
    """
    fill, pending, _ = parts
    nodes = tuple(lazy._node for lazy in pending)
    index = fill(nodes)
    signature = rules.plan(operator.getitem, (target._node, index), {})
    if signature is None:
        return None

    def run(base: Any, *found: Any) -> Any:
        """Index the computed array with the computed index."""
        return base[fill(found)]

    if graph.pending_count() >= graph.PENDING_LIMIT:
        evaluate.compute_pending(Trigger(graph.LIMIT, "the limit of pending nodes"))
    node = graph.deferred(run, "getitem", (target._node,) + nodes, {}, signature.shape, signature.dtype, signature.kind,
                          _strict(target, pending))
    made = LazyArray(node)
    if signature.shape != () and rules.is_doubtful_view(index):
        # `x[np.argmax(w)]`: a view of a row if the index is a number when computed. Not known yet.
        made._cell.doubted = True
        target._cell.doubtful.append(weakref.ref(made))
    return made


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
        if self._node.shape == () and _is_number(self._node):
            return _defer_or_call(ufunc, ufunc.__name__, f"np.{ufunc.__name__}", (self, other))
        if _may_defer_write(self) and _fits_elementwise(apply, self, (other,)):
            _defer_write(self, symbol, perform, (other,))
            return self
        graph.fallback(graph.WRITE, symbol)
        trigger = Trigger(graph.WRITE, symbol)
        evaluate.compute_pending(trigger)
        evaluate.compute([self._node], trigger)
        value = self._node.value
        result = apply(value, _valued(other))
        return self if result is value else LazyArray(graph.leaf(result, symbol, graph.AT_ONCE, self._node.strict))
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
    def into(view: np.ndarray, *values: Any) -> None:
        """Make the call with its result written into the part of the copy the target is a view of."""
        function(*values, out=view)

    if not isinstance(target, LazyArray) or not _may_defer_write(target) or not _fits_elementwise(into, target, args):
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
        deferred = _try_defer(method, name, operands, kwargs, ruled_as=function)
        return deferred if deferred is not None else _at_once(method, f"ndarray.{name}", operands, kwargs)
    return reduce


class LazyArray:
    """An array whose values may not have been computed yet.

    Its shape and element type are known from the start, but for one length
    of what a pending mask selects. It is not an `ndarray`: an `ndarray` owns
    memory that anything may read at any time, and a pending array has none.
    """

    __slots__ = ("_cell", "_steps", "_seen", "__weakref__")
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
        self._cell = cell if cell is not None else _Cell(node)
        self._steps = steps
        # For a view: the cell's contents the node was made from, and the node.
        self._seen = (self._cell.node, node)

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
        return np.transpose(self)

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
            An iterator of `self[0]`, `self[1]`, ..., each deferred.

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
        call = f"np.{ufunc.__name__}" if method == "__call__" else f"np.{ufunc.__name__}.{method}"
        if out is not None:
            if method == "__call__" and len(out) == 1 and not kwargs and _write_out(ufunc, ufunc.__name__, inputs, out[0]):
                return out[0]
            _write(getattr(ufunc, method), call, inputs, {**kwargs, "out": out}, out)
            return out[0] if len(out) == 1 else out
        if method == WRITING_UFUNC_METHOD:
            return _write(getattr(ufunc, method), call, inputs, kwargs, inputs[:1])
        if method == "__call__":
            deferred = _try_defer(ufunc, ufunc.__name__, inputs, kwargs)
            if deferred is not None:
                return deferred
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
            mask is not known until then.
        """
        parts = _index_parts(index)
        if parts is None or not parts[1]:
            settled = index if parts is None else parts[0](())
            return _defer_or_call(operator.getitem, "getitem", "x[index]", (self, settled))
        selected = _select(self, parts)
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
            parts = _index_parts(index)
            if parts is not None and _assignment_fits(self, parts, value):
                fill = parts[0]

                def assign(view: np.ndarray, assigned: Any, *found: Any) -> None:
                    """Assign into the part of the copy the target is a view of, through the computed index."""
                    view[fill(found)] = assigned

                _defer_write(self, "setitem", assign, (value, *parts[1]))
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
        return np.reshape(self, shape[0] if len(shape) == 1 else shape)

    def transpose(self, *axes: Any) -> "LazyArray":
        """Permute the axes, deferred.

        Args:
            *axes: The new order, as one tuple or as separate integers; none to reverse.

        Returns:
            A pending lazy array.
        """
        if not axes:
            return np.transpose(self)
        return np.transpose(self, axes[0] if len(axes) == 1 and not isinstance(axes[0], int) else axes)

    def swapaxes(self, first: int, second: int) -> "LazyArray":
        """Exchange two axes, deferred.

        Args:
            first: One axis.
            second: The other.

        Returns:
            A pending lazy array.
        """
        return np.swapaxes(self, first, second)

    # ---- Operators.

    __add__, __radd__ = _binary(np.add), _binary(np.add, reflected=True)
    __sub__, __rsub__ = _binary(np.subtract), _binary(np.subtract, reflected=True)
    __mul__, __rmul__ = _binary(np.multiply), _binary(np.multiply, reflected=True)
    __truediv__, __rtruediv__ = _binary(np.true_divide), _binary(np.true_divide, reflected=True)
    __floordiv__, __rfloordiv__ = _binary(np.floor_divide), _binary(np.floor_divide, reflected=True)
    __mod__, __rmod__ = _binary(np.remainder), _binary(np.remainder, reflected=True)
    __pow__, __rpow__ = _binary(np.power), _binary(np.power, reflected=True)
    __matmul__, __rmatmul__ = _binary(np.matmul), _binary(np.matmul, reflected=True)
    __and__, __rand__ = _binary(np.bitwise_and), _binary(np.bitwise_and, reflected=True)
    __or__, __ror__ = _binary(np.bitwise_or), _binary(np.bitwise_or, reflected=True)
    __xor__, __rxor__ = _binary(np.bitwise_xor), _binary(np.bitwise_xor, reflected=True)
    __lt__, __le__ = _binary(np.less), _binary(np.less_equal)
    __gt__, __ge__ = _binary(np.greater), _binary(np.greater_equal)
    __eq__, __ne__ = _binary(np.equal), _binary(np.not_equal)
    __neg__, __pos__ = _unary(np.negative), _unary(np.positive)
    __abs__, __invert__ = _unary(np.absolute), _unary(np.invert)
    __iadd__, __isub__ = _in_place(operator.iadd, "x += y", np.add), _in_place(operator.isub, "x -= y", np.subtract)
    __imul__ = _in_place(operator.imul, "x *= y", np.multiply)
    __itruediv__ = _in_place(operator.itruediv, "x /= y", np.true_divide)
