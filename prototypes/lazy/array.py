"""`LazyArray`: what the user holds. A handle on one node of the graph.

Every NumPy call that meets a lazy array arrives in this file, by one of
three roads: an operator or method of the class, `__array_ufunc__` (NumPy's
ufuncs, and an `ndarray`'s own operators), or `__array_function__` (NumPy's
functions). Each call then takes one of three paths:

- deferred: `rules.plan` accepts it, and a node joins the graph;
- made at once: NumPy makes the call on gathered operands, and the result is
  a lazy array that already has its values;
- a write: everything pending is computed first, then NumPy writes.

`gather()` is the way to ask for values. Anything else that needs them
(`float(x)`, `if x > 0:`, `np.asarray(x)`, a library) gets them too: they are
computed on the spot and the log says so, once per line. An array made by a
strict stand-in raises `NotGathered` there instead.
"""
from __future__ import annotations

import operator
from typing import Any, Callable, Iterator

import numpy as np

from . import evaluate, explain, graph, record, rules
from .graph import Node, NotGathered, Trigger

# NumPy functions that answer from the shape alone, and the attribute that answers each.
SHAPE_QUESTIONS = {np.shape: "shape", np.ndim: "ndim", np.size: "size"}
# NumPy functions that write into their first argument.
WRITING_FUNCTIONS = frozenset({np.copyto, np.putmask, np.place, np.put, np.fill_diagonal})
# Methods of `ndarray` that write into the array they are called on.
WRITING_METHODS = frozenset({"sort", "fill", "put", "partition", "resize", "itemset", "setfield", "byteswap"})
# The one ufunc method that writes into its first argument.
WRITING_UFUNC_METHOD = "at"


def wrap(value: Any, source: str = "array", strict: bool = False) -> "LazyArray":
    """Make a lazy array of something that already has values.

    Args:
        value: A lazy array, which is returned as it is; or an `ndarray` or
            NumPy scalar, which is held by reference; or anything
            `np.asarray` accepts.
        source: What made the array, for a recording: "array", or the name
            of the stand-in's function, such as "np.zeros".
        strict: Whether the new array refuses to give its values without
            `gather()`. A lazy array that is passed in keeps what it has.

    Returns:
        A lazy array on a computed node.
    """
    if isinstance(value, LazyArray):
        return value
    if not isinstance(value, (np.ndarray, np.generic)):
        value = np.asarray(value)
    return LazyArray(graph.leaf(value, source, strict=strict))


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
    return LazyArray(graph.deferred(operation, name, unwrapped, kwargs, signature.shape, signature.dtype, signature.kind,
                                    _strict(args, kwargs)))


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
    evaluate.compute([lazy._node for lazy in _lazies_in((args, kwargs))], Trigger(graph.AT_ONCE, name))
    return _made_lazy(operation(*_valued(args), **_valued(kwargs)), name, strict)


def _made_lazy(result: Any, name: str, strict: bool) -> Any:
    """Turn the result of a call made at once into lazy arrays.

    Args:
        result: What NumPy returned.
        name: The call's name, for messages and the log.
        strict: Whether a strict array was among the call's arguments.

    Returns:
        A lazy array for an array or a NumPy scalar; the same kind of tuple
        or list for a tuple or list of those; None for None. Anything else
        (a Python bool, float, list of numbers) is values leaving without
        `gather()`: it is returned as it is, and the log says so.

    Raises:
        NotGathered: For such a value when a strict array was involved.
    """
    if isinstance(result, (np.ndarray, np.generic)):
        return LazyArray(graph.leaf(result, name, graph.AT_ONCE, strict))
    if result is None:
        return None
    if isinstance(result, (list, tuple)) and all(isinstance(item, (np.ndarray, np.generic)) for item in result):
        items = [LazyArray(graph.leaf(item, name, graph.AT_ONCE, strict)) for item in result]
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


def _in_place(apply: Callable[[Any, Any], Any], symbol: str) -> Callable[..., Any]:
    """Build the method for an in-place operator, which is a write.

    Args:
        apply: The function of the `operator` module, such as `operator.iadd`.
        symbol: The operator as written, for the counters.

    Returns:
        A method that computes everything pending and hands the operator to
        NumPy on the computed value. NumPy returns the same array when it
        wrote in place, and then the lazy array is returned unchanged, so
        every name for it sees the write; for a NumPy scalar, which cannot be
        written, NumPy returns a new one and so does the method.
    """
    def write(self, other):
        """Apply the in-place operator with NumPy's meaning."""
        graph.fallback(graph.WRITE, symbol)
        trigger = Trigger(graph.WRITE, symbol)
        evaluate.compute_pending(trigger)
        evaluate.compute([self._node], trigger)
        value = self._node.value
        result = apply(value, _valued(other))
        return self if result is value else LazyArray(graph.leaf(result, symbol, graph.AT_ONCE, self._node.strict))
    return write


def _reduction(function: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Build the method form of a reduction.

    Args:
        function: The NumPy function whose rule gives the shape and element type, such as `np.sum`.
        name: The method's name, such as "sum".

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

    Its shape and element type are known from the start. It is not an
    `ndarray`: an `ndarray` owns memory that anything may read at any time,
    and a pending array has none.
    """

    __slots__ = ("_node", "__weakref__")
    # So that type(x) prints the name a user imports, not the file the class is written in.
    __module__ = "prototypes.lazy"
    # Arrays compare element by element, so they cannot be hashed. As for `ndarray`.
    __hash__ = None

    def __init__(self, node: Node) -> None:
        """Make a handle on a node.

        Args:
            node: The node of the graph this array stands for.
        """
        self._node = node

    # ---- Known without computing.

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape NumPy would give the array."""
        return self._node.shape

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
        return int(np.prod(self._node.shape, dtype=np.intp))

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
        return self._node.shape[0]

    def __repr__(self) -> str:
        """Describe the array without computing it and without showing values."""
        return f"LazyArray({self.dtype}, shape={self.shape}, {self._node.state})"

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
        return (self[position] for position in range(self._node.shape[0]))

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
            _write(func, call, args, kwargs, target if isinstance(target, tuple) else (target,))
            return target
        if func in WRITING_FUNCTIONS:
            return _write(func, call, args, kwargs, args[:1])
        deferred = _try_defer(func, func.__name__, args, kwargs)
        return deferred if deferred is not None else _at_once(func, call, args, kwargs)

    # ---- Indexing.

    def __getitem__(self, index: Any) -> "LazyArray":
        """Index the array.

        Args:
            index: What was written between the brackets.

        Returns:
            A deferred view for integers, slices, `None` and `...`. An index
            holding an array or a mask is made at once: the shape of `x[mask]`
            depends on values, and a lazy array's shape is always known.
        """
        return _defer_or_call(operator.getitem, "getitem", "x[index]", (self, index))

    def __setitem__(self, index: Any, value: Any) -> None:
        """Assign into the array: a write, with NumPy's meaning.

        Args:
            index: What was written between the brackets.
            value: What to assign. A lazy array's values are used.
        """
        _write(operator.setitem, "x[index] = v", (self, index, value), {}, (self,))

    # ---- Methods the graph knows. Any other method is `ndarray`'s, through `__getattr__`.

    sum = _reduction(np.sum, "sum")
    mean = _reduction(np.mean, "mean")
    max = _reduction(np.max, "max")
    min = _reduction(np.min, "min")
    prod = _reduction(np.prod, "prod")
    var = _reduction(np.var, "var")
    std = _reduction(np.std, "std")

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
    __iadd__, __isub__ = _in_place(operator.iadd, "x += y"), _in_place(operator.isub, "x -= y")
    __imul__, __itruediv__ = _in_place(operator.imul, "x *= y"), _in_place(operator.itruediv, "x /= y")
