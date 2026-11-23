"""Which NumPy calls are deferred, and the shape and element type of each without computing it.

`plan()` is the one entry point. It answers with a `Signature` when a call
can join the graph, raises NumPy's own exception when the call is wrong, and
answers None when the call is not one this layer defers: the caller then
makes it at once. When in doubt the answer is None; no shape or type is ever
guessed.

One length may be not known: how many elements a pending mask selects. The
shape of `x[mask]` has a stand-in there (`graph.unknown_length`), and so has
what is computed from it. Such an operand is deferred only where the result's
shape does not depend on the real length, and an error the real length would
cause is raised when the graph is computed, not on the line.

Three devices do the work:

- The shape comes from a rule per kind of operation.
- The element type comes from NumPy: the real call is made on stand-ins with
  one element per dimension (a probe), with Python numbers exactly as
  written, so promotion cannot disagree with NumPy's.
- An error comes from NumPy: the real call is made on stand-ins that have the
  operand's shape and no memory (a ghost), and NumPy raises before it reads
  anything.
"""
from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from . import graph
from .graph import Node

# Reductions that are deferred, as functions. `np.amax` and `np.amin` are
# separate objects from `np.max` and `np.min` in NumPy 2. `np.any` and `np.all`
# are reductions like the others: the same shape rule, and NumPy's answer on a
# probe for the element type, which is `bool`.
REDUCTIONS = frozenset({np.sum, np.mean, np.max, np.min, np.amax, np.amin, np.prod, np.var, np.std, np.any, np.all})
# Reductions that answer with a position. They take one axis or none, never several.
POSITIONS = frozenset({np.argmax, np.argmin})
# The only keywords a deferred reduction may carry.
REDUCTION_KEYWORDS = frozenset({"axis", "keepdims"})
# Functions NumPy answers with a view of their first argument.
VIEWS = frozenset({np.transpose, np.swapaxes, np.reshape, np.expand_dims, np.squeeze, np.ravel})
# The views that put elements in a new arrangement. Their shape is worked out
# from the number of elements, so they are not deferred on an array with a
# length that is not known.
REARRANGEMENTS = frozenset({np.reshape, np.ravel})
# Functions that work element by element and are not ufuncs.
SELECTIONS = frozenset({np.where})
# Functions that give their one argument again, as a new array.
COPIES = frozenset({np.copy})
# `np.dot` is deferred for operands of these numbers of dimensions, where it
# is the matrix product. With more, `np.dot` and `@` are different operations.
DOT_DIMENSIONS = (1, 2)
# An element type of no size. An array of it takes no memory whatever its
# shape, so indexing one by arrays gives NumPy's shape and NumPy's errors for
# nothing; see `_indexed_shape`.
NO_SIZE = np.dtype([])
# `dtype.kind` of the element types that are deferred: boolean, signed and
# unsigned integer, floating and complex. Dates, strings and objects are not.
DEFERRED_KINDS = frozenset("biufc")
# Python types that count as a number written in the program.
PYTHON_NUMBERS = (bool, int, float, complex)

# The kinds of deferred operation.
ELEMENTWISE, MATRIX_PRODUCT, REDUCTION, VIEW, INDEXING = "elementwise", "matmul", "reduction", "view", "indexing"
# The kind of a node that is a write into an array: a copy of what the array
# held, with the write made into it. Not planned here; `array.py` makes it.
WRITE = "write"
# The view functions whose result always shares its argument's memory.
# `np.reshape` and `np.ravel` are not among them: they copy when the memory's
# layout does not allow a view, which is not known before the argument is computed.
SURE_VIEWS = frozenset({np.transpose, np.swapaxes, np.expand_dims, np.squeeze})


@dataclass(frozen=True)
class Signature:
    """What is known about a deferred call's result before it is computed."""

    # Shape of the result.
    shape: tuple[int, ...]
    # Element type of the result.
    dtype: np.dtype
    # The kind of deferred operation: ELEMENTWISE, MATRIX_PRODUCT, REDUCTION, VIEW or INDEXING.
    kind: str


def plan(operation: Callable[..., Any], args: tuple, kwargs: dict) -> Signature | None:
    """Decide whether a call is deferred, and work out its result's shape and element type.

    Args:
        operation: The NumPy callable: a ufunc, a function, or `operator.getitem`.
        args: Its positional arguments, with a `Node` for each lazy operand.
        kwargs: Its keyword arguments.

    Returns:
        The signature of the result, or None when the call must be made at once.

    Raises:
        Exception: Whatever NumPy raises for these shapes and types.
    """
    kind = _kind(operation, args, kwargs)
    if kind is None or not all(_admissible(operand) for operand in _array_operands(kind, args)):
        return None
    unknown = not all(graph.is_known(_shape(operand)) for operand in _array_operands(kind, args))
    if unknown and not _holds_for_any_length(kind, operation, args, kwargs):
        return None
    try:
        shape = _SHAPE_RULES[kind](operation, args, kwargs)
    except Exception:
        if unknown:
            return None  # The real length may be one the call is right for: NumPy decides, at once.
        if kind == INDEXING:
            raise  # The rule asked NumPy itself; the exception is NumPy's.
        _let_numpy_raise(operation, args, kwargs)
        return None  # NumPy accepts what the rule refused: the rule is wrong, so do not defer.
    if shape is None or not _is_sound(shape):
        return None
    return Signature(shape, _element_type(kind, operation, args, kwargs), kind)


def is_array_index(index: Any) -> bool:
    """Say whether an index selects by arrays in a way this layer defers.

    Args:
        index: What was written between the brackets, with a `Node` for each
            lazy array that is still pending.

    Returns:
        True when every item is an integer, a slice, `None`, `...` or an
        array, and the arrays are either arrays of integers and masks that
        have their values (`ndarray`s, lists) and pending arrays of integers,
        or one pending mask alone. With a pending mask the result has a
        length that is not known; anything it would have to broadcast with
        is left to NumPy.
    """
    items = index if isinstance(index, tuple) else (index,)
    arrays = masks = 0
    for item in items:
        if isinstance(item, list):
            item = np.asarray(item)
        if isinstance(item, Node):
            if item.dtype.kind not in "biu" or not graph.is_known(item.shape) or 0 in item.shape:
                return False
            if item.dtype.kind == "b":
                if not item.shape:
                    return False  # A boolean without dimensions adds a dimension; NumPy's to handle.
                masks += 1
            else:
                arrays += 1
        elif isinstance(item, np.ndarray):
            if item.dtype.kind not in "biu":
                return False
            arrays += 1
        elif not is_basic_index(item):
            return False
    return (arrays > 0 and masks == 0) or (masks == 1 and arrays == 0)


def is_doubtful_view(index: Any) -> bool:
    """Say whether an index of arrays may give a view after all.

    Args:
        index: An index `is_array_index` accepts.

    Returns:
        True when every array in it is a pending array of integers without
        dimensions. If that is a number when computed, as the result of
        `np.argmax` is, NumPy gives a view of the row it names; if it is an
        array without dimensions, a copy. Which is not known yet.
    """
    items = index if isinstance(index, tuple) else (index,)
    arrays = [item for item in items if isinstance(item, (Node, np.ndarray, list))]
    return all(isinstance(item, Node) and item.shape == () for item in arrays)


def is_basic_index(index: Any) -> bool:
    """Say whether an index selects a view: integers, slices, `None` and `...` only.

    Args:
        index: What was written between the brackets.

    Returns:
        False for an index holding an array, a list or a mask, whose result
        NumPy copies and whose shape may depend on values.
    """
    items = index if isinstance(index, tuple) else (index,)
    return all(item is None or item is Ellipsis or isinstance(item, slice)
               or (isinstance(item, (int, np.integer)) and not isinstance(item, bool)) for item in items)


def _kind(operation: Callable[..., Any], args: tuple, kwargs: dict) -> str | None:
    """Classify a call.

    Args:
        operation: The NumPy callable.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Returns:
        One of the five kinds, or None for a call this layer does not defer.
    """
    operands = all(_is_operand(arg) for arg in args)
    if operation is np.dot:
        arrays = len(args) == 2 and not kwargs and all(isinstance(arg, (Node, np.ndarray)) for arg in args)
        return MATRIX_PRODUCT if arrays and all(len(arg.shape) in DOT_DIMENSIONS for arg in args) else None
    if operation in COPIES:
        return ELEMENTWISE if len(args) == 1 and not kwargs and isinstance(args[0], Node) else None
    if isinstance(operation, np.ufunc):
        if kwargs or not operands or len(args) != operation.nin:
            return None  # where=, dtype=, casting= and the like are NumPy's to handle now.
        if operation is np.matmul:
            return MATRIX_PRODUCT
        return ELEMENTWISE if operation.signature is None and operation.nout == 1 else None
    if operation in SELECTIONS:
        return ELEMENTWISE if operands and len(args) == 3 and not kwargs else None
    if operation in REDUCTIONS:
        return REDUCTION if _is_plain_reduction(args, kwargs) else None
    if operation in POSITIONS:
        return REDUCTION if _is_plain_reduction(args, kwargs) and _names_one_axis(args, kwargs) else None
    if operation is operator.getitem:
        if len(args) != 2 or not isinstance(args[0], Node):
            return None
        if is_basic_index(args[1]):
            return VIEW
        return INDEXING if is_array_index(args[1]) else None
    if operation in VIEWS:
        rest = args[1:] + tuple(kwargs.values())
        of_a_node = bool(args) and isinstance(args[0], Node) and not any(isinstance(item, Node) for item in rest)
        if operation is np.ravel and (len(args) != 1 or kwargs):
            return None  # `order=` chooses another arrangement; NumPy's to handle.
        return VIEW if of_a_node else None
    return None


def _names_one_axis(args: tuple, kwargs: dict) -> bool:
    """Say whether a reduction's axis is one axis or none.

    Args:
        args: The operand, and the axis when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        False for a tuple of axes or anything else: `np.argmax` takes none of
        those, and its error is NumPy's to raise.
    """
    axis = args[1] if len(args) > 1 else kwargs.get("axis")
    return axis is None or (isinstance(axis, (int, np.integer)) and not isinstance(axis, bool))


def _holds_for_any_length(kind: str, operation: Callable[..., Any], args: tuple, kwargs: dict) -> bool:
    """Say whether a call's shape rule is right whatever a length that is not known turns out to be.

    Args:
        kind: The call's kind.
        operation: The NumPy callable.
        args: Its positional arguments, one of them with such a length.
        kwargs: Its keyword arguments.

    Returns:
        True for an element-by-element call, a matrix product and a
        reduction: where their shape rule succeeds on the stand-in, it
        succeeds on any length. For views: True for the ones that move or add
        axes, and for an index that takes the whole of the axis in question
        with `:`. False for `np.reshape` and `np.ravel`, for `np.squeeze`
        without an axis (the length may be 1), for an index that takes part
        of that axis (`x[mask][:5]` has five elements or fewer) and for an
        index of arrays.
    """
    if kind in (ELEMENTWISE, MATRIX_PRODUCT, REDUCTION):
        return True
    if kind != VIEW or operation in REARRANGEMENTS:
        return False
    if operation is np.squeeze:
        return len(args) > 1 or "axis" in kwargs
    if operation is operator.getitem:
        return _takes_whole(args[0].shape, args[1])
    return True


def _takes_whole(shape: tuple[int, ...], index: Any) -> bool:
    """Say whether a basic index takes, with `:`, the whole of every axis whose length is not known.

    Args:
        shape: The shape of the array indexed.
        index: Integers, slices, `None` and `...`.

    Returns:
        True when each such axis meets `:` or is not named by the index at all.
    """
    items = index if isinstance(index, tuple) else (index,)
    named = sum(1 for item in items if item is not None and item is not Ellipsis)
    axis = 0
    for item in items:
        if item is None:
            continue
        if item is Ellipsis:
            axis += len(shape) - named
            continue
        if axis >= len(shape):
            return False  # More indices than axes: NumPy's error to raise.
        if not graph.is_known((shape[axis],)) and not (isinstance(item, slice) and item == slice(None)):
            return False
        axis += 1
    return True


def _is_sound(shape: tuple[int, ...]) -> bool:
    """Say whether a shape a rule gave can be relied on.

    Args:
        shape: The shape of a result.

    Returns:
        False when it has two lengths that are not known, whose stand-ins
        NumPy cannot multiply, or a length that came out of arithmetic on a
        stand-in and stands for nothing.
    """
    unknown = [length for length in shape if length >= graph.UNKNOWN_FLOOR]
    return len(unknown) <= 1 and all(graph.is_issued(length) for length in unknown)


def _is_plain_reduction(args: tuple, kwargs: dict) -> bool:
    """Say whether a reduction's arguments are ones this layer defers.

    Args:
        args: The operand, and the axis when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True for one node as the operand, an axis given at most once (by
        position or as `axis=`) and no keyword other than `axis` and
        `keepdims`. Anything else is NumPy's to handle at once, including the
        error for an axis given twice.
    """
    if not 1 <= len(args) <= 2 or not isinstance(args[0], Node):
        return False
    by_position = len(args) == 2
    if by_position and ("axis" in kwargs or isinstance(args[1], Node)):
        return False
    return set(kwargs) <= REDUCTION_KEYWORDS


def _is_operand(arg: Any) -> bool:
    """Say whether an argument is something an element-by-element call can take.

    Args:
        arg: One positional argument.

    Returns:
        True for a node, an exact `ndarray`, a NumPy scalar and a Python
        number. A list, a subclass of `ndarray` or any other array type is not:
        what NumPy does with those is left to NumPy.
    """
    return isinstance(arg, (Node, np.generic, *PYTHON_NUMBERS)) or type(arg) is np.ndarray


def _array_operands(kind: str, args: tuple) -> tuple:
    """Return the arguments of a call that are arrays or numbers, not axes or indices.

    Args:
        kind: The call's kind.
        args: Its positional arguments.

    Returns:
        Every argument for an element-by-element call or a matrix product;
        the first alone for a reduction or a view.
    """
    return args if kind in (ELEMENTWISE, MATRIX_PRODUCT) else args[:1]


def _admissible(operand: Any) -> bool:
    """Say whether an operand's element type and shape allow deferring.

    Args:
        operand: A node, an `ndarray`, a NumPy scalar or a Python number.

    Returns:
        False for an element type outside `DEFERRED_KINDS` and for an array
        with a dimension of length zero, where NumPy has rules of its own
        (`np.max` of nothing raises).
    """
    if isinstance(operand, PYTHON_NUMBERS):
        return True
    return operand.dtype.kind in DEFERRED_KINDS and 0 not in operand.shape


def _shape(operand: Any) -> tuple[int, ...]:
    """Return an operand's shape.

    Args:
        operand: A node, an `ndarray`, a NumPy scalar or a Python number.

    Returns:
        Its shape; the empty tuple for a number.
    """
    return () if isinstance(operand, PYTHON_NUMBERS) else operand.shape


def _elementwise_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of an element-by-element call: its operands broadcast together.

    Args:
        operation: The NumPy callable. Not used.
        args: The operands.
        kwargs: Not used.

    Returns:
        The broadcast shape.

    Raises:
        ValueError: When the shapes do not broadcast.
    """
    return np.broadcast_shapes(*(_shape(arg) for arg in args))


def _matmul_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of a matrix product, by NumPy's rule for `@`.

    A one-dimensional operand is a row on the left and a column on the right,
    and that dimension is dropped from the result. Leading dimensions broadcast.

    Args:
        operation: `np.matmul`. Not used.
        args: The two operands.
        kwargs: Not used.

    Returns:
        The shape of the product.

    Raises:
        ValueError: For an operand without dimensions, or inner sizes that differ.
    """
    left, right = _shape(args[0]), _shape(args[1])
    if not left or not right:
        raise ValueError("matmul takes no scalar")
    rows = left if len(left) > 1 else (1,) + left
    columns = right if len(right) > 1 else right + (1,)
    if rows[-1] != columns[-2]:
        raise ValueError("inner sizes differ")
    kept = (() if len(left) == 1 else (rows[-2],)) + (() if len(right) == 1 else (columns[-1],))
    return np.broadcast_shapes(rows[:-2], columns[:-2]) + kept


def _reduced_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of a reduction: the operand's, without the reduced axes or with them at length one.

    Args:
        operation: The reduction. Not used.
        args: The one operand, then the axis if it was given by position.
        kwargs: `axis` (None, an int or a tuple of ints) if it was given by keyword, and `keepdims`.

    Returns:
        The reduced shape.

    Raises:
        AxisError: For an axis the operand does not have.
    """
    shape, axis = _shape(args[0]), args[1] if len(args) > 1 else kwargs.get("axis")
    reduced = range(len(shape)) if axis is None else normalize_axis_tuple(axis, len(shape))
    if kwargs.get("keepdims", False):
        return tuple(1 if position in reduced else length for position, length in enumerate(shape))
    return tuple(length for position, length in enumerate(shape) if position not in reduced)


def _view_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of a view: NumPy's own answer, on a stand-in without memory.

    Args:
        operation: The view function, or `operator.getitem`.
        args: The node, then the axes, shape or index as written.
        kwargs: Keyword arguments as written.

    Returns:
        The shape NumPy gives. Making a view of a ghost reads no data.
        `np.ravel` is worked out here instead: it would copy the ghost.

    Raises:
        Exception: Whatever NumPy raises for a bad axis, shape or index.
    """
    if operation is np.ravel:
        return (int(np.prod(args[0].shape, dtype=np.intp)),)
    return operation(ghost(args[0]), *args[1:], **kwargs).shape


def _indexed_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...] | None:
    """Shape of an array indexed by arrays: NumPy's own answer, on an array of no size.

    Args:
        operation: `operator.getitem`. Not used.
        args: The node, then the index, with a `Node` for each pending array in it.
        kwargs: Not used.

    Returns:
        The shape NumPy gives. Unlike a view, the result is a new array, so a
        stand-in of the operand's element type would cost its size in memory
        to learn a shape. One of an element type of no size costs nothing.

        A pending array of integers is stood in for by zeros of its shape,
        which are in bounds whatever the array indexed; what it really holds
        is checked against the bounds when the graph is computed. A pending
        mask is stood in for by one that selects nothing, so NumPy checks
        its shape against the array's, and the one length of 0 in the answer
        is where the number selected goes: the mask's stand-in is put there.
        None when that place cannot be told apart, because a slice of the
        index selects nothing as well.

    Raises:
        IndexError: NumPy's, for an index out of bounds, index arrays that do
            not broadcast or a mask of the wrong shape.
    """
    index = args[1]
    items = index if isinstance(index, tuple) else (index,)
    stood_in = tuple(index_stand_in(item) for item in items)
    shape = np.broadcast_to(np.zeros((), NO_SIZE), args[0].shape)[stood_in if isinstance(index, tuple) else stood_in[0]].shape
    mask = next((item for item in items if isinstance(item, Node) and item.dtype.kind == "b"), None)
    if mask is None:
        return shape
    if shape.count(0) != 1:
        return None
    return tuple(graph.unknown_length(mask) if length == 0 else length for length in shape)


def index_stand_in(item: Any) -> Any:
    """Return what stands in for one item of an index while NumPy works out a shape.

    Args:
        item: An item of an index.

    Returns:
        For a pending array, one zero or one False broadcast to its shape,
        which takes no memory; anything else as it is.
    """
    if isinstance(item, Node):
        return np.broadcast_to(np.zeros((), np.bool_ if item.dtype.kind == "b" else np.intp), item.shape)
    return item


# The rule that gives the shape of each kind.
_SHAPE_RULES = {ELEMENTWISE: _elementwise_shape, MATRIX_PRODUCT: _matmul_shape,
                REDUCTION: _reduced_shape, VIEW: _view_shape, INDEXING: _indexed_shape}


def _element_type(kind: str, operation: Callable[..., Any], args: tuple, kwargs: dict) -> np.dtype:
    """Element type of a call's result, from NumPy itself.

    Args:
        kind: The call's kind.
        operation: The NumPy callable.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Returns:
        The operand's type for a view. Otherwise the type of the real call
        made on probes, so `float32 * 2.0` stays `float32` and the sum of
        `int8` is `int64`, because NumPy says so.

    Raises:
        Exception: Whatever NumPy raises for these types, such as a Python
            integer that does not fit the array's type.
    """
    if kind in (VIEW, INDEXING):
        return args[0].dtype
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # A probe holds zeros: 0 / 0 is not the user's doing.
        return np.result_type(operation(*(probe(arg) for arg in args), **kwargs))


def probe(operand: Any) -> Any:
    """Return a stand-in with an operand's element type and number of dimensions, and one element.

    Args:
        operand: A node, an `ndarray`, a NumPy scalar or a Python number.

    Returns:
        An array of zeros of shape (1, ..., 1) for an array; the operand
        itself for a number, so NumPy sees a Python number as one.
    """
    if isinstance(operand, (np.generic, *PYTHON_NUMBERS)):
        return operand
    return np.zeros((1,) * len(operand.shape), operand.dtype)


def ghost(operand: Any) -> Any:
    """Return a stand-in with an operand's shape and element type and no memory.

    Args:
        operand: A node or anything else.

    Returns:
        For a node, one zero broadcast to the node's shape: eight bytes
        whatever the shape. Anything else is returned as it is.
    """
    if isinstance(operand, Node):
        return np.broadcast_to(np.zeros((), operand.dtype), operand.shape)
    return operand


def _let_numpy_raise(operation: Callable[..., Any], args: tuple, kwargs: dict) -> None:
    """Make the real call on ghosts, so that a wrong call raises NumPy's own exception.

    Args:
        operation: The NumPy callable.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Raises:
        Exception: NumPy's, with NumPy's message. NumPy checks shapes before
            it reads an element, so nothing is computed on the way.
    """
    with np.errstate(all="ignore"):
        operation(*(ghost(arg) for arg in args), **kwargs)
