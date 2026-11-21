"""Which NumPy calls are deferred, and the shape and element type of each without computing it.

`plan()` is the one entry point. It answers with a `Signature` when a call
can join the graph, raises NumPy's own exception when the call is wrong, and
answers None when the call is not one this layer defers: the caller then
makes it at once. When in doubt the answer is None; no shape or type is ever
guessed.

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

from .graph import Node

# Reductions that are deferred, as functions. `np.amax` and `np.amin` are
# separate objects from `np.max` and `np.min` in NumPy 2.
REDUCTIONS = frozenset({np.sum, np.mean, np.max, np.min, np.amax, np.amin, np.prod, np.var, np.std})
# The only keywords a deferred reduction may carry.
REDUCTION_KEYWORDS = frozenset({"axis", "keepdims"})
# Functions NumPy answers with a view of their first argument.
VIEWS = frozenset({np.transpose, np.swapaxes, np.reshape, np.expand_dims, np.squeeze})
# Functions that work element by element and are not ufuncs.
SELECTIONS = frozenset({np.where})
# `dtype.kind` of the element types that are deferred: boolean, signed and
# unsigned integer, floating and complex. Dates, strings and objects are not.
DEFERRED_KINDS = frozenset("biufc")
# Python types that count as a number written in the program.
PYTHON_NUMBERS = (bool, int, float, complex)

# The kinds of deferred operation.
ELEMENTWISE, MATRIX_PRODUCT, REDUCTION, VIEW = "elementwise", "matmul", "reduction", "view"


@dataclass(frozen=True)
class Signature:
    """What is known about a deferred call's result before it is computed."""

    # Shape of the result.
    shape: tuple[int, ...]
    # Element type of the result.
    dtype: np.dtype


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
    try:
        shape = _SHAPE_RULES[kind](operation, args, kwargs)
    except Exception:
        _let_numpy_raise(operation, args, kwargs)
        return None  # NumPy accepts what the rule refused: the rule is wrong, so do not defer.
    return Signature(shape, _element_type(kind, operation, args, kwargs))


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
               or (isinstance(item, int) and not isinstance(item, bool)) for item in items)


def _kind(operation: Callable[..., Any], args: tuple, kwargs: dict) -> str | None:
    """Classify a call.

    Args:
        operation: The NumPy callable.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Returns:
        One of the four kinds, or None for a call this layer does not defer.
    """
    operands = all(_is_operand(arg) for arg in args)
    if isinstance(operation, np.ufunc):
        if kwargs or not operands or len(args) != operation.nin:
            return None  # where=, dtype=, casting= and the like are NumPy's to handle now.
        if operation is np.matmul:
            return MATRIX_PRODUCT
        return ELEMENTWISE if operation.signature is None and operation.nout == 1 else None
    if operation in SELECTIONS:
        return ELEMENTWISE if operands and len(args) == 3 and not kwargs else None
    if operation in REDUCTIONS:
        plain = len(args) == 1 and isinstance(args[0], Node) and set(kwargs) <= REDUCTION_KEYWORDS
        return REDUCTION if plain else None
    if operation in VIEWS or operation is operator.getitem:
        rest = args[1:] + tuple(kwargs.values())
        of_a_node = bool(args) and isinstance(args[0], Node) and not any(isinstance(item, Node) for item in rest)
        if operation is operator.getitem:
            return VIEW if of_a_node and len(args) == 2 and is_basic_index(args[1]) else None
        return VIEW if of_a_node else None
    return None


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
        args: The one operand.
        kwargs: `axis` (None, an int or a tuple of ints) and `keepdims`.

    Returns:
        The reduced shape.

    Raises:
        AxisError: For an axis the operand does not have.
    """
    shape, axis = _shape(args[0]), kwargs.get("axis")
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

    Raises:
        Exception: Whatever NumPy raises for a bad axis, shape or index.
    """
    return operation(_ghost(args[0]), *args[1:], **kwargs).shape


# The rule that gives the shape of each kind.
_SHAPE_RULES = {ELEMENTWISE: _elementwise_shape, MATRIX_PRODUCT: _matmul_shape,
                REDUCTION: _reduced_shape, VIEW: _view_shape}


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
    if kind == VIEW:
        return args[0].dtype
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # A probe holds zeros: 0 / 0 is not the user's doing.
        return np.result_type(operation(*(_probe(arg) for arg in args), **kwargs))


def _probe(operand: Any) -> Any:
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


def _ghost(operand: Any) -> Any:
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
        operation(*(_ghost(arg) for arg in args), **kwargs)
