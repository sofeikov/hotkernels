"""Which NumPy calls are deferred, and the shape and element type of each without computing it.

`plan()` is the one entry point. It answers with a `Signature` when a call
can join the graph, raises NumPy's own exception when the call is wrong, and
answers None when the call is not one this layer defers: the caller then
makes it at once. When in doubt the answer is None; no shape or type is ever
guessed.

One length may be not known: how many elements a pending mask selects, or how
many elements of an array are not zero. The shape of `x[mask]` and of
`np.flatnonzero(a)` has a stand-in there (`graph.unknown_length`), and so has
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

Those calls cost several times what writing the node does, and a program
asks the same question on every turn of a loop. So `plan()` remembers its
answers. The answer to a call depends on the operation, on the shape and
element type of each array among the arguments and on the other arguments as
written, and on nothing else: not on any array's values, which a stand-in
never has. `_key` writes exactly those down, and a call whose key was seen
before gets the answer that was worked out then. Whatever `_key` cannot write
down, it does not: the answer is then worked out as if nothing were
remembered. The shape of a plain index (`x[i]`, `x[2:6]`, `x[i, 1:]`) is the
one answer worked out here instead of asked of NumPy, since the index differs
on every turn; `basic_shape` gives it only where it is sure.

A length that is not known is a different integer for every mask, so a call
on what a mask selected would never be seen twice. For those the key is
written with the stand-ins replaced by placeholders, numbered in the order
they are met, and the answer is kept with the same placeholders in its
shape: the rules treat a stand-in as a length equal to itself and to nothing
else, so the answer for one mask is the answer for any other with the
stand-ins exchanged. An index that holds pending arrays is remembered the
same way.
"""
from __future__ import annotations

import math
import operator
import warnings
from typing import Any, Callable, NamedTuple

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from . import graph
from .graph import Node

# Reductions that are deferred, as functions. `np.amax` and `np.amin` are
# separate objects from `np.max` and `np.min` in NumPy 2. `np.any` and `np.all`
# are reductions like the others: the same shape rule, and NumPy's answer on a
# probe for the element type, which is `bool`.
REDUCTIONS = frozenset({np.sum, np.mean, np.max, np.min, np.amax, np.amin, np.prod, np.var, np.std, np.any, np.all})
# `np.count_nonzero` is a reduction with the shape rule of the others, and one condition of its own.
# Without an axis NumPy 2.3 and later answer with a NumPy scalar, as `np.sum` does. Earlier versions
# answer with a Python `int`, which the rest of an expression promotes by other rules than it does an
# `intp`, so the element type worked out for what follows could be wrong: there that call is made at once.
COUNT_IS_A_NUMPY_SCALAR = isinstance(np.count_nonzero(np.zeros(1)), np.generic)
# Reductions that answer with a position. They take one axis or none, never several.
# `np.nanargmax` and `np.nanargmin` raise for a slice that is all NaN, which depends on
# the values: that error comes when the graph is computed, with a note naming the line.
POSITIONS = frozenset({np.argmax, np.argmin, np.nanargmax, np.nanargmin})
# The only keywords a deferred reduction may carry.
REDUCTION_KEYWORDS = frozenset({"axis", "keepdims"})
# Running totals and running products: one result per element, each depending on those before it.
SCANS = frozenset({np.cumsum, np.cumprod})
# The only keyword a deferred running total may carry.
SCAN_KEYWORDS = frozenset({"axis"})
# Searches of a sorted array: one position per element looked for.
SEARCHES = frozenset({np.searchsorted})
# The values `side` may have in a deferred search.
SEARCH_SIDES = ("left", "right")
# Functions NumPy answers with a view of their first argument.
VIEWS = frozenset({np.transpose, np.swapaxes, np.reshape, np.expand_dims, np.squeeze, np.ravel})
# The views that put elements in a new arrangement. Their shape is worked out
# from the number of elements, so they are not deferred on an array with a
# length that is not known.
REARRANGEMENTS = frozenset({np.reshape, np.ravel})
# Functions that work element by element and are not ufuncs.
SELECTIONS = frozenset({np.where})
# Functions that round each element of their first argument. Not ufuncs: they take
# the number of decimals as a second argument, which is a setting and not an array.
ROUNDINGS = frozenset({np.round, np.around})
# Casts. `x.astype(t)` is ruled as `ndarray`'s own method, called as a function of the
# array; `np.astype(x, t)` is the same call, where this NumPy has it.
CASTS = frozenset({np.ndarray.astype, getattr(np, "astype", np.ndarray.astype)})
# Keywords of a cast that change nothing when they have these values, which are NumPy's
# defaults. With any other value the cast is made at once.
CAST_DEFAULTS = {"order": "K", "casting": "unsafe", "subok": True, "device": None}
# The element-by-element calls with one operand: what follows it is a setting (the number
# of decimals, the element type), passed to NumPy as written and never stood in for.
ONE_OPERAND = ROUNDINGS | CASTS
# Functions that give the elements of their first argument again in another order along
# an axis, as a new array of the same shape and element type.
ORDERINGS = frozenset({np.partition})
# Keywords of such a call that change nothing when they have these values, which are
# NumPy's defaults. With any other value the call is made at once.
ORDERING_DEFAULTS = {"kind": "introselect", "order": None}
# Functions that answer with the positions of the elements that are not zero. How many
# there are depends on the values, so the result has a length that is not known.
NONZEROS = frozenset({np.nonzero, np.flatnonzero, np.argwhere})
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
# The kind of a call in ORDERINGS.
ORDERING = "ordering"
# The kind of a call in NONZEROS.
NONZERO = "nonzero"
# The kinds of the calls in SCANS and in SEARCHES.
SCAN, SEARCH = "scan", "search"
# The kind of a node that is a write into an array: a copy of what the array
# held, with the write made into it. Not planned here; `array.py` makes it.
WRITE = "write"
# The view functions whose result always shares its argument's memory.
# `np.reshape` and `np.ravel` are not among them: they copy when the memory's
# layout does not allow a view, which is not known before the argument is computed.
SURE_VIEWS = frozenset({np.transpose, np.swapaxes, np.expand_dims, np.squeeze})

# Answers of `plan` remembered at most, before the memory is emptied and starts again.
PLAN_MEMORY_LIMIT = 8192
# The answer to a call with a length that is not known is remembered only if
# its arrays have, between them and with that length left out, no more
# elements than this. A stand-in is below 2**41 and an element takes at most
# 32 bytes, so no size NumPy works out on a stand-in of such a call can pass
# 2**62, whichever mask the stand-in is of: NumPy raises for a size that
# does not fit 63 bits, and it must raise for every mask or for none.
UNKNOWN_MEMORY_ELEMENTS = 2 ** 16
# Types of an argument that is written into a key as its type and its value:
# two of them that compare equal and are of one type are the same argument to NumPy.
_PLAIN_SETTINGS = frozenset({int, bool, float, complex, str, type(None), type(Ellipsis)})
# Python numbers that are whole: as an operand of a ufunc, the value decides
# whether NumPy raises (`uint8 + 300`, an integer to the power -1).
_WHOLE_NUMBERS = frozenset({int, bool})
# Python numbers with a fraction: as an operand of a ufunc only the type
# counts. NumPy 2 takes the element type of the result from the types alone,
# and under the settings a probe runs with, no value of one makes a ufunc raise.
_FRACTIONAL_NUMBERS = frozenset({float, complex})
# NumPy's scalar types for which the same holds, and those that are whole.
_FRACTIONAL_SCALARS = frozenset(kind for kind in np.sctypeDict.values() if issubclass(kind, np.inexact))
_WHOLE_SCALARS = frozenset(kind for kind in np.sctypeDict.values() if issubclass(kind, (np.integer, np.bool_)))


class Signature(NamedTuple):
    """What is known about a deferred call's result before it is computed.

    A tuple with names: it cannot be changed, so one that is remembered can
    be handed to every call it answers, and making one costs little, which
    counts for the calls that get theirs made anew (`_plan_marked`).
    """

    # Shape of the result.
    shape: tuple[int, ...]
    # Element type of the result.
    dtype: np.dtype
    # The kind of deferred operation: ELEMENTWISE, MATRIX_PRODUCT, REDUCTION, VIEW, INDEXING, ORDERING, NONZERO, SCAN or SEARCH.
    kind: str
    # The number `graph.spec_of` gives the shape and element type, for the
    # node that is made; 0 when it was not worked out.
    spec: int = 0


# The answers of `plan` so far, by what `_key` writes down of a call.
_plans: dict[tuple, Signature] = {}
# The element type NumPy's resolver gave for a ufunc and the types of its operands, with the call's kind, by
# the ufunc and one item per operand: the identity of an array's or a NumPy scalar's `dtype` object, which is
# NumPy's own and lives as long as NumPy does, or the type of a Python number. The answer does not depend on
# any shape, so a call that differs from an earlier one in its shapes alone needs only the shapes broadcast,
# which the C core does itself (`first_ufunc` in `_core.c`).
_types: dict[tuple, tuple[np.dtype, str]] = {}
# A Python `int` among a ufunc's operands can make NumPy raise by its value: one that does not fit the array's
# integer type, a negative power of an integer. Within these bounds it cannot, and NumPy's resolver answers for it
# by its type as for a float: from 0 to SMALL_WHOLE beside arrays of any type, which every integer type holds; and
# up to LARGE_WHOLE either side of zero beside arrays with a fraction only, which take it as a float, or, where
# the ufunc wants a whole number there (`np.ldexp`), as a 32-bit one, which holds it. A test holds both to the
# call for every ufunc.
SMALL_WHOLE, LARGE_WHOLE = 127, 2 ** 31 - 1
# Entries of `_types` kept at most.
TYPE_MEMORY_LIMIT = 4096
# The shapes and element types that allow deferring, by their number: what
# `_admissible` says of an operand, remembered.
_admissible_specs: dict[int, bool] = {}


def _forget_plans() -> None:
    """Empty what `plan` remembers: the numbers its keys were written with are being forgotten."""
    _plans.clear()
    _admissible_specs.clear()


graph.spec_memory_emptied.append(_forget_plans)


def plan(operation: Callable[..., Any], args: tuple, kwargs: dict) -> Signature | None:
    """Decide whether a call is deferred, and work out its result's shape and element type.

    Args:
        operation: The NumPy callable: a ufunc, a function, or `operator.getitem`.
        args: Its positional arguments, with a `Node` for each lazy operand.
        kwargs: Its keyword arguments.

    Returns:
        The signature of the result, or None when the call must be made at
        once. For `np.nonzero`, whose result is a tuple with one array per
        dimension of its operand, the signature of each of those arrays.

    Raises:
        Exception: Whatever NumPy raises for these shapes and types.
    """
    key = _key(operation, args, kwargs, None)
    if key is None:
        return _plan_marked(operation, args, kwargs)
    signature = _plans.get(key)
    if signature is None:
        signature = _plan(operation, args, kwargs)
        if signature is not None:
            signature = _remember(key, signature)
    return signature


def _remember(key: tuple, signature: Signature) -> Signature:
    """Keep the answer to a call.

    Args:
        key: What `_key` wrote down of the call.
        signature: The answer `_plan` gave: not None, since a call that is
            made at once is looked at again every time.

    Returns:
        The signature as it is kept, with the number of its shape and
        element type where the shape has no placeholder and no stand-in.
    """
    if len(_plans) >= PLAN_MEMORY_LIMIT:
        _plans.clear()
    numbered = all(0 <= length < graph.UNKNOWN_FLOOR for length in signature.shape)
    kept = _plans[key] = Signature(signature.shape, signature.dtype, signature.kind,
                                   graph.spec_of(signature.shape, signature.dtype) if numbered else 0)
    return kept


def _plan_marked(operation: Callable[..., Any], args: tuple, kwargs: dict) -> Signature | None:
    """Answer a call that `_key` could not write down plainly: one with a length that is not known, or an index.

    Args:
        operation: The NumPy callable.
        args: Its positional arguments, with a `Node` for each lazy operand.
        kwargs: Its keyword arguments.

    Returns:
        What `_plan` answers. The key is written with a placeholder for each
        stand-in, and the answer kept under it has the same placeholders in
        its shape; a call that finds it gets it with its own stand-ins put
        back. A call that still cannot be written down, or whose arrays are
        too large for the answer to be sure to hold for every stand-in, is
        worked out as if nothing were remembered.
    """
    marks: dict[int, int] = {}
    key = _key(operation, args, kwargs, marks) if _is_moderate(args[:1] if operation is operator.getitem else args) else None
    if key is None:
        return _plan(operation, args, kwargs)
    kept = _plans.get(key)
    if kept is None:
        signature = _plan(operation, args, kwargs)
        if signature is not None:
            # With the placeholders of this call's stand-ins. A stand-in the call did not bring is not expected; the answer is then not kept.
            shape = tuple(marks.get(length, length) for length in signature.shape)
            if graph.is_known(shape):
                _remember(key, Signature(shape, signature.dtype, signature.kind))
        return signature
    if not marks:
        return kept
    stood_for = {placeholder: length for length, placeholder in marks.items()}
    shape = [stood_for[length] if length < 0 else length for length in kept.shape]
    return Signature(tuple(shape), kept.dtype, kept.kind)


def _is_moderate(operands: tuple) -> bool:
    """Say whether a call's arrays are small enough for its answer to be remembered under placeholders.

    Args:
        operands: The call's positional arguments; for an index, the array
            indexed alone, since a pending array in an index is stood in for
            by its own shape and never by a length that is not known.

    Returns:
        True when no node among them has a length that is not known: NumPy
        is then shown no stand-in at all. Otherwise, True when the nodes and
        `ndarray`s have, between them and with that length left out, at most
        UNKNOWN_MEMORY_ELEMENTS elements.
    """
    elements = 1
    unknown = False
    for operand in operands:
        if type(operand) is Node or type(operand) is np.ndarray:
            for length in operand.shape:
                if length < graph.UNKNOWN_FLOOR:
                    elements *= length
                else:
                    unknown = True
    return not unknown or elements <= UNKNOWN_MEMORY_ELEMENTS


def _key(operation: Callable[..., Any], args: tuple, kwargs: dict, marks: dict[int, int] | None) -> tuple | None:
    """Write down everything the answer to a call depends on, or decline.

    Args:
        operation: The NumPy callable.
        args: Its positional arguments, with a `Node` for each lazy operand.
        kwargs: Its keyword arguments.
        marks: None to decline a call with a length that is not known, and
            an index. Otherwise a dictionary, empty at first, that gains a
            placeholder for each stand-in the call brings: a negative
            integer, by the stand-in. Such a call is then written down with
            the placeholders, as `_marked_part` says.

    Returns:
        A tuple that two calls share only if `_plan` answers them alike, up
        to the stand-ins that `marks` names: the operation, then for each
        argument what `_plan` reads of it. None for a call whose answer is
        not remembered: one with an argument this function does not know how
        to write down, and the positions of what is not zero, whose mark is
        a new one.
    """
    if type(operation) is np.ufunc:
        if kwargs:
            return None  # Not deferred; see `_kind`.
        parts = [operation]
        for arg in args:
            # A node is the commonest operand: its number is read here, without the call.
            part = (arg.spec or graph.spec(arg) or _marked_part(arg, marks)) if type(arg) is Node else operand_part(arg)
            if not part:
                return None
            parts.append(part)
        return tuple(parts)
    if operation is operator.getitem:
        return None if marks is None else _index_key(args, marks)
    if operation in NONZEROS:
        return None
    # `np.count_nonzero` is deferred or not by a fact about this NumPy, which a test changes.
    parts = [operation, COUNT_IS_A_NUMPY_SCALAR] if operation is np.count_nonzero else [operation]
    for arg in args:
        kind = type(arg)
        if kind is Node:
            part = arg.spec or graph.spec(arg) or _marked_part(arg, marks)
        elif kind is int:
            part = (int, arg)  # An axis: what `_setting_part` gives, without the call.
        else:
            part = _array_part(arg) or _setting_part(arg)
        if not part:
            return None
        parts.append(part)
    for name, value in kwargs.items():
        part = _setting_part(value)
        if not part:
            return None
        parts.append(name)
        parts.append(part)
    return tuple(parts)


def _marked_part(node: Node, marks: dict[int, int] | None) -> int:
    """Write down a node with a length that is not known, with a placeholder for the stand-in.

    Args:
        node: A node whose shape has a stand-in.
        marks: The placeholders given so far in this call, by stand-in; it
            gains one for a stand-in met for the first time. None when the
            call is being written down without placeholders.

    Returns:
        The number `graph.spec_of` gives the node's element type with its
        shape, the stand-in replaced by its placeholder: -1 for the first
        stand-in of the call, -2 for the second. Two operands selected by
        one mask get one placeholder, so the key says they have one length.
        0 when `marks` is None.
    """
    if marks is None:
        return 0
    shape = [length if length < graph.UNKNOWN_FLOOR else marks.setdefault(length, -len(marks) - 1) for length in node.shape]
    return graph.spec_of(tuple(shape), node.dtype)


def _index_key(args: tuple, marks: dict[int, int]) -> tuple | None:
    """Write down `x[index]` for a key.

    Args:
        args: The node indexed, then the index, with a `Node` for each lazy array in it that is still pending.
        marks: The placeholders of the call's stand-ins; see `_key`.

    Returns:
        The node, then the index item by item: a pending array by its shape
        and element type, a whole number, `None` and `...` as written, a
        slice by its three parts. The length a pending mask selects is the
        mask's own stand-in, which `marks` gains, so the answer is kept with
        a placeholder where that length goes. None for an index that holds
        an `ndarray` or a list, whose values NumPy checks against the bounds.
    """
    if len(args) != 2 or type(args[0]) is not Node:
        return None
    target, index = args
    plain = type(index) is not tuple
    if plain and isinstance(index, tuple):
        return None
    parts = [operator.getitem, target.spec or graph.spec(target) or _marked_part(target, marks), plain]
    for item in ((index,) if plain else index):
        kind = type(item)
        if kind is Node:
            part = item.spec or graph.spec(item) or _marked_part(item, marks)
            if item.dtype.kind == "b":
                marks.setdefault(graph.unknown_length(item), -len(marks) - 1)
        elif kind is slice:
            part = (slice, _setting_part(item.start), _setting_part(item.stop), _setting_part(item.step))
            if None in part:
                return None
        else:
            part = _setting_part(item)
        if not part:
            return None
        parts.append(part)
    return tuple(parts)


def _array_part(arg: Any) -> int:
    """Write down a node or an `ndarray` for a key.

    Args:
        arg: One argument of a call.

    Returns:
        For a node, the number of its shape and element type; for an exact
        `ndarray` the same number negated, since some rules take a node
        where they refuse an `ndarray`. 0 for anything else, and for a shape
        with a length that is not known. `_plan` reads nothing else of
        either: a probe and a ghost have the shape and the type and no values.
    """
    kind = type(arg)
    if kind is Node:
        return arg.spec or graph.spec(arg)
    if kind is np.ndarray:
        return -graph.spec_of(arg.shape, arg.dtype)
    return 0


def operand_part(arg: Any) -> Any:
    """Write down one operand of a ufunc for a key.

    Args:
        arg: A node, an `ndarray`, a number or anything else.

    Returns:
        What `_array_part` gives for a node or an `ndarray`. For a number
        with a fraction, Python's or NumPy's, its type: see
        `_FRACTIONAL_NUMBERS`. For a whole number its type and its value.
        A false value for anything else, a subclass of these included.
    """
    kind = type(arg)
    if kind is Node:
        return arg.spec or graph.spec(arg)
    if kind in _FRACTIONAL_NUMBERS or kind in _FRACTIONAL_SCALARS:
        return kind
    if kind in _WHOLE_NUMBERS:
        return (kind, arg)
    if kind is np.ndarray:
        return -graph.spec_of(arg.shape, arg.dtype)
    if kind in _WHOLE_SCALARS:
        return (kind, arg.item())
    return None


def _setting_part(arg: Any) -> tuple | None:
    """Write down an argument that is not an array, for a key: an axis, a shape, an element type, a number.

    Args:
        arg: One argument of a call, positional or by keyword.

    Returns:
        The type with the value for a Python number, a string, `None` and
        `...`; the same for a tuple of those, item by item, so that `(0, 1)`
        and `(0.0, True)`, which compare equal, are told apart; the type
        itself for a type, such as `np.float32`. None for anything else:
        a list, a `dtype` object, a lazy array inside a tuple.
    """
    kind = type(arg)
    if kind in _PLAIN_SETTINGS:
        return (kind, arg)
    if kind is tuple:
        items = tuple(_setting_part(item) for item in arg)
        return None if None in items else (tuple, items)
    if kind is type:
        return (type, arg)
    return None


def basic_shape(shape: tuple[int, ...], index: Any) -> tuple[int, ...] | None:
    """Work out the shape NumPy gives an array indexed by whole numbers, slices, `None` and `...`.

    Args:
        shape: The shape of the array, every length known.
        index: What was written between the brackets.

    Returns:
        The shape of the view, or None wherever this function is not sure
        and NumPy has to be asked: an index out of bounds or too long, which
        NumPy raises for; a step of zero; a second `...`; and anything that
        is not exactly an `int`, a slice of `int`s and `None`s, `None` or
        `...`. A `bool`, a NumPy integer and a slice that holds a lazy
        array are among those.
    """
    kind = type(index)
    if kind is int:
        # The commonest index: one whole number, which takes one row.
        if shape and -shape[0] <= index < shape[0]:
            return shape[1:]
        return None
    if kind is tuple:
        items = index
    elif kind is slice:
        items = (index,)
    else:
        return None
    dimensions = len(shape)
    axis = 0
    whole = True
    lengths = []
    for item in items:
        kind = type(item)
        if kind is int:
            if axis >= dimensions or not -shape[axis] <= item < shape[axis]:
                return None
            axis += 1
        elif kind is slice:
            if axis >= dimensions:
                return None
            start, stop, step = item.start, item.stop, item.step
            if step is None and start is None and stop is None:
                lengths.append(shape[axis])
            elif (step is None or (type(step) is int and step != 0)) and (start is None or type(start) is int) \
                    and (stop is None or type(stop) is int):
                # Python's own arithmetic for a slice, which is NumPy's for whole numbers.
                lengths.append(len(range(*item.indices(shape[axis]))))
            else:
                return None
            axis += 1
        elif item is None:
            lengths.append(1)
        elif item is Ellipsis:
            if not whole:
                return None
            whole = False
            taken = sum(1 for other in items if other is not None and other is not Ellipsis)
            if taken > dimensions:
                return None
            # `...` stands for the axes the other items do not take.
            skipped = dimensions - taken
            lengths.extend(shape[axis:axis + skipped])
            axis += skipped
        else:
            return None
    lengths.extend(shape[axis:])
    return tuple(lengths)


def broadcasts_into(shape: tuple[int, ...], into: tuple[int, ...]) -> bool:
    """Say whether an array of one shape is sure to broadcast to another, as an assignment needs.

    Args:
        shape: The shape of what is assigned.
        into: The shape assigned into, every length known.

    Returns:
        True when `shape` has no more dimensions than `into` and each of its
        lengths, counted from the last, is 1 or the length it meets. False
        otherwise, which is not a refusal: NumPy also takes a value with
        more dimensions when the extra ones have length 1, and the caller
        asks NumPy about whatever this function does not vouch for. A
        length that is not known equals no known length, so it is not
        vouched for either.
    """
    ahead = len(into) - len(shape)
    if ahead < 0:
        return False
    for position, length in enumerate(shape):
        if length != 1 and length != into[ahead + position]:
            return False
    return True


def is_admissible(node: Node) -> bool:
    """Say whether a node's element type and shape allow deferring, as `_admissible` does, remembering the answer.

    Args:
        node: A node whose shape is known.

    Returns:
        What `_admissible` says of it.
    """
    number = node.spec or graph.spec(node)
    found = _admissible_specs.get(number)
    if found is None:
        found = _admissible(node)
        if number:
            _admissible_specs[number] = found
    return found


def _plan(operation: Callable[..., Any], args: tuple, kwargs: dict) -> Signature | None:
    """Work out the answer of `plan` for one call, from the rules and from NumPy, remembering nothing.

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
    if kind is None:
        return None
    # One pass over the operands for what `_admissible` and `graph.is_known` say of each.
    unknown = False
    for operand in _array_operands(kind, operation, args):
        if isinstance(operand, PYTHON_NUMBERS):
            continue
        if operand.dtype.kind not in DEFERRED_KINDS or 0 in operand.shape:
            return None
        unknown = unknown or not graph.is_known(operand.shape)
    if unknown and not _holds_for_any_length(kind, operation, args, kwargs):
        return None
    try:
        shape = _SHAPE_RULES[kind](operation, args, kwargs)
    except Exception:
        if unknown:
            return None  # The real length may be one the call is right for: NumPy decides, at once.
        if kind in (INDEXING, ORDERING):
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
        array, and the arrays are one of three things: arrays of integers
        and masks that have their values (`ndarray`s, lists) with pending
        arrays of integers; one pending mask alone; or pending positions in
        one dimension whose number is not known, all with the same mark,
        as the arrays of `np.nonzero(a)` are. In the last two cases the
        result has a length that is not known; anything it would have to
        broadcast with is left to NumPy.
    """
    items = index if isinstance(index, tuple) else (index,)
    arrays = masks = 0
    marks = set()
    for item in items:
        if isinstance(item, list):
            item = np.asarray(item)
        if isinstance(item, Node):
            if item.dtype.kind not in "biu" or 0 in item.shape:
                return False
            if not graph.is_known(item.shape):
                if item.dtype.kind == "b" or len(item.shape) != 1:
                    return False
                marks.add(item.shape[0])
            elif item.dtype.kind == "b":
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
    if marks:
        return len(marks) == 1 and arrays == 0 and masks == 0
    return (arrays > 0 and masks == 0) or (masks == 1 and arrays == 0)


def index_mark(item: Any) -> int | None:
    """Return the mark of the length that one item of an index selects, when that length is not known.

    Args:
        item: An item of an index.

    Returns:
        For a pending mask, the stand-in for how many elements it selects.
        For pending positions whose number is not known, the mark their own
        shape carries. None for anything else.
    """
    if not isinstance(item, Node):
        return None
    if item.dtype.kind == "b":
        return graph.unknown_length(item)
    return None if graph.is_known(item.shape) else item.shape[0]


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
        One of the kinds named above, or None for a call this layer does not defer.
    """
    if operation is np.dot:
        arrays = len(args) == 2 and not kwargs and all(isinstance(arg, (Node, np.ndarray)) for arg in args)
        return MATRIX_PRODUCT if arrays and all(len(arg.shape) in DOT_DIMENSIONS for arg in args) else None
    if operation in COPIES:
        return ELEMENTWISE if len(args) == 1 and not kwargs and isinstance(args[0], Node) else None
    if isinstance(operation, np.ufunc):
        if kwargs or len(args) != operation.nin or not _all_operands(args):
            return None  # where=, dtype=, casting= and the like are NumPy's to handle now.
        if operation is np.matmul:
            return MATRIX_PRODUCT
        return ELEMENTWISE if operation.signature is None and operation.nout == 1 else None
    if operation in SELECTIONS:
        return ELEMENTWISE if len(args) == 3 and not kwargs and _all_operands(args) else None
    if operation in ROUNDINGS:
        return ELEMENTWISE if _is_plain_rounding(args, kwargs) else None
    if operation in CASTS:
        return ELEMENTWISE if _is_plain_cast(operation, args, kwargs) else None
    if operation in ORDERINGS:
        return ORDERING if _is_plain_ordering(args, kwargs) else None
    if operation in NONZEROS:
        # Of an array with dimensions only: NumPy refuses `np.nonzero` of a number, and has
        # answers of its own for the other two.
        return NONZERO if len(args) == 1 and not kwargs and isinstance(args[0], Node) and args[0].shape != () else None
    if operation in REDUCTIONS:
        return REDUCTION if _is_plain_reduction(args, kwargs) else None
    if operation in POSITIONS:
        return REDUCTION if _is_plain_reduction(args, kwargs) and _names_one_axis(args, kwargs) else None
    if operation is np.count_nonzero:
        counted = COUNT_IS_A_NUMPY_SCALAR or not _reduces_everything(args, kwargs)
        return REDUCTION if counted and _is_plain_reduction(args, kwargs) else None
    if operation in SCANS:
        return SCAN if _is_plain_scan(args, kwargs) else None
    if operation in SEARCHES:
        return SEARCH if _is_plain_search(args, kwargs) else None
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


def _is_plain_cast(operation: Callable[..., Any], args: tuple, kwargs: dict) -> bool:
    """Say whether a cast is one this layer defers: one NumPy always answers with a new array.

    Args:
        operation: `np.ndarray.astype`, standing for the method, or `np.astype`.
        args: The operand, and the element type when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True for one node cast to an element type this layer defers, given
        once, with `copy=` absent or True, or False when the element type is
        another than the operand's: NumPy then has to make a new array. With
        `copy=False` and the operand's own type it returns the array itself,
        which is not an operation of a graph but another name for memory.
        False also for `order=`, `casting=`, `subok=` or `device=` with a
        value other than NumPy's default, for the function on an operand
        without dimensions (some versions of NumPy refuse a number there,
        and whether it is one is not known before computing), and for
        anything NumPy does not read as an element type, whose error is
        NumPy's to raise.
    """
    named = "dtype" in kwargs
    if len(args) + named != 2 or not isinstance(args[0], Node):
        return False
    if operation is not np.ndarray.astype and (named or args[0].shape == ()):
        return False  # `np.astype` takes the type by position only, and no number.
    settings = {key: value for key, value in kwargs.items() if key not in ("dtype", "copy")}
    for key, value in settings.items():
        if key not in CAST_DEFAULTS or not (isinstance(value, type(CAST_DEFAULTS[key])) and value == CAST_DEFAULTS[key]):
            return False
    wanted = kwargs["dtype"] if named else args[1]
    if isinstance(wanted, Node):
        return False
    try:
        target = np.dtype(wanted)
    except TypeError:
        return False
    copy = kwargs.get("copy", True)
    return target.kind in DEFERRED_KINDS and (copy is True or (copy is False and target != args[0].dtype))


def _is_plain_rounding(args: tuple, kwargs: dict) -> bool:
    """Say whether a rounding's arguments are ones this layer defers.

    Args:
        args: The operand, and the number of decimals when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True for one node as the operand and, at most once, a whole number of
        decimals, by position or as `decimals=`. The result then has the
        operand's shape; the decimals are a setting, not an operand
        (`ONE_OPERAND`). Anything else is NumPy's to handle at once: `out=`, which is
        a write, decimals given twice, and decimals that are not a whole
        number, for which NumPy raises.
    """
    if not 1 <= len(args) <= 2 or not isinstance(args[0], Node) or not set(kwargs) <= {"decimals"}:
        return False
    if len(args) == 2 and kwargs:
        return False
    decimals = args[1] if len(args) == 2 else kwargs.get("decimals", 0)
    return isinstance(decimals, (int, np.integer))


def _is_plain_ordering(args: tuple, kwargs: dict) -> bool:
    """Say whether a partition's arguments are ones this layer defers.

    Args:
        args: The operand, then `kth` and the axis when they were given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True for one node as the operand, with `kth` and the axis as whole
        numbers written in the program (`kth` may be a list or tuple of
        them, the axis may be None), and `kind=` and `order=` absent or at
        NumPy's default. Whether `kth` and the axis fit the array, and
        whether each was given exactly once, is not checked here: the shape
        rule asks NumPy, which raises its own error. Anything else is made at
        once: a `kth` that is an array or has to be computed, another
        `kind=`, an `order=`.
    """
    if not args or len(args) > 3 or not isinstance(args[0], Node):
        return False
    for key, value in kwargs.items():
        if key in ("kth", "axis"):
            continue
        if key not in ORDERING_DEFAULTS or not (isinstance(value, type(ORDERING_DEFAULTS[key])) and value == ORDERING_DEFAULTS[key]):
            return False
    kth = args[1] if len(args) > 1 else kwargs.get("kth", 0)
    axis = args[2] if len(args) > 2 else kwargs.get("axis")
    positions = kth if isinstance(kth, (list, tuple)) else (kth,)
    return all(isinstance(item, (int, np.integer)) for item in (*positions, *(() if axis is None else (axis,))))


def _is_plain_scan(args: tuple, kwargs: dict) -> bool:
    """Say whether a running total's arguments are ones this layer defers.

    Args:
        args: The operand, and the axis when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True for one node with at least one dimension as the operand, and one
        axis or none, given at most once. `dtype=` is NumPy's to handle at
        once, and so is an operand without dimensions.
    """
    if not 1 <= len(args) <= 2 or not isinstance(args[0], Node) or not args[0].shape:
        return False
    if len(args) == 2 and "axis" in kwargs:
        return False
    return set(kwargs) <= SCAN_KEYWORDS and _names_one_axis(args, kwargs)


def _is_plain_search(args: tuple, kwargs: dict) -> bool:
    """Say whether a search's arguments are ones this layer defers.

    Args:
        args: The sorted array, what is looked for, and `side` when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True for a sorted array of one dimension, something an
        element-by-element call could take to look for, a node among the
        two, and at most a `side` of "left" or "right" given once. With
        `sorter=`, or a sorted array of another number of dimensions, the
        call and its error are NumPy's.
    """
    if not 2 <= len(args) <= 3 or not all(_is_operand(arg) for arg in args[:2]):
        return False
    if not any(isinstance(arg, Node) for arg in args[:2]) or len(_shape(args[0])) != 1:
        return False
    if (len(args) == 3 and kwargs) or not set(kwargs) <= {"side"}:
        return False
    side = args[2] if len(args) == 3 else kwargs.get("side", SEARCH_SIDES[0])
    return isinstance(side, str) and side in SEARCH_SIDES


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


def _reduces_everything(args: tuple, kwargs: dict) -> bool:
    """Say whether a reduction is written so that NumPy answers with one number, not an array.

    Args:
        args: The operand, and the axis when it was given by position.
        kwargs: Keyword arguments as written.

    Returns:
        True with no axis, or `None` for it, and no `keepdims`.
    """
    axis = args[1] if len(args) > 1 else kwargs.get("axis")
    return axis is None and not kwargs.get("keepdims", False)


def _holds_for_any_length(kind: str, operation: Callable[..., Any], args: tuple, kwargs: dict) -> bool:
    """Say whether a call's shape rule is right whatever a length that is not known turns out to be.

    Args:
        kind: The call's kind.
        operation: The NumPy callable.
        args: Its positional arguments, one of them with such a length.
        kwargs: Its keyword arguments.

    Returns:
        True for an element-by-element call, a matrix product, a
        reduction, a running total and a search: where their shape rule
        succeeds on the stand-in, it succeeds on any length, and a running
        total of all the elements of several dimensions gives a product of
        lengths, which `_is_sound` turns down. For views: True for the ones that move or add
        axes, and for an index that takes the whole of the axis in question
        with `:`. False for `np.reshape` and `np.ravel`, for `np.squeeze`
        without an axis (the length may be 1), for an index that takes part
        of that axis (`x[mask][:5]` has five elements or fewer), for an
        index of arrays, for a partition, whose `kth` is checked against
        the length, and for the positions of what is not zero, whose mark
        is made for an operand of a known shape.
    """
    if kind in (ELEMENTWISE, MATRIX_PRODUCT, REDUCTION, SCAN, SEARCH):
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
    unknown = 0
    for length in shape:
        if length >= graph.UNKNOWN_FLOOR:
            unknown += 1
            if unknown > 1 or not graph.is_issued(length):
                return False
    return True


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


def _all_operands(args: tuple) -> bool:
    """Say whether every argument is something an element-by-element call can take.

    Args:
        args: The positional arguments of a call.

    Returns:
        What `_is_operand` says of each, together.
    """
    for arg in args:
        if not _is_operand(arg):
            return False
    return True


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


def _array_operands(kind: str, operation: Callable[..., Any], args: tuple) -> tuple:
    """Return the arguments of a call that are arrays or numbers, not axes, indices or element types.

    Args:
        kind: The call's kind.
        operation: The NumPy callable.
        args: Its positional arguments.

    Returns:
        Every argument for an element-by-element call or a matrix product;
        the first two for a search; the first alone for a reduction, a
        running total, a view, a rounding or a cast. They come first among
        the arguments, whatever the call.
    """
    if kind == SEARCH:
        return args[:2]
    return args if kind in (ELEMENTWISE, MATRIX_PRODUCT) and operation not in ONE_OPERAND else args[:1]


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
        operation: The NumPy callable.
        args: The operands, and for a rounding or a cast the setting after its one operand.
        kwargs: Not used.

    Returns:
        The broadcast shape. For a rounding or a cast, the shape of its operand.

    Raises:
        ValueError: When the shapes do not broadcast.
    """
    shapes = [_shape(arg) for arg in _array_operands(ELEMENTWISE, operation, args)]
    # The commonest case needs no arithmetic: every operand has one shape, or is a number. A length that is not
    # known goes to NumPy all the same, which refuses a size that does not fit the machine.
    arrays = [shape for shape in shapes if shape != ()]
    if not arrays:
        return ()
    if all(shape == arrays[0] for shape in arrays) and graph.is_known(arrays[0]):
        return arrays[0]
    return np.broadcast_shapes(*shapes)


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


def _scanned_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of a running total: the operand's along an axis, or all its elements in one dimension without one.

    Args:
        operation: The running total. Not used.
        args: The operand, then the axis if it was given by position.
        kwargs: `axis` if it was given by keyword.

    Returns:
        The shape NumPy gives. The number of elements is multiplied as
        Python integers, which do not overflow on a length that is a stand-in.

    Raises:
        AxisError: For an axis the operand does not have.
    """
    shape, axis = _shape(args[0]), args[1] if len(args) > 1 else kwargs.get("axis")
    if axis is None:
        return (math.prod(shape),)
    normalize_axis_tuple(axis, len(shape))
    return shape


def _searched_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of a search: one position for each element looked for.

    Args:
        operation: The search. Not used.
        args: The sorted array, then what is looked for.
        kwargs: Not used.

    Returns:
        The shape of what is looked for, whatever the sorted array's length.
    """
    return _shape(args[1])


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
    shape = args[0].shape
    # The shape of a view does not depend on the element type, so where every length is known the stand-in is an
    # array of an element type of no size, which NumPy makes in a tenth of the time `np.broadcast_to` takes. A
    # length that is not known is larger than NumPy lets an array of its own be.
    viewed = np.empty(shape, NO_SIZE) if graph.is_known(shape) else ghost(args[0])
    return operation(viewed, *args[1:], **kwargs).shape


def _ordered_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of a partition: NumPy's own answer, on an array of no size.

    Args:
        operation: `np.partition`.
        args: The node, then `kth` and the axis as written.
        kwargs: Keyword arguments as written.

    Returns:
        The shape NumPy gives: the operand's, or its elements in one
        dimension for `axis=None`. The result is a new array, so the
        stand-in has an element type of no size and takes no memory, as for
        indexing by arrays. NumPy checks `kth` and the axis before it moves
        anything, and then still walks the elements, which costs a few
        nanoseconds each when they have no size.

    Raises:
        Exception: NumPy's, for a `kth` out of bounds, an axis the array
            lacks, a `kth` that is not a whole number, or an argument given
            twice or not at all.
    """
    return operation(np.empty(args[0].shape, NO_SIZE), *args[1:], **kwargs).shape


def _nonzero_shape(operation: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[int, ...]:
    """Shape of the positions of what is not zero: one length that is not known.

    Args:
        operation: `np.nonzero`, `np.flatnonzero` or `np.argwhere`.
        args: The one node.
        kwargs: Not used.

    Returns:
        `(?,)` for `np.flatnonzero`, and for each array of the tuple
        `np.nonzero` gives; `(?, ndim)` for `np.argwhere`. The mark is the
        operand's, the same for all three: it stands for how many of the
        node's elements are not zero. For a mask that is how many it
        selects, so `x[mask]` and `np.flatnonzero(mask)` carry one mark and
        are known to have one length.
    """
    count = graph.unknown_length(args[0])
    return (count, len(args[0].shape)) if operation is np.argwhere else (count,)


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
        Pending positions whose number is not known are stood in for by no
        positions at all, and their mark is put where the 0 is in the same
        way. None when that place cannot be told apart, because a slice of
        the index selects nothing as well.

    Raises:
        IndexError: NumPy's, for an index out of bounds, index arrays that do
            not broadcast or a mask of the wrong shape.
    """
    index = args[1]
    items = index if isinstance(index, tuple) else (index,)
    quick = _positions_shape(args[0].shape, items, isinstance(index, tuple))
    if quick is not None:
        return quick
    stood_in = tuple(index_stand_in(item) for item in items)
    shape = np.broadcast_to(np.zeros((), NO_SIZE), args[0].shape)[stood_in if isinstance(index, tuple) else stood_in[0]].shape
    mark = next((index_mark(item) for item in items if index_mark(item) is not None), None)
    if mark is None:
        return shape
    if shape.count(0) != 1:
        return None
    return tuple(mark if length == 0 else length for length in shape)


def _positions_shape(shape: tuple[int, ...], items: tuple, is_tuple: bool) -> tuple[int, ...] | None:
    """Shape of an array indexed by large arrays of positions, without NumPy walking them; or None.

    NumPy's indexing of an array of an element type of no size walks every
    position of the index: 8.7 ms for a table of 3.3 million positions on
    the Intel Core i5-14600KF, where taking the elements themselves takes
    2.5 ms. So the shape `_indexed_shape` asks for cost more than the call
    it stands for (SALib's `A[r]`, the speech features' frames). Here NumPy
    is asked twice, on stand-ins of one and of two positions along each
    dimension of the arrays' broadcast shape; the dimensions where the two
    answers differ are where that shape goes. Whether the positions of an
    array that has its values are in bounds is asked of its least and
    largest element, of its own element type, which NumPy checks as it
    would the whole array. Pending positions are zeros, as in
    `index_stand_in`.

    Anything that is not plainly such an index is left to `_indexed_shape`:
    a mask, a list, positions whose number is not known, unsigned positions
    of eight bytes, an empty array or a dimension of length 0, small
    arrays, and every index NumPy refuses here, so that its error is raised
    by NumPy's own walk with the message the eager line gives.

    Args:
        shape: The shape of the array indexed. All its lengths are known.
        items: The items of the index.
        is_tuple: Whether the index was a tuple of these items, or the one item alone.

    Returns:
        The shape NumPy gives, or None to have it found the general way.
    """
    if 0 in shape:
        return None
    arrays, elements = [], 0
    for place, item in enumerate(items):
        if isinstance(item, Node):
            if item.dtype.kind not in _POSITION_KINDS or not graph.is_known(item.shape):
                return None
        elif isinstance(item, np.ndarray):
            kind = item.dtype.kind
            if kind not in _POSITION_KINDS or (kind == "u" and item.dtype.itemsize >= _WIDE_UNSIGNED) or item.size == 0:
                return None
        elif isinstance(item, (list, tuple)) or (hasattr(item, "__array__") and not isinstance(item, np.generic)):
            return None
        else:
            continue
        arrays.append(place)
        elements += math.prod(item.shape)
    if elements < QUICK_POSITIONS:
        return None
    try:
        broadcast = np.broadcast_shapes(*(items[place].shape for place in arrays))
        if 0 in broadcast or len(broadcast) > _QUICK_DIMENSIONS:
            return None
        ghost = np.broadcast_to(np.zeros((), NO_SIZE), shape)

        def answer(stand_ins: dict[int, Any]) -> tuple[int, ...]:
            """NumPy's shape for the index with some of its items replaced."""
            parts = tuple(stand_ins.get(place, item) for place, item in enumerate(items))
            return ghost[parts if is_tuple else parts[0]].shape

        ones = answer({place: np.zeros((1,) * len(broadcast), np.intp) for place in arrays})
        twos = answer({place: np.zeros((2,) * len(broadcast), np.intp) for place in arrays})
        bounds = {place: (np.zeros((), np.intp) if isinstance(items[place], Node)
                          else np.array([items[place].min(), items[place].max()], items[place].dtype)) for place in arrays}
        answer(bounds)
    except Exception:
        # NumPy refuses the index: its own walk below raises the error with the eager line's message.
        return None
    if len(ones) != len(twos):
        return None
    differ = [axis for axis, (one, two) in enumerate(zip(ones, twos)) if one != two]
    if len(differ) != len(broadcast) or (differ and differ[-1] - differ[0] != len(differ) - 1):
        return None
    if not differ:
        return ones
    return ones[:differ[0]] + broadcast + ones[differ[-1] + 1:]


# The element kinds of arrays of positions that `_positions_shape` takes, and the size from which it declines
# unsigned ones: a position of eight bytes above the largest signed one is not ordered as NumPy casts it.
_POSITION_KINDS = frozenset("iu")
_WIDE_UNSIGNED = 8
# How many positions an index must hold in all for `_positions_shape` to be worth asking: below this NumPy's own
# walk takes a few microseconds, and the two stand-ins and the least and largest element cost as much.
QUICK_POSITIONS = 4096
# The most dimensions the arrays of an index may broadcast to for `_positions_shape`: its stand-in of two positions
# along each holds 2 ** this many.
_QUICK_DIMENSIONS = 8


def index_stand_in(item: Any) -> Any:
    """Return what stands in for one item of an index while NumPy works out a shape.

    Args:
        item: An item of an index.

    Returns:
        For a pending array, one zero or one False broadcast to its shape,
        which takes no memory; for pending positions whose number is not
        known, no positions; anything else as it is.
    """
    if isinstance(item, Node) and not graph.is_known(item.shape):
        return np.zeros(0, np.intp)
    if isinstance(item, Node):
        return np.broadcast_to(np.zeros((), np.bool_ if item.dtype.kind == "b" else np.intp), item.shape)
    return item


# The rule that gives the shape of each kind.
_SHAPE_RULES = {ELEMENTWISE: _elementwise_shape, MATRIX_PRODUCT: _matmul_shape,
                REDUCTION: _reduced_shape, VIEW: _view_shape, INDEXING: _indexed_shape, ORDERING: _ordered_shape,
                NONZERO: _nonzero_shape, SCAN: _scanned_shape, SEARCH: _searched_shape}


def _element_type(kind: str, operation: Callable[..., Any], args: tuple, kwargs: dict) -> np.dtype:
    """Element type of a call's result, from NumPy itself.

    Args:
        kind: The call's kind.
        operation: The NumPy callable.
        args: Its positional arguments.
        kwargs: Its keyword arguments.

    Returns:
        The operand's type for a view, a selection by arrays and a
        partition. Otherwise the type of the real call
        made on probes, so `float32 * 2.0` stays `float32` and the sum of
        `int8` is `int64`, because NumPy says so. Only the array operands
        are stood in for: an axis or an element type is passed as written.

    Raises:
        Exception: Whatever NumPy raises for these types, such as a Python
            integer that does not fit the array's type.
    """
    if kind in (VIEW, INDEXING, ORDERING):
        return args[0].dtype
    if type(operation) is np.ufunc:
        resolved = _resolved_type(operation, args)
        if resolved is not None:
            if len(_types) >= TYPE_MEMORY_LIMIT:
                _types.clear()
            _types[(operation, *(arg if type(arg) is type else id(arg) for arg in _resolver_types(args)))] = (resolved, kind)
            return resolved
    if kind == NONZERO:
        return np.flatnonzero(probe(args[0])).dtype  # The three answer with one type, and `np.nonzero` with a tuple.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # A probe holds zeros: 0 / 0 is not the user's doing.
        operands = len(_array_operands(kind, operation, args))
        return np.result_type(operation(*(probe(arg) for arg in args[:operands]), *args[operands:], **kwargs))


def _resolved_type(operation: np.ufunc, args: tuple) -> np.dtype | None:
    """Ask NumPy for a ufunc's result type from the operands' types alone, where that is the whole answer.

    `ufunc.resolve_dtypes` is the step of a ufunc call that picks the loop
    and the result's type. For operands that are arrays and numbers with a
    fraction, nothing else in a call on zeros can change the answer or
    raise, so the call itself, with its arrays of one element, its error
    state and its warning filter, need not be made. A test holds the two
    ways to each other for every ufunc of NumPy.

    Args:
        operation: A ufunc the rules have accepted.
        args: Its operands.

    Returns:
        The element type of the result, or None wherever the call has to be
        made to know: a whole number among the operands, whose value can
        make NumPy raise; an element type that is not NumPy's own object for
        it (another byte order, metadata), where the call's result need not
        be the resolver's; and whenever the resolver raises, so that the
        error comes from the call, in the call's words.
    """
    given = _resolver_types(args)
    if given is None:
        return None
    try:
        return operation.resolve_dtypes((*given, None))[-1]
    except Exception:
        return None


def _resolver_types(args: tuple) -> list | None:
    """Write a ufunc's operands down as NumPy's resolver takes them.

    Args:
        args: The operands.

    Returns:
        For each operand, its `dtype` object if it is an array or a NumPy
        scalar with a fraction, and the type `float`, `complex` or `int` for
        a Python number, which NumPy lets take the array's type. None when an
        operand is anything else, has an element type that is not NumPy's
        own object for it, or is an `int` outside the bounds within which
        its value cannot make NumPy raise (`SMALL_WHOLE`, `LARGE_WHOLE`).
    """
    given = []
    small = fractional = True
    for arg in args:
        kind = type(arg)
        if kind in _FRACTIONAL_NUMBERS:
            given.append(kind)
            continue
        if kind is int:
            if not -LARGE_WHOLE <= arg <= LARGE_WHOLE:
                return None
            small = small and 0 <= arg <= SMALL_WHOLE
            given.append(kind)
            continue
        if kind is not Node and kind is not np.ndarray and kind not in _FRACTIONAL_SCALARS:
            return None
        dtype = arg.dtype
        if np.dtype(dtype.type) is not dtype:
            return None
        fractional = fractional and dtype.kind in "fc"
        given.append(dtype)
    return given if small or fractional else None


def keep(key: tuple, shape: tuple[int, ...], dtype: np.dtype, kind: str, types: tuple | None = None) -> Signature:
    """Keep an answer the C core worked out for a ufunc: the element type from NumPy's resolver, the shape by its own rule.

    Args:
        key: What `_key` writes down of the call.
        shape: The shape the core's rule gave: the operands' shapes broadcast, or the matrix product's.
        dtype: The element type NumPy's resolver gave for the operands' types.
        kind: The call's kind.
        types: The key of `_types` to keep the element type under, when the
            core has just asked the resolver; None when it found the type there.

    Returns:
        The signature as `_remember` keeps it.
    """
    if types is not None:
        if len(_types) >= TYPE_MEMORY_LIMIT:
            _types.clear()
        _types[types] = (dtype, kind)
    return _remember(key, Signature(shape, dtype, kind))


def first(key: tuple | None, operation: Callable[..., Any], args: tuple, kwargs: dict) -> Signature | None:
    """Answer a call the C core found no remembered answer for: what `plan` does once its key is not found.

    Args:
        key: The key the core wrote for the call, as `_key` writes it without
            placeholders; None when it wrote none, and `plan` is to see to everything.
        operation: The NumPy callable.
        args: Its positional arguments, with a `Node` for each lazy operand.
        kwargs: Its keyword arguments.

    Returns:
        What `plan` returns.

    Raises:
        Exception: Whatever NumPy raises for these shapes and types.
    """
    if key is None:
        return plan(operation, args, kwargs)
    signature = _plan(operation, args, kwargs)
    return None if signature is None else _remember(key, signature)


def probe(operand: Any) -> Any:
    """Return a stand-in with an operand's element type and number of dimensions, and one element.

    Args:
        operand: A node, an `ndarray`, a NumPy scalar or a Python number, or
            an argument that is no operand: an axis, a tuple of axes, `None`, a `side`.

    Returns:
        An array of zeros of shape (1, ..., 1) for an array; anything else
        as it is, so NumPy sees a Python number as one and an axis as written.
    """
    if not isinstance(operand, (Node, np.ndarray)):
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
