"""The graph: one node per deferred operation, and the list of nodes not computed yet.

A node is either pending (it has an operation and no value), computed (it has
a value and has dropped its operands) or failed (computing it raised). A leaf
is a node born computed. Nodes refer to their operands; nothing refers to a
node except its consumers and the `LazyArray` a user holds, so an
intermediate nobody holds disappears when its consumer has been computed.
"""
from __future__ import annotations

import itertools
import logging
import os
import sys
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

# A frame in one of these directories is the layer's own code or NumPy's, never
# the line a user wrote. The layer's directory is matched exactly, so that the
# example programs in its `programs/` subdirectory count as user code.
_LAYER_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
_NUMPY_DIRECTORY = os.path.dirname(os.path.abspath(np.__file__)) + os.sep

# Pending nodes above which everything pending is computed before another is
# added, so a long loop that never gathers cannot hold an unbounded graph.
PENDING_LIMIT = 100_000

# Guards the pending list and every evaluation. Re-entrant, because computing
# everything pending computes nodes one by one under the same lock.
LOCK = threading.RLock()

# Hands out node numbers. Operands are always numbered below their consumer.
_numbers = itertools.count()

# A length that is not known until a mask is computed: how many elements the
# mask selects. This prototype stands it in, inside a shape, as an integer no
# array on this machine can have (2**40 elements of one byte are a terabyte):
# this floor plus the mask's node number. NumPy's own shape arithmetic then
# carries it through broadcasting, products and reductions, and two
# selections by the same mask have the same length. The package is to have a
# type of its own for it.
UNKNOWN_FLOOR = 2 ** 40
# Node numbers stay below this, so a stand-in is below the floor plus this.
# Twice a stand-in, which a reshape could produce, is not: it stands for nothing.
UNKNOWN_SPAN = 2 ** 32
# What a length that is not known is written as, for a reader.
UNKNOWN_TEXT = "?"

# The package this module is in: "hotkernels.lazy" once it is in the package.
_PACKAGE = __name__.rpartition(".")[0]
# Two loggers, so that standard logging configuration chooses what is seen.
# A call the layer could not defer, which NumPy made at once: logged at INFO.
eager_log = logging.getLogger(f"{_PACKAGE}.eager")
# A graph computed early, by a write into a lazy array or because a value was
# needed without `gather()`: logged at DEBUG. So the package's logger at INFO
# shows the eager calls alone, and at DEBUG the cuts as well.
cut_log = logging.getLogger(f"{_PACKAGE}.cut")

# What can make the layer compute: the kinds of `Trigger`.
GATHER, AT_ONCE, WRITE, NEEDED, LIMIT, PENDING = "gather", "at once", "write", "needed", "limit", "pending"

# One line of the log: what happened, the call, the file's name and the line.
_FALLBACK_LINE = "%s: %s at %s:%d"
# The word a fallback of each kind starts its line with.
FALLBACK_WORDS = {AT_ONCE: "eager", WRITE: "write", NEEDED: "needed"}


class NotGathered(TypeError):
    """Raised when something needs the values of a strict lazy array and `gather()` was not called.

    A `TypeError`, because that is what Python raises when an object cannot
    be turned into the number or the array that was asked for.
    """


@dataclass(frozen=True)
class Origin:
    """The line of the user's code that wrote an operation."""

    # File of the user's code.
    filename: str
    # Line in that file.
    lineno: int
    # Name of the module the line is in, which the warnings machinery reports.
    module: str
    # That module's `__warningregistry__`, where Python remembers which
    # warnings it has already shown for a line.
    registry: dict = field(compare=False, repr=False)


@dataclass(frozen=True)
class Trigger:
    """What made the layer compute."""

    # One of GATHER, AT_ONCE (a call outside the deferred set), WRITE (a write
    # into a lazy array), NEEDED (something needed values and `gather()` was
    # not called), LIMIT (too many nodes pending) and PENDING (nothing did: the
    # nodes are not computed).
    kind: str
    # The call as a reader would name it: "gather", "np.argmax", "x[index] = v".
    call: str


class Observer(Protocol):
    """Something that is told what the layer does: the recording mode of `record.py`."""

    def made(self, node: "Node", source: str) -> None:
        """Hear that a node was made.

        Args:
            node: The node, pending or a leaf.
            source: For a leaf, what made the array: "array", "np.zeros", "np.argmax".
        """

    def computing(self, nodes: list["Node"]) -> None:
        """Hear which nodes are about to be computed, while they still have their operands.

        Args:
            nodes: The pending nodes one computation will compute.
        """

    def fell_back(self, kind: str, call: str, origin: "Origin") -> None:
        """Hear that the layer computed without being asked.

        Args:
            kind: AT_ONCE, WRITE or NEEDED.
            call: The call as a reader would name it.
            origin: The line of the user's code where it happened.
        """

    def computed(self, numbers: list[int], failed: set[int], roots: list[int], trigger: Trigger) -> None:
        """Hear that one computation is over.

        Args:
            numbers: Numbers of the nodes it computed or failed, in the order it took them.
            failed: The numbers among them whose computation raised.
            roots: Numbers of the nodes whose values were asked for.
            trigger: What made the layer compute.
        """


# The observer in effect, or None. One at a time; `record.py` sets it for the
# length of a `with` block.
observer: Observer | None = None


@dataclass(eq=False, slots=True, weakref_slot=True)
class Node:
    """One operation of the graph, or an array that already exists."""

    # A count that rises with every node. Ascending order is a valid order of
    # evaluation, and it is the order the eager program would have run in.
    number: int
    # What `explain()` and error messages call the operation: "add", "matmul",
    # "getitem", or "array" for a leaf.
    name: str
    # The NumPy callable this node stands for. None once the node is computed
    # or has failed, and for a leaf.
    operation: Callable[..., Any] | None
    # Positional arguments of the call. A `Node` is an operand in the graph; an
    # `ndarray` is held by reference; a Python number, an index or an axis is
    # kept exactly as written, so NumPy applies its own rules to it.
    args: tuple
    # Keyword arguments of the call, none of them a `Node`.
    kwargs: dict
    # Shape of the result, known when the node is made. One length may be a
    # stand-in (`unknown_length`) until the node is computed: how many
    # elements a pending mask selects.
    shape: tuple[int, ...]
    # Element type of the result, known when the node is made.
    dtype: np.dtype
    # Where the user wrote the operation. None for a leaf.
    origin: Origin | None = None
    # The `np.errstate` in effect when the operation was written. None for a leaf.
    errstate: dict | None = None
    # The result: an `ndarray`, or a NumPy scalar where NumPy returns one.
    # Meaningful only when `operation` and `failure` are both None.
    value: Any = None
    # The exception computing this node raised, or inherited from an operand.
    failure: BaseException | None = None
    # For a deferred operation, its kind as `rules` names it: "elementwise",
    # "matmul", "reduction", "view", "indexing", "ordering", "nonzero" or "write". For a leaf, AT_ONCE when it is the
    # result of a call made at once, and empty for an array from outside.
    kind: str = ""
    # True for an array that refuses to give its values without `gather()`:
    # one made by a strict stand-in, or computed from such an array.
    strict: bool = False
    # True when the memory of the value may be known outside the layer: an
    # `ndarray` the user handed in, one `gather()` handed out, one that a call
    # made at once may have kept a view of. A write into such memory has to
    # happen in that memory, at once; a write into any other can be deferred.
    exposed: bool = False

    @property
    def pending(self) -> bool:
        """Whether the node still has to be computed."""
        return self.operation is not None

    @property
    def state(self) -> str:
        """One word for the node: "pending", "computed" or "failed"."""
        if self.failure is not None:
            return "failed"
        return "pending" if self.pending else "computed"


@dataclass
class Counts:
    """What the layer has done, for the tests and for a later report."""

    # Operations added to the graph.
    deferred: int = 0
    # Calls made at once and writes into lazy arrays, by the call's name.
    at_once: dict[str, int] = field(default_factory=dict)
    # Times values were needed without `gather()`, by what needed them.
    needed: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        """Start counting from nothing."""
        self.deferred = 0
        self.at_once = {}
        self.needed = {}
        SITES.clear()


# The one set of counters of the process.
COUNTS = Counts()

# Every node that has no value yet, held weakly: pending work that the program
# drops without gathering is forgotten, never computed.
_pending: weakref.WeakSet[Node] = weakref.WeakSet()


# The fallbacks already logged, as (kind, call, file, line): each line of the
# user's code is reported once, however often it runs.
_logged: set[tuple[str, str, str, int]] = set()
# How often the layer computed without being asked, by (kind, call, file,
# line): every occurrence, for `report.lazy_report`.
SITES: dict[tuple[str, str, str, int], int] = {}


def fallback(kind: str, call: str) -> None:
    """Count one time the layer computed without being asked, and log it once per line of the user's code.

    Args:
        kind: AT_ONCE, WRITE or NEEDED.
        call: The call as a reader would name it: "np.argmax", "x[index] = v", "float()".
    """
    counts = COUNTS.needed if kind == NEEDED else COUNTS.at_once
    counts[call] = counts.get(call, 0) + 1
    origin = caller()
    site = (kind, call, origin.filename, origin.lineno)
    SITES[site] = SITES.get(site, 0) + 1
    if observer is not None:
        observer.fell_back(kind, call, origin)
    logger, level = (eager_log, logging.INFO) if kind == AT_ONCE else (cut_log, logging.DEBUG)
    if logger.isEnabledFor(level) and site not in _logged:
        _logged.add(site)
        logger.log(level, _FALLBACK_LINE, FALLBACK_WORDS[kind], call, os.path.basename(origin.filename), origin.lineno)


def leaf(value: Any, source: str = "array", kind: str = "", strict: bool = False, exposed: bool = False) -> Node:
    """Make a node for an array or NumPy scalar that already exists.

    Args:
        value: An `ndarray` or a NumPy scalar. It is held by reference, not copied.
        source: What made the array, for an observer: "array" for one that
            came from outside, or the call's name, such as "np.zeros".
        kind: AT_ONCE for the result of a call made at once; empty otherwise.
        strict: Whether the array refuses to give its values without `gather()`.
        exposed: Whether the array's memory may be known outside the layer.

    Returns:
        A computed node with that value.
    """
    node = Node(number=next(_numbers), name="array", operation=None, args=(), kwargs={},
                shape=value.shape, dtype=value.dtype, value=value, kind=kind, strict=strict, exposed=exposed)
    if observer is not None:
        observer.made(node, source)
    return node


def deferred(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict,
             shape: tuple[int, ...], dtype: np.dtype, kind: str, strict: bool = False,
             origin: "Origin | None" = None) -> Node:
    """Make a pending node and add it to the pending list.

    Args:
        operation: The NumPy callable to make later.
        name: What to call it in `explain()` and in messages.
        args: Its positional arguments, with `Node`s for operands in the graph.
        kwargs: Its keyword arguments.
        shape: Shape of the result.
        dtype: Element type of the result.
        kind: The kind of deferred operation, as `rules` names it.
        strict: Whether the array refuses to give its values without `gather()`.
        origin: The line that wrote the operation, when it is not the line
            running now: a view is made again, after a write, for the line
            that first made it. None for the caller's line.

    Returns:
        The node. It remembers the line and the `np.errstate` in effect.
    """
    node = Node(number=next(_numbers), name=name, operation=operation, args=args, kwargs=kwargs,
                shape=shape, dtype=dtype, origin=origin or caller(), errstate=np.geterr(), kind=kind, strict=strict)
    with LOCK:
        _pending.add(node)
    COUNTS.deferred += 1
    if observer is not None:
        observer.made(node, name)
    return node


def unknown_length(mask: Node) -> int:
    """Return the stand-in for the number of elements a pending mask selects.

    The same stand-in serves for how many elements of any node are not zero
    (`np.flatnonzero(a)`), which for a mask is the number it selects.

    Args:
        mask: The node of a boolean array, or of any array whose elements that are not zero are counted.

    Returns:
        An integer at or above UNKNOWN_FLOOR: the same one each time for the
        same node, and a different one for every other node.
    """
    return UNKNOWN_FLOOR + mask.number


def is_known(shape: tuple[int, ...]) -> bool:
    """Say whether every length of a shape is known.

    Args:
        shape: A node's shape.

    Returns:
        False when a length is a stand-in for one that depends on values.
    """
    return all(length < UNKNOWN_FLOOR for length in shape)


def is_issued(length: int) -> bool:
    """Say whether an integer is a stand-in that `unknown_length` gives.

    Args:
        length: A length at or above UNKNOWN_FLOOR.

    Returns:
        False for an integer that arithmetic on a stand-in produced, which stands for nothing.
    """
    return UNKNOWN_FLOOR <= length < UNKNOWN_FLOOR + UNKNOWN_SPAN


def shown(shape: tuple[int, ...]) -> tuple[int | None, ...]:
    """Return a shape as it is shown and recorded.

    Args:
        shape: A node's shape.

    Returns:
        The same lengths, with None for each that is not known.
    """
    return tuple(length if length < UNKNOWN_FLOOR else None for length in shape)


def shape_text(shape: tuple[int, ...]) -> str:
    """Write a shape for a reader.

    Args:
        shape: A node's shape.

    Returns:
        The shape as Python writes a tuple, with `?` for a length that is not known: "(?, 2)".
    """
    lengths = [UNKNOWN_TEXT if length is None else str(length) for length in shown(shape)]
    return f"({lengths[0]},)" if len(lengths) == 1 else f"({', '.join(lengths)})"


def caller() -> Origin:
    """Find the line of the user's code that is running.

    Returns:
        The innermost frame that belongs neither to this layer nor to NumPy:
        `np.sum(x)` reaches the layer through NumPy's own Python.
    """
    frame = sys._getframe(1)
    while frame.f_back is not None and is_internal(frame.f_code.co_filename):
        frame = frame.f_back
    registry = frame.f_globals.setdefault("__warningregistry__", {})
    return Origin(frame.f_code.co_filename, frame.f_lineno, frame.f_globals.get("__name__", "<unknown>"), registry)


def is_internal(filename: str) -> bool:
    """Say whether a file is the layer's own code or NumPy's.

    Args:
        filename: A code object's file name.

    Returns:
        True for a module directly in this directory and for any file of NumPy.
    """
    path = os.path.abspath(filename)
    return os.path.dirname(path) == _LAYER_DIRECTORY or path.startswith(_NUMPY_DIRECTORY)


def pending_count() -> int:
    """Return how many nodes are waiting to be computed."""
    return len(_pending)


def pending_nodes() -> list[Node]:
    """Return the nodes waiting to be computed, the newest first.

    Returns:
        A list in descending number, so that `pop()` yields them in the order
        they were written.
    """
    with LOCK:
        return sorted((node for node in _pending if node.pending), key=lambda node: node.number, reverse=True)


def settle(node: Node) -> None:
    """Take a node off the pending list and drop its operands.

    Args:
        node: A node that has just been computed or has failed.
    """
    node.operation, node.args, node.kwargs = None, (), {}
    _pending.discard(node)
