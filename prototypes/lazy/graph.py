"""The graph: one node per deferred operation, and the list of nodes not computed yet.

A node is either pending (it has an operation and no value), computed (it has
a value and has dropped its operands) or failed (computing it raised). A leaf
is a node born computed. Nodes refer to their operands; nothing refers to a
node except its consumers and the `LazyArray` a user holds, so an
intermediate nobody holds disappears when its consumer has been computed.
"""
from __future__ import annotations

import itertools
import os
import sys
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

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


class NotGathered(TypeError):
    """Raised when something needs the values of a lazy array and `gather()` was not called.

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
    # Shape of the result, known when the node is made.
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
    # Calls made at once, by the name NumPy gives the function.
    at_once: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        """Start counting from nothing."""
        self.deferred = 0
        self.at_once = {}


# The one set of counters of the process.
COUNTS = Counts()

# Every node that has no value yet, held weakly: pending work that the program
# drops without gathering is forgotten, never computed.
_pending: weakref.WeakSet[Node] = weakref.WeakSet()


def leaf(value: Any) -> Node:
    """Make a node for an array or NumPy scalar that already exists.

    Args:
        value: An `ndarray` or a NumPy scalar. It is held by reference, not copied.

    Returns:
        A computed node with that value.
    """
    return Node(number=next(_numbers), name="array", operation=None, args=(), kwargs={},
                shape=value.shape, dtype=value.dtype, value=value)


def deferred(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict,
             shape: tuple[int, ...], dtype: np.dtype) -> Node:
    """Make a pending node and add it to the pending list.

    Args:
        operation: The NumPy callable to make later.
        name: What to call it in `explain()` and in messages.
        args: Its positional arguments, with `Node`s for operands in the graph.
        kwargs: Its keyword arguments.
        shape: Shape of the result.
        dtype: Element type of the result.

    Returns:
        The node. It remembers the caller's line and the `np.errstate` in effect.
    """
    node = Node(number=next(_numbers), name=name, operation=operation, args=args, kwargs=kwargs,
                shape=shape, dtype=dtype, origin=caller(), errstate=np.geterr())
    with LOCK:
        _pending.add(node)
    COUNTS.deferred += 1
    return node


def caller() -> Origin:
    """Find the line of the user's code that is running.

    Returns:
        The innermost frame that belongs neither to this layer nor to NumPy:
        `np.sum(x)` reaches the layer through NumPy's own Python.
    """
    frame = sys._getframe(1)
    while frame.f_back is not None and _is_internal(frame.f_code.co_filename):
        frame = frame.f_back
    registry = frame.f_globals.setdefault("__warningregistry__", {})
    return Origin(frame.f_code.co_filename, frame.f_lineno, frame.f_globals.get("__name__", "<unknown>"), registry)


def _is_internal(filename: str) -> bool:
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
