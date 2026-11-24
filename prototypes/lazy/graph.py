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
from types import CodeType, FrameType
from typing import Any, Callable, Protocol

import numpy as np



class _NoRecord:
    """Stands where NumPy's record of the error settings stands, on a NumPy that keeps none this layer knows of."""

    @staticmethod
    def get() -> object:
        """Return an object no earlier call returned, so that no remembered settings are taken for the ones in effect."""
        return object()


try:
    # Where NumPy 2 keeps the error settings in effect: a context variable whose
    # value is replaced, never changed, by `np.seterr` and `np.errstate`. It is
    # private to NumPy; without it `np.geterr()` is asked for every node.
    from numpy._core.umath import _extobj_contextvar as _error_settings
except ImportError:  # A NumPy that keeps them elsewhere.
    _error_settings = _NoRecord

# A frame in one of these directories is the layer's own code or NumPy's, never
# the line a user wrote. The layer's directory is matched exactly, so that the
# example programs in its `programs/` subdirectory count as user code.
_LAYER_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
_NUMPY_DIRECTORY = os.path.dirname(os.path.abspath(np.__file__)) + os.sep

# Pending nodes above which everything pending is computed before another is
# added, so a long loop that never gathers cannot hold an unbounded graph.
PENDING_LIMIT = 100_000

# Guards every evaluation. Re-entrant, because computing everything pending
# computes nodes one by one under the same lock. Making a node does not take
# it: a node joins the pending set in one step, and the set is read as a whole.
LOCK = threading.RLock()

# Lines of the user's code remembered at most, before the memory is emptied
# and starts again. A program has as many as it has places that write an
# operation; the bound is for one that compiles code as it runs.
ORIGIN_MEMORY_LIMIT = 4096
# Shapes with their element types remembered at most, before that memory and
# the answers of `rules.plan` that were found through it are emptied.
SPEC_MEMORY_LIMIT = 8192
# The name a module's dictionary keeps Python's record of shown warnings under.
_REGISTRY_NAME = "__warningregistry__"
# What a line is said to be in when its dictionary of globals has no `__name__`.
_UNKNOWN_MODULE = "<unknown>"

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
    # "matmul", "reduction", "view" and the others there. For a leaf, AT_ONCE when it is the
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
    # A number that stands for the shape and the element type together, as
    # `spec` gives it: two nodes with one number have the same shape and the
    # same `dtype` object. 0 while it has not been asked for, and for a shape
    # with a length that is not known. It is what `rules.plan` remembers its
    # answers by; nothing else reads it.
    spec: int = 0

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
# drops without gathering is forgotten, never computed. A set of weak
# references, each of which takes itself out of the set when its node dies.
# That is what `weakref.WeakSet` is; this one is kept by hand because both
# steps, adding a reference and dropping a dead one, are then single calls
# into C, and a node is made and dropped once per operation of the program.
_pending: set[weakref.ref] = set()
# The same set under a name other modules may read the length of, which is
# what `pending_count()` returns: the check against PENDING_LIMIT is made for
# every node, and this spares it a call.
NOT_COMPUTED = _pending
# Called with the reference of a node that has died.
_forget = _pending.discard

# Whether each file seen on the stack is the layer's own code or NumPy's, by
# the name its code objects carry. A name that is not a path (`<stdin>`) is
# judged against the working directory the first time it is seen.
_internal_files: dict[str, bool] = {}
# The lines of the user's code found so far, by the identity of the code
# object and the offset of the instruction running in it. An entry holds the
# code object, so that the identity stays its own; the line; and the line's
# registry of warnings and module name again, which are compared on every
# use. `_core.c` reads an entry in this order.
_origins: dict[tuple[int, int], tuple[CodeType, "Origin", dict, str]] = {}
# The error settings last seen and the dictionary `np.geterr()` gave for them.
_last_errstate: tuple[Any, dict] = (None, {})
# The number of each shape and element type met so far, by the shape and the
# identity of the `dtype` object. An entry holds the `dtype`, so that the
# identity stays its own.
_specs: dict[tuple[tuple[int, ...], int], tuple[int, np.dtype]] = {}
# Hands out those numbers, from 1. One is never given twice, so a number a
# node still carries after the memory was emptied can match no later entry.
_spec_numbers = itertools.count(1)
# Called when the memory of shapes is emptied: `rules` empties its own, whose
# entries were found by the numbers that are now forgotten.
spec_memory_emptied: list[Callable[[], None]] = []


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
    node = Node(next(_numbers), "array", None, (), {}, value.shape, value.dtype, None, None, value, None, kind, strict,
                exposed)
    if observer is not None:
        observer.made(node, source)
    return node


def deferred(operation: Callable[..., Any], name: str, args: tuple, kwargs: dict,
             shape: tuple[int, ...], dtype: np.dtype, kind: str, strict: bool = False,
             origin: "Origin | None" = None, spec: int = 0, frame: FrameType | None = None) -> Node:
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
        spec: The number of the shape and element type, when the caller has
            it already; 0 leaves it to be worked out if it is ever asked for.
        frame: A frame to look for the caller's line from: the one that
            called into the layer, when the caller has it at hand. Every
            frame between it and this one is the layer's own, so the line
            found is the same and fewer frames are walked. None to look from here.

    Returns:
        The node. It remembers the line and the `np.errstate` in effect.
    """
    remembered = _last_errstate
    node = Node(next(_numbers), name, operation, args, kwargs, shape, dtype,
                origin or caller(frame or sys._getframe(1)),
                remembered[1] if remembered[0] is _error_settings.get() else errstate(),
                None, None, kind, strict, False, spec)
    _pending.add(weakref.ref(node, _forget))
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
    for length in shape:
        if length >= UNKNOWN_FLOOR:
            return False
    return True


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


def errstate() -> dict:
    """Return the `np.errstate` in effect, as `np.geterr()` gives it.

    Returns:
        A dictionary that is shared by every node written under the same
        settings and must not be changed. NumPy replaces its record of the
        settings whenever they change, so the dictionary made for one record
        is right for as long as that record is the one in effect.
    """
    global _last_errstate
    settings = _error_settings.get()
    seen, found = _last_errstate
    if seen is not settings:
        found = np.geterr()
        _last_errstate = (settings, found)
    return found


def spec(node: Node) -> int:
    """Return the number that stands for a node's shape and element type, working it out the first time.

    Args:
        node: Any node.

    Returns:
        The same number for every node of that shape whose element type is
        that `dtype` object, and for no other. 0 when a length of the shape
        is not known: such a node has no number until it is computed.
    """
    found = node.spec
    if not found:
        found = node.spec = spec_of(node.shape, node.dtype)
    return found


def spec_of(shape: tuple[int, ...], dtype: np.dtype) -> int:
    """Return the number that stands for a shape and an element type.

    Args:
        shape: A shape.
        dtype: An element type. It is told from others by which object it
            is, not by what it equals: two `dtype` objects that compare
            equal can still differ, in metadata for one, and NumPy's answers
            may carry the difference.

    Returns:
        The number, or 0 when a length of the shape is not known.
    """
    key = (shape, id(dtype))
    found = _specs.get(key)
    if found is not None:
        return found[0]
    if not is_known(shape):
        return 0
    if len(_specs) >= SPEC_MEMORY_LIMIT:
        _specs.clear()
        for emptied in spec_memory_emptied:
            emptied()
    number = next(_spec_numbers)
    _specs[key] = (number, dtype)
    return number


def caller(frame: FrameType | None = None) -> Origin:
    """Find the line of the user's code that is running.

    Args:
        frame: Where to start looking; None for the frame that called this function.

    Returns:
        The innermost frame, from there outwards, that belongs neither to
        this layer nor to NumPy: `np.sum(x)` reaches the layer through
        NumPy's own Python. A line that was found before is returned as the
        same object.
    """
    if frame is None:
        frame = sys._getframe(1)
    # Only a frame of the user's code is ever remembered, so one that is found here is the answer
    # without a look at its file: the case of an operator, which the user's line calls directly.
    origin = _remembered(frame)
    if origin is not None:
        return origin
    walked = frame
    while True:
        filename = walked.f_code.co_filename
        internal = _internal_files.get(filename)
        if internal is None:
            internal = _internal_files[filename] = is_internal(filename)
        if not internal:
            break
        outer = walked.f_back
        if outer is None:
            break
        walked = outer
    if walked is not frame:
        origin = _remembered(walked)
        if origin is not None:
            return origin
    code, names = walked.f_code, walked.f_globals
    origin = Origin(code.co_filename, walked.f_lineno, names.get("__name__", _UNKNOWN_MODULE),
                    names.setdefault(_REGISTRY_NAME, {}))
    if not internal:
        # A stack that is the layer's and NumPy's all the way up ends at a frame that is not the user's: not remembered.
        if len(_origins) >= ORIGIN_MEMORY_LIMIT:
            _origins.clear()
        _origins[(id(code), walked.f_lasti)] = (code, origin, origin.registry, origin.module)
    return origin


def _remembered(frame: FrameType) -> Origin | None:
    """Return the line a frame is at, if it was found before and still describes the frame.

    Args:
        frame: Any frame.

    Returns:
        The `Origin` made for the same instruction of the same code object,
        which fixes the file and the line. The module's name and its
        registry of warnings are looked up again, since a program may
        replace either, and the same code may run under other globals. None
        when nothing is remembered or one of them differs.
    """
    found = _origins.get((id(frame.f_code), frame.f_lasti))
    if found is not None:
        names = frame.f_globals
        if names.get(_REGISTRY_NAME) is found[2] and names.get("__name__", _UNKNOWN_MODULE) == found[3]:
            return found[1]
    return None


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
    # `list()` copies the set in one step, so a node that dies or is made
    # while the copy is read changes nothing here.
    alive = (reference() for reference in list(_pending))
    return sorted((node for node in alive if node is not None and node.pending), key=lambda node: node.number, reverse=True)


def settle(node: Node) -> None:
    """Take a node off the pending list and drop its operands.

    Args:
        node: A node that has just been computed or has failed.
    """
    node.operation, node.args, node.kwargs = None, (), {}
    # A reference to a living node is equal to any other reference to it.
    _pending.discard(weakref.ref(node))
