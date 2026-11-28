"""Computing nodes: one at a time, each with the NumPy call it stands for.

Left alone, this makes exactly the calls the eager program would have made,
on the same operands, in the order they were written, so the results are
NumPy's by construction. Two things are done differently from the letter
of the graph, and give the same values: a write, which is defined as a copy
of the array with the write made into it, is made into the array's own
memory when nothing will read what the array held before (`reusable`); and
a call element by element writes its result into the memory of an operand
that nothing else reads, as NumPy reuses a temporary's (`_lendable`).

Two things can be put in from outside, and both are empty unless something
sets them. `planner` is asked, before a computation starts, which of its
nodes one kernel call can compute together: `fuse.py` sets it, and a
computation it plans is held to the same results. `meter` is told how long
each call took, for the tools that measure a program.
"""
from __future__ import annotations

import sys
import time
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from . import graph, rules
from .graph import Node, Trigger


class Plan(Protocol):
    """What a planner answers with: which nodes of one computation are computed together, and how."""

    # The numbers of the nodes that are computed inside a group and have no value of their own afterwards.
    inner: set[int]
    # The groups, by the number of the node each gives the value of.
    groups: dict[int, Any]

    def run(self, group: Any, node: Node) -> bool:
        """Compute a group's node from what the group reads.

        Args:
            group: A value of `groups`.
            node: The pending node it is under.

        Returns:
            True when the node has its value and is settled. False when the
            group could not be computed that way; the node and the nodes
            inside the group are then computed one by one.
        """

    def members(self, group: Any) -> list[Node]:
        """Return the nodes inside a group, oldest first, without its own node."""


# Asked for a plan of each computation: called with the pending nodes about to be computed, newest first, it
# returns a `Plan`, or None to have every node computed by its own call. None unless `fuse.enable()` has set it.
planner: Callable[[list[Node]], Plan | None] | None = None
# Told what each call took: called with the node, after it has its value, and the nanoseconds. None unless a
# measuring tool has set it.
meter: Callable[[Node, int], None] | None = None
# The C core's way of computing one node whose call is plain, or None: `array.use_core` sets it. It does what
# `_compute_one` does for a node that is no write, has no keyword and reads values that are there, and says
# False, having done nothing, for any other.
compute_plain: Callable[[Node], bool] | None = None
# The C core's loop over the nodes of a computation, or None: `array.use_core` sets it. It takes the nodes in the
# order `compute` and `compute_pending` take them, notes a node inside a group, computes a plain one, and hands
# every other to the Python that takes one node. Not used while a meter times each call.
run_nodes: Callable[..., None] | None = None
# The C core's `_needed`, or None.
needed: Callable[[list[Node]], list[Node]] | None = None


@dataclass
class Clock:
    """How much computing the layer has done, for the tools that measure a program."""

    # Computations that took at least one node, the nodes they took, and the most one of them took.
    computations: int = 0
    nodes: int = 0
    largest: int = 0
    # Nanoseconds spent inside `compute` and `compute_pending`: planning, the calls, and what an observer does.
    ns: int = 0
    # How many of the two functions are running, one inside another. Time is counted by the outermost.
    depth: int = 0

    def reset(self) -> None:
        """Start counting from nothing."""
        self.computations = self.nodes = self.largest = self.ns = 0

    def enter(self, nodes: int) -> int:
        """Note that a computation starts.

        Args:
            nodes: How many nodes it is about to take.

        Returns:
            The time it started, for `leave`.
        """
        if nodes:
            self.computations += 1
            self.nodes += nodes
            self.largest = max(self.largest, nodes)
        self.depth += 1
        return time.perf_counter_ns()

    def leave(self, started: int) -> None:
        """Note that a computation is over.

        Args:
            started: What `enter` returned for it.
        """
        self.depth -= 1
        if not self.depth:
            self.ns += time.perf_counter_ns() - started


# The one clock of the process.
CLOCK = Clock()


@dataclass
class _Progress:
    """What one computation has done so far, for an observer. Numbers only: it must not keep a node alive."""

    # Numbers of the nodes taken so far, in the order they were taken.
    numbers: list[int] = field(default_factory=list)
    # The numbers among them that failed.
    failed: set[int] = field(default_factory=set)


class _Quiet:
    """What the calls of one computation run under: their warnings caught, and NumPy's error settings as each wants.

    A node's call runs under the `np.errstate` its operation was written
    under, and a warning NumPy issues in it would name a line of this file,
    so it is caught and issued again at the line that wrote the operation.
    Both used to be set up around every call, which cost more than a call
    on a small array does. Here they are set up once for the computation:
    the warnings are caught for all of it and looked at after each call,
    and the error settings are changed only when a node wants others than
    the node before it. `catch_warnings` changes the process's filters;
    that is safe here only because every evaluation holds the layer's lock.
    """

    __slots__ = ("catcher", "caught", "found", "outer", "active")

    def __enter__(self) -> "_Quiet":
        """Start catching warnings, and note the error settings in effect."""
        # The warnings caught and not yet given a line, as `catch_warnings` records them.
        self.caught: list[warnings.WarningMessage] = []
        # The warnings waiting to be issued again: (warning, the line that wrote the operation).
        self.found: list[tuple[warnings.WarningMessage, graph.Origin]] = []
        # The error settings before the computation, and the ones in effect now. A node's settings are a
        # dictionary shared by every node written under them, so the same object nearly always.
        self.outer = self.active = np.geterr()
        self._catch()
        return self

    def _catch(self) -> None:
        """Catch every warning from here on."""
        self.catcher = warnings.catch_warnings(record=True)
        self.caught = self.catcher.__enter__()
        warnings.simplefilter("always")

    def under(self, errstate: dict | None) -> None:
        """Put NumPy's error settings as a node wants them.

        Args:
            errstate: The settings the node's operation was written under; None for a node that has none.
        """
        if errstate is not None and errstate != self.active:
            np.seterr(**errstate)
        if errstate is not None:
            self.active = errstate

    def note(self, origin: graph.Origin) -> None:
        """Give the warnings just caught the line of the node whose call issued them.

        Args:
            origin: Where the user wrote that node's operation.
        """
        self.found.extend((warning, origin) for warning in self.caught)
        self.caught.clear()

    def issue(self) -> None:
        """Issue the warnings found so far again, each at its line, and go on catching.

        The user's own filters apply, so this is done outside the catching:
        a filter that makes a warning an error raises here, with the
        computation stopped at the node that warned, as it used to be.

        Raises:
            Warning: A warning the user's filters turn into an error.
        """
        found, self.found = self.found, []
        self.catcher.__exit__(None, None, None)
        try:
            for warning, origin in found:
                _reissue(warning, origin)
        finally:
            self._catch()

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        """Put the filters and the error settings back, and issue what was found.

        Raises:
            Warning: A warning the user's filters turn into an error, unless the computation is already raising.
        """
        self.catcher.__exit__(None, None, None)
        if self.active != self.outer:
            np.seterr(**self.outer)
        found, self.found = self.found, []
        for warning, origin in found:
            try:
                _reissue(warning, origin)
            except Warning:
                if error is None:
                    raise


def compute(roots: Iterable[Node], trigger: Trigger) -> None:
    """Compute some nodes and whatever they need.

    Args:
        roots: The nodes whose values are wanted.
        trigger: What asked for them, for an observer.

    Raises:
        Exception: What computing a needed node raised, now or at an earlier attempt.
    """
    roots = list(roots)
    with graph.LOCK:
        todo = needed(roots) if needed is not None else _needed(roots)
        if not todo:
            # Nothing to compute: the commonest case by far, a value asked for of an array that has it.
            for root in roots:
                if root.failure is not None:
                    raise root.failure
            return
        started = CLOCK.enter(len(todo))
        progress = _begin(todo)
        try:
            plan = planner(todo) if planner is not None else None
            with _Quiet() as quiet:
                if run_nodes is not None and meter is None:
                    run_nodes(todo, None if plan is None else plan.inner, None if plan is None else plan.groups, quiet,
                              progress.numbers, _taker(plan, progress, quiet), _failer(progress), False)
                while todo:
                    # pop() drops the list's reference, so an intermediate is freed as
                    # soon as its last consumer has been computed.
                    node = todo.pop()
                    if plan is None or not _planned(plan, node, progress, quiet):
                        _step(node, progress, quiet)
                    if quiet.found:
                        quiet.issue()
        finally:
            _end(progress, roots, trigger)
            CLOCK.leave(started)
        for root in roots:
            if root.failure is not None:
                raise root.failure


def compute_pending(trigger: Trigger) -> None:
    """Compute everything that is pending, in the order it was written.

    This is what a write does first: afterwards the program is in the state
    the eager program would be in. A node that fails for the first time
    raises here. A node that depends on an earlier failure is marked failed
    without raising: it raises when it is gathered.

    Args:
        trigger: What made the layer compute, for an observer.

    Raises:
        Exception: What computing a node raised.
    """
    with graph.LOCK:
        todo = graph.pending_nodes()
        if not todo:
            return
        started = CLOCK.enter(len(todo))
        progress = _begin(todo)
        node = None
        try:
            plan = planner(todo) if planner is not None else None
            # The nodes a plan computes inside a group have no value afterwards. Nothing should hold one, or the
            # plan would not have put it inside; if something does, it is computed below, since after this
            # function nothing may be pending.
            skipped = [weakref.ref(node) for node in todo if node.number in plan.inner] if plan is not None else []
            node = None
            with _Quiet() as quiet:
                if run_nodes is not None and meter is None:
                    run_nodes(todo, None if plan is None else plan.inner, None if plan is None else plan.groups, quiet,
                              progress.numbers, _pending_taker(plan, progress, quiet), _failer(progress), True)
                while todo:
                    node = todo.pop()
                    if not node.pending:
                        continue
                    inherited = _inherited_failure(node)
                    if inherited is not None:
                        _fail(node, inherited)
                        progress.numbers.append(node.number)
                        progress.failed.add(node.number)
                    elif plan is None or not _planned(plan, node, progress, quiet):
                        _step(node, progress, quiet)
                    if quiet.found:
                        quiet.issue()
                node = None  # The loop's last node must not be kept alive by this name while the skipped ones are looked at.
                for reference in reversed(skipped):
                    node = reference()
                    if node is not None and node.pending:
                        _compute_one(node, quiet)
        finally:
            _end(progress, [], trigger)
            CLOCK.leave(started)


def _taker(plan: Plan | None, progress: _Progress, quiet: _Quiet) -> Callable[[Node], None]:
    """Build what takes one node of `compute`'s loop, for the C core's loop to call for a node it does not take.

    Args:
        plan: The computation's plan, or None.
        progress: The computation's record.
        quiet: What the computation's calls run under.

    Returns:
        A function of the node that does what one turn of the loop does but for the look at warnings found.
    """
    def take(node: Node) -> None:
        """Take one node: the plan's way if it has one, by its own call otherwise."""
        if plan is None or not _planned(plan, node, progress, quiet):
            _step(node, progress, quiet)
    return take


def _pending_taker(plan: Plan | None, progress: _Progress, quiet: _Quiet) -> Callable[[Node], None]:
    """Build what takes one node of `compute_pending`'s loop, for the C core's loop to call.

    Args:
        plan: The computation's plan, or None.
        progress: The computation's record.
        quiet: What the computation's calls run under.

    Returns:
        A function of the node that does what one turn of that loop does but for the look at warnings found.
    """
    def take(node: Node) -> None:
        """Take one node that is still pending: failed with its operand, the plan's way, or by its own call."""
        inherited = _inherited_failure(node)
        if inherited is not None:
            _fail(node, inherited)
            progress.numbers.append(node.number)
            progress.failed.add(node.number)
        elif plan is None or not _planned(plan, node, progress, quiet):
            _step(node, progress, quiet)
    return take


def _failer(progress: _Progress) -> Callable[[Node, BaseException], None]:
    """Build what the C core's loop calls with a node whose plain call raised.

    Args:
        progress: The computation's record.

    Returns:
        A function of the node and the error that does what `_compute_one` and `_step` do with one: the error
        gets a note of the line, the node is failed for good, and the record has its number among the failed.
    """
    def failed(node: Node, error: BaseException) -> None:
        """Note the line on the error and fail the node."""
        error.add_note(f"in the lazy operation '{node.name}' written at {node.origin.filename}:{node.origin.lineno}")
        _fail(node, error)
        progress.failed.add(node.number)
    return failed


def _planned(plan: Plan, node: Node, progress: _Progress, quiet: _Quiet) -> bool:
    """Take one node of a planned computation the plan's way, if the plan has one for it.

    Args:
        plan: The computation's plan.
        node: The next pending node, in the order written.
        progress: The computation's record, which gains the node's number if it is taken here.
        quiet: What the computation's calls run under.

    Returns:
        True when nothing is left to do for the node: it is inside a group,
        whose own node will give what it would have, or it is a group's node
        and has its value. False when the node is to be computed by its own
        call: it is in no group, or its group could not be computed
        together, in which case the nodes inside the group have been
        computed one by one first.

    Raises:
        Exception: What computing a node inside a group raised, one by one.
    """
    number = node.number
    if number in plan.inner:
        progress.numbers.append(number)
        return True
    group = plan.groups.get(number)
    if group is None:
        return False
    started = time.perf_counter_ns() if meter is not None else 0
    if plan.run(group, node):
        progress.numbers.append(number)
        if meter is not None:
            meter(node, time.perf_counter_ns() - started)
        return True
    for member in plan.members(group):
        if member.pending:
            _measured(member, quiet)
    return False


def _begin(todo: list[Node]) -> _Progress:
    """Start one computation, telling the observer what it will take.

    Args:
        todo: The pending nodes about to be computed.

    Returns:
        An empty record of progress.
    """
    if graph.observer is not None and todo:
        graph.observer.computing(todo)
    return _Progress()


def _step(node: Node, progress: _Progress, quiet: _Quiet) -> None:
    """Compute one node of a computation and note it.

    Args:
        node: A pending node whose operands are computed or failed.
        progress: The computation's record, which gains the node's number.
        quiet: What the computation's calls run under.

    Raises:
        Exception: What computing the node raised, or the failure of an operand.
    """
    progress.numbers.append(node.number)
    try:
        if meter is None:
            _compute_one(node, quiet)
        else:
            _measured(node, quiet)
    except Exception:
        progress.failed.add(node.number)
        raise


def _measured(node: Node, quiet: _Quiet) -> None:
    """Compute one node, telling the meter what it took if there is one.

    Args:
        node: A pending node whose operands are computed or failed.
        quiet: What the computation's calls run under.

    Raises:
        Exception: What computing the node raised, or the failure of an operand.
    """
    if meter is None:
        _compute_one(node, quiet)
        return
    started = time.perf_counter_ns()
    _compute_one(node, quiet)
    meter(node, time.perf_counter_ns() - started)


def _end(progress: _Progress, roots: list[Node], trigger: Trigger) -> None:
    """Finish one computation, telling the observer what it took.

    Args:
        progress: The computation's record.
        roots: The nodes whose values were asked for.
        trigger: What made the layer compute.
    """
    if graph.observer is not None and progress.numbers:
        graph.observer.computed(progress.numbers, progress.failed, [root.number for root in roots], trigger)
    graph.release_pending()


def _needed(roots: Iterable[Node]) -> list[Node]:
    """Collect the pending nodes some roots depend on, themselves included.

    Args:
        roots: The nodes whose values are wanted.

    Returns:
        The pending nodes, in descending number, so that `pop()` yields them
        in the order they were written. Found with a stack, not by recursion:
        a loop of a thousand steps makes a graph a thousand nodes deep.
    """
    found: dict[int, Node] = {}
    stack = [root for root in roots if root.operation is not None]
    while stack:
        node = stack.pop()
        number = node.number
        if number not in found:
            found[number] = node
            for arg in node.args:
                if type(arg) is Node and arg.operation is not None:
                    stack.append(arg)
    if len(found) < 2:
        return list(found.values())
    return [found[number] for number in sorted(found, reverse=True)]


def _old_contents(write: Node) -> tuple[int, int]:
    """Count the references to what a write's array held before it: to the node, and to the node's array.

    Args:
        write: A node whose first operand is a computed node with a value.

    Returns:
        (`sys.getrefcount` of the node, the same of its value), this
        function's own references among them.
    """
    old = write.args[0]
    return sys.getrefcount(old), sys.getrefcount(old.value)


def _view_counts(write: Node, memory: np.ndarray) -> list[tuple[int, int]]:
    """Count the references to each operand of a write that is a view of the array written into.

    Args:
        write: A write's node.
        memory: The array the write's first operand holds.

    Returns:
        For each different operand after the first that is a computed node
        whose value is a view made directly of `memory`: (references to the
        node, less one for each place the write has it at; references to
        its value), this function's own among them.
    """
    counts = []
    seen: set[int] = set()
    for arg in write.args[1:]:
        if type(arg) is Node and arg.number not in seen:
            seen.add(arg.number)
            value = arg.value
            if type(value) is np.ndarray and value.base is memory:
                del value
                places = sum(1 for other in write.args if other is arg)
                counts.append((sys.getrefcount(arg) - places, sys.getrefcount(arg.value)))
    return counts


def _view_operands(write: Node, memory: np.ndarray) -> int:
    """Count the operands of a write that are views of the array written into and that the write alone reads.

    `p[:, -1] = p[:, -2]` reads a view of the array it writes. The view's
    array refers to the array it is a view of, which is one of the
    references `_old_contents` counts; it is the write's own.

    Args:
        write: A write's node.
        memory: The array the write's first operand holds.

    Returns:
        How many of the views `_view_counts` lists have no reference to the node or to its value but the write's.
    """
    return sum(1 for nodes, arrays in _view_counts(write, memory) if nodes <= _VIEW_ALONE[0] and arrays <= _VIEW_ALONE[1])


def _calibrated() -> tuple[tuple[int, int], tuple[int, int]]:
    """Measure what the counts above report for objects nothing else refers to.

    Measured, not assumed: an interpreter may count its own references
    differently, and the counts depend on how the functions are written.

    Returns:
        (`_old_contents` of a write whose old contents nothing else refers
        to, `_view_counts` of a view operand that the write alone reads).
    """
    kind = np.dtype(np.float64)
    write = Node(0, "write", None, (Node(0, "array", None, (), {}, (2,), kind, value=np.zeros(2)),), {}, (2,), kind)
    alone = _old_contents(write)
    memory = write.args[0].value
    write.args = (write.args[0], Node(1, "view", None, (), {}, (1,), kind, value=memory[:1]))
    return alone, _view_counts(write, memory)[0]


# What `_old_contents` reports when the write is the only reader, and what `_view_operands` sees for a view
# that the write alone reads.
_VIEW_ALONE: tuple[int, int] = (0, 0)
_ALONE, _VIEW_ALONE = _calibrated()
# Counts of writes made in place and of writes that copied, for the tools that measure a program.
WRITES = {"in_place": 0, "copied": 0}
# The fewest bytes an operand must have for `_lendable` to lend its memory: NumPy's own threshold for reusing a
# temporary (`NPY_MIN_ELIDE_BYTES`, 256 KiB). Below it a new array is memory the allocator has just had back. The
# C core's `LEND_BYTES` is the same.
LEND_BYTES = 256 * 1024
# Calls computed by Python that wrote into an operand's memory (`_lendable`); the core counts its own (`lent_count`).
LENT = {"python": 0}


def _lend_counts(node: Node, place: int) -> tuple[int, int]:
    """Count the references to one operand of a node, and to its value.

    Args:
        node: A pending node.
        place: Where the operand is among the node's operands. It is a computed node.

    Returns:
        (`sys.getrefcount` of the operand, less one for each place the node has it at; the same of its value),
        this function's own references among them.
    """
    operand = node.args[place]
    return sys.getrefcount(operand) - _places(node.args, operand), sys.getrefcount(operand.value)


def _places(args: tuple, operand: Node) -> int:
    """Count the places of one operand among a node's operands.

    Args:
        args: The node's operands.
        operand: One of them.

    Returns:
        How many of them are that object.
    """
    return sum(1 for arg in args if arg is operand)


def _calibrated_lending() -> tuple[int, int]:
    """Measure what `_lend_counts` reports for an operand that nothing but the node reads.

    Returns:
        `_lend_counts` of the only operand of a node that nothing else refers to.
    """
    kind = np.dtype(np.float64)
    node = Node(1, "negative", np.negative, (Node(0, "array", None, (), {}, (2,), kind, value=np.zeros(2)),), {}, (2,), kind)
    return _lend_counts(node, 0)


# What `_lend_counts` reports for an operand that the node alone reads.
_LENT_ALONE = _calibrated_lending()


def _lendable(node: Node) -> Node | None:
    """Return the operand whose memory a node's call may write its result into, as NumPy reuses a temporary's.

    NumPy computes `a * b + c` by writing the sum into the memory of
    `a * b`, a temporary nothing else holds, and frees each temporary as
    soon as it is read, so the next array of that size is pages already in
    use. A graph's intermediates are each held by a node until their last
    reader has run, and were each a new array: computed node by node,
    pvlib's solar position took 1.05x the eager program's time, and takes
    0.95x with this (docs/lazy-codebases-findings.md). The result is the same when it is written into an operand's memory
    with `out=`: an operation element by element reads each element before
    it writes it. That is done when all of this holds:

    - the call is a ufunc of one output, with an operand for each input and
      no keyword, element by element;
    - every array operand lies in C order or has no dimensions, so that
      NumPy's result out of place is in C order, as the operand lent is;
    - the operand lent is a computed node, its value an `ndarray` that owns
      its memory and can be written, of the result's shape and element
      type, of `LEND_BYTES` or more, and not exposed;
    - nothing else reads it: a pending node holds its operands and a lazy
      array its node, so a reference to it beyond this node's is a reader
      or a name the user holds; and nothing but the node refers to its
      value, so no view of it is alive.

    CPython counts references exactly. Where a count is higher than it has
    to be the call makes a new array, which is always right.

    Args:
        node: A pending node whose operands are computed.

    Returns:
        The operand to lend its memory, or None.
    """
    operation, args = node.operation, node.args
    if (type(operation) is not np.ufunc or node.kwargs or operation.nout != 1 or operation.nin != len(args)
            or node.kind != rules.ELEMENTWISE):
        return None
    for arg in args:
        value = arg.value if type(arg) is Node else arg
        if type(value) is np.ndarray and value.ndim and not value.flags.c_contiguous:
            return None
    value = None
    for place in range(len(args)):
        arg = args[place]
        if type(arg) is not Node or arg.operation is not None or arg.failure is not None or arg.exposed:
            continue
        memory = arg.value
        if (type(memory) is not np.ndarray or memory.base is not None or not memory.flags.writeable
                or memory.dtype != node.dtype or memory.shape != node.shape or memory.nbytes < LEND_BYTES):
            continue
        memory = arg = None
        if _lend_counts(node, place) == _LENT_ALONE:
            return args[place]
    return None


def reusable(write: Node, held: int = 0, apart: int = 0) -> bool:
    """Say whether a write may be made into the memory its array already has, instead of into a copy.

    A write is defined as a copy of what the array held with the write
    made into it, so that an operation written before the write still reads
    the old contents whenever it is computed. The copy is not needed when
    nothing will read the old contents again, and then the write is what
    NumPy does: into the memory, in place. That is so when all of this holds:

    - the old contents are computed, in an array that owns its memory and
      that nothing outside the layer can know;
    - no pending node reads them: a pending node refers to its operands,
      and a computed one has let go of them, so a reference to the old
      node beyond the write's own is a reader, or a lazy array that still
      stands for the old contents;
    - no computed view of them is alive: a view's array refers to the
      array it views. The views the write itself reads are its own; NumPy
      takes care of a source that overlaps its destination.

    CPython counts references exactly. Where a count is higher than it has
    to be the write copies, which is always right.

    Args:
        write: A pending node of kind WRITE.
        held: References to the old contents' node that the caller knows of and that read nothing after the
            write: a plan's own record of what it computes together with the write.
        apart: Computed views of the old contents that the caller knows to lie wholly outside the part the
            write changes. Such a view shows the same elements before the write and after it, whoever reads it
            and whenever, so its reference to the array is no reader's.

    Returns:
        True when the write's first operand may be overwritten.
    """
    args = write.args
    old = args[0] if args else None
    if type(old) is not Node or old.operation is not None or old.failure is not None or old.exposed:
        return False
    memory = old.value
    if type(memory) is not np.ndarray or memory.base is not None or not memory.flags.writeable:
        return False
    del old
    nodes, arrays = _old_contents(write)
    # `memory` is one more reference to the array than the calibration had.
    spare = arrays - 1 - _ALONE[1]
    if nodes - held > _ALONE[0] or spare < 0:
        return False
    return spare == 0 or spare == _view_operands(write, memory) + apart


def _compute_one(node: Node, quiet: _Quiet) -> None:
    """Compute one node whose operands are all computed or failed.

    Args:
        node: A pending node.
        quiet: What the computation's calls run under.

    Raises:
        Exception: The failure of an operand, or what the NumPy call raised,
            with a note naming the line that wrote the operation.
    """
    if compute_plain is not None and node.errstate is quiet.active:
        origin = node.origin
        try:
            done = compute_plain(node)
        except Exception as error:
            error.add_note(f"in the lazy operation '{node.name}' written at {origin.filename}:{origin.lineno}")
            _fail(node, error)
            raise
        if done:
            if quiet.caught:
                quiet.note(origin)
            return
    operation, moved, out = node.operation, None, {}
    if node.kind == rules.WRITE:
        in_place = getattr(operation, "in_place", None)
        if in_place is not None and reusable(node):
            operation, moved = in_place, node.args[0]
            WRITES["in_place"] += 1
        else:
            WRITES["copied"] += 1
    else:
        moved = _lendable(node)
        if moved is not None:
            out = {"out": moved.value}
            LENT["python"] += 1
    values = []
    for arg in node.args:
        if type(arg) is Node:
            if arg.failure is not None:
                _fail(node, arg.failure)
                raise arg.failure
            arg = arg.value
        values.append(arg)
    if node.errstate is not quiet.active:
        quiet.under(node.errstate)
    try:
        value = operation(*values, **node.kwargs, **out)
    except Exception as error:
        error.add_note(f"in the lazy operation '{node.name}' written at {node.origin.filename}:{node.origin.lineno}")
        _fail(node, error)
        raise
    if moved is not None:
        # The old node's memory is the new node's now. Nothing reads the old one again; if anything did, it
        # would find no value there rather than the new contents under the old name.
        moved.value = None
    node.value = value
    if not graph.is_known(node.shape):
        node.shape = np.shape(value)  # The length a mask selected is known now.
    origin = node.origin
    graph.settle(node)
    if quiet.caught:
        quiet.note(origin)


def _reissue(warning: warnings.WarningMessage, origin: graph.Origin) -> None:
    """Issue a recorded warning again, at the line that wrote the operation.

    The user's own filters apply, and the module's registry makes Python show
    a warning once per line, as it would have for the eager program.

    Args:
        warning: A warning NumPy issued while a node was computed.
        origin: Where the user wrote that node's operation.
    """
    warnings.warn_explicit(warning.message, warning.category, origin.filename, origin.lineno,
                           module=origin.module, registry=origin.registry)


def _inherited_failure(node: Node) -> BaseException | None:
    """Return the failure of one of a node's operands, if any has failed.

    Args:
        node: A pending node.

    Returns:
        The exception of the first failed operand, or None.
    """
    return next((arg.failure for arg in node.args if isinstance(arg, Node) and arg.failure is not None), None)


def _fail(node: Node, error: BaseException) -> None:
    """Mark a node as failed for good.

    Args:
        node: The node.
        error: What to raise whenever the node is gathered.
    """
    node.failure = error
    graph.settle(node)
