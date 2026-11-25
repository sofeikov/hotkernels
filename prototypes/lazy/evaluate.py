"""Computing nodes: one at a time, each with the NumPy call it stands for.

Left alone, this makes exactly the calls the eager program would have made,
on the same operands, in the order they were written, so the results are
NumPy's by construction.

Two things can be put in from outside, and both are empty unless something
sets them. `planner` is asked, before a computation starts, which of its
nodes one kernel call can compute together: `fuse.py` sets it, and a
computation it plans is held to the same results. `meter` is told how long
each call took, for the tools that measure a program.
"""
from __future__ import annotations

import time
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from . import graph
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
        todo = _needed(roots)
        started = CLOCK.enter(len(todo))
        progress = _begin(todo)
        try:
            plan = planner(todo) if planner is not None and todo else None
            while todo:
                # pop() drops the list's reference, so an intermediate is freed as
                # soon as its last consumer has been computed.
                node = todo.pop()
                if plan is None or not _planned(plan, node, progress):
                    _step(node, progress)
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
        started = CLOCK.enter(len(todo))
        progress = _begin(todo)
        node = None
        try:
            plan = planner(todo) if planner is not None and todo else None
            # The nodes a plan computes inside a group have no value afterwards. Nothing should hold one, or the
            # plan would not have put it inside; if something does, it is computed below, since after this
            # function nothing may be pending.
            skipped = [weakref.ref(node) for node in todo if node.number in plan.inner] if plan is not None else []
            node = None
            while todo:
                node = todo.pop()
                if not node.pending:
                    continue
                inherited = _inherited_failure(node)
                if inherited is not None:
                    _fail(node, inherited)
                    progress.numbers.append(node.number)
                    progress.failed.add(node.number)
                elif plan is None or not _planned(plan, node, progress):
                    _step(node, progress)
            node = None  # The loop's last node must not be kept alive by this name while the skipped ones are looked at.
            for reference in reversed(skipped):
                node = reference()
                if node is not None and node.pending:
                    _compute_one(node)
        finally:
            _end(progress, [], trigger)
            CLOCK.leave(started)


def _planned(plan: Plan, node: Node, progress: _Progress) -> bool:
    """Take one node of a planned computation the plan's way, if the plan has one for it.

    Args:
        plan: The computation's plan.
        node: The next pending node, in the order written.
        progress: The computation's record, which gains the node's number if it is taken here.

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
            _measured(member)
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


def _step(node: Node, progress: _Progress) -> None:
    """Compute one node of a computation and note it.

    Args:
        node: A pending node whose operands are computed or failed.
        progress: The computation's record, which gains the node's number.

    Raises:
        Exception: What computing the node raised, or the failure of an operand.
    """
    progress.numbers.append(node.number)
    try:
        _measured(node)
    except Exception:
        progress.failed.add(node.number)
        raise


def _measured(node: Node) -> None:
    """Compute one node, telling the meter what it took if there is one.

    Args:
        node: A pending node whose operands are computed or failed.

    Raises:
        Exception: What computing the node raised, or the failure of an operand.
    """
    if meter is None:
        _compute_one(node)
        return
    started = time.perf_counter_ns()
    _compute_one(node)
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
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.pending and node.number not in found:
            found[node.number] = node
            stack.extend(arg for arg in node.args if isinstance(arg, Node))
    return [found[number] for number in sorted(found, reverse=True)]


def _compute_one(node: Node) -> None:
    """Compute one node whose operands are all computed or failed.

    Args:
        node: A pending node.

    Raises:
        Exception: The failure of an operand, or what the NumPy call raised,
            with a note naming the line that wrote the operation.
    """
    inherited = _inherited_failure(node)
    if inherited is not None:
        _fail(node, inherited)
        raise inherited
    try:
        value, caught = _call(node)
    except Exception as error:
        error.add_note(f"in the lazy operation '{node.name}' written at {node.origin.filename}:{node.origin.lineno}")
        _fail(node, error)
        raise
    node.value = value
    if not graph.is_known(node.shape):
        node.shape = np.shape(value)  # The length a mask selected is known now.
    graph.settle(node)
    for warning in caught:
        _reissue(warning, node.origin)


def _call(node: Node) -> tuple[Any, list[warnings.WarningMessage]]:
    """Make a node's NumPy call on its operands' values.

    Args:
        node: A pending node whose operands are computed.

    Returns:
        (the result, the warnings NumPy issued while computing it). The call
        runs under the `np.errstate` the operation was written under, and its
        warnings are recorded instead of shown, since they would name a line
        of this file. `catch_warnings` changes the process's filters; that is
        safe here only because every evaluation holds the layer's lock.
    """
    args = tuple(arg.value if isinstance(arg, Node) else arg for arg in node.args)
    with warnings.catch_warnings(record=True) as caught, np.errstate(**node.errstate):
        warnings.simplefilter("always")
        value = node.operation(*args, **node.kwargs)
    return value, caught


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
