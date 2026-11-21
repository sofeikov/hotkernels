"""Computing nodes: one at a time, each with the NumPy call it stands for.

This is the whole of the first build's executor. It makes exactly the calls
the eager program would have made, on the same operands, in the order they
were written, so the results are NumPy's by construction. Anything that
computes a graph differently later replaces `_call` and is held to the same
results.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import graph
from .graph import Node, Trigger


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
        progress = _begin(todo)
        try:
            while todo:
                # pop() drops the list's reference, so an intermediate is freed as
                # soon as its last consumer has been computed.
                _step(todo.pop(), progress)
        finally:
            _end(progress, roots, trigger)
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
        progress = _begin(todo)
        try:
            while todo:
                node = todo.pop()
                if not node.pending:
                    continue
                inherited = _inherited_failure(node)
                if inherited is None:
                    _step(node, progress)
                else:
                    _fail(node, inherited)
                    progress.numbers.append(node.number)
                    progress.failed.add(node.number)
        finally:
            _end(progress, [], trigger)


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
        _compute_one(node)
    except Exception:
        progress.failed.add(node.number)
        raise


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
