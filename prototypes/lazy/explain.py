"""The text of a pending graph: one operation per line, with the line of the program that wrote it."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .graph import Node

# Where this package keeps its templates.
TEMPLATES = Path(__file__).resolve().parent / "templates"
# The template of `explain()`.
EXPLAIN_TEMPLATE = "explain.txt.j2"


@dataclass(frozen=True)
class Row:
    """One line of `explain()`."""

    # How deep the node is below the array being explained: 0 for the array itself.
    depth: int
    # The operation's name, or "array" for something that already has values.
    name: str
    # Element type of the node's result.
    dtype: str
    # Shape of the node's result, as Python prints a tuple.
    shape: str
    # File name and line of the user's code that wrote the operation; empty for a leaf.
    where: str
    # True when this node was printed higher up, so its operands are not printed again.
    repeated: bool


def render(root: Node) -> str:
    """Return the pending graph of a node as text.

    Args:
        root: The node to explain.

    Returns:
        One line per node, operands indented under their consumer.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                              trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)
    return environment.get_template(EXPLAIN_TEMPLATE).render(rows=rows(root)).strip("\n")


def rows(root: Node) -> list[Row]:
    """Walk a node's graph in the order `explain()` prints it.

    Args:
        root: The node to explain.

    Returns:
        A row per node, consumer before operands. A pending node reached a
        second time is one row marked as repeated: a loop makes a graph in
        which each step uses the step before twice, and printing it as a tree
        would double with every step. Walked with a stack, for the same graphs.
    """
    result: list[Row] = []
    seen: set[int] = set()
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        repeated = node.pending and node.number in seen
        result.append(_row(node, depth, repeated))
        if node.pending and not repeated:
            seen.add(node.number)
            operands = [arg for arg in node.args if isinstance(arg, Node)]
            stack.extend((operand, depth + 1) for operand in reversed(operands))
    return result


def _row(node: Node, depth: int, repeated: bool) -> Row:
    """Describe one node.

    Args:
        node: The node.
        depth: Its depth below the array being explained.
        repeated: Whether it was printed higher up.

    Returns:
        The row.
    """
    where = "" if node.origin is None else f"{os.path.basename(node.origin.filename)}:{node.origin.lineno}"
    return Row(depth, node.name if node.failure is None else f"{node.name} (failed)", str(node.dtype),
               str(node.shape), where, repeated)
