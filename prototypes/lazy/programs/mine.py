"""Mines the corpus for the shapes of expression worth a fused kernel.

    uv run --group apps python -m prototypes.lazy.programs.mine --size large

Each program of `corpus.py` is run with the lazy stand-in, computed one node
at a time as the first build computes it. Before every computation the
expressions among its nodes are read off (`fuse.find`), and every call is
timed. So each expression has a measured weight: the time NumPy spent on the
operations in it. The shapes that can be cut out of the expressions are
ranked by that weight, and a library of shapes is chosen one at a time, each
the shape that takes the most passes over memory out of what is left.

Mining says which shapes occur and what they cost. It does not say that a
shape may be fused: that is `fuse.py`'s rule about which operations a
kernel may contain, and the judge's check of each kernel against NumPy.

Timings are only meaningful on a machine that is doing nothing else.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from prototypes.lazy import evaluate, fuse, get_lazy, graph, rules
from prototypes.lazy.array import LazyArray
from prototypes.lazy.graph import Node
from prototypes.lazy.programs import corpus

log = logging.getLogger(__name__)

# Where the layer keeps its templates, and the template of this tool's report.
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
REPORT_TEMPLATE = "mine.txt.j2"
# Nanoseconds in a millisecond.
NS_PER_MS = 1e6
# The classes a node's time is counted under, in the order a table lists them.
IN_EXPRESSIONS, SIMPLE, OTHER_ELEMENTWISE = "in expressions", "simple, alone", "other element by element"
CLASSES = (IN_EXPRESSIONS, SIMPLE, OTHER_ELEMENTWISE, rules.MATRIX_PRODUCT, rules.REDUCTION, rules.WRITE,
           rules.INDEXING, "the rest")
# Kinds of node that have a class of their own.
OWN_CLASS = frozenset({rules.MATRIX_PRODUCT, rules.REDUCTION, rules.WRITE, rules.INDEXING})
# How many rows the tables of shapes, of whole expressions and of operations outside them have.
SHAPE_ROWS, EXPRESSION_ROWS, OUTSIDE_ROWS = 24, 12, 12
# The shapes considered when the library is chosen: the heaviest this many.
CANDIDATES = 400
# The longest name of an expression a table prints in full.
NAME_WIDTH = 96

app = typer.Typer(add_completion=False)


@dataclass
class _Mined:
    """One expression as the miner keeps it: its shape, without the nodes."""

    # The expression's name, every operation kept.
    name: str
    # How many operations it has.
    operations: int
    # Its terms, the top first: for each, the places in this list of the terms it reads.
    below: list[list[int]]
    # For each term, the shapes that can be cut with it at their top: (name, the places of the terms the shape reads).
    shapes: list[list[tuple[str, list[int]]]]
    # Times it was computed, the elements each time added up, and the nanoseconds NumPy took for its operations.
    count: int = 0
    elements: int = 0
    ns: int = 0


@dataclass
class ProgramMine:
    """What mining one program found."""

    # The program.
    name: str
    # Operations deferred, computations made, and the nodes of the largest computation.
    deferred: int = 0
    graphs: int = 0
    largest: int = 0
    # Nanoseconds spent computing nodes, by class of CLASSES.
    ns: dict[str, int] = field(default_factory=dict)
    # Nanoseconds by "name dtype" for element-by-element nodes outside every expression.
    outside: dict[str, int] = field(default_factory=dict)
    # The distinct expressions, by name.
    expressions: dict[str, _Mined] = field(default_factory=dict)
    # The largest expression's number of operations.
    most_operations: int = 0

    @property
    def total_ns(self) -> int:
        """Nanoseconds spent computing nodes."""
        return sum(self.ns.values())

    def share(self, name: str) -> float:
        """Return one class's share of the time spent computing, in percent."""
        return 100.0 * self.ns.get(name, 0) / self.total_ns if self.total_ns else 0.0


class _Miner:
    """Watches one program's computations: the planner and the meter of `evaluate` for the length of its run."""

    def __init__(self, name: str) -> None:
        """Start on one program.

        Args:
            name: The program's name.
        """
        self.mine = ProgramMine(name)
        # The expression each node about to be computed is in, by the node's number.
        self._home: dict[int, _Mined] = {}

    def plan(self, todo: list[Node]) -> None:
        """Read the expressions off one computation's nodes, and plan nothing: every node is computed by its own call.

        Args:
            todo: The pending nodes about to be computed.
        """
        mine = self.mine
        mine.graphs += 1
        mine.largest = max(mine.largest, len(todo))
        self._home = {}
        expressions, _ = fuse.find(todo)
        for expression in expressions:
            name = fuse.canonical(expression)
            mined = mine.expressions.get(name)
            if mined is None:
                mined = mine.expressions[name] = _detached(name, expression.term)
                mine.most_operations = max(mine.most_operations, mined.operations)
            mined.count += 1
            mined.elements += expression.elements
            for term in fuse.terms_of(expression.term):
                self._home[term.node.number] = mined
            self._home[expression.root.number] = mined

    def measure(self, node: Node, ns: int) -> None:
        """Count the time of one call under its class.

        Args:
            node: The node, computed.
            ns: Nanoseconds its call took.
        """
        mine = self.mine
        home = self._home.get(node.number)
        if home is not None:
            home.ns += ns
            label = IN_EXPRESSIONS
        elif node.kind == rules.ELEMENTWISE:
            simple = node.dtype == fuse.FLOAT64 and node.name in _SIMPLE_NAMES
            label = SIMPLE if simple else OTHER_ELEMENTWISE
            if not simple:
                key = f"{node.name} {node.dtype}"
                mine.outside[key] = mine.outside.get(key, 0) + ns
        else:
            label = node.kind if node.kind in OWN_CLASS else CLASSES[-1]
        mine.ns[label] = mine.ns.get(label, 0) + ns


# The names of the nodes whose operation a fused kernel can make.
_SIMPLE_NAMES = frozenset(ufunc.__name__ for table in (fuse.ARITHMETIC, fuse.UNARY, fuse.COMPARISONS, fuse.SCALED)
                          for ufunc in table)


def _detached(name: str, top: fuse.Term) -> _Mined:
    """Write an expression down as places in a list, so that keeping it keeps no node alive.

    Args:
        name: The expression's name.
        top: Its top term.

    Returns:
        The expression, with the shapes that can be cut at each of its terms.
    """
    terms = list(fuse.terms_of(top))
    place = {id(term): index for index, term in enumerate(terms)}
    below = [[place[id(operand)] for operand in term.operands if type(operand) is fuse.Term] for term in terms]
    shapes = [[(shape.name, [place[id(vector)] for vector in shape.vectors if type(vector) is fuse.Term])
               for shape in fuse.shapes_at(term)] for term in terms]
    return _Mined(name, len(terms), below, shapes)


def _passes(mined: _Mined, library: set[str]) -> int:
    """Count the fewest passes over memory that compute an expression with a library of shapes.

    Args:
        mined: The expression.
        library: The names of the shapes a kernel exists for.

    Returns:
        The number of kernel calls and NumPy calls together.
    """
    best = [0] * len(mined.below)
    # A term reads only terms after it in the list, so the list is walked from its end.
    for index in range(len(best) - 1, -1, -1):
        cost = 1 + sum(best[lower] for lower in mined.below[index])
        for name, reads in mined.shapes[index]:
            if name in library:
                cost = min(cost, 1 + sum(best[lower] for lower in reads))
        best[index] = cost
    return best[0]


@dataclass(frozen=True)
class Chosen:
    """One shape of the chosen library."""

    # The shape's name.
    name: str
    # Its operations.
    operations: int
    # The share of all time spent in expressions that the library saves once this shape is in it, in percent.
    saved: float
    # The programs it occurs in.
    programs: tuple[str, ...]


def choose(mines: list[ProgramMine], size: int) -> list[Chosen]:
    """Choose a library of shapes one at a time, each the one that saves the most time with those before it.

    The time an expression of k operations takes is taken to be k passes of
    equal cost, and a tiling of p passes to save (k - p) of them. Each
    program counts by its measured time, so the corpus is weighed by where
    NumPy spent it.

    Args:
        mines: What mining each program found.
        size: How many shapes to choose.

    Returns:
        The shapes in the order chosen.
    """
    expressions = [(mine.name, mined) for mine in mines for mined in mine.expressions.values()]
    total = sum(mined.ns for _, mined in expressions) or 1
    weight: dict[str, float] = {}
    where: dict[str, set[str]] = {}
    for program, mined in expressions:
        for shapes in mined.shapes:
            for name, _ in shapes:
                weight[name] = weight.get(name, 0.0) + mined.ns / mined.operations
                where.setdefault(name, set()).add(program)
    candidates = sorted(weight, key=weight.get, reverse=True)[:CANDIDATES]
    library: set[str] = set()
    chosen: list[Chosen] = []
    saved_now = 0.0
    for _ in range(size):
        best_name, best_saved = None, saved_now
        for name in candidates:
            if name in library:
                continue
            trial = library | {name}
            saved = sum(mined.ns * (mined.operations - _passes(mined, trial)) / mined.operations
                        for _, mined in expressions if any(name == shape for shapes in mined.shapes for shape, _ in shapes))
            saved += sum(mined.ns * (mined.operations - _passes(mined, library)) / mined.operations
                         for _, mined in expressions if not any(name == shape for shapes in mined.shapes for shape, _ in shapes))
            if saved > best_saved:
                best_name, best_saved = name, saved
        if best_name is None:
            break
        library.add(best_name)
        saved_now = best_saved
        chosen.append(Chosen(best_name, best_name.count("("), 100.0 * saved_now / total, tuple(sorted(where[best_name]))))
    return chosen


def _values(result: Any) -> Iterator[Any]:
    """List the arrays and numbers in a program's result, whatever tuples and lists they are in.

    Args:
        result: What a program returned.

    Yields:
        Each array or number.
    """
    if isinstance(result, (tuple, list)):
        for item in result:
            yield from _values(item)
    else:
        yield result


def mine_program(program: corpus.Program, size: str) -> ProgramMine:
    """Run one program lazily, node by node, and collect its expressions and its times.

    Args:
        program: The program.
        size: `corpus.SMALL` or `corpus.LARGE`.

    Returns:
        What was found.
    """
    inputs = program.make(size)
    miner = _Miner(program.name)
    before = graph.COUNTS.deferred
    planner, meter = evaluate.planner, evaluate.meter
    evaluate.planner, evaluate.meter = miner.plan, miner.measure
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for value in _values(program.run(get_lazy(), inputs)):
                if isinstance(value, LazyArray):
                    value.gather()
    finally:
        evaluate.planner, evaluate.meter = planner, meter
    miner.mine.deferred = graph.COUNTS.deferred - before
    return miner.mine


def render(mines: list[ProgramMine], chosen: list[Chosen], size: str) -> str:
    """Write the report as text.

    Args:
        mines: What mining each program found.
        chosen: The library chosen from them.
        size: The size the programs were run at.

    Returns:
        The tables.
    """
    whole = sorted(((mine.name, mined) for mine in mines for mined in mine.expressions.values()),
                   key=lambda item: item[1].ns, reverse=True)[:EXPRESSION_ROWS]
    outside: dict[str, int] = {}
    for mine in mines:
        for key, ns in mine.outside.items():
            outside[key] = outside.get(key, 0) + ns
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                              lstrip_blocks=True)
    return environment.get_template(REPORT_TEMPLATE).render(
        mines=mines, chosen=chosen, size=size, classes=CLASSES, whole=whole, ns_per_ms=NS_PER_MS, name_width=NAME_WIDTH,
        outside=sorted(outside.items(), key=lambda item: item[1], reverse=True)[:OUTSIDE_ROWS],
        expression_ns=sum(mine.ns.get(IN_EXPRESSIONS, 0) for mine in mines)).strip("\n")


@app.command()
def main(size: str = typer.Option(corpus.LARGE, help="The size the programs are run at: small or large."),
         program: list[str] = typer.Option(None, help="A program to mine; may be given several times. Default: all."),
         shapes: int = typer.Option(SHAPE_ROWS, help="How many shapes the library is chosen to.")) -> None:
    """Mine the corpus and print what was found.

    Args:
        size: The size the programs are run at.
        program: The names of the programs to mine, or none for all of them.
        shapes: How many shapes the library is chosen to.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asked = corpus.by_name(list(program) if program else None)
    for each in asked:
        if not each.available:
            log.info("skipped %s: it needs %s (uv run --group apps ...)", each.name, ", ".join(each.requires))
    mines = [mine_program(each, size) for each in asked if each.available]
    log.info("%s", render(mines, choose(mines, shapes), size))


if __name__ == "__main__":
    app()
