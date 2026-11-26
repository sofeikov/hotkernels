"""Says how far fusing element-by-element operations could take each program, and how far the fused run is from it.

    uv run --group apps --group workloads python -m prototypes.lazy.programs.limit --size large

Fusion removes two things from an operation on arrays: the pass over memory
that reads its operands and writes its result, and the call into NumPy. It
cannot remove the arithmetic itself. So every operation has a part no fusion
touches, and that part can be measured: the same NumPy loop on arrays that
fit the first-level cache, per element, with the cost of the call taken off.

For each program of `corpus.py` this tool:

- times the program eagerly;
- runs it lazily, node by node, with every call timed, and puts each
  operation in a class by what a fused kernel would need to make it;
- measures, for each different operation, what an element costs in the
  cache, and takes that times the elements as the part that has to stay;
- times the fused run.

The **floor** of a program is its eager time less everything that could be
removed from its element-by-element operations and its reductions over
every element: what a fusion of all of them, of any kind, with no cost of
its own, would leave. What lies between the fused run and the floor is what
is left to gain; the classes say which operations it is in.

Timings are only meaningful on a machine that is doing nothing else.
"""
from __future__ import annotations

import gc
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy
import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from prototypes.lazy import array, evaluate, fuse, get_lazy, rules
from prototypes.lazy.graph import Node
from prototypes.lazy.programs import corpus

log = logging.getLogger(__name__)

# Where the layer keeps its templates, and the template of this tool's tables.
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
TABLE_TEMPLATE = "limit.txt.j2"
# Nanoseconds in a millisecond.
NS_PER_MS = 1e6
# The classes an operation of the graph is counted under. The first five are what fusion could reach: the
# operations a kernel makes now; comparisons, `np.where` and logic, which give or read a mask; float64 operations
# that are a call into a mathematics library; the same kinds of operation on other element types; and reductions.
ARITHMETIC, MASKS, LIBRARY, OTHER_TYPES, REDUCTIONS = "arithmetic", "masks", "library", "other types", "reductions"
# The rest fusion of element-by-element operations does not reach: matrix products; writes and selections by
# arrays, which move elements; and everything else the layer defers (views, running totals, sorting, searching).
PRODUCTS, MOVES, REST = "products", "moves", "rest"
REACHABLE = (ARITHMETIC, MASKS, LIBRARY, OTHER_TYPES, REDUCTIONS)
CLASSES = (*REACHABLE, PRODUCTS, MOVES, REST)
# The two lengths at which an operation is timed in the cache. Arrays of both fit the first-level cache of the
# machines this runs on, three at a time; the time for one element is the slope between the two, which leaves
# the cost of the call out.
SHORT, LONG = 512, 4096
# How many times an operation is timed at each length; the quickest counts.
CALIBRATION_RUNS = 200
# The values the made-up operands of a calibration have: away from zero, one and anything a loop treats apart.
LOW, HIGH = 0.5, 1.5
# Seed of the made-up operands.
SEED = 0

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Signature:
    """What decides an operation's time for one element: the call, and the element type of each operand."""

    # The class the operation is counted under.
    label: str
    # The callable the node holds, and the name the node has.
    operation: Any = field(compare=False)
    name: str = ""
    # The node's kind, as `rules` names it.
    kind: str = ""
    # The result's element type.
    dtype: str = ""
    # For each operand: its element type and whether it is as large as the largest (an array the loop walks) or
    # smaller (read as one value); or, for a number written in the program, its type's name and the number.
    operands: tuple = ()


@dataclass
class Spent:
    """What the operations of one signature took in a program."""

    # Nanoseconds NumPy took for them, node by node.
    ns: int = 0
    # How many calls that was, and how many elements they walked: the largest of each call's operands and result.
    calls: int = 0
    elements: int = 0


@dataclass
class Limit:
    """What was measured of one program."""

    # The program.
    name: str
    # Nanoseconds of the eager run, of the lazy run node by node, and of the fused run: the quickest of each.
    eager_ns: int = 0
    lazy_ns: int = 0
    fused_ns: int = 0
    # Nanoseconds NumPy took for the nodes of each class, in the lazy run.
    ns: dict[str, int] = field(default_factory=dict)
    # Of that, what the same loops take on arrays in the cache with the call left out: the part that stays.
    kept: dict[str, int] = field(default_factory=dict)
    # How many operations each class had.
    calls: dict[str, int] = field(default_factory=dict)
    # Nanoseconds the fused run spent inside kernels' programs, and how many nodes they computed.
    kernel_ns: int = 0
    fused_nodes: int = 0

    def removable(self, label: str) -> int:
        """Return what fusion could take off a class at most: its time less what its arithmetic costs in the cache."""
        return max(0, self.ns.get(label, 0) - self.kept.get(label, 0))

    @property
    def reachable_ns(self) -> int:
        """Nanoseconds that could be removed over all the classes fusion could reach."""
        return sum(self.removable(label) for label in REACHABLE)

    @property
    def graph_ns(self) -> int:
        """Nanoseconds NumPy took for every node of the graph."""
        return sum(self.ns.values())

    @property
    def outside_ns(self) -> int:
        """Nanoseconds of the eager run that are no node's: the program's own code, a library's, calls made at once."""
        return max(0, self.eager_ns - self.graph_ns)

    @property
    def floor_ns(self) -> int:
        """Nanoseconds a fusion of every reachable operation, with no cost of its own, would leave."""
        return max(1, self.eager_ns - min(self.reachable_ns, self.eager_ns - 1))

    @property
    def left_ns(self) -> int:
        """Nanoseconds between the fused run and the floor."""
        return self.fused_ns - self.floor_ns


def _operand(arg: Any, largest: int) -> tuple:
    """Write down what one operand contributes to a signature.

    Args:
        arg: A node, an `ndarray`, a NumPy scalar or a number written in the program.
        largest: The elements of the call's largest operand or result.

    Returns:
        (element type, whether the loop walks it) for an array, and (type name, the value) for a number.
    """
    if isinstance(arg, (Node, numpy.ndarray)):
        return (str(arg.dtype), _elements(arg.shape) == largest and largest > 1)
    if isinstance(arg, numpy.generic):
        return (str(arg.dtype), False)
    if isinstance(arg, (bool, int, float, complex)):
        return (type(arg).__name__, arg)
    return ("object", repr(arg))


def _elements(shape: tuple[int, ...]) -> int:
    """Count the elements of a shape, taking a length that is not known for one element."""
    count = 1
    for length in shape:
        count *= length if length < 2 ** 40 else 1
    return count


def classify(node: Node) -> str:
    """Put a pending node in a class.

    Args:
        node: A node about to be computed.

    Returns:
        One of CLASSES.
    """
    if node.kind == rules.REDUCTION:
        return REDUCTIONS
    if node.kind == rules.MATRIX_PRODUCT:
        return PRODUCTS
    if node.kind in (rules.WRITE, rules.INDEXING):
        return MOVES
    if node.kind != rules.ELEMENTWISE:
        return REST
    read = fuse.operation_of(node)
    types = [arg.dtype for arg in node.args if isinstance(arg, (Node, numpy.ndarray, numpy.generic))]
    if node.dtype == fuse.BOOLEAN or any(kind == fuse.BOOLEAN for kind in types) or (read is not None and read[0] == fuse.WHERE):
        return MASKS
    if node.dtype != fuse.FLOAT64 or any(kind != fuse.FLOAT64 for kind in types):
        return OTHER_TYPES
    return ARITHMETIC if read is not None else LIBRARY


class _Watcher:
    """Watches one program's lazy run: the planner and the meter of `evaluate` for its length."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        # The signature and the elements of each node about to be computed, by the node's number.
        self._waiting: dict[int, tuple[Signature, int]] = {}
        # What each signature took.
        self.spent: dict[Signature, Spent] = {}

    def plan(self, todo: list[Node]) -> None:
        """Write down what each node of a computation is, while it still has its operands; plan nothing.

        Args:
            todo: The pending nodes about to be computed.
        """
        for node in todo:
            largest = max([_elements(node.shape)] + [_elements(arg.shape) for arg in node.args
                                                      if isinstance(arg, (Node, numpy.ndarray))])
            signature = Signature(classify(node), node.operation, node.name, node.kind, str(node.dtype),
                                  tuple(_operand(arg, largest) for arg in node.args))
            self._waiting[node.number] = (signature, largest)

    def measure(self, node: Node, ns: int) -> None:
        """Count the time of one call under its signature.

        Args:
            node: The node, computed.
            ns: Nanoseconds its call took.
        """
        found = self._waiting.pop(node.number, None)
        if found is None:
            return
        spent = self.spent.setdefault(found[0], Spent())
        spent.ns += ns
        spent.calls += 1
        spent.elements += found[1]


def _made_up(operand: tuple, length: int, rng: numpy.random.Generator) -> Any:
    """Make an operand for a calibration.

    Args:
        operand: What `_operand` wrote down.
        length: How many elements an array the loop walks is to have.
        rng: Where the values come from.

    Returns:
        An array of that element type with `length` elements, or one element for an operand read as one value;
        or the number itself.
    """
    kind, walked = operand
    if kind in ("bool", "int", "float", "complex"):
        return walked
    dtype = numpy.dtype(kind)
    count = length if walked else 1
    if dtype.kind == "b":
        values = rng.random(count) < 0.5
    elif dtype.kind == "c":
        values = rng.uniform(LOW, HIGH, count) + 1j * rng.uniform(LOW, HIGH, count)
    else:
        values = rng.uniform(LOW, HIGH, count)
    made = values.astype(dtype)
    return made if walked else made.reshape(())


def _quickest(call: Callable[[], Any]) -> int:
    """Time a call CALIBRATION_RUNS times and return the quickest, in nanoseconds."""
    best = None
    for _ in range(CALIBRATION_RUNS):
        started = time.perf_counter_ns()
        call()
        taken = time.perf_counter_ns() - started
        best = taken if best is None or taken < best else best
    return best or 0


def in_cache(signature: Signature) -> float | None:
    """Measure what one element of an operation costs on arrays in the cache, the call left out.

    Args:
        signature: The operation.

    Returns:
        Nanoseconds for one element: the slope of the call's time between SHORT and LONG elements. None for an
        operation that cannot be made on made-up operands, whose whole time is then counted as staying.
    """
    if any(kind == "object" for kind, _ in signature.operands):
        return None
    rng = numpy.random.default_rng(SEED)
    times = []
    for length in (SHORT, LONG):
        operands = [_made_up(operand, length, rng) for operand in signature.operands]
        try:
            with warnings.catch_warnings(), numpy.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                signature.operation(*operands)
                times.append(_quickest(lambda: signature.operation(*operands)))
        except Exception:  # Whatever a made-up operand does to an operation: it is then not calibrated.
            return None
    return max(0.0, (times[1] - times[0]) / (LONG - SHORT))


def _values(result: Any) -> Iterator[Any]:
    """List the arrays and numbers in a program's result, whatever tuples and lists they are in."""
    if isinstance(result, (tuple, list)):
        for item in result:
            yield from _values(item)
    else:
        yield result


def _run(program: corpus.Program, inputs: tuple, np: Any) -> None:
    """Run a program with a module and gather what it returns, in one walk of the graph."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        array.gather(*_values(program.run(np, inputs)))


def _timed(program: corpus.Program, inputs: tuple, np: Any, runs: int) -> int:
    """Time a program several times with a module and return the quickest run, in nanoseconds."""
    best = None
    for _ in range(runs):
        gc.collect()
        started = time.perf_counter_ns()
        _run(program, inputs, np)
        taken = time.perf_counter_ns() - started
        best = taken if best is None or taken < best else best
    return best or 0


def measure(program: corpus.Program, size: str, runs: int) -> Limit:
    """Measure one program's floor and where it stands.

    Args:
        program: The program.
        size: `corpus.SMALL` or `corpus.LARGE`.
        runs: How many times each way is timed.

    Returns:
        What was measured.
    """
    inputs = program.make(size)
    limit = Limit(program.name)
    lazy = get_lazy()
    limit.eager_ns = _timed(program, inputs, numpy, runs)
    limit.lazy_ns = _timed(program, inputs, lazy, runs)
    watcher = _Watcher()
    planner, meter = evaluate.planner, evaluate.meter
    evaluate.planner, evaluate.meter = watcher.plan, watcher.measure
    try:
        _run(program, inputs, lazy)
    finally:
        evaluate.planner, evaluate.meter = planner, meter
    slopes: dict[tuple, float | None] = {}
    for signature, spent in watcher.spent.items():
        label = signature.label
        limit.ns[label] = limit.ns.get(label, 0) + spent.ns
        limit.calls[label] = limit.calls.get(label, 0) + spent.calls
        if label not in REACHABLE:
            continue
        key = (signature.name, signature.kind, signature.dtype, signature.operands)
        if key not in slopes:
            slopes[key] = in_cache(signature)
        slope = slopes[key]
        stays = spent.ns if slope is None else min(spent.ns, int(slope * spent.elements))
        limit.kept[label] = limit.kept.get(label, 0) + stays
    if fuse.enable():
        try:
            limit.fused_ns = _timed(program, inputs, lazy, runs)
            limit.kernel_ns, limit.fused_nodes = _in_kernels(program, inputs, lazy)
        finally:
            fuse.disable()
    return limit


def _in_kernels(program: corpus.Program, inputs: tuple, lazy: Any) -> tuple[int, int]:
    """Run a program fused once more and add up the time its planned groups took.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        lazy: The stand-in for `numpy`.

    Returns:
        (nanoseconds inside the groups' own computing, the nodes they computed).
    """
    spent = [0]
    run = fuse.Plan.run

    def timed(self: fuse.Plan, group: Any, node: Node) -> bool:
        """Compute a group as the plan does, and add up the time."""
        started = time.perf_counter_ns()
        try:
            return run(self, group, node)
        finally:
            spent[0] += time.perf_counter_ns() - started

    fuse.STATS.reset()
    fuse.Plan.run = timed
    try:
        _run(program, inputs, lazy)
    finally:
        fuse.Plan.run = run
    return spent[0], fuse.STATS.fused_nodes


def render(limits: list[Limit], size: str, runs: int) -> str:
    """Write the tables as text.

    Args:
        limits: One per program.
        size: The size the programs were run at.
        runs: How many times each way was timed.

    Returns:
        The tables.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                              lstrip_blocks=True, keep_trailing_newline=True)
    return environment.get_template(TABLE_TEMPLATE).render(
        limits=limits, size=size, runs=runs, ns_per_ms=NS_PER_MS, classes=CLASSES, reachable=REACHABLE, short=SHORT, long=LONG)


@app.command()
def main(size: str = typer.Option(corpus.LARGE, help="The size the programs are run at: small or large."),
         program: list[str] = typer.Option(None, help="A program to measure; may be given several times. Default: all."),
         runs: int = typer.Option(5, help="How many times each way is timed; the quickest counts.")) -> None:
    """Measure how far fusion could take each program, and print the tables.

    Args:
        size: `corpus.SMALL` or `corpus.LARGE`.
        program: Names of the programs to measure; every program when empty.
        runs: How many times each way is timed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    limits = []
    for asked in corpus.by_name(list(program) if program else None):
        if not asked.available:
            log.info("%s is skipped: it needs %s", asked.name, ", ".join(asked.requires))
            continue
        if not asked.eager:
            continue  # A program that gathers by name has no eager run to set its floor against.
        limits.append(measure(asked, size, runs))
    log.info("%s", render(limits, size, runs).rstrip("\n"))


if __name__ == "__main__":
    app()
