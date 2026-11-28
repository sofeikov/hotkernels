"""Says how far fusing element-by-element operations could take each program, and how far the fused run is from it.

    uv run --group apps --group workloads python -m prototypes.lazy.programs.limit --size large

Fusion removes two things from an operation on arrays: the pass over memory
that reads its operands and writes its result, and the call into NumPy. It
cannot remove the arithmetic itself. So every operation has a part no fusion
touches, and that part can be measured: the same NumPy loop on arrays that
fit the first-level cache, per element, with the cost of the call taken off.

`hotkernels.install()` is in effect for all of it but one timing, since
that is how the package is used: the floor is of the eager program with the
single-operation kernels serving it, and the fused run has them too.

For each program of `corpus.py` this tool:

- times the program four ways: eagerly as it is, eagerly with the kernels
  installed, lazily node by node, and fused;
- runs it lazily, node by node, with every call timed, and puts each
  operation in a class by what a fused kernel would need to make it;
- measures, for each different operation, what an element costs in the
  cache, on the first elements of the operands the program gave it, and
  takes that times the elements as the part that has to stay;
- runs it fused once more with every call NumPy still makes timed, and holds
  each against what its loop costs in the cache.

The four ways are timed in turn: each round runs the program once every
way, and the quickest run of each way counts. Timed one way after another,
a slow stretch of the machine or a state of the process falls on all the
runs of one way, and `at best` and `now`, which are against NumPy's run,
timed first, are too high by what that run lost. `--one-after-another` asks
for that order.

The values matter: what `erf` or `exp` costs depends on its argument, so an
operation is timed on the program's own numbers, not on made-up ones, and
on many different pieces of them, each once a round: a loop that branches
on its data runs faster over the same piece again and again than it ever
does in the program.

The **floor** of a program is its installed eager time less everything that could be
removed from its element-by-element operations and its reductions over
every element: what a fusion of all of them, of any kind, with no cost of
its own, would leave. What lies between the fused run and the floor is what
is left to gain; the classes say which operations it is in.

The floor is of the eager program, and a kernel may run below it, since it
makes several operations for each load and store. So the last table is of
the fused run itself, in three parts: the time inside the kernels'
programs; the calls NumPy still makes for operations a kernel could make,
with what of each would go if it cost what its loop costs in the cache; and
the rest of the run, which is the program's own code, a library's, matrix
products, moves and the layer. The sum of what would go is the most a
further fusion of anything not yet fused could take off that run.

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

from prototypes.lazy import array, evaluate, fuse, get_lazy, graph, rules
from prototypes.lazy.graph import Node
from prototypes.lazy.programs import corpus

log = logging.getLogger(__name__)

# Where the layer keeps its templates, and the template of this tool's tables.
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
TABLE_TEMPLATE = "limit.txt.j2"
# Nanoseconds in a millisecond.
NS_PER_MS = 1e6
# The ways a program is timed, in the order of a round: eagerly with nothing installed, and under
# `hotkernels.install()` eagerly, lazily node by node, and lazily with fusion.
NUMPY, INSTALLED, LAZY, FUSED = "numpy", "installed", "lazy", "fused"
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
# How many pieces of its operands an operation is timed on at each length, at most, and how many rounds over all
# of them are timed; the quickest round counts.
MOST_PIECES, CALIBRATION_ROUNDS = 64, 5

# The smallest part of a program's eager time its floor may be, as a divisor, and still be printed: a hundredth.
LEAST_FLOOR = 100
# The package's logger. It says at INFO that it was installed and uninstalled, which timing in turn does once a
# round for every program; the tool lets only its warnings through.
PACKAGE = "hotkernels"

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
    # Nanoseconds of the eager run with nothing installed: what the others are measured against.
    numpy_ns: int = 0
    # Nanoseconds of the eager run, of the lazy run node by node, and of the fused run, all three under
    # `hotkernels.install()`: the quickest of each.
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
    # Nanoseconds NumPy's own calls took in the fused run for the nodes of each class that no program computed,
    # and of that, what the same loops take in the cache: the part that would stay whatever computed them.
    still_ns: dict[str, int] = field(default_factory=dict)
    still_kept: dict[str, int] = field(default_factory=dict)

    def still_removable(self, label: str) -> int:
        """Return what could at most be taken off the calls NumPy still makes for a class in the fused run."""
        return max(0, self.still_ns.get(label, 0) - self.still_kept.get(label, 0))

    @property
    def still_reachable_ns(self) -> int:
        """Nanoseconds that could still be removed from the fused run, over the classes fusion could reach."""
        return sum(self.still_removable(label) for label in REACHABLE)

    @property
    def unreachable_ns(self) -> int:
        """Nanoseconds of the fused run that are not in its kernels and not in a call fusion could reach."""
        return self.fused_ns - self.kernel_ns - sum(self.still_ns.get(label, 0) for label in REACHABLE)

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
    def has_floor(self) -> bool:
        """Whether the floor says anything: not when what could be removed is all but nothing of the eager time.

        That is a program on arrays of a few elements, where an operation
        costs what its call into NumPy does and the arithmetic in the cache
        is next to nothing beside it.
        """
        return self.floor_ns * LEAST_FLOOR > self.eager_ns

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
    """Computes one program's nodes one at a time, timing each and, once for each different operation, its cost in the cache.

    It is a plan as `evaluate` asks a planner for one: every node it can
    time is a group of its own, which it computes by the node's own call.
    A node it does not take (a write, a node with a failed operand) is
    computed by `evaluate` as ever, and counted through the meter.
    """

    def __init__(self) -> None:
        """Start with nothing seen."""
        # What `evaluate` reads of a plan: no node is inside a group, and each group is one node's signature.
        self.inner: set[int] = set()
        self.groups: dict[int, tuple[Signature, int]] = {}
        # What each signature took, and what one element of it costs in the cache; None where that was not measured.
        self.spent: dict[Signature, Spent] = {}
        self.slopes: dict[Signature, float | None] = {}

    def plan(self, todo: list[Node]) -> "_Watcher":
        """Write down what each node of a computation is, while it still has its operands.

        Args:
            todo: The pending nodes about to be computed.

        Returns:
            This object, as the computation's plan.
        """
        self.groups = {}
        for node in todo:
            largest = max([_elements(node.shape)] + [_elements(arg.shape) for arg in node.args
                                                      if isinstance(arg, (Node, numpy.ndarray))])
            signature = Signature(classify(node), node.operation, node.name, node.kind, str(node.dtype),
                                  tuple(_operand(arg, largest) for arg in node.args))
            self.groups[node.number] = (signature, largest)
        return self

    def run(self, group: tuple[Signature, int], node: Node) -> bool:
        """Compute one node by its own call, timed; and time the call in the cache if its like was not yet.

        Args:
            group: The node's signature and the elements of its largest operand or result.
            node: The pending node.

        Returns:
            True when the node has its value. False for a write or a node with an operand that has none, which
            `evaluate` computes.
        """
        signature, largest = group
        values = []
        for arg in node.args:
            if type(arg) is Node:
                if arg.operation is not None or arg.failure is not None or arg.value is None:
                    return False
                arg = arg.value
            values.append(arg)
        if node.kind == rules.WRITE:
            return False
        operation, kwargs = node.operation, node.kwargs
        with warnings.catch_warnings(), numpy.errstate(**(node.errstate or {})):
            warnings.simplefilter("ignore")
            started = time.perf_counter_ns()
            value = operation(*values, **kwargs)
            taken = time.perf_counter_ns() - started
        node.value = value
        if not graph.is_known(node.shape):
            node.shape = numpy.shape(value)
        graph.settle(node)
        self._count(signature, largest, taken)
        if signature.label in REACHABLE and signature not in self.slopes:
            self.slopes[signature] = in_cache(operation, values, kwargs, largest)
        return True

    def members(self, group: tuple[Signature, int]) -> list[Node]:
        """Return the nodes inside a group that was not computed: none, since a group here is one node."""
        return []

    def _count(self, signature: Signature, largest: int, ns: int) -> None:
        """Add one call to what its signature took."""
        spent = self.spent.setdefault(signature, Spent())
        spent.ns += ns
        spent.calls += 1
        spent.elements += largest

    def measure(self, node: Node, ns: int) -> None:
        """Count the time of a call `evaluate` made itself, under its signature.

        Args:
            node: The node, computed.
            ns: Nanoseconds its call took, or the whole of `run` for a node computed there, which is not counted twice.
        """
        found = self.groups.get(node.number)
        if found is not None and (node.kind == rules.WRITE or found[0] not in self.spent):
            self._count(found[0], found[1], ns)


def _piece(value: Any, largest: int, start: int, length: int) -> Any:
    """Cut a piece out of an operand, for a calibration.

    Args:
        value: An operand as the program had it.
        largest: The elements of the call's largest operand or result.
        start: Where the piece starts among the elements of an array the loop walks, in the order of its rows.
        length: How many elements the piece has.

    Returns:
        A contiguous array of `length` elements for an array as large as the largest, its elements from `start`
        on over and over if fewer are left; one element for a smaller array; anything else as it is.
    """
    if not isinstance(value, numpy.ndarray):
        return value
    if value.size != largest or largest <= 1:
        return value.reshape(-1)[:1].reshape(()) if value.size else value
    cut = numpy.array(value.flat[start:start + length])
    return cut if cut.size == length else numpy.resize(cut, length)


def in_cache(operation: Callable[..., Any], values: list, kwargs: dict, largest: int) -> float | None:
    """Measure what one element of an operation costs on arrays in the cache, the call left out.

    Args:
        operation: The node's callable.
        values: Its operands as the program had them.
        kwargs: Its keywords.
        largest: The elements of its largest operand or result.

    Returns:
        Nanoseconds for one element: the slope of the call's time between pieces of SHORT and of LONG elements,
        over up to MOST_PIECES different pieces of the operands. None for an operation that cannot be made on
        such pieces, whose whole time is then counted as staying.
    """
    count = max(1, min(MOST_PIECES, largest // LONG))
    times = []
    for length in (SHORT, LONG):
        pieces = [[_piece(value, largest, place * LONG, length) for value in values] for place in range(count)]
        best = None
        try:
            with warnings.catch_warnings(), numpy.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                for _ in range(CALIBRATION_ROUNDS):
                    started = time.perf_counter_ns()
                    for operands in pieces:
                        operation(*operands, **kwargs)
                    taken = time.perf_counter_ns() - started
                    best = taken if best is None or taken < best else best
        except Exception:  # Whatever a piece of an operand does to an operation: it is then not calibrated.
            return None
        times.append((best or 0) / count)
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


def _timed_fused(program: corpus.Program, inputs: tuple, runs: int) -> int:
    """Time a program's fused run several times and return the quickest, in nanoseconds.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        runs: How many times to run it.

    Returns:
        The quickest run, or 0 where the fused kernels are not built.
    """
    if not fuse.enable(install=False):  # The caller has the single-operation kernels installed around this.
        return 0
    try:
        return _timed(program, inputs, get_lazy(), runs)
    finally:
        fuse.disable()


def _installed_ways(program: corpus.Program, inputs: tuple, runs: int) -> dict[str, int]:
    """Time the three ways that run under `hotkernels.install()`, one after another.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        runs: How many times each way is run before the next.

    Returns:
        The quickest run in nanoseconds by INSTALLED, LAZY and FUSED.
    """
    return {INSTALLED: _timed(program, inputs, numpy, runs), LAZY: _timed(program, inputs, get_lazy(), runs),
            FUSED: _timed_fused(program, inputs, runs)}


def one_after_another(program: corpus.Program, inputs: tuple, runs: int, enabled: Callable[[], Any]) -> dict[str, int]:
    """Time a program one way after another: every run of NumPy's first, then every run of the next way.

    What lasts a few seconds then falls on one way and not on the next; see `in_turn`.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        runs: How many times each way is timed.
        enabled: `hotkernels.enabled`: a block inside which the single-operation kernels are installed.

    Returns:
        The quickest run in nanoseconds by NUMPY, INSTALLED, LAZY and FUSED; FUSED is 0 where fusion is not built.
    """
    times = {NUMPY: _timed(program, inputs, numpy, runs)}
    with enabled():
        times.update(_installed_ways(program, inputs, runs))
    return times


def in_turn(program: corpus.Program, inputs: tuple, runs: int, enabled: Callable[[], Any]) -> dict[str, int]:
    """Time a program every way in turn, round after round, and keep the quickest run of each way.

    Timed one way after another, a slow stretch of the machine or a state of the process falls on all the runs
    of one way. In turn it falls on a round of every way, as in `fusion_time.alternately`;
    docs/lazy-codebases-findings.md, "Issue 323", has what the order was worth.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        runs: How many rounds; each times every way once.
        enabled: `hotkernels.enabled`: a block inside which the single-operation kernels are installed.

    Returns:
        The quickest run in nanoseconds by NUMPY, INSTALLED, LAZY and FUSED; FUSED is 0 where fusion is not built.
    """
    times: dict[str, int] = {}
    for _ in range(runs):
        once = {NUMPY: _timed(program, inputs, numpy, 1)}
        with enabled():
            once.update(_installed_ways(program, inputs, 1))
        times = {way: min(ns, times.get(way, ns)) for way, ns in once.items()}
    return times


def measure(program: corpus.Program, size: str, runs: int, enabled: Callable[[], Any], alternate: bool = True) -> Limit:
    """Measure one program's floor and where it stands.

    Args:
        program: The program.
        size: `corpus.SMALL` or `corpus.LARGE`.
        runs: How many times each way is timed.
        enabled: `hotkernels.enabled`: a block inside which the single-operation kernels are installed.
        alternate: Whether the ways are timed in turn, round after round (`in_turn`), and not one after another.

    Returns:
        What was measured.
    """
    inputs = program.make(size)
    limit = Limit(program.name)
    times = (in_turn if alternate else one_after_another)(program, inputs, runs, enabled)
    limit.numpy_ns, limit.eager_ns, limit.lazy_ns, limit.fused_ns = (times[way] for way in (NUMPY, INSTALLED, LAZY, FUSED))
    with enabled():
        _installed(program, inputs, limit)
    return limit


def _installed(program: corpus.Program, inputs: tuple, limit: Limit) -> None:
    """Measure where one program's time is, node by node and fused, with the single-operation kernels installed.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        limit: Where what is measured goes.
    """
    lazy = get_lazy()
    watcher = _Watcher()
    planner, meter = evaluate.planner, evaluate.meter
    evaluate.planner, evaluate.meter = watcher.plan, watcher.measure
    try:
        _run(program, inputs, lazy)
    finally:
        evaluate.planner, evaluate.meter = planner, meter
    for signature, spent in watcher.spent.items():
        label = signature.label
        limit.ns[label] = limit.ns.get(label, 0) + spent.ns
        limit.calls[label] = limit.calls.get(label, 0) + spent.calls
        if label not in REACHABLE:
            continue
        slope = watcher.slopes.get(signature)
        stays = spent.ns if slope is None else min(spent.ns, int(slope * spent.elements))
        limit.kept[label] = limit.kept.get(label, 0) + stays
    if fuse.enable(install=False):  # `measure` has the kernels installed around this.
        try:
            limit.kernel_ns, limit.fused_nodes = _in_kernels(program, inputs, lazy)
            limit.still_ns, limit.still_kept = _still_numpys(program, inputs, lazy, watcher)
        finally:
            fuse.disable()


class _Remaining:
    """Notes, in a fused run, what the calls NumPy still makes take, by the class of their node.

    It stands between `evaluate` and the planner: the plan is fusion's own,
    and each node of it is written down while it still has its operands,
    as `_Watcher` writes one down, so that a call the meter reports can be
    put in its class and held against what its like costs in the cache.
    """

    def __init__(self, planner: Callable[[list[Node]], Any], slopes: dict[Signature, float | None]) -> None:
        """Start with nothing seen.

        Args:
            planner: Fusion's planner.
            slopes: What one element of each operation costs in the cache, from the node-by-node run.
        """
        self.planner, self.slopes = planner, slopes
        # The nodes of the computation at hand: their signature and largest operand, and those a program computes.
        self.known: dict[int, tuple[Signature, int]] = {}
        self.grouped: set[int] = set()
        # By class: nanoseconds NumPy's calls took, and what the same loops take in the cache.
        self.ns: dict[str, int] = {}
        self.kept: dict[str, int] = {}

    def plan(self, todo: list[Node]) -> Any:
        """Write the nodes down, and plan as fusion does.

        Args:
            todo: The pending nodes about to be computed.

        Returns:
            Fusion's plan.
        """
        self._note(todo)
        plan = self.planner(todo)
        self.grouped = set(plan.groups) if plan is not None else set()
        return plan

    def _note(self, todo: list[Node]) -> None:
        """Write down the signature and the largest operand of each node.

        A function of its own, so that no name of it still refers to a node
        when fusion plans: the planner fuses an operation only when nothing
        but the operation that reads it refers to its node, and a loop
        variable left on the oldest node of each computation kept that
        node's expression from being fused, in this run alone.

        Args:
            todo: The pending nodes about to be computed.
        """
        self.known = {}
        for node in todo:
            largest = max([_elements(node.shape)] + [_elements(arg.shape) for arg in node.args
                                                      if isinstance(arg, (Node, numpy.ndarray))])
            signature = Signature(classify(node), node.operation, node.name, node.kind, str(node.dtype),
                                  tuple(_operand(arg, largest) for arg in node.args))
            self.known[node.number] = (signature, largest)

    def measure(self, node: Node, ns: int) -> None:
        """Count a call that was not a program's under its node's class.

        Args:
            node: The node, computed.
            ns: Nanoseconds its computing took.
        """
        found = self.known.get(node.number)
        if found is None or node.number in self.grouped:
            return
        signature, largest = found
        self.ns[signature.label] = self.ns.get(signature.label, 0) + ns
        slope = self.slopes.get(signature)
        stays = ns if slope is None else min(ns, int(slope * largest))
        self.kept[signature.label] = self.kept.get(signature.label, 0) + stays


def _still_numpys(program: corpus.Program, inputs: tuple, lazy: Any, watcher: _Watcher) -> tuple[dict[str, int], dict[str, int]]:
    """Run a program fused once more and add up what NumPy's own calls still take in it.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        lazy: The stand-in for `numpy`.
        watcher: The node-by-node run's watcher, for what each operation costs in the cache.

    Returns:
        (nanoseconds of NumPy's calls by class, what of that stays in the cache by class).
    """
    remaining = _Remaining(evaluate.planner, watcher.slopes)
    planner, meter = evaluate.planner, evaluate.meter
    evaluate.planner, evaluate.meter = remaining.plan, remaining.measure
    try:
        _run(program, inputs, lazy)
    finally:
        evaluate.planner, evaluate.meter = planner, meter
    return remaining.ns, remaining.kept


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


def render(limits: list[Limit], size: str, runs: int, alternate: bool) -> str:
    """Write the tables as text.

    Args:
        limits: One per program.
        size: The size the programs were run at.
        runs: How many times each way was timed.
        alternate: Whether the ways were timed in turn and not one after another.

    Returns:
        The tables.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                              lstrip_blocks=True, keep_trailing_newline=True)
    return environment.get_template(TABLE_TEMPLATE).render(
        limits=limits, size=size, runs=runs, alternate=alternate, ns_per_ms=NS_PER_MS, classes=CLASSES, reachable=REACHABLE, short=SHORT, long=LONG)


@app.command()
def main(size: str = typer.Option(corpus.LARGE, help="The size the programs are run at: small or large."),
         program: list[str] = typer.Option(None, help="A program to measure; may be given several times. Default: all."),
         runs: int = typer.Option(5, help="How many times each way is timed; the quickest counts."),
         alternate: bool = typer.Option(True, "--alternate/--one-after-another",
                                        help="Time the ways in turn, round after round, so that each meets the "
                                             "process in the same state; or every run of one way before the next.")) -> None:
    """Measure how far fusion could take each program, and print the tables.

    Args:
        size: `corpus.SMALL` or `corpus.LARGE`.
        program: Names of the programs to measure; every program when empty.
        runs: How many times each way is timed.
        alternate: Whether the ways are timed in turn and not one after another.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger(PACKAGE).setLevel(logging.WARNING)
    import hotkernels
    limits = []
    for asked in corpus.by_name(list(program) if program else None):
        if not asked.available:
            log.info("%s is skipped: it needs %s", asked.name, ", ".join(asked.requires))
            continue
        if not asked.eager:
            continue  # A program that gathers by name has no eager run to set its floor against.
        limits.append(measure(asked, size, runs, hotkernels.enabled, alternate))
    log.info("%s", render(limits, size, runs, alternate).rstrip("\n"))


if __name__ == "__main__":
    app()
