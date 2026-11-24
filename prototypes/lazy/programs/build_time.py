"""Times how long the layer takes to write a program's graph down, with nothing computed.

    uv run python -m prototypes.lazy.programs.build_time

Each program is run with the lazy stand-in and its result is dropped without
a gather, so the time is the building alone: the operators, the rules that
give each node its shape and element type, the node. It does not depend on
the size of the arrays, which are never read. Beside it is the time NumPy
takes to compute the same program on the arrays used here, which are small,
for a sense of scale: on arrays this small, computing is quicker than
writing down. A second table times single operations, each made and dropped.

Timings are only meaningful on a machine that is doing nothing else.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy
import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from prototypes.lazy import get_lazy, graph
from prototypes.lazy.programs import attention, regression

log = logging.getLogger(__name__)

# Where the layer keeps its templates.
TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
# The template of the two tables.
TABLE_TEMPLATE = "build_time.txt.j2"
# Seed of the random data. The data's values play no part in building a graph.
SEED = 0
# Examples and features of the regression, and its gradient steps.
EXAMPLES, FEATURES, STEPS = 200, 8, 50
# Tokens, width and heads of the transformer block, and how many blocks are run one after another.
TOKENS, WIDTH, HEADS, BLOCKS = 16, 32, 4, 10
# The block's feed-forward layer is this many times as wide as the block.
WIDENING = 4
# Turns of the loop that writes into its arrays and of the loop that selects by a mask.
TURNS = 100
# How many times a single operation is made for one timing of it.
REPEATS = 20_000
# Nanoseconds in a millisecond and in a microsecond.
NS_PER_MS, NS_PER_US = 1e6, 1e3

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class ProgramTime:
    """What building one program's graph took."""

    # The program, as a reader would name it.
    name: str
    # Operations it added to the graph.
    operations: int
    # Time to build the graph, in milliseconds: the quickest of the runs.
    build_ms: float
    # The same per operation, in microseconds.
    per_operation_us: float
    # Time NumPy took to compute the program on the same arrays, in milliseconds: the quickest of the runs.
    numpy_ms: float


@dataclass(frozen=True)
class OperationTime:
    """What making one node, and dropping it, took."""

    # The operation as it is written.
    written: str
    # Time for one, in nanoseconds: the quickest of the runs.
    ns: float


def updates(np: Any, start: Any, row: Any, turns: int) -> Any:
    """Update an array in place, turn after turn: a write through an index, an operator, a view.

    Args:
        np: The `numpy` module, or the lazy stand-in for it.
        start: The array's first contents, one row per turn at least.
        row: What is assigned to one row on each turn.
        turns: How many turns to take.

    Returns:
        The array after the last turn.
    """
    x = np.array(start)
    for turn in range(turns):
        x[turn] = row
        x += 1.0
        x[turn, 1:] *= 2.0
    return x


def selections(np: Any, points: Any, turns: int) -> list:
    """Take the mean of the rows a condition selects, with another condition on each turn.

    Args:
        np: The `numpy` module, or the lazy stand-in for it.
        points: One point per row.
        turns: How many conditions to try.

    Returns:
        The mean of the selected rows for each condition. With the lazy
        stand-in each selection has a length that is not known.
    """
    points = np.asarray(points)
    means = []
    for turn in range(turns):
        near = points[:, 0] > -1.0 + 0.01 * turn
        means.append(points[near].mean(axis=0))
    return means


def block_parameters(rng: numpy.random.Generator) -> dict[str, numpy.ndarray]:
    """Draw float32 weights for one transformer block.

    Args:
        rng: The generator to draw from.

    Returns:
        Parameter name -> array, in the shapes `attention.block` expects.
    """
    shapes = {"gain_1": (WIDTH,), "bias_1": (WIDTH,), "gain_2": (WIDTH,), "bias_2": (WIDTH,),
              "w_qkv": (WIDTH, 3 * WIDTH), "b_qkv": (3 * WIDTH,), "w_out": (WIDTH, WIDTH), "b_out": (WIDTH,),
              "w_up": (WIDTH, WIDENING * WIDTH), "b_up": (WIDENING * WIDTH,),
              "w_down": (WIDENING * WIDTH, WIDTH), "b_down": (WIDTH,)}
    return {name: rng.standard_normal(shape).astype(numpy.float32) for name, shape in shapes.items()}


def blocks(tokens: Any, parameters: dict, count: int) -> Any:
    """Run a transformer block several times, each on the result of the one before.

    Args:
        tokens: Activations, (tokens, width): an `ndarray`, or a lazy array to have the graph written.
        parameters: The block's weights, by name.
        count: How many blocks.

    Returns:
        The activations after the last block.
    """
    for _ in range(count):
        tokens = attention.block(tokens, parameters, HEADS)
    return tokens


def quickest(run: Callable[[], Any], runs: int) -> tuple[float, int]:
    """Time a function several times and keep the quickest.

    Args:
        run: The function. What it returns is dropped after the clock is read.
        runs: How many times to run it.

    Returns:
        (the quickest time in nanoseconds, the operations the last run added to the graph).
    """
    best = float("inf")
    operations = 0
    for _ in range(runs):
        gc.collect()
        before = graph.COUNTS.deferred
        started = time.perf_counter_ns()
        result = run()
        best = min(best, time.perf_counter_ns() - started)
        operations = graph.COUNTS.deferred - before
        del result
    return best, operations


def programs(runs: int) -> list[ProgramTime]:
    """Time the building of each program's graph, and NumPy computing the same program.

    Args:
        runs: How many times each is run; the quickest counts.

    Returns:
        One row per program.
    """
    rng = numpy.random.default_rng(SEED)
    features = rng.standard_normal((EXAMPLES, FEATURES))
    targets = features @ rng.standard_normal(FEATURES)
    tokens = rng.standard_normal((TOKENS, WIDTH)).astype(numpy.float32)
    parameters = block_parameters(rng)
    start, row = numpy.zeros((TURNS, FEATURES)), rng.standard_normal(FEATURES)
    lazy = get_lazy()
    cases: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        (f"Linear regression, {STEPS} gradient steps",
         lambda: regression.fit(lazy, features, targets, STEPS), lambda: regression.fit(numpy, features, targets, STEPS)),
        (f"Transformer block, {BLOCKS} in a row",
         lambda: blocks(lazy.asarray(tokens), parameters, BLOCKS), lambda: blocks(tokens, parameters, BLOCKS)),
        (f"Writes into an array, {TURNS} turns",
         lambda: updates(lazy, start, row, TURNS), lambda: updates(numpy, start, row, TURNS)),
        (f"Rows a mask selects, {TURNS} masks",
         lambda: selections(lazy, features, TURNS), lambda: selections(numpy, features, TURNS)),
    ]
    rows = []
    for name, build, compute in cases:
        build()  # Once unmeasured, as a program's loop is after its first turn.
        build_ns, operations = quickest(build, runs)
        numpy_ns, _ = quickest(compute, runs)
        rows.append(ProgramTime(name, operations, build_ns / NS_PER_MS, build_ns / operations / NS_PER_US, numpy_ns / NS_PER_MS))
    return rows


def repeated(make: Callable[[], Any]) -> Callable[[], None]:
    """Build a function that makes one operation REPEATS times, dropping each result.

    Args:
        make: Makes the operation once.

    Returns:
        The function to time.
    """
    def run() -> None:
        """Make the operation REPEATS times."""
        for _ in range(REPEATS):
            make()
    return run


def operations(runs: int) -> list[OperationTime]:
    """Time single operations, each made and dropped.

    Args:
        runs: How many times each timing is taken; the quickest counts.

    Returns:
        One row per operation. The time of the loop that makes them, measured empty, is taken off.
    """
    lazy = get_lazy()
    rng = numpy.random.default_rng(SEED)
    a, b = lazy.asarray(rng.standard_normal((3, 4))), lazy.asarray(rng.standard_normal((3, 4)))
    c = a + b
    mask = c[:, 0] > 0.5
    selected = c[mask]
    cases: list[tuple[str, Callable[[], Any]]] = [
        ("a * b", lambda: a * b), ("a * 2.0", lambda: a * 2.0), ("-a", lambda: -a), ("a @ b.T", lambda: a @ b.T),
        ("np.exp(a), the real module", lambda: numpy.exp(a)), ("np.exp(a), the stand-in", lambda: lazy.exp(a)),
        ("a.sum()", lambda: a.sum()), ("a.mean(axis=-1, keepdims=True)", lambda: a.mean(axis=-1, keepdims=True)),
        ("np.sum(a, axis=0), the real module", lambda: numpy.sum(a, axis=0)),
        ("np.where(a > b, a, b)", lambda: numpy.where(a > b, a, b)),
        ("c[1]", lambda: c[1]), ("c[1, 1:3]", lambda: c[1, 1:3]), ("c.T", lambda: c.T), ("c.reshape(4, 3)", lambda: c.reshape(4, 3)),
        ("c[mask], the mask pending", lambda: c[mask]),
        ("selected * 2.0, a length not known", lambda: selected * 2.0),
        ("selected.mean(axis=0), a length not known", lambda: selected.mean(axis=0)),
    ]
    empty, _ = quickest(repeated(lambda: None), runs)
    rows = []
    for written, make in cases:
        make()
        taken, made = quickest(repeated(make), runs)
        # `a @ b.T` and `np.where(a > b, a, b)` make two nodes each time: the time is for the line as written.
        rows.append(OperationTime(written if made == REPEATS else f"{written}  ({made // REPEATS} nodes)", (taken - empty) / REPEATS))
    return rows


def render(program_rows: list[ProgramTime], operation_rows: list[OperationTime], runs: int) -> str:
    """Write the two tables as text.

    Args:
        program_rows: What `programs` returned.
        operation_rows: What `operations` returned.
        runs: How many runs each timing is the quickest of.

    Returns:
        The tables.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    return environment.get_template(TABLE_TEMPLATE).render(programs=program_rows, operations=operation_rows, runs=runs,
                                                           examples=EXAMPLES, features=FEATURES, tokens=TOKENS, width=WIDTH).strip("\n")


@app.command()
def main(runs: int = typer.Option(9, help="How many times each timing is taken; the quickest counts.")) -> None:
    """Time the building of the example programs' graphs and of single operations, and print both tables.

    Args:
        runs: How many times each timing is taken; the quickest counts.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log.info("%s", render(programs(runs), operations(runs), runs))


if __name__ == "__main__":
    app()
