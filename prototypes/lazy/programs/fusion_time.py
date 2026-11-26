"""Times each program of the corpus four ways: NumPy, and with hotkernels installed eagerly, lazily and fused.

    uv run --group apps python -m prototypes.lazy.programs.fusion_time --size large

`hotkernels.install()` is in effect for everything but the first way, since
that is how the package is used: each NumPy call it has a kernel for is
served by the kernel. For each program:

- **NumPy** is the program as it is, with `import numpy as np` and nothing
  installed. It is what every other way is measured against.
- **Installed** is the same eager program after `hotkernels.install()`, one
  operation at a time, with no graph. It is what the package gives today,
  and what a lazy run has to beat to be worth its graph.
- **Lazy** is the same program with the stand-in for `np`, its graph
  computed one node at a time, each with the call the program wrote.
- **Fused** is the lazy run with `fuse.enable()`: expressions the library
  has kernels for are computed by one kernel call each. Its time against
  NumPy's is the figure the layer is judged by.

The results of the three installed ways are compared bit for bit before
anything is printed, and the run stops if they differ.

A lazy run's time is split in two. *Computing* is the time inside the
layer's two computing functions: NumPy's calls or the kernels', and for a
fused run the planning before them, which is given by itself as well.
*Building* is everything else: the program's own Python, writing the nodes,
and whatever NumPy computes at once. With `--plain` the lazy and the fused
run are also timed with nothing installed, which says what the graph and
its kernels do by themselves.

A second table says what the graphs were: how many, how large, how many
expressions were found in them, how many nodes a kernel computed in place of
a NumPy call, how many calls and writes the fused run made at once instead
of writing a node, how many deferred writes it made in place, and how many
bytes of results were written to memory each way.

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
TABLE_TEMPLATE = "fusion_time.txt.j2"
# Nanoseconds in a millisecond, and bytes in a megabyte.
NS_PER_MS, BYTES_PER_MB = 1e6, 1e6
# The ways a program is run: eagerly, lazily node by node, and lazily with fusion.
NUMPY, LAZY, FUSED = "numpy", "lazy", "fused"
# The eager run under `hotkernels.install()`, as a key of `Row.timings`.
INSTALLED = "installed"

app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Timing:
    """The quickest of several runs of a program one way."""

    # Nanoseconds from the start of the program to its last result.
    total_ns: int
    # Of that, nanoseconds inside the layer's computing functions. 0 for NumPy.
    computing_ns: int = 0
    # Of that, nanoseconds planning the computations. 0 unless fused.
    planning_ns: int = 0

    @property
    def building_ns(self) -> int:
        """Nanoseconds outside the computing functions: the program's Python, the nodes, calls made at once."""
        return self.total_ns - self.computing_ns


@dataclass
class Shape:
    """What a program's graphs were, from one run that is not timed."""

    # Operations deferred, computations made, the nodes of the largest computation.
    deferred: int = 0
    graphs: int = 0
    largest_graph: int = 0
    # The elements of the largest array a node gave.
    largest_node: int = 0
    # Bytes of new arrays the nodes' calls wrote: results that own their memory. A write into an array that
    # exists is not counted: its node holds that array, whether it was written in place or copied first.
    written: int = 0
    # Calls and writes made as the program wrote them, without a node (`array.at_once_below`).
    at_once: int = 0
    written_at_once: int = 0
    # Deferred writes computed into the array's own memory, and into a copy of it.
    in_place: int = 0
    copied: int = 0


@dataclass
class Row:
    """Everything measured for one program."""

    # The program's name and one line on its size.
    name: str
    size: str
    # The quickest run each way, by NUMPY, INSTALLED, LAZY and FUSED: the last three under `hotkernels.install()`.
    timings: dict[str, Timing] = field(default_factory=dict)
    # The lazy ways with nothing installed, by LAZY and FUSED, when asked for.
    plain: dict[str, Timing] = field(default_factory=dict)
    # What the graphs were, node by node and fused.
    lazy: Shape = field(default_factory=Shape)
    fused: Shape = field(default_factory=Shape)
    # What fusion did in the fused run that was not timed.
    stats: fuse.Stats = field(default_factory=fuse.Stats)

    def ratio(self, slower: str, faster: str) -> float:
        """Return how many times quicker one way is than another."""
        return self.timings[slower].total_ns / self.timings[faster].total_ns


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


def _finish(result: Any) -> list:
    """Gather every lazy array of a program's result, all in one walk of the graph.

    One walk and not one for each array: a write into an array is made in
    place only when nothing still waits to read what the array held, and
    with the results asked for one at a time the operations behind a later
    result are still waiting while an earlier one is computed.

    Args:
        result: What a program returned.

    Returns:
        The values, as `ndarray`s and numbers.
    """
    return list(array.gather(*_values(result)))


def _run(program: corpus.Program, inputs: tuple, way: str) -> list:
    """Run a program one way, to its values.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        way: NUMPY, LAZY or FUSED.

    Returns:
        Its results as `ndarray`s and numbers.
    """
    if way == NUMPY:
        return _finish(program.run(numpy, inputs))
    if way == FUSED:
        fuse.enable(install=False)  # `measure` sees to what is installed for each way.
    try:
        return _finish(program.run(get_lazy(), inputs))
    finally:
        fuse.disable()


def _same(first: list, second: list) -> bool:
    """Say whether two runs gave the same results, in type, shape and bits.

    Args:
        first: One run's values.
        second: The other's.

    Returns:
        True when they match value for value.
    """
    if len(first) != len(second):
        return False
    for one, other in zip(first, second):
        one, other = numpy.asarray(one), numpy.asarray(other)
        if one.dtype != other.dtype or one.shape != other.shape or one.tobytes() != other.tobytes():
            return False
    return True


def quickest(program: corpus.Program, inputs: tuple, way: str, runs: int) -> Timing:
    """Time a program one way several times and keep the quickest run.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        way: NUMPY, LAZY or FUSED.
        runs: How many times to run it.

    Returns:
        The quickest run's times.
    """
    best: Timing | None = None
    for _ in range(runs):
        gc.collect()
        evaluate.CLOCK.reset()
        fuse.STATS.reset()
        started = time.perf_counter_ns()
        values = _run(program, inputs, way)
        taken = Timing(time.perf_counter_ns() - started, evaluate.CLOCK.ns, fuse.STATS.planning_ns)
        del values
        if best is None or taken.total_ns < best.total_ns:
            best = taken
    assert best is not None
    return best


def shape_of(program: corpus.Program, inputs: tuple, way: str) -> tuple[Shape, fuse.Stats, list]:
    """Run a program lazily once, not timed, and note what its graphs were.

    Args:
        program: The program.
        inputs: What `program.make` returned.
        way: LAZY or FUSED.

    Returns:
        (what the graphs were, what fusion did, the program's values).
    """
    shape = Shape()

    def note(node: Node, _ns: int) -> None:
        """Count what one call wrote."""
        value = node.value
        if isinstance(value, numpy.ndarray):
            shape.largest_node = max(shape.largest_node, value.size)
            if value.flags.owndata and node.kind != rules.WRITE:
                shape.written += value.nbytes

    evaluate.CLOCK.reset()
    fuse.STATS.reset()
    evaluate.WRITES.update(in_place=0, copied=0)
    made, wrote = array.at_once_counts()
    before, evaluate.meter = graph.COUNTS.deferred, note
    try:
        values = _run(program, inputs, way)
    finally:
        evaluate.meter = None
    shape.deferred = graph.COUNTS.deferred - before
    shape.at_once, shape.written_at_once = array.at_once_counts()[0] - made, array.at_once_counts()[1] - wrote
    shape.in_place, shape.copied = evaluate.WRITES["in_place"], evaluate.WRITES["copied"]
    shape.graphs, shape.largest_graph = evaluate.CLOCK.computations, evaluate.CLOCK.largest
    stats = fuse.Stats(**vars(fuse.STATS))
    stats.calls = dict(stats.calls)
    return shape, stats, values


def measure(program: corpus.Program, size: str, runs: int, enabled: Callable[[], Any], plain: bool) -> Row:
    """Measure one program every way.

    Args:
        program: The program.
        size: `corpus.SMALL` or `corpus.LARGE`.
        runs: How many times each way is timed; the quickest counts.
        enabled: `hotkernels.enabled`: a block inside which the single-operation kernels are installed.
        plain: Whether to time the lazy ways with nothing installed as well.

    Returns:
        The program's row.

    Raises:
        AssertionError: When a lazy run's results are not the eager program's bit for bit, with the kernels
            installed or without.
    """
    inputs = program.make(size)
    row = Row(program.name, program.sizes[size])
    row.timings[NUMPY] = quickest(program, inputs, NUMPY, runs)
    with enabled():
        served = _run(program, inputs, NUMPY)
        row.lazy, _, lazily = shape_of(program, inputs, LAZY)
        row.fused, row.stats, fused = shape_of(program, inputs, FUSED)
        assert _same(served, lazily), f"{program.name}: under hotkernels.install() the lazy run does not give the eager program's results"
        assert _same(served, fused), f"{program.name}: under hotkernels.install() the fused run does not give the eager program's results"
        del served, lazily, fused
        row.timings[INSTALLED] = quickest(program, inputs, NUMPY, runs)
        for way in (LAZY, FUSED):
            row.timings[way] = quickest(program, inputs, way, runs)
    if plain:
        expected = _run(program, inputs, NUMPY)
        for way in (LAZY, FUSED):
            assert _same(expected, _run(program, inputs, way)), f"{program.name}: the {way} run does not give NumPy's results"
            row.plain[way] = quickest(program, inputs, way, runs)
    return row


def render(rows: list[Row], size: str, runs: int) -> str:
    """Write the tables as text.

    Args:
        rows: One row per program.
        size: The size the programs were run at.
        runs: How many runs each timing is the quickest of.

    Returns:
        The tables.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                              lstrip_blocks=True)
    calls: dict[str, int] = {}
    for row in rows:
        for name, count in row.stats.calls.items():
            calls[name] = calls.get(name, 0) + count
    return environment.get_template(TABLE_TEMPLATE).render(
        rows=rows, size=size, runs=runs, ns_per_ms=NS_PER_MS, bytes_per_mb=BYTES_PER_MB, core=array.core_in_use(),
        numpy=NUMPY, installed=INSTALLED, lazy=LAZY, fused=FUSED, calls=sorted(calls.items(), key=lambda item: item[1], reverse=True),
        kernels=len({kernel.name for kernel in fuse.LIBRARY.values()})).strip("\n")


@app.command()
def main(size: str = typer.Option(corpus.LARGE, help="The size the programs are run at: small or large."),
         program: list[str] = typer.Option(None, help="A program to time; may be given several times. Default: all."),
         runs: int = typer.Option(5, help="How many times each way is timed; the quickest counts."),
         plain: bool = typer.Option(False, help="Also time the lazy ways with nothing installed.")) -> None:
    """Time the corpus four ways and print the tables.

    Args:
        size: The size the programs are run at.
        program: The names of the programs to time, or none for all of them that run with NumPy.
        runs: How many times each way is timed.
        plain: Whether to time the lazy and the fused run with nothing installed as well.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not fuse.enable(install=False):
        log.error("the fused kernels are not built: uv run python -m prototypes.lazy.build_fused")
        raise typer.Exit(1)
    fuse.disable()
    import hotkernels
    asked = corpus.by_name(list(program) if program else None)
    for each in asked:
        if not each.available:
            log.info("skipped %s: it needs %s (uv run --group apps ...)", each.name, ", ".join(each.requires))
    chosen = [each for each in asked if each.eager and each.available]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = [measure(each, size, runs, hotkernels.enabled, plain) for each in chosen]
    log.info("%s", render(rows, size, runs))


if __name__ == "__main__":
    app()
