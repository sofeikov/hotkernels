# A prototype of the lazy NumPy layer

The design in [docs/lazy-design.md](../../docs/lazy-design.md), written out so it can be read and tried. It is not part of the `hotkernels` package and nothing imports it.

```python
from prototypes.lazy import get_lazy

np = get_lazy()                 # where `import numpy as np` stood
x = np.arange(6.0) * 2 + 1      # written down, not computed
print(x)                        # LazyArray(float64, shape=(6,), pending)
x.gather()                      # array([ 1.,  3.,  5.,  7.,  9., 11.])
```

Try it from the repository's root with `uv run python`.

## What it does

- Defers a narrow set of NumPy operations into a graph: element-by-element operations, the matrix product, the common reductions with `argmax` and `argmin`, running totals, `np.searchsorted`, views, indexing by arrays, and `np.partition`.
- Defers `x[mask]` for a mask not yet computed, and `np.flatnonzero`, `np.nonzero` and `np.argwhere`: the result has one length that is not known, `shape=(?, 4)`, and what is computed from it stays deferred where the shape of the result does not depend on that length.
- Makes every other NumPy call at once, on gathered operands, and returns a lazy array that has its values.
- Reads one element of an array that has its values on the spot, `x[i]` or `x[r, c]`, as a lazy array that has its value: a Python loop over such an array adds no node and no graph per element.
- Makes a write into an array (`w -= step`, `x[i] = v`, `out=x`) a node of the graph, so a loop that updates its arrays in place is one graph. A write into memory the user also holds is made at once.
- Gives values through `gather()`, and computes on the spot where something else needs them (`float(x)`, `if x > 0:`, `np.asarray(x)`), saying so in the log once per line and counting every time for `lazy_report()`. A number read from an array that has its values computes nothing and is not counted. `get_lazy(strict=True)` gives arrays that raise `NotGathered` there instead.
- Computes the graph one node at a time, each with the call the user wrote, spelled the same way, so results are NumPy's and nothing is faster. With `hotkernels.install()` in effect those calls reach the kernels as the eager program's do.
- Writes a node in 1.4 to 2.5 µs in Python, and in 0.4 to 0.7 µs with its C core built. What a node needs and a loop repeats is remembered: the shape and element type of a call, the line that wrote it, the error settings in effect. Each memory gives the answer that asking again would give. The core is the quick road of the Python: it finds a remembered answer, and for the first call of a ufunc, a reduction or a view function it asks NumPy for the element type as the Python does and works the shape out by a rule it holds beside Python's. A program built once in a fresh process takes 0.17 ms for 23 operations and 0.33 ms for the 55 of a transformer block, where NumPy's first run of the same programs takes 0.08 ms and 0.27 ms.
- Patches nothing: `get_lazy()` returns an object, and the `numpy` module and other modules' imports are as they were. An `ndarray` that exists enters with `np.asarray(a)`.

## Reading it

| File | Holds |
|---|---|
| [graph.py](graph.py) | The node, and the list of nodes not computed yet |
| [rules.py](rules.py) | Which calls are deferred, and their shape and element type without computing |
| [evaluate.py](evaluate.py) | Computing nodes |
| [array.py](array.py) | `LazyArray`: operators, NumPy's two protocols, the refusals, `gather()` |
| [namespace.py](namespace.py) | The stand-in for the `numpy` module |
| [_core.c](_core.c), [build_core.py](build_core.py) | The quick road for writing a node, in C, and the command that builds it. Optional |
| [explain.py](explain.py) | The text of a pending graph |
| [record.py](record.py) | The graph as an object, and the mode that records the graphs a block of code produced |
| [report.py](report.py) | What was computed without being asked, ranked: `lazy_report()` |
| [view.py](view.py), [templates/](templates) | The page: the program's code and its graphs. `graph.html.j2`, `graph.js`, `graph.css` |
| [programs/](programs) | A linear regression and a transformer block, each run both ways by the tests; a regression that gathers as it goes; `pages.py`, which writes the page of all three; and `build_time.py`, which times how long their graphs take to write |

## Looking at its graphs

```bash
uv run python -m prototypes.lazy.programs.pages --out lazy-graphs.html --show
```

Runs the three example programs in one recorded block and opens the page of their graphs. The page loads its layout library from a CDN, so it needs a network connection. In your own code:

```python
from prototypes.lazy import get_lazy, record_graphs

np = get_lazy()
with record_graphs() as recording:
    ...                       # NumPy code on lazy arrays
recording.show()              # or recording.save("graphs.html"), recording.to_json()
print(recording.report())     # what NumPy had to compute at once, and where graphs were cut, ranked
```

The page opens on your code. `record_graphs(source="lines")` or `source="none"` makes pages that carry less of it.

To see as it runs what the layer could not defer, turn its logger on: `logging.getLogger("prototypes.lazy").setLevel(logging.INFO)`. `DEBUG` adds the places where a graph was computed early.

The design of the two is in [docs/lazy-recording-design.md](../../docs/lazy-recording-design.md) and [docs/lazy-graph-page-design.md](../../docs/lazy-graph-page-design.md).

## Building the C core

```bash
uv run python -m prototypes.lazy.build_core
```

Compiles `_core.c` with clang into this directory. The layer takes the core when it is there and is the same without it, only slower to write a graph; `prototypes.lazy.array.core_in_use()` says which. Build it again after changing `_core.c`.

## Timing it

```bash
uv run python -m prototypes.lazy.programs.build_time
```

Writes each example program's graph down and drops it without a gather, and prints the time for the graph and per operation, beside the time NumPy takes to compute the same program on the small arrays used. A second table times single lines, each made and dropped. `--no-core` times Python alone. Run it on a machine that is doing nothing else.

## Testing it

```bash
uv run --group dev pytest tests/test_lazy_prototype.py tests/test_lazy_graphs.py tests/test_lazy_building.py
```

Each test holds one rule of the design to what the real `numpy` module gives for the same lines. Those of `test_lazy_building.py` hold what the layer remembers while it writes a graph to what it would work out from nothing, and what the C core answers for a first call to what the rules answer. Every test runs twice, in Python alone and with the C core, which is built for the run; the second is skipped where clang is missing.
