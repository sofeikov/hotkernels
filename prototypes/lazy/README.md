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
- Makes a write into an array (`w -= step`, `x[i] = v`, `out=x`) a node of the graph, so a loop that updates its arrays in place is one graph. A write into memory the user also holds is made at once. When a write is computed it goes into the array's own memory, as NumPy's does, unless something still reads what the array held; then into a copy.
- Gives values through `gather()`, and computes on the spot where something else needs them (`float(x)`, `if x > 0:`, `np.asarray(x)`), saying so in the log once per line and counting every time for `lazy_report()`. A number read from an array that has its values computes nothing and is not counted. `get_lazy(strict=True)` gives arrays that raise `NotGathered` there instead.
- Computes the graph one node at a time, each with the call the user wrote, spelled the same way, so results are NumPy's. `x ** 2` is NumPy's operator, which squares, not `np.power`. With `hotkernels.install()` in effect those calls reach the kernels as the eager program's do.
- With `fuse.enable()`, computes an expression the graph holds, such as `a + s * b`, a whole stencil or a polynomial of 153 operations, by kernel calls made block by block in one pass over memory, with NumPy's bits; and makes a call on a small array at once instead of writing a node for it. See [Fusion](#fusion).
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
| [_core.c](_core.c), [build_core.py](build_core.py) | The layer's hot path in C: writing a node, making a small call where the program wrote it, reading the expressions for fusion; and the command that builds it. Optional |
| [explain.py](explain.py) | The text of a pending graph |
| [record.py](record.py) | The graph as an object, and the mode that records the graphs a block of code produced |
| [report.py](report.py) | What was computed without being asked, ranked: `lazy_report()` |
| [view.py](view.py), [templates/](templates) | The page: the program's code and its graphs. `graph.html.j2`, `graph.js`, `graph.css` |
| [fuse.py](fuse.py) | Fusion: reading expressions off the graph, the library of shapes, covering an expression with kernel calls, and computing a group. Optional: nothing imports it until `fuse.enable()` |
| [_fused.c](_fused.c), [build_fused.py](build_fused.py) | The run of an expression's kernel calls, block by block, on arrays of any shape, in C, and the command that builds it with the kernels |
| [kernels/](kernels) | The fused kernels, one directory each, checked by the judge as the kernels of `kernels/` are |
| [make_kernel.py](make_kernel.py) | Writes a kernel from the name of its shape: the spec, the C reference and the NEON assembly |
| [programs/](programs) | A linear regression and a transformer block, each run both ways by the tests; a regression that gathers as it goes; `pages.py`, which writes the page of all three; and `build_time.py`, which times how long their graphs take to write. For fusion: `corpus.py`, twenty-six programs that run with NumPy or with the stand-in, three of them copied into `vendor/` and the rest the applications and workloads of `delivery/`; `mine.py`, which ranks the shapes of expression in them; `fusion_time.py`, which times each four ways; and `limit.py`, which measures what part of each program's time no fusion could remove, what the fused run leaves above it by kind of operation, and what the calls NumPy still makes in the fused run could at most lose |

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

Compiles `_core.c` with clang into this directory; it needs NumPy's headers, which the installed NumPy brings. The layer takes the core when it is there and gives the same results without it, only slower: every graph is then written and read by Python, and no call is made at once. `prototypes.lazy.array.core_in_use()` says which. Build it again after changing `_core.c`.

## Timing it

```bash
uv run python -m prototypes.lazy.programs.build_time
```

Writes each example program's graph down and drops it without a gather, and prints the time for the graph and per operation, beside the time NumPy takes to compute the same program on the small arrays used. A second table times single lines, each made and dropped. `--no-core` times Python alone. Run it on a machine that is doing nothing else.

## Fusion

The design and what was measured are in [docs/lazy-fusion-design.md](../../docs/lazy-fusion-design.md).

```bash
uv run python -m prototypes.lazy.build_fused      # assemble the fused kernels into the module that calls them
```

```python
from prototypes.lazy import fuse, get_lazy

fuse.enable()                   # False, and nothing changes, if the module is not built; installs hotkernels too
np = get_lazy()
w = np.asarray(weights) - 0.1 * np.asarray(gradient)
w.gather()                      # one kernel call, one pass, NumPy's bits
fuse.STATS                      # what was planned, which kernels ran, what was left to NumPy
fuse.disable()
```

- **What is fused** is an expression of additions, subtractions, multiplications, divisions, maxima, minima, square roots, comparisons under `np.where` and a few more on float64 arrays, whose intermediates nothing else reads. Everything else is computed by NumPy's own calls, as before.
- **An expression is a program of kernel calls,** run a block of 4,096 elements at a time, so that what one kernel gives the next stays in the cache and only the result is written to memory.
- **Rows of a few elements are walked as one run,** where NumPy's loop is called for each row: `table - row` on rows of 2 to 15 elements is one kernel call, 1.8 to 9 times as fast as NumPy's; a slice with elements between its rows is walked straight through them; and one operation of arithmetic on such rows, at any pitch or with a value for each row, is a loop of the module's own, 1.1 to 5.4 times NumPy's call. A product of three arrays stored into short rows of a result that steps over elements, `Ai[..., ip, kr] = a * b * mask[...]`, is such a loop too, 1.5 times NumPy's.
- **`hotkernels.install()` is in effect with it.** `fuse.enable()` installs the package's single-operation kernels unless told not to (`install=False`), so a program gains from both, and `fuse.disable()` takes them out again if it was what installed them.
- **Eighty kernels**, NEON for ARM64, in [kernels/](kernels/README.md): fourteen written by hand, and sixty-six that `make_kernel.py` wrote from the names of their shapes, the single operations among them. On another architecture each is built from its C reference.
- **A write is part of its expression:** `u[1:-1, 1:-1] = e` is computed straight into that part of `u`.
- **Small arrays are not planned for, and not written down either.** With fusion on, a call that gives fewer than 32,768 elements on arrays that have their values is made at once, and so is a call of any size that no kernel could make (a comparison, a matrix product, `np.exp`, anything complex or float32), and a write when nothing is pending. `fuse.MIN_ELEMENTS` and `fuse.AT_ONCE_BELOW` are the two limits; `array.at_once_below(n)` is the second by itself. A call made at once costs 15 to 115 ns over NumPy's own, and needs the C core.

Four tools, needing `--group apps --group workloads` for the programs that call a library:

```bash
uv run --group apps --group workloads python -m prototypes.lazy.programs.mine --size large          # which shapes occur, and what they cost
uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --size large   # each program: NumPy, and with hotkernels installed eagerly, lazily and fused
uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --plain        # the lazy runs with nothing installed as well
uv run --group apps --group workloads python -m prototypes.lazy.programs.limit --size large         # each program: what no fusion could remove, what is left, and what of the fused run could still go
uv run python -m prototypes.lazy.make_kernel "sub(V0,mul(S0,V1))"                                   # write the kernel of a shape
```

The timing tools run the lazy layer with `hotkernels.install()` in effect, since that is how the package is used, and measure against the program on plain NumPy.

Run them on a machine that is doing nothing else.

## Testing it

```bash
uv run --group apps --group workloads --group dev pytest tests/test_lazy_prototype.py tests/test_lazy_graphs.py tests/test_lazy_building.py tests/test_lazy_fusion.py tests/test_lazy_at_once.py
```

Each test holds one rule of the design to what the real `numpy` module gives for the same lines. Those of `test_lazy_building.py` hold what the layer remembers while it writes a graph to what it would work out from nothing, and what the C core answers for a first call to what the rules answer. Those of `test_lazy_fusion.py` hold every fused result, warning and error to NumPy's, and run the programs of the corpus with fusion on; they build the fused kernels for the run and are skipped where clang is missing. Those of `test_lazy_at_once.py` hold writes made in place, calls made at once and random programs of views and writes to NumPy's arrays, and run the corpus as the timing tool runs it. Every test runs twice, in Python alone and with the C core, which is built for the run; the second is skipped where clang is missing.
