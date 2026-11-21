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

- Defers a narrow set of NumPy operations into a graph: element-by-element operations, the matrix product, the common reductions, and views.
- Makes every other NumPy call at once, on gathered operands, and returns a lazy array that has its values.
- Gives values through `gather()` only. Anything else that would need them (`float(x)`, `if x > 0:`, `np.asarray(x)`) raises `NotGathered`.
- Computes the graph one node at a time, each with the call the user wrote, spelled the same way, so results are NumPy's and nothing is faster. With `hotkernels.install()` in effect those calls reach the kernels as the eager program's do.
- Patches nothing: `get_lazy()` returns an object, and the `numpy` module and other modules' imports are as they were. An `ndarray` that exists enters with `np.asarray(a)`.

## Reading it

| File | Holds |
|---|---|
| [graph.py](graph.py) | The node, and the list of nodes not computed yet |
| [rules.py](rules.py) | Which calls are deferred, and their shape and element type without computing |
| [evaluate.py](evaluate.py) | Computing nodes |
| [array.py](array.py) | `LazyArray`: operators, NumPy's two protocols, the refusals, `gather()` |
| [namespace.py](namespace.py) | The stand-in for the `numpy` module |
| [explain.py](explain.py) | The text of a pending graph |
| [programs/](programs) | A linear regression and a transformer block, each run both ways by the tests |

## Testing it

```bash
uv run --group dev pytest tests/test_lazy_prototype.py
```

Each test holds one rule of the design to what the real `numpy` module gives for the same lines.
