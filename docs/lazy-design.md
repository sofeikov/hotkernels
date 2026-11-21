# A lazy NumPy layer: `hotkernels.get_lazy()`

This is a proposal for review. Nothing here is implemented yet.

It covers the smallest build that is worth having: NumPy code whose operations are written down as a graph instead of being computed, and computed when the user asks. **It makes nothing faster and is not meant to.** Speed is out of scope here. The graph is what fusion needs, and fusion is what a kernel for a whole expression needs; [What the graph is for](#what-the-graph-is-for) says how the three connect, without planning the second and third.

[core-design.md](core-design.md#later) left fusion for later because it "cannot be delivered transparently", and [gradient-step-findings.md](gradient-step-findings.md#can-a-hook-fuse-the-statement) found why: no hook sees an expression whole, and a result that has no values yet is unsound for a real `ndarray`. The delivery design named the other road in its first table: an array type of the project's own reaches anything, in code that has opted in. This document designs that type, and keeps the opting in to one line.

## Summary

- **One line turns it on, one call gets the values.** `np = hotkernels.get_lazy()` stands where `import numpy as np` stood, and `x.gather()` returns an ordinary `numpy.ndarray`.
- **The code in between stays NumPy code.** Operators, `np.exp(x)`, `x.mean(axis=0)`, `X.T @ r`, slicing. No new functions to learn.
- **A lazy array is a type of its own, not an `ndarray`.** NumPy already hands every call that meets such a type to it, through two protocols it provides for the purpose. A probe found that this reaches operators, ufuncs, functions, methods of ufuncs and `out=`, also when the code calls the real `numpy` module. See [What was verified](#what-was-verified).
- **A narrow set of operations is deferred; all the rest of NumPy still works.** Anything outside the set is computed at once by NumPy, on gathered operands, and its result is a lazy array that already has its values. So nothing a program calls is refused, and the set can grow one operation at a time.
- **The contract is one sentence:** what a program gets from `gather()` is what the same lines give with `import numpy as np`, in type, element type, shape and bits. It is checked by running programs both ways.
- **Shape and type errors are raised on the line that caused them,** with NumPy's own message, because shapes and types are worked out when the line runs. Only the values wait.
- **The first build computes the graph one node at a time, each with the NumPy call it stands for.** No kernel, no C, no fusion. The result is NumPy's by construction, and that is the point of the first build: the graph, its rules and its check exist before anything is allowed to compute differently.
- **A throwaway prototype of about 430 lines ran two programs under these rules:** a linear regression by gradient descent and a transformer block's forward pass. Both give NumPy's results bit for bit.
- **Eight decisions are asked for,** at the [end](#decisions-wanted).

## What the user writes

**A linear regression.** The only changes from eager NumPy are the first line and the last.

```python
import hotkernels
np = hotkernels.get_lazy()                 # where `import numpy as np` stood

X, y = np.asarray(features), np.asarray(targets)
X = (X - X.mean(axis=0)) / X.std(axis=0)   # nothing is computed yet
w = np.zeros(X.shape[1])                   # shapes are known without computing

for step in range(200):
    residual = X @ w - y
    loss = (residual ** 2).mean()
    w = w - 0.05 * (2 / len(y)) * (X.T @ residual)
    if step % 20 == 0:
        print(f"step {step}: loss {loss:.4f}")   # a value is needed, so it is computed here

weights = w.gather()                       # a numpy.ndarray
```

**Attention, with the real `numpy` module.** Code that already exists, and code in other files, need not change at all: one lazy input is enough, because NumPy passes every call that meets a lazy array to it.

```python
import numpy as np
import hotkernels

lazy = hotkernels.get_lazy()

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def attention(q, k, v, mask):
    return softmax(q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1]) + mask) @ v

out = attention(lazy(q), k, v, mask)       # q is lazy; k, v and mask are plain ndarrays
out.shape                                  # (4, 64, 24), known now
out.gather()                               # computed now
```

**Values on demand.** Where Python needs a value, it gets one, and nothing has to be gathered by hand:

```python
if loss < 1e-6: ...        # computes loss
float(loss); f"{loss:.3f}" # compute loss
plt.plot(losses)           # matplotlib asks NumPy for an array: computed
loss                       # <lazy float64 (), pending>: looking does not compute
```

**Several at once, and a loop that keeps its state lazy.**

```python
w_now, b_now = hotkernels.gather(w, b)     # two results, one walk of the graph

for batch in batches:
    w = w - lr * gradient(w, batch)
    w.gather()                             # compute this step now; w stays a lazy array
```

**Looking at the graph.** `loss.explain()` prints what is pending, one operation per line, with the line of the program that wrote it. From the prototype, for the loss of the first step above:

```text
mean      float64 ()            regression.py:15
  power     float64 (20000,)      regression.py:15
    subtract  float64 (20000,)      regression.py:14
      matmul    float64 (20000,)      regression.py:14
        divide    float64 (20000, 30)   regression.py:10
          subtract  float64 (20000, 30)   regression.py:10
            array     float64 (20000, 30)
            mean      float64 (30,)         regression.py:10
              array     float64 (20000, 30)
          std       float64 (30,)         regression.py:10
            array     float64 (20000, 30)
        array     float64 (30,)
      array     float64 (20000,)
```

### The whole interface

| Call | Does |
|---|---|
| `hotkernels.get_lazy()` | Returns a stand-in for the `numpy` module: the same names, and every array it makes is lazy. Called on an array, `lazy(x)`, it wraps that array. It changes nothing in NumPy |
| `x.gather()` | Computes `x` and returns what NumPy would have returned for it: an `ndarray`, or a NumPy scalar where NumPy gives one. `x` is still a lazy array afterwards, now computed |
| `hotkernels.gather(a, b, ...)` | The same for several arrays, as a tuple. With no argument it computes everything pending |
| `x.explain()` | The pending graph of `x` as text. Computes nothing |

Rules:

- **Importing does nothing, and neither does `get_lazy()`** beyond building the stand-in. NumPy is not changed. Only code that holds a lazy array behaves differently.
- **It does not need `install()`,** a kernel or the binding. Where `install()` does nothing because the machine is not supported, `get_lazy()` works the same. With `install()` in effect, the NumPy calls the layer makes are served by kernels like any others.
- **It never refuses a NumPy call.** A call outside the deferred set is made at once.
- **Every error a user sees for a wrong shape or type is NumPy's,** raised on the line that wrote the operation.

## What a lazy array is

### Known at once, computed later

| Known when the line runs, without computing | Computed later |
|---|---|
| `x.shape`, `x.dtype`, `x.ndim`, `x.size`, `len(x)` | The values |
| Whether the shapes fit and the types combine: NumPy's exception, on that line | Warnings and errors that depend on values: division by zero, overflow |

### Three kinds of operation

| Kind | What happens | Which |
|---|---|---|
| Deferred | A node joins the graph. Nothing is computed | The set in the next table |
| Computed at once | The lazy operands are gathered, NumPy makes the call, and the result is a lazy array that has its values | Everything else in NumPy: `np.linalg.solve`, `np.argsort`, `x[mask]`, `x.cumsum()`, any method or function not listed |
| A write | Everything pending is computed first, in the order it was written, and then NumPy makes the call | `x[i] = v`, `x += y`, `out=`, `np.copyto`, `a += x` with `a` an `ndarray`, and NumPy asking a lazy array for its values |

The third row is what keeps NumPy's meaning when memory is changed in place. Once everything pending is computed, the program is in the state the eager program would be in, and the write is NumPy's own. It is never more work than eager NumPy would have done, since eager NumPy had computed all of it already.

### The deferred set

Narrow in kinds, not in count. The first build computes each node with NumPy's own call, so admitting an operation costs a shape rule and nothing else.

| Kind | Deferred | Not deferred |
|---|---|---|
| Element by element | Every ufunc with one output and no core dimensions (`np.add`, `np.exp`, `np.tanh`, `np.maximum`, `np.less`, ...), on arrays and numbers given by position; the operators that call them: `+ - * / // % **`, unary `-`, `abs`, the six comparisons, `& \| ^ ~`; `np.where(c, a, b)` | A call with `where=`, `dtype=` or `casting=` |
| Matrix product | `@`, `np.matmul` | `np.dot`, `np.einsum`, `np.linalg` |
| Reductions | `sum`, `mean`, `max`, `min`, `prod`, `var`, `std`, as functions and as methods, with `axis` and `keepdims` | Any other argument; `argmax`, `cumsum`, `any` |
| Views | `x.T`, `np.transpose`, `np.swapaxes`, `reshape`, `np.expand_dims`, `np.squeeze`, and indexing with integers, slices, `None` and `...` | Indexing with an array or a mask: the shape of `x[mask]` depends on values, so it is computed at once |

Never deferred, whatever the operation: an operand with a dimension of length zero, an element type that is not boolean, integer, floating or complex, and an operand that is neither a lazy array, an exact `ndarray`, a NumPy scalar nor a Python number.

### When the values are computed

| Event | What is computed |
|---|---|
| `x.gather()`, `hotkernels.gather(a, b)` | Those arrays and what they need |
| `float(x)`, `int(x)`, `bool(x)` as in `if x > 0:`, `f"{x:.3f}"`, `x.item()`, `x.tolist()`, iterating over `x` | That array |
| An operation outside the deferred set | Its operands |
| A write, and `hotkernels.gather()` with no argument | Everything pending |
| `np.asarray(x)`, and any library that asks NumPy to turn `x` into an array | Everything pending. The layer cannot see why NumPy asks, and `a[i] = x` on an `ndarray` asks the same way |
| More than 100,000 nodes pending | Everything pending, so a long loop that never gathers cannot hold an unbounded graph |
| `x.shape`, `x.dtype`, `len(x)`, `print(x)`, `x.explain()` | Nothing |

A node is computed once. Its value is kept while the program holds the lazy array, and an intermediate nobody holds is freed as soon as its last consumer has been computed. Pending work that the program drops without asking for it is never computed.

### What differs from NumPy

A user has to know five things.

1. **A pending expression reads an `ndarray` when it is gathered, not when it is written down.** `y = lazy(a) * 2`, then `a[0] = 5` on the `ndarray` itself, then `y.gather()`: the result has the 5. Writes made through a lazy array are safe, by the rule above; a write made directly to an `ndarray` that a pending expression has read is not seen by the layer. Gather before such a write. `np.array(a)` copies, as it does in NumPy, and `np.asarray(a)` and `lazy(a)` do not.
2. **A lazy array is not an `ndarray`.** `isinstance(x, np.ndarray)` is False, and code that needs the memory itself, a Cython function taking `double[:]` for one, has to be given `x.gather()`.
3. **Warnings and errors that depend on values come at the gather.** They are the same warnings, issued once, and each names the line that wrote the operation, not the line that gathered. `np.errstate` is remembered per operation, so `with np.errstate(divide="ignore"): r = a / b` stays silent wherever `r` is gathered. A `warnings.catch_warnings()` block around the line does not reach the gather.
4. **An operation that failed stays failed.** Under `np.errstate(over="raise")` the exception is raised by the first gather that needs the operation, with a note naming the line that wrote it, and again by any later gather that needs it. A write that computes everything pending raises it once, and not again for what depended on it.
5. **The work happens at the gather,** so that is where time passes and memory is taken, and what `KeyboardInterrupt` interrupts.

## What was verified

Two throwaway scripts, not in the repository: a probe of NumPy's protocols, and a prototype of the layer in about 430 lines of Python. Apple M1 Pro, macOS, CPython 3.12.7, NumPy 2.5.5.

### What reaches a type that is not an `ndarray`

A class with `__array_ufunc__`, `__array_function__`, `__array_priority__` and the operators, given to the real `numpy` module beside an `ndarray` `a` and a matrix `m`:

| Spelling | Arrives as |
|---|---|
| `np.exp(x)`, `np.sqrt(x)`, `np.tanh(x)`, `np.maximum(x, 0)` | `__array_ufunc__` |
| `a + x`, `np.add(a, x)`, `np.float64(2) + x` | `__array_ufunc__`: the `ndarray`'s own operator hands the call over |
| `x + a`, `2.0 + x` | The type's own `__add__` and `__radd__` |
| `m @ x`, `np.matmul(m, x)` | `__array_ufunc__`. `x @ m` is the type's `__matmul__` |
| `np.where`, `np.sum`, `np.mean`, `np.clip`, `np.dot`, `np.transpose`, `np.swapaxes`, `np.concatenate([a, x])`, `np.argmax`, `np.einsum`, `np.linalg.norm`, `np.linalg.solve`, `np.zeros_like` | `__array_function__`, with the function and its arguments as written |
| `np.add.reduce(x)` | `__array_ufunc__`, method `reduce` |
| `a += x`, `np.multiply(a, 2, out=x)` | `__array_ufunc__` with `out`: the write is seen before it happens |
| `np.asarray(x)` | `__array__` |

Not reached, because no lazy array is among the arguments: `np.zeros(3)`, a random generator's methods, a file loader. That is what the stand-in for the module is for.

### The prototype

It defers the set above, computes everything else at once, and evaluates node by node. Both programs were written once as functions of `np` and run with the `numpy` module and with the stand-in. The two examples under [What the user writes](#what-the-user-writes) were also run exactly as printed there, with the same result.

| Program | Deferred | Computed at once | Result against eager NumPy |
|---|---|---|---|
| Linear regression: standardised features, 50 gradient steps with an L2 penalty, no gather inside the loop, on 20,000 × 30 `float64` | 1,004 operations | 0 | Weights, bias and all 50 losses identical, bit for bit |
| Transformer block: layer norm, four-head attention with a causal mask, softmax, GELU, two residuals, `float32` parameters, 64 × 96 | 112 | 0 | Identical. Also identical when the block calls the real `numpy` module and only its input is lazy |

Behaviours checked one by one, each matching the rule this document states:

- **Errors on the line.** `a + b` with shapes (6,) and (4,) raises NumPy's "operands could not be broadcast together"; a matrix product with mismatched inner sizes raises NumPy's `matmul` message; `np.sum(x, axis=3)` on one dimension raises `AxisError`; `int8_array + 300` raises NumPy's `OverflowError`.
- **Promotion is NumPy's.** `float32 * 2.0` stays `float32` and `float32 * np.float64(2)` gives `float64`; `int8 + 1` stays `int8`; `sum` of `int8` gives `int64`; `mean` of `int32` gives `float64`. The transformer block comes out `float64` under both, because `np.sqrt(2 / np.pi)` is an `np.float64`.
- **Warnings.** `a / zeros` gathered later warns "divide by zero encountered in divide" with the file and line of the division. Written under `np.errstate(divide="ignore")` and gathered outside it: no warning. Under `errstate(divide="raise")`: `FloatingPointError` at the gather, with the note.
- **Writes.** `a += x` on an `ndarray` leaves `a` the same `ndarray` and updated, and an expression written before it has the old values. `x[1:3]` assigned through, as a view, changes `x`. `w -= step` on a lazy array keeps the object, so another name for it sees the change. `s += 1` on the result of a `sum` gives a new scalar, as NumPy does. `int_array += 0.5` raises NumPy's casting error.
- **Types.** `gather()` of a `sum` is an `np.float64`, not an array.

Building a node costs Python time: 0.3 to 0.5 µs in the bare probe, where NumPy adds two arrays of 16 elements in 0.26 µs, on a busy machine (load average 8 to 10). The first build is therefore slower than NumPy on small arrays. That is accepted here.

## Design

### Where it lives

A subpackage of pure Python, `hotkernels/src/hotkernels/lazy/`, and two names exported from the package: `get_lazy` and `gather`.

| Module | Holds |
|---|---|
| `graph.py` | The node, the list of pending nodes, the lock |
| `array.py` | The lazy array: operators, the two protocols, the ways out |
| `rules.py` | Which calls are deferred, and the shape of each |
| `evaluate.py` | Computing a node, a set of nodes, everything pending |
| `namespace.py` | The stand-in for the `numpy` module |
| `explain.py`, `templates/explain.txt.j2` | The text of a pending graph |

### The node

| Field | Meaning |
|---|---|
| Operation | The NumPy callable the node stands for: a ufunc, a function such as `np.sum`, or indexing |
| Operands | Lazy arrays, `ndarray`s held by reference, and Python numbers kept exactly as written, so that NumPy's rule for them (a Python `2.0` does not widen a `float32` array) is applied by NumPy |
| Shape, element type | Worked out when the node is made |
| Value | Empty until computed. Then the operands are dropped |
| Written at | File and line of the user's code |
| Error state | The `np.errstate` in effect when it was written, when that is not the default |
| Number | A count that rises with every node. Operands are always older than their consumer, so ascending order is a valid order of evaluation, and it is the order the eager program ran in |
| Failure | The exception, if computing it raised |

A lazy array is a handle on one node. A leaf is a node born with its value.

### Shape and element type without computing

- **The shape comes from a rule,** one per kind: NumPy's `np.broadcast_shapes` for element-by-element operations; the matrix product's rule; the axes removed for a reduction. For a view, the operation is made on a stand-in that has the operand's shape and no memory (`np.broadcast_to` of one zero), which NumPy answers without touching data.
- **The element type comes from NumPy itself:** the real call is made on arrays with one element per dimension, with the Python numbers as written. That costs a microsecond or two and cannot disagree with NumPy about promotion.
- **An error is NumPy's.** When the rule finds the shapes do not fit, the real call is made on the stand-ins without memory, and NumPy raises its own exception with its own message before it reads anything. If NumPy does not raise, the rule was wrong to doubt, and the call is computed at once.
- **When in doubt, do not defer.** A call the rules do not recognise in every detail is computed at once. The layer never guesses a shape or a type.

### Computing

- `gather` collects the nodes its arrays need that have no value, without recursion, and computes them in ascending number: the node's own NumPy call, on its operands' values, under its error state.
- Floating-point events are collected through `np.errstate(call=...)` instead of being left to warn from inside the layer, and each is issued as the `RuntimeWarning` NumPy words, at the file and line the node was written at. Modes other than `warn` are left to NumPy.
- An exception gets a note naming that line (`add_note`), is kept on the node, and is raised.
- The first build therefore makes exactly the calls the eager program makes, on the same operands. Whatever computes a graph differently later is held to the same results by the check below.

### Pending work and writes

Every node without a value is in one list, held weakly. A write computes that list in ascending number and then lets NumPy write. In-place operators on a lazy array are handed to NumPy's in-place operator on the computed value: if NumPy returns the same array, the lazy array is returned unchanged in identity; if it returns a new scalar, a new lazy array is.

One lock guards the list and every evaluation. Lazy arrays can be used from several threads; their work is done one gather at a time. Free-threaded Python is not supported, as for the rest of the package.

### The stand-in for the module

`get_lazy()` returns an object that looks names up in `numpy`:

- A type or a constant (`np.float32`, `np.pi`, `np.newaxis`) is returned as it is.
- A submodule (`np.linalg`, `np.random`) is returned as another stand-in.
- A function is returned wrapped: `ndarray` arguments become leaves, the real function is called, NumPy's protocols route it to the lazy array, and an `ndarray` that comes back is made a leaf. So `np.zeros(3)` is lazy, and `np.exp(x)` is the same call it is with the real module.

An `ndarray` from somewhere else, a generator's `standard_normal` or `pandas`, is a plain `ndarray` until it meets a lazy array or is wrapped.

### Checking it

A lazy layer is trusted for the same reason a kernel is: a check that does not depend on reading the code. `uv run delivery lazy-check` runs:

1. **Programs both ways.** The two examples, and the applications in `delivery/apps/` that need only NumPy (`mlp_train`, `min_char_rnn`, `veros_ocean`, `particle_filter`), each with its module's `np` replaced by the stand-in. Results, exceptions and warnings are compared as `delivery check` compares them.
2. **Random programs.** Sequences of operations from the deferred set and outside it, over random shapes, element types and special values, with gathers and writes at random points, run eagerly and lazily. Compared: type, element type, shape, bits, the exception's type and message, the warnings and their lines, and the final state of every `ndarray` the program was given.
3. **Each supported NumPy,** in CI, since the protocols are NumPy's.

## What the graph is for

Not part of this build, and no gain is claimed. It is here so that the first build is shaped for what comes after.

- **A fusible group can be read off the graph:** connected element-by-element nodes of one output shape, optionally ending in a reduction. In the regression above, `(X @ w - y) ** 2` then `mean` is one group after the product; in the transformer block the eight operations of GELU are one. `explain()` is where such groups would first be shown.
- **A group can be computed three ways,** and the graph does not care which: node by node with NumPy, as now; with NumPy's own loops a cache-sized block at a time, which removes passes over memory and keeps NumPy's bits; or with one kernel for the whole expression.
- **A fused kernel fits the existing contract unchanged.** Its `spec.py` already declares its oracle as a NumPy expression (`np.where(c, a, b)` for `where_f64`). Run that oracle on lazy arrays and the result is the graph the kernel computes: the specification is its own pattern, and the judge checks a fused kernel as it checks any other.
- **The graph carries facts a hook has to read memory for.** A mask that came from a comparison holds only 0 and 1; a temporary the graph owns overlaps nothing. The binding scans for both today.
- **The graph says which chains a program runs,** with their sizes: the "way to identify which chains are worth fusing" that [core-design.md](core-design.md#later) says fusion first needs.

## Considered and not proposed

| Option | Why not |
|---|---|
| Make real `ndarray`s lazy, process-wide, as `install()` is process-wide | Unsound: an `ndarray`'s memory is public C API, and anything may read it without calling something a hook could see ([gradient-step-findings.md](gradient-step-findings.md#can-a-hook-fuse-the-statement)). Laziness has to be a type of its own, in code that asked for it |
| A subclass of `ndarray` | The same: a subclass has a data pointer that must hold values at all times |
| Copy every `ndarray` a lazy expression reads | It would remove the first of the five differences. It costs a copy of every input each time it is met, and a loop that reads a large plain `ndarray` in every step copies it in every step. Offered as [decision 3](#decisions-wanted) |
| Refuse to give a value without `gather()`: raise on `float(x)`, `if x:`, `np.asarray(x)` | It breaks plotting, printing and every library call, for the sake of making computation visible. `explain()` and a later report make it visible without breaking them. Offered as decision 2 |
| Defer operations whose shape depends on values (`x[mask]`, `np.unique`) | Every lazy array would lose its shape, and shape errors their line. Computing these at once keeps the rule that shape is always known |
| A decorator, `@hotkernels.lazy`, that wraps a function's arguments and gathers its results | Sugar over this design, and a good one for a hot function. Left until the layer under it exists |
| Dask, JAX or numexpr | Each has semantics of its own: chunked and parallel arrays, another array library with its own type rules, expressions as strings. The contract here is NumPy's results bit for bit, so that kernels judged against NumPy can compute the graph |
| The array API standard (`__array_namespace__`) as the way in | It serves libraries written for it. Existing NumPy code calls `np.exp(x)`, which the two protocols above serve. It can be added beside them |

## Exit criterion

1. **Two programs run with one changed line and one `gather()`:** `examples/lazy_regression.py` and `examples/lazy_attention.py`, with results identical to eager NumPy, and no operation in either computed at once.
2. **`uv run delivery lazy-check` passes** in all three of its parts.
3. **Shape and type errors are NumPy's and are raised on the line,** and value-dependent warnings name the line that wrote the operation, for every deferred kind.
4. **`get_lazy()` works with and without `install()`,** and `delivery check` and NumPy's test suite are unaffected by it, since it changes nothing in NumPy.
5. **The user-facing pages say what a user must know:** `docsite/pages/start.md` and `hotkernels/README.md` carry the examples and the five differences.

No criterion is about speed.

## Decisions wanted

| # | Question | Recommended | The alternative |
|---|---|---|---|
| 1 | What `get_lazy()` returns | A stand-in for the `numpy` module, which also wraps an array when called: both spellings above from one object | A wrapping function only, keeping `import numpy as np` everywhere. A process-wide switch is not offered: it is unsound |
| 2 | A value is needed and `gather()` was not called (`float(x)`, `if x > 0:`, `np.asarray(x)`) | Compute it silently | Raise and ask for `gather()`, perhaps as `get_lazy(strict=True)` |
| 3 | An `ndarray` read by a pending expression | Held by reference; the user gathers before writing to it directly | Copied when first met: no rule to remember, one copy per input |
| 4 | `x[i] = v` and `x += y` on a lazy array | Writes with NumPy's meaning, after everything pending is computed | Refused: lazy arrays immutable, the user writes to gathered arrays |
| 5 | `print(x)` | Describes a pending array without computing; shows the values of a computed one | Always computes, as NumPy would show values |
| 6 | After `x.gather()` | `x` stays a lazy array holding its values; `gather()` returns that array itself, not a copy | Return a copy, so that writing to the result cannot change `x` |
| 7 | The first deferred set | The table above, with linear regression and a transformer block as the two programs it must run whole | A narrower set: element by element and matrix product only |
| 8 | Its home | Inside the `hotkernels` package, as `hotkernels.get_lazy()`, in pure Python and independent of `install()` | A package of its own, since it depends on nothing in hotkernels |

## Not examined

- **Other array types as operands.** A masked array, an `np.matrix` or a pandas Series beside a lazy array is computed at once by this design; which of the two types NumPy lets decide such a call was not probed.
- **Pickling and copying a lazy array.** Presumably a gather; not tried.
- **NumPy releases other than 2.5.5.** Both protocols are years old and documented, and the supported range starts at 2.3; the check runs on each.
- **A program of real size.** The two programs are a few dozen lines each. The applications in `delivery/apps/` are the next test and are part of the check.
