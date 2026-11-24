# A lazy NumPy layer: `hotkernels.get_lazy()`

This is a proposal for review. Nothing here is in the package. A prototype that follows this document is in [`prototypes/lazy/`](../prototypes/lazy), with tests in [`tests/test_lazy_prototype.py`](../tests/test_lazy_prototype.py); it is there to be read and argued with, and [What was verified](#what-was-verified) says what it showed. Getting the graph out and drawing it have documents of their own: [lazy-recording-design.md](lazy-recording-design.md) and [lazy-graph-page-design.md](lazy-graph-page-design.md).

The document covers the smallest build that is worth having: NumPy code whose operations are written down as a graph instead of being computed, and computed when the user asks. **It makes nothing faster and is not meant to.** Speed is out of scope here. The graph is what fusion needs, and fusion is what a kernel for a whole expression needs; [What the graph is for](#what-the-graph-is-for) says how the three connect, without planning the second and third.

[core-design.md](core-design.md#later) left fusion for later because it "cannot be delivered transparently", and [gradient-step-findings.md](gradient-step-findings.md#can-a-hook-fuse-the-statement) found why: no hook sees an expression whole, and a result that has no values yet is unsound for a real `ndarray`. The delivery design named the other road in its first table: an array type of the project's own reaches anything, in code that has opted in. This document designs that type, and keeps the opting in to one line.

## Summary

- **One line turns it on, one call gets the values.** `np = hotkernels.get_lazy()` stands where `import numpy as np` stood, and `x.gather()` returns an ordinary `numpy.ndarray`.
- **Nothing is patched.** `get_lazy()` returns an object, and the line binds it to a name in the module that wrote it. The `numpy` module, `sys.modules` and every other module's import are as they were, so it does not matter what was imported before or in which order. See [One line, and nothing is patched](#one-line-and-nothing-is-patched).
- **The code in between stays NumPy code.** Operators, `np.exp(x)`, `x.mean(axis=0)`, `X.T @ r`, slicing. No new functions to learn.
- **A lazy array is a type of its own, `hotkernels.LazyArray`, not an `ndarray`.** Element types do not change: `x.dtype` is NumPy's. [Types](#types-what-changes-for-the-user) says where the new type shows in a program and what a user has to change: nothing to run, one thing if their code asks `isinstance(x, np.ndarray)`.
- **NumPy itself delivers the calls.** It hands every call that meets such a type to it, through two protocols it provides for the purpose. That reaches operators, ufuncs, functions, methods of ufuncs and `out=`, also in another file that imports the real `numpy` module.
- **A narrow set of operations is deferred; all the rest of NumPy still works.** Anything outside the set is computed at once by NumPy, on gathered operands, and its result is a lazy array that already has its values. The log says so.
- **A program runs without being rewritten.** `gather()` is the way to ask for values, and nothing else is refused: `float(x)`, `if x > 0:`, `np.asarray(x)` and a library that wants an `ndarray` each get what NumPy would give, computed on the spot. Every such place can be found afterwards, in the log, in a ranked summary and on the page of the program's code; none of them stops the program. See [When a value is needed](#when-a-value-is-needed) and [Why it computes instead of raising](#why-it-computes-instead-of-raising).
- **A loop does not make the layer compute.** `w -= step`, `x[i] = v` and `out=x` on an array the layer made are nodes of the graph like any other operation. Fifty steps of a training loop are one graph, computed when a value is needed. See [Writes](#writes).
- **Strict is a switch.** `hotkernels.get_lazy(strict=True)` gives arrays that raise `NotGathered` in those places instead. It is for a hot path that must not compute by accident, and for the layer's own tests ([Strict arrays](#strict-arrays)).
- **The contract is one sentence:** what a program gets from `gather()` is what the same lines give with `import numpy as np`, in type, element type, shape and bits. It is checked by running programs both ways.
- **Shape and type errors are raised on the line that caused them,** with NumPy's own message, because shapes and types are worked out when the line runs. Only the values wait, and with them the one length that is a value: how many elements a mask selects.
- **The first build computes the graph one node at a time, each with the call the user wrote.** No kernel, no C, no fusion. The result is NumPy's by construction, and that is the point of the first build: the graph, its rules and its check exist before anything is allowed to compute differently.
- **With `hotkernels.install()` in effect those calls reach the kernels,** as the eager program's calls do, because they are the same calls spelled the same way. See [With `hotkernels.install()`](#with-hotkernelsinstall).
- **The graph can be looked at:** as text, as an object, and as a page in the browser, with a mode that records the graphs a block of code produced. See [Debugging tools](#debugging-tools).
- **The prototype runs two programs under these rules,** a linear regression by gradient descent and a transformer block's forward pass, with NumPy's results bit for bit.
- **What is not settled is listed** under [Open questions](#open-questions).

## What the user writes

### What an array is

The first thing to try: make an array and look at it.

```python
>>> import hotkernels
>>> np = hotkernels.get_lazy()
>>> x = np.arange(6.0)
>>> type(x)
<class 'hotkernels.LazyArray'>
>>> x.dtype, x.shape
(dtype('float64'), (6,))
>>> y = x * 2 + 1
>>> print(y)
LazyArray(float64, shape=(6,), pending)
>>> y.gather()
array([ 1.,  3.,  5.,  7.,  9., 11.])
>>> type(y.gather())
<class 'numpy.ndarray'>
>>> print(y)
LazyArray(float64, shape=(6,), computed)
```

- **`type(x)` is `hotkernels.LazyArray`,** whatever the array holds. There is one class.
- **`x.dtype` and `x.shape` are NumPy's own answers,** known without computing. The exception is what a mask selects, `x[mask]`, whose length is not known until the mask is ([A length that depends on values](#a-length-that-depends-on-values)).
- **`print(x)` describes the array and never computes or shows values:** the class, the element type, the shape, and whether it is pending, computed or failed. Values are what `gather()` returns, and the `ndarray` it returns prints as an `ndarray` does.
- **`gather()` gives exactly what NumPy would have given.** For a result NumPy returns as a scalar, that is the scalar: `type(y.sum())` is `LazyArray`, and `type(y.sum().gather())` is `numpy.float64`.

The prototype prints `prototypes.lazy.LazyArray`, since that is where it lives.

### A linear regression

The one change from eager NumPy is the first line. The `gather()` at the end is how the program says where it wants its values; the loss printed inside the loop is computed without one.

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
        print(f"step {step}: loss {loss:.4f}")   # a value is needed: computed here

weights = w.gather()                       # a numpy.ndarray
```

With the logger `hotkernels.lazy` at `DEBUG`, the run says where a graph was computed early. The line runs ten times and is reported once:

```text
needed: Formatting with '.4f' at regression.py:18
```

### Code in another file, unchanged

A lazy array stays lazy in code that knows nothing of the layer. `model.py` imports the real `numpy` module and is not changed; NumPy passes each of its calls to the lazy array the call meets.

```python
# model.py: as it was
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def attention(q, k, v, mask):
    return softmax(q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1]) + mask) @ v
```

```python
# main.py
import hotkernels
from model import attention

np = hotkernels.get_lazy()
out = attention(np.asarray(q), k, v, mask)   # q enters lazy; k, v and mask are plain ndarrays
out.shape                                    # (4, 64, 24), known now
out.gather()                                 # computed now
```

Only `q` was made lazy. `k`, `v` and `mask` join the graph as inputs when an operation meets them, and all nine operations of `attention` are deferred.

### One line, and nothing is patched

`np = hotkernels.get_lazy()` is an ordinary assignment. `get_lazy()` returns an object that answers the names the `numpy` module has, and the line binds it to a name in the module that wrote the line.

- **Nothing in NumPy or in Python's import system is changed.** `sys.modules["numpy"]` is the real module, `numpy.zeros` is the real function, and every other module's `import numpy` gives what it always gave. The stand-in is not an import hook and replaces no import. Patching an import has to happen before the first import of the module, which a real program cannot promise once something imports something that imports NumPy; here there is no import to get in ahead of, and the line works the same wherever it stands and whatever was imported before it.
- **Arrays enter in three ways.** A function of the stand-in that makes an array makes a lazy one: `np.zeros(3)`, `np.arange(10)`, `np.array(data)`. An `ndarray` that exists is taken in with `np.asarray(a)`, by reference and without a copy. And a plain `ndarray` that meets a lazy array in an operation joins the graph as an input.
- **It covers the module that wrote the line, and every function a lazy array is handed to,** in whatever file. It does not cover arrays that other modules make with their own `numpy`: a helper's `np.zeros(n)`, a random generator's `standard_normal`, a file loader, a pandas column. Those are plain `ndarray`s, computed eagerly, until they meet a lazy array or are taken in.
- **A part of the program that stays eager costs graph, not correctness.** What it computes is NumPy's, at once; it is only missing from the graph. One case is different: code that writes directly into an `ndarray` after a pending expression has read it ([What differs from NumPy](#what-differs-from-numpy), item 3, and the first of the [open questions](#open-questions)).
- **What the line costs is the editor.** After it, an editor or a type checker no longer knows that `np` is NumPy: no completion on `np.`, no checking of calls. The program is not affected. [Other ways in](#other-ways-in) has an import spelling that could give that back.

### When a value is needed

`gather()` is the way to ask for values. A program that needs one without asking gets it all the same, as the line would give it with NumPy:

```python
>>> total = y.sum()
>>> float(total)
36.0
>>> if total > 30: print("over thirty")
over thirty
>>> f"{total:.1f}"
'36.0'
>>> numpy.asarray(y)                 # the real module
array([ 1.,  3.,  5.,  7.,  9., 11.])
>>> plain += y                       # plain is an ndarray of ones
>>> plain
array([ 2.,  4.,  6.,  8., 10., 12.])
```

Each of those lines made the layer compute a graph early. With the logger `hotkernels.lazy` at `DEBUG` it says so, once per line of code:

```text
needed: float() at session.py:10
needed: A truth test (if, while, and, or, not) at session.py:11
needed: Formatting with '.1f' at session.py:13
needed: Conversion to an ndarray at session.py:14
needed: np.add at session.py:16
```

A call outside the deferred set, which NumPy had to make at once, is a line of another kind, and it is the only kind at `INFO`:

```text
eager: np.argsort at session.py:18
```

The two are apart on purpose. `eager` lines say what the deferred set does not cover yet, and are what one usually opens a log for. The others say where the program's graph was cut. `hotkernels.lazy_report()` has every occurrence of both, ranked; [What the layer says](#what-the-layer-says) has the whole of it.

A strict stand-in gives arrays that refuse instead:

```python
>>> np = hotkernels.get_lazy(strict=True)
>>> total = (np.arange(6.0) * 2 + 1).sum()
>>> float(total)
NotGathered: float() needs the values of a lazy array: call .gather() first
>>> if total > 30: ...
NotGathered: A truth test (if, while, and, or, not) needs the values of a lazy array: call .gather() first
>>> numpy.asarray(total)
NotGathered: Conversion to an ndarray needs the values of a lazy array: call .gather() first
>>> float(total.gather())
36.0
```

`NotGathered` is a `TypeError`, which is what Python raises when an object cannot be turned into the number or array that was asked for. The full list of what is computed, and how much, is under [What is computed without `gather()`](#what-is-computed-without-gather).

### Several at once, and a loop that keeps its state lazy

```python
w_now, b_now = hotkernels.gather(w, b)     # two results, one walk of the graph

for batch in batches:
    w = w - lr * gradient(w, batch)
    w.gather()                             # compute this step now; w stays a lazy array
```

### Looking at the graph

`loss.explain()` returns what is pending, one operation per line, with the line of the program that wrote it. From the prototype, for the loss of the first step of the regression above:

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

An operation that several others use is printed once and marked "(shown above)" where it comes again. Without that, the graph of a loop, in which each step uses the step before twice, would double in length with every step.

Text is enough for a dozen operations. For more, and for a graph that has already been computed, [Debugging tools](#debugging-tools) has the graph as an object, as a page in the browser, and a mode that records the graphs a block of code produced.

### The whole interface

| Name | What it is |
|---|---|
| `hotkernels.get_lazy(strict=False)` | Returns a stand-in for the `numpy` module: the same names, and every array it makes is lazy. It changes nothing in NumPy. With `strict=True` its arrays, and everything computed from them, raise where a value is needed and `gather()` was not called |
| `x.gather()` | Computes `x` and returns what NumPy would have returned for it: an `ndarray`, or a NumPy scalar where NumPy gives one. `x` is still a lazy array afterwards, now computed |
| `hotkernels.gather(a, b, ...)` | The same for several arrays, as a tuple. With no argument it computes everything pending |
| `x.explain()` | The pending graph of `x` as text. Computes nothing |
| `x.graph()` | The pending graph of `x` as an object that can be written as JSON or shown in the browser. Computes nothing |
| `hotkernels.record_graphs()` | For a `with` block: keeps the graphs the block's code produced, the computed ones included, and what was computed without being asked. Its page opens on the program's code |
| `hotkernels.LazyArray` | The type of every lazy array, for `isinstance` and for annotations |
| `hotkernels.NotGathered` | Raised by a strict array when something needs its values and `gather()` was not called. A `TypeError` |
| The loggers `hotkernels.lazy.eager` and `hotkernels.lazy.cut` | The lines of the program where NumPy had to make a call at once (`INFO`), and where a graph was computed early (`DEBUG`). Each line once |
| `hotkernels.lazy_report()` | Every occurrence of both, ranked: which call, how often, on which lines. Printing it gives a table |

The names `graph` and `record_graphs` are provisional.

Rules:

- **Importing does nothing, and neither does `get_lazy()`** beyond building the stand-in. NumPy is not changed. Only code that holds a lazy array behaves differently.
- **It does not need `install()`,** a kernel or the binding. Where `install()` does nothing because the machine is not supported, `get_lazy()` works the same. With `install()` in effect, the calls the layer makes reach the kernels: [With `hotkernels.install()`](#with-hotkernelsinstall).
- **It refuses nothing unless it is asked to.** A call outside the deferred set is made at once, a value that is needed is computed, and the log says so both times. Only a strict array raises.
- **Every error a user sees for a wrong shape or type is NumPy's,** raised on the line that wrote the operation.

## Types: what changes for the user

"Type" means two things for an array, and only one of them changes.

- **The element type does not change.** `x.dtype` is the `numpy.dtype` object NumPy would give the array. `np.float32`, `x.astype(...)` and promotion are NumPy's: `float32 * 2.0` stays `float32`, `float32 * np.float64(2)` becomes `float64`, the sum of `int8` is `int64`. The layer does not decide any of this; it asks NumPy ([Shape and element type without computing](#shape-and-element-type-without-computing)).
- **The container is one new class, `hotkernels.LazyArray`.** It cannot be `ndarray` or a subclass of it: an `ndarray` owns memory that any code may read at any time without calling anything, and a pending array has no memory yet ([gradient-step-findings.md](gradient-step-findings.md#can-a-hook-fuse-the-statement)). One class serves every result, including those NumPy returns as scalars: before `gather()` the layer has computed nothing, so everything is a `LazyArray` with a shape, and the shape of a scalar is `()`.

Where a type shows in a program, and whether the user changes anything:

| In the program | With lazy arrays | Change needed |
|---|---|---|
| `x.dtype`, `np.float32`, `x.astype(np.float32)` | NumPy's own, unchanged | None |
| `type(x)`, `print(x)` | `hotkernels.LazyArray`; a description without values | None |
| An annotation: `def step(w: np.ndarray) -> np.ndarray` | Python does not enforce annotations, so the function runs. The annotation is no longer accurate | None to run. For accuracy: `np.ndarray \| hotkernels.LazyArray` |
| `isinstance(x, np.ndarray)` | False | Yes. The one place a program has to change: `isinstance(x, (np.ndarray, hotkernels.LazyArray))`. It is also the one place a program can go another way without a word: code that tests it takes its other branch |
| The result of a reduction: `loss = (r ** 2).mean()` | A `LazyArray` of shape `()`. `loss.gather()` is an `np.float64`, as NumPy's `loss` is | None. `float(loss)`, `loss < tolerance` and `f"{loss:.3f}"` compute it; `gather()` says so in the code |
| A library that takes an array: pandas, matplotlib | The library asks NumPy for an array and is given an `ndarray`, computed on the spot | None. `x.gather()` says so in the code |
| Code that reads memory directly: a Cython function taking `double[:]`, `memoryview(x)` | `TypeError`: a lazy array has no buffer to give | Pass `x.gather()` |
| An editor or a type checker | `np` is not known to be NumPy after `np = hotkernels.get_lazy()`: no completion on `np.`, no checking of calls | None to run. [Other ways in](#other-ways-in) has the spelling that could restore it |

Two things are not in this design:

- **Making `isinstance(x, np.ndarray)` true.** The stand-in could hand out, as `np.ndarray`, a type whose instance check accepts lazy arrays. It would work only in the user's own module, since every other module asks the real `numpy.ndarray`, and it would tell code that it holds memory when it does not.
- **Type stubs.** What a type checker should be told a lazy array is, and how the stand-in's names can be typed so that `np.` completes.

## What a lazy array is

### Known at once, computed later

| Known when the line runs, without computing | Computed later |
|---|---|
| `x.shape`, `x.dtype`, `x.ndim`, `x.size`, `len(x)` | The values, and the length of what a pending mask selects |
| Whether the shapes fit and the types combine: NumPy's exception, on that line | Warnings and errors that depend on values: division by zero, overflow |

### Three kinds of operation

| Kind | What happens | Which |
|---|---|---|
| Deferred | A node joins the graph. Nothing is computed | The set in the next table |
| Made at once | The lazy operands are gathered, NumPy makes the call, and the result is a lazy array that has its values | Everything else in NumPy that gives arrays back: `np.linalg.solve`, `np.argsort`, `np.concatenate`, `x.astype(np.float32)`, any method or function not listed |
| A write into a lazy array | A node joins the graph: the array's contents after the write. Nothing is computed. Only where the array's memory is known outside the layer is everything pending computed and the write made at once | `x[i] = v`, `x += y`, `np.multiply(a, 2, out=x)`; see [Writes](#writes) |

The third row is what lets a loop be one graph; [Writes](#writes) says how NumPy's meaning of a write is kept.

**A call made at once is not an error.** It computes its operands without being asked, and nothing leaves the lazy world: the result is a `LazyArray`. Raising here would stop nearly every real program at its first `np.argsort` until the deferred set had grown to cover it. Each one is logged and counted, and each is a place where the program's graph is cut, which the [debugging tools](#debugging-tools) show.

### The deferred set

Narrow in kinds, not in count. The first build computes each node with NumPy's own call, so admitting an operation costs a shape rule and nothing else.

| Kind | Deferred | Not deferred |
|---|---|---|
| Element by element | Every ufunc with one output and no core dimensions (`np.add`, `np.exp`, `np.tanh`, `np.maximum`, `np.less`, ...), on arrays and numbers given by position; the operators that call them: `+ - * / // % **`, unary `-`, `abs`, the six comparisons, `& \| ^ ~`; `np.where(c, a, b)`; `np.copy(x)` and `x.copy()`; `np.round(x, decimals)`, `np.around` and `x.round(decimals)`, with a whole number of decimals by position or as `decimals=`, or none; a cast to a boolean, integer, floating or complex type, `x.astype(t)` and `np.astype(x, t)` | A call with `where=`, `dtype=` or `casting=`; a cast with `copy=False` to the array's own type, which gives the array itself, or with `order=`, `casting=` or `subok=` other than NumPy's default |
| Matrix product | `@`, `np.matmul`; `np.dot(a, b)` and `a.dot(b)` for arrays of one or two dimensions, where it is the same product | `np.dot` with more dimensions or with a number, where it is another operation; `np.einsum`, `np.linalg` |
| Reductions | `sum`, `mean`, `max`, `min`, `prod`, `var`, `std`, `any`, `all`, as functions and as methods, with no argument, with an axis by position or as `axis=`, and with `keepdims=`. `argmax` and `argmin` the same way, with one axis or none; `np.nanargmax` and `np.nanargmin` too, whose error for a slice that is all NaN depends on the values and is raised when the graph is computed. `np.count_nonzero`, with the same arguments as `sum`; its element type is `intp` | Any other argument, such as `where=`; `np.count_nonzero(a)` without an axis on a NumPy before 2.3, which answers it with a Python `int` |
| Running totals | `np.cumsum` and `np.cumprod`, as functions and as methods, over all the elements or along one axis | With `dtype=`; an array without dimensions |
| Searches | `np.searchsorted(a, v)` and `a.searchsorted(v)` for a sorted array of one dimension, with `side=` if given | With `sorter=` |
| Views | `x.T`, `np.transpose`, `np.swapaxes`, `reshape`, `ravel`, `np.expand_dims`, `np.squeeze`, and indexing with integers, slices, `None` and `...` | `ravel` with an `order`; one element of an array that has its values, which is read on the spot (below) |
| Ordering | `np.partition(a, kth)` with `kth` one whole number or several, along the last axis, another axis or `axis=None`. The result is a new array of the operand's shape and element type, with NumPy's own choice of what lands on each side | `kind=` other than the default, `order=`, a `kth` that has to be computed; `a.partition(kth)`, which writes into `a` ([Writes](#writes)); `np.sort`, `np.argsort`, `np.argpartition` |
| Indexing by arrays | `x[rows]`, `x[[0, 2, 4]]`, `x[rows, columns]`, `x[1:, columns]`, with arrays or lists of integers, and `x[mask]`, `x[mask, 2:]` with a mask. An array in the index may be a lazy array still pending: `x[np.argmax(w)]`, `x[order[:k]]`, `x[speed > limit]`, `x[np.flatnonzero(m)]`, `w[np.nonzero(m)]` | A pending mask beside another array in one index; positions whose number is not known beside any array but those of the same `np.nonzero`; an index array of another element type |
| Positions of what is not zero | `np.flatnonzero(a)`, `np.nonzero(a)`, `a.nonzero()` and `np.argwhere(a)`, of an array with dimensions. How many there are is [a length that depends on values](#a-length-that-depends-on-values) | Of a number, and of an array that itself has such a length; `np.where(c)` with one argument, `np.unique` |

**An array made like another needs no value.** `np.zeros_like(x)`, `np.ones_like(x)`, `np.empty_like(x)` and `np.full_like(x, v)` use the shape and element type of `x`, which are known. The new array is made on the spot and `x` is not computed for it.

**`any` and `all` are deferred although their result is nearly always tested.** `if not known.all():` needs the value on the line either way, so deferring moves the cut from the call to the `if` and costs nothing. The log then names the truth test as what needed a value, which is the program asking, and not `all` as a call the layer could not defer. Where the result stays in arithmetic the graph goes on: `np.where(mask.any(axis=1), a, b)`, `x.all(axis=0) & y`.

**A lazy array in an index need not have its values.** An array of integers still pending is an operand of the node like any other. The shape of the result follows from the shape of the index, so it is known; what the index holds is checked against the bounds when the graph is computed, and an index out of bounds raises NumPy's error then, with a note naming the line. An index that has its values is checked on its line.

**One element of an array that has its values is read on the spot.** `x[i]` and `x[r, c]`, with one whole number for each dimension, on an array that is not pending, give a lazy array that has its value: NumPy's number, of NumPy's type, with no node and nothing counted.

- **A Python loop asks for one element per turn:** `for i in positions:`, `flags[i]`, `valid[r, c]`. Deferred, each element is a node, and a graph of its own as soon as Python uses it as a list index or tests it: four such loops over the 13 positions a mask selects, two of them reading another array at each position, are 66 graphs. Read on the spot they are one, the positions themselves.
- **Nothing is lost by reading it.** Nothing is pending under an array that has its values, and one element is a number: a copy, which no later write has to reach. A row or a slice of the same array is a view and stays deferred, so that writes are seen through it ([Writes](#writes)).
- **A pending array is left alone.** Its elements stay deferred: reading one would compute the array on a line that asked for no value. Once something has computed it, by the first element a loop uses or by a `gather()`, the rest of the loop reads.
- **The rule is on the index, not on iteration.** `flags[i]` and `valid[r, c]` are not iteration; `for v in x:` gets the same elements because it gives `x[0]`, `x[1]`, and so on.
- **The element is the one there when the line runs,** which is NumPy's. That holds for an `ndarray` taken in by reference too: a write made directly to it afterwards does not change an element already read ([What differs from NumPy](#what-differs-from-numpy), the third).
- **The type is still `LazyArray`,** and a strict array's element is strict ([Strict arrays](#strict-arrays)).

Never deferred, whatever the operation: an operand with a dimension of length zero, an element type that is not boolean, integer, floating or complex, and an operand that is neither a lazy array, an exact `ndarray`, a NumPy scalar nor a Python number.

### A length that depends on values

`x[mask]` has as many elements as the mask has true ones. When the mask is a lazy array not yet computed, nobody knows that number. Making the selection at once, to learn it, computes the mask and everything it was made from, at every such line: a loop that caps, filters or groups by a mask would be computed at every step.

So the selection is deferred, and its shape has one length marked as not known:

```python
>>> fast = speed > limit
>>> particles[fast]
LazyArray(float64, shape=(?, 4), pending)
```

- **What is computed from it is deferred wherever the result's shape does not depend on the number.** An operation with a number, with an array of length one along that axis, or with another selection by the same mask, which has the same length whatever it is. A matrix product. A reduction: `points[nearest == c].mean(axis=0)` has shape `(2,)`, which is known. A running total along an axis, and a search, whichever of its two arrays has the length. A transpose, a new axis, and an index that takes the whole of that axis with `:`.
- **An assignment through the mask is deferred** when what is assigned is a number or was made from a selection by the same mask. `particles[fast, 2:] *= (limit / speed[fast])[:, None]` is six nodes and computes nothing.
- **Where NumPy's answer depends on the number, the call is made at once:** part of that axis (`x[mask][:5]` has five elements or fewer), one element of it, `reshape` and `ravel`, `np.partition`, whose `kth` has to fit the length, an operation with an array of a known length or with a selection by another mask, `np.zeros_like`. The selection is computed and NumPy decides, with its own error if the lengths do not fit.
- **Asking for the length is asking for a value.** `x[mask].shape`, `len(x[mask])`, `.size` and iteration compute the selection, and the log says `needed: The shape of what a mask selects`. A strict array raises `NotGathered`. `x.ndim` and `x.dtype` are known as always.
- **An error that depends on the number is raised when the graph is computed,** with a note naming the line that wrote the operation: `np.max` of a selection that turns out empty. A mask of the wrong shape for the array is NumPy's error on the line, as before.
- **The positions of what is not zero have such a length too.** `np.flatnonzero(a)` has shape `(?,)`, each array of `np.nonzero(a)` has shape `(?,)` and `np.argwhere(a)` has `(?, a.ndim)`, with `intp` as the element type. The mark is the one a selection by `a` as a mask carries, so `x[m]`, `np.flatnonzero(m)` and the arrays of `np.nonzero(m)` are known to have one length, whatever it is. An index of such positions is deferred as an index by a pending mask is, `x[np.flatnonzero(m)]`, `w[np.nonzero(m)]`, `x[np.flatnonzero(m), 2:]`, and so is an assignment through one, on the same condition. `np.nonzero` gives a tuple, and a node has one value: each array of the tuple is a node of its own that makes the call and keeps its part, so the first build calls `np.nonzero` once per dimension.
- **A mask that has its values is not this case.** An `ndarray` of booleans, or a lazy one already computed, selects a number of elements that is known, and the result has an ordinary shape.

### When the values are computed

| Event | What is computed |
|---|---|
| `x.gather()`, `hotkernels.gather(a, b)` | Those arrays and what they need |
| A call made at once | Its operands |
| A write into a lazy array whose memory is known outside the layer, and `hotkernels.gather()` with no argument | Everything pending |
| A number, a truth value or a string is needed and `gather()` was not called | The array asked, and what it needs |
| An array is handed out, or an `ndarray` is written from a lazy array, and `gather()` was not called | Everything pending |
| More than 100,000 nodes pending | Everything pending, so a long loop that never gathers cannot hold an unbounded graph |
| `x.shape`, `x.dtype`, `len(x)`, `print(x)`, `x.explain()`, `x.graph()`, `np.shape(x)`, `for row in x:` | Nothing |
| The same of what a pending mask selects, except `x.dtype` and `print(x)` | That array, and what it needs |

A node is computed once. Its value is kept while the program holds the lazy array, and an intermediate nobody holds is freed as soon as its last consumer has been computed. Pending work that the program drops without asking for it is never computed.

### What is computed without `gather()`

| What the program does | What it gets | What is computed first |
|---|---|---|
| `float(x)`, `int(x)`, `complex(x)` | A Python number | `x` |
| `if x > 0:`, `while`, `and`, `or`, `not`, `bool(x)` | A truth value | `x` |
| `range(n)` | An integer | `n` |
| `f"{x:.3f}"` | The value, formatted. `f"{x}"` is the description and computes nothing | `x` |
| `v in x` | A truth value | `x` |
| `x.tolist()`, `x.item()`, `np.allclose(x, y)`, `np.array_equal(x, y)` | NumPy's plain Python value | The call's operands |
| `np.asarray(x)`, `np.array(x)` with the real module, `plt.plot(x)`, `rng.permutation(x)`, `a[i] = x` and `a[x]` on a plain `a` | NumPy is given an `ndarray` | Everything pending |
| `a += x`, `np.multiply(x, 2, out=a)`, `np.copyto(a, x)` with a plain `a` | NumPy writes `x`'s values into `a` | Everything pending |
| `x.flat`, `x.real`, `x.data`, `x.ctypes` | The attribute of the computed array | Everything pending |
| `for row in x:`, `list(x)` | Lazy views of the first dimension, `x[0]`, `x[1]`, ...; of one dimension that has its values, its elements, each with its value | Nothing: no value is needed |
| `memoryview(x)`, a function that takes a buffer | `TypeError` | Nothing: a lazy array has no buffer |

Two rules decide how much is computed.

- **A number, a truth value or a string: the array asked, and no more.** Nothing that leaves can be written through, so nothing else has to be settled first. From an array that has its values nothing is computed at all, and the line is not logged or counted ([What the layer says](#what-the-layer-says)).
- **An array handed out, or an `ndarray` about to be written: everything pending.** Whoever receives an `ndarray` may write into it, and `a[i] = x` on a plain `a` reaches a lazy array the same way `np.asarray(x)` does. A pending operation may still read the memory that is about to change. Once nothing is pending, the program is in the state the eager program would be in, and whatever happens next is NumPy's. It is the rule for a write that is made at once ([Writes](#writes)), applied wherever memory changes hands, and it is never more work than eager NumPy had done by that line.

**What continues from an array that left is plain NumPy.** `np.asarray(x)` is an `ndarray`; what a library computes from it is computed at once and is not in any graph, until a result meets a lazy array again. The program is right. It may be less lazy than it looks, which is what the log is for.

### Why it computes instead of raising

- **A real program needs a value every few lines.** It prints a loss, tests a tolerance, hands an array to a plotting or a data library, checks `if mask.any():`. If each of those raised, the program would have to be rewritten before its first graph could be looked at, and every small operation the layer did not expect would end an attempt. The point of the first build is to run existing programs and see their graphs.
- **The answer is NumPy's.** Computing on the spot is what eager NumPy does on every line. The layer adds no rule of its own to the result, only the one rule above about what is computed first.
- **Nothing is hidden.** The log names each such line once. A recording has each as a graph of its own, with what needed the value and where ([lazy-recording-design.md](lazy-recording-design.md#what-one-graph-is)), and the page marks those graphs. So the places where a program's graph is cut can be found and counted without the program having to stop at each.
- **Coverage can grow under a program that already runs.** An operation outside the deferred set costs laziness, not the run, so the set can be widened by looking at what real programs log. Other projects took the same road: Dask's arrays compute when `float()`, `bool()` or `np.asarray()` asks, and PyTorch's backend for Apple GPUs runs an operator it lacks on the CPU, with a warning, when [`PYTORCH_ENABLE_MPS_FALLBACK`](https://docs.pytorch.org/docs/main/mps_environment_variables.html) is set.

What it costs:

- **"It runs" stops meaning "it is lazy".** A program whose first line hands its input to a library runs to the end with nothing in a graph. The log and the page are how that is seen; nothing else will say it.
- **A value can be computed where the author did not think of it,** inside a loop for instance, cutting the graph at every step. Again it is right, and visible, and slower than it could be once anything computes a graph faster than node by node.

### Strict arrays

`hotkernels.get_lazy(strict=True)` gives a stand-in whose arrays raise `NotGathered` wherever the table above computes. Iteration and `print(x)` need no value and work as before.

- **It is how an unnoticed road to the memory is found.** NumPy asks an object for `__array_interface__` and `__array_struct__` before it calls `__array__`. A lazy array that answered those, for instance by passing unknown attributes on to the computed array, would hand over its memory without the layer hearing of it: no log line, and nothing pending computed first. With computing on the spot the results still look right whenever the array happens to be computed already, so nothing shows. On a strict array every known road raises, so a road nobody thought of is the one that does not, and a test finds it. The layer's own tests run strict for this reason, and a lazy array passes on methods only, never a double-underscore name ([What an unknown attribute gets](#what-an-unknown-attribute-gets)).
- **It is for a hot path that must not compute by accident.** Code that has been made lazy on purpose can say that a value needed without `gather()` is a mistake there.
- **Strictness belongs to the array, not to the process.** What is computed from a strict array is strict, in whatever file and beside whatever other array. A program can be strict in one part and not in another, and nothing global is switched.
- **A strict array raises even when it is already computed.** No work would be hidden in that case, but the program would then behave differently depending on whether something had been gathered earlier.
- **One element of a strict array that has its values is a strict array.** `x[i]` reads it as it does for any array, and no value has left: `int(x[i])`, `if x[i]:` and `names[i]` in a loop over `x` raise `NotGathered`, and `x[i].gather()` gives the number.

### Writes

Programs write into arrays all the time, and loops most of all: `w -= learning_rate * gradient`, `total += term`, `x[mask] = 0`, `np.clip(g, -5, 5, out=g)`. If each write made the layer compute, a training loop would be computed a statement at a time and no graph would be larger than a line. So a write is an operation of the graph.

```python
w = np.zeros(3)             # the layer made it
before = w + 1              # reads what w holds now
for step in range(50):
    w -= 0.1 * (w - 1)      # nothing is computed: w now stands for a new node
w.gather()                  # fifty steps, one graph
before.gather()             # array([1., 1., 1.]): what w held when the line ran
```

- **The array gets new contents; nothing that exists is changed.** `w -= s` makes a node for "what `w` held, with `s` subtracted in place", and `w` stands for that node from then on. An operation written earlier keeps reading the node `w` stood for when it was written, so it has the old contents whenever it is computed. That is why nothing has to be computed first.
- **Another name for the array sees the write,** as in NumPy: the lazy array is the same object under both names, and it is the object that changes what it stands for.
- **A view sees it too, and a write through a view reaches the array.** A view (`x[1:3]`, `x.T`, `x[:, 2]`) is kept as "this view of whatever `x` holds", not as a node of its own. After a write into `x` it is the view of the new contents; a write into the view is a write into `x` at that place. `x[i] += v`, which Python carries out as take the view, update it, assign it back, comes out as one write.
- **A number is not written into.** For a result NumPy gives as a scalar, `s += v` makes a new scalar and leaves the old one alone. The layer does the same: a new lazy array.
- **Shapes, bounds and types are checked on the line.** An index out of bounds, a value that does not fit the place it is assigned to and a result that cannot be cast into the array raise NumPy's own errors when the write is written. The assignment is tried on arrays of an element type of no size, and the types on arrays of one element. An index that holds a lazy array still pending is the exception: what an array of integers holds is checked against the bounds when the graph is computed, and through a mask only a number, or an array made from a selection by the same mask, is assigned without computing, since nothing else is sure to fit.

Other systems that build a graph put assignment in it: TensorFlow's graphs had an operation for assigning to a variable, and JAX has `x.at[i].set(v)`. The difference here is that the program is not rewritten: it says `x[i] = v`.

**Where the write is still made at once.** NumPy's meaning of a write is that the memory changes, there and then, for everyone who can see it. Deferring is sound only while nobody outside the layer can see that memory:

| The array | Why its memory may be seen | A write into it |
|---|---|---|
| Made by the stand-in from no array (`np.zeros`, `np.arange`, `np.random.randn`, a generator's `standard_normal`), or the result of a deferred operation | It cannot be | A node |
| An `ndarray` the user handed in: `np.asarray(a)`, or `a` met in an operation | The user holds `a` | At once, in `a`, after everything pending |
| An array that was gathered, or handed to NumPy or a library | The user holds what `gather()` returned, which is the array itself | At once, until a later deferred operation gives the name new contents |
| An array with a live `reshape` or `ravel` of it, or the `reshape` or `ravel` itself | NumPy shares memory between the two or copies, depending on a layout not known before computing | At once |
| An array with a live row taken by a position still pending, `x[np.argmax(w)]`, or that row | NumPy gives a view of the row when the position is a number and a copy when it is an array without dimensions, which is not known before computing | At once |
| An array a call made at once may have returned a view of: `np.atleast_2d(x)`, `x.astype(x.dtype, copy=False)` | The result and `x` overlap | At once, for both |

At once means what a write meant before: everything pending is computed in the order written, and NumPy writes into the memory. It is always right and it cuts the graph, and it is logged as `write`.

**What it costs.** A deferred write is computed as a copy of what the array held, with the write made into the copy. NumPy's write in place makes no copy. The results are the same; the work is not, until whatever computes a graph sees that nothing else reads the old contents and writes into them. That is left for when computing a graph differently is taken up, with every other question of speed.

### What differs from NumPy

A user has to know six things.

1. **A value needed without `gather()` is computed where it is needed,** and what continues from an array that left is plain NumPy. The program is right and may be less lazy than it looks; the log and a recording say where. See [What is computed without `gather()`](#what-is-computed-without-gather).
2. **A lazy array is not an `ndarray`.** `isinstance(x, np.ndarray)` is False. See [Types](#types-what-changes-for-the-user).
3. **A pending expression reads an `ndarray` when it is gathered, not when it is written down.**

   ```python
   a = loader.read()        # a plain ndarray: [0., 1., 2.]
   y = np.asarray(a) * 2    # pending; np is the stand-in
   a[0] = 5                 # written directly to the ndarray
   y.gather()               # array([10., 2., 4.]); eager NumPy had computed y before the write: [0., 2., 4.]
   ```

   Writes made through a lazy array are safe, by the rule for writes; a write made directly to an `ndarray` that a pending expression has read is not seen by the layer. Gather before such a write. `np.array(a)` copies, as it does in NumPy, and `np.asarray(a)` does not. One element of an array that has its values is read on its line, so `np.asarray(a)[0]` is the element from before the write, as eager NumPy has it. This is the one way a program that is partly lazy and partly eager can get a result eager NumPy would not give, and whether inputs should be copied instead is an [open question](#open-questions).
4. **Warnings and errors that depend on values come at the gather.** They are the same warnings, with NumPy's words, and each names the line that wrote the operation, not the line that gathered. Two errors about shape are among them, because they depend on values that are not there yet: an index out of bounds, when the index is a lazy array still pending, and whatever follows from how many elements a pending mask selects. `np.nanargmax` of a slice that is all NaN is an error of this kind as well. `np.errstate` is remembered per operation, so `with np.errstate(divide="ignore"): r = a / b` stays silent wherever `r` is gathered. A `warnings.catch_warnings()` block around the line does not reach the gather. A warning NumPy issues while it computes comes at the gather in the same way when it does not depend on values: the `ComplexWarning` of `z.astype(float)`.
5. **An operation that failed stays failed.** Under `np.errstate(over="raise")` the exception is raised by the first gather that needs the operation, with a note naming the line that wrote it, and again by any later gather that needs it. A write that computes everything pending raises it once, and not again for what depended on it.
6. **The work happens at the gather,** so that is where time passes and memory is taken, and what `KeyboardInterrupt` interrupts. A loop that writes in place does its work there too, and for now with a copy per write ([Writes](#writes)).

## With `hotkernels.install()`

`get_lazy()` and `install()` are independent, and they work together. A gather makes NumPy calls on real `ndarray`s, so the pointers `install()` changed inside NumPy serve those calls as they serve the eager program's. It takes one rule to make that hold.

**A node makes the call the user wrote:** the same function or method, with the arguments that were given, given the same way. `x.sum()` is computed as `value.sum()`, not as `np.sum(value, axis=None, keepdims=False)`.

The values are the same either way. The reason is that a kernel's hook decides by how a call is spelled. `sum_f64` takes `x.sum()` and `np.sum(x)`, and leaves a sum that names `axis` or `keepdims`, even as `axis=None`, to NumPy ([delivery-design.md](delivery-design.md#which-hook-serves-which-kernel) has each hook's conditions). A layer that tidied every call into one form would give NumPy's results and miss the kernels. The same holds for anything else that looks at how it was called, so the rule is not about hotkernels alone.

What follows from it:

- **Order does not matter.** `get_lazy()` may be called before or after `install()`; it holds no pointer of NumPy's.
- **The results are the eager program's under the same installation.** The lazy program makes the calls the eager one makes, on the same operands, through the same pointers. With `install()` as it is by default those are NumPy's bits.
- **`report()` counts the layer's dry runs.** The call that works out an element type, on arrays of one element ([Shape and element type without computing](#shape-and-element-type-without-computing)), is a real call, and a kernel that takes arrays of any size takes it. The answer is remembered ([What is remembered while a graph is written](#what-is-remembered-while-a-graph-is-written)), so the dry run is made the first time an operation meets operands of those shapes and element types and not again: a deferred `sum` or `np.where` is counted once more than it is computed, however often its line runs.
- **Three methods are computed through NumPy's function of the same name:** `x.reshape(...)`, `x.transpose(...)` and `x.swapaxes(...)`, and `x.T` through `np.transpose`. They make views and compute no values. An operator is computed as the ufunc it calls: `a + b` as `np.add(a, b)`.

Measured on the prototype: [With `install()` in effect](#with-install-in-effect).

## Debugging tools

The first build makes nothing faster. What it produces is the graph, so the graph has to be easy to look at, first of all for the people building the layer. Each of the two tools beyond `explain()` has a document of its own and a prototype.

| Tool | What it gives | Designed in |
|---|---|---|
| `x.explain()` | The pending graph of one array as text | [Looking at the graph](#looking-at-the-graph) |
| `x.graph()`, `hotkernels.record_graphs()` | The graph as plain data, and a mode that keeps every graph a block of code produced, the computed ones included | [lazy-recording-design.md](lazy-recording-design.md) |
| `recording.show()` | One page in the browser. It opens on the program's code, with what each line wrote and what it made the layer compute, and has each graph one click away: folded where it is large, with search and selection | [lazy-graph-page-design.md](lazy-graph-page-design.md) |
| The log, `hotkernels.lazy_report()` | What NumPy had to compute at once and where graphs were computed early: as it happens, and ranked | [What the layer says](#what-the-layer-says) |

What the main design needs to know of them:

- **One graph per computation.** Each time the layer computes, the nodes computed together are one recorded graph, with what made it compute and on which line. That is the unit anything computing a graph differently will be handed, and every one of them is a place where the program's graph was cut.
- **A mode, because a node forgets.** A node drops its operation and operands once computed, which is what frees intermediates. So graphs are recorded only inside a `with` block, as plain data without values, and the layer is told what to record through one hook.
- **A large graph is folded.** The record carries the operations as a tree: the lines that wrote them, the calls of functions, the steps of loops. The page draws a group of more than five operations as one box and unfolds it on a click.

![The regression's graph on the page: one line drawn in full, the loop of 50 steps as one box](images/lazy-graph-page-loop.png)

## What was verified

A throwaway probe of NumPy's protocols, not in the repository, and the prototype in [`prototypes/lazy/`](../prototypes/lazy). Apple M1 Pro, macOS, CPython 3.12.7, NumPy 2.5.5.

### What reaches a type that is not an `ndarray`

A class with `__array_ufunc__`, `__array_function__` and the operators, given to the real `numpy` module beside an `ndarray` `a` and a matrix `m`:

| Spelling | Arrives as |
|---|---|
| `np.exp(x)`, `np.sqrt(x)`, `np.tanh(x)`, `np.maximum(x, 0)` | `__array_ufunc__` |
| `a + x`, `np.add(a, x)`, `np.float64(2) + x` | `__array_ufunc__`: the `ndarray`'s own operator hands the call over |
| `x + a`, `2.0 + x` | The type's own `__add__` and `__radd__` |
| `m @ x`, `np.matmul(m, x)` | `__array_ufunc__`. `x @ m` is the type's `__matmul__` |
| `np.where`, `np.sum`, `np.mean`, `np.clip`, `np.dot`, `np.transpose`, `np.swapaxes`, `np.concatenate([a, x])`, `np.argmax`, `np.einsum`, `np.linalg.norm`, `np.linalg.solve`, `np.zeros_like`, `np.shape` | `__array_function__`, with the function and its arguments as written |
| `np.add.reduce(x)` | `__array_ufunc__`, method `reduce` |
| `a += x`, `np.multiply(a, 2, out=x)` | `__array_ufunc__` with `out`: the write is seen before it happens |
| `np.asarray(x)`, `a[i] = x`, `a[x]` | NumPy looks for `__array_interface__` and `__array_struct__` on the object, then calls `__array__`. An exception raised there reaches the caller as it is |

Not reached, because no lazy array is among the arguments: `np.zeros(3)`, a random generator's methods, a file loader. That is what the stand-in for the module is for.

### The prototype

Nine modules of Python and one optional C source, laid out as the package would be ([Where it lives](#where-it-lives)), so that a comment on a file is a comment on a piece of the design. It follows this document: the deferred set above, every other call at once, a value that is needed computed and logged, strict arrays that refuse instead, node-by-node evaluation with the call as written. `uv run --group dev pytest tests/test_lazy_prototype.py` runs 54 tests, one or more per rule stated here, each comparing with the real `numpy` module. The 16 tests of the graph tools are in `tests/test_lazy_graphs.py` and are described in those tools' documents. The 21 tests of what is remembered while a graph is written, and of the C core, are in `tests/test_lazy_building.py` ([What writing a node costs](#what-writing-a-node-costs)).

| Program | Deferred | Made at once | Result against eager NumPy |
|---|---|---|---|
| Linear regression ([`programs/regression.py`](../prototypes/lazy/programs/regression.py)): standardised features, 50 gradient steps with an L2 penalty, no gather inside the loop, 20,000 × 30 `float64`, run with the stand-in | 954 operations | 0 | Coefficients, intercept and all 50 losses identical, bit for bit |
| Transformer block ([`programs/attention.py`](../prototypes/lazy/programs/attention.py)): layer norm, four-head attention with a causal mask, softmax, GELU, two residuals, `float32` parameters, 64 × 96, in a module that imports the real `numpy`, with one lazy input | 55 | 0 | Identical |

The examples under [What the user writes](#what-the-user-writes) were also run as printed there, the attention example as two files. The log lines shown there are the prototype's, word for word; their file names and line numbers are those of the scripts that were run.

What the tests hold, each matching a rule of this document:

- **Errors on the line.** A sum of shapes (6,) and (4,), a matrix product with mismatched inner sizes, `np.sum(x, axis=3)` on one dimension, a `reshape` to the wrong size, an index out of range, `int8_array + 300` and `float & float` each raise the exception NumPy raises, with the same message, when the line runs.
- **Promotion is NumPy's,** for ten expressions that mix Python numbers, NumPy scalars, integers, booleans and reductions. The transformer block comes out `float64` under both, because `np.sqrt(2 / np.pi)` is an `np.float64`.
- **Warnings.** `a / zeros` gathered later issues NumPy's two warnings, word for word, with the file and line of the division. Written under `np.errstate(divide="ignore")` and gathered outside it: none. Under `errstate(over="raise")`: `FloatingPointError` at the gather, with the note, and the same exception again at the next gather.
- **A value needed without `gather()`:** thirteen ways to a number, a truth value, a string or a plain Python value each give what the same line gives with NumPy. So do `np.asarray`, `np.array` and a random generator's `permutation`, and four ways of writing lazy values into an `ndarray`.
- **Everything pending first.** An `ndarray` that a pending expression has read is written from a lazy array: the pending expression has the values from before the write, as eager NumPy has. A `float()` computes what it needs and leaves other pending work pending.
- **The log.** A line that needs a value three times in a loop is logged once and counted three times. A call made at once is an `INFO` line of one logger; a write into a lazy array and a value needed are `DEBUG` lines of another; with the package's logger at `INFO` only the first is seen.
- **The summary.** `lazy_report()` has every occurrence by call and by line, the most frequent first, in three parts, and starts again from nothing when asked.
- **Strict arrays.** Nineteen ways to a value each raise `NotGathered`, an `ndarray` that a refused write named is unchanged, and strictness follows an array through the real module and beside lenient and plain arrays. Neither kind of array has `__array_interface__` or `__array_struct__`.
- **Iteration** gives lazy views of the first dimension and computes nothing, for a strict array too.
- **A Python loop over an array that has its values:** the positions of a mask iterated three times, used as list indices, tested through a second array and read from a third by row and column record one graph, the positions themselves, with two operations deferred and nothing made at once. Each element is a `LazyArray` that has its value, with NumPy's type and bits; negative positions and NumPy integers too. A position out of bounds raises NumPy's message on the line. A row, an element of a pending array and an index with `...` stay deferred. A number read from an array that has its values is not counted as a value needed, a conversion to an `ndarray` is, and so is a number that had to be computed. An element read before a write, through the layer or directly to the `ndarray`, keeps its value.
- **An element of a strict array** is strict and has its value: five ways to a Python value from it raise `NotGathered`, `gather()` gives NumPy's number, and arithmetic with it stays strict.
- **Deferred writes.** In-place operators, an assignment, `out=`, `np.clip(..., out=)` and an assignment through a pending mask into an array the layer made compute nothing and give NumPy's values; an expression written before them has the old contents; another name is the same object.
- **Views and writes.** Rows, columns and a transpose of one array follow writes into it, and writes through each of them reach it and each other, with NumPy's values. `base[2] -= 4` is one write.
- **Numbers.** `s += 1` on the result of a sum gives a new lazy array, deferred, and the old one keeps its value.
- **Writes made at once.** Into an `ndarray` the user handed in, the user's array changes on the line. Into an array that was gathered, the gathered array changes. With a `reshape` or a `ravel` of an array alive, writes into any of them are seen through the others as NumPy shows them.
- **Errors on the line.** An index out of bounds, too many values, the same through a view, and a result that cannot be cast each raise NumPy's message when the write is written, and the array is unchanged. `int_array += 0.5` raises NumPy's casting error.
- **Random generators.** `np.random.default_rng()` through the stand-in draws NumPy's numbers as lazy arrays into which a write is deferred.
- **Calls made at once** (`np.linalg.norm`, `x.argsort()`, `np.cumsum` with `dtype=`, `np.searchsorted` with `sorter=`, `np.argsort`, `np.concatenate`, `np.dot` of three dimensions, `np.sort`, `ravel` in another order, part of what a pending mask selects) give NumPy's results as computed lazy arrays.
- **`np.dot` and copies:** the product of matrices and vectors as a function, as a method and with a plain operand first, and `np.copy` and `x.copy()`, are each deferred and give NumPy's bits; a copy is a copy. Shapes that do not fit raise NumPy's message.
- **Indexing by integer arrays:** an array, a list, two arrays, a slice with an array, a computed lazy array and a two-dimensional index are each deferred with NumPy's shape and bits; an index out of bounds raises NumPy's message on the line. A pending index, alone and beside an array, a mask that has its values and a list of booleans are deferred too; a pending index out of bounds raises NumPy's message when it is gathered.
- **A pending mask:** `x[mask]` is a node whose description shows `shape=(?, 3)`. A reduction along the mask and of everything, arithmetic with the same mask's selection, a new axis, a matrix product, a position and a transpose of it are each deferred, with nothing made at once and no value needed, and give NumPy's bits, also when the mask selects nothing. A mask of the wrong shape raises NumPy's message on the line.
- **The length asked for:** `.shape`, `len()` and `.size` of a selection compute it, are counted as values needed, and a strict array raises `NotGathered`; afterwards the description shows the real shape.
- **What depends on the length** (part of a selection, one element of it, `reshape`, `ravel`, an operation with an array of a known length, `zeros_like`) is made at once and gives NumPy's result; two selections of different lengths raise NumPy's message.
- **Assignment through pending indices:** `p[fast, 2:] *= (c / speed[fast])[:, None]`, `p[fast, 0] = p[fast, 1] * 3`, the same through a mask that selects nothing, `p[rows, columns] -= 1` with pending positions and `p[np.argmin(s)] = 0` compute nothing and give NumPy's values. An array of a known length assigned through a pending mask is made at once.
- **Positions of what is not zero:** `np.flatnonzero` of one dimension, of two and of numbers, both arrays of `np.nonzero`, the method, `np.argwhere`, arithmetic and a reduction of the positions, rows and elements indexed by them, alone, with a slice and after an integer, and arithmetic with what the same mask selects are each deferred, with nothing made at once and no value needed, and give NumPy's type and bits, also when nothing is not zero. An assignment through them of a number and of what was read through them is a deferred write; of an array of a known length it is made at once. `len()` and iteration compute the positions and are counted as a value needed; a strict array raises `NotGathered`. Positions of two masks in one index, positions beside a known array, part of the positions, the positions in what a pending mask selects and of a number, and `np.where` with one argument are made at once and give NumPy's result. `np.nonzero` of a number raises NumPy's message on the line, and positions out of bounds for another array raise it when gathered. An assignment through a pending mask beside an array is made at once by NumPy.
- **Positions and `ravel`:** `np.argmax` and `np.argmin` as functions and as methods, with an axis and with `keepdims`, `x.ravel()`, `np.ravel`, a row and an element named by pending positions are each deferred with NumPy's shape, type and bits. `np.argmax` with two axes raises NumPy's message. A row named by a pending position and a `ravel` show later writes into the array, and writes into them reach it, as NumPy has it.
- **Rounding:** `np.round`, `np.around` and `x.round()`, with decimals by position and by keyword, negative decimals, on `float64`, `float32` and integers, of a number, through the stand-in, of what a pending mask selects and with the result read from afterwards, are each deferred with NumPy's shape, type and bits, and the node keeps the arguments as written. Decimals that are not a whole number, decimals given twice and too many arguments raise what NumPy raises, on the line. With `out=`, into the array that is read, into another and through a view, each is a deferred write with NumPy's values; a result that cannot be cast into `out` raises NumPy's message on the line.
- **Positions that skip NaN:** `np.nanargmax` and `np.nanargmin`, with an axis by keyword and by position, with `keepdims`, through the stand-in, of a selection by pending positions and of what a pending mask selects, and a row named by one, are each deferred with NumPy's shape, type and bits. Two axes and an axis the array lacks raise NumPy's message on the line. A slice that is all NaN raises NumPy's `ValueError` when it is gathered, with a note naming the line.
- **Casts:** `x.astype(t)` to `float32`, to integers by name, to booleans and to the array's own type, with the type by keyword, with `copy=False` to another type, with the defaults spelled out, `np.astype(x, t)`, a cast at the end of an expression, of a mask that is read on, of a number and of what a pending mask selects are each deferred with NumPy's shape, type and bits. `copy=False` to the array's own type gives the array itself and is made at once, as are another `casting=` or `order=`, a cast to strings and `np.astype` of a number. A type NumPy does not know, a cast its rule forbids and a type given twice raise NumPy's message on the line. The warning of a cast from complex to real comes at the gather and names the line of the cast.
- **`np.partition`:** along the last axis, another axis by keyword and by position, with several positions, with `kth` by keyword, with `axis=None`, with the defaults spelled out, through the stand-in, and with the two best of each column read from it and the gap between them, each is deferred with NumPy's shape, type and bits. A `kth` out of bounds, an axis the array lacks, a `kth` that is not a whole number, an array without dimensions and a `kth` given twice or not at all raise NumPy's message on the line. A write into the result is a deferred write that does not reach the operand. Of what a pending mask selects and with a `kth` that has to be computed it is made at once, and `a.partition(kth)` is still a write made at once.
- **Running totals and searches:** `np.cumsum` over all the elements, along an axis given by keyword and by position, as a method, and on `int8`, whose total is `int64`; `np.cumprod`; `np.searchsorted` as a function and a method, from the right, for one number, with an `ndarray` as the sorted array; a running total of what a pending mask selects and a search for it. Each is deferred with nothing made at once, and gives NumPy's type and bits. An axis the array lacks raises NumPy's message on the line. A resampling written as a particle filter writes it, a running total with its last element assigned and then searched, is one graph. `np.sum(x, (0, 1))` and `np.sum(x, None)`, with the axis by position, give NumPy's result; they raised an `AttributeError` of the layer's own until this test.
- **An array like another:** `zeros_like`, `ones_like` and `full_like` of a pending array leave it pending and count as nothing computed.
- **The stand-in:** arrays it makes are lazy, types and constants are NumPy's own objects, `np.linalg.norm` and `np.add.reduce` work through it, `np.asarray(x)` returns a lazy `x` without computing, `np.asarray(a)` holds an `ndarray` by reference, and `np.array(x)` copies.
- **Nothing is patched:** after `get_lazy()`, `sys.modules["numpy"]` and `numpy.zeros` are the objects they were, and a module that imports the real `numpy` gets plain `ndarray`s from it and lazy arrays from lazy input.
- **`any` and `all`:** as functions and as methods, along one axis and two, by position, with `keepdims`, of masks and of numbers, as the condition of `np.where`, beside another mask and of what a pending mask selects, each is deferred with NumPy's shape, type and bits. An axis the array lacks raises NumPy's message on the line, and `where=` is made at once. `if x.any():` computes the reduction and is counted as a value needed, with no call made at once; a strict array raises `NotGathered` there.
- **A call is kept as written:** `x.sum()`, `x.sum(0)`, `x.mean(axis=1, keepdims=True)`, `np.sum(x)`, `np.max(x, 1)`, `np.std(x, axis=0)` and `np.sum(x, (0, 1))` are each deferred with the arguments given, given the same way, and an axis given twice raises what NumPy raises.
- **Counting:** `np.count_nonzero` of two masks combined, of numbers, along one axis and two, by position, with `keepdims`, through the stand-in, of what a pending mask selects and inside arithmetic is each deferred with NumPy's shape, type and bits, and nothing is made at once. An axis the array lacks raises NumPy's message on the line. On a NumPy that answers the count without an axis with a Python `int`, that call alone is made at once.
- **Housekeeping:** a shared operand is printed once by `explain()`; past the limit of pending nodes everything pending is computed; pending work nobody holds is dropped without being computed.

### What writing a node costs

`uv run python -m prototypes.lazy.programs.build_time` writes each program's graph down with the stand-in and drops it without a gather, so the time is the writing alone. It does not depend on the size of the arrays, which are never read. Nodes are written by the C core ([The C core](#the-c-core)) where it is built, and by Python with `--no-core`. The quickest of 15 runs:

| Program | Operations | In Python | With the C core | Per operation, with the core | NumPy computing it, on 200 × 8 `float64` and 16 × 32 `float32` |
|---|---|---|---|---|---|
| The linear regression, 50 gradient steps | 954 | 1.65 ms | 0.50 ms | 0.53 µs | 0.42 ms |
| The transformer block, 10 in a row | 550 | 1.30 ms | 0.49 ms | 0.89 µs | 1.18 ms |
| A loop that writes into its array (`x[i] = v`, `x += 1.0`, `x[i, 1:] *= 2.0`), 100 turns | 400 | 0.89 ms | 0.29 ms | 0.72 µs | 0.12 ms |
| The mean of the rows a mask selects, 100 masks | 400 | 1.33 ms | 0.44 ms | 1.09 µs | 0.59 ms |

| One line, made and dropped | In Python | With the C core |
|---|---|---|
| `a * b`, `a * 2.0`, `-a`, `a.sum()`, `c[1]` | 1.4 to 1.6 µs | 0.40 to 0.42 µs |
| `c[1, 1:3]`, `c.T`, `a.mean(axis=-1, keepdims=True)` | 1.9 µs | 0.45 to 0.50 µs |
| `np.exp(a)` and `np.sum(a, axis=0)` through the real module | 1.7 to 1.9 µs | 0.60 µs |
| `np.exp(a)` through the stand-in | 2.2 µs | 0.67 µs |
| `selected * 2.0` and `selected.mean(axis=0)`, where `selected` has a length that is not known | 3.2 to 3.3 µs | 0.57 to 0.61 µs |
| `c[mask]`, the mask pending | 4.5 µs | 0.58 µs |
| `c.reshape(4, 3)` | 2.5 µs | 0.66 µs |
| `np.add(a, 2.0, out=x)`, `np.clip(x, -1, 1, out=x)` | 2.9 to 3.2 µs | 0.70 µs |
| `x[mask] = 0.0`, `x[mask, 1:] = 2.0`, `x[rows] = 1.0`, the mask and the positions pending | 3.9 to 4.6 µs | 0.58 to 0.75 µs |

- **A node costs as much as NumPy's own work on a thousand elements.** NumPy multiplies two `float64` arrays of 16 elements in 0.23 µs, of 1,024 in 0.45 µs, of 4,096 in 0.8 µs and of 8,192 in 1.9 µs. With the core, writing `a * b` down takes what computing it takes at about 1,000 elements; in Python, at about 7,000. The regression's whole graph is written in the time NumPy computes it on 200 examples. The first build, which computes each node with the call as written, is therefore slower than NumPy on small arrays and close to it on large ones. That is accepted here; what the graph is for comes after ([What the graph is for](#what-the-graph-is-for)).
- **Nearly all of it would be spent asking, if every node asked.** What a node needs is the same on every turn of a loop, and each answer costs more than the node: NumPy's answer for a shape and an element type, on stand-ins, 4 to 6.5 µs; the line of the user's code, 1.3 µs for each frame of the layer between it and the node; `np.geterr()`, 0.4 µs. A node written by asking all three each time takes 14 to 19 µs. The layer remembers the answers ([What is remembered while a graph is written](#what-is-remembered-while-a-graph-is-written)): found again, they cost 0.25 to 0.4 µs, 0.2 µs and 0.05 µs in Python, and a node takes 1.4 to 2.5 µs. What is left then is the interpreter's own work, which the C core does in 0.4 to 0.7 µs.
- **What is remembered is what would be worked out.** `tests/test_lazy_building.py` holds each memory to the answer found from nothing. Every kind of deferred call gets, the second time, the signature the rules and NumPy give. Calls that Python finds equal and NumPy does not (`axis=1` and `axis=True`, `uint8 + 1` and `uint8 + 300`, a node and an `ndarray`, two byte orders, two `dtype` objects that differ in metadata) are told apart. For each of NumPy's 42 ufuncs of two or more operands, seven element types and every place among the operands, 39 float and complex values give one answer per type of the number, which is what lets `x * rate` be remembered whatever `rate` is. `rules.basic_shape` gives NumPy's shape for 4,000 random indices wherever it answers, and passes on every one NumPy raises for. Each road into the layer names the line that wrote the operation, the second time as the first. Forty random programs of 150 operations each, with writes, masks, views read after a write and errors, give the same nodes, lines, fallbacks, exceptions and values with every memory switched off, with the memories empty, and with them full. Sixteen wrong versions of the memories, each one changed line, were tried against these tests by a script outside the repository; each fails at least one.
- **The core writes the nodes Python writes.** Every test of the layer runs twice, in Python alone and with the core (`tests/conftest.py`): 91 tests each way. Among them the random programs, where the run with every memory off is in Python and the runs with them on are the core's. Of twenty-two wrong versions of the core, each one changed line, nineteen fail a test. Two of the others write a key Python never stores, so the core finds nothing by it and Python takes the call; the third writes `a[(m,)]` for `a[m]`, which NumPy reads alike.
- **Without the core, the floor is what the bare probe takes.** The probe, a class with the operators that writes down the operation and its operands and nothing else, takes 0.3 to 0.5 µs for a node in Python, measured on a busy machine (load average 8 to 10). The core writes a whole node of this design, with its shape, type, line, error settings and its place in the pending set, in 0.41 µs.

### With `install()` in effect

The same machine, with `hotkernels.install()` as it is by default: 43 of 49 kernels reachable. Counts of calls are from `hotkernels.report()`. Nothing was timed.

| What was run | Result |
|---|---|
| Both programs of the table above, lazily, with the stand-in made before `install()` for one and after it for the other | Identical, bit for bit, to the eager programs with `install()` in effect and to the eager programs without it. 1,009 operations deferred, none made at once |
| The regression's kernel calls. One call of the program is served by a kernel, `(w ** 2).sum()` in the penalty, once per step | Eagerly, `sum_f64` took 50 calls. Lazily it took 51: the 50 at the gather and one dry run for the element type, on the first step. Run a second time, the lazy program took 50 |
| `np.where(a > 0, a, b)` through the stand-in and through the real module, `np.clip(a, -1, 1)`, `np.argmax(a)`, `np.searchsorted(table, a)`, `a[a > 0]` and `np.maximum(a, b).sum()`, on lazy `float64` arrays of 100,000 elements | Each result is a lazy array equal to NumPy's, and each call was taken by its kernel: `where_f64`, `clip_f64`, `argmax_f64`, `searchsorted_f64`, `compress_f64`, `sum_f64` |
| `np.argmax(a)`, `np.argmin(a)`, `a[a > 0]` and `a.ravel().sum()` on a lazy `float64` array of 100,000 elements, each deferred and then gathered | Each equal to NumPy's. The kernels for `np.argmax` and `np.argmin` took two calls each, the dry run for the element type and the gather; the one for `x[mask]` took one; the one for `np.sum` two. Written and gathered a second time, each took one: the gather |
| `np.cumsum` on `int64` and `np.searchsorted` on `float64`, 100,000 elements, each deferred and then gathered | Each equal to NumPy's, and each kernel took two calls: the dry run and the gather. A second time, the gather alone |
| `np.round(a, 2)` on a lazy `float64` array, deferred and then gathered | Equal to NumPy's. `round_f64` takes arrays of up to 4,096 elements. At 1,000 elements it took two calls, the dry run for the element type and the gather, where the eager call is one; at 100,000 it took the dry run alone, and the eager call none. A second time: one call at 1,000 elements, none at 100,000 |
| `np.cumsum` on `int32`, a stable `np.argsort`, `a.astype(np.float32)` and `a += b`, on lazy arrays | Each result is NumPy's. No kernel's count moved for these, for the lazy call or for the eager one; in-place adds are served by a loop the binding does not count |

One test runs the regression both ways under `hotkernels.enabled()` and holds that every kernel the eager program reached is reached at least as often by the lazy one. No operation of the transformer block is served by a kernel.

### Ten programs with one changed line

Run once on the same machine, by a script that is not in the repository. Each program was run with NumPy and then with its `np` replaced by the stand-in and nothing else changed, under a recording, and the results were compared. Five were written for the run; five are code the repository already has, run smaller than they come.

| Program | Deferred | Graphs | Largest graph | What it shows |
|---|---|---|---|---|
| Veros ocean model, equation of state (`delivery/apps/vendor/pyhpc/`), a grid of 4,096 points, its inputs taken in with `np.asarray` | 366 | 1 | 366 | A 48-term polynomial as one graph, every operation element by element |
| The Mandelbrot set, 30 iterations on 300 × 240, nothing written in place | 187 | 4 | 179 | A loop as one graph |
| Heat on a plate, 60 steps written into a slice | 785 | 5 | 261 | Twenty steps between two reports of the temperature are one graph |
| Veros ocean model, isoneutral mixing, the same grid | 698 | 19 | 119 | Its arrays are the user's, taken in with `np.asarray`: 20 assignments into them are made at once |
| `delivery/apps/mlp_train.py`, one epoch on 1,280 examples | 516 | 11 | 50 | A step is one graph, ended by `float(loss)` |
| Karpathy's `min-char-rnn.py`, 3 iterations, with its import line as the one line changed | 3,245 | 27 | 2,059 | Three iterations of training are one graph. The rest are the twenty characters it samples, each needing a value |
| `delivery/apps/particle_filter.py`, 20 steps | 1,327 | 24 | 140 | A step with its masks and its resampling is one graph, ended by a test (`if ess < ...:`) |
| k-means, four clusters of 500 points | 99 | 10 | 19 | An iteration with a mask per cluster is one graph, ended by `np.stack` of the centres and a test for convergence |
| PCA of 2,000 × 12 | 25 | 3 | 11 | `np.linalg.eigh` and `np.argsort` made at once |
| A Monte Carlo option price, its data from `np.random.default_rng()` | 13 | 2 | 10 | The paths, their running total and the payoff are one graph, ended by the price being formatted |

- **All ten give NumPy's results, bit for bit,** without a line of theirs changed but the one that says what `np` is.
- **A loop is one graph.** The recurrent network writes into an array at nearly every line of its inner loop, and its three iterations of training are one graph of 2,059 operations. The perceptron's step is one graph of 50. Over the ten programs 40 writes are still made at once: 20 assignments into the ocean routine's arrays, which are the user's, and 20 assignments of a tuple of two lazy arrays, a value whose shape the layer does not look into.
- **A mask does not end a graph.** The particle filter caps the speed of its particles through a mask at every step, and at most steps the mask selects none of them: the selection, the division and the assignment back are nodes, and a step is one graph of up to 140 operations. k-means takes the mean of the points of each cluster through a mask.
- **What ends a graph now is a value being needed.** 85 times over the ten programs: a test (`if shift < tolerance:`), a loss turned into a `float`, a number formatted for a line of output, an array handed to a function of NumPy's that the layer does not see into. Those are the program asking, and no rule of the layer removes them.
- **7 calls are still made at once:** `np.stack` (4), and `np.concatenate`, `np.linalg.eigh` and `np.argsort` once each.
- **A long function needed a group of its own.** The equation of state is one function of a hundred lines that are not alike, and it opened as 366 boxes. A call of many members is now cut into stretches of lines ([lazy-recording-design.md](lazy-recording-design.md#groups)).

## Design

### Where it lives

A subpackage of Python with one optional C source, `hotkernels/src/hotkernels/lazy/`, and the names of [the whole interface](#the-whole-interface) exported from the package. The prototype has the same modules:

| Module | Holds | In the prototype |
|---|---|---|
| `graph.py` | The node, the list of pending nodes, the lock, `NotGathered` | [graph.py](../prototypes/lazy/graph.py) |
| `rules.py` | Which calls are deferred, and the shape and element type of each | [rules.py](../prototypes/lazy/rules.py) |
| `evaluate.py` | Computing a node, a set of nodes, everything pending | [evaluate.py](../prototypes/lazy/evaluate.py) |
| `array.py` | The lazy array: operators, the two protocols, the refusals, `gather()` | [array.py](../prototypes/lazy/array.py) |
| `_core.c` | The quick road for writing a node, in C ([The C core](#the-c-core)) | [_core.c](../prototypes/lazy/_core.c), built by [build_core.py](../prototypes/lazy/build_core.py) |
| `namespace.py` | The stand-in for the `numpy` module | [namespace.py](../prototypes/lazy/namespace.py) |
| `explain.py`, `templates/explain.txt.j2` | The text of a pending graph | [explain.py](../prototypes/lazy/explain.py) |
| `record.py` | The graph as an object, and the recording mode | [record.py](../prototypes/lazy/record.py) |
| `report.py`, `templates/report.txt.j2` | What was computed without being asked, ranked | [report.py](../prototypes/lazy/report.py) |
| `view.py`, `templates/graph.html.j2`, `graph.js`, `graph.css` | The program's code and its graphs as a page | [view.py](../prototypes/lazy/view.py), [templates/](../prototypes/lazy/templates) |

`graph` knows nothing of the others; `rules` and `evaluate` work on nodes and never see a `LazyArray`; `array` is the only module the user's calls enter. `record` hears of nodes and computations through one hook that `graph` offers, and `view` sees only what `record` wrote down. Whatever computes a graph differently later replaces one function of `evaluate`.

The prototype stays outside the package on purpose. Inside `hotkernels/` it would be shipped to users and would read as the implementation; in `prototypes/` it is a study object that the implementation is free to depart from.

### The node

| Field | Meaning |
|---|---|
| Number | A count that rises with every node. Operands are always older than their consumer, so ascending order is a valid order of evaluation, and it is the order the eager program ran in |
| Operation | The callable the user wrote: a ufunc, a function such as `np.sum`, a method such as `x.sum`, or indexing. Empty once the node is computed |
| Operands | Other nodes, `ndarray`s held by reference, and every other argument exactly as written: which were given, by position or by keyword, and Python numbers as Python numbers. NumPy's rule for a Python `2.0` (it does not widen a `float32` array) is then applied by NumPy, and a call reaches the code that serves that spelling ([With `hotkernels.install()`](#with-hotkernelsinstall)) |
| Shape, element type | Worked out when the node is made |
| Kind | Element by element, matrix product, reduction, view, indexing by arrays, ordering, positions of what is not zero, running total, search, or a write: which rule deferred it. What a fusible group is read from, and what the page colours by |
| Strict | Whether the array refuses to give its values without `gather()`. True for an array a strict stand-in made, and for anything computed from a strict array |
| Exposed | Whether the memory of the value may be known outside the layer. A write into such memory is made at once ([Writes](#writes)) |
| Written at | File and line of the user's code, and that module's record of warnings already shown |
| Error state | The `np.errstate` in effect when it was written |
| Value | Empty until computed. Then the operands are dropped |
| Failure | The exception, if computing it raised |

A node written on a line that wrote one before shares that line's record, and nodes written under the same `np.errstate` share its record; neither is changed once made. A node also carries one number that stands for its shape and element type together, which is what the answers of the next section are remembered by.

A lazy array is a handle on one node. A leaf is a node born with its value. Nodes refer to their operands and nothing refers to a node but its consumers and the lazy array a user holds, which is why an intermediate nobody holds is freed when its consumer has been computed.

### Shape and element type without computing

- **The shape comes from a rule,** one per kind: NumPy's `np.broadcast_shapes` for element-by-element operations; the matrix product's rule; the axes removed for a reduction; the operand's shape for a running total along an axis, and its number of elements without one; the shape of what is looked for, for a search. For a view, the operation is made on a stand-in that has the operand's shape and no memory (`np.broadcast_to` of one zero), which NumPy answers without touching data. Indexing by arrays makes a new array, which a stand-in of the operand's type would pay for in memory; there the stand-in has an element type of no size, and NumPy gives the shape, and its own error for an index out of bounds, for no memory at all. `np.partition` is asked the same way: NumPy checks `kth` and the axis against the stand-in and raises its own error on the line. It then walks elements of no size, a few nanoseconds each, which the first build accepts.
- **A lazy array in an index is stood in for** while NumPy works the shape out: an array of integers by zeros of its shape, which are in bounds whatever the array indexed, and a mask by one that selects nothing, which NumPy checks against the array's shape.
- **A length that is not known is a mark in the shape.** NumPy's answer for a mask that selects nothing has one length of zero; the mark of that mask's node is put there, so two selections by one mask carry the same mark. The positions of what is not zero in a node carry that node's mark, and an index of such positions is stood in for by no positions, with the mark put where NumPy's answer has its zero. A rule treats it as a length like any other where its answer holds for every length: broadcasting it with itself, with 1 or with a number, a product, a reduction, moving axes. The call is made at once where that fails: when the rule does not accept the mark as it stands, when two marks would be in one shape, for part of the axis the mark is on, and for `reshape` and `ravel`, whose shape is arithmetic on lengths. When the node is computed, its shape becomes the real one. The prototype's mark is an integer larger than any array this machine can hold, so that NumPy's own shape arithmetic carries it; the package is to have a type of its own for it.
- **The element type comes from NumPy itself:** the real call is made on arrays with one element per dimension, with the Python numbers as written, and with what is not an operand (an axis, an element type) as written too. That costs a microsecond or two and cannot disagree with NumPy about promotion. The answer is remembered, so a loop pays for it on its first turn ([What is remembered while a graph is written](#what-is-remembered-while-a-graph-is-written)). For a method such as `x.sum()` the dry run uses NumPy's function of the same name, `np.sum`.
- **An error is NumPy's.** When the rule finds the shapes do not fit, the real call is made on the stand-ins without memory, and NumPy raises its own exception with its own message before it reads anything. If NumPy does not raise, the rule was wrong to doubt, and the call is made at once.
- **When in doubt, do not defer.** A call the rules do not recognise in every detail is made at once. The layer never guesses a shape or a type.

### What is remembered while a graph is written

A loop writes the same operations on every turn, and each asks what the turn before asked. The layer remembers the answers. Each memory is an exact one: it gives the answer that asking again would give, or it gives nothing and the question is asked.

- **The answer for a call's shape and element type.** It depends on the operation, on the shape and element type of each array among the arguments, and on the other arguments as written, and on nothing else: a stand-in has an operand's shape and type and none of its values. Those are written down as a key, and a call whose key was seen before gets the answer worked out then.
  - *An array* is written as one number for its shape and its `dtype` object. The object, not what it equals: two `dtype` objects that compare equal can differ in metadata, which NumPy's answer carries. A node and an `ndarray` of one shape and type are written differently, since a rule may take one and refuse the other.
  - *An argument that is no array* (an axis, a shape, an element type, `keepdims`) is written with its type and its value, item by item for a tuple, because `1`, `1.0` and `True` are equal to Python and three different arguments to NumPy.
  - *A number that is an operand of a ufunc* is written with its type alone if it has a fraction, Python's or NumPy's: NumPy 2 takes the result's type from the operands' types, and no value of a float makes a ufunc raise on zeros with the warnings off. `x * rate` is one key whatever `rate` is. A whole number is written with its value, which decides whether NumPy raises: `uint8 + 300`, an integer to the power −1.
  - *Anything else is not written down,* and the call is answered as if nothing were remembered: a list, a `dtype` object as an argument, a subclass of any of these types. So is a call that is made at once or that raises: it is looked at again each time.
- **A length that is not known, under a placeholder.** Its mark is a different integer for every mask, so a call on what a mask selects would never be seen twice. The key is written with a placeholder for each mark, numbered in the order they are met, and the answer is kept with the same placeholders in its shape; a later call gets it with its own marks put back. That is sound because a rule treats a mark as a length equal to itself and to nothing else. One thing breaks it: NumPy refuses a size past 63 bits, and for an array some eight million wide that limit falls between one mark and the next. The answer is therefore remembered only for calls whose arrays have at most 65,536 elements beside the length that is not known, where no mark can reach the limit. An index that holds pending arrays is remembered the same way, with the mark of its mask as a placeholder.
- **The shape of a plain index is worked out, not asked for.** `x[i]` has another `i` on every turn, so its answer is of no use for the next, and NumPy's answer costs 4 µs. For an index of Python integers, slices of them, `None` and `...`, the layer works the shape out in 0.1 µs, by Python's own arithmetic for slices, which is NumPy's. It answers only where it is sure: an index out of bounds or too long, a step of zero, a NumPy integer, a slice that holds anything but integers go to NumPy, which answers or raises its own error. An assignment `x[i] = v` through such an index is known to fit when `v`'s shape broadcasts to the shape selected; any other is asked of NumPy as before.
- **The line of the user's code.** It follows from the code object that is running and how far into it the frame is, so it is found once per place in the program. The module's name and its record of warnings are looked up again each time, since a program may replace either and one code object may run under two sets of globals. The methods the user calls pass their caller's frame down, so the stack is not walked through the layer's own frames; whether a file is the layer's or NumPy's is decided once per file name, so a name that is not a path (`<stdin>`) is judged against the working directory in effect the first time it is seen.
- **The error settings.** NumPy keeps them in one record that it replaces, never changes, when `np.seterr` or `np.errstate` changes them. The dictionary `np.geterr()` gave for a record is right for as long as that record is the one in effect, so it is asked for again only when the record is another.
- **Whether an in-place operator's types fit** (`x += y`, a ufunc with `out=x`), by the same key as a call: the operator, the target and the operands. Only that it fits is remembered; a write that does not fit raises, or is made at once, every time.
- **Whether an assignment through an index of pending arrays fits** (`x[mask] = v`, `x[rows] = v`): NumPy's answer on stand-ins depends on the shape assigned into, the index as a call's key writes one down, and the value's shape with a placeholder for a length that is not known, and on no array's values.

Each memory is bounded: when it is full it is emptied and starts again, so a program that makes shapes, types or code as it runs cannot grow one without end. A number that stood for a shape is never given out twice, so a node that still carries one from before means what it meant.

What is left of a node's 1.5 µs is the interpreter's own work, in these shares for `a * b`: finding the line again, 0.2 µs; the key and the look-up of the answer, 0.25 µs; the node, 0.25 µs; the lazy array and its cell, 0.17 µs; adding the node to the set of pending nodes, counting it and reading the error settings, 0.25 µs; the operator's method and the pass over its arguments, 0.35 µs. That is some ten calls of Python functions and five objects: the node, its tuple of operands, its weak reference in the pending set, the lazy array and the cell. None of the six shares is a question that could be asked less often; each is work the design gives every node. Going below it means doing that work outside the interpreter, which is the next section.

### The C core

`prototypes/lazy/_core.c` is the hot path in C: about 2,400 lines that write one node of the graph when every answer it needs is remembered. It takes `a * b` from 1.5 µs to 0.41 µs.

- **It is the quick road of functions that exist in Python, and nothing else.** `graph.deferred`, `array._try_defer`, `array._plain_item`, `array._select` for an index that holds pending arrays, the in-place operators, `x[i] = v` through a plain index and through one that holds pending arrays, a write through `out=`, the view methods, and the stand-in's `np.name` and `np.name(...)` each have a road in the core for the calls where everything the Python function would look up is already in the memories of the section above. The Python functions are the definition. A node the core makes is, field for field, the node they would have made.
- **It reads the memories and never writes them.** The core builds the key Python builds and looks it up. Where it finds nothing, or meets anything it is not sure of (an argument it does not know how to write down, a subclass, a collection that may hold a lazy array, a pending set at its limit, an array that has its values and is read by one element), it returns a marker and the Python function takes the call unchanged: works the answer out, raises NumPy's error, makes the call at once, and fills the memory, so that the next such call is one the core can take. A mistake in a key the core writes therefore costs time and not correctness, unless it makes two calls look alike that are not; the tests of [What writing a node costs](#what-writing-a-node-costs) are aimed at that.
- **The objects are the Python classes' own.** `Node`, `LazyArray`, the cell, the view step and the write are declared with `__slots__`, so an instance is a C structure of object pointers at offsets the class's member descriptors give. The core reads the offsets once and makes an instance by allocating it and storing its fields. No `__init__` runs, so each constructor in the core stores what the `__init__` stores and says so. For this the node of a deferred write, of a write through `out=` and of an index of pending arrays calls an object of a small class where it called a function made on the spot, in Python as in C.
- **It stands on the class where the methods stood.** A `Road` is a callable the core defines; `array.use_core(core)` puts one in place of each operator, each reduction, `__array_ufunc__`, `__array_function__`, `__getitem__`, `__setitem__`, the in-place operators, `reshape`, `transpose`, `swapaxes`, `ravel` and `T`, and keeps the Python method as the road's fallback. `a * b` then goes from the interpreter's `BINARY_OP` to C without a Python frame in between, which is also why the user's line is the frame the core is called from. `use_core(None)` puts the methods back.
- **What a test or the layer rebinds is read each time:** the observer, the limit of pending nodes, the error settings' record. So a recording, a patched limit and `np.errstate` behave with the core as without it.
- **It is optional and says which version it is.** `python -m prototypes.lazy.build_core` compiles it with clang into the layer's directory; the layer takes it if it is there and runs the same without it. The core carries a version number that `array.py` asks for, so a module built from an older source is left unused.
- **What still goes the Python way, and why it cannot be the core's.**
  - *The first call of each kind.* It takes 6.4 µs for `a * b`, of which 3.1 µs are NumPy's own calls and the warning filter around them: the broadcast, the call on two arrays of one element, the type of its result. Those cannot be skipped. The element type and the error are NumPy's by definition here, and getting them without asking NumPy means writing NumPy's promotion rules a second time, which the design refuses ([Shape and element type without computing](#shape-and-element-type-without-computing)). The other half is the rules in Python, which C could shorten. It is paid once per operation and operand types: 21 times for the regression's 954 operations, and never again however long the loop runs.
  - *An index that holds a list, an `ndarray` or a lazy array that has its values.* NumPy checks such values against the bounds and raises on the line. The values can differ on every call, so NumPy's check has to be made on every call: 2.3 µs of the 5.7 µs such an index takes. There is no remembered answer that would be right.
  - *A call made at once, and a write made at once.* They are not written into a graph: they compute, after everything pending. On 200,000 elements `np.sort` takes 210 µs through the layer and 207 µs in NumPy.
- **What it leans on.** CPython's layout of an object with slots, its frames (`PyEval_GetFrame`, the offset of the running instruction), its context variables and weak references. It compiles against the headers of Python 3.11, 3.12 and 3.14 and was run on 3.12. It assumes the lock the interpreter holds; free-threaded Python is not supported, as for the rest of the package.

### Computing

- `gather` collects the nodes its arrays need that have no value, with a stack and not by recursion, and computes them in ascending number: the node's own call, on its operands' values, under its error state.
- **Warnings are recorded and issued again.** Left alone, NumPy's warning would name a line inside the layer. Each node's call runs inside `warnings.catch_warnings(record=True)`, and what NumPy issued is issued again with NumPy's own message and category, at the file and line the node was written at, through that module's record of warnings shown, so Python shows it once per line as it would for the eager program. Recording changes the process's warning filters for the length of the call, which is safe only because every evaluation holds the layer's lock.
- An exception gets a note naming that line (`add_note`), is kept on the node, and is raised.
- The first build therefore makes exactly the calls the eager program makes, spelled the same way, on the same operands. Whatever computes a graph differently later is held to the same results by the check below.

### Pending work and writes

Every node without a value is in one list, held weakly.

**A lazy array stands on a cell.** A cell is one array as NumPy knows it: memory that several lazy arrays may be looking at. An array of its own has a cell to itself, and the cell holds the node of its contents. A view shares the cell of the array it is a view of, and keeps the view operations that lead from the cell's contents to it: `x[1:3].T` is the cell of `x` and two steps.

- **Reading a lazy array** takes the cell's node, for an array of its own. For a view it takes the view of the cell's node, as deferred nodes; these are made again when a write has given the cell a new node since the view was last read.
- **A deferred write gives the cell a new node.** Its operands are the cell's node as it was, and what the write reads. Its value is a copy of the old contents; the view steps of the array written into are applied to the copy, and the write is made into that: `view += v`, `view[index] = v`, `ufunc(..., out=view)`. The array and all its views read the new node from then on.
- **Which views are views.** Indexing by integers, NumPy's integers among them, and slices, `transpose`, `swapaxes`, `expand_dims` and `squeeze` always share memory, and are tracked as steps. An index that leaves no dimension gives a number, a copy. `reshape` and `ravel` share memory when the layout allows and copy when it does not, which is not known for an array not yet computed: the result gets a cell of its own, and while it lives neither it nor the array it came from has a write deferred. A row taken by a position still pending is treated the same way, since NumPy gives a view for a number and a copy for an array without dimensions.
- **Exposure is a mark on a node.** It is set when the node is made for an `ndarray` the user handed in; when `gather()`, `np.asarray` or an attribute hands the array out; and on an argument of a call made at once whose result may overlap it (`np.may_share_memory`), with the result. A write into a cell whose node is exposed is made at once.
- **A write made at once** computes the pending list in ascending number and then lets NumPy write. In-place operators are handed to NumPy's in-place operator on the computed value: the lazy array is returned unchanged in identity, so every name for it sees the write.

**Which writes are deferred:** the in-place operators `+= -= *= /=`; `x[index] = v`, also through an index that holds lazy arrays still pending: arrays of integers, or one mask, or positions whose number is not known, when what is assigned is a number or was made from a selection by the same mask or the same positions; an index the layer would not defer a read through is not deferred for a write either; a ufunc with `out=x`; `np.clip(..., out=x)`; `np.round(..., out=x)` and `np.around(..., out=x)`. **Which are made at once whatever the array:** `np.copyto`, `np.putmask`, `np.place`, `np.put`, `np.fill_diagonal`, `ufunc.at`, and the methods `sort`, `fill`, `put`, `partition`, `resize`, `itemset`, `setfield` and `byteswap`. A NumPy function that writes into an argument and is on neither list would be made at once without the pending work being computed first; the lists have to be checked against NumPy as part of the build.

One lock guards every evaluation. Lazy arrays can be used from several threads; their work is done one gather at a time. Writing a node does not take the lock: a node joins the set of pending nodes in one step, the set is copied in one step when a computation reads it, and a node made while another thread computes is not part of that computation either way. Free-threaded Python is not supported, as for the rest of the package.

### Values that leave without `gather()`

Every road by which a value can leave a lazy array without `gather()` goes through one function. It is given what asked (`float()`, `Conversion to an ndarray`, `np.add`), whether an array is handed out, and the nodes whose values are needed.

- **For a strict array it raises `NotGathered`** and computes nothing.
- **Otherwise it counts and logs,** then computes: everything pending first if an array is handed out or an `ndarray` is about to be written, and then the nodes asked for. The computation is told what triggered it, so a recording has it as a graph of kind `needed`.
- **A number, a truth value or a string asked of arrays that have their values is given without counting or logging.** Nothing is computed for it, so no graph is cut and a recording gains no graph. An array handed out is counted whether or not it has its values: what continues from it is plain NumPy.
- **The roads are:** the conversions Python asks an object for (`__float__`, `__int__`, `__complex__`, `__bool__`, `__index__`, `__contains__`, and `__format__` with a specification); `__array__`, which is how NumPy and every library built on it ask; a ufunc or a function whose `out` or target is a plain `ndarray`; an attribute of `ndarray` that is not a method; and a call made at once whose answer is a plain Python value.
- **It is told to the log, to the summary and to a recording** ([What the layer says](#what-the-layer-says)).
- **Iteration is not one of the roads.** `for row in x:` gives `x[0]`, `x[1]`, and so on, each a deferred view: the length is the shape's, and no value is needed unless a mask decides that length. Over one dimension of an array that has its values each is an element read on the spot ([The deferred set](#the-deferred-set)), which is still a lazy array.

### What the layer says

Three things happen without being asked, and they are not the same to a reader:

| Word | What happened | Logger | Level |
|---|---|---|---|
| `eager` | A call outside the deferred set: NumPy made it at once, and the result is still lazy | `hotkernels.lazy.eager` | `INFO` |
| `write` | A write into a lazy array computed everything pending first | `hotkernels.lazy.cut` | `DEBUG` |
| `needed` | A value was needed without `gather()` and was computed | `hotkernels.lazy.cut` | `DEBUG` |

- **Which is seen is ordinary logging configuration.** `logging.getLogger("hotkernels.lazy").setLevel(logging.INFO)` shows the eager calls and nothing else, which is the list of what the deferred set should cover next. `DEBUG` adds the places where a graph was cut. Either logger can be set by itself. Nothing is shown until one of them is turned on.
- **A line is short and each line of code gets one:** the word, the call, the file and the line. The layer keeps the set of what it has reported.
- **Every occurrence is counted by line,** whether or not anything is logged, and `hotkernels.lazy_report()` ranks the counts: the call a program makes six hundred times at once is at the top. A recording keeps the same counts for its block (`recording.report()`). For k-means on four clusters:

  ```text
  Made at once by NumPy: not in the deferred set
         4  np.stack                    kmeans.py:18 × 4
         1  np.concatenate              kmeans.py:12

  A graph computed early: a write into a lazy array
         none

  A graph computed early: a value needed without gather()
         4  A truth test (if, while, and, or, not)  kmeans.py:21 × 4
         1  Formatting with '.2f'       kmeans.py:25
  ```

- **Only a line that computed is counted as a value needed.** `float(x)`, `if x[i]:` or `names[i]` on an array that has its values cuts no graph, and the summary is the list of where graphs were cut. So a Python loop over the elements of an array is counted once, at the read that computed the array, and not at all when the array came with its values: the summary counts the arrays a loop made the layer compute, not the elements it read. Counting each read under a line of its own ("22 elements of one array") was the other way to say it; it would name lines where nothing can be made lazier.
- **More can go under the same name.** How long a graph took or how much memory it held would be further loggers of `hotkernels.lazy`, chosen the same way. None is designed.

All of this is for a person reading a run, and each part has its lever: which logger and which level; the summary instead of the stream; how much of the program's code a page carries ([lazy-recording-design.md](lazy-recording-design.md#how-much-code-a-page-carries)); what a page shows folded.

### The stand-in for the module

`get_lazy()` returns an object that looks names up in `numpy`. It holds a reference to the module and changes nothing in it.

- A type or a constant (`np.float32`, `np.pi`, `np.newaxis`, `np.errstate`) is returned as it is.
- A submodule (`np.linalg`, `np.random`) is returned as another stand-in.
- An array a function makes when it is given no array (`np.zeros(3)`, `np.arange(10)`, `np.array([1, 2])`) is the layer's own: nothing outside can know its memory, so a write into it can be deferred.
- `np.random.default_rng()` and `np.random.RandomState()` return objects of NumPy's, whose methods the stand-in would never see. They are returned inside a stand-in of their own, so that `rng.standard_normal(n)` is a lazy array, and the layer's own.
- A function or a ufunc is returned wrapped: `ndarray` arguments become lazy arrays, the real function is called, NumPy's protocols route it to the lazy arrays, and an `ndarray` that comes back is made lazy. So `np.zeros(3)` is lazy, and `np.exp(x)` is the same call it is with the real module. A wrapped ufunc keeps its methods, so `np.add.reduce` works.
- `np.array`, `np.asarray` and their like are answered by the stand-in itself. They have no protocol through which a lazy array could answer, so the real ones would ask it for its values. `np.asarray(x)` returns a lazy `x` as it is, without computing; `np.asarray(a)` takes an `ndarray` in by reference; `np.array(x)` copies, as in NumPy.

### What an unknown attribute gets

A lazy array answers a method it does not define with `ndarray`'s, made at once: `x.clip(0, 1)`, `x.argsort()`, `x.repeat(2)`. Two limits keep that from becoming a way out:

- **No double-underscore name is passed on.** `__array_interface__` and `__array_struct__` would hand NumPy the memory behind the layer's back; see [Strict arrays](#strict-arrays).
- **An attribute that is not a method is the values themselves:** `x.flat`, `x.real`, `x.data`. The array is computed, with everything else pending, and the attribute of the result is returned; a strict array raises.

### Checking it

A lazy layer is trusted for the same reason a kernel is: a check that does not depend on reading the code. `uv run delivery lazy-check` runs:

1. **Programs both ways.** The two examples, and the applications in `delivery/apps/` that need only NumPy (`mlp_train`, `min_char_rnn`, `veros_ocean`, `particle_filter`), each with its module's `np` replaced by the stand-in and nothing else changed. Results, exceptions and warnings are compared as `delivery check` compares them, and what each run logged is kept: it is the list of what the deferred set should cover next.
2. **Random programs.** Sequences of operations from the deferred set and outside it, over random shapes, element types and special values, with gathers and writes at random points, run eagerly and lazily. Compared: type, element type, shape, bits, the exception's type and message, the warnings and their lines, and the final state of every `ndarray` the program was given.
3. **Each supported NumPy,** in CI, since the protocols are NumPy's.
4. **With `install()` in effect.** The same programs once more: results compared with the eager program under the same installation, and every kernel the eager program reaches reached at least as often.

The prototype's tests are the first and fourth parts of this in small.

## What the graph is for

Not part of this build, and no gain is claimed. It is here so that the first build is shaped for what comes after.

- **A fusible group can be read off the graph:** connected element-by-element nodes of one output shape, optionally ending in a reduction. In the regression above, `(X @ w - y) ** 2` then `mean` is one group after the product; in the transformer block the eight operations of GELU are one. The page of the [debugging tools](#debugging-tools) is where such groups would first be shown, as boxes.
- **A group can be computed three ways,** and the graph does not care which: node by node with NumPy, as now; with NumPy's own loops a cache-sized block at a time, which removes passes over memory and keeps NumPy's bits; or with one kernel for the whole expression.
- **A fused kernel fits the existing contract unchanged.** Its `spec.py` already declares its oracle as a NumPy expression (`np.where(c, a, b)` for `where_f64`). Run that oracle on lazy arrays and the result is the graph the kernel computes: the specification is its own pattern, and the judge checks a fused kernel as it checks any other.
- **The graph carries facts a hook has to read memory for.** A mask that came from a comparison holds only 0 and 1; a temporary the graph owns overlaps nothing. The binding scans for both today.
- **The graph says which chains a program runs,** with their sizes: the "way to identify which chains are worth fusing" that [core-design.md](core-design.md#later) says fusion first needs.

## Other ways in

Two more ways to get lazy arrays into a program. Neither is in the first build, and both are as safe as `get_lazy()` in the respect that matters: nothing is patched, so neither depends on what was imported first.

### The stand-in as a module that can be imported

```python
import hotkernels.numpy as np
```

The object `get_lazy()` returns, reachable by an import statement. Other array libraries offer themselves in place of NumPy's module the same way: `import jax.numpy as jnp`, `import autograd.numpy as np`, `import cupy as cp`.

- **It gives what an assignment cannot:** `from hotkernels.numpy import exp, zeros` and `import hotkernels.numpy.linalg as la`.
- **It can give the editor back.** A stub file beside the module can tell editors and type checkers that its names are NumPy's, so `np.` completes and calls are checked. Not tried.
- **It reads as what it is:** another module, not a change to NumPy. Nobody takes an import of a different module for a patch.
- **It costs a second spelling of one thing,** beside `get_lazy()`, and a module named `numpy` inside the package.

### A decorator on one function

```python
@hotkernels.lazy
def step(w, X, y):
    r = X @ w - y
    return w - 0.05 * (X.T @ r)
```

The function's `ndarray` arguments enter lazy, and what it returns is gathered before the caller gets it.

- **Nothing lazy leaves the function.** The caller passes `ndarray`s and receives `ndarray`s, so no type changes outside it, no `gather()` is written anywhere, and nothing outside has to know.
- **It suits a hot function in a program that otherwise stays as it is.**
- **Its limits:** the graph ends at every return, so a loop that calls the function gets one graph per call; and arrays the body makes with its module's own `numpy` are eager, as in any module that has not taken the stand-in.
- **It is sugar over this layer,** a few lines once the layer exists.

## Considered and not proposed

| Option | Why not |
|---|---|
| Replace the `numpy` module for the whole process: put the stand-in in `sys.modules`, or patch NumPy's own functions | It fails twice. Import order: a module that imported NumPy earlier holds the real module, and `from numpy import zeros` holds the real function for good, so the patch reaches some of a program and not the rest, depending on what was imported first. And it is unsound even in the right order: every library, and NumPy's own Python code, would be handed arrays without memory, in code that never asked for them |
| Make real `ndarray`s lazy, process-wide, as `install()` is process-wide | Unsound: an `ndarray`'s memory is public C API, and anything may read it without calling something a hook could see ([gradient-step-findings.md](gradient-step-findings.md#can-a-hook-fuse-the-statement)). Laziness has to be a type of its own, in code that asked for it |
| A subclass of `ndarray` | The same: a subclass has a data pointer that must hold values at all times |
| A function that wraps one array, `lazy(a)`, with `import numpy as np` kept | Every array has to be wrapped where it enters, and `np.zeros(3)` stays eager. The stand-in's `np.asarray(a)` takes an existing array in, in the one place it is needed |
| Raise wherever a value is needed, with no way to compute | A real program needs a value every few lines and would have to be rewritten before its first graph could be looked at ([Why it computes instead of raising](#why-it-computes-instead-of-raising)). Raising is kept as a switch |
| One switch for strictness for the whole process | A flag on the array needs nothing global, and lets one part of a program be strict while a library it calls is not |
| Log every time a line computes, not once | A loop of ten thousand steps would log the same line ten thousand times. The counters and a recording have every occurrence |
| Compute everything pending at every write into a lazy array | It is always right, and it makes a loop a row of graphs of a line each: over ten programs, 1,133 writes and no graph of a training loop above 15 operations. It is kept for memory the layer cannot be sure of ([Writes](#writes)) |
| Refuse writes into lazy arrays | The commonest idiom of a training loop, `w -= step`, would have to be rewritten |
| Refuse calls outside the deferred set as well | Nearly every real program would stop at its first such call until the set grew. They compute without letting a value out, so they are made at once and counted |
| Keep every shape known, and make `x[mask]` at once when the mask is pending | Every shape error stays on its line and no rule has to think about a length. But a program that selects by a mask inside its loop then computes at every step: the particle filter and k-means had graphs a few lines long ([A length that depends on values](#a-length-that-depends-on-values)) |
| Defer the other operations whose shape depends on values (`np.unique`, `np.where` with one argument) | The same mark would serve, as it does for `np.nonzero`, `np.flatnonzero` and `np.argwhere` since a program called them. No program has called these two, so they are made at once until one shows the need |
| Give elements that have their values from iteration only, and leave `x[i]` deferred | `flags[i]` and `valid[r, c]` inside a loop are not iteration, and they were half of what one program read element by element. The rule on the index covers iteration with it ([The deferred set](#the-deferred-set)) |
| Read one element of a pending array on the spot as well | It would compute the array on a line that asked for no value, and a strict array would have to refuse an index |
| Dask, JAX or numexpr | Each has semantics of its own: chunked and parallel arrays, another array library with its own type rules, expressions as strings. The contract here is NumPy's results bit for bit, so that kernels judged against NumPy can compute the graph |
| The array API standard (`__array_namespace__`) as the way in | It serves libraries written for it. Existing NumPy code calls `np.exp(x)`, which the two protocols above serve. It can be added beside them |

## Exit criterion

1. **A loop that writes into its arrays in place is one graph** until a value is needed, with NumPy's results.
1. **Two programs run with one changed line:** `examples/lazy_regression.py` and `examples/lazy_attention.py`, with results identical to eager NumPy, and no operation in either made at once.
2. **`uv run delivery lazy-check` passes** in all four of its parts.
3. **Shape and type errors are NumPy's and are raised on the line,** and value-dependent warnings name the line that wrote the operation, for every deferred kind.
4. **A value needed without `gather()` is computed, logged once per line and recorded,** with everything pending computed first where an array is handed out. On a strict array each of those raises `NotGathered`, and no road hands NumPy the memory unnoticed.
5. **`get_lazy()` works with and without `install()`.** With it, results are the eager program's under the same installation and every kernel the eager program reaches is reached. `delivery check` and NumPy's test suite are unaffected by `get_lazy()`, since it changes nothing in NumPy.
6. **The graphs of a block of code can be recorded, written as JSON and opened as a page,** and the regression's 50 steps open folded on one screen ([lazy-recording-design.md](lazy-recording-design.md), [lazy-graph-page-design.md](lazy-graph-page-design.md)).
7. **The user-facing pages say what a user must know:** `docsite/pages/start.md` and `hotkernels/README.md` carry the examples, the table of types and the six differences.

No criterion is about speed.

## Open questions

Each has a behaviour in the prototype, so that the prototype is defined.

**An `ndarray` read by a pending expression: held by reference, or copied?**

```python
a = loader.read()        # [0., 1., 2.]
y = np.asarray(a) * 2
a[0] = 5
y.gather()               # by reference: [10., 2., 4.]      copied when met: [0., 2., 4.], as eager NumPy
```

- *By reference* (the prototype): nothing is copied, and the user has a rule to remember: gather before writing directly to an `ndarray` a pending expression has read. A module that never took the stand-in can break that rule without knowing the layer exists, which makes this the one place a partly lazy program can differ from NumPy. It is also why an array handed out without `gather()` takes everything pending with it.
- *Copied when first met:* no rule to remember and no way to get a result eager NumPy would not give. Every plain `ndarray` is copied each time an expression meets it, so a loop that reads the same large `ndarray` in every step copies it in every step unless the user takes it in once.
- A third possibility, not designed: by reference, with a check mode that remembers a fingerprint of each input and raises at the gather if it changed.

**What a deferred write costs, and which further writes to defer.**

- *A copy per write* (the prototype): right, and more work than NumPy's write in place. Writing into the old contents when nothing else reads them belongs to whatever computes a graph.
- *`reshape`:* whether its result shares memory could be learnt without computing, by asking NumPy to reshape an array of the right layout and no memory with `copy=False`. Not tried; until then a live `reshape` sends writes the other way.
- *The functions and methods that write* (`np.copyto`, `x.sort()`, `x.fill(v)`): each could be a node in the same way. None is.

**What `gather()` returns, and what the lazy array is afterwards.**

- *The array itself, and the lazy array keeps it* (the prototype): no copy. Its memory is then known outside the layer, so the next write into that lazy array is made at once ([Writes](#writes)). Writing into the returned `ndarray` changes what the lazy array holds, as it would in NumPy, where the two are one object: after `v = x.gather(); v[0] = 100`, an expression built from `x` afterwards sees the 100.
- *A copy:* the result cannot change `x`, at the price of a copy per gather.

**The first deferred set.**

- *The table under [The deferred set](#the-deferred-set)* (the prototype): seven kinds, enough that the regression and the transformer block run without one call made at once. Since each node is computed by the call as written, a wider set costs shape rules only.
- *Narrower, element by element and matrix product only:* less to get right first, but reductions, transposes and slices would each cut the graph, and the two programs would be made of many small graphs instead of a few whole ones.
- *Wider, by what programs log* ([Ten programs with one changed line](#ten-programs-with-one-changed-line)): `np.stack` and `np.concatenate`, whose shapes follow from their arguments'. Each costs a shape rule.

**A length that is not known.**

- *What carries it.* The prototype's mark is an integer no array can have, which lets NumPy's shape arithmetic work on it unchanged and has to be kept away from every place that would allocate by it. A type of its own in the shape is the sound form, and it means the layer's own shape rules in place of NumPy's for the operations that meet it.
- *What asking for it computes.* The prototype computes the selection. Counting the mask would be enough, and would compute less.
- *Errors that move.* An error that depends on the number selected is raised when the graph is computed. A strict stand-in could refuse to defer such a selection, so that a program run strictly keeps every shape error on its line.

**The C core in the package.**

- *Built on request, beside the Python* (the prototype): `build_core` calls clang, and the layer works without the result. Right for a study object.
- *Built with the package:* `hotkernels` already compiles C for its binding, so the core would be one more source of that build and would ship in the wheels. The Python roads stay as the definition and the fallback, and the test that runs every test both ways stays with them.
- *How far it goes.* The core is a quick road, not a second implementation: it holds no rule for a shape or a type and decides nothing that Python does not decide first. Moving the rules themselves into C would make the first call of each kind cheaper, which a loop does not notice, at the price of two sets of rules to keep alike. Not proposed.

**How much to say when the layer computes without being asked.**

- *Two loggers and two levels, each line of code once, and a summary on request* (the prototype; [What the layer says](#what-the-layer-says)): quiet unless asked, and readable when asked.
- *A warning the first time, as Python's `warnings` would give:* seen without setting up logging, and noise for a program that means what it does.
- What further loggers should exist under the same name, for time or memory, is not designed.

**The graph tools** have their open questions in their own documents: [recording](lazy-recording-design.md#open-questions) and [the page](lazy-graph-page-design.md#open-questions).

## Not examined

- **Other array types as operands.** A masked array, an `np.matrix` or a pandas Series beside a lazy array is made at once by this design; which of the two types NumPy lets decide such a call was not probed.
- **The list of calls that write.** It was written from memory of NumPy's API, not checked against it.
- **Other roads to the memory.** The buffer protocol and `__dlpack__` are not defined by the prototype, so for every lazy array, strict or not, `memoryview(x)` raises a `TypeError` and `np.from_dlpack(x)` an `AttributeError` that names `__dlpack__`. Whether anything else reaches values was not searched for systematically: pickling, `copy.copy` and `np.save` were not looked at. [Issue 270](https://github.com/sofeikov/hotkernels/issues/270) is that search.
- **A codebase that uses other libraries.** The ten programs use NumPy alone. pandas, matplotlib, SciPy and scikit-learn are not installed where they were run, so nothing here says what happens when an array is handed to one of them.
- **Pickling and copying a lazy array.** Not tried.
- **`install(inexact=True)` and the bottleneck hooks with the layer.** Only the default `install()` was run.
- **The other architecture.** Everything here was run on ARM64 macOS. The layer's Python is the same everywhere, and its C core is portable C that was compiled and run on ARM64 macOS only; which kernel takes which call differs by platform, and the test of the kernels makes no assumption about it.
- **A stub file for the stand-in.** Whether one makes `np.` complete in an editor was not tried.
- **NumPy releases other than 2.5.5.** Both protocols are years old and documented, and the supported range starts at 2.3; the check runs on each.
- **A program at its real size.** The repository's applications were run smaller than they come, once, by a script outside the repository. They are not yet part of a check that anyone can run again.
