# A lazy NumPy layer: `hotkernels.get_lazy()`

This is a proposal for review. Nothing here is in the package. A prototype that follows this document rule for rule is in [`prototypes/lazy/`](../prototypes/lazy), with tests in [`tests/test_lazy_prototype.py`](../tests/test_lazy_prototype.py); it is there to be read and argued with, and [What was verified](#what-was-verified) says what it showed.

The document covers the smallest build that is worth having: NumPy code whose operations are written down as a graph instead of being computed, and computed when the user asks. **It makes nothing faster and is not meant to.** Speed is out of scope here. The graph is what fusion needs, and fusion is what a kernel for a whole expression needs; [What the graph is for](#what-the-graph-is-for) says how the three connect, without planning the second and third.

[core-design.md](core-design.md#later) left fusion for later because it "cannot be delivered transparently", and [gradient-step-findings.md](gradient-step-findings.md#can-a-hook-fuse-the-statement) found why: no hook sees an expression whole, and a result that has no values yet is unsound for a real `ndarray`. The delivery design named the other road in its first table: an array type of the project's own reaches anything, in code that has opted in. This document designs that type, and keeps the opting in to one line.

## Summary

- **One line turns it on, one call gets the values.** `np = hotkernels.get_lazy()` stands where `import numpy as np` stood, and `x.gather()` returns an ordinary `numpy.ndarray`.
- **The code in between stays NumPy code.** Operators, `np.exp(x)`, `x.mean(axis=0)`, `X.T @ r`, slicing. No new functions to learn.
- **A lazy array is a type of its own, `hotkernels.LazyArray`, not an `ndarray`.** Element types do not change: `x.dtype` is NumPy's. [Types](#types-what-changes-for-the-user) says where the new type shows in a program and what a user has to change: nothing to run, one thing if their code asks `isinstance(x, np.ndarray)`.
- **NumPy itself delivers the calls.** It hands every call that meets such a type to it, through two protocols it provides for the purpose. That reaches operators, ufuncs, functions, methods of ufuncs and `out=`, also when the code calls the real `numpy` module.
- **A narrow set of operations is deferred; all the rest of NumPy still works.** Anything outside the set is computed at once by NumPy, on gathered operands, and its result is a lazy array that already has its values.
- **Values leave only through `gather()`, for now.** `float(x)`, `if x > 0:`, `np.asarray(x)` and every other way to a value raise `NotGathered` and say to call `gather()`. Decided in review for the prototyping stage: fail loudly until it is understood how silent computing should work, which is the target and is [issue 262](https://github.com/sofeikov/hotkernels/issues/262). [Why it fails loudly](#why-it-fails-loudly-for-now) has the reasons and what trying it found.
- **The contract is one sentence:** what a program gets from `gather()` is what the same lines give with `import numpy as np`, in type, element type, shape and bits. It is checked by running programs both ways.
- **Shape and type errors are raised on the line that caused them,** with NumPy's own message, because shapes and types are worked out when the line runs. Only the values wait.
- **The first build computes the graph one node at a time, each with the NumPy call it stands for.** No kernel, no C, no fusion. The result is NumPy's by construction, and that is the point of the first build: the graph, its rules and its check exist before anything is allowed to compute differently.
- **The prototype runs two programs under these rules,** a linear regression by gradient descent and a transformer block's forward pass, with NumPy's results bit for bit.
- **Of the eight decisions asked for, four were settled in review and four are left for later review;** see [Decisions](#decisions).

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
- **`x.dtype` and `x.shape` are NumPy's own answers,** known without computing.
- **`print(x)` describes the array and never computes or shows values:** the class, the element type, the shape, and whether it is pending, computed or failed. Values are what `gather()` returns, and the `ndarray` it returns prints as an `ndarray` does.
- **`gather()` gives exactly what NumPy would have given.** For a result NumPy returns as a scalar, that is the scalar: `type(y.sum())` is `LazyArray`, and `type(y.sum().gather())` is `numpy.float64`.

The prototype prints `prototypes.lazy.LazyArray`, since that is where it lives.

### A linear regression

The changes from eager NumPy are the first line and the two `gather()` calls.

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
        print(f"step {step}: loss {loss.gather():.4f}")   # a value is needed: gathered here

weights = w.gather()                       # a numpy.ndarray
```

### Attention, with the real `numpy` module

Code that already exists, and code in other files, need not change at all: one lazy input is enough, because NumPy passes every call that meets a lazy array to it.

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

### Two spellings from one object

The two examples above use `get_lazy()` in two ways. They are not rival designs: the second is a part of the first.

```python
# A: the stand-in for the module
np = hotkernels.get_lazy()
x = np.zeros(1000)            # lazy: functions that make arrays make lazy ones
y = np.exp(x) + 1             # lazy

# B: wrapping an array
import numpy as np
lazy = hotkernels.get_lazy()
x = lazy(np.zeros(1000))      # np.zeros is the real one and eager; lazy() wraps its result
y = np.exp(x) + 1             # lazy, because x is
```

- **After the first array, the two are the same.** `np.exp(x) + 1` is deferred in both, by the same code: the lazy array receives the call from NumPy. The stand-in's `np.exp` calls the real one.
- **What A adds is the making of arrays.** `np.zeros(3)`, `np.arange(10)` and `np.array(data)` have no array among their arguments for NumPy to hand the call to, so with the real module they are eager. The stand-in makes their results lazy. With A the user changes the import and nothing else; with B they keep the import and wrap each array where it enters.
- **What A costs is the editor.** After `np = hotkernels.get_lazy()` an editor or a type checker no longer knows that `np` is NumPy, so completion and checking on `np.` are lost. B keeps them.
- **One object does both,** so the question was only whether A should exist. It does, for now ([decision 1](#decisions)): it is the one-line change, and the cost falls on tools, not on the program.

An `ndarray` that comes from neither, a random generator's `standard_normal` or a pandas column, is a plain `ndarray` until it meets a lazy array or is wrapped.

### Values leave only through `gather()`

```python
>>> total = y.sum()
>>> float(total)
NotGathered: float() needs the values of a lazy array: call .gather() first
>>> if total > 30: ...
NotGathered: A truth test (if, while, and, or, not) needs the values of a lazy array: call .gather() first
>>> f"{total:.1f}"
NotGathered: Formatting with '.1f' needs the values of a lazy array: call .gather() first
>>> numpy.asarray(y)
NotGathered: Conversion to an ndarray needs the values of a lazy array: call .gather() first
>>> plain += y                       # plain is an ndarray
NotGathered: np.add would write the values of a lazy array into an ndarray: call .gather() on the lazy array first
>>> float(total.gather())
36.0
```

`NotGathered` is a `TypeError`, which is what Python raises when an object cannot be turned into the number or array that was asked for. The full list of what raises is under [When the values are computed](#when-the-values-are-computed).

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

### The whole interface

| Name | What it is |
|---|---|
| `hotkernels.get_lazy()` | Returns a stand-in for the `numpy` module: the same names, and every array it makes is lazy. Called on an array, `lazy(x)`, it wraps that array. It changes nothing in NumPy |
| `x.gather()` | Computes `x` and returns what NumPy would have returned for it: an `ndarray`, or a NumPy scalar where NumPy gives one. `x` is still a lazy array afterwards, now computed |
| `hotkernels.gather(a, b, ...)` | The same for several arrays, as a tuple. With no argument it computes everything pending |
| `x.explain()` | The pending graph of `x` as text. Computes nothing |
| `hotkernels.LazyArray` | The type of every lazy array, for `isinstance` and for annotations |
| `hotkernels.NotGathered` | Raised when something needs values and `gather()` was not called. A `TypeError` |

Rules:

- **Importing does nothing, and neither does `get_lazy()`** beyond building the stand-in. NumPy is not changed. Only code that holds a lazy array behaves differently.
- **It does not need `install()`,** a kernel or the binding. Where `install()` does nothing because the machine is not supported, `get_lazy()` works the same. With `install()` in effect, the NumPy calls the layer makes are served by kernels like any others.
- **It never refuses a NumPy call on lazy arrays that gives arrays back.** A call outside the deferred set is made at once.
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
| `isinstance(x, np.ndarray)` | False | Yes. The one place a program has to change: `isinstance(x, (np.ndarray, hotkernels.LazyArray))` |
| The result of a reduction: `loss = (r ** 2).mean()` | A `LazyArray` of shape `()`. `loss.gather()` is an `np.float64`, as NumPy's `loss` is | `gather()` where the number is used |
| A library that takes an array: pandas, matplotlib, a Cython function taking `double[:]` | `NotGathered`: the library asks NumPy for an array, or for the memory | Pass `x.gather()` |
| An editor or a type checker | With the stand-in, `np` is not known to be NumPy: no completion on `np.`. With wrapping, the tools see NumPy and believe every array is an `ndarray` | None to run. See [Two spellings](#two-spellings-from-one-object) |

Two things were considered and left out:

- **Making `isinstance(x, np.ndarray)` true.** The stand-in could hand out, as `np.ndarray`, a type whose instance check accepts lazy arrays. It would work only in the user's own module, since every other module asks the real `numpy.ndarray`, and it would tell code that it holds memory when it does not. Not while the layer fails loudly; it can be asked again with [issue 262](https://github.com/sofeikov/hotkernels/issues/262).
- **Type stubs.** What a type checker should be told a lazy array is, and whether `get_lazy()` can be typed so that `np.` completes, is not designed here.

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
| Made at once | The lazy operands are gathered, NumPy makes the call, and the result is a lazy array that has its values | Everything else in NumPy that gives arrays back: `np.linalg.solve`, `np.argsort`, `x[mask]`, `x.cumsum()`, any method or function not listed |
| A write into a lazy array | Everything pending is computed first, in the order it was written, and then NumPy makes the call | `x[i] = v`, `x += y`, `np.multiply(a, 2, out=x)`, `np.copyto(x, v)`, `x.sort()` |

The third row is what keeps NumPy's meaning when memory is changed in place. Once everything pending is computed, the program is in the state the eager program would be in, and the write is NumPy's own. It is never more work than eager NumPy would have done, since eager NumPy had computed all of it already. Whether lazy arrays should allow writes at all is [decision 4](#decisions), left open; this is the prototype's behaviour until then.

**A call made at once stays quiet, and that is deliberate.** It computes its operands without being asked, but nothing leaves the lazy world: the result is a `LazyArray`, and the program still has to call `gather()` to see a value. Making these calls raise as well would stop nearly every real program at its first `np.argmax` until the deferred set had grown to cover it. They are counted, so a later report can show where a program's graph was cut.

### The deferred set

Narrow in kinds, not in count. The first build computes each node with NumPy's own call, so admitting an operation costs a shape rule and nothing else.

| Kind | Deferred | Not deferred |
|---|---|---|
| Element by element | Every ufunc with one output and no core dimensions (`np.add`, `np.exp`, `np.tanh`, `np.maximum`, `np.less`, ...), on arrays and numbers given by position; the operators that call them: `+ - * / // % **`, unary `-`, `abs`, the six comparisons, `& \| ^ ~`; `np.where(c, a, b)` | A call with `where=`, `dtype=` or `casting=` |
| Matrix product | `@`, `np.matmul` | `np.dot`, `np.einsum`, `np.linalg` |
| Reductions | `sum`, `mean`, `max`, `min`, `prod`, `var`, `std`, as functions and as methods, with `axis` and `keepdims` by keyword | Any other argument; `argmax`, `cumsum`, `any` |
| Views | `x.T`, `np.transpose`, `np.swapaxes`, `reshape`, `np.expand_dims`, `np.squeeze`, and indexing with integers, slices, `None` and `...` | Indexing with an array or a mask: the shape of `x[mask]` depends on values, so it is made at once |

Never deferred, whatever the operation: an operand with a dimension of length zero, an element type that is not boolean, integer, floating or complex, and an operand that is neither a lazy array, an exact `ndarray`, a NumPy scalar nor a Python number.

### When the values are computed

| Event | What is computed |
|---|---|
| `x.gather()`, `hotkernels.gather(a, b)` | Those arrays and what they need |
| A call made at once | Its operands |
| A write into a lazy array, and `hotkernels.gather()` with no argument | Everything pending |
| More than 100,000 nodes pending | Everything pending, so a long loop that never gathers cannot hold an unbounded graph |
| `x.shape`, `x.dtype`, `len(x)`, `print(x)`, `x.explain()`, `np.shape(x)` | Nothing |

A node is computed once. Its value is kept while the program holds the lazy array, and an intermediate nobody holds is freed as soon as its last consumer has been computed. Pending work that the program drops without asking for it is never computed.

What raises `NotGathered` instead of computing:

| What the program does | Why it would need values |
|---|---|
| `float(x)`, `int(x)`, `complex(x)` | Python wants one number |
| `if x > 0:`, `while`, `and`, `or`, `not`, `bool(x)` | Python wants one truth value |
| `range(n)`, `a[i]` with a lazy `i` and a plain `a` | Python or NumPy wants an integer |
| `f"{x:.3f}"` | A format of a value. `f"{x}"` is the description and is allowed |
| `for row in x:`, `list(x)`, `v in x` | The elements themselves |
| `np.asarray(x)`, `np.array(x)` with the real module, `plt.plot(x)`, `rng.permutation(x)`, `a[i] = x` on a plain `a` | NumPy asks the object for an array |
| `a += x`, `np.multiply(x, 2, out=a)`, `np.copyto(a, x)` with a plain `a` | Lazy values written into an `ndarray` |
| `x.flat`, `x.real`, `x.data`, `x.ctypes` | Attributes that are the memory |
| `x.tolist()`, `x.item()`, `np.allclose(x, y)`, `np.array_equal(x, y)` | A call whose answer is a plain Python value. It is made, and then refused |

Each is refused whether or not the array already has its values.

### Why it fails loudly, for now

The first version of this design computed silently wherever a value was needed. Review turned that round for the prototyping stage: raise, and aim for silent computing once it is understood how it should work ([issue 262](https://github.com/sofeikov/hotkernels/issues/262)). Trying it on the prototype supported the decision.

- **It costs the two programs almost nothing.** Both run unchanged with every implicit way out raising. In the regression example above one line differs from the silent version: `{loss:.4f}` became `{loss.gather():.4f}`.
- **It found a hole within minutes.** The first prototype answered any attribute it did not define with the computed array's. NumPy asks an object for `__array_interface__` and `__array_struct__` before it calls `__array__`, so `np.asarray(x)`, `a[i] = x` and `rng.permutation(x)` were handed the memory without the layer noticing, and the rule "when NumPy asks, compute everything pending first" was never reached. Silent computing had hidden that: the results were right in the tests and the rule was dead. The prototype now forwards methods only, and no double-underscore name.
- **It shows where a program needs values,** which is what silent computing has to be designed from. Each `gather()` a user is made to write is a point where the graph is cut.
- **It removes a rule nobody could check.** `np.asarray(x)` and `a[i] = x` reach a lazy array the same way, and the second is a write. Computing silently there means computing everything pending, in case something pending reads `a`. With a refusal there is nothing to get wrong.

Three choices inside the decision:

- **It raises even when the array is already computed.** No work would be hidden in that case, but a program would then behave differently depending on whether something had been gathered earlier, and `if loss < tolerance:` would raise anyway, since the comparison is a new pending array.
- **Writing lazy values into an `ndarray` raises too.** `a += x` with a plain `a` computes `x` and puts its values into memory the layer does not own. That is a way out of the lazy world, and `gather()` is the only one: `a += x.gather()`.
- **`print(x)` never raises.** A description costs nothing and a debugger or a log line must not fail.

### What differs from NumPy

A user has to know six things.

1. **Values come from `gather()` and from nowhere else.** See above.
2. **A lazy array is not an `ndarray`.** `isinstance(x, np.ndarray)` is False. See [Types](#types-what-changes-for-the-user).
3. **A pending expression reads an `ndarray` when it is gathered, not when it is written down.**

   ```python
   a = numpy.arange(3.0)
   y = lazy(a) * 2      # pending
   a[0] = 5             # written directly to the ndarray
   y.gather()           # array([10., 2., 4.]); eager NumPy had computed y before the write: [0., 2., 4.]
   ```

   Writes made through a lazy array are safe, by the rule for writes; a write made directly to an `ndarray` that a pending expression has read is not seen by the layer. Gather before such a write. `np.array(a)` copies, as it does in NumPy, and `np.asarray(a)` and `lazy(a)` do not. Whether inputs should be copied instead is [decision 3](#decisions), left open.
4. **Warnings and errors that depend on values come at the gather.** They are the same warnings, with NumPy's words, and each names the line that wrote the operation, not the line that gathered. `np.errstate` is remembered per operation, so `with np.errstate(divide="ignore"): r = a / b` stays silent wherever `r` is gathered. A `warnings.catch_warnings()` block around the line does not reach the gather.
5. **An operation that failed stays failed.** Under `np.errstate(over="raise")` the exception is raised by the first gather that needs the operation, with a note naming the line that wrote it, and again by any later gather that needs it. A write that computes everything pending raises it once, and not again for what depended on it.
6. **The work happens at the gather,** so that is where time passes and memory is taken, and what `KeyboardInterrupt` interrupts.

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

Six modules of pure Python, laid out as the package would be ([Where it lives](#where-it-lives)), so that a comment on a file is a comment on a piece of the design. It follows this document: the deferred set above, every other call at once, values only through `gather()`, node-by-node evaluation. `uv run --group dev pytest tests/test_lazy_prototype.py` runs 19 tests, one or more per rule stated here, each comparing with the real `numpy` module.

| Program | Deferred | Made at once | Result against eager NumPy |
|---|---|---|---|
| Linear regression ([`programs/regression.py`](../prototypes/lazy/programs/regression.py)): standardised features, 50 gradient steps with an L2 penalty, no gather inside the loop, 20,000 × 30 `float64`, run with the stand-in | 954 operations | 0 | Coefficients, intercept and all 50 losses identical, bit for bit |
| Transformer block ([`programs/attention.py`](../prototypes/lazy/programs/attention.py)): layer norm, four-head attention with a causal mask, softmax, GELU, two residuals, `float32` parameters, 64 × 96, calling the real `numpy` module with one lazy input | 55 | 0 | Identical |

The two examples under [What the user writes](#what-the-user-writes) were also run as printed there.

What the tests hold, each matching a rule of this document:

- **Errors on the line.** A sum of shapes (6,) and (4,), a matrix product with mismatched inner sizes, `np.sum(x, axis=3)` on one dimension, a `reshape` to the wrong size, an index out of range, `int8_array + 300` and `float & float` each raise the exception NumPy raises, with the same message, when the line runs.
- **Promotion is NumPy's,** for ten expressions that mix Python numbers, NumPy scalars, integers, booleans and reductions. The transformer block comes out `float64` under both, because `np.sqrt(2 / np.pi)` is an `np.float64`.
- **Warnings.** `a / zeros` gathered later issues NumPy's two warnings, word for word, with the file and line of the division. Written under `np.errstate(divide="ignore")` and gathered outside it: none. Under `errstate(over="raise")`: `FloatingPointError` at the gather, with the note, and the same exception again at the next gather.
- **Values leave only through `gather()`:** twenty ways from the table above each raise `NotGathered`, and an `ndarray` that a refused write named is unchanged.
- **Writes into lazy arrays.** `x[1:3]` assigned through, as a view, changes `x`, and an expression written before it has the old values. `w -= step` keeps the object, so another name for it sees the change. `s += 1` on the result of a `sum` gives a new scalar, as NumPy does. `int_array += 0.5` raises NumPy's casting error.
- **Calls made at once** (`x[mask]`, `x[[0, 2, 4]]`, `np.linalg.norm`, `x.cumsum()`, `np.argmax`, `np.concatenate`, `np.dot`, `np.sort`) give NumPy's results as computed lazy arrays.
- **The stand-in:** arrays it makes are lazy, types and constants are NumPy's own objects, `np.linalg.norm` and `np.add.reduce` work through it, `np.asarray(x)` returns `x` without computing, and `np.array(x)` copies.
- **Housekeeping:** a shared operand is printed once by `explain()`; past the limit of pending nodes everything pending is computed; pending work nobody holds is dropped without being computed.

Building a node costs Python time: 0.3 to 0.5 µs in the bare probe, where NumPy adds two arrays of 16 elements in 0.26 µs, on a busy machine (load average 8 to 10). The first build is therefore slower than NumPy on small arrays. That is accepted here.

## Design

### Where it lives

A subpackage of pure Python, `hotkernels/src/hotkernels/lazy/`, and four names exported from the package: `get_lazy`, `gather`, `LazyArray` and `NotGathered`. The prototype has the same modules:

| Module | Holds | In the prototype |
|---|---|---|
| `graph.py` | The node, the list of pending nodes, the lock, `NotGathered` | [graph.py](../prototypes/lazy/graph.py) |
| `rules.py` | Which calls are deferred, and the shape and element type of each | [rules.py](../prototypes/lazy/rules.py) |
| `evaluate.py` | Computing a node, a set of nodes, everything pending | [evaluate.py](../prototypes/lazy/evaluate.py) |
| `array.py` | The lazy array: operators, the two protocols, the refusals, `gather()` | [array.py](../prototypes/lazy/array.py) |
| `namespace.py` | The stand-in for the `numpy` module | [namespace.py](../prototypes/lazy/namespace.py) |
| `explain.py`, `templates/explain.txt.j2` | The text of a pending graph | [explain.py](../prototypes/lazy/explain.py) |

`graph` knows nothing of the others; `rules` and `evaluate` work on nodes and never see a `LazyArray`; `array` is the only module the user's calls enter. Whatever computes a graph differently later replaces one function of `evaluate`.

The prototype stays outside the package on purpose. Inside `hotkernels/` it would be shipped to users and would read as the implementation; in `prototypes/` it is a study object that the implementation is free to depart from.

### The node

| Field | Meaning |
|---|---|
| Number | A count that rises with every node. Operands are always older than their consumer, so ascending order is a valid order of evaluation, and it is the order the eager program ran in |
| Operation | The NumPy callable the node stands for: a ufunc, a function such as `np.sum`, or indexing. Empty once the node is computed |
| Operands | Other nodes, `ndarray`s held by reference, and Python numbers, indices and axes kept exactly as written, so that NumPy's rule for them (a Python `2.0` does not widen a `float32` array) is applied by NumPy |
| Shape, element type | Worked out when the node is made |
| Written at | File and line of the user's code, and that module's record of warnings already shown |
| Error state | The `np.errstate` in effect when it was written |
| Value | Empty until computed. Then the operands are dropped |
| Failure | The exception, if computing it raised |

A lazy array is a handle on one node. A leaf is a node born with its value. Nodes refer to their operands and nothing refers to a node but its consumers and the lazy array a user holds, which is why an intermediate nobody holds is freed when its consumer has been computed.

### Shape and element type without computing

- **The shape comes from a rule,** one per kind: NumPy's `np.broadcast_shapes` for element-by-element operations; the matrix product's rule; the axes removed for a reduction. For a view, the operation is made on a stand-in that has the operand's shape and no memory (`np.broadcast_to` of one zero), which NumPy answers without touching data.
- **The element type comes from NumPy itself:** the real call is made on arrays with one element per dimension, with the Python numbers as written. That costs a microsecond or two and cannot disagree with NumPy about promotion.
- **An error is NumPy's.** When the rule finds the shapes do not fit, the real call is made on the stand-ins without memory, and NumPy raises its own exception with its own message before it reads anything. If NumPy does not raise, the rule was wrong to doubt, and the call is made at once.
- **When in doubt, do not defer.** A call the rules do not recognise in every detail is made at once. The layer never guesses a shape or a type.

### Computing

- `gather` collects the nodes its arrays need that have no value, with a stack and not by recursion, and computes them in ascending number: the node's own NumPy call, on its operands' values, under its error state.
- **Warnings are recorded and issued again.** Left alone, NumPy's warning would name a line inside the layer. Each node's call runs inside `warnings.catch_warnings(record=True)`, and what NumPy issued is issued again with NumPy's own message and category, at the file and line the node was written at, through that module's record of warnings shown, so Python shows it once per line as it would for the eager program. Recording changes the process's warning filters for the length of the call, which is safe only because every evaluation holds the layer's lock.
- An exception gets a note naming that line (`add_note`), is kept on the node, and is raised.
- The first build therefore makes exactly the calls the eager program makes, on the same operands. Whatever computes a graph differently later is held to the same results by the check below.

### Pending work and writes

Every node without a value is in one list, held weakly. A write into a lazy array computes that list in ascending number and then lets NumPy write. In-place operators are handed to NumPy's in-place operator on the computed value: if NumPy returns the same array, the lazy array is returned unchanged in identity, so every name for it sees the write; if it returns a new scalar, a new lazy array is.

**Which calls write is a list the layer keeps:** `out=`, `x[i] = v`, the in-place operators, `np.copyto`, `np.putmask`, `np.place`, `np.put`, `np.fill_diagonal`, `ufunc.at`, and the methods `sort`, `fill`, `put`, `partition`, `resize`, `itemset`, `setfield` and `byteswap`. A NumPy function that writes into an argument and is not on the list would be made at once without the pending work being computed first. The list has to be checked against NumPy as part of the build.

One lock guards the list and every evaluation. Lazy arrays can be used from several threads; their work is done one gather at a time. Free-threaded Python is not supported, as for the rest of the package.

### The stand-in for the module

`get_lazy()` returns an object that looks names up in `numpy`:

- A type or a constant (`np.float32`, `np.pi`, `np.newaxis`, `np.errstate`) is returned as it is.
- A submodule (`np.linalg`, `np.random`) is returned as another stand-in.
- A function or a ufunc is returned wrapped: `ndarray` arguments become lazy arrays, the real function is called, NumPy's protocols route it to the lazy arrays, and an `ndarray` that comes back is made lazy. So `np.zeros(3)` is lazy, and `np.exp(x)` is the same call it is with the real module. A wrapped ufunc keeps its methods, so `np.add.reduce` works.
- `np.array`, `np.asarray` and their like are answered by the stand-in itself. They have no protocol through which a lazy array could answer, so the real ones would ask it for its values. `np.asarray(x)` returns a lazy `x` as it is, without computing; `np.array(x)` copies, as in NumPy.

### What an unknown attribute gets

A lazy array answers a method it does not define with `ndarray`'s, made at once: `x.cumsum()`, `x.astype(np.float32)`, `x.argmax()`. Two limits keep that from becoming a way out:

- **No double-underscore name is forwarded.** `__array_interface__` and `__array_struct__` would hand NumPy the memory behind the layer's back; see [Why it fails loudly](#why-it-fails-loudly-for-now).
- **An attribute that is not a method raises `NotGathered`.** `x.flat`, `x.real`, `x.data` are the values.

### Checking it

A lazy layer is trusted for the same reason a kernel is: a check that does not depend on reading the code. `uv run delivery lazy-check` runs:

1. **Programs both ways.** The two examples, and the applications in `delivery/apps/` that need only NumPy (`mlp_train`, `min_char_rnn`, `veros_ocean`, `particle_filter`), each with its module's `np` replaced by the stand-in and a `gather()` where it reports. Results, exceptions and warnings are compared as `delivery check` compares them.
2. **Random programs.** Sequences of operations from the deferred set and outside it, over random shapes, element types and special values, with gathers and writes at random points, run eagerly and lazily. Compared: type, element type, shape, bits, the exception's type and message, the warnings and their lines, and the final state of every `ndarray` the program was given.
3. **Each supported NumPy,** in CI, since the protocols are NumPy's.

The prototype's tests are the first part of this in small.

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
| Compute silently wherever a value is needed | The target, not the first build: [issue 262](https://github.com/sofeikov/hotkernels/issues/262) and [Why it fails loudly](#why-it-fails-loudly-for-now) |
| Refuse calls outside the deferred set as well | Nearly every real program would stop at its first such call until the set grew. They compute without letting a value out, so they are made at once and counted |
| Defer operations whose shape depends on values (`x[mask]`, `np.unique`) | Every lazy array would lose its shape, and shape errors their line. Making these at once keeps the rule that shape is always known |
| A decorator, `@hotkernels.lazy`, that wraps a function's arguments and gathers its results | Sugar over this design, and a good one for a hot function. Left until the layer under it exists |
| Dask, JAX or numexpr | Each has semantics of its own: chunked and parallel arrays, another array library with its own type rules, expressions as strings. The contract here is NumPy's results bit for bit, so that kernels judged against NumPy can compute the graph |
| The array API standard (`__array_namespace__`) as the way in | It serves libraries written for it. Existing NumPy code calls `np.exp(x)`, which the two protocols above serve. It can be added beside them |

## Exit criterion

1. **Two programs run with one changed line and a `gather()` where a value is used:** `examples/lazy_regression.py` and `examples/lazy_attention.py`, with results identical to eager NumPy, and no operation in either made at once.
2. **`uv run delivery lazy-check` passes** in all three of its parts.
3. **Shape and type errors are NumPy's and are raised on the line,** and value-dependent warnings name the line that wrote the operation, for every deferred kind.
4. **Every way to a value other than `gather()` raises `NotGathered`,** and no road hands NumPy the memory unnoticed.
5. **`get_lazy()` works with and without `install()`,** and `delivery check` and NumPy's test suite are unaffected by it, since it changes nothing in NumPy.
6. **The user-facing pages say what a user must know:** `docsite/pages/start.md` and `hotkernels/README.md` carry the examples, the table of types and the six differences.

No criterion is about speed.

## Decisions

### Settled in review

Review of [pull request 261](https://github.com/sofeikov/hotkernels/pull/261), 21 November 2026.

| # | Question | Decision | What was not chosen |
|---|---|---|---|
| 1 | What `get_lazy()` returns | One object that is the stand-in for the `numpy` module and also wraps an array when called. See [Two spellings](#two-spellings-from-one-object) for what each is and what the stand-in costs | A wrapping function alone. A process-wide switch was never on offer: it is unsound |
| 2 | A value is needed and `gather()` was not called | Raise `NotGathered`, for the prototyping stage. Silent computing is the target: [issue 262](https://github.com/sofeikov/hotkernels/issues/262) | Computing silently from the start |
| 5 | `print(x)` | A description, always: class, element type, shape, state. Never values, never a computation | Showing the values of a computed array; computing in order to show them |
| 8 | Its home | Inside the `hotkernels` package, as `hotkernels.get_lazy()`, in pure Python and independent of `install()` | A package of its own |

### Left for later review

Each has a behaviour in the prototype, so that the prototype is defined; none is settled.

**3. An `ndarray` read by a pending expression: held by reference, or copied?**

```python
a = numpy.arange(3.0)
y = lazy(a) * 2
a[0] = 5
y.gather()      # by reference: [10., 2., 4.]      copied when met: [0., 2., 4.], as eager NumPy
```

- *By reference* (the prototype): nothing is copied, and the user has a rule to remember: gather before writing directly to an `ndarray` a pending expression has read.
- *Copied when first met:* no rule to remember and no way to get a result eager NumPy would not give. Every plain `ndarray` is copied each time an expression meets it, so a loop that reads the same large `ndarray` in every step copies it in every step unless the user wraps it once.
- A third possibility, not designed: by reference, with a check mode that remembers a fingerprint of each input and raises at the gather if it changed.

**4. `x[i] = v` and `x += y` on a lazy array: NumPy's meaning, or refused?**

- *NumPy's meaning* (the prototype): everything pending is computed, then NumPy writes, and another name for the same array sees the write. Existing code with in-place updates (`w -= lr * g`, `dhidden[hidden <= 0] = 0`) runs unchanged. The price is that each such statement computes everything pending, which will matter once computing a graph whole is worth something.
- *Refused:* lazy arrays cannot be changed; the user gathers and writes to the `ndarray`. Simpler, and consistent with failing loudly, but the commonest idiom of a training loop then needs rewriting.

**6. What `gather()` returns, and what the lazy array is afterwards.**

- *The array itself, and the lazy array keeps it* (the prototype): no copy. Writing into the returned `ndarray` changes what the lazy array holds, as it would in NumPy, where the two are one object: after `v = x.gather(); v[0] = 100`, an expression built from `x` afterwards sees the 100.
- *A copy:* the result cannot change `x`, at the price of a copy per gather.

**7. The first deferred set.**

- *The table under [The deferred set](#the-deferred-set)* (the prototype): four kinds, enough that the regression and the transformer block run without one call made at once. Since each node is computed by NumPy's own call, a wider set costs shape rules only.
- *Narrower, element by element and matrix product only:* less to get right first, but reductions, transposes and slices would each cut the graph, and the two programs would be made of many small graphs instead of a few whole ones.

## Not examined

- **Other array types as operands.** A masked array, an `np.matrix` or a pandas Series beside a lazy array is made at once by this design; which of the two types NumPy lets decide such a call was not probed.
- **The list of calls that write.** It was written from memory of NumPy's API, not checked against it.
- **Other roads to the memory.** The buffer protocol and `__dlpack__` are not defined by the prototype, so they fail; whether anything else reaches values was not searched for systematically. [Issue 262](https://github.com/sofeikov/hotkernels/issues/262) lists it.
- **Pickling and copying a lazy array.** Not tried.
- **NumPy releases other than 2.5.5.** Both protocols are years old and documented, and the supported range starts at 2.3; the check runs on each.
- **A program of real size.** The two programs are a few dozen lines each. The applications in `delivery/apps/` are the next test and are part of the check.
