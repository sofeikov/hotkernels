<p class="lede">Every kernel in this report makes one NumPy call faster. A line of real code is rarely one call: <code>np.exp(a * b + c).sum()</code> is four, and NumPy makes them one after another, each reading its inputs from memory and writing a whole new array for the next to read back. Doing the four in one pass needs something that sees the expression whole, and no hook into NumPy ever does. This chapter is about a design that does: NumPy code whose operations are written down as a graph instead of being computed, and computed when the program asks.</p>

::: note A design with a prototype, not part of the package
What this chapter describes is a proposal, [docs/lazy-design.md](repo:docs/lazy-design.md), with two documents of its own for recording the graph and drawing it, [docs/lazy-recording-design.md](repo:docs/lazy-recording-design.md) and [docs/lazy-graph-page-design.md](repo:docs/lazy-graph-page-design.md), and a prototype that follows all three, in [prototypes/lazy/](repo:prototypes/lazy). Nothing here is installed with `hotkernels`, and the names below (`hotkernels.get_lazy()`, `gather()`, `record_graphs()`) are the ones the design proposes. **The layer by itself makes nothing faster, and is not meant to.** Its job is to write the graph down correctly and give NumPy's results bit for bit. What the graph is then used for, kernels that compute a whole expression in one pass, is the subject of the next two chapters, [Fusion]([[ root ]]report/fusion/) and [What fusion gained]([[ root ]]report/fusion-results/); [the last section](#what-the-graph-is-for) of this one is the bridge. Every behaviour shown in the figures was run on the prototype, and every timing was taken on an Apple M1 Pro.
:::

## One call at a time

NumPy is **eager**: each operation is computed the moment the line runs. That is what makes it easy to use, since every array always has its values, and it has a cost that no single faster call can remove. Take the GELU activation from a transformer:

```python
y = 0.5 * x * (1 + np.tanh(k * (x + 0.044715 * x ** 3)))
```

That one line is eight NumPy calls. Each reads one or two whole arrays and writes a third, a **temporary**, that exists only for the next call to read back. For an array of a million float64 values, every temporary is 8 MB. A loop that took one element of `x` at a time through the whole formula would read `x` once and write `y` once.

<figure class="widget">
<div data-widget="passes" data-expr="gelu"></div>
<figcaption>Bytes moved to compute an expression one call at a time, as NumPy does, and in one pass over the data. This is counting, not timing. Move the slider past about 1.5 million elements and a temporary no longer fits in the M1 Pro's 12 MB second-level cache: every pass then goes to main memory, and <a href="[[ root ]]report/machine/#fact-four-memory-has-a-speed-limit">memory is the limit</a>.</figcaption>
</figure>

The kernels of this project cannot help here, for a reason [the delivery chapter]([[ root ]]report/delivery/) depends on. A kernel reaches unmodified code through a hook inside NumPy, and a hook sees one call: `x ** 3`, then later and separately `0.044715 * …`. It cannot hold the first result back to combine it with the second, because the first result is an `ndarray`, and an `ndarray`'s memory is public. Any code may read it the moment the call returns, without calling anything a hook could see. An `ndarray` has to have its values from the instant it exists.

So an expression can only be computed as a whole by something that is not an `ndarray`: an array that stands for a value not computed yet. Then the program's operations can be collected first, as a **graph**, and computed together when a value is actually needed. That is what **lazy** means. The rest of this chapter is how to do it to existing NumPy code without breaking any of it.

## Writing it down instead

The design asks the program to change one line:

```python
import hotkernels
np = hotkernels.get_lazy()        # where `import numpy as np` stood

x = np.arange(6.0)
y = x * 2 + 1                     # written down, not computed
print(y)                          # LazyArray(float64, shape=(6,), pending)
y.gather()                        # array([ 1.,  3.,  5.,  7.,  9., 11.])
```

From that line on, every array the program makes is a `LazyArray`. Operating on one does not compute anything: it adds a **node** to the graph, a record of which operation to make on which operands. Each node knows its shape and element type at once, because those follow from the operands' shapes and types and not from their values. `gather()` is how the program asks for values: it computes what the array needs and returns exactly what NumPy would have returned, an ordinary `numpy.ndarray`, or a NumPy scalar where NumPy gives one: `type(y.sum())` is `LazyArray`, and `type(y.sum().gather())` is `numpy.float64`.

Step through four short programs and watch the graph grow. Switch to **NumPy** to see the same lines run eagerly.

<figure class="widget">
<div data-widget="lazytrace" data-program="expression"></div>
<figcaption>Four programs run one line at a time. A dashed box is pending: written down, with no values yet. A filled box has its values, and the numbered badge says which computation produced it. The dark labels are the program's variables and the node each one stands for. Everything shown, down to the printed text and the log lines, is what the prototype gives.</figcaption>
</figure>

Four things to take from it:

- **A pending array still answers questions about itself.** `x.shape`, `x.dtype`, `len(x)` and `print(x)` compute nothing. Values are the only thing that waits.
- **Nothing is computed that nobody asks for.** In *A mask*, asking for the length of the selection computes the selection and leaves the mean pending. Pending work that the program drops without asking for it is never computed at all.
- **Each computation is one graph.** The nodes computed together, because one thing asked for them, are the unit that anything computing a graph differently is handed.
- **A value needed is computed, not refused.** In *A value needed*, the `if` needs True or False on its line. The layer computes what the test reads, logs the line, and the program carries on.

A node is computed once. Its value is kept for as long as the program holds the lazy array, and an intermediate that nobody holds is freed as soon as the last node that reads it has been computed, as NumPy frees a temporary. Where nothing else reads it, a large intermediate's memory takes the result of the call that reads it, as NumPy writes `a * b + c` into the memory of `a * b`.

## One line, and nothing patched

`np = hotkernels.get_lazy()` is an ordinary assignment. `get_lazy()` returns an object that answers the names the `numpy` module has, and the line binds it to a name in one module. That is the whole mechanism, and it is deliberate.

- **NumPy is not changed.** `sys.modules["numpy"]` is the real module, `numpy.zeros` is the real function, and every other module's `import numpy` gives what it always gave. So it does not matter what was imported before the line, or in which order.
- **Arrays enter in three ways.** A function of the stand-in that makes an array makes a lazy one: `np.zeros(3)`, `np.arange(10)`, and the numbers a generator from `np.random` draws. An existing `ndarray` is taken in with `np.asarray(a)`, without a copy. And a plain `ndarray` that meets a lazy array in an operation joins the graph as an input.
- **Laziness travels with the array, into code that knows nothing of it.** A model in another file, written with `import numpy as np`, stays lazy when it is handed a lazy array:

```python
# model.py, unchanged
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
```

Call `softmax` with a lazy array and all of its operations are nodes. How that can work, when `model.py` calls the real `np.exp`, is the next section.

What the line costs is the editor. After it, an editor or a type checker no longer knows that `np` is NumPy: no completion on `np.`, no checking of calls. The program is not affected. [Other ways in](#other-ways-in-and-why-not) has a spelling that could give the editor back.

## How the calls find the layer

Python and NumPy both let an object that is not an `ndarray` take part in their operations, through methods with fixed names, a convention called a **protocol**:

- **Python's operators.** `x * 2` calls `x.__mul__(2)`, `x -= y` calls `x.__isub__(y)`, `x[i]` calls `x.__getitem__(i)`. A class decides what its own operators do.
- **`__array_ufunc__`.** Every element-by-element function of NumPy (`np.exp`, `np.add`, `np.maximum` and some ninety others, called **ufuncs**) first looks at its arguments. If one of them has this method, NumPy calls it instead of computing. This is also what happens when an `ndarray`'s own operator meets a lazy array: `a + x` with a plain `a` is handed over.
- **`__array_function__`.** The same for NumPy's other functions: `np.sum`, `np.where`, `np.transpose`, `np.argsort`, with the function and its arguments exactly as they were written.
- **Conversions.** `float(x)` calls `x.__float__()`, an `if` calls `x.__bool__()`, and `np.asarray(x)` asks for `x.__array__()`. These are the moments a value has to leave.

A `LazyArray` defines all of them. So NumPy itself delivers every call that meets a lazy array, in whatever file. Pick a line and follow it:

<figure class="widget">
<div data-widget="lazyroute"></div>
<figcaption>How each spelling reaches the layer, what the layer does with it, and what the line gets back. x is a pending lazy array of shape (4, 3), a a plain ndarray of the same shape, w an array the stand-in made, numpy the real module and np the stand-in. The results and log lines are the prototype's.</figcaption>
</figure>

Every call ends up as one of three kinds:

| Kind | What happens | Which calls |
|---|---|---|
| **Deferred** | A node joins the graph. Nothing is computed | The set below |
| **Made at once** | The lazy operands are computed, NumPy makes the call, and the result is a lazy array that already has its values | Everything else that returns arrays: `np.argsort`, `np.concatenate`, `np.linalg.solve`, any argument the rules do not recognise |
| **A write** | A node for the array's new contents. Nothing is computed | `w -= step`, `x[i] = v`, `np.add(a, b, out=x)`, in an array only the layer can see; see [Writes](#writes-and-why-a-loop-stays-one-graph) |

A call made at once is not an error. Refusing it would stop nearly every real program at its first `np.argsort` until the deferred set had grown to cover it. The program runs, and the log names the line, so the set can grow from what real programs actually call.

The deferred set is narrow in kinds and wide in count:

| Kind | Examples |
|---|---|
| Element by element | every ufunc with one output, the operators, `np.where(c, a, b)`, `np.round`, `x.astype(t)`, `np.copy` |
| Matrix product | `@`, `np.matmul`, `np.dot` of one or two dimensions |
| Reductions | `sum`, `mean`, `max`, `min`, `prod`, `var`, `std`, `any`, `all`, `argmax`, `argmin`, `np.nanargmax`, `np.count_nonzero`, with an axis and `keepdims` |
| Running totals and searches | `np.cumsum`, `np.cumprod`, `np.searchsorted` |
| Views | `x.T`, `reshape`, `ravel`, `np.transpose`, `np.swapaxes`, `np.expand_dims`, `np.squeeze`, slicing, `x.view()` and `x.view(type(x))` |
| Indexing by arrays | `x[rows]`, `x[rows, columns]`, `x[mask]`, also with an index that is itself pending: `x[np.argmax(w)]`, `x[speed > limit]` |
| Ordering and positions | `np.partition`, `np.nonzero`, `np.flatnonzero`, `np.argwhere` |

Because the layer computes each node with NumPy's own call, admitting another operation into the set costs a rule for its shape and nothing else.

## Shapes without values

The layer has to know what NumPy would make of a call before anything is computed: the shape and element type of the result, and whether the call is an error at all. It never works these out by rules of its own where NumPy can be asked.

- **The element type is NumPy's answer.** For a ufunc, NumPy's own resolver, `np.multiply.resolve_dtypes`, gives the result's type from the operands' types. Where nothing that cheap exists, the real call is made on arrays of one element each, with the Python numbers as written. So `float32 * 2.0` stays `float32` and `float32 * np.float64(2)` becomes `float64`, as in NumPy, because it is NumPy deciding.
- **The shape comes from a rule or from a stand-in.** Element-by-element shapes follow NumPy's broadcasting, and the matrix product and reductions have short rules of their own. For a view or an index, NumPy is asked to make the same view of an array that has the operand's shape and no memory at all, an array of an element type of size zero, and the shape is read from the answer.
- **An error is NumPy's, raised on the line.** If the shapes do not fit, the real call is made on those stand-ins, and NumPy raises its own exception with its own message before it reads a byte: `x + numpy.arange(4.0)` on a (4, 3) array fails with *operands could not be broadcast together with shapes (4,3) (4,)*, on that line, as it would eagerly.
- **When in doubt, do not defer.** A call the rules do not recognise in every detail is made at once. The layer never guesses a shape or a type.

One length cannot be known: how many elements a mask selects. `particles[speed > limit]` has as many rows as the mask has true values, which is a value. The selection is deferred anyway, and its shape carries a mark, `(?, 4)`. Whatever is computed from it stays deferred wherever the result's shape does not depend on that number: arithmetic with a number, a mean along that axis, a matrix product, an assignment back through the same mask. Where NumPy's answer does depend on it, as in the first five rows `x[mask][:5]`, the call is made at once. Asking for the length, with `len()` or `.shape`, is asking for a value, and computes the selection.

Making every such selection at once was the simpler design, and it keeps every shape known and every shape error on its line. It was measured and not taken: a program that selects by a mask inside its loop then computes at every step, and the particle filter and k-means of the programs below had graphs a few lines long.

## When a value is needed

Values leave a lazy array in two ways. `gather()` is the one the program writes. The other is everything else that needs a real number or a real array: `float(loss)`, `if error < tolerance:`, `f"{loss:.4f}"`, `np.asarray(x)` with the real module, and every library that asks NumPy for an array, such as a plotting call.

The layer **computes there instead of raising**. A real program needs a value every few lines, and if each of those raised, it would have to be rewritten before its first graph could even be looked at. Computing on the spot is exactly what eager NumPy does on every line, so the answer is NumPy's. Two rules decide how much is computed:

- **A number, a truth value or a string: the array asked for, and no more.** Nothing that leaves as a Python number can be written into, so nothing else needs settling.
- **An array handed out, or an `ndarray` about to be written: everything pending.** Whoever receives an `ndarray` may write into it, and a pending operation may still read the memory about to change. Once nothing is pending, the program is in the state the eager program would be in.

What continues from an array that left is plain NumPy: `np.asarray(x)` is an `ndarray`, and what a library computes from it is computed at once and is in no graph. The program is right; it may be less lazy than it looks. Nothing is hidden, though. With the logger `hotkernels.lazy` turned on, each such line is reported once, however often it runs:

```text
eager: np.argsort at session.py:18
needed: float() at session.py:10
needed: A truth test (if, while, and, or, not) at session.py:11
```

An `eager` line is a call the deferred set does not cover yet, and the only kind shown at `INFO`: it is the list of what the set should cover next. A `needed` line is a place where the program's graph was cut. `hotkernels.lazy_report()` ranks every occurrence of both, the call a program makes six hundred times at once at the top.

### Several at once, and a loop that keeps its state lazy

`hotkernels.gather(w, b)` computes several arrays in one walk of the graph and returns them as a tuple; with no argument it computes everything pending. Inside a loop, `w.gather()` computes this step now and leaves `w` a lazy array, now with its values, so the next step's operations are nodes again:

```python
w_now, b_now = hotkernels.gather(w, b)     # two results, one walk of the graph

for batch in batches:
    w = w - lr * gradient(w, batch)
    w.gather()                             # compute this step now; w stays a lazy array
```

A long loop that never gathers cannot grow its graph without end: past 100,000 pending nodes the layer computes everything pending.

### Strict arrays

For code that must not compute by accident, `hotkernels.get_lazy(strict=True)` gives arrays that raise `NotGathered`, a `TypeError`, at each of those places instead:

```python
>>> np = hotkernels.get_lazy(strict=True)
>>> total = (np.arange(6.0) * 2 + 1).sum()
>>> float(total)
NotGathered: float() needs the values of a lazy array: call .gather() first
>>> float(total.gather())
36.0
```

Strictness belongs to the array and to everything computed from it, not to the process, so one part of a program can be strict and the rest not. A strict array raises even when it has been computed already; otherwise the program would behave differently depending on whether something had happened to gather it earlier.

Strict arrays exist for a second reason, which is how the layer is tested. NumPy can reach an object's memory through attributes such as `__array_interface__` and `__array_struct__`, which it asks for before it calls `__array__`. A lazy array that answered them, say by passing every unknown attribute on to its computed value, would hand its memory over with no log line and with nothing pending computed first. With computing on the spot the results would still look right whenever the array happened to be computed already, so nothing would show. On a strict array every known road to the values raises, so a road nobody thought of is the one that does not, and a test finds it. The layer's own tests run strict for that reason, and a lazy array passes on ordinary methods only, never a name with double underscores.

## Writes, and why a loop stays one graph

Programs write into arrays all the time, and loops most of all: `w -= learning_rate * gradient`, `x[mask] = 0`, `np.clip(g, -5, 5, out=g)`. If every write made the layer compute, a training loop would be computed one statement at a time and no graph would be longer than a line. So a write is a node too.

```python
w = np.zeros(3)             # the layer made it
before = w + 1              # reads what w holds now
for step in range(50):
    w -= 0.1 * (w - 1)      # nothing is computed: w now stands for a new node
w.gather()                  # fifty steps, one graph
before.gather()             # array([1., 1., 1.]): what w held when the line ran
```

- **The array gets new contents; nothing that exists is changed.** `w -= s` writes a node for "what `w` held, with `s` subtracted", and the name `w` stands for that node from then on. An operation written earlier keeps reading the node `w` stood for when it was written, so it gets the old contents whenever it is computed. That is why nothing has to be computed first.
- **Every name and every view sees the write.** The lazy array is one object under all its names, and it is the object that moves to the new node. A view such as `w[1:]` or `w.T` is kept as "this view of whatever `w` holds", and a write through a view is a write into `w` at that place.
- **Where the memory can be seen outside the layer, the write is made at once.** NumPy's meaning of a write is that memory changes, there and then, for everyone who can see it. An array the stand-in made is seen by nobody else. An `ndarray` the user handed in, or one `gather()` returned, is held by the user, and a write into it computes everything pending and then writes, as NumPy would.

Other systems that build a graph put assignment in it: TensorFlow's graphs had an operation for assigning to a variable, and JAX has `x.at[i].set(v)`. The difference here is that the program is not rewritten: it says `x[i] = v`.

A deferred write is defined as a copy of what the array held with the write made into the copy. It is computed that way only when something still reads the old contents. When the write is computed, the layer asks who can still read them: an operation written before the write and not computed yet, or a view that something still holds. If nobody can, the write is made into the array's own memory, as NumPy makes it. Reference counts answer the question, and CPython keeps them exactly, so a count that is higher than it needs to be costs a copy, never a wrong value. Which way it goes can depend on what is asked for together: with `x.gather()` alone, an operation that read `x` before the write and belongs to another result is still pending, and the write copies; `hotkernels.gather(x, y)` computes both in one walk, the reader first, and the write is made in place.

The figure is a linear regression of 30 gradient steps, run on the prototype under each setting. It shows the graphs the layer computed:

<figure class="widget">
<div data-widget="lazycuts" data-look="5" data-write="inplace"></div>
<figcaption>The graphs a 30-step regression is computed in, each as wide as its number of operations. Printing the loss needs a value and cuts the graph; a write into an array only the layer can see does not; a write into an ndarray of your own cuts it at every step. Hatched: operations written and never computed, the losses nobody printed. Recorded with the prototype's <code>record_graphs()</code>.</figcaption>
</figure>

The obvious alternative, computing everything pending at every write, was measured too. It is always right, and over the ten programs below it made 1,133 writes and left no graph of a training loop longer than 15 operations. With writes as nodes, three iterations of Karpathy's character-level recurrent network, which writes into an array on nearly every line of its inner loop, are one graph of 2,059 operations.

## What changes for the user

"Type" means two things for an array, and only one of them changes. **The element type does not.** `x.dtype` is the `numpy.dtype` object NumPy would give the array, and promotion is NumPy's: the sum of `int8` is `int64`. **The container is one new class, `hotkernels.LazyArray`.** It cannot be `ndarray` or a subclass of one, for the reason the chapter began with: an `ndarray` owns memory any code may read at any time, and a pending array has no memory yet. One class serves every result, including those NumPy returns as scalars; the shape of a scalar is `()`.

Where that shows in a program:

| In the program | With lazy arrays | Change needed |
|---|---|---|
| `x.dtype`, `np.float32`, `x.astype(np.float32)` | NumPy's own | None |
| `type(x)`, `print(x)` | `hotkernels.LazyArray`; a description, without values | None |
| An annotation, `def step(w: np.ndarray)` | Python does not enforce it, so the function runs | None to run |
| `isinstance(x, np.ndarray)` | False | Yes: `isinstance(x, (np.ndarray, hotkernels.LazyArray))`. The one place a program has to change, and the one place it can take another branch without a word |
| `x.view(type(x))`, how NumPy code keeps the kind of array it was given | The view `x.view()`, deferred | None where the array viewed is lazy |
| A reduction's result, `loss = (r ** 2).mean()` | A `LazyArray` of shape `()`; `loss.gather()` is an `np.float64` | None: `float(loss)` and `f"{loss:.3f}"` compute it |
| `np.isscalar(loss)` | With the stand-in, NumPy's answer for the value. The real `numpy.isscalar` says False of every lazy array | None in a module whose `np` is the stand-in; a library that asks with its own `numpy` is handed `loss.gather()` |
| A library that takes an array: pandas, matplotlib | It asks NumPy for an array and gets an `ndarray`, computed on the spot | None |
| Code that reads memory directly: Cython's `double[:]`, `memoryview(x)` | `TypeError`: a lazy array has no buffer to give | Pass `x.gather()` |

A program that is partly lazy has six things to know, and the design lists them all:

1. **A value needed without `gather()` is computed where it is needed,** and what continues from an array that left is plain NumPy. Right, and perhaps less lazy than it looks; the log says where.
2. **A lazy array is not an `ndarray`**, as in the table.
3. **A pending expression reads a plain `ndarray` when it is gathered, not when it is written down.** This is the one that can change a result:

   ```python
   a = loader.read()        # a plain ndarray: [0., 1., 2.]
   y = np.asarray(a) * 2    # pending; np is the stand-in
   a[0] = 5                 # written directly to the ndarray
   y.gather()               # array([10., 2., 4.]); eager NumPy computed y before the write: [0., 2., 4.]
   ```

   Writes made through a lazy array are safe, by the rule for writes; a write made directly to the `ndarray` is not seen by the layer. Gather before such a write. Whether inputs should be copied instead is the first of the design's [open questions](#what-is-not-settled).
4. **Warnings and errors that depend on values come at the gather.** They are NumPy's warnings with NumPy's words, and each names the file and line that wrote the operation, not the line that gathered. `np.errstate` is remembered with each node, so `r = a / b` written under `np.errstate(divide="ignore")` stays silent wherever `r` is gathered.
5. **An operation that failed stays failed.** Under `np.errstate(over="raise")` the exception is raised by the first gather that needs the operation, with a note naming its line, and again by any later one.
6. **The work happens at the gather,** so that is where time passes, memory is taken, and `KeyboardInterrupt` lands.

## The same answer, bit for bit

The design's contract is one sentence: **what a program gets from `gather()` is what the same lines give with `import numpy as np`**, in type, element type, shape and bits. Without fusion it holds by construction, because the graph is computed one node at a time, in the order the program wrote it, **each node with the call the user wrote**, spelled the same way. `x.sum()` is computed as `value.sum()`, not as `np.sum(value, axis=None)`. Warnings that NumPy issues while a graph is computed are caught and issued again at the line that wrote the node, with NumPy's message and category, so Python shows each once per line as it would for the eager program.

That last rule looks pedantic and is not. With `hotkernels.install()` in effect, [a kernel's hook decides by how a call is spelled]([[ root ]]report/delivery/#one-call-several-hooks): the `sum_f64` kernel takes `x.sum()` and leaves a sum that names its `axis`, even `axis=None`, to NumPy. A layer that tidied every call into one form would give NumPy's results and miss the kernels. Kept as written, the lazy program makes the calls the eager one makes, on the same operands, through the same pointers, and reaches the same kernels: `np.where`, `np.clip`, `np.argmax`, `np.searchsorted`, `x[mask]` and `np.sum` on lazy arrays were each served by their kernel.

### With `hotkernels.install()`

`get_lazy()` and `install()` are independent, and either may come first: the layer holds no pointer of NumPy's. They are meant to be used together, and fusion runs that way by default: `fuse.enable()` has `install()` in effect unless it is told not to. Three things follow.

- **The results are the eager program's under the same installation.** With `install()` as it is by default, those are NumPy's bits.
- **`hotkernels.report()` counts one dry run more.** Working out an element type makes the real call on arrays of one element, and a kernel that takes arrays of any size takes that call. The answer is remembered, so the dry run is made the first time a line meets operands of those shapes and types and not again. In the regression, `sum_f64` took 50 calls eagerly and 51 lazily: the 50 at the gather and one dry run on the first step.
- **A call the layer never makes is a kernel the program loses.** `grid[mask]` with `grid` a plain `ndarray` and `mask` a lazy array is NumPy's own indexing of an `ndarray`, and the hook on `x[mask]` sees an index that is not a boolean `ndarray` and leaves it. A Mandelbrot program whose grids came from `np.mgrid`, which the stand-in did not make lazy, ran 16% behind the eager program with the kernels installed, measured as part of fusion. Every array the stand-in can make is now lazy, the ones `np.mgrid`, `np.ogrid`, `np.r_` and `np.c_` make among them, and the program runs 1% behind.

### How it is checked

- **Tests in three files** (`tests/test_lazy_prototype.py`, `test_lazy_graphs.py`, `test_lazy_building.py`), each holding one rule of the design to what the real `numpy` module gives for the same lines. Every test runs twice, in Python alone and with the C core described below, except one, a test of the core's own answers, which runs with the core only.
- **Ten programs with one changed line**, five written for the purpose and five from the repository's applications, all give NumPy's results bit for bit:

| Program | Operations deferred | Graphs | Largest graph |
|---|---|---|---|
| Ocean model, equation of state (48-term polynomial) | 366 | 1 | 366 |
| The Mandelbrot set, 30 iterations | 187 | 4 | 179 |
| Heat on a plate, 60 steps written into a slice | 785 | 5 | 261 |
| Ocean model, isoneutral mixing | 698 | 19 | 119 |
| `mlp_train.py`, one epoch | 516 | 11 | 50 |
| Karpathy's `min-char-rnn.py`, 3 iterations | 3,245 | 27 | 2,059 |
| A particle filter, 20 steps, with masks and resampling | 1,327 | 24 | 140 |
| k-means, four clusters | 99 | 10 | 19 |
| PCA of 2,000 × 12 | 25 | 3 | 11 |
| A Monte Carlo option price | 13 | 2 | 10 |

What still cuts their graphs is the program asking for a value, 85 times over the ten: a test for convergence, a loss turned into a `float`, a number formatted for output. Those are the program asking, and no rule of the layer removes them. Only seven calls over all ten programs were made at once: `np.stack` four times, `np.concatenate`, `np.linalg.eigh` and `np.argsort` once each.

## What writing a graph costs

Writing a node down takes time, and with small arrays it can take longer than NumPy computing the node would. A node needs its shape and element type, the line of the user's code that wrote it (for the log and for warnings), and the error settings in effect. Asking all of that for every node costs 14 to 19 µs; NumPy multiplies two arrays of 8,192 float64 values in 1.9 µs.

### What is remembered, and why it is safe

A loop asks the same questions on every turn. So the layer **remembers** the answers, and each memory is an exact one: it gives the answer that asking again would give, or it gives nothing and the question is asked. That needs the key to hold exactly what the answer depends on, and nothing that only looks the same.

- **A shape and an element type depend on the operation, on each array's shape and element type, and on every other argument as written,** and on no value: the stand-ins NumPy is asked about have no values. So that is the key. An array is written into it as a number for its shape and its `dtype` object, the object and not what it compares equal to, since two `dtype`s that are equal can differ in metadata that NumPy's answer carries.
- **An argument that is no array is written with its type and its value,** because `1`, `1.0` and `True` are equal to Python and three different arguments to NumPy: `axis=1` and `axis=True` must not share an answer.
- **A number that is an operand is written by what decides NumPy's answer.** A float is written by its type alone: NumPy 2 takes the result's type from the operands' types, and no float makes a ufunc raise on the stand-ins, so `x * rate` is one key whatever `rate` is. A whole number is written with its value, which can decide whether NumPy raises: `uint8 + 1` is fine and `uint8 + 300` is an error.
- **A length that is not known is written as a placeholder,** numbered in the order met, since every mask has its own mark and a call on what a mask selects would otherwise never be seen twice.
- **Anything else is not written down,** and the call is answered from nothing: a list, a subclass, a call that raised or was made at once.
- **The line of the user's code** is found once per place in the program, from the code object and the instruction running, and **the error settings** are asked again only when NumPy has replaced the record that holds them, which it does whenever `np.errstate` changes them.

Every memory is bounded, and emptied when full. How such a memory is shown to be right is the interesting part. A test holds each one to the answer worked out from nothing, for every kind of deferred call, and tells apart the pairs that Python finds equal and NumPy does not. Forty random programs of 150 operations each, with writes, masks, views read after a write and errors, give the same nodes, lines, exceptions and values with every memory switched off, with the memories empty, and with them full. And the tests themselves were tested: sixteen wrong versions of the memories, each one changed line, were tried against them, and each fails at least one.

### The C core

With the memories, writing a node takes 1.4 to 2.5 µs in Python, and what is left is the interpreter's own work: some ten calls of Python functions and five objects per node. Going below that means doing the work outside the interpreter. The prototype's **C core**, [_core.c](repo:prototypes/lazy/_core.c), is that: a second, quicker path, written in C, for the Python functions that write a node.

- **The Python is the definition.** Each path in the core stands for a Python function that exists, and a node the core makes is, field for field, the node that function would have made. Wherever the core is not sure (an argument it cannot write into a key, a subclass, shapes that do not fit), it hands the call to the Python unchanged.
- **It decides three things about a shape itself,** broadcasting, the matrix product and the axes of a reduction, each also written in Python and held to it by a test, and nothing about an element type: every type is still NumPy's answer.
- **It is optional.** `python -m prototypes.lazy.build_core` compiles it, and the layer runs the same without it.
- **It is tested as the memories are.** Every test of the layer runs with it and without it, and of forty-one wrong versions of the core, each one changed line, thirty-nine fail a test. The other two change nothing a program can see.

<figure class="widget">
<div data-widget="nodecost" data-road="remember"></div>
<figcaption>The time to write a * b down as a node, asking every question each time, remembering the answers in Python, and with the C core, beside NumPy's time to compute a * b on two float64 arrays of 16 to 8,192 elements. Both axes are logarithmic. The dashed line marks the size at which writing the node and computing it cost the same. Measured on an Apple M1 Pro, the quickest of several runs.</figcaption>
</figure>

For whole programs, written down and dropped without a gather, so that the time is the writing alone; the quickest of 15 runs on the Apple M1 Pro:

| Program | Operations | In Python | With the C core | NumPy computing it, on 200 × 8 and 16 × 32 arrays |
|---|---|---|---|---|
| The regression, 50 gradient steps | 954 | 1.67 ms | 0.42 ms | 0.43 ms |
| A transformer block, 10 in a row | 550 | 1.35 ms | 0.37 ms | 1.17 ms |

With the core, a node costs what NumPy's own work costs on about five hundred elements. So computing a graph node by node is slower than NumPy on small arrays, and close to it on large ones. For the layer by itself that was accepted: it is there to get the graph right. The first time a line runs costs more, because it asks its questions then: the regression of 50 steps, built once in a fresh process, takes 1.17 ms where asking every time took 17.1 ms, and NumPy computes it for the first time in 0.46 ms. The core now does more than write nodes: for fusion it reads the expressions off a computation, runs the loop that computes the nodes, and makes small calls at once without the graph, which [the fusion chapter]([[ root ]]report/fusion/) describes.

## Looking at the graph

A layer that makes nothing faster by itself produces a graph and nothing else, so the graph has to be easy to look at. `y.explain()` prints what is pending, one operation per line with the line that wrote it:

```text
sum       float64 ()            program.py:4
  exp       float64 (6,)          program.py:3
    divide    float64 (6,)          program.py:3
      add       float64 (6,)          program.py:2
        multiply  float64 (6,)          program.py:2
          array     float64 (6,)
```

An operation that several others read is printed once and marked where it comes again; without that, the text of a loop, in which each step reads the step before twice, would double with every step. Text is enough for a dozen operations. For more there is a recording mode and a page.

### One graph per computation

A node forgets its operation and its operands once it is computed, which is what lets the memory of intermediates go. So graphs are kept only when asked, inside a `with` block, as plain data that holds no array:

```python
with hotkernels.record_graphs() as recording:
    weights, losses = fit(np, features, targets)
    hotkernels.gather(weights, *losses)

recording.graphs        # one graph per computation, in the order they ran
recording.show()        # the page, in the browser
```

A recorded graph is **the nodes one computation computed**, with what made the layer compute and on which line: a `gather`; a call made `at once` that needed its operands; a `write` made at once into memory known outside the layer; a value `needed` without `gather()`; the `limit` of pending nodes; and what was still `pending` when the block ended. That unit is chosen because it is exactly what anything that computes a graph differently is handed, and because every one of those triggers is a cut: seeing them, with their lines, is what tells a user why their program's graph is in forty pieces. The `needed` ones matter most, since nobody wrote `gather()` there and the code does not show them. Graphs that are alike carry one signature, so a loop that gathers at every step is one graph, fifty times. The prototype's three example programs in one block give 15 graphs and 1,336 operations, and 78 more that were written and dropped: the losses of the steps nobody reported.

### Folding a loop

A graph of a loop is too large to read node by node: 954 operations for 50 steps of the regression. So each recorded graph carries its operations a second time, as a tree of groups: the calls of a function, the operations of one run of a line, and the steps of a loop. Calls come from the stack at the moment a node is made, with the layer's and NumPy's frames left out; one call of a function is told from the next by its frame object. Loops are not read from the source at all, which is not always there, and not traced, which would slow every line. They are found in what was recorded: runs of members that are alike and follow each other. Try it:

<figure class="widget">
<div data-widget="lazyfold" data-steps="50" data-stage="repeats"></div>
<figcaption>How the recording turns the regression's 954 operations into a tree that opens on four operations and one box. The regression is written here as one function, with a line that can make one step differ. Choose the number of steps, or make one step write one operation more, and step through the three moves. The tree below is what the page opens with; click a group to unfold it. The rules are the prototype's, ported to the page, and for every setting the tree was checked against the one the prototype recorded.</figcaption>
</figure>

The rules, each a constant of [record.py](repo:prototypes/lazy/record.py) chosen by looking at three programs:

- **A line is a run.** Consecutive operations written on one line under one call are one member.
- **A member's signature** is a digest of what a reader sees: each operation's name, element type, shape and line. Two runs of one line are alike when their operations are.
- **A repeat is a run of members that comes three times or more in a row.** At each member the search tries every period up to 64 members and keeps the one that covers the most. It is greedy from the left, so where one step differs, the steps it finds may begin in the middle of a turn of the loop; the figure's "the 2nd" shows it.
- **A long repeat lists its ends.** Above twelve steps it lists the first two and the last and holds the others in one group, split into groups of fifty when there are more.
- **A long call is cut into stretches** of eight members when it has more than sixteen, so a function of a hundred lines that are not alike does not open as a hundred boxes. The ocean model's equation of state, one function of a hundred lines, is why the rule exists: it opened as 366 boxes.
- **The tree cannot lose anything.** Every node is in the graph whatever the tree says; a wrong grouping costs a reader a click.

### The page

`recording.show()` writes one HTML file with the graphs in it as JSON, and opens it. It opens on the program's code, because a drawing of a graph says what was computed together and not where in the program that is. Every line says how many operations it wrote, which graphs they went into, and what it made the layer compute:

<figure>
<img src="[[ root ]]static/images/lazy-graph-page-code.png" alt="The graph page showing a program's code: the lines of the selected graph marked, and on each line how many operations it wrote, which graphs they went into, and marks such as needed and eager for what it made the layer compute" loading="lazy">
<figcaption>The prototype's page, on the code of a regression that prints its loss every fifth step. The list on the left is the recording's graphs, with what triggered each; the marks on the lines are the log in another form, put on the lines they are about.</figcaption>
</figure>

The drawing of a graph is one click from its line. A layout library places the boxes, and a group of more than five operations is drawn folded, as a stack of cards that says what it holds. While the top of a graph holds one folded group and nothing else, that group is opened, since a page that opens on one box says nothing. For the regression that is the four operations of its first line and one box for the loop:

<figure>
<img src="[[ root ]]static/images/lazy-graph-page-loop.png" alt="The graph page of the prototype: a list of graphs on the left, the regression's graph in the middle with the loop of 50 steps drawn as one stacked box, and a panel of counts on the right" loading="lazy">
<figcaption>The regression's graph. Colours are kinds of operation: element by element, matrix product, reduction, view, write. The stacked box is the loop, × 50, and unfolds one level on a click.</figcaption>
</figure>

The colours make the shape fusion looks for visible at a glance: a chain of element-by-element operations ending in a reduction. Graphs are cut, so a graph often reads what an earlier one computed; such a node is drawn dashed, labelled with the graph that computed it, and a click goes there:

<figure>
<img src="[[ root ]]static/images/lazy-graph-page-earlier.png" alt="A graph on the page that reads three results of earlier graphs, drawn dashed and labelled with the graph each came from; the outputs the gather asked for have a thick border" loading="lazy">
<figcaption>One of the graphs of the loop that prints its loss every fifth step. The three dashed nodes were computed by earlier graphs; the thick border and <code>out</code> mark what the gather asked for.</figcaption>
</figure>

Search finds operations inside folded groups and unfolds what hides them, and the list of graphs can be put largest first, because in a real program the large graphs are few and not where the list starts: the particle filter's 33 entries start 52, 3, 51, 3, and largest first puts 140, 60, 52 at the top. To see it for yourself, this records the three example programs and opens their page:

```text
uv run python -m prototypes.lazy.programs.pages --out lazy-graphs.html --show
```

The prototype's page loads its layout library from a CDN, so it needs a network connection; the package's would carry the library inside it.

## Other ways in, and why not

Two more ways to get lazy arrays into a program were designed and left out of the prototype. Both are as safe as `get_lazy()` in the respect that matters: nothing is patched.

- **A module that can be imported:** `import hotkernels.numpy as np`, the way `import jax.numpy as jnp` and `import cupy as cp` offer themselves in place of NumPy. It gives what an assignment cannot, `from hotkernels.numpy import exp`, and a stub file beside it could tell an editor that its names are NumPy's, which would give completion back; that was not tried. It costs a second spelling of one thing.
- **A decorator on one function**, `@hotkernels.lazy`: the function's `ndarray` arguments enter lazy and what it returns is gathered before the caller gets it. Nothing lazy leaves the function and no type changes outside it, which suits one hot function in a program that otherwise stays as it is. Its limit is that the graph ends at every return, so a loop that calls the function gets one graph per call. It is a few lines once the layer exists.

And the ways that were considered and rejected:

| Option | Why not |
|---|---|
| Replace the `numpy` module for the whole process, in `sys.modules` or by patching NumPy's functions | It fails twice. A module that imported NumPy first, or wrote `from numpy import zeros`, keeps the real one, so the result depends on import order. And even in the right order, every library and NumPy's own Python would be handed arrays without memory, in code that never asked for them |
| Make real `ndarray`s lazy, or a subclass of `ndarray` | An `ndarray`'s memory is public C API, and a subclass has a data pointer that must hold values at all times. Laziness has to be a type of its own, in code that asked for it |
| Raise wherever a value is needed | A real program would have to be rewritten before its first graph could be looked at. Raising is kept, as strict arrays |
| One switch for strictness for the whole process | A flag on the array needs nothing global, and lets one part of a program be strict while a library it calls is not |
| Dask, JAX or numexpr | Each has semantics of its own: chunked and parallel arrays, another library's type rules, expressions as strings. The contract here is NumPy's bits, so that kernels judged against NumPy can compute the graph |

## What the graph is for

Everything above is there so that a graph exists, is right, and costs little to write. Its use is the one the chapter opened with: once an expression is in the graph whole, it can be computed in one pass instead of one call at a time. In the graph, the eight operations of GELU are visibly what they are: element-by-element nodes of one shape, each intermediate read once, by the node above it. Nothing else needs those intermediates, so nothing needs them to exist in memory.

[The next chapter]([[ root ]]report/fusion/) builds that, in the prototype. A **pattern** is a small tree of operations with named inputs, such as `a + s * b`, and a kernel written for it computes it in one pass. The patterns are not a list kept by hand: each fused kernel's `spec.py` already names the NumPy expression it must equal, and that expression, run on lazy arrays, writes a graph whose shape is the pattern. A long expression is covered with several patterns, and their kernel calls are run as a program a block of 4,096 elements at a time, so what one kernel gives the next stays in the cache. Eighty kernels do this, fourteen written by hand and sixty-six by a program from the names of their shapes, and each passes [the judge]([[ root ]]report/judge/) against its NumPy expression. Every operation in them is the single IEEE operation NumPy makes, in the program's order, so the bits are still NumPy's.

Fusion changes one thing about this layer. With it switched on, a call on arrays that all have their values, if it gives fewer than 32,768 elements or is one no kernel could make, is NumPy's call at that line and not a node: on small arrays a node costs more than the work it describes, and there is nothing to fuse. The graphs this chapter draws are the layer with fusion off, where every operation the layer defers is a node.

[The chapter after it]([[ root ]]report/fusion-results/) measures the result on twenty-six programs, on the Apple M1 Pro, run as the package is used, with `hotkernels.install()` in effect. Against the same programs on plain NumPy, all twenty-six run within 1% of NumPy's speed or faster, up to 5.1x. Where a program is expressions on large arrays the gain is fusion's: a cavity flow 2.80x, a weather model's horizontal diffusion 2.36x, an ocean model's equation of state 1.74x. Against the eager programs with the same kernels installed, the fused run is ahead on seven, by 1.10x to 2.79x, within 3% on fifteen, and behind on four.

The graph carries one more thing a hook cannot have: facts it would otherwise read memory for. A mask that came from a comparison holds only 0 and 1, and a temporary the graph owns overlaps nothing. The binding of [the delivery chapter]([[ root ]]report/delivery/) scans memory for both today.

## What is not settled

The design lists its open questions, and the prototype picks one answer for each so that it is defined. Among them:

- **Should an `ndarray` be copied when a pending expression first reads it?** Holding it by reference costs nothing and leaves the third of the six things to remember; copying removes it at the price of a copy each time an expression meets the array.
- **What carries a length that is not known?** The prototype uses an integer larger than any array can be, so that NumPy's own shape arithmetic works on it unchanged, and keeps it away from anything that would allocate by it. A type of its own is the sound form.
- **How wide should the deferred set be?** It covers both of the design's example programs with no call made at once. The ten programs asked for `np.stack` and `np.concatenate` next; each costs a rule for its shape.
- **Should the C core ship in the package**, built with the binding's C and in the wheels, with the Python as its definition and fallback?
- **What one recorded graph is.** One per computation, as above, shows where a program was cut. The connected pieces of everything a block wrote would show its structure instead. Both can be had from the same records; the question is which the page opens on.

Everything above was run on ARM64 macOS with NumPy 2.5.5 and Python 3.12. The layer's design did not try other libraries handed a lazy array, such as pandas or matplotlib; of the programs fusion was measured on, one hands lazy arrays to scikit-learn. The layer's C core is portable C that was compiled and run on ARM64 macOS only.
