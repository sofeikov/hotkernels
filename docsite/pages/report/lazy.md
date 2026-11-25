<p class="lede">Every kernel in this report makes one NumPy call faster. A line of real code is rarely one call: <code>np.exp(a * b + c).sum()</code> is four, and NumPy makes them one after another, each reading its inputs from memory and writing a whole new array for the next to read back. Doing the four in one pass needs something that sees the expression whole, and no hook into NumPy ever does. This chapter is about a design that does: NumPy code whose operations are written down as a graph instead of being computed, and computed when the program asks.</p>

::: note A design with a prototype, not part of the package
What this chapter describes is a proposal under review, [docs/lazy-design.md](repo:docs/lazy-design.md), and a prototype that follows it, in [prototypes/lazy/](repo:prototypes/lazy). Nothing here is installed with `hotkernels`, and the names below (`hotkernels.get_lazy()`, `gather()`) are the ones the design proposes. **The layer this chapter describes makes nothing faster, and is not meant to.** Its first job is to write the graph down correctly and give NumPy's results bit for bit; what the graph is for comes [at the end](#what-the-graph-is-for), and a first build of it, kernels that compute a whole expression in one pass, is a document of its own, [docs/lazy-fusion-design.md](repo:docs/lazy-fusion-design.md). Every behaviour shown in the figures was run on the prototype, and every timing was taken on an Apple M1 Pro.
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

From that line on, every array the program makes is a `LazyArray`. Operating on one does not compute anything: it adds a **node** to the graph, a record of which operation to make on which operands. Each node knows its shape and element type at once, because those follow from the operands' shapes and types and not from their values. `gather()` is how the program asks for values: it computes what the array needs and returns exactly what NumPy would have returned, an ordinary `numpy.ndarray`.

Step through four short programs and watch the graph grow. Switch to **NumPy** to see the same lines run eagerly.

<figure class="widget">
<div data-widget="lazytrace" data-program="expression"></div>
<figcaption>Four programs run one line at a time. A dashed box is pending: written down, with no values yet. A filled box has its values, and the numbered badge says which computation produced it. The dark labels are the program's variables and the node each one stands for. Everything shown, down to the printed text and the log lines, is what the prototype gives.</figcaption>
</figure>

Four things to take from it:

- **A pending array still answers questions about itself.** `x.shape`, `x.dtype`, `len(x)` and `print(x)` compute nothing. Values are the only thing that waits.
- **Nothing is computed that nobody asks for.** In *A mask*, asking for the length of the selection computes the selection and leaves the mean pending.
- **Each computation is one graph.** The nodes computed together, because one thing asked for them, are the unit that anything computing a graph differently would one day be handed.
- **A value needed is computed, not refused.** In *A value needed*, the `if` needs True or False on its line. The layer computes what the test reads, logs the line, and the program carries on.

## One line, and nothing patched

`np = hotkernels.get_lazy()` is an ordinary assignment. `get_lazy()` returns an object that answers the names the `numpy` module has, and the line binds it to a name in one module. That is the whole mechanism, and it is deliberate.

- **NumPy is not changed.** `sys.modules["numpy"]` is the real module, `numpy.zeros` is the real function, and every other module's `import numpy` gives what it always gave. Replacing the module for the whole process was considered and rejected twice over: a module that imported NumPy first would keep the real one, so the result would depend on import order; and every library would be handed arrays without memory in code that never asked for them.
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
| Views | `x.T`, `reshape`, `ravel`, `np.transpose`, `np.swapaxes`, `np.expand_dims`, `np.squeeze`, slicing |
| Indexing by arrays | `x[rows]`, `x[rows, columns]`, `x[mask]`, also with an index that is itself pending: `x[np.argmax(w)]`, `x[speed > limit]` |
| Ordering and positions | `np.partition`, `np.nonzero`, `np.flatnonzero`, `np.argwhere` |

## Shapes without values

The layer has to know what NumPy would make of a call before anything is computed: the shape and element type of the result, and whether the call is an error at all. It never works these out by rules of its own where NumPy can be asked.

- **The element type is NumPy's answer.** For a ufunc, NumPy's own resolver, `np.multiply.resolve_dtypes`, gives the result's type from the operands' types. Where nothing that cheap exists, the real call is made on arrays of one element each, with the Python numbers as written. So `float32 * 2.0` stays `float32` and `float32 * np.float64(2)` becomes `float64`, as in NumPy, because it is NumPy deciding.
- **The shape comes from a rule or from a stand-in.** Element-by-element shapes follow NumPy's broadcasting, and the matrix product and reductions have short rules of their own. For a view or an index, NumPy is asked to make the same view of an array that has the operand's shape and no memory at all, an array of an element type of size zero, and the shape is read from the answer.
- **An error is NumPy's, raised on the line.** If the shapes do not fit, the real call is made on those stand-ins, and NumPy raises its own exception with its own message before it reads a byte: `x + numpy.arange(4.0)` on a (4, 3) array fails with *operands could not be broadcast together with shapes (4,3) (4,)*, on that line, as it would eagerly.
- **When in doubt, do not defer.** A call the rules do not recognise in every detail is made at once. The layer never guesses a shape or a type.

One length cannot be known: how many elements a mask selects. `particles[speed > limit]` has as many rows as the mask has true values, which is a value. The selection is deferred anyway, and its shape carries a mark, `(?, 4)`. Whatever is computed from it stays deferred wherever the result's shape does not depend on that number: arithmetic with a number, a mean along that axis, a matrix product, an assignment back through the same mask. Where NumPy's answer does depend on it, as in the first five rows `x[mask][:5]`, the call is made at once.

## When a value is needed

Values leave a lazy array in two ways. `gather()` is the one the program writes. The other is everything else that needs a real number or a real array: `float(loss)`, `if error < tolerance:`, `f"{loss:.4f}"`, `np.asarray(x)` with the real module, and every library that asks NumPy for an array, such as a plotting call.

The layer **computes there instead of raising**. A real program needs a value every few lines, and if each of those raised, it would have to be rewritten before its first graph could even be looked at. Computing on the spot is exactly what eager NumPy does on every line, so the answer is NumPy's. Two rules decide how much is computed:

- **A number, a truth value or a string: the array asked for, and no more.** Nothing that leaves as a Python number can be written into, so nothing else needs settling.
- **An array handed out, or an `ndarray` about to be written: everything pending.** Whoever receives an `ndarray` may write into it, and a pending operation may still read the memory about to change. Once nothing is pending, the program is in the state the eager program would be in.

Nothing is hidden. With the logger `hotkernels.lazy` turned on, each such line is reported once, however often it runs:

```text
eager: np.argsort at session.py:18
needed: float() at session.py:10
needed: A truth test (if, while, and, or, not) at session.py:11
```

An `eager` line is a call the deferred set does not cover yet, and the only kind shown at `INFO`. A `needed` line is a place where the program's graph was cut. `hotkernels.lazy_report()` ranks every occurrence of both.

For code that must not compute by accident, `hotkernels.get_lazy(strict=True)` gives arrays that raise `NotGathered`, a `TypeError`, at each of those places instead. Strictness belongs to the array and to everything computed from it, not to the process, so one part of a program can be strict and the rest not. The layer's own tests run strict, for a second reason: NumPy can reach an object's memory through attributes such as `__array_interface__` without calling anything. On a strict array every known road to the values raises, so a road nobody thought of is the one that does not, and a test finds it.

## Writes, and why a loop stays one graph

Programs write into arrays all the time, and loops most of all: `w -= learning_rate * gradient`, `x[mask] = 0`, `np.clip(g, -5, 5, out=g)`. If every write made the layer compute, a training loop would be computed one statement at a time and no graph would be longer than a line. So a write is a node too.

- **The array gets new contents; nothing that exists is changed.** `w -= s` writes a node for "what `w` held, with `s` subtracted", and the name `w` stands for that node from then on. An operation written earlier keeps reading the node `w` stood for when it was written, so it gets the old contents whenever it is computed. *A loop that writes*, in the step-through above, shows exactly this: `before` keeps the zeros.
- **Every name and every view sees the write.** The lazy array is one object under all its names, and it is the object that moves to the new node. A view such as `w[1:]` or `w.T` is kept as "this view of whatever `w` holds", and a write through a view is a write into `w` at that place.
- **Where the memory can be seen outside the layer, the write is made at once.** NumPy's meaning of a write is that memory changes, there and then, for everyone who can see it. An array the stand-in made is seen by nobody else. An `ndarray` the user handed in, or one `gather()` returned, is held by the user, and a write into it computes everything pending and then writes, as NumPy would.

The figure is a linear regression of 30 gradient steps, run on the prototype under each setting. It shows the graphs the layer computed:

<figure class="widget">
<div data-widget="lazycuts" data-look="5" data-write="inplace"></div>
<figcaption>The graphs a 30-step regression is computed in, each as wide as its number of operations. Printing the loss needs a value and cuts the graph; a write into an array only the layer can see does not; a write into an ndarray of your own cuts it at every step. Hatched: operations written and never computed, the losses nobody printed. Recorded with the prototype's <code>record_graphs()</code>.</figcaption>
</figure>

Run over ten real programs, unchanged but for their `np`, this rule is what lets a loop stay whole. Computing everything at every write was measured too: it is always right, and over the same ten programs it left no graph of a training loop longer than 15 operations. With writes as nodes, three iterations of Karpathy's character-level recurrent network, which writes into an array on nearly every line of its inner loop, are one graph of 2,059 operations.

## The same answer, bit for bit

The design's contract is one sentence: **what a program gets from `gather()` is what the same lines give with `import numpy as np`**, in type, element type, shape and bits. It holds by construction in the first build, because the graph is computed one node at a time, in the order the program wrote it, **each node with the call the user wrote**, spelled the same way. `x.sum()` is computed as `value.sum()`, not as `np.sum(value, axis=None)`.

That last rule looks pedantic and is not. With `hotkernels.install()` in effect, [a kernel's hook decides by how a call is spelled]([[ root ]]report/delivery/#five-places-to-stand): the `sum_f64` kernel takes `x.sum()` and leaves a sum that names its `axis`, even `axis=None`, to NumPy. A layer that tidied every call into one form would give NumPy's results and miss the kernels. Kept as written, the lazy program makes the calls the eager one makes, on the same operands, through the same pointers, and reaches the same kernels: `np.where`, `np.clip`, `np.argmax`, `np.searchsorted`, `x[mask]` and `np.sum` on lazy arrays were each served by their kernel.

How it is checked:

- **94 tests**, each holding one rule of the design to what the real `numpy` module gives for the same lines. All but one run twice, in Python alone and with the C core described below; the one is a test of the core itself.
- **Errors and warnings stay NumPy's.** A division by zero that is gathered later issues NumPy's warning with its own words, at the file and line of the division, not of the gather. `np.errstate` is remembered with each node.
- **Ten programs with one changed line**, five written for the purpose and five from the repository's applications, all give NumPy's results bit for bit:

| Program | Operations deferred | Graphs | Largest graph |
|---|---|---|---|
| Ocean model, equation of state (48-term polynomial) | 366 | 1 | 366 |
| The Mandelbrot set, 30 iterations | 187 | 4 | 179 |
| Heat on a plate, 60 steps written into a slice | 785 | 5 | 261 |
| Karpathy's `min-char-rnn.py`, 3 iterations | 3,245 | 27 | 2,059 |
| A particle filter, 20 steps, with masks and resampling | 1,327 | 24 | 140 |
| k-means, four clusters | 99 | 10 | 19 |

What still cuts their graphs is the program asking for a value, 85 times over the ten: a test for convergence, a loss turned into a `float`, a number formatted for output. Only seven calls over all ten programs were made at once: `np.stack` four times, `np.concatenate`, `np.linalg.eigh` and `np.argsort` once each.

A program that is partly lazy has six things to know, and the design lists them all. The one that can change a result is this: a pending expression reads a plain `ndarray` when it is gathered, not when it is written down, so a write made directly to that `ndarray` in between is seen. Gather before such a write. Whether inputs should be copied instead is one of the design's open questions.

## What writing a graph costs

Writing a node down takes time, and with small arrays it can take longer than NumPy computing the node would. A node needs its shape and element type, the line of the user's code that wrote it (for the log and for warnings), and the error settings in effect. Asking all of that for every node costs 14 to 19 µs; NumPy multiplies two arrays of 8,192 float64 values in 1.9 µs.

But a loop asks the same questions on every turn. So the layer **remembers** the answers, keyed by exactly what the answer depends on: the operation, the shape and element type of each array, and every other argument with its type and value, because `1`, `1.0` and `True` are equal to Python and three different arguments to NumPy. A test holds every memory to the answer that asking again would give. Then a prototype in C, the **core**, does the rest of a node's work outside the interpreter:

<figure class="widget">
<div data-widget="nodecost" data-road="remember"></div>
<figcaption>The time to write a * b down as a node, by each of the three roads, beside NumPy's time to compute a * b on two float64 arrays of 16 to 8,192 elements. Both axes are logarithmic. The dashed line marks the size at which writing the node and computing it cost the same. Measured on an Apple M1 Pro, the quickest of several runs.</figcaption>
</figure>

For whole programs, written down and dropped without a gather, so that the time is the writing alone:

| Program | Operations | In Python | With the C core | NumPy computing it, on small arrays |
|---|---|---|---|---|
| The regression, 50 gradient steps | 954 | 1.67 ms | 0.52 ms | 0.43 ms |
| A transformer block, 10 in a row | 550 | 1.35 ms | 0.50 ms | 1.17 ms |

With the core, a node costs what NumPy's own work costs on a little under a thousand elements. On small arrays the first build is therefore slower than NumPy, and close to it on large ones. That is accepted: the first build is there to get the graph right. Computing it faster is what [docs/lazy-fusion-design.md](repo:docs/lazy-fusion-design.md) takes up.

## Looking at the graph

The first build produces a graph and no speed, so the graph has to be easy to look at. `y.explain()` prints what is pending, one operation per line with the line that wrote it:

```text
sum       float64 ()            program.py:4
  exp       float64 (6,)          program.py:3
    divide    float64 (6,)          program.py:3
      add       float64 (6,)          program.py:2
        multiply  float64 (6,)          program.py:2
          array     float64 (6,)
```

For more than a dozen operations there is a recording mode and a page. `hotkernels.record_graphs()`, around a block of code in a `with` statement, keeps every graph the block produced, the computed ones included, as plain data, and `recording.show()` opens them in the browser. The page opens on the program's code, with each line marked by what it wrote and what it made the layer compute, and draws each graph with large parts folded: the 954 operations of the regression's 50 steps open as four operations and one box for the loop.

<figure>
<img src="[[ root ]]static/images/lazy-graph-page-loop.png" alt="The graph page of the prototype: a list of graphs on the left, the regression's graph in the middle with the loop of 50 steps drawn as one stacked box, and a panel of counts on the right" loading="lazy">
<figcaption>The prototype's page for the regression. Colours are kinds of operation: element by element, matrix product, reduction, view, write. The stacked box is the loop, × 50, and unfolds one level on a click.</figcaption>
</figure>

## What the graph is for

None of this is part of the first build, and no gain is claimed here. It is why the first build is shaped the way it is. The third way of the second point has a first build in the prototype, with fourteen kernels and its gain measured on twelve programs: [docs/lazy-fusion-design.md](repo:docs/lazy-fusion-design.md).

- **A fusible group can be read off the graph:** connected element-by-element nodes of one output shape, perhaps ending in a reduction. The eight operations of GELU are one; in the regression, `(X @ w - y) ** 2` and its `mean` are one after the matrix product. [Figure 1](#one-call-at-a-time) counts what computing such a group in one pass would save.
- **A group can be computed three ways, and the graph does not care which:** node by node with NumPy, as now; with NumPy's own calls a cache-sized block at a time, which removes the passes over main memory and keeps NumPy's bits; or with one kernel for the whole expression.
- **A fused kernel fits the existing contract unchanged.** A kernel's spec already names its oracle as a NumPy expression. Run that expression on lazy arrays and the result is the graph the kernel computes: the specification is its own pattern, and [the judge]([[ root ]]report/judge/) checks a fused kernel as it checks any other.
- **The graph carries facts a hook has to read memory for.** A mask that came from a comparison holds only 0 and 1; a temporary the graph owns overlaps nothing. The binding scans for both today.

## What is not settled

The design lists its open questions, and the prototype picks one answer for each so that it is defined. Among them:

- **Should an `ndarray` be copied when a pending expression first reads it?** Holding it by reference costs nothing and leaves the one rule above to remember; copying removes the rule at the price of a copy per use.
- **What should a deferred write cost?** Today it is computed as a copy with the write made into it. Writing into the old contents when nothing else reads them is left to whatever computes a graph faster.
- **What carries a length that is not known?** The prototype uses an integer larger than any array can be, so that NumPy's own shape arithmetic works on it unchanged. A type of its own is the sound form.
- **Should the C core ship in the package**, built with the binding's C and in the wheels, with the Python as its definition and fallback?

Everything above was run on ARM64 macOS with NumPy 2.5.5 and Python 3.12. Other libraries handed a lazy array, such as pandas or matplotlib, and the other architecture were not tried.
