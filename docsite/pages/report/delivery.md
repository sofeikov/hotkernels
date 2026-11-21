<p class="lede">At the end of the last chapter there were 23 native functions, each proven correct on a flat buffer and none reachable from Python. The remaining problem sounds like plumbing and is closer to half the project: make <code>x[mask]</code> in somebody's existing notebook run a kernel, without that person changing a character, and without breaking anything on the day their array is strided, or a subclass, or of a type no kernel handles.</p>

## What "no friction" rules out

The promise is two lines:

```python
import hotkernels
hotkernels.install()
```

That rules out the easy designs. A wrapper type (`hk.array(x)`) or a decorator would reach anything, but only in code the user has rewritten. Offering faster functions under new names (`hk.argmax`) is the same thing. The kernels have to be reached by the NumPy calls people already write, in their own code and inside pandas or scikit-learn.

The obvious way to do that is to replace names: assign a new function to `np.argmax`. It is also the wrong way, and it is worth seeing why, because the alternative shapes everything else.

| | Replace the name `np.argmax` | Change one pointer inside the existing function object |
|---|---|---|
| `np.argmax(x)` | Reached | Reached |
| `from numpy import argmax`, done before `install()` | Not reached | Reached |
| NumPy calling its own function, as `np.nanargmax` does | Not reached | Reached |
| `np.argmax` is still the same object, pickles, keeps its signature | No | Yes |
| A library that uses `np.argmax` as a dictionary key | Breaks if imported afterwards | Works |

Replacing a name only affects code that looks the name up afterwards. Changing the object affects every reference to it that exists. So the binding never replaces anything. It keeps NumPy's objects and changes a function pointer inside them, and `uninstall()` puts the pointer back.

## Five places to stand

No single hook reaches everything. `np.clip` is a function backed by a "ufunc", NumPy's machinery for elementwise operations. `x.argmax()` is a method. And `x[mask]` is neither: it is syntax, which Python turns into a call through a pointer stored in the array's type. An early version of the design recorded indexing as a known gap that might never be reachable.

It turned out that everything can be reached, with five kinds of hook at five depths.

```mermaid Where the five hooks sit. A deeper hook is reached by more ways of writing the same operation and needs fewer checks, because NumPy has already done its own. A shallower hook can skip NumPy's fixed costs.
flowchart TD
  code["Your code"]
  code --> f["np.argmax(x), np.clip(x, lo, hi),<br/>np.histogram(...), np.where(mask)"]
  code --> m["x.argmax(), x.clip(lo, hi),<br/>x.cumsum(), mask.nonzero()"]
  code --> i["x[mask], x[mask] = v"]

  f --> hf["<b>Function hook</b><br/>the call pointer inside<br/>the NumPy function object"]
  m --> hm["<b>Method hook</b><br/>the C function pointer in<br/>ndarray's method table"]
  i --> hi["<b>Indexing hook</b><br/>the two pointers behind a[i] and a[i] = v<br/>in ndarray's type"]
  hf -.->|"where no kernel applies"| inner["NumPy's own code"]
  hm -.-> inner
  inner --> wrap["the clip ufunc's<br/>machinery"]
  wrap --> hl["<b>Inner-loop hook</b><br/>the ufunc's loop for one element type"]
  inner --> hs["<b>Type-slot hook</b><br/>the argmax / argmin routine<br/>an element type carries"]

  class hf,hm,hi,hl,hs hot
```

| Hook | What is changed | Does NumPy offer this for the purpose? |
|---|---|---|
| **Inner loop** | A ufunc's loop for one element type | Yes: a public C API exists for exactly this |
| **Type slot** | The `argmax` and `argmin` entries each element type holds | The accessor is public; writing through it is not a documented use |
| **Function** | The call pointer inside a NumPy function object | No |
| **Method** | The C function pointer in `ndarray`'s method table, and in the ufunc type's for `accumulate` and `reduceat` | No |
| **Indexing** | The two pointers behind `a[i]` and `a[i] = v` in `ndarray`'s type | No |

The honest reading of the last column: one hook is sanctioned, one is a grey area, and three reach into NumPy's objects in ways NumPy never invited. With the two sanctioned-or-nearly hooks alone, 3 of the 15 operations could be served. The project decided to use all five, and built the defences described below to make that safe.

The rule for choosing is: **use the deepest hook that exists for an operation**, because it is reached by the most spellings and arrives after NumPy has already validated everything; **add the function or method hook on top where the fixed cost of the call is worth removing**.

One library is reached without NumPy at all. pandas and empyrical send NaN-aware reductions to bottleneck, whose `nanmean`, `nanstd`, `nanmin` and `nanmax` are C function objects of CPython's own type. The function hook works on them unchanged: their call pointer is set, so every holder of the object reaches the binding, and a call no kernel takes goes on to bottleneck as CPython would have sent it. Four kernels, checked against bottleneck rather than NumPy, serve them along one axis of a two-dimensional array ([the design](https://github.com/sofeikov/hotkernels/blob/main/docs/bottleneck-design.md)). hotkernels never imports bottleneck; it hooks it when it is imported.

The syntax and its function forms meet different hooks. `x[mask]` and `x[mask] = v` reach the indexing hook; `np.compress(mask, x)`, `np.extract(mask, x)`, `np.putmask(x, mask, v)` and `np.copyto(x, v, where=mask)` are function objects, and each has a function hook in front of the same kernel. `np.where(mask)` needs one too: it calls NumPy's C code for `nonzero` directly and never reaches the method `np.nonzero` calls. The method and `np.where(mask)` also take a C-ordered boolean matrix, for the `nonzero_2d_bool` kernel, and return what NumPy returns for one: two views of the columns of a single array of (row, column) pairs, with the same strides and base. The function forms flatten their arrays, so they take a contiguous `x` of any shape where the syntax takes one dimension.

Try it. The figure below encodes the binding's actual rules.

<figure class="widget">
<div data-widget="router"></div>
<figcaption>Pick a NumPy statement and describe the array. The figure shows which hook the call reaches, the checks it has to pass, and whether a kernel or NumPy's own code runs. Try a strided array, another element type, or 16 elements instead of a million.</figcaption>
</figure>

## The rule that keeps it safe

A hook has one job: decide, very quickly, whether this exact call is one a kernel handles. The binding follows a rule that makes the decision easy to reason about:

> The binding never converts an argument and never raises an error of its own. A call either meets a kernel's conditions exactly as it stands, or goes to NumPy with the caller's own arguments.

So there is no "close enough". An array of float32 where the kernel wants float64 is not converted; it goes to NumPy. An unusual keyword is not interpreted; it goes to NumPy. Every error message and warning a user ever sees is NumPy's own, because every call that could produce one was passed along untouched. The conditions, shared by the function, method and indexing hooks:

- The array's type is exactly `ndarray`, not a subclass.
- Its element type is exactly the kernel's, in native byte order.
- It is contiguous. Alignment is not required.
- It has at least the kernel's minimum length.
- Every argument the kernel does not implement is absent or at its default.
- Scalars are inside the domain the kernel was verified on: for `clip`, `lo <= hi`.
- An output the caller supplies is writable and does not overlap an input.

The two deep hooks need almost none of this. By the time NumPy calls an inner loop or a type slot, it has settled the element type and described the memory layout in the arguments. A strided call simply shows up with a step that is not the element size, and is handed back.

## What a detour costs

The indexing hook sits on *every* index operation on *every* array in the process: `x[3]`, `x[1:5]`, all of them. If it cost 20 ns, hotkernels would make most NumPy programs slower. So this is the number the whole delivery design is judged by:

| Call that reaches a hook and is left to NumPy | Plain NumPy | With hotkernels | Difference |
|---|---|---|---|
| `x[3]` | 60.3 ns | 60.8 ns | +0.5 ns |
| `x[1:5]` | 88.8 ns | 89.4 ns | +0.6 ns |
| `x.astype(np.float64)`, uint8, 16 elements | 198 ns | 201 ns | +3 ns |
| `np.clip(x[::2], -0.5, 0.5)`, 16 elements | 1.24 µs | 1.26 µs | within noise |
| `np.sum(x)`, int64, 16 elements | 1.08 µs | 1.08 µs | within noise |

**About a nanosecond.** Getting there took the same kind of measured iteration as the kernels, and three findings from it are instructive:

- **Keep the rare work out of line.** The compiler had folded the code for *taking* a kernel into the hook, so every index operation saved twelve registers and reserved a scratch buffer before its first test: 2 to 4 ns on every `x[3]`. Forcing that code into a separate function leaves the indexing hook at nine instructions: is the index a boolean array? No: jump to NumPy.
- **An atomic counter was too expensive.** `report()` counts calls. A properly atomic increment cost more than the rest of a fallback put together. The counters are updated with a plain load and store; two threads counting at the same instant can lose a count, which is acceptable for statistics.
- **Test the cheap, decisive thing first.** `np.cumsum` on a float array once spent about 60 ns comparing keyword names before discovering it had no kernel for that element type. Checking the array first fixed it.

## The other fixed cost: NumPy's own

A deep hook delivers the kernel but does nothing about what happens above it. Through the type slot alone, `np.argmax` on 16 elements still costs about 560 ns, nearly all of it NumPy's Python-level wrapper. That is why `argmax` has three hooks: the type slot for everything, and the function and method hooks, which can answer a simple call before the wrapper runs.

| `np.argmax(x)`, 16 float32 elements | Time |
|---|---|
| Plain NumPy | 527 ns |
| With hotkernels | 37.0 ns |

A 14x speedup on 16 elements has almost nothing to do with assembly: the loop is 2 ns either way. It is the wrapper being skipped. The same effect makes `np.sum` on 16 bytes 32x faster.

`np.clip` shows both halves. Its kernel was first reached only through the clip ufunc's inner loop, beneath `np.clip`'s Python wrapper, `x.clip`, NumPy's `_methods._clip` and the ufunc machinery: on 16 elements it gained nothing (1.00x), at a million elements 5x to 11x. A function hook on `np.clip` and a method hook on `x.clip` now call the kernel before any of that runs, and 16 elements take 120 ns instead of 1.2 µs. They take only the calls whose result they can reproduce without the machinery: bounds that are plain numbers, or NumPy scalars of the array's own type, so that the result keeps the element type NumPy would give it, and no `out`. Anything else goes on to the ufunc, where the loop hook still serves it. The gradient-clipping idiom, `np.clip(g, -5, 5, out=g)`, was at first left to NumPy, because the kernels are wrong when their output is their input; the function, method and loop hooks now clip it in place through a buffer of the binding's, so the kernel never sees overlapping buffers. On int64, which has no kernel, the same two hooks run a plain C loop of the binding's own for arrays of up to 4,096 elements: on 16 elements 117 ns instead of 1.8 µs, measured on x86-64. Above that size NumPy's own loop is as good, and the call is NumPy's. `np.round` on float64 is served the same way, by function hooks on `np.round` and `np.around` and a method hook on `x.round`: a loop that multiplies, rounds and divides as NumPy's three passes do, on arrays of up to 4,096 elements. `np.median`, a Python function in NumPy that partitions a copy and takes a mean, is answered on float64 arrays of up to 128 elements by sorting a copy on the stack. One more family of loops is there for another reason: an accumulation down the columns of a float64 matrix (`a.cumprod(axis=0)`, `np.fmax.accumulate(a, axis=0)`) reaches the ufunc's `accumulate` method, where NumPy walks one column at a time with a row between elements. A loop of the binding computes the same operation on the same operands a row at a time, about 7x to 10x faster at 253 × 4,789 on an Apple M1 Pro, and hands a sum or product to NumPy at the row where it first meets a NaN or a floating-point flag, so that every warning stays NumPy's. One loop of the binding's own is there for a different reason: on Apple Silicon, `w -= b` and `w += b` on float64 arrays of 131,072 elements (1 MB) or more go through the inner loops of `subtract` and `add` to a loop that prefetches the output for a store. After Accelerate has multiplied by a large matrix on several cores, updating that matrix in place costs NumPy's loop up to 15 times its usual time, and this loop removes the cost; everywhere else it runs at NumPy's speed. [The gradient-step findings](https://github.com/sofeikov/hotkernels/blob/main/docs/gradient-step-findings.md) have the measurements.

There is a mirror-image case. The type slot must take the interpreter lock (see below), which costs about 40 ns per call, more than the kernel saves on a short array. So that hook leaves runs under 128 elements to NumPy. A method already holds the lock, so `x.argmax()` on a short contiguous array is served by the method hook instead: 55 ns against NumPy's 172 ns at 16 elements. Where a hook cannot win, it stands aside: the exit criterion was that no entry point is slower than NumPy at any size measured.

## Scratch output and the cost of allocation

Chapter 5 left a debt. The mask kernels store more than they keep, so they need an output buffer with room for all $n$ elements, of which the tail ends up holding rubbish. NumPy, by contrast, counts the selected elements first and allocates exactly that many.

The binding pays the debt like this:

```mermaid What happens on x[mask] once hotkernels is installed. The kernel never sees memory the caller owns, and the caller never sees the scratch tail.
sequenceDiagram
  participant P as Python
  participant H as Indexing hook
  participant K as compress_f64
  participant N as NumPy
  P->>H: x[mask]
  H->>H: is the index a boolean ndarray?
  alt not a mask, or conditions not met
    H->>N: original function, same arguments
    N-->>P: result
  else plain contiguous float64 and a mask of the same shape
    H->>H: scratch buffer of n elements (on the stack up to 2 KB)
    H->>H: scan the mask once: every byte 0 or 1?
    H->>K: x, mask, scratch, n
    K-->>H: count
    H->>N: new array of exactly count elements
    H->>H: copy count elements, free the scratch
    H-->>P: result
  end
```

Two other strategies were measured and lost: shrinking the big buffer in place was slower than copying out (555 µs against 301 µs at a million elements, half selected). The cost shows in the results: from Python, `x[mask]` on float64 is 15.8x faster than NumPy where the kernel alone is 22.5x. The buffer, the copy and the mask scan take about a third of the gain.

The mask scan is there for a reason nobody expected. A NumPy boolean array is supposed to hold bytes of 0 or 1, but nothing stops one holding a 2 (through a view of other data, for instance). The kernels multiply the mask bytes as numbers; a 2 would make them count past the buffer's capacity and write outside it. The judge never generates such a mask, so the kernels pass. The binding reads the mask once first, and any byte above 1 sends the call to NumPy.

## The contract gaps

That mask check is one of several places where binding the kernels to real calls exposed an input on which a kernel is wrong or unsafe and its spec did not say so. The binding is the last place such inputs can be stopped.

| Input | What the kernel would do | What the binding does |
|---|---|---|
| A mask byte other than 0 or 1 | Write outside the buffer | Scans the mask; sends the call to NumPy |
| `np.clip(x, 0, 1, out=x)`: output is the input | Wrong on some data: a `-0.0` or a signalling NaN is changed before it is read again | Copies 4 KB of the array at a time to a buffer of its own and has the kernel clip the copy back into place |
| Another thread writes the array during `argmax` | Reads it twice; may read past the end | Holds the interpreter lock while the kernel runs |
| A histogram range whose bin edges are not strictly increasing | Returns counts where NumPy raises an error | Builds the edges first, as NumPy does; sends the call to NumPy if they are not increasing |
| `clip` bounds that are arrays, reversed, or NaN | Matches only NumPy's scalar, ordered case | Takes the kernel for ordered scalar bounds only |
| A NaN among the sample points of `np.interp` | Read before the start of its buffers, or return something other than NumPy | Reads the sample points once; sends the call to NumPy. Only worth doing when there are enough queries to pay for the pass, so a call with more than 64 sample points per query goes to NumPy as well |
| Floating-point status flags left set by a kernel | NumPy would turn them into spurious warnings | Clears them after `clip`; for float `cumsum`, discards the kernel's output if any flag was raised and lets NumPy compute and warn |
| A float32 or float64 sum whose array is not aligned to its elements' size | Add in NumPy's pairwise order, where NumPy copies such an array in blocks of 8,192 elements and adds the blocks' sums one after another | Sends the call to NumPy |
| A float32 or float64 sum that is not finite | Return a NaN whose bits depend on which operand of an addition came first, and no overflow warning | Discards the kernel's result and lets NumPy compute and warn |

The binding's check is what handles each row today. Fixing the kernels themselves, so that the check is no longer needed, is still open for the mask, in-place and `argmax` rows. The lesson generalises: a function verified in isolation is verified *on the inputs its tests can express*. Integrating it is a second round of verification, against what real callers can actually pass.

## Threads

NumPy releases Python's interpreter lock around long loops so that other threads can run. The binding follows NumPy's lead with three exceptions, each forced by a kernel property described earlier:

- **Inner loop (`clip`)**: the kernel runs wherever NumPy's loop ran, lock held or released as NumPy decided.
- **Function, method and indexing hooks**: release the lock above 8,192 elements, as a kernel is a pure function touching no Python object.
- **Type slot (`argmax`)**: *takes* the lock, because the kernels read the array twice and are not safe against a concurrent writer.
- **Masked assignment**, written as `x[mask] = v`, `np.putmask` or `np.copyto(..., where=mask)`: keeps the lock, because the kernel [rewrites unselected elements]([[ root ]]report/selection/#masked-assignment-write-what-you-did-not-change) with their own bits. That blocks other Python threads for about 180 µs per million elements. The float32 and int32 kernel writes only the selected elements, and keeps the lock all the same, so that the four element types behave alike.

Free-threaded Python, which has no such lock, is not supported: `install()` does nothing there and says why.

## Installing is a test

Three of the five hooks depend on the internal layout of NumPy's objects. A future NumPy release could route a call differently, and a hook would silently stop firing, or worse, fire for something else. So installation does not trust itself.

```mermaid How install() applies one hook. A hook that cannot demonstrate that it works is removed again, and NumPy is left as it was for that operation.
sequenceDiagram
  participant U as install()
  participant B as Binding
  participant N as NumPy
  U->>B: apply hook, remember the pointer it replaced
  U->>N: one small public call with a known answer
  N->>B: (the call reaches the hook, if the hook works)
  B-->>U: result
  alt the kernel's counter moved and the answer is right
    U->>U: keep the hook
  else counter did not move, or wrong answer
    U->>B: put the original pointer back
    U->>U: record why, for report()
  end
```

Each hook goes in one at a time and proves itself with a real call through NumPy's public interface. If the kernel's counter did not move, or the answer is wrong, the hook is removed and `report()` says so. The failure mode of a NumPy that routes a call differently is therefore "that operation is not accelerated", never "that operation is wrong".

A NumPy that routes the call the same way but answers it differently is another matter, and the self-test does not see it. NumPy 2.0 makes every zero `np.clip` returns +0.0, where later releases and the kernel keep its sign; before 2.3, `np.count_nonzero(x)` returns a Python `int` where later releases and the binding return `np.intp`. On those releases every hook passed its self-test and stayed in place. The differential test below found both, run against each NumPy 2 release, and the package now requires NumPy 2.3 and refuses an older one. Continuous integration runs that test against the oldest NumPy the package claims and the newest on PyPI.

`uninstall()` restores the pointers in reverse order, and only a pointer that is still the one hotkernels put there. If something else has patched the same place since, it is left alone and reported.

## Testing the binding

The judge proves a kernel right as a native function. It says nothing about the code between NumPy and the kernel, and that code decides, on every call, whether a kernel's conditions hold. A wrong check there is a wrong answer in someone's program. It has its own test.

**A differential test of calls, not of kernels.** Each case is a Python statement. It is run on plain NumPy and again with hotkernels installed, and the two must agree in result type, element type, shape and bytes; in the exception raised; in the warnings issued; and in leaving the inputs unchanged. The cases are a cross product per entry point:

| Dimension | Values |
|---|---|
| Element types | The kernel's, and its neighbours |
| Shapes | 0-d, empty, one element, 1-D, 2-D |
| Layouts | Contiguous, strided, reversed, misaligned, read-only, Fortran order, other byte order |
| Array kinds | `ndarray`, a subclass, a masked array, a memory map, a list |
| Spellings | Positional, keyword, `out=`, `axis=`, method and function forms |
| Values | The judge's value classes: NaN, infinities, signed zeros, ties |

That is **302,364 calls over the 23 hooked NumPy operations and bottleneck's four functions** (on each architecture; 96,734 over 15 when the first kernels were bound), and all agree.

**Each case also states whether a kernel should have run**, and the counters are checked against it. Without that, a check that quietly stopped matching would pass every comparison, since NumPy's fallback is always right, while delivering nothing.

**NumPy's own test suite**, run once on plain NumPy and once with the kernels installed, gives the same outcome for each of its **48,659 tests** (NumPy 2.5.5 on x86-64, with every exact kernel and loop installed). This is the broadest evidence available for "never break anything", and it cost nothing to write. CI runs it on both platforms for every pull request and every release wheel.

The test earned its keep while it was being written. It found two defects in the binding: `np.count_nonzero` returned a Python `int` where NumPy returns `np.intp`, and `np.histogram` issued NumPy's overflow warnings twice for a range wider than a double can hold. Neither was a wrong number. Both were differences a user's code could have observed.

::: note What this test does not cover
The binding runs without guard pages. A wrong length passed to a correct kernel is caught only if it changes a result. The kernels' memory safety is established by the judge; the binding's is established by review and by the differential test, which is weaker.
:::

## Where that leaves things

Of the 49 kernels, 47 are installed by `install()` and the other 2 with `install(inexact=True)`; 4 of the 47 serve bottleneck's functions, and are applied once bottleneck is imported. `searchsorted` and `interp` were bound late: they were written with one length for the sorted array and the needles, which no real call has, and had to be given a second before a hook could hand them anything. The last kernel added is the right side of the search, which `np.digitize` asks for by default. The stable `argsort` kernels came from a study of whole applications, where sorting was the largest cost no kernel touched; they are reached through a hook on the `argsort` method, because NumPy 2.5 no longer sorts through the per-type table that `argmax` is hooked into. The search on int64 came from a production pipeline, whose calls with one `np.int64` needle all reached the hook and went to NumPy ([issue 160](https://github.com/sofeikov/hotkernels/issues/160)). An ordinary NumPy program that uses eleven of these operations on two million elements runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.

The last chapter collects every number, and is candid about which of them mean less than they appear to.
