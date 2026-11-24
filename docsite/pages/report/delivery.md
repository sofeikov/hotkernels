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

## Where the hooks sit

No single hook reaches everything. `np.clip` is a function backed by a "ufunc", NumPy's machinery for elementwise operations. `x.argmax()` is a method. `x[mask]` is neither: it is syntax, which Python turns into a call through a pointer stored in the array's type. And `a + b` is an operator, which `ndarray` turns into a call of the ufunc object `np.add`. An early version of the design recorded indexing as a known gap that might never be reachable.

It turned out that everything can be reached, with five kinds of hook at five depths. A sixth, on the ufunc objects themselves, came later, for the cost of NumPy's call path on small arrays: see [The other fixed cost](#the-other-fixed-cost-numpys-own). A seventh is the function hook again, set in bottleneck's function objects instead of NumPy's.

```mermaid Where the hooks in NumPy sit. A deeper hook is reached by more ways of writing the same operation and needs fewer checks, because NumPy has already done its own. A shallower hook can skip NumPy's fixed costs.
flowchart TD
  code["Your code"]
  code --> f["np.argmax(x), np.clip(x, lo, hi),<br/>np.histogram(...), np.where(mask)"]
  code --> m["x.argmax(), x.clip(lo, hi),<br/>x.cumsum(), mask.nonzero()"]
  code --> i["x[mask], x[mask] = v"]
  code --> o["a + b, a > 0, np.exp(a),<br/>x += y"]

  f --> hf["<b>Function hook</b><br/>the call pointer inside<br/>the NumPy function object"]
  m --> hm["<b>Method hook</b><br/>the C function pointer in<br/>the type's method table"]
  i --> hi["<b>Indexing hook</b><br/>the two pointers behind a[i] and a[i] = v<br/>in ndarray's type"]
  o --> hu["<b>Ufunc call hook</b><br/>the call pointer inside<br/>each ufunc object"]
  hf -.->|"where no kernel applies"| inner["NumPy's own code"]
  hm -.-> inner
  hu -.->|"where the lane does not apply"| wrap
  inner --> wrap["a ufunc's<br/>machinery"]
  wrap --> hl["<b>Inner-loop hook</b><br/>the ufunc's loop for one element type"]
  inner --> hs["<b>Type-slot hook</b><br/>the argmax / argmin routine<br/>an element type carries"]

  class hf,hm,hi,hl,hs,hu hot
```

| Hook | What is changed | Does the owner offer this for the purpose? |
|---|---|---|
| **Inner loop** | A ufunc's loop for one element type | Yes: a public C API exists for exactly this |
| **Type slot** | The `argmax` and `argmin` entries each element type holds | The accessor is public; writing through it is not a documented use |
| **Function** | The call pointer inside a NumPy function object | No |
| **Method** | The C function pointer in `ndarray`'s method table, and in the ufunc type's for `accumulate` and `reduceat` | No |
| **Indexing** | The two pointers behind `a[i]` and `a[i] = v` in `ndarray`'s type | No |
| **Ufunc call** | The call pointer inside each of NumPy's ufunc objects | No. The loops it then calls are fetched through an interface NumPy does offer, to compilers such as numba |
| **bottleneck** | The call pointer inside four of bottleneck's function objects | No |

The honest reading of the last column: one hook is sanctioned, one is a grey area, and the others reach into objects in ways their owners never invited. With the two sanctioned-or-nearly hooks alone, 3 of the first design's 15 operations could be served. The project decided to use them all, and built the defences described below to make that safe.

The rule for choosing is: **use the deepest hook that exists for an operation**, because it is reached by the most spellings and arrives after NumPy has already validated everything; **add the function or method hook on top where the fixed cost of the call is worth removing**.

One library is reached without NumPy at all. pandas and empyrical send NaN-aware reductions to bottleneck, whose `nanmean`, `nanstd`, `nanmin` and `nanmax` are C function objects of CPython's own type. The function hook works on them unchanged: their call pointer is set, so every holder of the object reaches the binding, and a call no kernel takes goes on to bottleneck as CPython would have sent it. Four kernels, checked against bottleneck rather than NumPy, serve them along one axis of a two-dimensional array ([the design](https://github.com/sofeikov/hotkernels/blob/main/docs/bottleneck-design.md)). hotkernels never imports bottleneck; it hooks it when it is imported.

The syntax and its function forms meet different hooks. `x[mask]` and `x[mask] = v` reach the indexing hook; `np.compress(mask, x)`, `np.extract(mask, x)`, `np.putmask(x, mask, v)` and `np.copyto(x, v, where=mask)` are function objects, and each has a function hook in front of the same kernel. `np.where(mask)` needs one too: it calls NumPy's C code for `nonzero` directly and never reaches the method `np.nonzero` calls. The method and `np.where(mask)` also take a C-ordered boolean matrix, for the `nonzero_2d_bool` kernel, and return what NumPy returns for one: two views of the columns of a single array of (row, column) pairs, with the same strides and base. The function forms flatten their arrays, so they take a contiguous `x` of any shape, as the syntax does.

## Each hook, exactly

What each kind of hook changes, in the terms of CPython and NumPy, how `install()` finds it, and which spellings reach it. The code is in [`hotkernels/src/hotkernels/_binding/`](https://github.com/sofeikov/hotkernels/tree/main/hotkernels/src/hotkernels/_binding), one file per kind. Every kind is applied the same way: read the pointer, keep it, write the binding's function in its place, and prove with a call through NumPy that the kernel is reached ([Installing is a test](#installing-is-a-test)). Every kind is removed the same way: put the kept pointer back, but only if the place still holds the binding's function.

**Function hook** (`hook_function.c`). NumPy's functions, `np.argmax`, `np.where`, `np.median` and the rest, are objects of one C type, `numpy._ArrayFunctionDispatcher`. Each wraps NumPy's implementation and runs the `__array_function__` protocol, through which other array libraries take over NumPy's functions, before calling it. CPython calls such an object through a function pointer stored in the object itself, its *vectorcall* pointer, at an offset its type publishes as `tp_vectorcall_offset`. The binding looks the function up in the `numpy` module, checks that its type has a pointer there and that it is set, keeps it and writes the binding's own function in its place. `np.argmax` stays the same object, with its name, signature and docstring, and every reference to it, from before `install()` or after, now calls through the new pointer.

The binding's function receives exactly what the caller wrote: the positional arguments, then the values of the keyword arguments, and a tuple of the keywords' names. It matches them against the function's parameter names itself, without building a dictionary, and takes a kernel only if every argument already is what the kernel needs. Otherwise it calls the pointer it kept with the same arguments, and NumPy runs as if the hook were not there. Because the hook runs before `__array_function__`, an array of another library (dask, CuPy, a pandas object) fails the first test, an exact `ndarray`, and NumPy dispatches it as before. Nineteen of NumPy's functions are hooked this way: `argmax`, `argmin`, `any`, `bincount`, `clip`, `compress`, `copyto`, `count_nonzero`, `extract`, `histogram`, `interp`, `isin`, `median`, `putmask`, `round` and `around`, `sum`, `unique` and `where`.

**Method hook** (`hook_method.c`). A method of a C type, `ndarray.argmax` for instance, is described by an entry in a table inside NumPy, a `PyMethodDef`: the method's name, its C function (`ml_meth`), and flags that say how that function is called (`ml_flags`). The attribute `np.ndarray.argmax` is a *method descriptor* that points to the entry. The binding fetches the descriptor, follows it to the entry and replaces the C function in it. Every way of calling the method reads that field at the time of the call: `x.argmax()`, `np.ndarray.argmax(x)`, and a bound method `f = x.argmax` saved before `install()`.

The flags matter. A method declared `METH_FASTCALL | METH_KEYWORDS` is called with an array of arguments and a tuple of keyword names, one declared `METH_VARARGS | METH_KEYWORDS` with a tuple and a dictionary, and the binding's function must have the shape the flags say. So the hook is applied only when the flags in NumPy's entry are the ones the binding's function was written for; a NumPy that changed them gets the hook left off, and `report()` says why. NumPy's own Python functions often do nothing but call the method: `np.cumsum(x)`, `np.nonzero(x)`, `np.searchsorted(a, v)`, `np.argsort(x)` and `np.round(x)` end in `x.cumsum()`, `x.nonzero()` and the rest, and `np.argwhere` and `np.digitize` call two of those, so all of them reach the method hook. Eleven methods of `ndarray` are hooked: `any`, `argmax`, `argmin`, `argsort`, `astype`, `clip`, `cumsum`, `nonzero`, `round`, `searchsorted` and `sum`. Two more are the ufunc type's, `accumulate` and `reduceat`. Every ufunc shares those two entries, so `np.fmax.accumulate(x)` and `np.add.accumulate(x)` reach the same function, which tells them apart by the object it is called on and lets any other ufunc's call go on after a pointer comparison or two.

**Indexing hook** (`hook_index.c`). CPython runs `a[i]` by calling a pointer in the type of `a`, `tp_as_mapping->mp_subscript`, and `a[i] = v` by calling its neighbour, `mp_ass_subscript`. For `ndarray` both point into NumPy, and the binding replaces the two in `ndarray`'s type object itself. CPython reads them from the type on every index operation and keeps nothing derived from them, so nothing has to be told of the change. `PyType_Modified`, the usual call after a type is changed, is deliberately not made: it would throw away the type's cache of attribute lookups, and CPython stops keeping that cache for a type modified a thousand times. Every index operation on every plain array now passes through the hook, so its first test is the only one most calls see: is the index an exact `ndarray` of booleans? For `x[3]` or `x[1:5]` it fails at once, and the call goes to NumPy ([What a detour costs](#what-a-detour-costs)). `operator.getitem(x, mask)` reaches the hook as the syntax does. `x.__getitem__(mask)`, written out, does not: that attribute is a wrapper CPython made when NumPy created the type, and it holds NumPy's original function and calls it directly.

**Type-slot hook** (`hook_slot.c`). NumPy keeps, for each element type, a table of C functions, `PyArray_ArrFuncs`, which the binding reaches through the type's descriptor (`PyDataType_GetArrFuncs`). Its `argmax` and `argmin` entries are where every route to `argmax` ends: `np.argmax(x)`, `x.argmax()`, `axis=` (one call per row or column) and `np.nanargmax`. NumPy calls the entry with one contiguous run of elements in native byte order, having copied the data first if the array was laid out any other way, so the hook checks nothing but the length. The binding replaces the float32 and float64 entries.

**Inner-loop hook** (`hook_loop.c`). A ufunc holds, for each combination of element types it supports, a C loop that NumPy calls with a pointer per operand, an element count, and a step in bytes per operand. `PyUFunc_ReplaceLoopBySignature`, in NumPy's C API for exactly this, swaps the loop for one combination and returns the old one. The binding replaces the float32 and float64 loops of `clip`, and on Apple Silicon the float64 loops of `subtract` and `add`. When NumPy calls a loop it has already converted the arguments, settled the element type and allocated the output, so the hook sees memory only: a strided operand shows up as a step that is not the element size, and goes to NumPy's loop.

**Ufunc call hook** (`hook_ufunc.c`). `np.add`, `np.exp` and every other ufunc are objects of type `np.ufunc`, and like NumPy's functions they are called through a pointer inside the object (the `vectorcall` field of NumPy's `PyUFuncObject`). `ndarray`'s operators call the same objects: `a + b` calls `np.add`, `a > 0` `np.greater`, `x += y` `np.add` with `out=x`. The binding walks the `numpy` module's dictionary, not `dir()`, which would import NumPy's lazily loaded submodules, and points every ufunc with one or two inputs, one output and no core dimensions at its own function, provided the ufunc's pointer is still NumPy's. Ufuncs of other libraries, such as `scipy.special.expit`, and ufuncs made after `install()` are not touched. First it checks that NumPy's interface for fetching loops answers in the form the binding reads (a capsule named `numpy_1.24_ufunc_call_info`), and leaves the hook off if not.

The binding's function, the *lane*, takes a call whose operands are exact, contiguous, aligned arrays of one shape, a matrix and a row of it, or an array and a number, with at most 16,384 elements in the result. It asks NumPy once for each ufunc and kinds of operand which inner loop NumPy would run, through `ufunc._resolve_dtypes_and_context` and `ufunc._get_strided_loop`, the route numba takes, and keeps the answer in a table of 2,048 entries; a combination that finds no room takes the place of an older one, which is asked for again when next needed. From then on the lane allocates the output, or takes the one it was given, and calls that loop with the counts and strides NumPy would pass. A row against a matrix is repeated into a buffer, as NumPy's buffered iterator does, and only when NumPy too would make a single call of its loop. A floating-point flag raised by the loop goes to NumPy's own handler, `PyUFunc_GiveFloatingpointErrors`, so warnings and `np.errstate` behave as before. If a loop is replaced with `PyUFunc_ReplaceLoopBySignature` after the lane fetched it, by hotkernels' inner-loop hook or by anyone else, the lane sees on the next call that the ufunc's table holds another function, and asks NumPy again. Every other call goes to NumPy's pointer with its arguments untouched.

**bottleneck hook** (`hook_bottleneck.c`). bottleneck's `nanmean`, `nanstd`, `nanmin` and `nanmax` are C functions of its module `bottleneck.reduce`, objects of CPython's type `builtin_function_or_method`. For their calling convention (`METH_VARARGS | METH_KEYWORDS`) CPython leaves the object's call pointer empty and calls the type's `tp_call` with a tuple and a dictionary. The binding sets the pointer, so every call of the object, however a library found it, comes to the binding, and a call no kernel takes goes on to `tp_call` with the caller's arguments, which does not come back through the pointer. hotkernels never imports bottleneck. If `bottleneck.reduce` is already imported when `install()` runs, the hooks are applied then. If not, `install()` puts a finder at the front of `sys.meta_path` that finds nothing itself: when `bottleneck.reduce` is imported, it wraps the loader the other finders return, and the moment the module has run, it applies the four hooks, each with its self-test. Until then `report()` shows their rows as waiting.

## One call, several hooks

A call passed on by one hook can meet another below it, and the deeper hook makes its own checks. `np.argmax(x[::2])` is strided, so the function hook passes it on; NumPy's `np.argmax` copies the run and calls the type slot, which takes it from 128 elements. `np.clip(x, lo, hi, out=y)` with `y` another array is passed on by the function hook, and by the method hook on `x.clip` that `np.clip` calls; NumPy then runs the `clip` ufunc, whose float32 or float64 loop takes it if `y` does not overlap `x`. `w -= b` on float64 meets the ufunc call hook, which takes it up to 16,384 elements; above that NumPy's `subtract` runs and calls its float64 loop, which on Apple Silicon is the inner-loop hook.

| You write | Hooks it meets, in order | What serves it when the conditions hold |
|---|---|---|
| `np.argmax(x)`, `np.argmin(x)` | Function, then type slot | The `argmax` and `argmin` kernels |
| `x.argmax()`, `x.argmin()` | Method, then type slot | The same kernels |
| `np.argmax(m, axis=1)`, `m.argmax(axis=1)`, and `argmin` | Function or method, then type slot once per row | The kernels once per row, or a loop of the binding's for rows under 6 float64 or 9 float32 elements |
| `np.argmax(x, axis=0)`, `x.argmax(axis=0)` | Function or method, which pass it on, then type slot once per column | The same kernels, on columns of 128 elements or more |
| `np.clip(x, lo, hi)`, `x.clip(lo, hi)` | Function or method, then the `clip` ufunc's inner loop | The `clip` kernels; on int64, a loop of the binding's |
| `np.round(x, d)`, `np.around(x, d)`, `x.round(d)` | Function, then method; or method | A loop of the binding's, `round_f64` |
| `np.median(x)` | Function | A loop of the binding's, `median_f64` |
| `x[mask]`, `x[mask] = v` | Indexing | The `compress` and `maskfill` kernels |
| `np.compress`, `np.extract`, `np.putmask`, `np.copyto(x, v, where=mask)` | Function | The same kernels |
| `np.nonzero(m)`, `m.nonzero()`, `np.argwhere(m)` | Method | `nonzero_bool`, `nonzero_2d_bool` |
| `np.where(mask)`, `np.where(mask, a, b)` | Function | The `nonzero` kernels; `where_f64` |
| `m.any(axis=1)`, `m.any(axis=0)`; `np.any(m, axis=...)` | Method; function | `any_rows_bool`, `any_cols_bool` |
| `np.sum(x)`; `x.sum()` | Function; method | The `sum` kernels |
| `np.count_nonzero`, `np.bincount`, `np.histogram`, `np.isin`, `np.interp`, `np.unique` | Function | Each its own kernels |
| `np.cumsum(x)`, `x.cumsum()` | Method | The `cumsum` kernels |
| `x.astype(np.float32)` | Method | `cast_u8_f32` |
| `np.searchsorted(a, v)`, `a.searchsorted(v)`, `np.digitize(x, bins)` | Method | The `searchsorted` kernels, one per side |
| `np.argsort(x, kind="stable")`, `x.argsort(kind="stable")` | Method | The `argsort` kernels |
| `np.maximum.accumulate(x)`, `np.fmax.accumulate(x)`, `np.fmax.reduceat(x, starts)` | Method of the ufunc type | `cummax_i64`, `cumfmax_f64`, `fmax_reduce_f64` |
| `a.cumsum(axis=0)`, `a.cumprod(axis=0)`, `np.add.accumulate(a)` and the rest down the columns of a matrix | Method: `cumsum`'s, or the ufunc type's `accumulate` | Loops of the binding's, `cumsum_rows_f64` and three more |
| `a + b`, `a * 2.0`, `a > 0`, `-a`, `np.exp(a)`, `x += y`, every NumPy ufunc with one or two inputs | Ufunc call | NumPy's own inner loop, called directly |
| `w -= b`, `w += b` on float64 | Ufunc call, then the inner loop of `subtract` or `add` (Apple Silicon) | NumPy's loop up to 16,384 elements; from 131,072, a loop of the binding's |
| `bn.nanmean`, `bn.nanstd`, `bn.nanmin`, `bn.nanmax` | bottleneck | The four `*_cols_f64` kernels |

Which element types and arguments each one takes is in [the table of Get started]([[ root ]]start/#what-gets-faster), and with every condition the binding tests in [Which hook serves which kernel](https://github.com/sofeikov/hotkernels/blob/main/docs/delivery-design.md#which-hook-serves-which-kernel), in the design of the binding. Try it here. The figure below encodes the binding's rules: the hooks a call meets, the checks each one makes, and what serves it.

<figure class="widget">
<div data-widget="router"></div>
<figcaption>Pick a NumPy statement and describe the array. The figure shows each hook the call reaches, the checks it makes, and whether a kernel, a loop of the binding's, NumPy's own loop called directly, or NumPy's own code runs. Try a strided array, another element type, a subclass, or 16 elements instead of a million.</figcaption>
</figure>

## The rule that keeps it safe

A hook has one job: decide, very quickly, whether this exact call is one a kernel handles. The binding follows a rule that makes the decision easy to reason about:

> The binding never converts an argument and never raises an error of its own. A call either meets a kernel's conditions exactly as it stands, or goes to NumPy with the caller's own arguments.

So there is no "close enough". An array of float32 where the kernel wants float64 is not converted; it goes to NumPy. An unusual keyword is not interpreted; it goes to NumPy. Every error message and warning a user ever sees is NumPy's own, because every call that could produce one was passed along untouched. The conditions, shared by the function, method, indexing, ufunc call and bottleneck hooks:

- The array's type is exactly `ndarray`, not a subclass.
- Its element type is exactly the kernel's, in native byte order.
- It is contiguous. Alignment is not required, except by the ufunc call hook, which calls the loops NumPy itself uses for aligned data.
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

**About a nanosecond.** The ufunc call hook's detour is longer, 3 to 4 ns, because its first tests look at every operand; it is paid on calls of 250 ns and more. Getting there took the same kind of measured iteration as the kernels, and three findings from it are instructive:

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

`np.clip` shows both halves. Its kernel was first reached only through the clip ufunc's inner loop, beneath `np.clip`'s Python wrapper, `x.clip`, NumPy's `_methods._clip` and the ufunc machinery: on 16 elements it gained nothing (1.00x), at a million elements 5x to 11x. A function hook on `np.clip` and a method hook on `x.clip` now call the kernel before any of that runs, and 16 elements take 120 ns instead of 1.2 µs. They take only the calls whose result they can reproduce without the machinery: bounds that are plain numbers, or NumPy scalars of the array's own type, so that the result keeps the element type NumPy would give it, and no `out`. Anything else goes on to the ufunc, where the loop hook still serves it. The gradient-clipping idiom, `np.clip(g, -5, 5, out=g)`, was at first left to NumPy, because the kernels are wrong when their output is their input; the function, method and loop hooks now clip it in place through a buffer of the binding's, so the kernel never sees overlapping buffers. On int64, which has no kernel, the same two hooks run a plain C loop of the binding's own for arrays of up to 4,096 elements: on 16 elements 117 ns instead of 1.8 µs, measured on x86-64. Above that size NumPy's own loop is as good, and the call is NumPy's. `np.round` on float64 is served the same way, by function hooks on `np.round` and `np.around` and a method hook on `x.round`: a loop that multiplies, rounds and divides as NumPy's three passes do, on arrays of up to 4,096 elements. `np.median`, a Python function in NumPy that partitions a copy and takes a mean, is answered on float64 arrays of up to 64 elements by sorting a copy on the stack, and on longer ones by finding the middle elements from the bits of the numbers ([Get started]([[ root ]]start/#things-worth-knowing)). One more family of loops is there for another reason: an accumulation down the columns of a float64 matrix (`a.cumprod(axis=0)`, `np.fmax.accumulate(a, axis=0)`) reaches the ufunc's `accumulate` method, where NumPy walks one column at a time with a row between elements. A loop of the binding computes the same operation on the same operands a row at a time, about 7x to 10x faster at 253 × 4,789 on an Apple M1 Pro, and hands a sum or product to NumPy at the row where it first meets a NaN or a floating-point flag, so that every warning stays NumPy's. One loop of the binding's own is there for a different reason: on Apple Silicon, `w -= b` and `w += b` on float64 arrays of 131,072 elements (1 MB) or more go through the inner loops of `subtract` and `add` to a loop that prefetches the output for a store. After Accelerate has multiplied by a large matrix on several cores, updating that matrix in place costs NumPy's loop up to 15 times its usual time, and this loop removes the cost; everywhere else it runs at NumPy's speed. [The gradient-step findings](https://github.com/sofeikov/hotkernels/blob/main/docs/gradient-step-findings.md) have the measurements.

The same is true of everything that is not a hooked function: `a + b` on 16 float64 elements costs about 250 ns, `a * 2.0` 400 ns and `a > 0` 500 ns, while the loops take a few. The ufunc call hook, on the call pointer inside each of NumPy's ufunc objects, which every operator of an array calls through, takes that out without a kernel of its own: it calls the inner loop NumPy would have run, directly ([Each hook, exactly](#each-hook-exactly)). The result is NumPy's by construction, and floating-point flags go to NumPy's own handler, so warnings and `np.errstate` are unchanged. It makes small elementwise calls 1.6x to 3.8x faster ([ufunc-lane-design.md](https://github.com/sofeikov/hotkernels/blob/main/docs/ufunc-lane-design.md)). What it cannot remove is NumPy's allocation of the result: 83 ns of the lane's 125 ns on 16 elements, spent on the memory handler's context variable, a capsule lookup and `tracemalloc`, none of which a binding can skip without bypassing a handler the user may have set.

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

NumPy releases Python's interpreter lock around long loops so that other threads can run. The binding follows NumPy's lead, with two exceptions forced by kernel properties described earlier:

- **Inner loop** (`clip`, and `add` and `subtract` on Apple Silicon): the loop runs wherever NumPy's loop ran, lock held or released as NumPy decided.
- **Function, method, indexing and bottleneck hooks**: release the lock above 8,192 elements, as a kernel is a pure function touching no Python object.
- **Ufunc call hook**: releases the lock around NumPy's loop above 500 elements, the threshold NumPy itself uses. A loop that needs the Python API is not taken at all.
- **Type slot (`argmax`)**, the first exception: *takes* the lock, because the kernels read the array twice and are not safe against a concurrent writer.
- **Masked assignment**, the second, written as `x[mask] = v`, `np.putmask` or `np.copyto(..., where=mask)`: keeps the lock, because the kernel [rewrites unselected elements]([[ root ]]report/selection/#masked-assignment-write-what-you-did-not-change) with their own bits. That blocks other Python threads for about 180 µs per million elements. The float32 and int32 kernel writes only the selected elements, and keeps the lock all the same, so that the four element types behave alike.

Free-threaded Python, which has no such lock, is not supported: `install()` does nothing there and says why.

## Installing is a test

Five of the seven kinds of hook depend on the layout of objects that NumPy, CPython or bottleneck never promised to keep: the function, method, indexing, ufunc call and bottleneck hooks. A future release could route a call differently, and a hook would silently stop firing, or worse, fire for something else. So installation does not trust itself.

```mermaid How install() applies one hook. A hook that cannot demonstrate that it works is removed again, and NumPy is left as it was for that operation.
sequenceDiagram
  participant U as install()
  participant B as Binding
  participant N as NumPy
  U->>N: small public calls, before any hook is applied
  N-->>U: NumPy's own answers
  U->>B: apply hook, remember the pointer it replaced
  U->>N: the same calls again
  N->>B: (the call reaches the hook, if the hook works)
  B-->>U: result
  alt the kernel's counter moved and every answer is NumPy's, in every detail
    U->>U: keep the hook
  else counter did not move, or an answer differs
    U->>B: put the original pointer back
    U->>U: record why, for report()
  end
```

Each hook goes in one at a time and proves itself with real calls through NumPy's public interface, made once before any hook is applied and once after. If the kernel's counter did not move, or an answer differs from NumPy's own, the hook is removed and `report()` says so. The answers are compared as the differential test below compares them: Python type, element type, shape, flags and bytes, so a result that is a Python `int` where NumPy's is `np.intp`, or `+0.0` where NumPy's is `-0.0`, counts as different. The probes' inputs hold a negative zero wherever the operation can return one. The failure mode of a NumPy that routes a call differently, or answers it differently in such a detail, is therefore "that operation is not accelerated", never "that operation is wrong". It costs about 0.6 ms of the 6.4 ms `install()` takes on an Apple M1 Pro.

The self-test did not always compare in full. NumPy 2.0 makes every zero `np.clip` returns +0.0, where later releases and the kernel keep its sign; before 2.3, `np.count_nonzero(x)` returns a Python `int` where later releases and the binding return `np.intp`. On those releases every hook passed the self-test of the time, which compared values alone, and stayed in place. The differential test below found both, run against each NumPy 2 release, and the package now requires NumPy 2.3 and refuses an older one. Continuous integration runs that test against the oldest NumPy the package claims and the newest on PyPI. A probe sees only its own inputs, so the differential test remains the wider guard.

`uninstall()` restores the pointers in reverse order, and only a pointer that is still the one hotkernels put there. If something else has patched the same place since, it is left alone and reported.

A hook that fails to apply or to prove itself is logged as a warning on the `hotkernels` logger and listed under the hooks that are not in place in `report()`. Calling `install()` again with the same arguments does nothing. With other arguments, `install(inexact=True)` after `install()` for instance, it removes every hook and applies them all again, self-tests included, and the counts in `report()` start again. `with hotkernels.enabled(...)` does the same on entering when its arguments differ from the installation it finds, and on leaving puts hotkernels back as it found it: not installed, or installed with the arguments it had.

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

That is **403,023 calls over the 24 hooked NumPy operations and bottleneck's four functions** (on each architecture; 96,734 over 15 when the first kernels were bound), and all agree. A quarter of them, 100,657, are elementwise ufunc calls for the ufunc call hook.

**Each case also states whether a kernel should have run**, and the counters are checked against it. Without that, a check that quietly stopped matching would pass every comparison, since NumPy's fallback is always right, while delivering nothing.

**NumPy's own test suite**, run once on plain NumPy and once with the kernels installed, gives the same outcome for each of its **48,659 tests** (NumPy 2.5.5 on x86-64, with every exact kernel and loop installed). This is the broadest evidence available for "never break anything", and it cost nothing to write. CI runs it on both platforms for every pull request that changes more than documentation, and for every release wheel.

The test earned its keep while it was being written. It found two defects in the binding: `np.count_nonzero` returned a Python `int` where NumPy returns `np.intp`, and `np.histogram` issued NumPy's overflow warnings twice for a range wider than a double can hold. Neither was a wrong number. Both were differences a user's code could have observed.

::: note What this test does not cover
The binding runs without guard pages. A wrong length passed to a correct kernel is caught only if it changes a result. The kernels' memory safety is established by the judge; the binding's is established by review and by the differential test, which is weaker.
:::

## Where that leaves things

Of the 49 kernels, 47 are installed by `install()` and the other 2 with `install(inexact=True)`; 4 of the 47 serve bottleneck's functions, and are applied once bottleneck is imported. `searchsorted` and `interp` were bound late: they were written with one length for the sorted array and the needles, which no real call has, and had to be given a second before a hook could hand them anything. The last kernel added is the right side of the search, which `np.digitize` asks for by default. The stable `argsort` kernels came from a study of whole applications, where sorting was the largest cost no kernel touched; they are reached through a hook on the `argsort` method, because NumPy 2.5 no longer sorts through the per-type table that `argmax` is hooked into. The search on int64 came from a production pipeline, whose calls with one `np.int64` needle all reached the hook and went to NumPy ([issue 160](https://github.com/sofeikov/hotkernels/issues/160)). An ordinary NumPy program that uses eleven of these operations on two million elements runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.

The last chapter collects every number, and is candid about which of them mean less than they appear to.
