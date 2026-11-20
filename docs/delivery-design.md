# Delivery: the binding and `install()`

This began as a proposal and is now implemented: the package is in [`hotkernels/`](../hotkernels) and its tests in [`delivery/`](../delivery). Where the building departed from the proposal, [What changed in the building](#what-changed-in-the-building) says so, and [Status](#status) holds the result against the exit criterion. The rest of the text stands as proposed.

It details the second half of phase 3 of [core-design.md](core-design.md). The first half produced 22 kernels that the judge finds correct and faster than NumPy ([kernel-findings.md](kernel-findings.md)). None of them can be called from Python. This document proposes how they reach unmodified NumPy code: the binding, `install()`, `report()`, and how all of it is tested.

The core design left two questions for the start of this phase: which patching mechanisms work on current NumPy, and whether indexing syntax and array methods can be reached at all. Both were answered with a throwaway probe before writing this, and the answers shape the design. They are in [What was verified](#what-was-verified).

The binding is the same C on every architecture. The figures in this document, the list of kernels installed and the limits the binding screens for are ARM64's: every measurement was taken on an Apple M1 Pro. x86-64 gets its own figures and its own list when the package is built there ([x86-64-design.md](x86-64-design.md#the-package-and-its-delivery)). They go beside these, not in place of them.

## Summary

- **Everything can be reached**, including `a[mask]`, `x[mask] = v` and `x.astype(...)`, which the core design recorded as a known gap. It takes five kinds of hook. One is an API NumPy provides for the purpose. The other four change NumPy's own objects in place.
- **A hook costs about 1 ns on a call that does not take a kernel.** Measured: 0.5 to 1.4 ns per call, on calls that cost 59 ns and up.
- **End to end from Python, the speedups hold.** At a million elements: `np.argmax` 8.6x, `np.clip` 4.9x, `x[mask]` 9.8x to 16x, `x[mask] = v` 7.6x to 21x. The mask selection figure is lower than the judge's 22.5x, because a result has to be allocated.
- **A hooked function or method can also skip NumPy's Python wrapper.** `np.argmax` on 16 elements drops from 573 ns to 65 ns. `np.clip` and `x.argmax()`, which first had only a deeper hook, got the same treatment later: on 16 elements `np.clip` drops from 1.2 µs to 120 ns and `x.argmax()` from 172 ns to 55 ns ([issue 17](https://github.com/sofeikov/kernel-arena/issues/17)). That is a win of the binding, not of a kernel.
- **Two kernels could not serve real calls when this was built.** `searchsorted_f64` and `interp_f64` took one length for both of their inputs. They were given a second and bound afterwards; see [two-sizes-design.md](two-sizes-design.md).
- **One decision replaces an earlier one.** The float `cumsum` kernels are not installed by default. See [Decisions made in review](#decisions-made-in-review).

## Status

Against the [exit criterion](#exit-criterion), on an Apple M1 Pro with NumPy 2.5.5:

| # | Criterion | Result |
|---|---|---|
| 1 | An unmodified script runs faster with the one line, and `report()` shows the kernels hit | Met. [`examples/workload.py`](../examples/workload.py) uses eleven operations on two million elements: 45 ms on plain NumPy, 8.4 ms installed, identical results. `uv run delivery demo` |
| 2 | `delivery check` passes | Met. 96,734 calls over the 15 hooked operations agree in result, exception, warnings and the state of their arguments, and the kernel ran exactly where each case says |
| 3 | NumPy's test suite gives the same outcome | Met. 47,234 tests, none differs. The tests that compile Fortran (`f2py`) are left out. CI now checks it on both platforms on every pull request and on every release wheel ([numpy-tests-ci-design.md](numpy-tests-ci-design.md)) |
| 4 | A fallback costs at most 2 ns more | Met for the calls where 2 ns can be measured: `x[3]`, `x[1:5]` and `x[3] = 1.0` cost 0.3 to 1.5 ns more. On calls of 130 ns and up the difference is inside the noise of the measurement, a percent or two either way |
| 5 | No installed entry point is slower than NumPy at any size | Met. The lowest figure was `np.clip` at 16 elements, 1.00x; with its function hook ([issue 17](https://github.com/sofeikov/kernel-arena/issues/17)) it is 9.8x, and the lowest is now `np.digitize` at 16 elements, 1.10x. The type slot needed a minimum size to get there; see below. Since the second shortlist this holds for the shapes `delivery time` measures, not for every shape: `np.fmax.reduceat(x, starts)` is faster than NumPy on runs of 16 (3.6x at 4,096 and 3.1x at 1,048,576 elements on this machine, with its ARM64 kernel and the binding's ARM64 read) and slower on long runs, where the binding reads the array once more for signalling NaNs: on this machine 1.3x to 1.8x on runs of 64, and on runs of 256 to 4,096 0.95x to 1.16x at 4,096 elements and 0.61x to 0.80x at 262,144 and 1,048,576; on an Intel Core i5-14600KF level on runs of 512 and 0.8x on runs of 4,096 in 262,144 elements |
| 6 | Every kernel has a verdict | Met. 21 installed, 2 installed only with `inexact=True` (`cumsum_f32`, `cumsum_f64`). `searchsorted_f64` and `interp_f64` were not bound at first (one length for two inputs) and are now |

The timings were taken on a machine in use (load average about 3). Every statement, at three sizes, is in [`delivery/results/apple-m1-pro.md`](../delivery/results/apple-m1-pro.md), which holds the most recent run rather than this one. At a million elements, from Python, in the run made for the exit criterion:

| Statement | Faster by | Judge, kernel alone, vs NumPy |
|---|---|---|
| `np.argmax(x)`, float32 | 8.98x | 8.82x |
| `np.clip(x, -0.5, 0.5)`, float32 | 11.2x | 11.1x idle, 8.67x under load |
| `np.clip(x, -0.5, 0.5)`, float64 | 5.01x | 4.88x idle, 3.65x under load |
| `x[mask]`, float32, half selected | 31.0x | 41.7x |
| `x[mask]`, float64, half selected | 15.8x | 22.5x |
| `x[mask] = 0.0`, float64, half selected | 20.3x | 21.4x |
| `np.nonzero(mask)`, half selected | 3.52x | 5.15x |
| `np.nonzero(mask)`, few selected | 13.9x | 19.2x |
| `np.cumsum(x)`, int64 | 13.6x | 12.5x |
| `np.cumsum(x)`, float32, `inexact=True` | 29.2x | 29.0x |
| `x.astype(np.float32)`, uint8 | 5.29x | 2.68x |
| `np.sum(x)`, uint8 | 15.6x | 15.7x |
| `np.count_nonzero(x)`, float64 | 3.46x | 3.79x |
| `np.histogram(x, 50, (-3.0, 3.0))` | 9.40x | 9.70x |
| `np.bincount(x)`, uint8 | 6.74x | 6.74x |
| `np.isin(x, eight_values)`, values close together | 18.2x | 75.6x |

Three rows are well below the judge's figure. The mask kernels pay for a buffer and a copy, and for reading the mask once to check it. For `np.isin` the difference is on NumPy's side: NumPy takes 2.2 ms for this call here and took 9.3 ms in the judge's run, and why has not been established. The kernel's own time is the same (122 µs here, 115 µs in review).

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| The type slot inherits NumPy's behaviour and serves every call | It takes the interpreter lock, and leaves runs under 128 elements to NumPy | The lock is needed because the kernels are not safe against a writer ([issue 9]({I}9)). Taking it costs about 40 ns per call, measured: `x.argmax()` on 16 float32 elements went from 174 ns to 210 ns with the kernel, and a row of 16 under `axis=1` from 10 ns to 45 ns. The kernel breaks even at 96 to 128 elements. Short calls without an axis are served by the function hook and, since [issue 17](https://github.com/sofeikov/kernel-arena/issues/17), the method hook, which already hold the lock |
| Counters are atomic | They are atomic objects updated by a separate load and store | An atomic add cost more than the rest of a fallback. Two threads counting the same thing at once can lose a count; `report()` is statistics, and the self-test at install runs on one thread |
| The indexing hook calls `PyType_Modified` | It does not | Nothing CPython caches depends on the two pointers. And CPython stops caching attribute lookups for a type that has been modified a thousand times, which a test that installs repeatedly reaches |
| One set of checks, shared | The same, with each hook's work kept out of line | The compiler folded the work of taking a kernel into the hook, so every index operation saved twelve registers and reserved the scratch buffer before its first test: 2 to 4 ns. Out of line, the indexing hook is nine instructions |
| Arguments are checked, then the array | The array first | `np.cumsum` on a float array spent about 60 ns comparing keyword names before finding it had no kernel for the type |
| `entry/`, one C file per NumPy operation | One C file per kind of hook | An operation's code is short and belongs with the calling convention of its hook |
| `x[mask] = v` for a Python float | Also a Python int | `x[mask] = 0` is the commonest spelling. NumPy converts the int to the same double |
| `np.isin` with an int64 array of test values | Also a list or tuple of Python ints | `np.isin(x, [1, 2, 3])` is the commonest spelling, and NumPy converts such a list to int64 |
| A scratch kernel runs on a buffer of `n` elements, then a copy | The same; the buffer is on the stack up to 2 KB | An allocation is most of a small call |
| `meson-python`, not yet tried | `meson-python`, in a `uv` workspace | It worked as hoped. The package is installed non-editable, with the kernel and binding sources as `uv` cache keys, so `uv run` rebuilds it when they change |
| Any NumPy 2, with the self-test as the guard (open decision 5) | NumPy 2.3 or later; `install()` refuses an older NumPy 2 | NumPy 2.0 to 2.2 answer `np.clip` and `np.count_nonzero` differently from later releases, the kernels follow the later answer, and the self-test does not see the difference. See [Which NumPy versions](#which-numpy-versions) |

Two defects in the binding were found by `delivery check` while it was being written, and are fixed: `np.count_nonzero` returned a Python `int` where NumPy returns `np.intp`, and `np.histogram` issued NumPy's overflow warnings twice for a range wider than a double can hold.

Not built, and recorded as issues:

| Left for later | Issue |
|---|---|
| The function forms of the mask operations: `np.compress`, `np.extract`, `np.putmask`, `np.copyto`, `np.where(mask)`. Done: each has a function hook; see [Which hook serves which kernel](#which-hook-serves-which-kernel) | [15](https://github.com/sofeikov/kernel-arena/issues/15) |
| Skipping NumPy's wrapper for `np.clip` and for methods on small arrays. Done: `np.clip` has a function hook, and `x.clip`, `x.argmax()` and `x.argmin()` method hooks; see [Calling the kernel above NumPy's wrapper](#calling-the-kernel-above-numpys-wrapper) | [17](https://github.com/sofeikov/kernel-arena/issues/17) |
| `delivery check` against the oldest and newest NumPy 2 releases. Done: it runs in CI, and it moved the floor to NumPy 2.3. See [Which NumPy versions](#which-numpy-versions) | [18](https://github.com/sofeikov/kernel-arena/issues/18) |
| Wheels, a build that needs no checkout, `python -m hotkernels` | [19](https://github.com/sofeikov/kernel-arena/issues/19) |
| `x[mask]` and `np.nonzero` keep about two thirds of the kernel's speed: the buffer, the copy and the mask check | [20](https://github.com/sofeikov/kernel-arena/issues/20) |
| Free-threaded Python | [21](https://github.com/sofeikov/kernel-arena/issues/21) |
| A case with repeated test values in the `isin_i64` spec, since the binding pads with repeats. Done: the spec has value classes that pad the set the way the binding does | [22](https://github.com/sofeikov/kernel-arena/issues/22) |

Open before this work and still open: in-place `cumsum` ([7](https://github.com/sofeikov/kernel-arena/issues/7); in-place `clip` is now served through a buffer of the binding's, [clip-in-place-design.md](clip-in-place-design.md)), mask bytes other than 0 or 1 in the kernels themselves ([8](https://github.com/sofeikov/kernel-arena/issues/8)), the kernels' safety against a writer during `argmax` (found in [9](https://github.com/sofeikov/kernel-arena/issues/9), now [56](https://github.com/sofeikov/kernel-arena/issues/56)), and whether float `cumsum` may be non-monotonic at all ([10](https://github.com/sofeikov/kernel-arena/issues/10)). The binding screens for each; none is fixed in the kernels.

## What the user gets

```python
import hotkernels
hotkernels.install()
```

The whole interface:

| Call | Does |
|---|---|
| `hotkernels.install(inexact=False)` | Applies every hook. Returns nothing. With `inexact=True` it also installs the kernels that are not bit-identical to NumPy |
| `hotkernels.uninstall()` | Removes every hook. NumPy is exactly as it was |
| `with hotkernels.enabled():` | `install()` on entry, `uninstall()` on exit if this block did the installing |
| `hotkernels.report()` | Returns what is installed and how often each kernel ran. Printing it gives a table |

Rules:

- **Importing does nothing.** NumPy is only changed by `install()`.
- **It covers the whole process**, every thread and every library that calls NumPy, not only the caller's own code. That is the point, and it is why the context manager is not a way to limit hotkernels to one thread.
- **`install()` never fails because of the environment.** On a machine, Python build or NumPy it does not support, it does nothing, and `report()` says why.
- **Calling it twice is harmless.** So is `uninstall()` without `install()`.
- **The binding never converts an argument and never raises an error of its own.** A call either meets a kernel's conditions exactly as it stands, or goes to NumPy with the caller's own arguments. Every error message and warning a user sees is NumPy's.

`print(hotkernels.report())` after running a script, as an illustration of the layout:

```
hotkernels: installed, 21 of 23 kernels (ARM64, NumPy 2.5.5)

NumPy call          Elements   Kernel       Result   Kernel calls   Left to NumPy
np.argmax           float32    assembly     exact          10,412               0
np.clip             float64    assembly     exact           2,000              17
x[mask]             float64    assembly     exact             640           1,203
x.astype(float32)   uint8      C            exact              12               0
np.cumsum           float64    assembly     within tolerance: not installed, needs install(inexact=True)
...
```

"Left to NumPy" counts calls that reached a hook and did not meet the kernel's conditions. It is how a user finds out that their arrays are strided, or of another type, and that nothing was gained.

## What was verified

A C extension of about 260 lines, written to be thrown away, applied each kind of hook to NumPy and recorded which calls arrived and what they cost. It linked four real kernels: `argmax_f32`, `clip_f64`, `compress_f64` and `maskfill_f64`. It is not in the repository.

- **Machine.** Apple M1 Pro, macOS 27.2.1, CPython 3.12.7, NumPy 2.5.5. Native.
- **Timing.** From Python, with and without the hook in alternating batches, the minimum of 11 to 15 batches. The machine was in use (load average about 3), so these are good to a nanosecond or two per call and a few percent at large sizes. The implementation measures again with the real binding.

### The five hooks

| Hook | What is changed | Provided by NumPy for this? |
|---|---|---|
| Inner loop | A ufunc's loop for one type, through `PyUFunc_ReplaceLoopBySignature` | Yes |
| Type slot | The `argmax` and `argmin` entries a NumPy type holds, through `PyDataType_GetArrFuncs` | The accessor is public API; writing through it is not a documented use |
| Function | The call pointer inside a NumPy function object such as `np.argmax` | No. Its position is given by the object's type, which is a CPython guarantee |
| Method | The C function pointer in `ndarray`'s method table, and since the second shortlist two entries of the ufunc type's: `accumulate` and `reduceat`, which every ufunc object shares | No |
| Indexing | The two pointers behind `a[i]` and `a[i] = v` in `ndarray`'s type | No |

The last three keep NumPy's objects and change one pointer inside them. The alternative is to replace the objects: assign a new function to `np.argmax`, or put a new method into `ndarray`'s dictionary. Both were tried. Replacing is worse in every respect measured:

| | Replace the name `np.argmax` | Hook the function object in place |
|---|---|---|
| `np.argmax(x)` | Reached | Reached |
| `from numpy import argmax` done before `install()` | Not reached | Reached |
| NumPy calling its own function (`np.nanargmax`) | Not reached | Reached |
| `np.argmax` is still the same object, pickles, keeps its signature | No | Yes |
| A library that registers `np.argmax` as a key for the `__array_function__` protocol | Breaks if it is imported after `install()` | Works |

### What each hook reaches

| Hook, tried on | Reached | Not reached |
|---|---|---|
| Type slot, float32 `argmax` | `np.argmax(x)`, `x.argmax()`, a function imported earlier, `np.nanargmax`, strided input (NumPy passes a contiguous copy), `axis=` on a 2-D array (one call per row or column) | Other types |
| Inner loop, float64 `clip` | `np.clip(x, 0, 1)`, `x.clip(0, 1)`, `out=`, in place, contiguous arrays of any shape (as one flat run) | `np.clip(x, 0, None)`, which NumPy sends to `np.maximum` |
| Function, `np.argmax` | Every call of the function, however it was imported or spelled | `x.argmax()` |
| Method, five methods | The method, a bound method saved earlier, and the NumPy function of the same name where that function calls the method: `np.searchsorted`, `np.digitize`, `np.clip`, `np.argmax` | `np.sum(x)`, which does not go through `x.sum()` |
| Method, `argsort` (built for [issue 188](https://github.com/sofeikov/hotkernels/issues/188)) | `x.argsort(...)`, `np.argsort(x, ...)`, which calls the method, `np.unique(x, return_index=True)`, and a masked array's `argsort`, which calls the method of the plain array underneath | `np.lexsort`, which reads the element type's table of sort functions, and `np.sort`. The type slot, which reached every sort up to NumPy 2.3, reaches only `np.lexsort` on NumPy 2.5: [sort-design.md](sort-design.md#the-hook) |
| Method, the ufunc type's `accumulate` (built with the second shortlist) | `np.maximum.accumulate(x)`, the same for every other ufunc, and a bound method saved earlier: one entry serves every ufunc object, so the wrapper tells them apart by `self` | None found. Since [issue 187](https://github.com/sofeikov/hotkernels/issues/187) it is known that NumPy's own `cumsum` and `cumprod` with an axis (`a.cumprod(axis=0)`, `np.cumsum(a, 0)`, `np.cumulative_sum`) look the method up on `np.add` or `np.multiply` and call it, so they arrive here too |
| Method, the ufunc type's `reduceat` (built with the second shortlist) | `np.fmax.reduceat(x, starts)`, the same for every other ufunc, and a bound method saved earlier, as for `accumulate` | Not examined, as for `accumulate`. Only `np.fmax` has a kernel |
| Indexing | `x[mask]`, `x[mask] = v`, `operator.getitem` | Subclasses of `ndarray`, whenever they were defined; `x.__getitem__(mask)` called by name |

Two details matter for the design:

- **The inner loop and the type slot are called after NumPy has done its own checking.** The type is settled and the memory layout is described by the arguments. A strided call shows up as a step that is not the element size and is handed back to NumPy's loop.
- **The method hook was run for the five methods that use CPython's fast calling convention** (`searchsorted`, `sum`, `astype`, `clip`, `argmax`). `nonzero`, `cumsum` and `compress` use the older tuple conventions. The same change applies to them with a wrapper of the matching shape. That was not run.

### What a hook costs

On calls that reach a hook and do not take a kernel, at 16 elements:

| Hook | Call | NumPy alone | With the hook | Difference |
|---|---|---|---|---|
| Function | `np.argmax(float64)` | 573.2 ns | 569.5 ns | Within noise |
| Inner loop | `np.clip(x[::2], -0.5, 0.5)` | 1,261.0 ns | 1,262.1 ns | +1.2 ns |
| Method | `x.astype(np.float32)` | 202.4 ns | 202.9 ns | +0.5 ns |
| Indexing | `x[3]` | 58.6 to 59.1 ns | 59.7 to 60.0 ns | +0.6 to +1.4 ns |
| Indexing | `x[1:5]` | 88.3 ns | 89.4 ns | +1.1 ns |

The type slot has no such row: the slot belongs to one type, so every call that arrives takes the kernel.

The indexing hook is the one that matters. It sits on every index operation on every array in the process, and adds 1% to 2% to the cheapest of them.

### End to end

Python statement to Python result, NumPy alone against the hooked call:

| Statement | Elements | NumPy | With the kernel | Faster by | Judge, kernel alone |
|---|---|---|---|---|---|
| `np.argmax(x)`, float32, type slot | 1,048,576 | 440.9 µs | 51.1 µs | 8.6x | 8.82x |
| `x.argmax()`, float32, type slot | 16 | 173.9 ns | 168.5 ns | 1.03x | |
| `np.argmax(x)`, float32, type slot | 16 | 562.8 ns | 555.8 ns | 1.01x | |
| `np.argmax(x)`, float32, function hook | 16 | 572.5 ns | 65.2 ns | 8.8x | |
| `np.clip(x, -0.5, 0.5)`, float64, inner loop | 1,048,576 | 661.5 µs | 135.1 µs | 4.9x | 4.88x idle, 3.65x under load |
| `np.clip(x, -0.5, 0.5)`, float64, inner loop | 16 | 1,167.0 ns | 1,154.6 ns | 1.01x | |
| `x[mask]`, float64, half selected | 16 | 249.0 ns | 159.4 ns | 1.6x | |
| `x[mask]`, float64, half selected | 4,096 | 9.66 µs | 0.97 µs | 10.0x | |
| `x[mask]`, float64, half selected | 1,048,576 | 4,814 µs | 301 µs | 16.0x | 22.5x |
| `x[mask]`, float64, few selected | 1,048,576 | 915.6 µs | 93.1 µs | 9.8x | 11.0x |
| `x[mask]`, float64, half selected | 16,777,216 | 76.95 ms | 6.30 ms | 12.2x | |
| `x[mask] = 0.0`, float64, half selected | 16 | 288.3 ns | 40.2 ns | 7.2x | |
| `x[mask] = 0.0`, float64, half selected | 1,048,576 | 3,799 µs | 179 µs | 21.2x | 21.4x |
| `x[mask] = 0.0`, float64, few selected | 1,048,576 | 817 µs | 108 µs | 7.6x | 7.48x |

What this says:

- **Kernels that write into memory the caller already has keep the judge's number.** `argmax`, `clip` and masked assignment land within a few percent of it.
- **Mask selection loses part of its number to allocation.** A scratch kernel needs room for all `n` elements. The probe allocated that, ran the kernel, and then either shrank the array in place or copied the result into one of the right size. Copying was faster: 301 µs against 555 µs at a million elements with half selected, and equal with few selected. The first number is the one in the table.
- **A deep hook does nothing for small arrays.** Through the type slot, `np.argmax` on 16 elements still costs 556 ns, nearly all of it NumPy's Python wrapper. The function hook removes the wrapper. This is the fixed cost the survey measured ([survey-design.md](survey-design.md#per-call-overhead)), and the function hook is the way to take it out.

## Design

### Which hook serves which kernel

Use the deepest hook that exists for an operation, because it reaches the most callers and needs the fewest checks. Add the function hook where the fixed cost of the call is worth removing.

The hook for an operation does not depend on the architecture. The "Kernels" column counts ARM64's. A condition that comes from a kernel's own limit, such as `clip`'s "output is not the input", belongs to that architecture's kernel; see [What a caller must know](kernel-findings.md#what-a-caller-must-know).

| NumPy operation | Kernels | Hook | Takes the kernel when |
|---|---|---|---|
| `argmax`, `argmin` | 4 | Type slot, plus the function and the method for calls without an axis and along the rows of a matrix | Always, for float32 and float64. The type slot from 128 elements; the function and the method at any size, for a contiguous array and every other argument absent or None. Along axis 1 or -1 of a C-ordered two-dimensional array with at least one column, `out` absent or None and no `keepdims`, the function and the method call the kernel once per row for rows of 6 (float64) or 9 (float32) elements and more, and run a loop of the binding's own, one row per vector lane, for narrower rows; counted under `argmax_rows_f64` and the like, loops built into the binding and not kernels ([argmax-rows-design.md](argmax-rows-design.md)). A Python call the slot sees row by row is counted once ([Counting](#counting)) |
| `clip` | 2 | Inner loop, plus the function and the method | Contiguous, scalar bounds with `lo <= hi`, output clear of the input or exactly the input; an in-place call is clipped through a buffer of the binding's ([clip-in-place-design.md](clip-in-place-design.md)). The function and the method call the kernel themselves when there is no `out`, or `out` is the array itself, and both bounds keep the array's element type; see [below](#calling-the-kernel-above-numpys-wrapper). On int64, which has no kernel, the function and the method run a C loop of the binding's own for 1 to 4,096 elements, both bounds Python ints that fit int64 or `np.int64`, `lo <= hi`, and no `out`; counted under `clip_i64`, a loop built into the binding and not a kernel. An int64 call left to NumPy is not counted, as before |
| `x[mask]` | 2 | Indexing | Both contiguous and of the same shape, of any number of dimensions; a plain `ndarray` indexed by a boolean `ndarray`. A mask of fewer dimensions selects rows and is NumPy's. float32, float64, and int64, which the float64 kernel selects as bit patterns |
| `np.compress(mask, x)`, `np.extract(mask, x)` | The same 2 | Function | Both contiguous and of the same size; `x` of any shape, which both functions flatten. For `np.compress` the mask is 1-D and there is no `axis` or `out`; for `np.extract` the mask may have any shape |
| `x[mask] = v` | 1 | Indexing | As for `x[mask]`, `x` writable and not sharing memory with the mask, and `v` a Python float or int |
| `np.putmask(x, mask, v)`, `np.copyto(x, v, where=mask)` | The same 1 | Function | As for `x[mask] = v`, with `x` of any shape. `np.putmask` needs only the sizes to match; `np.copyto` needs the same shape and no `casting` |
| `nonzero` | 1 | Method | 1-D contiguous boolean array |
| `np.where(mask)` | The same 1 | Function | One argument, a 1-D contiguous boolean array. `np.where` calls NumPy's C code for `nonzero` directly, so the method hook does not see it |
| `np.where(mask, a, b)` | 1 | Function | Three positional arguments: a contiguous boolean array of at least one dimension, and two contiguous arrays of its exact shape, both float64 or both int64 (which the float64 kernel moves as bit patterns); or, for float64, one such array and a Python float, a Python int a double holds exactly or an `np.float64` on the other side. The result is a new array, so it cannot overlap an input. The condition is read for bytes other than 0 and 1 first. A scalar side is spread over a buffer on the stack and the kernel runs a block at a time. Counted as its own operation, `where` |
| `searchsorted` | 2 | Method | The array one-dimensional, contiguous float64; the needles a contiguous float64 array of any shape or a Python number; `side` absent, `'left'` or `'right'`, each side with its own kernel ([searchsorted-right-design.md](searchsorted-right-design.md)); no `sorter` |
| `argsort`, stable | 3 | Method | The array one-dimensional float64, int64 or uint64 in native byte order, contiguous or not (a view with a step or a reversed one, as `np.flip` returns and scikit-learn's metrics sort, is copied element by element) (C's `long long` and its unsigned form included, which are NumPy types of their own and 64 bits wide), with 256 to 2^26 - 1 elements; `kind` the string `'stable'` or `'mergesort'` with `stable` absent or None, or `kind` absent or None and `stable` True; `axis` absent or the Python int -1 or 0; `order` absent or None; `descending` absent. NumPy reads only the first letter of `kind`, and the other spellings of a stable sort (`'Stable'`, `'m'`) are left to it. The array is looked at first: one with no element below the one before it is answered `0, 1, 2, ...` by the hook, one in strictly descending order `n-1, ..., 0`, and one that is nearly in order (at most eight elements out of order, or one run that leaves out fewer than one element in 32) is NumPy's, whose timsort is the faster sort there. The kernel gets a copy of the array, made with the interpreter lock held, and sorts it with the lock released, into a buffer of the binding's own of which the first `n` words are copied to the result: four words of memory per element in all, and a call that cannot have them goes to NumPy. `np.argsort(x, kind=...)` calls the method, as do `np.unique(x, return_index=True)` and a masked array's `argsort`. The default kind is not served: [sort-design.md](sort-design.md#the-default-kind). A call on one of the three element types that is not taken is counted; on any other type it goes on uncounted. Counted as its own operation, `argsort` |
| `cumsum` | 4 | Method | Contiguous, no `axis`, `dtype` or `out`. The two float kernels only with `inexact=True`. A boolean array, which has a kernel of its own and gives `np.intp`, is read for bytes other than 0 and 1 first |
| `astype` | 1 | Method | Contiguous `uint8` to `float32`, other arguments at their defaults |
| `np.round(x, d)`, `np.around(x, d)`, `x.round(d)` | None: a loop of the binding's own, `round_f64` | Function (both names) and method | Contiguous float64 of 1 to 4,096 elements, `decimals` absent or a Python int from 0 to 15, `out` absent or None. The loop computes `rint(x * 10^d) / 10^d`, as NumPy does in three passes; the floating-point flags are cleared before it and read after it, and a flag NumPy reports (overflow in the multiply, invalid for a signalling NaN) drops the result and sends the call to NumPy, which warns. Counted as its own operation, `round`, at the method hook, which every call the function hooks pass on reaches. See [below](#calling-the-kernel-above-numpys-wrapper) |
| `np.maximum.accumulate(x)`, `np.fmax.accumulate(x)` | 2 | Method, of the ufunc type | One-dimensional, contiguous, at least one element; `axis` absent, 0 or -1, no `dtype` or `out`. int64 under either ufunc (`cummax_i64`); float64 under `np.fmax` only (`cumfmax_f64`), after the array is read for signalling NaNs, which NumPy does not ignore and the kernel does, and for a negative zero or NaNs of more than one bit pattern, where NumPy's answer differs between its x86-64 and ARM64 builds; an array with any of these is NumPy's. `np.maximum` on float64 propagates NaN and is NumPy's. Any other ufunc's `accumulate` goes on after two pointer comparisons, uncounted. Counted as its own operation, `accumulate` |
| Accumulation along the first axis of a matrix: `np.add`, `np.multiply`, `np.maximum` and `np.fmax`'s `accumulate`, and `a.cumsum(axis=0)`, `a.cumprod(axis=0)`, `np.cumsum`, `np.cumprod`, `np.nancumsum`, `np.cumulative_sum` and `np.cumulative_prod` with `axis=0`, which call it | None: four loops of the binding's own, `cumsum_rows_f64`, `cumprod_rows_f64`, `cummax_rows_f64`, `cumfmax_rows_f64` | Method, of the ufunc type, and `method.cumsum` | A plain C-ordered float64 array of two or more dimensions with at least one row and one column; `axis` the Python int 0 or `-ndim` (or absent, for `accumulate`); no `dtype`; `out` absent, None, or a plain C-ordered writable float64 array of the same shape that is the input's memory exactly or shares no byte with it. The loop computes `out[i] = op(out[i-1], x[i])` a row at a time, where NumPy goes a column at a time, so every element is the same operation on the same operands. For `np.add` and `np.multiply` the floating-point flags are read after each block of 16 KB of rows, and each row says whether it made a NaN; at the first block that raised a flag NumPy reports or made a NaN, NumPy's own `accumulate` finishes the call in place from the row before the block, and warns or raises as `np.errstate` says. `np.maximum` and `np.fmax` read the array first with the guard of `cumfmax_f64`. `a.cumsum(axis=0)` is taken at its method hook, so as not to be counted as passed on there; a call to `np.add` or `np.multiply` not taken is not counted. See [accumulate-rows-design.md](accumulate-rows-design.md) |
| `np.fmax.reduceat(x, starts)` | 1 | Method, of the ufunc type | `x` one-dimensional, contiguous float64, at least one element; `starts` a one-dimensional contiguous int64 array with at least one element, every one in `[0, n)` (each start is read once, in the loop over the runs, and one outside sends the call to NumPy, which raises); `axis` absent, 0 or -1, no `dtype` or `out`. Lists and other index types are NumPy's to convert. `fmax_reduce_f64` reduces one run, so the binding calls it once per run of two or more elements and copies a single element itself, by NumPy's rule: run i ends at the next start, or at n, and is the single element at its start when it does not start before it ends. Three guards send the whole call to NumPy: a signalling NaN anywhere in `x`, looked for a block of 2,048 elements at a time ahead of the runs; and, after the runs, a zero result while `x` holds a negative zero, or a NaN result while its NaNs are of more than one bit pattern. Any other ufunc's `reduceat` goes on after one pointer comparison, uncounted. Counted as its own operation, `reduceat` |
| `m.any(axis=1)`, `np.any(m, axis=1)` | 1 | Method and function | A C-ordered two-dimensional boolean array with at least one column; `axis` the Python int 1 or -1, by position or as `axis=`, and no other argument (`out`, `keepdims`, `where`, and for the method `dtype`). The function hook is needed because `np.any` calls `np.logical_or.reduce` itself for a plain `ndarray` and never reaches the method. `m.any()` with no argument goes on at the method hook's first test, uncounted; a call on such a matrix with arguments that is not taken is counted. The matrix is read for bytes other than 0 and 1 a block of 16,384 bytes at a time, just before the kernel reads the block. Zero rows are taken; zero columns are NumPy's. Counted as its own operation, `any` |
| `w -= b`, `w += b`, `np.subtract(w, b, out=w)`, `np.add(w, b, out=w)`, and the in-place add of NumPy's elision of temporaries | None: loops of the binding's own, `subtract_inplace_f64` and `add_inplace_f64`, ARM64 only | Inner loop of `subtract` and of `add` | float64; the output is the first input; all three operands step by 8 bytes; the second input is the output or shares no byte with it; 131,072 elements (1 MB) or more. The loop handles one 128-byte line per step and prefetches the output 4,096 bytes ahead for a store, which removes the cost of a read followed by a write to lines a threaded matrix product of Accelerate has left in other cores' caches; elsewhere it runs at NumPy's speed. Same instructions in the same order as NumPy's loop, so the result is bit-identical. Calls left to NumPy are not counted: every float64 add and subtract reaches the hook. Not built on x86-64. See [gradient-step-findings.md](gradient-step-findings.md) |
| `np.median(x)`, `np.median(x, axis=None)` | None: a loop of the binding's own, `median_f64` | Function | The array alone, by position, or with `axis=None` by position or keyword: a contiguous float64 array of any shape with at least one element and no NaN. Up to 64 elements they are copied to a buffer on the stack and sorted; above that the two middle elements are selected exactly from the bits of the elements, without a copy and with the interpreter lock released from 8,192 elements ([median-large-design.md](median-large-design.md)). The result is the middle element, or the two middle elements added and divided by two, as NumPy's mean of them is, as an `np.float64`. Sent to NumPy: a sum of the middle elements that overflows or is a NaN (infinities of both signs), a half that underflows, and a zero median when the array holds a negative zero. Every call not taken is counted, as its own operation, `median`. See [below](#calling-the-kernel-above-numpys-wrapper) |
| `sum` | 2 | Method and function | Contiguous `uint8` or boolean, no other arguments. A boolean array is read for bytes other than 0 and 1 first, a block at a time, and one that holds any is NumPy's. On ARM64 also a contiguous, aligned, one-dimensional float32 array with no other argument: `sum_f32` makes NumPy's additions in NumPy's pairwise order and returns its `np.float32` bit for bit. An empty array sums to `+0.0`, as in NumPy. A result that is not finite (a NaN, an infinity, an overflow) is thrown away and the call passed on, so the NaN's bits and the warning are NumPy's. An explicit `axis`, even `None` or `0`, goes to NumPy, as for `uint8`; so does a matrix, which NumPy sums in the same order as far as its sources say, not yet shown by `delivery check`. An array whose elements are not aligned to their size is NumPy's: NumPy copies it in blocks of its buffer size, 8,192 elements, and adds the blocks' sums one after another, another order. After the kernel the four flags NumPy reports are cleared and the inexact flag is left as the additions left it, which is what NumPy leaves. On x86-64 float32 stays with NumPy until the kernel is measured there ([coverage-gaps-design.md](coverage-gaps-design.md)) |
| `count_nonzero` | 2 | Function | Contiguous, no `axis` |
| `interp` | 1 | Function | Three contiguous float64 arrays, `xp` and `fp` one-dimensional and the same length; no `left`, `right` or `period`; no NaN in `xp`; at most 64 sample points per query |
| `histogram` | 1 | Function | Contiguous float64, an integer bin count, no weights or density |
| `bincount` | 3 | Function | One-dimensional, contiguous, no weights, `minlength` absent or a Python int >= 0. `uint8` (`bincount_u8`) whatever its values. `int16` (`bincount_i16`) and `int64` (`bincount_i64`) with at least one element: the array is read once for its largest element as an unsigned number, compiled for AVX2 with four running maxima, and a negative element, which reads as too large, sends the call to NumPy, which raises. The result has NumPy's length, the largest element plus one or `minlength`, and the kernel is given exactly that many counters, which it zeroes itself, so every element is inside its range. A result longer than 65,536 counts and longer than the array is NumPy's, whose memory comes zeroed from the system where the kernel would write every count; so is a largest element of the largest int64, whose length overflows. Empty arrays are NumPy's. The lock is released for the scan and for the kernel |
| `isin` | 1 | Function | Contiguous int64, at most eight test values, flags at their defaults |
| `unique` | 1 | Function | `np.unique(x)` on an int64 `ndarray` of any shape or layout with at least one element (one that is not C-contiguous is copied in C order first, as NumPy's `flatten` does); `return_index`, `return_inverse` and `return_counts` absent or `False`, `axis` absent or None, `equal_nan` absent or a bool, `sorted` absent or `True`. The array is read once for its smallest and largest elements, and the kernel takes the call when the largest minus the smallest is below 256 or below the element count; it marks the values present in a table of `span` values on the stack (up to 2 KB) or from `malloc`, and the values are copied into a new array of exactly their count. A wider range is NumPy's after the read. The lock is released for the read and the kernel from 8,192 elements. Counted as its own operation, `unique`. See [unique-design.md](unique-design.md) |
| `bottleneck.nanmean`, `.nanstd`, `.nanmin`, `.nanmax` | 4 | Function, of bottleneck's function objects ([bottleneck-design.md](bottleneck-design.md)) | A plain two-dimensional float64 array in native byte order, at least one element along each axis; `axis` a Python int naming either axis, by position or keyword; for `nanstd`, `ddof` absent or a Python int in C `int` range; no other argument. The kept axis has a stride of 8 bytes and the reduced axis a positive stride that is a multiple of 8. The result is bottleneck's, bit for bit, as a new array. Applied only when `bottleneck.reduce` is in `sys.modules`, or later by a finder on `sys.meta_path` when it is imported; never by importing it; left out by `install(bottleneck=False)`. Every call not taken goes on to the type's `tp_call` and is counted, as `bn.nanmean` and the rest |

"Contiguous" for a kernel that works element by element or reduces the whole array means any shape. NumPy's inner loops already work that way: a contiguous 40 by 25 array arrived at the `clip` loop as one run of 1,000.

The function forms of the mask operations (`np.compress`, `np.extract`, `np.putmask`, `np.copyto`, `np.where(mask)`) were not needed for the exit criterion and came after it ([issue 15](https://github.com/sofeikov/kernel-arena/issues/15)). Each is a function object of the type the function hook already changes, and each shares its kernel's checks with the indexing or method hook, including the scan of the mask for bytes other than 0 or 1. They differ from the syntax in what NumPy itself requires, and the table says how:

- **They flatten.** `np.compress` without an axis, `np.extract` and `np.putmask` read their arrays in C order whatever the shape, so a contiguous array of any shape is one flat buffer to the kernel. `np.copyto` broadcasts the mask against `x` instead, and a mask of another shape, even of the same size, is NumPy's to broadcast or refuse.
- **`np.compress(mask, x, out=o)` is NumPy's.** The compress kernels write scratch output, and a scratch kernel must never see a buffer the caller owns.
- **Only the masked operation is counted.** A call left to NumPy counts under "Left to NumPy" when it is the operation the kernel serves. `np.copyto` without `where` is a plain copy and passes through the hook without being counted. `np.where(mask, x, y)`, a selection between two arrays, has a kernel of its own since the second shortlist and is counted under `where`.

From Python, with `uv run delivery time` on the Apple M1 Pro, NumPy 2.5.5, half the mask selected. The machine was running other work (load average about 7), so the figures are rough. The run now in `delivery/results/`, made at a load average of 2.0, has all five within a few percent of them:

| Statement | 16 elements | 4,096 | 1,048,576 |
|---|---|---|---|
| `np.compress(mask, x)`, float64 | 6.20x | 9.33x | 6.20x |
| `np.extract(mask, x)`, float64 | 11.8x | 9.95x | 6.26x |
| `np.putmask(x, mask, 0.0)`, float64 | 4.03x | 5.36x | 18.4x |
| `np.copyto(x, 0.0, where=mask)`, float64 | 3.66x | 9.33x | 20.8x |
| `np.where(mask)` | 2.46x | 3.93x | 3.44x |

NumPy's own `np.compress` is faster than its `x[mask]` (1.97 ms against 5.1 ms at a million elements), so the same kernel gains less there. A call that passes through one of the five hooks to NumPy costs nothing measurable: `np.copyto(x, 0.0)` and `np.where(mask, 1.0, 0.0)` were within 1.3% of plain NumPy at every size.

The keywords each hook accepts are the ones NumPy 2.0 to 2.5 accept: `np.putmask` takes `x` by position only, and `np.where` takes no keywords at all. They were checked on 2.0.2, 2.2.6, 2.3.5 and 2.5.5, and in each the five are function objects of the type the hook expects.

#### Calling the kernel above NumPy's wrapper

`np.clip(x, lo, hi)` runs NumPy's Python wrapper, then `x.clip`, then NumPy's `_methods._clip` in Python, then the `clip` ufunc's machinery, and only then the inner loop the loop hook replaces. On 16 elements that path costs about 1.2 µs and the kernel 2 ns. `x.argmax()` has a shorter path, but the type slot leaves runs under 128 elements to NumPy. Both were left for after the exit criterion ([issue 17](https://github.com/sofeikov/kernel-arena/issues/17)). Four hooks now sit above them: a function hook on `np.clip` and method hooks on `x.clip`, `x.argmax()` and `x.argmin()`, in the kinds of hook that already existed.

- **`np.clip` and `x.clip` call the kernel themselves**, into a new array of the input's shape, when the array is a contiguous float32 or float64 `ndarray` with at least one element, both bounds are given, `out` is absent, None or, for an in-place clip, the writeable array itself (then the array is clipped through a buffer of the binding's and returned, [clip-in-place-design.md](clip-in-place-design.md)), nothing else is passed, and `lo <= hi` once the bounds are in the array's element type. Everything else goes on unchanged and reaches the loop hook, which serves what it served before, `out=` included.
- **The result's element type is NumPy's.** NumPy 2 promotes by NEP 50: a Python float or int is a weak scalar and takes the array's type, a NumPy scalar or a 0-d array is not. A float32 array clipped to Python floats stays float32; clipped to `np.float64` bounds it gives float64. So a bound is taken when it is a Python float, a Python int, or a NumPy scalar of exactly the array's type, and any other bound sends the call to NumPy, whose loop hook may then take the float64 call it makes.
- **A bound is converted as NumPy converts it.** For a float32 array NumPy rounds a Python float to the nearest float32 with a C cast, and so does the hook. A float beyond float32's range makes NumPy warn ("overflow encountered in cast"), and one below its smallest normal number raises the underflow flag; both go to NumPy, so the conversion here raises no flag NumPy reports. A Python int is taken only when the element type holds it exactly (2²⁴ + 1 is not a float32; a float64 holds every int up to 2⁵³), where every conversion agrees. `lo <= hi` is tested after the conversion and without raising a flag for a NaN, so `np.clip(x, 0.100000001, 0.1)` has equal bounds for a float32 array, as the loop sees them, and reversed ones for a float64 array. A bool goes to NumPy.
- **The spellings are NumPy's.** `np.clip` takes `a_min` and `a_max` by position or keyword, and `min=` and `max=` where NumPy has them. NumPy 2.1 added those two, so every NumPy hotkernels installs on has them; the hook still reads, when it is applied, whether the Python function behind `np.clip` has them as keyword-only parameters, and leaves them to NumPy where it does not (on NumPy 2.0 they reach the ufunc, which raises). Both spellings at once, a bound left out, and a None bound are NumPy's: it raises, or calls `np.maximum` or `np.minimum`. `x.clip` takes `min`, `max` and `out`, by position or keyword, as NumPy's `_methods._clip` does.
- **`out=` stays with the loop hook.** Writing into the caller's buffer needs the ufunc's rules for broadcasting and casting and the binding's test that the output is not the input ([issue 7](https://github.com/sofeikov/kernel-arena/issues/7)). The loop hook applies all of them already, and on the small arrays where the wrapper matters `out=` is rare.
- **Status flags and threads are as in the loop hook.** The flags are cleared after the kernel, as NumPy's clip loop clears them, and the lock is released from 8,192 elements, as in the other hooks. The output is a new array, or the array itself clipped through a buffer, so no kernel call sees an output that overlaps its input.
- **`x.argmax()` and `x.argmin()` take the kernel at every size** for a contiguous float32 or float64 array, with every argument absent or None: `x.argmax()`, `x.argmax(None)`, and `x.argmax(axis=None, out=None)`, which is how NumPy's wrapper calls the method for `np.argmax(x, axis=None)` and `np.argmax(a=x)`. A method runs with the interpreter lock held and keeps it, as the function hook does. An axis, an `out` array or `keepdims` go on to the type slot.
- **Short int64 arrays get a loop of the binding's own.** No kernel serves `np.clip` on int64, and NumPy's int64 clip loop is quick; on 32 elements the call costs about 1.75 µs, nearly all of it the call path above. So `np.clip` and `x.clip` on a contiguous int64 array of 1 to 4,096 elements, with both bounds Python ints that fit int64 or `np.int64` (NumPy keeps the result int64 for both, and raises for some Python ints outside the range and accepts others) and `lo <= hi`, run a plain C loop in the hook file, `min(max(x, lo), hi)`, into a new array. It is compiled for the kernels' instruction level (`HK_TARGET_KERNEL_LEVEL`), where it is a compare and a blend per four elements. Measured from Python on the Intel Core i5-14600KF, it is 15x faster than NumPy at 16 elements and 3.6x at 4,096; it stays ahead further (2.0x at 16,384, 1.5x at 32,768), but the limit is kept at 4,096, among the sizes it was chosen from, so the loop never needs to release the interpreter lock. Above the limit the call is NumPy's and costs under a percent more.
- **`np.round` on float64 gets one too.** `np.round(x, d)` and `np.around(x, d)`, two function objects with one body in NumPy 2, call NumPy's `_wrapfunc`, then the method `x.round`, then NumPy's `PyArray_Round`, which makes three ufunc calls: `np.multiply` by 10^d, `np.rint` in place and `np.true_divide` by 10^d, or `np.rint` alone for d = 0. On 16 elements that is about 1.1 µs. Function hooks on both names and a method hook on `x.round` run a loop that does the same three operations on each element, in the same order and the same rounding mode, so the result is the same bits, the sign of a zero included: a contiguous float64 array of 1 to 4,096 elements, `decimals` absent or a Python int from 0 to 15 (10^15 is a double exactly; negative decimals divide first and are NumPy's), and `out` absent or None. NumPy reports the flags of those operations as warnings or errors, by `np.errstate`; the loop clears MXCSR's flags before it and reads them after it, and if overflow or invalid is set (a product beyond the largest double, a signalling NaN) the result is dropped and the call goes to NumPy. Underflow cannot arise: a subnormal number times a power of ten is exact while it stays subnormal. The loop is compiled for the kernels' level, where `rint` is one `vroundpd`; for any x86-64 it would be a call into the C library per element. Measured from Python on the Intel Core i5-14600KF, `np.round(x, 2)` is 7.4x faster than NumPy at 16 elements and 2.2x at 4,096, and `np.round(x)` 3.5x and 1.7x; at 8,192 `np.round(x)` is 1.1x, so the limit is 4,096. A call either hook passes on reaches the method, and only the method counts it.
- **`np.median` on float64 gets one as well, and replaces all of NumPy's function.** `np.median(x)` is Python throughout: `_ureduce`, `_median`, `np.partition` of a copy at the middle (and at the end, for a NaN check), `np.mean` of the middle one or two elements, and a NaN check. About 4.8 µs at 32 elements. The function hook takes `np.median(x)` with the array alone, by position, on a contiguous float64 array of any shape (NumPy flattens without an axis) of 1 to 128 elements (since extended to every size and to `axis=None`; see below), reads it for NaN, copies it to a buffer on the stack, sorts the copy by insertion and returns the middle element, or the two middle elements added and then divided by two, which is what NumPy's mean of them computes, as an `np.float64`. The result is the same bits, with three exceptions that go to NumPy: an array holding a NaN; a sum that overflows or meets infinities of both signs, or a half that underflows, for which NumPy warns or raises by `np.errstate` (recognised from the numbers, not from the status flags, since the compiler may move inline arithmetic across a read of MXCSR); and a zero median when the array holds a negative zero, whose sign depends on where NumPy's partition leaves equal elements (and NumPy 2.5's sum of a lone `-0.0` is `+0.0`). The sort takes time in the square of the count. Measured from Python on the Intel Core i5-14600KF on normally distributed values, `np.median(x)` is 47x faster than NumPy at 16 elements, 13x at 64, 3.2x at 128 and 0.86x at 256, so the limit is 128; on values in descending order, the sort's worst case, it is 1.48x at 128. A call not taken costs two tests and a count before NumPy's function runs. Since then the loop takes arrays of every size: above 64 elements, where the sort had become slower than the alternative, it selects the middle elements by their bits, counting them by the eleven bits below those they all share and keeping only the digit that holds the middle ([median-large-design.md](median-large-design.md)), 7.8x faster than NumPy at a million elements on the Apple M1 Pro. The same change takes `np.median(x, axis=None)`, Astropy's spelling, at every size.
- **Accumulations along the first axis are loops of the binding too**, at every size: there the gain is the order in which memory is walked, which the judge, timing a kernel on flat buffers, would not see. See [accumulate-rows-design.md](accumulate-rows-design.md).
- **These loops are not kernels.** Where the gain is NumPy's call path and not the loop, a kernel would add nothing the judge could measure. The second shortlist (decision 4 of [second-shortlist-design.md](second-shortlist-design.md#decisions-still-open)) settled that such loops belong in the binding, as plain C, for short arrays only: no `kernels/` directory, no assembly, no judge, checked against NumPy by `delivery check` like the rest of the binding. They have counters like a kernel's, so that `install()` can probe them and `report()` show them, with implementation `C` and result `exact`; the count of kernels on the report's first line leaves them out.
- **Nothing is counted twice.** None of the four hooks counts a call it passes on: the call reaches the type slot or the clip loop, whose hooks count it, or no kernel at all (an integer array, a None bound), which was not counted before either.
- **The self-test proves the kernel, not which hook ran it, for clip.** The `x.argmax()` probes use 16 elements, which only the method hook takes. The clip loop takes every call the function and method hooks take, so their probes show a right result from a clip kernel. `delivery check` closes the gap: it removes the loop hooks for a moment and requires `np.clip` and `x.clip` on both types to reach the kernels without them.

From Python, with `uv run delivery time --record` on the Apple M1 Pro, NumPy 2.5.5, in a run that started at a load average of 2.0 on a machine that also hosts other work. "Before" is the run recorded before these hooks, at a load average of 3.8:

| Statement | 16 elements, before | 16 elements | 4,096, before | 4,096 | 1,048,576 |
|---|---|---|---|---|---|
| `np.clip(x, -0.5, 0.5)`, float32 | 1.02x | 11.0x (1.31 µs to 119 ns) | 2.46x | 11.3x | 9.92x |
| `np.clip(x, -0.5, 0.5)`, float64 | 1.00x | 9.76x (1.16 µs to 118 ns) | 2.31x | 6.91x | 4.93x |
| `np.clip(x, 0, 1)`, float64 | 1.01x | 11.1x | 2.07x | 6.06x | 4.78x |
| `x.clip(-0.5, 0.5)`, float32 | not timed | 7.54x (791 ns to 105 ns) | not timed | 10.1x | 11.7x |
| `x.argmax()`, float32 | 1.00x, left to NumPy | 3.11x (172 ns to 55 ns) | 5.95x | 11.2x | 8.87x |

At a million elements the direct path and the loop hook run the same kernel on the same fresh output, and measured alternately in one process they took 138 µs and 142 µs for float64. The 4,096-element figures gain as well as the 16-element ones because the ufunc machinery costs about 1 µs at any size.

A call these hooks pass on pays for the tests. `np.clip(x, -0.5, None)` on 16 float64 elements went from 1.03 µs to 1.04 µs (+1.4%), `np.clip(x[::2], -0.5, 0.5)` +0.9% and `x.argmax()` on int64 +0.7%: a percent or two of a call that costs a microsecond, inside the noise the results file states. Getting there took one change to code every hook shares. Keyword names were compared with a `strlen` and a `memcmp` per parameter, which cost about 20 ns on `x.clip(lo, None, out=None)`, the call `np.clip` makes when it passes a None bound on; they are now compared in a loop of the binding's own.

### Considered and not proposed

- **Replacing names instead of hooking objects.** Compared above.
- **`cumsum` through the inner loop of `add`.** NumPy runs `np.cumsum` as that loop with the output one element ahead of its first input, so the call can be recognised, and it would reach `axis=` and `np.add.accumulate` as well. Two things count against it. The loop is handed the array from its second element on, with the first already stored, and the kernels cannot start from a value. And the check would sit on every float64 addition in the process: +1.0 ns per call, measured.
- **`np.sum` on `uint8` through the inner loop.** NumPy converts the bytes to 64-bit integers in chunks before the loop sees them: 13 loop calls for 100,000 elements. The kernel would never see a byte.

### When a call takes a kernel

One set of checks, written once and shared by the function, method and indexing hooks:

- The array's type is exactly `ndarray`. Subclasses, masked arrays and memory-mapped arrays go to NumPy.
- Its element type is exactly the kernel's, in native byte order.
- It is contiguous. Alignment is not required; the kernels do not need it.
- It has at least the kernel's `min_n` elements.
- Every argument the kernel does not implement is absent or at its default.
- Scalars are inside the domain the kernel's spec declares. For `clip` that is `lo <= hi`, which a NaN bound also fails.
- An output the caller supplies is writable and does not overlap an input.
- The size is at or above any minimum that end-to-end timing sets for the entry point (see [Exit criterion](#exit-criterion)).

Anything else is passed to the original function pointer with the same arguments. Nothing is copied or converted on the way, which is why a fallback costs a nanosecond.

### Inputs outside the kernels' contracts

Review of the ARM64 kernels found inputs where a kernel is wrong, or unsafe, and its spec does not say so. The binding is the only place they can be stopped. Each costs a check or a pass:

| Input | What the kernel does | What the binding does |
|---|---|---|
| A mask byte other than 0 or 1, which a boolean array can hold ([issue 8](https://github.com/sofeikov/kernel-arena/issues/8)) | Counts past the capacity and writes outside the buffer | Reads the mask once before the kernel. Any byte above 1 sends the call to NumPy |
| The output is the input ([issue 7](https://github.com/sofeikov/kernel-arena/issues/7)) | `clip` and float `cumsum` are wrong on some data | `clip`: copies each run of 4 KB to a buffer on the stack and has the kernel clip the copy back over it ([clip-in-place-design.md](clip-in-place-design.md)); an output that overlaps the input only in part goes to NumPy. `cumsum`: sends the call to NumPy |
| Another thread writes the array during `argmax` ([issue 9](https://github.com/sofeikov/kernel-arena/issues/9)) | Reads it twice and may read past the end | Every `argmax` kernel call runs with the interpreter lock held: the type-slot hook takes it back from NumPy, and the function and method hooks never release it. That stops Python threads, and does not stop code that writes without the lock. Bounding the kernels' scans is [issue 56](https://github.com/sofeikov/kernel-arena/issues/56) |
| A histogram range whose bin edges are not strictly increasing ([issue 9](https://github.com/sofeikov/kernel-arena/issues/9)) | Returns counts where NumPy raises | Builds the edges first, as NumPy does, and sends the call to NumPy if they are not strictly increasing |
| `clip` with array bounds ([issue 9](https://github.com/sofeikov/kernel-arena/issues/9)) | Matches only NumPy's scalar-bound loop | Takes the kernel for scalar bounds only |
| A NaN among `np.interp`'s sample points ([issue 9](https://github.com/sofeikov/kernel-arena/issues/9)) | At `xp[0]` it reads before the start of `xp` and `fp`; elsewhere it returns something other than NumPy | Reads `xp` once before the kernel. Any NaN sends the call to NumPy. The pass costs about 0.13 ns per sample point, so the kernel is taken only with at most 64 sample points per query |
| Input to `searchsorted` or `interp` that is not sorted | Stays inside its buffers and returns a meaningless result, not the same one as NumPy's | Nothing: NumPy does not check either. See [two-sizes-design.md](two-sizes-design.md#inputs-outside-the-kernels-contracts) |
| Another thread writes the array during a stable `argsort` | The kernel reads its input three times and sizes the places it scatters to from the first reading. Elements that change in between can send a write outside its buffers | Copies the array with the interpreter lock held and gives the kernel the copy, so nothing a thread does to the array afterwards reaches it, whether or not that thread holds the lock. The copy is about a fiftieth of the sort |
| Floating-point status flags ([issue 9](https://github.com/sofeikov/kernel-arena/issues/9)) | `clip` leaves the invalid flag set; float `cumsum` raises overflow and invalid where NumPy's sequential sum does not; `cumfmax_f64` and `fmax_reduce_f64` (x86-64) leave the invalid flag set after a NaN, quiet ones included, through `vmaxpd`. NumPy turns flags into warnings, and leaves none set after its own loops | `clip`: clears the flags after the kernel, in each of its three hooks, which is what NumPy's own loop does. `cumsum`: clears them before, and if any is set afterwards discards the kernel's output and lets NumPy compute and warn. `accumulate` and `reduceat`: clear them after the kernel. Float `sum` (ARM64): after its own `np.sum` NumPy leaves the inexact flag as its additions left it and the other four clear, and the kernel makes the same additions, so only the four NumPy reports are cleared, by reading FPSR and writing it only when one is set. On x86-64 only MXCSR's flags are cleared, the only ones an SSE or AVX kernel can set: about 1 ns, against 46 ns for the C library's `feclearexcept` |
| The sign of a zero maximum with zeros of both signs present, and which NaN an all-NaN run returns, in `np.fmax.reduceat` | `fmax_reduce_f64` returns the bits NumPy's AVX2 loop returns, which follow its four lanes. NumPy picks its AVX2 or its AVX-512 loop when it starts, and the AVX-512 loop has eight | Sends the whole call to NumPy when `x` holds a negative zero and a result is a zero, or holds NaNs of more than one pattern and a result is a NaN. Elsewhere every correct maximum is one bit pattern, so the binding does not depend on which loop NumPy runs |

### Results and memory

- **A kernel never writes into a caller's buffer unless the operation is defined that way.** `clip` with `out=` and masked assignment are. Scratch kernels are not: [judge-design.md](judge-design.md#scratch-output-writing-inside-the-capacity) requires that a scratch kernel only ever sees a buffer the binding owns.
- **Scratch output.** Allocate `n` elements, run the kernel, copy the reported elements into an array of exactly that size, free the first. This is the faster of the two ways tried. With few elements selected the large buffer barely shows: 9.8x end to end against the judge's 11.0x for the kernel alone.
- **Result types match NumPy's.** `np.count_nonzero` returns a Python `int`, `np.sum` on `uint8` a `np.uint64`, on a boolean array a `np.intp` and on float32 a `np.float32`, `np.histogram` a pair of counts and edges. Edges are produced by calling the NumPy function that NumPy itself uses, so they are identical by construction.
- **`bincount_u8` always counts into 256 entries.** NumPy returns one entry more than the largest value present. The binding cuts the result to that length.

### Threads

- **The inner-loop hook inherits NumPy's behaviour.** The kernel runs where NumPy's loop ran, with the lock held or released as NumPy decided.
- **The type-slot hook takes the lock.** NumPy releases it around `argmax`, and the kernels are not safe against a writer; see the table above.
- **The other hooks release the interpreter lock above a size threshold**, as NumPy does. A kernel is a pure leaf function and touches no Python object.
- **Masked assignment keeps the lock**, in all three spellings: `x[mask] = v`, `np.putmask` and `np.copyto(..., where=mask)`. The ARM64 kernel rewrites elements the mask does not select with their own bits ([kernel-findings.md](kernel-findings.md#masked-assignment)). Holding the lock means no other Python thread can write one of them in between. It blocks other threads for about 180 µs per million elements.
- **Free-threaded Python is not supported.** `install()` does nothing there. Changing a shared function pointer while other threads run needs its own design.

### Counting

Each hook adds one to a per-kernel counter when it takes a kernel and to a per-operation counter when it passes the call on. `report()` reads them. The type slot of `argmax` and `argmin` is called once per row when NumPy splits a call with `axis=`; while the method hook waits for such a call, the slot records what it did in a mark kept per thread, and the method hook counts the whole call once, under the kernel the slot took or as left to NumPy ([argmax-rows-design.md](argmax-rows-design.md#counting)). Inner loops run with the lock released, so the counters are atomic objects; see [What changed in the building](#what-changed-in-the-building) for how they are updated.

### Installing and uninstalling

- **Order.** Hooks go in one at a time, each recording the pointer it replaced in a table.
- **Each hook proves itself.** After applying a hook, `install()` makes one small call through NumPy's public interface with a known answer. If the kernel's counter did not move, or the answer is wrong, that hook is removed again and `report()` says so. This is the defence against a future NumPy that routes a call differently: the kernel is quietly not installed, and nothing breaks.
- **`uninstall()` restores in reverse order**, and only a pointer that is still ours. If something else has patched the same place since, it is left alone and reported.
- **Supported environment.** CPython with the interpreter lock and NumPy 2.3 or a later NumPy 2, on a supported platform. There are two: macOS on ARM64, and Linux on x86-64 with a processor of the x86-64-v3 level ([x86-64-design.md](x86-64-design.md#target)). On Linux `install()` reads `/proc/cpuinfo` and needs `avx`, `avx2`, `fma`, `bmi1`, `bmi2`, `abm` (LZCNT), `movbe` and `popcnt`; Linux lists the first two only when it also saves the 256-bit registers, which is the other half of the requirement. Anywhere else, and on an x86-64 processor without them, `install()` is a no-op and `report()` names what is missing.
- **On Linux x86-64** the same hooks serve the same calls with the x86-64 kernels: `uv run delivery check` passes there (96,734 cases, 63,630 taken by a kernel, none failed, on the Intel Core i5-14600KF with NumPy 2.5.5), and the timings from Python are in [delivery/results/intel-r-core-tm-i5-14600kf.md](../delivery/results/intel-r-core-tm-i5-14600kf.md). No hook, condition or binding constraint was changed for it. Two things the binding screens for because of the ARM64 kernels are probably not needed by the x86-64 ones (in-place `clip`, which the binding now serves through a buffer on both platforms, [clip-in-place-design.md](clip-in-place-design.md), and mask bytes other than 0 and 1); the screens are left in force on both platforms until a person decides otherwise ([kernel-findings.md](kernel-findings.md#what-a-caller-must-know)).

### Which NumPy versions

This was open decision 5. The recommendation was to rely on the self-test at install rather than pin to tested versions, and to run `delivery check` against the oldest and newest NumPy 2 releases ([issue 18](https://github.com/sofeikov/kernel-arena/issues/18)). Doing so showed that the self-test alone is not enough, and the package now claims NumPy 2.3 and later.

**What was found.** `delivery check` on an Apple M1 Pro, CPython 3.12.7, with the extension built against NumPy 2.5.5 and each release laid over it at run time:

| NumPy | Self-test at install | `delivery check` |
|---|---|---|
| 2.0.2 | Every hook kept, 21 of 23 kernels installed | 2,462 of 77,481 calls differ: 708 `np.clip`, 1,754 `np.count_nonzero` |
| 2.1.0, 2.2.0 | The same | 1,754 differ, all `np.count_nonzero` |
| 2.3.0 | The same | Pass, also on CPython 3.13.5 |
| 2.5.5, the newest | The same | Pass, also on CPython 3.14.6 |

- **`np.clip`.** NumPy 2.0 makes every zero it returns +0.0. From 2.1 a zero keeps the sign of the element or bound it came from, and so does the kernel: `np.clip(-0.0, 0.0, 1.0)` is +0.0 on 2.0 and -0.0 from 2.1.
- **`np.count_nonzero`.** Before 2.3 the call without `axis` returns a Python `int`. From 2.3 it returns `np.intp`, and so does the binding, which was made to match NumPy 2.5.5 while it was being built ([What changed in the building](#what-changed-in-the-building)).
- **The self-test passed on all of them.** A probe checks that its kernel ran and that the answer has the right value. It does not check the result's type, or the sign of a zero. On NumPy 2.0 to 2.2 every hook therefore stayed in place and returned what a later NumPy would, which is not what the NumPy in use returns. The self-test guards against a NumPy that routes a call elsewhere; it does not guard against one whose answer differs in a detail. That is what this check is for.
- **A build against the newest NumPy runs on the oldest.** The build takes NumPy from the build requirements, which resolve to the newest release for the Python being built for: 2.5.5 here, as for the wheels. NumPy's headers target an old C API by default (1.25 in the 2.5 headers), and the binding uses nothing newer, so the extension imports and every hook applies on 2.0.2. The differences above are NumPy's behaviour, not the build.
- **A hook that fails its self-test fails `delivery check`.** No release from 2.0 to 2.5.5 has a hook that does not fit, so this was shown by making one fail: with the probe of `loop.clip.f32` forced to fail, `install()` removed the hook, and `delivery check` failed 4,186 `np.clip` cases with "the kernel did not take the call", because each case states which kernel should take it.

**What was decided.**

- **The floor is NumPy 2.3**, the oldest release whose answers the kernels reproduce, in the package's dependencies and in its build requirements. Making the binding answer as the NumPy in use does would mean a version test in each hook for differences the self-test cannot see; raising the floor was preferred to that. NumPy 2.3 was released in June 2025.
- **`install()` refuses an older NumPy 2**, as it refuses NumPy 1, and `report()` says why. A dependency floor is not enough on its own: installing an older NumPy afterwards replaces it, and pip only reports the conflict.
- **No pin above.** A new NumPy is served without a release of hotkernels, guarded by the self-test, as recommended.
- **CI runs `delivery check` at both ends** on the macOS ARM64 runner, in the `numpy-range` job of [`ci.yml`](../.github/workflows/ci.yml): exactly the floor read from `hotkernels/pyproject.toml`, on CPython 3.13, the newest Python NumPy 2.3.0 has wheels for; and the newest NumPy on PyPI, relocked at run time, on CPython 3.14. The `judge` job still checks the NumPy in `uv.lock`. A NumPy release that changes an answer fails the newest step on the next pull request, without anyone updating the lock.

### Package and build

```
hotkernels/                  the installable package, its own distribution
  pyproject.toml
  meson.build
  tools/generate.py          build time: C prototypes and the kernel table, from the specs
  src/hotkernels/
    __init__.py              install, uninstall, enabled, report
    environment.py           whether this interpreter is supported
    hooks.py                 every hook, its kernels and its self-test
    report.py                the report's data and its rendering
    templates/               the Jinja template for the report
    _binding/                the C extension
      binding.h              counters and the shared checks
      module.c               hook table, counters, the interface to Python
      hook_slot.c            type slot: argmax, argmin
      hook_loop.c            inner loop: clip
      hook_function.c        function objects: argmax, count_nonzero, sum, histogram, bincount, isin,
                             interp, compress, extract, putmask, copyto, where, clip, unique
      hook_method.c          methods: astype, sum, nonzero, cumsum, searchsorted, argmax, argmin, clip,
                             any, round, and the ufunc type's accumulate and reduceat
      hook_index.c           indexing: x[mask], x[mask] = v
delivery/                    development tool: `uv run delivery check | demo | time | numpy-tests`
examples/workload.py         a NumPy program written without hotkernels in mind
```

- **`kernels/` stays the only source.** The build takes `kernels/<name>/<arch>/kernel.s` for the architecture being built where it exists, and `naive.c` where it does not, compiled with the flags the judge used. The C prototypes come from the specs, through the code the judge already has for it.
- **Two distributions.** `hotkernels` is what a user installs. The judge, the survey and the new `delivery` tool stay in the development distribution. The repository becomes a `uv` workspace holding both.
- **The package depends on NumPy and Jinja.** Jinja is there for the report, by the repository's rule on generated text, and is imported only when a report is rendered.
- **Installing needs a compiler for now.** Wheels are under "Later" in the core design, and so is `python -m hotkernels script.py`.

### Testing the binding

The judge proves a kernel right as a native function. It says nothing about the code between NumPy and the kernel, and that code decides, for every call, whether a kernel's conditions hold. A wrong check there is a wrong result in someone's program. It gets its own test, `uv run delivery check`:

- **Differential.** Every case is a Python call. It is run on plain NumPy and again with hotkernels installed, and the two must agree in result type, element type, shape and bytes (or the kernel's tolerance), in the exception raised, in the warnings issued, and in leaving the inputs unchanged.
- **Cases are a cross product**, per entry point: element types (the kernel's and its neighbours), shapes (0-d, empty, one element, 1-D, 2-D), layouts (contiguous, strided, reversed, misaligned, read-only, Fortran order, other byte order), array kinds (`ndarray`, a subclass, a masked array, a memory map, a list), every spelling of the call (positional, keyword, `out=`, `axis=`, method and function), and values from the judge's value classes.
- **Each case states whether a kernel should run**, and the counters are checked against it. A check that silently stops matching would otherwise pass every comparison while delivering nothing.
- **NumPy's own test suite**, run once on plain NumPy and once with the exact kernels installed, must give the same outcome for every test. This is the broadest evidence available for "never break anything", and it costs nothing to write.

`uv run delivery time` measures each entry point from Python, with and without hotkernels, at 16, 4,096 and 1,048,576 elements, for calls that take the kernel and calls that fall back.

What this does not cover: the binding runs without guard pages. A wrong length passed to a correct kernel is caught only if it changes a result.

## What the kernels need first

Binding the kernels to real calls shows where a kernel is narrower than the NumPy call it stands for. These are kernel or judge changes and go through their own pull requests.

| Kernel | Limit | Consequence | Proposed |
|---|---|---|---|
| `searchsorted_f64`, `interp_f64` | One `n` was the length of the sorted array and of the needles | Only a call where the two lengths happened to be equal could use the kernel | Done: the number of needles is its own argument ([two-sizes-design.md](two-sizes-design.md)), and both are bound |
| `clip_f32`, `clip_f64` | Input and output must not overlap: in place they are wrong on some data ([issue 7](https://github.com/sofeikov/kernel-arena/issues/7)) | `np.clip(x, 0, 1, out=x)` is clipped through a buffer of the binding's, one copy of each 4 KB run more than a clip into a new array | Done: the binding uses a temporary ([clip-in-place-design.md](clip-in-place-design.md)). Hardening the kernels would save the copy |
| `isin_i64` | Exactly eight test values | Fewer than eight can be padded by repeating one, which does not change membership. More than eight go to NumPy | None: the spec has value classes padded the same way, so the judge checks what the binding sends ([issue 22](https://github.com/sofeikov/kernel-arena/issues/22)) |
| `histogram_f64` | A range must be given | For `range=None` the binding calls NumPy's `min` and `max` first, as NumPy does | Nothing |
| Every operation | One or two element types each | Other types fall back | More kernels; "Later" in the core design |

## Order of work

| Step | What | Why here |
|---|---|---|
| 1 | The package, the build, `install()`, `uninstall()`, `enabled()`, `report()`, counters, the self-test at install. `clip` through the inner loop and `argmax`/`argmin` through the type slot. `delivery check` and `delivery time` | Proves the path with the two hooks that need no checks on the Python side, and builds the tests everything else is held to |
| 2 | Function hook: `count_nonzero`, `sum`, `histogram`, `bincount`, `isin`, and the small-array path for `argmax` and `argmin` | The first hook into an object NumPy does not offer for it, on calls where a mistake is easiest to see |
| 3 | Method hook: `astype`, `sum`, `nonzero`, `cumsum` | Reaches methods; includes the two older calling conventions that were not probed |
| 4 | Indexing hook: `x[mask]`, `x[mask] = v` | Last, because it touches every index operation in the process. It can be left out without affecting the rest |
| 5 | The exit measurements; README updated | |

In parallel, as kernel work: the second size for `searchsorted_f64` and `interp_f64`, then their hooks (done since); in-place `clip`. Also still open from [kernel-findings.md](kernel-findings.md#what-is-next): the locked re-run of every kernel and the leaderboard.

## Exit criterion

The core design's, made measurable:

1. **A demonstration script**, written against plain NumPy with the spellings people use (`x[mask]`, methods, functions), runs faster with the one line added, and `report()` shows every installed kernel being hit.
2. **`uv run delivery check` passes** for every entry point.
3. **NumPy's test suite gives the same outcome** with the exact kernels installed as without.
4. **A call that falls back costs at most 2 ns more than on plain NumPy**, for every hook, as measured by `delivery time`.
5. **No installed entry point is slower than NumPy at any size measured.** Where the binding loses at small sizes, the entry point gets a minimum size and smaller calls go to NumPy.
6. **Every kernel has a recorded verdict:** installed, installed only with `inexact=True`, or not bound, with the reason.

## Out of scope

- Wheels, PyPI and `python -m hotkernels`. "Later" in the core design.
- Free-threaded Python and sub-interpreters.
- x86-64 and Linux, which are designed in [x86-64-design.md](x86-64-design.md).
- Any array that is not a plain `ndarray`.
- New kernels and new element types.

## Decisions made in review

| Question | Decision |
|---|---|
| Change NumPy's own objects in place (function objects, `ndarray`'s method table)? Only the inner-loop hook is an interface NumPy offers for this, and it reaches `clip` alone. With the type slot as well, 3 of the 15 operations are reached | Yes |
| Hook indexing? It is the only way to `x[mask]` and `x[mask] = v`, and adds about 1 ns to every index operation on every array | Yes |
| Are kernels that are not bit-identical installed by default? The float `cumsum` kernels are not monotonic for non-negative input ([issue 10](https://github.com/sofeikov/kernel-arena/issues/10)), which breaks `np.searchsorted(np.cumsum(p), r)` | No. `install()` installs exact kernels only; `install(inexact=True)` adds the rest. This replaces the core design's decision to install them and mark them in `report()` |

## Decisions still open

Each has a recommendation. The implementation follows the recommendations until they are decided otherwise.

| # | Question | Recommendation |
|---|---|---|
| 1 | **Masked assignment rewrites unselected elements** (open question 1 in the findings; about the ARM64 kernel) | Yes, with the interpreter lock held during the kernel |
| 2 | **`clip` reads and writes the floating-point status register** (open question 2 in the findings; about the ARM64 kernels) | Yes, as the findings recommend, with the flags cleared afterwards |
| 3 | **Remove NumPy's Python wrapper for small arrays** where a deeper hook already delivers the kernel (`np.argmax` at 16 elements: 573 ns to 65 ns)? It means a second hook and a second set of checks for the same operation | Yes for `argmax` and `argmin` in step 2. `clip` after the exit criterion: its wrapper has more arguments to get right. Done for both, and for `x.argmax()` on short arrays, in [issue 17](https://github.com/sofeikov/kernel-arena/issues/17); see [Calling the kernel above NumPy's wrapper](#calling-the-kernel-above-numpys-wrapper) |
| 4 | **Build system for the package** | `meson-python`: it compiles assembly sources without special handling and is what NumPy uses. Not yet tried in this repository |

Decision 5, which NumPy versions, is made: see [Which NumPy versions](#which-numpy-versions).
