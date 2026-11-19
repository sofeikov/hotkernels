There is very little to do. Install the package, call `install()` once, and NumPy uses the kernels for the rest of the process.

## Requirements

| | |
|---|---|
| Machine | A Mac with Apple Silicon (ARM64), or Linux on x86-64 with a processor of the x86-64-v3 level (AVX2, FMA, BMI2: Intel since 2013, AMD since 2015) |
| Python | 3.11 or later, the standard build with the interpreter lock |
| NumPy | 2.3 or a later NumPy 2 |

Wheels are published for CPython 3.11 to 3.14. With another Python the package is built from source when it is installed, which takes a few seconds and needs the Xcode command line tools (`xcode-select --install`) on a Mac, or a C compiler and GNU binutils on Linux.

The figures on this site were measured on an Apple M1 Pro. On x86-64 the same calls are served by a second set of kernels, written separately, and NumPy is faster to begin with for some of them, so the gains are not the same.

On any other machine the package still installs, with no compiler, and `install()` does nothing, so code that calls it stays portable.

## Install

```bash
pip install hotkernels
```

or `uv pip install hotkernels`.

To work on the kernels themselves, clone the repository and run `uv sync`; the judge and the other development tools come with it.

## Without changing your code

`python -m hotkernels` runs a script with hotkernels installed, as `python` would run it:

```bash
python -m hotkernels my_script.py --its --own --arguments
python -m hotkernels -m pytest            # a module, as python -m runs it
python -m hotkernels --report my_script.py  # print report() to standard error at the end
```

`--inexact` installs the kernels that are not bit-identical to NumPy as well.

## Turn it on

```python
import numpy as np
import hotkernels

hotkernels.install()            # NumPy now uses the kernels, in the whole process

x = np.random.standard_normal(1_000_000)
x[x > 0]                        # served by a kernel
np.clip(x, 0, 1)                # served by a kernel
x[::2].sum()                    # not something a kernel handles: NumPy does it, as before

hotkernels.uninstall()          # NumPy is exactly as it was
```

That is the whole interface:

| Call | What it does |
|---|---|
| `hotkernels.install()` | Points NumPy at every kernel that returns NumPy's result bit for bit |
| `hotkernels.install(inexact=True)` | Also installs the two that do not: `np.cumsum` on float32 and float64 |
| `hotkernels.uninstall()` | Puts every pointer back |
| `with hotkernels.enabled():` | Installs for the duration of a block |
| `hotkernels.report()` | What is installed, and how often each kernel ran |

Importing the package changes nothing; only `install()` does. Calling it twice is harmless, and so is `uninstall()` without `install()`.

::: note It covers the whole process
`install()` changes NumPy itself, so every thread and every library that calls NumPy gets the kernels, not only your own code. That is the point: pandas, scikit-learn or your own helpers speed up without knowing. It also means `with hotkernels.enabled():` is a way to limit hotkernels in time, not to one thread.
:::

## See what it did

`print(hotkernels.report())` after your code has run:

```text
hotkernels 0.2.0: installed, 30 of 32 kernels (NumPy 2.5.5)

NumPy call         Elements Kernel    Result            Kernel calls  Left to NumPy
np.argmax          float32  assembly  exact                        0              0
np.argmax          float64  assembly  exact                        1
np.clip            float64  assembly  exact                        1
x[mask]            float64  assembly  exact                        1
x[mask] = v        float64  assembly  exact                        1              0
np.cumsum          int64    assembly  exact                        1              0
np.cumsum          float64  assembly  within tolerance not installed: needs install(inexact=True)
x.astype(float32)  uint8    C         exact                        1              3
np.sum             uint8    assembly  exact                        1              0
...
```

**Left to NumPy** is the column to read when nothing got faster. It counts calls that reached a hook and did not meet a kernel's conditions: another element type, a strided array, an argument the kernel does not implement. A large number there tells you what your arrays look like.

## What gets faster

A kernel takes a call only when the array is a plain, contiguous `ndarray` of the kernel's element type and every other argument is one the kernel implements. Any other call goes to NumPy as it was written.

| You write | Element types | A kernel takes it when |
|---|---|---|
| `np.argmax(x)`, `np.argmin(x)`, `x.argmax()`, `x.argmin()` | float32, float64 | For these types, including strided input and `axis=` (one kernel call per row or column). A strided run or a row shorter than 128 elements stays with NumPy, which is as fast there |
| `np.clip(x, lo, hi)`, `x.clip(lo, hi)` | float32, float64; int64 on arrays of up to 4,096 elements | Scalar bounds with `lo <= hi`; the output is not the input. Small arrays gain most with bounds that are Python numbers and no `out`. For int64, both bounds are Python ints or `np.int64` and there is no `out`: see below |
| `x[mask]` | float32, float64, int64 | The mask has the shape of `x`, of any number of dimensions. A mask over the rows of a two-dimensional `x` stays with NumPy |
| `np.compress(mask, x)`, `np.extract(mask, x)` | float32, float64, int64 | `x` and `mask` the same size, `x` of any shape. For `np.compress` the mask is one-dimensional and there is no `axis` or `out` |
| `x[mask] = v` | float64 | As for `x[mask]`, and `v` is a Python float or int |
| `np.putmask(x, mask, v)`, `np.copyto(x, v, where=mask)` | float64 | `v` is a Python float or int, and `x` and `mask` are the same size (`np.putmask`) or the same shape (`np.copyto`, with no `casting`), of any shape |
| `np.nonzero(mask)`, `mask.nonzero()`, `np.where(mask)` | bool | One-dimensional; `np.where` with nothing but the mask |
| `np.where(mask, a, b)` | float64, int64 | `mask` is a boolean array and `a` and `b` are arrays of its exact shape and one element type, of any number of dimensions; or, for float64, one of `a` and `b` is such an array and the other a single number: a Python float, a Python int up to 2^53, or an `np.float64`. Shapes that would broadcast, and mixed element types, stay with NumPy |
| `np.cumsum(x)`, `x.cumsum()` | int64, bool; float32 and float64 with `inexact=True` | No `axis`, `dtype` or `out` |
| `x.astype(np.float32)` | uint8 | Other arguments at their defaults |
| `np.round(x, decimals)`, `np.around(x, decimals)`, `x.round(decimals)` | float64, on arrays of up to 4,096 elements | `decimals` left out or a Python int from 0 to 15; no `out`. Values whose rounding overflows, and signalling NaNs, go to NumPy, which warns: see below |
| `np.maximum.accumulate(x)` | int64 | `x` is one-dimensional, with at least one element; no `dtype` or `out`, and `axis` left out, 0 or -1 |
| `np.fmax.accumulate(x)` | int64, float64 | As for `np.maximum.accumulate`, and a float64 `x` holds no signalling NaN |
| `np.fmax.reduceat(x, starts)` | float64 | `x` is one-dimensional, with at least one element; `starts` is a one-dimensional int64 array, not a list, with at least one start and every start inside `x`; no `dtype` or `out`, and `axis` left out, 0 or -1; `x` holds no signalling NaN. Fastest on short runs: see below |
| `m.any(axis=1)`, `np.any(m, axis=1)` | bool | `m` is two-dimensional, in C order, with at least one column; `axis` is the number 1 or -1, by position or by name, and no other argument is given |
| `np.median(x)` | float64, on arrays of up to 128 elements | The array alone, of any shape, with no other argument; no NaN. A zero median beside a negative zero, and a mean of the two middle elements that overflows, goes to NumPy: see below |
| `np.sum(x)`, `x.sum()` | uint8, bool | No other arguments |
| `np.count_nonzero(x)` | int64, float64 | No `axis` |
| `np.histogram(x, bins, range)` | float64 | An integer bin count; no weights or density |
| `np.bincount(x)`, `np.bincount(x, minlength=m)` | uint8, int16, int64 | `x` is one-dimensional; no weights; `minlength`, if given, is a Python int. For int16 and int64, `x` has at least one element, none negative, and the result is no longer than 65,536 counts or than `x` |
| `np.isin(x, values)` | int64 | At most eight test values; flags at their defaults |
| `np.searchsorted(a, v)`, `a.searchsorted(v)`, `np.digitize(x, bins)` | float64 | `a` is one-dimensional; `v` is an array of any shape or a Python number; either `side`; no `sorter`. For `np.digitize`, with or without `right=True`, when the bins increase |
| `np.interp(x, xp, fp)` | float64 | `xp` and `fp` are one-dimensional and the same length; no `left`, `right` or `period`; `xp` holds no NaN; at most 64 sample points per query |
| `np.hypot(a, b)` | float64, on x86-64 Linux | `a`, `b` and the output are contiguous where NumPy's loop meets them, of any shape, and the output is a new array, one clear of both inputs, or `a` or `b` itself (`out=a`); and the C library computes `hypot` as the kernel does, which `install()` checks: see below. A single number as either input, strided arrays, float32, and data whose result overflows or underflows stay with NumPy |

"Contiguous" means any shape for the operations that work element by element or reduce the whole array: a contiguous 40 by 25 array reaches the `clip` kernel as one run of 1,000 elements.

How much faster each one is, from Python and at three sizes, is in [the results]([[ root ]]report/results/).

## Things worth knowing

- **Float `cumsum` is opt-in for a reason.** The fast kernels add in a different order from NumPy. The result is as accurate, but not bit-identical, and the output of a non-negative input is not guaranteed to be non-decreasing. Code such as `np.searchsorted(np.cumsum(p), r)` relies on that. Use `install(inexact=True)` only when it does not matter to you. [Why it cannot be both exact and fast]([[ root ]]report/arithmetic/#cumsum-the-chain-you-cannot-break).
- **`x[mask] = v` rewrites elements it does not change**, and so do `np.putmask` and `np.copyto(x, v, where=mask)`. On a dense mask the kernel stores unselected elements back with their own bits. The result is identical; the array must be writable throughout, which a normal array is.
- **A mask holding a byte other than 0 or 1 goes to NumPy.** NumPy treats any non-zero byte as True, and a boolean view of other data can hold a 2. Every mask is read once before a kernel sees it, in the indexing syntax, in the function forms, in `mask.sum()`, in `np.cumsum(mask)`, in `np.where(mask, a, b)` and in `m.any(axis=1)` alike.
- **The mask functions are counted under the indexing syntax.** In `report()`, `np.compress` and `np.extract` count on the `x[mask]` rows, `np.putmask` and `np.copyto` on `x[mask] = v`, and `np.where(mask)` on `np.nonzero`. `np.copyto` without `where` is another operation and is not counted at all. `np.where(mask, a, b)` has a row of its own, `np.where(c, a, b)`, whose element type reads `float64`: it counts int64 calls too, for the same reason as `x[mask]` below. `mask.sum()` counts on the `np.sum` row, whose element type reads `uint8`: one kernel adds the bytes of both. In the same way `x[mask]` on int64 counts on the float64 row of `x[mask]`: that kernel moves 64-bit elements without computing on them, so it serves both types.
- **`np.maximum.accumulate` on floats stays with NumPy.** It carries a NaN forward from where it appears, which no kernel does. `np.fmax.accumulate`, the running maximum that skips NaN, is served on float64 and gives NumPy's result bit for bit, the sign of a zero and the payload of a leading NaN included.
- **A float64 array holding a signalling NaN goes to NumPy in `np.fmax.accumulate`.** NumPy turns one into a quiet NaN and forgets the maximum so far; the kernel would skip it. Arithmetic never produces a signalling NaN, so you only meet one in data built bit by bit, but every float64 array is read for them before the kernel sees it.
- **`np.fmax.reduceat` gains most on short runs.** On runs of 16 it is several times faster than NumPy. On runs of several hundred elements it is about as fast as NumPy, and on long runs of an array too large for the processor's second-level cache it can be slower (0.8x at runs of 1,024 in a million elements): every element is read once more, for signalling NaNs. Give `starts` as an int64 array; a list or int32 starts go to NumPy.
- **`np.fmax.reduceat` leaves two kinds of answer to NumPy.** When `x` holds a negative zero and the maximum of some run is a zero, or `x` holds NaNs of different bit patterns and some run is all NaN, the whole call goes to NumPy. Which zero or which NaN NumPy returns there depends on the vector instructions it picked when it started, which differ between processors, and hotkernels does not guess. Values from ordinary arithmetic have one NaN pattern, and a zero maximum among negative values only matters if a negative zero is present.
- **Only `m.any(axis=1)` is served, not its neighbours.** `m.all(axis=1)`, `m.any(axis=0)`, `m.any()` and `np.logical_or.reduce(m, axis=1)` go to NumPy, and so does `m.any(axis=1)` with `keepdims`, `out` or `where`, or with the axis given as a NumPy integer. A matrix of 22 columns is about 40 times faster than NumPy at a million elements; rows of four columns gain most and rows of one, two or eight least, though all are well ahead of NumPy.
- **`np.bincount` on int16 and int64 reads the array twice.** NumPy's result is one count longer than the largest element, so the array is read once for its smallest and largest elements before it is counted. An array holding a negative element goes to NumPy, which raises its `ValueError` as before. So does a result much longer than the array, such as `np.bincount(np.array([10**9]))`: NumPy gets that memory from the system already zeroed, and the kernel would write every count. The gain is largest on int16 (about 4.5x at a million elements) and smaller on int64 (about 1.9x), whose array is four times the size to read twice. Other integer types, int32 and uint16 among them, and `weights` stay with NumPy.
- **`np.clip(x, 0, 1)` is faster by less than `np.clip(x, -1, 1)`.** A lower bound of exactly zero needs extra care for negative zeros. It is still several times faster than NumPy.
- **A small `np.clip` is fast when its bounds are plain numbers.** `np.clip(x, lo, hi)` and `x.clip(lo, hi)` skip NumPy's own call path, most of a call on a small array, when the bounds are Python floats or ints, or NumPy scalars of the array's own type, and there is no `out`, `dtype`, `casting` or `where`. The result has the element type NumPy gives it: a float32 array clipped to Python floats stays float32. Other calls, such as `np.float64` bounds on a float32 array (NumPy returns float64), `out=`, or a bound of `None`, go through NumPy's call path first; the kernel still serves them where it served them before, and they gain only on larger arrays.
- **`np.clip` on int64, `np.round` and `np.median` are served on short arrays only.** On a short array most of a NumPy call is NumPy's own call path, about 1.8 µs for `np.clip`, 1.1 µs for `np.round(x, 2)` and 4.8 µs for `np.median`, and hotkernels answers it with a plain loop of its own in about 0.1 µs. On longer arrays NumPy's own loops are as good or better, and the call goes to NumPy as before: above 4,096 elements for `np.clip` and `np.round`, above 128 for `np.median`, whose loop sorts a copy of the array. In `report()` each has a row of its own, marked `C`, which the count of kernels on the first line leaves out: they are part of hotkernels, not judged kernels.
- **`np.clip` on int64 needs integer bounds.** Python ints that fit int64, or `np.int64`, with `lo <= hi`, and no `out`. A float bound gives NumPy's float64 result, and is NumPy's.
- **`np.round` gives NumPy's result bit for bit**, because it does what NumPy does, in the same order: multiply by the power of ten, round to the nearest whole number with ties to even, divide. So `np.round(-0.004, 2)` is `-0.0` and `np.round(0.125, 2)` is `0.12`, as with NumPy. Negative `decimals`, more than 15, and `decimals` given as a NumPy integer go to NumPy. So does a call where the multiplication overflows, or the array holds a signalling NaN: NumPy computes it and warns, or raises under `np.errstate`, as it always did.
- **`np.median` gives NumPy's result bit for bit**: the middle element, or the two middle elements added and halved, as NumPy's mean of them does. Some calls go to NumPy, which then answers and warns as it always did: an array holding a NaN; two middle elements whose sum overflows, or that are infinities of both signs; a half that is too small to be exact; and a median of zero when the array holds a negative zero anywhere, since which zero NumPy returns then depends on how its partition orders equal elements (NumPy 2.5 returns `+0.0`). `np.median(x, axis=...)` and every other argument, `out` and `overwrite_input` included, are NumPy's.
- **`np.hypot` is served only where your C library computes it the way the kernel does.** NumPy calls the C library's `hypot` for every element, and C libraries do not all round it the same way: they differ in the last bit for a fraction of a percent to about ten percent of results. The kernel computes the algorithm of glibc 2.35 and later, so on such a system it gives NumPy's result bit for bit, NaNs and infinities included. `install()` checks this on your machine, comparing the kernel with the C library on 20,000 pairs of numbers; where they differ in a single bit, `report()` lists the hook as not in place and `np.hypot` is NumPy's, as before. On ARM64 the call runs the C library's own `hypot`, with no gain.
- **`np.hypot` gains on arrays of hundreds of elements and more.** On a short array nearly all of a call is NumPy's own call path, which this hook sits beneath: 1.2x at 16 elements, about 4.7x at 4,096 and 6.8x at a million, measured on x86-64. A call whose data overflow or underflow, such as `np.hypot(1e308, 1e308)` among its elements, is computed again by NumPy, which warns or raises as `np.errstate` says, so such a call takes longer than before.
- **`np.isin` is fastest when the test values are close together**: all between 0 and 65,534, or all within 32,767 of the first one. Otherwise it runs at the speed of compiled C, which is still ahead of NumPy.
- **`np.searchsorted` and `np.interp` trust you that the array is sorted, as NumPy does.** Neither checks. On an array that is not sorted both return a meaningless result, and the kernel's is not the same meaningless result as NumPy's. Sorted data, including NaNs at the end of `a`, gives NumPy's answer bit for bit.
- **`np.digitize` is served when its bins increase.** It calls `np.searchsorted`, on either side. With decreasing bins it searches them reversed, as a view no kernel takes, and runs at NumPy's speed.
- **`np.interp` with a few queries against a long table stays with NumPy.** A NaN among the sample points is something the kernel cannot handle, so `xp` is read once before every call. With more than 64 sample points per query that pass would cost more than the kernel saves.
- **Free-threaded Python is not supported.** `install()` does nothing there, and `report()` says so.
- **NumPy 2.0 to 2.2 are not supported.** They answer a few calls differently from later releases (`np.clip` drops the sign of a zero on 2.0, and `np.count_nonzero(x)` gives a Python `int` before 2.3), and the kernels give the later answer. Installing hotkernels brings NumPy 2.3 or later; with an older NumPy in place `install()` does nothing, and `report()` says so.
- **Subclasses mostly stay with NumPy.** `x[mask]`, the methods and the hooked functions take a kernel only for a plain `ndarray`; masked arrays, memory maps and your own subclasses go to NumPy there. `clip` and `argmax` are also hooked beneath NumPy's own handling of array types, so for a subclass they serve whatever plain buffer NumPy hands them, and NumPy wraps the result as it always did.

## Checking it yourself

From a clone of the repository:

```bash
uv run delivery demo          # run examples/workload.py with and without hotkernels, and compare
uv run delivery check         # 170,973 NumPy calls, both ways; results, errors and warnings must agree
uv run delivery time          # time each statement from Python on your machine
uv run delivery numpy-tests   # NumPy's own test suite, both ways; about ten minutes
```

`delivery demo` is the quickest way to see the effect: an ordinary NumPy program on two million elements that runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.
