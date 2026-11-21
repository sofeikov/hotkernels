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
| `hotkernels.install(bottleneck=False)` | Leaves bottleneck's functions alone: see [What gets faster](#what-gets-faster) |
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
hotkernels 0.3.1: installed, 40 of 42 kernels (NumPy 2.5.5)

NumPy call         Elements Kernel    Result            Kernel calls  Left to NumPy
np.argmax          float32  assembly  exact                        0              0
np.argmax          float64  assembly  exact                        1
argmax(axis=1)     float64  C         exact                        1
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

Both columns count the calls you wrote. `np.argmax(m, axis=1)` is one call, however many rows NumPy splits it into, whether a kernel took its rows or NumPy did. The rows marked C are loops built into hotkernels rather than kernels; the first line does not count them among the kernels.

## What gets faster

A kernel takes a call only when the array is a plain, contiguous `ndarray` of the kernel's element type and every other argument is one the kernel implements. Any other call goes to NumPy as it was written.

| You write | Element types | A kernel takes it when |
|---|---|---|
| `np.argmax(x)`, `np.argmin(x)`, `x.argmax()`, `x.argmin()` | float32, float64 | For these types, including strided input and `axis=` (one kernel call per row or column). A strided run or a row shorter than 128 elements stays with NumPy, which is as fast there, except as in the next row |
| `np.argmax(m, axis=1)`, `m.argmax(axis=1)`, and `argmin` | float32, float64 | `m` is two-dimensional, in C order, with at least one column; `axis` is the number 1 or -1, by position or by name; no `out` and no `keepdims`. Rows of any length: narrow rows, such as the scores of a classifier with a few classes, gain most |
| `np.clip(x, lo, hi)`, `x.clip(lo, hi)` | float32, float64; int64 on arrays of up to 4,096 elements | Scalar bounds with `lo <= hi`; the output is a new array or the array itself (`np.clip(x, lo, hi, out=x)`), not another array that overlaps it. Small arrays gain most with bounds that are Python numbers and no `out` other than `x` itself. For int64, both bounds are Python ints or `np.int64` and there is no `out`: see below |
| `x[mask]` | float32, float64, int64 | The mask has the shape of `x`, of any number of dimensions. A mask over the rows of a two-dimensional `x` stays with NumPy |
| `np.compress(mask, x)`, `np.extract(mask, x)` | float32, float64, int64 | `x` and `mask` the same size, `x` of any shape. For `np.compress` the mask is one-dimensional and there is no `axis` or `out` |
| `x[mask] = v` | float64 | As for `x[mask]`, and `v` is a Python float or int |
| `np.putmask(x, mask, v)`, `np.copyto(x, v, where=mask)` | float64 | `v` is a Python float or int, and `x` and `mask` are the same size (`np.putmask`) or the same shape (`np.copyto`, with no `casting`), of any shape |
| `np.nonzero(mask)`, `mask.nonzero()`, `np.where(mask)` | bool | One-dimensional; `np.where` with nothing but the mask |
| `np.nonzero(m)`, `m.nonzero()`, `np.where(m)`, and `np.argwhere(m)`, which calls `np.nonzero` | bool; Apple Silicon only | `m` is two-dimensional and in C order, of any shape, empty ones included. Other layouts and three or more dimensions stay with NumPy |
| `np.where(mask, a, b)` | float64, int64 | `mask` is a boolean array and `a` and `b` are arrays of its exact shape and one element type, of any number of dimensions; or, for float64, one of `a` and `b` is such an array and the other a single number: a Python float, a Python int up to 2^53, or an `np.float64`. Shapes that would broadcast, and mixed element types, stay with NumPy |
| `np.cumsum(x)`, `x.cumsum()` | int64, bool; float32 and float64 with `inexact=True` | No `axis`, `dtype` or `out` |
| `x.astype(np.float32)` | uint8 | Other arguments at their defaults |
| `np.round(x, decimals)`, `np.around(x, decimals)`, `x.round(decimals)` | float64, on arrays of up to 4,096 elements | `decimals` left out or a Python int from 0 to 15; no `out`. Values whose rounding overflows, and signalling NaNs, go to NumPy, which warns: see below |
| `np.maximum.accumulate(x)` | int64 | `x` is one-dimensional, with at least one element; no `dtype` or `out`, and `axis` left out, 0 or -1 |
| `np.fmax.accumulate(x)` | int64, float64 | As for `np.maximum.accumulate`, and a float64 `x` holds no signalling NaN, no negative zero, and NaNs of one bit pattern only |
| `a.cumsum(axis=0)`, `np.cumsum(a, axis=0)`, `a.cumprod(axis=0)`, `np.cumprod(a, axis=0)`, `np.add.accumulate(a)`, `np.multiply.accumulate(a)`, `np.maximum.accumulate(a)`, `np.fmax.accumulate(a)` | float64 | `a` has two or more dimensions, is in C order, and has at least one row and one column; the axis is the first (the number 0 or `-a.ndim`, or left out for `.accumulate`); no `dtype`; `out` left out, or a C-ordered float64 array of `a`'s shape that is `a` itself or does not overlap it. For the maximum and `np.fmax`, `a` holds no signalling NaN, no negative zero, and NaNs of one bit pattern only: see below |
| `np.fmax.reduceat(x, starts)` | float64 | `x` is one-dimensional, with at least one element; `starts` is a one-dimensional int64 array, not a list, with at least one start and every start inside `x`; no `dtype` or `out`, and `axis` left out, 0 or -1; `x` holds no signalling NaN. Fastest on short runs: see below |
| `m.any(axis=1)`, `np.any(m, axis=1)` | bool | `m` is two-dimensional, in C order, with at least one column; `axis` is the number 1 or -1, by position or by name, and no other argument is given |
| `w -= b`, `w += b`, `np.subtract(w, b, out=w)`, `np.add(w, b, out=w)` | float64, on arrays of 131,072 elements (1 MB) or more; Apple Silicon only | The result goes into the first operand, and `w` and `b` are contiguous arrays of one shape. Gains only in one situation: see below |
| `np.median(x)`, `np.median(x, axis=None)` | float64 | The array alone, of any shape, with no other argument than `axis=None`; no NaN. A zero median beside a negative zero, and a mean of the two middle elements that overflows, goes to NumPy: see below |
| `np.argsort(x, kind="stable")`, `x.argsort(kind="stable")` | float64, int64, uint64 | `x` is one-dimensional with at least 256 elements, and need not be contiguous: a reversed view or one with a step is taken too; `kind` is `"stable"` or `"mergesort"`, or `stable=True` is given with no `kind`; no `order` or `descending`, and `axis` left out, 0 or -1. The default kind stays with NumPy: see below |
| `np.sum(x)`, `x.sum()` | uint8, bool; float32 and float64 on Apple Silicon | No other arguments. For the floats, `x` is one-dimensional and its sum is finite: see below |
| `np.count_nonzero(x)` | int64, float64 | No `axis` |
| `np.histogram(x, bins, range)` | float64 | An integer bin count; no weights or density |
| `np.bincount(x)`, `np.bincount(x, minlength=m)` | uint8, int16, int64 | `x` is one-dimensional; no weights; `minlength`, if given, is a Python int. For int16 and int64, `x` has at least one element, none negative, and the result is no longer than 65,536 counts or than `x` |
| `np.unique(x)` | int64 | At least one element, of any shape or layout; no `return_index`, `return_inverse` or `return_counts` (or each `False`), no `axis`, `sorted` left at `True`. The largest element minus the smallest is less than 256 or than the number of elements |
| `np.isin(x, values)` | int64 | At most eight test values; flags at their defaults |
| `np.searchsorted(a, v)`, `a.searchsorted(v)`, `np.digitize(x, bins)` | float64 | `a` is one-dimensional; `v` is an array of any shape or a Python number; either `side`; no `sorter`. For `np.digitize`, with or without `right=True`, when the bins increase |
| `np.interp(x, xp, fp)` | float64 | `xp` and `fp` are one-dimensional and the same length; no `left`, `right` or `period`; `xp` holds no NaN; at most 64 sample points per query |
| `bottleneck.nanmean(a, axis)`, `bottleneck.nanstd(a, axis, ddof)`, `bottleneck.nanmin(a, axis)`, `bottleneck.nanmax(a, axis)`, and pandas and empyrical calling them | float64 | `a` is two-dimensional, and the axis that is kept has its elements next to each other: `axis=0` of a C-ordered array or of a rolling-window view, `axis=1` of a Fortran-ordered one. `axis` and `ddof` are Python ints. The result is bottleneck's own, bit for bit. One-dimensional arrays and `axis=None` stay with bottleneck: see below |

"Contiguous" means any shape for the operations that work element by element or reduce the whole array: a contiguous 40 by 25 array reaches the `clip` kernel as one run of 1,000 elements.

How much faster each one is, from Python and at three sizes, is in [the results]([[ root ]]report/results/).

## Things worth knowing

- **Float `cumsum` is opt-in for a reason.** The fast kernels add in a different order from NumPy. The result is as accurate, but not bit-identical, and the output of a non-negative input is not guaranteed to be non-decreasing. Code such as `np.searchsorted(np.cumsum(p), r)` relies on that. Use `install(inexact=True)` only when it does not matter to you. [Why it cannot be both exact and fast]([[ root ]]report/arithmetic/#cumsum-the-chain-you-cannot-break).
- **`np.sum` on float32 and float64 gives NumPy's result bit for bit, on Apple Silicon.** NumPy adds the elements in a fixed order, in blocks of up to 128 whose sums it adds in pairs, and hotkernels makes exactly the same additions in the same order, only several blocks at a time. So, unlike float `cumsum`, it is installed by default. On an Apple M1 Pro `np.sum(x)` on float32 is about 20x faster than NumPy on 16 elements, 10x on 4,096 and 3x on a million, and on float64 about 20x, 6x and 1.9x. Only `np.sum(x)` and `x.sum()` with nothing else are taken: `axis=None`, `axis=0`, `dtype`, `out`, `keepdims`, `initial` and `where` go to NumPy, and so do a matrix, a strided array, and an array whose elements are not aligned to their size (4 or 8 bytes), which NumPy itself adds in another order. A sum that overflows or meets a NaN or an infinity is handed to NumPy, so that the result and the warning are NumPy's own. On x86-64 nothing changes yet: the call stays with NumPy until it has been measured there, and `report()` says so.
- **`x[mask] = v` rewrites elements it does not change**, and so do `np.putmask` and `np.copyto(x, v, where=mask)`. On a dense mask the kernel stores unselected elements back with their own bits. The result is identical; the array must be writable throughout, which a normal array is.
- **A mask holding a byte other than 0 or 1 goes to NumPy.** NumPy treats any non-zero byte as True, and a boolean view of other data can hold a 2. Every mask is read once before a kernel sees it, in the indexing syntax, in the function forms, in `mask.sum()`, in `np.cumsum(mask)`, in `np.where(mask, a, b)`, in `m.any(axis=1)` and in `np.nonzero(m)` alike.
- **The mask functions are counted under the indexing syntax.** In `report()`, `np.compress` and `np.extract` count on the `x[mask]` rows, `np.putmask` and `np.copyto` on `x[mask] = v`, and `np.where(mask)` on `np.nonzero`. `np.copyto` without `where` is another operation and is not counted at all. `np.where(mask, a, b)` has a row of its own, `np.where(c, a, b)`, whose element type reads `float64`: it counts int64 calls too, for the same reason as `x[mask]` below. `mask.sum()` counts on the `np.sum` row, whose element type reads `uint8`: one kernel adds the bytes of both. In the same way `x[mask]` on int64 counts on the float64 row of `x[mask]`: that kernel moves 64-bit elements without computing on them, so it serves both types.
- **`np.nonzero` on a boolean matrix returns what NumPy returns, views and all.** For a two-dimensional mask NumPy builds one array of (row, column) pairs and returns two views of its columns; hotkernels does the same, so the arrays you get have the same strides and the same base as NumPy's. A matrix of 40 rows of 22 with 5% True is about 5x faster than NumPy on an Apple M1 Pro, and a million elements 15x to 27x. It is served on Apple Silicon only so far; on x86-64 the call stays with NumPy. In `report()` it has a row of its own, `np.nonzero, matrix`.
- **`np.maximum.accumulate` on a one-dimensional float array stays with NumPy.** It carries a NaN forward from where it appears, which no kernel does. Down the columns of a float64 matrix it is served: see below. `np.fmax.accumulate`, the running maximum that skips NaN, is served on float64 and gives NumPy's result bit for bit, the payload of a leading NaN included.
- **A float64 array holding a signalling NaN goes to NumPy in `np.fmax.accumulate`.** NumPy turns one into a quiet NaN and forgets the maximum so far; the kernel would skip it. Arithmetic never produces a signalling NaN, so you only meet one in data built bit by bit, but every float64 array is read for them before the kernel sees it.
- **A float64 array holding a negative zero, or NaNs of different bit patterns, also goes to NumPy in `np.fmax.accumulate`.** There NumPy's own answer depends on how it was built: its x86-64 and ARM64 builds give a running maximum of `0.0` and `-0.0` different signs. You get whatever your NumPy gives. The same read finds them.
- **Accumulating down the columns of a matrix gives NumPy's result bit for bit.** `a.cumprod(axis=0)`, `a.cumsum(axis=0)`, `np.maximum.accumulate(a, axis=0)` and `np.fmax.accumulate(a, axis=0)` on a C-ordered float64 matrix compute each element exactly as NumPy does, the previous row's result and this row's element, but a row at a time instead of a column at a time: about 7x to 10x faster than NumPy at 253 × 4,789 on an Apple M1 Pro, and in place too (`a.cumprod(axis=0, out=a)`). It is plain C inside hotkernels, not a judged kernel, with rows of its own in `report()`. A sum or product that meets a NaN, overflows or underflows is finished by NumPy from the row where that happened, so the warnings, or the error under `np.errstate`, are NumPy's own. The maximum and `np.fmax` leave a matrix holding a signalling NaN, a negative zero or NaNs of different bit patterns to NumPy, as `np.fmax.accumulate` does on one dimension. Other axes, Fortran order, other element types and `dtype=` go to NumPy.
- **`np.fmax.reduceat` gains on short runs only.** On runs of 16 it is faster than NumPy: about 11x at a million elements on an Intel Core i5-14600KF with 5% NaN (2.2x without), and about 3x on an Apple M1 Pro. Longer runs lose, because every element is read once more, for signalling NaNs, before the maximum is taken. On x86-64 runs of several hundred elements are about as fast as NumPy, and long runs of an array too large for the processor's second-level cache can be slower (0.8x at runs of 1,024 in a million elements). On Apple Silicon runs of 64 are still faster (1.3x at a million elements), and runs of 256 or more are slower from about 262,144 elements (0.6x to 0.8x of NumPy); on 4,096 elements they are about level. If your runs are long, leave this call to NumPy with `uninstall()` or outside `enabled()`. Give `starts` as an int64 array; a list or int32 starts go to NumPy.
- **`np.fmax.reduceat` leaves two kinds of answer to NumPy.** When `x` holds a negative zero and the maximum of some run is a zero, or `x` holds NaNs of different bit patterns and some run is all NaN, the whole call goes to NumPy. Which zero or which NaN NumPy returns there depends on the vector instructions it picked when it started, which differ between processors, and hotkernels does not guess. Values from ordinary arithmetic have one NaN pattern, and a zero maximum among negative values only matters if a negative zero is present.
- **Only the stable `argsort` is served.** `np.argsort(x, kind="stable")`, `kind="mergesort"` and `stable=True` have one right answer, equal elements in their original order, and the kernel returns it: the same array as NumPy, NaNs last and zeros of both signs in the order they came. `np.argsort(x)` with no `kind` is not a stable sort. On an array with equal elements its answer depends on how NumPy's quicksort happens to partition, which no other sort reproduces, so it stays with NumPy. If your code does not depend on that order, asking for `kind="stable"` is the faster call with hotkernels installed: about ten times faster at a million elements than NumPy's default kind as well. `np.unique(x, return_index=True)` and scikit-learn's ranking metrics ask for the stable kind themselves.
- **A stable `argsort` borrows memory while it runs:** 32 bytes per element, freed before it returns, where NumPy's own stable sort borrows 4. A million elements borrow 32 MB. If that memory cannot be had the call goes to NumPy.
- **Sorted input is recognised.** An array already in order, or in strictly descending order, is answered without sorting. One that is nearly in order, with a handful of elements out of place or a few new ones appended to a sorted array, goes to NumPy, whose sort is built for exactly that and is faster there than the kernel.
- **Only `m.any(axis=1)` is served, not its neighbours.** `m.all(axis=1)`, `m.any(axis=0)`, `m.any()` and `np.logical_or.reduce(m, axis=1)` go to NumPy, and so does `m.any(axis=1)` with `keepdims`, `out` or `where`, or with the axis given as a NumPy integer. A matrix of 22 columns is about 40 times faster than NumPy at a million elements on an Intel Core i5-14600KF and about 24 times on an Apple M1 Pro; rows of four columns gain most. On x86-64 rows of one, two or eight columns gain least, though all are well ahead of NumPy. On Apple Silicon every width measured, from one column to 1,000, is faster than the compiled loop it replaces. From Python it is ahead of NumPy as well, with one exception: rows of 1,000 columns that hold a True are about 1.6 times slower than NumPy at a million elements, because every byte is checked before the kernel runs while NumPy stops at each row's first True. A matrix of one column is about 1.4 times faster than NumPy.
- **`np.argmax(m, axis=1)` gains most on narrow rows.** NumPy spends 6 to 20 ns on each row, whatever its length; hotkernels under 1 ns on a row of up to four elements and about 2 ns on a row of ten. On 300,000 rows of ten float64 that is about 3.5x faster than NumPy, and on rows of two to five 12x to 20x (Apple M1 Pro; not yet measured on x86-64). Rows of six float64 or nine float32 elements and more go to the kernel one row at a time, which holds Python's interpreter lock while it runs, where NumPy releases it. `axis=0`, arrays of three or more dimensions, and `keepdims=True` go to NumPy, and long columns there still reach the kernel.
- **`np.bincount` on int16 and int64 reads the array twice.** NumPy's result is one count longer than the largest element, so the array is read once for its smallest and largest elements before it is counted. An array holding a negative element goes to NumPy, which raises its `ValueError` as before. So does a result much longer than the array, such as `np.bincount(np.array([10**9]))`: NumPy gets that memory from the system already zeroed, and the kernel would write every count. The gain at a million elements is largest on int16 (about 9x on an Apple M1 Pro, 4.5x on an Intel Core i5-14600KF) and smaller on int64 (about 4x and 1.9x), whose array is four times the size to read twice. Other integer types, int32 and uint16 among them, and `weights` stay with NumPy.
- **`np.unique` is fast on labels, not on arbitrary integers.** Class labels, a few distinct values close together, are what scikit-learn validates on every call, and those are about 18x faster at a million elements on an Apple M1 Pro and about 39x on an Intel Core i5-14600KF. The array is read once for its smallest and largest elements first; when they are far apart, compared with the number of elements, the call goes to NumPy, which costs it that one read: well under 1% of what NumPy then takes. `np.unique` with counts, an inverse or indices, with `sorted=False`, or on any type other than int64, goes to NumPy as before.
- **`np.clip(x, 0, 1)` is faster by less than `np.clip(x, -1, 1)`.** A lower bound of exactly zero needs extra care for negative zeros. It is still several times faster than NumPy.
- **A small `np.clip` is fast when its bounds are plain numbers.** `np.clip(x, lo, hi)` and `x.clip(lo, hi)` skip NumPy's own call path, most of a call on a small array, when the bounds are Python floats or ints, or NumPy scalars of the array's own type, and there is no `dtype`, `casting` or `where`, and no `out` unless it is the array itself. The result has the element type NumPy gives it: a float32 array clipped to Python floats stays float32. Other calls, such as `np.float64` bounds on a float32 array (NumPy returns float64), `out=` another array, or a bound of `None`, go through NumPy's call path first; the kernel still serves them where it served them before, and they gain only on larger arrays.
- **`np.clip(x, lo, hi, out=x)` clips in place.** The gradient-clipping idiom, `np.clip(g, -5, 5, out=g)` or `g.clip(-5, 5, out=g)`, is served on float32 and float64 arrays under the same conditions as `np.clip(g, -5, 5)`, and returns `g` itself, as NumPy does. It costs one copy more than clipping into a new array: the kernels must not write over the data they are reading, so hotkernels copies 4 KB of the array at a time to a buffer of its own and clips the copy back into place. An `out` that is any other array goes through NumPy's call path first, and one that overlaps `x` without being `x` goes to NumPy. On int64 the in-place call is NumPy's.
- **`np.clip` on int64 and `np.round` are served on short arrays only.** On a short array most of a NumPy call is NumPy's own call path, about 1.8 µs for `np.clip`, 1.1 µs for `np.round(x, 2)` and 4.8 µs for `np.median`, and hotkernels answers it with a plain loop of its own in about 0.1 µs. On longer arrays NumPy's own loops are as good or better, and above 4,096 elements `np.clip` and `np.round` go to NumPy as before. In `report()` each has a row of its own, marked `C`, as `np.median` has, which the count of kernels on the first line leaves out: they are part of hotkernels, not judged kernels.
- **`np.median` is fast on large arrays too.** Above 64 elements hotkernels does not sort: it finds the middle elements from the bits of the numbers, reading the array three times, where NumPy partitions a copy of it. On an Apple M1 Pro that is about 8x faster than NumPy on a million normally distributed float64 (9.2 ms to 1.2 ms), about 7x on an astronomical image of 1,024 × 1,024, and 4x on 1,000 elements. On data already in order, or nearly, NumPy's own partition is quick, and the gain is smaller: 1.7x on a million sorted elements, 1.2x on a smooth curve, and on a sorted array with a little noise added hotkernels is slower, 0.74x at a million elements. It takes a buffer of up to eight bytes an element, usually a few per cent of that, where NumPy copies the whole array, and releases the interpreter lock from 8,192 elements. Astropy's robust statistics (`biweight_location`, `biweight_scale`, `median_absolute_deviation`) call `np.median(data, axis=None)` and are served; `np.nanmedian`, bottleneck's `nanmedian`, `np.percentile` and `np.quantile` are not.
- **`w -= b` and `w += b` on float64 gain after a large matrix product, on Apple Silicon only.** When Accelerate multiplies by a large matrix, it does so on several cores, and updating that matrix in place afterwards, as the gradient step `w -= lr * g` of a training loop does, takes NumPy up to 15 times as long as it would otherwise. hotkernels runs these updates with a loop of its own that asks for each part of the array ahead of time in a way that removes the delay: in a small training loop on an Apple M1 Pro the step went from about 460 µs to about 95 µs, and the whole program from 260 ms to 153 ms. Everywhere else it runs at about NumPy's speed, on an array already in the processor's cache or in a program without a large matrix product: from 0.99x to 1.07x of NumPy, measured on its own. Arrays under 131,072 elements go to NumPy, since there the loop was up to 2% slower and the delay was not seen. The result is NumPy's bit for bit. `w = w - b`, a scalar such as `w -= 0.5`, and float32 go to NumPy. On x86-64 nothing changes: whether the same delay arises there has not been measured. In `report()` the two rows, `x += y` and `x -= y`, are marked `C` and count no calls left to NumPy, since every float64 add and subtract passes their hook. Setting `VECLIB_MAXIMUM_THREADS=1` before NumPy starts removes the delay too, and made that training faster still (170 ms), but it changes every matrix product in the program, so hotkernels does not do it for you.
- **`np.clip` on int64 needs integer bounds.** Python ints that fit int64, or `np.int64`, with `lo <= hi`, and no `out`. A float bound gives NumPy's float64 result, and is NumPy's.
- **`np.round` gives NumPy's result bit for bit**, because it does what NumPy does, in the same order: multiply by the power of ten, round to the nearest whole number with ties to even, divide. So `np.round(-0.004, 2)` is `-0.0` and `np.round(0.125, 2)` is `0.12`, as with NumPy. Negative `decimals`, more than 15, and `decimals` given as a NumPy integer go to NumPy. So does a call where the multiplication overflows, or the array holds a signalling NaN: NumPy computes it and warns, or raises under `np.errstate`, as it always did.
- **`np.median` gives NumPy's result bit for bit**: the middle element, or the two middle elements added and halved, as NumPy's mean of them does. Some calls go to NumPy, which then answers and warns as it always did: an array holding a NaN; two middle elements whose sum overflows, or that are infinities of both signs; a half that is too small to be exact; and a median of zero when the array holds a negative zero anywhere, since which zero NumPy returns then depends on how its partition orders equal elements (NumPy 2.5 returns `+0.0`). `np.median(x, axis=None)` is served as `np.median(x)` is; any other `axis`, and every other argument, `out`, `keepdims` and `overwrite_input` included, are NumPy's.
- **`np.isin` is fastest when the test values are close together**: all between 0 and 65,534, or all within 32,767 of the first one. Otherwise it runs at the speed of compiled C, which is still ahead of NumPy.
- **`np.searchsorted` and `np.interp` trust you that the array is sorted, as NumPy does.** Neither checks. On an array that is not sorted both return a meaningless result, and the kernel's is not the same meaningless result as NumPy's. Sorted data, including NaNs at the end of `a`, gives NumPy's answer bit for bit.
- **`np.digitize` is served when its bins increase.** It calls `np.searchsorted`, on either side. With decreasing bins it searches them reversed, as a view no kernel takes, and runs at NumPy's speed.
- **`np.interp` with a few queries against a long table stays with NumPy.** A NaN among the sample points is something the kernel cannot handle, so `xp` is read once before every call. With more than 64 sample points per query that pass would cost more than the kernel saves.
- **bottleneck's own functions are served too, when bottleneck is installed.** pandas and empyrical, among others, send NaN-aware reductions to [bottleneck](https://github.com/pydata/bottleneck) instead of NumPy, so no NumPy hook sees them. hotkernels serves `nanmean`, `nanstd`, `nanmin` and `nanmax` along one axis of a two-dimensional float64 array, with bottleneck's result bit for bit, however your code or the library got hold of the function. At 252 × 4,789, the rolling windows of a year over twenty years of daily returns, they are about 4x to 7x faster than bottleneck on an Apple M1 Pro. hotkernels never imports bottleneck: if it is not imported when you call `install()`, its functions are served from the moment it is, and `report()` says "waiting" until then. `install(bottleneck=False)` leaves them alone. In `report()` they are the `bn.*` rows, and on those rows "Left to NumPy" counts calls left to bottleneck, such as one-dimensional arrays, `axis=None`, float32 and pandas' `Series.std()`. If your bottleneck was built so that it rounds `nanstd` differently from the kernel, which `install()` checks against bottleneck's own answer, `nanstd` stays with bottleneck and `report()` lists it under the hooks that are not in place.
- **Free-threaded Python is not supported.** `install()` does nothing there, and `report()` says so.
- **NumPy 2.0 to 2.2 are not supported.** They answer a few calls differently from later releases (`np.clip` drops the sign of a zero on 2.0, and `np.count_nonzero(x)` gives a Python `int` before 2.3), and the kernels give the later answer. Installing hotkernels brings NumPy 2.3 or later; with an older NumPy in place `install()` does nothing, and `report()` says so.
- **Subclasses mostly stay with NumPy.** `x[mask]`, the methods and the hooked functions take a kernel only for a plain `ndarray`; masked arrays, memory maps and your own subclasses go to NumPy there. `clip` and `argmax` are also hooked beneath NumPy's own handling of array types, so for a subclass they serve whatever plain buffer NumPy hands them, and NumPy wraps the result as it always did.

## Checking it yourself

From a clone of the repository:

```bash
uv run delivery demo          # run examples/workload.py with and without hotkernels, and compare
uv run delivery check         # 243,601 calls, NumPy's and bottleneck's, both ways; results, errors and warnings must agree
uv run delivery time          # time each statement from Python on your machine
uv run delivery numpy-tests   # NumPy's own test suite, both ways; a minute or two
```

`delivery demo` is the quickest way to see the effect: an ordinary NumPy program on two million elements that runs in 45 ms on plain NumPy and 8.4 ms with the one line added, with the same results bit for bit.
