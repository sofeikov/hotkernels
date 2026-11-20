# hotkernels

Faster NumPy on Apple Silicon and on x86-64 Linux, with one line and no other change to your code.

```python
import hotkernels
hotkernels.install()
```

`install()` points NumPy at hand-written assembly kernels, ARM64 NEON or x86-64 AVX2, for the calls where NumPy is slowest. Every kernel is checked against NumPy's own output, including NaN, infinities, signed zeros and ties, and is memory-safe by test. A call no kernel handles goes to NumPy exactly as it was written.

## Install

```bash
pip install hotkernels
```

Wheels are for macOS on Apple Silicon and for Linux on x86-64 (glibc), CPython 3.11 to 3.14, with NumPy 2.3 or a later NumPy 2. Another Python on one of those builds from source, which needs the Xcode command line tools on a Mac, or a C compiler and GNU binutils on Linux. On Linux the processor must have the x86-64-v3 instruction sets (AVX2, FMA, BMI1, BMI2, LZCNT, MOVBE, POPCNT), which Intel processors have had since 2013 and AMD's since 2015; on an older one, and on any other machine, `install()` does nothing and `report()` says why, so code that calls it stays portable.

The speed figures on this page were measured on an Apple M1 Pro. On x86-64 NumPy has vector loops of its own for some of these calls, so the gains differ: larger for some, close to none for `clip` and `astype`.

## What gets faster

`np.argmax`, `np.argmin` (also along the rows of a matrix: `np.argmax(m, axis=1)` on 300,000 rows of ten `float64` is about 3.5x faster than NumPy on an Apple M1 Pro, and on rows of two or three about 15x), `np.clip` (on `int64` for arrays of up to 4,096 elements), `x[mask]` (and `np.compress`, `np.extract`), `x[mask] = v` (and `np.putmask`, `np.copyto(..., where=mask)`), `np.nonzero` (and `np.where(mask)`), `np.where(mask, a, b)` on arrays of one shape, `np.cumsum` (on boolean masks too), `np.maximum.accumulate` and `np.fmax.accumulate` on one-dimensional arrays, accumulations down the columns of a `float64` matrix (`a.cumsum(axis=0)`, `a.cumprod(axis=0)`, `np.maximum.accumulate(a, axis=0)`, `np.fmax.accumulate(a, axis=0)`; about 7x to 10x faster at 253 × 4,789 on an Apple M1 Pro), `np.fmax.reduceat` on short runs (runs of 16 are about 3x faster than NumPy at a million elements on an Apple M1 Pro; runs of 256 or more are slower than NumPy on large arrays, by up to 1.6x on Apple Silicon), `m.any(axis=1)` and `m.any(axis=0)` on two-dimensional boolean arrays, `np.round` on `float64` arrays of up to 4,096 elements, `np.median` and `np.median(x, axis=None)` on `float64`, a stable `np.argsort` (`kind="stable"`, `kind="mergesort"` or `stable=True`) on one-dimensional `float64`, `int64` and `uint64` arrays of 256 elements or more, `w -= b` and `w += b` on `float64` arrays of 131,072 elements (1 MB) or more on Apple Silicon (faster only right after a large matrix product has read `w` on several cores, as in a training loop's gradient step; elsewhere NumPy's speed), `x.astype(np.float32)` from `uint8`, `np.sum` on `uint8` and on boolean masks, `np.sum(x)` and `x.sum()` on one-dimensional `float32` arrays (NumPy's result bit for bit: the same additions in NumPy's order, several of its blocks at a time; on an Intel Core i5-14600KF about 24x faster than NumPy on 16 elements and 2.6x on a million; a sum that is not finite is left to NumPy), `np.count_nonzero`, `np.histogram`, `np.bincount` (on `uint8`, `int16` and `int64`), `np.unique` on `int64` labels (a range of values no wider than 256 or than the array; without counts, an inverse or indices), `np.isin`, `np.searchsorted` on `float64` and on `int64` with `int64` needles (and `np.digitize`, which calls it) and `np.interp`, for the element types each kernel covers. At a million elements, from Python, `x[mask]` on `float32` is about 31x faster and `np.histogram` about 9x. On small arrays most of a NumPy call is NumPy's own call path, which hotkernels skips where it can: on 16 elements `np.clip(x, -0.5, 0.5)` is about 10x faster, when its bounds are plain numbers and there is no `out` other than `x` itself (`np.clip(x, lo, hi, out=x)` clips in place), and `np.argmax(x)` about 14x. `np.clip` on a short `int64` array, and `np.round` and `np.median` on a short `float64` one, which no kernel serves, are answered by plain C loops inside hotkernels for the same reason: `np.median` of 16 elements about 45x faster, measured on x86-64. On larger arrays `np.median` is a loop of hotkernels too, at every size, which finds the middle elements from the bits of the numbers instead of partitioning a copy: about 8x faster at a million `float64` on an Apple M1 Pro, though about level with NumPy, whose partition is quick there, on data already in order. The accumulations down the columns of a matrix are plain C loops too, at every size: NumPy walks a column at a time, a row apart between elements, and hotkernels walks the rows.

Libraries such as pandas and empyrical send some NaN-aware reductions to [bottleneck](https://github.com/pydata/bottleneck) rather than NumPy. When bottleneck is installed, hotkernels also serves `bottleneck.nanmean`, `nanstd`, `nanmin` and `nanmax` along one axis of a two-dimensional float64 array whose kept axis is contiguous (`axis=0` of a C-ordered array or of a rolling-window view), with bottleneck's result bit for bit, about 4x to 7x faster at 252 × 4,789. It never imports bottleneck; the functions are served from when it is imported.

## Using it

```python
import numpy as np
import hotkernels

hotkernels.install()            # NumPy now uses the kernels, in the whole process
x = np.random.standard_normal(1_000_000)
x[x > 0]                        # served by a kernel

print(hotkernels.report())      # what is installed, and how often each kernel ran
hotkernels.uninstall()          # NumPy is exactly as it was
```

- `hotkernels.install(inexact=True)` also installs `np.cumsum` on `float32` and `float64`, which are within a declared tolerance of NumPy rather than bit-identical.
- `hotkernels.install(bottleneck=False)` leaves bottleneck's functions alone.
- `with hotkernels.enabled():` installs for the duration of a block.

## Without changing your code

```bash
python -m hotkernels my_script.py --its --own --arguments
python -m hotkernels -m pytest
python -m hotkernels --report my_script.py   # print report() to standard error at the end
```

Documentation, the full results and how the kernels are made: [sofeikov.github.io/hotkernels](https://sofeikov.github.io/hotkernels/) and [github.com/sofeikov/hotkernels](https://github.com/sofeikov/hotkernels).
