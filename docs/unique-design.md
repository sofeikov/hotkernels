# `np.unique` on integer labels

This began as a proposal and is now built. Where the building departed from it, [What changed in the building](#what-changed-in-the-building) says so, and [Status](#status) holds the result against the exit criterion. It answers [issue 189](https://github.com/sofeikov/hotkernels/issues/189), found by the applications study ([applications-findings.md](applications-findings.md)): scikit-learn calls `np.unique` on every array of labels it is given, and in `sklearn_metrics` those calls are a quarter of the run.

The figures here are ARM64's, measured on an Apple M1 Pro with NumPy 2.5.5. No x86-64 hardware was at hand; see [x86-64](#x86-64).

## Summary

- **The call is `np.unique(y)` with nothing else**, on a one-dimensional `int64` array of a few distinct values. The `sklearn_metrics` application makes 67 such calls and no other spelling: 62 on a million or 300,000 elements, of two or ten values.
- **NumPy hashes every element.** `np.unique` on a million labels of two values takes 5.9 ms, on 300,000 of ten 1.8 ms. Reading a million `int64` for their minimum and maximum takes 0.2 ms in NumPy.
- **A new kernel, `unique_i64`, built like the bincount kernels.** Given the smallest value and the width of the range, it marks the values present in a table that is its own output buffer, and reads the table out in order. The binding reads the array once for its minimum and maximum first, and gives the kernel the range when it is narrow.
- **A function hook on `np.unique`.** Every NumPy release the package supports, 2.3 to 2.5, sends this call through the same Python code, which ends in a sorted array of the distinct values. The hook answers it before that code runs.
- **A wide range costs one read of the array** before the call goes to NumPy: about 0.2 ms at a million elements, where NumPy then takes 99 to 168 ms.
- **On x86-64 the kernel ships as its naive C**, as the second shortlist's kernels did on ARM64 until they had assembly there. That makes it the first C kernel on x86-64, and the Linux wheels then have to be built with clang, as [release-design.md](release-design.md#more-than-one-platform) says.

## Status

Against the [exit criterion](#exit-criterion), on an Apple M1 Pro with NumPy 2.5.5. Other work was running on the machine throughout (load averages from 3 to 22), so every figure is rough and the ratios are what to read.

| # | Criterion | Result |
|---|---|---|
| 1 | `judge check unique_i64` passes | Met, with `--unlocked`: 21,573 cases. The spec is new on this branch, and the branch's `pyproject.toml` and `uv.lock` differ from `main` |
| 2 | `judge run` faster than NumPy at every size | Met. 41.6x over NumPy and 1.44x over the naive C across the four timed cases; 39.5x to 40.0x over NumPy at a million elements of two or ten values (160 µs), 25.4x with a range of 256. Load average 22 |
| 3 | `delivery check` passes with the new cases | Met: 8,249 cases for `unique`, 981 taken by the kernel |
| 4 | Several times faster from Python, not slower on small or wide arrays | Met. `uv run delivery time --only unique`, load average 4.3: two values 13.2x at 16 elements, 19.6x at 4,096, 17.7x at 1,048,576 (6.55 ms to 371 µs); ten values 15.0x, 20.6x, 18.2x. A wide range, left to NumPy after the read: +0.4% at 16 elements in an alternating run of its own, -0.3% at a million |
| 5 | `sklearn_metrics` faster, identical results | Met. `uv run --group apps delivery apps --only sklearn_metrics`: 1,046 ms on NumPy, 790 ms with `install()`, 1.33x, results identical, all 67 `np.unique` calls taken. Alternating seven times with and without the `np.unique` hook at a load average of 20: 1,071 ms without it, 806 ms with it |

## What is slow today

`np.unique(x)` from Python, NumPy 2.5.5, the minimum of seven batches, at a load average of 3.3:

| Array | `np.unique(x)` | `x.min()` and `x.max()` |
|---|---|---|
| 16 elements, two values | 1.30 µs | 0.68 µs |
| 4,096 elements, ten values | 25.2 µs | 1.41 µs |
| 300,000 elements, ten values | 1.78 ms | 0.059 ms |
| 1,000,000 elements, two values | 5.91 ms | 0.21 ms |
| 1,000,000 elements, 0 to 999,999 | 99.2 ms | 0.21 ms |
| 1,000,000 elements, uniform over ±2^62 | 168 ms | 0.20 ms |

The answer to the fourth row is two numbers.

## How NumPy answers it

From `numpy/lib/_arraysetops_impl.py`, the same in 2.3.0, 2.4.0 and 2.5.5 (read from the installed sources of each):

1. `np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, *, equal_nan=True, sorted=True)` is a function object made by `array_function_dispatch`, the type every other function hook changes. It calls `np.asanyarray(ar)` and, when `axis` is None or the array has one dimension, `_unique1d`.
2. `_unique1d` flattens the array, which copies it. With no `return_index`, `return_inverse` or `return_counts`, and an array that is not masked, it calls `_unique_hash`, a C function of `numpy._core._multiarray_umath`, which puts every element into a hash set and returns the distinct values in the set's order. With `sorted` true the result is then sorted in place.
3. Every other combination sorts the flattened copy and compares neighbours.

So for an integer array the answer is the same on every supported release, whichever path NumPy takes: a new one-dimensional array of the array's element type holding each distinct value once, in ascending order. `equal_nan` changes nothing for integers. `sorted=False` does: the result is then in the hash set's order, which belongs to NumPy's C++ and is not reproduced here.

NumPy 2.3 is the oldest release the package supports ([delivery-design.md](delivery-design.md#which-numpy-versions)), and it has `sorted` and `equal_nan` as keyword-only parameters, as 2.5 does. In 2.3 `equal_nan` is not passed to `_unique_hash`; for integers that makes no difference.

## What scikit-learn calls

`sklearn_metrics` was run with every function of the `np.unique` family wrapped to record its arguments, the array's type and layout, and its caller (scikit-learn as locked in the `apps` group, NumPy 2.5.5):

| Calls | Spelling | Array | Caller |
|---|---|---|---|
| 20 | `np.unique(y)` | int64, contiguous, 1,000,000 or 300,000 | `sklearn.utils._unique._attach_unique` |
| 36 | `np.unique(y, return_index=False, return_inverse=False, return_counts=False)` | int64, contiguous, the same sizes | `array_api_compat`'s `unique_values`, for `type_of_target`, `unique_labels` and `_check_pos_label_consistency` |
| 3 | the same | int64, one column of a 1,000,000 × 2 array (stride 16 bytes) | the same |
| 3 | `np.unique(y)` | int64, two contiguous, one a column | `roc_auc_score`, `calibration_curve` |
| 5 | as the 36 | int64, under 1,000 elements | the same |

Every one of the 67 is the plain call, spelt with or without the three flags at False. None asks for counts, an inverse, indices or `sorted=False`, and none goes through `np.unique_values`, which asks for `sorted=False`. Four are on a column, which is not contiguous. `LabelEncoder`, which scikit-learn's estimators use in `fit`, asks for `return_inverse=True`; it is not in this application and is left to NumPy (see [Out of scope](#out-of-scope)).

## Where the call is caught

| Entry point | Reaches | Verdict |
|---|---|---|
| The function object `np.unique` | Every call of `np.unique`, however imported, including `array_api_compat`'s, which calls `np.unique` by name | Proposed |
| `_unique_hash` | Only the plain call; it is reached after `asanyarray`, `flatten`'s copy and `_array_converter` in Python, and it is private: a release may rename or remove it | Not proposed |
| `np.unique_values` | Asks for `sorted=False`: the hash set's order | Not proposed |

The hook takes a call when all of these hold, and passes everything else to NumPy unchanged:

- `ar` is exactly an `ndarray` of `int64` in native byte order, with at least one dimension and at least one element. Any layout: one that is not C-contiguous is first copied in C order, as `flatten` does.
- `return_index`, `return_inverse` and `return_counts` are absent or `False`, by position or keyword; `axis` is absent or None; `equal_nan` is absent, `True` or `False`; `sorted` is absent or `True`. Any other value, even one Python treats as false such as `0`, is NumPy's to interpret.
- The range is narrow: the largest element minus the smallest, plus one, is at most `max(n, 256)` for `n` elements. 256 values are 2 KB of table, the stack buffer the binding already keeps; above that the table is no larger than the array, so it costs at most about one more pass over as many bytes.

Empty arrays and 0-d arrays are NumPy's. So are other integer types, bool, and `np.longlong` on Linux, which is a type of its own there.

## The kernel

```c
// The distinct values of x that lie in [lo, lo + span), in ascending order,
// written to out[0..count). Returns count. Elements outside the range are
// ignored. out has room for span values and may be written anywhere inside it.
size_t unique_i64(const int64_t *x, int64_t *out, size_t n, int64_t lo, size_t span);
```

- **The output is the table.** It has `span` elements, so it can hold every distinct value of the range, and the kernel may mark presence in it before it compacts the marked positions to the front. It is declared `scratch=True` ([judge-design.md](judge-design.md#scratch-output-writing-inside-the-capacity)), as the compress kernels are: only the returned count of elements is compared with NumPy. The binding owns the buffer, as it must for a scratch kernel, and copies the values into the result.
- **Elements outside the range are ignored,** as the bincount kernels ignore elements outside their counters. The kernel is then safe on any input, and the judge's general integer classes test that. The range is computed in unsigned arithmetic, so `lo + span` may pass the largest `int64`.
- **The spec's oracle** is `np.unique` of the elements inside the range; what is timed as NumPy is `np.unique(x)` on data inside it.
- **The naive C** zeroes the table, stores a one at each element's place, and writes `lo + j` for each marked place `j`. Assembly is kept only where the judge finds it faster than that C.

The minimum and maximum are read by the binding, not the kernel, as the largest element is for the bincount kernels: the kernel needs the range before it reads its first element. The scan runs at the speed of a read; on ARM64 it keeps several running minima and maxima, as the bincount scans do.

## The binding

- A function hook on `np.unique`, in `hook_function.c`, counted as a new operation, `unique`.
- The scan, then the kernel on a buffer of `span` values, on the stack up to 2 KB and from `malloc` above, then a new array of exactly the count, of the input's element type, into which the values are copied. A call whose buffer or result cannot be allocated goes to NumPy, which raises its own error if it has one.
- The interpreter lock is released for the scan and the kernel from 8,192 elements, as for bincount. A thread that writes the array between the two reads can change which values are found, never make the kernel write outside its buffer.
- `install()` proves the hook with a probe whose answer has values below zero and gaps in the range.

## Tests

- **The judge** checks the kernel at every size against its oracle, with value classes of labels, of classes, of values just inside and just outside the range, of the extremes of `int64`, and scalar sets that put the range at zero, below it, at both ends of `int64` and at widths from 1 to 65,536.
- **`delivery check`** adds cases for `np.unique` on int64 of every layout and shape, of narrow and wide ranges and at the limit of the range, with every spelling of the flags, and on the element types next to it, each with the kernel it must reach or NumPy.
- **`delivery time`** times `np.unique(x)` at its usual three sizes, and a call that falls back after the scan.
- **`delivery apps`** times `sklearn_metrics` before and after.

## x86-64

When this was designed there was no x86-64 machine at hand, and the kernel shipped as its naive C on x86-64, compiled with clang for the x86-64-v3 level. It has x86-64 assembly since ([What changed in the building](#what-changed-in-the-building)); the rest of this section is why the Linux wheels are built with clang, which the `argsort` and bottleneck kernels, C on x86-64 still, go on needing.

The Linux wheels are built by `cibuildwheel` in a manylinux image whose compiler is gcc. [release-design.md](release-design.md#more-than-one-platform) records that this is acceptable only while no kernel ships as C on x86-64, and that the day one does, the image has to install clang and the build point `CC` at it. This change makes that day: `[tool.cibuildwheel.linux]` gets both.

## Documents that change

`docs/delivery-design.md` (the hook table and the file list), `docs/kernel-findings.md`, `docs/release-design.md`, `README.md`, `hotkernels/README.md`, `docsite/pages/start.md`, `docsite/pages/index.md` and the report chapters that quote the number of kernels, `survey/` with a candidate and its result, and a release note.

## Exit criterion

1. `uv run judge check unique_i64` passes.
2. `uv run judge run unique_i64` reports the kernel faster than NumPy at every size.
3. `uv run delivery check` passes with the new cases.
4. `np.unique(x)` from Python is several times faster than NumPy at a million elements of two values and 300,000 of ten, and not slower at small sizes or on a wide range.
5. `sklearn_metrics` is faster with `install()` than before this change, with identical results.

## Out of scope

- **`return_counts`, `return_inverse`, `return_index`.** `np.unique(y, return_inverse=True)` is what `LabelEncoder` calls in an estimator's `fit`, and `return_counts=True` what class weights are computed with. The same table gives both (counts are what the bincount kernels compute), but they need a second kernel or a second output, and no call in the application asks for them. Recorded as [issue 200](https://github.com/sofeikov/hotkernels/issues/200).
- **Other element types.** Labels from Python lists are `int64` on both platforms; `int32` and `uint8` labels can follow with kernels of their own.
- **`sorted=False` and `np.unique_values`.** Their order is NumPy's hash set's.
- **Labels spread over 65 or more values, on x86-64.** The x86-64 kernel's window takes ten neighbouring values in a range of 256, which is all the spec times; spread labels are a store each ([issue 241](https://github.com/sofeikov/hotkernels/issues/241)).

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| The range is tested in unsigned arithmetic, so `lo + span` may pass the largest `int64` | The kernel is defined only for ranges whose last value, `lo + span - 1`, is at most the largest `int64` | A range that wraps past it gives its values in unsigned order, not NumPy's; the judge found it in the reference. The binding never passes one: its range runs from the smallest element to the largest |
| The naive C, with assembly where it beats it | ARM64 assembly: a 64-bit set built with `ushl` for ranges of up to 64 values, without the subtraction for labels from zero, and a byte table in the top of `out` above that | 1.44x over the naive C overall; `kernels/unique_i64/arm64/attempts.md` |
| An attempts log on each architecture | An ARM64 log only at first; the x86-64 log came with the x86-64 kernel | Nothing had been tried on x86-64, and the documentation site refuses a log without a numbered attempt |
| The naive C on x86-64 until someone with the hardware writes assembly | x86-64 assembly, written afterwards on an Intel Core i5-14600KF: the set by `vpsllvq`, and in front of the byte table a window of 64 values that follows the data | 3.00x over the naive C overall, 1.3x to 1.9x at a million elements; from Python `np.unique(x)` went from 374 µs to 279 µs at a million elements and from 1.49 µs to 835 ns at 4,096. `kernels/unique_i64/x86_64/attempts.md` |
| A survey entry | The group `applications` in `survey/candidates.py`, recorded in `survey/results/apple-m1-pro-unique.md` | NumPy is 11.1x behind the naive C from 65,536 elements up |

Decision 2, the limit on the range, is left as recommended. Decision 1 was left for review: the kernel shipped as C on x86-64, until its assembly was written, and the Linux wheels are built with clang. That change to the wheel build has not been run: the `build-linux` job of the release workflow runs it on the pull request, since it touches `kernels/` and `hotkernels/`.

## Decisions still open

| # | Question | Recommendation |
|---|---|---|
| 1 | Ship the kernel as C on x86-64 and build the Linux wheels with clang, or leave `np.unique` to NumPy on x86-64 until it has assembly? | Ship it as C: the gain is NumPy's hash, not the last factor a hand-written loop would add |
| 2 | Is `max(n, 256)` the right limit on the range? | Yes, until a measurement says otherwise |
