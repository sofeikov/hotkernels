# Coverage gaps on a real workload: which calls get a kernel

This is a proposal for review. [Issue 160](https://github.com/sofeikov/hotkernels/issues/160) lists the NumPy calls of a production pipeline that hotkernels 0.3.0 passed to NumPy because their element type, axis or dimension has no kernel. This document goes through that list, says for each call whether a kernel is worth writing, and proposes the kernels, their hooks and the order of work.

Every figure here is x86-64's, measured on an Intel Core i5-14600KF. Nothing was measured on ARM64 for this document. The same issue was taken up on ARM64 separately, in [coverage-gaps-arm64.md](coverage-gaps-arm64.md), whose pull requests reached `main` first: they built `sum_f32`, `sum_f64` and `nonzero_2d_bool` with NEON assembly, installed on ARM64 only. The steps for `np.sum` and `np.nonzero` below take those kernels' specs and references and bring the kernels to x86-64.

## Summary

- **Four calls get a kernel with x86-64 assembly:** `np.searchsorted` on int64 (both sides), `np.sum` on float32, `m.any(axis=0)` on a two-dimensional boolean array, and `np.nonzero` on a two-dimensional boolean array. That is five kernels, since the search has one per side.
- **Three calls get no kernel because compiled C is already at the limit of the processor:** `np.sum` on int64 and `astype` from float64 to float32 and from int16 to int64. A hook with a C loop would still make them faster from Python. None is built here; the figures are recorded [below](#not-built) for a decision.
- **Two items of the issue are not kernels at all:** `np.around` on a zero-dimensional float64, and a breakdown of `report()`'s "Left to NumPy" by reason. Both are changes to the binding.
- **One sibling the issue does not ask for:** `np.sum` on float64 has the same room as float32, by the same technique. It is the last row of the order of work.
- **The rule for what is built:** a kernel is built where NumPy is behind the simplest C loop and assembly has room over that loop, measured before it is proposed. Where NumPy is already as fast as the C, or the C leaves assembly nothing, the call is recorded and left.

## What changed in the building

This section is added to as the steps of [the order of work](#order-of-work) are built. Each entry says what was built, where it departs from the text below, and which open decision it settled.

- **Row 1, `searchsorted_i64` and `searchsorted_right_i64`, on x86-64.** Spec, reference, x86-64 kernels and binding in one pull request, so the kernels' runs are `--unlocked`: the specs were not on `main` to lock against. The specs and references were committed before the first line of assembly and not changed after. They are the float64 specs with int64, and four more needle counts in the boundary sets (two to five). `searchsorted_i64` is 5.63x over the naive C across the judge's sizes and both needle counts and 3.68x over NumPy; `searchsorted_right_i64` 5.00x and 3.71x. Against NumPy with 1,000 needles that is 4.0x to 5.1x up to 4,096 elements and 1.2x to 1.45x from 1,048,576, less than float64 has, as expected. Built as proposed: the method hook of `searchsorted` (there is no function hook; `np.searchsorted` calls the method), a contiguous one-dimensional int64 array, and for the needles an int64 array of any shape, an `np.int64` or a Python int that fits int64. Three things the design did not say. The kernels are not a port of the float64 ones as they stand: two steps of a search in one pass over the needles, a first pass that needs no gather, and 256 needles in step were 1.09x, 1.06x and 1.01x in turn over the port, and those changes would apply to the float64 kernels, which were not touched. No call is left to NumPy for its size: from Python the hook was never slower than NumPy on lengths from 16 to 16,777,216 with 1 to 100,000 needles, so the binding has no size test. And an `np.intp` needle is taken, being the same type as `np.int64` on both platforms, while `np.longlong`, which is as wide, is not. From Python the pipeline's call, `np.searchsorted(a, f)` with one `np.int64`, is 2.1x faster at 16, 4,096 and 1,048,576 elements (559 ns to 264 ns at 16), and 1.9x with `side='right'`; with as many needles as elements 1.5x, 3.3x and 1.2x, at a load average of 1.4. [kernel-findings.md](kernel-findings.md#npsearchsorted-on-int64) has the rest. One attempt, a pass unrolled over two vectors, the judge rated significantly faster for the left kernel (1.006x) and not for the right; it was kept in neither, which is for a person to confirm. Open decision 4, needles of other integer types, stays open. On ARM64 the kernels ship as their C.

## What was measured

Each call of the issue's list against the simplest C loop for it, compiled with the judge's flags (`clang -O3 -march=native`), by a throwaway script that is not in the repository. The minimum of seven samples on one pinned performance core, at a load average of about 0.6. NumPy 2.5.5, CPython 3.13, clang 21.1.8. The C side writes into a buffer it is given where NumPy allocates its result, so at the largest sizes the comparison favours the C. These are naive timings, good enough to rank candidates; the judge measures each kernel again before it ships.

Nanoseconds per call, NumPy and then the C:

| Call | 16 elements | 4,096 | 65,536 | 1,048,576 |
|---|---|---|---|---|
| `x.sum()`, float32 | 277, 2.1 | 787, 441 | 8,051, 7,073 | 125,951, 114,226 |
| `x.sum()`, float64 | 277, 1.9 | 800, 393 | 8,250, 6,321 | 146,264, 105,901 |
| `x.sum()`, int64 | 281, 1.9 | 615, 112 | 4,981, 2,580 | 109,484, 105,905 |
| `x.astype(np.float32)`, float64 | 173, 1.2 | 824, 216 | 9,551, 4,897 | 170,774, 169,355 |
| `x.astype(np.int64)`, int16 | 173, 1.3 | 686, 194 | 7,571, 8,573 | 187,176, 179,664 |

The C for the float sums is NumPy's own pairwise order, not a running sum, so its result is NumPy's bit for bit.

Two-dimensional boolean arrays with a tenth of the elements True, by shape:

| Call | (30, 31) | (40, 22) | (2,978, 22) | (47,662, 22) | (4,096, 256) |
|---|---|---|---|---|---|
| `m.any(axis=0)` | 818, 98 | 898, 104 | 29,451, 6,770 | 464,748, 108,207 | 48,194, 13,206 |
| `np.nonzero(m)` | 1,802, 352 | 1,736, 333 | 140,803, 27,709 | 2,243,324, 454,161 | 2,052,412, 454,569 |

`np.searchsorted(a, v)` on int64, nanoseconds per needle, by the length of `a` and the number of needles:

| Length of `a` | 1 needle | 64 needles | 1,000 needles |
|---|---|---|---|
| 16 | 488, 2.3 | 10.3, 3.1 | 3.1, 3.1 |
| 4,096 | 509, 5.1 | 15.0, 8.0 | 7.7, 20.7 |
| 65,536 | 516, 11.4 | 17.9, 10.4 | 10.6, 42.3 |
| 1,048,576 | 541, 15.0 | 22.9, 13.4 | 18.6, 129.9 |

With an `np.int64` needle in place of an array of one, the NumPy call takes 584 to 633 ns at every length.

## The kernels

All five are for contiguous arrays only, as every other kernel is. Each gets a `spec.py` and a `naive.c` that both architectures share, and assembly for x86-64. On ARM64 each ships as its naive C until someone writes the NEON; that is not part of this work.

### 1. `searchsorted_i64` and `searchsorted_right_i64`

`np.searchsorted(a, v)` and `np.searchsorted(a, v, side='right')` for a sorted int64 array and int64 needles. The twins of `searchsorted_f64` and `searchsorted_right_f64`, with the same signature and the same specs, int64 in place of float64: `(a, v, out, n, needles)`, needles drawn from `a` as well as random. There is no NaN, so the order is plain signed comparison.

- **Why assembly:** the C takes one needle at a time, and from 4,096 elements a mispredicted branch or a cache miss per step is its whole cost: 130 ns per needle against a million elements. The float64 kernels search for many needles at once, and that is what these take over.
- **What to expect against NumPy:** less than float64 has. NumPy's int64 search is already about 3x faster than its float64 search with many needles (18.6 ns per needle against a million elements here; 33 against 104 in [the survey](../survey/results/intel-core-i5-14600kf.md)). The larger gain for a caller is the short call: one needle costs NumPy about 600 ns, nearly all of it its call path, and the float64 hook answers the same call in under 300.
- **The hook:** the function and method hooks of `searchsorted`, which serve float64 today. Taken: a contiguous one-dimensional int64 `a`, and `v` an int64 array, an `np.int64`, or a Python int that fits int64; `side` either way; no `sorter`. The single-needle path the float64 kernels have is the one the issue's 717,457 calls need.
- **Left to NumPy:** needles of another integer type, which NumPy converts by its own rules; any other element type of `a`.

### 2. `sum_f32`

`np.sum(x)` and `x.sum()` on float32.

NumPy does not add a float array from left to right. It adds in blocks of at most 128 elements, each block with eight running sums that are combined at its end, and it combines the blocks as a balanced tree. A kernel must give those bits, so it must keep that order. What it need not keep is the order in time: the blocks are independent of each other until they are combined. NumPy's loop works through one block at a time and waits on each add, four cycles for eight elements. A kernel that keeps one vector register per block and works through eight blocks at once is limited by loads instead.

- **Measured:** a prototype with AVX2 intrinsics, for lengths that are a multiple of 1,024, gives NumPy's bits and takes 78 ns at 4,096 elements, where the C in NumPy's order takes 441 and `x.sum()` 787; 2,000 ns at 65,536 against 7,073 and 8,051; 45,030 ns at 1,048,576 against 114,226 and 125,951.
- **The order was checked:** the C written from NumPy's source (`pairwise_sum` in `loops_utils.h.src`) gave `np.sum`'s bits for every length from 1 to 699 and six longer ones, on values spread over six decimal orders of magnitude, for float32 and float64. The reduction adds all `n` elements to a starting value of zero, so the sum of a lone `-0.0` is `+0.0`.
- **The domain:** inputs whose sum and partial sums are all finite. Which of two different NaNs an add returns depends on the order of its operands, which is the compiler's choice in NumPy's build, and a sum that overflows or adds infinities of both signs makes NumPy warn. So the binding gives the call to NumPy whenever the kernel's result is not finite. An add cannot return from infinity or NaN to a finite number, so a finite result means no step overflowed or was invalid, and the test costs one comparison.
- **The hook:** the function and method hooks of `sum`, which serve uint8 and bool today. Taken: a contiguous one-dimensional float32 array and no other argument. The result is an `np.float32`.
- **A risk:** the order is NumPy's source, not its documentation. The kernel is probed against NumPy at `install()`, as every kernel is, and the probe must use lengths that reach every part of the order: below 8, a block with a remainder, and several levels of the tree.

### 3. `any_cols_bool`

`m.any(axis=0)` and `np.any(m, axis=0)` on a C-ordered boolean matrix: for each column, whether any row holds a True. The mirror of `any_rows_bool`, with its signature: the matrix as `n` elements and the row width as a scalar, `n // cols` whole rows; it writes `cols` results.

- **Why assembly:** the C or's each row into the results in memory, 0.10 ns per byte on a (30, 31) matrix. For a width of up to a few hundred the results fit in vector registers for the whole call, and a row is then one or two loads and ors.
- **The hook:** the function and method hooks of `any`, which take `axis=1` today. Taken: `axis` given as the Python int 0 or -2, a two-dimensional C-contiguous boolean array, no other argument. The matrix is read for bytes other than 0 and 1 first, as for `axis=1`.
- **To decide in the building:** what the kernel returns when there are no whole rows, and the largest width it takes with the results in registers.

### 4. `nonzero_2d_bool`

`np.nonzero(m)` on a two-dimensional C-ordered boolean array.

NumPy's result for two dimensions is one array of shape (count, 2), row and column of each True element in C order, and the two arrays it returns are views of that array's columns. So the kernel writes pairs: `(m, out, n, cols)`, the matrix as `n` elements with rows of `cols`, `out` with room for two int64 per element, and it returns the number of int64 written. The oracle is `np.argwhere(m).ravel()`.

- **Why assembly:** the C stores a row number and a column number for every element and advances only on a True, 0.37 ns per element whatever the share of True. The one-dimensional kernel, `nonzero_bool`, takes the mask a block at a time and is 1.78x over its C on x86-64 with half the elements True and 4.57x with 5%; the same approach applies with a row number beside each index.
- **The hook:** the function and method hooks of `nonzero`, which take one-dimensional masks today. Taken: a two-dimensional C-contiguous boolean array. The result is a tuple of two views of one (count, 2) array, as NumPy's is. `np.argwhere(m)` could return the same array whole; whether its hook is worth having is for the building to say.
- **To decide in the building:** whether the count is taken first, so the result is allocated at its final size, as the one-dimensional hook does.

### 5. `sum_f64`, a sibling

`np.sum(x)` on float64 is not in the issue: the pipeline's sums are on float32 and int64. It has the same order, the same room (800 ns at 4,096 elements, 393 in C) and, with two registers per block, the same technique. It is the commonest reduction in NumPy. It is built last, after `sum_f32` has shown what the technique gives under the judge.

## Not built

| Call | Why not | What a hook with a C loop would give |
|---|---|---|
| `np.sum` on int64 | The compiler turns the C into vector adds that are limited by loads already; a loop unrolled by hand to eight sums took the same time (110 ns against 112 at 4,096 elements). Assembly has nothing to add. | 5.5x at 4,096 elements, 1.9x at 65,536, nothing at 1,048,576, and the call path of a short array (281 ns at 16 elements) |
| `astype` float64 to float32 | The conversion is one instruction per four elements and runs once per cycle; the compiled C takes 0.053 ns per element at 4,096, which is that. | 3.8x at 4,096 elements, 2.0x at 65,536, nothing at 1,048,576. The issue's calls are on arrays of at most 128 elements and on strided slices, where the call path is the cost |
| `astype` int16 to int64 | The same: one widening instruction and one store per four elements, 0.047 ns per element in C. | 3.5x at 4,096 elements, nothing from 65,536. The issue's calls are on at most 16 elements |
| `astype` int64 to intp, float32 to float32 | A copy. `memcpy` is the kernel. | The call path only |
| `np.around(x, 1)` on a zero-dimensional float64 | Not a loop. The `round` hook takes arrays of 1 to 4,096 elements and could take a scalar. | The call path only |
| `report()`: "Left to NumPy" by reason | Not a kernel: a counter per reason in the binding. | Nothing in speed; it is how a user would have found this list without a profiler |
| `x[mask] = v`, `x[mask]`, `np.median`, `np.argmax`, `np.isin` | The issue could not say what these calls pass, so there is nothing to design against. | Unknown |

"C is enough" here means that assembly would not beat the C, not that NumPy is as fast as the C. For the first four rows a kernel that ships as its C, as `cast_u8_f32` does on ARM64, would serve the call. They are left out because this work is the assembly; whether to add them as C is [an open decision](#open-decisions).

## Order of work

One pull request per row, each based on the one before, since all of them change the binding, its tables in the documentation and the count of kernels. Spec, reference, x86-64 kernel and binding are in one pull request, so the kernel's runs are `--unlocked`: the spec is not on `main` to lock against. The spec and reference are committed before the first line of assembly and not changed after.

| Step | Pull request | Kernels |
|---|---|---|
| 0 | This document | |
| 1 | `np.searchsorted` on int64 | `searchsorted_i64`, `searchsorted_right_i64` |
| 2 | `np.sum` on float32 | `sum_f32` |
| 3 | `m.any(axis=0)` | `any_cols_bool` |
| 4 | `np.nonzero` in two dimensions | `nonzero_2d_bool` |
| 5 | `np.sum` on float64 | `sum_f64` |

Each kernel keeps its assembly only if the judge finds it significantly faster than its naive C; one that does not ships as the C, and its pull request says so.

## Open decisions

1. **C kernels for the calls where C is enough.** `np.sum` on int64 and the two casts would each be 3.5x to 5.5x faster than NumPy at 4,096 elements from a C loop alone. Not built here.
2. **`sum_f64`.** Not asked for by the issue. Proposed because it is the same kernel twice as wide and a far commoner call.
3. **ARM64.** Each kernel built here ships as its naive C there, except where the ARM64 work above has a NEON kernel of its own (`sum_f32`, `sum_f64`, `nonzero_2d_bool`). Whether NEON is worth writing for any of them needs the survey's question asked on that machine.
4. **Needles of other integer types.** `np.searchsorted` on an int64 array with int32 needles is NumPy's here. If a recorded run shows such calls, the binding could widen them.
