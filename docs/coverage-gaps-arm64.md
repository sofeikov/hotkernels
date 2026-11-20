# Coverage gaps on a real workload, on ARM64

[Issue 160](https://github.com/sofeikov/hotkernels/issues/160) lists the NumPy calls of a production pipeline that hotkernels passed to NumPy because their element type, axis or dimension had no kernel. This document is the ARM64 side of the answer: what each call costs on an Apple M1 Pro, which calls got a kernel with ARM64 assembly, and which did not and why.

Every figure here is ARM64's, measured on an Apple M1 Pro with NumPy 2.5.5 and Apple clang 21. The issue's own figures are x86-64's.

## Summary

- **Five calls have room for a kernel on ARM64:** `np.searchsorted` on int64 (both sides), `np.sum` on float32 and on float64 with NumPy's bits, `m.any(axis=0)` on a boolean matrix, and `np.nonzero` on a boolean matrix.
- **Three of the issue's calls get no kernel,** because the compiler's loop is already NumPy's speed: `astype` between float64, float32, int16 and int64, and `np.sum` on int64. What they would gain is the fixed cost of a call on a short array, which is a loop inside the binding and not a kernel. `np.round` on a single number is the same case. They are recorded under [What is not built](#what-is-not-built).
- **`np.sum` on floats can be exact.** NumPy adds in a fixed order, eight running sums over blocks of up to 128 elements with the blocks added in pairs, and a plain C loop that adds in the same order gives the same bits 2.0x faster than NumPy on float32 and 1.6x on float64. The blocks do not depend on each other, which no compiler uses and assembly can.
- **The issue's workload is short arrays in Python loops.** There a kernel's own time is a few nanoseconds of a call of 500 to 800, so most of the gain for that workload is the hook, and the assembly is for the long arrays of other callers.

## NumPy against naive C

The survey's method ([survey-design.md](survey-design.md)): the NumPy statement against the simplest correct C loop, compiled with the judge's flags, timed inside C. The candidates are the group `coverage gaps` in `survey/candidates.py`, and the recorded run is [apple-m1-pro-coverage-gaps.md](../survey/results/apple-m1-pro-coverage-gaps.md), taken at a load average of about 2.3. Each ratio is NumPy's time over the C's; the last column is NumPy's time for one call on one element, which is its fixed cost.

| Call | Naive C | 16 | 4,096 | 1,048,576 | NumPy, one element |
|---|---|---|---|---|---|
| `np.searchsorted(a, key)` on int64, one needle | binary search | 205x | 53.9x | 21.8x | 640 ns |
| `np.searchsorted(a, v)` on int64, as many needles as elements | binary search | 8.90x | 0.66x | 0.46x | 536 ns |
| `np.nonzero(m)`, 22 columns, 5% True | two stores per True element | 14.7x | 5.04x | 3.91x | 452 ns |
| `np.nonzero(m)`, 22 columns, half True | the same | 15.6x | 5.55x | 2.09x | 474 ns |
| `m.any(axis=0)`, 31 columns | each row or'ed into the result | 21.2x | 3.06x | 2.30x | 570 ns |
| `m.any(axis=0)`, 30 rows | the same | 20.3x | 6.53x | 1.13x | 571 ns |
| `np.sum(x)` on float32 | NumPy's order of additions | 244x | 4.58x | 2.00x | 841 ns |
| `np.sum(x)` on float64 | NumPy's order of additions | 243x | 4.47x | 1.63x | 826 ns |
| `np.sum(x)` on int64 | one running sum | 413x | 3.78x | 1.03x | 826 ns |
| `x.astype(np.float32)` on float64 | one conversion per element | 8.99x | 1.80x | 1.01x | 177 ns |
| `x.astype(np.int64)` on int16 | the same | 8.13x | 1.50x | 1.00x | 166 ns |

What the table says:

- **Every call is far behind at 16 elements.** That is NumPy's call path, 170 to 840 ns, against a loop of a few nanoseconds. The issue's workload lives in this column.
- **Four calls are behind where the call path no longer matters:** `np.nonzero` on a matrix (2.1x to 3.9x at a million elements), `m.any(axis=0)` on a narrow matrix (2.3x), and `np.sum` on float32 and float64 (2.0x and 1.6x).
- **Two are level:** the casts and `np.sum` on int64. Clang turns each loop into NEON at full width and so, in effect, does NumPy.
- **One is ahead of the naive C:** a search with many needles on int64. NumPy's int64 comparison compiles to a conditional select, and the naive loop's branch is mispredicted half the time. So an int64 search kernel must not ship as its C on ARM64: it needs assembly that keeps many searches in step, as the float64 kernels do, before a hook serves arrays of int64 needles here.

## Built on ARM64

One entry per kernel, added by the pull request that builds it: what was built and what the judge measured.

## What is not built

Recorded so the question is not asked again without new figures. "Level" means the NumPy call is within a tenth of the naive C from 65,536 elements up, on the Apple M1 Pro.

| Call | Finding | What would help instead |
|---|---|---|
| `x.astype(np.float32)` on float64 | Level: 1.02x. Clang's loop is NEON at full width, 0.11 ns per element, and so is NumPy's | A loop in the binding for short contiguous arrays: NumPy takes 177 ns for one element, the C with its allocation 20 ns |
| `x.astype(np.int64)` on int16 | Level: 1.01x | The same |
| `x.astype(np.intp)` on int64, `x.astype(np.float32)` on float32 | A copy; nothing to compute | The same |
| `astype` on a strided source (`ctx[..., idx].astype("float32")`) | Not measured: the kernels take contiguous buffers, and a strided one would be copied first, which is the cast's own cost | Nothing here |
| `np.sum(x)` on int64 | Level: 1.07x, and 1.03x at a million elements. Assembly could use more running sums than clang's four, a gain inside the first-level cache only | The `sum` hooks could take short int64 arrays with a C loop: NumPy's 826 ns for one element is all call path |
| `np.around(x, 1)` on one float64 number | Not a loop | The `round` hooks taking a 0-d array or an `np.float64`, with the arithmetic they already do |
| A count per reason in `report()` | Not a kernel | A change to the binding's counters and the report; it would have attributed the calls the issue could not (`x[mask]`, `x[mask] = v`, `np.median`, `np.argmax`, `np.isin`) |

The first five rows are one piece of work: loops inside the binding for short arrays, as [the second shortlist's decision 4](second-shortlist-design.md) settled for `np.clip` on int64, `np.round` and `np.median`. It is not done here, since no kernel comes of it.
