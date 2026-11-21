[% macro cell(c) %][% if c %][[ c.text ]][% if not c.significant %] <small>n.s.</small>[% endif %][% endif %][% endmacro %]
<p class="lede">This chapter has two halves. The first is every number the project has, with where it came from. The second is the list of ways those numbers are narrower than they look. A speedup table without the second half is an advertisement.</p>

## What a user gets: timings from Python

These are the figures that matter to someone running NumPy code: one Python statement, timed from call to result, on plain NumPy and again after `hotkernels.install()`. They include everything: NumPy's call path, the hook, the checks, any buffer the binding allocates.

<figure class="widget">
<div data-widget="pytimes"></div>
<figcaption>Recorded timings of Python statements at three sizes, on plain NumPy and with hotkernels installed. Switch to "Calls left to NumPy" to see what a hook costs a call it does not serve. Read from the recorded result in the repository when this site was built.</figcaption>
</figure>

| Statement | Elements | 16 elements | 1,048,576 elements |
|---|---|---|---|
| `np.argmax(x)` | float32 | 13.2x | 8.89x |
| `x.argmax()` | float32 | 2.98x | 8.89x |
| `np.clip(x, -0.5, 0.5)` | float32 | 11.1x | 11.3x |
| `np.clip(x, -0.5, 0.5)` | float64 | 9.81x | 4.69x |
| `x[mask]`, half selected | float32 | 2.48x | 30.7x |
| `x[mask]`, half selected | float64 | 2.53x | 16.3x |
| `x[mask] = 0.0`, half selected | float64 | 5.19x | 20.7x |
| `np.nonzero(mask)`, few selected | bool | 1.25x | 13.4x |
| `np.nonzero(m)`, rows of 22, few True | bool | 1.27x | 26.9x |
| `np.cumsum(x)` | int64 | 1.94x | 12.3x |
| `x.astype(np.float32)` | uint8 | 1.90x | 5.15x |
| `np.sum(x)` | uint8 | 29.4x | 15.6x |
| `np.count_nonzero(x)` | float64 | 3.34x | 3.78x |
| `np.histogram(x, 50, (-3.0, 3.0))` | float64 | 6.13x | 9.33x |
| `np.bincount(x)` | uint8 | 2.27x | 6.75x |
| `np.isin(x, eight_values)` | int64 | 72.6x | 18.2x |
| `np.searchsorted(a, v)` | float64 | 1.49x | 8.83x |
| `np.searchsorted(a, v, side='right')` | float64 | 1.45x | 7.71x |
| `np.digitize(x, bins)`, 1,000 bin edges | float64 | 1.10x | 26.8x |
| `np.interp(x, xp, fp)` | float64 | 5.61x | 21.0x |
| `np.cumsum(mask)`, half True | bool | 2.77x | 19.2x |
| `np.maximum.accumulate(x)` | int64 | 2.93x | 14.2x |
| `np.fmax.accumulate(x)` | float64 | 2.85x | 12.8x |
| `np.fmax.reduceat(x, starts)`, runs of 16, 5% NaN | float64 | 3.46x | 3.05x |
| `m.any(axis=1)`, rows of 22 | bool | 5.52x | 24.0x |
| `np.bincount(x)`, codes 0 to 63 | int16 | 4.48x | 9.44x |
| `np.bincount(x)`, codes 0 to 63 | int64 | 2.71x | 4.26x |
| `np.where(c, a, b)`, half true | float64 | 4.79x | 1.60x |

The search and `np.interp` rows are timed with as many needles as elements, and `np.digitize` with 1,000 bin edges. The run started at a load average of 2.2, on a machine that also runs other work. The `np.nonzero(m)` row, on a matrix and on Apple Silicon only, is from a later run, on 20 November 2026 at a load average of 1.6.

Read the two columns as two different stories. At a million elements the number is the kernel. At 16 elements it is mostly NumPy's own call path being skipped: `np.sum` on 16 bytes is 29x faster because a 1.2 µs wrapper was replaced by a 40 ns call, not because of how the bytes are added, and `np.clip` is 11x faster because the kernel is called before NumPy's wrapper and ufunc machinery run.

**A whole program.** `examples/workload.py` in the repository is an ordinary NumPy script, written with no knowledge of hotkernels: it cleans a signal, summarises an 8-bit image and looks up some ids, using eleven of these operations on two million elements. It runs in **45 ms on plain NumPy and 8.4 ms with the one line added**, and every result is identical bit for bit.

**The cost of being there.** A call that reaches a hook and is not served costs 0.3 to 1.5 ns more than on plain NumPy: `x[3]` takes 66.7 ns on plain NumPy and 66.6 ns with hotkernels. Hooks that read keyword arguments cost a little more, and on calls that cost more: `np.clip(x, -0.5, None)`, which no kernel serves, goes from 1.02 µs to 1.04 µs.

## What the kernels do: timings as native functions

These are the judge's figures: each kernel called from C, against the naive C reference compiled with `clang -O3`, and against the NumPy statement called from Python. "All sizes" is the geometric mean over eleven sizes from 16 to 16,777,216 elements with its 95% interval; the four columns after it are single sizes. A row whose input is marked † is the mean of ten runs, and its brackets hold the lowest and highest of the ten instead.

| Kernel | Input | vs naive C, all sizes | at 16 | at 4,096 | at 1M | at 16M | vs NumPy at 1M |
|---|---|---|---|---|---|---|---|
[% for r in data.findings %]
| `[[ r.kernel ]]` | [[ r.input ]] | [% if r.c_kernel %]ships as its C[% else %][[ cell(r.vs_c) ]][% if r.vs_c and r.vs_c.interval %] <small>[[ r.vs_c.interval ]]</small>[% endif %][% endif %] | [[ cell(r.vs_c_at[0]) ]] | [[ cell(r.vs_c_at[1]) ]] | [[ cell(r.vs_c_at[2]) ]] | [[ cell(r.vs_c_at[3]) ]] | [[ cell(r.vs_numpy_1m) ]] |
[% endfor %]

"n.s." marks a difference the judge did not find statistically significant. Kernels timed on two kinds of input have a row for each and one for both together.

Three patterns are worth reading out of that table:

- **Speedups over the compiler peak in the middle.** At 16 elements fixed work dominates; at 16 million, memory does. The largest ratios are at 4,096 elements, where the data fits the first-level cache and the loop is all that is being measured.
- **The NumPy column and the C column measure different things.** `argmax_f32` is 26.6x over naive C and 8.82x over NumPy: NumPy's `argmax` was already better than naive C. `isin_i64` on wide values ties the C and is still 1.83x over NumPy. `clip` gains little over the compiler and a lot over NumPy. The C column is what assembly added. The NumPy column is what the user gains.
- **The two search kernels gain at the largest sizes**, where everything else fades, because overlapping cache misses is what they do.

### How these were measured

- **Machine.** Apple M1 Pro, macOS, NumPy 2.5.5, Apple clang 21. Native, one core.
- **Method.** 30 interleaved samples per contender per size; ratio of medians; paired bootstrap interval, as in [chapter 3]([[ root ]]report/judge/#deciding-what-faster-means).
- **Passes.** Most rows come from one pass on an idle machine. The rows of the eight second-shortlist kernels, the last in the table, are one recorded run each, taken on a quiet machine at a load average of 1.4 to 2.9: locked for six, and `--unlocked` for `fmax_reduce_f64` and `cumfmax_f64`, whose narrowed specs are not yet on the main branch. The `clip` and search rows, marked †, were taken later on a machine that could not be made quiet (load average about 3 for most runs), so each is the mean of ten locked runs. Up to about 65,536 elements those runs agree within 1%; from a million elements up single runs differ by 10 to 15%.
- **Unlocked.** The judge normally refuses to time a kernel whose spec differs from the main branch. The single-pass runs were made in the same change that introduced the specs, with that refusal overridden, and the judge marks such results as untrusted. Nothing was recorded as an official result from those runs for that reason.

## Where a result is narrower than its number

### `np.isin`: it depends on the test values, and on their order

The 75.6x over NumPy applies when all eight test values lie between 0 and 65,534, or within 32,767 of the first one. For arbitrary 64-bit values the kernel is the compiled C: 1.83x over NumPy. Because the second window is centred on the first test value, the same set of values can be fast in one order and slow in another. At 16 elements the kernel is slower than naive C on wide values (0.87x), because it has tried and rejected the fast set-ups first.

### `np.clip`: it depends on the bounds, and with a zero bound on the data

- With neither `lo = +0.0` nor `hi = -0.0`, the two-operation path runs: about 1.6x over the compiled C in cache.
- With `lo = +0.0`, as in `np.clip(x, 0, 1)`, a three-operation path runs: about 1.3x in cache, *if the data holds no negative zeros*. One negative zero among the first 7,264 float32 elements (3,632 for float64) sends the rest of the array to the compare-and-select loop, however long the array is. Rounded data is such data.
- With `hi = -0.0` the kernel is the compare-and-select loop: the speed of the C.
- For reversed or NaN bounds the kernel is not used at all.

Against NumPy it is 3.5x to 11x faster in all of the cases it handles. The conditions affect how much of that is owed to assembly.

### Float `np.cumsum`: faster because it is a different sum

Not bit-identical, by necessity ([chapter 6]([[ root ]]report/arithmetic/#floats-exact-and-fast-are-mutually-exclusive)), and not monotonic on non-negative input. Installed only with `install(inexact=True)`. Running sums above about $10^{153}$ in float64 take the slow path. Arrays shorter than 32 float64 or 20 float32 elements are summed in NumPy's order and are exact.

### The mask kernels: weakest between the densities that are timed

Timed at 5% and 50% selected. Near 10% the switch between the sparse and dense plans mispredicts, and `nonzero_bool` measured about 2.8 times its 5% cost per element there. From Python, `x[mask]` and `np.nonzero` keep roughly two thirds of the kernel's speed, because of the buffer, the copy and the mask scan.

### `x[mask] = v`: writes elements it does not change

Identical results, but unselected elements in dense regions are stored back with their own bits. The array must be writable throughout. The binding holds the interpreter lock during the kernel so that no other Python thread can write in between.

### Scalar arguments: verified for the values listed, and no others

`clip` is checked with twelve pairs of bounds; `histogram` with sixteen bin counts and ranges; masked assignment with six values. The judge has value classes for arrays and none for scalars. A combination nobody listed is untested.

## What the judge does not see

The judge is the reason any of this can be trusted, so its blind spots are the most important caveats in the report. Each of these was found the hard way.

| Blind spot | What it hid, or could hide |
|---|---|
| **Scalar combinations nobody listed** | The `clip` reference disagreeing with NumPy for reversed bounds; `clip`'s dependence on its bounds; the `histogram` overflow bug at the top of the float64 range, which no listed range was near |
| **Data that follows the scalars** | Nothing places elements on a histogram's bin edges, the only place a histogram goes wrong. That evidence comes from separate differential tests |
| **One input per size from 4,096 elements up** | A branch predictor memorises a repeated input, so branchy code looks faster than it is up to about 16,384 elements. This understates the masked-assignment kernel at those sizes, and hides the mask kernels' own mispredictions |
| **Densities and distributions between the ones timed** | The mask kernels near 10% density; `bincount` on all-equal bytes, which runs at 2.1 ns per byte below 8,192 elements |
| **Paths behind a size cap** | `interp`'s large-array path: NumPy's `interp` is too slow to act as an oracle beyond 65,537 elements, so that path was compared with the reference outside the judge, without guard pages |
| **A tolerance that widens with size** | For float32 `cumsum` beyond some tens of thousands of elements, the bound is wide enough that a single skipped element would pass |
| **Inputs outside what a spec can express** | Mask bytes other than 0 or 1; an output that is the input; a concurrent writer. All found when the kernels met real calls, and all screened by the binding |
| **Prefetch addresses** | Two kernels issue prefetch hints up to 256 KB past the end of very large arrays. A prefetch never faults and reads nothing, so this is safe, and invisible |

The pattern is consistent: the judge verifies what its generated inputs exercise. Every defect that got past it was in a region the generator did not reach, and each one that was found led to a new value class, a new scalar set or a new check in the binding.

## What is not delivered

- **`np.digitize` with decreasing bins.** It searches the bins reversed, as a view no kernel takes.
- **`np.interp` with `left`, `right` or `period`**, and with more than 64 sample points per query, where reading the sample points for NaN would cost more than the kernel saves.
- **One or two element types per operation.** `np.argmax` on int32, `x[mask]` on float16, `np.sum` on int64: all go to NumPy. (NumPy is already fast on several of those.)
- **Plain arrays only.** Subclasses reaching the function, method and indexing hooks go to NumPy.
- **`np.compress` with `axis` or `out`.** The compress kernels write past the elements they keep, so they never run on a buffer the caller owns.
- **In-place `clip`.** `np.clip(x, 0, 1, out=x)` goes to NumPy.
- **No wheels.** Installing compiles the package.
- **One platform.** macOS on Apple Silicon. Anywhere else `install()` does nothing.
- **Free-threaded Python** is not supported.
- **NumPy before 2.3** is not supported. Earlier NumPy 2 releases answer `np.clip` and `np.count_nonzero` slightly differently, and the kernels give the later answer.

## Decisions that are still open

Three choices were made provisionally, each with a recommendation the implementation follows:

| Question | Why it is a question | Current answer |
|---|---|---|
| May masked assignment rewrite unselected elements? | It is the whole of the gain on dense masks, and it is observable by a concurrent writer | Yes, with the interpreter lock held |
| May `clip` read and write the floating-point status register? | Without it the kernel is the compare-and-select loop, which ties the compiler in cache. It is more than "a pure function over its buffers" suggests | Yes, as a stated exception; flags are only ever added |
| Should float `cumsum` test for overflow once per block instead of four times? | About 12% faster from 256 to 16,384 elements. The proof that infinities land where NumPy has them would no longer hold near the largest finite value | No |

## One machine

Every figure in this report is from a single Apple M1 Pro. The techniques are general: they rest on latency, ports, prediction and caches, which every modern processor has. The constants are not. The number of accumulators, the block sizes, the thresholds between paths were all tuned to one core by measurement, and another core would want them measured again. The repository contains everything needed to do that: the judge, the specs and the logs. On an Apple Silicon Mac, `uv run judge run <kernel>` reproduces any row of the table above in about half a minute.
