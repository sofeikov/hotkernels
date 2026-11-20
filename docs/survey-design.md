# The survey: design and findings

This is a proposal for review. It details phase 2 of [core-design.md](core-design.md): where NumPy is slow, which operations are worth a kernel, and in what order to build them.

The answer depends on the architecture, because NumPy's own vector code does: an operation NumPy leaves scalar on one can be vectorised on another. The method is shared. The results, the shortlist and the order of work are given per architecture. Two have been surveyed: ARM64, and x86-64 as planned in [x86-64-design.md](x86-64-design.md).

Unlike the other design documents, the measuring part is already done. The harness in [`survey/`](../survey) exists and the numbers below are real. What is proposed, and open to change, is the ranking and the order of work.

The survey was run before the judge existed, so it uses its own small harness. The timings are "naive timings": good enough to rank candidates, not good enough to defend a 10% difference. Every candidate is measured again by the judge before it ships.

A second list of candidates, taken from the NumPy calls of a program in use, is in [second-shortlist-design.md](second-shortlist-design.md).

## What the survey is

A command-line tool, `uv run survey`, that for each candidate operation times the NumPy statement a user would write against the simplest correct C loop and reports how far NumPy is behind. It answers two questions per candidate:

1. **Is NumPy slow here?** Measured: NumPy time divided by naive C time.
2. **Can assembly beat the C as well?** Partly measured (did clang vectorise the loop?), partly judgement (is there a known technique the compiler will not find?).

The survey is a development tool, like the judge. It is not part of the `hotkernels` package.

## Layout

```
survey/
  naive.c        naive C baselines for every candidate, with their timing loops
  candidates.py  the candidate list: NumPy statement, C kernel, input, expected result
  measure.py     builds the C, checks it against NumPy, times both
  render.py      prepares the numbers for the templates
  templates/     Jinja templates for the report and the progress lines
  cli.py         the `survey run` and `survey report` commands
  results/       recorded results, committed: one JSON file and one report per run
```

The baselines in `survey/naive.c` are survey code. A candidate that goes forward gets its own reviewed `naive.c` and `spec.py` under `kernels/`, as the kernel contract requires.

## Method

- **Contenders.** For each candidate, the NumPy statement a user would write, against the simplest correct C loop compiled with `clang -O3` and tuned for the machine it runs on (`-mcpu=native` on ARM64, `-march=native` on x86-64). The flags are the judge's flags for a kernel's `naive.c`.
- **Timing.** C is timed inside C. NumPy is timed from Python and includes its call overhead. Samples are interleaved and the median of up to 11 is reported. The thread is kept on the performance cores: on macOS it asks for the highest scheduling class, and on Linux it is pinned to one performance core, claimed the way the judge claims its core, so a survey and a judge run never share one. A reference loop is timed before and after the run, and its drift is recorded with the result.
- **What else is recorded.** The vector instruction sets NumPy uses on the machine, since NumPy's own loops are the other contender. On macOS, whether Low Power Mode was on. On Linux, the CPU the thread was pinned to and anything the judge's checks found that holds the processor back: turbo switched off, a power-saving frequency setting, other work running.
- **Sizes.** 1, 16, 256, 4,096, 65,536, 1,048,576 and 16,777,216 elements.
- **Allocation.** Where NumPy allocates its result, the C side also mallocs and frees its output on every call. Otherwise the gap would partly be the allocator, which no kernel can remove.
- **Where the output sits.** A candidate that writes into a buffer the caller supplies gets that buffer half a memory page after its first input, counted within the page, and clear of every other input. Left to the allocator, two power-of-two arrays made one after the other often sit 16 bytes apart within the page, and on x86-64 that alone makes NumPy up to 8x slower ([below](#where-the-output-sits)). Each result row records the distances. Where the result is allocated on every call, the allocator still decides.
- **Correctness.** Every naive C kernel is checked against NumPy on the timed input.
- **Was the C vectorised?** Taken from clang's own loop-vectoriser remarks.
To reproduce:

```
uv run survey run
```

`uv run survey report <file.json>` regenerates the report of an existing result file.

"Gap" below always means NumPy time divided by naive C time, as a geometric mean over sizes of 65,536 and up. A gap of 4x means the naive loop is four times faster than NumPy.

## Recorded runs

| Architecture | Machine and run | Full tables |
|---|---|---|
| ARM64 | Apple M1 Pro, macOS 27.2.1, NumPy 2.5.5, Apple clang 21. The reference loop drifted by 0.6%. All 65 variants agree with NumPy. Taken before the survey placed output buffers | [apple-m1-pro.md](../survey/results/apple-m1-pro.md) (61 variants) and [apple-m1-pro-branchless.md](../survey/results/apple-m1-pro-branchless.md) (4 follow-up variants) |
| x86-64 | Intel Core i5-14600KF, Linux 7.0, NumPy 2.5.5 using its AVX2 loops (it requires X86_V2 and found X86_V3), clang 21.1.8. Pinned to one performance core, no warnings. The reference loop drifted by less than 0.1%. All 65 variants agree with NumPy | [intel-core-i5-14600kf.md](../survey/results/intel-core-i5-14600kf.md) |
| x86-64, second shortlist | The same machine, NumPy 2.5.5, clang 21.1.8, pinned to one performance core, no warnings. The reference loop drifted by less than 0.1%. The 13 variants of [second-shortlist-design.md](second-shortlist-design.md) and no others; all agree with NumPy | [intel-core-i5-14600kf-second-shortlist.md](../survey/results/intel-core-i5-14600kf-second-shortlist.md) |
| ARM64, second shortlist | Apple M1 Pro, macOS 27.2.1, NumPy 2.5.5, Apple clang 21, at a load average of about 2.5. The reference loop drifted by 1.6%. The same 13 variants and no others; all agree with NumPy | [apple-m1-pro-second-shortlist.md](../survey/results/apple-m1-pro-second-shortlist.md) |
| ARM64, applications | Apple M1 Pro, macOS 27.2.1, NumPy 2.5.5, Apple clang 21, at a load average of about 16, so rough. The reference loop drifted by 0.8%. The group `applications`: `np.unique` on int64 labels of two and ten values ([unique-design.md](unique-design.md)), 11.1x behind the naive C from 65,536 elements up; both agree with NumPy | [apple-m1-pro-unique.md](../survey/results/apple-m1-pro-unique.md) |
| ARM64, argsort | Apple M1 Pro, macOS 27.2.1, NumPy 2.5.5, Apple clang 21, on a machine in use (load average 4 to 8). The reference loop drifted by 0.4%. The eight variants of [sort-design.md](sort-design.md) and no others; all agree with NumPy | [apple-m1-pro-argsort.md](../survey/results/apple-m1-pro-argsort.md) |
| ARM64, coverage gaps | Apple M1 Pro, macOS 27.2.1, NumPy 2.5.5, Apple clang 21, at a load average of about 2.3. The reference loop drifted by 1.4%. The group `coverage gaps`, the calls a production pipeline left to NumPy ([coverage-gaps-design.md](coverage-gaps-arm64.md)), with `searchsorted_i64` and `cast_f64_f32` from the earlier groups: 12 variants, all agreeing with NumPy | [apple-m1-pro-coverage-gaps.md](../survey/results/apple-m1-pro-coverage-gaps.md) |

## Results on ARM64

### What the numbers say

1. **Most of the phase 2 list is slower than a naive C loop.** Of the candidates named in the core design, only `argmax`/`argmin`, `any`/`all`, `count_nonzero` on booleans and `searchsorted` on integers are ahead of naive C. Most of the rest is behind, by 1.5x to 7x; `take` through `a[idx]` and `clip` on integers are level.
2. **The biggest gaps are often closed by the compiler alone.** `clip` on floats (7x), `astype` from `uint8` to `float32` (5.5x), `np.sum` on `uint8` (3.3x) and `count_nonzero` on non-boolean data (2.9x) are all loops that clang vectorises by itself. Assembly would tie the C, so these ship as C kernels (decided in review). **Phase 3 measured otherwise:** assembly beat the C on five of the six, see [kernel-findings.md](kernel-findings.md).
3. **The assembly targets are the loops clang leaves scalar.** Mask selection, `nonzero`, `argmax`, `cumsum`, `searchsorted`, `interp`, `histogram` and `bincount` are not vectorised by the compiler, and each has a known hand-written technique.
4. **Branch misprediction is a large part of NumPy's cost on masks.** `x[mask]` at 50% density costs NumPy 4.8 ns per element. A C loop with no data-dependent branch does it in 0.34 ns: 14x faster, before any SIMD.
5. **Where NumPy already has NEON, it is very good.** `np.max`, `np.any`, `np.all`, `count_nonzero` on booleans and `packbits` run at 0.01 to 0.05 ns per element, 10x to 50x ahead of naive C. These are not targets.
6. **Stepped slices are not a target.** Copying and adding over `a[::2]` tie naive C (gap 1.03x to 1.09x), and `np.sum` and `np.argmax` on a stepped slice are ahead of it.

**Sorting, surveyed after the applications study.** A stable `argsort` was on neither shortlist: the survey's question is whether NumPy loses to the loop a programmer would write, and for a sort it does not. A bottom-up merge sort of indices in C is level with NumPy's timsort on int64 and uint64 keys (0.88x to 1.07x from 4,096 elements up) and half as fast on float64. The gap is in the algorithm: a plain radix sort of (key, index) pairs, eight passes of eight bits, is 6.4x to 8.8x faster than NumPy at a million elements and 8.6x to 22.6x at sixteen million, and 4x to 5x faster than NumPy's default, unstable kind. That is why the `argsort` kernels are a different algorithm in C and not NumPy's loop in assembly ([sort-design.md](sort-design.md)).

### Shortlist

Ranked by how likely a hand-written kernel is to beat both NumPy and C, then by the size of the gap. The first two columns of numbers are measured. The last column is a judgement and has not been tested.

| # | Candidate | NumPy vs naive C | ns per element at 1M: NumPy / C | Does clang vectorise it? | Technique the compiler will not find |
|---|---|---|---|---|---|
| 1 | Mask selection, `a[mask]` | 1.6x to 1.8x behind the branchy loop; 14x behind the branchless loop at 50% density, 2.7x at 5% | f64, 50%: 4.75 / 0.34 (branchless) | No | Turn each block of mask bytes into a shuffle index and compact with a table lookup |
| 2 | `nonzero` on booleans | 1.5x behind at 5% density. At 50% NumPy beats the branchy loop but is 1.9x behind the branchless one | 50%: 0.72 / 0.38 (branchless) | No | Same compaction as 1, emitting indices |
| 3 | `argmax` / `argmin` on floats | NumPy is ahead of naive C (0.33x f32, 0.54x f64) | f32: 0.43 / 1.29; f64: 0.71 / 1.29 | No | Vector max with index tracking and a NaN check per block. NumPy has NEON here, but its float `argmax` is 8x slower than its own `np.max` (0.05) and 2.5x to 4x slower than its integer `argmax` (0.17), so there is room |
| 4 | `searchsorted` on floats (and `digitize`) | 1.7x to 4.7x behind up to 1M; ahead at 16M | f64: 142 / 84 per needle | No | Branchless search, several needles advanced together so cache misses overlap. NumPy's int64 path (39 ns per needle) shows what is reachable |
| 5 | `interp` | 3.4x to 5.3x behind from 4,096 up | 679 / 127 per point | Not the search | Same search core as 4, then the interpolation step in SIMD |
| 6 | `cumsum` | 3.7x behind on floats, 6.5x on int64 | f64: 3.61 / 0.97; i64: 2.18 / 0.34 | No | SIMD prefix sum. Exact for integers. For floats it reorders additions, so it ships in tolerance mode (accepted in review); in exact mode the C loop is already at the limit of one addition per element |
| 7 | `histogram`, uniform bins | 7.1x behind | 6.26 / 0.88 | No | Compute bin numbers in SIMD, count into several tables to avoid store-to-load stalls |
| 8 | Masked assignment, `x[mask] = v` | 1.2x behind; both NumPy and C cost 3.4 to 3.9 ns per element at 50% density | 3.85 / 3.40 | Not usefully | Load, select, store whole vectors. The compiler may not do this because it writes elements the source code does not write |
| 9 | `bincount` | 4.2x behind | 1.37 / 0.33 | No | Several count tables. C is already at about one cycle per element, so the gain over C is likely small |
| 10 | `isin` with a small test set | 5.7x behind for 8 test values | 7.86 / 1.08 | Yes, the inner compare | Compare against all test values held in registers. Gain over C unclear |

Also measured and worth keeping in mind:

- **`np.minimum.accumulate`** (1.4x behind). A SIMD prefix minimum is exact, unlike a float prefix sum. Low usage.
- **`np.add.at`** (2.4x behind). No SIMD technique; a C-level win only.
- **`np.take(a, idx)`** is 3.2x behind naive C, while `a[idx]` is level with it (1.07x). The function form is slow, not the gather.
- **`np.sum(mask)`** costs 0.21 ns per element; `np.count_nonzero(mask)` costs 0.03. Routing one to the other is a 7x win with no new kernel.

#### C kernels

These have large gaps but clang vectorises the naive loop, so assembly is expected to tie C. They ship as C. The agent may still attempt assembly; it replaces the C only if the judge says it is significantly faster.

| Candidate | NumPy vs naive C | ns per element at 1M: NumPy / C |
|---|---|---|
| `np.clip` on float32 | 7.1x behind | 0.65 / 0.09 |
| `np.clip` on float64 | 3.5x behind | 0.65 / 0.18 |
| `astype`, uint8 to float32 | 5.5x behind | 0.32 / 0.06 |
| `np.sum` on uint8 | 3.3x behind | 0.19 / 0.06 |
| `count_nonzero` on int64 / float64 | 2.9x behind | 0.33 / 0.10 |
| `astype`, int64 to float64 | 1.8x behind | 0.34 / 0.18 |

`np.clip` on int64 is level with C, so the float gap is specific to NumPy's float loop.

#### Not targets

NumPy is level with or ahead of naive C: `np.max`, `np.sum` on floats, `any`, `all`, `count_nonzero` on booleans, `packbits`, `argmax` on booleans and integers, `searchsorted` on int64, `np.where(c, a, b)`, `np.diff`, `np.dot` on int64, `np.array_equal`, `a[idx]`, `a[idx] = v`, `astype` from float64 to float32, `np.correlate`, and everything measured on stepped slices.

### Per-call overhead

NumPy's time on a one-element array, which is almost entirely fixed cost. The naive C kernel takes a few nanoseconds here, or about 25 ns when it has to allocate.

| Fixed cost | Calls |
|---|---|
| 12,000 to 18,000 ns | `np.histogram`, `np.isin` |
| 1,100 to 2,100 ns | `np.clip`, `np.diff` |
| 700 to 1,250 ns | `np.sum`, `np.max`, `np.cumsum`, `np.any`, `np.all`, `np.array_equal`, `np.take(..., out=)` |
| 400 to 650 ns | `np.argmax`, `np.argmin`, `np.searchsorted`, `np.interp`, `np.add.at`, `np.dot`, `np.where` |
| 140 to 310 ns | `np.nonzero`, `np.bincount`, `np.packbits`, `np.count_nonzero`, `astype`, `x[mask]`, `x[mask] = v` |
| 60 to 100 ns | `a[idx]`, `a[idx] = v` |

Two consequences:

- The floor for a call that goes straight into C is around 60 to 150 ns. Calls costing 500 ns or more spend most of that in Python-level wrappers, and a patched function with a direct path could remove it. That is a win in the binding, not in a kernel, and it is largest for `histogram`, `isin`, `clip` and the reductions.
- It sets the budget for phase 3: our own binding has to cost well under 100 ns per call, or small arrays get slower.

### Order of work

Every candidate above is tried; none is dropped in advance. The judge decides whether each ships as assembly, as C, or not at all. The order puts shared techniques together and the cheapest evidence first.

| Step | Candidates | Why here |
|---|---|---|
| 1 | `np.clip` on floats, as C | Proves the delivery path with a kernel that needs no assembly. `np.clip` is backed by a ufunc, so it also tests the inner-loop patching mechanism |
| 2 | Mask selection and `nonzero` | Largest gap the compiler does not close; one compaction technique, two kernels; `compress_f32` is already a judge fixture. Uses scratch output |
| 3 | `argmax` / `argmin` on floats | The harder case: beating NumPy where it already has NEON. Already a judge fixture |
| 4 | `searchsorted` on float64, then `interp` | One search core serves `searchsorted`, `digitize` and `interp` |
| 5 | `cumsum`: integers exact, then floats in tolerance mode | Integers first because they need no tolerance |
| 6 | `histogram`, `bincount` | Share the several-count-tables technique |
| 7 | Masked assignment | Only `np.putmask` and `np.copyto` are reached until the indexing gap is resolved |
| 8 | `isin` with a small test set | Least certain gain over C |
| 9 | Remaining C kernels: `astype`, `np.sum` on uint8, `count_nonzero` on int64 and float64 | No assembly expected; can be done at any point |

## Results on x86-64

From the recorded run on the Intel Core i5-14600KF. NumPy uses its AVX2 loops on this machine. Where a figure moved by more than a tenth between that run and a second one taken straight after, both ends are given.

### What the numbers say

1. **Three ARM64 targets are not targets against naive C here.** NumPy has AVX2 loops for them.
   - `argmax` and `argmin` on floats run at 0.08 ns per element for `float32` and 0.17 for `float64`. The naive loop, which clang leaves scalar, takes 0.76. NumPy is 4x to 9x ahead of it in the caches and 2.4x to 5x beyond them, and a kernel has to beat NumPy's vector loop, not a scalar one.
   - `np.clip(x, lo, hi, out=out)` is 1.3x to 1.8x behind while the data fits the second-level cache and level from 1,048,576 elements up. It was 3.5x to 7x behind on ARM64.
   - `astype` from `uint8` to `float32` is level (0.09 ns per element against 0.09). It was 5.5x behind on ARM64. `count_nonzero` on `int64` is level to 1.8x behind, where it was 2.9x.
2. **The loops NumPy leaves scalar are the same ones as on ARM64.** Masked assignment costs NumPy 2.8 to 3.1 ns per element, mask selection 3.2 to 3.6, `histogram` 3.8, `cumsum` 1.4 to 1.7, `isin` with eight test values 4.6 to 6.3, `bincount` 0.66. Naive C is 2.2x to 27x ahead on each of them at 65,536 and 1,048,576 elements.
3. **clang closes more of the gap here.** It vectorises the masked assignment loop, which it left scalar on ARM64: 0.11 to 0.17 ns per element against NumPy's 3, a gap of 17x to 27x in the caches and 7x beyond them. It also vectorises `np.where(c, a, b)`, where NumPy takes a branch per element: 4.5x at 1,048,576 elements and 10x at 65,536, on an operation that was level on ARM64.
4. **`np.sum` on `uint8` is further behind than on ARM64:** 4x to 5.7x from 65,536 elements up, 0.23 to 0.26 ns per element against 0.05. `np.sum(mask)` is the same loop (2.6x to 3.4x behind), and ten times slower than `np.count_nonzero(mask)` at 0.025.
5. **The search is a contest with NumPy, not with naive C.** For `searchsorted` on `float64` the naive binary search is level with NumPy up to 65,536 elements (70 ns per needle each) and behind it above: 179 against 104 at 1,048,576. `interp` is level to 1.4x behind. NumPy's `int64` search does the same job in 10 ns per needle at 65,536 and 33 at 1,048,576, which is the figure a kernel is measured against.
6. **Where NumPy already has vector code it is far ahead, as on ARM64.** `np.max` (0.06 ns per element), `np.sum` on floats (0.12 to 0.15), `count_nonzero` on booleans (0.025), `any`, `all` and `packbits` (0.01 to 0.02) are 1.5x to 20x ahead of naive C.
7. **Per element, NumPy is 1.2x to 4x faster here than on the M1 Pro** on the loops it leaves scalar (`cumsum` on `float64` 1.7 against 3.6 ns, `histogram` 3.8 against 6.3, `interp` 164 against 679 per point), and no faster on `sum` over `uint8` (0.23 to 0.26 against 0.19) and `count_nonzero` on `float64`. The naive loops are not uniformly faster either: the float `cumsum` chain is 2.6x faster (0.38 against 0.97) and `argmax` 1.7x, `histogram` is no faster, and the binary search takes twice as long at 1,048,576 elements, where the table no longer fits this machine's second-level cache. So the gaps are of the same order as on ARM64 for the scalar loops, and smaller wherever NumPy brought AVX2.
8. **At 16,777,216 elements the rows that allocate measure the allocator.** The C side gets a new 128 MB block on every call and takes a page fault for every 4 KB of it: 32,769 per call, counted. NumPy asks Linux for large pages and takes 4,562. NumPy is then 2x ahead on `clip`, `astype` and `np.diff` whatever the loop does. The `out=` rows at that size are the ones to read: `clip` is level, `cumsum` is 2.2x to 3.9x behind.

### Where the output sits

One finding is about the machine, not about any operation. An x86-64 processor decides whether a load has to wait for an earlier store by comparing only the address bits within a 4 KB page. If the output buffer starts a few bytes after the input, counted within the page, each store looks like a store to the elements about to be loaded, and the loop stalls.

Measured apart from the survey's rows: one block of memory, `x` at its start, `out` placed the given number of bytes further on within the page, 1,048,576 `float64` elements, ns per element.

| `out` starts this many bytes after `x`, within the page | `np.clip(x, lo, hi, out=out)` | `np.add(x, 1.0, out=out)` | Naive C `clip` |
|---|---|---|---|
| 0 | 0.24 | 0.24 | 0.25 |
| 8 | 1.88 | 0.49 | 0.39 |
| 16 | 1.86 | 0.47 | 0.35 |
| 32 | 0.97 | 0.49 | 0.39 |
| 64 | 0.51 | 0.51 | 0.38 |
| 128 | 0.29 | 0.29 | 0.40 |
| 256 | 0.25 | 0.24 | 0.25 |
| 2,048 | 0.24 | 0.23 | 0.25 |
| 4,088 | 0.24 | 0.24 | 0.25 |

- **It is the common case for power-of-two sizes.** Two arrays of 8 MB made one after the other come from the same heap once a program has freed one large array, and then sit 8 MB and 16 bytes apart: 16 bytes within the page. A first recorded attempt had every `out=` candidate in that position at 65,536 and 1,048,576 elements, and showed `np.clip` 4x to 5x behind naive C where it is level. Large blocks that come straight from the system are page-aligned and sit at distance 0, which is harmless.
- **What the survey does about it** is in the [method](#method): the output buffer is placed half a page from the input.
- **What it means for the kernels.** A kernel that reads one buffer and writes another meets this in a user's program. One that does not slow down in that position would be several times ahead of NumPy there on `clip`, where it is otherwise level. That is a candidate for [open issue 1](x86-64-design.md#candidates-that-apply-to-every-kernel). The judge now places its timing buffers and times this position in a table of its own: [judge-design.md](judge-design.md#where-the-output-buffer-sits).
- **Where it shows.** The judge's sweep adds what the table above could not: for a compiled C running sum the cost is 5x to 7x at 262,144 and 1,048,576 `int64` elements, where the data lies in the third-level cache, and nothing where it fits the second-level cache or is larger than all of them.

### Shortlist

The same kernels as on ARM64, in the order they are worth on this architecture. The numbers are measured. The last column is a judgement.

| # | Candidate | NumPy vs naive C | ns per element at 1,048,576: NumPy / C | Does clang vectorise it? | What it means for a kernel |
|---|---|---|---|---|---|
| 1 | `np.sum` on `uint8` | 4x to 5.7x behind | 0.26 / 0.05 | Yes | The C loop adds about four bytes per clock cycle. One instruction can add 32, so there is room above the C as well |
| 2 | Mask selection, `a[mask]` | 14x to 16x behind the branchless loop in the caches at 50% density, 3.4x to 4.1x at 5%. 1.1x to 1.4x behind the branchy loop | `float64`, 50%: 3.35 / 0.24 (branchless) | No | NumPy pays for a mispredicted branch on every other element. The compaction technique applies unchanged |
| 3 | `nonzero` on booleans | 1.8x to 2x behind the branchless loop in the caches at 50%, 1.8x to 3.6x at 5% | 50%: 0.42 / 0.21; 5%: 0.69 / 0.19 | No | Same compaction, emitting indices |
| 4 | Masked assignment | 17x to 27x behind in the caches, 7x beyond | 3.05 / 0.17 | Yes | The compiler takes most of it. Whether assembly adds to the C is open |
| 5 | `cumsum` with `out=` | `int64` 5.6x to 7.6x behind in the caches, floats 4.3x to 4.5x; 2.2x to 3.9x at 16,777,216 | `int64`: 1.45 / 0.24; `float64`: 1.69 / 0.38; `float32`: 1.63 / 0.38 | No | The scalar chain is already one addition per element. A vector prefix sum has to beat 0.19 ns for `int64` |
| 6 | `histogram`, uniform bins | 3.9x to 4.1x behind from 65,536 elements up | 3.76 / 0.97 | No | Bin numbers in vectors, counting stays scalar |
| 7 | `bincount` | 2.2x to 2.5x behind | 0.66 / 0.27 | No | Several count tables. Small gain over C expected |
| 8 | `isin` with a small test set | 2.2x to 3x behind | 6.32 / 2.08 | Yes, the inner compare | Table lookup for close test values, as on ARM64 |
| 9 | `searchsorted` on `float64`, `interp` | Level up to 65,536, NumPy 1.7x to 2.5x ahead above | 104 / 179 per needle; `interp` 164 / 175 per point | No | Measured against NumPy. Its `int64` search at 33 ns per needle shows what is reachable |
| 10 | `count_nonzero` on `float64` | 5.8x to 9.7x behind at 65,536, 3.6x to 3.9x at 1,048,576, 1.8x at 16,777,216 | 0.39 / 0.10 | Yes | Likely ships as C |
| 11 | `count_nonzero` on `int64` | Level to 1.8x behind | 0.13 / 0.11 | Yes | NumPy has a vector loop. Little to gain |
| 12 | `argmax` / `argmin` on floats | NumPy 2.4x to 9x ahead | `float32`: 0.08 / 0.76; `float64`: 0.17 / 0.76 | No | The opponent is NumPy's AVX2 loop. Its `float64` loop runs at half the speed of its `float32` one, and both are slower than its own `np.max` (0.06), so some room is left |
| 13 | `np.clip` on floats | 1.3x to 1.8x behind in the second-level cache, level above | `float32`: 0.13 / 0.13; `float64`: 0.27 / 0.26 | Yes | Ships as C if at all, unless the output sits just behind the input ([above](#where-the-output-sits)) |
| 14 | `astype`, `uint8` to `float32` | Level | 0.09 / 0.09 | Yes | Nothing to install is the expected verdict |

Also measured:

- **`np.where(c, a, b)`** is 4.5x to 10x behind naive C while its result fits the caches. It is not one of the kernels and was level on ARM64. It would be a C kernel.
- **`np.take(a, idx)`** is level with naive C here; it was 3.2x behind on ARM64. With `out=` it is still 1.2x to 6x behind.
- **`np.add.at`** (2x behind), **`np.minimum.accumulate`** (up to 1.7x) and **`np.dot` on `int64`** (1.2x to 1.7x) are behind by about what they were on ARM64.
- **Stepped slices** are level with naive C or ahead of it, as on ARM64.

### Per-call overhead

NumPy's time on a one-element array. The naive C kernel takes 1 to 6 ns here, and more where it clears a table first.

| Fixed cost | Calls |
|---|---|
| 10,000 to 16,500 ns | `np.histogram`, `np.isin` |
| 980 to 1,770 ns | `np.clip`, `np.sum` on `uint8` and booleans |
| 590 to 860 ns | `np.sum` on floats, `np.max`, `np.cumsum`, `np.any`, `np.all`, `np.array_equal`, `np.diff` |
| 390 to 580 ns | `np.argmax`, `np.argmin`, `np.searchsorted`, `np.interp`, `np.add.at`, `np.dot`, `np.where`, `np.take` |
| 170 to 330 ns | `np.nonzero`, `np.bincount`, `np.packbits`, `astype`, `x[mask] = v`, `np.copyto`, `x[mask]` when it selects something |
| 55 to 100 ns | `np.count_nonzero`, `a[idx]`, `a[idx] = v`, `x[mask]` when it selects nothing |

These lie between level with the M1 Pro's and a third below them, and fall into the same groups. The budget for the binding is the same: well under 100 ns per call.

### Order of work

The order proposed in [x86-64-design.md](x86-64-design.md#the-survey) stands. The run confirms it and moves nothing.

| Step | Candidates | Why here |
|---|---|---|
| 1 | `np.sum` on `uint8` | The largest gap that holds at every size, one instruction does the work, and it proves the pipeline |
| 2 | Mask selection and `nonzero`, then masked assignment | The largest gaps the compiler does not close; one way of handling a mask, four kernels |
| 3 | `cumsum`: `int64`, then the floats | Large gap, and the first kernels that write a buffer as long as the one they read, so the first to meet the [placement](#where-the-output-sits) question |
| 4 | `searchsorted` on `float64`, both sides, then `interp` | One search core. NumPy's `int64` path sets the mark |
| 5 | `histogram`, `bincount` | Share the several-count-tables technique |
| 6 | `isin` | Depends on the table lookup carrying over |
| 7 | `count_nonzero` on `float64` and `int64` | Likely C, or nothing for `int64` |
| 8 | `argmax` and `argmin` on floats, `clip` on floats | NumPy has AVX2 here. Each needs a verdict; a win is not expected outside the caches |
| 9 | `astype` from `uint8` to `float32` | Expected verdict: not installed |

## How the kernels get built

The survey lands in its own pull request. The kernels follow in separate ones, because the judge refuses to run a kernel whose `spec.py` or `naive.c` differs from the main branch:

1. **This pull request:** the survey harness, the results and this document.
2. **Specs and baselines:** `kernels/<name>/spec.py` and `naive.c` for the candidates, reviewed by a person, since everything is measured against them.
3. **One pull request per kernel or kernel family:** the assembly from `/goal` runs against the judge, with its `attempts.md` and recorded result. A candidate where assembly does not beat the C ships the C.

## Known gap: indexing syntax

`a[mask]`, `x[mask] = v` and `x.astype(...)` are indexing syntax or array methods. They are not module functions and not ufuncs, so none of the three patching mechanisms in the core design obviously reaches them. Their function forms are reachable: `np.compress`, `np.extract`, `np.putmask` and `np.copyto`.

This limits what three shortlist entries are worth until it is resolved:

| Candidate | Reached today | Not reached |
|---|---|---|
| Mask selection | `np.compress(mask, a)`, `np.extract(mask, a)` | `a[mask]` |
| Masked assignment | `np.putmask(x, mask, v)`, `np.copyto(x, v, where=mask)` | `x[mask] = v` |
| `astype` C kernels | nothing | `x.astype(...)` |

The kernels are still worth building: the judge can measure them, and the function forms benefit at once. Whether the syntax can be reached is investigated during step 1 of the order of work. The gap is also recorded in [core-design.md](core-design.md) under phase 3.

## The baseline for mask selection

The branchy loop is the simplest C for `a[mask]`, but the branchless loop is just as short and 9x faster at 50% density, so the branchless loop is the honest baseline for mask selection and `nonzero`. It leaves a stray element just past the reported output, inside the buffer's capacity, and a kernel that stores a block at a time leaves up to a block of them. The judge's default output rule rejects that.

Decided in review: a spec may declare an output buffer as scratch, which lets the kernel write anywhere inside its capacity. The trade-offs, including what the stray last element means for the binding, are set out in [judge-design.md](judge-design.md#scratch-output-writing-inside-the-capacity). The judge implements it, with `compress_scratch_f32` as its fixture.

## Limits of this survey

- One machine and one run per architecture, one input distribution per candidate (two densities for the mask operations).
- On the x86-64 machine two runs taken one after the other agree to within 1% to 3% for the typical ratio. At 65,536 and 1,048,576 elements about a quarter of the ratios moved by more than a tenth, most of them candidates that allocate their result: there the allocator places the output, and the placement changes the time ([Where the output sits](#where-the-output-sits)). Three megabyte-sized arrays also come close to the 24 MB cache that all cores share, with a desktop session running.
- Where NumPy allocates its result, sizes whose result is tens of megabytes compare the two allocators as much as the two loops. On Linux that favours NumPy, which asks for large pages.
- At sizes up to 4,096 the C loop sees the same input on every repeat, so the branch predictor learns it. Ratios for the mask operations at small sizes flatter the C. The ranking uses sizes of 65,536 and up.
- NumPy's numbers include call overhead; the C numbers include no binding cost. At small sizes this flatters the C, as the judge design already notes.
- The last column of the shortlist is a judgement about what assembly can do. Nothing in this survey measures assembly.
- `histogram` was measured with 64 bins over [0, 1), where the naive bin calculation is exact. NumPy's rule for values on a bin edge needs care in a real kernel.
- The search candidates use random needles. Sorted or clustered needles behave differently and were not measured.

## Decisions made in review

| Question | Decision |
|---|---|
| Ship plain C where assembly only ties it? | Yes |
| Choose three kernels for phase 3? | No. All candidates are tried |
| Tolerance-mode float `cumsum`? | Acceptable. Float `cumsum` may reorder additions under a declared tolerance |
| Indexing syntax not reachable | Recorded as a known gap here and in the core design |
| May a kernel write inside its output capacity? | Yes, per buffer, when the spec declares it scratch. Strict by default; trade-offs in the judge design |
