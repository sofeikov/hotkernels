# searchsorted_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. Every run is `--unlocked`: the spec and the reference came in the same pull request as the kernel and were not on `main` to lock against. They were committed before the first line of assembly and not changed after.

What the compiler does with `naive.c` (`clang -O3 -march=native`): one search at a time with a branch per step. The judge times the same needles again and again, so with 64 needles the branches are learnt and the naive C takes 5 to 15 ns per needle up to 1,048,576 elements; with 1,000 needles they are not, and it takes 30 ns per needle at 1,024 elements and 139 ns at 1,048,576. NumPy's int64 search is about three times faster than its float64 search: 6.5 ns per needle at 1,024 elements and 24 ns at 1,048,576 with 1,000 needles.

## 1. Baseline: the float64 kernel with a signed integer compare

- **Idea:** `searchsorted_f64` as it stands after its x86-64 attempt 3, which is where that log's measurements led: a search is a fixed number of steps that depends only on n (`half = (len + 1) / 2; if a[lo + half - 1] < needle: lo += half; len -= half`), up to 64 searches in step with their positions kept in `out` and their needles in `v`, 512 in step from 262,144 elements, four searches per `vpgatherqq`. The compare is `vpcmpgtq` with the needles as its first operand, which must be a register, so the needles are loaded where the float64 kernel compares against memory. There is no NaN: the test of the needles for NaN and the pass that fixes their answers are gone. One to three needles are searched one at a time in general registers with `cmovg`.
- **check:** pass
- **compare:** not run (first version). `run`: 4.69x over naive C overall (64 needles 2.46x, 1,000 needles 8.96x), 3.11x over NumPy (4.37x and 2.21x).
- **Kept:** yes
- **Notes:** about 15% slower than the float64 kernel at 64 to 16,384 elements (132.6 ns for 64 needles at 1,024 elements, 113.4 there). `vpcmpgtq` runs on one port of this processor with a latency of three cycles, and the float compare on two. Against NumPy with 1,000 needles the kernel is 3.0x to 3.5x up to 4,096 elements and 1.2x from 1,048,576.

## 2. Two steps of each search in one pass over the chunk

- **Idea:** a pass loaded four positions, took one step and stored them. With the gather counted at about thirteen instructions' worth of the processor's front end, a step of four searches was about 23, of which the load of the positions, the load of the needles, the store and the loop are five that do not depend on how many steps the pass takes. Two steps in a pass pay them once: the positions and the needles stay in registers between the two gathers. From 524,288 elements a pass takes one step as before.
- **check:** pass
- **compare:** significantly faster, 1.089x [1.087 – 1.092]
- **Kept:** yes
- **Notes:** 1.15x to 1.18x up to 4,096 elements for both needle counts, 1.05x to 1.07x at 16,384 to 262,144, nothing above. Tried first with two steps at every size: 1.072x [1.066 – 1.078] overall, but 0.95x, 0.91x and 0.88x at 1,048,576, 4,194,304 and 16,777,216 with 1,000 needles. There the probes miss the second-level cache, and the second gather of a pass waits for the first, so fewer misses are in flight; hence the threshold. Three and four steps in a pass were slower than two in a benchmark outside the judge (1,000 needles against 1,024 elements: 1.71 us with two, 2.03 with three, 2.32 with four, 1.98 with one): the gathers of a pass form a chain, and the instructions that wait for it fill the scheduler.

## 3. The first pass takes two or three steps without a gather

- **Idea:** the first step already needed no gather, since every search probes `a[half - 1]`. The second probes one of two elements and the third one of four, all known before the pass. The probe is chosen with the masks of the compares before it: `e0 ^ (m1 & (e0 ^ e1))` for the second step, and the same twice over for the third, with the differences of the candidates computed once per chunk. That is two instructions for the second probe and six for the third, against a gather. The number of steps is the number of bits of n; the first pass takes three when it is odd and two when it is even, so the passes with gathers are always pairs.
- **check:** pass
- **compare:** significantly faster, 1.064x [1.060 – 1.069]
- **Kept:** yes
- **Notes:** 1.22x at 16 elements, where two of five steps lost their gather, 1.16x at 64, 1.09x at 1,024, 1.05x at 65,536, and 1.01x to 1.02x from 1,048,576, where the first steps are the ones that stay in the cache. In a benchmark outside the judge the first two steps alone gave 9% at 16 elements and 3% at 1,024 with 1,000 needles, and the third step 4% and 2% more.

## 4. Chunks of 256 needles while `a` fits the second-level cache

- **Idea:** the float64 kernel settled on 64 searches in step when a pass took one step and its measure was gathers in flight. Each chunk pays for its halves, the candidates of its first pass and the ends of its loops; with 256 needles in a chunk that is paid a quarter as often, and the positions and needles of a chunk are still only 4 KB of the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.008 – 1.011]
- **Kept:** yes
- **Notes:** a call with 64 needles is one chunk either way and does not change. With 1,000 needles 1.04x at 16 elements, 1.03x at 1,024, 1.01x at 16,384 and nothing from 65,536. In a benchmark outside the judge 128 needles gave half of that, and 32 needles in a chunk was 9% to 27% slower than 64.

## 5. Two steps in a pass for the early steps of a large array

- **Idea:** from 524,288 elements every pass takes one step, because the probes of the last steps miss the second-level cache. The first steps of a search probe few different elements, which stay in the cache whatever the size of `a`; so take two steps in a pass while `len` is at least `n >> 14` (at most 16,384 different elements probed), and one step after that.
- **check:** pass
- **compare:** significantly slower, 0.980x [0.978 – 0.981]
- **Kept:** no
- **Notes:** unchanged below 524,288 elements, as it must be. Slower at every size above: with 64 needles 0.97x, 0.96x and 0.94x at 1,048,576, 4,194,304 and 16,777,216, with 1,000 needles 0.92x, 0.87x and 0.94x. The early probes of an array of 8 MB or more are few, but each is on a page of its own; the guess is that the second gather of a pass waits longer for the first there, as it does when the first misses the cache. Not looked into further: counters are not available on this machine.

## 6. A pass unrolled over two vectors of searches

- **Idea:** eight searches per iteration of a pass, so the loop's counter and branch are paid once for two vectors.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.005 – 1.008]
- **Kept:** no, against the letter of the rule; for the reviewer to decide. The same change to `searchsorted_right_i64` made no significant difference (1.002x [0.999 – 1.004]), and the two kernels are one source. Keeping it here alone would make them differ in the shape of their main loop for six parts in a thousand.
- **Notes:** 0.99x at 16 and 1,024 elements with 64 needles, 1.01x from 65,536; with 1,000 needles 1.00x at 16 elements and 1.01x above. A pass with an odd number of vectors needs a second copy of its body.

## Measured outside the judge

A benchmark of its own (1,000 needles against 1,024 elements unless said, one pinned performance core at 5.28 GHz, the minimum of 40 samples) was used to find what limits the loop, and to rule out three other shapes before any was given to the judge. None of these is a judged attempt.

- **What a step costs.** The baseline took 0.193 ns per needle and step: 4.1 cycles per vector of four searches. A loop of gathers alone takes 2.2 to 2.4 cycles per gather. The same loop with the gather replaced by a load takes 2.1 cycles, and removing any one other instruction from it changes nothing; an empty loop takes 1.1. So the loop is limited by the number of instructions the front end delivers, the gather counts for about thirteen of them, and nothing is gained by shortening a dependency. Larger chunks did not change the cost of a step, so it is not the ends of the loops either.
- **Probes loaded one at a time and put together** (`vmovq`, `vpinsrq`, `vinserti128`, from four scalar loads of the positions) in place of the gather: 7% faster than the baseline's gather with one step in a pass (1.84 us against 1.98), limited by loads then, ten per vector. As the first of two steps in a pass, with a gather for the second, 3% faster for the left side (1.65 us against 1.71) and nothing for the right. Not taken further: four more loads and two more registers for a gain the second step does not share.
- **No vectors at all:** positions in `out`, one search at a time in general registers, two to four steps in a pass, `cmp` against memory and `cmov`. Three instructions per step when the position is kept as a pointer, where the gather form has about four and a half per search. 2.20 us with four steps in a pass, 2.08 with three, 2.14 with two, 2.73 with one, against 1.71 for two gathers in a pass. Each step waits eight cycles for the one before it, and the scheduler fills with instructions that wait.
- **A fourth step without a gather** was not tried: it selects one of eight candidates, about fourteen instructions against the eighteen of a step with a gather.
- **Where the buffers sit.** With `out` 16 bytes behind `v` within the page the benchmark is 4% slower at 4,096 elements with 64 needles (130.8 ns against 125.8), and with `out` or `v` 16 bytes off a cache line 3% and under 1%. Where `out` sits against `a` made no difference.

## Where it stands

Final `judge run`, recorded, at a load average of 0.7:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 42.0 ns | 526.7 ns | 12.5x [12.46 – 12.62] | 678.3 ns | 16.2x [16.15 – 16.17] |
| 64 | 63.6 ns | 976.2 ns | 15.3x [15.31 – 15.44] | 763.3 ns | 12.0x [12.00 – 12.01] |
| 256 | 91.8 ns | 308.1 ns | 3.36x [3.35 – 3.36] | 834.4 ns | 9.09x [9.08 – 9.10] |
| 1,024 | 118.3 ns | 362.4 ns | 3.06x [3.06 – 3.06] | 912.7 ns | 7.71x [7.71 – 7.72] |
| 4,096 | 130.5 ns | 393.1 ns | 3.01x [3.01 – 3.02] | 984.5 ns | 7.55x [7.54 – 7.55] |
| 16,384 | 174.5 ns | 454.2 ns | 2.60x [2.60 – 2.61] | 1.1 us | 6.11x [6.10 – 6.11] |
| 65,536 | 238.4 ns | 518.9 ns | 2.18x [2.17 – 2.18] | 1.2 us | 4.93x [4.93 – 4.94] |
| 262,144 | 307.3 ns | 583.7 ns | 1.90x [1.90 – 1.90] | 1.3 us | 4.09x [4.08 – 4.09] |
| 1,048,576 | 885.8 ns | 895.1 ns | 1.01x [1.01 – 1.01] | 1.8 us | 2.07x [2.06 – 2.07] |
| 4,194,304 | 1.5 us | 2.0 us | 1.31x [1.31 – 1.32] | 2.5 us | 1.67x [1.67 – 1.67] |
| 16,777,216 | 2.8 us | 3.5 us | 1.23x [1.21 – 1.26] | 4.0 us | 1.43x [1.38 – 1.45] |
| overall |  |  | 2.86x [2.86 – 2.87] |  | 5.11x [5.10 – 5.12] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 606.3 ns | 7.2 us | 11.9x [11.83 – 11.90] | 3.1 us | 5.11x [5.11 – 5.12] |
| 64 | 954.1 ns | 14.6 us | 15.3x [15.24 – 15.29] | 4.2 us | 4.45x [4.44 – 4.45] |
| 256 | 1.3 us | 21.7 us | 16.8x [16.77 – 16.85] | 5.4 us | 4.18x [4.16 – 4.18] |
| 1,024 | 1.6 us | 29.6 us | 18.2x [18.14 – 18.27] | 6.5 us | 4.03x [4.02 – 4.04] |
| 4,096 | 1.9 us | 20.3 us | 10.4x [10.37 – 10.49] | 7.7 us | 3.97x [3.97 – 3.97] |
| 16,384 | 3.1 us | 30.3 us | 9.90x [9.84 – 9.95] | 9.0 us | 2.93x [2.93 – 2.93] |
| 65,536 | 4.2 us | 42.5 us | 10.00x [9.97 – 10.03] | 10.7 us | 2.51x [2.51 – 2.52] |
| 262,144 | 6.8 us | 61.5 us | 9.06x [9.02 – 9.09] | 13.5 us | 2.00x [1.99 – 2.00] |
| 1,048,576 | 13.1 us | 134.8 us | 10.3x [10.26 – 10.40] | 18.9 us | 1.45x [1.45 – 1.45] |
| 4,194,304 | 30.4 us | 221.9 us | 7.30x [7.10 – 7.49] | 35.7 us | 1.18x [1.15 – 1.19] |
| 16,777,216 | 54.8 us | 437.8 us | 7.98x [7.94 – 8.02] | 66.3 us | 1.21x [1.20 – 1.21] |
| overall |  |  | 11.1x [11.06 – 11.12] |  | 2.65x [2.65 – 2.66] |

All distributions: 5.63x [5.62 – 5.64] over naive C, 3.68x [3.68 – 3.69] over NumPy

With the output buffer 16 bytes behind its input (64 needles; "apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 181.9 ns | 131.2 ns | 0.72x [0.72 – 0.72] | 397.4 ns | 2.18x [2.18 – 2.19] |
| 16,384 | 178.3 ns | 179.1 ns | 1.00x [1.00 – 1.01] | 452.8 ns | 2.54x [2.54 – 2.54] |
| 65,536 | 236.6 ns | 240.4 ns | 1.02x [1.02 – 1.02] | 519.9 ns | 2.20x [2.20 – 2.20] |
| 262,144 | 368.5 ns | 309.5 ns | 0.84x [0.84 – 0.84] | 581.6 ns | 1.58x [1.58 – 1.58] |
| 1,048,576 | 647.6 ns | 631.0 ns | 0.97x [0.97 – 0.98] | 699.1 ns | 1.08x [1.08 – 1.08] |
| 4,194,304 | 1.5 us | 1.5 us | 0.99x [0.99 – 0.99] | 2.0 us | 1.29x [1.28 – 1.29] |
| 16,777,216 | 3.0 us | 2.6 us | 0.88x [0.86 – 0.92] | 3.5 us | 1.16x [1.13 – 1.22] |
| overall |  |  | 0.91x [0.91 – 0.92] |  | 1.63x [1.63 – 1.64] |

No stall that the placement explains. The two figures that stand out, 0.72x at 4,096 elements and 0.84x at 262,144, did not repeat: two more runs straight after gave 131.2 ns against 130.9 and 131.4 against 131.1 at 4,096, and 308.6 against 309.7 and 311.2 against 320.9 at 262,144. The same two sizes move in the main table from one run to the next with the output clear (262,144 elements: 307.3 ns here, 384.2 and 382.9 in those two runs; `searchsorted_right_i64` took 149.1 ns at 4,096 in its recorded run and 132.8 in a `compare`). At 4,096 elements `a` is 32 KB of a 48 KB first-level cache and at 262,144 it is all of the second-level cache, so which probes hit depends on where the judge's buffers were allocated in that process.

What holds it:

- **In the caches:** the front end. A step with a gather is about eighteen instructions for four searches, thirteen of them the gather, and a pass adds five; 1,000 needles against 1,024 elements take 0.15 ns per needle and step, which is 0.8 cycles.
- **Against NumPy with many needles:** NumPy's int64 search is about three times faster than its float64 search, so the kernel is 4.0x to 5.1x over it up to 4,096 elements with 1,000 needles, 2.0x at 262,144 and 1.2x from 4,194,304, where both wait for memory.
- **Beyond the second-level cache:** the misses, as in the float64 kernel. One step in a pass and 512 searches in step from there.
- **64 needles against a large array:** 1.0x to 1.3x over the naive C from 1,048,576 up. The judge repeats the same 64 needles, so the naive C's branches are learnt and its probes are in the cache; a caller's needles are new each time.

The candidates the design lists for this family (`docs/x86-64-design.md`):

| Candidate | Outcome |
|---|---|
| A gather loads four probes in one instruction, the compare and the move of the base stay in vectors | Kept (attempts 1 and 2), `vpgatherqq` |
| Scalar loads with `cmov`, sixteen or more searches interleaved | Measured outside the judge with the positions in `out`: slower than the gathers. Kept for one to three needles |
| Gathers are slow on older AMD processors and on Intel processors with the gather mitigation | Used, in every path from four needles up: from the fourth step of a search on, and at every step for the last needles of a count that is not a multiple of four. Untested elsewhere |
| Masked loads and stores for leftover elements | Not needed: a leftover is one more vector that overlaps the one before |
| Non-temporal stores; alignment of data | Not tried: the output is as small as the needles, and the probes are single elements |
| An output just behind the input | No cost in the judge's close table that repeats |
| Alignment of code | The loops of the passes start on a 32-byte boundary. Not varied |

**What differs from `searchsorted_f64` on x86-64.** The compare, and no pass for NaN needles. Then attempts 2 to 4, which do not depend on the element type: two steps in a pass, a first pass without a gather, and chunks of 256. They were not tried on the float64 kernels in this work.

**For a caller.**

- **One needle** is searched in general registers, about eight cycles a step. The judge times no call with fewer than 64 needles, so nothing here was tuned for it; from Python the call path is the cost.
- **From Python,** `uv run delivery time --only searchsorted` at a load average of 1.4: `np.searchsorted(a, f)` with one `np.int64` takes 264 ns against NumPy's 559 at 16 elements, 269 against 561 at 4,096 and 293 against 599 at 1,048,576. With as many needles as elements the call is 1.5x, 3.3x and 1.2x faster at those sizes, and with a million needles against 1,000 elements 1.9x.
- **Four to fifteen needles** are one to four vectors of searches, each a chain of gathers with nothing else in flight. By a script outside the repository, from Python, four needles against 65,536 elements and more are 1.00x to 1.14x of NumPy, where one needle is 1.4x and sixteen 1.25x to 1.4x. No count was slower than NumPy at any length from 16 to 16,777,216 elements, with random or sorted needles.
- **Unsorted input:** the result is meaningless, and the kernel stays inside its buffers by construction: `lo + len <= n` holds at every step for any data, and the candidates of the first pass are at fixed positions inside `a`.
- **`out` is used as working storage** during the call: an element holds the search's current position until its last step.

Left untried: the first gather of a pass replaced by loads put together, for the left side only; a fourth step without a gather; a prefetch of the next pass's probes for arrays beyond the caches.

Found here and worth trying on `searchsorted_f64` and `searchsorted_right_f64`: attempts 2 to 4.
