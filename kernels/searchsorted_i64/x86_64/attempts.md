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
