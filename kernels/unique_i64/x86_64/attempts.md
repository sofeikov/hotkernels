# unique_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge, locked: `spec.py` and `naive.c` are on `main`. Other work was running on other cores of the machine, at a load average of 3 to 5, so sizes from 1,048,576 elements up are noisier than usual and single figures there are not to be trusted.

The ARM64 kernel and its log were read first. Its two paths are kept, a 64-bit set for at most 64 values and a table of bytes inside `out` above that, but the set is built differently: AVX2's `vpsllvq` gives zero for a count of 64 or more, where NEON's `ushl` reads only the low byte of the count, so an element far outside the range needs no test and no second reading of its chunk.

What the compiler does with `naive.c` (`clang -O3 -march=native`, clang 21.1.8): one store of eight bytes per element behind a compare and a branch. That is 0.15 to 0.16 ns per element from 4,096 to 262,144 elements, on two values and on ten, with ten or 256 values of range: 0.84 cycles, the store. At 16,777,216 elements 0.37 to 0.44 ns, where the 128 MB of input come from memory. With a range of 256 the table of 2 KB is cleared and read back, 61 ns at 16 elements. NumPy takes 10 ns per element at 262,144 elements, so the C is already about 55x ahead of it; what is left for assembly is the store per element.

## 1. A 64-bit set by `vpsllvq` for at most 64 values, a byte table inside `out` above that

- **Idea:** for `span <= 64`, value `lo + d` is bit `d` of a set. Four elements per vector: `vpaddq` of `-lo` from memory, `vpsllvq` of 1 by the result, `vpor` into one of four sets; sixteen elements per step, then single vectors, and a last vector that ends at `n` and overlaps the one before. A count of 64 or more shifts to zero, so nothing is tested; bits from `span` to 63 are dropped by `bzhi` at the end and the set is read out with `tzcnt` and `blsr`. Fewer than four elements go one at a time. For `span > 64`, the last `span` bytes of `out` are a table of one byte per value, marked with a byte store behind a compare and a branch, four elements per step, and read out 32 bytes per `vpmovmskb`. The mask of the table's last 32 bytes is taken before the first value is written, since the values written to the front reach those bytes before the steps do (proof at `.Lread_start` in `kernel.s`).
- **check:** pass
- **compare:** n/a (first version). `run`: 1.73x [1.72 – 1.75] over the naive C overall, 86.0x over NumPy.
- **Kept:** yes
- **Notes:** the set, default range: 2.9x to 4.2x from 1,024 to 16,384 elements (0.061 ns per element against 0.18 at 1,024), 3.7x at 16 elements of two values and 2.0x at 16 of ten, 1.25x to 1.32x from 4,194,304 up, where both read memory. The same code took 254 ns and 207 ns for 4,096 elements in the two distributions, and 5.8 µs and 3.8 µs for 65,536: the judge's two inputs sit differently against the cache lines. The byte table, range of 256: 5.2x to 5.9x at 16 elements (10 to 12 ns against 61), 1.3x at 1,024, 1.0x to 1.2x from 4,096 up, where both are one store per element. With the output 16 bytes behind the input: 1.02x of the placement apart overall, nothing to report.

## 2. No subtraction when the labels start at zero or above and end below 64

- **Idea:** attempt 2 of the ARM64 kernel, which pays more here. When `0 <= lo` and `lo + span <= 64`, an element is its own bit number, and `vpsllvq` takes its counts straight from memory: two instructions per four elements, a shift with a load in it and an or, where there were three. The set is shifted down by `lo` before `bzhi`. These are the ranges the binding passes for labels 0 and 1 or 0 to 9.
- **check:** pass
- **compare:** significantly faster, 1.030x [1.026 – 1.034]
- **Kept:** yes
- **Notes:** default range, both distributions: 1.17x at 64 elements, 1.20x to 1.25x from 256 to 4,096 (0.041 ns per element against 0.053 at 4,096), 1.04x to 1.09x from 16,384 to 262,144, level from 1,048,576 up. The range of 256 came out 0.96x to 0.97x although no instruction of the table path changed: its loops moved by 96 bytes. Where a loop lies costs this kernel a few percent either way (see attempt 4).

## 3. Loads aligned to 32 bytes from 32 elements

- **Idea:** a 32-byte load that straddles two cache lines costs a cycle more here, and an array from the allocator starts 16 bytes into a line, so half the loads of the set's loop straddled. A set does not mind an element read twice: the first vector is taken where `x` starts and the steps go on from the next multiple of 32 bytes. Only when `x` is a multiple of 8, so that a step still starts at an element, and only from 32 elements, since the extra vector costs more than it saves below that.
- **check:** pass
- **compare:** significantly faster, 1.056x [1.028 – 1.086]
- **Kept:** yes
- **Notes:** default range: 1.11x to 1.12x at 256 elements, 1.36x to 1.45x at 65,536 and 262,144 on the input that sat badly (3.5 µs against 5.4 at 65,536) and 1.14x to 1.17x, not significant, on the other; level at 16 elements, 0.98x at 16,384. The machine was busy and the intervals are wide.

## 4. Table path: sixteen elements tested together, then stored without a test each

- **Idea:** the table path's loop is a load, a subtraction, a compare, a branch and a byte store per element. Test sixteen at once instead (`vpaddq` of `2^63 - lo`, `vpcmpgtq` against `2^63 + span - 1`, the four results or'ed, `vptest`) and, when none is outside the range, store each at `table - lo + x` with nothing but its load in front; a step with an element outside goes one at a time as before.
- **check:** pass
- **compare:** significantly faster, 1.041x [1.032 – 1.050]
- **Kept:** no
- **Notes:** the gain is at one size: 1.16x to 1.17x at 4,096 elements with the range of 256, level at 16, 65,536 and above. Not kept, although the judge would have allowed it: attempt 5 takes the same inputs twice as fast without storing at all, and what is left for the table is data spread over the range, which the judge does not time. On that data, in a benchmark of my own on the same machine (uniform over 256 values), this loop was slower than the one with a branch per element: 0.180 against 0.168 ns per element at 4,096, 0.223 against 0.196 at 65,536, 0.247 against 0.234 at 1,048,576. The stores are the limit either way, about one per cycle when they go to different cache lines.

## 5. Table path: a 64-value window that follows the data, and no table while it holds

- **Idea:** one store per element is what the table costs, 0.77 cycles when the stores share a cache line and a cycle or more when they do not, against a third of a cycle for the bits path. So a 64-bit set takes the elements first, as on the bits path but for a window of 64 values somewhere in the range: `t = x - (window's first value)`, `1 << t` by `vpsllvq`, and the or of a chunk's sixteen `t` tested with `vptest` for a bit from 64 up. A chunk that passes is done: seventeen instructions for sixteen elements. A chunk that fails is marked in the table one element at a time; the bits it set are of values that are there, so they stay. A table is only made at the first failure; without one the values are read straight out of the set.
- **What a failed chunk leads to:** it is read for the smallest and largest of its values. Less than 64 apart, it is one cluster, and the window moves to around their middle, after the set has gone to the table; if the window it replaces had lasted no more than sixteen chunks, the clusters do not last. Wider than that, it is spread or holds a stray element; if the last failure was within sixteen chunks, or this is the first chunk of all, the data is spread. Data that does not suit a window, either way, is marked in the table for a stretch without trying it: 1,024 elements, and four times as much each time the chunk after a stretch fails wide again, up to 65,536. A failed chunk costs about 9 to 17 ns, which is what the window saves over some sixteen chunks; hence the sixteen.
- **check:** pass. Also outside the judge, against the naive C: 1,400,000 random calls of this form over 19 spans from 1 to 65,536 and random ones up to 600, sizes up to 20,000, five kinds of `lo` with both ends of int64 among them, and nine shapes of data (spread, one cluster, sorted, runs of clusters, a cluster with elements elsewhere in the range, with elements far outside it, spread then clustered), `x` and `out` each ending at an inaccessible page, one call in four with `x` not a multiple of 8: none wrong.
- **compare:** significantly faster, 1.386x [1.377 – 1.394]
- **Kept:** yes
- **Notes:** range of 256, both distributions, whose ten values lie in one window: 1.9x to 2.8x at 16 elements (3.8 and 6.9 ns against 10.4 and 12.3), 2.1x to 2.6x from 64 to 65,536 (0.068 ns per element against 0.146 at 4,096), 1.25x to 1.32x at 1,048,576 and 1.22x to 1.23x at 16,777,216, where the input comes from memory. The default range is untouched, 1.00x.
- **The give-up rule was revised before the attempt was kept.** Its first form moved the window after every failure, counted four failures in a row, each within sixteen chunks of the last, and then left the window out for 256 elements, doubling. `compare` gave that form 1.407x [1.398 – 1.416], and attempts 6 to 9 were first judged on top of it. A benchmark of my own then showed what it cost labels spread evenly over the range, which fail every chunk: at 256 elements over 256 values 114 ns, against 82 for the table alone (attempt 3) and 121 for the naive C; over 100 values 96 ns against 62 and 67, slower than the C. The form above takes 91 and 72 ns there. Judged as a change of its own on top of attempt 9, the revision came out `significantly slower`, 0.994x [0.991 – 0.996]: the judge's inputs never fail a chunk, so all it saw was three values written below the stack pointer on entry, which are now written at the first failure instead. A change the judge cannot see cannot be kept on its own, so the branch was taken back to attempt 3 and this attempt, 7 and 9 were judged again in their final form, each against the one before; the figures in this log are from those runs.
- **What the judge cannot see:** both timed inputs of the range of 256 are ten values next to one another. See "Where it stands" for other data.

## 6. Table path: the window's loads aligned to 32 bytes

- **Idea:** attempt 3 for the window's chunks: the first chunk where `x` starts, the rest from a multiple of 32 bytes, from 128 elements.
- **check:** pass
- **compare:** no significant difference, 0.994x [0.980 – 1.009]
- **Kept:** no
- **Notes:** not significantly faster. Judged on top of the first form of attempt 5, whose chunk loop is the one kept. The window's chunk is seventeen instructions on the three vector ports for four loads, so a load that waits a cycle is not what holds it up, as it was for the two instructions per load of attempt 2.

## 7. Table path: a window that starts at the value 0 needs no subtraction

- **Idea:** attempt 2 for the window. When the window's first value is 0, `t` is the element itself: the chunk loads its four vectors and shifts, thirteen instructions for sixteen elements where there were seventeen. The first window is put there when the first element is from 0 to 63 and 0 is in the range, and a window moved after a failure is put there when the failed chunk's values are all from 0 to 63. Only when `lo` is at most 0, so that the window never starts below it. These are the labels the binding passes: counted from 0, with a range wider than 64.
- **check:** pass. 400,000 random calls against the naive C, none wrong.
- **compare:** significantly faster, 1.055x [1.044 – 1.067]
- **Kept:** yes
- **Notes:** range of 256, both distributions: 1.11x and 1.34x at 16 elements, 1.10x to 1.14x at 256, 1.28x at 4,096 (0.053 ns per element against 0.068), 1.10x at 65,536, not significant there, 1.01x at 1,048,576 and 1.03x at 16,777,216. The default range is untouched.

## 8. Bits path: eight vectors per step

- **Idea:** the loop of attempt 2 takes 0.21 cycles per element in the first-level cache (0.040 ns at 4,096 elements, at 5.29 GHz) where two instructions per vector on three ports would allow 0.17. Halve the loop's own add, compare and branch per element: eight vectors per step into the same four sets.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.993 – 1.004]
- **Kept:** no
- **Notes:** not significantly faster. In a benchmark of my own the loop also came out the same aligned to 32 bytes, and with the four shifted vectors or'ed in pairs before they reach two sets: 0.040 ns per element at 4,096 each time. At 256 elements, where the whole input is in the cache from the last call, it is 0.035 ns per element, 0.19 cycles; the loop is at what the three ports give it.

## 9. A prefetch 2,048 bytes ahead from 64 KB of input

- **Idea:** from 16,384 to 262,144 elements the input is in the second-level cache and the bits path takes 0.053 ns per element, where it takes 0.040 in the first. Ask for the two cache lines 2,048 bytes on with `prefetcht0` in each step of sixteen. In the first-level cache the two instructions cost a tenth (0.045 ns per element at 4,096 in a benchmark of my own, with any distance, a distance of 0 included), so each of the four vector loops (the two of the bits path, the two of the window) has a second copy with the prefetch, taken from 65,536 bytes of `x`. A prefetch reads nothing: an address past the end of `x` is ignored.
- **check:** pass. 400,000 random calls against the naive C with sizes up to 20,000 elements, none wrong.
- **compare:** significantly faster, 1.030x [1.018 – 1.043]
- **Kept:** yes
- **Notes:** default range: 1.17x at 16,384 elements, 1.18x to 1.19x at 65,536 (3.0 µs against 3.5), 1.05x to 1.07x at 262,144, 1.06x to 1.09x at 4,194,304 and 16,777,216; 4,096 and below untouched. Range of 256: 0.91x to 0.92x at 16,384 (1.0 µs against 950 ns), level at 65,536, 1.13x to 1.14x at 4,194,304 and 1.10x at 16,777,216. Distances of 256, 512 and 1,024 bytes gained less in my benchmark (0.051, 0.050 and 0.047 ns per element at 131,072 against 0.046 for 2,048 and 0.053 without); one line per step instead of two gained half. `prefetcht0` is the one instruction here whose worth depends on the machine's caches: the size it starts at is this processor's 48 KB first-level cache rounded up.

## 10. The window's loops prefetch only beyond the second-level cache

- **Idea:** attempt 9 cost the range of 256 a twelfth at 16,384 elements: the window's chunk is thirteen instructions on the vector ports, the loads are not what holds it up inside the second-level cache, and the two prefetches are two instructions more. Start the window's prefetch at 2 MB of input, the size of that cache, and leave the bits path at 64 KB.
- **check:** pass
- **compare:** no significant difference, 1.004x [0.998 – 1.009]
- **Kept:** no
- **Notes:** not significantly faster. Range of 256: 1.07x to 1.08x at 65,536 elements, 1.03x at 16,384 inside a wide interval (0.89 to 1.19), level everywhere else. The slower size of attempt 9 stays: with a range of 256, 16,384 elements take about 1.0 µs where they took 950 ns without the prefetch.

## Where it stands

`uv run judge run unique_i64 --record`, locked, at a load average of 2.6 (the judge still calls that a noisy machine): 3.00x [2.97 – 3.04] over the naive C and 148x over NumPy across the four timed cases. Default range: 3.3x to 5.6x over the C from 256 to 65,536 elements (0.041 ns per element at 4,096, 0.046 at 65,536), 2.1x to 3.8x at 16, 1.5x to 1.9x at 1,048,576 inside wide intervals, 1.4x to 1.6x at 16,777,216, where both read memory. Range of 256: 12x and 19x at 16 elements (3.4 and 5.2 ns against 65 and 62, the C's table of 2 KB), 3.4x to 4.1x at 1,024, 2.9x to 3.0x at 4,096, 2.1x to 2.7x from 16,384 to 262,144, 1.3x to 1.4x at 1,048,576. Against NumPy at 1,048,576 elements, 70x to 83x. No size of any timed case is slower than the C.

With the output 16 bytes behind the input: 1.01x [1.00 – 1.02] of the placement apart overall, not significant; 0.93x at 262,144 elements (19.3 µs against 18.0) and level or faster at the six other sizes. The kernel stores nothing while it reads `x` on the timed data, so there is nothing for a load to wait for.

From Python, `uv run delivery time --only unique` at a load average of 2.4 to 2.6, with this kernel and with the naive C it replaces: at 4,096 elements 835 ns against 1.49 µs, at 1,048,576 elements 279 µs against 374 µs, at 16 elements 93 ns against 98. The binding reads the array once for its smallest and largest elements before it calls the kernel, and at a million elements that read is now the larger half of the call.

**Data the judge does not time**, range of 256 unless said, in a benchmark of my own on the same machine at a load average of 2.4 (the smallest of three runs; the kernel, the table alone as in attempt 3, the naive C):

| Labels | Elements | Kernel | Table alone | Naive C |
|---|---|---|---|---|
| Uniform over all 256 values | 16 | 15.9 ns | 12.4 ns | 86.5 ns |
| | 256 | 91.4 ns | 82.1 ns | 121 ns |
| | 4,096 | 704 ns | 779 ns | 814 ns |
| | 1,048,576 | 235 µs | 237 µs | 305 µs |
| Uniform over 100 values, range of 100 | 16 | 14.4 ns | 11.4 ns | 38.8 ns |
| | 256 | 71.7 ns | 61.8 ns | 67.5 ns |
| | 4,096 | 639 ns | 615 ns | 742 ns |
| | 1,048,576 | 212 µs | 212 µs | 275 µs |
| Sorted, every value equally often | 256 | 134 ns | 104 ns | 108 ns |
| | 1,024 | 224 ns | 216 ns | 217 ns |
| | 4,096 | 533 ns | 653 ns | 652 ns |
| | 65,536 | 5.6 µs | 9.9 µs | 9.6 µs |
| Ten neighbouring values, one element in 500 elsewhere | 256 | 28.0 ns | 49.9 ns | 101 ns |
| | 4,096 | 339 ns | 600 ns | 647 ns |
| | 65,536 | 5.6 µs | 9.9 µs | 11.2 µs |

Two rows are slower than the naive C: 100 evenly spread values at 256 elements, by 6%, and sorted labels at 256 elements, one of each value, by 24% (3% at 1,024). Both are a failed chunk or two before the kernel gives the window up. Spread labels are otherwise a store each, at the speed of the table alone, which is 1.2x to 1.3x over the C from 4,096 elements. A set of two 64-bit words would take 65 to 128 values in about half that time, by attempt 5's arithmetic (six instructions per four elements), but `compare` has no input that would show it, so it was not built.

No `pext`, `pdep`, gather or masked store is used. `prefetcht0` is (attempt 9).
