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
