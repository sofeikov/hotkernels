# unique_i64 attempts (arm64)

All runs on an Apple M1 Pro (macOS, native), with `--unlocked`: the spec and naive C were new on this branch and not yet on `main`, and the branch's `pyproject.toml` and `uv.lock` carry the applications study's dependency group. Other agents were timing on the same machine, at a load average of about 2.5 to 3.5, so figures from about 100,000 elements up are noisier than usual.

No x86-64 machine was at hand when this log was written, and the kernel shipped as its naive C there. The x86-64 kernel was written afterwards and has a log of its own, `kernels/unique_i64/x86_64/attempts.md`: its set is built with AVX2's `vpsllvq`, which shifts each 64-bit lane by its own count and gives zero for counts of 64 or more, so it needs neither the test of a chunk nor the second reading below.

What the naive C costs here (Apple clang, `-O3`): about 0.24 ns per element from 1,024 elements up, on two values and on ten, with ten or 256 values of range: one store per element, unrolled eight times with a test and branch each. With a range of 256 the table of 2 KB is cleared by `bzero` and read back, about 110 ns at 16 elements.

## 1. A 64-bit set for at most 64 values, a byte table inside `out` above that

- **Idea:** for `span <= 64`, value `lo + d` is bit `d` of a set: NEON makes `1 << d` with `ushl` for two elements per vector, or'ed into four sets, and ors `d` itself into four more; over a chunk of 64 elements the or of the `d` is below 64 exactly when every `d` is, and a chunk where it is not is read again one element at a time with the exact test (`ushl` reads only the low byte of `d`). Bits from `span` to 63 are values outside the range and are masked off at the end; the set is read out with `rbit` and `clz`. Fewer than 64 elements go one at a time. For `span > 64`, the last `span` bytes of `out` are a table of one byte per value, marked with `strb` four elements per step, and read out eight bytes at a time, skipping words of zeros; the values written to the front never reach a table byte not yet read (proof in the header of `kernel.s`). A first version used `out` itself as a table of int64 flags, with a branch-free compaction: 0.62x to 0.91x of the naive C with a range of 256, since it cleared and read eight times the bytes.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.31x [1.31 – 1.31] over the naive C overall, 37.8x over NumPy.
- **Kept:** yes
- **Notes:** the set: 1.27x to 1.30x from 64 elements up (0.19 ns per element against 0.24 at a million), 1.20x at 16 elements of two values and 0.88x at 16 of ten. The byte table with a range of 256: 3.7x to 3.9x at 16 elements (30 ns against 112), 1.08x at 4,096, 0.98x to 0.99x from a million up, where both are one store per element.

## 2. No subtraction when the labels start at zero or above and end below 64

- **Idea:** when `0 <= lo` and `lo + span <= 64`, an element is its own bit number, so the vector steps drop their `sub`: twelve NEON instructions per eight elements where there were sixteen. The path runs with `lo = 0` and `span = lo + span`, its exact re-reading of a chunk included, and the set is shifted down by the real `lo` before it is read out. These are the ranges the binding passes for labels 0 and 1 or 0 to 9.
- **check:** pass
- **compare:** significantly faster, 1.100x [1.098 – 1.102]
- **Kept:** yes
- **Notes:** 1.19x to 1.25x on both distributions with the default range from 1,024 elements up (155 µs against 193 µs at a million, about 0.15 ns per element); unchanged with a range of 256, which takes the byte table.
