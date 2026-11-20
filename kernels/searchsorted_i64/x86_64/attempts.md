# searchsorted_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. Every run is `--unlocked`: the spec and the reference came in the same pull request as the kernel and were not on `main` to lock against. They were committed before the first line of assembly and not changed after.

What the compiler does with `naive.c` (`clang -O3 -march=native`): one search at a time with a branch per step. The judge times the same needles again and again, so with 64 needles the branches are learnt and the naive C takes 5 to 15 ns per needle up to 1,048,576 elements; with 1,000 needles they are not, and it takes 30 ns per needle at 1,024 elements and 139 ns at 1,048,576. NumPy's int64 search is about three times faster than its float64 search: 6.5 ns per needle at 1,024 elements and 24 ns at 1,048,576 with 1,000 needles.

## 1. Baseline: the float64 kernel with a signed integer compare

- **Idea:** `searchsorted_f64` as it stands after its x86-64 attempt 3, which is where that log's measurements led: a search is a fixed number of steps that depends only on n (`half = (len + 1) / 2; if a[lo + half - 1] < needle: lo += half; len -= half`), up to 64 searches in step with their positions kept in `out` and their needles in `v`, 512 in step from 262,144 elements, four searches per `vpgatherqq`. The compare is `vpcmpgtq` with the needles as its first operand, which must be a register, so the needles are loaded where the float64 kernel compares against memory. There is no NaN: the test of the needles for NaN and the pass that fixes their answers are gone. One to three needles are searched one at a time in general registers with `cmovg`.
- **check:** pass
- **compare:** not run (first version). `run`: 4.69x over naive C overall (64 needles 2.46x, 1,000 needles 8.96x), 3.11x over NumPy (4.37x and 2.21x).
- **Kept:** yes
- **Notes:** about 15% slower than the float64 kernel at 64 to 16,384 elements (132.6 ns for 64 needles at 1,024 elements, 113.4 there). `vpcmpgtq` runs on one port of this processor with a latency of three cycles, and the float compare on two. Against NumPy with 1,000 needles the kernel is 3.0x to 3.5x up to 4,096 elements and 1.2x from 1,048,576.
