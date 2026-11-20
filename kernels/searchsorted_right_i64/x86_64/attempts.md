# searchsorted_right_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, pinned to one performance core by the judge. Every run is `--unlocked`: the spec and the reference came in the same pull request as the kernel and were not on `main` to lock against. This kernel and `searchsorted_i64` are generated from one source and differ in the compare and the conditional move; each attempt was made on both and judged on each. The reasoning and the measurements behind an attempt are in [that kernel's log](../../searchsorted_i64/x86_64/attempts.md), under the same number.

## 1. Baseline: the left kernel with `<=` for `<`

- **Idea:** side='right' advances when `a[lo + half - 1] <= needle`, which is when the probe is not greater than the needle: `vpcmpgtq` with the needles as its second operand, straight from memory, and `vpandn` to take `half` where the compare is false. `cmovge` in the path for one to three needles.
- **check:** pass
- **compare:** not run (first version). `run`: 4.31x over naive C overall (64 needles 2.48x, 1,000 needles 7.47x), 3.22x over NumPy (4.55x and 2.27x).
- **Kept:** yes
- **Notes:** one instruction fewer per four searches than the left kernel, and the same time: 131.9 ns for 64 needles at 1,024 elements against 132.6.
