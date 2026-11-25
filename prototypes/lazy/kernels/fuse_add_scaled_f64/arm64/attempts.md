# fuse_add_scaled_f64 attempts (arm64)

The first fused kernel written, and the one the others were measured from. All figures are from an Apple M1 Pro, with the kernel's spec not yet on `main` (`--unlocked`).

## 1. Multiply, then add, eight elements per iteration, overlapping tail

- **Idea:** `fmul` then `fadd` on four vectors of two per iteration, the multiplier in a register, with a remainder of 1 to 7 elements done as one more block that ends exactly at the end of the arrays, and fewer than 8 elements one at a time. No `fmla`: a fused multiply-add rounds once and gives other bits than NumPy's two passes.
- **check:** pass, 98,555 cases
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** against the compiled `naive.c` it is 1.00x overall [1.00 – 1.01]: clang vectorises the loop the same way, and from 65,536 elements up both run at the speed memory is read and written. Against NumPy's two passes (`a + s * b` from Python): 4.4x at 4,096 elements, 1.98x at 16,384, 1.65x at 65,536 and 262,144, 1.4x to 1.5x from a million up.

## 2. Non-temporal loads (`ldnp`) in the loop

- **Idea:** as `clip_f64` attempt 3 in `kernels/`. `ldnp` reads the same bytes as `ldp`; the hint is what lets a loop that streams through its inputs run at the speed of the system `memmove` once the arrays no longer fit the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.041x [1.039 – 1.043]
- **Kept:** yes
- **Notes:** 262,144: 1.23x, 1,048,576: 1.20x, 4,194,304: 1.06x, 16,777,216: 1.02x. Nothing changes to 4,096 elements; 16,384 measured 0.97x [0.97 – 0.97]. This is the one place the assembly is ahead of the compiled C, which does not emit the instruction.
