# fuse_scaled_sum_f64 attempts (arm64)

All figures are from an Apple M1 Pro, with the kernel's spec not yet on `main` (`--unlocked`).

## 1. Add, then multiply, eight elements per iteration, overlapping tail

- **Idea:** `fadd` then `fmul` by the number in a register, on four vectors of two per iteration. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time.
- **check:** pass, 98,555 cases
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** the loop of `fuse_add_scaled_f64`, whose log has the reasons for its shape.

## 2. Non-temporal loads (`ldnp`) in the loop

- **Idea:** as `fuse_add_scaled_f64` attempt 2: `ldnp` reads the same bytes as `ldp`, and the hint lets a loop that streams through its inputs keep up with the memory once the arrays no longer fit the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.044x [1.040 – 1.048]
- **Kept:** yes
- **Notes:** 65,536: 1.01x, 262,144: 1.25x, 1,048,576: 1.21x, 4,194,304: 1.06x, 16,777,216: 1.02x. Nothing changes below 65,536 elements.

## As it stands

`uv run judge run fuse_scaled_sum_f64 --unlocked`: 1.04x over the compiled `naive.c` across the judge's sizes [1.04 – 1.04]. Against NumPy's passes for the same expression, called from Python: 4,096: 4.04x, 65,536: 1.89x, 1,048,576: 1.56x, 16,777,216: 1.56x.
