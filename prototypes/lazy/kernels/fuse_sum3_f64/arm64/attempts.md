# fuse_sum3_f64 attempts (arm64)

All figures are from an Apple M1 Pro, with the kernel's spec not yet on `main` (`--unlocked`).

## 1. Two additions, eight elements per iteration, overlapping tail

- **Idea:** `fadd` of the inner pair, then `fadd` of the third array, on four vectors of two per iteration. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time.
- **check:** pass, 60,304 cases
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** the loop of `fuse_add_scaled_f64`, whose log has the reasons for its shape.

## 2. Non-temporal loads (`ldnp`) in the loop

- **Idea:** as `fuse_add_scaled_f64` attempt 2: `ldnp` reads the same bytes as `ldp`, and the hint lets a loop that streams through its inputs keep up with the memory once the arrays no longer fit the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.019x [1.012 – 1.027]
- **Kept:** yes
- **Notes:** 65,536: 1.00x, 262,144: 1.08x, 1,048,576: 1.08x, 4,194,304: 1.05x, 16,777,216: 1.01x. Nothing changes below 65,536 elements.

## As it stands

`uv run judge run fuse_sum3_f64 --unlocked`: 1.02x over the compiled `naive.c` across the judge's sizes [1.02 – 1.03]. Against NumPy's passes for the same expression, called from Python: 4,096: 3.22x, 65,536: 1.53x, 1,048,576: 1.56x, 16,777,216: 1.36x.
