# fuse_flux_update_f64 attempts (arm64)

All figures are from an Apple M1 Pro, with the kernel's spec not yet on `main` (`--unlocked`).

## 1. Five operations in the expression's order, eight elements per iteration, overlapping tail

- **Idea:** `fsub`, `fadd`, `fsub`, `fmul`, `fsub` on four vectors of two per iteration: six loads and a store per vector, the registers of the first two arrays reused for the last two. No `fmls`. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time.
- **check:** pass, 120,208 cases
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** the loop of `fuse_add_scaled_f64`, whose log has the reasons for its shape.

## 2. Non-temporal loads (`ldnp`) in the loop

- **Idea:** as `fuse_add_scaled_f64` attempt 2: `ldnp` reads the same bytes as `ldp`, and the hint lets a loop that streams through its inputs keep up with the memory once the arrays no longer fit the first-level cache.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.997 – 1.000]
- **Kept:** no
- **Notes:** 65,536: 1.00x, 262,144: 1.05x, 1,048,576: 0.94x, 4,194,304: 1.00x, 16,777,216: 0.99x. Nothing changes below 65,536 elements. Reverted: not significantly faster.

## As it stands

`uv run judge run fuse_flux_update_f64 --unlocked`: 0.99x over the compiled `naive.c` across the judge's sizes [0.99 – 1.00]. Against NumPy's passes for the same expression, called from Python: 4,096: 2.70x, 65,536: 1.81x, 1,048,576: 1.71x, 16,777,216: 1.74x.
