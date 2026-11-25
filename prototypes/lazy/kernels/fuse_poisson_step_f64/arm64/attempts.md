# fuse_poisson_step_f64 attempts (arm64)

All figures are from an Apple M1 Pro, with the kernel's spec not yet on `main` (`--unlocked`).

## 1. Eight operations in the expression's order, eight elements per iteration, overlapping tail

- **Idea:** two `fadd`, three `fmul` by numbers in registers, `fadd`, `fdiv` by a number in a register and `fsub`, on four vectors of two per iteration: five loads and a store per vector. No `fmla`, no `fmls`, and a real division: a product with the reciprocal rounds differently. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time.
- **check:** pass, 280,900 cases
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** the loop of `fuse_add_scaled_f64`, whose log has the reasons for its shape.

## 2. Non-temporal loads (`ldnp`) in the loop

- **Idea:** as `fuse_add_scaled_f64` attempt 2: `ldnp` reads the same bytes as `ldp`, and the hint lets a loop that streams through its inputs keep up with the memory once the arrays no longer fit the first-level cache.
- **check:** pass
- **compare:** significantly faster, 1.008x [1.006 – 1.010]
- **Kept:** yes
- **Notes:** 65,536: 1.00x, 262,144: 1.07x, 1,048,576: 1.01x, 4,194,304: 1.02x, 16,777,216: 1.00x. Nothing changes below 65,536 elements. The gain is smaller the more arrays the loop reads: 1.2x with two, 1.1x with three.

## As it stands

`uv run judge run fuse_poisson_step_f64 --unlocked`: 1.00x over the compiled `naive.c` across the judge's sizes [1.00 – 1.00]. Against NumPy's passes for the same expression, called from Python: 4,096: 5.05x, 65,536: 3.13x, 1,048,576: 2.72x, 16,777,216: 2.72x.
