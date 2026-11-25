# fuse_sum_squares_f64 attempts (arm64)

All figures are from an Apple M1 Pro, with the kernel's spec not yet on `main` (`--unlocked`).

## 1. Squares added in NumPy's pairwise order, eight elements per iteration

- **Idea:** a leaf of 8 to 128 elements keeps NumPy's eight running sums in four registers, squares each group of eight (`fmul`) and adds it (`fadd`), combines the sums with `faddp` in NumPy's order and adds the last n mod 8 squares one by one. Longer arrays are NumPy's recursive split, walked with a stack of ranges and a stack of sums in a frame below `sp`, since a kernel calls nothing.
- **check:** pass, 20,368 cases
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** `kernels/sum_f64` is the same walk over the elements themselves, with groups of leaves summed in one loop; this is the plain form of it.

## 2. Non-temporal loads (`ldnp`) in the loop

- **Idea:** as `fuse_add_scaled_f64` attempt 2: `ldnp` reads the same bytes as `ldp`, and the hint lets a loop that streams through its inputs keep up with the memory once the arrays no longer fit the first-level cache.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.998 – 1.004]
- **Kept:** no
- **Notes:** 65,536: 1.00x, 262,144: 1.00x, 1,048,576: 1.02x, 4,194,304: 1.00x, 16,777,216: 1.00x. Nothing changes below 65,536 elements. Reverted: not significantly faster.

## As it stands

`uv run judge run fuse_sum_squares_f64 --unlocked`: 1.49x over the compiled `naive.c` across the judge's sizes [1.49 – 1.49]. Against NumPy's passes for the same expression, called from Python: 4,096: 7.71x, 65,536: 3.16x, 1,048,576: 2.86x, 16,777,216: 2.76x.
