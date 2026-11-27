# fuse_sum_squares_f64 attempts (x86_64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. None: the C reference

- **Idea:** keep `naive.c`, NumPy's pairwise order with the squares added in, before writing any assembly.
- **check:** pass
- **compare:** n/a
- **Kept:** yes (the C)
- **Notes:** `run`, naive C against NumPy: 14.9x [14.89 – 14.97] overall; 419.2 ns against 2.8 us at 4,096 elements, 116.0 us against 472.6 us at 1,048,576, 4.0x to 4.2x from 65,536 up. Clang vectorises the eight running sums of a block. An AVX2 loop would have to keep NumPy's tree as `kernels/sum_f64/x86_64/kernel.s` does, with a multiplication before each addition (no multiply-add), and could gain at most what that kernel gains over its own C; not tried.
