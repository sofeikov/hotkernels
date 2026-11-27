# fuse_sub_v_s_f64 attempts (x86_64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. The loop `make_kernel.py` writes for the shape

- **Idea:** `sub(V0,S0)` as 1 operation on two ymm registers of four per iteration, each rounding once, in the order of NumPy's passes; every load of an iteration before its store, every address the array's base plus one index. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time. Written by `prototypes/lazy/make_kernel.py`, not by hand.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.02x [1.02 – 1.03] overall against `naive.c`; 512.6 ns against 524.8 ns at 4,096 elements, 284.5 us against 278.2 us at 1,048,576.
- **Kept:** no
- **Notes:** level: clang vectorises `naive.c` at the judge's `-O3 -march=native`, and beyond the first-level cache both wait for memory. On x86-64 the kernel is built from its C reference. `make_kernel.py --arch x86_64` passes over a kernel that has this log; `--force` writes the loop again.
