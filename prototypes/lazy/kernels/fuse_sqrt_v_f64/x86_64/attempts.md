# fuse_sqrt_v_f64 attempts (x86_64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. The loop `make_kernel.py` writes for the shape

- **Idea:** `sqrt(V0)` as 1 operation on two ymm registers of four per iteration, each rounding once, in the order of NumPy's passes; every load of an iteration before its store, every address the array's base plus one index. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time. Written by `prototypes/lazy/make_kernel.py`, not by hand.
- **check:** pass
- **compare:** n/a (first version). `run`: 5.46x [5.46 – 5.47] overall against `naive.c`; 2.3 us against 8.5 us at 4,096 elements, 617.0 us against 4.84 ms at 1,048,576.
- **Kept:** yes
- **Notes:** `naive.c` calls `__builtin_sqrt`, which keeps C's rule that a square root of a negative number sets `errno`; at the judge's flags clang then calls the library for every element and does not vectorise. `vsqrtpd` is the instruction NumPy's loop executes, and gives its bits. The gain is that and nothing else: in-cache it is the square root's throughput, beyond it memory.
