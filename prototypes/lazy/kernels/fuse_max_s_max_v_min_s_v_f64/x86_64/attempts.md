# fuse_max_s_max_v_min_s_v_f64 attempts (x86_64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. The loop `make_kernel.py` writes for the shape

- **Idea:** `max(S0,max(V0,min(S1,V1)))` as 3 operations on two ymm registers of four per iteration, each rounding once, in the order of NumPy's passes; every load of an iteration before its store, every address the array's base plus one index. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time. Written by `prototypes/lazy/make_kernel.py`, not by hand.
- **check:** pass
- **compare:** n/a (first version). `run`: 0.75x [0.75 – 0.76] overall against `naive.c`; 1.0 us against 604.8 ns at 4,096 elements, 781.6 us against 707.5 us at 1,048,576.
- **Kept:** no
- **Notes:** slower: the loop makes every maximum and minimum with three instructions (`vcmpunordpd`, `vblendvpd`, `vmaxpd`). Clang sees that whether `s0` or `s1` is a NaN does not change in the loop and tests it once, so two of the three operations are one instruction each in `naive.c`, and only `max(a, ...)` takes three. A loop that tested the numbers once would be the next attempt. On x86-64 the kernel is built from its C reference. `make_kernel.py --arch x86_64` passes over a kernel that has this log; `--force` writes the loop again.
