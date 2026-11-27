# fuse_div_neg_add_v_mul_v_v_sub_min_s_v_s_f64 attempts (arm64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. The loop `make_kernel.py` writes for the shape

- **Idea:** `div(neg(add(V0,mul(V1,V2))),sub(min(S0,V3),S1))` as 6 instructions on four vectors of two per iteration, each rounding once, in the order of NumPy's passes; every load of an iteration before its store. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time. Written by `prototypes/lazy/make_kernel.py`, not by hand.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** not tuned and not timed by itself. The loads are plain `ldp`: the kernels written by hand read with `ldnp`, which their logs found quicker from 262,144 elements up, but this one is called for the most part on blocks of an expression that stay in the cache.
