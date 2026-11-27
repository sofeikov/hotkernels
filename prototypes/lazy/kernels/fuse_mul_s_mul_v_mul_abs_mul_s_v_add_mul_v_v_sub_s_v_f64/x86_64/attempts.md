# fuse_mul_s_mul_v_mul_abs_mul_s_v_add_mul_v_v_sub_s_v_f64 attempts (x86_64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. The loop `make_kernel.py` writes for the shape

- **Idea:** `mul(S0,mul(V0,mul(abs(mul(S1,V1)),add(mul(V2,V3),sub(S2,V4)))))` as 8 operations on two ymm registers of four per iteration, each rounding once, in the order of NumPy's passes; every load of an iteration before its store, every address the array's base plus one index. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time. Written by `prototypes/lazy/make_kernel.py`, not by hand.
- **check:** fail: the reference, not the loop. `naive.c is wrong` on `n=1 scalars=nan_s1 a=nan_all`: `abs(s1 * b)` makes the NaN of `s1` a NaN with the sign bit clear, a second kind beside the inputs' NaN with it set, and clang orders the operands of the products otherwise than NumPy's passes, so the reference returns the other of the two.
- **compare:** n/a
- **Kept:** no
- **Notes:** a kernel whose reference the judge refuses is not kept, so on x86-64 this one is built from its C reference. The domain of the fused specs (one kind of NaN in the inputs, `one_nan`) does not cover a NaN that an absolute value makes inside the expression.
