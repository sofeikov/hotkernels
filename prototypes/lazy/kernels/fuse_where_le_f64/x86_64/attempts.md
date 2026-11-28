# fuse_where_le_f64 attempts (x86_64)

The kernel's spec is not on `main` yet, so the judge is run with `--unlocked`.

## 1. The loop `make_kernel.py` writes for the shape

- **Idea:** `where(le(V0,S0),V1,V2)` as 2 operations on two ymm registers of four per iteration, each rounding once, in the order of NumPy's passes; every load of an iteration before its store, every address the array's base plus one index. A remainder of 1 to 7 elements is one more block that ends exactly at the end of the arrays, and fewer than 8 elements are done one at a time. Written by `prototypes/lazy/make_kernel.py`, not by hand.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.16x [1.16 – 1.16] overall against `naive.c`; 753.7 ns against 873.7 ns at 4,096 elements, the length of a block of a program, 855.0 µs against 878.5 µs at 1,048,576, and level from 65,536 up. Taken at a load average of 3.2, with other sessions running.
- **Kept:** yes
- **Notes:** kept for two reasons. It is ahead of the compiled C at the lengths a program calls it with, 1.16x to 1.53x up to 16,384 elements. And its comparison is `vcmppd` with `_CMP_LE_OQ`, which raises no flag for a NaN: clang compiles the reference's `islessequal` to `vcmpnlepd`, the signalling comparison, since it takes the floating-point flags for unread, and the layer drops a result whose kernel raised the invalid-operation flag ([the design](../../../../../docs/lazy-fusion-design.md#floating-point-exceptions)). With the output 16 bytes behind its input the loop is level with itself clear of it or ahead, 0.95x to 1.41x. No `pext`, `pdep`, gather or masked store.
