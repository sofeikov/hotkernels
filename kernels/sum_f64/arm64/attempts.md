# sum_f64 attempts (arm64)

All runs on an Apple M1 Pro (performance core), macOS, native. **Every judge command in this run takes `--unlocked`**: the kernel's `spec.py` and `naive.c` are new on this branch and not yet on `main` (docs/coverage-gaps-design.md, "Order of work"). `spec.py`, `naive.c` and `judge/` were not edited after the reference was committed. Other agents timed kernels on the same machine throughout, at load averages of 3 to 6, so every figure here is indicative.

The float32 kernel, `kernels/sum_f32/arm64/`, came first; its log says what was found about NumPy's order, the freedom the sign of a zero gives, and this core's first-level cache. This kernel is the same idea at another width: a vector of eight elements is four NEON registers, so a group of eight leaves would need 32 registers of running sums and the largest group here is four.

**Correctness beyond the judge.** Before every commit of `kernel.s` the same differential test as for float32 compared the kernel with `np.sum` on every n from 0 to 5,000 at four element offsets, in three distributions, every n to 1,100 in five more (denormals, mixed zeros, negative zeros, values near the largest double, infinities and NaNs), and about 4,500 random n up to 6,000,000 at random offsets: bits equal whenever NumPy's sum is finite, a result that is not finite whenever NumPy's is not.

## 1. Baseline: the tree walked one leaf at a time

- **Idea:** float32's first version at this width: an explicit stack of nodes below `sp`, split until a leaf; each leaf summed into four registers of two running sums and combined with four `faddp`; a final `fadd` of `+0.0`.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** 1.10x the naive C overall, 2.10x at 16, 1.36x at 64, then level with it (0.97x to 1.01x) from 1,024 up: 352 ns at 4,096. Load average 4.7.
