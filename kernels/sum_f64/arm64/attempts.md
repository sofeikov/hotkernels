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

## 2. Leaves side by side: groups of two and four

- **Idea:** float32's attempts 2 and 4 together: a node of 17 to 32 vectors is two leaves, of 34 to 64 four (33 is split further), summed in one loop with each leaf in its own four registers, so four leaves are sixteen chains. The longer leaves' last vector is added after the loop and the shorter ones add 64 bytes of `+0.0`, skipped when every leaf has q vectors. A group of eight would need 32 registers of running sums.
- **check:** pass
- **compare:** significantly faster, 1.073x [1.072 – 1.074]
- **Kept:** yes
- **Notes:** 1.23x at 256, 1.33x at 1,024, 1.29x at 4,096 (272 ns), 1.23x at 16,384; but 0.96x to 0.97x from 65,536 up, beyond the first-level cache, where the baseline's one leaf at a time ran at the speed of the naive C. Load average 4.2.

## 3. Four leaves in a diagonal order when q is even

- **Idea:** float32's attempt 3: when q is even, eight steps at a time in a diagonal order, leaf j's step k beside leaf 0's step k + j, so that loads issued together sit at different offsets of their 256-byte blocks.
- **check:** pass
- **compare:** significantly faster, 1.028x [1.019 – 1.036]
- **Kept:** yes
- **Notes:** 1.08x at 1,024 and 16,384, 1.09x at 4,096 (251 ns); 1.01x to 1.02x from 65,536 to 1,048,576. Load average 4.0.

## 4. Leaves one at a time beyond the first-level cache

- **Idea:** attempt 2 lost 3% to 4% beyond the first-level cache, where the baseline, one leaf at a time and so one stream through memory, had run at the speed of the naive C. So an array of more than 16,384 elements (128 KB) is walked down to its leaves and they are summed one after another; the out-of-order core still overlaps neighbouring leaves.
- **check:** pass
- **compare:** significantly faster, 1.011x [1.007 – 1.016]
- **Kept:** yes
- **Notes:** 1.03x from 262,144 to 4,194,304, 1.04x at 16,777,216, nothing in the first-level cache. A first try with 16,384 itself on the long path was 0.74x there: an array of exactly 128 KB still sits in the first-level cache. Load average 4.8.

## 5. One leaf as a straight run entered at its length

- **Idea:** float32's attempts 5 and 20: 8 to 128 elements are tested with one unsigned compare and summed by 15 steps written out, entered by a computed branch at the step that leaves exactly the leaf's vectors after its first.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.004 – 1.014]
- **Kept:** yes
- **Notes:** 1.08x at 16 (1.6 ns), 1.02x at 64, nothing elsewhere. A leaf of eight float64 vectors is four chains of seven additions, bound by their latency more than float32's two chains were by the loop. The indirect branch is predicted in the judge, which calls one size at a time. Load average 4.5.
