# sum_f32 attempts (arm64)

All runs on an Apple M1 Pro (performance core), macOS, native. **Every judge command in this run takes `--unlocked`**: the kernel's `spec.py` and `naive.c` are new on this branch and not yet on `main` (docs/coverage-gaps-design.md, "Order of work"). `spec.py`, `naive.c` and `judge/` were not edited after the reference was committed. Three other agents timed kernels on the same machine throughout, at load averages of 6 to 16, so every figure here is indicative.

**Correctness beyond the judge.** The judge sweeps every size to 65 and then powers of two and their neighbours; NumPy's tree has an irregular shape between those. Before every commit of `kernel.s` a differential test outside the repository compared the kernel with `np.sum` on every n from 0 to 5,000 at four element offsets, in three distributions (values of very different magnitudes, normal values, pairs that nearly cancel), every n to 1,100 in five more (denormals, mixed zeros, negative zeros, values near the largest float, infinities and NaNs scattered), and about 4,500 random n up to 6,000,000 at random offsets: bits equal whenever NumPy's sum is finite, a result that is not finite whenever NumPy's is not.

**What is free and what is not.** Every addition must be NumPy's: the same two operands, rounded once. The operands of an addition may be swapped. The sign of a zero is the one freedom beyond that: a zero operand changes nothing in a sum that is not zero, so the sign of an intermediate zero can only change the sign of a result that is zero, and NumPy's result is `0 + pairwise(x)`, never `-0.0`. A kernel may therefore drop NumPy's leading `0 +` and its `0 +` of the shorter paths, as long as the result it returns is not `-0.0`: one operand of the last addition that cannot be `-0.0` is enough.

## 1. Baseline: the tree walked one leaf at a time

- **Idea:** NumPy's recursion as an explicit stack below `sp`: a node is split at half its vectors (8 elements each) until it is a leaf of at most 16 vectors, or 15 and the tail on the right edge; each leaf is summed into two NEON registers of four running sums and combined with three `faddp`; a marker on the node stack says when the sums of two children are to be added. A final `fadd` of `+0.0` gives NumPy's sign of zero.
- **check:** pass
- **compare:** n/a (first version)
- **Kept:** yes
- **Notes:** 1.31x the naive C overall, 297 ns at 4,096 (load average 9.9). The leaves are independent, and the processor's out-of-order window already overlaps some of them.

## 2. Leaves side by side: groups of two, four and eight

- **Idea:** a node of 17 to 32 vectors is two leaves, of 34 to 64 four, of 68 to 128 eight (33 and 65 to 67 have leaves at two depths and are split further). Such a node's leaves are summed in one loop, each in its own pair of registers, so eight leaves give sixteen independent chains of additions; the leaves have q or q + 1 vectors, and the last vector of a longer leaf is added after the loop, a shorter leaf adding a vector of `+0.0` from the stack instead. The leaf sums are combined in the node's order by `faddp` across leaves.
- **check:** pass
- **compare:** significantly faster, 1.264x [1.255 – 1.272]
- **Kept:** yes
- **Notes:** 1.84x at 4,096 (159 ns, 0.039 ns an element), 1.81x at 16,384; 1.13x to 1.15x in the second-level cache; nothing at 16 and 64, which are one leaf. Load average 6.6.
