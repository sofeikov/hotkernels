# sum_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. **Every judge command in this run takes `--unlocked`**: the kernel's `spec.py` and `naive.c` are new on this branch and not yet on `main`, taken verbatim from the ARM64 session's branch `claude/gaps160-03-sum-f32`. `spec.py`, `naive.c` and `judge/` were not edited after the reference was committed. Two other agents timed kernels on the same machine at the same time, so figures here are indicative; the load average is given with each.

What the compiler does with `naive.c`: NumPy's tree, one leaf at a time, each leaf's additions waiting on the one before. 0.11 ns per element from 1,024 to 262,144.

**Correctness beyond the judge.** The judge sweeps every size to 65 and then powers of two and their neighbours. Before every commit of `kernel.s` a differential test outside the repository (`difftest.py` in the run's scratch folder) assembled the file, called it through `ctypes` and compared its bits with `np.sum`: every n from 0 to 5,000, twice each, once ending exactly at a `PROT_NONE` page and once starting 0 to 7 floats after a 32-byte boundary, plus 300 random n up to 8,388,608 at random offsets, in seven distributions (values over 120 binary orders of magnitude of both signs, normal values, negative zeros, mixed signed zeros, denormals, values near the largest float halved until the sum is finite, pairs that nearly cancel). A copy with the split at the ceiling of m/2 instead of the floor failed 9,411 of 20,024 inputs, and a copy with the group's leaf sums combined in another order failed 10,945.

## 1. Baseline: the tree walked with an explicit stack, groups of eight leaves side by side

- **Idea:** 0 to 7 elements one after another from +0.0; 8 to 128 one leaf in one `ymm`, combined with two `vhaddps`, then the tail. Beyond 128 the tree is walked with a stack of 16-byte frames below `rsp` (the right half and its length, then the left half's sum). A node of 68 to 128 vectors not on the right edge is a group: its eight leaves, three levels down, are summed in one loop, one `ymm` each, so eight independent chains of `vaddps` with a memory operand; leaves of q + 1 vectors add their last one after the loop, the others a vector of +0.0 chosen with `cmova`. The leaf sums and then the node's sum are combined with `vhaddps` and one `vperm2f128` pair in NumPy's order. The last leaf's t elements are added one by one.
- **check:** pass
- **compare:** n/a (first version). `run`: 2.78x [2.77 – 2.79] over the naive C overall, 14.4x over NumPy.
- **Kept:** yes
- **Notes:** 6.3x at 1,024 (17.7 ns) and 6.9x at 4,096 (65.3 ns, 0.016 ns an element); 3.4x in the second-level cache, 2.1x from 1,048,576 to 4,194,304, 1.26x at 16,777,216 (2.16 ms, memory). 1.6x at 256, which is two single leaves taken one after another by the walk; 1.7x at 16 and 2.2x at 64. Load average 1.6.

## 2. Combine a leaf's eight sums with shuffles instead of two `vhaddps`

- **Idea:** `vmovshdup` and an add give r0 + r1, r2 + r3, r4 + r5 and r6 + r7 in the even lanes, `vpermilps` and a second add the two halves' sums in lane 0 of each 128-bit half: six instructions and a shorter chain than two `vhaddps` (three micro-operations each), the extract and the add.
- **check:** pass
- **compare:** significantly faster, 1.027x [1.022 – 1.032]
- **Kept:** yes
- **Notes:** 1.32x at 256 (12.7 ns), whose two leaves are each combined on the walk's critical path; 1.01x at 1,024 and 4,096; 0.99x in the second-level cache, inside the noise of the other agents' runs. Load average 0.8.

## 3. Groups of two and four leaves

- **Idea:** a node of 17 to 32 vectors not on the right edge is two leaves, of 34 to 64 four (33 and 65 to 67 have leaves at two depths and are split again, as before). Each is summed in one loop with a `ymm` per leaf, as the group of eight is, instead of one leaf at a time with a trip up the walk between them.
- **check:** pass
- **compare:** significantly faster, 1.038x [1.029 – 1.047]
- **Kept:** yes
- **Notes:** 1.50x at 256 (8.3 ns), a single group of two; nothing elsewhere among the timed sizes, which from 1,024 up are all groups of eight. Sizes whose walk ends in nodes of 17 to 64 vectors, such as 272 to 512, gain the same way but are not timed. Load average 1.0.
