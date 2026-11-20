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

## 6. One group of two without the walk

- **Idea:** float32's attempt 13: 136 to 256 elements that are a multiple of 8 are summed as one group of two straight from the entry, without the frame and the stacks.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.995 – 1.006]
- **Kept:** no
- **Notes:** 1.01x at 256 (16.2 ns): at this width a group of two is eight chains of 16 additions, and their latency, not the walk around them, is what 256 elements cost. Load average 8.2.

## 7. `ldnp` in the leaves of the walk

- **Idea:** the non-temporal load pair in the loop of the leaf the walk uses, which since attempt 4 sums every leaf of an array beyond the first-level cache (and the odd leaves of shorter ones).
- **check:** pass
- **compare:** significantly faster, 1.010x [1.005 – 1.015]; again 1.010x [1.002 – 1.019]
- **Kept:** yes
- **Notes:** beyond the second-level cache 1.09x to 1.12x at 4,194,304 and 1.06x to 1.07x at 16,777,216; but 0.95x at 262,144 and 0.96x to 0.98x at 65,536, in it. Load average 20 to 24 during both runs, from other work: rough.

## 8. `ldnp` only above 4 MB

- **Idea:** keep attempt 7's gain beyond the second-level cache and drop its loss in it: the walk's leaves load with `ldp` up to 524,288 elements (4 MB) and with `ldnp` above.
- **check:** pass
- **compare:** significantly faster, 1.013x [1.006 – 1.019]
- **Kept:** yes
- **Notes:** 1.10x at 262,144 and 1.02x at 65,536, recovering attempt 7's loss; unchanged at 1,048,576 and above. Load average 19, from other work: rough.

## 9. Test for a group of four first

- **Idea:** float32's attempt 10: in an array that fits the first-level cache a node is tested for a group of four with one unsigned compare first.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.999 – 1.003]
- **Kept:** no
- **Notes:** 1.01x at 4,096 within the noise. A group of four here is 512 elements, half of float32's group of eight, and the tests are a smaller share still of its loads than float32's were. Load average 17.

## 10. Groups of two beyond the first-level cache

- **Idea:** two neighbouring leaves are two streams 1 KB apart; perhaps few enough for the caches beyond the first, where attempt 4 sums leaves one at a time.
- **check:** pass
- **compare:** significantly slower, 0.989x [0.984 – 0.994]
- **Kept:** no
- **Notes:** 0.93x at 4,194,304 and 0.96x at 16,777,216 (where the pairs' loads are not the non-temporal ones of attempt 8), nothing elsewhere. Load average 12.

## Where this run stopped

Ten attempts, the last two rejected; the run was cut short at the coordinator's request (the binding and documents for `sum_f64` are left to the stacked x86-64 work). Recorded `uv run judge run sum_f64 --unlocked --record` at a load average of 7.8, flagged as noisy by the judge: 1.25x [1.24 – 1.25] over the naive C and 9.85x over NumPy overall; 1.40x to 1.45x from 1,024 to 16,384 elements; **slower than the naive C from 65,536 to 1,048,576 (0.97x, 0.97x, 0.91x [0.86 – 0.95])**, 1.07x and 1.03x beyond. Beyond the first-level cache the kernel sums one leaf at a time, as the C does, and is level with it at best: the gain there is open.

## 11. Beyond the first-level cache: groups summed leaf after leaf, without loops

- **Is the loss real?** An outside timer, the kernel and the naive C (built with the judge's flags) timed alternately, six rounds each, at a load average of 14: C/kernel 0.974 at 65,536, 0.973 at 262,144 and 0.867 at 1,048,576 (1.063 at 4,194,304). Real.
- **What the C does.** clang's code for the leaf is NumPy's loop, as ours is: no prefetch, five accumulators. It walks the tree by recursion, whose returns the processor predicts. Our walk pops a node or a combine marker from its stack after every leaf, and which one comes next follows the bits of a counter: a branch mispredicted about once a leaf, and the leaf's loop exit as well. Beyond the first-level cache each misprediction throws away the loads already issued for the next leaves. Outside timer, one change at a time: prefetch in the leaf 0.86x of the C (rejected); post-indexed or single-register loads, groups of two, plain-order groups of four, all 0.92x to 0.98x; the leaf as a straight run entered at its length (no loop exit), 0.99x; and both of the following together 1.06x to 1.07x.
- **Idea:** beyond 16,384 elements, a node of 17 to 32 or 34 to 64 vectors has its two or four leaves summed one after another, in the order of memory, each leaf a straight run entered at its length, and their sums combined in registers in the node's order: one pass of the walk per four leaves instead of one per leaf, and no loop exits. `ldnp` (attempt 8) starts at 2,097,152 elements (16 MB) instead of 524,288: at 1,048,576 (8 MB, inside the second-level cache) it cost 13%.
- **check:** pass
- **compare:** significantly faster, 1.030x [1.014 – 1.045]
- **Kept:** yes
- **Notes:** 1.11x at 65,536 and 262,144, 1.13x [1.00 – 1.28] at 1,048,576, nothing elsewhere. Outside timer against the naive C: 1.064x, 1.075x and 1.069x at the three sizes, 1.035x at 4,194,304. Load average 11.

## 12. Eight leaves in sequence beyond the first-level cache

- **Idea:** attempt 11 for a node of 68 to 128 vectors: its eight leaves one after another, their sums in eight registers, combined in the node's order. One pass of the walk per eight leaves.
- **check:** pass
- **compare:** significantly faster, 1.004x [1.001 – 1.007]
- **Kept:** yes
- **Notes:** 1.02x at 65,536 and 262,144, 1.01x at 1,048,576. Outside timer against the naive C: 1.097x, 1.107x, 1.071x. Load average 8.7.

## 13. The same sequence of leaves above 16 MB, with `ldnp`

- **Idea:** above 2,097,152 elements the walk still summed one leaf at a time with a loop (attempt 8's `ldnp` leaves). There too a group's leaves now run one after another as straight runs, with `ldnp`.
- **check:** pass
- **compare:** significantly faster, 1.003x [1.000 – 1.006]
- **Kept:** yes
- **Notes:** 1.02x at 4,194,304 and 16,777,216; 0.99x [0.99 – 1.00] at 65,536, a path this change does not touch (one more compare per leaf). Outside timer against the naive C: 1.067x at 1,048,576, 1.070x at 4,194,304, 1.030x at 16,777,216. Load average 7.5.
