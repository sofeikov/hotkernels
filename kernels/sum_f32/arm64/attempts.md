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

## 3. Eight leaves in a diagonal order when q is even

- **Idea:** a microbenchmark of the eight-leaf loop alone ran at the load limit (about 28 ns for 1,024 elements) when the leaves started 544 bytes apart and 20% slower (34 ns) when they started 256, 512 or 1,024 bytes apart, as they do for every power-of-two size. Loads of the same 32 bytes of different 256-byte blocks seem to contend in the first-level cache on this core. So when q is even, eight steps are taken at once in a diagonal order, leaf j's step k beside leaf 0's step k + j, which puts the loads issued together in different 32-byte positions; each leaf still adds its vectors in order. When q is odd the plain order already does that. The running sums start at `+0.0` so that all steps are alike.
- **check:** pass
- **compare:** significantly faster, 1.061x [1.058 – 1.065]
- **Kept:** yes
- **Notes:** 1.14x at 1,024 and 4,096 (137 ns), 1.15x at 16,384, 1.06x in the second-level cache. Load average 4.4.

## 4. No extra step when every leaf of a group has q vectors

- **Idea:** a group of 8 leaves whose vectors are a multiple of 8 (of 4 for four leaves, of 2 for two) has leaves of q vectors each, so the step that adds the longer leaves' last vector, and the work of finding those vectors, is skipped. A group of two needs no search at all: its right leaf is the longer one exactly when m is odd.
- **check:** pass
- **compare:** significantly faster, 1.045x [1.044 – 1.046]
- **Kept:** yes
- **Notes:** 1.06x at 256, 1.11x at 1,024, 1.09x at 4,096 (125 ns), 1.05x in the second-level cache. Every timed size from 256 up is a power of two, whose groups all have equal leaves; other sizes keep the cost. Load average 4.0.

## 5. A straight path for one leaf, two steps per trip

- **Idea:** 8 to 128 elements, a single leaf, are tested with one unsigned compare and summed in code of their own, two vectors per trip of the loop, instead of the leaf macro the walk uses.
- **check:** pass
- **compare:** significantly faster, 1.047x [1.046 – 1.047]
- **Kept:** yes
- **Notes:** 1.25x at 16 (1.3 ns), 1.33x at 64 (2.9 ns), nothing elsewhere. Load average 6.2.

## 6. The diagonal in steps of 64 bytes

- **Idea:** a microbenchmark of the group loop over blocks in the second-level cache ran 5% faster when each visit to a leaf loaded 64 bytes, two vectors, instead of 32: 3.37 us against 3.55 us for 65,536 elements, where one stream at the load limit takes 3.15 us. So the diagonal block of 8 vectors became 4 visits of two vectors per leaf.
- **check:** pass
- **compare:** significantly slower, 0.990x [0.989 – 0.992]
- **Kept:** no
- **Notes:** 0.94x at 1,024, 0.98x at 4,096 and 16,384, nothing in the second-level cache. A block of four visits has only 11 diagonals, with few leaves active at its two ends, and each visit's two additions to a running sum wait for each other; the microbenchmark's block was 8 visits long.

## 7. A diagonal block of 16 vectors in steps of 64 bytes

- **Idea:** attempt 6 with the microbenchmark's block length: when at least 16 steps remain and q is even, a block of 8 visits of two vectors per leaf, 15 diagonals, as measured there.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.001 – 1.011]
- **Kept:** yes
- **Notes:** 1.04x from 65,536 to 1,048,576 (57.2 us), as the microbenchmark said; but 0.95x at 1,024 and 0.97x to 0.98x at 4,096 and 16,384, in the first-level cache. Load average 3.5.

## 8. The 64-byte diagonal only beyond the first-level cache

- **Idea:** keep attempt 7's gain and drop its loss: the walk notes on entry whether the array is at least 32,768 elements (128 KB, the first-level cache of a performance core), and only then do groups of eight take blocks of 16 vectors in 64-byte visits.
- **check:** pass
- **compare:** significantly faster, 1.007x [1.005 – 1.010]
- **Kept:** yes
- **Notes:** 1.05x at 1,024, 1.03x at 4,096 and 16,384 (back to attempt 5's figures), unchanged from 65,536 up. Load average 3.3.

## 9. Prefetch the next group's leaf starts beyond the first-level cache

- **Idea:** in an array of at least 32,768 elements, each group of eight asks with `prfm pldl1keep` for the first line of each leaf of the next group, taken to be as long as this one and to start where it ends, so that the eight new streams do not all start on a miss. `prfm` never faults; past the end of the array it does nothing.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.003 – 1.009]
- **Kept:** yes
- **Notes:** 1.02x at 262,144 and 1,048,576 (55.6 us), 1.01x at 65,536, nothing in the first-level cache or beyond the second. Load average 3.0.

## 10. Test for a group of eight first

- **Idea:** a node is tested for a group of eight (68 to 128 vectors) with one unsigned compare before anything else, since nearly all of a long array is such groups; the other classes follow.
- **check:** pass
- **compare:** significantly faster, 1.004x [1.003 – 1.006]
- **Kept:** yes
- **Notes:** 1.01x at 1,024 and 16,384, nothing elsewhere: five instructions per group of 1,024 elements. Load average 4.0.

## 11. No loop for 16 to 23 elements

- **Idea:** two vectors and straight to the combining, without the tests that lead into the loop.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.999 – 1.003]
- **Kept:** no
- **Notes:** 1.3 ns at 16 either way. Load average 4.5.

## 12. Two steps per trip in a group of two

- **Idea:** the loop of a group of two leaves (256 elements is one such group) takes two steps per trip, as the single leaf does since attempt 5.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.995 – 1.005]
- **Kept:** no
- **Notes:** 9.8 ns at 256 either way. A group of two is four chains of additions, bound by their latency, not by the loop's instructions. Load average 6.3.

## 13. One group of two without the walk

- **Idea:** 136 to 256 elements that are a multiple of 8 are a single group of two leaves; they are summed straight from the entry, without the frame, the node stack and the stack of sums. The group's code became a macro used in both places.
- **check:** pass
- **compare:** significantly faster, 1.008x [1.007 – 1.010]
- **Kept:** yes
- **Notes:** 1.11x at 256 (8.8 ns), nothing elsewhere. Load average 5.6.

## 14. One group of eight without the walk

- **Idea:** as attempt 13 for groups of eight: 544 to 1,024 elements that are a multiple of 64 are one group of eight equal leaves, which needs nothing of the frame but the flag of attempt 8; the entry sets that and an empty node stack, and the push after the group returns.
- **check:** pass
- **compare:** significantly faster, 1.002x [1.001 – 1.003]
- **Kept:** yes
- **Notes:** 1.02x at 1,024 (29.0 ns), nothing elsewhere. Load average 5.0.

## 15. `ldnp` in the 64-byte diagonal

- **Idea:** the non-temporal load pair in the blocks that only arrays of 32,768 elements and more take (attempt 8).
- **check:** pass
- **compare:** significantly faster, 1.009x [1.003 – 1.015]
- **Kept:** yes
- **Notes:** 1.10x at 4,194,304 (265 us) and 1.05x at 16,777,216, beyond the second-level cache; 0.98x [0.96 – 1.00] at 262,144, nothing elsewhere. Load average 4.1.

## 16. A diagonal of 16 single steps in the first-level cache

- **Idea:** in the first-level cache, a diagonal block of 16 steps of one vector (23 diagonals) instead of two blocks of 8 (15 each), so that fewer diagonals are short ones at a block's ends.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.999 – 1.002]
- **Kept:** no
- **Notes:** 124.9 ns at 4,096 against 124.4. The out-of-order core already fills the short diagonals from the next block. Load average 4.6.

## 17. `ldnp` only beyond the second-level cache

- **Idea:** attempt 15 cost 0.98x at 262,144 within its interval; so the 64-byte blocks use `ldp` from 32,768 elements and `ldnp` only from 2,097,152 (8 MB).
- **check:** pass
- **compare:** no significant difference, 1.002x [0.998 – 1.005]
- **Kept:** no
- **Notes:** 1.02x [1.00 – 1.04] at 262,144, 1.01x at 1,048,576: the loss attempt 15 showed there was mostly noise. Load average 5.0.

## 18. Prefetch two groups ahead

- **Idea:** attempt 9's prefetch at twice the distance, the leaf starts of the group after next.
- **check:** pass
- **compare:** no significant difference, 0.999x [0.998 – 1.001]
- **Kept:** no
- **Notes:** nothing at any size. Load average 4.2.

## 19. Loops aligned to 16 bytes

- **Idea:** `.p2align 4` before the loop of the one-leaf path, the diagonal blocks, the plain step of a group of eight and the walk's node label.
- **check:** pass
- **compare:** significantly slower, 0.991x [0.990 – 0.992]
- **Kept:** no
- **Notes:** 0.90x at 64 (3.3 ns): the padding put `nop`s in the one-leaf path ahead of its loop. Nothing elsewhere. Load average 4.3.

## 20. One leaf as a straight run entered at its length

- **Idea:** the one-leaf path's loop replaced by 15 steps written out, entered by a computed branch at the step that leaves exactly as many as the leaf has vectors after its first.
- **check:** pass
- **compare:** significantly faster, 1.017x [1.014 – 1.019]
- **Kept:** yes
- **Notes:** 1.21x at 64 (2.4 ns), nothing elsewhere. The judge calls with one size at a time, so the indirect branch is always predicted; calls of changing sizes would mispredict it, as they mispredicted the loop's exit before. Load average 4.5.

## 21. A group of two as a straight run entered at its length

- **Idea:** attempt 20 for the group of two: 15 steps written out, entered by a computed branch.
- **check:** pass
- **compare:** no significant difference, 1.001x [0.997 – 1.004]
- **Kept:** no
- **Notes:** 9.0 ns at 256 either way: the group of two is bound by the latency of its four chains (attempt 12), not by its loop. Load average 6.0.
