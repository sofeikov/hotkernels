# sum_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. **Every judge command in this run takes `--unlocked`**: the kernel's `spec.py` and `naive.c` are new on this branch, taken verbatim from `claude/gaps160-04-sum-f64`, and not yet on `main`. They were not edited afterwards, nor was `judge/`.

What the compiler does with `naive.c` (`clang -O3 -march=native`): one leaf at a time, its eight running sums in two `ymm` registers, so two chains of dependent additions: about 0.1 ns per element (four cycles of add latency per four elements per chain) from 1,024 to 16,777,216 elements. NumPy is the same order with its per-call overhead.

**Correctness beyond the judge.** Before every commit of `kernel.s`, a differential test outside the repository (`difftest.py`, named in the report of this run) compared the kernel with `np.sum` bit for bit: every n from 0 to 5,000 at each element offset within 32 bytes and ending at an inaccessible page, in seven distributions (values over sixty orders of magnitude of both signs, normal values, mixed signed zeros, all -0.0, denormals, tiny values mixed with -0.0, values near the largest double scaled until the sum is finite), and 300 random n up to 8,388,608. A copy with two of the quad's combining additions swapped fails it.

## 1. Baseline: the tree walked with an explicit stack, leaves in pairs and quads

- **Idea:** the reference's tree walked without calls: a stack of (address, length) entries for pending right halves, and of finished left sums, in a 1 KB frame below `rsp`. A node whose two halves are leaves (a pair) is summed with both leaves side by side, four `ymm` chains; a node whose four grandchildren are leaves (a quad, 272 to 512 elements) with all four side by side, eight chains, enough for both vector adders. The leaves of a pair or quad are q or q + 1 blocks of eight long; all are summed for q blocks, then the longer ones add their last block and the shorter a block of +0.0 from `.rodata` (the header says why that changes no result). The n mod 8 leftovers go after the last leaf's eight sums are combined. Fewer than 8 elements: the chain from +0.0; up to 128: one leaf.
- **check:** pass
- **compare:** n/a (first version). `run`: 1.62x [1.62 – 1.62] over the naive C overall, 9.24x over NumPy.
- **Kept:** yes
- **Notes:** 1.97x at 16, 1.56x at 64, 2.70x at 256, 2.95x at 1,024 (34.0 ns), 2.72x at 4,096 (147 ns), 2.16x at 16,384, 1.42x at 65,536, 1.20x at 262,144; **slower than the naive C beyond the second-level cache**: 0.96x at 1,048,576, 0.94x at 4,194,304, 0.91x at 16,777,216. Load average 1.0.

## 2. A quad's leaves one after another from 262,144 elements

- **Idea:** beyond the second-level cache attempt 1 was slower than the naive C, which reads one stream; a quad reads four, a leaf (1 KB) apart. From `SEQ_MIN` = 262,144 elements a quad's four leaves are summed one after another in memory order, each into its own two registers; with no dependence between them the out-of-order core overlaps them. Same combination as before.
- **check:** pass
- **compare:** significantly faster, 1.013x [1.004 – 1.022]
- **Kept:** yes
- **Notes:** 1.02x at 1,048,576, 1.06x at 4,194,304, 1.08x at 16,777,216; no change elsewhere (262,144 itself 1.00x). Other judges were running at the same time.

## 3. Aligned loads in a quad, the running sums rotated

- **Idea:** the judge places its buffers 16 bytes apart, as NumPy's allocator does, and an `x` that is not a multiple of 32 bytes makes every other 32-byte load straddle two cache lines. A micro-benchmark (warm caches, one quad of 512 elements per call) gave 15.9 ns with `x` on a 64-byte boundary and 17.4 ns 16 bytes off it; at 16,384 elements, in the second-level cache, 760 ns against 1,100 to 1,180. Every leaf starts a multiple of 64 bytes from `x`, so all have the same offset t (elements, 0 to 3) from a 32-byte boundary. A quad now loads 32-byte-aligned chunks from t elements before each leaf: lane p of the even chunks' sum then holds r(p - t mod 8), of the odd chunks' r(4 + p - t mod 8), each still added in the reference order. A leaf's first chunk has its lanes before the leaf cleared to +0.0 and its last chunk, shared with the next leaf, its lanes after the leaf; the array's first and last chunk are read with `vmaskmovpd`, so nothing outside the buffer is loaded. For t = 2 `vhaddpd` still pairs r0 with r1 and so on and the combination only takes the pairs from other lanes (two `vpermilpd` per quad); for t = 1 and 3 each leaf is turned back with two `vpermpd` and two `vblendpd`. Only a quad below `SEQ_MIN` does this. **`vmaskmovpd`** (a masked load, two per quad) is used here.
- **check:** pass
- **compare:** significantly faster, 1.041x [1.022 – 1.061]
- **Kept:** yes
- **Notes:** 1.35x at 16,384 (804 ns), 1.05x at 4,096, 1.02x at 1,024, 0.99x at 256 (a pair, not touched: the dispatch costs a little), nothing elsewhere. In the micro-benchmark t = 1 and t = 3 (which the judge does not time) are 3% slower than before in the first-level cache (151.7 ns against 146.9 at 4,096) and 24% faster at 16,384 (822 to 848 ns against 1,115). A first try that turned the leaves back through memory (three stores and two overlapping loads per leaf) was 30% slower in the first-level cache: the loads wait for the stores to reach the cache.

## 4. Quads side by side up to 1,048,576 elements

- **Idea:** with attempt 3 the quads read the second-level cache without split loads, so the one-stream order of attempt 2 may no longer pay at 262,144 elements (2 MB, the size of that cache). `SEQ_MIN` from 262,144 to 1,048,576.
- **check:** pass
- **compare:** significantly faster, 1.017x [1.010 – 1.023]
- **Kept:** yes
- **Notes:** 1.21x at 262,144 (17.4 us against 21.3), nothing elsewhere.

## 5. Quads side by side up to 4,194,304 elements

- **Idea:** attempt 4 one step further: `SEQ_MIN` 4,194,304, so that 1,048,576 (8 MB, in the third-level cache) also takes the side-by-side quads.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.992 – 1.007]
- **Kept:** no
- **Notes:** 1.02x at 1,048,576 (98.5 us against 100.7), 1.01x at 4,194,304, the rest level. Beyond the second-level cache both orders run at about 84 GB/s, the third-level cache's rate for one core, and the order does not matter.

## 6. Two of a quad's eight chains as `vfmadd231pd` with 1.0

- **Idea:** this core adds on two ports (1 and 5), and the shuffles of each quad's combination take port 5 too; its fused multiply-add also runs on port 0, which the loop leaves idle. `vfmadd231pd s, one, [x]` computes x * 1.0 + s with one rounding, and x * 1.0 is x exactly, so the result is the addition s + x, bit for bit, signed zeros and denormals included: no multiplication of the reference is fused, and no addition is changed. Two of the eight chains of the side-by-side quad loop use it (the second register of leaves 0 and 2); 1.0 is broadcast to `ymm15` when the walk starts.
- **check:** pass
- **compare:** significantly faster, 1.011x [1.007 – 1.015]
- **Kept:** yes
- **Notes:** 1.05x at 1,024, 1.06x at 4,096 (132.5 ns), nothing elsewhere. In the micro-benchmark three or four such chains instead of two change nothing (125.2 and 125.4 ns at 4,096 with `x` on a 64-byte boundary, against 125.6): what is left is the loads, two 32-byte loads a cycle on this core.

## 7. The quad loop on four pointers instead of base plus index

- **Idea:** knock-outs in the micro-benchmark (warm, `x` on a 64-byte boundary, one quad of 512 elements): the loop with its adds replaced by plain loads took 11.3 ns, with no combination 14.6, the whole quad 15.7. The loads alone were not the limit. A `vaddpd` or `vfmadd231pd` with a base-plus-index memory operand is split into two micro-operations when it is renamed, so a step of eight was 18 at the six-wide rename. Each leaf now has its own pointer, advanced by 64 bytes a step: 13 micro-operations a step. Whether a leaf has q + 1 blocks is then read from the distance to the next leaf's pointer.
- **check:** pass
- **compare:** significantly faster, 1.034x [1.030 – 1.037]
- **Kept:** yes
- **Notes:** 1.19x at 1,024 (26.9 ns), 1.18x at 4,096 (112.1 ns), 1.02x at 16,384 and 65,536, 1.01x at 262,144. Micro-benchmark: 12.9 ns a quad on a 64-byte boundary, 13.9 ns 16 bytes off it.
- **Not taken to the judge:** `prefetcht0` 512 or 2,048 bytes ahead on each of the quad's four pointers, for the second-level cache. Micro-benchmark, 16 bytes off a boundary: 865 and 815 ns at 16,384 against 794, and 136 ns at 4,096 against 113.

## 8. The pair on two pointers, and a pair of the whole array without the walk

- **Idea:** attempt 7 for the pair loop, and 129 to 247 elements (and 248 and 256), a node whose halves are both leaves, summed straight from the entry: no frame, no 1.0 broadcast and no masks. The pair became a macro used in both places.
- **check:** pass
- **compare:** significantly faster, 1.009x [1.001 – 1.017]
- **Kept:** yes
- **Notes:** 1.17x at 256 (8.4 ns against 9.4); 0.99x at 1,024 and 16,384, a path this does not change.

## 9. `prefetcht0` 2 KB ahead in a leaf summed on its own

- **Idea:** a run recorded after attempt 8 had the kernel 0.98x [0.97 – 0.99] the naive C at 16,777,216 elements (4.18 ms against 4.12) and level at 1,048,576 and 4,194,304. There every quad's leaves are summed one after another (attempt 2), one stream from memory. One `prefetcht0` per 64-byte step, `SEQ_AHEAD` = 2,048 bytes ahead, in that loop only; the extra load-port operation costs nothing at the speed of memory.
- **check:** pass
- **compare:** significantly faster, 1.013x [1.010 – 1.015]
- **Kept:** yes
- **Notes:** 1.07x at 4,194,304 and at 16,777,216 (4.05 ms against 4.25), level at 1,048,576.

## 10. `prefetcht0` 4 KB ahead

- **Idea:** attempt 9 with `SEQ_AHEAD` = 4,096.
- **check:** pass
- **compare:** no significant difference, 1.009x [1.000 – 1.019]
- **Kept:** no
- **Notes:** 1.05x at 4,194,304 and 1.03x at 16,777,216 over attempt 9, 1.02x [0.98 – 1.07] at 1,048,576, nothing else; overall just short of the rule.

## Where this run stands

Ten attempts, seven kept. Where each size now stops, on this processor:

- **Up to 128 elements** one leaf, two chains of dependent additions: the reference's own latency, less its overhead (1.95x at 16, 1.58x at 64).
- **In the first-level cache** the quads' loop is at the loads: a quad of 512 elements takes 12.9 ns warm on a 64-byte boundary, and the same loop with its adds replaced by plain loads 11.3 ns. Neither more `vfmadd231pd` chains (attempt 6) nor fewer passes of the walk could take more than the remaining 1.6 ns, which is the combination: 4,096 elements (eight quads) cost 104 ns in the micro-benchmark, eight times one quad.
- **In the second-level cache** (16,384 to 262,144 elements) about 32 bytes a cycle, that cache's rate for one core with four streams; software prefetch made it slower.
- **Beyond it** the third-level cache and memory, at the rate the naive C also gets from them; attempt 9's prefetch gives memory a few percent.

The recorded run was taken again when the binding was added, after the spec and reference had reached `main`: `uv run judge run sum_f64 --record`, locked, on the Intel Core i5-14600KF at a load average of 0.9 to 1.0 with no other judge running. 1.88x [1.88 – 1.89] over the naive C and 10.8x over NumPy. At 1,048,576 elements it is 1.20x [1.19 – 1.20] (85.3 us against 102.0), where the run of this log's building, at a load average of 1.7 to 1.9 with other judges sharing the third-level cache, had 0.99x: that size is in the third-level cache, and its figure depends on what else is using it. The function has no output buffer, so `run` prints no second table.
