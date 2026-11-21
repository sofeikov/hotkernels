# isin_i32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core), Linux, pinned to one performance core by the judge. The spec and reference are new in this pull request, so every judge command ran with `--unlocked`.

What the compiler does with `naive.c` (`clang -O3`, x86-64-v3 as the Linux wheels build it): it vectorises, eight elements per register, eight `vpcmpeqd` against the broadcast test values, an OR tree and a narrowing of the masks to bytes: 0.16 ns per element at every size up to 1,048,576 and for any values. NumPy takes 8.3 ns per element at 1,048,576 on small values and 0.73 ns on values spread over the int32 range.

The plan was `isin_i64`'s x86-64 kernel at half the width: 16-bit keys where the test values allow, 32-bit compares where they do not. Two things are simpler at 32 bits. `vpackssdw` narrows 32 to 16 bits in one step, and its saturation needs no check of the block: an element outside the test values' range becomes -32,768 or 32,767, which is no test key once the test keys are kept from -32,767 to 32,766. And a wrapping subtraction moves any window of 65,534 values onto that range, so the window can sit wherever the test values are, which `isin_i64` left untried.

## 1. 16-bit keys by a wrapping subtraction and a saturating pack; 32-bit compares otherwise

- **Idea:** 32 elements per block. When the test values lie within 65,533 of each other, each element x becomes c - x with c = smallest test value + 32,766 (one `vpsubd` with the load folded in), two `vpackssdw` give 32 saturated 16-bit keys in two registers, and sixteen `vpcmpeqw` with seven ORs per register test them; `vpacksswb`, a `vpermd` and `vpabsb` give 32 bytes in element order. Otherwise eight `vpcmpeqd` per eight elements against test values broadcast in registers. 8 to 31 elements are done eight at a time at 32 bits, fewer one at a time.
- **check:** pass. Differential test outside the judge against `np.isin`: 33,480 calls, n from 0 to 199 and random up to 4,088, every 4-byte offset within 32 bytes, x and out ending at an inaccessible page, test sets of spread 0, 3, 255, 65,532 to 65,535 and 70,000 at random bases, the edges -32,768 / -32,767 / 32,766 / 32,767 and both ends of int32, elements drawn from the test values, from just outside the window, from where a wrong window or a wrong saturation would match, and at random. No mismatch. The test was checked by breaking the kernel: a spread limit one too large gives 4,140 mismatches, a wrong `vpermd` order 12,908.
- **compare:** not run (first committed version). `run`: 1.32x over naive C overall (`mostly_misses` 1.51x, `wide_values` 1.16x), 55.8x over NumPy.
- **Kept:** yes
- **Notes:** narrow path 0.088 ns per element at 4,096, about 15 cycles per 32 elements for 39 vector instructions.

## 2. A fused compare and branch closes each loop; the last block after it

- **Idea:** the loops closed with `cmp`, `jae`, `add`, `cmp`, `cmova`, `jmp`, six scalar instructions, some of them on the ports the vector instructions need. Now `add`, `cmp`, `jbe`, and the overlapping last block is done once after the loop, the block's code a macro used twice.
- **check:** pass; differential test 0 mismatches.
- **compare:** significantly faster, 1.027x [1.026 – 1.028]
- **Kept:** yes
- **Notes:** 1.04x on the narrow path, nothing on the wide one; 1.13x at 16 elements, which runs neither loop, so that part is code placement.

## 3. Test values from -32,767 to 32,766 are their own keys

- **Idea:** a set of small integers needs no subtraction: every value is already a key in range. A second narrow loop loads the elements straight into the packs (`vpackssdw` with the second operand from memory). Taken when the smallest test value is at least -32,767 and the largest at most 32,766; other sets within 65,533 of each other keep the subtraction.
- **check:** pass; differential test 0 mismatches.
- **compare:** significantly faster, 1.071x [1.070 – 1.071]
- **Kept:** yes
- **Notes:** 1.15x on `mostly_misses` up to 4,194,304, more than four instructions in 39 would suggest. `wide_values` unchanged.

## Where it stands

Final `judge run` (load average 0.39 before it):

`mostly_misses`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.9 ns | 3.4 ns | 1.17x [1.17 – 1.18] | 7.8 us | 2689x [2684 – 2694] |
| 64 | 7.3 ns | 13.1 ns | 1.80x [1.79 – 1.80] | 7.9 us | 1083x [1082 – 1084] |
| 256 | 23.3 ns | 41.6 ns | 1.78x [1.78 – 1.78] | 8.6 us | 369x [369 – 369] |
| 1,024 | 78.6 ns | 162.9 ns | 2.07x [2.07 – 2.08] | 10.5 us | 133x [133 – 133] |
| 4,096 | 299.9 ns | 651.6 ns | 2.17x [2.17 – 2.17] | 19.2 us | 64.1x [64.05 – 64.11] |
| 16,384 | 1.2 us | 2.6 us | 2.19x [2.19 – 2.19] | 79.0 us | 66.5x [66.48 – 66.57] |
| 65,536 | 4.7 us | 10.4 us | 2.23x [2.23 – 2.23] | 224.2 us | 48.2x [48.13 – 48.18] |
| 262,144 | 18.9 us | 41.5 us | 2.20x [2.20 – 2.20] | 721.3 us | 38.2x [38.14 – 38.20] |
| 1,048,576 | 81.1 us | 166.4 us | 2.05x [2.04 – 2.07] | 8.74 ms | 108x [107 – 109] |
| 4,194,304 | 915.7 us | 1.01 ms | 1.10x [1.09 – 1.20] | 36.80 ms | 40.2x [39.87 – 41.28] |
| 16,777,216 | 3.25 ms | 4.10 ms | 1.26x [1.25 – 1.27] | 85.59 ms | 26.3x [26.23 – 26.54] |
| overall |  |  | 1.76x [1.76 – 1.78] |  | 126x [126 – 126] |

`wide_values`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.9 ns | 3.3 ns | 1.16x [1.16 – 1.16] | 7.8 us | 2731x [2729 – 2732] |
| 64 | 10.4 ns | 13.0 ns | 1.25x [1.25 – 1.25] | 7.8 us | 755x [754 – 756] |
| 256 | 35.8 ns | 41.6 ns | 1.16x [1.16 – 1.16] | 7.9 us | 221x [221 – 222] |
| 1,024 | 140.3 ns | 163.0 ns | 1.16x [1.16 – 1.16] | 8.9 us | 63.5x [63.32 – 63.68] |
| 4,096 | 558.7 ns | 652.0 ns | 1.17x [1.17 – 1.17] | 10.4 us | 18.6x [18.56 – 18.59] |
| 16,384 | 2.2 us | 2.6 us | 1.16x [1.16 – 1.16] | 16.2 us | 7.21x [7.21 – 7.22] |
| 65,536 | 8.9 us | 10.4 us | 1.16x [1.16 – 1.16] | 37.6 us | 4.21x [4.21 – 4.21] |
| 262,144 | 35.7 us | 41.5 us | 1.16x [1.16 – 1.16] | 126.1 us | 3.53x [3.53 – 3.53] |
| 1,048,576 | 144.0 us | 166.5 us | 1.16x [1.16 – 1.16] | 760.5 us | 5.28x [5.28 – 5.28] |
| 4,194,304 | 753.2 us | 880.2 us | 1.17x [1.16 – 1.18] | 4.95 ms | 6.58x [6.51 – 6.66] |
| 16,777,216 | 3.50 ms | 4.04 ms | 1.16x [1.15 – 1.16] | 30.29 ms | 8.67x [8.64 – 8.72] |
| overall |  |  | 1.17x [1.17 – 1.17] |  | 29.9x [29.92 – 29.99] |

All distributions: 1.44x [1.44 – 1.44] over naive C, 61.4x [61.40 – 61.54] over NumPy.

With the output buffer 16 bytes behind its input (`mostly_misses`): 1.04x [1.04 – 1.04] overall against the output clear of the input, from 0.99x to 1.20x by size. The close placement costs nothing.

What holds it:

- **Narrow path, up to 1,048,576:** the vector ports. Per 32 elements, with the test values their own keys: two packs, sixteen compares, fourteen ORs, a pack, a `vpermd` and a `vpabsb`, 35 instructions on three ports, 11.7 cycles; measured about 12.
- **Narrow path, from 4,194,304:** memory, 5 bytes per element; 1.1x to 1.3x over the naive C.
- **Wide path:** 32 compares per 32 elements on the two ports that take them, 16 cycles, and 28 ORs beside them: about 22 cycles for 65 instructions on three ports, the compiler's loop written by hand. 1.16x.

Stopped here: both paths are at a limit of the ports.

Left untried:

- **8-bit keys** for test values within 253 of each other (`np.isin(x, [1, 2, 3])` padded to eight is one): a second saturating pack, `vpacksswb`, gives 32 keys in one register and halves the compares, about 24 instructions per 32 elements. Neither of the spec's distributions has such a set, so `compare` could not have kept it.
- **Masked loads and stores for leftover elements:** not needed, the leftover is one more block overlapping the one before.

Instructions slow on other x86-64 processors: none used.

**What differs from the ARM64 kernel for a caller:** nothing. Both take any eight int32 test values; the ARM64 kernel is the naive C.
