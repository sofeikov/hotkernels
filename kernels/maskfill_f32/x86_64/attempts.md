# maskfill_f32 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The spec and the naive C are new in this run, so every judge command was run with `--unlocked`.

What the compiler does with `naive.c` (`clang -O3 -march=native`): unlike `maskfill_f64`'s, it vectorises. Eight mask bytes are widened with `vpmovzxbd`, shifted to the sign bit with `vpslld 31` and stored with `vmaskmovps`, four times per iteration, after a check that x and mask do not overlap. So the naive C is already a masked store per eight elements, 0.03 ns per element at 1,024, and that is the bar. (GCC 15 makes a branch per element; the judge builds with clang.)

Masked stores (`vmaskmovps`) are used by the kept kernel. They are fast on this core and reported slow on some AMD processors.

## 1. Baseline: `maskfill_f64`'s kernel with eight lanes

- **Idea:** the x86-64 `maskfill_f64` scheme. Blocks of 32: one `vpcmpeqb` and `vpmovmskb` give the mask word; a block with at most four selected elements stores the value to each with `tzcnt`, `blsr` and a `cmovnc`, no load; any other block is blended eight elements at a time (`vpmovsxbd`, `vpcmpeqd` with zero, `vpand`, `vpandn`, `vpor`, store back). Groups of 8, then a scalar loop.
- **check:** pass
- **compare:** not run (first version). `run`: 0.86x over naive C overall (half_selected 0.83x, few_selected 0.90x), 25.4x over NumPy.
- **Kept:** yes, as the first version, to have a HEAD to compare against.
- **Notes:** half_selected 0.53x to 0.69x from 64 to 4,096 elements, level from 16,384. The blend takes five vector instructions on three ports for eight elements, where the compiler's masked store takes three and has no load of x. The sparse block did not beat four masked stores either: few_selected 0.69x to 0.84x up to 16,384.

## 2. A masked store in place of the blend

- **Idea:** what the compiler does, written out. `vpmovzxbd`, `vpcmpgtd` against zero (all ones in a lane whose mask byte is not zero, whatever its value, where the compiler's shift takes bit 0 only) and `vmaskmovps`. No load of x, and an unselected element is not written at all. No sparse path. Blocks of 32, groups of 8, and the last 8 elements by one masked store that overlaps the group before it (storing the value twice to an element changes nothing); below 8 elements a scalar loop.
- **check:** pass
- **compare:** significantly faster, 1.178x [1.153 – 1.204]
- **Kept:** yes
- **Notes:** `run`: 0.98x over naive C (0.88x to 1.20x by size), 29.0x over NumPy. It is the compiler's loop, so level with it is what it should be. Unlike `maskfill_f64` (its attempt 3, not kept there for this reason), this kernel never rewrites an element the mask does not select.

## 3. Bring x to a 32-byte boundary

- **Idea:** the judge's arrays start 16 bytes into a cache line, so every second masked store straddles two lines. One masked store does the first eight elements, then x and mask move on by up to seven elements to a 32-byte boundary of x (an x that is not 4-byte aligned moves on by whole elements and stays unaligned).
- **check:** pass
- **compare:** significantly faster, 1.061x [1.057 – 1.065]
- **Kept:** yes
- **Notes:** 1.38x to 1.43x at 65,536 and 262,144, 1.10x at 1,048,576, level from 256 to 16,384, 0.87x to 0.92x at 16 and 64 for the extra store.

## 4. No alignment below 64 elements

- **Idea:** recover the small sizes attempt 3 lost by skipping the head below 64 elements.
- **check:** pass
- **compare:** significantly slower, 0.991x [0.989 – 0.994]
- **Kept:** no
- **Notes:** 0.90x at 16 (one more compare and branch on that path), level at 64 and everywhere else.

## 5. Prefetch x's lines ahead of the masked stores

- **Idea:** a masked store does not load, so nothing asks for x's lines before the store needs them (`maskfill_f64`'s attempt 3 suspected this). Two `prefetcht0` per block of 32, 512 bytes ahead.
- **check:** pass
- **compare:** significantly slower, 0.983x [0.981 – 0.985]
- **Kept:** no
- **Notes:** 1.06x to 1.10x at 16,384 to 262,144 with half selected, 0.92x to 0.95x up to 4,096, and few_selected 0.93x almost everywhere. Two attempts in a row not faster: the run stops here.

## Where it stands

Final `judge run` (load average 0.97 before, 1.04 after):

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.9 ns | 2.1 ns | 1.12x [1.12 – 1.12] | 308.8 ns | 167x [167 – 167] |
| 64 | 3.4 ns | 2.9 ns | 0.84x [0.84 – 0.84] | 357.7 ns | 104x [104 – 104] |
| 256 | 8.5 ns | 9.1 ns | 1.07x [1.07 – 1.07] | 548.1 ns | 64.8x [64.78 – 64.91] |
| 1,024 | 30.0 ns | 31.0 ns | 1.03x [1.03 – 1.03] | 1.4 us | 46.2x [46.17 – 46.21] |
| 4,096 | 115.5 ns | 116.0 ns | 1.00x [1.00 – 1.00] | 4.6 us | 40.0x [39.70 – 40.14] |
| 16,384 | 858.2 ns | 1.1 us | 1.32x [1.32 – 1.32] | 39.2 us | 45.7x [45.59 – 45.91] |
| 65,536 | 3.4 us | 3.5 us | 1.02x [1.02 – 1.02] | 188.9 us | 55.1x [55.04 – 55.19] |
| 262,144 | 14.0 us | 17.9 us | 1.28x [1.27 – 1.28] | 804.3 us | 57.3x [57.20 – 57.37] |
| 1,048,576 | 89.7 us | 96.9 us | 1.08x [1.08 – 1.08] | 3.21 ms | 35.8x [35.82 – 35.85] |
| 4,194,304 | 389.0 us | 405.3 us | 1.04x [1.02 – 1.07] | 12.87 ms | 33.1x [32.52 – 33.62] |
| 16,777,216 | 3.42 ms | 3.44 ms | 1.01x [1.00 – 1.01] | 51.82 ms | 15.2x [15.11 – 15.21] |
| overall |  |  | 1.07x [1.06 – 1.07] |  | 50.5x [50.38 – 50.56] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 1.9 ns | 2.1 ns | 1.10x [1.10 – 1.10] | 254.9 ns | 135x [135 – 135] |
| 64 | 3.4 ns | 2.8 ns | 0.83x [0.83 – 0.84] | 302.9 ns | 90.3x [90.25 – 90.38] |
| 256 | 8.7 ns | 7.6 ns | 0.87x [0.87 – 0.87] | 351.2 ns | 40.3x [40.33 – 40.37] |
| 1,024 | 28.9 ns | 28.2 ns | 0.97x [0.97 – 0.98] | 590.0 ns | 20.4x [20.38 – 20.40] |
| 4,096 | 111.2 ns | 113.5 ns | 1.02x [1.02 – 1.02] | 1.4 us | 13.0x [12.99 – 13.02] |
| 16,384 | 587.1 ns | 770.0 ns | 1.31x [1.31 – 1.31] | 4.7 us | 7.97x [7.97 – 7.98] |
| 65,536 | 2.0 us | 2.0 us | 1.00x [1.00 – 1.00] | 29.4 us | 14.9x [14.85 – 14.93] |
| 262,144 | 7.9 us | 11.0 us | 1.39x [1.39 – 1.40] | 179.6 us | 22.7x [22.65 – 22.72] |
| 1,048,576 | 70.1 us | 76.2 us | 1.09x [1.08 – 1.09] | 765.9 us | 10.9x [10.91 – 10.94] |
| 4,194,304 | 287.0 us | 285.5 us | 0.99x [0.99 – 1.00] | 3.07 ms | 10.7x [10.66 – 10.73] |
| 16,777,216 | 2.93 ms | 3.08 ms | 1.05x [1.04 – 1.06] | 12.53 ms | 4.28x [4.26 – 4.29] |
| overall |  |  | 1.05x [1.04 – 1.05] |  | 19.7x [19.71 – 19.73] |

All distributions: 1.06x [1.06 – 1.06] over naive C, 31.5x [31.52 – 31.58] over NumPy.

The kernel writes no buffer other than x, so the judge has no close-placement table for it.

What holds it:

- **One masked store per eight elements, about 1.2 cycles each in the first-level cache,** 0.028 ns per element from 1,024 to 4,096 whatever the mask. The naive C is the same loop by the compiler, so the margin over it is small: 1.06x overall, all of it from the 32-byte alignment of attempt 3 (1.3x to 1.4x at 16,384 and 262,144), with 0.83x to 0.87x at 64 and 256 where the alignment's extra store is not paid back.
- **Ships as assembly** because it is significantly faster than the naive C on the judge, and because the naive C reaches this speed only when built with `-march=native`; a package built for x86-64 at large is not.
- **It never writes an element the mask does not select**, unlike `maskfill_f64` on either architecture.

The candidates the design lists for `maskfill_f64` (`docs/x86-64-design.md`), as they came out here:

| Candidate | Outcome |
|---|---|
| A masked store of only the selected elements | Kept (attempt 2) |
| The load, blend and store | Rejected (attempt 1) |
| Masked loads and stores for leftover elements | The last 8 by an overlapping masked store; below 8 elements a scalar loop, since the mask bytes past n may not be read |
| Non-temporal stores | Not tried: a masked non-temporal store does not exist in AVX2 |
| Alignment of data | Kept (attempt 3) |
| Prefetch ahead of the masked stores | Rejected (attempt 5) |
| An output just behind the input | Does not apply: one buffer |
| Instructions slow on other x86-64 processors | `vmaskmovps` |

Left untried: a sparse path for blocks with no selected element (at 5% selected, one block of 32 in five is empty, too few to pay for a branch the predictor cannot learn).

Found here and worth trying on ARM64: nothing; the kernel ships there as its naive C.
