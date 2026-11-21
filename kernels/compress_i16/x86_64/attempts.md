# compress_i16 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The spec and the reference were new in this run, so every judge command ran with `--unlocked`; they were committed before any assembly and not changed after.

What the compiler does with `naive.c` (`clang -O3 -march=native`, and the same for the x86-64-v3 of the Linux wheel): the scalar loop, about 0.19 ns per element at every size up to 1,048,576, as for `compress_f32` and `compress_u8`.

The technique is `compress_u8`'s with 16-bit elements: a group of eight is one `xmm` register, the table entry for a mask byte holds the two byte indices of each selected element, so an entry is 16 bytes and the table 4 KB.

## 1. Baseline: `vpshufb` by a 256-entry table per eight elements

- **Idea:** `compress_u8`'s baseline with 16-byte groups: `vmovdqu` in and out, the write position counted in elements and scaled by 2 in the store's address, each group's table offset from the mask word by `rorx` and `and` after one shift by 4.
- **check:** pass
- **compare:** not run (first version). `run`: 3.26x [3.26 – 3.26] over naive C overall (half_selected 3.24x, few_selected 3.27x), 29.7x over NumPy.
- **Kept:** yes
- **Notes:** 0.042 ns per element at 1,024. `compress_u8`'s attempt 3 (each group's byte by `movzx` from `al` or `ah`) does not carry over: an entry of 16 bytes cannot be addressed as the byte scaled in the address, whose largest scale is 8, so each byte would need a shift as well, the same count as now. Its attempt 2 fails the same way.

## 2. Sixteen elements per `vpshufb` on a `ymm` register

- **Idea:** halve the loads of x and the shuffles. `vpshufb` shuffles each 16-byte lane by its own control, so two groups' entries make one control (`vmovdqa` and `vinserti128` from memory); the low lane is stored with `vmovdqu`, the high one with `vextracti128` to memory at the advanced position.
- **check:** pass
- **compare:** significantly slower, 0.983x [0.973 – 0.994]
- **Kept:** no
- **Notes:** the same 14 instruction slots per 16 elements as two steps of eight, and no size gained.

## 3. `pdep` spreads the mask word into four scaled table offsets

- **Idea:** one `pdep` with the mask `0x0ff00ff00ff00ff0` moves the four bytes of the mask word into the four 16-bit fields of `rax`, each already shifted left by 4. The groups' offsets are then `movzx` of `ax`, `eax` shifted right by 16, `rorx` by 32 and `movzx`, and `rax` shifted right by 48: seven instructions per block for the four offsets, against nine. The constant lives in `rcx`, so the block loop counts the elements left in `r11` and the leftover count is taken from it.
- **check:** pass
- **compare:** significantly faster, 1.017x [1.011 – 1.022]
- **Kept:** yes
- **Notes:** 1.01x to 1.03x at most sizes. `pdep` is fast on this processor and slow on AMD processors before Zen 3 (microcode, tens of cycles), where this kernel would lose much of its margin. Stopped here: two instruction slots fewer per block of 32 gave 1.7%, the loop is at about 33 slots per block on a core that renames six per cycle, and the one arrangement with fewer loads cost as many slots (attempt 2).

## Where it stands

Final `judge run` (load average 1.1, nothing else running):

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 3.8 ns | 1.61x [1.60 – 1.61] | 268.4 ns | 114x [114 – 114] |
| 64 | 4.6 ns | 13.9 ns | 3.00x [3.00 – 3.01] | 330.5 ns | 71.5x [71.44 – 71.59] |
| 256 | 11.7 ns | 47.1 ns | 4.03x [4.02 – 4.03] | 574.8 ns | 49.1x [49.07 – 49.13] |
| 1,024 | 43.1 ns | 192.4 ns | 4.47x [4.46 – 4.47] | 1.7 us | 38.5x [38.49 – 38.60] |
| 4,096 | 168.9 ns | 776.2 ns | 4.60x [4.59 – 4.60] | 5.6 us | 33.0x [32.96 – 33.08] |
| 16,384 | 820.8 ns | 3.1 us | 3.78x [3.78 – 3.78] | 46.3 us | 56.4x [56.31 – 56.46] |
| 65,536 | 3.3 us | 12.8 us | 3.93x [3.93 – 3.93] | 216.5 us | 66.4x [66.33 – 66.48] |
| 262,144 | 13.3 us | 50.0 us | 3.75x [3.75 – 3.76] | 894.9 us | 67.1x [67.05 – 67.17] |
| 1,048,576 | 60.8 us | 203.2 us | 3.34x [3.24 – 3.39] | 3.60 ms | 59.2x [57.33 – 59.96] |
| 4,194,304 | 258.5 us | 807.2 us | 3.12x [3.09 – 3.18] | 14.45 ms | 55.9x [55.38 – 57.03] |
| 16,777,216 | 2.06 ms | 3.68 ms | 1.79x [1.77 – 1.80] | 58.35 ms | 28.3x [28.10 – 28.49] |
| overall |  |  | 3.24x [3.23 – 3.25] |  | 54.3x [54.08 – 54.34] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 3.8 ns | 1.59x [1.59 – 1.60] | 194.4 ns | 81.1x [80.82 – 81.39] |
| 64 | 4.2 ns | 12.8 ns | 3.08x [3.07 – 3.08] | 258.3 ns | 61.9x [61.80 – 61.91] |
| 256 | 11.7 ns | 47.2 ns | 4.04x [4.03 – 4.05] | 323.5 ns | 27.7x [27.63 – 27.69] |
| 1,024 | 43.2 ns | 192.8 ns | 4.46x [4.46 – 4.47] | 566.2 ns | 13.1x [13.09 – 13.11] |
| 4,096 | 158.5 ns | 775.8 ns | 4.89x [4.89 – 4.90] | 1.6 us | 10.00x [9.98 – 10.02] |
| 16,384 | 725.1 ns | 3.2 us | 4.41x [4.41 – 4.41] | 5.0 us | 6.90x [6.89 – 6.91] |
| 65,536 | 3.1 us | 12.8 us | 4.17x [4.16 – 4.17] | 44.8 us | 14.6x [14.57 – 14.64] |
| 262,144 | 11.6 us | 50.1 us | 4.32x [4.31 – 4.32] | 211.6 us | 18.2x [18.18 – 18.26] |
| 1,048,576 | 47.2 us | 201.3 us | 4.26x [4.25 – 4.26] | 868.5 us | 18.4x [18.35 – 18.40] |
| 4,194,304 | 217.2 us | 807.2 us | 3.72x [3.68 – 3.75] | 3.45 ms | 15.9x [15.68 – 16.01] |
| 16,777,216 | 1.55 ms | 3.42 ms | 2.20x [2.20 – 2.22] | 14.13 ms | 9.10x [9.06 – 9.16] |
| overall |  |  | 3.57x [3.57 – 3.57] |  | 18.5x [18.51 – 18.55] |

All distributions: 3.40x [3.40 – 3.40] over naive C, 31.7x [31.65 – 31.74] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 158.0 ns | 168.9 ns | 1.07x [1.07 – 1.07] | 775.8 ns | 4.91x [4.90 – 4.92] |
| 16,384 | 809.4 ns | 797.5 ns | 0.99x [0.98 – 0.99] | 3.1 us | 3.83x [3.83 – 3.83] |
| 65,536 | 3.2 us | 3.3 us | 1.01x [1.01 – 1.01] | 12.5 us | 3.90x [3.89 – 3.90] |
| 262,144 | 13.1 us | 13.3 us | 1.01x [1.01 – 1.02] | 50.2 us | 3.82x [3.82 – 3.82] |
| 1,048,576 | 58.0 us | 95.5 us | 1.65x [1.65 – 1.65] | 203.2 us | 3.51x [3.50 – 3.51] |
| 4,194,304 | 427.7 us | 480.1 us | 1.12x [1.11 – 1.14] | 845.4 us | 1.98x [1.96 – 1.99] |
| 16,777,216 | 1.96 ms | 2.05 ms | 1.05x [1.04 – 1.05] | 3.48 ms | 1.77x [1.77 – 1.77] |
| overall |  |  | 1.11x [1.11 – 1.11] |  | 3.20x [3.20 – 3.20] |

The close table's row at 1,048,576 (1.65x) is the "apart" timing of that run, 95.5 us against 60.8 us in the first table: noise in the reference, not a gain from the close placement. The other rows are 0.99x to 1.12x.

What holds it at each size:

- **16:** the call and the scalar loop of the last 1 to 7 elements.
- **64 to 4,096:** the instruction count, 0.04 ns per element.
- **16,384 to 4,194,304:** 0.05 to 0.06 ns per element with half selected, the cache-line traffic of an output twice `compress_u8`'s, as `compress_f32`'s log found for 32-bit elements.
- **16,777,216:** memory. 2.06 ms for 32 MB of data and 16 MB of mask with half selected, 1.8x over the naive C.

The candidates the design lists for this family (`docs/x86-64-design.md`), where they differ from `compress_f32`:

| Candidate | Outcome |
|---|---|
| 32 mask bytes to a 32-bit mask in one `vpmovmskb` | Kept (attempt 1) |
| A shuffle moves the selected elements to the front by index | Kept (attempt 1): `vpshufb` on eight 16-bit elements; on `ymm` rejected (attempt 2) |
| The index from a table | Kept: 256 entries of 16 bytes, 4 KB |
| `pext`/`pdep` | `pdep` kept to spread the mask word (attempt 3) |
| `popcnt` advances the write pointer | Kept, as an element count |
| Masked loads and stores for leftover elements | Not tried, as for `compress_f32` |
| Alignment of data, prefetch | Not tried: both were slower on `compress_f32` |
| Instructions slow on other x86-64 processors | `pdep` (attempt 3) |
