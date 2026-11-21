# compress_u8 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The spec and the reference were new in this run, so every judge command ran with `--unlocked`; they were committed before any assembly and not changed after.

What the compiler does with `naive.c` (`clang -O3 -march=native`, and the same for the x86-64-v3 of the Linux wheel): the scalar loop of `compress_f32`, one load, one store and one add of the mask byte per element, about 0.19 ns per element at every size up to 1,048,576. NumPy takes 3.5 ns per element for `x[mask]` at 1,048,576 with half selected.

The technique is `compress_f32`'s (read its log): a mask word of 32 bits from one `vpcmpeqb` and one `vpmovmskb`, a table indexed by each byte of it, a shuffle that moves the selected elements of a group of eight to the front, a store of the whole group at the write position, and `popcnt` to advance it. For bytes the shuffle is `vpshufb` on an `xmm` register, the group is eight bytes (`vmovq` in and out) and an entry is eight byte indices, so the table is 2 KB.

## 1. Baseline: `vpshufb` by a 256-entry table per eight elements

- **Idea:** `compress_f32`'s loop with `vpshufb` in place of `vpermps`: blocks of 32 elements, each group's table offset from the mask word by `rorx` and `and`, eight bytes loaded with `vmovq`, shuffled by the table entry as the instruction's memory operand, stored with `vmovq`. `vpshufb` reads 16 bytes of its control, so the table has eight bytes of padding after its last entry. 8 to 31 leftover elements take one table step per 8, the last 1 to 7 the scalar loop of the reference.
- **check:** pass
- **compare:** not run (first version). `run`: 3.70x [3.69 – 3.70] over naive C overall (half_selected 3.62x, few_selected 3.78x), 35.3x over NumPy.
- **Kept:** yes
- **Notes:** branch-free, so both distributions cost the same: 0.044 ns per element at 1,024, about 1.9 cycles per group of eight, the same as `compress_f32` per group.

## 2. The mask word below the stack pointer, each group's byte loaded back

- **Idea:** what `compress_f32`'s log left untried. Store the mask word once in the red zone and load each group's byte with `movzx`, the entry addressed as the byte scaled by 8: one load per group in place of a shift and an `and`, five instruction slots fewer per block of 32.
- **check:** pass
- **compare:** significantly slower, 0.978x [0.973 – 0.982]
- **Kept:** no
- **Notes:** 1.03x to 1.08x from 64 to 4,096, 0.90x to 0.96x from 65,536 up. The store forwarding puts about five cycles between the compare and the first table load of each block.

## 3. Each group's mask byte by one `movzx` from `al` or `ah`

- **Idea:** the same saving without the memory round trip. The mask word stays in `eax`; the groups are `al` and `ah`, then `al` and `ah` again after one `shr eax, 16`, each moved to `ecx` with `movzx` and scaled by 8 in the table address. `ah` can only be read into a register without a REX prefix, so the group's byte goes to `ecx`, and the count of leftover elements moves to `r8`. Five instructions per block for the four offsets, against nine.
- **check:** pass
- **compare:** significantly faster, 1.101x [1.091 – 1.111]
- **Kept:** yes
- **Notes:** 1.10x to 1.18x at every size from 256 to 1,048,576, 1.02x to 1.04x at 16,777,216. 31 instruction slots per block of 32 on a core that renames six per cycle is 5.2 cycles; the loop takes about 6 at 4,096. That is the limit named in `compress_f32`'s log, the instruction count, and the stopping point here: every other arrangement tried on either width (attempt 2 here, attempts 2 and 3 of `compress_i16`) costs as many slots or more.

## Where it stands

Final `judge run` (load average 1.1, nothing else running):

`half_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 4.1 ns | 1.73x [1.73 – 1.73] | 275.9 ns | 117x [117 – 117] |
| 64 | 3.4 ns | 12.2 ns | 3.60x [3.59 – 3.60] | 339.7 ns | 101x [100 – 101] |
| 256 | 10.6 ns | 47.1 ns | 4.44x [4.43 – 4.45] | 591.4 ns | 55.7x [55.62 – 55.77] |
| 1,024 | 40.8 ns | 192.4 ns | 4.72x [4.71 – 4.72] | 1.7 us | 41.6x [41.55 – 41.63] |
| 4,096 | 153.0 ns | 775.8 ns | 5.07x [5.06 – 5.08] | 5.7 us | 37.0x [36.94 – 37.04] |
| 16,384 | 622.9 ns | 3.1 us | 4.97x [4.97 – 4.97] | 46.8 us | 75.1x [75.02 – 75.15] |
| 65,536 | 2.5 us | 12.4 us | 4.89x [4.89 – 4.90] | 221.6 us | 87.6x [87.46 – 87.67] |
| 262,144 | 10.0 us | 50.3 us | 5.04x [5.04 – 5.05] | 919.4 us | 92.2x [92.07 – 92.33] |
| 1,048,576 | 46.4 us | 202.4 us | 4.36x [4.32 – 4.42] | 3.69 ms | 79.6x [78.84 – 80.53] |
| 4,194,304 | 184.6 us | 802.8 us | 4.35x [4.08 – 4.47] | 14.77 ms | 80.0x [75.07 – 82.30] |
| 16,777,216 | 1.23 ms | 3.28 ms | 2.66x [2.65 – 2.67] | 59.46 ms | 48.2x [47.98 – 48.43] |
| overall |  |  | 3.99x [3.97 – 4.01] |  | 69.6x [69.27 – 69.89] |

`few_selected`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 2.4 ns | 3.9 ns | 1.66x [1.64 – 1.68] | 183.6 ns | 77.4x [76.59 – 77.94] |
| 64 | 3.3 ns | 12.0 ns | 3.58x [3.57 – 3.58] | 262.3 ns | 78.3x [78.23 – 78.45] |
| 256 | 10.5 ns | 47.6 ns | 4.54x [4.53 – 4.57] | 336.1 ns | 32.0x [31.92 – 32.08] |
| 1,024 | 38.4 ns | 192.8 ns | 5.03x [5.02 – 5.03] | 608.4 ns | 15.9x [15.85 – 15.88] |
| 4,096 | 147.4 ns | 775.9 ns | 5.26x [5.26 – 5.27] | 1.7 us | 11.2x [11.23 – 11.26] |
| 16,384 | 785.7 ns | 3.2 us | 4.13x [4.13 – 4.13] | 5.1 us | 6.53x [6.52 – 6.54] |
| 65,536 | 2.6 us | 12.5 us | 4.89x [4.88 – 4.90] | 46.2 us | 18.0x [17.95 – 18.12] |
| 262,144 | 10.1 us | 50.3 us | 4.99x [4.99 – 4.99] | 215.6 us | 21.4x [21.34 – 21.41] |
| 1,048,576 | 41.1 us | 202.1 us | 4.92x [4.92 – 4.92] | 882.1 us | 21.5x [21.47 – 21.51] |
| 4,194,304 | 179.3 us | 805.7 us | 4.49x [4.46 – 4.53] | 3.50 ms | 19.5x [19.32 – 19.68] |
| 16,777,216 | 1.04 ms | 3.26 ms | 3.12x [3.11 – 3.15] | 14.29 ms | 13.7x [13.62 – 13.78] |
| overall |  |  | 4.06x [4.06 – 4.07] |  | 21.7x [21.66 – 21.72] |

All distributions: 4.03x [4.01 – 4.04] over naive C, 38.9x [38.76 – 38.95] over NumPy

With the output buffer 16 bytes behind its input (`half_selected`; "apart" is the same kernel with the output clear of the input):

`half_selected`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 146.6 ns | 153.5 ns | 1.05x [1.04 – 1.05] | 775.8 ns | 5.29x [5.29 – 5.29] |
| 16,384 | 656.2 ns | 610.9 ns | 0.93x [0.93 – 0.93] | 3.1 us | 4.72x [4.72 – 4.72] |
| 65,536 | 2.4 us | 2.5 us | 1.01x [1.01 – 1.01] | 12.4 us | 5.09x [5.09 – 5.09] |
| 262,144 | 10.1 us | 9.9 us | 0.99x [0.98 – 0.99] | 50.8 us | 5.05x [5.04 – 5.05] |
| 1,048,576 | 41.0 us | 46.4 us | 1.13x [1.13 – 1.13] | 200.7 us | 4.90x [4.90 – 4.90] |
| 4,194,304 | 178.9 us | 191.1 us | 1.07x [1.06 – 1.08] | 808.7 us | 4.52x [4.50 – 4.54] |
| 16,777,216 | 1.29 ms | 1.33 ms | 1.03x [1.03 – 1.04] | 3.26 ms | 2.53x [2.52 – 2.53] |
| overall |  |  | 1.03x [1.03 – 1.03] |  | 4.47x [4.47 – 4.48] |

No stall: the write position falls behind the read position as elements are dropped. The figures in the close table vary by up to 0.07x either way between runs.

What holds it at each size:

- **16:** the call and the scalar loop of the last 1 to 7 elements, as for `compress_f32`.
- **64 to 4,194,304:** the instruction count, 0.04 ns per element from 256 up (0.05 at 64). Unlike `compress_f32`, the output and the input are a quarter as many bytes per element, so the cache-line traffic of the output does not hold it before memory does.
- **16,777,216:** memory. 1.23 ms for 16 MB of data and 16 MB of mask with half selected, 2.7x over the naive C.

The candidates the design lists for this family (`docs/x86-64-design.md`), where they differ from `compress_f32`:

| Candidate | Outcome |
|---|---|
| 32 mask bytes to a 32-bit mask in one `vpmovmskb` | Kept (attempt 1) |
| A shuffle moves the selected elements to the front by index | Kept (attempt 1): `vpshufb` on eight bytes |
| The index from a table | Kept: 256 entries of 8 bytes, 2 KB |
| The index from `pext` with no table | Not tried as a shuffle. `pext` on eight data bytes in a general register with a byte mask is the whole compaction in one instruction, but it needs the mask bytes widened to `0xff` (a multiply, which assumes bytes of 0 or 1, or a `pdep`), and `pext`, the multiply and `popcnt` all need port 1: three cycles per group of eight, against about 1.5 for the shuffle |
| `popcnt` advances the write pointer | Kept |
| Sixteen elements per step | Not tried: a 16-byte step needs the control assembled from two entries (a load and a `vpblendd`) and the second half stored with `vmovhps`, the same 14 slots per 16 elements as two steps of eight. `compress_i16`'s attempt 2 measured the equivalent there: 0.98x |
| Masked loads and stores for leftover elements | Not tried, as for `compress_f32` |
| Alignment of data, prefetch | Not tried: both were slower on `compress_f32`, and this kernel is held by its instruction count up to memory |
| Instructions slow on other x86-64 processors | None used |
