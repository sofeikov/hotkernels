# nonzero_2d_bool attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The spec is not on `main` yet, so every judge command ran with `--unlocked`; `spec.py` and `naive.c` are as on `claude/gaps160-06-nonzero-2d-bool` and were not changed.

What the compiler does with `naive.c`: the scalar loop, two 8-byte stores per element, about 0.44 ns per element at every width and share from 1,024 elements up.

## 1. Baseline: blocks of 32, a sparse path with a multiplication, table groups for dense blocks, narrow rows in whole-row blocks

- **Idea:** 32 mask bytes become a word of 32 bits. A block with at most four true elements takes four positions with `tzcnt` and `blsr`; each position f gets its row as the high half of 2f * ceil(2^63 / width) (`mulx`) and its column as f - row * width, stored with two 8-byte stores. A denser block goes eight elements at a time: the group's byte indexes a table of positions, `vpbroadcastq` puts them in every quadword and four `vpshufb` spread them into four registers of two (0, position) pairs, to which (row, column) of the group's first element is added; four 32-byte stores. A group that runs past a row's end adds (1, -width) to the pairs past it, chosen by a byte compare of the positions with the distance to the row's end (out of line, one branch per group). Rows of 1 to 8 elements go in blocks of whole rows (30 bytes for width 3), so each position's row within the block and column are fixed for the width; they come from tables of 128 bytes per width, looked up with one `vpshufb` per group. Under 32 elements, the last 0 to 31 and matrices of 2^31 elements or more run one element at a time.
- **check:** pass
- **compare:** not run (first version). `run`: 2.91x over naive C overall (load average 0.7). At 1,024 elements: sparse 5.35x (width 22), 6.65x (3), 5.42x (32), 5.59x (1,000); dense 2.46x, 3.07x, 3.36x, 3.41x. At 1,048,576: sparse 5.2x to 6.6x, dense 1.6x to 1.8x. At 16 elements, below one block, 0.72x: the one-element loop is slower than the compiler's.
- **Kept:** yes

## 2. Sparse positions without the multiplication: the block's (row, column) carried from block to block

- **Idea:** a knock-out of `mulx` and `imul` from the sparse path took width 22 at 5% from 0.081 to 0.045 ns per element (micro-benchmark, 4,096 elements): `tzcnt`, `mulx` and `imul` all run on port 1, three per position. Instead the row and column of each block's first element are carried: the column advances by 32 mod width and the row by 32 / width, plus one where the column passes the width (one `div` on entry for widths up to 32, a compare, `cmov` and `sbb` per block). Up to 32 elements a row, a sparse position p is at c + p < 64 of the block's row, and a 128-byte table for the width gives its row within the block and its column. Wider rows: a block inside one row stores (r, c + p) as they are; a block across one row end compares c + p with the width. Dense blocks start from the carried (row, column) as well. With no multiplication there is no limit on n any more.
- **check:** pass
- **compare:** significantly faster, 1.135x [1.133 – 1.138]
- **Kept:** yes
- **Notes:** micro-benchmark at 4,096: width 22 at 5% 0.081 to 0.055 ns per element, dense widths 32 and 1,000 0.140 to 0.128; width 3 unchanged (it had no multiplication).

## 3. Fewer instructions per dense group

- **Idea:** knock-outs of the dense group (half the stores, stores to one aligned place, no `vpshufb`, no write-position advance, no table load) each gained only 0 to 15%: the group is held by its instruction count, about 100 per block of 32, not by one port or by the stores. So: a dense block that lies in one row (c + 32 <= width) skips the per-group row-end compare and branch; the group's byte comes from `al` or `ah`, one shift per two groups; the write position advances by one `lea r9, [r9 + rcx * 2]` (two cycles on the chain, still under the group's time).
- **check:** pass
- **compare:** significantly faster, 1.025x [1.024 – 1.027]
- **Kept:** yes
- **Notes:** dense widths 32 and 1,000 1.04x, width 22 1.03x, width 3 no significant difference.

## 4. The one-element loop unrolled by two, the write position by one `lea`

- **Idea:** under 32 elements, and for the last 0 to 31, the kernel runs one element at a time. Micro-benchmark at 16 and 31 elements, width 22: 0.58 ns per element against the compiler's 0.46 (unrolled by four, branchless). Two elements per pass, the column reset left as a branch (it predicts: the pattern repeats every row), the write position advanced by `lea r9, [r9 + rax * 2]`: 0.43 to 0.47 ns per element.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.001 – 1.012]
- **Kept:** yes
- **Notes:** the only timed size it reaches is 16 elements: width 3 1.22x and 1.25x there, the other widths no significant difference.

## 5. One table path for every width up to 32: a window of (row, column) bytes per dense group

- **Idea:** with the block's column c carried, every position of a block is at c + p < 64. A table of 128 bytes per width (4 KB for widths 1 to 32) holds the row and the column of each c + p as two bytes. A dense group loads the 16 bytes of the table at its first column, the pairs of its eight positions, and one `vpshufb` with a table of byte indices (2o, 2o + 1 per position o present) picks those of its true elements; four `vpshufb` spread them into the four registers and the block's row is added. Sparse blocks read the same table. This replaces both the row-end path of widths 9 to 32, which width 22 took in a third of its groups, and the narrow blocks of whole rows, which left up to 4 of 32 bytes unused (30 for width 3).
- **check:** pass
- **compare:** significantly faster, 1.042x [1.036 – 1.049]
- **Kept:** yes
- **Notes:** sparse width 3 1.17x, dense width 22 1.07x, dense width 3 1.06x, sparse width 22 1.03x; widths 32 and 1,000 no significant difference (they do not take the new path, or take it as before). Micro-benchmark at 4,096, dense: width 22 0.158 to 0.122 ns per element, width 3 0.144 to 0.114.
