# nonzero_2d_bool attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The spec is not on `main` yet, so every judge command ran with `--unlocked`; `spec.py` and `naive.c` are as on `claude/gaps160-06-nonzero-2d-bool` and were not changed.

What the compiler does with `naive.c`: the scalar loop, two 8-byte stores per element, about 0.44 ns per element at every width and share from 1,024 elements up.

## 1. Baseline: blocks of 32, a sparse path with a multiplication, table groups for dense blocks, narrow rows in whole-row blocks

- **Idea:** 32 mask bytes become a word of 32 bits. A block with at most four true elements takes four positions with `tzcnt` and `blsr`; each position f gets its row as the high half of 2f * ceil(2^63 / width) (`mulx`) and its column as f - row * width, stored with two 8-byte stores. A denser block goes eight elements at a time: the group's byte indexes a table of positions, `vpbroadcastq` puts them in every quadword and four `vpshufb` spread them into four registers of two (0, position) pairs, to which (row, column) of the group's first element is added; four 32-byte stores. A group that runs past a row's end adds (1, -width) to the pairs past it, chosen by a byte compare of the positions with the distance to the row's end (out of line, one branch per group). Rows of 1 to 8 elements go in blocks of whole rows (30 bytes for width 3), so each position's row within the block and column are fixed for the width; they come from tables of 128 bytes per width, looked up with one `vpshufb` per group. Under 32 elements, the last 0 to 31 and matrices of 2^31 elements or more run one element at a time.
- **check:** pass
- **compare:** not run (first version). `run`: 2.91x over naive C overall (load average 0.7). At 1,024 elements: sparse 5.35x (width 22), 6.65x (3), 5.42x (32), 5.59x (1,000); dense 2.46x, 3.07x, 3.36x, 3.41x. At 1,048,576: sparse 5.2x to 6.6x, dense 1.6x to 1.8x. At 16 elements, below one block, 0.72x: the one-element loop is slower than the compiler's.
- **Kept:** yes
