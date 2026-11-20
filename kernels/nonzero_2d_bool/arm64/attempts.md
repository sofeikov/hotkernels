# nonzero_2d_bool attempts (arm64)

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine (load averages of 5 to 10), so intervals are wider than usual and figures are indicative.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every element's (row, column) pair with one `stp`, advance the write pointer by 16 bytes times the mask byte, step the column and wrap it with `csel` / `cinc`.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** `judge run`: 1.14x the naive C at every width and density, about 1 ns per element.

## 2. Chunks of up to 8 bytes that stop at the end of a row, a position table and `st2`

- **Idea:** `nonzero_bool`'s dense step, made two-dimensional by never letting a chunk cross a row: each chunk is 8 bytes or what is left of the row, loaded as one word with the bytes past its end masked off. Two multiplications give an 8-bit mask and the true count; the mask indexes the 256-entry table of positions; the positions are widened to 64 bits and added to the chunk's first column, and four `st2` store them interleaved with copies of the row: eight (row, column) pairs, of which the write pointer keeps `count`. The column and row advance with `csel` / `cinc`. The last 0 to 7 bytes are scalar.
- **check:** pass
- **compare:** significantly faster, 1.57x [1.50 – 1.64]
- **Kept:** yes
- **Notes:** width 32 and 1,000 about 2.6x (0.38 ns per element), width 3 0.95x: a chunk is 3 bytes there and costs more than three scalar steps. Even at width 32 the loop runs at about 10 cycles per chunk, which is the chain through the column: length, add, compare, select, every chunk.

## 3. A loop over rows, with an inner loop of whole 8-byte chunks

- **Idea:** take the chain through the column out of the chunk loop. An outer loop per row sets the row registers and the row's end; an inner loop runs whole 8-byte chunks with no masking and only pointer increments and a vector add for the column; one masked chunk takes what is left of the row.
- **check:** pass
- **compare:** significantly faster, 1.18x [1.18 – 1.19]
- **Kept:** yes
- **Notes:** default width 1.52x sparse and 1.22x dense, width 1,000 1.34x dense, width 3 0.96x (one masked chunk per row of 3, plus the row's bookkeeping). The sparse tables at widths 32 and 1,000 read about 1.9 ms at 1M elements for both versions, against 0.3 ms at the default width; noise from the other agents' runs, to be rechecked.

## 4. `zip1` / `zip2` and `stp` in place of `st2`

- **Idea:** the four `st2` of 32 bytes each look like the limit (about 7 cycles per chunk at width 32). Pair the row with each column register by `zip1` / `zip2` instead and store the eight pairs with four `stp q`. One row register instead of four copies.
- **check:** pass
- **compare:** significantly faster, 1.13x [1.12 – 1.14]
- **Kept:** yes
- **Notes:** about 1.25x at the default width and at width 3 dense, 1.02 to 1.10x at widths 32 and 1,000 above 64K elements. The slow sparse tables of attempt 3 at widths 3, 32 and 1,000 appeared again for both versions (2 ms at 1M at width 32, against 0.29 ms dense); a `judge run --quick` straight after had sparse width 32 at 0.29 ns per element, and a timer outside the judge 0.30. So it is the machine (the process sharing cores with other agents' runs), not the kernel.

## 5. Sparse path: 64 bytes to a bitmask, six extractions, row by a multiplication

- **Idea:** `nonzero_bool`'s sparse step, which does not care about rows. 64 mask bytes become a 64-bit mask (`ushl`, three rounds of `addp`, `rbit`) and a count. With at most 6 true, six unconditional `clz` / `lsr` / `bic` steps take the positions; each position f becomes its row without a division, row = high half of 2f times M = ceil(2^63 / w) (`umulh`), exact while f * w < 2^63, and its column f - row * w (`msub`); one `stp` per pair. More than 6 true sends the block to the dense row loop of attempt 4, which now starts anywhere in a row and runs to the end of the row past the block. After a dense stretch, at most one true byte in eight sends the next bytes back to the sparse path, otherwise a stretch of 512 bytes more. The sparse path is skipped when n >= 2^31 (the product could overflow) and the width is taken as at most n.
- **check:** pass
- **compare:** significantly faster, 1.81x [1.76 – 1.87]
- **Kept:** yes
- **Notes:** sparse 2.3x at the default width, 6.8x at width 3, 3.4x at width 32, 2.6x at width 1,000 (about 0.1 ns per element); dense 0.94 to 0.95x at widths 22, 32 and 1,000, all of it at small sizes (n=16 to 1,024: the failed first sparse block), and 1.02x at width 3.

## 6. Narrow rows: chunks of whole rows with row and column tables

- **Idea:** for widths under 8 the row loop handles one row per masked chunk. Instead, take as many whole rows as fit in 8 bytes (8 at width 1, 6 at width 3, 5 at width 5) as one chunk, which always starts a row. The positions from the table index two 8-byte tables built for the width (`tbl`): the row within the chunk and the column. `zip1` interleaves them byte by byte, three rounds of widening give 64-bit (row, column) pairs with the chunk's first row added by the last `uaddw`, and four `stp` store eight pairs. The row loop hands over to this loop at the start of a row, and takes back the last bytes of the mask.
- **check:** pass
- **compare:** significantly faster, 1.13x [1.13 – 1.13]
- **Kept:** yes
- **Notes:** width 3 dense 2.05x (0.38 ns per element, from 0.80), sparse 1.33x, n=16 1.38x; the other widths unchanged but for 0.97x at n=16 (one more compare and branch per row).

## 7. A second sparse tier: 12 extractions for 7 to 12 true bytes

- **Idea:** as `nonzero_bool` attempt 5. At 5% true about one 64-byte block in 25 holds more than 6, and each such block went to the dense code for at least 64 bytes and the rest of the row, at about 0.24 ns per element against 0.1. A second copy of the extraction takes 12 positions; only blocks with more than 12 go dense.
- **check:** pass
- **compare:** significantly faster, 1.11x [1.10 – 1.11]
- **Kept:** yes
- **Notes:** sparse 1.11x to 1.41x (1.41x at width 1,000, where a dense trip ran to the end of a 1,000-byte row); dense unchanged. Sparse is now about 0.08 ns per element at every width.

## 8. Two dense chunks per step in the wide row loop

- **Idea:** load 16 mask bytes with one `ldp` and handle two 8-byte chunks per iteration of the inner loop, with one more single chunk when 8 to 15 bytes of the row are left: one loop branch and compare per 16 bytes.
- **check:** pass
- **compare:** significantly faster, 1.006x [1.004 – 1.008]
- **Kept:** yes
- **Notes:** dense widths 32 and 1,000 1.03 to 1.04x (0.27 ns per element at 1M), the rest unchanged.

## 9. Narrow chunks of 16 bytes

- **Idea:** a narrow chunk of attempt 6 holds as many whole rows as fit in 8 bytes, which leaves bytes unused (6 of 8 at width 3, 5 at width 5). Take as many as fit in 16 bytes instead (15 at width 3 and 5, 14 at width 7, 16 at widths 1, 2 and 4), as two words handled by the same steps, the second word's positions moved up by 8 and looked up in row and column tables of 16 entries.
- **check:** pass
- **compare:** significantly faster, 1.03x [1.03 – 1.03]
- **Kept:** yes
- **Notes:** width 3 dense 1.26x (0.31 ns per element at 1M), n=16 1.32x at width 3; nothing else moved.

## 10. Wide rows: a sparse block inside one row skips the multiplication

- **Idea:** in rows of 512 bytes or more, most 64-byte blocks lie in one row. For a block of at most 6 true bytes there, take its row and first column once (one `umulh`, one `msub`) and give each extracted position that row and the first column plus its offset: five instructions per position instead of eight. A block that crosses a row takes the general steps; with rows of at least 512, at most one block in eight does.
- **check:** pass
- **compare:** significantly faster, 1.025x [1.024 – 1.026]
- **Kept:** yes
- **Notes:** sparse width 1,000 1.26x (0.061 ns per element at 1M); the other sparse tables 0.98 to 0.99x at the large sizes, the cost of the width test on every block.

## 11. The width test out of the sparse loop

- **Idea:** attempt 10 tests the width on every sparse block. Give rows of 512 or more a sparse loop of their own, chosen when the sparse path is entered and after a block of more than 6, so the loop for narrower rows is attempt 9's again.
- **check:** pass
- **compare:** significantly faster, 1.008x [1.006 – 1.010]
- **Kept:** yes
- **Notes:** sparse 1.01 to 1.02x at every width, 1.08x at width 1,000 at 1M; dense unchanged.

## 12. Dense: a 32 KB table of ready-made pairs

- **Idea:** replace the widening and the `zip`s of a dense chunk (15 vector instructions) by a table of 256 entries of eight (0, position) pairs each, loaded with four `ldp q` and added to the chunk's (row, column) with eight `add`. The table base needs a register, so the start of `out` moves to the stack.
- **check:** pass
- **compare:** significantly slower, 0.92x [0.92 – 0.92]
- **Kept:** no
- **Notes:** dense wide tables 0.79x: the four 32-byte loads from a 32 KB table per chunk cost more than the arithmetic they replace; n=16 0.84x from the stack traffic.
