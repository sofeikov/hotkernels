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
