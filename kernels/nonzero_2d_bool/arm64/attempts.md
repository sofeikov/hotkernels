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
