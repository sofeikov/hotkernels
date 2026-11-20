# any_cols_bool attempts (arm64)

**Two specs.** Attempts 1 to 23 were judged against a spec of this kernel's own, written on an
ARM64 branch before the x86-64 one existed. It differed from the spec beside this file in four
ways: the matrix was declared `u8` with the oracle `(m != 0).any(axis=0)`, so the judge fed it every
byte value, where this spec declares `bool` and feeds 0 and 1 (other bytes are checked by `delivery
check`); the kernel returned `width`, where here it returns nothing; it timed widths 31, 3, 8, 48,
100 and 1,000, where this spec times 31, 8, 64 and 256; and it had one timing distribution, a mask
3% True, where this spec has sparse (3%) and dense (50%). The kernel of attempt 20 passed this
spec's `check` unchanged; attempt 24 adapts its header and return. From attempt 24 on everything is
judged against this spec.

All runs on an Apple M1 Pro (macOS), native, `--unlocked` (the spec is new on its branch), while
three other kernel runs were timing in other worktrees: load averages of 5 to 11 are noted with
each figure, and sizes from about 262,144 elements up are noisier than usual.

What the compiler does with `naive.c` (from the judge's figures): the inner loop over a row is
vectorised, but it loads and stores the result on every row, so each row waits for the store of
the row before it: rows of 31 at 0.22 ns per element, rows of 3 at 1.4, rows of 1,000 at 0.021
(where a row is long enough to hide that).

## 1. Baseline: accumulators in registers, a path per kind of width

- **Idea:** keep the column results in registers across rows and store once. Rows of 1 to 16:
  the whole rows as one stream of 16-byte loads that step by the largest multiple of the width
  under 16, so lane i of every load is column i mod width; eight accumulators; the last 16 bytes
  moved to the same lanes by `tbl`; the lanes folded onto the first `width` by `tbl` with shifts
  of width, 2 width, 4 width, 8 width; under 16 bytes of rows, two loads placed by `tbl`. Rows of
  17 to 32: first and last 16 bytes of each row, four rows a step. Rows of 33 to 64: first and
  last 32, two rows a step. Wider rows: tiles of 16 rows, blocks of 64 columns loaded from `out`,
  or'ed and stored back; `out` cleared first and passed through `umin` with 1 last. Every path
  turns its or'ed bytes into 0 or 1 by `umin` with 1, so any byte value is right.
- **check:** pass
- **compare:** none (first version). `run` (load average 10.8): 10.5x [10.44 – 10.52] over the
  naive C and 55.7x over NumPy over all six tables. Width 31 16.7x, 3 64.4x, 8 23.5x, 48 10.9x,
  100 4.24x, 1,000 1.13x.
- **Kept:** yes
- **Notes:** width 1,000 is slower than the C with no row or one row (0.93x to 0.97x at 16 to 256
  elements, 0.63x at 1,024: the clearing and normalising passes over `out` and a tile per row)
  and at 16,777,216 elements (0.82x, noisy); faster from 4,096 to 4,194,304.

## 2. Wide rows: no clearing and normalising passes over `out`

- **Idea:** the first tile starts its blocks from zero instead of loading `out`, and the last
  tile stores its blocks through `umin` with 1, so `out` is neither cleared first nor passed
  over again at the end.
- **check:** pass
- **compare:** significantly faster, 1.037x [1.035 – 1.039] (load average 6.4); width 1,000
  2.11x at 1,024 elements (one row, 65.6 ns to 31.0 ns), 1.66x at 4,096, 1.16x over its table;
  width 100 1.08x. The other widths unchanged.
- **Kept:** yes

## 3. No whole row: wider stores of the zeros

- **Idea:** a matrix with no whole row was cleared 16 bytes a store; now 32-byte pairs (`stp`),
  64 bytes a step for widths over 64, the last ones ending at the width.
- **check:** pass
- **compare:** significantly faster, 1.049x [1.048 – 1.050] (load average 8.5); width 1,000 at
  16 to 256 elements (no row) 1.93x to 1.95x, 22.4 ns to 11.6 ns.
- **Kept:** yes

## 4. Wide rows above 8 MB: non-temporal loads, in line

- **Idea:** width 1,000 was slower than the C at 16,777,216 elements (0.82x in attempt 1's run):
  the tiles read sixteen rows side by side, the C one row after another. From 8,388,608 bytes,
  more than the second-level cache, the row pairs of the wide path load with `ldnp`; the loop is
  chosen per block by a flag set at entry, and sits after the ordinary one.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.999 – 1.001] (load average 4.2); width
  1,000 1.11x at 16,777,216 elements, but 0.97x at 4,096.
- **Kept:** no

## 5. The same, with the non-temporal loop out of line

- **Idea:** attempt 4 with the `ldnp` loop moved after the return, so the ordinary loop and the
  code around it keep their place.
- **check:** pass
- **compare:** significantly faster, 1.002x [1.001 – 1.003] (load average 3.9); width 1,000
  1.17x [1.07 – 1.27] at 16,777,216 elements (441 µs to 370 µs), 0.99x at 4,096.
- **Kept:** yes

## 6. Rows of 33 to 48: three loads a row

- **Idea:** a row of 33 to 48 bytes is covered by its first 32 and its last 16; the path of 33
  to 64 loaded the last 32. Three loads a row in place of four, two rows a step.
- **check:** pass
- **compare:** significantly faster, 1.029x [1.029 – 1.029] (load average 5.0); width 48 1.35x
  from 4,096 to 65,536 elements, 1.18x from 262,144 to 4,194,304, 1.19x over its table.
- **Kept:** yes

## 7. Rows of 1 to 16: the load step from a table

- **Idea:** the step of the narrow path's loads, the largest multiple of the width up to 16,
  came from a `udiv` and a `mul`; now from a 17-byte table indexed by the width.
- **check:** pass
- **compare:** significantly faster, 1.002x [1.001 – 1.002] (load average 3.4); widths 3 and 8
  1.02x to 1.03x at 1,024 and 4,096 elements, the small sizes it was aimed at unchanged.
- **Kept:** yes

## 8. Wide rows: tiles of 32 rows

- **Idea:** tiles of 32 rows in place of 16 halve the loads and stores of `out` per row.
- **check:** pass
- **compare:** significantly slower, 0.999x [0.998 – 1.000] (load average 3.4); width 100 1.08x
  to 1.22x from 4,096 to 65,536 elements but 0.92x from 262,144 up; width 1,000 1.04x to 1.06x
  from 65,536 to 4,194,304 but 0.69x at 16,777,216.
- **Kept:** no
- **Notes:** beyond the first-level cache, more rows side by side cost more than the saved
  traffic to `out` gains.

## 9. Rows of 65 to 128: every column in registers

- **Idea:** attempt 8 showed the per-block work of the tiles weighs on width 100. A row of 65 to
  128 bytes is its first 64 and its last 64: eight accumulators hold every column, one row a
  step, one store of each at the end, no tiles.
- **check:** pass
- **compare:** significantly faster, 1.025x [1.024 – 1.027] (load average 3.4); width 100 1.22x
  to 1.65x up to 65,536 elements, 1.07x to 1.09x beyond, 1.17x over its table.
- **Kept:** yes

## 10. Rows of more than 128: blocks of 128 columns, one row a step

- **Idea:** blocks of 128 columns (eight accumulators, a cache line of the M1's 128 bytes) in
  place of 64 columns two rows a step: half the blocks per tile, and each row's loads of a block
  stay in one line when the row is aligned.
- **check:** pass
- **compare:** significantly faster, 1.015x [1.014 – 1.016] (load average 4.0); width 1,000
  1.53x at 1,024 elements (one row), 1.35x at 4,096, 1.07x to 1.09x at 16,384 and 65,536,
  0.96x to 0.97x from 262,144 to 4,194,304, 1.26x at 16,777,216 (noisy); 1.10x over its table.
- **Kept:** yes

## 11. Rows of 65 to 112: seven loads a row

- **Idea:** a row of up to 112 bytes is covered by its first 64 and its last 48: seven loads a
  row in place of eight.
- **check:** pass
- **compare:** significantly faster, 1.012x [1.010 – 1.013] (load average 5.3); width 100 1.08x
  to 1.14x at every size up to 4,194,304, 1.08x over its table; width 8 0.97x at 4,096 alone.
- **Kept:** yes

## 12. Rows of 1, 2, 4, 8 and 16: contiguous pairs of loads

- **Idea:** when the narrow path's step is 16 its loads are contiguous: four `ldp` a step with
  immediate offsets in place of eight `ldr` with register offsets, out of line.
- **check:** pass
- **compare:** significantly faster, 1.001x [1.001 – 1.002] (load average 4.2); width 8 1.06x at
  4,096 elements, 1.03x at 1,024, 1.01x over its table.
- **Kept:** yes

## 13. Rows of 17 to 32: a prefetch per step

- **Idea:** from 262,144 elements width 31 runs at about 80 GB/s, the second-level cache's pace;
  one `prfm pldl1keep` 1,024 bytes ahead per step of four rows, to start those fetches earlier.
- **check:** pass
- **compare:** significantly slower, 0.989x [0.987 – 0.990] (load average 4.4); width 31 0.86x
  to 0.92x from 4,096 elements up: the prefetch takes a load slot from a loop limited by loads,
  as it did for `any_rows_bool`.
- **Kept:** no

## 14. No division for widths of 17 to 128

- **Idea:** the paths of 17 to 128 count bytes left down by the width instead of rows, so the
  `udiv` at entry is taken only by the narrow and the wide paths; "no whole row" is `n < width`.
- **check:** pass
- **compare:** significantly slower, 0.997x [0.996 – 0.998] (load average 3.8); widths 31, 48
  and 100 unchanged (the division's latency was already hidden), width 3 0.94x to 0.96x up to
  256 elements behind the extra compare at entry.
- **Kept:** no

## 15. Wide rows: tiles of 8 rows

- **Idea:** with blocks of 128 columns, fewer rows side by side may suit the caches beyond the
  first level better (attempt 8 found more rows worse there).
- **check:** pass
- **compare:** no significant difference, 1.000x [1.000 – 1.001] (load average 4.7); width
  1,000 1.24x at 16,777,216 elements, 0.94x to 0.97x from 16,384 to 4,194,304.
- **Kept:** no

## 16. Wide rows: tiles of 8 rows above 8 MB only

- **Idea:** attempt 15's gain without its loss: half a tile when the matrix is larger than the
  second-level cache, the size from which the loads are already non-temporal.
- **check:** pass
- **compare:** significantly faster, 1.004x [1.002 – 1.005] (load average 5.2); width 1,000
  1.26x [1.18 – 1.33] at 16,777,216 elements (376 µs to 292 µs), nothing else moved.
- **Kept:** yes

## 17. Widths dividing 16 above 8 MB: non-temporal loads

- **Idea:** the contiguous loop of attempt 12 with `ldnp` from HUGE_N, as attempt 5 did for the
  wide rows.
- **check:** pass
- **compare:** significantly faster, 1.001x [1.001 – 1.002] (load average 4.6); width 8 1.13x
  [1.10 – 1.16] at 16,777,216 elements, 0.98x to 0.99x at 256 and 1,024.
- **Kept:** yes

## 18. Rows of 33 to 48 above 8 MB: non-temporal pairs

- **Idea:** the pair of each row's first 32 bytes by `ldnp` from HUGE_N, out of line; the last
  16 keep `ldr` (register offset, which `ldnp` does not have).
- **check:** pass
- **compare:** significantly faster, 1.001x [1.001 – 1.002] (load average 3.8); width 48 1.17x
  [1.14 – 1.20] at 16,777,216 elements, but 0.96x to 0.97x at 256 and 1,024 behind the size test.
- **Kept:** yes

## 19. Rows of 65 to 112 above 8 MB: non-temporal pairs

- **Idea:** attempt 18 for the path of attempt 11: its three pairs by `ldnp` from HUGE_N, out of
  line.
- **check:** pass
- **compare:** significantly faster, 1.001x [1.000 – 1.002] (load average 3.1); width 100 1.11x
  [1.07 – 1.14] at 16,777,216 elements, 0.97x at 256.
- **Kept:** yes

## 20. Rows of 1 to 16: the lane numbers loaded once

- **Idea:** the store of up to 16 results loaded the lane-number constant again although the
  narrow path holds it in `v31`; the narrow path now enters the store after that load.
- **check:** pass
- **compare:** significantly faster, 1.001x [1.000 – 1.002] (load average 3.2); widths 3 and 8
  1.01x to 1.02x up to 4,096 elements.
- **Kept:** yes

## 21. Rows of 1 to 16: the fold unrolled, four steps always

- **Idea:** the fold of the lanes onto the first `width` was a loop of one to four steps
  (shifts of width, 2, 4 and 8 width); now all four indices are made at once and all four steps
  taken, an index past 15 giving zeros.
- **check:** pass
- **compare:** significantly slower, 0.997x [0.995 – 0.999] (load average 5.2); width 3 1.20x
  and 1.12x at 16 and 64 elements, but width 8, which needs one step, 0.83x to 0.93x up to 1,024.
- **Kept:** no

## 22. The same, leaving as soon as the lanes are covered

- **Idea:** attempt 21's indices made at once, and a test of the width after each step, so
  width 8 and up take one step.
- **check:** pass
- **compare:** no significant difference, 1.000x [0.998 – 1.001] (load average 7.0); width 3
  1.15x at 16 elements, width 8 0.91x to 0.96x up to 1,024.
- **Kept:** no
- **Notes:** the loop's branch was predicted well in a timing run of one width; the gain of
  either attempt at one width is the loss at the other.

## 23. Rows of 17 to 32: eight rows a step

- **Idea:** the step of four rows is 19 instructions for eight loads, close to what the front
  end issues in the time of the loads; eight rows a step (sixteen loads from two positions) halve
  the adds and branches per row.
- **check:** pass
- **compare:** significantly slower, 0.999x [0.998 – 0.999] (load average 5.3); width 31 1.02x at
  16,384 elements, 0.90x at 64 (two rows, one more test).
- **Kept:** no

## Where it stopped

Twenty-three attempts, the last three without a gain. In the first-level cache every path is at
the M1's three 16-byte loads a cycle, or near it: width 31 at 0.72 cycles a row (two loads),
widths 3 and 8 at about 43 bytes a cycle. Beyond it the kernel runs at the pace of the
second-level cache, about 80 GB/s, whatever the width; a prefetch (attempt 13) took load slots
and lost. Left untried: non-temporal loads above 8 MB for rows of 17 to 32 and of 1 to 16 other
than divisors of 16, whose loads take a register offset that `ldnp` does not have (rows of 17 to
32 would need the masked pair `any_rows_bool` uses); the kernel is already 12x to 70x the C
there.

## 24. The kernel of attempt 20 under this spec

- **Idea:** none beyond the spec: the header says `void any_cols_bool(const bool *m, bool *out,
  size_t n, size_t cols)`, and the return no longer sets `x0`. Every path, result and store is
  attempt 20's.
- **check:** pass, 40,357 cases. Outside the judge, the kernel against a plain loop on matrices of
  arbitrary bytes (any byte, rare `0x80`, rare 2 to 255, all zero; 33 widths from 1 to 4,097, 14
  sizes from 0 to 300,000, a guard byte after `out`): 1,848 cases pass.
- **compare:** none (first ARM64 version under this spec). `run` (load average 2.3 to 4.2): 8.34x
  [8.34 – 8.34] over the naive C and 70.7x over NumPy over all eight tables. Width 31 16.9x, 8
  26.2x, 64 4.50x, 256 2.44x; dense and sparse the same to 1%.
- **Kept:** yes
- **Notes:** width 256 takes the wide path's tiles of 16 rows and blocks of 128 columns: 1.85x
  over the C from 262,144 elements, 2.05x to 3.6x below; width 64 4.6x to 7.9x. No size is
  slower than the C or NumPy. Untimed widths against the C (scratch timer, 3% True, load average
  3.3) at 930 / 4,096 / 1,048,576 elements: 1 128x / 302x / 186x, 3 111x / 175x / 111x, 17 18.9x /
  23.1x / 16.9x, 33 16.2x / 20.3x / 14.2x, 48 18.6x / 24.9x / 15.4x, 65 4.45x / 4.79x / 3.28x, 100
  7.94x / 9.73x / 6.27x, 129 2.46x / 2.83x / 1.96x, 200 2.60x / 3.23x / 2.02x, 1,000 2.35x / 2.30x /
  1.34x.

## 25. Rows of 129 to 256: every column in registers

- **Idea:** width 256 is this spec's weakest table (2.44x over the C). A row of 129 to 256 bytes
  is its first 128 and its last 128: sixteen accumulators (`v16` to `v31`) hold every column, one
  row a step, sixteen loads, and one store of each at the end; no tiles and no loads and stores of
  `out` per block. Out of line, after the other paths.
- **check:** pass; the arbitrary-bytes check outside the judge passes (1,848 cases).
- **compare:** significantly faster, 1.025x [1.024 – 1.026] (load average 2.9); width 256 1.08x to
  1.24x up to 4,194,304 elements, 1.10x over each of its two tables, 0.98x at 16,777,216 (no
  `ldnp` on this path).
- **Kept:** yes

## 26. Rows of 129 to 256 above 8 MB: non-temporal loads

- **Idea:** the loop of attempt 25 with `ldnp` from HUGE_N, out of line, as the other paths do.
- **check:** pass; arbitrary-bytes check passes.
- **compare:** significantly faster, 1.001x [1.001 – 1.001] (load average 2.4); width 256 1.12x at
  16,777,216 elements on the sparse table, 0.96x to 0.98x at 256 and 1,024 behind the size test.
- **Kept:** yes

## 27. Rows of 1 to 16, fewer than eight loads in all: one accumulator

- **Idea:** width 8 at 16 elements (two rows) was 1.83x over the C, the smallest gain of any
  table: the narrow path cleared eight accumulators and set up seven offsets for a matrix of one
  or two loads. Now the test for a step of eight loads comes first, and a matrix that has none
  takes a loop of one accumulator, out of line, then the last 16 bytes and the fold as before.
- **check:** pass; arbitrary-bytes check passes.
- **compare:** significantly faster, 1.007x [1.006 – 1.007] (load average 1.9); width 8 1.26x at
  16 elements (3.7 ns to 2.9 ns), 1.07x at 64, 0.99x at 1,024; 1.03x over each of its tables.
- **Kept:** yes

## 28. One row wider than 16: a copy through `umin`

- **Idea:** with one whole row the results are that row's bytes, each through `umin` with 1. A
  test for one row after the narrow paths sends it to a copy, 32 bytes a step, the last 32 ending
  at the row's end; rows of 17 to 32 as two overlapping halves. No accumulators to clear.
- **check:** pass; arbitrary-bytes check passes.
- **compare:** significantly faster, 1.007x [1.007 – 1.007] (load average 1.8); width 64 1.29x at
  64 elements (2.8 ns to 2.2 ns), width 256 1.14x at 256; 0.98x to 0.99x on widths 31 and 64 at
  256 to 1,024 elements behind the extra test.
- **Kept:** yes

## 29. Rows of 49 to 64: two positions, the last 32 bytes by register offsets

- **Idea:** the loop of two rows moved four positions (each row's start and its last 32 bytes).
  Now two positions, the row's first 32 bytes by `ldp` and its last 32 by two `ldr` with register
  offsets: the same six loads per row pair in load count, two adds instead of four.
- **check:** pass; arbitrary-bytes check passes.
- **compare:** significantly faster, 1.007x [1.005 – 1.008] (load average 1.5); width 64 1.04x to
  1.08x from 256 to 65,536 elements on both tables, 1.03x over each.
- **Kept:** yes
- **Notes:** width 64 now takes 0.43 ns a row at 4,096 elements, about 1.37 cycles, against the
  1.33 of four loads a row on three load units. This path has no `ldnp` loop above 8 MB.

## 30. Rows of 49 to 64 above 8 MB: non-temporal pairs

- **Idea:** the loop of attempt 29 with each row's first 32 bytes by `ldnp` from HUGE_N, out of
  line; the last 32 keep `ldr` with register offsets.
- **check:** pass; arbitrary-bytes check passes.
- **compare:** significantly faster, 1.005x [1.005 – 1.005] (load average 3.0); width 64 1.14x at
  16,777,216 elements on both tables, 1.17x at 64 elements (one row, Lrow1: the code moved), 0.96x
  to 0.98x at 256 and 1,024 behind the size test.
- **Kept:** yes
