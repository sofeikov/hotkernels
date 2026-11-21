# any_cols_bool attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written.

What the compiler does with `naive.c` (`clang -O3 -march=native`): it or's each row into the results in memory, 32 bytes at a time where the row is that long, and makes the results 0 or 1 in a last pass. With rows of 31 it takes 0.08 ns per element from 1,024 to 1,048,576 elements (2.5 ns per row); with rows of 8, 0.17 ns per element; with rows of 64, 0.03 ns; with rows of 256, 0.012 ns. NumPy takes 700 ns for a call of 16 elements and 0.33 ns per element for rows of 31 at 1,048,576.

The timed sizes are element counts. With the default width of 31, 16 elements are no row, 64 are two rows and 4,096 are 132 rows; with a width of 256, both 16 and 64 elements are no row and 256 are one.

The kernel does the same work whatever the elements are, so the two timing distributions differ only by where their buffers were placed.

## 1. Baseline: the results in registers for the whole call

- **Idea:** the results are or'ed together in `ymm` registers and made 0 or 1 once, by `vpminub` with 1, when they are stored. Rows of 32 and more: strips of 32 columns, each with a register for a whole pass over the rows; eight strips in a pass (`vpor` with a memory operand, eight per row) while 256 columns are left, then four, two and one, and the columns left under 32 by one more strip that ends at the row's end and overlaps the one before. A pass of one strip takes two rows per step into two registers. Rows under 32: one register for the whole call. A 32-byte load at a row's start holds as many whole rows as fit in 32 bytes (`step` bytes, from a table of 32 entries), and the loads are `step` apart, two per pass of the loop into two registers; what a load holds above `step` belongs to rows with a load of their own and is never looked at. The rows that start in the last 31 bytes of `m` are taken from one load of the last 32 bytes, moved down to where a load at the first of them would have put them, with zeros above. Then the rows inside the register are folded onto the first, one move and `vpor` per row, and the first `cols` bytes are stored as their first 16, 8 or 4 and their last 16, 8 or 4, which overlap. Moving a register's bytes down by a count is two `vpshufb` with index vectors read from a table at the count, one for the bytes that stay in their half and one for those that cross. A matrix under 32 bytes is copied to zeros below the stack pointer by two moves that overlap and loaded from there, which costs a failed store forward. One division, as the reference has, for the row count.
- **check:** pass
- **compare:** n/a (first version). `run`: 5.37x [5.35 – 5.39] over naive C over all eight tables, 66.2x over NumPy. Rows of 31: 8.28x (sparse) and 8.26x (dense); rows of 8: 11.7x and 11.3x; rows of 64: 4.58x and 4.46x; rows of 256: 2.03x and 1.86x.
- **Kept:** yes
- **Notes:** rows of 31 at 4,096 elements: 26.0 ns for 132 rows, 0.18 ns per row, about one cycle, where two loads per cycle would allow half that; half the loads straddle two cache lines. Rows of 64 at 16,384: 0.33 ns per row for two loads. Rows of 256 at 16,384: 0.89 ns per row for eight loads. Rows of 8 at 16 elements, the copy through the stack: 7.1 ns against 6.0 for the reference, the one figure below it (0.85x). Rows of 256 at 16 and 64 elements, no row: 4.0 ns against 4.2. The table with the output 16 bytes behind the input is level with the other: 1.00x from 4,096 to 4,194,304.

## 2. No division before the loops, on either path

- **Idea:** count the rows by address instead of dividing. Rows of 32 and more: a pass ends at the last address a whole row starts at, `m + n - cols`, so the row count is never needed. Rows under 32: the loads end 32 bytes before the end of `m` rather than of its whole rows, the division still runs but only its remainder is wanted, to clear the partial row out of the last 32 bytes (`vpand` with a mask read from a table at the remainder), and a matrix with no row went through the copy on the stack.
- **check:** pass
- **compare:** significantly slower, 0.987x [0.979 – 0.994]
- **Kept:** no
- **Notes:** the wide tables gained (rows of 64: 1.04x and 1.06x, 1.41x at 16 elements where there is no row and now no division). The default width lost, 0.89x: at 16 elements, no row, 6.4 ns against 1.9, because the path for a matrix under 32 bytes was taken where zeros were stored directly before; and at 256 elements 4.0 ns against 3.4. The division was not what the loop waited for in the first place: its branch is predicted, so the loads ran beside the division already, and the last load waits for the quotient in one form and for the remainder, then for the mask, in the other.

## 3. No division on the wide path only

- **Idea:** the wide half of attempt 2 alone. The narrow path as in attempt 1, except that a matrix shorter than one row stores its zeros before the division.
- **check:** pass
- **compare:** significantly faster, 1.024x [1.011 – 1.037]
- **Kept:** yes
- **Notes:** rows of 31: 1.04x, all of it at 16 elements. Rows of 64: 1.03x. Rows of 8: 1.02x. Rows of 256: 1.01x, not significant.

## 4. A matrix under 32 bytes built in a register

- **Idea:** no copy through the stack. The matrix's first 16, 8 or 4 bytes and its last 16, 8 or 4 are loaded, the last moved down by one `vpshufb` to follow the first with zeros above, and the two joined (`vinserti128`, `vpunpcklqdq`, `vpunpckldq`); one to three bytes go through a general register. The load of 32 bytes that two smaller stores had just written could not be forwarded and waited for them to reach the cache.
- **check:** pass
- **compare:** significantly faster, 1.022x [1.003 – 1.040]
- **Kept:** yes
- **Notes:** the judge times this path at one point, 16 elements in rows of 8: 4.0 ns against 7.1, 1.82x and 1.76x in the two tables, and 3.3 ns for the reference became 0.67x of the kernel's time where it was 0.85x the other way. Every other point is level. A single row under 32 wide, `(1, 31)`, takes the same path; the judge has no timed point there.

## 5. Four loads per step in the narrow loop

- **Idea:** four `vpor` with a memory operand into four registers per pass of the narrow loop, the loads a `step` apart (`rdx`, `rdx + step`, `rdx + 2 * step`, `rdx + 3 * step`, the multiples kept in registers), and the position moved by one `lea` with a scaled index, whose two cycles are what four loads take anyway. The loads left under four are taken one at a time.
- **check:** pass
- **compare:** significantly faster, 1.028x [1.019 – 1.037]
- **Kept:** yes
- **Notes:** rows of 8: 1.08x in both tables (1.38x at 256 elements, 1.32x at 1,024, level from 16,384). Rows of 31: 1.00x, not significant: 4,096 elements take 24.8 ns, 0.19 ns per row, one cycle, and that is the cost of the loads. A 32-byte load every 31 bytes straddles two cache lines 31 times in 64, and such a load takes a cycle more than the half cycle of one that does not. In a benchmark outside the judge, with the matrix on a 64-byte boundary, 4,096 elements in rows of 8, whose loads then never straddle, went from 26.8 ns to 16.5 ns; three bytes off the boundary, from 31.2 ns to 26.9 ns.

## 6. Two rows per step in the pass of two strips

- **Idea:** the pass that serves rows of 64 to 127 takes one row, two loads, per step, and a step is about 1.3 cycles. Two rows per step into four registers, as the pass of one strip does, with a row without a pair taken after the loop.
- **check:** pass
- **compare:** significantly faster, 1.012x [1.009 – 1.016], and 1.014x [1.009 – 1.018] when run again
- **Kept:** no
- **Notes:** dropped although the judge rated it faster, because the gain is not where the change is. Rows of 64, the only timed width that takes this pass: 1.01x not significant and 1.02x. Rows of 256: 1.04x, all of it at 16 and 64 elements, where there is no row and the kernel stores zeros: 3.2 ns against 4.2. That code was not changed; it moved, and in the kernel as committed the `jmp` after its loop straddles a 32-byte boundary of the code, which it no longer did with this change in front of it. In a benchmark outside the judge the pass itself was slower for a matrix of few rows (one row of 64: 1.70 ns against 1.51; four rows: 2.26 ns against 1.89), level from 16 rows up with the matrix five bytes off a 64-byte boundary, and 1.15x faster from 64 rows with the matrix on one. The binding leaves a matrix of no rows to NumPy, so the point that gained is never reached from Python.

## 7. The function and its loops on 32-byte boundaries

- **Idea:** take the gain of attempt 6 on purpose: `.p2align 5` for the function, for the head of each loop and for the path of no rows, so that where a loop and the jump after it lie within a 32-byte block of code does not depend on what stands in front of them.
- **check:** pass
- **compare:** no significant difference, 0.996x [0.989 – 1.004]
- **Kept:** no
- **Notes:** rows of 256: 1.03x and 1.02x, the no-row points again. Rows of 64: 0.97x and 0.96x. Rows of 8: 0.99x in both tables. Rows of 31: 1.01x and 1.00x. The padding in front of a loop is run once per pass, and a call of a few rows is a few passes.

## Where the kernel stands

- **Rows under 32, in the first-level cache:** the loads. One 32-byte load per row or per `step` bytes, and a load that straddles two cache lines takes a cycle more: 0.19 ns per row of 31 at 4,096 elements, one cycle, where half the loads straddle. A row of 31 has two places a 32-byte load can cover it from, one byte apart, so there is no choosing a load that does not straddle.
- **Rows of 32 and more, in the first-level cache:** the same. Rows of 256 at 16,384 elements take 0.89 ns per row for eight loads with the matrix on a 64-byte boundary in a benchmark outside the judge, and 1.58 ns five bytes off it.
- **Beyond the first-level cache:** how fast the cache below delivers. 0.009 ns per element for rows of 31 from 65,536 to 1,048,576 elements, 0.013 at 4,194,304.
- **A call of a few rows:** about 2 ns for two rows of 31 and 6 ns for the `(30, 31)` matrix the kernel was asked for. The division on the narrow path costs 0.2 to 0.6 ns of that for one to eight rows and nothing from 30, measured outside the judge by leaving it out.

- **The recorded run,** `uv run judge run any_cols_bool --record --unlocked` at a load average of 1.2 to 1.3: 5.72x [5.69 – 5.74] over the naive C across the eight tables and 71.2x over NumPy. Rows of 31: 8.55x and 8.48x. Rows of 8: 14.2x in both. Rows of 64: 4.81x and 4.75x. Rows of 256: 1.71x and 2.00x. The one figure below the reference is rows of 256 with no row, 16 and 64 elements: 4.5 ns for 256 zeros against 4.2 to 4.6, a figure that moved between 3.2 and 4.7 ns from run to run and with where the code lay (attempt 6). The table with the output 16 bytes behind the input: 1.04x of the other overall, level from 4,096 to 4,194,304.
- **From Python,** at a load average of 0.6 to 0.7: `m.any(axis=0)` on a mask of 30 by 31 takes 95 ns against NumPy's 840 ns and on one of 1 by 31 96 ns against 503 ns, by a script outside the repository; `uv run delivery time --only any` on rows of 31 gives 5.5x at 16 elements, 17.9x at 4,096 and 37x at 1,048,576. The binding does not read the matrix for bytes other than 0 and 1: the kernel takes any non-zero byte as True on every path, as NumPy does along this axis.

`vpshufb` with an index vector read from a table is used to move bytes within a register; no `pext`, `pdep`, gather or masked store is used.

Left untried: strips that start on a 32-byte boundary of the memory for rows whose width is a multiple of 32, where every row then lies the same way and only the first and last strip of a row can straddle; more than eight strips in a pass for rows of several hundred; the division replaced by a multiplication with a reciprocal read from a table.
