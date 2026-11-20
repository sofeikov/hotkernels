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
