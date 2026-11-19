# A second shortlist: what a program in use calls

This is a proposal for review. It adds a second list of candidates to the one in [survey-design.md](survey-design.md). The first list came from operations chosen in advance. This one comes from the NumPy calls of one program in production use: which of them the kernels serve, which they miss, and which no kernel covers. It proposes what to build, in what order, and how the work is tracked across the two architectures.

Every figure here is x86-64's, measured on an Intel Core i5-14600KF. No ARM64 figure has been taken for these candidates.

## Summary

- **The source is one private codebase.** A data pipeline of about 14,000 lines of Python that leans on NumPy, polars and numba. Its NumPy calls were recorded while its own test suite ran, and the operations no kernel serves were timed against naive C in the survey's manner. No code from it appears here; calls are given by their NumPy spelling.
- **The existing kernels are reached.** Thirteen of the 21 installed kernels ran, 213,063 times in all, most of them `np.searchsorted` and `x[mask]` on float64. Eight were never reached.
- **Three new kernels where assembly has room:** `np.cumsum` on a boolean array (NumPy is 18x behind naive C at 131,072 elements), a running maximum (`np.maximum.accumulate`, `np.fmax.accumulate`: 8.6x and 1.3x, on a C loop the compiler leaves scalar), and `np.hypot` (7 ns per element; 12x behind the square-root form, which is not bit-identical).
- **Four where NumPy is behind and the gain over compiled C is uncertain:** `np.where(c, a, b)` (3.4x to 8.5x), reductions over short rows such as `m.any(axis=1)` (20x at every size), `np.fmax.reduceat` (5x at every size) and `np.bincount` on 16-bit and 64-bit integers (3.7x). They are worth building either way; the judge decides whether each ships as assembly or as C.
- **Three calls an existing kernel could take with a change to the binding only:** `mask.sum()`, `x[mask]` on int64, and `x[mask] = v` on a two-dimensional array.
- **Most of the recorded time is not in a loop at all.** `np.median`, `np.round` and `np.clip` on int64 took 63% of it, on arrays of about 30 elements, where NumPy's own call path is nearly the whole cost. These are hooks above NumPy's wrapper, as `np.clip` on floats already has, not assembly.
- **What is not known.** The program's real runs were not recorded, only its tests, whose arrays hold tens of elements. So the list says which calls occur and with which element types, not how a real run's time divides. The order below is provisional until one is recorded.

## What changed in the building

This section is added to as the steps of [the order of work](#order-of-work) are built. Each entry says what was built, where it departs from the text below, and which open decision it settled.

- **Step 1, the survey, on x86-64.** The candidates of rows 1 to 7 are the group `second shortlist` in `survey/candidates.py`, with their loops in `survey/naive.c`: `cumsum_bool`, `cummax_i64`, `cumfmax_f64`, `fmax_reduceat_f64`, `any_rows_bool`, `bincount_i16`, `hypot_f64` against the C library's `hypot` and against the square-root form, `where_i64`, and `np.where(c, a, np.nan)`. `where_f64`, `sum_bool` and `bincount_i64` were candidates already. The recorded run on the Intel Core i5-14600KF is [intel-core-i5-14600kf-second-shortlist.md](../survey/results/intel-core-i5-14600kf-second-shortlist.md). It agrees with [the table below](#numpy-against-naive-c) where the sizes are close, with one exception: `m.any(axis=1)` is 8x behind in the survey, not 20x. The script's loop had the row width of 22 written into it, which the compiler unrolled; the survey's takes the width as an argument, as a kernel must, and that loop costs 0.08 ns per element, not 0.03. The survey compares results with a NaN in the same place as equal, which the two candidates that produce NaN needed. ARM64 is not recorded yet ([issue 102](https://github.com/sofeikov/hotkernels/issues/102)).
- **Row 8, `mask.sum()` and `np.sum(mask)`.** Built as proposed: the function and the method hook of `sum` send a contiguous boolean array to `sum_u8`, and the result is the `np.intp` NumPy returns. The mask is read for bytes other than 0 and 1 first, as for the other mask kernels, since NumPy counts such a byte once and the kernel would add its value. The check and the sum go through the array 16,384 bytes at a time, so the second pass reads the first-level cache. The check costs more than the kernel: 0.012 ns per byte against 0.005. From Python on the Intel Core i5-14600KF, `np.sum(mask)` is 33x faster at 16 elements, 22x at 4,096 and 12.6x at 1,048,576; `mask.sum()` 15x, 15x and 12.6x.
- **Row 9, `x[mask]` on int64.** Built for selection only: `x[mask]`, `np.compress(mask, x)` and `np.extract(mask, x)` on a contiguous int64 array go to `compress_f64` under the conditions float64 has, and the result is int64. Neither architecture's kernel computes on an element, so the bits arrive unchanged; `delivery check` runs the int64 extremes through every size and mask. From Python on the Intel Core i5-14600KF `x[mask]` on int64 is 2.5x faster at 16 elements, 6.3x at 4,096 and 10.3x at 1,048,576, the same as float64. Assignment, `x[mask] = v` on int64, is not built. The 412 recorded assignments to int64 arrays all assigned an array of values, which `maskfill_f64` does not take for any type, and a scalar would need NumPy's rules for storing a Python number into an integer array reproduced. It stays in [issue 104](https://github.com/sofeikov/hotkernels/issues/104)'s discussion as a possible follow-up.
- **Row 10, `x[mask] = v` in more than one dimension.** Built as proposed, and for `x[mask]` as well: the indexing hook took both only for one-dimensional arrays, and now takes them whenever the mask has the shape of `x`. NumPy then reads both in C order, which for contiguous arrays is what the kernels do with two flat buffers. A mask of fewer dimensions selects rows, and one of the same size and another shape is an error; both stay with NumPy. From Python on the Intel Core i5-14600KF, on an array of four columns with half the mask True, `x[mask] = 0.0` is 7.2x faster at 16 elements and 9.9x at 4,096 (8.3x and 9.2x in the recorded run of the closing pull request), and `x[mask]` 2.8x and 6.1x, and 10.2x at 1,048,576: the figures of the one-dimensional call.
- **Row 1, `cumsum_bool`, on x86-64.** Spec, reference, x86-64 kernel and binding in one pull request, so the kernel's runs are `--unlocked`: the spec was not on `main` to lock against. The spec and reference were committed before the first line of assembly and not changed after. The kernel is 1.89x over the naive C across the judge's sizes: 3.0x inside the first-level cache, 1.9x in the second-level cache, where writing eight bytes per element is the limit, and 2.6x beyond the caches with stores that bypass them. The hook is `cumsum`'s method hook, as proposed. One thing the design did not say: the kernel adds the mask's bytes, so the binding reads the mask for bytes other than 0 and 1 first, as it does for the other mask kernels. From Python `np.cumsum(mask)` is 3.1x, 9.7x and 9.5x faster at 16, 4,096 and 1,048,576 elements (3.0x, 9.6x and 13.9x in the recorded run of the closing pull request). On ARM64 the kernel ships as its C until [issue 109](https://github.com/sofeikov/hotkernels/issues/109).
- **Row 4, `where_f64`, on x86-64.** Spec, reference, x86-64 kernel and binding in one pull request, so the kernel's runs are `--unlocked`, as for row 1. The kernel is 1.37x over the naive C across the judge's sizes and both distributions, 8.1x over NumPy; it is limited by memory from the second-level cache out, and stores past the caches from 24 MB of output. The hook is `np.where`'s function hook, as proposed, which now serves both forms: the one-argument form still counts under `nonzero`, and the three-argument form is a new operation, `where`, so the package now hooks 16 operations. Built as proposed for two arrays of one shape, float64 or int64 (the int64 call takes the same kernel, which moves bit patterns, as expected), and for a float64 array beside a number: a Python float, a Python int a double holds exactly or an `np.float64`, on either side. Three things the design did not say. The binding reads the condition for bytes other than 0 and 1 first, as for the other mask kernels, although the x86-64 kernel would take any non-zero byte as NumPy does: the spec promises only 0 and 1 and ARM64 runs the C. Shapes that broadcast, an int64 array beside a number, and mixed element types stay with NumPy. And a number is not spread over a buffer of n elements: the kernel runs 256 elements at a time against a buffer of 256 on the stack. With a buffer of n, `np.where(c, a, np.nan)` was no faster than NumPy at 1,048,576 elements (1.01x); in blocks it is 2.5x, at the cost of never storing past the caches. From Python `np.where(c, a, b)` on float64 with half the condition True is 4.7x, 3.2x and 6.0x faster at 16, 4,096 and 1,048,576 elements, and `np.where(c, a, np.nan)` with 95% True 4.4x, 3.0x and 2.5x, at a load average of about 3. Two of the kernel's attempts, a masked store and stores past the caches from a lower size, were left for a person to decide; see [kernel-findings.md](kernel-findings.md#npwherec-a-b). On ARM64 the kernel ships as its C until [issue 114](https://github.com/sofeikov/hotkernels/issues/114), which also settles open decision 5 for this row.
- **Row 2, `cummax_i64` and `cumfmax_f64`, on x86-64, and decision 2.** Spec, reference, x86-64 kernels and binding in one pull request, so the kernels' runs are `--unlocked`, as for rows 1 and 4. `cummax_i64` is 1.16x over the naive C across the judge's sizes and both distributions, 12.0x over NumPy; `cumfmax_f64` 8.55x and 13.3x. The int64 kernel avoids the maximum where one compare shows nothing changes (blocks that are flat or rising), because AVX2 has no 64-bit integer maximum and clang's naive C, a predicted branch per element, is already at 0.15 ns. The float64 kernel turns NaN into minus infinity and keeps a running maximum inside each vector, a chain of one `vmaxpd` per eight elements; the naive C is a chain of eleven cycles per element. Decision 2 is settled for the method: `accumulate` is an entry in the ufunc type's method table, and the method hook changes its function pointer as it changes `ndarray`'s, so the package still has five kinds of hook, not the fifth the decision foresaw. Three reasons. On a short array nearly all of the call is NumPy's `accumulate` machinery (about 220 ns at 16 elements), which only a hook above it removes. `reduceat`, row 6, calls the inner loop once per run, so an inner-loop hook could never serve it, and it can join the same table. And every other ufunc's `accumulate` pays two pointer comparisons. The hook is `ufunc.accumulate`, a new operation `accumulate`, so the package now hooks 17 operations. Built as proposed: `np.maximum.accumulate` and `np.fmax.accumulate` on int64 both take `cummax_i64`, `np.fmax.accumulate` on float64 takes `cumfmax_f64`, and `np.maximum.accumulate` on float64, which propagates NaN, stays with NumPy. One thing the design did not say: NumPy does not ignore a signalling NaN in `np.fmax` (the C library's `fmax` returns a quiet NaN and the maximum so far is lost), the spec leaves signalling NaNs out, and the binding reads a float64 array for them first. The same read leaves to NumPy an array that holds a negative zero or NaNs of more than one bit pattern: NumPy's sign for a zero running maximum among zeros of both signs differs between its x86-64 and ARM64 builds (seen in CI on Apple Silicon), and the kernel reproduces the x86-64 one. The read takes about 0.12 ns per element, near the kernel's own time. Only one-dimensional arrays are taken; an `axis` over a two-dimensional array is NumPy's. From Python `np.maximum.accumulate(x)` on int64 is 2.6x, 11.6x and 6.6x faster at 16, 4,096 and 1,048,576 elements, and `np.fmax.accumulate(x)` on float64 2.8x, 6.8x and 5.0x, at a load average of about 1 (in the recorded run of the closing pull request, after row 6 made the read for signalling NaNs faster and before the read also looked for negative zeros and mixed NaNs: 2.6x, 11.2x and 5.3x, and 2.8x, 8.2x and 5.5x). On ARM64 both kernels ship as their C until [issue 122](https://github.com/sofeikov/hotkernels/issues/122).
- **Row 6, `np.fmax.reduceat`, on x86-64.** Spec, reference, x86-64 kernel and binding in one pull request, so the kernel's runs are `--unlocked`, as for the rows before. The design named a kernel `fmax_reduceat_f64` that takes the array and the starts. What was built is `fmax_reduce_f64`, the NaN-ignoring maximum of one run (`np.fmax.reduce(x)`), which the binding calls once per run: the judge cannot describe a kernel whose second buffer holds indices into the first without being extended, a call costs a few nanoseconds against a run of 16 elements, and the same kernel can later serve `np.fmax.reduce`. It is 6.63x over the naive C across the judge's sizes, 3.4x at 16 elements. The hook is `ufunc.reduceat`, beside `ufunc.accumulate` in the ufunc type's method table, as decision 2 foresaw; a new operation `reduceat`, so the package now hooks 18 operations. Four things the design did not say. The sign of a zero maximum with zeros of both signs present, and which NaN a run of NaNs returns, follow the four lanes of NumPy's AVX2 loop, which the spec and the kernel reproduce; NumPy's AVX-512 loop has eight, so the binding sends a call to NumPy when the array holds a negative zero and a result is a zero, or NaNs of more than one bit pattern and a result is a NaN, and does not depend on which loop NumPy runs. A signalling NaN anywhere sends the call to NumPy, as for `np.fmax.accumulate`; the array is read for one a block of 2,048 elements at a time just ahead of the runs, because a separate first pass made long runs on arrays beyond the second-level cache 0.6x of NumPy. The functions that read the array for these guards are compiled for AVX2, which `install()` already requires on x86-64, and the read for signalling NaNs, shared with `np.fmax.accumulate`, went from 0.17 to about 0.05 ns per element. And `vmaxpd` leaves the invalid flag set after a NaN, which NumPy's own loop does not; the binding of row 2 did not clear it, and both hooks now clear MXCSR's flags after the kernel. Only an int64 array of starts is taken; a list and int32 starts are NumPy's to convert. From Python on runs of 16 with 5% NaN `np.fmax.reduceat(x, starts)` is 2.7x, 12.7x and 10.9x faster at 16, 4,096 and 1,048,576 elements, at a load average of about 0.5; without NaN 2.9x at 4,096 and 2.2x at 1,048,576. On runs of several hundred elements it is about NumPy's speed, and on runs of 4,096 in an array of 262,144 elements 0.8x. `np.logical_or.reduceat`, which the program also calls, is not built. On ARM64 the kernel ships as its C until [issue 127](https://github.com/sofeikov/hotkernels/issues/127).
- **Row 5, `any_rows_bool`, on x86-64.** Spec, reference, x86-64 kernel and binding in one pull request, so the kernel's runs are `--unlocked`, as for the rows before. Step 7 expected a new shape for the judge, a kernel over rows; none was needed. The kernel takes the matrix as `n` elements and the row width as a scalar argument, writes one result per whole row and returns the number of rows, and the spec gives the width a set of values the judge sweeps (17 widths from 1 to 1,000) and times (22, 4 and 64). It is 6.94x over the naive C across the judge's sizes and its six tables: 5.0x on rows of 22, 17.6x on rows of 4, which have a path of their own, and 3.8x on rows of 64. The hooks are as proposed, the method and the function: `np.any(m, axis=1)` calls `np.logical_or.reduce` itself for a plain `ndarray` and never reaches the method. A new operation `any`, so the package now hooks 19 operations. Only `axis` given as the Python int 1 or -1 and no other argument is taken; `m.any()` with no argument goes to NumPy at the method hook's first test. One thing the design did not say: the matrix is read for bytes other than 0 and 1 first, as for the other mask kernels, and here it is required: on the kernel's path for fewer than 32 elements with a width other than 4, a byte above 1 changes the results of the rows beside it. The check goes 16,384 bytes at a time just ahead of the kernel and is compiled for AVX2, as row 6's guards are; it still costs about 0.007 ns per byte against the kernel's 0.012. From Python on rows of 22 `m.any(axis=1)` is 5.1x, 16.2x and 42.2x faster at 16, 4,096 and 1,048,576 elements, and `np.any(m, axis=1)` 8.0x, 16.8x and 42.2x, at a load average of about 0.4. `m.all(axis=1)` and the other reductions over short rows are not built. On ARM64 the kernel ships as its C until [issue 132](https://github.com/sofeikov/hotkernels/issues/132).
- **Row 7, `bincount_i16` and `bincount_i64`, on x86-64.** Spec, reference, x86-64 kernels and binding in one pull request, so the kernels' runs are `--unlocked`, as for the rows before. The kernels take the number of counters and ignore elements outside it, so they are safe on any input. `bincount_i16` is 1.26x over the naive C across the judge's four tables, 6.7x over NumPy; `bincount_i64` 1.16x and 4.0x. The several-tables technique of `bincount_u8` became two: a table of pairs for at most 64 counters, one increment per two elements, and a second table where a sample of the data repeats, up to 1,024 counters, both from 8,192 elements; below that the kernels tie the C. The hook is the function hook of `np.bincount`, as proposed, and the operation `bincount` is not new. Three things the design did not say. The binding reads the array once for its largest element first, since NumPy sizes its result by it and raises for a negative element; the scan, compiled for AVX2 with four running maxima, costs about 0.1 ns per int64 element, and it is why `np.bincount` on int64 gains 1.9x at a million elements from Python where int16 gains 4.6x. A result longer than 65,536 counts and longer than the array is NumPy's, whose zeroed memory from `calloc` costs it nothing there. And NumPy 2.5.5 itself writes outside its result for an int64 array whose largest element is the largest int64. On ARM64 both kernels ship as their C until [issue 137](https://github.com/sofeikov/hotkernels/issues/137).
- **Row 12, `np.clip` on int64, and decision 4.** Decision 4 is settled: the short-array hooks belong here, and they are not kernels. In the program's tests these calls were on arrays of tens of elements, where nearly all of a call is NumPy's own call path and the speed of the loop is irrelevant, so each is a plain C loop inside the binding, taken only up to a limit in size: no `kernels/` directory, no assembly, no judge, checked against NumPy by `delivery check` like the rest of the binding. Each limit is the largest power of two among 16, 64, 256, 1,024 and 4,096 elements at which the hooked call, measured from Python, is still at least 1.5x faster than NumPy; above it the call goes to NumPy as before. A loop has a counter like a kernel's, so `install()` probes it and `report()` shows it, as `C` and `exact`, on a row that the count of kernels on the first line leaves out. Built as proposed for `np.clip(x, lo, hi)` and `x.clip(lo, hi)`, through the function and method hooks of `clip`, which already served floats: a contiguous int64 array of 1 to 4,096 elements, both bounds Python ints that fit int64 or `np.int64`, `lo <= hi`, and no `out` or other argument. One thing the design did not say: the loop is compiled for the kernels' instruction level (`HK_TARGET_KERNEL_LEVEL`), where it is a compare and a blend per four elements, so it stays ahead of NumPy well past 4,096 elements (2.0x at 16,384, 1.5x at 32,768); the limit is kept at 4,096, the largest size it was chosen among, which also keeps the loop under the interpreter lock. From Python, `np.clip(x, 2, 7)` on int64 is 15.4x faster at 16 elements (1.8 µs to 117 ns) and 3.6x at 4,096, and `x.clip(2, 7)` 12.3x and 3.0x, at a load average of about 0.6; at 1,048,576 elements the call is NumPy's and costs 0.8% more. The loop is the same C on ARM64, where it was not run.
- **Row 13, `np.round` on float64.** A loop built into the binding, as decision 4 settled with row 12: `np.round(x, d)`, `np.around(x, d)` and `x.round(d)` on a contiguous float64 array of 1 to 4,096 elements, `decimals` absent or a Python int from 0 to 15, and no `out`. NumPy's `PyArray_Round` makes three ufunc calls, multiply by 10^d, `rint`, divide by 10^d (`rint` alone for d = 0), and the loop does the same three operations per element in the same order, so the result is the same bits, `-0.0` from a negative value that rounds to zero included; `delivery check` found no difference on the judge's value classes, on ties and on signed zeros for every number of decimals taken. Three things the design did not say. NumPy turns the flags of those operations into warnings, so the flags are cleared before the loop and read after it, and an overflow in the multiply or a signalling NaN sends the call to NumPy, which warns; underflow cannot arise, since a subnormal number times a power of ten is exact while it stays subnormal. `np.around` is a second function object in NumPy 2, with the same body, and has a function hook of its own; the method hook counts every call the two pass on, as a new operation, `round`, so the package now hooks 20 operations. And the loop is compiled for the kernels' instruction level, where `rint` is a `vroundpd`; the division is then most of its time, about 0.4 ns per element. From Python, `np.round(x, 2)` is 7.6x faster at 16 elements (1.16 µs to 153 ns) and 2.2x at 4,096, `x.round(2)` 5.6x and 2.1x, and `np.round(x)` 3.8x and 1.5x (1.8x at 4,096 in the recorded run of the closing pull request), at a load average of about 1.3; at 8,192 elements `np.round(x)` was 1.1x, which set the limit at 4,096. Above it the call is NumPy's and costs under 1% more. The loop is the same C on ARM64, where it was not run.
- **Row 11, `np.median` on float64.** A loop built into the binding, as decision 4 settled with row 12, and as that decision foresaw the hook replaces all of NumPy's Python function for the calls it takes: `np.median(x)` with the array alone, by position, on a contiguous float64 array of any shape with 1 to 128 elements and no NaN. The array is copied to a buffer on the stack and sorted by insertion, and the result is the middle element, or the two middle elements added and divided by two, which is what NumPy's mean of them computes, as an `np.float64`; `delivery check` found it bit-identical on the judge's value classes at every size up to the limit. Three things the design did not say. The guards the brief foresaw are recognised from the numbers, not from the status flags: a sum that overflows or meets infinities of both signs, which NumPy warns about, and also a half that underflows, which NumPy reports under `np.errstate(under=...)`. A zero median beside a negative zero goes to NumPy wherever the negative zero is, since NumPy 2.5's mean of a lone `-0.0` is `+0.0` as well as its partition deciding which zeros are in the middle. And the limit is lower than for the other two: the sort takes time in the square of the count, and the hooked call was 47x faster than NumPy at 16 elements, 13x at 64, 3.2x at 128 and 0.86x at 256 on normally distributed values (1.48x at 128 on values in descending order, the sort's worst case). A new operation, `median`, so the package now hooks 21 operations. From Python, `np.median(x)` on 16 float64 elements is 44x faster (4.93 µs to 112 ns), at a load average of about 0.7; above 128 elements the call is NumPy's. On 4,096 elements a call left to NumPy measured 8% to 11% slower with hotkernels installed, float32 3% to 4%, and the same on the branch before this hook existed, so it is not the hook's test; the last entry explains it. The loop is the same C on ARM64, where it was not run.
- **The stack as a whole, on x86-64.** The closing pull request built nothing new; it checked the top of the stack and recorded its figures on a quiet machine (load average 0.3 to 1.1). `judge selftest` passes; `judge check --unlocked` passes for the eight new kernels, and plain `judge check` for `argmax_f64`, `compress_f64` and `sum_u8`, whose locked files are unchanged from `main`; `delivery check` agrees on 166,020 calls over the 21 operations, 94,638 of them taken by a kernel or a loop; and NumPy's own test suite, which no pull request of the stack had run, gives the same outcome for all 48,659 tests with and without hotkernels (47,480 pass both ways). The judge's reports of the eight kernels and `delivery time` for the Intel Core i5-14600KF were recorded again, and where a figure quoted above differed from them by more than a tenth the recorded one is given beside it. The 8% to 11% that row 11 found on `np.median(x)` at 4,096 float64 elements, a call no hook takes, is how `delivery time` measured, not a cost of the hooks. The calls just after `install()` run slow for about 3 ms while the core and its caches recover from its probes, about 0.4 ms lost in all, the same as after a 1.2 ms `time.sleep` and the same on `main`, and the tool timed the hotkernels batch, of 10 ms, right after `install()`. With hooks in place and no `install()` just before, the call takes NumPy's time. The tool now makes half a batch of untimed calls before every batch on both sides, and the recorded run has the call 2.8% faster with hotkernels installed, which is noise.
- **Row 3, `np.hypot` on float64, built for the owner to judge.** Option (a) of [issue 96](https://github.com/sofeikov/hotkernels/issues/96)'s last comment: the exact kernel with an inner-loop hook and a self-test at `install()`. Decision 1 stays the owner's; this pull request is one answer to it, at the top of the stack, so that closing it changes nothing below. The reference is a loop that calls the C library's `hypot`, which is NumPy's result everywhere; the x86-64 kernel computes the corrected algorithm of glibc's `e_hypot.c` without fused multiply-add, bit for bit equal to glibc 2.43's `hypot` on 36 million pairs over the whole range, and passes `judge check` only where the C library is that algorithm. What makes that safe to ship is the hook's probe: `install()` compares the kernel with NumPy's own loop on 20,000 pairs of ordinary and special values and keeps the hook only if every bit agrees, which a kernel differing by a single fused multiply-add fails on 16 of them. The binding goes further than the design said in two ways. The floating-point flags set before the kernel are kept, and a call on which the kernel raises one NumPy reports is computed again by NumPy's loop, so the warnings are NumPy's; the kernel raises one more flag than the C library, underflow, on pairs whose smaller input is negligible and very small, which costs those calls time and nothing else. And an output that is an input is computed in blocks of 256 elements on the stack, so that NumPy's loop can still redo a flagged block. Over the naive C 5.25x; from Python 1.18x at 16 elements, 4.65x at 4,096 and 6.75x at a million: the gain is on large arrays only, as the issue foresaw, since the recorded calls were on arrays of tens of elements, where a hook on the ufunc's own call (option (b)) would be needed. A new operation, `hypot`, so the package now hooks 22 operations; `delivery check` agrees on 170,982 calls, 4,953 of them `np.hypot`. On ARM64 the kernel is its naive C, the C library's own `hypot`: no gain and nothing that can differ.

What of this design is built: step 1 on x86-64, rows 8 to 10 in the binding, rows 1 to 7 as nine kernels with x86-64 assembly, and rows 11 to 13 as loops inside the binding. Row 3, `hypot_f64`, is built as one answer to decision 1 for the owner to judge ([issue 96](https://github.com/sofeikov/hotkernels/issues/96)), in the last pull request of the stack. What is not: the ARM64 assembly of the nine kernels, which ship there as their C until their issues are closed; the survey's run on ARM64 ([issue 102](https://github.com/sofeikov/hotkernels/issues/102)); and the recorded real run of decision 3, which needs the program's production data. Decisions 2 and 4 are settled, by rows 2 and 12. Decision 1 is informed, not settled: the survey recorded `np.hypot` against both C forms, and row 3 built option (a) for review; whether to keep it, or to want option (b) or (d) instead, is still for the owner and a real run to show. Decision 5 is open with the ARM64 survey.

## Where the candidates come from

### The program

A pipeline that parses large feeds into tables, aligns two time series, and computes reports and model inputs from them. Per run it holds arrays of about 130,000 elements (one per time step) and about three million (one per time step and tracked object), and it also works through many short arrays in Python loops. Its heaviest numeric loop is compiled with numba, which calls no NumPy function that hotkernels can reach.

It is private. This document names no file, function or data of it.

### What was recorded

Four things, by throwaway scripts that are not in the repository:

1. **Calls.** `sys.monitoring` (Python 3.12 and later) reported every call from the program's own code into NumPy, with the caller's line, the size and element type of the first array, and the time to return. Operators and subscripts are not calls and are not seen. It ran twice: once for functions and ufuncs on plain NumPy (1,074,562 calls from 381 call sites), which the times below are from, and once with array methods as well and hotkernels installed (1,124,106 calls from 493 call sites).
2. **Subscripts.** The program's modules were rewritten as they were imported, so that `a[b]` and `a[b] = v` recorded the array's type and dimensions when `b` was a boolean or integer array. One module, the one holding the numba code, was left out, and 28 tests that pass otherwise failed under the rewrite, so these counts are a lower bound.
3. **`report()`.** The same test run with `hotkernels.install()`: kernel calls and calls left to NumPy.
4. **NumPy against naive C.** For each operation no kernel serves, the NumPy statement against the simplest C loop, compiled with the judge's flags (`clang -O3 -march=native`), at 32, 512, 8,192, 131,072 and 2,097,152 elements. Where NumPy allocates its result the C side allocates and frees its output on every call. The median of seven interleaved samples, on one pinned performance core. Every C loop was checked against NumPy on the timed input.

Machine: Intel Core i5-14600KF, Linux 7.0, CPython 3.13.15, NumPy 2.5.3, clang 21.1.8, hotkernels 0.2.0. Load average 0.5 to 0.7. The timings were taken twice; where a figure moved by more than a tenth between the two passes, both ends are given. They are naive timings, as the survey's are: good enough to rank candidates, and each candidate is measured again by the judge before it ships.

### Limits

- **Tests, not runs.** The program's test suite ran, 703 tests, on arrays of tens of elements. A real run needs data that was not available. The call counts say what the code does; they do not say how a real run's time divides between short arrays and long ones.
- **The recorded calls are a small part of the run.** The calls the program's own code makes into NumPy functions took about 2.5 s of a test run of 45 s, once the tracer's cost is taken off. Arithmetic and comparisons written as operators are not in that figure. The rest is parsing, polars, compiled code and the tests themselves.
- **The tracer costs about 0.26 µs per call,** which is in every time below. A call that takes 0.2 µs untraced shows as 0.5.
- **One program, one architecture.** NumPy's own vector code differs by architecture, so a gap here says nothing certain about ARM64.
- **The outcome of the tests did not change.** With hotkernels installed the same tests passed and the same failed as without it.

## What the existing kernels took

From `report()` over the test run:

| NumPy call | Elements | Kernel calls | Left to NumPy |
|---|---|---|---|
| `np.searchsorted` | float64 | 78,592, and 805 with `side='right'` | 6,852 |
| `x[mask]` | float64 | 91,014 | 101,693 |
| `x[mask]` | float32 | 7,488 | |
| `np.argmin` | float64 | 14,018 | 0 |
| `np.nonzero` | bool | 12,782 | 832 |
| `x[mask] = v` | float64 | 5,891 | 8,448 |
| `np.cumsum` | int64 | 1,373 | 5,756 |
| `np.clip` | float64 | 426 | 12 |
| `np.isin` | int64 | 424 | 123 |
| `np.argmin` | float32 | 235 | |
| `np.argmax` | float64, float32 | 13, 2 | 264 |
| `np.bincount` | uint8 | 0 | 261 |
| `np.interp` | float64 | 0 | 1,521 |
| `np.histogram` | float64 | 0 | 3 |
| `np.sum` | uint8 | 0 | 20,709 |
| `x.astype(np.float32)` | uint8 | 0 | 44,622 |
| `np.count_nonzero` | int64, float64 | 0 | 1 |

"Left to NumPy" is counted per NumPy call, for every caller in the process: the program, NumPy's own Python code (`np.unique` and `np.median` index by mask themselves), polars and the test tools. What the program's own code does is in the other records:

- **`np.searchsorted`:** 94% of its calls are on float64 and are served. The rest are on float32 (3,154), int64 (1,656) and int32 (412).
- **`x[mask]` in its own code:** 3,065 on a one-dimensional int64 array, 1,966 on a two-dimensional float64 array with a mask over its rows, 556 on float64 and 470 on float32 in one dimension, 615 on strided arrays. Only the float ones in one dimension are served.
- **`x[mask] = v` in its own code:** 3,920 on float64 in one dimension with a number, which are served. Left: 880 on a two-dimensional float64 array with a mask of the same shape, 390 on float32, 412 on int64 with an array of values, and 1,137 that select rows of a two-dimensional array.
- **`np.cumsum`:** 1,889 of 2,378 calls are on a boolean array. No kernel takes them.
- **`np.clip`:** 78,131 of 80,913 calls are on an int64 array, clamping the indices a search returned. They never reach a hook, so `report()` does not count them. The float calls with one bound, `np.clip(x, 0, None)`, go to NumPy's `maximum` and are not counted either.
- **`np.bincount`:** on int16 and int64, never uint8.
- **`np.sum` and `astype`:** never on uint8.
- **`a[idx]` with an integer array** is the commonest subscript of all: 18,507, nearly three times the mask selections. It is not a target; see [below](#not-targets).

## Where the time went

The calls that took the most recorded time, without hotkernels installed:

| Call | Calls | Share of recorded time | Mean elements | Elements |
|---|---|---|---|---|
| `np.median` | 169,039 | 38% | 30 | float64 |
| `np.round(x, 2)` | 37,083 | 16% | 19 | float64, some float32 |
| `np.clip` | 80,913 | 9% | 31 | int64 |
| `np.where(c, a, b)` | 99,153 | 6% | 79 | int64, float64 |
| `np.searchsorted` | 86,077 | 5% | 648 | float64 |
| `np.isnan` | 178,970 | 3% | 58 | float64 |
| `np.abs` | 159,938 | 3% | 32 | float64 |
| `np.maximum` | 81,113 | 2% | 30 | int64 |
| `np.isclose` | 1,641 | 2% | | numbers, not arrays |
| `np.diff` | 9,579 | 2% | 22 | float64, int64, float32 |
| `np.hypot` | 78,268 | 2% | 31 | float64 |
| `np.unique` | 1,955 | 2% | 16 | int64 |
| `np.cumsum` | 2,378 | 1% | 34 | bool, some int64 and float64 |
| `np.flatnonzero` | 10,226 | 1% | 17 | bool |

Two things stand out:

- **One loop makes most of these calls.** Each pass searches a sorted array, clamps the indices with `np.clip`, picks between two candidates with `np.where`, takes `np.hypot`, drops NaN with a mask and takes `np.median`, on arrays as long as a few dozen items. That length does not grow with the data: a real run makes more passes, not longer ones.
- **At these sizes the time is NumPy's call path.** `np.median` of 32 elements takes 4.8 µs, `np.clip` on 32 int64 elements 1.75 µs, `np.round(x, 2)` 1.1 µs. The C loops take 58, 6 and 12 ns.

## NumPy against naive C

NumPy's time divided by the naive C loop's. Above 1 the C loop is faster.

| NumPy statement | Elements | 32 | 512 | 8,192 | 131,072 | 2,097,152 | ns per element at 131,072: NumPy / C |
|---|---|---|---|---|---|---|---|
| `np.cumsum(mask)` | bool | 156x | 15.4x | 8.73x | 18.5x | 1.52x | 3.60 / 0.19 |
| `m.any(axis=1)`, rows of 22 | bool | 138x | 41.0x | 18.3x | 20.5x | 20.5x | 0.71 / 0.03 |
| `np.hypot(a, b)`, against `sqrt(a*a + b*b)` | float64 | 18.3x | 6.36x | 11.6x | 12.2x | 10.2x | 7.09 / 0.58 |
| `np.hypot(a, b)`, against a loop calling `hypot` | float64 | 4.11x | 1.48x | 1.24x | 1.08x | 1.09x | 7.05 / 6.52 |
| `np.maximum.accumulate(x)` | int64 | 28.6x | 9.09x – 10.4x | 9.33x – 10.7x | 8.62x | 2.97x | 1.55 / 0.18 |
| `np.fmax.accumulate(x)` | float64 | 10.4x | 1.44x | 1.20x | 1.33x | 1.31x | 2.28 / 1.71 |
| `np.where(c, a, b)`, half true | float64 | 44.2x | 5.39x | 2.29x | 8.49x | 3.59x | 2.63 / 0.30 |
| `np.where(c, a, b)`, half true | int64 | 44.3x | 5.41x | 2.59x – 6.83x | 8.42x | 3.58x – 3.96x | 2.62 / 0.31 |
| `np.where(c, a, np.nan)`, 95% true | float64 | 76.0x | 7.22x | 3.00x | 3.35x | 1.62x | 0.60 / 0.17 |
| `np.where(np.isnan(x), 0, x)` | float64 | 175x | 17.0x | 4.07x | 4.32x | 2.31x | 0.72 / 0.16 |
| `np.fmax.reduceat(x, starts)`, runs of 16 | float64 | 18.7x | 5.13x | 5.33x | 5.35x | 4.98x | 3.38 / 0.63 |
| `mask.sum()` | bool | 249x | 29.1x | 6.01x | 4.53x | 4.29x | 0.21 / 0.05 |
| `np.bincount(x, minlength=64)` | int16 | 41.3x | 8.71x | 4.71x | 3.71x | 4.27x – 4.96x | 0.75 / 0.20 |
| `np.argsort(x, kind="stable")`, 50 distinct values | int16 | 26.5x | 5.77x | 2.33x – 3.41x | 2.90x | 3.27x | 2.72 / 0.94 |
| `np.round(x, 2)` | float64 | 93.6x | 7.49x | 2.00x | 1.71x | 2.50x | 0.66 / 0.38 |
| `np.median(x)` | float64 | 84.3x | 7.26x | 3.80x | 0.67x | 1.17x | 4.65 / 6.98 |
| `np.clip(x, lo, hi)` | int64 | 279x | 20.5x | 2.88x | 1.40x | 0.96x | 0.23 / 0.16 |
| `np.diff(x)` | float64 | 125x – 143x | 15.6x – 19.6x | 2.88x – 3.39x | 1.11x – 1.85x | 0.97x | 0.19 – 0.29 / 0.16 |
| `np.repeat(a, counts)` | int64 | 13.2x | 2.45x | 2.30x – 2.59x | 1.29x | 0.78x | 6.97 / 5.35 |
| `a[idx]`, sorted indices | float64 | 19.2x | 4.37x | 1.91x | 1.53x | 1.00x | 0.47 / 0.30 |
| `mask.astype(float)` | bool | 35.7x | 4.57x – 6.29x | 1.54x – 1.83x | 1.30x – 1.64x | 1.06x – 1.31x | 0.13 – 0.17 / 0.10 |
| `np.isnan(x)` | float64 | 38.2x | 3.71x | 0.76x | 0.55x | 0.67x | 0.09 / 0.16 |
| `np.nanmax(x)` | float64 | 32.6x | 1.69x – 2.68x | 0.16x | 0.06x | 0.06x | 0.10 / 1.74 |

How to read it:

- **The column at 32 elements is the call, not the loop.** Every row is 10x or more behind there, including the ones NumPy wins at size. A kernel only gets that factor if a hook reaches it before NumPy's wrapper.
- **A gap that stays above 2x from 8,192 to 2,097,152 elements is in the loop.** That is `m.any(axis=1)`, `np.hypot`, `np.fmax.reduceat`, `np.where`, `mask.sum()`, `np.bincount` and `np.maximum.accumulate`.
- **`np.cumsum(mask)` falls to 1.5x at 2,097,152 elements** because it writes eight bytes for every byte it reads, and beyond the caches the writing is the cost.
- **`np.where` with a mask that is half true is NumPy's worst case:** it takes a branch per element. With a mask that is 95% true it is 3.4x behind at 131,072 elements, not 8.5x. The int64 figure at 8,192 elements moved between passes, 2.6x and 6.8x, as a branch predictor that has half learnt a repeated mask would make it.
- **The naive C is not the limit for the loops that carry a value from one element to the next.** clang's own remarks say it leaves the running maxima and `np.cumsum(mask)` scalar, and `np.bincount` as well. It vectorises `np.where`, `mask.sum()`, `np.clip`, `np.round`, the square-root form of `hypot` and the loop over one run of `np.fmax.reduceat`.
- **`sqrt(a*a + b*b)` is not `np.hypot`.** On two million pairs of normally distributed values 16.7% of the results differ, each by one unit in the last place. It also overflows where `hypot` does not.

## The shortlist

Each row is a family: one NumPy operation, one or more kernels. "Hook" says how the call would reach it, with the hooks of [delivery-design.md](delivery-design.md#which-hook-serves-which-kernel).

### New kernels where assembly has room

| # | Kernel | NumPy call | NumPy vs naive C at 131,072 | Hook | Technique |
|---|---|---|---|---|---|
| 1 | `cumsum_bool` | `np.cumsum(mask)`, int64 result | 18.5x behind | Method, as `cumsum_i64` | A prefix sum of bytes within a block, widened to 64 bits and added to a carried total. Exact |
| 2 | `cummax_i64`, `cumfmax_f64` | `np.maximum.accumulate(x)`, `np.fmax.accumulate(x)` | 8.6x, 1.3x | Open: see [decision 2](#decisions-still-open) | A prefix maximum within a vector, then against the carried maximum. Exact for both, unlike a float prefix sum. `np.minimum.accumulate` was in the first survey for the same reason (1.4x behind on x86-64) and had no caller then |
| 3 | `hypot_f64` | `np.hypot(a, b)` | 12.2x behind the square-root form | Inner loop | Open: see [decision 1](#decisions-still-open). NumPy calls the C library once per element |

### New kernels where the gain over compiled C is uncertain

clang vectorises the first three of these loops, and for the fourth the gain over C is known to be small. NumPy is behind all four, so each is worth a kernel; where assembly does not beat the C, the C ships, as `cast_u8_f32` does on ARM64. The same guess was made for six kernels in the first survey and was wrong for five ([kernel-findings.md](kernel-findings.md)), so assembly is attempted for each.

| # | Kernel | NumPy call | NumPy vs naive C at 131,072 | Hook | Note |
|---|---|---|---|---|---|
| 4 | `where_f64` | `np.where(c, a, b)` with two arrays, or an array and a number | 3.4x to 8.5x behind | Function: `np.where` is hooked already, for its one-argument form | The first survey found it 4.5x to 10x behind on x86-64 and level on ARM64. A selection moves elements without reading them as numbers, so the int64 call is expected to take the same code |
| 5 | `any_rows_bool` | `m.any(axis=1)` on a two-dimensional boolean array | 20x behind, at every size | Method and function | NumPy pays its per-row cost on every short row. The C loop was compiled for rows of exactly 22; a kernel takes any row length. [workloads-design.md](workloads-design.md#what-was-verified) found the same shape in scikit-learn: `argmax(axis=1)` over rows of three |
| 6 | `fmax_reduceat_f64` | `np.fmax.reduceat(x, starts)` | 5.4x behind, at every size | Open: see [decision 2](#decisions-still-open) | Timed on runs of 16 elements. The program also calls `np.logical_or.reduceat` |
| 7 | `bincount_i16`, `bincount_i64` | `np.bincount(x)` on int16 and int64 | 3.7x behind for int16 | Function, as `bincount_u8` | NumPy first converts the array to 64-bit integers. The first survey had the int64 loop 2.4x behind. The several-tables technique of `bincount_u8` applies, and there it was worth 1.04x to 1.24x over the C |

### Existing kernels, more calls

No new assembly. Each needs cases in `delivery check`, and the first two a spec that says the kernel is right for the type.

| # | Call | Served by | What changes |
|---|---|---|---|
| 8 | `mask.sum()`, `np.sum(mask)` | `sum_u8` | The binding accepts a boolean array and returns NumPy's type for it. A mask holding bytes other than 0 and 1 goes to NumPy, as for the mask kernels ([issue 8](https://github.com/sofeikov/hotkernels/issues/8)). NumPy is 4.5x behind naive C here, and `sum_u8` is ahead of the C |
| 9 | `x[mask]` and `x[mask] = v` on int64 | `compress_f64`, `maskfill_f64` | Neither kernel reads an element as a number on either architecture; they move 64-bit values. A spec for int64 would confirm it, and the binding would accept the type. This is the commonest mask selection in the program's own code |
| 10 | `x[mask] = v` on a two-dimensional array with a mask of the same shape | `maskfill_f64` | The indexing hook asks for one dimension. `np.putmask` already takes any shape through the same kernel |

`x[mask] = v` on float32 would need a kernel of its own, `maskfill_f32`. It is not proposed until a run shows it on long arrays.

### Calls on short arrays

Where the gain is the call path. Each is a function or method hook that calls a plain C loop before NumPy's wrapper runs, as the hooks for `np.clip`, `x.argmax()` and `x.argmin()` do ([issue 17](https://github.com/sofeikov/hotkernels/issues/17)). None is expected to need assembly, and none is faster than NumPy on long arrays.

| # | Call | NumPy per call at 32 elements | Naive C | At 131,072 elements |
|---|---|---|---|---|
| 11 | `np.median(x)`, one-dimensional float64 | 4.8 µs | 58 ns | NumPy is ahead of a naive selection, 0.67x. The hook would take short arrays only |
| 12 | `np.clip(x, lo, hi)` on int64 | 1.75 µs | 6 ns | 1.4x behind, level at 2,097,152 |
| 13 | `np.round(x, decimals)` on float64 | 1.1 µs | 12 ns | 1.7x behind: NumPy makes three passes |

`np.diff` (0.9 µs against 6 ns) and `np.nanmax` and `np.nanmin` (1.5 µs against 45 ns) have the same shape and fewer calls. They wait for a run that shows them.

### Not targets

Measured, and level with naive C or ahead of it at size:

- **`np.isnan`** and **`np.nanmax`**: NumPy has vector code for both. `np.nanmax` is 17x ahead of the naive loop at 131,072 elements.
- **`a[idx]` with an integer array:** level at 2,097,152 elements, 1.5x behind at 131,072. A loop of AVX2 gathers was no faster than the scalar loop in the caches and slower beyond them.
- **`np.diff`, `np.repeat`, `mask.astype(float)`:** level from 131,072 elements up.
- **`np.argsort(x, kind="stable")` on a few distinct small integers:** a counting sort is 2.9x to 3.3x ahead of NumPy's radix sort, but it is a different algorithm for a special case, the program calls it once per run, and the scatter does not vectorise. Not proposed.
- **`np.searchsorted` on int64 and float32:** 6% of the program's searches. NumPy's int64 search was 5x ahead of naive C in the first survey.
- **Code compiled with numba.** It does not call NumPy at run time.

## Order of work

Provisional. A recorded real run ([decision 3](#decisions-still-open)) may reorder it or remove rows.

| Step | Candidates | Why here |
|---|---|---|
| 1 | The candidates of rows 1 to 7 added to the survey and recorded on both architectures | The figures above are from a script outside the repository, and from one architecture. The survey is where a candidate's gap is recorded |
| 2 | Rows 8 to 10: `mask.sum()`, int64 mask selection and assignment, two-dimensional `x[mask] = v` | No new assembly, and the kernels exist on both architectures |
| 3 | Row 1: `cumsum_bool` | Largest gap that assembly can add to; the hook exists; exact |
| 4 | Row 4: `where_f64` | Most call sites of any candidate; the function hook exists |
| 5 | Rows 11 to 13: the short-array hooks | Most of the recorded time, in C only. After step 2 because they add hooks, not conditions |
| 6 | Row 2: the running maxima, and row 6: `reduceat` | Both wait for decision 2 |
| 7 | Row 5: reductions over short rows | A new shape for the judge: a kernel over rows |
| 8 | Row 7: `bincount` on int16 and int64 | Technique known; one caller per run |
| 9 | Row 3: `hypot_f64` | Waits for decision 1 |

## How the work is tracked

A family's kernels are written twice, once per architecture, on different machines and at different times. The spec and the naive C are written once, and so is the binding.

- **One issue per family,** holding what is shared: the NumPy call, the conditions, the figures from this document, and any decision.
- **Sub-issues under it,** each closed by one pull request:

  | Sub-issue | Label | Waits for |
  |---|---|---|
  | `spec.py` and `naive.c`, and the survey rows | `spec` | |
  | The x86-64 kernel | `arch:x86-64` | The spec merged, since the judge locks it against `main` |
  | The ARM64 kernel | `arch:arm64` | The spec merged |
  | The binding, `delivery check` cases and the user-facing pages | `binding` | One architecture's kernel |

- **The two architecture sub-issues are independent.** The package already installs a different number of kernels per architecture, so one can be released before the other. Each pull request leaves its own release note.
- **Rows 8 to 13 are one issue each,** with no architecture sub-issues: the binding and a C loop are the same on both.
- **A milestone holds the round together** and gives its order.

The labels `spec`, `binding`, `arch:x86-64` and `arch:arm64` do not exist yet. The issues are opened when this document is merged.

## Decisions still open

1. **`np.hypot`: bit-identical or not.** NumPy's result is the C library's `hypot`, which differs between glibc and macOS and may differ between versions of each. Three options: reproduce each library's algorithm in the kernel and check it against that library; ship the square-root form under `install(inexact=True)`, as float `cumsum` is, with a stated bound and its overflow range left to NumPy; or not build it. Proposed: decide after a real run shows how much time `np.hypot` takes.
2. **A hook for `accumulate` and `reduceat`.** *Settled with row 2: the ufunc type's method is hooked, by the method hook; see [What changed in the building](#what-changed-in-the-building).* Both are methods of the ufunc object, not of the array. Either the inner loop of `np.maximum` and `np.fmax` is replaced and recognises the pattern of an accumulation, as the `clip` loop recognises scalar bounds, or the ufunc type's method is hooked, which is a fifth kind of hook. The first reaches every caller; whether an inner loop can tell an accumulation from a reduction by its arguments has not been checked.
3. **A recorded real run.** The same recording over one real run of the program gives each call's share at real sizes and settles the order. It needs the program's production data.
4. **Whether short-array hooks belong here.** *Settled with row 12: they do, as plain C loops inside the binding for short arrays only, not as kernels; see [What changed in the building](#what-changed-in-the-building).* Rows 11 to 13 are C loops behind a hook, and their gain is NumPy's wrapper. `np.clip` on floats set the precedent. `np.median` goes further: it is a Python function in NumPy, and the hook replaces all of it for the cases it takes.
5. **ARM64.** Nothing here was measured there. Step 1 of the order does it. A candidate that is level with naive C on ARM64, as `np.where` was in the first survey, is built for x86-64 only.
6. **Recording as a feature.** The call recording answers the open question of [issue 89](https://github.com/sofeikov/hotkernels/issues/89): `sys.monitoring` sees every call from chosen code into NumPy at about 0.26 µs each, and does not see operators or subscripts. Whether the scripts become `hotkernels.profile()` is that issue's to decide.

## Out of scope

- Any change to the program the calls come from.
- Fusing the calls of its inner loop into one. [core-design.md](core-design.md#later) lists fusion without a plan; this loop is an example of a chain worth fusing.
- Kernels for strided arrays and for row selection of a two-dimensional array, which the program also does.

## Documents that change

| Document | Change |
|---|---|
| `docs/survey-design.md` | A pointer to this document |

When the candidates are built, each pull request updates the documents its kernel or hook makes untrue, as [AGENTS.md](../AGENTS.md#documentation-moves-with-the-change) lists them. This document changes nothing a user of the package gets, so it leaves no release note.
