# The ufunc lane: elementwise calls on small arrays through NumPy's own loops

Proposed, and now built: `hotkernels/src/hotkernels/_binding/hook_ufunc.c`. [Status](#status) holds the result and [What changed in the building](#what-changed-in-the-building) says where the building departed from the proposal; the rest of the text stands as proposed. Every figure is from an Apple M1 Pro, CPython 3.13.12 and NumPy 2.5.5, taken from Python with the two sides alternating and the best of seven or nine batches, while other work ran on the machine (load average 3 to 6). Read the ratios more than the nanoseconds.

## Status

Against the [exit criterion](#exit-criterion):

| Criterion | Result |
|---|---|
| `delivery check` passes, with the `ufunc` operation | Met on ARM64: 403,021 cases, none failing. The `ufunc` operation has 100,657, of which the lane must take 49,922 and took 51,766 (the rest of what it took are cases where only the results are compared). x86-64: CI |
| NumPy's own test suite has the same outcome with and without `install()` | Met: 47,248 tests, none differing, after the change in [What changed in the building](#what-changed-in-the-building) for replaced loops |
| `a + b` at least 1.8x and `a * 2.0` at least 2.5x faster at 16 float64 elements | Met: 1.98x and 2.58x in `delivery time`; 1.92x and 2.87x in alternating batches of `timeit` |
| No case of `delivery time` the lane serves is slower than NumPy | Met. An earlier run, at load average 5.2, read `np.exp(a)` at 4,096 elements as 0.96x; the run below has it at 1.00x, and alternating batches put it level (14.97 against 14.74 µs best, 15.50 against 15.64 µs median) |
| A call that falls back costs at most 2 ns more | Not met: 3 to 4 ns, about 1.3% of the cheapest such call. 1.3 ns of it is the count of calls left to NumPy, which was kept ([below](#what-changed-in-the-building)) |

`uv run delivery time --only ufunc`, load average 4.05. At a million elements every statement is left to NumPy, and the column says how much that costs:

| Statement | Elements | 16 | 4,096 | 1,048,576, left to NumPy |
|---|---|---|---|---|
| `a + b` | float64 | 246 → 124 ns, 1.98x | 807 → 686 ns, 1.18x | -0.2% |
| `a * 2.0` | float64 | 419 → 163 ns, 2.58x | 890 → 628 ns, 1.42x | +0.1% |
| `a > 0` | float64 | 531 → 166 ns, 3.20x | 1.95 → 1.58 µs, 1.24x | -0.0% |
| `np.exp(a)` | float64 | 253 → 158 ns, 1.60x | 16.1 → 16.0 µs, 1.00x | +0.0% |
| `a * 2.0` | float32 | 518 → 162 ns, 3.20x | 798 → 432 ns, 1.85x | +0.4% |
| `i + 1` | int64 | 500 → 163 ns, 3.06x | 910 → 566 ns, 1.61x | +0.0% |
| `np.maximum(x, y, out=x)` | float64 | 236 → 99 ns, 2.39x | 911 → 772 ns, 1.18x | +0.1% |
| `m - row`, 4 columns | float64 | 554 → 148 ns, 3.75x | 6.83 → 4.55 µs, 1.50x | +0.1% |
| `a + b`, strided, declined | float64 | +1.3% | +0.4% | +2.0% |
| `a + b`, declined | float16 | +1.1% | +0.1% | -0.0% |

From Python in alternating batches at 16 float64 elements: `a + b` 250 → 130 ns (1.92x), `a - b` 1.91x, `a > b` 1.90x, `-a` 1.85x, `a * 2.0` 406 → 141 ns (2.87x), `a > 0` 509 → 145 ns (3.51x), `a * np.float64(2)` 2.63x, `np.add(x, b, out=x)` 212 → 72 ns (2.97x), and `M - w` on a 100 x 12 matrix 1,076 → 573 ns (1.88x). `np.isclose(a, b)`, a Python function of several ufunc calls, is 1.25x faster without a hook of its own. Calls the lane declines: a strided `a + b` +3.4 ns, float16 +4.4 ns, `np.add(a, b, where=True)` +3.4 ns, complex `a + 1j` within noise.

x86-64 was not measured: there is no such machine here. The lane is the same C there, and the call path it skips is NumPy's on both.

## What changed in the building

- **Floating-point errors go to NumPy's own handler, and nothing is computed twice.** Proposed: throw the result away and let NumPy compute the call again, which needed the inputs untouched and so a buffer for a given output. Built: after the loop the lane reads the four flags NumPy reports and, if any is set, clears them and calls `PyUFunc_GiveFloatingpointErrors(name, flags)`, the function of NumPy's C API its own ufuncs call after their loops. It warns, raises, calls or does nothing as `np.errstate` says, with NumPy's message. The loop writes straight into a given output, `x += y` included, as NumPy's does, and an output keeps what the loop wrote when the call raises, as NumPy's does. A Python exception the loop sets, such as an integer power's `ValueError` for a negative exponent, is raised as it is. Under `np.errstate(all="ignore")` NumPy leaves the flags set after the call and the lane clears them; nothing in Python can tell. A call that raises is not counted as taken.
- **A row against a matrix is one loop call over a tiled row, not one call per row.** Spying on NumPy's loop (a replacement of the float64 `add` loop that records each call's count and strides) showed that NumPy's buffered iterator copies the row, repeated, into its buffer and calls the loop over chunks of whole rows: one call of 72 elements for 12 rows of 6, two calls of 8,190 and 3,810 for 2,000 rows of 6, one call per row only for rows longer than the buffer. The lane does the same: it repeats the row into a buffer (on the stack up to 2 KB) and makes one call, and it takes such a call only when NumPy too makes one, that is when the result has at most as many elements as the whole rows that fit in NumPy's default buffer of 8,192. With one call, an error stops both at the same element: per row, the lane went on to the rows after one that raised, where NumPy stops. A row of one element is left to NumPy, whose iterator drops the dimension of one. `np.setbufsize` changes NumPy's chunks and not the lane's; an elementwise result does not depend on them.
- **`np.maximum` and `np.minimum` with an output given positionally go to NumPy**, which deprecates that spelling from 2.5 and warns. The warning was the only difference the first sweep found once the rest was right.
- **The table evicts.** Proposed: a full table sends new combinations to NumPy. `delivery check` alone made more than 1,024 combinations and the lane stopped serving the later ones. Built: 2,048 entries, and a combination that finds no room within eight slots takes the first one's place. A call already running holds the capsule of its loop, so evicting its entry frees nothing it uses.
- **A replaced loop is noticed.** NumPy's own suite has a test that calls `np.negative`, replaces its float64 loop with `PyUFunc_ReplaceLoopBySignature`, and expects the next call to use the new loop. The lane had kept the loop it was given and failed it, the one difference in NumPy's 47,248 tests. Built: when NumPy's loop wraps a legacy loop, the entry also keeps which entry of the ufunc's type table that is and the function it held; every call compares that function with the one the table holds now, one load and a comparison, and asks NumPy again if it changed. That also covers hotkernels' own inner-loop hooks, which replace loops the same way.
- **The ufuncs are found in the module's dictionary**, not through `dir()` and `getattr`: NumPy imports a submodule on first access to its name, and `install()` should not.
- **The self-test's flags probe checks the error, not the counter.** `a / 0.0` under `np.errstate(divide="raise")` must raise `FloatingPointError`, which shows the flag reached NumPy's handler, and give infinities under `divide="ignore"`. A plain call after them moves the counter.
- **A call that falls back costs 3 to 4 ns, not 2.** The decline path tests an exact `ndarray` first and settles it by its own checks, and looks at keywords and outputs only when there are any. What is left is the call through the lane and the count of calls left to NumPy, 1.3 ns of the total. The count was kept: it is what `report()` shows of the calls the lane did not serve, and it is what told shap's calls apart in [What it does for whole programs](#what-it-does-for-whole-programs).
- **`report()` shows the lane as NumPy's.** Its row is "elementwise ufuncs", element type "numeric", and "NumPy" in the Kernel column, not "C", because the loops are NumPy's; a line under the table says what its columns count.
- **`delivery check` builds a case's arguments before it reads the counters.** The judge's value classes make ufunc calls of their own, which the lane counts. Building the arguments first, then reading the counters, then making the call, leaves every other operation as it was.
- **NumPy can be nondeterministic itself.** `np.power` on int64 with a negative exponent into a float32 output gave two different outputs over six runs of plain NumPy: after the loop raises, its buffered cast copies what was left in its buffer. Those cases are left out of `delivery check`.

## Summary

- **Most of a small NumPy call is the call, not the arithmetic.** `a + b` on two float64 arrays of 16 elements takes 255 ns, and the loop that adds them takes a few. `a * 2.0` takes 410 ns and `a > 0` 516 ns, because NumPy converts the Python number on every call. hotkernels takes this cost out of the calls it hooks (`np.sum(x)` on 16 elements: 867 ns to 46 ns). It does nothing for the elementwise operators and functions, `+`, `*`, `>`, `np.exp`, `np.maximum` and the rest, which are most of the NumPy calls a numerical program makes.
- **Proposed: a lane through NumPy's own loops.** One hook on the call pointer of each of NumPy's ufunc objects, as the function hook does for `np.argmax`. A call whose operands are plain contiguous arrays of one shape, or such an array and a number, takes a short path. It allocates the output, or uses the `out` it was given, and calls the inner loop NumPy itself would have called. Anything else goes to NumPy unchanged.
- **The loop is NumPy's, so the bytes are NumPy's.** The lane asks NumPy once per operand types which loop it would use, through the interface NumPy provides for this to other compilers (`ufunc._resolve_dtypes_and_context` and `ufunc._get_strided_loop`, the route numba takes). It then calls that loop with the same element count and strides NumPy would pass. No arithmetic in the binding, no kernel, and nothing for the judge.
- **What it gives, from a throwaway prototype.** About 2x on `a + b` at 16 float64 elements, 3x on `a * 2.0` and 3.5x on `a > 0`; 1.7x at 1,024 elements; nothing measurable above about 16,384. A sweep of 5,317 calls over every one- and two-operand NumPy ufunc, seven element types, Python and NumPy numbers on either side and special values gave the same bytes, types, warnings and errors with and without it.
- **What it cannot give.** The lane's own floor is about 125 ns from Python on 16 elements: about 20 ns for Python to reach the ufunc at all, and 83 ns for NumPy to allocate and free the result. A whole program gains only as much as its NumPy calls cost. In shap's explainers NumPy is a quarter to a third of the time, and the prototype made them 2% to 7% faster ([What it does for whole programs](#what-it-does-for-whole-programs)).

## What a small call costs

From Python, 16 float64 elements unless the line says otherwise. "hotkernels today" is with `install()` on `main`.

| Statement | NumPy | hotkernels today |
|---|---|---|
| `len(a)`: the cost of any call from Python | 23 ns | 20 ns |
| `np.empty(16)` | 126 ns | 123 ns |
| `a.copy()` | 162 ns | 138 ns |
| `a + b`, `a - b`, `a * b`, `a / b` | 242 to 255 ns | the same |
| `np.add(a, b)` | 261 ns | the same |
| `a * 2.0` | 401 ns | the same |
| `a > 0` | 516 ns | the same |
| `np.exp(a)` | 266 ns | the same |
| `np.abs(a)` | 236 ns | the same |
| `np.maximum(a, b)` | 265 ns | the same |
| `np.sum(a)` | 867 ns | 46 ns |
| `np.argmax(a)` | 609 ns | 47 ns |

The last two rows are what a hook on the call itself does: it skips NumPy's Python wrapper and its argument handling and goes straight to a kernel. The rows above them reach no hook.

Inside a C extension, one allocation and release of a 16-element float64 array through `PyArray_NewFromDescr` takes 83 ns. Sampling that loop shows where the time goes: the memory handler is a context variable read on every allocation; it is a capsule whose name is compared with `strcmp` on every allocation and release; and every allocation and release is reported to `tracemalloc`, which takes the GIL state through thread-local storage. The context variable and the GIL state are thread-local lookups, which cost a call into the dynamic linker on macOS (`_tlv_get_addr`).

## What was verified

A C extension of about 300 lines, in the session's scratch space and not in the repository, applied the lane to every object of type `np.ufunc` in the `numpy` namespace.

- **Speed.** From Python, plain NumPy against the prototype, float64 unless stated:

| Statement | 16 elements | 256 | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---|---|---|---|---|---|
| `a + b` | 264 → 127 ns, 2.1x | 1.8x | 1.7x | 1.3x | 1.05x | 1.02x |
| `a * 2.0` | 410 → 138 ns, 3.0x | 2.5x | 2.2x | 1.6x | 1.35x | 1.03x |
| `np.exp(a)` | 268 → 168 ns, 1.6x | 1.1x | 1.05x | 1.0x | 1.0x | 1.0x |

At 16 elements: `a > 0` 3.4x, `a > b` 1.9x, `-a` 1.9x, `np.abs(a)` 1.8x, `np.maximum(a, b)` 1.7x, `np.isfinite(a)` 1.7x, `f * 2.0` on float32 3.8x, `i + 1` on int64 3.6x, `c & c` on bool 2.0x.

The saving is a fixed 130 to 300 ns a call. The prototype tested the floating-point flags with the C library's `fetestexcept` and `feclearexcept`, 15 ns of its time; the binding reads the status register directly in about 1 ns (`hk_float_flags_reported`), so the built lane should be a little faster than this. It is about the same: 124 to 130 ns for `a + b`, so the flags were not where the time went.

- **Exactness.** Every NumPy ufunc with one or two inputs and one output, applied to arrays of float64, float32, int64, int32, uint8 and bool and to a float64 array of zeros of both signs, infinities, NaN, the largest and smallest numbers; each array with each other array of its length, and with each of 2.0, -0.5, 3, 0, -7, 1e300 and 2⁶⁰ on either side; under `np.errstate(all="warn")`. The result's bytes and type, the warnings and the exception, if any, compared with plain NumPy: 5,317 calls, no difference. The first version of the prototype differed on 11: integer `power` with a negative exponent, whose loop raises `ValueError` without returning an error code. Testing for a Python exception after every loop fixed them, and is part of the design.
- **shap's test suite** passed the same way with and without `install()` (385 passed, 156 skipped). That run was hotkernels alone, without the prototype. The suite was not run under the prototype; `delivery check` and NumPy's own test suite are the tests of the built lane ([Testing](#testing)).

## What it does for whole programs

shap was the occasion for this. Its explainers are NumPy-heavy in the sense that they make hundreds of thousands of NumPy calls on arrays of a few dozen to a few hundred elements, and hotkernels as it is made no measurable difference to them (within ±3%). The table below is shap's master branch on 2026-11-23, its explainers on scikit-learn models with 12 features.

| Explainer | NumPy | hotkernels | hotkernels and the prototype lane |
|---|---|---|---|
| Kernel | 171.1 ms | 1.03x | 1.075x |
| Permutation | 412.6 ms | 1.00x | 1.034x |
| Partition | 1,470.7 ms | 1.01x | 1.031x |
| Sampling | 202.8 ms | 1.01x | 1.021x |
| Exact, Linear, Tree | | 1.00x | 0.99x to 1.01x |

With the lane as built, which also takes outputs, NumPy numbers and rows, in a later run (load average about 3): Kernel 1.097x, Sampling 1.089x, Linear 1.072x, Permutation 1.069x, Partition 1.047x, Exact 0.991x, Tree 1.000x, against plain NumPy, every result the same bytes as NumPy's. shap's test suite gave the same outcomes (385 passed, 156 skipped), the lane serving 1,313,993 of its ufunc calls and leaving 484,893 to NumPy.

The gain is small because the most it could be is small. Sampling with macOS's `sample`, classified as `delivery apps --profile` does: in the Partition explainer the Python interpreter has 54% of the time, NumPy 24% and memory management 17%; in the Permutation explainer 36%, 30% and 18%. The Kernel explainer spends 62% waiting in OpenMP threads that scikit-learn's k-means starts. If every NumPy call cost nothing, Partition would be at most 1.3x faster. The interpreter's share is shap's own Python, the scipy sparse matrices it builds and the models' predict functions.

The lane took 40% to 57% of the ufunc calls in Partition, Permutation and Kernel, and 10% in Sampling. The rest fell back for these reasons:

| Reason | Partition | Permutation | Sampling |
|---|---|---|---|
| An output given positionally (`x += y`, `np.add(a, b, out)`) | 41,824 | 48,000 | 811 |
| A keyword (nearly always `out=`) | 24,799 | 0 | 1,830 |
| A NumPy number on one side (`x - x.mean()`) | 3,205 | 2,099 | 31,291 |
| An array against a row of it (`(m, n)` with `(n,)`) | 59,402 | 6,596 | 1,471 |
| An array that is not contiguous | 97,474 | 0 | 421 |
| Operands that need a cast (`int64` with `float64`) | 28,000 | 0 | 0 |

The prototype took none of these. The design takes the first four.

## Design

### The hook

Every object of type `np.ufunc` holds its own call pointer, at the offset `PyUFunc_Type` publishes as `tp_vectorcall_offset` (the field `vectorcall` of `PyUFuncObject`). The ufunc hook points it at the binding, exactly as the function hook does for `np.argmax`'s object. `np.add(a, b)` calls it directly. `a + b` calls it too: NumPy's `ndarray.__add__` calls the `add` ufunc through `PyObject_CallFunctionObjArgs`, which goes through the call pointer. `a += b` calls `add(a, b, a)` the same way. So do comparisons, unary minus, `abs()` and every other operator NumPy implements with a ufunc. Names imported before `install()` and NumPy's own internal callers reach the hook too, because the object is not replaced.

Which objects are hooked: every exact `np.ufunc` that is an attribute of the `numpy` module, with one or two inputs, one output and no core signature (not `np.matmul`). That covers every operator of `ndarray`. Ufuncs other libraries create, such as `scipy.special.expit`, are objects of the same type and could be hooked the same way. They are left out, because hotkernels never imports another library, and a library imported after `install()` would need its ufuncs found then ([Open decisions](#open-decisions)). Every ufunc NumPy creates starts with the same call function, `ufunc_generic_vectorcall`. A ufunc whose pointer is anything else at `install()` has been changed by someone else and is left alone.

### When a call takes the lane

All of these, else the call goes to NumPy's own pointer with its arguments untouched:

- **No keyword, or only `out`.** `out=` may be an array or a one-element tuple.
- **The positional arguments are the inputs, or the inputs and one output.**
- **Each input is one of these:**
  - an array of exactly type `ndarray`, C-contiguous, aligned, in native byte order, with at least one dimension, of bool, int8 to int64, uint8 to uint64, float32 or float64;
  - a Python `float` or `int` of exactly those types (not `bool`), when the other input is an array;
  - a NumPy number of exactly one of those types (`np.float64(0.5)`), when the other input is an array.
- **The arrays fit together:**
  - they have the same shape;
  - or the smaller one's shape is the trailing part of the larger one's, as a row `(n,)` against a matrix `(m, n)`, with the larger one first or second.
- **At most 16,384 elements in the result** ([Open decisions](#open-decisions)).
- **NumPy's loop for these operand types needs no cast of an array input**, and its output type is one of the types above. `int64 + float64` needs a cast and goes to NumPy, as does any loop that needs the Python API.
- **A given output** is an exact `ndarray`, C-contiguous, aligned, writeable, native, of the result's shape and exactly the loop's output type. It is either the same memory as an input of the result's shape (`x += y`) or overlaps no input.
- **A number fits the loop's type exactly.** A Python float is taken by a float64 loop, and by a float32 loop unless it is finite and rounds to an infinity. A Python int is taken by an integer loop when it is in that type's range, and by a float loop when its magnitude is at most 2⁵³ for float64 or 2²⁴ for float32, where the conversion is exact. Anything else goes to NumPy, which raises, warns or rounds as it does.

### The loop

For each combination of ufunc, input types and layout, the lane asks NumPy once:

1. `ufunc._resolve_dtypes_and_context((type0, type1, None))` gives the types NumPy resolves the call to and a capsule. For a Python number the type passed is `float` or `int`, which NumPy treats as a number of no fixed type, as it does in a real call (NEP 50). For a NumPy number it is the number's dtype.
2. `ufunc._get_strided_loop(capsule, fixed_strides=...)` puts in the capsule the inner loop, its data and two flags: whether the loop needs the Python API, and whether it can raise floating-point exceptions. The fixed strides are the ones NumPy itself passes for that call: each array's element size, and zero for a number, which NumPy turns into a zero-dimensional array and broadcasts.

The answer is kept in a table of the binding, keyed by the ufunc object and the two operand kinds. A kind is an element type and whether it is an array, a NumPy number or a Python number. The table holds a reference to the capsule, whose lifetime NumPy ties to the loop's data. An entry that cannot be served is kept too, so the question is not asked again. The table has a fixed number of entries; when it is full, a call with a new combination goes to NumPy. `uninstall()`, and every `install()`, empties it, because hotkernels' own inner-loop hooks can change which loop NumPy returns.

On each call the lane then:

1. converts the number, if there is one, into a buffer on the stack, as the loop's type;
2. for an output to allocate: allocates it with `PyArray_NewFromDescr` with the loop's output type and the larger input's shape. This is NumPy's allocator and memory handler, as for any array;
3. calls the loop:
   - with equal shapes, once over all the elements, with each array's element size as its stride and zero for a number. That is NumPy's own trivial loop, which it uses whenever the operands are contiguous arrays of one shape or zero-dimensional;
   - for a row against a matrix, once per row, the row's stride zero between calls, which is the inner loop NumPy's iterator reaches for those operands.
4. hands the result back: the new array, or the given output with its reference count raised, which is what NumPy returns.

Interpreting `_get_strided_loop`'s capsule is the one thing here NumPy calls unstable: its layout carries the name `numpy_1.24_ufunc_call_info` and may change with a NumPy release. The binding checks the capsule's name. A NumPy whose capsule has another name leaves the hook off, and `report()` says why. The install-time self-test then covers the rest, as for every hook.

### Errors, flags and the given output

NumPy clears the floating-point status flags before a loop that can raise them and tests them after. Under `np.errstate` it then warns, raises, calls a function or does nothing. The lane does the first half and hands the second to NumPy:

- Before such a loop it clears the four flags NumPy reports (invalid, divide by zero, overflow, underflow), as NumPy does. After the loop it tests them, and tests whether the loop set a Python exception or returned an error. If any of these happened, it throws away what it computed and calls NumPy's own pointer with the original arguments. NumPy computes again, with the same loop on the same inputs, and warns, raises or ignores as the error state says. The result is NumPy's in every respect, at the price of computing twice the calls that raise a flag.
- That needs the inputs unchanged when NumPy computes again, and with `x += y` the output is an input. So a call with a given output computes into a buffer of the binding first: 2 KB on the stack (`HK_STACK_SCRATCH_BYTES`), or one allocation for a larger result. It copies the buffer into the output only once the loop has finished without a flag or an error. Elementwise, each output element depends only on the same elements of the inputs, so where it is computed does not change its value.
- The flags NumPy does not report, inexact above all, are left as the loop left them, which is what NumPy does.
- Under `np.errstate(all="ignore")` NumPy does not test the flags at all. The lane does not read the error state, so a call there that raises a flag is computed twice and returns NumPy's result.

### The interpreter lock

NumPy releases the GIL for a loop over more than 500 elements that does not need the Python API (`NPY_BEGIN_THREADS_THRESHOLDED`). The lane does the same at the same count. Below it, both hold the lock throughout.

### Counting and the report

One counter of calls the lane served, shown in `report()` as one row: the NumPy call is "elementwise ufuncs", for example `a + b` or `np.exp(a)`, the element type is "numeric" and the kernel is "C". It is not counted among the kernels, as the loops built into the binding are not. Its "Left to NumPy" column counts the calls that reached the hook and went to NumPy. That is every ufunc call in the process that the lane did not serve, and it costs a load and a store, as every other hook's count does.

### The self-test

At `install()`, after the hook is applied, a set of probes runs, each of which must give the right answer and move the lane's counter by exactly one:

- `a + b` on float64;
- `a * 2.0` on float32, so a Python number is converted for a float32 loop;
- `i > 3` on int64;
- `np.subtract(m, row)` with a matrix and a row;
- `x += y` on float64, which must leave `x` holding the sum;
- `a / 0.0`, which raises the divide-by-zero flag. It must leave the counter where it was and return NumPy's infinities.

If any fails, the hook is removed and `report()` says which probe failed.

### Testing

- **`delivery check`** gets an operation of its own, `ufunc`. Its cases:
  - every hooked ufunc, over each element type of the lane and float16 and complex128, which it must decline;
  - the values of the judge's value classes;
  - sizes on both sides of 500 and of 16,384;
  - each operand as an array, a Python number, a NumPy number, a zero-dimensional array and an array subclass;
  - equal shapes, a row against a matrix and other broadcasts;
  - contiguous, strided, reversed and Fortran-ordered arrays;
  - outputs: none, `out=` the input, `out=` another array, `out=` an overlapping view, `out=` of the wrong type;
  - each under the default error state and under `np.errstate(all="raise")`.

  Each case says whether the lane must take it. Results, types, warnings, exceptions and the contents of a given output must match NumPy's.
- **NumPy's own test suite** (`uv run delivery numpy-tests`) must give the same outcomes with and without `install()`. Nothing tests the operators harder than NumPy's own tests do.
- **`delivery time`** gets cases for `a + b`, `a * 2.0`, `a > 0`, `np.exp(a)` and `x += y` at 16, 1,024 and 16,384 elements, and one that falls back (a strided array), to show it costs nothing measurable.

## Considered and not proposed

- **A faster allocation.** 83 ns of the lane's 125 are NumPy creating and destroying the output. The binding could allocate the data itself and hand NumPy an array that owns it, skipping the context variable, the capsule and `tracemalloc`. It would bypass a memory handler a user has set, and the output would be invisible to `tracemalloc`. NumPy could look the handler up more cheaply itself; that is a change for NumPy, not for this package.
- **Recycling outputs.** Keeping the arrays the lane creates on a free list when Python releases them would mean replacing `ndarray`'s deallocator and bringing objects back after their release. It is not done.
- **The binding's own loops.** Writing the arithmetic in the binding, or calling the kernels, would make the lane responsible for matching every NumPy loop bit for bit, transcendental functions included. NumPy's loops are already there and are exactly right by definition.
- **Calls that need a cast, and arrays that are not contiguous.** NumPy runs these through buffers and an iterator whose chunking would have to be reproduced exactly, and the loop it chooses for a stride it does not know in advance can differ from the one for a fixed stride. They go to NumPy.
- **Deferring and fusing calls.** The lazy layer ([PR 261](https://github.com/sofeikov/hotkernels/pull/261)) removes the per-call cost differently: it builds a graph of calls and runs them together. The lane makes the calls a program makes now cheaper, without changing what it means to call NumPy. They do not depend on each other.
- **The Python wrappers.** `np.mean(a)` costs 1.4 µs, `a.std()` 4.5 µs and `np.isclose(a, b)` 8.0 µs at 16 elements, because they are Python functions that make several ufunc calls each. The lane makes those inner calls cheaper. A hook for each wrapper, as for `np.sum`, would remove the wrapper itself. That is a later proposal of its own: one function at a time, each with its own exactness argument.

## Order of work

1. The hook, the table and the loop call, for equal shapes and numbers, with no given output.
2. The self-test probes and the counter in `report()`.
3. The `ufunc` operation of `delivery check`, then NumPy's own test suite.
4. A given output, then a row against a matrix, each with its cases.
5. `delivery time` cases, the figures in the README and `docs/kernel-findings.md`, the user pages and the release note.
6. x86-64: the same code. `delivery check` and the timings on the Intel machine.

## Exit criterion

- `delivery check` passes on both architectures, with the `ufunc` operation's cases.
- NumPy's own test suite has the same outcome with and without `install()`.
- From Python at 16 float64 elements, on the Apple M1 Pro: `a + b` at least 1.8x and `a * 2.0` at least 2.5x faster than NumPy. No case of `delivery time` that the lane serves is slower than NumPy at any size, and a call that falls back costs at most 2 ns more (built: 3 to 4 ns; see [What changed in the building](#what-changed-in-the-building)).

## Open decisions

1. **The largest result the lane serves.** Proposed 16,384 elements: above it the measured gain is under 5% for every statement measured. Holding the lane to 500, below NumPy's GIL threshold, would leave the lock out of it, at the cost of the 1.3x to 1.7x measured from 1,024 to 4,096 elements.
2. **Other libraries' ufuncs.** `scipy.special` makes ufuncs scikit-learn calls on small arrays, `expit` among them. Hooking them means finding them after the library is imported, as the bottleneck hooks are found, or walking every live object of type `np.ufunc` at `install()`.
3. **More element types.** float16, complex64 and complex128 have loops like the others. Leaving them out keeps the number conversions to the types where they are certain to be exact.
