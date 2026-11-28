# Fusion by pattern in the lazy layer

This is fusion in the prototype of the lazy layer ([lazy-design.md](lazy-design.md), [`prototypes/lazy/`](../prototypes/lazy)), and what it measured. Nothing here is in the package. The layer writes a program's NumPy operations down as a graph; this document is about computing that graph faster than one NumPy call per operation, with NumPy's bits.

The approach is the plainest one there is. A **pattern** is a small tree of operations with named inputs, such as `a + s * b`. Where the graph holds that tree, a kernel written for it computes it in one pass over memory. A longer expression is covered with several patterns, and their kernels are called block by block, so that what one gives the next never leaves the cache. An operation no kernel is written for is computed as before, so nothing is lost by a pattern that is missing.

## Summary

- **A fused kernel gives NumPy's bits, and the judge checks it.** Every operation in one is a single IEEE operation that NumPy makes with the same instruction, in the program's order, each rounded by itself. No multiply-add is fused and no sum is regrouped. There are eighty kernels, in NEON assembly for ARM64 (on x86-64 two are AVX2 assembly and the rest their compiled C, which is as fast there: [Another processor](#another-processor)), and each passes the judge against its NumPy expression on 20,000 to 370,000 cases. See [Why the bits are NumPy's](#why-the-bits-are-numpys).
- **The pattern is the kernel's own oracle.** A kernel's `spec.py` names the NumPy expression it must equal. Run on lazy arrays, that expression writes a graph, and the shape of that graph is the pattern. Nothing else says what a kernel serves. See [The library](#the-library).
- **A whole expression is one pass over memory, whatever its length.** Its kernel calls are run a block of 4,096 elements at a time, so what one kernel gives the next stays in the cache. For that every operation has a kernel, the single operations too. See [An expression block by block](#an-expression-block-by-block).
- **Most kernels are written by a program from the name of their shape.** Fourteen were written by hand; `make_kernel.py` wrote sixty-six, each with its spec and its C reference, and the judge holds them to the same bits. See [The kernels](#the-kernels).
- **Which shapes to write was mined, not guessed.** Twenty-six programs were run lazily with every call timed: every program this repository times with and without `hotkernels.install()`, and six more. See [Finding the shapes](#finding-the-shapes).
- **A write is made where the array is.** A deferred write is a copy only when something still reads what the array held, and with fusion an expression assigned to part of an array is computed straight into the part. See [Writes](#writes).
- **What cannot be fused is not written down.** With fusion on, a call that gives fewer than 32,768 elements, or that no kernel could make, on arrays that have their values, is NumPy's call at that line, not a node, and its result is one object that holds NumPy's value. See [As the graph is built](#as-the-graph-is-built).
- **The C core reads the graph as well as writing it.** The expressions of a computation are read off its nodes in C, by the functions of the Python reader one for one. See [Planning](#planning).
- **Rows of a few elements are a kernel's as well.** NumPy's loop is called once for each row of an array it cannot walk as one run, which for rows of two or four elements costs several times their arithmetic. A kernel walks such rows as one run or straight through what lies between them, and one operation by itself is then worth a kernel. See [Computing a group](#computing-a-group).
- **The layer runs as the package is used: with `hotkernels.install()` in effect.** `fuse.enable()` installs the package's single-operation kernels unless told not to, and the figure the layer is judged by is the fused run with them against the program on plain NumPy; beside it stands the eager program with the same kernels, which is what the package gives with no graph. See [The programs, four ways](#the-programs-four-ways).
- **All twenty-six programs then run within 1% of NumPy's speed or faster, up to 5.1x.** Where a program is expressions on large arrays the gain is fusion's: a cavity flow 2.80x, a weather model's horizontal diffusion 2.36x, an ocean model's equation of state 1.74x, a particle filter 1.43x, isoneutral mixing 1.15x, turbulent kinetic energy 1.13x, a regression 1.10x. Where it is a library's work or single calls on large arrays the gain is the installed kernels', up to 5.1x, which the lazy run keeps to within 4%.
- **Against the eager program with the same kernels the fused run is ahead on seven programs,** by 1.10x to 2.79x, within 3% on fifteen, and behind on four: three on small arrays, at 0.73x, 0.87x and 0.91x, where a call made at once costs 26 to 135 ns over NumPy's own, and one at 0.96x. See [What the measurements show](#what-the-measurements-show).
- **A kernel the lazy run does not reach is a loss, and one was found.** A plain `ndarray` indexed by a lazy mask is not a call the layer makes, so the package's kernel for `x[mask]` was never asked: the second Mandelbrot program ran 16% behind the eager program with the kernels installed, and runs 1% behind with the arrays `np.mgrid` makes lazy.
- **What is left is measured, program by program, and what could take it was tried.** A tool splits each fused run into the time in its kernels, the calls NumPy still makes that a kernel could make, and the rest. What a fusion of everything not yet fused could take off, at no cost of its own, is under half a percent of the run for twenty programs and 6% for the two it is largest in. A loop written for each thing that is left says what a kernel for it could do at best, and the two that showed a gain are built. The four programs behind the eager one write no graph, and writing their graphs alone takes longer than the programs do. See [What is left](#what-is-left).
- **Twenty more programs were run, those of issue 298, and they found five things in the layer, none a missing pattern.** Two gain as the corpus's stencils do, 2.31x and 1.84x. The rest are on small arrays, where a write into an array the program was handed took 3.9 µs, a recurrence over rows copied its array for every row, and a small call waited behind a large one: one filter took 11.6 s for NumPy's 37 ms and takes 41 ms. See [The programs of issue 298](#the-programs-of-issue-298).
- **On x86-64 the programs gain as on ARM64, with the kernels' C.** The AVX2 loop the tool writes there is level with clang's vectorisation of the C references for all but two kernels, so those two are assembly and the rest C; NumPy's maximum and its NaNs differ from ARM64's in ways the judge found. See [Another processor](#another-processor).
- **What is parked is in four issues:** whether the limits set on the M1 Pro should move on x86-64 ([309](https://github.com/sofeikov/hotkernels/issues/309)), what a call on a small array costs ([310](https://github.com/sofeikov/hotkernels/issues/310)), the size at which a short expression is worth fusing ([311](https://github.com/sofeikov/hotkernels/issues/311)), and the graph page showing what was fused ([312](https://github.com/sofeikov/hotkernels/issues/312)).

## What is fused

### An expression

An **expression** is a tree of element-by-element operations on float64 arrays in which every intermediate result is read once, by the operation above it, and by nothing else.

```python
w = w - rate * gradient          # two operations, one intermediate: `rate * gradient`
lap = 4.0 * u[1:-1, 1:-1] - (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])    # five, four intermediates
```

NumPy computes the second line in five passes over memory and allocates four temporaries. A kernel written for the shape of that tree reads five arrays and writes one, once.

| What it may contain | Why |
|---|---|
| `+`, `-`, `*`, `/`, `np.maximum`, `np.minimum`, unary `-`, `abs`, `np.sqrt`, `np.square`, `x ** 2` | Each is one IEEE operation, and NumPy's loop for it executes the one instruction a kernel would. `x ** 2`, `np.square(x)` and `x * x` are all one multiplication, and so is `x ** np.int64(2)` |
| `np.radians`, `np.degrees`, `np.deg2rad`, `np.rad2deg` | NumPy's loop for each is one multiplication by a constant folded in double, `x * (NPY_PI / 180.0)` or `x * (180.0 / NPY_PI)`, and `np.pi / 180.0` is that constant. Each is read as `mul` of its operand and the constant (`fuse.SCALED`); it had NumPy's bits on 100 million values of every class ([lazy-inexact-design.md](lazy-inexact-design.md#npradians-and-npdegrees-one-multiplication-built)) |
| A comparison, as the condition of `np.where(c, a, b)` | The comparison gives a mask, which a kernel keeps in a register and never writes to memory |
| A sum or a mean of every element, at the top | NumPy's order of addition can be reproduced; see [Reductions](#reductions) |

| What ends an expression | Why |
|---|---|
| `np.exp`, `np.log`, `np.tanh`, `np.arctan2`, `x ** 3`, any power but 2, `%` | NumPy calls the platform's mathematics library for these, one element at a time. A kernel is a leaf function and calls nothing, and a second implementation would not have the library's bits. Whether the layer should have a kind that is not bit-identical for them was measured and is the user's decision; [lazy-inexact-design.md](lazy-inexact-design.md) recommends not, and keeps this rule. `x % 1.0` has the bits of `x - np.floor(x)`, but the kernels have no `floor` |
| An array of any element type but float64, a NumPy scalar that is neither a float64 nor a whole number (`np.float32(0.5)`, a complex number) | The kernels are for float64. The transformer block of the corpus is float32 but for its attention scores, which a division by a float64 scalar makes float64, and only those are fused |
| An intermediate that is read twice, or that the program holds a name for | Its value is wanted again, so it has to exist; computing past it would compute it twice. It is an operand of the expressions that read it |
| An intermediate of another shape than the result | It belongs to another loop |
| The matrix product, a reduction along an axis, indexing by arrays | Not element by element. Each is computed by NumPy and is an operand of what reads it |

**An operand is a vector or a number.** Along the innermost axis, each thing an expression reads is either an array as long as the result (`V` in a pattern) or the same number for the whole run (`S`): a Python number, a float64 scalar, a whole number of NumPy's, a float64 array without dimensions, or one of length one along that axis, such as the mean of each row kept with `keepdims=True`. Along the outer axes an operand may broadcast in any way.

**A whole number of NumPy's is the float64 NumPy makes of it.** A table of whole-number coefficients gives an `np.int64` for `table[row, 0]`, and `table[row, 0] * x` on a float64 `x` is float64. NumPy converts such a scalar to float64 once, by C's conversion, and then runs the float64 loop; `float(k)` is the same conversion, so the scalar is a number to a kernel by that value ([issue 318](https://github.com/sofeikov/hotkernels/issues/318)).

- **Which:** a NumPy scalar of a boolean type or of any integer type, with a sign or without, of any value, and an array or a pending result without dimensions of such a type (`np.asarray(3)`, `counts.sum()`).
- **A value a float64 does not hold is rounded as NumPy rounds it,** to the nearest: `np.int64(2**53 + 1)` is 2**53 and `np.uint64(2**64 - 1)` is 2**64, for NumPy's loop and for the kernel.
- **Where:** wherever a float is a number. Either side of `+`, `-`, `*` and `/`, of `np.maximum` and `np.minimum`, of a comparison; either value of `np.where`; an in-place operator and a write into part of an array. `x ** np.int64(2)` is `np.power`'s loop handed the exponent 2.0, which multiplies, as it does for a Python `2`: 0.29 ns an element on a million where `x ** np.int64(3)` takes 9.0 ns, and `x * x`'s bits.
- **What still ends an expression:** a whole-number array with a dimension, of one element as of many, since that is an array of another type; an `np.float32` or a complex scalar; a span of time, which is an `np.signedinteger` by its class and no number; and a Python integer beyond 2**53, as before.
- **The result's type is never changed by it.** A node is inside an expression only if NumPy's answer for it is float64, and with a whole number as one operand that is so only when the other is a float64 array. `float32 * np.int64(3)` is float64 as well, but its array is not a kernel's, and `int64 * np.int64(3)` stays int64.
- **How it is held to NumPy:** each of twenty-three such scalars, in twenty-seven expressions at five lengths, against the real `numpy` and against the same expression written with `float(k)`, which has to be computed by the same kernels to the same bits; `float(k)` against NumPy's conversion at every power of two and either side of it, for int64 and uint64; and the two readers of expressions against each other on all of them.

**The same array may be read twice.** `a + s * a` is the pattern `add(V0,mul(S0,V1))`, served by the kernel of `a + s * b` given one pointer twice. Each place an array is read is a leaf of its own, so a pattern's name does not depend on which of its arrays happen to be the same one.

**The operands of a sum or a product may be in either order.** `c + a * b` and `b * a + c` are one pattern.

### Reductions

`(x ** 2).sum()` is the pattern `sum(sq(V0))`: the square and the sum in one pass, with no array of squares.

NumPy does not add a contiguous array from left to right. It adds pairwise: eight running sums over blocks of up to 128 elements, combined in a fixed tree whose shape depends on the length ([`kernels/sum_f64/spec.py`](../kernels/sum_f64/spec.py) has the rules). A fused reduction is exact only because it makes the same additions in the same order on the values it computes. `fuse_sum_squares_f64` walks NumPy's tree with two small stacks below the stack pointer and squares each element as it meets it; a test compares it with `np.sum(x ** 2)` at every length to 700 and at several to 300,000.

- **The mean is the same kernel.** NumPy's mean is its sum divided by the count, so `mean(sq(V0))` is an entry of the library that calls the sum kernel and divides.
- **Only an array that lies in one run is reduced by a kernel.** NumPy adds such an array, of any number of dimensions, in the order it lies in memory, and whichever of its dimensions is the slowest there: `np.sum(x.T)` has the bits of `np.sum(x)` for a contiguous `x`, compared on eight shapes of two and three dimensions in every order of their dimensions. So the arrays of a call are all C-contiguous, or all one run in one other order of their dimensions, with no dimension of length one. For any other layout NumPy's order is another, and the call is left to NumPy.
- **Only over every element.** A reduction along an axis has another order again (down the columns it is not pairwise at all), and is left to NumPy.

### Writes

The lazy layer makes a write into an array it owns a node of the graph, defined as a copy of what the array held with the write made into it, and computes it into the array's own memory when nothing reads what the array held ([lazy-design.md](lazy-design.md#writes)). With fusion switched on the write and what it stores are one expression, and three kinds of write cost less still. All give the bits of the copy and the write.

| The program writes | Node by node | With fusion |
|---|---|---|
| `w -= s * g`, on a whole array | `s * g` into a temporary, then `-=` into `w` | The operator is an operation of the expression: `sub(V0,mul(S0,V1))`, one kernel call, no temporary. Into a new array; or into `w` itself when nothing reads the old `w` and no floating-point exception is reported under the line's settings (below) |
| `u[1:-1, 1:-1] = e` | `e` into a temporary, which is assigned into `u` | `e` computed straight into that part of `u`, by its program. No temporary |
| `u[1:-1] += s * v` | `s * v` into a temporary, then `+=` into the part | One program that reads the part and gives its new contents: written into the part itself where no exception is reported, and otherwise into a temporary of the part, which is then assigned |
| `np.add(a, b, out=x); np.multiply(x, 0.5, out=x)` | Two calls into `x` | One kernel call, `mul(S0,add(V0,V1))`, into the memory `x` had before the first of the two |
| `e = x % c - d; e /= s`, with numbers `c`, `d` and `s` | `x % c - d` into an array, then `/=` into it | The subtraction and the operator are one kernel call, `div(sub(V0,S0),S1)`, into new memory: what the array held is an operation the program makes itself, so there is no memory of the array's to write into |

- **A kernel writes where the array is only when its result can still be dropped.** A kernel's result is dropped when it raised a floating-point exception that NumPy reports, and NumPy's calls are then made on the same operands ([Floating-point exceptions](#floating-point-exceptions)). That needs the operands to be what they were. So a program writes into the array itself when it reads nothing of what the array held (`u[1:-1, 1:-1] = e` with no `u` in `e`), or when nothing is reported under the `np.errstate` the lines were written under. Otherwise its result goes to new memory, for a whole array, or to a temporary of the part, which is assigned into the array once the result is known to be kept: a pass more, and no copy of the array. The array is copied only when something else still reads what it held, as for any write, and only once the program's result is known to be kept; at a line whose program has been kept before, the copy is made first and the program writes the part of the copy, a pass fewer.
- **The result may be one of the operands, element for element.** `x -= s * y` made in `x` reads and writes the same memory. A kernel loads the eight elements of a step from every array before it stores them, so this is right for a length that is a multiple of eight, and for fewer than eight, which are done one at a time. The module makes every such call as the whole steps and then the elements left over. A result that overlaps an operand in any other way (`u[1:] = u[:-1] + 1`) is declined before a kernel runs, and NumPy computes it.
- **A copy keeps the array's layout,** as a write in place does, so that a later sum adds in NumPy's order.
- **A program may read other parts of the array it writes.** `y[i, :] = a * x[i, :] + b * y[i - 1, :] + c * y[i - 2, :]` reads two rows of the array it writes a third of. The rows read are views computed before the program runs, and each view's array refers to the array it is a view of, which the rule for writing in place takes for a reader of what the array held. So the write went to a copy of the whole array, once for every row: the Deriche filter, 1,500 rows of 1.5 million elements, took 11.6 s for NumPy's 37 ms. A view that shares no byte with the part written shows the same elements before the write and after it, whoever reads it, so such views among what the program reads are not counted as readers (`_Group._apart`), and the row is written where it is. A view that does share bytes with the part is counted as before.

## An expression block by block

A kernel covers a shape of up to eight operations. The equation of state of an ocean model is one expression of 153. Covered with shapes of the library it is 64 kernel calls, and if each wrote a whole array for the next to read, there would be 64 passes over memory where NumPy makes 153: better, and still far from one.

So the kernel calls of an expression are a **program**, and the program is run a block at a time.

```text
for each block of 4,096 elements:
    buffer 0 = fuse_mul_v_v(V3, V4)                  # a * b
    buffer 1 = fuse_add_scaled(V0, buffer 0, S0)     # c + s * (a * b)
    ...
    out      = fuse_sub_product(buffer 7, buffer 2, V9)
```

- **What one kernel gives the next is a buffer of one block,** 32 KB, reused for every block. It is written and read within the cache. Only the expression's operands are read from memory and only its result is written to it, however many operations lie between.
- **Every operation needs a kernel for that,** also one that stands alone between two shapes. So there is a kernel for each single operation: `add(V0,V1)`, `mul(S0,V0)`, `div(V0,S0)`, `sqrt(V0)` and the rest. They fuse nothing by themselves. An operation by itself in a program, with nothing to fuse with, is still NumPy's call, with one exception.
- **One operation is fused where NumPy would make it a short row at a time.** NumPy walks operands it cannot take as one run a row at a time, and for rows of a few elements the call of its loop costs more than the row's arithmetic. So an operation by itself is an expression when its result's innermost dimension is shorter than 16 and its operands do not all lie in one run. A kernel walks them as one run where it can; the four operations of arithmetic are otherwise made by a loop of the module's own over the rows ([Computing a group](#computing-a-group)). Timed on 100,000 rows of 2, 4, 8 and 15 elements against NumPy's one call:

  | The operation | Its layout | How it is made | The layer against NumPy |
  |---|---|---|---|
  | `table - row` | An operand that is the same for every row | One kernel call, the row repeated in a buffer | 9.1, 6.9, 2.6 and 1.8 times as fast |
  | `wide[:, half:] * 0.1` | A slice with as much again between its rows | The module's loop over the rows | 3.1, 2.2, 3.4 and 1.3 times |
  | `wide[:, :half] - wide[:, half:]` | Two such slices | The same | 5.4, 4.0, 3.6 and 2.6 times |
  | `wide[:, :half] * other` | Rows at two pitches | The same | 2.9, 2.1, 2.1 and 1.7 times |
  | `table * column[:, None]` | One value for each row | The same | 1.5, 1.1, 2.4 and 1.5 times |
  | `np.abs(wide[:, half:])` | A slice, and no operation of arithmetic | A kernel's walk straight through, up to 8 elements a row | 1.5, 1.4 and 1.5 times; 0.64 at 15, where it is left to NumPy |
  | `table * other` | Both in one run | NumPy's call: it walks them as one run too | Never an expression |

  A view still pending has no layout to ask and is taken for a slice that cuts its rows; if it does not, the kernel walks it as one run, at NumPy's speed. A call of the kernel for each row, which is what the module had before its own loop, was 0.2 to 0.9 times NumPy's call on these layouts.
- **A program is the tiles that feed one another.** A tile whose operand is an operation with no kernel (`np.where`, `x ** 3`) reads that operation's node, which NumPy computes first as an array; what lies below it is a program of its own.
- **The steps are in the order of the tree, the deepest first.** A step takes the buffer for its result when the steps it reads have been written, and a buffer is free again as soon as the step that read it is written. A chain of calls, each reading the one before, runs in two buffers however long it is, and a long polynomial in a handful.
- **The block is 4,096 elements.** Timed from 256 to 32,768 on the programs of the corpus, the equation of state runs in 83 ms at 256, 77 ms at 1,024, 72 ms at 4,096 and 68 ms at 32,768, and isoneutral mixing the other way, 203 ms to 209 ms; the cavity flow and the others do not move. A program of one step has nothing to keep in the cache and is one call for the whole run.

[`_fused.program`](../prototypes/lazy/_fused.c) checks a program before it computes anything: every step against its kernel's number of operands, every place a step reads against what was written before it, and that only the last step writes the result.

## The kernels

Eighty, in [`prototypes/lazy/kernels/`](../prototypes/lazy/kernels/README.md), one directory each with a spec, a C reference, the assembly and its log, as the kernels of `kernels/` have. The judge finds a kernel by name in either place ([judge-design.md](judge-design.md#layout)) and holds each to its NumPy expression, bit for bit, on 20,000 to 370,000 cases. The package builds from `kernels/` alone and does not contain these.

### Fourteen written by hand

The nine shapes of two operations that weigh most across programs, the one reduction that three programs make, and four whole expressions of the two stencil programs.

| Kernel | Computes | In the corpus |
|---|---|---|
| `fuse_add_scaled_f64` | `a + s * b` | Eight programs |
| `fuse_sub_scaled_f64` | `a - s * b` | A gradient step: five programs |
| `fuse_add_product_f64` | `a + b * c` | Four programs |
| `fuse_sub_product_f64` | `a - b * c` | Four programs |
| `fuse_product3_f64` | `a * (b * c)` | The three Veros routines |
| `fuse_scaled_sum_f64` | `s * (a + b)` | Five programs |
| `fuse_mul_shifted_f64` | `a * (s + b)` | Horner's rule, GELU: three programs |
| `fuse_mul_difference_f64` | `a * (b - c)` | A coefficient times a finite difference: six programs |
| `fuse_sum3_f64` | `a + (b + c)` | Four programs |
| `fuse_sum_squares_f64` | `np.sum(x * x)`, and the mean | An L2 penalty, a loss: four programs |
| `fuse_laplacian5_f64` | `s * a - (b + c + d + e)` | The horizontal diffusion |
| `fuse_flux_limit_f64` | `np.where(a * (b - c) > lo, fill, d)` | The same, twice |
| `fuse_flux_update_f64` | `a - b * (c - d + e - f)` | The same, its last line |
| `fuse_poisson_step_f64` | `((a + b) * s0 + (c + d) * s1) / s2 - s3 * e` | The cavity flow's pressure iteration, eight operations |


### Sixty-six written from the names of their shapes

A shape's name says everything its kernel has to do: which operations, on which operands, in which order. The loop for it is always the same loop: load eight elements of each array, make the operations in the order the tree gives, each one instruction that rounds once, and store. [`make_kernel.py`](../prototypes/lazy/make_kernel.py) writes it.

```bash
uv run python -m prototypes.lazy.make_kernel "sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))"
uv run judge check fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64 --unlocked
```

- **It writes the four files of a kernel:** `spec.py`, whose oracle is the shape as a NumPy expression, with every number tried at zero of either sign, at an overflow, a denormal, an infinity and a NaN; `naive.c`; `arm64/kernel.s`; `x86_64/kernel.s`; and the first entry of each log. `--arch x86_64` writes the x86-64 loop alone for a kernel that is there, reading the shape off its oracle, so the kernels written by hand get one as well.
- **The kernel is then a kernel like the others.** Nothing the tool writes is trusted: the judge checks it against NumPy, and the pattern it serves is read off its oracle. A test holds the files in the repository to what the tool writes today.
- **Registers are given to the tree by the usual rule,** the operand that needs more of them first, four registers to a value. A number is held in every lane of one register. Shapes of up to six arrays and eight numbers fit.
- **Which shapes:** the eighteen single operations (the four of arithmetic on two arrays and on an array and a number either way round, a maximum and a minimum, a negation, an absolute value, a square root, a square), the forty-six shapes of two to six operations that cover the corpus's expressions best, as the miner chose them one at a time, and the two halves of an advection flux, four operations and eight ([What is left](#what-is-left)).
- **`np.maximum` and `np.minimum` are the processor's `fmax` and `fmin` on ARM64,** at every length and position: a NaN wins over a number, and of two zeros the maximum is the positive one. On x86-64 they are `vmaxpd` and `vminpd` with a NaN of the first operand carried through: a NaN still wins, and of two zeros the result is the second operand. C's `fmax` is another function, so the reference spells out both rules, one for each architecture.
- **One shape was refused by the judge, and rightly.** For `(s0 + (-|a|)) / s1` the compiler turns the C reference's `s0 + (-|a|)` into `s0 - |a|`, which gives a NaN the other sign. The assembly was NumPy's; the reference was not, and a kernel whose reference is wrong is not kept.
- **Not tuned.** A written kernel loads with plain `ldp`, or `vmovupd` on x86-64, and is as the tool left it. Against the compiled C it is expected level, as the kernels written by hand are: the loops are ones clang vectorises as written.
- **What it does not write:** a comparison and `np.where`, whose mask is kept in a register; a reduction; float32.

### The library

**A kernel's pattern is read off its oracle.** A spec's oracle is the NumPy expression the judge holds the kernel to:

```python
def oracle(a, b, s):
    return a + s * b
```

`fuse.py` calls it with lazy arrays and two numbers. The oracle writes a graph without computing anything, the expression is read off that graph as off any other, and its name, `add(V0,mul(S0,V1))`, is the pattern. Which leaf is which argument is known because the arrays passed in are the leaves read.

- **The specification is its own pattern.** What a kernel is matched against and what the judge checks it against are one function, so they cannot disagree. A kernel whose oracle is not one expression of the operations above, or does not read each argument once, has no pattern and is refused when the library is loaded.
- **A name is the tree as written,** with the operands of a sum, a product, a maximum or a minimum in a fixed order: `sub(V0,mul(S0,V1))`. Two expressions of one shape have one name whichever way round the program wrote them. On x86-64 a maximum and a minimum keep the program's order: of two zeros NumPy's loop there gives the second operand, so `np.maximum(x, 0.0)` and `np.maximum(0.0, x)` are different functions.
- **Arguments come in one order:** the input arrays, the output array, the length, then the numbers. [`_fused.c`](../prototypes/lazy/_fused.c) calls every kernel with eight integer arguments and eight floating-point ones, of which the kernel reads those it has; both ARM64 and x86-64 pass that many in registers or in order on the stack, so one call reaches any kernel.

- **A shape is preferred to NumPy's call where the passes are as many.** A kernel's pass is a step of a program and stays in the cache; NumPy's is a pass over memory. Among shapes the largest that gives the fewest passes is taken.

## Finding the shapes

### The corpus

Twenty-six programs, in [`prototypes/lazy/programs/corpus.py`](../prototypes/lazy/programs/corpus.py). Each runs with `import numpy as np` or with the stand-in for it, and nothing else of it changes; all give NumPy's results bit for bit lazily, with fusion and without.

| Programs | What they are | From |
|---|---|---|
| `regression`, `monitor`, `attention` | A linear regression, the same gathering its loss as it goes, a transformer block in float32 | This repository, `prototypes/lazy/programs/` |
| `perceptron`, `char_rnn`, `particle_filter`, `veros_eos`, `veros_isoneutral`, `veros_tke` | A two-layer perceptron, Karpathy's `min-char-rnn.py`, a bootstrap particle filter, three routines of the Veros ocean model | `delivery/apps/`, the four applications that are NumPy code |
| `cavity_flow`, `hdiff`, `black_scholes` | A lid-driven cavity flow and a weather model's horizontal diffusion from npbench, Black-Scholes from dpbench | [`prototypes/lazy/programs/vendor/`](../prototypes/lazy/programs/vendor/README.md), copied unchanged |
| `trimesh_mesh`, `sklearn_metrics`, `empyrical_stats`, `skimage_pipeline`, `pandas_sales`, `astropy_timeseries` | Six programs that are calls into a library | `delivery/apps/`, the other six applications |
| `example`, `flame`, `azimint_naive`, `azimint_hist`, `mandelbrot1`, `mandelbrot2`, `make_decision`, `pandas_filter`, `equalize_hist` | The nine workloads: this repository's example and its fractal flame, five published benchmarks copied unchanged, and two calls into a library | `delivery/workloads/` |

That is every program the repository times with and without `hotkernels.install()` ([README](../README.md#whole-programs)), beside the three examples and the three benchmarks added for this work.

- **A program that is calls into a library changes its one line like any other.** The `np` of the module that calls the library is the stand-in, and the arrays it is given are the stand-in's. The library has its own `numpy` and asks for `ndarray`s, so each array is made where it is handed over, and what the library computes is in no graph. Such a program says what the layer costs a program it can do nothing for.
- **One library does not take a lazy array.** scikit-image's `regionprops_table` asks `np.isscalar` of the mean of an image it was handed, with its own `numpy`. That function looks at the type of its argument, and for a lazy image the mean is a lazy array without dimensions, not a number: the library fails on it. `skimage_pipeline` is given plain arrays, and nothing of it is deferred. The stand-in's `np.isscalar` answers for the value, which serves a module that computes with the stand-in and not this one ([lazy-design.md](lazy-design.md#what-a-library-asks-a-lazy-array)).
- **One program needed the layer to learn something:** `mandelbrot2` assigns to `x.shape`, which a lazy array now takes ([lazy-design.md](lazy-design.md#pending-work-and-writes)).

### Mining

`uv run --group apps python -m prototypes.lazy.programs.mine` runs each program lazily, one node at a time. Before every computation the expressions among its nodes are read off, and every call is timed. So each expression has a measured weight: the time NumPy spent on the operations in it.

- **It is the graph that is mined, not the source.** The graph is the program after Python has run: loops unrolled, numbers worked out, views resolved. `dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]` is one multiplication of an array by a number.
- **The weight is time, not count.** A shape that occurs a thousand times on arrays of a hundred elements weighs less than one that occurs once on ten million.
- **Mining says which shapes occur and what they cost. It does not say that a shape may be fused.** That is the rule of [What is fused](#what-is-fused), and the judge's check of each kernel.

Where NumPy's time goes, at the large size, as shares of the time spent computing nodes, for the seventeen programs whose NumPy code is their own:

```text
program            deferred  graphs  largest        ms in expressio simple, alon other elemen       matmul    reduction        write     indexing     the rest
regression              384      22      261      46.6        29.9%         0.3%         0.0%        46.8%        22.8%         0.0%         0.0%         0.2%
attention               220       1      220      29.1         4.7%        11.7%        57.8%        21.2%         4.1%         0.0%         0.0%         0.5%
perceptron             8854     565       33      38.8        26.4%         3.6%         6.1%        22.8%        10.5%        26.6%         2.5%         1.6%
char_rnn              32423     207    25387      60.7        39.4%         6.0%         7.7%        30.9%         2.9%         5.6%         0.0%         7.5%
particle_filter        2537      43      280     321.5        38.2%        14.4%        20.5%         4.4%         1.3%         1.5%         5.3%        14.3%
veros_eos               366       1      366     175.7        85.5%        11.0%         3.6%         0.0%         0.0%         0.0%         0.0%         0.0%
veros_isoneutral        718       7      255     227.4        77.0%         0.0%        22.7%         0.0%         0.0%         0.1%         0.0%         0.2%
veros_tke               382       7      272      85.3        86.1%         3.2%         4.5%         0.0%         0.0%         6.0%         0.0%         0.3%
cavity_flow            6920       2     6856     165.2        94.7%         0.0%         2.9%         0.0%         0.0%         1.3%         0.0%         1.0%
hdiff                    40       1       40     109.1        93.5%         6.4%         0.0%         0.0%         0.0%         0.0%         0.0%         0.0%
black_scholes            28       2       25     104.1        20.9%         7.5%        71.7%         0.0%         0.0%         0.0%         0.0%         0.0%
example                  13       9        4      20.9         0.0%         2.3%        11.5%         0.0%        17.7%         1.7%        43.6%        23.2%
flame                  7418     416       53     117.5         7.4%         4.3%        17.5%         0.0%         0.1%        35.4%        35.2%         0.0%
azimint_naive         10001       1    10001     183.0         0.0%         2.8%        78.2%         0.0%         1.3%         0.8%        16.8%         0.0%
mandelbrot1            1205       2     1053     154.3         0.0%        44.4%        16.6%         0.0%         0.0%        17.9%        21.1%         0.0%
mandelbrot2             727     322        9     138.2         0.0%        33.0%         6.7%         0.0%         0.0%        32.1%        28.1%         0.0%
make_decision        491520       5   100000     545.0         0.0%        13.7%        29.5%         0.0%        20.1%        21.8%        14.9%         0.0%
```

- **A program made of arithmetic is nearly all expression:** 94% of the horizontal diffusion's time and of the cavity flow's, and 77% to 86% of the three Veros routines'.
- **A program made of calls, of masks or of small arrays is not.** 72% of Black-Scholes is `erf`, `log` and `exp`; 78% of the naive azimuthal integration is comparisons and the masks made of them; the two Mandelbrot programs are complex arithmetic and selections by a mask, with no expression at all.

### The shapes

An expression of the corpus has between two and 153 operations. The miner cuts every expression into the smaller trees it contains, up to six operations, and counts each by the time of the expressions it occurs in. A library is then chosen one shape at a time: the next shape is the one that removes the most passes, given the shapes already chosen.

The first twenty-four, with the share of the removable passes that the library removes once each is in it:

```text
  1    6.4%  2 operations  add(V0,mul(V1,V2))                           flame, particle_filter, veros_eos, veros_isoneutral, veros_tke
  2   12.6%  6 operations  sub(div(add(V0,mul(S0,add(V1,V2))),S1),mul(S2,V3)) cavity_flow
  3   17.3%  2 operations  add(V0,mul(S0,V1))                           attention, cavity_flow, flame, particle_filter, perceptron, regression, veros_eos, veros_isoneutral, veros_tke
  4   21.0%  5 operations  sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))   hdiff
  5   24.5%  2 operations  mul(V0,sub(V1,V2))                           black_scholes, cavity_flow, flame, hdiff, veros_eos, veros_isoneutral, veros_tke
  6   27.6%  6 operations  div(neg(add(V0,mul(V1,V2))),sub(min(S0,V3),S1)) veros_isoneutral
  7   30.5%  2 operations  mul(V0,mul(V1,V2))                           veros_eos, veros_isoneutral, veros_tke
  8   32.7%  4 operations  where(gt(mul(V0,sub(V1,V2)),S0),S1,V3)       hdiff
  9   34.4%  2 operations  mul(S0,add(V0,V1))                           attention, cavity_flow, flame, particle_filter, regression, veros_isoneutral
 10   36.1%  2 operations  mul(V0,add(S0,V1))                           attention, black_scholes, veros_eos
 11   37.8%  5 operations  sub(V0,mul(V1,sub(add(V2,sub(V3,V4)),V5)))   hdiff
 12   39.2%  6 operations  add(V0,mul(V1,mul(mul(V2,mul(S0,V3)),sq(V4)))) veros_isoneutral
 13   40.6%  5 operations  add(V0,mul(max(S0,mul(V1,V2)),mul(V3,V4)))   veros_isoneutral
 14   42.0%  5 operations  sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))   cavity_flow
 15   43.4%  3 operations  add(V0,add(V1,add(V2,V3)))                   char_rnn, hdiff, veros_eos, veros_isoneutral
 16   44.6%  2 operations  mul(V0,mul(S0,V1))                           attention, flame, veros_eos, veros_isoneutral, veros_tke
 17   45.8%  2 operations  sub(V0,mul(S0,V1))                           cavity_flow, perceptron, regression, veros_tke
 18   47.0%  4 operations  div(add(S0,neg(abs(V0))),S1)                 veros_isoneutral
 19   47.9%  3 operations  add(sq(V0),sq(V1))                           flame, particle_filter
 20   48.7%  2 operations  add(V0,add(S0,V1))                           veros_eos, veros_tke
 21   49.5%  2 operations  div(sub(V0,V1),S0)                           cavity_flow, veros_tke
 22   50.2%  4 operations  sub(sub(V0,mul(V1,V2)),mul(V3,V4))           cavity_flow, veros_eos
 23   50.9%  6 operations  add(mul(S0,add(V0,V1)),sub(V2,mul(S1,sub(V3,V4)))) cavity_flow
 24   51.6%  3 operations  mul(V0,add(V1,sub(V2,V3)))                   black_scholes, veros_eos
```

- **The tail is long.** Twenty-two shapes remove half of the passes that fusing every expression whole would remove, and sixty remove 61%; the sixtieth still adds a tenth of a percent.
- **That is why a program has a kernel for every step.** No library of whole shapes covers a long tail, and a pass that is left is a pass over memory. With the single operations as kernels, what a missing shape costs is a step within the cache, not a pass.

## Planning

Before each computation `evaluate` asks its planner which nodes are computed together ([lazy-design.md](lazy-design.md#computing)). `fuse.plan` answers in four steps.

1. **Writes into whole arrays become their one call** ([Writes](#writes)), whatever else is planned.
2. **Expressions are read off the nodes,** the newest first. An operation is the top of an expression unless an expression already found contains it; below it, each operand is inside the expression if it is an operation a kernel can make, of the same shape, about to be computed, read by this operation alone, and held by nothing else. An expression is read 200 operations deep at most. One of a single operation is kept when it is a store, or when NumPy would make it a short row at a time ([An expression block by block](#an-expression-block-by-block)).
3. **Each expression is covered with shapes of the library by the fewest passes,** found from the leaves up: at each operation, the shapes with that operation at their top are held against the tree. A shape whose top operation could not take the operation's operands, by what each is (a number, an array, a named operation), is not walked at all.
4. **The tiles that feed one another become one program,** a group under the node whose value it gives. The plan is the groups.

**What "held by nothing else" means.** An intermediate the program has a name for, or that a node outside this computation reads, will be asked for again. Both are a reference to the node, and CPython counts references exactly: a node of a computation is referred to by the nodes of it that read it, by the one list it is in, and by nothing else unless one of the two holds. The count decides only what is worth fusing. A node that a plan skips stays pending and is computed when asked for, so a wrong count costs time and never a result. Two things this found while it was built are worth knowing by anyone who touches the code: a `for` loop's variable still refers to the last node after the loop, and pytest's rewriting of `assert` keeps every value of the expression in it alive. Both make a node look held.

**Nothing planned outlives the computation.** A plan refers to nodes and so to arrays. If anything kept a plan alive past its computation, those arrays would stay in memory until Python's collector of cycles next ran, and every result in between would be written to fresh memory where NumPy's own calls reuse what was just freed. This happened in the building: two functions that called themselves by a name of the function they were made in formed a cycle, one run of an ocean routine took 46,000 page faults where the run without fusion takes one, and was a quarter slower for it. A test now runs a fused computation with the collector off and holds that no node is left.

**An expression with one value for each row has a second program.** `dx[:, :, None]` beside a grid is one value for each row. A kernel takes one value for a number in a call, so the expression is planned with each row a run, which is quickest where a kernel can read the other arrays where they lie. Where one of them steps over elements and has to be gathered, a block is many rows, and the module gives the program up ([Computing a group](#computing-a-group)). So where rows are shorter than 32, such an expression is written a second time with the value read as an array, covered with shapes again, and the second program is kept beside the first for when the first is given up: the module fills a block with each value as often as its row is long. It is kept as its steps and the places of what it reads among the first program's operands, with no reference to a node, for the reason of the paragraph above; the first version of it kept the nodes, and the kinetic energy routine took 116 ms for 92 ms.

**A plan is remembered by the shape of its expression.** A loop writes the same expressions at every turn, and covering a tree with shapes is most of what planning one costs. What is kept is the program and where each thing it reads is among the expression's operands, by the tree's operations and the pattern of its operands. The next expression of that shape is planned by a look-up.

**The expressions are read by the C core** ([`_core.c`](../prototypes/lazy/_core.c), `find`), when the core is what writes the nodes. Reading is a walk over every node of the computation with a few questions asked of each, and in Python that walk was three quarters of what planning cost once plans were remembered.

- **It is the functions of `fuse._Finder`, one for one,** and gives the same `Expression`, `Term` and `Leaf` objects to the Python that plans them. The Python reader is the definition, and is what runs without the core. A test runs both on every computation of eight programs of the corpus and holds what they find to be the same.
- **What it notes of each node is in one table by the node,** where the Python has a set and two dictionaries by node number: whether the computation takes it, how often its nodes read it, whether something else holds it, whether an expression found so far has it inside.
- **It writes down what planning reads off each tree:** the tree's shape as the key a plan is remembered by, the nodes of its operations in order, and the error settings they were written under. Planning an expression whose plan is remembered is then a look-up and one object made.
- **Planning the cavity flow's 260 expressions costs 3 ms a run** of 55 ms: about 2 ms to read them and the rest to place each. Read by the Python reader it costs 16 ms, and without the memory of plans 45 ms.

**An expression is read 200 operations deep at most.** A loop that adds to one large array a few thousand times, never asking for a value, writes one chain that long. Reading, covering and placing a tree each walk it a call per level, and Python allows a thousand. Below the limit an operand is read as an array: it is then the top of an expression of its own, which is computed first, and every operation of the chain is still a kernel's. Where the cut falls costs one array written to memory per 200 operations.

**Arrays of fewer than 32,768 elements are left alone.** Writing a node, planning it and coming back to it costs about a microsecond, and a pass of NumPy's over fewer elements than that costs little more. A computation none of whose nodes is that large is not read at all. With fusion on, such operations are not nodes in the first place ([As the graph is built](#as-the-graph-is-built)). Where the floor is matters little above 16,384 and a great deal below. The fractal flame, whose arrays have 25,000 elements, runs in 149 ms with the floor at 16,384, 148 ms at 32,768 and 147 ms at 65,536, and the perceptron, whose arrays have 8,000, in 35 ms at all three; at 8,192 they run in 235 ms and 76 ms. No other program of the corpus moves by 1%.

## Computing a group

A group is computed when `evaluate` reaches the node it gives the value of. [`_fused.c`](../prototypes/lazy/_fused.c) stands between the kernels, which are leaf functions over contiguous runs, and arrays of any shape.

- **It walks the outer axes and runs the program once per contiguous run,** block by block. Every outer axis that every operand walks straight through joins the run, so a C-contiguous expression of any number of dimensions is one run. `u[1:-1, 1:-1]` of a grid is one run per row.
- **An array that is the same along an outer axis does not end a run.** `dzt[None, None, :]` beside a grid of 204 x 204 x 26 would make every run 26 elements long. Instead the run is lengthened all the same, and that array is read from a buffer that holds its 26 elements over and over for the length of a block.
- **An array that steps over elements is gathered.** A column of a table, `particles[:, 0]`, or a grid taken at one time level, `tke[:, :, :, tau]`, is not memory a kernel can read. The shape is then walked in the order of its rows, a block at a time: an array that lies in that order is read where it is, and any other is copied into a buffer of its own, by a loop written for it. The result's block is copied out if the result steps too.
  - *Each array gathered walks its own rows,* and a row is as many of its innermost dimensions as it walks at one step: a grid at one time level is every third element through two or three dimensions, thousands of elements for one loop, where a row of the grid is 26.
  - *An array read in two places of the program is gathered once.*
  - *One value for each row is filled in,* each as often as its row is long, when the program is one that reads it as an array ([Planning](#planning)).
  - *A copy costs about what a pass of NumPy's over the array does:* 458 µs for a million elements three apart, since reading one element in three pulls every cache line of an array three times the size. So a program is run this way only when it saves more passes than it makes copies: when it has more operations than arrays to copy, and three operations for each array copied or filled when a value for each row is filled in. One kernel of two operations on two such arrays was timed 0.6 ms a million elements slower than NumPy's two calls; five operations on two arrays and a value for each row took 3.99 ms for NumPy's 2.43 ms; the advection flux, seventeen operations on three and one, 3.94 ms and 5.36 ms in its two directions for NumPy's 6.9 ms in each.
- **Short rows at one pitch are walked straight through.** A slice that cuts the innermost axis, `x[2:-2, 2:-2, :-1]` of a grid 26 deep, is rows of 25 elements every 26, which cannot be joined: a row does not end where the next begins. But where every array of the expression has the same elements between its rows, the steps walk each array from its first element to its last, the elements between the rows included, and only the rows are copied out of the last step's buffer. Nothing is copied in, at the cost of arithmetic on elements nobody asked for. A flag that only those elements raised drops the result like any other.
  - *For rows of 16 to 31 elements, when a quarter of the walk at most lies between them.* With more between, a program for each row is quicker.
  - *For rows shorter than 16, up to eight times the row's length apart.* NumPy's loop is called once for each row of an array it cannot walk as one run, and for a row of a few elements the call costs more than the arithmetic: `particles[:, :2] += particles[:, 2:] * dt` on 100,000 rows takes NumPy 1,256 µs, and the same arithmetic on 200,000 elements in one run 82 µs. So a walk that computes eight times what was asked for still wins. On 400,000 elements in rows of 2 to 15, every 1.25 to 8 times their length, a program of one kernel is 1.0 to 2.8 times NumPy's passes and one of two kernels 1.05 to 3.0 times.
- **Short rows at more than one pitch are a program each, from 16 elements.** A slice of a grid beside an array computed from one has rows 25 long at pitches of 26 and 25. Each row is then a run: every step is a call for that row, by a walk that has worked out once everything that is the same for every row. On `(a + b) * c - d` over four million elements that is 1.06 to 1.21 times NumPy's three passes for rows of 16 to 28, and 0.70 to 0.88 for rows of 4 to 12, which are left to NumPy. A program of one kernel is a call for each row down to two elements, 1.2 to 2 times NumPy's two passes, when the kernel reads three arrays or numbers or more; one that reads fewer is one operation or two, and NumPy's pass is quicker than a call for each row. Rows of one element, which `x[:, :1]` has, are never a call each: that ran 3 to 9 times slower than NumPy.
- **One operation of arithmetic on rows shorter than 16 is a loop of the module's own** (`program_rows`). A program that is one kernel of `+`, `-`, `*` or `/`, on two arrays or an array and a number, is not worth a call for each row, and need not be walked through what lies between the rows either: the module makes the operation itself, by two loops with nothing in them but the rows' arithmetic, which is the processor's one instruction for the operation, as in NumPy's loop and in the kernel. The arrays may lie at any pitch, the number may be another for each row (`weights[:, None]`), and the result may be one of the arrays. Written out by themselves such loops take 105 µs for `table * weights[:, None]` on 100,000 rows of four and 90 µs for `table[:, :2] += made`, where NumPy takes 294 µs and 861 µs.
- **A product of three arrays stored into short rows of a result that steps is a loop of the module's own as well** (`program_stored`). Isoneutral mixing has sixteen lines such as `Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * sxb * maskW[2:-2, 2:-2, :-1]`: the result is every fourth element of an array with two more axes after its grid, in rows of 25, and what it stores is read from arrays at pitches of 25 and 26. NumPy makes the two products a row at a time and stores the rows at the step, 1,537 µs for a million elements, of which 731 µs the products. Here each row is made into a buffer on the stack, in the kernel's order `a * (b * c)`, and stored from there: 1,001 µs. It is the one expression such a loop is written for, because it is the one the corpus stores this way; what was tried before it is in [What was built on it and did not gain](#what-was-built-on-it-and-did-not-gain).
- **It reads float64 arrays** with strides that are positive and do not grow from one axis of the walk to the next. The walk is in the order the output lies in memory: the module sorts the output's axes by stride, the largest outermost, and leaves two of one stride in the order they are written in, which is what NumPy's iterator does with the operands of an operation. For an output in C order that is the order written, and NumPy's result for such operands is laid out in C order, which the output is.
- **Arrays that all lie in one other order are walked in that order.** `np.mgrid[...].transpose(1, 2, 0) % 1` has shape (2048, 2048, 2) and strides (16384, 8, 33554432): its last axis is the slowest in memory. NumPy gives the result of an operation the order its operands lie in, and its loops walk such an array as it lies, not as four million rows of two. So does the module. `fuse.py` hands it new memory in C order first, which is right for nearly every program and costs the others nothing; when that is declined and the first array of the result's shape has a stride larger than the one before it, the memory is given back and new memory is asked for with that array's layout (`np.empty_like`), which the module then sorts and walks. A write into an array the program holds is walked in that array's order from the start.
  - *The rule is narrower than in C order,* because the layout NumPy gives a result is decided operation by operation, and the module sees only the arrays an expression reads. Every array has every axis of the shape, or is one element: an array that is the same along an axis leaves the order of that axis to the other operand of its operation, and `row + column` is in C order whatever a third array is. Where two axes are walked in the other order than they are written in, the outer one's stride is the larger and not merely as large, since of two equal strides NumPy takes the order written. And no axis has length one: NumPy gives such an axis no stride when it sorts, where it puts the axis then depends on which of its two loops an operation took, and with it the strides of the result. Operands in two different orders are declined too; NumPy's result for those is in C order where they disagree.
  - *Everything else is as in C order,* in the sorted walk: slices of a transposed array are a run for each row, one that steps is gathered, the result may be one of the arrays.
  - *One operation by itself gains nothing from it and loses nothing.* `a.T * b.T` on 100,000 rows of 2, 3, 8 and 15 elements, which is planned as an expression because the arrays are not C-contiguous, takes 0.87 to 1.00 times NumPy's call as one kernel call; left to NumPy after being declined it took 0.80 to 0.93. Three operations on the same arrays are 1.44 to 1.79 times NumPy's three passes. Intel Core i5-14600KF.
- **It declines anything else:** a reversed array, arrays that lie in different orders (`a.T + b`), an array out of order beside a row or a column, or with an axis of length one, an output that overlaps an operand without being it, rows shorter than 16 elements at more than one pitch or more than eight times their length apart, an array that steps with fewer than 32 elements at one step, unless it is the result of the product above, and a program with too few operations for the arrays it would copy. The group's nodes are then computed by their own calls, in order. Declining is decided before anything is computed or copied.

### Floating-point exceptions

NumPy reports overflow, division by zero and invalid operations as warnings, or as errors under `np.errstate(over="raise")`, with the name of the operation and on the program's line. A kernel of several operations cannot say which of them raised a flag.

So the module clears the processor's flags before a kernel runs and reads them after. If a flag is raised that NumPy would report under the `np.errstate` the operations were written under, **the kernel's result is dropped and NumPy's own calls are made**, one node at a time. The warning, the error, its words and its line are then NumPy's by construction, and so are the values.

- **An exception the program silenced costs nothing.** Under `np.errstate(all="ignore")`, and for underflow always, the kernel's result is kept.
- **An expression that raises one is computed twice.** The line that wrote it is remembered, and its expressions are left to NumPy from then on, so a loop pays once.
- **A comparison raises a flag NumPy does not report.** The processor's comparison marks a NaN operand as an invalid operation; NumPy's comparison clears the flag. The fused flux limiter is therefore computed by NumPy's calls when its data has a NaN, unless the program has silenced invalid operations.

### A write made at once

A write into memory the program holds makes the layer compute everything pending first, after which nothing may be pending ([lazy-design.md](lazy-design.md#writes)). The nodes inside a group have no value of their own. Nothing should hold one, or the plan would not have put it inside; `evaluate` checks all the same, and computes any that is still alive after the planned computation, since the state after it must not depend on a count of references.

## As the graph is built

A node costs time three times: when it is written, when its computation is planned, and when the loop that computes it comes to it. NumPy makes a call on an array of a hundred elements in a third of a microsecond. A program on small arrays runs two to eleven times slower lazily than eagerly for that reason alone, and nothing that computes a graph faster can win that back, since there is nothing to fuse in it.

So with fusion on, **what is small is done as the program writes it,** by the C core (`array.at_once_below`):

| The program writes | What happens |
|---|---|
| A call that gives fewer than 32,768 elements, on arrays that all have their values | NumPy's call, there and then. The result is a lazy array that has its values |
| A ufunc or an operator on small arrays one of which is still pending | What it reads is computed, and then the same. A small call never waits |
| A call of any size that no kernel could make, on arrays that all have their values: a comparison, a selection by a mask, a matrix product, `np.exp`, anything complex or float32 | The same. Nothing written on top of it could be computed together with it, so its node would only cost |
| A view, a transpose or an element of an array that has its values | NumPy's own, whatever the size: a view costs nothing to take |
| A write, when no node at all is pending | NumPy's write, into the array's own memory |
| A ufunc that names its output, `np.maximum(x, 0, out=x)`, when no node at all is pending; `a += x` on a plain `ndarray` `a` with `x` the layer's | NumPy's call with the output's memory as `out`, whether the output is the layer's array, one the program handed in, or a plain `ndarray` |
| A call through the stand-in on plain `ndarray`s, `np.sqrt(a)`, `np.dot(a, b)`; an operator of a plain array with one of the layer's | The same as on the layer's arrays, by the same limits: a plain array is its own value |
| `np.clip(x, lo, hi)`, `np.average`: a function the layer has no rule for, whose result is a new array | NumPy's call on the values, whatever the size, counted as a call outside the deferred set |
| Anything else: arithmetic on float64 that gives a large result, a large call with an operand still pending, a write while something is pending | A node, as before |

- **It is what the eager program does at that line.** The value is NumPy's, and so are an error and a warning, raised where the program wrote the call. Nothing later can tell the difference: the result is a lazy array like any other, and a large operation on it is deferred.
- **A small call does not wait for what it reads.** `y[i, :] = a * out[i, :] + b * y[i - 1, :]` in a loop, with `out` a large array still pending, is thousands of operations on rows of a thousand elements, and each waited as a node because the row of `out` it read had no value yet. The Deriche filter of [issue 298](https://github.com/sofeikov/hotkernels/issues/298) wrote 36,000 nodes that way and took 170 ms for NumPy's 37 ms. A pending operand of fewer than 32,768 elements is now computed, with whatever it reads, and the call is made at once on its value: 49 ms. What is given up is a fusion across the peek: `t = a + b; s = t[0] * 2; u = t * c` computes `t` by itself where it could have been one pass with `u`. That is one pass of one array, against a node for every small operation that follows.
- **A write at once is sound when nothing is pending.** The copy a deferred write is defined by exists for operations written before the write and not computed yet. With no node pending there is none, and writing into the memory is NumPy's own meaning, views included.
- **That holds for an array the program was handed as for one the layer made.** A write into an array that came in through `np.asarray` went to Python, which is where a write that waits has to be looked at: such an array may be read by anyone. A write made now is the eager program's write, whoever reads it, so the core makes it: 0.3 to 0.4 µs where Python took 3.9 µs and NumPy takes 0.2 µs. The symmetric rank update and the Gauss-Seidel stencil of issue 298, which update the array they are given a row or an element at a time, ran 7 and 25 times slower than eagerly through Python's road.
- **A view taken at once is NumPy's view and nothing more,** so a write into an array that such a view may be looking at does not wait: everything pending is computed and the write is made in the memory, where the view sees it. Whether one may be looking is read off the reference NumPy's view holds to its array ([lazy-design.md](lazy-design.md#pending-work-and-writes)). Of the cavity flow's 1,220 writes into its grids, 1,196 still wait: a slice of a grid with a write pending is deferred with the write, and holds nothing of NumPy's.
- **An in-place operator made at once on rows of a few elements is its own expression, and an assignment.** `particles[:, :2] += particles[:, 2:] * dt` writes through a view taken at once, so it does not wait, and NumPy's in-place addition is a call of its loop for each row of two: 861 µs for 100,000 rows. The layer makes `view + other` as an expression, which the module's loop over the rows computes, and assigns the result's rows by a loop of the module's as well (`_fused.assign`). A warning the sum raises is NumPy's for that operation on that line; an error leaves the array as it was.
- **The result is one object.** A lazy array is a handle on a cell that holds a node, three objects, and making and freeing them cost 155 ns of the 180 ns a small `a + b` took over NumPy's 265 ns. The result of a call made at once is a lazy array that holds NumPy's value and nothing else, taken from the instances the core has kept; its node and its cell are made if something asks for them, which the next small call does not ([lazy-design.md](lazy-design.md#the-c-core)).
- **The call is made where the interpreter first looks.** The core's functions stand in the class's table for the operators, for `x[i]` and for `x[i] = v`, so `a + b` is one call from the interpreter to NumPy's ufunc. A reduction method, `x += y`, `float(x)` and a name of the stand-in (`np.tanh`) are answered by the core as well. A call through the stand-in (`np.tanh(x)`, `np.dot(a, b)`) is made without NumPy's way round through its protocols, and for the functions that always give a new array without the rules being asked.
- **A library's calls on its own arrays are made the same way.** A library validates what it is given with its own `numpy`, so its arrays are plain whatever its `np` is, and code written for speed computes in place: `np.maximum(X, 0, out=X)`, `np.exp(X, out=X)`, `np.clip(y, eps, 1 - eps)`, `param += update`. Each went through the stand-in's Python, which makes every plain array lazy, then through NumPy's protocol, and for a write or a call outside the deferred set through the Python that computes everything pending first, with nothing pending. The core makes them where the stand-in is called ([lazy-design.md](lazy-design.md#the-c-core)). On arrays of 200 × 100, over NumPy's own call, before and after ([issue 320](https://github.com/sofeikov/hotkernels/issues/320), x86-64):

  | Call | NumPy | Plain arrays, before | after | The layer's arrays, before | after |
  |---|---|---|---|---|---|
  | `np.maximum(x, 0, out=x)` | 4.27 µs | +5.95 µs | +0.27 µs | +4.94 µs | +0.18 µs |
  | `np.exp(x, out=x)` | 35.35 µs | +5.57 µs | +0.30 µs | +4.54 µs | +0.20 µs |
  | `np.clip(x, lo, hi)` | 4.20 µs | +8.12 µs | +0.88 µs | +7.28 µs | +0.87 µs |
  | `x += y`, `y` the layer's | 2.44 µs | +4.29 µs | +0.27 µs | | |
  | `x + y`, `y` the layer's | 5.78 µs | +0.57 µs | +0.15 µs | | |
  | `np.sqrt(x)` | 11.59 µs | +1.14 µs | +0.10 µs | +0.11 µs | +0.11 µs |
  | `np.dot(s, s)` | 1.68 µs | +1.97 µs | +0.12 µs | | |
  | `np.sum(x, axis=0)` | 4.42 µs | +1.60 µs | +0.20 µs | | |

  `np.clip` is outside the deferred set, and each call of it is counted on its line for `lazy_report()` by the function that counts every such call: 0.7 µs of its 0.9. scikit-learn's perceptron, which makes 1,800 such calls in a run, went from 0.89x of NumPy to 0.95x fused, and from 0.66x to 0.98x at a tenth of the size ([lazy-codebases-findings.md](lazy-codebases-findings.md#out-npclip-and-plain-operands-issue-320)). A write while something is pending still computes first, in Python: `param += update` with `update` an expression that waits.
- **What it costs is 15 to 115 ns a call over NumPy's own,** on sixteen elements, each line in a loop of 300,000 turns:

  | The program writes | NumPy | On a lazy array | Over |
  |---|---|---|---|
  | `a + b` | 258 ns | 291 ns | 33 ns |
  | `a * 2.0` | 415 ns | 409 ns | nothing |
  | `(a + b) * a - b` | 784 ns | 875 ns | 91 ns |
  | `a[3]` | 48 ns | 79 ns | 31 ns |
  | `a[3] = 1.0` | 40 ns | 56 ns | 15 ns |
  | `a[1:-1]` | 82 ns | 113 ns | 31 ns |
  | `a.sum()` | 362 ns | 426 ns | 64 ns |
  | `np.tanh(a)` | 297 ns | 373 ns | 76 ns |
  | `np.dot(a, b)` | 368 ns | 484 ns | 116 ns |
  | `s + 1.5`, with `s = a.max()` | 51 ns | 98 ns | 48 ns |

  An operator on a lazy array calls the ufunc directly, where `ndarray`'s own operator first looks at what the other operand is; that is why `a * 2.0` costs nothing. A view holds NumPy's view and nothing else ([lazy-design.md](lazy-design.md#the-c-core)); while it stood on the cell of the array it was taken of, with a step, it cost 112 ns over NumPy's. A function that is no ufunc, called through the stand-in, costs the most: the stand-in's own call, and NumPy's Python wrapper around the function. Arithmetic on a number that came out of an array is the NumPy scalar's own operator, called on the value: through the operator's ufunc it cost 565 ns over the scalar's.
- **Which calls wait is the planner's to say.** `fuse.enable()` names the callables a kernel can make and their element type; a large call that is none of them, or reads an array of another type, is made at once. One whole number without dimensions, `table[i, j]` of a lazy table of integers, is not an array of another type there: a kernel takes it as a number, so the call waits. A program of masks, or of complex numbers, writes no graph at all then.
- **The graphs of a program on small arrays are gone,** and with them what the layer could show of it. That is the trade, and it is why the limit is a switch: without `fuse.enable()` every operation the layer defers is a node, and the tools that look at graphs see them all.

Three more things are done for every program, fused or not. A deferred write is computed into the array's own memory when nothing reads what the array held ([lazy-design.md](lazy-design.md#writes)). The loop that computes nodes catches warnings once for a computation and changes the error settings only when a node wants others, where it did both around every call. And `x ** e` is NumPy's operator, which squares for an exponent of 2 where `np.power` runs its general loop, ten times slower.

## Why the bits are NumPy's

The argument is per operation, and the judge checks it per kernel.

| Hazard | What a kernel does | How it is checked |
|---|---|---|
| A product and a sum fused into one multiply-add, which rounds once | `fmul` then `fadd`, never `fmla`. The C reference starts with `#pragma STDC FP_CONTRACT OFF`; without it clang fuses the two at the judge's flags | The judge compares bits with NumPy's `a + s * b` on every size to 65, on random values and on its special ones. Compiled without the pragma, the reference's loop is `fmla` |
| A division made as a product with a reciprocal | `fdiv` by the number | The same, with divisors that are zero, tiny and infinite |
| Additions regrouped | The program's brackets are the pattern's: `add(V0,add(V1,V2))` adds the last two first. A reduction makes NumPy's pairwise additions | The judge, and a test at every length to 700 |
| Signed zeros, NaN, infinities, denormals | Whatever the instruction gives, which is what NumPy's loop gives | The judge's value classes |
| A result laid out differently, so that a later sum adds in another order | Operands are taken only where NumPy's result lies in the order the module walks, C order or the one other order every array lies in, and the output is laid out so | Tests that hold bits, strides and flags to NumPy's in every order of two to four dimensions, and of layouts the kernels do not read |
| An output that overlaps an input | Every load of a block comes before its store, so the result may be exactly one of the arrays, for a length the module chooses; any other overlap is declined before a kernel runs | Tests of a program written in place at every length around its blocks, and of the module's refusal |
| A warning or an error | The flags are read and NumPy's calls are made | Tests of each, with the line |

**One thing is not specified: which NaN comes of two.** When an operation meets two NaNs of different sign or payload, the processor returns its first operand's. Which operand is first in NumPy's loop is its compiler's choice, and a pattern does not record how the program ordered a sum. Whether a result is NaN is always NumPy's; which NaN it is, is NumPy's whenever a program has one kind of NaN, which is what arithmetic and `np.nan` give.

## What was measured

Everything below is from an Apple M1 Pro, NumPy 2.5.5, CPython 3.13, with the layer's nodes written by its C core. The kernels' specs are not on `main` yet, so the judge's runs are `--unlocked`. The tables of the programs from that machine were timed one way after another, NumPy's first, by the tool as it then was; [issue 323](https://github.com/sofeikov/hotkernels/issues/323) and [the findings on it](lazy-codebases-findings.md#issue-323-the-ways-timed-in-turn-by-default) have what that can move, and they could not be taken again without the machine.

### The kernels written by hand, alone

From `uv run judge run <kernel> --unlocked`. The NumPy column is the spec's oracle called from Python, the expression as a program writes it, so the speedup against it is what fusing that expression gains at that size, a call from Python included. Naive C is the compiled reference.

How many times faster than NumPy's passes, by the number of elements, and than the compiled C over all of the judge's sizes:

| Kernel | Operations | 4,096 | 65,536 | 1,048,576 | 16,777,216 | Over naive C |
|---|---|---|---|---|---|---|
| `fuse_add_scaled_f64` | 2 | 4.37x | 1.86x | 1.66x | 1.54x | 1.05x |
| `fuse_sub_scaled_f64` | 2 | 4.34x | 1.75x | 2.03x | 1.65x | 1.04x |
| `fuse_add_product_f64` | 2 | 3.21x | 1.54x | 1.60x | 1.36x | 1.03x |
| `fuse_sub_product_f64` | 2 | 3.24x | 1.53x | 1.70x | 1.43x | 1.03x |
| `fuse_product3_f64` | 2 | 3.25x | 1.53x | 1.58x | 1.36x | 1.03x |
| `fuse_scaled_sum_f64` | 2 | 4.04x | 1.89x | 1.56x | 1.56x | 1.04x |
| `fuse_mul_shifted_f64` | 2 | 4.35x | 1.92x | 1.69x | 1.54x | 1.05x |
| `fuse_mul_difference_f64` | 2 | 3.26x | 1.52x | 1.58x | 1.36x | 1.02x |
| `fuse_sum3_f64` | 2 | 3.22x | 1.53x | 1.56x | 1.36x | 1.02x |
| `fuse_sum_squares_f64` | 2 | 7.71x | 3.16x | 2.86x | 2.76x | 1.49x |
| `fuse_flux_limit_f64` | 4 | 6.98x | 3.16x | 3.00x | 2.47x | 6.52x |
| `fuse_laplacian5_f64` | 5 | 3.50x | 2.19x | 1.79x | 1.91x | 1.00x |
| `fuse_flux_update_f64` | 5 | 2.70x | 1.81x | 1.71x | 1.74x | 0.99x |
| `fuse_poisson_step_f64` | 8 | 5.05x | 3.13x | 2.72x | 2.72x | 1.00x |

- **Two operations in one pass are worth 1.4x to 2x on arrays that fit no cache, not 2x.** Both ways are limited by memory there, and a pass of NumPy's that reads two arrays and writes one is not twice the traffic of a kernel that reads three and writes one. A kernel with a number in place of an array (`a + s * b`) reads less and gains more than one with three arrays (`a + b * c`): 1.54x against 1.36x at 16.8 million elements.
- **More operations gain more, but less than their number.** Eight operations in one pass are 2.7x, five are 1.7x to 1.9x. The stencil kernels read five and six arrays, and reading is what they wait for.
- **At 4,096 elements the figure is mostly calls.** Three to seven times, of which a good part is what each call into NumPy from Python costs. Inside the layer a node costs more than that again, so the figure there is not smaller.
- **The assembly is level with the compiled C, with three exceptions.** Clang vectorises these loops as written, and beyond the first-level cache both wait for memory. The loads with a non-temporal hint are 1.05x to 1.2x ahead at 262,144 to 1,048,576 elements in the kernels that read two or three arrays, which is where the 2% to 5% over all sizes comes from; with five or six arrays the hint gains nothing that is significant. The flux limiter is 6.5x ahead: its C reference, a comparison that feeds a selection, is compiled to a loop of one element at a time. The sum of squares is 1.5x ahead of a C reference the compiler vectorises only in part.
- **Each kernel's log has its runs,** in `arm64/attempts.md` beside it, and the x86-64 runs in `x86_64/attempts.md`.

### The programs, four ways

`uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --size large --plain`. `hotkernels.install()` is in effect for every way but the first, since that is how the package is used: a call it has a kernel for is served by the kernel. **NumPy** is the program as it is, with nothing installed, and what every other way is measured against. **Installed** is the same eager program after `hotkernels.install()`, one operation at a time, with no graph: what the package gives today. **Lazy** is the program with the stand-in, every operation a node, computed one node at a time. **Fused** is the same with `fuse.enable()`: programs of kernels, small calls made at once. The results of the three installed ways are compared bit for bit before anything is printed. Times are the quickest of 5 runs, in milliseconds, from the start of the program to its last result, taken at load averages of 1.5 to 3.1, so they are rough to a few percent; a program's own NumPy time moved by up to 10% between runs of this table.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             38.55     38.91      38.46     0.24    38.22      35.00    30.22     0.67     4.12    1.10x    1.10x     1.11x
attention              28.65     28.56      28.64     0.44    28.20      28.88     1.05     0.13    27.71    0.99x    0.99x     0.99x
perceptron             32.36     25.33      55.83    24.16    31.67      27.70    27.34     0.04     0.31    2.02x    1.17x     0.91x
char_rnn               32.63     28.62      90.29    45.20    45.09      32.72    32.72     0.00     0.00    2.76x    1.00x     0.87x
particle_filter       358.14    320.63     332.63    84.96   247.67     250.32   102.01     2.80   145.52    1.33x    1.43x     1.28x
veros_eos             128.96    123.73     116.92     0.68   116.24      73.97     0.69     0.48    72.80    1.58x    1.74x     1.67x
veros_isoneutral      218.80    219.14     228.37     8.54   219.83     189.67    20.66     0.88   168.13    1.20x    1.15x     1.16x
veros_tke              99.66     97.04     100.92    23.11    77.81      87.93    28.47     0.52    58.94    1.15x    1.13x     1.10x
cavity_flow           155.31    154.48     169.43     7.52   161.91      55.44     7.65     2.83    44.95    3.06x    2.80x     2.79x
hdiff                  69.41     66.00      69.07     5.80    63.27      29.43     5.80     0.04    23.59    2.35x    2.36x     2.24x
black_scholes          87.29     86.84      88.47     1.88    86.58      85.49     1.88     0.04    83.57    1.03x    1.02x     1.02x
trimesh_mesh          174.86    137.24     137.39   137.29     0.10     137.29   137.29     0.00     0.00    1.00x    1.27x     1.00x
sklearn_metrics       978.27    272.56     278.66   277.32     1.35     277.37   277.37     0.00     0.00    1.00x    3.53x     0.98x
empyrical_stats      1259.40    327.30     327.84   327.84     0.00     327.32   327.32     0.00     0.00    1.00x    3.85x     1.00x
skimage_pipeline     1194.85   1113.44    1112.37  1112.37     0.00    1112.23  1112.23     0.00     0.00    1.00x    1.07x     1.00x
pandas_sales          474.16    443.03     443.75   443.75     0.00     442.98   442.98     0.00     0.00    1.00x    1.07x     1.00x
astropy_timeseries   1526.09   1295.04    1295.07  1293.20     1.87    1290.92  1289.81     0.08     1.04    1.00x    1.18x     1.00x
example                45.01      8.46       8.82     4.22     4.60       8.84     7.45     0.01     1.37    1.00x    5.09x     0.96x
flame                 140.35     88.19     166.28    86.31    79.97      91.04    88.66     0.03     2.34    1.83x    1.54x     0.97x
azimint_naive         171.71    154.52     168.94     6.62   162.32     155.33   155.33     0.00     0.00    1.09x    1.11x     0.99x
azimint_hist          122.10     71.16      71.26    71.22     0.04      71.57    71.57     0.00     0.00    1.00x    1.71x     0.99x
mandelbrot1           149.59    141.72     145.15     1.20   143.95     142.47   142.47     0.00     0.00    1.02x    1.05x     0.99x
mandelbrot2           161.51    145.84     167.32    23.96   143.36     147.12   147.12     0.00     0.00    1.14x    1.10x     0.99x
make_decision         143.15     85.28    1352.01  1039.20   312.81     116.80   116.80     0.00     0.00   11.58x    1.23x     0.73x
pandas_filter         174.40     33.92      33.88    33.88     0.00      33.90    33.90     0.00     0.00    1.00x    5.14x     1.00x
equalize_hist         141.24    126.55     126.75   126.67     0.08     126.65   126.65     0.00     0.00    1.00x    1.12x     1.00x
```

`build` is the time outside the layer's computing functions: the program's own Python, the nodes, and whatever is made at once. `compute` is the time inside them, less the planning.

`vs NumPy` is the figure the layer is judged by: the fused run with the kernels installed against the program on plain NumPy. `vs inst.` is the same run against the eager program with the same kernels, which says what the graph adds to what the package already gives.

The lazy and the fused run again with nothing installed, beside the same two eager runs. This says what the graph and its kernels do by themselves; it is not how the layer would be used.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             38.55     38.91      38.26     0.24    38.02      33.24    28.68     0.64     3.91    1.15x    1.16x     1.17x
attention              28.65     28.56      28.67     0.45    28.23      29.35     1.06     0.11    28.18    0.98x    0.98x     0.97x
perceptron             32.36     25.33      62.90    24.42    38.47      35.39    35.13     0.02     0.24    1.78x    0.91x     0.72x
char_rnn               32.63     28.62      93.86    45.25    48.60      36.53    36.53     0.00     0.00    2.57x    0.89x     0.78x
particle_filter       358.14    320.63     368.42    84.84   283.57     286.57   102.28     2.90   181.38    1.29x    1.25x     1.12x
veros_eos             128.96    123.73     117.18     0.67   116.50      73.80     0.70     0.48    72.62    1.59x    1.75x     1.68x
veros_isoneutral      218.80    219.14     227.57     8.61   218.95     190.29    21.98     0.87   167.45    1.20x    1.15x     1.15x
veros_tke              99.66     97.04     102.94    23.07    79.87      90.03    28.39     0.52    61.12    1.14x    1.11x     1.08x
cavity_flow           155.31    154.48     169.71     7.47   162.24      55.33     7.64     2.81    44.88    3.07x    2.81x     2.79x
hdiff                  69.41     66.00      72.57     5.80    66.77      29.47     5.82     0.03    23.62    2.46x    2.36x     2.24x
black_scholes          87.29     86.84      88.30     1.88    86.41      85.42     1.88     0.04    83.49    1.03x    1.02x     1.02x
trimesh_mesh          174.86    137.24     175.10   175.00     0.10     174.71   174.71     0.00     0.00    1.00x    1.00x     0.79x
sklearn_metrics       978.27    272.56    1056.80  1053.75     3.05    1054.41  1054.41     0.00     0.00    1.00x    0.93x     0.26x
empyrical_stats      1259.40    327.30    1289.58  1289.58     0.00    1289.75  1289.75     0.00     0.00    1.00x    0.98x     0.25x
skimage_pipeline     1194.85   1113.44    1192.11  1192.11     0.00    1191.55  1191.55     0.00     0.00    1.00x    1.00x     0.93x
pandas_sales          474.16    443.03     474.16   474.16     0.00     474.06   474.06     0.00     0.00    1.00x    1.00x     0.93x
astropy_timeseries   1526.09   1295.04    1505.51  1503.18     2.33    1514.93  1513.86     0.08     1.00    0.99x    1.01x     0.85x
example                45.01      8.46      45.30    24.52    20.77      45.36    43.81     0.01     1.54    1.00x    0.99x     0.19x
flame                 140.35     88.19     219.64   101.91   117.73     143.32   140.97     0.03     2.32    1.53x    0.98x     0.62x
azimint_naive         171.71    154.52     186.01     6.72   179.29     173.02   173.02     0.00     0.00    1.08x    0.99x     0.89x
azimint_hist          122.10     71.16     122.19   122.15     0.04     122.13   122.13     0.00     0.00    1.00x    1.00x     0.58x
mandelbrot1           149.59    141.72     152.96     1.16   151.80     150.02   150.02     0.00     0.00    1.02x    1.00x     0.94x
mandelbrot2           161.51    145.84     184.06    23.87   160.19     161.80   161.80     0.00     0.00    1.14x    1.00x     0.90x
make_decision         143.15     85.28    1412.19  1036.43   375.75     176.99   176.99     0.00     0.00    7.98x    0.81x     0.48x
pandas_filter         174.40     33.92     173.99   173.99     0.00     174.21   174.21     0.00     0.00    1.00x    1.00x     0.19x
equalize_hist         141.24    126.55     141.61   141.52     0.09     141.40   141.40     0.00     0.00    1.00x    1.00x     0.89x
```

What the graphs were, and what the fused run did with them:

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
regression              384       1      384      3200000        83     281     41     41      41     41      0       0       0      179.2      59.2
attention               220       1      220       262144       184      33      8      8      16     16      0       0       0      174.7     135.4
perceptron             8854     565       33       128000         6    7908      2      2       4      2      0       0       0      163.7       2.4
char_rnn              32423     202    28606        10000         0   27923      0      0       0      0      0       0       0      165.8       0.0
particle_filter        2537      41      480       400000      2356     160    422    422     462    500      0     338      19     1166.5     640.9
veros_eos               366       1      366      1082016       363       3     37     37     140    285      0       0       0     3142.2     675.2
veros_isoneutral        718       1      690      4328064       442      27    121    121     226    266      9      31       0     3230.5    1074.4
veros_tke               382       6      281      3246048       260      28     38     38      77    115     11      28       3     1316.1     448.4
cavity_flow            6920       1     6900       160801      6043      47    260    260     560   2980      0    1196       0     4129.6     304.6
hdiff                    40       1       40     10650240        29       0      4      4       4     15      0       1       0     1540.2     422.1
black_scholes            28       1       28      2000000        28       0      5      5       8     14      0       2       0      416.0     192.0
trimesh_mesh              2       1        2       245760         0       2      0      0       0      0      0       0       0        2.9       0.0
sklearn_metrics           7       4        2      1000000         0       7      0      0       0      0      0       0       0       11.4       0.0
empyrical_stats           0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
skimage_pipeline          0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
pandas_sales              0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
astropy_timeseries      235      52       11        62113        20     130      0      0       0      0      0       0       0       11.1       9.9
example                  13       2        9      2000000         3       9      0      0       0      0      0       1       0       68.1      18.0
flame                  7418     416       53       589824         6    6855      1      1       1      0      0       1       0      299.3       8.6
azimint_naive         10001       1    10001       200000         0    9001      0      0       0      0      0       0       0      601.6       0.0
azimint_hist              1       1        1         1000         0       1      0      0       0      0      0       0       0        0.0       0.0
mandelbrot1            1205       1     1205       220900         0     904      0      0       0      0      0       0       0     1126.2       0.0
mandelbrot2            1132     239       19       902500         0     565      0      0       0      0      0       0       0     1278.1       0.0
make_decision        491520       5   100000        81920         0  409600      0      0       0      0      0       0       0       83.9       0.0
pandas_filter             0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
equalize_hist             8       4        4      2890000         0       5      0      0       0      0      0       0       0        0.0       0.0
```

`deferred ops` to `largest node` are of the lazy run. `deferred fused` and `at once fused` are the operations the fused run still wrote as nodes, and the calls and writes it made as the program wrote them. `expr.` and `tiled` are the expressions found in the fused run's graphs and those a program was planned for; `fused nodes` the nodes computed inside a program; `given up` the programs not run, or dropped. `writes in place` and `copied` are the fused run's deferred writes. `MB written` is the results the calls wrote.

### Small sizes

The same at each program's small size, where arrays have hundreds to tens of thousands of elements, with the kernels installed:

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression              0.47      0.36       1.17     0.54     0.64       0.48     0.48     0.00     0.00    2.47x    0.99x     0.77x
attention               1.25      1.16       1.74     0.43     1.31       1.55     1.55     0.00     0.00    1.12x    0.80x     0.75x
perceptron              4.30      3.43       7.18     2.84     4.34       3.72     3.45     0.02     0.24    1.93x    1.16x     0.92x
char_rnn                6.58      5.88      18.19    10.66     7.54       7.24     7.24     0.00     0.00    2.51x    0.91x     0.81x
particle_filter         4.26      3.73       6.25     2.43     3.82       4.39     4.39     0.00     0.00    1.42x    0.97x     0.85x
veros_eos               6.56      6.58       7.12     0.30     6.82       5.80     0.35     0.49     4.95    1.23x    1.13x     1.14x
veros_isoneutral       14.98     14.97      16.32     1.52    14.80      13.98     1.82     0.71    11.45    1.17x    1.07x     1.07x
veros_tke               7.30      7.16       8.08     2.24     5.84       7.97     2.69     0.45     4.83    1.01x    0.92x     0.90x
cavity_flow             5.53      5.14      13.61     6.15     7.46       5.48     5.48     0.00     0.00    2.48x    1.01x     0.94x
hdiff                   1.81      1.72       1.79     0.25     1.53       0.90     0.22     0.03     0.65    1.98x    2.01x     1.91x
black_scholes           0.23      0.22       0.30     0.08     0.22       0.25     0.25     0.00     0.00    1.20x    0.92x     0.88x
trimesh_mesh           11.36      9.49       9.57     9.53     0.04       9.52     9.52     0.00     0.00    1.01x    1.19x     1.00x
sklearn_metrics        48.68     19.54      20.98    20.83     0.15      20.38    20.38     0.00     0.00    1.03x    2.39x     0.96x
empyrical_stats      1248.41    356.94     356.95   356.95     0.00     357.11   357.11     0.00     0.00    1.00x    3.50x     1.00x
skimage_pipeline       99.72     91.69      91.41    91.41     0.00      91.34    91.34     0.00     0.00    1.00x    1.09x     1.00x
pandas_sales           30.25     28.74      28.66    28.66     0.00      28.75    28.75     0.00     0.00    1.00x    1.05x     1.00x
astropy_timeseries    207.25    190.45     192.43   191.48     0.94     191.10   191.10     0.00     0.00    1.01x    1.08x     1.00x
example                44.99      8.48       8.84     4.23     4.61       8.84     7.45     0.01     1.38    1.00x    5.09x     0.96x
flame                  16.94      9.73      21.71    12.02     9.69      10.46     9.99     0.02     0.45    2.07x    1.62x     0.93x
azimint_naive          20.19     18.42      32.48     6.76    25.72      19.03    19.03     0.00     0.00    1.71x    1.06x     0.97x
azimint_hist           12.32      7.16       7.29     7.26     0.03       7.24     7.24     0.00     0.00    1.01x    1.70x     0.99x
mandelbrot1            15.71     14.81      17.41     1.10    16.32      14.98    14.98     0.00     0.00    1.16x    1.05x     0.99x
mandelbrot2            16.31     14.53      21.95     5.44    16.51      14.85    14.85     0.00     0.00    1.48x    1.10x     0.98x
make_decision          14.83      8.55      74.53    45.84    28.69      11.93    11.93     0.00     0.00    6.25x    1.24x     0.72x
pandas_filter          17.55      3.57       3.56     3.56     0.00       3.57     3.57     0.00     0.00    1.00x    4.92x     1.00x
equalize_hist          12.97     11.55      11.78    11.73     0.06      11.69    11.69     0.00     0.00    1.01x    1.11x     0.99x
```

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
regression              954       1      954         1600         0     904      0      0       0      0      0       0       0        0.4       0.0
attention               550       1      550         2048         0     520      0      0       0      0      0       0       0        3.0       0.0
perceptron              958      61       33       128000         6     852      2      2       4      2      0       0       0       23.6       2.4
char_rnn               5585     202     2497        10000         0    5135      0      0       0      0      0       0       0       17.9       0.0
particle_filter        1267      21      240         8000         0     940      0      0       0      0      0       0       0       11.7       0.0
veros_eos               366       1      366        72171       363       3     39     39     142    285      0       0       0      209.6      45.0
veros_isoneutral        718       1      690       288684       454      31    135    135     206    222     19      31       0      196.6      94.8
veros_tke               382       6      281       216513       232      50     37     37      53     73     17      28       2       82.5      44.0
cavity_flow            5900       1     5875         3721         0    3850      0      0       0      0      0       0       0       81.8       0.0
hdiff                    40       1       40       261360        29       0      4      4       4     15      0       1       0       36.8      10.1
black_scholes            28       1       28         4096         0      28      0      0       0      0      0       0       0        0.9       0.0
trimesh_mesh              2       1        2        15360         0       2      0      0       0      0      0       0       0        0.2       0.0
sklearn_metrics           7       4        2        50000         0       7      0      0       0      0      0       0       0        0.6       0.0
empyrical_stats           0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
skimage_pipeline          0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
pandas_sales              0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
astropy_timeseries      235      52       11        12451         0     150      0      0       0      0      0       0       0        2.2       0.0
example                  13       2        9      2000000         3       9      0      0       0      0      0       1       0       68.1      18.0
flame                  1870     108       53       147456         4    1726      1      1       1      0      0       0       0       31.3       2.0
azimint_naive         10001       1    10001        20000         0    9001      0      0       0      0      0       0       0       60.2       0.0
azimint_hist              1       1        1         1000         0       1      0      0       0      0      0       0       0        0.0       0.0
mandelbrot1            1205       1     1205        22500         0     904      0      0       0      0      0       0       0      114.4       0.0
mandelbrot2            1132     239       19        90000         0     565      0      0       0      0      0       0       0      126.9       0.0
make_decision         49152       1    49152         8192         0   40960      0      0       0      0      0       0       0        8.4       0.0
pandas_filter             0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
equalize_hist             8       4        4       262144         0       5      0      0       0      0      0       0       0        0.0       0.0
```

## What the measurements show

**Every program, by what the fused run with the kernels installed is against plain NumPy** at the large size; what the eager program gains from the same kernels; and the fused run against that eager program. The last column is the fused run with nothing installed, against plain NumPy.

| Program | Installed, eager, against NumPy | Fused and installed against NumPy | Fused and installed against installed | Fused alone against NumPy | What it is made of |
|---|---|---|---|---|---|
| `pandas_filter` | 5.14x | 5.14x | 1.00x | 1.00x | pandas, on plain arrays |
| `example` | 5.32x | 5.09x | 0.96x | 0.99x | Single calls on two million elements |
| `empyrical_stats` | 3.85x | 3.85x | 1.00x | 0.98x | empyrical, on plain arrays |
| `sklearn_metrics` | 3.59x | 3.53x | 0.98x | 0.93x | scikit-learn, handed lazy arrays |
| `cavity_flow` | 1.01x | 2.80x | 2.79x | 2.81x | Stencil expressions of 6 to 24 operations, each assigned into part of a grid |
| `hdiff` | 1.05x | 2.36x | 2.24x | 2.36x | Four expressions of a stencil, each one kernel written by hand |
| `veros_eos` | 1.04x | 1.74x | 1.67x | 1.75x | Polynomials, one of 153 operations |
| `azimint_hist` | 1.72x | 1.71x | 0.99x | 1.00x | `np.histogram` |
| `flame` | 1.59x | 1.54x | 0.97x | 0.98x | 7,400 operations on arrays of 25,000 elements |
| `particle_filter` | 1.12x | 1.43x | 1.28x | 1.25x | Random draws, `arctan2`, `exp`; expressions on columns of a table, and on its rows of two and four elements |
| `trimesh_mesh` | 1.27x | 1.27x | 1.00x | 1.00x | trimesh, on plain arrays |
| `make_decision` | 1.68x | 1.23x | 0.73x | 0.81x | 490,000 operations on arrays of 32 |
| `astropy_timeseries` | 1.18x | 1.18x | 1.00x | 1.01x | astropy, on plain arrays |
| `perceptron` | 1.28x | 1.17x | 0.91x | 0.91x | 8,900 operations, most on arrays of 8,000 elements |
| `veros_isoneutral` | 1.00x | 1.15x | 1.16x | 1.15x | Expressions on slices of a grid; 9 of 121 are given up for arrays that step |
| `veros_tke` | 1.03x | 1.13x | 1.10x | 1.11x | A tridiagonal solve, masks, expressions on arrays that step |
| `equalize_hist` | 1.12x | 1.12x | 1.00x | 1.00x | scikit-image's histogram equalisation |
| `azimint_naive` | 1.11x | 1.11x | 0.99x | 0.99x | A thousand masks over 200,000 elements |
| `regression` | 0.99x | 1.10x | 1.11x | 1.16x | A matrix product, a residual, a squared loss |
| `mandelbrot2` | 1.11x | 1.10x | 0.99x | 1.00x | Complex arithmetic on arrays that shrink to nothing |
| `skimage_pipeline` | 1.07x | 1.07x | 1.00x | 1.00x | scikit-image, on plain arrays |
| `pandas_sales` | 1.07x | 1.07x | 1.00x | 1.00x | pandas, on plain arrays |
| `mandelbrot1` | 1.06x | 1.05x | 0.99x | 1.00x | Complex arithmetic through a mask |
| `black_scholes` | 1.01x | 1.02x | 1.02x | 1.02x | `erf`, `log`, `exp`; arithmetic between them |
| `char_rnn` | 1.14x | 1.00x | 0.87x | 0.89x | 32,000 operations on arrays of 100 to 10,000 |
| `attention` | 1.00x | 0.99x | 0.99x | 0.98x | float32 matrix products and `tanh` |

- **The two kinds of kernel add up, and neither gets in the other's way.** Six of the seven programs fusion speeds up gain 6% or nothing from `install()` eagerly, because no kernel of the package is in their expressions, and their fused runs are within 2% with it and without. The programs that are a library's work or single calls gain 1.07x to 5.3x from `install()` eagerly and nothing from fusion, and their fused runs keep the kernels' gain to within 4%. The particle filter has both: 322 ms eagerly with the kernels for 359 ms without, 289 ms fused without them, and 253 ms with both. The regression's and the transformer block's eager times moved by 3% between the two eager runs of this table, and that is all their first column says.
- **Where the fused run is behind the eager program with the same kernels, it is by what a call made at once costs.** `make_decision`, the recurrent network and the perceptron make 410,000, 28,000 and 7,900 calls at once and write no node: 117 ms, 32.7 ms and 27.7 ms against 85 ms, 28.6 ms and 25.3 ms. The example is 4% behind, 0.4 ms, and the fractal flame 3%, 3 ms. The naive azimuthal integration was 4% behind as well, 7 ms, most of it arithmetic on a number that came out of an array, `rmax * i / npt`, which went through the operator's ufunc at 600 ns an operation where the eager program's scalar has its own in 50 ns. It is the scalar's own now ([lazy-design.md](lazy-design.md#the-c-core)), and the program runs level.
- **A kernel the lazy run does not reach shows in the same column.** The second Mandelbrot program takes its grids from `np.mgrid`, which gave plain `ndarray`s, and selects from them by a lazy mask: NumPy's own indexing, with the package's kernel for `x[mask]` never asked. It ran in 172 ms fused with the kernels installed, the same as without them, where the eager program runs in 148 ms with them. With the arrays `np.mgrid`, `np.ogrid`, `np.r_` and `np.c_` make lazy ([lazy-design.md](lazy-design.md#the-stand-in-for-the-module)) it runs in 148 ms, for the eager program's 147 ms. The package's own count of the calls it served was then compared, eager run against fused, for eleven other programs, and none loses a kernel that way: the counts differ by the layer's dry runs on one element, by the calls a fused kernel made in a package kernel's place, and for scikit-learn by the library's own different road.
- **One pass for a whole expression is worth what its passes were.** The cavity flow's fused run writes 0.3 GB of new arrays where the node-by-node run writes 4.1 GB, and computes in 45 ms what took 163 ms. Its 560 kernel calls are 260 programs; 2,980 of its 6,900 nodes have no array of their own. The equation of state writes 0.7 GB instead of 3.1 GB and computes in 72 ms instead of 116 ms: 153 operations in 64 kernel calls, all on blocks.
- **A write in place is the other half of the cavity flow.** Its 1,220 writes into slices and boundary rows are made where the array is: all of them in the lazy run, and in the fused one 1,196 by their programs and 24 at once. As copies they were 54 ms.
- **A kernel for every step is what covers a long tail.** Sixty-nine of the eighty kernels ran over the corpus, the single operations among the most called: `add(S0,V0)` 67 times, `mul(S0,V0)` 63, `sub(V0,S0)` 56. Each of those is a step inside a block, where otherwise it is a pass of NumPy's between two kernels.
- **The particle filter runs 1.43x, most of fusion's part on rows of two and four elements.** Its table is 100,000 rows of four numbers, and NumPy makes `particles - mean`, `centred * weights[:, None]` and `particles[:, :2] += particles[:, 2:] * dt` a row at a time: 544 µs, 294 µs and 1,256 µs, where the same arithmetic in one run takes 101 µs, 101 µs and 82 µs. The first is one kernel's call now and the others the module's loop over the rows ([Computing a group](#computing-a-group)), and the expressions on the table's columns are gathered into blocks. With the rows left to NumPy the fused run was 2% ahead of NumPy with nothing installed, where it is 25% ahead. Nineteen of its writes into the table go to a copy, because an earlier reader of the table still waits.
- **Planning is 5% of the cavity flow's fused run,** 2.8 ms of 55 ms for 6,900 nodes, with the expressions read by the C core; read in Python it was 15 ms of 69 ms. For every other program it is about 1% or less.
- **A program of masks or of complex numbers writes no graph.** The naive azimuthal integration makes 9,000 calls at once, comparisons and selections on 200,000 elements, and is 10% ahead of NumPy, level with the eager program with the kernels. The two Mandelbrot programs are 1% behind theirs.
- **At the small sizes the horizontal diffusion, the equation of state and isoneutral mixing gain from fusion,** 2.01x, 1.13x and 1.07x against NumPy, and the programs that are a library's work keep the kernels' gain, up to 5.1x. Six programs are behind NumPy there, at 0.81x to 0.98x, and against the eager program with the kernels the fused run is from 0.72x to 1.91x: at these sizes most calls are small, and each pays what a call made at once costs.
- **Arrays that step are gathered where the program is long enough to pay for the copies.** The two Veros routines keep their state with a time level as its last axis, so `tke[:, :, :, tau]` is every third element. The kinetic energy's advection flux, seventeen operations with its Courant number on three such arrays, is two kernels written for its halves and one copy of each array, and the routine runs in 88 ms where it ran in 93 ms. 11 of its 38 programs and 9 of isoneutral mixing's 121 are still given up: one to five operations, where the copies cost what the passes save. Isoneutral mixing's sixteen stores of a product into every fourth element, which were given up as well, are a loop of the module's own, and the routine runs in 190 ms where it ran in 197 ms. [What is left](#what-is-left) has the measurements.
- **Calls into a mathematics library are the wall for Black-Scholes and the transformer block.** `erf`, `log` and `exp` are 86% of the first, `tanh` and float32 arithmetic most of the second.
- **A library handed a lazy array may take longer over it.** scikit-learn's metrics are 2% slower on lazy arrays than on `ndarray`s with the kernels installed, and 7% with nothing installed, whatever the layer does with the seven operations of its own.
- **How long a plan lives matters more than how good it is.** A reference kept past a computation costs a copy or a page fault where NumPy would have reused memory: a cycle of references in a planner costs 46,000 page faults a run, and a record of the part written into, kept by a group that has been given up, a copy of the array for each of 21 writes. Both are tested for.


## What is left

This section answers one question with measurements: what could fusing by pattern still gain on these twenty-six programs, run as the package is used, with `hotkernels.install()` in effect. Four of them run at or below what the tool described here takes for the best any fusion could do. Fifteen have 10% at most to gain from any fusion at all. Four are on arrays below the limit at which anything is written down. Three have something left on large arrays, and for those it says in which operations, what a loop written for exactly that could do at best, and what was tried.

### How far any fusion could go

Fusion removes two things from an operation on arrays: the pass over memory that reads its operands and writes its result, and the call into NumPy. It cannot remove the arithmetic. So every operation has a part no fusion touches, and it can be measured: the same NumPy loop on arrays that fit the first-level cache, per element, with the cost of the call taken off.

[`programs/limit.py`](../prototypes/lazy/programs/limit.py) does that for every program. It runs the program lazily with every call timed, puts each operation in a class by what a kernel would need to make it, and times each different operation in the cache on pieces of the operands the program itself gave it, since what `erf` or `exp` costs depends on its argument. Everything but the first column is measured with the single-operation kernels installed. The **floor** of a program is its installed eager time less everything that could be removed from its element-by-element operations and its reductions over every element: what a fusion of all of them, of any kind, at no cost of its own, would leave.

`uv run --group apps --group workloads python -m prototypes.lazy.programs.limit --size large`:

```text
How far fusion could take each program (large size, the quickest of 5 runs, milliseconds)

program                NumPy installed     floor     fused  at best       now     left   left of
                                                                                       fused run
regression             42.69     41.62     32.58     36.02    1.31x     1.19x     3.43      9.5%
attention              28.80     28.89     26.73     29.02    1.08x     0.99x     2.30      7.9%
perceptron             32.07     25.18     20.71     27.75    1.55x     1.16x     7.04     25.4%
char_rnn               32.37     28.47     21.89     32.56    1.48x     0.99x    10.67     32.8%
particle_filter       364.18    325.48    254.44    253.97    1.43x     1.43x    -0.47     -0.2%
veros_eos             129.95    127.41    111.25     74.43    1.17x     1.75x   -36.82    -49.5%
veros_isoneutral      221.07    220.56    171.17    190.04    1.29x     1.16x    18.87      9.9%
veros_tke             101.65     97.92     73.28     89.07    1.39x     1.14x    15.79     17.7%
cavity_flow           158.22    157.69    116.08     56.20    1.36x     2.82x   -59.88   -106.5%
hdiff                  69.95     66.40     62.22     29.75    1.12x     2.35x   -32.47   -109.2%
black_scholes          88.27     87.42     84.64     86.03    1.04x     1.03x     1.39      1.6%
trimesh_mesh          175.50    137.26    137.24    137.08    1.28x     1.28x    -0.16     -0.1%
sklearn_metrics       980.95    274.46    274.41    279.19    3.57x     3.51x     4.78      1.7%
empyrical_stats      1289.00    328.06    328.06    328.08    3.93x     3.93x     0.02      0.0%
skimage_pipeline     1205.52   1119.40   1119.40   1117.58    1.08x     1.08x    -1.82     -0.2%
pandas_sales          476.44    444.19    444.19    443.74    1.07x     1.07x    -0.45     -0.1%
astropy_timeseries   1511.45   1282.49   1282.30   1282.64    1.18x     1.18x     0.33      0.0%
example                45.01      8.51      8.21      8.86    5.48x     5.08x     0.65      7.4%
flame                 140.30     87.98     79.19     90.55    1.77x     1.55x    11.36     12.5%
azimint_naive         171.77    154.57    149.22    156.43    1.15x     1.10x     7.21      4.6%
azimint_hist          122.12     71.24     71.23     71.30    1.71x     1.71x     0.07      0.1%
mandelbrot1           148.62    141.19    129.83    141.84    1.14x     1.05x    12.01      8.5%
mandelbrot2           161.05    145.74    145.34    146.77    1.11x     1.10x     1.43      1.0%
make_decision         143.50     85.55         -    117.95        -     1.22x        -         -
pandas_filter         174.44     34.03     34.03     33.91    5.13x     5.14x    -0.12     -0.4%
equalize_hist         141.27    126.43    126.42    126.51    1.12x     1.12x     0.09      0.1%

`NumPy` is the program as it is; every other time is under `hotkernels.install()`. `floor` is the installed eager
time less everything that could be removed from the program's element-by-element operations and its reductions: what
fusing all of them, at no cost, would leave. `at best` is NumPy over the floor, `now` NumPy over the fused run, and
`left` the fused run less the floor. A program with `-` for a floor has none that means
anything: its arrays are so small that what could be removed from an operation, the call, is all the operation costs.

Where the time is with the kernels installed, node by node, and how much of each class could be removed

program              outside        arithmetic             masks           library       other types        reductions          products             moves              rest
                   the graph    ms   removable    ms   removable    ms   removable    ms   removable    ms   removable                ms                ms                ms
regression              3.55     7.35     3.61     0.00     0.00     0.00     0.00     0.00     0.00     9.39     5.43             21.30              0.00              0.04
attention               0.16     2.57     0.51     0.00     0.00    17.03     0.16     1.96     1.09     1.12     0.41              5.92              0.00              0.12
perceptron              0.00     5.32     3.30     0.75     0.22     1.06     0.27     0.00     0.00     4.54     0.68              8.38              7.06              0.31
char_rnn                0.00     3.93     3.28     0.00     0.00     2.21     1.60     0.00     0.00     1.70     1.70             14.84             18.36              1.73
particle_filter        75.90    92.11    70.31     1.50     0.12    64.91     0.30     0.97     0.09     1.96     0.21             13.40             62.03             12.71
veros_eos              10.71   110.53    16.14     0.00     0.00     6.17     0.02     0.00     0.00     0.00     0.00              0.00              0.00              0.00
veros_isoneutral        0.23   140.32    49.39     0.00     0.00    52.36     0.00     0.00     0.00     0.00     0.00              0.00             27.52              0.12
veros_tke              19.35    57.65    21.96    10.19     2.66     0.00     0.00     0.02     0.02     0.00     0.00              0.00             10.62              0.08
cavity_flow             0.00   148.24    41.60     0.00     0.00     4.79     0.00     0.00     0.00     0.00     0.00              0.00             11.25              0.47
hdiff                   2.76    48.39     2.34    13.33     1.83     0.00     0.00     0.00     0.00     0.00     0.00              0.00              1.91              0.01
black_scholes           0.00    12.73     1.76     0.00     0.00    74.39     1.02     0.00     0.00     0.00     0.00              0.00              0.82              0.00
trimesh_mesh          137.17     0.00     0.00     0.00     0.00     0.03     0.01     0.06     0.01     0.00     0.00              0.00              0.00              0.00
sklearn_metrics       273.22     0.00     0.00     0.51     0.04     0.00     0.00     0.00     0.00     0.74     0.01              0.00              0.00              0.00
empyrical_stats       328.06     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
skimage_pipeline     1119.40     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
pandas_sales          444.19     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
astropy_timeseries   1281.28     0.32     0.13     0.00     0.00     0.00     0.00     0.59     0.00     0.13     0.06              0.00              0.14              0.02
example                 3.98     0.48     0.10     1.35     0.00     0.35     0.00     0.17     0.04     0.78     0.16              0.00              0.86              0.56
flame                  13.45     7.23     4.06     1.74     0.74    16.68     3.43     1.06     0.53     0.05     0.03              0.00             47.76              0.00
azimint_naive           0.00     3.56     3.56   142.04     0.00     0.00     0.00     0.00     0.00     1.81     1.79              0.00             14.37              0.00
azimint_hist           71.23     0.00     0.00     0.00     0.00     0.00     0.00     0.01     0.01     0.00     0.00              0.00              0.00              0.00
mandelbrot1             0.00     0.00     0.00    11.13     0.00     0.00     0.00    79.41    11.36     0.00     0.00              0.00             52.47              0.00
mandelbrot2             5.70     0.00     0.00     7.55     0.17     0.00     0.00    46.59     0.22     0.00     0.00              0.00             85.88              0.01
make_decision           0.00    21.50    21.19     0.00     0.00     0.00     0.00    76.52    70.34    12.12    12.00              0.00            151.32              0.00
pandas_filter          34.03     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
equalize_hist         126.41     0.00     0.00     0.00     0.00     0.00     0.00     0.01     0.01     0.00     0.00              0.00              0.00              0.01

`outside the graph` is the eager time that is no node's: the program's own code, a library's, and calls the layer
does not defer. A class's `removable` is its time less what the same NumPy loops take for as many elements on arrays
of 512 to 4096 elements, with the cost of the call left out: the passes over memory and the calls.

What the kernels took for the nodes they computed

program                 nodes   kernels ms     NumPy ms  in cache ms
regression                 41         3.33         7.35         3.74
attention                  16         0.84         2.57         2.07
perceptron                  2         0.10         5.32         2.02
char_rnn                    0         0.00         3.93         0.65
particle_filter           500        44.75        92.11        21.80
veros_eos                 285        55.61       110.53        94.39
veros_isoneutral          266       101.57       140.32        90.93
veros_tke                 115        36.39        57.65        35.69
cavity_flow              2980        37.15       148.24       106.63
hdiff                      15        18.66        48.39        46.04
black_scholes              14         7.63        12.73        10.97
trimesh_mesh                0         0.00         0.00         0.00
sklearn_metrics             0         0.00         0.00         0.00
empyrical_stats             0         0.00         0.00         0.00
skimage_pipeline            0         0.00         0.00         0.00
pandas_sales                0         0.00         0.00         0.00
astropy_timeseries          0         0.00         0.32         0.20
example                     0         0.00         0.48         0.38
flame                       0         0.45         7.23         3.17
azimint_naive               0         0.00         3.56         0.00
azimint_hist                0         0.00         0.00         0.00
mandelbrot1                 0         0.00         0.00         0.00
mandelbrot2                 0         0.00         0.00         0.00
make_decision               0         0.00        21.50         0.30
pandas_filter               0         0.00         0.00         0.00
equalize_hist               0         0.00         0.00         0.00

`NumPy ms` is what NumPy took for every operation of the class a kernel can make, and `in cache ms` what its loops
take for as many elements in the cache.

What NumPy still makes in the fused run, and how much of that could be removed

program              kernels             arithmetic                 masks               library             other types              the rest of      could
                          ms                    ms  removable         ms  removable         ms  removable         ms  removable    the run ms   still go
regression              3.33                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00         32.69       0.00
attention               0.84                  1.30       0.52       0.00       0.00      17.41       0.58       1.83       1.07          6.56       2.57
perceptron              0.10                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00         27.64       0.00
char_rnn                0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00         32.56       0.00
particle_filter        44.75                  0.57       0.26       1.45       0.07      58.84       0.28       0.00       0.00        146.81       0.87
veros_eos              55.61                 11.59       2.42       0.00       0.00       6.18       0.02       0.00       0.00          1.06       2.45
veros_isoneutral      101.57                 12.81       7.14       0.00       0.00      56.08       3.86       0.00       0.00         19.58      11.00
veros_tke              36.39                  9.42       4.88       6.57       0.78       0.00       0.00       0.00       0.00         36.68       5.67
cavity_flow            37.15                  0.00       0.00       0.00       0.00       4.67       0.24       0.00       0.00         14.38       0.24
hdiff                  18.66                  5.38       0.00       0.00       0.00       0.00       0.00       0.00       0.00          5.71       0.00
black_scholes           7.63                  2.69       0.44       0.00       0.00      74.75       1.38       0.00       0.00          0.96       1.82
trimesh_mesh            0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        137.08       0.00
sklearn_metrics         0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        279.19       0.00
empyrical_stats         0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        328.08       0.00
skimage_pipeline        0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00       1117.58       0.00
pandas_sales            0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        443.74       0.00
astropy_timeseries      0.00                  0.21       0.01       0.00       0.00       0.00       0.00       0.67       0.08       1281.76       0.09
example                 0.00                  0.47       0.09       0.68       0.00       0.00       0.00       0.00       0.00          7.71       0.10
flame                   0.45                  0.08       0.03       0.00       0.00       1.22       0.02       0.00       0.00         88.80       0.05
azimint_naive           0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        156.43       0.00
azimint_hist            0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00         71.30       0.00
mandelbrot1             0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        141.84       0.00
mandelbrot2             0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        146.77       0.00
make_decision           0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        117.95       0.00
pandas_filter           0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00         33.91       0.00
equalize_hist           0.00                  0.00       0.00       0.00       0.00       0.00       0.00       0.00       0.00        126.51       0.00

The fused run in three parts. `kernels` is the time inside its programs. The classes are the calls NumPy still
makes for nodes no program computed, with what of each would go if it cost what its loop costs in the cache.
`the rest of the run` is everything else: the program's own code, a library's, matrix products, moving elements,
and the layer. `could still go` is the sum of the removable parts: what a fusion of all that is left, at no cost,
would take off the fused run.
```

- **The floor is not a floor for a kernel.** It is what NumPy's own loops would take with no pass over memory and no call. A kernel makes several operations for each load and store, which NumPy's loops cannot, so the cavity flow, the horizontal diffusion and the equation of state run below it: the cavity flow's kernels took 37.2 ms for arithmetic that NumPy's loops need 106.6 ms for in the cache. The particle filter is at it.
- **So the floor says little about a run that already has kernels in it, and the last table is of that run itself.** It has three parts: the time inside the kernels' programs, the calls NumPy still makes for operations a kernel could make, and everything else. The last column is what those calls would lose if each cost what its loop costs in the cache: the most a fusion of everything not yet fused could take off, if it cost nothing. It is 11.0 ms for isoneutral mixing and 5.7 ms for the kinetic energy, 6% of each; 2.6 ms for the transformer block, 2.5 ms for the equation of state, 1.8 ms for Black-Scholes; and under half a percent of the run for the other twenty-one.
- **Fifteen programs have 10% at most to gain by the floor,** and nothing by the last table. Their time is outside the graph, in a library or in calls the layer does not defer, or in a mathematics library, matrix products, or moving elements: a selection by a mask or by positions, a write through one.
- **Four programs are on arrays below the limit.** `make_decision`, on arrays of 32 elements, has no floor that means anything: what the tool finds removable there is the call into NumPy, which is nearly all such an operation costs. The floors of the recurrent network, the perceptron and the fractal flame, whose arrays have 100 to 25,000 elements, are low for the same reason; see [small arrays](#small-arrays-the-graph-costs-more-than-the-program) below.
- **The regression's figures move with its matrix products,** which are 26 ms of its 38 ms. It read 1.37x against NumPy in the run before this one and reads 1.10x here; the fused run is 3.5 ms ahead of the node-by-node run in this one and 4.0 ms in that. Its last table has nothing left in it.
- **What the tool finds removable moves between runs** for an operation whose cost depends on its values: the complex arithmetic of the first Mandelbrot program came out at 11 ms to 12 ms in four runs and 20 ms in another.

### Where the rest is, class by class

| What is left | In which programs | How much | Why a pattern does not take it |
|---|---|---|---|
| Calls into the platform's mathematics library: `exp`, `log`, `erf`, `tanh`, `arctan2`, `%`, `hypot` | Black-Scholes, isoneutral mixing, the particle filter, the fractal flame, the transformer block | 226 ms over the five, of which the tool finds 7.8 ms removable | It costs the same in the cache as out of it. A kernel is a leaf function and calls nothing; a second implementation would not have the library's bits |
| Calls on small arrays | `make_decision`, the recurrent network, the perceptron | All of what they are behind the eager program with the kernels: 32 ms of 117 ms, 4.1 ms of 32.7 ms, 2.4 ms of 27.7 ms | Nothing is written down for them, so there is nothing to match. Writing a node costs more than the call it would save |
| Arrays that step over elements, in rows of 25 or 26, in expressions of one to five operations | The two Veros routines, whose state has a time level as its last axis | 11 of the kinetic energy's 38 programs and 9 of isoneutral mixing's 121 are given up: at most 5.7 ms and 11.0 ms by the last table | A copy for each such array costs what the passes saved, and a loop that reads them where they lie is 1.14 times NumPy at best; see below |
| Complex arithmetic | The first Mandelbrot program | The tool finds 11 to 20 ms removable of 81 ms | A loop written for it gains nothing: see below |
| Comparisons and selections standing alone | The kinetic energy, the naive azimuthal integration, the horizontal diffusion | 2.5 ms of 10.2 ms, 1.4 ms of 151 ms, 1.9 ms of 13.3 ms | A mask is a byte for an element, which the float64 kernels neither read nor write, and its pass moves an eighth of the memory |
| float32 | The transformer block | 1.1 ms of 2.0 ms | No kernel is written for float32, and the time is in the matrix products and `tanh` |

### The best a kernel for it could do

The tool's figure is what would go if a fusion cost nothing. What a kernel could really do is found by writing the loop: one C loop for exactly the thing that is left, compiled with contraction off so that every operation is rounded by itself, timed against NumPy on the program's own layout and compared bit for bit. None of these loops is in the code; each says whether its kernels would be worth writing.

| What the loop is for | NumPy | The loop | What it says |
|---|---|---|---|
| `Z[I]**2 + C[I]` on complex128 as one pass, in the first Mandelbrot program run whole | 149.1 to 149.6 ms | 149.5 ms | Nothing to gain: the program's time is in `abs`, the masks and the selections, and the two passes fused are over what a mask selected. And the loop's bits are not NumPy's, whose complex power is another formula |
| One loop for each whole expression, however many arrays it reads, in place of a program of kernels: the equation of state's 35 expressions on arrays in one run | 25.9 ms by the kernels' programs | 29.2 ms | Larger kernels are not better in general. A loop over a dozen arrays reads them all from memory at once, where a program reads two or three at a step and keeps the rest in the cache |
| The same for Black-Scholes, the regression and isoneutral mixing | 4.8 ms, 1.9 ms and 12.6 ms by the kernels' programs | 3.2 ms, 1.9 ms and 8.0 ms | 1.6 ms of 87 ms, nothing, and 4.6 ms of 200 ms: under what a run of these programs moves by |
| The advection flux of the kinetic energy, thirteen operations, its three arrays that step read where they lie | 4,439 µs | 1,705 µs | 2.6 times, and the layer now has most of it: two kernels written for the flux's halves and each array gathered once make it in 2.85 ms for NumPy's 4.64 ms on the routine's own layout, result stored and all |
| `(a - b) * m` with `a` and `b` at a step, the commonest expression the two Veros routines give up | 1,151 µs | 1,006 µs | 1.14 times: 0.15 ms an expression, so about 2 ms in the kinetic energy and 3 ms in isoneutral mixing, 2% of each, if every expression they give up were this one |
| `Ai[..., ip, kr] = taper * sx * mask[...]`, a product of three stored into every fourth element in rows of 25: sixteen lines of isoneutral mixing | 1,537 µs | 985 µs | 1.56 times, and the layer has it: the module's own loop makes the statement in 1,001 µs, and isoneutral mixing runs 5 to 9 ms sooner ([Computing a group](#computing-a-group)) |
| An array that steps copied into a run, by itself | | 458 µs | What a copy costs before any arithmetic: a third of NumPy's two passes over it. It is why gathering loses on one kernel and gains little on thirteen operations |
| `np.where(x > 0, a, b)` with the comparison inside | 842 µs | 545 µs | 1.5 times on what the tool puts at 2.5 ms in the kinetic energy: 1 ms of 93 ms |

### What was built on it and did not gain

Each of these was built in the layer and timed, and none is in the code. The figures are from the same machine as the tables, in runs with nothing installed: each compares two builds of the module or the core on the same program.

- **A mathematics library called a block at a time.** Black-Scholes was written with NumPy's own loops on blocks of 4,096 elements, `erf`, `log` and `exp` among them, into buffers used again: 84.5 to 87.7 ms against 86.2 ms for the program as written, with the same bits. The arithmetic between the library calls is 13 ms of its 87 ms, and the kernels have it already.
- **Arrays that step gathered for every program that reads one,** over the rows they walk at one step: the kinetic energy took 99.8 to 100.8 ms where it took 93.9 to 95.7 ms. The programs it lost on are one kernel that reads two such arrays, where two copies replace one pass. That is where the rule of operations against copies comes from, and with it gathering is in the code ([Computing a group](#computing-a-group)).
- **A program for each short row, in place of declining,** for rows of 25 at more than one pitch: isoneutral mixing 194 to 215 ms against 196 to 201 ms, the kinetic energy 94 to 99 ms against 97 ms. On an expression by itself it is 1.16 times NumPy's passes, and it is what the module does from 16 elements a row; on the two routines it cannot be told from nothing.
- **A call of the kernel for each row, with the row stored at the result's step afterwards,** for the stores of isoneutral mixing that the module's loop now makes: 1,632 µs a statement for NumPy's 1,553 µs, and the routine 198.3 and 200.5 ms against 198.1 and 198.7 ms. The call for a row of 25 costs what NumPy's second pass did. The same statement as a loop with no call in it is in the code.
- **Rows of 16 to 31 elements copied into blocks,** for any program with an array that steps: isoneutral mixing 195.6 and 198.1 ms against 197.6 and 199.7 ms, and the kinetic energy 90.4 and 90.2 ms against 89.4 and 89.3 ms. One routine's gain is the other's loss, and both are within what a run moves by.
- **Rows of 25 at more than one pitch copied into blocks,** for a program of two steps or more, in place of a call of every step for each row: 2.35 ns an element where the calls take 1.85 ns, on the four of isoneutral mixing's programs it applied to. Those programs read five arrays and write one, 48 bytes an element; at 1.85 ns they move memory, and a copy of each array that lies otherwise is one more pass.
- **Large views left pending,** so that a write through one could wait and be fused with what it stores: the particle filter the same, 325 to 326 ms either way, and the cavity flow 54.1 to 54.7 ms against 52.0 to 52.1 ms, for the nodes its thousands of views then are.

### Small arrays: the graph costs more than the program

Four programs of the corpus are behind the eager program with the same kernels, and they are the four on arrays below the limit at which the layer writes anything down. Their fused run has no graph: it is the eager program with a lazy array around each value.

| Program | Elements to an array | Eager, kernels installed | Writing its graph, nothing computed | Fused run | Fusion forced on every array | Forced from 512 elements |
|---|---|---|---|---|---|---|
| `make_decision` | 32 | 85.3 ms | 1039.2 ms | 116.8 ms | 862 ms | 120 ms |
| The recurrent network | 100 to 10,000 | 28.6 ms | 45.2 ms | 32.7 ms | 163 ms | 103 ms |
| The perceptron | 8,000 | 25.3 ms | 24.2 ms | 27.7 ms | 125 ms | 99 ms |
| The fractal flame | 25,000 | 88.2 ms | 86.3 ms | 91.0 ms | 194 ms | 194 ms |

- **What they are behind by is the lazy array around each value.** An operator made at once costs 26 to 47 ns over NumPy's own, a view 55 ns, a function called through the stand-in 90 to 135 ns ([As the graph is built](#as-the-graph-is-built)). Statement by statement in `make_decision`'s loop, on 32 elements: `e - symbols` 499 ns for 459 ns, `abs(d)` 376 for 337, `a ** 2` 224 for 179, `np.argmin(s)` 208 for 95, `symbols[i]` 130 for 84, `out[i] = v` 121 for 95; the body 1,555 ns for 1,166 ns.
- **A pattern needs the graph, and the graph alone costs more than the program.** The third column is the time it takes to write these programs' operations down as nodes, with nothing computed. For three of the four that is more than the eager program takes for everything, and for the fourth it is nearly all of it. Whatever computed those graphs, at any speed, would start behind.
- **And computing them is slower still, which was measured.** With every operation written down and planned whatever its size, the programs take two to seven times what they take with calls made at once; from 512 elements, two to three and a half times for the three whose arrays are that large. A kernel saves a pass over memory, and an array of a few thousand elements is in the cache either way; what is left to save is NumPy's call, about 0.1 µs, against a node, a plan and a program of kernel calls.
- **What would take them is not a pattern.** It is a lazy array that costs less to make, which is the layer's own work and has a floor at an object made and freed for each operation; or compiling the body of the loop once from a trace of it, which is another design.

### What a pattern could still take

A pattern, in this document, is a tree of operations found in a graph and computed by code written beforehand: a kernel, a program of kernels, a loop of the module's own, a walk of the driver. It needs three things. The operations have to be written down as a graph. Among them there has to be a tree that is element by element on float64. And what replaces NumPy's calls for it has to cost less than they do. Every millisecond of a fused run is in one of four places, which the limit tool measures apart ([How far any fusion could go](#how-far-any-fusion-could-go)), and in each of them one of the three fails or the gain has a measured bound.

| Program | Fused run | In its kernels' programs | NumPy's calls a kernel could make: arithmetic, masks, a library's functions, other types | Of that, gone if each cost its loop in the cache | The rest of the run |
|---|---|---|---|---|---|
| `veros_isoneutral` | 189.7 ms | 101.6 ms | 68.9 ms | 11.00 ms, 5.8% | 19.6 ms |
| `veros_tke` | 87.9 ms | 36.4 ms | 16.0 ms | 5.67 ms, 6.4% | 36.7 ms |
| `attention` | 28.9 ms | 0.8 ms | 20.5 ms | 2.57 ms, 8.9% | 6.6 ms |
| `veros_eos` | 74.0 ms | 55.6 ms | 17.8 ms | 2.45 ms, 3.3% | 1.1 ms |
| `black_scholes` | 85.5 ms | 7.6 ms | 77.4 ms | 1.82 ms, 2.1% | 1.0 ms |
| `particle_filter` | 250.3 ms | 44.8 ms | 60.9 ms | 0.87 ms, 0.3% | 146.8 ms |

**1. Inside the kernels' programs.** The pattern has been applied. What it costs there is the arithmetic and the memory it moves, and three things could lower it. Each was measured.

- *A larger kernel:* one loop for a whole expression in place of a program of kernels is slower on the equation of state, level on the regression, and within what a run moves by on Black-Scholes and isoneutral mixing ([The best a kernel for it could do](#the-best-a-kernel-for-it-could-do)).
- *Another walk for short rows:* three were built for the rows of 25 in the Veros routines and dropped ([What was built on it and did not gain](#what-was-built-on-it-and-did-not-gain)). The programs that are a call for each such row take 1.85 ns an element, and copying their rows into blocks made that 2.35 ns: what a program does that is bound by the memory it moves, when it is given one more pass. The one walk that gained, a loop with no call in it for a product stored at a step, is in the code.
- *A kernel tuned by hand:* the fourteen written by hand are level with the compiled C beyond the first-level cache ([The kernels written by hand, alone](#the-kernels-written-by-hand-alone)).

**2. Calls NumPy still makes for operations a kernel could make.** The fourth column of the table is all of them, and the fifth is what would go if each cost what its loop costs in the cache, which is what a fusion at no cost of its own would take off. The table has the six programs it is more than a tenth of a millisecond for; for the other twenty it is nothing or next to it. In the two Veros routines it is the expressions of one to five operations on arrays that step, and the column overstates them: a loop written for the commonest, which reads the arrays where they lie, is 1.14 times NumPy's calls and not a fusion at no cost, about 1 ms of each routine. It would need each kernel written a second time with a step for every array it reads, and the judge to describe such arrays. In the equation of state and Black-Scholes it is single operations between expressions, which a kernel makes no faster than NumPy does, and in the transformer block float32 and `tanh`, on arrays of a few thousand elements.

**3. The rest of the run.** The program's own Python, a library's code, matrix products, selections and writes through masks and positions, sorting, random draws, and the layer. None of it is a tree of element-by-element operations on float64, so there is nothing to match. The kernels the package installs take what they can of it, with or without a graph.

**4. Programs that write no graph.** The four above. Writing the graph costs more than the program.

So on these programs a further pattern has nothing left that a measurement can see. That is a statement about the twenty-six programs of the corpus, on one machine, and it is weaker than it sounds in two ways, which the next two sections are about: other programs, and another processor.

### The programs of issue 298

[Issue 298](https://github.com/sofeikov/hotkernels/issues/298) lists 21 programs that come out as one graph each when every operation is deferred: stencils, convolutions and rank updates written as Python loops, filters, and chains of arithmetic. One is in the corpus already. The other twenty were run as the corpus is, each at the size the issue names, with their inputs copied before each run since most of them write into what they are given.

| Program | What it is | NumPy | Eager, kernels installed | Fused and installed | Against NumPy | Against installed | Expressions planned | Calls made at once |
|---|---|---|---|---|---|---|---|---|
| `jacobi_2d` | A stencil of five slices, 80 steps on 350 x 350 | 41.02 ms | 40.27 ms | 17.79 ms | 2.31x | 2.26x | 158 | 0 |
| `heat_3d` | A stencil in three dimensions, 50 steps on 40 x 40 x 40 | 30.22 ms | 30.07 ms | 16.39 ms | 1.84x | 1.83x | 98 | 0 |
| `slowparts` | Products of `tanh` terms slice by slice, arrays of 10 to 200 | 1.08 ms | 0.63 ms | 0.86 ms | 1.26x | 0.73x | 0 | 2,200 |
| `adi` | An implicit solver, a column at a time | 64.40 ms | 50.92 ms | 60.00 ms | 1.07x | 0.85x | 0 | 142,720 |
| `syr2k` | A rank update, a row at a time | 42.41 ms | 31.26 ms | 39.90 ms | 1.06x | 0.78x | 0 | 107,940 |
| `jacobi_1d` | A stencil of three slices, 3,000 steps on 12,000 | 102.98 ms | 102.79 ms | 101.36 ms | 1.02x | 1.01x | 0 | 23,992 |
| `ex088` | The Game of Life on 50 x 50 whole numbers | 2.57 ms | 2.41 ms | 2.55 ms | 1.01x | 0.95x | 0 | 1,800 |
| `covariance` | A loop over columns, a matrix product in each | 393.50 ms | 389.14 ms | 390.86 ms | 1.01x | 1.00x | 0 | 5,602 |
| `diffusion` | A stencil written into a second array, 128 x 128 | 7.72 ms | 7.69 ms | 7.76 ms | 0.99x | 0.99x | 0 | 1,200 |
| `correlation` | A loop over columns, a matrix product in each | 391.33 ms | 396.26 ms | 397.05 ms | 0.99x | 1.00x | 0 | 4,204 |
| `wdist` | A double loop over columns of 70 elements | 7.99 ms | 5.83 ms | 8.32 ms | 0.96x | 0.70x | 0 | 29,400 |
| `conv2d_bias` | A convolution as a Python loop over output positions | 54.18 ms | 54.16 ms | 56.49 ms | 0.96x | 0.96x | 0 | 10,801 |
| `multiple_sum` | A sum for each row of 100 elements | 0.90 ms | 0.90 ms | 0.95 ms | 0.95x | 0.95x | 0 | 300 |
| `lenet` | Convolutions as Python loops over output positions | 30.73 ms | 31.63 ms | 32.52 ms | 0.94x | 0.97x | 0 | 11,171 |
| `resnet` | Convolutions as Python loops over output positions | 34.64 ms | 34.96 ms | 36.87 ms | 0.94x | 0.95x | 1 | 7,076 |
| `go_fast` | `trace += np.tanh(a[i, i])` in a loop, then `a + trace` | 14.85 ms | 15.11 ms | 16.44 ms | 0.90x | 0.92x | 0 | 12,000 |
| `deriche` | Recursive filters along rows and columns | 36.51 ms | 31.97 ms | 40.51 ms | 0.90x | 0.79x | 2 | 39,958 |
| `local_maxima` | One element and one neighbour at a time | 10.90 ms | 11.07 ms | 14.97 ms | 0.73x | 0.74x | 0 | 43,200 |
| `syrk` | A rank update, a row at a time | 32.80 ms | 28.45 ms | 46.86 ms | 0.70x | 0.61x | 0 | 120,200 |
| `seidel_2d` | A stencil a row at a time, each row from the one before | 43.92 ms | 42.31 ms | 97.39 ms | 0.45x | 0.43x | 0 | 548,800 |

- **Two gain, and they are the two that are expressions on large arrays.** The Jacobi stencil in two dimensions and the heat equation in three are covered by kernels the corpus had asked for already, `fuse_add_v_add_v_add_v_v_f64` and `fuse_scaled_sum_f64` among them.
- **The others are on arrays below the limit,** rows and columns of ten to a few thousand elements or single elements, and write no graph. They are the fourth place above.
- **They found five things in the layer that the corpus had not,** all in what a program on small arrays costs, and none a missing pattern:
  - *A write into an array the program was handed went to Python:* 3.9 µs a write for NumPy's 0.2 µs. The symmetric rank update took 204 ms for 34 ms and the Gauss-Seidel stencil 1,111 ms for 44 ms. The core makes such a write now ([As the graph is built](#as-the-graph-is-built)).
  - *A recurrence over the rows of an array copied the array for every row:* the Deriche filter took 11.6 s for 37 ms ([Writes](#writes)).
  - *A small call waited for a large array still pending,* and every operation after it with it ([As the graph is built](#as-the-graph-is-built)).
  - *`x += y` on a number taken out of an array went through the ufunc,* 599 ns for the scalar's 34 ns, and so did an operator between a NumPy scalar of the program's own and a number or a row ([lazy-design.md](lazy-design.md#the-c-core)).
  - *The set of pending nodes kept the table of its largest moment,* so that after one graph of 44,000 nodes each later write walked 131,072 empty places: the Game of Life took 8.8 ms after another program's graph and 3.1 ms by itself ([lazy-design.md](lazy-design.md#pending-work-and-writes)).
- **What is left behind NumPy there is a loop over single elements.** The Gauss-Seidel stencil updates one element at a time and runs at 0.45x of NumPy; the search for local maxima, on single elements too, at 0.73x; the symmetric rank update, on rows of 1 to 200 elements, at 0.70x. A lazy array around a number costs 60 to 100 ns an operation where NumPy's scalar costs 35 to 60 ns. The other seventeen are within 10% of NumPy or ahead of it, and five of those are behind the eager program with the kernels installed by more, since what the kernels gain on rows of a hundred elements is about what the lazy arrays cost. [Issue 310](https://github.com/sofeikov/hotkernels/issues/310) has the costs and what could lower them.

### Another processor

Everything above was measured on an Apple M1 Pro. This is the same on x86-64, an Intel Core i5-14600KF under Linux with NumPy 2.5.5 and CPython 3.13, as [issue 309](https://github.com/sofeikov/hotkernels/issues/309) asked.

**The kernels.** `make_kernel.py --arch x86_64` writes the same walk of the tree in AVX2, eight elements a step in two `ymm` registers of four, and wrote it for the seventy-nine kernels that compute an expression element by element. All seventy-nine pass the judge, and each was timed against its compiled C with `uv run judge run`. Seventy-six are level with it, 0.96x to 1.02x over all sizes: clang vectorises the C references at the judge's `-O3 -march=native` as it does on ARM64, and beyond the first-level cache both wait for memory. One is slower, `max(S0,max(V0,min(S1,V1)))` at 0.75x, because clang tests once whether a number is a NaN where the written loop tests it every step. Two are far ahead, `np.sqrt(a)` at 5.46x and `s / np.sqrt(a)` at 3.50x, because the C reference's `sqrt` keeps C's rule that a square root of a negative number sets `errno`, and clang then calls the library for every element. So on x86-64 those two are assembly and the other seventy-eight are their C references, which is what the repository does wherever a kernel's assembly does not beat its compiler. Each kernel's `x86_64/attempts.md` has its figures. The sum of squares is a reduction, which the tool does not write; its C is 14.9x NumPy's two passes over all sizes and 4x from 65,536 elements up, and no assembly was written for it.

**What NumPy does there that it does not do on ARM64.** Three things, each found by the judge:

- **`np.maximum` and `np.minimum` give the second operand of two zeros,** whatever its sign, where ARM64's `fmax` gives the positive zero. The C references had ARM64's rule and failed the judge on x86-64; they now have both, one for each architecture. And since `np.maximum(x, 0.0)` and `np.maximum(0.0, x)` then differ where `x` is `-0.0`, `fuse.py` takes a maximum's and a minimum's operands only in the order the program wrote them there. The long expression of the tests loses one fused node to it: its `np.maximum(a, 0.25)` would need a kernel for `max(V0,S0)`, which the library does not have.
- **An invalid operation gives a NaN with its sign bit set,** where ARM64's is `np.nan`'s bits. A program that holds `np.nan` and also makes `inf - inf` has two kinds of NaN, and which of two an operation returns is which operand comes first. NumPy's own loops do not keep to one order: `a + b` returns `a`'s NaN in blocks of eight and `b`'s in the elements left over, and `a + s` the number's NaN from nine elements up and the array's below. A C compiler is free to swap the operands of a sum. So each fused spec's domain is now inputs whose NaNs all have the bits of this processor's invalid operations (`one_nan`), and the judge checks that a result is a NaN wherever NumPy's is, but not which of two. On ARM64 that changes nothing. What a program with both kinds gets is NumPy's NaN, or the other one; [the kernels' README](../prototypes/lazy/kernels/README.md#what-is-not-specified) already said so.
- **One C reference cannot be NumPy's even then.** In `s0 * (a * (|s1 * b| * (c * d + (s2 - e))))` the absolute value makes a NaN with the sign bit clear out of one with it set, a second kind inside the expression, and clang orders the products otherwise than NumPy's passes. The judge refuses the reference, so that kernel has no assembly on x86-64 either.

**The programs, four ways,** with the command of [The programs, four ways](#the-programs-four-ways), the quickest of 5 rounds. This table and the next were taken again for [issue 323](https://github.com/sofeikov/hotkernels/issues/323), with the ways timed in turn and no result of a program kept while it is timed ([what that is](lazy-codebases-findings.md#issue-323-the-ways-timed-in-turn-by-default)). They were taken pinned to one performance core with one BLAS thread (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 6`), while other work ran on the machine, at load averages of 2.1 to 3.5. So a program of matrix products has them on one thread here, where the first take of these tables had BLAS on every core.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             53.31     52.92      54.72     0.24    54.48      52.25    44.39     0.78     7.07    1.05x    1.02x     1.01x
attention              53.93     53.94      59.19     0.47    58.72      58.24     2.18     0.15    55.91    1.02x    0.93x     0.93x
perceptron             41.57     35.66      79.05    29.89    49.16      38.24    37.78     0.04     0.41    2.07x    1.09x     0.93x
char_rnn               22.36     19.37      78.53    44.84    33.69      23.16    23.16     0.00     0.00    3.39x    0.97x     0.84x
particle_filter       388.52    356.90     409.27   125.64   283.63     328.66   137.61     3.97   187.08    1.25x    1.18x     1.09x
veros_eos             234.84    235.21     268.87     2.88   265.99     139.26     1.56     0.42   137.29    1.93x    1.69x     1.69x
veros_isoneutral      402.86    403.77     427.89    42.27   385.62     314.11    55.11     1.03   257.97    1.36x    1.28x     1.29x
veros_tke             176.90    169.86     170.17    30.36   139.81     128.05    40.63     0.55    86.87    1.33x    1.38x     1.33x
cavity_flow           199.82    199.33     247.27     7.67   239.59      64.73    10.22     2.80    51.71    3.82x    3.09x     3.08x
hdiff                 178.19    167.70     209.71    10.74   198.97      77.03    10.70     0.05    66.28    2.72x    2.31x     2.18x
black_scholes          78.88     78.76      83.25     3.88    79.38      70.58     3.88     0.04    66.66    1.18x    1.12x     1.12x
trimesh_mesh          139.99    120.70     121.50   121.26     0.24     120.25   120.25     0.00     0.00    1.01x    1.16x     1.00x
sklearn_metrics      1025.79    316.25     321.95   317.32     4.62     321.34   321.34     0.00     0.00    1.00x    3.19x     0.98x
empyrical_stats       939.03    433.44     434.62   434.62     0.00     433.68   433.68     0.00     0.00    1.00x    2.17x     1.00x
skimage_pipeline     1155.21   1099.37    1092.21  1092.21     0.00    1091.82  1091.82     0.00     0.00    1.00x    1.06x     1.01x
pandas_sales          418.97    408.71     408.13   408.13     0.00     409.18   409.18     0.00     0.00    1.00x    1.02x     1.00x
astropy_timeseries   1465.95   1298.77    1318.68  1317.04     1.64    1290.45  1289.67     0.10     0.68    1.02x    1.14x     1.01x
example                36.63     14.06      14.58     6.60     7.98      14.15    12.36     0.01     1.78    1.03x    2.59x     0.99x
flame                 115.48     73.78     160.97    85.88    75.09      78.42    76.18     0.04     2.19    2.05x    1.47x     0.94x
azimint_naive          60.97     71.51      86.32     6.53    79.79      71.56    71.56     0.00     0.00    1.21x    0.85x     1.00x
azimint_hist           90.44     59.41      59.74    59.70     0.04      59.80    59.80     0.00     0.00    1.00x    1.51x     0.99x
mandelbrot1            95.02     88.72     100.40     1.31    99.09      96.06    96.06     0.00     0.00    1.05x    0.99x     0.92x
mandelbrot2           146.62    155.72     185.78    28.56   157.22     156.13   156.13     0.00     0.00    1.19x    0.94x     1.00x
make_decision         116.34     71.56    1326.81  1055.96   270.85      98.67    98.67     0.00     0.00   13.45x    1.18x     0.73x
pandas_filter         162.77    132.15     131.88   131.88     0.00     132.11   132.11     0.00     0.00    1.00x    1.23x     1.00x
equalize_hist         113.58    104.44     104.49   104.39     0.11     104.71   104.71     0.00     0.00    1.00x    1.08x     1.00x
```

- **Where the programs are expressions on large arrays the gains are about those of ARM64:** the cavity flow 3.09x, the horizontal diffusion 2.31x, the equation of state 1.69x, turbulent kinetic energy 1.38x, isoneutral mixing 1.28x, the particle filter 1.18x. This is with seventy-eight of the eighty kernels compiled C: fusion's gain is the passes it saves, not the instructions.
- **The regression is level, 1.02x, where the first take had it at 2.26x.** That was its NumPy column: 64.44 ms in that take and 137.76 ms in the table after it, with BLAS on every core, against a fused run of 28.54 ms. On one thread its NumPy run is 53.31 ms here and 55.55 ms in the next table, and 44 ms of its 52 ms fused run is calls made at once, its matrix products among them.
- **Turbulent kinetic energy was 1.62x in the first take and is 1.38x.** That one is the order. One way after another its NumPy run, timed first, took 200 to 205 ms in three takes; in turn it took 177 to 188 ms in four, with the fused run at 123 to 137 ms either way.
- **Four programs are behind NumPy by 3% or more:** the naive azimuthal integration at 0.85x, the attention block at 0.93x, the second Mandelbrot set at 0.94x and the character RNN at 0.97x. The attention block's eager run is the same with the kernels installed and without (53.93 ms and 53.94 ms), where the first take had the installed run slower; its lazy runs are 4 to 5 ms behind both. The azimuthal integration and the Mandelbrot set are behind with the kernels installed and no graph (60.97 ms for NumPy and 71.51 ms installed, 146.62 ms and 155.72 ms), and their fused runs are level with that. A second take had the azimuthal integration at 0.91x; unpinned, earlier the same day, its two eager runs were level at 56 to 58 ms. Why the installed run is slower there when pinned was not looked into.
- **Against the eager program with the same kernels** the small-array programs are behind as on ARM64, by the cost of a call made at once: the decision tree at 0.73x, the character RNN at 0.84x, the first Mandelbrot set at 0.92x, the perceptron and the attention block at 0.93x, the flame graph at 0.94x.
- **A second take, at load averages of 3.2 to 3.8, had times up to 28% longer and nearly the same ratios.** The equation of state took 298.50 ms on NumPy for 234.84 ms, isoneutral mixing 488.70 ms for 402.86 ms. Fused against NumPy, nineteen of the twenty-six were within 0.05 of this table. The other seven: the equation of state 1.79x for 1.69x, the horizontal diffusion 2.18x for 2.31x, `sklearn_metrics` 3.05x for 3.19x, `empyrical_stats` 2.09x for 2.17x, `example` 2.49x for 2.59x, `pandas_filter` 1.29x for 1.23x, and the azimuthal integration as above. Timed in turn, a slow stretch of the machine falls on every way.

**How far fusion could go,** with the command of [How far any fusion could go](#how-far-any-fusion-could-go), in turn and pinned as the table above:

```text
program                NumPy installed     floor     fused  at best       now     left   left of
                                                                                       fused run
regression             55.55     56.27     41.60     53.07    1.34x     1.05x    11.47     21.6%
attention              54.26     54.36     43.00     54.99    1.26x     0.99x    12.00     21.8%
perceptron             41.91     35.96     30.69     38.45    1.37x     1.09x     7.76     20.2%
char_rnn               22.39     19.44     13.91     22.95    1.61x     0.98x     9.04     39.4%
particle_filter       393.82    362.68    278.47    336.57    1.41x     1.17x    58.11     17.3%
veros_eos             245.10    245.00    107.91    145.59    2.27x     1.68x    37.68     25.9%
veros_isoneutral      409.07    409.01    264.70    315.52    1.55x     1.30x    50.82     16.1%
veros_tke             184.83    174.90     94.10    133.24    1.96x     1.39x    39.14     29.4%
cavity_flow           193.11    193.66    126.60     62.62    1.53x     3.08x   -63.98   -102.2%
hdiff                 185.03    172.85     34.51     83.30    5.36x     2.22x    48.79     58.6%
black_scholes          79.31     79.29     64.89     70.90    1.22x     1.12x     6.01      8.5%
trimesh_mesh          141.19    121.93    121.77    121.92    1.16x     1.16x     0.16      0.1%
sklearn_metrics      1036.17    323.04    322.58    328.39    3.21x     3.16x     5.81      1.8%
empyrical_stats      1021.51    479.20    479.20    480.27    2.13x     2.13x     1.06      0.2%
skimage_pipeline     1184.29   1130.21   1130.21   1111.75    1.05x     1.07x   -18.46     -1.7%
pandas_sales          470.46    463.72    463.72    482.24    1.01x     0.98x    18.53      3.8%
astropy_timeseries   1572.22   1393.60   1393.06   1417.32    1.13x     1.11x    24.26      1.7%
example                38.27     14.96     11.19     15.39    3.42x     2.49x     4.20     27.3%
flame                 119.22     76.25     63.19     85.28    1.89x     1.40x    22.09     25.9%
azimint_naive          64.94     73.17     45.67     73.84    1.42x     0.88x    28.18     38.2%
azimint_hist           96.20     64.20     64.19     62.90    1.50x     1.53x    -1.29     -2.1%
mandelbrot1           101.08     95.20     66.67    103.26    1.52x     0.98x    36.59     35.4%
mandelbrot2           151.96    165.07    158.20    167.76    0.96x     0.91x     9.56      5.7%
make_decision         118.41     72.61         -    100.10        -     1.18x        -         -
pandas_filter         157.30    120.77    120.77    118.46    1.30x     1.33x    -2.32     -2.0%
equalize_hist         114.47    103.66    103.65    103.77    1.10x     1.10x     0.13      0.1%
```

- **Where the most is left:** the particle filter (58.1 ms, 17% of its fused run), isoneutral mixing (50.8 ms, 16%), the horizontal diffusion (48.8 ms, 59%), turbulent kinetic energy (39.1 ms, 29%), the equation of state (37.7 ms, 26%) and the first Mandelbrot set (36.6 ms, 35%). The cavity flow's fused run is below its floor: the floor is the eager program less what could be removed from it, and fusion also removes the copies its writes made.
- **The floor moves more than the runs do.** It is a difference of two sums of single calls' times, taken in one run. The equation of state's was 67.26 ms in the first take and is 107.91 ms here, the horizontal diffusion's 43.46 ms and 34.51 ms; their fused runs moved by 1%.
- **The two tables agree on a program's NumPy time now.** The regression's is 53.31 ms above and 55.55 ms here, where the first take had 64.44 ms and 137.76 ms. Twenty of the twenty-six are within 5% between the two tables and the other six within 13%, `pandas_sales` the furthest; other work ran on the machine through both.
- **The limits were not moved.** The block of 4,096 elements, the shortest rows, `OPERATIONS_A_COPY` and the 32,768 elements below which a call is made at once are the ones the M1 Pro set; with them the programs above are where they are. Whether other values do better on x86-64 was not measured.

### Ten programs from other codebases

Ten more programs were run on x86-64 as the corpus is, each the NumPy code of a published project with the stand-in where its `import numpy as np` stood: perlin-numpy, the convolutional network of *Deep Learning from Scratch*, Matplotlib's hill shading, pvlib's solar position, scikit-image's colour difference, scikit-learn's perceptron, SALib's Sobol' analysis, prysm's Zernike polynomials, Astropy's models and python_speech_features. [lazy-codebases-findings.md](lazy-codebases-findings.md) has them.

- **The fused run is 0.88x to 1.14x of NumPy on them,** ahead on three by 1.08x to 1.14x, level on five and behind on two. All give NumPy's bits. One of the two behind, pvlib's solar position at 0.97x, is 1.02x to 1.05x with a whole number of NumPy's taken as a number ([what was built on it](lazy-codebases-findings.md#issue-318-a-whole-number-of-numpys-is-a-number-of-an-expression)). The perceptron, the 0.88x, is 0.95x since.
- **No pattern was missing:** every expression found on arrays a kernel can read was planned.
- **A third of their time is a mathematics library's functions,** `cos`, `sin`, `x ** 3`, `%`, which end an expression; a sixth is reductions along an axis. What a kind that is not bit-identical would be worth is measured in [lazy-inexact-design.md](lazy-inexact-design.md): glibc's vector functions in place of NumPy's own loops make pvlib's solar position 2.17x eagerly and 2.27x fused, so the gain is the library's and not fusion's, and the design recommends no such kind in the layer. `np.radians` and `np.degrees`, one multiplication each, are operations of an expression now.
- **They found things in the layer that the corpus had not:** an array left transposed by the call that made it was declined, where NumPy walks it as it lies, and is walked in the order of memory since ([Computing a group](#computing-a-group)); rows of four with a value for each row are declined in a program of several steps; a NumPy scalar of a whole-number type ended an expression, and is a number now ([What is fused](#an-expression)); a ufunc called with `out=`, `np.clip` and a function called on plain arrays cost 1 to 8 µs over NumPy's call, and are made by the core since, at 0.1 to 0.9 µs, which took scikit-learn's perceptron from 0.89x to 0.95x ([As the graph is built](#as-the-graph-is-built)); a lazy run computed node by node was up to 1.4x slower than the eager program, from a shape rule that walked a table of positions a second time and from calls that each made a new array where NumPy reuses a temporary's memory, and is 0.82x to 1.11x of it since a call writes into an operand nothing else reads ([lazy-design.md](lazy-design.md#computing)); and `type(x)` of a lazy array stopped Matplotlib, whose `np.empty(...).view(type(x))` a lazy array takes since ([lazy-design.md](lazy-design.md#what-a-library-asks-a-lazy-array)): the hill shading runs on a lazy elevation model and is level with where it was.
- **The order the ways are timed in was worth several percent,** and both timing tools now time the ways in turn and keep no result of a program while they time ([what was built on it](lazy-codebases-findings.md#issue-323-the-ways-timed-in-turn-by-default)). The x86-64 tables of [Another processor](#another-processor) were taken again that way; the tables from the M1 Pro were taken one way after another.

### What is open in the layer, and is not fusion

- **What handing a lazy array to a library costs.** Scoring a classifier with scikit-learn is 2% slower with lazy arrays handed in than with `ndarray`s when the kernels are installed, and 7% with nothing installed, with seven operations deferred in all. The time is not in the layer: the library takes a longer road for an array type it does not know. And scikit-image fails on one. A program whose work is a library's gains nothing from the layer; whether it should lose anything is open.
- **More of NumPy in the deferred set.** The nine workloads make calls the layer does not defer: `np.histogram`, `np.bincount`, `np.interp`, `np.isin`, `np.unique`, `np.stack`, `x.astype(...)`, `np.clip` without `out=`. None of them sits inside an expression the kernels could take, but for `np.clip`, which is a maximum and a minimum, and a cast, which is a kernel nobody wrote.
- **The limit below which a call is made at once.** 32,768 elements is in the middle of a range, 16,384 to 65,536, over which no program of the corpus runs 1% differently. Two programs have arrays near it, which is two programs' evidence. Timed by itself, `(a + b) * a` on 40,000 elements takes 22.9 µs fused for NumPy's 17.7 µs, and is ahead from about 70,000: a computation costs about 6 µs whatever is in it. [Issue 311](https://github.com/sofeikov/hotkernels/issues/311).
- **What a call made at once costs,** which is all the programs on small arrays are behind by: [issue 310](https://github.com/sofeikov/hotkernels/issues/310).
- **The graph page does not show fusion:** which operations a program of kernels made together, and which were given up. [Issue 312](https://github.com/sofeikov/hotkernels/issues/312).
- **A sum at the end of a program, and reductions along an axis.** `sum(sq(V0))` is a kernel of its own; a sum of a longer expression computes the expression into an array first, since NumPy's order of addition is over the whole array and not over blocks. `np.sum(e ** 2, axis=-1)` and `np.max(np.abs(e))` are left to NumPy. The tool finds 6 ms removable in the regression's reductions, of which its kernel for a sum of squares has a part, and under 2 ms in any other program on large arrays.

## What changed outside `fuse.py`

- **`evaluate.py` asks a planner,** and has a meter and a clock for the tools. With none set it computes node by node. It decides, for each write it computes, whether the array's own memory may be written (`reusable`), and it catches warnings and puts the error settings once for a computation. [lazy-design.md](lazy-design.md#computing) says what a planner may and may not change, and [its section on writes](lazy-design.md#writes) when a write copies.
- **`array.py` and the C core** have `at_once_below`, the roads for calls made at once, `x ** e` as `operator.pow`, and `x.shape = s`. The core makes and frees the layer's objects, gives a call made at once a lazy array that holds its value alone, stands in the class's table for the operators, takes the nodes of a computation one after the other, and reads the expressions for `fuse.find` ([lazy-design.md](lazy-design.md#the-c-core)). A view taken at once holds NumPy's view alone, and a write does not wait while one may be looking ([lazy-design.md](lazy-design.md#pending-work-and-writes)). Without the switch set, a graph is written node for node as before, which the layer's tests of what is written check.
- **`namespace.py`** makes the arrays of `np.mgrid`, `np.ogrid`, `np.r_` and `np.c_` lazy, so that what a program does with its grids is the layer's to make, and a kernel of the package's to serve ([lazy-design.md](lazy-design.md#the-stand-in-for-the-module)).
- **The judge looks for a kernel in a second place,** `prototypes/lazy/kernels/`, and locks its spec and reference to `main` there as it does in `kernels/`. Nothing else in the judge changed.
- **`programs/corpus.py`** has every program the repository times, `fusion_time.py` times each four ways, and `limit.py` measures what no fusion could remove from each ([What is left](#what-is-left)). `programs/codebases.py` has the ten programs of [Ten programs from other codebases](#ten-programs-from-other-codebases), which the same tools run by name.
- **Nothing was added to the deferred set.** `x.shape = s` is a view, which was already deferred; the calls the new programs make at once are listed under [What is open in the layer](#what-is-open-in-the-layer-and-is-not-fusion).
