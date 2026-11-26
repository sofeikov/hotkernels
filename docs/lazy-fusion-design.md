# Fusion by pattern in the lazy layer

This is fusion in the prototype of the lazy layer ([lazy-design.md](lazy-design.md), [`prototypes/lazy/`](../prototypes/lazy)), and what it measured. Nothing here is in the package. The layer writes a program's NumPy operations down as a graph; this document is about computing that graph faster than one NumPy call per operation, with NumPy's bits.

The approach is the plainest one there is. A **pattern** is a small tree of operations with named inputs, such as `a + s * b`. Where the graph holds that tree, a kernel written for it computes it in one pass over memory. A longer expression is covered with several patterns, and their kernels are called block by block, so that what one gives the next never leaves the cache. An operation no kernel is written for is computed as before, so nothing is lost by a pattern that is missing.

## Summary

- **A fused kernel gives NumPy's bits, and the judge checks it.** Every operation in one is a single IEEE operation that NumPy makes with the same instruction, in the program's order, each rounded by itself. No multiply-add is fused and no sum is regrouped. There are eighty kernels, in NEON assembly for ARM64, and each passes the judge against its NumPy expression on 20,000 to 370,000 cases. See [Why the bits are NumPy's](#why-the-bits-are-numpys).
- **The pattern is the kernel's own oracle.** A kernel's `spec.py` names the NumPy expression it must equal. Run on lazy arrays, that expression writes a graph, and the shape of that graph is the pattern. Nothing else says what a kernel serves. See [The library](#the-library).
- **A whole expression is one pass over memory, whatever its length.** Its kernel calls are run a block of 4,096 elements at a time, so what one kernel gives the next stays in the cache. For that every operation has a kernel, the single operations too. See [An expression block by block](#an-expression-block-by-block).
- **Most kernels are written by a program from the name of their shape.** Fourteen were written by hand; `make_kernel.py` wrote sixty-six, each with its spec and its C reference, and the judge holds them to the same bits. See [The kernels](#the-kernels).
- **Which shapes to write was mined, not guessed.** Twenty-six programs were run lazily with every call timed: every program this repository times with and without `hotkernels.install()`, and six more. See [Finding the shapes](#finding-the-shapes).
- **A write is made where the array is.** A deferred write is a copy only when something still reads what the array held, and with fusion an expression assigned to part of an array is computed straight into the part. See [Writes](#writes).
- **What cannot be fused is not written down.** With fusion on, a call that gives fewer than 32,768 elements, or that no kernel could make, on arrays that have their values, is NumPy's call at that line, not a node, and its result is one object that holds NumPy's value. See [As the graph is built](#as-the-graph-is-built).
- **The C core reads the graph as well as writing it.** The expressions of a computation are read off its nodes in C, by the functions of the Python reader one for one. See [Planning](#planning).
- **Rows of a few elements are a kernel's as well.** NumPy's loop is called once for each row of an array it cannot walk as one run, which for rows of two or four elements costs several times their arithmetic. A kernel walks such rows as one run or straight through what lies between them, and one operation by itself is then worth a kernel. See [Computing a group](#computing-a-group).
- **The layer runs as the package is used: with `hotkernels.install()` in effect.** `fuse.enable()` installs the package's single-operation kernels unless told not to, and the figure the layer is judged by is the fused run with them against the program on plain NumPy; beside it stands the eager program with the same kernels, which is what the package gives with no graph. See [The programs, four ways](#the-programs-four-ways).
- **Twenty-five of the twenty-six programs then run at NumPy's speed or faster, up to 5.1x.** Where a program is expressions on large arrays the gain is fusion's: a cavity flow 2.76x, a weather model's horizontal diffusion 2.35x, an ocean model's equation of state 1.75x, a particle filter 1.42x, a regression 1.37x, turbulent kinetic energy 1.13x, isoneutral mixing 1.11x. Where it is a library's work or single calls on large arrays the gain is the installed kernels', up to 5.1x, which the lazy run keeps to within 4%.
- **Against the eager program with the same kernels the fused run is ahead on seven programs,** by 1.10x to 2.75x, within 3% on fifteen, and behind on four: three on small arrays, at 0.72x, 0.87x and 0.93x, where a call made at once costs 15 to 115 ns over NumPy's own, and one at 0.96x. See [What the measurements show](#what-the-measurements-show).
- **A kernel the lazy run does not reach is a loss, and one was found.** A plain `ndarray` indexed by a lazy mask is not a call the layer makes, so the package's kernel for `x[mask]` was never asked: the second Mandelbrot program ran 16% behind the eager program with the kernels installed, and runs 1% behind with the arrays `np.mgrid` makes lazy.
- **What is left is measured, program by program, and what could take it was tried.** A tool takes from each program what no fusion could remove. Four programs run at or below that, fifteen have 10% at most to gain from any fusion, four are on arrays below the limit at which anything is written down, and three have something left. A loop written for each thing that is left says what a kernel for it could do at best: nothing for complex arithmetic, nothing in general for one kernel a whole expression, and 2.6 times NumPy for one flux on arrays that step, which two kernels and one gathering of each array now make. What the loops leave is under what a run of its program moves by. See [What is left](#what-is-left).

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
| `+`, `-`, `*`, `/`, `np.maximum`, `np.minimum`, unary `-`, `abs`, `np.sqrt`, `np.square`, `x ** 2` | Each is one IEEE operation, and NumPy's loop for it executes the one instruction a kernel would. `x ** 2`, `np.square(x)` and `x * x` are all one multiplication |
| A comparison, as the condition of `np.where(c, a, b)` | The comparison gives a mask, which a kernel keeps in a register and never writes to memory |
| A sum or a mean of every element, at the top | NumPy's order of addition can be reproduced; see [Reductions](#reductions) |

| What ends an expression | Why |
|---|---|
| `np.exp`, `np.log`, `np.tanh`, `np.arctan2`, `x ** 3`, any power but 2, `%` | NumPy calls the platform's mathematics library for these, one element at a time. A kernel is a leaf function and calls nothing, and a second implementation would not have the library's bits |
| An array of any element type but float64, a NumPy scalar of another type | The kernels are for float64. The transformer block of the corpus is float32 but for its attention scores, which a division by a float64 scalar makes float64, and only those are fused |
| An intermediate that is read twice, or that the program holds a name for | Its value is wanted again, so it has to exist; computing past it would compute it twice. It is an operand of the expressions that read it |
| An intermediate of another shape than the result | It belongs to another loop |
| The matrix product, a reduction along an axis, indexing by arrays | Not element by element. Each is computed by NumPy and is an operand of what reads it |

**An operand is a vector or a number.** Along the innermost axis, each thing an expression reads is either an array as long as the result (`V` in a pattern) or the same number for the whole run (`S`): a Python number, a float64 scalar, an array without dimensions, or an array of length one along that axis, such as the mean of each row kept with `keepdims=True`. Along the outer axes an operand may broadcast in any way.

**The same array may be read twice.** `a + s * a` is the pattern `add(V0,mul(S0,V1))`, served by the kernel of `a + s * b` given one pointer twice. Each place an array is read is a leaf of its own, so a pattern's name does not depend on which of its arrays happen to be the same one.

**The operands of a sum or a product may be in either order.** `c + a * b` and `b * a + c` are one pattern.

### Reductions

`(x ** 2).sum()` is the pattern `sum(sq(V0))`: the square and the sum in one pass, with no array of squares.

NumPy does not add a contiguous array from left to right. It adds pairwise: eight running sums over blocks of up to 128 elements, combined in a fixed tree whose shape depends on the length ([`kernels/sum_f64/spec.py`](../kernels/sum_f64/spec.py) has the rules). A fused reduction is exact only because it makes the same additions in the same order on the values it computes. `fuse_sum_squares_f64` walks NumPy's tree with two small stacks below the stack pointer and squares each element as it meets it; a test compares it with `np.sum(x ** 2)` at every length to 700 and at several to 300,000.

- **The mean is the same kernel.** NumPy's mean is its sum divided by the count, so `mean(sq(V0))` is an entry of the library that calls the sum kernel and divides.
- **Only a C-contiguous array is reduced by a kernel.** NumPy adds such an array, of any number of dimensions, in the order it lies in memory. For any other layout its order is another, and the call is left to NumPy.
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

- **It writes the four files of a kernel:** `spec.py`, whose oracle is the shape as a NumPy expression, with every number tried at zero of either sign, at an overflow, a denormal, an infinity and a NaN; `naive.c`; `arm64/kernel.s`; and the first entry of the log.
- **The kernel is then a kernel like the others.** Nothing the tool writes is trusted: the judge checks it against NumPy, and the pattern it serves is read off its oracle. A test holds the files in the repository to what the tool writes today.
- **Registers are given to the tree by the usual rule,** the operand that needs more of them first, four registers to a value. A number is held in every lane of one register. Shapes of up to six arrays and eight numbers fit.
- **Which shapes:** the eighteen single operations (the four of arithmetic on two arrays and on an array and a number either way round, a maximum and a minimum, a negation, an absolute value, a square root, a square), the forty-six shapes of two to six operations that cover the corpus's expressions best, as the miner chose them one at a time, and the two halves of an advection flux, four operations and eight ([What is left](#what-is-left)).
- **`np.maximum` and `np.minimum` are the processor's `fmax` and `fmin` on ARM64,** at every length and position: a NaN wins over a number, and of two zeros the maximum is the positive one. C's `fmax` is another function, so the reference spells the rule out.
- **One shape was refused by the judge, and rightly.** For `(s0 + (-|a|)) / s1` the compiler turns the C reference's `s0 + (-|a|)` into `s0 - |a|`, which gives a NaN the other sign. The assembly was NumPy's; the reference was not, and a kernel whose reference is wrong is not kept.
- **Not tuned.** A written kernel loads with plain `ldp` and is as the tool left it. Against the compiled C it is expected level, as the kernels written by hand are: the loops are ones clang vectorises as written.
- **What it does not write:** a comparison and `np.where`, whose mask is kept in a register; a reduction; float32.

### The library

**A kernel's pattern is read off its oracle.** A spec's oracle is the NumPy expression the judge holds the kernel to:

```python
def oracle(a, b, s):
    return a + s * b
```

`fuse.py` calls it with lazy arrays and two numbers. The oracle writes a graph without computing anything, the expression is read off that graph as off any other, and its name, `add(V0,mul(S0,V1))`, is the pattern. Which leaf is which argument is known because the arrays passed in are the leaves read.

- **The specification is its own pattern.** What a kernel is matched against and what the judge checks it against are one function, so they cannot disagree. A kernel whose oracle is not one expression of the operations above, or does not read each argument once, has no pattern and is refused when the library is loaded.
- **A name is the tree as written,** with the operands of a sum, a product, a maximum or a minimum in a fixed order: `sub(V0,mul(S0,V1))`. Two expressions of one shape have one name whichever way round the program wrote them.
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
- **One library does not take a lazy array.** scikit-image's `regionprops_table` asks `np.isscalar` of the mean of an image it was handed. For a lazy image that mean is a lazy array without dimensions, not a number, and the library fails on it. `skimage_pipeline` is given plain arrays, and nothing of it is deferred.
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
- **It reads float64 arrays** with strides that are positive and do not grow from one axis to the next. NumPy's result for such operands is laid out in C order, which the output is.
- **It declines anything else:** a reversed or Fortran-ordered array, a transposed one, an output that overlaps an operand without being it, rows shorter than 16 elements at more than one pitch or more than eight times their length apart, an array that steps with fewer than 32 elements at one step, and a program with too few operations for the arrays it would copy. The group's nodes are then computed by their own calls, in order. Declining is decided before anything is computed or copied.

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
| A call of any size that no kernel could make, on arrays that all have their values: a comparison, a selection by a mask, a matrix product, `np.exp`, anything complex or float32 | The same. Nothing written on top of it could be computed together with it, so its node would only cost |
| A view, a transpose or an element of an array that has its values | NumPy's own, whatever the size: a view costs nothing to take |
| A write, when no node at all is pending | NumPy's write, into the array's own memory |
| Anything else: arithmetic on float64 that gives a large result, a call with an operand still pending, a write while something is pending | A node, as before |

- **It is what the eager program does at that line.** The value is NumPy's, and so are an error and a warning, raised where the program wrote the call. Nothing later can tell the difference: the result is a lazy array like any other, and a large operation on it is deferred.
- **A write at once is sound when nothing is pending.** The copy a deferred write is defined by exists for operations written before the write and not computed yet. With no node pending there is none, and writing into the memory is NumPy's own meaning, views included.
- **A view taken at once is NumPy's view and nothing more,** so a write into an array that such a view may be looking at does not wait: everything pending is computed and the write is made in the memory, where the view sees it. Whether one may be looking is read off the reference NumPy's view holds to its array ([lazy-design.md](lazy-design.md#pending-work-and-writes)). Of the cavity flow's 1,220 writes into its grids, 1,196 still wait: a slice of a grid with a write pending is deferred with the write, and holds nothing of NumPy's.
- **An in-place operator made at once on rows of a few elements is its own expression, and an assignment.** `particles[:, :2] += particles[:, 2:] * dt` writes through a view taken at once, so it does not wait, and NumPy's in-place addition is a call of its loop for each row of two: 861 µs for 100,000 rows. The layer makes `view + other` as an expression, which the module's loop over the rows computes, and assigns the result's rows by a loop of the module's as well (`_fused.assign`). A warning the sum raises is NumPy's for that operation on that line; an error leaves the array as it was.
- **The result is one object.** A lazy array is a handle on a cell that holds a node, three objects, and making and freeing them cost 155 ns of the 180 ns a small `a + b` took over NumPy's 265 ns. The result of a call made at once is a lazy array that holds NumPy's value and nothing else, taken from the instances the core has kept; its node and its cell are made if something asks for them, which the next small call does not ([lazy-design.md](lazy-design.md#the-c-core)).
- **The call is made where the interpreter first looks.** The core's functions stand in the class's table for the operators, for `x[i]` and for `x[i] = v`, so `a + b` is one call from the interpreter to NumPy's ufunc. A reduction method, `x += y`, `float(x)` and a name of the stand-in (`np.tanh`) are answered by the core as well. A call through the stand-in (`np.tanh(x)`, `np.dot(a, b)`) is made without NumPy's way round through its protocols, and for the functions that always give a new array without the rules being asked.
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
- **Which calls wait is the planner's to say.** `fuse.enable()` names the callables a kernel can make and their element type; a large call that is none of them, or reads an array of another type, is made at once. A program of masks, or of complex numbers, writes no graph at all then.
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
| A result laid out differently, so that a later sum adds in another order | Operands are taken only where NumPy's result is in C order, and the output is | A test of layouts the kernels do not read |
| An output that overlaps an input | Every load of a block comes before its store, so the result may be exactly one of the arrays, for a length the module chooses; any other overlap is declined before a kernel runs | Tests of a program written in place at every length around its blocks, and of the module's refusal |
| A warning or an error | The flags are read and NumPy's calls are made | Tests of each, with the line |

**One thing is not specified: which NaN comes of two.** When an operation meets two NaNs of different sign or payload, the processor returns its first operand's. Which operand is first in NumPy's loop is its compiler's choice, and a pattern does not record how the program ordered a sum. Whether a result is NaN is always NumPy's; which NaN it is, is NumPy's whenever a program has one kind of NaN, which is what arithmetic and `np.nan` give.

## What was measured

Everything below is from an Apple M1 Pro, NumPy 2.5.5, CPython 3.13, with the layer's nodes written by its C core. The kernels' specs are not on `main` yet, so the judge's runs are `--unlocked`.

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
- **Each kernel's log has its runs,** in `arm64/attempts.md` beside it.

### The programs, four ways

`uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --size large --plain`. `hotkernels.install()` is in effect for every way but the first, since that is how the package is used: a call it has a kernel for is served by the kernel. **NumPy** is the program as it is, with nothing installed, and what every other way is measured against. **Installed** is the same eager program after `hotkernels.install()`, one operation at a time, with no graph: what the package gives today. **Lazy** is the program with the stand-in, every operation a node, computed one node at a time. **Fused** is the same with `fuse.enable()`: programs of kernels, small calls made at once. The results of the three installed ways are compared bit for bit before anything is printed. Times are the quickest of 5 runs, in milliseconds, from the start of the program to its last result, taken at load averages of 1.6 to 4.2, so they are rough to a few percent; a program's own NumPy time moved by up to 10% between runs of this table.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             42.94     43.06      35.38     0.27    35.12      31.39     7.91     0.29    23.19    1.13x    1.37x     1.37x
attention              28.76     28.84      28.94     0.45    28.48      28.77     0.89     0.07    27.81    1.01x    1.00x     1.00x
perceptron             32.29     25.38      56.38    24.60    31.77      27.22    26.90     0.02     0.30    2.07x    1.19x     0.93x
char_rnn               32.44     28.49     102.19    56.79    45.40      32.71    32.71     0.00     0.00    3.12x    0.99x     0.87x
particle_filter       359.17    321.95     332.03    84.81   247.22     253.37   104.55     2.87   145.95    1.31x    1.42x     1.27x
veros_eos             129.09    125.48     116.76     0.68   116.08      73.67     0.69     0.50    72.48    1.58x    1.75x     1.70x
veros_isoneutral      219.81    219.83     227.51     8.61   218.90     197.52    22.97     0.91   173.65    1.15x    1.11x     1.11x
veros_tke              99.54     97.12     100.97    23.38    77.60      88.34    28.67     0.53    59.14    1.14x    1.13x     1.10x
cavity_flow           156.24    155.82     170.42     7.47   162.96      56.61     8.85     2.88    44.87    3.01x    2.76x     2.75x
hdiff                  69.33     66.01      69.01     5.79    63.22      29.54     5.74     0.04    23.76    2.34x    2.35x     2.23x
black_scholes          87.23     87.18      88.26     1.89    86.38      84.87     1.90     0.04    82.92    1.04x    1.03x     1.03x
trimesh_mesh          175.20    137.48     137.50   137.40     0.11     137.38   137.38     0.00     0.00    1.00x    1.28x     1.00x
sklearn_metrics       978.32    272.88     285.66   284.31     1.35     278.19   278.19     0.00     0.00    1.03x    3.52x     0.98x
empyrical_stats      1255.66    325.03     324.83   324.83     0.00     326.19   326.19     0.00     0.00    1.00x    3.85x     1.00x
skimage_pipeline     1197.12   1112.07    1113.57  1113.57     0.00    1113.06  1113.06     0.00     0.00    1.00x    1.08x     1.00x
pandas_sales          474.84    443.02     443.26   443.26     0.00     442.88   442.88     0.00     0.00    1.00x    1.07x     1.00x
astropy_timeseries   1514.17   1321.09    1310.86  1308.97     1.89    1323.12  1321.95     0.08     1.08    0.99x    1.14x     1.00x
example                49.19     10.85      11.31     5.41     5.90      11.30     9.68     0.01     1.61    1.00x    4.35x     0.96x
flame                 142.21     88.07     171.64    90.90    80.74      91.17    88.82     0.03     2.32    1.88x    1.56x     0.97x
azimint_naive         171.28    155.42     170.03     6.72   163.31     156.10   156.10     0.00     0.00    1.09x    1.10x     1.00x
azimint_hist          122.94     71.36      71.54    71.50     0.04      71.51    71.51     0.00     0.00    1.00x    1.72x     1.00x
mandelbrot1           150.62    143.28     146.81     1.21   145.60     143.82   143.82     0.00     0.00    1.02x    1.05x     1.00x
mandelbrot2           163.15    147.27     181.10    33.62   147.48     148.46   148.46     0.00     0.00    1.22x    1.10x     0.99x
make_decision         142.52     85.33    1404.30  1083.96   320.34     119.00   119.00     0.00     0.00   11.80x    1.20x     0.72x
pandas_filter         175.31     34.16      34.16    34.16     0.00      34.24    34.24     0.00     0.00    1.00x    5.12x     1.00x
equalize_hist         141.56    126.78     127.13   127.04     0.09     126.88   126.88     0.00     0.00    1.00x    1.12x     1.00x
```

`build` is the time outside the layer's computing functions: the program's own Python, the nodes, and whatever is made at once. `compute` is the time inside them, less the planning.

`vs NumPy` is the figure the layer is judged by: the fused run with the kernels installed against the program on plain NumPy. `vs inst.` is the same run against the eager program with the same kernels, which says what the graph adds to what the package already gives.

The lazy and the fused run again with nothing installed, beside the same two eager runs. This says what the graph and its kernels do by themselves; it is not how the layer would be used.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             42.94     43.06      37.66     0.24    37.43      30.96     7.80     0.27    22.89    1.22x    1.39x     1.39x
attention              28.76     28.84      28.77     0.42    28.34      28.56     0.87     0.07    27.62    1.01x    1.01x     1.01x
perceptron             32.29     25.38      64.41    25.51    38.89      35.15    34.78     0.03     0.34    1.83x    0.92x     0.72x
char_rnn               32.44     28.49     105.87    56.77    49.10      36.92    36.92     0.00     0.00    2.87x    0.88x     0.77x
particle_filter       359.17    321.95     368.12    84.91   283.21     289.33   105.02     3.00   181.32    1.27x    1.24x     1.11x
veros_eos             129.09    125.48     116.50     0.67   115.84      73.67     0.71     0.49    72.47    1.58x    1.75x     1.70x
veros_isoneutral      219.81    219.83     227.90     8.55   219.35     196.62    21.53     0.92   174.17    1.16x    1.12x     1.12x
veros_tke              99.54     97.12     103.26    23.37    79.89      90.22    28.54     0.52    61.15    1.14x    1.10x     1.08x
cavity_flow           156.24    155.82     170.40     7.45   162.95      56.58     8.87     2.90    44.82    3.01x    2.76x     2.75x
hdiff                  69.33     66.01      72.44     5.79    66.65      29.65     5.76     0.03    23.86    2.44x    2.34x     2.23x
black_scholes          87.23     87.18      88.33     1.86    86.47      84.99     1.89     0.04    83.07    1.04x    1.03x     1.03x
trimesh_mesh          175.20    137.48     175.39   175.28     0.11     174.82   174.82     0.00     0.00    1.00x    1.00x     0.79x
sklearn_metrics       978.32    272.88    1062.52  1059.45     3.07    1056.04  1056.04     0.00     0.00    1.01x    0.93x     0.26x
empyrical_stats      1255.66    325.03    1257.80  1257.80     0.00    1260.06  1260.06     0.00     0.00    1.00x    1.00x     0.26x
skimage_pipeline     1197.12   1112.07    1192.60  1192.60     0.00    1191.08  1191.08     0.00     0.00    1.00x    1.01x     0.93x
pandas_sales          474.84    443.02     475.07   475.07     0.00     474.68   474.68     0.00     0.00    1.00x    1.00x     0.93x
astropy_timeseries   1514.17   1321.09    1560.24  1557.92     2.33    1536.15  1535.03     0.09     1.03    1.02x    0.99x     0.86x
example                49.19     10.85      50.19    28.24    21.95      47.91    46.14     0.01     1.76    1.05x    1.03x     0.23x
flame                 142.21     88.07     223.36   105.25   118.11     143.44   141.09     0.03     2.33    1.56x    0.99x     0.61x
azimint_naive         171.28    155.42     186.46     6.82   179.64     172.92   172.92     0.00     0.00    1.08x    0.99x     0.90x
azimint_hist          122.94     71.36     122.68   122.64     0.04     122.74   122.74     0.00     0.00    1.00x    1.00x     0.58x
mandelbrot1           150.62    143.28     155.39     1.17   154.21     151.97   151.97     0.00     0.00    1.02x    0.99x     0.94x
mandelbrot2           163.15    147.27     195.67    33.33   162.34     164.11   164.11     0.00     0.00    1.19x    0.99x     0.90x
make_decision         142.52     85.33    1470.75  1079.24   391.51     180.63   180.63     0.00     0.00    8.14x    0.79x     0.47x
pandas_filter         175.31     34.16     175.50   175.50     0.00     174.41   174.41     0.00     0.00    1.01x    1.01x     0.20x
equalize_hist         141.56    126.78     141.98   141.89     0.09     141.54   141.54     0.00     0.00    1.00x    1.00x     0.90x
```

What the graphs were, and what the fused run did with them:

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
regression              384       1      384      3200000       378       6     41     41      41     41      0       0       0      179.2      89.6
attention               220       1      220       262144       198      19      8      8      16     16      0       0       0      174.7     135.4
perceptron             8854     565       33       128000        14    7900      2      2       4      2      0       0       0      163.7       2.7
char_rnn              32423     202    28606        10000         0   27923      0      0       0      0      0       0       0      165.8       0.0
particle_filter        2537      41      480       400000      2436      59    422    422     462    500      0     338      19     1166.5     640.9
veros_eos               366       1      366      1082016       363       3     37     37     140    285      0       0       0     3142.2     675.2
veros_isoneutral        718       1      690      4328064       442      27    121    121     214    242     21      31       0     3230.5    1266.7
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
regression              0.52      0.40       1.19     0.54     0.65       0.50     0.50     0.00     0.00    2.41x    1.04x     0.82x
attention               1.25      1.17       1.79     0.45     1.34       1.61     1.61     0.00     0.00    1.11x    0.78x     0.73x
perceptron              4.42      3.60       7.42     3.01     4.41       3.97     3.58     0.03     0.36    1.87x    1.11x     0.91x
char_rnn                6.63      6.02      20.73    12.95     7.77       7.37     7.37     0.00     0.00    2.81x    0.90x     0.82x
particle_filter         4.37      3.82       6.35     2.47     3.87       4.65     4.65     0.00     0.00    1.36x    0.94x     0.82x
veros_eos               6.78      6.68       7.28     0.33     6.95       5.96     0.34     0.49     5.13    1.22x    1.14x     1.12x
veros_isoneutral       15.09     15.17      16.49     1.58    14.91      14.62     1.96     0.71    11.95    1.13x    1.03x     1.04x
veros_tke               7.37      7.27       8.24     2.29     5.95       7.93     2.66     0.45     4.83    1.04x    0.93x     0.92x
cavity_flow             5.50      5.20      13.66     6.17     7.49       5.51     5.51     0.00     0.00    2.48x    1.00x     0.94x
hdiff                   1.83      1.74       1.82     0.26     1.56       0.93     0.24     0.03     0.65    1.96x    1.98x     1.87x
black_scholes           0.24      0.23       0.31     0.09     0.22       0.27     0.27     0.00     0.00    1.14x    0.88x     0.85x
trimesh_mesh           11.44      9.54       9.65     9.60     0.04       9.58     9.58     0.00     0.00    1.01x    1.19x     1.00x
sklearn_metrics        48.75     19.47      21.80    21.64     0.16      20.44    20.44     0.00     0.00    1.07x    2.38x     0.95x
empyrical_stats      1277.36    358.19     358.69   358.69     0.00     358.20   358.20     0.00     0.00    1.00x    3.57x     1.00x
skimage_pipeline      102.08     93.73      93.90    93.90     0.00      94.34    94.34     0.00     0.00    1.00x    1.08x     0.99x
pandas_sales           32.31     30.07      29.98    29.98     0.00      30.02    30.02     0.00     0.00    1.00x    1.08x     1.00x
astropy_timeseries    201.27    183.74     188.96   187.86     1.10     183.59   183.59     0.00     0.00    1.03x    1.10x     1.00x
example                45.88      8.51       8.87     4.25     4.62       8.86     7.47     0.01     1.38    1.00x    5.18x     0.96x
flame                  16.98      9.77      21.79    12.08     9.71      10.50    10.02     0.02     0.45    2.08x    1.62x     0.93x
azimint_naive          20.28     18.46      32.50     6.68    25.82      19.12    19.12     0.00     0.00    1.70x    1.06x     0.97x
azimint_hist           12.37      7.19       7.40     7.36     0.04       7.31     7.31     0.00     0.00    1.01x    1.69x     0.98x
mandelbrot1            15.70     14.85      17.48     1.10    16.39      14.99    14.99     0.00     0.00    1.17x    1.05x     0.99x
mandelbrot2            16.46     14.62      24.36     7.76    16.60      14.95    14.95     0.00     0.00    1.63x    1.10x     0.98x
make_decision          14.33      8.71      75.78    46.27    29.50      11.81    11.81     0.00     0.00    6.42x    1.21x     0.74x
pandas_filter          17.63      3.58       3.59     3.59     0.00       3.59     3.59     0.00     0.00    1.00x    4.92x     1.00x
equalize_hist          13.00     11.58      11.85    11.79     0.05      11.71    11.71     0.00     0.00    1.01x    1.11x     0.99x
```

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
regression              954       1      954         1600         0     904      0      0       0      0      0       0       0        0.4       0.0
attention               550       1      550         2048         0     520      0      0       0      0      0       0       0        3.0       0.0
perceptron              958      61       33       128000        14     844      2      2       4      2      0       0       0       23.6       2.7
char_rnn               5585     202     2497        10000         0    5135      0      0       0      0      0       0       0       17.9       0.0
particle_filter        1267      21      240         8000         0     940      0      0       0      0      0       0       0       11.7       0.0
veros_eos               366       1      366        72171       363       3     39     39     142    285      0       0       0      209.6      45.0
veros_isoneutral        718       1      690       288684       454      31    135    135     194    198     31      31       0      196.6     106.2
veros_tke               382       6      281       216513       252      39     39     39      58     79     17      28       2       82.5      44.3
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
| `pandas_filter` | 5.13x | 5.12x | 1.00x | 1.01x | pandas, on plain arrays |
| `example` | 4.53x | 4.35x | 0.96x | 1.03x | Single calls on two million elements |
| `empyrical_stats` | 3.86x | 3.85x | 1.00x | 1.00x | empyrical, on plain arrays |
| `sklearn_metrics` | 3.59x | 3.52x | 0.98x | 0.93x | scikit-learn, handed lazy arrays |
| `cavity_flow` | 1.00x | 2.76x | 2.75x | 2.76x | Stencil expressions of 6 to 24 operations, each assigned into part of a grid |
| `hdiff` | 1.05x | 2.35x | 2.23x | 2.34x | Four expressions of a stencil, each one kernel written by hand |
| `veros_eos` | 1.03x | 1.75x | 1.70x | 1.75x | Polynomials, one of 153 operations |
| `azimint_hist` | 1.72x | 1.72x | 1.00x | 1.00x | `np.histogram` |
| `flame` | 1.61x | 1.56x | 0.97x | 0.99x | 7,400 operations on arrays of 25,000 elements |
| `particle_filter` | 1.12x | 1.42x | 1.27x | 1.24x | Random draws, `arctan2`, `exp`; expressions on columns of a table, and on its rows of two and four elements |
| `regression` | 1.00x | 1.37x | 1.37x | 1.39x | A matrix product, a residual, a squared loss |
| `trimesh_mesh` | 1.27x | 1.28x | 1.00x | 1.00x | trimesh, on plain arrays |
| `make_decision` | 1.67x | 1.20x | 0.72x | 0.79x | 490,000 operations on arrays of 32 |
| `perceptron` | 1.27x | 1.19x | 0.93x | 0.92x | 8,900 operations, most on arrays of 8,000 elements |
| `astropy_timeseries` | 1.15x | 1.14x | 1.00x | 0.99x | astropy, on plain arrays |
| `veros_tke` | 1.02x | 1.13x | 1.10x | 1.10x | A tridiagonal solve, masks, an advection flux on arrays that step |
| `equalize_hist` | 1.12x | 1.12x | 1.00x | 1.00x | scikit-image's histogram equalisation |
| `veros_isoneutral` | 1.00x | 1.11x | 1.11x | 1.12x | Expressions on slices of a grid; 21 of 121 are given up for arrays that step |
| `mandelbrot2` | 1.11x | 1.10x | 0.99x | 0.99x | Complex arithmetic on arrays that shrink to nothing |
| `azimint_naive` | 1.10x | 1.10x | 1.00x | 0.99x | A thousand masks over 200,000 elements |
| `skimage_pipeline` | 1.08x | 1.08x | 1.00x | 1.01x | scikit-image, on plain arrays |
| `pandas_sales` | 1.07x | 1.07x | 1.00x | 1.00x | pandas, on plain arrays |
| `mandelbrot1` | 1.05x | 1.05x | 1.00x | 0.99x | Complex arithmetic through a mask |
| `black_scholes` | 1.00x | 1.03x | 1.03x | 1.03x | `erf`, `log`, `exp`; arithmetic between them |
| `attention` | 1.00x | 1.00x | 1.00x | 1.01x | float32 matrix products and `tanh` |
| `char_rnn` | 1.14x | 0.99x | 0.87x | 0.88x | 32,000 operations on arrays of 100 to 10,000 |

- **The two kinds of kernel add up, and neither gets in the other's way.** Six of the seven programs fusion speeds up gain 6% or nothing from `install()` eagerly, because no kernel of the package is in their expressions, and their fused runs are within 2% with it and without. The programs that are a library's work or single calls gain 1.07x to 5.3x from `install()` eagerly and nothing from fusion, and their fused runs keep the kernels' gain to within 4%. The particle filter has both: 322 ms eagerly with the kernels for 359 ms without, 289 ms fused without them, and 253 ms with both. The regression's and the transformer block's eager times moved by 3% between the two eager runs of this table, and that is all their first column says.
- **Where the fused run is behind the eager program with the same kernels, it is by what a call made at once costs.** `make_decision`, the recurrent network and the perceptron make 410,000, 28,000 and 7,900 calls at once and write no node: 119 ms, 32.7 ms and 27.2 ms against 85 ms, 28.5 ms and 25.4 ms. The example is 4% behind, 0.5 ms, and the fractal flame 3%, 3 ms. The naive azimuthal integration was 4% behind as well, 7 ms, most of it arithmetic on a number that came out of an array, `rmax * i / npt`, which went through the operator's ufunc at 600 ns an operation where the eager program's scalar has its own in 50 ns. It is the scalar's own now ([lazy-design.md](lazy-design.md#the-c-core)), and the program runs level.
- **A kernel the lazy run does not reach shows in the same column.** The second Mandelbrot program takes its grids from `np.mgrid`, which gave plain `ndarray`s, and selects from them by a lazy mask: NumPy's own indexing, with the package's kernel for `x[mask]` never asked. It ran in 172 ms fused with the kernels installed, the same as without them, where the eager program runs in 148 ms with them. With the arrays `np.mgrid`, `np.ogrid`, `np.r_` and `np.c_` make lazy ([lazy-design.md](lazy-design.md#the-stand-in-for-the-module)) it runs in 148 ms, for the eager program's 147 ms. The package's own count of the calls it served was then compared, eager run against fused, for eleven other programs, and none loses a kernel that way: the counts differ by the layer's dry runs on one element, by the calls a fused kernel made in a package kernel's place, and for scikit-learn by the library's own different road.
- **One pass for a whole expression is worth what its passes were.** The cavity flow's fused run writes 0.3 GB of new arrays where the node-by-node run writes 4.1 GB, and computes in 45 ms what took 163 ms. Its 560 kernel calls are 260 programs; 2,980 of its 6,900 nodes have no array of their own. The equation of state writes 0.7 GB instead of 3.1 GB and computes in 72 ms instead of 116 ms: 153 operations in 64 kernel calls, all on blocks.
- **A write in place is the other half of the cavity flow.** Its 1,220 writes into slices and boundary rows are made where the array is: all of them in the lazy run, and in the fused one 1,196 by their programs and 24 at once. As copies they were 54 ms.
- **A kernel for every step is what covers a long tail.** Sixty-nine of the eighty kernels ran over the corpus, the single operations among the most called: `add(S0,V0)` 67 times, `mul(S0,V0)` 63, `sub(V0,S0)` 56. Each of those is a step inside a block, where otherwise it is a pass of NumPy's between two kernels.
- **The particle filter runs 1.42x, most of fusion's part on rows of two and four elements.** Its table is 100,000 rows of four numbers, and NumPy makes `particles - mean`, `centred * weights[:, None]` and `particles[:, :2] += particles[:, 2:] * dt` a row at a time: 544 µs, 294 µs and 1,256 µs, where the same arithmetic in one run takes 101 µs, 101 µs and 82 µs. The first is one kernel's call now and the others the module's loop over the rows ([Computing a group](#computing-a-group)), and the expressions on the table's columns are gathered into blocks. With the rows left to NumPy the fused run was 2% ahead of NumPy with nothing installed, where it is 24% ahead. Nineteen of its writes into the table go to a copy, because an earlier reader of the table still waits.
- **Planning is 5% of the cavity flow's fused run,** 2.9 ms of 57 ms for 6,900 nodes, with the expressions read by the C core; read in Python it was 15 ms of 69 ms. For every other program it is about 1% or less.
- **A program of masks or of complex numbers writes no graph.** The naive azimuthal integration makes 9,000 calls at once, comparisons and selections on 200,000 elements, and is 10% ahead of NumPy, level with the eager program with the kernels. The two Mandelbrot programs are 1% behind theirs.
- **At the small sizes the horizontal diffusion, the equation of state and isoneutral mixing gain from fusion,** 1.97x, 1.14x and 1.03x against NumPy, and the programs that are a library's work keep the kernels' gain, up to 5.2x. Five programs are behind NumPy there, at 0.78x to 0.94x, and against the eager program with the kernels the fused run is from 0.73x to 1.87x: at these sizes most calls are small, and each pays what a call made at once costs.
- **Arrays that step are gathered where the program is long enough to pay for the copies.** The two Veros routines keep their state with a time level as its last axis, so `tke[:, :, :, tau]` is every third element. The kinetic energy's advection flux, seventeen operations with its Courant number on three such arrays, is two kernels written for its halves and one copy of each array, and the routine runs in 88 ms where it ran in 93 ms. 11 of its 38 programs and 21 of isoneutral mixing's 121 are still given up: one to five operations, where the copies cost what the passes save. [What is left](#what-is-left) has the measurements.
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
regression             41.49     38.59     29.92     32.82    1.39x     1.26x     2.90      8.8%
attention              29.27     28.71     26.59     28.68    1.10x     1.02x     2.09      7.3%
perceptron             32.93     25.20     20.13     28.52    1.64x     1.15x     8.39     29.4%
char_rnn               32.60     28.81     22.67     33.13    1.44x     0.98x    10.46     31.6%
particle_filter       371.57    331.45    258.05    263.80    1.44x     1.41x     5.75      2.2%
veros_eos             136.64    135.37    121.69     74.63    1.12x     1.83x   -47.05    -63.0%
veros_isoneutral      224.11    223.98    174.64    199.56    1.28x     1.12x    24.92     12.5%
veros_tke             101.49     98.80     74.46     89.28    1.36x     1.14x    14.82     16.6%
cavity_flow           159.08    158.33    115.64     57.74    1.38x     2.76x   -57.90   -100.3%
hdiff                  70.37     66.83     62.82     30.04    1.12x     2.34x   -32.78   -109.1%
black_scholes          88.60     88.65     86.20     86.60    1.03x     1.02x     0.40      0.5%
trimesh_mesh          176.98    139.27    139.26    137.89    1.27x     1.28x    -1.36     -1.0%
sklearn_metrics       984.82    276.74    276.68    281.65    3.56x     3.50x     4.96      1.8%
empyrical_stats      1272.36    329.04    329.04    329.51    3.87x     3.86x     0.47      0.1%
skimage_pipeline     1219.41   1132.56   1132.56   1135.71    1.08x     1.07x     3.16      0.3%
pandas_sales          479.78    448.76    448.76    449.94    1.07x     1.07x     1.19      0.3%
astropy_timeseries   1573.30   1343.67   1343.45   1338.07    1.17x     1.18x    -5.38     -0.4%
example                49.16     10.87     10.65     11.21    4.61x     4.39x     0.55      4.9%
flame                 141.33     88.38     79.25     91.21    1.78x     1.55x    11.96     13.1%
azimint_naive         173.68    157.63    151.67    158.72    1.15x     1.09x     7.05      4.4%
azimint_hist          124.11     71.60     71.59     72.46    1.73x     1.71x     0.87      1.2%
mandelbrot1           153.61    144.86    132.48    146.19    1.16x     1.05x    13.71      9.4%
mandelbrot2           164.89    149.75    149.23    149.78    1.10x     1.10x     0.55      0.4%
make_decision         147.31     85.63         -    120.12        -     1.23x        -         -
pandas_filter         174.81     34.24     34.24     34.16    5.11x     5.12x    -0.08     -0.2%
equalize_hist         141.66    126.73    126.72    126.92    1.12x     1.12x     0.20      0.2%

`NumPy` is the program as it is; every other time is under `hotkernels.install()`. `floor` is the installed eager
time less everything that could be removed from the program's element-by-element operations and its reductions: what
fusing all of them, at no cost, would leave. `at best` is NumPy over the floor, `now` NumPy over the fused run, and
`left` the fused run less the floor. A program with `-` for a floor has none that means
anything: its arrays are so small that what could be removed from an operation, the call, is all the operation costs.

Where the time is with the kernels installed, node by node, and how much of each class could be removed

program              outside        arithmetic             masks           library       other types        reductions          products             moves              rest
                   the graph    ms   removable    ms   removable    ms   removable    ms   removable    ms   removable                ms                ms                ms
regression              1.43     7.02     3.28     0.00     0.00     0.00     0.00     0.00     0.00     9.41     5.39             20.68              0.00              0.04
attention               0.12     2.64     0.55     0.00     0.00    17.03     0.06     1.96     1.08     1.14     0.42              5.71              0.00              0.10
perceptron              0.00     5.57     3.72     0.79     0.28     1.06     0.28     0.00     0.00     4.89     0.79              8.73              7.37              0.34
char_rnn                0.00     3.76     3.13     0.00     0.00     2.15     1.58     0.00     0.00     1.44     1.43             14.67             18.33              1.78
particle_filter        79.25    93.80    72.07     1.56     0.21    64.88     0.66     1.02     0.15     2.14     0.32             13.64             62.52             12.65
veros_eos              19.55   109.64    13.67     0.00     0.00     6.18     0.02     0.00     0.00     0.00     0.00              0.00              0.00              0.00
veros_isoneutral        1.96   140.96    48.96     0.00     0.00    52.75     0.38     0.00     0.00     0.00     0.00              0.00             28.17              0.13
veros_tke              20.17    57.86    21.94    10.06     2.38     0.00     0.00     0.03     0.02     0.00     0.00              0.00             10.60              0.08
cavity_flow             0.00   148.65    42.69     0.00     0.00     4.83     0.00     0.00     0.00     0.00     0.00              0.00             11.32              0.47
hdiff                   2.84    48.60     2.21    13.48     1.81     0.00     0.00     0.00     0.00     0.00     0.00              0.00              1.89              0.02
black_scholes           0.38    12.82     1.77     0.00     0.00    74.68     0.69     0.00     0.00     0.00     0.00              0.00              0.78              0.00
trimesh_mesh          139.20     0.00     0.00     0.00     0.00     0.03     0.01     0.04     0.00     0.00     0.00              0.00              0.00              0.00
sklearn_metrics       275.50     0.00     0.00     0.51     0.04     0.00     0.00     0.00     0.00     0.74     0.01              0.00              0.00              0.00
empyrical_stats       329.04     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
skimage_pipeline     1132.56     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
pandas_sales          448.76     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
astropy_timeseries   1342.38     0.35     0.14     0.00     0.00     0.01     0.01     0.63     0.02     0.13     0.06              0.00              0.16              0.02
example                 5.05     0.63     0.00     1.37     0.04     0.64     0.00     0.32     0.00     0.79     0.17              0.00              1.28              0.77
flame                  13.40     7.21     4.07     1.77     0.75    16.86     3.74     1.05     0.54     0.06     0.04              0.00             48.03              0.00
azimint_naive           0.00     3.63     3.63   144.32     0.52     0.00     0.00     0.00     0.00     1.83     1.81              0.00             14.84              0.00
azimint_hist           71.59     0.00     0.00     0.00     0.00     0.00     0.00     0.01     0.01     0.00     0.00              0.00              0.00              0.00
mandelbrot1             0.00     0.00     0.00    11.51     0.07     0.00     0.00    81.59    12.31     0.00     0.00              0.00             55.93              0.00
mandelbrot2             3.49     0.00     0.00     7.65     0.14     0.00     0.00    47.71     0.38     0.00     0.00              0.00             90.88              0.02
make_decision           0.00    20.88    20.55     0.00     0.00     0.00     0.00    79.50    73.19    11.31    11.19              0.00            156.07              0.00
pandas_filter          34.24     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00     0.00              0.00              0.00              0.00
equalize_hist         126.71     0.00     0.00     0.00     0.00     0.00     0.00     0.01     0.01     0.00     0.00              0.00              0.00              0.01

`outside the graph` is the eager time that is no node's: the program's own code, a library's, and calls the layer
does not defer. A class's `removable` is its time less what the same NumPy loops take for as many elements on arrays
of 512 to 4096 elements, with the cost of the call left out: the passes over memory and the calls.

What the kernels took for the nodes they computed

program                 nodes   kernels ms     NumPy ms  in cache ms
regression                 41         3.79         7.02         3.75
attention                  16         0.80         2.64         2.09
perceptron                  2         0.13         5.57         1.85
char_rnn                    0         0.00         3.76         0.62
particle_filter           500        45.60        93.80        21.73
veros_eos                 285        56.09       109.64        95.97
veros_isoneutral          242        88.46       140.96        92.00
veros_tke                 115        36.21        57.86        35.92
cavity_flow              2980        36.87       148.65       105.96
hdiff                      15        18.97        48.60        46.39
black_scholes              14         7.53        12.82        11.05
trimesh_mesh                0         0.00         0.00         0.00
sklearn_metrics             0         0.00         0.00         0.00
empyrical_stats             0         0.00         0.00         0.00
skimage_pipeline            0         0.00         0.00         0.00
pandas_sales                0         0.00         0.00         0.00
astropy_timeseries          0         0.00         0.35         0.21
example                     0         0.00         0.63         0.63
flame                       0         0.46         7.21         3.14
azimint_naive               0         0.00         3.63         0.00
azimint_hist                0         0.00         0.00         0.00
mandelbrot1                 0         0.00         0.00         0.00
mandelbrot2                 0         0.00         0.00         0.00
make_decision               0         0.00        20.88         0.33
pandas_filter               0         0.00         0.00         0.00
equalize_hist               0         0.00         0.00         0.00

`NumPy ms` is what NumPy took for every operation of the class a kernel can make, and `in cache ms` what its loops
take for as many elements in the cache.
```

- **The floor is not a floor for a kernel.** It is what NumPy's own loops would take with no pass over memory and no call. A kernel makes several operations for each load and store, which NumPy's loops cannot, so the cavity flow, the horizontal diffusion and the equation of state run below it: the cavity flow's kernels took 36.9 ms for arithmetic that NumPy's loops need 106.0 ms for in the cache. The particle filter is at it, 6 ms of 264 ms above.
- **Fifteen programs have 10% at most to gain.** Their time is outside the graph, in a library or in calls the layer does not defer, or in a mathematics library, matrix products, or moving elements: a selection by a mask or by positions, a write through one. Their floors are 0% to 10% below the eager program with the kernels, which is the whole of what any fusion could add, and their fused runs are within 5% of that program.
- **Four programs are on arrays below the limit.** `make_decision`, on arrays of 32 elements, has no floor that means anything: what the tool finds removable there is the call into NumPy, which is nearly all such an operation costs. The floors of the recurrent network, the perceptron and the fractal flame, whose arrays have 100 to 25,000 elements, are low for the same reason; see [small arrays](#small-arrays-what-a-call-costs) below.
- **Three have something left:** the kinetic energy 15 ms and isoneutral mixing 25 ms, 17% and 12% of their fused runs, and the regression 2.9 ms, which was 4.5 ms in another run and nothing in one with nothing installed; its own time moves by 3 ms between runs. The kinetic energy was 19 ms above before its flux was taken.
- **What the tool finds removable moves between runs** for an operation whose cost depends on its values: the complex arithmetic of the first Mandelbrot program came out at 11 ms to 12 ms in four runs and 20 ms in another.

### Where the rest is, class by class

| What is left | In which programs | How much | Why a pattern does not take it |
|---|---|---|---|
| Calls into the platform's mathematics library: `exp`, `log`, `erf`, `tanh`, `arctan2`, `%`, `hypot` | Black-Scholes, isoneutral mixing, the particle filter, the fractal flame, the transformer block | 226 ms over the five, of which the tool finds 7.8 ms removable | It costs the same in the cache as out of it. A kernel is a leaf function and calls nothing; a second implementation would not have the library's bits |
| Calls on small arrays | `make_decision`, the recurrent network, the perceptron | All of what they are behind the eager program with the kernels: 34 ms of 119 ms, 4.2 ms of 32.7 ms, 1.8 ms of 27.2 ms | Nothing is written down for them, so there is nothing to match. Writing a node costs more than the call it would save |
| Arrays that step over elements, in rows of 25 or 26, in expressions of one to five operations | The two Veros routines, whose state has a time level as its last axis | 11 of the kinetic energy's 38 programs and 21 of isoneutral mixing's 121 are given up | A copy for each such array costs what the passes saved, and a loop that reads them where they lie is 1.14 times NumPy at best; see below |
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
| An array that steps copied into a run, by itself | | 458 µs | What a copy costs before any arithmetic: a third of NumPy's two passes over it. It is why gathering loses on one kernel and gains little on thirteen operations |
| `np.where(x > 0, a, b)` with the comparison inside | 842 µs | 545 µs | 1.5 times on what the tool puts at 2.5 ms in the kinetic energy: 1 ms of 93 ms |

### What was built on it and did not gain

Each of these was built in the layer and timed, and none is in the code. The figures are from the same machine as the tables, in runs with nothing installed: each compares two builds of the module or the core on the same program.

- **A mathematics library called a block at a time.** Black-Scholes was written with NumPy's own loops on blocks of 4,096 elements, `erf`, `log` and `exp` among them, into buffers used again: 84.5 to 87.7 ms against 86.2 ms for the program as written, with the same bits. The arithmetic between the library calls is 13 ms of its 87 ms, and the kernels have it already.
- **Arrays that step gathered for every program that reads one,** over the rows they walk at one step: the kinetic energy took 99.8 to 100.8 ms where it took 93.9 to 95.7 ms. The programs it lost on are one kernel that reads two such arrays, where two copies replace one pass. That is where the rule of operations against copies comes from, and with it gathering is in the code ([Computing a group](#computing-a-group)).
- **A program for each short row, in place of declining,** for rows of 25 at more than one pitch: isoneutral mixing 194 to 215 ms against 196 to 201 ms, the kinetic energy 94 to 99 ms against 97 ms. On an expression by itself it is 1.16 times NumPy's passes, and it is what the module does from 16 elements a row; on the two routines it cannot be told from nothing.
- **Large views left pending,** so that a write through one could wait and be fused with what it stores: the particle filter the same, 325 to 326 ms either way, and the cavity flow 54.1 to 54.7 ms against 52.0 to 52.1 ms, for the nodes its thousands of views then are.

### Small arrays: what a call costs

The three programs on small arrays write no graph, and what they are behind the eager program by is what a call made at once costs over NumPy's own, 15 to 115 ns ([As the graph is built](#as-the-graph-is-built)). Fusing needs the operations written down first. A node costs 0.32 to 0.35 µs to write ([lazy-design.md](lazy-design.md#what-writing-a-node-costs)), and NumPy makes the whole call on 16 elements in 0.26 to 0.42 µs. So an expression of `k` operations on arrays that small costs `k` nodes, a plan and the running of a group, against `k` calls that each cost about what the node does. The measurement is the limit itself ([Planning](#planning)): with nodes written down from 8,192 elements instead of 32,768, the fractal flame, whose arrays have 25,000, runs in 235 ms where it runs in 148 ms, and the perceptron, whose arrays have 8,000, in 76 ms for 35 ms. What would take these programs is compiling the body of their loops once, from a trace of it, which is another design and not a pattern.

### What a pattern could still take

Nothing that a loop written for it shows to be worth a kernel:

- **The expressions of one to five operations on arrays that step,** which the two Veros routines still give up: a loop that reads such arrays where they lie is 1.14 times NumPy's calls on the commonest of them, about 2% of each routine if every one gained that. It would need each kernel written a second time with a step for every array it reads, and the judge to describe arrays that step.
- **Complex arithmetic, larger kernels, masks:** the loops above.

That is a statement about these twenty-six programs on this machine:

- **x86-64.** The kernels are ARM64 only; on another architecture each is built from its C reference, which the judge holds to the same bits. `make_kernel.py` writes NEON; the same tree walk would write AVX2. Nothing was timed there.
- **More programs.** [Issue 298](https://github.com/sofeikov/hotkernels/issues/298) lists 21 programs that come out as one graph each, most of them stencils and loops over rows. What is left on this corpus says nothing of them.

### What is open in the layer, and is not fusion

- **What handing a lazy array to a library costs.** Scoring a classifier with scikit-learn is 2% slower with lazy arrays handed in than with `ndarray`s when the kernels are installed, and 7% with nothing installed, with seven operations deferred in all. The time is not in the layer: the library takes a longer road for an array type it does not know. And scikit-image fails on one. A program whose work is a library's gains nothing from the layer; whether it should lose anything is open.
- **More of NumPy in the deferred set.** The nine workloads make calls the layer does not defer: `np.histogram`, `np.bincount`, `np.interp`, `np.isin`, `np.unique`, `np.stack`, `x.astype(...)`, `np.clip` without `out=`. None of them sits inside an expression the kernels could take, but for `np.clip`, which is a maximum and a minimum, and a cast, which is a kernel nobody wrote.
- **The limit below which a call is made at once.** 32,768 elements is in the middle of a range, 16,384 to 65,536, over which no program of the corpus runs 1% differently. Two programs have arrays near it, which is two programs' evidence.
- **A sum at the end of a program, and reductions along an axis.** `sum(sq(V0))` is a kernel of its own; a sum of a longer expression computes the expression into an array first, since NumPy's order of addition is over the whole array and not over blocks. `np.sum(e ** 2, axis=-1)` and `np.max(np.abs(e))` are left to NumPy. The tool finds 6 ms removable in the regression's reductions, of which its kernel for a sum of squares has a part, and under 2 ms in any other program on large arrays.

## What changed outside `fuse.py`

- **`evaluate.py` asks a planner,** and has a meter and a clock for the tools. With none set it computes node by node. It decides, for each write it computes, whether the array's own memory may be written (`reusable`), and it catches warnings and puts the error settings once for a computation. [lazy-design.md](lazy-design.md#computing) says what a planner may and may not change, and [its section on writes](lazy-design.md#writes) when a write copies.
- **`array.py` and the C core** have `at_once_below`, the roads for calls made at once, `x ** e` as `operator.pow`, and `x.shape = s`. The core makes and frees the layer's objects, gives a call made at once a lazy array that holds its value alone, stands in the class's table for the operators, takes the nodes of a computation one after the other, and reads the expressions for `fuse.find` ([lazy-design.md](lazy-design.md#the-c-core)). A view taken at once holds NumPy's view alone, and a write does not wait while one may be looking ([lazy-design.md](lazy-design.md#pending-work-and-writes)). Without the switch set, a graph is written node for node as before, which the layer's tests of what is written check.
- **`namespace.py`** makes the arrays of `np.mgrid`, `np.ogrid`, `np.r_` and `np.c_` lazy, so that what a program does with its grids is the layer's to make, and a kernel of the package's to serve ([lazy-design.md](lazy-design.md#the-stand-in-for-the-module)).
- **The judge looks for a kernel in a second place,** `prototypes/lazy/kernels/`, and locks its spec and reference to `main` there as it does in `kernels/`. Nothing else in the judge changed.
- **`programs/corpus.py`** has every program the repository times, `fusion_time.py` times each four ways, and `limit.py` measures what no fusion could remove from each ([What is left](#what-is-left)).
- **Nothing was added to the deferred set.** `x.shape = s` is a view, which was already deferred; the calls the new programs make at once are listed under [What is open in the layer](#what-is-open-in-the-layer-and-is-not-fusion).
