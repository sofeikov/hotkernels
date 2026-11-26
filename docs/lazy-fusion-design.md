# Fusion by pattern in the lazy layer

This is fusion in the prototype of the lazy layer ([lazy-design.md](lazy-design.md), [`prototypes/lazy/`](../prototypes/lazy)), and what it measured. Nothing here is in the package. The layer writes a program's NumPy operations down as a graph; this document is about computing that graph faster than one NumPy call per operation, with NumPy's bits.

The approach is the plainest one there is. A **pattern** is a small tree of operations with named inputs, such as `a + s * b`. Where the graph holds that tree, a kernel written for it computes it in one pass over memory. A longer expression is covered with several patterns, and their kernels are called block by block, so that what one gives the next never leaves the cache. An operation no kernel is written for is computed as before, so nothing is lost by a pattern that is missing.

## Summary

- **A fused kernel gives NumPy's bits, and the judge checks it.** Every operation in one is a single IEEE operation that NumPy makes with the same instruction, in the program's order, each rounded by itself. No multiply-add is fused and no sum is regrouped. There are seventy-eight kernels, in NEON assembly for ARM64, and each passes the judge against its NumPy expression on 20,000 to 280,000 cases. See [Why the bits are NumPy's](#why-the-bits-are-numpys).
- **The pattern is the kernel's own oracle.** A kernel's `spec.py` names the NumPy expression it must equal. Run on lazy arrays, that expression writes a graph, and the shape of that graph is the pattern. Nothing else says what a kernel serves. See [The library](#the-library).
- **A whole expression is one pass over memory, whatever its length.** Its kernel calls are run a block of 4,096 elements at a time, so what one kernel gives the next stays in the cache. For that every operation has a kernel, the single operations too. See [An expression block by block](#an-expression-block-by-block).
- **Most kernels are written by a program from the name of their shape.** Fourteen were written by hand; `make_kernel.py` wrote sixty-four, each with its spec and its C reference, and the judge holds them to the same bits. See [The kernels](#the-kernels).
- **Which shapes to write was mined, not guessed.** Twenty-six programs were run lazily with every call timed: every program this repository times with and without `hotkernels.install()`, and six more. See [Finding the shapes](#finding-the-shapes).
- **A write is made where the array is.** A deferred write is a copy only when something still reads what the array held, and with fusion an expression assigned to part of an array is computed straight into the part. See [Writes](#writes).
- **What cannot be fused is not written down.** With fusion on, a call that gives fewer than 32,768 elements, or that no kernel could make, on arrays that have their values, is NumPy's call at that line, not a node. See [As the graph is built](#as-the-graph-is-built).
- **Where a program is expressions on large arrays, the fused run is 1.1x to 2.3x faster than plain NumPy:** a cavity flow 2.27x, a weather model's horizontal diffusion 2.34x, an ocean model's equation of state 1.72x, a regression 1.18x, isoneutral mixing 1.12x. Against the eager program with the single-operation kernels installed the figures are the same, since those kernels serve none of these expressions.
- **Where a program is a library's work, masks, or one call at a time, the lazy layer adds nothing and costs little:** sixteen of the twenty-six run within 5% of NumPy fused, and with `hotkernels.install()` in effect they keep what the package gives eagerly, up to 5.3x.
- **Where a program is many calls on small arrays, the fused run is still slower than NumPy, by 1.2x to 1.9x,** where the lazy run is slower by 2x to 12x. What is left is what a lazy array costs to make, two tenths of a microsecond a call. See [What the measurements show](#what-the-measurements-show).

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
- **Every operation needs a kernel for that,** also one that stands alone between two shapes. So there is a kernel for each single operation: `add(V0,V1)`, `mul(S0,V0)`, `div(V0,S0)`, `sqrt(V0)` and the rest. They fuse nothing by themselves. An operation by itself in a program, with nothing to fuse with, is still NumPy's call.
- **A program is the tiles that feed one another.** A tile whose operand is an operation with no kernel (`np.where`, `x ** 3`) reads that operation's node, which NumPy computes first as an array; what lies below it is a program of its own.
- **The steps are in the order of the tree, the deepest first,** and a buffer is free again as soon as the step that read it is written, so a long polynomial needs a handful of buffers.
- **The block is 4,096 elements.** Timed from 256 to 32,768 on the programs of the corpus, the equation of state runs in 83 ms at 256, 77 ms at 1,024, 72 ms at 4,096 and 68 ms at 32,768, and isoneutral mixing the other way, 203 ms to 209 ms; the cavity flow and the others do not move. A program of one step has nothing to keep in the cache and is one call for the whole run.

[`_fused.program`](../prototypes/lazy/_fused.c) checks a program before it computes anything: every step against its kernel's number of operands, every place a step reads against what was written before it, and that only the last step writes the result.

## The kernels

Seventy-eight, in [`prototypes/lazy/kernels/`](../prototypes/lazy/kernels/README.md), one directory each with a spec, a C reference, the assembly and its log, as the kernels of `kernels/` have. The judge finds a kernel by name in either place ([judge-design.md](judge-design.md#layout)) and holds each to its NumPy expression, bit for bit, on 20,000 to 280,000 cases. The package builds from `kernels/` alone and does not contain these.

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


### Sixty-four written from the names of their shapes

A shape's name says everything its kernel has to do: which operations, on which operands, in which order. The loop for it is always the same loop: load eight elements of each array, make the operations in the order the tree gives, each one instruction that rounds once, and store. [`make_kernel.py`](../prototypes/lazy/make_kernel.py) writes it.

```bash
uv run python -m prototypes.lazy.make_kernel "sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))"
uv run judge check fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64 --unlocked
```

- **It writes the four files of a kernel:** `spec.py`, whose oracle is the shape as a NumPy expression, with every number tried at zero of either sign, at an overflow, a denormal, an infinity and a NaN; `naive.c`; `arm64/kernel.s`; and the first entry of the log.
- **The kernel is then a kernel like the others.** Nothing the tool writes is trusted: the judge checks it against NumPy, and the pattern it serves is read off its oracle. A test holds the files in the repository to what the tool writes today.
- **Registers are given to the tree by the usual rule,** the operand that needs more of them first, four registers to a value. A number is held in every lane of one register. Shapes of up to six arrays and eight numbers fit.
- **Which shapes:** the eighteen single operations (the four of arithmetic on two arrays and on an array and a number either way round, a maximum and a minimum, a negation, an absolute value, a square root, a square), and the forty-six shapes of two to six operations that cover the corpus's expressions best, as the miner chose them one at a time.
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
2. **Expressions are read off the nodes,** the newest first. An operation is the top of an expression unless an expression already found contains it; below it, each operand is inside the expression if it is an operation a kernel can make, of the same shape, about to be computed, read by this operation alone, and held by nothing else.
3. **Each expression is covered with shapes of the library by the fewest passes,** found from the leaves up: at each operation, the shapes with that operation at their top are held against the tree. A shape whose top operation could not take the operation's operands, by what each is (a number, an array, a named operation), is not walked at all.
4. **The tiles that feed one another become one program,** a group under the node whose value it gives. The plan is the groups.

**What "held by nothing else" means.** An intermediate the program has a name for, or that a node outside this computation reads, will be asked for again. Both are a reference to the node, and CPython counts references exactly: a node of a computation is referred to by the nodes of it that read it, by the one list it is in, and by nothing else unless one of the two holds. The count decides only what is worth fusing. A node that a plan skips stays pending and is computed when asked for, so a wrong count costs time and never a result. Two things this found while it was built are worth knowing by anyone who touches the code: a `for` loop's variable still refers to the last node after the loop, and pytest's rewriting of `assert` keeps every value of the expression in it alive. Both make a node look held.

**Nothing planned outlives the computation.** A plan refers to nodes and so to arrays. If anything kept a plan alive past its computation, those arrays would stay in memory until Python's collector of cycles next ran, and every result in between would be written to fresh memory where NumPy's own calls reuse what was just freed. This happened in the building: two functions that called themselves by a name of the function they were made in formed a cycle, one run of an ocean routine took 46,000 page faults where the run without fusion takes one, and was a quarter slower for it. A test now runs a fused computation with the collector off and holds that no node is left.

**A plan is remembered by the shape of its expression.** A loop writes the same expressions at every turn, and covering a tree with shapes was most of what planning one cost. What is kept is the program and where each thing it reads is among the expression's operands, by the tree's operations and the pattern of its operands. The next expression of that shape is planned by a look-up. Planning the cavity flow's 260 expressions costs 16 ms with that memory and 45 ms without it.

**Arrays of fewer than 32,768 elements are left alone.** Reading an expression off a graph costs about two microseconds a node, and a pass of NumPy's over fewer elements than that costs little more. A computation none of whose nodes is that large is not read at all. With fusion on, such operations are not nodes in the first place ([As the graph is built](#as-the-graph-is-built)). The floor is where it is because of one program, the fractal flame, whose arrays have 25,000 elements: it runs in 158 ms with the floor at 16,384 and in 145 ms at 32,768, and no other program of the corpus changes.

## Computing a group

A group is computed when `evaluate` reaches the node it gives the value of. [`_fused.c`](../prototypes/lazy/_fused.c) stands between the kernels, which are leaf functions over contiguous runs, and arrays of any shape.

- **It walks the outer axes and runs the program once per contiguous run,** block by block. Every outer axis that every operand walks straight through joins the run, so a C-contiguous expression of any number of dimensions is one run. `u[1:-1, 1:-1]` of a grid is one run per row.
- **An array that is the same along an outer axis does not end a run.** `dzt[None, None, :]` beside a grid of 204 x 204 x 26 would make every run 26 elements long. Instead the run is lengthened all the same, and that array is read from a buffer that holds its 26 elements over and over for the length of a block.
- **An array that steps over elements is gathered.** A column of a grid, `particles[:, 0]`, is not memory a kernel can read. Its elements are copied into a block for the steps, and the result's block is copied out if the result steps too: a copy for each such array, within the cache.
- **It reads float64 arrays** with strides that are positive and do not grow from one axis to the next. NumPy's result for such operands is laid out in C order, which the output is.
- **It declines anything else:** a reversed or Fortran-ordered array, a transposed one, an output that overlaps an operand without being it. And a program of several steps whose runs are shorter than 32 elements: a slice that cuts the innermost axis, `x[2:-2, 2:-2, :-1]`, cannot be joined across rows, and for rows of 25 elements gathering was timed slower than NumPy's own passes (isoneutral mixing: 217 ms gathered, 206 ms declined). The group's nodes are then computed by their own calls, in order. Declining is decided before anything is computed or copied.

### Floating-point exceptions

NumPy reports overflow, division by zero and invalid operations as warnings, or as errors under `np.errstate(over="raise")`, with the name of the operation and on the program's line. A kernel of several operations cannot say which of them raised a flag.

So the module clears the processor's flags before a kernel runs and reads them after. If a flag is raised that NumPy would report under the `np.errstate` the operations were written under, **the kernel's result is dropped and NumPy's own calls are made**, one node at a time. The warning, the error, its words and its line are then NumPy's by construction, and so are the values.

- **An exception the program silenced costs nothing.** Under `np.errstate(all="ignore")`, and for underflow always, the kernel's result is kept.
- **An expression that raises one is computed twice.** The line that wrote it is remembered, and its expressions are left to NumPy from then on, so a loop pays once.
- **A comparison raises a flag NumPy does not report.** The processor's comparison marks a NaN operand as an invalid operation; NumPy's comparison clears the flag. The fused flux limiter is therefore computed by NumPy's calls when its data has a NaN, unless the program has silenced invalid operations.

### A write made at once

A write into memory the program holds makes the layer compute everything pending first, after which nothing may be pending ([lazy-design.md](lazy-design.md#writes)). The nodes inside a group have no value of their own. Nothing should hold one, or the plan would not have put it inside; `evaluate` checks all the same, and computes any that is still alive after the planned computation, since the state after it must not depend on a count of references.

## As the graph is built

A node costs time three times: when it is written, when its computation is planned, and when the loop that computes it comes to it. NumPy makes a call on an array of a hundred elements in a third of a microsecond. A program on small arrays runs two to twelve times slower lazily than eagerly for that reason alone, and nothing that computes a graph faster can win that back, since there is nothing to fuse in it.

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
- **What it costs is two tenths of a microsecond a call** over NumPy's own: the lazy array, its cell and its node are still made. A sample of the process making small additions has the making and freeing of the three objects at about a third of the time, and NumPy's own call at about half. A call through the stand-in (`np.tanh(x)`, `np.dot(a, b)`) is made without NumPy's way round through its protocols, and for the functions that always give a new array without the rules being asked.
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

`uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --size large --install`. **NumPy** is the program as it is. **Installed** is the same eager program after `hotkernels.install()`: every call the package has a kernel for is served by it, one operation at a time, with no graph. **Lazy** is the program with the stand-in, every operation a node, computed one node at a time. **Fused** is the same with `fuse.enable()`: programs of kernels, small calls made at once. The results of all four are compared bit for bit before anything is printed. Times are the quickest of 5 runs, in milliseconds, from the start of the program to its last result, taken at load averages of 3.2 to 3.5, so they are rough to a few percent.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             39.26     37.79      37.53     0.42    37.11      33.31     7.76     0.71    24.84    1.13x    1.18x     1.13x
attention              30.26     29.10      30.94     0.59    30.35      30.01     1.06     0.45    28.50    1.03x    1.01x     0.97x
perceptron             33.09     25.05      68.27    26.55    41.72      40.43    40.01     0.07     0.35    1.69x    0.82x     0.62x
char_rnn               32.96     28.45     109.41    48.81    60.60      48.79    48.79     0.00     0.00    2.24x    0.68x     0.58x
particle_filter       367.58    331.03     375.78    88.17   287.61     376.83   104.46     7.30   265.07    1.00x    0.98x     0.88x
veros_eos             126.10    127.48     117.42     0.78   116.64      73.36     0.79     1.40    71.17    1.60x    1.72x     1.74x
veros_isoneutral      221.71    221.75     229.75     8.88   220.87     197.32     8.65     1.72   186.95    1.16x    1.12x     1.12x
veros_tke             100.25     97.49     104.12    23.51    80.61     100.66    27.22     1.20    72.24    1.03x    1.00x     0.97x
cavity_flow           156.26    155.03     173.40     8.72   164.68      68.92     9.04    15.48    44.40    2.52x    2.27x     2.25x
hdiff                  70.10     66.57      73.21     5.85    67.36      29.93     5.86     0.12    23.94    2.45x    2.34x     2.22x
black_scholes          87.22     87.18      88.26     1.97    86.29      85.36     1.92     0.13    83.31    1.03x    1.02x     1.02x
trimesh_mesh          175.12    137.77     176.41   176.31     0.11     175.06   175.06     0.00     0.00    1.01x    1.00x     0.79x
sklearn_metrics       982.24    272.53    1065.46  1062.35     3.11    1058.64  1058.64     0.00     0.00    1.01x    0.93x     0.26x
empyrical_stats      1261.58    325.69    1265.52  1265.52     0.00    1267.94  1267.94     0.00     0.00    1.00x    0.99x     0.26x
skimage_pipeline     1197.90   1122.41    1200.11  1200.11     0.00    1196.98  1196.98     0.00     0.00    1.00x    1.00x     0.94x
pandas_sales          481.34    446.68     480.21   480.21     0.00     479.03   479.03     0.00     0.00    1.00x    1.00x     0.93x
astropy_timeseries   1526.09   1336.60    1558.04  1555.68     2.36    1511.46  1510.17     0.29     1.01    1.03x    1.01x     0.88x
example                45.01      8.48      45.48    24.59    20.88      45.31    43.75     0.03     1.54    1.00x    0.99x     0.19x
flame                 140.77     88.24     225.68   106.47   119.21     145.63   143.17     0.06     2.40    1.55x    0.97x     0.61x
azimint_naive         172.40    155.25     192.54     8.99   183.55     178.83   178.83     0.00     0.00    1.08x    0.96x     0.87x
azimint_hist          122.62     71.45     122.85   122.81     0.03     122.73   122.73     0.00     0.00    1.00x    1.00x     0.58x
mandelbrot1           150.76    143.82     155.23     1.47   153.77     152.72   152.72     0.00     0.00    1.02x    0.99x     0.94x
mandelbrot2           162.85    147.08     343.99   206.87   137.13     172.82   172.82     0.00     0.00    1.99x    0.94x     0.85x
make_decision         145.54     88.83    1815.16  1188.30   626.86     280.38   280.38     0.00     0.00    6.47x    0.52x     0.32x
pandas_filter         175.83     34.54     178.61   178.61     0.00     174.87   174.87     0.00     0.00    1.02x    1.01x     0.20x
equalize_hist         141.90    126.53     142.85   142.76     0.09     142.26   142.26     0.00     0.00    1.00x    1.00x     0.89x
```

`build` is the time outside the layer's computing functions: the program's own Python, the nodes, and whatever is made at once. `compute` is the time inside them, less the planning.

The lazy and the fused run again, this time with the single-operation kernels installed as well, beside the same two eager runs. This is the comparison a user of the package would make: the last column is the fused run under `install()` against the eager program under `install()`.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression             39.26     37.79      36.57     0.38    36.19      31.77     7.88     0.70    23.19    1.15x    1.24x     1.19x
attention              30.26     29.10      28.92     0.53    28.39      29.21     1.00     0.44    27.77    0.99x    1.04x     1.00x
perceptron             33.09     25.05      60.87    26.37    34.50      33.15    32.75     0.08     0.32    1.84x    1.00x     0.76x
char_rnn               32.96     28.45     108.27    52.26    56.01      44.11    44.11     0.00     0.00    2.45x    0.75x     0.64x
particle_filter       367.58    331.03     359.82    93.42   266.40     336.85   103.97     7.33   225.55    1.07x    1.09x     0.98x
veros_eos             126.10    127.48     117.57     0.88   116.69      73.31     0.78     1.40    71.13    1.60x    1.72x     1.74x
veros_isoneutral      221.71    221.75     231.18     8.90   222.27     197.78     8.64     1.72   187.43    1.17x    1.12x     1.12x
veros_tke             100.25     97.49     102.35    23.63    78.72      98.74    27.27     1.20    70.27    1.04x    1.02x     0.99x
cavity_flow           156.26    155.03     174.07     8.73   165.34      69.60     9.13    15.50    44.97    2.50x    2.25x     2.23x
hdiff                  70.10     66.57      69.74     5.83    63.91      29.87     5.81     0.13    23.94    2.34x    2.35x     2.23x
black_scholes          87.22     87.18      88.86     1.91    86.95      85.50     1.90     0.12    83.48    1.04x    1.02x     1.02x
trimesh_mesh          175.12    137.77     137.78   137.67     0.11     137.67   137.67     0.00     0.00    1.00x    1.27x     1.00x
sklearn_metrics       982.24    272.53     285.15   283.81     1.34     278.03   278.03     0.00     0.00    1.03x    3.53x     0.98x
empyrical_stats      1261.58    325.69     327.12   327.12     0.00     326.38   326.38     0.00     0.00    1.00x    3.87x     1.00x
skimage_pipeline     1197.90   1122.41    1120.65  1120.65     0.00    1122.12  1122.12     0.00     0.00    1.00x    1.07x     1.00x
pandas_sales          481.34    446.68     445.89   445.89     0.00     445.80   445.80     0.00     0.00    1.00x    1.08x     1.00x
astropy_timeseries   1526.09   1336.60    1327.04  1325.19     1.84    1327.83  1326.54     0.28     1.00    1.00x    1.15x     1.01x
example                45.01      8.48       8.88     4.28     4.60       8.90     7.49     0.03     1.38    1.00x    5.06x     0.95x
flame                 140.77     88.24     171.93    90.88    81.05      93.25    90.76     0.05     2.44    1.84x    1.51x     0.95x
azimint_naive         172.40    155.25     175.62     9.12   166.49     161.79   161.79     0.00     0.00    1.09x    1.07x     0.96x
azimint_hist          122.62     71.45      71.60    71.56     0.03      71.47    71.47     0.00     0.00    1.00x    1.72x     1.00x
mandelbrot1           150.76    143.82     147.66     1.51   146.14     145.01   145.01     0.00     0.00    1.02x    1.04x     0.99x
mandelbrot2           162.85    147.08     343.89   206.72   137.18     173.06   173.06     0.00     0.00    1.99x    0.94x     0.85x
make_decision         145.54     88.83    1700.07  1157.97   542.10     198.64   198.64     0.00     0.00    8.56x    0.73x     0.45x
pandas_filter         175.83     34.54      34.71    34.71     0.00      34.29    34.29     0.00     0.00    1.01x    5.13x     1.01x
equalize_hist         141.90    126.53     127.08   126.99     0.09     126.95   126.95     0.00     0.00    1.00x    1.12x     1.00x
```

What the graphs were, and what the fused run did with them:

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
regression              384       1      384      3200000       378       6     41     41      41     41      0       0       0      179.2      89.6
attention               220       1      220       262144       198      19      8      8      16     16      0       0       0      174.7     135.4
perceptron             8854     565       33       128000        14    7900      2      2       4      2      0       0       0      163.7       2.7
char_rnn              32423     202    28606        10000         0   27923      0      0       0      0      0       0       0      165.8       0.0
particle_filter        2537      41      480       400000      2436      59    321    321     321    481     40     378      40     1166.5     605.7
veros_eos               366       1      366      1082016       363       3     37     37     140    285      0       0       0     3142.2     675.2
veros_isoneutral        718       1      690      4328064       582      19     97     97     116    186     37      47       0     3230.5    1745.3
veros_tke               382       6      281      3246048       286      26     39     39      47     46     18      33       1     1316.1    1063.9
cavity_flow            6920       1     6900       160801      6874       4    260    260     560   3000      0    1220       0     4129.6     304.9
hdiff                    40       1       40     10650240        29       0      4      4       4     15      0       1       0     1540.2     422.1
black_scholes            28       1       28      2000000        28       0      5      5       8     14      0       2       0      416.0     192.0
trimesh_mesh              2       1        2       245760         0       2      0      0       0      0      0       0       0        2.9       0.0
sklearn_metrics           7       4        2      1000000         0       7      0      0       0      0      0       0       0       11.4       0.0
empyrical_stats           0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
skimage_pipeline          0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
pandas_sales              0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
astropy_timeseries      235      52       11        62113        20     130      0      0       0      0      0       0       0       11.1       9.9
example                  13       2        9      2000000         3       9      0      0       0      0      0       1       0       68.0      18.0
flame                  7418     416       53       589824         6    6855      0      0       0      0      0       1       0      299.3       8.6
azimint_naive         10001       1    10001       200000         0    9001      0      0       0      0      0       0       0      601.6       0.0
azimint_hist              1       1        1         1000         0       1      0      0       0      0      0       0       0        0.0       0.0
mandelbrot1            1205       1     1205       220900         0     904      0      0       0      0      0       0       0     1126.2       0.0
mandelbrot2             727     321        9       902500         0     483      0      0       0      0      0       0       0      924.8       0.0
make_decision        491520       5   100000        81920         0  409600      0      0       0      0      0       0       0       83.9       0.0
pandas_filter             0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
equalize_hist             8       4        4      2890000         0       5      0      0       0      0      0       0       0        0.0       0.0
```

`deferred ops` to `largest node` are of the lazy run. `deferred fused` and `at once fused` are the operations the fused run still wrote as nodes, and the calls and writes it made as the program wrote them. `expr.` and `tiled` are the expressions found in the fused run's graphs and those a program was planned for; `fused nodes` the nodes computed inside a program; `given up` the programs not run, or dropped. `writes in place` and `copied` are the fused run's deferred writes. `MB written` is the results the calls wrote.

### Small sizes

The same at each program's small size, where arrays have hundreds to tens of thousands of elements:

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
regression              0.48      0.40       1.91     0.76     1.15       0.93     0.93     0.00     0.00    2.06x    0.52x     0.43x
attention               1.24      1.17       2.26     0.68     1.58       1.72     1.72     0.00     0.00    1.32x    0.72x     0.68x
perceptron              4.42      3.51       8.48     3.09     5.39       5.14     4.75     0.07     0.32    1.65x    0.86x     0.68x
char_rnn                6.67      6.01      22.84    12.04    10.80      11.19    11.19     0.00     0.00    2.04x    0.60x     0.54x
particle_filter         4.51      3.82       8.03     3.08     4.95       5.75     5.75     0.00     0.00    1.40x    0.78x     0.66x
veros_eos               7.40      6.96       7.93     0.43     7.50       7.16     0.47     1.43     5.25    1.11x    1.03x     0.97x
veros_isoneutral       16.10     16.19      18.21     1.96    16.25      17.74     1.79     1.71    14.24    1.03x    0.91x     0.91x
veros_tke               7.71      7.81       9.34     2.68     6.66      10.09     2.85     1.13     6.11    0.93x    0.76x     0.77x
cavity_flow             5.59      5.42      18.12     7.67    10.45       7.95     7.95     0.00     0.00    2.28x    0.70x     0.68x
hdiff                   1.88      1.89       2.04     0.33     1.72       1.07     0.28     0.13     0.66    1.91x    1.76x     1.77x
black_scholes           0.27      0.26       0.38     0.13     0.25       0.32     0.32     0.00     0.00    1.21x    0.87x     0.82x
trimesh_mesh           12.20      9.55      11.73    11.69     0.04      11.68    11.68     0.00     0.00    1.00x    1.04x     0.82x
sklearn_metrics        50.07     20.39      57.10    56.83     0.27      58.68    58.68     0.00     0.00    0.97x    0.85x     0.35x
empyrical_stats      1270.18    361.83    1268.40  1268.40     0.00    1298.71  1298.71     0.00     0.00    0.98x    0.98x     0.28x
skimage_pipeline      100.94     92.56     101.25   101.25     0.00     102.58   102.58     0.00     0.00    0.99x    0.98x     0.90x
pandas_sales           30.87     28.77      30.15    30.15     0.00      30.16    30.16     0.00     0.00    1.00x    1.02x     0.95x
astropy_timeseries    210.53    194.64     211.32   210.01     1.31     211.16   211.16     0.00     0.00    1.00x    1.00x     0.92x
example                49.11      8.70      46.72    25.53    21.19      45.48    43.89     0.03     1.56    1.03x    1.08x     0.19x
flame                  17.06      9.83      30.30    15.90    14.40      18.44    17.92     0.04     0.48    1.64x    0.93x     0.53x
azimint_naive          20.30     18.73      40.51     9.07    31.44      26.19    26.19     0.00     0.00    1.55x    0.78x     0.72x
azimint_hist           12.64      7.24      12.65    12.62     0.03      12.47    12.47     0.00     0.00    1.01x    1.01x     0.58x
mandelbrot1            15.71     15.11      18.74     1.42    17.32      16.39    16.39     0.00     0.00    1.14x    0.96x     0.92x
mandelbrot2            16.56     14.66      49.35    33.66    15.68      18.36    18.36     0.00     0.00    2.69x    0.90x     0.80x
make_decision          14.45      8.58     107.45    56.04    51.41      25.82    25.82     0.00     0.00    4.16x    0.56x     0.33x
pandas_filter          17.66      3.73      17.63    17.63     0.00      17.54    17.54     0.00     0.00    1.01x    1.01x     0.21x
equalize_hist          13.09     11.57      13.49    13.43     0.07      13.25    13.25     0.00     0.00    1.02x    0.99x     0.87x
```

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
regression              954       1      954         1600         0     904      0      0       0      0      0       0       0        0.4       0.0
attention               550       1      550         2048         0     520      0      0       0      0      0       0       0        3.0       0.0
perceptron              958      61       33       128000        14     844      2      2       4      2      0       0       0       23.6       2.7
char_rnn               5585     202     2497        10000         0    5135      0      0       0      0      0       0       0       17.9       0.0
particle_filter        1267      21      240         8000         0     940      0      0       0      0      0       0       0       11.7       0.0
veros_eos               366       1      366        72171       363       3     37     37     140    285      0       0       0      209.6      45.0
veros_isoneutral        718       1      690       288684       578      23     95     95     114    182     37      47       0      196.6     105.9
veros_tke               382       6      281       216513       270      37     37     37      41     39     18      32       0       82.5      66.3
cavity_flow            5900       1     5875         3721         0    3850      0      0       0      0      0       0       0       81.8       0.0
hdiff                    40       1       40       261360        29       0      4      4       4     15      0       1       0       36.8      10.1
black_scholes            28       1       28         4096         0      28      0      0       0      0      0       0       0        0.9       0.0
trimesh_mesh              2       1        2        15360         0       2      0      0       0      0      0       0       0        0.2       0.0
sklearn_metrics           7       4        2        50000         0       7      0      0       0      0      0       0       0        0.6       0.0
empyrical_stats           0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
skimage_pipeline          0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
pandas_sales              0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
astropy_timeseries      235      52       11        12451         0     150      0      0       0      0      0       0       0        2.2       0.0
example                  13       2        9      2000000         3       9      0      0       0      0      0       1       0       68.0      18.0
flame                  1870     108       53       147456         4    1726      0      0       0      0      0       0       0       31.3       2.0
azimint_naive         10001       1    10001        20000         0    9001      0      0       0      0      0       0       0       60.2       0.0
azimint_hist              1       1        1         1000         0       1      0      0       0      0      0       0       0        0.0       0.0
mandelbrot1            1205       1     1205        22500         0     904      0      0       0      0      0       0       0      114.4       0.0
mandelbrot2             727     321        9        90000         0     555      0      0       0      0      0       0       0       91.8       0.0
make_decision         49152       1    49152         8192         0   40960      0      0       0      0      0       0       0        8.4       0.0
pandas_filter             0       0        0            0         0       0      0      0       0      0      0       0       0        0.0       0.0
equalize_hist             8       4        4       262144         0       5      0      0       0      0      0       0       0        0.0       0.0
```

## What the measurements show

**Every program, by what the fused run is against plain NumPy** at the large size, and against the eager program under `install()` when the fused run has the single-operation kernels too:

| Program | Lazy against NumPy | Fused against NumPy | Fused and installed against installed | What it is made of |
|---|---|---|---|---|
| `hdiff` | 0.96x | 2.34x | 2.23x | Four expressions of a stencil, each one kernel written by hand |
| `cavity_flow` | 0.90x | 2.27x | 2.23x | Stencil expressions of 6 to 24 operations, each assigned into part of a grid |
| `veros_eos` | 1.07x | 1.72x | 1.74x | Polynomials, one of 153 operations |
| `regression` | 1.05x | 1.18x | 1.19x | A matrix product, a residual, a squared loss |
| `veros_isoneutral` | 0.97x | 1.12x | 1.12x | Expressions on slices of a grid; 37 of 97 are on rows too short |
| `black_scholes` | 0.99x | 1.02x | 1.02x | `erf`, `log`, `exp`; arithmetic between them |
| `trimesh_mesh`, `empyrical_stats`, `skimage_pipeline`, `pandas_sales`, `astropy_timeseries`, `azimint_hist`, `pandas_filter`, `equalize_hist`, `example` | 0.98x to 1.00x | 0.99x to 1.01x | 0.95x to 1.01x | A library's work, or single calls on large arrays |
| `mandelbrot1` | 0.97x | 0.99x | 0.99x | Complex arithmetic through a mask |
| `attention` | 0.98x | 1.01x | 1.00x | float32 matrix products and `tanh` |
| `veros_tke` | 0.96x | 1.00x | 0.99x | A tridiagonal solve, masks, small expressions |
| `azimint_naive` | 0.90x | 0.96x | 0.96x | A thousand masks over 200,000 elements |
| `flame` | 0.62x | 0.97x | 0.95x | 7,400 operations on arrays of 25,000 elements |
| `particle_filter` | 0.98x | 0.98x | 0.98x | Random draws, `arctan2`, `exp`; expressions on columns of a table |
| `mandelbrot2` | 0.47x | 0.94x | 0.85x | Complex arithmetic on arrays that shrink to nothing |
| `sklearn_metrics` | 0.92x | 0.93x | 0.98x | scikit-learn, handed lazy arrays |
| `perceptron` | 0.48x | 0.82x | 0.76x | 8,900 operations, most on arrays of 8,000 elements |
| `char_rnn` | 0.30x | 0.68x | 0.64x | 32,000 operations on arrays of 100 to 10,000 |
| `make_decision` | 0.08x | 0.52x | 0.45x | 490,000 operations on arrays of 32 |

- **One pass for a whole expression is worth what its passes were.** The cavity flow's fused run writes 0.3 GB of new arrays where the node-by-node run writes 4.1 GB, and computes in 44 ms what took 165 ms. Its 560 kernel calls are 260 programs; 3,000 of its 6,900 nodes have no array of their own. The equation of state writes 0.7 GB instead of 3.1 GB and computes in 71 ms instead of 117 ms: 153 operations in 64 kernel calls, all on blocks.
- **A write in place is the other half of the cavity flow.** Its 1,220 writes into slices and boundary rows are all made where the array is, in the lazy run and in the fused one. As copies they were 54 ms.
- **A kernel for every step is what covers a long tail.** Sixty-eight of the seventy-eight kernels ran over the corpus, the single operations among the most called: `add(S0,V0)` 67 times, `mul(S0,V0)` 40, `div(V0,S0)` 21. Each of those is a step inside a block, where otherwise it is a pass of NumPy's between two kernels.
- **The single-operation kernels and the fused ones do not meet, and do not get in each other's way.** Under `hotkernels.install()` the eager particle filter runs in 331 ms instead of 368 ms, and its fused run in 337 ms instead of 377 ms: the calls the package serves are the same calls in the graph. The nine programs that are a library's work or single calls gain eagerly from 1.07x to 5.3x, and their fused runs under `install()` are within 1% of that, the example at 5% behind. The five that fusion speeds up gain 5% or nothing from `install()` eagerly, because no kernel of the package is in their expressions.
- **Planning is 22% of the cavity flow's fused run,** 15 ms of 69 ms for 6,900 nodes: reading the expressions off the graph, in Python. For every other program it is about 2% or less.
- **A program of masks or of complex numbers writes no graph.** The naive azimuthal integration makes 9,000 calls at once, comparisons and selections on 200,000 elements, and is 4% behind NumPy where the lazy run is 10% behind. The first Mandelbrot program is 1% behind, the second 6%.
- **A program on small arrays pays for the lazy array, not for the graph.** `make_decision` makes 410,000 calls at once and writes no node: 280 ms against NumPy's 146 ms, where the lazy run takes 1,815 ms. The recurrent network makes 28,000 and takes 49 ms against 33 ms; the perceptron 7,900, 40 ms against 33 ms. Under `install()` the perceptron's fused run is level with plain NumPy and 0.76x of the eager program with the kernels, which are the same calls.
- **At the small sizes nothing but the horizontal diffusion gains,** 1.76x; the fused run is otherwise 0.5x to 1.0x of plain NumPy. It is 1.0x to 4.2x quicker than the lazy run all the same, the kinetic energy excepted, whose arrays of 72,000 to 290,000 elements are planned for in 1.1 ms of a run of 10.1 ms and mostly declined.
- **The particle filter gains nothing from fusion.** Its expressions are on columns of a table, which are gathered into blocks, and 40 of its writes into the table go to a copy because an earlier reader of the table still waits. Planning is 7 ms, and the passes saved pay for that and no more: 377 ms fused, 376 ms lazily, 368 ms on NumPy.
- **Rows too short to join are left to NumPy:** 37 of isoneutral mixing's 97 expressions and 18 of the kinetic energy's 39. They are slices that cut the innermost axis of a grid 26 deep.
- **Calls into a mathematics library are the wall for Black-Scholes and the transformer block.** `erf`, `log` and `exp` are 72% of the first, `tanh` and float32 arithmetic most of the second.
- **A library handed a lazy array may take longer over it.** scikit-learn's metrics are 8% slower on lazy arrays than on `ndarray`s, whatever the layer does with the seven operations of its own; under `install()` the difference is 2%.
- **How long a plan lives matters more than how good it is.** A reference kept past a computation costs a copy or a page fault where NumPy would have reused memory: a cycle of references in a planner costs 46,000 page faults a run, and a record of the part written into, kept by a group that has been given up, a copy of the array for each of 21 writes. Both are tested for.


## What is not done

Each of these is a decision or a piece of work, in the order of what the measurements say it is worth.

- **What a result is, on small arrays.** A call made at once still makes a lazy array, its cell and its node: two tenths of a microsecond over a call NumPy makes in three. The programs on small arrays are 1.2 to 1.8 times slower than plain NumPy for that and for nothing else. A result that has its values needs none of the three until something defers an operation on it; holding the `ndarray` in the lazy array itself, and making the node when one is first asked for, is the next thing to try.
- **Reading expressions off a graph, in C.** Covering an expression with shapes is remembered by the expression's shape; reading the expression is not, and is Python: two microseconds a node, a fifth of the cavity flow's fused run. The reading could be remembered by the line that wrote the root, checked against the nodes as it is replayed, or done by the C core as the nodes are written.
- **Operations a kernel does not make, inside a program.** A comparison and `np.where` give or read a mask; `exp`, `tanh`, `erf`, `x ** 3` call the platform's mathematics library; float32 and complex128 have no kernels. Each ends a program and is a pass over memory. The module could call NumPy's own loop for such an operation on a block's buffers, between two kernel steps: NumPy's bits by construction, and no pass over memory. That would reach Black-Scholes, the transformer block, the two Mandelbrot programs and the masks of the Veros routines.
- **Rows too short to join.** `x[2:-2, 2:-2, :-1]` of a grid 26 deep is rows of 25 elements that cannot be walked as one run, and a program on them is declined, as are most of the expressions of isoneutral mixing that are given up. Gathering such rows into blocks was tried and lost to NumPy's own passes by 5%, because each row is a call to `memcpy` for each array. Copying the rows of a block with one loop written for it, or letting a kernel take a row length and a row step, are the two ways left.
- **Tuning the written kernels.** They are as the tool wrote them, with plain loads. The kernels written by hand gained up to 1.2x from non-temporal loads on arrays read from memory; whether a kernel should load differently for an operand that is a buffer in the cache and one that is an array is not measured.
- **Reductions along an axis, and reductions other than a sum.** `np.sum(e ** 2, axis=-1)` and `np.max(np.abs(e))` occur in the programs and are left to NumPy.
- **A sum at the end of a program.** `sum(sq(V0))` is a kernel of its own. A sum of a long expression computes the expression into an array first, since NumPy's order of addition is over the whole array and not over blocks.
- **The limit below which a call is made at once.** 32,768 elements is the count at which the fractal flame, the one program with arrays near it, runs quickest. It is one program's evidence.
- **What handing a lazy array to a library costs.** Scoring a classifier with scikit-learn is 8% slower with lazy arrays handed in than with `ndarray`s, with seven operations deferred in all. The time is not in the layer: the library takes a longer road for an array type it does not know. And scikit-image fails on one. A program whose work is a library's gains nothing from the layer; whether it should lose anything is open.
- **x86-64.** The kernels are ARM64 only; on another architecture each is built from its C reference, which the judge holds to the same bits. `make_kernel.py` writes NEON; the same tree walk would write AVX2. Nothing was timed there.
- **More programs.** [Issue 298](https://github.com/sofeikov/hotkernels/issues/298) lists 21 programs that come out as one graph each, most of them stencils and loops over rows.
- **More of NumPy in the deferred set.** The nine workloads make calls the layer does not defer: `np.histogram`, `np.bincount`, `np.interp`, `np.isin`, `np.unique`, `np.stack`, `x.astype(...)`, `np.clip` without `out=`. None of them sits inside an expression the kernels could take, but for `np.clip`, which is a maximum and a minimum, and a cast, which is a kernel nobody wrote.

## What changed outside `fuse.py`

- **`evaluate.py` asks a planner,** and has a meter and a clock for the tools. With none set it computes node by node. It decides, for each write it computes, whether the array's own memory may be written (`reusable`), and it catches warnings and puts the error settings once for a computation. [lazy-design.md](lazy-design.md#computing) says what a planner may and may not change, and [its section on writes](lazy-design.md#writes) when a write copies.
- **`array.py` and the C core** have `at_once_below`, the roads for calls made at once, `x ** e` as `operator.pow`, and `x.shape = s`. Without the switch set, a graph is written node for node as before, which the layer's tests of what is written check.
- **The judge looks for a kernel in a second place,** `prototypes/lazy/kernels/`, and locks its spec and reference to `main` there as it does in `kernels/`. Nothing else in the judge changed.
- **`programs/corpus.py`** has every program the repository times, and `fusion_time.py` times each four ways.
- **Nothing was added to the deferred set.** `x.shape = s` is a view, which was already deferred; the calls the new programs make at once are listed under [What is not done](#what-is-not-done).
