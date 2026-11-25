# Fusion by pattern in the lazy layer

This is a first build, in the prototype of the lazy layer ([lazy-design.md](lazy-design.md), [`prototypes/lazy/`](../prototypes/lazy)), and what it measured. Nothing here is in the package. The layer writes a program's NumPy operations down as a graph; this document is about computing parts of that graph by one kernel call each instead of one NumPy call per operation.

The approach is the plainest one there is. A **pattern** is a small tree of operations with named inputs, such as `a + s * b`. Where the graph holds that tree, a kernel written for it computes it in one pass over memory. An expression no pattern fits is computed as before, so nothing is lost by a pattern that is missing.

## Summary

- **A fused kernel gives NumPy's bits, and the judge checks it.** Every operation in one is a single IEEE operation that NumPy makes with the same instruction, in the program's order, each rounded by itself. No multiply-add is fused and no sum is regrouped. Fourteen kernels were written, in NEON assembly for ARM64, and each passes the judge against its NumPy expression on 20,000 to 280,000 cases. See [Why the bits are NumPy's](#why-the-bits-are-numpys).
- **The pattern is the kernel's own oracle.** A kernel's `spec.py` names the NumPy expression it must equal. Run on lazy arrays, that expression writes a graph, and the shape of that graph is the pattern. Nothing else says what a kernel serves. See [The library](#the-library).
- **Which patterns to write was mined, not guessed.** Twelve programs, three of them published benchmarks copied unchanged, were run lazily with every call timed. Their expressions were read off the graphs and ranked by the time NumPy spends in them. See [Finding the shapes](#finding-the-shapes).
- **Where a program is an expression, fusion is worth 1.2x to 2.5x.** With fourteen kernels, the horizontal diffusion of a weather model runs 2.5x faster than the lazy layer without fusion and 2.3x faster than plain NumPy; an ocean model's equation of state 1.23x and 1.22x; a cavity flow 1.39x faster than the lazy layer and level with NumPy; a regression 1.11x faster than the lazy layer and level with NumPy. The other seven of the eleven programs timed are within 4% of the lazy run either way. See [What was measured](#what-was-measured).
- **A kernel alone is 1.4x to 2x faster than NumPy's passes from a million elements up,** for a shape of two operations, and 1.7x to 3x for the stencils of four to eight; all are further ahead on arrays that fit a cache. Hand-written assembly is level with the compiled C except where it loads with a non-temporal hint, which the compiler does not emit: up to 1.2x at 262,144 to 1,048,576 elements.
- **Three things limit it, and each is measured.** The long tail of shapes: fourteen kernels remove well under half of the passes that could be removed. Calls into a mathematics library (`exp`, `tanh`, `erf`), which a kernel cannot make with NumPy's bits and which are most of the time in Black-Scholes and in the transformer block. And the layer's own costs, which fusion does not touch: a deferred write still copies the array, and reading expressions off a graph costs about two microseconds a node. See [What the measurements show](#what-the-measurements-show).

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
| An array of any element type but float64, a NumPy scalar of another type | The first build has kernels for float64. The transformer block of the corpus is float32 but for its attention scores, which a division by a float64 scalar makes float64, and only those are fused |
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
- **Only over every element.** A reduction along an axis has another order again (down the columns it is not pairwise at all), and is not in the first build.

### Writes

The lazy layer makes a write into an array it owns a node of the graph: a copy of what the array held, with the write made into the copy ([lazy-design.md](lazy-design.md#writes)). Read that way, a write costs a pass more than NumPy's write in place. With fusion switched on, three kinds of write cost less, and all three give the bits of the copy and the write.

| The program writes | As written, the node computes | With fusion |
|---|---|---|
| `w -= s * g`, on a whole array | A copy of `w`, then `-=` into it | The operator is an operation of the expression: `sub(V0,mul(S0,V1))`, one kernel call into a new array. No copy. Where no kernel fits, NumPy's one call, `np.subtract(w, t)` |
| `u[1:-1, 1:-1] = e` | `e` into a temporary; a copy of `u`; the temporary assigned into the copy | A copy of `u`, and `e` computed straight into that part of the copy: by a kernel if one fits the top of `e`, and otherwise by NumPy's call with the part as its `out=`. No temporary. Where the part is all of `u` (`out[:, :, :] = e`), no copy either |
| `u[1:-1] += s * v` | A copy of `u`, then `+=` into the part | A copy of `u`, and one kernel call that reads the part of the old array and writes the part of the copy |

A copy and `-=` into it has the bits of the subtraction because everything is float64: the same loop subtracts the same operands, and nothing is cast on the way into the array. The copy a write makes is laid out in C order whatever the old array was, so the one call is asked for C order too (`order="C"`).

**What is still copied.** A write into part of an array copies the whole array first, as the layer defines it. NumPy, which writes in place, copies nothing. Writing into the old contents when nothing else can read them is not done here, and is the largest cost left in a program that updates its arrays in place: see [What the measurements show](#what-the-measurements-show).

## Finding the shapes

### The corpus

Twelve programs, in [`prototypes/lazy/programs/corpus.py`](../prototypes/lazy/programs/corpus.py). Each runs with `import numpy as np` or with the stand-in for it, and nothing else of it changes; all twelve give NumPy's results bit for bit lazily, with fusion and without.

| Program | What it is | From |
|---|---|---|
| `regression` | Linear regression by gradient descent | This repository, `prototypes/lazy/programs/` |
| `monitor` | The same, gathering its loss every fifth step | The same |
| `attention` | A transformer block's forward pass, float32 | The same |
| `perceptron` | A two-layer perceptron trained by mini-batches | `delivery/apps/mlp_train.py` |
| `char_rnn` | Karpathy's `min-char-rnn.py` | `delivery/apps/vendor/min_char_rnn/`, copied unchanged |
| `particle_filter` | A bootstrap particle filter | `delivery/apps/particle_filter.py` |
| `veros_eos`, `veros_isoneutral`, `veros_tke` | Three routines of the Veros ocean model, as pyhpc-benchmarks publishes them | `delivery/apps/vendor/pyhpc/`, copied unchanged |
| `cavity_flow` | Lid-driven cavity flow, step 11 of *CFD Python: the 12 steps to Navier-Stokes*, as npbench publishes it | [`prototypes/lazy/programs/vendor/`](../prototypes/lazy/programs/vendor/README.md), copied unchanged |
| `hdiff` | Horizontal diffusion, a stencil of the COSMO weather model, from npbench | The same |
| `black_scholes` | Black-Scholes option pricing, from Intel's dpbench | The same |

The last three were added for this work, as programs whose authors did not have the lazy layer in mind. They were chosen to differ: a stencil with nothing but arithmetic, a stencil that writes into its arrays at every step, and a formula that is mostly calls into a mathematics library. A fourth candidate, an N-body simulation, is published under the GPL and was left out.

### Mining

`uv run --group apps python -m prototypes.lazy.programs.mine` runs each program lazily, one node at a time, as the first build computes it. Before every computation the expressions among its nodes are read off, and every call is timed. So each expression has a measured weight: the time NumPy spent on the operations in it.

- **It is the graph that is mined, not the source.** The graph is the program after Python has run: loops unrolled, numbers worked out, views resolved. `dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]` is one multiplication of an array by a number.
- **The weight is time, not count.** A shape that occurs a thousand times on arrays of a hundred elements weighs less than one that occurs once on ten million.
- **Mining says which shapes occur and what they cost. It does not say that a shape may be fused.** That is the rule of [What is fused](#what-is-fused), and the judge's check of each kernel.

Where NumPy's time goes, at the large size of each program, as shares of the time spent computing nodes:

```text
program            deferred  graphs  largest        ms in expressio simple, alon other elemen       matmul    reduction        write     indexing     the rest
regression              384      22      261      54.3        37.5%         0.6%         0.0%        42.4%        19.3%         0.0%         0.0%         0.2%
monitor                 165       6       32      37.3        14.2%         5.5%         0.0%        59.7%        20.2%         0.0%         0.0%         0.4%
attention               220       1      220      29.3         5.2%        11.9%        57.6%        20.2%         4.3%         0.0%         0.0%         0.8%
perceptron             8854     565       33      66.1        34.8%         5.7%         5.9%        17.6%        10.0%        19.9%         3.1%         3.0%
char_rnn              32423     207    25387     140.7        38.8%         8.1%        10.1%        24.8%         2.9%         4.0%         0.0%        11.4%
particle_filter        2537      43      280     337.5        42.4%        10.7%        19.6%         4.6%         1.4%         2.3%         5.0%        14.2%
veros_eos               366       1      366     187.6        85.4%        10.5%         4.2%         0.0%         0.0%         0.0%         0.0%         0.0%
veros_isoneutral        718       7      255     262.2        79.3%         0.0%        19.8%         0.0%         0.0%         0.2%         0.0%         0.5%
veros_tke               382       7      272      98.0        82.5%         2.8%         3.9%         0.0%         0.0%        10.0%         0.0%         0.8%
cavity_flow            6920       2     6856     220.2        83.0%         0.0%         2.7%         0.0%         0.0%        10.8%         0.0%         3.5%
hdiff                    40       1       40     119.7        91.3%         8.5%         0.0%         0.0%         0.0%         0.0%         0.0%         0.2%
black_scholes            28       2       25      97.8        17.3%         6.3%        76.4%         0.0%         0.0%         0.0%         0.0%         0.0%
```

- **A program made of arithmetic is nearly all expression:** 91% of the horizontal diffusion's time, and 79% to 85% of the cavity flow's and the three Veros routines'.
- **A program made of calls is not.** 76% of Black-Scholes is `erf`, `log` and `exp`; 58% of the transformer block is `tanh`, `exp` and float32 arithmetic; the regressions spend 42% and 60% of their time in the matrix product.
- **The heaviest operations outside every expression** over the corpus are `tanh` (58 ms), `erf` (57 ms), `log` (27 ms), `arctan2` (27 ms) and `exp` (25 ms), beside 1,006 ms in expressions.

### The shapes

An expression of the corpus has between two and 153 operations, and no kernel is written for a tree of 153. So the miner cuts every expression into the smaller trees it contains, up to six operations, and counts each by the time of the expressions it occurs in. A library is then chosen one shape at a time, each the shape that removes the most passes together with those already chosen, an expression being covered by as few kernel calls as the library allows.

The first twenty-four, with the share of the removable passes that the library removes once each is in it:

```text
  1    6.0%  6 operations  sub(div(add(V0,mul(S0,add(V1,V2))),S1),mul(S2,V3)) cavity_flow
  2   11.5%  2 operations  add(V0,mul(V1,V2))                           particle_filter, veros_eos, veros_isoneutral, veros_tke
  3   15.9%  2 operations  add(V0,mul(S0,V1))                           attention, cavity_flow, particle_filter, perceptron, regression, veros_eos, veros_isoneutral, veros_tke
  4   19.7%  3 operations  add(V0,add(V1,add(V2,V3)))                   char_rnn, hdiff, veros_eos, veros_isoneutral
  5   23.3%  2 operations  mul(V0,mul(V1,V2))                           veros_eos, veros_isoneutral, veros_tke
  6   26.6%  4 operations  where(gt(mul(V0,sub(V1,V2)),S0),S1,V3)       hdiff
  7   29.3%  6 operations  div(neg(add(V0,mul(V1,V2))),sub(min(S0,V3),S1)) veros_isoneutral
  8   31.4%  2 operations  mul(V0,sub(V1,V2))                           black_scholes, cavity_flow, hdiff, veros_eos, veros_isoneutral, veros_tke
  9   33.2%  6 operations  add(V0,mul(V1,mul(mul(V2,mul(S0,V3)),sq(V4)))) veros_isoneutral
 10   34.8%  2 operations  mul(S0,add(V0,V1))                           attention, cavity_flow, particle_filter, regression, veros_isoneutral
 11   36.4%  5 operations  sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))   hdiff
 12   38.0%  2 operations  mul(V0,add(S0,V1))                           attention, black_scholes, veros_eos
 13   39.5%  3 operations  add(sq(V0),sq(V1))                           particle_filter
 14   41.0%  5 operations  sub(V0,mul(V1,sub(add(V2,sub(V3,V4)),V5)))   hdiff
 15   42.3%  2 operations  sub(V0,mul(S0,V1))                           cavity_flow, monitor, perceptron, regression, veros_tke
 16   43.6%  5 operations  sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))   cavity_flow
 17   44.9%  5 operations  add(V0,mul(max(S0,mul(V1,V2)),mul(V3,V4)))   veros_isoneutral
 18   46.0%  2 operations  mul(V0,mul(S0,V1))                           attention, veros_eos, veros_isoneutral, veros_tke
 19   47.0%  4 operations  div(add(S0,neg(abs(V0))),S1)                 veros_isoneutral
 20   48.0%  2 operations  div(sub(V0,V1),S0)                           cavity_flow, veros_tke
 21   48.8%  4 operations  mul(add(V0,V1),sub(S0,sq(V2)))               char_rnn
 22   49.6%  5 operations  add(add(S0,V0),mul(V1,add(S1,mul(S2,V2))))   veros_eos
 23   50.4%  2 operations  sum(sq(V0))                                  particle_filter, perceptron, regression
 24   51.0%  4 operations  sub(sub(V0,mul(V1,V2)),mul(V3,V4))           cavity_flow, veros_eos
```

- **The tail is long.** Ten shapes remove 35% of the passes that fusing every expression whole would remove, twenty-four remove 51% and forty 58%; the fortieth still adds a third of a percent.
- **Two kinds of shape are worth a kernel.** Small ones that many programs use: `a + b * c` and `a + s * b` are in four and eight of the twelve. And whole expressions of one hot loop: the five-point Laplacian of the diffusion is a single shape that one program spends a third of its time in.
- **The same shape of two operations is many patterns.** `a + s * b`, `a - s * b`, `s * (a + b)`, `a * (s + b)`: each is its own kernel. For sums, differences and products of arrays and numbers there are 56 such shapes of two operations, and more with division.

### The fourteen kernels

Chosen from that list: the nine shapes of two operations that weigh most across programs, the one reduction that three programs make, and four whole expressions of the two stencil programs.

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

They are in [`prototypes/lazy/kernels/`](../prototypes/lazy/kernels/README.md), one directory each with a spec, a C reference, the assembly and its log, as the kernels of `kernels/` have. The judge finds a kernel by name in either place ([judge-design.md](judge-design.md#layout)); the package builds from `kernels/` alone and does not contain these.

## The library

**A kernel's pattern is read off its oracle.** A spec's oracle is the NumPy expression the judge holds the kernel to:

```python
def oracle(a, b, s):
    return a + s * b
```

`fuse.py` calls it with lazy arrays and two numbers. The oracle writes a graph without computing anything, the expression is read off that graph as off any other, and its name, `add(V0,mul(S0,V1))`, is the pattern. Which leaf is which argument is known because the arrays passed in are the leaves read.

- **The specification is its own pattern.** What a kernel is matched against and what the judge checks it against are one function, so they cannot disagree. A kernel whose oracle is not one expression of the operations above, or does not read each argument once, has no pattern and is refused when the library is loaded.
- **A name is the tree as written,** with the operands of a sum, a product, a maximum or a minimum in a fixed order: `sub(V0,mul(S0,V1))`. Two expressions of one shape have one name whichever way round the program wrote them.
- **Arguments come in one order:** the input arrays, the output array, the length, then the numbers. [`_fused.c`](../prototypes/lazy/_fused.c) calls every kernel with eight integer arguments and eight floating-point ones, of which the kernel reads those it has; both ARM64 and x86-64 pass that many in registers or in order on the stack, so one call reaches any kernel.

## Planning

Before each computation `evaluate` asks its planner which nodes are computed together ([lazy-design.md](lazy-design.md#computing)). `fuse.plan` answers in four steps.

1. **Writes into whole arrays become their one call** ([Writes](#writes)), whatever else is planned.
2. **Expressions are read off the nodes,** the newest first. An operation is the top of an expression unless an expression already found contains it; below it, each operand is inside the expression if it is an operation a kernel can make, of the same shape, about to be computed, read by this operation alone, and held by nothing else.
3. **Each expression is covered with shapes of the library by the fewest passes.** At each operation, every shape with that operation at its top is held against the tree. A shape's vector leaf fits an array the expression reads, or any operation below, whose value is then computed first by its own tile or its own NumPy call. The fewest passes are found from the leaves up, which is exact for a tree. An expression of 24 operations in the cavity flow is computed in 16 passes, eight of them kernel calls; its pressure step of eight operations in one.
4. **Each tile is a group:** the nodes it covers, the kernel, and what it reads. The plan is the groups, by the node each gives the value of.

**What "held by nothing else" means.** An intermediate the program has a name for, or that a node outside this computation reads, will be asked for again. Both are a reference to the node, and CPython counts references exactly: a node of a computation is referred to by the nodes of it that read it, by the one list it is in, and by nothing else unless one of the two holds. The count decides only what is worth fusing. A node that a plan skips stays pending and is computed when asked for, so a wrong count costs time and never a result. Two things this found while it was built are worth knowing by anyone who touches the code: a `for` loop's variable still refers to the last node after the loop, and pytest's rewriting of `assert` keeps every value of the expression in it alive. Both make a node look held.

**Nothing planned outlives the computation.** A plan refers to nodes and so to arrays. If anything kept a plan alive past its computation, those arrays would stay in memory until Python's collector of cycles next ran, and every result in between would be written to fresh memory where NumPy's own calls reuse what was just freed. This happened in the building: two functions that called themselves by a name of the function they were made in formed a cycle, one run of an ocean routine took 46,000 page faults where the run without fusion takes one, and was a quarter slower for it. A test now runs a fused computation with the collector off and holds that no node is left.

**Arrays of fewer than 16,384 elements are left alone.** Reading an expression off a graph costs about two microseconds a node, and a plan about ten more. A pass of NumPy's over fewer elements than that costs no more, so there is nothing to win back. A computation none of whose nodes is that large is not read at all.

## Computing a group

A group is computed when `evaluate` reaches the node it gives the value of. [`_fused.c`](../prototypes/lazy/_fused.c) stands between the kernel, which is a leaf function over contiguous runs, and arrays of any shape.

- **It walks the outer axes and calls the kernel once per contiguous run.** Every outer axis that every operand walks straight through joins the run, so a C-contiguous expression of any number of dimensions is one call. `u[1:-1, 1:-1]` of a grid is one call per row. In the horizontal diffusion, whose arrays are sliced along two of three axes, a run is a whole slab of 41,000 elements.
- **It reads float64 arrays that step by one element along the innermost axis,** with strides that are positive and do not grow from one axis to the next. NumPy's result for such operands is laid out in C order, which the output is.
- **It declines anything else:** a column of a grid (`particles[:, 0]`), a reversed or Fortran-ordered array, a transposed one. The group's nodes are then computed by their own calls, in order. Declining is decided before anything is computed and costs nothing.
- **The output overlaps no input.** It is a new array, or part of a new copy.

### Floating-point exceptions

NumPy reports overflow, division by zero and invalid operations as warnings, or as errors under `np.errstate(over="raise")`, with the name of the operation and on the program's line. A kernel of several operations cannot say which of them raised a flag.

So the module clears the processor's flags before a kernel runs and reads them after. If a flag is raised that NumPy would report under the `np.errstate` the operations were written under, **the kernel's result is dropped and NumPy's own calls are made**, one node at a time. The warning, the error, its words and its line are then NumPy's by construction, and so are the values.

- **An exception the program silenced costs nothing.** Under `np.errstate(all="ignore")`, and for underflow always, the kernel's result is kept.
- **An expression that raises one is computed twice.** The line that wrote it is remembered, and its expressions are left to NumPy from then on, so a loop pays once.
- **A comparison raises a flag NumPy does not report.** The processor's comparison marks a NaN operand as an invalid operation; NumPy's comparison clears the flag. The fused flux limiter is therefore computed by NumPy's calls when its data has a NaN, unless the program has silenced invalid operations.

### A write made at once

A write into memory the program holds makes the layer compute everything pending first, after which nothing may be pending ([lazy-design.md](lazy-design.md#writes)). The nodes inside a group have no value of their own. Nothing should hold one, or the plan would not have put it inside; `evaluate` checks all the same, and computes any that is still alive after the planned computation, since the state after it must not depend on a count of references.

## Why the bits are NumPy's

The argument is per operation, and the judge checks it per kernel.

| Hazard | What a kernel does | How it is checked |
|---|---|---|
| A product and a sum fused into one multiply-add, which rounds once | `fmul` then `fadd`, never `fmla`. The C reference starts with `#pragma STDC FP_CONTRACT OFF`; without it clang fuses the two at the judge's flags | The judge compares bits with NumPy's `a + s * b` on every size to 65, on random values and on its special ones. Compiled without the pragma, the reference's loop is `fmla` |
| A division made as a product with a reciprocal | `fdiv` by the number | The same, with divisors that are zero, tiny and infinite |
| Additions regrouped | The program's brackets are the pattern's: `add(V0,add(V1,V2))` adds the last two first. A reduction makes NumPy's pairwise additions | The judge, and a test at every length to 700 |
| Signed zeros, NaN, infinities, denormals | Whatever the instruction gives, which is what NumPy's loop gives | The judge's value classes |
| A result laid out differently, so that a later sum adds in another order | Operands are taken only where NumPy's result is in C order, and the output is | A test of layouts the kernels do not read |
| An output that overlaps an input | It never does | By construction |
| A warning or an error | The flags are read and NumPy's calls are made | Tests of each, with the line |

**One thing is not specified: which NaN comes of two.** When an operation meets two NaNs of different sign or payload, the processor returns its first operand's. Which operand is first in NumPy's loop is its compiler's choice, and a pattern does not record how the program ordered a sum. Whether a result is NaN is always NumPy's; which NaN it is, is NumPy's whenever a program has one kind of NaN, which is what arithmetic and `np.nan` give.

## What was measured

Everything below is from an Apple M1 Pro, NumPy 2.5.5, CPython 3.13, with the layer's nodes written by its C core. The kernels' specs are not on `main` yet, so the judge's runs are `--unlocked`.

### The kernels alone

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

### The programs, three ways

`uv run --group apps python -m prototypes.lazy.programs.fusion_time --size large`. **NumPy** is the program as it is. **Lazy** is the program with the stand-in, computed one node at a time. **Fused** is the same with `fuse.enable()`. The results of the three are compared bit for bit before anything is printed. A lazy run's time is split: `compute` is the time inside the layer's computing functions, less the planning, which is `plan`; `build` is everything else, the program's own Python, the writing of the nodes, and what NumPy computes at once.

```text
program            NumPy ms    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy
regression            38.55      44.11     0.44    43.67      39.84     0.46     0.74    38.63    1.11x   0.97x
attention             28.93      29.42     0.51    28.91      29.58     0.52     0.52    28.54    0.99x   0.98x
perceptron            33.27      96.97    28.80    68.17     100.13    28.95     3.77    67.41    0.97x   0.33x
char_rnn              32.48     194.69    55.39   139.30     203.22    59.95     8.68   134.58    0.96x   0.16x
particle_filter      359.28     408.76    86.48   322.28     396.02    86.44     7.52   302.05    1.03x   0.91x
veros_eos            126.50     127.90     0.82   127.08     104.03     0.81     2.40   100.82    1.23x   1.22x
veros_isoneutral     219.38     261.14     9.01   252.13     257.63     9.10     2.83   245.70    1.01x   0.85x
veros_tke             99.57     114.88    23.55    91.33     113.87    23.57     1.54    88.77    1.01x   0.87x
cavity_flow          157.77     224.67     9.63   215.04     161.25     9.74    24.81   126.70    1.39x   0.98x
hdiff                 69.74      75.13     5.81    69.32      29.98     5.81     0.19    23.97    2.51x   2.33x
black_scholes         87.01      89.43     1.91    87.51      86.82     1.92     0.21    84.70    1.03x   1.00x
```

The sizes:

```text
regression        200,000 examples of 16 features, 20 steps
attention         256 tokens of width 256, 4 heads, 4 blocks
perceptron        12,000 examples, batches of 128, 2 epochs
char_rnn          30 iterations of 25 characters, 100 hidden units
particle_filter   100,000 particles, 40 steps
veros_eos         a grid of 204 × 204 × 26 points
veros_isoneutral  a grid of 204 × 204 × 26 points
veros_tke         a grid of 204 × 204 × 26 points
cavity_flow       a grid of 401 × 401, 20 steps of 10 pressure iterations
hdiff             a grid of 256 × 256 × 160
black_scholes     2,000,000 options
```

What the graphs were, and what fusion did with them. `expr.` are the expressions found, `tiled` those a kernel was planned for, `fused nodes` the nodes computed inside a kernel call instead of by a call of their own, `given up` the kernel calls not made or dropped, `whole` and `part writes` the writes computed by one NumPy call instead of a copy and a call, and `MB written` the results the calls wrote:

```text
program           deferred  graphs  largest largest node   expr.  tiled  kernel  fused  given  whole   part    lazy MB  fused MB
                       ops            graph    elements                  calls  nodes     up writes writes    written   written
regression             384      22      261      3200000      41     20      20     20      0      0      0      179.2     147.2
attention              220       1      220       262144       8      8       8      8      0      0      0      174.7     157.9
perceptron            8854     565       33       128000       2      0       0      0      0   1504      0      203.8     203.8
char_rnn             32423     207    25387        10000       0      0       0      0      0   4050      0      312.8     312.8
particle_filter       2537      43      280       400000     321    161     161    161      0    200      0     1693.2    1434.0
veros_eos              366       1      366      1082016      37     26      94     94      0      0      0     3142.2    2328.5
veros_isoneutral       718       7      255      4328064      97     61      55     57     24      0      3     4052.8    3566.4
veros_tke              382       7      272      3246048      35     15      13     16     12      0      7     1747.8    1574.8
cavity_flow           6920       2     6856       160801     260    260     580   1980      0      0      0     5699.0    3177.2
hdiff                   40       1       40     10650240       4      4       4     15      0      0      0     1624.1     505.9
black_scholes           28       2       25      2000000       5      2       3      3      0      0      2      448.0     368.0
```

### Small arrays

The same with `--size small`. Where every array is below the floor of 16,384 elements nothing is planned, and the fused run is the lazy run with each graph looked at once.

```text
program            NumPy ms    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy
regression             0.46       4.29     0.96     3.33       4.38     0.94     0.09     3.34    0.98x   0.11x
attention              1.23       3.73     0.72     3.01       3.81     0.71     0.06     3.04    0.98x   0.32x
perceptron             4.26      11.62     3.39     8.23      12.09     3.42     0.51     8.15    0.96x   0.35x
char_rnn               6.52      37.07    13.25    23.82      38.28    13.55     1.27    23.46    0.97x   0.17x
particle_filter        4.63      11.10     3.19     7.91      11.99     3.22     0.94     7.83    0.93x   0.39x
veros_eos              6.56       8.92     0.43     8.49       9.85     0.44     2.41     7.00    0.91x   0.67x
veros_isoneutral      15.14      20.61     2.05    18.56      23.52     1.97     2.71    18.84    0.88x   0.64x
veros_tke              7.29      10.09     2.44     7.65      11.36     2.49     1.36     7.51    0.89x   0.64x
cavity_flow            5.30      32.29     7.93    24.37      33.16     7.90     0.78    24.49    0.97x   0.16x
hdiff                  1.82       2.07     0.27     1.80       1.11     0.27     0.19     0.66    1.86x   1.63x
black_scholes          0.22       0.40     0.10     0.30       0.40     0.10     0.01     0.30    1.00x   0.54x
```

```text
program           deferred  graphs  largest largest node   expr.  tiled  kernel  fused  given  whole   part    lazy MB  fused MB
                       ops            graph    elements                  calls  nodes     up writes writes    written   written
regression             954      52      651         1600       0      0       0      0      0      0      0        0.4       0.4
attention              550       1      550         2048       0      0       0      0      0      0      0        3.0       3.0
perceptron             958      61       33       128000       2      0       0      0      0    160      0       27.9      27.9
char_rnn              5585     207     2059        10000       0      0       0      0      0    405      0       32.7      32.7
particle_filter       1267      23      140         8000       0      0       0      0      0    100      0       17.1      17.1
veros_eos              366       1      366        72171      37     26      94     94      0      0      0      209.6     155.3
veros_isoneutral       718       7      255       288684      97     61      55     57     24      0      3      251.4     222.0
veros_tke              382       7      272       216513      34     15      13     16     12      0      7      111.2     100.5
cavity_flow           5900       2     5831         3721       0      0       0      0      0      0      0      108.6     108.6
hdiff                   40       1       40       261360       4      4       4     15      0      0      0       38.7      12.0
black_scholes           28       2       25         4096       0      0       0      0      0      0      0        0.9       0.9
```

The small sizes:

```text
regression        200 examples of 8 features, 50 steps
attention         16 tokens of width 32, 4 heads, 10 blocks
perceptron        1,280 examples, batches of 128, 2 epochs
char_rnn          3 iterations of 25 characters, 100 hidden units
particle_filter   2,000 particles, 20 steps
veros_eos         a grid of 81 × 81 × 11 points
veros_isoneutral  a grid of 81 × 81 × 11 points
veros_tke         a grid of 81 × 81 × 11 points
cavity_flow       a grid of 61 × 61, 25 steps of 5 pressure iterations
hdiff             a grid of 64 × 64 × 60
black_scholes     4,096 options
```

### With the single-operation kernels installed

`--install`: every way run again under `hotkernels.install()`, at the large size. The fused run's results are compared with NumPy's there as well.

```text
program            NumPy ms    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy
regression            41.67      43.99     0.45    43.54      39.86     0.47     0.75    38.64    1.10x   1.05x
attention             28.61      29.31     0.51    28.79      29.47     0.52     0.52    28.43    0.99x   0.97x
perceptron            25.99      88.12    28.53    59.58      91.95    28.93     3.73    59.29    0.96x   0.28x
char_rnn              28.71     195.48    60.09   135.39     201.88    60.28     8.73   132.87    0.97x   0.14x
particle_filter      320.63     370.33    86.22   284.11     358.80    86.14     7.45   265.20    1.03x   0.89x
veros_eos            127.63     128.07     0.81   127.26     104.43     0.82     2.42   101.19    1.23x   1.22x
veros_isoneutral     219.70     261.41     9.05   252.36     257.75     9.14     2.86   245.75    1.01x   0.85x
veros_tke             97.14     112.96    23.53    89.43     111.69    23.60     1.56    86.52    1.01x   0.87x
cavity_flow          155.67     225.06     9.58   215.48     161.14     9.79    24.82   126.53    1.40x   0.97x
hdiff                 66.18      71.57     5.79    65.78      29.93     5.80     0.20    23.92    2.39x   2.21x
black_scholes         87.07      89.16     1.91    87.24      86.53     1.93     0.22    84.38    1.03x   1.01x
```

## What the measurements show

**A program gains what share of its time is in expressions the library covers.**

| Program | Share of NumPy's time in expressions | Of the passes fusing could remove, removed by the 14 kernels | Fused against lazy | Fused against NumPy |
|---|---|---|---|---|
| `hdiff` | 91% | 100% | 2.51x | 2.33x |
| `cavity_flow` | 83% | 53% | 1.39x | 0.98x |
| `veros_eos` | 85% | 33% | 1.23x | 1.22x |
| `regression` | 38% | 47% | 1.11x | 0.97x |
| `particle_filter` | 42% | 48% | 1.03x | 0.91x |
| `black_scholes` | 17% | 25% | 1.03x | 1.00x |
| `veros_isoneutral` | 79% | 26% | 1.01x | 0.85x |
| `veros_tke` | 83% | 18% | 1.01x | 0.87x |
| `attention` | 5% | 50% | 0.99x | 0.98x |
| `perceptron` | 35% | 65%, all of it on arrays below the floor | 0.97x | 0.33x |
| `char_rnn` | 39% | 41%, all of it on arrays below the floor | 0.96x | 0.16x |

The second column is from the miner and the third from covering each program's expressions with the library, counted by elements at the small size; both are properties of the program and the library, not of a run.

- **Where the whole program is a few expressions and each has its kernel, the gain is the kernels' own.** The horizontal diffusion is four expressions of 18 operations, covered by four kernel calls. It writes 506 MB of results where the node-by-node run writes 1,624 MB, and runs 2.5x faster than that run and 2.3x faster than the program on plain NumPy. On a grid forty times smaller it is 1.9x and 1.6x.
- **Where the expressions are long and the shapes are small, a third of the passes go.** The equation of state is a polynomial of 153 operations in one expression. Shapes of two operations cover it with 45 kernel calls and leave 63 operations to NumPy: 108 passes for 153, and a run 1.23x faster. The two other Veros routines have more of their time in expressions the library covers least, and stay level.
- **Fusion does not make up for a write that copies.** The cavity flow has 1,980 of its nodes computed inside 580 kernel calls and runs 1.39x faster than without fusion, and that only brings it level with plain NumPy. The node-by-node run writes 5.7 GB of results for it and the fused run 3.2 GB. Each of its 1,220 writes, into the inside of a field or into one boundary row, copies the whole field first, and those writes are 54 ms of the fused run's 159 ms of computing. NumPy, which updates the three fields in place, makes none of those copies.
- **On small arrays the layer's own cost is all there is.** The perceptron and the recurrent network work on arrays of ten to twelve thousand elements, below the floor, so nothing in them is fused and their times are the lazy layer's: three and six times NumPy's. `build`, the time outside computing, is 29 ms of the perceptron's 97 and 55 ms of the network's 195, and computing alone costs two and four times what NumPy takes for the whole program. Fusion has nothing to say about either; the two figures are where that time is.
- **Planning is not free, and on arrays of a hundred thousand elements it is what decides.** A plan costs about ten microseconds and each node two to seven, the more the longer its expression. At the large size that is 2% of the equation of state's run and 15% of the cavity flow's, whose graph of 6,856 nodes is planned in 25 ms. At the small size, where the Veros arrays have 72,000 to 290,000 elements, planning takes 1.4 to 2.7 ms of runs of 10 to 24 ms and the three routines are 9% to 12% slower fused than not, although their computing is quicker. The perceptron and the recurrent network pay 4 ms and 9 ms to have their 565 and 207 graphs looked at and left alone.
- **The single-operation kernels and the fused ones do not meet.** Under `hotkernels.install()` the eager perceptron runs in 26 ms instead of 33 and the particle filter in 321 instead of 359, and the lazy and the fused runs gain the same calls: the ones NumPy still makes in them. A fused group reaches no hook, so the programs fusion helps most are the ones the installed kernels help least: the horizontal diffusion, the cavity flow and the equation of state move by 5% or less with them.
- **No kernel call was dropped for an exception in these programs.** The 24 calls given up in isoneutral mixing and the 12 in the kinetic energy were declined for layout, before anything was computed. In isoneutral mixing the operand steps three elements at a time along the innermost axis: one component of an array of vectors.
- **Calls into a mathematics library are the wall for two programs.** Black-Scholes spends 76% of its time in `erf`, `log` and `exp`, and the three kernel calls it gets move nothing. Its arithmetic between those calls is `(a - b + c) * y` and `P * d1 - Se * d2`, shapes the library covers only in part.
- **How long a plan lives matters more than how good it is.** The first version of the planner kept its plans alive until Python's collector of cycles ran, and that alone made a fused run slower than an unfused one ([Planning](#planning)). NumPy's own way of computing an expression, one temporary after another, hands each result memory that was freed a moment ago and is still in the cache. A fused run has to keep that property, or it loses in page faults what it wins in passes.

## What is not done

Each of these is a decision or a piece of work, in the order of what the measurements say it is worth.

- **Writing into the old contents of an array.** A deferred write copies the array first. When nothing else can read what the array held (no pending node, no lazy array, no view of the memory, and the memory never left the layer), the write could be made in place, as NumPy makes it. For a kernel that also needs no operand to overlap the part written; the cavity flow's stencils read copies the program itself makes, so they qualify. It is the largest single cost left in the two programs that update their arrays in place, and it is a question of the layer, not of fusion: the count of references that says a node is held is the tool, and here a wrong count would be a wrong result, where in planning it is only lost time.
- **Planning that costs less.** A loop writes the same expressions on every turn, and each plan reads them again. A plan remembered by the shape of its graph, as the layer remembers a node's shape and type, would cost a look-up; the reading itself, in C beside the core that writes the nodes, would cost a fraction of what it does in Python. Either would let the floor of 16,384 elements come down.
- **The long tail of shapes.** Fourteen kernels were written by hand. The loop of every element-by-element kernel is the same but for its loads and its operations, so a kernel for a shape could be written by a program from the shape's name and then checked by the judge like any other: the judge, not the author, is what makes a kernel trusted. Whether such kernels belong in the repository beside hand-written ones, and how many, is not decided here.
- **float32.** The transformer block is float32 throughout and nothing in it is fused. A float32 kernel is the float64 one at four lanes a vector.
- **Operands that are not contiguous along the innermost axis.** A column of a grid, `particles[:, 0]`, steps by a row at a time, and the kernels do not read it: 24 calls of isoneutral mixing and 12 of the kinetic energy are declined for that. A kernel could take a stride per input; whether it would beat NumPy's own strided loop is not measured.
- **Short runs.** An expression over `(400,000, 4)` with a row of four broadcast along it is 400,000 kernel calls of four elements, and is slower than NumPy's two passes (3.5 ms against 2.7 ms). The module should decline a run that short; where the line is, is not measured.
- **Reductions along an axis, and reductions other than a sum.** `np.sum(e ** 2, axis=-1)` and `np.max(np.abs(e))` occur in the programs and are left to NumPy.
- **Calls into a mathematics library.** `np.exp(a * b + c)` cannot be one kernel with NumPy's bits. It can still be computed without three passes over main memory: NumPy's own loops run over a block that fits the cache, one operation after another, which keeps NumPy's bits by construction. That is a different executor from this one, and the one [lazy-design.md](lazy-design.md#what-the-graph-is-for) names beside it.
- **x86-64.** The kernels are ARM64 only; on another architecture each is built from its C reference, which the judge holds to the same bits. Nothing was timed there.
- **More programs.** [Issue 298](https://github.com/sofeikov/hotkernels/issues/298) lists 21 programs that come out as one graph each, most of them stencils and loops over rows. They are the next corpus to mine, and with in-place writes the ones this approach should suit best.
- **What the layer costs on small arrays.** Not fusion's subject, and the tables show it: computing a node costs about three microseconds beyond NumPy's call, 1.1 of them the recording of warnings and 0.7 the `np.errstate` entered per node, and a perceptron on batches of 128 runs three times slower lazily than eagerly, fused or not. A group computed by a kernel pays neither, which is part of what fusion gains on arrays of a few thousand elements.

## What changed outside `fuse.py`

- **`evaluate.py` asks a planner,** and has a meter and a clock for the tools. With none set it computes as before. [lazy-design.md](lazy-design.md#computing) says what a planner may and may not change.
- **The judge looks for a kernel in a second place,** `prototypes/lazy/kernels/`, and locks its spec and reference to `main` there as it does in `kernels/`. Nothing else in the judge changed, and `uv run judge selftest` passes.
- **Nothing in the lazy layer's rules changed.** The deferred set was looked at for what fusion would need added to it, and nothing was: over the twelve programs two calls are made at once, `np.median` once and `ndarray.flatten` four times, and neither is inside an expression. What cuts these programs' graphs is the program asking for a value (a loss turned into a `float`, a test, an index), which no rule removes.
