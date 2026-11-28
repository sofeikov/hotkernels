<p class="lede">The lazy layer of the last chapter writes a program's NumPy operations down as a graph and computes it one node at a time, with the call the program wrote. That gives NumPy's results and nothing faster. This chapter is about computing the graph faster, with the same bits: an expression of many operations is computed in one pass over memory, by kernels written for the shapes of tree that real programs contain. Nothing about it is clever. Every rule below exists either to keep the result bit for bit NumPy's, or to make sure that fusing never costs more than it saves.</p>

::: note A prototype, not part of the package
Everything here is in the lazy layer's prototype, [prototypes/lazy/](repo:prototypes/lazy), and its design document, [docs/lazy-fusion-design.md](repo:docs/lazy-fusion-design.md). None of it is installed with `hotkernels`. The eighty fused kernels live in [prototypes/lazy/kernels/](repo:prototypes/lazy/kernels/README.md), and the judge checks them as it checks the kernels of `kernels/`. The trees, names and programs in the figures were produced by running the prototype's own code on the expressions shown. Every timing was taken on an Apple M1 Pro with NumPy 2.5.5, and what fusion gained over whole programs is the [next chapter]([[ root ]]report/fusion-results/).
:::

## Why a whole expression in one pass

Take one line of a weather model, the five-point Laplacian of the horizontal diffusion stencil in npbench:

```python
lap = 4.0 * u[1:-1, 1:-1] - (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
```

NumPy computes it as five calls, one per operation. Each call reads its operands from memory and writes a whole new array, a **temporary**, which the next call reads back. Five passes over memory and four temporaries, for one value per element that a single loop could compute by reading the five slices once and writing the result once. [The first figure of the lazy chapter]([[ root ]]report/lazy/#one-call-at-a-time) counts the bytes this costs. Once the arrays are larger than the caches, [time follows bytes]([[ root ]]report/machine/#fact-four-memory-has-a-speed-limit), and the extra passes are most of the time.

Only the graph can see the expression. A kernel installed by `hotkernels.install()` reaches a program through a hook inside NumPy, and a hook sees one call at a time. It cannot hold `u[2:, 1:-1] + u[:-2, 1:-1]` back to combine it with the next addition, because the result of the first call is an `ndarray`, and anyone may read an `ndarray`'s memory the moment it exists ([the lazy chapter]([[ root ]]report/lazy/) has the argument). A lazy array has no memory until the program asks for its values, so by the time the layer computes anything, the whole expression is in the graph.

The approach taken is the plainest one there is:

- A **pattern** is a small tree of operations with named inputs, such as `a + s * b`.
- A **fused kernel** is written for each pattern. It computes the whole tree in one pass over memory.
- A longer expression is **covered** with several patterns, and their kernels are called a block at a time, so that what one gives the next stays in the cache.
- An operation no kernel is written for is computed by NumPy, as before. A missing pattern loses nothing.

## What an expression is

An **expression** is a tree of element-by-element operations on float64 arrays in which every intermediate result is read once, by the operation above it, and by nothing else. `w - rate * gradient` is two operations and one intermediate, `rate * gradient`. The Laplacian above is five operations and four intermediates.

What an expression may contain is decided by one rule: **each operation must be a single IEEE operation that NumPy makes with the same instruction.** Then a kernel that makes the same operations in the same order rounds every element the same way NumPy's separate passes do, and its bits are NumPy's.

| An expression may contain | Why |
|---|---|
| `+`, `-`, `*`, `/`, `np.maximum`, `np.minimum`, unary `-`, `abs`, `np.sqrt`, `np.square`, `x ** 2` | Each is one IEEE operation, and NumPy's loop for it executes the one instruction a kernel would. `x ** 2`, `np.square(x)` and `x * x` are all one multiplication |
| `np.radians`, `np.degrees` | NumPy's loop for each is one multiplication by a constant, which a kernel makes with the same float64 |
| A comparison, as the condition of `np.where(c, a, b)` | The comparison gives a mask, which a kernel keeps in a register and never writes to memory |
| A sum or a mean of every element, at the top | NumPy's order of addition can be reproduced exactly ([below](#a-sum-adds-in-numpys-order)) |
| A sum or a mean along a last axis of two to seven elements, of an array or of a product | NumPy adds a row that short from left to right, starting from 0.0 ([below](#a-sum-adds-in-numpys-order)) |

Everything else ends an expression, and each ending has a reason:

| What ends an expression | Why |
|---|---|
| `np.exp`, `np.log`, `np.tanh`, `np.arctan2`, `x ** 3`, any power but 2, `%` | NumPy calls the platform's mathematics library for these. A kernel is a leaf function that calls nothing, and a second implementation of `exp` would not have the library's bits |
| An array of any element type but float64, a NumPy scalar that is neither float64 nor a whole number | The kernels are for float64 |
| An intermediate read twice, or one the program still holds a name for | Its value is wanted again, so it has to exist; computing past it would compute it twice. It becomes an operand of the expressions that read it |
| An intermediate of another shape than the result | It belongs to another loop |
| The matrix product, any other reduction along an axis, indexing by arrays | Not element by element. NumPy computes each, and the result is an operand of what reads it |

Each thing an expression reads, an **operand**, is one of two kinds along the innermost axis. A **V** is an array as long as the result. An **S** is the same number for the whole row: a Python number, a float64 scalar, a whole number of NumPy's such as the `np.int64` that indexing a table of integers gives, an array with no dimensions, or an array of length one along that axis, such as a row mean kept with `keepdims=True`. NumPy converts a whole number to float64 before its float64 loop runs, and the kernel is handed the same float64. Along the outer axes an operand may broadcast in any way.

A pattern's **name** writes its tree out with numbered leaves. `a + s * b` is `add(V0,mul(S0,V1))`. Two rules make one name stand for every way a program might write the same shape:

- **Each place an array is read is a leaf of its own.** `a + s * a` is also `add(V0,mul(S0,V1))`, served by the same kernel with one pointer given twice.
- **The two operands of a sum, a product, a maximum or a minimum are put in a fixed order.** `c + a * b` and `b * a + c` are both `add(V0,mul(V1,V2))`. Only the two sides of one operation swap. Which additions happen first is never changed, because that changes the bits.

Pick an expression, most of them lines from the programs the prototype was tested on, and see what the planner sees:

<figure class="widget">
<div data-widget="fusetree" data-example="scaled"></div>
<figcaption>The tree of each expression, root on the left, as the prototype reads it off the graph: circles are operations inside the expression, square leaves are arrays, round leaves numbers, and dashed leaves operations outside it that NumPy computes first. The pattern names are <code>fuse.canonical</code>'s, and <b>Kernel calls</b> shows <code>fuse.tile</code>'s cover with the library of eighty kernels. Recorded by running each line on the prototype's lazy arrays.</figcaption>
</figure>

Three things the figure shows are worth saying in words:

- **The graph is the program after Python has run.** In the cavity flow's pressure step,

  ```python
  dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
  ```

  is one multiplication of an array by a number, because `dx` and `dy` are Python floats and Python folds them first. Loops are unrolled the same way: six turns of `x = x * 1.01 + b` are one expression of twelve operations.
- **An ending splits an expression; it does not lose it.** `np.exp(a * b + c) * 2.0 + a` is two expressions with NumPy's exponential between them. An intermediate read twice does the same: in the equation of state, `t11 = sa * (v15 + v16 * ct)` is read by two later lines, so it is an expression of three operations by itself.
- **A real line is a large tree.** The cavity flow's update of `u` is 24 operations on 15 slices. One expression of the Veros equation of state has 153.

## The library: a pattern is a kernel's own specification

Every kernel of this project has a `spec.py`, and every spec names an **oracle**: the NumPy expression the [judge]([[ root ]]report/judge/) holds the kernel to, bit for bit. For a fused kernel the oracle is the expression it fuses:

```python
def oracle(a, b, s):
    return a + s * b
```

The planner learns which pattern a kernel serves by calling that oracle with lazy arrays. The oracle writes a graph without computing anything, the expression is read off the graph exactly as off a program's, and its name, `add(V0,mul(S0,V1))`, is the pattern. Which leaf is which argument is known because the arrays passed in are the leaves read.

**So the specification is its own pattern.** What a kernel is matched against in a program and what the judge checks it against are one function; they cannot disagree. There is no table of patterns anywhere to keep in step with the kernels. A kernel whose oracle is not one expression of the operations above, or that reads an argument twice, has no pattern and is refused when the library is loaded.

Every kernel takes its arguments in one order: the input arrays, the output, the length, then the numbers. The module that calls the kernels, [_fused.c](repo:prototypes/lazy/_fused.c), calls every one of them with eight integer arguments and eight floating-point ones, of which each kernel reads those it has. ARM64 and x86-64 both pass that many in registers or in order on the stack, so one call reaches any kernel.

### Fourteen written by hand

The first kernels are the shapes that weigh most across the programs: nine shapes of two operations, the one reduction of every element the programs make, and four whole expressions of the two stencil programs.

| Kernel | Computes | Found in |
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

A shape's name says everything its kernel has to do: which operations, on which operands, in which order. And the loop for it is always the same loop: load eight elements of each array, make the operations in the order the tree gives, each with one instruction that rounds once, and store. So most kernels are written by a program, [make_kernel.py](repo:prototypes/lazy/make_kernel.py), from the name alone:

```bash
uv run python -m prototypes.lazy.make_kernel "mul(V0,add(S0,mul(S1,V1)))"
uv run judge check fuse_mul_v_add_s_mul_s_v_f64 --unlocked
```

That shape is a line of the Veros equation of state, `sa * (v15 + v16 * ct)`, which the figure above shows. The tool writes the files every kernel has:

- **`spec.py`**, whose oracle is the shape as a NumPy expression, `a * (s0 + (s1 * b))`, and whose sets of numbers try each of them at zero of either sign, at a value that overflows, a denormal, an infinity and a NaN.
- **`naive.c`**, the C reference, the same loop in one line. It starts with `#pragma STDC FP_CONTRACT OFF`, for the reason of [the hazards below](#why-the-bits-are-numpys).
- **`arm64/kernel.s`**, the NEON assembly. Its loop, eight elements a step in four vectors of two (shortened, with comments added):

  ```text
  ldp     q16, q17, [x0]          // a: 8 elements in four registers
  ldp     q18, q19, [x0, #32]
  ldp     q20, q21, [x1]          // b
  ldp     q22, q23, [x1, #32]
  fmul    v20.2d, v30.2d, v20.2d  // s1 * b, four instructions; s1 is in every lane of v30
  ...
  fadd    v20.2d, v31.2d, v20.2d  // s0 + (s1 * b)
  ...
  fmul    v16.2d, v16.2d, v20.2d  // a * (s0 + (s1 * b))
  ...
  stp     q16, q17, [x2]          // the 8 results
  stp     q18, q19, [x2, #32]
  ```

- **`arm64/attempts.md`**, the first entry of the log every kernel keeps.
- **`x86_64/kernel.s`** and **`x86_64/attempts.md`**, the same loop in AVX2, eight elements a step in two `ymm` registers of four, and its log. On x86-64 a kernel's assembly is kept only where it beats its compiled C, which it does for two kernels ([Another processor]([[ root ]]report/fusion-results/#another-processor)).

The rest of the loop is the same for every shape. Every load of a step comes before its store, so the output may be exactly one of the inputs ([Writes](#writes-cost-less-with-fusion)). A remainder of one to seven elements is one more step that ends exactly at the end of the arrays and overlaps the step before it, and fewer than eight elements are done one at a time. Registers are given to the tree by the usual rule, the operand that needs more of them computed first, four registers to a value, and each number is held in every lane of one register. Shapes of up to six arrays and eight numbers fit.

**Nothing the tool writes is trusted.** The judge checks the kernel against NumPy as it checks a kernel written by hand, and the pattern it serves is read off its oracle. A test holds the files in the repository to what the tool writes today. The written kernels are not tuned: they load with plain `ldp` and stay as the tool left them.

Which sixty-six: the eighteen **single operations** (the four of arithmetic on two arrays and on an array and a number either way round, a maximum and a minimum, a negation, an absolute value, a square root, a square); the **forty-six shapes** of two to six operations that cover the programs' expressions best, chosen one at a time by a **miner**, a tool that runs the programs and weighs every shape by the time NumPy spent on it (the next chapter has [how]([[ root ]]report/fusion-results/#which-shapes-to-write)); and the two halves of an advection flux, of four operations and eight. What the tool does not write is a comparison with `np.where`, a reduction, or float32.

::: aside The one shape the judge refused
The miner asked for `div(add(S0,neg(abs(V0))),S1)`, which is `(s0 + (-|a|)) / s1`, from the isoneutral mixing routine of Veros. The assembly the tool wrote makes exactly those operations, and is NumPy's. The judge failed the kernel anyway, and it was right to: the *C reference* was wrong. The compiler had turned the reference's `s0 + (-|a|)` into `s0 - |a|`. For ordinary numbers the two are equal. For a NaN they are not: negating a NaN flips its sign bit, while a subtraction passes the NaN it is given through with its sign, so the reference gave a NaN of the other sign from NumPy's. A kernel whose reference is wrong is not kept, so this shape has no kernel. It is a small instance of the whole chapter's point: two ways of writing the same arithmetic are not the same program, and only a check against NumPy's bits tells.
:::

`np.maximum` and `np.minimum` need the same care. On ARM64 NumPy's loops execute the processor's `fmax` and `fmin`, under which a NaN wins over a number and the maximum of two zeros is the positive one. On x86-64 they execute `vmaxpd` and `vminpd` with a NaN of the first operand carried through, under which a NaN still wins but the maximum of two zeros is the second operand, whatever its sign; there `np.maximum(x, 0.0)` and `np.maximum(0.0, x)` differ, and the layer matches a maximum's operands only in the order the program wrote them. C's `fmax` function is something else, so the reference spells out each architecture's rule.

## A whole expression, block by block

A kernel covers a shape of up to eight operations. The Veros equation of state is one expression of 153, and covered with shapes of the library it is 64 kernel calls. If each call wrote a whole array for the next to read, that would be 64 passes over memory where NumPy makes 153: better, and still far from one.

So the kernel calls of an expression are a **program**, and the program is run a block at a time:

```text
for each block of 4,096 elements:
    buffer 0 = kernel 1 (operands)
    buffer 1 = kernel 2 (operands, buffer 0)
    ...
    result   = kernel k (buffers, operands)
```

- **What one kernel gives the next is a buffer of one block,** 32 KB, reused for every block. It is written and read within the cache. Only the expression's operands are read from memory and only its result is written to it, however many operations lie between.
- **The steps are in the order of the tree, the deepest first.** A step takes a buffer for its result once the steps it reads have been written, and a buffer is free again as soon as the step that read it is done. A chain of calls, each reading the one before, runs in two buffers however long it is.
- **A block is 4,096 elements.** Timed from 256 to 32,768 on the programs, the equation of state ran in 83 ms with blocks of 256, 77 ms at 1,024, 72 ms at 4,096 and 68 ms at 32,768, and isoneutral mixing the other way, from 203 ms to 209 ms; the other programs did not move. A program of one step has nothing to keep in the cache and is one call for the whole run.
- **A program is checked before it computes anything.** `_fused.program` checks every step against its kernel's number of operands, every place a step reads against what was written before it, and that only the last step writes the result.

Step through one block of the cavity flow's update of `u`, the five calls the planner chose for it in the first figure:

<figure class="widget">
<div data-widget="fuseprogram" data-program="cavity"></div>
<figcaption>One block of a program, step by step: what each kernel reads from memory and from the buffers, and where it writes. The programs are the ones the prototype's planner wrote for each expression, with the library of eighty kernels. The counts under the figure are counting, not timing: each array a call reads or writes, once over its elements; for the program, each slice it reads counts once, since a block a later step reads again is still in the cache.</figcaption>
</figure>

### Every operation needs a kernel

The library has to have a kernel for each single operation, `add(V0,V1)`, `mul(S0,V0)`, `div(V0,S0)`, `sqrt(V0)` and the rest, though they fuse nothing by themselves. The reason is the long tail of shapes. Real expressions are made of many different trees, and the shapes the miner found trail off slowly: twenty-two shapes remove half the passes that fusing every expression whole would remove, sixty remove 61%, and the sixtieth still adds a tenth of a percent. No library of whole shapes covers that tail, and a pass left to NumPy in the middle of a program is a pass over memory. With every single operation a kernel too, a shape that is missing costs a step inside the cache, not a pass. *No shape fits* in the figure is such an expression: five kernels of one operation each, and still one pass.

An operation by itself, with nothing to fuse with, is still NumPy's call, with one exception that comes [further down](#computing-a-group-on-arrays-of-any-shape): rows of a few elements.

### Covering a tree by the fewest passes

Which shapes compute which part of a tree is found from the leaves up. At each operation, the shapes of the library that have that operation at their top are held against the tree; a shape whose top operation could not take the operation's operands, by what each one is (a number, an array, a named operation), is not walked at all. The cover with the fewest passes wins. A kernel's step stays in the cache and NumPy's call is a pass over memory, so where the two give as many passes the kernel is preferred, and among kernels the largest shape.

## Why the bits are NumPy's

The argument is made per operation, and the judge checks it per kernel. Each operation a fused kernel makes is one IEEE operation that NumPy's loop makes with the same instruction. The operations come in the program's order and each result is rounded by itself, exactly as when NumPy writes it to a temporary and reads it back. Each element then goes through the same roundings as in NumPy, so the result has the same bits. That argument has holes wherever an optimiser would normally help, and each is closed:

| Hazard | What a kernel does | How it is checked |
|---|---|---|
| A product and a sum fused into one multiply-add, which rounds once | `fmul` then `fadd`, never `fmla`. The C reference starts with `#pragma STDC FP_CONTRACT OFF`; without it clang fuses the two at the judge's flags | The judge compares bits with NumPy's `a + s * b` on every size to 65, on random values and on its special ones. Compiled without the pragma, the reference's loop is `fmla` |
| A division made as a product with a reciprocal | `fdiv` by the number | The same, with divisors that are zero, tiny and infinite |
| Additions regrouped | The program's brackets are the pattern's: `add(V0,add(V1,V2))` adds the last two first. A reduction makes NumPy's pairwise additions | The judge, and a test at every length to 700 |
| Signed zeros, NaN, infinities, denormals | Whatever the instruction gives, which is what NumPy's loop gives | The judge's value classes |
| A result laid out differently, so that a later sum adds in another order | Operands are taken only where NumPy's result lies in the order the module walks, C order or the one other order every array lies in, and the output is laid out so | Tests that hold bits, strides and flags to NumPy's in every order of two to four dimensions, and of layouts the kernels do not read |
| An output that overlaps an input | Every load of a block comes before its store, so the result may be exactly one of the arrays; any other overlap is declined before a kernel runs | Tests of a program written in place at every length around its blocks, and of the module's refusal |
| A warning or an error | The flags are read and NumPy's calls are made ([below](#floating-point-exceptions-are-numpys-too)) | Tests of each, with the line |

The first hazard is the one a compiler creates without being asked. A **fused multiply-add** computes `a * b + c` exactly and rounds once. That is more accurate, and it is not what NumPy computes: NumPy rounds the product to float64 when it writes the temporary, then rounds the sum. C lets a compiler contract the two into one instruction unless the source forbids it, which is what the pragma in every reference does; without it, clang makes the multiply-add at the flags the judge compiles with. Try values:

<figure class="widget">
<div data-widget="fusefma" data-preset="scaled"></div>
<figcaption>c ± a × b with two roundings, as NumPy and every fused kernel compute it, and with one, as a fused multiply-add would. Both are computed in the page: the first with ordinary float64 arithmetic, the second exactly, with integers, and rounded once to the nearest float64. The bits that differ are marked. Edit the numbers, or draw random ones from [0, 1).</figcaption>
</figure>

Draw random numbers in the figure and keep count: the two differ on ordinary inputs, not only on contrived ones. A kernel that differed from NumPy on some inputs would make a program's results depend on whether it ran lazily, which is the one thing the layer promises they do not.

### A sum adds in NumPy's order

`(x ** 2).sum()` is the pattern `sum(sq(V0))`: the square and the sum in one pass, with no array of squares. A sum has an order, and NumPy's is not left to right. For a contiguous array it adds **pairwise**: eight running sums over blocks of up to 128 elements, combined in a fixed tree whose shape depends on the length ([kernels/sum_f64/spec.py](repo:kernels/sum_f64/spec.py) has the rules). Floating-point addition is not associative, so a sum in any other order is another number.

A fused reduction is exact only because it makes the same additions in the same order on the values it computes. `fuse_sum_squares_f64` walks NumPy's tree with two small stacks below the stack pointer and squares each element as it meets it; a test compares it with `np.sum(x ** 2)` at every length to 700 and at several up to 300,000. The mean is the same kernel, since NumPy's mean is its sum divided by the count. A kernel reduces only an array that lies in one run, C-contiguous or left in another order by a transpose, which NumPy adds in memory order whatever its dimensions, and only over every element. A sum along an axis adds in yet another order (down the columns it is not pairwise at all) and is left to NumPy, but for one case: along a last axis of fewer than eight elements NumPy adds each row from left to right, starting from 0.0, so `np.sum(a * b, axis=-1)` on rows of two to seven is made by one loop of the layer's own, with no array of products, wherever NumPy's result would be in C order. `np.max` and `np.min` along such rows follow no one rule on x86-64 and are left to NumPy.

### Floating-point exceptions are NumPy's too

NumPy reports overflow, division by zero and invalid operations as warnings, or as errors under `np.errstate(over="raise")`, with the name of the operation and on the program's line. A kernel of several operations cannot say which of its operations raised a flag. So the module **clears the processor's flags before a kernel runs and reads them after.** If a flag is up that NumPy would report under the `np.errstate` the operations were written under, the kernel's result is dropped and NumPy's own calls are made, one node at a time. The warning, the error, its words and its line are then NumPy's by construction, and so are the values.

- **An exception the program silenced costs nothing.** Under `np.errstate(all="ignore")`, and for underflow always, the kernel's result is kept.
- **An expression that raises one is computed twice.** The line that wrote it is remembered, and its expressions are left to NumPy from then on, so a loop pays once.
- **A comparison raises a flag NumPy does not report.** The processor's comparison marks a NaN operand as an invalid operation, and NumPy's comparison clears that flag. The fused flux limiter is therefore computed by NumPy's calls when its data has a NaN, unless the program has silenced invalid operations. The kernel for `np.where(a <= s, b, c)` compares with the quiet comparison on x86-64, which raises nothing, so data with a NaN is still the kernel's.

### The one thing not specified: which NaN

When an operation meets two NaNs of different sign or payload, the processor returns its first operand's. Which operand is first in NumPy's loop is its compiler's choice, and a pattern's name, with its operands in a fixed order, does not record how the program ordered a sum. So whether a result is NaN is always NumPy's answer; which NaN it is, is NumPy's whenever a program has one kind of NaN, which is what arithmetic and `np.nan` give.

## Planning a computation

When the program asks for a value, the layer lists the pending nodes the value needs and asks a **planner** which of them to compute together. [fuse.py](repo:prototypes/lazy/fuse.py) answers in four steps:

1. **Writes into whole arrays become their one call** ([Writes](#writes-cost-less-with-fusion)).
2. **Expressions are read off the nodes,** the newest first, since the operation written last is the top of its tree. An operation is the top of an expression unless an expression already found contains it. Below it, each operand is inside the expression if it is an operation a kernel can make, of the same shape, about to be computed, read by this operation alone, and held by nothing else.
3. **Each expression is covered with shapes of the library** by the fewest passes.
4. **The tiles that feed one another become one program,** a group under the node whose value it gives. The plan is the groups.

**"Held by nothing else" is a reference count.** An intermediate the program has a name for, or that a node outside this computation reads, will be asked for again. Both are references to the node, and CPython counts references exactly: a node of a computation is referred to by the nodes of it that read it, by the one list it is in, and by nothing else unless one of the two holds. The count decides only what is *worth* fusing. A node a plan skips stays pending and is computed when asked for, so **a wrong count costs time and never a result.** Two things this found while it was built: a `for` loop's variable still refers to the last node after the loop, and pytest's rewriting of `assert` keeps every value of the expression in it alive. Both make a node look held.

**Nothing planned outlives the computation.** A plan refers to nodes, and nodes to arrays. If a plan stayed alive past its computation, its arrays would stay in memory until Python's cycle collector next ran, and every result in between would go to fresh memory where NumPy's calls reuse what was just freed. This happened while the module was built: two functions that called themselves through a name of the function they were made in formed a reference cycle, and one run of an ocean routine took 46,000 page faults, where the run without fusion took one, and ran a quarter slower. A test now runs a fused computation with the collector off and checks that no node is left.

**A plan is remembered by the shape of its expression.** A loop writes the same expressions at every turn, and covering a tree with shapes is most of what planning costs. What is kept is the program, and where each thing it reads is among the expression's operands, keyed by the tree's operations and the pattern of its operands. The next expression of that shape is planned by a look-up.

**The expressions are read in C.** Reading is a walk over every node of the computation with a few questions asked of each, and in Python that walk was three quarters of what planning cost once plans were remembered. The C core that writes the nodes ([the lazy chapter]([[ root ]]report/lazy/#what-writing-a-graph-costs)) also reads the expressions, by the functions of the Python reader one for one. The Python reader is the definition, and a test runs both on every computation of eight programs and checks that they find the same expressions. Planning the cavity flow's 260 expressions costs 3 ms of a 55 ms run: about 2 ms to read them and the rest to place each. With the Python reader it cost 16 ms, and without remembered plans 45 ms.

**An expression is read 200 operations deep at most.** A loop that adds to one large array a few thousand times without asking for a value writes one chain that long, and reading, covering and placing a tree each walk it a call per level, of which Python allows a thousand. Below the limit an operand is read as an array, the top of an expression of its own, computed first. Every operation of the chain is still a kernel's; where the cut falls costs one array written to memory per 200 operations.

**Arrays of fewer than 32,768 elements are left alone.** Writing a node, planning it and coming back to it costs about a microsecond, and a NumPy pass over fewer elements than that costs little more, so there is nothing to win back. Where the floor is matters little above 16,384 and a great deal below: the fractal flame, whose arrays have 25,000 elements, ran in 149 ms with the floor at 16,384, 148 ms at 32,768 and 147 ms at 65,536, and the perceptron, whose arrays have 8,000, in 35 ms at all three; with the floor at 8,192 they took 235 ms and 76 ms. No other program moved by 1%.

**An expression with one value for each row gets a second program.** `dx[:, :, None]` beside a grid gives each row of the grid one value. A kernel takes each of its numbers as one value for the whole call, so such an expression is planned with each row a run of its own and the row's value as the number. That is quickest when the kernel can read the other arrays where they lie. When one of them steps over elements and has to be gathered ([below](#computing-a-group-on-arrays-of-any-shape)), a block spans many rows, no one number fits it, and that plan is given up. So where rows are shorter than 32, the expression is also planned with the value read as an array, and the second program is kept beside the first for when the first is given up; the module then fills a buffer with each value repeated as often as its row is long. The same second program runs where rows are shorter than 16 and nothing is gathered: `1 - weight * (1 - rgb)` on an image with one weight for each pixel is rows of three or four, a call of every step for each such row is declined, and with each weight spread over its row inside the block the program is 1.7 to 3.1 times NumPy's passes.

## Computing a group on arrays of any shape

A kernel is a leaf function over contiguous runs of float64. Programs use slices, columns, broadcast rows and views of every kind. [_fused.c](repo:prototypes/lazy/_fused.c) stands between the two: it walks the outer axes of the arrays and runs the program once for each contiguous run, block by block. What it does with each layout, and when it gives a layout back to NumPy:

<figure class="widget">
<div data-widget="fuselayout" data-layout="contiguous"></div>
<figcaption>How the module lays a program over the memory of its arrays: one square per element, a row of squares per row of memory, coloured by the run it is computed in. The grids are schematic; the text names the real sizes. Rules and figures from docs/lazy-fusion-design.md, "Computing a group", timed on an Apple M1 Pro.</figcaption>
</figure>

The rules in short:

- **A C-contiguous expression of any number of dimensions is one run,** and a slice of a grid is one run per row.
- **Arrays that all lie in one other order are walked in that order.** `np.mgrid[...].transpose(1, 2, 0) % 1` is an array whose last axis is the slowest in memory. NumPy gives a result the order its operands lie in and walks it as it lies; the module sorts the axes by stride as NumPy does, and the result is laid out as NumPy's. Only where every array has every axis and none has length one: otherwise the order of NumPy's result depends on each operation of the expression.
- **An operand that is the same along an outer axis does not end a run.** It is read from a buffer that repeats it for the length of a block.
- **An array that steps over elements is gathered** into a buffer a block at a time, but only where the program saves more passes than the copies cost. A copy costs about what a NumPy pass over the array does.
- **Rows of a few elements are worth a kernel even for one operation,** because NumPy calls its loop once per row and the call costs more than the row's arithmetic. On 100,000 rows of 2, 4, 8 and 15 elements, `table - row` (an operand the same for every row) ran 9.1, 6.9, 2.6 and 1.8 times as fast as NumPy's one call, and `wide[:, :half] - wide[:, half:]` 5.4, 4.0, 3.6 and 2.6 times; a call of the kernel for each row, which the module made before it had its own loop, was 0.2 to 0.9 times NumPy's speed on these layouts.
- **Anything else is declined before anything is computed,** and the group's nodes are computed by their own calls, in order.

## Writes cost less with fusion

The lazy layer makes a write into an array it owns a node of the graph, defined as a copy of what the array held with the write made into it, and computes it into the array's own memory when nothing reads what the array held ([the lazy chapter]([[ root ]]report/lazy/#writes-and-why-a-loop-stays-one-graph)). With fusion the write and what it stores are one expression, and several kinds of write cost less still. All give the bits of the copy and the write.

| The program writes | Node by node | With fusion |
|---|---|---|
| `w -= s * g`, on a whole array | `s * g` into a temporary, then `-=` into `w` | The operator is an operation of the expression, `sub(V0,mul(S0,V1))`: one kernel call, no temporary. Into new memory, or into `w` itself when nothing reads the old `w` and no exception is reported (below) |
| `u[1:-1, 1:-1] = e` | `e` into a temporary, which is assigned into `u` | `e` computed straight into that part of `u` by its program. No temporary |
| `u[1:-1] += s * v` | `s * v` into a temporary, then `+=` into the part | One program that reads the part and gives its new contents, written into the part itself where no exception is reported, and otherwise into a temporary of the part, which is then assigned |
| `np.add(a, b, out=x); np.multiply(x, 0.5, out=x)` | Two calls into `x` | One kernel call, `mul(S0,add(V0,V1))`, into the memory `x` had before the first of the two |

**A kernel writes where the array is only when its result can still be dropped.** A result is dropped when the kernel raised an exception NumPy reports, and then NumPy's calls are made on the same operands, which need to be what they were. So a program writes into the array itself when it reads nothing of what the array held (`u[1:-1, 1:-1] = e` with no `u` in `e`), or when nothing is reported under the `np.errstate` the lines were written under. Otherwise its result goes to new memory, or to a temporary of the part that is assigned once the result is known to be kept: one pass more, and still no copy of the array.

**The result may be one of the operands, element for element.** `x -= s * y` made in `x` reads and writes the same memory. Every kernel loads all of a step's elements from every array before it stores them, so that is right for a length that is a multiple of eight, and for fewer than eight, which are done one at a time. The module makes every such call as the whole steps and then the elements left over, because the kernel's last, overlapping step would read elements it had already written. A result that overlaps an operand any other way (`u[1:] = u[:-1] + 1`) is declined before a kernel runs.

**A program may read other parts of the array it writes.**

```python
y[i, :] = a * x[i, :] + b * y[i - 1, :] + c * y[i - 2, :]
```

reads two rows of the array it writes a third of. A row read is a view, and a view refers to its array, which the rule for writing in place counted as a reader of what the array held. So the write went to a copy of the whole array, once for every row: the Deriche filter, 1,500 rows of 1.5 million elements, took 11.6 s for NumPy's 37 ms. A view that shares no byte with the part written shows the same elements before and after the write, whoever reads it, so such views are no longer counted as readers, and the row is written where it is. A view that does share bytes is counted as before.

## As the graph is built: small calls are made at once

A node costs time three times: when it is written, when its computation is planned, and when the loop that computes it comes to it. NumPy makes a call on an array of a hundred elements in a third of a microsecond. A program on small arrays ran two to eleven times slower lazily than eagerly for that alone, and nothing that computes a graph faster can win it back, because on small arrays there is nothing to fuse.

So with fusion on, **what is small is done as the program writes it**, by the C core:

| The program writes | What happens |
|---|---|
| A call that gives fewer than 32,768 elements, on arrays that all have their values | NumPy's call, there and then. The result is a lazy array that has its values |
| An operator on small arrays one of which is still pending | What it reads is computed, and then the same. A small call never waits |
| A call of any size that no kernel could make, on arrays that all have their values: a comparison, a selection by a mask, a matrix product, `np.exp`, anything complex or float32 | The same. Nothing written on top of it could be computed together with it, so its node would only cost |
| `x <= s` on a float64 array of 32,768 elements or more and a number: the one comparison a kernel selects by with the array read where it lies | It waits, so that the `np.where` written on top of it is computed with it. Every other comparison is made at once |
| A view, a transpose or an element of an array that has its values | NumPy's own, whatever the size: a view costs nothing to take |
| A write, when no node at all is pending | NumPy's write, into the array's own memory |
| Anything else: arithmetic on float64 that gives a large result, a large call with an operand still pending, a write while something is pending | A node, as before |

**This is what the eager program does at that line.** The value is NumPy's, and so are an error and a warning, raised where the program wrote the call. Nothing later can tell the difference: the result is a lazy array like any other, and a large operation on it is deferred. The result is one object, not the three a node needs, and the core's functions stand in the class's own table for the operators, so `a + b` is one call from the interpreter to NumPy's ufunc. What it costs is 15 to 115 ns a call over NumPy's own, timed on the M1 Pro on sixteen elements with each line in a loop of 300,000 turns:

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

`a * 2.0` costs nothing because an operator on a lazy array calls the ufunc directly, where `ndarray`'s own operator first looks at what the other operand is. A function that is not a ufunc, called through the stand-in for the `numpy` module, costs the most: the stand-in's own call, and NumPy's Python wrapper around the function.

**A small call does not wait for what it reads.** `y[i, :] = a * out[i, :] + b * y[i - 1, :]` in a loop, with `out` a large array still pending, is thousands of operations on rows of a thousand elements. Each one used to wait as a node, because the row of `out` it read had no value yet: the Deriche filter wrote 36,000 nodes that way and took 170 ms for NumPy's 37 ms. A pending operand of fewer than 32,768 elements is now computed, with whatever it reads, and the call is made at once on its value: 49 ms.

Two things are given up, and the design says so:

- **Fusion across a peek.** In `t = a + b; s = t[0] * 2; u = t * c`, the small call `t[0] * 2` computes `t` by itself, where `t` and `u` could have been one pass. That is one pass of one array, against a node for every small operation that follows.
- **The graphs of a program on small arrays.** With fusion on they are not written, so what the layer could show of such a program is gone. That is why the limit is a switch: without `fuse.enable()` every operation the layer defers is a node, and the tools that look at graphs see them all.

## What it gained

The layer runs as the package is used, with `hotkernels.install()` in effect, so the single calls between expressions still reach the package's own kernels. Measured on the M1 Pro over twenty-six programs, every one ran within 1% of NumPy's speed or faster with fusion on. Where a program is expressions on large arrays the gain was fusion's: the cavity flow 2.80x, the weather model's horizontal diffusion 2.36x, the ocean model's equation of state 1.74x. [The next chapter]([[ root ]]report/fusion-results/) has the four-way timings program by program, how the shapes were mined, what the measurements show, and what is left that no fusion could take.
