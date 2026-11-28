<p class="lede">The previous chapter built the machinery: expressions read off the graph, patterns read off the kernels' own specs, programs of kernel calls run a block at a time, every result NumPy's bit for bit. This chapter asks what it bought. Twenty-six programs were run four ways, with and without the graph, with and without the package's kernels, and every millisecond of the fused runs was then accounted for: what fusion took, what it could still take, and what no fusion of any kind could touch.</p>

::: note A prototype, measured on one machine
Everything here is from [docs/lazy-fusion-design.md](repo:docs/lazy-fusion-design.md) and the prototype in [prototypes/lazy/](repo:prototypes/lazy). None of it is in the package. Every timing was taken on an Apple M1 Pro with NumPy 2.5.5 and CPython 3.13, with the layer's nodes written by its C core. The fused kernels' specs are not on `main`, so the judge's runs of them are `--unlocked`. The same was timed on x86-64, and [the end of the chapter](#another-processor) has what it found. How fusion works is in [the fusion chapter]([[ root ]]report/fusion/), and the lazy layer it runs on in [the lazy chapter]([[ root ]]report/lazy/).
:::

## How it was measured

A program is run four ways. Each way changes one thing from the one before it, so the difference between two neighbours is the effect of that one thing.

| Way | What runs | What it says |
|---|---|---|
| **NumPy** | The program as it is, `import numpy as np`, nothing installed | The baseline every other way is measured against |
| **Installed** | The same eager program after `hotkernels.install()`, one call at a time, no graph | What the package gives today |
| **Lazy** | The program with the lazy stand-in for `np`: every operation a node, computed one node at a time with the call the program wrote | What writing the graph costs, with nothing gained from it |
| **Fused** | The lazy run with `fuse.enable()`: expressions computed by programs of kernels, small calls made at once | What the graph is for |

**`hotkernels.install()` is in effect for every way but the first.** That is how the package is used: a call it has a kernel for, such as `x[mask]` or `np.histogram`, is served by that kernel whether the program is eager or lazy. `fuse.enable()` installs the package's single-operation kernels unless it is told not to. So two kinds of kernel are at work in a fused run: the package's, one call each, and the fused ones, one expression each.

**The headline figure is the fused run with the kernels installed, against plain NumPy.** Beside it stands the eager program with the same kernels, against plain NumPy. The ratio of the two, the fused run against the installed one, is what the graph adds to what the package already gives. The lazy and the fused run were also timed with nothing installed, to say what the graph and its kernels do by themselves; that is not how the layer would be used, and it is the secondary figure here.

The details of the run:

- **The results are compared bit for bit before anything is printed.** The installed, lazy and fused results of every program must be identical, or the tool stops.
- **A time is the quickest of 5 runs,** in milliseconds, from the start of the program to its last result.
- **The ways were timed one after another,** NumPy's five runs first. The tool now times them in turn, a round of every way at a time, because the way timed first can be the slowest by several percent on a program of large temporaries ([issue 323](https://github.com/sofeikov/hotkernels/issues/323)). The tables of this chapter from the M1 Pro were not taken again; the x86-64 figures at [the end](#another-processor) were.
- **The machine was not idle.** The load average was 1.5 to 3.1, so the figures are rough to a few percent, and a program's own NumPy time moved by up to 10% between runs of the table. So the summaries below count two ways within 3% of each other as level.
- **A lazy run's time is split in two, a fused run's in three.** *Build* is everything outside the layer's computing functions: the program's own Python, writing the nodes, and whatever is made at once. *Plan* is the fused run's planning: finding the expressions and choosing the kernels. *Compute* is the time inside the computing functions, less the planning: NumPy's calls or the kernels'.

The tool is [prototypes/lazy/programs/fusion_time.py](repo:prototypes/lazy/programs/fusion_time.py):

```bash
uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --size large --plain
```

## Twenty-six programs

The corpus, in [prototypes/lazy/programs/corpus.py](repo:prototypes/lazy/programs/corpus.py), is every program this repository times with and without `hotkernels.install()`, and six more added for this work. Each runs with `import numpy as np` or with the stand-in, and nothing else in it changes.

| Programs | What they are | From |
|---|---|---|
| `regression`, `monitor`, `attention` | A linear regression, the same recording its loss as it goes, a transformer block in float32 | Examples written for the lazy layer |
| `perceptron`, `char_rnn`, `particle_filter`, `veros_eos`, `veros_isoneutral`, `veros_tke` | A two-layer perceptron, Karpathy's `min-char-rnn.py`, a bootstrap particle filter, three routines of the Veros ocean model | The four applications in `delivery/apps/` that are NumPy code |
| `cavity_flow`, `hdiff`, `black_scholes` | A lid-driven cavity flow and a weather model's horizontal diffusion from npbench, Black-Scholes from dpbench | Published benchmarks, copied unchanged into [vendor/](repo:prototypes/lazy/programs/vendor/README.md) |
| `trimesh_mesh`, `sklearn_metrics`, `empyrical_stats`, `skimage_pipeline`, `pandas_sales`, `astropy_timeseries` | Six programs that are calls into a library | The other six applications in `delivery/apps/` |
| `example`, `flame`, `azimint_naive`, `azimint_hist`, `mandelbrot1`, `mandelbrot2`, `make_decision`, `pandas_filter`, `equalize_hist` | This repository's example and its fractal flame, five published benchmarks, and two calls into a library | The nine workloads in `delivery/workloads/` |

Each program has two sizes. At the **large** size its arrays are large enough to leave the caches; at the **small** size they have hundreds to tens of thousands of elements.

Three things about the corpus are worth knowing before reading its results.

- **A program that is calls into a library changes its one line like any other.** The `np` of the module that calls pandas or scikit-learn is the stand-in, and the arrays it hands over are lazy. The library has its own `numpy` and asks for `ndarray`s, so each array is computed where it is handed over, and what the library does is in no graph. These programs say what the layer costs a program it can do nothing for.
- **One library does not take a lazy array.** scikit-image's `regionprops_table` asks `np.isscalar` of the mean of an image it was given. For a lazy image that mean is a lazy array without dimensions, not a number, and the library fails. `skimage_pipeline` is therefore handed plain arrays, and nothing of it is deferred.
- **One program needed the layer to learn something.** `mandelbrot2` assigns to `x.shape`, which a lazy array now accepts.

## Where NumPy's time goes

Before a kernel was written, the corpus was **mined**: [prototypes/lazy/programs/mine.py](repo:prototypes/lazy/programs/mine.py) runs each program lazily, one node at a time, and before every computation reads the expressions out of the graph and times every call. Each expression then has a measured weight: the time NumPy spent on the operations in it.

- **It is the graph that is mined, not the source.** The graph is the program after Python has run it: loops unrolled, numbers worked out, views resolved. This line of the cavity flow has nine arithmetic operators, and in the graph it is one multiplication of an array by a number, because everything before `b` is arithmetic on Python floats:

  ```python
  dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
  ```
- **The weight is time, not count.** A shape that occurs a thousand times on arrays of a hundred elements weighs less than one that occurs once on ten million.
- **Mining says which shapes occur and what they cost, not that they may be fused.** Whether a tree may be one kernel is the rule of what an expression may contain, and the judge's check of each kernel against NumPy.

The first thing mining gives is where each program's time goes. An operation counts as **in an expression** when it is one a fused kernel could make and sits in a tree of two or more such operations; **simple, alone** is the same operations with nothing next to them to fuse with; **other element by element** is a call into a mathematics library, such as `np.exp`, or arithmetic on another element type than float64.

<figure class="widget">
<div data-widget="fusionshapes" data-view="shares"></div>
<figcaption>Where NumPy's time goes in the seventeen programs whose NumPy code is their own, at the large size, as shares of the time spent computing nodes. Sorted by the share in expressions, which is what fusion can work on. Pick a program for its figures. From <code>mine.py --size large</code> on the M1 Pro, as the design document gives it.</figcaption>
</figure>

The figure predicts most of what follows.

- **A program made of arithmetic is nearly all expression:** 94% of the horizontal diffusion's time and of the cavity flow's, and 77% to 86% of the three Veros routines'. These are where fusion has something to take.
- **A program made of calls, masks or small arrays is not.** 72% of Black-Scholes is `erf`, `log` and `exp`, which a kernel cannot make. 78% of the naive azimuthal integration is comparisons and the masks made of them. The two Mandelbrot programs are complex arithmetic and selections by a mask, with no expression at all.

## Which shapes to write

An expression of the corpus has between two and 153 operations. Writing a kernel for each whole expression would cover the corpus and nothing else, so the miner cuts every expression into the smaller trees it contains, up to six operations, and weighs each by the time of the expressions it occurs in. The library is then chosen **greedily, one shape at a time**. The miner counts an expression of *k* operations as *k* passes of equal cost, NumPy's measured time for it spread evenly over them, and an expression covered by shapes as one pass for each shape and for each operation left over. The next shape is the one that saves the most of NumPy's time in expressions, counted that way, given the shapes already chosen ([mine.py](repo:prototypes/lazy/programs/mine.py), `choose`).

A shape is written as its pattern's name, the tree in prefix form: `add(V0,mul(S0,V1))` is `V0 + S0 * V1`, where a `V` is an array as long as the result and an `S` is one number for the whole run. The [fusion chapter]([[ root ]]report/fusion/) has how a name is read off a graph.

<figure class="widget">
<div data-widget="fusionshapes" data-view="library" data-shape="1"></div>
<figcaption>The first twenty-four shapes the miner chose, in order. Each bar is what one more shape adds; the line is the share of NumPy's time in expressions that the library saves so far, each operation counted as one pass. Pick a bar for the shape and the NumPy expression it stands for; pick a program to see which shapes it has. The sixtieth shape is drawn by itself because the document lists only the first twenty-four.</figcaption>
</figure>

- **The tail is long.** Twenty-two shapes save half of NumPy's time in expressions, counted that way, and sixty save 61%; the sixtieth still adds a tenth of a percent. Among the twenty-four listed, every shape of five or six operations serves one program, and every shape that five programs or more share has two operations.
- **That is why every step has a kernel.** No library of whole shapes covers a tail that long, and a pass left to NumPy between two kernels is a pass over memory. So the library has a kernel for each of the eighteen single operations as well, and a program of kernels can always finish an expression a step at a time inside the cache. With that, what a missing shape costs is a step within the cache, not a pass.

The library that came of it has eighty kernels: fourteen written by hand, and sixty-six that [make_kernel.py](repo:prototypes/lazy/make_kernel.py) writes from a shape's name (the eighteen single operations, forty-six mined shapes of two to six operations, and the two halves of an advection flux). Sixty-nine of the eighty ran over the corpus, and the single operations are among the most called: `add(S0,V0)` 67 times, `mul(S0,V0)` 63, `sub(V0,S0)` 56. Each of those is a step inside a block where it would otherwise be a pass of NumPy's between two kernels.

## One kernel alone

Before whole programs, the fourteen kernels written by hand were timed alone by the judge, each against its NumPy expression called from Python, as a program writes it. The speedup is what fusing that one expression gains at that size, the call from Python included.

<figure class="widget">
<div data-widget="fusionkernels"></div>
<figcaption>How many times faster each hand-written kernel computes its expression than NumPy's passes do, at four sizes, coloured by the number of operations. Pick a kernel for what it computes and how it compares with the compiled C reference. From <code>uv run judge run &lt;kernel&gt; --unlocked</code> on the M1 Pro.</figcaption>
</figure>

- **Two operations in one pass are worth 1.4x to 2x on arrays that fit no cache, not 2x.** Out there, NumPy and the kernel both wait for memory, so what counts is arrays moved, not operations. Count them for `a + s * b`: NumPy reads `b` and writes a temporary, then reads `a` and the temporary and writes the result, five arrays moved; the kernel reads `a` and `b` and writes the result, three. For `a + b * c` it is six against four. That is why the kernel with a number in place of an array gains more, 1.54x against 1.36x at 16.8 million elements. (The counting is this chapter's, not a measurement; the two speedups are the judge's.)
- **More operations gain more, but less than their number.** Eight operations in one pass, the cavity flow's pressure iteration, are 2.7x; five are 1.7x to 1.9x. The stencil kernels read five and six arrays, and reading is what they wait for.
- **At 4,096 elements the figure is mostly calls.** Three to seven times, of which a good part is what each call into NumPy from Python costs.
- **The assembly is level with the compiled C, with three exceptions.** Clang vectorises these loops as written, and beyond the first-level cache both wait for memory. Loads with a non-temporal hint put the assembly 1.05x to 1.2x ahead at 262,144 to 1,048,576 elements in the kernels that read two or three arrays, which is the 2% to 5% in the last column. The flux limiter is 6.5x ahead because its C reference, a comparison feeding a selection, compiles to a loop of one element at a time; the sum of squares is 1.5x ahead of a reference the compiler vectorises only in part.

## Every program, four ways

Here are the four ways side by side, for every program, against plain NumPy.

<figure class="widget">
<div data-widget="fusiontimes" data-size="large" data-program="cavity_flow"></div>
<figcaption>The twenty-six programs, each against plain NumPy on a logarithmic scale. The dot is the fused run with the kernels installed, the ring the eager program with the same kernels, and the segment between them what the graph adds. Pick a program for its four times, how each lazy run splits into build, plan and compute, and what its graphs were. Switch to small arrays to see where the graph costs more than it saves. From <code>fusion_time.py</code> on the M1 Pro, the quickest of 5 runs.</figcaption>
</figure>

**All twenty-six programs run within 1% of NumPy's speed or faster, up to 5.1x.** Where a program is expressions on large arrays, the gain is fusion's: the cavity flow 2.80x, the horizontal diffusion 2.36x, the equation of state 1.74x, the particle filter 1.43x, isoneutral mixing 1.15x, the turbulent kinetic energy 1.13x, the regression 1.10x. Where a program is a library's work or single calls on large arrays, the gain is the installed kernels', up to 5.1x, and the lazy run keeps it to within 4%.

**Against the eager program with the same kernels, the fused run is ahead on seven programs,** by 1.10x to 2.79x, within 3% on fifteen, and behind on four: three on small arrays, at 0.73x, 0.87x and 0.91x, and the example at 0.96x.

## What the measurements show

### The two kinds of kernel add up

Six of the seven programs that fusion speeds up gain 6% or nothing from `install()` eagerly, because no kernel of the package is in their expressions, and their fused runs are within 2% with the kernels and without. The programs that are a library's work or single calls gain 1.07x to 5.3x from `install()` eagerly and nothing from fusion, and their fused runs keep the kernels' gain. Neither kind gets in the other's way.

The particle filter has both. It takes 358.1 ms on NumPy, 320.6 ms eagerly with the kernels, 286.6 ms fused without them, and 250.3 ms with both.

### Where the fused run is behind, it is what a call made at once costs

With fusion on, a call that gives fewer than 32,768 elements is not written down: it is NumPy's call, made at that line, and its result is a lazy array that holds NumPy's value (the [fusion chapter]([[ root ]]report/fusion/) has the rule). That is right for small arrays, where a node costs more than the call, but the lazy array around the value still costs something: 26 to 47 ns over NumPy's own for an operator, 55 ns for a view, 90 to 135 ns for a function called through the stand-in.

`make_decision`, the recurrent network and the perceptron make 410,000, 28,000 and 7,900 calls at once and write no node at all. Their fused runs take 116.8 ms, 32.7 ms and 27.7 ms, against 85.3 ms, 28.6 ms and 25.3 ms for the eager program with the kernels. The example is 4% behind, 0.4 ms, and the fractal flame 3%, 3 ms.

The naive azimuthal integration was 4% behind too, 7 ms, and that one was found and fixed. Most of it was arithmetic on a number that came out of an array, `rmax * i / npt`, which went through the operator's ufunc at 600 ns an operation where the eager program's NumPy scalar does it in 50 ns. The layer now uses the scalar's own operator, and the program runs level.

### A kernel the lazy run did not reach

A loss of this kind shows in the same column. The second Mandelbrot program takes its grids from `np.mgrid`, which gave plain `ndarray`s, and selects from them by a lazy mask. That is NumPy's own indexing, so the package's kernel for `x[mask]` was never asked: the fused run took 172 ms, the same as with nothing installed, where the eager program with the kernels took 148 ms. The stand-in now makes the arrays of `np.mgrid`, `np.ogrid`, `np.r_` and `np.c_` lazy, and the fused run is level with the eager one (147.1 ms against 145.8 ms in the table).

After that, the package's own count of the calls its kernels served was compared, eager run against fused, for eleven other programs. None loses a kernel that way: the counts differ only by the layer's dry runs on one element, by calls a fused kernel made in a package kernel's place, and, for scikit-learn, by the library's own different road for an array type it does not know.

### One pass for a whole expression is worth what its passes were

The cavity flow shows it most plainly. Its node-by-node run writes 4.1 GB of new arrays and computes in 162 ms; the fused run writes 0.3 GB and computes in 45 ms. Its 560 kernel calls make 260 programs, and 2,980 of its 6,900 nodes never have an array of their own. The equation of state writes 0.7 GB instead of 3.1 GB and computes in 73 ms instead of 116 ms: 153 operations in 64 kernel calls, all on blocks. In the figure above, the last number under a program is this, the memory written node by node and fused.

### Writes made where the array is

The cavity flow's other half is its writes. It makes 1,220 writes into slices and boundary rows of its grids, and every one is made in the array's own memory: in the fused run 1,196 by their programs of kernels and 24 at once. Made as copies, which is what a deferred write is defined as, they were 54 ms.

### Rows of two and four elements

The particle filter's table is 100,000 rows of four numbers, and NumPy makes `particles - mean`, `centred * weights[:, None]` and `particles[:, :2] += particles[:, 2:] * dt` one row at a time: 544 µs, 294 µs and 1,256 µs, where the same arithmetic in one run takes 101 µs, 101 µs and 82 µs. The first is now one kernel's call, the other two a loop of the layer's own over the rows, and the expressions on the table's columns are gathered into blocks. With the rows left to NumPy, the fused run was 2% ahead of NumPy with nothing installed; it is 25% ahead. Nineteen of its writes into the table still go to a copy, because an earlier reader of the table still waits.

### Arrays that step

The two Veros routines keep their state with a time level as its last axis, so `tke[:, :, :, tau]` is every third element: an array that **steps** over memory rather than lying in one run. A kernel reads arrays in one run, so such an array has to be gathered into one first, and the copy costs a pass. The layer gathers where the expression is long enough to pay for it. The kinetic energy's advection flux, seventeen operations with its Courant number on three such arrays, is two kernels and one copy of each array, and the routine runs in 88 ms where it ran in 93 ms. Eleven of its 38 programs and 9 of isoneutral mixing's 121 are still given up: one to five operations, where the copies cost what the passes save. [What is left](#what-is-left) measures what that leaves.

### Planning costs little

Planning is 5% of the cavity flow's fused run, 2.8 ms of 55 ms for 6,900 nodes, with the expressions read by the C core; read in Python it was 15 ms of 69 ms. For every other program it is about 1% or less.

### A library is a wall

Calls into a mathematics library are the wall for Black-Scholes and the transformer block: `erf`, `log` and `exp` are 86% of the first, `tanh` and float32 arithmetic most of the second. And a library handed a lazy array may take longer over it: scikit-learn's metrics are 2% slower on lazy arrays than on `ndarray`s with the kernels installed, and 7% with nothing installed, whatever the layer does with the seven operations of its own.

### At the small size

Switch the figure to small arrays. The horizontal diffusion, the equation of state and isoneutral mixing still gain from fusion, 2.01x, 1.13x and 1.07x against NumPy, and the programs that are a library's work or single calls keep the kernels' gain, up to 5.1x. Six programs are behind NumPy, the transformer block furthest at 0.80x, and against the eager program with the kernels the fused run is from 0.72x to 1.91x. At these sizes most calls are small, and each pays what a call made at once costs.

### How long a plan lives matters

A reference kept past a computation costs a copy or a page fault where NumPy would have reused memory. A cycle of references in a planner cost 46,000 page faults a run, and a record of the part written into, kept by a group that had been given up, cost a copy of the array for each of 21 writes. Both were fixed, and both are tested for.

## What is left

The rest of the chapter answers one question with measurements: what could fusing by pattern still gain on these twenty-six programs, run as the package is used? The answer is given against a **floor**, defined next: the time a program would take if every pass over memory and every call that a fusion could remove were gone. In short, four programs already run at or below their floor; fifteen have 10% at most to gain from any fusion at all; four are on arrays below the limit of 32,768 elements, under which nothing is written down; and three have something left on large arrays, for which the sections below say in which operations, what a loop written for exactly that could do at best, and what was tried.

### The floor: what no fusion could remove

Fusion removes two things from an operation on arrays: the pass over memory that reads its operands and writes its result, and the call into NumPy. It cannot remove the arithmetic. So every operation has a part no fusion touches, and it can be measured: the same NumPy loop on arrays that fit the first-level cache, per element, with the cost of the call taken off.

[prototypes/lazy/programs/limit.py](repo:prototypes/lazy/programs/limit.py) does that for every program, with the kernels installed. It runs the program lazily with every call timed, puts each operation in a class by what a kernel would need to make it, and times each different operation in the cache on pieces of the operands the program itself gave it, because what `erf` or `exp` costs depends on its argument. The **floor** of a program is its installed eager time less everything that could be removed from its element-by-element operations and its reductions over every element: what a fusion of all of them, of any kind, at no cost of its own, would leave.

```bash
uv run --group apps --group workloads python -m prototypes.lazy.programs.limit --size large
```

<figure class="widget">
<div data-widget="fusionfloor" data-view="floor" data-program="cavity_flow"></div>
<figcaption>Each program's times as shares of its time on plain NumPy. The bar is the eager program with the kernels installed, the black tick the floor, and the hatched part between them what a perfect fusion of everything could remove. The dot is the fused run. A dot left of the tick beat the floor. Pick a program for its figures. From <code>limit.py --size large</code> on the M1 Pro, a separate run from the four-ways table, so its times differ from that table's by a few percent.</figcaption>
</figure>

- **The floor is not a floor for a kernel.** It is what NumPy's own loops would take with no pass over memory and no call. A kernel makes several operations for each load and store, which NumPy's loops cannot, so the cavity flow, the horizontal diffusion and the equation of state run below it: the cavity flow's kernels took 37.2 ms for arithmetic that NumPy's loops need 106.6 ms for in the cache. The particle filter is at it.
- **Fifteen programs have 10% at most to gain by the floor.** Their time is outside the graph, in a library or in calls the layer does not defer, or in a mathematics library, matrix products, or moving elements: a selection by a mask or by positions, a write through one.
- **Four programs are on arrays below the limit.** `make_decision`, on arrays of 32 elements, has no floor that means anything: what the tool finds removable there is the call into NumPy, which is nearly all such an operation costs. The floors of the recurrent network, the perceptron and the fractal flame, on arrays of 100 to 25,000 elements, are low for the same reason; [Small arrays](#small-arrays-the-graph-costs-more-than-the-program) below has why fusing them loses.
- **The regression's figures move with its matrix products,** which are 26 ms of its 38 ms. Its fused run read 1.37x against NumPy in an earlier run of the tool, 1.19x in this one, and 1.10x in the four-ways table.
- **What is removable moves between runs** for an operation whose cost depends on its values: the complex arithmetic of the first Mandelbrot program came out at 11 to 12 ms in four runs and 20 ms in another.

### The fused run in three parts

Since the kernels already run below the floor, the floor says little about a run that has kernels in it. So the tool also splits each fused run itself into three parts: the time inside the kernels' programs; the calls NumPy still makes for operations a kernel could make; and everything else. If each of those NumPy calls cost only what its loop costs in the cache, the difference would be the most a fusion of everything not yet fused could still take off, at no cost of its own.

<figure class="widget">
<div data-widget="fusionfloor" data-view="parts" data-program="veros_isoneutral"></div>
<figcaption>Each fused run, with the kernels installed, split into the time inside its kernels' programs, the calls NumPy still makes that a kernel could make (green: the part that would go if each cost its loop in the cache), and the rest of the run. The number at the right is that green part as a share of the run. From the last table of <code>limit.py --size large</code>.</figcaption>
</figure>

What could still go is 11.0 ms for isoneutral mixing and 5.7 ms for the kinetic energy, 6% of each; 2.6 ms for the transformer block, 2.5 ms for the equation of state, 1.8 ms for Black-Scholes; and under half a percent of the run for the other twenty-one.

### Where the rest is, class by class

| What is left | In which programs | How much | Why a pattern does not take it |
|---|---|---|---|
| Calls into the platform's mathematics library: `exp`, `log`, `erf`, `tanh`, `arctan2`, `%`, `hypot` | Black-Scholes, isoneutral mixing, the particle filter, the fractal flame, the transformer block | 226 ms over the five, of which the tool finds 7.8 ms removable | It costs the same in the cache as out of it. A kernel calls nothing, and a second implementation would not have the library's bits |
| Calls on small arrays | `make_decision`, the recurrent network, the perceptron | All of what they are behind the eager program: 32 ms of 117 ms, 4.1 ms of 32.7 ms, 2.4 ms of 27.7 ms | Nothing is written down for them, so there is nothing to match. Writing a node costs more than the call it would save |
| Arrays that step, in rows of 25 or 26, in expressions of one to five operations | The two Veros routines | 11 of the kinetic energy's 38 programs and 9 of isoneutral mixing's 121 given up: at most 5.7 ms and 11.0 ms | A copy for each such array costs what the passes save, and a loop that reads them where they lie is 1.14 times NumPy at best |
| Complex arithmetic | The first Mandelbrot program | 11 to 20 ms removable of 81 ms | A loop written for it gains nothing; see below |
| Comparisons and selections standing alone | The kinetic energy, the naive azimuthal integration, the horizontal diffusion | 2.5 ms of 10.2 ms, 1.4 ms of 151 ms, 1.9 ms of 13.3 ms | A mask is a byte an element, which the float64 kernels neither read nor write, and its pass moves an eighth of the memory |
| float32 | The transformer block | 1.1 ms of 2.0 ms | No kernel is written for float32, and the time is in the matrix products and `tanh` |

### The best a kernel for it could do

The tool's figure is what would go if a fusion cost nothing. What a kernel could really do was found by writing the loop: one C loop for exactly the thing that is left, compiled with floating-point contraction off so that every operation is rounded by itself, timed against NumPy on the program's own layout and compared bit for bit. None of these loops is in the code; each says whether its kernels would be worth writing.

| What the loop is for | NumPy | The loop | What it says |
|---|---|---|---|
| `Z[I]**2 + C[I]` on complex128 as one pass, in the first Mandelbrot program | 149.1 to 149.6 ms | 149.5 ms | Nothing to gain: the time is in `abs`, the masks and the selections. And the loop's bits are not NumPy's, whose complex power is another formula |
| One loop for each whole expression of the equation of state, in place of a program of kernels | 25.9 ms by the kernels' programs | 29.2 ms | Larger kernels are not better. A loop over a dozen arrays reads them all from memory at once, where a program reads two or three at a step and keeps the rest in the cache |
| The same for Black-Scholes, the regression and isoneutral mixing | 4.8, 1.9 and 12.6 ms by the kernels' programs | 3.2, 1.9 and 8.0 ms | 1.6 ms of 87, nothing, and 4.6 ms of 200: under what a run moves by |
| The kinetic energy's advection flux, thirteen operations, its three stepping arrays read where they lie | 4,439 µs | 1,705 µs | 2.6 times, and the layer now has most of it: two kernels for the flux's halves and each array gathered once make it in 2.85 ms for NumPy's 4.64 ms |
| `(a - b) * m` with `a` and `b` at a step, the commonest expression the Veros routines give up | 1,151 µs | 1,006 µs | 1.14 times: about 2 ms in the kinetic energy and 3 ms in isoneutral mixing, 2% of each, if every expression they give up were this one |
| `Ai[..., ip, kr] = taper * sx * mask[...]`, a product stored into every fourth element, sixteen lines of isoneutral mixing | 1,537 µs | 985 µs | 1.56 times, and the layer has it: a loop of its own makes the statement in 1,001 µs, and the routine runs 5 to 9 ms sooner |
| An array that steps copied into a run, by itself | | 458 µs | What a copy costs before any arithmetic: a third of NumPy's two passes over it. It is why gathering loses on one kernel and gains little on thirteen operations |
| `np.where(x > 0, a, b)` with the comparison inside | 842 µs | 545 µs | 1.5 times on what the tool puts at 2.5 ms in the kinetic energy: 1 ms of 93 |

### What was built and did not gain

Each of these was built in the layer, timed against the build without it on the same program, and taken out again. These comparisons were run with nothing installed, since each compares two builds of the layer, not the layer against NumPy.

- **A mathematics library called a block at a time.** Black-Scholes with NumPy's own loops for `erf`, `log` and `exp` on blocks of 4,096 elements, into buffers used again: 84.5 to 87.7 ms against 86.2 ms as written. The arithmetic between the library calls is 13 ms of 87, and the kernels have it already.
- **Gathering every array that steps, for every program that reads one.** The kinetic energy took 99.8 to 100.8 ms where it took 93.9 to 95.7 ms: a program of one kernel that reads two such arrays makes two copies to save one pass. That is where the rule of operations against copies comes from.
- **A program for each short row, in place of declining,** for rows of 25 at more than one pitch: isoneutral mixing 194 to 215 ms against 196 to 201 ms, the kinetic energy 94 to 99 ms against 97 ms. On the two routines it cannot be told from nothing.
- **A call of the kernel for each row, stored at the result's step afterwards.** 1,632 µs a statement for NumPy's 1,553 µs: the call for a row of 25 costs what NumPy's second pass did. The same statement as a loop with no call in it is what the layer kept.
- **Rows of 16 to 31 elements copied into blocks,** for any program with an array that steps: one routine's gain was the other's loss, both within what a run moves by.
- **Rows of 25 at more than one pitch copied into blocks,** in place of a call of every step for each row: the four programs of isoneutral mixing it applied to went from 1.85 ns an element to 2.35 ns. They read five arrays and write one, 48 bytes an element; at 1.85 ns they are waiting on memory, and a copy is one more pass.
- **Large views left pending,** so that a write through one could be fused with what it stores: the particle filter the same, the cavity flow 54.1 to 54.7 ms against 52.0 to 52.1 ms, for the nodes its thousands of views then are.

### Small arrays: the graph costs more than the program

Four programs of the corpus are behind the eager program with the same kernels, and they are the four on arrays below the limit at which the layer writes anything down. Their fused run has no graph: it is the eager program with a lazy array around each value.

| Program | Elements to an array | Eager, kernels installed | Writing its graph, nothing computed | Fused run | Fusion forced on every array | Forced from 512 elements |
|---|---|---|---|---|---|---|
| `make_decision` | 32 | 85.3 ms | 1039.2 ms | 116.8 ms | 862 ms | 120 ms |
| The recurrent network | 100 to 10,000 | 28.6 ms | 45.2 ms | 32.7 ms | 163 ms | 103 ms |
| The perceptron | 8,000 | 25.3 ms | 24.2 ms | 27.7 ms | 125 ms | 99 ms |
| The fractal flame | 25,000 | 88.2 ms | 86.3 ms | 91.0 ms | 194 ms | 194 ms |

- **What they are behind by is the lazy array around each value.** Statement by statement in `make_decision`'s loop, on 32 elements: `e - symbols` 499 ns for NumPy's 459 ns, `abs(d)` 376 for 337, `a ** 2` 224 for 179, `np.argmin(s)` 208 for 95, `symbols[i]` 130 for 84, `out[i] = v` 121 for 95; the body of the loop 1,555 ns for 1,166 ns.
- **A pattern needs the graph, and the graph alone costs more than the program.** The fourth column is the time to write these programs' operations down as nodes, with nothing computed. For three of the four it is more than the eager program takes for everything, and for the fourth nearly all of it. Whatever computed those graphs, at any speed, would start behind.
- **Computing them is slower still, and that was measured.** With every operation written down and planned whatever its size, the programs take two to seven times what they take with calls made at once. A kernel saves a pass over memory, and an array of a few thousand elements is in the cache either way; what is left to save is NumPy's call, about 0.1 µs, against a node, a plan and a program of kernel calls.
- **What would take them is not a pattern.** It is a lazy array that costs less to make, which is the layer's own work and has a floor at an object made and freed for each operation; or compiling the body of the loop once from a trace of it, which is another design.

### What a pattern could still take

A pattern here is a tree of operations found in a graph and computed by code written beforehand: a kernel, a program of kernels, a loop of the layer's own. It needs three things. The operations have to be written down as a graph. Among them there has to be a tree that is element by element on float64. And what replaces NumPy's calls has to cost less than they do. Every millisecond of a fused run is in one of four places, and in each of them one of the three fails or the gain has a measured bound.

1. **Inside the kernels' programs.** The pattern has been applied. A larger kernel was slower or level; other walks of short rows were built and dropped; the hand-written kernels are level with compiled C beyond the first-level cache.
2. **Calls NumPy still makes for operations a kernel could make.** The green part of the figure above: more than a tenth of a millisecond in six programs, nothing or next to it in the other twenty. In the Veros routines it is short expressions on arrays that step, where the best loop is 1.14 times NumPy, about 1 ms of each routine; taking it would need every kernel written a second time with a step for each array, and the judge to describe such arrays. In the equation of state and Black-Scholes it is single operations between expressions, which a kernel makes no faster than NumPy; in the transformer block, float32 and `tanh`.
3. **The rest of the run.** The program's own Python, a library's code, matrix products, selections and writes through masks and positions, sorting, random draws, and the layer. None of it is a tree of element-by-element operations on float64. The package's own kernels take what they can of it, with or without a graph.
4. **Programs that write no graph.** The four above, where writing the graph costs more than the program.

::: takeaway On these twenty-six programs, a further pattern has nothing left that a measurement can see
That is a statement about one corpus on one machine, and it is weaker than it sounds in two ways: other programs, and another processor.
:::

### The programs of issue 298

[Issue 298](https://github.com/sofeikov/hotkernels/issues/298) lists 21 programs that come out as one graph each when every operation is deferred: stencils, convolutions and rank updates written as Python loops, filters, and chains of arithmetic. One is in the corpus already. The other twenty were run as the corpus is, at the size the issue names, with their inputs copied before each run since most of them write into what they are given.

| Program | What it is | NumPy | Eager, installed | Fused, installed | Against NumPy | Against installed | Calls made at once |
|---|---|---|---|---|---|---|---|
| `jacobi_2d` | A stencil of five slices, 80 steps on 350 x 350 | 41.02 ms | 40.27 ms | 17.79 ms | 2.31x | 2.26x | 0 |
| `heat_3d` | A stencil in three dimensions, 50 steps on 40 x 40 x 40 | 30.22 ms | 30.07 ms | 16.39 ms | 1.84x | 1.83x | 0 |
| `slowparts` | Products of `tanh` terms slice by slice, arrays of 10 to 200 | 1.08 ms | 0.63 ms | 0.86 ms | 1.26x | 0.73x | 2,200 |
| `adi` | An implicit solver, a column at a time | 64.40 ms | 50.92 ms | 60.00 ms | 1.07x | 0.85x | 142,720 |
| `syr2k` | A rank update, a row at a time | 42.41 ms | 31.26 ms | 39.90 ms | 1.06x | 0.78x | 107,940 |
| `jacobi_1d` | A stencil of three slices, 3,000 steps on 12,000 | 102.98 ms | 102.79 ms | 101.36 ms | 1.02x | 1.01x | 23,992 |
| `ex088` | The Game of Life on 50 x 50 whole numbers | 2.57 ms | 2.41 ms | 2.55 ms | 1.01x | 0.95x | 1,800 |
| `covariance` | A loop over columns, a matrix product in each | 393.50 ms | 389.14 ms | 390.86 ms | 1.01x | 1.00x | 5,602 |
| `diffusion` | A stencil written into a second array, 128 x 128 | 7.72 ms | 7.69 ms | 7.76 ms | 0.99x | 0.99x | 1,200 |
| `correlation` | A loop over columns, a matrix product in each | 391.33 ms | 396.26 ms | 397.05 ms | 0.99x | 1.00x | 4,204 |
| `wdist` | A double loop over columns of 70 elements | 7.99 ms | 5.83 ms | 8.32 ms | 0.96x | 0.70x | 29,400 |
| `conv2d_bias` | A convolution as a Python loop over output positions | 54.18 ms | 54.16 ms | 56.49 ms | 0.96x | 0.96x | 10,801 |
| `multiple_sum` | A sum for each row of 100 elements | 0.90 ms | 0.90 ms | 0.95 ms | 0.95x | 0.95x | 300 |
| `lenet` | Convolutions as Python loops over output positions | 30.73 ms | 31.63 ms | 32.52 ms | 0.94x | 0.97x | 11,171 |
| `resnet` | Convolutions as Python loops over output positions | 34.64 ms | 34.96 ms | 36.87 ms | 0.94x | 0.95x | 7,076 |
| `go_fast` | `trace += np.tanh(a[i, i])` in a loop, then `a + trace` | 14.85 ms | 15.11 ms | 16.44 ms | 0.90x | 0.92x | 12,000 |
| `deriche` | Recursive filters along rows and columns | 36.51 ms | 31.97 ms | 40.51 ms | 0.90x | 0.79x | 39,958 |
| `local_maxima` | One element and one neighbour at a time | 10.90 ms | 11.07 ms | 14.97 ms | 0.73x | 0.74x | 43,200 |
| `syrk` | A rank update, a row at a time | 32.80 ms | 28.45 ms | 46.86 ms | 0.70x | 0.61x | 120,200 |
| `seidel_2d` | A stencil a row at a time, each row from the one before | 43.92 ms | 42.31 ms | 97.39 ms | 0.45x | 0.43x | 548,800 |

- **Two gain, and they are the two that are expressions on large arrays.** The Jacobi stencil in two dimensions and the heat equation in three are covered by kernels the corpus had already asked for.
- **The others are on arrays below the limit,** rows and columns of ten to a few thousand elements or single elements, and write no graph: the fourth place above.
- **They found five things in the layer that the corpus had not,** all in what a program on small arrays costs, and none of them a missing pattern. Each is fixed:
  - *A write into an array the program was handed went through Python:* 3.9 µs a write for NumPy's 0.2 µs. The symmetric rank update took 204 ms for 34 ms and the Gauss-Seidel stencil 1,111 ms for 44 ms. The C core makes such a write now, in 0.3 to 0.4 µs.
  - *A recurrence over the rows of an array copied the array for every row:* the Deriche filter, 1,500 rows, took 11.6 s for NumPy's 37 ms. A view that shares no byte with the row written is no longer counted as a reader of it.
  - *A small call waited for a large array still pending,* and every operation after it with it: the Deriche filter wrote 36,000 nodes that way and took 170 ms. A small call now computes what it reads first: 49 ms.
  - *`x += y` on a number taken out of an array went through the ufunc,* 599 ns for the NumPy scalar's 34 ns.
  - *The set of pending nodes kept the table of its largest moment,* so that after one graph of 44,000 nodes each later write walked 131,072 empty places: the Game of Life took 8.8 ms after another program's graph and 3.1 ms by itself.
- **What is left behind NumPy there is a loop over single elements.** The Gauss-Seidel stencil updates one element at a time and runs at 0.45x; the search for local maxima at 0.73x; the symmetric rank update, on rows of 1 to 200 elements, at 0.70x. A lazy array around a number costs 60 to 100 ns an operation where NumPy's scalar costs 35 to 60 ns. The other seventeen are within 10% of NumPy or ahead, and five of those are further behind the eager program with the kernels, since what the kernels gain on rows of a hundred elements is about what the lazy arrays cost.

### Another processor

Everything above was measured on an Apple M1 Pro. The same was run on x86-64, an Intel Core i5-14600KF under Linux with NumPy 2.5.5, and the details are in [the design's section](repo:docs/lazy-fusion-design.md#another-processor). The figures here are of the second take of its tables: the four ways timed in turn, on one performance core with one BLAS thread, while other work ran on the machine.

- **The kernels there are compiled C, with two exceptions.** The kernel tool writes the same loop in AVX2 for every kernel that computes an expression element by element, and the judge passed seventy-nine of them. Timed against their C references, seventy-six are level (0.96x to 1.02x) and one is slower: clang vectorises those loops as well. Two are 5.5x and 3.5x ahead, `np.sqrt(a)` and `s / np.sqrt(a)`, because C's `sqrt` keeps its `errno` rule and is not vectorised. Only those two are assembly there.
- **The programs gain as on ARM64.** Fused against plain NumPy: the cavity flow 3.09x, the horizontal diffusion 2.31x, the equation of state 1.69x, turbulent kinetic energy 1.38x, isoneutral mixing 1.28x, the particle filter 1.18x. Fusion's gain is the passes it saves, and C saves them as well as assembly. The regression is level, 1.02x, with its matrix products on one thread. Four programs are behind NumPy by 3% or more: the naive azimuthal integration at 0.85x and the second Mandelbrot set at 0.94x, as they are with the kernels installed and no graph, the attention block at 0.93x and the character RNN at 0.97x.
- **The first take of those tables had two figures that were not fusion's.** The regression read 2.26x because its NumPy run, with BLAS on every core, took 64 ms in one table and 138 ms in the next; and turbulent kinetic energy read 1.62x because its NumPy run was timed first, 200 to 205 ms where it takes 177 to 188 ms timed in turn.
- **NumPy is not the same function there.** `np.maximum` of two zeros is its second operand, not the positive zero, so a maximum's operands are matched only in the order the program wrote them. An invalid operation such as `inf - inf` gives a NaN with its sign bit set, unlike `np.nan`, and NumPy's own loops return one or the other of two NaNs depending on where an element falls. The judge now checks that a fused result is a NaN wherever NumPy's is, and not which of two.
- **The layer's limits were not moved.** The block of 4,096 elements and the size below which a call is made at once are still the M1 Pro's. Whether x86-64 wants others is [issue 309](https://github.com/sofeikov/hotkernels/issues/309).

### Ten programs from other codebases

Ten more programs were run on x86-64, each the NumPy code of a published project with the stand-in where its `import numpy as np` stood: perlin-numpy, the convolutional network of the book *Deep Learning from Scratch*, Matplotlib's hill shading, pvlib's solar position, scikit-image's colour difference, scikit-learn's perceptron, SALib's Sobol' analysis, prysm's Zernike polynomials, Astropy's models and python_speech_features. The tables and the reasons are in [the findings](repo:docs/lazy-codebases-findings.md).

- **They gain little: the fused run is 0.88x to 1.14x of NumPy.** Astropy's models 1.14x, prysm 1.10x, scikit-image 1.08x, the convolutional network 1.07x (from `install()`, not from fusion), five level, pvlib 0.97x and scikit-learn's perceptron 0.88x. All give NumPy's bits.
- **A third of their time is a mathematics library's functions:** `cos`, `sin`, `x ** 3`, `%`, `cbrt`. Those end an expression, since a kernel's result has to be the library's bits, and a program that is sums of cosines has little between them.
- **No pattern was missing.** Every expression found on arrays a kernel can read was planned. What was given up was for how the arrays lie: one left transposed by the call that made it, rows of four colour values with one value for each pixel, and a channel of an image or a column of a table, whose elements are not next to each other.
- **`install()` gains on one of the ten,** 1.08x: the other nine make almost no call the package has a kernel for.
- **The layer costs time where the corpus had not shown it.** A program computed node by node is up to 1.4x slower than eagerly, because NumPy reuses the memory of its temporaries and a graph holds each one; and a ufunc called with `out=`, `np.clip` and a function called on a library's plain arrays cost 1 to 8 µs over NumPy's call.

### What is open in the layer, and is not fusion

- **What handing a lazy array to a library costs.** scikit-learn is 2% slower with lazy arrays handed in, with the kernels installed, and the time is not in the layer: the library takes a longer road for an array type it does not know. scikit-image fails on one. Whether a program whose work is a library's should lose anything is open.
- **More of NumPy in the deferred set.** The workloads make calls the layer does not defer: `np.histogram`, `np.bincount`, `np.interp`, `np.isin`, `np.unique`, `np.stack`, `x.astype(...)`, `np.clip` without `out=`. None sits inside an expression the kernels could take, but for `np.clip`, which is a maximum and a minimum, and a cast, which is a kernel nobody wrote.
- **The limit below which a call is made at once.** 32,768 elements is in the middle of a range, 16,384 to 65,536, over which no program of the corpus runs 1% differently, and only two programs have arrays near it. Timed by itself, `(a + b) * a` on 40,000 elements takes 22.9 µs fused for NumPy's 17.7 µs, and is ahead from about 70,000: a computation costs about 6 µs whatever is in it.
- **Reductions along an axis, and a sum of a longer expression.** `sum(sq(V0))` is a kernel of its own; a sum of a longer expression computes the expression into an array first, because NumPy's order of addition is over the whole array, not over blocks. `np.sum(e ** 2, axis=-1)` and `np.max(np.abs(e))` are left to NumPy. The tool finds 6 ms removable in the regression's reductions, and under 2 ms in any other program on large arrays.

What is parked is in four issues:

- [Issue 309](https://github.com/sofeikov/hotkernels/issues/309): whether the limits set on the M1 Pro should move on x86-64.
- [Issue 310](https://github.com/sofeikov/hotkernels/issues/310): what a call on a small array costs through a lazy array, and what could lower it.
- [Issue 311](https://github.com/sofeikov/hotkernels/issues/311): the size at which a short expression is worth fusing.
- [Issue 312](https://github.com/sofeikov/hotkernels/issues/312): the graph page showing what was fused: which operations a program of kernels made together, and which were given up.
