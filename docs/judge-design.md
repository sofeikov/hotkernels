# The judge: design

This is a proposal for review. Nothing here is implemented yet. It details phase 1 of [core-design.md](core-design.md).

## What the judge is

The judge is a command-line tool in this repository that takes one kernel and does two things:

1. **Decides whether it is correct.** Does it match NumPy on every input we can throw at it, without touching memory it does not own? This is an automatic pass or fail.
2. **Measures how fast it is.** How does it compare with the naive C version and with NumPy, and is the difference statistically significant? This is a report. Whether a kernel is a win worth shipping is decided by a person reading it.

A coding agent iterating on a kernel runs the judge after every change. Nobody reviews the assembly line by line, so the judge is the only thing standing between a subtly wrong kernel and a user's data. The correctness side has to be strict, and the timing side has to be honest about its own uncertainty.

The judge is a development tool. It is not part of the `hotkernels` package that users install.

## Commands

All commands run through `uv`, so there is nothing to set up beyond cloning the repository.

| Command | What it does | Time budget |
|---|---|---|
| `uv run judge check <kernel>` | Lint and correctness only | About 10 seconds |
| `uv run judge bench <kernel>` | Timing report only | About 1 minute |
| `uv run judge run <kernel>` | Correctness, then the timing report | About 1.5 minutes |
| `uv run judge compare <kernel> --against <git-ref>` | Is the working copy faster than an earlier version of the same kernel? | About 1 minute |
| `uv run judge selftest` | Proves the judge catches known-bad kernels | About 20 seconds |
| `uv run judge leaderboard` | Regenerates the results table | Seconds |

The exit code reflects correctness only: 0 if the kernel is correct, non-zero if not. Speed never fails a run. `check` is fast on purpose: agents run it on every edit and run the slower commands only once `check` passes.

## Layout

```
judge/                 the tool (Python, plus a small amount of C and assembly)
  fixtures/            kernels used only to test the judge itself
kernels/<name>/        one directory per kernel (the contract from core-design.md)
  spec.py
  naive.c
  <arch>/kernel.s      arm64 or x86_64
  <arch>/attempts.md   what was tried on that architecture
results/               recorded results, committed
.judge/                build output and raw runs, not committed
```

The judge is written in Python because NumPy is the reference for correctness. It compiles kernels by calling `clang` directly with flags it controls. There is no build system in the proof of concept.

## The kernel spec

`spec.py` tells the judge how to call a kernel and what the right answer is.

**This format is provisional.** It is the smallest thing that covers the two fixture kernels. When the survey or a real kernel needs something it cannot express, the format changes. Nothing else in the design depends on its details.

A kernel that returns a single value:

```python
import numpy as np
from judge.spec import Kernel, In, Size

def oracle(x):
    return np.argmax(x)

KERNEL = Kernel(
    name="argmax_f32",
    returns="i64",
    args=[In("x", "f32", len="n"), Size("n")],
    oracle=oracle,
    mode="exact",
    min_n=1,   # np.argmax raises on empty input; the kernel is never called with n = 0
)
```

A kernel that writes a variable amount of output:

```python
import numpy as np
from judge.spec import Kernel, In, Out, Size

def oracle(x, mask):
    return x[mask]

KERNEL = Kernel(
    name="compress_f32",
    returns="usize",   # number of elements written
    args=[
        In("x", "f32", len="n"),
        In("mask", "bool", len="n"),
        Out("out", "f32", capacity="n", used="return"),
        Size("n"),
    ],
    oracle=oracle,
    mode="exact",
)
```

- `oracle` is a Python function over the inputs. Its result is the correct answer.
- `mode` is `"exact"` or a tolerance in units in the last place, as defined in the core design, optionally with the scale the unit is taken at.
- A spec can also supply its own input generators when the standard ones are not enough.

From the signature the judge generates the C glue it needs to call the kernel. Kernel authors never write harness code.

### What a spec can express

The first real kernels needed more than the two fixtures do. Each addition below exists because a surveyed kernel could not be written without it.

| Feature | What it does | Needed by |
|---|---|---|
| `Scalar("lo", "f64", -0.5)` | A scalar argument and its default value. Integers go in the integer argument registers and floats in the floating-point ones. The oracle receives the scalars after the arrays: `oracle(x, lo, hi)` | `clip` (bounds), `histogram` (bin count and range), masked assignment (the value) |
| `Kernel(scalar_sets=[ScalarSet("zero_lo", {"lo": 0.0, "hi": 1.0})])` | Further combinations of scalar values to check, and optionally time. See [Scalar sets](#scalar-sets) | `clip`, `histogram`, masked assignment |
| `In("test", "i64", len=8)` | An input whose length is fixed and does not follow `n` | `isin` (the set of test values) |
| `In("v", "f64", len="needles")` | An input whose length is the value of an integer scalar, so a kernel can have two lengths that vary independently. See [A second length](#a-second-length) | `searchsorted` (the needles), `interp` (the queries) |
| `Out("counts", "i64", capacity=256)` | An output whose size is fixed | `bincount` |
| `Out("counts", "i64", capacity="nbins")` | An output whose size is the value of an integer scalar, so it changes with the scalar set | `histogram` |
| `Out("out", "i64", capacity=Times(3), used="return", scratch=True)` | An output with a whole number of elements per element of `n`. With `scratch=True` it lends a kernel working memory in proportion to its input: the kernel reports the elements that are its result, they are compared with the oracle, and the rest of the buffer is its to use. See [Scratch output](#scratch-output-writing-inside-the-capacity) | `argsort` (the result and the two buffers a radix sort moves its elements between) |
| `In("a", "f64", prepare=np.sort)` | A function applied to every generated input, for kernels defined only on some inputs. Timing inputs that a distribution supplies do not pass through it | `searchsorted` and `interp` (sorted data); `cumfmax_f64` and `fmax_reduce_f64` (no `-0.0` where a zero is the maximum: a domain that leaves out the inputs the binding sends to NumPy, where NumPy's builds disagree) |
| `In("test", "i64", len=8, classes={"padded_3": fn})` | Value classes of one argument alone, tried next to the ones every argument of its type gets, for inputs a caller sends that the general classes do not produce. Each function takes the argument's default random array and returns the array to use. They are checked like any plain class, pairings included, and are not timed | `isin_i64` (test values padded by repeating the first, as the binding pads them) |
| `In("v", "f64", len="needles", draws_from="a")` | One more value class, `from_a`, whose elements are elements of `a` as the kernel receives it. Tried with `a` random and with `a` holding repeated values at every size and in every scalar set. See [Values drawn from another argument](#values-drawn-from-another-argument) | `searchsorted` (needles equal to elements, at every path), `interp` (queries on sample points) |
| `InOut("x", "f64")` | A buffer the kernel modifies in place. It starts as an input and is compared in full with the oracle's result | masked assignment |
| `Kernel(numpy=fn)` | What is timed as NumPy when that is not the oracle. An in-place oracle has to copy its input; the statement a user writes does not | masked assignment |
| `mode={"tolerance_ulp": 4, "scale": fn}` | A tolerance measured at a magnitude the spec computes from the inputs, instead of at the size of the result. See [Comparison](#comparison) | float `cumsum` |
| `Kernel(check_max_n=65537)` | Caps the sizes used for correctness, for an oracle too slow to run on millions of elements per case. Timing still uses the full sweep | `interp` |
| No `arm64/kernel.s` | The kernel is its naive C. `check` verifies the C against NumPy and `run` reports C against NumPy | every kernel before its assembly exists, and kernels shipped as C |

Two limits follow from these and are deliberate:

- **A scalar is tested with the values the spec lists and no others.** The judge has value classes for arrays and none for scalars. A kernel's behaviour for a combination that is not in the default set or in `scalar_sets` is not verified.
- **An in-place buffer is timed on its own output.** After the first repeat the kernel sees the result of the previous one. That is only a fair timing for operations that give the same result when applied twice, which masked assignment is.

### A second length

`np.searchsorted(a, v)` has two lengths that have nothing to do with each other: the sorted array's and the number of needles. A spec has one `Size`. The second length is a scalar that sizes a buffer:

```python
args=[In("a", "f64", prepare=np.sort), In("v", "f64", len="needles"), Out("out", "i64", capacity="needles"),
      Size("n"), Scalar("needles", "usize", 64)],
scalar_sets=[ScalarSet("none", {"needles": 0}), ScalarSet("k17", {"needles": 17}),
             ScalarSet("many", {"needles": 1000}, timed=True)],
```

- **The `Size` is the length that selects the kernel's code path**, here the sorted array's. It gets the whole sweep: every value up to 65, every power of two with its neighbours, and the timing sweep.
- **The other length takes the values the scalar sets give it**, and is checked and timed like any scalar: the default set in full, each further set with the thinner sweep, `timed=True` sets with a timing table of their own. The spec lists the values that matter. For a kernel that handles 16, 64 or 128 needles at once those are the counts either side of each width.
- **A buffer sized by a scalar is placed by that length.** Its guard page sits at its own end, a positional value class (`nan_at@p`) is tried at every index of its own length when that is 65 or fewer, and a case with a length of zero has an empty buffer.
- **The pool of timing inputs is sized by the largest buffer of the call**, so that 1,000 needles against a 16-element array do not get 64 inputs that no longer fit the cache.
- **The oracle receives the scalar** like any other, after the arrays, and can ignore it.

This was chosen over a second `Size` because it adds nothing to the case description, the timing loop or the reports; [two-sizes-design.md](two-sizes-design.md#why-not-a-second-size) has the comparison. What it costs is the first limit above: the second length is verified for the values the spec lists.

Prepared inputs are kept between cases of one size. `prepare=np.sort` used to run for every case, although most cases of a size differ in one argument and share the other's array. The fixed cases therefore come size by size, every scalar set of one size before the next size.

### Values drawn from another argument

A search answers differently on its two sides only for a needle that equals an element of the array. `np.searchsorted(a, v)` and `np.searchsorted(a, v, side='right')` differ there and nowhere else, and `np.interp` picks between the values of repeated sample points there. Value classes are drawn one argument at a time, so random needles against a random array never equal an element. Ties came only from the pairings of plain classes (`ties` with `ties`, `equal` with `equal`), which run in the default scalar set and up to 4,096 elements, and from fresh cases, which stop at 5,000. Both search kernels take a path of their own from 1,048,576 elements, and only with 128 needles or more, which the default set does not have. A mutation run showed a wrong comparison there passes the judge ([issue 48](https://github.com/sofeikov/kernel-arena/issues/48)).

A spec says that one argument's values may be drawn from another's:

```python
args=[In("a", "f64", prepare=np.sort), In("v", "f64", len="needles", draws_from="a"), ...]
```

- **It gives the argument one more value class, `from_<source>`** (here `v=from_a`). Each of its elements is an element of the source as the kernel receives it, after the source's own value class and its `prepare`, at an index drawn at random with replacement. A case description stays reproducible: it names the source's class and the drawn one, and the seed gives the indices.
- **The source is built first**, whatever the order of the arguments: `interp_f64` takes its queries `x` before its sample points `xp`. A source has the same element type as the argument drawn from it, and does not itself draw from another argument. A source with no elements leaves the drawn argument its random data.
- **It is tried wherever the argument's other classes are**: with the source random in the single-argument sweep, against every plain class of the source in the pairings, and in fresh cases.
- **And at every size, in every scalar set, against the source's class of repeated values**: `ties` for floats, `small` for integers. A needle equal to an element with no equal neighbour shows a search's comparison only at the probe that lands on that element. An element in a run of equal ones shows it wherever the run is cut, which is what the first step of a search needs: it compares with one fixed element, `a[p - 1]`, and goes wrong only when elements equal to the needle lie on both sides of it. Above 65,536 elements, where the sweep keeps one class per argument, the drawn class is tried against the random source as well. One memory arrangement per case, rotating, as in the thinner sweep.
- **Prepared arrays are kept for every class of a size**, not only the last one, so that a sorted source is not sorted again for each scalar set.

Where it is used:

| Kernel | Drawn | From |
|---|---|---|
| `searchsorted_f64`, `searchsorted_right_f64` | `v` (needles) | `a` (the sorted array) |
| `interp_f64` | `x` (queries) | `xp` (the sample points) |

What it costs, measured on the Apple M1 Pro with other jobs running beside it (load average 7 to 15), in alternating runs of the judge on `main` and with this change:

| Kernel | Cases | `judge check` | Peak memory |
|---|---|---|---|
| `searchsorted_f64`, `searchsorted_right_f64` | 141,872 to 145,599 (324 of the new ones above 65,536 elements) | 18.2 s to 14.5 s | 618 MB to 890 MB (`searchsorted_f64`) |
| `interp_f64` | 155,063 to 157,062 | 15.4 s to 13.1 s | |

The check got faster although it runs more cases: keeping the sorted arrays of every class of a size, rather than the last one, saves sorting them again for each scalar set, and that was most of the time before. The same arrays are the extra memory.

What the mutation run showed, with the comparison of one part of a kernel put back to the other side's (the tables are in the kernels' `attempts.md`):

| Kernel | Part | Before | After |
|---|---|---|---|
| `searchsorted_right_f64` | First step of the long-array path (`Lbig_init`) | Passes | Fails: `n=1048577 scalars=k128 a=ties v=from_a` |
| `searchsorted_right_f64` | Later steps of it (`Lbig_probe`) | Passes | Fails: the same case |
| `searchsorted_f64` | `Lbig_init` | Passes | Fails: `n=2097151 scalars=k128 a=ties v=from_a` |
| `searchsorted_f64` | `Lbig_probe` | Passes | Fails: `n=1048577 scalars=k128 a=ties v=from_a` |
| `interp_f64` | Later steps of the 64-query search, and the sixteen-query count | Pass every fixed case; caught by a fresh case in nine runs of ten | Fail on a fixed case: `x=from_xp xp=ties` at 64 and 16 points |
| `interp_f64` | First step of the 64-query search | Passes | Passes, and cannot fail: see below |

`interp_f64`'s first step is not a gap. When it takes the lower window for a query equal to `a[p - 1]`, the search ends at `p`, inside the run of equal sample points; the interpolation then divides by a zero width, the result is NaN, and the kernel recomputes that query by the reference's own steps. Every outcome is right, so no input can tell the mutation apart.

Why this and not the issue's other option, the pairings of plain classes extended to one large size per path boundary: the pairings run in the default scalar set, and the long-array path is not reached there; extending them to every set multiplies eleven classes of each argument at sizes where an array takes tens of milliseconds to build. And no pairing of plain classes gives a needle equal to an element of random data.

Why a keyword and not a function in `classes`: those functions take the argument's own random array. Drawing needs another argument's finished array and a random generator, and the judge has to know the dependency to build the arguments in order and to pick the source's class. One recipe covers the three kernels; a function form can follow when a kernel needs another.

What it does not give: a needle equal to `a[p - 1]` when the array has no repeated values comes up only by chance, at about one needle in `n`. The first step of a search is checked on ties through the `ties` source, not through random data.

### Scalar sets

A scalar often chooses a path through the kernel. `clip` takes a two-operation path unless a bound is a zero of the wrong sign, and `histogram` trusts its first bin estimate only when the bin count and range pass a check at entry. A kernel tested with one value of its scalars is tested on one of those paths. The first version of the judge did that, and it hid two things in `clip` that were found by other means: the reference disagrees with NumPy for reversed and NaN bounds, and the kernel is fast for some bounds only.

A spec therefore lists further combinations:

```python
KERNEL = Kernel(
    name="clip_f64",
    returns="void",
    args=[In("x", "f64"), Out("out", "f64"), Size("n"), Scalar("lo", "f64", -0.5), Scalar("hi", "f64", 0.5)],
    oracle=lambda x, lo, hi: np.clip(x, lo, hi),
    scalar_sets=[
        ScalarSet("zero_lo", {"lo": 0.0, "hi": 1.0}, timed=True),
        ScalarSet("neg_zero_hi", {"lo": -1.0, "hi": -0.0}),
        ScalarSet("infinite", {"lo": float("-inf"), "hi": float("inf")}),
    ],
)
```

- **Sets, not value classes.** Scalars are rarely independent: a lower bound must not exceed the upper one, and a bin count belongs to a range. Classes drawn per scalar would mostly produce combinations outside the kernel's domain. A set names one valid combination; a scalar it leaves out keeps its default.
- **The default set** is every `Scalar`'s own value. It always exists and gets the full sweep of cases, exactly as before, so a kernel without scalar sets runs the cases it always ran.
- **Every further set gets a thinner sweep:** every value class at the same sizes (less the exact powers of two above 65,536), but one memory arrangement per case instead of five, three indices per positional class instead of all, and no pairings of classes. What has to be shown is that each path handles every size and every kind of value; the default sweep has already shown the rest. A set costs about 2,700 cases and under a second for a one-input kernel.
- **Fresh cases** draw their set at random from all of them.
- **A case names its set:** `n=17 arr=end seed=1 scalars=zero_lo x=zeros`. The field is left out for the default set.
- **The reference is checked on every set too.** A `naive.c` that disagrees with NumPy for a listed combination fails as `reference`, which is how a wrong reference for some bounds would now be caught, provided the combination is listed.
- **`timed=True`** adds a full timing sweep with that set, reported as its own table (`random, scalars=zero_lo`) and included in the overall figure. `compare` times it as well, so an edit that helps one set of scalars and hurts another shows both. Each timed set adds about half a minute to `run` and `compare`, so they should be few. The timing loop takes the scalars as arguments; they are not compiled into it.

What this does not give: a combination nobody listed is still untested, and the array data does not follow the scalars. `histogram` is checked with sixteen bin counts and ranges, but the judge's data is random and seldom lands within rounding distance of a bin edge, which is where a histogram goes wrong. Evidence about edges still comes from a differential test outside the judge.

### Why a Python file

The oracle is Python code, so a Python file per kernel exists either way. Putting the signature in the same file means one file and one format per kernel.

The alternatives considered:

| Format | Why not |
|---|---|
| TOML plus a separate Python oracle | Two files to keep in step for no gain. TOML on its own cannot hold the oracle beyond a one-line expression. |
| JSON | No comments, and the same two-file problem. |
| YAML | Needs an extra dependency, has well-known parsing surprises, and the same two-file problem. |

The cost of Python is that a spec is executable code, not inert data. That is acceptable here because specs are written or reviewed by a person and locked against agent edits (see [Keeping the agent honest](#keeping-the-agent-honest)).

## Stage 1: lint

Cheap static checks on the assembly source, run before anything is executed. A kernel fails if it:

- calls any external function, or makes a system call
- declares writable data
- uses a register the platform reserves (`x18` on Apple platforms)
- exports any symbol other than the kernel itself

A kernel is a pure function over the buffers it is given. These rules keep it that way.

The rules are the same everywhere. What counts as a call, a system call or a reserved register is listed under [Per architecture](#per-architecture).

## Stage 2: correctness

The kernel and `naive.c` are both run against the oracle. Checking `naive.c` as well matters: if it is wrong, the speed comparison is meaningless.

### Inputs

Every case is generated from a fixed seed, so a failure can always be reproduced.

- **Sizes.** Every size from `min_n` to 65, then each power of two up to about four million, together with one less and one more. Small sizes catch mistakes in leftover-element handling; the neighbours of powers of two catch mistakes at block boundaries.
- **Values.** Chosen per data type:
  - random data
  - all elements equal
  - sorted ascending and descending
  - floating point: NaN, both infinities, both signed zeros, denormals
  - integers: the minimum and maximum values
- **Positions.** For sizes up to 65, the interesting element (the maximum, a NaN, a single negative zero, the only `True` in a mask) is placed at every possible index in turn. This catches a kernel that is right except in one lane of one vector. At larger sizes it is placed at the start, the middle and the end, which is what reaches code that only runs once a kernel is well into an array: `clip` with a zero lower bound redoes a chunk when it meets a negative zero among ordinary values, and nothing but a single one far into the array gets it there.
- **Ties.** Duplicates of the winning value at several positions, since NumPy defines which one wins. For a search, needles equal to elements of the array, drawn from it at every size (see [Values drawn from another argument](#values-drawn-from-another-argument)).
- **Alignment.** Each case runs with the buffers at several different offsets from an aligned address.
- **Fresh random cases.** After the fixed cases, a batch generated from a new seed on every run. The seed is printed. This stops a kernel from passing by fitting the fixed cases.

### Memory safety

Buffers are not ordinary allocations. The judge maps them so that:

- **Reads and writes past the end crash.** Each buffer ends exactly at the boundary of an inaccessible page, and a second arrangement starts it exactly after one. Any access outside the buffer, by even one byte, faults.
- **Writes to inputs crash.** Input buffers are mapped read-only for the duration of the call.
- **Writes past the used output are detected.** The unused part of an output buffer is filled with a known pattern and checked afterwards. A spec can relax this for one buffer; see [Scratch output](#scratch-output-writing-inside-the-capacity).

The rule is strict: a kernel may not read outside its buffers at all.

#### What the strict rule costs

A SIMD kernel handles several elements per step, say four. An array of 37 elements is nine full steps plus one element left over. There are two ways to deal with the leftover:

- **Stay inside the buffer.** Handle the last element with separate, slower code. This is extra code and extra branches.
- **Read past the end.** Load one more full step of four, and ignore the three values that are not part of the array. This is shorter and faster, because every step looks the same.

Reading past the end changes nothing, so it is harmless as long as the extra bytes belong to the program. Memory is handed out in fixed-size pages. If the array happens to end at the very edge of the last page the program owns, the extra read lands on memory it does not own and the program crashes. Code that uses this trick therefore has to check how close it is to a page edge first.

The trick only matters for tiny arrays, where the leftover is most of the work. For an array of a million elements the leftover is a rounding error. The rule stays strict because it is simple to state to an agent and simple to verify. It can be relaxed for specific kernels later if the survey shows that tiny arrays are where the wins are.

Some architectures offer a third way: a masked load, which names the elements it wants and does not touch the others, so nothing outside the buffer is read and nothing faults. x86-64 has one and ARM64 NEON does not. Whether it satisfies the rule is [open issue 3](x86-64-design.md#open-issues) of the x86-64 design.

#### Scratch output: writing inside the capacity

*Decided in review of the survey.*

By default a kernel may write only the output it reports. A spec can relax this for one output buffer by declaring it as scratch:

```python
Out("out", "f32", capacity="n", used="return", scratch=True)
```

A kernel may then write anywhere inside the declared capacity of that buffer. Only the reported part is compared with NumPy; the rest holds unspecified values. Writing even one byte past the capacity is still a failure, as is every other memory rule.

**Why it is needed.** Mask selection (`a[mask]`) copies the selected elements to the output. The simple loop asks "is this element selected?" before each store, and on an unpredictable mask the processor guesses that answer wrong about half the time. The fast loop does not ask: it stores every element at the current output position, and moves the position forward only if the element was selected. An unselected element is overwritten by the next one. The survey measured this on ARM64 at 0.34 ns per element against 3.2 ns for the loop that asks. A vector kernel does the same with a whole block of elements per store.

**The last-element problem.** Overwriting only works while there is a next element. Whatever was stored last and not selected stays where it is, just past the reported output. With the scalar loop that is at most one stray element; with a vector kernel it is up to one block minus one. The reported count and the reported elements are correct. The slots after them are not empty, and not what they held before the call.

That has consequences, which is why scratch is opt-in per buffer and not the default:

| Consequence | What it means in practice |
|---|---|
| The tail of the buffer is garbage | The binding must hand the user exactly the reported elements and never a view of the whole buffer. Otherwise values the mask excluded become visible |
| The buffer must be big enough for the stray writes | A block store at output position `k` is safe only because `k` never runs ahead of the input position, so a capacity of `n` always suffices. A binding that counts the selected elements first and allocates exactly that many cannot use a scratch kernel on that buffer: the last block would run past the end |
| Capacity `n` costs memory | NumPy counts first and allocates only what is selected. Allocating `n` for a mask that selects 1% touches a hundred times more memory before the result is shrunk. The binding has to choose between that and a count pass followed by a strict tail |
| A caller's own buffer cannot be scratch | With `np.compress(mask, a, out=buf)` the buffer belongs to the user. Stray writes past the result would change data the user can see. The binding must use a strict kernel, or a temporary buffer, whenever the output is not its own |
| The judge checks less | For a scratch buffer the judge no longer detects a kernel that stores one element too many inside the capacity. A wrong count or a wrong reported element is still caught by the comparison, and a store past the capacity is still caught by the guard page |
| Two correct kernels can differ | The tail is unspecified, so byte-for-byte comparison covers the reported part only |

**Working memory.** A capacity can be a multiple of `n`, `capacity=Times(3)`, and a kernel that needs memory in proportion to its input uses a scratch output for it: the stable `argsort` kernels report `n` indices at the start of a buffer of `3n` and keep the elements they are moving in the rest ([sort-design.md](sort-design.md#the-judge)). The judge supports one output per kernel, and this needed nothing else of it. Every consequence in the table holds, the first most of all: the buffer is the binding's own, and the user is handed a copy of the reported part.

The assembly still needs care at the end of the input: the leftover elements that do not fill a block are handled one at a time, because a full block load would read past the input, which stays forbidden.

### Calling convention

Assembly can corrupt its caller in ways that tests pass by luck. The judge calls each kernel through a small assembly wrapper that fills every register the kernel is required to preserve with known values, calls the kernel, and verifies they are unchanged, along with the stack pointer. There is one such wrapper per architecture. What each one checks is listed under [Per architecture](#per-architecture).

### Comparison

- **Exact mode:** the output bytes must equal the oracle's bytes.
- **Tolerance mode:** each value must be within the declared number of units in the last place. NaN and infinity must appear in exactly the same positions.
- **Tolerance with a scale:** by default the unit in the last place is taken at the size of the value itself. That is wrong for a running sum: terms that cancel leave a small result carrying the rounding error of the large terms, and a reordered float `cumsum` then differs from NumPy's by millions of units of the result while being exactly as accurate. A spec can therefore supply a scale function with its tolerance, `mode={"tolerance_ulp": 4, "scale": fn}`, which returns per output element the magnitude the unit is taken at. For `cumsum` that is `(i + 1) * (|x[0]| + ... + |x[i]|)`, the standard bound on the error of adding in any order. The price is a weaker check where the bound is wide: for float32 beyond some tens of thousands of elements it no longer notices a single skipped element. Small sizes, where every length and alignment is tried, are not affected.
- The return value is compared the same way as output buffers.

### Crashes

Each kernel is checked in a separate process. If it crashes, the judge reports which case was running and fails the kernel; the judge itself keeps going.

### Failure output

Output is written for an agent to act on: short, and specific enough to reproduce. Only the first failure is shown.

```
FAIL argmax_f32 (arm64): wrong result
  case:     nan_at_index  n=37  nan_index=33  offset=4  seed=1234
  expected: 33
  got:      0
  rerun:    uv run judge check argmax_f32 --case nan_at_index:37:33:4:1234
```

## Stage 3: speed

Timing is best effort for now. The method below is careful enough to trust a clear result, and it gets stricter if the project starts finding results worth defending.

### What is timed

| Contender | How it is timed |
|---|---|
| The assembly kernel | Inside C, called in a loop through a function pointer |
| `naive.c` | The same loop; compiled by the judge with `clang -O3`, tuned for the machine it runs on (`-mcpu=native` on ARM64, `-march=native` on x86-64) |
| NumPy | The oracle function, called from Python |

The kernel directory cannot change compiler flags, so the C baseline cannot be weakened.

The NumPy number includes NumPy's own per-call overhead, because that is what a user experiences. The assembly number does not yet include the cost of our binding, which does not exist until phase 3. So at small sizes the comparison with NumPy flatters the kernel, and the comparison with naive C is the honest one. The report says this next to the NumPy column. Phase 3 measures end to end.

### Sizes

A fixed sweep: 16, 64, 256, 1,024, 4,096, 16,384, 65,536, 262,144, 1,048,576, 4,194,304 and 16,777,216 elements. This spans data that fits in the fastest cache through to data far larger than any cache.

### Method

- **Repeat count.** For each size the loop count is calibrated so one sample takes at least 1 ms.
- **Samples.** 300 samples per contender per size. Where a single call of some contender lasts longer than 1 ms, fewer fit the same time, and never fewer than 30; every contender at that size gets the same number. Why the samples are short and many is under [Short samples](#short-samples).
- **Interleaving.** Samples from the contenders are alternated, so slow drift in the machine affects all of them equally.
- **Varied inputs.** At small sizes the loop cycles through a pool of different inputs. Timing one input repeatedly lets the processor memorise its branch pattern, which makes branchy code look faster than it is on real data.
- **Input shapes.** Where speed depends on the data (how many mask elements are true, where the maximum sits), the spec lists a few named distributions and each is timed.
- **Performance cores.** The timing thread is kept on a performance core. On macOS it asks for the highest scheduling priority class, which does that.
- **Where the output sits.** The judge decides where each output buffer of a timed call lies relative to that call's inputs, and does not leave it to the allocator. See [Where the output buffer sits](#where-the-output-buffer-sits).

### What is reported

There are no pass or fail thresholds on speed. For each size, and overall, the judge reports:

- the median time of each contender
- the speedup of the kernel over naive C and over NumPy
- a 95% confidence interval for each speedup, computed by resampling the measured samples
- whether the speedup is statistically significant, meaning the interval does not include 1.0

The overall figure is the geometric mean of the per-size speedups, with its own interval.

```
argmax_f32 (arm64, Apple M1 Pro)        vs naive C              vs NumPy (includes call overhead)
        n      kernel      speedup   95% interval          speedup   95% interval
       16      3.1 ns       1.02x    0.97 – 1.06  n.s.      310x     ...
    4,096      210 ns       2.41x    2.36 – 2.47             3.9x    ...
  ...
  overall                   2.10x    2.05 – 2.16             ...
```

### Short samples

*Changed after the first x86-64 kernel. The method used to be 30 samples of 10 ms.*

**What went wrong.** On the x86-64 machine, with a desktop session running, `compare` was given the same kernel as both contenders, seven times. Three of the seven verdicts were "significant" (0.975x, 1.011x, 0.994x). Every size up to 1,048,576 bytes read 1.00x each time. The two largest, 4,194,304 and 16,777,216, differed by up to 17% between two copies of the same code.

**Why.** Those two sizes do not fit the 2 MB cache a core has to itself. They are served from the cache and the memory that all cores share, so the memory traffic of the other cores slows them, and on a desktop that traffic comes in bursts a few milliseconds long, many times a second. Single calls show it: at 4,194,304 bytes one call took 39.5 to 40.0 us nine times out of ten, while samples of 249 calls in a row averaged anything from 39.8 to 68 us. A 10 ms sample nearly always contains part of a burst, so its time is a mixture, and the median of 30 mixtures moves with how much each one caught.

**What changed.** The same measuring time, cut finer: 300 samples of 1 ms. A 1 ms sample is either clear of the bursts or inside one. Most are clear, and the median is then the undisturbed time. One function timed against itself twenty times over, largest difference between the two medians:

| Size | 30 samples of 10 ms | 100 samples of 3 ms | 300 samples of 1 ms |
|---|---|---|---|
| 4,096 | 0.11% | 0.03% | 0.03% |
| 65,536 | 0.45% | 0.04% | 0.03% |
| 1,048,576 | 1.02% | 0.14% | 0.14% |
| 4,194,304 | 16.3% | 0.68% | 0.10% |
| 16,777,216 | 16.5% | 1.27% | 0.32% |

- **The statistics are unchanged:** medians, the paired bootstrap, the geometric mean. Only the samples they are computed from are shorter and more numerous.
- **A run takes as long as before.** The time given to a size is what 300 samples of 1 ms take per contender. If one call of a contender is longer than 1 ms, fewer samples are taken of every contender at that size, down to 30.
- **Results recorded before the change** were taken with 30 samples of 10 ms. On the Apple M1 Pro they were taken with nothing else running; whether that machine shows the same effect has not been measured.

### Where the output buffer sits

*Added when the survey was run on x86-64.*

**What goes wrong.** An x86-64 processor decides whether a load has to wait for an earlier store by comparing only the address bits within a 4 KB memory page. If the output buffer starts a few bytes after the input, counted within the page, each store looks like a store to the elements about to be loaded, and the loop stalls. The naive C running sum over 1,048,576 `int64` then takes 1.46 ms instead of 0.29 ms. The slowdown is largest 8 to 32 bytes after the input, visible at 128 and gone at 256.

**Why the judge has to care.** It used to take its timing buffers from NumPy's allocator. Two arrays of a power-of-two size made one after the other often sit exactly 16 bytes apart within the page, because the allocator puts a 16-byte header before each. Which sizes of a sweep landed there changed from one process to the next. A `compare` could then be decided by the allocator.

**What the judge does.**

- **Every timing is taken with the output clear of the inputs.** Each output buffer starts either exactly where each input of its call does, counted within the page, or at least 256 bytes after it. This is what `run`, `bench`, `compare` and the selftest time.
- **`run` and `bench` add one table that times the bad case on purpose:** the output 16 bytes behind its input. A caller's arrays are often there, so a kernel that stalls in that position is slower for real users than its main table says. The table has the kernel in the close placement, the same code with the output clear (`apart`), and naive C in the close placement. A speedup below 1 against `apart` is what the placement costs.
- **The two placements differ in nothing else.** In both, the output starts at the same position within a 64-byte cache line as it would 16 bytes after the input. Without that, the two would also differ in how many vector stores straddle two lines, which alone moved times by up to 17%.
- **The input an output is placed against** is the first one with as many elements as the output, since that is the one a kernel reads in step with its writes; otherwise the first input.
- **Only `Out` buffers are placed.** An in-place buffer is its own input.
- **The outputs of a pool of small inputs are carved from one block**, each as low as its rule allows, so they stay as close together as the inputs are.

```
cumsum_i64: output buffer 16 bytes behind its input, counted within a memory page

           n     naive C      apart*  speedup  [95% interval]
       4,096    783.1 ns    782.8 ns    1.00x  [1.00 – 1.00] n.s.
      65,536     12.5 us     12.5 us    1.00x  [1.00 – 1.00] n.s.
     262,144    449.4 us     65.6 us    0.15x  [0.14 – 0.15]
   1,048,576     1.46 ms    294.1 us    0.20x  [0.20 – 0.20]
   4,194,304     2.57 ms     2.61 ms    1.02x  [1.01 – 1.03]
```

That is `cumsum_i64` as plain C on the Intel Core i5-14600KF. The cost shows where the data lies in the third-level cache, and not where it fits the second-level cache or is larger than all of them.

**Limits.**

- **The close table covers one distribution and the default scalars**, and starts at the first size that is timed on a single input. The outputs of a pool of small inputs cannot each be 16 bytes behind their own input and stay close to one another, and spreading them over many pages would be timed instead.
- **The close table is not part of the overall figure** and plays no part in `compare`. Whether a kernel has to do well in it is for a person to decide.
- **NumPy is not in it, and its column in the main table is not placed.** The oracle allocates its own result, so the allocator still decides where that lies. A NumPy time can be several times higher at one size than at its neighbours for this reason. `uv run delivery time` measures real calls and is the place to compare with NumPy.
- **The same rule runs on ARM64.** It costs nothing where the processor compares whole addresses, and the close table then reads 1.00x. That has not been measured yet.

### Noise

Before and after the sweep the judge times a small fixed reference loop. If the two disagree by more than 3%, or the platform says the machine is throttled (Low Power Mode on macOS), the report carries a prominent warning that the machine was noisy. The numbers are still shown; the reader decides whether to rerun.

### Comparing two versions of a kernel

During iteration the question is narrower: is this edit faster than what we had? `judge compare` builds the kernel from an earlier git revision alongside the working copy and times the two interleaved. It reports one of three outcomes:

- significantly faster
- significantly slower
- no significant difference

An agent in a `/goal` run keeps an edit only when `compare` says significantly faster. This is bookkeeping for the iteration loop, not a decision about shipping. Whether the finished kernel is worth shipping stays with a person.

**Rounds.** *Added after the first x86-64 kernel.* `compare` takes its measurement in five rounds, each in a process of its own, with a fifth of the samples in each.

- **Why.** Two builds of the same source are not the same inside one process. They are loaded at different addresses, and that alone can cost one of them a mispredicted branch per call at one size. Every sample of that process carries the same offset, so an interval built from those samples does not contain it, and with [short samples](#short-samples) that interval is only a few tenths of a percent wide. On the x86-64 machine, 41 comparisons of a kernel with itself were off by up to 0.7% overall and by up to 4% at a single size, and most of them were called significant.
- **What a round changes.** A new process loads both builds somewhere else and draws the offset again. The spread between rounds is therefore the uncertainty that matters.
- **The statistics.** Within a round a speedup is the ratio of the two median times, as everywhere else. Across rounds it is the geometric mean of those ratios, and its 95% interval comes from how much the rounds differ (Student's t on the logarithms, four degrees of freedom). The overall figure is treated the same way: one geometric mean over the sizes per round, then the interval across rounds. No fixed margin is applied.
- **What it costs.** The same measuring time, plus five process starts.
- **`run` and `bench` are not changed.** They report one process. A single size of theirs can be off by a few percent for this reason, which is small against the speedups they report and is why an edit is judged by `compare`.

## Results

- Every `judge run` writes its full report as JSON under `.judge/`, which is not committed.
- `judge run --record` also copies the report into `results/`, one file per kernel, architecture and machine. Recording is a deliberate step taken when a person decides the result is worth keeping. Recorded results are committed for now; this can be revisited.
- A recorded result holds the timings and intervals per size, the processor model, the operating system, compiler and NumPy versions, and the git revisions of the kernel, its spec and the judge.
- `judge leaderboard` turns the recorded results into `LEADERBOARD.md`: one row per kernel with its speedup over naive C and over NumPy.

The NumPy version is pinned in the lock file. Results are only comparable across the same NumPy version, since NumPy's own speed changes between releases.

## Keeping the agent honest

An agent told to make a number go up will sometimes do it the wrong way. The judge closes the obvious routes:

- **Limited write access.** During a kernel run, the only files allowed to change are that kernel's `<arch>/kernel.s` and the `attempts.md` beside it. The judge checks with git that the spec, `naive.c` and the judge itself match the main branch, and fails the run otherwise.
- **A human approves the reference.** `spec.py` and `naive.c` are reviewed by a person when a kernel is created, because everything else is measured against them.
- **Fixed compiler flags** for the baseline, as above.
- **Fresh random cases** on every correctness run, as above.

## Testing the judge

`judge selftest` runs the judge against fixture kernels that are wrong in known ways and confirms each one is caught for the right reason. This is the exit criterion for phase 1. The fixtures are checked at the same time, one thread each; a fixture's own checks run one after another, since each rebuilds its naive reference and harness into the same files. The two steps that build a fixture outside its thread, the timing buffers and the slow variant, run after the threads, so the slow variant is timed with nothing else running.

There are two fixture kernels, matching the two spec examples above: `argmax_f32` (returns a value) and `compress_f32` (writes variable-length output). Each has a correct, deliberately simple assembly version, plus broken variants:

A third fixture, `compress_scratch_f32`, covers [scratch output](#scratch-output-writing-inside-the-capacity). It is the same operation as `compress_f32` with the output declared `scratch=True`. Its correct kernel and its naive C both leave a stray element past the reported output, and must pass. The same assembly is a broken variant of `compress_f32`, where it must fail, so the two fixtures together prove the rule applies only where the spec asks for it.

A fourth, `tally_u8`, covers [scalar sets](#scalar-sets). It counts elements by their remainder modulo a scalar bucket count, which also sizes its output. Both of its broken variants are correct for the default bucket count and wrong for another, so they are caught only if the other sets are really run and the output is really sized by the scalar of the case.

A fifth, `rank_u8`, covers [a second length](#a-second-length). For each of a scalar number of needles it counts the elements of an array smaller than the needle. One broken variant reads a byte past the needles, which is caught only if that buffer is placed against the guard page by the scalar's value and not by `n`. The other handles the default number of needles whatever the scalar says.

`rank_u8` also draws its needles from its array, and a selftest step confirms what [Values drawn from another argument](#values-drawn-from-another-argument) promises without running a kernel: at every size the fixture is checked at, in every scalar set, there is a case whose needles are all elements of the array as the kernel receives it, and the array holds repeated values.

A capacity that is a multiple of `n` (`capacity=Times(3)`) has no fixture of its own: a selftest step builds a spec with one and confirms that the output is sized at that many elements per element at several sizes, and that a factor that is not a whole number of at least 1 is refused. The kernels that use it, the stable `argsort` kernels, are what run it against guard pages.

| Broken variant | Must be caught by |
|---|---|
| Skips the last element when the size is not a multiple of the vector width | Correctness |
| Ignores NaN | Correctness |
| Returns the last of several tied maxima instead of the first | Correctness |
| Only correct when the buffer is aligned | Correctness |
| Reads one vector past the end of the input | Memory safety |
| Writes one element past the used output | Memory safety |
| Writes one element past the capacity of a scratch output | Memory safety |
| Writes to an input buffer | Memory safety |
| Hard-codes the default value of a scalar | Correctness, in a non-default scalar set |
| Clears as many output elements as the default scalar value gives, when the case gives fewer | Memory safety, in a non-default scalar set |
| Reads one byte past an input whose length is a scalar | Memory safety |
| Handles the default number of elements of an input whose length is a scalar | Correctness, in a non-default scalar set |
| Overwrites a register it must preserve | Calling convention |
| Uses a register the platform reserves (`x18` on ARM64) | Lint |
| Uses an instruction outside the level (x86-64) | Lint |
| Returns without clearing the upper halves of the vector registers (x86-64) | Calling convention |
| Correct but slower than naive C | Speed report: shown as significantly slower |

The second half of the exit criterion is repeatability: ten consecutive `judge bench` runs of the correct fixture on an idle machine agree to within 3% at every size.

The fixtures test the judge. They are not candidates for shipping; phase 2 decides which kernels are worth writing.

## What agents get

Phase 1 also delivers the two documents that turn the judge into an agent loop. Their content follows directly from this design, so they are written alongside the implementation and not designed separately:

- **`AGENTS.md`:** the kernel contract, the assembly conventions for the platform, the judge commands, how to read a failure, the files that may be edited, and the one rule: keep a change only if `judge check` passes and `judge compare` says it is significantly faster.
- **Goal-spec template:** the prompt for a `/goal` run on one kernel. It sets out:
  - **The target.** A statistically significant overall speedup of at least 10% over naive C. This is a target given to the agent, not a threshold in the judge.
  - **Do not stop at the target.** After reaching 10% the agent keeps going until it has exhausted the optimisations it can think of.
  - **If the target cannot be reached.** The agent reports its best result and explains why 10% was not possible, for example that the compiler already vectorises the naive loop well, or that the operation is limited by memory speed.
  - **The log.** Every attempt goes into `attempts.md` with what was tried and what the judge said.
  - **The final report.** The best speedup with its confidence interval, the optimisations that worked, the ones that did not, and what was left untried.

## Per architecture

The stages above state rules. This section says how each platform enforces them. Both columns describe what is built. Why x86-64 is done this way is in [x86-64-design.md](x86-64-design.md#the-judge).

| | ARM64 on macOS | x86-64 on Linux |
|---|---|---|
| Kernel source | `kernels/<name>/arm64/kernel.s` | `kernels/<name>/x86_64/kernel.s`, GNU assembler in Intel syntax |
| Assembled by | The C compiler | GNU `as`, because it can be told which instructions exist |
| Naive C | `clang -O3 -mcpu=native` | `clang -O3 -march=native` |
| Lint: calls | `bl`, `blr` and their authenticated forms | `call` |
| Lint: system calls | `svc`, `hvc`, `smc` | `syscall`, `sysenter`, `int` |
| Lint: reserved register | `x18` | None |
| Instruction-set level | Not needed: every Apple Silicon Mac runs what the M1 runs | The kernel is assembled a second time with x86-64-v3 stated, and anything the assembler then refuses fails as lint. The development machine runs newer instructions, so a crash cannot be relied on to catch one |
| Must be preserved | `x19` to `x28`, `x29`, the stack pointer, the low 64 bits of `v8` to `v15` | `rbx`, `rbp`, `r12` to `r15`, the stack pointer. Also: the direction flag clear, the rounding mode and exception masks unchanged, the upper halves of the vector registers cleared before returning, which the wrapper asks the processor about |
| Read-only constants | `__TEXT,__const` or `__TEXT,__literal16` | `.rodata` |
| Performance cores | The highest scheduling priority class | The timing thread pinned to one performance core. The core is claimed through a lock file, so judges running at once take different cores |
| Output placement | The same rule; no effect is known and none has been measured | The reason for the rule: an output just behind its input stalls the loop. See [Where the output buffer sits](#where-the-output-buffer-sits) |
| Object format | Mach-O: symbols start with `_`, libraries end in `.dylib` | ELF: plain symbol names, `.so`, and every kernel must declare that it needs no executable stack |
| Sign of a throttled machine | Low Power Mode | Turbo switched off, a power-saving governor or energy preference, a load average of 2 or more, or a core shared with another judge |
| Integer arguments | Eight in registers | Six in registers, a seventh and eighth on the stack |
| Leftover elements | A scalar loop, or a final vector that overlaps the previous one | The same, and possibly a masked load: [open issue 3](x86-64-design.md#open-issues) |

Everything else is the same on both: the cases, the guarded buffers, the comparison with NumPy, the statistics, the locked files, and the selftest's broken variants. The variant that uses a reserved register has no x86-64 counterpart. x86-64 has two of its own: one uses an instruction outside the level, and one returns without clearing the upper halves of the vector registers.

## Out of scope

- Hardware performance counters (cycles, cache misses). On macOS they need elevated privileges. Wall-clock time with the method above is enough for now.
- Timing in CI. Shared runners are too noisy. CI will run `judge check` and `judge selftest` without the speed fixture.
- Multi-dimensional or non-contiguous arrays. Kernels see flat, contiguous buffers; anything else falls back to NumPy in the binding.
- x86-64 was out of scope for phase 1. It is designed in [x86-64-design.md](x86-64-design.md) and summarised under [Per architecture](#per-architecture).

## Decisions made in review

| Question | Decision |
|---|---|
| Spec format | One Python file per kernel, holding the signature and the oracle. Provisional; changes when a kernel needs it to. |
| Win thresholds | None. The judge reports speedups with confidence intervals and significance; a person decides. |
| Memory rule | Strict: no reads outside the buffer. |
| Writes inside the output capacity | Allowed per buffer when the spec declares it `scratch`. Strict by default. The tail is then garbage, so the binding must return only the reported elements and must not use a scratch kernel on a caller's buffer or an exactly sized one. |
| Keeping edits during iteration | The agent keeps an edit when `judge compare` says it is significantly faster. |
| Agent target | At least 10% over naive C, then continue until options are exhausted. If 10% is not possible, report best effort and why. |
| Committing results | Committed for now; revisit later. |
| NumPy timing | Best effort: timed from Python with call overhead, labelled as such. Stricter if results warrant it. |
| Where timing outputs sit | The judge places them: clear of the inputs for every timing, and `run` adds a table with the output 16 bytes behind its input. |
| Fixture kernels | `argmax_f32`, `compress_f32` and `compress_scratch_f32`. |
