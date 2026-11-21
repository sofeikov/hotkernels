<p class="lede">The kernels in this project are several hundred lines of assembly each, written by a coding agent, and no person has reviewed them line by line. That is only defensible if something else stands between a subtly wrong kernel and a user's data. That something is a command-line tool called the judge, and it is the most important piece of the project.</p>

The judge takes one kernel and answers two questions. **Is it correct?** Does it return what NumPy returns on every input that can be thrown at it, without touching memory it does not own? That is an automatic pass or fail. **Is it faster?** By how much, compared with naive C and with NumPy, and is the difference real or noise? That is a measurement with an interval around it.

An agent runs the judge after every edit. So the correctness side has to be strict enough that passing means something, and the timing side has to know the limits of its own precision, because an agent told to make a number go up will happily climb noise.

## What the judge is given

Each kernel comes with a small Python file that states its contract: how to call it, and what the right answer is.

```python
import numpy as np
from judge.spec import In, Kernel, Size

def oracle(x):
    """Index of the maximum, as NumPy defines it."""
    return np.argmax(x)

KERNEL = Kernel(
    name="argmax_f32",
    returns="i64",
    args=[In("x", "f32"), Size("n")],
    oracle=oracle,
    min_n=1,  # np.argmax raises on empty input; the kernel is never called with n = 0
)
```

The **oracle** is the key idea. Correctness is not defined by a written description of what `argmax` should do with NaNs and ties. It is defined by calling NumPy. Whatever NumPy returns for an input is, by definition, the right answer for that input, including every corner case nobody thought to write down.

A kernel that writes a variable amount of output declares that too:

```python
KERNEL = Kernel(
    name="compress_f32",
    returns="usize",   # number of elements written
    args=[
        In("x", "f32"),
        In("mask", "bool"),
        Out("out", "f32", capacity="n", used="return", scratch=True),
        Size("n"),
    ],
    oracle=lambda x, mask: x[mask],
)
```

From the signature the judge generates the C glue to call the kernel. The author of a kernel never writes test code, and so cannot write test code that is too kind.

```mermaid What the judge's check and compare commands do with a kernel. Each stage runs only if the one before passed.
flowchart TD
  src["kernel.s"] --> lint["<b>Lint</b><br/>no calls, no system calls,<br/>no writable data,<br/>no reserved registers"]
  lint --> build["<b>Build</b><br/>assemble the kernel,<br/>compile naive.c"]
  build --> correct["<b>Correctness</b><br/>thousands of generated cases,<br/>each on guarded memory,<br/>against the NumPy oracle"]
  correct --> time["<b>Timing</b><br/>interleaved samples per size,<br/>bootstrap interval"]
  time --> verdict["significantly faster /<br/>no significant difference /<br/>significantly slower"]
  lint -.->|"fail"| stop["FAIL<br/>first failure only,<br/>with a command to rerun it"]
  build -.->|"fail"| stop
  correct -.->|"fail"| stop
  class correct,verdict hot
  class stop bad
```

## Lint: a kernel is a pure function

Before anything runs, the source is checked against a few static rules. A kernel may not call another function, make a system call, declare writable data, export any symbol but itself, or use the one register the platform reserves. These rules make a kernel a **pure leaf function** over the buffers it is given: it cannot have side effects, keep state between calls, or reach anything outside its arguments. Everything the later stages test is then everything the kernel can do.

## Generating the inputs

A vector kernel has more ways to be wrong than a scalar loop. It can be right for lengths that are a multiple of its block size and wrong for the rest. It can be right in three of its four lanes. It can be right until the interesting element lands on a block boundary. The cases are built to hit all of these on purpose, and every one comes from a fixed seed so that a failure can be reproduced exactly.

| Dimension | What is tried | What it catches |
|---|---|---|
| **Size** | Every length from the minimum to 65, then every power of two from 128 to 4,194,304 with the lengths one below and one above it | Leftover handling; block boundaries; paths that only exist for long arrays |
| **Values** | Random; all equal; ascending; descending; few distinct values. For floats also NaN, both infinities, both zeros, denormals, extremes. For integers the minimum and maximum | Special values and ties |
| **Position** | For lengths up to 65, the one interesting element (the maximum, a NaN, a lone negative zero, the only `True`) is placed at *every* index in turn; for longer arrays at the start, middle and end | A kernel that is right except in one lane of one vector |
| **Placement in memory** | Five arrangements per case: flush against the end of accessible memory, flush against the start, and three deliberately misaligned | Reads and writes outside the buffer; hidden reliance on alignment |
| **Scalars** | Every combination of scalar arguments the spec lists, such as twelve pairs of bounds for `clip` | A path chosen by a scalar that the default value never takes |
| **Fresh cases** | 400 more, from a new random seed on every run; the seed is printed | A kernel fitted to the fixed cases |

For `argmax_f32` that comes to 18,081 cases, and they run in about five seconds. The naive C reference is put through the same cases: if the reference were wrong, every speed comparison against it would be meaningless.

## Memory safety, by construction

This is the part of the judge that makes hand-written assembly acceptable at all. A kernel that reads sixteen bytes past the end of an array returns the right answer and passes every comparison. It will also, one day, crash someone's program, when their array happens to end at the edge of the memory the process owns.

So the judge does not hope to notice. It arranges for the mistake to be fatal every time. Kernel arguments are not ordinary allocations. Each one is placed in an **arena**: a run of normal memory pages with a page on each side that the process is forbidden to touch. A buffer is positioned so that its last byte is the last accessible byte, and in a second arrangement so that its first byte is the first accessible one. Any access outside the buffer, in either direction, by even one byte, is a hardware fault and the check fails.

<figure class="widget">
<div data-widget="lanes" data-n="37" data-mode="tail"></div>
<figcaption>The same figure as in the previous chapter, as the judge sees it. With the array pressed against an inaccessible page, "read past the end" is no longer a harmless shortcut: the first over-long load faults. The other two strategies never leave the buffer.</figcaption>
</figure>

Three further rules are enforced the same way:

- **Inputs are read-only.** For the duration of the call, input buffers are mapped without write permission. A kernel that scribbles on its input faults.
- **Output beyond what was reported is checked.** If a kernel says it wrote *k* elements, the rest of the output buffer must still hold the pattern the judge filled it with.
- **Each kernel runs in its own process.** A crash is caught, reported with the case that caused it, and counted as a failure.

::: aside The one relaxation: scratch output
The fast way to do `x[mask]` stores more than it keeps: every block is written whole, and the write position then advances only by the number of selected elements, so the surplus is overwritten by the next store. The last store leaves a few stray elements just past the result. A spec can permit this for one buffer by declaring it `scratch=True`. The kernel may then write anywhere inside the buffer's capacity, and only the reported elements are compared with NumPy. Writing one byte past the capacity is still a fault.

The same declaration lends a kernel working memory. A capacity can be a multiple of `n` (`capacity=Times(3)`), and the stable `argsort` kernels report `n` indices at the start of such a buffer and move their elements about in the rest of it.

This has a price, paid in the binding: the buffer has to be large enough for all `n` elements, the user must only ever be handed the reported part, and a scratch kernel can never be run on a buffer the user owns. [Chapter 7]([[ root ]]report/delivery/#scratch-output-and-the-cost-of-allocation) picks that up.
:::

### The register check

Assembly can damage its caller in a way no output comparison sees. By convention a function may freely overwrite some registers and must leave others exactly as it found them. A kernel that breaks this returns the right answer and corrupts a variable in whatever called it, sometimes.

The judge calls every kernel through a small wrapper that fills every register the kernel must preserve with known values, makes the call, and verifies them afterwards, along with the stack pointer. This wrapper is the only architecture-specific code in the judge.

## Exact, or within a stated bound

Each kernel declares how its output is compared with NumPy's.

**Exact** is the default: the output bytes must equal NumPy's bytes. Not "close", and not "equal as numbers": `-0.0` and `+0.0` compare equal as numbers and are different bytes, and two NaNs with different payloads are different bytes. Twenty-one of the 23 kernels are exact. For anything that selects, searches, counts, copies or compares, exactness costs nothing and a tolerance would only hide bugs.

**Tolerance** exists for the two float `cumsum` kernels, which regroup additions and so cannot be bit-identical. The natural unit is the **ulp** (unit in the last place): the gap between adjacent floating-point numbers at a given magnitude. Stating the bound in ulps makes one rule work for float32 and float64 alike.

The subtle part is: an ulp *of what?* The obvious choice is the size of the result. That is wrong for a running sum. If large terms cancel, the result is small but still carries the rounding error of the large terms, in NumPy's output as much as in a kernel's. Measured in ulps of a result that has cancelled to nearly zero, two equally accurate sums can differ by millions of units.

The standard error bound for summing in any order scales with the number of terms and the sum of their magnitudes. So the judge allows element $i$ of a running sum an error of

$$\lvert s_i - \hat{s}_i \rvert \;\le\; 4\,\operatorname{ulp}\!\Big( (i+1) \sum_{j \le i} \lvert x_j \rvert \Big),$$

where $\hat{s}_i$ is NumPy's value. NaNs and infinities get no tolerance at all: they must appear in exactly the positions where NumPy has them. The bound is the honest one, and it has a cost the project records openly: for float32 arrays beyond some tens of thousands of elements it grows wide enough that the check would no longer notice a single skipped element. Small sizes, where every length and alignment is tried, are unaffected.

## When it fails

Output is written for an agent to act on: one failure, and exactly how to see it again.

```text
FAIL argmax_f32 (arm64): wrong result
  wrong return value
  case:     n=2 arr=mid2 seed=1 x=nan_at@1
  expected: 1
  got:      0
  rerun:    uv run judge check argmax_f32 --case 'n=2 arr=mid2 seed=1 x=nan_at@1'
```

The case line is the recipe for the input: two elements, placed at a deliberately misaligned address, random data with a NaN at index 1. The category at the top is one of six: `lint`, `build`, `wrong result`, `memory safety`, `calling convention` or `crash`.

## Deciding what "faster" means

Timing is where agent-driven optimisation usually goes wrong. Run the same binary twice and the times differ by a percent or two. An agent that keeps every edit that came out ahead will accept, over a hundred attempts, dozens of changes that did nothing, and the final kernel will be a pile of superstition.

The judge's method is built around that problem.

- **Everything is timed inside C.** The kernel and the naive reference are called in a loop through a function pointer, so the measurement is the loop and not the cost of getting there.
- **Eleven sizes**, each four times the last, from 16 elements to 16,777,216: from data that fits in the fastest cache to data far larger than any.
- **Calibrated repeats.** For each size the repeat count is set so that one sample takes at least 1 ms.
- **Three hundred samples per contender per size**, fewer where a single call is longer than a sample. The figures in this report were taken with the earlier setting, 30 samples of 10 ms, on a machine with nothing else running. Short samples came with the second architecture: on a desktop machine, memory traffic from other cores arrives in bursts of a few milliseconds, a 10 ms sample nearly always catches part of one, and the median of 30 such samples moved by up to 16% for arrays larger than a core's own cache. A 1 ms sample is either clear of the bursts or inside one, and the median of 300 is the undisturbed time.
- **Interleaved.** The samples alternate between the contenders, so sample *i* of each was taken at nearly the same moment. Slow drift in the machine (temperature, another process waking up) then hits both.
- **A pool of inputs at small sizes.** The loop cycles through different inputs, so a branch predictor cannot memorise one and make branchy code look better than it is.
- **Performance cores.** The timing thread asks macOS for its highest scheduling class.
- **Placed outputs.** The judge decides where an output buffer sits relative to the inputs, and does not leave it to the allocator. On x86-64 an output a few bytes behind its input, counted within a memory page, stalls the loop, so every timing keeps them clear, and a second table times the bad case on purpose.

The speedup at one size is the ratio of the two median times. Its uncertainty comes from a **paired bootstrap**: draw as many sample indices as there are samples, with replacement, take the medians of *both* contenders at those same indices, form the ratio; repeat 2,000 times; the 2.5th and 97.5th percentiles of those ratios are the 95% interval. Resampling in pairs is what makes interleaving pay off: noise the two contenders shared cancels in each ratio instead of inflating the interval. A speedup is **significant** when its interval excludes 1.0.

<figure class="widget">
<div data-widget="bootstrap"></div>
<figcaption>The judge's decision, simulated. Set a true speedup and a noise level, and watch what 30 interleaved samples and a paired bootstrap make of it. Then repeat the measurement 200 times: a change that does nothing is still called "significantly faster" in a small share of runs, and a real 1% gain is often missed when the machine is noisy.</figcaption>
</figure>

The overall figure for a kernel is the geometric mean of its per-size speedups, $\big(\prod_{k=1}^{11} r_k\big)^{1/11}$, the right average for ratios. Its interval is built from the same resamples: the geometric mean is taken across sizes within each bootstrap draw, and the percentiles are taken of that.

Two more guards:

- **A noise canary.** Before and after the sweep the judge times a small fixed loop. If the two disagree by more than 3%, or the machine is in Low Power Mode, the report carries a warning that the machine was noisy. The numbers are still shown.
- **No thresholds on speed.** The judge reports speedups, intervals and significance. Whether a finished kernel is worth shipping is decided by a person reading the report.

### One edit against the last

During iteration the question is narrower: is this edit faster than the version before it? `judge compare` builds the kernel from the last commit alongside the working copy and times the two against each other, interleaved. It prints one of three verdicts:

```text
RESULT: significantly faster, 1.021x [1.017 – 1.032]
RESULT: no significant difference, 1.002x [0.998 – 1.004]
RESULT: significantly slower, 0.949x [0.943 – 0.956]
```

Only the first lets an edit stay. The middle line is from a real attempt whose point estimate was above 1; a looser rule would have kept it.

The comparisons in this report were each taken in one process. `compare` now takes five rounds, each in a process of its own, and builds its interval from how much the rounds differ. Two builds of the same source are loaded at different addresses, which can cost one of them a mispredicted branch per call; within one process that offset is the same in every sample, so no amount of sampling shows it. A new process draws it again.

## Keeping the agent honest

An optimiser pointed at a metric will find the cheap way to move it, if there is one. The judge closes the obvious routes.

- **Two files may change.** During a kernel run the agent may edit that kernel's assembly file and its attempts log, nothing else. The judge checks with git that the spec, the naive reference and the judge itself match the main branch, and refuses to run otherwise.
- **A person approves the reference.** The spec and `naive.c` are reviewed when a kernel is created, because everything is measured against them.
- **The baseline cannot be weakened.** The C is compiled by the judge with flags it controls; nothing in the kernel directory can change them.
- **Fresh random cases every run**, so a kernel cannot be fitted to a fixed test set.

## Who judges the judge

A test suite that has never failed proves little. So the judge ships with deliberately broken kernels, each wrong in one known way, and a self-test that passes only if every one is rejected *for the right reason*.

| A fixture kernel that... | Must be caught as |
|---|---|
| Skips the last element when the length is not a multiple of the vector width | wrong result |
| Ignores NaN | wrong result |
| Returns the last of several tied maxima instead of the first | wrong result |
| Is only correct when the buffer is aligned | wrong result |
| Reads one vector past the end of its input | memory safety |
| Writes one element past the output it reported | memory safety |
| Writes one element past the capacity of a scratch buffer | memory safety |
| Writes to an input | memory safety |
| Hard-codes the default value of a scalar argument | wrong result, in another scalar set |
| Reads one byte past an input whose length is a scalar argument | memory safety |
| Overwrites a register it must preserve | calling convention |
| Uses the register the platform reserves | lint |
| Is correct but slower than naive C | reported as significantly slower |

The self-test is the gate for any change to the judge, and it runs in continuous integration on every pull request that changes more than documentation.

## What the judge had to learn

The judge was designed around two toy kernels and then met twenty real ones. Each of these exists because a real kernel could not be expressed, or was wrongly passed, without it:

- **Scalar sets.** The first version tested one value per scalar argument. For `clip` that meant one pair of bounds, which exercised one of the kernel's three paths and hid the fact that the reference loop disagreed with NumPy for reversed bounds. A spec now lists named combinations (twelve for `clip`, sixteen bin counts and ranges for `histogram`), each checked on kernel and reference alike.
- **A scale for tolerances**, as described above, after a correct regrouped `cumsum` failed by millions of ulps of a cancelled result.
- **A lone negative zero** as a value class, after a `clip` kernel whose repair path stored nothing was found to pass every case, because no case reached that path and then continued.
- **Inputs that must be sorted**, fixed-size buffers, buffers sized by a scalar, and buffers modified in place.
- **Values drawn from another argument.** A search answers differently on its two sides only for a needle equal to an element, and random needles never equal one. Such needles came only from small inputs, and a search that compared the wrong way on its path for arrays of a million elements or more passed every case. A spec can now say that the needles are drawn from the array; the judge then draws them, from an array with repeated values, at every size and with every needle count.
- **A second length.** `np.searchsorted(a, v)` has two lengths that have nothing to do with each other, and the judge had one. The first search kernels were therefore written and judged with as many needles as elements, a shape no real call has, and could not be installed. An input may now take its length from a scalar argument, and the spec lists the needle counts that matter: none, one, and the counts either side of every group width the kernel works in.

It still has blind spots: scalar combinations nobody listed, input densities between the ones it times, a single repeated input at large sizes. They are listed with their consequences in [chapter 8]([[ root ]]report/results/#what-the-judge-does-not-see), because a tool like this is only trustworthy if you know where it is not looking.
