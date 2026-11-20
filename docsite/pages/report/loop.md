<p class="lede">With a judge that can be trusted, writing a kernel stops being a craft and becomes a search. A coding agent proposes a change, the judge rules on it, and the change is kept or thrown away. The agent needs no taste and no supervision, only a rule it cannot argue with. This chapter is about that rule, and about what [[ data.attempts.total ]] recorded attempts look like when you lay them end to end.</p>

## The one rule

An agent working on a kernel may keep a change only if both of these hold:

1. `judge check` passes: the kernel is still correct on every case.
2. `judge compare --against HEAD` reports `RESULT: significantly faster`.

Otherwise the change is reverted. Each kept change is committed before the next is tried, so the last commit is always the best version so far and the next comparison is always against it.

```mermaid The loop an agent runs, unattended, for one kernel. It ends when the agent has no ideas left that it has not tried.
stateDiagram-v2
  [*] --> Idea
  Idea --> Edit: change kernel.s
  Edit --> Check: judge check
  Check --> Revert: fails
  Check --> Compare: passes
  Compare --> Commit: significantly faster
  Compare --> Revert: no difference, or slower
  Commit --> Log
  Revert --> Log
  Log --> Idea: next idea
  Log --> [*]: ideas exhausted
```

That is hill-climbing with a statistical gate, and three details make it work.

**The gate is a hypothesis test, not a comparison of two numbers.** "Faster" means the 95% interval of the speedup excludes 1.0. An edit that measures 1.002x with an interval of [0.998 – 1.004] is reverted, even though its point estimate is above one.

**The agent can touch two files.** Its kernel's assembly, and its kernel's log. The spec, the reference loop and the judge are compared with the main branch on every run, and a difference fails the run. An agent cannot make its kernel look better by weakening what it is measured against.

**Every attempt is written down, including the failures.** Each kernel has an `attempts.md` per architecture, beside its `kernel.s`, with one entry per attempt: the idea, what `check` said, what `compare` said, whether it was kept, and what was learned. Later runs read the log first, so a dead end is walked once.

The instruction given to an agent is short. Reach a statistically significant overall speedup of at least 10% over naive C; having reached it, keep going until the ideas run out; if 10% cannot be reached, report the best result and explain why. The explanation is part of the deliverable: "the compiler already emits exactly this" and "this size is limited by memory, not by the loop" are both valid results.

## The log, all of it

Across the 40 kernels the logs hold **[[ data.attempts.total ]] attempts: [[ data.attempts.kept ]] kept and [[ data.attempts.reverted ]] reverted.** Of the ones taken to `judge compare`, [[ data.attempts.verdicts.faster ]] came out significantly faster, [[ data.attempts.verdicts.same ]] showed no significant difference and [[ data.attempts.verdicts.slower ]] were significantly slower. The rest were first versions with nothing to compare against, or ideas screened with a quick private measurement and dropped before reaching the judge.

Here they are. Pick a kernel, then click through its attempts.

<figure class="widget">
<div data-widget="attempts" data-kernel="argmax_f32"></div>
<figcaption>Every logged attempt for every kernel, read from the <code>attempts.md</code> files in the repository when this site was built. The line is the cumulative speedup of the kept changes over the kernel's first version; hollow points are attempts that were measured and thrown away.</figcaption>
</figure>

A few kernels are worth looking at specifically:

- **`argmax_f32`** is the classic shape. One structural idea (attempt 2) is worth 13x. The fourteen attempts after it fight over the last few percent, and ten of them are reverted.
- **`sum_u8`** shows a kernel hitting hard limits one after another: the core's load ports, then the second-level cache, then the cost of a function call. Its log ends with an explanation of why nothing more is possible at each size.
- **`cumsum_f64`** changes character halfway. The first attempts produce an exact kernel that ties naive C; then the tolerance mode arrives and the same kernel gains nearly 4x in one step.
- **`cast_u8_f32`** is a kernel that lost. Five attempts, a best of 1.05x with an interval that touches 1.0, and a verdict: ship the C.

## What the logs teach

Reading all of them, a handful of lessons repeat. None is specific to assembly.

### One idea does the work; the rest is polish

In almost every log there is a single attempt, usually the second, that carries most of the final speedup: block maxima for `argmax` (13.3x), table compaction for `compress_f32` (2.5x), pairwise byte accumulation for `sum_u8` (2.2x on top of a 2.4x first version), regrouping for float `cumsum` (3.8x to 5.3x). These are algorithmic changes: a different way of organising the computation, of the kind a compiler cannot invent. What follows is a long tail of 1% to 7% gains from shorter paths for small arrays, fewer instructions in the loop, a better order of branches at entry.

### After the big idea, latency matters more than instruction count

`argmax_f32`, attempt 9, replaced the final step of the kernel with a version that used fewer instructions: 30 vector operations in place of 37. It came out *slower*, 0.962x [0.960 – 0.964]. The log's diagnosis:

> The extra cost is about 24 cycles per call from 1,024 up, which matches the extra length of the dependency chain, not the operation count. Lesson: after the block loop the final phase is paid for by its latency, so it should be kept short and shallow, not merely small.

Attempt 10 applied the lesson in the other direction, restructuring the loop so that the address of the best block is ready before the loop ends, and gained 10% at 1,024 elements. This is [fact two]([[ root ]]report/machine/#fact-two-instructions-overlap-unless-they-wait-for-each-other) from the last chapter, discovered empirically by an agent and written down for the next one.

### Explicit prefetching almost never helps

Agents tried software prefetch on `argmax`, `sum`, the count kernels, the search kernels and others. On a single forward pass through an array it was slower nearly every time: 5% slower overall for `argmax` (0.949x [0.943 – 0.956]), because the prefetch instructions compete with the loads for the same slots and the hardware prefetcher was already keeping up. It paid in two situations: arrays several times larger than the second-level cache, and `interp` on large tables, where the addresses are not a forward walk and the kernel knows them before the hardware can guess.

### The rule cannot keep what the judge cannot see

`argmax_f32`, attempt 8, made the kernel find a NaN much faster once it knew a NaN was present. Measured privately, an array of 16,384 elements with a NaN at the end went from 5,711 ns to 898 ns. The judge reported no significant difference, because it times random data without NaNs, and that path never ran. The log entry:

> The judge times random data without NaN, so it cannot see this path, and the rule says not to keep it.

The change was reverted. That is the rule working as designed, and it is also a real cost of the design: an optimisation that only helps inputs the judge does not time is invisible to the loop. The fix is not to bend the rule but to add the input to the judge, which happened for other kernels (a second distribution for `isin`, negative-zero data for `clip`).

### A significant result can still be noise

At a 95% level, a change that does nothing will be called significantly faster about once in forty tries. With [[ data.attempts.total ]] attempts that is not a remote possibility, and the logs show agents meeting it. `argmax_f32`, attempt 7, removed one instruction from the path for 16 elements:

> Three runs: 1.004x [1.001 – 1.007], then 1.002x [0.997 – 1.007] and 1.002x [0.998 – 1.010].

Significant once, then not twice. It was not kept. Attempts 11 to 13 chase a 6% gain at one size that came with a 3 to 4% loss at two others; aligning the loop ruled out code placement as the cause, and the log records the effect as not understood and not pursued. An effect that shows at one size, from a one-instruction change, and does not survive a rerun is treated as noise until shown otherwise.

### Sometimes the answer is "this cannot be beaten"

Several logs end not with a number but with an argument:

- **Exact float `cumsum`.** Each output is the previous output plus one element, so $n$ elements need $n$ additions one after another, at the latency of an add: 0.93 ns. Naive C and an exact assembly kernel both measured 0.95 to 0.97 ns per element. No exact kernel can be faster. This was only discovered because the first specs were written as exact by mistake, and the agent's log explained why it could not reach the target.
- **`isin` with arbitrary 64-bit test values.** Eight compares per vector of two elements is the work, the compiler emits exactly that, and there is no wider compare on this hardware.
- **`cast_u8_f32`.** The compiler's loop already stores one 16-byte vector per cycle, which is all this core can store.

These are as valuable as the wins. Each one replaces "we did not find anything" with "here is the bound, and here is the measurement sitting on it".

### Private measurements have a place, clearly labelled

Agents often used a quick micro-benchmark of their own to choose between designs before spending 30 seconds on a judge run: which loop shape, how many accumulators, what prefetch distance. The byte-sum kernel's accumulator count was chosen this way, from the measurements in the [chain figure]([[ root ]]report/machine/#fact-two-instructions-overlap-unless-they-wait-for-each-other). The logs mark every such number as measured outside the judge. They guide the search; only the judge's verdict decides what is kept.

## Why this works at all

It is fair to ask why an agent, and not a person, and why a loop, and not a design.

A kernel like this has a very large space of small variations (block sizes, how many independent streams, the order of entry checks, which of three equivalent instruction sequences) and the right choice depends on details of one specific processor that are tedious to reason about and cheap to measure. A person would try five variations and stop. An agent tries sixteen, logs all of them, and does not mind that ten were wasted.

What the agent cannot be trusted with is the judgement of whether it succeeded. That is why the interesting engineering in this project is not in the kernels. It is in the two things the agent is not allowed to touch: the reference that defines "correct", and the judge that defines "faster".

The next two chapters look at what the search found.
