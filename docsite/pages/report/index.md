<p class="lede">NumPy is the layer everything else in scientific Python stands on, and for most of what it does it is very hard to beat. This report is about the places where it is not: a short list of everyday operations where the loop inside NumPy runs several times slower than the processor allows, and what it took to replace those loops without anyone's code having to change.</p>

It is written for someone who has used NumPy for years, is comfortable with the mathematics, and has never needed to read assembly. There is no assembly tutorial here. What there is instead is the reasoning: where the time goes, what the hardware can do that a compiler will not ask of it, how you prove a hand-written loop is right when nobody is going to read it, and what the numbers look like when you are honest about their limits.

## The short version

- **The opportunity is real but narrow.** Measured against the simplest correct C loop, NumPy on an Apple M1 Pro is behind by 1.5x to 7x on boolean masks, `cumsum`, `clip`, `histogram`, `bincount`, `searchsorted` and a few others, and level or ahead on almost everything else. [Chapter 1]([[ root ]]report/numpy/).
- **The gains come from four ideas**, not from cleverness with instructions: do several elements per step, keep steps independent of each other, never let a branch depend on the data, and respect the caches. [Chapter 2]([[ root ]]report/machine/).
- **Correctness is decided by a program.** A judge compares each kernel with NumPy on thousands of generated inputs (18,081 for `argmax`), with every buffer placed against memory the process may not touch, and decides with a bootstrap test whether a change made it faster. [Chapter 3]([[ root ]]report/judge/).
- **The kernels were written by coding agents in a loop** against that judge: [[ data.attempts.total ]] logged attempts across 49 kernels, of which [[ data.attempts.kept ]] were kept and [[ data.attempts.reverted ]] thrown away. [Chapter 4]([[ root ]]report/loop/).
- **Twenty-one of 22 beat the compiler**, by anything from 1.1x to 23x, and all 22 beat NumPy at every size measured. [Chapters 5]([[ root ]]report/selection/) and [6]([[ root ]]report/arithmetic/).
- **They reach unmodified code** by changing a handful of function pointers inside NumPy. A call no kernel takes costs about a nanosecond more. [Chapter 7]([[ root ]]report/delivery/).
- **Several results are narrower than their headline**, and the judge has blind spots. [Chapter 8]([[ root ]]report/results/).

## The whole system on one page

```mermaid How a kernel gets from an idea to your NumPy call. Solid arrows are things that are built; the dashed arrow is the loop an agent runs unattended.
flowchart TD
  survey["<b>Survey</b><br/>time NumPy against naive C<br/>for 61 operation variants"]
  spec["<b>Spec and reference</b><br/>spec.py: signature and NumPy oracle<br/>naive.c: the simplest correct loop"]
  agent["<b>Coding agent</b><br/>edits one file: kernel.s"]
  judge["<b>Judge</b><br/>lint, thousands of cases on guarded memory,<br/>calling convention, bootstrap timing"]
  log["attempts.md<br/>every attempt, kept or not"]
  kernels["<b>23 kernels</b><br/>22 assembly, 1 plain C"]
  binding["<b>Binding</b><br/>five kinds of hook into NumPy"]
  check["<b>Delivery check</b><br/>287,895 calls with and without,<br/>plus NumPy's own test suite"]
  user["<b>Your code</b><br/>import hotkernels; hotkernels.install()"]

  survey -->|"shortlist: where NumPy is behind"| spec
  spec -->|"reviewed by a person, then locked"| agent
  agent -->|"check, then compare"| judge
  judge -.->|"correct and significantly faster: commit<br/>anything else: revert"| agent
  agent --> log
  judge -->|"final verdict"| kernels
  kernels --> binding
  binding --> check
  check --> user

  class judge,kernels hot
  class user np
```

Three things in that picture carry the weight of the project:

1. **The reference.** For every kernel a person writes two small files: a spec that says how to call the kernel and which NumPy expression gives the right answer, and the most naive C loop that computes the same thing. Everything else is measured against these, so they are reviewed by a person and then locked.
2. **The judge.** It is the only reviewer the assembly ever gets. It has to be strict about correctness and honest about the uncertainty in a timing.
3. **The binding.** A kernel is a native function over a flat buffer. Getting `x[mask]` in someone's notebook to call it, without breaking anything when the array is strided, or a subclass, or of another type, is a separate problem and about half the work.

## Reading guide

Each chapter stands on the ones before it, but they are written to be read on their own if you only care about one.

<ol class="chapter-cards">
[% for chapter in chapters[1:] %]
<li><a href="[[ root ]][[ chapter.url ]]"><span class="num">[[ chapter.chapter ]]</span><b>[[ chapter.title ]]</b><span class="what">[[ chapter.summary ]]</span></a></li>
[% endfor %]
</ol>

## Ground rules for the numbers

Every figure in this report was measured on one machine: an Apple M1 Pro running macOS, NumPy 2.5.5 and Apple clang 21, natively, on a single core. Three kinds of number appear, and they are not interchangeable:

| Number | What is timed | Where it is honest |
|---|---|---|
| **vs naive C** | The kernel against the reference loop compiled with `clang -O3`, both called from C | Everywhere. This is the comparison that says what hand-written assembly added |
| **vs NumPy** | The kernel called from C against the NumPy statement called from Python | At large sizes. At small sizes it flatters the kernel, because NumPy pays its per-call cost and the kernel does not |
| **from Python** | One Python statement, on plain NumPy and again after `install()` | Everywhere. This is what a user gets, and it includes the binding |

When a speedup carries an interval, as in 1.21x [1.19 – 1.23], the bracket is a 95% bootstrap confidence interval, and "significant" means that interval excludes 1.0. [Chapter 3]([[ root ]]report/judge/#deciding-what-faster-means) explains how it is computed and lets you play with it.
