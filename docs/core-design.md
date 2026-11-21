# hotkernels: proof-of-concept design

This is a proposal for review. Nothing here is implemented yet.

It covers the proof of concept only: a trustworthy judge, a survey of where NumPy is slow, and the surveyed kernels delivered end to end. Everything beyond that is listed under [Later](#later) without a plan.

## What we are building

A Python package, `hotkernels`, that makes existing NumPy code faster with one line and no other changes:

```python
import hotkernels
hotkernels.install()
```

Behind that line sits a set of kernels. Each one is checked by an automated judge and only shipped if it is correct against NumPy and faster than NumPy. Where a naive C loop compiled with `clang -O3` is enough, the kernel is that C. Where hand-written assembly, written and tuned by a coding agent, beats the C as well, the kernel is the assembly.

The first target was ARM64 NEON on Apple Silicon. The second is x86-64 on Linux, designed in [x86-64-design.md](x86-64-design.md). The design keeps architectures apart from the start (see [Architectures](#architectures)).

Priorities, in order:

1. Real, measured speedups.
2. A user experience with no friction.
3. Learning to run coding agents against an objective judge. Understanding every line of the assembly comes later.

## Principles

1. **No friction.** One install command, one line to enable. No code changes, no configuration, no build step for the user.
2. **Correct by a declared standard.** Every kernel declares how its output is compared with NumPy's (see [Correctness policy](#correctness-policy)). Exact is the default.
3. **Never break anything.** If a kernel does not handle an input (dtype, memory layout, size, platform), the call goes to NumPy untouched. On an unsupported machine, `install()` does nothing.
4. **Only ship real wins.** A kernel must beat NumPy. Assembly is shipped only when it also beats the naive C baseline; when it merely matches the compiler, the C is shipped instead, because C is easier to maintain.
5. **Visible, reversible.** The user can see what was patched, how often each kernel ran, which kernels are inexact, and undo it all.

## What we know so far

One throwaway experiment was run on an M1 Pro: `out = max(0, a*x + y)` on float32, comparing NumPy, C and hand-written NEON.

| Finding | Number |
|---|---|
| Hand-written assembly vs `clang -O3` C | Tie (0.99x to 1.04x from 128 elements up) |
| One fused pass vs NumPy's three passes | 3.7x to 5.7x faster from 262,144 elements up |
| Cost of calling a kernel through `ctypes` | About 690 ns per call |
| Kernel time itself at 16 elements | About 2 ns |

Consequences:

- Simple elementwise maths is not worth writing in assembly. The targets must be operations the compiler cannot vectorise on its own.
- The binding layer matters as much as the kernels. A slow call path wipes out any win on small arrays.
- Fusing several operations into one pass is a large win, but it cannot be delivered transparently and is out of scope for the proof of concept.

These are findings about one machine and one compiler. They set the direction of the project and have not been repeated on x86-64.

## Kernel contract

A kernel is a directory:

```
kernels/<name>/
  spec.py          signature, dtypes, NumPy oracle, correctness mode
  naive.c          naive C reference: the simplest correct loop
  arm64/kernel.s   hand-written NEON (absent for a kernel shipped as C on ARM64)
  x86_64/kernel.s  hand-written AVX2 (designed in x86-64-design.md)
  <arch>/attempts.md   log of what was tried on that architecture and what the judge said
```

- `spec.py` and `naive.c` are shared by all architectures.
- Whether a kernel ships as assembly, as C or not at all is decided per architecture.
- `naive.c` has three jobs: it is the baseline the assembly must beat, a second check on correctness alongside NumPy, and the shipped implementation when no assembly beats it.
- `attempts.md` records every iteration, including failures, so later runs do not repeat dead ends.

## Correctness policy

Each kernel declares one of two modes in its spec.

- **`exact` (default).** Output must be byte-for-byte equal to NumPy's, including NaN, infinities and signed zero. This applies to every kernel that does no floating-point arithmetic: selection, search, counting, copying, comparison. For these, exactness costs nothing and a tolerance would only hide bugs.
- **`tolerance`.** For kernels that reorder floating-point arithmetic (sums, cumulative sums, dot products), where bit-identical output is often impossible at full speed. The bound is relative and expressed in units in the last place, so the same rule works for float32 and float64. A fixed absolute bound such as 1e-12 does not: float32 carries only about 7 significant digits. For a running sum the unit is taken at the size of the terms added so far, not at the size of the result: a result that has cancelled to almost nothing still carries the rounding error of its large terms, in NumPy's output as much as in a kernel's.

`report()` marks every installed kernel that is not exact.

## Architectures

Two are in the design, and both are built: ARM64 on macOS, and x86-64 on Linux, whose kernels were written to [x86-64-design.md](x86-64-design.md).

| Shared by every architecture | Per architecture |
|---|---|
| `spec.py` and `naive.c`: the contract, the oracle, the correctness mode | `kernel.s`, and the reasoning behind it |
| The judge's rules: what correct means, what a kernel may touch, how speed is compared | How the judge enforces them: the calling-convention wrapper, the lint rules, how the timing thread is kept on a fast core |
| The survey's method | The survey's results, and so the shortlist and the order of work |
| The binding and its hooks | Which kernels are installed, and what each asks of its caller |
| The principles above | Every measured number |

- **A kernel is carried to another architecture in spirit.** The idea and the proofs carry over. The instructions, the block sizes and every statement about what the hardware can do are written again. See [x86-64-design.md](x86-64-design.md#carrying-a-kernel-over).
- **The judge builds and tests whichever kernel directory matches the machine it runs on.**
- **Timings are only ever taken on native hardware.** x86-64 numbers come from a real x86-64 machine, never from Rosetta.
- **Documents say which architecture a statement is about.** What holds for an operation is stated once. What holds for one architecture or one machine sits under a heading that names it.
- **A processor that lacks the instructions a kernel needs gets no kernels.** On x86-64 `install()` checks for them at run time; see [x86-64-design.md](x86-64-design.md#target).

## Phases

Each phase has an exit criterion. The next phase does not start until it is met.

### Phase 1: the judge and the agent setup

The judge is the foundation. Agents can only iterate unattended if its verdict can be trusted. Once the judge, the kernel contract and `AGENTS.md` exist, writing a kernel is a `/goal` run with a kernel spec, not a separate phase.

**How:**

- Implement the kernel contract above.
- Correctness check. Run the kernel against NumPy under its declared mode across:
  - sizes 0, 1, every size up to 65, then powers of two and their neighbours, up to a few million
  - special values: NaN, infinities, signed zero, denormals, extreme integers
  - unaligned pointers
  - guard pages after the buffers, so a read or write past the end crashes instead of passing silently
- Speed check. Time each kernel inside C, not through Python, across a fixed size sweep. Report best-of-N against NumPy and the naive C baseline.
- Output a machine-readable result per kernel (pass or fail, timings per size) and a human-readable leaderboard.
- Write `AGENTS.md`: the kernel contract, the platform's assembly conventions, how to run the judge, and the single rule (keep a change only if the judge says correct and faster).
- Write a goal-spec template: what a `/goal` prompt for one kernel contains.

**Exit criterion:** the judge correctly rejects a set of deliberately broken kernels (off-by-one tail, wrong NaN handling, out-of-bounds read, wrong tie-breaking) and gives timings that are stable to within a few percent between runs.

### Phase 2: survey

Find out where NumPy is actually slow on ARM64 before choosing what to write. NumPy already has NEON code for some operations, so guessing is unreliable. The survey is repeated for every architecture, because NumPy's vector code differs between them.

**How:**

- Write the naive C baseline for a list of candidates and measure NumPy against it. Candidates: `argmax`/`argmin`, boolean mask selection (`a[mask]`), `count_nonzero`, `nonzero`, `cumsum`, `searchsorted`, `clip`, `any`/`all`, `take`, `isin`, and operations on stepped slices such as `a[::2]`.
- Measure NumPy's fixed per-call overhead on tiny arrays separately, since that is a different kind of win.
- Rank candidates by how far NumPy is behind C, and by whether a known SIMD technique exists that the compiler will not find.

**Exit criterion:** a ranked shortlist with numbers. Every candidate on it goes forward to phase 3; none is dropped in advance.

**Status:** a first pass is in [survey-design.md](survey-design.md): ten assembly candidates and six C-only candidates for ARM64, with numbers, and a recorded run with its own shortlist for x86-64.

### Phase 3: the shortlisted kernels, end to end

Take the shortlisted kernels to `install()`, in the order given in [survey-design.md](survey-design.md). Each candidate is tried; the judge decides whether it ships as assembly, as C, or not at all.

**How:**

- Prove the delivery path with the first kernel before writing the others:
  - Build a real C extension module for the binding. No `ctypes`.
  - Establish which patching mechanisms work on current NumPy. There are three, and each reaches different calls:

    | Mechanism | Reaches | Does not reach |
    |---|---|---|
    | Replace module functions (`np.argmax`) | Function-style calls, including from other libraries | Method calls (`a.argmax()`), names imported before patching |
    | Replace NumPy's inner loops through its C API | Anything routed through a ufunc: operators, methods, reductions | Operations that are not ufuncs |
    | Decorator or own array type | Anything | Code the user has not opted in |

    **Known gap: indexing syntax and array methods.** `a[mask]`, `x[mask] = v` and `x.astype(...)` are not module functions and are not ufuncs, so none of the three mechanisms obviously reaches them without the user opting in. Their function forms are reachable: `np.compress`, `np.extract`, `np.putmask`, `np.copyto`. This matters because mask selection is the largest gap the survey found that the compiler does not close (see [survey-design.md](survey-design.md)). Until a way to reach the syntax is found, the mask kernels and the `astype` C kernels only speed up code that uses the function forms. Finding out whether the syntax can be reached is part of the first task of this phase.

    The second mechanism is recalled from NumPy's C API and from how Intel's `mkl_umath` works. It has not been checked against NumPy 2.x. Verifying it is the first task of this phase.

    **Checked since:** all three work on NumPy 2.5.5, and indexing syntax and methods can be reached by changing `ndarray`'s own type. The mechanisms, what each reaches and what each costs are in [delivery-design.md](delivery-design.md).
  - Implement `install()`, `uninstall()`, a context manager, and `report()`.
  - Implement the fallback path and measure its cost. A call that falls back must cost almost nothing extra.
- Produce each kernel with `/goal` runs against the judge, logging every attempt in `attempts.md`.

**Exit criterion:**

- Every shortlisted candidate has a recorded verdict from the judge: shipped as assembly (beats NumPy and naive C), shipped as C (beats NumPy), or rejected with the reason logged.
- An unmodified script runs faster after adding the one line, with output that meets each kernel's correctness mode, and `report()` shows the kernels being hit.
- The overhead of a patched call that falls back is measured and negligible.

**Status:** the kernels have had a first pass, recorded in [kernel-findings.md](kernel-findings.md): 22 of the 23 have assembly that beats naive C and one ships as C. The binding and `install()` are built as [delivery-design.md](delivery-design.md) describes: 21 of the 23 kernels are installed, and that document holds the result against this exit criterion. The two search kernels are not bound until they take a second size. Kernels added since from a second shortlist are recorded in [second-shortlist-design.md](second-shortlist-design.md).

## Non-goals

- Reimplementing all of NumPy.
- Beating BLAS at matrix multiplication. Apple's Accelerate already uses dedicated hardware for it.
- Multithreading. Single-core performance only, to keep measurements clean.
- Linux ARM64.

## Later

Mentioned so they are not forgotten. None of these is planned here.

- **AVX-512.** A second tier of x86-64 kernels, on a machine that has it. The first tier is designed in [x86-64-design.md](x86-64-design.md).
- **Publishing.** Wheels on PyPI so installation needs no compiler, plus `python -m hotkernels script.py` for zero code changes. The release pipeline and its GitHub setup are proposed in [release-design.md](release-design.md).
- **More kernels.** Work down the survey shortlist.
- **Fusion.** Fusing a chain of operations into a single pass was the largest win measured. It first needs a way to identify which chains are worth fusing, which is a project of its own. The applications study offered the gradient step `w -= lr * g` as the first such chain, and [gradient-step-findings.md](gradient-step-findings.md) measured it: fusing it would make the training loop that showed it 4% faster, no hook can reach it (the multiply has run before the subtraction is seen, and deferring it is unsound for a real `ndarray`), and the cost the study had read as two passes was the in-place subtraction after a threaded matrix product, which a loop of the binding's own now removes on ARM64. Fusion of statements written with operators stays out of reach of the hooks; an explicit fused function would need the user to change their code, for a gain measured at 4% to 5%. A lazy array type the user opts into with one line, which sees a whole expression before anything is computed, is proposed in [lazy-design.md](lazy-design.md); it designs the graph and leaves fusion for after it.

## Decisions made in review

| Question | Decision |
|---|---|
| Package and import name | `hotkernels` (`kernel-arena` is taken on PyPI) |
| Licence | MIT |
| Bit-identical always? | Per kernel: exact by default, declared tolerance for float arithmetic |
| How hands-on with the assembly? | Gains first, via `/goal` iteration; understanding later |
| Other platforms | x86-64 on Linux, designed in [x86-64-design.md](x86-64-design.md); Linux ARM64 out of scope |
| Agent loop as its own phase? | No; it is the judge, the contract and `AGENTS.md` |
| Ship plain C kernels? | Yes, when C beats NumPy and assembly does not beat the C |
| How many kernels in phase 3? | All shortlisted candidates are tried, not a chosen three |
| Float `cumsum` that reorders additions? | Acceptable, in tolerance mode. `report()` marks it as inexact. Since narrowed in [delivery-design.md](delivery-design.md): installed only with `install(inexact=True)` |
| Indexing syntax and methods (`a[mask]`, `x.astype`) | Recorded as a known gap in the delivery path; to be investigated at the start of phase 3 |
