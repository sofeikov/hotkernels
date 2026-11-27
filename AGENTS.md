# Working in this repository

hotkernels is a set of kernels for NumPy operations, hand-written assembly where it beats the compiler and plain C where it does not, checked and timed by an automated judge. Read [docs/core-design.md](docs/core-design.md) for the project, [docs/judge-design.md](docs/judge-design.md) for the judge [docs/delivery-design.md](docs/delivery-design.md) for how the kernels reach NumPy and [docs/release-design.md](docs/release-design.md) for how releases are made. The kernels exist for two architectures: ARM64 on macOS, and x86-64 on Linux, written to the plan in [docs/x86-64-design.md](docs/x86-64-design.md).

This file is for coding agents. The first half applies to any work here; the second half applies when you are writing or tuning a kernel.

## General rules

- Design changes go through a pull request with a document in `docs/` before any code is written.
- Run everything through `uv`: `uv run judge <command>`. There is no other setup.
- `uv run judge selftest` must pass before any change to `judge/` is proposed.
- `uv run delivery check` must pass before any change to `hotkernels/` or to a kernel that is installed is proposed. It runs every hooked NumPy call with and without hotkernels and compares them.
- Timings are only meaningful on native hardware. Never report numbers taken under emulation.
- A pull request that changes what a user of the package gets leaves a release note. See [Release notes](#release-notes).
- READMEs describe the present state only: what exists and how to use it. No phases, status, roadmap or development history. Plans belong in `docs/`, and history belongs in git. When a change makes a README inaccurate, update it in the same change.
- A change is not finished until every text that describes it is right. See [Documentation moves with the change](#documentation-moves-with-the-change).

## Documentation moves with the change

The documentation and the explanations written for users are part of the change, not a follow-up. Update them in the same pull request as the code, every time, without being asked.

**Find what the change made untrue.** Search the repository for the kernel's name, the NumPy call, and any count or figure the change moves (`grep -rn "20 of the 22" --include='*.md' .`). A statement that was true before the change and is false after it is a defect of the change, wherever it is written. The interactive figures in `docsite/static/widgets/` hold rules and numbers as code, so search them too (`grep -rn "compress_f32" docsite/`).

**Where to look.**

| The change affects | Update |
|---|---|
| Which NumPy calls a kernel takes, or the conditions for it | `docsite/pages/start.md` ("What gets faster", "Things worth knowing"), `hotkernels/README.md`, the hook table in `docs/delivery-design.md`, `docsite/pages/report/delivery.md` (the table in "One call, several hooks"), and the figure that traces a call through the hooks, `docsite/static/widgets/router.js`: a new call gets an entry there, and a changed condition, element type or kernel changes its entry |
| A hook: one added or removed, a new kind of hook, what a hook changes inside NumPy, CPython or another library, which spellings reach it, or whether it releases the interpreter lock | `docsite/pages/report/delivery.md` (the diagram and the table of kinds in "Where the hooks sit", "Each hook, exactly", "One call, several hooks", "Threads", "Installing is a test"), the kinds and entries in `docsite/static/widgets/router.js`, the hook table in `docs/delivery-design.md`, the header comment of the hook's file in `hotkernels/src/hotkernels/_binding/` |
| The Python interface: an argument or behaviour of `install()`, `uninstall()`, `enabled()` or `report()`, a field of `Report`, what is logged | The interface table and "See what it did" in `docsite/pages/start.md`, `hotkernels/README.md`, the "Using hotkernels" section of `README.md`, "What the user gets" in `docs/delivery-design.md`, "Installing is a test" in `docsite/pages/report/delivery.md` |
| What `delivery check` runs: cases or operations added or removed | The total it prints, in `README.md` ("Checked from Python"), `docsite/pages/start.md` ("Checking it yourself"), `docsite/pages/report/delivery.md` ("Testing the binding") and the diagram in `docsite/pages/report/index.md` |
| How many kernels exist, are installed, or are bit-identical | `README.md`, `hotkernels/README.md`, `docsite/pages/index.md`, the report chapters under `docsite/pages/report/` that quote the count |
| A speed figure | `README.md` (the docsite reads its tables from there), `docs/kernel-findings.md`, `delivery/results/` for figures taken from Python. Only figures from native hardware, with the size they were measured at |
| How a kernel works, what it requires of its caller, or a limit it has | `docs/kernel-findings.md`, the kernel's chapter under `docsite/pages/report/`, the header comment of its `kernel.s` |
| The judge: the spec format, a command, a field of a case description | `docs/judge-design.md`, `kernels/README.md`, the kernel half of this file, `docsite/pages/report/judge.md` |
| What `report()` prints | The sample output in `docsite/pages/start.md` |
| An interactive figure's subject: anything a figure in `docsite/static/widgets/` encodes, which its header comment names | The figure itself. Open the page after `uv run docsite serve` and try the controls, on a narrow window as well, and the Play button where there is one |
| The tools: a command added, renamed or removed | The "Using ..." section of `README.md` |

**Design documents.** A design in `docs/` is written before the code. When the building departs from it, say so in the document in the same pull request (a "What changed in the building" section, as in `docs/delivery-design.md`), and resolve what it listed as open. Do not leave a design that describes something other than what was built.

**Explain it to a user.** Where the change is visible to someone who calls NumPy, the user-facing pages (`docsite/pages/start.md`, `hotkernels/README.md`) say what they get and what they must know, in their terms: the call they write, not the hook or the file that serves it. This is in addition to the release note, which is one bullet; the pages are where the conditions and the caveats go.

**Say what you updated.** The pull request description lists the documents changed, or says that the change makes no existing statement untrue and how that was checked.

`uv run docsite build` must succeed after any change to `docsite/`, to `README.md`, or to a file the site reads its figures from.

## Python code standards

These apply to all Python in this repository.

- **Command lines use `typer`.** Never `argparse`. The one exception is code that ships in the `hotkernels` package (`python -m hotkernels`): it reads `sys.argv` by hand, so the package gains no command-line dependency. See [docs/distribution-design.md](docs/distribution-design.md).
- **Output uses the `logging` module.** Never a bare `print`. Each module gets `log = logging.getLogger(__name__)`; the CLI configures the handler.
- **Generated text uses Jinja templates**, wherever it is produced: reports, tables, Markdown, progress lines, generated C. Each package keeps its templates in its own `templates/` directory (`judge/templates/`, `survey/templates/`). Python passes values in; it does not assemble multi-line or column-aligned text with string formatting, and it does not build a report out of consecutive `print` or `log` calls.
- **Every module, class and function has a docstring**, including private and nested ones. Say what it does and anything a reader could not guess from the signature.
- **Docstrings describe every parameter and the return value**, in `Args:` and `Returns:` sections: what each one means, its unit where it has one, and any constraint on it. A one-line docstring is enough only for a function with no parameters and an obvious result, or a nested function a few lines long.
- **Every dataclass field has a comment** directly above it saying what it means.
- **No magic constants.** Numbers and strings with meaning get a named module-level constant, with a comment explaining the value where it is not obvious.
- **Comment the non-obvious.** Where code depends on something outside itself, such as a calling convention, a file format or a memory layout, explain it at the point of use.
- **Small functions over closures that mutate outer state.** Return values; do not use `nonlocal` counters.

## Release notes

A release gathers short notes left by the pull requests since the last one. You write the note for your pull request.

**When a note is needed.** Your pull request changes `hotkernels/`, or a kernel the package installs (its `kernels/<name>/<arch>/kernel.s`, `naive.c` or `spec.py`), and a user of the package would notice: a call served by a new kernel, a faster kernel, a fixed result or crash, a supported Python or NumPy version gained or lost, different `report()` output. Changes that touch only `judge/`, `survey/`, `delivery/`, `docsite/`, `release/`, `docs/` or a README need no note. CI checks this with `uv run release check-note`; run it yourself before pushing.

**When it is not.** A change in those paths that no user would notice, such as a comment, a rename or a refactor with identical behaviour, gets the `no release note` label instead. Say in the pull request description why.

**Where it goes.** One file per pull request, `release-notes/upcoming/<PR number>.<type>.md`. The number is only known once the pull request exists, so open the pull request first and add the note in a second commit. The type is one of:

| Type | For |
|---|---|
| `kernel` | A NumPy call or element type that a kernel now serves |
| `speed` | A kernel or the binding got faster |
| `fix` | A wrong result, a crash, or a call that went to the wrong place |
| `compat` | A Python or NumPy version gained or lost, a new platform, a change to what `install()` installs by default |
| `change` | Anything else a user would notice |

If a pull request does two of these, pick the one a user cares about most. One note per pull request, never one per commit.

**How to write it.**

- One to three sentences, plain Markdown, no heading. It becomes one bullet in the release notes.
- Write for someone who calls NumPy, not for someone who works on this repository. Name the NumPy call and element type (`np.cumsum` on `int32`), not the kernel's file name or the hook that serves it.
- Say what changed for them, not how. "`np.clip(x, lo, hi, out=x)` gave wrong results on some data and now goes to NumPy", not "added an aliasing check in hook_loop.c".
- A speed figure is allowed if it comes from `uv run judge run` or `uv run delivery time` on native hardware. Give the size it was measured at. It will be re-measured at release. Once the package ships for more than one architecture, also name the one the figure was measured on.
- Do not mention the pull request number in the text; the release notes add it.

Example, `release-notes/upcoming/31.kernel.md`:

```markdown
`np.cumsum` on `int32` is served by a kernel: about 11x faster than NumPy at a million elements.
```

**Releases are not yours to make.** Do not run `uv run release prepare` or `uv run release tag`, change the package version, or push tags, unless you are asked to.

## Writing or tuning a kernel

### The one rule

Keep a change to a kernel only if both of these hold:

1. `uv run judge check <kernel>` passes.
2. `uv run judge compare <kernel> --against HEAD` reports `RESULT: significantly faster`.

If either fails, revert the change. Commit each kept change before trying the next, so `HEAD` is always the best version so far.

### Files you may edit

During a kernel run you may change exactly two files:

- `kernels/<name>/<arch>/kernel.s`
- `kernels/<name>/<arch>/attempts.md`

A fused kernel of the lazy layer's prototype is in `prototypes/lazy/kernels/<name>/` instead, laid out the same way and found by the judge under the same name. The same two files are yours there, and everything below holds for it; what a fused kernel is held to beyond that is in [prototypes/lazy/kernels/README.md](prototypes/lazy/kernels/README.md). It is not installed by the package, so it never needs a release note. A fused kernel that `prototypes/lazy/make_kernel.py` wrote from the name of its shape is tuned the same way: its `kernel.s` is yours from then on, and its log says what the tool wrote and what you changed.

When the run is over and its pull request is open, and the kernel is one the package installs, add the run's one release note: a `speed` note summarising it, as described in [Release notes](#release-notes). That is the only other file a run creates.

Everything else is locked: `spec.py`, `naive.c`, and all of `judge/`. The judge compares those against the `main` branch and fails the run if they differ. Do not pass `--unlocked`; it exists for people creating a kernel or changing the judge, and it marks the results as untrusted.

If you believe the spec, the naive reference or the judge is wrong, stop and say so. Do not work around it.

### Commands

| Command | Use it for | Time |
|---|---|---|
| `uv run judge check <kernel>` | After every edit. Lint and correctness. | About 5 seconds, up to 25 for a kernel with many scalar sets |
| `uv run judge compare <kernel> --against HEAD` | After `check` passes. Is this edit faster? | About 30 seconds |
| `uv run judge run <kernel>` | The full report against naive C and NumPy. | About 30 seconds |
| `uv run judge check <kernel> --case '<case>'` | Rerun one failing case. | Instant |

`compare` and `run` both run `check` first.

The judge decides where a timed call's output buffer sits relative to its inputs: clear of them. For a kernel with an output buffer, `run` prints a second table with the output 16 bytes behind its input, a position callers often hit and where a loop can stall on x86-64. It plays no part in `compare`. If a kernel is slower there than with the output clear, say so in `attempts.md` with the table's figures; an attempt aimed at that table is judged by `run`, since `compare` does not see it.

### Reading a failure

```
FAIL argmax_f32 (arm64): wrong result
  wrong return value
  case:     n=2 arr=mid2 seed=1 x=nan_at@1
  expected: 1
  got:      0
  rerun:    uv run judge check argmax_f32 --case 'n=2 arr=mid2 seed=1 x=nan_at@1'
```

Only the first failure is shown. The case line says how the input was built:

- `n` is the element count.
- `arr` is where the buffers sit in memory:
  - `end`: the buffer ends exactly at an inaccessible page.
  - `start`: the buffer starts exactly after one.
  - `mid1` to `mid3`: deliberately misaligned.
- `x=nan_at@1` means argument `x` is random data with a NaN at index 1. Other value classes are named the same way (`max_at`, `neg_zero_at`, `ties`, `denormal`, `all_true`, and so on); see `judge/cases.py`.
- `v=from_a` means the elements of `v` were drawn from `a` as the kernel received it, so some of them equal elements of `a`: the needles of a search, where a tie decides the answer. The spec declares which argument may be drawn from which.
- `scalars=zero_lo`, when present, names the set of scalar arguments the call was made with. The sets and their values are listed in the kernel's `spec.py`; without the field the scalars had their default values. A kernel must be right for every set, and `compare` times every set marked `timed`.

The failure categories:

| Category | Meaning |
|---|---|
| `lint` | The source breaks a static rule (see below). Nothing was run. |
| `build` | The assembler rejected the file. |
| `wrong result` | The output differs from NumPy's. |
| `memory safety` | The kernel touched memory outside its buffers, wrote to an input, or wrote past the output it reported using. A crash with `SIGBUS` or `SIGSEGV` is almost always this. |
| `calling convention` | The kernel changed a register it must preserve. |
| `crash` | Any other crash, or the kernel did not finish. |

### What a kernel must be

A kernel is a pure leaf function over the buffers it is given. The judge enforces all of this:

- **No reads or writes outside the buffers, by even one byte.** This is strict: loading a full vector that extends past the end of an array is a failure, even though it would usually be harmless. Handle leftover elements inside the buffer, for example with a scalar loop, or with a final vector that overlaps the previous one and ends exactly at the end of the array.
- **Inputs are read-only.** Writing to an input buffer crashes. A buffer the spec declares `InOut` is the exception: the kernel reads and modifies it in place, and may store an element's own bits back to it.
- **Write only the output you report.** If the kernel returns a count, it must not write beyond that many elements. The one exception is an output the spec declares `scratch=True`: there the kernel may write anywhere inside the buffer's capacity, and only the reported elements have to be right. Writing past the capacity is always a failure. A scratch output can be larger than `n` (`capacity=Times(3)`) when the kernel is lent working memory that way. Check `spec.py` before relying on this; the default is strict.
- **No function calls, no system calls, no writable data.** Read-only constants are fine; the section they go in is listed under the architecture below.
- **Export exactly one symbol:** the kernel itself.
- **Match NumPy exactly** unless the spec declares a tolerance. This includes NaN, infinities, signed zeros and which element wins a tie.

### More than one architecture

`spec.py` and `naive.c` are shared. Each architecture has its own `kernel.s`, and the judge builds the one that matches the machine it runs on. Kernels exist for both, ARM64 on macOS and x86-64 on Linux; what was tried on each is in the `attempts.md` beside each `kernel.s`, and what was found in [docs/kernel-findings.md](docs/kernel-findings.md).

- **The other architecture's kernel and log are reading material, not a template.** Take the idea and the proofs from them. Choose the instructions, block sizes and paths for the processor you are on.
- **A dead end there is not a dead end here.** An attempt rejected on another architecture was rejected on that processor. Do not skip it because of that log; try it and record what the judge says.
- **Say which architecture a statement is about.** In `attempts.md` and in `docs/`, a limit such as "the core loads three vectors per cycle" belongs to one processor. Name it.
- **A change that removes a caveat or needs a rule changed is not yours to keep.** If a kernel on your architecture would not need a constraint the other one has, or would need the judge to allow something it forbids, report it with the judge's figures and stop.

### ARM64 on macOS

- The symbol for C function `foo` is `_foo`. Declare it with `.globl _foo` and align with `.p2align 2`.
- Labels that start with `L` are local to the file. Use them for everything except the kernel symbol.
- Arguments arrive in `x0` to `x7` (integers and pointers) and `v0` to `v7` (floating point). The result goes in `x0` or `v0`. The two kinds are numbered separately: in `f(double *x, size_t n, double lo, double hi)` the pointer is `x0`, `n` is `x1`, `lo` is `d0` and `hi` is `d1`.
- A leaf kernel may use stack memory below `sp` if it restores `sp` before returning and keeps it 16-byte aligned while it is moved.
- You may freely use `x0` to `x17` and `v0` to `v7`, `v16` to `v31`.
- You must preserve `x19` to `x28`, `x29`, the stack pointer, and the low 64 bits of `v8` to `v15`. Prefer not to use them at all.
- Never use `x18`. The platform reserves it.
- `//` and `;` both start a comment.
- Read-only constants go in `.section __TEXT,__const` or `__TEXT,__literal16`.
- Unaligned vector loads and stores are allowed and cost nothing extra on this hardware. Buffers are not guaranteed to be aligned.
- Kernels are only ever called with `n >= min_n` from the spec.

### x86-64 on Linux

- The file starts with `.intel_syntax noprefix`. It is GNU assembler syntax, and the judge assembles it with GNU `as`.
- The symbol for C function `foo` is `foo`. Declare it with `.globl foo` and `.type foo, @function`, and align with `.p2align 4`.
- Labels that start with `.L` are local to the file. Use them for everything except the kernel symbol.
- Integer and pointer arguments arrive in `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`, in that order, and floating-point arguments in `xmm0` to `xmm7`. The result goes in `rax` or `xmm0`. The two kinds are numbered separately: in `f(double *x, size_t n, double lo, double hi)` the pointer is `rdi`, `n` is `rsi`, `lo` is `xmm0` and `hi` is `xmm1`. A seventh integer argument is on the stack at `[rsp + 8]`.
- You may freely use `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8` to `r11`, and all of `xmm0` to `xmm15` and `ymm0` to `ymm15`. No vector register is preserved across a call.
- You must preserve `rbx`, `rbp`, `r12` to `r15` and the stack pointer. Prefer not to use them at all. Leave the direction flag clear and the rounding mode and exception masks in MXCSR as you found them.
- **Run `vzeroupper` before every `ret` if the kernel wrote any `ymm` register.** The judge asks the processor whether the upper halves are still in use and fails the kernel as `calling convention` if they are.
- A leaf kernel may use the 128 bytes below `rsp` without moving it. If it moves `rsp`, it restores it before returning and keeps it 16-byte aligned.
- **Use only x86-64-v3 instructions:** AVX2, FMA, BMI1, BMI2, LZCNT, MOVBE, POPCNT and everything older. No AVX-512, and nothing newer than the level even where this machine runs it (AVX-VNNI, GFNI, VAES, SHA, ADX). No MMX and no x87. The judge refuses anything else as `lint`.
- `#` starts a comment, and so does `/* */`. `//` does not.
- Read-only constants go in `.section .rodata` and are addressed relative to `rip`: `[rip + .Lconstants]`.
- End the file with `.section .note.GNU-stack,"",@progbits`. Without it the library would need an executable stack, and the judge fails the kernel as `lint`.
- Unaligned vector loads and stores are allowed, and buffers are not guaranteed to be aligned. Do not assume they are free: a 32-byte access that straddles two cache lines costs more than one that does not. Measure.
- `pext` and `pdep`, gathers and masked stores are allowed. They are fast on this machine and slow on some other x86-64 processors, so name each use in `attempts.md`.
- **An output just behind the input stalls a loop.** If the output buffer starts 8 to 128 bytes after the input, counted within a 4 KB page, the processor takes each store for a store to the elements about to be loaded. The second table of `run` times exactly that. A kernel may test the two addresses on entry and take a path that does not stall.
- The timing thread is pinned to one performance core. Several judges can run at once and each takes a different core, but they still share the last-level cache and memory, so sizes from about a million elements up are only trustworthy when nothing else is running.
- Kernels are only ever called with `n >= min_n` from the spec.
- **What this processor costs,** measured while the first kernels were written (the Intel Core i5-14600KF; [docs/kernel-findings.md](docs/kernel-findings.md#x86-64-1)): a `lea` with a scaled index takes two cycles, so keep a position that advances every step as a count and add to it. `vblendvpd` and `vpblendvb` run once per cycle; three logical instructions run three per cycle. Compares, maxima and float adds have two of the three vector ports. Two 32-byte loads per cycle, and one store per cycle unless two go to the same cache line. A 32-byte load that straddles two cache lines costs a cycle more, a store two. Beyond the first-level cache a loop that writes as much as it reads runs at the speed of a copy, about 0.115 ns per eight bytes, whatever its arithmetic.
- **Load ahead of the stores** in a kernel that reads one buffer and writes another of the same size: load the next vectors before storing the current ones. That is what keeps the close-placement table of `run` level.

### The attempts log

Record every attempt in `kernels/<name>/<arch>/attempts.md`, beside the `kernel.s` it is about, including the ones you reverted. Each architecture has a log of its own, numbered from 1. Later runs read this file so they do not repeat dead ends; the other architecture's log is reading material, as its kernel is. A kernel that ships as C on an architecture keeps its log there all the same (`kernels/cast_u8_f32/arm64/attempts.md`). One entry per attempt:

```markdown
## 7. Unroll the main loop to four vectors per iteration

- **Idea:** hide the latency of the compare by keeping four independent maxima.
- **check:** pass
- **compare:** significantly faster, 1.21x [1.19 – 1.23]
- **Kept:** yes
- **Notes:** gain is all above 1,024 elements; small sizes unchanged.
```

For a rejected attempt, say why in one line: failed `check` (and how), not significantly faster, or significantly slower.
