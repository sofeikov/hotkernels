# x86-64: design

This began as a proposal for review. Every step of its [order of work](#order-of-work) has been carried out: the judge and the survey run on Linux x86-64, all 23 kernels have an x86-64 verdict, the package builds, installs and passes `delivery check` there, and the findings are written up. What is left is a person's: the decisions listed in [kernel-findings.md](kernel-findings.md#decisions), one reference correction to review, and [open issue 6](#open-issues). It details the "x86-64 kernels" item under [Later](core-design.md#later) in the core design: which x86-64 to target, what has to change in the judge, the survey and the package, how the existing kernels are carried over, and how the existing documents are amended so that they explain two architectures and not one.

## Summary

- **Target: x86-64-v3 on Linux.** AVX2, FMA, BMI1, BMI2, LZCNT and MOVBE, with 256-bit vectors. It is the most the development machine can run as a named level, and it is the level NumPy itself dispatches to on that machine.
- **Assembly is GNU assembler in Intel syntax**, in `kernels/<name>/x86_64/kernel.s`, built through the C compiler as on ARM64.
- **A kernel is carried over in spirit, not line by line.** `spec.py` and `naive.c` are shared. The assembly is written again for the same contract, starting from what makes the operation slow and not from the ARM64 instructions.
- **x86-64 has techniques ARM64 does not, and lacks some ARM64 has.** Which of them pay was [open issue 1](#open-issue-1-techniques-particular-to-x86-64), a list of candidates per kernel family that every x86-64 kernel run had to try and report on. It is closed: [what became of each](#what-became-of-the-candidates).
- **The speedups are not the ARM64 ones.** NumPy already uses AVX2 for `argmax`, `clip` and `astype` on x86-64. The measured figures are in [kernel-findings.md](kernel-findings.md#x86-64); they came out close to what the first measurement suggested, which was that the mask, search, histogram and `sum` kernels can match or beat their ARM64 figures against NumPy, the `cumsum` kernels reach about half, and `argmax`, `clip`, `count_nonzero` on `int64` and the `uint8` cast gain little or nothing outside the caches.
- **The existing documents are amended first.** They state ARM64 reasoning as if it were general. Each is restructured to separate what is true of an operation from what is true of one architecture, before any x86-64 result is written into it.

## What changed in the building

The judge and the survey now run on Linux x86-64 (steps 3 and 4 of the [order of work](#order-of-work)). Where the building departed from the proposal:

| Proposed | Built | Why |
|---|---|---|
| Kernels assembled by the C compiler driver | By GNU `as` directly | It is the assembler that can be told which instructions exist. Stated with the level, it refused every instruction beyond x86-64-v3 that was tried and accepted the level's own. clang's assembler does not: given `-march=x86-64-v3` it assembled a GFNI instruction and an AVX-512 one without complaint |
| The level check: preferred the assembler, otherwise a disassembly | The kernel is assembled twice: plainly, where a failure is a build error, then with the level stated, where a failure is lint and names the instruction | The two failures mean different things to whoever reads them |
| Whether the wrapper can ask the processor about `vzeroupper` was to be confirmed | It can, with `xgetbv`. A 256-bit write sets the bit even when it writes zeros, and `vzeroupper` clears it | Measured on this machine. Where a processor cannot report it the check is skipped |
| The timing thread pinned in the C harness | Pinned from Python, to the highest-numbered performance core that no other judge holds, one thread per physical core | Several agents run judges at once. Without a claim they would all pin to the same core |
| Noise: governor, turbo, load average | Turbo off, a governor or energy preference that saves power, a load average of 2 or more, a shared core | "powersave" is the normal governor under the drivers that let the processor manage itself, and is only a problem under the others |
| Not mentioned | Every kernel must end with the note that it needs no executable stack | Current Linux refuses to load a library that needs one |
| Not mentioned | The wrapper passes a seventh and eighth integer argument on the stack | The convention has six integer registers, and the judge's spec allows eight arguments |
| The attempts logs moved beside each kernel in this step | Done later, in a pull request of its own after the kernels: `kernels/<name>/<arch>/attempts.md` | It touches every kernel directory and the documentation site, and is easier to review alone |
| The survey ported by changing its clock and scheduling calls, its flags and its reading of the vectoriser remarks | The remarks needed no change, since clang is the compiler on both. The survey takes the compiler, the flags, the library suffix and the core claim from the judge | One place decides how C is built and where a timing thread runs. A survey and a judge run never share a core |
| Not mentioned | The survey records which vector instruction sets NumPy uses, the pinned CPU, and the judge's warnings about turbo, power saving and load | NumPy's own loops are the other contender, and which ones it has is what differs between architectures |
| Not mentioned | The survey places each output buffer half a page from its input | Left to the allocator the output often sat 16 bytes after the input within the page, which made NumPy up to 8x slower and the ratios meaningless. See [open issue 8](#open-issues) |
| 30 samples of 10 ms per size | 300 samples of 1 ms | With a desktop session running, sizes larger than a core's own cache are slowed by other cores' memory traffic in bursts of a few milliseconds. Two copies of the same code differed by up to 17% at 4 MB and 16 MB, and by 0.1% and 0.3% with short samples. See [judge-design.md](judge-design.md#short-samples) |
| `compare` in one process | `compare` in five rounds, each in a process of its own, with the interval taken from the spread between rounds | Where a build is loaded changes its speed by up to 0.7% overall. Within one process that is the same in every sample, so identical code was called significantly different. See [judge-design.md](judge-design.md#comparing-two-versions-of-a-kernel) |
| Not mentioned | The judge places the output buffers of every timed call clear of the inputs, and `run` adds a table that times the output 16 bytes behind its input | The same finding, in the judge: its naive C for `cumsum_i64` took four times as long at the one size where the allocator had put the buffers 16 bytes apart. See [judge-design.md](judge-design.md#where-the-output-buffer-sits) |

The kernels, the package and the findings (steps 5 to 7). Where the building departed from the proposal:

| Proposed | Built | Why |
|---|---|---|
| The attempts logs moved beside each kernel, one per architecture, before the kernels | The kernel runs appended an "x86-64" part to each shared log; the logs were split into `kernels/<name>/arm64/attempts.md` and `kernels/<name>/x86_64/attempts.md` afterwards, in one pull request | A kernel run may edit two files, and moving 23 logs is a change of its own. The documentation site counts the ARM64 logs as before, 188 attempts, and the x86-64 ones for its chapter on that architecture |
| One family per run, each a `/goal` run of its own | One agent, family after family, in one session; one pull request per family, stacked | It was asked for that way. The rule per change was the same |
| A kernel's findings written by the pull request that produces them | Written once, at the top of the stack | A kernel run edits two files |
| Not mentioned | The documentation site has a chapter on the second architecture, with its figures read from the findings table, the recorded Python timings and the x86-64 logs | The site described ARM64 only |
| `cast_u8_f32` last and probably dropped; `clip` shipping as C | Both have assembly, by small margins | The compiler does not emit the two-instruction `clip`, and stores past the caches help both at the largest size. The verdicts are a person's to confirm |
| `interp_f64` like any other kernel | Its reference had to be corrected first, in a pull request of its own, and its kernel was written and judged unlocked on top of it. Its recorded figures are from a locked run on `main` after both were merged | NumPy's x86-64 build does not fuse the multiply and the add, and the shared reference did ([issue 73](https://github.com/sofeikov/hotkernels/issues/73)) |
| `tools/generate.py` already picks the kernel directory by architecture | It always took ARM64's. Now it writes a kernel set per architecture, and a source distribution carries both | One source distribution serves both platforms |
| The run-time check asks the processor and the operating system | It reads `/proc/cpuinfo`, which lists `avx` and `avx2` only when Linux saves the 256-bit registers | No compiled code is needed before the check has passed |
| The wheel's compiler has to be clang | The Linux wheels are built by the manylinux image's gcc | No kernel shipped as C on x86-64, so the compiler built only the binding. Since `unique_i64`, which shipped as C there until it had assembly, as the `argsort` kernels and bottleneck's four still do, the build installs clang in the image and uses it ([release-design.md](release-design.md#more-than-one-platform)) |
| The release job for other platforms on Linux | Moved to Linux on ARM64 | A build on Linux x86-64 now needs a compiler |
| Not mentioned | `delivery time` names the machine and pins its thread as the judge does | It recorded "unknown machine" on Linux and was not pinned |

**Checked with clang.** The selftest passes with clang 21.1.8 as the compiler, which is what the design asks for, and continuous integration runs it on Linux. It was developed with gcc standing in, before clang was installed on this machine, and passes with both. The choice of compiler matters as much as [open issue 2](#open-issues) expected: the naive C for `sum_u8` takes 49 ns for 1,024 bytes built by clang and 127 ns built by gcc.

**Repeatability.** On this machine, with a desktop session running, five runs of the ten-run check gave worst spreads of 2.0%, 2.3%, 4.7%, 7.9% and 17.6% against a target of 3%. The last was taken while other tests ran on the machine. Sizes up to 1,048,576 elements stayed within 2.3% in four of the five. The misses were at 256 elements once, and otherwise at 4,194,304: sixteen megabytes of `float32`, which only just fits the 24 MB cache that every core shares, so anything else that uses the cache moves it. **Since the judge takes 300 samples of 1 ms** in place of 30 of 10 ms, the same check passes with the desktop session running: the worst spread over ten runs was 1.7%, at 16 and at 16,777,216 elements, and 0.4% or less from 256 to 1,048,576.

## The machine

| | |
|---|---|
| Processor | Intel Core i5-14600KF (Raptor Lake Refresh) |
| Cores | 6 performance cores with two threads each (CPUs 0 to 11), 8 efficiency cores (CPUs 12 to 19) |
| Clock | Up to 5.3 GHz |
| Caches | 2 MB second-level cache per performance core, 24 MB third-level cache shared |
| Has | AVX2, FMA, BMI1, BMI2, LZCNT, POPCNT, MOVBE; beyond x86-64-v3 also AVX-VNNI, GFNI, VAES, SHA |
| Does not have | AVX-512 |
| System | Linux 7.0, clang 21.1.8, GNU assembler 2.46. gcc 15.2 is installed and not used. No nasm |

## What was measured before writing this

The survey was run once on this machine to see where NumPy stands, before it was ported. It is not a recorded result. **A recorded run has since replaced it:** [survey-design.md](survey-design.md#results-on-x86-64), taken with clang. The NumPy figures below agree with it. The naive C figures do not all, because this first look built the C with gcc. The section is kept as it was written, as the basis the design was reviewed on.

- **How.** `uv run survey run`, pinned to one performance core, NumPy 2.5.5. The survey is not ported, so it ran through a temporary stand-in outside the repository that sent its `clang` calls to gcc 15.2 with `-march=native` and supplied the two macOS functions its C file calls. The reference loop drifted by 0.03% over the run.
- **What it is good for.** NumPy's time per element, and how far a compiled C loop is from it. Sizes from 1,048,576 elements up showed scatter between variants of the same operation, so those are read as rough.
- **What it is not.** A judge result. No x86-64 kernel existed when this was written, so every kernel figure below is an estimate from the instruction counts a kernel would need and from what the caches deliver.

| Family | NumPy here, ns per element at 1,048,576 | NumPy on the M1 Pro | Kernel against NumPy at 1,048,576, estimated | ARM64 kernel, measured |
|---|---|---|---|---|
| `sum_u8` | 0.21 | 0.19 | 25x to 40x | 15.7x |
| `compress_f32`, half selected | 3.4 | 5.0 | 25x to 40x | 41.7x |
| `compress_f64`, half selected | 3.4 | 4.75 | 13x to 20x | 22.5x |
| `nonzero_bool`, half / few selected | 0.42 / 0.69 | 0.71 / 0.85 | 4x to 6x / 15x to 25x | 5.15x / 19.2x |
| `maskfill_f64`, half selected | 3.0 | 3.85 | 20x to 30x | 21.4x |
| `isin_i64`, close test values | 6.3 | 7.86 | 40x to 60x | 75.6x |
| `histogram_f64` | 3.7 | 6.26 | 10x to 15x | 9.70x |
| `searchsorted_f64` | 100 per needle | 142 | 7x to 10x | 8.84x |
| `interp_f64` | 163 per query | 679 | 5x to 9x | 22.7x |
| `cumsum_i64` / `f64` / `f32` | 1.2 to 1.9 | 2.2 / 3.6 / 3.5 | 5x to 8x / 6x to 9x / 10x to 15x | 12.5x / 20.1x / 29.0x |
| `count_nonzero_f64` | 0.39 | 0.38 | About 4x | 3.79x |
| `count_nonzero_i64` | 0.12 | 0.33 | 1.1x to 1.3x | 3.25x |
| `argmax_f32` / `f64` | 0.08 / 0.17 | 0.43 / 0.71 | 1.5x to 2x | 8.82x / 6.86x |
| `clip_f32` | 0.08 to 0.10 at 65,536 | 0.67 | 1x to 1.5x | 8.67x |
| `cast_u8_f32` | 0.09 | 0.32 | None | 2.68x |

What the measurement says:

1. **NumPy is a different opponent here.** Per element it runs 1.3x to 5x faster than on the M1 Pro. Part is the clock. The rest is that NumPy has AVX2 loops for `argmax`, `clip` and `astype`, which were scalar or slow on ARM64. The ARM64 shortlist does not carry over unchanged.
2. **The operations NumPy leaves scalar are the same ones.** Mask selection, `nonzero`, masked assignment, `isin`, `histogram`, `cumsum`, `searchsorted` and `interp` are as far behind as on ARM64. These are where the ARM64 speedups can be matched.
3. **Memory is the ceiling sooner.** One core reads about 30 GB/s from main memory, and the second-level cache holds 2 MB. A kernel that is 4x to 6x ahead of NumPy inside the caches (`argmax`, `clip`, `count_nonzero_i64`) is level with it at 16,777,216 elements.
4. **The compiler matters more than expected.** gcc 15.2 left the survey's `argmax` and `clip` loops scalar and branching: `clip` on `float32` took 1.9 ns per element at 65,536 against NumPy's 0.08. With that compiler as the baseline, assembly would look better than it is. See [open issue 2](#open-issues).
5. **Fixed cost per call is as large as on ARM64.** 0.5 µs for `np.argmax` and 1.0 to 1.2 µs for `np.clip` on tiny arrays. Small sizes will win for the same reason they do on ARM64.

## Target

- **Instruction set: x86-64-v3.** Every kernel may use AVX2, FMA, BMI1, BMI2, LZCNT, MOVBE and POPCNT and nothing newer. Intel has had these since Haswell (2013) and AMD since 2015.
  - *Why not less.* The 128-bit SSE levels halve the vector width for a set of machines that is by now small.
  - *Why not more.* AVX-512 cannot run on this machine, and timings are only taken on native hardware. It would help the mask kernels most, since it has a compress instruction and masked operations throughout, and it belongs to a later design on a machine that has it.
  - *The level has to be enforced.* This processor also runs AVX-VNNI, GFNI and VAES, which are not in x86-64-v3. A kernel that used one would pass the judge here and crash on a Haswell. The judge must reject it statically; see [The judge](#the-judge).
- **Assembler: GNU assembler syntax with `.intel_syntax noprefix`**, assembled by the C compiler driver. Both gcc and clang accept it, so the toolchain stays the one the judge already uses. nasm would add a dependency to the judge, the package build and the wheels.
- **Platform: Linux, System V calling convention.** Windows passes arguments differently and preserves some vector registers, so the same file would not work there. Intel Macs use the System V convention but another object format. Both are out of scope.
- **At run time** `install()` checks that the processor has every instruction set of the level and that the operating system saves the 256-bit registers. Otherwise it does nothing and `report()` says why, as on any unsupported machine.

## Carrying a kernel over

`spec.py` and `naive.c` are shared by every architecture, so the contract is already written: the same inputs, the same oracle, the same correctness mode. What is new per architecture is `kernel.s` and the reasoning behind it.

The 9,603 lines of ARM64 assembly are not translated. An instruction-for-instruction port would inherit choices made for NEON's instructions, 128-bit vectors and the M1's core, and would miss what x86-64 offers. A kernel is carried over at the level of its idea:

- **What carries over.** Why the operation is slow in NumPy and in naive C; the shape of the fix (reduce blocks and locate the winner afterwards, store without asking and advance by the count, run many searches in step); what NumPy's semantics force (which tie wins, where a NaN lands, which bits a signed zero has); and the proofs (the `cumsum` overflow argument, the `histogram` error bound).
- **What does not.** The choice of instructions, block sizes, the number of accumulators, where the path for small arrays starts, table layouts, and every statement of the form "this is the hardware's limit".
- **Dead ends do not carry over either.** An attempt rejected on ARM64 was rejected on that core. Software prefetch, aligning the loop, non-temporal stores and larger tables were all slower or no faster there. Each is untested here and must not be skipped because an ARM64 log says so.

Three kinds of difference decide how a kernel changes:

| Difference | Examples |
|---|---|
| Instructions x86-64 has and NEON does not | A vector compare straight to a bit mask in a general register (`vpmovmskb`, `vmovmskps`); bit scans and bit extraction (`tzcnt`, `blsr`, `pext`, `pdep`); a sum of bytes in one instruction (`vpsadbw`); masked loads and stores (`vmaskmovps`, `vpmaskmovq`); gathers (`vgatherqpd`); a permute across the whole 256-bit register by a variable index (`vpermd`); a test of a whole register for zero (`vptest`); non-temporal stores; `rep stos` and `rep movs` |
| Instructions NEON has and x86-64-v3 does not | Table lookup across up to four registers (`tbl`); pairwise add-and-accumulate (`uadalp`) and pairwise maximum (`umaxp`); a maximum that returns NaN when either operand is one (`fmax`); a count of set bits per byte (`cnt`); reductions of a whole vector to a scalar (`addv`, `umaxv`); unsigned integer compares and 64-bit integer minimum and maximum |
| A different machine | 256-bit vectors, but most shuffles, shifts and packs work on the two 128-bit halves separately; 16 vector registers against 32, none of them preserved across a call; 9 freely usable general registers against 18; load and arithmetic can be one instruction; a higher clock, so a dependent chain of scalar operations is cheaper (a float add chain costs 0.38 ns per element here against 0.93 on the M1 Pro); a smaller memory bandwidth per core; performance and efficiency cores that differ far more than on the M1 |

## Open issue 1: techniques particular to x86-64

**Closed.** Each entry below was a hypothesis for the kernel run of its family to try, keep or reject under the usual rule, and record. The table is kept as it was proposed; [what became of the candidates](#what-became-of-the-candidates) follows it.

### Why it is an issue and not a list of instructions

Several candidates do more than make a loop faster. They would remove a caveat the ARM64 kernel has, change what the binding must guarantee, or touch a rule of the judge. Those have to be decided by a person, with a judge result in hand, and they need a place where they are tracked until then.

### Candidates per family

| Family | What the ARM64 kernel does | Candidates on x86-64 | What it would change |
|---|---|---|---|
| `argmax`, `argmin` | Reduces blocks with `fmax`, which returns NaN if either operand is one, so a NaN shows up in the block compare. Scans the winning block with scalar code | `vmaxps` returns its second operand when either is NaN, so a NaN can be lost and needs its own test; one `vcmpunordps` covers two vectors. The first position equal to the maximum is `vcmpeqps`, `vmovmskps` and `tzcnt`, with no scalar scan. The same pair finds the first NaN | The opponent is NumPy's AVX2 loop at 0.08 ns per element, not a scalar loop. Whether a block scheme beats it outside the first-level cache is the question |
| `compress`, `nonzero` | Eight mask bytes become an index into a 256-entry table of shuffle patterns for `tbl`. Sparse masks use a 64-bit mask and `clz`. Switches between the two by measured density | 32 mask bytes become a 32-bit mask in one `vpmovmskb`. `vpermd` moves eight `float32` or four `float64` to the front of the register by index. The index comes from a table, or from `pext` with no table at all. `popcnt` advances the write pointer. Sparse masks: `tzcnt` and `blsr` per set bit. A block whose mask is all zeros or all ones is recognised with one compare | If the dense path is cheap enough the switch by density can go, and with it the weak spot near 10% that the judge does not time. `pext` is slow on AMD processors before Zen 3: see [open issue 4](#open-issues) |
| `maskfill_f64` | Loads 16 elements, blends in the value, stores all 16. Unselected elements are rewritten with their own bits | `vmaskmovpd` stores only the selected elements. Or the same load, blend and store | A masked store does not touch unselected elements. Decision 1 in [kernel-findings.md](kernel-findings.md#decisions) would then not be needed on x86-64, and the binding's constraint would differ by architecture. Masked stores are reported to be slow on some AMD processors |
| `isin_i64` | Narrows each element to a 16-bit key with saturation, then four `tbl` lookups on its nibbles. Otherwise eight 64-bit compares per two elements, which ties the compiler | `vpshufb` is the same nibble lookup. `vpackssdw` narrows with saturation, within each 128-bit half. `vpcmpeqq` compares four elements at once | The general case, which "cannot be beaten" on ARM64 because no wider compare exists, has one here. Whether the compiler already emits it decides if assembly wins there |
| `searchsorted`, `interp` | Branch-free binary search for many needles in step, one scalar probe load per needle | `vgatherqpd` loads four probes in one instruction and the compare and the move of the base stay in vectors. Or scalar loads with `cmov`, sixteen or more searches interleaved. `interp` divides four per vector with `vdivpd`. NumPy's x86-64 build may not fuse the multiply and the add as its ARM64 build does, and the kernel has to do whatever NumPy does | Gathers are slow on older AMD processors and on Intel processors that carry the gather mitigation. Which form is faster here, and how much the gather form loses elsewhere, is unknown |
| `histogram_f64` | Two elements per vector. A lane that is out of range, NaN or in the last bin goes to scalar code | Four per vector. `vcvttpd2dq` returns the same value, 0x80000000, for NaN and for anything too large to convert, so a NaN needs no test of its own: it fails the same unsigned compare as an index past the last bin. There is no scatter, so the counter updates stay scalar | Fewer special cases. The error-bound proof carries over unchanged and must be rechecked against the instruction that rounds here |
| `bincount_u8` | Eight input bytes per load, four count tables on the stack so equal neighbours do not wait on each other | The same scheme with an increment that addresses memory directly. `rep stosq` to clear the tables. The number of tables is retested | Small. The limit is one counter update per element on both architectures |
| `cumsum` | Prefix sums inside a block from the inputs alone, the running total added to a whole block. Floats: a sum of squares per chunk proves no overflow | The byte shift `vpslldq` works inside each 128-bit half, so a prefix sum across the register needs an extra permute. Non-temporal stores above the third-level cache. The overflow test could be a compare and a `vmovmskpd` instead of a sum of squares | The scalar chain is 0.19 ns per element for `int64` here. The vector scheme has to beat that with the extra permute, which is not certain |
| `sum_u8` | `uadalp` into 16-bit accumulators, sixteen per step, widened every 128 steps | `vpsadbw` against zero adds 32 bytes into four 64-bit sums in one instruction. No widening schedule and no overflow bookkeeping | Simpler and expected to be faster than the ARM64 kernel relative to NumPy. The first kernel to write, to prove the pipeline |
| `count_nonzero` | Three levels of `umaxp` narrow eight vectors to sixteen bytes before one test | `vpcmpeqq` against zero and a subtract into an accumulator. Or `vmovmskpd` and `popcnt`. For floats the sign is shifted out first | Level with NumPy above the second-level cache whatever is done |
| `clip` | `fmin(fmax(x, lo), hi)` differs from NumPy on signed zeros and signalling NaNs. So: a path chosen by the bounds, a chunk redone when a -0.0 is met, and a read of the floating-point status register | `vmaxps` and `vminps` return the second operand when the operands are unordered or both zero, which is exactly C's `a > b ? a : b`. With the bound first and the data second, two instructions per vector compute the reference bit for bit for every ordered pair of bounds | All three `clip` caveats in [kernel-findings.md](kernel-findings.md#where-the-result-is-narrower-than-the-number) would not exist on x86-64. The compiler very likely emits the same two instructions, so the kernel would ship as C |
| `cast_u8_f32` | Ships as C | `vpmovzxbd` from memory and `vcvtdq2ps` | NumPy ties the C here. Expected: nothing to install on x86-64 |

### Candidates that apply to every kernel

- **Masked loads and stores for leftover elements.** A masked load does not fault on the elements it leaves out, so a kernel could finish an array with one masked vector, with no scalar loop and no overlapping vector. Whether that satisfies "no reads outside the buffers, by even one byte" is a question about the rule: [open issue 3](#open-issues).
- **Non-temporal stores** for outputs larger than the third-level cache: `clip`, `cumsum`, `compress`. They skip reading the destination before writing it. They need aligned addresses and change what the caller finds in its caches afterwards.
- **An output that sits just behind the input.** Found by the survey: when the output buffer starts 8 to 128 bytes after the input, counted within a 4 KB page, the processor takes each store for a store to the elements about to be loaded. `np.clip` then takes 8x as long, `np.add` 2x and the naive C `clip` 1.5x ([survey-design.md](survey-design.md#where-the-output-sits)). Two power-of-two arrays allocated one after the other are often exactly there. A kernel could test the two addresses on entry and take a path that does not stall: loading further ahead of its stores, or running from the end of the arrays to the start. Every kernel that reads one buffer and writes another is concerned: `clip`, `cumsum`, `cast`, `compress`. On `clip`, where a kernel is otherwise level with NumPy, this may be the only place it wins. The judge times it in its own table: [judge-design.md](judge-design.md#where-the-output-buffer-sits). The cost shows where the data lies in the third-level cache.
- **Alignment of data.** A 32-byte load that straddles two cache lines costs more than one that does not. AGENTS.md says unaligned loads cost nothing; that was measured on the M1 and is not assumed here.
- **Alignment of code.** Where a loop sits relative to 32- and 64-byte boundaries affects how this processor decodes it.
- **Instructions slow on other x86-64 processors.** `pext` and `pdep`, gathers and masked stores are fast here and not everywhere. With one machine the judge cannot see that: [open issue 4](#open-issues).
- **Instructions beyond the level that this processor has.** AVX-VNNI could sum bytes, GFNI could transform masks. Using them needs a second tier and a run-time choice between two kernels: [open issue 5](#open-issues).

### What became of the candidates

Per family, in a line each; the reasons and the judge's figures are in each kernel's `x86_64/attempts.md` and in [kernel-findings.md](kernel-findings.md#results-by-family).

| Family | Kept | Rejected, with the judge's result | Not tried, and why |
|---|---|---|---|
| `argmax`, `argmin` | A NaN test of its own beside `vmaxps`, one `vcmpunordps` per two vectors; position by `vcmpeqps`, `vmovmskps`, `tzcnt`; the same for the first NaN. A block scheme beats NumPy's AVX2 loop at every size | | Aligned loads, a prefetch: left for a second run |
| `compress`, `nonzero` | One `vpmovmskb` per 32 mask bytes; `vpermd` by a table; `popcnt` for the write position; `tzcnt` and `blsr` for sparse blocks in `compress_f64` and `nonzero_bool` | The sparse path in `compress_f32` (1.005x, a wash); aligned loads (0.960x); a prefetch (0.980x) | `pext` in place of the table: more instructions, and the loop is held by its instruction count. A branch for all-zero blocks: taken at random on the judge's masks. The switch by density has gone from `compress_f32`; in the other two it is per block and has no weak spot |
| `maskfill_f64` | Load, blend and store, with a store-only path for sparse blocks | Alignment (no significant difference) | **`vmaskmovpd`: measured (1.062x overall, 1.6x up to 4,096 elements, 0.86x at 65,536) and left for a person,** [issue 70](https://github.com/sofeikov/hotkernels/issues/70) |
| `isin_i64` | `vpcmpeqq` for the general case, which ties the compiler (1.08x); narrowing with the pack instructions for test values from 0 to 32,767 | A window that starts at the smallest test value (0.77x against the naive C) | `vpshufb` nibble tables: with sixteen keys per register eight compares cost the same |
| `searchsorted`, `interp` | `vgatherqpd` for the probes, 64 or 512 searches in step; `vdivpd` for `interp`; an unfused multiply and add, as NumPy's build has | Sixteen searches in step in registers (the baseline; 64 in step is 1.92x over it) | Scalar loads with `cmov` for more than three needles: about six instructions per search step against four |
| `histogram_f64` | Four per vector; `vcvttpd2dq` makes the NaN test free; scalar counter updates. The proof was rechecked and changed: a multiply by a precomputed scale in place of the divide, fixed point, the last bin on the fast path | Two count tables (0.939x) | |
| `bincount_u8` | An increment that addresses memory; `rep stosb` to clear; four tables from 32,768 bytes | Four tables from 2,048 bytes | Two and eight tables |
| `cumsum` | The prefix sum across the register with one permute and an overlapping load; non-temporal stores from 24 MB; the overflow test as a compare of magnitudes | 16-byte vectors at every size for `int64` (0.991x; kept from 512 to 131,071 elements); aligned stores as a general step; adds as fused multiply-adds by one (0.979x) | |
| `sum_u8` | `vpsadbw`, mixed with `vpmaddubsw`, which brings a widening schedule back | | |
| `count_nonzero` | A compare with zero and a subtract | | `vmovmskpd` and `popcnt`: three instructions per four elements against two |
| `clip` | Two instructions with the bound first, bit for bit for every ordered pair of bounds. The compiler does not emit them, so the kernel is assembly | | **The three ARM64 caveats do not exist in the kernel; recording that is left for a person** |
| `cast_u8_f32` | `vpmovzxbd` and `vcvtdq2ps`, by a small margin over the C | | |

The candidates that applied to every kernel:

| Candidate | Outcome |
|---|---|
| Masked loads and stores for leftover elements | Not used by any kernel. Every kernel finishes with an overlapping vector, a scalar loop, or both. [Open issue 3](#open-issues) therefore never needed a decision |
| Non-temporal stores | Kept from 24 MB of output in `cumsum`, `clip` and `cast`: 1.2x to 2.1x at the sizes that reach them. Not usable in `compress`, whose stores overlap |
| An output that sits just behind the input | Stalled the first `cumsum_f64` by 2.5x at two sizes. Loading ahead of the stores removes it there and in `cumsum_i64`, `cumsum_f32` and `clip`. `histogram_f64` loses up to a sixth at some sizes; the mask kernels and `cast` nothing |
| Alignment of data | Kept in `sum_u8`, `count_nonzero`, and for the stores of `clip`, `cast` and the large-array path of `cumsum`. Rejected in `compress_f32`, `maskfill_f64` and `cumsum_i64`: the judge places the output 16 bytes after the input within a cache line, so one of the two is always off |
| Alignment of code | Hot loops start on 64-byte boundaries throughout. Varied only in `sum_u8` |
| Instructions slow on other x86-64 processors | `vgatherqpd` in the three search kernels. No `pext`, `pdep` or masked store in any kept kernel |
| Instructions beyond the level | None used. The judge refuses them |

### How the issue is worked and closed

1. **The goal spec for an x86-64 run names the family's candidates.** The run tries each one, or says why it could not, and records the attempt in the log like any other.
2. **The other architecture's kernel and log are reading material.** A run reads them for the idea and for the proofs. It does not start from that code, and it does not treat that log's rejections as settled.
3. **A candidate that changes a caveat, a binding constraint or a judge rule is not kept on the run's own authority.** The run reports it with the judge's figures, and a person decides. The rows for `maskfill`, `clip` and the masked tail are of that kind.
4. **What is found goes into [kernel-findings.md](kernel-findings.md)** under the family's x86-64 heading: what was tried, what held, and what the limit is on this processor.
5. **Ideas flow back.** A technique found here that does not depend on x86-64 instructions is noted for a later ARM64 run of the same kernel.
6. **Closed when** every family has its x86-64 entry in the findings and each row above is marked kept, rejected with the judge's result, or not tried with a reason. Both are done.

## The judge

The judge's design holds: it builds and tests whichever kernel directory matches the machine. In practice the code assumes macOS in more places than the one wrapper file the design names.

| Part | Today | Change |
|---|---|---|
| Compiler (`judge/build.py`) | Calls `clang` by name; `-mcpu=native` on ARM64 | clang on both architectures ([open issue 2](#open-issues), decided), which has to be installed on this machine. `-march=native` is already chosen for x86-64 |
| Libraries | `.dylib` | `.so` on Linux, built position-independent |
| Exported symbols | Expects `_name`, as Mach-O spells it | ELF spells it `name` |
| Writable data | Looks for `segname __DATA` in `otool -l` | Reads the section flags of the ELF object |
| Lint (`judge/lint.py`) | Rules for ARM64 only; any other architecture passes unchecked | Rules for x86-64: `call`, `syscall`, `sysenter` and `int` are forbidden, `#` starts a comment, read-only constants go in `.rodata`. There is no reserved register |
| Instruction-set level | Not needed: every Apple Silicon Mac runs what the M1 runs | New static check that a kernel uses nothing beyond x86-64-v3. Preferred: assemble with the level stated, so the assembler refuses anything newer. To be confirmed that the chosen compiler's assembler can do it; otherwise the object is disassembled and checked |
| Calling convention (`judge/c/abi_arm64.s`) | Checks `x19` to `x28`, `x29`, `sp`, `d8` to `d15` | New `abi_x86_64.s`: `rbx`, `rbp`, `r12` to `r15` and `rsp` unchanged; the direction flag clear; the rounding mode and exception masks unchanged; the upper halves of the vector registers cleared with `vzeroupper` before returning. Whether the wrapper can ask the processor for that directly is to be confirmed; otherwise it is a lint rule |
| Performance cores (`judge/c/harness.c`) | Asks macOS for the highest scheduling class; does nothing elsewhere | Pins the timing thread to one performance core, taken from the list Linux publishes. An efficiency core runs 256-bit code at a fraction of the speed, so an unpinned run is not a result |
| Noise | Reference loop before and after; macOS Low Power Mode | Reference loop unchanged. On Linux: the frequency governor, whether turbo is on, and the load average |
| Machine name | `sysctl machdep.cpu.brand_string` | The model name from `/proc/cpuinfo` |
| Guarded buffers (`judge/memory.py`) | `mmap` and `mprotect` through libc | Nothing: the constants come from Python's `mmap` module |
| Fixtures | Four correct ARM64 kernels and thirteen broken variants | The same set for x86-64. `uses_x18` has no counterpart; its place is taken by a variant that uses an instruction outside the level |

- **Exit criterion.** `uv run judge selftest` passes on this machine, including the repeatability half: ten consecutive timing runs of the correct fixture agree to within 3% at every size.
- **The attempts log moves beside its kernel:** `kernels/<name>/<arch>/attempts.md`. Today's logs are ARM64 logs and say so only in passing. A log per architecture keeps an x86-64 run from reading ARM64 dead ends as its own. `cast_u8_f32` has a log and no assembly; its log moves to `arm64/` with a note that the kernel ships as C there.
- **A kernel's verdict is per architecture.** The same kernel can ship as assembly on one, as C on another, and not be installed on a third because it does not beat NumPy there. Principle 4 of the core design is applied to each architecture separately.

## The survey

- **Port.** Three things are macOS-only: the clock and scheduling calls in `survey/naive.c`, the compiler flags, and the reading of clang's vectoriser remarks. The stand-in used for the measurement above shows the size of the change.
- **Run and record** on this machine under `survey/results/`, with clang.
- **A shortlist per architecture.** The ARM64 order of work was set by where NumPy is slow on ARM64. The x86-64 order is set by the recorded run, which is now in [survey-design.md](survey-design.md#results-on-x86-64) and confirms what was expected from the first measurement: `sum_u8`, the mask kernels, `cumsum`, `searchsorted` and `interp`, `histogram` and `bincount`, `isin`, `count_nonzero`, then `argmax` and `clip`, with `cast_u8_f32` last and probably dropped.

## The package and its delivery

| Part | Today | Change |
|---|---|---|
| `hotkernels/src/hotkernels/environment.py` | Supported: macOS on ARM64 | Also Linux on x86-64, when the processor and the operating system pass the check under [Target](#target) |
| `hotkernels/meson.build` | Builds the extension on macOS ARM64 only | Also on Linux x86-64. `tools/generate.py` already picks the kernel directory by architecture |
| C kernels in a wheel | Compiled with `-mcpu=apple-m1` | On x86-64 compiled with `-march=x86-64-v3`. The option takes a flag per architecture |
| Wheels | cibuildwheel, macOS ARM64 | Also Linux x86-64. The compiler inside the build image has to be the one the judge timed with, or the C kernels in the wheel are not the ones that were judged |
| CI | `judge selftest --no-speed` and `delivery check` on a macOS runner | The same two on a Linux x86-64 runner |
| Which kernels are installed | One list | One per architecture, from the verdicts. The table `kernels.json` is already generated per build |
| Binding constraints | The table "What a caller must know" in kernel-findings.md | A column per architecture. The `clip` status-register constraint and the `maskfill` rewrite may be ARM64 only |
| `uv run delivery time` | One results file per machine | Unchanged. The timing thread is pinned as in the judge |

`uv run delivery check` must pass on x86-64 before anything is installed there, as on ARM64.

## Amending the existing documents

The documents were written when there was one architecture, and they give ARM64 reasoning as the reasoning. "No wider compare exists on this hardware", "three 16-byte loads per cycle, which is the core's limit" and "unaligned loads cost nothing extra" are true of one core. A reader carrying a kernel over needs to know which sentences are about the operation and which are about the M1.

**The rule for every document:** what is true of an operation or of the contract is stated once, for every architecture. What is true of one architecture or one machine goes under a heading that names it. No sentence states an ARM64 fact as a general one.

| Document | What is ARM64-specific in it today | Amendment |
|---|---|---|
| [core-design.md](core-design.md) | The first target; "What we know so far" (one M1 experiment); "the judge contains nothing architecture-specific"; the run-time check "not designed yet"; x86-64 under Later | Architectures section rewritten: what is shared, what is per architecture, and a pointer here. The M1 experiment is labelled as such. Later loses the x86-64 item |
| [judge-design.md](judge-design.md) | The layout; `clang -O3 -mcpu=native`; the `x18` lint rule; the scheduling class and Low Power Mode; performance counters "on macOS"; the fixture table; x86-64 under Out of scope | A section "Per architecture" holding the table above. The stages describe the rule; the per-architecture section says how each platform enforces it. The cost of the strict memory rule gains the masked-load case |
| [survey-design.md](survey-design.md) | Its question is "where NumPy is slow on ARM64"; the findings ("where NumPy already has NEON, it is very good"); the shortlist and order of work | The method is stated once. Results, findings, shortlist and order of work each get a section per architecture, because NumPy's own vector code differs by architecture and so does the answer |
| [kernel-findings.md](kernel-findings.md) | Almost all of it: every "Technique", "Why it is this far ahead", "What did not help" and "Limit"; the summary table; the caller constraints; the decisions | Each family is split into three parts. **The operation:** why it is slow, what NumPy's semantics force, the proofs. **On ARM64:** today's text. **On x86-64:** written as results arrive. The summary table, "Where the result is narrower", "What the judge does not see" and "What a caller must know" get a column or a row per architecture. Each decision says which architectures it applies to |
| [delivery-design.md](delivery-design.md) | Supported environment; the machine everything was verified on; what a hook costs; which hook serves which kernel; the build | Supported environment per platform. Verified figures labelled with their machine, with the x86-64 ones beside them. The kernel list per architecture |
| [release-design.md](release-design.md) | Wheels for Apple Silicon; `apple-m1`; macOS runners; the re-measurement on one machine; note paths that name `arm64/kernel.s` | Wheels and jobs per platform. Paths become `<arch>/kernel.s`. Re-measurement needs one machine per platform: [open issue 6](#open-issues) |
| [AGENTS.md](../AGENTS.md) | The section "ARM64 on macOS"; `__TEXT,__const`; "unaligned loads cost nothing extra on this hardware"; the paths in the release-note rule; where the attempts log is | A section "x86-64 on Linux" beside the ARM64 one: symbol names, argument and result registers, what may be used and what must be preserved, the 128 bytes below the stack pointer a leaf may use, `vzeroupper`, the instruction-set level. A rule that the other architecture's kernel and log are for reading, not a template. A speed figure in a release note names its platform |
| [goal-spec-template.md](goal-spec-template.md) | Nothing, but it assumes one architecture | A step to read the other architecture's kernel and log for the idea, and, for x86-64, the family's candidates from open issue 1 with the duty to report on each |
| `README.md`, `kernels/README.md`, `hotkernels/README.md` | The layout and the supported platform | Updated in the change that makes them inaccurate, as the repository's rule requires. They describe the present only |
| `kernels/<name>/attempts.md` | ARM64 logs without the label | Moved to `kernels/<name>/arm64/attempts.md` |

**When.** The restructuring is its own pull request and comes before any x86-64 code. It adds no result and changes no conclusion: it moves existing ARM64 text under ARM64 headings and writes the shared part of each family from what the documents already say. The x86-64 sections are then filled in by the pull requests that produce the results they describe. Doing it in this order gives every x86-64 kernel run a place to write its reasoning, and makes a missing explanation visible as an empty section.

## Order of work

| Step | What | Done when |
|---|---|---|
| 1 | This document | Reviewed and merged |
| 2 | The existing documents restructured by architecture | Merged; no x86-64 content yet |
| 3 | The judge on Linux x86-64: build, lint, level check, wrapper, pinning, fixtures; AGENTS.md section; attempts logs moved | `uv run judge selftest` passes on this machine, including repeatability, and still passes on the M1 |
| 4 | The survey ported and recorded | A result file for this machine and an x86-64 shortlist |
| 5 | The kernels, one family per run, starting with `sum_u8` | Every kernel has an x86-64 verdict from the judge: assembly, C, or not installed, with the reason. **Done:** all 23 have assembly; four verdicts are marked for a person to confirm. `interp_f64` was written unlocked and timed again locked once its reference was corrected on `main` |
| 6 | The package: environment check, build, wheels, CI | `uv run delivery check` passes on this machine; `uv run delivery time` recorded. **Done** |
| 7 | The findings written up and open issue 1 closed | Every family has its x86-64 section; one `compat` release note. **Done** |

Steps 3 to 6 each take their own pull request or several. Step 5 follows AGENTS.md: one rule, two files per run.

## Exit criterion

- Every one of the 23 kernels has a recorded verdict on x86-64.
- An unmodified NumPy script runs faster on this machine after `hotkernels.install()`, with output that meets each kernel's correctness mode.
- [kernel-findings.md](kernel-findings.md) explains, per family, what was done on x86-64 and why it differs from ARM64.
- Every candidate in open issue 1 is marked kept, rejected or not tried, with the reason.

## Out of scope

- AVX-512, and any second tier above x86-64-v3.
- Windows and Intel Macs.
- Tuning for AMD processors or for efficiency cores. Kernels are timed on one Intel performance core.
- Linux on ARM64, as in the core design.
- Multithreading.

## Open issues

1. **Techniques particular to x86-64.** [Above](#open-issue-1-techniques-particular-to-x86-64). Worked through the kernel runs and closed with the findings.
2. **Which compiler builds the naive C on Linux.** *Decided in review: clang on both architectures.* The core design says clang. The machine has gcc 15.2, which left `argmax` and `clip` scalar in the survey's loops. With one compiler the baseline means the same thing on each architecture and "ships as C" is decided against the same compiler. It has to be installed here, and the wheel build has to use it too.
3. **Do masked loads and stores satisfy the strict memory rule?** *Not decided, and not needed: no kernel uses one.* They do not fault on, read or write the elements they leave out, so the judge's guard pages would pass them. Recommended: allowed, and said so in the rule, because nothing outside the buffer is accessed. A masked access that reaches an inaccessible page is slow on this processor, so it is for leftover elements only.
4. **Instructions that are slow on other x86-64 processors.** `pext` and `pdep` on AMD before Zen 3, gathers on older AMD and on Intel processors with the gather mitigation, masked stores on some AMD processors. The judge sees one machine. *Decided in review: allowed*, with each use named in the findings as untested elsewhere, until a second machine exists. The alternative was to forbid them in loops and accept slower kernels here.
5. **Instructions beyond the level that this machine has.** AVX-VNNI and GFNI. *Not used by any kernel; the judge refuses them.* Recommended: not used. A second tier needs a run-time choice between two kernels, which is the AVX-512 design.
6. **Releases with two platforms.** A release re-measures every speed it quotes. With two platforms that needs both machines, or notes that quote one platform and say so. **Still open, with a proposal.** The release workflow builds and tests wheels for both platforms. How a release's figures for the platform the maintainer is not sitting at are re-measured has a proposed rule in [release-design.md](release-design.md#re-measuring-on-two-platforms-proposed): a figure names its machine, the timings run once per platform, and a platform whose own files did not change since its recorded results may be carried. It is the maintainer's to accept.
7. **One findings document or two.** *Decided in review: one*, split by family and then by architecture, so the two are read side by side and the caller constraints sit in one table. The alternative was a second document per architecture, which is shorter to write and loses the comparison.
8. **Where the judge puts its timing buffers.** *Decided and built.* The time of a loop that reads one buffer and writes another depends on where the two sit relative to each other within a page, by up to 8x for NumPy and 5x to 7x for a compiled C running sum ([survey-design.md](survey-design.md#where-the-output-sits)). The judge took its timing buffers from NumPy's allocator, so the allocator decided which case a kernel was timed in. Now the judge places them: every timing is taken with the output clear of the inputs, and `run` adds a table with the output 16 bytes behind its input, because that is what a user's two power-of-two arrays get. The rule and its limits are in [judge-design.md](judge-design.md#where-the-output-buffer-sits). Still open: whether a kernel must do well in the close table to ship. Recommended: decide it on the first kernel that shows a cost there. That kernel exists now: `histogram_f64` loses up to a sixth at some sizes with its counters just behind the start of the data, and `nonzero_bool` a fifth at 4,096 elements. Every kernel that copies or accumulates into an output of the input's size loads ahead of its stores and shows no cost.

## Decisions made in review

| Question | Decision |
|---|---|
| Compiler for the naive C on Linux | clang, as on ARM64 |
| Instructions that are fast here and slow on other x86-64 processors | Allowed. Each use is named in the findings as untested elsewhere |
| One findings document or one per architecture | One, split by family and then by architecture |
