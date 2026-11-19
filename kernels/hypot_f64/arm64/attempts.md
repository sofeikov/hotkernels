# hypot_f64 attempts (arm64)

All runs on an Apple M1 Pro (Firestorm performance core), macOS 27.2.1, native. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the draft pull request #158 and not on `main` yet. Neither was changed here.

The reference is a loop that calls the C library's `hypot`, which is what NumPy calls for every element. The kernel computes the `hypot` of macOS's `libsystem_m` on arm64, so it is exact against the reference only where the C library is that algorithm. What the compiler does with `naive.c`: a call per element, 1.75 ns per element.

## What macOS's hypot is, and how it was established

Read from its instructions: a program in a scratch directory looked `hypot` up with `dlsym`, wrote its first 400 words out as `.word` directives, and `objdump -d` disassembled the assembled result (`lldb` could not attach). The function is 29 instructions with no loop and no table, plus two constants, 2^768 and 2^-768:

```
x = |a|, y = |b|, big = fmaxnm(x, y), small = fmin(x, y), eb and es their exponent fields
eb = 0x7ff            -> big + big       (an infinity, or both NaN; hypot(inf, NaN) = inf)
small = 0             -> big
small NaN             -> small + small
eb - es > 27          -> small = 0
eb >= 0x5fe (2^511)   -> sqrt(fma(small', small', big' * big')) * 2^768,  inputs scaled by 2^-768
eb < 0x21b (2^-484)   -> sqrt(fma(small', small', big' * big')) * 2^-768, inputs scaled by 2^768
otherwise             -> sqrt(fma(small, small, big * big))
```

A C model of exactly that sequence (`fmaxnm` and `fmin` by inline assembly, compiled without contraction and at `-O0` so the compiler does not compute both scalings and select, which raises overflow and underflow the library does not) was run beside the library on 80,000,000 pairs, 10 million from each of eight classes: standard normal; mixed magnitudes with exponent differences 0 to 60; random bit patterns (every NaN, infinity, denormal); exponents within 4 of the thresholds 0x5fe, 0x21b, 0x7fe and 1 with differences 0 to 39; exponent differences 24 to 31 around the 27 of the algorithm; a list of special values (zeros of both signs, infinities, quiet and signalling NaNs with payloads, the smallest denormal, the largest double, the neighbours of 2^511 and 2^-484) beside anything; exponents from -1076 to 1023; denormal pairs. 0 differ in their bits, NaN payloads included, and 0 differ in the floating-point flags raised (invalid, divide by zero, overflow, underflow, inexact).

It is not correctly rounded. Decided exactly with GMP integers (the square of the hypotenuse compared with the squares of the midpoints around the result), 10,000,000 pairs:

| Pairs | Not correctly rounded | More than one unit off |
|---|---|---|
| 2,000,000 standard normal | 317,854 (15.9%) | 0 |
| 2,000,000 mixed, exponent difference 0 to 60 | 155,412 (7.8%) | 0 |
| 1,999,262 exponents from -1060 to 1020, result normal | 8,824 (0.44%) | 0 |
| 2,000,000 constructed within 2^-47 of a unit of a midpoint (a = b^2 - j) | 1,020,527 (51.0%) | 0 |
| 2,000,000 Pythagorean triples scaled by powers of two | 12,835 (0.64%) | 0 |

Against the same library on 20,000,000 ordinary pairs (the first two classes above): `sqrt(x*x + y*y)` differs on 498,074 (2.5%), `sqrt(fma(x, x, y*y))` with the operands in argument order on 1,117,468 (5.6%), and glibc's corrected `e_hypot.c` (the x86-64 kernel's algorithm) on 2,374,312 (11.9%).

## The test outside the judge

`difftest.c` in a scratch directory, assembled with `kernel.s` and run beside the library's `hypot`:

1. Rounds of up to a million pairs (minus 0 to 3), one class per round (the eight above, zeros of either sign beside any finite value, and all classes mixed element by element), each computed with out clear of the inputs, out equal to a and out equal to b; a, b and out at offsets of 0 to 24 bytes from the allocation, and some not a multiple of 8. Bit for bit, NaN payloads included.
2. Every size from 0 to 300, five placements of the buffers, the three aliasings, 128 guard bytes around out.
3. Two million calls of 1 to 11 elements, comparing the flags the kernel raises (invalid, divide by zero, overflow, underflow) with those the library raises on the same elements.

## 1. Baseline: two elements per step, a bit test, the library's routine for the rest

- **Idea:** `fabs`, `fmax`, `fmin`; big's bits minus 2^-484's compared unsigned with the width of [2^-484, 2^511), which also catches a NaN; then `fmul`, `fmla`, `fsqrt`. A vector that fails the test, the last element of an odd n, and every element when FPCR is not round-to-nearest without flush-to-zero go to a scalar routine that is the library's instructions. The header of `kernel.s` proves the two steps the vector path leaves out change nothing there.
- **check:** pass
- **compare:** n/a (first version). `run` (CI jobs running on the machine): 3.64x [3.64 – 3.64] over naive C, 5.66x over NumPy.
- **Kept:** yes
- **Notes:** 0.46 ns per element from 256 up, against 1.75 for the library call. `fsqrt` on two doubles measured on this core outside the judge: 0.64 ns throughput (0.32 ns per element), 4.1 ns latency. The test outside the judge: 5,999,910 results, 0 differ; sizes 0 to 300, 4,515 runs, 0 differ, no guard byte changed; flags on 2,000,000 calls, none raised that the library did not raise, none lacking.

## 2. Eight elements per step, tested together

- **Idea:** four vectors loaded with `ldp`, tested with one reduction (`orr` of the four masks, `addp`, `fmov`, `cbnz`), computed and stored; a step that fails goes through the two-element loop. Fewer reductions and branches per element.
- **check:** pass
- **compare:** significantly slower, 0.994x [0.992 – 0.995]
- **Kept:** no
- **Notes:** 0.93x to 0.99x by size on `mixed`. Timing variants of the loop outside the judge showed what limits it: without the test (`sub`, `cmhs`, `addp`, `fmov`, `cbnz`) the loop runs at 0.31 ns per element, the throughput of `fsqrt`; with it, 0.45. The test's five instructions per vector, not the loop overhead, are the cost.

## 3. Blocks computed without a test per vector, checked once per block

- **Idea:** when out is neither a nor b, blocks of up to 64 elements are computed with no test per vector; each vector's big goes into a running `umax` and `umin` of its bits as 32-bit lanes (the high halves are what matter; the low halves are ignored by comparing the 64-bit lanes with 2^511 and 2^-484 as bits, whose low halves are zero). After the block one test: if a big was out of [2^-484, 2^511) or a NaN, FPSR is put back as it was before the block (read at entry with `mrs`, written with `msr` only here) and the block is computed again from its inputs on the checked path of attempt 1. Out equal to a or b takes the checked path throughout. Two instructions per vector instead of five.
- **check:** pass
- **compare:** significantly faster, 1.213x [1.211 – 1.215]
- **Kept:** yes
- **Notes:** 1.18x to 1.23x at every size. 0.38 ns per element from 256 up. The test outside the judge: 5,999,910 results, 0 differ; sizes 0 to 300, 4,515 runs, 0 differ, no guard byte changed; flags on 2,000,000 calls of 1 to 11 elements, none extra, none lacking.

## 4. Two vectors per iteration of the block loop

- **Idea:** `ldp`/`stp` of two vectors, two pairs of running maxima and minima so the accumulators are not one chain, blocks a multiple of four.
- **check:** pass
- **compare:** significantly slower, 0.908x [0.908 – 0.909]
- **Kept:** no
- **Notes:** 0.41 ns per element against 0.38 outside the judge too. Reordering the one-vector loop (the arithmetic before the running maximum and minimum) made no difference outside the judge (0.385 against 0.380) and was not run in it.

## 5. Blocks of 256 elements

- **Idea:** a quarter of the per-block work (the reduction, the branch, resetting the accumulators).
- **check:** pass
- **compare:** significantly slower, 0.986x [0.986 – 0.987]
- **Kept:** no
- **Notes:** the per-block work was already about 2% of a block of 64; nothing to gain, and a block with a special value costs four times as much to compute again.
