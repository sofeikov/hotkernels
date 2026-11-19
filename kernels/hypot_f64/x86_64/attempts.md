# hypot_f64 attempts (x86_64)

All runs on an Intel Core i5-14600KF (Raptor Lake, performance core, 48 KB first-level and 2 MB second-level cache per core, 24 MB third-level cache shared), Linux, glibc 2.43, pinned to one performance core by the judge. The runs are `--unlocked`: `spec.py` and `naive.c` are new in the same pull request and not on `main` yet. Neither was changed after the first line of assembly was written.

The reference is a loop that calls the C library's `hypot`, which is what NumPy calls for every element. The kernel computes the corrected algorithm of glibc's `e_hypot.c` without fused multiply-add (the sequence is in the header of `kernel.s`), so it is exact against the reference only where the C library is that algorithm, as glibc 2.43 here is. What the compiler does with `naive.c`: a call per element, 4.7 to 7.3 ns per element. NumPy takes about the same per element and about 300 ns per call more.

Beyond the judge, every kept attempt was checked with a program outside the repository that assembles `kernel.s` and calls it beside glibc's `hypot`: twelve rounds of a million pairs (minus 0 to 3), each with out clear of both inputs, then out equal to a, then out equal to b, so 36 million results; the pairs drawn from nine classes (standard normal; uniform; exponents from -100 to 100; random bit patterns, so every NaN, infinity and denormal; exponents from -1060 to 1040; a list of special values, both zeros, quiet and signalling NaNs with payloads, the smallest denormal, the largest double, and the neighbours of 2^511 and 2^-459; ordinary values with a quarter of them zero; exponents from -80 to 80; values next to 2^511 and 2^-459); a, b and out at offsets of 0 to 24 bytes from the allocation and once each an a and an out that are not a multiple of 8; 64 guard bytes around out. Results compared bit for bit, NaN payloads included. A second part calls the kernel on two million groups of one or four elements and compares the floating-point flags it raises (invalid, divide by zero, overflow, underflow) with those of glibc's `hypot` on the same elements.

## 1. Baseline: four elements per step, a bit test for the common case, a scalar routine for the rest

- **Idea:** both inputs of a group of four are tested on their bits before any floating-point instruction touches them: the absolute value plus a bias, compared as a signed integer with a limit, is one `vpaddq` and one `vpcmpgtq` per input for "in [2^-459, 2^511]". Such a group computes both branches of the core, blends them on `h <= 2 ay`, divides, and blends in `ax + ay` where `ax >= ay * 2^54`. A group with another element is tested again: if every element is finite and at most 2^511 and every pair with `ay` below 2^-459 is one whose result is `ax + ay` (a zero beside anything finite), it is computed the same way with 1.0 in place of those pairs' inputs, so that the unused core raises no flag. Anything else, and the last one to three elements, go to a scalar routine with the whole algorithm, specials and both scaled paths included. No `vmaxpd` before the test, because it raises the invalid flag on a quiet NaN.
- **check:** pass
- **compare:** n/a (first version). `run`: 5.12x [5.12 – 5.13] over naive C overall (random 5.51x, mixed 4.76x), 6.94x over NumPy.
- **Kept:** yes
- **Notes:** 1.06 ns per element from 256 to 4,096 (about 23 cycles per group of four), 1.20 at 16,384 and above. The program outside the judge: 35,999,946 results, 0 differ, no NaN with another payload, no guard byte changed. Flags: the kernel never lacks one glibc raises; it raises underflow where glibc does not in 2% of random bit patterns and in none of the pairs with exponents from -200 to 200. Those are pairs where `ay` is negligible beside `ax` (glibc returns `ax + ay` without computing anything) and `ay * ay / ax` is below 2^-1022, so the core computed and thrown away underflows. The close-placement table: 0.92x to 0.99x of the first.

## 2. Two groups per step, tested together

- **Idea:** while eight elements are left, two groups are loaded and tested with one `vptest`, then computed one after the other and stored, the first before the second's arithmetic starts (both groups' inputs are loaded by then, so out equal to a or b is still read before it is written). When either group has an element outside [2^-459, 2^511], the first group takes the path of one group, and the second follows. Loop overhead and the test's branch halve; the divide and the square root are unchanged.
- **check:** pass (the first version read past the buffers at n = 8: after a single group the pair loop restarted at element 4; a pair now starts only where eight elements are left)
- **compare:** significantly faster, 1.052x [1.050 – 1.054]
- **Kept:** yes
- **Notes:** 1.03x to 1.05x up to 65,536, 1.06x to 1.08x from 262,144. The program outside the judge, its flag part now with calls of 1, 4, 8 and 11 elements: 35,999,946 results, 0 differ; the kernel never lacks a flag glibc raises.

## 3. Stores past the caches from 24 MB of output

- **Idea:** from 3,145,728 elements, with out at a multiple of 8 bytes, the first zero to three elements go to the scalar routine so that out is at a multiple of 32 bytes, and the pairs of groups store with `vmovntpd`, followed by one `sfence`, as `where_f64` and the cumulative sums do from the same size. The kernel moves 24 bytes per element, and these stores take away the 8 bytes an ordinary store reads before it overwrites a line.
- **check:** pass. The first version wrote past the end of out at some sizes from the threshold up: after the unaligned first elements a group no longer started at a multiple of four, and the bound of the single groups assumed it did. Found by the program outside the judge (guard bytes changed at 3,145,730 elements), whose sizes the judge's correctness cases do not reach. The bounds of groups and pairs became `n - 3` and `n - 7` compared as signed. A second program then ran every size from 0 to 299 and the 81 sizes from 40 below the threshold to 40 above it, and 4,194,309, at five placements of a and of out (one not a multiple of 8), with out clear and out equal to a and to b: 24,714 runs, no failure.
- **compare:** no significant difference, 1.000x [1.000 – 1.001]
- **Kept:** no
- **Notes:** 1.01x to 1.02x at 4,194,304 and 16,777,216, nothing elsewhere. At 1.1 ns per element the kernel needs about 22 GB/s of memory, well inside what this machine gives a copy; the arithmetic, not memory, holds it at every size.

## Where it stands

Final `judge run` (unlocked, recorded): 5.25x [5.25 – 5.26] over naive C over both distributions, 7.31x over NumPy.

| n | random: kernel | naive C | speedup [95%] | NumPy | mixed: kernel | naive C | speedup [95%] |
|---|---|---|---|---|---|---|---|
| 16 | 17.8 ns | 51.4 ns | 2.88x [2.86 – 2.89] | 334.3 ns | 17.7 ns | 42.8 ns | 2.41x [2.41 – 2.41] |
| 64 | 67.1 ns | 322.0 ns | 4.80x [4.78 – 4.82] | 650.0 ns | 66.9 ns | 285.7 ns | 4.27x [4.25 – 4.28] |
| 256 | 261.6 ns | 1.4 us | 5.22x [5.20 – 5.23] | 1.9 us | 261.9 ns | 1.2 us | 4.55x [4.53 – 4.56] |
| 1,024 | 1.0 us | 5.4 us | 5.20x [5.18 – 5.22] | 6.7 us | 1.0 us | 4.8 us | 4.61x [4.60 – 4.63] |
| 4,096 | 4.1 us | 22.0 us | 5.34x [5.33 – 5.35] | 25.7 us | 4.1 us | 19.3 us | 4.66x [4.65 – 4.67] |
| 16,384 | 16.5 us | 113.3 us | 6.86x [6.85 – 6.86] | 113.4 us | 16.8 us | 98.5 us | 5.88x [5.88 – 5.88] |
| 65,536 | 66.1 us | 463.4 us | 7.01x [7.00 – 7.02] | 460.1 us | 67.6 us | 408.7 us | 6.04x [6.04 – 6.05] |
| 262,144 | 267.1 us | 1.87 ms | 7.00x [6.99 – 7.04] | 1.85 ms | 270.3 us | 1.66 ms | 6.13x [6.12 – 6.14] |
| 1,048,576 | 1.14 ms | 7.57 ms | 6.65x [6.58 – 6.69] | 7.52 ms | 1.17 ms | 6.68 ms | 5.69x [5.67 – 5.71] |
| 4,194,304 | 4.77 ms | 30.28 ms | 6.35x [6.32 – 6.39] | 32.05 ms | 4.84 ms | 26.88 ms | 5.55x [5.51 – 5.58] |
| 16,777,216 | 18.95 ms | 120.63 ms | 6.37x [6.36 – 6.38] | 127.34 ms | 18.89 ms | 106.60 ms | 5.64x [5.64 – 5.66] |
| overall | | | 5.63x [5.63 – 5.64] | | | | 4.90x [4.90 – 4.90] |

Over NumPy: 7.64x overall on random (18.7x at 16, 6.2x to 7.0x from 4,096 up), 6.99x on mixed. The two distributions take the kernel the same time; they differ in the C library's, which is faster on mixed, where more pairs return `ax + ay` early.

The second table, output 16 bytes behind a within the page: 0.96x to 0.99x of the first everywhere except 16,384 (0.89x, 18.9 us against 16.8 us). The kernel loads both groups of a pair before it stores either, but the next pair's loads come after those stores.

What holds it: **the divider.** `vsqrtpd` and `vdivpd` on 256 bits share one unit. Measured outside the judge on this core, a loop of independent pairs of one of each takes 3.78 ns (`vsqrtpd` alone 2.27 ns, `vdivpd` alone 1.52 ns); the kernel takes 4.0 to 4.1 ns per group of four from 256 to 262,144 elements, 94% of that bound. Everything else in a group (about 45 instructions, among them the range test, two blends and both branches of the core) runs beside it. Beyond the caches it needs about 22 GB/s, which this machine gives without stalling it (attempt 3).

Exactness outside the judge, on the final kernel: 36 million results against glibc 2.43's `hypot` over the whole range of doubles, out clear and equal to each input, 0 differences, NaN payloads included; 24,714 runs over every size from 0 to 299 and around 3,145,728 at several placements and aliasings, no failure, no guard byte changed (both programs are described at the top of this file).

What the kernel requires of its caller:

- **The C library's `hypot` must be the corrected algorithm of glibc's `e_hypot.c`** (glibc from about 2.35) for the results to be NumPy's. Elsewhere the kernel computes the same function to within about one unit in the last place, but not NumPy's bits. A binding checks that at run time.
- out is a, or b, or overlaps neither.
- Floating-point flags: never fewer than the C library raises for the same elements. One more, underflow, for a pair where one input is less than 2^-54 of the other and its square over the other is below 2^-1022 (2% of random bit patterns, none with exponents from -200 to 200). A binding that turns flags into warnings must not keep a result whose call raised a flag the library would not have.
- No alignment. Any `n`.

Instructions that are slow on some other x86-64 processors: none (`vpcmpgtq`, `vptest`, `vblendvpd`, `vsqrtpd`, `vdivpd`).

Not tried:

- **A square root without the divider:** `vrsqrtps` and Newton steps with fused multiply-adds, then a correction to the correctly rounded root. Exactness would rest on a proof about the correction step that random testing cannot replace, and the divide would remain.
- **Loads of the next pair ahead of this pair's stores,** for the second table's 16,384: a pair's test and its branch sit between the loads and the arithmetic, so the loads of the next pair would have to be taken before its test.
- **Bringing out to a multiple of 32 bytes:** one store per 20 cycles; a store that straddles two lines cannot matter at that rate.
