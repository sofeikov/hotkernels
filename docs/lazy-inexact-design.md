# An inexact kind for the lazy layer: what it would be worth

[Issue 315](https://github.com/sofeikov/hotkernels/issues/315) asks whether fusion should have a kind that is not bit-identical: `fuse.enable(inexact=True)`, with vector `cos`, `sin`, `exp` and powers by a small whole number inside an expression. A third of the time of ten programs from other codebases is such functions ([lazy-codebases-findings.md](lazy-codebases-findings.md#what-is-not-supported)), and they end every expression they are in.

This document has the measurements that decision needs, taken on an Intel Core i5-14600KF under Linux, and a recommendation. **The decision is the user's and is not taken here.** Nothing inexact is built: `fuse.py` behaves as before, but for two exact operations, `np.radians` and `np.degrees`, which need no decision and are built ([Exact forms](#exact-forms)).

## The decision

> **Open.** Should the lazy layer have an inexact kind?
>
> **Recommended: no, not in the layer.** If the gain is wanted, it belongs in the package, as loops for NumPy's own `np.cos`, `np.sin` and the others installed only with `install(inexact=True)`, which the lazy layer then inherits unchanged. Measured below: such loops alone, with no graph, make pvlib's solar position 2.17x; fusion then adds what it adds now, 1.05x, for 2.27x; and computing `cos` inside a kernel would add at most about 4% more.
>
> What that recommendation leaves open, for a design of the package's: whether a loop may call the platform's vector library (glibc's `libmvec`), which a judged kernel may not; the tolerance in units in the last place; and whether powers by a whole number may be repeated multiplication.

## Summary

- **NumPy computes these functions one element at a time here, by glibc's scalar functions.** NumPy 2.5.5 finds x86-64-v3 and no AVX-512 on this processor, and its float64 `cos`, `sin`, `tan`, `exp`, `cbrt`, `arctan2`, `hypot` and `power` gave the bits of a plain C loop over glibc's `cos` and the others on every element the five programs hand them. In the programs they take 2.3 to 15.5 ns an element.
- **glibc's vector functions take 0.5 to 4.7 ns an element on the same arguments, and differ from NumPy's on 15% to 64% of them.** The largest difference is 4 units in the last place (ULP) for `cos`, `sin`, `cbrt` and `arctan2`, 1 for `pow` and `hypot`. A power by a whole number as repeated multiplication takes 0.56 ns an element where NumPy's `pow` takes 6.9 ns, and is 3 to 5 ULP from it on a third to a half of the elements.
- **Inexact is not rare.** It is not a matter of a few hard cases: a vector `cos` gives another last bit than NumPy's for two elements in three of pvlib's.
- **The gain is the library's, not fusion's.** With NumPy's own loops for these functions replaced by glibc's vector functions process-wide, an experiment outside the repository, pvlib's eager program is 2.17x, and fused 2.27x, of NumPy as it is: fusion adds the 1.05x it adds today. scikit-image's colour difference is 1.42x eagerly and 1.65x fused, prysm's wavefront 1.12x and 1.29x. Astropy's models lose, 0.74x eagerly and 1.07x fused, because glibc's vector `pow` is slower than its scalar one on their fractional exponents.
- **Inside an expression it would add little more.** With the library calls at the vector figure, what is left of them that fusion could take, their passes over memory, is 174 ms of pvlib's fused 1,419 ms, 16 ms of scikit-image's 265 ms, 45 ms of prysm's 349 ms.
- **In the layer it would break the layer's rule and the tools' check.** The layer gives NumPy's bits, and every tool compares the fused run with the eager program bit for bit. A loop of the package's changes both alike; a kernel of the layer's changes one.
- **`np.radians` and `np.degrees` are exact, and built.** NumPy's loop for each is one multiplication by a folded constant; a kernel's product by the same float64 had the bits on 100 million values of every class. Each is now `mul` of its operand and the constant in both readers of expressions. pvlib's fused run computes 66 more nodes inside a program and writes 0.55 GB less, and is level in time: the conversions were 75 ms of NumPy's 2.7 s.
- **`x % 1.0` is `x - np.floor(x)` bit for bit, and is not built.** Shown on the same 100 million values and argued below. It would need an operation the kernels do not have, and the one program with it, perlin-numpy, computes it on an array the module declines for its layout ([issue 316](https://github.com/sofeikov/hotkernels/issues/316)).

## How the package treats a kernel that is not bit-identical

The package has had such a kind since the float `cumsum` kernels, which add in another order ([core-design.md](core-design.md), [delivery-design.md](delivery-design.md)):

- **A tolerance in the spec.** The judge compares bits unless a spec declares `mode={"tolerance_ulp": n}`, optionally with a scale; NaN and infinity must still be where NumPy has them ([judge-design.md](judge-design.md)).
- **Off unless asked for.** `hotkernels.install(inexact=True)` installs them, `python -m hotkernels --inexact` too, and `report()` marks them. Two of the package's kernels are of that kind.
- **A kernel is a leaf function.** It calls nothing; the judge fails a kernel that makes a call. `libmvec`'s `_ZGVdN4v_cos` is a call into glibc.

The lazy layer has no such kind. Its one rule is that every result, warning and error is NumPy's ([lazy-fusion-design.md](lazy-fusion-design.md#why-the-bits-are-numpys)), and a fused kernel may contain only operations that are one IEEE operation NumPy makes with the same instruction.

## What was measured

Everything on one performance core with one BLAS thread, NumPy 2.5.5, glibc 2.43, at the programs' large size, while two other sessions were working on the machine (load averages 2.5 to 4.0). The scripts are not in the repository: a C file of loops over glibc's scalar and vector functions, a small extension that puts glibc's vector functions in place of NumPy's float64 loops through `PyUFunc_ReplaceLoopBySignature` (the interface the package's `hook_loop.c` uses for `clip`), and wrappers of the layer's evaluation that see each call's operands.

### Each function on the programs' own arguments

Each program was run lazily, node by node, and at every call of one of these functions on float64 its operands were taken as they were and handed to NumPy's function, to a C loop over glibc's scalar function, to glibc's vector function (`_ZGVdN4v_cos` and its siblings: AVX2, four lanes), and for a whole exponent to repeated multiplication, `((x * x) * x) * ...`. Each was timed (the quickest of three) and compared with NumPy's result on every element. Nanoseconds an element:

| Program | Function | Elements | Arguments | NumPy | glibc, scalar | glibc, vector | Multiplication | Vector: largest ULP, share differing | Multiplication: largest ULP, share differing |
|---|---|---|---|---|---|---|---|---|---|
| `pvlib_solar` | `np.cos` | 286 M | -4,058 to 10,880 | 4.21 | 4.12 | 0.59 | | 4, 64% | |
| | `np.sin` | 80 M | -4,058 to 10,880 | 4.09 | 3.97 | 0.55 | | 3, 58% | |
| | `x ** k`, k of 3 to 10 | 23 M | 0.0024 to 0.26 | 6.86 | 6.77 | 2.92 | 0.56 | 1, 26% | 5, 36% |
| | `np.arctan2` | 4 M | -1 to 1 | 6.62 | 6.81 | 1.25 | | 3, 29% | |
| | `np.tan` | 3 M | -756 to 2,797 | 2.90 | 2.82 | 0.64 | | 3, 33% | |
| `skimage_delta_e` | `np.cbrt` | 6 M | 0.009 to 1 | 9.35 | 9.12 | 1.24 | | 4, 53% | |
| | `np.cos` | 4 M | -1.1 to 24 | 10.95 | 10.94 | 1.10 | | 3, 63% | |
| | `np.hypot` | 4 M | -86 to 98 | 9.18 | 10.13 | 3.25 | | 1, 14% | |
| | `np.arctan2` | 2 M | -108 to 94 | 15.46 | 15.48 | 4.21 | | 3, 33% | |
| | `x ** 2.4` | 6 M | 0.09 to 1 | 7.08 | 7.08 | 2.91 | | 1, 26% | |
| | `x ** 7` | 2 M | 0.27 to 133 | 7.15 | 6.93 | 4.68 | 1.34 | 1, 26% | 4, 50% |
| | `np.sin` | 2 M | -1.6 to 1.6 | 4.75 | 4.72 | 0.74 | | 3, 15% | |
| | `np.exp` | 1 M | -121 to 0 | 2.29 | 2.28 | 0.53 | | 3, 44% | |
| `prysm_zernike` | `np.sin` | 7 M | -22 to 22 | 4.18 | 4.33 | 2.66 | | 3, 63% | |
| | `np.cos` | 7 M | -22 to 22 | 4.27 | 4.15 | 1.52 | | 3, 64% | |
| | `x ** k`, k of 1 to 7 | 6 M | 0 to 1.41 | 5.89 | 6.98 | 2.82 | 0.56 | 1, 21% | 3, 33% |
| | `np.arctan2` | 1 M | -12.5 to 12.5 | 6.81 | 6.96 | 1.28 | | 3, 32% | |
| | `np.hypot` | 1 M | -12.5 to 12.5 | 3.20 | 2.61 | 0.89 | | 1, 1% | |
| `astropy_models` | `np.exp` | 8 M | -3,389 to 5 | 2.69 | 4.51 | 2.72 | | 3, 40% | |
| | `x ** y`, y of -2.5 to 0.85 | 13 M | 0 to 1,103 | 4.94 | 6.92 | 6.23 | | 1, 26% | |

- **NumPy's results are glibc's scalar results,** bit for bit, on every element of every call but one kind: `x ** 0.5`, which NumPy computes as `np.sqrt`, 1 ULP from glibc's `pow` on some elements. That and the times say NumPy has no vector loop of its own for these on this processor.
- **The scalar times depend on the data.** pvlib's `cos` is 4.2 ns an element in the program and 10.9 ns on the same values shuffled: glibc's functions branch on the size of the argument, and pvlib's arguments change slowly along an array.
- **Repeated multiplication is the quickest and the least exact.** Up to 5 ULP for `x ** 10`, as each product rounds once.
- **glibc's vector `pow` is not always quicker.** On Astropy's fractional exponents, 6.2 ns an element against NumPy's 4.9 ns.
- **perlin-numpy has none of these of any weight:** its `np.cos` and `np.sin` are on 512 gradients. Its 294 ms is `grid % 1`, below.

### The programs with every such call at the vector figure

Each program run as it is, eagerly and fused, with NumPy's loops and then with glibc's vector functions in place of NumPy's float64 loops of `cos`, `sin`, `tan`, `exp`, `log`, `cbrt`, `arctan2`, `hypot` and `power` (a whole exponent of 3 to 16 by repeated multiplication), the four ways in turn, the quickest of 7 rounds, milliseconds:

| Program | NumPy | Fused | Vector loops, eager | Vector loops, fused | Fused against NumPy | Vector, eager, against NumPy | Vector, fused, against NumPy |
|---|---|---|---|---|---|---|---|
| `pvlib_solar` | 2,767.42 | 2,660.46 | 1,277.66 | 1,218.73 | 1.04x | 2.17x | 2.27x |
| `skimage_delta_e` | 431.37 | 389.20 | 303.01 | 261.00 | 1.11x | 1.42x | 1.65x |
| `prysm_zernike` | 416.52 | 380.71 | 370.40 | 322.57 | 1.09x | 1.12x | 1.29x |
| `astropy_models` | 207.13 | 180.35 | 279.84 | 192.75 | 1.15x | 0.74x | 1.07x |

The same with one group of functions replaced at a time, in four later takes at higher load, each a row of its own take:

| Replaced | `pvlib_solar` eager, fused | `skimage_delta_e` | `prysm_zernike` | `astropy_models` |
|---|---|---|---|---|
| None (fused against NumPy, the four takes) | 1.05x to 1.09x | 1.09x to 1.15x | 1.09x to 1.12x | 1.15x to 1.20x |
| `cos`, `sin`, `tan`, `arctan2` | 1.86x, 1.99x | 1.20x, 1.35x | 1.06x, 1.24x | 1.03x, 1.15x |
| `power`, a whole exponent by multiplication | 1.05x, 1.10x | 0.99x, 1.09x | 1.00x, 1.13x | 0.76x, 1.05x |
| `power`, all by vector `pow` | 0.97x, 1.07x | 0.97x, 1.07x | 0.94x, 1.04x | 0.74x, 1.01x |
| `exp`, `log`, `cbrt`, `hypot` | 1.00x, 1.10x | 1.18x, 1.33x | 1.02x, 1.14x | 1.06x, 1.22x |

- **pvlib's gain is `cos` and `sin`.** The powers are 5% of it, and only by multiplication.
- **A vector `pow` loses wherever it is used for everything.** Astropy's fractional exponents are slower by it, and a whole exponent is twelve times quicker by multiplication than by NumPy's `pow` but only twice by the vector one.
- **The eager and the fused run still agree bit for bit** with the vector loops in place, on three of the four: the layer calls the same NumPy functions. Astropy's do not, and why was not traced. With NumPy's own loops they agree, as they always have; the likeliest reason is a power the eager program and the layer reach by two roads, `x ** y` and `np.power(x, y)`, which glibc's scalar `pow` and NumPy's fast paths make agree and the vector `pow` does not.

### What fusion could still take with the library at the vector figure

`limit` on the four programs, with NumPy's loops and with the vector ones, one take after the other (milliseconds):

| Program | NumPy | Fused | Floor | At best | Library calls in the fused run | Of them, passes over memory | Fused with vector loops | Floor with vector loops | At best, with vector loops, against NumPy |
|---|---|---|---|---|---|---|---|---|---|
| `pvlib_solar` | 2,932.30 | 2,748.84 | 1,401.75 | 2.09x | 1,881.77 | 678.78 | 1,419.40 | 608.47 | 4.82x |
| `skimage_delta_e` | 423.61 | 366.42 | 282.46 | 1.50x | 225.18 | 69.21 | 264.54 | 216.25 | 1.96x |
| `prysm_zernike` | 412.51 | 369.90 | 255.00 | 1.62x | 127.21 | 78.39 | 348.54 | 236.22 | 1.75x |
| `astropy_models` | 202.69 | 179.27 | 102.16 | 1.98x | 90.88 | 9.00 | 206.06 | 172.79 | 1.17x |

- **Inside an expression a vector `cos` would save its passes over memory,** which `limit` counts as removable: with the vector loops, 174 ms of pvlib's library calls, 16 ms of scikit-image's, 45 ms of prysm's. The fused run would then be at most about 1,245 ms for pvlib, 2.36x of the NumPy of that take, 249 ms for scikit-image, 1.70x, and 304 ms for prysm, 1.36x. That is an estimate, not a measurement: it takes the calls' compute as it is in a loop of their own.
- **pvlib's floor falls to 608 ms,** and the arithmetic that is not yet fused (`s += a * np.cos(b + c * x)` on arrays NumPy reuses) is most of what is left. The library is no longer the limit there.

### What the user would see

The programs' results, eagerly with NumPy's loops and with the vector ones, compared value by value:

| Program | Result | Values differing | Largest difference, relative | Median ULP of those that differ |
|---|---|---|---|---|
| `pvlib_solar` | zenith, apparent and true | 59% | 5.0e-14 | 1 |
| | elevation, apparent and true | 86% | 1.3e-10 | 3 |
| | azimuth | 22% | 5.2e-10 | 2 |
| | equation of time | 0.3% | 4.7e-8, near its zeros | 8,192 |
| `skimage_delta_e` | the colour difference | 97% | 9.5e-13 | 25 |
| `prysm_zernike` | the wavefront | 76% | 1.0e-9 | 2 |
| | the image of a point | 99.7% | 9.2e-11 | 238 |
| `astropy_models` | the image | 66% | 3.7e-15 | 2 |

- **Most results change in their last bits,** and a few by much more where a result is near zero: the equation of time in minutes, which crosses zero, moves by 1.5e-11 minutes at most, which is 4.7e-8 of a value near zero.
- **For what these programs compute, none of it matters:** a sun's position to 1e-12 degrees, a colour difference to 1e-12. Whether that holds for a user's program is the user's to judge, which is why the package's inexact kind is off unless asked for.

## What an inexact kind would be

If the user decides for one, this is the design that would follow from the measurements. It is written down so that the decision is about something definite.

### Where it lives

| | A kind of the layer's: `fuse.enable(inexact=True)` | A kind of the package's: loops under `install(inexact=True)` |
|---|---|---|
| What changes | A fused kernel may contain `cos`, `sin`, `exp`, a power by a whole number; their bits are the kernel's | NumPy's own float64 loop for `np.cos` and the others is the package's, for every call, eager or lazy |
| What it is worth, pvlib | 2.27x measured with the calls at the vector figure, about 2.36x estimated inside expressions | 2.17x eagerly, 2.27x with fusion on top |
| The layer's rule | Broken: the fused run no longer gives the eager program's bits, and every tool's comparison needs a tolerance | Kept: the layer gives what NumPy gives, whatever NumPy's loops are |
| Programs that do not use the layer | Nothing | Everything that calls `np.cos` |
| Kernels | One for each shape that contains such a function, on two architectures | One loop for each function and element type |

The second gives almost all of the first's gain, keeps the layer's one rule, and reaches every program rather than those run lazily. That is the reason for the recommendation.

### Whose code

- **glibc's `libmvec` is a call.** A kernel calls nothing, and the judge fails one that does. A loop that calls `_ZGVdN4v_cos` would be a new kind of thing in the package, the platform's code behind a hook, judged as a whole call through `delivery check` and a tolerance rather than as a kernel.
- **A kernel of one's own is a polynomial and an argument reduction.** Vector `cos` and `sin` for every float64 need the reduction of a huge argument modulo π/2 (pvlib's reach 10,880; NumPy's accept 1e308), which is most of the work and of the risk. A kernel could take arguments up to a bound and hand the rest to NumPy, as the package's kernels hand on what they do not take.
- **Repeated multiplication is a kernel's already.** `x ** k` for a small whole `k` as products is a shape of single multiplications; only its result is not `pow`'s. It is the one inexact operation the layer's kernels could make with no new code.

### Which functions and what tolerance

The figures above suggest the list and the bound, to be set by the judge on its value classes and not by these programs alone:

| Function | Measured here, largest ULP | A tolerance would be |
|---|---|---|
| `cos`, `sin`, `tan` | 4 | 4 |
| `arctan2`, `cbrt`, `exp` | 4 | 4 |
| `hypot`, `pow` | 1 | 2 |
| `x ** k` by multiplication, k up to 10 | 5 | k − 1 roundings: 4 ULP for `x ** 5` and growing with k |

NaN and infinities where NumPy has them, as the judge's tolerance mode requires. Warnings: an inexact loop must raise NumPy's floating-point flags for the same elements (overflow for `exp` of large arguments, invalid for `cos` of an infinity), which glibc's vector functions were not checked for here.

### How it would be tested

- **The judge's tolerance mode,** per function, on its value classes, with arguments up to the reduction bound and beyond it.
- **The programs' own results,** with a tolerance per output that a user of the program would accept, which is the program's question and not NumPy's: [What the user would see](#what-the-user-would-see).
- **The layer's tests as they are.** With the kind in the package they need no change: they hold the layer to whatever NumPy computes.

### ARM64

There is no ARM64 machine here, and nothing above was measured on one. macOS has no `libmvec`; its vector functions are Accelerate's (`vvcos` and the others), with their own errors. Whether NumPy has vector loops of its own for these functions on Apple's processors was not checked.

### What it does to the tools

`fusion_time` and `limit` compare the fused run's results with the eager program's, bit for bit, before they time. With a kind of the package's, both runs use the same loops and the comparison stands. With a kind of the layer's, the comparison would need a tolerance per result, and "the fused run gives the eager program's results" would no longer be what the tables say.

## Exact forms

Two of the functions in the findings' list have a form a kernel can make with NumPy's bits. They need no decision.

### `np.radians` and `np.degrees`: one multiplication, built

**Why the bits are NumPy's.** NumPy's loop for `np.radians` and `np.deg2rad` is `npy_deg2rad`, `x * (NPY_PI / 180.0)`; for `np.degrees` and `np.rad2deg` it is `npy_rad2deg`, `x * (180.0 / NPY_PI)`. The constant is a division of two doubles folded by the compiler, and `np.pi / 180.0` in Python is the same division of the same doubles: 0.017453292519943295 and 57.29577951308232. So each is one IEEE multiplication, and a kernel's product by that float64 has its bits. A product does not depend on the order of its operands, NaN's included, since only one of them can be a NaN.

**How it was checked.** All four functions against `x * c` and `c * x` on 100 million values: random bit patterns of every exponent, uniform values, values of every magnitude, NaNs with payloads of both signs, signalling ones, infinities, zeros of both signs, subnormals, every power of two and its neighbours. No value differed. The tests hold the same on 200,000 such values and every power of two.

**What changed.**

- **`fuse.SCALED`** holds the four functions and their constants, and `fuse.operation_of` reads `np.radians(x)` as `mul` of `x` and the constant, which the program did not write. The C core's reader does the same (`operation_of` in `_core.c` writes the two operands into places its caller keeps).
- **They wait for a kernel.** With fusion on, a large `np.radians` on an array that has its values was a call made at once; it is a node now, as a multiplication is.
- **A warning is NumPy's.** `np.degrees` of a value above about 3e306 overflows; the kernel's flag drops its result and NumPy's own calls are made, so the warning says `degrees` and names the line.
- **What is left to NumPy:** a conversion of a float32 array or of whole numbers, whose operand is not a kernel's.

**Before and after.** `pvlib_solar` and `skimage_delta_e` at the large size, the quickest of 7 rounds, the ways in turn, one performance core, one BLAS thread, milliseconds; this change and its base, alternately, at load averages of 3.5 to 5.6:

| Program | Take | | NumPy | Lazy | Fused | Fused, nothing installed | Fused against NumPy | Fused against lazy |
|---|---|---|---|---|---|---|---|---|
| `pvlib_solar` | 18:23 | after | 3,242.69 | 3,353.52 | 2,817.61 | 2,822.31 | 1.15x | 1.19x |
| | 18:26 | before | 3,192.61 | 3,491.38 | 2,970.54 | 2,950.87 | 1.07x | 1.18x |
| | 18:29 | after | 2,943.84 | 3,113.11 | 2,774.87 | 2,818.57 | 1.06x | 1.12x |
| | 18:32 | before | 2,890.80 | 3,024.78 | 2,686.06 | 2,735.28 | 1.08x | 1.13x |
| `skimage_delta_e` | 18:23 | after | 419.36 | 412.53 | 383.47 | 382.24 | 1.09x | 1.08x |
| | 18:26 | before | 419.82 | 406.17 | 378.43 | 380.30 | 1.11x | 1.07x |
| | 18:29 | after | 414.56 | 400.31 | 374.23 | 374.91 | 1.11x | 1.07x |
| | 18:32 | before | 435.78 | 422.81 | 392.56 | 392.26 | 1.11x | 1.08x |

- **Level.** The machine moved by 10% between takes, more than the conversions weigh: at most their 75 ms of NumPy's passes, 2.5% of pvlib.
- **What it does is in the counts,** which do not move from take to take: pvlib's kernel calls 873 before and 938 after, its nodes computed inside a program 953 and 1,019, its new arrays 11,014.5 MB and 10,459.4 MB. 66 more of its nodes are computed inside a program; which of its 89 conversions are not, and why, was not looked into.
- **scikit-image's `(np.rad2deg(Hbar) - 275) / 25` is one program now:** one kernel call and one node more inside a program, 8 MB less written, nothing in time.

### `x % 1.0`: the same bits as `x - np.floor(x)`, not built

**Why.** NumPy's remainder is `npy_divmod`: `mod = fmod(x, 1.0)`, exact; if `mod` is not zero and its sign differs from the divisor's, `mod + 1.0`; if it is zero, `+0.0`. For `x` of either sign that is not a whole number, `fmod` is `x - trunc(x)`, and `mod + 1.0` for a negative `x` is the exact value `x - floor(x)` rounded once, which is what `x - np.floor(x)` computes: `floor` is exact and the subtraction rounds once. For a whole `x`, zeros included, both give `+0.0`, since `x - x` is `+0.0` when rounding to nearest. For an infinity both give an invalid operation's NaN, and a NaN passes through both. On the 100 million values above, `x % 1.0` and `x - np.floor(x)` had the same bits on every one, and so did `x % 1` and `x % np.int64(1)`.

**Why it is not built.**

- **The kernels have no `floor`.** Every operation in a kernel's name is one of `add`, `sub`, `mul`, `div`, `max`, `min`, `neg`, `abs`, `sqrt`, `sq` and the comparisons. `floor` is one instruction on both processors (`vroundpd` with mode 9, `frintm`) and exact, so it would have NumPy's bits; but it is a new operation for `make_kernel.py`, its templates for two architectures, and the judge, and `x % 1.0` is one node that would be two operations of a pattern, which the readers of expressions have never had.
- **It would gain nothing on these programs now.** Its one program of weight is perlin-numpy's `np.mgrid[...].transpose(1, 2, 0) % 1`, 294 ms, on the transposed array that the module declines for its layout, as it declines the interpolant on the result ([issue 316](https://github.com/sofeikov/hotkernels/issues/316)). pvlib's `% 360.0` is another divisor, which [issue 222](https://github.com/sofeikov/hotkernels/issues/222) is about.

So it waits for issue 316, with the proof above.

## Recommendation

1. **No inexact kind in the lazy layer.** The layer's rule stays: NumPy's bits.
2. **If the gain is wanted, a design of the package's** for float64 `np.cos` and `np.sin` first, installed only with `install(inexact=True)`, with the questions of [The decision](#the-decision). pvlib's solar position would be about 2.2x eagerly by it and 2.3x fused.
3. **Exact forms as they are found.** `np.radians` and `np.degrees` are built; `x % 1.0` waits for issue 316.

## What is open

- **The decision above,** the user's.
- **Whether glibc's vector functions raise NumPy's flags for the same elements,** not checked.
- **Why Astropy's eager and fused runs differ under the vector loops,** when they agree under NumPy's: not traced. It says that some call reaches NumPy by two roads that give one result only with glibc's scalar functions, which no test holds on another library.
- **ARM64,** for want of the machine.
