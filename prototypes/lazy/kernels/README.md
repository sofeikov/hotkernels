# Fused kernels

One directory per kernel, laid out as the kernels of [`kernels/`](../../../kernels/README.md) are, and checked and timed by the same judge:

```
prototypes/lazy/kernels/<name>/
  spec.py            signature, NumPy oracle
  naive.c            the simplest correct C loop
  arm64/kernel.s     NEON assembly, written by hand or by make_kernel.py from the shape's name
  arm64/attempts.md  log of what was tried on ARM64 and what the judge said
```

A fused kernel computes a whole expression in one pass over memory, where NumPy makes one pass per operation and allocates a temporary for each intermediate. A longer expression is computed by several of them, block by block, with what one gives the next kept in the cache. The design is [docs/lazy-fusion-design.md](../../../docs/lazy-fusion-design.md). They are kernels of the lazy layer's prototype and are not built into the `hotkernels` package.

```bash
uv run judge check fuse_add_scaled_f64 --unlocked    # lint, then every correctness case against NumPy
uv run judge run fuse_add_scaled_f64 --unlocked      # the same, then timings against naive C and NumPy
uv run python -m prototypes.lazy.build_fused         # assemble them into the module the layer calls
uv run python -m prototypes.lazy.make_kernel "sub(V0,mul(S0,V1))"    # write the kernel of a shape
```

`--unlocked` is needed until a kernel's `spec.py` and `naive.c` are on `main`, as for any new kernel. In the timing tables the `NumPy` column is the spec's oracle, the expression as a program writes it, so the speedup against it is what fusing the expression gains at that size, less what a call from Python costs.

## The kernels written by hand

| Kernel | Computes | Pattern |
|---|---|---|
| `fuse_add_scaled_f64` | `a + s * b` | `add(V0,mul(S0,V1))` |
| `fuse_sub_scaled_f64` | `a - s * b` | `sub(V0,mul(S0,V1))` |
| `fuse_add_product_f64` | `a + b * c` | `add(V0,mul(V1,V2))` |
| `fuse_sub_product_f64` | `a - b * c` | `sub(V0,mul(V1,V2))` |
| `fuse_product3_f64` | `a * (b * c)` | `mul(V0,mul(V1,V2))` |
| `fuse_scaled_sum_f64` | `s * (a + b)` | `mul(S0,add(V0,V1))` |
| `fuse_mul_shifted_f64` | `a * (s + b)` | `mul(V0,add(S0,V1))` |
| `fuse_mul_difference_f64` | `a * (b - c)` | `mul(V0,sub(V1,V2))` |
| `fuse_sum3_f64` | `a + (b + c)` | `add(V0,add(V1,V2))` |
| `fuse_sum_squares_f64` | `np.sum(x * x)`, and the mean | `sum(sq(V0))`, `mean(sq(V0))` |
| `fuse_laplacian5_f64` | `s * a - (b + c + d + e)` | `sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))` |
| `fuse_flux_limit_f64` | `np.where(a * (b - c) > lo, fill, d)` | `where(gt(mul(V0,sub(V1,V2)),S0),S1,V3)` |
| `fuse_flux_update_f64` | `a - b * (c - d + e - f)` | `sub(V0,mul(V1,sub(add(V2,sub(V3,V4)),V5)))` |
| `fuse_poisson_step_f64` | `((a + b) * s0 + (c + d) * s1) / s2 - s3 * e` | `sub(div(add(mul(S0,add(V0,V1)),mul(S1,add(V2,V3))),S2),mul(S3,V4))` |

`V` is an array read along the innermost axis and `S` a number. The operands of a sum or a product are written in a fixed order in a pattern's name; the program may have them either way round.

## The kernels written from a shape's name

A shape's name says everything its kernel has to do, and the loop for it is always the same loop: load eight elements of each array, make the operations in the tree's order, store. [`make_kernel.py`](../make_kernel.py) writes it: the `spec.py`, whose oracle is the shape as a NumPy expression, the `naive.c`, the `arm64/kernel.s` and the first entry of its log. The kernel is then a kernel like the others: the judge holds it to NumPy's bits, and `fuse.py` reads its pattern off its oracle.

- **The name is the shape,** read from left to right with `v` for an array and `s` for a number: `fuse_sub_v_mul_s_v_f64` is `sub(V0,mul(S0,V1))`, and `fuse_add_v_v_f64` is `add(V0,V1)`.
- **Which shapes:** the list is `SHAPES` in `make_kernel.py`, and `--all` writes whichever of them have no directory. The single operations (`add(V0,V1)`, `mul(S0,V0)`, `div(V0,S0)`, `sqrt(V0)`, `max(S0,V0)` and the rest, eighteen of them), the forty-six shapes of two to six operations that cover the corpus's expressions best, as [`programs/mine.py`](../programs/mine.py) chose them, and the two halves of an advection flux, four operations and eight, written whole because a program of six kernel calls for them took twice what one loop does.
- **A kernel of one operation fuses nothing by itself.** It is what lets a whole expression be computed block by block with a kernel for every step ([docs/lazy-fusion-design.md](../../../docs/lazy-fusion-design.md#an-expression-block-by-block)). An operation by itself in a program is still NumPy's call.
- **What it writes no kernel for:** a comparison and `np.where`, a sum; more than six arrays; and a shape whose C reference the compiler rewrites into other bits, which the judge finds (`s0 + (-|a|)` becomes `s0 - |a|`, and a NaN changes sign).
- **`np.maximum` and `np.minimum` are `fmax` and `fmin`,** which is what NumPy's own loops execute on ARM64: a NaN wins over a number, and of two zeros the maximum is the positive one. The C reference spells that rule out, since C's `fmax` is another function.
- **Not tuned.** A written kernel loads with plain `ldp` and is as the tool left it. Tuning one is a run like any other, in its own `attempts.md`; `--force` would write over it.

## What every fused kernel is held to

- **The pattern is the oracle's.** `fuse.py` runs a spec's oracle on lazy arrays and reads the pattern off the graph it writes. Nothing else says which expressions a kernel serves, so the pattern cannot disagree with what the judge checks. An oracle has to be one expression of the operations in [`fuse.py`](../fuse.py) that reads each of its arguments once.
- **One IEEE operation per operation of the expression, in the expression's order.** Each is the instruction NumPy's loop for that operation executes, and each result is rounded by itself. A product and a sum are never fused into a multiply-add, a division is never a product with a reciprocal, and additions are never regrouped. That is the whole argument for the bits being NumPy's, and the judge checks it on every size to 65 and on NaN, infinities, signed zeros, denormals and the largest values.
- **`naive.c` starts with `#pragma STDC FP_CONTRACT OFF`.** Without it the compiler fuses `a + s * b` into a multiply-add at the judge's flags, and the reference itself fails the check.
- **Arguments come in one order:** the input arrays, the output array (unless the kernel returns a number), the length, then the numbers. All float64. [`_fused.c`](../_fused.c) calls every kernel one way and relies on it; `build_fused.py` refuses a spec in another order.
- **Every load of a block comes before its store.** A kernel computes eight elements a step, and reads all it needs of them from every array before it writes them. So the result may be written where one of the arrays is, element for element: `x -= s * y` made in `x`. That holds for a length that is a multiple of eight, and below eight, where the elements are done one at a time. For any other length a kernel finishes with a block that ends with the arrays and overlaps the block before it, computed from the arrays again, which would then read what it had written; `_fused.c` never asks for that, and makes such a call as the whole blocks and then the elements left over. An output that overlaps an array in any other way is declined before anything is written.
- **A reduction adds in NumPy's order.** `np.sum` adds a contiguous array pairwise, in a tree whose shape depends on the length ([`kernels/sum_f64/spec.py`](../../../kernels/sum_f64/spec.py)). A fused reduction makes the same additions in the same order on the values it computes; one that accumulated from left to right would be a different number. The layer calls it only for C-contiguous arrays, which NumPy sums in the order they lie in memory.
- **Floating-point exceptions are the caller's to read.** A kernel raises the flags its instructions raise and reports nothing. The layer reads them after the call, and where NumPy would have warned or raised under the program's `np.errstate` it drops the kernel's result and makes NumPy's own calls, which report it with NumPy's words on the program's line. A comparison is the one place a kernel raises a flag NumPy does not report (invalid, for a NaN); the result is right, and the expression is computed a second time by NumPy.

## What is not specified

**Which NaN comes of two.** When an operation meets two NaNs of different sign or payload the processor returns its first operand's. Which operand is first in NumPy's loop is its compiler's choice, and a pattern does not record how the program ordered a sum's operands. Whether a result is NaN is always NumPy's; which NaN it is, is NumPy's whenever a program has one kind of NaN, which is what arithmetic and `np.nan` give. The judge's NaNs are all one NaN.

## Adding one

1. Find the shape: `uv run --group apps python -m prototypes.lazy.programs.mine` ranks the shapes that occur in the corpus by the time NumPy spends on them.
2. For a shape of the operations `make_kernel.py` has a rule for, add its name to `SHAPES` there and run `uv run python -m prototypes.lazy.make_kernel --all`, then `uv run judge check <name> --unlocked`. That is the whole of it; go to step 5.
3. Otherwise write `spec.py` with the expression as its oracle, and `naive.c`. `uv run judge check <name> --unlocked` must pass on the C before any assembly is written.
4. Write `arm64/kernel.s`, and `attempts.md` as for any kernel ([AGENTS.md](../../../AGENTS.md#writing-or-tuning-a-kernel)). Load every array of a block before the block is stored.
5. `uv run python -m prototypes.lazy.build_fused`, then `uv run --group dev pytest tests/test_lazy_fusion.py`, whose table of expected patterns gains a line for a kernel written by hand.
