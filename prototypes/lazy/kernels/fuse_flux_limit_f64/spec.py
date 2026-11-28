"""A flux limiter: `np.where(a * (b - c) > lo, fill, d)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer: the four passes NumPy makes for the
expression, a difference, a product, a comparison into a mask and a
selection, as one pass with no temporary and no mask. The pattern it
serves is `where(gt(mul(V0,sub(V1,V2)),S0),S1,V3)`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus it
is the horizontal diffusion of npbench, which sets a flux to zero wherever
it runs against the gradient: `np.where(res * (x1 - x0) > 0, 0, res)`, with
`res` given twice, as `a` and as `d`.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself, and the selection copies bits. A
product that is NaN compares false, as in NumPy, and `d` is taken. What
that leaves open, and what a fused kernel may assume of its caller, is in
[the kernels' README](../README.md).

The comparison raises the invalid-operation flag for a NaN product, which
NumPy's own comparison clears before it returns. The layer therefore
computes the expression with NumPy's calls whenever the flag is raised and
the program has not silenced it; the kernel's result is right either way.

**The domain** is inputs whose NaNs all have the bits of the NaN this
processor's invalid operations give (`one_nan`), so that every NaN of the
computation is one NaN and which of two an operation returns does not show.
That is NumPy's `np.nan` on ARM64; on x86-64 it has the sign bit set as well.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The NaN this processor's invalid operations give: `np.nan`'s bits on ARM64, those with the sign bit set on x86-64.
with np.errstate(invalid="ignore"):
    DEFAULT_NAN = np.float64(np.inf) - np.float64(np.inf)

# The default threshold and the default value put where the product is above it: the program's two zeros.
LO, FILL = 0.0, 0.0

SCALAR_SETS = [
    ScalarSet("negative_zero_threshold", {"lo": -0.0}),
    ScalarSet("positive_threshold", {"lo": 0.25, "fill": 1.5}),
    ScalarSet("negative_threshold", {"lo": -0.25, "fill": -1.5}),
    ScalarSet("negative_zero_fill", {"fill": -0.0}),
    ScalarSet("infinite_threshold", {"lo": float("inf"), "fill": 2.0}),
    ScalarSet("lowest_threshold", {"lo": float("-inf"), "fill": 2.0}),
    ScalarSet("nan_threshold", {"lo": DEFAULT_NAN, "fill": 2.0}),
    ScalarSet("nan_fill", {"fill": DEFAULT_NAN}),
    ScalarSet("infinite_fill", {"fill": float("inf")}),
]


def oracle(a, b, c, d, lo, fill):
    """The expression as a program writes it.

    Args:
        a: The factor of the product that is tested, float64.
        b: The array subtracted from, float64, of the same length.
        c: The array subtracted, the same.
        d: What is kept where the product is not above the threshold, the same.
        lo: The threshold.
        fill: What is put where the product is above it.

    Returns:
        `np.where(a * (b - c) > lo, fill, d)`, a new float64 array.
    """
    return np.where(a * (b - c) > lo, fill, d)


def one_nan(x):
    """Bring an input into the domain: every NaN in it the NaN this processor's invalid operations give.

    Of two NaNs an operation returns one, chosen by the order of its operands, which a compiler is free to
    swap in a sum or a product, and which NumPy's own loops on x86-64 take one way in blocks of eight and
    the other way in the elements left over. With one NaN the choice does not show.

    Args:
        x: A generated float64 array.

    Returns:
        `x` itself when it holds no NaN; else a copy in which every NaN is DEFAULT_NAN.
    """
    nan = np.isnan(x)
    if not nan.any():
        return x
    y = x.copy()
    y[nan] = DEFAULT_NAN
    return y


KERNEL = Kernel(
    name="fuse_flux_limit_f64",
    returns="void",
    args=[In("a", "f64", prepare=one_nan), In("b", "f64", prepare=one_nan), In("c", "f64", prepare=one_nan), In("d", "f64", prepare=one_nan), Out("out", "f64"), Size("n"),
          Scalar("lo", "f64", LO), Scalar("fill", "f64", FILL)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
