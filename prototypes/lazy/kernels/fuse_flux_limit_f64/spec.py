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
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The default threshold and the default value put where the product is above it: the program's two zeros.
LO, FILL = 0.0, 0.0

SCALAR_SETS = [
    ScalarSet("negative_zero_threshold", {"lo": -0.0}),
    ScalarSet("positive_threshold", {"lo": 0.25, "fill": 1.5}),
    ScalarSet("negative_threshold", {"lo": -0.25, "fill": -1.5}),
    ScalarSet("negative_zero_fill", {"fill": -0.0}),
    ScalarSet("infinite_threshold", {"lo": float("inf"), "fill": 2.0}),
    ScalarSet("lowest_threshold", {"lo": float("-inf"), "fill": 2.0}),
    ScalarSet("nan_threshold", {"lo": float("nan"), "fill": 2.0}),
    ScalarSet("nan_fill", {"fill": float("nan")}),
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


KERNEL = Kernel(
    name="fuse_flux_limit_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), In("c", "f64"), In("d", "f64"), Out("out", "f64"), Size("n"),
          Scalar("lo", "f64", LO), Scalar("fill", "f64", FILL)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
