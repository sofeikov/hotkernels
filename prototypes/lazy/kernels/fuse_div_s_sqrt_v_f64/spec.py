"""`s0 / np.sqrt(a)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer, written by `prototypes/lazy/make_kernel.py`
from the name of its shape, `div(S0,sqrt(V0))`:
the 2 passes NumPy makes for the expression, each into a temporary, as
one pass with none.
`prototypes/lazy/fuse.py` reads the pattern it serves off this file's oracle.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself. What that leaves open, and what a
fused kernel may assume of its caller, is in [the kernels' README](../README.md).
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The default number: no power of two, so a product or a quotient with one is rounded.
S0 = 0.37
# A number that overflows ordinary values, and one that brings them down to denormals.
HUGE, TINY = 1e300, 1e-310

SCALAR_SETS = [
    ScalarSet("zero_s0", {"s0": 0.0}),
    ScalarSet("negative_zero_s0", {"s0": -0.0}),
    ScalarSet("huge_s0", {"s0": HUGE}),
    ScalarSet("tiny_s0", {"s0": TINY}),
    ScalarSet("infinite_s0", {"s0": float("inf")}),
    ScalarSet("nan_s0", {"s0": float("nan")}),
]


def oracle(a, s0):
    """The expression as a program writes it.

    Args:
        a: The array the shape calls V0, float64.
        s0: The number the shape calls S0.

    Returns:
        `s0 / np.sqrt(a)`, a new float64 array.
    """
    return s0 / np.sqrt(a)


KERNEL = Kernel(
    name="fuse_div_s_sqrt_v_f64",
    returns="void",
    args=[
        In("a", "f64"),
        Out("out", "f64"),
        Size("n"),
        Scalar("s0", "f64", S0),
    ],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
