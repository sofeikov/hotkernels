"""A number times the sum of two arrays: `s * (a + b)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer: the passes NumPy makes for the expression,
`a + b` into a temporary and then `s *` that, as one pass with no temporary.
The pattern it serves is `mul(S0,add(V0,V1))`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus it is
an average of two neighbours: the cavity flow, isoneutral mixing, the
transformer block.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself, so the bits are NumPy's for every
input. What that leaves open, and what a fused kernel may assume of its
caller, is in [the kernels' README](../README.md).
"""
from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The default number: no power of two, so a product with it is rounded.
S = 0.37
# A number that overflows ordinary values, and one that brings them down to denormals.
HUGE, TINY = 1e300, 1e-310

SCALAR_SETS = [
    ScalarSet("negative", {"s": -2.5}),
    ScalarSet("zero", {"s": 0.0}),
    ScalarSet("negative_zero", {"s": -0.0}),
    ScalarSet("one", {"s": 1.0}),
    ScalarSet("huge", {"s": HUGE}),
    ScalarSet("tiny", {"s": TINY}),
    ScalarSet("infinite", {"s": float("inf")}),
    ScalarSet("nan", {"s": float("nan")}),
]



def oracle(a, b, s):
    """The expression as a program writes it.

    Args:
        a: One array of the sum, float64.
        b: The other, float64, of the same length.
        s: The multiplier.

    Returns:
        `s * (a + b)`, a new float64 array.
    """
    return s * (a + b)


KERNEL = Kernel(
    name="fuse_scaled_sum_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), Out("out", "f64"), Size("n"), Scalar("s", "f64", S)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
