"""A five-point Laplacian: `s * a - (b + c + d + e)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer: the passes NumPy makes for the expression,
`s * a`, three sums and a difference, each into a temporary, as one pass
with no temporary. The pattern it serves is
`sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus it is
the horizontal diffusion of npbench, where it is the heaviest expression of
the corpus.

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



def oracle(a, b, c, d, e, s):
    """The expression as a program writes it.

    Args:
        a: The centre points, float64.
        b: One neighbour of each, float64, of the same length.
        c: The second neighbour.
        d: The third.
        e: The fourth.
        s: The centre's weight: 4 for a Laplacian.

    Returns:
        `s * a - (b + c + d + e)`, a new float64 array.
    """
    return s * a - (b + c + d + e)


KERNEL = Kernel(
    name="fuse_laplacian5_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), In("c", "f64"), In("d", "f64"), In("e", "f64"), Out("out", "f64"),
          Size("n"), Scalar("s", "f64", S)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
