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
    ScalarSet("nan", {"s": DEFAULT_NAN}),
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
    name="fuse_laplacian5_f64",
    returns="void",
    args=[In("a", "f64", prepare=one_nan), In("b", "f64", prepare=one_nan), In("c", "f64", prepare=one_nan), In("d", "f64", prepare=one_nan), In("e", "f64", prepare=one_nan), Out("out", "f64"),
          Size("n"), Scalar("s", "f64", S)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
