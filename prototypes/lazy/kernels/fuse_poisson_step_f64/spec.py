"""One Jacobi step of a pressure Poisson equation: `((a + b) * s0 + (c + d) * s1) / s2 - s3 * e` on float64.

NumPy's result bit for bit. A fused kernel of the lazy layer: the eight
passes NumPy makes for the expression, each into a temporary, as one pass
with none. The pattern it serves is
`sub(div(add(mul(S0,add(V0,V1)),mul(S1,add(V2,V3))),S2),mul(S3,V4))`,
which `prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus
it is the inner loop of the cavity flow of CFD Python:

    p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                      (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                     (2 * (dx**2 + dy**2)) -
                     dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

with the four numbers worked out by Python before NumPy sees them.

Each element goes through the same eight IEEE operations in the same order
as in NumPy's passes, each rounded by itself: the division is a division,
not a product with a reciprocal. What that leaves open, and what a fused
kernel may assume of its caller, is in [the kernels' README](../README.md).

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

# The default numbers: none a power of two, so every product and the quotient are rounded.
S0, S1, S2, S3 = 0.37, 0.71, 2.3, 0.11
# A number that overflows ordinary values, and one that brings them down to denormals.
HUGE, TINY = 1e300, 1e-310

SCALAR_SETS = [
    ScalarSet("grid", {"s0": 2.5e-5, "s1": 2.5e-5, "s2": 1e-4, "s3": 6.25e-6}),
    ScalarSet("negative", {"s0": -0.37, "s1": 0.71, "s2": -2.3, "s3": -0.11}),
    ScalarSet("zero_weights", {"s0": 0.0, "s1": -0.0, "s3": 0.0}),
    ScalarSet("zero_divisor", {"s2": 0.0}),
    ScalarSet("negative_zero_divisor", {"s2": -0.0}),
    ScalarSet("ones", {"s0": 1.0, "s1": 1.0, "s2": 1.0, "s3": 1.0}),
    ScalarSet("huge", {"s0": HUGE, "s1": HUGE, "s3": HUGE}),
    ScalarSet("tiny", {"s0": TINY, "s1": TINY, "s2": HUGE, "s3": TINY}),
    ScalarSet("tiny_divisor", {"s2": TINY}),
    ScalarSet("infinite", {"s0": float("inf"), "s2": float("inf")}),
    ScalarSet("nan", {"s1": DEFAULT_NAN}),
    ScalarSet("nan_divisor", {"s2": DEFAULT_NAN}),
]


def oracle(a, b, c, d, e, s0, s1, s2, s3):
    """The expression as a program writes it.

    Args:
        a: One neighbour along the first direction, float64.
        b: The other, float64, of the same length.
        c: One neighbour along the second direction, the same.
        d: The other, the same.
        e: The right-hand side at each point, the same.
        s0: The weight of the first pair's sum.
        s1: The weight of the second pair's sum.
        s2: What the weighted sum is divided by.
        s3: The weight of the right-hand side.

    Returns:
        `((a + b) * s0 + (c + d) * s1) / s2 - s3 * e`, a new float64 array.
    """
    return ((a + b) * s0 + (c + d) * s1) / s2 - s3 * e


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
    name="fuse_poisson_step_f64",
    returns="void",
    args=[In("a", "f64", prepare=one_nan), In("b", "f64", prepare=one_nan), In("c", "f64", prepare=one_nan), In("d", "f64", prepare=one_nan), In("e", "f64", prepare=one_nan), Out("out", "f64"), Size("n"),
          Scalar("s0", "f64", S0), Scalar("s1", "f64", S1), Scalar("s2", "f64", S2), Scalar("s3", "f64", S3)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
