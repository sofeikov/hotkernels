"""`(a + b) * (s0 * c)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer, written by `prototypes/lazy/make_kernel.py`
from the name of its shape, `mul(add(V0,V1),mul(S0,V2))`:
the 3 passes NumPy makes for the expression, each into a temporary, as
one pass with none.
`prototypes/lazy/fuse.py` reads the pattern it serves off this file's oracle.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself. What that leaves open, and what a
fused kernel may assume of its caller, is in [the kernels' README](../README.md).

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
    ScalarSet("nan_s0", {"s0": DEFAULT_NAN}),
]


def oracle(a, b, c, s0):
    """The expression as a program writes it.

    Args:
        a: The array the shape calls V0, float64.
        b: The array the shape calls V1, float64, of the same length.
        c: The array the shape calls V2, float64, of the same length.
        s0: The number the shape calls S0.

    Returns:
        `(a + b) * (s0 * c)`, a new float64 array.
    """
    return (a + b) * (s0 * c)


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
    name="fuse_mul_add_v_v_mul_s_v_f64",
    returns="void",
    args=[
        In("a", "f64", prepare=one_nan),
        In("b", "f64", prepare=one_nan),
        In("c", "f64", prepare=one_nan),
        Out("out", "f64"),
        Size("n"),
        Scalar("s0", "f64", S0),
    ],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
