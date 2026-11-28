"""An array less the product of two: `a - b * c` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer: the passes NumPy makes for the expression,
`b * c` into a temporary and then `a -` that, as one pass with no temporary.
The pattern it serves is `sub(V0,mul(V1,V2))`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus it is
Black-Scholes, the cavity flow, the horizontal diffusion and the equation of
state.

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

from judge.spec import In, Kernel, Out, Size

# The NaN this processor's invalid operations give: `np.nan`'s bits on ARM64, those with the sign bit set on x86-64.
with np.errstate(invalid="ignore"):
    DEFAULT_NAN = np.float64(np.inf) - np.float64(np.inf)



def oracle(a, b, c):
    """The expression as a program writes it.

    Args:
        a: The array subtracted from, float64.
        b: The first factor, float64, of the same length.
        c: The second factor, the same.

    Returns:
        `a - b * c`, a new float64 array.
    """
    return a - b * c


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
    name="fuse_sub_product_f64",
    returns="void",
    args=[In("a", "f64", prepare=one_nan), In("b", "f64", prepare=one_nan), In("c", "f64", prepare=one_nan), Out("out", "f64"), Size("n")],
    oracle=oracle,
)
