"""Hypotenuse: `np.hypot(a, b)` for float64 data.

`out[i]` is `sqrt(a[i]**2 + b[i]**2)` as the C library's `hypot` computes it,
without overflow or underflow in the intermediate squares. NumPy calls that
function for every element, so the reference, `naive.c`, is a loop that calls
it too, and is NumPy's result on every platform.

**Exact against one algorithm.** `hypot` is not correctly rounded, and C
libraries differ in which of two neighbouring doubles they return for a
fraction of a percent to about ten percent of pairs. A kernel matches NumPy
bit for bit only where it computes the same sequence of operations as the C
library it runs beside. The x86-64 kernel computes the corrected algorithm of
glibc's `e_hypot.c`, which glibc uses from about version 2.35, without fused
multiply-add: it passes on such a system and fails elsewhere. A binding that
ships it verifies it against the C library where it runs.

`out` may be exactly `a` or exactly `b`, as when NumPy's loop is called in
place; it does not overlap either in any other way.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Powers of two the `mixed` distribution spreads its magnitudes over: from
# 2^-MIXED_EXPONENT to 2^MIXED_EXPONENT. Wide enough that for many pairs one
# input is less than 2^-54 of the other, where the result is the larger one,
# and well inside the range where no scaling is needed (2^-459 to 2^511).
MIXED_EXPONENT = 80


def oracle(a, b):
    """The hypotenuse of each pair of elements of a and b."""
    return np.hypot(a, b)


def _random(rng, n):
    """Build one timing input of standard normal values.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count of each array.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}


def _spread(rng, n):
    """Return n standard normal values scaled by random powers of two.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count.

    Returns:
        A float64 array whose magnitudes range from about 2^-MIXED_EXPONENT
        to 2^MIXED_EXPONENT.
    """
    exponents = rng.integers(-MIXED_EXPONENT, MIXED_EXPONENT + 1, n)
    return np.ldexp(rng.standard_normal(n), exponents)


def _mixed(rng, n):
    """Build one timing input whose magnitudes are spread over many powers of two.

    Args:
        rng: The judge's NumPy random generator.
        n: Element count of each array.

    Returns:
        The kernel's input arrays by argument name.
    """
    return {"a": _spread(rng, n), "b": _spread(rng, n)}


KERNEL = Kernel(
    name="hypot_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), Out("out", "f64"), Size("n")],
    oracle=oracle,
    distributions={"random": _random, "mixed": _mixed},
)
