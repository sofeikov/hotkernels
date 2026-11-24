"""Clamp to a range: `np.clip(x, LO, HI)` for float64 data.

`out` does not overlap `x`, not even as `x` itself: in place the ARM64
kernels redo part of the array from `x` after storing to `out` and give a
wrong zero or NaN (issue 7). The binding clips in place through a buffer of
its own.

The kernel is defined for `lo <= hi` with neither bound NaN. Outside that
NumPy returns `hi` for every non-NaN element (reversed bounds) or NaN (a NaN
bound) and `naive.c` does not, so whatever calls the kernel must send such
calls to NumPy.

The judge checks the pairs of bounds in SCALAR_SETS as well as the default pair,
and times the default pair and `zero_lo`, each on the three kinds of data in
DISTRIBUTIONS. Negative zeros are among them because a zero lower bound is the one
case where NumPy's result depends on the sign of a zero element, and a kernel
that is fast for such a bound has to do something about them.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

LO, HI = -0.5, 0.5

# A denormal magnitude and one near the largest finite value of the type.
TINY, BIG = 1e-310, 1e308

# Further pairs of bounds, all with lo <= hi. The zero bounds matter most:
# NumPy keeps the sign of a zero element that a bound of the other sign would
# otherwise replace, and `np.clip(x, 0, 1)` is probably the commonest call.
SCALAR_SETS = [
    ScalarSet("zero_lo", {"lo": 0.0, "hi": 1.0}, timed=True),
    ScalarSet("neg_zero_lo", {"lo": -0.0, "hi": 1.0}),
    ScalarSet("zero_hi", {"lo": -1.0, "hi": 0.0}),
    ScalarSet("neg_zero_hi", {"lo": -1.0, "hi": -0.0}),
    ScalarSet("both_zero", {"lo": 0.0, "hi": 0.0}),
    ScalarSet("zeros_apart", {"lo": -0.0, "hi": 0.0}),
    ScalarSet("equal", {"lo": 0.25, "hi": 0.25}),
    ScalarSet("infinite", {"lo": float("-inf"), "hi": float("inf")}),
    ScalarSet("half_open", {"lo": 0.5, "hi": float("inf")}),
    ScalarSet("denormal", {"lo": -TINY, "hi": TINY}),
    ScalarSet("wide", {"lo": -BIG, "hi": BIG}),
]


# Share of elements that are -0.0 in the "sparse_negative_zeros" timing data.
SPARSE_NEGATIVE_ZERO_SHARE = 0.001


def _random(rng, n):
    """Timing data: normally distributed values, none of them a zero."""
    return {"x": rng.standard_normal(n).astype(np.float64)}


def _sparse_negative_zeros(rng, n):
    """Timing data: normally distributed values with an occasional -0.0."""
    x = rng.standard_normal(n).astype(np.float64)
    x[rng.random(n) < SPARSE_NEGATIVE_ZERO_SHARE] = -0.0
    return {"x": x}


def _rounded(rng, n):
    """Timing data: normal values rounded to integers, as counts or labels are.

    Rounding a value between -0.5 and 0 gives -0.0, so about a fifth of the
    elements are negative zeros and another fifth positive zeros.
    """
    return {"x": np.round(rng.standard_normal(n)).astype(np.float64)}


DISTRIBUTIONS = {
    "random": _random,
    "sparse_negative_zeros": _sparse_negative_zeros,
    "rounded": _rounded,
}


def oracle(x, lo, hi):
    """x with every element below lo raised to lo and every element above hi lowered to hi."""
    return np.clip(x, lo, hi)


KERNEL = Kernel(
    name="clip_f64",
    returns="void",
    args=[In("x", "f64"), Out("out", "f64"), Size("n"), Scalar("lo", "f64", LO), Scalar("hi", "f64", HI)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
    distributions=DISTRIBUTIONS,
)
