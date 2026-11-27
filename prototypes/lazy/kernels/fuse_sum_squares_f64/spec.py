"""The sum of the squares of a float64 array: `np.sum(x * x)`, NumPy's result bit for bit.

A fused kernel of the lazy layer: the two passes NumPy makes for the
expression, `x * x` into a temporary and then the sum of that, as one pass
with no temporary. The pattern it serves is `sum(sq(V0))`, which
`prototypes/lazy/fuse.py` reads off this file's oracle; `np.sum(x ** 2)`,
`(x ** 2).sum()` and `np.square(x).sum()` are the same pattern, and the mean
of the squares is this sum divided by the number of elements, as NumPy
computes a mean. In the corpus it is the L2 penalty of the regressions and
of the perceptron, and the effective sample size of the particle filter.

**The order of the additions is NumPy's.** NumPy sums a contiguous array
pairwise (`pairwise_sum` in `numpy/_core/src/umath/loops_utils.h.src`;
`kernels/sum_f64/spec.py` has the rules), and a fused reduction is exact
only because it makes the same additions in the same order, on the squares:

- Fewer than 8 squares are added one after another into a sum that starts
  at +0.0.
- Up to 128 are added into eight running sums, square i into sum i mod 8,
  which are combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
  the last n mod 8 squares are added to that one after another.
- More than 128 are split at n / 2 rounded down to a multiple of 8, and the
  sums of the two halves are added.
- The reduction starts from +0.0: the result is 0 + pairwise.

Each square is one multiplication, rounded before it is added: no
multiply-add is fused.

**The domain is every input.** A square is never negative, so nothing
cancels: finite inputs give a finite sum or +inf, an infinity gives +inf,
and a NaN gives NaN. The judge's NaNs are all one NaN; which NaN comes of
two different ones is not specified, as for every fused kernel
([the kernels' README](../README.md)).

The layer calls it for arrays of any shape that are C-contiguous, which
NumPy also sums in the order they lie in memory.
"""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """The expression as a program writes it.

    Args:
        x: The array, float64, any length.

    Returns:
        `np.sum(x * x)` as a float64 scalar: the squares added in NumPy's pairwise order.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sum(x * x)


KERNEL = Kernel(
    name="fuse_sum_squares_f64",
    returns="f64",
    args=[In("x", "f64"), Size("n")],
    oracle=oracle,
)
