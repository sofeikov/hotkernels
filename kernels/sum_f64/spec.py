"""Sum of a float64 array: `np.sum(x)`, NumPy's result bit for bit.

NumPy adds the elements of a contiguous array in a fixed order, its pairwise
summation (`pairwise_sum` in `numpy/_core/src/umath/loops_utils.h.src`, the
same in NumPy 2.3 to 2.5), so a kernel that makes the same additions gives
the same bits, and the spec is exact:

- Fewer than 8 elements are added one after another into a sum that starts
  at +0.0.
- Up to 128 elements are added into eight running sums, element i into sum
  i mod 8, which are combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) +
  (r6 + r7)); the last n mod 8 elements are added to that one after another.
- More than 128 elements are split at n / 2 rounded down to a multiple of 8,
  and the sums of the two halves are added.
- The reduction starts from +0.0: the result is 0 + pairwise(x, n), so a sum
  of negative zeros is +0.0, and the sum of no elements is +0.0.

The operands of each addition may be swapped, since a float addition of
finite values is commutative, but no addition may be fused, reassociated or
done at a higher precision.

**The domain** is the inputs whose sum in that order is finite. Once one
addition gives an infinity or a NaN, every later one does too, so these are
the inputs whose additions are all finite. Outside it the result depends on
things that are not the order of the additions: which NaN of two an addition
returns, which the compiler decides by the order it put the operands in, and
NumPy's warning on overflow. The binding sends a call to NumPy whenever the
kernel's result is not finite, so a kernel only has to give a result that is
not finite there too, which any kernel making the same additions does.
Signed zeros, denormals and the largest finite values are in the domain.

`x` is passed through `_finite_sum` before every call the judge makes, which
leaves an input in the domain as it is and brings any other into it.

The judge sweeps every size to 65 and then powers of two and their
neighbours. The tree's shape depends on n in an irregular way, so a kernel
should also be compared with `np.sum` at every size to a few thousand and at
random sizes to several million; the judge's fresh cases reach 5,000.
"""
import numpy as np

from judge.spec import In, Kernel, Size


def oracle(x):
    """The sum of the elements of x.

    Args:
        x: The input array, float64, any length.

    Returns:
        The sum as a float64 scalar, added in NumPy's pairwise order.
    """
    return np.sum(x)


def _finite_sum(x):
    """Bring an input into the domain: an array whose sum in NumPy's order is finite.

    Args:
        x: A generated float64 array.

    Returns:
        `x` itself when its sum is finite. Otherwise a new array in which
        every NaN and infinity is +0.0 and the rest is halved until the
        sum is finite: an array of the largest values keeps
        values that are large, and one of zeros, denormals or ordinary
        values with a NaN in it keeps them.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        if np.isfinite(np.sum(x)):
            return x
        y = np.where(np.isfinite(x), x, np.float64(0.0)).astype(np.float64)
        while not np.isfinite(np.sum(y)):
            y = y * np.float64(0.5)
    return y


KERNEL = Kernel(
    name="sum_f64",
    returns="f64",
    args=[In("x", "f64", prepare=_finite_sum), Size("n")],
    oracle=oracle,
)
