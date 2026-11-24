"""Running sum: `np.cumsum(x)` for float32 data, within a declared tolerance.

`out` does not overlap `x`, not even as `x` itself: the ARM64 kernels redo a
chunk that fails their fast path from `x` after storing it, and in place give
wrong sums after an infinity or a NaN (issue 7). The binding takes the kernel
only for a call without `out`, which writes into a new array.

A fast running sum adds the elements in a different order from NumPy, which
rounds differently, so the output is compared with a tolerance. NaN and
infinity must still appear exactly where NumPy has them.
"""
import numpy as np

from judge.spec import TOLERANCE_SCALE, TOLERANCE_ULP, In, Kernel, Out, Size

# Allowed difference from NumPy, in units in the last place of `error_scale`.
# Adding the first i + 1 elements in any order is off from their true sum by at
# most about i * e/2 * (|x[0]| + ... + |x[i]|), where e is the gap between 1.0
# and the next float. Two orders can therefore differ by twice that. A unit in
# the last place of a value v is at least v * e/2, so two units of the scale
# cover it; four leave room for the terms of the bound left out here.
ALLOWED_ULP = 4


def oracle(x):
    """The running sum of x, accumulated left to right in the type of x."""
    return np.cumsum(x)


def error_scale(x):
    """The magnitude the allowed error of each output element is measured at.

    Args:
        x: The input array.

    Returns:
        Per element i, (i + 1) * (|x[0]| + ... + |x[i]|) as float64. The size
        of out[i] itself is no use as a scale: terms that cancel leave a small
        result carrying the rounding error of the large terms.
    """
    magnitudes = np.cumsum(np.abs(x.astype(np.float64)))
    return magnitudes * np.arange(1, len(x) + 1)


KERNEL = Kernel(
    name="cumsum_f32",
    returns="void",
    args=[In("x", "f32"), Out("out", "f32"), Size("n")],
    oracle=oracle,
    mode={TOLERANCE_ULP: ALLOWED_ULP, TOLERANCE_SCALE: error_scale},
)
