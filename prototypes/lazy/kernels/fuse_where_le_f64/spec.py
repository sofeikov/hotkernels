"""A selection by a threshold: `np.where(a <= s, b, c)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer: the two passes NumPy makes for the
expression, a comparison into a mask and a selection, as one pass with no
mask in memory. The pattern it serves is `where(le(V0,S0),V1,V2)`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. It is the last
step of a program and not much by itself: what it selects between is what
the steps before it made, a block at a time. In the programs from other
codebases it is Matplotlib's overlay blend,
`np.where(rgb <= 0.5, 2 * intensity * rgb, 1 - 2 * (1 - intensity) * (1 - rgb))`.

The selection copies bits, so a NaN of `b` or `c` comes through with its
sign and payload, and a zero with its sign. A NaN in `a` is not below
anything, so `c` is taken, as NumPy's comparison and `np.where` give.

The comparison is meant as the quiet one, which raises no flag for a NaN,
where C's `<=` raises the invalid-operation flag. NumPy's own comparison
clears that flag before it returns and reports nothing, so a kernel that
never raises it is NumPy's behaviour, and the layer keeps its result for
an image with NaNs in it. The x86-64 assembly compares so (`vcmppd` with
`_CMP_LE_OQ`). The C reference says so too (`islessequal`), but a compiler
that takes the flags for unread may compile it to the signalling
comparison, and clang does on x86-64; where the kernel is built from the
reference and raises the flag, the layer drops its result and makes
NumPy's calls, as for every flag NumPy would not have reported, so the
values are NumPy's either way. What a fused kernel may assume of its
caller is in [the kernels' README](../README.md).
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The default threshold: the middle of a colour channel between 0 and 1, the overlay blend's.
THRESHOLD = 0.5

SCALAR_SETS = [
    ScalarSet("zero_threshold", {"s": 0.0}),
    ScalarSet("negative_zero_threshold", {"s": -0.0}),
    ScalarSet("negative_threshold", {"s": -0.25}),
    ScalarSet("infinite_threshold", {"s": float("inf")}),
    ScalarSet("lowest_threshold", {"s": float("-inf")}),
    ScalarSet("nan_threshold", {"s": float("nan")}),
]


def oracle(a, b, c, s):
    """The expression as a program writes it.

    Args:
        a: The array that is tested, float64.
        b: What is taken where `a` is at most the threshold, float64, of the same length.
        c: What is taken elsewhere, a NaN of `a` included; the same.
        s: The threshold.

    Returns:
        `np.where(a <= s, b, c)`, a new float64 array.
    """
    return np.where(a <= s, b, c)


KERNEL = Kernel(
    name="fuse_where_le_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), In("c", "f64"), Out("out", "f64"), Size("n"), Scalar("s", "f64", THRESHOLD)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
