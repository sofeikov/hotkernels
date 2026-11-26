"""`np.square(a) + np.square(b)` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer, written by `prototypes/lazy/make_kernel.py`
from the name of its shape, `add(sq(V0),sq(V1))`:
the 3 passes NumPy makes for the expression, each into a temporary, as
one pass with none.
`prototypes/lazy/fuse.py` reads the pattern it serves off this file's oracle.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself. What that leaves open, and what a
fused kernel may assume of its caller, is in [the kernels' README](../README.md).
"""
import numpy as np

from judge.spec import In, Kernel, Out, Size


def oracle(a, b):
    """The expression as a program writes it.

    Args:
        a: The array the shape calls V0, float64.
        b: The array the shape calls V1, float64, of the same length.

    Returns:
        `np.square(a) + np.square(b)`, a new float64 array.
    """
    return np.square(a) + np.square(b)


KERNEL = Kernel(
    name="fuse_add_sq_v_sq_v_f64",
    returns="void",
    args=[
        In("a", "f64"),
        In("b", "f64"),
        Out("out", "f64"),
        Size("n"),
    ],
    oracle=oracle,
)
