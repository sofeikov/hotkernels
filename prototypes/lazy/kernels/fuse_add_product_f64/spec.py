"""An array plus the product of two: `a + b * c` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer: the passes NumPy makes for the expression,
`b * c` into a temporary and then `a +` that, as one pass with no temporary.
The pattern it serves is `add(V0,mul(V1,V2))`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus it is
the Veros routines and the particle filter.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself, so the bits are NumPy's for every
input. What that leaves open, and what a fused kernel may assume of its
caller, is in [the kernels' README](../README.md).
"""
from judge.spec import In, Kernel, Out, Size



def oracle(a, b, c):
    """The expression as a program writes it.

    Args:
        a: The array added to, float64.
        b: The first factor, float64, of the same length.
        c: The second factor, the same.

    Returns:
        `a + b * c`, a new float64 array.
    """
    return a + b * c


KERNEL = Kernel(
    name="fuse_add_product_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), In("c", "f64"), Out("out", "f64"), Size("n")],
    oracle=oracle,
)
