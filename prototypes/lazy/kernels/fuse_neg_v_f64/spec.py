"""`-a` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer, written by `prototypes/lazy/make_kernel.py`
from the name of its shape, `neg(V0)`:
one operation, one pass. It fuses nothing by itself; it is a step of an
expression computed block by block, where every step is a kernel.
`prototypes/lazy/fuse.py` reads the pattern it serves off this file's oracle.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself. What that leaves open, and what a
fused kernel may assume of its caller, is in [the kernels' README](../README.md).
"""
from judge.spec import In, Kernel, Out, Size


def oracle(a):
    """The expression as a program writes it.

    Args:
        a: The array the shape calls V0, float64.

    Returns:
        `-a`, a new float64 array.
    """
    return -a


KERNEL = Kernel(
    name="fuse_neg_v_f64",
    returns="void",
    args=[
        In("a", "f64"),
        Out("out", "f64"),
        Size("n"),
    ],
    oracle=oracle,
)
