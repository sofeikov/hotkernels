"""`(a - b) / c` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer, written by `prototypes/lazy/make_kernel.py`
from the name of its shape, `div(sub(V0,V1),V2)`:
the 2 passes NumPy makes for the expression, each into a temporary, as
one pass with none.
`prototypes/lazy/fuse.py` reads the pattern it serves off this file's oracle.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself. What that leaves open, and what a
fused kernel may assume of its caller, is in [the kernels' README](../README.md).
"""
from judge.spec import In, Kernel, Out, Size


def oracle(a, b, c):
    """The expression as a program writes it.

    Args:
        a: The array the shape calls V0, float64.
        b: The array the shape calls V1, float64, of the same length.
        c: The array the shape calls V2, float64, of the same length.

    Returns:
        `(a - b) / c`, a new float64 array.
    """
    return (a - b) / c


KERNEL = Kernel(
    name="fuse_div_sub_v_v_v_f64",
    returns="void",
    args=[
        In("a", "f64"),
        In("b", "f64"),
        In("c", "f64"),
        Out("out", "f64"),
        Size("n"),
    ],
    oracle=oracle,
)
