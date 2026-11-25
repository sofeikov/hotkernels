"""A field less a coefficient times the divergence of two fluxes: `a - b * (c - d + e - f)` on float64,
NumPy's result bit for bit.

A fused kernel of the lazy layer: the passes NumPy makes for the expression,
a difference, a sum, a difference, a product and a difference, each into a
temporary, as one pass with no temporary. The pattern it serves is
`sub(V0,mul(V1,sub(add(V2,sub(V3,V4)),V5)))`, which
`prototypes/lazy/fuse.py` reads off this file's oracle. In the corpus it is
the last line of the horizontal diffusion of npbench.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself, so the bits are NumPy's for every
input. What that leaves open, and what a fused kernel may assume of its
caller, is in [the kernels' README](../README.md).
"""
from judge.spec import In, Kernel, Out, Size



def oracle(a, b, c, d, e, f):
    """The expression as a program writes it.

    Args:
        a: The field, float64.
        b: The coefficient at each point, float64, of the same length.
        c: The first flux at each point's far side.
        d: The first flux at its near side.
        e: The second flux at its far side.
        f: The second flux at its near side.

    Returns:
        `a - b * (c - d + e - f)`, a new float64 array.
    """
    return a - b * (c - d + e - f)


KERNEL = Kernel(
    name="fuse_flux_update_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), In("c", "f64"), In("d", "f64"), In("e", "f64"), In("f", "f64"),
          Out("out", "f64"), Size("n")],
    oracle=oracle,
)
