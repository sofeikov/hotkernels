"""`s0 * (a + (b - (s1 * c)))` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer, written by `prototypes/lazy/make_kernel.py`
from the name of its shape, `mul(S0,add(V0,sub(V1,mul(S1,V2))))`:
the 4 passes NumPy makes for the expression, each into a temporary, as
one pass with none.
`prototypes/lazy/fuse.py` reads the pattern it serves off this file's oracle.

Each element goes through the same IEEE operations in the same order as in
NumPy's passes, each rounded by itself. What that leaves open, and what a
fused kernel may assume of its caller, is in [the kernels' README](../README.md).
"""
from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The default numbers: no power of two, so a product or a quotient with one is rounded.
S0, S1 = 0.37, -1.9
# A number that overflows ordinary values, and one that brings them down to denormals.
HUGE, TINY = 1e300, 1e-310

SCALAR_SETS = [
    ScalarSet("zero_s0", {"s0": 0.0}),
    ScalarSet("negative_zero_s0", {"s0": -0.0}),
    ScalarSet("huge_s0", {"s0": HUGE}),
    ScalarSet("tiny_s0", {"s0": TINY}),
    ScalarSet("infinite_s0", {"s0": float("inf")}),
    ScalarSet("nan_s0", {"s0": float("nan")}),
    ScalarSet("zero_s1", {"s1": 0.0}),
    ScalarSet("negative_zero_s1", {"s1": -0.0}),
    ScalarSet("huge_s1", {"s1": HUGE}),
    ScalarSet("tiny_s1", {"s1": TINY}),
    ScalarSet("infinite_s1", {"s1": float("inf")}),
    ScalarSet("nan_s1", {"s1": float("nan")}),
]


def oracle(a, b, c, s0, s1):
    """The expression as a program writes it.

    Args:
        a: The array the shape calls V0, float64.
        b: The array the shape calls V1, float64, of the same length.
        c: The array the shape calls V2, float64, of the same length.
        s0: The number the shape calls S0.
        s1: The number the shape calls S1.

    Returns:
        `s0 * (a + (b - (s1 * c)))`, a new float64 array.
    """
    return s0 * (a + (b - (s1 * c)))


KERNEL = Kernel(
    name="fuse_mul_s_add_v_sub_v_mul_s_v_f64",
    returns="void",
    args=[
        In("a", "f64"),
        In("b", "f64"),
        In("c", "f64"),
        Out("out", "f64"),
        Size("n"),
        Scalar("s0", "f64", S0),
        Scalar("s1", "f64", S1),
    ],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
