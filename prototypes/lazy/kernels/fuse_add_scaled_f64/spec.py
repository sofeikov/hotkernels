"""An array plus a number times an array: `a + s * b` on float64, NumPy's result bit for bit.

A fused kernel of the lazy layer (docs/lazy-fusion-design.md): the two
passes NumPy makes for the expression, `s * b` into a temporary and then
`a +` that, as one pass with no temporary. The pattern it serves is
`add(V0,mul(S0,V1))`, which `prototypes/lazy/fuse.py` reads off this
file's oracle.

**Why the bits are NumPy's.** Each element goes through the same two IEEE
operations in the same order: the product is rounded to a float64, then the
sum is. No multiply-add is fused, which would round once. The domain is
every input: infinities, NaN, signed zeros and denormals give what the two
instructions give, which is what NumPy's loops execute.

**One thing is not specified:** which NaN comes out when an operation meets
two NaNs of different sign or payload. The processor returns its first
operand's, so the answer depends on the order of the operands of each
instruction, which NumPy's compiler chose and a pattern does not record
(`a + s * b` and `s * b + a` are one pattern). The judge's NaNs are all one
NaN, and so are a program's unless it makes others on purpose.

`out` does not overlap `a` or `b`. The layer always gives a fused kernel an
output of its own.
"""
from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# The default multiplier: no power of two, so the product is rounded.
S = 0.37
# A multiplier that overflows ordinary values, and one that brings them down to denormals.
HUGE, TINY = 1e300, 1e-310

SCALAR_SETS = [
    ScalarSet("negative", {"s": -2.5}),
    ScalarSet("zero", {"s": 0.0}),
    ScalarSet("negative_zero", {"s": -0.0}),
    ScalarSet("one", {"s": 1.0}),
    ScalarSet("huge", {"s": HUGE}),
    ScalarSet("tiny", {"s": TINY}),
    ScalarSet("infinite", {"s": float("inf")}),
    ScalarSet("nan", {"s": float("nan")}),
]


def oracle(a, b, s):
    """The expression as a program writes it.

    Args:
        a: The array added to, float64.
        b: The array multiplied, float64, of the same length.
        s: The multiplier.

    Returns:
        `a + s * b`, a new float64 array.
    """
    return a + s * b


KERNEL = Kernel(
    name="fuse_add_scaled_f64",
    returns="void",
    args=[In("a", "f64"), In("b", "f64"), Out("out", "f64"), Size("n"), Scalar("s", "f64", S)],
    oracle=oracle,
    scalar_sets=SCALAR_SETS,
)
