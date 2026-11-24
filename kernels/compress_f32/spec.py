"""Boolean mask selection: `x[mask]` for float32 data.

Every mask byte is 0 or 1. NumPy takes any non-zero byte of a boolean array
as True; a caller that may hold other bytes checks for them before the call,
and the binding sends such a mask to NumPy.
"""
from judge.spec import USED_RETURN, In, Kernel, Out, Size

# Share of mask elements that are True in each timing distribution.
HALF_SELECTED = 0.5
FEW_SELECTED = 0.05


def oracle(x, mask):
    """The elements of x where mask is True, as NumPy selects them."""
    return x[mask]


def _masked(fraction):
    """Return a timing distribution in which `fraction` of the mask is True."""
    def make(rng, n):
        """Build random data and a random mask for one timing input."""
        return {"x": rng.standard_normal(n).astype("float32"), "mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="compress_f32",
    returns="usize",  # number of elements written
    args=[
        In("x", "f32"),
        In("mask", "bool"),
        # Scratch: the kernel may store more than it keeps, anywhere inside the
        # n-element capacity. Only the returned count of elements is compared.
        Out("out", "f32", used=USED_RETURN, scratch=True),
        Size("n"),
    ],
    oracle=oracle,
    distributions={"half_selected": _masked(HALF_SELECTED), "few_selected": _masked(FEW_SELECTED)},
)
