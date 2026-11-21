"""Boolean mask selection: `x[mask]` for 8-bit data (uint8, int8 and bool by their bits)."""
import numpy as np

from judge.spec import USED_RETURN, In, Kernel, Out, Size

# Share of mask elements that are True in each timing distribution.
HALF_SELECTED = 0.5
FEW_SELECTED = 0.05

# Bound of the timing data: every uint8 value.
UINT8_END = 256


def oracle(x, mask):
    """The elements of x where mask is True, as NumPy selects them."""
    return x[mask]


def _masked(fraction):
    """Return a timing distribution in which `fraction` of the mask is True."""
    def make(rng, n):
        """Build random data and a random mask for one timing input."""
        return {"x": rng.integers(0, UINT8_END, size=n, dtype=np.uint8), "mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="compress_u8",
    returns="usize",  # number of elements written
    args=[
        In("x", "u8"),
        In("mask", "bool"),
        # Scratch: the kernel may store more than it keeps, anywhere inside the
        # n-element capacity. Only the returned count of elements is compared.
        Out("out", "u8", used=USED_RETURN, scratch=True),
        Size("n"),
    ],
    oracle=oracle,
    distributions={"half_selected": _masked(HALF_SELECTED), "few_selected": _masked(FEW_SELECTED)},
)
