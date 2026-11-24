"""Indices of the True elements of a boolean array: `np.nonzero(mask)[0]`.

Every mask byte is 0 or 1. NumPy takes any non-zero byte of a boolean array
as True; a caller that may hold other bytes checks for them before the call,
and the binding sends such a mask to NumPy.
"""
import numpy as np

from judge.spec import USED_RETURN, In, Kernel, Out, Size

# Share of mask elements that are True in each timing distribution.
HALF_SELECTED = 0.5
FEW_SELECTED = 0.05


def oracle(mask):
    """The indices where mask is True, ascending, as int64."""
    return np.nonzero(mask)[0]


def _masked(fraction):
    """Return a timing distribution in which `fraction` of the mask is True."""
    def make(rng, n):
        """Build a random mask for one timing input."""
        return {"mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="nonzero_bool",
    returns="usize",  # number of indices written
    args=[
        In("mask", "bool"),
        # Scratch: see compress_f32. Only the returned count of indices is compared.
        Out("out", "i64", used=USED_RETURN, scratch=True),
        Size("n"),
    ],
    oracle=oracle,
    distributions={"half_selected": _masked(HALF_SELECTED), "few_selected": _masked(FEW_SELECTED)},
)
