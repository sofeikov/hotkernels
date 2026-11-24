"""Masked assignment of a scalar: `x[mask] = VALUE` on float32, in place.

Every mask byte is 0 or 1. NumPy takes any non-zero byte of a boolean array
as True; a caller that may hold other bytes checks for them before the call,
and the binding sends such a mask to NumPy.

The judge checks the values in SCALAR_SETS as well as the default one, and times
the default.
"""
from judge.spec import In, InOut, Kernel, Scalar, ScalarSet, Size

VALUE = 0.0

# Further values to assign. The kernel must store the value's bits unchanged,
# so the ones here are those an arithmetic shortcut would get wrong.
SCALAR_SETS = [
    ScalarSet("negative_zero", {"value": -0.0}),
    ScalarSet("ordinary", {"value": 1.5}),
    ScalarSet("nan", {"value": float("nan")}),
    ScalarSet("infinity", {"value": float("-inf")}),
    # The smallest positive float32 denormal, 2**-149.
    ScalarSet("denormal", {"value": 2.0**-149}),
]

# Share of mask elements that are True in each timing distribution.
HALF_SELECTED = 0.5
FEW_SELECTED = 0.05


def oracle(x, mask, value):
    """A copy of x with value written wherever mask is True."""
    result = x.copy()
    result[mask] = value
    return result


def assign(x, mask, value):
    """The statement a user writes, timed as NumPy. Applying it twice changes nothing."""
    x[mask] = value


def _masked(fraction):
    """Return a timing distribution in which `fraction` of the mask is True."""
    def make(rng, n):
        """Build random data and a random mask for one timing input."""
        return {"x": rng.standard_normal(n).astype("float32"), "mask": rng.random(n) < fraction}
    return make


KERNEL = Kernel(
    name="maskfill_f32",
    returns="void",
    args=[InOut("x", "f32"), In("mask", "bool"), Size("n"), Scalar("value", "f32", VALUE)],
    oracle=oracle,
    numpy=assign,
    scalar_sets=SCALAR_SETS,
    distributions={"half_selected": _masked(HALF_SELECTED), "few_selected": _masked(FEW_SELECTED)},
)
