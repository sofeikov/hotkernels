"""Binary search on integers: `np.searchsorted(a, v)` for n sorted int64 values and any number of int64 needles.

The twin of `searchsorted_f64` for int64. There is no NaN, so the order is
plain signed comparison and every needle has one answer.
"""
import numpy as np

from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# Needles in the default scalar set: one full group of the widest path that
# works inside the cache. This set gets the full sweep and is timed.
NEEDLES = 64
# Needles in the second timed set: several groups and a leftover in every
# path (1,000 leaves 8 over groups of 16, 40 over 64 and 104 over 128).
MANY_NEEDLES = 1000
# A kernel searches for several needles at once, in groups of 4, 16, 64 or
# 128, and handles leftovers apart, so the counts either side of each width
# are the ones that matter. Set name -> number of needles.
BOUNDARY_NEEDLES = {
    "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "k15": 15, "k16": 16, "k17": 17,
    "k63": 63, "k65": 65,
    "k127": 127, "k128": 128, "k129": 129,
}


def oracle(a, v, needles):
    """For each needle, the first index at which it could be inserted to keep `a` sorted."""
    return np.searchsorted(a, v)


KERNEL = Kernel(
    name="searchsorted_i64",
    returns="void",
    args=[
        # The integer value classes, sorted: runs of equal elements (`small`,
        # `equal`), and the type's minimum and maximum at the ends
        # (`extremes`, `min_at`, `max_at`).
        In("a", "i64", prepare=np.sort),
        # Drawn from `a` as well as random: a needle equal to an element is
        # where the two sides of the search differ, and random data never
        # gives one. The other classes give needles below and above every
        # element, and the type's extremes as needles.
        In("v", "i64", len="needles", draws_from="a"),
        Out("out", "i64", capacity="needles"),
        # The length of a. It selects the kernel's code path and the depth of
        # the search, so it is the size that is swept.
        Size("n"),
        Scalar("needles", "usize", NEEDLES),
    ],
    oracle=oracle,
    scalar_sets=[ScalarSet(name, {"needles": count}) for name, count in BOUNDARY_NEEDLES.items()]
    + [ScalarSet("many", {"needles": MANY_NEEDLES}, timed=True)],
)
