"""Membership in a small set: `np.isin(x, test)` on int32 with TEST_VALUES int32 test values."""
import numpy as np

from judge.spec import In, Kernel, Out, Size

# Number of test values. Small sets are where NumPy's sort-based method is slowest
# relative to comparing against every value.
TEST_VALUES = 8


def oracle(x, test):
    """For each element of x, whether it equals any of the test values."""
    return np.isin(x, test)


def _padded(kept, order=None):
    """Make a value class of test values padded the way the binding pads a short set.

    `np.isin(x, [1, 2, 3])` reaches the kernel as `[1, 2, 3, 1, 1, 1, 1, 1]`:
    the binding fills the set to TEST_VALUES by repeating its first value
    (`isin_test_values` in `hotkernels/src/hotkernels/_binding/hook_function.c`).
    Most calls from Python are therefore made with repeated test values.

    Args:
        kept: How many of the random test values are kept, from 1 to
            TEST_VALUES - 1. The rest are replaced by the first.
        order: Where the first value lies among the others, which matters
            because the kernel chooses its path from `test[0]` and the spread
            of the set: FIRST_SMALLEST, FIRST_LARGEST, or None to leave the
            values in the order they were drawn.

    Returns:
        A function from the random test values to the padded set.
    """
    def build(test):
        """Order the values as asked, then repeat the first over the tail."""
        padded = np.sort(test) if order else test.copy()
        if order == FIRST_LARGEST:
            padded = padded[::-1].copy()
        padded[kept:] = padded[0]
        return padded
    return build


def _pairs(test):
    """Half the test values, each one twice: repeats that are not the first value."""
    return np.repeat(test[: TEST_VALUES // 2], 2)


# Orders `_padded` can put the values in before padding.
FIRST_SMALLEST = "first_smallest"
FIRST_LARGEST = "first_largest"

# Value classes of the test values, next to the general integer ones. The
# general ones already give eight equal values (`equal`) and a few values
# repeated at random (`small`); these give the shapes the binding sends.
TEST_CLASSES = {
    "padded_1": _padded(1),
    "padded_2": _padded(2),
    "padded_5": _padded(5),
    "padded_7": _padded(7),
    "padded_3_first_smallest": _padded(3, FIRST_SMALLEST),
    "padded_3_first_largest": _padded(3, FIRST_LARGEST),
    "pairs": _pairs,
}


def _mostly_misses(rng, n):
    """Timing input: values and test values drawn from a range wide enough that few match."""
    return {"x": rng.integers(0, 1000, n, dtype=np.int32),
            "test": rng.integers(0, 1000, TEST_VALUES, dtype=np.int32)}


def _wide_values(rng, n):
    """Timing input: values and test values spread over the whole int32 range.

    Small non-negative values, as in `mostly_misses`, allow tricks that narrow
    the comparison. This distribution shows what a kernel does when they do not
    apply. Half of x is drawn from the test values so that matches occur.
    """
    info = np.iinfo(np.int32)
    test = rng.integers(info.min, info.max, TEST_VALUES, dtype=np.int32)
    x = rng.integers(info.min, info.max, n, dtype=np.int32)
    hit = rng.random(n) < 0.5
    x[hit] = rng.choice(test, int(hit.sum()))
    return {"x": x, "test": test}


KERNEL = Kernel(
    name="isin_i32",
    returns="void",
    args=[In("x", "i32"), In("test", "i32", len=TEST_VALUES, classes=TEST_CLASSES), Out("out", "bool"), Size("n")],
    oracle=oracle,
    distributions={"mostly_misses": _mostly_misses, "wide_values": _wide_values},
)
