"""Row and column of each True element of a boolean matrix: `np.argwhere(m)`, the pairs behind `np.nonzero(m)`.

`m` holds `n` elements in rows of `width`, in C order. For each True element,
in C order, the kernel writes its row and its column to `out` as two int64:
row first, then column. That is `np.argwhere(m).ravel()`, and it is the
layout NumPy itself builds for `np.nonzero` on a matrix: one int64 array of
(row, column) pairs, of which the two arrays of the tuple are views with a
stride of 16 bytes. The kernel returns how many int64 it wrote, which is two
per True element.

A partial last row, when `n` is not a whole number of rows, is a row like the
others: element `i` is at row `i // width` and column `i % width` wherever it
is. A caller that passes whole rows, as the binding does, never sees the
difference. When `width` is larger than `n` every element is in row 0.

The kernel is defined for `width >= 1`. Every byte of `m` is 0 or 1, as in
`nonzero_bool`; a caller that may hold other non-zero bytes checks for them
before the call, and the binding sends such a mask to NumPy.

`out` has room for `2 * n` int64 and is scratch, as `nonzero_bool`'s output
is: the kernel may write anywhere inside it, and only the reported elements
are compared. A caller allocates the full capacity and copies out what is
reported.

The judge sweeps `n` for each width in SCALAR_SETS, so no rows, one row, a
partial last row and a width larger than the whole matrix are covered at
every width. It times the default width and the sets marked `timed`, each at
a sparse and a dense share of True.
"""
import numpy as np

from judge.spec import USED_RETURN, In, Kernel, Out, Scalar, ScalarSet, Size, Times

# Row width in the default scalar set: the matrices the kernel was asked for
# hold about 30 to 50 rows of 22 columns.
DEFAULT_WIDTH = 22

# Further row widths that are checked and not timed: the narrowest, and the
# widths either side of each block size a kernel may work in (8, 16, 32 and
# 64 bytes), then two that are several blocks long.
CHECKED_WIDTHS = (1, 2, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 100)
# Row widths that are timed as well, so that a path a kernel takes for a
# width class is seen by `compare`: a narrow row of fewer than 8 elements, a
# row that is a whole number of 16-byte vectors, and a wide row.
TIMED_WIDTHS = (3, 32, 1000)

# Output elements per True element: its row and its column.
PAIR = 2

# Share of elements that are True in each timing distribution: the survey's
# two shares (survey/results/apple-m1-pro-coverage-gaps.md).
SPARSE_SHARE = 0.05
DENSE_SHARE = 0.5


def _scalar_sets():
    """Return one scalar set per row width, named `width_<w>`, ascending by width."""
    widths = sorted(CHECKED_WIDTHS + TIMED_WIDTHS)
    return [ScalarSet(f"width_{w}", {"width": w}, timed=w in TIMED_WIDTHS) for w in widths]


def oracle(m, width):
    """The (row, column) pair of every True element of m, in C order, flattened.

    Args:
        m: The boolean elements, row after row.
        width: Elements per row, at least 1.

    Returns:
        An int64 array of twice as many elements as m has True ones: row,
        column, row, column, and so on. For whole rows it equals
        `np.argwhere(m.reshape(-1, width)).ravel()`.
    """
    flat = np.flatnonzero(m).astype(np.int64)
    return np.stack((flat // width, flat % width), axis=1).ravel()


def numpy_call(m, width):
    """What a user writes: `np.nonzero` on the whole rows of m as a matrix.

    The oracle builds the pairs from flat indices, which costs NumPy more than
    the call it stands for, so the NumPy column times this instead.

    Args:
        m: The boolean elements, row after row.
        width: Elements per row, at least 1.

    Returns:
        NumPy's tuple of row indices and column indices. A partial last row
        is left out; at the sizes timed it is at most one row of many.
    """
    rows = len(m) // width
    return np.nonzero(m[:rows * width].reshape(rows, width))


def _filled(share):
    """Return a timing distribution in which `share` of the elements are True.

    Args:
        share: Probability that an element is True, from 0 to 1.

    Returns:
        A function of a random generator and an element count that builds the
        input of one timed call.
    """
    def make(rng, n):
        """Build one random matrix, as its n elements in C order."""
        return {"m": rng.random(n) < share}
    return make


KERNEL = Kernel(
    name="nonzero_2d_bool",
    returns="usize",  # number of int64 written: two per True element
    args=[
        In("m", "bool"),
        # Scratch, as nonzero_bool's output: only the returned count is compared.
        Out("out", "i64", capacity=Times(PAIR), used=USED_RETURN, scratch=True),
        Size("n"),
        Scalar("width", "usize", DEFAULT_WIDTH),
    ],
    oracle=oracle,
    numpy=numpy_call,
    scalar_sets=_scalar_sets(),
    distributions={"sparse": _filled(SPARSE_SHARE), "dense": _filled(DENSE_SHARE)},
)
