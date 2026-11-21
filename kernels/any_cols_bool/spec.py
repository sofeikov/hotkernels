"""Whether each column of a boolean matrix holds a True: `m.any(axis=0)` for a two-dimensional array.

`m` holds `n` elements: `n // cols` whole rows of `cols` elements in C order,
and possibly a partial row after them, which is ignored. The kernel writes
`cols` results, one per column, each exactly 0 or 1. When `cols` is larger than
`n` there are no rows and every result is False, which is what NumPy gives for
a matrix of no rows.

The kernel is defined for `cols >= 1`, and for any bytes in `m`: a result is
whether its column holds a byte that is not zero, which is what NumPy's
`any(axis=0)` computes for a boolean array whose bytes are not all 0 or 1. The
judge's boolean data holds only 0 and 1, so it does not check that; the cases
of `uv run delivery check` do, from Python.

The judge sweeps `n` for each width in SCALAR_SETS, so no rows, one row and a
partial last row are covered at every width. It times the default width and
the sets marked `timed`.
"""
from judge.spec import In, Kernel, Out, Scalar, ScalarSet, Size

# Row width in the default scalar set: the masks the kernel was asked for are
# 31 and 30 columns wide, and 31 is one short of a 32-byte vector.
DEFAULT_COLS = 31

# Further row widths that are checked and not timed: the narrowest, the widths
# either side of each vector and block size a kernel may work in (4, 8, 16 and
# 32 bytes, and 2, 4, 8 and 16 vectors of 32), widths of which a 32-byte vector
# holds a different number of whole rows, and one of several hundred that is
# no multiple of any of them.
CHECKED_COLS = (1, 2, 3, 4, 7, 9, 13, 15, 16, 17, 22, 30, 32, 33, 63, 65, 96, 127, 128, 129, 255, 257, 511, 512,
                513, 1000)
# Row widths that are timed as well: a narrow row, of which a vector holds
# several, one of two 32-byte vectors and one of eight.
TIMED_COLS = (8, 64, 256)

# Share of elements that are True in each timing distribution. With the sparse
# share about four columns in ten of a 16-row matrix hold a True; with the
# dense share every column of a matrix of a few rows does.
SPARSE_SHARE = 0.03
DENSE_SHARE = 0.5


def _scalar_sets():
    """Return one scalar set per row width, named `cols_<width>`, ascending by width."""
    widths = sorted(CHECKED_COLS + TIMED_COLS)
    return [ScalarSet(f"cols_{width}", {"cols": width}, timed=width in TIMED_COLS) for width in widths]


def oracle(m, cols):
    """Whether each column of the whole rows of `cols` elements of m holds a True.

    Args:
        m: The boolean elements, row after row.
        cols: Elements per row, at least 1.

    Returns:
        One boolean per column: an array of `cols` elements, all False when m
        is shorter than one row.
    """
    rows = len(m) // cols
    return m[:rows * cols].reshape(rows, cols).any(axis=0)


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
    name="any_cols_bool",
    returns="void",
    args=[
        In("m", "bool"),
        Out("out", "bool", capacity="cols"),
        Size("n"),
        Scalar("cols", "usize", DEFAULT_COLS),
    ],
    oracle=oracle,
    scalar_sets=_scalar_sets(),
    distributions={"sparse": _filled(SPARSE_SHARE), "dense": _filled(DENSE_SHARE)},
)
