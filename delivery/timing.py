"""Times NumPy calls from Python, with and without hotkernels installed.

The judge times a kernel as a native function. This times what a user gets:
the Python statement, through NumPy's call path and the binding, to a Python
result. Two kinds of call are timed. A call a kernel takes shows the speedup
that is delivered. A call that reaches a hook and is left to NumPy shows what
the hook costs everyone else.

Timings are only meaningful on native hardware and an idle machine. The load
average is recorded with every run.
"""
from __future__ import annotations

import importlib
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import hotkernels
from hotkernels.environment import BINDING_MODULE
from hotkernels.hooks import ON_ARM64

from judge import cores
from judge.cli import machine

from .check import operation_kernels

ROOT = Path(__file__).resolve().parent.parent

# Element counts timed: where the cost of the call dominates, where the data
# fits the first-level cache, and where it fits no cache.
SIZES = (16, 4096, 1_048_576)
# Alternating batches per measurement. The minimum of each side is reported.
ROUNDS = 9
# Time one batch aims for, in nanoseconds, and the limits on calls per batch.
TARGET_BATCH_NS = 10_000_000
# Untimed calls made after each switch between plain NumPy and hotkernels, as a
# share of a batch. install() runs its probes for about a millisecond, and the
# calls after any such interruption (a short sleep does the same) run slow for
# about 3 ms while the core and its caches recover: np.median at 4,096 float64
# elements lost about 0.4 ms there, which made a 10 ms batch with hotkernels
# installed 9% slower than NumPy's for a call no hook takes. Half a batch is
# past that on both sides.
WARMUP_SHARE = 0.5
MIN_CALLS = 3
MAX_CALLS = 200_000
# Seed of all timing data.
SEED = 7
# Columns of the two-dimensional arrays timed. Every timed size is a multiple of it.
TIMED_COLUMNS = 4
# Columns of the timed boolean matrices of any(axis=1): the width its kernel
# was timed for by the judge.
ANY_COLUMNS = 22
# Columns of the timed boolean matrices of any(axis=0): the width of the
# masks its kernel was asked for.
ANY_COLS_COLUMNS = 31
# Rows of the boolean matrices of the workload np.nonzero on a matrix was
# asked for: 30 to 50 rows of ANY_COLUMNS columns.
NONZERO_ROWS = 40
# Share of a mask that is True in the two mask distributions.
HALF = 0.5
FEW = 0.05
# Share of a condition that is True in the timed np.where with a scalar.
MOSTLY = 0.95
# Bounds, bin count and range used by the timed clip and histogram calls.
LO, HI = -0.5, 0.5
BINS = 50
RANGE = (-3.0, 3.0)
# Largest value of the "narrow" isin data, and its eight test values.
ISIN_SPAN = 1000
ISIN_TEST = [1, 50, 999, 0, 7, 500, 501, 2]
# Number of distinct codes of the timed int16 and int64 bincount arrays: the
# counter count their kernels were timed for.
BINCOUNT_CODES = 64
# Distinct values of the timed np.unique arrays: binary labels and ten
# classes, the labels of the application it was written for.
UNIQUE_LABELS = (2, 10)
# Range of the np.unique array that is left to NumPy after the binding's scan:
# wider than any array timed, so the range test fails at every size.
UNIQUE_WIDE = 1 << 40
# Length of the runs of the timed np.fmax.reduceat, and the share of its
# elements that are NaN.
REDUCEAT_RUN = 16
REDUCEAT_NAN = 0.05
# Bounds of the timed np.clip on int64: Python ints inside the range of the
# small integers it clips, so that elements on both sides change.
INT_LO, INT_HI = 2, 7
# Largest element count each loop built into the binding for short arrays
# takes: its limit in the binding (CLIP_I64_MAX_N in hook_function.c,
# ROUND_F64_MAX_N in hook_method.c). The median loop takes arrays of every
# size timed.
CLIP_I64_MAX_N = 4096
ROUND_F64_MAX_N = 4096
# Decimals of the timed np.round: the commonest in the program the second
# shortlist came from.
DECIMALS = 2
# Most rows of the timed matrices accumulated along their first axis: about a
# year of trading days, as in empyrical's rolling windows. A matrix of n
# elements has min(ROWS_WINDOW, isqrt(n)) rows: 4 x 4, 64 x 64 and 256 x 4,096.
ROWS_WINDOW = 256
# Standard deviation of the factors of the timed cumulative product, which are
# near 1 as daily returns plus one are.
ROWS_SPREAD = 0.01

# Element count from which the argsort kernels take a stable argsort: the
# binding's limit (ARGSORT_MIN_N in hook_method.c).
ARGSORT_MIN_N = 256
# The share of a sorted array that the "nearly sorted" timing shuffles at its
# end: few enough that the binding leaves the call to NumPy's timsort.
ARGSORT_SHUFFLED_SHARE = 100
# Length of the sorted array in the search statements that hold it fixed while
# the number of needles grows: the shape of a lookup in a small table.
SMALL_TABLE = 1000


@dataclass(frozen=True)
class Timed:
    """One statement to time."""

    # Operation the statement belongs to.
    operation: str
    # The statement, as a user writes it.
    statement: str
    # Element type of the data, and anything else that defines the input.
    data: str
    # Builds the arguments for an element count.
    make: Callable[[int], tuple]
    # Runs the statement on the built arguments.
    call: Callable[..., object]
    # Kernel that must take the call, or None for a call left to NumPy.
    kernel: str | None
    # Whether the statement needs install(inexact=True).
    inexact: bool = False
    # Element count from which the kernel takes the call. Below it the
    # statement is measured as one left to NumPy.
    min_n: int = 0
    # Largest element count at which the kernel takes the call, or None for
    # no limit. The loops built into the binding for short arrays have one;
    # above it the statement is measured as one left to NumPy.
    max_n: int | None = None
    # Element counts to time the statement at, when they are not SIZES: for a
    # shape that a workload has and the usual sizes do not.
    sizes: tuple[int, ...] | None = None

    def kernel_at(self, n: int) -> str | None:
        """Return the kernel that takes the call at `n` elements, or None."""
        below_limit = self.max_n is None or n <= self.max_n
        return self.kernel if n >= self.min_n and below_limit else None


@dataclass(frozen=True)
class Measurement:
    """One statement at one size."""

    # Operation the statement belongs to.
    operation: str
    # The statement, as a user writes it.
    statement: str
    # Element type of the data, and anything else that defines the input.
    data: str
    # Kernel that took the call, or None for a call left to NumPy.
    kernel: str | None
    # Element count.
    n: int
    # Best time per call on plain NumPy, in nanoseconds.
    numpy_ns: float
    # Best time per call with hotkernels installed, in nanoseconds.
    hotkernels_ns: float

    @property
    def speedup(self) -> float:
        """Return how many times faster the call is with hotkernels."""
        return self.numpy_ns / self.hotkernels_ns

    @property
    def extra_ns(self) -> float:
        """Return what hotkernels adds to the call, in nanoseconds."""
        return self.hotkernels_ns - self.numpy_ns


def _rng(n: int) -> np.random.Generator:
    """Return the generator for timing data of one size."""
    return np.random.default_rng([SEED, n])


def _floats(dtype: type, n: int) -> tuple:
    """Build normally distributed floats."""
    return (_rng(n).standard_normal(n).astype(dtype),)


def _ints(dtype: type, n: int) -> tuple:
    """Build small integers, about a tenth of them zero."""
    return (_rng(n).integers(0, 10, n).astype(dtype),)


def _unsigned(n: int) -> tuple:
    """Build uint64 values drawn from the whole range of the type, as hashes are."""
    return (_rng(n).integers(0, np.iinfo(np.uint64).max, n, dtype=np.uint64, endpoint=True),)


def _sorted_floats(n: int) -> tuple:
    """Build normally distributed float64 values in ascending order."""
    return (np.sort(_rng(n).standard_normal(n)),)


def _nearly_sorted_floats(n: int) -> tuple:
    """Build sorted float64 values whose last one in ARGSORT_SHUFFLED_SHARE is in random order."""
    (x,) = _sorted_floats(n)
    tail = max(n // ARGSORT_SHUFFLED_SHARE, 2)
    x[n - tail:] = _rng(n + 1).standard_normal(tail)
    return (x,)


def _stable_argsort(x: np.ndarray) -> np.ndarray:
    """Call `np.argsort(x, kind='stable')`."""
    return np.argsort(x, kind="stable")


def _spread_ints(n: int) -> tuple:
    """Build int64 values drawn from the whole range of the type, so that a new maximum is rare after the start."""
    info = np.iinfo(np.int64)
    return (_rng(n).integers(info.min, info.max, n, dtype=np.int64, endpoint=True),)


def _runs(n: int, nan: float) -> tuple:
    """Build normally distributed float64 values and the starts of runs of REDUCEAT_RUN elements.

    Args:
        n: Element count.
        nan: Share of the elements that are NaN, all of one bit pattern.

    Returns:
        (values, starts), the starts an int64 array beginning at 0.
    """
    rng = _rng(n)
    x = rng.standard_normal(n)
    x[rng.random(n) < nan] = np.nan
    return x, np.arange(0, n, REDUCEAT_RUN, dtype=np.int64)


def _codes(dtype: type, n: int) -> tuple:
    """Build codes uniform in [0, BINCOUNT_CODES) of an integer type.

    Args:
        dtype: The element type, such as np.int16.
        n: Element count.

    Returns:
        (codes,).
    """
    return (_rng(n).integers(0, BINCOUNT_CODES, n).astype(dtype),)


def _labels(classes: int, n: int) -> tuple:
    """Build int64 labels uniform in [0, classes).

    Args:
        classes: Number of distinct values.
        n: Element count.

    Returns:
        (labels,).
    """
    return (_rng(n).integers(0, classes, n),)


def _bytes(n: int) -> tuple:
    """Build uniformly random bytes."""
    return (_rng(n).integers(0, 256, n, dtype=np.uint8),)


def _masked(dtype: type, share: float, n: int) -> tuple:
    """Build data and a mask that selects `share` of it."""
    rng = _rng(n)
    return (rng.standard_normal(n).astype(dtype), rng.random(n) < share)


def _masked_rows(dtype: type, share: float, n: int) -> tuple:
    """Build two-dimensional data of TIMED_COLUMNS columns and a mask of its shape that selects `share` of it."""
    x, mask = _masked(dtype, share, n)
    return (x.reshape(-1, TIMED_COLUMNS), mask.reshape(-1, TIMED_COLUMNS))


def _selected(dtype: type, share: float, n: int) -> tuple:
    """Build a mask that is True for `share` of its elements and two arrays of floats to select from."""
    rng = _rng(n)
    return (rng.random(n) < share, rng.standard_normal(n).astype(dtype), rng.standard_normal(n).astype(dtype))


def _fortran_matrix(share: float, n: int) -> tuple:
    """Build the matrix of `_matrix` in Fortran order, which the binding leaves to NumPy.

    Args:
        share: Probability that an element is True, from 0 to 1.
        n: Element count asked for, as for `_matrix`, but at least two
            rows: a matrix of one row is in C order as well.

    Returns:
        (matrix,).
    """
    (m,) = _matrix(share, max(n, 2 * ANY_COLUMNS))
    return (np.asfortranarray(m),)


def _mask(share: float, n: int) -> tuple:
    """Build a mask that is True for `share` of its elements."""
    return (_rng(n).random(n) < share,)


def _matrix(share: float, n: int, columns: int = ANY_COLUMNS) -> tuple:
    """Build a boolean matrix of `columns` columns that is True for `share` of its elements.

    Args:
        share: Probability that an element is True, from 0 to 1.
        n: Element count asked for. The matrix has `n // columns` rows, and
            one when that is none, so it holds about `n` elements, and
            `columns` at the smallest size.
        columns: Elements per row, at least 1.

    Returns:
        (matrix,), C-ordered.
    """
    rows = max(n // columns, 1)
    return (_rng(n).random((rows, columns)) < share,)


def _window(n: int, center: float = 0.0, spread: float = 1.0) -> tuple:
    """Build a C-ordered float64 matrix of about `n` normally distributed elements, for accumulations down its columns.

    Args:
        n: Element count asked for; a perfect square at or below 65,536, and
            a multiple of ROWS_WINDOW above it.
        center: Mean of the elements.
        spread: Standard deviation of the elements.

    Returns:
        (matrix,), of min(ROWS_WINDOW, isqrt(n)) rows.
    """
    rows = min(ROWS_WINDOW, math.isqrt(n))
    return (center + spread * _rng(n).standard_normal((rows, n // rows)),)


def _narrow_ints(n: int) -> tuple:
    """Build int64 values close together, where isin's test values match often."""
    return (_rng(n).integers(0, ISIN_SPAN, n),)


def _search(table: int | None, n: int) -> tuple:
    """Build a sorted array and `n` needles for `np.searchsorted`.

    Args:
        table: Length of the sorted array, or None for as many elements as needles.
        n: Number of needles.

    Returns:
        (sorted array, needles), both float64.
    """
    rng = _rng(n)
    return (np.sort(rng.standard_normal(table or n)), rng.standard_normal(n))


def _int_search(table: int | None, n: int) -> tuple:
    """Build a sorted int64 array and `n` int64 needles for `np.searchsorted`.

    Args:
        table: Length of the sorted array, or None for as many elements as needles.
        n: Number of needles.

    Returns:
        (sorted array, needles), both int64 and drawn from the whole range of the type.
    """
    info = np.iinfo(np.int64)
    rng = _rng(n)
    draw = partial(rng.integers, info.min, info.max, dtype=np.int64, endpoint=True)
    return (np.sort(draw(table or n)), draw(n))


def _int_search_one(n: int) -> tuple:
    """Build a sorted int64 array of `n` elements and one needle, an `np.int64`.

    The call of a program that looks up one value at a time, as
    `np.searchsorted(frames, f)` in a loop does.

    Args:
        n: Length of the sorted array.

    Returns:
        (sorted int64 array, np.int64 needle).
    """
    table, needles = _int_search(n, 1)
    return (table, needles[0])


def _int_search_int32_needles(n: int) -> tuple:
    """Build a sorted int64 array and `n` int32 needles, which no search kernel takes.

    Args:
        n: Number of elements, and of needles.

    Returns:
        (sorted int64 array, int32 needles).
    """
    info = np.iinfo(np.int32)
    table, _ = _int_search(None, n)
    return (table, _rng(n).integers(info.min, info.max, n, dtype=np.int32, endpoint=True))


def _search_float32_needles(n: int) -> tuple:
    """Build a sorted float64 array and `n` float32 needles, which no search kernel takes.

    Args:
        n: Number of elements, and of needles.

    Returns:
        (sorted float64 array, float32 needles).
    """
    table, needles = _search(None, n)
    return (table, needles.astype(np.float32))


def _search_right(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Call `np.searchsorted(a, v, side='right')`.

    Args:
        a: The sorted array.
        v: The needles.

    Returns:
        The index after the last element not greater than each needle.
    """
    return np.searchsorted(a, v, side="right")


def _digitize(bins: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Call `np.digitize(x, bins)`, which takes the values first and the bins second.

    Args:
        bins: The increasing bin edges.
        x: The values to place.

    Returns:
        The bin index of each value.
    """
    return np.digitize(x, bins)


def _interp(points: int | None, n: int) -> tuple:
    """Build `n` queries and the sample points for `np.interp`.

    Args:
        points: Number of sample points, or None for as many as queries.
        n: Number of queries.

    Returns:
        (x, xp, fp), all float64, with xp sorted.
    """
    rng = _rng(n)
    count = points or n
    return (rng.standard_normal(n), np.sort(rng.standard_normal(count)), rng.standard_normal(count))


def _assign(value: float, x: np.ndarray, mask: np.ndarray) -> None:
    """Run `x[mask] = value`. Running it twice changes nothing more."""
    x[mask] = value


def _compress(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return `np.compress(mask, x)`, which takes the mask first."""
    return np.compress(mask, x)


def _extract(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return `np.extract(mask, x)`, which takes the mask first."""
    return np.extract(mask, x)


def _putmask(value: float, x: np.ndarray, mask: np.ndarray) -> None:
    """Run `np.putmask(x, mask, value)`. Running it twice changes nothing more."""
    np.putmask(x, mask, value)


def _copyto(value: float, x: np.ndarray, mask: np.ndarray) -> None:
    """Run `np.copyto(x, value, where=mask)`. Running it twice changes nothing more."""
    np.copyto(x, value, where=mask)


def _clip_in_place(x: np.ndarray) -> np.ndarray:
    """Run `np.clip(x, -0.5, 0.5, out=x)`. Running it twice changes nothing more.

    Args:
        x: The array, clipped in place.

    Returns:
        `x`, as NumPy returns it.
    """
    return np.clip(x, LO, HI, out=x)


def _clip_method_in_place(x: np.ndarray) -> np.ndarray:
    """Run `x.clip(-0.5, 0.5, out=x)`. Running it twice changes nothing more.

    Args:
        x: The array, clipped in place.

    Returns:
        `x`, as NumPy returns it.
    """
    return x.clip(LO, HI, out=x)


def _two_floats(n: int) -> tuple:
    """Build two arrays of normally distributed float64 values."""
    rng = _rng(n)
    return rng.standard_normal(n), rng.standard_normal(n)


def _subtract_in_place(w: np.ndarray, b: np.ndarray) -> None:
    """Run `w -= b`."""
    w -= b


def _add_in_place(w: np.ndarray, b: np.ndarray) -> None:
    """Run `w += b`."""
    w += b


def _subtract_scalar(w: np.ndarray) -> None:
    """Run `w -= 0.5`, whose scalar steps by zero."""
    w -= HI


# The loops of the binding that take float64 adds and subtracts in place,
# where they are built: ARM64 only. Elsewhere the statements are timed as
# calls left to NumPy, and so they are below INPLACE_MIN_N elements.
INPLACE_MIN_N = 131072
# The float sum kernels take their calls on ARM64 only, until measured on
# x86-64 (docs/coverage-gaps-arm64.md).
SUM_F32 = "sum_f32" if ON_ARM64 else None
SUM_F64 = "sum_f64" if ON_ARM64 else None
# The kernel of np.nonzero on a boolean matrix, which is installed on ARM64
# only; elsewhere those statements are timed as calls left to NumPy.
NONZERO_2D = "nonzero_2d_bool" if ON_ARM64 else None
SUBTRACT_IN_PLACE = "subtract_inplace_f64" if ON_ARM64 else None
ADD_IN_PLACE = "add_inplace_f64" if ON_ARM64 else None


def _set_element(x: np.ndarray) -> None:
    """Run `x[3] = 1.0`."""
    x[3] = 1.0


# Statements a kernel takes: at least one per installed kernel.
TAKEN: tuple[Timed, ...] = (
    Timed("argmax", "np.argmax(x)", "float32", partial(_floats, np.float32), np.argmax, "argmax_f32"),
    Timed("argmax", "x.argmax()", "float32", partial(_floats, np.float32), lambda x: x.argmax(), "argmax_f32"),
    Timed("argmax", "np.argmax(x)", "float64", partial(_floats, np.float64), np.argmax, "argmax_f64"),
    Timed("argmin", "np.argmin(x)", "float32", partial(_floats, np.float32), np.argmin, "argmin_f32"),
    Timed("argmin", "np.argmin(x)", "float64", partial(_floats, np.float64), np.argmin, "argmin_f64"),
    Timed("clip", "np.clip(x, -0.5, 0.5)", "float32", partial(_floats, np.float32), lambda x: np.clip(x, LO, HI), "clip_f32"),
    Timed("clip", "np.clip(x, -0.5, 0.5)", "float64", partial(_floats, np.float64), lambda x: np.clip(x, LO, HI), "clip_f64"),
    Timed("clip", "np.clip(x, 0, 1)", "float64", partial(_floats, np.float64), lambda x: np.clip(x, 0, 1), "clip_f64"),
    Timed("clip", "x.clip(-0.5, 0.5)", "float32", partial(_floats, np.float32), lambda x: x.clip(LO, HI), "clip_f32"),
    Timed("clip", "np.clip(x, -0.5, 0.5, out=x)", "float64", partial(_floats, np.float64), _clip_in_place, "clip_f64"),
    Timed("clip", "x.clip(-0.5, 0.5, out=x)", "float32", partial(_floats, np.float32), _clip_method_in_place, "clip_f32"),
    Timed("clip", "np.clip(x, 2, 7)", "int64", partial(_ints, np.int64), lambda x: np.clip(x, INT_LO, INT_HI),
          "clip_i64", max_n=CLIP_I64_MAX_N),
    Timed("clip", "x.clip(2, 7)", "int64", partial(_ints, np.int64), lambda x: x.clip(INT_LO, INT_HI), "clip_i64",
          max_n=CLIP_I64_MAX_N),
    Timed("round", "np.round(x, 2)", "float64", partial(_floats, np.float64), lambda x: np.round(x, DECIMALS),
          "round_f64", max_n=ROUND_F64_MAX_N),
    Timed("round", "x.round(2)", "float64", partial(_floats, np.float64), lambda x: x.round(DECIMALS), "round_f64",
          max_n=ROUND_F64_MAX_N),
    Timed("round", "np.round(x)", "float64", partial(_floats, np.float64), np.round, "round_f64",
          max_n=ROUND_F64_MAX_N),
    Timed("inplace", "w -= b", "float64", _two_floats, _subtract_in_place, SUBTRACT_IN_PLACE,
          min_n=INPLACE_MIN_N),
    Timed("inplace", "w += b", "float64", _two_floats, _add_in_place, ADD_IN_PLACE, min_n=INPLACE_MIN_N),
    Timed("median", "np.median(x)", "float64", partial(_floats, np.float64), np.median, "median_f64"),
    Timed("compress", "x[mask]", "float32, half selected", partial(_masked, np.float32, HALF), lambda x, m: x[m], "compress_f32"),
    Timed("compress", "x[mask]", "float64, half selected", partial(_masked, np.float64, HALF), lambda x, m: x[m], "compress_f64"),
    Timed("compress", "x[mask]", "float64, few selected", partial(_masked, np.float64, FEW), lambda x, m: x[m], "compress_f64"),
    Timed("compress", "x[mask]", "int64, half selected", partial(_masked, np.int64, HALF), lambda x, m: x[m], "compress_f64"),
    Timed("maskfill", "x[mask] = 0.0", "float64, half selected", partial(_masked, np.float64, HALF), partial(_assign, 0.0), "maskfill_f64"),
    Timed("compress", "np.compress(mask, x)", "float64, half selected", partial(_masked, np.float64, HALF), _compress, "compress_f64"),
    Timed("compress", "np.extract(mask, x)", "float64, half selected", partial(_masked, np.float64, HALF), _extract, "compress_f64"),
    Timed("maskfill", "x[mask] = 0.0", "float64, few selected", partial(_masked, np.float64, FEW), partial(_assign, 0.0), "maskfill_f64"),
    Timed("maskfill", "x[mask] = 0.0", "float64, two-dimensional, half selected", partial(_masked_rows, np.float64, HALF), partial(_assign, 0.0), "maskfill_f64"),
    Timed("compress", "x[mask]", "float64, two-dimensional, half selected", partial(_masked_rows, np.float64, HALF), lambda x, m: x[m], "compress_f64"),
    Timed("maskfill", "np.putmask(x, mask, 0.0)", "float64, half selected", partial(_masked, np.float64, HALF), partial(_putmask, 0.0), "maskfill_f64"),
    Timed("maskfill", "np.copyto(x, 0.0, where=mask)", "float64, half selected", partial(_masked, np.float64, HALF), partial(_copyto, 0.0), "maskfill_f64"),
    Timed("nonzero", "np.nonzero(mask)", "bool, half selected", partial(_mask, HALF), np.nonzero, "nonzero_bool"),
    Timed("nonzero", "np.nonzero(mask)", "bool, few selected", partial(_mask, FEW), np.nonzero, "nonzero_bool"),
    Timed("nonzero", "np.where(mask)", "bool, half selected", partial(_mask, HALF), np.where, "nonzero_bool"),
    *(Timed("nonzero", "np.nonzero(m)", f"bool, {ANY_COLUMNS} columns, {name}", partial(_matrix, share), np.nonzero,
            NONZERO_2D, sizes=sizes)
      for sizes in ((NONZERO_ROWS * ANY_COLUMNS,), None)
      for name, share in (("few True", FEW), ("half True", HALF))),
    Timed("nonzero", "m.nonzero()", f"bool, {ANY_COLUMNS} columns, few True", partial(_matrix, FEW),
          lambda m: m.nonzero(), NONZERO_2D),
    Timed("nonzero", "np.where(m)", f"bool, {ANY_COLUMNS} columns, few True", partial(_matrix, FEW), np.where,
          NONZERO_2D),
    Timed("nonzero", "np.argwhere(m)", f"bool, {ANY_COLUMNS} columns, few True", partial(_matrix, FEW), np.argwhere,
          NONZERO_2D),
    Timed("where", "np.where(mask, a, b)", "float64, half true", partial(_selected, np.float64, HALF), np.where, "where_f64"),
    Timed("where", "np.where(mask, a, np.nan)", "float64, mostly true", partial(_selected, np.float64, MOSTLY),
          lambda m, a, b: np.where(m, a, np.nan), "where_f64"),
    Timed("cumsum", "np.cumsum(x)", "int64", partial(_ints, np.int64), np.cumsum, "cumsum_i64"),
    Timed("accumulate", "np.maximum.accumulate(x)", "int64", _spread_ints, np.maximum.accumulate, "cummax_i64"),
    Timed("accumulate", "np.fmax.accumulate(x)", "float64", partial(_floats, np.float64), np.fmax.accumulate,
          "cumfmax_f64"),
    Timed("accumulate", "np.fmax.accumulate(a, axis=0)", "float64 matrix", _window,
          lambda a: np.fmax.accumulate(a, axis=0), "cumfmax_rows_f64"),
    Timed("accumulate", "np.maximum.accumulate(a, axis=0)", "float64 matrix", _window,
          lambda a: np.maximum.accumulate(a, axis=0), "cummax_rows_f64"),
    Timed("accumulate", "a.cumprod(axis=0)", "float64 matrix near 1", partial(_window, center=1.0, spread=ROWS_SPREAD),
          lambda a: a.cumprod(axis=0), "cumprod_rows_f64"),
    Timed("cumsum", "a.cumsum(axis=0)", "float64 matrix", _window, lambda a: a.cumsum(axis=0), "cumsum_rows_f64"),
    Timed("reduceat", "np.fmax.reduceat(x, starts)", f"float64, runs of {REDUCEAT_RUN}, {REDUCEAT_NAN:.0%} NaN",
          partial(_runs, nan=REDUCEAT_NAN), np.fmax.reduceat, "fmax_reduce_f64"),
    Timed("cumsum", "np.cumsum(mask)", "bool, half selected", partial(_mask, HALF), np.cumsum, "cumsum_bool"),
    Timed("any", "m.any(axis=1)", f"bool, {ANY_COLUMNS} columns, few True", partial(_matrix, FEW),
          lambda m: m.any(axis=1), "any_rows_bool"),
    Timed("any", "np.any(m, axis=1)", f"bool, {ANY_COLUMNS} columns, few True", partial(_matrix, FEW),
          lambda m: np.any(m, axis=1), "any_rows_bool"),
    Timed("any", "m.any(axis=0)", f"bool, {ANY_COLS_COLUMNS} columns, few True",
          partial(_matrix, FEW, columns=ANY_COLS_COLUMNS), lambda m: m.any(axis=0), "any_cols_bool"),
    Timed("any", "np.any(m, axis=0)", f"bool, {ANY_COLS_COLUMNS} columns, few True",
          partial(_matrix, FEW, columns=ANY_COLS_COLUMNS), lambda m: np.any(m, axis=0), "any_cols_bool"),
    Timed("cumsum", "np.cumsum(x)", "float32", partial(_floats, np.float32), np.cumsum, "cumsum_f32", inexact=True),
    Timed("cumsum", "np.cumsum(x)", "float64", partial(_floats, np.float64), np.cumsum, "cumsum_f64", inexact=True),
    Timed("astype", "x.astype(np.float32)", "uint8", _bytes, lambda x: x.astype(np.float32), "cast_u8_f32"),
    Timed("sum", "np.sum(x)", "uint8", _bytes, np.sum, "sum_u8"),
    Timed("sum", "x.sum()", "uint8", _bytes, lambda x: x.sum(), "sum_u8"),
    Timed("sum", "np.sum(mask)", "bool, half selected", partial(_mask, HALF), np.sum, "sum_u8"),
    Timed("sum", "mask.sum()", "bool, half selected", partial(_mask, HALF), lambda m: m.sum(), "sum_u8"),
    Timed("sum", "np.sum(x)", "float32", partial(_floats, np.float32), np.sum, SUM_F32),
    Timed("sum", "x.sum()", "float32", partial(_floats, np.float32), lambda x: x.sum(), SUM_F32),
    Timed("sum", "np.sum(x)", "float64", partial(_floats, np.float64), np.sum, SUM_F64),
    Timed("sum", "x.sum()", "float64", partial(_floats, np.float64), lambda x: x.sum(), SUM_F64),
    Timed("count_nonzero", "np.count_nonzero(x)", "int64", partial(_ints, np.int64), np.count_nonzero, "count_nonzero_i64"),
    Timed("count_nonzero", "np.count_nonzero(x)", "float64", partial(_ints, np.float64), np.count_nonzero, "count_nonzero_f64"),
    Timed("histogram", "np.histogram(x, 50, (-3.0, 3.0))", "float64", partial(_floats, np.float64), lambda x: np.histogram(x, BINS, RANGE), "histogram_f64"),
    Timed("bincount", "np.bincount(x)", "uint8", _bytes, np.bincount, "bincount_u8"),
    Timed("bincount", "np.bincount(x)", f"int16, codes 0 to {BINCOUNT_CODES - 1}", partial(_codes, np.int16),
          np.bincount, "bincount_i16"),
    Timed("bincount", "np.bincount(x)", f"int64, codes 0 to {BINCOUNT_CODES - 1}", partial(_codes, np.int64),
          np.bincount, "bincount_i64"),
    *(Timed("unique", "np.unique(x)", f"int64, {classes} values", partial(_labels, classes), np.unique, "unique_i64")
      for classes in UNIQUE_LABELS),
    Timed("isin", "np.isin(x, eight_values)", "int64, values close together", _narrow_ints, lambda x: np.isin(x, ISIN_TEST), "isin_i64"),
    Timed("searchsorted", "np.searchsorted(a, v)", "float64, as many needles as elements", partial(_search, None), np.searchsorted, "searchsorted_f64"),
    Timed("searchsorted", "np.searchsorted(a, v)", "float64, 1,000 elements", partial(_search, SMALL_TABLE), np.searchsorted, "searchsorted_f64"),
    Timed("searchsorted", "np.searchsorted(a, v, side='right')", "float64, as many needles as elements", partial(_search, None), _search_right, "searchsorted_right_f64"),
    Timed("searchsorted", "np.searchsorted(a, v, side='right')", "float64, 1,000 elements", partial(_search, SMALL_TABLE), _search_right, "searchsorted_right_f64"),
    Timed("searchsorted", "np.digitize(x, bins)", "float64, 1,000 bin edges", partial(_search, SMALL_TABLE), _digitize, "searchsorted_right_f64"),
    Timed("searchsorted", "np.searchsorted(a, f)", "int64, one np.int64 needle", _int_search_one, np.searchsorted, "searchsorted_i64"),
    Timed("searchsorted", "np.searchsorted(a, f, side='right')", "int64, one np.int64 needle", _int_search_one, _search_right, "searchsorted_right_i64"),
    Timed("searchsorted", "np.searchsorted(a, v)", "int64, as many needles as elements", partial(_int_search, None), np.searchsorted, "searchsorted_i64"),
    Timed("searchsorted", "np.searchsorted(a, v)", "int64, 1,000 elements", partial(_int_search, SMALL_TABLE), np.searchsorted, "searchsorted_i64"),
    Timed("searchsorted", "np.searchsorted(a, v, side='right')", "int64, as many needles as elements", partial(_int_search, None), _search_right, "searchsorted_right_i64"),
    Timed("searchsorted", "np.searchsorted(a, v, side='right')", "int64, 1,000 elements", partial(_int_search, SMALL_TABLE), _search_right, "searchsorted_right_i64"),
    Timed("interp", "np.interp(x, xp, fp)", "float64, as many queries as points", partial(_interp, None), np.interp, "interp_f64"),
    Timed("interp", "np.interp(x, xp, fp)", "float64, 1,000 points", partial(_interp, SMALL_TABLE), np.interp, "interp_f64"),
    Timed("argsort", "np.argsort(x, kind='stable')", "float64", partial(_floats, np.float64), _stable_argsort,
          "argsort_f64", min_n=ARGSORT_MIN_N),
    Timed("argsort", "x.argsort(kind='mergesort')", "float64", partial(_floats, np.float64),
          lambda x: x.argsort(kind="mergesort"), "argsort_f64", min_n=ARGSORT_MIN_N),
    Timed("argsort", "np.argsort(x, kind='stable')", "int64, the whole range", _spread_ints, _stable_argsort,
          "argsort_i64", min_n=ARGSORT_MIN_N),
    Timed("argsort", "np.argsort(x, kind='stable')", "int64, ten values", partial(_ints, np.int64), _stable_argsort,
          "argsort_i64", min_n=ARGSORT_MIN_N),
    Timed("argsort", "np.argsort(x, kind='stable')", "uint64, the whole range", _unsigned, _stable_argsort,
          "argsort_u64", min_n=ARGSORT_MIN_N),
    Timed("argsort", "np.argsort(x, kind='stable')", "float64, already sorted", _sorted_floats, _stable_argsort,
          "argsort_f64", min_n=ARGSORT_MIN_N),
    # What scikit-learn's ranking metrics call on their scores.
    Timed("argsort", "np.argsort(np.flip(x), kind='stable')", "float64, a reversed view",
          partial(_floats, np.float64), lambda x: np.argsort(np.flip(x), kind="stable"), "argsort_f64",
          min_n=ARGSORT_MIN_N),
)

# Statements that reach a hook and are left to NumPy: what a hook costs a
# call it does not serve.
LEFT: tuple[Timed, ...] = (
    Timed("argmax", "np.argmax(x)", "int64", partial(_ints, np.int64), np.argmax, None),
    Timed("argmax", "x.argmax()", "int64", partial(_ints, np.int64), lambda x: x.argmax(), None),
    Timed("clip", "np.clip(x[::2], -0.5, 0.5)", "float64, strided", partial(_floats, np.float64), lambda x: np.clip(x[::2], LO, HI), None),
    Timed("clip", "np.clip(x, -0.5, None)", "float64", partial(_floats, np.float64), lambda x: np.clip(x, LO, None), None),
    Timed("round", "np.round(x, 2)", "float32", partial(_floats, np.float32), lambda x: np.round(x, DECIMALS), None),
    Timed("median", "np.median(x)", "float32", partial(_floats, np.float32), np.median, None),
    Timed("inplace", "a + b", "float64", _two_floats, np.add, None),
    Timed("inplace", "w -= 0.5", "float64", partial(_floats, np.float64), _subtract_scalar, None),
    Timed("inplace", "w -= b", "float32", lambda n: tuple(a.astype(np.float32) for a in _two_floats(n)),
          _subtract_in_place, None),
    Timed("astype", "x.astype(np.float64)", "uint8", _bytes, lambda x: x.astype(np.float64), None),
    Timed("sum", "np.sum(x)", "int64", partial(_ints, np.int64), np.sum, None),
    Timed("sum", "np.sum(x[::2])", "float32, strided", partial(_floats, np.float32), lambda x: np.sum(x[::2]), None),
    Timed("count_nonzero", "np.count_nonzero(x)", "int32", partial(_ints, np.int32), np.count_nonzero, None),
    Timed("cumsum", "np.cumsum(x)", "float64, exact kernels only", partial(_floats, np.float64), np.cumsum, None),
    Timed("accumulate", "np.add.accumulate(x)", "int64", partial(_ints, np.int64), np.add.accumulate, None),
    Timed("accumulate", "np.maximum.accumulate(x)", "float64", partial(_floats, np.float64), np.maximum.accumulate,
          None),
    Timed("any", "m.any()", "bool, one-dimensional", partial(_mask, HALF), lambda m: m.any(), None),
    Timed("any", "m.any(axis=0, keepdims=True)", f"bool, {ANY_COLUMNS} columns", partial(_matrix, FEW),
          lambda m: m.any(axis=0, keepdims=True), None),
    Timed("reduceat", "np.add.reduceat(x, starts)", f"float64, runs of {REDUCEAT_RUN}",
          partial(_runs, nan=0.0), np.add.reduceat, None),
    Timed("histogram", "np.histogram(x, 50, (-3.0, 3.0), density=False)", "float64", partial(_floats, np.float64), lambda x: np.histogram(x, BINS, RANGE, density=False), None),
    Timed("bincount", "np.bincount(x)", "int32", partial(_ints, np.int32), np.bincount, None),
    Timed("unique", "np.unique(x)", "int64, wide range", partial(_labels, UNIQUE_WIDE), np.unique, None),
    Timed("isin", "np.isin(x, nine_values)", "int64", _narrow_ints, lambda x: np.isin(x, ISIN_TEST + [3]), None),
    Timed("searchsorted", "np.searchsorted(a, v)", "float64 array, float32 needles", _search_float32_needles, np.searchsorted, None),
    Timed("searchsorted", "np.searchsorted(a, v)", "int64 array, int32 needles", _int_search_int32_needles, np.searchsorted, None),
    Timed("interp", "np.interp(x, xp, fp, left=0.0)", "float64", partial(_interp, None), lambda x, xp, fp: np.interp(x, xp, fp, left=0.0), None),
    Timed("nonzero", "x.nonzero()", "int64", partial(_ints, np.int64), lambda x: x.nonzero(), None),
    Timed("nonzero", "np.nonzero(m)", f"bool, {ANY_COLUMNS} columns, Fortran order", partial(_fortran_matrix, FEW),
          np.nonzero, None),
    Timed("compress", "x[3]", "float64", partial(_floats, np.float64), lambda x: x[3], None),
    Timed("compress", "x[1:5]", "float64", partial(_floats, np.float64), lambda x: x[1:5], None),
    Timed("compress", "x[mask]", "int32, half selected", partial(_masked, np.int32, HALF), lambda x, m: x[m], None),
    Timed("maskfill", "x[3] = 1.0", "float64", partial(_floats, np.float64), _set_element, None),
    Timed("maskfill", "x[mask] = 0.0", "float32, half selected", partial(_masked, np.float32, HALF), partial(_assign, 0.0), None),
    Timed("maskfill", "np.copyto(x, 0.0)", "float64", partial(_floats, np.float64), lambda x: np.copyto(x, 0.0), None),
    Timed("where", "np.where(mask, 1.0, 0.0)", "bool, half selected", partial(_mask, HALF), lambda m: np.where(m, 1.0, 0.0), None),
    Timed("where", "np.where(mask, a, b)", "float32, half true", partial(_selected, np.float32, HALF), np.where, None),
    Timed("argsort", "np.argsort(x)", "float64, the default kind", partial(_floats, np.float64), np.argsort, None),
    Timed("argsort", "np.argsort(x, kind='stable')", "float64, nearly sorted",
          _nearly_sorted_floats, _stable_argsort, None),
    Timed("argsort", "np.argsort(x, kind='stable')", "float32", partial(_floats, np.float32), _stable_argsort, None),
)


def _batch(call: Callable, arguments: tuple, calls: int) -> float:
    """Time `calls` calls and return nanoseconds per call."""
    clock = time.perf_counter_ns
    repeats = range(calls)
    start = clock()
    for _ in repeats:
        call(*arguments)
    return (clock() - start) / calls


def _calls_per_batch(call: Callable, arguments: tuple) -> int:
    """Choose how many calls make a batch of about TARGET_BATCH_NS."""
    call(*arguments)
    once = max(_batch(call, arguments, 1), 1.0)
    return int(min(max(TARGET_BATCH_NS / once, MIN_CALLS), MAX_CALLS))


def _kernel_ran(entry: Timed, arguments: tuple, kernel: str | None) -> bool:
    """Make one call with hotkernels installed and say whether the expected thing happened.

    Args:
        entry: The statement.
        arguments: Its arguments.
        kernel: The kernel expected to take the call, or None.

    Returns:
        For a statement with a kernel, whether that kernel took the call; for
        one left to NumPy, whether none of its operation's kernels did.
    """
    binding = importlib.import_module(BINDING_MODULE)
    before = binding.counters()[0]
    entry.call(*arguments)
    after = binding.counters()[0]
    if kernel:
        return after[kernel] > before[kernel]
    return all(after[k] == before[k] for k in operation_kernels(entry.operation))


def measure(entry: Timed, n: int) -> Measurement:
    """Time one statement at one size, alternating plain NumPy and hotkernels.

    Args:
        entry: The statement.
        n: Element count.

    Returns:
        The best time of each side over ROUNDS alternating batches.

    Raises:
        RuntimeError: If the call was not routed as the entry says, in which
            case the timing would describe something else.
    """
    arguments = entry.make(n)
    kernel = entry.kernel_at(n)
    hotkernels.uninstall()
    calls = _calls_per_batch(entry.call, arguments)
    # At least one untimed call, so the first timed call never follows a switch.
    warmup = max(int(calls * WARMUP_SHARE), 1)
    plain, installed = [], []
    for _ in range(ROUNDS):
        hotkernels.uninstall()
        _batch(entry.call, arguments, warmup)
        plain.append(_batch(entry.call, arguments, calls))
        hotkernels.install(inexact=entry.inexact)
        if not _kernel_ran(entry, arguments, kernel):
            raise RuntimeError(f"{entry.statement} ({entry.data}, n={n}) was not routed as expected")
        _batch(entry.call, arguments, warmup)
        installed.append(_batch(entry.call, arguments, calls))
    hotkernels.uninstall()
    return Measurement(entry.operation, entry.statement, entry.data, kernel, n, min(plain), min(installed))


def run(entries: Iterable[Timed], sizes: tuple[int, ...] = SIZES) -> list[Measurement]:
    """Time every entry at every size."""
    return [measure(entry, n) for entry in entries for n in (entry.sizes or sizes)]


def _output_of(command: list[str]) -> str:
    """Return a command's standard output, stripped, or an empty string if it fails."""
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_average() -> float:
    """Return the one-minute load average, to two decimals. Above 1.0 the machine is in use."""
    return round(os.getloadavg()[0], 2)


def describe_machine() -> dict:
    """Record what a run was measured on.

    Pins the calling thread first, as the judge does, where the system allows
    it: on a processor with performance and efficiency cores an unpinned run
    is not a result.

    Returns:
        Processor, operating system, NumPy and Python versions, git revision,
        the one-minute load average when the run started, and the CPU the
        timing thread is pinned to, or None where threads cannot be pinned
        (macOS). A load average well above zero means the timings were taken
        on a busy machine.
    """
    claim = cores.claim()
    return {
        "cpu": machine(),
        "pinned_cpu": claim.cpu if claim else None,
        "os": platform.platform(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "git": _output_of(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "load_average": load_average(),
    }


def as_rows(measurements: list[Measurement]) -> list[dict]:
    """Convert measurements to plain dictionaries for a JSON file."""
    return [asdict(m) for m in measurements]


def from_rows(rows: list[dict]) -> list[Measurement]:
    """Rebuild measurements from the dictionaries of a JSON file."""
    return [Measurement(**row) for row in rows]
