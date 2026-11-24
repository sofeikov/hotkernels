"""What hotkernels installs: the kernels, the hooks that deliver them, and a
self-test for each hook.

A hook is one pointer inside NumPy that the extension module can point at the
binding. This module says which hooks exist, which kernels each one serves,
and how to prove after applying a hook that the kernel really is reached and
answers as NumPy does. The proof is a probe: a few small NumPy calls, made
once on NumPy with no hook applied and once with the hook. `install()` keeps a
hook only if every probe's answers are the same in every detail a caller
could see (type, element type, shape, flags and bytes, so the sign of a zero
too) and the kernel's call counter moved. A NumPy that routes a call
differently from the one the hooks were written for, or answers it
differently, then leaves hotkernels switched off for that call, with NumPy
untouched.
"""
from __future__ import annotations

import operator
import platform
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

# Kinds of hook, as shown to the user.
TYPE_SLOT = "type slot"
INNER_LOOP = "inner loop"
FUNCTION = "function"
METHOD = "method"
INDEXING = "indexing"
BOTTLENECK = "bottleneck"
UFUNC = "ufunc call"

# What report() shows as the implementation of the ufunc lane: NumPy's own
# inner loops, which the binding calls without NumPy's call path.
NUMPY_LOOPS = "NumPy"

# The module that defines bottleneck's functions. A bottleneck hook is applied
# only when it is in sys.modules; hotkernels never imports it.
BOTTLENECK_MODULE = "bottleneck.reduce"
# Shape of the matrix the bottleneck probes reduce along axis 0, and the seed
# of its values. With these, the square and the add of nanstd rounded once
# give another result than rounded twice in several columns, so a bottleneck
# build that rounds the other way fails the nanstd probe.
BOTTLENECK_PROBE_SHAPE = (40, 37)
BOTTLENECK_PROBE_SEED = 192

# Names platform.machine() gives an ARM64 processor: arm64 on macOS, aarch64 on Linux.
ARM64_MACHINES = ("arm64", "aarch64")
# Whether this is an ARM64 machine, where the binding has the in-place add and
# subtract loops. They are not built for other architectures.
ON_ARM64 = platform.machine() in ARM64_MACHINES
# The operation of the in-place add and subtract loops. The binding does not
# count the calls they leave to NumPy, since every float64 add and subtract
# reaches them, so the report leaves that column blank for them.
INPLACE = "inplace"
# Why the in-place loops are not installed on another architecture.
INPLACE_ARM64_ONLY = "ARM64 only: see docs/gradient-step-findings.md"
# Element count of the in-place probes: the fewest the loops take
# (INPLACE_MIN_N in hook_loop.c).
PROBE_INPLACE_N = 131072
# Value the in-place probes add to or subtract from each element.
PROBE_INPLACE_STEP = 0.375

# Element count of a probe array. Large enough for every size threshold in
# the binding, small enough that install() stays fast.
PROBE_N = 1000
# Element count of a probe that must be served by a function hook and not by
# the type slot behind it. The slot leaves runs this short to NumPy.
SMALL_PROBE_N = 16
# Position of the one element a probe expects a search to find.
PROBE_INDEX = 617
SMALL_PROBE_INDEX = 11
# Bounds the clip probes clamp to.
PROBE_LO, PROBE_HI = -0.5, 0.5
# Bounds the int64 clip probes clamp to: Python ints inside the probe's values,
# so that elements on both sides change.
PROBE_INT_BOUNDS = (-3, 5)
# Number of decimals the round probes round to, and the step between their
# values: a multiple of 0.375 times 100 ends in .5 whenever it is not whole,
# so the probes meet ties, which go to the even neighbour.
PROBE_DECIMALS = 2
PROBE_ROUND_STEP = 0.375
# How many of every `PROBE_STRIDE` elements a probe selects or sets.
PROBE_STRIDE = 3
# Number of bins and range of the histogram probe.
PROBE_BINS = 4
PROBE_RANGE = (0.0, 4.0)
# Byte value the bincount, sum and astype probes fill their array with.
PROBE_BYTE = 7
# Value the masked-assignment probe writes into a float array.
PROBE_FILL = 2.5
# Value the masked-assignment probe writes into an integer array.
PROBE_INT_FILL = 7

# Length of the runs the reduceat probe cuts its array into: the length the
# kernel was timed for.
PROBE_RUN = 16

# Columns of the matrix the argmax(axis=1) probes search: narrower than the
# rows the binding gives to the kernels, so its own loop takes them.
PROBE_ROWS_COLS = 3

# Columns of the matrix the any(axis=1) and any(axis=0) probes reduce: the
# width the first kernel was timed for. PROBE_N is a whole number of such
# rows' worth or more.
PROBE_COLS = 22
# A byte other than 1 that one True of the any(axis=0) probe is stored as.
# Its top bit is set, so a kernel that compared bytes as signed fails.
PROBE_TRUE_BYTE = 0x80

# Number of distinct codes the int16 and int64 bincount probes count: their
# element i is i modulo it. PROBE_N is not a multiple of it, so the first
# codes are counted once more than the others, and a kernel that dropped or
# misplaced a code would fail.
PROBE_CODES = 41

# Rows of the matrices the probes of the row loops accumulate down: PROBE_N
# elements in 25 rows of 40 columns.
PROBE_ROWS = 25
# Largest factor in the cumulative-product probe: its products are at most
# 3 ** PROBE_ROWS, below 2 ** 53, so every one is exact in float64.
PROBE_FACTOR = 3
# The argsort probes sort the values (PROBE_SORT_STEP * i modulo PROBE_N)
# divided by PROBE_SORT_TIES: the step has no factor in common with PROBE_N,
# so the values are shuffled, and the division leaves them in groups of equal
# elements, which a sort that is not stable would put in another order.
PROBE_SORT_STEP = 617
PROBE_SORT_TIES = 4

# The float sum probes add the values ((i * PROBE_SORT_STEP) mod
# PROBE_SUM_N - PROBE_SUM_N // 2) / PROBE_SUM_DIVISOR + PROBE_SUM_OFFSET.
# PROBE_SUM_N is more than one leaf of NumPy's pairwise summation (128
# elements) and not a multiple of 8, so the sum goes through splits, the
# leaves' eight running sums and the elements left over. With these values
# the float32 and the float64 sum in NumPy's order each differ from the sum
# added one after another, from a tree split at n / 2 not rounded to a
# multiple of 8, from leaves of 64 or 256 elements, and from leaves whose
# running sums are added in turn: a kernel that adds in any of those orders
# fails the probe.
PROBE_SUM_N = 1003
PROBE_SUM_DIVISOR = 7
PROBE_SUM_OFFSET = 0.1

# Element count of the sorted array the search probes look in. It differs
# from PROBE_N so that a kernel which confused its two lengths would fail.
PROBE_TABLE_N = 37


@dataclass(frozen=True)
class Probe:
    """One self-test call for one kernel."""

    # Kernel whose call counter must move when `call` runs.
    kernel: str
    # Makes the NumPy calls the kernel should take, on inputs it builds
    # afresh each time, and returns everything a caller could look at
    # afterwards: the results, and any array the calls changed in place.
    # `install()` runs it once with no hook applied and once with the hook,
    # and compares the two with `describe`.
    call: Callable[[], object]
    # True for a kernel that is not bit-identical to NumPy. Its probe only
    # runs when such kernels were asked for.
    inexact: bool = False


@dataclass(frozen=True)
class Hook:
    """One pointer inside NumPy that can be pointed at the binding."""

    # Name of the hook in the extension module.
    name: str
    # Kind of hook: one of the constants above.
    kind: str
    # One probe per kernel the hook serves.
    probes: tuple[Probe, ...]
    # Module of another library that holds the hooked object, or None for a
    # hook in NumPy. Such a hook is applied only while the module is in
    # sys.modules.
    module: str | None = None


@dataclass(frozen=True)
class KernelInfo:
    """How one kernel appears to a user."""

    # Kernel name, as in `kernels/`.
    name: str
    # The NumPy call the kernel stands in for, as a user writes it.
    numpy_call: str
    # Element type of the arrays it handles, as a NumPy type name.
    elements: str
    # Name of the operation the kernel serves, as the binding's counters of
    # calls passed on to NumPy name it, or None for a kernel without a hook.
    # An operation the binding does not count, such as INPLACE, has no
    # figure of calls left to NumPy.
    operation: str | None
    # Names of the hooks that deliver the kernel. Empty for one without a hook.
    hooks: tuple[str, ...]
    # Why the kernel has no hook, or None when it has one.
    unbound_reason: str | None = None
    # True for a loop built into the binding, for calls on short arrays or
    # for accumulations along the first axis of a matrix, or for in-place adds
    # and subtracts, which is plain C in the hook files and not a kernel from
    # `kernels/`: the judge never sees it, and `report()` shows it as exact C
    # and does not count it among the kernels.
    builtin: bool = False
    # What `report()` shows in its Kernel column for a row built into the
    # binding that is not C of its own, or None for C: "NumPy" for the ufunc
    # lane, which calls NumPy's own loops.
    implementation: str | None = None


def describe(value: object) -> tuple:
    """Reduce a probe's answer to a value that is equal for two answers exactly when a caller could not tell them apart.

    The comparison of `delivery check` (`describe` in delivery/check.py),
    which this package cannot import: an array is its type, element type,
    shape, the flags a caller can see and its bytes, so +0.0 differs from
    -0.0 and NaNs are compared by their bits; a NumPy scalar is its type,
    element type and bytes; a tuple or a list is described element by
    element; anything else is its type and its repr, so the Python int 1
    differs from np.intp(1), and the Python float 0.0 from -0.0.

    Args:
        value: What a probe's calls gave.

    Returns:
        A nested tuple of plain values, compared with `==`.
    """
    if isinstance(value, np.ndarray):
        contents = repr(value.tolist()) if value.dtype.hasobject else np.ascontiguousarray(value).tobytes()
        flags = (value.flags.c_contiguous, value.flags.f_contiguous, value.flags.writeable)
        return ("array", type(value).__name__, value.dtype.str, value.shape, flags, contents)
    if isinstance(value, np.generic):
        return ("scalar", type(value).__name__, value.dtype.str, value.tobytes())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(describe(item) for item in value))
    return ("object", type(value).__name__, repr(value))


def _marked(dtype: type, n: int, index: int, value: float) -> np.ndarray:
    """Return `n` zeros of `dtype` with `value` at `index`."""
    x = np.zeros(n, dtype)
    x[index] = value
    return x


def _argext_method(dtype: type, method: str, value: float) -> np.intp:
    """Probe the type slot: `x.argmax()` or `x.argmin()` on a long array with one marked element.

    Args:
        dtype: Element type of the probe array.
        method: "argmax" or "argmin".
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        What the method returned.
    """
    return getattr(_marked(dtype, PROBE_N, PROBE_INDEX, value), method)()


def _argext_function(dtype: type, function: Callable, value: float) -> np.intp:
    """Probe the function hook: `np.argmax(x)` or `np.argmin(x)` on a short array.

    The array is shorter than the type slot serves, so only the function hook
    can have taken the call.

    Args:
        dtype: Element type of the probe array.
        function: `np.argmax` or `np.argmin`.
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        What the function returned.
    """
    return function(_marked(dtype, SMALL_PROBE_N, SMALL_PROBE_INDEX, value))


def _argext_small_method(dtype: type, method: str, value: float) -> np.intp:
    """Probe the method hook: `x.argmax()` or `x.argmin()` on a short array.

    The array is shorter than the type slot serves, so only the method hook
    can have taken the call.

    Args:
        dtype: Element type of the probe array.
        method: "argmax" or "argmin".
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        What the method returned.
    """
    return getattr(_marked(dtype, SMALL_PROBE_N, SMALL_PROBE_INDEX, value), method)()


def _argext_rows(dtype: type, rows: Callable[[np.ndarray], np.ndarray], value: float) -> np.ndarray:
    """Probe the rows path of a function or method hook: argmax or argmin along axis 1 of a narrow matrix.

    Row r holds `value` at column r modulo PROBE_ROWS_COLS and zeros
    elsewhere, so the answer moves from row to row; a row's other zeros tie
    with each other and never win.

    Args:
        dtype: Element type of the probe matrix.
        rows: Makes the call on the matrix: `np.argmax(m, axis=1)` and so on.
        value: The marked element: positive for argmax, negative for argmin.

    Returns:
        What the call returned.
    """
    count = PROBE_N // PROBE_ROWS_COLS
    m = np.zeros((count, PROBE_ROWS_COLS), dtype)
    m[np.arange(count), np.arange(count) % PROBE_ROWS_COLS] = value
    return rows(m)


def _argext_rows_function(function: Callable) -> Callable[[np.ndarray], np.ndarray]:
    """Return `function(m, axis=1)` as a function of `m`, for `np.argmax` or `np.argmin`."""
    return lambda m: function(m, axis=1)


def _argext_rows_method(method: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return `m.argmax(axis=1)` or `m.argmin(axis=1)` as a function of `m`."""
    return lambda m: getattr(m, method)(axis=1)


def _clip_with(dtype: type, clip: Callable[[np.ndarray, float, float], np.ndarray]) -> np.ndarray:
    """Probe a clip hook: values from -1 to 1, with zeros of both signs, clamped to the probe bounds.

    A zero keeps its sign through `np.clip` from NumPy 2.1 on, as it does
    through the kernels; NumPy 2.0 made every zero it returned +0.0. The
    zeros are inside the bounds, so no bound ties with them.

    The loop hook takes every call the function and method hooks take, so
    the probe of those two proves that the result is right and that a clip
    kernel ran, not which hook ran it.

    Args:
        dtype: Element type of the probe array.
        clip: Clips its first argument to the bounds given by the other two.

    Returns:
        The clipped array.
    """
    x = np.linspace(-1.0, 1.0, PROBE_N).astype(dtype)
    x[PROBE_INDEX] = -0.0
    x[SMALL_PROBE_INDEX] = 0.0
    return clip(x, PROBE_LO, PROBE_HI)


def _clip_method(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return `x.clip(lo, hi)`."""
    return x.clip(lo, hi)


def _clip(dtype: type) -> np.ndarray:
    """Probe the clip loop or the function hook: `np.clip(x, lo, hi)`."""
    return _clip_with(dtype, np.clip)


def _inplace(in_place: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> tuple[bool, np.ndarray]:
    """Probe an in-place loop: `w += b` or `w -= b` on PROBE_INPLACE_N float64 elements.

    Args:
        in_place: The in-place operator, `operator.iadd` or `operator.isub`.

    Returns:
        Whether the operator returned `w` itself, and `w` afterwards.
    """
    w = np.arange(PROBE_INPLACE_N, dtype=np.float64)
    b = np.full(PROBE_INPLACE_N, PROBE_INPLACE_STEP)
    return in_place(w, b) is w, w


def _clip_i64_with(clip: Callable[[np.ndarray, int, int], np.ndarray]) -> np.ndarray:
    """Probe the int64 path of a clip hook: SMALL_PROBE_N values around zero clamped to PROBE_INT_BOUNDS.

    Args:
        clip: Clips its first argument to the bounds given by the other two.

    Returns:
        The clipped array.
    """
    x = np.arange(SMALL_PROBE_N, dtype=np.int64) - SMALL_PROBE_N // 2
    lo, hi = PROBE_INT_BOUNDS
    return clip(x, lo, hi)


def _round_with(round_: Callable[[np.ndarray, int], np.ndarray]) -> tuple[np.ndarray, list]:
    """Probe a round hook: SMALL_PROBE_N multiples of PROBE_ROUND_STEP rounded to PROBE_DECIMALS decimals.

    The multiples meet ties, which go to the even neighbour, and the small
    negative ones round to -0.0. Each value is also rounded alone, as an
    np.float64 and as a zero-dimensional array.

    Args:
        round_: Rounds its first argument to the number of decimals given by
            the second.

    Returns:
        The rounded array, and the list of the values rounded alone.
    """
    scale = 10.0 ** PROBE_DECIMALS
    x = (np.arange(SMALL_PROBE_N) - SMALL_PROBE_N // 2) * PROBE_ROUND_STEP / scale
    alone = [round_(number, PROBE_DECIMALS) for v in x for number in (v, np.array(v))]
    return round_(x, PROBE_DECIMALS), alone


def _round_method(x: np.ndarray, decimals: int) -> np.ndarray:
    """Return `x.round(decimals)`."""
    return x.round(decimals)


def _median() -> list:
    """Probe the function hook: `np.median(x)` on the halves of 1 to n, short and long.

    SMALL_PROBE_N and SMALL_PROBE_N - 1 halves, in descending order, reach
    the loop that sorts a copy. PROBE_N and PROBE_N - 1, shuffled by
    PROBE_SORT_STEP (which has no factor in common with either), reach the
    selection by bits that serves longer arrays; an array in order would be
    answered from its middle places without it. The even counts are asked
    as `np.median(x, axis=None)`. The even count's median is the mean of its
    two middle elements and the odd count's its middle element, so a loop
    that took the wrong element, or did not order them, fails.

    Returns:
        The four medians.
    """
    answers = []
    for n in (SMALL_PROBE_N, SMALL_PROBE_N - 1, PROBE_N, PROBE_N - 1):
        order = np.arange(n, 0, -1) if n <= SMALL_PROBE_N else np.arange(n) * PROBE_SORT_STEP % n + 1
        x = order * 0.5
        answers.append(np.median(x, axis=None) if n % 2 == 0 else np.median(x))
    return answers


def _stride_mask() -> np.ndarray:
    """Return a boolean mask that is True at every PROBE_STRIDE-th position."""
    return np.arange(PROBE_N) % PROBE_STRIDE == 0


def _selection_values(dtype: type) -> np.ndarray:
    """Return the array the selection and masked-assignment probes work on: 0 to PROBE_N - 1, with -0.0 first.

    The first PROBE_STRIDE elements are set to -0.0, which is a negative zero
    in a float type and a plain zero in any other. The stride mask selects
    the first of them and not the others, so a kernel that lost the sign of
    a zero it moved, or of one it should have left, fails.

    Args:
        dtype: Element type of the array.

    Returns:
        A new array of PROBE_N elements.
    """
    x = np.arange(PROBE_N).astype(dtype)
    x[:PROBE_STRIDE] = -0.0
    return x


def _compress(dtype: type) -> np.ndarray:
    """Probe the indexing hook: `x[mask]`."""
    return _selection_values(dtype)[_stride_mask()]


def _compress_function(dtype: type, function: Callable) -> np.ndarray:
    """Probe a function hook: `np.compress(mask, x)` or `np.extract(mask, x)`.

    Args:
        dtype: Element type of the probe array.
        function: `np.compress` or `np.extract`; both take the mask first.

    Returns:
        What the function returned.
    """
    return function(_stride_mask(), _selection_values(dtype))


def _assign(x: np.ndarray, mask: np.ndarray, fill: object) -> None:
    """Run `x[mask] = fill`."""
    x[mask] = fill


def _putmask(x: np.ndarray, mask: np.ndarray, fill: object) -> None:
    """Run `np.putmask(x, mask, fill)`."""
    np.putmask(x, mask, fill)


def _copyto(x: np.ndarray, mask: np.ndarray, fill: object) -> None:
    """Run `np.copyto(x, fill, where=mask)`."""
    np.copyto(x, fill, where=mask)


def _maskfill(assign: Callable[[np.ndarray, np.ndarray, object], None], dtype: type = np.float64,
              fill: object = PROBE_FILL) -> np.ndarray:
    """Probe a masked assignment: `x[mask] = value` or one of its function forms.

    Args:
        assign: Sets the elements of its first argument where its second is
            True to its third, in place.
        dtype: Element type of the probe array.
        fill: The value written, of a kind the binding takes for `dtype`.

    Returns:
        The array afterwards.
    """
    x = _selection_values(dtype)
    assign(x, _stride_mask(), fill)
    return x


def _nonzero_method(mask: np.ndarray) -> tuple:
    """Return `mask.nonzero()`."""
    return mask.nonzero()


def _nonzero(function: Callable[[np.ndarray], tuple]) -> tuple:
    """Probe `mask.nonzero()` or `np.where(mask)` on the stride mask.

    Args:
        function: Takes the mask and returns NumPy's tuple of index arrays.

    Returns:
        What the function returned.
    """
    return function(_stride_mask())


def _nonzero_2d(function: Callable[[np.ndarray], tuple]) -> tuple:
    """Probe `mask.nonzero()` or `np.where(mask)` on a boolean matrix: the stride mask in PROBE_ROWS rows.

    Args:
        function: Takes the mask and returns NumPy's tuple of index arrays.

    Returns:
        What the function returned.
    """
    return function(_stride_mask().reshape(PROBE_ROWS, -1))


def _where() -> np.ndarray:
    """Probe the function hook: `np.where(mask, a, b)` on float64, with a mask True at every PROBE_STRIDE-th position.

    The two arrays differ at every position, so an element taken from the
    wrong one fails the probe. The first element of `a`, which is taken, and
    the second of `b`, which is taken too, are -0.0.

    Returns:
        What the call returned.
    """
    a = np.arange(PROBE_N, dtype=np.float64)
    b = -a - 1.0
    a[0] = -0.0
    b[1] = -0.0
    return np.where(_stride_mask(), a, b)


def _cumsum(dtype: type) -> np.ndarray:
    """Probe the method hook: `x.cumsum()` on ones, whose running sum is exact in every type."""
    return np.ones(PROBE_N, dtype).cumsum()


def _cumsum_bool() -> np.ndarray:
    """Probe the method hook: `mask.cumsum()` on a mask that is True at every PROBE_STRIDE-th position."""
    return _stride_mask().cumsum()


def _running_max_probe() -> list[float]:
    """Return the values of the running-maximum probes, as Python floats.

    0, 1, -2, 3, -4, ...: the odd elements are each a new maximum and the even
    ones never are, so a kernel that kept the wrong one fails. Every value is
    a whole number below 2^53, exact in both int64 and float64.
    """
    return [float(i if i % 2 else -i) for i in range(PROBE_N)]


def _cummax(ufunc: np.ufunc) -> np.ndarray:
    """Probe the ufunc method hook: `ufunc.accumulate(x)` on int64, for np.maximum or np.fmax.

    Args:
        ufunc: `np.maximum` or `np.fmax`; on integers both are the running maximum.

    Returns:
        What the call returned.
    """
    return ufunc.accumulate(np.array(_running_max_probe(), np.int64))


def _cumfmax() -> np.ndarray:
    """Probe the ufunc method hook: `np.fmax.accumulate(x)` on float64 with a NaN at every PROBE_STRIDE-th position after the first.

    A kernel that let a NaN through, or that NumPy's NaN-propagating
    `np.maximum` answered, fails.
    """
    values = _running_max_probe()
    for i in range(PROBE_STRIDE, PROBE_N, PROBE_STRIDE):
        values[i] = float("nan")
    return np.fmax.accumulate(np.array(values, np.float64))


def _rows_probe(call: Callable[[np.ndarray], np.ndarray], values: list[float]) -> np.ndarray:
    """Probe a row loop: an accumulation along the first axis of a PROBE_ROWS-row float64 matrix.

    Args:
        call: Makes the NumPy call on the matrix.
        values: The matrix's elements in C order.

    Returns:
        What the call returned.
    """
    return call(np.array(values, np.float64).reshape(PROBE_ROWS, -1))


def _rows_cumsum(call: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Probe the row loop of np.add: `call` sums the running-maximum probe values down the columns, exactly."""
    return _rows_probe(call, _running_max_probe())


def _rows_cumprod() -> np.ndarray:
    """Probe the row loop of np.multiply: `m.cumprod(axis=0)` on whole numbers 1 to PROBE_FACTOR, exactly."""
    values = [float(1 + i % PROBE_FACTOR) for i in range(PROBE_N)]
    return _rows_probe(lambda m: m.cumprod(axis=0), values)


def _rows_cummax() -> np.ndarray:
    """Probe the row loop of np.maximum: `np.maximum.accumulate(m)` on the running-maximum probe values."""
    return _rows_probe(np.maximum.accumulate, _running_max_probe())


def _rows_cumfmax() -> np.ndarray:
    """Probe the row loop of np.fmax: `np.fmax.accumulate(m)` with a NaN at every PROBE_STRIDE-th element after the first row.

    The first row holds numbers, so no result is a NaN; a loop that let a NaN
    through, or propagated it as np.maximum does, fails.
    """
    values = _running_max_probe()
    cols = PROBE_N // PROBE_ROWS
    for i in range(cols, PROBE_N, PROBE_STRIDE):
        values[i] = float("nan")
    return _rows_probe(np.fmax.accumulate, values)


def _fmax_reduceat() -> np.ndarray:
    """Probe the ufunc method hook: `np.fmax.reduceat(x, starts)` on float64, in runs of PROBE_RUN.

    The values are those of the running-maximum probes, with a NaN at every
    PROBE_STRIDE-th position; every run holds numbers, so no result is a NaN.
    The last start is repeated, which gives the first of the two the single
    element at that start, as NumPy does when a start is not below the next.
    A kernel that let a NaN through, or reduced the wrong elements, fails.
    """
    values = _running_max_probe()
    for i in range(PROBE_STRIDE, PROBE_N, PROBE_STRIDE):
        values[i] = float("nan")
    starts = list(range(0, PROBE_N, PROBE_RUN))
    starts.insert(-1, starts[-1])
    return np.fmax.reduceat(np.array(values, np.float64), np.array(starts, np.intp))


def _any_rows(reduce: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Probe an any(axis=1) hook on a boolean matrix of PROBE_COLS columns.

    Every PROBE_STRIDE-th row holds one True, at a column that moves from row
    to row; the others hold none. A kernel that reduced the wrong elements or
    along the wrong axis fails.

    Args:
        reduce: Takes the matrix and returns NumPy's `m.any(axis=1)` for it.

    Returns:
        What the call returned.
    """
    rows = PROBE_N // PROBE_COLS
    m = np.zeros((rows, PROBE_COLS), np.bool_)
    marked = np.arange(0, rows, PROBE_STRIDE)
    m[marked, marked % PROBE_COLS] = True
    return reduce(m)


def _any_method(m: np.ndarray) -> np.ndarray:
    """Return `m.any(axis=1)`."""
    return m.any(axis=1)


def _any_function(m: np.ndarray) -> np.ndarray:
    """Return `np.any(m, axis=1)`."""
    return np.any(m, axis=1)


def _any_cols(reduce: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Probe an any(axis=0) hook on a boolean matrix of PROBE_COLS columns.

    Every PROBE_STRIDE-th column holds one True, at a row that moves from
    column to column; the others hold none. The first of them is stored as
    the byte PROBE_TRUE_BYTE rather than 1, which NumPy takes as True and
    answers with a 1. A kernel that reduced the wrong elements or along the
    wrong axis, or that passed such a byte on, fails.

    Args:
        reduce: Takes the matrix and returns NumPy's `m.any(axis=0)` for it.

    Returns:
        What the call returned.
    """
    rows = PROBE_N // PROBE_COLS
    m = np.zeros((rows, PROBE_COLS), np.bool_)
    marked = np.arange(0, PROBE_COLS, PROBE_STRIDE)
    m[(marked * PROBE_STRIDE) % rows, marked] = True
    m.view(np.uint8)[0, 0] = PROBE_TRUE_BYTE
    return reduce(m)


def _any_cols_method(m: np.ndarray) -> np.ndarray:
    """Return `m.any(axis=0)`."""
    return m.any(axis=0)


def _any_cols_function(m: np.ndarray) -> np.ndarray:
    """Return `np.any(m, axis=0)`."""
    return np.any(m, axis=0)


def _astype() -> np.ndarray:
    """Probe the method hook: `x.astype(np.float32)` on bytes."""
    return np.full(PROBE_N, PROBE_BYTE, np.uint8).astype(np.float32)


def _sum_method() -> np.generic:
    """Probe the method hook: `x.sum()` on bytes."""
    return np.full(PROBE_N, PROBE_BYTE, np.uint8).sum()


def _sum_function() -> np.generic:
    """Probe the function hook: `np.sum(x)` on bytes."""
    return np.sum(np.full(PROBE_N, PROBE_BYTE, np.uint8))


def _sum_float(dtype: type, call: Callable[[np.ndarray], object]) -> object:
    """Probe `np.sum(x)` or `x.sum()` on floats whose sum depends on the order of the additions.

    Args:
        dtype: np.float32 or np.float64.
        call: Makes the call on the array.

    Returns:
        What the call returned.
    """
    steps = (np.arange(PROBE_SUM_N) * PROBE_SORT_STEP) % PROBE_SUM_N - PROBE_SUM_N // 2
    return call((steps / PROBE_SUM_DIVISOR + PROBE_SUM_OFFSET).astype(dtype))


def _sum_method_of(x: np.ndarray) -> object:
    """Return `x.sum()`."""
    return x.sum()


def _count_nonzero(dtype: type) -> object:
    """Probe the function hook: `np.count_nonzero(x)` on the stride mask as `dtype`."""
    return np.count_nonzero(_stride_mask().astype(dtype))


def _histogram() -> tuple:
    """Probe the function hook: `np.histogram` on values that fill each bin equally."""
    return np.histogram((np.arange(PROBE_N) % PROBE_BINS) + 0.5, PROBE_BINS, PROBE_RANGE)


def _bincount() -> np.ndarray:
    """Probe the function hook: `np.bincount` on equal bytes."""
    return np.bincount(np.full(PROBE_N, PROBE_BYTE, np.uint8))


def _bincount_codes(dtype: type) -> np.ndarray:
    """Probe the function hook: `np.bincount` on int16 or int64 codes that cycle through PROBE_CODES values.

    Args:
        dtype: np.int16 or np.int64.

    Returns:
        What the call returned.
    """
    return np.bincount(np.arange(PROBE_N, dtype=dtype) % PROBE_CODES)


def _unique() -> np.ndarray:
    """Probe the function hook: `np.unique` on PROBE_N int64 values with gaps, below zero and above it.

    The elements are -3, 0 and 7 repeated, so a kernel that wrote every value
    of the range, or lost the sign, or did not sort, fails.
    """
    return np.unique(np.resize(np.array([7, -3, 0], np.int64), PROBE_N))


def _isin(dtype: type = np.int64) -> np.ndarray:
    """Probe the function hook: `np.isin` with two test values.

    Args:
        dtype: Element type of the array searched, np.int64 or np.int32.

    Returns:
        What the call returned.
    """
    return np.isin(np.arange(PROBE_N, dtype=dtype), [PROBE_INDEX, SMALL_PROBE_INDEX])


def _searchsorted() -> list:
    """Probe the method hook: `np.searchsorted` for needles between whole numbers.

    The needles are searched as float64 and as float32, which the float64
    kernel takes converted.
    """
    table = np.arange(PROBE_TABLE_N, dtype=np.float64)
    needles = (np.arange(PROBE_N) % PROBE_TABLE_N) + 0.5
    return [np.searchsorted(table, v) for v in (needles, needles.astype(np.float32))]


def _searchsorted_right() -> list:
    """Probe the method hook: `np.searchsorted` with `side='right'` for needles that are elements.

    Every needle equals one element, so the left side's answer is one less at
    each of them: a call that reached the wrong kernel fails. The needles are
    searched as float64 and as float32.
    """
    table = np.arange(PROBE_TABLE_N, dtype=np.float64)
    needles = (np.arange(PROBE_N) % PROBE_TABLE_N).astype(np.float64)
    return [np.searchsorted(table, v, side="right") for v in (needles, needles.astype(np.float32))]


def _searchsorted_i64(side: str) -> np.ndarray:
    """Probe the method hook: `np.searchsorted` on int64 for needles that are elements.

    Every needle equals one element, so the two sides differ by one at each of
    them: a call that reached the kernel of the other side fails.

    Args:
        side: The `side` argument of the call, 'left' or 'right'.

    Returns:
        What the call returned.
    """
    table = np.arange(PROBE_TABLE_N, dtype=np.int64)
    return np.searchsorted(table, np.arange(PROBE_N, dtype=np.int64) % PROBE_TABLE_N, side=side)


def _interp() -> np.ndarray:
    """Probe the function hook: `np.interp` through points on the line y = 2x."""
    points = np.arange(PROBE_TABLE_N, dtype=np.float64)
    queries = (np.arange(PROBE_N) % (PROBE_TABLE_N - 1)) + 0.5
    return np.interp(queries, points, 2.0 * points)


def _bottleneck_probe_matrix() -> np.ndarray:
    """Return the matrix the bottleneck probes reduce: random values with every special value bottleneck treats apart.

    Column 0 is all NaN, column 1 all -0.0, column 2 zeros of both signs ending
    in +0.0 and column 3 ending in -0.0, column 4 +inf and NaN, column 5 -inf
    and NaN; the rest are normal values with a NaN in every seventh row.
    """
    rows, cols = BOTTLENECK_PROBE_SHAPE
    x = np.random.default_rng(BOTTLENECK_PROBE_SEED).standard_normal((rows, cols))
    x[::7, 6:] = np.nan
    x[:, 0] = np.nan
    x[:, 1] = -0.0
    x[:, 2] = np.where(np.arange(rows) % 2, 0.0, -0.0)
    x[:, 3] = np.where(np.arange(rows) % 2, -0.0, 0.0)
    x[:, 4] = np.where(np.arange(rows) % 2, np.inf, np.nan)
    x[:, 5] = np.where(np.arange(rows) % 2, -np.inf, np.nan)
    return x


def _bottleneck(function: str) -> list:
    """Probe a bottleneck hook: the function along axis 0 of the probe matrix, in two layouts.

    The layouts are C order and a view of rolling windows. bottleneck is in
    sys.modules when this runs. With the hook removed this is bottleneck's
    own loop, so `install()` compares the kernel with the bottleneck in use.

    Args:
        function: "nanmean", "nanstd", "nanmin" or "nanmax"; nanstd is called
            with ddof=1, as empyrical calls it.

    Returns:
        The two results.
    """
    call = getattr(sys.modules[BOTTLENECK_MODULE], function)
    keywords = {"ddof": 1} if function == "nanstd" else {}
    x = _bottleneck_probe_matrix()
    windows = np.lib.stride_tricks.as_strided(x.ravel(), (x.shape[0], x.size - x.shape[0] + 1), (8, 8))
    return [call(a, axis=0, **keywords) for a in (x, windows)]


def _argsort(dtype: type) -> np.ndarray:
    """Probe the method hook: `np.argsort(x, kind="stable")` on shuffled values with ties.

    The values come in groups of PROBE_SORT_TIES equal elements, which a sort
    that is not stable would put in another order.

    Args:
        dtype: Element type of the probe array.

    Returns:
        What the call returned.
    """
    values = [(PROBE_SORT_STEP * i % PROBE_N) // PROBE_SORT_TIES for i in range(PROBE_N)]
    return np.argsort(np.array(values, dtype), kind="stable")


# Element count of the ufunc lane's probes: far below every threshold of the lane.
LANE_PROBE_N = 16
# Rows and columns of the matrix the lane's row probe subtracts a row from.
LANE_PROBE_SHAPE = (4, 5)
# The value the lane's comparison probe compares with.
LANE_PROBE_THRESHOLD = 3


def _lane_values(offset: float, step: float, n: int = LANE_PROBE_N) -> list[float]:
    """Return `n` floats, `offset + step * i`, all exact in float32.

    Args:
        offset: The first value.
        step: The difference between neighbours; a power of two or a small multiple of one.
        n: How many.

    Returns:
        The values, as Python floats.
    """
    return [offset + step * i for i in range(n)]


def _lane_add() -> np.ndarray:
    """Probe the ufunc lane: `a + b` on two float64 arrays of one shape."""
    return np.array(_lane_values(0.25, 0.5)) + np.array(_lane_values(-3.0, 0.125))


def _lane_number_f32() -> np.ndarray:
    """Probe the ufunc lane: `f * 2.0` with a Python float, on float32.

    The Python float must be converted to float32, as NumPy converts a Python
    number of no fixed type (NEP 50), and the result must stay float32.
    """
    return np.array(_lane_values(-1.5, 0.25), np.float32) * 2.0


def _lane_compare() -> np.ndarray:
    """Probe the ufunc lane: `i > 3` on int64, a comparison with a Python int."""
    return np.arange(LANE_PROBE_N, dtype=np.int64) > LANE_PROBE_THRESHOLD


def _lane_row() -> np.ndarray:
    """Probe the ufunc lane: `np.subtract(m, row)`, a row taken from every row of a matrix."""
    rows, cols = LANE_PROBE_SHAPE
    matrix = [_lane_values(r, 0.5, cols) for r in range(rows)]
    return np.subtract(np.array(matrix), np.array(_lane_values(0.75, -0.25, cols)))


def _lane_inplace() -> tuple[bool, np.ndarray]:
    """Probe the ufunc lane with an output: `x += y`, then `np.add(x, y, out=x)`.

    Returns:
        Whether the second call returned `x` itself, and `x` afterwards.
    """
    x, y = np.array(_lane_values(1.0, 0.5)), np.array(_lane_values(0.0, 0.25))
    x += y
    return np.add(x, y, out=x) is x, x


def _lane_flags() -> tuple[bool, np.ndarray, np.ndarray]:
    """Probe the ufunc lane's floating-point errors: `a / 0.0` under `np.errstate(divide="raise")`.

    The lane must hand the divide-by-zero flag to NumPy's handler, which
    raises here, and must return infinities with the flag ignored. A last
    plain call moves the lane's counter.

    Returns:
        Whether the division raised FloatingPointError, its result with the
        flag ignored, and the result of the plain call.
    """
    a = np.array(_lane_values(1.0, 1.0))
    raised = False
    try:
        with np.errstate(divide="raise"):
            a / 0.0
    except FloatingPointError:
        raised = True
    with np.errstate(divide="ignore"):
        result = a / 0.0
    return raised, result, a + a


# Every hook, in the order install() applies them: the deepest first.
HOOKS: tuple[Hook, ...] = (
    Hook("slot.argmax.f32", TYPE_SLOT, (Probe("argmax_f32", partial(_argext_method, np.float32, "argmax", 1.0)),)),
    Hook("slot.argmax.f64", TYPE_SLOT, (Probe("argmax_f64", partial(_argext_method, np.float64, "argmax", 1.0)),)),
    Hook("slot.argmin.f32", TYPE_SLOT, (Probe("argmin_f32", partial(_argext_method, np.float32, "argmin", -1.0)),)),
    Hook("slot.argmin.f64", TYPE_SLOT, (Probe("argmin_f64", partial(_argext_method, np.float64, "argmin", -1.0)),)),
    Hook("loop.clip.f32", INNER_LOOP, (Probe("clip_f32", partial(_clip, np.float32)),)),
    Hook("loop.clip.f64", INNER_LOOP, (Probe("clip_f64", partial(_clip, np.float64)),)),
    *((
        Hook("loop.add.f64", INNER_LOOP, (Probe("add_inplace_f64", partial(_inplace, operator.iadd)),)),
        Hook("loop.subtract.f64", INNER_LOOP,
             (Probe("subtract_inplace_f64", partial(_inplace, operator.isub)),)),
    ) if ON_ARM64 else ()),
    Hook("function.argmax", FUNCTION, (
        Probe("argmax_f32", partial(_argext_function, np.float32, np.argmax, 1.0)),
        Probe("argmax_f64", partial(_argext_function, np.float64, np.argmax, 1.0)),
        Probe("argmax_rows_f32", partial(_argext_rows, np.float32, _argext_rows_function(np.argmax), 1.0)),
        Probe("argmax_rows_f64", partial(_argext_rows, np.float64, _argext_rows_function(np.argmax), 1.0)),
    )),
    Hook("function.argmin", FUNCTION, (
        Probe("argmin_f32", partial(_argext_function, np.float32, np.argmin, -1.0)),
        Probe("argmin_f64", partial(_argext_function, np.float64, np.argmin, -1.0)),
        Probe("argmin_rows_f32", partial(_argext_rows, np.float32, _argext_rows_function(np.argmin), -1.0)),
        Probe("argmin_rows_f64", partial(_argext_rows, np.float64, _argext_rows_function(np.argmin), -1.0)),
    )),
    Hook("function.count_nonzero", FUNCTION, (
        Probe("count_nonzero_i64", partial(_count_nonzero, np.int64)),
        Probe("count_nonzero_f64", partial(_count_nonzero, np.float64)),
    )),
    Hook("function.sum", FUNCTION, (
        Probe("sum_u8", _sum_function),
        Probe("sum_f32", partial(_sum_float, np.float32, np.sum)),
        Probe("sum_f64", partial(_sum_float, np.float64, np.sum)),
    )),
    Hook("function.histogram", FUNCTION, (Probe("histogram_f64", _histogram),)),
    Hook("function.bincount", FUNCTION, (
        Probe("bincount_u8", _bincount),
        Probe("bincount_i16", partial(_bincount_codes, np.int16)),
        Probe("bincount_i64", partial(_bincount_codes, np.int64)),
    )),
    Hook("function.isin", FUNCTION, (
        Probe("isin_i64", _isin),
        Probe("isin_i32", partial(_isin, np.int32)),
    )),
    Hook("function.interp", FUNCTION, (Probe("interp_f64", _interp),)),
    Hook("function.compress", FUNCTION, (
        Probe("compress_f32", partial(_compress_function, np.float32, np.compress)),
        Probe("compress_f64", partial(_compress_function, np.float64, np.compress)),
        Probe("compress_u8", partial(_compress_function, np.uint8, np.compress)),
        Probe("compress_i16", partial(_compress_function, np.int16, np.compress)),
    )),
    Hook("function.extract", FUNCTION, (
        Probe("compress_f32", partial(_compress_function, np.float32, np.extract)),
        Probe("compress_f64", partial(_compress_function, np.float64, np.extract)),
        Probe("compress_u8", partial(_compress_function, np.uint8, np.extract)),
        Probe("compress_i16", partial(_compress_function, np.int16, np.extract)),
    )),
    Hook("function.putmask", FUNCTION, (
        Probe("maskfill_f64", partial(_maskfill, _putmask)),
        Probe("maskfill_f32", partial(_maskfill, _putmask, np.float32)),
    )),
    Hook("function.copyto", FUNCTION, (
        Probe("maskfill_f64", partial(_maskfill, _copyto)),
        Probe("maskfill_f32", partial(_maskfill, _copyto, np.float32)),
    )),
    Hook("function.where", FUNCTION, (
        Probe("nonzero_bool", partial(_nonzero, np.where)),
        Probe("nonzero_2d_bool", partial(_nonzero_2d, np.where)),
        Probe("where_f64", _where),
    )),
    Hook("function.clip", FUNCTION, (
        Probe("clip_f32", partial(_clip, np.float32)),
        Probe("clip_f64", partial(_clip, np.float64)),
        Probe("clip_i64", partial(_clip_i64_with, np.clip)),
    )),
    Hook("function.any", FUNCTION, (
        Probe("any_rows_bool", partial(_any_rows, _any_function)),
        Probe("any_cols_bool", partial(_any_cols, _any_cols_function)),
    )),
    Hook("function.round", FUNCTION, (Probe("round_f64", partial(_round_with, np.round)),)),
    Hook("function.around", FUNCTION, (Probe("round_f64", partial(_round_with, np.around)),)),
    Hook("function.median", FUNCTION, (Probe("median_f64", _median),)),
    Hook("function.unique", FUNCTION, (Probe("unique_i64", _unique),)),
    Hook("method.astype", METHOD, (Probe("cast_u8_f32", _astype),)),
    Hook("method.sum", METHOD, (
        Probe("sum_u8", _sum_method),
        Probe("sum_f32", partial(_sum_float, np.float32, _sum_method_of)),
        Probe("sum_f64", partial(_sum_float, np.float64, _sum_method_of)),
    )),
    Hook("method.nonzero", METHOD, (
        Probe("nonzero_bool", partial(_nonzero, _nonzero_method)),
        Probe("nonzero_2d_bool", partial(_nonzero_2d, _nonzero_method)),
    )),
    Hook("method.cumsum", METHOD, (
        Probe("cumsum_i64", partial(_cumsum, np.int64)),
        Probe("cumsum_f32", partial(_cumsum, np.float32), inexact=True),
        Probe("cumsum_f64", partial(_cumsum, np.float64), inexact=True),
        Probe("cumsum_bool", _cumsum_bool),
        Probe("cumsum_rows_f64", partial(_rows_cumsum, lambda m: m.cumsum(axis=0))),
    )),
    Hook("method.searchsorted", METHOD, (
        Probe("searchsorted_f64", _searchsorted),
        Probe("searchsorted_right_f64", _searchsorted_right),
        Probe("searchsorted_i64", partial(_searchsorted_i64, "left")),
        Probe("searchsorted_right_i64", partial(_searchsorted_i64, "right")),
    )),
    Hook("method.argmax", METHOD, (
        Probe("argmax_f32", partial(_argext_small_method, np.float32, "argmax", 1.0)),
        Probe("argmax_f64", partial(_argext_small_method, np.float64, "argmax", 1.0)),
        Probe("argmax_rows_f32", partial(_argext_rows, np.float32, _argext_rows_method("argmax"), 1.0)),
        Probe("argmax_rows_f64", partial(_argext_rows, np.float64, _argext_rows_method("argmax"), 1.0)),
    )),
    Hook("method.argmin", METHOD, (
        Probe("argmin_f32", partial(_argext_small_method, np.float32, "argmin", -1.0)),
        Probe("argmin_f64", partial(_argext_small_method, np.float64, "argmin", -1.0)),
        Probe("argmin_rows_f32", partial(_argext_rows, np.float32, _argext_rows_method("argmin"), -1.0)),
        Probe("argmin_rows_f64", partial(_argext_rows, np.float64, _argext_rows_method("argmin"), -1.0)),
    )),
    Hook("method.clip", METHOD, (
        Probe("clip_f32", partial(_clip_with, np.float32, _clip_method)),
        Probe("clip_f64", partial(_clip_with, np.float64, _clip_method)),
        Probe("clip_i64", partial(_clip_i64_with, _clip_method)),
    )),
    Hook("ufunc.accumulate", METHOD, (
        Probe("cummax_i64", partial(_cummax, np.maximum)),
        Probe("cummax_i64", partial(_cummax, np.fmax)),
        Probe("cumfmax_f64", _cumfmax),
        Probe("cumsum_rows_f64", partial(_rows_cumsum, np.add.accumulate)),
        Probe("cumprod_rows_f64", _rows_cumprod),
        Probe("cummax_rows_f64", _rows_cummax),
        Probe("cumfmax_rows_f64", _rows_cumfmax),
    )),
    Hook("ufunc.reduceat", METHOD, (Probe("fmax_reduce_f64", _fmax_reduceat),)),
    Hook("method.any", METHOD, (
        Probe("any_rows_bool", partial(_any_rows, _any_method)),
        Probe("any_cols_bool", partial(_any_cols, _any_cols_method)),
    )),
    Hook("method.round", METHOD, (Probe("round_f64", partial(_round_with, _round_method)),)),
    Hook("method.argsort", METHOD, (
        Probe("argsort_f64", partial(_argsort, np.float64)),
        Probe("argsort_i64", partial(_argsort, np.int64)),
        Probe("argsort_u64", partial(_argsort, np.uint64)),
    )),
    Hook("index.get", INDEXING, (
        Probe("compress_f32", partial(_compress, np.float32)),
        Probe("compress_f64", partial(_compress, np.float64)),
        # The kernels move bits, so int32 goes to the float32 one.
        Probe("compress_f32", partial(_compress, np.int32)),
        # int8 and bool go to compress_u8, uint16 to compress_i16.
        Probe("compress_u8", partial(_compress, np.uint8)),
        Probe("compress_u8", partial(_compress, np.bool_)),
        Probe("compress_i16", partial(_compress, np.int16)),
    )),
    Hook("index.set", INDEXING, (
        Probe("maskfill_f64", partial(_maskfill, _assign)),
        Probe("maskfill_f32", partial(_maskfill, _assign, np.float32)),
        # The kernels move bits: int64 goes to the float64 one, int32 to the
        # float32 one.
        Probe("maskfill_f64", partial(_maskfill, _assign, np.int64, PROBE_INT_FILL)),
        Probe("maskfill_f32", partial(_maskfill, _assign, np.int32, PROBE_INT_FILL)),
    )),
    # bottleneck's functions, applied only when bottleneck is imported.
    *(Hook(f"bottleneck.{function}", BOTTLENECK,
           (Probe(f"{function}_cols_f64", partial(_bottleneck, function)),),
           module=BOTTLENECK_MODULE)
      for function in ("nanmean", "nanstd", "nanmin", "nanmax")),
    # The ufunc lane, last: the inner-loop hooks above can change which loop
    # NumPy hands it.
    Hook("ufunc.call", UFUNC, (
        Probe("ufunc_lane", _lane_add),
        Probe("ufunc_lane", _lane_number_f32),
        Probe("ufunc_lane", _lane_compare),
        Probe("ufunc_lane", _lane_row),
        Probe("ufunc_lane", _lane_inplace),
        Probe("ufunc_lane", _lane_flags),
    )),
)

# The hooks in bottleneck, which install(bottleneck=False) leaves out.
BOTTLENECK_HOOKS = tuple(hook.name for hook in HOOKS if hook.module == BOTTLENECK_MODULE)

# The hooks that deliver the compress kernels: `x[mask]`, `np.compress` and `np.extract`.
COMPRESS_HOOKS = ("index.get", "function.compress", "function.extract")

# Every kernel, and every loop built into the binding, in the order the report
# lists them. The rows of one operation are adjacent.
KERNELS: tuple[KernelInfo, ...] = (
    KernelInfo("argmax_f32", "np.argmax", "float32", "argmax", ("slot.argmax.f32", "function.argmax", "method.argmax")),
    KernelInfo("argmax_f64", "np.argmax", "float64", "argmax", ("slot.argmax.f64", "function.argmax", "method.argmax")),
    # Loops built into the binding, for rows too narrow for the kernels in
    # np.argmax(m, axis=1) and m.argmax(axis=1); wider rows count under the
    # kernels above.
    KernelInfo("argmax_rows_f32", "argmax(axis=1)", "float32", "argmax", ("function.argmax", "method.argmax"), builtin=True),
    KernelInfo("argmax_rows_f64", "argmax(axis=1)", "float64", "argmax", ("function.argmax", "method.argmax"), builtin=True),
    KernelInfo("argmin_f32", "np.argmin", "float32", "argmin", ("slot.argmin.f32", "function.argmin", "method.argmin")),
    KernelInfo("argmin_f64", "np.argmin", "float64", "argmin", ("slot.argmin.f64", "function.argmin", "method.argmin")),
    KernelInfo("argmin_rows_f32", "argmin(axis=1)", "float32", "argmin", ("function.argmin", "method.argmin"), builtin=True),
    KernelInfo("argmin_rows_f64", "argmin(axis=1)", "float64", "argmin", ("function.argmin", "method.argmin"), builtin=True),
    KernelInfo("clip_f32", "np.clip", "float32", "clip", ("loop.clip.f32", "function.clip", "method.clip")),
    KernelInfo("clip_f64", "np.clip", "float64", "clip", ("loop.clip.f64", "function.clip", "method.clip")),
    # A loop built into the binding, for short arrays.
    KernelInfo("clip_i64", "np.clip", "int64", "clip", ("function.clip", "method.clip"), builtin=True),
    KernelInfo("compress_f32", "x[mask]", "float32", "compress", COMPRESS_HOOKS),
    KernelInfo("compress_f64", "x[mask]", "float64", "compress", COMPRESS_HOOKS),
    # Each also counts the other element types of its width: int8 and bool,
    # and uint16.
    KernelInfo("compress_u8", "x[mask]", "uint8", "compress", COMPRESS_HOOKS),
    KernelInfo("compress_i16", "x[mask]", "int16", "compress", COMPRESS_HOOKS),
    KernelInfo("maskfill_f32", "x[mask] = v", "float32", "maskfill", ("index.set", "function.putmask", "function.copyto")),
    KernelInfo("maskfill_f64", "x[mask] = v", "float64", "maskfill", ("index.set", "function.putmask", "function.copyto")),
    KernelInfo("nonzero_bool", "np.nonzero", "bool", "nonzero", ("method.nonzero", "function.where")),
    # The same call on a C-ordered matrix.
    KernelInfo("nonzero_2d_bool", "np.nonzero, matrix", "bool", "nonzero", ("method.nonzero", "function.where")),
    KernelInfo("cumsum_i64", "np.cumsum", "int64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f32", "np.cumsum", "float32", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f64", "np.cumsum", "float64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_bool", "np.cumsum", "bool", "cumsum", ("method.cumsum",)),
    # A loop built into the binding: the sum down the columns of a matrix.
    KernelInfo("cumsum_rows_f64", "np.cumsum axis=0", "float64", "cumsum", ("method.cumsum", "ufunc.accumulate"),
               builtin=True),
    KernelInfo("cast_u8_f32", "x.astype(float32)", "uint8", "astype", ("method.astype",)),
    KernelInfo("sum_u8", "np.sum", "uint8", "sum", ("function.sum", "method.sum")),
    KernelInfo("sum_f32", "np.sum", "float32", "sum", ("function.sum", "method.sum")),
    KernelInfo("sum_f64", "np.sum", "float64", "sum", ("function.sum", "method.sum")),
    KernelInfo("count_nonzero_i64", "np.count_nonzero", "int64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("count_nonzero_f64", "np.count_nonzero", "float64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("histogram_f64", "np.histogram", "float64", "histogram", ("function.histogram",)),
    KernelInfo("bincount_u8", "np.bincount", "uint8", "bincount", ("function.bincount",)),
    KernelInfo("bincount_i16", "np.bincount", "int16", "bincount", ("function.bincount",)),
    KernelInfo("bincount_i64", "np.bincount", "int64", "bincount", ("function.bincount",)),
    KernelInfo("isin_i64", "np.isin", "int64", "isin", ("function.isin",)),
    KernelInfo("isin_i32", "np.isin", "int32", "isin", ("function.isin",)),
    KernelInfo("searchsorted_f64", "np.searchsorted", "float64", "searchsorted", ("method.searchsorted",)),
    # Indented under the row above: the same call with one argument more, in a
    # column the report keeps to 19 characters.
    KernelInfo("searchsorted_right_f64", "  side='right'", "float64", "searchsorted", ("method.searchsorted",)),
    KernelInfo("searchsorted_i64", "np.searchsorted", "int64", "searchsorted", ("method.searchsorted",)),
    KernelInfo("searchsorted_right_i64", "  side='right'", "int64", "searchsorted", ("method.searchsorted",)),
    KernelInfo("interp_f64", "np.interp", "float64", "interp", ("function.interp",)),
    KernelInfo("where_f64", "np.where(c, a, b)", "float64", "where", ("function.where",)),
    # np.maximum.accumulate does not fit the report's column of 19 characters.
    KernelInfo("cummax_i64", "maximum.accumulate", "int64", "accumulate", ("ufunc.accumulate",)),
    KernelInfo("cumfmax_f64", "fmax.accumulate", "float64", "accumulate", ("ufunc.accumulate",)),
    # Loops built into the binding: accumulations down the columns of a matrix.
    KernelInfo("cumprod_rows_f64", "np.cumprod axis=0", "float64", "accumulate", ("ufunc.accumulate",),
               builtin=True),
    KernelInfo("cummax_rows_f64", "maximum.acc axis=0", "float64", "accumulate", ("ufunc.accumulate",),
               builtin=True),
    KernelInfo("cumfmax_rows_f64", "fmax.acc axis=0", "float64", "accumulate", ("ufunc.accumulate",),
               builtin=True),
    KernelInfo("fmax_reduce_f64", "fmax.reduceat", "float64", "reduceat", ("ufunc.reduceat",)),
    KernelInfo("any_rows_bool", "m.any(axis=1)", "bool", "any", ("method.any", "function.any")),
    KernelInfo("any_cols_bool", "m.any(axis=0)", "bool", "any", ("method.any", "function.any")),
    KernelInfo("unique_i64", "np.unique", "int64", "unique", ("function.unique",)),
    # The stable kinds only: kind='stable', kind='mergesort' and stable=True.
    KernelInfo("argsort_f64", "np.argsort, stable", "float64", "argsort", ("method.argsort",)),
    KernelInfo("argsort_i64", "np.argsort, stable", "int64", "argsort", ("method.argsort",)),
    KernelInfo("argsort_u64", "np.argsort, stable", "uint64", "argsort", ("method.argsort",)),
    # A loop built into the binding, for short arrays.
    KernelInfo("round_f64", "np.round", "float64", "round", ("function.round", "function.around", "method.round"),
               builtin=True),
    KernelInfo("median_f64", "np.median", "float64", "median", ("function.median",), builtin=True),
    # Loops built into the binding for adds and subtracts in place, on ARM64.
    KernelInfo("add_inplace_f64", "x += y", "float64", INPLACE, ("loop.add.f64",),
               unbound_reason=None if ON_ARM64 else INPLACE_ARM64_ONLY, builtin=True),
    KernelInfo("subtract_inplace_f64", "x -= y", "float64", INPLACE, ("loop.subtract.f64",),
               unbound_reason=None if ON_ARM64 else INPLACE_ARM64_ONLY, builtin=True),
    # NumPy's own inner loops, called directly by the ufunc lane for
    # elementwise calls on small arrays: `a + b`, `a > 0`, `np.exp(a)`.
    KernelInfo("ufunc_lane", "elementwise ufuncs", "numeric", "ufunc", ("ufunc.call",), builtin=True,
               implementation=NUMPY_LOOPS),
    # Calls a library sends to bottleneck rather than NumPy: `bn.nanmean(a,
    # axis=0)` on a matrix whose rows' elements are adjacent.
    *(KernelInfo(f"{function}_cols_f64", f"bn.{function}", "float64", f"bn.{function}", (f"bottleneck.{function}",))
      for function in ("nanmean", "nanstd", "nanmin", "nanmax")),
)
