"""What hotkernels installs: the kernels, the hooks that deliver them, and a
self-test for each hook.

A hook is one pointer inside NumPy that the extension module can point at the
binding. This module says which hooks exist, which kernels each one serves,
and how to prove after applying a hook that the kernel really is reached. The
proof is a probe: one small NumPy call with a known answer. `install()` keeps
a hook only if every probe gives the right answer and the kernel's call
counter moved. A NumPy that routes a call differently from the one the hooks
were written for then leaves hotkernels switched off for that call, with
NumPy untouched.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import Callable

import numpy as np

from .environment import BINDING_MODULE

# Kinds of hook, as shown to the user.
TYPE_SLOT = "type slot"
INNER_LOOP = "inner loop"
FUNCTION = "function"
METHOD = "method"
INDEXING = "indexing"

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
# Value the masked-assignment probe writes.
PROBE_FILL = 2.5

# Length of the runs the reduceat probe cuts its array into: the length the
# kernel was timed for.
PROBE_RUN = 16

# Columns of the matrix the any(axis=1) probes reduce: the width the kernel
# was timed for. PROBE_N is a whole number of such rows' worth or more.
PROBE_COLS = 22

# Number of distinct codes the int16 and int64 bincount probes count: their
# element i is i modulo it. PROBE_N is not a multiple of it, so the first
# codes are counted once more than the others, and a kernel that dropped or
# misplaced a code would fail.
PROBE_CODES = 41

# Number of pairs the hypot probe compares, and the seed they are drawn with.
# Algorithms for hypot that differ do so on a fraction of a percent to about
# ten percent of ordinary pairs. The closest one tried, the kernel's own with
# one fused multiply-add in its first sum, differs on 16 of the probe's
# PROBE_HYPOT_ORDINARY_N ordinary pairs, and on none of the others.
PROBE_HYPOT_N = 20000
PROBE_HYPOT_ORDINARY_N = 14000
PROBE_HYPOT_SEED = 96
# Exponents, as powers of two, the hypot probe draws its magnitudes between.
# The close ones span -30 to 30, so that most pairs are computed in full
# rather than returning the larger operand; the spread ones span -200 to 200.
# The large ones lie above 2^511, where the algorithm scales its inputs down,
# and stay below 2^1000 so that no result overflows. The small ones lie below
# 2^-459, where it scales them up, and stay above 2^-1000 so that no result
# is a denormal, whose rounding raises the underflow flag and would send the
# call to NumPy's own loop.
PROBE_HYPOT_CLOSE = (-30, 30)
PROBE_HYPOT_SPREAD = (-200, 200)
PROBE_HYPOT_LARGE = (512, 1000)
PROBE_HYPOT_SMALL = (-1000, -460)
# A quiet NaN with a payload of its own and the sign bit set: the result of
# hypot for a NaN operand is a sum of the operands, which keeps one of them.
PROBE_HYPOT_NAN = np.array([0xFFF8000000001234], dtype=np.uint64).view(np.float64)[0]

# Element count of the sorted array the search probes look in. It differs
# from PROBE_N so that a kernel which confused its two lengths would fail.
PROBE_TABLE_N = 37


@dataclass(frozen=True)
class Probe:
    """One self-test call for one kernel."""

    # Kernel whose call counter must move when `call` runs.
    kernel: str
    # Makes one NumPy call the kernel should take and returns whether the
    # answer was right.
    call: Callable[[], bool]
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


@dataclass(frozen=True)
class KernelInfo:
    """How one kernel appears to a user."""

    # Kernel name, as in `kernels/`.
    name: str
    # The NumPy call the kernel stands in for, as a user writes it.
    numpy_call: str
    # Element type of the arrays it handles, as a NumPy type name.
    elements: str
    # Name of the operation in the binding's counters of calls passed on to
    # NumPy, or None for a kernel without a hook.
    operation: str | None
    # Names of the hooks that deliver the kernel. Empty for one without a hook.
    hooks: tuple[str, ...]
    # Why the kernel has no hook, or None when it has one.
    unbound_reason: str | None = None
    # True for a loop built into the binding for calls on short arrays, which
    # is plain C in the hook files and not a kernel from `kernels/`: the
    # judge never sees it, and `report()` shows it as exact C and does not
    # count it among the kernels.
    builtin: bool = False


def _marked(dtype: type, n: int, index: int, value: float) -> np.ndarray:
    """Return `n` zeros of `dtype` with `value` at `index`."""
    x = np.zeros(n, dtype)
    x[index] = value
    return x


def _argext_method(dtype: type, method: str, value: float) -> bool:
    """Probe the type slot: `x.argmax()` or `x.argmin()` on a long array.

    Args:
        dtype: Element type of the probe array.
        method: "argmax" or "argmin".
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        Whether the method found the marked element.
    """
    return int(getattr(_marked(dtype, PROBE_N, PROBE_INDEX, value), method)()) == PROBE_INDEX


def _argext_function(dtype: type, function: Callable, value: float) -> bool:
    """Probe the function hook: `np.argmax(x)` or `np.argmin(x)` on a short array.

    The array is shorter than the type slot serves, so only the function hook
    can have taken the call.

    Args:
        dtype: Element type of the probe array.
        function: `np.argmax` or `np.argmin`.
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        Whether the function found the marked element.
    """
    return int(function(_marked(dtype, SMALL_PROBE_N, SMALL_PROBE_INDEX, value))) == SMALL_PROBE_INDEX


def _argext_small_method(dtype: type, method: str, value: float) -> bool:
    """Probe the method hook: `x.argmax()` or `x.argmin()` on a short array.

    The array is shorter than the type slot serves, so only the method hook
    can have taken the call.

    Args:
        dtype: Element type of the probe array.
        method: "argmax" or "argmin".
        value: The one non-zero element: positive for argmax, negative for argmin.

    Returns:
        Whether the method found the marked element.
    """
    x = _marked(dtype, SMALL_PROBE_N, SMALL_PROBE_INDEX, value)
    return int(getattr(x, method)()) == SMALL_PROBE_INDEX


def _clip_with(dtype: type, clip: Callable[[np.ndarray, float, float], np.ndarray]) -> bool:
    """Probe a clip hook: values from -1 to 1 clamped to the probe bounds.

    The loop hook takes every call the function and method hooks take, so
    the probe of those two proves that the result is right and that a clip
    kernel ran, not which hook ran it.

    Args:
        dtype: Element type of the probe array.
        clip: Clips its first argument to the bounds given by the other two.

    Returns:
        Whether the result has the array's element type and the right values.
    """
    x = np.linspace(-1.0, 1.0, PROBE_N).astype(dtype)
    result = clip(x, PROBE_LO, PROBE_HI)
    return result.dtype == dtype and bool(np.array_equal(result, np.minimum(np.maximum(x, PROBE_LO), PROBE_HI)))


def _clip_method(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return `x.clip(lo, hi)`."""
    return x.clip(lo, hi)


def _clip(dtype: type) -> bool:
    """Probe the clip loop or the function hook: `np.clip(x, lo, hi)`."""
    return _clip_with(dtype, np.clip)


def _clip_i64_with(clip: Callable[[np.ndarray, int, int], np.ndarray]) -> bool:
    """Probe the int64 path of a clip hook: SMALL_PROBE_N values around zero clamped to PROBE_INT_BOUNDS.

    Args:
        clip: Clips its first argument to the bounds given by the other two.

    Returns:
        Whether the result is int64 and holds the clamped values.
    """
    x = np.arange(SMALL_PROBE_N, dtype=np.int64) - SMALL_PROBE_N // 2
    lo, hi = PROBE_INT_BOUNDS
    result = clip(x, lo, hi)
    return result.dtype == np.int64 and result.tolist() == [min(max(v, lo), hi) for v in x.tolist()]


def _round_with(round_: Callable[[np.ndarray, int], np.ndarray]) -> bool:
    """Probe a round hook: SMALL_PROBE_N multiples of PROBE_ROUND_STEP rounded to PROBE_DECIMALS decimals.

    The expected values are computed in plain Python: `round` of a float to
    an int goes to the even neighbour on a tie, as `rint` does, and the
    products are exact.

    Args:
        round_: Rounds its first argument to the number of decimals given by
            the second.

    Returns:
        Whether the result is float64 and holds the rounded values.
    """
    scale = 10.0 ** PROBE_DECIMALS
    x = (np.arange(SMALL_PROBE_N) - SMALL_PROBE_N // 2) * PROBE_ROUND_STEP / scale
    result = round_(x, PROBE_DECIMALS)
    return result.dtype == np.float64 and result.tolist() == [round(v * scale) / scale for v in x.tolist()]


def _round_method(x: np.ndarray, decimals: int) -> np.ndarray:
    """Return `x.round(decimals)`."""
    return x.round(decimals)


def _median() -> bool:
    """Probe the function hook: `np.median(x)` on SMALL_PROBE_N and SMALL_PROBE_N - 1 halves in descending order.

    The even count's median is the mean of its two middle elements and the
    odd count's its middle element, so a loop that took the wrong element,
    or did not sort, fails. Halves of small whole numbers and their means are
    exact.

    Returns:
        Whether both medians are np.float64 and right.
    """
    answers = []
    for n in (SMALL_PROBE_N, SMALL_PROBE_N - 1):
        x = np.arange(n, 0, -1) * 0.5
        result = np.median(x)
        answers.append(type(result) is np.float64 and float(result) == (n + 1) * 0.25)
    return all(answers)


def _stride_mask() -> np.ndarray:
    """Return a boolean mask that is True at every PROBE_STRIDE-th position."""
    return np.arange(PROBE_N) % PROBE_STRIDE == 0


def _compress(dtype: type) -> bool:
    """Probe the indexing hook: `x[mask]`."""
    x = np.arange(PROBE_N).astype(dtype)
    return bool(np.array_equal(x[_stride_mask()], x[::PROBE_STRIDE]))


def _compress_function(dtype: type, function: Callable) -> bool:
    """Probe a function hook: `np.compress(mask, x)` or `np.extract(mask, x)`.

    Args:
        dtype: Element type of the probe array.
        function: `np.compress` or `np.extract`; both take the mask first.

    Returns:
        Whether the function selected every PROBE_STRIDE-th element.
    """
    x = np.arange(PROBE_N).astype(dtype)
    return bool(np.array_equal(function(_stride_mask(), x), x[::PROBE_STRIDE]))


def _assign(x: np.ndarray, mask: np.ndarray) -> None:
    """Run `x[mask] = PROBE_FILL`."""
    x[mask] = PROBE_FILL


def _putmask(x: np.ndarray, mask: np.ndarray) -> None:
    """Run `np.putmask(x, mask, PROBE_FILL)`."""
    np.putmask(x, mask, PROBE_FILL)


def _copyto(x: np.ndarray, mask: np.ndarray) -> None:
    """Run `np.copyto(x, PROBE_FILL, where=mask)`."""
    np.copyto(x, PROBE_FILL, where=mask)


def _maskfill(assign: Callable[[np.ndarray, np.ndarray], None]) -> bool:
    """Probe a masked assignment: `x[mask] = value` or one of its function forms.

    Args:
        assign: Sets the elements of its first argument where its second is
            True to PROBE_FILL, in place.

    Returns:
        Whether exactly the selected elements were set.
    """
    x = np.arange(PROBE_N, dtype=np.float64)
    expected = x.copy()
    expected[::PROBE_STRIDE] = PROBE_FILL
    assign(x, _stride_mask())
    return bool(np.array_equal(x, expected))


def _nonzero_method(mask: np.ndarray) -> tuple:
    """Return `mask.nonzero()`."""
    return mask.nonzero()


def _nonzero(function: Callable[[np.ndarray], tuple]) -> bool:
    """Probe `mask.nonzero()` or `np.where(mask)`.

    Args:
        function: Takes the mask and returns NumPy's tuple of index arrays.

    Returns:
        Whether the one index array holds every PROBE_STRIDE-th position.
    """
    (indices,) = function(_stride_mask())
    return bool(np.array_equal(indices, np.arange(0, PROBE_N, PROBE_STRIDE)))


def _where() -> bool:
    """Probe the function hook: `np.where(mask, a, b)` on float64, with a mask True at every PROBE_STRIDE-th position.

    The two arrays differ at every position, so an element taken from the
    wrong one fails the probe.
    """
    a = np.arange(PROBE_N, dtype=np.float64)
    b = -a - 1.0
    expected = b.copy()
    expected[::PROBE_STRIDE] = a[::PROBE_STRIDE]
    result = np.where(_stride_mask(), a, b)
    return result.dtype == np.float64 and bool(np.array_equal(result, expected))


def _hypot_pairs() -> tuple[np.ndarray, np.ndarray]:
    """Return the pairs the hypot probe compares, in a fixed order.

    First PROBE_HYPOT_ORDINARY_N ordinary pairs, where algorithms for hypot
    disagree most: half standard normal, half with magnitudes from 2^-30 to
    2^30. They come first and on their own, so that the kernel computes them
    four at a time on its vector path, the path that carries nearly all of a
    real call. Then, shuffled together, one block for each kind of operand
    the algorithm treats on a path of its own: magnitudes from 2^-200 to
    2^200, where one operand is often negligible beside the other; both
    operands above 2^511; one of them above 2^511; both below 2^-459; a
    denormal beside a value near 2^-1000 and beside 1.0; a zero of either
    sign beside anything; an infinity beside a finite value, a NaN or another
    infinity; a quiet NaN beside a finite value or another NaN; and a value
    near 2^-300 beside an ordinary one. None of them raises a floating-point
    flag in the C library's hypot (no result overflows or is a denormal, and
    no NaN is signalling), so the kernel takes the call when it is right.

    Returns:
        The arrays a and b, PROBE_HYPOT_N float64 elements each.
    """
    rng = np.random.default_rng(PROBE_HYPOT_SEED)

    def spread(low: int, high: int, count: int) -> np.ndarray:
        """Return values of either sign with exponents drawn from low to high.

        Args:
            low: Smallest power of two of a magnitude.
            high: Largest power of two of a magnitude.
            count: Number of values.

        Returns:
            A float64 array of `count` values.
        """
        signs = rng.choice([-1.0, 1.0], count)
        return np.ldexp(signs * rng.uniform(0.5, 1.0, count), rng.integers(low, high + 1, count))

    half = PROBE_HYPOT_ORDINARY_N // 2
    a = [rng.standard_normal(half), spread(*PROBE_HYPOT_CLOSE, half)]
    b = [rng.standard_normal(half), spread(*PROBE_HYPOT_CLOSE, half)]
    blocks = 16
    k = (PROBE_HYPOT_N - PROBE_HYPOT_ORDINARY_N) // blocks
    denormals = np.ldexp(rng.uniform(0.0, 1.0, k), -1022)
    pairs = (
        (spread(*PROBE_HYPOT_SPREAD, k), spread(*PROBE_HYPOT_SPREAD, k)),
        (spread(*PROBE_HYPOT_LARGE, k), spread(*PROBE_HYPOT_LARGE, k)),
        (spread(*PROBE_HYPOT_LARGE, k), spread(*PROBE_HYPOT_SPREAD, k)),
        (spread(*PROBE_HYPOT_SMALL, k), spread(*PROBE_HYPOT_SMALL, k)),
        (denormals, spread(PROBE_HYPOT_SMALL[0], PROBE_HYPOT_SMALL[0] + 20, k)),
        (denormals, np.ones(k)),
        (np.zeros(k), spread(*PROBE_HYPOT_SPREAD, k)),
        (-np.zeros(k), spread(*PROBE_HYPOT_LARGE, k)),
        (np.zeros(k), denormals),
        (np.zeros(k), -np.zeros(k)),
        (np.full(k, np.inf), spread(*PROBE_HYPOT_SPREAD, k)),
        (spread(*PROBE_HYPOT_SPREAD, k), np.full(k, -np.inf)),
        (np.full(k, np.inf), np.full(k, PROBE_HYPOT_NAN)),
        (np.full(k, PROBE_HYPOT_NAN), spread(*PROBE_HYPOT_SPREAD, k)),
        (np.full(k, np.nan), np.full(k, PROBE_HYPOT_NAN)),
        (spread(*PROBE_HYPOT_SPREAD, k), np.ldexp(spread(-10, 10, k), -300)),
    )
    # Shuffled, so that the kernel meets these operands inside groups of
    # other kinds and not only in groups of their own.
    order = rng.permutation(blocks * k)
    a.append(np.concatenate([x for x, _ in pairs])[order])
    b.append(np.concatenate([y for _, y in pairs])[order])
    return np.concatenate(a), np.concatenate(b)


def _hypot() -> bool:
    """Probe the hypot loop: the kernel must give the C library's bits on this machine.

    The kernel computes one particular algorithm for hypot, and NumPy calls
    whatever the C library has, so this probe is what decides whether the
    hook may stay. It computes np.hypot on the pairs of `_hypot_pairs` twice:
    first on strided views of them, a call the hook leaves to NumPy's own
    loop and so to the C library, then on contiguous arrays, which the kernel
    takes. It passes only if the kernel's counter did not move for the first
    call, moved for the second, and every result has the same bits, NaN
    payloads included. A call that raised a floating-point flag would go to
    NumPy's loop and leave the counter where it was, so it cannot pass by
    that route either; both calls run under np.errstate(all="raise"), which
    turns such a flag into an exception, and an exception fails the probe.

    Returns:
        True when the kernel took the contiguous call and matched the C
        library on every pair.
    """
    binding = importlib.import_module(BINDING_MODULE)
    a, b = _hypot_pairs()
    # Every second element of arrays twice as long: a step of 16 bytes.
    wide_a = np.zeros(2 * PROBE_HYPOT_N)
    wide_b = np.zeros(2 * PROBE_HYPOT_N)
    wide_a[::2] = a
    wide_b[::2] = b
    before = binding.counters()[0]["hypot_f64"]
    with np.errstate(all="raise"):
        reference = np.hypot(wide_a[::2], wide_b[::2])
        if binding.counters()[0]["hypot_f64"] != before:
            return False
        result = np.hypot(a, b)
    if binding.counters()[0]["hypot_f64"] == before:
        return False
    return result.dtype == np.float64 and bool(np.array_equal(result.view(np.uint64), reference.view(np.uint64)))


def _cumsum(dtype: type) -> bool:
    """Probe the method hook: `x.cumsum()` on ones, whose running sum is exact in every type."""
    return bool(np.array_equal(np.ones(PROBE_N, dtype).cumsum(), np.arange(1, PROBE_N + 1).astype(dtype)))


def _cumsum_bool() -> bool:
    """Probe the method hook: `mask.cumsum()` on a mask that is True at every PROBE_STRIDE-th position."""
    mask = _stride_mask()
    result = mask.cumsum()
    return result.dtype == np.intp and bool(np.array_equal(result, np.arange(PROBE_N) // PROBE_STRIDE + 1))


def _running_max_probe() -> list[float]:
    """Return the values of the running-maximum probes, as Python floats.

    0, 1, -2, 3, -4, ...: the odd elements are each a new maximum and the even
    ones never are, so a kernel that kept the wrong one fails. Every value is
    a whole number below 2^53, exact in both int64 and float64.
    """
    return [float(i if i % 2 else -i) for i in range(PROBE_N)]


def _running_max(values: list[float]) -> list[float]:
    """Return the running maximum of `values` in plain Python, a NaN never replacing the value so far.

    Args:
        values: The elements; the first is not NaN.

    Returns:
        Element i is the largest of the elements up to i that are not NaN.
    """
    return list(accumulate(values, lambda so_far, v: v if v > so_far else so_far))


def _cummax(ufunc: np.ufunc) -> bool:
    """Probe the ufunc method hook: `ufunc.accumulate(x)` on int64, for np.maximum or np.fmax.

    Args:
        ufunc: `np.maximum` or `np.fmax`; on integers both are the running maximum.

    Returns:
        Whether the result is int64 and holds the running maximum.
    """
    values = _running_max_probe()
    result = ufunc.accumulate(np.array(values, np.int64))
    return result.dtype == np.int64 and result.tolist() == [int(v) for v in _running_max(values)]


def _cumfmax() -> bool:
    """Probe the ufunc method hook: `np.fmax.accumulate(x)` on float64 with a NaN at every PROBE_STRIDE-th position after the first.

    A kernel that let a NaN through, or that NumPy's NaN-propagating
    `np.maximum` answered, fails.
    """
    values = _running_max_probe()
    for i in range(PROBE_STRIDE, PROBE_N, PROBE_STRIDE):
        values[i] = float("nan")
    result = np.fmax.accumulate(np.array(values, np.float64))
    return result.dtype == np.float64 and result.tolist() == _running_max(values)


def _fmax_reduceat() -> bool:
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
    ends = [*starts[1:], PROBE_N]
    expected = [max(v for v in values[start:max(end, start + 1)] if v == v) for start, end in zip(starts, ends)]
    result = np.fmax.reduceat(np.array(values, np.float64), np.array(starts, np.intp))
    return result.dtype == np.float64 and result.tolist() == expected


def _any_rows(reduce: Callable[[np.ndarray], np.ndarray]) -> bool:
    """Probe an any(axis=1) hook on a boolean matrix of PROBE_COLS columns.

    Every PROBE_STRIDE-th row holds one True, at a column that moves from row
    to row; the others hold none. A kernel that reduced the wrong elements or
    along the wrong axis fails.

    Args:
        reduce: Takes the matrix and returns NumPy's `m.any(axis=1)` for it.

    Returns:
        Whether the result is a one-dimensional boolean array with True at
        exactly the rows that hold one.
    """
    rows = PROBE_N // PROBE_COLS
    m = np.zeros((rows, PROBE_COLS), np.bool_)
    marked = np.arange(0, rows, PROBE_STRIDE)
    m[marked, marked % PROBE_COLS] = True
    result = reduce(m)
    return result.dtype == np.bool_ and bool(np.array_equal(result, np.arange(rows) % PROBE_STRIDE == 0))


def _any_method(m: np.ndarray) -> np.ndarray:
    """Return `m.any(axis=1)`."""
    return m.any(axis=1)


def _any_function(m: np.ndarray) -> np.ndarray:
    """Return `np.any(m, axis=1)`."""
    return np.any(m, axis=1)


def _astype() -> bool:
    """Probe the method hook: `x.astype(np.float32)` on bytes."""
    result = np.full(PROBE_N, PROBE_BYTE, np.uint8).astype(np.float32)
    return result.dtype == np.float32 and bool(np.all(result == PROBE_BYTE))


def _sum_method() -> bool:
    """Probe the method hook: `x.sum()` on bytes."""
    return int(np.full(PROBE_N, PROBE_BYTE, np.uint8).sum()) == PROBE_N * PROBE_BYTE


def _sum_function() -> bool:
    """Probe the function hook: `np.sum(x)` on bytes."""
    return int(np.sum(np.full(PROBE_N, PROBE_BYTE, np.uint8))) == PROBE_N * PROBE_BYTE


def _count_nonzero(dtype: type) -> bool:
    """Probe the function hook: `np.count_nonzero(x)`."""
    return int(np.count_nonzero(_stride_mask().astype(dtype))) == len(range(0, PROBE_N, PROBE_STRIDE))


def _histogram() -> bool:
    """Probe the function hook: `np.histogram` on values that fill each bin equally."""
    x = (np.arange(PROBE_N) % PROBE_BINS) + 0.5
    counts, edges = np.histogram(x, PROBE_BINS, PROBE_RANGE)
    return bool(np.all(counts == PROBE_N // PROBE_BINS)) and len(edges) == PROBE_BINS + 1


def _bincount() -> bool:
    """Probe the function hook: `np.bincount` on equal bytes."""
    counts = np.bincount(np.full(PROBE_N, PROBE_BYTE, np.uint8))
    return len(counts) == PROBE_BYTE + 1 and int(counts[PROBE_BYTE]) == PROBE_N


def _bincount_codes(dtype: type) -> bool:
    """Probe the function hook: `np.bincount` on int16 or int64 codes that cycle through PROBE_CODES values.

    Args:
        dtype: np.int16 or np.int64.

    Returns:
        Whether the result is an intp array of PROBE_CODES counts, each the
        number of positions below PROBE_N that leave its code modulo
        PROBE_CODES.
    """
    counts = np.bincount(np.arange(PROBE_N, dtype=dtype) % PROBE_CODES)
    expected = [len(range(code, PROBE_N, PROBE_CODES)) for code in range(PROBE_CODES)]
    return counts.dtype == np.intp and counts.tolist() == expected


def _isin() -> bool:
    """Probe the function hook: `np.isin` with two test values."""
    x = np.arange(PROBE_N, dtype=np.int64)
    result = np.isin(x, [PROBE_INDEX, SMALL_PROBE_INDEX])
    return int(result.sum()) == 2 and bool(result[PROBE_INDEX]) and bool(result[SMALL_PROBE_INDEX])


def _searchsorted() -> bool:
    """Probe the method hook: `np.searchsorted` for needles between whole numbers."""
    table = np.arange(PROBE_TABLE_N, dtype=np.float64)
    needles = (np.arange(PROBE_N) % PROBE_TABLE_N) + 0.5
    expected = (np.arange(PROBE_N) % PROBE_TABLE_N) + 1
    return bool(np.array_equal(np.searchsorted(table, needles), expected))


def _searchsorted_right() -> bool:
    """Probe the method hook: `np.searchsorted` with `side='right'` for needles that are elements.

    Every needle equals one element, so the left side's answer is one less at
    each of them: a call that reached the wrong kernel fails.
    """
    table = np.arange(PROBE_TABLE_N, dtype=np.float64)
    needles = (np.arange(PROBE_N) % PROBE_TABLE_N).astype(np.float64)
    expected = (np.arange(PROBE_N) % PROBE_TABLE_N) + 1
    return bool(np.array_equal(np.searchsorted(table, needles, side="right"), expected))


def _interp() -> bool:
    """Probe the function hook: `np.interp` through points on the line y = 2x.

    Halves and doubles of small whole numbers are exact, so the answer is too.
    """
    points = np.arange(PROBE_TABLE_N, dtype=np.float64)
    queries = (np.arange(PROBE_N) % (PROBE_TABLE_N - 1)) + 0.5
    return bool(np.array_equal(np.interp(queries, points, 2.0 * points), 2.0 * queries))


# Every hook, in the order install() applies them: the deepest first.
HOOKS: tuple[Hook, ...] = (
    Hook("slot.argmax.f32", TYPE_SLOT, (Probe("argmax_f32", partial(_argext_method, np.float32, "argmax", 1.0)),)),
    Hook("slot.argmax.f64", TYPE_SLOT, (Probe("argmax_f64", partial(_argext_method, np.float64, "argmax", 1.0)),)),
    Hook("slot.argmin.f32", TYPE_SLOT, (Probe("argmin_f32", partial(_argext_method, np.float32, "argmin", -1.0)),)),
    Hook("slot.argmin.f64", TYPE_SLOT, (Probe("argmin_f64", partial(_argext_method, np.float64, "argmin", -1.0)),)),
    Hook("loop.clip.f32", INNER_LOOP, (Probe("clip_f32", partial(_clip, np.float32)),)),
    Hook("loop.clip.f64", INNER_LOOP, (Probe("clip_f64", partial(_clip, np.float64)),)),
    Hook("loop.hypot.f64", INNER_LOOP, (Probe("hypot_f64", _hypot),)),
    Hook("function.argmax", FUNCTION, (
        Probe("argmax_f32", partial(_argext_function, np.float32, np.argmax, 1.0)),
        Probe("argmax_f64", partial(_argext_function, np.float64, np.argmax, 1.0)),
    )),
    Hook("function.argmin", FUNCTION, (
        Probe("argmin_f32", partial(_argext_function, np.float32, np.argmin, -1.0)),
        Probe("argmin_f64", partial(_argext_function, np.float64, np.argmin, -1.0)),
    )),
    Hook("function.count_nonzero", FUNCTION, (
        Probe("count_nonzero_i64", partial(_count_nonzero, np.int64)),
        Probe("count_nonzero_f64", partial(_count_nonzero, np.float64)),
    )),
    Hook("function.sum", FUNCTION, (Probe("sum_u8", _sum_function),)),
    Hook("function.histogram", FUNCTION, (Probe("histogram_f64", _histogram),)),
    Hook("function.bincount", FUNCTION, (
        Probe("bincount_u8", _bincount),
        Probe("bincount_i16", partial(_bincount_codes, np.int16)),
        Probe("bincount_i64", partial(_bincount_codes, np.int64)),
    )),
    Hook("function.isin", FUNCTION, (Probe("isin_i64", _isin),)),
    Hook("function.interp", FUNCTION, (Probe("interp_f64", _interp),)),
    Hook("function.compress", FUNCTION, (
        Probe("compress_f32", partial(_compress_function, np.float32, np.compress)),
        Probe("compress_f64", partial(_compress_function, np.float64, np.compress)),
    )),
    Hook("function.extract", FUNCTION, (
        Probe("compress_f32", partial(_compress_function, np.float32, np.extract)),
        Probe("compress_f64", partial(_compress_function, np.float64, np.extract)),
    )),
    Hook("function.putmask", FUNCTION, (Probe("maskfill_f64", partial(_maskfill, _putmask)),)),
    Hook("function.copyto", FUNCTION, (Probe("maskfill_f64", partial(_maskfill, _copyto)),)),
    Hook("function.where", FUNCTION, (
        Probe("nonzero_bool", partial(_nonzero, np.where)),
        Probe("where_f64", _where),
    )),
    Hook("function.clip", FUNCTION, (
        Probe("clip_f32", partial(_clip, np.float32)),
        Probe("clip_f64", partial(_clip, np.float64)),
        Probe("clip_i64", partial(_clip_i64_with, np.clip)),
    )),
    Hook("function.any", FUNCTION, (Probe("any_rows_bool", partial(_any_rows, _any_function)),)),
    Hook("function.round", FUNCTION, (Probe("round_f64", partial(_round_with, np.round)),)),
    Hook("function.around", FUNCTION, (Probe("round_f64", partial(_round_with, np.around)),)),
    Hook("function.median", FUNCTION, (Probe("median_f64", _median),)),
    Hook("method.astype", METHOD, (Probe("cast_u8_f32", _astype),)),
    Hook("method.sum", METHOD, (Probe("sum_u8", _sum_method),)),
    Hook("method.nonzero", METHOD, (Probe("nonzero_bool", partial(_nonzero, _nonzero_method)),)),
    Hook("method.cumsum", METHOD, (
        Probe("cumsum_i64", partial(_cumsum, np.int64)),
        Probe("cumsum_f32", partial(_cumsum, np.float32), inexact=True),
        Probe("cumsum_f64", partial(_cumsum, np.float64), inexact=True),
        Probe("cumsum_bool", _cumsum_bool),
    )),
    Hook("method.searchsorted", METHOD, (
        Probe("searchsorted_f64", _searchsorted),
        Probe("searchsorted_right_f64", _searchsorted_right),
    )),
    Hook("method.argmax", METHOD, (
        Probe("argmax_f32", partial(_argext_small_method, np.float32, "argmax", 1.0)),
        Probe("argmax_f64", partial(_argext_small_method, np.float64, "argmax", 1.0)),
    )),
    Hook("method.argmin", METHOD, (
        Probe("argmin_f32", partial(_argext_small_method, np.float32, "argmin", -1.0)),
        Probe("argmin_f64", partial(_argext_small_method, np.float64, "argmin", -1.0)),
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
    )),
    Hook("ufunc.reduceat", METHOD, (Probe("fmax_reduce_f64", _fmax_reduceat),)),
    Hook("method.any", METHOD, (Probe("any_rows_bool", partial(_any_rows, _any_method)),)),
    Hook("method.round", METHOD, (Probe("round_f64", partial(_round_with, _round_method)),)),
    Hook("index.get", INDEXING, (
        Probe("compress_f32", partial(_compress, np.float32)),
        Probe("compress_f64", partial(_compress, np.float64)),
    )),
    Hook("index.set", INDEXING, (Probe("maskfill_f64", partial(_maskfill, _assign)),)),
)

# The hooks that deliver the compress kernels: `x[mask]`, `np.compress` and `np.extract`.
COMPRESS_HOOKS = ("index.get", "function.compress", "function.extract")

# Every kernel, and every loop built into the binding, in the order the report
# lists them. The rows of one operation are adjacent.
KERNELS: tuple[KernelInfo, ...] = (
    KernelInfo("argmax_f32", "np.argmax", "float32", "argmax", ("slot.argmax.f32", "function.argmax", "method.argmax")),
    KernelInfo("argmax_f64", "np.argmax", "float64", "argmax", ("slot.argmax.f64", "function.argmax", "method.argmax")),
    KernelInfo("argmin_f32", "np.argmin", "float32", "argmin", ("slot.argmin.f32", "function.argmin", "method.argmin")),
    KernelInfo("argmin_f64", "np.argmin", "float64", "argmin", ("slot.argmin.f64", "function.argmin", "method.argmin")),
    KernelInfo("clip_f32", "np.clip", "float32", "clip", ("loop.clip.f32", "function.clip", "method.clip")),
    KernelInfo("clip_f64", "np.clip", "float64", "clip", ("loop.clip.f64", "function.clip", "method.clip")),
    # A loop built into the binding, for short arrays.
    KernelInfo("clip_i64", "np.clip", "int64", "clip", ("function.clip", "method.clip"), builtin=True),
    KernelInfo("compress_f32", "x[mask]", "float32", "compress", COMPRESS_HOOKS),
    KernelInfo("compress_f64", "x[mask]", "float64", "compress", COMPRESS_HOOKS),
    KernelInfo("maskfill_f64", "x[mask] = v", "float64", "maskfill", ("index.set", "function.putmask", "function.copyto")),
    KernelInfo("nonzero_bool", "np.nonzero", "bool", "nonzero", ("method.nonzero", "function.where")),
    KernelInfo("cumsum_i64", "np.cumsum", "int64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f32", "np.cumsum", "float32", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_f64", "np.cumsum", "float64", "cumsum", ("method.cumsum",)),
    KernelInfo("cumsum_bool", "np.cumsum", "bool", "cumsum", ("method.cumsum",)),
    KernelInfo("cast_u8_f32", "x.astype(float32)", "uint8", "astype", ("method.astype",)),
    KernelInfo("sum_u8", "np.sum", "uint8", "sum", ("function.sum", "method.sum")),
    KernelInfo("count_nonzero_i64", "np.count_nonzero", "int64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("count_nonzero_f64", "np.count_nonzero", "float64", "count_nonzero", ("function.count_nonzero",)),
    KernelInfo("histogram_f64", "np.histogram", "float64", "histogram", ("function.histogram",)),
    KernelInfo("bincount_u8", "np.bincount", "uint8", "bincount", ("function.bincount",)),
    KernelInfo("bincount_i16", "np.bincount", "int16", "bincount", ("function.bincount",)),
    KernelInfo("bincount_i64", "np.bincount", "int64", "bincount", ("function.bincount",)),
    KernelInfo("isin_i64", "np.isin", "int64", "isin", ("function.isin",)),
    KernelInfo("searchsorted_f64", "np.searchsorted", "float64", "searchsorted", ("method.searchsorted",)),
    # Indented under the row above: the same call with one argument more, in a
    # column the report keeps to 19 characters.
    KernelInfo("searchsorted_right_f64", "  side='right'", "float64", "searchsorted", ("method.searchsorted",)),
    KernelInfo("interp_f64", "np.interp", "float64", "interp", ("function.interp",)),
    KernelInfo("where_f64", "np.where(c, a, b)", "float64", "where", ("function.where",)),
    # np.maximum.accumulate does not fit the report's column of 19 characters.
    KernelInfo("cummax_i64", "maximum.accumulate", "int64", "accumulate", ("ufunc.accumulate",)),
    KernelInfo("cumfmax_f64", "fmax.accumulate", "float64", "accumulate", ("ufunc.accumulate",)),
    KernelInfo("fmax_reduce_f64", "fmax.reduceat", "float64", "reduceat", ("ufunc.reduceat",)),
    KernelInfo("any_rows_bool", "m.any(axis=1)", "bool", "any", ("method.any", "function.any")),
    KernelInfo("hypot_f64", "np.hypot", "float64", "hypot", ("loop.hypot.f64",)),
    # A loop built into the binding, for short arrays.
    KernelInfo("round_f64", "np.round", "float64", "round", ("function.round", "function.around", "method.round"),
               builtin=True),
    KernelInfo("median_f64", "np.median", "float64", "median", ("function.median",), builtin=True),
)
