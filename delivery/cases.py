"""The calls `delivery check` makes: every hooked operation, across element
types, sizes, memory layouts, array kinds, argument spellings and values.

A case is one Python call. It says how to build fresh arguments, how to make
the call, and whether a kernel should take it. The binding decides for every
call whether a kernel's conditions hold, so a case that must fall back to
NumPy is as much a test as one that must not.

Values come from the judge's value classes (`judge/cases.py`), so the binding
sees the same NaNs, infinities, signed zeros and ties the kernels were judged
on. Scalar arguments come from each kernel's own spec.
"""
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import itertools
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from judge import cases as judge_cases
from judge.spec import DEFAULT_SCALARS, Kernel, load_spec, np_dtype

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"

# Expectation meaning "a kernel may or may not take this call". NumPy
# normalises some inputs before a hook sees them (it copies a strided array
# before argmax, for example), and whether it does is NumPy's business. The
# results must match either way.
ANY = "*"

# Seed of all generated data. A case builds the same arguments every time.
SEED = 1

# Sizes for the sweep over values: every power of two a kernel changes path
# at, and its neighbours, up to a size no kernel treats as small.
VALUE_SIZES = (1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129,
               255, 256, 257, 511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 2049,
               4096, 4097, 8191, 8192, 8193, 20000, 100003)
# Sizes for the sweep over layouts and spellings: one below every threshold
# in the binding, one in the middle, and one above the size from which a hook
# releases the interpreter lock.
LAYOUT_SIZES = (16, 1000, 20000)
# A few sizes for sweeps that multiply by many scalar values.
FEW_SIZES = (1, 16, 100, 1000, 8193)
# Columns of the two-dimensional layouts.
COLUMNS = 2
# Bytes other than 1 that a boolean array can hold for True. NumPy accepts
# them; the mask kernels must never see them.
BAD_MASK_BYTES = (2, 255)
# Element count of the one-off cases that need a single ordinary size.
ORDINARY_N = 1000
# Shape of the three-dimensional layout of ORDINARY_N elements.
CUBE_SHAPE = (-1, 5, 4)
# Element count of a mask whose only bad byte is its last: more than two of
# the blocks in which the binding checks a mask before summing it.
SUM_BOOL_LATE_N = 40000
# Every how many bytes the self-overlapping mask case sets one to True.
OVERLAP_EVERY = 3

# Value classes whose sums can overflow, underflow or meet an infinity. A
# float cumsum kernel drops its output for NumPy's when that happens.
MAY_RAISE = frozenset({"denormal", "extremes", "inf_mix", "posinf_at", "neginf_at"})

# How a spelling of a call relates to its kernel.
TAKES = "takes"      # a plain array must be handled by the kernel
NEVER = "never"      # the kernel must not handle it, whatever the array
MAYBE = "maybe"      # NumPy decides; only the results are compared


@dataclass(frozen=True)
class Case:
    """One call, made once on plain NumPy and once with hotkernels installed."""

    # Operation the call belongs to: a key of the binding's pass counters.
    operation: str
    # What the call is, in enough words to find it again.
    label: str
    # Builds the arguments. Called afresh for each of the two runs and must
    # return equal arguments both times.
    make: Callable[[], tuple]
    # Makes the call on the built arguments and returns its result.
    call: Callable[..., object]
    # Name of the kernel that must take the call, None when none of the
    # operation's kernels may, or ANY.
    expect: str | None
    # Spec of the kernel when its result is compared within the spec's
    # tolerance; None for a bitwise comparison.
    tolerance: Kernel | None = None


@dataclass(frozen=True)
class Variant:
    """One way an array can be laid out or wrapped."""

    # Short name used in case labels.
    name: str
    # Builds the variant from a contiguous one-dimensional array.
    build: Callable[[np.ndarray], object]
    # Whether the result is what a kernel needs: exactly an ndarray, native
    # byte order, C-contiguous, at least one dimension.
    plain: bool
    # Number of dimensions of what `build` returns.
    dims: int = 1
    # Whether the result can be written to.
    writeable: bool = True
    # False for a wrapper whose own methods decide what reaches NumPy, so
    # that no expectation about kernels can be stated for it.
    predictable: bool = True
    # Smallest element count the variant can be built from.
    min_n: int = 0
    # False for a variant that is the same as the contiguous one when
    # elements are one byte wide.
    for_single_bytes: bool = True


@dataclass(frozen=True)
class Spelling:
    """One way of writing a call to an operation, taking the array(s) it works on."""

    # The call as text, for labels.
    text: str
    # Makes the call.
    call: Callable[..., object]
    # TAKES, NEVER or MAYBE.
    kind: str = TAKES
    # Element count from which the kernel takes the call; below it NumPy must.
    min_n: int = 0
    # True when NumPy may hand the hook a normalised copy of an array that is
    # not plain, so that such an array may still reach the kernel.
    normalises: bool = False
    # Whether a contiguous array of two or more dimensions is taken too.
    any_shape: bool = True
    # Whether the sweep over every value class at every size runs this
    # spelling. False for a spelling that reaches the same code as another
    # one after its arguments are read, which the layouts still cover.
    sweep_values: bool = True


class _Subclass(np.ndarray):
    """A do-nothing subclass of ndarray. The binding must leave it to NumPy."""


# Files behind the memory-mapped variants. Removed when the process exits.
_MEMMAP_DIR = tempfile.TemporaryDirectory(prefix="delivery-memmap-")
atexit.register(_MEMMAP_DIR.cleanup)
_memmap_counter = itertools.count()


def _strided(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values as every second element of an array twice as long."""
    return np.repeat(a, 2)[::2]


def _reversed(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values in order, stored back to front with a negative step."""
    return a[::-1].copy()[::-1]


def _misaligned(a: np.ndarray) -> np.ndarray:
    """Return a writable copy of `a` that starts one byte off any alignment."""
    buffer = bytearray(1 + a.nbytes)
    buffer[1:] = a.tobytes()
    return np.frombuffer(buffer, dtype=a.dtype, count=len(a), offset=1)


def _readonly(a: np.ndarray) -> np.ndarray:
    """Return a copy of `a` that refuses writes."""
    b = a.copy()
    b.flags.writeable = False
    return b


def _byteswapped(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values stored in the other byte order."""
    return a.astype(a.dtype.newbyteorder())


def _memmap(a: np.ndarray) -> np.ndarray:
    """Return `a`'s values in a memory-mapped file."""
    path = Path(_MEMMAP_DIR.name) / f"{next(_memmap_counter)}.bin"
    mapped = np.memmap(path, dtype=a.dtype, mode="w+", shape=a.shape)
    mapped[:] = a
    return mapped


def _rows(a: np.ndarray) -> np.ndarray:
    """Return the first whole rows of `a` as a C-ordered two-dimensional array."""
    return a[: len(a) // COLUMNS * COLUMNS].reshape(-1, COLUMNS).copy()


VARIANTS: tuple[Variant, ...] = (
    Variant("contiguous", np.copy, plain=True),
    Variant("strided", _strided, plain=False),
    Variant("reversed", _reversed, plain=False),
    Variant("misaligned", _misaligned, plain=True),
    Variant("readonly", _readonly, plain=True, writeable=False),
    Variant("byteswapped", _byteswapped, plain=False, for_single_bytes=False),
    Variant("subclass", lambda a: a.copy().view(_Subclass), plain=False),
    Variant("masked_array", lambda a: np.ma.array(a.copy()), plain=False, predictable=False),
    Variant("memmap", _memmap, plain=False, min_n=1),
    Variant("list", lambda a: a.tolist(), plain=False),
    Variant("rows", _rows, plain=True, dims=2, min_n=2 * COLUMNS),
    Variant("fortran_rows", lambda a: np.asfortranarray(_rows(a)), plain=False, dims=2, min_n=2 * COLUMNS),
    Variant("zero_d", lambda a: np.array(a[0]), plain=False, dims=0, min_n=1),
)
CONTIGUOUS = VARIANTS[0]


def spec(kernel: str) -> Kernel:
    """Load the spec of a kernel by name."""
    return load_spec(KERNELS_DIR / kernel)


def values(dtype: str, n: int, cls: str = judge_cases.RANDOM, position: int = 0, arg: str = "x") -> np.ndarray:
    """Build one array from a value class of the judge.

    Args:
        dtype: Element type as a spec name, such as "f64".
        n: Element count.
        cls: Value class name, such as "nan_some".
        position: Index of the interesting element, for positional classes.
        arg: Argument name. Different names give different random data.

    Returns:
        A contiguous one-dimensional array. The same arguments always give
        the same array.
    """
    base = judge_cases.base(dtype, n, SEED, arg)
    return judge_cases.make(dtype, cls, position % n if n else 0, n, SEED, arg, base)


def value_arrays(dtype: str, sizes: tuple[int, ...] = VALUE_SIZES, arg: str = "x") -> Iterator[tuple[int, str, str, Callable[[], np.ndarray]]]:
    """Yield every value class of an element type at every size.

    Args:
        dtype: Element type as a spec name.
        sizes: Element counts to build.
        arg: Argument name, used in labels and to vary the random data.

    Yields:
        (element count, value class, label, function that builds the array).
        Positional classes are built with their element first, in the middle
        and last.
    """
    plain, positional = judge_cases.classes(dtype)
    for n in sizes:
        for cls in plain:
            yield n, cls, f"n={n} {arg}={cls}", partial(values, dtype, n, cls, 0, arg)
        for cls in positional:
            for position in sorted({0, n // 2, n - 1}):
                yield n, cls, f"n={n} {arg}={cls}@{position}", partial(values, dtype, n, cls, position, arg)


def variants_for(dtype: str, n: int) -> Iterator[Variant]:
    """Yield the variants that can be built from `n` elements of `dtype`."""
    single_bytes = np_dtype(dtype).itemsize == 1
    for variant in VARIANTS:
        if n >= variant.min_n and (variant.for_single_bytes or not single_bytes):
            yield variant


def _built(dtype: str, n: int, variant: Variant, arg: str = "x") -> tuple:
    """Build the one argument of a unary case: random values in a variant."""
    return (variant.build(values(dtype, n, arg=arg)),)


def _made(maker: Callable[[], np.ndarray]) -> tuple:
    """Build the one argument of a unary case from an array maker."""
    return (maker(),)


def expectation(kernel: str | None, spelling: Spelling, variant: Variant, n: int) -> str | None:
    """Say whether a kernel should take a call.

    Args:
        kernel: The kernel for the array's element type, or None if the
            operation has none for it.
        spelling: How the call is written.
        variant: How the array is laid out.
        n: Element count.

    Returns:
        The kernel name, None, or ANY.
    """
    # A list has no element type of its own: NumPy converts it, and the
    # result may be an array some kernel takes.
    if variant.name == "list" and spelling.normalises and spelling.kind != NEVER:
        return ANY
    if kernel is None or spelling.kind == NEVER:
        return None
    if spelling.kind == MAYBE or not variant.predictable:
        return ANY
    shape_ok = variant.dims == 1 or (variant.dims > 1 and spelling.any_shape)
    if variant.plain and shape_ok:
        return kernel if n >= spelling.min_n else None
    # NumPy flags every empty array as contiguous, whatever its strides say.
    return ANY if spelling.normalises or n == 0 else None


def unary_cases(operation: str, kernels: dict[str, str], neighbours: tuple[str, ...],
                spellings: tuple[Spelling, ...], empty_takes: bool = True,
                tolerance: bool = False) -> Iterator[Case]:
    """Yield the cases of an operation that takes one array.

    Three sweeps: every value class at every size on a plain array, for the
    spellings that take the kernel; every layout and every spelling at a few
    sizes, for the kernel's element types and their neighbours; and an empty
    array.

    Args:
        operation: Operation name.
        kernels: Element type (spec name) -> kernel name.
        neighbours: Element types the operation has no kernel for.
        spellings: The ways of writing the call.
        empty_takes: Whether the kernel takes an empty plain array.
        tolerance: Whether results are compared within the kernel's tolerance.

    Yields:
        The cases.
    """
    for dtype, kernel in kernels.items():
        within = spec(kernel) if tolerance else None
        for spelling in spellings:
            if spelling.kind != TAKES or not spelling.sweep_values:
                continue
            for n, cls, label, maker in value_arrays(dtype):
                expect = expectation(kernel, spelling, CONTIGUOUS, n)
                # A kernel compared within a tolerance hands a call back to
                # NumPy when its arithmetic raises a floating-point exception.
                if tolerance and cls in MAY_RAISE:
                    expect = ANY
                yield Case(operation, f"{spelling.text} {dtype} {label}", partial(_made, maker), spelling.call,
                           expect, within)
    for dtype in (*kernels, *neighbours):
        kernel = kernels.get(dtype)
        within = spec(kernel) if tolerance and kernel else None
        for n, spelling in itertools.product(LAYOUT_SIZES, spellings):
            for variant in variants_for(dtype, n):
                yield Case(operation, f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call,
                           expectation(kernel, spelling, variant, n), within)
        for spelling in spellings:
            expect = expectation(kernel, spelling, CONTIGUOUS, 0) if empty_takes else None
            yield Case(operation, f"{spelling.text} {dtype} n=0", partial(_built, dtype, 0, CONTIGUOUS),
                       spelling.call, expect, within)


# ------------------------------------------------------------- operations

# Bit patterns of NaNs the judge's value classes do not produce: a signalling
# NaN, and a negative quiet NaN with a payload, per element size in bytes.
# Operations that copy or compare elements must keep them bit for bit, and
# the clip kernels treat signalling NaNs on a path of their own.
_UNUSUAL_NANS = {
    4: (np.uint32, 0x7FA00001, 0xFFC00123),
    8: (np.uint64, 0x7FF4000000000001, 0xFFF8000000000123),
}
# Every how many elements each of the two patterns is placed.
_SIGNALLING_EVERY = 5
_NEGATIVE_EVERY = 7


def unusual_nans(dtype: str, n: int) -> np.ndarray:
    """Build random floats with signalling and negative NaNs spread through them.

    Args:
        dtype: "f32" or "f64".
        n: Element count.

    Returns:
        A contiguous array whose every fifth element is a signalling NaN and
        every seventh, from the third, a negative quiet NaN with a payload.
    """
    x = values(dtype, n)
    unsigned, signalling, negative = _UNUSUAL_NANS[x.dtype.itemsize]
    bits = x.view(unsigned)
    bits[::_SIGNALLING_EVERY] = signalling
    bits[2::_NEGATIVE_EVERY] = negative
    return x


def unusual_nan_cases(operation: str, kernels: dict[str, str], text: str, call: Callable,
                      min_n: int = 0) -> Iterator[Case]:
    """Yield one case per size on data that holds signalling and negative NaNs.

    Args:
        operation: Operation name.
        kernels: Element type ("f32", "f64") -> kernel name.
        text: The call as text, for labels.
        call: Makes the call on the array.
        min_n: Element count from which the kernel takes the call.

    Yields:
        The cases.
    """
    for dtype, kernel in kernels.items():
        for n in VALUE_SIZES:
            yield Case(operation, f"{text} {dtype} n={n} x=signalling and negative NaNs",
                       partial(_made, partial(unusual_nans, dtype, n)), call, kernel if n >= min_n else None)


def argext_cases(name: str) -> Iterator[Case]:
    """Yield the cases of `argmax` or `argmin`.

    Args:
        name: "argmax" or "argmin".

    Yields:
        The cases. The function hook takes `np.argmax(x)` and the method hook
        `x.argmax()` and its spellings with every argument None, on a plain
        array of any size; NumPy's wrapper sends `np.argmax(x, axis=None)`
        to the method. The type slot behind every other spelling takes runs
        from 128 elements (HK_SLOT_MIN_N in the binding), and NumPy may
        copy an array into shape before the slot sees it.
    """
    function, nan_function = getattr(np, name), getattr(np, f"nan{name}")

    def method(*args, **kwargs) -> Callable[[object], object]:
        """Return a call of the method on its one argument, with these arguments.

        Args:
            *args: Positional arguments of the method call.
            **kwargs: Keyword arguments of the method call.

        Returns:
            A function of the array that makes the call.
        """
        return lambda x: getattr(x, name)(*args, **kwargs)

    spellings = (
        Spelling(f"np.{name}(x)", function, normalises=True),
        Spelling(f"x.{name}()", method(), normalises=True),
        Spelling(f"x.{name}(None)", method(None), normalises=True, sweep_values=False),
        Spelling(f"x.{name}(axis=None, out=None)", method(axis=None, out=None), normalises=True, sweep_values=False),
        Spelling(f"np.{name}(x, axis=None)", lambda x: function(x, axis=None), normalises=True, sweep_values=False),
        Spelling(f"np.{name}(a=x)", lambda x: function(a=x), normalises=True, sweep_values=False),
        Spelling(f"np.{name}(x, axis=0)", lambda x: function(x, axis=0), MAYBE),
        Spelling(f"np.{name}(x, axis=-1)", lambda x: function(x, axis=-1), MAYBE),
        Spelling(f"x.{name}(0)", method(0), MAYBE),
        Spelling(f"np.{name}(x, keepdims=True)", lambda x: function(x, keepdims=True), MAYBE),
        Spelling(f"x.{name}(keepdims=True)", method(keepdims=True), MAYBE),
        Spelling(f"x.{name}(keepdims=False)", method(keepdims=False), MAYBE),
        Spelling(f"np.{name}(x, out=o)", lambda x: function(x, out=np.empty((), np.intp)), MAYBE),
        Spelling(f"x.{name}(out=o)", lambda x: getattr(x, name)(out=np.empty((), np.intp)), MAYBE),
        Spelling(f"np.nan{name}(x)", nan_function, MAYBE),
        Spelling(f"x.{name}(bogus=1)", method(bogus=1), NEVER),
        Spelling(f"x.{name}(None, None, None, None)", method(None, None, None, None), NEVER),
    )
    kernels = {"f32": f"{name}_f32", "f64": f"{name}_f64"}
    # NumPy raises on an empty array, so no kernel takes one.
    yield from unary_cases(name, kernels, ("i64", "i32", "u8"), spellings, empty_takes=False)
    yield from unusual_nan_cases(name, kernels, f"np.{name}(x)", function)
    yield from unusual_nan_cases(name, kernels, f"x.{name}()", spellings[1].call)


# Layouts NumPy passes to the clip loop as they are, so that an in-place
# call arrives with the output equal to the input.
IN_PLACE_UNBUFFERED = frozenset({"contiguous", "rows", "readonly"})


def _clip_bounds(kernel: str) -> Iterator[tuple[str, float, float]]:
    """Yield every pair of bounds a clip kernel's spec lists, default first."""
    k = spec(kernel)
    for set_name in k.scalar_set_names:
        scalars = k.scalar_values(set_name)
        yield set_name, scalars["lo"], scalars["hi"]


def _clip_in_place(x, lo, hi):
    """Clip `x` into itself and return it."""
    return np.clip(x, lo, hi, out=x)


def _clip_method_in_place(x, lo, hi):
    """Clip `x` into itself with the method and return it."""
    return x.clip(lo, hi, out=x)


def _clip_into_fresh(x, lo, hi):
    """Clip `x` into a new array passed as `out`."""
    return np.clip(x, lo, hi, out=np.empty_like(x))


def _clip_where(x, lo, hi):
    """Clip `x` into an array of zeros where `x` is positive, and return that array."""
    return np.clip(x, lo, hi, out=np.zeros_like(x), where=np.greater(x, 0))


# Bounds a float32 array cannot hold exactly, which NumPy rounds to float32.
CLIP_ROUNDED = (0.1, 0.9)
# Python ints as bounds, which NumPy converts to the array's element type.
CLIP_INTS = (-1, 1)
# A lower bound above the upper one as doubles that rounds to the same
# float32: reversed for a float64 array, equal bounds for a float32 one.
CLIP_EQUAL_IN_F32 = (0.1 + 1e-9, 0.1)
# A Python int a float32 cannot hold, one beyond what a double holds exactly,
# and one beyond the range of a float32. NumPy rounds all three, and warns
# for the last on a float32 array.
CLIP_INT_NOT_F32, CLIP_INT_NOT_F64, CLIP_INT_TOO_LARGE = 2**24 + 1, 2**60, 2**200


def _clip_bound_cases(dtype: str, kernel: str, default: tuple[str, float, float]) -> Iterator[Case]:
    """Yield the sweep over values for bounds of each kind NumPy converts.

    Args:
        dtype: "f32" or "f64".
        kernel: The array's clip kernel.
        default: The spec's default set, as (name, lo, hi).

    Yields:
        One case per value class at FEW_SIZES for each kind of bound: the
        method with the default bounds, Python floats a float32 cannot hold,
        Python ints, NumPy scalars of the array's type, and, for float32,
        bounds that are reversed as doubles and equal as float32. For float64
        the same bounds are reversed, and no kernel may take the call.
    """
    scalar = np_dtype(dtype).type
    same_type = (scalar(default[1]), scalar(default[2]))
    for _, _, label, maker in value_arrays(dtype, FEW_SIZES):
        yield Case("clip", f"x.clip(lo, hi) {dtype} {label}", partial(_made, maker),
                   partial(_clip_method, default[1], default[2]), kernel)
        for lo, hi in (CLIP_ROUNDED, CLIP_INTS, same_type):
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                       partial(_clip3, np.clip, lo, hi), kernel)
        lo, hi = CLIP_EQUAL_IN_F32
        if dtype == "f32":
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                       partial(_clip3, np.clip, lo, hi), kernel)
    # Reversed for float64. Built from plain random values: the makers of
    # some value classes call np.clip themselves, which a kernel takes.
    if dtype == "f64":
        lo, hi = CLIP_EQUAL_IN_F32
        for n in FEW_SIZES:
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} n={n}", partial(_built, dtype, n, CONTIGUOUS),
                       partial(_clip3, np.clip, lo, hi), None)


def clip_cases() -> Iterator[Case]:
    """Yield the cases of `clip`.

    Three hooks serve clip. The function hook on `np.clip` and the method
    hook on `x.clip` call the kernel directly for a contiguous float32 or
    float64 array with no `out` and two bounds that keep the result in the
    array's element type (Python floats and ints, NumPy scalars of the
    array's type), lo <= hi. Every other call goes on to the `clip` ufunc,
    whose loop hook takes contiguous data with scalar bounds, lo <= hi, when
    the output is not the input. NumPy buffers arrays that are misaligned or
    in the other byte order and hands the loop contiguous chunks, so those
    may reach the kernel too. The counters do not say which of the hooks took
    a call; `installation_checks` proves the direct path on its own. int64,
    which the function and method hooks serve on short arrays with a loop of
    the binding's own, has its cases in `clip_i64_cases`.
    """
    kernels = {"f32": "clip_f32", "f64": "clip_f64"}
    for dtype, kernel in kernels.items():
        default = next(_clip_bounds(kernel))
        for set_name, lo, hi in _clip_bounds(kernel):
            sizes = VALUE_SIZES if set_name in (DEFAULT_SCALARS, "zero_lo") else FEW_SIZES
            for _, _, label, maker in value_arrays(dtype, sizes):
                yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) {dtype} {label}", partial(_made, maker),
                           partial(_clip3, np.clip, lo, hi), kernel)
        yield from _clip_bound_cases(dtype, kernel, default)
    yield from unusual_nan_cases("clip", kernels, "np.clip(x, -0.5, 0.5)", partial(_clip3, np.clip, -0.5, 0.5))
    yield from unusual_nan_cases("clip", kernels, "np.clip(x, 0.0, 1.0)", partial(_clip3, np.clip, 0.0, 1.0))
    yield from unusual_nan_cases("clip", kernels, "x.clip(-0.5, 0.5)", partial(_clip_method, -0.5, 0.5))
    lo, hi, nan = -0.5, 0.5, float("nan")
    spellings = (
        Spelling("np.clip(x, lo, hi)", partial(_clip3, np.clip, lo, hi), normalises=True),
        Spelling("x.clip(lo, hi)", partial(_clip_method, lo, hi), normalises=True),
        Spelling("np.clip(x, min=lo, max=hi)", lambda x: np.clip(x, min=lo, max=hi), normalises=True),
        Spelling("np.clip(x, max=hi, min=lo)", lambda x: np.clip(x, max=hi, min=lo), normalises=True),
        Spelling("np.clip(x, a_min=lo, a_max=hi)", lambda x: np.clip(x, a_min=lo, a_max=hi), normalises=True),
        Spelling("np.clip(a=x, a_min=lo, a_max=hi)", lambda x: np.clip(a=x, a_min=lo, a_max=hi), normalises=True),
        Spelling("np.clip(x, lo, hi, None)", lambda x: np.clip(x, lo, hi, None), normalises=True),
        Spelling("np.clip(x, lo, hi, out=None)", lambda x: np.clip(x, lo, hi, out=None), normalises=True),
        Spelling("x.clip(min=lo, max=hi)", lambda x: x.clip(min=lo, max=hi), normalises=True),
        Spelling("x.clip(lo, hi, None)", lambda x: x.clip(lo, hi, None), normalises=True),
        Spelling("x.clip(lo, hi, out=None)", lambda x: x.clip(lo, hi, out=None), normalises=True),
        Spelling("np.clip(x, -1, 1)", lambda x: np.clip(x, -1, 1), normalises=True),
        Spelling("np.clip(x, 0.1, 0.9)", lambda x: np.clip(x, 0.1, 0.9), normalises=True),
        Spelling("np.clip(x, -inf, inf)", lambda x: np.clip(x, -np.inf, np.inf), normalises=True),
        Spelling("np.clip(x, lo, inf)", lambda x: np.clip(x, lo, np.inf), normalises=True),
        Spelling("np.clip(x, -1e300, 1e300)", lambda x: np.clip(x, -1e300, 1e300), normalises=True),
        Spelling("np.clip(x, -1e-300, 1e-300)", lambda x: np.clip(x, -1e-300, 1e-300), normalises=True),
        Spelling("np.clip(x, -1e-40, 1e-40)", lambda x: np.clip(x, -1e-40, 1e-40), normalises=True),
        Spelling("np.clip(x, 2**24 + 1, 2**25 + 2)", lambda x: np.clip(x, CLIP_INT_NOT_F32, 2 * CLIP_INT_NOT_F32), normalises=True),
        Spelling("np.clip(x, -2**60, 2**60)", lambda x: np.clip(x, -CLIP_INT_NOT_F64, CLIP_INT_NOT_F64), normalises=True),
        Spelling("np.clip(x, -2**200, 2**200)", lambda x: np.clip(x, -CLIP_INT_TOO_LARGE, CLIP_INT_TOO_LARGE), normalises=True),
        Spelling("np.clip(x, False, True)", lambda x: np.clip(x, False, True), normalises=True),
        Spelling("np.clip(x, np.float32(lo), np.float32(hi))", lambda x: np.clip(x, np.float32(lo), np.float32(hi)), normalises=True),
        Spelling("np.clip(x, lo, np.float32(hi))", lambda x: np.clip(x, lo, np.float32(hi)), normalises=True),
        Spelling("x.clip(np.float32(lo), np.float32(hi))", lambda x: x.clip(np.float32(lo), np.float32(hi)), normalises=True),
        Spelling("np.clip(x, lo, hi, out=fresh)", lambda x: _clip_into_fresh(x, lo, hi), normalises=True),
        Spelling("np.clip(x, np.float64(lo), np.float64(hi))", lambda x: np.clip(x, np.float64(lo), np.float64(hi)), MAYBE),
        Spelling("np.clip(x, lo, np.float64(hi))", lambda x: np.clip(x, lo, np.float64(hi)), MAYBE),
        Spelling("x.clip(np.float64(lo), np.float64(hi))", lambda x: x.clip(np.float64(lo), np.float64(hi)), MAYBE),
        Spelling("np.clip(x, np.array(lo), np.array(hi))", lambda x: np.clip(x, np.array(lo), np.array(hi)), MAYBE),
        Spelling("np.clip(x, lo_array, hi_array)", lambda x: np.clip(x, np.full(np.shape(x), lo), np.full(np.shape(x), hi)), MAYBE),
        Spelling("np.clip(x, lo, hi, dtype=np.float64)", lambda x: np.clip(x, lo, hi, dtype=np.float64), MAYBE),
        Spelling("np.clip(x, lo, hi, casting='unsafe')", lambda x: np.clip(x, lo, hi, casting="unsafe"), MAYBE),
        Spelling("np.clip(x, lo, hi, out=zeros, where=x > 0)", lambda x: _clip_where(x, lo, hi), MAYBE),
        Spelling("x.clip(lo, hi, order='F')", lambda x: x.clip(lo, hi, order="F"), MAYBE),
        Spelling("np.clip(x, lo, hi, out=x)", lambda x: _clip_in_place(x, lo, hi), NEVER, normalises=True),
        Spelling("x.clip(lo, hi, out=x)", lambda x: _clip_method_in_place(x, lo, hi), NEVER, normalises=True),
        Spelling("np.clip(x, lo, None)", lambda x: np.clip(x, lo, None), NEVER),
        Spelling("np.clip(x, None, hi)", lambda x: np.clip(x, None, hi), NEVER),
        Spelling("np.clip(x, min=lo)", lambda x: np.clip(x, min=lo), NEVER),
        Spelling("np.clip(x, max=hi)", lambda x: np.clip(x, max=hi), NEVER),
        Spelling("np.clip(x, lo)", lambda x: np.clip(x, lo), NEVER),
        Spelling("np.clip(x, a_min=lo, max=hi)", lambda x: np.clip(x, a_min=lo, max=hi), NEVER),
        Spelling("np.clip(x, lo, hi, min=lo)", lambda x: np.clip(x, lo, hi, min=lo), NEVER),
        Spelling("np.clip(x, lo, hi, bogus=1)", lambda x: np.clip(x, lo, hi, bogus=1), NEVER),
        Spelling("np.clip(x, lo, hi, None, None)", lambda x: np.clip(x, lo, hi, None, None), NEVER),
        Spelling("x.clip(lo)", lambda x: x.clip(lo), NEVER),
        Spelling("x.clip(max=hi)", lambda x: x.clip(max=hi), NEVER),
        Spelling("x.clip()", lambda x: x.clip(), NEVER),
        Spelling("x.clip(lo, hi, None, None)", lambda x: x.clip(lo, hi, None, None), NEVER),
        Spelling("x.clip(a_min=lo, a_max=hi)", lambda x: x.clip(a_min=lo, a_max=hi), NEVER),
        Spelling("np.clip(x, hi, lo) reversed", lambda x: np.clip(x, hi, lo), NEVER),
        Spelling("x.clip(hi, lo) reversed", lambda x: x.clip(hi, lo), NEVER),
        Spelling("np.clip(x, nan, hi)", lambda x: np.clip(x, nan, hi), NEVER),
        Spelling("np.clip(x, lo, nan)", lambda x: np.clip(x, lo, nan), NEVER),
        Spelling("x.clip(nan, hi)", lambda x: x.clip(nan, hi), NEVER),
        Spelling("np.clip(x, np.float32(nan), np.float32(hi))", lambda x: np.clip(x, np.float32(nan), np.float32(hi)), NEVER),
    )
    for dtype in ("f32", "f64", "i64"):
        kernel = kernels.get(dtype)
        for n, spelling in itertools.product((0, *LAYOUT_SIZES), spellings):
            for variant in variants_for(dtype, n):
                expect = expectation(kernel, spelling, variant, n)
                # An in-place call on an array NumPy has to buffer reaches the
                # loop with separate buffers, which the kernel may take.
                if spelling.kind == NEVER and spelling.normalises and kernel and variant.name not in IN_PLACE_UNBUFFERED:
                    expect = ANY
                # NumPy converts integers to float64 for float bounds, and the
                # float64 loop may then take the converted data.
                if dtype == "i64" and spelling.kind != NEVER:
                    expect = ANY
                # The loop is not called at all for an empty array.
                if n == 0 and expect == kernel:
                    expect = None
                yield Case("clip", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call, expect)
    yield from clip_i64_cases()


# The largest int64 array the binding's own clip loop takes: CLIP_I64_MAX_N
# in hook_function.c. Above it the call is NumPy's.
CLIP_I64_MAX_N = 4096
# Sizes of the int64 sweep over values: every size of the general sweep up to
# and beyond the limit, and one below it.
CLIP_I64_SIZES = tuple(sorted({*VALUE_SIZES, CLIP_I64_MAX_N - 1}))
# Sizes of the int64 sweep over layouts and spellings: empty, the general
# ones, and the limit and one above it.
CLIP_I64_LAYOUT_SIZES = (0, *LAYOUT_SIZES, CLIP_I64_MAX_N, CLIP_I64_MAX_N + 1)
# The extremes of int64, as Python ints.
I64_MIN, I64_MAX = -2**63, 2**63 - 1
# Bounds of the int64 sweep over values, every pair with lo <= hi, as the
# types NumPy keeps an int64 result for: Python ints and np.int64. Some cut
# off much of the judge's data and some little; the extremes cut off none.
CLIP_I64_BOUNDS = (
    (-1, 1),
    (0, 2**62),
    (-2**40, 2**40),
    (5, 5),
    (I64_MIN, I64_MAX),
    (I64_MIN, I64_MIN),
    (I64_MAX, I64_MAX),
    (np.int64(-1000), np.int64(1000)),
    (-10, np.int64(10)),
)
# The bounds swept at every size; the others are swept at FEW_SIZES and
# around the limit.
CLIP_I64_FULL_SWEEP = 2
# Element types beside int64 that the loop must leave to NumPy, whatever the
# bounds. On Linux `longlong` is a second 64-bit type with a type number of
# its own.
CLIP_I64_NEIGHBOURS = ("i32", "u64", "i16")


# Text in a spelling of an int64 clip that marks a bound given as a NumPy
# integer or array rather than a Python int.
_CLIP_NUMPY_BOUND_MARKERS = ("np.int", "np.longlong", "np.array", "_array")


def _clip_i64_expect(n: int) -> str | None:
    """Return the kernel a plain contiguous int64 clip of `n` elements with int64 bounds should take."""
    return "clip_i64" if 1 <= n <= CLIP_I64_MAX_N else None


def _longlong(n: int) -> tuple:
    """Build random int64 values as an array of `np.longlong`, which is another type number on Linux."""
    return (values("i64", n).astype(np.longlong),)


def clip_i64_cases() -> Iterator[Case]:
    """Yield the cases of `np.clip` and `x.clip` on int64.

    The function and method hooks run the binding's own loop for a plain
    contiguous int64 array of 1 to CLIP_I64_MAX_N elements, both bounds a
    Python int that fits int64 or an np.int64, lo <= hi, and no `out` or
    anything else. Every other call is NumPy's: a bound of another type
    changes the result's type or is converted by NumPy's own rules, and a
    float bound sends NumPy to its float64 loop, which the clip loop hook
    may then take, so no expectation is stated for it.
    """
    sizes_near_limit = (CLIP_I64_MAX_N - 1, CLIP_I64_MAX_N, CLIP_I64_MAX_N + 1)
    for index, (lo, hi) in enumerate(CLIP_I64_BOUNDS):
        sizes = CLIP_I64_SIZES if index < CLIP_I64_FULL_SWEEP else (*FEW_SIZES, *sizes_near_limit)
        for n, _, label, maker in value_arrays("i64", sizes):
            yield Case("clip", f"np.clip(x, {lo!r}, {hi!r}) i64 {label}", partial(_made, maker),
                       partial(_clip3, np.clip, lo, hi), _clip_i64_expect(n))
            yield Case("clip", f"x.clip({lo!r}, {hi!r}) i64 {label}", partial(_made, maker),
                       partial(_clip_method, lo, hi), _clip_i64_expect(n))
    lo, hi, low, big = -100, 100, np.int64(-100), 2**63
    spellings = (
        Spelling("np.clip(x, lo, hi)", partial(_clip3, np.clip, lo, hi)),
        Spelling("x.clip(lo, hi)", partial(_clip_method, lo, hi)),
        Spelling("np.clip(x, min=lo, max=hi)", lambda x: np.clip(x, min=lo, max=hi)),
        Spelling("np.clip(x, a_min=lo, a_max=hi)", lambda x: np.clip(x, a_min=lo, a_max=hi)),
        Spelling("np.clip(a=x, a_min=lo, a_max=hi)", lambda x: np.clip(a=x, a_min=lo, a_max=hi)),
        Spelling("np.clip(x, lo, hi, None)", lambda x: np.clip(x, lo, hi, None)),
        Spelling("np.clip(x, lo, hi, out=None)", lambda x: np.clip(x, lo, hi, out=None)),
        Spelling("x.clip(min=lo, max=hi)", lambda x: x.clip(min=lo, max=hi)),
        Spelling("x.clip(lo, hi, None)", lambda x: x.clip(lo, hi, None)),
        Spelling("x.clip(lo, hi, out=None)", lambda x: x.clip(lo, hi, out=None)),
        Spelling("np.clip(x, np.int64(lo), np.int64(hi))", lambda x: np.clip(x, low, np.int64(hi))),
        Spelling("x.clip(np.int64(lo), hi)", lambda x: x.clip(low, hi)),
        Spelling("np.clip(x, lo, lo)", lambda x: np.clip(x, lo, lo)),
        Spelling("np.clip(x, -2**63, 2**63 - 1)", lambda x: np.clip(x, I64_MIN, I64_MAX)),
        Spelling("np.clip(x, -0.5, 0.5)", lambda x: np.clip(x, -0.5, 0.5), MAYBE),
        Spelling("np.clip(x, lo, 100.0)", lambda x: np.clip(x, lo, 100.0), MAYBE),
        Spelling("np.clip(x, np.uint64(0), np.uint64(hi))", lambda x: np.clip(x, np.uint64(0), np.uint64(hi)), MAYBE),
        Spelling("np.clip(x, np.float64(lo), hi)", lambda x: np.clip(x, np.float64(lo), hi), MAYBE),
        Spelling("np.clip(x, hi, lo) reversed", lambda x: np.clip(x, hi, lo), NEVER),
        Spelling("x.clip(hi, lo) reversed", lambda x: x.clip(hi, lo), NEVER),
        Spelling("np.clip(x, 2**63, 2**63 + 1)", lambda x: np.clip(x, big, big + 1), NEVER),
        Spelling("np.clip(x, -2**63 - 1, hi)", lambda x: np.clip(x, I64_MIN - 1, hi), NEVER),
        Spelling("np.clip(x, lo, 2**64)", lambda x: np.clip(x, lo, 2**64), NEVER),
        Spelling("np.clip(x, np.int32(lo), hi)", lambda x: np.clip(x, np.int32(lo), hi), NEVER),
        Spelling("np.clip(x, np.longlong(lo), hi)", lambda x: np.clip(x, np.longlong(lo), hi), NEVER),
        Spelling("np.clip(x, False, True)", lambda x: np.clip(x, False, True), NEVER),
        Spelling("np.clip(x, np.array(lo), hi)", lambda x: np.clip(x, np.array(lo), hi), NEVER),
        Spelling("np.clip(x, lo_array, hi_array)", lambda x: np.clip(x, np.full(np.shape(x), lo), np.full(np.shape(x), hi)), NEVER),
        Spelling("np.clip(x, lo, None)", lambda x: np.clip(x, lo, None), NEVER),
        Spelling("np.clip(x, None, hi)", lambda x: np.clip(x, None, hi), NEVER),
        Spelling("x.clip(lo)", lambda x: x.clip(lo), NEVER),
        Spelling("np.clip(x, lo, hi, out=x)", lambda x: _clip_in_place(x, lo, hi), NEVER),
        Spelling("x.clip(lo, hi, out=x)", lambda x: _clip_method_in_place(x, lo, hi), NEVER),
        Spelling("np.clip(x, lo, hi, out=fresh)", lambda x: _clip_into_fresh(x, lo, hi), NEVER),
        Spelling("np.clip(x, lo, hi, dtype=np.int64)", lambda x: np.clip(x, lo, hi, dtype=np.int64), NEVER),
        Spelling("np.clip(x, lo, hi, casting='unsafe')", lambda x: np.clip(x, lo, hi, casting="unsafe"), NEVER),
        Spelling("np.clip(x, lo, hi, out=zeros, where=x > 0)", lambda x: _clip_where(x, lo, hi), NEVER),
        Spelling("np.clip(x, lo, hi, bogus=1)", lambda x: np.clip(x, lo, hi, bogus=1), NEVER),
        Spelling("x.clip(lo, hi, order='F')", lambda x: x.clip(lo, hi, order="F"), NEVER),
    )
    for dtype in ("i64", *CLIP_I64_NEIGHBOURS):
        kernel = "clip_i64" if dtype == "i64" else None
        for n, spelling in itertools.product(CLIP_I64_LAYOUT_SIZES, spellings):
            for variant in variants_for(dtype, n):
                expect = expectation(kernel, spelling, variant, n)
                # The two-dimensional layouts hold the whole rows of n elements.
                size = n // COLUMNS * COLUMNS if variant.dims == 2 else n
                # Above the limit and for an empty array the call is NumPy's.
                if expect == kernel and not 1 <= size <= CLIP_I64_MAX_N:
                    expect = None
                # np.clip converts a list to an array of NumPy's choosing and
                # calls its method, which the method hook serves: for int64
                # values an int64 array, for the others whatever NumPy makes
                # of them. A list has no method of its own.
                if variant.name == "list" and spelling.text.startswith("np."):
                    expect = _clip_i64_expect(n) if kernel and spelling.kind == TAKES else ANY
                # A float bound sends NumPy to the float64 clip loop, which the
                # loop hook may take. So does a NumPy integer bound beside a
                # uint64 array: NumPy promotes the pair to float64.
                numpy_bound = any(marker in spelling.text for marker in _CLIP_NUMPY_BOUND_MARKERS)
                if spelling.kind == MAYBE or (dtype == "u64" and numpy_bound):
                    expect = ANY
                yield Case("clip", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call, expect)
    for n in (1, 16, CLIP_I64_MAX_N):
        yield Case("clip", f"np.clip(x, -100, 100) longlong n={n}", partial(_longlong, n),
                   partial(_clip3, np.clip, -100, 100), None)


# ------------------------------------------------------------------ round

# The largest float64 array the binding's own round loop takes:
# ROUND_F64_MAX_N in hook_method.c. Above it the call is NumPy's.
ROUND_F64_MAX_N = 4096
# Sizes of the sweep over values: every size of the general sweep up to and
# beyond the limit, and one below it.
ROUND_SIZES = tuple(sorted({*VALUE_SIZES, ROUND_F64_MAX_N - 1}))
# Sizes of the sweep over layouts and spellings: empty, the general ones, and
# the limit and one above it.
ROUND_LAYOUT_SIZES = (0, *LAYOUT_SIZES, ROUND_F64_MAX_N, ROUND_F64_MAX_N + 1)
# Numbers of decimals the binding takes, 0 to 15. The first two are swept at
# every size, the others at FEW_SIZES and around the limit.
ROUND_DECIMALS = (0, 2, 1, 3, 5, 8, 10, 15)
ROUND_FULL_SWEEP = 2
# Value classes on which multiplying by a power of ten can overflow or
# underflow, so that NumPy warns or not depending on the values: the call is
# taken exactly when it raises no flag, which the case does not predict.
ROUND_MAY_RAISE = frozenset({"denormal", "extremes"})
# Element types beside float64 that the loop must leave to NumPy.
ROUND_NEIGHBOURS = ("f32", "i64", "i32")
# Values whose rounding a sign or a tie decides: negative values that round
# to a zero, which must give -0.0 where NumPy's does, both zeros, and ties at
# 0 and at 2 decimals, which go to the even neighbour.
ROUND_SIGNS = (-0.004, -0.3, 0.004, -0.0, 0.0, -0.5, 0.5, 1.5, 2.5, -2.5, -0.005, 0.005, 0.125, -0.125,
               0.375, 1e-17, -1e-17, 4503599627370495.5, -4503599627370495.5)
# A value whose product with 100 overflows, for which NumPy warns or raises
# as np.errstate says, and a subnormal value. A subnormal number times a
# power of ten is exact while it stays subnormal, and a whole number divided
# by one is not tiny, so rounding never underflows: that call is taken.
ROUND_OVERFLOWS = 1e307
ROUND_SUBNORMAL = 1.2345e-310


def _round_expect(n: int, decimals: int = 0, cls: str = "") -> str | None:
    """Say whether the binding's loop takes np.round on a plain float64 array.

    Args:
        n: Element count.
        decimals: The number of decimals, from 0 to 15.
        cls: The value class of the array, or "" for ordinary values.

    Returns:
        "round_f64", None, or ANY where the values decide.
    """
    if not 1 <= n <= ROUND_F64_MAX_N:
        return None
    return ANY if decimals and cls in ROUND_MAY_RAISE else "round_f64"


def _round_call(decimals, x):
    """Call `np.round(x, decimals)`."""
    return np.round(x, decimals)


def _round_method(decimals, x):
    """Call `x.round(decimals)`."""
    return x.round(decimals)


def _round_into(x, decimals, target: str):
    """Call `np.round(x, decimals, out=...)` into `x` itself or into a new array, and return the output.

    Args:
        x: The array.
        decimals: The number of decimals.
        target: "self" to round into `x`, "fresh" for a new array.

    Returns:
        The output array.
    """
    out = x if target == "self" else np.empty_like(x)
    return np.round(x, decimals, out=out)


def _round_errstate(setting: str, decimals: int, x):
    """Call `np.round(x, decimals)` with every floating-point error set to `setting`.

    Args:
        setting: A value np.errstate takes, such as "raise" or "warn".
        decimals: The number of decimals.
        x: The array.

    Returns:
        The rounded array.
    """
    with np.errstate(all=setting):
        return np.round(x, decimals)


def _filled(value: float, n: int) -> tuple:
    """Build the one argument of a unary case: random float64 values with `value` at every third element."""
    x = values("f64", n)
    x[::3] = value
    return (x,)


def round_cases() -> Iterator[Case]:
    """Yield the cases of `np.round`, `np.around` and `x.round`.

    The function hooks of np.round and np.around and the method hook of
    x.round run the binding's own loop for a plain contiguous float64 array
    of 1 to ROUND_F64_MAX_N elements, decimals absent or a Python int from
    0 to 15, and no `out`. A call whose arithmetic raises a flag NumPy
    reports, overflow or underflow in the multiply or a signalling NaN, goes
    to NumPy, which warns or raises. Every other call is NumPy's.
    """
    sizes_near_limit = (ROUND_F64_MAX_N - 1, ROUND_F64_MAX_N, ROUND_F64_MAX_N + 1)
    for index, decimals in enumerate(ROUND_DECIMALS):
        sizes = ROUND_SIZES if index < ROUND_FULL_SWEEP else (*FEW_SIZES, *sizes_near_limit)
        for n, cls, label, maker in value_arrays("f64", sizes):
            expect = _round_expect(n, decimals, cls)
            yield Case("round", f"np.round(x, {decimals}) f64 {label}", partial(_made, maker),
                       partial(_round_call, decimals), expect)
            yield Case("round", f"x.round({decimals}) f64 {label}", partial(_made, maker),
                       partial(_round_method, decimals), expect)
    for decimals in ROUND_DECIMALS:
        signs = np.array(ROUND_SIGNS)
        yield Case("round", f"np.round(x, {decimals}) f64 signs and ties", partial(_made, signs.copy),
                   partial(_round_call, decimals), "round_f64")
        for n in (1, 16, ROUND_F64_MAX_N):
            yield Case("round", f"np.round(x, {decimals}) f64 n={n} x=signalling and negative NaNs",
                       partial(_made, partial(unusual_nans, "f64", n)), partial(_round_call, decimals), None)
    # The guards: a flag NumPy reports sends the call to NumPy, which warns,
    # raises or stays silent as np.errstate says.
    for n in (1, 16, ROUND_F64_MAX_N):
        for decimals, value, flagged in ((2, ROUND_OVERFLOWS, True), (2, ROUND_SUBNORMAL, False),
                                          (0, ROUND_OVERFLOWS, False), (0, ROUND_SUBNORMAL, False)):
            expect = None if flagged else "round_f64"
            make = partial(_filled, value, n)
            yield Case("round", f"np.round(x, {decimals}) f64 n={n} x={value!r} every third", make,
                       partial(_round_call, decimals), expect)
            for setting in ("raise", "warn", "ignore"):
                yield Case("round", f"np.errstate(all={setting!r}) np.round(x, {decimals}) f64 n={n} x={value!r} every third",
                           make, partial(_round_errstate, setting, decimals), expect)
    spellings = (
        Spelling("np.round(x, 2)", partial(_round_call, 2)),
        Spelling("np.round(x)", np.round),
        Spelling("np.around(x, 2)", lambda x: np.around(x, 2)),
        Spelling("np.around(x)", np.around),
        Spelling("x.round(2)", partial(_round_method, 2)),
        Spelling("x.round()", lambda x: x.round()),
        Spelling("np.round(x, decimals=2)", lambda x: np.round(x, decimals=2)),
        Spelling("np.round(a=x, decimals=2)", lambda x: np.round(a=x, decimals=2)),
        Spelling("np.round(x, 2, None)", lambda x: np.round(x, 2, None)),
        Spelling("np.round(x, 2, out=None)", lambda x: np.round(x, 2, out=None)),
        Spelling("np.round(x, 0)", partial(_round_call, 0)),
        Spelling("np.round(x, 15)", partial(_round_call, 15)),
        Spelling("x.round(decimals=2)", lambda x: x.round(decimals=2)),
        Spelling("x.round(2, None)", lambda x: x.round(2, None)),
        Spelling("x.round(2, out=None)", lambda x: x.round(2, out=None)),
        Spelling("x.round(out=None, decimals=2)", lambda x: x.round(out=None, decimals=2)),
        Spelling("np.round(x, -1)", partial(_round_call, -1), NEVER),
        Spelling("np.round(x, 16)", partial(_round_call, 16), NEVER),
        Spelling("np.round(x, 2**70)", partial(_round_call, 2**70), NEVER),
        Spelling("x.round(-2)", partial(_round_method, -2), NEVER),
        Spelling("np.round(x, np.int64(2))", partial(_round_call, np.int64(2)), NEVER),
        Spelling("x.round(np.int32(2))", partial(_round_method, np.int32(2)), NEVER),
        Spelling("np.round(x, True)", partial(_round_call, True), NEVER),
        Spelling("np.round(x, 2.0)", partial(_round_call, 2.0), NEVER),
        Spelling("np.round(x, None)", partial(_round_call, None), NEVER),
        Spelling("np.round(x, '2')", partial(_round_call, "2"), NEVER),
        Spelling("np.round(x, 2, out=x)", lambda x: _round_into(x, 2, "self"), NEVER),
        Spelling("np.round(x, 2, out=fresh)", lambda x: _round_into(x, 2, "fresh"), NEVER),
        Spelling("x.round(2, out=x)", lambda x: x.round(2, out=x), NEVER),
        Spelling("x.round(2, decimals=2)", lambda x: x.round(2, decimals=2), NEVER),
        Spelling("x.round(2, None, None)", lambda x: x.round(2, None, None), NEVER),
        Spelling("x.round(bogus=1)", lambda x: x.round(bogus=1), NEVER),
        Spelling("np.round(x, 2, bogus=1)", lambda x: np.round(x, 2, bogus=1), NEVER),
    )
    for dtype in ("f64", *ROUND_NEIGHBOURS):
        kernel = "round_f64" if dtype == "f64" else None
        for n, spelling in itertools.product(ROUND_LAYOUT_SIZES, spellings):
            for variant in variants_for(dtype, n):
                expect = expectation(kernel, spelling, variant, n)
                # The two-dimensional layouts hold the whole rows of n elements.
                size = n // COLUMNS * COLUMNS if variant.dims == 2 else n
                # Above the limit and for an empty array the call is NumPy's.
                if expect == kernel and not 1 <= size <= ROUND_F64_MAX_N:
                    expect = None
                # np.round converts a list to an array of NumPy's choosing and
                # calls its method, which the method hook serves: a list of
                # Python floats is a float64 array, whatever the type it was
                # built from. A list has no method of its own.
                if variant.name == "list" and spelling.text.startswith("np.") and spelling.kind == TAKES:
                    expect = ANY if dtype != "f64" else _round_expect(n)
                yield Case("round", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call, expect)


# ----------------------------------------------------------------- median

# The largest float64 array the binding's own median loop takes:
# MEDIAN_F64_MAX_N in hook_function.c. Above it the call is NumPy's.
MEDIAN_F64_MAX_N = 128
# Sizes of the sweep over values: every size of the general sweep up to the
# limit and one beyond it, every size from 1 to 9 for the odd and even
# cases, and the limit's neighbours. Above the limit nothing is built.
MEDIAN_SIZES = tuple(sorted({*(n for n in VALUE_SIZES if n <= 2 * MEDIAN_F64_MAX_N), *range(1, 10),
                             MEDIAN_F64_MAX_N - 1, MEDIAN_F64_MAX_N, MEDIAN_F64_MAX_N + 1}))
# Sizes of the sweep over layouts and spellings: empty, a short one, the limit
# and one above it, and the general ones.
MEDIAN_LAYOUT_SIZES = (0, 16, MEDIAN_F64_MAX_N, MEDIAN_F64_MAX_N + 1, *LAYOUT_SIZES[1:])
# Element types beside float64 that the loop must leave to NumPy.
MEDIAN_NEIGHBOURS = ("f32", "i64", "i32")
# Small arrays that probe each guard and the cases on either side of it, as
# (label, values, whether the loop takes it). A zero median goes to NumPy
# when a negative zero is present, wherever it is; sums that overflow or
# meet infinities of both signs, and halves that underflow, go to NumPy,
# which warns or raises; infinities of one sign are taken.
_DBL_MAX, _DBL_MIN, _DBL_TRUE_MIN = 1.7976931348623157e308, 2.2250738585072014e-308, 5e-324
MEDIAN_GUARDS = (
    ("one -0.0", (-0.0,), False),
    ("one 0.0", (0.0,), True),
    ("-0.0 and 0.0", (-0.0, 0.0), False),
    ("two -0.0", (-0.0, -0.0), False),
    ("two 0.0", (0.0, 0.0), True),
    ("-1 and 1, a zero median without -0.0", (-1.0, 1.0), True),
    ("-1, -0.0, 0.0, 1: a zero median from both zeros", (-1.0, -0.0, 0.0, 1.0), False),
    ("-1, -0.0, 1, 2: -0.0 beside a median of 0.5", (-1.0, -0.0, 1.0, 2.0), True),
    ("-0.0 away from the middle", (-0.0, 1.0, 2.0, 3.0, 4.0), True),
    ("0.0 and -0.0 with zero in the middle", (0.0, -0.0, 0.0), False),
    ("the largest two", (_DBL_MAX, _DBL_MAX), False),
    ("the largest and its negative", (_DBL_MAX, -_DBL_MAX), True),
    ("the largest three", (_DBL_MAX, _DBL_MAX, _DBL_MAX), True),
    ("inf and -inf", (np.inf, -np.inf), False),
    ("inf and inf", (np.inf, np.inf), True),
    ("-inf, -inf, 5", (-np.inf, -np.inf, 5.0), True),
    ("inf and 1", (np.inf, 1.0), True),
    ("the smallest subnormal and 0", (_DBL_TRUE_MIN, 0.0), False),
    ("two smallest subnormals", (_DBL_TRUE_MIN, _DBL_TRUE_MIN), True),
    ("three smallest subnormals and 0: an exact half", (_DBL_TRUE_MIN, _DBL_TRUE_MIN, _DBL_TRUE_MIN, 0.0), True),
    ("two zeros and two smallest subnormals", (0.0, 0.0, _DBL_TRUE_MIN, _DBL_TRUE_MIN), False),
    ("the smallest normal and 0", (_DBL_MIN, 0.0), True),
    ("the smallest normal and the next: the sum rounds, the half is exact", (_DBL_MIN, np.nextafter(_DBL_MIN, 1.0)), True),
    ("the smallest normal and the smallest subnormal: an inexact half", (_DBL_MIN, _DBL_TRUE_MIN), False),
    ("twice the smallest normal and 0", (2 * _DBL_MIN, 0.0), True),
    ("a NaN", (1.0, np.nan, 2.0), False),
    ("only NaN", (np.nan,), False),
)


def _median_expect(x: np.ndarray) -> str | None:
    """Say whether the binding's loop takes np.median on a plain contiguous float64 array.

    Args:
        x: The array, as the call receives it.

    Returns:
        "median_f64" when the array has 1 to MEDIAN_F64_MAX_N elements, no
        NaN, a mean of its middle elements that raises no floating-point
        error, and no negative zero when the median is a zero; else None.
    """
    n = x.size
    if not 1 <= n <= MEDIAN_F64_MAX_N or np.isnan(x).any():
        return None
    middle = np.sort(x.ravel())[(n - 1) // 2:n // 2 + 1]
    try:
        with np.errstate(all="raise"):
            median = middle[0] if n % 2 else (middle[0] + middle[1]) / np.float64(2)
    except FloatingPointError:
        return None
    if median == 0 and np.signbit(x[x == 0]).any():
        return None
    return "median_f64"


def _median_errstate(setting: str, x):
    """Call `np.median(x)` with every floating-point error set to `setting`.

    Args:
        setting: A value np.errstate takes, such as "raise" or "warn".
        x: The array.

    Returns:
        The median.
    """
    with np.errstate(all=setting):
        return np.median(x)


def _median_out(x):
    """Call `np.median(x, out=o)` into a new zero-dimensional float64 array and return it."""
    return np.median(x, out=np.empty((), np.float64))


def median_cases() -> Iterator[Case]:
    """Yield the cases of `np.median`.

    The function hook runs the binding's own loop for `np.median(x)` with the
    array alone, by position, a plain contiguous float64 array of any shape
    with 1 to MEDIAN_F64_MAX_N elements and no NaN, when the mean of its
    middle elements raises nothing NumPy reports and the median is not a
    zero beside a negative zero. Every other call is NumPy's.
    """
    for n, _, label, maker in value_arrays("f64", MEDIAN_SIZES):
        yield Case("median", f"np.median(x) f64 {label}", partial(_made, maker), np.median, _median_expect(maker()))
        cube = n % 4 == 0 and n >= 4
        if cube:
            shaped = partial(np.reshape, maker(), (-1, 2, 2))
            yield Case("median", f"np.median(x) f64 three dimensions {label}", partial(_made, shaped), np.median,
                       _median_expect(maker()))
    for label, numbers, taken in MEDIAN_GUARDS:
        x = np.array(numbers, np.float64)
        expect = "median_f64" if taken else None
        make = partial(_made, x.copy)
        yield Case("median", f"np.median(x) f64 {label}", make, np.median, expect)
        for setting in ("raise", "warn", "ignore"):
            yield Case("median", f"np.errstate(all={setting!r}) np.median(x) f64 {label}", make,
                       partial(_median_errstate, setting), expect)
    for n in (1, 2, 16, MEDIAN_F64_MAX_N):
        yield Case("median", f"np.median(x) f64 n={n} x=signalling and negative NaNs",
                   partial(_made, partial(unusual_nans, "f64", n)), np.median, None)
    spellings = (
        Spelling("np.median(x)", np.median),
        Spelling("np.median(a=x)", lambda x: np.median(a=x), NEVER),
        Spelling("np.median(x, None)", lambda x: np.median(x, None), NEVER),
        Spelling("np.median(x, axis=None)", lambda x: np.median(x, axis=None), NEVER),
        Spelling("np.median(x, axis=0)", lambda x: np.median(x, axis=0), NEVER),
        Spelling("np.median(x, axis=-1)", lambda x: np.median(x, axis=-1), NEVER),
        Spelling("np.median(x, out=o)", _median_out, NEVER),
        Spelling("np.median(x, overwrite_input=True)", lambda x: np.median(x, overwrite_input=True), NEVER),
        Spelling("np.median(x, overwrite_input=False)", lambda x: np.median(x, overwrite_input=False), NEVER),
        Spelling("np.median(x, keepdims=True)", lambda x: np.median(x, keepdims=True), NEVER),
        Spelling("np.median(x, keepdims=False)", lambda x: np.median(x, keepdims=False), NEVER),
        Spelling("np.median(x, bogus=1)", lambda x: np.median(x, bogus=1), NEVER),
        Spelling("np.nanmedian(x)", np.nanmedian, NEVER),
        Spelling("np.percentile(x, 50)", lambda x: np.percentile(x, 50), NEVER),
    )
    for dtype in ("f64", *MEDIAN_NEIGHBOURS):
        kernel = "median_f64" if dtype == "f64" else None
        for n, spelling in itertools.product(MEDIAN_LAYOUT_SIZES, spellings):
            for variant in variants_for(dtype, n):
                expect = expectation(kernel, spelling, variant, n)
                # The two-dimensional layouts hold the whole rows of n elements.
                size = n // COLUMNS * COLUMNS if variant.dims == 2 else n
                if expect == kernel and not 1 <= size <= MEDIAN_F64_MAX_N:
                    expect = None
                yield Case("median", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call, expect)


def _clip3(function, lo, hi, x):
    """Call `function(x, lo, hi)`."""
    return function(x, lo, hi)


def _clip_method(lo, hi, x):
    """Call `x.clip(lo, hi)`."""
    return x.clip(lo, hi)


def _bad_mask(n: int, byte: int) -> np.ndarray:
    """Return a boolean array whose True elements are stored as `byte`, not 1.

    NumPy treats any non-zero byte as True. The mask kernels do not, so the
    binding must keep such a mask away from them.
    """
    raw = values("bool", n, arg="mask").astype(np.uint8) * byte
    return raw.view(np.bool_)


def _x_and_mask(dtype: str, n: int, x_variant: Variant, mask_maker: Callable[[], object]) -> tuple:
    """Build the two arguments of a mask case."""
    return (x_variant.build(values(dtype, n)), mask_maker())


def _mask_makers(n: int) -> Iterator[tuple[str, Callable[[], np.ndarray]]]:
    """Yield every boolean value class at one size, as (label, maker)."""
    for _, _, label, maker in value_arrays("bool", (n,), arg="mask"):
        yield label.split(" ", 1)[1], maker


def _compress_into(x, mask):
    """Run `np.compress(mask, x, out=o)` into an array of exactly the right size and return `o`."""
    out = np.empty(np.count_nonzero(mask), np.asarray(x).dtype)
    return np.compress(mask, x, out=out)


def _reshaped(maker: Callable[[], np.ndarray], order: str = "C") -> Callable[[], np.ndarray]:
    """Return a maker of the same values as `maker`'s, in COLUMNS columns of the given order.

    Args:
        maker: Builds a one-dimensional array whose length COLUMNS divides.
        order: "C" or "F", the memory order of the two-dimensional result.

    Returns:
        A function that builds the two-dimensional array.
    """
    return lambda: np.asarray(maker().reshape(-1, COLUMNS), order=order)


def _cube(maker: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    """Return a maker of the same values as `maker`'s, in three dimensions.

    Args:
        maker: Builds a one-dimensional array whose length CUBE_SHAPE's
            last two lengths divide.

    Returns:
        A function that builds the three-dimensional array, in C order.
    """
    return lambda: maker().reshape(CUBE_SHAPE).copy()


def _compress_shape_cases(dtype: str, kernel: str | None) -> Iterator[Case]:
    """Yield the cases of `np.compress` and `np.extract` with arrays of two dimensions.

    Without an axis both flatten `x` in C order, so a contiguous `x` of any
    shape is taken; `np.extract` flattens the mask too, and `np.compress`
    raises for one of two dimensions.

    Args:
        dtype: Element type of `x`, as a spec name.
        kernel: The compress kernel for that type, or None.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    x_flat = partial(values, dtype, n)
    mask_flat = partial(values, "bool", n, arg="mask")
    shapes = (
        ("x two-dimensional", _reshaped(x_flat), mask_flat, kernel, kernel),
        ("x two-dimensional, Fortran order", _reshaped(x_flat, "F"), mask_flat, None, None),
        ("both two-dimensional", _reshaped(x_flat), _reshaped(mask_flat), None, kernel),
        ("mask two-dimensional, Fortran order", x_flat, _reshaped(mask_flat, "F"), None, None),
        ("x zero-dimensional", lambda: np.array(x_flat()[0]), lambda: np.array(True), None, None),
    )
    for text, x_maker, mask_maker, by_compress, by_extract in shapes:
        make = lambda x_maker=x_maker, mask_maker=mask_maker: (x_maker(), mask_maker())
        yield Case("compress", f"np.compress(mask, x) {dtype} {text}", make, _compress_call, by_compress)
        yield Case("compress", f"np.extract(mask, x) {dtype} {text}", make, _extract_call, by_extract)


def _select_shape_cases(dtype: str, kernel: str | None) -> Iterator[Case]:
    """Yield the cases of `x[mask]` with arrays of more than one dimension.

    A mask of the shape of `x` selects elements in C order, which the kernel
    takes for contiguous arrays. A mask of fewer dimensions selects rows, and
    one of the same size and another shape is an error.

    Args:
        dtype: Element type of `x`, as a spec name.
        kernel: The compress kernel for that type, or None.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    x_flat = partial(values, dtype, n)
    mask_flat = partial(values, "bool", n, arg="mask")
    row_mask = partial(values, "bool", n // COLUMNS, arg="mask")
    shapes = (
        ("both two-dimensional", _reshaped(x_flat), _reshaped(mask_flat), kernel),
        ("both three-dimensional", _cube(x_flat), _cube(mask_flat), kernel),
        ("both two-dimensional, Fortran order", _reshaped(x_flat, "F"), _reshaped(mask_flat, "F"), None),
        ("x two-dimensional, mask over its rows", _reshaped(x_flat), row_mask, None),
        ("x two-dimensional, mask of the same size and one dimension", _reshaped(x_flat), mask_flat, None),
        ("both two-dimensional, shapes transposed", _reshaped(x_flat), lambda: mask_flat().reshape(COLUMNS, -1), None),
        ("both empty and two-dimensional", lambda: np.zeros((0, COLUMNS), np_dtype(dtype)),
         lambda: np.zeros((0, COLUMNS), np.bool_), kernel),
    )
    for text, x_maker, mask_maker, expect in shapes:
        make = lambda x_maker=x_maker, mask_maker=mask_maker: (x_maker(), mask_maker())
        yield Case("compress", f"x[mask] {dtype} {text}", make, lambda x, mask: x[mask], expect)


def _compress_call(x, mask):
    """Return `np.compress(mask, x)`."""
    return np.compress(mask, x)


def _extract_call(x, mask):
    """Return `np.extract(mask, x)`."""
    return np.extract(mask, x)


# Value classes of `x` in the sweep of mask selection over every size and
# mask: the ones whose bit patterns a kernel that computed on its elements
# would change.
_COMPRESS_VALUE_CLASSES = {"f32": ("random", "nan_some"), "f64": ("random", "nan_some"), "i64": ("random", "extremes")}


def compress_cases() -> Iterator[Case]:
    """Yield the cases of `x[mask]`, `np.compress(mask, x)` and `np.extract(mask, x)`.

    int64 elements are selected by the float64 kernel, which moves 64-bit
    elements without computing on them.
    """
    kernels = {"f32": "compress_f32", "f64": "compress_f64", "i64": "compress_f64"}
    select = Spelling("x[mask]", lambda x, mask: x[mask], any_shape=False)
    functions = (
        Spelling("np.compress(mask, x)", _compress_call),
        Spelling("np.extract(mask, x)", _extract_call),
    )
    others = (
        *functions,
        Spelling("np.compress(mask, x, axis=None, out=None)", lambda x, mask: np.compress(mask, x, axis=None, out=None)),
        Spelling("np.compress(condition=mask, a=x)", lambda x, mask: np.compress(condition=mask, a=x)),
        Spelling("np.extract(condition=mask, arr=x)", lambda x, mask: np.extract(condition=mask, arr=x)),
        Spelling("np.compress(mask, x, axis=0)", lambda x, mask: np.compress(mask, x, axis=0), NEVER),
        Spelling("np.compress(mask, x, out=o)", _compress_into, NEVER),
        Spelling("np.compress(mask.astype(int), x)", lambda x, mask: np.compress(np.asarray(mask).astype(int), x), NEVER),
        Spelling("np.extract(mask.astype(int), x)", lambda x, mask: np.extract(np.asarray(mask).astype(int), x), NEVER),
        Spelling("x[mask,]", lambda x, mask: x[mask,], NEVER),
        Spelling("x[mask][:3]", lambda x, mask: x[mask][:3], any_shape=False),
    )
    for dtype, kernel in kernels.items():
        for n in VALUE_SIZES:
            for mask_label, mask_maker in _mask_makers(n):
                for x_class in _COMPRESS_VALUE_CLASSES[dtype]:
                    make = partial(_values_and_mask, dtype, n, x_class, mask_maker)
                    yield Case("compress", f"x[mask] {dtype} n={n} x={x_class} {mask_label}", make, select.call, kernel)
                for spelling in functions:
                    make = partial(_values_and_mask, dtype, n, "random", mask_maker)
                    yield Case("compress", f"{spelling.text} {dtype} n={n} {mask_label}", make, spelling.call, kernel)
    for dtype in ("f32", "f64", "i64", "u64", "i32", "u8"):
        kernel = kernels.get(dtype)
        for n in (0, *LAYOUT_SIZES):
            plain_mask = partial(values, "bool", n, arg="mask")
            for spelling in (select, *others):
                for variant in variants_for(dtype, n):
                    if variant.dims != 1:
                        continue
                    yield Case("compress", f"{spelling.text} {dtype} n={n} x={variant.name}",
                               partial(_x_and_mask, dtype, n, variant, plain_mask), spelling.call,
                               expectation(kernel, spelling, variant, n))
                for variant in variants_for("bool", n):
                    if variant.dims != 1:
                        continue
                    mask_maker = partial(_variant_of, "bool", n, variant, "mask")
                    yield Case("compress", f"{spelling.text} {dtype} n={n} mask={variant.name}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, mask_maker), spelling.call,
                               expectation(kernel, spelling, variant, n))
            if n:
                for spelling in (select, *functions):
                    for byte in BAD_MASK_BYTES:
                        yield Case("compress", f"{spelling.text} {dtype} n={n} mask bytes are {byte}",
                                   partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_bad_mask, n, byte)),
                                   spelling.call, None)
                    for change, text in ((-1, "one short"), (1, "one long")):
                        other = partial(values, "bool", n + change, arg="mask")
                        yield Case("compress", f"{spelling.text} {dtype} n={n} mask {text}",
                                   partial(_x_and_mask, dtype, n, CONTIGUOUS, other), spelling.call, None)
        yield from _select_shape_cases(dtype, kernel)
        yield from _compress_shape_cases(dtype, kernel)


def _values_and_mask(dtype: str, n: int, x_class: str, mask_maker: Callable[[], np.ndarray]) -> tuple:
    """Build `x` from a value class and a mask from a maker."""
    return (values(dtype, n, x_class), mask_maker())


def _variant_of(dtype: str, n: int, variant: Variant, arg: str) -> object:
    """Build random values of `dtype` in a variant."""
    return variant.build(values(dtype, n, arg=arg))


def _assign(value, x, mask):
    """Run `x[mask] = value` and return `x`."""
    x[mask] = value
    return x


def _putmask(value, x, mask):
    """Run `np.putmask(x, mask, value)` and return `x`."""
    np.putmask(x, mask, value)
    return x


def _copyto(value, x, mask):
    """Run `np.copyto(x, value, where=mask)` and return `x`."""
    np.copyto(x, value, where=mask)
    return x


def _overlapping(n: int) -> tuple:
    """Build an array and a boolean mask that is a view of the array's own bytes."""
    x = np.zeros(n, np.float64)
    x.view(np.uint8)[:n:OVERLAP_EVERY] = 1
    return (x, x.view(np.bool_)[:n])


# Values assigned through a mask, with whether the kernel takes them: a
# Python float, or a Python int that converts to a double.
_FILL_VALUES: tuple[tuple[str, object, bool], ...] = (
    ("0.0", 0.0, True), ("1.5", 1.5, True), ("-0.0", -0.0, True), ("nan", float("nan"), True),
    ("-inf", float("-inf"), True), ("5e-324", 5e-324, True), ("0", 0, True), ("-3", -3, True),
    ("2**53", 2**53, True), ("2**53 + 1", 2**53 + 1, True), ("2**70", 2**70, True),
    ("10**400", 10**400, False), ("True", True, False), ("np.float64(2.0)", np.float64(2.0), False),
    ("np.float32(2.0)", np.float32(2.0), False), ("array", np.array(7.0), False), ("'1.5'", "1.5", False),
    ("None", None, False), ("1j", 1j, False),
)


def _copyto_casting(value, x, mask):
    """Run `np.copyto(x, value, casting='same_kind', where=mask)` and return `x`."""
    np.copyto(x, value, casting="same_kind", where=mask)
    return x


def _copyto_positional(value, x, mask):
    """Run `np.copyto(x, value, 'same_kind', mask)` and return `x`."""
    np.copyto(x, value, "same_kind", mask)
    return x


def _copyto_keywords(value, x, mask):
    """Run `np.copyto(dst=x, src=value, where=mask)` and return `x`."""
    np.copyto(dst=x, src=value, where=mask)
    return x


def _copyto_unmasked(value, x, mask):
    """Run `np.copyto(x, value)`, which has no mask, and return `x`."""
    np.copyto(x, value)
    return x


def _putmask_keywords(value, x, mask):
    """Run `np.putmask(x, mask=mask, values=value)` and return `x`."""
    np.putmask(x, mask=mask, values=value)
    return x


def _putmask_list(value, x, mask):
    """Run `np.putmask(x, mask, [value, value])`, which repeats a list of values, and return `x`."""
    np.putmask(x, mask, [value, value])
    return x


def _int_mask(maker: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    """Return a maker of the same mask as `maker`'s, stored as int64 0 and 1."""
    return lambda: maker().astype(np.int64)


# The ways to write a masked assignment of a scalar: as text, the function
# that makes it, and whether the kernel may take it. All but the first are
# function hooks.
_ASSIGNMENTS: tuple[tuple[str, Callable, bool], ...] = (
    ("x[mask] = v", _assign, True),
    ("np.putmask(x, mask, v)", _putmask, True),
    ("np.copyto(x, v, where=mask)", _copyto, True),
    ("np.putmask(x, mask=mask, values=v)", _putmask_keywords, True),
    ("np.copyto(dst=x, src=v, where=mask)", _copyto_keywords, True),
    ("np.copyto(x, v, 'same_kind', mask)", _copyto_positional, False),
    ("np.copyto(x, v, casting='same_kind', where=mask)", _copyto_casting, False),
    ("np.copyto(x, v)", _copyto_unmasked, False),
    ("np.putmask(x, mask, [v, v])", _putmask_list, False),
)
# The function forms among them that take the kernel under the same
# conditions as `x[mask] = v`.
_FUNCTION_ASSIGNMENTS = _ASSIGNMENTS[1:3]


def _maskfill_shape_cases(dtype: str, takes: str | None) -> Iterator[Case]:
    """Yield the masked assignments whose arrays have two dimensions or none.

    `np.putmask` needs only the sizes of `x` and the mask to match and reads
    both in C order. `np.copyto` broadcasts the mask against `x`, so the
    kernel takes it only for matching shapes. `x[mask] = v` is taken for a
    mask of the shape of `x`; a mask of another shape selects rows or is an
    error, and is NumPy's.

    Args:
        dtype: Element type of `x`, as a spec name.
        takes: The kernel for that type, or None.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    x_flat = partial(values, dtype, n)
    mask_flat = partial(values, "bool", n, arg="mask")
    shapes = (
        # (text, x, mask, taken by x[mask] = v, by putmask, by copyto)
        ("both two-dimensional", _reshaped(x_flat), _reshaped(mask_flat), takes, takes, takes),
        ("both three-dimensional", _cube(x_flat), _cube(mask_flat), takes, takes, takes),
        ("both two-dimensional, shapes transposed", _reshaped(x_flat), lambda: mask_flat().reshape(COLUMNS, -1),
         None, takes, None),
        ("x two-dimensional, mask over its rows", _reshaped(x_flat), partial(values, "bool", n // COLUMNS, arg="mask"),
         None, None, None),
        ("x two-dimensional, mask one-dimensional", _reshaped(x_flat), mask_flat, None, takes, None),
        ("x one-dimensional, mask two-dimensional", x_flat, _reshaped(mask_flat), None, takes, None),
        ("x two-dimensional, Fortran order", _reshaped(x_flat, "F"), _reshaped(mask_flat, "F"), None, None, None),
        ("x zero-dimensional", lambda: np.array(x_flat()[0]), lambda: np.array(True), None, None, None),
        ("x empty two-dimensional", lambda: np.zeros((0, COLUMNS), np_dtype(dtype)),
         lambda: np.zeros((0, COLUMNS), np.bool_), takes, takes, takes),
    )
    for text, x_maker, mask_maker, *expects in shapes:
        make = lambda x_maker=x_maker, mask_maker=mask_maker: (x_maker(), mask_maker())
        for (spelling, function, _), expect in zip(_ASSIGNMENTS[:3], expects):
            yield Case("maskfill", f"{spelling} with v=1.5 {dtype} {text}", make, partial(function, 1.5), expect)


def maskfill_cases() -> Iterator[Case]:
    """Yield the cases of `x[mask] = value`, `np.putmask` and `np.copyto(..., where=mask)`."""
    kernel = "maskfill_f64"
    for n in VALUE_SIZES:
        for mask_label, mask_maker in _mask_makers(n):
            for text, value, _ in _FILL_VALUES[:4]:
                yield Case("maskfill", f"x[mask] = {text} f64 n={n} {mask_label}",
                           partial(_values_and_mask, "f64", n, "random", mask_maker), partial(_assign, value), kernel)
            for spelling, function, _ in _FUNCTION_ASSIGNMENTS:
                yield Case("maskfill", f"{spelling} with v=1.5 f64 n={n} {mask_label}",
                           partial(_values_and_mask, "f64", n, "random", mask_maker), partial(function, 1.5), kernel)
    for dtype in ("f64", "f32", "i64"):
        takes = kernel if dtype == "f64" else None
        for n in (0, *LAYOUT_SIZES):
            plain_mask = partial(values, "bool", n, arg="mask")
            for spelling, function, by_kernel in _ASSIGNMENTS:
                for text, value, taken in _FILL_VALUES:
                    yield Case("maskfill", f"{spelling} with v={text} {dtype} n={n}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, plain_mask), partial(function, value),
                               takes if taken and by_kernel else None)
            for spelling, function, _ in _ASSIGNMENTS[:3]:
                for variant in variants_for(dtype, n):
                    if variant.dims != 1 or variant.name == "list":
                        continue
                    expect = takes if variant.plain and variant.writeable else None
                    if takes and (not variant.predictable or (n == 0 and not variant.plain)):
                        expect = ANY
                    yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} x={variant.name}",
                               partial(_x_and_mask, dtype, n, variant, plain_mask), partial(function, 1.5), expect)
                for variant in variants_for("bool", n):
                    if variant.dims != 1:
                        continue
                    expect = takes if variant.plain else (ANY if takes and n == 0 else None)
                    yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask={variant.name}",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_variant_of, "bool", n, variant, "mask")),
                               partial(function, 1.5), expect)
                yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask=int64",
                           partial(_x_and_mask, dtype, n, CONTIGUOUS, _int_mask(plain_mask)), partial(function, 1.5), None)
                if n:
                    for byte in BAD_MASK_BYTES:
                        yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask bytes are {byte}",
                                   partial(_x_and_mask, dtype, n, CONTIGUOUS, partial(_bad_mask, n, byte)),
                                   partial(function, 1.5), None)
                    short = partial(values, "bool", n - 1, arg="mask")
                    yield Case("maskfill", f"{spelling} with v=1.5 {dtype} n={n} mask one short",
                               partial(_x_and_mask, dtype, n, CONTIGUOUS, short), partial(function, 1.5), None)
        yield from _maskfill_shape_cases(dtype, takes)
    for spelling, function, _ in _ASSIGNMENTS[:3]:
        yield Case("maskfill", f"{spelling} with v=1.5 where mask is a view of x", partial(_overlapping, ORDINARY_N),
                   partial(function, 1.5), None)


def nonzero_cases() -> Iterator[Case]:
    """Yield the cases of `nonzero` on booleans."""
    spellings = (
        Spelling("mask.nonzero()", lambda m: m.nonzero(), any_shape=False),
        Spelling("np.nonzero(mask)", np.nonzero, any_shape=False, normalises=True),
        Spelling("np.flatnonzero(mask)", np.flatnonzero, normalises=True),
        Spelling("np.argwhere(mask)", np.argwhere, MAYBE),
        Spelling("np.where(mask)", np.where, any_shape=False),
        Spelling("np.where(mask, 1.0, 0.0)", lambda m: np.where(m, 1.0, 0.0), NEVER),
        Spelling("np.where(mask, mask, mask)", lambda m: np.where(m, m, m), NEVER),
    )
    # The kernel is not called for an empty mask's sake: it takes it and returns nothing.
    yield from unary_cases("nonzero", {"bool": "nonzero_bool"}, ("u8", "f64", "i64"), spellings)
    # The spellings that would take the kernel for a clean mask.
    taking = (spellings[0], spellings[1], spellings[4])
    for n in LAYOUT_SIZES:
        for byte in BAD_MASK_BYTES:
            for spelling in taking:
                yield Case("nonzero", f"{spelling.text} n={n} mask bytes are {byte}",
                           lambda n=n, byte=byte: (_bad_mask(n, byte),), spelling.call, None)


def cumsum_cases(inexact: bool) -> Iterator[Case]:
    """Yield the cases of `cumsum`.

    Args:
        inexact: False for the int64 and bool kernels and the float kernels' absence;
            True for the float kernels, which are compared within their
            specs' tolerance and only exist under install(inexact=True).

    Yields:
        The cases.
    """
    spellings = (
        Spelling("x.cumsum()", lambda x: x.cumsum()),
        Spelling("np.cumsum(x)", np.cumsum, normalises=True),
        Spelling("x.cumsum(None)", lambda x: x.cumsum(None)),
        Spelling("x.cumsum(axis=None, dtype=None)", lambda x: x.cumsum(axis=None, dtype=None)),
        Spelling("np.cumsum(x, axis=0)", lambda x: np.cumsum(x, axis=0), NEVER),
        Spelling("x.cumsum(0)", lambda x: x.cumsum(0), NEVER),
        Spelling("x.cumsum(dtype=np.float64)", lambda x: x.cumsum(dtype=np.float64), NEVER),
        Spelling("x.cumsum(out=o)", lambda x: x.cumsum(out=np.empty(np.size(x), np.asarray(x).dtype)), NEVER),
        Spelling("np.add.accumulate(x)", np.add.accumulate, NEVER),
    )
    if inexact:
        yield from unary_cases("cumsum", {"f32": "cumsum_f32", "f64": "cumsum_f64"}, (), spellings,
                               empty_takes=False, tolerance=True)
    else:
        yield from unary_cases("cumsum", {"i64": "cumsum_i64", "bool": "cumsum_bool"}, ("f32", "f64", "i32", "u8"),
                               spellings, empty_takes=False)
        # A mask holding a byte other than 0 or 1 counts once per element in NumPy.
        for n, byte, spelling in itertools.product(LAYOUT_SIZES, BAD_MASK_BYTES, spellings[:2]):
            yield Case("cumsum", f"{spelling.text} bool n={n} mask bytes are {byte}",
                       partial(_made, partial(_bad_mask, n, byte)), spelling.call, None)


def astype_cases() -> Iterator[Case]:
    """Yield the cases of `astype` from uint8 to float32."""
    spellings = (
        Spelling("x.astype(np.float32)", lambda x: x.astype(np.float32)),
        Spelling("x.astype('float32')", lambda x: x.astype("float32")),
        Spelling("x.astype('f4')", lambda x: x.astype("f4")),
        Spelling("x.astype(np.dtype('float32'))", lambda x: x.astype(np.dtype("float32"))),
        Spelling("x.astype(dtype=np.float32)", lambda x: x.astype(dtype=np.float32), NEVER),
        Spelling("x.astype(np.float32, copy=False)", lambda x: x.astype(np.float32, copy=False), NEVER),
        Spelling("x.astype(np.float32, order='F')", lambda x: x.astype(np.float32, order="F"), NEVER),
        Spelling("x.astype(np.float64)", lambda x: x.astype(np.float64), NEVER),
        Spelling("x.astype('>f4')", lambda x: x.astype(">f4"), NEVER),
        Spelling("x.astype(float)", lambda x: x.astype(float), NEVER),
        Spelling("np.asarray(x, dtype=np.float32)", lambda x: np.asarray(x, dtype=np.float32), NEVER),
    )
    yield from unary_cases("astype", {"u8": "cast_u8_f32"}, ("i8", "bool", "u16", "f64"), spellings)


def _late_bad_mask(n: int, byte: int) -> np.ndarray:
    """Return a boolean array of zeros whose last element is stored as `byte`.

    The binding checks a mask block by block; this one is clean until its end.
    """
    raw = np.zeros(n, np.uint8)
    raw[-1] = byte
    return raw.view(np.bool_)


def sum_cases() -> Iterator[Case]:
    """Yield the cases of `sum` on uint8 and on booleans, which the same kernel serves."""
    spellings = (
        Spelling("np.sum(x)", np.sum),
        Spelling("x.sum()", lambda x: x.sum()),
        Spelling("np.sum(x, axis=0)", lambda x: np.sum(x, axis=0), NEVER),
        Spelling("np.sum(x, axis=None)", lambda x: np.sum(x, axis=None), NEVER),
        Spelling("x.sum(dtype=np.uint8)", lambda x: x.sum(dtype=np.uint8), NEVER),
        Spelling("np.sum(x, initial=5)", lambda x: np.sum(x, initial=5), NEVER),
        Spelling("np.sum(x, keepdims=True)", lambda x: np.sum(x, keepdims=True), NEVER),
        Spelling("np.add.reduce(x)", np.add.reduce, NEVER),
        Spelling("np.sum(a=x)", lambda x: np.sum(a=x), NEVER),
    )
    yield from unary_cases("sum", {"u8": "sum_u8", "bool": "sum_u8"}, ("i8", "u16", "f64"), spellings)
    # A mask holding a byte other than 0 or 1 counts once per element in NumPy.
    for n, byte, spelling in itertools.product(LAYOUT_SIZES, BAD_MASK_BYTES, spellings[:2]):
        yield Case("sum", f"{spelling.text} bool n={n} mask bytes are {byte}",
                   partial(_made, partial(_bad_mask, n, byte)), spelling.call, None)
        yield Case("sum", f"{spelling.text} bool n={SUM_BOOL_LATE_N} last mask byte is {byte}",
                   partial(_made, partial(_late_bad_mask, SUM_BOOL_LATE_N, byte)), spelling.call, None)


def count_nonzero_cases() -> Iterator[Case]:
    """Yield the cases of `count_nonzero`."""
    spellings = (
        Spelling("np.count_nonzero(x)", np.count_nonzero),
        Spelling("np.count_nonzero(x, axis=0)", lambda x: np.count_nonzero(x, axis=0), NEVER),
        Spelling("np.count_nonzero(x, axis=None)", lambda x: np.count_nonzero(x, axis=None), NEVER),
        Spelling("np.count_nonzero(x, keepdims=True)", lambda x: np.count_nonzero(x, keepdims=True), NEVER),
        Spelling("np.count_nonzero(a=x)", lambda x: np.count_nonzero(a=x), NEVER),
    )
    kernels = {"i64": "count_nonzero_i64", "f64": "count_nonzero_f64"}
    yield from unary_cases("count_nonzero", kernels, ("i32", "f32", "bool", "u8"), spellings)
    yield from unusual_nan_cases("count_nonzero", {"f64": "count_nonzero_f64"}, "np.count_nonzero(x)", np.count_nonzero)


def _histogram_sets() -> Iterator[tuple[str, int, float, float]]:
    """Yield every bin count and range the histogram kernel's spec lists."""
    k = spec("histogram_f64")
    for set_name in k.scalar_set_names:
        scalars = k.scalar_values(set_name)
        yield set_name, int(scalars["nbins"]), scalars["lo"], scalars["hi"]


def histogram_cases() -> Iterator[Case]:
    """Yield the cases of `histogram`.

    The kernel takes a contiguous float64 array, an integer bin count and a
    range of two plain numbers, or no range. NumPy raises for a range that is
    not finite or whose bin edges are not strictly increasing, and widens an
    empty one; all of those must be left to it.
    """
    kernel = "histogram_f64"
    for set_name, bins, lo, hi in _histogram_sets():
        sizes = VALUE_SIZES if set_name == DEFAULT_SCALARS else FEW_SIZES
        for _, _, label, maker in value_arrays("f64", sizes):
            yield Case("histogram", f"np.histogram(x, {bins}, ({lo!r}, {hi!r})) {label}", partial(_made, maker),
                       lambda x, bins=bins, lo=lo, hi=hi: np.histogram(x, bins, (lo, hi)), kernel)
    finite = ("random", "equal", "asc", "desc", "ties", "zeros", "denormal")
    for n in FEW_SIZES:
        for cls in judge_cases.classes("f64")[0]:
            # Without a range NumPy uses the data's minimum and maximum. It
            # raises if they are not finite and widens the range if they are
            # equal, so only data with two distinct finite extremes is taken.
            data = values("f64", n, cls)
            usable = cls in finite and n > 1 and np.isfinite(data).all() and data.min() < data.max()
            for text, call in (("np.histogram(x)", np.histogram), ("np.histogram(x, 7)", lambda x: np.histogram(x, 7))):
                yield Case("histogram", f"{text} n={n} x={cls}", partial(_made, partial(values, "f64", n, cls)),
                           call, kernel if usable else None)
    spellings = (
        Spelling("np.histogram(x, 50, (-3.0, 3.0))", lambda x: np.histogram(x, 50, (-3.0, 3.0))),
        Spelling("np.histogram(x, bins=50, range=(-3.0, 3.0))", lambda x: np.histogram(x, bins=50, range=(-3.0, 3.0))),
        Spelling("np.histogram(a=x, bins=5, range=(0, 1))", lambda x: np.histogram(a=x, bins=5, range=(0, 1))),
        Spelling("np.histogram(x, range=(-1, 1))", lambda x: np.histogram(x, range=(-1, 1))),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), None, None)", lambda x: np.histogram(x, 50, (-3.0, 3.0), None, None)),
        Spelling("np.histogram(x, 50, [-3.0, 3.0])", lambda x: np.histogram(x, 50, [-3.0, 3.0]), NEVER),
        Spelling("np.histogram(x, 50, (np.float64(-3), 3.0))", lambda x: np.histogram(x, 50, (np.float64(-3), 3.0)), NEVER),
        Spelling("np.histogram(x, np.int64(50), (-3.0, 3.0))", lambda x: np.histogram(x, np.int64(50), (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), density=True)", lambda x: np.histogram(x, 50, (-3.0, 3.0), density=True), NEVER),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), density=False)", lambda x: np.histogram(x, 50, (-3.0, 3.0), density=False), NEVER),
        Spelling("np.histogram(x, 50, (-3.0, 3.0), weights=w)", lambda x: np.histogram(x, 50, (-3.0, 3.0), weights=np.ones(np.shape(x))), NEVER),
        Spelling("np.histogram(x, 'auto')", lambda x: np.histogram(x, "auto"), NEVER),
        Spelling("np.histogram(x, [-1.0, 0.0, 2.0])", lambda x: np.histogram(x, [-1.0, 0.0, 2.0]), NEVER),
        Spelling("np.histogram(x, 0, (-3.0, 3.0))", lambda x: np.histogram(x, 0, (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, -4, (-3.0, 3.0))", lambda x: np.histogram(x, -4, (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, True, (-3.0, 3.0))", lambda x: np.histogram(x, True, (-3.0, 3.0)), NEVER),
        Spelling("np.histogram(x, 5, (3.0, -3.0)) reversed", lambda x: np.histogram(x, 5, (3.0, -3.0)), NEVER),
        Spelling("np.histogram(x, 5, (1.0, 1.0)) empty", lambda x: np.histogram(x, 5, (1.0, 1.0)), NEVER),
        Spelling("np.histogram(x, 5, (nan, 1.0))", lambda x: np.histogram(x, 5, (float("nan"), 1.0)), NEVER),
        Spelling("np.histogram(x, 5, (0.0, inf))", lambda x: np.histogram(x, 5, (0.0, float("inf"))), NEVER),
        Spelling("np.histogram(x, 5, (0, 2**70))", lambda x: np.histogram(x, 5, (0, 2**70)), NEVER),
        Spelling("np.histogram(x, 1000, (1e16, 1e16 + 2)) too many bins", lambda x: np.histogram(x, 1000, (1e16, 1e16 + 2)), NEVER),
        Spelling("np.histogram(x, 5, (0.0, 1.0, 2.0))", lambda x: np.histogram(x, 5, (0.0, 1.0, 2.0)), NEVER),
        Spelling("np.histogram(x, bins=5, bogus=1)", lambda x: np.histogram(x, bins=5, bogus=1), NEVER),
    )
    for dtype in ("f64", "f32", "i64"):
        for n, spelling in itertools.product((0, *LAYOUT_SIZES), spellings):
            for variant in variants_for(dtype, n):
                yield Case("histogram", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call,
                           expectation(kernel if dtype == "f64" else None, spelling, variant, n))


def _bincount_spellings(sweep_extra: bool) -> tuple[Spelling, ...]:
    """Return the spellings of `np.bincount` that the uint8, int16 and int64 cases share.

    Args:
        sweep_extra: Whether the spellings with `minlength` that take a
            kernel are swept over every value class too. The uint8 cases do
            that; for int16 and int64 the sweep over code ranges has its own
            values of `minlength`.

    Returns:
        The spellings. The ones that take a kernel do so on a plain array
        whose elements are all in the range NumPy accepts.
    """
    return (
        Spelling("np.bincount(x)", np.bincount, any_shape=False),
        Spelling("np.bincount(x, minlength=0)", lambda x: np.bincount(x, minlength=0), any_shape=False,
                 sweep_values=sweep_extra),
        Spelling("np.bincount(x, minlength=10)", lambda x: np.bincount(x, minlength=10), any_shape=False,
                 sweep_values=sweep_extra),
        Spelling("np.bincount(x, minlength=256)", lambda x: np.bincount(x, minlength=256), any_shape=False,
                 sweep_values=sweep_extra),
        Spelling("np.bincount(x, minlength=300)", lambda x: np.bincount(x, minlength=300), any_shape=False,
                 sweep_values=sweep_extra),
        Spelling("np.bincount(x, None, 256)", lambda x: np.bincount(x, None, 256), any_shape=False,
                 sweep_values=sweep_extra),
        Spelling("np.bincount(x, weights=None)", lambda x: np.bincount(x, weights=None), any_shape=False,
                 sweep_values=sweep_extra),
        Spelling("np.bincount(x, weights=w)", lambda x: np.bincount(x, weights=np.ones(np.shape(x))), NEVER),
        Spelling("np.bincount(x, minlength=-1)", lambda x: np.bincount(x, minlength=-1), NEVER),
        Spelling("np.bincount(x, minlength=None)", lambda x: np.bincount(x, minlength=None), NEVER),
        Spelling("np.bincount(x, minlength=5.0)", lambda x: np.bincount(x, minlength=5.0), NEVER),
        Spelling("np.bincount(x, minlength=True)", lambda x: np.bincount(x, minlength=True), NEVER),
        Spelling("np.bincount(x, minlength=np.int64(5))", lambda x: np.bincount(x, minlength=np.int64(5)), NEVER),
        Spelling("np.bincount(x=x)", lambda x: np.bincount(x=x), NEVER),
        Spelling("np.bincount(x, None, 0, 1)", lambda x: np.bincount(x, None, 0, 1), NEVER),
    )


# The int16 and int64 kernels of bincount, by element type.
_BINCOUNT_KERNELS = {"i16": "bincount_i16", "i64": "bincount_i64"}
# Element types np.bincount accepts or refuses that have no kernel: they stay
# with NumPy. "longlong" is a 64-bit integer type NumPy keeps apart from
# int64 on Linux, where int64 is C's long.
_BINCOUNT_NEIGHBOURS = ("i32", "u16", "u64", "i8", "bool")
# Counters the int16 and int64 kernels are always given; above it a call is
# taken only when it has no more counters than elements. The binding's
# BINCOUNT_FREE_BINS.
BINCOUNT_FREE_BINS = 65536
# Most counts a case lets NumPy's result hold: 32 MB of them.
BINCOUNT_CASE_LIMIT = 1 << 22
# A code that makes NumPy's result longer than BINCOUNT_FREE_BINS and longer
# than any array of the cases, but small enough to run.
BINCOUNT_FAR = 1_000_000
# The largest int64, for which NumPy's length overflows.
_INT64_MAX = np.iinfo(np.int64).max
# Element counts of the sweep over code ranges: the kernels change path at
# 8,192 elements, and from it the interpreter lock is released.
BINCOUNT_SIZES = (1, 2, 16, 100, 1000, 8191, 8192, 8193, 20000, 100003)
# A large code among small ones: inside the counters every call is given, and
# for int64 one past them.
BINCOUNT_LARGE = 30000
BINCOUNT_BEYOND = 70000
# The ranges of codes of the sweep: name -> the codes are drawn from
# [0, span). The kernels have a path for at most 64 counters, used from 8,192
# elements, and one for up to 1,024 where the data repeats, as "0..3" does
# with a `minlength` above 64.
_BINCOUNT_RANGES = {
    "zeros": 1,
    "0..3": 4,
    "0..63": 64,
    "0..64": 65,
    "0..999": 1000,
    "0..1023": 1024,
    "0..1024": 1025,
    "0..4999": 5000,
}


def _bincount_takes(x: np.ndarray, minlength: int = 0) -> bool:
    """Say whether the binding takes `np.bincount(x, minlength=...)` for a plain int16 or int64 x.

    It is the binding's rule written again in NumPy: at least one element,
    none negative, the largest not the largest int64, and no more counters
    than BINCOUNT_FREE_BINS or the element count, whichever is more.

    Args:
        x: The codes, contiguous and one-dimensional.
        minlength: The `minlength` given, 0 when none was.

    Returns:
        True when the kernel must take the call.
    """
    if len(x) == 0 or int(x.min()) < 0 or int(x.max()) == _INT64_MAX:
        return False
    bins = max(int(x.max()) + 1, minlength)
    return bins <= max(BINCOUNT_FREE_BINS, len(x))


def _bincount_expect(dtype: str, maker: Callable[[], np.ndarray], minlength: int = 0) -> str | None:
    """Return the kernel for a plain call on the array `maker` builds, or None when the call is NumPy's."""
    return _BINCOUNT_KERNELS[dtype] if _bincount_takes(maker(), minlength) else None


def _codes(dtype: str, n: int, span: int, large: int | None = None, last: bool = False) -> np.ndarray:
    """Build codes for np.bincount.

    Args:
        dtype: "i16" or "i64".
        n: Element count.
        span: The codes are drawn from [0, span).
        large: A code placed in the middle, or None.
        last: Whether the largest code, `span - 1`, is placed last as well,
            so that the element that sizes the result is the last read.

    Returns:
        A contiguous one-dimensional array.
    """
    x = np.random.default_rng([SEED, n, span]).integers(0, span, n).astype(np_dtype(dtype))
    if large is not None and n:
        x[n // 2] = large
    if last and n:
        x[-1] = span - 1
    return x


def _with_negative(dtype: str, n: int, position: int) -> np.ndarray:
    """Build codes in [0, 64) with -1 at `position`: NumPy raises ValueError."""
    x = _codes(dtype, n, 64)
    x[position] = -1
    return x


def _minlength_call(minlength: int | None) -> Callable[[np.ndarray], np.ndarray]:
    """Return a call of np.bincount with `minlength` given, or without it when None."""
    if minlength is None:
        return np.bincount
    return lambda x: np.bincount(x, minlength=minlength)


def _bincount_range_cases(dtype: str) -> Iterator[Case]:
    """Yield np.bincount on int16 or int64 codes in several ranges, with and without `minlength`.

    Each range is taken at every size in BINCOUNT_SIZES, plain and with
    BINCOUNT_LARGE in the middle or its largest code last, and with
    `minlength` absent, 0, below the largest code plus one and above it.

    Args:
        dtype: "i16" or "i64".

    Yields:
        The cases.
    """
    for (name, span), n in itertools.product(_BINCOUNT_RANGES.items(), BINCOUNT_SIZES):
        shapes = (("", None, False), (f" one {BINCOUNT_LARGE}", BINCOUNT_LARGE, False), (" largest last", None, True))
        for suffix, large, last in shapes:
            maker = partial(_codes, dtype, n, span, large, last)
            for minlength in (None, 0, span // 2, span + 100):
                text = "np.bincount(x)" if minlength is None else f"np.bincount(x, minlength={minlength})"
                yield Case("bincount", f"{text} {dtype} n={n} x={name}{suffix}", partial(_made, maker),
                           _minlength_call(minlength), _bincount_expect(dtype, maker, minlength or 0))


def _bincount_value_cases(dtype: str) -> Iterator[Case]:
    """Yield np.bincount on every value class of the judge, and of the kernel's spec, at every size.

    Most of the judge's integer classes hold negative values, for which NumPy
    raises; the binding must leave those calls to it, and the same error must
    come back.

    Args:
        dtype: "i16" or "i64".

    Yields:
        The cases.
    """
    for n, cls, label, maker in value_arrays(dtype):
        if _not_runnable(maker()):
            continue
        yield Case("bincount", f"np.bincount(x) {dtype} {label}", partial(_made, maker), np.bincount,
                   _bincount_expect(dtype, maker))
    own = spec(_BINCOUNT_KERNELS[dtype]).args[0].classes
    for (cls, build), n in itertools.product(own.items(), VALUE_SIZES):
        maker = partial(_spec_class, build, dtype, n)
        if _not_runnable(maker()):
            continue
        yield Case("bincount", f"np.bincount(x) {dtype} n={n} x={cls}", partial(_made, maker), np.bincount,
                   _bincount_expect(dtype, maker))


def _not_runnable(x: np.ndarray) -> bool:
    """Say whether plain NumPy's np.bincount on `x` cannot be run as a case.

    Two kinds. NumPy 2.5.5 sizes its result as the largest element plus one;
    for the largest int64, with no negative element, that overflows and NumPy
    writes outside the array it allocated, and the process aborts or crashes.
    And a result of more than BINCOUNT_CASE_LIMIT counts, which the value
    classes with large elements ask for (up to 2^62): NumPy takes it from
    memory the system zeroes only when touched, and the check, which hashes
    every byte of a result, would touch all of it. The binding leaves both to
    NumPy, unchanged; smaller results with far more counters than elements are
    among the guard cases.

    Args:
        x: The codes.

    Returns:
        True when the case must not be made.
    """
    if len(x) == 0 or int(x.min()) < 0:
        return False
    return int(x.max()) == _INT64_MAX or int(x.max()) >= BINCOUNT_CASE_LIMIT


def _spec_class(build: Callable[[np.ndarray], np.ndarray], dtype: str, n: int) -> np.ndarray:
    """Build an array from one of a spec's own value classes, which derive their values from the default array."""
    return build(judge_cases.base(dtype, n, SEED, "x"))


def _bincount_guard_cases(dtype: str) -> Iterator[Case]:
    """Yield the calls the int16 and int64 binding must leave to NumPy, and the edges of what it takes.

    A negative element first, in the middle and last; more counters than the
    binding gives (BINCOUNT_BEYOND, and a large `minlength`) with few
    elements, where NumPy's zeroed memory costs it nothing, and with more
    elements than counters, which is taken. For int64 the largest int64,
    which NumPy itself does not survive (_not_runnable), is not among them.

    Args:
        dtype: "i16" or "i64".

    Yields:
        The cases.
    """
    for n in (1, 16, 1000, 8193, 20000):
        for position in sorted({0, n // 2, n - 1}):
            maker = partial(_with_negative, dtype, n, position)
            yield Case("bincount", f"np.bincount(x) {dtype} n={n} x=-1@{position}", partial(_made, maker),
                       np.bincount, None)
    for n, minlength in ((16, BINCOUNT_FREE_BINS), (16, BINCOUNT_FREE_BINS + 1), (100003, 100003),
                         (100003, 100004)):
        maker = partial(_codes, dtype, n, 64)
        yield Case("bincount", f"np.bincount(x, minlength={minlength}) {dtype} n={n} x=0..63", partial(_made, maker),
                   _minlength_call(minlength), _bincount_expect(dtype, maker, minlength))
    if dtype != "i64":
        return
    for n, large in ((16, BINCOUNT_BEYOND), (100003, BINCOUNT_BEYOND), (16, BINCOUNT_FAR), (100003, BINCOUNT_FAR)):
        maker = partial(_codes, dtype, n, 64, large)
        yield Case("bincount", f"np.bincount(x) {dtype} n={n} x=0..63 one {large}", partial(_made, maker),
                   np.bincount, _bincount_expect(dtype, maker))


def _bincount_layout_cases() -> Iterator[Case]:
    """Yield every spelling of np.bincount in every layout, on codes in [0, 64).

    The int16 and int64 kernels and their neighbouring element types, at
    LAYOUT_SIZES and on an empty array, which is NumPy's.

    Yields:
        The cases.
    """
    spellings = _bincount_spellings(sweep_extra=False)
    for dtype in (*_BINCOUNT_KERNELS, *_BINCOUNT_NEIGHBOURS):
        kernel = _BINCOUNT_KERNELS.get(dtype)
        for n, spelling in itertools.product(LAYOUT_SIZES, spellings):
            for variant in variants_for(dtype, n):
                maker = partial(_codes, "bool" if dtype == "bool" else dtype, n, 2 if dtype == "bool" else 64)
                yield Case("bincount", f"{spelling.text} {dtype} n={n} x=0..63 layout={variant.name}",
                           partial(_variant_codes, maker, variant), spelling.call,
                           expectation(kernel, spelling, variant, n))
        for spelling in spellings:
            yield Case("bincount", f"{spelling.text} {dtype} n=0", partial(_built, dtype, 0, CONTIGUOUS),
                       spelling.call, None)
    for n in LAYOUT_SIZES:
        yield Case("bincount", f"np.bincount(x) longlong n={n} x=0..63",
                   lambda n=n: (_codes("i64", n, 64).astype(np.longlong),), np.bincount, None)


def _variant_codes(maker: Callable[[], np.ndarray], variant: Variant) -> tuple:
    """Build the one argument of a bincount layout case: codes from `maker` in a variant."""
    return (variant.build(maker()),)


def bincount_cases() -> Iterator[Case]:
    """Yield the cases of `bincount`: uint8, int16 and int64, and the element types that stay with NumPy.

    uint8 is taken whatever its values. int16 and int64 are taken when no
    element is negative and the result is not far longer than the array; see
    the binding's bincount_codes.
    """
    spellings = _bincount_spellings(sweep_extra=True)
    yield from unary_cases("bincount", {"u8": "bincount_u8"}, (), spellings)
    for value in (0, 1, 254, 255):
        yield Case("bincount", f"np.bincount(x) every byte is {value}", lambda value=value: (np.full(ORDINARY_N, value, np.uint8),),
                   np.bincount, "bincount_u8")
    for dtype in _BINCOUNT_KERNELS:
        yield from _bincount_value_cases(dtype)
        yield from _bincount_range_cases(dtype)
        yield from _bincount_guard_cases(dtype)
    yield from _bincount_layout_cases()


def _isin_x(spread: str, n: int) -> np.ndarray:
    """Build the array searched by an isin case.

    Args:
        spread: "narrow" for values from 0 to 1,000, where a small set of test
            values matches often; "wide" for the whole int64 range.
        n: Element count.

    Returns:
        A contiguous int64 array.
    """
    rng = np.random.default_rng([SEED, n])
    if spread == "narrow":
        return rng.integers(0, 1000, n)
    info = np.iinfo(np.int64)
    x = rng.integers(info.min, info.max, n, endpoint=True)
    x[::3] = info.max
    x[1::7] = info.min
    return x


# Test values for isin, with whether the kernel takes them: one to eight
# int64 values, as an array or as a list or tuple of Python ints.
_INT64 = np.iinfo(np.int64)
_ISIN_TESTS: tuple[tuple[str, Callable[[], object], bool], ...] = (
    ("[5]", lambda: [5], True),
    ("[1, 2, 3]", lambda: [1, 2, 3], True),
    ("(1, 2, 3)", lambda: (1, 2, 3), True),
    ("[3, 3, 3]", lambda: [3, 3, 3], True),
    ("eight values", lambda: [1, 50, 999, 0, 7, 500, 501, 2], True),
    ("eight wide values", lambda: [_INT64.max, _INT64.min, 0, -1, 1, 2**40, -(2**40), 65535], True),
    ("[30000, 59990, 5, 100, 1000, 20000, 40000, -10]", lambda: [30000, 59990, 5, 100, 1000, 20000, 40000, -10], True),
    ("[-10, 59990, 5, 100, 1000, 20000, 40000, 30000]", lambda: [-10, 59990, 5, 100, 1000, 20000, 40000, 30000], True),
    ("array of 2", lambda: np.array([4, 999]), True),
    ("array of 8", lambda: np.arange(8) * 100, True),
    ("array of 9", lambda: np.arange(9) * 100, False),
    ("nine values", lambda: list(range(9)), False),
    ("[]", lambda: [], False),
    ("empty array", lambda: np.array([], np.int64), False),
    ("[1, 2.5]", lambda: [1, 2.5], False),
    ("[1, True]", lambda: [1, True], False),
    ("[2**63]", lambda: [2**63], False),
    ("[-2**63 - 1]", lambda: [-(2**63) - 1], False),
    ("int32 array", lambda: np.array([1, 2], np.int32), False),
    ("strided array", lambda: np.arange(16)[::4], False),
    ("two-dimensional array", lambda: np.array([[1, 2], [3, 4]]), False),
    ("scalar 5", lambda: 5, False),
    ("set", lambda: {1, 2}, False),
)


def isin_cases() -> Iterator[Case]:
    """Yield the cases of `isin` on int64."""
    kernel = "isin_i64"
    for n, spread in itertools.product(VALUE_SIZES, ("narrow", "wide")):
        for text, test, taken in _ISIN_TESTS:
            yield Case("isin", f"np.isin(x, {text}) n={n} x={spread}", lambda n=n, spread=spread, test=test: (_isin_x(spread, n), test()),
                       np.isin, kernel if taken else None)
    spellings = (
        Spelling("np.isin(x, [1, 2, 3])", lambda x: np.isin(x, [1, 2, 3])),
        Spelling("np.isin(x, [1, 2, 3], invert=True)", lambda x: np.isin(x, [1, 2, 3], invert=True), NEVER),
        Spelling("np.isin(x, [1, 2, 3], assume_unique=True)", lambda x: np.isin(x, [1, 2, 3], assume_unique=True), NEVER),
        Spelling("np.isin(x, [1, 2, 3], kind='sort')", lambda x: np.isin(x, [1, 2, 3], kind="sort"), NEVER),
        Spelling("np.isin(element=x, test_elements=[1, 2, 3])", lambda x: np.isin(element=x, test_elements=[1, 2, 3]), NEVER),
    )
    for dtype in ("i64", "i32", "u8", "f64"):
        for n, spelling in itertools.product((0, *LAYOUT_SIZES), spellings):
            for variant in variants_for(dtype, n):
                yield Case("isin", f"{spelling.text} {dtype} n={n} layout={variant.name}",
                           partial(_built, dtype, n, variant), spelling.call,
                           expectation(kernel if dtype == "i64" else None, spelling, variant, n))


# ---------------------------------------------------- searchsorted and interp

# Lengths of the sorted array, and numbers of needles, either side of every
# boundary at which the search kernels change path: 16, 64 and 128 needles at
# once, and one much longer array.
TABLE_SIZES = (0, 1, 2, 15, 16, 17, 63, 64, 65, 1000, 4097)
NEEDLE_COUNTS = (0, 1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 1000)
# Length from which `searchsorted_f64` takes its path for arrays beyond the
# cache, and a length above it.
LARGE_TABLE_N = 1_048_576
BEYOND_LARGE_TABLE_N = LARGE_TABLE_N + 3
# Numbers of needles tried against the two long arrays: one, and either side
# of the group of 128 that the path for long arrays works in.
LARGE_TABLE_NEEDLES = (1, 64, 127, 128, 129, 1000)
# (value class of the array, value class of the needles) tried against the
# two long arrays.
LARGE_TABLE_PAIRS = (("random", "nan_some"), ("random", "hits"), ("equal", "hits"))
# Number of needles in the cases that vary the spelling or the layout of a
# search: more than one group of 64, with a leftover.
SPELLING_NEEDLES = 100
# (sample points, queries) of the interp cases whose sample points hold a NaN.
NAN_POINT_SHAPES = ((1, SPELLING_NEEDLES), (2, SPELLING_NEEDLES), (ORDINARY_N, SPELLING_NEEDLES), (ORDINARY_N, ORDINARY_N))
# Value classes for the sorted array of a search: ordinary values, repeated
# values, infinities at both ends, and NaNs, which NumPy sorts last.
TABLE_CLASSES = ("random", "ties", "inf_mix", "nan_some")
# Value classes for the needles.
NEEDLE_CLASSES = ("random", "ties", "inf_mix", "nan_some", "nan_all", "extremes", "zeros")
# Value classes for the sample points of interp. A NaN is not among them:
# NumPy's answer then depends on which elements its search probes.
POINT_CLASSES = ("random", "ties", "inf_mix", "equal", "zeros")
# Value classes for the values at the sample points.
FP_CLASSES = ("random", "nan_some", "inf_mix", "extremes", "equal")
# Mirrors INTERP_MAX_POINTS_PER_QUERY in the binding: with more sample points
# than this per query, np.interp is left to NumPy.
INTERP_MAX_POINTS_PER_QUERY = 64


def _table(n: int, cls: str = judge_cases.RANDOM) -> np.ndarray:
    """Build a sorted float64 array of `n` elements from a value class, NaNs last."""
    return np.sort(values("f64", n, cls, arg="a"))


def _hits(table: np.ndarray, count: int) -> np.ndarray:
    """Build `count` needles that are elements of `table`, so every search is an exact hit.

    Args:
        table: The array the needles are drawn from.
        count: Number of needles.

    Returns:
        A float64 array of `count` elements; zeros when the table is empty.
    """
    if len(table) == 0:
        return np.zeros(count)
    picks = np.random.default_rng([SEED, len(table), count]).integers(0, len(table), count)
    return table[picks].copy()


def _search_arguments(n: int, table_class: str, count: int, needle_class: str) -> tuple:
    """Build the sorted array and the needles of a searchsorted case.

    Args:
        n: Length of the sorted array.
        table_class: Value class of the sorted array.
        count: Number of needles.
        needle_class: Value class of the needles, or "hits" for needles taken
            from the array itself.

    Returns:
        (sorted array, needles).
    """
    table = _table(n, table_class)
    if needle_class == "hits":
        return table, _hits(table, count)
    return table, values("f64", count, needle_class, arg="v")


def _search_layout_expectation(kernel: str, variant: Variant, n: int, any_shape: bool) -> str | None:
    """Say whether a search kernel takes a call in which one array is a variant.

    Args:
        kernel: The kernel the call would reach.
        variant: How the array in question is laid out.
        n: Its element count.
        any_shape: Whether the kernel accepts the array with two or more
            dimensions: true for needles and queries, false for the sorted
            array and the sample points.

    Returns:
        The kernel name, None, or ANY.
    """
    if not variant.predictable:
        return ANY
    if variant.plain and (variant.dims == 1 or (variant.dims > 1 and any_shape)):
        return kernel
    # NumPy flags every empty array as contiguous, whatever its strides say.
    return ANY if n == 0 else None


def _search_right(a, v):
    """Call `np.searchsorted(a, v, side='right')`.

    Args:
        a: The sorted array.
        v: The needles.

    Returns:
        What NumPy, or the kernel in its place, returns.
    """
    return np.searchsorted(a, v, side="right")


# The two sides of a search: (the call as a case names it, the call, the kernel
# that takes it). The sides differ only for a needle that equals an element
# and for a NaN needle, so the "hits" class is what tells them apart.
SEARCH_SIDES = (
    ("np.searchsorted(a, v)", np.searchsorted, "searchsorted_f64"),
    ("np.searchsorted(a, v, side='right')", _search_right, "searchsorted_right_f64"),
)


def _search_grid_cases(text: str, search: Callable, kernel: str) -> Iterator[Case]:
    """Yield the cases of one side of a search over sizes and value classes.

    Args:
        text: The call, as the cases name it.
        search: Makes the call from (sorted array, needles).
        kernel: The kernel expected to take every case.

    Returns:
        An iterator over the cases.
    """
    for n, count in itertools.product(TABLE_SIZES, NEEDLE_COUNTS):
        # Every class of needle on an ordinary array, then every class of array.
        pairs = [("random", needle_class) for needle_class in (*NEEDLE_CLASSES, "hits")]
        pairs += [(table_class, needle_class) for table_class in TABLE_CLASSES[1:] for needle_class in ("random", "hits")]
        for table_class, needle_class in pairs:
            yield Case("searchsorted", f"{text} a={table_class} n={n} v={needle_class} needles={count}",
                       partial(_search_arguments, n, table_class, count, needle_class), search, kernel)
    # The long arrays get needles with NaNs among them, and needles that are
    # elements, so that the path for long arrays is seen to compare the right
    # way through the binding as well as in the judge, which draws needles
    # from the array at every size. An array of one repeated value is there for the
    # first step of the search, which only shows a wrong comparison when
    # elements equal to the needle lie on both sides of a[p - 1], p being the
    # largest power of two not above the length.
    for n in (LARGE_TABLE_N, BEYOND_LARGE_TABLE_N):
        for count, (table_class, needle_class) in itertools.product(LARGE_TABLE_NEEDLES, LARGE_TABLE_PAIRS):
            yield Case("searchsorted", f"{text} a={table_class} n={n} v={needle_class} needles={count}",
                       partial(_search_arguments, n, table_class, count, needle_class), search, kernel)


def searchsorted_cases() -> Iterator[Case]:
    """Yield the cases of `searchsorted` on float64, for both sides."""
    kernel, right = "searchsorted_f64", "searchsorted_right_f64"
    for side in SEARCH_SIDES:
        yield from _search_grid_cases(*side)

    n, count = ORDINARY_N, SPELLING_NEEDLES
    arguments = partial(_search_arguments, n, "random", count, "random")
    spellings = (
        ("a.searchsorted(v)", lambda a, v: a.searchsorted(v), kernel),
        ("np.searchsorted(a, v, 'left')", lambda a, v: np.searchsorted(a, v, "left"), kernel),
        ("np.searchsorted(a, v, side='left', sorter=None)", lambda a, v: np.searchsorted(a, v, side="left", sorter=None), kernel),
        ("a.searchsorted(v=v)", lambda a, v: a.searchsorted(v=v), kernel),
        ("np.searchsorted(a, v, side='right')", lambda a, v: np.searchsorted(a, v, side="right"), right),
        ("np.searchsorted(a, v, 'right')", lambda a, v: np.searchsorted(a, v, "right"), right),
        ("a.searchsorted(v, 'right')", lambda a, v: a.searchsorted(v, "right"), right),
        ("a.searchsorted(v=v, side='right')", lambda a, v: a.searchsorted(v=v, side="right"), right),
        ("np.searchsorted(a, v, side='right', sorter=None)", lambda a, v: np.searchsorted(a, v, side="right", sorter=None), right),
        ("np.searchsorted(a, v, sorter=np.arange(len(a)))", lambda a, v: np.searchsorted(a, v, sorter=np.arange(len(a))), None),
        ("np.searchsorted(a, v, side='right', sorter=np.arange(len(a)))", lambda a, v: np.searchsorted(a, v, side="right", sorter=np.arange(len(a))), None),
        ("np.digitize(v, a)", lambda a, v: np.digitize(v, a), right),
        ("np.digitize(v, a, right=True)", lambda a, v: np.digitize(v, a, right=True), kernel),
        # Needles that are bin edges: the two sides of digitize differ at every one.
        ("np.digitize(a, a)", lambda a, v: np.digitize(a, a), right),
        ("np.digitize(a, a, right=True)", lambda a, v: np.digitize(a, a, right=True), kernel),
        # Decreasing bins: digitize searches them reversed, which is a view
        # that is not contiguous...
        ("np.digitize(v, a[::-1].copy())", lambda a, v: np.digitize(v, a[::-1].copy()), None),
        ("np.digitize(v, a[::-1].copy(), right=True)", lambda a, v: np.digitize(v, a[::-1].copy(), right=True), None),
        # ...unless the bins were a reversed view to begin with: reversed
        # again they are the contiguous array.
        ("np.digitize(v, a[::-1])", lambda a, v: np.digitize(v, a[::-1]), right),
        ("np.digitize(v, a[::-1], right=True)", lambda a, v: np.digitize(v, a[::-1], right=True), kernel),
        ("np.searchsorted(a, 0.25, side='right')", lambda a, v: np.searchsorted(a, 0.25, side="right"), right),
        ("np.searchsorted(a, a[3], side='right')", lambda a, v: np.searchsorted(a, float(a[3]), side="right"), right),
        ("np.searchsorted(a, float('nan'), side='right')", lambda a, v: np.searchsorted(a, float("nan"), side="right"), right),
        ("np.searchsorted(a, a, side='right')", lambda a, v: np.searchsorted(a, a, side="right"), right),
        ("np.searchsorted(a, 0.25)", lambda a, v: np.searchsorted(a, 0.25), kernel),
        ("np.searchsorted(a, float('nan'))", lambda a, v: np.searchsorted(a, float("nan")), kernel),
        ("np.searchsorted(a, 1)", lambda a, v: np.searchsorted(a, 1), kernel),
        ("np.searchsorted(a, 2**60 + 1)", lambda a, v: np.searchsorted(a, 2**60 + 1), None),
        ("np.searchsorted(a, True)", lambda a, v: np.searchsorted(a, True), None),
        ("np.searchsorted(a, np.float64(0.25))", lambda a, v: np.searchsorted(a, np.float64(0.25)), None),
        ("np.searchsorted(a, np.array(0.25))", lambda a, v: np.searchsorted(a, np.array(0.25)), None),
        ("np.searchsorted(a, v.tolist())", lambda a, v: np.searchsorted(a, v.tolist()), None),
        ("np.searchsorted(a, v.astype(np.float32))", lambda a, v: np.searchsorted(a, v.astype(np.float32)), None),
        ("np.searchsorted(a, v.astype(np.int64))", lambda a, v: np.searchsorted(a, v.astype(np.int64)), None),
        ("np.searchsorted(a.astype(np.float32), v)", lambda a, v: np.searchsorted(a.astype(np.float32), v), None),
        ("np.searchsorted(a.astype(np.int64), v)", lambda a, v: np.searchsorted(a.astype(np.int64), v), None),
        ("np.searchsorted(a, a)", lambda a, v: np.searchsorted(a, a), kernel),
        ("np.searchsorted(a, v, side='up')", lambda a, v: np.searchsorted(a, v, side="up"), None),
        ("np.searchsorted(a, v, side=None)", lambda a, v: np.searchsorted(a, v, side=None), None),
    )
    for text, call, expect in spellings:
        yield Case("searchsorted", f"{text} n={n} needles={count}", arguments, call, expect)

    for (text, search, kernel), size in itertools.product(SEARCH_SIDES, (0, *LAYOUT_SIZES)):
        for variant in variants_for("f64", size):
            # A list has no method of its own: NumPy converts it and calls the
            # method of the array it made.
            expect = ANY if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=False)
            yield Case("searchsorted", f"{text} n={size} needles={count} a layout={variant.name}",
                       lambda size=size, variant=variant: (variant.build(_table(size)), values("f64", count, arg="v")),
                       search, expect)
            expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=True)
            yield Case("searchsorted", f"{text} n={n} needles={size} v layout={variant.name}",
                       lambda size=size, variant=variant: (_table(n), variant.build(values("f64", size, arg="v"))),
                       search, expect)


def _interp_arguments(n: int, count: int, x_class: str, xp_class: str, fp_class: str) -> tuple:
    """Build the queries, sample points and values of an interp case.

    Args:
        n: Number of sample points.
        count: Number of queries.
        x_class: Value class of the queries, or "hits" for queries that are
            sample points.
        xp_class: Value class of the sample points, which are sorted.
        fp_class: Value class of the values at the sample points.

    Returns:
        (x, xp, fp).
    """
    xp = np.sort(values("f64", n, xp_class, arg="xp"))
    x = _hits(xp, count) if x_class == "hits" else values("f64", count, x_class, arg="x")
    return x, xp, values("f64", n, fp_class, arg="fp")


def _interp_with_nan(n: int, count: int, position: int) -> tuple:
    """Build an interp call whose sample points hold a NaN at `position`."""
    x, xp, fp = _interp_arguments(n, count, "random", "random", "random")
    xp[position] = np.nan
    return x, xp, fp


def _interp_takes(n: int, count: int) -> bool:
    """Say whether the binding gives an interp call to the kernel by its two lengths.

    Args:
        n: Number of sample points.
        count: Number of queries.

    Returns:
        True when there are few enough sample points per query for the pass
        over them to be worth making.
    """
    return n // INTERP_MAX_POINTS_PER_QUERY <= count


def interp_cases() -> Iterator[Case]:
    """Yield the cases of `interp` on float64."""
    kernel = "interp_f64"
    for n, count in itertools.product(TABLE_SIZES[1:], NEEDLE_COUNTS):
        expect = kernel if _interp_takes(n, count) else None
        triples = [(x_class, "random", "random") for x_class in (*NEEDLE_CLASSES, "hits")]
        triples += [(x_class, xp_class, "random") for xp_class in POINT_CLASSES[1:] for x_class in ("random", "hits")]
        triples += [("random", "random", fp_class) for fp_class in FP_CLASSES[1:]]
        triples += [("hits", "ties", "inf_mix"), ("inf_mix", "inf_mix", "inf_mix"), ("nan_some", "ties", "nan_some")]
        for x_class, xp_class, fp_class in triples:
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={count} x={x_class} xp={xp_class} fp={fp_class}",
                       partial(_interp_arguments, n, count, x_class, xp_class, fp_class), np.interp, expect)
    for n, count in NAN_POINT_SHAPES:
        for position in sorted({0, n // 2, n - 1}):
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={count} xp has a NaN at {position}",
                       partial(_interp_with_nan, n, count, position), np.interp, None)

    n, count = ORDINARY_N, SPELLING_NEEDLES
    arguments = partial(_interp_arguments, n, count, "random", "random", "random")
    spellings = (
        ("np.interp(x, xp, fp, None, None, None)", lambda x, xp, fp: np.interp(x, xp, fp, None, None, None), kernel),
        ("np.interp(x, xp, fp, left=None, right=None, period=None)", lambda x, xp, fp: np.interp(x, xp, fp, left=None, right=None, period=None), kernel),
        ("np.interp(x=x, xp=xp, fp=fp)", lambda x, xp, fp: np.interp(x=x, xp=xp, fp=fp), kernel),
        ("np.interp(x, xp, fp, left=0.0)", lambda x, xp, fp: np.interp(x, xp, fp, left=0.0), None),
        ("np.interp(x, xp, fp, right=0.0)", lambda x, xp, fp: np.interp(x, xp, fp, right=0.0), None),
        ("np.interp(x, xp, fp, period=1.0)", lambda x, xp, fp: np.interp(x, xp, fp, period=1.0), None),
        ("np.interp(0.25, xp, fp)", lambda x, xp, fp: np.interp(0.25, xp, fp), None),
        ("np.interp(x.tolist(), xp, fp)", lambda x, xp, fp: np.interp(x.tolist(), xp, fp), None),
        ("np.interp(x[:5], xp, fp)", lambda x, xp, fp: np.interp(x[:5], xp, fp), None),
        ("np.interp(x[:0], xp, fp)", lambda x, xp, fp: np.interp(x[:0], xp, fp), None),
        ("np.interp(x, xp, fp[:-1])", lambda x, xp, fp: np.interp(x, xp, fp[:-1]), None),
        ("np.interp(x, xp[:0], fp[:0])", lambda x, xp, fp: np.interp(x, xp[:0], fp[:0]), None),
        ("np.interp(x, xp, fp.astype(np.float32))", lambda x, xp, fp: np.interp(x, xp, fp.astype(np.float32)), None),
        ("np.interp(x, xp, fp.astype(np.complex128))", lambda x, xp, fp: np.interp(x, xp, fp.astype(np.complex128)), None),
        ("np.interp(x, xp.astype(np.float32), fp)", lambda x, xp, fp: np.interp(x, xp.astype(np.float32), fp), None),
        ("np.interp(x.astype(np.float32), xp, fp)", lambda x, xp, fp: np.interp(x.astype(np.float32), xp, fp), None),
        ("np.interp(x.astype(np.int64), xp, fp)", lambda x, xp, fp: np.interp(x.astype(np.int64), xp, fp), None),
        ("np.interp(x, xp, xp)", lambda x, xp, fp: np.interp(x, xp, xp), kernel),
        ("np.interp(xp, xp, fp)", lambda x, xp, fp: np.interp(xp, xp, fp), kernel),
    )
    for text, call, expect in spellings:
        yield Case("interp", f"{text} n={n} queries={count}", arguments, call, expect)

    for size in (0, *LAYOUT_SIZES):
        for variant in variants_for("f64", size):
            expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=True)
            yield Case("interp", f"np.interp(x, xp, fp) n={n} queries={size} x layout={variant.name}",
                       lambda size=size, variant=variant: (variant.build(values("f64", size, arg="x")), _table(n), values("f64", n, arg="fp")),
                       np.interp, expect if _interp_takes(n, size) else None)
            if size == 0:
                continue
            for which in ("xp", "fp"):
                expect = None if variant.name == "list" else _search_layout_expectation(kernel, variant, size, any_shape=False)
                yield Case("interp", f"np.interp(x, xp, fp) n={size} queries={size} {which} layout={variant.name}",
                           lambda size=size, variant=variant, which=which: (
                               values("f64", size, arg="x"),
                               variant.build(_table(size)) if which == "xp" else _table(size),
                               variant.build(values("f64", size, arg="fp")) if which == "fp" else values("f64", size, arg="fp")),
                           np.interp, expect)


# ------------------------------------------------------------------- where

# Element types np.where(c, a, b) is taken for when `a` and `b` are both
# arrays of it, with the kernel that takes it: int64 elements are moved by the
# float64 kernel as bit patterns.
_WHERE_KERNELS = {"f64": "where_f64", "i64": "where_f64"}
# Element types of two arrays that np.where leaves to NumPy.
_WHERE_NEIGHBOURS = ("f32", "u64", "i32", "u8", "bool")
# Argument name of the condition, which gives it random data of its own.
_CONDITION = "mask"


def _signalling_nan() -> np.float64:
    """Return an np.float64 holding a signalling NaN with a payload, which the result must keep bit for bit."""
    unsigned, signalling, _ = _UNUSUAL_NANS[8]
    return np.array([signalling], unsigned).view(np.float64)[0]


# Scalars given as `a` or `b` beside a float64 array: (text, maker, taken).
# Taken are those that leave NumPy's result float64 and that it stores
# exactly: a Python float, a Python int a double holds exactly, an
# np.float64. The rest change the result's type or value, or raise.
_WHERE_SCALARS: tuple[tuple[str, Callable[[], object], bool], ...] = (
    ("1.5", lambda: 1.5, True),
    ("nan", lambda: float("nan"), True),
    ("-0.0", lambda: -0.0, True),
    ("-inf", lambda: float("-inf"), True),
    ("0", lambda: 0, True),
    ("-2**53", lambda: -(2**53), True),
    ("np.float64(-0.0)", lambda: np.float64(-0.0), True),
    ("np.float64(signalling NaN)", _signalling_nan, True),
    ("2**53 + 1", lambda: 2**53 + 1, False),
    ("10**400", lambda: 10**400, False),
    ("True", lambda: True, False),
    ("np.float32(1.5)", lambda: np.float32(1.5), False),
    ("np.int64(3)", lambda: np.int64(3), False),
    ("1j", lambda: 1j, False),
    ("np.array(1.5)", lambda: np.array(1.5), False),
    ("None", lambda: None, False),
)


def _where(c, a, b):
    """Return `np.where(c, a, b)`."""
    return np.where(c, a, b)


def _built_each(*makers: Callable[[], object]) -> tuple:
    """Build the arguments of a case, one maker per argument, in order."""
    return tuple(maker() for maker in makers)


def _given(value: object) -> object:
    """Return `value`: a maker of an argument that is the same object every time, such as a constant."""
    return value


def _unusual_nans_reversed(n: int) -> np.ndarray:
    """Return the float64 array of `unusual_nans`, back to front, so that its NaNs fall elsewhere."""
    return unusual_nans("f64", n)[::-1].copy()


def _where_value_cases() -> Iterator[Case]:
    """Yield `np.where(c, a, b)` over every value class at every size, all three plain arrays.

    Both sides get the same value class (with data of their own), so the
    sweep covers each class on each side. Then every class of the condition,
    the NaNs the judge does not make, and each scalar taken, on each side.
    """
    for dtype, kernel in _WHERE_KERNELS.items():
        for n in VALUE_SIZES:
            condition = partial(values, "bool", n, arg=_CONDITION)
            sides = zip(value_arrays(dtype, (n,), "a"), value_arrays(dtype, (n,), "b"))
            for (_, _, label, a_maker), (_, _, _, b_maker) in sides:
                yield Case("where", f"np.where(c, a, b) {dtype} {label} and b alike",
                           partial(_built_each, condition, a_maker, b_maker), _where, kernel)
            a_maker = partial(values, dtype, n, arg="a")
            b_maker = partial(values, dtype, n, arg="b")
            for mask_label, mask_maker in _mask_makers(n):
                yield Case("where", f"np.where(c, a, b) {dtype} n={n} c={mask_label.split('=', 1)[1]}",
                           partial(_built_each, mask_maker, a_maker, b_maker), _where, kernel)
    for n in VALUE_SIZES:
        condition = partial(values, "bool", n, arg=_CONDITION)
        yield Case("where", f"np.where(c, a, b) f64 n={n} a and b=signalling and negative NaNs",
                   partial(_built_each, condition, partial(unusual_nans, "f64", n), partial(_unusual_nans_reversed, n)),
                   _where, "where_f64")
        array = partial(values, "f64", n, arg="a")
        for text, scalar, taken in _WHERE_SCALARS:
            if not taken:
                continue
            yield Case("where", f"np.where(c, a, {text}) f64 n={n}", partial(_built_each, condition, array, scalar),
                       _where, "where_f64")
            yield Case("where", f"np.where(c, {text}, b) f64 n={n}", partial(_built_each, condition, scalar, array),
                       _where, "where_f64")


def _where_layout_cases() -> Iterator[Case]:
    """Yield `np.where(c, a, b)` with each of the three arguments in every layout, the others plain.

    Only plain one-dimensional arrays are taken. A two-dimensional array beside
    one-dimensional ones is NumPy's to broadcast or refuse, and a
    zero-dimensional condition is broadcast.
    """
    spelling = Spelling("np.where(c, a, b)", _where, any_shape=False)
    for dtype in (*_WHERE_KERNELS, *_WHERE_NEIGHBOURS):
        kernel = _WHERE_KERNELS.get(dtype)
        for n in (0, *LAYOUT_SIZES):
            plain = {
                "c": partial(values, "bool", n, arg=_CONDITION),
                "a": partial(values, dtype, n, arg="a"),
                "b": partial(values, dtype, n, arg="b"),
            }
            for arg, arg_dtype, name in (("c", "bool", _CONDITION), ("a", dtype, "a"), ("b", dtype, "b")):
                for variant in variants_for(arg_dtype, n):
                    makers = {**plain, arg: partial(_variant_of, arg_dtype, n, variant, name)}
                    yield Case("where", f"np.where(c, a, b) {dtype} n={n} {arg}={variant.name}",
                               partial(_built_each, makers["c"], makers["a"], makers["b"]), _where,
                               expectation(kernel, spelling, variant, n))


def _where_shape_cases(dtype: str, kernel: str) -> Iterator[Case]:
    """Yield `np.where(c, a, b)` with arrays of more than one dimension, and shapes that broadcast.

    Arrays of one shape and C order are taken whatever their number of
    dimensions. Shapes that differ are NumPy's to broadcast or refuse.

    Args:
        dtype: Element type of `a` and `b`, as a spec name.
        kernel: The kernel that takes the call for that type.

    Yields:
        The cases.
    """
    n = ORDINARY_N
    c = partial(values, "bool", n, arg=_CONDITION)
    a = partial(values, dtype, n, arg="a")
    b = partial(values, dtype, n, arg="b")
    shapes = (
        ("all two-dimensional", _reshaped(c), _reshaped(a), _reshaped(b), kernel),
        ("all three-dimensional", _cube(c), _cube(a), _cube(b), kernel),
        ("all two-dimensional, Fortran order", _reshaped(c, "F"), _reshaped(a, "F"), _reshaped(b, "F"), None),
        ("c two-dimensional, Fortran order", _reshaped(c, "F"), _reshaped(a), _reshaped(b), None),
        ("b over the rows of two-dimensional c and a", _reshaped(c), _reshaped(a), lambda: b()[:COLUMNS].copy(), None),
        ("c of one column against two-dimensional a and b", lambda: c()[: n // COLUMNS].reshape(-1, 1).copy(),
         _reshaped(a), _reshaped(b), None),
        ("c of one row against two-dimensional a and b", lambda: c()[:COLUMNS].copy(), _reshaped(a), _reshaped(b), None),
        ("a and b of one element", c, lambda: a()[:1].copy(), lambda: b()[:1].copy(), None),
        ("c two-dimensional, a and b of the same size and one dimension", _reshaped(c), a, b, None),
        ("all empty and two-dimensional", lambda: np.zeros((0, COLUMNS), np.bool_),
         lambda: np.zeros((0, COLUMNS), np_dtype(dtype)), lambda: np.zeros((0, COLUMNS), np_dtype(dtype)), kernel),
    )
    for text, c_maker, a_maker, b_maker, expect in shapes:
        yield Case("where", f"np.where(c, a, b) {dtype} {text}", partial(_built_each, c_maker, a_maker, b_maker),
                   _where, expect)
    if dtype == "f64":
        yield Case("where", "np.where(c, a, nan) f64 c and a two-dimensional",
                   partial(_built_each, _reshaped(c), _reshaped(a), partial(_given, float("nan"))), _where, kernel)
        yield Case("where", "np.where(c, 0, b) f64 c and b three-dimensional",
                   partial(_built_each, _cube(c), partial(_given, 0), _cube(b)), _where, kernel)
        yield Case("where", "np.where(c, a, nan) f64 c two-dimensional, a of one dimension",
                   partial(_built_each, _reshaped(c), a, partial(_given, float("nan"))), _where, None)


# Calls of np.where on plain float64 arrays, written other ways: (text,
# call, taken). The call receives the condition and two float64 arrays.
_WHERE_SPELLINGS: tuple[tuple[str, Callable, bool], ...] = (
    ("np.where(c, a, a)", lambda c, a, b: np.where(c, a, a), True),
    ("np.where(c, b, a)", lambda c, a, b: np.where(c, b, a), True),
    ("np.where(c, x=a, y=b)", lambda c, a, b: np.where(c, x=a, y=b), False),
    ("np.where(c, a)", lambda c, a, b: np.where(c, a), False),
    ("np.where(c, a, b, b)", lambda c, a, b: np.where(c, a, b, b), False),
    ("np.where(c.astype(int), a, b)", lambda c, a, b: np.where(c.astype(np.int64), a, b), False),
    ("np.where(c.view(np.uint8), a, b)", lambda c, a, b: np.where(c.view(np.uint8), a, b), False),
    ("np.where(c, c, c)", lambda c, a, b: np.where(c, c, c), False),
    ("np.where(c, 1.0, 0.0)", lambda c, a, b: np.where(c, 1.0, 0.0), False),
    ("np.where(c, 0, nan)", lambda c, a, b: np.where(c, 0, float("nan")), False),
    ("np.where(c, a, b.astype(np.float32))", lambda c, a, b: np.where(c, a, b.astype(np.float32)), False),
    ("np.where(c, a.astype(np.float32), b)", lambda c, a, b: np.where(c, a.astype(np.float32), b), False),
    ("np.where(c, a.astype(np.int64), b)", lambda c, a, b: np.where(c, a.astype(np.int64), b), False),
    ("np.where(c, a.astype(np.int64), 0)", lambda c, a, b: np.where(c, a.astype(np.int64), 0), False),
    ("np.where(c, 1.5, b.astype(np.int64))", lambda c, a, b: np.where(c, 1.5, b.astype(np.int64)), False),
    ("np.where(c, a, b.astype('>f8'))", lambda c, a, b: np.where(c, a, b.astype(">f8")), False),
)


def where_cases() -> Iterator[Case]:
    """Yield the cases of `np.where(c, a, b)`.

    float64 and int64 pairs are taken, the int64 ones by the float64 kernel
    moving bit patterns, and so are float64 arrays beside a scalar that keeps
    the result float64. Everything else, and a condition holding a byte other
    than 0 or 1, is NumPy's.
    """
    yield from _where_value_cases()
    yield from _where_layout_cases()
    for dtype, kernel in _WHERE_KERNELS.items():
        yield from _where_shape_cases(dtype, kernel)
    for n in (0, *LAYOUT_SIZES):
        condition = partial(values, "bool", n, arg=_CONDITION)
        floats = partial(values, "f64", n, arg="a")
        others = partial(values, "f64", n, arg="b")
        for text, call, taken in _WHERE_SPELLINGS:
            yield Case("where", f"{text} f64 n={n}", partial(_built_each, condition, floats, others), call,
                       "where_f64" if taken else None)
        for dtype in _WHERE_KERNELS:
            array = partial(values, dtype, n, arg="a")
            for text, scalar, taken in _WHERE_SCALARS:
                # Beside an int64 array no scalar is taken: a float changes the
                # result's type, and an int is NumPy's to fit.
                expect = "where_f64" if taken and dtype == "f64" else None
                yield Case("where", f"np.where(c, a, {text}) {dtype} n={n}",
                           partial(_built_each, condition, array, scalar), _where, expect)
                yield Case("where", f"np.where(c, {text}, b) {dtype} n={n}",
                           partial(_built_each, condition, scalar, array), _where, expect)
        if n:
            for byte in BAD_MASK_BYTES:
                bad = partial(_bad_mask, n, byte)
                for dtype in _WHERE_KERNELS:
                    yield Case("where", f"np.where(c, a, b) {dtype} n={n} c bytes are {byte}",
                               partial(_built_each, bad, partial(values, dtype, n, arg="a"),
                                       partial(values, dtype, n, arg="b")), _where, None)
                yield Case("where", f"np.where(c, a, nan) f64 n={n} c bytes are {byte}",
                           partial(_built_each, bad, floats, partial(_given, float("nan"))), _where, None)
                # With a scalar the binding checks the condition block by block;
                # this one is clean until its last element.
                late = partial(_late_bad_mask, n, byte)
                yield Case("where", f"np.where(c, nan, b) f64 n={n} c bytes are {byte} at the end only",
                           partial(_built_each, late, partial(_given, float("nan")), others), _where, None)
                yield Case("where", f"np.where(c, a, b) f64 n={n} c bytes are {byte} at the end only",
                           partial(_built_each, late, floats, others), _where, None)


# ------------------------------------------------------------- accumulate

# Bit patterns of the short float64 cases of the running maximum. NumPy
# computes an accumulation of two elements with its operands the other way
# round from a longer one, which shows only where two elements compare equal
# or are both NaN: two zeros of opposite sign, and two NaNs with different
# payloads. Each pattern is also run with a third element.
_POS_ZERO = 0x0000000000000000
_NEG_ZERO = 0x8000000000000000
_QUIET_NAN = 0x7FF8000000000000
_PAYLOAD_NAN = 0xFFF8000000000123
_OTHER_NAN = 0x7FF80000000ABCDE
_ONE = 0x3FF0000000000000
_NEG_INF = 0xFFF0000000000000
_SHORT_FLOATS: tuple[tuple[int, ...], ...] = (
    (_POS_ZERO, _NEG_ZERO), (_NEG_ZERO, _POS_ZERO), (_NEG_ZERO, _NEG_ZERO), (_POS_ZERO, _POS_ZERO),
    (_QUIET_NAN, _PAYLOAD_NAN), (_PAYLOAD_NAN, _QUIET_NAN), (_QUIET_NAN, _ONE), (_ONE, _QUIET_NAN),
    (_PAYLOAD_NAN, _NEG_ZERO), (_NEG_INF, _QUIET_NAN), (_ONE, _ONE),
    (_POS_ZERO, _NEG_ZERO, _POS_ZERO), (_NEG_ZERO, _POS_ZERO, _NEG_ZERO), (_NEG_ZERO, _POS_ZERO, _POS_ZERO),
    (_QUIET_NAN, _PAYLOAD_NAN, _OTHER_NAN), (_PAYLOAD_NAN, _NEG_ZERO, _POS_ZERO), (_QUIET_NAN, _ONE, _OTHER_NAN),
)
# The int64 values of the short cases: two or three elements in each order,
# with the extremes of the type.
_INT64_MIN, _INT64_MAX = int(np.iinfo(np.int64).min), int(np.iinfo(np.int64).max)
_SHORT_INTS: tuple[tuple[int, ...], ...] = (
    (5, 3), (3, 5), (5, 5), (_INT64_MIN, _INT64_MAX), (_INT64_MAX, _INT64_MIN), (-1, 0, -1), (_INT64_MAX, -1, 0),
)
# Element count of the case whose only signalling NaN is its last element:
# above the size from which the binding releases the interpreter lock.
ACCUMULATE_LATE_N = 20000


def _maximum_accumulate(x):
    """Return `np.maximum.accumulate(x)`."""
    return np.maximum.accumulate(x)


def _fmax_accumulate(x):
    """Return `np.fmax.accumulate(x)`."""
    return np.fmax.accumulate(x)


def _accumulate_spellings(ufunc: np.ufunc) -> tuple[Spelling, ...]:
    """Return the ways of writing an accumulation with `ufunc`, np.maximum or np.fmax.

    The kernels take a one-dimensional array with `axis` absent, 0 or -1 and
    no `dtype` or `out`, however the arguments are passed. A contiguous array
    of two dimensions is NumPy's, even with the default axis.

    Args:
        ufunc: `np.maximum` or `np.fmax`.

    Returns:
        The spellings, the ones that take a kernel first.
    """
    name = f"np.{ufunc.__name__}.accumulate"
    plain = _maximum_accumulate if ufunc is np.maximum else _fmax_accumulate
    return (
        Spelling(f"{name}(x)", plain, any_shape=False),
        Spelling(f"{name}(x, 0)", lambda x: ufunc.accumulate(x, 0), any_shape=False, sweep_values=False),
        Spelling(f"{name}(x, axis=-1)", lambda x: ufunc.accumulate(x, axis=-1), any_shape=False,
                 sweep_values=False),
        Spelling(f"{name}(array=x)", lambda x: ufunc.accumulate(array=x), any_shape=False, sweep_values=False),
        Spelling(f"{name}(x, 0, None, None)", lambda x: ufunc.accumulate(x, 0, None, None), any_shape=False,
                 sweep_values=False),
        Spelling(f"{name}(x, axis=0, dtype=None, out=None)",
                 lambda x: ufunc.accumulate(x, axis=0, dtype=None, out=None), any_shape=False, sweep_values=False),
        Spelling(f"{name}(x, axis=None)", lambda x: ufunc.accumulate(x, axis=None), NEVER),
        Spelling(f"{name}(x, axis=1)", lambda x: ufunc.accumulate(x, axis=1), NEVER),
        Spelling(f"{name}(x, axis=np.int64(0))", lambda x: ufunc.accumulate(x, axis=np.int64(0)), NEVER),
        Spelling(f"{name}(x, dtype=x.dtype)", lambda x: ufunc.accumulate(x, dtype=np.asarray(x).dtype), NEVER),
        Spelling(f"{name}(x, out=o)",
                 lambda x: ufunc.accumulate(x, out=np.empty(np.shape(x), np.asarray(x).dtype)), NEVER),
    )


# Accumulations through the same hook that no kernel serves.
_OTHER_ACCUMULATIONS = (
    Spelling("np.minimum.accumulate(x)", np.minimum.accumulate, NEVER),
    Spelling("np.add.accumulate(x)", np.add.accumulate, NEVER),
    Spelling("np.fmin.accumulate(x)", np.fmin.accumulate, NEVER),
)


def _from_bits(patterns: tuple[int, ...]) -> np.ndarray:
    """Return a float64 array holding exactly the given bit patterns."""
    return np.array(patterns, np.uint64).view(np.float64)


def _quiet_nans_only(n: int) -> np.ndarray:
    """Return the float64 array of `unusual_nans` with each signalling NaN made quiet.

    The negative quiet NaNs with a payload stay, so the kernel takes it.
    """
    x = unusual_nans("f64", n)
    bits = x.view(np.uint64)
    bits[::_SIGNALLING_EVERY] = _QUIET_NAN
    return x


def _late_signalling_nan(n: int) -> np.ndarray:
    """Return random float64 values whose only signalling NaN is the last element."""
    x = values("f64", n)
    x.view(np.uint64)[-1] = _UNUSUAL_NANS[8][1]
    return x


def accumulate_cases() -> Iterator[Case]:
    """Yield the cases of `np.maximum.accumulate` and `np.fmax.accumulate`.

    Both ufuncs take cummax_i64 on int64. np.fmax takes cumfmax_f64 on
    float64 unless the array holds a signalling NaN; np.maximum on float64
    propagates NaN and is NumPy's. Other ufuncs' accumulations pass through
    the hook uncounted.
    """
    yield from unary_cases("accumulate", {"i64": "cummax_i64"}, ("f64", "f32", "i32", "u64"),
                           (*_accumulate_spellings(np.maximum), *_OTHER_ACCUMULATIONS), empty_takes=False)
    yield from unary_cases("accumulate", {"i64": "cummax_i64", "f64": "cumfmax_f64"}, ("f32", "i32", "u64"),
                           _accumulate_spellings(np.fmax), empty_takes=False)
    # np.maximum on float64 over every value class, NaNs included: NumPy's.
    for n, _, label, maker in value_arrays("f64", FEW_SIZES):
        yield Case("accumulate", f"np.maximum.accumulate(x) f64 {label}", partial(_made, maker),
                   _maximum_accumulate, None)
    for patterns in _SHORT_FLOATS:
        text = " ".join(f"{p:#x}" for p in patterns)
        yield Case("accumulate", f"np.fmax.accumulate(x) f64 x=[{text}]", partial(_made, partial(_from_bits, patterns)),
                   _fmax_accumulate, "cumfmax_f64")
        yield Case("accumulate", f"np.maximum.accumulate(x) f64 x=[{text}]",
                   partial(_made, partial(_from_bits, patterns)), _maximum_accumulate, None)
    for numbers in _SHORT_INTS:
        for text, call in (("np.maximum.accumulate(x)", _maximum_accumulate), ("np.fmax.accumulate(x)", _fmax_accumulate)):
            yield Case("accumulate", f"{text} i64 x={list(numbers)}",
                       partial(_made, partial(np.array, numbers, np.int64)), call, "cummax_i64")
    # A signalling NaN is NumPy's: it answers with a quiet NaN where the kernel would skip it.
    for n in VALUE_SIZES:
        yield Case("accumulate", f"np.fmax.accumulate(x) f64 n={n} x=signalling and negative NaNs",
                   partial(_made, partial(unusual_nans, "f64", n)), _fmax_accumulate, None)
        yield Case("accumulate", f"np.fmax.accumulate(x) f64 n={n} x=negative NaNs with a payload",
                   partial(_made, partial(_quiet_nans_only, n)), _fmax_accumulate, "cumfmax_f64")
    yield Case("accumulate", f"np.fmax.accumulate(x) f64 n={ACCUMULATE_LATE_N} x=signalling NaN last only",
               partial(_made, partial(_late_signalling_nan, ACCUMULATE_LATE_N)), _fmax_accumulate, None)
    # The kernel's vmaxpd sets the invalid flag for a NaN; NumPy's loop leaves it clear.
    for n in FEW_SIZES:
        yield Case("accumulate", f"np.fmax.accumulate(x) f64 n={n} x=nan_some invalid flag after",
                   partial(_made, partial(values, "f64", n, "nan_some")),
                   partial(_invalid_flag_after, _fmax_accumulate), "cumfmax_f64")


# ------------------------------------------------- floating-point status flags

# The C library, whose fenv functions read and clear the processor's
# floating-point status flags. Found by name, so that it is glibc's libm on
# Linux and libSystem's on macOS.
_LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
# The invalid-operation flag of fenv.h. Its value is 1 both on x86-64 Linux and
# on ARM64 macOS.
_FE_INVALID = 1


def _invalid_flag_after(call: Callable[..., object], *arguments: object) -> tuple:
    """Make a call with the invalid-operation flag cleared, and say whether the call left it set.

    NumPy clears the flag after its own loops, so a caller that reads the
    flags afterwards, as C code beside NumPy can, finds it clear. A kernel
    whose instructions set it for a NaN operand must leave it clear too.

    Args:
        call: Makes the call on the arguments.
        arguments: The call's arguments.

    Returns:
        (the call's result, whether the invalid flag was set afterwards).
    """
    _LIBM.feclearexcept(_FE_INVALID)
    result = call(*arguments)
    return result, bool(_LIBM.fetestexcept(_FE_INVALID))


# ---------------------------------------------------------------- reduceat

# Run lengths of the regular starts of np.fmax.reduceat: runs of one element,
# which the binding copies, the shortest runs the kernel is called for, the
# length it was timed for, and a long one.
_REDUCEAT_RUNS = (1, 2, 3, 16, 100)
# Run length of the starts in the sweep over every value class, the layouts
# and the spellings.
_REDUCEAT_RUN = 16
# One start in this many, on average, in the irregular starts.
_REDUCEAT_SPARSE = 8
# Every how many elements the starts with one repeated start are placed.
_REDUCEAT_REPEAT_STEP = 3
# Bit patterns of float64 values, for the arrays built to hit the binding's
# guards: the quiet bit, and the sign bit with the magnitude.
_QUIET_BIT = 0x0008000000000000
_MAGNITUDE = 0x7FFFFFFFFFFFFFFF
# Element count of the arrays built to hit the binding's guards: ten groups
# of four.
_REDUCEAT_GUARD_N = 40
# Expectation of a reduceat case that `_reduceat_expect` works out.
_WORKED_OUT = "worked out"
# The two-dimensional layout, whose rows the default axis reduces.
_ROWS = next(variant for variant in VARIANTS if variant.name == "rows")


def _early_signalling_nan(n: int) -> np.ndarray:
    """Return random float64 values whose only signalling NaN is the first element."""
    x = values("f64", n)
    x.view(np.uint64)[0] = _UNUSUAL_NANS[8][1]
    return x


def _fmax_reduceat(x, starts):
    """Return `np.fmax.reduceat(x, starts)`."""
    return np.fmax.reduceat(x, starts)


def _regular_starts(run: int, n: int) -> np.ndarray:
    """Return the int64 starts of runs of `run` elements over `n`, from 0."""
    return np.arange(0, n, run, dtype=np.int64)


def _irregular_starts(n: int) -> np.ndarray:
    """Return increasing int64 starts at random positions of `n` elements, not always including 0."""
    rng = np.random.default_rng([SEED, n])
    return np.unique(rng.integers(0, n, max(1, n // _REDUCEAT_SPARSE))).astype(np.int64)


def _repeated_starts(n: int) -> np.ndarray:
    """Return regular int64 starts with the middle one given twice, so that the first of the two is one element."""
    starts = _regular_starts(_REDUCEAT_REPEAT_STEP, n)
    middle = len(starts) // 2
    return np.insert(starts, middle, starts[middle])


def _decreasing_starts(n: int) -> np.ndarray:
    """Return int64 starts that go back twice, so that two runs are one element each."""
    return np.array([0, n // 2, n // 4, 3 * n // 4, n - 1], np.int64)


def _last_start(n: int) -> np.ndarray:
    """Return the regular starts with n - 1 added as the last: a last run of one element."""
    return np.append(_regular_starts(_REDUCEAT_RUN, n), np.int64(n - 1))


def _fixed_starts(starts: tuple[int, ...]) -> np.ndarray:
    """Return the given starts as an int64 array."""
    return np.array(starts, np.int64)


# The starts that are in range for any n of at least one: (text, maker of n).
_REDUCEAT_STARTS: tuple[tuple[str, Callable[[int], object]], ...] = (
    *((f"runs of {run}", partial(_regular_starts, run)) for run in _REDUCEAT_RUNS),
    ("irregular", _irregular_starts),
    ("[0]", lambda n: _fixed_starts((0,))),
    ("[n // 2]", lambda n: _fixed_starts((n // 2,))),
    ("one repeated", _repeated_starts),
    ("decreasing", _decreasing_starts),
    ("last n - 1", _last_start),
)

# Starts that the binding leaves to NumPy whatever the array: out of range
# (NumPy raises IndexError), empty, indices of another type, which NumPy
# converts, and a list.
_REDUCEAT_FOREIGN_STARTS: tuple[tuple[str, Callable[[int], object]], ...] = (
    ("[0, n]", lambda n: _fixed_starts((0, n))),
    ("[0, -1]", lambda n: _fixed_starts((0, -1))),
    ("[n]", lambda n: _fixed_starts((n,))),
    ("empty", lambda n: np.empty(0, np.int64)),
    ("int32", lambda n: _regular_starts(_REDUCEAT_RUN, n).astype(np.int32)),
    ("list", lambda n: _regular_starts(_REDUCEAT_RUN, n).tolist()),
)


def _reduceat_takes(x: np.ndarray, starts: np.ndarray) -> bool:
    """Say whether the binding takes `np.fmax.reduceat(x, starts)` for a plain float64 x and int64 starts.

    It is the binding's rule written again in NumPy, so that the sweep over
    the value classes checks both ways of every guard: no index outside
    [0, n); no signalling NaN; and the call is left to NumPy when the array
    holds a negative zero and a result is a zero, or holds NaNs of more than
    one bit pattern and a result is a NaN.

    Args:
        x: The array, contiguous float64, one-dimensional.
        starts: The starts, contiguous int64, one-dimensional.

    Returns:
        True when the kernel must take the call.
    """
    n = len(x)
    if n == 0 or len(starts) == 0 or starts.min() < 0 or starts.max() >= n:
        return False
    bits = x.view(np.uint64)
    nan = (bits & np.uint64(_MAGNITUDE)) > np.uint64(_NEG_INF & _MAGNITUDE)
    if (nan & ((bits & np.uint64(_QUIET_BIT)) == 0)).any():
        return False
    # The values of the results do not depend on which bits a run returns,
    # so NumPy's own call says which are zeros and NaNs.
    result = np.fmax.reduceat(x, starts)
    if (bits == np.uint64(_NEG_ZERO)).any() and (result == 0).any():
        return False
    return not (len(np.unique(bits[nan])) > 1 and np.isnan(result).any())


def _reduceat_expect(x_maker: Callable[[], np.ndarray], starts_maker: Callable[[], np.ndarray]) -> str | None:
    """Return the kernel for a plain call built by the two makers, or None when the call is NumPy's."""
    return "fmax_reduce_f64" if _reduceat_takes(x_maker(), starts_maker()) else None


def _reduceat_case(label: str, x_maker: Callable[[], object], starts_maker: Callable[[], object],
                   call: Callable = _fmax_reduceat, expect: str | None = _WORKED_OUT) -> Case:
    """Build one reduceat case.

    Args:
        label: What the case is, after the call's text.
        x_maker: Builds the array.
        starts_maker: Builds the starts.
        call: Makes the call on the array and the starts.
        expect: The expectation. _WORKED_OUT, the default, has
            `_reduceat_expect` work it out for a plain call.

    Returns:
        The case.
    """
    if expect == _WORKED_OUT:
        expect = _reduceat_expect(x_maker, starts_maker)
    return Case("reduceat", label, partial(_built_each, x_maker, starts_maker), call, expect)


def _float_bits(*patterns: int) -> Callable[[], np.ndarray]:
    """Return a maker of the float64 array holding exactly the given bit patterns."""
    return partial(_from_bits, tuple(patterns))


def _reduceat_guard_cases() -> Iterator[Case]:
    """Yield the cases built to hit the guards, each with its expectation stated, not worked out.

    The array is ten groups of four: a negative zero, a positive zero and
    negative values, or NaNs of two patterns, with the runs placed so that a
    run's result is or is not the ambiguous value.
    """
    groups = _REDUCEAT_GUARD_N // 4
    minus_one, two = 0xBFF0000000000000, 0x4000000000000000
    both_zeros = _float_bits(*(_NEG_ZERO, _POS_ZERO, minus_one, _NEG_ZERO) * groups)
    starts_of_four = partial(_regular_starts, 4, _REDUCEAT_GUARD_N)
    yield _reduceat_case("np.fmax.reduceat(x, s) zeros of both signs, a zero result", both_zeros, starts_of_four,
                         expect=None)
    # A negative zero that no result is: every run of four ends at a 2.0.
    zeros_then_two = _float_bits(*(_NEG_ZERO, _POS_ZERO, minus_one, two) * groups)
    yield _reduceat_case("np.fmax.reduceat(x, s) a negative zero, no zero result", zeros_then_two, starts_of_four,
                         expect="fmax_reduce_f64")
    # Positive zeros only: a zero result with no negative zero stays with the kernel.
    positive_zeros = _float_bits(*(_POS_ZERO, minus_one, _POS_ZERO, minus_one) * groups)
    yield _reduceat_case("np.fmax.reduceat(x, s) positive zeros, a zero result", positive_zeros, starts_of_four,
                         expect="fmax_reduce_f64")
    two_nans = _float_bits(*(_QUIET_NAN, _PAYLOAD_NAN, _QUIET_NAN, _PAYLOAD_NAN) * groups)
    yield _reduceat_case("np.fmax.reduceat(x, s) all NaN of two patterns", two_nans, starts_of_four, expect=None)
    one_nan = _float_bits(*(_PAYLOAD_NAN,) * _REDUCEAT_GUARD_N)
    yield _reduceat_case("np.fmax.reduceat(x, s) all NaN of one pattern", one_nan, starts_of_four,
                         expect="fmax_reduce_f64")
    # NaNs of two patterns, none a result: every run of four holds a number.
    nans_and_one = _float_bits(*(_QUIET_NAN, _PAYLOAD_NAN, _ONE, _OTHER_NAN) * groups)
    yield _reduceat_case("np.fmax.reduceat(x, s) NaNs of three patterns, no NaN result", nans_and_one,
                         starts_of_four, expect="fmax_reduce_f64")
    # The same array with runs of one element: a result is a NaN.
    yield _reduceat_case("np.fmax.reduceat(x, s) NaNs of three patterns, runs of 1", nans_and_one,
                         partial(_regular_starts, 1, _REDUCEAT_GUARD_N), expect=None)
    for n in VALUE_SIZES:
        regular = partial(_regular_starts, _REDUCEAT_RUN, n)
        yield _reduceat_case(f"np.fmax.reduceat(x, s) n={n} x=signalling and negative NaNs s=runs of 16",
                             partial(unusual_nans, "f64", n), regular, expect=None)
        yield _reduceat_case(f"np.fmax.reduceat(x, s) n={n} x=negative NaNs with a payload s=runs of 16",
                             partial(_quiet_nans_only, n), regular)
    # The array is read for signalling NaNs a block at a time ahead of the
    # runs; one in an element no run uses still sends the call to NumPy.
    yield _reduceat_case(f"np.fmax.reduceat(x, s) n={ACCUMULATE_LATE_N} x=signalling NaN first s=[n // 2]",
                         partial(_early_signalling_nan, ACCUMULATE_LATE_N),
                         partial(_fixed_starts, (ACCUMULATE_LATE_N // 2,)), expect=None)
    yield _reduceat_case(f"np.fmax.reduceat(x, s) n={ACCUMULATE_LATE_N} x=signalling NaN last only s=decreasing",
                         partial(_late_signalling_nan, ACCUMULATE_LATE_N),
                         partial(_decreasing_starts, ACCUMULATE_LATE_N), expect=None)
    late = partial(_late_signalling_nan, ACCUMULATE_LATE_N)
    yield _reduceat_case(f"np.fmax.reduceat(x, s) n={ACCUMULATE_LATE_N} x=signalling NaN last only", late,
                         partial(_regular_starts, _REDUCEAT_RUN, ACCUMULATE_LATE_N), expect=None)


# Spellings of np.fmax.reduceat(x, s), and whether the kernel takes them on a
# plain array: (text, call, taken). `axis` -1 names the one axis as 0 does.
_REDUCEAT_SPELLINGS: tuple[tuple[str, Callable, bool], ...] = (
    ("np.fmax.reduceat(x, s)", _fmax_reduceat, True),
    ("np.fmax.reduceat(x, s, 0)", lambda x, s: np.fmax.reduceat(x, s, 0), True),
    ("np.fmax.reduceat(x, s, axis=0)", lambda x, s: np.fmax.reduceat(x, s, axis=0), True),
    ("np.fmax.reduceat(x, s, axis=-1)", lambda x, s: np.fmax.reduceat(x, s, axis=-1), True),
    ("np.fmax.reduceat(array=x, indices=s)", lambda x, s: np.fmax.reduceat(array=x, indices=s), True),
    ("np.fmax.reduceat(x, s, 0, None, None)", lambda x, s: np.fmax.reduceat(x, s, 0, None, None), True),
    ("np.fmax.reduceat(x, s, axis=0, dtype=None, out=None)",
     lambda x, s: np.fmax.reduceat(x, s, axis=0, dtype=None, out=None), True),
    ("np.fmax.reduceat(x, s, axis=None)", lambda x, s: np.fmax.reduceat(x, s, axis=None), False),
    ("np.fmax.reduceat(x, s, axis=np.int64(0))", lambda x, s: np.fmax.reduceat(x, s, axis=np.int64(0)), False),
    ("np.fmax.reduceat(x, s, dtype=np.float64)", lambda x, s: np.fmax.reduceat(x, s, dtype=np.float64), False),
    ("np.fmax.reduceat(x, s, out=o)",
     lambda x, s: np.fmax.reduceat(x, s, out=np.empty(np.shape(s), np.float64)), False),
    ("np.maximum.reduceat(x, s)", np.maximum.reduceat, False),
    ("np.add.reduceat(x, s)", np.add.reduceat, False),
    ("np.fmin.reduceat(x, s)", np.fmin.reduceat, False),
)


def _reduceat_layout_cases() -> Iterator[Case]:
    """Yield every layout of the array and of the starts, every spelling, and the neighbouring element types.

    Only a plain one-dimensional float64 array and a plain one-dimensional
    int64 array of starts are taken. The starts of a two-dimensional array
    are those of its rows, which the default axis reduces.
    """
    for n in LAYOUT_SIZES:
        regular = partial(_regular_starts, _REDUCEAT_RUN, n)
        for variant in variants_for("f64", n):
            rows = n // COLUMNS if variant.dims == 2 else n
            starts = partial(_regular_starts, _REDUCEAT_RUN, rows) if variant.dims == 2 else regular
            expect = "fmax_reduce_f64" if variant.plain and variant.dims == 1 else None
            if not variant.predictable:
                expect = ANY
            yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 n={n} x layout={variant.name}",
                                 partial(_variant_of, "f64", n, variant, "x"), starts, expect=expect)
        # Runs of two, so that even at the smallest size there are several
        # starts: NumPy flags an array of one element contiguous whatever its
        # strides, and the kernel may then take it.
        pairs = _regular_starts(2, n)
        for variant in variants_for("i64", len(pairs)):
            expect = "fmax_reduce_f64" if variant.plain and variant.dims == 1 else None
            if not variant.predictable:
                expect = ANY
            yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 n={n} s=runs of 2 layout={variant.name}",
                                 partial(values, "f64", n), partial(variant.build, pairs), expect=expect)
        for text, call, taken in _REDUCEAT_SPELLINGS:
            yield _reduceat_case(f"{text} f64 n={n}", partial(values, "f64", n), regular, call,
                                 "fmax_reduce_f64" if taken else None)
        for dtype in ("f32", "i64", "i32"):
            yield _reduceat_case(f"np.fmax.reduceat(x, s) {dtype} n={n}", partial(values, dtype, n), regular,
                                 expect=None)
        rows = partial(_variant_of, "f64", n, _ROWS, "x")
        yield _reduceat_case(f"np.fmax.reduceat(x, s, axis=1) f64 n={n} x=rows", rows,
                             partial(_fixed_starts, (0, 1, 1)),
                             lambda x, s: np.fmax.reduceat(x, s, axis=1), None)
        for text, starts in _REDUCEAT_FOREIGN_STARTS:
            yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 n={n} s={text}", partial(values, "f64", n),
                                 partial(starts, n), expect=None)
    for text, starts in _REDUCEAT_FOREIGN_STARTS:
        yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 n=0 s={text}", partial(values, "f64", 0),
                             partial(starts, 0), expect=None)


def reduceat_cases() -> Iterator[Case]:
    """Yield the cases of `np.fmax.reduceat`.

    The kernel takes a plain one-dimensional float64 array and plain
    one-dimensional int64 starts in [0, n), and runs once per run; see the
    binding's try_reduceat. Every value class at every size with runs of 16
    and irregular starts; a few sizes with every kind of starts; the guards;
    layouts, spellings and neighbouring types; and whether the call leaves
    the invalid-operation flag set.
    """
    for n, _, label, maker in value_arrays("f64"):
        for text, starts in (("runs of 16", partial(_regular_starts, _REDUCEAT_RUN, n)),
                             ("irregular", partial(_irregular_starts, n))):
            yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 {label} s={text}", maker, starts)
    for n, _, label, maker in value_arrays("f64", FEW_SIZES):
        for text, starts in _REDUCEAT_STARTS:
            yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 {label} s={text}", maker, partial(starts, n))
    yield from _reduceat_guard_cases()
    yield from _reduceat_layout_cases()
    for n in FEW_SIZES:
        nan_some = partial(values, "f64", n, "nan_some")
        yield _reduceat_case(f"np.fmax.reduceat(x, s) f64 n={n} x=nan_some invalid flag after",
                             nan_some, partial(_regular_starts, _REDUCEAT_RUN, n),
                             partial(_invalid_flag_after, _fmax_reduceat))


# ------------------------------------------------------------------- any

# Row widths of the any(axis=1) sweep: either side of each vector and block
# size the kernel may work in (8, 16, 32 and 64 bytes), its own paths (rows of
# 4 have one), the width it was timed for, and two long rows.
ANY_COLS = (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 22, 31, 32, 33, 63, 64, 65, 100, 1000)
# Row counts of the sweep. A further count per width puts the matrix above
# the size from which the binding releases the interpreter lock
# (_any_large_rows).
ANY_ROWS = (1, 2, 3, 100)
# Element count from which the binding releases the interpreter lock
# (HK_NOGIL_MIN_N in binding.h).
NOGIL_MIN_N = 8192
# Width and row count of the matrix whose only bad byte is its last: more
# than two of the blocks of 16,384 bytes in which the binding checks a matrix.
ANY_LATE_COLS = 22
ANY_LATE_ROWS = 2000
# Every how many rows the matrix with some rows True holds one, and the step
# by which that True moves along the columns from one such row to the next.
ANY_MARKED_EVERY = 3
ANY_MARKED_STEP = 5
# Width and row count of the matrices of the layouts and spellings sweep.
ANY_LAYOUT_SHAPES = ((3, 7), (100, 22), (1000, 22))


def _any_large_rows(cols: int) -> int:
    """Return the fewest rows of `cols` elements that the binding runs without the interpreter lock."""
    return NOGIL_MIN_N // cols + 1


def _any_matrix(maker: Callable[[], np.ndarray], cols: int) -> np.ndarray:
    """Return `maker`'s elements as a C-ordered matrix of `cols` columns.

    Args:
        maker: Builds a one-dimensional array whose length `cols` divides.
        cols: Elements per row, at least 1.

    Returns:
        The matrix, of its own memory.
    """
    return maker().reshape(-1, cols).copy()


def _any_random(rows: int, cols: int, dtype: str = "bool") -> np.ndarray:
    """Return random elements of `dtype` as a C-ordered matrix of `rows` by `cols`."""
    return values(dtype, rows * cols, arg="mask").reshape(rows, cols).copy()


def _any_sparse(rows: int, cols: int) -> np.ndarray:
    """Return a boolean matrix of `rows` by `cols` in which some rows hold a True and the others none.

    Random data of a few hundred elements has a True in nearly every row,
    which a kernel that reduced the wrong elements could still match.
    """
    m = np.zeros((rows, cols), np.bool_)
    marked = np.arange(0, rows, ANY_MARKED_EVERY)
    m[marked, (marked * ANY_MARKED_STEP) % cols] = True
    return m


def _any_bad(rows: int, cols: int, byte: int) -> np.ndarray:
    """Return a boolean matrix whose True elements are stored as `byte`, not 1.

    NumPy takes any non-zero byte as True. On one of the kernel's paths a
    byte above 1 changes the results of the rows beside it, so the binding
    must keep such a matrix away from the kernel.
    """
    return _bad_mask(rows * cols, byte).reshape(rows, cols)


def _any_late_bad(byte: int) -> np.ndarray:
    """Return a boolean matrix of zeros whose last element is stored as `byte`: clean until its last block."""
    return _late_bad_mask(ANY_LATE_ROWS * ANY_LATE_COLS, byte).reshape(ANY_LATE_ROWS, ANY_LATE_COLS)


def _any_out(m):
    """Run `m.any(axis=1, out=o)` into a new array of the right length and return `o`."""
    return m.any(axis=1, out=np.empty(np.shape(m)[0], np.bool_))


def _any_function_out(m):
    """Run `np.any(m, axis=1, out=o)` into a new array of the right length and return `o`."""
    return np.any(m, axis=1, out=np.empty(np.shape(m)[0], np.bool_))


def _any_where(m):
    """Run `m.any(axis=1, where=w)` with a `where` that leaves out the first column."""
    where = np.ones(np.shape(m), np.bool_)
    where[:, 0] = False
    return m.any(axis=1, where=where)


# The spellings of any(axis=1) that take the kernel, first the two the sweep
# over every value runs.
_ANY_TAKING: tuple[tuple[str, Callable], ...] = (
    ("m.any(axis=1)", lambda m: m.any(axis=1)),
    ("np.any(m, axis=1)", lambda m: np.any(m, axis=1)),
    ("m.any(axis=-1)", lambda m: m.any(axis=-1)),
    ("m.any(1)", lambda m: m.any(1)),
    ("m.any(-1)", lambda m: m.any(-1)),
    ("np.any(m, axis=-1)", lambda m: np.any(m, axis=-1)),
    ("np.any(m, 1)", lambda m: np.any(m, 1)),
    ("np.any(m, -1)", lambda m: np.any(m, -1)),
)
# Spellings of the same operation, or of one beside it, that must stay with NumPy.
_ANY_NEVER: tuple[tuple[str, Callable], ...] = (
    ("m.any()", lambda m: m.any()),
    ("np.any(m)", np.any),
    ("m.any(axis=0)", lambda m: m.any(axis=0)),
    ("np.any(m, axis=0)", lambda m: np.any(m, axis=0)),
    ("m.any(axis=None)", lambda m: m.any(axis=None)),
    ("np.any(m, axis=None)", lambda m: np.any(m, axis=None)),
    ("m.any(axis=(1,))", lambda m: m.any(axis=(1,))),
    ("np.any(m, axis=(1,))", lambda m: np.any(m, axis=(1,))),
    ("m.any(axis=np.int64(1))", lambda m: m.any(axis=np.int64(1))),
    ("np.any(m, axis=np.int64(1))", lambda m: np.any(m, axis=np.int64(1))),
    ("m.any(axis=1, keepdims=True)", lambda m: m.any(axis=1, keepdims=True)),
    ("np.any(m, axis=1, keepdims=True)", lambda m: np.any(m, axis=1, keepdims=True)),
    ("m.any(axis=1, keepdims=False)", lambda m: m.any(axis=1, keepdims=False)),
    ("m.any(1, None)", lambda m: m.any(1, None)),
    ("np.any(m, 1, None)", lambda m: np.any(m, 1, None)),
    ("m.any(axis=1, out=o)", _any_out),
    ("np.any(m, axis=1, out=o)", _any_function_out),
    ("m.any(axis=1, where=w)", _any_where),
    ("np.any(a=m, axis=1)", lambda m: np.any(a=m, axis=1)),
    ("m.any(axis=2)", lambda m: m.any(axis=2)),
    ("m.all(axis=1)", lambda m: m.all(axis=1)),
    ("np.all(m, axis=1)", lambda m: np.all(m, axis=1)),
    ("np.logical_or.reduce(m, axis=1)", lambda m: np.logical_or.reduce(m, axis=1)),
)
# Ways a boolean matrix can be laid out or wrapped, built from a C-ordered
# one, and whether the kernel takes it.
_ANY_LAYOUTS: tuple[tuple[str, Callable[[np.ndarray], object], str | None], ...] = (
    ("contiguous", np.copy, "any_rows_bool"),
    ("readonly", _readonly, "any_rows_bool"),
    ("fortran", np.asfortranarray, None),
    ("strided rows", lambda m: np.repeat(m, 2, axis=0)[::2], None),
    ("strided columns", lambda m: np.repeat(m, 2, axis=1)[:, ::2], None),
    ("transposed", lambda m: np.ascontiguousarray(m.T).T, None),
    ("subclass", lambda m: m.copy().view(_Subclass), None),
    ("memmap", _memmap, None),
    ("masked_array", lambda m: np.ma.array(m.copy()), ANY),
    ("list", lambda m: m.tolist(), None),
)


def _any_laid_out(build: Callable[[np.ndarray], object], rows: int, cols: int) -> object:
    """Return a random boolean matrix of `rows` by `cols` as `build` lays it out."""
    return build(_any_random(rows, cols))


def _any_cube(rows: int, cols: int) -> np.ndarray:
    """Return a random boolean matrix of `rows` by `cols` as a C-ordered array of shape (rows, 1, cols)."""
    return _any_random(rows, cols).reshape(rows, 1, cols)


def _any_flat(rows: int, cols: int) -> np.ndarray:
    """Return the elements of a random boolean matrix of `rows` by `cols` as one dimension."""
    return _any_random(rows, cols).ravel()


def _any_case(label: str, maker: Callable[[], object], call: Callable, expect: str | None) -> Case:
    """Return one case of `any`, its one argument built by `maker`."""
    return Case("any", label, partial(_made, maker), call, expect)


def _any_value_cases() -> Iterator[Case]:
    """Yield every boolean value class at every width and row count, for the two main spellings.

    Positional classes put their element first, in the middle and last of
    the matrix's elements, and so in its first, a middle and its last row.
    """
    for cols in ANY_COLS:
        for rows in (*ANY_ROWS, _any_large_rows(cols)):
            for _, _, label, maker in value_arrays("bool", (rows * cols,), arg="m"):
                shape = f"{rows}x{cols} {label.split(' ', 1)[1]}"
                for text, call in _ANY_TAKING[:2]:
                    yield _any_case(f"{text} bool {shape}", partial(_any_matrix, maker, cols), call,
                                    "any_rows_bool")
            yield _any_case(f"m.any(axis=1) bool {rows}x{cols} some rows", partial(_any_sparse, rows, cols),
                            _ANY_TAKING[0][1], "any_rows_bool")
        for text, call in _ANY_TAKING[:2]:
            yield _any_case(f"{text} bool 0x{cols}", partial(_any_random, 0, cols), call, "any_rows_bool")


def _any_layout_cases() -> Iterator[Case]:
    """Yield every spelling, every layout and every element type beside bool, at a few shapes."""
    for rows, cols in ANY_LAYOUT_SHAPES:
        shape = f"{rows}x{cols}"
        matrix = partial(_any_random, rows, cols)
        for text, call in _ANY_TAKING:
            yield _any_case(f"{text} bool {shape}", matrix, call, "any_rows_bool")
        for text, call in _ANY_NEVER:
            yield _any_case(f"{text} bool {shape}", matrix, call, None)
        for name, build, expect in _ANY_LAYOUTS:
            for text, call in _ANY_TAKING[:2]:
                yield _any_case(f"{text} bool {shape} layout={name}", partial(_any_laid_out, build, rows, cols), call,
                                expect)
        for dtype in ("u8", "i64", "f64"):
            for text, call in _ANY_TAKING[:2]:
                yield _any_case(f"{text} {dtype} {shape}", partial(_any_random, rows, cols, dtype), call, None)
        for text, call in _ANY_TAKING[:2]:
            yield _any_case(f"{text} bool {shape} as 3-D", partial(_any_cube, rows, cols), call, None)
            yield _any_case(f"{text} bool {rows * cols} in 1-D", partial(_any_flat, rows, cols), call, None)
    for text, call in _ANY_TAKING[:2] + _ANY_NEVER[:2]:
        for rows in (0, 3):
            yield _any_case(f"{text} bool {rows}x0", partial(_any_random, rows, 0), call, None)
        yield _any_case(f"{text} bool 0x0", partial(_any_random, 0, 0), call, None)


def _any_bad_byte_cases() -> Iterator[Case]:
    """Yield matrices holding bytes other than 0 and 1, which must stay with NumPy.

    The widths and row counts include matrices of fewer than 32 elements of
    a width other than 4, the path on which such a byte would change the
    results of other rows.
    """
    for byte in BAD_MASK_BYTES:
        for cols in (1, 3, 4, 7, 22, 33, 100):
            for rows in (1, 3, 100):
                for text, call in _ANY_TAKING[:2]:
                    yield _any_case(f"{text} bool {rows}x{cols} mask bytes are {byte}",
                                    partial(_any_bad, rows, cols, byte), call, None)
        for text, call in _ANY_TAKING[:2]:
            yield _any_case(f"{text} bool {ANY_LATE_ROWS}x{ANY_LATE_COLS} last mask byte is {byte}",
                            partial(_any_late_bad, byte), call, None)


def any_cases() -> Iterator[Case]:
    """Yield the cases of `m.any(axis=1)` and `np.any(m, axis=1)` on boolean matrices.

    The kernel takes a plain, C-ordered, two-dimensional boolean array with
    at least one column whose bytes are all 0 or 1, with `axis` 1 or -1 as a
    Python int and no other argument; see the binding's hk_try_any_rows.
    """
    yield from _any_value_cases()
    yield from _any_layout_cases()
    yield from _any_bad_byte_cases()


# ------------------------------------------------------------------ hypot

# Value classes whose pairs make NumPy's hypot raise a floating-point flag
# when both arrays hold them: underflow for two denormals, overflow for two
# of the largest doubles. The kernel then hands the call to NumPy's loop, or
# raises no flag at the sizes where the class holds no such pair, so either
# side may compute it.
HYPOT_MAY_RAISE = frozenset({"denormal", "extremes"})
# Element types beside float64. float32 has a loop of its own, which is not
# hooked. Integers are converted to float64 in buffers, contiguous ones, that
# NumPy's float64 loop receives, so the kernel may take them.
HYPOT_OTHER_LOOPS = ("f32",)
HYPOT_CONVERTED = ("i64", "i32")
# Elements the binding computes at a time when the output is one of the
# inputs (HK_STACK_SCRATCH_BYTES of doubles), and sizes around it.
HYPOT_BLOCK = 256
HYPOT_BLOCK_SIZES = (HYPOT_BLOCK - 1, HYPOT_BLOCK, HYPOT_BLOCK + 1, 3 * HYPOT_BLOCK + 5)
# Pairs whose hypot raises a flag in NumPy's loop: overflow from two of the
# largest doubles, underflow from two denormals. Planted in otherwise
# ordinary arrays, to show that the warnings are NumPy's.
HYPOT_OVERFLOWS = (1.5e308, 1.5e308)
HYPOT_UNDERFLOWS = (3e-320, 4e-320)
# A pair where b is negligible beside a (less than 2^-54 of it) and b * b / a
# is below 2^-1022: NumPy raises nothing, and the kernel raises underflow in
# the part of its arithmetic it throws away, so the call goes to NumPy.
HYPOT_NEGLIGIBLE_UNDERFLOW = (2.0 ** 300, 2.0 ** -400)
# Settings of np.errstate the flagged calls are made under.
ERRSTATE_SETTINGS = ("raise", "warn", "ignore")


def _hypot(a, b):
    """Return `np.hypot(a, b)`."""
    return np.hypot(a, b)


def _hypot_out(a, b):
    """Return `np.hypot(a, b, out=o)` into a fresh output array."""
    return np.hypot(a, b, out=np.empty(np.shape(a)))


def _hypot_into_a(a, b):
    """Return `np.hypot(a, b, out=a)`: the output is the first input."""
    return np.hypot(a, b, out=a)


def _hypot_into_b(a, b):
    """Return `np.hypot(a, b, out=b)`: the output is the second input."""
    return np.hypot(a, b, out=b)


def _hypot_self(a, b):
    """Return `np.hypot(a, a)`: one array as both inputs."""
    return np.hypot(a, a)


def _hypot_self_into(a, b):
    """Return `np.hypot(a, a, out=a)`: one array as both inputs and the output."""
    return np.hypot(a, a, out=a)


def _hypot_strided_out(a, b):
    """Return the output array of `np.hypot(a, b, out=o[::2])`, every second element of a larger array."""
    o = np.zeros(2 * len(a))
    np.hypot(a, b, out=o[::2])
    return o


def _hypot_overlap(a, b):
    """Return `np.hypot(a[1:], b[1:], out=a[:-1])`: the output overlaps the first input one element behind."""
    return np.hypot(a[1:], b[1:], out=a[:-1])


def _hypot_where(a, b):
    """Return `np.hypot(a, b, out=o, where=m)` with a mask True at every third element."""
    mask = np.zeros(len(a), dtype=bool)
    mask[::OVERLAP_EVERY] = True
    return np.hypot(a, b, out=np.zeros(len(a)), where=mask)


def _hypot_dtype(dtype: type, a, b):
    """Return `np.hypot(a, b, dtype=dtype)`."""
    return np.hypot(a, b, dtype=dtype)


def _hypot_reduce(a, b):
    """Return `np.hypot.reduce(a)`."""
    return np.hypot.reduce(a)


def _hypot_accumulate(a, b):
    """Return `np.hypot.accumulate(a)`."""
    return np.hypot.accumulate(a)


def _hypot_outer(a, b):
    """Return `np.hypot.outer(a[:30], b[:30])`."""
    return np.hypot.outer(a[:30], b[:30])


def _hypot_at(a, b):
    """Return a after `np.hypot.at(a, [0, 1, 1, -1], b[:4])`, which computes in place at indices."""
    np.hypot.at(a, [0, 1, 1, -1], b[:4])
    return a


def _hypot_errstate(setting: str, call: Callable, a, b):
    """Make a hypot call under np.errstate(all=setting).

    Args:
        setting: A value np.errstate takes, such as "raise" or "warn".
        call: The hypot call, taking a and b.
        a: First input.
        b: Second input.

    Returns:
        What the call returns.
    """
    with np.errstate(all=setting):
        return call(a, b)


# The ways of writing a call that the kernel takes for two plain arrays, and
# those it must not or may: (text, call, expectation for float64 arrays of
# HYPOT_BLOCK or more elements). A scalar has a step of zero, so a call with
# one is NumPy's; a strided output is too. NumPy copies an operand that
# overlaps the output partly, and the copy may reach the kernel. A masked
# call and the ufunc methods other than a call run NumPy's machinery around
# the loop, which decides what the loop sees.
_HYPOT_SPELLINGS: tuple[tuple[str, Callable, str | None], ...] = (
    ("np.hypot(a, b)", _hypot, "hypot_f64"),
    ("np.hypot(a, b, out=o)", _hypot_out, "hypot_f64"),
    ("np.hypot(a, b, out=a)", _hypot_into_a, "hypot_f64"),
    ("np.hypot(a, b, out=b)", _hypot_into_b, "hypot_f64"),
    ("np.hypot(a, a)", _hypot_self, "hypot_f64"),
    ("np.hypot(a, a, out=a)", _hypot_self_into, "hypot_f64"),
    ("np.hypot(a, b, dtype=np.float64)", partial(_hypot_dtype, np.float64), "hypot_f64"),
    ("np.hypot(a, b, dtype=np.float32)", partial(_hypot_dtype, np.float32), None),
    ("np.hypot(a, 2.5)", lambda a, b: np.hypot(a, 2.5), None),
    ("np.hypot(-0.0, b)", lambda a, b: np.hypot(-0.0, b), None),
    ("np.hypot(a, np.float64(3))", lambda a, b: np.hypot(a, np.float64(3)), None),
    ("np.hypot(a, b[:1])", lambda a, b: np.hypot(a, b[:1]), None),
    ("np.hypot(a, b, out=o[::2])", _hypot_strided_out, None),
    ("np.hypot(a[1:], b[1:], out=a[:-1])", _hypot_overlap, ANY),
    ("np.hypot(a, b, out=o, where=m)", _hypot_where, ANY),
    ("np.hypot.reduce(a)", _hypot_reduce, None),
    ("np.hypot.accumulate(a)", _hypot_accumulate, None),
    ("np.hypot.outer(a[:30], b[:30])", _hypot_outer, ANY),
    ("np.hypot.at(a, i, b[:4])", _hypot_at, ANY),
)


def _hypot_layout_expect(dtype: str, variant: Variant, n: int) -> str | None:
    """Say whether the kernel takes `np.hypot(a, b)` with one operand in a layout and the other plain.

    The hook sits under NumPy's machinery, so what decides is what reaches the
    float64 loop: contiguous doubles are taken whatever array they came from.
    A strided operand reaches the loop with its step and is NumPy's; most
    other layouts reach it through NumPy's buffers or as they are, which NumPy
    decides.

    Args:
        dtype: Element type of both operands, as a spec name.
        variant: Layout of the one operand that is not plain.
        n: Element count.

    Returns:
        The kernel name, None, or ANY.
    """
    # A two-dimensional operand of n elements does not broadcast against a
    # one-dimensional one of n: NumPy raises.
    if n == 0 or variant.dims == 2:
        return None
    # A list is converted to float64, whatever it was made from.
    if variant.name == "list":
        return ANY
    if dtype in HYPOT_OTHER_LOOPS:
        return None
    if dtype in HYPOT_CONVERTED:
        return ANY
    if variant.name in ("contiguous", "readonly"):
        return "hypot_f64"
    if variant.name == "strided":
        return None
    return ANY


def _planted(n: int, pair: tuple[float, float], arg: str, index: int) -> np.ndarray:
    """Return random float64 values with one element replaced by a pair's member.

    Args:
        n: Element count.
        pair: The two values planted, one in a and one in b.
        arg: "a" or "b", which also picks the member of the pair.
        index: Position of the planted element.

    Returns:
        A contiguous float64 array.
    """
    x = values("f64", n, arg=arg)
    x[index] = pair[0 if arg == "a" else 1]
    return x


def _hypot_value_cases() -> Iterator[Case]:
    """Yield `np.hypot(a, b)` over every float64 value class at every size, both arrays plain.

    Each class on both sides at once, each on a beside random b, and each on
    b beside random a; then the NaNs the judge does not make.
    """
    for n in VALUE_SIZES:
        random_a = partial(values, "f64", n, arg="a")
        random_b = partial(values, "f64", n, arg="b")
        sides = zip(value_arrays("f64", (n,), "a"), value_arrays("f64", (n,), "b"))
        for (_, cls, label, a_maker), (_, _, _, b_maker) in sides:
            yield Case("hypot", f"np.hypot(a, b) f64 {label} and b alike",
                       partial(_built_each, a_maker, b_maker), _hypot,
                       ANY if cls in HYPOT_MAY_RAISE else "hypot_f64")
            yield Case("hypot", f"np.hypot(a, b) f64 {label}, b random",
                       partial(_built_each, a_maker, random_b), _hypot, "hypot_f64")
        for _, _, label, b_maker in value_arrays("f64", (n,), "b"):
            yield Case("hypot", f"np.hypot(a, b) f64 a random, {label}",
                       partial(_built_each, random_a, b_maker), _hypot, "hypot_f64")
        # A signalling NaN raises the invalid flag in NumPy's loop, which then
        # warns; every array of unusual_nans starts with one.
        yield Case("hypot", f"np.hypot(a, b) f64 n={n} a and b=signalling and negative NaNs",
                   partial(_built_each, partial(unusual_nans, "f64", n), partial(_unusual_nans_reversed, n)),
                   _hypot, None)


def _hypot_layout_cases() -> Iterator[Case]:
    """Yield `np.hypot(a, b)` with each operand in every layout, the other plain, for float64 and its neighbours."""
    for dtype in ("f64", *HYPOT_OTHER_LOOPS, *HYPOT_CONVERTED):
        for n in (0, *LAYOUT_SIZES):
            plain = {"a": partial(values, dtype, n, arg="a"), "b": partial(values, dtype, n, arg="b")}
            for arg in ("a", "b"):
                for variant in variants_for(dtype, n):
                    makers = {**plain, arg: partial(_variant_of, dtype, n, variant, arg)}
                    yield Case("hypot", f"np.hypot(a, b) {dtype} n={n} {arg}={variant.name}",
                               partial(_built_each, makers["a"], makers["b"]), _hypot,
                               _hypot_layout_expect(dtype, variant, n))
        # Two arrays of different element types: NumPy converts one of them.
        if dtype != "f64":
            for n in LAYOUT_SIZES:
                yield Case("hypot", f"np.hypot(a, b) f64 and {dtype} n={n}",
                           partial(_built_each, partial(values, "f64", n, arg="a"), partial(values, dtype, n, arg="b")),
                           _hypot, ANY)


def _hypot_spelling_cases() -> Iterator[Case]:
    """Yield every spelling of the call, on float64 arrays around the binding's block size and the lock threshold."""
    for n in (1, 16, *HYPOT_BLOCK_SIZES, *LAYOUT_SIZES[1:]):
        make = partial(_built_each, partial(values, "f64", n, arg="a"), partial(values, "f64", n, arg="b"))
        for text, call, expect in _HYPOT_SPELLINGS:
            # NumPy describes the one element of an array of one as it likes:
            # its step may be anything, so the loop may or may not see 8.
            yield Case("hypot", f"{text} f64 n={n}", make, call, ANY if n == 1 else expect)
    for text, call, _ in _HYPOT_SPELLINGS[:2]:
        for n in (1, ORDINARY_N):
            yield Case("hypot", f"{text} f64 n={n} two-dimensional, b a row",
                       partial(_built_each, partial(_rows_of, n, "a"), partial(values, "f64", COLUMNS, arg="b")),
                       call, ANY)


def _rows_of(n: int, arg: str) -> np.ndarray:
    """Return random float64 values as a C-ordered array of COLUMNS columns, at least one row.

    Args:
        n: Element count, rounded up to a whole row.
        arg: Argument name, to vary the random data.

    Returns:
        A two-dimensional contiguous array.
    """
    rows = max(1, -(-n // COLUMNS))
    return values("f64", rows * COLUMNS, arg=arg).reshape(rows, COLUMNS)


def _hypot_flag_cases() -> Iterator[Case]:
    """Yield calls whose data make a floating-point flag, under every np.errstate setting.

    The kernel's result is dropped and NumPy's loop computes the call again,
    so the result, the warning or the exception are NumPy's. With the output
    one of the inputs the binding computes block by block, so a flag in a
    late block leaves the earlier blocks to the kernel and the rest to NumPy,
    which must still see the inputs of the rest as they were. The pair whose
    flag only the kernel raises is NumPy's too.
    """
    # The pair only the kernel's vector path raises a flag for goes to NumPy
    # when it falls in a group of four, and is the kernel's when it falls in
    # the last one to three elements, which the kernel does one at a time.
    pairs = (("overflow", HYPOT_OVERFLOWS, None), ("underflow", HYPOT_UNDERFLOWS, None),
             ("negligible", HYPOT_NEGLIGIBLE_UNDERFLOW, ANY))
    for n in (1, 16, *HYPOT_BLOCK_SIZES, LAYOUT_SIZES[-1]):
        for (name, pair, expect), index in itertools.product(pairs, sorted({0, n // 2, n - 1})):
            make = partial(_built_each, partial(_planted, n, pair, "a", index), partial(_planted, n, pair, "b", index))
            for text, call, _ in _HYPOT_SPELLINGS[:4]:
                for setting in ERRSTATE_SETTINGS:
                    yield Case("hypot", f"np.errstate(all={setting!r}) {text} f64 n={n} {name} at {index}",
                               make, partial(_hypot_errstate, setting, call), expect)


def hypot_cases() -> Iterator[Case]:
    """Yield the cases of `np.hypot(a, b)`.

    The hook is the float64 loop of the hypot ufunc. The kernel takes a loop
    call whose three operands are contiguous and whose output is clear of the
    inputs or is one of them, and whose data raise no floating-point flag;
    see the binding's loop_hypot_f64.
    """
    yield from _hypot_value_cases()
    yield from _hypot_layout_cases()
    yield from _hypot_spelling_cases()
    yield from _hypot_flag_cases()


# Every operation that has a hook, with its exact cases.
OPERATIONS: dict[str, Callable[[], Iterator[Case]]] = {
    "argmax": partial(argext_cases, "argmax"),
    "argmin": partial(argext_cases, "argmin"),
    "clip": clip_cases,
    "compress": compress_cases,
    "maskfill": maskfill_cases,
    "nonzero": nonzero_cases,
    "cumsum": partial(cumsum_cases, False),
    "astype": astype_cases,
    "sum": sum_cases,
    "count_nonzero": count_nonzero_cases,
    "histogram": histogram_cases,
    "bincount": bincount_cases,
    "isin": isin_cases,
    "searchsorted": searchsorted_cases,
    "interp": interp_cases,
    "where": where_cases,
    "accumulate": accumulate_cases,
    "reduceat": reduceat_cases,
    "any": any_cases,
    "round": round_cases,
    "median": median_cases,
    "hypot": hypot_cases,
}

# Operations that have kernels which are not bit-identical to NumPy, with the
# cases that run under install(inexact=True).
INEXACT_OPERATIONS: dict[str, Callable[[], Iterator[Case]]] = {
    "cumsum": partial(cumsum_cases, True),
}
