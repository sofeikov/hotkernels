"""The operations the survey measures.

A candidate pairs one NumPy statement with one naive C kernel from `naive.c`.
Candidates are plain data: every field that needs computing is a Python
expression held as a string and evaluated in the candidate's environment, the
same way `timeit` evaluates the NumPy statement. The environment holds `np`,
`rng` (a seeded generator), `n` (the element count), `ptr` (address of an
array), the helper functions in this module, and every array the setup built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Names of the phase 2 candidate groups, as they appear in the result files.
ARGMAX, MASK, COUNT, NONZERO, CUMSUM = "argmax/argmin", "a[mask]", "count_nonzero", "nonzero", "cumsum"
SEARCH, CLIP, ANYALL, TAKE, ISIN = "searchsorted", "clip", "any/all", "take", "isin"
STEPPED = "stepped slices"
# Operations NumPy is known to be good at; they check that the harness is sane.
CONTROL = "control"
# Operations beyond the phase 2 list.
EXTRA, WIDER = "extra", "wider"
# Operations taken from the calls of a program in use: docs/second-shortlist-design.md.
SECOND = "second shortlist"
# Operations taken from the applications study: docs/applications-findings.md.
APPS = "applications"
# Sorting, from the applications study: docs/sort-design.md.
SORT = "argsort"
# Calls a second program in use left to NumPy: docs/coverage-gaps-arm64.md.
GAPS = "coverage gaps"

# Mask densities measured: the worst case for a branch predictor, and a sparse mask.
DENSITIES = (0.5, 0.05)
# Search candidates use at most this many needles, so a 16M haystack stays quick to time.
MAX_NEEDLES = 1 << 20
# Sizes of the fixed-size inputs and outputs.
ISIN_TEST_VALUES = 8
BINCOUNT_BINS = 256
ADD_AT_BINS = 4096
CORRELATE_TAPS = 8
HISTOGRAM_BINS = 64
# Second shortlist: the columns of the array reduced over rows, the length of
# each run reduced by `reduceat`, the bins of the int16 bincount, and the share
# of NaN in the inputs of the maxima that ignore it.
ROW_COLUMNS = 22
RUN_LENGTH = 16
CODE_BINS = 64
NAN_SHARE = 0.05
# Share of True in the mask of `np.where(c, a, nan)`, which callers use to
# blank the few elements that fail a test.
MOSTLY_TRUE = 0.95
# Distinct values of the np.unique labels, binary and ten classes, and the
# table the naive C is given for them.
UNIQUE_LABELS = (2, 10)
UNIQUE_TABLE = 64
# Elements of output the argsort candidates have per element of input: the
# result, and the working memory of the C sorts (see survey/naive.c).
ARGSORT_OUT_PER_ELEMENT = 5
# Coverage gaps: the columns of the mask reduced down its columns, the rows of
# the one reduced over a few long rows, and the columns of the mask whose True
# elements are located. The shapes the program had were about 30 by 31 and 30
# to 50 by 22.
ANY_COLUMNS = 31
ANY_ROWS = 30
NONZERO_COLUMNS = 22
# Relative tolerances for kernels whose NumPy counterpart orders floating-point
# operations differently (pairwise sums, fused multiply-add).
RTOL_F32_SUM = 1e-2
RTOL_F64_SUM = 1e-6
RTOL_F64 = 1e-9


@dataclass(frozen=True)
class Cand:
    """One NumPy statement and the naive C kernel it is measured against."""

    # Unique name, used in tables and by `--only`.
    key: str
    # Which group of candidates this belongs to.
    group: str
    # The NumPy statement that is timed.
    stmt: str
    # Name of the naive C kernel in naive.c.
    cfn: str
    # C argument types, one letter each: p pointer, i int64, f float, d double.
    sig: str
    # Environment to build before timing: name -> expression, evaluated in order.
    setup: dict[str, str]
    # C arguments, one expression per letter of `sig`.
    args: tuple[str, ...]
    # Expression for what the C kernel produced; `ret` is its return value.
    got: str
    # Expression for what NumPy says the answer is.
    expect: str
    # Relative tolerance for comparing `got` with `expect`; None means exact.
    rtol: float | None = None
    # Name of the output buffer in the environment, if the kernel writes one.
    out: str | None = None
    # True when the NumPy statement allocates its result. The C side then
    # allocates and frees its output on every call too.
    alloc: bool = False
    # Expression for the number of elements processed per call.
    work: str = "n"


def last_true(n: int) -> np.ndarray:
    """A boolean array that is False except for its last element."""
    x = np.zeros(n, bool)
    x[-1] = True
    return x


def scattered_sum(idx: np.ndarray, v: np.ndarray, bins: int) -> np.ndarray:
    """What `np.add.at` produces on a zeroed array of `bins` elements."""
    want = np.zeros(bins)
    np.add.at(want, idx, v)
    return want


def assigned(a: np.ndarray, idx: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A copy of `a` after `a[idx] = v`."""
    want = a.copy()
    want[idx] = v
    return want


def filled(x: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    """A copy of `x` after `x[mask] = value`."""
    want = x.copy()
    want[mask] = value
    return want


# Helper functions that setup expressions may call.
HELPERS = {f.__name__: f for f in (last_true, scattered_sum, assigned, filled)}

# How to draw random input of each element type.
_RANDOM = {
    "f32": "rng.random(n, dtype=np.float32)",
    "f64": "rng.random(n)",
    "i64": "rng.integers(-(1 << 40), 1 << 40, n)",
}
_DTYPE = {"f32": "np.float32", "f64": "np.float64", "i64": "np.int64"}


def _argmax() -> list[Cand]:
    """argmax and argmin over contiguous arrays.

    Returns:
        The candidates of this group, in measuring order.
    """
    cands = [
        Cand(f"argmax_{t}", ARGMAX, "np.argmax(x)", f"argmax_{t}", "pii",
             {"x": _RANDOM[t]}, ("ptr(x)", "n", "1"), "ret", "np.argmax(x)")
        for t in ("f32", "f64", "i64")
    ]
    cands.append(Cand("argmin_f64", ARGMAX, "np.argmin(x)", "argmin_f64", "pii",
                      {"x": _RANDOM["f64"]}, ("ptr(x)", "n", "1"), "ret", "np.argmin(x)"))
    return cands


def _controls() -> list[Cand]:
    """Reductions where NumPy is expected to win.

    Returns:
        The candidates of this group, in measuring order.
    """
    return [
        Cand("max_f32 (control)", CONTROL, "np.max(x)", "max_f32", "pip",
             {"x": _RANDOM["f32"], "res": "np.empty(1, np.float32)"},
             ("ptr(x)", "n", "ptr(res)"), "res[0]", "np.max(x)"),
        Cand("sum_f32 (control)", CONTROL, "np.sum(x)", "sum_f32", "piip",
             {"x": _RANDOM["f32"], "res": "np.empty(1, np.float32)"},
             ("ptr(x)", "n", "1", "ptr(res)"), "res[0]", "np.sum(x)", rtol=RTOL_F32_SUM),
        Cand("sum_f64 (control)", CONTROL, "np.sum(x)", "sum_f64", "piip",
             {"x": _RANDOM["f64"], "res": "np.empty(1)"},
             ("ptr(x)", "n", "1", "ptr(res)"), "res[0]", "np.sum(x)", rtol=RTOL_F64_SUM),
        Cand("diff_f64 (control)", CONTROL, "np.diff(x)", "diff_f64", "ppi",
             {"x": _RANDOM["f64"], "out": "np.empty(n)"},
             ("ptr(x)", "ptr(out)", "n"), "out[:n - 1]", "np.diff(x)", out="out", alloc=True),
    ]


def _mask_selection() -> list[Cand]:
    """Boolean mask selection, with the branchy and the branchless naive loop.

    Returns:
        The candidates of this group, in measuring order.
    """
    cands = []
    for p in DENSITIES:
        for t in ("f32", "f64"):
            setup = {"x": f"rng.random(n).astype({_DTYPE[t]})", "mask": f"rng.random(n) < {p}",
                     "out": f"np.empty(n, {_DTYPE[t]})"}
            args = ("ptr(x)", "ptr(mask)", "ptr(out)", "n")
            cands.append(Cand(f"compress_{t} p={p}", MASK, "x[mask]", f"compress_{t}", "pppi",
                              setup, args, "out[:ret]", "x[mask]", out="out", alloc=True))
            if t == "f64":
                cands.append(Cand(f"compress_f64 p={p} (branchless C)", MASK, "x[mask]",
                                  "compress_f64_branchless", "pppi",
                                  setup, args, "out[:ret]", "x[mask]", out="out", alloc=True))
    return cands


def _counting() -> list[Cand]:
    """count_nonzero, and the sums that amount to counting.

    Returns:
        The candidates of this group, in measuring order.
    """
    inputs = {
        "bool": "rng.random(n) < 0.5",
        "i64": "rng.integers(0, 3, n)",
        "f64": "rng.random(n) * (rng.random(n) < 0.5)",
    }
    cands = [
        Cand(f"count_nonzero_{t}", COUNT, "np.count_nonzero(x)", f"count_nonzero_{t}", "pi",
             {"x": expr}, ("ptr(x)", "n"), "ret", "np.count_nonzero(x)")
        for t, expr in inputs.items()
    ]
    cands.append(Cand("sum_bool", WIDER, "np.sum(x)", "count_nonzero_bool", "pi",
                      {"x": inputs["bool"]}, ("ptr(x)", "n"), "ret", "np.sum(x)"))
    cands.append(Cand("sum_u8", WIDER, "np.sum(x)", "sum_u8", "pi",
                      {"x": "rng.integers(0, 256, n, dtype=np.uint8)"},
                      ("ptr(x)", "n"), "ret", "int(np.sum(x))"))
    return cands


def _nonzero() -> list[Cand]:
    """nonzero on a boolean array, with the branchy and the branchless naive loop.

    Returns:
        The candidates of this group, in measuring order.
    """
    cands = []
    for p in DENSITIES:
        setup = {"mask": f"rng.random(n) < {p}", "out": "np.empty(n, np.int64)"}
        for label, cfn in (("", "nonzero_bool"), (" (branchless C)", "nonzero_bool_branchless")):
            cands.append(Cand(f"nonzero_bool p={p}{label}", NONZERO, "np.nonzero(mask)[0]", cfn, "ppi",
                              setup, ("ptr(mask)", "ptr(out)", "n"),
                              "out[:ret]", "np.nonzero(mask)[0]", out="out", alloc=True))
    return cands


def _accumulations() -> list[Cand]:
    """cumsum with and without out=, and a running minimum.

    Returns:
        The candidates of this group, in measuring order.
    """
    inputs = {"f32": _RANDOM["f32"], "f64": _RANDOM["f64"], "i64": "rng.integers(-1000, 1000, n)"}
    cands = []
    for t, expr in inputs.items():
        setup = {"x": expr, "out": f"np.empty(n, {_DTYPE[t]})"}
        args = ("ptr(x)", "ptr(out)", "n")
        cands.append(Cand(f"cumsum_{t}", CUMSUM, "np.cumsum(x)", f"cumsum_{t}", "ppi",
                          setup, args, "out", "np.cumsum(x)", out="out", alloc=True))
        cands.append(Cand(f"cumsum_{t} out=", CUMSUM, "np.cumsum(x, out=out)", f"cumsum_{t}", "ppi",
                          setup, args, "out", "np.cumsum(x)", out="out"))
    cands.append(Cand("cummin_f64", EXTRA, "np.minimum.accumulate(x)", "cummin_f64", "ppi",
                      {"x": _RANDOM["f64"], "out": "np.empty(n)"}, ("ptr(x)", "ptr(out)", "n"),
                      "out", "np.minimum.accumulate(x)", out="out", alloc=True))
    return cands


def _searching() -> list[Cand]:
    """searchsorted and interp: n sorted values, random needles, reported per needle.

    Returns:
        The candidates of this group, in measuring order.
    """
    inputs = {"f64": "rng.random({})", "i64": "rng.integers(0, 1 << 40, {})"}
    cands = [
        Cand(f"searchsorted_{t}", SEARCH, "np.searchsorted(a, v)", f"searchsorted_{t}", "pipip",
             {"m": f"min(n, {MAX_NEEDLES})", "a": f"np.sort({expr.format('n')})",
              "v": expr.format("m"), "out": "np.empty(m, np.int64)"},
             ("ptr(a)", "n", "ptr(v)", "m", "ptr(out)"),
             "out", "np.searchsorted(a, v)", out="out", alloc=True, work="m")
        for t, expr in inputs.items()
    ]
    cands.append(Cand("interp_f64", WIDER, "np.interp(v, xp, fp)", "interp_f64", "pipppi",
                      {"m": f"min(n, {MAX_NEEDLES})", "xp": "np.sort(rng.random(n))", "fp": "rng.random(n)",
                       "v": "rng.random(m)", "out": "np.empty(m)"},
                      ("ptr(v)", "m", "ptr(xp)", "ptr(fp)", "n", "ptr(out)"),
                      "out", "np.interp(v, xp, fp)", rtol=RTOL_F64, out="out", alloc=True, work="m"))
    return cands


def _clip() -> list[Cand]:
    """clip with and without out=.

    Returns:
        The candidates of this group, in measuring order.
    """
    variants = [
        ("f32", "f", _RANDOM["f32"], "0.25", "0.75"),
        ("f64", "d", _RANDOM["f64"], "0.25", "0.75"),
        ("i64", "i", "rng.integers(0, 1000, n)", "250", "750"),
    ]
    cands = []
    for t, letter, expr, lo, hi in variants:
        setup = {"x": expr, "out": f"np.empty(n, {_DTYPE[t]})", "lo": lo, "hi": hi}
        args = ("ptr(x)", "ptr(out)", "n", "lo", "hi")
        sig = "ppi" + letter + letter
        cands.append(Cand(f"clip_{t}", CLIP, "np.clip(x, lo, hi)", f"clip_{t}", sig,
                          setup, args, "out", "np.clip(x, lo, hi)", out="out", alloc=True))
        cands.append(Cand(f"clip_{t} out=", CLIP, "np.clip(x, lo, hi, out=out)", f"clip_{t}", sig,
                          setup, args, "out", "np.clip(x, lo, hi)", out="out"))
    return cands


def _any_all() -> list[Cand]:
    """any and all on inputs where no early exit is possible, and find-first-True.

    Returns:
        The candidates of this group, in measuring order.
    """
    return [
        Cand("any_bool (all False)", ANYALL, "np.any(x)", "any_bool", "pi",
             {"x": "np.zeros(n, bool)"}, ("ptr(x)", "n"), "bool(ret)", "bool(np.any(x))"),
        Cand("all_bool (all True)", ANYALL, "np.all(x)", "all_bool", "pi",
             {"x": "np.ones(n, bool)"}, ("ptr(x)", "n"), "bool(ret)", "bool(np.all(x))"),
        Cand("argmax_bool (find first)", WIDER, "np.argmax(x)", "argmax_u8", "pii",
             {"x": "last_true(n)"}, ("ptr(x)", "n", "1"), "ret", "np.argmax(x)"),
    ]


def _indexing() -> list[Cand]:
    """Gather and scatter with random indices.

    Returns:
        The candidates of this group, in measuring order.
    """
    setup = {"a": "rng.random(n)", "idx": "rng.integers(0, n, n)", "out": "np.empty(n)"}
    args = ("ptr(a)", "n", "ptr(idx)", "n", "ptr(out)")
    return [
        Cand("take_f64 a[idx]", TAKE, "a[idx]", "take_f64", "pipip",
             setup, args, "out", "a[idx]", out="out", alloc=True),
        Cand("take_f64 np.take", TAKE, "np.take(a, idx)", "take_f64", "pipip",
             setup, args, "out", "a[idx]", out="out", alloc=True),
        Cand("take_f64 np.take out=", TAKE, "np.take(a, idx, out=out)", "take_f64", "pipip",
             setup, args, "out", "a[idx]", out="out"),
        Cand("put_f64 a[idx]=v", WIDER, "a[idx] = v", "put_f64", "pippi",
             {"a": "rng.random(n)", "idx": "rng.integers(0, n, n)", "v": "rng.random(n)",
              "want": "assigned(a, idx, v)"},
             ("ptr(a)", "n", "ptr(idx)", "ptr(v)", "n"), "a", "want"),
        Cand(f"add_at_f64 {ADD_AT_BINS} bins", WIDER, "np.add.at(acc, idx, v)", "add_at_f64", "pppi",
             {"idx": f"rng.integers(0, {ADD_AT_BINS}, n)", "v": "rng.random(n)",
              "want": f"scattered_sum(idx, v, {ADD_AT_BINS})", "acc": f"np.zeros({ADD_AT_BINS})"},
             ("ptr(acc)", "ptr(idx)", "ptr(v)", "n"), "acc", "want"),
    ]


def _membership_and_binning() -> list[Cand]:
    """isin with a small test set, bincount and histogram.

    Returns:
        The candidates of this group, in measuring order.
    """
    return [
        Cand(f"isin_i64 m={ISIN_TEST_VALUES}", ISIN, "np.isin(x, test)", "isin_i64", "pipip",
             {"x": "rng.integers(0, 1000, n)", "test": f"rng.integers(0, 1000, {ISIN_TEST_VALUES})",
              "out": "np.empty(n, bool)"},
             ("ptr(x)", "n", "ptr(test)", str(ISIN_TEST_VALUES), "ptr(out)"),
             "out", "np.isin(x, test)", out="out", alloc=True),
        Cand(f"bincount_i64 {BINCOUNT_BINS} bins", EXTRA, f"np.bincount(x, minlength={BINCOUNT_BINS})",
             "bincount_i64", "pipi",
             {"x": f"rng.integers(0, {BINCOUNT_BINS}, n)", "out": f"np.empty({BINCOUNT_BINS}, np.int64)"},
             ("ptr(x)", "n", "ptr(out)", str(BINCOUNT_BINS)),
             "out", f"np.bincount(x, minlength={BINCOUNT_BINS})", out="out", alloc=True),
        Cand(f"histogram_f64 {HISTOGRAM_BINS} bins", WIDER,
             f"np.histogram(x, bins={HISTOGRAM_BINS}, range=(0.0, 1.0))", "histogram_f64", "pipidd",
             {"x": _RANDOM["f64"], "out": f"np.empty({HISTOGRAM_BINS}, np.int64)"},
             ("ptr(x)", "n", "ptr(out)", str(HISTOGRAM_BINS), "0.0", "1.0"),
             "out", f"np.histogram(x, bins={HISTOGRAM_BINS}, range=(0.0, 1.0))[0]", out="out", alloc=True),
    ]


def _elementwise() -> list[Cand]:
    """Elementwise operations beyond the phase 2 list.

    Returns:
        The candidates of this group, in measuring order.
    """
    casts = [
        ("cast_f64_f32", _RANDOM["f64"], "np.float32"),
        ("cast_u8_f32", "rng.integers(0, 256, n, dtype=np.uint8)", "np.float32"),
        ("cast_i64_f64", _RANDOM["i64"], "np.float64"),
    ]
    cands = [
        Cand(name, WIDER, f"x.astype({dst})", name, "ppi",
             {"x": expr, "out": f"np.empty(n, {dst})"}, ("ptr(x)", "ptr(out)", "n"),
             "out", f"x.astype({dst})", out="out", alloc=True)
        for name, expr, dst in casts
    ]
    cands += [
        Cand("where_f64", EXTRA, "np.where(c, a, b)", "where_f64", "ppppi",
             {"c": "rng.random(n) < 0.5", "a": "rng.random(n)", "b": "rng.random(n)", "out": "np.empty(n)"},
             ("ptr(c)", "ptr(a)", "ptr(b)", "ptr(out)", "n"),
             "out", "np.where(c, a, b)", out="out", alloc=True),
        Cand("maskfill_f64 x[mask]=0", WIDER, "x[mask] = 0.0", "maskfill_f64", "ppid",
             {"x": "rng.random(n)", "mask": "rng.random(n) < 0.5", "want": "filled(x, mask, 0.0)"},
             ("ptr(x)", "ptr(mask)", "n", "0.0"), "x", "want"),
        Cand("dot_i64", WIDER, "np.dot(x, y)", "dot_i64", "ppi",
             {"x": "rng.integers(-1000, 1000, n)", "y": "rng.integers(-1000, 1000, n)"},
             ("ptr(x)", "ptr(y)", "n"), "ret", "int(np.dot(x, y))"),
        Cand("array_equal_f64 (equal)", WIDER, "np.array_equal(x, y)", "array_equal_f64", "ppi",
             {"x": "rng.random(n)", "y": "x.copy()"},
             ("ptr(x)", "ptr(y)", "n"), "bool(ret)", "bool(np.array_equal(x, y))"),
        Cand("packbits", WIDER, "np.packbits(x)", "packbits", "ppi",
             {"x": "rng.random(n) < 0.5", "out": "np.empty((n + 7) // 8, np.uint8)"},
             ("ptr(x)", "ptr(out)", "n"), "out", "np.packbits(x)", out="out", alloc=True),
        Cand(f"correlate_f64 k={CORRELATE_TAPS}", WIDER, "np.correlate(x, k, 'valid')", "correlate_f64", "pppii",
             {"x": f"rng.random(n + {CORRELATE_TAPS} - 1)", "k": f"rng.random({CORRELATE_TAPS})",
              "out": "np.empty(n)"},
             ("ptr(x)", "ptr(k)", "ptr(out)", "n", str(CORRELATE_TAPS)),
             "out", "np.correlate(x, k, 'valid')", rtol=RTOL_F64, out="out", alloc=True),
    ]
    return cands


def _stepped_slices() -> list[Cand]:
    """Operations on `xs[::2]` and `x[::-1]`; n is the number of elements used.

    Returns:
        The candidates of this group, in measuring order.
    """
    two = {"xs": "rng.random(2 * n)", "ys": "rng.random(2 * n)", "out": "np.empty(n)"}
    add_args = ("ptr(xs)", "ptr(ys)", "ptr(out)", "n", "2")
    return [
        Cand("argmax_f64 [::2]", STEPPED, "np.argmax(xs[::2])", "argmax_f64", "pii",
             {"xs": "rng.random(2 * n)"}, ("ptr(xs)", "n", "2"), "ret", "np.argmax(xs[::2])"),
        Cand("sum_f64 [::2]", STEPPED, "np.sum(xs[::2])", "sum_f64", "piip",
             {"xs": "rng.random(2 * n)", "res": "np.empty(1)"},
             ("ptr(xs)", "n", "2", "ptr(res)"), "res[0]", "np.sum(xs[::2])", rtol=RTOL_F64_SUM),
        Cand("copy_f64 [::2] out=", STEPPED, "np.copyto(out, xs[::2])", "copy_f64_strided", "ppii",
             {"xs": "rng.random(2 * n)", "out": "np.empty(n)"},
             ("ptr(xs)", "ptr(out)", "n", "2"), "out", "xs[::2]", out="out"),
        # A reversed slice starts at the last element and steps backwards.
        Cand("copy_f64 [::-1] out=", STEPPED, "np.copyto(out, x[::-1])", "copy_f64_strided", "ppii",
             {"x": "rng.random(n)", "out": "np.empty(n)"},
             ("ptr(x) + x.itemsize * (n - 1)", "ptr(out)", "n", "-1"), "out", "x[::-1]", out="out"),
        Cand("add_f64 [::2]", STEPPED, "xs[::2] + ys[::2]", "add_f64_strided", "pppii",
             two, add_args, "out", "xs[::2] + ys[::2]", out="out", alloc=True),
        Cand("add_f64 [::2] out=", STEPPED, "np.add(xs[::2], ys[::2], out=out)", "add_f64_strided", "pppii",
             two, add_args, "out", "xs[::2] + ys[::2]", out="out"),
    ]


def _second_shortlist() -> list[Cand]:
    """The candidates of docs/second-shortlist-design.md that had no survey row.

    `np.where` on float64, `np.sum` of a mask and `np.bincount` on int64 are
    measured by earlier groups.

    Returns:
        The candidates of this group, in measuring order.
    """
    with_nan = f"np.where(rng.random(n) < {NAN_SHARE}, np.nan, rng.random(n))"
    pair = {"a": "rng.standard_normal(n)", "b": "rng.standard_normal(n)", "out": "np.empty(n)"}
    pair_args = ("ptr(a)", "ptr(b)", "ptr(out)", "n")
    return [
        Cand("cumsum_bool", SECOND, "np.cumsum(mask)", "cumsum_bool", "ppi",
             {"mask": "rng.random(n) < 0.5", "out": "np.empty(n, np.int64)"},
             ("ptr(mask)", "ptr(out)", "n"), "out", "np.cumsum(mask)", out="out", alloc=True),
        Cand("cummax_i64", SECOND, "np.maximum.accumulate(x)", "cummax_i64", "ppi",
             {"x": _RANDOM["i64"], "out": "np.empty(n, np.int64)"},
             ("ptr(x)", "ptr(out)", "n"), "out", "np.maximum.accumulate(x)", out="out", alloc=True),
        Cand("cumfmax_f64", SECOND, "np.fmax.accumulate(x)", "cumfmax_f64", "ppi",
             {"x": with_nan, "out": "np.empty(n)"},
             ("ptr(x)", "ptr(out)", "n"), "out", "np.fmax.accumulate(x)", out="out", alloc=True),
        Cand(f"fmax_reduceat_f64 runs of {RUN_LENGTH}", SECOND, "np.fmax.reduceat(x, starts)",
             "fmax_reduceat_f64", "pppii",
             {"x": with_nan, "starts": f"np.arange(0, n, {RUN_LENGTH}, dtype=np.int64)",
              "out": "np.empty(len(starts))"},
             ("ptr(x)", "ptr(starts)", "ptr(out)", "n", "len(starts)"),
             "out", "np.fmax.reduceat(x, starts)", out="out", alloc=True),
        Cand(f"any_rows_bool {ROW_COLUMNS} columns", SECOND, "m.any(axis=1)", "any_rows_bool", "ppii",
             {"m": f"rng.random((max(n // {ROW_COLUMNS}, 1), {ROW_COLUMNS})) < 0.03",
              "out": "np.empty(len(m), bool)"},
             ("ptr(m)", "ptr(out)", "len(m)", str(ROW_COLUMNS)),
             "out", "m.any(axis=1)", out="out", alloc=True, work="m.size"),
        Cand(f"bincount_i16 {CODE_BINS} bins", SECOND, f"np.bincount(x, minlength={CODE_BINS})",
             "bincount_i16", "pipi",
             {"x": f"rng.integers(0, {CODE_BINS}, n).astype(np.int16)", "out": f"np.empty({CODE_BINS}, np.int64)"},
             ("ptr(x)", "n", "ptr(out)", str(CODE_BINS)),
             "out", f"np.bincount(x, minlength={CODE_BINS})", out="out", alloc=True),
        # The C library's hypot is what NumPy calls, so the two agree exactly.
        Cand("hypot_f64", SECOND, "np.hypot(a, b)", "hypot_f64", "pppi",
             pair, pair_args, "out", "np.hypot(a, b)", out="out", alloc=True),
        Cand("hypot_f64 (square-root form)", SECOND, "np.hypot(a, b)", "hypot_sqrt_f64", "pppi",
             pair, pair_args, "out", "np.hypot(a, b)", rtol=RTOL_F64, out="out", alloc=True),
        Cand("where_i64", SECOND, "np.where(c, a, b)", "where_i64", "ppppi",
             {"c": "rng.random(n) < 0.5", "a": _RANDOM["i64"], "b": _RANDOM["i64"],
              "out": "np.empty(n, np.int64)"},
             ("ptr(c)", "ptr(a)", "ptr(b)", "ptr(out)", "n"),
             "out", "np.where(c, a, b)", out="out", alloc=True),
        Cand("where_f64 y=nan", SECOND, "np.where(c, a, np.nan)", "where_scalar_f64", "pppid",
             {"c": f"rng.random(n) < {MOSTLY_TRUE}", "a": "rng.random(n)", "out": "np.empty(n)"},
             ("ptr(c)", "ptr(a)", "ptr(out)", "n", "np.nan"),
             "out", "np.where(c, a, np.nan)", out="out", alloc=True),
    ]


def _applications() -> list[Cand]:
    """The candidates found by the applications study: `np.unique` on int64 labels.

    Returns:
        The candidates of this group, in measuring order.
    """
    return [
        Cand(f"unique_i64 {labels} values", APPS, "np.unique(x)", "unique_i64", "pipi",
             {"x": f"rng.integers(0, {labels}, n)", "out": f"np.empty({UNIQUE_TABLE}, np.int64)"},
             ("ptr(x)", "n", "ptr(out)", str(UNIQUE_TABLE)),
             "out[:ret]", "np.unique(x)", out="out", alloc=True)
        for labels in UNIQUE_LABELS
    ]


def _argsort() -> list[Cand]:
    """Stable argsort of 64-bit keys, against a merge sort and against a radix sort in C.

    The default kind is measured too, against the radix sort, on data without
    equal elements: there every correct sort gives the same answer, and with
    ties NumPy's quicksort gives an order of its own.

    Returns:
        The candidates of this group, in measuring order.
    """
    out = f"np.empty({ARGSORT_OUT_PER_ELEMENT} * n, np.int64)"
    data = {
        "f64": _RANDOM["f64"],
        "i64": "rng.integers(-(1 << 62), 1 << 62, n)",
        "u64": "rng.integers(0, 1 << 63, n).astype(np.uint64) * 2",
    }
    args = ("ptr(x)", "ptr(out)", "n")
    cands = []
    for t, x in data.items():
        for how, cfn in (("merge sort", f"argsort_merge_{t}"), ("radix sort", f"argsort_radix_{t}")):
            cands.append(Cand(f"argsort_{t} stable ({how})", SORT, "np.argsort(x, kind='stable')", cfn, "ppi",
                              {"x": x, "out": out}, args, "out[:n]", "np.argsort(x, kind='stable')",
                              out="out", alloc=True))
    for t in ("f64", "i64"):
        cands.append(Cand(f"argsort_{t} default kind (radix sort)", SORT, "np.argsort(x)", f"argsort_radix_{t}",
                          "ppi", {"x": data[t], "out": out}, args, "out[:n]", "np.argsort(x)",
                          out="out", alloc=True))
    return cands


def _coverage_gaps() -> list[Cand]:
    """The calls of docs/coverage-gaps-arm64.md that had no survey row.

    `np.searchsorted` on int64 with the left side and `astype` from float64 to
    float32 are measured by earlier groups.

    Returns:
        The candidates of this group, in measuring order.
    """
    haystack = {"m": f"min(n, {MAX_NEEDLES})", "a": "np.sort(rng.integers(0, 1 << 40, n))",
                "v": "rng.integers(0, 1 << 40, m)", "out": "np.empty(m, np.int64)"}
    one = {"a": "np.sort(rng.integers(0, 1 << 40, n))", "v": "rng.integers(0, 1 << 40, 1)", "key": "v[0]",
           "out": "np.empty(1, np.int64)"}
    cands = [
        Cand("searchsorted_i64 side=right", GAPS, "np.searchsorted(a, v, side='right')", "searchsorted_right_i64",
             "pipip", haystack, ("ptr(a)", "n", "ptr(v)", "m", "ptr(out)"),
             "out", "np.searchsorted(a, v, side='right')", out="out", alloc=True, work="m"),
        Cand("searchsorted_i64 one needle", GAPS, "np.searchsorted(a, key)", "searchsorted_i64",
             "pipip", one, ("ptr(a)", "n", "ptr(v)", "1", "ptr(out)"),
             "out[0]", "np.searchsorted(a, key)", out="out", work="1"),
    ]
    cands += [
        Cand(f"sum_{t} (NumPy's order)", GAPS, "np.sum(x)", f"sum_pairwise_{t}", "pip",
             {"x": f"rng.standard_normal(n, dtype={_DTYPE[t]})", "res": f"np.empty(1, {_DTYPE[t]})"},
             ("ptr(x)", "n", "ptr(res)"), "res[0]", "np.sum(x)")
        for t in ("f32", "f64")
    ]
    cands += [
        Cand("sum_i64", GAPS, "np.sum(x)", "sum_i64", "pi",
             {"x": _RANDOM["i64"]}, ("ptr(x)", "n"), "ret", "int(np.sum(x))"),
        Cand(f"any_cols_bool {ANY_COLUMNS} columns", GAPS, "m.any(axis=0)", "any_cols_bool", "ppii",
             {"m": f"rng.random((max(n // {ANY_COLUMNS}, 1), {ANY_COLUMNS})) < 0.03",
              "out": f"np.empty({ANY_COLUMNS}, bool)"},
             ("ptr(m)", "ptr(out)", "len(m)", str(ANY_COLUMNS)),
             "out", "m.any(axis=0)", out="out", alloc=True, work="m.size"),
        Cand(f"any_cols_bool {ANY_ROWS} rows", GAPS, "m.any(axis=0)", "any_cols_bool", "ppii",
             {"m": f"rng.random(({ANY_ROWS}, max(n // {ANY_ROWS}, 1))) < 0.03",
              "out": "np.empty(m.shape[1], bool)"},
             ("ptr(m)", "ptr(out)", str(ANY_ROWS), "m.shape[1]"),
             "out", "m.any(axis=0)", out="out", alloc=True, work="m.size"),
    ]
    cands += [
        Cand(f"nonzero_2d_bool {NONZERO_COLUMNS} columns p={p}", GAPS, "np.nonzero(m)", "nonzero_2d_bool", "ppii",
             {"m": f"rng.random((max(n // {NONZERO_COLUMNS}, 1), {NONZERO_COLUMNS})) < {p}",
              "out": "np.empty(2 * m.size, np.int64)"},
             ("ptr(m)", "ptr(out)", "len(m)", str(NONZERO_COLUMNS)),
             "np.concatenate((out[:ret], out[m.size:m.size + ret]))", "np.concatenate(np.nonzero(m))",
             out="out", alloc=True, work="m.size")
        for p in DENSITIES
    ]
    cands.append(Cand("cast_i16_i64", GAPS, "x.astype(np.int64)", "cast_i16_i64", "ppi",
                      {"x": "rng.integers(-(1 << 15), 1 << 15, n).astype(np.int16)", "out": "np.empty(n, np.int64)"},
                      ("ptr(x)", "ptr(out)", "n"), "out", "x.astype(np.int64)", out="out", alloc=True))
    return cands


def candidates() -> list[Cand]:
    """List every candidate the survey measures.

    Returns:
        All candidates of all groups, in measuring order.
    """
    groups = (_argmax, _controls, _mask_selection, _counting, _nonzero, _accumulations, _searching,
              _clip, _any_all, _indexing, _membership_and_binning, _elementwise, _stepped_slices,
              _second_shortlist, _applications, _argsort, _coverage_gaps)
    return [cand for group in groups for cand in group()]
