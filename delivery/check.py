"""Runs the cases: each call once on plain NumPy and once with hotkernels
installed, and compares everything a caller could observe.

Two runs of a case agree when the result has the same type, element type,
shape, flags and bytes (or, for a kernel that declares a tolerance, values
within it), the same exception was raised with the same message, the same
warnings were issued, and the arguments are in the same state afterwards.
On top of that each case says whether a kernel should have taken the call,
and the binding's counters are checked against it: a condition in the
binding that silently stopped matching would pass every comparison while
delivering nothing.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import importlib
import multiprocessing
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

import hotkernels
from hotkernels.environment import BINDING_MODULE
from hotkernels.hooks import BOTTLENECK_HOOKS, FUNCTION, HOOKS, KERNELS, METHOD

from .cases import ANY, INEXACT_OPERATIONS, OPERATIONS, Case

log = logging.getLogger(__name__)

# Bytes of a digest that stands in for an array's contents.
DIGEST_BYTES = 16
# Longest description of a value shown in a failure.
SHOWN_CHARACTERS = 300

# Bits of the significand NumPy reports (`np.finfo(np.longdouble).nmant`)
# for the x87 80-bit extended format, which a long double is on x86-64 Linux.
# The format stores 64 significand bits with the leading one explicit, and
# NumPy counts the 63 after it.
X87_SIGNIFICAND_BITS = 63
# Bytes of an x87 80-bit value: 8 of significand and 2 of sign and exponent.
# NumPy stores it in 16 bytes and leaves the 6 after these as they were, so
# they differ from run to run and say nothing about the value.
X87_VALUE_BYTES = 10

# What a case ended in.
RETURNED = "returned"
RAISED = "raised"

# Shares each operation's cases are split into when the check runs in several
# processes. Enough that the operation with the most work (clip, about 40
# seconds of it on an Apple M1 Pro) does not leave one process running alone
# at the end; few enough that each share is still a second or more of work
# for the larger operations, against 5 ms to install and uninstall.
SHARDS_PER_OPERATION = 12
# How worker processes are started. Spawned, not forked: a fork of a process
# whose NumPy has started its BLAS threads can deadlock, and spawn behaves the
# same on macOS and Linux.
START_METHOD = "spawn"


@dataclass(frozen=True)
class Outcome:
    """Everything observable about one run of a case."""

    # (RETURNED, description of the result) or (RAISED, exception type, message).
    ending: tuple
    # (category name, message) of every warning issued, in order.
    warnings: tuple
    # Description of the arguments after the call.
    arguments: tuple
    # The result itself, kept only for cases compared within a tolerance.
    raw: object = None


@dataclass(frozen=True)
class Failure:
    """One case whose two runs disagree, or whose kernel expectation failed."""

    # The case's label.
    label: str
    # What disagreed.
    problem: str
    # What plain NumPy gave, shortened.
    expected: str
    # What the run with hotkernels gave, shortened.
    got: str


@dataclass(frozen=True)
class Summary:
    """The outcome of one operation's cases."""

    # Operation name, with a suffix for the run under install(inexact=True).
    operation: str
    # Number of cases run.
    cases: int
    # Number of cases a kernel took.
    taken: int
    # The cases that failed.
    failures: tuple[Failure, ...]


def _digest(data: bytes) -> str:
    """Return a short hash that stands in for a buffer's contents."""
    return hashlib.blake2b(data, digest_size=DIGEST_BYTES).hexdigest()


def _value_bytes(array: np.ndarray) -> bytes:
    """Return the bytes of an array's elements that hold their values, in C order.

    That is every byte, except for a long double or complex long double in
    the x87 80-bit format (x86-64 Linux): there only the first
    X87_VALUE_BYTES of each 16-byte long double are kept, since NumPy leaves
    the padding after them as it was. On ARM64 macOS a long double is a
    double, and on ARM64 Linux an IEEE quad with no padding; both keep every
    byte.

    Args:
        array: Any array, of any layout.

    Returns:
        The bytes to compare: equal for two arrays of one element type and
        shape exactly when their elements have the same bits.
    """
    contiguous = np.ascontiguousarray(array)
    x87 = np.finfo(np.longdouble).nmant == X87_SIGNIFICAND_BITS
    if x87 and contiguous.dtype.type in (np.longdouble, np.clongdouble):
        # A complex long double is two long doubles, each padded the same way.
        elements = contiguous.view(np.uint8).reshape(-1, np.dtype(np.longdouble).itemsize)
        return elements[:, :X87_VALUE_BYTES].tobytes()
    return contiguous.tobytes()


def describe(value: object) -> tuple:
    """Reduce a value to a comparable description.

    An array becomes its type, element type, shape, flags and a hash of its
    bytes, so +0.0 differs from -0.0 and NaNs are compared by their bits. A
    masked array adds its mask. Containers are described element by element.
    An x87 long double is compared by its value bytes only (`_value_bytes`).

    Args:
        value: A result, or an argument after a call.

    Returns:
        A nested tuple that is equal for two values exactly when a caller
        could not tell them apart.
    """
    if isinstance(value, np.ma.MaskedArray):
        return ("masked", type(value).__name__, describe(np.asarray(value.data)), describe(np.ma.getmaskarray(value)))
    if isinstance(value, np.ndarray):
        contents = repr(value.tolist()) if value.dtype.hasobject else _digest(_value_bytes(value))
        flags = (value.flags.c_contiguous, value.flags.f_contiguous, value.flags.writeable)
        return ("array", type(value).__name__, value.dtype.str, value.shape, flags, contents)
    if isinstance(value, np.generic):
        return ("scalar", type(value).__name__, value.dtype.str, _digest(_value_bytes(np.asarray(value))))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(describe(item) for item in value))
    return ("object", type(value).__name__, repr(value))


def capture(case: Case, arguments: tuple | None = None) -> Outcome:
    """Run one case and record everything observable about it.

    Args:
        case: The case.
        arguments: Its arguments, built by `case.make()` for this run, or
            None to build them here. Building them first lets a caller read
            the counters after the arguments exist and before the call: the
            ufunc lane counts the ufunc calls that build them.

    Returns:
        The outcome, with the raw result kept when the case has a tolerance.
    """
    if arguments is None:
        arguments = case.make()
    raw = None
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        try:
            raw = case.call(*arguments)
            ending = (RETURNED, describe(raw))
        except Exception as error:  # Any exception is a result to compare.
            ending = (RAISED, type(error).__name__, str(error))
    recorded = tuple((w.category.__name__, str(w.message)) for w in issued)
    return Outcome(ending, recorded, describe(arguments), raw if case.tolerance else None)


def _within_tolerance(case: Case, expected: Outcome, got: Outcome) -> bool:
    """Say whether a result is within the tolerance its kernel's spec declares.

    The rule is the judge's: NaN and infinity must be in the same positions,
    and a finite element may differ by the spec's number of units in the last
    place, taken at the magnitude the spec's scale function gives.

    Args:
        case: A case with a tolerance, whose one argument is the input array.
        expected: The run on plain NumPy.
        got: The run with hotkernels.

    Returns:
        True when both runs returned arrays of one type and shape that agree
        within the tolerance.
    """
    a, b = expected.raw, got.raw
    if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
        return False
    if type(a) is not type(b) or a.dtype != b.dtype or a.shape != b.shape:
        return False
    (x,) = case.make()
    scale = np.asarray(case.tolerance.tolerance_scale(np.asarray(x).ravel()), dtype=np.float64)
    special = ~np.isfinite(a) | ~np.isfinite(b)
    identical = (a == b) | (np.isnan(a) & np.isnan(b))
    with np.errstate(invalid="ignore", over="ignore"):
        magnitude = np.minimum(np.abs(scale), np.finfo(a.dtype).max).astype(a.dtype)
        close = np.abs(b - a) <= case.tolerance.tolerance_ulp * np.spacing(magnitude)
    return bool(np.where(special, identical, close).all())


def _shorten(value: object) -> str:
    """Format a value for a failure message, cut to a readable length."""
    text = repr(value)
    return text if len(text) <= SHOWN_CHARACTERS else text[:SHOWN_CHARACTERS] + "..."


def _disagreement(case: Case, expected: Outcome, got: Outcome) -> Failure | None:
    """Compare the two runs of a case.

    Args:
        case: The case.
        expected: The run on plain NumPy.
        got: The run with hotkernels installed.

    Returns:
        A Failure naming the first thing that differs, or None.
    """
    if expected.ending != got.ending:
        both_returned = expected.ending[0] == got.ending[0] == RETURNED
        if not (case.tolerance and both_returned and _within_tolerance(case, expected, got)):
            return Failure(case.label, "result differs", _shorten(expected.ending), _shorten(got.ending))
    if expected.warnings != got.warnings:
        return Failure(case.label, "warnings differ", _shorten(expected.warnings), _shorten(got.warnings))
    if expected.arguments != got.arguments:
        return Failure(case.label, "arguments differ after the call", _shorten(expected.arguments), _shorten(got.arguments))
    return None


def _expectation_failure(case: Case, own_kernels: tuple[str, ...], before: dict, after: dict) -> Failure | None:
    """Check the binding's counters against what a case expects.

    Args:
        case: The case.
        own_kernels: The kernels of the case's operation.
        before: Calls taken per kernel before the case ran.
        after: The same afterwards.

    Returns:
        A Failure when the expected kernel did not run, or when one of the
        operation's kernels ran for a call that must be left to NumPy.
    """
    if case.expect == ANY:
        return None
    ran = [kernel for kernel in own_kernels if after[kernel] > before[kernel]]
    if case.expect is None and ran:
        return Failure(case.label, "a kernel took a call that belongs to NumPy", "no kernel", ", ".join(ran))
    if case.expect is not None and after[case.expect] <= before[case.expect]:
        return Failure(case.label, "the kernel did not take the call", case.expect, ", ".join(ran) or "no kernel")
    return None


def operation_kernels(operation: str) -> tuple[str, ...]:
    """Return the names of the kernels that serve an operation."""
    return tuple(kernel.name for kernel in KERNELS if kernel.operation == operation)


@dataclass(frozen=True)
class CaseFilter:
    """Which cases a run keeps. A dataclass rather than a closure, so that it
    can be sent to the worker processes."""

    # Keep only cases of the operation with this name; None keeps every operation.
    only: str | None = None
    # Keep only cases whose label contains this text; None keeps every case.
    text: str | None = None

    def __call__(self, case: Case) -> bool:
        """Say whether a case passes both filters.

        Args:
            case: The case.

        Returns:
            True when the case runs.
        """
        return (self.only is None or case.operation == self.only) and (self.text is None or self.text in case.label)


def summary_name(operation: str, inexact: bool) -> str:
    """Return the name an operation's summary carries.

    Args:
        operation: Operation name.
        inexact: Whether its cases run under install(inexact=True).

    Returns:
        The operation's name, with a suffix for the inexact run.
    """
    return f"{operation} (inexact)" if inexact else operation


def run_operation(operation: str, cases: Iterable[Case], inexact: bool, keep: Callable[[Case], bool]) -> Summary:
    """Run one operation's cases on plain NumPy, then with hotkernels installed.

    Args:
        operation: Operation name.
        cases: The cases.
        inexact: Whether to install the kernels that are not bit-identical.
        keep: Says which cases to run.

    Returns:
        The summary. hotkernels is left uninstalled.
    """
    selected = [case for case in cases if keep(case)]
    hotkernels.uninstall()
    expected = [capture(case) for case in selected]
    hotkernels.install(inexact=inexact)
    binding = importlib.import_module(BINDING_MODULE)
    own_kernels = operation_kernels(operation)
    failures, taken = [], 0
    try:
        for case, want in zip(selected, expected):
            arguments = case.make()
            before = binding.counters()[0]
            got = capture(case, arguments)
            after = binding.counters()[0]
            taken += any(after[kernel] > before[kernel] for kernel in own_kernels)
            failure = _disagreement(case, want, got) or _expectation_failure(case, own_kernels, before, after)
            if failure:
                failures.append(failure)
    finally:
        hotkernels.uninstall()
    return Summary(summary_name(operation, inexact), len(selected), taken, tuple(failures))


# Name of the summary line for the checks of install() and uninstall() themselves.
INSTALLATION = "install and uninstall"
# What a function or method hook's name is an attribute of, by the part of the
# name before the dot: `function.sum` is np.sum, `method.sum` is
# np.ndarray.sum and `ufunc.reduceat` is np.ufunc.reduceat. Each such object
# must be the same object after install() and after uninstall().
_HOOK_OWNERS = {"function": np, "method": np.ndarray, "ufunc": np.ufunc}
# The hooks behind the clip ufunc's loop, which also take every call the
# function and method hooks of clip take.
_CLIP_LOOP_HOOKS = ("loop.clip.f32", "loop.clip.f64")
# Element count of the arrays the check of clip's direct path clips.
_DIRECT_CLIP_N = 16


# bottleneck's functions whose call pointers hotkernels changes.
_BOTTLENECK_FUNCTIONS = ("nanmean", "nanstd", "nanmin", "nanmax")

# Run in a fresh interpreter: install() must not import bottleneck, and an
# import after it must be served, with bottleneck's own answer, and stop being
# served after uninstall(). Prints "ok" when all of that holds.
_LATE_IMPORT_PROGRAM = """
import sys, numpy as np, hotkernels
hotkernels.install()
assert "bottleneck" not in sys.modules, "install() imported bottleneck"
import bottleneck
x = np.random.default_rng(0).standard_normal((300, 41))
taken = bottleneck.nanstd(x, axis=0, ddof=1)
calls = {k.name: k.calls for k in hotkernels.report().kernels}["nanstd_cols_f64"]
assert calls == 1, f"the late import was not served: {calls} calls"
hotkernels.uninstall()
assert bottleneck.nanstd(x, axis=0, ddof=1).tobytes() == taken.tobytes(), "results differ"
assert not any(type(f).__name__ == "_BottleneckFinder" for f in sys.meta_path), "the finder is left behind"
print("ok")
"""

# Run in a fresh interpreter: a `bottleneck.reduce` whose functions are not
# what the hook was written for (here Python functions) must leave the four
# hooks off, say so in report(), and keep the functions working. Prints "ok".
_FOREIGN_LAYOUT_PROGRAM = """
import sys, types, hotkernels
fake = types.ModuleType("bottleneck.reduce")
for name in ("nanmean", "nanstd", "nanmin", "nanmax"):
    setattr(fake, name, lambda a, axis=None, ddof=0: "bottleneck's own")
sys.modules["bottleneck.reduce"] = fake
hotkernels.install()
report = hotkernels.report()
assert report.installed, "install() did not install the rest"
assert sum("bottleneck" in p for p in report.problems) == 4, report.problems
assert fake.nanmean(None, axis=0) == "bottleneck's own"
print("ok")
"""


def _runs_clean(program: str) -> bool:
    """Run a check in a fresh interpreter and say whether it printed "ok".

    Args:
        program: Python source that prints "ok" when its statement holds.

    Returns:
        Whether it did, exiting without an error.
    """
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        log.debug("fresh interpreter failed: %s", done.stderr)
    return done.returncode == 0 and done.stdout.strip() == "ok"


def _bottleneck_objects() -> list[object]:
    """Return bottleneck's functions that hotkernels changes pointers in."""
    import bottleneck

    return [getattr(bottleneck, name) for name in _BOTTLENECK_FUNCTIONS]


def _numpy_objects() -> list[object]:
    """Return the NumPy functions and methods hotkernels changes pointers in.

    Read from the package's own table of hooks, so a hook added there is
    checked here without being listed again. The ufunc call hook changes a
    field of every ufunc, not an attribute, and is not of either kind.
    """
    names = (hook.name.split(".", 1) for hook in HOOKS if hook.kind in (FUNCTION, METHOD))
    return [getattr(_HOOK_OWNERS[owner], attribute) for owner, attribute in names]


def _kernel_calls(binding) -> int:
    """Return how many calls all kernels have taken."""
    return sum(binding.counters()[0].values())


def _exercise() -> None:
    """Make one call to each kind of hook that a kernel takes when installed."""
    x = np.arange(1000, dtype=np.float64)
    mask = x > 500
    x.argmax(), np.clip(x, 0.0, 1.0), np.count_nonzero(x), x.astype(np.uint8).sum(), x[mask]
    x[mask] = 0.0


def _clip_direct_path(binding) -> bool:
    """Say whether `np.clip` and `x.clip` reach the clip kernels without the clip loop.

    The loop hook takes every call the function and method hooks take, so
    the kernel counters cannot tell which ran. With the loop hooks removed
    for the moment, only the direct path can move them.

    Args:
        binding: The extension module, with hotkernels installed.

    Returns:
        True when each of the four calls, `np.clip` and `x.clip` on float32
        and on float64, moved its kernel's counter and gave NumPy's answer.
    """
    for hook in _CLIP_LOOP_HOOKS:
        binding.remove(hook)
    try:
        served = []
        for dtype, kernel in ((np.float32, "clip_f32"), (np.float64, "clip_f64")):
            x = np.linspace(-1.0, 1.0, _DIRECT_CLIP_N, dtype=dtype)
            expected = np.minimum(np.maximum(x, dtype(-0.5)), dtype(0.5))
            for clip in (np.clip, np.ndarray.clip):
                before = binding.counters()[0][kernel]
                result = clip(x, -0.5, 0.5)
                right = result.dtype == dtype and np.array_equal(result, expected)
                served.append(right and binding.counters()[0][kernel] == before + 1)
        return all(served)
    finally:
        for hook in _CLIP_LOOP_HOOKS:
            binding.apply(hook)


# Shapes for the checks that one Python call of argmax counts once in
# report(), however many times NumPy calls the type slot for it: rows of
# three, which the slot passes on one by one, and columns of 1,000, which the
# slot takes one by one with `axis=0`.
_COUNTED_ROWS_SHAPE = (300, 3, 2)
_COUNTED_COLUMNS_SHAPE = (1000, 4)


def _counted_once(binding, call: Callable[[], object], column: int) -> bool:
    """Say whether one Python call moved the argmax counters of `report()` by exactly one.

    Args:
        binding: The extension module, with hotkernels installed.
        call: Makes one call of np.argmax or x.argmax that NumPy splits into
            many calls of the type slot.
        column: 0 for the kernel counters ("Kernel calls"), 1 for the
            operation counters ("Left to NumPy").

    Returns:
        True when the counters of that column together moved by one and
        those of the other column did not move.
    """
    before = binding.counters()
    call()
    after = binding.counters()
    moved = [sum(after[c].values()) - sum(before[c].values()) for c in (0, 1)]
    return moved[column] == 1 and moved[1 - column] == 0


def installation_checks() -> Summary:
    """Check install(), uninstall(), enabled() and report() themselves.

    Returns:
        A summary with one failure per statement that does not hold: NumPy's
        objects keep their identity, uninstall() leaves no hook behind, a
        second install() changes nothing, enabled() restores the state it
        found, arguments included, float cumsum is only taken under install(inexact=True), and
        one Python call of argmax counts once in report() however many
        times NumPy calls the type slot for it.
    """
    binding = importlib.import_module(BINDING_MODULE)
    checks: list[tuple[str, bool]] = []
    hotkernels.uninstall()
    before = _numpy_objects()
    bottleneck_before = _bottleneck_objects()

    hotkernels.install()
    checks.append(("NumPy's objects are the same objects after install()", all(a is b for a, b in zip(before, _numpy_objects()))))
    checks.append(("bottleneck's functions are the same objects after install()",
                   all(a is b for a, b in zip(bottleneck_before, _bottleneck_objects()))))
    checks.append(("bottleneck's functions are hooked by install()", all(binding.applied(h) for h in BOTTLENECK_HOOKS)))
    _exercise()
    calls = _kernel_calls(binding)
    checks.append(("kernels take calls after install()", calls >= 6))
    hotkernels.install()
    checks.append(("a second install() does not reset the counters", _kernel_calls(binding) == calls))
    checks.append(("report() counts the calls", sum(k.calls for k in hotkernels.report().kernels) == calls))
    checks.append(("report() renders when installed", "installed" in str(hotkernels.report())))
    checks.append(("np.clip and x.clip reach the kernels without the clip loop", _clip_direct_path(binding)))
    rows = np.ones(_COUNTED_ROWS_SHAPE)
    checks.append(("one np.argmax over many short rows counts once as left to NumPy",
                   _counted_once(binding, lambda: np.argmax(rows, axis=-1), 1)))
    columns = np.ones(_COUNTED_COLUMNS_SHAPE)
    checks.append(("one x.argmax(axis=0) over many long columns counts once as a kernel call",
                   _counted_once(binding, lambda: columns.argmax(axis=0), 0)))
    np.cumsum(np.ones(1000))
    checks.append(("float cumsum is left to NumPy by install()", binding.counters()[0]["cumsum_f64"] == 0))
    with hotkernels.enabled():
        pass
    checks.append(("enabled() leaves an existing installation in place", hotkernels.report().installed))
    with hotkernels.enabled(inexact=True):
        np.cumsum(np.ones(1000))
        checks.append(("enabled(inexact=True) inside install() takes float cumsum",
                       binding.counters()[0]["cumsum_f64"] == 1))
    checks.append(("enabled(inexact=True) inside install() puts install()'s arguments back",
                   hotkernels.report().installed and not hotkernels.report().inexact))

    hotkernels.uninstall()
    hotkernels.install(bottleneck=False)
    checks.append(("install(bottleneck=False) leaves bottleneck alone",
                   not any(binding.applied(h) for h in BOTTLENECK_HOOKS)
                   and all("bottleneck=False" in k.note for k in hotkernels.report().kernels if k.numpy_call.startswith("bn."))))
    hotkernels.install()
    checks.append(("install() after install(bottleneck=False) hooks bottleneck",
                   all(binding.applied(h) for h in BOTTLENECK_HOOKS)))

    hotkernels.install(inexact=True)
    np.cumsum(np.ones(1000))
    checks.append(("float cumsum is taken under install(inexact=True)", binding.counters()[0]["cumsum_f64"] == 1))

    hotkernels.uninstall()
    binding.reset_counters()
    _exercise()
    checks.append(("no hook is applied after uninstall()", not any(binding.applied(name) for name in binding.hooks())))
    checks.append(("no kernel takes a call after uninstall()", _kernel_calls(binding) == 0))
    checks.append(("NumPy's objects are the same objects after uninstall()", all(a is b for a, b in zip(before, _numpy_objects()))))
    checks.append(("bottleneck's functions are the same objects after uninstall()",
                   all(a is b for a, b in zip(bottleneck_before, _bottleneck_objects()))))
    checks.append(("install() does not import bottleneck, and serves it once imported", _runs_clean(_LATE_IMPORT_PROGRAM)))
    checks.append(("a bottleneck of another layout is left alone, and install() still works",
                   _runs_clean(_FOREIGN_LAYOUT_PROGRAM)))
    checks.append(("report() renders when not installed", "not installed" in str(hotkernels.report())))
    with hotkernels.enabled():
        checks.append(("enabled() installs for its block", hotkernels.report().installed))
    checks.append(("enabled() uninstalls what it installed", not hotkernels.report().installed))

    failures = tuple(Failure(statement, "does not hold", "true", "false") for statement, holds in checks if not holds)
    return Summary(INSTALLATION, len(checks), 0, failures)


def run(operations: dict, inexact_operations: dict, keep: Callable[[Case], bool]) -> Iterator[Summary]:
    """Run every operation, the exact kernels first.

    Args:
        operations: Operation name -> function yielding its cases, run under
            a plain install().
        inexact_operations: The same for cases run under install(inexact=True).
        keep: Says which cases to run.

    Yields:
        One summary per operation that had at least one case to run.
    """
    for inexact, table in ((False, operations), (True, inexact_operations)):
        for operation, make_cases in table.items():
            summary = run_operation(operation, make_cases(), inexact, keep)
            if summary.cases:
                yield summary


@functools.cache
def _operation_cases(operation: str, inexact: bool) -> tuple[Case, ...]:
    """Build an operation's cases once per process.

    A worker process runs several shares of one operation; the cases are built
    on its first and kept for the rest. They cannot be sent from the parent,
    since they hold functions, so each process builds its own. Building is
    deterministic: every process gets the same cases in the same order.

    Args:
        operation: Operation name, a key of OPERATIONS or INEXACT_OPERATIONS.
        inexact: Which of the two tables to read.

    Returns:
        The operation's cases, in order.
    """
    table = INEXACT_OPERATIONS if inexact else OPERATIONS
    return tuple(table[operation]())


def _every_case(case: Case) -> bool:
    """Keep every case: the filter of a share whose cases were filtered already."""
    return True


def run_shard(operation: str, inexact: bool, shard: int, shards: int, keep: CaseFilter) -> Summary:
    """Run one share of an operation's cases, in a worker process.

    The cases the filter keeps are split into `shards` contiguous blocks of
    nearly equal size, so the failures of the blocks, put together in block
    order, are in the order a run in one process gives.

    Args:
        operation: Operation name.
        inexact: Whether its cases run under install(inexact=True).
        shard: Which block to run, from 0 to `shards` - 1.
        shards: How many blocks the cases are split into.
        keep: Says which cases to run.

    Returns:
        The summary of this block alone. hotkernels is left uninstalled.
    """
    kept = [case for case in _operation_cases(operation, inexact) if keep(case)]
    start, stop = len(kept) * shard // shards, len(kept) * (shard + 1) // shards
    return run_operation(operation, kept[start:stop], inexact, _every_case)


def _merge(parts: list[Summary]) -> Summary:
    """Put the summaries of an operation's blocks together.

    Args:
        parts: One summary per block, in block order. They share a name.

    Returns:
        The operation's summary, as a run in one process would give it.
    """
    failures = tuple(failure for part in parts for failure in part.failures)
    return Summary(parts[0].operation, sum(p.cases for p in parts), sum(p.taken for p in parts), failures)


def run_parallel(operations: dict, inexact_operations: dict, keep: CaseFilter, jobs: int) -> Iterator[Summary]:
    """Run every operation in worker processes, each split into shares.

    Each share installs and uninstalls hotkernels in its own process, as
    `run_operation` does in one, so no share can see another's state.

    Args:
        operations: Operation name -> function yielding its cases, run under a
            plain install(). Only the names are used: the workers build the
            cases from the same tables.
        inexact_operations: The same for cases run under install(inexact=True).
        keep: Says which cases to run.
        jobs: Number of worker processes, at least 1.

    Yields:
        One summary per operation that had at least one case to run, in the
        order the operations finish.

    Raises:
        RuntimeError: A worker process died, as a crash in a kernel makes it.
    """
    work = [(operation, inexact) for inexact, table in ((False, operations), (True, inexact_operations))
            for operation in table]
    parts: dict[tuple[str, bool], list[Summary | None]] = {key: [None] * SHARDS_PER_OPERATION for key in work}
    context = multiprocessing.get_context(START_METHOD)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        # Every operation's first share before any operation's second, so the
        # operations finish in a spread and the progress lines come steadily.
        futures = {pool.submit(run_shard, operation, inexact, shard, SHARDS_PER_OPERATION, keep): (operation, inexact, shard)
                   for shard in range(SHARDS_PER_OPERATION) for operation, inexact in work}
        try:
            for future in as_completed(futures):
                operation, inexact, shard = futures[future]
                done = parts[(operation, inexact)]
                done[shard] = future.result()
                if all(part is not None for part in done):
                    summary = _merge(done)
                    if summary.cases:
                        yield summary
        except BrokenProcessPool as error:
            raise RuntimeError("a worker process died, as a crash in a kernel or in NumPy makes it; "
                               "rerun with --jobs 1 to see which case") from error
