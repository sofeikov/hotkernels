"""Tests of the order in which the lazy layer's two timing tools time the ways a program is run.

`fusion_time` and `limit` time a program as it is on NumPy, eagerly with the
kernels installed, lazily node by node, and fused. A process that has run a
program before runs it faster, so the tools time the ways in turn: a round
runs every way once, and the quickest run of each way counts. Every run of
one way before the first of the next is still there to be asked for.

The tests of the order put a recorder where a tool times a run, so they
time nothing; two more run each tool on a small program in both orders.
"""
from __future__ import annotations

import contextlib
import inspect
from typing import Any, Callable, Iterator

import numpy
import pytest

from prototypes.lazy import fuse
from prototypes.lazy.programs import corpus, fusion_time, limit

# How many rounds the recorded timings are asked for, and how many the tools really run a program for.
ROUNDS, REAL_RUNS = 3, 2
# The program the tools are really run on, at its small size.
PROGRAM = "regression"
# What the recorder reports for a run, in nanoseconds: each way is quickest in a different round.
TAKEN = {"numpy": [30, 10, 20], "installed": [21, 31, 11], "lazy": [12, 22, 32], "fused": [33, 23, 13]}
# A round of the tools: (the way, whether the kernels are installed for it).
INSTALLED_ROUND = [("numpy", False), ("installed", True), ("lazy", True), ("fused", True)]
PLAIN_ROUND = [("lazy", False), ("fused", False)]


class Recorder:
    """Stands where a tool times a run: notes each call and reports a time from `TAKEN`."""

    def __init__(self) -> None:
        """Start with nothing timed and nothing installed."""
        # Each timing asked for: (the way, whether the kernels were installed, how many runs were asked for).
        self.calls: list[tuple[str, bool, int]] = []
        # Whether a block of `enabled` is open.
        self.installed = False

    @contextlib.contextmanager
    def enabled(self) -> Iterator[None]:
        """Stand in for `hotkernels.enabled`: a block inside which the kernels count as installed."""
        self.installed = True
        try:
            yield
        finally:
            self.installed = False

    def take(self, way: str, runs: int) -> int:
        """Note one timing and return what it took.

        Args:
            way: The way that was timed: a key of `TAKEN`, where an eager run with the kernels installed is
                still "numpy".
            runs: How many runs were asked for.

        Returns:
            Nanoseconds: the next of the way's times in `TAKEN`, by how often the way has been timed so far.
        """
        name = "installed" if way == "numpy" and self.installed else way
        seen = sum(1 for other, installed, _ in self.calls if (other, installed) == (way, self.installed))
        self.calls.append((way, self.installed, runs))
        return TAKEN[name][seen % ROUNDS]

    def order(self) -> list[tuple[str, bool]]:
        """Return the timings in the order they were asked for, an installed eager run named "installed"."""
        return [("installed" if way == "numpy" and installed else way, installed) for way, installed, _ in self.calls]


@pytest.fixture
def in_fusion_time(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Put a recorder where `fusion_time` times a program one way."""
    recorder = Recorder()

    def quickest(_program: Any, _inputs: tuple, way: str, runs: int) -> fusion_time.Timing:
        """Report a time in place of `fusion_time.quickest`."""
        return fusion_time.Timing(recorder.take(way, runs))

    monkeypatch.setattr(fusion_time, "quickest", quickest)
    return recorder


@pytest.fixture
def in_limit(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Put a recorder where `limit` times a program one way."""
    recorder = Recorder()

    def timed(_program: Any, _inputs: tuple, np: Any, runs: int) -> int:
        """Report a time in place of `limit._timed`."""
        return recorder.take("numpy" if np is numpy else "lazy", runs)

    def timed_fused(_program: Any, _inputs: tuple, runs: int) -> int:
        """Report a time in place of `limit._timed_fused`."""
        return recorder.take("fused", runs)

    monkeypatch.setattr(limit, "_timed", timed)
    monkeypatch.setattr(limit, "_timed_fused", timed_fused)
    return recorder


def quickest_of(way: str) -> int:
    """Return the least of the times the recorder reports for a way."""
    return min(TAKEN[way])


# ---- The order.

def test_fusion_time_runs_every_way_once_a_round(in_fusion_time: Recorder):
    """In turn: NumPy, installed, lazy and fused once each, round after round, and the quickest of each is kept."""
    timings, plain = fusion_time.alternately(None, (), ROUNDS, in_fusion_time.enabled, plain=False)
    assert in_fusion_time.order() == INSTALLED_ROUND * ROUNDS
    assert all(runs == 1 for _, _, runs in in_fusion_time.calls)
    assert {way: timing.total_ns for way, timing in timings.items()} == {way: quickest_of(way) for way in TAKEN}
    assert plain == {}


def test_fusion_time_times_the_plain_lazy_ways_in_the_same_round(in_fusion_time: Recorder):
    """With `--plain` a round ends with the lazy and the fused run with nothing installed."""
    timings, plain = fusion_time.alternately(None, (), ROUNDS, in_fusion_time.enabled, plain=True)
    assert in_fusion_time.order() == (INSTALLED_ROUND + PLAIN_ROUND) * ROUNDS
    assert set(timings) == set(TAKEN) and set(plain) == {"lazy", "fused"}


def test_limit_runs_every_way_once_a_round(in_limit: Recorder):
    """In turn: the four ways once each, round after round, and the quickest of each is kept."""
    times = limit.in_turn(None, (), ROUNDS, in_limit.enabled)
    assert in_limit.order() == INSTALLED_ROUND * ROUNDS
    assert all(runs == 1 for _, _, runs in in_limit.calls)
    assert times == {way: quickest_of(way) for way in TAKEN}


def test_limit_one_after_another_finishes_a_way_before_the_next(in_limit: Recorder):
    """The order that can still be asked for: each way timed once, for all its runs, NumPy's first."""
    times = limit.one_after_another(None, (), ROUNDS, in_limit.enabled)
    assert in_limit.order() == INSTALLED_ROUND
    assert all(runs == ROUNDS for _, _, runs in in_limit.calls)
    assert set(times) == set(TAKEN)


@pytest.mark.parametrize("tool", [fusion_time, limit])
def test_in_turn_is_the_default(tool: Any):
    """Both tools time in turn unless asked otherwise, from the command line and when `measure` is called."""
    assert inspect.signature(tool.measure).parameters["alternate"].default is True
    option = inspect.signature(tool.main).parameters["alternate"].default
    assert option.default is True
    assert option.param_decls == ("--alternate/--one-after-another",)


# ---- The tools themselves, on a small program.

@pytest.fixture
def program() -> corpus.Program:
    """Return the small program the tools are run on: skipped where fusion or the package is not built."""
    pytest.importorskip("hotkernels")
    if not fuse.enable(install=False):
        pytest.skip("the fused kernels are not built")
    fuse.disable()
    return corpus.by_name([PROGRAM])[0]


def enabled() -> Callable[[], Any]:
    """Return `hotkernels.enabled`, imported when a test needs it."""
    import hotkernels
    return hotkernels.enabled


@pytest.mark.parametrize("alternate", [True, False])
def test_fusion_time_measures_a_program_in_either_order(program: corpus.Program, alternate: bool):
    """Either order gives a time for each of the four ways and the two plain ones, and the table says which order."""
    row = fusion_time.measure(program, corpus.SMALL, REAL_RUNS, enabled(), True, alternate)
    assert set(row.timings) == {fusion_time.NUMPY, fusion_time.INSTALLED, fusion_time.LAZY, fusion_time.FUSED}
    assert set(row.plain) == {fusion_time.LAZY, fusion_time.FUSED}
    assert all(timing.total_ns > 0 for timing in list(row.timings.values()) + list(row.plain.values()))
    heading = fusion_time.render([row], corpus.SMALL, REAL_RUNS, alternate).splitlines()[0]
    assert ("the ways timed in turn" in heading) == alternate
    assert ("one way after another" in heading) != alternate


@pytest.mark.parametrize("alternate", [True, False])
def test_limit_measures_a_program_in_either_order(program: corpus.Program, alternate: bool):
    """Either order gives the four times the floor is set against, and the table says which order."""
    measured = limit.measure(program, corpus.SMALL, REAL_RUNS, enabled(), alternate)
    assert min(measured.numpy_ns, measured.eager_ns, measured.lazy_ns, measured.fused_ns) > 0
    heading = limit.render([measured], corpus.SMALL, REAL_RUNS, alternate).splitlines()[0]
    assert ("the ways timed in turn" in heading) == alternate
    assert ("one way after another" in heading) != alternate
