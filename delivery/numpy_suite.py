"""Runs NumPy's own test suite on plain NumPy and again with hotkernels
installed, and compares the outcome of every test.

NumPy's tests know nothing of hotkernels, which makes them the broadest
evidence available that installing it breaks nothing: a test that passes on
plain NumPy and fails with hotkernels is a behaviour some user relies on.
Only the exact kernels are installed; a kernel that is not bit-identical is
expected to fail tests that compare bits.

A test whose outcome differs is run again, both ways, before it is reported,
so that a test that fails at random does not fail the comparison. See
docs/numpy-tests-ci-design.md.

The two runs of the whole suite are made at the same time, in two processes,
which halves the wall time on a machine with a core to spare. The runs of the
differing tests are made one after the other: a test that fails only when
another run shares the machine then agrees when run again, and is not reported.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pytest_plugin import INSTALL_VALUE, INSTALL_VARIABLE, OUTCOMES_VARIABLE

# The pytest plugin that installs hotkernels and records outcomes.
PLUGIN = "delivery.pytest_plugin"
# NumPy's own selection for an ordinary test run: everything not marked slow.
MARKER = "not slow"
# Part of NumPy whose tests are left out: they compile Fortran, need tools
# this project does not, and touch nothing hotkernels changes.
SKIPPED_PACKAGE = "f2py"
# Outcome recorded for a test that one of the two runs did not report.
MISSING = "not run"
# Most differing tests that are run again. More differences than this mean a
# crash or a broken hook, not chance, and running them again could take as
# long as the suite; they are reported as they stand.
RERUN_LIMIT = 100
# Prefix that makes pytest read arguments from a file, one per line.
ARGUMENTS_FILE_PREFIX = "@"


@dataclass(frozen=True)
class Difference:
    """One test whose outcome differs between the two runs."""

    # pytest's id of the test.
    test: str
    # Outcome on plain NumPy.
    plain: str
    # Outcome with hotkernels installed.
    installed: str


@dataclass(frozen=True)
class Comparison:
    """The result of the two runs."""

    # Number of tests either run reported.
    tests: int
    # Number that passed on plain NumPy.
    passed_plain: int
    # Number that passed with hotkernels installed.
    passed_installed: int
    # Every test whose outcome differs: in the first runs and again when run
    # a second time, or in the first runs alone when `rerun` is False.
    differences: tuple[Difference, ...]
    # Tests whose outcomes differed in the first runs and agreed when run
    # again, with their outcomes from the first runs.
    unrepeated: tuple[Difference, ...]
    # Whether the differing tests were run again. False when there were none,
    # or more than RERUN_LIMIT.
    rerun: bool


def _start_suite(install: bool, outcomes: Path, select: str | None, tests: Path | None = None) -> subprocess.Popen:
    """Start one run of NumPy's tests in a subprocess that records every outcome.

    Args:
        install: Whether hotkernels is installed for the run.
        outcomes: File the outcomes are written to.
        select: A pytest `-k` expression to run part of the suite, or None.
            Ignored when `tests` is given.
        tests: A file of pytest node ids, one per line, to run only those
            tests; None for the whole suite. Node ids are relative to the
            current directory, as the first run recorded them, so both runs
            must start in the same directory.

    Returns:
        The running subprocess. Its exit status says whether tests failed,
        which is an outcome to compare and not an error of this tool.
    """
    environment = dict(os.environ, **{OUTCOMES_VARIABLE: str(outcomes)})
    environment[INSTALL_VARIABLE] = INSTALL_VALUE if install else ""
    command = [sys.executable, "-m", "pytest", "-p", PLUGIN, "-q", "--no-header", "-p", "no:cacheprovider"]
    if tests is not None:
        # The node ids go through a file so that their number cannot exceed
        # the length of a command line.
        command.append(f"{ARGUMENTS_FILE_PREFIX}{tests}")
    else:
        command += ["--pyargs", "numpy", "-m", MARKER, "--continue-on-collection-errors",
                    f"--ignore={Path(np.__file__).parent / SKIPPED_PACKAGE}"]
        if select:
            command += ["-k", select]
    # pytest's output is not needed: the plugin writes the outcomes.
    return subprocess.Popen(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_outcomes(path: Path) -> dict[str, str]:
    """Read the outcomes a run recorded.

    Args:
        path: The file the run was told to write.

    Returns:
        Test id -> outcome. Empty if the run wrote nothing, as when pytest
        itself crashed; every test of that run then counts as not run.
    """
    return json.loads(path.read_text()) if path.exists() else {}


def _differences(plain: dict[str, str], installed: dict[str, str],
                 tests: list[str]) -> tuple[Difference, ...]:
    """Find the tests whose outcomes differ between two runs.

    Args:
        plain: Outcomes on plain NumPy.
        installed: Outcomes with hotkernels installed.
        tests: The tests to compare, in the order to report them.

    Returns:
        One difference per test whose outcomes differ. A test missing from a
        run has the outcome MISSING there.
    """
    return tuple(
        Difference(test, plain.get(test, MISSING), installed.get(test, MISSING))
        for test in tests if plain.get(test, MISSING) != installed.get(test, MISSING)
    )


def _run_both(directory: Path, name: str, select: str | None, tests: Path | None = None,
              together: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    """Run the suite, or some of its tests, plain and with hotkernels.

    Args:
        directory: Where the outcome files are written.
        name: Distinguishes the files of this pair of runs from another pair's.
        select: A pytest `-k` expression, or None. Ignored with `tests`.
        tests: A file of node ids to run only those tests, or None.
        together: Whether the two runs are made at the same time rather than
            one after the other.

    Returns:
        The outcomes on plain NumPy and with hotkernels installed.
    """
    plain_file, installed_file = directory / f"{name}-plain.json", directory / f"{name}-installed.json"
    if together:
        runs = [_start_suite(False, plain_file, select, tests), _start_suite(True, installed_file, select, tests)]
        for run in runs:
            run.wait()
    else:
        _start_suite(False, plain_file, select, tests).wait()
        _start_suite(True, installed_file, select, tests).wait()
    return _read_outcomes(plain_file), _read_outcomes(installed_file)
def _passed(outcomes: dict[str, str]) -> int:
    """Count the tests whose call phase passed."""
    return sum(1 for outcome in outcomes.values() if outcome == "call:passed")


def compare(select: str | None = None) -> Comparison:
    """Run the suite twice, run any differing test twice again, and compare.

    Args:
        select: A pytest `-k` expression to run part of the suite, or None
            for all of it.

    Returns:
        The comparison. A test missing from one run counts as a difference.
    """
    with tempfile.TemporaryDirectory(prefix="delivery-numpy-") as name:
        directory = Path(name)
        plain, installed = _run_both(directory, "suite", select, together=True)
        tests = sorted(set(plain) | set(installed))
        first = _differences(plain, installed, tests)
        if not first or len(first) > RERUN_LIMIT:
            return Comparison(len(tests), _passed(plain), _passed(installed), first, (), False)
        ids = directory / "differing.txt"
        ids.write_text("".join(f"{difference.test}\n" for difference in first))
        plain_again, installed_again = _run_both(directory, "again", None, ids)
    # A test that neither second run reported was not run again, and keeps
    # its difference from the first runs rather than passing unseen.
    not_rerun = {d.test for d in first if d.test not in plain_again and d.test not in installed_again}
    again = {d.test: d for d in _differences(plain_again, installed_again, [d.test for d in first])}
    differences = tuple(again.get(d.test, d) for d in first if d.test in again or d.test in not_rerun)
    reported = {difference.test for difference in differences}
    return Comparison(
        len(tests), _passed(plain), _passed(installed), differences,
        tuple(difference for difference in first if difference.test not in reported), True,
    )
