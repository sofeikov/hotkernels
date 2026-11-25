"""Workload runs kept over time, and the series read back from them.

`delivery workloads --history DIR` writes one file per run under
`DIR/history/<machine>/`. The files live on the orphan branch `perf-history`,
are never edited, and are read back through git with no checkout: `delivery
history` prints them, and the documentation site draws them. The design is in
docs/workloads-history-design.md.

This module needs only the standard library and git, so that the site can be
built on a machine where hotkernels has no kernels.
"""
from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import median

log = logging.getLogger(__name__)

# The repository: the tools run from a checkout of it.
ROOT = Path(__file__).resolve().parent.parent
# Directory of the history files, at the top of the data branch and of the
# directory given to --history.
HISTORY_DIR = "history"
# Where the history is read from: the data branch as fetched from GitHub. A
# reader runs `git fetch origin perf-history` first.
DATA_BRANCH = "perf-history"
DEFAULT_REF = f"origin/{DATA_BRANCH}"
# Load average above which a run is marked as taken on a busy machine.
BUSY_LOAD = 1.0
# The sizes, smallest first, as delivery/workloads/ names them. Repeated
# here so that reading the history imports no workload.
SIZE_ORDER = ("S", "M", "L")
# The size the headline is the geometric mean over.
HEADLINE_SIZE = "M"
# What started a run: the nightly workflow, a person, or a one-off run of a past commit.
NIGHTLY, MANUAL, BACKFILL = "nightly", "manual", "backfill"
SOURCES = (NIGHTLY, MANUAL, BACKFILL)
# A run's file name: the UTC minute it started, then the short commit.
# Colons are left out, as some file systems do not allow them.
_STAMP_FORMAT = "%Y-%m-%dT%H%MZ"
# Characters of the commit hash in a file name.
SHORT_COMMIT = 7
# Where the release notes of the pull requests since the last release are kept,
# and the name of one: <pull request number>.<type>.md (AGENTS.md, "Release notes").
UPCOMING_NOTES = "release-notes/upcoming"
_NOTE_NAME = re.compile(r"(\d+)\.([a-z]+)\.md$")
# Separates the fields of a `git log` line asked for below; a tab never occurs in a hash or a date.
_FIELD_SEP = "\t"


class HistoryError(Exception):
    """The history cannot be written or read: a dirty tree, or a missing ref."""


def _git(*args: str, cwd: Path = ROOT, stdin: str | None = None) -> str:
    """Run git and return its standard output.

    Args:
        *args: The git command and its arguments.
        cwd: The repository to run it in.
        stdin: Text fed to the command, if any.

    Returns:
        The output, with the trailing newline left in.

    Raises:
        subprocess.CalledProcessError: If git fails.
    """
    return subprocess.run(["git", *args], cwd=cwd, input=stdin, capture_output=True, text=True, check=True).stdout


def is_dirty(cwd: Path = ROOT) -> bool:
    """Return whether tracked files differ from the commit checked out. Untracked files do not count."""
    return bool(_git("status", "--porcelain", "--untracked-files=no", cwd=cwd).strip())


def head_commit(cwd: Path = ROOT) -> tuple[str, str]:
    """Return the full hash of the commit checked out and its committer date, in ISO 8601."""
    commit, date = _git("log", "-1", f"--format=%H{_FIELD_SEP}%cI", cwd=cwd).strip().split(_FIELD_SEP)
    return commit, date


def now_utc() -> datetime:
    """Return the current time in UTC, to the second."""
    return datetime.now(UTC).replace(microsecond=0)


def run_meta(meta: dict, sizes: list[str], source: str, started: datetime, hotkernels_version: str) -> dict:
    """Add what a history file records beyond the report of `--record`.

    Args:
        meta: The run's description from `workload_timing.describe_run()`.
        sizes: The size names the run covered.
        source: One of SOURCES.
        started: When the run started, in UTC.
        hotkernels_version: The version of the package that was timed.

    Returns:
        A new dictionary: `meta` with commit, commit_date, recorded_at, dirty,
        sizes, hotkernels and source added. load_average_end is added once the
        run has finished.
    """
    commit, commit_date = head_commit()
    return {**meta, "commit": commit, "commit_date": commit_date, "recorded_at": started.isoformat(),
            "dirty": is_dirty(), "sizes": list(sizes), "hotkernels": hotkernels_version, "source": source}


def run_path(directory: Path, machine: str, started: datetime, commit: str) -> Path:
    """Return where a run's file goes.

    Args:
        directory: The directory given to --history; the files go under its history/.
        machine: The machine's slug, as --record names its files.
        started: When the run started, in UTC.
        commit: The full hash of the commit timed.

    Returns:
        DIR/history/<machine>/<UTC date and time>-<short commit>.json.
    """
    return directory / HISTORY_DIR / machine / f"{started.strftime(_STAMP_FORMAT)}-{commit[:SHORT_COMMIT]}.json"


@dataclass(frozen=True)
class Run:
    """One recorded run of the workloads, as read from the data branch."""

    # Path of the file on the data branch: history/<machine>/<name>.json.
    path: str
    # The machine's slug: the directory the file is in.
    machine: str
    # The run's description: the fields of `--record` and those run_meta adds.
    meta: dict
    # One row per workload and size, as `workload_timing.as_rows` writes them.
    rows: tuple[dict, ...]

    @property
    def commit(self) -> str:
        """Return the full hash of the commit timed."""
        return self.meta["commit"]

    @property
    def commit_date(self) -> datetime:
        """Return the date of the commit timed."""
        return datetime.fromisoformat(self.meta["commit_date"])

    @property
    def recorded_at(self) -> datetime:
        """Return when the run started."""
        return datetime.fromisoformat(self.meta["recorded_at"])

    @property
    def placed_at(self) -> datetime:
        """Return where the run sits on the time axis.

        A run sits at the time it started. A backfill run times a past commit,
        so it sits at that commit's date, among the runs it would have had.
        """
        return self.commit_date if self.meta["source"] == BACKFILL else self.recorded_at

    @property
    def busy(self) -> bool:
        """Return whether the machine was in use at the start or at the end of the run."""
        loads = (self.meta.get("load_average"), self.meta.get("load_average_end"))
        return any(load is not None and load > BUSY_LOAD for load in loads)

    def row(self, workload: str, size: str) -> dict | None:
        """Return the row of one workload at one size, or None if the run did not time it.

        Args:
            workload: The workload's name.
            size: The size's name.

        Returns:
            The row, only if its results were identical both ways and it was timed.
        """
        for row in self.rows:
            if row["workload"] == workload and row["size"] == size and row["identical"] and row["hotkernels_ns"]:
                return row
        return None

    def versions(self, libraries: tuple[str, ...]) -> tuple:
        """Return what has to stay the same for this run to be compared with another.

        Args:
            libraries: Distribution names of the libraries that matter beyond NumPy.

        Returns:
            Python's version, NumPy's, and those of the given libraries.
        """
        installed = self.meta.get("libraries", {})
        return (self.meta["python"], self.meta["numpy"], *(installed.get(name) for name in sorted(libraries)))


def ref_exists(ref: str = DEFAULT_REF, cwd: Path = ROOT) -> bool:
    """Return whether the ref the history is read from exists in the repository."""
    try:
        _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=cwd)
    except subprocess.CalledProcessError:
        return False
    return True


def _read_blobs(ref: str, paths: list[str], cwd: Path) -> list[str]:
    """Read several files of a ref through one `git cat-file --batch`.

    Args:
        ref: The branch or commit.
        paths: Paths of files in it.
        cwd: The repository.

    Returns:
        The text of each file, in the order of `paths`.
    """
    # The batch output is, per file, a header "<hash> blob <size>", the
    # content, and a newline. Sizes are in bytes, so the output is read as bytes.
    out = subprocess.run(["git", "cat-file", "--batch"], cwd=cwd, check=True, capture_output=True,
                         input="".join(f"{ref}:{p}\n" for p in paths).encode()).stdout
    texts, at = [], 0
    for path in paths:
        end = out.index(b"\n", at)
        header = out[at:end].decode().split()
        if len(header) != 3 or header[1] != "blob":
            raise HistoryError(f"{ref}:{path}: git cat-file said {' '.join(header)}")
        size = int(header[2])
        texts.append(out[end + 1:end + 1 + size].decode())
        at = end + 1 + size + 1
    return texts


def load_runs(ref: str = DEFAULT_REF, cwd: Path = ROOT) -> list[Run]:
    """Read every run on the data branch.

    Args:
        ref: The branch to read, usually DEFAULT_REF.
        cwd: The repository.

    Returns:
        The runs, ordered by where they sit on the time axis (Run.placed_at)
        and, at the same place, by when they ran.

    Raises:
        HistoryError: If the ref does not exist.
    """
    if not ref_exists(ref, cwd):
        raise HistoryError(f"{ref} not found. Fetch it first: git fetch origin {DATA_BRANCH}")
    paths = [p for p in _git("ls-tree", "-r", "--name-only", ref, "--", HISTORY_DIR, cwd=cwd).splitlines()
             if p.endswith(".json")]
    runs = []
    for path, text in zip(paths, _read_blobs(ref, paths, cwd)):
        data = json.loads(text)
        runs.append(Run(path, Path(path).parent.name, data["meta"], tuple(data["rows"])))
    return sorted(runs, key=lambda r: (r.placed_at, r.recorded_at))


@dataclass(frozen=True)
class Group:
    """Consecutive runs of one series that timed the same commit under the same versions.

    Their spread is how far the figure moves by chance; the chart draws it as a band.
    """

    # The commit timed.
    commit: str
    # The date of the commit timed.
    commit_date: datetime
    # (run, speedup) for each run in the group, in the order they ran.
    points: tuple[tuple[Run, float], ...]
    # Whether the versions differ from the group before, so the two must not be joined.
    break_before: bool

    @property
    def value(self) -> float:
        """Return the median speedup of the group's runs."""
        return median(speedup for _, speedup in self.points)


@dataclass(frozen=True)
class Series:
    """The runs of one workload at one size on one machine, or the headline when workload is None."""

    # The machine's slug.
    machine: str
    # The workload's name, or None for the headline.
    workload: str | None
    # The size's name.
    size: str
    # The groups, in time order.
    groups: tuple[Group, ...] = field(default=())


def _group(machine: str, workload: str | None, size: str, values: list[tuple[Run, float, tuple]]) -> Series:
    """Split one series' runs into groups of one commit under one set of versions.

    Args:
        machine: The machine's slug.
        workload: The workload's name, or None for the headline.
        size: The size's name.
        values: (run, speedup, what must not change) per run, in time order.

    Returns:
        The series. Consecutive runs fall into the same group while the commit
        and the versions stay the same.
    """
    groups: list[Group] = []
    previous = None
    for run, speedup, key in values:
        last = groups[-1] if groups else None
        if last and last.commit == run.commit and previous == key:
            groups[-1] = Group(last.commit, last.commit_date, (*last.points, (run, speedup)), last.break_before)
        else:
            groups.append(Group(run.commit, run.commit_date, ((run, speedup),), previous is not None and previous != key))
        previous = key
    return Series(machine, workload, size, tuple(groups))


def workload_series(runs: list[Run], machine: str, workload: str, size: str) -> Series:
    """Build the series of one workload at one size on one machine.

    Args:
        runs: Every run, in time order (as load_runs returns them).
        machine: The machine's slug.
        workload: The workload's name.
        size: The size's name.

    Returns:
        The series; empty if no run timed it. A change of Python, NumPy or a
        library the workload needs is a break.
    """
    values = []
    for run in runs:
        row = run.row(workload, size) if run.machine == machine else None
        if row:
            values.append((run, row["speedup"], run.versions(tuple(row.get("requires", ())))))
    return _group(machine, workload, size, values)


def headline_series(runs: list[Run], machine: str, size: str = HEADLINE_SIZE) -> Series:
    """Build the headline of one machine: the geometric mean of the workloads' speedups per run.

    Args:
        runs: Every run, in time order (as load_runs returns them).
        machine: The machine's slug.
        size: The size the mean is over.

    Returns:
        The series. A change of the set of workloads, or of a version any of
        them depends on, is a break.
    """
    values = []
    for run in runs:
        if run.machine != machine:
            continue
        rows = [run.row(r["workload"], size) for r in run.rows if r["size"] == size]
        rows = [r for r in rows if r]
        if not rows:
            continue
        mean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
        libraries = tuple(sorted({name for r in rows for name in r.get("requires", ())}))
        key = (tuple(sorted(r["workload"] for r in rows)), run.versions(libraries))
        values.append((run, mean, key))
    return _group(machine, None, size, values)


def machines(runs: list[Run]) -> list[str]:
    """Return the slugs of the machines with runs, in the order they first appear."""
    return list(dict.fromkeys(run.machine for run in runs))


@dataclass(frozen=True)
class Change:
    """A merge to main that changed what a user gets: it added a release note."""

    # The merge commit's full hash.
    commit: str
    # Its committer date: where the marker sits on the time axis.
    date: datetime
    # The pull request's number, from the note's file name.
    pull_request: int
    # The note's type: kernel, speed, fix, compat or change.
    kind: str
    # The note, in Markdown.
    note: str


def _has_commit(commit: str, cwd: Path) -> bool:
    """Return whether the repository has the given commit."""
    try:
        _git("cat-file", "-e", f"{commit}^{{commit}}", cwd=cwd)
    except subprocess.CalledProcessError:
        return False
    return True


def _changes_between(earlier: str, later: str, cwd: Path) -> list[Change]:
    """Find the merges that added a release note between two commits.

    Args:
        earlier: The commit of the earlier run.
        later: The commit of the later run.
        cwd: The repository, with the full history of main.

    Returns:
        One change per note added, oldest first.
    """
    # With --first-parent, a merge's files are those it brought into main
    # (git >= 2.31 implies --diff-merges=first-parent), and the branch's own
    # commits are not listed. The format line comes first, then the files it added.
    text = _git("log", "--first-parent", "--reverse", "--diff-filter=A", "--name-only",
                f"--format=@{_FIELD_SEP}%H{_FIELD_SEP}%cI", f"{earlier}..{later}", "--", UPCOMING_NOTES, cwd=cwd)
    changes, commit, date = [], None, None
    for line in text.splitlines():
        if line.startswith("@" + _FIELD_SEP):
            _, commit, iso = line.split(_FIELD_SEP)
            date = datetime.fromisoformat(iso)
            continue
        match = _NOTE_NAME.search(line)
        if match and commit:
            note = _git("show", f"{commit}:{line}", cwd=cwd).strip()
            changes.append(Change(commit, date, int(match.group(1)), match.group(2), note))
    return changes


def changes(runs: list[Run], cwd: Path = ROOT) -> list[Change]:
    """Find the merges that changed the package between consecutive commits that were timed.

    Built from git each time, not stored with the runs, so that the markers
    can be improved without running anything again.

    Args:
        runs: Every run, in any order.
        cwd: The repository, with the full history of main.

    Returns:
        The changes, oldest first. A commit the repository does not have is
        skipped with a warning, as are the changes on either side of it.
    """
    # The merges between two commits are found in commit order, whatever order the runs were taken in.
    commits = list(dict.fromkeys(run.commit for run in sorted(runs, key=lambda r: r.commit_date)))
    present = []
    for commit in commits:
        if _has_commit(commit, cwd):
            present.append(commit)
        else:
            log.warning("history: commit %s is not in this repository; fetch main with its full history", commit[:SHORT_COMMIT])
    found: dict[tuple[str, int], Change] = {}
    for earlier, later in pairwise(present):
        for change in _changes_between(earlier, later, cwd):
            found.setdefault((change.commit, change.pull_request), change)
    return sorted(found.values(), key=lambda c: c.date)
