"""The delivery tool: tests and times hotkernels from Python.

Run with `uv run delivery <command>`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

import hotkernels

from . import app_profile as P
from . import app_timing as A
from . import apps as applications
from . import check as C
from . import demo as D
from . import history as H
from .gradient_step import study as G
from . import numpy_suite
from . import timing as T
from . import workload_timing as W
from . import workloads as WL
from .cases import INEXACT_OPERATIONS, OPERATIONS
from .render import render

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# Exit codes.
EXIT_OK = 0
EXIT_FAILED = 1

# Failures printed per operation unless --all-failures is given.
SHOWN_FAILURES = 5
# Recorded timings, committed to the repository: one JSON file and one report per machine.
RESULTS = Path(__file__).resolve().parent / "results"
# What the file names of a recorded run of `delivery apps` end with.
APPS_SUFFIX = "-apps"
# What the record of each set of `delivery apps` adds to APPS_SUFFIX. The first
# set's record keeps the name it had before there were two.
SET_SUFFIX = {"first": "", "second": "-second"}
# The report that says what each set's figures mean, under docs/.
SET_FINDINGS = {"first": "applications-findings.md", "second": "applications-second-findings.md"}
# Load average above which a timing run is marked as taken on a busy machine.
BUSY_LOAD = H.BUSY_LOAD
# A kernel-taking call counts as slower than NumPy below this speedup. The
# margin is the noise of a best-of-several timing.
SLOWER_THAN = 0.97


@app.callback()
def main() -> None:
    """Configure logging for every command: plain messages on standard error."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The commands install and uninstall hotkernels many times over; its own
    # message for each would bury their output.
    logging.getLogger(hotkernels.__name__).setLevel(logging.WARNING)


def _table_order(operations: dict, inexact: dict) -> dict[str, int]:
    """Give each summary name its place in the report.

    Args:
        operations: The operations run under a plain install(), in order.
        inexact: The operations run under install(inexact=True), in order.

    Returns:
        Summary name -> position: the exact operations first, then the inexact
        ones, each in the order of its table, as a run in one process gives them.
    """
    names = [C.summary_name(name, False) for name in operations] + [C.summary_name(name, True) for name in inexact]
    return {name: position for position, name in enumerate(names)}


@app.command()
def check(
    only: Optional[str] = typer.Option(None, help="Run one operation, for example `clip`."),
    case: Optional[str] = typer.Option(None, help="Run only cases whose label contains this text."),
    all_failures: bool = typer.Option(False, help="Print every failure, not the first few per operation."),
    jobs: int = typer.Option(os.cpu_count() or 1, min=1,
                             help="Worker processes. Default: one per core. 1 runs everything in this process."),
) -> None:
    """Run every hooked NumPy call with and without hotkernels and compare them.

    Args:
        only: Operation to restrict the run to.
        case: Text a case's label must contain.
        all_failures: Whether to print every failure.
        jobs: Number of worker processes; 1 runs the cases in this process.
    """
    operations = {name: make for name, make in OPERATIONS.items() if only in (None, name)}
    inexact = {name: make for name, make in INEXACT_OPERATIONS.items() if only in (None, name)}
    keep = C.CaseFilter(only, case)
    installation = [] if only or case else [C.installation_checks()]
    finished = C.run(operations, inexact, keep) if jobs == 1 else C.run_parallel(operations, inexact, keep, jobs)
    summaries = []
    for summary in finished:
        log.info("%s: %d cases, %d failed", summary.operation, summary.cases, len(summary.failures))
        summaries.append(summary)
    order = _table_order(operations, inexact)
    summaries = installation + sorted(summaries, key=lambda summary: order[summary.operation])
    failures = sum(len(s.failures) for s in summaries)
    shown = max((len(s.failures) for s in summaries), default=0) if all_failures else SHOWN_FAILURES
    log.info("\n%s", render("check.txt.j2", summaries=summaries, shown=shown,
                            total_cases=sum(s.cases for s in summaries),
                            total_taken=sum(s.taken for s in summaries), total_failures=failures).rstrip("\n"))
    raise typer.Exit(EXIT_OK if failures == 0 else EXIT_FAILED)


def _slug(text: str) -> str:
    """Turn a processor name into a file name: lower case, words joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _time_context(meta: dict, measurements: list) -> dict:
    """Build the variables both timing templates take.

    Args:
        meta: What the run was measured on.
        measurements: Every measurement of the run.

    Returns:
        The measurements split into calls a kernel took and calls left to
        NumPy, the kernel-taking calls that came out slower than NumPy, and
        the metadata.
    """
    taken = [m for m in measurements if m.kernel]
    return {
        "meta": meta,
        "taken": taken,
        "left": [m for m in measurements if not m.kernel],
        "slower": [m for m in taken if m.speedup < SLOWER_THAN],
        "busy_load": BUSY_LOAD,
        "rounds": T.ROUNDS,
    }


@app.command()
def time(
    only: Optional[str] = typer.Option(None, help="Time one operation, for example `clip`."),
    record: bool = typer.Option(False, help="Write the results to delivery/results/."),
) -> None:
    """Time NumPy statements from Python, with and without hotkernels.

    Only meaningful on native hardware and an idle machine.

    Args:
        only: Operation to restrict the run to.
        record: Whether to write a JSON file and a Markdown report.
    """
    meta = T.describe_machine()
    entries = [entry for entry in (*T.TAKEN, *T.LEFT) if only in (None, entry.operation)]
    measurements = T.run(entries)
    context = _time_context(meta, measurements)
    log.info("%s", render("time.txt.j2", **context).rstrip("\n"))
    if record:
        RESULTS.mkdir(exist_ok=True)
        stem = _slug(meta["cpu"]) or "unknown-machine"
        (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": T.as_rows(measurements)}, indent=2) + "\n")
        (RESULTS / f"{stem}.md").write_text(render("time.md.j2", **context))
        log.info("recorded in %s", RESULTS / f"{stem}.md")


@app.command()
def demo() -> None:
    """Run examples/workload.py, an unmodified NumPy program, with and without hotkernels."""
    outcome = D.run()
    log.info("%s", render("demo.txt.j2", demo=outcome).rstrip("\n"))
    raise typer.Exit(EXIT_OK if outcome.identical and outcome.speedup > 1 else EXIT_FAILED)


def _workload_cases(only: Optional[str], sizes: tuple[str, ...]) -> tuple[list, list, list]:
    """Choose which workloads run, at which sizes.

    Args:
        only: Name of the one workload to run, or None for all.
        sizes: Names of the sizes to run.

    Returns:
        (workloads chosen, (workload, size) pairs to run, (name, missing
        libraries) for each chosen workload that cannot run here).
    """
    chosen = [w for w in WL.load() if only in (None, w.name)]
    runnable = [w for w in chosen if not w.missing()]
    skipped = [(w.name, w.missing()) for w in chosen if w.missing()]
    cases = [(w, name) for w in runnable for name in sizes if name in w.sizes]
    return runnable, cases, skipped


def _sizes(text: Optional[str]) -> tuple[str, ...]:
    """Read the --size option.

    Args:
        text: Size names separated by commas, such as "M,L", or None.

    Returns:
        The names, in the order of SIZE_NAMES; every size but the largest if None.

    Raises:
        typer.BadParameter: If a name is not a size.
    """
    if text is None:
        return WL.SIZE_NAMES[:-1]
    names = {name.strip() for name in text.split(",")}
    if not names <= set(WL.SIZE_NAMES):
        raise typer.BadParameter(f"sizes are {', '.join(WL.SIZE_NAMES)}, separated by commas", param_hint="--size")
    return tuple(name for name in WL.SIZE_NAMES if name in names)


def _write_history(directory: Path, meta: dict, rows: list[dict], sizes: tuple[str, ...], source: str,
                   started: datetime) -> Path:
    """Write one run's history file.

    Args:
        directory: The directory given to --history.
        meta: The run's description, taken when it started.
        rows: The run's rows, as `workload_timing.as_rows` makes them.
        sizes: The size names the run covered.
        source: What started the run, one of history.SOURCES.
        started: When the run started, in UTC.

    Returns:
        The file written.
    """
    full = H.run_meta(meta, list(sizes), source, started, hotkernels.__version__)
    full["load_average_end"] = T.load_average()
    path = H.run_path(directory, _slug(meta["cpu"]) or "unknown-machine", started, full["commit"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": full, "rows": rows}, indent=2) + "\n")
    return path


@app.command()
def workloads(
    only: Optional[str] = typer.Option(None, help="Run one workload, for example `azimint_naive`."),
    size: Optional[str] = typer.Option(None, help="Sizes to run, separated by commas: S, M, L. Default: S,M."),
    record: bool = typer.Option(False, help="Write the results to delivery/results/."),
    history: Optional[Path] = typer.Option(
        None, help="Also write the run to DIR/history/<machine>/, for the data branch. Refused from a dirty tree."),
    source: str = typer.Option(H.MANUAL, help="What started the run, as the history records it: "
                               + ", ".join(H.SOURCES) + "."),
) -> None:
    """Run code not written for hotkernels with and without it: results, times and counts.

    Needs the libraries of the `workloads` dependency group: `uv run --group
    workloads delivery workloads`. Times are only meaningful on native
    hardware and an idle machine.

    Args:
        only: Workload to restrict the run to.
        size: Sizes to run, separated by commas.
        record: Whether to write a JSON file and a Markdown report.
        history: Directory to write a history file under, or None.
        source: What started the run: nightly, manual or backfill.
    """
    sizes = _sizes(size)
    if source not in H.SOURCES:
        raise typer.BadParameter(f"source must be one of {', '.join(H.SOURCES)}", param_hint="--source")
    # A history file names the commit it timed; with changes in tracked files it would not say what ran.
    if history is not None and H.is_dirty():
        raise typer.BadParameter("tracked files differ from the commit; commit or stash them first",
                                 param_hint="--history")
    started = H.now_utc()
    meta = W.describe_run()
    chosen, cases, skipped = _workload_cases(only, sizes)
    outcomes = W.run(cases)
    failed = [f"{o.workload} ({o.size})" for o in outcomes if not o.identical]
    context = {"meta": meta, "outcomes": outcomes, "skipped": skipped, "failed": failed,
               "workloads": chosen, "busy_load": BUSY_LOAD}
    log.info("%s", render("workloads.txt.j2", **context).rstrip("\n"))
    if record:
        RESULTS.mkdir(exist_ok=True)
        stem = f"{_slug(meta['cpu']) or 'unknown-machine'}-workloads"
        (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": W.as_rows(outcomes)}, indent=2) + "\n")
        (RESULTS / f"{stem}.md").write_text(render("workloads.md.j2", **context))
        log.info("recorded in %s", RESULTS / f"{stem}.md")
    if history is not None:
        log.info("history in %s", _write_history(history, meta, W.as_rows(outcomes), sizes, source, started))
    raise typer.Exit(EXIT_OK if not failed else EXIT_FAILED)


def _history_lines(series: H.Series) -> list[dict]:
    """Turn a series into one line per run for the history report.

    Args:
        series: The series.

    Returns:
        Per run: when it ran, the commit, the versions, the times where the
        series is one workload's, the speedup, and the marks that apply.
    """
    lines = []
    for group in series.groups:
        for index, (run, speedup) in enumerate(group.points):
            row = run.row(series.workload, series.size) if series.workload else None
            marks = [mark for mark, applies in (("break", group.break_before and index == 0), ("busy", run.busy),
                                                (run.meta["source"], run.meta["source"] != H.NIGHTLY)) if applies]
            lines.append({"recorded_at": run.recorded_at.strftime("%Y-%m-%d %H:%M"), "commit": run.commit[:H.SHORT_COMMIT],
                          "numpy": run.meta["numpy"], "numpy_ns": row["numpy_ns"] if row else None,
                          "hotkernels_ns": row["hotkernels_ns"] if row else None, "speedup": speedup, "marks": marks})
    return lines


@app.command(name="history")
def history_report(
    machine: Optional[str] = typer.Option(None, help="Machine, as its history directory is named. Default: this one."),
    workload: Optional[str] = typer.Option(None, help="One workload. Default: the headline and every workload."),
    size: str = typer.Option(H.HEADLINE_SIZE, help="The size: S, M or L."),
    ref: str = typer.Option(H.DEFAULT_REF, help="The branch the runs are read from."),
) -> None:
    """Print the recorded workload runs of one machine, in the order they ran.

    Reads the data branch through git; run `git fetch origin perf-history` first.

    Args:
        machine: The machine's slug, or None for the one this runs on.
        workload: The one workload to show, or None for the headline and all of them.
        size: The size to show.
        ref: The branch the runs are read from.
    """
    try:
        runs = H.load_runs(ref)
    except H.HistoryError as error:
        log.error("%s", error)
        raise typer.Exit(EXIT_FAILED)
    machine = machine or _slug(T.machine())
    names = [workload] if workload else list(dict.fromkeys(
        row["workload"] for run in runs if run.machine == machine for row in run.rows))
    series = [] if workload else [H.headline_series(runs, machine, size)]
    series += [H.workload_series(runs, machine, name, size) for name in names]
    blocks = [{"workload": s.workload, "lines": _history_lines(s)} for s in series]
    log.info("%s", render("history.txt.j2", machine=machine, size=size, ref=ref, blocks=blocks,
                          machines=H.machines(runs), busy_load=BUSY_LOAD).rstrip("\n"))


def _apps_context(meta: dict, items: list[dict], skipped: dict[str, list[str]], passed: bool) -> dict:
    """Build the variables both templates of `delivery apps` take.

    Args:
        meta: What the run was measured on.
        items: One {"timing": AppTiming, "profile": AppProfile or None} per
            application and size, in the order they ran.
        skipped: Application -> the modules it needs that are missing.
        passed: Whether every result was identical.

    Returns:
        The items, and the same items grouped by application: a list of
        {"name", "title", "source", "items"}, each application once, its
        sizes in the order they ran.
    """
    groups: dict[str, dict] = {}
    for item in items:
        timing = item["timing"]
        group = groups.setdefault(timing.name, {"name": timing.name, "title": timing.title, "source": timing.source,
                                                "items": []})
        group["items"].append(item)
    chosen_set = _set_of(meta)
    return {"meta": meta, "items": items, "applications": list(groups.values()), "skipped": skipped,
            "busy_load": BUSY_LOAD, "rounds": A.ROUNDS, "shortest_round_ms": A.SHORTEST_ROUND_MS, "passed": passed,
            "set_name": chosen_set, "default_set": applications.DEFAULT_SET, "findings": SET_FINDINGS[chosen_set]}


def _set_of(meta: dict) -> str:
    """Return the set of applications a run was of.

    Args:
        meta: What the run was measured on. Records written before there were
            two sets do not name one; they are of the first.

    Returns:
        One of `apps.SETS`.
    """
    return meta.get("set", applications.DEFAULT_SET)


def _apps_stem(meta: dict) -> str:
    """Return the file name, without suffix, of a machine's record of `delivery apps`."""
    return (_slug(meta["cpu"]) or "unknown-machine") + APPS_SUFFIX + SET_SUFFIX[_set_of(meta)]


@app.command()
def apps(
    set_name: str = typer.Option(applications.DEFAULT_SET, "--set",
                                 help="The set of applications: `first` or `second`."),
    only: Optional[str] = typer.Option(None, help="Run one application of either set, for example `mlp_train`."),
    size: Optional[str] = typer.Option(None, help="Run one size, `small` or `large`. Both by default."),
    profile: bool = typer.Option(False, help="Also sample where the time goes: `sample` on macOS, `perf` on Linux."),
    record: bool = typer.Option(False, help="Write the results to delivery/results/."),
    allow_busy: bool = typer.Option(False, help="Record even while GitHub Actions runs are unfinished."),
) -> None:
    """Run whole applications with and without hotkernels, and say where their time goes.

    Needs the libraries of the `apps` dependency group: `uv run --group apps
    delivery apps`. Only meaningful on native hardware and an idle machine.

    Args:
        set_name: The set of applications to run, one of `apps.SETS`. Each
            set is recorded apart.
        only: Application to restrict the run to. Its set is the run's,
            whatever `set_name` says.
        size: Size to restrict the run to, one of `apps.SIZE_NAMES`.
        profile: Whether to sample each application's native stacks both ways.
        record: Whether to write a JSON file and a Markdown report.
        allow_busy: Whether to record although `gh run list` shows runs that
            have not finished. This machine hosts the CI runners, and a run
            taken beside them inflates the shares of the profile.
    """
    if size not in (None, *applications.SIZE_NAMES):
        log.error("no size %r: the sizes are %s", size, ", ".join(applications.SIZE_NAMES))
        raise typer.Exit(EXIT_FAILED)
    if set_name not in applications.SETS:
        log.error("no set %r: the sets are %s", set_name, ", ".join(applications.SETS))
        raise typer.Exit(EXIT_FAILED)
    if only is not None and only not in applications.NAMES:
        log.error("no application %r: the applications are %s", only, ", ".join(applications.NAMES))
        raise typer.Exit(EXIT_FAILED)
    if only is not None:
        set_name = next(name for name, members in applications.SETS.items() if only in members)
    unfinished = A.unfinished_runs() if record else None
    if record and unfinished is None:
        log.warning("could not ask `gh run list` whether CI runs are unfinished; recording anyway")
    elif record and unfinished and not allow_busy:
        log.error("%d GitHub Actions runs are unfinished (`gh run list`). They share this machine and would "
                  "inflate the profile's shares. Wait for them, or pass --allow-busy.", unfinished)
        raise typer.Exit(EXIT_FAILED)
    sampler = None
    if profile:
        sampler, reason = P.find_sampler()
        if sampler is None:
            log.info("no profile: %s. Times and counts only.", reason)
    meta = T.describe_machine() | {"set": set_name, "libraries": A.library_versions(),
                                   "unfinished_ci_runs": unfinished,
                                   "sampler": sampler.description if sampler else None}
    if record and meta["load_average"] > BUSY_LOAD:
        log.warning("load average %s: the times will be marked rough", meta["load_average"])
    names = [name for name in applications.SETS[set_name] if only in (None, name)]
    sizes = [name for name in applications.SIZE_NAMES if size in (None, name)]
    skipped = {name: applications.missing(name) for name in names if applications.missing(name)}
    items = []
    for name in names:
        if name in skipped:
            continue
        for chosen in sizes:
            log.info("running %s, %s", name, chosen)
            items.append({"timing": A.measure(name, chosen),
                          "profile": P.profile(name, chosen, sampler) if sampler else None})
    passed = bool(items) and all(item["timing"].identical for item in items)
    context = _apps_context(meta, items, skipped, passed)
    log.info("%s", render("apps.txt.j2", **context).rstrip("\n"))
    if record:
        RESULTS.mkdir(exist_ok=True)
        stem = _apps_stem(meta)
        rows = [{"timing": A.as_row(item["timing"]),
                 "profile": P.as_row(item["profile"]) if item["profile"] else None} for item in items]
        (RESULTS / f"{stem}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")
        (RESULTS / f"{stem}.md").write_text(render("apps.md.j2", **context))
        log.info("recorded in %s", RESULTS / f"{stem}.md")
    raise typer.Exit(EXIT_OK if passed else EXIT_FAILED)


@app.command(name="apps-report")
def apps_report(
    path: Path = typer.Argument(..., help="A JSON record written by `delivery apps --record`."),
) -> None:
    """Write the Markdown report of a recorded run of `delivery apps` again from its JSON.

    Reads records written before the applications had two sizes as well. The
    report goes beside the JSON, under the same name.

    Args:
        path: The JSON record.
    """
    recorded = json.loads(path.read_text())
    items = []
    for row in recorded["rows"]:
        timing = A.from_row(row["timing"])
        items.append({"timing": timing,
                      "profile": P.from_row(row["profile"], timing.size) if row["profile"] else None})
    context = _apps_context(recorded["meta"], items, {}, all(item["timing"].identical for item in items))
    report = path.with_suffix(".md")
    report.write_text(render("apps.md.j2", **context))
    log.info("written %s", report)


@app.command(name="numpy-tests")
def numpy_tests(
    select: Optional[str] = typer.Option(None, help="A pytest -k expression, to run part of the suite."),
) -> None:
    """Run NumPy's own tests with and without hotkernels and compare every outcome.

    About a minute or two. Only the exact kernels are installed. A test whose
    outcome differs is run again both ways, and reported only if it differs again.

    Args:
        select: A pytest `-k` expression restricting the run.
    """
    comparison = numpy_suite.compare(select)
    log.info("%s", render("numpy_tests.txt.j2", comparison=comparison,
                          limit=numpy_suite.RERUN_LIMIT).rstrip("\n"))
    raise typer.Exit(EXIT_OK if not comparison.differences else EXIT_FAILED)


if __name__ == "__main__":
    app()


@app.command("gradient-step")
def gradient_step(
    rounds: int = typer.Option(G.ROUNDS, help="Trainings and runs per way of writing the update."),
) -> None:
    """Measure where the time of the gradient step `w -= lr * g` goes, and what fusing it would save.

    The study of docs/gradient-step-findings.md: mlp_train and min_char_rnn,
    with loops and a caching allocator of its own. macOS on ARM64 only, and
    only meaningful on an idle machine.

    Args:
        rounds: Trainings and application runs per way, alternating.
    """
    meta = T.describe_machine()
    log.info("timing the update inside mlp_train")
    ways, split = G.in_loop(rounds)
    log.info("timing w -= b after each reader")
    readers = G.after_reader()
    log.info("running both applications on two allocators")
    allocators, cache_counts = G.allocator(rounds)
    log.info("timing min_char_rnn's accumulations")
    accumulations = G.accumulations()
    log.info("%s", render("gradient_step.txt.j2", meta=meta, threads_variable=G.THREADS_VARIABLE,
                            threads=G.threads_setting(), measured=G.MEASURED, rounds=rounds, ways=ways,
                            split=split, repeats=G.REPEATS, readers=readers, prefetch_ahead=G.PREFETCH_AHEAD,
                            allocators=allocators, cache_counts=cache_counts,
                            accumulations=accumulations).rstrip("\n"))
