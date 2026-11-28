"""The judge: checks hotkernels assembly kernels against NumPy and times them.

Run with `uv run judge <command>`.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from . import bench as B
from . import rounds as R
from . import selftest as S
from .build import ARCH, BUILD, CC, ROOT, BuildError, build_asm
from .check import KERNEL, NAIVE, Built, CheckResult, build_all, check, format_result
from .placement import CLOSE_BYTES
from .render import render
from .spec import Kernel

log = logging.getLogger(__name__)

KERNELS = ROOT / "kernels"
# The kernels of the lazy layer's prototype: one per fused expression (docs/lazy-fusion-design.md). They are checked
# and timed as any other kernel, and are not built into the package, which takes its kernels from KERNELS alone.
PROTOTYPE_KERNELS = ROOT / "prototypes" / "lazy" / "kernels"
# Where a kernel is looked for by name, in this order.
KERNEL_ROOTS = (KERNELS, PROTOTYPE_KERNELS)
FIXTURES = Path(__file__).resolve().parent / "fixtures"
# Recorded results, committed to the repository.
RESULTS = ROOT / "results"
# The latest report of every run. Not committed.
RUNS = ROOT / ".judge" / "runs"
LEADERBOARD = ROOT / "LEADERBOARD.md"

# Exit codes. Speed never fails a run; only correctness and misuse do.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2

# Contender names used in timing reports.
NUMPY = "numpy"
NEW, OLD = "new", "old"
_BENCH_LABELS = {KERNEL: "kernel", NAIVE: "naive C", NUMPY: "NumPy*", B.PLACED_APART: "apart*"}
# Printed under the table that times the close placement.
CLOSE_NOTE = ("'apart' is the same code with its output clear of the input, as in the table above. "
              "A speedup below 1 is what the close placement costs.")
_COMPARE_LABELS = {NEW: "new", OLD: "old"}

NUMPY_NOTE = (
    "NumPy is timed from Python and includes its per-call overhead; the kernel is timed "
    "without a binding. At small sizes the NumPy column flatters the kernel. Naive C is the "
    "honest baseline."
)

# Where Linux describes the processor, and the field that holds its name.
_CPUINFO = Path("/proc/cpuinfo")
_CPUINFO_MODEL = "model name"

# Branch the reference files must match during a kernel run.
_REFERENCE_BRANCH = "main"
# Anything that is not a lowercase letter or digit becomes a dash in a file name.
_SLUG_SEPARATORS = re.compile(r"[^a-z0-9]+")
_JSON_INDENT = 2

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

KernelArg = typer.Argument(..., help="Kernel name: a directory under kernels/ or prototypes/lazy/kernels/.")
UnlockedOpt = typer.Option(
    False, "--unlocked",
    help="Allow modified reference files. For people creating a kernel or changing the "
         "judge; never for agent runs.",
)
QuickOpt = typer.Option(False, "--quick", help="Two sizes and a third of the samples, as a smoke test.")


@app.callback()
def configure(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress while timing."),
) -> None:
    """Set up logging. Reports go to standard output as plain lines."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command in the repository and return the completed process."""
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def resolve(name: str) -> tuple[Path, bool]:
    """Find a kernel by name. Returns (directory, whether it is a judge fixture)."""
    for root in KERNEL_ROOTS:
        if (root / name / "spec.py").exists():
            return root / name, False
    if (FIXTURES / name / "spec.py").exists():
        return FIXTURES / name, True
    log.error("unknown kernel %r: no %s", name,
              " or ".join(f"{root.relative_to(ROOT).as_posix()}/{name}/spec.py" for root in KERNEL_ROOTS))
    raise typer.Exit(EXIT_REFUSED)


def reference_changes(kernel_dir: Path) -> list[str]:
    """List the locked files that differ from the reference branch.

    The locked files are the judge itself, its dependencies, and the kernel's
    spec and naive reference: everything a kernel is measured against.

    Args:
        kernel_dir: The kernel's directory, under one of KERNEL_ROOTS.

    Returns:
        The paths that differ or are not tracked, relative to the repository's root.
    """
    if _git("rev-parse", "--verify", "--quiet", _REFERENCE_BRANCH).returncode != 0:
        return []
    relative = kernel_dir.relative_to(ROOT).as_posix()
    locked = ["judge", "pyproject.toml", "uv.lock", f"{relative}/spec.py", f"{relative}/naive.c"]
    changed = _git("diff", "--name-only", _REFERENCE_BRANCH, "--", *locked).stdout.split()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *locked).stdout.split()
    return sorted(set(changed + untracked))


def guard(name: str, kernel_dir: Path, is_fixture: bool, unlocked: bool) -> bool:
    """Refuse to run when the reference files were modified.

    Args:
        name: The kernel's name, for the message.
        kernel_dir: Its directory.
        is_fixture: Whether it is a judge fixture.
        unlocked: Whether the caller asked to run anyway.

    Returns:
        True when the reference files are verified unchanged, and False
        when they differ but `unlocked` allowed the run anyway. Fixtures are
        part of the judge and are not guarded.
    """
    if is_fixture:
        return True
    changed = reference_changes(kernel_dir)
    if not changed:
        return True
    log.info(render("guard.txt.j2", unlocked=unlocked, changed=changed, name=name, arch=ARCH))
    if unlocked:
        return False
    raise typer.Exit(EXIT_REFUSED)


def machine() -> str:
    """Return a human-readable name for this machine's processor."""
    if platform.system() == "Darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip()
        if brand:
            return brand
    if _CPUINFO.exists():
        for line in _CPUINFO.read_text().splitlines():
            field, _, value = line.partition(":")
            if field.strip() == _CPUINFO_MODEL and value.strip():
                return value.strip()
    return platform.processor() or platform.machine()


def environment(locked: bool) -> dict:
    """Describe where and how a report was produced, for the JSON record."""
    clang = subprocess.run([CC, "--version"], capture_output=True, text=True).stdout
    return {
        "arch": ARCH,
        "machine": machine(),
        "os": f"{platform.system()} {platform.release()}",
        "clang": clang.splitlines()[0] if clang else "unknown",
        "numpy": np.__version__,
        "python": platform.python_version(),
        "git_head": _git("rev-parse", "--short", "HEAD").stdout.strip(),
        "git_dirty": bool(_git("status", "--porcelain").stdout.strip()),
        "reference_locked": locked,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _log_progress(dist: str, n: int) -> None:
    """Log that one size has been timed. Shown only with --verbose."""
    log.debug("  timed %s n=%s", dist, f"{n:,}")


def _sweep(quick: bool) -> tuple[list[int], int]:
    """Return the (sizes, samples) to time: the full sweep, or the quick one."""
    return (B.QUICK_SIZES, B.QUICK_SAMPLES) if quick else (B.SIZES, B.SAMPLES)


def run_bench(built: Built, quick: bool, locked: bool) -> dict:
    """Time the kernel against naive C and NumPy and return the full report."""
    k = built.k
    harness = B.Harness(k, built.harness)
    naive_fn, _naive_lib = B.function_address(built.naive, k.name)
    contenders: dict[str, int | None] = {NAIVE: naive_fn, NUMPY: None}
    # A kernel without assembly is its naive C, measured against NumPy alone.
    subject, baselines = NAIVE, [NUMPY]
    if built.kernel:
        kernel_fn, _kernel_lib = B.function_address(built.kernel, k.name)
        contenders = {KERNEL: kernel_fn, **contenders}
        subject, baselines = KERNEL, [NAIVE, NUMPY]
    sizes, samples = _sweep(quick)

    canary_before = harness.canary()
    data = B.measure(k, harness, contenders, sizes, samples, _log_progress)
    close = _close_output(k, harness, subject, contenders, sizes, samples)
    canary_after = harness.canary()

    report = B.analyse(data, subject, baselines)
    report.update(kernel=k.name, samples=samples, subject=subject, baselines=baselines,
                  warnings=B.noise_warnings(canary_before, canary_after, harness.core),
                  environment=environment(locked))
    if close:
        report["close_output"] = close
    return report


def _close_output(k: Kernel, harness: B.Harness, subject: str, contenders: dict[str, int | None],
                  sizes: list[int], samples: int) -> dict | None:
    """Time the subject with its output buffer just behind its input.

    Args:
        k: The kernel's spec.
        harness: The compiled timing loop.
        subject: Name of the contender the report is about.
        contenders: Contender name -> native function address, or None for the
            NumPy oracle, which is left out.
        sizes: Element counts to time.
        samples: Timing samples per contender per size.

    Returns:
        A report in the shape `B.analyse` returns, of the subject in the close
        placement against itself placed apart and against the other native
        contenders in the close placement. It also names its `baselines` and
        the distance in `gap_bytes`. None for a kernel without an output
        buffer that can be placed.
    """
    if not B.has_placed_output(k):
        return None
    native = {name: fn for name, fn in contenders.items() if fn is not None}
    baselines = [B.PLACED_APART] + [name for name in native if name != subject]
    data = B.measure_close(k, harness, subject, native, sizes, samples, _log_progress)
    return {**B.analyse(data, subject, baselines), "baselines": baselines, "gap_bytes": CLOSE_BYTES}


def log_bench(report: dict) -> None:
    """Log the timing table for a kernel against naive C and NumPy."""
    env = report["environment"]
    title = (f"{report['kernel']} ({env['arch']}, {env['machine']}), NumPy {env['numpy']}, "
             f"up to {report['samples']} samples per point")
    log.info(render("timing.txt.j2", title=title, warnings=report["warnings"], report=report,
                    kernel=report["subject"], baselines=report["baselines"], labels=_BENCH_LABELS,
                    notes=[f"* {NUMPY_NOTE}"], verdict=None).rstrip())
    close = report.get("close_output")
    if close:
        title = (f"{report['kernel']}: output buffer {close['gap_bytes']} bytes behind its input, "
                 "counted within a memory page")
        log.info("\n" + render("timing.txt.j2", title=title, warnings=[], report=close,
                               kernel=report["subject"], baselines=close["baselines"],
                               labels=_BENCH_LABELS, notes=[f"* {CLOSE_NOTE}"], verdict=None).rstrip())


def save_run(report: dict, record: bool) -> None:
    """Write the report as JSON under .judge/runs, and under results/ when recording."""
    env = report["environment"]
    text = json.dumps(report, indent=_JSON_INDENT) + "\n"
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{report['kernel']}-{env['arch']}.json"
    path.write_text(text)
    log.info("\nfull report: %s", path.relative_to(ROOT))
    if record:
        slug = _SLUG_SEPARATORS.sub("-", env["machine"].lower()).strip("-")
        target = RESULTS / report["kernel"] / f"{env['arch']}-{slug}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        log.info("recorded:    %s", target.relative_to(ROOT))


def _checked(kernel: str, kernel_dir: Path, case: Optional[str] = None) -> CheckResult:
    """Run the correctness check, log its outcome, and exit if it failed."""
    result = check(kernel_dir, case=case)
    log.info(format_result(result, kernel))
    if not result.ok:
        raise typer.Exit(EXIT_FAILED)
    return result


@app.command("check")
def check_command(
    kernel: str = KernelArg,
    case: Optional[str] = typer.Option(None, help="Run a single case, as printed by a failure."),
    unlocked: bool = UnlockedOpt,
) -> None:
    """Lint and correctness only. Fast enough to run after every edit."""
    kernel_dir, is_fixture = resolve(kernel)
    guard(kernel, kernel_dir, is_fixture, unlocked)
    _checked(kernel, kernel_dir, case)


@app.command()
def bench(kernel: str = KernelArg, quick: bool = QuickOpt, unlocked: bool = UnlockedOpt) -> None:
    """Timing report only, without the correctness check."""
    kernel_dir, is_fixture = resolve(kernel)
    locked = guard(kernel, kernel_dir, is_fixture, unlocked)
    built, failure, k = build_all(kernel_dir)
    if failure:
        log.info(format_result(CheckResult(k.name, False, failure), kernel))
        raise typer.Exit(EXIT_FAILED)
    report = run_bench(built, quick, locked)
    log_bench(report)
    save_run(report, record=False)


@app.command()
def run(
    kernel: str = KernelArg,
    quick: bool = QuickOpt,
    record: bool = typer.Option(False, "--record", help="Also save the report under results/."),
    unlocked: bool = UnlockedOpt,
) -> None:
    """Correctness, then the timing report against naive C and NumPy."""
    kernel_dir, is_fixture = resolve(kernel)
    locked = guard(kernel, kernel_dir, is_fixture, unlocked)
    result = _checked(kernel, kernel_dir)
    log.info("")
    report = run_bench(result.built, quick, locked)
    report["correctness"] = {"cases": result.cases, "fresh_seed": result.fresh_seed}
    log_bench(report)
    save_run(report, record)


def _build_old_version(built: Built, revision: str) -> Path:
    """Build the kernel as it was at an earlier git revision.

    Exits with a message when the kernel did not exist then, or no longer
    builds or passes lint.
    """
    k = built.k
    relative = built.asm.relative_to(ROOT).as_posix()
    shown = _git("show", f"{revision}:{relative}")
    if shown.returncode != 0:
        log.error("\ncannot compare: %s does not exist at %s", relative, revision)
        raise typer.Exit(EXIT_REFUSED)
    source = BUILD / k.name / "old.s"
    source.write_text(shown.stdout)
    try:
        dylib, issues = build_asm(k, source, OLD)
    except BuildError as e:
        log.error("\ncannot compare: the version at %s does not build: %s", revision, e)
        raise typer.Exit(EXIT_REFUSED)
    if issues:
        log.error("\ncannot compare: the version at %s fails lint: %s", revision, issues[0])
        raise typer.Exit(EXIT_REFUSED)
    return dylib


def _compare_verdict(overall: dict) -> str:
    """Describe the overall difference as one of three outcomes, with its interval."""
    if not overall["significant"]:
        outcome = "no significant difference"
    elif overall["speedup"] > 1:
        outcome = "significantly faster"
    else:
        outcome = "significantly slower"
    return (f"{outcome} ({overall['speedup']:.3f}x, "
            f"95% interval {overall['lo']:.3f} – {overall['hi']:.3f})")


@app.command()
def compare(
    kernel: str = KernelArg,
    against: str = typer.Option(..., help="Git revision to compare with, for example HEAD."),
    quick: bool = QuickOpt,
    unlocked: bool = UnlockedOpt,
) -> None:
    """Is the working copy of a kernel faster than an earlier version of it?

    Both versions are timed interleaved, which detects much smaller
    differences than comparing against numbers recorded earlier. The timing
    is taken in several rounds, each in a process of its own, because where a
    build happens to be loaded changes its speed a little and a difference
    has to be larger than that to count: see COMPARE_ROUNDS in judge/bench.py.
    """
    kernel_dir, is_fixture = resolve(kernel)
    guard(kernel, kernel_dir, is_fixture, unlocked)
    built = _checked(kernel, kernel_dir).built
    k = built.k
    if not built.kernel:
        log.error("\ncannot compare: %s has no %s assembly", kernel, ARCH)
        raise typer.Exit(EXIT_REFUSED)
    old_dylib = _build_old_version(built, against)

    harness = B.Harness(k, built.harness)
    sizes, samples = _sweep(quick)
    per_round = max(1, samples // B.COMPARE_ROUNDS)
    canary_before = harness.canary()
    try:
        measured = R.measure(k, kernel_dir, built.harness, {NEW: built.kernel, OLD: old_dylib},
                             sizes, per_round, B.COMPARE_ROUNDS, harness.core)
    except RuntimeError as e:
        log.error("\ncannot compare: %s", e)
        raise typer.Exit(EXIT_FAILED)
    canary_after = harness.canary()

    report = B.analyse_rounds(measured, NEW, [OLD])
    log.info("\n" + render(
        "timing.txt.j2",
        title=(f"{k.name} ({ARCH}): working copy vs {against}, "
               f"{B.COMPARE_ROUNDS} rounds of up to {per_round} samples per point"),
        warnings=B.noise_warnings(canary_before, canary_after, harness.core),
        report=report, kernel=NEW, baselines=[OLD], labels=_COMPARE_LABELS, notes=[],
        verdict=_compare_verdict(report["overall"][f"vs_{OLD}"]),
    ).rstrip())


@app.command()
def selftest(
    no_speed: bool = typer.Option(False, "--no-speed", help="Skip the timing fixture, for CI."),
    stability: bool = typer.Option(
        False, "--stability", help="Repeat the timing sweep and report how much it varies."),
) -> None:
    """Prove the judge catches kernels that are wrong in known ways."""
    passed = S.stability() if stability else S.selftest(speed=not no_speed)
    if not passed:
        raise typer.Exit(EXIT_FAILED)


@app.command()
def leaderboard() -> None:
    """Regenerate LEADERBOARD.md from the recorded results."""
    results = [json.loads(path.read_text()) for path in sorted(RESULTS.glob("*/*.json"))]
    LEADERBOARD.write_text(render("leaderboard.md.j2", results=results, numpy_note=NUMPY_NOTE))
    log.info("wrote %s (%d results)", LEADERBOARD.name, len(results))


if __name__ == "__main__":
    app()
