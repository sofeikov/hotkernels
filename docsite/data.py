"""Numbers and logs the site shows, read from the repository at build time.

Nothing here is typed in by hand. Every figure on the site that comes from a
measurement is parsed out of the file that records it: the kernels' attempts
logs, the results tables in the README and in docs/kernel-findings.md, and the
recorded Python timings. A file whose layout has changed fails the build
instead of producing a page with holes in it.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from delivery import history as H

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels"
README = ROOT / "README.md"
FINDINGS = ROOT / "docs" / "kernel-findings.md"
PYTHON_TIMINGS = ROOT / "delivery" / "results" / "apple-m1-pro.json"
PYTHON_TIMINGS_X86_64 = ROOT / "delivery" / "results" / "intel-r-core-tm-i5-14600kf.json"
# Where the workloads' history is read from, in order: the data branch as
# fetched from GitHub, then a local branch of the same name, for a build on a
# machine that records runs and has not pushed them.
HISTORY_REFS = (H.DEFAULT_REF, H.DATA_BRANCH)

# The architectures kernels are written for, by the name of their directory
# under kernels/<name>/. Each holds that architecture's kernel.s and attempts.md.
ARM64, X86_64 = "arm64", "x86_64"
# The heading in kernel-findings.md whose table holds the x86-64 results. The
# ARM64 table is the first in the file; this one is found by its heading.
_X86_64_HEADING = "### x86-64"
# The size of the one single-size column the x86-64 table and the x86-64
# chapter quote.
X86_64_SIZE = 1048576

# "## 7. Unroll the main loop": the heading that opens one attempt in an attempts log.
_ATTEMPT_HEADING = re.compile(r"^## (\d+)\. (.+)$")
# "- **Idea:** text": one field of an attempt.
_FIELD = re.compile(r"^- \*\*([^:*]+):\*\* ?(.*)$")
# "1.21x [1.19 – 1.23]": a speedup with its 95% interval, as the judge prints it.
_RATIO = re.compile(r"(\d+(?:\.\d+)?)x \[(\d+(?:\.\d+)?) – (\d+(?:\.\d+)?)\]")
# "23.4x": a speedup on its own.
_SPEEDUP = re.compile(r"(\d+(?:\.\d+)?)x")

# A heading appended to a log while parsing, so its last attempt is closed like the others.
_END_OF_LOG = "## end"
# Fields every attempt carries, by the name the site uses for them.
_ATTEMPT_FIELDS = {"Idea": "idea", "check": "check", "compare": "compare", "Kept": "kept", "Notes": "notes"}

# What `judge compare` said about an attempt.
FASTER, SLOWER, SAME, NOT_COMPARED = "faster", "slower", "same", "none"
# Phrases that mark a compare line as "no significant difference". They are
# tested first, because "not significantly faster" contains "significantly faster".
_SAME_PHRASES = ("no significant", "not significant")
# How a compare line starts when the attempt was never timed against its
# predecessor. Any ratio on such a line is against something else, such as naive C.
_NOT_COMPARED_PREFIXES = ("n/a", "not run", "not applicable")
# The first attempt of a log has no predecessor to be compared with.
_FIRST_ATTEMPT = 1

# Header cells that identify the two results tables.
_FINDINGS_HEADER = "| Kernel | Input | vs naive C |"
_README_HEADER = "| NumPy call | Elements | M1 Pro: vs NumPy |"
# Marks a difference the judge did not find statistically significant.
_NOT_SIGNIFICANT = "n.s."
# Sizes of the four single-size columns of the findings table, in order.
FINDINGS_SIZES = [16, 4096, 1048576, 16777216]


def _verdict(number: int, compare: str) -> str:
    """Classify what `judge compare` said about an attempt.

    Args:
        number: The attempt's number in its log.
        compare: The text of the attempt's compare field.

    Returns:
        FASTER, SLOWER, SAME, or NOT_COMPARED when the attempt was never timed
        against its predecessor.
    """
    text = compare.lower().lstrip()
    if number == _FIRST_ATTEMPT or text.startswith(_NOT_COMPARED_PREFIXES):
        return NOT_COMPARED
    if any(phrase in text for phrase in _SAME_PHRASES):
        return SAME
    if "significantly faster" in text:
        return FASTER
    if "significantly slower" in text:
        return SLOWER
    return NOT_COMPARED


def _attempt(number: int, title: str, fields: dict[str, str], inline: Callable[[str], str]) -> dict:
    """Build the record of one attempt.

    Args:
        number: The attempt's number in its log.
        title: The attempt's heading, as Markdown.
        fields: The attempt's fields by site name, each as Markdown.
        inline: Renders a line of Markdown to HTML.

    Returns:
        The attempt, with its text as HTML and its compare line broken down
        into a verdict and, where one was printed, a speedup with its interval.
    """
    compare = fields.get("compare", "")
    verdict = _verdict(number, compare)
    # A ratio is only the attempt's own when compare ran against its predecessor.
    ratio = _RATIO.search(compare) if verdict != NOT_COMPARED else None
    return {
        "n": number,
        "title": inline(title),
        "idea": inline(fields.get("idea", "")),
        "check": inline(fields.get("check", "")),
        "compare": inline(compare),
        "notes": inline(fields.get("notes", "")),
        "kept": fields.get("kept", "").lower().startswith("yes"),
        "verdict": verdict,
        "speedup": float(ratio.group(1)) if ratio else None,
        "lo": float(ratio.group(2)) if ratio else None,
        "hi": float(ratio.group(3)) if ratio else None,
    }


def _parse_attempts(text: str, inline: Callable[[str], str]) -> list[dict]:
    """Read every numbered attempt out of one attempts log.

    Args:
        text: The contents of an attempts.md.
        inline: Renders a line of Markdown to HTML.

    Returns:
        The attempts in the order they were made. Sections of the log that are
        not numbered attempts (final results, notes) are skipped.
    """
    attempts: list[dict] = []
    number, title, fields, last = None, "", {}, None
    for line in text.splitlines() + [_END_OF_LOG]:
        heading = _ATTEMPT_HEADING.match(line)
        if line.startswith("## "):
            if number is not None:
                attempts.append(_attempt(number, title, fields, inline))
            number, title, fields, last = (int(heading.group(1)), heading.group(2), {}, None) if heading else (None, "", {}, None)
            continue
        if number is None:
            continue
        field = _FIELD.match(line)
        if field:
            last = _ATTEMPT_FIELDS.get(field.group(1).strip())
            if last:
                fields[last] = field.group(2)
        elif last and line.strip() and not line.startswith("- "):
            # A field that wraps onto a second line.
            fields[last] += " " + line.strip()
    return attempts


def attempts(inline: Callable[[str], str], arch: str = ARM64) -> dict:
    """Collect the attempts log of every kernel on one architecture.

    Args:
        inline: Renders a line of Markdown to HTML.
        arch: ARM64 or X86_64: which `kernels/<name>/<arch>/attempts.md` to read.

    Returns:
        `kernels`, one entry per kernel with its attempts and whether it has
        assembly on that architecture, and the totals over all of them:
        `total`, `kept`, `reverted`, and how many attempts got each verdict
        from `judge compare`.
    """
    kernels = []
    for log_path in sorted(KERNELS.glob(f"*/{arch}/attempts.md")):
        tried = _parse_attempts(log_path.read_text(), inline)
        if not tried:
            raise ValueError(f"{log_path}: no numbered attempts found")
        kernels.append({
            # kernels/<name>/<arch>/attempts.md
            "name": log_path.parent.parent.name,
            "assembly": (log_path.parent / "kernel.s").exists(),
            "attempts": tried,
        })
    everything = [a for k in kernels for a in k["attempts"]]
    kept = sum(a["kept"] for a in everything)
    return {
        "kernels": kernels,
        "total": len(everything),
        "kept": kept,
        "reverted": len(everything) - kept,
        "verdicts": {v: sum(a["verdict"] == v for a in everything) for v in (FASTER, SLOWER, SAME, NOT_COMPARED)},
    }


def _table(path: Path, header: str) -> list[list[str]]:
    """Read the rows of one Markdown table.

    Args:
        path: The Markdown file.
        header: How the table's header line starts.

    Returns:
        The cells of every body row, stripped of surrounding spaces.
    """
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(header)]
    if not starts:
        raise ValueError(f"{path}: no table starting with {header!r}")
    rows = []
    # Skip the header and the |---|---| line under it.
    for line in lines[starts[0] + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _speedup_cell(cell: str) -> dict | None:
    """Read one cell of the findings table.

    Args:
        cell: A cell such as "23.4x [23.39 – 23.41]", "1.00x n.s." or "".

    Returns:
        The speedup, its interval where the cell has one and whether it is
        significant; None for an empty cell or one without a number. `text`
        and `interval` hold the figures as the source prints them, with the
        precision the judge chose.
    """
    value = _SPEEDUP.search(cell)
    if not value:
        return None
    ratio = _RATIO.search(cell)
    return {
        "x": float(value.group(1)),
        "lo": float(ratio.group(2)) if ratio else None,
        "hi": float(ratio.group(3)) if ratio else None,
        "text": value.group(0),
        "interval": f"[{ratio.group(2)} – {ratio.group(3)}]" if ratio else None,
        "significant": _NOT_SIGNIFICANT not in cell,
    }


def findings() -> list[dict]:
    """The judge's results per kernel and input, from docs/kernel-findings.md.

    Returns:
        One entry per row of the summary table: the kernel, the input it was
        timed on, its speedup over naive C overall and at four sizes, and over
        NumPy overall and at 1,048,576 elements.
    """
    rows = []
    for cells in _table(FINDINGS, _FINDINGS_HEADER):
        kernel, kind, overall_c, *at_sizes, overall_numpy, numpy_1m = cells
        rows.append({
            "kernel": kernel.strip("`"),
            "input": kind,
            "vs_c": _speedup_cell(overall_c),
            "vs_c_at": [_speedup_cell(c) for c in at_sizes],
            "vs_numpy": _speedup_cell(overall_numpy),
            "vs_numpy_1m": _speedup_cell(numpy_1m),
            "c_kernel": "C kernel" in overall_c,
        })
    return rows


def _range_cell(cell: str) -> dict | None:
    """Read one cell of the README's results table.

    Args:
        cell: A cell holding one speedup ("29.0x"), two ("8.82x, 8.86x" or
            "17.3x – 41.7x"), or words.

    Returns:
        The lowest and highest speedup in the cell, or None if it has none.
    """
    values = [float(v) for v in _SPEEDUP.findall(cell)]
    return {"lo": min(values), "hi": max(values)} if values else None


def headline() -> list[dict]:
    """The headline results per NumPy call, from the README.

    Returns:
        One entry per row of the README's kernel table that has a figure for
        the Apple M1 Pro: the call, its element type, and its speedup over
        NumPy and over `clang -O3` at 1,048,576 elements on that machine, each
        as a range.
    """
    rows = []
    # The table also has a column for the Apple M4 Max and one for the Intel
    # Core i5-14600KF; the site's chart is of the Apple M1 Pro alone.
    for call, elements, vs_numpy, vs_c, _m4_max, _x86_64 in _table(README, _README_HEADER):
        if _range_cell(vs_numpy) is None:
            # A kernel that ships as its C on ARM64 and was timed on x86-64 only.
            continue
        rows.append({
            "call": call.replace("`", ""),
            "elements": elements,
            "vs_numpy": _range_cell(vs_numpy),
            "vs_c": _range_cell(vs_c),
            "vs_c_text": vs_c.replace("`", ""),
        })
    return rows


def python_timings(path: Path = PYTHON_TIMINGS) -> dict:
    """The recorded timings of Python statements, with and without hotkernels.

    Args:
        path: The delivery tool's recorded result for one machine.

    Returns:
        Its contents: `meta` describes the machine and `rows` holds one
        statement at one size each.
    """
    return json.loads(path.read_text())


def findings_x86_64() -> list[dict]:
    """The x86-64 results table of docs/kernel-findings.md.

    Returns:
        One entry per row, with the cells as written: `kernel` (without its
        backticks, and with any footnote mark kept in `mark`), `input`,
        `vs_c`, `vs_numpy` and `vs_numpy_1m`. A kernel timed on two kinds of
        input has both figures in one cell, separated by a slash.
    """
    lines = FINDINGS.read_text().splitlines()
    headings = [i for i, line in enumerate(lines) if line == _X86_64_HEADING]
    if not headings:
        raise ValueError(f"{FINDINGS}: no heading {_X86_64_HEADING!r}")
    starts = [i for i in range(headings[0], len(lines)) if lines[i].startswith(_FINDINGS_HEADER)]
    if not starts:
        raise ValueError(f"{FINDINGS}: no results table under {_X86_64_HEADING!r}")
    rows = []
    # Skip the header and the |---|---| line under it.
    for line in lines[starts[0] + 2:]:
        if not line.startswith("|"):
            break
        kernel, kind, vs_c, vs_numpy, vs_numpy_1m = (cell.strip() for cell in line.strip().strip("|").split("|"))
        name, _, mark = kernel.partition("` ")
        rows.append({"kernel": name.strip("`"), "mark": mark, "input": kind, "vs_c": vs_c,
                     "vs_numpy": vs_numpy, "vs_numpy_1m": vs_numpy_1m})
    if not rows:
        raise ValueError(f"{FINDINGS}: the x86-64 results table is empty")
    return rows


def python_at(timings: dict, n: int) -> list[dict]:
    """The statements a kernel took at one size, from a recorded timing run.

    Args:
        timings: What `python_timings` returned.
        n: The element count to keep.

    Returns:
        One entry per statement timed at `n` elements and served by a kernel:
        `statement`, `data`, and `speedup`, NumPy's time over hotkernels'.
    """
    return [{"statement": row["statement"], "data": row["data"], "speedup": row["numpy_ns"] / row["hotkernels_ns"]}
            for row in timings["rows"] if row["kernel"] and row["n"] == n]


def _history_run(index: int, run: H.Run) -> dict:
    """Describe one recorded run for the history figure.

    Args:
        index: The run's position in commit order, by which the series refer to it.
        run: The run.

    Returns:
        What the night's table shows, with the run's versions and load.
    """
    meta = run.meta
    return {
        "id": index, "machine": run.machine, "commit": run.commit[:H.SHORT_COMMIT],
        "commit_date": meta["commit_date"], "recorded_at": meta["recorded_at"], "source": meta["source"],
        "numpy": meta["numpy"], "python": meta["python"], "hotkernels": meta["hotkernels"],
        "load_average": meta.get("load_average"), "load_average_end": meta.get("load_average_end"), "busy": run.busy,
        "rows": [{"workload": r["workload"], "size": r["size"], "identical": r["identical"],
                  "numpy_ns": r["numpy_ns"], "hotkernels_ns": r["hotkernels_ns"], "speedup": r["speedup"],
                  "calls": sum(c["calls"] for c in r["counts"]),
                  "left": sum(c["left_to_numpy"] for c in r["counts"])} for r in run.rows],
    }


def _history_series(series: H.Series, ids: dict[str, int]) -> dict:
    """Describe one series for the history figure.

    Args:
        series: The series.
        ids: Each run's id, by the path of its file.

    Returns:
        The machine, workload (None for the headline) and size, and per group
        of runs of one commit: its date, the median speedup, whether it follows
        a break, and (run id, speedup) for each run.
    """
    return {
        "machine": series.machine, "workload": series.workload, "size": series.size,
        "groups": [{"commit": g.commit[:H.SHORT_COMMIT], "date": g.commit_date.isoformat(), "value": g.value,
                    "break": g.break_before, "runs": [[ids[run.path], speedup] for run, speedup in g.points]}
                   for g in series.groups],
    }


def history(inline: Callable[[str], str]) -> dict:
    """Read the workloads' recorded runs and the merges between them, for the history page.

    Args:
        inline: Renders one line of Markdown to HTML, for the release notes.

    Returns:
        The ref read (None if no history is available, and everything else
        empty), the machines, the runs, the workloads, the series (a headline
        per machine and one per workload and size), and the changes.
    """
    ref = next((r for r in HISTORY_REFS if H.ref_exists(r)), None)
    if ref is None:
        log.warning("no workloads history: fetch it with `git fetch origin %s`", H.DATA_BRANCH)
        runs = []
    else:
        runs = H.load_runs(ref)
    ids = {run.path: index for index, run in enumerate(runs)}
    machines = H.machines(runs)
    titles = {r["workload"]: (r["title"], r["ours"]) for run in runs for r in run.rows}
    sizes = [size for size in H.SIZE_ORDER if any(r["size"] == size for run in runs for r in run.rows)]
    series = [H.headline_series(runs, machine) for machine in machines]
    series += [H.workload_series(runs, machine, name, size) for machine in machines for name in titles for size in sizes]
    return {
        "ref": ref, "busy_load": H.BUSY_LOAD, "headline_size": H.HEADLINE_SIZE,
        "machines": [{"slug": m, "cpu": next(r.meta["cpu"] for r in runs if r.machine == m)} for m in machines],
        "workloads": [{"name": name, "title": title, "ours": ours} for name, (title, ours) in titles.items()],
        "runs": [_history_run(index, run) for index, run in enumerate(runs)],
        "series": [_history_series(s, ids) for s in series if s.groups],
        "changes": [{"commit": c.commit[:H.SHORT_COMMIT], "date": c.date.isoformat(), "pull_request": c.pull_request,
                     "kind": c.kind, "note": inline(c.note)} for c in H.changes(runs)],
    }
