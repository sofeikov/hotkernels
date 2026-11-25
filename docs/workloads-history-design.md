# Workloads over time: nightly runs and a history chart

This is a proposal for review. It extends [workloads-design.md](workloads-design.md). Every night a scheduled workflow runs `uv run delivery workloads` on the maintainer's Mac and keeps the result. A page of the documentation site draws the history: how much faster the workloads run with hotkernels, night by night, with the changes that moved the figure marked on it. It builds on the runner in [pull request 90](https://github.com/sofeikov/hotkernels/pull/90) and needs it merged first.

The model is [doesjitgobrrr](https://doesjitgobrrr.fastapicloud.dev/?goals=5%2C10&changes=1), which tracks CPython's JIT against its interpreter on pyperformance. It runs every night, draws one geometric-mean speedup per machine over time, marks the days a relevant pull request merged, and opens a per-benchmark table for the night a reader clicks.

## Summary

- **Today a recorded run replaces the one before it.** `delivery workloads --record` writes `delivery/results/<machine>-workloads.json`, and the previous run on that machine is gone. Nothing can say whether a workload became faster or slower, or when.
- **A run per commit is too many.** Most commits are kernel attempts, documents or tools, often dozens a day. A run per night is enough to see a change, and finds the merge that caused it to within a day.
- **Proposed: a nightly workflow** on the self-hosted Mac, and only there. If the Mac is offline or busy, that night has no run. A hosted runner never stands in.
- **One file per run, on a data branch.** The workflow writes `history/<machine>/<UTC date and time>-<commit>.json` to an orphan branch, `perf-history`, and never changes a file once written. `main` gets no commits from the workflow.
- **Each run records what can move the figures**, not only the commit: the NumPy and library versions, the load average and the set of workloads. `np.searchsorted` went from 6.40x to 1.48x faster than NumPy between the M1 Pro and the M4 Max with the kernel unchanged ([README.md](../README.md#results)). A change of NumPy version on one machine can do the same.
- **A history page on the site:** the geometric mean of the workloads' speedups per night, the merges that change the package marked on the time axis, and a table per night. Drawn by a widget like the site's other charts, with no library.
- **`uv run delivery history`** prints the same series in a terminal.
- **A night with no new commit still runs.** Runs of one commit on different nights show how far a figure moves by chance, which nothing measures yet.
- **Not proposed:** asv, hosted runners, a run per commit, and the judge and `delivery time`. See [Considered and not proposed](#considered-and-not-proposed).

## What is wrong today

Three tools record timings, and each keeps only the latest run per machine:

| Tool | Writes | A new run |
|---|---|---|
| `judge run --record` | `results/<kernel>/<arch>-<machine>.json` | Replaces it |
| `delivery time --record` | `delivery/results/<machine>.json` and `.md` | Replaces them |
| `delivery workloads --record` | `delivery/results/<machine>-workloads.json` and `.md` | Replaces them |

Git keeps the old versions, so history exists in principle. In practice a recorded file changes only when someone chooses to record, its old versions are spread across unrelated commits, and nothing reads them back.

This proposal covers the workloads. They are what a user gets, and they are few. The judge and `delivery time` can take the same format later; see [Decisions still open](#decisions-still-open).

## Design

### One file per run

`delivery workloads` gains `--history DIR`. With it, the run also writes:

```
DIR/history/<machine>/<UTC date and time>-<commit>.json
```

for example `history/apple-m4-max/2026-11-20T0200Z-6a4fe1c.json`. The machine part is the slug `--record` already uses. `--record` keeps writing the latest report to `delivery/results/` as now; the two options are independent.

A file is never edited after it is written. A run that turns out to be wrong is deleted, in a commit that says why. One file per run, not one file per machine with a line appended per run, so that two writers never conflict: the nightly workflow, and a person who records a run on another machine and pushes it.

### What a run records

The same JSON `--record` writes today (`meta` and `rows`), with these fields added to `meta`:

| Field | Is |
|---|---|
| `commit` | The full commit hash |
| `commit_date` | The commit's date, so that runs can be put in commit order without the repository |
| `recorded_at` | When the run started, in UTC |
| `dirty` | Whether tracked files differed from the commit. Untracked files do not count, unlike the judge's `git_dirty` |
| `sizes` | The sizes run |
| `hotkernels` | The package version |
| `load_average_end` | The one-minute load average when the run finished. `load_average` is taken at the start |
| `source` | `nightly` for the workflow, `manual` otherwise |

`libraries`, `numpy`, `python`, `os` and `load_average` are already there.

`--history` refuses to write from a dirty tree, since the commit would not say what was run.

### The data branch

`perf-history` is an orphan branch: it shares no history with `main` and holds only `history/`. Why not `main`, as `--record` does now:

- A commit a night on `main` from a bot is noise in its log, and every open branch would have to merge it.
- The files are data, not source. Nothing in the package or the tools reads them at build time except the site.

A file is about 7 KB for four workloads at two sizes, so a year of nights is about 2.5 MB.

`delivery history` and the site read the branch through git, with no checkout: `git ls-tree` and `git show` on `origin/perf-history`. A reader runs `git fetch origin perf-history` first; `delivery history` says so if the ref is missing.

### The nightly workflow

A new workflow, `.github/workflows/nightly.yml`.

**Trigger.** `schedule`, at 02:00 UTC, and `workflow_dispatch` to run it by hand. Scheduled workflows run on the default branch, so the code timed is `main` as it stands at that moment. The jobs run only in `sofeikov/hotkernels`, not in forks (`if: github.repository == ...`).

**Jobs:**

| Job | Runs on | What it does |
|---|---|---|
| `gate` | `ubuntu-latest` | Reads the repository's runners, as `pick-runner.yml` does, with the same `RUNNER_STATUS_TOKEN`. Passes only if a runner labelled `hotkernels-mac` is online and none is busy. Otherwise it writes the reason to the job summary and the night has no run |
| `run` | `hotkernels-mac` | Waits up to 15 minutes for the one-minute load average to fall under `BUSY_LOAD` (1.0, as `delivery time` uses). Then runs `delivery workloads --size M,L --history` into a worktree of `perf-history`, commits the new file and pushes it. A run that never saw a quiet minute still runs, and its file says so |
| `site` | `ubuntu-latest` | Starts `pages.yml` with `gh workflow run`, so the site shows the night's run. A push made with the workflow's own token starts no other workflow; a `workflow_dispatch` is the exception |

**Permissions.** `run` needs `contents: write` to push to `perf-history`, and `site` needs `actions: write`. Nothing else. The Mac only runs code from `main`.

**What can still disturb a run.** A CI job that reaches the Mac while the run is going. `pick-runner.yml` sends jobs to the Mac even when it is busy, and they queue behind the run if the Mac takes one job at a time. If it takes more, the load average at the end shows it, and the run is marked busy. Proposed: no more than that. Pausing CI for the run would block pull requests for its length.

### Series and breaks

A **series** is the runs of one workload at one size on one machine. Within a series, consecutive runs are comparable only if NumPy, Python and the workload's own libraries have the same versions. Where one of them changes, the series has a **break**: the chart draws a gap, and `delivery history` marks the line. A speedup that moves at a break is NumPy's or the library's doing until shown otherwise.

The **headline** is the geometric mean, over the workloads, of their speedup at size `M`, one per run. It is comparable only while the set of workloads is the same, so adding or removing a workload is a break in the headline too. Workloads will be added often at first; the per-workload charts under the headline carry the history across those breaks.

Runs are ordered by `recorded_at`, when they started. A backfill run times a past commit, so it takes its commit's date instead.

### Changes on the time axis

The chart marks each merge to `main` that changes what a user gets, between the commits of two consecutive runs. Those are the merges that add a release note ([AGENTS.md](../AGENTS.md#release-notes)), and the note is the text a reader wants: "`np.cumsum` on `int32` is served by a kernel".

The site finds them when it is built. `pages.yml` checks out `main` with its full history, and fetches `perf-history`. For each pair of consecutive runs, `git log --first-parent <earlier>..<later> -- release-notes/upcoming` gives the merges, and the note each one added. A marker shows the pull request's number, its note type and the note. A release moves the notes out of `upcoming/`, which is a deletion and makes no marker.

Built from git at each build, not stored with the run, so the markers can be improved later without running anything again.

### The page

A new page, "Over time", in the report section of the site. From the top:

1. **The headline chart.** Geometric-mean speedup against date, one line per machine, with a line at 1.0x. Gaps at breaks. A run with a busy machine is drawn as a hollow dot. A toggle shows the change markers, with the note in a tooltip.
2. **The night's table.** Clicking or tabbing to a dot fills a table under the chart with that run's workloads: time without and with hotkernels, the speedup, and `report()`'s kernel and "Left to NumPy" counts. Sortable, as the other tables on the site.
3. **One small chart per workload,** speedup at `M` and `L` against date, with the same markers.
4. **What a run is:** the machine, the sizes, the schedule, and a link to the data branch.

The data goes on the element as JSON, as the site's other widgets take it. A new widget, `history.js`, draws it as SVG, with the colours, axes and read-out of `series.js`. No chart library.

### The command

```
uv run delivery history [--machine SLUG] [--workload NAME] [--size S|M|L]
```

Defaults: the machine it runs on, every workload, size `M`. For each series, one line per run: date, commit, NumPy version, the time without and with hotkernels, the speedup, and marks for a break, a busy machine and a manual run. Rendered from a Jinja template, as the other reports are.

### Noise

How far a workload's figure moves between two runs of the same commit is not measured yet. In [pull request 90](https://github.com/sofeikov/hotkernels/pull/90) two workloads were run twice at the same size, at a load average of about 2, and agreed within 1%; two pairs say little.

Whether the Mac is quiet at night is not known either. [Issue 51](https://github.com/sofeikov/hotkernels/issues/51) found a load average of 2.2 to 5.5 on an evening with no CI job running, from the window server and the desktop apps, never under 1.0. If 02:00 is no better, every run is marked busy, and the figures are worth what the noise band says. A speedup is a ratio of two times taken in the same run, and moves less with the load than either time does. The first week of runs answers both questions.

The nightly runs measure the noise without extra cost: a night with no new commit runs the same code as the night before. Once a commit has several runs, the chart draws the range of its runs as a band. Neither the page nor the command calls a change a regression or a gain until that band is known.

## Considered and not proposed

| Option | Why not |
|---|---|
| A run per commit | Most commits do not change the package, and a branch can have dozens a day. Each run takes the Mac for its length. A night finds a change to within one day's merges, and the markers name them |
| Hosted runners | Shared virtual machines, and a different host from one night to the next. Their figures would move with the host, and [AGENTS.md](../AGENTS.md) takes timings on native hardware only |
| A hosted Linux series beside the Mac's | The same objection. An x86-64 series waits for a quiet x86-64 machine, [issue 51](https://github.com/sofeikov/hotkernels/issues/51) |
| History files on `main`, under `delivery/results/history/` | A bot commit a night on `main`. See [The data branch](#the-data-branch) |
| Workflow artifacts | No commit anywhere, but GitHub deletes them after 90 days at most |
| [asv](https://asv.readthedocs.io/) | It is made for this: it builds the package at each commit, stores results per commit and machine, and publishes charts. But it builds each commit in its own environment, and hotkernels is a `uv` workspace member built by meson. Teaching asv that build costs more than the files and the widget here. The decision against asv as the runner, in [workloads-design.md](workloads-design.md#considered-and-not-proposed), stands |
| A chart library | The site draws every chart with its own widgets, which follow its theme. One more line chart does not need a library |

## Documents that change

| Document | Change |
|---|---|
| `docs/workloads-design.md` | A link here from "What the results are used for" |
| `README.md` | `delivery history`, and `--history`, in "Using the delivery tool"; a link to the new page |
| `docs/release-design.md` | "What runs on GitHub" lists `nightly.yml` |
| `docsite/pages/report/` | The new page, and its place in the report's index |
| `.github/workflows/pages.yml` | Full history of `main`, and a fetch of `perf-history`, before the build |

No release note: nothing under `hotkernels/` changes.

## Setup outside the repository

- Create the `perf-history` branch, empty, once.
- If a ruleset ever covers all branches, exempt `perf-history` for GitHub Actions.
- `RUNNER_STATUS_TOKEN` already exists for `pick-runner.yml`; `gate` reads the same list of runners.

## Order of work

1. This document, reviewed. Pull request 90 merged.
2. `--history`, the new `meta` fields, the refusal on a dirty tree, and `delivery history`.
3. `nightly.yml` and the `perf-history` branch. One run started by hand.
4. The page and the widget, once a series has a week of runs.

## Exit criterion

- `delivery workloads --history DIR` writes a history file, and refuses to from a dirty tree.
- The nightly workflow runs on the Mac and pushes a file to `perf-history`, and has no run on a night the Mac is offline.
- `delivery history` shows the runs of one series in order, and a break where the NumPy version changes. Checked with a run under each of two NumPy versions.
- The page shows the headline, the markers of the merges between two runs, and a night's table.

## Out of scope

- History for `judge run` and `delivery time`.
- Deciding automatically whether a change is significant.
- Filling in history for commits before the first nightly run, except the one-off of [Decisions still open](#decisions-still-open).

## Decisions still open

1. **The judge and `delivery time`.** Their recorded runs have the same problem. The same layout, `history/<machine>/<run>.json`, on the same branch, would serve them. Proposed: yes, in a later pull request, once the workloads' history has been used.
2. **Goal lines.** doesjitgobrrr draws lines at 5% and 10% faster, the JIT's targets. hotkernels has no target for the workloads yet. Proposed: the line at 1.0x only, and a goal line once a target is set.
3. **Sizes and time.** `M` and `L` each night, at 02:00 UTC. `S` takes little time but measures mostly Python's call overhead; `L` takes most of the run. To be set once the length of a run is known.
4. **One backfill run.** Whether to time the workloads once against the binding at a few past commits (the first `install()`, issue 17's function hooks, the two-sizes change), to give the history a start. Proposed: yes, as a one-off, with `source: backfill`.

## What changed in the building

Steps 2, 3 and 4 are built. The page came before the week of nightly runs, from a run recorded by hand.

- **`--size` takes a list**, `--size M,L`, instead of a new `--sizes`.
- **Each row records `requires`**, the libraries the workload needs, so a break is decided per workload without importing the workloads.
- **The time axis is the date of the run.** It was first the commit's date, with runs of one commit drawn above one another. Read as a date axis, that put three nights of runs on the morning the commit was merged. Each run now sits at `recorded_at`, a backfill run at its commit's date, and the line goes through every run. Consecutive runs of one commit under the same versions still form a group, and the band encloses the group's dots: its width is the nights, its height the spread. The headline and the per-workload charts use the same rule.
- **Close merges share one marker.** Many pull requests merge within the same hour, and their labels overlapped.
- **Where the site reads the runs:** `origin/perf-history`, else a local `perf-history` branch, else no runs and a warning. `pages.yml` warns instead of failing while the branch does not exist.
- **The night's table sorts by its own column buttons.** No table on the site sorted before, so there was nothing to follow.
- **The site's footer** said every figure came from one Apple M1 Pro. It now says each figure names its machine.
- **`nightly.yml` builds before it waits.** `uv sync` builds hotkernels and the libraries first, so the build's own load is over before the wait for a quiet minute starts.
- **A run whose results differ is kept.** The workflow pushes its file and then fails, and the site is rebuilt either way.
- **A push that races a manual run merges and retries**, up to three times. Runs never share a file, so the merge cannot conflict.
- **`docs/release-design.md` is not changed.** Its "What runs on GitHub" describes `release.yml` alone; `nightly.yml` is described here and in its own comments.
