<p class="lede">A speedup recorded once says how fast the package was on one day. This page keeps every run of the workloads: programs and library calls not written for hotkernels, timed on plain NumPy and after <code>hotkernels.install()</code>. It shows whether a change to the package made them faster, slower, or neither, and when.</p>

## The workloads, run by run

<figure class="widget">
<div data-widget="history" data-repo="[[ site.repo ]]"></div>
<figcaption>The headline is the geometric mean, over the workloads, of how many times faster each runs with hotkernels at size M. Each dot is one run, placed at the date it ran. A dashed line is a merge to <code>main</code> that left a release note; choose it to read the note. Choose a dot to see that run's workloads in the table, and below it one chart per workload at sizes M and L.</figcaption>
</figure>

## How to read it

**A gap in a line** means that Python, NumPy or a library a workload calls changed between two runs. A speedup that moves at a gap is the new version's doing until shown otherwise. Adding or removing a workload is also a gap in the headline, so the charts per workload carry the history across it.

**A band** encloses consecutive runs of one commit: its width is the nights they ran on, its height their spread. A night with no new commit runs the same code again, so the band shows how far a figure moves by chance. Until the bands are known, a step smaller than them is not a change.

**A hollow dot** is a run taken while the machine was in use: a load average above 1.0 when the run started or when it finished. The speedup is a ratio of two times taken in the same run, and moves less with the load than either time does.

## What a run is

- **Where:** the maintainer's Apple M4 Max, the self-hosted runner `hotkernels-mac`, and only there. A night the Mac is offline or busy with CI has no run. A hosted runner never stands in, since its figures would move with the host.
- **When:** every night at 02:00 UTC, timing `main` as it stands then. A person can also record a run by hand; it is marked as such.
- **What:** `delivery workloads` at sizes M and L, with `--history`. Each workload runs both ways, results are compared bit for bit, and the times are the best of alternating runs. The workloads are listed in [docs/workloads-design.md]([[ site.repo ]]/blob/main/docs/workloads-design.md).
- **Where it is kept:** one file per run on the branch [`perf-history`]([[ site.repo ]]/tree/perf-history), never edited once written. `uv run delivery history` prints the same runs in a terminal after `git fetch origin perf-history`.

The design is in [docs/workloads-history-design.md]([[ site.repo ]]/blob/main/docs/workloads-history-design.md).
