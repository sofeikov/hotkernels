# The CI queue

How `ci.yml` decides which of its jobs a push needs, so that the macOS jobs spend the maintainer's runners only on work whose result someone will read. It changes when jobs run and which ones; it does not change what any job does or how long it takes.

## Summary

- **The wait is in the queue, not in the jobs.** A run of `ci.yml` takes about ten minutes once it has its runners. Its two macOS jobs, `judge` and `numpy-range`, run on the four self-hosted runners on the maintainer's Mac (`.github/runners/setup.sh 4`), about ten minutes each. When pushes come together, the jobs queue behind each other: ten merges to `main` within two minutes on 21 November asked for twenty macOS jobs, and the last run finished 100 minutes after its push.
- **Much of the queue is work nobody reads.** Between 16 and 21 November, 78 of the 127 runs on `main` and 71 of the 273 runs on pull requests had a newer push on the same branch before they finished. Of the last 120 pull requests, 18 changed only documentation, which no job of `ci.yml` reads.
- **Proposed:**
  1. A newer push to a pull request cancels its unfinished run. On `main`, the running run finishes and a newer push replaces the one still waiting.
  2. A change that touches only documentation skips the three test jobs.
  3. On a pull request, `numpy-range` runs only when the change touches what it checks on ARM64: the package, an ARM64 kernel, the delivery check's cases, the lock or the project's dependencies. On `main` it runs after every merge, as today.
- **Not proposed here:** making the jobs themselves faster (`delivery check` in parallel processes, the two runs of NumPy's test suite at once, the selftest's checks in parallel). That changes `judge/` and `delivery/` and is a design of its own.

## The numbers

Read from the GitHub API for `ci.yml` and the repository's pull requests.

| | Count |
|---|---|
| Runs on `main`, 16–21 November | 127 |
| of which a newer push to `main` arrived before the run finished | 78 |
| minutes those runs went on after the newer push | 1,947 |
| Runs on pull requests, same days | 273 |
| of which a newer push to the same pull request arrived before the run finished | 71 |
| minutes those runs went on after the newer push | 1,001 |
| Pull requests, the last 120 | 120 |
| that change only documentation | 18 |
| that touch what `numpy-range` checks (see [below](#numpy-range-where-the-numpy-version-matters)) | 76 |

The minutes are wall clock and include time queued, so they overstate the runner time lost. They show the size of the problem, not the saving.

A macOS job, measured from the runs of 21 November:

| Step | Time |
|---|---|
| `judge selftest --no-speed` | about 55 s |
| two `judge check --unlocked` | about 1 min |
| `delivery check` | 4.5 to 6 min |
| `delivery numpy-tests` | about 2 min |
| `numpy-range`: `delivery check` on the oldest and the newest NumPy | 10 to 12 min |

## Design

### A newer push replaces an unfinished run

`ci.yml` gets a concurrency group per pull request, and one for `main`:

```yaml
concurrency:
  group: ci-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

- **On a pull request** the newer push cancels the older run, wherever it is. Only the newest commit of a pull request is merged, so only its result is read. `release.yml` already does the same.
- **On `main`** nothing is cancelled once it runs. GitHub keeps one run going and one waiting per group, and a newer push replaces the waiting one. A burst of ten merges becomes two or three runs instead of ten.
- **What `main` gives up.** A merge commit in the middle of a burst gets no run of its own. If the newest run fails, the cause is one of the merges since the last run that passed, and `git log` lists them. Every commit that a run does reach is checked in full, so no failure is hidden; only its author may take a minute longer to find.
- **Why not cancel on `main` too.** A run cancelled halfway through a burst would leave `main` with no result at all until the burst ends. Letting the running one finish keeps the gap between results to one run.

### Documentation skips the test jobs

A new job, `scope`, runs on `ubuntu-latest` alongside `runner`. It reads the paths the change touches and passes them to `.github/scripts/ci-scope.sh`, which prints two outputs: `tests` and `numpy_range`. `judge`, `judge-linux` and `numpy-range` need `scope` and run only when its output says so.

- **Documentation** is any path ending in `.md`, anything under `docs/` or `docsite/`, and `LICENSE`. None of the commands in `ci.yml` reads any of them: `judge`, `delivery` and `tests/` read no Markdown, and the site is built by `pages.yml` from `main`.
- **Which paths.** For a pull request, the files of the whole pull request (the API's list of its files), so a pull request that changes code and then only documentation still runs everything on its last push. For a push to `main`, the files changed between the commit before the push and the commit pushed (the API's comparison of the two). Anything else, such as a run started by hand, has no list and runs everything.
- **When the list cannot be trusted, everything runs.** An empty list, a failed request, a push that creates a branch, or a list at the API's limit (3,000 files for a pull request, 300 for a comparison) counts as a change to everything.
- **A skipped job is shown as skipped.** No check is required by branch protection on this repository, so a skipped job blocks nothing. The run still appears on the pull request with its jobs marked skipped, which tells a reader why nothing was tested.
- **Why a job rather than `paths-ignore`.** `paths-ignore` filters a whole workflow, so it could skip documentation but not decide about `numpy-range` alone. One script holds both rules and can be run on a list of paths by hand, which is how it was checked (below).

### `numpy-range` where the NumPy version matters

`numpy-range` runs `delivery check` on ARM64 macOS against the oldest NumPy the package claims and the newest on PyPI. On a pull request it runs when a path other than Markdown is one of these:

| Path | Why |
|---|---|
| `hotkernels/` | The hooks, the conditions under which they take a call, and the binding to NumPy |
| `kernels/<name>/arm64/`, `kernels/<name>/spec.py`, `kernels/<name>/naive.c` | The package builds every kernel under `kernels/` for its platform without a change to `hotkernels/` (`hotkernels/tools/generate.py`), so a new or changed ARM64 kernel, or the C a kernel ships as where it has no assembly, changes the answers on ARM64 |
| `delivery/cases.py`, `delivery/check.py` | The cases and the comparison the job runs |
| `uv.lock`, `pyproject.toml` | Which NumPy and which bottleneck the other jobs use |
| `.github/workflows/ci.yml`, `.github/scripts/ci-scope.sh` | The job and this rule themselves |

On `main` it runs after every merge that is not documentation, as today.

- **What it leaves to `main`.** A pull request that touches none of these paths changes nothing the package does on ARM64: an x86-64 kernel, the judge, the survey, the timing commands, recorded results. `judge-linux` checks the x86-64 kernels against the NumPy in `uv.lock`. A NumPy release that changes an answer is found on the next run on `main`, where before it was found on the next pull request of any kind.
- **What it saves.** Of the last 120 pull requests, 44 would not have run `numpy-range` on their first push: the 18 that are documentation, and 26 that change the survey, the judge, the delivery tool's timing commands, other workflows, a prototype, x86-64 kernels or recorded results.

### A run by hand

`ci.yml` gains `workflow_dispatch`, so the whole of it can be run on any branch from the Actions tab or with `gh workflow run ci.yml --ref <branch>`. Such a run has no list of paths and runs every job.

## Effect

Counted on the first push of each of the last 120 pull requests: 240 macOS jobs before, 178 after (102 `judge`, 76 `numpy-range`), a quarter fewer. On top of that, the runs that a newer push supersedes are stopped (pull requests) or never started (`main`), which is where the long waits of a burst came from. How much of the 100-minute waits this removes depends on how pushes arrive and is not predicted here; the runs after the change show it.

## The number of runners

The queue's capacity is the number of runners: four on the maintainer's Mac, an Apple M1 Pro with eight performance cores, two efficiency cores and 32 GB of memory.

- **More runners add capacity only while cores are free.** The jobs do not keep their cores busy. With all four runners busy on 21 November, each job's Python process used 30% to 71% of a core, about 2.5 cores in all; `delivery check` alone, run by hand, spends about 65% of its time on the processor. The load average was 12.9 at the same moment, so the rest of each job's time is spent waiting, not computing. Six runners fit on the eight performance cores by this count.
- **Spotlight indexes the jobs' files.** At the same moment `mds_stores`, Spotlight's indexer, used 82% of a core: each job builds a virtual environment and the package under `/Users/ci-runner`, and Spotlight indexes what it writes. Adding `/Users/ci-runner` to Spotlight's Privacy list in System Settings stops that; it is a setting of the machine, not of this repository.
- **What it costs.** Every job on the Mac slows the others and whatever else runs on it, including timings taken by hand, which are only trustworthy when nothing else runs.
- **Which comes first.** This design takes jobs out of the queue; it is the cheaper change and is measured first. Making the jobs faster with several processes each would change the arithmetic: a job that uses four cores leaves room for fewer runners. The runner count is best decided after both.

## How it was checked

- **The script on real changes.** `ci-scope.sh` was run on the file lists of the last 120 pull requests. The 18 it calls documentation were read: each changes only design documents, READMEs, `AGENTS.md`, attempts logs, release notes or the site. The 26 that run tests but not `numpy-range` were read too: none adds or changes an ARM64 kernel, a hook or a delivery case.
- **A long list.** 200,000 paths give the right answer, so the script holds no limit of its own below the API's.
- **On GitHub.** The pull request that brings this design runs every job, because it changes `ci.yml`. A pull request that changes only a Markdown file, opened on top of it and then closed, shows the three test jobs skipped. A second push to the pull request while its run is going shows the first run cancelled.
