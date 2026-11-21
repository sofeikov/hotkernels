# NumPy's test suite in CI

This began as a proposal and is now implemented: `delivery/numpy_suite.py`, and a step in `ci.yml` and `release.yml`. Where the building departed from the proposal, [What changed in the building](#what-changed-in-the-building) says so. It answers [issue 92](https://github.com/sofeikov/hotkernels/issues/92): run NumPy's own test suite with hotkernels installed on every change that could break NumPy behaviour, and before every release.

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| A test is reported only if it differs in the second pair of runs too | Also reported, with its first outcomes, if neither second run reported it at all | A node id the second runs could not collect would otherwise pass unseen |

## Summary

- **`uv run delivery numpy-tests` runs only by hand today.** It is step 4 of a release in [release-design.md](release-design.md#the-release-pull-request), on the maintainer's machine. No workflow runs it. `delivery check` covers the hooked calls with the cases this project wrote; NumPy's 47,000 tests cover behaviour this project did not think of.
- **It is fast enough to run on every pull request.** Both runs together take 75 seconds on an Apple M4 Max. See [What was verified](#what-was-verified).
- **Proposed:** one step in each of the two existing CI jobs that run `delivery check`, one per platform, and one step in each of the two release jobs that test the wheels. A test whose outcome differs is run again both ways before it is reported, so a test that fails at random does not fail the build.
- **Not proposed:** a cached plain-NumPy baseline, a path filter, or the suite at the ends of the NumPy range. See [Considered and not proposed](#considered-and-not-proposed).

## What was verified

`uv run delivery numpy-tests` on `main`, native:

- **Machine.** Apple M4 Max, macOS, CPython 3.14.8, NumPy 2.5.5, pytest 9.2.0, hypothesis 6.170.15. Load average 1.6.
- **Result.** 47,262 tests reported, 46,766 passed both ways, no outcome differs.
- **Time.** 75 seconds for both runs, wall clock, with the package already built. The run uses one core.

The README says "about ten minutes"; that figure is older than the current machine and package. A hosted `macos-15` runner has an M1, and the Linux runner an x86-64 processor of similar single-core speed; both are expected to take two to three minutes for the two runs.

## Design

### Running a differing test again

`delivery numpy-tests` runs the suite plain and with hotkernels installed, at the same time in two subprocesses, and compares the outcomes of every test. When some outcomes differ, it then runs only those tests again, plain and installed, each in a fresh subprocess, one after the other. It reports a test only if its outcomes differ in the second pair of runs too, so a test that fails only while the other run shares the machine is not reported.

- **Why.** Some of NumPy's tests can fail at random: tests that depend on threads, timing or free memory. Without a second run, such a test fails CI for a reason that has nothing to do with the change. Hypothesis tests are not among them: an installed NumPy has no `pytest.ini`, so its `conftest.py` loads the profile with `derandomize=True`, and both runs draw the same inputs.
- **How.** The outcomes file names each test by pytest's node id, a path under the environment's `site-packages`. The node ids go into a file that pytest reads as arguments (`pytest @ids.txt`), so their number does not meet a command-line limit.
- **What is printed.** The report keeps its totals and adds one line: how many outcomes differed in the first runs and agreed in the second. Those tests are listed under the line, so a flaky test stays visible without failing the run.
- **A limit.** If more than 100 outcomes differ, they are not run again and all are reported. That many differences means a crash or a broken hook, not chance, and running them again could take as long as the suite.

### Where it runs

| Workflow | Job | Runs on | What it tests |
|---|---|---|---|
| `ci.yml` | `judge` | Apple Silicon, chosen by `pick-runner.yml` | The package built from the checkout |
| `ci.yml` | `judge-linux` | `ubuntu-latest`, x86-64 | The package built from the checkout |
| `release.yml` | `test` | Apple Silicon, per Python 3.11 to 3.14 | Each macOS release wheel |
| `release.yml` | `test-linux` | `ubuntu-latest`, per Python 3.11 to 3.14 | Each Linux release wheel |

Each is one step after `delivery check`: `uv run delivery numpy-tests`, or `uv run --no-sync delivery numpy-tests` in the release jobs, where the wheel has replaced the package built from the checkout. The `pypi` and `testpypi` jobs already need `test` and `test-linux`, so a release cannot skip the suite.

A step in an existing job, not a job of its own: a new macOS job waits for a runner of its own, which on GitHub's hosted runners can take a quarter of an hour (see `pick-runner.yml`). A step adds two to three minutes to a job that has its runner already.

The suite runs on the NumPy in `uv.lock`, on the Python the job uses. The `numpy-range` job keeps checking both ends of the supported range with `delivery check`.

### Documents

The implementation updates, in the same pull request:

- [release-design.md](release-design.md): step 4 of the release pull request no longer lists `delivery numpy-tests`; CI runs it on the release wheels. [What runs on GitHub](release-design.md#what-runs-on-github) says so.
- [README.md](../README.md) and [docsite/pages/start.md](../docsite/pages/start.md): the time the command takes, and that CI runs it.
- [delivery-design.md](delivery-design.md): exit criterion 3 says where it is now checked.

No release note: the change touches `delivery/` and the workflows, not the package.

## Considered and not proposed

- **A cached plain-NumPy baseline.** Issue 92 proposed to run the plain half once per environment and keep its outcomes in an `actions/cache` entry, keyed by the NumPy, Python, pytest and hypothesis versions, the platform, and a hash of `delivery/numpy_suite.py` and `delivery/pytest_plugin.py`. That saves one run of the suite, which the issue estimated at five minutes. It takes 37 seconds on an M4 Max, and about a minute on a hosted runner. The saving does not pay for a new option, a cache key of six parts, and a baseline that can be stale. Running a differing test again also covers what the baseline's staleness check was for.
- **Committing baselines under `delivery/results/`.** They would be valid for one platform and one set of versions, and stale at the next lock update. With no cache, there is nothing to share.
- **A separate job with a path filter.** Path filters in GitHub Actions apply to a whole workflow, not to a job. A separate workflow that runs only on changes to `hotkernels/` and `kernels/` would spare documentation pull requests two to three minutes, and would cost each pull request it runs on the wait for a second macOS runner. A step in the existing job is cheaper on the pull requests that matter.
- **The suite at the ends of the NumPy range.** Three runs per job instead of one. `delivery check` already runs at both ends; a difference that only NumPy's tests find and only at one end of the range is possible but is not what this change is for. It can be added later as a step in `numpy-range`.

## Order of work

1. This document.
2. Run a differing test again, in `delivery/numpy_suite.py` and its template, with the report line.
3. The steps in `ci.yml` and `release.yml`.
4. The documents listed under [Documents](#documents).
