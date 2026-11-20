# Releases: how `hotkernels` gets to PyPI

This began as a proposal and is now implemented: the commands in [`release/`](../release), the workflows [`release.yml`](../.github/workflows/release.yml) and [`release-notes.yml`](../.github/workflows/release-notes.yml), and the instructions in [AGENTS.md](../AGENTS.md#release-notes). Where the building departed from the proposal, [What changed in the building](#what-changed-in-the-building) says so. The rest of the text stands as proposed.

It answers the core design's note under [Later](core-design.md#later) that "the release pipeline and its GitHub setup get their own design document", and [issue 24](https://github.com/sofeikov/hotkernels/issues/24). It covers when a release happens, how its notes are written, how the version is set, and what runs on GitHub to build and publish it. What goes into a wheel is [issue 19](https://github.com/sofeikov/hotkernels/issues/19); the building settled part of it, below.

## What changed in the building

| Proposed | Built | Why |
|---|---|---|
| A wheel's contents and flags left to issue 19 | Wheels only, no source distribution. C kernels in a wheel are compiled for `apple-m1` through a new build option, `kernel_cpu` in `hotkernels/meson.options`; a build from a checkout still uses `-mcpu=native`, as the judge does | A wheel built on a CI runner with `-mcpu=native` would carry whatever that runner's chip allows, and could crash on an M1. Every later Apple chip runs M1 code. The judge still times `-mcpu=native`, which on an M1 is the same code. A source distribution still cannot build outside the checkout, so `pip` on another platform finds nothing to install. [distribution-design.md](distribution-design.md) adds one |
| Each wheel installed into a clean environment outside the checkout, then `delivery check` | Two tests. cibuildwheel installs each wheel alone into a fresh environment and checks `install()` puts kernels in place with no hook failing. Then, per Python, the wheel replaces the package in the workspace environment and `delivery check` runs against it | `delivery check` needs the repository's `delivery/` package, so it runs in the workspace; the fresh-environment test is the one that proves the wheel needs nothing from the checkout |
| `release.yml` runs on tags | Also on pull requests that touch the package, kernels, the judge's build code or the release tooling, building and testing the wheels without publishing | A change that breaks a wheel is caught before it merges rather than at release time |
| The release-note check part of CI | Its own workflow, `release-notes.yml`, rerun when a label is added or removed | Adding `no release note` clears the failure without a new push. A skipped job counts as passing |
| `report()` prints the version | It does, on its first line: `hotkernels 0.1.0: installed, ...`. `hotkernels.__version__` reads the installed package's metadata | |
| Notes may come from `kernels/<name>/arm64/kernel.s` or `naive.c` | Also `spec.py` | A spec generates the C prototypes and decides whether a kernel is exact, so it changes the package |

### Who can publish

Three things have to line up for an upload to PyPI, and only the maintainer can make each happen:

1. **PyPI accepts uploads for `hotkernels` only through trusted publishing**, from repository `sofeikov/hotkernels`, workflow `release.yml`, environment `pypi`. No API token is stored in the repository or its settings.
2. **The `pypi` environment needs the maintainer's approval on every run** and accepts only deployments from tags matching `v*`. A job that names the environment from a branch is refused; one from a tag waits for the click.
3. **Only an administrator, which is the maintainer alone, can create, move or delete a `v*` tag**: the repository ruleset "release tags".

A pull request from a fork runs `release.yml` with no write permission and no access to the environment, and its `pypi` job is skipped since it is not a tag. Someone who edited `release.yml` on a branch would still need the environment, and so the approval.

## Summary

- **Merging to `main` does not publish.** `main` is always releasable; a release is a deliberate step taken every few weeks, or at once for a fix to a wrong result.
- **Each pull request leaves a note.** A pull request that changes what a user of the package gets adds one short file to `release-notes/upcoming/`. CI fails such a pull request without one. The notes pile up between releases.
- **A release starts with a release pull request.** It is an ordinary pull request that does three things: bumps the version, gathers the notes into one document with a summary written by hand that spans every change since the last release, and updates the headline numbers re-measured on native hardware. Merging it publishes nothing.
- **Pushing a tag publishes.** After the release pull request is merged, the maintainer runs `uv run release tag`, which tags the merged commit `v<version>` with the version it reads from the repository, so the two cannot disagree. The tag starts a workflow that checks the tag against the version, builds the wheels, tests them, waits for the maintainer's approval, uploads to PyPI, and creates the GitHub release with the notes as its text.
- **Uploads come only from GitHub Actions, never from a laptop.** PyPI accepts an upload from that one workflow through trusted publishing; no password or token exists anywhere to upload with.

## Why not publish on every merge

Publishing on every merge would release the judge's own changes, documentation fixes and half-finished kernel work as new versions, each with a one-line note. A user reading the history of the package would learn little from it. The project also has a rule that a release cannot be automated around: timings are only meaningful on native hardware, so the speedups a release claims have to be taken on a real machine by a person, not on a CI runner.

Releasing by hand from a laptop has the opposite problem: nothing checks that what was uploaded is what `main` holds, and the notes are written from memory.

The proposal sits between the two. Changes accumulate on `main`, each with its note written while its author still knows what it does. A release is cut when there is something worth telling users, and everything after the decision to cut it runs in CI.

## The note each pull request leaves

A note is a Markdown file of one to three sentences, written for someone who uses `hotkernels`, not for someone who works on it:

```
release-notes/upcoming/31.kernel.md
```

```markdown
`np.cumsum` on `int32` is served by a kernel: about 11x faster than NumPy at a million elements.
```

The file name is the pull request number and a type. The types become the sections of the release notes:

| Type | Section | For |
|---|---|---|
| `kernel` | New kernels | A NumPy call that a kernel now serves, or a new element type |
| `speed` | Faster | A kernel or the binding got faster |
| `fix` | Fixes | A wrong result, a crash, or a call that went to the wrong place |
| `compat` | Compatibility | A Python or NumPy version gained or lost, a new platform, a change to what `install()` installs by default |
| `change` | Other changes | Anything else a user would notice, such as `report()` output |

**Which pull requests need one.** Those that change `hotkernels/`, or a kernel the package installs (`kernels/<name>/<arch>/kernel.s` or `naive.c`). A CI job checks this. A pull request in those paths that a user would not notice, such as a comment or a refactor, gets the `no release note` label and the job passes. Pull requests that touch only the judge, the survey, the documentation site or `docs/` need nothing.

**Who writes it.** Whoever opens the pull request, which is usually an agent. [AGENTS.md](../AGENTS.md) tells agents when a note is needed, how to name it and how to write it. The note is short; the long text is written once per release, in the highlights.

**One note per pull request, not per commit.** A kernel run keeps many commits; its pull request leaves one `speed` note saying what the run achieved.

**Numbers in a note are allowed but provisional.** A note may quote the judge's figure at the time. The release pull request re-measures, and the release notes quote the re-measured figures.

This is how NumPy writes its own release notes (`doc/release/upcoming_changes/`), with the same tool, [towncrier](https://towncrier.readthedocs.io/). Towncrier renders the notes through a Jinja template, which is the repository's rule for generated text, and its `check` command is the CI job.

## The release pull request

Nothing in GitHub treats it differently from any other pull request. What makes it the release pull request is its content: the version bump, `release-notes/<version>.md`, and an empty `release-notes/upcoming/`. CI runs on it as usual. Merging it does not publish anything; it puts on `main` the commit that the tag will name.

Anyone can open it; the maintainer decides to merge it. One command prepares it:

```
uv run release prepare 0.2.0
```

It:

1. Sets the version to `0.2.0` in the one place the package keeps it ([Versions](#versions)).
2. Runs towncrier, which writes `release-notes/0.2.0.md` from the notes in `upcoming/` and deletes them.
3. Leaves a **Highlights** section at the top of that file to be written by hand, with the list of pull requests merged since the last tag as material.

Then, on the maintainer's machine:

4. `uv run judge selftest` and `uv run delivery check`. NumPy's own test suite is not on this list: `release.yml` runs `delivery numpy-tests` against every wheel ([numpy-tests-ci-design.md](numpy-tests-ci-design.md)).
5. The timings: `uv run delivery time`, and `uv run judge run` for any kernel whose note claims a speed. The results files under `delivery/results/` and `kernels/` are updated in the same pull request, and the README's tables if they changed.
6. The highlights: a few paragraphs on what this release means for a user, with the re-measured numbers. This is where the notes become longer than any one pull request. An agent can draft it from the notes and the pull request descriptions; the maintainer edits it.

The finished notes look like this:

```markdown
# hotkernels 0.2.0

## Highlights

Three more NumPy calls are served by kernels, ... (hand-written)

Measured on an Apple M1 Pro, NumPy 2.5.5, Python 3.13.

## New kernels

- `np.cumsum` on `int32` is served by a kernel: ... (#31)

## Fixes

- `np.clip(x, lo, hi, out=x)` gave wrong results on some data and now goes to NumPy. (#34)
```

Fixes to wrong results come first within Fixes, and are named in the highlights. A user who stays on the previous version needs to know what they are exposed to.

### Publishing it

Once the release pull request is merged, on an up-to-date `main`:

```
uv run release tag
```

It reads the version from `meson.build`, refuses if the working tree differs from `origin/main`, if `v<version>` exists already, or if `release-notes/<version>.md` is missing, and otherwise creates the annotated tag `v<version>` on that commit and pushes it. Typing `git tag` by hand does the same, and the workflow checks the result either way; the command only saves typing the version twice.

**How the tag and the version stay in step.** The version is written in one file, `meson.build`, and changed only by the release pull request. The tag is made from that file, never typed. The workflow's first job reads the file at the tagged commit and stops the release if the tag says anything else. So a tag that disagrees with the package's version is never published; it is deleted and pushed again.

### When to release

There is no fixed calendar. A release is cut when `release-notes/upcoming/` holds something a user would want, which is expected every few weeks while kernels are being added. Two exceptions:

- **A fix to a wrong result is released at once**, as a patch release, even if it is the only note.
- **Notes that wait a long time are a prompt.** A scheduled workflow opens an issue when notes have been waiting in `upcoming/` for more than four weeks. It only reminds; it does not release.

## Versions

Semantic versioning, read for this package:

| Part | Bumped when | Example |
|---|---|---|
| Minor (`0.2.0`) | New kernels or operations, a change to what `install()` does by default, a supported Python or NumPy version dropped | Most releases |
| Patch (`0.2.1`) | Only fixes and speed | A wrong-result fix released at once |
| Major (`1.0.0`) | When the maintainer decides the public API (`install`, `uninstall`, `enabled`, `report`) is stable | Not planned |

**One place holds the version.** Today it is written twice, in `hotkernels/pyproject.toml` and in `hotkernels/meson.build`. `meson-python` can take it from `meson.build` when `pyproject.toml` declares it dynamic, so that becomes the one place. The package gains `hotkernels.__version__`, and `report()` prints it, so a bug report says which release it is about.

**The first release is `0.1.0`**, the version the package carries today. Only a `0.0.1` placeholder is on PyPI, and the first notes describe the package as it stands rather than a change from an earlier version.

**Pre-releases** use the same path: a release pull request for `0.3.0rc1`, tagged `v0.3.0rc1`, publishes a pre-release, which `pip` installs only when asked. Worth it for a release that changes default behaviour; otherwise optional.

The root package, `hotkernels-judge`, is not published and keeps no version of its own beyond what `uv` needs.

## What runs on GitHub

A new workflow, `.github/workflows/release.yml`.

**Trigger.** A pushed tag that matches `v*`. It can also be started by hand on an existing tag, to retry a release that failed part-way, or on any commit to upload to TestPyPI as a rehearsal.

**Jobs, in order:**

| Job | Runs on | What it does |
|---|---|---|
| `version` | Ubuntu | Checks the tag against the version in `meson.build` at the tagged commit, that the commit is on `main`, that `release-notes/<version>.md` exists and that `upcoming/` is empty. Any mismatch stops the release before anything is built |
| `build` | macOS 15, Apple Silicon | Builds the source distribution and one wheel per supported Python with [cibuildwheel](https://cibuildwheel.pypa.io/). What the wheel contains and the compiler flags it uses are [issue 19](https://github.com/sofeikov/hotkernels/issues/19) |
| `test` | macOS 15, Apple Silicon | Installs each wheel into a clean environment, outside the checkout, and runs `delivery check` and `delivery numpy-tests` against it. This is the test that the wheel works without the repository next to it |
| `publish` | Ubuntu, environment `pypi` | Waits for the maintainer's approval, then uploads with [trusted publishing](https://docs.pypi.org/trusted-publishers/) |
| `release` | Ubuntu | Creates a GitHub release for the tag with `release-notes/<version>.md` as its text and the wheels attached |

**The order protects what cannot be undone.** PyPI never accepts the same version twice, even after a file is deleted. Nothing is uploaded until every wheel is built and tested. If a run fails before the upload, the tag is deleted, the problem fixed on `main` (with the release notes amended if needed), and the tag pushed again on the new commit. If it fails after the upload, only the `release` job is rerun.

**The approval** is the last point to stop a release. Pushing the tag has already said "release this commit"; the approval says "these built files". It costs one click, and is kept for the first few releases.

**Timings are not taken in CI.** The runners are too noisy, as `ci.yml` already notes. The numbers in the notes come from step 5 of the release pull request.

## Setup outside the repository

Done once by the maintainer:

| Where | What |
|---|---|
| PyPI | The name `hotkernels` is held already, by a placeholder `0.0.1` whose repository link still uses the old name `kernel-arena`; the first release replaces that metadata. Add a trusted publisher to the project: owner `sofeikov`, repository `hotkernels`, workflow `release.yml`, environment `pypi`. Revoke any API token made for the placeholder upload |
| TestPyPI | A pending trusted publisher for project `hotkernels`: the same, with environment `testpypi` |
| GitHub, environments | `pypi` with the maintainer as required reviewer, limited to tags matching `v*`; `testpypi` limited to `main`, without a reviewer |
| GitHub, tag ruleset | Only the maintainer may create, move or delete tags matching `v*`, so no one else can start a release |
| GitHub, labels | Create `no release note` |
| GitHub, branch protection | Require `ci` and the release-note check on `main` |

## Order of work

| Step | What | Depends on |
|---|---|---|
| 1 | The version in one place, `__version__`, `report()` prints it | Nothing |
| 2 | towncrier, `release-notes/`, the template, `uv run release prepare` and `uv run release tag`, the CI check for notes, the label | Nothing. The instructions in AGENTS.md land with this design, so pull requests leave notes from now on; this step adds the check that enforces them |
| 3 | `release.yml` up to `release`, publishing to TestPyPI only | Step 1; a wheel that builds, even if it is not yet the final one |
| 4 | Answers to issue 19: a source distribution that builds without the checkout, the flags for C kernels, `python -m hotkernels` | Its own decisions |
| 5 | The first release, `0.1.0`, to PyPI | Steps 1 to 4, and [issue 18](https://github.com/sofeikov/hotkernels/issues/18) if the release is to claim a NumPy range |

Steps 1 and 2 can land now; they are useful before anything is published, because notes pile up from the next pull request.

## More than one platform

The pipeline was built for one platform, macOS on ARM64. The second, Linux on x86-64 ([x86-64-design.md](x86-64-design.md#the-package-and-its-delivery)), is in the workflow as follows:

- **`build` and `test` run once per platform**, each on a runner of that platform. `build` makes the source distribution and the macOS wheels; `build-linux` takes that source distribution and builds the Linux wheels from it with `cibuildwheel` in a manylinux container; `test` and `test-linux` run `delivery check` against each wheel. Nothing is uploaded until every platform's wheels are built and tested: the publishing jobs need both test jobs and collect both sets of wheels.
- **Linux wheels are `manylinux` on x86-64 only.** No `musllinux` wheel: the kernels have only been judged and checked with glibc. A Linux wheel installs on any x86-64 processor and `install()` does nothing on one below the x86-64-v3 level.
- **The Linux wheels are compiled by clang, which is installed into the manylinux image.** A C kernel in a wheel has to be compiled by the compiler the judge timed it with, which is clang, and the image's own compiler is gcc. While every kernel had assembly on x86-64, assembled by GNU `as` as the judge does, gcc built only the binding and that was acceptable. `unique_i64` was the first kernel to ship as C on x86-64 ([unique-design.md](unique-design.md#x86-64)), so `[tool.cibuildwheel.linux]` installs clang before the build and sets `CC` to it. That kernel has x86-64 assembly since; the three `argsort` kernels and bottleneck's four are C there and need clang still.
- **The job for other platforms moved to Linux on ARM64**, since a build on Linux x86-64 now needs a compiler.
- **macOS jobs that publish nothing may run on the maintainer's own Mac.** Hosted macOS runners are often not assigned for a quarter of an hour. [`pick-runner.yml`](../.github/workflows/pick-runner.yml) sends the judge job of `ci.yml` and the `build` and `test` jobs of a pull request to a self-hosted runner when it is online, busy or not, and to a hosted one otherwise. It is never used for a pull request from a fork or for a tag, so a published wheel is never built or tested on a personal machine. On the Mac, `build` makes the wheels with `uv build` and the settings `[tool.cibuildwheel]` gives, because `cibuildwheel` installs Pythons system-wide and the runner's account may not; a pull request therefore does not exercise `cibuildwheel` itself unless it falls back to a hosted runner. A tag always does.
- **C kernels in a wheel are compiled for the oldest processor that platform's wheels claim to run on:** `apple-m1` on macOS, the x86-64-v3 level on Linux. The `kernel_cpu` build option carries either.
- **A note's speed figure names its platform** once the package ships for more than one.
- **Re-measuring needs every platform.** Step 5 of the release pull request runs on the maintainer's machine, which is one platform. How the other platform's figures are re-measured was [open issue 6](x86-64-design.md#open-issues) of the x86-64 design. A rule is proposed below; it is not decided.

### Re-measuring on two platforms: proposed

**Proposed, for the maintainer to accept, change or replace.** Nothing in the release tool enforces it yet.

1. **A figure names its machine.** Every speed in the release notes and in the highlights says which processor it was measured on. AGENTS.md already asks this of a note once the package ships for more than one architecture.
2. **Step 5 runs once per platform, on a machine of that platform, on the release pull request.** `uv run delivery time --record` writes `delivery/results/<machine>.md`, one file per machine, with the revision it was taken at, so the two runs do not touch each other's file and can be committed to the same pull request from two machines. `uv run judge run` for a kernel whose note claims a speed runs on the machine of the note's architecture.
3. **A platform may be carried instead of re-measured when nothing of its own changed.** If no file under `kernels/*/<arch>/`, no shared `spec.py` or `naive.c`, and nothing under `hotkernels/` differs between the revision recorded in that platform's results file and the release commit, its figures are the recorded ones and the notes quote them with that revision's date. Otherwise the platform is re-measured or the release waits.
4. **The highlights end with one line per machine,** in place of today's single "Measured on" line.

Why this and not the alternatives:

- **Both machines every time** is rule 2 without rule 3. It is the honest default and costs a second machine for every release, including one that only changed the other platform's kernels.
- **One platform's figures only,** with the other's left out of the notes, makes a release say nothing about half its users.
- **Measuring in CI** needs runners that are quiet, and the judge's own design says shared runners are too noisy to time anything.

What would make rule 3 checkable: `uv run release verify` comparing each results file's recorded revision with the tagged commit by the paths above. That is a change to the release tool and is not made here.

## Out of scope

- What a wheel contains, its compiler flags, and the source distribution's contents: [issue 19](https://github.com/sofeikov/hotkernels/issues/19).
- The NumPy versions the package claims: NumPy 2.3 and later, decided in [delivery-design.md](delivery-design.md#which-numpy-versions) ([issue 18](https://github.com/sofeikov/hotkernels/issues/18)).
- conda-forge. Worth it after PyPI works.
- Publishing the release notes on the documentation site. A small change to `docsite/` once there are notes to show.

## Decisions made in review

| Question | Decision |
|---|---|
| A note per pull request, or write the notes at release time from the list of merged pull requests? | A short note per pull request, written by whoever opens it, usually an agent. The long text is the release's highlights |
| towncrier, or a few lines of our own? | towncrier |
| What starts the release: merging the release pull request, or pushing a tag? | Pushing a tag. The tag is made from the version in the repository by `uv run release tag`, and the workflow refuses a tag that disagrees with it |
| An approval before the upload? | Yes, for the first few releases |
| First version on PyPI | `0.1.0` |
| Do pull requests from agents, including kernel runs, follow the same rule? | Yes, with explicit instructions in [AGENTS.md](../AGENTS.md) |
