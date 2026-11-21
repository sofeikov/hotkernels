"""Tests of `delivery apps` that need no timing: its sizes, its records and the Linux profile.

The Linux profile is tested on the output of `perf script`: one capture from an
ARM64 Linux container (tests/data/perf-script-aarch64.txt, its stacks cut to
six frames) and samples written by hand in the form perf prints on x86-64.
Run with `uv run --group dev pytest tests`.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from delivery import app_profile as P
from delivery import app_timing as A
from delivery import apps, cli

# Captured `perf script -F comm,tid,ip,sym,dso` output of `sklearn_metrics`,
# small, in a Docker container on ARM64 Linux: three samples or fewer of each
# file found at the top of a stack.
CAPTURE = Path(__file__).resolve().parent / "data" / "perf-script-aarch64.txt"
# A copy of the first record of `delivery apps`, delivery/results/apple-m1-pro-apps.json
# as written before applications had two sizes, with its report rewritten by
# `delivery apps-report` in the layout of two sizes. A copy, so a new record
# can replace the committed one.
FIRST_RECORD = Path(__file__).resolve().parent / "data" / "apps-record-one-size.json"
# The compiled files of the capture that belong to packages, as `package_images` maps them.
CAPTURE_PACKAGES = {
    "_multiarray_umath.cpython-312-aarch64-linux-gnu.so": "numpy",
    "_sparsetools.cpython-312-aarch64-linux-gnu.so": "scipy",
}
# Hand-written samples in the form perf prints on x86-64: a symbol with its
# offset, a C++ symbol with parentheses, an address with no symbol in a file
# with none, the kernel, OpenBLAS as NumPy's wheel ships it, hotkernels, and
# a sample whose top frame is libm's exp. The comm of the third holds a space.
X86_SCRIPT = """\
python3 4242
\t    7f3a1b2c3d4e DOUBLE_subtract+0x4e (/venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so)
\t    7f3a1b2c0000 generic_wrapped_legacy_loop+0x20 (/venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so)

python3 4242
\t    7f3a1b2c3d50 DOUBLE_subtract+0x52 (/venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so)

pool worker 4243
\t    7f3a00001234 dgemm_kernel_HASWELL (/venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-ff651d7f.so)
\t    7f3a00000010 [unknown] ([unknown])

python3 4242
\t    7f3a1b2c9999 void np::sort::(anonymous namespace)::aquicksort_<np::sort::Tag<double> >(void*, long*, long) (/venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so)

python3 4242
\tffffffff81a0b2c3 clear_page_erms ([kernel.kallsyms])
\tffffffff81a00000 do_anonymous_page ([kernel.kallsyms])
\t    7f3a1b2c0000 PyDataMem_UserNEW (/venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so)

python3 4242
\t    7f3a22220000 searchsorted_f64 (/venv/lib/python3.12/site-packages/hotkernels/_kernels.cpython-312-x86_64-linux-gnu.so)

python3 4242
\t    7f3a33330000 [unknown] (/usr/lib/x86_64-linux-gnu/libc.so.6)

python3 4242
\t    7f3a44440000 __exp1 (/usr/lib/x86_64-linux-gnu/libm.so.6)

"""
# The packages of the hand-written samples' files.
X86_PACKAGES = {
    "_multiarray_umath.cpython-312-x86_64-linux-gnu.so": "numpy",
    "libscipy_openblas64_-ff651d7f.so": "numpy.libs",
    "_kernels.cpython-312-x86_64-linux-gnu.so": "hotkernels",
}
# A C++ name from the hand-written samples, as long as perf prints it.
CXX_NAME = "void np::sort::(anonymous namespace)::aquicksort_<np::sort::Tag<double> >(void*, long*, long)"


def _first_timing() -> A.AppTiming:
    """Return the first timing of the first record, to stand in for a measurement."""
    return A.from_row(json.loads(FIRST_RECORD.read_text())["rows"][0]["timing"])


def test_every_application_has_every_size():
    """Each application declares a description for both sizes, and they differ."""
    for name in apps.NAMES:
        module = apps.load(name)
        assert tuple(module.SIZES) == apps.SIZE_NAMES, name
        assert module.SIZES["small"] != module.SIZES["large"], name


def test_every_application_is_in_one_set():
    """The sets partition the applications, and the default set is the first study's."""
    members = [name for names in apps.SETS.values() for name in names]
    assert sorted(members) == sorted(set(members))
    assert tuple(members) == apps.NAMES
    assert apps.SETS[apps.DEFAULT_SET][0] == "mlp_train"


def test_each_set_records_under_its_own_name():
    """The first set keeps the record name it had; the second records beside it."""
    assert cli._apps_stem({"cpu": "Apple M1 Pro"}) == "apple-m1-pro-apps"
    assert cli._apps_stem({"cpu": "Apple M1 Pro", "set": "first"}) == "apple-m1-pro-apps"
    assert cli._apps_stem({"cpu": "Apple M1 Pro", "set": "second"}) == "apple-m1-pro-apps-second"


def test_particle_filter_resamples_in_proportion_to_the_weights():
    """Systematic resampling copies each particle about weight × n times and never indexes past the end."""
    import numpy as np

    module = apps.load("particle_filter")
    weights = np.array([0.5, 0.0, 0.25, 0.25])
    chosen = module.systematic_resample(np.repeat(weights, 250) / 250, np.random.default_rng(0))
    assert chosen.min() >= 0 and chosen.max() < 1_000
    assert np.bincount(chosen // 250, minlength=4).tolist() == [500, 0, 250, 250]


def test_mlp_train_is_larger_at_the_large_size():
    """The perceptron has more inputs and hidden units at the large size."""
    module = apps.load("mlp_train")
    small, large = module.make_data("small"), module.make_data("large")
    assert small[0].shape[1] < large[0].shape[1]
    assert small[-1] < large[-1]


def test_min_char_rnn_runs_unchanged_at_the_large_size():
    """The vendored program trains on a text of a thousand characters with no change but the listed rewrites."""
    module = apps.load("min_char_rnn")
    program, text = module.make_data("large")
    assert len(set(text)) == module.LARGE_VOCABULARY
    assert len(set(module.make_data("small")[1])) == module.SMALL_VOCABULARY < module.LARGE_VOCABULARY
    weights = module.run(program, text)
    assert weights[0].shape == (100, module.LARGE_VOCABULARY)


def test_repeats_make_a_round_last_long_enough():
    """A short run is repeated until a round lasts SHORTEST_ROUND_MS; a long one is not."""
    assert A.repeats_for(A.SHORTEST_ROUND_MS * 2) == 1
    assert A.repeats_for(A.SHORTEST_ROUND_MS / 8) == 8
    assert A.repeats_for(0.0) >= 1


def test_parse_perf_counts_the_top_frame_of_each_captured_sample():
    """Every sample of the capture is counted once, by its first frame, and classed by file."""
    profile = P.parse_perf(CAPTURE.read_text(), CAPTURE_PACKAGES)
    assert profile.samples == 24
    by_kind = {kind: round(share * profile.samples) for kind, share in profile.classes.items()}
    assert by_kind == {
        P.PYTHON: 3, P.NUMPY: 3, P.BLAS: 0, P.MATHS: 3, P.HOTKERNELS: 0,
        # libc, the kernel, libgcc_s, the dynamic loader and libstdc++.
        P.SYSTEM: 3 + 3 + 3 + 2 + 1,
        P.OTHER: 3,
    }
    names = {function.name for function in profile.functions}
    assert "amerge_at_<npy::floating_point_type<double, (NPY_TYPES)12>, double, false>"[:P.NAME_WIDTH] in names
    assert P.NO_SYMBOLS in names
    assert all("/" not in function.image for function in profile.functions)


def test_parse_perf_reads_x86_names():
    """Offsets are dropped, C++ names kept whole, files named by file name, and every class is reached."""
    profile = P.parse_perf(X86_SCRIPT, X86_PACKAGES)
    assert profile.samples == 8
    shares = {(f.name, f.image): f.share for f in profile.functions}
    numpy_file = "_multiarray_umath.cpython-312-x86_64-linux-gnu.so"
    assert shares[("DOUBLE_subtract", numpy_file)] == pytest.approx(2 / 8)
    assert (CXX_NAME[:P.NAME_WIDTH], numpy_file) in shares
    assert profile.classes[P.NUMPY] == pytest.approx(3 / 8)
    assert profile.classes[P.BLAS] == pytest.approx(1 / 8)
    assert profile.classes[P.SYSTEM] == pytest.approx(2 / 8)
    assert profile.classes[P.HOTKERNELS] == pytest.approx(1 / 8)
    assert profile.classes[P.MATHS] == pytest.approx(1 / 8)
    assert ("clear_page_erms", "[kernel.kallsyms]") in shares
    assert (P.NO_SYMBOLS, "libc.so.6") in shares


def test_parse_perf_of_nothing():
    """An empty capture gives an empty profile, not a division by zero."""
    profile = P.parse_perf("", {})
    assert profile.functions == ()
    assert sum(profile.classes.values()) == 0


@pytest.mark.parametrize("image, kind", [
    ("libm.so.6", P.MATHS), ("libc.so.6", P.SYSTEM), ("ld-linux-x86-64.so.2", P.SYSTEM),
    ("[kernel.kallsyms]", P.SYSTEM), ("[vdso]", P.SYSTEM), ("libstdc++.so.6.0.33", P.SYSTEM),
    ("libpython3.12.so.1.0", P.PYTHON), ("python3.12", P.PYTHON), ("libopenblas64_p-r0-0cf96a72.3.23.dev.so", P.BLAS),
    ("libmkl_core.so.2", P.BLAS), ("libgomp.so.1", P.OTHER), ("[unknown]", P.OTHER),
])
def test_linux_files_are_classed_as_on_macos(image, kind):
    """The files of a Linux system go to the classes of the macOS profile."""
    assert P.classify(image, {}) == kind


def _linux_with_perf(monkeypatch):
    """Make find_sampler see a Linux system on which perf is installed."""
    monkeypatch.setattr(P.sys, "platform", "linux")
    monkeypatch.setattr(P.shutil, "which", lambda tool: f"/usr/bin/{tool}")


def test_no_perf_is_said(monkeypatch):
    """Without perf the reason names the package to install."""
    monkeypatch.setattr(P.sys, "platform", "linux")
    monkeypatch.setattr(P.shutil, "which", lambda tool: None)
    sampler, reason = P.find_sampler()
    assert sampler is None and "linux-perf" in reason


def test_paranoid_two_samples_user_space_only(monkeypatch):
    """At the kernel's default level a user who is not root samples user space, and the record says so."""
    _linux_with_perf(monkeypatch)
    monkeypatch.setattr(P, "paranoid_level", lambda: 2)
    monkeypatch.setattr(P.os, "geteuid", lambda: 1000)
    tried = []
    monkeypatch.setattr(P, "_perf_allows", lambda event: tried.append(event) or "")
    sampler, reason = P.find_sampler()
    assert tried == [P.PERF_USER_EVENT] and sampler == P.Sampler(P.PERF, P.PERF_USER_EVENT) and reason == ""
    assert "kernel's time is not in the shares" in sampler.description


def test_root_samples_the_kernel_too(monkeypatch):
    """Root samples user space and the kernel at any level."""
    _linux_with_perf(monkeypatch)
    monkeypatch.setattr(P, "paranoid_level", lambda: 2)
    monkeypatch.setattr(P.os, "geteuid", lambda: 0)
    monkeypatch.setattr(P, "_perf_allows", lambda event: "")
    assert P.find_sampler()[0] == P.Sampler(P.PERF, P.PERF_EVENT)


def test_paranoid_three_is_explained(monkeypatch):
    """Where the level forbids sampling, the reason gives the level and the sysctl that allows it."""
    _linux_with_perf(monkeypatch)
    monkeypatch.setattr(P, "paranoid_level", lambda: 3)
    monkeypatch.setattr(P.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(P, "_perf_allows", lambda event: "No permission to enable cpu-clock:u event.")
    sampler, reason = P.find_sampler()
    assert sampler is None
    assert "perf_event_paranoid is 3" in reason and "sysctl kernel.perf_event_paranoid=2" in reason


def test_a_container_that_forbids_perf_is_explained(monkeypatch):
    """Where the level allows sampling and perf still fails, the reason points at the container."""
    _linux_with_perf(monkeypatch)
    monkeypatch.setattr(P, "paranoid_level", lambda: 2)
    monkeypatch.setattr(P.os, "geteuid", lambda: 0)
    monkeypatch.setattr(P, "_perf_allows", lambda event: "No permission to enable cpu-clock event.")
    sampler, reason = P.find_sampler()
    assert sampler is None and "container" in reason


@pytest.mark.skipif(shutil.which(P.PERF) is None, reason="needs perf")
def test_perf_on_this_machine():
    """Where perf exists, find_sampler either finds a sampler or says why not."""
    sampler, reason = P.find_sampler()
    assert (sampler is None) != (reason == "")


def test_unfinished_runs_are_counted(monkeypatch):
    """Runs whose status is anything but completed count as unfinished."""
    statuses = [{"status": "completed"}, {"status": "in_progress"}, {"status": "queued"}]
    answer = subprocess.CompletedProcess([], 0, stdout=json.dumps(statuses), stderr="")
    monkeypatch.setattr(A.subprocess, "run", lambda *args, **kwargs: answer)
    assert A.unfinished_runs() == 2


def test_unfinished_runs_without_gh(monkeypatch):
    """Without `gh`, or when it fails, the count is unknown."""
    def missing(*args, **kwargs):
        """Fail as running a program that is not installed does."""
        raise FileNotFoundError("gh")
    monkeypatch.setattr(A.subprocess, "run", missing)
    assert A.unfinished_runs() is None
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="not logged in")
    monkeypatch.setattr(A.subprocess, "run", lambda *args, **kwargs: failed)
    assert A.unfinished_runs() is None


def test_record_refused_while_ci_runs(monkeypatch, tmp_path):
    """--record refuses before timing anything while CI runs are unfinished, and --allow-busy overrides."""
    monkeypatch.setattr(A, "unfinished_runs", lambda: 3)
    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    measured = []

    def fake_measure(name, size):
        """Stand in for timing an application, and note that it was asked for."""
        measured.append((name, size))
        return _first_timing()

    monkeypatch.setattr(A, "measure", fake_measure)
    runner = CliRunner()
    refused = runner.invoke(cli.app, ["apps", "--only", "mlp_train", "--record"])
    assert refused.exit_code == cli.EXIT_FAILED and measured == [] and not list(tmp_path.iterdir())
    allowed = runner.invoke(cli.app, ["apps", "--only", "mlp_train", "--record", "--allow-busy"])
    assert allowed.exit_code == cli.EXIT_OK, allowed.output
    assert measured == [("mlp_train", "small"), ("mlp_train", "large")]
    recorded = json.loads(next(tmp_path.glob("*-apps.json")).read_text())
    assert recorded["meta"]["unfinished_ci_runs"] == 3
    assert len(recorded["rows"]) == 2


def test_one_size_can_be_chosen(monkeypatch):
    """--size runs one size; an unknown size is refused."""
    measured = []

    def fake_measure(name, size):
        """Stand in for timing an application, and note the size asked for."""
        measured.append(size)
        return _first_timing()

    monkeypatch.setattr(A, "measure", fake_measure)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["apps", "--only", "mlp_train", "--size", "small"]).exit_code == cli.EXIT_OK
    assert measured == ["small"]
    assert runner.invoke(cli.app, ["apps", "--size", "medium"]).exit_code == cli.EXIT_FAILED


def test_first_record_is_read_and_rendered_as_committed(tmp_path):
    """The record written before sizes reads with each application's size, and renders to the committed report."""
    rows = json.loads(FIRST_RECORD.read_text())["rows"]
    timings = [A.from_row(row["timing"]) for row in rows]
    assert {timing.name: timing.size for timing in timings} == A.SINGLE_SIZE
    copy = tmp_path / FIRST_RECORD.name
    shutil.copy(FIRST_RECORD, copy)
    result = CliRunner().invoke(cli.app, ["apps-report", str(copy)])
    assert result.exit_code == cli.EXIT_OK, result.output
    assert copy.with_suffix(".md").read_text() == FIRST_RECORD.with_suffix(".md").read_text()


def test_a_new_record_round_trips():
    """A row written by as_row reads back to the same timing and profile."""
    row = json.loads(FIRST_RECORD.read_text())["rows"][1]
    timing = A.from_row(row["timing"])
    profile = P.from_row(row["profile"], timing.size)
    assert A.from_row(A.as_row(timing)) == timing
    assert P.from_row(P.as_row(profile), "") == profile
