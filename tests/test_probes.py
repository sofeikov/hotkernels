"""Tests of the self-test `install()` runs: a hook stays only if its probes answer as NumPy did, in every detail.

Each probe is run once with its hook not applied and once with it applied,
and the two answers are compared with `hotkernels.hooks.describe`. These
tests stand in for a NumPy whose answer differs from the kernel's in one
detail, the Python type of a result, its element type or the sign of a zero,
by changing the answer recorded before the hook was applied. They need a
platform hotkernels installs on, and are skipped elsewhere.
Run with `uv run --group dev pytest tests`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import hotkernels
from hotkernels import hooks
from hotkernels.environment import unsupported_reason

pytestmark = pytest.mark.skipif(unsupported_reason() is not None, reason="hotkernels does not install here")

# The hook whose probes the end-to-end test makes fail: np.count_nonzero,
# whose result was a Python int before NumPy 2.3 and np.intp after.
COUNT_NONZERO_HOOK = "function.count_nonzero"


def _as_python_int(answer: object) -> object:
    """Return `np.count_nonzero`'s answer as NumPy before 2.3 gave it: a Python int.

    Args:
        answer: The answer of a count_nonzero probe, an np.intp.

    Returns:
        The same count as a Python int.
    """
    return int(answer)


def _zeros_positive(answer: np.ndarray) -> np.ndarray:
    """Return a clip probe's answer as NumPy 2.0 gave it: every zero +0.0.

    Args:
        answer: The clipped array.

    Returns:
        A copy with each zero, of either sign, replaced by +0.0.
    """
    result = answer.copy()
    result[result == 0] = 0.0
    return result


def _as_int32(answer: np.ndarray) -> np.ndarray:
    """Return an index array as int32 rather than np.intp: the same values, another element type.

    Args:
        answer: The answer of an argsort probe.

    Returns:
        The array cast to int32.
    """
    return answer.astype(np.int32)


@pytest.fixture
def binding():
    """Give the extension module with no hook applied, and leave none applied afterwards.

    Yields:
        The extension module.
    """
    hotkernels.uninstall()
    module = hotkernels._load_binding()
    yield module
    hotkernels.uninstall()
    for hook in reversed(hooks.HOOKS):
        if module.applied(hook.name):
            module.remove(hook.name)


def _hook(name: str) -> hooks.Hook:
    """Return the hook of that name.

    Args:
        name: A hook name, as in `hooks.HOOKS`.

    Returns:
        The hook.
    """
    return next(hook for hook in hooks.HOOKS if hook.name == name)


@pytest.mark.parametrize(("left", "right"), [
    (1, np.intp(1)),
    (0.0, -0.0),
    (np.float64(0.0), np.float64(-0.0)),
    (np.array([0.0, 1.0]), np.array([-0.0, 1.0])),
    (np.zeros(3, np.float32), np.zeros(3, np.float64)),
    (np.zeros((2, 3)), np.zeros((3, 2))),
    (np.zeros((2, 3)), np.asfortranarray(np.zeros((2, 3)))),
    (np.zeros(3), np.zeros(3).view(np.recarray)),
    ((np.zeros(3),), [np.zeros(3)]),
])
def test_describe_tells_apart(left: object, right: object) -> None:
    """Two answers a caller could tell apart have different descriptions."""
    assert hooks.describe(left) != hooks.describe(right)


def test_describe_compares_nan_by_bits() -> None:
    """Equal answers have equal descriptions, NaN included, which `==` alone would call unequal."""
    assert hooks.describe(np.array([np.nan, -0.0])) == hooks.describe(np.array([np.nan, -0.0]))


@pytest.mark.parametrize("name", ["function.count_nonzero", "loop.clip.f64", "method.argsort"])
def test_hook_kept_when_answers_match(binding, name: str) -> None:
    """A hook whose probes answer as NumPy did before it was applied stays in place."""
    hook = _hook(name)
    expected = hotkernels._expected(hook, False)
    assert hotkernels._apply(binding, hook, False, expected) is None
    assert binding.applied(name)


@pytest.mark.parametrize(("name", "change"), [
    ("function.count_nonzero", _as_python_int),
    ("loop.clip.f64", _zeros_positive),
    ("method.argsort", _as_int32),
])
def test_hook_removed_when_one_detail_differs(binding, name: str, change: Callable[[object], object]) -> None:
    """A hook whose first probe's answer differs from NumPy's in one detail only is removed again."""
    hook = _hook(name)
    expected = list(hotkernels._expected(hook, False))
    plain = hook.probes[0].call()
    changed = change(plain)
    # The change is one a comparison of values alone would not see.
    assert bool(np.all(np.asarray(changed) == np.asarray(plain)))
    expected[0] = hooks.describe(changed)
    problem = hotkernels._apply(binding, hook, False, tuple(expected))
    assert problem == f"{name}: removed again, its self-test for {hook.probes[0].kernel} failed"
    assert not binding.applied(name)


def test_install_leaves_the_hook_off(binding, monkeypatch) -> None:
    """install() leaves off a hook whose NumPy answers with a Python int, and keeps every other."""
    real = hotkernels._expected

    def older_numpy(hook: hooks.Hook, inexact: bool) -> tuple:
        """Answer as `_expected` does, but np.count_nonzero's probes as NumPy before 2.3 did.

        Args:
            hook: The hook, not applied.
            inexact: As for `_expected`.

        Returns:
            The descriptions of the probes' answers.
        """
        if hook.name != COUNT_NONZERO_HOOK:
            return real(hook, inexact)
        return tuple(hooks.describe(_as_python_int(probe.call())) for probe in hook.probes)

    monkeypatch.setattr(hotkernels, "_expected", older_numpy)
    hotkernels.install(bottleneck=False)
    failed = f"{COUNT_NONZERO_HOOK}: removed again, its self-test for count_nonzero_i64 failed"
    assert hotkernels._state.problems == [failed]
    assert not binding.applied(COUNT_NONZERO_HOOK)
    assert binding.applied("function.sum")
    assert type(np.count_nonzero(np.ones(4))) is np.intp
