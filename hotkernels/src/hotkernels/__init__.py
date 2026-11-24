"""hotkernels: faster kernels for NumPy operations, installed with one line.

    import hotkernels
    hotkernels.install()

`install()` points a handful of function pointers inside NumPy at kernels that
were checked against NumPy and timed against it. Existing NumPy code then
runs faster with no other change. A call a kernel does not handle goes to
NumPy exactly as it was written. `uninstall()` puts every pointer back, and
`report()` shows what is installed and how often each kernel ran.

Importing this package changes nothing. Only `install()` does, and it covers
the whole process: every thread and every library that calls NumPy.

It also serves four functions of bottleneck, `nanmean`, `nanstd`, `nanmin` and
`nanmax`, which libraries such as pandas and empyrical call in place of
NumPy's. Those hooks are applied only once bottleneck is imported; hotkernels
never imports it, and `install(bottleneck=False)` leaves them out.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.metadata
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from . import report as _report
from .environment import BINDING_MODULE, unsupported_reason
from .hooks import BOTTLENECK_HOOKS, BOTTLENECK_MODULE, HOOKS, Hook, Probe, describe
from .report import Report

__all__ = ["install", "uninstall", "enabled", "report", "Report", "__version__"]

# The installed release, from the package metadata; meson.build is its source.
__version__ = importlib.metadata.version(__name__)

log = logging.getLogger(__name__)


@dataclass
class _State:
    """What install() has done in this process."""

    # Whether install() is in effect.
    installed: bool = False
    # Whether kernels that are not bit-identical to NumPy were asked for.
    inexact: bool = False
    # Whether the hooks in bottleneck were asked for.
    bottleneck: bool = True
    # Why install() did nothing, when it did nothing.
    unsupported: str | None = None
    # One line per hook that could not be installed or removed cleanly.
    problems: list[str] = field(default_factory=list)


# The one installation state of the process, and the lock that guards it.
_state = _State()
_lock = threading.RLock()


def _load_binding():
    """Return the extension module that switches pointers and counts calls."""
    return importlib.import_module(BINDING_MODULE)


def _active_probes(hook: Hook, inexact: bool) -> tuple[Probe, ...]:
    """Return the probes of a hook that install() runs.

    Args:
        hook: The hook.
        inexact: Whether kernels that are not bit-identical to NumPy were
            asked for. Their probes are left out otherwise.

    Returns:
        The probes, in the hook's order.
    """
    return tuple(probe for probe in hook.probes if inexact or not probe.inexact)


def _answer(probe: Probe) -> tuple | None:
    """Run one probe's calls and describe what they gave.

    Args:
        probe: The probe.

    Returns:
        The description of its answer (`describe`), or None when it raised.
    """
    try:
        return describe(probe.call())
    except Exception:  # A probe that raises has failed, whatever it raised.
        log.debug("probe for %s raised", probe.kernel, exc_info=True)
        return None


def _expected(hook: Hook, inexact: bool) -> tuple[tuple | None, ...]:
    """Run a hook's probes before the hook is applied: the answers it must give.

    Args:
        hook: The hook, not applied.
        inexact: As for `_active_probes`.

    Returns:
        One description per probe that install() runs, in order; None for one
        that raised.
    """
    return tuple(_answer(probe) for probe in _active_probes(hook, inexact))


def _probe_passes(binding, probe: Probe, expected: tuple | None) -> bool:
    """Run one self-test call and say whether its kernel took it and answered as NumPy did.

    Args:
        binding: The extension module.
        probe: The call to make and the kernel that should take it.
        expected: The description of the probe's answer with the hook not
            applied, from `_expected`, or None when that run raised.

    Returns:
        True when the answer is the same as without the hook in every detail
        `describe` keeps and the kernel's call counter moved.
    """
    if expected is None:
        return False
    before = binding.counters()[0][probe.kernel]
    answer = _answer(probe)
    return answer is not None and answer == expected and binding.counters()[0][probe.kernel] > before


def _apply(binding, hook: Hook, inexact: bool, expected: tuple[tuple | None, ...]) -> str | None:
    """Apply one hook and keep it only if every probe passes.

    Args:
        binding: The extension module.
        hook: The hook to apply.
        inexact: Whether kernels that are not bit-identical to NumPy were
            asked for. Their probes are skipped otherwise.
        expected: The answers of the hook's probes before it was applied,
            from `_expected` with the same `inexact`.

    Returns:
        None when the hook is in place, otherwise a line saying why it is not.
        NumPy is as it was in that case.
    """
    try:
        binding.apply(hook.name)
    except Exception as error:  # Any failure here means NumPy is not shaped as expected.
        return f"{hook.name}: could not be applied ({error})"
    for probe, answer in zip(_active_probes(hook, inexact), expected):
        if not _probe_passes(binding, probe, answer):
            binding.remove(hook.name)
            return f"{hook.name}: removed again, its self-test for {probe.kernel} failed"
    return None


def _apply_all(binding, hooks: list[Hook], inexact: bool) -> list[str]:
    """Apply hooks in order, each proved by its probes against the answers given before any of them was applied.

    The probes all run first, with none of `hooks` applied, so each hook is
    compared with NumPy as it was, not with NumPy and the hooks before it.

    Args:
        binding: The extension module.
        hooks: The hooks, none of them applied.
        inexact: As for `_apply`.

    Returns:
        One line per hook that is not in place.
    """
    expected = [_expected(hook, inexact) for hook in hooks]
    problems = (_apply(binding, hook, inexact, answers) for hook, answers in zip(hooks, expected))
    return [problem for problem in problems if problem]


class _BottleneckLoader(importlib.abc.Loader):
    """Loads `bottleneck.reduce` with its own loader, then applies the bottleneck hooks."""

    def __init__(self, loader) -> None:
        """Wrap the loader the import system found.

        Args:
            loader: The loader of `bottleneck.reduce`'s spec.
        """
        self.loader = loader

    def create_module(self, spec):
        """Create the module as its own loader does.

        Args:
            spec: The module's spec.

        Returns:
            The module, or None for the default.
        """
        return self.loader.create_module(spec)

    def exec_module(self, module) -> None:
        """Run the module as its own loader does, then apply the bottleneck hooks.

        Args:
            module: The module, already in sys.modules.
        """
        self.loader.exec_module(module)
        _bottleneck_imported()

    def __getattr__(self, name: str):
        """Answer anything else as the module's own loader does."""
        return getattr(self.loader, name)


class _BottleneckFinder(importlib.abc.MetaPathFinder):
    """Notices the import of `bottleneck.reduce`, so its hooks can be applied then.

    It finds nothing itself. For `bottleneck.reduce` it asks the finders after
    it, and wraps the loader of the spec they return.
    """

    def find_spec(self, fullname: str, path, target=None):
        """Return the spec the other finders give `bottleneck.reduce`, with a wrapped loader.

        Args:
            fullname: The module being imported.
            path: The parent package's search path.
            target: The module being reloaded, or None.

        Returns:
            The spec, or None for any other module, which the finders after
            this one then look for as usual.
        """
        if fullname != BOTTLENECK_MODULE:
            return None
        for finder in sys.meta_path:
            find = getattr(finder, "find_spec", None)
            if finder is self or find is None:
                continue
            spec = find(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is not None:
                spec.loader = _BottleneckLoader(spec.loader)
            return spec
        return None


# The finder that waits for bottleneck, while it is in sys.meta_path.
_finder: _BottleneckFinder | None = None


def _add_finder() -> None:
    """Put the finder at the front of sys.meta_path, once."""
    global _finder
    if _finder is None:
        _finder = _BottleneckFinder()
        sys.meta_path.insert(0, _finder)


def _drop_finder() -> None:
    """Take the finder out of sys.meta_path, if it is there."""
    global _finder
    if _finder is not None:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(_finder)
        _finder = None


def _apply_bottleneck(binding) -> list[str]:
    """Apply the hooks in bottleneck, each proved by its probe against bottleneck's own answer.

    bottleneck must be in sys.modules. A hook whose function is not shaped as
    the binding expects, or whose probe fails, is left off.

    Args:
        binding: The extension module.

    Returns:
        One line per hook that is not in place.
    """
    return _apply_all(binding, [hook for hook in HOOKS if hook.name in BOTTLENECK_HOOKS], False)


def _bottleneck_imported() -> None:
    """Apply the bottleneck hooks when bottleneck is imported after install().

    The counters were reset by install(), so the probes' own calls are taken
    off again. Never raises: an import must not fail because of hotkernels.
    """
    with _lock:
        _drop_finder()
        if not (_state.installed and _state.bottleneck):
            return
        try:
            binding = _load_binding()
            before = binding.counters()[0]
            problems = _apply_bottleneck(binding)
            after = binding.counters()[0]
            for kernel, calls in after.items():
                if calls > before[kernel]:
                    binding.forget(kernel, calls - before[kernel])
        except Exception as error:  # Never let the user's import fail.
            problems = [f"bottleneck hooks: could not be applied ({error})"]
        _state.problems.extend(problems)
        for problem in problems:
            log.warning("hotkernels: %s", problem)


def _remove_all(binding) -> list[str]:
    """Remove every hook that is in place, in reverse order.

    Args:
        binding: The extension module.

    Returns:
        One line per hook whose pointer had been changed by something else
        since install() and was therefore left alone.
    """
    problems = []
    for hook in reversed(HOOKS):
        if binding.applied(hook.name) and not binding.remove(hook.name):
            problems.append(f"{hook.name}: left in place, something else changed it after install()")
    return problems


def install(inexact: bool = False, bottleneck: bool = True) -> None:
    """Make NumPy use the kernels, for the whole process.

    Does nothing on a platform, Python build or NumPy that is not supported,
    and never raises because of one; `report()` then says why. Calling it
    again with the same arguments does nothing.

    Args:
        inexact: Also install the kernels whose results are not bit-identical
            to NumPy's. Today those are the float32 and float64 `cumsum`
            kernels, which regroup the additions: their output stays within a
            declared tolerance but, unlike NumPy's, is not monotonic for
            non-negative input.
        bottleneck: Also serve bottleneck's `nanmean`, `nanstd`, `nanmin` and
            `nanmax`, bit-identical to bottleneck's own results. If bottleneck
            is not imported yet, they are served from when it is; hotkernels
            never imports it.
    """
    global _state
    with _lock:
        if _state.installed and _state.inexact == inexact and _state.bottleneck == bottleneck:
            return
        reason = unsupported_reason()
        if reason:
            _state = _State(unsupported=reason)
            log.info("hotkernels is not installed: %s", reason)
            return
        binding = _load_binding()
        _remove_all(binding)
        _drop_finder()
        binding.set_inexact(inexact)
        problems = _apply_all(binding, [hook for hook in HOOKS if hook.module is None], inexact)
        if bottleneck and BOTTLENECK_MODULE in sys.modules:
            problems += _apply_bottleneck(binding)
        elif bottleneck:
            _add_finder()
        binding.reset_counters()
        _state = _State(installed=True, inexact=inexact, bottleneck=bottleneck, problems=problems)
        for problem in problems:
            log.warning("hotkernels: %s", problem)
        log.info("hotkernels installed: %d of %d hooks", len(HOOKS) - len(problems), len(HOOKS))


def uninstall() -> None:
    """Put every pointer back, so NumPy is as it was before `install()`.

    Does nothing when hotkernels is not installed.
    """
    global _state
    with _lock:
        if not _state.installed:
            return
        binding = _load_binding()
        _drop_finder()
        problems = _remove_all(binding)
        binding.set_inexact(False)
        _state = _State(problems=problems)
        for problem in problems:
            log.warning("hotkernels: %s", problem)
        log.info("hotkernels uninstalled")


@contextlib.contextmanager
def enabled(inexact: bool = False, bottleneck: bool = True) -> Iterator[None]:
    """Install hotkernels for the duration of a `with` block.

    The kernels are installed for the whole process while the block runs, not
    only for the thread that entered it. If hotkernels was already installed,
    the block leaves it installed.

    Args:
        inexact: As for `install()`.
        bottleneck: As for `install()`.

    Returns:
        A context manager that yields nothing.
    """
    with _lock:
        was_installed = _state.installed
        install(inexact, bottleneck)
    try:
        yield
    finally:
        if not was_installed:
            uninstall()


def report() -> Report:
    """Say what is installed and how often each kernel has run.

    Returns:
        A `Report`. Printing it gives a table with one row per kernel: the
        NumPy call it serves, whether it is assembly or C, whether it is
        bit-identical to NumPy, how many calls it took, and how many calls
        reached its hook and were left to NumPy.
    """
    with _lock:
        taken: dict[str, int] = {}
        passed: dict[str, int] = {}
        applied: frozenset[str] = frozenset()
        if _state.installed:
            binding = _load_binding()
            taken, passed = binding.counters()
            applied = frozenset(hook.name for hook in HOOKS if binding.applied(hook.name))
        return _report.build(__version__, _state.installed, _state.unsupported, _state.inexact, np.__version__,
                             applied, taken, passed, tuple(_state.problems), _pending_hooks())


def _pending_hooks() -> dict[str, str]:
    """Say why each hook in bottleneck that is not applied is not, when that is by design.

    Returns:
        Hook name -> the reason, for the hooks left out by
        `install(bottleneck=False)` or waiting for bottleneck to be imported.
        A hook that failed is not here: install() lists it as a problem.
    """
    if not _state.installed:
        return {}
    if not _state.bottleneck:
        return dict.fromkeys(BOTTLENECK_HOOKS, _report.LEFT_OUT)
    if _finder is not None:
        return dict.fromkeys(BOTTLENECK_HOOKS, _report.WAITING)
    return {}
