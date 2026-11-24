"""What `hotkernels.report()` returns, and how it is printed.

A report is plain data: one row per kernel, one per loop built into the
binding, and a few facts about the installation. Printing it
renders a Jinja template into a table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .hooks import KERNELS, KernelInfo

# File inside the installed package that says how each kernel is implemented
# and whether it is bit-identical to NumPy. Written at build time from the
# kernel specs.
KERNEL_TABLE = "kernels.json"
TEMPLATES = Path(__file__).resolve().parent / "templates"
REPORT_TEMPLATE = "report.txt.j2"

# What the report says in its "Kernel" column for a loop built into the
# binding, which has no entry in the kernel table.
BUILTIN_IMPLEMENTATION = "C"

# What the report says in its "Result" column.
EXACT = "exact"
WITHIN_TOLERANCE = "within tolerance"

# Why a kernel that has a hook is not installed.
NOT_INSTALLED = "hotkernels is not installed"
NEEDS_INEXACT = "not installed: needs install(inexact=True)"
# Why a kernel served through bottleneck is not installed, when that is by design.
LEFT_OUT = "not installed: install(bottleneck=False)"
WAITING = "waiting: bottleneck is not imported yet"


@dataclass(frozen=True)
class KernelReport:
    """One kernel's row in a report, or the row of a loop built into the binding."""

    # Kernel name, as in `kernels/`.
    name: str
    # The NumPy call the kernel stands in for, as a user writes it.
    numpy_call: str
    # Element type of the arrays it handles.
    elements: str
    # "assembly" or "C".
    implementation: str
    # EXACT when the kernel is bit-identical to NumPy, else WITHIN_TOLERANCE.
    result: str
    # Whether calls can reach the kernel now.
    installed: bool
    # Why not, when `installed` is False; empty otherwise.
    note: str
    # Calls the kernel has taken since install().
    calls: int
    # Calls of the kernel's operation that reached a hook and went to NumPy,
    # because no kernel's conditions were met. Counted per operation, so it is
    # set on the first row of an operation and None on the others, and on an
    # operation the binding does not count.
    left_to_numpy: int | None
    # True for a loop built into the binding, which is not one of the kernels
    # the report's first line counts.
    builtin: bool = False
    # True when the binding does not count the calls of the kernel's
    # operation that it leaves to NumPy.
    uncounted: bool = False


@dataclass(frozen=True)
class Report:
    """The state of hotkernels in this process."""

    # The hotkernels release in use, so a pasted report says which one it is.
    version: str
    # Whether install() is in effect.
    installed: bool
    # Why install() did nothing, when it did nothing; None otherwise.
    unsupported: str | None
    # Whether kernels that are not bit-identical to NumPy were asked for.
    inexact: bool
    # The NumPy version in use.
    numpy_version: str
    # One row per kernel, in a fixed order.
    kernels: tuple[KernelReport, ...]
    # One line per hook that could not be installed or removed cleanly.
    problems: tuple[str, ...]

    @property
    def installed_count(self) -> int:
        """Return how many kernels calls can reach now, leaving out the loops built into the binding."""
        return sum(1 for kernel in self.kernels if kernel.installed and not kernel.builtin)

    @property
    def kernel_count(self) -> int:
        """Return how many kernels there are, leaving out the loops built into the binding."""
        return sum(1 for kernel in self.kernels if not kernel.builtin)

    def __str__(self) -> str:
        """Render the report as a table."""
        return render(self)


def load_kernel_table() -> dict[str, dict]:
    """Read the build-time table of kernels.

    Returns:
        Kernel name -> {"implementation": "assembly" or "C", "exact": bool}.
        Empty where the package was built without its extension.
    """
    table = resources.files(__package__).joinpath(KERNEL_TABLE)
    if not table.is_file():
        return {}
    return json.loads(table.read_text())


def _note(info: KernelInfo, exact: bool, installed: bool, inexact: bool, hooks_applied: bool,
          pending: dict[str, str]) -> str:
    """Say why a kernel is not installed.

    Args:
        info: The kernel.
        exact: Whether the kernel is bit-identical to NumPy.
        installed: Whether install() is in effect.
        inexact: Whether inexact kernels were asked for.
        hooks_applied: Whether at least one of the kernel's hooks is in place.
        pending: Hook name -> why it is not applied, for hooks left off by
            design rather than by a failure.

    Returns:
        The reason, or an empty string for a kernel that is installed.
    """
    if info.unbound_reason:
        return info.unbound_reason
    if not installed:
        return NOT_INSTALLED
    if not exact and not inexact:
        return NEEDS_INEXACT
    if not hooks_applied and info.hooks and all(hook in pending for hook in info.hooks):
        return pending[info.hooks[0]]
    if not hooks_applied:
        return "not installed: its hook failed, see below"
    return ""


def build(version: str, installed: bool, unsupported: str | None, inexact: bool, numpy_version: str,
          applied_hooks: frozenset[str], taken: dict[str, int], passed: dict[str, int],
          problems: tuple[str, ...], pending: dict[str, str] | None = None) -> Report:
    """Assemble a report from the installation's state and the binding's counters.

    Args:
        version: The hotkernels release in use.
        installed: Whether install() is in effect.
        unsupported: Why install() did nothing, or None.
        inexact: Whether inexact kernels were asked for.
        numpy_version: The NumPy version in use.
        applied_hooks: Names of the hooks that are in place.
        taken: Kernel name -> calls the kernel took.
        passed: Operation name -> calls its hooks passed on to NumPy. An
            operation the binding does not count is absent.
        problems: One line per hook that failed.
        pending: Hook name -> why it is not applied, for the hooks in
            bottleneck when they are left out or wait for its import.

    Returns:
        The report.
    """
    table = load_kernel_table()
    rows, seen_operations = [], set()
    for info in KERNELS:
        # A loop built into the binding is not in the table: it is C, or
        # NumPy's own loop for the ufunc lane, and bit-identical to NumPy,
        # which `delivery check` holds it to.
        builtin = {"implementation": info.implementation or BUILTIN_IMPLEMENTATION}
        entry = builtin if info.builtin else table.get(info.name, {})
        exact = entry.get("exact", True)
        hooks_applied = any(hook in applied_hooks for hook in info.hooks)
        note = _note(info, exact, installed, inexact, hooks_applied, pending or {})
        first_of_operation = info.operation is not None and info.operation not in seen_operations
        seen_operations.add(info.operation)
        rows.append(KernelReport(
            name=info.name,
            numpy_call=info.numpy_call,
            elements=info.elements,
            implementation=entry.get("implementation", ""),
            result=EXACT if exact else WITHIN_TOLERANCE,
            installed=not note,
            note=note,
            calls=taken.get(info.name, 0),
            left_to_numpy=passed.get(info.operation) if first_of_operation else None,
            builtin=info.builtin,
            uncounted=info.operation is not None and bool(passed) and info.operation not in passed,
        ))
    return Report(version, installed, unsupported, inexact, numpy_version, tuple(rows), problems)


def _thousands(value: int) -> str:
    """Format an integer with thousands separators."""
    return f"{value:,}"


def render(report: Report) -> str:
    """Render a report through its template.

    Jinja is imported here, so that a program that never prints a report
    never loads it.

    Args:
        report: The report to render.

    Returns:
        The table as text, without a trailing newline.
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True)
    env.filters.update(ljust=str.ljust, rjust=str.rjust, thousands=_thousands)
    return env.get_template(REPORT_TEMPLATE).render(report=report).rstrip("\n")
