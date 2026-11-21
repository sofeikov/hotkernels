"""The summary of what the layer computed without being asked: which calls, how often, on which lines.

    print(lazy_report())

The log names each line once. This is the other half: every occurrence,
ranked, so that the call a program makes six hundred times at once is at the
top. Calls the layer could not defer come first, since they say what the
deferred set should cover next; then the places where a graph was computed
early, by a write or because a value was needed without `gather()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import graph

# Where this package keeps its templates.
TEMPLATES = Path(__file__).resolve().parent / "templates"
# The template of the summary.
REPORT_TEMPLATE = "report.txt.j2"
# How many lines of code a row names before it says how many more there are.
PLACES_SHOWN = 4


@dataclass(frozen=True)
class Row:
    """One call of the summary."""

    # The call as a reader would name it: "np.dot", "x += y", "float()".
    call: str
    # How many times it made the layer compute.
    count: int
    # The lines it happened on, as ("file.py:12", times), the most frequent first.
    places: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Report:
    """What the layer computed without being asked, ranked."""

    # Calls outside the deferred set, which NumPy made at once.
    eager: tuple[Row, ...]
    # Writes into lazy arrays, each of which computed everything pending first.
    writes: tuple[Row, ...]
    # Values needed without `gather()`.
    needed: tuple[Row, ...]

    def __str__(self) -> str:
        """Return the summary as a table."""
        return render(self)


def lazy_report(reset: bool = False) -> Report:
    """Summarise what the layer has computed without being asked, in this process.

    Args:
        reset: Whether to start counting from nothing afterwards.

    Returns:
        The report. Printing it gives a table.
    """
    with graph.LOCK:
        report = build(graph.SITES)
        if reset:
            graph.SITES.clear()
    return report


def build(sites: Mapping[tuple[str, str, str, int], int]) -> Report:
    """Rank counts kept by line into a report.

    Args:
        sites: (kind, call, file, line) -> times, as `graph.SITES` and a recording keep them.

    Returns:
        The report: one row per call in each of its three parts, the most
        frequent first.
    """
    return Report(_rows(sites, graph.AT_ONCE), _rows(sites, graph.WRITE), _rows(sites, graph.NEEDED))


def render(report: Report) -> str:
    """Write a report as text.

    Args:
        report: The report.

    Returns:
        A table of its three parts; a part with nothing in it says so.
    """
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined,
                              trim_blocks=True, lstrip_blocks=True)
    parts = [("Made at once by NumPy: not in the deferred set", report.eager),
             ("A graph computed early: a write into a lazy array", report.writes),
             ("A graph computed early: a value needed without gather()", report.needed)]
    return environment.get_template(REPORT_TEMPLATE).render(parts=parts, shown=PLACES_SHOWN).strip("\n")


def _rows(sites: Mapping[tuple[str, str, str, int], int], kind: str) -> tuple[Row, ...]:
    """Gather the counts of one kind into rows.

    Args:
        sites: (kind, call, file, line) -> times.
        kind: graph.AT_ONCE, graph.WRITE or graph.NEEDED.

    Returns:
        One row per call of that kind, the most frequent first, each with its
        lines, the most frequent first. Lines are named by the file's name
        without its directory.
    """
    by_call: dict[str, dict[str, int]] = {}
    for (site_kind, call, path, line), times in sites.items():
        if site_kind == kind:
            places = by_call.setdefault(call, {})
            place = f"{os.path.basename(path)}:{line}"
            places[place] = places.get(place, 0) + times
    rows = [Row(call, sum(places.values()), tuple(sorted(places.items(), key=lambda item: -item[1])))
            for call, places in by_call.items()]
    return tuple(sorted(rows, key=lambda row: -row.count))
