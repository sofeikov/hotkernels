"""The graph as an object, and a mode that records the graphs a block of code produced.

    with record_graphs() as recording:
        ...                          # NumPy code on lazy arrays
    recording.graphs                 # one Graph per computation, in order
    recording.save("graphs.html")    # the page of view.py

A node forgets its operation and its operands once it is computed, which is
what frees intermediates. So a graph can only be looked at while it is pending
(`x.graph()`), unless something wrote it down as it was made. That is what a
recording does. While its block runs, each node made leaves a record of plain
data, and each time the layer computes, the nodes computed together become one
`Graph`, with what made the layer compute and on which line.

Everything a `Graph` holds is plain data: no array and no live node. Keeping
one keeps no memory alive, and nothing done to it changes the program.

A graph also carries its operations as a tree of groups, for a reader and for
the page's folding: the operations one line wrote, the calls of a function,
and repeated runs of the same lines, which are the steps of a loop.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import linecache
import os
import sys
import weakref
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Any, Iterable

import numpy as np

from . import graph, report, view
from .graph import Node, Trigger

# A group holding more operations than this is drawn folded by the page.
FOLD_ABOVE = 5
# How many times in a row a run of operations must come for it to be a repeat.
MIN_REPEATS = 3
# The longest run, in members of one group, that is looked for as the body of
# a repeat. It bounds the search: a loop body of more statements is not folded.
MAX_PERIOD = 64
# A repeat of up to this many steps lists them all. A longer one lists its
# first HEAD_STEPS and last TAIL_STEPS and puts the rest in one group.
SHOWN_STEPS, HEAD_STEPS, TAIL_STEPS = 12, 2, 1
# The most members a group of left-out steps holds; beyond it, it is split
# into groups of groups, so that unfolding one never shows thousands of nodes.
REST_BLOCK = 50
# A call that wrote more members than MOST_MEMBERS, a function of a hundred
# lines for instance, has them cut into blocks of BLOCK_MEMBERS that follow
# each other, so that it does not open as a hundred boxes at once.
MOST_MEMBERS, BLOCK_MEMBERS = 16, 8
# Longest text of one argument kept in a record, in characters.
ARGUMENT_LENGTH = 60
# Characters of a digest kept as a signature. Twelve hex digits: 48 bits.
SIGNATURE_LENGTH = 12
# Longest label of a group taken from a line of code, in characters.
LABEL_LENGTH = 72
# Code names of frames that are not calls a reader wrote: a generator
# expression or a comprehension runs in a frame of its own, inside the
# function whose line it is on.
INLINE_FRAMES = frozenset({"<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>"})

# The kinds of node in a graph.
OPERATION, INPUT, EARLIER = "operation", "input", "earlier"
# The category of an input that came from outside the layer. The other
# category of an input is graph.AT_ONCE; an operation's is its kind in `rules`.
ARRAY = "array"
# The states of a node in a graph.
PENDING, COMPUTED, FAILED = "pending", "computed", "failed"
# The kinds of group.
BLOCK, CALL, LINE, LINES, REPEAT, STEP, REST = "block", "call", "line", "lines", "repeat", "step", "rest"
# What the root group of a recorded graph and of `x.graph()` is called.
BLOCK_LABEL, PENDING_LABEL = "the recorded block", "pending under this array"
# The index of a graph that is not part of a recording.
NO_INDEX = -1
# How much of the user's code a page carries. WHOLE_FILES: every file that
# wrote an operation, in full, so the page can show the program with its
# graphs beside it. WRITTEN_LINES: only the lines that wrote an operation or
# made the layer compute. NO_SOURCE: no code at all, file names and line
# numbers only, for a page that is passed on.
WHOLE_FILES, WRITTEN_LINES, NO_SOURCE = "files", "lines", "none"
# Indent of `to_json`, in spaces.
JSON_INDENT = 1


@dataclass(frozen=True)
class Frame:
    """One function call on the way from the recorded block to an operation."""

    # A number that tells this call of the function from every other call of it.
    invocation: int
    # The function's qualified name, or "<module>".
    function: str
    # Full path of the file the function is in.
    path: str
    # The line that was running in it.
    line: int


@dataclass(frozen=True)
class GraphNode:
    """One node of a graph, as plain data."""

    # The node's number. Negative for an `ndarray` that is not a node of the
    # layer: one that met a lazy array in an operation.
    id: int
    # OPERATION; INPUT for an array that came from outside this graph and was
    # never an operation of the recording; EARLIER for an operation an earlier
    # graph computed.
    kind: str
    # The operation's name ("add", "matmul"), or for an input what made it:
    # "np.zeros", "ndarray", "np.argmax".
    name: str
    # For an operation its kind in `rules`: "elementwise", "matmul",
    # "reduction" or "view". For an input, ARRAY, or graph.AT_ONCE when it is
    # the result of a call made at once.
    category: str
    # Element type of the result, as NumPy prints it.
    dtype: str
    # Shape of the result.
    shape: tuple[int, ...]
    # Size of the result in bytes.
    nbytes: int
    # File name and line of the user's code that wrote the operation, or where
    # an input entered; empty when not known.
    where: str
    # The text of that line.
    code: str
    # The ids of the nodes it reads. Empty for an input.
    operands: tuple[int, ...]
    # Every argument of the call in the order written: an int is the id of an
    # operand, a string is anything else as text ("2.0", "axis=0").
    arguments: tuple[int | str, ...]
    # PENDING, COMPUTED or FAILED.
    state: str
    # For EARLIER, the index of the graph that computed it; None when it was
    # computed before the recording began, and for every other kind.
    graph: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return the node as built-in types, for JSON."""
        return {"id": self.id, "kind": self.kind, "name": self.name, "category": self.category,
                "dtype": self.dtype, "shape": list(self.shape), "nbytes": self.nbytes, "where": self.where,
                "code": self.code, "operands": list(self.operands), "arguments": list(self.arguments),
                "state": self.state, "graph": self.graph}


@dataclass(frozen=True)
class Group:
    """Some operations of a graph that belong together, for a reader and for folding."""

    # BLOCK for the whole graph; CALL for one call of a function; LINE for the
    # operations one run of one line wrote; LINES for a stretch of a long call,
    # several lines that follow each other; REPEAT for runs that are alike and
    # follow each other, the steps of a loop; STEP for one of them; REST for
    # the steps of a long repeat that are not listed singly.
    kind: str
    # What to call it: the line's code, "softmax()", "× 50", "step 3 of 50".
    label: str
    # Name of the file its lines are in; empty for a BLOCK.
    file: str
    # The first and the last line it covers; (0, 0) when not known.
    lines: tuple[int, int]
    # For a CALL, the text of the line that made the call; for a LINE, and for
    # a REPEAT of one line, the line's own text. Empty otherwise.
    code: str
    # How many operations it holds, in all its groups.
    operations: int
    # For a REPEAT, how many steps it has. 0 otherwise.
    repeats: int
    # The same for two groups that are alike in every operation, shape, type and line.
    signature: str
    # Its members in the order written: an int is the id of an operation.
    children: tuple["Group | int", ...]

    @property
    def where(self) -> str:
        """File name and line or lines, as "regression.py:32-36"; empty when not known."""
        if not self.file:
            return ""
        first, last = self.lines
        return f"{self.file}:{first}" if first == last else f"{self.file}:{first}-{last}"

    def to_dict(self) -> dict[str, Any]:
        """Return the group and everything under it as built-in types, for JSON."""
        return {"kind": self.kind, "label": self.label, "where": self.where, "code": self.code,
                "operations": self.operations, "repeats": self.repeats, "signature": self.signature,
                "children": [child if isinstance(child, int) else child.to_dict() for child in self.children]}


@dataclass(frozen=True)
class Graph:
    """The nodes one computation computed, or the pending nodes under one array."""

    # Its place among the graphs of a recording, from 0; NO_INDEX for `x.graph()`.
    index: int
    # What made the layer compute it: graph.GATHER, AT_ONCE, WRITE or LIMIT;
    # graph.PENDING for nodes that were not computed.
    trigger: str
    # The trigger as a reader would name it: "gather", "np.argmax", "x[index] = v".
    call: str
    # File name and line of the trigger; empty when not known.
    where: str
    # The text of that line.
    code: str
    # Inputs first, in the order first read, then operations in the order written.
    nodes: tuple[GraphNode, ...]
    # Ids of the nodes that were asked for; where nothing in particular was
    # (a write, the limit), the operations nothing else in the graph reads.
    outputs: tuple[int, ...]
    # The operations as a tree of groups.
    groups: Group
    # The same for two graphs that are alike in every node, argument and line.
    signature: str

    def to_dict(self) -> dict[str, Any]:
        """Return the graph as built-in types, for JSON."""
        return {"index": self.index, "trigger": self.trigger, "call": self.call, "where": self.where,
                "code": self.code, "nodes": [node.to_dict() for node in self.nodes],
                "outputs": list(self.outputs), "groups": self.groups.to_dict(), "signature": self.signature}

    def to_json(self) -> str:
        """Return the graph as JSON text."""
        return json.dumps(self.to_dict(), indent=JSON_INDENT)

    def page(self) -> str:
        """Return the graph as one HTML page; see `view.py`. It carries the lines that wrote the graph, not whole files."""
        return view.page(_page_data([self], 0, {}, (), WRITTEN_LINES), "Lazy graph")

    def save(self, path: str | os.PathLike) -> Path:
        """Write the graph's page to a file.

        Args:
            path: Where to write it.

        Returns:
            The path written.
        """
        return view.save(self.page(), path)

    def show(self) -> Path:
        """Write the graph's page to a temporary file and open it in the browser.

        Returns:
            The path of the file.
        """
        return view.show(self.page())

    def _repr_html_(self) -> str:
        """Return the page for a notebook to show under the cell."""
        return view.framed(self.page())


@dataclass(slots=True)
class _Record:
    """What a recording keeps of one node: plain data, written down when the node is made."""

    # The node's number; negative for an `ndarray` that is not a node.
    id: int
    # OPERATION or INPUT.
    kind: str
    # As `GraphNode.name`.
    name: str
    # As `GraphNode.category`.
    category: str
    # As `GraphNode.dtype`.
    dtype: str
    # As `GraphNode.shape`.
    shape: tuple[int, ...]
    # As `GraphNode.nbytes`.
    nbytes: int
    # Full path of the file of the user's code that wrote it; empty when not known.
    path: str
    # The line in that file; 0 when not known.
    line: int
    # As `GraphNode.operands`.
    operands: tuple[int, ...]
    # As `GraphNode.arguments`.
    arguments: tuple[int | str, ...]
    # The calls on the way from the recorded block to the line, outermost
    # first. Empty for a node that was not made while a block was recorded.
    frames: tuple[Frame, ...]
    # The node, held weakly, so the recording can tell at the end whether a
    # pending node is still pending or was dropped. None for an input.
    node: weakref.ref | None
    # PENDING, COMPUTED or FAILED.
    state: str
    # Index of the graph that computed it; None while it has not been computed.
    graph: int | None = None


class Recording:
    """The graphs a block of code produced. Made by `record_graphs()`, used as a `with` block."""

    def __init__(self, path: str | os.PathLike | None = None, source: str = WHOLE_FILES) -> None:
        """Prepare a recording; nothing is recorded until the `with` block is entered.

        Args:
            path: Where to write the page when the block ends, or None to write nothing.
            source: How much of the user's code its pages carry unless told
                otherwise: WHOLE_FILES, WRITTEN_LINES or NO_SOURCE.
        """
        # The graphs, in the order they were computed. Public: this is the result.
        self.graphs: list[Graph] = []
        # How many operations the block wrote and then dropped without computing them.
        self.dropped = 0
        self._path = path
        # How much code a page carries when `page`, `save` or `show` is not told.
        self.source = source
        # How often the layer computed without being asked during the block,
        # by (kind, call, file, line), as `graph.SITES` keeps it for the process.
        self.sites: dict[tuple[str, str, str, int], int] = {}
        # Full paths of the files that wrote an operation or made the layer compute.
        self._files: set[str] = set()
        # Record of every node seen, by id.
        self._records: dict[int, _Record] = {}
        # For each plain `ndarray` met as an operand, by `id()`: the array held
        # weakly, and the id it was given. An `id()` can be used again once an
        # array is gone, which the weak reference tells.
        self._arrays: dict[int, tuple[weakref.ref, int]] = {}
        # Hands out the negative ids of those arrays.
        self._array_ids = itertools.count(-1, -1)
        # Hands out the numbers that tell one call of a function from another.
        self._invocations = itertools.count(1)
        # The user's frames the last node was made under, outermost first,
        # each with its invocation number. Held strongly on purpose: a frame
        # that is still referenced cannot have its address used again, so
        # comparing by identity tells one call of a function from the next.
        self._stack: list[tuple[FrameType, int]] = []
        # The frame of the `with` block. Frames above it are not the block's business.
        self._base: FrameType | None = None

    def __enter__(self) -> "Recording":
        """Start recording.

        Returns:
            The recording itself.

        Raises:
            RuntimeError: When another recording is in effect.
        """
        with graph.LOCK:
            if graph.observer is not None:
                raise RuntimeError("graphs are already being recorded: one recording at a time")
            self._base = sys._getframe(1)
            graph.observer = self
        return self

    def __exit__(self, *exception: Any) -> None:
        """Stop recording, add what is still pending as a last graph, and write the page if a path was given.

        Args:
            *exception: What the block raised, if it did. It is not swallowed.
        """
        with graph.LOCK:
            graph.observer = None
            self._close()
        if self._path is not None:
            self.save(self._path)

    # ---- What the layer tells a recording; see graph.Observer.

    def made(self, node: Node, source: str) -> None:
        """Write down a node that was just made.

        Args:
            node: The node, pending or a leaf.
            source: For a leaf, what made the array.
        """
        with graph.LOCK:
            self._ensure(arg for arg in node.args if isinstance(arg, Node))
            self._records[node.number] = self._record(node, source, self._frames())

    def fell_back(self, kind: str, call: str, origin: graph.Origin) -> None:
        """Count one time the layer computed without being asked.

        Args:
            kind: graph.AT_ONCE, graph.WRITE or graph.NEEDED.
            call: The call as a reader would name it.
            origin: The line of the user's code where it happened.
        """
        with graph.LOCK:
            site = (kind, call, origin.filename, origin.lineno)
            self.sites[site] = self.sites.get(site, 0) + 1
            self._files.add(origin.filename)

    def computing(self, nodes: list[Node]) -> None:
        """Write down any node about to be computed that was made before the block.

        Args:
            nodes: The pending nodes one computation will compute.
        """
        with graph.LOCK:
            self._ensure(nodes)

    def computed(self, numbers: list[int], failed: set[int], roots: list[int], trigger: Trigger) -> None:
        """Turn the nodes one computation took into a graph.

        Args:
            numbers: Numbers of the nodes it computed or failed.
            failed: The numbers among them whose computation raised.
            roots: Numbers of the nodes whose values were asked for.
            trigger: What made the layer compute.
        """
        with graph.LOCK:
            index = len(self.graphs)
            for number in numbers:
                record = self._records[number]
                record.graph, record.state = index, FAILED if number in failed else COMPUTED
            origin = graph.caller()
            self._files.add(origin.filename)
            self.graphs.append(self._graph(index, sorted(numbers), roots, trigger, origin.filename, origin.lineno))

    # ---- The result.

    def to_dict(self) -> dict[str, Any]:
        """Return every graph as built-in types, for JSON, with what was computed unasked and what was dropped."""
        return {"graphs": [each.to_dict() for each in self.graphs], "events": _events(self.sites), "dropped": self.dropped}

    def to_json(self) -> str:
        """Return every graph as JSON text."""
        return json.dumps(self.to_dict(), indent=JSON_INDENT)

    def report(self) -> report.Report:
        """Summarise what the layer computed without being asked during the block.

        Returns:
            The calls made at once, the writes and the values needed, each
            ranked by how often, with their lines. Printing it gives a table.
        """
        return report.build(self.sites)

    def page(self, source: str | None = None) -> str:
        """Return all the graphs as one HTML page; see `view.py`.

        Args:
            source: How much of the user's code the page carries: WHOLE_FILES,
                WRITTEN_LINES or NO_SOURCE. None for what the recording was made with.

        Returns:
            The page's HTML.
        """
        return view.page(_page_data(self.graphs, self.dropped, self.sites, sorted(self._files), source or self.source),
                         "Lazy graphs")

    def save(self, path: str | os.PathLike, source: str | None = None) -> Path:
        """Write the page to a file.

        Args:
            path: Where to write it.
            source: As for `page`.

        Returns:
            The path written.
        """
        return view.save(self.page(source), path)

    def show(self, source: str | None = None) -> Path:
        """Write the page to a temporary file and open it in the browser.

        Args:
            source: As for `page`.

        Returns:
            The path of the file.
        """
        return view.show(self.page(source))

    def _repr_html_(self) -> str:
        """Return the page for a notebook to show under the cell."""
        return view.framed(self.page())

    # ---- Records.

    def pending_graph(self, root: Node) -> Graph:
        """Build the graph of what is pending under one node, without computing it.

        Args:
            root: The node of the array asked about.

        Returns:
            A graph whose trigger is graph.PENDING. For a node that is
            computed already it holds that one node, as an input.
        """
        pending = _pending_under(root)
        self._ensure(pending or [root])
        numbers = [node.number for node in pending]
        trigger = Trigger(graph.PENDING, "pending")
        if numbers:
            return self._graph(NO_INDEX, numbers, [root.number], trigger, "", 0, PENDING_LABEL)
        only = _node(self._records[root.number], INPUT)
        return Graph(NO_INDEX, trigger.kind, trigger.call, "", "", (only,), (only.id,),
                     _Tree({}).group(BLOCK, PENDING_LABEL, []), _digest((only.name, only.dtype, only.shape)))

    def _close(self) -> None:
        """End the recording: what is still pending becomes a last graph, and what was dropped is counted."""
        alive, self.dropped = [], 0
        for record in self._records.values():
            if record.kind != OPERATION or record.graph is not None:
                continue
            node = record.node()
            if node is not None and node.pending:
                alive.append(record.id)
            else:
                self.dropped += 1
        if alive:
            trigger = Trigger(graph.PENDING, "still pending when the block ended")
            path, line = (self._base.f_code.co_filename, self._base.f_lineno) if self._base is not None else ("", 0)
            self.graphs.append(self._graph(len(self.graphs), sorted(alive), [], trigger, path, line))
        # Let go of the user's frames, and with them of their local variables.
        self._stack, self._base = [], None

    def _ensure(self, nodes: Iterable[Node]) -> None:
        """Write down nodes the recording has not seen, and whatever pending work they read.

        These are nodes made before the block began, or outside any recording
        when `x.graph()` asks. No frame is known for them, only the line each
        remembers. Found with a stack, not by recursion, and written in
        ascending number so that operands are known before their consumer.

        Args:
            nodes: Nodes that may be unknown.
        """
        unknown: dict[int, Node] = {}
        stack = list(nodes)
        while stack:
            node = stack.pop()
            if node.number in self._records or node.number in unknown:
                continue
            unknown[node.number] = node
            stack.extend(arg for arg in node.args if isinstance(arg, Node))
        for number in sorted(unknown):
            self._records[number] = self._record(unknown[number], ARRAY, ())

    def _record(self, node: Node, source: str, frames: tuple[Frame, ...]) -> _Record:
        """Write down one node.

        Args:
            node: The node. Its operands must be known already.
            source: For a node that has its value, what made the array.
            frames: The calls the node was made under, or empty when not known.

        Returns:
            An OPERATION record for a pending node; an INPUT record for one
            that has its value, whether it is a leaf or was computed before
            the recording saw it.
        """
        if frames:
            path, line = frames[-1].path, frames[-1].line
        elif node.origin is not None:
            path, line = node.origin.filename, node.origin.lineno
        else:
            path, line = "", 0
        if path:
            self._files.add(path)
        nbytes = int(np.prod(node.shape, dtype=np.int64)) * node.dtype.itemsize
        if not node.pending:
            name = source if node.origin is None else node.name
            return _Record(node.number, INPUT, name, node.kind or ARRAY, str(node.dtype), tuple(node.shape), nbytes,
                           path, line, (), (), frames, None, node.state)
        operands, arguments = self._arguments(node, path, line)
        return _Record(node.number, OPERATION, node.name, node.kind, str(node.dtype), tuple(node.shape), nbytes,
                       path, line, operands, arguments, frames, weakref.ref(node), PENDING)

    def _arguments(self, node: Node, path: str, line: int) -> tuple[tuple[int, ...], tuple[int | str, ...]]:
        """Describe a pending node's arguments.

        Args:
            node: The node.
            path: File of the line that wrote it, for a plain `ndarray` first met here.
            line: That line.

        Returns:
            (the ids of the nodes and arrays it reads, every argument in the
            order written: an id for an operand, text for anything else).
        """
        operands: list[int] = []
        arguments: list[int | str] = []
        for arg in node.args:
            if isinstance(arg, Node):
                number = arg.number
            elif isinstance(arg, np.ndarray):
                number = self._array(arg, path, line)
            else:
                arguments.append(_index_text(arg) if node.name == "getitem" else _text(arg))
                continue
            operands.append(number)
            arguments.append(number)
        arguments.extend(f"{key}={_text(value)}" for key, value in node.kwargs.items())
        return tuple(operands), tuple(arguments)

    def _array(self, array: np.ndarray, path: str, line: int) -> int:
        """Return the id of a plain `ndarray` used as an operand, writing it down the first time.

        Args:
            array: The array. It is not a node: it met a lazy array in an operation.
            path: File of the line where it was met.
            line: That line.

        Returns:
            A negative id, the same for the same array object while it lives.
        """
        known = self._arrays.get(id(array))
        if known is not None and known[0]() is array:
            return known[1]
        number = next(self._array_ids)
        self._arrays[id(array)] = (weakref.ref(array), number)
        self._records[number] = _Record(number, INPUT, "ndarray", ARRAY, str(array.dtype), tuple(array.shape),
                                        int(array.nbytes), path, line, (), (), (), None, COMPUTED)
        return number

    def _frames(self) -> tuple[Frame, ...]:
        """Find the user's calls that are running, from the recorded block down to the current line.

        Returns:
            One `Frame` per function of the user's code on the stack,
            outermost first, starting at the block. Frames of the layer and
            of NumPy are left out, and a generator expression is counted as
            a line of the function it is written in. When the block's frame
            is not on this stack (another thread), the innermost user frame
            alone.
        """
        live: list[FrameType] = []
        frame = sys._getframe(1)
        while frame is not None:
            if not graph.is_internal(frame.f_code.co_filename):
                live.append(frame)
            if frame is self._base:
                break
            frame = frame.f_back
        else:
            live = live[:1]
        live.reverse()
        stack: list[tuple[FrameType, int]] = []
        same = True
        for depth, each in enumerate(live):
            # A frame object that is still the one seen last time is the same
            # call. Once one differs, everything under it is a new call.
            same = same and depth < len(self._stack) and self._stack[depth][0] is each
            stack.append((each, self._stack[depth][1] if same else next(self._invocations)))
        self._stack = stack
        frames: list[Frame] = []
        for each, invocation in stack:
            if frames and each.f_code.co_name in INLINE_FRAMES:
                frames[-1] = replace(frames[-1], line=each.f_lineno)
            else:
                name = each.f_code.co_qualname.replace(".<locals>", "")
                frames.append(Frame(invocation, name, each.f_code.co_filename, each.f_lineno))
        return tuple(frames)

    def _graph(self, index: int, numbers: list[int], roots: list[int], trigger: Trigger,
               path: str, line: int, label: str = BLOCK_LABEL) -> Graph:
        """Build the graph of some operations.

        Args:
            index: The graph's index, or NO_INDEX.
            numbers: Ids of its operations, ascending.
            roots: Ids of the nodes that were asked for.
            trigger: What made the layer compute them.
            path: File of the trigger's line, or empty.
            line: The trigger's line.
            label: What to call the root group.

        Returns:
            The graph: the operations, and before them every node they read
            that is not among them.
        """
        inside = set(numbers)
        records = [self._records[number] for number in numbers]
        read = dict.fromkeys(operand for record in records for operand in record.operands)
        outside = [self._records[operand] for operand in read if operand not in inside]
        nodes = tuple([_node(record, EARLIER if record.kind == OPERATION else INPUT) for record in outside]
                      + [_node(record, OPERATION) for record in records])
        asked = [number for number in dict.fromkeys(roots) if number in inside]
        outputs = tuple(asked or [number for number in numbers if number not in read])
        groups = _Tree({record.id: record for record in records}).build(records, label)
        where, code = _place(path, line)
        return Graph(index, trigger.kind, trigger.call, where, code, nodes, outputs, groups,
                     _graph_signature(trigger, where, nodes))


def record_graphs(path: str | os.PathLike | None = None, source: str = WHOLE_FILES) -> Recording:
    """Record the graphs a block of code produces.

        with record_graphs() as recording:
            ...
        recording.show()

    Args:
        path: Where to write the page when the block ends. None writes nothing.
        source: How much of the user's code a page of the recording carries:
            WHOLE_FILES ("files"), WRITTEN_LINES ("lines") or NO_SOURCE ("none").

    Returns:
        A recording, to be used as a `with` block. Its `graphs` are filled as
        the block runs: one per computation, and a last one for what is
        still pending when the block ends.
    """
    return Recording(path, source)


def graph_of(root: Node) -> Graph:
    """Return the pending graph under one node as an object. Computes nothing.

    Args:
        root: The node of the array asked about.

    Returns:
        A graph whose trigger is graph.PENDING. While a block is being
        recorded its nodes carry the calls they were made under; otherwise
        only the line each was written at.
    """
    with graph.LOCK:
        recording = graph.observer if isinstance(graph.observer, Recording) else Recording()
        return recording.pending_graph(root)


def _pending_under(root: Node) -> list[Node]:
    """Collect the pending nodes one node depends on, itself included.

    Args:
        root: The node asked about.

    Returns:
        The pending nodes in ascending number. Found with a stack: the graph
        of a loop is as deep as the loop is long.
    """
    found: dict[int, Node] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.pending and node.number not in found:
            found[node.number] = node
            stack.extend(arg for arg in node.args if isinstance(arg, Node))
    return [found[number] for number in sorted(found)]


def _node(record: _Record, kind: str) -> GraphNode:
    """Turn a record into a node of one graph.

    Args:
        record: The record.
        kind: OPERATION when the graph computes it; EARLIER or INPUT when the
            graph only reads it.

    Returns:
        The node. One that is only read carries no operands: what it was
        computed from belongs to the graph that computed it.
    """
    where, code = _place(record.path, record.line)
    inside = kind == OPERATION
    return GraphNode(record.id, kind, record.name, record.category, record.dtype, record.shape, record.nbytes,
                     where, code, record.operands if inside else (), record.arguments if inside else (),
                     record.state, record.graph if kind == EARLIER else None)


def _place(path: str, line: int) -> tuple[str, str]:
    """Name a line of the user's code.

    Args:
        path: Full path of the file, or empty when not known.
        line: The line.

    Returns:
        ("file.py:12", the text of the line). The file's name alone, without
        its directory: a page may be passed on, and a path says where on a
        machine the code lives. Two empty strings when the file is not known.
    """
    if not path:
        return "", ""
    return f"{os.path.basename(path)}:{line}", linecache.getline(path, line).strip()


def _text(value: Any) -> str:
    """Write an argument that is not an array as text.

    Args:
        value: A Python number, a NumPy scalar, an axis, a shape, anything.

    Returns:
        Its `repr`, cut to ARGUMENT_LENGTH characters.
    """
    text = repr(value)
    return text if len(text) <= ARGUMENT_LENGTH else text[:ARGUMENT_LENGTH - 1] + "…"


def _index_text(index: Any) -> str:
    """Write an index the way it is written between brackets.

    Args:
        index: What was between the brackets: an int, a slice, None, `...`, or a tuple of those.

    Returns:
        "1:3, None, 0" for `(slice(1, 3), None, 0)`.
    """
    return ", ".join(_index_item(item) for item in (index if isinstance(index, tuple) else (index,)))


def _index_item(item: Any) -> str:
    """Write one item of an index.

    Args:
        item: An int, a slice, None or `...`.

    Returns:
        "1:3" for a slice, "..." for an ellipsis, the `repr` otherwise.
    """
    if item is Ellipsis:
        return "..."
    if not isinstance(item, slice):
        return repr(item)
    start, stop = ("" if bound is None else repr(bound) for bound in (item.start, item.stop))
    return f"{start}:{stop}" if item.step is None else f"{start}:{stop}:{item.step!r}"


def _digest(parts: Any) -> str:
    """Return a short, stable name for some data: the same data gives the same name in every run.

    Args:
        parts: Tuples, strings and numbers.

    Returns:
        SIGNATURE_LENGTH hex digits of a digest of its `repr`.
    """
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:SIGNATURE_LENGTH]


def _graph_signature(trigger: Trigger, where: str, nodes: tuple[GraphNode, ...]) -> str:
    """Name a graph so that two graphs alike in every respect but their node numbers get the same name.

    Args:
        trigger: What made the layer compute it.
        where: The trigger's line.
        nodes: Its nodes.

    Returns:
        A signature. Node ids are replaced by positions in the graph, so the
        graph of one step of a loop has the signature of the next.
    """
    position = {node.id: place for place, node in enumerate(nodes)}
    described = [(node.kind, node.name, node.dtype, node.shape, node.where,
                  tuple(position[arg] if isinstance(arg, int) else arg for arg in node.arguments)) for node in nodes]
    return _digest((trigger.kind, trigger.call, where, described))


class _Tree:
    """Builds the tree of groups of one graph's operations."""

    def __init__(self, records: dict[int, _Record]) -> None:
        """Prepare to group some operations.

        Args:
            records: The graph's operations, by id.
        """
        self._records = records
        # For each operation, the calls it was made under with the block's
        # own frame first; see `_path`.
        self._paths = {number: _path(record) for number, record in records.items()}
        # For each operation, a signature of what a reader sees of it.
        self._signatures = {number: _digest((record.name, record.dtype, record.shape, record.path, record.line))
                            for number, record in records.items()}

    def build(self, records: list[_Record], label: str) -> Group:
        """Group a graph's operations.

        Args:
            records: The operations, ascending.
            label: What to call the root.

        Returns:
            The root group.
        """
        return self.group(BLOCK, label, self._members(records, 0))

    def group(self, kind: str, label: str, children: list[Group | int], file: str = "",
              lines: tuple[int, int] = (0, 0), code: str = "", repeats: int = 0) -> Group:
        """Make one group, counting its operations and working out its signature.

        Args:
            kind: The kind of group.
            label: What to call it.
            children: Its members.
            file: Name of the file its lines are in.
            lines: The first and last line it covers.
            code: The text of its line, for a LINE or a CALL.
            repeats: The number of steps, for a REPEAT.

        Returns:
            The group. Two groups have one signature when their kind, their
            lines and their members' signatures are the same; a CALL's label,
            the function's name, counts too. The label of a STEP does not.
        """
        operations = sum(1 if isinstance(child, int) else child.operations for child in children)
        named = label if kind == CALL else ""
        signature = _digest((kind, named, file, lines, repeats, [self._signature(child) for child in children]))
        return Group(kind, label, file, lines, code, operations, repeats, signature, tuple(children))

    def _members(self, records: list[_Record], depth: int) -> list[Group | int]:
        """Group the operations one call of a function wrote, or the block itself.

        Args:
            records: The operations made under that call, ascending.
            depth: How many calls down from the block the call is.

        Returns:
            The call's members in the order written: an operation written on
            a line by itself, a LINE group for several from one run of a
            line, a CALL group for the operations a function called from
            here wrote; and REPEAT groups where runs of those are alike.
        """
        runs: list[tuple[tuple[int, int | None], list[_Record]]] = []
        for record in records:
            path = self._paths[record.id]
            # What tells one run from the next: the line running in this call,
            # and which call was made from it, if the operation is in one.
            key = (path[depth].line, path[depth + 1].invocation if len(path) > depth + 1 else None)
            if runs and runs[-1][0] == key:
                runs[-1][1].append(record)
            else:
                runs.append((key, [record]))
        members: list[Group | int] = []
        for (line, invocation), run in runs:
            here = self._paths[run[0].id][depth]
            file, code = os.path.basename(here.path), linecache.getline(here.path, line).strip()
            if invocation is not None:
                function = self._paths[run[0].id][depth + 1].function
                members.append(self.group(CALL, f"{function}()", self._members(run, depth + 1), file, (line, line), code))
            elif len(run) == 1:
                members.append(run[0].id)
            else:
                ids: list[Group | int] = [record.id for record in run]
                inner = self._repeats(ids)
                if len(inner) == 1 and isinstance(inner[0], Group):
                    # The line is the body of a loop and nothing else: the
                    # repeat stands for it, with the line's text.
                    members.append(replace(inner[0], code=code))
                    continue
                label = code if len(code) <= LABEL_LENGTH else code[:LABEL_LENGTH - 1] + "…"
                members.append(self.group(LINE, label or f"{file}:{line}", inner, file, (line, line), code))
        return self._stretches(self._repeats(members))

    def _stretches(self, members: list[Group | int]) -> list[Group | int]:
        """Cut a long list of members into LINES groups of members that follow each other.

        Args:
            members: The members of one call, in order.

        Returns:
            The members as they are when there are at most MOST_MEMBERS.
            Otherwise blocks of BLOCK_MEMBERS, and blocks of blocks when that
            is still too many, each named by the lines it covers.
        """
        if len(members) <= MOST_MEMBERS:
            return members
        size = BLOCK_MEMBERS
        while len(members) > size * MOST_MEMBERS:
            size *= BLOCK_MEMBERS
        stretches: list[Group | int] = []
        for start in range(0, len(members), size):
            stretch = members[start:start + size]
            if len(stretch) == 1:
                stretches.append(stretch[0])
                continue
            file, lines = self._span(stretch)
            stretches.append(self.group(LINES, f"lines {lines[0]} to {lines[1]}", self._stretches(stretch), file, lines))
        return stretches

    def _repeats(self, members: list[Group | int]) -> list[Group | int]:
        """Replace runs of members that are alike and follow each other by REPEAT groups.

        Args:
            members: The members of one group, in order.

        Returns:
            The members, with each repeat as one group. Greedy from the left:
            at each member the repeat that covers the most members is taken.
        """
        signatures = [self._signature(member) for member in members]
        found: list[Group | int] = []
        start = 0
        while start < len(members):
            repeats, period = _longest_repeat(signatures, start)
            if repeats < MIN_REPEATS:
                found.append(members[start])
                start += 1
                continue
            steps = [members[start + step * period:start + (step + 1) * period] for step in range(repeats)]
            found.append(self._repeat(steps))
            start += repeats * period
        return found

    def _repeat(self, steps: list[list[Group | int]]) -> Group:
        """Make the REPEAT group of some steps that are alike.

        Args:
            steps: The members of each step.

        Returns:
            The group. A step of one member is that member; a step of several
            is a STEP group. A long repeat lists its first and last steps and
            holds the rest in one REST group.
        """
        count = len(steps)
        file, lines = self._span(steps[0])
        listed: list[Group | int] = []
        for place, step in enumerate(steps):
            if len(step) == 1:
                listed.append(step[0])
            else:
                listed.append(self.group(STEP, f"step {place + 1} of {count}", step, *self._span(step)))
        if count > SHOWN_STEPS:
            rest = self._rest(listed[HEAD_STEPS:count - TAIL_STEPS], HEAD_STEPS + 1)
            listed = listed[:HEAD_STEPS] + [rest] + listed[count - TAIL_STEPS:]
        return self.group(REPEAT, f"× {count}", listed, file, lines, repeats=count)

    def _rest(self, steps: list[Group | int], first: int) -> Group:
        """Make the REST group of the steps a long repeat does not list singly.

        Args:
            steps: Those steps.
            first: The number of the first of them within the repeat, from 1.

        Returns:
            A group of at most REST_BLOCK members: the steps themselves, or
            REST groups of them when there are more.
        """
        label = f"steps {first} to {first + len(steps) - 1}"
        if len(steps) <= REST_BLOCK:
            return self.group(REST, label, steps, *self._span(steps))
        size = REST_BLOCK
        while len(steps) > size * REST_BLOCK:
            size *= REST_BLOCK
        blocks: list[Group | int] = [self._rest(steps[start:start + size], first + start)
                                     for start in range(0, len(steps), size)]
        return self.group(REST, label, blocks, *self._span(steps))

    def _signature(self, member: Group | int) -> str:
        """Return the signature of a member: an operation's or a group's.

        Args:
            member: The id of an operation, or a group.

        Returns:
            The signature.
        """
        return self._signatures[member] if isinstance(member, int) else member.signature

    def _span(self, members: list[Group | int]) -> tuple[str, tuple[int, int]]:
        """Find the file and the lines some members cover.

        Args:
            members: Operations and groups.

        Returns:
            (the file's name, (first line, last line)), over the members that
            are in the same file as the first. ("", (0, 0)) for no members.
        """
        places: list[tuple[str, int, int]] = []
        for member in members:
            if isinstance(member, int):
                record = self._records[member]
                places.append((os.path.basename(record.path), record.line, record.line))
            else:
                places.append((member.file, *member.lines))
        if not places:
            return "", (0, 0)
        file = places[0][0]
        same = [(first, last) for name, first, last in places if name == file]
        return file, (min(first for first, _ in same), max(last for _, last in same))


def _path(record: _Record) -> tuple[Frame, ...]:
    """Return the calls an operation was made under, with one frame for the block first.

    Args:
        record: An operation's record.

    Returns:
        The record's frames, or one frame made of its line when none are
        known. The first frame's invocation is set to 0 in every path, so that
        all the operations of a graph hang under one root whatever was known
        about the block's own frame.
    """
    frames = record.frames or (Frame(0, "", record.path, record.line),)
    return (replace(frames[0], invocation=0),) + frames[1:]


def _longest_repeat(signatures: list[str], start: int) -> tuple[int, int]:
    """Find the repeat starting at one member that covers the most members.

    Args:
        signatures: The signatures of a group's members, in order.
        start: The member to start at.

    Returns:
        (how many times the run comes, how many members the run has). (1, 1)
        when nothing at `start` comes MIN_REPEATS times in a row.
    """
    best = (1, 1)
    longest = min(MAX_PERIOD, (len(signatures) - start) // MIN_REPEATS)
    for period in range(1, longest + 1):
        repeats = 1
        while _same_run(signatures, start, start + repeats * period, period):
            repeats += 1
        if repeats >= MIN_REPEATS and repeats * period > best[0] * best[1]:
            best = (repeats, period)
    return best


def _same_run(signatures: list[str], first: int, second: int, period: int) -> bool:
    """Say whether two runs of members are alike.

    Args:
        signatures: The signatures of a group's members.
        first: Where one run starts.
        second: Where the other starts.
        period: How many members each has.

    Returns:
        True when the second run lies within the list and matches the first member by member.
    """
    if second + period > len(signatures):
        return False
    return all(signatures[first + offset] == signatures[second + offset] for offset in range(period))


def _events(sites: dict[tuple[str, str, str, int], int]) -> list[dict[str, Any]]:
    """Turn the counts of what was computed unasked into rows, the most frequent first.

    Args:
        sites: (kind, call, file, line) -> times.

    Returns:
        One dict per line and call: kind, call, where, the line's text, and the count.
    """
    rows = []
    for (kind, call, path, line), count in sorted(sites.items(), key=lambda item: -item[1]):
        where, code = _place(path, line)
        rows.append({"kind": kind, "call": call, "where": where, "code": code, "count": count})
    return rows


def _page_data(graphs: list[Graph], dropped: int, sites: dict[tuple[str, str, str, int], int],
               files: Iterable[str], source: str) -> dict[str, Any]:
    """Gather what the page needs.

    Args:
        graphs: The graphs to show.
        dropped: How many operations were written and dropped without being computed.
        sites: (kind, call, file, line) -> times the layer computed unasked.
        files: Full paths of the files that wrote an operation or made the layer compute.
        source: WHOLE_FILES, WRITTEN_LINES or NO_SOURCE.

    Returns:
        The graphs and events as built-in types, the constant the page folds
        by, and `sources`: per file, the text of its lines by number. With
        WRITTEN_LINES only the lines that did something are there; with
        NO_SOURCE none are, and every text of a line is removed from the
        graphs as well.
    """
    data = {"graphs": [each.to_dict() for each in graphs], "events": _events(sites), "dropped": dropped,
            "fold_above": FOLD_ABOVE, "sources": []}
    if source == NO_SOURCE:
        _remove_code(data)
        return data
    texts = _written_lines(data)
    if source == WHOLE_FILES:
        for path in files:
            lines = linecache.getlines(path)
            if lines:
                texts[os.path.basename(path)] = {number: line.rstrip("\n") for number, line in enumerate(lines, start=1)}
    data["sources"] = [{"file": name, "text": {str(number): line for number, line in sorted(lines.items())}}
                       for name, lines in sorted(texts.items())]
    return data


def _written_lines(data: dict[str, Any]) -> dict[str, dict[int, str]]:
    """Collect the lines of code a page's graphs and events already carry.

    Args:
        data: The page's data.

    Returns:
        File name -> line number -> the line's text, for every line that
        wrote an operation, triggered a computation or made the layer compute.
    """
    texts: dict[str, dict[int, str]] = {}
    carried = [node for each in data["graphs"] for node in each["nodes"]] + data["graphs"] + data["events"]
    for item in carried:
        name, _, number = item["where"].rpartition(":")
        if name:
            texts.setdefault(name, {})[int(number)] = item["code"]
    return texts


def _remove_code(data: dict[str, Any]) -> None:
    """Take every text of a line of code out of a page's data, in place.

    Args:
        data: The page's data. Afterwards it names files and lines and holds
            no code: a LINE group is called by its place instead of its text.
    """
    for item in [node for each in data["graphs"] for node in each["nodes"]] + data["graphs"] + data["events"]:
        item["code"] = ""
    stack = [each["groups"] for each in data["graphs"]]
    while stack:
        group = stack.pop()
        group["code"] = ""
        if group["kind"] == LINE:
            group["label"] = group["where"]
        stack.extend(child for child in group["children"] if isinstance(child, dict))
