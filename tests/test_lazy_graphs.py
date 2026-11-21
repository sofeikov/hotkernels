"""Tests of the prototype's graph tools: the recording mode, the graph as an object, and the page.

They hold what docs/lazy-recording-design.md and docs/lazy-graph-page-design.md
say. The drawing itself runs in a browser and is not tested here. Run with
`uv run --group dev pytest tests/test_lazy_graphs.py`.
"""
from __future__ import annotations

import json
import sys

import numpy
import pytest

from prototypes.lazy import LazyArray, gather, get_lazy, graph, record, record_graphs, view
from prototypes.lazy.record import BLOCK, CALL, EARLIER, INPUT, LINE, OPERATION, REPEAT, REST, STEP

lazy = get_lazy()

# Steps of the loops in these tests: enough for a repeat, and for one that is too long to list.
FEW_STEPS, MANY_STEPS = 4, 30


def scaled(x: LazyArray, factor: float) -> LazyArray:
    """Multiply and shift an array: two operations on one line, inside a function.

    Args:
        x: A lazy array.
        factor: What to multiply by.

    Returns:
        A pending lazy array.
    """
    return x * factor + 1


def kinds(group: record.Group) -> list[str]:
    """List the kinds of a group's members.

    Args:
        group: A group.

    Returns:
        "op" for an operation and the group's kind for a group, in order.
    """
    return ["op" if isinstance(child, int) else child.kind for child in group.children]


def operations(recorded: record.Graph) -> list[record.GraphNode]:
    """Return a graph's operations.

    Args:
        recorded: A graph.

    Returns:
        Its nodes of kind OPERATION, in order.
    """
    return [node for node in recorded.nodes if node.kind == OPERATION]


def test_one_graph_per_computation_with_what_triggered_it():
    """A gather, a call made at once, a write and what is left pending each give one graph, in order, with their lines."""
    first = sys._getframe().f_lineno + 2
    with record_graphs() as recording:
        x = lazy.arange(6.0)
        y = x * 2 + 1
        y.gather()
        top = numpy.argmax(y - 3)
        z = lazy.zeros(6)
        late = y / 4
        z[0] = 1.0
        left = z + late
    assert [(each.trigger, each.call) for each in recording.graphs] == [
        (graph.GATHER, "gather"), (graph.AT_ONCE, "np.argmax"), (graph.WRITE, "x[index] = v"),
        (graph.PENDING, "still pending when the block ended")]
    assert [each.index for each in recording.graphs] == [0, 1, 2, 3]
    lines = [first + offset for offset in (2, 3, 6)]
    assert [each.where for each in recording.graphs[:3]] == [f"test_lazy_graphs.py:{line}" for line in lines]
    assert recording.graphs[0].code == "y.gather()"
    assert [[node.name for node in operations(each)] for each in recording.graphs] == [
        ["multiply", "add"], ["subtract"], ["divide"], ["add"]]
    assert recording.dropped == 0 and graph.observer is None
    assert type(top) is LazyArray and left._node.pending


def test_a_node_is_plain_data_and_says_where_it_came_from():
    """Operations carry their line, code, operands and arguments as written; inputs say what made them."""
    weights = numpy.arange(3.0)
    line = sys._getframe().f_lineno + 3
    with record_graphs() as recording:
        x = lazy.ones((2, 3), dtype=numpy.float32)
        total = (x * 2.0 @ weights).sum(axis=0)
        index = numpy.argmax(total)
        picked = total + index
        picked.gather()
    made, later = recording.graphs[0], recording.graphs[1]
    by_name = {node.name: node for node in made.nodes}
    assert set(by_name) == {"np.ones", "ndarray", "multiply", "matmul", "sum"}
    assert (by_name["np.ones"].kind, by_name["np.ones"].category, by_name["ndarray"].kind) == (INPUT, record.ARRAY, INPUT)
    assert by_name["ndarray"].id < 0 and by_name["ndarray"].shape == (3,) and by_name["ndarray"].nbytes == 24
    multiply, matmul, total_node = by_name["multiply"], by_name["matmul"], by_name["sum"]
    assert (multiply.dtype, multiply.shape, multiply.nbytes, multiply.category) == ("float32", (2, 3), 24, "elementwise")
    assert multiply.arguments == (by_name["np.ones"].id, "2.0") and multiply.operands == (by_name["np.ones"].id,)
    assert matmul.operands == (multiply.id, by_name["ndarray"].id) and matmul.category == "matmul"
    assert total_node.arguments == (matmul.id, "axis=0") and total_node.category == "reduction"
    assert total_node.where == f"test_lazy_graphs.py:{line}" and total_node.code == "total = (x * 2.0 @ weights).sum(axis=0)"
    assert made.outputs == (total_node.id,) and all(node.state == record.COMPUTED for node in made.nodes)
    read = {node.name: node for node in later.nodes if node.kind != OPERATION}
    assert (read["sum"].kind, read["sum"].graph, read["sum"].operands) == (EARLIER, 0, ())
    assert (read["np.argmax"].kind, read["np.argmax"].category) == (INPUT, graph.AT_ONCE)
    json.dumps(recording.to_dict())


def test_groups_follow_lines_calls_and_loops():
    """Operations of one line are a LINE, of one call a CALL, and alike runs that follow each other a REPEAT."""
    with record_graphs() as recording:
        x = lazy.arange(4.0)
        y = scaled(x, 2.0)
        for _ in range(FEW_STEPS):
            y = y - 0.1 * y
            y = scaled(y, 0.5)
        y.gather()
    root = recording.graphs[0].groups
    assert (root.kind, root.operations, kinds(root)) == (BLOCK, 2 + 4 * FEW_STEPS, [CALL, REPEAT])
    call, loop = root.children
    assert (call.label, call.code, call.operations) == ("scaled()", "y = scaled(x, 2.0)", 2)
    assert kinds(call) == [LINE] and call.children[0].label == "return x * factor + 1"
    assert (loop.label, loop.repeats, loop.operations) == (f"× {FEW_STEPS}", FEW_STEPS, 4 * FEW_STEPS)
    assert kinds(loop) == [STEP] * FEW_STEPS and loop.children[2].label == f"step 3 of {FEW_STEPS}"
    assert kinds(loop.children[0]) == [LINE, CALL] and len({step.signature for step in loop.children}) == 1
    assert loop.lines[1] - loop.lines[0] == 1 and loop.where.startswith("test_lazy_graphs.py:")


def test_a_long_loop_lists_its_ends_and_holds_the_rest():
    """A repeat of many steps shows its first two and its last, with the others in one group."""
    with record_graphs() as recording:
        w = lazy.zeros(3)
        for _ in range(MANY_STEPS):
            w = w - 0.1 * (w * 2)
        w.gather()
    loop = recording.graphs[0].groups.children[0]
    assert (loop.kind, loop.repeats) == (REPEAT, MANY_STEPS)
    assert kinds(loop) == [STEP, STEP, REST, STEP] and loop.code == "w = w - 0.1 * (w * 2)"
    rest = loop.children[2]
    assert rest.label == f"steps 3 to {MANY_STEPS - 1}" and rest.operations == 3 * (MANY_STEPS - 3)
    assert len(rest.children) == MANY_STEPS - 3 and loop.operations == 3 * MANY_STEPS


def test_a_generator_expression_is_a_line_of_its_function():
    """Operations written inside a generator expression belong to the function it stands in, not to a call."""
    with record_graphs() as recording:
        x = lazy.arange(6.0)
        halves = tuple(x[start:start + 3] * 2 for start in (0, 3))
        gather(*halves)
    root = recording.graphs[0].groups
    assert kinds(root) == [LINE] and root.children[0].operations == 4


def test_graphs_that_are_alike_share_a_signature():
    """A loop that gathers at every step gives one graph per step, alike from the second on."""
    with record_graphs() as recording:
        w = lazy.zeros(3)
        for _ in range(FEW_STEPS):
            w = w - 0.1 * (w * 2)
            w.gather()
    signatures = [each.signature for each in recording.graphs]
    assert len(recording.graphs) == FEW_STEPS and len(set(signatures[1:])) == 1 and signatures[0] != signatures[1]
    assert [node.kind for node in recording.graphs[1].nodes if node.kind != OPERATION] == [EARLIER]


def test_recording_changes_no_result_and_keeps_no_array():
    """A recorded program gives the bits it gives unrecorded, and its graphs survive the arrays."""
    data = numpy.random.default_rng(0).standard_normal((50, 4))

    def program(np):
        """Standardise the columns and take a product; a few operations of each kind."""
        x = np.asarray(data)
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        return (x.T @ x).sum()

    expected = program(numpy)
    with record_graphs() as recording:
        got = program(lazy).gather()
    assert type(got) is type(expected) and got.tobytes() == expected.tobytes()
    del got
    assert sum(len(operations(each)) for each in recording.graphs) == 7
    assert recording._stack == [] and recording._base is None


def test_a_failure_is_recorded_as_failed():
    """A node whose computation raises is in the graph with the state "failed"."""
    with record_graphs() as recording, numpy.errstate(over="raise"):
        big = lazy.array([1e308]) * 10
        with pytest.raises(FloatingPointError):
            big.gather()
    assert [node.state for node in operations(recording.graphs[0])] == [record.FAILED]


def test_dropped_work_is_counted_and_only_one_recording_runs():
    """Operations written and dropped without being computed are counted; a second recording is refused."""
    with record_graphs() as recording:
        unused = lazy.arange(3.0) * 2
        del unused
        with pytest.raises(RuntimeError):
            with record_graphs():
                pass
    assert recording.graphs == [] and recording.dropped == 1


def test_the_pending_graph_of_one_array():
    """x.graph() gives what is pending under x as an object and computes nothing; outside a recording too."""
    x = lazy.arange(4.0)
    y = (x * 2).sum()
    pending = y.graph()
    assert (pending.index, pending.trigger) == (record.NO_INDEX, graph.PENDING) and y._node.pending
    assert [(node.kind, node.name, node.state) for node in pending.nodes] == [
        (INPUT, "array", record.COMPUTED), (OPERATION, "multiply", record.PENDING), (OPERATION, "sum", record.PENDING)]
    assert pending.outputs == (pending.nodes[-1].id,) and pending.groups.label == record.PENDING_LABEL
    assert json.loads(pending.to_json())["nodes"][1]["name"] == "multiply"
    y.gather()
    done = y.graph()
    assert [(node.kind, node.state) for node in done.nodes] == [(INPUT, record.COMPUTED)] and done.groups.operations == 0


def test_the_page_holds_the_graphs_and_survives_code_that_ends_a_script(tmp_path):
    """The page is one file: the graphs as JSON, the layout library by URL, and no way for a line of code to break out."""
    target = tmp_path / "graphs.html"
    with record_graphs(target) as recording:
        x = lazy.arange(3.0)
        y = x * 2  # </script><b>
        y.gather()
    text = target.read_text()
    assert text == recording.page() and text.count("</script>") == 3 and view.LAYOUT_LIBRARY_URL in text
    embedded = text.split('<script id="data" type="application/json">')[1].split("</script>")[0]
    data = json.loads(embedded)
    assert data["fold_above"] == record.FOLD_ABOVE and len(data["graphs"]) == 1
    assert data["graphs"][0]["nodes"][1]["code"] == "y = x * 2  # </script><b>"
    assert data["graphs"][0]["groups"]["children"] == [data["graphs"][0]["nodes"][1]["id"]]
    assert recording._repr_html_().startswith("<iframe srcdoc=") and "<b>" not in recording._repr_html_().split("srcdoc=")[1][:200]
