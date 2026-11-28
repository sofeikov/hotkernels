"""Tests of the lazy layer's prototype, prototypes/lazy: each rule of docs/lazy-design.md, held to NumPy.

Every result is compared with what the same lines give with the real `numpy`
module: type, element type, shape and bits. Run with
`uv run --group dev pytest tests/test_lazy_prototype.py`.
"""
from __future__ import annotations

import gc
import logging
import sys
import warnings

import numpy
import pytest

from prototypes.lazy import LazyArray, NotGathered, gather, get_lazy, graph, lazy_report, record_graphs, rules
from prototypes.lazy.programs import attention, regression

# Seed of every random array in these tests.
SEED = 0
# Examples, features and gradient steps of the regression.
EXAMPLES, FEATURES, STEPS = 2_000, 12, 30
# Tokens, width and heads of the transformer block. The feed-forward layer is four times as wide.
TOKENS, WIDTH, HEADS, WIDENING = 16, 24, 4, 4
# Players, and figures kept for each, of the roster that a Python loop reads element by element.
ROSTER, FIGURES = 22, 3

lazy = get_lazy()
# A stand-in whose arrays refuse to give their values without gather().
strict = get_lazy(strict=True)


def same(expected, got) -> bool:
    """Say whether two results are the same in type, element type, shape and bits.

    Args:
        expected: What eager NumPy returned.
        got: What `gather()` returned.

    Returns:
        True when nothing distinguishes them.
    """
    return (type(expected) is type(got) and expected.dtype == got.dtype and expected.shape == got.shape
            and expected.tobytes() == got.tobytes())


def block_parameters(rng: numpy.random.Generator) -> dict:
    """Draw float32 weights for one transformer block.

    Args:
        rng: The generator to draw from.

    Returns:
        Parameter name -> array, in the shapes `attention.block` expects.
    """
    shapes = {"gain_1": (WIDTH,), "bias_1": (WIDTH,), "gain_2": (WIDTH,), "bias_2": (WIDTH,),
              "w_qkv": (WIDTH, 3 * WIDTH), "b_qkv": (3 * WIDTH,), "w_out": (WIDTH, WIDTH), "b_out": (WIDTH,),
              "w_up": (WIDTH, WIDENING * WIDTH), "b_up": (WIDENING * WIDTH,),
              "w_down": (WIDENING * WIDTH, WIDTH), "b_down": (WIDTH,)}
    return {name: rng.standard_normal(shape).astype(numpy.float32) for name, shape in shapes.items()}


def raised(call) -> BaseException:
    """Make a call that must raise, and return what it raised.

    Args:
        call: A function of no arguments.

    Returns:
        The exception.
    """
    with pytest.raises(Exception) as caught:
        call()
    return caught.value


def test_a_lazy_array_says_what_it_is():
    """The type is LazyArray, the element type is NumPy's own, and printing computes nothing."""
    x = lazy.arange(6.0)
    y = x * 2 + 1
    assert type(x) is LazyArray and str(type(x)) == "<class 'prototypes.lazy.LazyArray'>"
    assert y.dtype is numpy.dtype("float64") and y.shape == (6,) and len(y) == 6 and y.ndim == 1 and y.size == 6
    assert repr(y) == str(y) == f"{y}" == "LazyArray(float64, shape=(6,), pending)"
    assert same(numpy.arange(6.0) * 2 + 1, y.gather())
    assert repr(y) == "LazyArray(float64, shape=(6,), computed)"
    assert type(y.sum()) is LazyArray and type(y.sum().gather()) is numpy.float64
    assert not isinstance(x, numpy.ndarray)


def test_regression_gives_numpys_results_with_the_stand_in():
    """The regression run with the stand-in for the module defers every operation and matches NumPy bit for bit."""
    rng = numpy.random.default_rng(SEED)
    features = rng.standard_normal((EXAMPLES, FEATURES))
    targets = features @ rng.standard_normal(FEATURES) + 0.1 * rng.standard_normal(EXAMPLES)
    expected = regression.fit(numpy, features, targets, STEPS)
    graph.COUNTS.reset()
    weights, intercept, losses = regression.fit(lazy, features, targets, STEPS)
    assert graph.COUNTS.deferred > 0 and graph.COUNTS.at_once == {}
    assert weights._node.pending and all(loss._node.pending for loss in losses)
    got = gather(weights, intercept, *losses)
    assert same(expected[0], got[0]) and same(expected[1], got[1])
    assert all(same(a, b) for a, b in zip(expected[2], got[2:], strict=True))


def test_a_transformer_block_is_lazy_when_its_input_is():
    """The block calls the real numpy module; one lazy input defers all of it, with NumPy's result."""
    rng = numpy.random.default_rng(SEED)
    params = block_parameters(rng)
    x = rng.standard_normal((TOKENS, WIDTH)).astype(numpy.float32)
    expected = attention.block(x, params, HEADS)
    graph.COUNTS.reset()
    out = attention.block(lazy.asarray(x), params, HEADS)
    assert type(out) is LazyArray and out.shape == expected.shape and out.dtype == expected.dtype
    assert graph.COUNTS.deferred > 0 and graph.COUNTS.at_once == {}
    assert same(expected, out.gather())


def test_a_strict_array_gives_values_only_through_gather():
    """On a strict array every implicit way to a value raises NotGathered, also when the array is already computed."""
    x = strict.arange(6.0)
    total = (x * 2).sum()
    total.gather()
    ways = {
        "float": lambda: float(total), "int": lambda: int(total), "bool": lambda: bool(total > 0),
        "if": lambda: 1 if total > 0 else 0, "format": lambda: f"{total:.3f}",
        "range": lambda: range(strict.arange(4).max()), "in": lambda: 3.0 in x, "flat": lambda: x.flat,
        "tolist": lambda: x.tolist(), "item": lambda: total.item(), "allclose": lambda: numpy.allclose(x, x),
    }
    for name, way in ways.items():
        assert isinstance(raised(way), NotGathered), name
    assert isinstance(raised(lambda: float(total)), TypeError)
    assert same(numpy.float64(30.0), total.gather())


def test_strictness_follows_the_array():
    """What is computed from a strict array is strict, through the real module and beside a plain or a lenient array."""
    hard = strict.arange(3.0)
    assert isinstance(raised(lambda: float(numpy.sum(hard * 2))), NotGathered)
    assert isinstance(raised(lambda: float((hard + lazy.arange(3.0) + numpy.ones(3)).sum())), NotGathered)
    assert isinstance(raised(lambda: int(numpy.argmax(hard))), NotGathered)
    assert float((lazy.arange(3.0) * 2).sum()) == 6.0


def test_numpy_cannot_take_the_values_of_a_strict_array_unnoticed():
    """NumPy asking for an array, an ndarray written from a lazy array, and the array interface are all refused."""
    x = strict.arange(6.0)
    plain = numpy.zeros(6)
    ways = {
        "asarray": lambda: numpy.asarray(x), "array": lambda: numpy.array(x),
        "ndarray[:] = lazy": lambda: plain.__setitem__(slice(None), x),
        "ndarray += lazy": lambda: plain.__iadd__(x), "out=ndarray": lambda: numpy.multiply(x, 2, out=plain),
        "copyto": lambda: numpy.copyto(plain, x), "index": lambda: plain[strict.arange(3)],
        "generator": lambda: numpy.random.default_rng(SEED).permutation(x),
    }
    for name, way in ways.items():
        assert isinstance(raised(way), NotGathered), name
    assert not plain.any()
    for name in ("__array_interface__", "__array_struct__"):
        assert not hasattr(x, name) and not hasattr(lazy.arange(3.0), name)


def test_a_value_needed_without_gather_is_what_numpy_gives():
    """By default every implicit way to a value computes it, and gives what the same line gives with NumPy."""
    data = numpy.arange(6.0)
    x = lazy.asarray(data.copy())
    total, plain_total = (x * 2).sum(), (data * 2).sum()
    graph.COUNTS.reset()
    pairs = {
        "float": (float(total), float(plain_total)), "int": (int(total), int(plain_total)),
        "complex": (complex(total), complex(plain_total)), "bool": (bool(total > 0), bool(plain_total > 0)),
        "if": (1 if total > 100 else 0, 0), "format": (f"{total:.3f}", f"{plain_total:.3f}"),
        "range": (range(lazy.arange(4).max()), range(numpy.arange(4).max())), "in": (3.0 in x, 3.0 in data),
        "tolist": (x.tolist(), data.tolist()), "item": (total.item(), plain_total.item()),
        "allclose": (numpy.allclose(x, x + 1e-12), True), "flat": (list(x.flat), list(data.flat)),
        "index": (list(data[lazy.arange(3)]), list(data[numpy.arange(3)])),
    }
    for name, (got, expected) in pairs.items():
        assert type(got) is type(expected) and got == expected, name
    assert same(data, numpy.asarray(x)) and numpy.asarray(x) is x.gather()
    assert same(data, numpy.array(x)) and numpy.array(x) is not x.gather()
    assert same(numpy.random.default_rng(SEED).permutation(data), numpy.random.default_rng(SEED).permutation(x))
    assert graph.COUNTS.needed["float()"] == 1 and graph.COUNTS.needed["Conversion to an ndarray"] >= 4


def test_lazy_values_written_into_an_ndarray_are_numpys():
    """ndarray[:] = lazy, ndarray += lazy, out=ndarray and copyto compute the lazy side and let NumPy write."""
    data = numpy.arange(4.0)
    cases = {
        "ndarray[:] = lazy": lambda np, plain, x: plain.__setitem__(slice(None), x * 2),
        "ndarray += lazy": lambda np, plain, x: plain.__iadd__(x * 2),
        "out=ndarray": lambda np, plain, x: numpy.multiply(x, 3, out=plain),
        "copyto": lambda np, plain, x: numpy.copyto(plain, x + 1),
    }
    for name, case in cases.items():
        expected, got = numpy.ones(4), numpy.ones(4)
        case(numpy, expected, data)
        case(lazy, got, lazy.asarray(data))
        assert same(expected, got), name


def test_handing_out_an_array_computes_everything_pending_first():
    """Before lazy values reach an ndarray, all pending work is computed: it may read what is about to change."""
    held = numpy.zeros(3)
    reader = lazy.asarray(held) + 1
    source = lazy.ones(3) * 5
    held[:] = source
    assert not reader._node.pending
    assert same(numpy.ones(3), reader.gather()) and same(numpy.full(3, 5.0), held)
    # A number that leaves is not an array someone can write through: only what it needs is computed.
    untouched = lazy.arange(3.0) * 2
    assert float((lazy.arange(3.0) + 1).sum()) == 6.0 and untouched._node.pending


def test_eager_calls_and_cuts_are_logged_apart_and_once_per_line(caplog):
    """A call made at once is an INFO line of one logger; a write or a value needed is a DEBUG line of another."""
    caplog.set_level(logging.DEBUG, logger="prototypes.lazy")
    graph.COUNTS.reset()
    x = lazy.asarray(numpy.arange(4.0))
    for _ in range(3):
        float((x * 2).sum())
    numpy.argsort(x)
    x[0] = 9.0
    lines = [(entry.name, entry.levelname, entry.getMessage()) for entry in caplog.records if entry.name.startswith("prototypes.lazy")]
    assert [(name, level) for name, level, _ in lines] == [
        ("prototypes.lazy.cut", "DEBUG"), ("prototypes.lazy.eager", "INFO"), ("prototypes.lazy.cut", "DEBUG")]
    assert lines[0][2].startswith("needed: float() at test_lazy_prototype.py:")
    assert lines[1][2].startswith("eager: np.argsort at test_lazy_prototype.py:")
    assert lines[2][2].startswith("write: x[index] = v at test_lazy_prototype.py:")
    assert graph.COUNTS.needed == {"float()": 3} and graph.COUNTS.at_once == {"np.argsort": 1, "x[index] = v": 1}


def test_at_info_only_eager_calls_are_logged(caplog):
    """With the package's logger at INFO, a line says what NumPy had to compute at once and nothing else."""
    caplog.set_level(logging.INFO, logger="prototypes.lazy")
    x = lazy.asarray(numpy.arange(4.0))
    float((x * 2).sum())
    numpy.sort(x)
    x[1] = 9.0
    lines = [entry.getMessage() for entry in caplog.records if entry.name.startswith("prototypes.lazy")]
    assert len(lines) == 1 and lines[0].startswith("eager: np.sort at ")


def test_the_report_ranks_what_was_computed_unasked():
    """lazy_report() has every occurrence by call and by line, the most frequent first, in three parts."""
    lazy_report(reset=True)
    x = lazy.asarray(numpy.arange(4.0))
    for _ in range(3):
        numpy.argsort(x * 2)
    numpy.sort(x)
    x[0] = float((x + 1).sum())
    found = lazy_report()
    assert [(row.call, row.count) for row in found.eager] == [("np.argsort", 3), ("np.sort", 1)]
    assert found.eager[0].places[0][1] == 3 and found.eager[0].places[0][0].startswith("test_lazy_prototype.py:")
    assert [(row.call, row.count) for row in found.writes] == [("x[index] = v", 1)]
    assert [(row.call, row.count) for row in found.needed] == [("float()", 1)]
    text = str(found).splitlines()
    assert text[0] == "Made at once by NumPy: not in the deferred set"
    assert text[1].split()[:2] == ["3", "np.argsort"] and text[1].endswith("× 3") and text[2].split()[:2] == ["1", "np.sort"]
    assert "A graph computed early: a write into a lazy array" in text and "none" not in str(found)
    assert lazy_report(reset=True).eager and not lazy_report().eager and "none" in str(lazy_report())


def test_iterating_needs_no_value():
    """Iteration gives lazy views of the first dimension, for a strict array too; a scalar is not iterable, as in NumPy."""
    data = numpy.arange(6.0).reshape(2, 3)
    for stand_in in (lazy, strict):
        rows = list(stand_in.asarray(data) * 2)
        assert len(rows) == 2 and all(type(row) is LazyArray and row._node.pending for row in rows)
        assert same(data[1] * 2, rows[1].gather())
    assert [float(value) for value in lazy.arange(3.0)] == [0.0, 1.0, 2.0]
    assert str(raised(lambda: iter(lazy.arange(3.0).sum()))) == str(raised(lambda: iter(numpy.array(3.0))))


def test_a_loop_over_an_array_that_has_its_values_makes_no_graph_per_element():
    """One element of an array that has its values is read without a node, so a Python loop over it is not a graph per turn."""
    rng = numpy.random.default_rng(SEED)
    away, keeper = rng.random(ROSTER) < 0.3, rng.random(ROSTER) < 0.4
    figures = rng.integers(0, 9, (ROSTER, FIGURES))
    names = [f"player {place}" for place in range(ROSTER)]
    expected = numpy.flatnonzero(~away)
    graph.COUNTS.reset()
    with record_graphs() as recording:
        here = lazy.flatnonzero(~lazy.asarray(away))
        kept, table = lazy.asarray(keeper), lazy.asarray(figures)
        elements = list(here)
        picked = [names[position] for position in here]
        keepers = [names[position] for position in here if kept[position]]
        read = [int(table[position, turn % FIGURES]) for turn, position in enumerate(here)]
        last, typed = table[-1, -1], table[numpy.int64(2), numpy.uint8(1)]
    # The positions were computed once, when the first loop asked how many there are; nothing after that computed.
    assert [(each.trigger, each.call) for each in recording.graphs] == [(graph.NEEDED, "The shape of what a mask selects")]
    assert graph.COUNTS.deferred == 2 and graph.COUNTS.at_once == {}
    assert graph.COUNTS.needed == {"The shape of what a mask selects": 1}
    assert len(elements) == len(expected) and all(type(element) is LazyArray and not element._node.pending for element in elements)
    assert all(same(plain, element.gather()) for plain, element in zip(expected, elements))
    assert picked == [names[position] for position in expected]
    assert keepers == [names[position] for position in expected if keeper[position]]
    assert read == [int(figures[position, turn % FIGURES]) for turn, position in enumerate(expected)]
    assert same(figures[-1, -1], last.gather()) and same(figures[2, 1], typed.gather())
    for bad in ((ROSTER, 0), (0, -FIGURES - 1), (0, 0, 0)):
        assert str(raised(lambda: table[bad])) == str(raised(lambda: figures[bad])), bad
    # A row is a view and what is pending stays pending: both are deferred as before.
    assert table[1]._node.pending and (table * 2)[1, 1]._node.pending and table[1, 1, ...]._node.pending
    assert same(figures[1], table[1].gather()) and same(figures[1, 1, ...], table[1, 1, ...].gather())
    # A number handed out is counted where something was computed for it; an array handed out, always.
    graph.COUNTS.reset()
    assert float(elements[0]) == float(expected[0]) and bool(kept[1]) is bool(keeper[1]) and f"{last:d}" == f"{figures[-1, -1]:d}"
    assert graph.COUNTS.needed == {}
    assert same(figures, numpy.asarray(table)) and graph.COUNTS.needed == {"Conversion to an ndarray": 1}
    assert float((table * 2)[1, 1]) == float(figures[1, 1] * 2) and graph.COUNTS.needed["float()"] == 1
    # The element is the one that was there when the line ran, as NumPy's is.
    outside, inside = numpy.arange(4.0), lazy.arange(4.0)
    before, first = lazy.asarray(outside)[2], inside[2]
    outside[2], inside[2] = 9.0, 9.0
    assert same(numpy.float64(2.0), before.gather()) and same(numpy.float64(2.0), first.gather())
    assert inside[2]._node.pending and same(numpy.float64(9.0), inside[2].gather())


def test_an_element_of_a_strict_array_is_strict():
    """One element of a strict array that has its values is read without a node and gives its value only through gather()."""
    data = numpy.arange(6.0).reshape(2, 3)
    x = strict.asarray(data)
    graph.COUNTS.reset()
    element, elements = x[1, 2], list(strict.arange(3))
    assert type(element) is LazyArray and not element._node.pending and element._node.strict
    assert all(not each._node.pending and each._node.strict for each in elements)
    for name, way in {"float": lambda: float(element), "if": lambda: 1 if element else 0, "format": lambda: f"{element:.1f}",
                      "index": lambda: ["a", "b", "c"][elements[1]], "loop": lambda: [range(3)[i] for i in strict.arange(3)]}.items():
        assert isinstance(raised(way), NotGathered), name
    assert same(data[1, 2], element.gather()) and same(numpy.int64(1), elements[1].gather())
    assert same(data[1, 2] * 2, (element * 2).gather()) and (element * 2)._node.strict
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}


def test_shape_and_type_errors_are_numpys_and_come_at_the_line():
    """A wrong shape, axis, product or number raises NumPy's exception with NumPy's message, with nothing computed."""
    cases = {
        "broadcast": lambda np: np.arange(6.0) + np.arange(4.0),
        "matmul": lambda np: np.ones((3, 4)) @ np.ones((3, 4)),
        "axis": lambda np: np.sum(np.arange(6.0), axis=3),
        "reshape": lambda np: np.arange(6.0).reshape(4, 2),
        "index": lambda np: np.arange(6.0)[7],
        "python int": lambda np: np.ones(3, dtype=np.int8) + 300,
        "types": lambda np: np.arange(3.0) & np.arange(3.0),
    }
    for name, case in cases.items():
        expected, got = raised(lambda: case(numpy)), raised(lambda: case(lazy))
        assert type(expected) is type(got) and str(expected) == str(got), name


def test_element_types_are_numpys():
    """Promotion follows NumPy for Python numbers, NumPy scalars, reductions and comparisons."""
    cases = {
        "float32 * python float": lambda np: np.ones(3, dtype=np.float32) * 2.0,
        "float32 * float64 scalar": lambda np: np.ones(3, dtype=np.float32) * np.float64(2),
        "int8 + python int": lambda np: np.ones(3, dtype=np.int8) + 1,
        "sum of int8": lambda np: np.ones(3, dtype=np.int8).sum(),
        "mean of int32": lambda np: np.ones(3, dtype=np.int32).mean(),
        "comparison": lambda np: np.arange(3.0) > 1,
        "where": lambda np: np.where(np.arange(3) > 1, np.ones(3, dtype=np.float32), 2.0),
        "int / int": lambda np: np.arange(3) / np.arange(1, 4),
        "bool + bool": lambda np: (np.arange(3) > 1) + (np.arange(3) > 0),
        "matmul": lambda np: np.ones((2, 3), dtype=np.float32) @ np.ones(3),
    }
    for name, case in cases.items():
        expected, got = case(numpy), case(lazy)
        assert got.dtype == expected.dtype and got.shape == expected.shape, name
        assert same(expected, got.gather()), name


def test_a_warning_names_the_line_that_wrote_the_operation():
    """A division by zero warns at the gather, with NumPy's words and the division's own line."""
    x = lazy.arange(3.0)
    line = sys._getframe().f_lineno + 1
    quotient = x / lazy.zeros(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quotient.gather()
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        numpy.arange(3.0) / numpy.zeros(3)
    assert [str(w.message) for w in caught] == [str(w.message) for w in expected] and len(caught) == 2
    assert all(w.filename == __file__ and w.lineno == line and w.category is RuntimeWarning for w in caught)


def test_errstate_is_remembered_per_operation():
    """An operation written under np.errstate keeps it wherever it is gathered."""
    with numpy.errstate(divide="ignore", invalid="ignore"):
        quiet = lazy.arange(3.0) / lazy.zeros(3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet.gather()
    assert caught == []


def test_a_failure_carries_a_note_and_stays():
    """Under errstate raise the gather raises, the note names the line, and gathering again raises the same."""
    line = sys._getframe().f_lineno + 2
    with numpy.errstate(over="raise"):
        big = lazy.array([1e308]) * 10
    dependant = big + 1
    error = raised(big.gather)
    assert isinstance(error, FloatingPointError) and f"{__file__}:{line}" in error.__notes__[0]
    assert raised(big.gather) is error and raised(dependant.gather) is error
    assert repr(big) == "LazyArray(float64, shape=(1,), failed)"


def test_a_failure_nobody_gathered_surfaces_once_at_a_write():
    """A write computes everything pending: a first failure is raised there once, and its dependants stay quiet."""
    with numpy.errstate(over="raise"):
        big = lazy.array([1e308]) * 10
    dependant = big + 1
    target = lazy.asarray(numpy.zeros(2))
    with pytest.raises(FloatingPointError):
        target += 1
    target += 1
    assert same(numpy.ones(2), target.gather())
    assert isinstance(raised(dependant.gather), FloatingPointError)


def test_writes_into_lazy_arrays_have_numpys_meaning():
    """Assignment through a view, in-place operators and out= compute what is pending and then write."""
    base = lazy.zeros(4)
    before = base + 1
    view = base[1:3]
    view[0] = 9.0
    assert same(numpy.array([0.0, 9.0, 0.0, 0.0]), base.gather()) and same(numpy.ones(4), before.gather())

    weights = lazy.zeros(4)
    alias = weights
    weights -= 0.5 * lazy.ones(4)
    assert weights is alias and same(numpy.full(4, -0.5), alias.gather())

    total = lazy.arange(4.0).sum()
    first = total
    total += 1
    assert total is not first and same(numpy.float64(6.0), first.gather()) and same(numpy.float64(7.0), total.gather())

    out = lazy.zeros(3)
    assert numpy.multiply(numpy.arange(3.0), 2, out=out) is out and same(numpy.array([0.0, 2.0, 4.0]), out.gather())

    counts = lazy.arange(3)
    error = raised(lambda: counts.__iadd__(0.5))
    assert type(error) is type(raised(lambda: numpy.arange(3).__iadd__(0.5)))


def test_a_write_into_an_array_of_the_layers_own_is_a_node():
    """In-place operators, assignment, out= and np.clip into such an array compute nothing and give NumPy's values."""
    def program(np):
        """Write into an array in five ways; return it, another name for it, and what read it before the writes."""
        w = np.arange(8.0).reshape(2, 4) * 1.0
        alias, before = w, w + 100
        w += 2
        w[1, 1:3] = np.ones(2) * 50
        w *= np.arange(4.0)
        np.multiply(w, 0.5, out=w)
        np.clip(w, 1.0, 40.0, out=w)
        w[w > 30] = -1.0
        return w, alias, before

    expected = program(numpy)
    graph.COUNTS.reset()
    w, alias, before = program(lazy)
    assert w is alias and w._node.pending and w._node.kind == "write" and graph.COUNTS.at_once == {}
    assert same(expected[2], before.gather()) and same(expected[0], w.gather())


def test_a_write_reaches_every_view_and_a_write_through_a_view_reaches_the_array():
    """Views are views of the array's contents now: they follow writes, and writes through them are writes into it."""
    def program(np):
        """Write through rows, columns and a transpose of one array, and read the views afterwards."""
        base = np.zeros((3, 4))
        row, column, turned = base[1], base[:, 2], base.T
        row += 1
        base[0, :2] = 7
        column *= 3
        turned[3, 2] = 9
        base[2] -= 4
        row[1:3] = row[0:2] + 10
        return base, row, column, turned

    expected = program(numpy)
    graph.COUNTS.reset()
    got = program(lazy)
    assert all(each._node.pending for each in got) and graph.COUNTS.at_once == {}
    for name, plain, mine in zip(("base", "row", "column", "turned"), expected, got, strict=True):
        assert same(plain, mine.gather()), name


# ---- `x.view(type(x))`: how NumPy code keeps the kind of array it was given.

# The ways a call asks for the same elements with no change of element type: each given the array, it makes the call.
WHOLE_VIEWS = {
    "no argument": lambda x: x.view(),
    "its own type": lambda x: x.view(type(x)),
    "its own type as type": lambda x: x.view(type=type(x)),
    "its own type as dtype": lambda x: x.view(dtype=type(x)),
    "ndarray": lambda x: x.view(numpy.ndarray),
    "ndarray as type": lambda x: x.view(type=numpy.ndarray),
}
# Calls of `view` that NumPy refuses, each given the array and its type.
REFUSED_VIEWS = {
    "its type and None": lambda x, kind: x.view(kind, None),
    "its type twice": lambda x, kind: x.view(kind, kind),
    "its type under both names": lambda x, kind: x.view(dtype=kind, type=kind),
    "another array type beside it": lambda x, kind: x.view(numpy.matrix, kind),
    "its type by position and dtype by name": lambda x, kind: x.view(kind, dtype=numpy.int64),
    "a keyword view does not have": lambda x, kind: x.view(kind, order="C"),
    "three arguments": lambda x, kind: x.view(None, kind, kind),
    "None as type": lambda x, kind: x.view(type=None),
    "an element type that does not fit": lambda x, kind: x.view(numpy.complex128, kind),
}


def laid_out_like(expected: numpy.ndarray, got: numpy.ndarray) -> bool:
    """Say whether two arrays lie in memory the same way, beyond what `same` compares.

    Args:
        expected: What eager NumPy returned.
        got: What `gather()` returned.

    Returns:
        True when the strides agree, and whether each is a view of other memory and may be written.
    """
    return (expected.strides == got.strides and (expected.base is None) == (got.base is None)
            and expected.flags.owndata == got.flags.owndata and expected.flags.writeable == got.flags.writeable)


@pytest.mark.parametrize("spelling", sorted(WHOLE_VIEWS))
def test_a_view_by_the_arrays_own_type_is_a_view_of_the_array(spelling: str):
    """`x.view(type(x))`, `x.view()` and `x.view(np.ndarray)` are deferred, and are NumPy's view: bits and layout."""
    view = WHOLE_VIEWS[spelling]
    special = numpy.array([[numpy.nan, -0.0, numpy.inf, 1.5], [-numpy.inf, 0.0, 5e-324, -numpy.nan]])

    def program(np):
        """View an array as it is made, one out of order, one that steps, one of one element, and one pending."""
        made = np.asarray(special)
        turned = np.asarray(special).T
        stepped = np.asarray(numpy.arange(24.0).reshape(4, 6))[::2, 1::2]
        one = np.asarray(numpy.array([7.0]))
        pending = np.asarray(numpy.arange(6, dtype=numpy.int32)) * 3
        return view(made), view(turned), view(stepped), view(one), view(pending), view(view(pending))

    expected = program(numpy)
    graph.COUNTS.reset()
    got = program(lazy)
    assert all(type(each) is LazyArray and each._node.pending for each in got) and graph.COUNTS.at_once == {}
    for plain, mine in zip(expected, got, strict=True):
        value = mine.gather()
        assert same(plain, value) and laid_out_like(plain, value)


def test_matplotlibs_array_of_normals_is_written_through_its_view():
    """`np.empty(shape).view(type(x))` is an array the program fills: the writes wait, and the values are NumPy's."""
    def program(np):
        """Make the normals of a surface as `LightSource.hillshade` does, and scale each to length one."""
        elevation = np.asarray(numpy.linspace(0.0, 5.0, 12).reshape(3, 4) ** 2)
        normal = np.empty(elevation.shape + (3,)).view(type(elevation))
        normal[..., 0] = -elevation
        normal[..., 1] = elevation * 0.5
        normal[..., 2] = 1
        length = 0
        for axis in range(3):
            length = length + normal[..., axis, np.newaxis] ** 2
        normal /= np.sqrt(length)
        return normal, normal[..., 2], normal.sum()

    expected = program(numpy)
    graph.COUNTS.reset()
    got = program(lazy)
    assert all(each._node.pending for each in got) and graph.COUNTS.at_once == {}
    for plain, mine in zip(expected, got, strict=True):
        assert same(plain, mine.gather())


def test_a_write_is_seen_through_a_view_by_type_both_ways():
    """The view and the array are one memory: a write into either shows in the other, before and after computing."""
    def program(np):
        """Write into an array and into its view by type, in turn."""
        base = np.zeros((2, 3)) + 1.0
        view = base.view(type(base))
        view[0, 1] = 5.0
        base[1] = view[0] * 2.0
        view *= -1.0
        again = view.view(type(view))
        again[1, 2] = numpy.nan
        return base, view, again

    expected = program(numpy)
    graph.COUNTS.reset()
    got = program(lazy)
    assert all(each._node.pending for each in got) and graph.COUNTS.at_once == {}
    for plain, mine in zip(expected, got, strict=True):
        assert same(plain, mine.gather())
    base, view, _ = got
    base[0, 0] = 9.0  # After computing: the same memory still.
    view[1, 0] = 8.0
    assert view.gather()[0, 0] == 9.0 and base.gather()[1, 0] == 8.0


def test_a_view_by_type_that_changes_the_elements_is_numpys_at_once():
    """An element type beside the array's own type is NumPy's reading of the same bytes; the type itself is dropped."""
    def program(np):
        """Read float64 as int64 and as pairs of float32, and int64 as NumPy reads it for `None`."""
        x = np.asarray(numpy.array([[1.5, -0.0], [numpy.nan, numpy.inf]]))
        whole = np.asarray(numpy.arange(4))
        return (x.view(numpy.int64, type(x)), x.view(dtype=numpy.float32, type=type(x)), x.view(numpy.int64),
                whole.view(None, type(whole)), whole.view(None))

    expected = program(numpy)
    graph.COUNTS.reset()
    got = program(lazy)
    assert graph.COUNTS.at_once == {"ndarray.view": len(expected)}
    for plain, mine in zip(expected, got, strict=True):
        assert same(plain, mine.gather()) and laid_out_like(plain, mine.gather())


def test_a_view_by_type_of_what_the_layer_does_not_defer_is_numpys_at_once():
    """No dimension, no element, an element type outside the deferred set: NumPy's own `view` on the value."""
    def program(np):
        """View a number, an array without dimensions, an empty array and an array of strings by their own type."""
        x = np.asarray(numpy.arange(6.0))
        number = x.sum()
        alone = np.asarray(numpy.array(3.0))
        empty = np.asarray(numpy.empty((0, 3)))
        words = np.asarray(numpy.array(["a", "bc"]))
        return (number.view(type(number)), alone.view(type(alone)), empty.view(type(empty)), words.view(type(words)),
                number.view(), alone.view(numpy.ndarray))

    expected = program(numpy)
    graph.COUNTS.reset()
    got = program(lazy)
    assert graph.COUNTS.at_once["ndarray.view"] == len(expected)
    for plain, mine in zip(expected, got, strict=True):
        assert same(plain, mine.gather())
    # A number stays a number, as `np.float64(15.0).view()` does; `x[...]` of it would be an array.
    assert type(got[0].gather()) is numpy.float64 and type(got[1].gather()) is numpy.ndarray


def test_a_view_by_type_of_a_selection_by_a_mask_is_numpys():
    """Rows a mask selects, some and none: the view is of what was selected, and a write through it reaches that."""
    def program(np):
        """Select rows by a mask, view each selection by its own type, and write through one view."""
        x = np.asarray(numpy.arange(12.0).reshape(4, 3))
        some, none = x[x[:, 0] > 2], x[x[:, 0] > 99]
        first, second = some.view(type(some)), none.view(type(none))
        first[0, 1] = -1.0
        return first, second, some, first * 2, second + 1

    expected = program(numpy)
    for plain, mine in zip(expected, program(lazy), strict=True):
        assert same(plain, mine.gather())


def test_a_view_by_type_keeps_the_kind_of_array_the_value_is():
    """A lazy array that holds a matrix or a masked array is viewed by NumPy's method: its own type keeps the kind."""
    data = numpy.arange(6.0).reshape(2, 3)
    for kind in (numpy.matrix, numpy.ma.MaskedArray):
        held = lazy.asarray(data).view(kind)
        plain = data.view(kind)
        assert type(held) is LazyArray and type(held.gather()) is kind
        own, stripped = held.view(type(held)), held.view(numpy.ndarray)
        assert same(plain.view(type(plain)), own.gather()) and same(plain.view(numpy.ndarray), stripped.gather())
        assert same(plain.view(), held.view().gather())


@pytest.mark.parametrize("spelling", sorted(REFUSED_VIEWS))
def test_a_view_numpy_refuses_is_refused_the_same_way(spelling: str):
    """An array type given twice, a keyword `view` lacks, an element type that does not fit: NumPy's error and words."""
    call = REFUSED_VIEWS[spelling]
    data = numpy.arange(6.0).reshape(2, 3)
    theirs = raised(lambda: call(data, numpy.ndarray))
    ours = raised(lambda: call(lazy.asarray(data), LazyArray))
    assert type(ours) is type(theirs) and str(ours) == str(theirs)


def test_a_view_by_type_of_a_strict_array_is_strict():
    """The view refuses its values without `gather()` as the array does, deferred or made at once."""
    x = strict.asarray(numpy.arange(6.0).reshape(2, 3))
    for view in (x.view(type(x)), x.view(numpy.int64, type(x))):
        assert view._node.strict
        with pytest.raises(NotGathered):
            float(view[0, 0])


def test_a_number_is_not_written_in_place():
    """`s += v` on what NumPy gives as a number makes a new lazy array, deferred, and leaves the old one alone."""
    s = (lazy.arange(4.0) * 2).sum()
    first = s
    graph.COUNTS.reset()
    s += 1
    s *= 2
    assert s is not first and s._node.pending and graph.COUNTS.at_once == {}
    assert same(numpy.float64(26.0), s.gather()) and same(numpy.float64(12.0), first.gather())


def test_a_write_into_memory_known_outside_is_made_at_once():
    """An array the user handed in, one that was gathered, and one a reshape may share are written in their memory."""
    mine = numpy.zeros(3)
    x = lazy.asarray(mine)
    x += 1
    assert mine.tolist() == [1.0, 1.0, 1.0] and not x._node.pending

    w = lazy.zeros(3) + 1
    held = w.gather()
    w += 1
    assert held is w.gather() and held.tolist() == [2.0, 2.0, 2.0]

    def program(np):
        """Write into an array through a reshape of it and through a ravel of it, and into it while they live."""
        base = np.zeros(6)
        shaped, flat = base.reshape(2, 3), base.ravel()
        shaped[0, 0] = 5
        flat[1] = 6
        base += 1
        return base, shaped, flat

    for name, plain, got in zip(("base", "reshape", "ravel"), program(numpy), program(lazy), strict=True):
        assert same(plain, got.gather()), name


def test_a_deferred_write_of_the_wrong_shape_or_place_raises_on_its_line():
    """An index out of bounds and a value that does not fit raise NumPy's error when written, and nothing changes."""
    w, plain = lazy.zeros((3, 4)), numpy.zeros((3, 4))
    cases = {
        "out of bounds": lambda x: x.__setitem__((5, 0), 1.0),
        "too many values": lambda x: x.__setitem__((0, slice(0, 2)), numpy.ones(3)),
        "through a view": lambda x: x[1].__setitem__(9, 1.0),
        "cast": lambda x: (x > 0).__iadd__(1.5),
    }
    for name, case in cases.items():
        assert str(raised(lambda: case(w))) == str(raised(lambda: case(plain))), name
    assert same(plain, w.gather())


def test_arrays_from_a_random_generator_are_lazy():
    """np.random.default_rng() through the stand-in gives a generator whose arrays are lazy and the layer's own."""
    rng, plain = lazy.random.default_rng(SEED), numpy.random.default_rng(SEED)
    x, expected = rng.standard_normal((3, 4)), plain.standard_normal((3, 4))
    assert type(x) is LazyArray and same(expected, (x * 1).gather())
    graph.COUNTS.reset()
    x -= 0.5
    expected -= 0.5
    assert x._node.pending and graph.COUNTS.at_once == {} and same(expected, x.gather())
    assert same(plain.permutation(5), rng.permutation(5).gather()) and rng.bit_generator is not None


def test_arrays_from_numpys_index_objects_are_lazy():
    """`np.mgrid[...]`, `np.ogrid[...]`, `np.r_[...]` and `np.c_[...]` through the stand-in give lazy arrays with NumPy's values."""
    grid, expected = lazy.mgrid[0:4, 0:3], numpy.mgrid[0:4, 0:3]
    assert type(grid) is LazyArray and same(expected, grid.gather())
    rows, columns = lazy.mgrid[0:4, 0:3]
    assert type(rows) is LazyArray and same(expected[0], rows.gather()) and same(expected[1], columns.gather())
    open_rows, open_columns = lazy.ogrid[0:4, 0:3]
    assert type(open_rows) is LazyArray and same(numpy.ogrid[0:4, 0:3][1], open_columns.gather())
    data = numpy.random.default_rng(SEED).standard_normal(5)
    x = lazy.asarray(data)
    assert same(numpy.r_[1:4, 0.5, data], lazy.r_[1:4, 0.5, data].gather())
    assert same(numpy.r_[data, data * 2], lazy.r_[x, x * 2].gather())
    assert same(numpy.c_[data, data], lazy.c_[x, x].gather())
    assert lazy.mgrid.sparse is numpy.mgrid.sparse
    # What the grids are for: a grid indexed by a lazy mask is the layer's indexing, with NumPy's result.
    keep = lazy.asarray(expected[0].ravel() > 1)
    flat = lazy.mgrid[0:4, 0:3][0]
    flat.shape = 12
    assert same(expected[0].ravel()[expected[0].ravel() > 1], flat[keep].gather())


def test_other_numpy_calls_are_made_at_once_and_stay_lazy():
    """A call outside the deferred set is NumPy's, on gathered operands, and its result is a computed lazy array."""
    data = numpy.random.default_rng(SEED).standard_normal(50)
    x = lazy.asarray(data)
    cases = {
        "part of what a mask selects": (x[x > 0][:3], data[data > 0][:3]),
        "norm": (numpy.linalg.norm(x * 2), numpy.linalg.norm(data * 2)),
        "a method outside the set": (x.argsort(), data.argsort()),
        "a running total of another type": (numpy.cumsum(x, dtype=numpy.float32), numpy.cumsum(data, dtype=numpy.float32)),
        "a search with a sorter": (numpy.searchsorted(x, 0.0, sorter=numpy.argsort(data)), numpy.searchsorted(data, 0.0, sorter=numpy.argsort(data))),
        "argsort": (numpy.argsort(-x), numpy.argsort(-data)),
        "ravel in another order": ((x * 2).reshape(5, 10).ravel(order="F"), (data * 2).reshape(5, 10).ravel(order="F")),
        "concatenate": (numpy.concatenate([x, data]), numpy.concatenate([data, data])),
        "dot of three dimensions": (numpy.dot(x.reshape(2, 5, 5), x.reshape(2, 5, 5)), numpy.dot(data.reshape(2, 5, 5), data.reshape(2, 5, 5))),
        "sort": (numpy.sort(x), numpy.sort(data)),
    }
    for name, (got, expected) in cases.items():
        assert type(got) is LazyArray and not got._node.pending, name
        assert same(expected, got.gather()), name


def test_running_totals_and_searches_are_deferred():
    """np.cumsum, np.cumprod and np.searchsorted join the graph, as functions and as methods, with NumPy's shapes and types."""
    data = numpy.random.default_rng(SEED).standard_normal((4, 6))
    counts = numpy.arange(1, 7, dtype=numpy.int8)
    x, n = lazy.asarray(data) + 0, lazy.asarray(counts) + numpy.int8(0)
    plain = data + 0
    table, plain_table = numpy.cumsum(numpy.abs(x[0])), numpy.cumsum(numpy.abs(plain[0]))
    graph.COUNTS.reset()
    cases = {
        "all the elements": (numpy.cumsum(x), numpy.cumsum(plain)), "along an axis": (numpy.cumsum(x, axis=1), numpy.cumsum(plain, axis=1)),
        "an axis by position": (numpy.cumsum(x, 0), numpy.cumsum(plain, 0)), "a method": (x.cumsum(axis=-1), plain.cumsum(axis=-1)),
        "small integers": (n.cumsum(), counts.cumsum()), "a running product": (numpy.cumprod(n), numpy.cumprod(counts)),
        "a search": (numpy.searchsorted(table, numpy.abs(x)), numpy.searchsorted(plain_table, numpy.abs(plain))),
        "a search from the right": (numpy.searchsorted(table, x[1], side="right"), numpy.searchsorted(plain_table, plain[1], side="right")),
        "a search for one number": (table.searchsorted(1.0), plain_table.searchsorted(1.0)),
        "a search of an ndarray": (numpy.searchsorted(plain_table, numpy.abs(x[2]), "right"), numpy.searchsorted(plain_table, numpy.abs(plain[2]), "right")),
        "what a mask selects, summed as it goes": (x[x[:, 0] > 0].cumsum(axis=0), plain[plain[:, 0] > 0].cumsum(axis=0)),
        "a search for what a mask selects": (numpy.searchsorted(table, x[0][x[0] > 0]), numpy.searchsorted(plain_table, plain[0][plain[0] > 0])),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.dtype == expected.dtype and got.ndim == expected.ndim, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert str(raised(lambda: numpy.cumsum(x, axis=2))) == str(raised(lambda: numpy.cumsum(plain, axis=2)))
    assert same(numpy.sum(plain, (0, 1)), numpy.sum(x, (0, 1)).gather()) and same(numpy.sum(plain, None), numpy.sum(x, None).gather())

    def resample(np, weights, positions):
        """Pick, for each position, the first weight whose running total reaches it, as a particle filter does."""
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        return np.asarray(weights)[np.searchsorted(cumulative, positions)]

    weights = numpy.abs(data[0]) / numpy.abs(data[0]).sum()
    positions = (0.5 + numpy.arange(6)) / 6
    graph.COUNTS.reset()
    got = resample(lazy, lazy.asarray(weights) + 0, positions)
    assert got._node.pending and graph.COUNTS.at_once == {} and same(resample(numpy, weights, positions), got.gather())


def test_dot_and_copy_are_deferred():
    """np.dot of vectors and matrices is deferred as the matrix product it is; so is a copy, which stays a copy."""
    data = numpy.random.default_rng(SEED).standard_normal((4, 6))
    x = lazy.asarray(data)
    cases = {
        "matrix by matrix": (numpy.dot(x, x.T), numpy.dot(data, data.T)),
        "matrix by vector": (numpy.dot(x, x[0]), numpy.dot(data, data[0])),
        "vector by vector": (numpy.dot(x[0], x[1]), numpy.dot(data[0], data[1])),
        "plain by lazy": (numpy.dot(data.T, x), numpy.dot(data.T, data)),
        "method": (x.dot(x.T), data.dot(data.T)),
        "copy": (numpy.copy(x), numpy.copy(data)),
        "copy, method": (x.copy(), data.copy()),
    }
    graph.COUNTS.reset()
    for name, (got, expected) in cases.items():
        assert type(got) is LazyArray and got._node.pending and got.shape == numpy.shape(expected), name
        assert same(expected, got.gather()), name
    assert graph.COUNTS.at_once == {}
    assert x.copy().gather() is not data and x.gather() is data
    assert str(raised(lambda: numpy.dot(x, x))) == str(raised(lambda: numpy.dot(data, data)))


def test_rounding_is_deferred():
    """np.round, np.around and x.round() join the graph element by element, and with out= they are a write."""
    data = numpy.random.default_rng(SEED).standard_normal((4, 6)) * 100
    x = lazy.asarray(data) + 0
    plain = data + 0
    single, whole = lazy.asarray(data.astype(numpy.float32)) * 2, lazy.arange(24).reshape(4, 6) * 7
    plain_single, plain_whole = data.astype(numpy.float32) * 2, numpy.arange(24).reshape(4, 6) * 7
    graph.COUNTS.reset()
    cases = {
        "np.round": (numpy.round(x), numpy.round(plain)),
        "decimals by position": (numpy.round(x * 0.37, 2), numpy.round(plain * 0.37, 2)),
        "decimals by keyword": (numpy.round(x, decimals=1), numpy.round(plain, decimals=1)),
        "np.around, to tens": (numpy.around(x, -1), numpy.around(plain, -1)),
        "method": (x.round(), plain.round()), "method with decimals": (x.round(3), plain.round(3)),
        "method, a NumPy integer by keyword": (x.round(decimals=numpy.int64(1)), plain.round(decimals=numpy.int64(1))),
        "float32": (numpy.round(single, 2), numpy.round(plain_single, 2)), "integers": (whole.round(-1), plain_whole.round(-1)),
        "a number": (numpy.round(x.sum(), 2), numpy.round(plain.sum(), 2)),
        "through the stand-in": (lazy.round(x / 3, 2), numpy.round(plain / 3, 2)),
        "what a mask selects": (numpy.round(x[x[:, 0] > 0], 1).sum(axis=0), numpy.round(plain[plain[:, 0] > 0], 1).sum(axis=0)),
        "read from afterwards": (numpy.round(x @ x.T, 2)[:, :2] + 1, numpy.round(plain @ plain.T, 2)[:, :2] + 1),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == numpy.shape(expected) and got.dtype == expected.dtype, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    # The call is kept as it was written, for whatever looks at how it is spelled.
    assert numpy.round(x, 2)._node.args[1:] == (2,) and x.round(decimals=1)._node.kwargs == {"decimals": 1}
    for wrong in (lambda a: numpy.round(a, 1.5), lambda a: numpy.round(a, 1, decimals=2), lambda a: a.round(1, None, 3)):
        error, expected = raised(lambda: wrong(x)), raised(lambda: wrong(plain))
        assert type(error) is type(expected) and str(error) == str(expected)

    def program(np, start):
        """Round into the array that is read, into another, and through a view."""
        a = np.array(start)
        b, c = a * 3, a / 7
        np.round(b, 1, out=b)
        np.around(a / 7, 2, out=a)
        np.round(c[1], out=c[1])
        return a, b, c

    graph.COUNTS.reset()
    written = program(lazy, data)
    assert all(each._node.pending for each in written) and graph.COUNTS.at_once == {}
    for name, expected, got in zip(("a", "b", "c"), program(numpy, data), written, strict=True):
        assert same(expected, got.gather()), name
    into, plain_into = lazy.arange(6), numpy.arange(6)
    assert str(raised(lambda: numpy.round(x[0], 1, out=into))) == str(raised(lambda: numpy.round(plain[0], 1, out=plain_into)))
    assert same(plain_into, into.gather())


def test_an_array_like_another_needs_no_value():
    """zeros_like, ones_like and full_like use the shape and type of a lazy array and do not compute it."""
    pending = lazy.arange(6.0).reshape(2, 3) * 2
    graph.COUNTS.reset()
    made = {"zeros": numpy.zeros_like(pending), "ones": lazy.ones_like(pending), "full": numpy.full_like(pending, 7.0),
            "typed": numpy.zeros_like(pending, dtype=numpy.int32)}
    assert pending._node.pending and graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    assert same(numpy.zeros((2, 3)), made["zeros"].gather()) and same(numpy.ones((2, 3)), made["ones"].gather())
    assert same(numpy.full((2, 3), 7.0), made["full"].gather()) and same(numpy.zeros((2, 3), numpy.int32), made["typed"].gather())
    made["zeros"][0, 0] = 1.0
    assert made["zeros"].gather()[0, 0] == 1.0


def test_indexing_by_integer_arrays_is_deferred():
    """x[rows], x[[0, 2]], x[rows, columns] are deferred, and so are an index still pending and a mask that has its values."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 5))
    x = lazy.asarray(data) + 0
    plain = data + 0
    rows, columns = numpy.array([4, 0, 2]), numpy.array([1, 1, 3])
    cases = {
        "an array": (x[rows], plain[rows]), "a list": (x[[0, 2, 4]], plain[[0, 2, 4]]),
        "two arrays": (x[rows, columns], plain[rows, columns]), "a slice and an array": (x[1:, columns], plain[1:, columns]),
        "a computed lazy array": (x[lazy.asarray(rows)], plain[rows]), "two dimensions": (x[rows.reshape(3, 1)], plain[rows.reshape(3, 1)]),
    }
    graph.COUNTS.reset()
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == expected.shape and got.dtype == expected.dtype, name
        assert same(expected, got.gather()), name
    assert graph.COUNTS.at_once == {}
    assert str(raised(lambda: x[numpy.array([0, 6])])) == str(raised(lambda: plain[numpy.array([0, 6])]))
    keep = numpy.array([True, False, True, True, False, False])
    late = {"a pending index": (x[lazy.arange(3) * 2], plain[numpy.arange(3) * 2]),
            "a pending index beside an array": (x[lazy.arange(3) + 1, columns], plain[numpy.arange(3) + 1, columns]),
            "a mask that has its values": (x[keep], plain[keep]), "a list of booleans": (x[list(keep)], plain[keep])}
    for name, (got, expected) in late.items():
        assert got._node.pending and got.shape == expected.shape and same(expected, got.gather()), name
    assert graph.COUNTS.at_once == {}
    beyond = x[lazy.arange(3) * 5]
    assert beyond._node.pending and str(raised(beyond.gather)) == str(raised(lambda: plain[numpy.arange(3) * 5]))


def test_a_pending_mask_selects_a_length_that_is_not_known():
    """x[mask] is a node; what is computed from it is deferred where the shape does not depend on how many it selects."""
    data = numpy.random.default_rng(SEED).standard_normal((40, 3))
    x = lazy.asarray(data) + 0
    plain = data + 0
    mask, plain_mask = x[:, 0] > 0, plain[:, 0] > 0
    graph.COUNTS.reset()
    chosen = x[mask]
    assert chosen._node.pending and repr(chosen) == "LazyArray(float64, shape=(?, 3), pending)" and chosen.ndim == 2
    cases = {
        "a reduction along the mask": (chosen.mean(axis=0), plain[plain_mask].mean(axis=0)),
        "a reduction of all": (chosen.sum(), plain[plain_mask].sum()),
        "arithmetic with the same mask's selection": (chosen * 2 - x[mask, :1], plain[plain_mask] * 2 - plain[plain_mask, :1]),
        "a new axis": ((1 / x[mask, 0])[:, None] * chosen, (1 / plain[plain_mask, 0])[:, None] * plain[plain_mask]),
        "a product": (chosen @ numpy.ones(3), plain[plain_mask] @ numpy.ones(3)),
        "a position": (chosen.argmax(axis=0), plain[plain_mask].argmax(axis=0)),
        "a transpose": (chosen.T, plain[plain_mask].T),
        "nothing selected": (x[x[:, 0] > 99].sum(axis=0), plain[plain[:, 0] > 99].sum(axis=0)),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert str(raised(lambda: x[mask[:5]])) == str(raised(lambda: plain[plain_mask[:5]]))


def test_a_length_that_is_not_known_is_computed_when_it_is_asked_for():
    """The shape of x[mask] is a value: asking computes it and the log says so; a strict array refuses."""
    data = numpy.random.default_rng(SEED).standard_normal(30)
    x = lazy.asarray(data) + 0
    graph.COUNTS.reset()
    chosen = x[x > 0]
    assert chosen.shape == data[data > 0].shape and len(x[x > 0]) == (data > 0).sum() and (x[x < 0] * 2).size == (data < 0).sum()
    assert graph.COUNTS.needed == {"The shape of what a mask selects": 3} and graph.COUNTS.at_once == {}
    assert repr(chosen) == f"LazyArray(float64, shape=({(data > 0).sum()},), computed)"
    hard = strict.asarray(data) + 0
    assert isinstance(raised(lambda: hard[hard > 0].shape), NotGathered)
    assert same(data[data > 0], hard[hard > 0].gather())


def test_what_depends_on_the_length_is_left_to_numpy():
    """Where NumPy's answer depends on how many a mask selects, the call is made at once and is NumPy's."""
    data = numpy.random.default_rng(SEED).standard_normal(30)
    x = lazy.asarray(data) + 0
    cases = {
        "part of the selection": (x[x > 0][:4], data[data > 0][:4]),
        "one element of it": (x[x > 0][0], data[data > 0][0]),
        "a new shape": (x[x > 0].reshape(-1, 1), data[data > 0].reshape(-1, 1)),
        "its elements in one dimension": (x[x > 0].ravel(), data[data > 0].ravel()),
        "all of it against an array": (x[x > -99] + numpy.arange(30.0), data + numpy.arange(30.0)),
        "an array made like it": (numpy.zeros_like(x[x > 0]), numpy.zeros_like(data[data > 0])),
    }
    for name, (got, expected) in cases.items():
        assert not got._node.pending and same(expected, got.gather()), name
    assert str(raised(lambda: x[x > 0] + x[x < 0])) == str(raised(lambda: data[data > 0] + data[data < 0]))


def test_an_assignment_through_a_pending_mask_is_a_node():
    """x[mask] *= f(y[mask]) computes nothing; an array of another length through a mask is NumPy's to check."""
    def program(np, data):
        """Cap the speed of the rows that are too fast, then mark and replace some elements."""
        p = np.array(data)
        speed = np.hypot(p[:, 2], p[:, 3])
        fast = speed > 1.0
        p[fast, 2:] *= (1.0 / speed[fast])[:, None]
        p[fast, 0] = p[fast, 1] * 3
        rows = np.argmax(p, axis=0)
        p[rows, np.arange(4)] -= 1.0
        p[np.argmin(speed)] = 0.0
        never = speed > 99.0
        p[never, 2:] /= speed[never][:, None]
        return p

    data = numpy.random.default_rng(SEED).standard_normal((50, 4))
    graph.COUNTS.reset()
    got = program(lazy, data)
    assert got._node.pending and graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    assert same(program(numpy, data), got.gather())
    x, plain = lazy.array(data[:, 0]), data[:, 0].copy()
    x[x > 0] = numpy.arange((plain > 0).sum(), dtype=float)
    plain[plain > 0] = numpy.arange((plain > 0).sum(), dtype=float)
    assert graph.COUNTS.at_once == {"x[index] = v": 1} and same(plain, x.gather())


def test_the_positions_of_what_is_not_zero_have_a_length_that_is_not_known():
    """np.flatnonzero, np.nonzero and np.argwhere are nodes of shape (?,) and (?, n); an index of them stays pending as x[mask] does."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 5))
    x = lazy.asarray(data) + 0
    plain = data + 0
    mask, plain_mask = x > 0, plain > 0
    graph.COUNTS.reset()
    flat, both, where = numpy.flatnonzero(mask[:, 0]), numpy.nonzero(mask), numpy.argwhere(mask)
    assert repr(flat) == "LazyArray(int64, shape=(?,), pending)" and repr(where) == "LazyArray(int64, shape=(?, 2), pending)"
    assert type(both) is tuple and [repr(each) for each in both] == [repr(flat)] * 2
    assert flat._node.kind == rules.NONZERO and both[1]._node.name == "nonzero[1]"
    cases = {
        "flatnonzero": (flat, numpy.flatnonzero(plain_mask[:, 0])),
        "flatnonzero of two dimensions": (numpy.flatnonzero(mask), numpy.flatnonzero(plain_mask)),
        "flatnonzero of numbers": (numpy.flatnonzero(numpy.trunc(x)), numpy.flatnonzero(numpy.trunc(plain))),
        "nonzero, rows": (both[0], numpy.nonzero(plain_mask)[0]), "nonzero, columns": (both[1], numpy.nonzero(plain_mask)[1]),
        "the method": ((x > 0.5).nonzero()[1], (plain > 0.5).nonzero()[1]),
        "argwhere": (where, numpy.argwhere(plain_mask)),
        "through the stand-in": (lazy.flatnonzero(x[0] < 0), numpy.flatnonzero(plain[0] < 0)),
        "arithmetic and a reduction": ((flat + 1).max(), (numpy.flatnonzero(plain_mask[:, 0]) + 1).max()),
        "rows by positions": (x[flat], plain[numpy.flatnonzero(plain_mask[:, 0])]),
        "rows by positions, and a slice": (x[flat, 1:], plain[numpy.flatnonzero(plain_mask[:, 0]), 1:]),
        "one row by positions": (x[2, numpy.flatnonzero(mask[2])], plain[2, numpy.flatnonzero(plain_mask[2])]),
        "elements by nonzero": (x[both], plain[numpy.nonzero(plain_mask)]),
        "the same length as the mask selects": (x[both] * 2 - x[mask], plain[numpy.nonzero(plain_mask)] * 2 - plain[plain_mask]),
        "nothing is not zero": (x[numpy.flatnonzero(x[:, 0] > 99)].sum(axis=0), plain[numpy.flatnonzero(plain[:, 0] > 99)].sum(axis=0)),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.ndim == expected.ndim and got.dtype == expected.dtype, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name

    def program(np, start):
        """Write through the positions of what is not zero: a number, what was read through them, and a known array."""
        p = np.array(start)
        low = np.flatnonzero(p[:, 0] < 0)
        p[low, 0] = 0.0
        high = np.flatnonzero(p[:, 1] > 0)
        p[high] = p[high] * 3
        cells = np.nonzero(p > 2)
        p[cells] = 2.0
        deferred = graph.COUNTS.at_once == {}
        some = np.flatnonzero(p[:, 2] > -99)
        p[some, 4] = np.arange(6.0)
        return p, deferred

    graph.COUNTS.reset()
    got, deferred = program(lazy, data)
    assert deferred and graph.COUNTS.at_once == {"x[index] = v": 1} and same(program(numpy, data)[0], got.gather())

    # Asking for the length is asking for a value, as for what a mask selects.
    graph.COUNTS.reset()
    late = numpy.flatnonzero(x[0] > 0)
    assert len(late) == len(numpy.flatnonzero(plain[0] > 0)) and [int(i) for i in late] == list(numpy.flatnonzero(plain[0] > 0))
    assert graph.COUNTS.needed["The shape of what a mask selects"] == 1 and graph.COUNTS.at_once == {}
    hard = strict.asarray(data) + 0
    assert isinstance(raised(lambda: len(numpy.flatnonzero(hard[0] > 0))), NotGathered)
    assert same(numpy.flatnonzero(plain[0] > 0), numpy.flatnonzero(hard[0] > 0).gather())

    # Where NumPy's answer depends on the number, the call is made at once and is NumPy's.
    at_once = {
        "positions of two masks in one index": (x[numpy.flatnonzero(x[:, 0] > -99), numpy.flatnonzero(x[0] > -99)[:1]], plain[numpy.arange(6), :1].ravel()),
        "positions beside a known array": (x[numpy.flatnonzero(x[:, 0] > -99), numpy.zeros(6, int)], plain[:, 0]),
        "part of the positions": (numpy.flatnonzero(mask)[:3], numpy.flatnonzero(plain_mask)[:3]),
        "of what a pending mask selects": (numpy.flatnonzero(x[x > 0] > 1), numpy.flatnonzero(plain[plain > 0] > 1)),
        "of a number": (numpy.flatnonzero(x.sum()), numpy.flatnonzero(plain.sum())),
        "np.where with one argument": (numpy.where(mask)[0], numpy.where(plain_mask)[0]),
    }
    for name, (got, expected) in at_once.items():
        assert not got._node.pending and same(expected, got.gather()), name
    for wrong in (lambda a: numpy.nonzero(a.sum()), lambda a: a.sum().nonzero(), lambda a: numpy.nonzero(a, 1)):
        error, expected = raised(lambda: wrong(x)), raised(lambda: wrong(plain))
        assert type(error) is type(expected) and str(error) == str(expected)
    # A pending mask beside an array in one index: whether they fit depends on the mask, so NumPy makes the write.
    def beside(np):
        """Assign through a mask that selects three rows and an array of three columns."""
        p = np.array(data)
        p[p[:, 0] > numpy.sort(data[:, 0])[2], numpy.array([0, 1, 2])] = -1.0
        return p

    assert same(beside(numpy), beside(lazy).gather())
    beyond = x[:3][numpy.flatnonzero(x[:, 0] > -99)]
    assert beyond._node.pending and str(raised(beyond.gather)) == str(raised(lambda: plain[:3][numpy.flatnonzero(plain[:, 0] > -99)]))


def test_positions_and_ravel_are_deferred():
    """np.argmax, np.argmin and ravel join the graph; a row named by a pending position stays NumPy's view."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 5))
    x = lazy.asarray(data) + 0
    plain = data + 0
    graph.COUNTS.reset()
    cases = {
        "argmax": (numpy.argmax(x), numpy.argmax(plain)), "argmin along an axis": (numpy.argmin(x, axis=1), numpy.argmin(plain, axis=1)),
        "methods": (x.argmax(axis=0) + x.argmin(0), plain.argmax(axis=0) + plain.argmin(0)),
        "keepdims": (numpy.argmax(x, axis=1, keepdims=True), numpy.argmax(plain, axis=1, keepdims=True)),
        "ravel": (x.ravel() * 2, plain.ravel() * 2), "np.ravel": (numpy.ravel(x.T), numpy.ravel(plain.T)),
        "a row by a pending position": (x[numpy.argmax(x[:, 0])], plain[numpy.argmax(plain[:, 0])]),
        "an element by pending positions": (x[x[:, 0].argmax(), x[0].argmin()], plain[plain[:, 0].argmax(), plain[0].argmin()]),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == expected.shape and got.dtype == expected.dtype, name
        assert same(expected, got.gather()), name
    assert graph.COUNTS.at_once == {}
    assert str(raised(lambda: numpy.argmax(x, axis=(0, 1)))) == str(raised(lambda: numpy.argmax(plain, axis=(0, 1))))

    def program(np, start):
        """Take the row a position names and the flat elements, then write into the array they may be views of."""
        base = np.array(start)
        row, flat = base[np.argmax(base[:, 0])], base.ravel()
        base += 1.0
        row[0] = -5.0
        flat[1] = 7.0
        return base, row, flat

    for name, expected, got in zip(("base", "row", "flat"), program(numpy, data), program(lazy, data), strict=True):
        assert same(expected, got.gather()), name


def test_positions_that_skip_nan_are_deferred():
    """np.nanargmax and np.nanargmin join the graph as np.argmax does; a slice that is all NaN raises when it is computed."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 5))
    data[1, 2] = data[4, 0] = numpy.nan
    x = lazy.asarray(data) + 0
    plain = data + 0
    graph.COUNTS.reset()
    cases = {
        "nanargmax": (numpy.nanargmax(x), numpy.nanargmax(plain)),
        "nanargmin along an axis": (numpy.nanargmin(x, axis=1), numpy.nanargmin(plain, axis=1)),
        "an axis by position": (numpy.nanargmax(x, 0), numpy.nanargmax(plain, 0)),
        "keepdims": (numpy.nanargmin(x, axis=0, keepdims=True), numpy.nanargmin(plain, axis=0, keepdims=True)),
        "through the stand-in": (lazy.nanargmax(x * 2), numpy.nanargmax(plain * 2)),
        "a selection by pending positions": (numpy.nanargmax(x[lazy.arange(3) * 2, 0]), numpy.nanargmax(plain[numpy.arange(3) * 2, 0])),
        "what a mask selects": (numpy.nanargmin(x[x[:, 1] > 0], axis=0), numpy.nanargmin(plain[plain[:, 1] > 0], axis=0)),
        "a row it names": (x[numpy.nanargmax(x[:, 0])], plain[numpy.nanargmax(plain[:, 0])]),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == numpy.shape(expected) and got.dtype == expected.dtype, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert str(raised(lambda: numpy.nanargmax(x, axis=(0, 1)))) == str(raised(lambda: numpy.nanargmax(plain, axis=(0, 1))))
    assert str(raised(lambda: numpy.nanargmin(x, axis=2))) == str(raised(lambda: numpy.nanargmin(plain, axis=2)))

    blank = numpy.full((2, 3), numpy.nan)
    late = numpy.nanargmax(lazy.asarray(blank) + 0, axis=1)
    assert late._node.pending
    error, expected = raised(late.gather), raised(lambda: numpy.nanargmax(blank, axis=1))
    assert type(error) is type(expected) is ValueError and str(error) == str(expected)
    assert "'nanargmax'" in error.__notes__[0] and "test_lazy_prototype.py" in error.__notes__[0]


def test_the_stand_in_answers_like_the_module():
    """Arrays made through the stand-in are lazy; types, constants, submodules and ufunc methods are NumPy's."""
    assert type(lazy.zeros(3)) is LazyArray and type(lazy.array([1, 2, 3])) is LazyArray
    assert lazy.pi == numpy.pi and lazy.float32 is numpy.float32 and lazy.ndarray is numpy.ndarray
    assert lazy.newaxis is None and lazy.errstate is numpy.errstate
    assert type(lazy.sqrt(2.0)) is numpy.float64
    assert same(numpy.linalg.norm(numpy.ones(4)), lazy.linalg.norm(lazy.ones(4)).gather())
    assert same(numpy.add.reduce(numpy.arange(4.0)), lazy.add.reduce(lazy.arange(4.0)).gather())
    assert lazy.add.nin == 2 and "zeros" in dir(lazy)

    pending = lazy.arange(4.0) * 2
    assert lazy.asarray(pending) is pending and pending._node.pending
    copy = lazy.array(pending)
    assert copy is not pending and copy.gather() is not pending.gather() and same(pending.gather(), copy.gather())

    data = numpy.arange(3.0)
    assert lazy.asarray(data).gather() is data
    assert lazy.array(data).gather() is not data


def test_shape_questions_compute_nothing():
    """np.shape, np.ndim and np.size answer from the node."""
    pending = lazy.ones((3, 4)) @ lazy.ones((4, 5))
    assert numpy.shape(pending) == (3, 5) and numpy.ndim(pending) == 2 and numpy.size(pending) == 15
    assert pending.T.shape == (5, 3) and pending.reshape(5, 3).shape == (5, 3) and pending[1:, None, 0].shape == (2, 1)
    assert pending._node.pending


def test_explain_prints_a_shared_operand_once():
    """The graph of a loop is printed with each operation once, and a repeat marked."""
    w = lazy.zeros(3)
    for _ in range(3):
        w = w - 0.1 * (w * 2)
    text = w.explain()
    lines = text.splitlines()
    assert lines[0].startswith("subtract") and "test_lazy_prototype.py:" in lines[0]
    assert sum("(shown above)" in line for line in lines) == 2
    assert sum(line.strip().startswith("array") for line in lines) == 2 and len(lines) == 13
    assert w._node.pending


def test_a_long_loop_cannot_hold_an_unbounded_graph(monkeypatch):
    """Past the limit of pending nodes, everything pending is computed before another node is added."""
    gather()
    monkeypatch.setattr(graph, "PENDING_LIMIT", 10)
    x = lazy.zeros(3)
    for _ in range(25):
        x = x + 1
    assert graph.pending_count() <= 10
    assert same(numpy.full(3, 25.0), x.gather())


def test_work_nobody_holds_is_never_computed():
    """A pending array that is dropped leaves the pending list without being computed."""
    gather()
    dropped = lazy.arange(5.0) * 3
    assert graph.pending_count() == 1
    del dropped
    gc.collect()
    assert graph.pending_count() == 0


def test_gather_takes_several_and_everything():
    """gather(a, b) returns both values, passes other objects through, and with no argument computes all."""
    a, b = lazy.arange(3.0) + 1, lazy.arange(3.0) * 2
    values = gather(a, 7, b)
    assert same(numpy.arange(3.0) + 1, values[0]) and values[1] == 7 and same(numpy.arange(3.0) * 2, values[2])
    c = a + b
    assert gather() == () and not c._node.pending


def test_get_lazy_patches_nothing():
    """get_lazy() returns an object: the numpy module, its functions and other modules' imports are as they were."""
    module, zeros = sys.modules["numpy"], numpy.zeros
    stand_in = get_lazy()
    assert sys.modules["numpy"] is module and numpy.zeros is zeros and stand_in is not module
    assert type(numpy.zeros(3)) is numpy.ndarray and type(stand_in.zeros(3)) is LazyArray
    # A module that imported numpy before or after is served by the real one, and stays lazy on lazy input.
    assert attention.np is numpy and type(attention.softmax(stand_in.ones((2, 3)))) is LazyArray
    assert not callable(stand_in)


def test_a_reduction_is_called_as_it_was_written():
    """A node keeps the arguments that were given, given the same way, for a method and for a function."""
    data = numpy.random.default_rng(SEED).standard_normal((4, 5))
    x = lazy.asarray(data)
    cases = {
        "method, bare": (x.sum(), data.sum(), (), {}),
        "method, axis by position": (x.sum(0), data.sum(0), (0,), {}),
        "method, keywords": (x.mean(axis=1, keepdims=True), data.mean(axis=1, keepdims=True), (), {"axis": 1, "keepdims": True}),
        "function, bare": (numpy.sum(x), numpy.sum(data), (), {}),
        "function, axis by position": (numpy.max(x, 1), numpy.max(data, 1), (1,), {}),
        "function, keywords": (numpy.std(x, axis=0), numpy.std(data, axis=0), (), {"axis": 0}),
        "function, two axes by position": (numpy.sum(x, (0, 1)), numpy.sum(data, (0, 1)), ((0, 1),), {}),
    }
    for name, (got, expected, args, kwargs) in cases.items():
        assert got._node.pending and got._node.args[1:] == args and got._node.kwargs == kwargs, name
        assert got.shape == numpy.shape(expected) and same(expected, got.gather()), name
    assert type(raised(lambda: x.sum(0, axis=0))) is type(raised(lambda: data.sum(0, axis=0)))
    assert str(raised(lambda: x.sum(axis=2))) == str(raised(lambda: data.sum(axis=2)))


def test_count_nonzero_is_a_reduction(monkeypatch):
    """np.count_nonzero joins the graph of the masks it counts, with axes and keepdims, and its errors are NumPy's."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 5))
    x = lazy.asarray(data) + 0
    plain = data + 0
    graph.COUNTS.reset()
    cases = {
        "two masks": (numpy.count_nonzero((x[:, 0] > 0) & ~(x[:, 1] > 0)), numpy.count_nonzero((plain[:, 0] > 0) & ~(plain[:, 1] > 0))),
        "numbers": (numpy.count_nonzero(numpy.trunc(x)), numpy.count_nonzero(numpy.trunc(plain))),
        "along an axis": (numpy.count_nonzero(x > 0, axis=0), numpy.count_nonzero(plain > 0, axis=0)),
        "an axis by position": (numpy.count_nonzero(x > 0, 1), numpy.count_nonzero(plain > 0, 1)),
        "two axes": (numpy.count_nonzero(x > 0, axis=(0, 1)), numpy.count_nonzero(plain > 0, axis=(0, 1))),
        "keepdims": (numpy.count_nonzero(x > 0, axis=1, keepdims=True), numpy.count_nonzero(plain > 0, axis=1, keepdims=True)),
        "through the stand-in": (lazy.count_nonzero(x < 0), numpy.count_nonzero(plain < 0)),
        "what a mask selects": (numpy.count_nonzero(x[x[:, 0] > 0] > 0.5, axis=0), numpy.count_nonzero(plain[plain[:, 0] > 0] > 0.5, axis=0)),
        "in arithmetic": (x.sum() / numpy.count_nonzero(x > 0), plain.sum() / numpy.count_nonzero(plain > 0)),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == numpy.shape(expected) and got.dtype == expected.dtype, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert str(raised(lambda: numpy.count_nonzero(x, axis=2))) == str(raised(lambda: numpy.count_nonzero(plain, axis=2)))
    # A NumPy that answers a count without an axis with a Python int: that call is made at once, the others are not.
    monkeypatch.setattr(rules, "COUNT_IS_A_NUMPY_SCALAR", False)
    whole, along = numpy.count_nonzero(x > 0), numpy.count_nonzero(x > 0, axis=0)
    assert not whole._node.pending and along._node.pending and graph.COUNTS.at_once == {"np.count_nonzero": 1}
    assert same(numpy.count_nonzero(plain > 0), whole.gather())


def test_any_and_all_are_reductions():
    """any and all join the graph, as functions and as methods; a truth test of one is a value needed, not a call made at once."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 5))
    x = lazy.asarray(data) + 0
    plain = data + 0
    graph.COUNTS.reset()
    cases = {
        "any": (numpy.any(x > 1), numpy.any(plain > 1)), "all": (numpy.all(x > -9), numpy.all(plain > -9)),
        "methods": ((x > 1).any() | (x > 0).all(), (plain > 1).any() | (plain > 0).all()),
        "along an axis": (numpy.any(x > 1, axis=1), numpy.any(plain > 1, axis=1)),
        "an axis by position": ((x > 0).all(0), (plain > 0).all(0)),
        "two axes and keepdims": (numpy.all(x > 0, axis=(0, 1), keepdims=True), numpy.all(plain > 0, axis=(0, 1), keepdims=True)),
        "numbers": (numpy.trunc(x).any(axis=0), numpy.trunc(plain).any(axis=0)),
        "a condition of np.where": (numpy.where((x > 1).any(axis=1), x[:, 0], x[:, 1]), numpy.where((plain > 1).any(axis=1), plain[:, 0], plain[:, 1])),
        "beside another mask": ((x > 0).all(axis=0) & (x[0] > 0), (plain > 0).all(axis=0) & (plain[0] > 0)),
        "what a mask selects": (x[x[:, 0] > 0].any(axis=0), plain[plain[:, 0] > 0].any(axis=0)),
        "a mask that selects nothing": ((x[x[:, 0] > 99] > 0).all(), (plain[plain[:, 0] > 99] > 0).all()),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == numpy.shape(expected) and got.dtype == expected.dtype, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert str(raised(lambda: numpy.any(x, axis=2))) == str(raised(lambda: numpy.any(plain, axis=2)))
    assert str(raised(lambda: x.all(axis=2))) == str(raised(lambda: plain.all(axis=2)))
    limited = numpy.any(x > 1, where=x > 0)
    assert not limited._node.pending and same(numpy.any(plain > 1, where=plain > 0), limited.gather())

    graph.COUNTS.reset()
    found = (x > 1).any()
    assert found._node.pending and bool(found) is bool((plain > 1).any()) and (not (x > 0).all()) is (not (plain > 0).all())
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {"A truth test (if, while, and, or, not)": 2}
    hard = strict.asarray(data) + 0
    assert isinstance(raised(lambda: bool((hard > 0).all())), NotGathered) and same((plain > 0).all(), (hard > 0).all().gather())


def test_a_cast_is_deferred():
    """x.astype(t) and np.astype(x, t) join the graph; a cast that may give the array itself is made at once."""
    data = numpy.random.default_rng(SEED).standard_normal((4, 6)) * 10
    x = lazy.asarray(data) + 0
    plain = data + 0
    graph.COUNTS.reset()
    cases = {
        "to float32": (x.astype(numpy.float32), plain.astype(numpy.float32)),
        "to integers, by name": ((x * 3).astype("int32"), (plain * 3).astype("int32")),
        "to booleans": (numpy.trunc(x / 5).astype(bool), numpy.trunc(plain / 5).astype(bool)),
        "to its own type": (x.astype(float), plain.astype(float)),
        "the type by keyword": (x.astype(dtype=numpy.int64), plain.astype(dtype=numpy.int64)),
        "copy=False to another type": (x.astype(numpy.float32, copy=False), plain.astype(numpy.float32, copy=False)),
        "defaults spelled out": (x.astype(numpy.int16, casting="unsafe", order="K"), plain.astype(numpy.int16, casting="unsafe", order="K")),
        "np.astype": (numpy.astype(x, numpy.float32), numpy.astype(plain, numpy.float32)),
        "at the end of an expression": (numpy.exp(-x * numpy.log(2) / 3).astype(numpy.float64), numpy.exp(-plain * numpy.log(2) / 3).astype(numpy.float64)),
        "a mask read on": (~(x > 0).astype(bool) & (x < 5), ~(plain > 0).astype(bool) & (plain < 5)),
        "a number": (x.sum().astype(numpy.float32), plain.sum().astype(numpy.float32)),
        "what a mask selects": (x[x[:, 0] > 0].astype(numpy.float32).sum(axis=0), plain[plain[:, 0] > 0].astype(numpy.float32).sum(axis=0)),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == numpy.shape(expected) and got.dtype == expected.dtype, name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert x.astype(numpy.float32, copy=False)._node.kwargs == {"copy": False}
    assert x.astype(float).gather() is not x.gather()

    graph.COUNTS.reset()
    at_once = {
        "copy=False to its own type": (x.astype(float, copy=False), plain.astype(float, copy=False)),
        "another casting": (x.astype(numpy.float32, casting="same_kind"), plain.astype(numpy.float32, casting="same_kind")),
        "another order": (x.astype(numpy.float32, order="F"), plain.astype(numpy.float32, order="F")),
        "to strings": (x.astype(str), plain.astype(str)),
        "np.astype of a number": (numpy.astype((x * 2).sum(), numpy.float32), (plain * 2).sum().astype(numpy.float32)),
    }
    for name, (got, expected) in at_once.items():
        assert not got._node.pending and same(expected, got.gather()), name
    assert graph.COUNTS.at_once == {"ndarray.astype": 4, "np.astype": 1}
    assert at_once["copy=False to its own type"][0].gather() is x.gather()
    for wrong in (lambda a: a.astype("nope"), lambda a: a.astype(numpy.int32, casting="safe"), lambda a: a.astype(float, dtype=int)):
        error, expected = raised(lambda: wrong(x)), raised(lambda: wrong(plain))
        assert type(error) is type(expected) and str(error) == str(expected)

    line = sys._getframe().f_lineno + 1
    real = (x * 1j).astype(float)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        real.gather()
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        (plain * 1j).astype(float)
    assert [str(w.message) for w in caught] == [str(w.message) for w in expected] and len(caught) == 1
    assert caught[0].filename == __file__ and caught[0].lineno == line and caught[0].category is expected[0].category


def test_partition_is_deferred():
    """np.partition joins the graph as a new array of its operand's shape and type; a kth that does not fit is NumPy's error on the line."""
    data = numpy.random.default_rng(SEED).standard_normal((7, 5))
    x = lazy.asarray(data) + 0
    plain = data + 0

    def gap(np, scores):
        """Mean gap between the best and the second best of each column."""
        top = np.partition(scores, -2, axis=0)[-2:]
        return (top[1] - top[0]).mean()

    graph.COUNTS.reset()
    cases = {
        "the last axis": (numpy.partition(x, 2), numpy.partition(plain, 2)),
        "an axis by keyword": (numpy.partition(x, -2, axis=0), numpy.partition(plain, -2, axis=0)),
        "an axis by position": (numpy.partition(x, 3, 0), numpy.partition(plain, 3, 0)),
        "several positions": (numpy.partition(x, (1, 3)), numpy.partition(plain, (1, 3))),
        "kth by keyword": (numpy.partition(x, kth=1, axis=1), numpy.partition(plain, kth=1, axis=1)),
        "no axis": (numpy.partition(x, 10, axis=None), numpy.partition(plain, 10, axis=None)),
        "defaults spelled out": (numpy.partition(x, 2, kind="introselect", order=None), numpy.partition(plain, 2, kind="introselect", order=None)),
        "integers": (numpy.partition(lazy.arange(35).reshape(7, 5) * 7 % 11, 2), numpy.partition(numpy.arange(35).reshape(7, 5) * 7 % 11, 2)),
        "through the stand-in": (lazy.partition(x * 2, 1), numpy.partition(plain * 2, 1)),
        "the two best of each column": (numpy.partition(x, -2, axis=0)[-2:], numpy.partition(plain, -2, axis=0)[-2:]),
        "the gap between them": (gap(lazy, x), gap(numpy, plain)),
    }
    for name, (got, expected) in cases.items():
        assert got._node.pending and got.shape == numpy.shape(expected) and got.dtype == expected.dtype, name
    assert numpy.partition(x, 2)._node.kind == rules.ORDERING
    for name, (got, expected) in cases.items():
        assert same(expected, got.gather()), name
    assert graph.COUNTS.at_once == {} and graph.COUNTS.needed == {}
    wrong = (lambda a: numpy.partition(a, 5), lambda a: numpy.partition(a, 1, axis=2), lambda a: numpy.partition(a, 1.5),
             lambda a: numpy.partition(a.sum(), 0), lambda a: numpy.partition(a), lambda a: numpy.partition(a, 1, kth=2),
             lambda a: numpy.partition(a, [1, 9]), lambda a: numpy.partition(a, 1, kind="quick"))
    for call in wrong:
        error, expected = raised(lambda: call(x)), raised(lambda: call(plain))
        assert type(error) is type(expected) and str(error) == str(expected)

    def program(np, start):
        """Partition into a new array and write into it and into the operand; then partition the operand in place."""
        a = np.array(start)
        parted = np.partition(a, 2)
        parted[0] = 0.0
        a += 1.0
        deferred = graph.COUNTS.at_once == {}
        a.partition(1)
        return a, parted, deferred

    graph.COUNTS.reset()
    a, parted, deferred = program(lazy, data)
    assert deferred and graph.COUNTS.at_once == {"ndarray.partition": 1}
    expected = program(numpy, data)
    assert same(expected[0], a.gather()) and same(expected[1], parted.gather())
    at_once = {"what a mask selects": (numpy.partition(x[x[:, 0] > -9], 1), numpy.partition(plain, 1)),
               "a kth to be computed": (numpy.partition(x, lazy.arange(3)[1]), numpy.partition(plain, 1))}
    for name, (got, expected) in at_once.items():
        assert not got._node.pending and same(expected, got.gather()), name


def kernel_calls(hotkernels) -> dict[str, int]:
    """Return how many calls each kernel of hotkernels has taken since it was installed.

    Args:
        hotkernels: The package, with `install()` in effect.

    Returns:
        Kernel name -> calls taken, for the kernels that took any.
    """
    return {kernel.name: kernel.calls for kernel in hotkernels.report().kernels if kernel.calls}


def test_a_gather_reaches_the_kernels_of_hotkernels():
    """With hotkernels.install() in effect, a lazy program gives the same bits and its calls reach the same kernels."""
    hotkernels = pytest.importorskip("hotkernels")
    rng = numpy.random.default_rng(SEED)
    features = rng.standard_normal((EXAMPLES, FEATURES))
    targets = features @ rng.standard_normal(FEATURES) + 0.1 * rng.standard_normal(EXAMPLES)
    plain = regression.fit(numpy, features, targets, STEPS)
    with hotkernels.enabled():
        if not hotkernels.report().installed:
            pytest.skip(f"hotkernels does not install here: {hotkernels.report().unsupported}")
        before = kernel_calls(hotkernels)
        expected = regression.fit(numpy, features, targets, STEPS)
        eager = kernel_calls(hotkernels)
        weights, intercept, losses = regression.fit(lazy, features, targets, STEPS)
        got = gather(weights, intercept, *losses)
        after = kernel_calls(hotkernels)
    for reference in (expected, plain):
        assert same(reference[0], got[0]) and same(reference[1], got[1])
        assert all(same(a, b) for a, b in zip(reference[2], got[2:], strict=True))
    taken_eagerly = {name: eager[name] - before.get(name, 0) for name in eager if eager[name] > before.get(name, 0)}
    if not taken_eagerly:
        pytest.skip("no kernel takes a call of this program on this platform")
    # On ARM64 the penalty's `(w ** 2).sum()` is the call a kernel takes, once per step. At least as
    # often, not equally often: the layer's dry run for an element type is a real call, and a kernel
    # that takes arrays of any size takes it too.
    assert all(after.get(name, 0) - eager[name] >= count for name, count in taken_eagerly.items())


def test_the_pending_set_gives_its_memory_back_after_a_large_graph():
    """A set keeps the table of its largest moment, and every look at what is pending walks it: after a graph of
    thousands of nodes the C core gives the table back, so that a later write does not pay for the graph before it.
    Without the core the table stays, since Python cannot find the set empty and clear it in one step."""
    small = sys.getsizeof(set())
    total = lazy.zeros(8)
    for step in range(5000):
        total = total + float(step)
    assert sys.getsizeof(graph._pending) > 16 * small
    expected = numpy.zeros(8)
    for step in range(5000):
        expected = expected + float(step)
    assert same(expected, total.gather())
    del total
    gc.collect()
    assert graph.pending_nodes() == []
    if graph.release is not None:
        assert sys.getsizeof(graph._pending) <= 4 * small
