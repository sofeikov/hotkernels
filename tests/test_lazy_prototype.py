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

from prototypes.lazy import LazyArray, NotGathered, gather, get_lazy, graph, lazy_report
from prototypes.lazy.programs import attention, regression

# Seed of every random array in these tests.
SEED = 0
# Examples, features and gradient steps of the regression.
EXAMPLES, FEATURES, STEPS = 2_000, 12, 30
# Tokens, width and heads of the transformer block. The feed-forward layer is four times as wide.
TOKENS, WIDTH, HEADS, WIDENING = 16, 24, 4, 4

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
    x = lazy.arange(4.0)
    for _ in range(3):
        float((x * 2).sum())
    numpy.argmax(x)
    x[0] = 9.0
    lines = [(entry.name, entry.levelname, entry.getMessage()) for entry in caplog.records if entry.name.startswith("prototypes.lazy")]
    assert [(name, level) for name, level, _ in lines] == [
        ("prototypes.lazy.cut", "DEBUG"), ("prototypes.lazy.eager", "INFO"), ("prototypes.lazy.cut", "DEBUG")]
    assert lines[0][2].startswith("needed: float() at test_lazy_prototype.py:")
    assert lines[1][2].startswith("eager: np.argmax at test_lazy_prototype.py:")
    assert lines[2][2].startswith("write: x[index] = v at test_lazy_prototype.py:")
    assert graph.COUNTS.needed == {"float()": 3} and graph.COUNTS.at_once == {"np.argmax": 1, "x[index] = v": 1}


def test_at_info_only_eager_calls_are_logged(caplog):
    """With the package's logger at INFO, a line says what NumPy had to compute at once and nothing else."""
    caplog.set_level(logging.INFO, logger="prototypes.lazy")
    x = lazy.arange(4.0)
    float((x * 2).sum())
    numpy.argmin(x)
    x[1] = 9.0
    lines = [entry.getMessage() for entry in caplog.records if entry.name.startswith("prototypes.lazy")]
    assert len(lines) == 1 and lines[0].startswith("eager: np.argmin at ")


def test_the_report_ranks_what_was_computed_unasked():
    """lazy_report() has every occurrence by call and by line, the most frequent first, in three parts."""
    lazy_report(reset=True)
    x = lazy.arange(4.0)
    for _ in range(3):
        numpy.argmax(x * 2)
    numpy.cumsum(x)
    x[0] = float((x + 1).sum())
    found = lazy_report()
    assert [(row.call, row.count) for row in found.eager] == [("np.argmax", 3), ("np.cumsum", 1)]
    assert found.eager[0].places[0][1] == 3 and found.eager[0].places[0][0].startswith("test_lazy_prototype.py:")
    assert [(row.call, row.count) for row in found.writes] == [("x[index] = v", 1)]
    assert [(row.call, row.count) for row in found.needed] == [("float()", 1)]
    text = str(found).splitlines()
    assert text[0] == "Made at once by NumPy: not in the deferred set"
    assert text[1].split()[:2] == ["3", "np.argmax"] and text[1].endswith("× 3") and text[2].split()[:2] == ["1", "np.cumsum"]
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
    target = lazy.zeros(2)
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


def test_other_numpy_calls_are_made_at_once_and_stay_lazy():
    """A call outside the deferred set is NumPy's, on gathered operands, and its result is a computed lazy array."""
    data = numpy.random.default_rng(SEED).standard_normal(50)
    x = lazy.asarray(data)
    cases = {
        "mask": (x[x > 0], data[data > 0]),
        "norm": (numpy.linalg.norm(x * 2), numpy.linalg.norm(data * 2)),
        "cumsum": (x.cumsum(), data.cumsum()),
        "argmax": (numpy.argmax(-x), numpy.argmax(-data)),
        "concatenate": (numpy.concatenate([x, data]), numpy.concatenate([data, data])),
        "dot of three dimensions": (numpy.dot(x.reshape(2, 5, 5), x.reshape(2, 5, 5)), numpy.dot(data.reshape(2, 5, 5), data.reshape(2, 5, 5))),
        "sort": (numpy.sort(x), numpy.sort(data)),
    }
    for name, (got, expected) in cases.items():
        assert type(got) is LazyArray and not got._node.pending, name
        assert same(expected, got.gather()), name


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
    """x[rows], x[[0, 2]], x[rows, columns] have a shape that needs no value; a mask or a pending index does not."""
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
    assert not x[x > 0]._node.pending and not x[lazy.arange(3) * 2]._node.pending
    assert same(plain[numpy.arange(3) * 2], x[lazy.arange(3) * 2].gather())


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
    }
    for name, (got, expected, args, kwargs) in cases.items():
        assert got._node.pending and got._node.args[1:] == args and got._node.kwargs == kwargs, name
        assert got.shape == numpy.shape(expected) and same(expected, got.gather()), name
    assert type(raised(lambda: x.sum(0, axis=0))) is type(raised(lambda: data.sum(0, axis=0)))
    assert str(raised(lambda: x.sum(axis=2))) == str(raised(lambda: data.sum(axis=2)))


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
