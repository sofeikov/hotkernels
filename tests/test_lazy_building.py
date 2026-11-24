"""Tests of what the lazy layer's prototype remembers while it writes a graph down.

Writing a node asks questions whose answers are the same on every turn of a
loop: the shape and element type of the call, the line that wrote it, the
error settings in effect, whether a write fits. The prototype remembers the
answers, and works the shape of a plain index out itself. Each test here
holds what is remembered to what is worked out from nothing, which is what
NumPy says. Each test runs twice, in Python alone and with the layer's C
core (`tests/conftest.py`), which takes the calls whose answers are
remembered. Run with `uv run --group dev pytest tests/test_lazy_building.py`.
"""
from __future__ import annotations

import gc
import operator
import random
import warnings
from typing import Any, Callable

import numpy
import pytest

from prototypes.lazy import LazyArray, NotGathered, array, get_lazy, graph, rules
from prototypes.lazy.programs import regression

# Seed of every random array and every random program in these tests.
SEED = 0
# Element types the brute-force tests go through.
ELEMENT_TYPES = (numpy.bool_, numpy.int8, numpy.uint8, numpy.int64, numpy.float32, numpy.float64, numpy.complex64)
# Fractional values that would show a dependence on the value if there were one: the ordinary, the
# signed zeros, the largest and smallest, the infinities and NaN.
FRACTIONS = (1.0, 0.0, -0.0, 0.5, -1.5, 1e300, -1e300, 5e-324, float("inf"), float("-inf"), float("nan"), 2.0 ** 70)
# Complex values chosen the same way.
COMPLEXES = (1j, 0j, complex("nan"), complex(1e300, -1e300), complex("inf"))
# Shapes the random indices and the random programs are tried on.
SHAPES = ((), (1,), (3,), (4,), (3, 4), (4, 3), (2, 3, 4), (1, 4), (3, 1), (4, 4), (6,))
# The length an axis is taken to have, for drawing an index, when its real length is not known.
ASSUMED_LENGTH = 4
# Random indices and random assignments tried.
INDEX_TRIALS, ASSIGNMENT_TRIALS = 4_000, 2_000
# Of the random indices NumPy accepts, the share `basic_shape` must answer rather than pass on.
ANSWERED_SHARE = 0.95
# Random programs run with and without the memories, and the steps of each.
PROGRAMS, PROGRAM_STEPS = 40, 150
# How often, out of a hundred, each kind of item is drawn for a random index: a whole number in or just out
# of bounds, a slice, `None`, `...`, and something a plain index does not hold.
INDEX_ITEMS = {"number": 35, "slice": 45, "none": 7, "ellipsis": 7, "other": 6}
# How often, out of a hundred, each kind of step is drawn for a random program.
STEPS_DRAWN = {"fresh": 5, "operator": 20, "ufunc": 8, "reduction": 10, "index": 13, "view": 8, "mask": 10, "in place": 8,
               "assign": 8, "stale view": 4, "out": 3, "errstate": 3}
# Of the programs, the share run with a strict stand-in; of the picks of an array made earlier, the share
# taken from the most recent ones, and how many those are.
STRICT_SHARE, RECENT_SHARE, RECENT = 0.1, 0.7, 8
# Of the picks of an operand, the shares that are a lazy array and a number; the rest are `ndarray`s.
LAZY_SHARE, NUMBER_SHARE = 0.6, 0.3
# Examples, features and gradient steps of the regression that is built twice.
EXAMPLES, FEATURES, STEPS = 50, 4, 20
# The width of an array for which NumPy's own limit on sizes decides the answer: one row short of 2**23, so
# that a first length of 2**40 and a little fits 63 bits and a first length of 2**40 and a little more does not.
WIDTH_AT_THE_LIMIT = 2 ** 23 - 1
# The numbers of two nodes whose marks, as first lengths of such an array, fall on either side of that limit.
MARK_THAT_FITS, MARK_THAT_DOES_NOT = 5, 200_000
# Turns of the loops that count how often NumPy is asked.
TURNS = 10

lazy = get_lazy()
# A stand-in whose arrays refuse to give their values without gather().
strict = get_lazy(strict=True)


def raised(call: Callable[[], Any]) -> BaseException:
    """Make a call that must raise, and return what it raised.

    Args:
        call: A function of no arguments.

    Returns:
        The exception.
    """
    with pytest.raises(Exception) as caught:
        call()
    return caught.value


def outcome(call: Callable[[], Any]) -> tuple:
    """Make a call and describe what happened, so that two calls can be compared.

    Args:
        call: A function of no arguments that returns a `rules.Signature` or None.

    Returns:
        ("made at once",) for None; the shape, the element type, which
        object the element type is and the kind for a signature; the type
        and message of the exception for a call that raises.
    """
    try:
        signature = call()
    except Exception as error:
        return (type(error).__name__, str(error))
    if signature is None:
        return ("made at once",)
    return (signature.shape, signature.dtype, id(signature.dtype), signature.kind)


def node_of(shape: tuple[int, ...], dtype: Any = numpy.float64) -> graph.Node:
    """Make a node that has its values: zeros of a shape and an element type.

    Args:
        shape: The shape.
        dtype: The element type.

    Returns:
        The node.
    """
    return graph.leaf(numpy.zeros(shape, dtype))


class Counted:
    """A function that counts its calls and then makes them."""

    def __init__(self, function: Callable[..., Any]) -> None:
        """Wrap a function.

        Args:
            function: The function to count the calls of.
        """
        self.function = function
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Count one call and make it.

        Args:
            *args: The call's positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            What the function returns.
        """
        self.calls += 1
        return self.function(*args, **kwargs)


def test_a_remembered_answer_is_the_one_worked_out():
    """For each kind of deferred call, the answer found again is the one NumPy and the rules give from nothing."""
    f8, f4, i1 = node_of((3, 4)), node_of((3, 4), numpy.float32), node_of((3, 4), numpy.int8)
    row, flat, mask = node_of((4,)), node_of((6,)), node_of((3, 4), numpy.bool_)
    calls = [
        (numpy.multiply, (f8, f8), {}), (numpy.multiply, (f4, 2.0), {}), (numpy.multiply, (2, f4), {}), (numpy.add, (f4, f8), {}),
        (numpy.add, (i1, numpy.float32(2)), {}), (numpy.add, (i1, numpy.int64(2)), {}), (numpy.add, (f8, numpy.ones(4)), {}),
        (numpy.matmul, (f8, row), {}), (numpy.power, (i1, 2), {}), (numpy.negative, (i1,), {}), (numpy.greater, (f8, 0), {}),
        (numpy.where, (mask, f8, 0.0), {}), (numpy.dot, (f8, row), {}), (numpy.copy, (f8,), {}),
        (numpy.sum, (f8,), {}), (numpy.sum, (i1, 0), {}), (numpy.mean, (f4,), {"axis": -1, "keepdims": True}),
        (numpy.max, (f8, (0, 1)), {}), (numpy.argmax, (f8,), {"axis": 1}), (numpy.count_nonzero, (mask,), {"axis": 0}),
        (numpy.any, (mask,), {}), (numpy.cumsum, (i1,), {}), (numpy.cumsum, (f8, 1), {}),
        (numpy.searchsorted, (flat, f8), {}), (numpy.searchsorted, (flat, 0.5, "right"), {}),
        (numpy.transpose, (f8,), {}), (numpy.transpose, (f8, (1, 0)), {}), (numpy.swapaxes, (f8, 0, 1), {}),
        (numpy.reshape, (f8, (4, 3)), {}), (numpy.reshape, (f8, (2, -1)), {}), (numpy.ravel, (f8,), {}),
        (numpy.expand_dims, (f8, 0), {}), (numpy.squeeze, (node_of((1, 4)),), {"axis": 0}),
        (numpy.round, (f8, 2), {}), (numpy.ndarray.astype, (f8, numpy.float32), {}), (numpy.ndarray.astype, (f8, "int64"), {}),
        (numpy.ndarray.astype, (f8,), {"dtype": float, "copy": True}), (numpy.partition, (f8, 1), {"axis": 0}),
    ]
    for operation, args, kwargs in calls:
        rules._plans.clear()
        worked_out = outcome(lambda: rules._plan(operation, args, kwargs))
        first, again = rules.plan(operation, args, kwargs), rules.plan(operation, args, kwargs)
        name = (operation.__name__, args[1:], kwargs)
        assert worked_out[0] != "made at once" and again is first, name
        assert outcome(lambda: first)[:2] == worked_out[:2] and first.kind == worked_out[3], name
        assert first.spec == graph.spec_of(first.shape, first.dtype) != 0, name


def test_calls_that_python_finds_equal_are_told_apart():
    """An answer is never given to a call that differs in something NumPy looks at, though Python finds the two equal."""
    f8, u1, whole = node_of((3, 4)), node_of((3,), numpy.uint8), node_of((3,), numpy.int64)
    pairs = [
        # True is 1 and 1.0 is 1 to Python; not to NumPy, as an axis or as `keepdims`.
        ((numpy.sum, (f8, 1), {}), (numpy.sum, (f8, True), {})),
        ((numpy.sum, (f8,), {"axis": (0, 1)}), (numpy.sum, (f8,), {"axis": (0.0, 1)})),
        ((numpy.swapaxes, (f8, 1, 0), {}), (numpy.swapaxes, (f8, 1.0, 0), {})),
        ((numpy.sum, (f8,), {"keepdims": 1.0}), (numpy.sum, (f8,), {"keepdims": 0.0})),
        # A whole number counts by its value: these raise where their neighbours do not.
        ((numpy.add, (u1, 1), {}), (numpy.add, (u1, 300), {})),
        ((numpy.power, (whole, 2), {}), (numpy.power, (whole, -1), {})),
        ((numpy.power, (whole, numpy.int64(2)), {}), (numpy.power, (whole, numpy.int64(-1)), {})),
        ((numpy.add, (u1, True), {}), (numpy.add, (u1, 1), {})),
        # The type of a number counts, and of a NumPy scalar.
        ((numpy.multiply, (u1, 2), {}), (numpy.multiply, (u1, 2.0), {})),
        ((numpy.multiply, (node_of((3,), numpy.float32), 2.0), {}), (numpy.multiply, (node_of((3,), numpy.float32), numpy.float64(2.0)), {})),
        # A node is not an `ndarray`: a reduction of an `ndarray` is NumPy's own business.
        ((numpy.sum, (f8,), {}), (numpy.sum, (numpy.zeros((3, 4)),), {})),
        # One element type under two byte orders.
        ((numpy.negative, (node_of((3,), numpy.dtype("<i4")),), {}), (numpy.negative, (node_of((3,), numpy.dtype(">i4")),), {})),
    ]
    rules._plans.clear()
    for first, second in pairs:
        for operation, args, kwargs in (first, second, first):
            remembered = outcome(lambda: rules.plan(operation, args, kwargs))
            worked_out = outcome(lambda: rules._plan(operation, args, kwargs))
            assert remembered[:2] == worked_out[:2] and remembered[3:] == worked_out[3:], (operation.__name__, args[1:], kwargs)
    assert outcome(lambda: rules.plan(numpy.add, (u1, 300), {}))[0] == "OverflowError"
    assert outcome(lambda: rules.plan(numpy.swapaxes, (f8, 1.0, 0), {}))[0] == "TypeError"


def test_an_element_type_that_only_equals_another_keeps_what_it_carries():
    """Two `dtype` objects that compare equal are remembered apart, so an answer carries the metadata of its own operand."""
    plain, labelled = numpy.dtype("float64"), numpy.dtype("float64", metadata={"unit": "m"})
    assert plain == labelled and plain is not labelled
    first, second = node_of((3,), plain), node_of((3,), labelled)
    assert graph.spec(first) != graph.spec(second)
    rules._plans.clear()
    for node in (first, second, first, second):
        for operation, args in ((numpy.negative, (node,)), (numpy.transpose, (node,)), (numpy.add, (node, node))):
            remembered, worked_out = rules.plan(operation, args, {}), rules._plan(operation, args, {})
            assert remembered.dtype.metadata == worked_out.dtype.metadata, (operation.__name__, node.dtype.metadata)
    assert rules.plan(numpy.transpose, (second,), {}).dtype is labelled


def test_the_value_of_a_fraction_never_changes_an_answer():
    """For every ufunc of NumPy, operand type and place, a float or complex operand's value changes neither the answer nor the error.

    That is what lets `x * 0.01 * step` be remembered as `x * float`. For a
    whole number the value does count, which `test_calls_that_python_finds_equal_are_told_apart` holds.
    """
    ufuncs = sorted({item for item in vars(numpy).values() if isinstance(item, numpy.ufunc) and item.nin > 1}, key=lambda item: item.__name__)
    scalars = [kind(value) for kind in (numpy.float16, numpy.float32, numpy.float64, numpy.longdouble) for value in (1.0, -1.5, float("inf"), float("nan"))]
    scalars += [kind(value) for kind in (numpy.complex64, numpy.complex128) for value in (1j, complex("nan"), complex("inf"))]
    nodes = [node_of(shape, dtype) for dtype in ELEMENT_TYPES for shape in ((2,), ())]
    checked = 0
    assert len(ufuncs) > 30
    for ufunc in ufuncs:
        for node in nodes:
            for place in range(ufunc.nin):
                seen: dict[type, tuple] = {}
                for value in (*FRACTIONS, *COMPLEXES, *scalars):
                    args = tuple(value if position == place else node for position in range(ufunc.nin))
                    answer = outcome(lambda: rules._plan(ufunc, args, {}))
                    # One answer per type of the number, whatever its value.
                    assert seen.setdefault(type(value), answer) == answer, (ufunc.__name__, node.dtype, place, value)
                    assert rules.operand_part(value) is type(value)
                    checked += 1
    assert checked > 30_000


def random_index(rng: random.Random, shape: tuple[int, ...]) -> Any:
    """Draw an index for an array: mostly whole numbers and slices, in and out of bounds, sometimes what is neither.

    Args:
        rng: The generator to draw from.
        shape: The shape of the array to be indexed.

    Returns:
        One item, or a tuple of up to four.
    """
    def item(length: int) -> Any:
        """Draw one item of an index, for an axis of a given length."""
        if length >= graph.UNKNOWN_FLOOR:
            length = ASSUMED_LENGTH  # A stand-in differs from run to run, and the draws must not.
        bounds = (None, 0, 1, 2, -1, -2, length, length + 3, -length - 2)
        kind = rng.choices(list(INDEX_ITEMS), list(INDEX_ITEMS.values()))[0]
        return {"number": lambda: rng.randint(-length - 1, length),
                "slice": lambda: slice(rng.choice(bounds), rng.choice(bounds), rng.choice((None, None, 1, 2, -1, -3, 0))),
                "none": lambda: None,
                "ellipsis": lambda: Ellipsis,
                "other": lambda: rng.choice((True, numpy.int64(0), [0], slice(numpy.int64(0), None), numpy.array([0]), 1.0))}[kind]()

    count = rng.choice((0, 1, 1, 2, 2, 3, 4))
    if count == 0:
        return item(shape[0] if shape else 1)
    return tuple(item(shape[position] if position < len(shape) else 2) for position in range(count))


def test_the_shape_of_a_plain_index_is_numpys():
    """`basic_shape` gives NumPy's shape wherever it answers, and passes on every index NumPy raises for."""
    rng = random.Random(SEED)
    valid = answered = 0
    for _ in range(INDEX_TRIALS):
        shape = rng.choice(SHAPES)
        index = random_index(rng, shape)
        ours = rules.basic_shape(shape, index)
        items = index if isinstance(index, tuple) else (index,)
        plain = all(type(item) is int or item is None or item is Ellipsis
                    or (type(item) is slice and all(part is None or type(part) is int for part in (item.start, item.stop, item.step)))
                    for item in items)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                numpys = numpy.empty(shape, rules.NO_SIZE)[index].shape
        except (IndexError, ValueError, TypeError):
            assert ours is None, (shape, index)
            continue
        assert ours is None or (plain and ours == numpys), (shape, index, ours, numpys)
        if plain:
            valid += 1
            answered += ours is not None
    assert valid > INDEX_TRIALS // 4 and answered >= ANSWERED_SHARE * valid


def test_an_assignment_that_is_vouched_for_fits():
    """Where `broadcasts_into` says a value fits a selection, NumPy assigns it; where NumPy would raise, it does not say so."""
    rng = random.Random(SEED)
    vouched = 0
    for _ in range(ASSIGNMENT_TRIALS):
        into, value = rng.choice(SHAPES), rng.choice(SHAPES)
        target = numpy.empty(into, rules.NO_SIZE)
        try:
            target[...] = numpy.broadcast_to(numpy.zeros((), rules.NO_SIZE), value)
            fits = True
        except ValueError:
            fits = False
        assert fits or not rules.broadcasts_into(value, into), (value, into)
        vouched += rules.broadcasts_into(value, into)
    assert vouched > ASSIGNMENT_TRIALS // 10
    assert not rules.broadcasts_into((graph.UNKNOWN_FLOOR + 5,), (3,))


def test_a_loop_asks_numpy_once(monkeypatch):
    """A program that runs a second time is written down without one question to NumPy, and gives the same results."""
    rng = numpy.random.default_rng(SEED)
    features, targets = rng.standard_normal((EXAMPLES, FEATURES)), rng.standard_normal(EXAMPLES)
    rules._plans.clear()
    counted = Counted(rules._plan)
    monkeypatch.setattr(rules, "_plan", counted)
    graph.COUNTS.reset()
    first = regression.fit(lazy, features, targets, STEPS)
    operations, asked = graph.COUNTS.deferred, counted.calls
    # Each line of the loop asks on its first turn. The intercept is a Python
    # number on the first turn and a lazy one afterwards, so two lines ask twice.
    assert operations > 15 * STEPS and 0 < asked < 40
    second = regression.fit(lazy, features, targets, STEPS)
    assert counted.calls == asked and graph.COUNTS.deferred == 2 * operations
    eager = regression.fit(numpy, features, targets, STEPS)
    for expected, one, other in zip((eager[0], eager[1], *eager[2]), (first[0], first[1], *first[2]), (second[0], second[1], *second[2])):
        assert numpy.asarray(expected).tobytes() == numpy.asarray(one.gather()).tobytes() == numpy.asarray(other.gather()).tobytes()


def test_a_length_that_is_not_known_keeps_its_own_mark(monkeypatch):
    """What one mask selects is remembered for the next mask, and each result carries the mark of its own mask."""
    data = numpy.random.default_rng(SEED).standard_normal((30, 4))
    x = lazy.asarray(data)
    rules._plans.clear()
    counted = Counted(rules._plan)
    monkeypatch.setattr(rules, "_plan", counted)
    asked = []
    for turn in range(TURNS):
        threshold = -1.0 + 0.2 * turn
        mask = x[:, 0] > threshold
        chosen = x[mask]
        mark = graph.unknown_length(mask._node)
        doubled, mean, both, column = chosen * 2.0, chosen.mean(axis=0), chosen + chosen, chosen[:, 1]
        assert chosen._node.shape == doubled._node.shape == both._node.shape == (mark, 4) and column._node.shape == (mark,)
        assert mean._node.shape == (4,) and all(each._node.pending for each in (doubled, mean, both, column))
        plain = data[data[:, 0] > threshold]
        for expected, got in ((plain * 2.0, doubled), (plain.mean(axis=0), mean), (plain + plain, both), (plain[:, 1], column)):
            assert expected.tobytes() == got.gather().tobytes() and expected.shape == got.shape
        asked.append(counted.calls)
    # Every question was asked on the first turn.
    assert asked[0] > 0 and asked[-1] == asked[0]
    # A call that brings two lengths that are not known, and whose answer has the second: each time, that call's own.
    column = data[:, 0]
    for low, high in ((-0.5, 0.2), (0.0, 0.6), (-1.0, 1.0)):
        ordered = lazy.asarray(numpy.sort(column))
        table_mask, sought_mask = ordered > low, x[:, 1] > high
        places = numpy.searchsorted(ordered[table_mask], x[:, 1][sought_mask])
        assert places._node.pending and places._node.shape == (graph.unknown_length(sought_mask._node),)
        expected = numpy.searchsorted(numpy.sort(column)[numpy.sort(column) > low], data[:, 1][data[:, 1] > high])
        assert expected.tobytes() == places.gather().tobytes()
    # Two selections of different lengths do not go together, remembered or not: NumPy's error, on the line.
    few, many = x[x[:, 0] > 0.5], x[x[:, 0] > -0.5]
    assert (few + few)._node.pending and (many + many)._node.pending
    assert str(raised(lambda: few + many)) == str(raised(lambda: data[data[:, 0] > 0.5] + data[data[:, 0] > -0.5]))


def test_an_answer_that_could_depend_on_the_mask_is_not_remembered():
    """Where NumPy's limit on sizes makes the answer depend on which mask it is, each mask's answer is worked out.

    A mark is 2**40 plus its node's number, and NumPy refuses a size past 63
    bits. For an array nearly 2**23 wide the product passes that limit for
    some marks and not for others, so the answer for one is not the answer
    for all, and `plan` keeps none.
    """
    def selection(mark: int) -> graph.Node:
        """Make up a node of a selection with a given mark and the width at the limit. It is never computed."""
        return graph.Node(0, "made up", None, (), {}, (graph.UNKNOWN_FLOOR + mark, WIDTH_AT_THE_LIMIT), numpy.dtype(numpy.float64))

    fits, does_not = selection(MARK_THAT_FITS), selection(MARK_THAT_DOES_NOT)
    rules._plans.clear()
    for node in (fits, does_not, fits, does_not):
        assert outcome(lambda: rules.plan(numpy.add, (node, node), {})) == outcome(lambda: rules._plan(numpy.add, (node, node), {}))
    assert rules.plan(numpy.add, (fits, fits), {}) is not None and rules.plan(numpy.add, (does_not, does_not), {}) is None
    if array.core_in_use():
        # The C core finds a remembered answer by the same rule, and so finds none here either.
        assert all(array._core.plan(numpy.add, (node, node), {}) is array._core.MISS for node in (fits, does_not))
    # A narrow selection is remembered: the second mask's answer is found, with its own mark.
    narrow = [graph.Node(0, "made up", None, (), {}, (graph.UNKNOWN_FLOOR + mark, 4), numpy.dtype(numpy.float64)) for mark in (7, 9)]
    before = len(rules._plans)
    answers = [rules.plan(numpy.add, (node, node), {}) for node in narrow]
    assert len(rules._plans) == before + 1 and [answer.shape for answer in answers] == [node.shape for node in narrow]
    if array.core_in_use():
        assert [array._core.plan(numpy.add, (node, node), {})[0] for node in narrow] == [node.shape for node in narrow]


def test_every_road_finds_the_line_that_wrote_the_operation():
    """An operator, a function of the real module, one of the stand-in, a method, an index and a write each name their own line."""
    data = numpy.random.default_rng(SEED).standard_normal((3, 4))
    x = lazy.asarray(data) + 0
    target = lazy.zeros((3, 4))
    roads = [
        lambda: x * 2,
        lambda: 2 - x,
        lambda: -x,
        lambda: numpy.exp(x),
        lambda: lazy.exp(x),
        lambda: numpy.sum(x, axis=0),
        lambda: lazy.sum(x, axis=0),
        lambda: x.sum(),
        lambda: x.mean(axis=1, keepdims=True),
        lambda: x.T,
        lambda: x.reshape(4, 3),
        lambda: x.transpose(1, 0),
        lambda: x.swapaxes(0, 1),
        lambda: x.ravel(),
        lambda: x.reshape((2, 6)),
        lambda: x.reshape(-1),
        lambda: x.transpose(),
        lambda: x.transpose((1, 0)),
        lambda: x[1],
        lambda: x[1, 1:3],
        lambda: x[numpy.int64(1)],
        lambda: x[x[:, 0] > 0],
        lambda: numpy.where(x > 0, x, 0.0),
        lambda: data * x,
        lambda: operator.iadd(target, 1.0),
        lambda: operator.setitem(target, 0, 2.0) or target,
        lambda: operator.setitem(target, (slice(None), 1), x[:, 0]) or target,
        lambda: numpy.add(x, x, out=target),
    ]
    for _ in range(2):  # The second time every line is found in the memory.
        for road in roads:
            made = road()
            origin = made._node.origin
            assert made._node.pending and origin.lineno == road.__code__.co_firstlineno, road.__code__.co_firstlineno
            assert origin.filename == __file__ and origin.module == __name__ and origin.registry is globals()["__warningregistry__"]


def here() -> graph.Origin:
    """Return the line this function is written on, as the layer finds it."""
    return graph.caller()


def test_a_line_is_found_again_and_checked_against_its_module():
    """A line found before is the same object, until its module's name or its record of warnings is another."""
    first, again = here(), here()
    assert again is first and first.filename == __file__ and first.module == __name__
    kept = globals()["__warningregistry__"]
    try:
        globals()["__warningregistry__"] = replaced = {}
        after = here()
        assert after is not first and after.registry is replaced and after.lineno == first.lineno
    finally:
        globals()["__warningregistry__"] = kept
    assert here().registry is kept
    # The same on the road an operator takes, which with the C core is the core's own look in the memory.
    x = lazy.arange(3.0)

    def written() -> graph.Origin:
        """Write one operation and return the line it names."""
        return (x + 1)._node.origin

    before = [written(), written()]
    try:
        globals()["__warningregistry__"] = replaced = {}
        assert written().registry is replaced and before[0] is before[1] and before[0].registry is kept
    finally:
        globals()["__warningregistry__"] = kept
    assert written().registry is kept
    operation = compile("y = x + 1", "<two modules, one line>", "exec")
    for names in ({"x": x, "__name__": "one"}, {"x": x, "__name__": "two"}, {"x": x, "__name__": "one"}):
        exec(operation, names)
        origin = names["y"]._node.origin
        assert origin.module == names["__name__"] and origin.registry is names["__warningregistry__"]
    # One code object under two sets of globals: each gets its own module's name and record.
    code = compile("origin = caller()", "<two modules>", "exec")
    one, two = {"caller": graph.caller, "__name__": "one"}, {"caller": graph.caller, "__name__": "two"}
    for names in (one, two, one):
        exec(code, names)
        assert names["origin"].module == names["__name__"] and names["origin"].registry is names["__warningregistry__"]


def test_error_settings_are_remembered_until_they_change(monkeypatch):
    """Nodes written under the same `np.errstate` share its record, and a change is seen by the next node."""
    x = lazy.arange(3.0) + 1
    outside = numpy.geterr()
    first, second = x + 1, x + 2
    assert first._node.errstate == outside and second._node.errstate is first._node.errstate
    with numpy.errstate(divide="ignore", over="raise"):
        inside = numpy.geterr()
        within, again = x / 0, x / 0
        assert within._node.errstate == inside != outside and again._node.errstate is within._node.errstate
        with numpy.errstate(invalid="raise"):
            assert (x * 2)._node.errstate == numpy.geterr() != inside
        assert (x * 2)._node.errstate == inside
    assert (x + 3)._node.errstate == outside == numpy.geterr()
    previous = numpy.seterr(under="raise")
    try:
        assert (x + 4)._node.errstate == numpy.geterr() != outside
    finally:
        numpy.seterr(**previous)
    # On a NumPy that keeps no record this layer knows of, NumPy is asked every time.
    monkeypatch.setattr(graph, "_error_settings", graph._NoRecord)
    with numpy.errstate(divide="ignore"):
        assert (x / 0)._node.errstate == numpy.geterr() != outside
    assert (x + 5)._node.errstate == outside


def test_a_write_that_fits_is_asked_about_once(monkeypatch):
    """An in-place operator in a loop asks NumPy once whether its types fit; a write that does not fit raises every time."""
    counted = Counted(array._types_fit)
    monkeypatch.setattr(array, "_types_fit", counted)
    array._fitting.clear()
    x, plain = lazy.zeros((3, 4)), numpy.zeros((3, 4))
    for turn in range(TURNS):
        x += 0.5 * turn
        x *= numpy.float64(1.5)
        plain += 0.5 * turn
        plain *= numpy.float64(1.5)
    assert counted.calls == 2 and x._node.pending and plain.tobytes() == x.gather().tobytes()
    small, plain_small = lazy.zeros(3, dtype=numpy.uint8), numpy.zeros(3, dtype=numpy.uint8)
    small += 1
    for _ in range(2):
        assert str(raised(lambda: operator.iadd(small, 300))) == str(raised(lambda: operator.iadd(plain_small, 300)))
        assert type(raised(lambda: operator.iadd(small, 1.5))) is type(raised(lambda: operator.iadd(plain_small, 1.5)))
    # What is written from a strict array is strict, the second time as the first.
    whole, part = strict.arange(3.0) + 0, strict.arange(2.0) + 0
    for _ in range(3):
        target, assigned = lazy.zeros(3), lazy.zeros(3)
        target += 0.0
        assigned[0] = 0.0
        assert not target._node.strict and not assigned._node.strict
        target += whole
        assigned[1:] = part
        assert target._node.pending and target._node.strict and assigned._node.pending and assigned._node.strict
        assert isinstance(raised(lambda: float(target.sum())), NotGathered)
        assert target.gather().tobytes() == numpy.arange(3.0).tobytes() and assigned.gather().tobytes() == numpy.array([0.0, 0.0, 1.0]).tobytes()
    out, plain_out = lazy.zeros((3, 4)), numpy.zeros((3, 4))
    before = counted.calls
    for turn in range(TURNS):
        numpy.add(x, turn * 1.0, out=out)
        numpy.add(plain, turn * 1.0, out=plain_out)
    assert counted.calls == before + 1 and plain_out.tobytes() == out.gather().tobytes()


def test_a_plain_assignment_does_not_ask_numpy(monkeypatch):
    """`x[i] = v` with whole numbers and slices is deferred without a question; anything else is asked about, and errors are NumPy's."""
    counted = Counted(array._assignment_fits)
    monkeypatch.setattr(array, "_assignment_fits", counted)
    array._assignable.clear()
    data = numpy.random.default_rng(SEED).standard_normal((4,))
    x, plain = lazy.zeros((6, 4)), numpy.zeros((6, 4))
    for turn in range(6):
        for target in (x, plain):
            target[turn] = data
            target[turn, 1:3] = 1.0
            target[turn, 0] = turn
            target[turn][2:] = data[2:]
            target.T[1, turn] = 7.0
    assert counted.calls == 0 and x._node.pending and plain.tobytes() == x.gather().tobytes()
    y, plain_y = lazy.zeros((6, 4)), numpy.zeros((6, 4))
    for target in (y, plain_y):
        target[numpy.int64(1)] = 1.0  # A NumPy integer: asked about.
        target[1:][numpy.int64(0):2] = 2.0  # Through a view that is not plain: asked about.
        target[0] = numpy.ones((1, 4))  # More dimensions than the selection: NumPy takes it; asked about.
    assert counted.calls == 3 and plain_y.tobytes() == y.gather().tobytes()
    for wrong in (lambda target: operator.setitem(target, 0, numpy.ones(5)), lambda target: operator.setitem(target, 9, 1.0),
                  lambda target: operator.setitem(target, (0, slice(None, None, 0)), 1.0)):
        assert str(raised(lambda: wrong(y))) == str(raised(lambda: wrong(plain_y)))
    # A view made while a length was not known wrote that down as its shape. Once the length is known,
    # an assignment through the view is checked against the real one: by NumPy, not from what was written down.
    values = numpy.arange(24.0).reshape(6, 4)
    chosen, plain_chosen = (lazy.asarray(values) + 0)[lazy.asarray(values)[:, 0] > 7], (values + 0)[values[:, 0] > 7]
    column, plain_column = chosen[:, 1], plain_chosen[:, 1]
    assert not graph.is_known(column._steps[-1].shape) and not column._steps[-1].plain
    float(chosen.sum())  # Computes the selection, which now has four rows, and hands no array out.
    assert (chosen * 1)._node.shape == (4, 4) and chosen[0]._node.shape == (4,)  # Read with its real shape, as a loop would have.
    before = counted.calls
    column[0] = -1.0
    plain_column[0] = -1.0
    assert counted.calls == before + 1 and chosen._node.pending
    assert str(raised(lambda: operator.setitem(column, 5, 1.0))) == str(raised(lambda: operator.setitem(plain_column, 5, 1.0)))
    assert plain_chosen.tobytes() == chosen.gather().tobytes()


def test_what_is_remembered_is_bounded(monkeypatch):
    """Each memory is emptied when it is full, and what is worked out afterwards is as right as before."""
    monkeypatch.setattr(rules, "PLAN_MEMORY_LIMIT", 6)
    monkeypatch.setattr(graph, "SPEC_MEMORY_LIMIT", 5)
    monkeypatch.setattr(graph, "ORIGIN_MEMORY_LIMIT", 3)
    monkeypatch.setattr(array, "FIT_MEMORY_LIMIT", 2)
    kept = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # The mean of no rows, where the mask selects none.
        for length in range(2, 14):
            data = numpy.arange(float(length * 2)).reshape(length, 2)
            x, plain = lazy.asarray(data), data
            table, plain_table = lazy.zeros((length, 2)), numpy.zeros((length, 2))
            for target, source in ((table, x), (plain_table, plain)):
                target += source * 2 + 1
                target[0] = source[-1]
                target -= source.sum(axis=0)
            results = (x.T @ x, (x * 2 + 1).sum(), x[1:, 0] - x[:-1, 1], x[x[:, 0] > 3].mean(axis=0), table)
            expected = (plain.T @ plain, (plain * 2 + 1).sum(), plain[1:, 0] - plain[:-1, 1], plain[plain[:, 0] > 3].mean(axis=0), plain_table)
            kept.append((results, expected))
            assert len(rules._plans) <= 6 and len(graph._specs) <= 5 and len(graph._origins) <= 3 and len(array._fitting) <= 2
        # Nodes written before a memory was emptied carry numbers it has forgotten; they compute as they should.
        for results, expected in kept:
            for got, plain in zip(results, expected):
                assert numpy.asarray(plain).tobytes() == numpy.asarray(got.gather()).tobytes()


def test_the_pending_set_holds_what_is_alive_and_not_computed():
    """A node leaves the pending set when it is computed and when nobody holds it, and the set is read newest first."""
    lazy_gather = get_lazy()
    gc.collect()
    before = graph.pending_count()
    a = lazy_gather.arange(4.0) * 2
    b, c = a + 1, a - 1
    assert graph.pending_count() == before + 3
    assert [node.number for node in graph.pending_nodes()][:3] == [c._node.number, b._node.number, a._node.number]
    b.gather()
    assert graph.pending_count() == before + 1 and not a._node.pending and c._node.pending
    del c
    gc.collect()
    assert graph.pending_count() == before


def test_the_stand_in_gives_one_wrapper_for_a_name_until_the_module_changes(monkeypatch):
    """`np.exp` through the stand-in is the same object each time, and another once the module has another function there."""
    x, data = lazy.arange(3.0), numpy.arange(3.0)
    wrapped = lazy.exp
    assert lazy.exp is wrapped and lazy.linalg is lazy.linalg and lazy.pi is numpy.pi
    assert wrapped(x).gather().tobytes() == numpy.exp(data).tobytes()
    monkeypatch.setattr(numpy, "exp", numpy.expm1)
    assert lazy.exp is not wrapped and lazy.exp(x).gather().tobytes() == numpy.expm1(data).tobytes()


def test_the_core_stands_in_for_the_python_and_can_be_put_away(lazy_core):
    """With the C core, an operator is a road that falls back to the Python method; put away, the method is back."""
    data = numpy.arange(12.0).reshape(3, 4)
    x = lazy.asarray(data)
    python_add, python_deferred = array._stood_in_for.get("__add__", LazyArray.__add__), array._python_deferred
    if lazy_core != "c":
        assert not array.core_in_use() and LazyArray.__add__ is python_add and graph.deferred is python_deferred
        return
    core = array._core
    roads = ("__add__", "__rmul__", "__neg__", "__lt__", "sum", "astype", "__array_ufunc__", "__array_function__", "__getitem__")
    assert all(type(vars(LazyArray)[name]) is core.Road and vars(LazyArray)[name].fallback is array._stood_in_for[name] for name in roads)
    assert graph.deferred is core.deferred
    # What the core does not take, the method does: a call made at once, an error, a bound method.
    assert (x + [1.0, 2.0, 3.0, 4.0]).gather().tobytes() == (data + [1.0, 2.0, 3.0, 4.0]).tobytes()
    assert str(raised(lambda: x + numpy.ones(5))) == str(raised(lambda: data + numpy.ones(5)))
    assert x.__add__(1).gather().tobytes() == (data + 1).tobytes() and x.sum(axis=0).gather().tobytes() == data.sum(axis=0).tobytes()
    # Switched off, every call is handed to the method; put away, the methods are the class's again.
    core.enable(False)
    try:
        assert (x * 2)._node.pending and (x * 2).gather().tobytes() == (data * 2).tobytes()
    finally:
        core.enable(True)
    try:
        array.use_core(None)
        assert LazyArray.__add__ is python_add and graph.deferred is python_deferred and (x * 2)._node.pending

        class Older:
            """Stands for a core built from another version of the source."""

            VERSION = array.CORE_VERSION + 1

        array.use_core(Older)
        assert not array.core_in_use() and LazyArray.__add__ is python_add
    finally:
        array.use_core(core)
    assert array.core_in_use() and (x * 2).gather().tobytes() == (data * 2).tobytes()


def test_a_selection_is_strict_if_its_index_or_its_array_is():
    """`x[mask]` refuses its values when the mask or the array is strict, and a view method keeps its arguments as Python puts them: the second time as the first."""
    data = numpy.arange(12.0).reshape(3, 4)
    plain_x, strict_x = lazy.asarray(data) + 0, strict.asarray(data) + 0
    row = lazy.arange(4.0) + 0
    for _ in range(3):
        by_strict_mask, of_strict_array, neither = plain_x[strict_x[:, 0] > 3], strict_x[plain_x[:, 0] > 3], plain_x[plain_x[:, 0] > 3]
        assert by_strict_mask._node.pending and by_strict_mask._node.strict and of_strict_array._node.strict and not neither._node.strict
        # One whole number is one axis, not a tuple of axes; several are a tuple; a tuple is itself.
        assert row.transpose(0)._node.args[1:] == ((0,),) and row.reshape(4)._node.args[1:] == (4,)
        assert plain_x.transpose(1, 0)._node.args[1:] == plain_x.transpose((1, 0))._node.args[1:] == ((1, 0),)
        assert plain_x.reshape(4, 3)._node.args[1:] == plain_x.reshape((4, 3))._node.args[1:] == ((4, 3),)
    assert neither.gather().tobytes() == data[data[:, 0] > 3].tobytes()


class Heard:
    """Hears what the layer does and writes it down in a form two runs of one program can be compared in."""

    def __init__(self) -> None:
        """Start with nothing heard."""
        # What was heard, in order.
        self.said: list[tuple] = []
        # The number of the first node heard of, which the others are counted from.
        self.first: int | None = None

    def place(self, number: int) -> int:
        """Count a node's number from the first node heard of.

        Args:
            number: A node's number.

        Returns:
            How many nodes after the first it was made.
        """
        if self.first is None:
            self.first = number
        return number - self.first

    def shape(self, shape: tuple[int, ...]) -> tuple:
        """Write a shape with the node a mark belongs to counted from the first node.

        Args:
            shape: A node's shape.

        Returns:
            The same lengths, with ("?", place) for one that is not known.
        """
        return tuple(length if length < graph.UNKNOWN_FLOOR else ("?", self.place(length - graph.UNKNOWN_FLOOR)) for length in shape)

    def argument(self, argument: Any) -> Any:
        """Describe one argument of a node.

        Args:
            argument: An operand, an axis, an index.

        Returns:
            The place of a node, the shape and type of an `ndarray`, the text of anything else.
        """
        if isinstance(argument, graph.Node):
            return ("node", self.place(argument.number))
        if isinstance(argument, numpy.ndarray):
            return ("ndarray", argument.shape, str(argument.dtype))
        if isinstance(argument, (tuple, list)):
            return (type(argument).__name__, *(self.argument(item) for item in argument))
        if isinstance(argument, slice):
            return ("slice", self.argument(argument.start), self.argument(argument.stop), self.argument(argument.step))
        return repr(argument)

    def made(self, node: graph.Node, source: str) -> None:
        """Hear that a node was made, as `graph.Observer` has it.

        Args:
            node: The node.
            source: What made it.
        """
        origin = None if node.origin is None else (node.origin.filename, node.origin.lineno, node.origin.module)
        self.said.append(("made", self.place(node.number), node.name, source, self.argument(node.args), sorted(node.kwargs), self.shape(node.shape),
                          str(node.dtype), node.kind, node.strict, node.exposed, node.pending, origin, node.errstate))

    def computing(self, nodes: list[graph.Node]) -> None:
        """Hear which nodes are about to be computed.

        Args:
            nodes: The pending nodes one computation will compute.
        """
        self.said.append(("computing", [self.place(node.number) for node in nodes]))

    def fell_back(self, kind: str, call: str, origin: graph.Origin) -> None:
        """Hear that the layer computed without being asked.

        Args:
            kind: What kind of fallback.
            call: The call that caused it.
            origin: The line it happened on.
        """
        self.said.append(("fell back", kind, call, origin.lineno))

    def computed(self, numbers: list[int], failed: set[int], roots: list[int], trigger: graph.Trigger) -> None:
        """Hear that one computation is over.

        Args:
            numbers: The nodes it took.
            failed: Those that failed.
            roots: Those that were asked for.
            trigger: What made the layer compute.
        """
        self.said.append(("computed", [self.place(number) for number in numbers], sorted(self.place(number) for number in failed),
                          [self.place(number) for number in roots], trigger.kind, trigger.call))


def random_program(seed: int, heard: Heard) -> list:
    """Run a program of random operations on lazy arrays, and describe everything it gave.

    The operations are drawn from one seeded generator, so two runs make the
    same calls for as long as the layer answers them alike, and anything
    they answer differently shows in what is returned or in what was heard.

    Args:
        seed: The seed of the program.
        heard: Gains what the layer did.

    Returns:
        One entry per step: what the step returned, or the exception it
        raised; then the gathered value of every lazy array made.
    """
    rng = random.Random(seed)
    values = numpy.random.default_rng(seed)
    np = get_lazy(strict=rng.random() < STRICT_SHARE)
    pool: list[Any] = []

    def data() -> numpy.ndarray:
        """Draw an `ndarray` of small whole numbers, of a random shape and element type."""
        return values.integers(-3, 4, size=rng.choice(SHAPES)).astype(rng.choice(ELEMENT_TYPES))

    def fresh() -> LazyArray:
        """Make a lazy array that has its values."""
        return rng.choice((lambda: np.asarray(data()), lambda: np.zeros(rng.choice(SHAPES), dtype=rng.choice(ELEMENT_TYPES)),
                           lambda: np.arange(rng.choice((3, 4, 6))), lambda: np.ones(rng.choice(SHAPES)) * 2.0))()

    def held() -> LazyArray:
        """Pick a lazy array made earlier, the recent ones more often."""
        arrays = [item for item in pool if isinstance(item, LazyArray)]
        return rng.choice(arrays[-RECENT:] if rng.random() < RECENT_SHARE else arrays) if arrays else fresh()

    def anything() -> Any:
        """Pick an operand: a lazy array, a number or an `ndarray`."""
        kind = rng.random()
        if kind < LAZY_SHARE:
            return held()
        if kind < LAZY_SHARE + NUMBER_SHARE:
            return rng.choice((0, 1, 2, -1, 300, 2.0, 0.5, float("nan"), True, 1j, numpy.float32(2), numpy.int64(-1), numpy.uint8(200)))
        return data()

    def axis() -> Any:
        """Pick an axis, a good one or a bad one."""
        return rng.choice((None, 0, -1, 1, 5, (0, 1), 1.0, True))

    def either(function: Callable[..., Any]) -> Callable[..., Any]:
        """Give a NumPy function as the real module has it or as the stand-in has it."""
        return rng.choice((function, getattr(np, function.__name__)))

    def by_operator(x: LazyArray) -> Any:
        """Apply a binary operator, the lazy array on either side."""
        symbol = rng.choice((operator.add, operator.sub, operator.mul, operator.truediv, operator.pow, operator.matmul, operator.and_,
                             operator.lt, operator.eq, operator.mod, operator.floordiv))
        return rng.choice((lambda: symbol(x, anything()), lambda: symbol(x, anything()), lambda: symbol(anything(), x)))()

    def by_ufunc(x: LazyArray) -> Any:
        """Call a ufunc of one operand or of two."""
        function = either(rng.choice((numpy.exp, numpy.sqrt, numpy.negative, numpy.abs, numpy.isnan, numpy.invert, numpy.maximum, numpy.hypot,
                                      numpy.power)))
        return function(x) if function.nin == 1 else function(x, anything())

    def by_reduction(x: LazyArray) -> Any:
        """Reduce, as a method or a function, with the axis given in each of the ways it can be."""
        name = rng.choice(("sum", "mean", "max", "prod", "std", "any", "all", "argmax", "cumsum"))
        call = rng.choice((getattr(x, name), lambda *args, **kwargs: getattr(numpy, name)(x, *args, **kwargs),
                           lambda *args, **kwargs: getattr(np, name)(x, *args, **kwargs)))
        return rng.choice((lambda: call(), lambda: call(axis()), lambda: call(axis=axis()), lambda: call(axis=axis(), keepdims=rng.choice((True, 0.0)))))()

    def by_view(x: LazyArray) -> Any:
        """Take a view, a cast, a rounding or a selection between two arrays."""
        return rng.choice((lambda: x.T, lambda: x.swapaxes(0, rng.choice((1, -1, 2))), lambda: x.reshape(rng.choice(((-1,), (4, 3), (2, -1)))),
                           lambda: x.ravel(), lambda: numpy.expand_dims(x, rng.choice((0, -1))), lambda: numpy.squeeze(x),
                           lambda: x.transpose(*rng.choice(((), (1, 0), (2, 0, 1)))), lambda: x.astype(rng.choice((numpy.float32, "int64", bool))),
                           lambda: x.round(rng.choice((0, 2))), lambda: numpy.where(x > 0, x, anything())))()

    def by_mask(x: LazyArray) -> Any:
        """Select by a pending mask or by its positions, and read or write what was selected."""
        mask = x > rng.choice((0, 0.5, -1))
        chosen = rng.choice((lambda: x[mask], lambda: x[mask], lambda: x[numpy.flatnonzero(mask)]))()
        return rng.choice((lambda: chosen, lambda: chosen * anything(), lambda: chosen.mean(axis=axis()), lambda: chosen + x[mask],
                           lambda: chosen + held()[mask], lambda: chosen[:, 0], lambda: chosen.T, lambda: numpy.cumsum(chosen),
                           lambda: operator.setitem(x, mask, anything()) or x, lambda: operator.setitem(x, mask, chosen * 2) or x))()

    def in_place(x: LazyArray) -> Any:
        """Apply an in-place operator, and say whether the array stayed the same object."""
        after = rng.choice((operator.iadd, operator.isub, operator.imul, operator.itruediv))(x, anything())
        return (after is x, after)

    def assign(x: LazyArray) -> Any:
        """Assign through a random index, or update through one."""
        index = random_index(rng, x._node.shape)
        if rng.choice((True, False)):
            x[index] = anything()
        else:
            x[index] += anything()
        return x

    def stale_view(x: LazyArray) -> Any:
        """Take a view, write into the array it is a view of, and read the view."""
        view = x[random_index(rng, x._node.shape)]
        x += 1
        return (view + 1, view.sum())

    def into_out(x: LazyArray) -> Any:
        """Call a ufunc or `np.clip` with the array as `out`."""
        return rng.choice((lambda: numpy.add(anything(), anything(), out=x), lambda: numpy.clip(x, -1, 1, out=x)))()

    def under_errstate(x: LazyArray) -> Any:
        """Divide under an `np.errstate` of its own."""
        with numpy.errstate(divide=rng.choice(("ignore", "raise", "warn"))):
            return x / anything()

    steps: dict[str, Callable[[LazyArray], Any]] = {
        "fresh": lambda x: fresh(), "operator": by_operator, "ufunc": by_ufunc, "reduction": by_reduction,
        "index": lambda x: x[random_index(rng, x._node.shape)], "view": by_view, "mask": by_mask, "in place": in_place,
        "assign": assign, "stale view": stale_view, "out": into_out, "errstate": under_errstate}

    def said(result: Any) -> Any:
        """Describe what a step returned."""
        if isinstance(result, LazyArray):
            return (repr(result).replace(str(result._node.shape), str(heard.shape(result._node.shape))), heard.place(result._node.number),
                    len(result._steps))
        if isinstance(result, tuple):
            return tuple(said(item) for item in result)
        return repr(result)

    def value(item: LazyArray) -> Any:
        """Gather a lazy array and describe its value, or the exception."""
        try:
            got = numpy.asarray(item.gather())
        except Exception as error:
            return (type(error).__name__, str(error))
        return (got.shape, str(got.dtype), repr(got.tolist()) if got.dtype == object else got.tobytes())

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(PROGRAM_STEPS):
            kind = rng.choices(list(STEPS_DRAWN), list(STEPS_DRAWN.values()))[0]
            try:
                result = steps[kind](held())
            except Exception as error:
                out.append((type(error).__name__, str(error)))
                continue
            out.append(said(result))
            pool.extend(item for item in (result if isinstance(result, tuple) else (result,)) if isinstance(item, LazyArray))
        out.extend(value(item) for item in pool)
    return out


def forget_everything() -> None:
    """Empty every memory of the layer, as at the start of a process."""
    rules._plans.clear()
    rules._admissible_specs.clear()
    array._fitting.clear()
    array._assignable.clear()
    graph._origins.clear()


def test_random_programs_are_written_alike_with_and_without_the_memories(monkeypatch):
    """A program gives the same nodes, lines, fallbacks, errors and values whether the layer remembers or works everything out.

    With the memories switched off every question goes the long way: to the
    rules, to NumPy's answer on stand-ins, up the stack. With them on, the
    program runs twice more, the second time with every answer already there:
    in Python, or by the C core when the test is run with it, where the
    second run is the one the core writes.
    """
    def run(seed: int) -> tuple[list, list]:
        """Run one random program under an observer, with nothing held from the run before."""
        gc.collect()
        gc.disable()  # So that both runs drop what nobody holds at the same moments.
        heard = Heard()
        monkeypatch.setattr(graph, "observer", heard)
        graph.COUNTS.reset()
        try:
            return random_program(seed, heard), heard.said
        finally:
            monkeypatch.setattr(graph, "observer", None)
            gc.enable()

    nodes = 0
    core = array._core
    for seed in range(PROGRAMS):
        with monkeypatch.context() as off:
            array.use_core(None)  # The C core reads the memories directly: off means Python alone.
            off.setattr(rules, "_key", lambda operation, args, kwargs, marks: None)
            off.setattr(rules, "basic_shape", lambda shape, index: None)
            off.setattr(array, "_fit_key", lambda written_by, target, operands: None)
            off.setattr(array, "_assignment_key", lambda target, parts, value: None)
            off.setattr(graph, "_remembered", lambda frame: None)
            off.setattr(graph, "_error_settings", graph._NoRecord)
            try:
                plain = run(seed)
            finally:
                array.use_core(core)
        forget_everything()
        first = run(seed)
        again = run(seed)
        for remembered in (first, again):
            for theirs, ours in zip(plain[1], remembered[1]):
                assert theirs == ours, (seed, theirs, ours)
            assert plain[0] == remembered[0] and len(plain[1]) == len(remembered[1]), seed
        nodes += sum(1 for item in plain[1] if item[0] == "made")
    assert nodes > 50 * PROGRAMS
