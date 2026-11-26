"""Tests of what the lazy layer does to run a program quickly without changing a value.

Three things, each held to NumPy's results bit for bit:

- a write made into the array's own memory, at computing time, when nothing
  reads what the array held before (`evaluate.reusable`);
- small calls made as the program writes them, and writes made at once in
  place when nothing is pending (`array.at_once_below`), which only the C
  core does;
- `x ** e` as NumPy's own operator, and `x.shape = s`.

Each test runs once in Python alone and once with the C core, as the other
tests of the layer do; the ones of calls made at once are skipped without it.
"""
from __future__ import annotations

import random
import warnings
from typing import Any, Callable, Iterator

import numpy
import pytest

from prototypes.lazy import LazyArray, array, evaluate, fuse, get_lazy, graph
from prototypes.lazy.programs import corpus

lazy = get_lazy()
# Seed of every random array and program in these tests.
SEED = 0
# A limit under which every array of these tests is small, and one under which none is.
EVERYTHING, NOTHING = 1_000_000, 0
# How many random programs are run, and how many steps each has.
PROGRAMS, STEPS = 60, 80
# The side of the square arrays the random programs work on, and of the large one that keeps a node pending.
SIDE, LARGE_SIDE = 6, 40


@pytest.fixture
def at_once() -> Iterator[None]:
    """Have small calls made at once for the length of a test: skipped without the C core, which makes them."""
    if not array.core_in_use():
        pytest.skip("calls are made at once by the C core only")
    array.at_once_below(EVERYTHING)
    try:
        yield
    finally:
        array.at_once_below(NOTHING)


def same(expected: Any, got: Any) -> bool:
    """Say whether two results are the same in element type, shape and bits."""
    expected, got = numpy.asarray(expected), numpy.asarray(got)
    return expected.dtype == got.dtype and expected.shape == got.shape and expected.tobytes() == got.tobytes()


def writes(call: Callable[[], Any]) -> tuple[int, int, Any]:
    """Run something and count the deferred writes it computed: (made in place, made into a copy, what it returned)."""
    evaluate.WRITES.update(in_place=0, copied=0)
    result = call()
    return evaluate.WRITES["in_place"], evaluate.WRITES["copied"], result


# ---- A write made in place when it is computed.

def test_a_loop_of_writes_copies_nothing():
    """One element set per turn: every write but none is made where the array is."""
    def program(np: Any) -> Any:
        """Fill an array one element at a time from pending values."""
        x = np.zeros(50)
        source = np.asarray(numpy.arange(50.0)) * 2.0
        for i in range(50):
            x[i] = source[i] + 1.0
        return x

    expected = program(numpy)
    in_place, copied, got = writes(lambda: program(lazy).gather())
    assert same(expected, got)
    assert copied == 0 and in_place == 50


def test_an_operation_written_before_a_write_reads_what_the_array_held():
    """`y = x + 1; x[0] = 5`: `y` is computed after the write and still has the old first element."""
    def program(np: Any) -> tuple:
        """Read an array, write into it, and return both."""
        x = np.asarray(numpy.arange(8.0)) * 1.0
        y = x + 1.0
        x[0] = 50.0
        x += 2.0
        return x, y

    expected = program(numpy)
    x, y = program(lazy)
    got = x.gather(), y.gather()  # `x` first: its writes are computed while `y` is still pending.
    assert same(expected[0], got[0]) and same(expected[1], got[1])


def test_a_write_that_reads_a_view_of_its_own_array_is_made_in_place():
    """`p[:, -1] = p[:, -2]` copies nothing, as with NumPy, and the view is of the contents before the write."""
    def program(np: Any) -> Any:
        """Copy columns and rows of an array onto its edges."""
        p = np.asarray(numpy.arange(30.0).reshape(5, 6)) * 1.0
        for _ in range(3):
            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[1:, 0] = p[:-1, 1]
        return p

    expected = program(numpy)
    in_place, copied, got = writes(lambda: program(lazy).gather())
    assert same(expected, got)
    assert copied == 0 and in_place == 9


def test_a_view_the_program_holds_keeps_a_write_out_of_the_memory_it_looks_at():
    """A computed view that something still reads is not written under: the write goes to a copy."""
    x = lazy.asarray(numpy.arange(12.0)) * 1.0
    view = x[2:6]
    first = (view * 1.0).gather()  # The view has its values now, in the memory of `x`.
    later = view + 0.5  # Pending, and reads that memory.
    x[3] = 99.0
    after = x.gather()
    assert same(numpy.arange(2.0, 6.0) + 0.5, later.gather())
    assert after[3] == 99.0 and same(numpy.arange(2.0, 6.0), first)


def test_a_write_keeps_the_layout_of_an_array_in_fortran_order():
    """A sum after a write adds in the order NumPy's does, which is the order the array's memory has."""
    rng = numpy.random.default_rng(SEED)
    a, b = (numpy.asfortranarray(rng.standard_normal((37, 53))) for _ in range(2))

    def program(np: Any) -> Any:
        """Write into a Fortran-ordered sum and add it up."""
        total = np.asarray(a) + np.asarray(b)
        total += 2.0
        total[3] = 1.5
        return total.sum(), total

    expected = program(numpy)
    got = program(lazy)
    assert same(expected[0], got[0].gather()) and same(expected[1], got[1].gather())
    assert got[1].gather().flags.f_contiguous


# ---- `x ** e` and `x.shape = s`.

@pytest.mark.parametrize("exponent", (2, 2.0, 0.5, -1, 1, 3, 0, -2, 1.5))
def test_a_power_is_the_operator_numpy_computes(exponent: Any):
    """`x ** e` has the bits of NumPy's operator for every exponent, the ones it has a quick way for among them."""
    rng = numpy.random.default_rng(SEED)
    for values in (rng.standard_normal(100) * 3, rng.standard_normal(100) + 1j * rng.standard_normal(100),
                   rng.standard_normal(100).astype(numpy.float32)):
        with numpy.errstate(all="ignore"):
            assert same(values ** exponent, (lazy.asarray(values) ** exponent).gather())
            assert same(exponent ** values, (exponent ** lazy.asarray(values)).gather())


def test_a_shape_can_be_assigned():
    """`x.shape = s` gives the array another shape where it is; a write through either shape is seen by both."""
    def program(np: Any) -> tuple:
        """Reshape an array in place, then write through a view made before and through the array."""
        x = np.asarray(numpy.arange(12.0)) * 1.0
        row = x[0:3]
        x.shape = (3, -1)
        x[1, 1] = 50.0
        row[0] = 70.0
        return x, row, x.shape

    expected = program(numpy)
    got = program(lazy)
    assert got[2] == expected[2] == (3, 4)
    assert same(expected[0], got[0].gather()) and same(expected[1], got[1].gather())


def test_a_shape_that_does_not_hold_the_elements_is_refused_in_numpys_words():
    """The size is checked when the shape is assigned, with NumPy's message."""
    x = lazy.asarray(numpy.arange(12.0)) * 1.0
    with pytest.raises(ValueError, match="cannot reshape array of size 12 into shape"):
        x.shape = (5, 2)
    with pytest.raises(ValueError, match="can only specify one unknown dimension"):
        x.shape = (-1, -1)


# ---- Small calls made at once.

def test_a_small_call_is_made_at_once_and_a_large_one_is_deferred(at_once: None):
    """Under a limit a call on arrays that have their values leaves nothing pending; at the limit it is a node."""
    array.at_once_below(100)
    small, large = lazy.asarray(numpy.arange(99.0)), lazy.asarray(numpy.arange(100.0))
    before = graph.COUNTS.deferred
    result = small * 2.0 + 1.0
    assert not result._node.pending and graph.COUNTS.deferred == before
    assert same(numpy.arange(99.0) * 2.0 + 1.0, result.gather())
    pending = large * 2.0
    assert pending._node.pending
    on_pending = pending[:5] + 1.0  # Small, but its operand has no values yet.
    assert on_pending._node.pending
    assert same((numpy.arange(100.0) * 2.0)[:5] + 1.0, on_pending.gather())
    total = large.sum()  # A small result of a large array that has its values.
    assert not total._node.pending and same(numpy.arange(100.0).sum(), total.gather())


def test_a_call_made_at_once_raises_and_warns_where_it_is_written(at_once: None):
    """NumPy's error is the program's at that line, as without the layer, and so is its warning."""
    a, b = lazy.asarray(numpy.ones(3)), lazy.asarray(numpy.ones(4))
    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        a + b
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in divide"):
        got = a / 0.0
    assert same(numpy.full(3, numpy.inf), got.gather())


def test_a_write_at_once_is_seen_through_every_view(at_once: None):
    """With nothing pending a write goes into the array's memory, and a view made before sees it, as with NumPy."""
    def program(np: Any) -> tuple:
        """Write through an array and through its views."""
        x = np.zeros((4, 5))
        row, column = x[1], x[:, 2]
        x[1, 2] = 7.0
        row[0] = 3.0
        column += 1.0
        x.T[4, 3] = 9.0
        flipped = x.T
        flipped *= 2.0
        return x, row, column

    expected = program(numpy)
    before = array.at_once_counts()[1]
    got = program(lazy)
    assert array.at_once_counts()[1] - before == 5 and not graph.pending_count()
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours.gather())


def test_a_write_is_deferred_while_something_is_pending(at_once: None):
    """A pending operation may read what the array holds, so the write waits its turn and the reader has the old contents."""
    array.at_once_below(100)
    large = lazy.asarray(numpy.arange(200.0)) * 1.0
    x = lazy.zeros(10)
    reader = large[:10] + x  # Pending: `large * 1.0` is.
    x[0] = 5.0
    assert x._node.pending
    assert same(numpy.arange(10.0), reader.gather())
    assert x.gather()[0] == 5.0


def test_an_array_made_of_no_array_is_the_layers_own(at_once: None):
    """`np.zeros(n)` through the stand-in is a lazy array that a write into can be deferred for."""
    made = lazy.zeros((3, 4))
    assert isinstance(made, LazyArray) and not made._node.exposed and made._node.kind == ""
    like = lazy.ones_like(made)
    assert isinstance(like, LazyArray) and same(numpy.ones((3, 4)), like.gather())


def random_program(np: Any, seed: int) -> list:
    """Run a program of random reads, views and writes on a few small arrays and one large, and return every array.

    Args:
        np: The `numpy` module or the lazy stand-in.
        seed: The seed of the program: the same calls are made for either module.

    Returns:
        The arrays the program ended with, the views it took among them.
    """
    rng = random.Random(seed)
    values = numpy.random.default_rng(seed)
    pool = [np.asarray(values.integers(-4, 5, (SIDE, SIDE)).astype(float)) * 1.0 for _ in range(3)]
    large = np.asarray(values.standard_normal((LARGE_SIDE, LARGE_SIDE)))

    def span(length: int) -> slice:
        """Draw a slice of an axis that takes at least one element of it."""
        start = rng.randrange(length)
        return slice(start, rng.randrange(start + 1, length + 1))

    def pick() -> Any:
        """Draw an array of the pool."""
        return rng.choice(pool)

    for _ in range(STEPS):
        kind = rng.randrange(12)
        target = pick()
        if kind == 0:
            other = pick()
            pool.append(target + other if target.shape == other.shape else target * 2.0)
        elif kind == 1 and target.ndim == 2:
            pool.append(target[span(target.shape[0]), span(target.shape[1])])
        elif kind == 2 and target.ndim == 2:
            pool.append(target[rng.randrange(target.shape[0])])
        elif kind == 3:
            target[..., rng.randrange(target.shape[-1])] = float(rng.randrange(-9, 10))
        elif kind == 4:
            target += 0.5
        elif kind == 5:
            target *= pick().sum()
        elif kind == 6 and target.ndim == 2:
            pool.append(target.T)
        elif kind == 7:
            target[target > 1.0] = -1.0
        elif kind == 8:
            np.clip(target, -6.0, 6.0, out=target)
        elif kind == 9:
            large = large * 1.0001 + 0.5  # Stays pending under a limit below its size.
        elif kind == 10 and target.shape == (SIDE, SIDE):
            target[span(SIDE), :] = large[:1, :SIDE]
        elif kind == 11 and target.ndim == 2 and target.shape[0] > 1:
            target[0] = target[-1] - target[0]
        if len(pool) > 8:
            pool.pop(rng.randrange(3, len(pool)))
    return pool + [large]


@pytest.mark.parametrize("limit", (NOTHING, SIDE * SIDE + 1, EVERYTHING))
def test_random_programs_of_views_and_writes_give_numpys_arrays(limit: int):
    """Reads, views and writes in any order give NumPy's arrays, whatever is made at once and whatever in place."""
    if limit and not array.core_in_use():
        pytest.skip("calls are made at once by the C core only")
    array.at_once_below(limit)
    try:
        for seed in range(PROGRAMS):
            expected = random_program(numpy, seed)
            got = array.gather(*random_program(lazy, seed))
            assert len(expected) == len(got)
            for place, (theirs, ours) in enumerate(zip(expected, got)):
                assert same(theirs, ours), (seed, place)
    finally:
        array.at_once_below(NOTHING)


@pytest.mark.parametrize("name", [program.name for program in corpus.programs() if program.eager])
def test_a_program_of_the_corpus_gives_numpys_results_as_it_is_run_for_speed(name: str):
    """Every program, run as `fusion_time` runs it fused: small calls at once, writes in place, kernels where they pay."""
    program = corpus.by_name([name])[0]
    if not program.available:
        pytest.skip(f"needs {', '.join(program.requires)}")
    if not fuse.enable():
        pytest.skip("the fused kernels are not built")
    try:
        inputs = program.make(corpus.SMALL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = list(_values(program.run(numpy, inputs)))
            got = array.gather(*_values(program.run(lazy, inputs)))
    finally:
        fuse.disable()
    assert len(expected) == len(got)
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours)


def _values(result: Any) -> Iterator[Any]:
    """List the arrays and numbers in a program's result, whatever tuples and lists they are in."""
    if isinstance(result, (tuple, list)):
        for item in result:
            yield from _values(item)
    else:
        yield result
