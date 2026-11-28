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

import gc
import random
import warnings
import weakref
from typing import Any, Callable, Iterator

import numpy
import pytest

from prototypes.lazy import LazyArray, array, evaluate, fuse, get_lazy, graph
from prototypes.lazy.programs import codebases, corpus

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
    """Under a limit a call on arrays that have their values leaves nothing pending; at the limit it is a node. A
    small call on an operand that waits has the operand computed, and what else waits goes on waiting."""
    array.at_once_below(100)
    small, large = lazy.asarray(numpy.arange(99.0)), lazy.asarray(numpy.arange(100.0))
    before = graph.COUNTS.deferred
    result = small * 2.0 + 1.0
    assert not result._node.pending and graph.COUNTS.deferred == before
    assert same(numpy.arange(99.0) * 2.0 + 1.0, result.gather())
    pending = large * 2.0
    assert pending._node.pending
    wider = pending + 1.0  # Large, on an operand that waits: it waits as well.
    assert wider._node.pending and pending._node.pending
    on_pending = pending[:5] + 1.0  # Small: what it reads is computed, and the call is made at once.
    assert not on_pending._node.pending and not pending._node.pending and wider._node.pending
    assert graph.pending_count() == 1
    assert same((numpy.arange(100.0) * 2.0)[:5] + 1.0, on_pending.gather())
    assert same(numpy.arange(100.0) * 2.0 + 1.0, wider.gather())
    total = large.sum()  # A small result of a large array that has its values.
    assert not total._node.pending and same(numpy.arange(100.0).sum(), total.gather())


def test_a_large_call_no_kernel_could_make_is_made_at_once(at_once: None):
    """Told what a kernel can make, the core makes every other call at once whatever its size, and lets those wait."""
    array.at_once_below(100, frozenset({numpy.add, numpy.multiply, numpy.absolute}), numpy.dtype(numpy.float64))
    values = numpy.arange(500.0)
    x = lazy.asarray(values)
    mask = x > 100.0  # A comparison gives a mask.
    assert not mask._node.pending and same(values > 100.0, mask.gather())
    picked = x[mask]  # How many it selects is known once it is made.
    assert not picked._node.pending and picked.shape == (399,) and same(values[values > 100.0], picked.gather())
    product = x @ x  # A matrix product.
    assert not product._node.pending and same(values @ values, product.gather())
    exponent = lazy.exp(x * 0.0)  # Its operand waits, so it does too.
    assert exponent._node.pending and same(numpy.exp(values * 0.0), exponent.gather())
    waits = x * 2.0 + 1.0  # Arithmetic on float64: a kernel's.
    assert waits._node.pending and same(values * 2.0 + 1.0, waits.gather())
    turned = numpy.arange(500.0) + 1j
    size = abs(lazy.asarray(turned))  # float64 from a complex array: no kernel reads that.
    assert not size._node.pending and same(abs(turned), size.gather())
    single = numpy.arange(500.0, dtype=numpy.float32)
    doubled = lazy.asarray(single) * 2.0  # float32.
    assert not doubled._node.pending and same(single * 2.0, doubled.gather())


def test_a_function_that_gives_a_new_array_is_called_at_once_on_small_arrays(at_once: None):
    """`np.dot`, `np.sum`, `np.copy` and their like on small arrays that have their values are NumPy's own calls."""
    rng = numpy.random.default_rng(SEED)
    a, b = rng.standard_normal((6, 4)), rng.standard_normal((4, 3))
    la, lb = lazy.asarray(a), lazy.asarray(b)
    cases = [
        (numpy.dot(a, b), lazy.dot(la, lb)), (numpy.sum(a, axis=0), lazy.sum(la, axis=0)),
        (numpy.sum(a), lazy.sum(la)), (numpy.max(a, axis=1, keepdims=True), lazy.max(la, axis=1, keepdims=True)),
        (numpy.argmax(b), lazy.argmax(lb)), (numpy.copy(a), lazy.copy(la)), (numpy.outer(a[0], b[0]), lazy.outer(la[0], lb[0])),
    ]
    before = graph.COUNTS.deferred
    for expected, got in cases:
        assert isinstance(got, LazyArray) and not got._node.pending and same(expected, got.gather())
    assert graph.COUNTS.deferred == before
    copied = lazy.copy(la)
    copied[0, 0] = 99.0  # A copy is an array of its own: the write does not reach `a`.
    assert a[0, 0] != 99.0 and copied.gather()[0, 0] == 99.0
    with pytest.raises(ValueError):
        lazy.dot(la, la)


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
    large = lazy.asarray(numpy.arange(200.0).reshape(20, 10))
    x = lazy.zeros(10)
    reader = large + x  # Pending: a large call waits.
    x[0] = 5.0
    assert x._node.pending
    assert same(numpy.arange(200.0).reshape(20, 10), reader.gather())
    assert x.gather()[0] == 5.0


def test_an_array_made_of_no_array_is_the_layers_own(at_once: None):
    """`np.zeros(n)` through the stand-in is a lazy array that a write into can be deferred for."""
    made = lazy.zeros((3, 4))
    assert isinstance(made, LazyArray) and not made._node.exposed and made._node.kind == ""
    like = lazy.ones_like(made)
    assert isinstance(like, LazyArray) and same(numpy.ones((3, 4)), like.gather())


# ---- What the result of a call made at once is made of, and how the core makes and frees the layer's objects.

def test_a_call_made_at_once_gives_an_array_that_holds_its_value_alone(at_once: None):
    """No node and no cell until something asks; then they are what any array that has its values has."""
    made = lazy.asarray(numpy.arange(6.0)) * 2.0 + 1.0
    assert type(made._held) is numpy.ndarray and made._seen == (graph.AT_ONCE, False) and not gc.is_tracked(made)
    assert made.shape == (6,) and made.dtype == numpy.dtype(numpy.float64)
    node = made._node
    assert type(made._held) is array._Cell and made._seen is None and gc.is_tracked(made)
    assert made._node is node and made._cell.node is node
    assert node.kind == graph.AT_ONCE and not node.pending and not node.exposed and not node.strict
    assert node.shape == (6,) and node.dtype == numpy.dtype(numpy.float64)
    assert same(numpy.arange(6.0) * 2.0 + 1.0, made.gather())


def test_an_array_that_holds_its_value_alone_keeps_strictness(at_once: None):
    """What is computed from a strict array refuses its values without `gather()`, node or no node."""
    strict = get_lazy(strict=True)
    made = strict.asarray(numpy.arange(6.0)) + 1.0
    total = made.sum()
    assert made._seen == (graph.AT_ONCE, True) and total._seen == (graph.AT_ONCE, True)
    with pytest.raises(graph.NotGathered):
        float(total)
    into = lazy.zeros(6) + 0.0
    into[2] = total  # A write made at once of something strict.
    assert into._seen == (graph.AT_ONCE, True)
    with pytest.raises(graph.NotGathered):
        float(into[2])
    assert made._node.strict and same(numpy.arange(6.0) + 1.0, made.gather())


def test_a_deferred_operation_on_an_array_that_holds_its_value_gives_it_a_node(at_once: None):
    """A large call waits, and what it reads becomes a node of the graph like any other."""
    array.at_once_below(100)
    small = lazy.zeros(50) + 3.0
    large = lazy.asarray(numpy.arange(200.0).reshape(4, 50))
    assert type(small._held) is numpy.ndarray
    pending = large + small
    assert pending._node.pending and type(small._held) is array._Cell
    assert small._node in pending._node.args
    assert same(numpy.arange(200.0).reshape(4, 50) + 3.0, pending.gather())


def test_a_view_taken_at_once_holds_numpys_view(at_once: None):
    """A slice and a transpose of an array that has its values are NumPy's views, held alone like any small result."""
    whole = lazy.zeros((4, 5)) + 1.0
    row, turned = whole[1], whole.T
    assert type(row._held) is numpy.ndarray and row._held.base is whole._held and turned._held.base is whole._held
    assert type(whole._held) is numpy.ndarray  # The array viewed needed no node and no cell for it.
    whole[1, 2] = 7.0  # Made at once, in the memory all three look at.
    assert row.gather()[2] == 7.0 and turned.gather()[2, 1] == 7.0
    inner = whole[1:][0][2:]  # A view of a view of a view: each is NumPy's view of the one array.
    row[3] = 8.0  # Through a view, into the array.
    assert same(numpy.array([7.0, 8.0, 1.0]), inner.gather()) and whole.gather()[1, 3] == 8.0
    node = row._node
    assert node.kind == graph.AT_ONCE and not node.pending and row._node is node
    assert same(numpy.array([1.0, 1.0, 7.0, 8.0, 1.0]), (row * 1.0).gather())


def test_a_view_of_a_strict_array_is_strict(at_once: None):
    """What is taken of a strict array refuses its values without `gather()`."""
    data = numpy.arange(12.0).reshape(3, 4)
    strict = get_lazy(strict=True).asarray(data) + 0.0
    row = strict[1]
    assert row._seen == (graph.AT_ONCE, True)
    with pytest.raises(graph.NotGathered):
        float(row[0])
    assert row._node.strict and same(data[1], row.gather())


def test_a_write_does_not_wait_while_a_view_taken_at_once_may_see_the_array(at_once: None):
    """Such a view shows what the memory holds. A write that waited would go to a copy, and the view would not see it."""
    def program(np: Any) -> Any:
        """Take views, keep something large waiting, and write into the array and through a view."""
        whole = np.zeros(10) + 0.0
        part, other = whole[2:6], whole[::2]
        waiting = np.asarray(numpy.arange(200.0)) * 1.0
        whole[3] = 9.0
        part[0] = 4.0
        whole += 1.0
        return part, other, whole, waiting, part * 2.0

    array.at_once_below(100)
    expected = program(numpy)
    got = program(lazy)
    assert not got[2]._node.pending  # Each write was made where it was written, after what was waiting.
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours.gather())
    alone = lazy.zeros(10) + 0.0
    waiting = lazy.asarray(numpy.arange(200.0)) * 1.0
    alone[3] = 7.0  # No view of this array: the write waits, as it always did.
    assert alone._node.pending and waiting._node.pending
    in_place, copied, value = writes(alone.gather)
    assert (in_place, copied) == (1, 0) and value[3] == 7.0


def test_a_view_taken_at_once_outlives_the_limit_it_was_taken_under(at_once: None):
    """With calls no longer made at once, a write into an array such a view looks at still does not wait."""
    whole = lazy.zeros(10) + 0.0
    part = whole[2:6]
    array.at_once_below(NOTHING)
    waiting = lazy.asarray(numpy.arange(200.0)) * 1.0
    assert waiting._node.pending
    whole[3] = 9.0
    assert not whole._node.pending and same(numpy.array([0.0, 9.0, 0.0, 0.0]), part.gather())


def test_an_in_place_operator_is_made_at_once_in_the_memory(at_once: None):
    """`x += y`, `x[i] -= v` and `s += v` on small arrays with nothing pending: NumPy's own, where the array is."""
    def program(np: Any) -> Any:
        """Update an array, one of its rows, and a number."""
        x = np.zeros((4, 3)) + 2.0
        y = np.asarray(numpy.arange(12.0).reshape(4, 3))
        x += y
        x[1] -= 1.0
        x[2, 1:] *= y[0, 1:]
        x /= 2.0
        total = x.sum()
        total += 1.5
        total -= x[0, 0]
        return x, total

    expected = program(numpy)
    _, written_before = array.at_once_counts()
    deferred_before = graph.COUNTS.deferred
    got = program(lazy)
    # Four updates, and the two views written back to where they are, which NumPy sees is nothing to do.
    assert array.at_once_counts()[1] - written_before == 6 and graph.COUNTS.deferred == deferred_before
    assert same(expected[0], got[0].gather()) and same(expected[1], got[1].gather())


def test_an_in_place_operator_that_does_not_fit_fails_as_numpys_does(at_once: None):
    """Made at once, the error is NumPy's own at that line, and the array is as it was."""
    x = lazy.zeros(3, dtype=numpy.int64) + 1
    with pytest.raises(TypeError):
        x += 0.5
    with pytest.raises(ValueError):
        x += lazy.asarray(numpy.arange(4))
    assert same(numpy.ones(3, dtype=numpy.int64), x.gather())


@pytest.mark.parametrize("method", ("sum", "mean", "max", "min", "prod", "std", "var", "argmax", "argmin", "cumsum", "any", "all"))
def test_a_reduction_method_is_numpys_method_called_at_once(at_once: None, method: str):
    """With no argument, with an axis by position and by keyword, and with `keepdims` where NumPy has it."""
    data = numpy.random.default_rng(SEED).standard_normal((4, 5))
    x = lazy.asarray(data) + 0.0
    before = graph.COUNTS.deferred
    assert same(getattr(data + 0.0, method)(), getattr(x, method)().gather())
    assert same(getattr(data + 0.0, method)(1), getattr(x, method)(1).gather())
    assert same(getattr(data + 0.0, method)(axis=0), getattr(x, method)(axis=0).gather())
    if method not in ("argmax", "argmin", "cumsum"):
        assert same(getattr(data + 0.0, method)(axis=0, keepdims=True), getattr(x, method)(axis=0, keepdims=True).gather())
    assert graph.COUNTS.deferred == before
    with pytest.raises(Exception) as theirs:
        getattr(data, method)(axis=5)
    with pytest.raises(type(theirs.value)):
        getattr(x, method)(axis=5)


def test_a_product_method_takes_another_small_array(at_once: None):
    """`x.dot(y)` with both operands holding their values."""
    data = numpy.arange(12.0).reshape(3, 4)
    x, y = lazy.asarray(data) + 0.0, lazy.asarray(data.T) * 1.0
    assert same(data.dot(data.T), x.dot(y).gather())
    assert same(data.dot(data.T), x.dot(data.T).gather())  # An array from outside: the ordinary road.


def test_a_value_leaves_an_array_that_has_it_without_a_word(at_once: None):
    """A number, a truth value and the array itself, asked of an array that holds its value while nothing waits."""
    data = numpy.arange(6.0).reshape(2, 3)
    x = lazy.asarray(data) + 1.0
    total = x.sum()
    needed = dict(graph.COUNTS.needed)
    assert float(total) == 21.0 and int(total) == 21 and complex(total) == 21 + 0j and bool(total)
    assert 3.0 in x and 99.0 not in x and range(10)[x.argmax()] == 5
    given = numpy.asarray(x)
    assert type(x._held) is array._Cell and x._node.exposed and given is x._node.value
    assert same(data + 1.0, given) and same((data + 1.0).astype(numpy.float32), numpy.asarray(x, dtype=numpy.float32))
    assert numpy.array(x, copy=True) is not given
    assert graph.COUNTS.needed == needed
    with pytest.raises(graph.NotGathered):
        float(get_lazy(strict=True).asarray(data).sum())
    with pytest.raises((TypeError, ValueError)):
        float(x)  # NumPy's own refusal for an array of six.


def test_an_operator_on_numbers_is_the_numpy_scalars_own(at_once: None):
    """`x.max() * i / n`: arithmetic on what a reduction or an element gave is NumPy's scalar arithmetic, with its type,
    its bits and its warning, and writes no node."""
    data = numpy.random.default_rng(SEED).standard_normal(50)
    whole = numpy.arange(1, 9)

    def program(np: Any) -> list:
        """Operators of every kind on float, integer and boolean numbers taken out of arrays."""
        x, k = np.asarray(data), np.asarray(whole)
        top, low, first, count = x.max(), x.min(), x[3], k.sum()
        return [top * 3 / 7, 2.5 - top, top - low, top ** 2, -top, abs(low), top // 0.25, top % 0.5, first + count,
                count // 3, count % 5, count * 2 - 1, count & 6, count | 1, count ^ 3, ~count, top < low, top >= 0.5,
                top == top, first != low, (top > 0) & (low < 0), 1.0 / (top - top + 2.0)]

    expected = program(numpy)
    deferred = graph.COUNTS.deferred
    got = program(lazy)
    assert graph.COUNTS.deferred == deferred
    for theirs, ours in zip(expected, got):
        value = ours.gather() if type(ours) is LazyArray else ours
        assert type(value) is type(theirs) and same(theirs, value)
    x = lazy.asarray(data)
    nothing = x.max() - x.max()
    with warnings.catch_warnings(record=True) as lazily:
        warnings.simplefilter("always")
        (x.max() / nothing).gather()
    with warnings.catch_warnings(record=True) as eagerly:
        warnings.simplefilter("always")
        data.max() / (data.max() - data.max())
    assert [str(warning.message) for warning in lazily] == [str(warning.message) for warning in eagerly] != []
    strict = get_lazy(strict=True).asarray(data)
    with pytest.raises(graph.NotGathered):
        float(strict.max() * 2.0)


def test_a_large_call_no_kernel_could_make_is_made_at_once_by_the_core():
    """A comparison, `np.exp` and `np.logical_and` on arrays past the limit, with fusion on: NumPy's call at that line."""
    if not array.core_in_use() or not fuse.enable():
        pytest.skip("needs the C core and the fused kernels")
    try:
        data = numpy.random.default_rng(SEED).standard_normal(2 * fuse.AT_ONCE_BELOW)
        x = lazy.asarray(data)
        made, deferred = array.at_once_counts()[0], graph.COUNTS.deferred
        low, high = x < 0.25, x >= -0.25
        both = lazy.logical_and(low, high)
        grown = lazy.exp(x)
        assert array.at_once_counts()[0] - made == 4 and graph.COUNTS.deferred == deferred
        assert same(numpy.logical_and(data < 0.25, data >= -0.25), both.gather()) and same(numpy.exp(data), grown.gather())
        waits = x * 2.0 + 1.0
        assert graph.COUNTS.deferred == deferred + 2 and same(data * 2.0 + 1.0, waits.gather())
    finally:
        fuse.disable()


def test_an_operator_in_place_on_a_number_is_the_numpy_scalars_own(at_once: None):
    """`trace += np.tanh(a[i, i])`: a NumPy scalar cannot be written, so the operator gives a new number by the
    scalar's own arithmetic, with its type and bits, and another name for the old number keeps the old one."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 6))
    whole = numpy.arange(1, 9)

    def program(np: Any) -> list:
        """In-place operators of every kind on numbers taken out of arrays, each with a second name for the old one."""
        a, k = np.asarray(data), np.asarray(whole)
        results = []
        trace = 0.0
        for i in range(6):
            trace += np.tanh(a[i, i])
        results.append(trace)
        for start, steps in ((a[1, 2], ("+= 1.5", "-= a[0, 0]", "*= 3", "/= a[2, 2]", "//= 0.25", "%= 0.5")),
                             (k.sum(), ("+= 1", "*= k[2]", "//= 3", "%= 5", "&= 6", "|= 1", "^= 3"))):
            for step in steps:
                names = {"x": start, "a": a, "k": k}
                exec(f"old = x\nx {step}", names)
                results.extend((names["x"], names["old"]))
        return results

    expected = program(numpy)
    deferred = graph.COUNTS.deferred
    got = program(lazy)
    assert graph.COUNTS.deferred == deferred
    for theirs, ours in zip(expected, got):
        value = ours.gather() if type(ours) is LazyArray else ours
        assert type(value) is type(theirs) and same(theirs, value)


def test_a_numpy_scalar_of_the_programs_own_is_a_number_to_the_layer(at_once: None):
    """`alpha * a[i, k]` and `k * row` with `alpha` and `k` NumPy scalars the program made itself: NumPy's values and
    types either way round, on a number and on an array, and no node."""
    data = numpy.random.default_rng(SEED).standard_normal((6, 6))

    def program(np: Any) -> list:
        """Operators between NumPy scalars of several types and numbers and rows taken out of an array."""
        a = np.asarray(data)
        k, single, whole = numpy.float64(1.5), numpy.float32(0.25), numpy.int64(3)
        return [k * a[1, 2], a[1, 2] * k, k - a[0, 0], a[0, 0] / k, single + a[2, 2], a[2, 2] - single, whole * a[3, 3],
                a[3, 3] ** whole, k * a[1, :], a[1, :] * k, single * a[:, 2], a[:, 2] / single, whole - a[4, :], a[4, :] // whole,
                k < a[1, 2], a[1, :] >= k]

    expected = program(numpy)
    deferred = graph.COUNTS.deferred
    got = program(lazy)
    assert graph.COUNTS.deferred == deferred
    for theirs, ours in zip(expected, got):
        value = ours.gather() if type(ours) is LazyArray else ours
        assert type(value) is type(theirs) and same(theirs, value)


def test_a_write_at_once_goes_into_an_array_the_program_was_handed(at_once: None):
    """An array that came in through `np.asarray` is written where it is when nothing is pending, a row or an element
    at a time, by an assignment and by an operator; the array handed in sees every write, as with NumPy."""
    def program(np: Any, mine: numpy.ndarray) -> Any:
        """A rank update a row at a time, and a stencil an element at a time, both into the array given."""
        c = np.asarray(mine)
        for i in range(1, 5):
            c[i, :i + 1] *= 0.5
            c[i, :i + 1] += 1.5 * c[i, 0] * c[:i + 1, 1]
            for j in range(1, 5):
                c[i, j] += c[i, j - 1]
                c[i, j] /= 9.0
        c[0, :] = c[1, :] - c[2, :]
        return c

    theirs, ours = numpy.arange(36.0).reshape(6, 6), numpy.arange(36.0).reshape(6, 6)
    expected = program(numpy, theirs)
    deferred, written = graph.COUNTS.deferred, array.at_once_counts()[1]
    got = program(lazy, ours)
    # Every statement is a write made at once: the two on a row, the two on each element, and the last.
    assert graph.COUNTS.deferred == deferred and array.at_once_counts()[1] - written >= 4 * 2 + 4 * 4 * 2 + 1
    assert same(expected, got.gather()) and same(theirs, ours)


def test_a_whole_number_of_numpys_is_an_index_like_pythons(at_once: None):
    """`x[k]` and `x[k] = v` with what `np.argmax` or a random choice gives, made at once like any other."""
    def program(np: Any) -> Any:
        """Read and write through indices that are NumPy's integers."""
        x = np.zeros((5, 2)) + 1.0
        k, j = numpy.int64(3), numpy.intp(1)
        x[k] = 4.0
        x[k, j] += 2.5
        x[numpy.int32(0)] = x[k] * 2.0
        return x, x[k], x[k, j], x[numpy.uint8(4)]

    expected = program(numpy)
    deferred = graph.COUNTS.deferred
    got = program(lazy)
    assert graph.COUNTS.deferred == deferred
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours.gather())


def test_the_objects_the_core_makes_are_freed_and_made_again():
    """Weak references are told, pending work that is dropped is forgotten, and a long chain is dropped whole."""
    a = lazy.asarray(numpy.arange(4.0))
    waiting = graph.pending_count()
    made = a + 1.0
    told: list = []
    watching = weakref.ref(made, told.append)
    node = weakref.ref(made._node)
    del made
    assert watching() is None and len(told) == 1 and node() is None and graph.pending_count() == waiting
    for _ in range(3):
        chain = a
        for _ in range(20_000):
            chain = chain + 1.0
        del chain
    assert graph.pending_count() == waiting
    again = [(a * float(turn)).gather() for turn in range(300)]
    assert all(same(numpy.arange(4.0) * float(turn), value) for turn, value in enumerate(again))


def test_a_subclass_of_the_layers_array_is_freed_its_own_way():
    """The core frees the classes' own instances; one of a subclass goes back the way the subclass has."""
    class Named(LazyArray):
        """A lazy array with a name."""

        __slots__ = ("label",)

    told: list = []
    for turn in range(200):
        named = Named(graph.leaf(numpy.arange(3.0) + turn))
        named.label = [turn]
        watching = weakref.ref(named, told.append)
        assert same(numpy.arange(3.0) + turn, named.gather())
        assert same(numpy.arange(3.0) + turn + 1.0, (named + 1.0).gather())
        assert same(2.0 - (numpy.arange(3.0) + turn), (2.0 - named).gather())
        assert same((numpy.arange(3.0) + turn) ** 2, (named ** 2).gather())
        del named
        assert watching() is None
    assert len(told) == 200


def test_arrays_the_core_made_outlive_it(at_once: None):
    """Put away, the core leaves its arrays to Python: they give their values, get their nodes and are freed."""
    core = array._core
    made = [lazy.zeros(3) + float(turn) for turn in range(300)]
    views = [each[1:] for each in made[:50]]
    array.use_core(None)
    try:
        assert all(same(numpy.zeros(3) + float(turn), each.gather()) for turn, each in enumerate(made))
        assert all(same(numpy.zeros(2) + float(turn), each.gather()) for turn, each in enumerate(views))
        assert all(gc.is_tracked(each) for each in made)
        del made, views
        gc.collect()
    finally:
        array.use_core(core)


def test_a_failed_computation_that_holds_its_arrays_is_collected(at_once: None):
    """An error keeps the frames it passed through, which hold the arrays: a cycle the collector has to see whole."""
    array.at_once_below(100)

    def failing() -> weakref.ref:
        """Compute something that raises, catch it, and return a weak reference to an array it read."""
        small = lazy.zeros(3) + 1.0
        large = lazy.asarray(numpy.arange(200.0)) * 1.0
        with numpy.errstate(divide="raise", invalid="raise"):
            bad = large / (large * 0.0) + small.sum()
        watching = weakref.ref(small)
        with pytest.raises(FloatingPointError):
            bad.gather()
        return watching

    watching = failing()
    gc.collect()
    assert watching() is None


def test_a_name_of_the_stand_in_follows_the_module(at_once: None, monkeypatch: pytest.MonkeyPatch):
    """`np.tanh` is the answer given before while NumPy has what it had, and a new one when it has something else."""
    first = lazy.tanh
    assert lazy.tanh is first
    x = lazy.asarray(numpy.arange(4.0)) + 0.0
    assert same(numpy.tanh(numpy.arange(4.0)), lazy.tanh(x).gather())
    monkeypatch.setattr(numpy, "tanh", numpy.sinh)
    assert lazy.tanh is not first
    assert same(numpy.sinh(numpy.arange(4.0)), lazy.tanh(x).gather())
    assert lazy._module is numpy and lazy._strict is False
    with pytest.raises(AttributeError):
        lazy.no_such_name


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


@pytest.mark.parametrize("name", [program.name for program in corpus.programs() + codebases.programs() if program.eager])
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
