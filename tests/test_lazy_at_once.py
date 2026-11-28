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

from prototypes.lazy import LazyArray, NotGathered, array, evaluate, fuse, get_lazy, graph, namespace, record_graphs, rules
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
# The values a ufunc has an edge at, beside ordinary ones: not a number, the infinities, the two zeros, a denormal.
EDGES = (numpy.nan, numpy.inf, -numpy.inf, 0.0, -0.0, 1.5, -2.5, 5e-324)
# How many rows `edges` gives an array, and by how many places it turns the values of a second array of them.
EDGE_ROWS, EDGE_TURN = 3, 3
# How many random programs of a library's calls are run, and how many steps each has.
LIBRARY_PROGRAMS, LIBRARY_STEPS = 40, 60


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


def edges(turn: int = 0) -> numpy.ndarray:
    """Make an array whose rows are the EDGES, each row turned one place further than the row before.

    Args:
        turn: By how many places the first row is turned, so that two arrays meet each edge with another.

    Returns:
        A new float64 array of EDGE_ROWS rows.
    """
    return numpy.array([numpy.roll(EDGES, turn + row) for row in range(EDGE_ROWS)])


def layout(value: Any) -> tuple:
    """Say how an array lies in memory: its strides and whether it is in C order, in Fortran order."""
    value = numpy.asarray(value)
    return value.strides, value.flags.c_contiguous, value.flags.f_contiguous


def raised(call: Callable[[], Any]) -> tuple[type, str]:
    """Run something that must fail, and return the type and the words of what it raised."""
    with pytest.raises(Exception) as caught:
        call()
    return caught.type, str(caught.value)


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


def test_a_view_by_the_arrays_own_type_taken_at_once_is_numpys_view(at_once: None):
    """`x.view(type(x))` of an array that has its values is NumPy's view of it, and a write into it does not wait."""
    def program(np: Any) -> Any:
        """Fill an array through its view by type, as Matplotlib fills its array of normals, and scale it in place."""
        slope = np.asarray(numpy.arange(12.0).reshape(3, 4)) * 0.25
        normal = np.empty(slope.shape + (3,)).view(type(slope))
        normal[..., 0] = -slope
        normal[..., 1] = slope * slope
        normal[..., 2] = 1
        normal /= np.sqrt((normal * normal).sum())
        return normal, normal.view(type(normal))[..., 0], slope.view(type(slope)) + 1.0

    expected = program(numpy)
    got = program(lazy)
    for theirs, ours in zip(expected, got, strict=True):
        assert same(theirs, ours.gather())
    whole = lazy.zeros((4, 5)) + 1.0
    view = whole.view(type(whole))
    assert type(view._held) is numpy.ndarray and view._held.base is whole._held
    waiting = lazy.asarray(numpy.arange(200.0)) * 1.0
    array.at_once_below(100)
    view[1, 2] = 7.0  # Made at once, in the memory both look at, after what was waiting.
    assert not waiting._node.pending and whole.gather()[1, 2] == 7.0 and view.gather().base is whole.gather()


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


def test_one_whole_number_beside_a_large_array_waits_for_a_kernel():
    """With fusion on, `table[r, c] * x + y` past the limit is two nodes and one kernel call: one whole number is a
    number to a kernel, not an array of another type. An array of whole numbers with a dimension still is one."""
    if not array.core_in_use() or not fuse.enable():
        pytest.skip("needs the C core and the fused kernels")
    try:
        rng = numpy.random.default_rng(SEED)
        data, more = rng.standard_normal(2 * fuse.AT_ONCE_BELOW), rng.standard_normal(2 * fuse.AT_ONCE_BELOW)
        table, counts = numpy.arange(12).reshape(3, 4) - 5, numpy.arange(100)
        x, y, t, c = lazy.asarray(data), lazy.asarray(more), lazy.asarray(table), lazy.asarray(counts)
        cases = [
            (table[1, 2] * data + more, lambda: t[1, 2] * x + y),
            (data * table[0, 0] + more, lambda: x * t[0, 0] + y),
            (numpy.multiply(table[2, 3], data) + more, lambda: lazy.multiply(t[2, 3], x) + y),
            (numpy.multiply(data, numpy.asarray(7)) + more, lambda: lazy.multiply(x, lazy.asarray(numpy.asarray(7))) + y),
            (counts.sum() * data + more, lambda: c.sum() * x + y),
            (data * (counts > 5).any() + more, lambda: x * (c > 5).any() + y),
            (data * ((data * 2.0 > 0.5).sum() - 3) + more, lambda: x * ((x * 2.0 > 0.5).sum() - 3) + y),
            (numpy.maximum(table[1, 1], data) + more, lambda: lazy.maximum(t[1, 1], x) + y),
        ]
        fuse.STATS.reset()
        for place, (expected, write) in enumerate(cases):
            deferred = graph.COUNTS.deferred
            waits = write()
            assert graph.COUNTS.deferred >= deferred + 2, place
            assert same(expected, waits.gather()), place
        assert fuse.STATS.calls == {"fuse_add_scaled_f64": 7, "fuse_max_s_v_f64": 1, "fuse_add_v_v_f64": 1}
        made, deferred = array.at_once_counts()[0], graph.COUNTS.deferred
        whole = x * lazy.asarray(numpy.asarray([3]))
        assert array.at_once_counts()[0] - made == 1 and graph.COUNTS.deferred == deferred
        assert same(data * numpy.asarray([3]), whole.gather())
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


# ---- A ufunc that names its output, a call on plain arrays, and a call outside the deferred set, made at once.

@pytest.mark.parametrize("held", ("made", "handed", "plain"))
def test_a_ufunc_that_names_its_output_writes_at_once_when_nothing_is_pending(at_once: None, held: str):
    """`np.maximum(x, 0, out=x)` and its like are NumPy's own call into the memory the output has, for an array the
    layer made, one the program handed in and a plain `ndarray` alike; the call gives its output."""
    def program(np: Any, x: Any, y: Any) -> tuple:
        """Compute in place, as a library written for speed does, through an array and through its views."""
        first = np.maximum(x, 0.0, out=x)
        np.exp(x, out=x)
        np.add(x, y, out=x)
        np.negative(x[1], out=x[1])
        np.multiply(x[:, 2:], y[:, :-2], out=x[:, 2:])
        np.copysign(x, y, out=x)
        return first, x

    theirs, ours = edges(), edges()
    with numpy.errstate(all="ignore"):
        expected = program(numpy, theirs, edges(EDGE_TURN))
        target = {"made": lambda: lazy.asarray(ours) * 1.0, "handed": lambda: lazy.asarray(ours), "plain": lambda: ours}[held]()
        deferred, written = graph.COUNTS.deferred, array.at_once_counts()[1]
        first, got = program(lazy, target, lazy.asarray(edges(EDGE_TURN)))
    assert graph.COUNTS.deferred == deferred and array.at_once_counts()[1] - written == 6 and not graph.pending_count()
    assert same(expected[1], got.gather() if held != "plain" else got)
    if held == "plain":
        # What the stand-in gives for a plain output is a lazy array on it, whose memory the program knows.
        assert isinstance(first, LazyArray) and first._node.value is ours and first._node.exposed
    else:
        assert first is target
    if held != "made":
        assert same(theirs, ours)  # The array the program holds has every write.


def test_a_ufunc_that_names_its_output_waits_or_computes_first_while_something_is_pending(at_once: None):
    """A pending operation reads what the output held: a write into the layer's own array waits its turn, and one
    into an array the program holds is made after everything pending."""
    array.at_once_below(100)

    def program(np: Any, mine: numpy.ndarray, plain: numpy.ndarray) -> tuple:
        """Read three arrays in an operation that waits, then write into each."""
        large = np.asarray(numpy.arange(200.0).reshape(20, 10))
        own, handed = np.zeros(10) + 1.0, np.asarray(mine)
        reader = large + own + handed + plain
        np.multiply(own, 3.0, out=own)
        waited = getattr(getattr(reader, "_node", None), "pending", None)
        np.add(handed, 5.0, out=handed)
        np.subtract(plain, own, out=plain)
        plain += own
        return reader, own, handed, plain, waited

    theirs = program(numpy, numpy.arange(10.0), numpy.arange(10.0) * 2.0)
    mine, plain = numpy.arange(10.0), numpy.arange(10.0) * 2.0
    ours = program(lazy, mine, plain)
    assert ours[4] is True  # The write into the layer's own array was a node: the reader still waited after it.
    for expected, got in zip(theirs[:4], array.gather(*ours[:4])):
        assert same(expected, got)
    assert same(theirs[2], mine) and ours[3] is plain


def test_a_ufunc_that_names_its_output_fails_as_numpys_does(at_once: None):
    """Made at once, the error is NumPy's own in type and words, and the output is as it was."""
    whole = numpy.arange(3)
    for make in (lambda: lazy.asarray(whole.copy()) + 0, lambda: lazy.asarray(whole.copy()), lambda: whole.copy()):
        target, theirs = make(), whole.copy()
        assert raised(lambda: lazy.add(target, 0.5, out=target)) == raised(lambda: numpy.add(theirs, 0.5, out=theirs))
        longer = numpy.arange(4)
        assert (raised(lambda: lazy.add(target, lazy.asarray(longer), out=target))
                == raised(lambda: numpy.add(theirs, longer, out=theirs)))
        assert raised(lambda: lazy.add(target, out=target))[0] is raised(lambda: numpy.add(theirs, out=theirs))[0]
        assert same(whole, array.gather(target)[0])


def test_a_ufunc_given_its_output_by_position_writes_into_it(at_once: None):
    """`np.add(a, b, c)` is a write into `c`: what was written before it reads what `c` held, and the call gives `c`."""
    array.at_once_below(100)

    def program(np: Any) -> tuple:
        """Read an array in an operation that waits, then name it as an output by position."""
        a, b, c = (np.asarray(numpy.arange(4.0) + shift) for shift in range(3))
        reader = np.asarray(numpy.arange(200.0).reshape(50, 4)) + c
        result = np.add(a, b, c)
        return reader, c, result, result is c

    theirs, ours = program(numpy), program(lazy)
    assert ours[3] == theirs[3]
    for expected, got in zip(theirs[:3], array.gather(*ours[:3])):
        assert same(expected, got)


def test_a_plain_array_updated_in_place_by_a_lazy_one_is_numpys_write(at_once: None):
    """`a += x` on an `ndarray` `a` reaches the lazy `x` as `np.add(a, x, out=(a,))`: NumPy's write into `a`, which
    is what the statement gives, with nothing counted as computed early when `x` has its values."""
    def program(np: Any, a: numpy.ndarray) -> numpy.ndarray:
        """Update a plain array by arrays of `np`'s, as a library updates its parameters."""
        x = np.asarray(edges()) * 1.0
        a += x
        a *= x[0]
        a[1] -= x[2]
        numpy.multiply(x, 2.0, out=a)
        a /= np.sqrt(np.abs(x) + 1.0)
        return a

    theirs, ours = numpy.ones((EDGE_ROWS, len(EDGES))), numpy.ones((EDGE_ROWS, len(EDGES)))
    with numpy.errstate(all="ignore"):
        expected = program(numpy, theirs)
        deferred, written, needed = graph.COUNTS.deferred, array.at_once_counts()[1], dict(graph.COUNTS.needed)
        got = program(lazy, ours)
    assert got is ours and same(expected, ours)
    assert graph.COUNTS.deferred == deferred and array.at_once_counts()[1] - written == 5
    assert graph.COUNTS.needed == needed


def test_a_plain_array_updated_by_a_lazy_one_that_waits_or_is_strict(at_once: None):
    """A pending array is computed for the write, as it was; the values of a strict one do not leave into an `ndarray`."""
    array.at_once_below(100)
    values = numpy.arange(200.0)
    theirs, ours = numpy.ones(200), numpy.ones(200)
    theirs += values * 2.0 + 1.0
    waiting = lazy.asarray(values) * 2.0 + 1.0
    assert waiting._node.pending
    ours += waiting
    assert same(theirs, ours) and not waiting._node.pending
    strict = get_lazy(strict=True)
    with pytest.raises(NotGathered):
        ours += strict.asarray(values)
    with pytest.raises(NotGathered):
        numpy.add(ours, strict.asarray(values) * 2.0, out=ours)
    assert same(theirs, ours)


def test_a_call_on_plain_arrays_through_the_stand_in_is_numpys_call(at_once: None):
    """A function or a ufunc of the stand-in on `ndarray`s, and an operator of an `ndarray` with a lazy array, are
    NumPy's call on the arrays as they are: its values, its element type and the way its result lies in memory."""
    a, b = edges(), edges(EDGE_TURN)
    columns = numpy.asfortranarray(a)
    lb = lazy.asarray(b)
    tiny = numpy.finfo(a.dtype).eps  # A NumPy scalar among the arguments, as a library's own constants are.
    before = graph.COUNTS.deferred
    with numpy.errstate(all="ignore"):
        cases = [
            (numpy.sqrt(a), lazy.sqrt(a)), (numpy.add(a, b), lazy.add(a, b)), (numpy.add(a, b), lazy.add(a, lb)),
            (a + b, a + lb), (b * a, lb * a), (a ** b, a ** lb), (a - 1.5, lazy.subtract(a, 1.5)),
            (numpy.maximum(a, tiny), lazy.maximum(a, tiny)), (numpy.less(a, 0.0), lazy.less(a, 0.0)),
            (numpy.exp(columns), lazy.exp(columns)), (columns + b, columns + lb), (numpy.negative(a[:, ::2]), lazy.negative(a[:, ::2])),
            (numpy.dot(a, b.T), lazy.dot(a, b.T)), (numpy.matmul(a, b.T), lazy.matmul(a, lb.T)),
            (numpy.sum(a, axis=0), lazy.sum(a, axis=0)), (numpy.max(a, axis=1, keepdims=True), lazy.max(a, axis=1, keepdims=True)),
            (numpy.argmax(b), lazy.argmax(b)), (numpy.outer(a[0], b[0]), lazy.outer(a[0], lb[0])),
            (numpy.sqrt(a[0, 5]), lazy.sqrt(a[0, 5])),
        ]
    for place, (expected, got) in enumerate(cases[:-1]):
        assert isinstance(got, LazyArray) and not got._node.pending, place
        assert same(expected, got.gather()) and layout(expected) == layout(got.gather()), place
    assert type(cases[-1][1]) is type(cases[-1][0]) and same(*cases[-1])  # Of a number, NumPy's number.
    assert graph.COUNTS.deferred == before
    made = lazy.exp(a)
    made[0, 0] = 99.0  # A result is an array of its own: the write does not reach `a`.
    assert numpy.isnan(a[0, 0]) and made.gather()[0, 0] == 99.0


def test_a_call_on_plain_arrays_waits_where_a_kernel_could_make_it(at_once: None):
    """What the planner can compute still waits, on plain arrays as on the layer's; a strict stand-in gives strict results."""
    array.at_once_below(10, frozenset({numpy.add, numpy.multiply}), numpy.dtype(numpy.float64))
    a, b = numpy.arange(40.0), numpy.arange(40.0)[::-1].copy()
    waits = lazy.add(a, b)
    assert waits._node.pending and same(a + b, waits.gather())
    by_operator = a * lazy.asarray(b)
    assert by_operator._node.pending and same(a * b, by_operator.gather())
    root = lazy.sqrt(a)  # No kernel of the planner's: made now, whatever its size.
    assert not root._node.pending and same(numpy.sqrt(a), root.gather())
    product = lazy.dot(a, b)
    assert not product._node.pending and same(numpy.dot(a, b), product.gather())
    whole = numpy.arange(40)
    counted = lazy.add(whole, whole)  # Another element type than the kernels'.
    assert not counted._node.pending and same(whole + whole, counted.gather())
    strict = get_lazy(strict=True)
    for made in (strict.sqrt(a), strict.dot(a, b), strict.add(a[:4], b[:4]), strict.clip(a, 1.0, 2.0)):
        assert made._node.strict
    target = a.copy()
    assert strict.negative(a, out=target)._node.strict and same(-a, target)
    own = lazy.zeros(40) + 1.0
    strict.add(a, 1.0, out=own)  # What a strict stand-in writes is strict, and so is the array from then on.
    assert own._node.strict


def test_a_call_made_on_plain_arrays_raises_and_warns_where_it_is_written(at_once: None):
    """A warning of a call the core makes is the program's, at the line that wrote it, under the error settings in
    effect there: for a call on plain arrays, a ufunc that names its output and a plain array updated in place."""
    ones = numpy.ones(3)
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in divide") as caught:
        lazy.divide(ones, 0.0)
    assert caught[0].filename == __file__
    with pytest.warns(RuntimeWarning, match="invalid value encountered in sqrt") as caught:
        below = lazy.asarray(-ones)
        lazy.sqrt(below, out=below)
    assert caught[0].filename == __file__ and numpy.isnan(below.gather()).all()
    with pytest.warns(RuntimeWarning, match="divide by zero encountered in divide") as caught:
        target = ones.copy()
        target /= lazy.zeros(3)
    assert caught[0].filename == __file__ and numpy.isinf(target).all()
    with numpy.errstate(divide="raise"):
        for call in (lambda: lazy.divide(ones, 0.0), lambda: lazy.divide(ones, 0.0, out=ones.copy()),
                     lambda: ones.copy().__itruediv__(lazy.zeros(3)), lambda: lazy.clip(lazy.asarray(ones) / 0.0, 0.0, 1.0)):
            with pytest.raises(FloatingPointError):
                call()


def test_a_call_outside_the_deferred_set_is_made_at_once_and_counted(at_once: None):
    """`np.clip` and `np.average` have no rule: NumPy's call on the values, on arrays of the layer's and plain ones,
    with every call counted on its line for `lazy_report()` as a call made at once."""
    a, b = edges(), edges(EDGE_TURN)
    la, lb = lazy.asarray(a), lazy.asarray(b)
    tiny = numpy.finfo(a.dtype).eps
    weights = numpy.arange(1.0, len(EDGES) + 1.0)
    finite = numpy.arange(12.0).reshape(3, 4)
    calls = [
        (lambda np, x, y: np.clip(x, -1.0, 1.0), 1), (lambda np, x, y: np.clip(x, tiny, 1 - tiny), 1),
        (lambda np, x, y: np.clip(x, None, 0.5), 1), (lambda np, x, y: np.clip(x, y, 2.0), 1),
        (lambda np, x, y: np.clip(x, 0.0, numpy.inf), 1), (lambda np, x, y: np.clip(x[0], -0.0, 0.0), 1),
        (lambda np, x, y: np.average(x, axis=0), 0), (lambda np, x, y: np.average(x, axis=1, weights=weights), 0),
        (lambda np, x, y: np.average(y), 0),
    ]
    clipped, deferred = graph.COUNTS.at_once.get("np.clip", 0), graph.COUNTS.deferred
    sites = sum(count for site, count in graph.SITES.items() if site[:3] == (graph.AT_ONCE, "np.clip", __file__))
    with numpy.errstate(all="ignore"):
        for place, (call, _) in enumerate(calls):
            expected = call(numpy, a, b)
            for got in (call(lazy, la, lb), call(lazy, a, b), call(lazy, a, lb)):
                assert isinstance(got, LazyArray) and not got._node.pending and got._node.kind == graph.AT_ONCE, place
                assert same(expected, got.gather()) and layout(expected) == layout(got.gather()), place
        both = lazy.average(lazy.asarray(finite), axis=0, weights=lazy.asarray(numpy.arange(1.0, 4.0)), returned=True)
    assert all(same(theirs, ours.gather()) for theirs, ours
               in zip(numpy.average(finite, axis=0, weights=numpy.arange(1.0, 4.0), returned=True), both, strict=True))
    made = 3 * sum(counted for _, counted in calls)
    assert graph.COUNTS.at_once["np.clip"] - clipped == made and graph.COUNTS.deferred == deferred
    assert sum(count for site, count in graph.SITES.items() if site[:3] == (graph.AT_ONCE, "np.clip", __file__)) - sites == made
    kept = lazy.clip(la, -1.0, 1.0)
    kept[0, 0] = 99.0  # A result is an array of its own.
    assert numpy.isnan(a[0, 0]) and kept.gather()[0, 0] == 99.0


def test_a_call_outside_the_deferred_set_computes_what_it_reads_and_takes_any_size(at_once: None):
    """A pending argument is computed for the call, as it was, and an array past the limit is clipped at once."""
    array.at_once_below(100)
    values = numpy.arange(200.0)
    waiting = lazy.asarray(values) * 2.0
    assert waiting._node.pending
    clipped = lazy.clip(waiting, 10.0, 50.0)
    assert not clipped._node.pending and not waiting._node.pending and same(numpy.clip(values * 2.0, 10.0, 50.0), clipped.gather())
    for large in (lazy.asarray(values), values):
        made = lazy.clip(large, 10.0, 50.0)
        assert not made._node.pending and same(numpy.clip(values, 10.0, 50.0), made.gather())
    into = lazy.asarray(values.copy())
    assert lazy.clip(into, 10.0, 50.0, out=into) is into and same(numpy.clip(values, 10.0, 50.0), into.gather())
    with pytest.raises(NotGathered):
        float(get_lazy(strict=True).average(values))


def test_the_functions_called_whatever_their_size_have_no_rule_and_give_an_array_of_their_own():
    """What `array.NEVER_DEFERRED` says of its functions, which the core's road rests on."""
    values = numpy.arange(12.0).reshape(3, 4)
    node = lazy.asarray(values)._node
    for function in array.NEVER_DEFERRED:
        assert rules.plan(function, (node,), {}) is None and rules.plan(function, (node, 0.0, 1.0), {}) is None
    for result in (numpy.clip(values, None, None), numpy.clip(values, -numpy.inf, numpy.inf), numpy.clip(values, values, values),
                   numpy.average(values, axis=0), numpy.average(values, axis=0, weights=numpy.ones(3)),
                   *numpy.average(values, axis=1, keepdims=True, returned=True)):
        assert not numpy.shares_memory(result, values)


def test_the_core_makes_these_calls_without_the_python_roads(at_once: None, monkeypatch: pytest.MonkeyPatch):
    """None of them reaches the Python that makes a write at once, makes a call at once or makes a plain array lazy:
    that way round is what each cost 1 to 8 µs over NumPy's own call for."""
    def reached(*args: Any, **kwargs: Any) -> None:
        """Fail the test: a Python road was taken."""
        raise AssertionError("the call went the Python road")

    a, b = numpy.arange(1.0, 13.0).reshape(3, 4), numpy.arange(12.0).reshape(3, 4)
    made, handed = lazy.zeros((3, 4)) + 1.0, lazy.asarray(b.copy())
    tiny = numpy.finfo(a.dtype).eps
    for owner, name in ((array, "_write"), (array, "_at_once"), (namespace, "wrap")):
        monkeypatch.setattr(owner, name, reached)
    for target in (made, handed, a.copy()):
        lazy.maximum(target, 2.0, out=target)
        lazy.exp(target, out=target)
        lazy.add(target, handed, out=target)
    plain = a.copy()
    plain += made
    plain *= handed
    numpy.subtract(plain, made, out=plain)
    results = [lazy.sqrt(a), lazy.add(a, b), a + made, handed - a, lazy.dot(a, b.T), lazy.sum(a, axis=0),
               lazy.clip(a, tiny, 1 - tiny), lazy.clip(made, 2.0, 5.0), lazy.average(handed, axis=0), lazy.maximum(a, tiny)]
    assert all(isinstance(result, LazyArray) and not result._node.pending for result in results)


def test_the_quick_roads_stand_aside_while_a_recording_listens(at_once: None):
    """A recording has a node for each array that comes in and a line for each write made at once, as without them."""
    theirs, ours = numpy.arange(6.0) - 2.0, numpy.arange(6.0) - 2.0
    numpy.maximum(theirs, 0.0, out=theirs)
    with record_graphs() as recording:
        lazy.maximum(ours, 0.0, out=ours)
        root = lazy.sqrt(ours)
        ours += root
        clipped = lazy.clip(root, 0.5, 1.0)
    theirs += numpy.sqrt(theirs)
    assert same(theirs, ours) and same(numpy.clip(numpy.sqrt(numpy.maximum(numpy.arange(6.0) - 2.0, 0.0)), 0.5, 1.0), clipped.gather())
    report = recording.report()
    assert [row.call for row in report.writes] == ["np.maximum"] and [row.call for row in report.eager] == ["np.clip"]


def library_program(np: Any, seed: int) -> list:
    """Run a program of the calls a library makes on its own arrays, and return every array.

    Half of the arrays are plain `ndarray`s whichever module computes, as a
    library's are once it has validated its input, and half are `np`'s. The
    calls are those of a library written for speed: a ufunc that names its
    output, a plain array updated in place, a function on plain arrays,
    `np.clip`, and now and then an operation large enough to wait.

    Args:
        np: The `numpy` module or the lazy stand-in.
        seed: The seed of the program: the same calls are made for either module.

    Returns:
        The arrays the program ended with.
    """
    rng = random.Random(seed)
    values = numpy.random.default_rng(seed)

    def draw() -> numpy.ndarray:
        """Draw a small array of numbers of both signs, zeros among them."""
        return values.integers(-3, 4, (SIDE, SIDE)).astype(float)

    pool = [draw(), draw(), np.asarray(draw()), np.asarray(draw()) * 1.0]
    large = np.asarray(values.standard_normal((LARGE_SIDE, LARGE_SIDE)))
    for _ in range(LIBRARY_STEPS):
        kind = rng.randrange(12)
        place = rng.randrange(len(pool))
        target, other = pool[place], rng.choice(pool)
        if kind == 0:
            np.maximum(target, 0.0, out=target)
        elif kind == 1:
            np.tanh(target, out=target)
        elif kind == 2:
            np.add(target, other, out=target)
        elif kind == 3:
            # Through `np`: a write from one `ndarray` into another is NumPy's alone, which the layer does not see.
            target += np.asarray(other)
        elif kind == 4:
            target *= np.sqrt(np.abs(other))
        elif kind == 5:
            pool[place] = np.clip(target, -2.5, 2.5)
        elif kind == 6:
            pool[place] = np.dot(target, other)
        elif kind == 7:
            target -= np.sum(other, axis=0)
        elif kind == 8:
            large = large * 1.0001 + 0.5  # Stays pending under a limit below its size.
        elif kind == 9:
            np.subtract(target, large[:SIDE, :SIDE], out=target)
        elif kind == 10:
            np.multiply(target[1:], other[:-1], out=target[1:])
        elif kind == 11:
            pool[place] = other / (np.abs(target) + 1.0)
    return pool + [large]


@pytest.mark.parametrize("limit", (NOTHING, SIDE * SIDE + 1, EVERYTHING))
def test_random_programs_of_a_librarys_calls_give_numpys_arrays(limit: int):
    """In-place ufuncs, updates of plain arrays and calls on them in any order give NumPy's arrays, whatever waits."""
    if limit and not array.core_in_use():
        pytest.skip("calls are made at once by the C core only")
    array.at_once_below(limit)
    try:
        for seed in range(LIBRARY_PROGRAMS):
            with numpy.errstate(all="ignore"):
                expected = library_program(numpy, seed)
                got = array.gather(*library_program(lazy, seed))
            assert len(expected) == len(got)
            for place, (theirs, ours) in enumerate(zip(expected, got)):
                assert same(theirs, ours), (seed, place)
    finally:
        array.at_once_below(NOTHING)


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
