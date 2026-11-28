"""Tests of fusion in the lazy layer's prototype, prototypes/lazy/fuse.py: every fused result held to NumPy.

The fused kernels themselves are checked by the judge (`uv run judge check
<name>`). These tests are of what stands around them: that a kernel's
pattern is the one its oracle writes, that the module which calls the
kernels walks arrays of every layout, and that a program computed with
fusion gives what the same lines give with the real `numpy` module, in type,
element type, shape, bits, warnings and errors. Run with
`uv run --group dev pytest tests/test_lazy_fusion.py`; the programs that need
SciPy are skipped without `--group apps`.

One thing about how these tests are written. pytest rewrites an `assert` so
that it can show the values in it, and keeps every value of the expression
until the statement is over. A lazy expression written inside an `assert`
therefore has all its intermediates held, and the layer, rightly, does not
compute past a value something still holds. So an expression is written on
a line of its own, or by `gathered`, and only its value goes into the `assert`.
"""
from __future__ import annotations

import gc
import subprocess
import warnings
from typing import Any, Iterator

import numpy
import pytest

from prototypes.lazy import LazyArray, array, evaluate, fuse, get_lazy, graph, make_kernel, record_graphs
from judge.spec import In, Scalar, load_spec
from prototypes.lazy.programs import codebases, corpus, limit

# Seed of every random array in these tests.
SEED = 0
# The shapes a kernel's pattern is expected to be, by kernel.
PATTERNS = {
    "fuse_add_product_f64": "add(V0,mul(V1,V2))",
    "fuse_add_scaled_f64": "add(V0,mul(S0,V1))",
    "fuse_flux_limit_f64": "where(gt(mul(V0,sub(V1,V2)),S0),S1,V3)",
    "fuse_flux_update_f64": "sub(V0,mul(V1,sub(add(V2,sub(V3,V4)),V5)))",
    "fuse_laplacian5_f64": "sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))",
    "fuse_mul_difference_f64": "mul(V0,sub(V1,V2))",
    "fuse_mul_shifted_f64": "mul(V0,add(S0,V1))",
    "fuse_poisson_step_f64": "sub(div(add(mul(S0,add(V0,V1)),mul(S1,add(V2,V3))),S2),mul(S3,V4))",
    "fuse_product3_f64": "mul(V0,mul(V1,V2))",
    "fuse_scaled_sum_f64": "mul(S0,add(V0,V1))",
    "fuse_sub_product_f64": "sub(V0,mul(V1,V2))",
    "fuse_sub_scaled_f64": "sub(V0,mul(S0,V1))",
    "fuse_sum3_f64": "add(V0,add(V1,V2))",
    "fuse_sum_squares_f64": "sum(sq(V0))",
}
# Lengths at which a kernel is tried through the layer: either side of its block of eight, and longer.
LENGTHS = (1, 2, 7, 8, 9, 15, 16, 17, 64, 100, 1000)
# Lengths at which the sum of squares is compared with NumPy's: every one up to here, where the pairwise tree
# changes shape with nearly every length, and a few beyond.
SWEEP, LONG = 700, (4_096, 10_001, 65_537, 300_000)
# The values that make floating-point arithmetic do something unusual.
SPECIAL = (0.0, -0.0, numpy.inf, -numpy.inf, numpy.nan, 1e308, -1e308, 5e-324, 1e-310)

lazy = get_lazy()


@pytest.fixture(scope="session")
def built_kernels() -> Any:
    """Build the fused kernels and the module that calls them, once for the whole run.

    Returns:
        None, or the reason they could not be built, as a string.
    """
    from prototypes.lazy import build_fused
    try:
        build_fused.build()
    except build_fused.NoCompiler as error:
        return str(error)
    except subprocess.CalledProcessError as error:
        return f"the fused kernels do not build here: {error.stderr}"
    return None


@pytest.fixture(autouse=True)
def fusing(built_kernels: Any) -> Iterator[None]:
    """Have every computation of a test planned for fusion, and none afterwards.

    Args:
        built_kernels: Why the kernels could not be built, or None.
    """
    if built_kernels is not None:
        pytest.skip(built_kernels)
    assert fuse.enable()
    fuse.STATS.reset()
    fuse._flagged.clear()
    # The tests are of what is fused, on arrays of every length; what is worth fusing is another question. So
    # no floor, and no small call made at once in place of a node: the tests of that turn it on themselves.
    floor, fuse.MIN_ELEMENTS = fuse.MIN_ELEMENTS, 1
    array.at_once_below(0)
    try:
        yield
    finally:
        fuse.MIN_ELEMENTS = floor
        fuse.disable()


def same(expected: Any, got: Any) -> bool:
    """Say whether two results are the same in type, element type, shape and bits.

    Args:
        expected: What eager NumPy returned.
        got: What `gather()` returned.

    Returns:
        True when nothing distinguishes them.
    """
    return (type(expected) is type(got) and expected.dtype == got.dtype and expected.shape == got.shape
            and expected.tobytes() == got.tobytes())


def same_but_for_which_nan(expected: Any, got: Any) -> bool:
    """Say whether two results are the same but for which NaN stands where both have one.

    Which NaN comes of two of different sign is the one thing the design
    leaves open (docs/lazy-fusion-design.md, "Why the bits are NumPy's").
    On x86-64 an invalid operation makes a NaN with its sign bit set and
    `np.nan` has it clear, so arrays that hold `np.nan` beside infinities
    and zeros have both kinds, and NumPy's own loops do not keep to one
    order of operands there.

    Args:
        expected: What eager NumPy returned, as an array.
        got: What `gather()` returned, as an array.

    Returns:
        True when they have one type, element type and shape, a NaN in the
        same places, and the same bits everywhere else.
    """
    if type(expected) is not type(got) or expected.dtype != got.dtype or expected.shape != got.shape:
        return False
    theirs, ours = numpy.isnan(expected), numpy.isnan(got)
    return bool((theirs == ours).all()) and expected[~theirs].tobytes() == got[~ours].tobytes()


def arrays(count: int, shape: tuple[int, ...], special: bool = False) -> list[numpy.ndarray]:
    """Draw float64 arrays of one shape.

    Args:
        count: How many.
        shape: Their shape.
        special: Whether to put a value of SPECIAL into about a tenth of the elements.

    Returns:
        The arrays.
    """
    rng = numpy.random.default_rng(SEED + count + sum(shape))
    drawn = [rng.standard_normal(shape) for _ in range(count)]
    if special:
        for array in drawn:
            places = rng.random(shape) < 0.1
            array[places] = rng.choice(SPECIAL, int(places.sum()))
    return drawn


def gathered(write: Any) -> Any:
    """Write a lazy expression and gather it, holding none of its intermediates.

    Args:
        write: A function of no arguments that writes the expression and returns its lazy array.

    Returns:
        What `gather()` gives.
    """
    return write().gather()


def calls() -> int:
    """Return how many kernel calls fusion has made since the test began."""
    return sum(fuse.STATS.calls.values())


def expressions(np: Any, a: Any, b: Any, c: Any, d: Any, e: Any, f: Any) -> dict[str, Any]:
    """Write one expression per kernel, as a program would, with a module of the caller's choice.

    Args:
        np: The `numpy` module or the lazy stand-in.
        a, b, c, d, e, f: Six arrays of one shape, `ndarray`s or lazy.

    Returns:
        Kernel name -> the expression's result.
    """
    return {
        "fuse_add_product_f64": a + b * c,
        "fuse_add_scaled_f64": a + 0.37 * b,
        "fuse_flux_limit_f64": np.where(a * (b - c) > 0, 0, a),
        "fuse_flux_update_f64": a - b * (c - d + e - f),
        "fuse_laplacian5_f64": 4.0 * a - (b + c + d + e),
        "fuse_mul_difference_f64": a * (b - c),
        "fuse_mul_shifted_f64": a * (1 + b),
        "fuse_poisson_step_f64": ((a + b) * 0.3 + (c + d) * 0.7) / 2.1 - 0.05 * e,
        "fuse_product3_f64": a * (b * c),
        "fuse_scaled_sum_f64": 0.5 * (a + b),
        "fuse_sub_product_f64": a - b * c,
        "fuse_sub_scaled_f64": a - 0.1 * b,
        "fuse_sum3_f64": a + (b + c),
        "fuse_sum_squares_f64": np.sum(a ** 2),
    }


def test_the_library_has_the_shape_each_kernels_oracle_writes():
    """A kernel's pattern is read off its spec's oracle, and is the shape the kernel was written for."""
    by_kernel = {kernel.name: name for name, kernel in fuse.LIBRARY.items() if not kernel.mean}
    # The kernels `make_kernel.py` wrote are named after their shapes; the others are in the table above.
    written = {make_kernel.name_of(shape): shape for shape in make_kernel.SHAPES}
    assert {name: shape for name, shape in by_kernel.items() if name not in written} == PATTERNS
    assert {name: shape for name, shape in by_kernel.items() if name in written} == written
    # A kernel that sums also serves the mean of the same expression.
    assert fuse.LIBRARY["mean(sq(V0))"].name == "fuse_sum_squares_f64" and fuse.LIBRARY["mean(sq(V0))"].mean


def test_an_oracle_that_is_no_expression_has_no_pattern():
    """An oracle that computes its result, or is more than one expression, is refused: it could not be matched."""
    with pytest.raises(fuse.NoPattern):
        fuse.pattern_of(lambda a: numpy.sort(a), 1, 0)
    with pytest.raises(fuse.NoPattern):
        fuse.pattern_of(lambda a, b: numpy.exp(a) + b * 2.0, 2, 0)
    with pytest.raises(fuse.NoPattern):
        fuse.pattern_of(lambda a, b: a * 2.0 + 1.0, 2, 0)  # Does not read `b`.


@pytest.mark.parametrize("length", LENGTHS)
def test_every_kernels_expression_gives_numpys_bits(length: int):
    """One expression per kernel, on ordinary and on special values, is NumPy's bit for bit and is fused."""
    for special in (False, True):
        operands = arrays(6, (length,), special)
        with numpy.errstate(all="ignore"):
            expected = expressions(numpy, *operands)
            got = expressions(lazy, *(lazy.asarray(operand) for operand in operands))
            for name, value in got.items():
                before = fuse.STATS.calls.get(name, 0)
                assert same(expected[name], value.gather()), name
                assert fuse.STATS.calls.get(name, 0) == before + 1, name
    assert fuse.STATS.declined == 0


def test_the_sum_of_squares_adds_in_numpys_order_at_every_length():
    """The fused sum of squares, and the mean, match NumPy at every length: the pairwise tree changes with each."""
    rng = numpy.random.default_rng(SEED)
    for length in [*range(1, SWEEP), *LONG]:
        x = rng.standard_normal(length) * 10.0 ** rng.integers(-3, 4)
        held = lazy.asarray(x)
        total, mean, squared = numpy.sum(held ** 2), (held * held).mean(), numpy.square(held).sum()
        assert same(numpy.sum(x ** 2), total.gather()), length
        assert same((x * x).mean(), mean.gather()), length
        assert same(numpy.square(x).sum(), squared.gather()), length
    assert fuse.STATS.calls["fuse_sum_squares_f64"] == 3 * (SWEEP - 1 + len(LONG))


def test_a_reduction_over_more_than_one_dimension_is_numpys():
    """A C-contiguous array of any shape is summed in the order it lies in memory, by NumPy and by the kernel."""
    for shape in ((7, 9), (40, 33), (3, 5, 17), (129, 2), (2, 200)):
        (x,) = arrays(1, shape)
        total, mean = numpy.sum(lazy.asarray(x) ** 2), (lazy.asarray(x) ** 2).mean()
        assert same(numpy.sum(x ** 2), total.gather()), shape
        assert same((x ** 2).mean(), mean.gather()), shape
    assert calls() == 10 and fuse.STATS.declined == 0


def test_a_reduction_of_an_array_that_is_not_contiguous_is_left_to_numpy():
    """Where NumPy's order of addition is not the order in memory, the kernel is not called and the bits are NumPy's."""
    (x,) = arrays(1, (40, 30))
    for view in (x[:, ::2], x.T, x[::3]):
        total = numpy.sum(lazy.asarray(view) ** 2)
        assert same(numpy.sum(view ** 2), total.gather())
    assert "fuse_sum_squares_f64" not in fuse.STATS.calls


def test_rows_slices_and_broadcast_operands_are_walked():
    """A kernel is called once per contiguous run: over slices of a grid, a row repeated, and a number per row."""
    a, b, c = arrays(3, (20, 33))
    row, column = a[3], b[:, :1]
    la, lb, lc = (lazy.asarray(operand) for operand in (a, b, c))
    cases = [
        (a[1:-1, 1:-1] + 0.5 * b[2:, 1:-1], la[1:-1, 1:-1] + 0.5 * lb[2:, 1:-1]),
        (a + b * row, la + lb * lazy.asarray(row)),
        (a + column * b, la + lazy.asarray(column) * lb),
        (a[:, 5:20] - b[:, 4:19] * c[:, 6:21], la[:, 5:20] - lb[:, 4:19] * lc[:, 6:21]),
        (a[::2] * (b[1::2] - c[::2]), la[::2] * (lb[1::2] - lc[::2])),
    ]
    for expected, got in cases:
        assert same(expected, got.gather())
    assert calls() == len(cases) and fuse.STATS.declined == 0


def test_a_layout_the_kernels_do_not_read_is_left_to_numpy():
    """A reversed array, a Fortran-ordered one and a transposed one are not walked; the results are NumPy's."""
    a, b = arrays(2, (30, 40))
    fortran = numpy.asfortranarray(a)
    cases = [
        (a[::-1] + 0.5 * b, lazy.asarray(a)[::-1] + 0.5 * lazy.asarray(b)),
        (fortran + 0.5 * fortran, lazy.asarray(fortran) + 0.5 * lazy.asarray(fortran)),
        (a.T + 0.5 * b.T, lazy.asarray(a).T + 0.5 * lazy.asarray(b).T),
    ]
    for expected, got in cases:
        value = got.gather()
        assert same(expected, value) and value.flags.c_contiguous == expected.flags.c_contiguous
    assert calls() == 0 and fuse.STATS.declined == len(cases)


# The shortest row the module gathers: `SHORTEST_RUN` of `_fused.c`.
SHORTEST_GATHERED = 32
# Shapes whose rows are shorter than a kernel is called for, and one whose rows are long, with what is cut off them.
SHORT_ROWS = (((7, 9, 26), 1), ((3, 40, 12), 2), ((50, 5), 1), ((6, 4, 5, 16), 3), ((4, 300), 2))


@pytest.mark.parametrize("shape, cut", SHORT_ROWS)
def test_rows_at_one_pitch_are_walked_straight_through(shape: tuple[int, ...], cut: int):
    """Slices that cut the innermost dimension: every array has the same elements between its rows, and none is copied."""
    a, b, c = arrays(3, shape)
    profile = arrays(1, (shape[-1],))[0]
    inner = (slice(1, -1),) * (len(shape) - 1)

    def program(np: Any) -> Any:
        """Stencil-like expressions on slices of three grids and a profile along the innermost dimension."""
        x, y, z, depth = np.asarray(a), np.asarray(b), np.asarray(c), np.asarray(profile)
        left, right = inner + (slice(None, -cut),), inner + (slice(cut, None),)
        first = (x[left] - y[right]) * z[left] + 0.25 * x[right]
        second = (x[left] + depth[:-cut]) * (y[left] - depth[cut:]) - z[right]
        return first, second

    expected = program(numpy)
    got = program(lazy)
    for theirs, ours in zip(expected, got):
        value = ours.gather()
        assert same(theirs, value) and value.flags.c_contiguous
    assert fuse.STATS.declined == 0 and fuse.STATS.fused_nodes >= 6


@pytest.mark.parametrize("shape, cut", SHORT_ROWS)
def test_arrays_at_more_than_one_pitch_are_gathered(shape: tuple[int, ...], cut: int):
    """A slice of a grid beside an array computed from one, a column that steps, and a result stored into a slice."""
    a, b, c = arrays(3, shape)

    def program(np: Any) -> Any:
        """Expressions whose operands lie at the grid's pitch, at their own, and stepping over elements."""
        x, y, z = np.asarray(a), np.asarray(b), np.asarray(c)
        left, right = (Ellipsis, slice(None, -cut)), (Ellipsis, slice(cut, None))
        made = x[left] * y[right] + 0.5 * z[left]  # An array of its own, one row after the other.
        mixed = (made - x[right]) * (y[left] + made) - 0.125 * z[right]
        stepped = (x[..., ::2] - y[..., ::2]) * z[..., ::2] + x[..., ::2]
        into = np.zeros(shape)
        into[left] = (made + y[left]) * (z[right] - 2.0 * x[left])
        into[right] += made * 0.5 - into[left] * y[right]
        return made, mixed, stepped, into

    expected = program(numpy)
    got = program(lazy)
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours.gather())
    # Rows of a few elements at more than one pitch are left to NumPy; longer ones are gathered.
    assert fuse.STATS.fused_nodes >= (8 if shape[-1] - cut >= SHORTEST_GATHERED else 0)


# The shortest row a program of several steps is run on by itself: `SHORTEST_ROW` of `_fused.c`.
SHORTEST_ROW = 16


@pytest.mark.parametrize("length", (4, 15, 16, 25, 31, 40))
def test_short_rows_at_two_pitches_are_a_program_each_from_sixteen_elements(length: int):
    """Slices of grids beside arrays of their own: no pitch to walk at, and nothing a kernel cannot read in place."""
    grid_a, grid_c = arrays(2, (6, 5, length + 1))
    b, d = arrays(2, (6, 5, length))
    x, z = lazy.asarray(grid_a), lazy.asarray(grid_c)
    got = (x[..., :-1] + lazy.asarray(b)) * z[..., :-1] - lazy.asarray(d)
    assert same((grid_a[..., :-1] + b) * grid_c[..., :-1] - d, got.gather())
    assert fuse.STATS.declined == (0 if length >= SHORTEST_ROW else 1)
    assert sum(fuse.STATS.calls.values()) == (2 if length >= SHORTEST_ROW else 0)


# How many times their length rows of fewer than `SHORTEST_ROW` elements may lie apart and be walked straight
# through: `MOST_SPREAD` of `_fused.c`.
MOST_SPREAD = 8


@pytest.mark.parametrize("width, pitch", ((1, 3), (2, 4), (2, 16), (2, 17), (6, 10), (15, 120), (15, 121), (16, 40), (30, 39)))
def test_short_rows_with_much_between_them_are_walked_or_left_to_numpy(width: int, pitch: int):
    """Columns of two tables: rows of a few elements are walked through up to eight times their length, longer rows
    are a program each, and rows of a few elements farther apart than that are NumPy's."""
    a, b = arrays(2, (70, pitch))
    x, y = lazy.asarray(a), lazy.asarray(b)
    got = (x[:, :width] - y[:, -width:]) * x[:, -width:] + 0.5 * y[:, :width]
    assert same((a[:, :width] - b[:, -width:]) * a[:, -width:] + 0.5 * b[:, :width], got.gather())
    made = width >= SHORTEST_ROW or pitch <= width * MOST_SPREAD
    if width > 1:  # A row of one element is a number to the planner, not an array.
        assert fuse.STATS.declined == (0 if made else 1) and (fuse.STATS.fused_nodes > 0) == made


@pytest.mark.parametrize("width", (2, 4, 8, 15, 16))
def test_one_operation_is_a_kernels_where_numpy_would_make_it_a_short_row_at_a_time(width: int):
    """`table - row`, a value for each row, a slice with elements between its rows and rows at two pitches are worth
    a kernel by themselves; two arrays in one run are not, nor anything but arithmetic on rows longer than eight."""
    table, other = arrays(2, (500, width))
    wide = arrays(1, (500, 2 * width))[0]
    row, column = arrays(1, (width,))[0], arrays(1, (500,))[0]
    t, o, w, r, c = (lazy.asarray(each) for each in (table, other, wide, row, column))
    short, slight = width < fuse.SHORT_ROW, width <= fuse.SLIGHT_ROW
    cases = (
        (lambda: t - r, table - row, short),
        (lambda: r * t, row * table, short),
        (lambda: w[:, width:] * 0.5, wide[:, width:] * 0.5, short),
        (lambda: 0.5 / w[:, width:], 0.5 / wide[:, width:], short),
        (lambda: w[:, :width] - w[:, width:], wide[:, :width] - wide[:, width:], short),
        (lambda: t * c[:, None], table * column[:, None], short),
        (lambda: c[:, None] - t, column[:, None] - table, short),
        (lambda: t / c[:, None], table / column[:, None], short),
        (lambda: w[:, :width] * o, wide[:, :width] * other, short),
        (lambda: o + w[:, width:], other + wide[:, width:], short),
        (lambda: lazy.abs(w[:, width:]), numpy.abs(wide[:, width:]), slight),
        (lambda: lazy.maximum(w[:, :width], w[:, width:]), numpy.maximum(wide[:, :width], wide[:, width:]), slight),
        (lambda: t * o, table * other, False),
    )
    for program, expected, by_kernel in cases:
        before = sum(fuse.STATS.calls.values())
        assert same(expected, program().gather())
        assert sum(fuse.STATS.calls.values()) - before == (1 if by_kernel else 0)


@pytest.mark.parametrize("shape", ((7, 5, 3), (4, 3, 6, 2), (9, 15), (3, 1, 4)))
def test_rows_of_one_operation_are_made_in_any_layout(shape: tuple[int, ...]):
    """The module's loop for one operation on short rows: arrays at any pitch, a value for each row, results in place
    and into parts, in two dimensions and more, with NumPy's bits."""
    width = shape[-1]
    wide_shape = shape[:-1] + (2 * width + 1,)
    a, b = arrays(2, wide_shape)
    values = arrays(1, shape[:-1] + (1,))[0]

    def program(np: Any) -> list:
        """One operation at a time on slices of two grids, into new arrays, in place and into a part of a third."""
        x, y, v = np.asarray(a.copy()), np.asarray(b.copy()), np.asarray(values)
        left, right = x[..., :width], y[..., width + 1:]
        made = [left + right, left - right, left * right, left / right, left * v, v - right, right / v, 2.5 - left, left / 0.5]
        into = np.zeros(wide_shape)
        into[..., 1:width + 1] = left * right
        x[..., :width] += right * 0.5
        y[..., width + 1:] -= v
        return made + [into, x, y]

    with numpy.errstate(all="ignore"):
        expected = program(numpy)
        got = program(lazy)
        for theirs, ours in zip(expected, got):
            assert same(theirs, ours.gather())


def test_enabling_fusion_installs_the_packages_kernels_and_disabling_takes_them_out():
    """`fuse.enable()` has `hotkernels.install()` in effect; an installation the program made itself is left alone."""
    hotkernels = pytest.importorskip("hotkernels")
    fuse.disable()
    hotkernels.uninstall()
    try:
        assert fuse.enable(install=False) and not hotkernels.report().installed
        fuse.disable()
        assert fuse.enable() and hotkernels.report().installed
        fuse.disable()
        assert not hotkernels.report().installed
        hotkernels.install()
        assert fuse.enable()
        fuse.disable()
        assert hotkernels.report().installed
    finally:
        hotkernels.uninstall()
        fuse.enable()


def test_an_operation_written_over_by_the_next_line_is_one_program_with_it():
    """`e = a - b; e /= s`: what the array held is an operation the program makes, so there is no memory of it yet."""
    a, b = arrays(2, (300, 40))
    x, y = lazy.asarray(a), lazy.asarray(b)
    error = (x - y + 3.0) * x - 3.0
    error /= 0.25
    expected = (a - b + 3.0) * a - 3.0
    expected /= 0.25
    assert same(expected, error.gather())
    assert fuse.STATS.declined == 0 and fuse.STATS.fused_nodes >= 4


@pytest.mark.parametrize("pitch", (2, 5, 40))
def test_a_program_of_one_kernel_on_rows_of_one_element_gives_numpys_values(pitch: int):
    """`x[:, :1]` of a table: no call is made for one element at a time."""
    a, b = arrays(2, (3000, pitch))
    x, y = lazy.asarray(a), lazy.asarray(b)
    got = x[:, :1] * y[:, -1:] + y[:, :1]
    assert same(a[:, :1] * b[:, -1:] + b[:, :1], got.gather())
    assert sum(fuse.STATS.calls.values()) <= 1


def test_what_lies_between_rows_is_never_written():
    """The steps walk through the elements between rows; the result's rows are copied, and nothing else is touched."""
    shape = (6, 8, 26)
    a, b = arrays(2, shape)
    canvas = numpy.full(shape, 7.0)

    def program(np: Any, into: Any) -> Any:
        """Store an expression on slices into the same slice of an array that holds something else."""
        x, y = np.asarray(a), np.asarray(b)
        left, right = (slice(1, -1), slice(1, -1), slice(None, -1)), (slice(1, -1), slice(1, -1), slice(1, None))
        into[left] = (x[left] - y[right]) * x[right] + 0.5 * y[left]
        return into

    expected = program(numpy, canvas.copy())
    got = program(lazy, lazy.asarray(canvas.copy()) + 0.0).gather()
    assert same(expected, got) and got[0, 0, 0] == 7.0 and got[1, 1, 25] == 7.0


def test_a_flag_raised_between_rows_drops_the_result_and_numpy_computes_it():
    """A division by an element between rows that is zero: the kernels' result is dropped, and the values are NumPy's."""
    shape = (5, 6, 26)
    a, b = arrays(2, shape)
    b = numpy.abs(b) + 1.0
    b[:, :, 25] = 0.0  # Never a divisor of a row below, and a divisor of what lies between rows.

    def program(np: Any) -> Any:
        """Divide slices that leave the last element of every row out."""
        x, y = np.asarray(a), np.asarray(b)
        left = (slice(None), slice(None), slice(None, -1))
        return (x[left] + 1.0) / y[left] - x[left]

    expected = program(numpy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = program(lazy).gather()
    assert same(expected, got)


def test_other_element_types_are_not_fused():
    """Only float64 is fused: float32 arrays and integers are computed by NumPy's calls, with NumPy's types."""
    a, b = arrays(2, (500,))
    single, whole = a.astype(numpy.float32), numpy.arange(500)
    cases = [
        (single + 0.5 * single, lazy.asarray(single) + 0.5 * lazy.asarray(single)),
        (a + 2 * whole, lazy.asarray(a) + 2 * lazy.asarray(whole)),
        (a + numpy.float32(0.5) * b, lazy.asarray(a) + numpy.float32(0.5) * lazy.asarray(b)),
        (whole + 3 * whole, lazy.asarray(whole) + 3 * lazy.asarray(whole)),
    ]
    for expected, got in cases:
        assert same(expected, got.gather())
    assert calls() == 0


# Whole numbers and booleans as NumPy has them, of every integer type: small ones, the ends of each type, and
# values a float64 does not hold exactly, which NumPy's conversion rounds (2**53 + 1 down, 2**64 - 1 up to 2**64).
WHOLE_NUMBERS = (
    numpy.int64(3), numpy.int64(-3), numpy.int64(0), numpy.int64(2 ** 53 + 1), numpy.int64(-(2 ** 53) - 1),
    numpy.int64(2 ** 63 - 1), numpy.int64(-(2 ** 63)), numpy.int64(2 ** 62 + 2 ** 9 + 1),
    numpy.uint64(2 ** 64 - 1), numpy.uint64(2 ** 64 - 1025), numpy.uint64(2 ** 63 + 1025), numpy.uint64(2 ** 53 + 3),
    numpy.int8(-128), numpy.uint8(255), numpy.int16(-7), numpy.uint16(65535), numpy.int32(-(2 ** 31)),
    numpy.uint32(2 ** 32 - 1), numpy.longlong(2 ** 53 + 1), numpy.ulonglong(2 ** 64 - 1), numpy.intp(5),
    numpy.bool_(True), numpy.bool_(False),
)
# How many of the twenty-seven expressions of `with_a_number` have a node computed inside a kernel's program, at
# least, on either architecture. The others are those where the operation that reads the number is NumPy's call
# for a float as well: `np.maximum(a, k)` has the number second, and x86-64 has no kernel of one operation for
# that order; an `np.where` on `<=` is no shape of the library.
MOSTLY_INSIDE = 20
# The length from which that is asked. On a few elements a program of two kernels is given up, for a float as for
# a whole number.
LONG_ENOUGH = 64
# Those of them a float64 does not hold: the conversion gives another number.
INEXACT = tuple(value for value in WHOLE_NUMBERS if int(float(value)) != int(value))


def with_a_number(np: Any, a: Any, b: Any, c: Any, k: Any) -> dict[str, Any]:
    """Write expressions that read one number, in every place an operation of a kernel takes one.

    Args:
        np: The `numpy` module or the lazy stand-in.
        a, b, c: Three float64 arrays of one shape, `ndarray`s or lazy.
        k: The number: a NumPy scalar, an array without dimensions, or a Python number.

    Returns:
        A name -> the expression's result. Each operator the kernels have
        is there with the number on either side, with `np.maximum` and
        `np.minimum`, a comparison against it, and it as either value of
        an `np.where`.
    """
    return {
        "a * k + b": a * k + b,
        "k * a + b": k * a + b,
        "a + k * b": a + k * b,
        "a - k * b": a - k * b,
        "a - b * k": a - b * k,
        "k * (a + b)": k * (a + b),
        "a * (k + b)": a * (k + b),
        "(a + k) * b": (a + k) * b,
        "(a - k) + b": (a - k) + b,
        "(k - a) + b": (k - a) + b,
        "a / k + b": a / k + b,
        "k / a + b": k / a + b,
        "(a * b) / k": (a * b) / k,
        "k / (a * b)": k / (a * b),
        "maximum(k, a) + b": np.maximum(k, a) + b,
        "maximum(a, k) + b": np.maximum(a, k) + b,
        "minimum(k, a) + b": np.minimum(k, a) + b,
        "minimum(a, k) + b": np.minimum(a, k) + b,
        "maximum(k, maximum(a, minimum(k, b)))": np.maximum(k, np.maximum(a, np.minimum(k, b))),
        "where(a * (b - c) > k, k, a)": np.where(a * (b - c) > k, k, a),
        "where(a * (b - c) > 0, a, k)": np.where(a * (b - c) > 0, a, k),
        "where(a * b <= k, c, a)": np.where(a * b <= k, c, a),
        "sqrt(abs(k * a)) - b": np.sqrt(np.abs(k * a)) - b,
        "-(a * k) + b": -(a * k) + b,
        "k * a + k * b + k * c": k * a + k * b + k * c,
        "sum(a * k + b)": np.sum(a * k + b),
        "mean(a - k * b)": np.mean(a - k * b),
    }


def fused_with(a: numpy.ndarray, b: numpy.ndarray, c: numpy.ndarray, k: Any) -> tuple[dict[str, Any], dict[str, tuple]]:
    """Compute the expressions of `with_a_number` lazily, one at a time, and say what fusion did for each.

    Args:
        a, b, c: Three float64 arrays of one shape.
        k: The number.

    Returns:
        (name -> what `gather()` gave; name -> the kernel calls it took, by
        kernel, with how many nodes were computed inside a program and how
        many programs were given up).
    """
    values, did = {}, {}
    for name in with_a_number(numpy, a, b, c, 1.0):
        fuse.STATS.reset()
        values[name] = gathered(lambda: with_a_number(lazy, lazy.asarray(a), lazy.asarray(b), lazy.asarray(c), k)[name])
        did[name] = (dict(fuse.STATS.calls), fuse.STATS.fused_nodes, fuse.STATS.declined)
    return values, did


@pytest.mark.parametrize("length", (0, 1, 7, LONG_ENOUGH, 1000))
@pytest.mark.parametrize("k", WHOLE_NUMBERS, ids=lambda k: f"{type(k).__name__}({k})")
def test_a_whole_number_of_numpys_is_a_number_as_its_float_is(k: Any, length: int):
    """`np.int64(k)` beside float64 arrays gives NumPy's bits, by the kernels `float(k)` is computed by.

    NumPy converts the scalar to float64 once and runs the float64 loop, so
    the expression is the one written with `float(k)`. NumPy itself is held
    to that here, bit for bit; the layer's run with the whole number to its
    run with the float, bit for bit and kernel for kernel; and the layer to
    NumPy, bit for bit on ordinary values. On the special values, which
    hold `np.nan` beside what makes the processor's own NaN, it is held to
    NumPy but for which NaN of two a result is, as a float is.
    """
    for special in (False, True):
        a, b, c = arrays(3, (length,), special)
        with numpy.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # The mean of no elements.
            expected = with_a_number(numpy, a, b, c, k)
            as_float = with_a_number(numpy, a, b, c, float(k))
            got, did = fused_with(a, b, c, k)
            got_as_float, did_as_float = fused_with(a, b, c, float(k))
        for name, wanted in expected.items():
            wanted, value = numpy.asarray(wanted), numpy.asarray(got[name])
            assert same(wanted, numpy.asarray(as_float[name])), name
            assert same(numpy.asarray(got_as_float[name]), value), name
            assert same_but_for_which_nan(wanted, value) if special else same(wanted, value), name
            assert did[name] == did_as_float[name], name
        if length:  # An array of no elements is NumPy's, whatever is written on it.
            assert did["a * k + b"] == ({"fuse_add_scaled_f64": 1}, 1, 0)
        if length >= LONG_ENOUGH:
            assert sum(1 for _, inside, _ in did.values() if inside) >= MOSTLY_INSIDE


@pytest.mark.parametrize("k", INEXACT, ids=lambda k: f"{type(k).__name__}({k})")
def test_a_whole_number_a_float64_does_not_hold_is_rounded_as_numpy_rounds_it(k: Any):
    """`a * np.int64(2**53 + 1) + b` is one kernel call with NumPy's bits: the conversion that rounds is NumPy's own."""
    a, b = arrays(2, (1000,))
    # Values near the number, where another rounding of it would show: a quotient of 1.0 and a difference of 0.0.
    near = float(k)
    around = [near, numpy.nextafter(near, numpy.inf), numpy.nextafter(near, -numpy.inf)]
    a[:6] = [*around, 1.0, -1.0, 0.5]
    # And products at the number and either side of it, for the comparisons against it: `2.0 * (b - a)`, and
    # `b * (b - a)`, which is the shape of a kernel with a mask.
    a[6:9], b[6:9] = 0.0, [value / 2.0 for value in around]
    a[9:12], b[9:12] = [-value for value in around], 1.0
    assert int(near) != int(k)
    held, other = lazy.asarray(a), lazy.asarray(b)
    with numpy.errstate(all="ignore"):
        cases = [
            (a * k + b, lambda: held * k + other, {"fuse_add_scaled_f64": 1}),
            (b - a / k, lambda: other - held / k, {"fuse_div_v_s_f64": 1, "fuse_sub_v_v_f64": 1}),
            (k / a + b, lambda: k / held + other, {"fuse_div_s_v_f64": 1, "fuse_add_v_v_f64": 1}),
            ((a - k) * b, lambda: (held - k) * other, {"fuse_sub_v_s_f64": 1, "fuse_mul_v_v_f64": 1}),
            ((k - a) + b, lambda: (k - held) + other, {"fuse_sub_s_v_f64": 1, "fuse_add_v_v_f64": 1}),
            (numpy.where(2.0 * (b - a) > k, k, a), lambda: lazy.where(2.0 * (other - held) > k, k, held), None),
            (numpy.where(b * (b - a) > k, k, a), lambda: lazy.where(other * (other - held) > k, k, held),
             {"fuse_flux_limit_f64": 1}),
        ]
        # Each comparison is false at the number and true just above it.
        assert len(numpy.unique(2.0 * (b - a)[6:9] > k)) == 2 and len(numpy.unique((b * (b - a))[9:12] > k)) == 2
        for expected, write, kernels in cases:
            fuse.STATS.reset()
            assert same(expected, gathered(write)), kernels
            assert kernels is None or fuse.STATS.calls == kernels
            assert fuse.STATS.fused_nodes >= 1 and fuse.STATS.declined == 0


def test_one_whole_number_in_an_array_without_dimensions_is_a_number():
    """`np.asarray(7)`, an element of a lazy table and a count still pending are numbers; each line is one kernel call."""
    a, b = arrays(2, (1000,), special=True)
    table = numpy.arange(12).reshape(3, 4) - 5
    counts = numpy.arange(1000)
    held, other = lazy.asarray(a), lazy.asarray(b)
    with numpy.errstate(all="ignore"):
        cases = [
            (numpy.asarray(7) * a + b, lambda: numpy.asarray(7) * held + other),
            (a * numpy.asarray(-7, dtype=numpy.int8) + b, lambda: held * numpy.asarray(-7, dtype=numpy.int8) + other),
            (a * numpy.asarray(7, dtype=">i4") + b, lambda: held * numpy.asarray(7, dtype=">i4") + other),
            (a * numpy.asarray(2 ** 64 - 1025, dtype=numpy.uint64) + b,
             lambda: held * numpy.asarray(2 ** 64 - 1025, dtype=numpy.uint64) + other),
            (a * numpy.asarray(True) + b, lambda: held * numpy.asarray(True) + other),
            (table[1, 2] * a + b, lambda: lazy.asarray(table)[1, 2] * held + other),
            (a * table[0, 0] + b, lambda: held * lazy.asarray(table)[0, 0] + other),
            (counts.sum() * a + b, lambda: lazy.asarray(counts).sum() * held + other),
            (a * (counts > 5).any() + b, lambda: held * (lazy.asarray(counts) > 5).any() + other),
            (numpy.multiply(table[2, 3], a) + b, lambda: lazy.multiply(lazy.asarray(table)[2, 3], held) + other),
        ]
        for place, (expected, write) in enumerate(cases):
            fuse.STATS.reset()
            assert same(expected, gathered(write)), place
            assert fuse.STATS.calls == {"fuse_add_scaled_f64": 1} and fuse.STATS.fused_nodes == 1, place


def test_a_square_by_a_whole_number_of_numpys_is_one_multiplication():
    """`x ** np.int64(2)` is `np.power`'s loop handed 2.0, which multiplies: the kernels' square, with `x * x`'s bits."""
    a, b = arrays(2, (1000,), special=True)
    held, other = lazy.asarray(a), lazy.asarray(b)
    with numpy.errstate(all="ignore"):
        for two in (numpy.int64(2), numpy.uint8(2), numpy.int32(2), numpy.longlong(2)):
            fuse.STATS.reset()
            assert same(a * a, a ** two) and same(a * a, numpy.power(a, two))
            assert same(a ** two + b, gathered(lambda: held ** two + other))
            assert same(numpy.power(a, two) - b, gathered(lambda: lazy.power(held, two) - other))
            assert same(numpy.sum(a ** two), gathered(lambda: numpy.sum(held ** two)))
            with_two = dict(fuse.STATS.calls)
            fuse.STATS.reset()
            assert same(a ** 2 + b, gathered(lambda: held ** 2 + other))
            assert same(numpy.power(a, 2) - b, gathered(lambda: lazy.power(held, 2) - other))
            assert same(numpy.sum(a ** 2), gathered(lambda: numpy.sum(held ** 2)))
            assert with_two == fuse.STATS.calls and with_two["fuse_sum_squares_f64"] == 1 and with_two["fuse_sq_v_f64"] == 2
        # Another exponent is the mathematics library's, a boolean is the exponent 1, and a number raised to an
        # array is no square: NumPy's calls, and nothing inside a kernel but the addition's own.
        fuse.STATS.reset()
        for expected, write in [(a ** numpy.int64(3) + b, lambda: held ** numpy.int64(3) + other),
                                (a ** numpy.bool_(True) + b, lambda: held ** numpy.bool_(True) + other),
                                (numpy.int64(2) ** a + b, lambda: numpy.int64(2) ** held + other),
                                (a ** numpy.asarray(2) + b, lambda: held ** numpy.asarray(2) + other)]:
            assert same(expected, gathered(write))
        assert fuse.STATS.fused_nodes == 0 and "fuse_sq_v_f64" not in fuse.STATS.calls


def test_a_whole_number_is_no_number_where_numpys_result_is_not_the_float64_loops():
    """Whole numbers with a dimension, beside other element types, of other kinds and too large a Python integer
    are left to NumPy, with NumPy's types and bits."""
    a, b = arrays(2, (500,))
    single, whole = a.astype(numpy.float32), numpy.arange(500)
    held, other = lazy.asarray(a), lazy.asarray(b)
    cases = [
        # An array of whole numbers with a dimension, of one element or of many: NumPy converts it element by element.
        (a * numpy.asarray([3]) + b, lambda: held * numpy.asarray([3]) + other),
        (a * numpy.asarray([[3]]) + b, lambda: held * lazy.asarray(numpy.asarray([[3]])) + other),
        (a * whole + b, lambda: held * lazy.asarray(whole) + other),
        # float32 with an int64 gives float64, but the array is not a kernel's; integers stay integers.
        (single * numpy.int64(3) + single, lambda: lazy.asarray(single) * numpy.int64(3) + lazy.asarray(single)),
        (whole * numpy.int64(3) + whole, lambda: lazy.asarray(whole) * numpy.int64(3) + lazy.asarray(whole)),
        (whole / numpy.int64(3) + whole, lambda: lazy.asarray(whole) / numpy.int64(3) + lazy.asarray(whole)),
        # Scalars that are no whole number, and arrays without dimensions of their types: a float32, a complex
        # number, a span of time, which is an `np.signedinteger` by its class.
        (a * numpy.float32(0.1) + b, lambda: held * numpy.float32(0.1) + other),
        (a * numpy.asarray(0.1, dtype=numpy.float32) + b, lambda: held * numpy.asarray(0.1, dtype=numpy.float32) + other),
        (a * single[3] + b, lambda: held * lazy.asarray(single)[3] + other),
        (a * numpy.asarray(2j) + b, lambda: held * numpy.asarray(2j) + other),
        (a * numpy.asarray(numpy.timedelta64(3, "s")) * 2.0, lambda: held * numpy.asarray(numpy.timedelta64(3, "s")) * 2.0),
        (a * numpy.complex128(2j) + b, lambda: held * numpy.complex128(2j) + other),
        (a * numpy.timedelta64(3, "s") * 2.0, lambda: held * numpy.timedelta64(3, "s") * 2.0),
        # A Python integer beyond what a float64 holds exactly is not taken, as before.
        (a * (2 ** 53 + 1) + b, lambda: held * (2 ** 53 + 1) + other),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for place, (expected, write) in enumerate(cases):
            assert same(expected, gathered(write)), place
    assert calls() == 0 and fuse.STATS.fused_nodes == 0


def test_a_write_with_a_whole_number_of_numpys_is_fused_as_one_with_a_float():
    """`w *= np.int64(3)`, `u[1:-1] += np.int64(2) * u[:-2]` and `np.multiply(x, np.int64(5), out=x)` are kernels' work."""
    start, gradient = arrays(2, (1000,))
    w, expected = lazy.array(start), start.copy()
    for step in range(3):
        w -= numpy.int64(2 ** 53 + 1) * lazy.asarray(gradient)
        w *= numpy.uint8(3)
        w += numpy.bool_(True)
        w /= numpy.int16(-7)
        expected -= numpy.int64(2 ** 53 + 1) * gradient
        expected *= numpy.uint8(3)
        expected += numpy.bool_(True)
        expected /= numpy.int16(-7)
    assert same(expected, w.gather())
    assert fuse.STATS.whole_writes == 12 and fuse.STATS.fused_nodes == 14
    grid, other = arrays(2, (30, 40))
    u, wanted = lazy.array(grid), grid.copy()
    fuse.STATS.reset()
    u[1:-1] += numpy.int64(2) * u[:-2]
    wanted[1:-1] += numpy.int64(2) * wanted[:-2]
    u[1:-1, 1:-1] = lazy.asarray(other)[1:-1, 1:-1] + numpy.uint64(2 ** 64 - 1) * u[2:, 1:-1]
    wanted[1:-1, 1:-1] = other[1:-1, 1:-1] + numpy.uint64(2 ** 64 - 1) * wanted[2:, 1:-1]
    assert same(wanted, u.gather())
    assert fuse.STATS.calls == {"fuse_add_scaled_f64": 2}
    x = lazy.zeros(200)
    fuse.STATS.reset()
    numpy.add(lazy.asarray(start[:200]), lazy.asarray(gradient[:200]), out=x)
    numpy.multiply(x, numpy.int64(5), out=x)
    assert same((start[:200] + gradient[:200]) * numpy.int64(5), x.gather())
    assert fuse.STATS.calls == {"fuse_scaled_sum_f64": 1}


def test_the_limit_tool_counts_an_operation_on_a_whole_number_of_numpys_as_arithmetic():
    """`limit` puts `a * np.int64(3)` with the operations a kernel makes, and a whole-number array under other types."""
    (data,) = arrays(1, (100,))
    a, whole = lazy.asarray(data), lazy.asarray(numpy.arange(100))
    cases = [
        (a * numpy.int64(3), limit.ARITHMETIC), (numpy.uint8(2) / a, limit.ARITHMETIC),
        (a * numpy.bool_(True), limit.ARITHMETIC), (a - numpy.asarray(7), limit.ARITHMETIC),
        (a + whole.sum(), limit.ARITHMETIC), (a ** numpy.int64(2), limit.ARITHMETIC),
        (a ** numpy.int64(3), limit.LIBRARY), (a > numpy.int64(3), limit.MASKS),
        (a * whole, limit.OTHER_TYPES), (whole * numpy.int64(3), limit.OTHER_TYPES),
        (a * numpy.float32(0.5), limit.OTHER_TYPES), (a * numpy.asarray([3]), limit.OTHER_TYPES),
    ]
    for place, (written, wanted) in enumerate(cases):
        assert limit.classify(written._node) == wanted, place


def test_both_readers_take_a_whole_number_of_numpys_alike(lazy_core: str, monkeypatch: pytest.MonkeyPatch):
    """The core's reader and `_Finder` find the same expressions on every line that reads a whole number."""
    if lazy_core != "c":
        pytest.skip("one reader only without the C core")
    compared = _read_both_ways(monkeypatch)
    a, b, c = arrays(3, (64,), special=True)
    table = numpy.arange(12).reshape(3, 4) - 5
    held, other = lazy.asarray(a), lazy.asarray(b)
    with numpy.errstate(all="ignore"):
        for k in (numpy.int64(2 ** 53 + 1), numpy.uint64(2 ** 64 - 1), numpy.bool_(True), numpy.asarray(7),
                  numpy.float32(0.5), numpy.timedelta64(3, "s"), 2 ** 53 + 1, numpy.asarray([3])):
            for name in with_a_number(numpy, a, b, c, 1.0):
                try:
                    expected = numpy.asarray(with_a_number(numpy, a, b, c, k)[name])
                except TypeError:
                    continue  # A span of time is no operand of most of these, for NumPy and so for the layer.
                got = gathered(lambda: with_a_number(lazy, held, other, lazy.asarray(c), k)[name])
                assert same(expected, numpy.asarray(got)), (k, name)
        for write in (lambda: lazy.asarray(table)[1, 2] * held + other, lambda: held ** numpy.int64(2) + other,
                      lambda: held ** numpy.int64(3) + other, lambda: lazy.asarray(numpy.arange(64)).sum() * held + other):
            gathered(write)
    assert sum(compared) > 100


def test_an_array_read_twice_is_one_kernel_call():
    """`a + s * a` is served by the kernel of `a + s * b`, given the same array twice."""
    a, b = arrays(2, (300,))
    held, other = lazy.asarray(a), lazy.asarray(b)
    assert same(a + 0.25 * a, gathered(lambda: lazy.asarray(a) + 0.25 * lazy.asarray(a)))
    assert same(a - a * b, gathered(lambda: held - held * other))
    assert fuse.STATS.calls == {"fuse_add_scaled_f64": 1, "fuse_sub_product_f64": 1}


def test_operands_in_either_order_are_one_shape():
    """The operands of a sum or a product may be written either way round: it is the same kernel."""
    a, b, c = arrays(3, (300,))
    la, lb, lc = (lazy.asarray(operand) for operand in (a, b, c))
    for expected, got in [(a + b * 0.5, la + lb * 0.5), (0.5 * b + a, 0.5 * lb + la), (b * 0.5 + a, lb * 0.5 + la),
                          ((b - c) * a, (lb - lc) * la), (c * b + a, lc * lb + la)]:
        assert same(expected, got.gather())
    assert fuse.STATS.calls == {"fuse_add_scaled_f64": 3, "fuse_mul_difference_f64": 1, "fuse_add_product_f64": 1}


def test_a_long_expression_is_covered_by_several_kernels():
    """An expression larger than any shape of the library is computed by several kernel calls and NumPy's own."""
    a, b, c, d = arrays(4, (400,))
    la, lb, lc, ld = (lazy.asarray(operand) for operand in (a, b, c, d))
    expected = (a + b * c) * (d - a) + 0.5 * (b + numpy.sqrt(numpy.abs(c)) * d) / (a * a + 1.0)
    got = (la + lb * lc) * (ld - la) + 0.5 * (lb + numpy.sqrt(numpy.abs(lc)) * ld) / (la * la + 1.0)
    assert same(expected, got.gather())
    assert calls() >= 2 and fuse.STATS.fused_nodes >= 2


def test_a_long_expression_is_one_program_with_no_array_between_its_kernels():
    """Every operation of an arithmetic expression is a step of one program: all its nodes but the last are inside."""
    a, b, c, d = arrays(4, (5000,), special=True)
    with numpy.errstate(all="ignore"):
        expected = ((a + b * c) * (d - a) + 0.5 * (b + numpy.sqrt(numpy.abs(c)) * d) / (a * a + 1.0)) - numpy.maximum(a, 0.25) * b
        got = gathered(lambda: ((lazy.asarray(a) + lazy.asarray(b) * lazy.asarray(c)) * (lazy.asarray(d) - lazy.asarray(a))
                                + 0.5 * (lazy.asarray(b) + numpy.sqrt(numpy.abs(lazy.asarray(c))) * lazy.asarray(d))
                                / (lazy.asarray(a) * lazy.asarray(a) + 1.0)) - numpy.maximum(lazy.asarray(a), 0.25) * lazy.asarray(b))
    assert same(expected, got)
    # Sixteen operations, one group: fifteen nodes have no array of their own. On x86-64 `np.maximum(a, 0.25)` is
    # not a kernel's: its operands are taken only in the order written there, and the library has `max(S0,V0)`.
    inside = 15 if "max" in fuse.COMMUTATIVE else 14
    assert fuse.STATS.fused_nodes == inside and fuse.STATS.declined == 0 and calls() < 16


@pytest.mark.parametrize("length", (1, 7, 8, 9, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 9000))
def test_a_program_is_numpys_at_every_length_around_its_blocks(length: int):
    """A program's blocks, its last one and the elements left of it give NumPy's bits, also written in place."""
    a, b, c = arrays(3, (length,), special=True)
    with numpy.errstate(all="ignore"):
        expected = (a - 0.5 * b) / (c * c + 2.0) - a * (b + c)
        got = gathered(lambda: (lazy.asarray(a) - 0.5 * lazy.asarray(b)) / (lazy.asarray(c) * lazy.asarray(c) + 2.0)
                       - lazy.asarray(a) * (lazy.asarray(b) + lazy.asarray(c)))
        assert same(expected, got)
        # The same with its result written where the first operand is.
        w, there = lazy.array(a), a.copy()
        w += (lazy.asarray(b) - w) * lazy.asarray(c)
        w *= 0.75
        there += (b - there) * c
        there *= 0.75
        assert same(there, w.gather())
    # Runs of fewer than 32 elements are left to NumPy: a program there is mostly the cost of its calls.
    assert fuse.STATS.flagged == 0 and (length < 32 or fuse.STATS.declined == 0)


def test_an_array_that_repeats_along_outer_dimensions_does_not_end_a_run():
    """`grid * levels[None, None, :]` is walked in long runs, the levels read from a buffer that repeats them."""
    grid, other = arrays(2, (9, 11, 26))
    levels, rows = arrays(1, (26,))[0], arrays(1, (11, 26))[0]
    cases = [
        ((grid + other * levels[None, None, :]) * 0.5 - grid, lambda: (lazy.asarray(grid) + lazy.asarray(other) * lazy.asarray(levels)[None, None, :]) * 0.5 - lazy.asarray(grid)),
        ((grid * levels - rows) / (other * other + 1.0), lambda: (lazy.asarray(grid) * lazy.asarray(levels) - lazy.asarray(rows)) / (lazy.asarray(other) * lazy.asarray(other) + 1.0)),
        (levels * (rows + levels) - grid, lambda: lazy.asarray(levels) * (lazy.asarray(rows) + lazy.asarray(levels)) - lazy.asarray(grid)),
    ]
    for expected, write in cases:
        assert same(expected, gathered(write))
    # The sum of the third, `rows + levels`, has another shape than the result and is NumPy's own call.
    assert fuse.STATS.declined == 0 and fuse.STATS.fused_nodes == 3 + 4 + 1


def test_columns_of_a_grid_are_gathered_into_blocks():
    """An array that steps over elements is copied into a block for the kernels, and the result is NumPy's."""
    table, other = arrays(2, (5000, 4))
    out, expected = lazy.zeros((5000, 3)), numpy.zeros((5000, 3))
    out[:, 1] = (lazy.asarray(table)[:, 0] - 0.5 * lazy.asarray(table)[:, 3]) * lazy.asarray(other)[:, 2] + 1.5
    expected[:, 1] = (table[:, 0] - 0.5 * table[:, 3]) * other[:, 2] + 1.5
    assert same(expected, out.gather())
    column = gathered(lambda: ((lazy.asarray(table)[:, 1] * lazy.asarray(table)[:, 2] - lazy.asarray(other)[:, 0]) / 3.0 + 1.0) * 0.5)
    assert same(((table[:, 1] * table[:, 2] - other[:, 0]) / 3.0 + 1.0) * 0.5, column)
    assert fuse.STATS.declined == 0 and calls() >= 2
    # As many columns to copy as operations: the copies cost what the program saves, and NumPy's calls make it.
    both = gathered(lambda: (lazy.asarray(table)[:, 1] - lazy.asarray(table)[:, 2]) * lazy.asarray(other)[:, 0])
    assert same((table[:, 1] - table[:, 2]) * other[:, 0], both) and fuse.STATS.declined == 1


@pytest.mark.parametrize("shape", ((30, 14, 26, 3), (9, 40, 12, 2), (6, 5, 33, 4)))
def test_arrays_that_step_through_several_dimensions_are_gathered_once(shape: tuple[int, ...]):
    """A grid with a time level as its last axis: every third element, the same way through two or three dimensions,
    read twice by one expression, with a result stored into part of another grid."""
    a, b = arrays(2, shape)
    grid = shape[:-1]
    c, d = arrays(2, grid)

    def program(np: Any) -> Any:
        """A flux of thirteen operations on slices of two such grids, and the same into a part."""
        vel, var = np.asarray(a)[..., 1], np.asarray(b)[..., 0]
        cr, rj = np.asarray(c), np.asarray(d)
        here, there = (slice(1, -2), slice(2, -2)), (slice(2, -1), slice(2, -2))
        flux = 1.5 * vel[here] * (var[there] + var[here]) * 0.5 - np.abs(1.5 * vel[here]) * ((1.0 - cr[here]) + rj[there] * cr[here]) * rj[here] * 0.5
        into = np.zeros(grid)
        into[here] = vel[here] * (var[there] + var[here]) * 0.5 - np.abs(vel[here]) * ((1.0 - cr[here]) + rj[there] * cr[here]) * rj[here]
        return flux, into

    expected = program(numpy)
    got = program(lazy)
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours.gather())
    rows = shape[1] - 4 if shape[2] < 32 else 1
    assert (fuse.STATS.declined == 0) == (rows * shape[2] >= 32)


@pytest.mark.parametrize("width", (2, 7, 16, 25, 31, 32, 40))
@pytest.mark.parametrize("special", (False, True))
def test_a_product_of_three_is_stored_into_short_rows_of_a_result_that_steps(width: int, special: bool):
    """`Ai[..., ip, kr] = taper * sx * mask[...]`: the result is every fourth element of an array with two more axes
    after its grid, in rows that end short of the next. Below 32 elements to a row the module makes the product by a
    loop of its own and stores each row at the result's step; the elements between are left as they were."""
    grid = (12, 9, width + 1)
    mask, deep = arrays(2, grid, special)
    taper, sx = arrays(2, (8, 5, width), special)

    def program(np: Any) -> Any:
        """Two such stores, and one of another expression, which no loop of the module's is written for."""
        Ai, Bi = np.ones(grid + (2, 2)), np.ones(grid + (2, 2))
        Ai[2:-2, 2:-2, :-1, 1, 0] = np.asarray(taper) * np.asarray(sx) * np.asarray(mask)[2:-2, 2:-2, :-1]
        Ai[2:-2, 2:-2, 1:, 0, 1] = np.asarray(deep)[1:-3, 3:-1, 1:] * np.asarray(sx) * np.asarray(mask)[2:-2, 2:-2, 1:]
        Bi[2:-2, 2:-2, :-1, 1, 1] = (np.asarray(taper) - np.asarray(deep)[1:-3, 3:-1, 1:]) / np.asarray(sx)
        return Ai, Bi

    with numpy.errstate(all="ignore"):
        expected = program(numpy)
        got = program(lazy)
        for theirs, ours in zip(expected, got):
            assert same(theirs, ours.gather())
    if not special and width < 32:
        assert fuse.STATS.calls.get("fuse_product3_f64") == 2


def test_a_recurrence_over_rows_is_written_into_the_array_a_row_at_a_time():
    """`y[i, :] = a * out[i, :] + b * y[i - 1, :] + c * y[i - 2, :]`: the program reads rows of the array it writes
    another row of. The rows read share no byte with the row written, so the write is made where the array is; a
    whole array copied for each row took 11.6 s on a filter of 1,500 rows where NumPy takes 37 ms. The two
    statements after the loop read bytes of the part they write, and those are made by way of a copy, as before."""
    image = arrays(1, (40, 64))[0]

    def program(np: Any) -> Any:
        """A large array made first, then each row of another from the two rows before it, and one from itself."""
        y = np.zeros_like(np.asarray(image)) + 1.0
        out = 0.5 * (y + np.asarray(image))
        y[0, :] = 0.25 * out[0, :]
        y[1, :] = 0.25 * out[1, :] + 0.5 * out[0, :] + 0.125 * y[0, :]
        for i in range(2, 40):
            y[i, :] = 0.25 * out[i, :] + 0.5 * out[i - 1, :] + 0.125 * y[i - 1, :] + 0.0625 * y[i - 2, :]
        y[5, :] = 2.0 * y[5, :] + y[4, :]
        y[7, 1:] = y[7, :-1] * 3.0 + out[7, 1:]
        return y

    expected = program(numpy)
    evaluate.WRITES.update(in_place=0, copied=0)
    assert same(expected, program(lazy).gather())
    assert evaluate.WRITES["copied"] == 2 and evaluate.WRITES["in_place"] >= 40


def test_the_driver_refuses_a_program_that_is_not_one():
    """`_fused.program` checks every step against its kernel and the places it names before it computes anything."""
    module = fuse._fused
    add = fuse.LIBRARY["add(V0,V1)"].index
    a, b, out = numpy.ones(64), numpy.ones(64), numpy.empty(64)
    assert module.program(((add, -1, 0, 1),), 2, (a, b), out, fuse.BLOCK_ELEMENTS) == 0 and (out == 2.0).all()
    for steps in (((add, 2, 0, 1),),                      # The last step has to write the result.
                  ((add, -1, 0, 1), (add, -1, 0, 1)),     # Only the last may.
                  ((add, -1, 0, 5),),                     # Reads a buffer nothing wrote.
                  ((add, 2, 0, 2), (add, -1, 0, 2)),      # Reads the buffer it writes.
                  ((add, -1, 0),),                        # Not the kernel's number of operands.
                  ((10 ** 6, -1, 0, 1),)):                # No such kernel.
        with pytest.raises(ValueError):
            module.program(steps, 2, (a, b), out, fuse.BLOCK_ELEMENTS)
    with pytest.raises(ValueError):
        module.program(((add, -1, 0, 1),), 2, (a, b), out, 100)   # A block is a multiple of eight.
    # A result that overlaps an operand, but is not the operand, is declined before anything is written.
    whole = numpy.arange(72.0)
    assert module.program(((add, -1, 0, 1),), 2, (whole[:64], b), whole[8:], fuse.BLOCK_ELEMENTS) == module.DECLINED
    assert (whole == numpy.arange(72.0)).all()
    # The operand itself is not: the result is written where it is.
    assert module.program(((add, -1, 0, 1),), 2, (whole[:67], numpy.ones(67)), whole[:67], fuse.BLOCK_ELEMENTS) == 0
    assert (whole[:67] == numpy.arange(67.0) + 1.0).all() and (whole[67:] == numpy.arange(67.0, 72.0)).all()


def test_an_expression_seen_before_is_planned_from_what_was_remembered(monkeypatch: pytest.MonkeyPatch):
    """The second expression of a shape is not covered with shapes again, and is computed as the first was."""
    a, b, c = arrays(3, (300,))
    fuse._RECIPES.clear()
    covered = []
    monkeypatch.setattr(fuse, "tile", lambda term, library, real=fuse.tile: (covered.append(1), real(term, library))[1])
    for turn in range(3):
        got = gathered(lambda: (lazy.asarray(a) - 0.5 * lazy.asarray(b)) * (lazy.asarray(c) + lazy.asarray(a)) / 3.0)
        assert same((a - 0.5 * b) * (c + a) / 3.0, got)
    assert len(covered) == 1 and len(fuse._RECIPES) == 1 and fuse.STATS.tiled == 3


@pytest.mark.parametrize("shape", make_kernel.SHAPES)
def test_a_kernel_written_from_its_shape_is_called_for_it_and_gives_numpys_bits(shape: str):
    """The oracle of a written kernel, run on lazy arrays, is computed by that kernel alone; one operation is NumPy's."""
    name = make_kernel.name_of(shape)
    spec = load_spec(make_kernel.KERNELS / name)
    count = sum(1 for arg in spec.args if type(arg) is In)
    numbers = [arg.value for arg in spec.args if type(arg) is Scalar]
    for length in (5, 64, 1001):
        operands = arrays(count, (length,), special=True)
        with numpy.errstate(all="ignore"):
            expected = spec.oracle(*operands, *numbers)
            got = gathered(lambda: spec.oracle(*(lazy.asarray(operand) for operand in operands), *numbers))
        assert same(expected, got), length
    # One operation by itself is no expression: it is NumPy's call. Anything larger is this kernel's, once a length.
    assert fuse.STATS.calls == ({} if shape.count("(") == 1 else {name: 3})


def test_the_kernels_in_the_repository_are_what_the_tool_writes(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    """Every written kernel's spec and reference are the tool's, and so is its assembly until its log has a second attempt.

    An x86-64 loop that was timed against the compiled C and not kept has a log and no `kernel.s`.
    """
    kept = make_kernel.KERNELS
    monkeypatch.setattr(make_kernel, "KERNELS", tmp_path)
    for shape in make_kernel.SHAPES:
        written = make_kernel.write(shape)
        theirs = kept / written.name
        for path in sorted(written.rglob("*")):
            if not path.is_file() or path.name == make_kernel.LOG_NAME:
                continue
            ours = kept / path.relative_to(tmp_path)
            if path.name == "kernel.s":
                log = (theirs / path.parent.name / make_kernel.LOG_NAME).read_text()
                if not ours.exists():
                    assert "- **Kept:** no" in log, path.parent.name
                    continue
                if "\n## 2." in log:
                    continue
            assert path.read_text() == ours.read_text(), path.name
    assert make_kernel.name_of("sub(V0,mul(S0,V1))") == "fuse_sub_v_mul_s_v_f64"
    with pytest.raises(make_kernel.CannotWrite):
        make_kernel.write("where(gt(V0,S0),V1,V2)")


def test_an_intermediate_the_program_holds_is_computed_once():
    """An intermediate with a name of its own is not computed inside a kernel: its value is wanted again."""
    a, b, c = arrays(3, (200,))
    product = lazy.asarray(b) * lazy.asarray(c)
    total = lazy.asarray(a) + product
    assert same(a + b * c, total.gather())
    assert calls() == 0 and not product._node.pending
    assert same(b * c, product.gather())


def test_an_intermediate_read_by_a_later_computation_is_computed_once():
    """An intermediate that a node outside the computation reads is not computed inside a kernel either."""
    a, b, c = arrays(3, (200,))
    la, lb, lc = (lazy.asarray(operand) for operand in (a, b, c))
    product = lb * lc
    first, second = la + product, product - la
    del product
    assert same(a + b * c, first.gather())
    assert calls() == 0
    assert same(b * c - a, second.gather())


def test_a_warning_is_numpys_and_names_the_line_that_wrote_the_operation():
    """A kernel that raises a floating-point exception NumPy reports is dropped, and NumPy's calls report it."""
    huge = numpy.full(64, 1e308)
    with warnings.catch_warnings(record=True) as expected:
        warnings.simplefilter("always")
        wanted = huge + 10.0 * huge
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pending = lazy.asarray(huge) + 10.0 * lazy.asarray(huge); line = pending._node.origin.lineno  # noqa: E702
        got = pending.gather()
    assert same(wanted, got)
    assert [str(warning.message) for warning in caught] == [str(warning.message) for warning in expected]
    assert [warning.lineno for warning in caught] == [line] and caught[0].filename == __file__
    assert fuse.STATS.flagged == 1 and calls() == 0


def test_an_exception_that_is_ignored_does_not_stop_a_kernel():
    """Under `np.errstate(over="ignore")` NumPy reports nothing, so the kernel's result is kept."""
    huge = numpy.full(64, 1e308)
    with numpy.errstate(over="ignore"):
        wanted = huge + 10.0 * huge
        pending = lazy.asarray(huge) + 10.0 * lazy.asarray(huge)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert same(wanted, pending.gather())
    assert fuse.STATS.flagged == 0 and calls() == 1


def test_an_exception_that_is_raised_is_raised_as_numpy_raises_it():
    """Under `np.errstate(over="raise")` the error is NumPy's, with the note naming the line, and stays the node's."""
    huge = lazy.asarray(numpy.full(64, 1e308))
    with numpy.errstate(over="raise"):
        pending = huge + 10.0 * huge
    for _ in range(2):
        with pytest.raises(FloatingPointError, match="overflow encountered in multiply") as caught:
            pending.gather()
        assert "written at" in caught.value.__notes__[0]


def test_a_comparison_of_a_nan_is_left_to_numpy_and_does_not_warn():
    """A fused comparison raises the invalid flag for a NaN, which NumPy's does not report: NumPy's calls answer."""
    a, b, c = arrays(3, (100,))
    a[5] = numpy.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wanted = numpy.where(a * (b - c) > 0, 0, a)
        got = numpy.where(lazy.asarray(a) * (lazy.asarray(b) - lazy.asarray(c)) > 0, 0, lazy.asarray(a)).gather()
    assert same(wanted, got) and fuse.STATS.flagged == 1
    with numpy.errstate(invalid="ignore"):
        pending = numpy.where(lazy.asarray(a) * (lazy.asarray(b) - lazy.asarray(c)) > 0, 0, lazy.asarray(a))
    silenced = pending.gather()
    assert same(wanted, silenced) and fuse.STATS.calls == {"fuse_flux_limit_f64": 1}


def test_an_in_place_operator_on_a_whole_array_is_one_call():
    """`w -= s * g` on an array the layer made is one kernel call, and a loop of them is NumPy's, step for step."""
    start, gradient = arrays(2, (1000,))
    w, expected = lazy.array(start), start.copy()
    alias = w
    for step in range(5):
        w -= 0.1 * lazy.asarray(gradient)
        w += lazy.asarray(gradient) * lazy.asarray(start)
        w *= 1.5
        expected -= 0.1 * gradient
        expected += gradient * start
        expected *= 1.5
    assert alias is w and same(expected, w.gather())
    # The fifteen writes and the ten products they read are one expression: each write is read once, by the next.
    # So all but the last node are computed inside one group, with no array between them.
    assert fuse.STATS.whole_writes == 15 and fuse.STATS.fused_nodes == 24


def test_a_write_into_a_whole_array_keeps_what_was_read_before_it():
    """An operation written before a write reads what the array held then, fused or not."""
    start, step = arrays(2, (300,))
    w = lazy.array(start)
    before = w + 0.5 * w
    w -= 0.1 * lazy.asarray(step)
    after = w * (1 + w)
    assert same((start - 0.1 * step) * (1 + (start - 0.1 * step)), after.gather())
    assert same(start + 0.5 * start, before.gather())


def test_an_expression_assigned_to_part_of_an_array_is_computed_into_it():
    """`u[1:-1, 1:-1] = e` computes `e` straight into the part of the new array: by a kernel, or by NumPy's `out=`."""
    grid, other = arrays(2, (30, 40))
    u, expected = lazy.array(grid), grid.copy()
    held = lazy.asarray(other)
    u[1:-1, 1:-1] = held[1:-1, 1:-1] + 3.0 * u[2:, 1:-1]
    expected[1:-1, 1:-1] = other[1:-1, 1:-1] + 3.0 * expected[2:, 1:-1]
    u[1:-1, 1:-1] = held[1:-1, 1:-1] - u[:-2, 1:-1]
    expected[1:-1, 1:-1] = other[1:-1, 1:-1] - expected[:-2, 1:-1]
    u[:, :] = u * (held - u)
    expected[:, :] = expected * (other - expected)
    assert same(expected, u.gather())
    # The one subtraction of the second line is a program of one step, written into the part like the others.
    assert fuse.STATS.calls == {"fuse_add_scaled_f64": 1, "fuse_sub_v_v_f64": 1, "fuse_mul_difference_f64": 1}


def test_a_loop_of_fused_writes_into_parts_of_an_array_copies_nothing():
    """Each expression is computed straight into its part of the array: no copy of the array, turn after turn."""
    grid, other = arrays(2, (30, 40))
    u, expected = lazy.array(grid), grid.copy()
    evaluate.WRITES.update(in_place=0, copied=0)
    for _ in range(4):
        u[1:-1, 1:-1] = lazy.asarray(other)[1:-1, 1:-1] * 0.5 + lazy.asarray(other)[2:, 1:-1]
        u[0, :] = u[1, :]
        u[2:, 3:] = (lazy.asarray(other)[2:, 3:] - 1.5) * lazy.asarray(other)[:-2, :-3]
        expected[1:-1, 1:-1] = other[1:-1, 1:-1] * 0.5 + other[2:, 1:-1]
        expected[0, :] = expected[1, :]
        expected[2:, 3:] = (other[2:, 3:] - 1.5) * other[:-2, :-3]
    assert same(expected, u.gather())
    assert evaluate.WRITES == {"in_place": 12, "copied": 0} and fuse.STATS.declined == 0


def test_an_in_place_operator_on_part_of_an_array_reads_what_the_part_held():
    """`u[1:-1] += s * v` is one kernel call that reads the part of the old array and writes the part of the new."""
    grid, other = arrays(2, (30, 40))
    u, expected = lazy.array(grid), grid.copy()
    u[1:-1] += 2.0 * u[:-2]
    expected[1:-1] += 2.0 * expected[:-2]
    u[:, 3:9] -= lazy.asarray(other)[:, 3:9] * u[:, 10:16]
    expected[:, 3:9] -= other[:, 3:9] * expected[:, 10:16]
    assert same(expected, u.gather())
    assert fuse.STATS.calls == {"fuse_add_scaled_f64": 1, "fuse_sub_product_f64": 1}


def test_a_ufunc_with_out_is_the_ufuncs_one_call():
    """`np.add(a, b, out=x)` into an array the layer made is NumPy's one call, whatever `x` held."""
    a, b = arrays(2, (200,))
    x = lazy.zeros(200)
    numpy.add(lazy.asarray(a), lazy.asarray(b), out=x)
    numpy.multiply(x, 0.5, out=x)
    assert same((a + b) * 0.5, x.gather())
    assert fuse.STATS.calls == {"fuse_scaled_sum_f64": 1}


def test_a_write_made_at_once_leaves_nothing_pending():
    """A write into memory the program holds computes everything pending first, the nodes inside kernels included."""
    a, b = arrays(2, (100,))
    plain = a.copy()
    held = lazy.asarray(plain)
    scaled = held + 0.5 * lazy.asarray(b)
    product = held * lazy.asarray(b)
    total = product + 0.5 * held
    held[0] = 7.0
    assert graph.pending_count() == 0 and plain[0] == 7.0
    assert same(a + 0.5 * b, scaled.gather()) and same(a * b + 0.5 * a, total.gather()) and same(a * b, product.gather())


def test_a_recording_still_has_every_node():
    """A node computed inside a kernel is in the recorded graph, as computed."""
    a, b = arrays(2, (50,))
    with record_graphs() as recording:
        (lazy.asarray(a) + 0.5 * lazy.asarray(b)).gather()
    (recorded,) = recording.graphs
    assert sorted(node.name for node in recorded.nodes if node.category == "elementwise") == ["add", "multiply"]
    assert all(node.state == "computed" for node in recorded.nodes)
    assert calls() == 1


def test_a_strict_array_is_fused_and_still_refuses():
    """Fusion changes nothing about what a strict array gives without `gather()`."""
    strict = get_lazy(strict=True)
    a, b = arrays(2, (50,))
    pending = strict.asarray(a) + 0.5 * strict.asarray(b)
    with pytest.raises(TypeError):
        float(pending[0])
    assert same(a + 0.5 * b, pending.gather()) and calls() == 1


def test_nothing_planned_outlives_the_computation():
    """The nodes inside a kernel call are freed when it returns, without Python's collector of cycles having to run.

    Were a plan to keep them, the arrays they read would stay in memory
    until the collector's next run, and every result in between would be
    written to fresh memory where NumPy's own calls reuse what was just freed.
    """
    a, b, c = arrays(3, (500,))
    la, lb, lc = (lazy.asarray(operand) for operand in (a, b, c))
    gc.collect()
    gc.disable()
    try:
        value = gathered(lambda: (la + lb * lc) * (lc - la) + 0.5 * (lb + la))
        pending = graph.pending_count()
    finally:
        gc.enable()
    assert same((a + b * c) * (c - a) + 0.5 * (b + a), value) and pending == 0 and calls() >= 2


@pytest.mark.parametrize("shape", ((12, 9, 26, 3), (5, 40, 12, 2), (4, 7, 31, 3)))
def test_a_program_given_up_for_a_value_of_each_row_has_a_second_that_reads_it_as_an_array(shape: tuple[int, ...]):
    """Grids taken at one time level beside one value for each row: as numbers the values cannot be gathered with the
    grids, so the expression's second program runs, with NumPy's bits, and leaves no node behind it."""
    a, b = arrays(2, shape)
    grid = shape[:-1]
    c = arrays(1, grid)[0]
    per_row, per_plane = arrays(1, (grid[1], 1))[0], arrays(1, (grid[0], 1, 1))[0]

    def program(np: Any) -> Any:
        """A flux of many operations on two such grids, scaled by a value for each row and one for each plane."""
        vel, var = np.asarray(a)[..., 1], np.asarray(b)[..., 0]
        cr, dx, dy = np.asarray(c), np.asarray(per_row), np.asarray(per_plane)
        speed = np.abs(dy * vel * 0.5 / dx)
        flux = dy * vel * (var + cr) * 0.5 - np.abs(dy * vel) * ((1.0 - cr) + speed * cr) * (var - cr) * 0.5
        into = np.zeros(grid)
        into[1:-1] = (dy * vel * (var + cr) * 0.5 - np.abs(dy * vel) * ((1.0 - cr) + cr * cr) * (var - cr) / dx)[1:-1]
        return flux, into

    expected = program(numpy)
    gc.collect()
    gc.disable()
    try:
        got = [each.gather() for each in program(lazy)]
        pending = graph.pending_count()
    finally:
        gc.enable()
    for theirs, ours in zip(expected, got):
        assert same(theirs, ours)
    assert pending == 0 and fuse.STATS.fused_nodes >= 12
    # A few operations on as many arrays to copy are not worth the copies, as a second program or as a first.
    declined = fuse.STATS.declined
    few = gathered(lambda: (lazy.asarray(a)[..., 1] - lazy.asarray(b)[..., 0]) / lazy.asarray(per_row))
    assert same((a[..., 1] - b[..., 0]) / per_row, few) and fuse.STATS.declined > declined


def test_a_line_whose_kernel_was_flagged_is_left_to_numpy_afterwards():
    """A loop whose arithmetic overflows computes the expression twice on its first turn and once on every other."""
    huge = lazy.asarray(numpy.full(64, 1e308))
    wanted = numpy.full(64, numpy.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            pending = huge + 10.0 * huge
            assert same(wanted, pending.gather())
    assert fuse.STATS.flagged == 1 and calls() == 0


def test_few_elements_are_left_to_numpy_when_a_floor_is_set(monkeypatch: pytest.MonkeyPatch):
    """Expressions on arrays below `fuse.MIN_ELEMENTS` are not read at all."""
    monkeypatch.setattr(fuse, "MIN_ELEMENTS", 1000)
    small, large = arrays(1, (999,))[0], arrays(1, (1000,))[0]
    assert same(small + 0.5 * small, gathered(lambda: lazy.asarray(small) + 0.5 * lazy.asarray(small)))
    assert calls() == 0
    assert same(large + 0.5 * large, gathered(lambda: lazy.asarray(large) + 0.5 * lazy.asarray(large)))
    assert calls() == 1


def test_a_node_held_by_the_program_is_seen_by_its_references():
    """The count of references that says a node's value is wanted again is right for the two cases it separates."""
    a, b = arrays(2, (10,))
    kept = lazy.asarray(a) * 2.0
    for held, expected in ((True, 1), (False, 0)):
        root = kept + lazy.asarray(b) if held else (lazy.asarray(a) * 2.0) + lazy.asarray(b)
        todo = evaluate._needed([root._node])
        inner = {node.number for node in todo if node is not root._node}
        assert len(inner & fuse.held_among(todo, fuse.count_uses(todo))) == expected
        del todo


def _written(part: Any) -> tuple:
    """Write a tree down with the number of the node at each term, and nothing that refers to a node."""
    if type(part) is fuse.Leaf:
        return (part.kind, part.place)
    return (part.op, part.node.number, *[_written(operand) for operand in part.operands])


def _read_operand(operand: Any) -> tuple:
    """Write down one thing an expression reads, without referring to it."""
    if type(operand) is graph.Node:
        return ("node", operand.number)
    if type(operand) is fuse.ViewOf:
        return ("part", operand.old.number, len(operand.steps))
    if type(operand) is numpy.ndarray:
        return ("array", id(operand))
    return (type(operand).__name__, repr(operand))


def _described(expressions: list) -> list:
    """Write down everything `fuse.find` says of each expression, as values that keep no node alive."""
    described = []
    for expression in expressions:
        target = expression.target
        steps = None if target is None else [(step.name, repr(step.tail), step.shape, step.plain) for step in target.steps]
        described.append((expression.root.number, expression.result, expression.shape, _written(expression.term),
                          [_read_operand(operand) for operand in expression.operands],
                          None if target is None else target.old.number, steps))
    return described


# The programs whose computations both readers of expressions are run on: the ones with the most to read, of
# every kind an expression has (an array, a sum, a part written into, a mask).
READ_BOTH_WAYS = ("cavity_flow", "hdiff", "veros_eos", "veros_isoneutral", "veros_tke", "particle_filter", "regression",
                  "black_scholes", "pvlib_solar")


def _read_both_ways(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Have every computation's expressions read by `_Finder` and by the C core, and the two compared.

    Args:
        monkeypatch: The test's, which puts `fuse.find` back afterwards.

    Returns:
        A list that gains, for each computation read from now on, how many expressions both readers found.
    """
    find, reading = fuse.find, fuse._reading_core
    compared: list[int] = []

    def both(todo: list, fewest: int = 1, alone: bool = False) -> tuple:
        """Read a computation's expressions in Python, then in C, and hand on the second after comparing them.

        What the first reading found is written down and dropped before the second: an expression refers to
        its nodes, and a node something refers to is one the reader does not compute past.
        """
        fuse._reading_core = lambda: None
        try:
            found, inside = find(todo, fewest, alone)
        finally:
            fuse._reading_core = reading
        theirs = (_described(found), sorted(inside))
        del found, inside
        found, inside = find(todo, fewest, alone)
        assert (_described(found), sorted(inside)) == theirs
        for expression in found:
            # What the core's reader writes down for planning is what planning works out from the tree.
            terms = list(fuse.terms_of(expression.term))
            assert expression.signature == fuse._signature(expression.term)
            assert [node.number for node in expression.nodes] == sorted(term.node.number for term in terms)
            assert {id(settings) for settings in expression.errstates} == {id(term.node.errstate) for term in terms}
        compared.append(len(found))
        return found, inside

    monkeypatch.setattr(fuse, "find", both)
    return compared


@pytest.mark.parametrize("name", READ_BOTH_WAYS)
def test_the_core_reads_the_expressions_python_reads(name: str, lazy_core: str, monkeypatch: pytest.MonkeyPatch):
    """With the C core `fuse.find` is the core's; on every computation of a program it finds what `_Finder` finds."""
    if lazy_core != "c":
        pytest.skip("one reader only without the C core")
    (program,) = corpus.by_name([name])
    if not program.available:
        pytest.skip(f"needs {', '.join(program.requires)}")
    compared = _read_both_ways(monkeypatch)
    inputs = program.make(corpus.SMALL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        array.gather(*_flat(program.run(lazy, inputs)))
    assert sum(compared) > 0


def _flat(result: Any) -> list:
    """List what a program returned, whatever tuples and lists it is in."""
    if isinstance(result, (tuple, list)):
        return [value for item in result for value in _flat(item)]
    return [result]


@pytest.mark.parametrize("depth", (fuse.MOST_DEPTH - 1, fuse.MOST_DEPTH, fuse.MOST_DEPTH + 1, 5 * fuse.MOST_DEPTH + 3))
def test_an_expression_too_deep_to_read_whole_is_read_in_pieces(depth: int):
    """A loop that adds to one array again and again writes one long chain: every operation is fused all the same."""
    def program(np: Any) -> Any:
        """Add to an array `depth` times, never asking for a value."""
        x = np.asarray(numpy.linspace(0.0, 1.0, 64))
        y = x
        for turn in range(depth):
            y = y + 0.5 if turn % 2 else y * 1.0625
        return y

    expected = program(numpy)
    result = program(lazy)
    got = result.gather()
    assert same(expected, got)
    # A piece is an operation and MOST_DEPTH more under it; a last piece of one operation is NumPy's call.
    pieces, left = divmod(depth, fuse.MOST_DEPTH + 1)
    assert fuse.STATS.expressions == pieces + (left > 1)
    assert fuse.STATS.declined == 0 and fuse.STATS.fused_nodes == depth - fuse.STATS.expressions - (left == 1)


@pytest.mark.parametrize("name", [program.name for program in corpus.programs() + codebases.programs() if program.eager])
def test_a_program_of_the_corpus_gives_numpys_results_with_fusion(name: str):
    """Each program run with the stand-in and fusion matches the program run with NumPy, bit for bit."""
    (program,) = corpus.by_name([name])
    if not program.available:
        pytest.skip(f"needs {', '.join(program.requires)}")
    inputs = program.make(corpus.SMALL)

    def values(result: Any) -> list:
        """Gather a program's results, whatever tuples and lists they are in."""
        if isinstance(result, (tuple, list)):
            return [value for item in result for value in values(item)]
        return [result.gather() if isinstance(result, LazyArray) else result]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = values(program.run(numpy, inputs))
        got = values(program.run(lazy, inputs))
    assert len(expected) == len(got)
    for wanted, value in zip(expected, got):
        assert same(numpy.asarray(wanted), numpy.asarray(value))
