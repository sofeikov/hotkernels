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
from prototypes.lazy.programs import corpus

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
    """Columns of a grid, a reversed array and a Fortran-ordered one are not walked; the results are NumPy's."""
    a, b = arrays(2, (30, 40))
    fortran = numpy.asfortranarray(a)
    cases = [
        (a[:, ::2] + 0.5 * b[:, ::2], lazy.asarray(a)[:, ::2] + 0.5 * lazy.asarray(b)[:, ::2]),
        (a[::-1] + 0.5 * b, lazy.asarray(a)[::-1] + 0.5 * lazy.asarray(b)),
        (fortran + 0.5 * fortran, lazy.asarray(fortran) + 0.5 * lazy.asarray(fortran)),
        (a.T + 0.5 * b.T, lazy.asarray(a).T + 0.5 * lazy.asarray(b).T),
    ]
    for expected, got in cases:
        value = got.gather()
        assert same(expected, value) and value.flags.c_contiguous == expected.flags.c_contiguous
    assert calls() == 0 and fuse.STATS.declined == len(cases)


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
    # Sixteen operations, one group: fifteen nodes have no array of their own.
    assert fuse.STATS.fused_nodes == 15 and fuse.STATS.declined == 0 and calls() < 16


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
    column = gathered(lambda: (lazy.asarray(table)[:, 1] * lazy.asarray(table)[:, 2] - lazy.asarray(other)[:, 0]) / 3.0)
    assert same((table[:, 1] * table[:, 2] - other[:, 0]) / 3.0, column)
    assert fuse.STATS.declined == 0 and calls() >= 2


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
    """Every written kernel's spec and reference are the tool's, and so is its assembly until its log has a second attempt."""
    kept = make_kernel.KERNELS
    monkeypatch.setattr(make_kernel, "KERNELS", tmp_path)
    for shape in make_kernel.SHAPES:
        written = make_kernel.write(shape)
        theirs = kept / written.name
        tuned = "\n## 2." in (theirs / make_kernel.ARCH / "attempts.md").read_text()
        for path in sorted(written.rglob("*")):
            if path.is_file() and path.name != "attempts.md" and not (tuned and path.name == "kernel.s"):
                assert path.read_text() == (kept / path.relative_to(tmp_path)).read_text(), path.name
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


@pytest.mark.parametrize("name", [program.name for program in corpus.programs() if program.eager])
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
