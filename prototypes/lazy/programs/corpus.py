"""The programs fusion is studied on: each can be run with NumPy or with the lazy stand-in for it.

Twenty-six programs. Three are this directory's examples, three are
published benchmarks copied unchanged into `vendor/`, and the rest are every
program the repository times with and without `hotkernels.install()`: the
ten applications of `delivery/apps/` (the three Veros routines count one
each) and the nine workloads of `delivery/workloads/`. A program is run by
`Program.run(np, inputs)`: with the real `numpy` module it is the eager
program, and with the stand-in every `np` in it is the stand-in, which is
the one line a user would change.

Inputs are plain `ndarray`s and numbers made once by `Program.make(size)`.
A program that writes into its inputs is given copies made with the module
it runs with (`np.array(a)`), so the eager and the lazy run start alike and
the lazy copies are the layer's own.

**A program that is calls into a library** (pandas, scikit-learn,
scikit-image, Astropy, trimesh, empyrical) changes its one line like any
other: the `np` of the module that calls the library is the stand-in, and
the arrays it is given are the stand-in's (`np.asarray(a)`, no copy). The
library has its own `numpy` and asks for `ndarray`s, so each array is made
where it is handed over and what the library computes is in no graph. Such a
program says what the layer costs a program it can do nothing for.
"""
from __future__ import annotations

import importlib
import importlib.util
import re
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Iterator

import numpy

from delivery import apps as applications
from delivery import workloads
from delivery.apps import min_char_rnn, mlp_train, particle_filter, veros_ocean
from prototypes.lazy.programs import attention, monitor, regression
from prototypes.lazy.programs.vendor.npbench import cavity_flow_numpy, hdiff_numpy

# The two sizes every program has.
SMALL, LARGE = "small", "large"
# Seed of the data the programs written here are given.
SEED = 0

# Linear regression: (examples, features, gradient steps) by size.
REGRESSION = {SMALL: (200, 8, 50), LARGE: (200_000, 16, 20)}
# Transformer block: (tokens, width, heads, blocks run one after another) by size.
ATTENTION = {SMALL: (16, 32, 4, 10), LARGE: (256, 256, 4, 4)}
# The block's feed-forward layer is this many times as wide as the block.
WIDENING = 4
# Two-layer perceptron: training examples by size; the network is `mlp_train`'s small one at both.
PERCEPTRON_EXAMPLES = {SMALL: 1_280, LARGE: 12_000}
# Iterations of training of the recurrent network, by size.
RNN_ITERATIONS = {SMALL: 3, LARGE: 30}
# Steps of the particle filter, by size; the number of particles is `particle_filter`'s.
FILTER_STEPS = {SMALL: 20, LARGE: 40}
# Cavity flow: (grid points per side, time steps, pressure iterations per step) by size. Density and viscosity are
# npbench's.
CAVITY = {SMALL: (61, 25, 5), LARGE: (401, 20, 10)}
CAVITY_DENSITY, CAVITY_VISCOSITY = 1.0, 0.1
# Horizontal diffusion: the grid (I, J, K) by size, npbench's S and paper sizes.
HDIFF = {SMALL: (64, 64, 60), LARGE: (256, 256, 160)}
# Black-Scholes: options priced, by size.
OPTIONS = {SMALL: 4_096, LARGE: 2_000_000}
# Black-Scholes: the ranges prices, strikes and maturities are drawn from, and the two rates, as dpbench has them.
PRICE_RANGE, STRIKE_RANGE, MATURITY_RANGE = (10.0, 50.0), (10.0, 50.0), (1.0, 2.0)
RISK_FREE, VOLATILITY = 0.1, 0.2
# The line of `min-char-rnn.py` that says what `np` is: the lazy run takes it out and supplies the stand-in.
RNN_IMPORT = (r"^import numpy as np$", "", "the module is supplied by the caller")
# The routines of the Veros ocean model, by the short name used here.
VEROS_ROUTINES = {"eos": "equation_of_state", "isoneutral": "isoneutral_mixing", "tke": "turbulent_kinetic_energy"}
# The module of dpbench's Black-Scholes, imported when it is run: it needs SciPy, which not every environment has.
BLACK_SCHOLES = "prototypes.lazy.programs.vendor.dpbench.black_scholes_numpy"
# The library two programs need beside NumPy: SciPy's error function, and its LAPACK for the Veros kinetic energy.
SCIPY = "scipy"
# The size of `delivery/workloads/` each workload is run at, by this corpus's sizes: about 10 ms and about 100 ms of
# plain NumPy on the machine they were set on.
WORKLOAD_SIZES = {SMALL: "S", LARGE: "M"}
# Where each workload's NumPy code is, as modules that have a global `np`, relative to `delivery.workloads`. A
# workload that is a call into a library has none. `example` is a file outside the package and is found by itself.
WORKLOAD_MODULES = {
    "example": (),
    "flame": ("flame",),
    "azimint_naive": ("vendor.npbench.azimint_naive_numpy",),
    "azimint_hist": ("vendor.npbench.azimint_hist_numpy",),
    "mandelbrot1": ("vendor.npbench.mandelbrot1_numpy",),
    "mandelbrot2": ("vendor.npbench.mandelbrot2_numpy",),
    "make_decision": ("vendor.numpy_benchmarks.make_decision",),
    "pandas_filter": (),
    "equalize_hist": (),
}
# The applications of `delivery/apps/` taken as they are, by the name each has here. The four others are in the corpus
# under names of their own: `perceptron`, `char_rnn`, `particle_filter` and the three Veros routines.
APPLICATIONS = {"trimesh_mesh": "trimesh_mesh", "sklearn_metrics": "sklearn_metrics",
                "empyrical_stats": "empyrical_stats", "skimage_pipeline": "skimage_pipeline",
                "pandas_sales": "pandas_sales", "astropy_timeseries": "astropy_timeseries"}


@dataclass(frozen=True)
class Program:
    """One program of the corpus."""

    # Short name, as a command line takes it.
    name: str
    # What it is, for a table.
    title: str
    # Where its code is.
    source: str
    # Makes its inputs for a size, SMALL or LARGE, with the real `numpy`: plain arrays and numbers.
    make: Callable[[str], tuple]
    # Runs it: (the `numpy` module or the lazy stand-in, the inputs) -> its results, lazy arrays where the stand-in
    # ran it, in any nesting of tuples and lists.
    run: Callable[[Any, tuple], Any]
    # One line per size saying how large its data is.
    sizes: dict[str, str]
    # False for a program that calls `gather()` itself and so runs with the stand-in only.
    eager: bool = True
    # The libraries it needs beside NumPy, by the names they are imported by.
    requires: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether every library the program needs is installed."""
        return all(importlib.util.find_spec(name) is not None for name in self.requires)


@contextmanager
def numpy_of(np: Any, *modules: ModuleType) -> Iterator[None]:
    """Make `np` the module some modules compute with, for the length of a block.

    This is the one changed line, made from outside: each module's global
    `np` is what its `import numpy as np` bound.

    Args:
        np: The `numpy` module or the lazy stand-in.
        *modules: Modules that have a global named `np`.
    """
    before = [module.np for module in modules]
    for module in modules:
        module.np = np
    try:
        yield
    finally:
        for module, original in zip(modules, before):
            module.np = original


def _regression_data(size: str) -> tuple:
    """Draw a regression problem.

    Args:
        size: SMALL or LARGE.

    Returns:
        (features, targets, gradient steps).
    """
    examples, features, steps = REGRESSION[size]
    rng = numpy.random.default_rng(SEED)
    x = rng.standard_normal((examples, features))
    return x, x @ rng.standard_normal(features) + 0.1 * rng.standard_normal(examples), steps


def _run_regression(np: Any, inputs: tuple) -> Any:
    """Run the linear regression."""
    features, targets, steps = inputs
    return regression.fit(np, features, targets, steps)


def _run_monitor(np: Any, inputs: tuple) -> Any:
    """Run the regression that gathers its loss as it goes."""
    features, targets, steps = inputs
    return monitor.fit(np, features, targets, steps)


def _attention_data(size: str) -> tuple:
    """Draw a transformer block's weights and input, float32 as such weights usually are.

    Args:
        size: SMALL or LARGE.

    Returns:
        (activations, the block's parameters by name, heads, blocks to run).
    """
    tokens, width, heads, blocks = ATTENTION[size]
    rng = numpy.random.default_rng(SEED)
    shapes = {"gain_1": (width,), "bias_1": (width,), "gain_2": (width,), "bias_2": (width,),
              "w_qkv": (width, 3 * width), "b_qkv": (3 * width,), "w_out": (width, width), "b_out": (width,),
              "w_up": (width, WIDENING * width), "b_up": (WIDENING * width,),
              "w_down": (WIDENING * width, width), "b_down": (width,)}
    scale = numpy.float32(1.0 / numpy.sqrt(width))
    parameters = {name: rng.standard_normal(shape).astype(numpy.float32) * scale for name, shape in shapes.items()}
    return rng.standard_normal((tokens, width)).astype(numpy.float32), parameters, heads, blocks


def _run_attention(np: Any, inputs: tuple) -> Any:
    """Run the transformer block several times, each on the result of the one before."""
    x, parameters, heads, blocks = inputs
    x = np.asarray(x)
    for _ in range(blocks):
        x = attention.block(x, parameters, heads)
    return x


def _perceptron_data(size: str) -> tuple:
    """Build the perceptron's classification problem, cut to this corpus's number of examples.

    Args:
        size: SMALL or LARGE.

    Returns:
        (training inputs, training labels, held-out inputs, held-out labels, hidden units).
    """
    x, y, x_held, y_held, hidden = mlp_train.make_data(SMALL)
    examples = PERCEPTRON_EXAMPLES[size]
    return x[:examples], y[:examples], x_held, y_held, hidden


def _run_perceptron(np: Any, inputs: tuple) -> Any:
    """Train the two-layer perceptron."""
    with numpy_of(np, mlp_train):
        return mlp_train.run(*inputs)


def _rnn_data(size: str) -> tuple:
    """Prepare `min-char-rnn.py` twice: as published, and without its import line.

    Args:
        size: SMALL or LARGE.

    Returns:
        (the compiled program, the same with `np` left for the caller to supply, the training text).
    """
    source = min_char_rnn.PROGRAM.read_text()
    rewrites = [(pattern, replacement) for pattern, replacement, _ in min_char_rnn.REWRITES]
    rewrites[-1] = (rewrites[-1][0], f"while n < {RNN_ITERATIONS[size]}:")
    for pattern, replacement in rewrites:
        source = re.sub(pattern, replacement, source, flags=re.MULTILINE)
    supplied = re.sub(RNN_IMPORT[0], RNN_IMPORT[1], source, flags=re.MULTILINE)
    path = str(min_char_rnn.PROGRAM)
    return compile(source, path, "exec"), compile(supplied, path, "exec"), min_char_rnn.TEXT.read_text()


def _run_rnn(np: Any, inputs: tuple) -> Any:
    """Run `min-char-rnn.py` from the start, with `np` as its module."""
    _, supplied, text = inputs

    class _Input:
        """Stands in for the file the program opens."""

        def read(self) -> str:
            """Return the training text."""
            return text

    def _silent(*_args: Any) -> None:
        """Take the program's progress lines and discard them."""

    numpy.random.seed(min_char_rnn.SEED)
    scope = {"open": lambda *_args: _Input(), "print": _silent, "__name__": "min_char_rnn", "np": np}
    exec(supplied, scope)
    return tuple(scope[name] for name in min_char_rnn.RESULTS)


def _filter_data(size: str) -> tuple:
    """Simulate the particle filter's measurements, cut to this corpus's number of steps.

    Args:
        size: SMALL or LARGE.

    Returns:
        (number of particles, measurements).
    """
    particles, measurements = particle_filter.make_data(size)
    return particles, measurements[:FILTER_STEPS[size]]


def _run_filter(np: Any, inputs: tuple) -> Any:
    """Run the particle filter."""
    with numpy_of(np, particle_filter):
        return particle_filter.run(*inputs)


def _veros(routine: str) -> tuple[Callable[[str], tuple], Callable[[Any, tuple], Any]]:
    """Build the two functions of one routine of the Veros ocean model.

    Args:
        routine: A value of VEROS_ROUTINES.

    Returns:
        (make, run) for a `Program`.
    """
    package = importlib.import_module(f"{veros_ocean.VENDOR}.{routine}")

    def make(size: str) -> tuple:
        """Generate the routine's inputs with pyhpc-benchmarks' own generator."""
        return tuple(package.generate_inputs(veros_ocean.GRID_POINTS[size]))

    def run(np: Any, inputs: tuple) -> Any:
        """Run the routine on copies of its inputs, which two of the three write into."""
        # Imported here and not above: the kinetic energy's module needs SciPy, which `Program.requires` says.
        module = package.try_import(veros_ocean.IMPLEMENTATION)
        with numpy_of(np, module):
            return module.run(*(np.array(array) for array in inputs))

    return make, run


def _cavity_data(size: str) -> tuple:
    """Set a lid-driven cavity at rest, as npbench's `initialize` does.

    Args:
        size: SMALL or LARGE.

    Returns:
        The arguments of `cavity_flow`: (nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu).
    """
    side, steps, iterations = CAVITY[size]
    u, v, p = (numpy.zeros((side, side)) for _ in range(3))
    spacing = 2 / (side - 1)
    dt = 0.1 / ((side - 1) * (side - 1))
    return side, side, steps, iterations, u, v, dt, spacing, spacing, p, CAVITY_DENSITY, CAVITY_VISCOSITY


def _run_cavity(np: Any, inputs: tuple) -> Any:
    """Run the cavity flow on copies of its three fields, which it writes into, and return them."""
    nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu = inputs
    u, v, p = np.array(u), np.array(v), np.array(p)
    with numpy_of(np, cavity_flow_numpy):
        cavity_flow_numpy.cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu)
    return u, v, p


def _hdiff_data(size: str) -> tuple:
    """Draw the fields of the horizontal diffusion, as npbench's `initialize` does.

    Args:
        size: SMALL or LARGE.

    Returns:
        (the input field with its halo of two points, the output field, the coefficients).
    """
    i, j, k = HDIFF[size]
    rng = numpy.random.default_rng(SEED)
    return rng.random((i + 4, j + 4, k)), rng.random((i, j, k)), rng.random((i, j, k))


def _run_hdiff(np: Any, inputs: tuple) -> Any:
    """Run the horizontal diffusion into a copy of its output field and return it."""
    in_field, out_field, coeff = (np.array(array) for array in inputs)
    with numpy_of(np, hdiff_numpy):
        hdiff_numpy.hdiff(in_field, out_field, coeff)
    return out_field


def _options_data(size: str) -> tuple:
    """Draw options to price, in dpbench's ranges.

    Args:
        size: SMALL or LARGE.

    Returns:
        (number of options, prices, strikes, maturities, risk-free rate, volatility, room for calls, room for puts).
    """
    count = OPTIONS[size]
    rng = numpy.random.default_rng(SEED)
    price, strike, maturity = (rng.uniform(low, high, count) for low, high in (PRICE_RANGE, STRIKE_RANGE, MATURITY_RANGE))
    return count, price, strike, maturity, RISK_FREE, VOLATILITY, numpy.zeros(count), -numpy.ones(count)


def _run_options(np: Any, inputs: tuple) -> Any:
    """Price the options into copies of the two output arrays and return them."""
    count, price, strike, maturity, rate, volatility, call, put = inputs
    price, strike, maturity, call, put = (np.array(array) for array in (price, strike, maturity, call, put))
    black_scholes_numpy = importlib.import_module(BLACK_SCHOLES)
    with numpy_of(np, black_scholes_numpy):
        black_scholes_numpy.black_scholes(count, price, strike, maturity, rate, volatility, call, put)
    return call, put


# Applications given plain `ndarray`s whatever module they run with. scikit-image's `regionprops_table` asks
# `np.isscalar` of the mean of an image it was handed; for a lazy image that mean is a lazy array without
# dimensions, not a number, and the library fails on it. With plain inputs nothing of the program is deferred.
PLAIN_INPUTS = frozenset({"skimage_pipeline"})


def _handed(np: Any, value: Any) -> Any:
    """Hand an input to a program as an array of the module it runs with.

    Args:
        np: The `numpy` module or the lazy stand-in.
        value: An input: an `ndarray`, a list or tuple of inputs, or anything else.

    Returns:
        `np.asarray(value)` for an `ndarray`, which copies nothing; a list or tuple with its items handed the same
        way; any other value as it is (a number, a pandas object).
    """
    if isinstance(value, numpy.ndarray):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_handed(np, item) for item in value)
    return value


def _workload(workload: workloads.Workload) -> Program:
    """Make a workload of `delivery/workloads/` a program of the corpus.

    Args:
        workload: The workload.

    Returns:
        The program, at the workload's S and M sizes; one that has a single size runs at it for both.
    """
    names = {size: workload_size if workload_size in workload.sizes else next(iter(workload.sizes))
             for size, workload_size in WORKLOAD_SIZES.items()}

    def make(size: str) -> tuple:
        """Build the workload's inputs at one size."""
        return workload.make_data(**workload.sizes[names[size]])

    def run(np: Any, inputs: tuple) -> Any:
        """Run the workload with `np` as the module of its NumPy code."""
        modules = [importlib.import_module(f"{workloads.__name__}.{name}") for name in WORKLOAD_MODULES[workload.name]]
        if workload.name == "example":
            from delivery.workloads import example
            modules = [example._program()]
        with numpy_of(np, *modules):
            return workload.run(*_handed(np, inputs))

    def described(arguments: dict) -> str:
        """Write one size's arguments as a line."""
        return ", ".join(f"{value:,} {name}" for name, value in arguments.items()) or "its own size, two million elements"

    return Program(workload.name, workload.title, "delivery/workloads/" + workload.name + ".py", make, run,
                   {size: described(workload.sizes[name]) for size, name in names.items()},
                   requires=tuple(workloads._import_name(name) for name in workload.requires))


def _application(name: str) -> Program:
    """Make an application of `delivery/apps/` a program of the corpus, as it is.

    Args:
        name: A key of APPLICATIONS.

    Returns:
        The program, at the application's two sizes.
    """
    module = applications.load(APPLICATIONS[name])

    def run(np: Any, inputs: tuple) -> Any:
        """Run the application with `np` as its module's NumPy."""
        with numpy_of(np, module):
            return module.run(*(inputs if name in PLAIN_INPUTS else _handed(np, inputs)))

    return Program(name, module.TITLE, "delivery/apps/" + APPLICATIONS[name] + ".py", module.make_data, run,
                   dict(module.SIZES), requires=tuple(module.REQUIRES))


def _described(table: dict[str, tuple], text: str) -> dict[str, str]:
    """Write one line per size from a table of sizes.

    Args:
        table: Size -> the numbers that size has.
        text: A format string taking those numbers.

    Returns:
        Size -> the line.
    """
    return {size: text.format(*numbers) for size, numbers in table.items()}


def programs() -> list[Program]:
    """Return the corpus, in the order its tables are written."""
    veros = {name: _veros(routine) for name, routine in VEROS_ROUTINES.items()}
    grids = {size: "a grid of {:,} × {:,} × {:,} points".format(*veros_ocean._shape(points))
             for size, points in veros_ocean.GRID_POINTS.items()}
    here, apps, vendor = "prototypes/lazy/programs/", "delivery/apps/", "prototypes/lazy/programs/vendor/"
    return [
        Program("regression", "Linear regression by gradient descent", here + "regression.py",
                _regression_data, _run_regression, _described(REGRESSION, "{:,} examples of {} features, {} steps")),
        Program("monitor", "The same regression, its loss gathered every fifth step", here + "monitor.py",
                _regression_data, _run_monitor, _described(REGRESSION, "{:,} examples of {} features, {} steps"),
                eager=False),
        Program("attention", "Transformer block, forward pass, float32", here + "attention.py",
                _attention_data, _run_attention, _described(ATTENTION, "{} tokens of width {}, {} heads, {} blocks")),
        Program("perceptron", "Two-layer perceptron, one epoch of mini-batches", apps + "mlp_train.py",
                _perceptron_data, _run_perceptron,
                {size: f"{count:,} examples, batches of {mlp_train.BATCH}, {mlp_train.EPOCHS} epochs"
                 for size, count in PERCEPTRON_EXAMPLES.items()}),
        Program("char_rnn", "Karpathy's min-char-rnn.py", apps + "vendor/min_char_rnn/",
                _rnn_data, _run_rnn, {size: f"{count} iterations of 25 characters, 100 hidden units"
                                      for size, count in RNN_ITERATIONS.items()}),
        Program("particle_filter", "Bootstrap particle filter", apps + "particle_filter.py",
                _filter_data, _run_filter, {size: f"{particle_filter.PARTICLES[size]:,} particles, {steps} steps"
                                            for size, steps in FILTER_STEPS.items()}),
        Program("veros_eos", "Veros ocean model: equation of state", apps + "vendor/pyhpc/equation_of_state/",
                *veros["eos"], grids),
        Program("veros_isoneutral", "Veros ocean model: isoneutral mixing", apps + "vendor/pyhpc/isoneutral_mixing/",
                *veros["isoneutral"], grids),
        Program("veros_tke", "Veros ocean model: turbulent kinetic energy",
                apps + "vendor/pyhpc/turbulent_kinetic_energy/", *veros["tke"], grids, requires=(SCIPY,)),
        Program("cavity_flow", "CFD Python: lid-driven cavity flow (npbench)", vendor + "npbench/cavity_flow_numpy.py",
                _cavity_data, _run_cavity, _described(CAVITY, "a grid of {0} × {0}, {1} steps of {2} pressure iterations")),
        Program("hdiff", "Horizontal diffusion, a weather-model stencil (npbench)", vendor + "npbench/hdiff_numpy.py",
                _hdiff_data, _run_hdiff, _described(HDIFF, "a grid of {} × {} × {}")),
        Program("black_scholes", "Black-Scholes option pricing (dpbench)", vendor + "dpbench/black_scholes_numpy.py",
                _options_data, _run_options, {size: f"{count:,} options" for size, count in OPTIONS.items()},
                requires=(SCIPY,)),
        *(_application(name) for name in APPLICATIONS),
        *(_workload(workload) for workload in workloads.load()),
    ]


def by_name(names: list[str] | None = None) -> list[Program]:
    """Pick programs of the corpus by name.

    Args:
        names: The names wanted, or None for every program.

    Returns:
        The programs, in the corpus's order.

    Raises:
        KeyError: For a name that is not a program's.
    """
    everything = programs()
    if not names:
        return everything
    known = {program.name for program in everything}
    for name in names:
        if name not in known:
            raise KeyError(f"no program named {name!r}; there are {', '.join(sorted(known))}")
    return [program for program in everything if program.name in names]
