"""Ten more programs, each the NumPy code of a published codebase, run as the corpus's programs are.

The corpus of `corpus.py` is this repository's own programs, benchmark
suites and calls into libraries that keep their own `numpy`. These ten are
the NumPy code of other people's projects with the stand-in put where their
`import numpy as np` stood: two copied unchanged into `vendor/`, and eight
that are functions of an installed library, whose modules' `np` is replaced
for the length of the run (`corpus.numpy_of`). So each says what
`hotkernels.install()` and the lazy layer with fusion do for code neither
was written against.

    uv run --group apps --group codebases python -m prototypes.lazy.programs.fusion_time --program codebases

They are not part of `corpus.programs()`: `corpus.by_name` finds them by
name, and takes `codebases` for all ten. What was measured on them is in
docs/lazy-codebases-findings.md.

**Where a library's code is made lazy.** A library's module computes with the
stand-in when its global `np` is the stand-in *and* an operand is a lazy
array: `a - b` on two `ndarray`s is NumPy's, whatever `np` is. So each
program hands its arrays over as the stand-in's (`np.asarray(a)`), and calls
the function that holds the arithmetic. Where a library turns what it is
handed into an `ndarray` with its own `numpy` before computing, the arrays
are plain again from there on; the findings say for which programs.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy

from prototypes.lazy.programs.corpus import LARGE, SEED, SMALL, Program, numpy_of
from prototypes.lazy.programs.vendor.perlin_numpy import perlin2d

# The name `corpus.by_name` takes for all ten programs.
SET = "codebases"
# Where the copied code is.
VENDOR = Path(__file__).resolve().parent / "vendor"

# Fractal noise: (side of the image in pixels, periods of the first octave along a side, octaves) by size. The side
# must be a multiple of the periods of the last octave, periods * 2 ** (octaves - 1).
PERLIN = {SMALL: (256, 4, 4), LARGE: (2048, 4, 5)}

# The copy of *Deep Learning from Scratch*: its directory, the modules of it that compute, and the file of its
# convolutional network, which is not in a package and is loaded by its path.
DLFS = VENDOR / "dlfs"
DLFS_MODULES = ("common.functions", "common.util", "common.layers", "common.optimizer", "common.trainer")
DLFS_NETWORK = ("simple_convnet", DLFS / "ch07" / "simple_convnet.py")
# The convolutional network: (training images, images in a batch) by size. One epoch is trained, so the iterations
# are the first divided by the second.
CONVNET = {SMALL: (100, 10), LARGE: (800, 100)}
# An image of the network's input, as MNIST's are: channels, height, width; and the classes it tells apart.
CONVNET_IMAGE, CONVNET_CLASSES = (1, 28, 28), 10
# Images the network's accuracy is measured on at the start of the epoch and at its end, as a share of a batch.
CONVNET_HELD = 1
# The book's learning rate for Adam in `ch07/train_convnet.py`.
CONVNET_RATE = 0.001

# Hill shading: the side of the elevation model in cells, by size; the size of a cell in metres; how many times the
# relief is exaggerated; the colour map; and the range of heights in metres.
HILLSHADE = {SMALL: 256, LARGE: 2048}
HILLSHADE_CELL, HILLSHADE_EXAGGERATION, HILLSHADE_COLOURS, HILLSHADE_RELIEF = 30.0, 2.0, "gist_earth", 1500.0
# Waves the synthetic terrain is the sum of: (periods across the model, height as a share of the relief).
TERRAIN_WAVES = ((1.5, 0.5), (4.0, 0.25), (9.0, 0.12), (21.0, 0.06))

# Solar position: timestamps by size (a year of hours, two years of minutes), and seconds between two of them.
SOLAR = {SMALL: (8_760, 3_600), LARGE: (1_051_200, 60)}
# Midnight, 1 January 2024 UTC, as seconds since 1970; the site (Golden, Colorado, as in the algorithm's report):
# latitude and longitude in degrees, elevation in metres, pressure in millibars, temperature in Celsius; the
# difference between terrestrial and universal time in seconds; and the atmospheric refraction in degrees.
SOLAR_START = 1_704_067_200
SOLAR_SITE = (39.742476, -105.1786, 1830.14, 820.0, 11.0)
SOLAR_DELTA_T, SOLAR_REFRACTION = 67.0, 0.5667

# Colour difference: the side of the two images in pixels, by size.
DELTA_E = {SMALL: 128, LARGE: 1024}
# How far the second image's pixels are from the first's, as a share of the range of a channel.
DELTA_E_NOISE = 0.05

# Perceptron: (examples, features, hidden units, epochs) by size. 784 features are MNIST's.
MLP = {SMALL: (1_000, 64, 32, 5), LARGE: (10_000, 784, 100, 3)}
MLP_CLASSES, MLP_BATCH = 10, 200

# Sensitivity analysis: the base sample by size, a power of two as Sobol' sequences want. The model is run on
# base * (2 * parameters + 2) points.
SOBOL = {SMALL: 1_024, LARGE: 32_768}
# Resamples for the confidence intervals: SALib's default.
SOBOL_RESAMPLES = 100
# Seed of the resampling. Not `corpus.SEED`: SALib tests `if seed:`, and with 0 draws from NumPy's global generator.
SOBOL_SEED = 1

# Optics: (samples across the pupil, Zernike polynomials summed) by size.
ZERNIKE = {SMALL: (128, 15), LARGE: (1024, 36)}
# The pupil's diameter in millimetres, the wavelength in microns, the focal length in millimetres, and how many times
# the pupil is padded before the transform that gives the image of a point.
PUPIL_DIAMETER, WAVELENGTH, FOCAL_LENGTH, PADDING = 25.0, 0.6328, 250.0, 2
# The largest coefficient of a polynomial, in nanometres of wavefront error.
ZERNIKE_ERROR = 50.0

# Sky image: (side in pixels, sources) by size.
SKY = {SMALL: (128, 6), LARGE: (1024, 12)}

# Speech features: seconds of sound by size, and samples a second.
SPEECH = {SMALL: 10, LARGE: 120}
SPEECH_RATE = 16_000
# The tones the synthetic sound is made of, in hertz, and the loudest sample as a share of an int16's range.
SPEECH_TONES, SPEECH_LEVEL = (220.0, 450.0, 1_300.0, 2_900.0), 0.5
INT16_MAX = 32_767


@contextmanager
def _named(np: Any, name: str, *modules: ModuleType) -> Iterator[None]:
    """Make `np` what some modules call by another name than `np`, for the length of a block.

    Args:
        np: The `numpy` module or the lazy stand-in.
        name: The global the modules hold NumPy under, such as `numpy`.
        *modules: The modules.
    """
    before = [getattr(module, name) for module in modules]
    for module in modules:
        setattr(module, name, np)
    try:
        yield
    finally:
        for module, original in zip(modules, before):
            setattr(module, name, original)


def _run_perlin(np: Any, inputs: tuple) -> Any:
    """Generate fractal noise. The gradients are drawn from NumPy's global generator, seeded here."""
    side, periods, octaves = inputs
    numpy.random.seed(SEED)
    with numpy_of(np, perlin2d):
        return perlin2d.generate_fractal_noise_2d((side, side), (periods, periods), octaves)


def _dlfs() -> list[ModuleType]:
    """Import the copied modules of *Deep Learning from Scratch*.

    The book's files import each other as `common.layers`, from a script's directory, and each appends the parent
    directory to `sys.path`. So the copy's directory is put on the path while they are imported, and the entries they
    appended are taken off again.

    Returns:
        The modules that compute, the network's last.
    """
    path = list(sys.path)
    sys.path.insert(0, str(DLFS))
    try:
        modules = [importlib.import_module(name) for name in DLFS_MODULES]
        name, file = DLFS_NETWORK
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, file)
            assert spec is not None and spec.loader is not None
            sys.modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sys.modules[name])
        return modules + [sys.modules[name]]
    finally:
        sys.path[:] = path


def _convnet_data(size: str) -> tuple:
    """Draw images and labels in MNIST's shapes and element types, as the book's `load_mnist` gives them.

    Args:
        size: SMALL or LARGE.

    Returns:
        (training images, their labels, held-out images, their labels, images in a batch).
    """
    examples, batch = CONVNET[size]
    held = CONVNET_HELD * batch
    rng = numpy.random.default_rng(SEED)
    images = rng.random((examples + held, *CONVNET_IMAGE), dtype=numpy.float32)
    labels = rng.integers(0, CONVNET_CLASSES, examples + held)
    return images[:examples], labels[:examples], images[examples:], labels[examples:], batch


def _run_convnet(np: Any, inputs: tuple) -> Any:
    """Train the book's convolutional network for one epoch with its trainer and Adam."""
    images, labels, held_images, held_labels, batch = inputs
    modules = _dlfs()
    numpy.random.seed(SEED)
    with numpy_of(np, *modules):
        network = modules[-1].SimpleConvNet(input_dim=CONVNET_IMAGE, output_size=CONVNET_CLASSES)
        trainer = sys.modules["common.trainer"].Trainer(
            network, np.asarray(images), np.asarray(labels), np.asarray(held_images), np.asarray(held_labels),
            epochs=1, mini_batch_size=batch, optimizer="Adam", optimizer_param={"lr": CONVNET_RATE},
            evaluate_sample_num_per_epoch=batch, verbose=False)
        trainer.train()
    return [network.params[name] for name in sorted(network.params)], trainer.train_loss_list


def _terrain(size: str) -> tuple:
    """Build an elevation model: a few waves and a little noise.

    Args:
        size: SMALL or LARGE.

    Returns:
        (heights in metres, the size of a cell in metres).
    """
    side = HILLSHADE[size]
    rng = numpy.random.default_rng(SEED)
    row, column = numpy.mgrid[0:side, 0:side] / side
    heights = sum(share * numpy.sin(2 * numpy.pi * periods * (row + phase)) * numpy.cos(2 * numpy.pi * periods * column)
                  for (periods, share), phase in zip(TERRAIN_WAVES, rng.random(len(TERRAIN_WAVES))))
    return HILLSHADE_RELIEF * (heights + 0.01 * rng.standard_normal((side, side))), HILLSHADE_CELL


def _run_hillshade(np: Any, inputs: tuple) -> Any:
    """Shade an elevation model with Matplotlib's `LightSource`, coloured by height.

    The model is handed over as the stand-in's array, like every other program's input.
    `LightSource.hillshade` makes its array of normals with `np.empty(...).view(type(elevation))`, to keep a masked
    array masked; on a lazy array that is a view by the array's own type, which the layer takes for `x.view()`.
    """
    import matplotlib
    from matplotlib import colors
    heights, cell = inputs
    with numpy_of(np, colors):
        return colors.LightSource(azdeg=315, altdeg=45).shade(
            np.asarray(heights), cmap=matplotlib.colormaps[HILLSHADE_COLOURS], blend_mode="overlay",
            vert_exag=HILLSHADE_EXAGGERATION, dx=cell, dy=cell)


def _solar_data(size: str) -> tuple:
    """List the moments the sun's position is wanted at.

    Args:
        size: SMALL or LARGE.

    Returns:
        (seconds since 1970 as float64, as pvlib passes them).
    """
    count, step = SOLAR[size]
    return (SOLAR_START + step * numpy.arange(count, dtype=numpy.float64),)


def _run_solar(np: Any, inputs: tuple) -> Any:
    """Compute the sun's position with pvlib's NumPy implementation of NREL's algorithm."""
    from pvlib import spa
    (moments,) = inputs
    with numpy_of(np, spa):
        return spa.solar_position_numpy(np.asarray(moments), *SOLAR_SITE, SOLAR_DELTA_T, SOLAR_REFRACTION, numthreads=1)


def _delta_e_data(size: str) -> tuple:
    """Draw an image and a copy of it with noise, both RGB in float64 between 0 and 1.

    Args:
        size: SMALL or LARGE.

    Returns:
        (the image, the noisy copy).
    """
    side = DELTA_E[size]
    rng = numpy.random.default_rng(SEED)
    image = rng.random((side, side, 3))
    return image, numpy.clip(image + DELTA_E_NOISE * rng.standard_normal(image.shape), 0.0, 1.0)


def _run_delta_e(np: Any, inputs: tuple) -> Any:
    """Convert two images to CIELAB and take the CIEDE2000 difference of each pixel, with scikit-image."""
    from skimage.color import colorconv, delta_e
    first, second = inputs
    with numpy_of(np, colorconv, delta_e):
        return delta_e.deltaE_ciede2000(colorconv.rgb2lab(np.asarray(first)), colorconv.rgb2lab(np.asarray(second)))


def _mlp_data(size: str) -> tuple:
    """Draw a classification problem.

    Args:
        size: SMALL or LARGE.

    Returns:
        (examples, their classes, hidden units, epochs).
    """
    examples, features, hidden, epochs = MLP[size]
    rng = numpy.random.default_rng(SEED)
    x = rng.random((examples, features))
    return x, (x @ rng.standard_normal((features, MLP_CLASSES))).argmax(axis=1), hidden, epochs


def _run_mlp(np: Any, inputs: tuple) -> Any:
    """Train scikit-learn's multi-layer perceptron with Adam for a few epochs."""
    from sklearn.neural_network import MLPClassifier, _base, _multilayer_perceptron, _stochastic_optimizers
    x, y, hidden, epochs = inputs
    with numpy_of(np, _base, _multilayer_perceptron, _stochastic_optimizers):
        model = MLPClassifier(hidden_layer_sizes=(hidden,), solver="adam", batch_size=MLP_BATCH, max_iter=epochs,
                              random_state=SEED).fit(np.asarray(x), np.asarray(y))
    return list(model.coefs_), list(model.intercepts_), model.loss_


def _sobol_data(size: str) -> tuple:
    """Sample SALib's Ishigami function as its documentation does.

    Args:
        size: SMALL or LARGE.

    Returns:
        (the problem, the function's value at each point of the sample).
    """
    from SALib.sample import sobol as sample
    from SALib.test_functions import Ishigami
    problem = {"num_vars": 3, "names": ["x1", "x2", "x3"], "bounds": [[-numpy.pi, numpy.pi]] * 3}
    return problem, Ishigami.evaluate(sample.sample(problem, SOBOL[size], seed=SEED))


def _run_sobol(np: Any, inputs: tuple) -> Any:
    """Compute Sobol' sensitivity indices and their confidence intervals with SALib."""
    from SALib.analyze import sobol
    problem, values = inputs
    with numpy_of(np, sobol):
        indices = sobol.analyze(problem, np.asarray(values), num_resamples=SOBOL_RESAMPLES, seed=SOBOL_SEED)
    return [indices[name] for name in ("S1", "S1_conf", "ST", "ST_conf", "S2", "S2_conf")]


def _zernike_data(size: str) -> tuple:
    """Draw a wavefront's Zernike coefficients.

    Args:
        size: SMALL or LARGE.

    Returns:
        (samples across the pupil, the coefficients in nanometres, one for each polynomial).
    """
    samples, modes = ZERNIKE[size]
    return samples, numpy.random.default_rng(SEED).uniform(-ZERNIKE_ERROR, ZERNIKE_ERROR, modes)


def _run_zernike(np: Any, inputs: tuple) -> Any:
    """Build a wavefront from Zernike polynomials with prysm and take the image of a point through it.

    prysm computes with `prysm.mathops.np`, an object made for exchanging NumPy for another library: the stand-in is
    put into it, which is the library's own one line.
    """
    from prysm import coordinates, geometry, mathops, polynomials, propagation
    samples, coefficients = inputs
    before, mathops.np._srcmodule = mathops.np._srcmodule, np
    try:
        x, y = coordinates.make_xy_grid(samples, diameter=PUPIL_DIAMETER)
        radius, angle = coordinates.cart_to_polar(x, y)
        orders = [polynomials.noll_to_nm(index) for index in range(1, len(coefficients) + 1)]
        basis = list(polynomials.zernike_nm_sequence(orders, radius / (PUPIL_DIAMETER / 2), angle))
        phase = polynomials.sum_of_2d_modes(basis, coefficients)
        pupil = propagation.Wavefront.from_amp_and_phase(geometry.circle(PUPIL_DIAMETER / 2, radius), phase,
                                                         WAVELENGTH, PUPIL_DIAMETER / samples)
        return phase, pupil.focus(FOCAL_LENGTH, Q=PADDING).intensity.data
    finally:
        mathops.np._srcmodule = before


def _sky_data(size: str) -> tuple:
    """Draw the sources of a sky image: stars as Gaussians and Moffat profiles, galaxies as Sérsic profiles.

    Args:
        size: SMALL or LARGE.

    Returns:
        (side of the image in pixels, a list of (the model's name, the arguments of its `evaluate` after x and y)).
    """
    side, count = SKY[size]
    rng = numpy.random.default_rng(SEED)
    sources = []
    for index in range(count):
        x, y, flux = float(rng.uniform(0, side)), float(rng.uniform(0, side)), float(rng.uniform(1, 100))
        width = float(rng.uniform(side / 64, side / 16))
        tilt = float(rng.uniform(0, numpy.pi))
        sources.append([("Gaussian2D", (flux, x, y, width, 0.6 * width, tilt)),
                        ("Moffat2D", (flux, x, y, width, 2.5)),
                        ("Sersic2D", (flux, 2 * width, float(rng.uniform(1, 4)), x, y, 0.4, tilt))][index % 3])
    return side, sources


def _run_sky(np: Any, inputs: tuple) -> Any:
    """Render sources onto an image with the formulas of Astropy's models, each evaluated over the whole image."""
    from astropy.modeling import functional_models
    side, sources = inputs
    with numpy_of(np, functional_models):
        y, x = np.mgrid[0:side, 0:side]
        image = np.zeros((side, side))
        for name, arguments in sources:
            image += getattr(functional_models, name).evaluate(x, y, *arguments)
        return image


def _speech_data(size: str) -> tuple:
    """Make a sound as a WAV file holds it: 16-bit samples.

    Args:
        size: SMALL or LARGE.

    Returns:
        (the samples as int16,).
    """
    rng = numpy.random.default_rng(SEED)
    moments = numpy.arange(SPEECH[size] * SPEECH_RATE) / SPEECH_RATE
    sound = sum(numpy.sin(2 * numpy.pi * tone * moments) for tone in SPEECH_TONES) / len(SPEECH_TONES)
    sound = 0.8 * sound + 0.2 * rng.standard_normal(moments.size)
    return ((SPEECH_LEVEL * INT16_MAX * sound / numpy.abs(sound).max()).astype(numpy.int16),)


def _run_speech(np: Any, inputs: tuple) -> Any:
    """Compute mel-frequency cepstral coefficients with python_speech_features, which calls NumPy `numpy`."""
    from python_speech_features import base, sigproc
    (sound,) = inputs
    with _named(np, "numpy", base, sigproc):
        return base.mfcc(np.asarray(sound), SPEECH_RATE)


def programs() -> list[Program]:
    """Return the ten programs, in the order their tables are written."""
    vendor = "prototypes/lazy/programs/vendor/"
    return [
        Program("perlin_noise", "perlin-numpy: fractal noise of several octaves", vendor + "perlin_numpy/perlin2d.py",
                lambda size: PERLIN[size], _run_perlin,
                {size: f"an image of {side:,} × {side:,}, {octaves} octaves" for size, (side, _, octaves) in PERLIN.items()}),
        Program("dlfs_convnet", "Deep Learning from Scratch: a convolutional network trained with Adam",
                vendor + "dlfs/", _convnet_data, _run_convnet,
                {size: f"{examples // batch} batches of {batch} images of 28 × 28"
                 for size, (examples, batch) in CONVNET.items()}),
        Program("mpl_hillshade", "Matplotlib: an elevation model shaded and coloured", "matplotlib.colors",
                _terrain, _run_hillshade, {size: f"{side:,} × {side:,} cells" for size, side in HILLSHADE.items()},
                requires=("matplotlib",)),
        Program("pvlib_solar", "pvlib: the sun's position by NREL's algorithm", "pvlib.spa",
                _solar_data, _run_solar, {size: f"{count:,} moments" for size, (count, _) in SOLAR.items()},
                requires=("pvlib",)),
        Program("skimage_delta_e", "scikit-image: RGB to CIELAB and the CIEDE2000 colour difference",
                "skimage.color.colorconv, skimage.color.delta_e", _delta_e_data, _run_delta_e,
                {size: f"two images of {side:,} × {side:,}" for size, side in DELTA_E.items()}, requires=("skimage",)),
        Program("sklearn_mlp", "scikit-learn: a multi-layer perceptron trained with Adam", "sklearn.neural_network",
                _mlp_data, _run_mlp,
                {size: f"{examples:,} examples of {features} features, {hidden} hidden units, {epochs} epochs"
                 for size, (examples, features, hidden, epochs) in MLP.items()}, requires=("sklearn",)),
        Program("salib_sobol", "SALib: Sobol' sensitivity indices with bootstrap intervals", "SALib.analyze.sobol",
                _sobol_data, _run_sobol,
                {size: f"a base sample of {base:,}, {SOBOL_RESAMPLES} resamples" for size, base in SOBOL.items()},
                requires=("SALib",)),
        Program("prysm_zernike", "prysm: a wavefront of Zernike polynomials and the image of a point", "prysm",
                _zernike_data, _run_zernike,
                {size: f"a pupil of {samples:,} × {samples:,}, {modes} polynomials"
                 for size, (samples, modes) in ZERNIKE.items()}, requires=("prysm",)),
        Program("astropy_models", "Astropy: stars and galaxies rendered from its models' formulas",
                "astropy.modeling.functional_models", _sky_data, _run_sky,
                {size: f"an image of {side:,} × {side:,}, {count} sources" for size, (side, count) in SKY.items()},
                requires=("astropy", "scipy")),
        Program("speech_mfcc", "python_speech_features: mel-frequency cepstral coefficients",
                "python_speech_features.base", _speech_data, _run_speech,
                {size: f"{seconds} s of sound at {SPEECH_RATE:,} samples a second" for size, seconds in SPEECH.items()},
                requires=("python_speech_features", "scipy")),
    ]
