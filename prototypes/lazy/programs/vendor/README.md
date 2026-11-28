# Code copied for the fusion corpus

Programs in this directory are other people's, copied unchanged. [`corpus.py`](../corpus.py) and [`codebases.py`](../codebases.py) supply their input and run each with NumPy and with the lazy stand-in; they do not change a NumPy call.

| File | Is | From | Licence |
|---|---|---|---|
| `npbench/cavity_flow_numpy.py` | Lid-driven cavity flow, step 11 of *CFD Python: the 12 steps to Navier-Stokes* by Lorena Barba and Gilbert Forsyth, as npbench publishes it | <https://github.com/spcl/npbench> at `f2d7f27d87ba4881991e62c18a607aa6ba52b260`, `npbench/benchmarks/cavity_flow/` | BSD 3-Clause, in `npbench/LICENSE`; the file's header has CFD Python's own notice |
| `npbench/hdiff_numpy.py` | Horizontal diffusion, a stencil of the COSMO weather model, adapted by npbench from GridTools' gt4py | The same repository and commit, `npbench/benchmarks/weather_stencils/hdiff/` | BSD 3-Clause, in `npbench/LICENSE` |
| `dpbench/black_scholes_numpy.py` | Black-Scholes pricing of European options, the NumPy implementation of Intel's dpbench. It needs SciPy for the error function | <https://github.com/IntelPython/dpbench> at `56319bb7ee8f142f545d16cbbc2fc2cad156b242`, `dpbench/benchmarks/default/black_scholes/` | Apache 2.0, in `dpbench/LICENSE` |
| `perlin_numpy/perlin2d.py` | Perlin noise and fractal noise in two dimensions, from Pierre Vigier's perlin-numpy | <https://github.com/pvigier/perlin-numpy> at `5e26837db14042e51166eb6cad4c0df2c1907016`, `perlin_numpy/` | MIT, in `perlin_numpy/LICENSE` |
| `dlfs/common/`, `dlfs/ch07/simple_convnet.py` | The layers, optimisers and trainer of Koki Saitoh's book *Deep Learning from Scratch*, and the convolutional network of its chapter 7. Of `common/` the six modules that network's training imports, and the package's empty `__init__.py` | <https://github.com/oreilly-japan/deep-learning-from-scratch> at `07b11618cdb5d5f78ad45464de0730efc07b0a4a`, `common/` and `ch07/` | MIT, in `dlfs/LICENSE.md` |

The inputs are not copied: `corpus.py` builds them as each suite's `initialize` does, a few lines each, and `codebases.py` draws its own in the shapes and element types the two programs are published with.
