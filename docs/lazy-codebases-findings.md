# Ten programs from other codebases, with `install()` and with fusion

The corpus fusion was built on ([lazy-fusion-design.md](lazy-fusion-design.md#the-corpus)) is this repository's own programs, benchmark suites, and calls into libraries that keep their own `numpy`. This document is about ten more programs, each the NumPy code of a published project that was written against neither the package nor the lazy layer, and about one question: what do `hotkernels.install()` and the lazy layer with fusion do for them, and where they do nothing or cost time, why.

Nothing here is in the package. The programs are in [`prototypes/lazy/programs/codebases.py`](../prototypes/lazy/programs/codebases.py) and are run by the corpus's tools.

## Summary

- **All ten give NumPy's results bit for bit,** node by node and fused, with the kernels installed and without. One library had to be handed a plain `ndarray`, and one seed had to be other than zero ([What got in the way of running them](#what-got-in-the-way-of-running-them)).
- **Fusion gains on three, by 1.08x to 1.14x, is level on five, and loses on two.** Against plain NumPy at the large size: Astropy's models 1.14x, prysm's Zernike wavefront 1.10x, scikit-image's colour difference 1.08x; the book's convolutional network 1.07x, which is `install()`'s gain and not fusion's; SALib 1.02x; perlin-numpy, Matplotlib's hill shading and the speech features 1.00x; pvlib's solar position 0.97x; scikit-learn's perceptron 0.88x. Nothing like the 2x to 3x of the corpus's stencils.
- **`install()` alone gains on one of the ten,** the convolutional network, 1.08x, from its kernels for `x[mask] = 0` and for `argmax` and sums along rows. On the other nine it is within 2% of NumPy either way: their time is not in calls the package has kernels for.
- **A third of all the time in the ten programs is a mathematics library's functions:** `cos`, `sin`, `x ** 3`, `x ** 2.4`, `%`, `cbrt`, `hypot`, `arctan2`, `exp`. 2.6 s of 7.9 s. No kernel of either kind may touch them, because the result has to be the library's bits. This is the largest thing the ten programs have that the corpus did not show at this weight.
- **The float64 arithmetic that fusion is for is 30% of the time, and a fifth of that was found and then given up for how the arrays lie.** An array left transposed by the call that made it, 300 ms; rows of four colour values with one value for each pixel, 100 ms; and arrays that step, a channel of an image (`lab[..., 0]`), a column of a table (`AB[:, j]`), every eighth element (`Y[::8]`), 30 ms in three programs. Each is how these codebases hold their data, and none is a missing pattern.
- **A lazy run computed node by node is slower than the eager program on six of the ten,** by up to 1.4x, before fusion is counted. NumPy computes `a * (1 - t) + t * b` reusing the memory of its temporaries; computed with every intermediate held, as a graph's are, the same line takes 1.4 to 1.7 times as long. Fusion wins that back and no more, which is why several programs are level.
- **Calls through the stand-in have slow roads the corpus did not meet:** a ufunc called with `out=`, `np.clip`, a function called on plain `ndarray`s, and `plain += lazy` cost 1.1 to 8.1 µs over NumPy's own call, where the same calls on the layer's arrays cost 0.05 to 0.2 µs. scikit-learn's perceptron makes 7,000 of them and is 0.88x for it.
- **The timing tool's order was worth several percent, and the tools time in turn now.** Timed one way after another, the way timed first, NumPy's, was the slowest whatever it was: pvlib ran 10% "faster installed" with no kernel serving a single call. `fusion_time` and `limit` time the ways in turn, by default since [issue 323](#issue-323-the-ways-timed-in-turn-by-default). Every table of the four ways below is taken that way.

## The programs

Each is run with `import numpy as np`, or with the stand-in for it put where that import stood: for the two copied programs and the eight library functions alike, the global `np` of the modules that compute (`corpus.numpy_of`). Inputs are made once with the real `numpy`, in the shapes and element types the project's own documentation uses, and handed over as the stand-in's arrays.

| Program | Is | From | What runs |
|---|---|---|---|
| `perlin_noise` | Fractal noise of five octaves | [perlin-numpy](https://github.com/pvigier/perlin-numpy), MIT, copied into `vendor/` | `generate_fractal_noise_2d` |
| `dlfs_convnet` | A convolutional network trained for one epoch with Adam | [*Deep Learning from Scratch*](https://github.com/oreilly-japan/deep-learning-from-scratch), MIT, copied into `vendor/` | `SimpleConvNet` of chapter 7 under the book's `Trainer`, on images of MNIST's shape |
| `mpl_hillshade` | An elevation model shaded and coloured | Matplotlib 3.11, `matplotlib.colors` | `LightSource.shade`, overlay blending |
| `pvlib_solar` | The sun's position by NREL's algorithm | pvlib 0.16, `pvlib.spa` | `solar_position_numpy` |
| `skimage_delta_e` | RGB to CIELAB and the CIEDE2000 difference of two images | scikit-image 0.26, `skimage.color` | `rgb2lab`, `deltaE_ciede2000` |
| `sklearn_mlp` | A perceptron of one hidden layer trained with Adam | scikit-learn 1.9, `sklearn.neural_network` | `MLPClassifier.fit` |
| `salib_sobol` | Sobol' sensitivity indices with bootstrap intervals | SALib 1.6, `SALib.analyze.sobol` | `analyze` on the Ishigami function |
| `prysm_zernike` | A wavefront of 36 Zernike polynomials and the image of a point through it | prysm 0.21, through `prysm.mathops.np` | `zernike_nm_sequence`, `sum_of_2d_modes`, `Wavefront.focus` |
| `astropy_models` | Twelve stars and galaxies rendered onto an image | Astropy 8.0, `astropy.modeling.functional_models` | `Gaussian2D.evaluate`, `Moffat2D.evaluate`, `Sersic2D.evaluate` |
| `speech_mfcc` | Mel-frequency cepstral coefficients of two minutes of sound | python_speech_features 0.6 | `mfcc` on 16-bit samples |

- **They were chosen for what they are, not for what they would show:** real projects whose arithmetic is NumPy written in Python, of ten different kinds (noise, a network from a textbook, plotting, solar energy, colour, a library's own network, sensitivity analysis, optics, astronomy, speech), with a licence that lets two be copied and the other eight installed. None is a stencil; the corpus has those.
- **prysm is the one made for it.** It computes with `prysm.mathops.np`, an object whose source module is meant to be exchanged for CuPy; the stand-in is put there, which is the library's own one line.
- **A library's code is lazy only from where an operand is lazy.** `a - b` on two `ndarray`s is NumPy's whatever `np` is. So each program hands its arrays over as the stand-in's and calls the function that holds the arithmetic: Astropy's `evaluate` and not the model's `__call__`, which makes plain arrays of its inputs with its own `numpy` first. scikit-learn validates what it is given the same way, so its perceptron computes on plain arrays, and only what its three modules' own `np` calls return is the layer's.

## How they were run

```bash
uv run --group apps --group codebases python -m prototypes.lazy.build_core
uv run --group apps --group codebases python -m prototypes.lazy.build_fused
OPENBLAS_NUM_THREADS=1 taskset -c 6 uv run --group apps --group codebases python -m prototypes.lazy.programs.fusion_time --program codebases --size large --plain --runs 7
OPENBLAS_NUM_THREADS=1 taskset -c 6 uv run --group apps --group codebases python -m prototypes.lazy.programs.mine --program codebases --size large
OPENBLAS_NUM_THREADS=1 taskset -c 6 uv run --group apps --group codebases python -m prototypes.lazy.programs.limit --program codebases --size large
```

`--program codebases` names all ten; each also goes by its own name. They are not part of `corpus.programs()`, so the corpus's tables and counts are what they were.

Everything is from an Intel Core i5-14600KF under Linux, NumPy 2.5.5, CPython 3.13, the layer's nodes written by its C core, the fused kernels as [the design's section on x86-64](lazy-fusion-design.md#another-processor) has them.

- **One performance core, and one thread for the matrix products.** Another session was using the machine while these were taken. Unpinned, with OpenBLAS on every core, scikit-learn's perceptron took anything from 223 ms to 1,267 ms and the convolutional network 842 ms to 1,544 ms from one run of the table to the next. Pinned to one core with one BLAS thread they repeat to about 2%. So the two programs that are matrix products are timed with those products on one thread, which makes the products a larger share of them than on an idle machine with all cores.
- **The ways in turn.** See [The order of timing](#the-order-of-timing).

## What was measured

At the large size, the quickest of 7 rounds, milliseconds. The columns are those of [the design's table](lazy-fusion-design.md#the-programs-four-ways): **NumPy** is the program as it is; **installed** the same eager program under `hotkernels.install()`; **lazy** the stand-in with every node computed by the call the program wrote; **fused** the same with `fuse.enable()`. The last three are with the kernels installed.

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
perlin_noise         1962.70   1966.52    2276.79   328.17  1948.61    1970.47   622.39     0.30  1347.78    1.16x    1.00x     1.00x
dlfs_convnet          792.00    733.20     708.21    97.01   611.19     742.91   248.41     1.73   492.77    0.95x    1.07x     0.99x
mpl_hillshade         279.84    277.64     280.90   109.89   171.01     279.30   133.01     0.11   146.19    1.01x    1.00x     0.99x
pvlib_solar          2646.48   2659.58    2778.97     2.17  2776.80    2715.62     2.36    10.95  2702.31    1.02x    0.97x     0.98x
skimage_delta_e       382.59    379.39     377.60    44.28   333.32     355.08    62.49     0.51   292.08    1.06x    1.08x     1.07x
sklearn_mlp           223.01    218.54     271.75   226.63    45.12     252.25   222.47     2.37    27.40    1.08x    0.88x     0.87x
salib_sobol           997.14    992.14    1413.45   533.93   879.52     976.04   895.05     0.59    80.40    1.45x    1.02x     1.02x
prysm_zernike         347.99    351.88     364.02   116.16   247.85     316.92   137.84     0.14   178.94    1.15x    1.10x     1.11x
astropy_models        192.26    188.42     212.61     4.77   207.84     169.33    46.50     0.13   122.69    1.26x    1.14x     1.11x
speech_mfcc            60.37     60.34      80.13    41.83    38.31      60.43    53.37     0.06     7.00    1.33x    1.00x     1.00x
```

The lazy and the fused run with nothing installed, against the same two eager runs:

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
perlin_noise         1962.70   1966.52    2277.63   331.42  1946.21    1965.01   621.45     0.31  1343.25    1.16x    1.00x     1.00x
dlfs_convnet          792.00    733.20     728.77    97.43   631.34     767.60   251.17     1.74   514.69    0.95x    1.03x     0.96x
mpl_hillshade         279.84    277.64     283.19   110.76   172.43     280.99   133.34     0.11   147.54    1.01x    1.00x     0.99x
pvlib_solar          2646.48   2659.58    2771.65     2.12  2769.53    2709.67     2.25    10.96  2696.46    1.02x    0.98x     0.98x
skimage_delta_e       382.59    379.39     378.70    47.64   331.06     358.14    65.84     0.50   291.80    1.06x    1.07x     1.06x
sklearn_mlp           223.01    218.54     277.34   231.17    46.18     257.84   227.89     2.40    27.55    1.08x    0.86x     0.85x
salib_sobol           997.14    992.14    1408.80   530.71   878.09     985.18   904.10     0.60    80.49    1.43x    1.01x     1.01x
prysm_zernike         347.99    351.88     363.91   116.11   247.80     317.91   138.61     0.15   179.16    1.14x    1.09x     1.11x
astropy_models        192.26    188.42     215.04     5.55   209.49     170.11    47.60     0.13   122.38    1.26x    1.13x     1.11x
speech_mfcc            60.37     60.34      86.50    47.56    38.94      66.17    58.05     0.06     8.06    1.31x    0.91x     0.91x
```

What the graphs were, and what the fused run did with them:

```text
program            deferred  graphs  largest largest node  deferred at once  expr.  tiled  kernel  fused  given  writes  writes    lazy MB  fused MB
                        ops            graph    elements     fused   fused                 calls  nodes     up in place  copied    written   written
perlin_noise            285      26      200      8388608       200      25     26     26      60     74      5       0       0     7717.7    5033.2
dlfs_convnet           3419      55      191      1875000      1135    1368     24     24      72     72      0     677       0     1228.1     919.2
mpl_hillshade           145      20       26     16777216        19      79      4      4       3      1      2       1       2     1073.9     973.1
pvlib_solar            2520       1     2520      2102400      2520       0    290    290     684    638      0     250       0    18532.7   13663.5
skimage_delta_e         184       7      138      3145728       168       2     21     21      56     58      4      14       0     1129.9     643.4
sklearn_mlp            8274     461       53        78400      1200    6921    450    450     750    750      0       0       0      764.2     282.2
salib_sobol             309      91       19      3276800       113     152     31     31      19     19     15       6       0     1900.3     404.0
prysm_zernike           204       4      191      4194304       194      10     45     45      86     93      0       0       0     1653.6     805.3
astropy_models          198       1      198      1048576       164      32     17     17      96    127      0       0       0     1543.5     302.0
speech_mfcc            1975       6     1951      4799600         6    1963      2      2       3      1      0       1       0      229.3      64.4
```

Forty of the eighty fused kernels ran, `fuse_add_scaled_f64` (`a + s * b`) the most, 402 times, then the single operations `s + a` (367), `s * a` (288) and `np.sqrt(a)` (163).

At the small size, the quickest of 15 rounds, where the arrays are a sixteenth to a hundredth as large:

```text
program             NumPy ms installed    lazy ms    build  compute   fused ms    build     plan  compute  vs lazy vs NumPy  vs inst.
perlin_noise           21.74     21.74      24.81     6.17    18.64      23.68     9.94     0.17    13.57    1.05x    0.92x     0.92x
dlfs_convnet          113.94    106.56     154.37    28.04   126.33     107.04    40.52     1.56    64.96    1.44x    1.06x     1.00x
mpl_hillshade           4.69      4.26       5.01     2.59     2.42       4.94     2.99     0.07     1.88    1.02x    0.95x     0.86x
pvlib_solar            19.72     19.40      21.85     2.11    19.74      19.84    19.84     0.00     0.00    1.10x    0.99x     0.98x
skimage_delta_e         5.36      5.30       6.21     1.41     4.79       5.91     4.04     0.05     1.81    1.05x    0.91x     0.90x
sklearn_mlp             5.70      4.87      13.01    10.83     2.17       8.71     8.71     0.00     0.00    1.49x    0.65x     0.56x
salib_sobol            14.45     14.45      31.79    19.49    12.29      17.41    15.61     0.15     1.64    1.83x    0.83x     0.83x
prysm_zernike           2.69      2.70       3.07     1.46     1.61       2.92     2.92     0.00     0.00    1.05x    0.92x     0.93x
astropy_models          1.24      1.27       1.43     0.31     1.12       1.29     1.29     0.00     0.00    1.11x    0.97x     0.99x
speech_mfcc             7.01      7.04      14.25     8.46     5.79       8.59     7.73     0.02     0.84    1.66x    0.82x     0.82x
```

There the fused run is behind NumPy on nine of the ten, at 0.65x to 0.99x, which is what [the design found for small arrays](lazy-fusion-design.md#small-arrays-the-graph-costs-more-than-the-program): a call made at once costs more than NumPy's own, and nothing is large enough to plan for. `install()` gains on three there: the perceptron 1.17x, the hill shading 1.10x, the convolutional network 1.07x.

### Where NumPy's time goes

From `mine`, which computes each graph one node at a time and times every call. Shares of the time spent computing nodes:

| Program | Nodes, ms | In expressions | Simple, alone | A library's functions, other types | Matrix products | Reductions | Writes | Indexing | The rest |
|---|---|---|---|---|---|---|---|---|---|
| `perlin_noise` | 1,983 | 43.3% | 3.9% | 14.8% | | 38.0% | | | |
| `dlfs_convnet` | 655 | 10.8% | 5.7% | 4.2% | 18.4% | 31.4% | 21.9% | 0.1% | 7.7% |
| `mpl_hillshade` | 171 | 80.8% | 3.2% | 5.2% | | 1.4% | 9.3% | | |
| `pvlib_solar` | 3,008 | 30.9% | 5.7% | 63.4% | | | | | |
| `skimage_delta_e` | 341 | 20.7% | 2.1% | 72.7% | | | 1.9% | 2.6% | |
| `sklearn_mlp` | 44 | 78.8% | 4.8% | 3.0% | 5.1% | 7.3% | | 0.1% | 0.8% |
| `salib_sobol` | 905 | 14.4% | | | | 30.5% | 0.1% | 55.1% | |
| `prysm_zernike` | 289 | 36.0% | 7.3% | 55.6% | | | 1.0% | | |
| `astropy_models` | 216 | 38.6% | 11.7% | 49.7% | | | | | |
| `speech_mfcc` | 44 | 11.2% | 27.2% | 24.8% | 9.3% | 3.0% | 1.5% | 22.9% | |

The nodes are not the whole program: the hill shading spends 109 ms of its 280 ms outside every graph, in calls the layer does not defer (`np.gradient`, `ndarray.take`, `ndarray.dot`), the perceptron 183 ms of 223 ms (the library's matrix products on plain arrays), prysm 133 ms of 348 ms (the Fourier transform), SALib 157 ms, the convolutional network 129 ms, the speech features 26 ms of 60 ms.

The heaviest element-by-element operations that are in no expression, over the ten programs:

| Operation | ms | In |
|---|---|---|
| `np.cos` | 1,311 | `pvlib_solar` 1,229, `skimage_delta_e` 45, `prysm_zernike` 37 |
| `np.sin` | 404 | `pvlib_solar` 339, `prysm_zernike` 55, `skimage_delta_e` 10 |
| `%` | 350 | `perlin_noise` 294 (`grid % 1`), `pvlib_solar` 56 (`x % 360.0`) |
| `**`, `np.power` | 336 | `pvlib_solar` 158 (`x ** 3`, `x ** 4`), `astropy_models` 84 (`z ** (1 / n)`, `x ** -alpha`), `skimage_delta_e` 57 (`x ** 2.4`, `x ** 7`), `prysm_zernike` 37 (`r ** m`) |
| `np.radians`, `np.degrees` | 81 | `pvlib_solar` |
| `np.arctan2` | 67 | `skimage_delta_e` 31, `pvlib_solar` 28, `prysm_zernike` 8 |
| `np.cbrt` | 55 | `skimage_delta_e` |
| `np.hypot` | 43 | `skimage_delta_e` 39, `prysm_zernike` 4 |
| `np.exp` | 42 | `astropy_models` 23, `prysm_zernike` 17 (complex), `skimage_delta_e` 2 |
| `+` on int64 | 10 | `speech_mfcc`, the table of positions it frames the sound by |

### What `install()` served

The package's own count of calls, for one run of each program: taken by a kernel, and those that reached a hook and were left to NumPy. `ufunc lane` is the package's loops for arithmetic on arrays of a few elements.

| Program | Taken by a kernel, eagerly | Left to NumPy at a hook | Eager gain |
|---|---|---|---|
| `perlin_noise` | 15 on the ufunc lane | `np.sum` 20 (along an axis of two elements), ufuncs 180 | 1.00x |
| `dlfs_convnet` | `x[mask] = 0` 27, `argmax` along rows 19, `np.sum` 19, `argmax` 3; 579 on the ufunc lane | `x[mask] = 0` 27 (a mask that is not contiguous), `np.sum` 40 (along an axis), `astype` 38, ufuncs 453 | 1.08x |
| `mpl_hillshade` | `np.clip` 4, `x[mask] = v` 4, `np.searchsorted` 3, `np.where` 1; 48 on the ufunc lane | `astype` 4, ufuncs 54 | 1.01x |
| `pvlib_solar` | none | ufuncs 2,531 | 1.00x |
| `skimage_delta_e` | `x[mask]` 13, `x[mask] = v` 1; 5 on the ufunc lane | `x[mask] = v` 13, `np.where` 2, `astype` 4, ufuncs 152 | 1.01x |
| `sklearn_mlp` | `np.clip` 150, `x[mask] = 0` 150, `np.sum` 150, `np.unique` 4, `x[mask]` 2, `np.searchsorted` 1; 6,459 on the ufunc lane | `np.sum` 451 (along an axis), ufuncs 3,468 | 1.02x |
| `salib_sobol` | 33 on the ufunc lane | `astype` 3, ufuncs 341 | 1.01x |
| `prysm_zernike` | `np.unique` 1; 2 on the ufunc lane | ufuncs 311 | 0.99x |
| `astropy_models` | none | ufuncs 220 | 1.02x |
| `speech_mfcc` | `np.where` 2; 14 on the ufunc lane | `np.sum` 1, ufuncs 15 | 1.00x |

- **Nine of the ten make almost no call the package has a kernel for.** Their work is arithmetic and a mathematics library's functions on large arrays, which the package leaves to NumPy because NumPy's own loops are at the speed of memory there ([issue 239](https://github.com/sofeikov/hotkernels/issues/239)), and reductions along an axis, indexing by integers and Fourier transforms, for which it has no kernel.
- **The fused run loses none of these kernels.** Counted the same way under fusion, each program's calls reach the same kernels. The convolutional network's count of `argmax` along rows falls from 19 to 8, and that is not a loss: see the program below.

## Program by program

### `astropy_models`, 1.14x

Twelve sources, each a formula over a million pixels. A Gaussian is `amplitude * np.exp(-(a * xdiff**2 + b * xdiff * ydiff + c * ydiff**2))`, and everything inside the exponential is one expression of nine operations; the Sérsic profile's radius is one of fourteen. 17 expressions are planned and 127 of the 198 nodes are computed inside a kernel's program: the new arrays written fall from 1.5 GB to 0.3 GB, and the nodes from 208 ms to 123 ms. What is left is `z ** (1 / n)` and `x ** -alpha`, 84 ms, and `np.exp`, 23 ms: half the nodes' time, the library's. The grid is `np.mgrid[0:side, 0:side]`, whole numbers as in Astropy's documentation, so the first subtraction of each source is on int64 and is NumPy's: 28 ms.

### `prysm_zernike`, 1.10x

The radial polynomials are a recurrence, `Pn = (a * x + b) * Pnm1 - c * Pnm2`, and every step of it is a kernel's shape; each polynomial is then `jac * azpiece * radialpiece`, a product of three arrays, 32 times. 45 expressions, all planned, none given up. Outside them are `np.sin(m * t)` and `np.cos(m * t)`, 92 ms, `r ** m`, 37 ms, and the complex arithmetic of the pupil, 20 ms; and outside the graph, 133 ms, the Fourier transform among it.

### `skimage_delta_e`, 1.08x

The CIEDE2000 formula is long, and its two largest expressions, of 24 and 16 operations, are each one program. But 73% of the nodes' time is `np.power(x, 2.4)` for the sRGB curve, `np.cbrt`, `np.cos`, `np.hypot` and `np.arctan2`. Four of 21 programs are given up: `a = 500.0 * (x - y)` and `b = 200.0 * (y - z)`, where `x`, `y` and `z` are the three channels of an image held with its channels last, each every third element of the array.

### `dlfs_convnet`, 1.07x against NumPy, 0.99x against the eager program with the kernels

- **The gain is `install()`'s:** 792 ms to 733 ms eagerly, from `out[self.mask] = 0` in the fully connected layer's ReLU, `np.argmax(col, axis=1)` in the pooling layer, and `np.sum`. The same assignment in the convolution's ReLU is left to NumPy 27 times: the mask is `x <= 0` of an array that `transpose(0, 3, 1, 2)` left out of order, and NumPy gives the mask that order.
- **The lazy run does less work than the program.** The trainer calls `network.loss` after each update, to print it; that forward pass computes the pooling layer's `arg_max`, which only a backward pass reads, and the next forward pass overwrites it. The same in the three passes that measure accuracy. A graph never computes what nothing reads: 11 of the 19 `argmax` calls are not made. Node by node the lazy run is the quickest of the four, 708 ms.
- **Fusion finds little and costs a little.** The time is the writes into slices of `im2col` and `col2im`, the matrix products, and `np.max` along rows of four. The 24 expressions planned are Adam's updates, `params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)`, 13 ms of kernels for 50 ms of NumPy. With fusion on, 1,368 calls are made at once, and the run is 35 ms behind the lazy one. Which of those calls it is in was not looked into.

### `salib_sobol`, 1.02x

- **Its time is not arithmetic.** 55% of the nodes' time is `A[r]`, a hundred resamples of 32,768 values taken by a table of positions, and 31% is `np.var` and `np.mean` along the first axis of the results. Neither kind of kernel has those.
- **15 of its 31 programs are given up, all for arrays that step, and they weigh nothing.** `separate_output_values` takes `A = Y[0:Y.size:step]`, every eighth value, and `AB[:, j]`, a column of a table of three. The first-order estimator `np.mean(B * (AB - A), axis=0)` then reads three arrays at steps of 64, 24 and 64 bytes, and a single kernel call that would need three copies first is declined, as [the design decided for the Veros routines](lazy-fusion-design.md#what-was-built-on-it-and-did-not-gain). Those are the base sample's 32,768 values, under a millisecond in all. The same arithmetic on the hundred resamples, 3.3 million elements that indexing made contiguous, is fused: 127 ms of NumPy's passes made 54 ms.
- **Node by node it is 1.4x slower than the program,** 1,413 ms for 997 ms, the most of the ten: [Memory a graph holds](#memory-a-graph-holds).

### `perlin_noise`, 1.00x

- **38% of it is a sum along an axis of two.** `np.sum(np.dstack((grid[:,:,0], grid[:,:,1])) * g00, 2)` is a dot product of two-vectors at each pixel, four times an octave: 754 ms, for which NumPy calls its loop once for each of four million rows. `install()` meets these 20 calls at its hook for `np.sum` and leaves them.
- **`grid % 1` is 294 ms,** 7 ns an element.
- **The interpolant is found and given up, five times, 300 ms.** `t*t*t*(t*(t*6 - 15) + 10)` is seven operations and one program. But `grid` is `np.mgrid[...].transpose(1, 2, 0) % 1`, and NumPy gives the result of `%` the order of its operand: an array of shape (2048, 2048, 2) whose last axis is the slowest in memory, 33 MB from one element to the next. NumPy's own loops walk such an array in the order it lies; the module that runs a program takes the last axis for the rows and declines.
- **What is fused,** 26 programs, the largest of 75 operations for the whole sum of octaves, takes 918 ms of NumPy's passes down to 241 ms. That is what the lazy run had lost: node by node it is 2,277 ms.

### `mpl_hillshade`, 1.00x

- **81% of the nodes' time is expressions, and the two that matter are given up.** The overlay blend is `np.where(rgb <= 0.5, 2 * intensity * rgb, 1 - 2 * (1 - intensity) * (1 - rgb))`, 101 ms, on rows of four colour values with one `intensity` for each pixel, an array of shape (2048, 2048, 1) beside one of (2048, 2048, 4). And the length of each normal, `sum_sq += arr[..., i, np.newaxis] ** 2` for each of three components and then `np.sqrt(sum_sq)`, 24 ms, reads three arrays that each step over three elements. One program of four is kept.
- **`install()` serves `np.clip`, `np.where` and the colour map's `x[mask] = v`,** and the program is level with it and without: those calls are 2% of it.

### `speech_mfcc`, 1.00x

The Fourier transform, the table of positions the sound is cut into frames by (int64, 4.8 million elements, built with `np.tile`) and the frames taken through it are most of it, and none is deferred or fusable. One expression of weight, `1.0 / NFFT * np.square(magspec)`, is fused. Node by node the lazy run is 0.75x of NumPy; fused it is level.

### `pvlib_solar`, 0.97x

- **It is the program with the most to fuse, and gains nothing.** 2,520 operations in one graph; 290 expressions, all planned, 684 kernel calls, 638 nodes computed inside a program, and no program given up. The kernels take 291 ms for what NumPy's passes took 814 ms.
- **63% of it is the library.** The algorithm is sums of `a * np.cos(b + c * x)` over the rows of a table, 64 terms for one quantity: `np.cos` alone is 1,229 ms, `np.sin` 339 ms, `x ** 3` and `x ** 4` 158 ms.
- **The eager program is already near what fusion makes of the rest.** `s += arr[row, 0] * np.cos(arr[row, 1] + arr[row, 2] * x)` makes few arrays as NumPy computes it, because NumPy reuses a temporary's memory and `+=` adds in place. The lazy run writes 18.5 GB of new arrays for it and the fused run 13.7 GB; node by node the program is 5% slower than eagerly, and fusion takes back 2%.
- **A whole number from a table ends an expression.** The nutation terms are `NUTATION_YTERM_ARRAY[row, 0]*x0 + ... + NUTATION_YTERM_ARRAY[row, 4]*x4`, and that table is int64. A NumPy scalar of another type than float64 is not an operand a kernel takes, so each of the 315 products is NumPy's pass, 176 ms that the limit tool counts under other element types, and only the sum of five is a program, where the whole line is nine operations. `np.int64(k) * x` and `float(k) * x` have the same bits.

### `sklearn_mlp`, 0.88x

- **Two thirds of it is five matrix products a batch,** in `safe_sparse_dot` on plain arrays, 141 ms of 223 ms. Nothing of either kind touches them.
- **What is fused is Adam.** `self.beta_1 * m + (1 - self.beta_1) * grad` and `-self.learning_rate * m / (np.sqrt(v) + self.epsilon)` on the 78,400 weights: 450 programs, 37 ms of NumPy's passes made 26 ms.
- **It loses on the 6,900 calls made at once.** The batch's arrays are 200 rows of 100 or 784, and the library's code is written to compute in place: `np.maximum(X, 0, out=X)`, `np.exp(X, out=X)`, `np.clip(y_prob, eps, 1 - eps)`, `param += update`. Each of those goes a slow road through the stand-in: [Calls through the stand-in](#calls-through-the-stand-in).

## What is not supported

The question this was run to answer: where a program does not gain, is it an operation that is missing, or a combination of operations that occurs and is not taken? In the order of what each weighs in the ten programs, of 7.9 s of NumPy in all:

| What | ms | Programs | Why it is not taken | Could it be |
|---|---|---|---|---|
| Functions of a mathematics library: `cos`, `sin`, `**` with an exponent other than 2, `%`, `cbrt`, `hypot`, `arctan2`, `exp` | 2,650 | `pvlib_solar`, `perlin_noise`, `skimage_delta_e`, `astropy_models`, `prysm_zernike` | A kernel calls nothing, and a second implementation would not have the library's bits | Not bit for bit. `x ** 3` takes 6.7 ns an element where a multiplication takes 0.35 ns, and `x * x * x` does not have its bits. Only as kernels of the package's inexact kind (`install(inexact=True)`), which the lazy layer does not have. `x % 1.0` is the exception: `x - np.floor(x)` gave the same bits on four million numbers of both signs, in 2.0 ns an element for 11.2 ns |
| Reductions along an axis | 1,200 | `perlin_noise` (754, rows of two), `salib_sobol` (265), `dlfs_convnet` (182, rows of four) | Fusion reduces over every element only; the package's kernels for sums along rows are for other shapes | `np.sum(a * b, axis=-1)` on rows of two or three is a dot product at each row, three operations on arrays that step. A kernel for rows of a few elements would be 38% of the noise program |
| Indexing by positions, writes into slices | 900 | `salib_sobol` (457, `A[r]`), `dlfs_convnet` (196, `im2col`), `pvlib_solar` (160, `s += ...`) | Not element by element | Taking by positions is a kernel of the package's, not of fusion: [issue 232](https://github.com/sofeikov/hotkernels/issues/232) is its sibling for scattering |
| An expression on an array left out of order by the call that made it | 300 | `perlin_noise` | The module takes the last axis for the rows; NumPy walks the array as it lies in memory | By walking a program's arrays in the order of memory when all of them share it, as NumPy's iterator does. No kernel changes |
| A whole-number NumPy scalar as an operand | 176 | `pvlib_solar` | Only float64 scalars and Python numbers are numbers to a kernel | Yes, with the bits: the conversion is the one NumPy makes. A table of whole-number coefficients is common in such series |
| Rows of three or four values with one value for each row, in a program of several steps | 100 | `mpl_hillshade` | One operation on such rows is a loop of the module's; a program of several is declined | It is the shape of every blend of a colour image with a mask or an intensity, `(H, W, 1)` beside `(H, W, 3)` |
| Whole-number arrays as the first operand | 38 | `astropy_models` (28), `speech_mfcc` (10) | The kernels are float64 | A cast is "a kernel nobody wrote" already in the design's list |
| Expressions given up for arrays that step: a channel, a column, every eighth element | 30 | `mpl_hillshade` (24), `skimage_delta_e` (7), `salib_sobol` (under 1) | A kernel reads arrays whose elements are next to each other. Copying each array first costs what the passes saved for a short program | By kernels that take a step for each array, which [the design names](lazy-fusion-design.md#what-a-pattern-could-still-take) and did not build for two programs' sake. Here it is three more programs, since it is how they hold their data, an image read by channel and a table by column; but it weighs little in them |

- **None of it is a missing pattern.** Every expression found on arrays a kernel can read was planned: 450 of 450 in the perceptron, 290 of 290 in the solar position, 45 of 45 in prysm, 24 of 24 in the network, 17 of 17 in Astropy's models. The library of shapes with a kernel for every single operation covers what these codebases write.
- **What is missing is on either side of the pattern:** the functions between the expressions, and the layouts under them.

## What gets in the way

Where the layer costs a program time or stops it, as these ten found it.

### Memory a graph holds

Six of the ten are slower node by node than eagerly: SALib 0.71x, the speech features 0.75x, the perceptron 0.82x, the noise 0.86x, Astropy's models 0.90x, the solar position 0.95x. The calls are the same calls.

NumPy computes `a * (1 - t) + t * b` reusing the memory of a temporary for the next operation's result where nothing else holds it, and frees each temporary as soon as it is read, so that the next array of that size is the same warm pages. A graph holds every intermediate from when it is computed until the graph is done. Timed by itself on four million elements, the expression takes 12.9 ms in one line and 17.8 ms with each intermediate held by a name, 1.38x; on one million, 2.9 ms and 4.9 ms, 1.69x.

So on a program of large temporaries the lazy run starts behind, and fusion's first gain is to win that back: the noise program's nodes go from 1,949 ms to 1,348 ms fused, and the run is where NumPy's was. The corpus's stencils did not show this because their fused runs go far past it.

### Calls through the stand-in

What a call costs over NumPy's own, on arrays of 200 × 100, with fusion on:

| Call | NumPy | Through the stand-in, plain arrays | Through the stand-in, its own arrays |
|---|---|---|---|
| `np.dot(s, s)` | 2.20 µs | +2.08 µs | +0.16 µs |
| `np.sum(x, axis=0)` | 4.88 µs | +1.59 µs | +0.21 µs |
| `np.sqrt(x)` | 11.60 µs | +1.12 µs | +0.10 µs |
| `np.maximum(x, 0, out=x)` | 4.28 µs | +5.88 µs | +5.00 µs |
| `np.exp(x, out=x)` | 35.28 µs | +5.77 µs | +4.74 µs |
| `np.clip(x, lo, hi)` | 4.22 µs | +8.14 µs | +7.53 µs |
| `x += y` | 2.74 µs | +4.49 µs, `y` the stand-in's | +0.05 µs |
| `x + y` | 5.97 µs | +0.45 µs, `y` the stand-in's | level |

- **On its own arrays the layer is where [the design measured it](lazy-fusion-design.md#as-the-graph-is-built),** 0.05 to 0.2 µs a call.
- **Three roads are 20 to 80 times that:** a ufunc called with `out=`, on any array; `np.clip`; and any function of the stand-in called on plain `ndarray`s, or a plain array updated in place by one of the layer's.
- **They are how a library computes.** A library's own arrays are plain, whatever its `np` is, and code written for speed writes in place. The perceptron is 0.88x for it, and 0.65x at its small size.

### The type of a lazy array

`LightSource.hillshade` makes its array of normals with `np.empty(elevation.shape + (3,)).view(type(elevation))`, so that a masked array stays masked. The type of a lazy array is not an array type; NumPy takes it for an element type and raises `TypeError: Cannot change data-type for array of references`. The program hands Matplotlib a plain `ndarray`, as the corpus does for scikit-image, which fails on `np.isscalar` of a lazy mean. It is the second library to ask what a lazy array is.

### The order of timing

`fusion_time` timed a program one way after another: NumPy first, then the three ways with the kernels installed, then with `--plain` the lazy ways with nothing installed. A process that has run a program several times runs it faster. Why was not measured; the likeliest reason is that the allocator has kept the earlier runs' memory, so that a large array is no longer new pages from the system. Three things in the first tables taken here were the order and nothing else:

- pvlib ran in 2,996 ms on NumPy and 2,724 ms installed, 1.10x, with no call taken by a kernel.
- prysm's fused run took 362 ms with the kernels installed and 318 ms without, in three tables out of three; computed node by node with each call timed, its nodes take 282.6 ms and 282.5 ms.
- The noise program's fused run was 0.99x of NumPy with the kernels installed and 1.05x without, in one table.

`--alternate` times every way once a round, round after round, and keeps the quickest of each. With it pvlib is 2,646 ms and 2,660 ms, and prysm 317 ms and 318 ms. The tables of [the design](lazy-fusion-design.md#the-programs-four-ways) were taken one way after another; a program there whose eager time moved between two tables, as the regression's did on x86-64, may be this.

### What got in the way of running them

- **Matplotlib,** as above.
- **SALib seeds its resampling with `if seed:`,** so a seed of 0 is no seed, and two eager runs differ. The program's seed is 1.
- **Nothing else.** `np.random` under the stand-in, a module that calls NumPy `numpy` and not `np`, a book's modules that append to `sys.path` and import each other by a bare name, a network that updates its weights in place through a second name for each array: all ran as written.

## What changed in the repository

- **`prototypes/lazy/programs/codebases.py`** has the ten programs. `corpus.by_name` finds them by name and takes `codebases` for all ten; `corpus.programs()` is the twenty-six it was.
- **`prototypes/lazy/programs/vendor/`** has two more directories, `perlin_numpy/` and `dlfs/`, each with its licence; [its README](../prototypes/lazy/programs/vendor/README.md) has the commits.
- **`fusion_time.py` can time the ways in turn** (`alternately`). In this study that was an option, `--alternate`; it is the default now, and `--one-after-another` asks for the order before it ([issue 323](#issue-323-the-ways-timed-in-turn-by-default)).
- **A dependency group, `codebases`:** Matplotlib, prysm, pvlib, python_speech_features, SALib, and setuptools below 81 for prysm's `pkg_resources`.
- **The two tests that run every program of the corpus fused** run these ten as well, at their small size, and skip one whose library is missing.
- **Nothing in the layer, in `fuse.py` or in a kernel changed.**

The reasons a program was given up, the calls each kernel of the package took line by line, and the times of single calls were measured with scripts that wrap `fuse._Group.run`, read `hotkernels`' counters between lines and time a call both ways. They are not in the repository.

## What is open

Each of these is a measurement above and a decision not taken here, with the issue it is parked in.

- **A mathematics library's functions under fusion,** [issue 315](https://github.com/sofeikov/hotkernels/issues/315). A third of these programs. The package has a kind of kernel that is not bit-identical, off unless asked for; the lazy layer has no such kind, and its design rests on NumPy's bits. Whether `fuse.enable(inexact=True)` should exist is a design question, and `cos`, `sin` and powers by a whole number are what it would be for. `x % 1.0` as `x - np.floor(x)` is in [issue 222](https://github.com/sofeikov/hotkernels/issues/222), which was open for the remainder already.
- **Walking a program's arrays in the order of memory,** [issue 316](https://github.com/sofeikov/hotkernels/issues/316).
- **Rows of three or four with a value for each row, in a program of several steps,** [issue 317](https://github.com/sofeikov/hotkernels/issues/317).
- **Whole-number NumPy scalars as numbers,** [issue 318](https://github.com/sofeikov/hotkernels/issues/318).
- **A sum of a product along a short last axis,** [issue 319](https://github.com/sofeikov/hotkernels/issues/319).
- **`out=`, `np.clip` and plain operands on the core's quick road,** [issue 320](https://github.com/sofeikov/hotkernels/issues/320).
- **Memory a graph holds,** [issue 321](https://github.com/sofeikov/hotkernels/issues/321): whether a node's value can be released, or its memory used again, as soon as its last reader has run.
- **A library that keeps the type of what it is given,** [issue 322](https://github.com/sofeikov/hotkernels/issues/322).
- **`--alternate` as the timing tool's default,** and the design's tables taken again with it, [issue 323](https://github.com/sofeikov/hotkernels/issues/323).
- **Taking by a table of positions,** which is half of SALib's analysis: [issue 224](https://github.com/sofeikov/hotkernels/issues/224), open already for `a[idx]`.
- **Kernels that read arrays at a step.** The design left them for two programs; three more have such arrays here, for 30 ms. No issue: it weighs too little in these to add to what the design says.
