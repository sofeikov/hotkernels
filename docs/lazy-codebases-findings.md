# Ten programs from other codebases, with `install()` and with fusion

The corpus fusion was built on ([lazy-fusion-design.md](lazy-fusion-design.md#the-corpus)) is this repository's own programs, benchmark suites, and calls into libraries that keep their own `numpy`. This document is about ten more programs, each the NumPy code of a published project that was written against neither the package nor the lazy layer, and about one question: what do `hotkernels.install()` and the lazy layer with fusion do for them, and where they do nothing or cost time, why.

Nothing here is in the package. The programs are in [`prototypes/lazy/programs/codebases.py`](../prototypes/lazy/programs/codebases.py) and are run by the corpus's tools.

## Summary

- **All ten give NumPy's results bit for bit,** node by node and fused, with the kernels installed and without. When they were measured one library, Matplotlib, had to be handed a plain `ndarray`; it takes a lazy one since ([`np.empty(...).view(type(x))`](#npemptyviewtypex-issue-322)). One seed had to be other than zero ([What got in the way of running them](#what-got-in-the-way-of-running-them)).
- **Fusion gains on three, by 1.08x to 1.14x, is level on five, and loses on two.** Against plain NumPy at the large size: Astropy's models 1.14x, prysm's Zernike wavefront 1.10x, scikit-image's colour difference 1.08x; the book's convolutional network 1.07x, which is `install()`'s gain and not fusion's; SALib 1.02x; perlin-numpy, Matplotlib's hill shading and the speech features 1.00x; pvlib's solar position 0.97x; scikit-learn's perceptron 0.88x, and 0.95x since the calls it loses on were given a quick road ([`out=`, `np.clip` and plain operands](#out-npclip-and-plain-operands-issue-320)). Nothing like the 2x to 3x of the corpus's stencils. pvlib's is 1.02x to 1.05x since a whole number from its tables is a number of an expression ([issue 318](#issue-318-a-whole-number-of-numpys-is-a-number-of-an-expression)). Since then perlin-numpy's arrays are walked in the order they lie in memory, and it gains as well, at 1.13x: [What was built on it](#what-was-built-on-it).
- **`install()` alone gains on one of the ten,** the convolutional network, 1.08x, from its kernels for `x[mask] = 0` and for `argmax` and sums along rows. On the other nine it is within 2% of NumPy either way: their time is not in calls the package has kernels for.
- **A third of all the time in the ten programs is a mathematics library's functions:** `cos`, `sin`, `x ** 3`, `x ** 2.4`, `%`, `cbrt`, `hypot`, `arctan2`, `exp`. 2.6 s of 7.9 s. No kernel of either kind may touch them, because the result has to be the library's bits. This is the largest thing the ten programs have that the corpus did not show at this weight. What a kind that is not bit-identical would be worth was measured ([issue 315](#issue-315-what-an-inexact-kind-would-be-worth)): glibc's vector functions in place of NumPy's loops make pvlib 2.17x with no graph, and the decision is left open.
- **The float64 arithmetic that fusion is for is 30% of the time, and a fifth of that was found and then given up for how the arrays lie.** An array left transposed by the call that made it, 300 ms; rows of four colour values with one value for each pixel, 100 ms; and arrays that step, a channel of an image (`lab[..., 0]`), a column of a table (`AB[:, j]`), every eighth element (`Y[::8]`), 30 ms in three programs. Each is how these codebases hold their data, and none is a missing pattern. The first of the three is walked since, and the second is one program since, with the selection written on it: Matplotlib's hill shading went from level to 1.3x of NumPy ([What was built on it](#what-was-built-on-it)).
- **A lazy run computed node by node was slower than the eager program on six of the ten,** by up to 1.4x, before fusion is counted. Two things were behind it, and neither was quite what was supposed. SALib and the speech features lost most of it to the shape rule of `A[r]`, which had NumPy walk every position of a large table a second time; the rest lost it to each node's call writing a new array where NumPy writes into a temporary's memory. Both are fixed: node by node SALib is 0.89x of the eager program where it was 0.65x, the speech features 0.92x where 0.76x, pvlib 1.05x where 0.95x, prysm 1.11x where 0.93x ([Memory a graph holds](#memory-a-graph-holds-issue-321)). What is left is the noise program, 0.95x: its pending octaves hold 2.2 GB at once where the eager program holds 0.5 GB.
- **Calls through the stand-in had slow roads the corpus did not meet:** a ufunc called with `out=`, `np.clip`, a function called on plain `ndarray`s, and `plain += lazy` cost 1.1 to 8.1 µs over NumPy's own call, where the same calls on the layer's arrays cost 0.05 to 0.2 µs. scikit-learn's perceptron makes 1,800 of them among its 7,000 calls at once and was 0.88x for it. The core makes them since, at 0.1 to 0.3 µs and 0.9 µs for `np.clip`, and the perceptron is 0.95x, and 0.98x at its small size where it was 0.66x ([`out=`, `np.clip` and plain operands](#out-npclip-and-plain-operands-issue-320)).
- **The timing tool's order was worth several percent, and the tools time in turn now.** Timed one way after another, the way timed first, NumPy's, was the slowest whatever it was: pvlib ran 10% "faster installed" with no kernel serving a single call. `fusion_time` and `limit` time the ways in turn, by default since [issue 323](#issue-323-the-ways-timed-in-turn-by-default), which also found that much of it was not the order: the tool kept a result of the program alive while it timed, and the runs after it reused memory. Every table of the four ways below is taken in turn, with that result kept.

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
- **The ways in turn,** with the tool as it was then, which kept one result of the program while it timed. See [The order of timing](#the-order-of-timing), and [what issue 323 found](#issue-323-the-ways-timed-in-turn-by-default) for what that result moved: the times of prysm's program by 13%, and no ratio in these tables by more than 0.02.

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
- **Node by node it was 1.4x slower than the program,** 1,413 ms for 997 ms, the most of the ten, and it was not memory: the shape rule of `A[r]` had NumPy walk the 3.3 million positions of the resampling table a second time, 9 ms for each of 39 calls. It is 0.89x since ([Memory a graph holds](#memory-a-graph-holds-issue-321)).

### `perlin_noise`, 1.00x

- **38% of it is a sum along an axis of two.** `np.sum(np.dstack((grid[:,:,0], grid[:,:,1])) * g00, 2)` is a dot product of two-vectors at each pixel, four times an octave: 754 ms, for which NumPy calls its loop once for each of four million rows. `install()` meets these 20 calls at its hook for `np.sum` and leaves them.
- **`grid % 1` is 294 ms,** 7 ns an element.
- **The interpolant is found and given up, five times, 300 ms.** `t*t*t*(t*(t*6 - 15) + 10)` is seven operations and one program. But `grid` is `np.mgrid[...].transpose(1, 2, 0) % 1`, and NumPy gives the result of `%` the order of its operand: an array of shape (2048, 2048, 2) whose last axis is the slowest in memory, 33 MB from one element to the next. NumPy's own loops walk such an array in the order it lies; the module that runs a program took the last axis for the rows and declined. It walks the order of memory since, and the program is at 1.13x: [What was built on it](#what-was-built-on-it).
- **What is fused,** 26 programs, the largest of 75 operations for the whole sum of octaves, takes 918 ms of NumPy's passes down to 241 ms. That is what the lazy run had lost: node by node it is 2,277 ms.

### `mpl_hillshade`, 1.00x

- **81% of the nodes' time is expressions, and the two that matter are given up.** The overlay blend is `np.where(rgb <= 0.5, 2 * intensity * rgb, 1 - 2 * (1 - intensity) * (1 - rgb))`, 101 ms, on rows of four colour values with one `intensity` for each pixel, an array of shape (2048, 2048, 1) beside one of (2048, 2048, 4). And the length of each normal, `sum_sq += arr[..., i, np.newaxis] ** 2` for each of three components and then `np.sqrt(sum_sq)`, 24 ms, reads three arrays that each step over three elements. One program of four is kept. The blend is one program since, and the program runs at 1.30x ([Rows of three or four with a value for each row](#rows-of-three-or-four-with-a-value-for-each-row)); the length of the normals is still given up.
- **`install()` serves `np.clip`, `np.where` and the colour map's `x[mask] = v`,** and the program is level with it and without: those calls are 2% of it.
- **These figures are of the elevation model handed over as a plain `ndarray`,** which `np.empty(...).view(type(elevation))` required then. Handed over lazy, as it is now, the program is level with this: [`np.empty(...).view(type(x))`](#npemptyviewtypex-issue-322).

### `speech_mfcc`, 1.00x

The Fourier transform, the table of positions the sound is cut into frames by (int64, 4.8 million elements, built with `np.tile`) and the frames taken through it are most of it, and none is deferred or fusable. One expression of weight, `1.0 / NFFT * np.square(magspec)`, is fused. Node by node the lazy run was 0.75x of NumPy, 15 ms of the 20 ms it was behind by the shape of `padsignal[indices]` worked out by walking the 4.8 million positions; it is 0.92x since ([Memory a graph holds](#memory-a-graph-holds-issue-321)). Fused it is level.

### `pvlib_solar`, 0.97x

- **It is the program with the most to fuse, and gains nothing.** 2,520 operations in one graph; 290 expressions, all planned, 684 kernel calls, 638 nodes computed inside a program, and no program given up. The kernels take 291 ms for what NumPy's passes took 814 ms.
- **63% of it is the library.** The algorithm is sums of `a * np.cos(b + c * x)` over the rows of a table, 64 terms for one quantity: `np.cos` alone is 1,229 ms, `np.sin` 339 ms, `x ** 3` and `x ** 4` 158 ms. With glibc's vector functions in NumPy's place the eager program is 2.17x and the fused one 2.27x, at another last bit for most results ([issue 315](#issue-315-what-an-inexact-kind-would-be-worth)). Its 89 `np.radians` and `np.degrees` are operations of an expression now.
- **The eager program is already near what fusion makes of the rest.** `s += arr[row, 0] * np.cos(arr[row, 1] + arr[row, 2] * x)` makes few arrays as NumPy computes it, because NumPy reuses a temporary's memory and `+=` adds in place. The lazy run writes 18.5 GB of new arrays for it and the fused run 13.7 GB; node by node the program was 5% slower than eagerly, and fusion took back 2%. Since a call writes into the memory of an operand nothing else reads, as NumPy's do, the run node by node is 1.05x of the eager program and the fused run 1.03x ([Memory a graph holds](#memory-a-graph-holds-issue-321)).
- **A whole number from a table ended an expression.** The nutation terms are `NUTATION_YTERM_ARRAY[row, 0]*x0 + ... + NUTATION_YTERM_ARRAY[row, 4]*x4`, and that table is int64. A NumPy scalar of another type than float64 was not an operand a kernel took, so each of the 315 products was NumPy's pass, 176 ms that the limit tool counted under other element types, and only the sum of five was a program, where the whole line is nine operations. `np.int64(k) * x` and `float(k) * x` have the same bits. It is a number now, and the line one program of nine operations: [issue 318](#issue-318-a-whole-number-of-numpys-is-a-number-of-an-expression).

### `sklearn_mlp`, 0.88x

- **Two thirds of it is five matrix products a batch,** in `safe_sparse_dot` on plain arrays, 141 ms of 223 ms. Nothing of either kind touches them.
- **What is fused is Adam.** `self.beta_1 * m + (1 - self.beta_1) * grad` and `-self.learning_rate * m / (np.sqrt(v) + self.epsilon)` on the 78,400 weights: 450 programs, 37 ms of NumPy's passes made 26 ms.
- **It loses on the 6,900 calls made at once.** The batch's arrays are 200 rows of 100 or 784, and the library's code is written to compute in place: `np.maximum(X, 0, out=X)`, `np.exp(X, out=X)`, `np.clip(y_prob, eps, 1 - eps)`, `param += update`. Each of those went a slow road through the stand-in: [Calls through the stand-in](#calls-through-the-stand-in). The core makes them since, and the program is 0.95x: [`out=`, `np.clip` and plain operands](#out-npclip-and-plain-operands-issue-320).

## What is not supported

The question this was run to answer: where a program does not gain, is it an operation that is missing, or a combination of operations that occurs and is not taken? In the order of what each weighs in the ten programs, of 7.9 s of NumPy in all:

| What | ms | Programs | Why it is not taken | Could it be |
|---|---|---|---|---|
| Functions of a mathematics library: `cos`, `sin`, `**` with an exponent other than 2, `%`, `cbrt`, `hypot`, `arctan2`, `exp` | 2,650 | `pvlib_solar`, `perlin_noise`, `skimage_delta_e`, `astropy_models`, `prysm_zernike` | A kernel calls nothing, and a second implementation would not have the library's bits | Not bit for bit. `x ** 3` takes 6.7 ns an element where a multiplication takes 0.35 ns, and `x * x * x` does not have its bits. Only as kernels of the package's inexact kind (`install(inexact=True)`), which the lazy layer does not have; measured and left to the user in [issue 315](#issue-315-what-an-inexact-kind-would-be-worth). `x % 1.0` is the exception: `x - np.floor(x)` has the same bits, shown on 100 million values of every class, but the kernels have no `floor`. `np.radians` and `np.degrees`, 81 ms, are one multiplication each and operations of an expression now |
| Reductions along an axis | 1,200 | `perlin_noise` (754, rows of two), `salib_sobol` (265), `dlfs_convnet` (182, rows of four) | Fusion reduces over every element only; the package's kernels for sums along rows are for other shapes | `np.sum(a * b, axis=-1)` on rows of two or three is a dot product at each row, three operations on arrays that step. A kernel for rows of a few elements would be 38% of the noise program |
| Indexing by positions, writes into slices | 900 | `salib_sobol` (457, `A[r]`), `dlfs_convnet` (196, `im2col`), `pvlib_solar` (160, `s += ...`) | Not element by element | Taking by positions is a kernel of the package's, not of fusion: [issue 232](https://github.com/sofeikov/hotkernels/issues/232) is its sibling for scattering |
| An expression on an array left out of order by the call that made it | 300 | `perlin_noise` | The module took the last axis for the rows; NumPy walks the array as it lies in memory | Done since, with no kernel changed: the module walks a program's arrays in the order of memory when all of them share it, as NumPy's iterator does ([What was built on it](#what-was-built-on-it)) |
| A whole-number NumPy scalar as an operand | 176 | `pvlib_solar` | Only float64 scalars and Python numbers were numbers to a kernel | Yes, with the bits: the conversion is the one NumPy makes. A table of whole-number coefficients is common in such series. Built: [issue 318](#issue-318-a-whole-number-of-numpys-is-a-number-of-an-expression) |
| Rows of three or four values with one value for each row, in a program of several steps | 100 | `mpl_hillshade` | One operation on such rows is a loop of the module's; a program of several is declined. Done since: each value is spread over its row inside a block, and the comparison and `np.where` are steps of the same program ([What was built on it](#what-was-built-on-it)) | It is the shape of every blend of a colour image with a mask or an intensity, `(H, W, 1)` beside `(H, W, 3)` |
| Whole-number arrays as the first operand | 38 | `astropy_models` (28), `speech_mfcc` (10) | The kernels are float64 | A cast is "a kernel nobody wrote" already in the design's list |
| Expressions given up for arrays that step: a channel, a column, every eighth element | 30 | `mpl_hillshade` (24), `skimage_delta_e` (7), `salib_sobol` (under 1) | A kernel reads arrays whose elements are next to each other. Copying each array first costs what the passes saved for a short program | By kernels that take a step for each array, which [the design names](lazy-fusion-design.md#what-a-pattern-could-still-take) and did not build for two programs' sake. Here it is three more programs, since it is how they hold their data, an image read by channel and a table by column; but it weighs little in them |

- **None of it is a missing pattern.** Every expression found on arrays a kernel can read was planned: 450 of 450 in the perceptron, 290 of 290 in the solar position, 45 of 45 in prysm, 24 of 24 in the network, 17 of 17 in Astropy's models. The library of shapes with a kernel for every single operation covers what these codebases write.
- **What is missing is on either side of the pattern:** the functions between the expressions, and the layouts under them.

## What gets in the way

Where the layer costs a program time or stops it, as these ten found it.

### Memory a graph holds

What follows is what the study found. [Memory a graph holds, issue 321](#memory-a-graph-holds-issue-321) has what it was when measured, and what was built.

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
- **The core makes all three since,** at 0.1 to 0.3 µs over NumPy's call, and `np.clip` at 0.9 µs: [`out=`, `np.clip` and plain operands](#out-npclip-and-plain-operands-issue-320). The table above is what they cost before.

### The type of a lazy array

`LightSource.hillshade` makes its array of normals with `np.empty(elevation.shape + (3,)).view(type(elevation))`, so that a masked array stays masked. The type of a lazy array is not an array type; NumPy takes it for an element type and raises `TypeError: Cannot change data-type for array of references`. When the tables above were taken the program handed Matplotlib a plain `ndarray` for that, as the corpus does for scikit-image, which fails on `np.isscalar` of a lazy mean. It is the second library to ask what a lazy array is.

Both questions have an answer now, where they reach the layer: `x.view(type(x))` on a lazy array is `x.view()`, and `np.isscalar` of the stand-in says what NumPy says of the value. Matplotlib is handed the lazy elevation model. scikit-image asks its own `numpy.isscalar`, which no lazy array can answer, and is still handed plain arrays. [`np.empty(...).view(type(x))`](#npemptyviewtypex-issue-322) has what was built and measured, and [the design](lazy-design.md#what-a-library-asks-a-lazy-array) the rule.

### The order of timing

`fusion_time` timed a program one way after another: NumPy first, then the three ways with the kernels installed, then with `--plain` the lazy ways with nothing installed. A process that has run a program several times runs it faster. Why was not measured; the likeliest reason is that the allocator has kept the earlier runs' memory, so that a large array is no longer new pages from the system. Three things in the first tables taken here were the order and nothing else:

- pvlib ran in 2,996 ms on NumPy and 2,724 ms installed, 1.10x, with no call taken by a kernel.
- prysm's fused run took 362 ms with the kernels installed and 318 ms without, in three tables out of three; computed node by node with each call timed, its nodes take 282.6 ms and 282.5 ms.
- The noise program's fused run was 0.99x of NumPy with the kernels installed and 1.05x without, in one table.

Timing in turn, every way once a round, round after round, with the quickest of each kept, was an option of the tool in this study, `--alternate`. With it pvlib was 2,646 ms and 2,660 ms, and prysm 317 ms and 318 ms.

**What it was** came out when the order was made the tool's default ([issue 323](#issue-323-the-ways-timed-in-turn-by-default)). The time is new pages from the system, as guessed, and it was measured there. But the second of the three was not the order: the tool kept a result of the program alive from its comparison of bits onwards, and every run after that reused memory where the runs before it had not. prysm's 362 ms was the run with nothing kept, and 318 ms the run under a kept result. In turn every way ran under the kept result, so the two figures agreed at the lower one. The tool keeps nothing now, and prysm's fused run is 361 ms with the kernels installed and 362 ms without.

### What got in the way of running them

- **Matplotlib,** as above, until the layer took `x.view(type(x))`.
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

## What was built on it

One subsection for each issue of [What is open](#what-is-open) that was taken up, with what it measured. The tables of [What was measured](#what-was-measured) are the study's and are as they were.

### Issue 323: the ways timed in turn by default

[Issue 323](https://github.com/sofeikov/hotkernels/issues/323) asked for `--alternate` as the default of `fusion_time`, for `limit` to time in turn, and for the design's tables again.

**What changed.**

- **`fusion_time` and `limit` time the ways in turn.** A round runs the program once on NumPy, once eagerly with the kernels installed, once lazily and once fused, and with `--plain` once each for the two lazy ways with nothing installed; the quickest run of each way counts. `--one-after-another` asks either tool for the order before. Each table's heading says which it was.
- **`fusion_time` keeps no result of a program while it times.** It did, and that was most of what the order had been taken for. See below.
- **Neither tool prints the package's lines about installing,** which a round would print twice.

**The time is new pages from the system, and it is a quarter of such a program.** The study guessed it and did not measure it. The equation of state of the corpus (`veros_eos`, 366 operations on arrays of 8.7 MB) was run alone in a new process, one way, pinned, with the page faults of each run counted. Every run after the first took the same:

| How freed memory is kept | NumPy | Page faults a run | Lazy | Page faults | Fused | Page faults |
|---|---|---|---|---|---|---|
| By glibc as it is | 245 ms | 60,738 | 270 ms | 36,476 | 138 ms | 33,108 |
| One result of the program still alive, as the tool had it | 196 ms | 12,071 | 225 ms | 0 | 99 ms | 0 |
| glibc told to give nothing back (`MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=4294967295`) | 181 ms | 0 | 224 ms | 0 | 99.5 ms | 0 |

- **glibc gives memory back to the system.** A large array has a mapping of its own, unmapped when the array is freed, or lies at the top of the heap, which is cut back when enough of it is free. The next array of that size is then new pages, each zeroed by the system: 249 MB of them in one eager run of this program. That is 64 ms of its 245 ms, and 38 ms of the fused run's 138 ms.
- **A process does not grow out of it.** Of twelve eager runs in a row, the eleven after the first took 244 to 251 ms each, with 60,228 page faults each. So "a process that has run a program several times runs it faster" is not what happens.
- **An array that stays alive changes it.** With one result of the program alive, 8.7 MB, the next runs took their memory with a fifth of the page faults or none. The likeliest reason is that memory freed below a live array is not the top of the heap and cannot be given back; that much was not measured.

**What the tool did.** To compare bits, `fusion_time` ran the program on NumPy and kept the result while it ran the lazy ways. With `--plain` that result stayed alive to the end of the program's measurement. Every run of the tool on the equation of state, in the order it made them, one way after another:

| Runs | Each took | Page faults a run |
|---|---|---|
| NumPy, timed | 244 to 250 ms | 57,155 to 61,353 |
| With the kernels installed, for the comparison of bits: NumPy, lazy and fused once each, the three results kept | 246 ms, 228 ms, 130 ms | 61,180, 7, 26 |
| NumPy, timed, the three results dropped | 246 to 248 ms | 61,247 |
| Lazy, timed | 271 ms | 36,475 |
| Fused, timed | 138 to 140 ms | 33,020 |
| NumPy once more, its result kept for the comparison of the plain ways | 246 ms | 61,247 |
| With nothing installed and that result alive: lazy, timed | 225 ms | 0 |
| Fused, timed | 99 ms | 0 |

So the two plain columns were of another state of the process than the four before them, and that is the second of the study's three findings about the order: prysm's fused run at 362 ms with the kernels installed and 318 ms without. Timed in turn with the option of the study, every round came after that result was made, and every way ran under it.

**Before and after,** each program on one performance core with one BLAS thread, the quickest of 7 rounds, taken within twenty minutes of each other at load averages of 1.0 to 2.4. NumPy, installed, and the fused run with the kernels installed and without, in milliseconds:

| Program | How it was timed | NumPy | Installed | Fused | Fused, nothing installed | Fused against NumPy |
|---|---|---|---|---|---|---|
| `veros_eos` | One way after another, the tool as it was | 229.31 | 229.14 | 137.41 | 99.23 | 1.67x, and 2.31x with nothing installed |
| | In turn, a result kept (`--alternate` of the study) | 182.24 | 184.53 | 99.16 | 98.94 | 1.84x |
| | One way after another, nothing kept | 235.32 | 235.24 | 139.64 | 139.38 | 1.69x |
| | In turn, nothing kept (the tool now) | 233.55 | 235.22 | 138.40 | 137.47 | 1.69x |
| `prysm_zernike` | In turn, a result kept | 351.36 | 353.25 | 315.40 | 315.79 | 1.11x |
| | One way after another, nothing kept | 396.96 | 376.63 | 360.18 | 360.55 | 1.10x |
| | In turn, nothing kept | 397.14 | 399.58 | 361.48 | 361.97 | 1.10x |
| `hdiff` | One way after another, the tool as it was | 188.92 | 178.36 | 89.29 | 89.56 | 2.12x |
| | In turn, nothing kept | 193.15 | 180.76 | 92.12 | 91.60 | 2.10x |
| `regression` | One way after another, the tool as it was | 51.15 | 51.45 | 49.48 | 49.23 | 1.03x |
| | In turn, nothing kept | 52.27 | 53.45 | 51.46 | 51.22 | 1.02x |
| `pvlib_solar` | One way after another, the tool as it was | 2,634.85 | 2,633.65 | 2,721.44 | 2,697.11 | 0.97x |
| | In turn, a result kept | 2,650.63 | 2,657.33 | 2,702.08 | 2,707.03 | 0.98x |
| | In turn, nothing kept | 2,678.86 | 2,701.34 | 2,755.16 | 2,762.00 | 0.97x |

- **The kept result moved two programs' times and one ratio.** The equation of state is 28% slower on NumPy and 40% slower fused with nothing kept, and its fused run is 1.69x of NumPy where the study's way of timing made it 1.84x. prysm's program is 13% to 15% slower every way, and its ratio is the same. The other eleven programs timed both ways, the horizontal diffusion, the regression and nine of the study's ten, moved by 4.8% or less, with other work running on the machine.
- **The order by itself moved one figure of these:** prysm's installed run, 376.63 ms after NumPy's 396.96 ms when each way's rounds come together, and 399.58 ms beside 397.14 ms in turn. That is a 5% gain from `install()` on a program where no kernel takes a call.
- **pvlib's 10% did not come back.** One way after another it took 2,635 ms on NumPy and 2,634 ms installed, where the study's first table had 2,996 ms and 2,724 ms. What state the machine or the process was in for that table is not known.
- **In the corpus the order moved turbulent kinetic energy.** All twenty-six programs were timed pinned with the tool as it was and as it is, one take after the other. `veros_tke` was 1.50x of NumPy with NumPy timed first (205.15 ms) and 1.38x in turn (176.90 ms), and 1.39x in a second take in turn; unpinned earlier the same day, with the tool as it was, 1.62x one way after another and 1.39x with `--alternate`. The equation of state's two fused columns, 141.41 ms and 103.90 ms in the tool as it was, are 139.26 ms and 139.86 ms. Five more programs' ratios moved by over 3% in both takes in turn: the regression from 1.10x to 1.02x and 1.04x, `pandas_sales` from 0.97x to 1.02x, the flame graph from 1.55x to 1.47x and 1.49x, `pandas_filter` from 1.35x to 1.23x and 1.29x, the naive azimuthal integration from 0.95x to 0.85x and 0.91x. With a program's time moving by up to 28% between two takes on this machine, at load averages of 2.1 to 3.8, that is not enough to call any of them the order.
- **`limit` shows what the order is for on a machine that is doing something else.** One way after another, at load averages of 2.3 to 2.5, it had the equation of state at 282.31 ms on NumPy and 253.01 ms installed, 1.12x from kernels that take none of its calls, and Black-Scholes at 99.60 ms and 80.45 ms fused, 1.24x. In turn, straight after, 245.10 ms and 245.00 ms, and 79.31 ms and 70.90 ms, 1.12x. A slow stretch of some seconds falls on all the runs of one way, or on one round of every way.

**What it means for the tables above.** They were timed in turn with a result kept, so each row is of one state of the process, and its ratios stand: taken again with nothing kept, no program's fused run against NumPy moved by more than 0.02. Its times do not all stand. prysm's are 13% to 15% longer now (397 ms on NumPy for the table's 348 ms, 361 ms fused for 317 ms), and scikit-image's 2% to 5%. The other eight are within 4%, which is what two takes differ by with other work on the machine.

**The design's tables.** Those of [its section on x86-64](lazy-fusion-design.md#another-processor), the four ways and the limit, were taken again in turn, pinned to one core with one BLAS thread while other work ran on the machine, and that section has what moved: the regression from 2.26x to 1.02x, which was its NumPy time with BLAS on every core and not the order, turbulent kinetic energy from 1.62x to 1.38x, which was the order, and the attention block from 0.83x to 0.93x. The tables from the Apple M1 Pro were timed one way after another and could not be taken again here. What the order can have moved in them is a program whose NumPy run, timed first, was slow for it, as `veros_tke`'s here. The kept result shows less there: the equation of state's fused run is 73.97 ms with the kernels installed and 73.80 ms without in that machine's tables, where it was 141 ms and 104 ms here. macOS has another allocator.

**What is not settled.** A program of large temporaries has two times, with pages given back and with pages kept, and they differ by a quarter. The tools now time the first, which is what a program run by itself gets from glibc. A program inside a longer process, with arrays alive above its own, gets the second. Which the layer should be judged by, and whether a fused run should keep the memory of its results itself, since that alone took the equation of state from 138 ms to 99 ms, is [issue 321](https://github.com/sofeikov/hotkernels/issues/321)'s question from another side; [issue 326](https://github.com/sofeikov/hotkernels/issues/326) has the figures and the two decisions.

### Issue 318: a whole number of NumPy's is a number of an expression

[Issue 318](https://github.com/sofeikov/hotkernels/issues/318) asked that `np.int64(k) * x` on a float64 array be an operation of an expression, as `float(k) * x` is, since the two have the same bits.

**What changed.**

- **A NumPy scalar of a boolean or whole-number type is a number to a kernel,** of any integer type and any value, and so is an array or a pending result without dimensions of such a type (`np.asarray(3)`, `counts.sum()`). It is one wherever a float is: either side of `+`, `-`, `*`, `/`, `np.maximum`, `np.minimum` and a comparison, either value of `np.where`, an in-place operator and a write into part of an array. Both readers of expressions, `fuse.py`'s and the C core's, take it, and a test holds one to the other.
- **`x ** np.int64(2)` is the square.** What NumPy does for it was measured, not assumed: `np.power(x, np.int64(2))` takes 0.29 ns an element on a million, as `np.power(x, 2)` and `np.square(x)` do, where `np.power(x, np.int64(3))` takes 9.0 ns; and it has the bits of `x * x` on a million values with infinities, NaN, zeros of both signs and denormals among them, on every third element of an array, and on arrays of 1 to 17 elements. The loop is handed the exponent as the float64 2.0 and multiplies. A boolean exponent is 1, and any other is the library's.
- **One whole number waits for a kernel.** With fusion on, a large call that reads an array of another type is made at once. A lazy array around one whole number, `table[i, j]` of a lazy table, was such an array; it is a number there now.
- **`limit` counts an operation on such a number as arithmetic,** where it counted it under other element types.

**Why the bits are NumPy's.** For a float64 array and a whole-number scalar NumPy's result is float64 for every one of these operations, and it gets there by converting the scalar once, with C's conversion, and running the float64 loop. A kernel is handed `float(k)`, the same conversion. A value a float64 does not hold is rounded to the nearest by both: `np.int64(2**53 + 1)` to 2**53, `np.uint64(2**64 - 1)` to 2**64. The tests hold `float(k)` to NumPy's conversion on 1,500 values, every power of two of int64 and uint64 and either side of it among them; twenty-three scalars in twenty-seven expressions at five lengths to the real `numpy` and to the same expression written with `float(k)`, which has to be computed by the same kernels to the same bits; ten values a float64 does not hold with the number itself and its two neighbours in the arrays, under a division, a subtraction and a comparison; and eleven wrong versions of the core's part, each one changed line, each fail a test.

**What is still left to NumPy,** each by a test: a whole-number array with a dimension, of one element as of many; float32 and integer arrays beside such a scalar, whose result is float64 or int64 and not a kernel's; an `np.float32` or complex scalar; an `np.timedelta64`, which is an `np.signedinteger` by its class; a Python integer beyond 2**53, as before.

**One thing the tests met that is not this change's.** With `np.nan` and infinities among the values, `0 * a + 0 * b + 0 * c` came out with another NaN than NumPy's in one element of a thousand, for a Python `0.0` as for `np.int64(0)`: `0 * inf` is the processor's NaN, with its sign bit set, `np.nan` has it clear, and which of two NaNs an addition returns is which operand comes first. That is what [the design leaves open](lazy-fusion-design.md#why-the-bits-are-numpys) on x86-64. So on arrays with such values the tests ask for a NaN wherever NumPy has one and NumPy's bits everywhere else, and for the bits of the layer's own run with `float(k)`.

**Before and after.** `pvlib_solar` and `prysm_zernike` at the large size, the quickest of 7 rounds, the ways in turn, one performance core, one BLAS thread, milliseconds. Each pair is the base branch and then this change, minutes apart, with two other sessions working on the machine at load averages of 1.7 to 3.4:

| Program | Take | | NumPy | Installed | Lazy | Fused | Fused against NumPy |
|---|---|---|---|---|---|---|---|
| `pvlib_solar` | 16:01, with the other nine | before | 2,739.41 | 2,715.42 | 2,855.62 | 2,787.30 | 0.98x |
| | 16:08 | after | 2,644.01 | 2,647.46 | 2,775.71 | 2,584.49 | 1.02x |
| | 16:17 | before | 2,797.12 | 2,827.71 | 2,967.68 | 2,857.80 | 0.98x |
| | 16:20 | after | 2,703.90 | 2,708.84 | 2,821.68 | 2,607.62 | 1.04x |
| | 16:23 | before | 2,752.77 | 2,733.65 | 2,865.26 | 2,815.64 | 0.98x |
| | 16:26 | after | 2,693.37 | 2,693.02 | 2,778.34 | 2,574.06 | 1.05x |
| `prysm_zernike` | 16:01 | before | 411.66 | 416.41 | 421.26 | 379.92 | 1.08x |
| | 16:08 | after | 408.28 | 414.17 | 416.05 | 369.13 | 1.11x |
| | 16:17 | before | 408.61 | 406.44 | 417.71 | 375.92 | 1.09x |
| | 16:20 | after | 399.51 | 401.56 | 419.39 | 365.11 | 1.09x |
| | 16:23 | before | 392.21 | 388.87 | 405.01 | 365.85 | 1.07x |
| | 16:26 | after | 387.37 | 389.16 | 399.11 | 352.79 | 1.10x |

What the fused run did, which does not move from take to take:

| Program | | Expressions, all planned | Kernel calls | Nodes inside a program | Given up | New arrays written |
|---|---|---|---|---|---|---|
| `pvlib_solar` | before | 290 | 684 | 638 | 0 | 13,663.5 MB |
| | after | 290 | 873 | 953 | 0 | 11,014.5 MB |
| `prysm_zernike` | before | 45 | 86 | 93 | 0 | 805.3 MB |
| | after | 45 | 98 | 105 | 0 | 704.6 MB |

- **pvlib's fused run is 7% to 9% quicker, and ahead of NumPy where it was behind.** 203 ms, 250 ms and 242 ms less in the three pairs, and 0.98x of NumPy made 1.02x to 1.05x. The eager runs of each later take were 2% to 3.5% quicker as well, which is the machine, so the pairs overstate it; a first pair, taken the other way round by the machine, NumPy 1% slower in the later take, had 2,757 ms made 2,651 ms, 4%, and 0.97x made 1.02x. Against the lazy run of the same take, which the change does not touch, the fused run went from 1.02x to 1.04x to 1.07x to 1.08x in all four.
- **The nutation series is 63 programs of nine operations.** `mine` reads each line as `add(add(add(add(mul(S0,V0),mul(S1,V1)),mul(S2,V2)),mul(S3,V3)),mul(S4,V4))`, where it was the 315 products alone and 63 sums of five arrays. The 315 products are the 315 more nodes inside a program, and 2.6 GB of arrays that are no longer written. A program is five kernel calls a block, one `s * a` and four `a + s * b`: the library has no kernel of the whole line.
- **prysm's is 11 to 13 ms quicker in each pair, 3%,** which is at what two takes differ by here. It has 27 such operations. The first polynomial of a recurrence, `alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)` with `alpha` an `np.int64`, is one program of four operations, six times, with two more nodes inside each. The others are `m * t` and `r ** m`, single operations that feed `np.sin`, `np.cos` and a power the library makes.
- **The other eight of the ten, and the corpus, have nothing for it.** Every float64 element-by-element operation of the thirty-six programs was looked at for such an operand, at the large size: 315 in pvlib, 27 in prysm, and 152 on arrays of one element in the two networks. The eight's counts of expressions, kernel calls and fused nodes are what they were, and the time to plan pvlib's 290 expressions is 10.9 ms before and after. Their times in the take with all ten moved by up to 6% either way, NumPy's with them.

**What was not built.** A kernel for the whole line, or for `a + s * b + t * c`, which `mine` now puts first of the shapes these two programs would gain from: the five calls of a program already keep what passes between them in the cache. An `np.float32` scalar and a Python integer beyond 2**53 beside float64, which have the bits by the same argument: no program of the thirty-six has one. The tables from an ARM64 machine: there is none here, and nothing in the change is the processor's.

### Issue 315: what an inexact kind would be worth

[Issue 315](https://github.com/sofeikov/hotkernels/issues/315) asked whether fusion should have a kind that is not bit-identical, and what it would be worth on pvlib, scikit-image, prysm and Astropy. The design, the measurements and the open decision are in [lazy-inexact-design.md](lazy-inexact-design.md). In short:

- **NumPy computes these functions by glibc's scalar functions here,** with their bits on every element of the programs, at 2.3 to 15.5 ns an element. glibc's vector functions take 0.5 to 4.7 ns on the same arguments and differ from NumPy's on 15% to 64% of them, by up to 4 ULP.
- **With glibc's vector functions in place of NumPy's loops,** an experiment outside the repository, the four programs against NumPy as it is, the four ways in turn:

| Program | Fused, as it is | Vector loops, eager | Vector loops, fused | At most, with the library inside expressions (estimate) |
|---|---|---|---|---|
| `pvlib_solar` | 1.04x | 2.17x | 2.27x | about 2.36x |
| `skimage_delta_e` | 1.11x | 1.42x | 1.65x | about 1.70x |
| `prysm_zernike` | 1.09x | 1.12x | 1.29x | about 1.36x |
| `astropy_models` | 1.15x | 0.74x | 1.07x | |

- **The gain is the library's and not fusion's,** so the design recommends no inexact kind in the layer, whose rule and whose tools' comparison it would break, and loops of the package's under `install(inexact=True)` if the gain is wanted. Astropy loses by glibc's vector `pow` on its fractional exponents.
- **Built, since they need no decision:** `np.radians`, `np.degrees`, `np.deg2rad` and `np.rad2deg` are `mul` of their operand and NumPy's constant, with NumPy's bits on 100 million values. pvlib computes 66 more nodes inside a program and writes 0.55 GB less; in time it is level, at load averages of 3.5 to 5.6, where the conversions weigh at most 2.5%.
- **Shown and not built:** `x % 1.0` has the bits of `x - np.floor(x)` on every value tried, but needs a `floor` the kernels do not have, and perlin-numpy's `grid % 1` is on an array the module declines for its layout.

### `np.empty(...).view(type(x))`, issue 322

**What was built.**

- **`x.view(type(x))` on a lazy array is `x.view()`.** Given to `view`, `LazyArray` means the kind of array the receiver is. `x.view()`, `x.view(np.ndarray)` and `x.view(type(x))` of an array with a dimension are the view `x[...]`, deferred as that index is, so a write through the view is a write into the array and can wait in the graph. Before, each was NumPy's method made at once, and a write into its result was made at once as well. A change of element type, another array type and an array without dimensions are still NumPy's at once.
- **`np.isscalar` of the stand-in answers for the value:** True for a lazy mean, where NumPy's function, which looks at the type alone, said False.
- **`mpl_hillshade` hands Matplotlib the lazy elevation model.** `LightSource.shade` runs to its end on it, lazily and fused, with NumPy's bits and NumPy's layout of the result. Nothing else stopped it: no masked array, no `isinstance`.

**What it measured.** `mpl_hillshade` before and after, one after the other three times at the large size and twice at the small, the quickest of 7 rounds (15 at the small size), the ways in turn, one performance core, one BLAS thread, with the kernels installed. Two other sessions were building and timing on the machine (load 4.1 to 4.4), and NumPy's own time moved by 3% between tables, so each row is to be read against its own NumPy column.

| Size | Taken | NumPy ms | lazy ms | fused ms | fused vs NumPy |
|---|---|---|---|---|---|
| large | before | 332.1, 330.8, 336.0 | 332.6, 330.2, 327.2 | 335.2, 332.1, 324.5 | 0.99x, 1.00x, 1.04x |
| large | after | 326.4, 334.7, 337.0 | 341.1, 331.3, 333.3 | 334.7, 330.1, 331.1 | 0.98x, 1.01x, 1.02x |
| small | before | 4.80, 4.83 | 5.57, 5.57 | 5.22, 5.17 | 0.92x, 0.93x |
| small | after | 4.82, 4.72 | 5.51, 5.51 | 5.28, 5.26 | 0.91x, 0.90x |

- **Level.** The fused run was at NumPy's time before and is at NumPy's time after, and the small size is within 3% of where it was.
- **Ten more operations are deferred, and none of them is fused.** The second table of `fusion_time`, at the large size:

  | | deferred ops | graphs | deferred fused | at once fused | expr. | tiled | kernel calls | fused nodes | given up | lazy MB | fused MB |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | before | 145 | 20 | 19 | 79 | 4 | 4 | 3 | 1 | 2 | 1,073.9 | 973.1 |
  | after | 155 | 21 | 20 | 81 | 4 | 4 | 3 | 1 | 2 | 1,107.4 | 1,006.6 |

- **Three of the ten are the elevation model's:** `vert_exag * elevation`, which `np.gradient` then needs at once, and `data.min()` and `data.max()`, which the colour map's norm needs at once. They were NumPy's calls on a plain array before, outside every count; the 33.5 MB more of new arrays is that product, now counted.
- **Seven are the array of normals.** `np.empty(...).view(np.ndarray)` gave a lazy array before as well, but one made at once, which a write does not wait for. Node by node, the view, the three assignments `normal[..., i] = ...` and `normal /= _vector_magnitude(normal)` are nodes now. With fusion on a view of an array that has its values is NumPy's view, taken at once, and a write into it is made at once ([the design](lazy-fusion-design.md#as-the-graph-is-built)), so the fused run makes those writes where the program wrote them, as before.
- **What they feed is not a kernel's.** The normals are written into every third element, and read back the same way by the length of each normal: arrays that step, which a kernel does not read ([What is not supported](#what-is-not-supported)). The rest of what the elevation model reaches is `np.gradient`, `ndarray.dot` on three dimensions and `np.clip`, made at once, and the colour map's norm, which makes a masked array of the model with `np.ma.array`, a conversion to an `ndarray`, and computes on plain arrays from there.
- **No other program is reached.** Of the corpus's twenty-six and these ten, run lazily and fused at the small size with both new answers watched, only `mpl_hillshade` calls `view` on a lazy array, and none asks the stand-in's `np.isscalar`. So no other program was timed again.

**What was not done.**

- **scikit-image's `regionprops_table` is still handed plain arrays** (`corpus.PLAIN_INPUTS`). It asks `np.isscalar` of each region's mean with its own `numpy`, and that function looks at the type of its argument and calls nothing of it. With the stand-in as the `np` of `skimage.measure._regionprops` as well, the corpus's `skimage_pipeline` runs on lazy images and gives NumPy's bits, lazily and fused; the program was left as it is, a library that keeps its own `numpy`. For the real `numpy.isscalar` to say True, a lazy number would have to be of a class of its own, registered as a number: [the design](lazy-design.md#types-what-changes-for-the-user) says what that takes and why it is not proposed.
- **`a.view(type(x))` on a plain `ndarray` `a` still raises.** It is NumPy's method. Matplotlib's line is reached because `np.empty` of the stand-in makes a lazy array.

### `out=`, `np.clip` and plain operands, issue 320

**What was built.** With fusion on, the C core makes four kinds of call where the stand-in or NumPy hands them to the layer, on the values, where each went through Python ([the design](lazy-design.md#the-c-core)):

- **A ufunc that names its output,** `np.maximum(x, 0, out=x)`, `np.exp(x, out=x)`, when nothing is pending: NumPy's call with the output's memory as `out`. The output may be an array the layer made, one the program handed in, or a plain `ndarray`. Before, only a write the layer could have deferred was made by the core, so an array that came in through `np.asarray` and every plain one went to `array._write`.
- **A plain array updated in place by one of the layer's,** `param += update`: NumPy makes it as `np.add(param, update, out=(param,))` and hands it to the lazy `update`. The same call, with `param` as it is.
- **A ufunc, an operator or a function that gives a new array, on plain arrays:** `np.sqrt(a)`, `np.dot(a, b)` and `np.sum(a, axis=0)` through the stand-in, `a + x` and `x * a` with `x` the layer's. The plain array is its own value. Before, the stand-in's Python made each plain argument lazy, a node, a cell and a lazy array, and handed the call to NumPy, which handed it back. A matrix product of two large arrays, which no kernel makes, is taken whatever its size.
- **`np.clip(x, lo, hi)` and `np.average`,** which the layer has no rule for: `array.NEVER_DEFERRED`, the functions whose call is never a node and whose result is a new array. NumPy's call on the values, at any size. Each call is still counted on its line as a call outside the deferred set, by `graph.fallback`, so `lazy_report()` and the log say what they said.
- **A NumPy scalar among the arguments** (`np.finfo(t).eps`, a library's constants) is a plain value to the stand-in's road. It sent the call to Python before.
- **None of it while a recording listens.** A recording has a node for each array that came in and a line for each write made at once, as before.

One thing was wrong and is fixed. `np.add(a, b, c)` through the stand-in, with the output by position, was taken by the core for a ufunc on three operands and made at once: `c` was written while an operation that had read it still waited, and the call gave a second lazy array on `c`'s memory. The core now takes a ufunc called by name only with as many operands as it has inputs, and the call goes the road of a write.

**What it measured.** Before and after one after the other, on one performance core with one BLAS thread, the ways in turn. Two other sessions were building and timing on the machine, at a load of 1.1 to 3.2.

A call over NumPy's own, on arrays of 200 × 100 with fusion on, the quickest of 7 loops of 5,000:

| Call | NumPy | Plain arrays, before | after | The layer's arrays, before | after |
|---|---|---|---|---|---|
| `np.maximum(x, 0, out=x)` | 4.27 µs | +5.95 µs | +0.27 µs | +4.94 µs | +0.18 µs |
| `np.exp(x, out=x)` | 35.35 µs | +5.57 µs | +0.30 µs | +4.54 µs | +0.20 µs |
| `np.clip(x, lo, hi)` | 4.20 µs | +8.12 µs | +0.88 µs | +7.28 µs | +0.87 µs |
| `x += y`, `y` the layer's | 2.44 µs | +4.29 µs | +0.27 µs | +0.40 µs | +0.40 µs |
| `x + y`, `y` the layer's | 5.78 µs | +0.57 µs | +0.15 µs | +0.11 µs | +0.13 µs |
| `np.sqrt(x)` | 11.59 µs | +1.14 µs | +0.10 µs | +0.11 µs | +0.11 µs |
| `np.dot(s, s)` | 1.68 µs | +1.97 µs | +0.12 µs | +0.65 µs | +0.66 µs |
| `np.sum(x, axis=0)` | 4.42 µs | +1.60 µs | +0.20 µs | +0.64 µs | +0.79 µs |

- **The three roads cost what a call on the layer's own arrays costs,** 0.1 to 0.3 µs, where they cost 1.1 to 8.1 µs.
- **`np.clip` is at 0.9 µs, and 0.7 µs of it is the count.** The call is outside the deferred set, and each one is counted on its line for `lazy_report()` and the log, by the Python function that counts every such call. With that function replaced by one that does nothing the call costs 0.24 µs over NumPy's.
- **The last two rows of the layer's own arrays did not change, and are not the layer's time.** The script makes its lazy arrays of a copy of the data, and NumPy's own `np.dot` and `np.sum` take 0.5 µs more or less by where an array lies: in another run of the same script NumPy's `np.dot` took 2.19 µs and the stand-in's 0.15 µs over it.

`sklearn_mlp`, with the kernels installed, the quickest of 7 rounds at the large size and of 15 at the small, each line one run of the tool:

| Size | Taken | NumPy ms | fused ms | fused vs NumPy | Calls and writes the core made at once |
|---|---|---|---|---|---|
| large | before | 226.5, 226.1, 225.7 | 255.5, 257.0, 253.9 | 0.89x, 0.88x, 0.89x | 6,921 |
| large | after | 226.0, 228.5, 240.1 | 238.6, 240.0, 250.3 | 0.95x, 0.95x, 0.96x | 7,971 |
| small | before | 5.66, 5.57 | 8.59, 8.48 | 0.66x, 0.66x | 1,379 |
| small | after | 5.66, 5.62 | 5.76, 5.66 | 0.98x, 0.99x | 1,579 |

- **17 ms of the 30 ms it was behind by are gone at the large size, and it is level with NumPy at the small.** Against the eager program with the kernels installed it is 0.93x where it was 0.87x, and 0.83x where it was 0.55x at the small size: `install()` gains 1.17x there, and the lazy array around each value is what is left.
- **1,800 of its calls went a Python road, and 150 do now:** `param += update` for the large weight matrix, whose `update` is an expression that waits. The planner computes it for the write, 27 ms in 150 computations, and the 4.5 µs of Python around each is what remains of the road.
- **What it is still behind by at the large size, 12 ms, is not these calls.** Adam's updates are 37 ms of NumPy's passes and 27 ms of kernels fused ([`sklearn_mlp`](#sklearn_mlp-088x)); around them the fused run writes and plans 450 expressions and makes 8,000 calls at once. Where the 12 ms are among those was not looked into.

The other nine, at the large size with the kernels installed, the quickest of 7 rounds, each pair of figures NumPy's time and the fused run's:

| Program | Before, ms | | After, ms | |
|---|---|---|---|---|
| `perlin_noise` | 1,991.0, 1,995.1 | 1.00x | 2,124.0, 2,093.7 | 1.01x |
| `dlfs_convnet` | 805.6, 755.4 | 1.07x | 834.5, 772.0 | 1.08x |
| `mpl_hillshade` | 286.9, 286.4 | 1.00x | 305.0, 302.2 | 1.01x |
| `pvlib_solar` | 2,793.0, 2,816.9 | 0.99x | 2,640.7, 2,707.9 | 0.98x |
| `skimage_delta_e` | 398.1, 365.7 | 1.09x | 402.4, 371.4 | 1.08x |
| `salib_sobol` | 1,076.3, 894.8 | 1.20x | 1,053.6, 906.5 | 1.16x |
| `prysm_zernike` | 358.6, 328.4 | 1.09x | 360.1, 328.2 | 1.10x |
| `astropy_models` | 203.1, 179.0 | 1.13x | 203.0, 180.3 | 1.13x |
| `speech_mfcc` | 70.9, 72.7 | 0.98x | 70.9, 72.6 | 0.98x |

- **Level, all nine.** None makes these calls more than a few tens of times in a run. NumPy's own time moved by up to 7% from one table to the next, so each ratio is to be read against the NumPy time beside it.
- **Three tables had a figure that another take did not.** `pvlib_solar` after, 2,911 ms fused for NumPy's 2,658, 0.91x, in a table where the same fused run with nothing installed took 2,735; `salib_sobol` after, 1,045 ms for 1,128, 1.08x, in one take of three whose other two were 945 for 1,111 and 907 for 1,054; and `prysm_zernike` and `speech_mfcc` 4% better in one take and level in the next. The machine was at a load of 2.4 to 3.2 for each. The rows above are the repeats.

Six programs of the corpus, at the large size with the kernels installed, the fused run in milliseconds, two takes each (three of `flame`):

| Program | Before | After | |
|---|---|---|---|
| `make_decision` | 99.1, 99.4 | 99.7, 100.7 | level |
| `char_rnn` | 22.9, 23.5 | 22.4, 22.5 | 0.97x of the time: level |
| `perceptron` | 38.0, 37.6 | 37.5, 37.4 | level |
| `flame` | 75.4, 73.9, 74.1 | 74.3, 74.1, 74.4 | level |
| `cavity_flow` | 63.9, 64.2 | 64.5, 66.7 | level |
| `hdiff` | 102.0, 100.2 | 99.6, 99.8 | level |

- **Nothing in the corpus moved.** Its four programs on small arrays compute on the layer's own arrays, by operators and assignments, and make none of these calls but `np.clip(g, -5, 5, out=g)` in the recurrent network, which was the core's already. They are behind the eager program by what a lazy array costs to make ([issue 310](https://github.com/sofeikov/hotkernels/issues/310)), which this does not touch.
- **`flame` took 78.5 and 77.5 ms after in the first two takes, 4% and 3% more than before,** and 74.3, 74.1 and 74.4 ms in three more taken with it alone, against 75.4, 73.9 and 74.1 before.

**What was not done.**

- **A write while something is pending still computes first, in Python.** `param += update` with `update` an expression that waits, 150 of the perceptron's 600 updates, goes to `array._write`, which has the planner compute what is pending: 4.5 µs of Python around a computation of 180 µs. Computing is `evaluate`'s, and the core does not start it for a write.
- **`np.clip` is still not in the deferred set.** It is made at once and ends an expression, where it is a maximum and a minimum that a kernel could take ([the design](lazy-fusion-design.md#what-is-open-in-the-layer-and-is-not-fusion)). Its count for `lazy_report()` is made by Python's `graph.fallback`, 0.7 µs of the 0.9 µs a call costs over NumPy's. The count was not written a second time in C for the 300 calls of the one program that makes them.
- **Other calls outside the deferred set go the Python road as before:** `np.pad`, `np.dstack`, `np.bincount`, `np.ptp`, `x.flatten()` and a ufunc with `where=` are what the ten programs make, 20 to 60 times each. Each would have to be shown to give an array that shares no memory with an argument before it could join `NEVER_DEFERRED`.
- **A library that calls its own `numpy.clip` on a lazy array** reaches the layer through NumPy's protocol, and is made by Python. The road built here is where the stand-in is called.
- **Nothing was measured on ARM64.** There is no such machine here. The roads are the same C on both.

### Memory a graph holds, issue 321

**What was measured first.** The six programs that were slower node by node than eagerly, with the kernels installed, the ways in turn on one performance core: the quickest of 5 runs, the minor page faults of that run (`resource.getrusage`), and the most bytes NumPy had allocated at once (`tracemalloc`). Then the same with glibc told to keep the memory it is given back (`MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` and `MALLOC_TOP_PAD_` set high), which takes new pages out of the time. Load 1.6 to 2.6.

| Program | Eager ms | Lazy ms | Faults, eager | Faults, lazy | Most allocated, eager | lazy | Lazy behind by, glibc keeping memory |
|---|---|---|---|---|---|---|---|
| `perlin_noise` | 2,000.7 | 2,314.0 | 70,277 | 113,237 | 541 MB | 2,185 MB | 87 ms of 313 |
| `pvlib_solar` | 2,748.7 | 2,881.4 | 24,368 | 18,541 | 353 MB | 169 MB | 142 ms of 133 |
| `salib_sobol` | 1,037.0 | 1,449.9 | 39,378 | 47,172 | 318 MB | 318 MB | 391 ms of 413 |
| `astropy_models` | 194.9 | 219.1 | 3,577 | 3,062 | 67 MB | 59 MB | 32 ms of 24 |
| `speech_mfcc` | 67.8 | 89.3 | 4,932 | 4,932 | 165 MB | 146 MB | 20 ms of 21 |

- **A node's value was already released at its last reader.** The list of a computation lets go of each node as it is taken, a computed node lets go of its operands, and nothing else holds an intermediate the program has no name for: on four of the five the lazy run holds as much memory at its peak as the eager one, or less.
- **New pages are two thirds of the noise program's loss, and none of the others'.** Its five octaves are one graph until the last line, and each pending octave holds what it reads, the four arrays `np.dstack` made and the repeated gradients, 400 MB an octave; eagerly each is read and freed at once. With glibc keeping memory the lazy run is 87 ms behind, where it was 313.
- **SALib and the speech features lost their time to a call made twice.** `cProfile` of the lazy run: 355 ms of SALib's 413 in `rules._indexed_shape`, the shape rule of `A[r]`, and 15 ms of the speech features' 21. The rule asked NumPy for the shape by indexing an array of an element type of no size, and NumPy walks every position to do it: 8.7 ms for SALib's table of 3.3 million positions, where `A[r]` itself takes 2.5 ms.
- **The rest is the memory of temporaries, and not new pages.** pvlib, Astropy's models and the noise program were as far behind with glibc keeping memory. A Python prototype that called each element-by-element node with `out=` the memory of an operand nothing else read took the node-by-node run of pvlib from 3,365 ms to 2,932 ms, Astropy's from 272 ms to 200 ms and the noise program's from 2,662 ms to 2,489 ms, with NumPy's bits: NumPy writes `a * b + c` into the memory of `a * b`, and its next array of that size is pages that are in the cache.

**What was built.**

- **The shape of `A[r]` without the walk** ([the design](lazy-design.md#shape-and-element-type-without-computing)). An index of arrays of integers holding 4,096 positions or more, with no mask and no list in it, is asked twice, on stand-ins of one and of two positions along each dimension of the arrays' broadcast shape; the dimensions where the two answers differ are where that shape goes. Whether an array that has its values is in bounds is asked of its least and largest element, of its own type. Every other index, and every index NumPy refuses there, is asked the whole way as before, so an error is NumPy's own walk's, with the eager line's message.
- **A call writes into the memory of an operand nothing else reads** ([the design](lazy-design.md#computing)). A ufunc of one output, element by element, with no keyword and every array operand in C order, is called with `out=` the value of an operand that is a computed node no lazy array stands for and no other node reads, an `ndarray` of the result's shape and type that owns its memory, has no view alive and holds 256 KiB or more, NumPy's own threshold. References are counted, as for a write made in place. Python's `evaluate._lendable` is the definition and the C core's road for a plain node does the same; core version 38. In one run the noise program lends 55 times, pvlib 1,439, the cavity flow 1,760, the perceptron 900; fused, pvlib 791 among the nodes left to NumPy, and the corpus's stencils none or a few.
- **Not built: a pool of blocks for intermediates.** Within a run the lazy layer frees an intermediate where the eager program would; what the noise program faults on is memory held at once, not memory given back and asked for again. Across runs a pool would hide what glibc's trimming costs the eager program as well, which is the timing tool's matter ([issue 326](https://github.com/sofeikov/hotkernels/issues/326)) and not the layer's.

**What it measured.** `fusion_time --plain --runs 7` at the large size, with the kernels installed, the ten programs and six of the corpus, the base of this change and this change one after the other, twice, each from its own working tree. Each figure is the lazy or the fused run against the eager program with the kernels installed in the same table, before → after, the first take and the second; then the second take's lazy and fused runs in milliseconds. Load 3.5 to 4.5 for the first take, 2.1 to 3.9 for the second. In the first take NumPy's own time fell by a third between the two tables of `regression` and `veros_eos`, so read each ratio against its own table.

| Program | Lazy vs installed | Fused vs installed | Lazy ms, take 2 | Fused ms, take 2 |
|---|---|---|---|---|
| `perlin_noise` | 0.92x → 0.96x, 0.93x → 0.95x | 1.00x → 0.99x, 1.00x → 0.99x | 2,091.2 → 2,068.8 | 1,949.6 → 1,987.7 |
| `pvlib_solar` | 0.93x → 1.03x, 0.95x → 1.05x | 1.02x → 1.02x, 0.99x → 1.03x | 2,833.3 → 2,600.5 | 2,722.4 → 2,650.6 |
| `salib_sobol` | 0.73x → 0.88x, 0.65x → 0.89x | 1.09x → 1.01x, 1.03x → 1.03x | 1,119.4 → 789.1 | 703.9 → 688.1 |
| `astropy_models` | 1.03x → 1.10x, 1.10x → 1.10x | 1.14x → 1.15x, 1.16x → 1.14x | 173.5 → 179.6 | 163.4 → 172.0 |
| `speech_mfcc` | 0.76x → 0.93x, 0.76x → 0.92x | 1.01x → 1.02x, 1.03x → 1.05x | 82.4 → 64.8 | 60.7 → 57.2 |
| `sklearn_mlp` | 0.78x → 0.81x, 0.81x → 0.82x | 0.91x → 0.93x, 0.93x → 0.92x | 273.6 → 270.4 | 238.1 → 239.1 |
| `dlfs_convnet` | 1.01x → 1.02x, 1.00x → 1.04x | 1.06x → 1.00x, 0.99x → 1.01x | 692.2 → 680.4 | 702.7 → 697.0 |
| `mpl_hillshade` | 0.99x → 1.07x, 0.98x → 1.07x | 0.99x → 1.04x, 0.99x → 1.02x | 266.0 → 252.0 | 265.8 → 263.4 |
| `skimage_delta_e` | 1.00x → 1.04x, 1.02x → 1.05x | 1.09x → 1.08x, 1.08x → 1.07x | 373.1 → 364.7 | 350.8 → 355.8 |
| `prysm_zernike` | 0.93x → 1.12x, 0.93x → 1.11x | 1.14x → 1.15x, 1.08x → 1.15x | 365.3 → 308.5 | 313.1 → 298.5 |
| `cavity_flow` | 0.81x → 0.94x, 0.80x → 0.93x | 3.12x → 3.08x, 3.10x → 3.09x | 238.5 → 209.3 | 61.3 → 63.2 |
| `hdiff` | 0.80x → 0.88x, 0.80x → 0.88x | 1.99x → 1.98x, 1.99x → 2.00x | 228.9 → 208.8 | 91.4 → 91.9 |
| `veros_eos` | 0.93x → 1.00x, 0.82x → 1.03x | 1.99x → 1.88x, 1.87x → 1.87x | 232.5 → 185.0 | 101.8 → 102.1 |
| `veros_isoneutral` | 0.94x → 1.04x, 0.94x → 1.03x | 1.31x → 1.35x, 1.31x → 1.29x | 418.1 → 383.5 | 301.4 → 304.4 |
| `black_scholes` | 1.01x → 1.15x, 1.02x → 1.13x | 1.17x → 1.24x, 1.16x → 1.22x | 88.4 → 80.9 | 77.3 → 75.0 |
| `regression` | 0.96x → 1.01x, 0.98x → 1.01x | 1.15x → 1.01x, 1.01x → 1.03x | 58.3 → 53.8 | 56.4 → 52.7 |

- **Node by node, ten of the sixteen gained more than 3% in both takes, and none lost.** SALib from 0.65x of the eager program to 0.89x, the speech features from 0.76x to 0.92x, both from the shape of `A[r]`; pvlib from 0.95x to 1.05x, prysm from 0.93x to 1.11x, the hill shading from 0.98x to 1.07x, the corpus's cavity flow from 0.80x to 0.93x and `veros_eos` from 0.82x to 1.03x, from lending memory. The noise program gained 4% and 2%, the perceptron, the network and Astropy's models 1% to 4% in one take and nothing in the other.
- **Fused, three gained and none lost.** `black_scholes` 1.16x to 1.22x and pvlib 0.99x to 1.03x in the second take, with the same direction in the first, and prysm 1.08x to 1.15x in the second take, where the first had it level. The rest are within 3% in at least one take and the other has them level or in the other direction: `veros_eos` and `regression` read 1.99x → 1.88x and 1.15x → 1.01x in the first take only because the eager runs of the two tables took a third less time in the second; their fused runs took 154 ms → 108 ms and 72 ms → 61 ms.
- **The issue's "Done when".** Node by node the noise program is 0.95x of the eager program and SALib 0.89x. What SALib is still behind by, 84 ms in the second take, is not new pages (it has as many faults and as much memory at its peak as the eager program) and was not looked into further. The noise program's is the 2.2 GB its pending octaves hold, and the faults of it: 84,197 a run where the eager program has 70,277, from 113,237 before.
- **Nothing was measured on ARM64.** There is no such machine here.

**What was not done.**

- **The memory pending work holds.** Lowering it means computing a graph before the program asks, when the operands its pending nodes hold pass a size: the noise program's octave by octave. That would cut the noise program's fused programs, the largest of which spans the whole sum of octaves, and is a decision about fusion, not about memory.
- **Writes through an index, `x[r] = v`,** still have the shape and the bounds checked by NumPy's walk on stand-ins without memory. None of the sixteen programs makes such a write with a large table.
- **A ufunc whose operands lie in another order than C's,** or that is called with a keyword, makes a new array as before: NumPy's result could lie otherwise than the operand lent.

### Arrays walked in the order of memory

Measured as the tables above were: one performance core, one BLAS thread, the ways in turn, the quickest of 7 rounds at the large size, before and after taken within the hour of each other on a machine other sessions were using, so a difference under 3% is none.

[Issue 316](https://github.com/sofeikov/hotkernels/issues/316). The module that runs a program sorts the axes of the array it fills by stride, the largest outermost, as NumPy's iterator sorts the operands of an operation, and reads every array in that order. Where the arrays of an expression all lie in one order other than C order, the result is new memory laid out as they are, which is the layout NumPy gives it. [The design](lazy-fusion-design.md#computing-a-group) has the rule.

- **`perlin_noise` gives up none, and is 1.13x of NumPy where it was level.** The interpolant's five programs run, six kernel calls each, block by block over the 8.4 million elements as one run. Timed twice each way:

| `perlin_noise` | NumPy ms | Fused ms | Of which computing | vs NumPy | Given up | Kernel calls | Nodes inside a program | New arrays written |
|---|---|---|---|---|---|---|---|---|
| Before | 2,012.50 | 2,027.08 | 1,373.59 | 0.99x | 5 | 60 | 74 | 5,033 MB |
| Before, with four other programs | 2,064.99 | 2,163.95 | 1,492.36 | 0.95x | | | | |
| After | 2,072.15 | 1,810.62 | 1,170.49 | 1.14x | 0 | 90 | 104 | 3,020 MB |
| After, with four other programs | 1,998.22 | 1,773.33 | 1,139.76 | 1.13x | | | | |

- **The fused run is 220 to 390 ms shorter,** for the 300 ms of NumPy's passes the study counted in the interpolant. What the kernels take for it was not timed apart.
- **The other nine do the same as before and take the same time.** Their counts of programs, kernel calls and programs given up are the ones in the tables above, to the last figure. Fused against plain NumPy, before and after:

| Program | Before: NumPy ms | Fused ms | vs NumPy | After: NumPy ms | Fused ms | vs NumPy |
|---|---|---|---|---|---|---|
| `perlin_noise` | 2,064.99 | 2,163.95 | 0.95x | 1,998.22 | 1,773.33 | 1.13x |
| `dlfs_convnet` | 876.30 | 808.95 | 1.08x | 799.28 | 755.53 | 1.06x |
| `mpl_hillshade` | 303.82 | 303.28 | 1.00x | 285.98 | 284.66 | 1.00x |
| `pvlib_solar` | 2,718.65 | 2,795.82 | 0.97x | 2,677.26 | 2,737.17 | 0.98x |
| `skimage_delta_e` | 400.30 | 367.94 | 1.09x | 389.31 | 358.74 | 1.09x |
| `sklearn_mlp` | 231.13 | 264.92 | 0.87x | 227.16 | 258.31 | 0.88x |
| `salib_sobol` | 1,077.06 | 1,056.19 | 1.02x | 1,068.40 | 1,063.55 | 1.00x |
| `prysm_zernike` | 344.77 | 319.48 | 1.08x | 344.50 | 317.05 | 1.09x |
| `astropy_models` | 195.26 | 170.89 | 1.14x | 193.01 | 168.37 | 1.15x |
| `speech_mfcc` | 71.38 | 68.35 | 1.04x | 68.62 | 65.91 | 1.04x |

- **The "before" was timed under a higher load,** 3.3 to 4.7 against 1.6 to 3.9, and in its first five programs NumPy's own times are 2% to 10% longer than in the "after". The ratio is what compares.
- **Nothing of the convolutional network is taken now.** It gave up no program before and gives up none now, and its 24 programs, 72 kernel calls and 72 nodes inside a program are what they were. Its arrays left out of order by `transpose(0, 3, 1, 2)` are read by a comparison (`x <= 0`), by `x.copy()`, which is in C order again, and by the pooling layer's copies of slices: no expression of arithmetic is written on one.
- **The corpus's programs with transposed or stepping arrays did not move.** Their counts are the same to the last figure. The horizontal diffusion read 2.11x and 2.01x in the first pair of runs, on arrays of 85 MB with other sessions running; timed again, before and after and after: 2.06x, 2.08x, 2.09x.

| Program | Before: NumPy ms | Fused ms | vs NumPy | After: NumPy ms | Fused ms | vs NumPy |
|---|---|---|---|---|---|---|
| `veros_isoneutral` | 400.67 | 307.22 | 1.30x | 387.35 | 297.88 | 1.30x |
| `veros_tke` | 200.59 | 145.66 | 1.38x | 189.90 | 136.12 | 1.40x |
| `particle_filter` | 425.30 | 383.57 | 1.11x | 412.28 | 369.24 | 1.12x |
| `cavity_flow` | 196.32 | 63.40 | 3.10x | 185.67 | 59.70 | 3.11x |
| `hdiff` | 195.09 | 92.47 | 2.11x | 201.31 | 100.26 | 2.01x |
| `hdiff`, again | 210.54 | 102.25 | 2.06x | 212.10, 211.80 | 101.86, 101.57 | 2.08x, 2.09x |

- **A sum of an expression on such arrays is a kernel's too.** NumPy adds an array that lies in one run in the order of memory whichever of its axes is the slowest, so `np.sum(x.T ** 2)` is the sum-of-squares kernel on the same memory. None of the ten programs writes one.
- **What is still NumPy's.** Arrays that lie in two orders (`a.T + b`), for which NumPy's own result is in C order. An array out of order beside a row or a column, and one with an axis of length one: there the layout of NumPy's result depends on each operation of the expression and on which of NumPy's two loops it took, and the module sees only the arrays. A reversed array, as before.
- **Found on the way, and not changed:** `np.zeros_like(x)`, `np.ones_like(x)` and `np.empty_like(x)` of a lazy array give an array in C order where NumPy keeps the order of `x`. `np.zeros_like(a.T)` has strides (8, 320) in NumPy and (240, 8) through the stand-in, with fusion or without. The layer makes such an array from the shape and element type alone ([lazy-design.md](lazy-design.md)).
- **Found as well, and not changed: with operands in two orders the layer's layout can differ from the eager program's.** NumPy computes `(a * 2.0) - b` by writing the subtraction into the temporary it holds, for arrays of 256 KB or more, so the result keeps the temporary's layout; with the temporary held by a name it allocates, and where the two operands disagree the new array is in C order. A graph holds every intermediate. For an `a` of shape (17, 17, 17, 40) with strides (105944, 328, 6232, 8) and a `b` with strides (152, 2736, 8, 98496), the line gives strides (92480, 320, 5440, 8) in NumPy and (92480, 5440, 320, 8) through the layer, with fusion or without, before this change and after; the bits are the same. Found by 3,700 random layouts, of which these were the only two that differed; no array of the ten programs is laid out so. Where every array lies in one order, which is all a kernel takes, both ways give that order.
- **One allocation had to be ordered.** The memory tried first in C order is given back before memory in the arrays' order is asked for. With both held at once, one multiplication on 200,000 elements took 308 µs where NumPy takes 85 µs; in order it takes 101 µs. Why was not measured; the likeliest reason is that the allocator returns two such blocks to the system when both are freed, and the next run starts on new pages.

### Rows of three or four with a value for each row

[Issue 317](https://github.com/sofeikov/hotkernels/issues/317). Three things were built, and the third is what made the first two count. [The design](lazy-fusion-design.md#computing-a-group) has the rules.

- **A program of several steps runs on rows shorter than 16 with one value for each row.** The value is read as an array and spread over its row inside a block, by a loop of the module's over many rows at once; the kernels read the other arrays where they lie. A row walked with its value held was the other way: it is a call of every step for each row, which the module declined already for rows that short, and for a program of one kernel it takes 16.3 ms on four million elements in rows of two where the spread value takes 3.8 ms and NumPy 11.4 ms.
- **A kernel that selects by a threshold,** `fuse_where_le_f64`, `np.where(a <= s, b, c)`: the two passes of the comparison and the selection as the last step of a program, with no mask in memory. It is written for x86-64, where its comparison raises no flag for a NaN, and is its C reference on ARM64, which was not at hand.
- **The comparison waits for its selection.** This is why `where(le(V0,S0),mul(S1,V1),sub(S2,mul(S3,sub(S4,V2))))` was not one program. The tool that mines expressions reads the lazy run, where every operation is a node. In the fused run a comparison of an array that has its value is made at once, as every call is that no kernel could make, so `rgb <= 0.5` was an array of masks before `np.where` was written, and the selection read three arrays from memory. Now `x <= s` on a float64 array of 32,768 elements or more and a number waits, as arithmetic does, because a kernel selects by it. Of the thirty-six programs four write one, 22 in all: the convolutional network and pvlib wrote theirs as nodes already, since what they compare is pending, and Matplotlib's and prysm's are new.

The blend by itself, `rgb` of 2,048 × 2,048 rows of four and of three, nothing installed, the quickest of 7:

| The overlay blend | NumPy ms | Fused before | With the value spread | And the selection in the program |
|---|---|---|---|---|
| Rows of four | 144.1 | 139.7, 1.03x | 118.3, 1.25x | 43.3, 3.33x |
| Rows of three | 112.1 | 108.1, 1.06x | 95.1, 1.20x | 33.8, 3.32x |

Each ratio is against NumPy as timed in the same run, 144 to 148 ms and 112 to 115 ms. The middle column was taken with the first version of the fill, a row at a time, which the loop over many rows has made 9 ms shorter on rows of four since.

- **`mpl_hillshade` computes its blend by one program of five kernel calls, and is 1.30x of NumPy where it was level.** The program is six of the blend's nine operations. `2 * intensity` is one operation on an array in one run, NumPy's call, and `2 * (1 - intensity)` is a program of its own, as before. Timed twice each way:

| `mpl_hillshade` | NumPy ms | Fused ms | Of which computing | vs NumPy | Given up | Kernel calls | Nodes inside a program | New arrays written |
|---|---|---|---|---|---|---|---|---|
| Before | 320.38 | 318.32 | 167.00 | 1.01x | 2 | 3 | 1 | 973 MB |
| Before, with two other programs | 343.45 | 343.84 | 188.64 | 1.00x | | | | |
| After | 334.56 | 256.60 | 107.44 | 1.30x | 1 | 7 | 6 | 436 MB |
| After, with two other programs | 348.32 | 262.41 | 106.64 | 1.33x | | | | |

- **The fused run is 62 to 81 ms shorter,** for the 101 ms of NumPy's passes the study counted in the blend.
- **The other nine do the same as before.** Their counts of programs, kernel calls and programs given up are the ones before, to the last figure, but for prysm's: the comparison of its circular mask, `rho <= radius`, waits now, so three more of its operations are nodes, three fewer are made at once, and one write is made by its node. It takes the same time, 367.18 ms before and 366.78 ms after, timed one after the other. Fused against plain NumPy:

| Program | Before: NumPy ms | Fused ms | vs NumPy | After: NumPy ms | Fused ms | vs NumPy |
|---|---|---|---|---|---|---|
| `perlin_noise` | 2,133.11 | 1,894.20 | 1.13x | 2,239.97 | 1,931.76 | 1.16x |
| `dlfs_convnet` | 890.89 | 834.06 | 1.07x | 864.86 | 811.46 | 1.07x |
| `mpl_hillshade` | 320.38 | 318.32 | 1.01x | 334.56 | 256.60 | 1.30x |
| `pvlib_solar` | 3,092.63 | 3,005.94 | 1.03x | 3,236.63 | 3,107.78 | 1.04x |
| `skimage_delta_e` | 416.66 | 375.03 | 1.11x | 409.10 | 386.54 | 1.06x |
| `sklearn_mlp` | 242.45 | 279.11 | 0.87x | 235.12 | 268.19 | 0.88x |
| `salib_sobol` | 1,031.60 | 1,067.56 | 0.97x | 1,012.63 | 986.77 | 1.03x |
| `prysm_zernike` | 405.18 | 361.06 | 1.12x | 348.93 | 320.23 | 1.09x |
| `astropy_models` | 214.50 | 189.30 | 1.13x | 194.58 | 171.39 | 1.14x |
| `speech_mfcc` | 68.04 | 70.82 | 0.96x | 61.11 | 62.71 | 0.97x |
| `skimage_delta_e`, again | 416.01 | 386.04 | 1.08x | 418.65 | 391.08 | 1.07x |
| `prysm_zernike`, again | 408.68 | 367.18 | 1.11x | 406.92 | 366.78 | 1.11x |

- **The machine was busier for the "before",** a load of 3.6 to 4.9 against 2.6 to 2.7, with two other sessions timing, and NumPy's own times differ by up to 14% between the two runs. Three ratios moved by 3% or more with nothing changed in what the programs do, scikit-image's and prysm's down and SALib's up. The first two were timed again, before and after one behind the other at a load of 3.7 to 3.9: level. SALib's was not.
- **The corpus's programs on short rows did not move.** Their counts are the same to the last figure, the particle filter's, whose rows of two and four the short-row walks were built for, among them. The naive azimuthal integration is the corpus's program of masks: its comparisons are `r1 <= radius`, a number and then an array, which no kernel selects by, so it writes no node, as before. Loads of 3.3 to 3.5 before and 2.4 to 3.4 after:

| Program | Before: NumPy ms | Fused ms | vs NumPy | After: NumPy ms | Fused ms | vs NumPy |
|---|---|---|---|---|---|---|
| `particle_filter` | 412.38 | 359.54 | 1.15x | 440.73 | 385.15 | 1.14x |
| `veros_isoneutral` | 469.48 | 349.36 | 1.34x | 454.19 | 343.93 | 1.32x |
| `veros_tke` | 254.54 | 190.79 | 1.33x | 229.43 | 164.26 | 1.40x |
| `azimint_naive` | 74.37 | 77.97 | 0.95x | 71.01 | 78.95 | 0.90x |

- **What is still NumPy's on such rows.** A program of one kernel with a value for each row is a call for each row, as before, though spread it would be 2 to 4 times as fast on rows of two to four: its first program runs, so the second is not asked. Rows cut out of wider ones, `rgba[..., :3]`, beside a value for each row, and a result that is such rows: the module would copy them a row of three at a time. A value for each row that is not float64, a mask or whole numbers, ends the expression, as any such array does.
- **The length of the normals, 24 ms, is still given up, and for its last axis.** `sum_sq += arr[..., i, np.newaxis] ** 2` reads three arrays of shape (2048, 2048, 1), each a column of the normals at a step of three elements. The module takes the axis of length one for the rows, so a row is one element, and declines. Written without that axis, `arr[..., i]`, the same expression is gathered and runs: 14.6 ms for NumPy's 33.3 ms, where with the axis the fused run takes 45.5 ms for NumPy's 40.7 ms. Not built here: it is arrays that step. It would be about 25 ms more of this program.
- **A NaN in the image does not cost the program.** A processor's ordered comparison raises the invalid-operation flag for a NaN, NumPy reports nothing for one, and the layer drops a result whose kernel raised a flag. The kernel's x86-64 loop compares quietly, so the result is kept.

## What is open

Each of these is a measurement above and a decision not taken here, with the issue it is parked in.

- **A mathematics library's functions under fusion,** [issue 315](https://github.com/sofeikov/hotkernels/issues/315). Measured ([above](#issue-315-what-an-inexact-kind-would-be-worth)) and written up in [lazy-inexact-design.md](lazy-inexact-design.md), whose decision is the user's: it recommends no inexact kind in the layer, and loops of the package's under `install(inexact=True)` if the gain is wanted. `x % 1.0` as `x - np.floor(x)` has NumPy's bits and waits for a `floor` among the kernels' operations and for [issue 316](https://github.com/sofeikov/hotkernels/issues/316); it is in [issue 222](https://github.com/sofeikov/hotkernels/issues/222), which was open for the remainder already.
- **Walking a program's arrays in the order of memory,** [issue 316](https://github.com/sofeikov/hotkernels/issues/316), is built: [What was built on it](#what-was-built-on-it). What it left to NumPy is open: arrays out of order beside a row or a column, and with an axis of length one.
- **Rows of three or four with a value for each row, in a program of several steps,** [issue 317](https://github.com/sofeikov/hotkernels/issues/317), is built: [What was built on it](#what-was-built-on-it). What it left: a program of one kernel on such rows is still a call for each row; rows cut out of wider ones (`rgba[..., :3]`) beside a value for each row are NumPy's; and an array whose last axis has length one, `arr[..., i, np.newaxis]`, is declined where the same column without that axis is gathered.
- **Whole-number NumPy scalars as numbers,** [issue 318](https://github.com/sofeikov/hotkernels/issues/318). Built ([above](#issue-318-a-whole-number-of-numpys-is-a-number-of-an-expression)). Left as they were: an `np.float32` scalar and a Python integer beyond 2**53 beside float64, which no program of the corpus or of these ten has, and a kernel for the whole line of the nutation series.
- **A sum of a product along a short last axis,** [issue 319](https://github.com/sofeikov/hotkernels/issues/319).
- **`out=`, `np.clip` and plain operands on the core's quick road,** [issue 320](https://github.com/sofeikov/hotkernels/issues/320): built ([above](#out-npclip-and-plain-operands-issue-320)). What is left is a write while something is pending, which computes first in Python, and the calls outside the deferred set other than `np.clip` and `np.average`.
- **Memory a graph holds,** [issue 321](https://github.com/sofeikov/hotkernels/issues/321): built ([above](#memory-a-graph-holds-issue-321)). A node's value was released at its last reader already; a call now writes into an operand's memory as NumPy's do, and the shape of `A[r]` no longer walks the table. What is left is the memory pending work holds, which takes the noise program to 2.2 GB, and which only computing earlier, at a cost to fusion, would lower.
- **A library that keeps the type of what it is given,** [issue 322](https://github.com/sofeikov/hotkernels/issues/322): built for `x.view(type(x))` and for `np.isscalar` of the stand-in ([above](#npemptyviewtypex-issue-322)). What is left is a library that asks with its own `numpy`, which is handed plain arrays.
- **The design's ARM64 tables taken again in turn,** [issue 323](https://github.com/sofeikov/hotkernels/issues/323). The tools time in turn by default now and the x86-64 tables were taken again ([below](#issue-323-the-ways-timed-in-turn-by-default)); the tables from the Apple M1 Pro were not, for want of the machine.
- **Which state of the allocator a program is timed in, and whether a fused run should keep the memory of its results,** [issue 326](https://github.com/sofeikov/hotkernels/issues/326): new pages from the system are a quarter of the equation of state's time.
- **Taking by a table of positions,** which is half of SALib's analysis: [issue 224](https://github.com/sofeikov/hotkernels/issues/224), open already for `a[idx]`.
- **Kernels that read arrays at a step.** The design left them for two programs; three more have such arrays here, for 30 ms. No issue: it weighs too little in these to add to what the design says.
