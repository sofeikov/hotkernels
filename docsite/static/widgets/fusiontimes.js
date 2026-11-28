// The twenty-six programs of the corpus, each run four ways.
//
// One row per program. The filled dot is the fused run with hotkernels
// installed, against the program on plain NumPy: the figure the lazy layer is
// judged by. The ring is the eager program with the same kernels installed,
// what the package gives with no graph. The segment between them is what the
// graph adds, green where the fused run is ahead by 3% or more and red where
// it is behind by as much. A diamond, at the large size only, is the fused run
// with nothing installed. Selecting a row shows the program's four times, how
// the lazy and the fused run split into building, planning and computing, and
// what its graphs were.
//
// Data: the tables of docs/lazy-fusion-design.md, "What was measured" ("The
// programs, four ways" and "Small sizes"), from
//   uv run --group apps --group workloads python -m prototypes.lazy.programs.fusion_time --size large --plain
// and the same at --size small, on an Apple M1 Pro, NumPy 2.5.5, CPython 3.13,
// the quickest of 5 runs. The numbers are the tables' own, copied by a script
// that reads the document; nothing was timed for this figure. The eager
// installed speedup is NumPy ms over installed ms, which the document's table
// in "What the measurements show" gives the same at the large size. The line
// "what it is made of" is that table's last column. The groups follow the
// document's reading of the programs in "What the measurements show" and
// "What is left".
//
// Each TIMES row: NumPy ms, installed ms, lazy ms, lazy build, lazy compute,
// fused ms, fused build, plan, fused compute, vs lazy, vs NumPy, vs installed.
// "plain" is the lazy and fused run with nothing installed (large size only).
// Each GRAPHS row: deferred ops, graphs, largest graph, largest node elements,
// deferred fused, at once fused, expressions, tiled, kernel calls, fused
// nodes, given up, writes in place, writes copied, lazy MB written, fused MB.
//
// Attributes:
//   data-size     large or small (default large)
//   data-program  the program selected first (default cavity_flow)

import { h, seg, stat, fmtX } from "./util.js";

const TIMES = {
  "large":{"regression":[38.55,38.91,38.46,0.24,38.22,35,30.22,0.67,4.12,1.1,1.1,1.11],
  "attention":[28.65,28.56,28.64,0.44,28.2,28.88,1.05,0.13,27.71,0.99,0.99,0.99],
  "perceptron":[32.36,25.33,55.83,24.16,31.67,27.7,27.34,0.04,0.31,2.02,1.17,0.91],
  "char_rnn":[32.63,28.62,90.29,45.2,45.09,32.72,32.72,0,0,2.76,1,0.87],
  "particle_filter":[358.14,320.63,332.63,84.96,247.67,250.32,102.01,2.8,145.52,1.33,1.43,1.28],
  "veros_eos":[128.96,123.73,116.92,0.68,116.24,73.97,0.69,0.48,72.8,1.58,1.74,1.67],
  "veros_isoneutral":[218.8,219.14,228.37,8.54,219.83,189.67,20.66,0.88,168.13,1.2,1.15,1.16],
  "veros_tke":[99.66,97.04,100.92,23.11,77.81,87.93,28.47,0.52,58.94,1.15,1.13,1.1],
  "cavity_flow":[155.31,154.48,169.43,7.52,161.91,55.44,7.65,2.83,44.95,3.06,2.8,2.79],
  "hdiff":[69.41,66,69.07,5.8,63.27,29.43,5.8,0.04,23.59,2.35,2.36,2.24],
  "black_scholes":[87.29,86.84,88.47,1.88,86.58,85.49,1.88,0.04,83.57,1.03,1.02,1.02],
  "trimesh_mesh":[174.86,137.24,137.39,137.29,0.1,137.29,137.29,0,0,1,1.27,1],
  "sklearn_metrics":[978.27,272.56,278.66,277.32,1.35,277.37,277.37,0,0,1,3.53,0.98],
  "empyrical_stats":[1259.4,327.3,327.84,327.84,0,327.32,327.32,0,0,1,3.85,1],
  "skimage_pipeline":[1194.85,1113.44,1112.37,1112.37,0,1112.23,1112.23,0,0,1,1.07,1],
  "pandas_sales":[474.16,443.03,443.75,443.75,0,442.98,442.98,0,0,1,1.07,1],
  "astropy_timeseries":[1526.09,1295.04,1295.07,1293.2,1.87,1290.92,1289.81,0.08,1.04,1,1.18,1],
  "example":[45.01,8.46,8.82,4.22,4.6,8.84,7.45,0.01,1.37,1,5.09,0.96],
  "flame":[140.35,88.19,166.28,86.31,79.97,91.04,88.66,0.03,2.34,1.83,1.54,0.97],
  "azimint_naive":[171.71,154.52,168.94,6.62,162.32,155.33,155.33,0,0,1.09,1.11,0.99],
  "azimint_hist":[122.1,71.16,71.26,71.22,0.04,71.57,71.57,0,0,1,1.71,0.99],
  "mandelbrot1":[149.59,141.72,145.15,1.2,143.95,142.47,142.47,0,0,1.02,1.05,0.99],
  "mandelbrot2":[161.51,145.84,167.32,23.96,143.36,147.12,147.12,0,0,1.14,1.1,0.99],
  "make_decision":[143.15,85.28,1352.01,1039.2,312.81,116.8,116.8,0,0,11.58,1.23,0.73],
  "pandas_filter":[174.4,33.92,33.88,33.88,0,33.9,33.9,0,0,1,5.14,1],
  "equalize_hist":[141.24,126.55,126.75,126.67,0.08,126.65,126.65,0,0,1,1.12,1]},"plain":{"regression":[38.55,38.91,38.26,0.24,38.02,33.24,28.68,0.64,3.91,1.15,1.16,1.17],
  "attention":[28.65,28.56,28.67,0.45,28.23,29.35,1.06,0.11,28.18,0.98,0.98,0.97],
  "perceptron":[32.36,25.33,62.9,24.42,38.47,35.39,35.13,0.02,0.24,1.78,0.91,0.72],
  "char_rnn":[32.63,28.62,93.86,45.25,48.6,36.53,36.53,0,0,2.57,0.89,0.78],
  "particle_filter":[358.14,320.63,368.42,84.84,283.57,286.57,102.28,2.9,181.38,1.29,1.25,1.12],
  "veros_eos":[128.96,123.73,117.18,0.67,116.5,73.8,0.7,0.48,72.62,1.59,1.75,1.68],
  "veros_isoneutral":[218.8,219.14,227.57,8.61,218.95,190.29,21.98,0.87,167.45,1.2,1.15,1.15],
  "veros_tke":[99.66,97.04,102.94,23.07,79.87,90.03,28.39,0.52,61.12,1.14,1.11,1.08],
  "cavity_flow":[155.31,154.48,169.71,7.47,162.24,55.33,7.64,2.81,44.88,3.07,2.81,2.79],
  "hdiff":[69.41,66,72.57,5.8,66.77,29.47,5.82,0.03,23.62,2.46,2.36,2.24],
  "black_scholes":[87.29,86.84,88.3,1.88,86.41,85.42,1.88,0.04,83.49,1.03,1.02,1.02],
  "trimesh_mesh":[174.86,137.24,175.1,175,0.1,174.71,174.71,0,0,1,1,0.79],
  "sklearn_metrics":[978.27,272.56,1056.8,1053.75,3.05,1054.41,1054.41,0,0,1,0.93,0.26],
  "empyrical_stats":[1259.4,327.3,1289.58,1289.58,0,1289.75,1289.75,0,0,1,0.98,0.25],
  "skimage_pipeline":[1194.85,1113.44,1192.11,1192.11,0,1191.55,1191.55,0,0,1,1,0.93],
  "pandas_sales":[474.16,443.03,474.16,474.16,0,474.06,474.06,0,0,1,1,0.93],
  "astropy_timeseries":[1526.09,1295.04,1505.51,1503.18,2.33,1514.93,1513.86,0.08,1,0.99,1.01,0.85],
  "example":[45.01,8.46,45.3,24.52,20.77,45.36,43.81,0.01,1.54,1,0.99,0.19],
  "flame":[140.35,88.19,219.64,101.91,117.73,143.32,140.97,0.03,2.32,1.53,0.98,0.62],
  "azimint_naive":[171.71,154.52,186.01,6.72,179.29,173.02,173.02,0,0,1.08,0.99,0.89],
  "azimint_hist":[122.1,71.16,122.19,122.15,0.04,122.13,122.13,0,0,1,1,0.58],
  "mandelbrot1":[149.59,141.72,152.96,1.16,151.8,150.02,150.02,0,0,1.02,1,0.94],
  "mandelbrot2":[161.51,145.84,184.06,23.87,160.19,161.8,161.8,0,0,1.14,1,0.9],
  "make_decision":[143.15,85.28,1412.19,1036.43,375.75,176.99,176.99,0,0,7.98,0.81,0.48],
  "pandas_filter":[174.4,33.92,173.99,173.99,0,174.21,174.21,0,0,1,1,0.19],
  "equalize_hist":[141.24,126.55,141.61,141.52,0.09,141.4,141.4,0,0,1,1,0.89]},"small":{"regression":[0.47,0.36,1.17,0.54,0.64,0.48,0.48,0,0,2.47,0.99,0.77],
  "attention":[1.25,1.16,1.74,0.43,1.31,1.55,1.55,0,0,1.12,0.8,0.75],
  "perceptron":[4.3,3.43,7.18,2.84,4.34,3.72,3.45,0.02,0.24,1.93,1.16,0.92],
  "char_rnn":[6.58,5.88,18.19,10.66,7.54,7.24,7.24,0,0,2.51,0.91,0.81],
  "particle_filter":[4.26,3.73,6.25,2.43,3.82,4.39,4.39,0,0,1.42,0.97,0.85],
  "veros_eos":[6.56,6.58,7.12,0.3,6.82,5.8,0.35,0.49,4.95,1.23,1.13,1.14],
  "veros_isoneutral":[14.98,14.97,16.32,1.52,14.8,13.98,1.82,0.71,11.45,1.17,1.07,1.07],
  "veros_tke":[7.3,7.16,8.08,2.24,5.84,7.97,2.69,0.45,4.83,1.01,0.92,0.9],
  "cavity_flow":[5.53,5.14,13.61,6.15,7.46,5.48,5.48,0,0,2.48,1.01,0.94],
  "hdiff":[1.81,1.72,1.79,0.25,1.53,0.9,0.22,0.03,0.65,1.98,2.01,1.91],
  "black_scholes":[0.23,0.22,0.3,0.08,0.22,0.25,0.25,0,0,1.2,0.92,0.88],
  "trimesh_mesh":[11.36,9.49,9.57,9.53,0.04,9.52,9.52,0,0,1.01,1.19,1],
  "sklearn_metrics":[48.68,19.54,20.98,20.83,0.15,20.38,20.38,0,0,1.03,2.39,0.96],
  "empyrical_stats":[1248.41,356.94,356.95,356.95,0,357.11,357.11,0,0,1,3.5,1],
  "skimage_pipeline":[99.72,91.69,91.41,91.41,0,91.34,91.34,0,0,1,1.09,1],
  "pandas_sales":[30.25,28.74,28.66,28.66,0,28.75,28.75,0,0,1,1.05,1],
  "astropy_timeseries":[207.25,190.45,192.43,191.48,0.94,191.1,191.1,0,0,1.01,1.08,1],
  "example":[44.99,8.48,8.84,4.23,4.61,8.84,7.45,0.01,1.38,1,5.09,0.96],
  "flame":[16.94,9.73,21.71,12.02,9.69,10.46,9.99,0.02,0.45,2.07,1.62,0.93],
  "azimint_naive":[20.19,18.42,32.48,6.76,25.72,19.03,19.03,0,0,1.71,1.06,0.97],
  "azimint_hist":[12.32,7.16,7.29,7.26,0.03,7.24,7.24,0,0,1.01,1.7,0.99],
  "mandelbrot1":[15.71,14.81,17.41,1.1,16.32,14.98,14.98,0,0,1.16,1.05,0.99],
  "mandelbrot2":[16.31,14.53,21.95,5.44,16.51,14.85,14.85,0,0,1.48,1.1,0.98],
  "make_decision":[14.83,8.55,74.53,45.84,28.69,11.93,11.93,0,0,6.25,1.24,0.72],
  "pandas_filter":[17.55,3.57,3.56,3.56,0,3.57,3.57,0,0,1,4.92,1],
  "equalize_hist":[12.97,11.55,11.78,11.73,0.06,11.69,11.69,0,0,1.01,1.11,0.99]}};

const GRAPHS = {
  "large":{"regression":[384,1,384,3200000,83,281,41,41,41,41,0,0,0,179.2,59.2],
  "attention":[220,1,220,262144,184,33,8,8,16,16,0,0,0,174.7,135.4],
  "perceptron":[8854,565,33,128000,6,7908,2,2,4,2,0,0,0,163.7,2.4],
  "char_rnn":[32423,202,28606,10000,0,27923,0,0,0,0,0,0,0,165.8,0],
  "particle_filter":[2537,41,480,400000,2356,160,422,422,462,500,0,338,19,1166.5,640.9],
  "veros_eos":[366,1,366,1082016,363,3,37,37,140,285,0,0,0,3142.2,675.2],
  "veros_isoneutral":[718,1,690,4328064,442,27,121,121,226,266,9,31,0,3230.5,1074.4],
  "veros_tke":[382,6,281,3246048,260,28,38,38,77,115,11,28,3,1316.1,448.4],
  "cavity_flow":[6920,1,6900,160801,6043,47,260,260,560,2980,0,1196,0,4129.6,304.6],
  "hdiff":[40,1,40,10650240,29,0,4,4,4,15,0,1,0,1540.2,422.1],
  "black_scholes":[28,1,28,2000000,28,0,5,5,8,14,0,2,0,416,192],
  "trimesh_mesh":[2,1,2,245760,0,2,0,0,0,0,0,0,0,2.9,0],
  "sklearn_metrics":[7,4,2,1000000,0,7,0,0,0,0,0,0,0,11.4,0],
  "empyrical_stats":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "skimage_pipeline":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "pandas_sales":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "astropy_timeseries":[235,52,11,62113,20,130,0,0,0,0,0,0,0,11.1,9.9],
  "example":[13,2,9,2000000,3,9,0,0,0,0,0,1,0,68.1,18],
  "flame":[7418,416,53,589824,6,6855,1,1,1,0,0,1,0,299.3,8.6],
  "azimint_naive":[10001,1,10001,200000,0,9001,0,0,0,0,0,0,0,601.6,0],
  "azimint_hist":[1,1,1,1000,0,1,0,0,0,0,0,0,0,0,0],
  "mandelbrot1":[1205,1,1205,220900,0,904,0,0,0,0,0,0,0,1126.2,0],
  "mandelbrot2":[1132,239,19,902500,0,565,0,0,0,0,0,0,0,1278.1,0],
  "make_decision":[491520,5,100000,81920,0,409600,0,0,0,0,0,0,0,83.9,0],
  "pandas_filter":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "equalize_hist":[8,4,4,2890000,0,5,0,0,0,0,0,0,0,0,0]},"small":{"regression":[954,1,954,1600,0,904,0,0,0,0,0,0,0,0.4,0],
  "attention":[550,1,550,2048,0,520,0,0,0,0,0,0,0,3,0],
  "perceptron":[958,61,33,128000,6,852,2,2,4,2,0,0,0,23.6,2.4],
  "char_rnn":[5585,202,2497,10000,0,5135,0,0,0,0,0,0,0,17.9,0],
  "particle_filter":[1267,21,240,8000,0,940,0,0,0,0,0,0,0,11.7,0],
  "veros_eos":[366,1,366,72171,363,3,39,39,142,285,0,0,0,209.6,45],
  "veros_isoneutral":[718,1,690,288684,454,31,135,135,206,222,19,31,0,196.6,94.8],
  "veros_tke":[382,6,281,216513,232,50,37,37,53,73,17,28,2,82.5,44],
  "cavity_flow":[5900,1,5875,3721,0,3850,0,0,0,0,0,0,0,81.8,0],
  "hdiff":[40,1,40,261360,29,0,4,4,4,15,0,1,0,36.8,10.1],
  "black_scholes":[28,1,28,4096,0,28,0,0,0,0,0,0,0,0.9,0],
  "trimesh_mesh":[2,1,2,15360,0,2,0,0,0,0,0,0,0,0.2,0],
  "sklearn_metrics":[7,4,2,50000,0,7,0,0,0,0,0,0,0,0.6,0],
  "empyrical_stats":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "skimage_pipeline":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "pandas_sales":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "astropy_timeseries":[235,52,11,12451,0,150,0,0,0,0,0,0,0,2.2,0],
  "example":[13,2,9,2000000,3,9,0,0,0,0,0,1,0,68.1,18],
  "flame":[1870,108,53,147456,4,1726,1,1,1,0,0,0,0,31.3,2],
  "azimint_naive":[10001,1,10001,20000,0,9001,0,0,0,0,0,0,0,60.2,0],
  "azimint_hist":[1,1,1,1000,0,1,0,0,0,0,0,0,0,0,0],
  "mandelbrot1":[1205,1,1205,22500,0,904,0,0,0,0,0,0,0,114.4,0],
  "mandelbrot2":[1132,239,19,90000,0,565,0,0,0,0,0,0,0,126.9,0],
  "make_decision":[49152,1,49152,8192,0,40960,0,0,0,0,0,0,0,8.4,0],
  "pandas_filter":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  "equalize_hist":[8,4,4,262144,0,5,0,0,0,0,0,0,0,0,0]}};

const MADE = {
  "pandas_filter":"pandas, on plain arrays","example":"Single calls on two million elements","empyrical_stats":"empyrical, on plain arrays","sklearn_metrics":"scikit-learn, handed lazy arrays","cavity_flow":"Stencil expressions of 6 to 24 operations, each assigned into part of a grid","hdiff":"Four expressions of a stencil, each one kernel written by hand","veros_eos":"Polynomials, one of 153 operations","azimint_hist":"`np.histogram`","flame":"7,400 operations on arrays of 25,000 elements","particle_filter":"Random draws, `arctan2`, `exp`; expressions on columns of a table, and on its rows of two and four elements","trimesh_mesh":"trimesh, on plain arrays","make_decision":"490,000 operations on arrays of 32","astropy_timeseries":"astropy, on plain arrays","perceptron":"8,900 operations, most on arrays of 8,000 elements","veros_isoneutral":"Expressions on slices of a grid; 9 of 121 are given up for arrays that step","veros_tke":"A tridiagonal solve, masks, expressions on arrays that step","equalize_hist":"scikit-image's histogram equalisation","azimint_naive":"A thousand masks over 200,000 elements","regression":"A matrix product, a residual, a squared loss","mandelbrot2":"Complex arithmetic on arrays that shrink to nothing","skimage_pipeline":"scikit-image, on plain arrays","pandas_sales":"pandas, on plain arrays","mandelbrot1":"Complex arithmetic through a mask","black_scholes":"`erf`, `log`, `exp`; arithmetic between them","char_rnn":"32,000 operations on arrays of 100 to 10,000","attention":"float32 matrix products and `tanh`"};

// Column positions in a TIMES row and a GRAPHS row.
const T = { numpy: 0, installed: 1, lazy: 2, lazyBuild: 3, lazyCompute: 4, fused: 5, build: 6, plan: 7, compute: 8, vsLazy: 9, vsNumPy: 10, vsInst: 11 };
const G = { deferred: 0, graphs: 1, largest: 2, atOnce: 5, tiled: 7, kernelCalls: 8, fusedNodes: 9, givenUp: 10, inPlace: 11, lazyMB: 13, fusedMB: 14 };

// What each program is made of, in the document's terms.
const GROUPS = [
  { id: "expr", label: "Expressions on float64 arrays", note: "where the gain is fusion's, at the large size",
    programs: ["cavity_flow", "hdiff", "veros_eos", "particle_filter", "veros_isoneutral", "veros_tke", "regression"] },
  { id: "math", label: "Calls into a mathematics library", note: "erf, log, exp, tanh: no kernel calls a library",
    programs: ["black_scholes", "attention"] },
  { id: "lib", label: "A library's work", note: "the arrays are handed to pandas, scikit-learn, …",
    programs: ["pandas_filter", "empyrical_stats", "sklearn_metrics", "trimesh_mesh", "astropy_timeseries", "equalize_hist", "skimage_pipeline", "pandas_sales"] },
  { id: "single", label: "Single calls", note: "where the gain is the installed kernels'",
    programs: ["example", "azimint_hist"] },
  { id: "masks", label: "Masks and complex numbers", note: "made at once: no graph is written",
    programs: ["azimint_naive", "mandelbrot2", "mandelbrot1"] },
  { id: "small", label: "Small arrays at every size", note: "below 32,768 elements: no graph is written",
    programs: ["flame", "make_decision", "perceptron", "char_rnn"] },
];
const GROUP_OF = Object.fromEntries(GROUPS.flatMap((g) => g.programs.map((p) => [p, g])));

// The speedup axis, logarithmic, and its ticks.
const DOMAIN = [0.7, 6];
const TICKS = [0.8, 1, 1.5, 2, 3, 5];
// A fused run within this factor of the eager one is drawn as level: runs move by a few percent.
const LEVEL = 0.03;

// The parts a run's time is split into, and their colours.
const PARTS = {
  build: { label: "build: the program's Python, the nodes, calls made at once", color: "var(--k-input)" },
  plan: { label: "plan", color: "var(--k-reduction)" },
  compute: { label: "compute: NumPy's calls or the kernels'", color: "var(--k-view)" },
};

export const css = `
.fz-list { display: grid; grid-template-columns: minmax(6.5rem, 10rem) minmax(0, 1fr) 3.4rem; column-gap: .6rem; align-items: center; }
.fz-axis { display: contents; }
.fz-axis .fz-track { height: 1.3rem; background: none; }
.fz-tick { position: absolute; top: 0; transform: translateX(-50%); font-size: .68rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.fz-group { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: .15rem .6rem; align-items: baseline; margin: .7rem 0 .2rem; padding-top: .35rem; border-top: 1px solid var(--rule); }
.fz-group b { font-size: .74rem; font-weight: 650; letter-spacing: .05em; text-transform: uppercase; color: var(--ink-soft); }
.fz-group span { font-size: .74rem; color: var(--muted); }
.fz-row { display: contents; cursor: pointer; }
.fz-name { font-family: var(--mono); font-size: .74rem; text-align: left; background: none; border: 0; padding: .22rem 0; color: var(--ink); cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fz-name:hover, .fz-name.on { color: var(--hot); }
.fz-name.on { font-weight: 700; }
.fz-track { position: relative; height: 1.25rem; border-radius: 4px; }
.fz-track.on { background: var(--hot-soft); }
.fz-one { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--np); opacity: .7; }
.fz-grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--rule); }
.fz-seg { position: absolute; top: 50%; height: 3px; margin-top: -1.5px; border-radius: 2px; background: var(--rule-strong); }
.fz-seg.good { background: var(--good); }
.fz-seg.bad { background: var(--bad); }
.fz-dot, .fz-ring, .fz-dia { position: absolute; top: 50%; transform: translate(-50%, -50%); }
.fz-dot { width: .72rem; height: .72rem; border-radius: 50%; background: var(--hot); z-index: 2; }
.fz-ring { width: .72rem; height: .72rem; border-radius: 50%; border: 2px solid var(--cc); background: var(--bg-raised); box-sizing: border-box; z-index: 1; }
.fz-dia { width: .55rem; height: .55rem; background: var(--k-matmul); transform: translate(-50%, -50%) rotate(45deg); z-index: 3; }
.fz-val { font-size: .76rem; font-weight: 650; font-variant-numeric: tabular-nums; text-align: right; color: var(--hot); }
.fz-legend { display: flex; flex-wrap: wrap; gap: .35rem 1.1rem; font-size: .76rem; color: var(--muted); margin-top: .8rem; }
.fz-legend span { display: inline-flex; align-items: center; gap: .35rem; }
.fz-legend .fz-dot, .fz-legend .fz-ring, .fz-legend .fz-dia { position: static; transform: none; display: inline-block; }
.fz-legend .fz-dia { transform: rotate(45deg); }
.fz-legend i.line { display: inline-block; width: 1px; height: .9rem; background: var(--np); }
.fz-legend i.seg { display: inline-block; width: 1.2rem; height: 3px; border-radius: 2px; }
.fz-check { display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; color: var(--ink-soft); cursor: pointer; }
.fz-check.off { opacity: .45; cursor: default; }
.fz-detail { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid var(--rule); }
.fz-title { display: flex; flex-wrap: wrap; gap: .2rem .7rem; align-items: baseline; }
.fz-title code { font-size: .95rem; font-weight: 700; }
.fz-title span { font-size: .78rem; color: var(--muted); }
.fz-made { font-size: .86rem; margin: .3rem 0 .8rem; color: var(--ink-soft); }
.fz-bars { display: grid; grid-template-columns: minmax(5.5rem, 9rem) minmax(0, 1fr) 4.6rem; gap: .35rem .6rem; align-items: center; }
.fz-bars .lab { font-size: .78rem; color: var(--ink-soft); }
.fz-bars .bar { display: flex; height: 1rem; border-radius: 3px; overflow: hidden; min-width: 2px; }
.fz-bars .bar i { display: block; height: 100%; }
.fz-bars .ms { font-size: .78rem; font-variant-numeric: tabular-nums; text-align: right; }
@media (max-width: 40rem) {
  .fz-list { grid-template-columns: minmax(5.2rem, 7rem) minmax(0, 1fr) 2.9rem; column-gap: .4rem; }
  .fz-name { font-size: .66rem; }
  .fz-bars { grid-template-columns: 5rem minmax(0, 1fr) 4rem; }
}
`;

// Where a speedup sits on the axis, as a percentage of the track.
function pos(x) {
  const v = Math.min(Math.max(x, DOMAIN[0]), DOMAIN[1]);
  return ((Math.log(v) - Math.log(DOMAIN[0])) / (Math.log(DOMAIN[1]) - Math.log(DOMAIN[0]))) * 100;
}

// Text with `code` spans, as the document writes it, made into elements.
function withCode(text) {
  return text.split("`").map((part, i) => (i % 2 ? h("code", {}, part) : part));
}

// The speedups of one program at one size.
function speeds(size, name) {
  const row = TIMES[size][name];
  const plain = size === "large" ? TIMES.plain[name] : null;
  return {
    inst: Math.round((row[T.numpy] / row[T.installed]) * 100) / 100,
    fused: row[T.vsNumPy],
    vsInst: row[T.vsInst],
    alone: plain ? plain[T.vsNumPy] : null,
  };
}

// One run's time as a bar of its parts; parts: [[ms, colour, label]].
function bar(label, total, max, parts) {
  return [
    h("span", { class: "lab" }, label),
    h("div", { class: "bar", style: `width: ${(total / max) * 100}%`, title: parts.map(([ms, , l]) => `${l}: ${ms.toFixed(2)} ms`).join("\n") },
      ...parts.map(([ms, color]) => h("i", { style: `width: ${total ? (ms / total) * 100 : 0}%; background: ${color}` }))),
    h("span", { class: "ms" }, `${total.toFixed(total >= 100 ? 1 : 2)} ms`),
  ];
}

export default async function mount(el) {
  let size = el.dataset.size === "small" ? "small" : "large";
  let selected = TIMES.large[el.dataset.program] ? el.dataset.program : "cavity_flow";
  let order = "grouped";
  let showAlone = false;

  const list = h("div", { class: "fz-list", role: "list" });
  const detail = h("div", { class: "fz-detail", "aria-live": "polite" });
  const aloneBox = h("input", { type: "checkbox", onchange: () => { showAlone = aloneBox.checked; render(); } });
  const aloneLabel = h("label", { class: "fz-check" }, aloneBox, "Also the fused run with nothing installed");

  function row(name) {
    const sp = speeds(size, name);
    const on = name === selected;
    const lo = Math.min(sp.inst, sp.fused), hi = Math.max(sp.inst, sp.fused);
    const tone = sp.vsInst >= 1 + LEVEL ? "good" : sp.vsInst <= 1 - LEVEL ? "bad" : "";
    const pick = () => { selected = name; render(); };
    const title = `${name}: fused and installed ${fmtX(sp.fused)} against NumPy; installed eagerly ${fmtX(sp.inst)}; fused against installed ${fmtX(sp.vsInst)}`
      + (sp.alone !== null ? `; fused alone ${fmtX(sp.alone)}` : "");
    return h("div", { class: "fz-row", role: "listitem", onclick: pick },
      h("button", { type: "button", class: `fz-name${on ? " on" : ""}`, title, "aria-pressed": on ? "true" : "false" }, name),
      h("div", { class: `fz-track${on ? " on" : ""}`, title },
        ...TICKS.filter((t) => t !== 1).map((t) => h("i", { class: "fz-grid", style: `left: ${pos(t)}%` })),
        h("i", { class: "fz-one", style: `left: ${pos(1)}%` }),
        h("i", { class: `fz-seg ${tone}`, style: `left: ${pos(lo)}%; width: ${pos(hi) - pos(lo)}%` }),
        h("i", { class: "fz-ring", style: `left: ${pos(sp.inst)}%` }),
        h("i", { class: "fz-dot", style: `left: ${pos(sp.fused)}%` }),
        showAlone && sp.alone !== null ? h("i", { class: "fz-dia", style: `left: ${pos(sp.alone)}%` }) : null),
      h("span", { class: "fz-val" }, fmtX(sp.fused)));
  }

  function renderList() {
    const axis = h("div", { class: "fz-axis", "aria-hidden": "true" },
      h("span", { class: "w-label" }, "program"),
      h("div", { class: "fz-track" }, ...TICKS.map((t) => h("span", { class: "fz-tick", style: `left: ${pos(t)}%` }, t === 1 ? "1x" : `${t}x`))),
      h("span", { class: "w-label", style: "text-align: right" }, "fused"));
    const children = [axis];
    if (order === "grouped") {
      for (const group of GROUPS) {
        children.push(h("div", { class: "fz-group" }, h("b", {}, group.label), h("span", {}, group.note)));
        const names = [...group.programs].sort((a, b) => speeds(size, b).fused - speeds(size, a).fused);
        children.push(...names.map(row));
      }
    } else {
      const key = order === "fused" ? "fused" : "vsInst";
      const names = Object.keys(TIMES[size]).sort((a, b) => speeds(size, b)[key] - speeds(size, a)[key]);
      children.push(...names.map(row));
    }
    list.replaceChildren(...children);
  }

  function renderDetail() {
    const t = TIMES[size][selected];
    const g = GRAPHS[size][selected];
    const plain = size === "large" ? TIMES.plain[selected] : null;
    const group = GROUP_OF[selected];
    const runs = [
      ["NumPy", t[T.numpy], [[t[T.numpy], "var(--np)", "NumPy"]]],
      ["Installed, eager", t[T.installed], [[t[T.installed], "var(--cc)", "installed"]]],
      ["Lazy, node by node", t[T.lazy], [[t[T.lazyBuild], PARTS.build.color, "build"], [t[T.lazyCompute], PARTS.compute.color, "compute"]]],
      ["Fused, installed", t[T.fused], [[t[T.build], PARTS.build.color, "build"], [t[T.plan], PARTS.plan.color, "plan"], [t[T.compute], PARTS.compute.color, "compute"]]],
    ];
    if (plain) runs.push(["Fused, nothing installed", plain[T.fused], [[plain[T.build], PARTS.build.color, "build"], [plain[T.plan], PARTS.plan.color, "plan"], [plain[T.compute], PARTS.compute.color, "compute"]]]);
    const max = Math.max(...runs.map((r) => r[1]));

    const vsN = stat("fused and installed, against NumPy", "hot");
    const vsI = stat("against the eager program with the kernels", t[T.vsInst] >= 1 + LEVEL ? "good" : t[T.vsInst] <= 1 - LEVEL ? "bad" : "");
    const calls = stat(g[G.tiled] ? `kernel calls, in ${g[G.tiled]} programs of kernels` : "kernel calls: nothing was fused");
    const mb = stat("MB of results written, node by node → fused");
    vsN.set(fmtX(t[T.vsNumPy]));
    vsI.set(fmtX(t[T.vsInst]));
    calls.set(String(g[G.kernelCalls]));
    const mbText = (v) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : String(v));
    mb.set(`${mbText(g[G.lazyMB])} → ${mbText(g[G.fusedMB])}`);

    let note;
    if (g[G.kernelCalls]) {
      const n = (v) => v.toLocaleString("en-US");
      const plural = (v, word) => `${n(v)} ${word}${v === 1 ? "" : "s"}`;
      note = `The fused run made ${plural(g[G.kernelCalls], "kernel call")} in ${plural(g[G.tiled], "program")} of kernels`
        + (g[G.fusedNodes] ? `, inside which ${plural(g[G.fusedNodes], "node")} had no array of their own. ` : ". ")
        + (g[G.inPlace] ? `It made ${plural(g[G.inPlace], "deferred write")} in the array's own memory. ` : "")
        + `It made ${n(g[G.atOnce])} ${g[G.atOnce] === 1 ? "call or write" : "calls and writes"} at once, as the program wrote ${g[G.atOnce] === 1 ? "it" : "them"}. `
        + (g[G.givenUp] ? `${g[G.givenUp]} of the programs were given up and left to NumPy. ` : "")
        + `Planning took ${t[T.plan].toFixed(2)} ms.`;
    } else if (g[G.deferred] === 0) {
      note = "Nothing of this program was deferred: its work is a library's, and the library is handed plain arrays. The lazy run is the eager one with the stand-in around it.";
    } else {
      note = `No kernel computed anything here. Of ${g[G.deferred].toLocaleString("en-US")} operations the lazy run deferred, the fused run made ${g[G.atOnce].toLocaleString("en-US")} at once, as the program wrote them, `
        + "because they were small or no kernel could make them. Its run is the eager program's with a lazy array around each value.";
    }

    detail.replaceChildren(
      h("div", { class: "fz-title" }, h("code", {}, selected), h("span", {}, group.label)),
      h("p", { class: "fz-made" }, ...withCode(MADE[selected])),
      h("div", { class: "fz-bars" }, ...runs.flatMap(([label, total, parts]) => bar(label, total, max, parts))),
      h("div", { class: "fz-legend" },
        h("span", {}, h("i", { class: "seg", style: "background: var(--np)" }), "NumPy"),
        h("span", {}, h("i", { class: "seg", style: "background: var(--cc)" }), "eager, kernels installed"),
        ...Object.values(PARTS).map((p) => h("span", {}, h("i", { class: "seg", style: `background: ${p.color}` }), p.label))),
      h("p", { class: "w-note" }, note),
      h("div", { class: "w-readout" }, vsN, vsI, calls, mb));
  }

  function render() {
    aloneBox.disabled = size !== "large";
    aloneLabel.classList.toggle("off", size !== "large");
    renderList();
    renderDetail();
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Arrays"),
      seg([{ value: "large", label: "large" }, { value: "small", label: "small" }], size, (v) => { size = v; render(); }),
      h("span", { class: "w-label" }, "Order"),
      seg([{ value: "grouped", label: "by what it is made of" }, { value: "fused", label: "by speedup" }, { value: "inst", label: "by gain over installed" }], order, (v) => { order = v; render(); }),
      aloneLabel),
    list,
    h("div", { class: "fz-legend" },
      h("span", {}, h("i", { class: "fz-dot" }), "fused, kernels installed, against NumPy"),
      h("span", {}, h("i", { class: "fz-ring" }), "eager, kernels installed, against NumPy"),
      h("span", {}, h("i", { class: "seg", style: "background: var(--good)" }), h("i", { class: "seg", style: "background: var(--bad)" }), "what the graph adds, ahead or behind by 3% or more"),
      h("span", {}, h("i", { class: "fz-dia" }), "fused, nothing installed (large only)"),
      h("span", {}, h("i", { class: "line" }), "NumPy, 1x")),
    detail);
  render();
}
