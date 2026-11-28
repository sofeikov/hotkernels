// How far any fusion could go, program by program, and what is left in the fused run.
//
// Two views, switchable. "The floor" draws each program's times as shares of
// its time on plain NumPy: the bar is the eager program with the kernels
// installed, the black tick is the floor (that time less everything a fusion
// of all its element-by-element operations and reductions, at no cost of its
// own, could remove), and the dot is the fused run with the kernels installed.
// A dot left of the tick is a run that beat the floor: a kernel makes several
// operations for each load and store, which NumPy's loops cannot. "The fused
// run in three parts" splits each fused run into the time inside its kernels'
// programs, the calls NumPy still makes that a kernel could make (with the
// part that would go if each cost what its loop costs in the cache), and the
// rest of the run.
//
// Data: docs/lazy-fusion-design.md, "How far any fusion could go", the first
// and last tables, from
//   uv run --group apps --group workloads python -m prototypes.lazy.programs.limit --size large
// on an Apple M1 Pro, NumPy 2.5.5, the quickest of 5 runs. This run is not the
// run of the four-ways table, so its times differ from that table's by a few
// percent. make_decision has no floor that means anything (its arrays are 32
// elements) and is drawn without one. The numbers are copied by a script that
// reads the document; nothing was timed for this figure.
//
// Each FLOOR row: NumPy, installed, floor, fused (ms), at best, now, left (ms), left of fused run (%).
// Each PARTS row: kernels ms; arithmetic, masks, library, other types (ms and removable ms each);
// the rest of the run ms; could still go ms.
//
// Attributes:
//   data-view     floor or parts (default floor)
//   data-program  the program picked first (default cavity_flow)

import { h, seg, stat, fmtX } from "./util.js";

const FLOOR = {
  "regression":[42.69,41.62,32.58,36.02,1.31,1.19,3.43,9.5],
  "attention":[28.8,28.89,26.73,29.02,1.08,0.99,2.3,7.9],
  "perceptron":[32.07,25.18,20.71,27.75,1.55,1.16,7.04,25.4],
  "char_rnn":[32.37,28.47,21.89,32.56,1.48,0.99,10.67,32.8],
  "particle_filter":[364.18,325.48,254.44,253.97,1.43,1.43,-0.47,-0.2],
  "veros_eos":[129.95,127.41,111.25,74.43,1.17,1.75,-36.82,-49.5],
  "veros_isoneutral":[221.07,220.56,171.17,190.04,1.29,1.16,18.87,9.9],
  "veros_tke":[101.65,97.92,73.28,89.07,1.39,1.14,15.79,17.7],
  "cavity_flow":[158.22,157.69,116.08,56.2,1.36,2.82,-59.88,-106.5],
  "hdiff":[69.95,66.4,62.22,29.75,1.12,2.35,-32.47,-109.2],
  "black_scholes":[88.27,87.42,84.64,86.03,1.04,1.03,1.39,1.6],
  "trimesh_mesh":[175.5,137.26,137.24,137.08,1.28,1.28,-0.16,-0.1],
  "sklearn_metrics":[980.95,274.46,274.41,279.19,3.57,3.51,4.78,1.7],
  "empyrical_stats":[1289,328.06,328.06,328.08,3.93,3.93,0.02,0],
  "skimage_pipeline":[1205.52,1119.4,1119.4,1117.58,1.08,1.08,-1.82,-0.2],
  "pandas_sales":[476.44,444.19,444.19,443.74,1.07,1.07,-0.45,-0.1],
  "astropy_timeseries":[1511.45,1282.49,1282.3,1282.64,1.18,1.18,0.33,0],
  "example":[45.01,8.51,8.21,8.86,5.48,5.08,0.65,7.4],
  "flame":[140.3,87.98,79.19,90.55,1.77,1.55,11.36,12.5],
  "azimint_naive":[171.77,154.57,149.22,156.43,1.15,1.1,7.21,4.6],
  "azimint_hist":[122.12,71.24,71.23,71.3,1.71,1.71,0.07,0.1],
  "mandelbrot1":[148.62,141.19,129.83,141.84,1.14,1.05,12.01,8.5],
  "mandelbrot2":[161.05,145.74,145.34,146.77,1.11,1.1,1.43,1],
  "make_decision":[143.5,85.55,null,117.95,null,1.22,null,null],
  "pandas_filter":[174.44,34.03,34.03,33.91,5.13,5.14,-0.12,-0.4],
  "equalize_hist":[141.27,126.43,126.42,126.51,1.12,1.12,0.09,0.1]};

const PARTS = {
  "regression":[3.33,0,0,0,0,0,0,0,0,32.69,0],
  "attention":[0.84,1.3,0.52,0,0,17.41,0.58,1.83,1.07,6.56,2.57],
  "perceptron":[0.1,0,0,0,0,0,0,0,0,27.64,0],
  "char_rnn":[0,0,0,0,0,0,0,0,0,32.56,0],
  "particle_filter":[44.75,0.57,0.26,1.45,0.07,58.84,0.28,0,0,146.81,0.87],
  "veros_eos":[55.61,11.59,2.42,0,0,6.18,0.02,0,0,1.06,2.45],
  "veros_isoneutral":[101.57,12.81,7.14,0,0,56.08,3.86,0,0,19.58,11],
  "veros_tke":[36.39,9.42,4.88,6.57,0.78,0,0,0,0,36.68,5.67],
  "cavity_flow":[37.15,0,0,0,0,4.67,0.24,0,0,14.38,0.24],
  "hdiff":[18.66,5.38,0,0,0,0,0,0,0,5.71,0],
  "black_scholes":[7.63,2.69,0.44,0,0,74.75,1.38,0,0,0.96,1.82],
  "trimesh_mesh":[0,0,0,0,0,0,0,0,0,137.08,0],
  "sklearn_metrics":[0,0,0,0,0,0,0,0,0,279.19,0],
  "empyrical_stats":[0,0,0,0,0,0,0,0,0,328.08,0],
  "skimage_pipeline":[0,0,0,0,0,0,0,0,0,1117.58,0],
  "pandas_sales":[0,0,0,0,0,0,0,0,0,443.74,0],
  "astropy_timeseries":[0,0.21,0.01,0,0,0,0,0.67,0.08,1281.76,0.09],
  "example":[0,0.47,0.09,0.68,0,0,0,0,0,7.71,0.1],
  "flame":[0.45,0.08,0.03,0,0,1.22,0.02,0,0,88.8,0.05],
  "azimint_naive":[0,0,0,0,0,0,0,0,0,156.43,0],
  "azimint_hist":[0,0,0,0,0,0,0,0,0,71.3,0],
  "mandelbrot1":[0,0,0,0,0,0,0,0,0,141.84,0],
  "mandelbrot2":[0,0,0,0,0,0,0,0,0,146.77,0],
  "make_decision":[0,0,0,0,0,0,0,0,0,117.95,0],
  "pandas_filter":[0,0,0,0,0,0,0,0,0,33.91,0],
  "equalize_hist":[0,0,0,0,0,0,0,0,0,126.51,0]};

// Column positions.
const F = { numpy: 0, installed: 1, floor: 2, fused: 3, atBest: 4, now: 5, left: 6, leftPct: 7 };
// The track of the floor view runs to this share of NumPy's time.
const FLOOR_MAX = 1.05;

export const css = `
.ff-rows { display: grid; grid-template-columns: minmax(6.5rem, 10rem) minmax(0, 1fr) 3.6rem; gap: .26rem .6rem; align-items: center; }
.ff-head { font-size: .68rem; color: var(--muted); position: relative; height: 1.1rem; }
.ff-head span { position: absolute; transform: translateX(-50%); font-variant-numeric: tabular-nums; }
.ff-name { font-family: var(--mono); font-size: .74rem; background: none; border: 0; padding: 0; text-align: left; color: var(--ink); cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ff-name.on, .ff-name:hover { color: var(--hot); }
.ff-name.on { font-weight: 700; }
.ff-track { position: relative; height: 1.15rem; cursor: pointer; border-radius: 3px; }
.ff-track.on { box-shadow: 0 0 0 2px var(--ink); }
.ff-inst { position: absolute; top: 0; bottom: 0; left: 0; background: var(--cc-soft); border-right: 2px solid var(--cc); border-radius: 3px 0 0 3px; }
.ff-gain { position: absolute; top: 0; bottom: 0; background: repeating-linear-gradient(135deg, transparent 0 3px, var(--good) 3px 4.5px); opacity: .55; }
.ff-floor { position: absolute; top: -3px; bottom: -3px; width: 2px; margin-left: -1px; background: var(--ink); }
.ff-dot { position: absolute; top: 50%; width: .7rem; height: .7rem; border-radius: 50%; background: var(--hot); transform: translate(-50%, -50%); z-index: 2; }
.ff-stack { display: flex; height: 1.15rem; border-radius: 3px; overflow: hidden; cursor: pointer; }
.ff-stack.on { outline: 2px solid var(--ink); outline-offset: 1px; }
.ff-stack i { display: block; height: 100%; }
.ff-val { font-size: .74rem; font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; }
.ff-legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .76rem; color: var(--muted); margin-top: .8rem; }
.ff-legend span { display: inline-flex; align-items: center; gap: .35rem; }
.ff-legend i { width: .85rem; height: .85rem; border-radius: 3px; display: inline-block; }
@media (max-width: 40rem) {
  .ff-rows { grid-template-columns: minmax(5rem, 7rem) minmax(0, 1fr) 3rem; }
  .ff-name { font-size: .66rem; }
}
`;

// The colours of the parts of a fused run.
const SLICES = {
  kernels: { label: "inside the kernels' programs", color: "var(--hot)" },
  calls: { label: "NumPy's calls a kernel could make", color: "var(--np)" },
  removable: { label: "of those, gone if each cost its loop in the cache", color: "var(--good)" },
  rest: { label: "the rest: the program's Python, libraries, matrix products, moving elements, the layer", color: "var(--rule-strong)" },
};

// A fused run's parts in ms: kernels, NumPy's calls less what could go, what could go, the rest.
function parts(name) {
  const r = PARTS[name];
  const calls = r[1] + r[3] + r[5] + r[7];
  const removable = r[2] + r[4] + r[6] + r[8];
  return { kernels: r[0], calls: calls - removable, removable, rest: r[9], could: r[10], total: r[0] + calls + r[9] };
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;

export default async function mount(el) {
  let view = el.dataset.view === "parts" ? "parts" : "floor";
  let picked = FLOOR[el.dataset.program] ? el.dataset.program : "cavity_flow";
  let order = "left";

  const rows = h("div", { class: "ff-rows" });
  const legend = h("div", { class: "ff-legend" });
  const note = h("p", { class: "w-note", "aria-live": "polite" });
  const readout = h("div", { class: "w-readout" });

  function names() {
    const all = Object.keys(FLOOR);
    if (order === "table") return all;
    if (view === "floor") {
      // Most left to gain first; a program with no floor last.
      return all.sort((a, b) => (FLOOR[b][F.leftPct] ?? -1e9) - (FLOOR[a][F.leftPct] ?? -1e9));
    }
    return all.sort((a, b) => parts(b).could / parts(b).total - parts(a).could / parts(a).total);
  }

  function floorRow(name) {
    const r = FLOOR[name];
    const on = name === picked;
    const at = (ms) => `${(Math.min(ms / r[F.numpy], FLOOR_MAX) / FLOOR_MAX) * 100}%`;
    const pick = () => { picked = name; render(); };
    const title = `${name}: NumPy ${r[F.numpy]} ms, installed ${r[F.installed]} ms, floor ${r[F.floor] ?? "none"}, fused ${r[F.fused]} ms`;
    return [
      h("button", { type: "button", class: `ff-name${on ? " on" : ""}`, onclick: pick, title }, name),
      h("div", { class: `ff-track${on ? " on" : ""}`, onclick: pick, title, role: "img", "aria-label": title },
        h("i", { class: "ff-inst", style: `width: ${at(r[F.installed])}` }),
        r[F.floor] !== null ? h("i", { class: "ff-gain", style: `left: ${at(r[F.floor])}; width: calc(${at(r[F.installed])} - ${at(r[F.floor])})` }) : null,
        r[F.floor] !== null ? h("i", { class: "ff-floor", style: `left: ${at(r[F.floor])}` }) : null,
        h("i", { class: "ff-dot", style: `left: ${at(r[F.fused])}` })),
      h("span", { class: "ff-val", style: "color: var(--hot)" }, fmtX(r[F.now])),
    ];
  }

  function partsRow(name) {
    const p = parts(name);
    const on = name === picked;
    const pick = () => { picked = name; render(); };
    const title = `${name}: kernels ${p.kernels.toFixed(2)} ms, NumPy's calls a kernel could make ${(p.calls + p.removable).toFixed(2)} ms of which ${p.removable.toFixed(2)} could go, the rest ${p.rest.toFixed(2)} ms`;
    return [
      h("button", { type: "button", class: `ff-name${on ? " on" : ""}`, onclick: pick, title }, name),
      h("div", { class: `ff-stack${on ? " on" : ""}`, onclick: pick, title, role: "img", "aria-label": title },
        ...["kernels", "calls", "removable", "rest"].map((k) => h("i", { style: `width: ${(p[k] / p.total) * 100}%; background: ${SLICES[k].color}` }))),
      h("span", { class: "ff-val", style: "color: var(--good)" }, pct(p.could / p.total)),
    ];
  }

  function render() {
    const head = view === "floor"
      ? h("div", { class: "ff-head" }, ...[0, 0.5, 1].map((v) => h("span", { style: `left: ${(v / FLOOR_MAX) * 100}%` }, v === 1 ? "NumPy" : `${v * 100}%`)))
      : h("div", { class: "ff-head" }, ...[0, 0.5, 1].map((v) => h("span", { style: `left: ${v * 100}%; ${v === 1 ? "transform: translateX(-100%)" : v === 0 ? "transform: none" : ""}` }, `${v * 100}%`)));
    rows.replaceChildren(
      h("span", { class: "w-label" }, "program"), head,
      h("span", { class: "w-label", style: "text-align: right" }, view === "floor" ? "now" : "could go"),
      ...names().flatMap(view === "floor" ? floorRow : partsRow));

    if (view === "floor") {
      legend.replaceChildren(
        h("span", {}, h("i", { style: "background: var(--cc-soft); border-right: 2px solid var(--cc)" }), "eager, kernels installed"),
        h("span", {}, h("i", { style: "background: repeating-linear-gradient(135deg, transparent 0 3px, var(--good) 3px 4.5px); opacity: .7" }), "what a fusion of everything, at no cost, could remove"),
        h("span", {}, h("i", { style: "background: var(--ink); width: 2px" }), "the floor"),
        h("span", {}, h("i", { style: "background: var(--hot); border-radius: 50%" }), "fused, kernels installed"),
        h("span", {}, "times as shares of the program's time on plain NumPy"));
      const r = FLOOR[picked];
      const items = [
        ["NumPy", `${r[F.numpy]} ms`, "np"], ["installed, eager", `${r[F.installed]} ms`, ""],
        ["the floor", r[F.floor] === null ? "none" : `${r[F.floor]} ms`, ""], ["fused, installed", `${r[F.fused]} ms`, "hot"],
        ["at best: NumPy over the floor", r[F.atBest] === null ? "–" : fmtX(r[F.atBest]), ""], ["now: NumPy over the fused run", fmtX(r[F.now]), "hot"],
      ];
      readout.replaceChildren(...items.map(([label, value, kind]) => { const st = stat(label, kind); st.set(value); return st; }));
      if (r[F.floor] === null) note.textContent = `${picked} has no floor that means anything: its arrays are 32 elements, and what could be removed from an operation that small, the call, is all the operation costs.`;
      else if (r[F.left] < 0) note.textContent = `${picked}'s fused run is ${Math.abs(r[F.left]).toFixed(2)} ms below the floor. The floor is what NumPy's own loops would take with no pass over memory and no call; a kernel makes several operations for each load and store, which NumPy's loops cannot, so it can run below it.`;
      else note.textContent = `${picked}'s fused run is ${r[F.left].toFixed(2)} ms above the floor, ${r[F.leftPct]}% of the run. That is the most any fusion of its element-by-element operations and reductions could still take, if it cost nothing; the second view says where in the run it is.`;
    } else {
      legend.replaceChildren(...Object.values(SLICES).map((sl) => h("span", {}, h("i", { style: `background: ${sl.color}` }), sl.label)), h("span", {}, "each bar is one fused run, whole"));
      const p = parts(picked);
      const items = [
        ["fused run", `${FLOOR[picked][F.fused]} ms`, "hot"],
        ["inside the kernels' programs", `${p.kernels.toFixed(2)} ms`, ""],
        ["NumPy's calls a kernel could make", `${(p.calls + p.removable).toFixed(2)} ms`, "np"],
        ["the rest of the run", `${p.rest.toFixed(2)} ms`, ""],
        ["could still go", `${p.could.toFixed(2)} ms`, "good"],
      ];
      readout.replaceChildren(...items.map(([label, value, kind]) => { const st = stat(label, kind); st.set(value); return st; }));
      note.textContent = p.could >= 1
        ? `What a fusion of everything not yet fused could take off ${picked}'s fused run, at no cost of its own, is ${p.could.toFixed(2)} ms, ${pct(p.could / p.total)}. It is in the calls NumPy still makes for operations a kernel could make.`
        : `For ${picked} what is not yet fused would save ${p.could.toFixed(2)} ms at most: ${pct(p.could / p.total)} of the run. Its time is in its kernels or in the rest of the run, which is no tree of element-by-element operations on float64.`;
    }
  }

  el.append(
    h("div", { class: "w-controls" },
      seg([{ value: "floor", label: "The floor" }, { value: "parts", label: "The fused run in three parts" }], view, (v) => { view = v; render(); }),
      h("span", { class: "w-label" }, "Order"),
      seg([{ value: "left", label: "most left first" }, { value: "table", label: "as the table" }], order, (v) => { order = v; render(); })),
    rows, legend, note, readout);
  render();
}
