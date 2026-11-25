// What writing one node of the graph costs, beside what NumPy takes to compute it.
//
// Writing a * b down is not free: the layer has to know the result's shape
// and element type, the line of the user's code, and the error settings in
// effect, and make the node. The figure puts three ways of doing that beside
// the time NumPy takes to multiply two float64 arrays, for sizes from 16 to
// 8,192 elements, and says at what size the two cost the same.
//
// Every number is a measurement from docs/lazy-design.md ("What writing a
// node costs"), on an Apple M1 Pro: the quickest of several runs, for a node
// written and dropped. NumPy's times are for np.multiply on two float64
// arrays. Between the sizes measured the curve is a straight line, since
// NumPy's time grows by a fixed cost per element.
//
// Attributes:
//   data-road  the road shown first: ask, remember or core (default remember)

import { h, s, seg, stat, scale, fmtN } from "./util.js";

// NumPy multiplying two float64 arrays: [elements, microseconds].
const NUMPY = [[16, 0.23], [1024, 0.45], [4096, 0.8], [8192, 1.9]];

// The three ways of writing a node, with the measured range in microseconds,
// and what the time is spent on, in microseconds, where it was measured.
const ROADS = {
  ask: {
    label: "Asking every time", lo: 14, hi: 19,
    what: "Every node asks NumPy for its shape and type, walks the stack for the user's line, and reads the error settings. The parts were timed one by one and do not add up to the whole: the line costs as much again for each frame walked.",
    parts: [["NumPy's answer for shape and type, on stand-ins", 4, 6.5], ["the user's line: each frame of the layer walked", 1.3, 1.3], ["np.geterr()", 0.4, 0.4], ["the rest of the node, as when remembering", 1.4, 1.6]],
  },
  remember: {
    label: "Remembering, in Python", lo: 1.4, hi: 1.6,
    what: "A loop asks the same questions every turn, so the answers are kept. What is left is the interpreter's own work.",
    parts: [["the operator's method and its arguments", 0.35, 0.35], ["the key, and the answer looked up", 0.25, 0.25], ["the node", 0.25, 0.25], ["the pending set, the count, the error settings", 0.25, 0.25], ["finding the line again", 0.2, 0.2], ["the lazy array and its cell", 0.17, 0.17]],
  },
  core: {
    label: "The C core", lo: 0.41, hi: 0.43,
    what: "The same work and the same memories, done in C: a * b goes from Python's operator straight into the core, with no Python frame between.",
    parts: [["the whole node, in C", 0.41, 0.43]],
  },
};

export const css = `
.nc-parts { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 1rem 1.4rem; align-items: start; }
@media (max-width: 680px) { .nc-parts { grid-template-columns: 1fr; } }
.nc-title { font-size: .72rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .4rem; }
.nc-grid { stroke: var(--rule); stroke-width: 1; }
.nc-np { fill: none; stroke: var(--np); stroke-width: 2.2; }
.nc-dot { fill: var(--bg-raised); stroke: var(--np); stroke-width: 1.8; }
.nc-band { fill: var(--hot-soft); stroke: var(--hot); stroke-width: 1.2; opacity: .45; }
.nc-band.on { opacity: 1; }
.nc-cross { stroke: var(--hot); stroke-width: 1.4; stroke-dasharray: 4 3; }
.nc-row { margin-bottom: .45rem; font-size: .78rem; }
.nc-row .lbl { display: flex; justify-content: space-between; gap: .6rem; }
.nc-row .lbl span:last-child { font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }
.nc-row .bar { height: .5rem; border-radius: 3px; background: var(--hot); margin-top: .15rem; }
`;

// NumPy's time at n elements, a straight line between the measured sizes.
function numpyAt(n) {
  for (let i = 1; i < NUMPY.length; i++) {
    const [n0, t0] = NUMPY[i - 1], [n1, t1] = NUMPY[i];
    if (n <= n1) return t0 + ((n - n0) / (n1 - n0)) * (t1 - t0);
  }
  return null;
}

// The size at which NumPy takes t microseconds, or null beyond what was measured.
function sizeFor(t) {
  for (let i = 1; i < NUMPY.length; i++) {
    const [n0, t0] = NUMPY[i - 1], [n1, t1] = NUMPY[i];
    if (t >= t0 && t <= t1) return n0 + ((t - t0) / (t1 - t0)) * (n1 - n0);
  }
  return null;
}

// "about 900": a size rounded to one significant figure.
function about(n) {
  const p = 10 ** Math.floor(Math.log10(n));
  return fmtN(Math.round(n / p) * p);
}

function chart(on) {
  const W = 420, H = 280, L = 40, R = 20, T = 10, B = 36;
  const x = scale([16, 8192], [L, W - R], true);
  const y = scale([0.1, 30], [H - B, T], true);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Microseconds against elements, both logarithmic: NumPy multiplying two arrays, and the cost of writing one node by each road" });
  for (const v of [0.1, 0.3, 1, 3, 10, 30]) {
    svg.append(s("line", { class: "nc-grid", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    svg.append(s("text", { class: "muted", x: L - 6, y: y(v) + 4, "text-anchor": "end", "font-size": 11.5 }, v));
  }
  for (const v of [16, 64, 256, 1024, 4096, 8192]) {
    svg.append(s("text", { class: "muted", x: x(v), y: H - B + 15, "text-anchor": "middle", "font-size": 11.5 }, fmtN(v)));
  }
  svg.append(s("text", { class: "muted", x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", "font-size": 11.5 }, "elements multiplied"));
  svg.append(s("text", { class: "muted", x: 11, y: (T + H - B) / 2, "text-anchor": "middle", "font-size": 11.5,
    transform: `rotate(-90 11 ${(T + H - B) / 2})` }, "microseconds"));

  for (const [key, road] of Object.entries(ROADS)) {
    svg.append(s("rect", { class: `nc-band${key === on ? " on" : ""}`, x: L, width: W - R - L, y: y(road.hi) - 1.5, height: Math.max(3, y(road.lo) - y(road.hi) + 3) }));
    svg.append(s("text", { x: W - R - 4, y: y(road.hi) - 5, "text-anchor": "end", "font-size": 11.5,
      style: `fill: var(--hot); font-weight: ${key === on ? 700 : 400}` }, road.label));
  }
  const points = [];
  for (let i = 0; i <= 80; i++) {
    const n = 16 * 512 ** (i / 80);
    points.push(`${x(n).toFixed(1)},${y(numpyAt(n)).toFixed(1)}`);
  }
  svg.append(s("polyline", { class: "nc-np", points: points.join(" ") }));
  for (const [n, t] of NUMPY) svg.append(s("circle", { class: "nc-dot", cx: x(n), cy: y(t), r: 3.8 }));
  svg.append(s("text", { x: x(40), y: y(0.23) + 16, "font-size": 11.5, style: "fill: var(--np); font-weight: 650" }, "NumPy computing a * b"));

  const road = ROADS[on];
  const n = sizeFor((road.lo + road.hi) / 2);
  if (n) {
    svg.append(s("line", { class: "nc-cross", x1: x(n), x2: x(n), y1: y((road.lo + road.hi) / 2), y2: H - B }));
  }
  return svg;
}

export default async function mount(el) {
  let on = ROADS[el.dataset.road] ? el.dataset.road : "remember";
  const plot = h("div", { class: "w-stage" });
  const parts = h("div", {});
  const cost = stat("to write a * b down", "hot");
  const level = stat("NumPy computes a * b on as many elements");
  const note = h("p", { class: "w-note", "aria-live": "polite" });

  function render() {
    const road = ROADS[on];
    plot.replaceChildren(chart(on));
    // Bars are to the scale of the road shown; the numbers beside them compare roads.
    const widest = Math.max(...road.parts.map((p) => p[2]));
    parts.replaceChildren(h("p", { class: "nc-title" }, "Where the time goes"),
      ...road.parts.map(([label, lo, hi]) => h("div", { class: "nc-row" },
        h("div", { class: "lbl" }, h("span", {}, label), h("span", {}, lo === hi ? `${lo} µs` : `${lo}–${hi} µs`)),
        h("div", { class: "bar", style: `width: ${Math.max(1.5, (100 * (lo + hi)) / 2 / widest)}%` }))),
      h("p", { class: "w-note" }, road.what));
    cost.set(`${road.lo}–${road.hi} µs`);
    const n = sizeFor((road.lo + road.hi) / 2);
    level.set(n ? `about ${about(n)}` : "more than 8,192");
    note.textContent = n
      ? `Writing the node costs what NumPy takes to multiply about ${about(n)} elements. Below that size, a program run lazily spends more time writing its graph than NumPy spends computing it; above it, less.`
      : "Writing the node costs more than NumPy takes to multiply 8,192 elements, the largest size timed. Nearly all of it is asking questions whose answers are the same on every turn of a loop.";
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Road"),
      seg(Object.entries(ROADS).map(([value, r]) => ({ value, label: r.label })), on, (v) => { on = v; render(); })),
    h("div", { class: "nc-parts" }, h("div", {}, h("p", { class: "nc-title" }, "One node beside NumPy's work"), plot), parts),
    note,
    h("div", { class: "w-readout" }, cost, level),
  );
  render();
}
