// Which road a line of NumPy code takes when it meets a lazy array.
//
// The reader picks a spelling. The figure shows how it reaches the layer
// (which of Python's or NumPy's protocols delivers it), what the layer does
// with it (writes a node, has NumPy make it at once, computes a value that is
// needed, ...), what the line gets back, and the line the layer logs.
//
// Each entry was run on prototypes/lazy with x a lazy float64 array of shape
// (4, 3) that is pending, a a plain ndarray of the same shape, w an array the
// stand-in made, numpy the real module and np the stand-in. The results and
// log lines are the prototype's; the log names the line as "line 1" here.
//
// Attributes:
//   data-call  the index of the spelling shown first (default 0)

import { h, s } from "./util.js";

// How a call arrives at the layer.
const ARRIVE = [
  ["op", "x's own operator", "__mul__, __getitem__, __isub__"],
  ["ufunc", "__array_ufunc__", "NumPy's ufuncs hand it over"],
  ["func", "__array_function__", "NumPy's functions hand it over"],
  ["method", "a method or attribute of x", "x.mean, x.shape"],
  ["convert", "a conversion Python asks for", "__float__, __bool__, __format__"],
  ["array", "__array__", "NumPy wants an ndarray"],
  ["standin", "the stand-in, np", "a function that makes arrays"],
  ["never", "never reaches the layer", "no lazy array is involved"],
];
// What the layer does with it.
const DECIDE = [
  ["defer", "in the deferred set: write a node", "defer"],
  ["write", "a write: a node for the new contents", "defer"],
  ["error", "shapes do not fit: NumPy's error, now", "error"],
  ["known", "known without computing", "known"],
  ["atonce", "outside the set: NumPy makes it at once", "atonce"],
  ["needed", "a value is needed: compute it now", "needed"],
  ["gather", "asked for: compute it now", "known"],
  ["leaf", "made by the layer, with its values", "known"],
  ["plain", "plain NumPy", "plain"],
];
// What the line gets back.
const RESULT = [
  ["pending", "a LazyArray, pending"],
  ["computed", "a LazyArray with its values"],
  ["python", "a Python value"],
  ["ndarray", "an ndarray"],
  ["raise", "an exception"],
];

const GROUPS = ["Operators", "NumPy's functions", "Methods and attributes", "Getting a value", "Writes", "Making arrays"];

// group, code, arrival, decision, result, what comes back, log line or null, explanation
const CALLS = [
  ["Operators", "x * 2", "op", "defer", "pending", "LazyArray(float64, shape=(4, 3), pending)", null,
    "Python calls x's own multiplication. The layer writes a node for np.multiply and works out its shape and element type from the operands' alone."],
  ["Operators", "a + x", "ufunc", "defer", "pending", "LazyArray(float64, shape=(4, 3), pending)", null,
    "a is an ordinary ndarray, so Python calls a's addition first. NumPy sees an operand it does not know and hands the whole call to x, through __array_ufunc__. The result is lazy all the same."],
  ["Operators", "x.T @ x", "op", "defer", "pending", "LazyArray(float64, shape=(3, 3), pending)", null,
    "Two nodes: a transpose, which is a view, and a matrix product. The shape (3, 3) follows from the matrix product's rule."],
  ["Operators", "x[x[:, 0] > 4]", "op", "defer", "pending", "LazyArray(float64, shape=(?, 3), pending)", null,
    "A selection by a mask that is still pending. How many rows it keeps is a value, so that length is shown as ?. It is deferred all the same."],
  ["Operators", "x[x[:, 0] > 4][:2]", "op", "atonce", "computed", "LazyArray(float64, shape=(2, 3), computed)", "eager: x[index] at line 1",
    "The first two of a number of rows nobody knows yet: the answer's shape depends on that number. The layer does not guess. The selection is computed and NumPy makes the slice."],
  ["Operators", "x + numpy.arange(4.0)", "op", "error", "raise", "ValueError: operands could not be broadcast together with shapes (4,3) (4,)", null,
    "Shapes are worked out when the line runs, so a shape that does not fit raises here, on this line, with NumPy's own message. Only the values wait."],
  ["NumPy's functions", "numpy.exp(x)", "ufunc", "defer", "pending", "LazyArray(float64, shape=(4, 3), pending)", null,
    "This is the real numpy module, in a file that has never heard of the layer. NumPy hands the call to the lazy array, so the operation is deferred anyway."],
  ["NumPy's functions", "numpy.sum(x, axis=0)", "func", "defer", "pending", "LazyArray(float64, shape=(3,), pending)", null,
    "np.sum is a function, not a ufunc. NumPy hands it over through __array_function__ with the arguments as they were written, and the node keeps them that way."],
  ["NumPy's functions", "numpy.argsort(x)", "func", "atonce", "computed", "LazyArray(int64, shape=(4, 3), computed)", "eager: np.argsort at line 1",
    "Sorting is not in the set of operations the layer defers. NumPy makes the call at once, on computed operands, and the result is a lazy array that has its values. The log says so, once per line of code."],
  ["NumPy's functions", "numpy.sum(x, where=x > 0)", "func", "atonce", "computed", "LazyArray(float64, shape=(), computed)", "eager: np.sum at line 1",
    "np.sum is deferred, but not with where=. A call the rules do not recognise in every detail is made at once."],
  ["Methods and attributes", "x.mean(axis=0)", "method", "defer", "pending", "LazyArray(float64, shape=(3,), pending)", null,
    "A reduction, deferred. When it is computed, the node calls value.mean(axis=0), the method as written, which matters for what serves the call (see With install())."],
  ["Methods and attributes", "x.shape", "method", "known", "python", "(4, 3)", null,
    "The shape and the element type are known from the moment the line that made x ran. Asking for them computes nothing."],
  ["Methods and attributes", "print(x)", "convert", "known", "python", "LazyArray(float64, shape=(4, 3), pending)", null,
    "A description, never the values. Printing a lazy array computes nothing."],
  ["Getting a value", "float(x.sum())", "convert", "needed", "python", "66.0", "needed: float() at line 1",
    "A Python number has to exist now. The layer computes what the sum needs, and no more, gives NumPy's answer, and logs the line. It does not refuse."],
  ["Getting a value", "if x.sum() > 30:", "convert", "needed", "python", "True", "needed: A truth test (if, while, and, or, not) at line 1",
    "if needs True or False on this line. The comparison is computed, as one graph, and the program goes on."],
  ["Getting a value", "f\"{x.sum():.2f}\"", "convert", "needed", "python", "'66.00'", "needed: Formatting with '.2f' at line 1",
    "Formatting with a specification needs the number. Without one, f\"{x}\" is the description and computes nothing."],
  ["Getting a value", "numpy.asarray(x)", "array", "needed", "ndarray", "array([[ 0.,  1.,  2.], …])", "needed: Conversion to an ndarray at line 1",
    "NumPy, or any library built on it, asks x for an ndarray. Whoever receives one may write into it, so everything pending is computed first. What follows from the ndarray is plain NumPy."],
  ["Getting a value", "x.gather()", "method", "gather", "ndarray", "array([[ 0.,  1.,  2.], …])", null,
    "The way to ask. It computes x and what it needs and returns exactly what NumPy would have returned. Nothing is logged: the program said where it wanted its values."],
  ["Writes", "w -= 1.0", "op", "write", "pending", "LazyArray(float64, shape=(3,), pending)", null,
    "w was made by the stand-in, so nothing outside the layer knows its memory. The write is a node for w's new contents, and w stands for it from now on. A loop of such writes is one graph."],
  ["Writes", "a += x", "ufunc", "needed", "ndarray", "a is changed in place", "needed: np.add at line 1",
    "a is an ordinary ndarray, and NumPy writes into it on this line. x's values have to exist, and everything pending is computed first."],
  ["Making arrays", "np.zeros(3)", "standin", "leaf", "computed", "LazyArray(float64, shape=(3,), computed)", null,
    "The stand-in's own function. The array has its values; it is lazy in that what is computed from it is written down, and a write into it can wait."],
  ["Making arrays", "numpy.zeros(3)", "never", "plain", "ndarray", "array([0., 0., 0.])", null,
    "The real module, given no lazy array, never meets the layer. A helper that makes its own arrays makes ordinary ones, until they meet a lazy array."],
];

export const css = `
.lr-list { display: grid; grid-template-columns: max-content 1fr; gap: .35rem .8rem; align-items: baseline; margin-bottom: 1rem; }
@media (max-width: 560px) { .lr-list { grid-template-columns: 1fr; } }
.lr-group-title { font-size: .68rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0; }
.lr-chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.lr-pick { font-family: var(--mono); font-size: .74rem; padding: .22rem .5rem; border-radius: 6px;
  border: 1px solid var(--rule-strong); background: var(--bg-raised); color: var(--ink); cursor: pointer; white-space: nowrap; }
.lr-pick:hover { border-color: var(--hot); }
.lr-pick.on { background: var(--ink); color: var(--bg); border-color: var(--ink); }
.lr-line { font-family: var(--mono); font-size: .95rem; background: var(--code-bg); padding: .5rem .8rem; border-radius: 8px; margin-bottom: .7rem; overflow-x: auto; white-space: pre; }
.lr-stage { overflow-x: auto; }
.lr-stage svg { display: block; width: 100%; min-width: 30rem; height: auto; }
.lr-box rect { fill: var(--bg-raised); stroke: var(--rule); stroke-width: 1; transition: fill .25s, stroke .25s; }
.lr-box text { font-size: 11.5px; fill: var(--muted); transition: fill .25s; }
.lr-box .sub { font-size: 9.5px; }
.lr-box.on rect { stroke-width: 2; }
.lr-box.on text { fill: var(--ink); font-weight: 650; }
.lr-box.on .sub { font-weight: 400; fill: var(--muted); }
.lr-col { font-size: 10.5px; font-weight: 700; letter-spacing: .06em; fill: var(--muted); text-transform: uppercase; }
.lr-path { fill: none; stroke-width: 2.4; }
.lr-got { display: grid; grid-template-columns: auto 1fr; gap: .25rem .8rem; margin-top: .8rem; font-size: .84rem; align-items: baseline; }
.lr-got dt { font-size: .7rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.lr-got dd { margin: 0; font-family: var(--mono); font-size: .78rem; overflow-x: auto; white-space: pre; }
.lr-got dd.log { color: var(--bad); }
.lr-got dd.none { color: var(--muted); font-family: var(--sans); font-style: italic; }
`;

// The colour a decision is drawn in.
const TONE = {
  defer: "var(--np)", known: "var(--good)", atonce: "var(--k-atonce)", needed: "var(--hot)", error: "var(--bad)", plain: "var(--muted)",
};
const SOFT = {
  defer: "var(--np-soft)", known: "var(--good-soft)", atonce: "var(--k-atonce-soft)", needed: "var(--hot-soft)", error: "var(--bad-soft)", plain: "var(--panel)",
};

// Geometry of the drawing, in SVG units.
const W = 680, TOP = 26, ROW = 41, BOX_H = 33;
const COLS = [{ x: 4, w: 190 }, { x: 224, w: 250 }, { x: 504, w: 172 }];

function column(svg, col, items, on, tone, soft, title) {
  const { x, w } = COLS[col];
  svg.append(s("text", { class: "lr-col", x, y: 14 }, title));
  const centres = new Map();
  items.forEach(([key, label, sub], i) => {
    const y = TOP + i * ROW;
    const active = key === on;
    const g = s("g", { class: `lr-box${active ? " on" : ""}` });
    g.append(s("rect", { x, y, width: w, height: BOX_H, rx: 7, style: active ? `stroke: ${tone}; fill: ${soft}` : "" }));
    g.append(s("text", { x: x + 9, y: y + (sub && col === 0 ? 14 : 20.5) }, label));
    if (sub && col === 0) g.append(s("text", { class: "sub", x: x + 9, y: y + 26 }, sub));
    svg.append(g);
    centres.set(key, y + BOX_H / 2);
  });
  return centres;
}

function diagram(call) {
  const [, , arrive, decide, result] = call;
  const tone = TONE[DECIDE.find((d) => d[0] === decide)[2]];
  const soft = SOFT[DECIDE.find((d) => d[0] === decide)[2]];
  const rows = Math.max(ARRIVE.length, DECIDE.length, RESULT.length);
  const H = TOP + rows * ROW;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Arrives as ${ARRIVE.find((a) => a[0] === arrive)[1]}; the layer: ${DECIDE.find((d) => d[0] === decide)[1]}; the line gets ${RESULT.find((r) => r[0] === result)[1]}` });
  const paths = s("g", {});
  svg.append(paths);
  const a = column(svg, 0, ARRIVE, arrive, tone, soft, "How it arrives");
  const d = column(svg, 1, DECIDE, decide, tone, soft, "What the layer does");
  const r = column(svg, 2, RESULT, result, tone, soft, "What the line gets");
  const link = (x1, y1, x2, y2) => {
    const mid = (x1 + x2) / 2;
    paths.append(s("path", { class: "lr-path", style: `stroke: ${tone}`, d: `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}` }));
  };
  link(COLS[0].x + COLS[0].w, a.get(arrive), COLS[1].x, d.get(decide));
  link(COLS[1].x + COLS[1].w, d.get(decide), COLS[2].x, r.get(result));
  return svg;
}

export default async function mount(el) {
  let index = Math.min(Math.max(Number(el.dataset.call) || 0, 0), CALLS.length - 1);
  const list = h("div", { class: "lr-list", role: "group", "aria-label": "Lines of code" });
  const line = h("div", { class: "lr-line" });
  const stage = h("div", { class: "lr-stage" });
  const got = h("dl", { class: "lr-got" });
  const note = h("p", { class: "w-note", "aria-live": "polite" });
  const picks = [];

  for (const group of GROUPS) {
    const chips = h("div", { class: "lr-chips" });
    CALLS.forEach((call, i) => {
      if (call[0] !== group) return;
      const b = h("button", { type: "button", class: "lr-pick", onclick: () => { index = i; render(); } }, call[1]);
      picks[i] = b;
      chips.append(b);
    });
    list.append(h("p", { class: "lr-group-title" }, group), chips);
  }

  function render() {
    const call = CALLS[index];
    picks.forEach((b, i) => { b.classList.toggle("on", i === index); b.setAttribute("aria-pressed", i === index); });
    line.textContent = call[1];
    stage.replaceChildren(diagram(call));
    got.replaceChildren(
      h("dt", {}, "Gets"), h("dd", {}, call[5]),
      h("dt", {}, "Logs"), call[6] ? h("dd", { class: "log" }, call[6]) : h("dd", { class: "none" }, "nothing"),
    );
    note.textContent = call[7];
  }

  el.append(list, line, stage, got, note);
  render();
}
