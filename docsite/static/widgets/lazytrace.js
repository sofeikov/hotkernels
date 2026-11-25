// A program run line by line, lazily or eagerly, with its graph beside it.
//
// Four short programs, each a list of steps. A step is one line running: it
// may write operations down (nodes), move a name to another node, compute
// some nodes together (one graph), print something, and log a line. In lazy
// mode a node is drawn hollow until it is computed; in eager mode every node
// is computed on the line that wrote it, as NumPy does.
//
// The steps are the prototype's behaviour, not a model of it: each program
// was run on prototypes/lazy under record_graphs(), and the nodes, shapes,
// graphs, printed text and log lines here are what it gave (NumPy 2.5.5,
// Apple M1 Pro). The log lines name the line of the listing, not of a file.
//
// Attributes:
//   data-program  the program shown first: expression, loop, needed or mask

import { h, s, seg, stat, button, reducedMotion } from "./util.js";

// Node kinds, as the prototype's recording names them, and how each is labelled.
const KINDS = {
  input: "an array with its values",
  elementwise: "element by element",
  reduction: "reduction",
  matmul: "matrix product",
  view: "view or index",
  write: "write into an array",
  atonce: "made at once by NumPy",
};

// Geometry of the drawing, in SVG units.
const NODE_W = 128, NODE_H = 36, LAYER_GAP = 22, COL_GAP = 18, PAD = 12, TAG_H = 16;
// Milliseconds between steps while playing.
const PLAY_MS = 1500;

// A node: id, the operation's name, its kind, its shape and element type, and the ids it reads.
const N = (id, name, kind, shape, dtype, ops = []) => ({ id, name, kind, shape, dtype, ops });

const PROGRAMS = {
  expression: {
    label: "An expression",
    code: [
      "x = np.arange(6.0)",
      "y = x * 2 + 1",
      "z = np.exp(y / 10)",
      "s = z.sum()",
      "print(s)",
      "print(s.shape, s.dtype)",
      "v = s.gather()",
    ],
    steps: [
      { line: 0, add: [N("a", "np.arange", "input", "(6,)", "float64")], names: { x: "a" },
        lazy: "np.arange makes an array that has its values from the start. It is a leaf of the graph: the place where data enters.",
        eager: "NumPy makes the array and fills it: six numbers in memory." },
      { line: 1, add: [N("m", "multiply", "elementwise", "(6,)", "float64", ["a"]), N("p", "add", "elementwise", "(6,)", "float64", ["m"])], names: { y: "p" },
        lazy: "Two operations are written down and nothing is computed. Their shape, (6,), and element type, float64, are known already: those follow from the operands' shapes and types, not from their values.",
        eager: "Two operations, two passes over the data: x * 2 makes a temporary array, and + 1 reads it and makes another." },
      { line: 2, add: [N("d", "divide", "elementwise", "(6,)", "float64", ["p"]), N("e", "exp", "elementwise", "(6,)", "float64", ["d"])], names: { z: "e" },
        lazy: "np.exp comes from NumPy itself, but NumPy sees that its argument is not an ndarray and hands the call to the lazy array. Another two nodes.",
        eager: "Two more passes, two more temporaries." },
      { line: 3, add: [N("s", "sum", "reduction", "()", "float64", ["e"])], names: { s: "s" },
        lazy: "A reduction. Its shape is (), a single number, and it is still a node like the others: nothing is computed.",
        eager: "The sum is computed: s is a number." },
      { line: 4, out: { lazy: "LazyArray(float64, shape=(), pending)", eager: "11.58127283892564" },
        lazy: "print describes a lazy array: its type, shape and state. It never shows values, so it computes nothing.",
        eager: "print shows the value." },
      { line: 5, out: { lazy: "() float64", eager: "() float64" },
        lazy: "Shape and element type are answered without computing.",
        eager: "The same answer." },
      { line: 6, compute: { ids: ["m", "p", "d", "e", "s"], trigger: "gather" }, out: { lazy: "np.float64(11.58127283892564)", eager: "(nothing to do: s has had its value since line 4)" },
        lazy: "gather() asks for the values. The five operations are computed together, in the order they were written, each with NumPy's own call: one graph. The result is what NumPy gives, bit for bit, as NumPy's own type.",
        eager: "With import numpy as np there is no gather: every value was computed on its line." },
    ],
  },
  loop: {
    label: "A loop that writes",
    code: [
      "w = np.zeros(3)",
      "before = w + 1",
      "for step in range(3):",
      "    w -= 0.1 * (w - 1)",
      "w.gather()",
      "before.gather()",
    ],
    steps: [
      { line: 0, add: [N("z", "np.zeros", "input", "(3,)", "float64")], names: { w: "z" },
        lazy: "An array of zeros, made by the layer, so nothing outside the layer can know its memory. That is what lets a write into it wait.",
        eager: "Three zeros in memory." },
      { line: 1, add: [N("b", "add", "elementwise", "(3,)", "float64", ["z"])], names: { before: "b" },
        lazy: "before reads what w holds now.",
        eager: "before is computed: [1, 1, 1]." },
      { line: 3, turn: 0, add: [N("s1", "subtract", "elementwise", "(3,)", "float64", ["z"]), N("m1", "multiply", "elementwise", "(3,)", "float64", ["s1"]), N("w1", "w -= …", "write", "(3,)", "float64", ["z", "m1"])], names: { w: "w1" },
        lazy: "w -= … would change memory in NumPy. Here it writes a node for \"what w held, with this subtracted\", and the name w now stands for that node. The old node is untouched, so before still reads the old contents.",
        eager: "NumPy subtracts into w's memory, in place." },
      { line: 3, turn: 1, add: [N("s2", "subtract", "elementwise", "(3,)", "float64", ["w1"]), N("m2", "multiply", "elementwise", "(3,)", "float64", ["s2"]), N("w2", "w -= …", "write", "(3,)", "float64", ["w1", "m2"])], names: { w: "w2" },
        lazy: "The second turn of the loop reads the node the first turn wrote. The loop is becoming one graph.",
        eager: "Another write into the same memory." },
      { line: 3, turn: 2, add: [N("s3", "subtract", "elementwise", "(3,)", "float64", ["w2"]), N("m3", "multiply", "elementwise", "(3,)", "float64", ["s3"]), N("w3", "w -= …", "write", "(3,)", "float64", ["w2", "m3"])], names: { w: "w3" },
        lazy: "Three turns, nine operations, nothing computed.",
        eager: "Three writes, nine operations, each computed as it ran." },
      { line: 4, compute: { ids: ["s1", "m1", "w1", "s2", "m2", "w2", "s3", "m3", "w3"], trigger: "gather" }, out: { lazy: "array([0.271, 0.271, 0.271])", eager: "(w is array([0.271, 0.271, 0.271]))" },
        lazy: "One gather computes the whole loop: one graph of nine operations.",
        eager: "Nothing left to compute." },
      { line: 5, compute: { ids: ["b"], trigger: "gather" }, out: { lazy: "array([1., 1., 1.])", eager: "(before is array([1., 1., 1.]))" },
        lazy: "before has what w held when its line ran, exactly as with NumPy. The writes made later did not reach it.",
        eager: "The same values: before was computed before the writes." },
    ],
  },
  needed: {
    label: "A value needed",
    code: [
      "x = np.arange(6.0)",
      "y = x * 2 + 1",
      "if y.sum() > 30:",
      "    y = y - 30",
      "order = np.argsort(-y)",
      "order.gather()",
    ],
    steps: [
      { line: 0, add: [N("a", "np.arange", "input", "(6,)", "float64")], names: { x: "a" },
        lazy: "A leaf, as before.", eager: "Six numbers in memory." },
      { line: 1, add: [N("m", "multiply", "elementwise", "(6,)", "float64", ["a"]), N("p", "add", "elementwise", "(6,)", "float64", ["m"])], names: { y: "p" },
        lazy: "Two nodes, nothing computed.", eager: "Two passes, two arrays." },
      { line: 2, add: [N("s", "sum", "reduction", "()", "float64", ["p"]), N("g", "greater", "elementwise", "()", "bool", ["s"])],
        compute: { ids: ["m", "p", "s", "g"], trigger: "needed" },
        log: "needed: A truth test (if, while, and, or, not) at line 3",
        lazy: "if needs True or False now, and no gather() was written. The layer does not refuse: it computes the four operations the test reads, as one graph, and logs the line. y.sum() was 36, so the test is True.",
        eager: "The sum is 36 and the test is True." },
      { line: 3, add: [N("d", "subtract", "elementwise", "(6,)", "float64", ["p"])], names: { y: "d" },
        lazy: "After the test, writing down starts again.", eager: "y is computed again." },
      { line: 4, add: [N("n", "negative", "elementwise", "(6,)", "float64", ["d"]), N("o", "np.argsort", "atonce", "(6,)", "int64", ["n"])], names: { order: "o" },
        compute: { ids: ["d", "n"], trigger: "at once", then: ["o"] },
        log: "eager: np.argsort at line 5",
        lazy: "np.argsort is not in the set of operations the layer defers. NumPy makes the call at once, on computed operands: the two pending operations it reads are computed first, a second graph. Its result is still a lazy array, one that already has its values.",
        eager: "The negation, then the sort." },
      { line: 5, out: { lazy: "array([5, 4, 3, 2, 1, 0])", eager: "(order is array([5, 4, 3, 2, 1, 0]))" },
        lazy: "Everything order needs has its values: gather computes nothing. Two graphs in all, each cut by something the program did.",
        eager: "Nothing to compute." },
    ],
  },
  mask: {
    label: "A mask",
    code: [
      "speed = np.array([3., 9., 1., 12.])",
      "fast = speed > 5",
      "chosen = speed[fast]",
      "m = chosen.mean()",
      "print(chosen)",
      "print(len(chosen))",
      "m.gather()",
    ],
    steps: [
      { line: 0, add: [N("a", "np.array", "input", "(4,)", "float64")], names: { speed: "a" },
        lazy: "Four numbers enter.", eager: "Four numbers in memory." },
      { line: 1, add: [N("g", "greater", "elementwise", "(4,)", "bool", ["a"])], names: { fast: "g" },
        lazy: "A mask, pending: nobody knows yet which elements are true.", eager: "The mask: [False, True, False, True]." },
      { line: 2, add: [N("i", "speed[fast]", "view", "(?,)", "float64", ["a", "g"])], names: { chosen: "i" },
        lazy: "How many elements a mask selects is a value: the number of trues. So the shape has a ? in it. The selection is still a node.",
        eager: "Two elements are selected." },
      { line: 3, add: [N("r", "mean", "reduction", "()", "float64", ["i"])], names: { m: "r" },
        lazy: "The mean of any number of elements is one number, so this shape is known: (). Work that does not depend on the ? stays pending.",
        eager: "The mean is computed: 10.5." },
      { line: 4, out: { lazy: "LazyArray(float64, shape=(?,), pending)", eager: "[ 9. 12.]" },
        lazy: "The description shows the length that is not known.", eager: "The values." },
      { line: 5, compute: { ids: ["g", "i"], trigger: "needed" }, shapes: { i: "(2,)" }, out: { lazy: "2", eager: "2" },
        log: "needed: The shape of what a mask selects at line 6",
        lazy: "Asking for the length is asking for a value. The mask and the selection are computed, one graph, and the ? becomes 2. The mean is left pending: nothing asked for it.",
        eager: "Two." },
      { line: 6, compute: { ids: ["r"], trigger: "gather" }, out: { lazy: "np.float64(10.5)", eager: "(m is 10.5)" },
        lazy: "The gather computes the one operation still pending.", eager: "Nothing to compute." },
    ],
  },
};

export const css = `
.lt-body { display: grid; grid-template-columns: minmax(15rem, 1fr) minmax(0, 1.25fr); gap: 1rem; align-items: start; }
@media (max-width: 760px) { .lt-body { grid-template-columns: 1fr; } }
.lt-code { margin: 0; padding: .6rem 0; background: var(--code-bg); border-radius: 10px; font-family: var(--mono); font-size: .8rem; line-height: 1.75; overflow-x: auto; }
.lt-line { display: flex; gap: .7rem; padding: 0 .8rem; white-space: pre; border-left: 3px solid transparent; transition: background .2s; }
.lt-line .ln { color: var(--muted); min-width: 1.1rem; text-align: right; user-select: none; }
.lt-line.done { color: var(--ink); }
.lt-line.todo { color: var(--muted); }
.lt-line.now { background: var(--hot-soft); border-left-color: var(--hot); color: var(--ink); }
.lt-turn { margin-left: auto; padding-left: .8rem; font-family: var(--sans); font-size: .7rem; font-weight: 650; color: var(--hot); }
.lt-console { margin-top: .7rem; background: var(--panel); border-radius: 10px; padding: .55rem .8rem; font-family: var(--mono); font-size: .76rem; min-height: 4.4rem; line-height: 1.6; overflow-x: auto; }
.lt-console .lt-out { color: var(--ink); white-space: pre-wrap; }
.lt-console .lt-log { color: var(--bad); white-space: pre-wrap; }
.lt-console .lt-empty { color: var(--muted); font-family: var(--sans); font-style: italic; }
.lt-console-title, .lt-graph-title { font-size: .7rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .25rem; font-family: var(--sans); }
.lt-graph-title { margin: 0 0 .4rem; }
.lt-graph { background: var(--bg); border: 1px solid var(--rule); border-radius: 10px; padding: .4rem; }
.lt-graph svg { display: block; width: 100%; height: auto; }
.lt-edge { fill: none; stroke: var(--rule-strong); stroke-width: 1.4; transition: stroke .3s, opacity .3s; }
.lt-edge.computed { stroke: var(--muted); }
.lt-node rect.box { stroke-width: 1.5; transition: fill .35s, stroke .35s; }
.lt-node.pending rect.box { fill: var(--bg-raised); stroke-dasharray: 4 3; }
.lt-node text { font-family: var(--sans); }
.lt-node .nm { font-size: 12px; font-weight: 650; fill: var(--ink); }
.w .lt-node .sh { font-size: 10px; fill: var(--muted); font-family: var(--mono); }
.lt-node.fresh rect.box { stroke-width: 2.6; }
.lt-tag rect { fill: var(--ink); }
.w .lt-tag text { fill: var(--bg); font-size: 10px; font-weight: 700; font-family: var(--mono); }
.lt-badge circle { stroke: var(--bg); stroke-width: 1.5; }
.w .lt-badge text { font-size: 9.5px; font-weight: 750; fill: #fff; }
.lt-legend { display: flex; flex-wrap: wrap; gap: .3rem .9rem; margin-top: .55rem; font-size: .74rem; color: var(--muted); }
.lt-legend span { display: inline-flex; align-items: center; gap: .3rem; }
.lt-legend i { width: .8rem; height: .8rem; border-radius: 3px; border: 1.5px solid; display: inline-block; }
.lt-legend i.pend { border-style: dashed; border-color: var(--muted); background: var(--bg-raised); }
.lt-buttons { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .7rem; }
.lt-story { margin-top: .7rem; padding: .7rem .9rem; border-radius: 10px; background: var(--panel); font-size: .9rem; line-height: 1.5; min-height: 3.6rem; }
.lt-story b { color: var(--hot); }
`;

// Colours of a kind, read from widgets.css: [fill, line].
const kindColours = (kind) => [`var(--k-${kind}-soft)`, `var(--k-${kind})`];

// Places every node the program will ever write, so the drawing does not move as it grows.
function layout(nodes) {
  const depth = new Map();
  for (const node of nodes) depth.set(node.id, node.ops.length ? 1 + Math.max(...node.ops.map((o) => depth.get(o))) : 0);
  const layers = [];
  for (const node of nodes) (layers[depth.get(node.id)] ||= []).push(node);
  const widest = Math.max(...layers.map((l) => l.length));
  const width = Math.max(2 * PAD + widest * NODE_W + (widest - 1) * COL_GAP, 300);
  const at = new Map();
  layers.forEach((layer, d) => {
    const span = layer.length * NODE_W + (layer.length - 1) * COL_GAP;
    layer.forEach((node, i) => at.set(node.id, {
      x: (width - span) / 2 + i * (NODE_W + COL_GAP),
      y: PAD + TAG_H + d * (NODE_H + LAYER_GAP),
    }));
  });
  return { at, width, height: PAD * 2 + TAG_H + layers.length * NODE_H + (layers.length - 1) * LAYER_GAP };
}

// What the program looks like after its first `count` steps, in one mode.
function stateAfter(program, count, lazy) {
  const nodes = new Map(), names = new Map(), out = [], logs = [];
  let graphs = 0, computations = 0;
  const fresh = new Set();
  for (let i = 0; i < count; i++) {
    const step = program.steps[i];
    const last = i === count - 1;
    for (const node of step.add || []) {
      nodes.set(node.id, { ...node, computed: node.kind === "input" || !lazy, graph: null });
      if (!lazy && node.kind !== "input") computations += 1;
      if (last) fresh.add(node.id);
    }
    for (const [name, id] of Object.entries(step.names || {})) names.set(name, id);
    if (step.shapes) for (const [id, shape] of Object.entries(step.shapes)) nodes.get(id).shape = shape;
    if (lazy && step.compute) {
      graphs += 1;
      for (const id of step.compute.ids) Object.assign(nodes.get(id), { computed: true, graph: graphs });
      for (const id of step.compute.then || []) Object.assign(nodes.get(id), { computed: true });
      if (last) step.compute.ids.forEach((id) => fresh.add(id));
    }
    if (!lazy && step.shapes) for (const [id, shape] of Object.entries(step.shapes)) nodes.get(id).shape = shape;
    if (step.out) out.push({ line: step.line, text: step.out[lazy ? "lazy" : "eager"] });
    if (lazy && step.log) logs.push(step.log);
  }
  // In eager mode a selection's length is known on its line.
  if (!lazy) for (const step of program.steps) if (step.shapes) for (const [id, shape] of Object.entries(step.shapes)) if (nodes.has(id)) nodes.get(id).shape = shape;
  if (!lazy) graphs = computations;
  return { nodes, names, out, logs, graphs, fresh };
}

function draw(program, state, place) {
  const svg = s("svg", { viewBox: `0 0 ${place.width} ${place.height}`, role: "img",
    "aria-label": `The graph so far: ${state.nodes.size} arrays and operations` });
  const at = place.at;
  for (const node of state.nodes.values()) {
    for (const op of node.ops) {
      if (!state.nodes.has(op)) continue;
      const a = at.get(op), b = at.get(node.id);
      const x1 = a.x + NODE_W / 2, y1 = a.y + NODE_H, x2 = b.x + NODE_W / 2, y2 = b.y;
      const mid = (y1 + y2) / 2;
      svg.append(s("path", { class: `lt-edge${node.computed ? " computed" : ""}`, d: `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`,
        "marker-end": "url(#lt-arrow)" }));
    }
  }
  svg.prepend(s("defs", {}, s("marker", { id: "lt-arrow", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" },
    s("path", { d: "M0,0 L8,4 L0,8 z", style: "fill: var(--muted)" }))));

  const tags = new Map();
  for (const [name, id] of state.names) (tags.get(id) || tags.set(id, []).get(id)).push(name);

  for (const node of state.nodes.values()) {
    const { x, y } = at.get(node.id);
    const [fillColour, line] = kindColours(node.kind);
    const g = s("g", { class: `lt-node ${node.computed ? "computed" : "pending"}${state.fresh.has(node.id) ? " fresh" : ""}` });
    g.append(s("rect", { class: "box", x, y, width: NODE_W, height: NODE_H, rx: 7,
      style: `stroke: ${line};${node.computed ? ` fill: ${fillColour};` : ""}` }));
    g.append(s("text", { class: "nm", x: x + NODE_W / 2, y: y + 15, "text-anchor": "middle" }, node.name));
    g.append(s("text", { class: "sh", x: x + NODE_W / 2, y: y + 29, "text-anchor": "middle" }, `${node.shape} ${node.dtype}`));
    if (node.graph) {
      const badge = s("g", { class: "lt-badge" });
      badge.append(s("circle", { cx: x + NODE_W - 2, cy: y + 2, r: 8, style: `fill: var(--g${((node.graph - 1) % 3) + 1})` }));
      badge.append(s("text", { x: x + NODE_W - 2, y: y + 5.5, "text-anchor": "middle" }, node.graph));
      g.append(badge);
    }
    const names = tags.get(node.id);
    if (names) {
      const label = names.join(", ");
      const w = 10 + label.length * 6.2;
      const tag = s("g", { class: "lt-tag" });
      tag.append(s("rect", { x: x + 4, y: y - TAG_H + 2, width: w, height: TAG_H - 2, rx: 4 }));
      tag.append(s("text", { x: x + 4 + w / 2, y: y - 4, "text-anchor": "middle" }, label));
      g.append(tag);
    }
    svg.append(g);
  }
  return svg;
}

export default async function mount(el) {
  let key = PROGRAMS[el.dataset.program] ? el.dataset.program : "expression";
  let lazy = true;
  let count = 0;
  let timer = null;

  const codeBox = h("pre", { class: "lt-code", "aria-label": "The program" });
  const consoleBox = h("div", { class: "lt-console", "aria-live": "polite" });
  const graphBox = h("div", { class: "lt-graph" });
  const story = h("div", { class: "lt-story", "aria-live": "polite" });
  const written = stat("operations written");
  const pending = stat("pending", "hot");
  const computed = stat("computed");
  const graphs = stat("computations");
  const back = button("Back", () => { stop(); if (count > 0) { count -= 1; render(); } });
  const next = button("Step", () => { stop(); advance(); }, "primary");
  const play = button("Play", () => (timer ? stop() : start()));
  const reset = button("Reset", () => { stop(); count = 0; render(); });

  const legend = h("div", { class: "lt-legend" },
    h("span", {}, h("i", { class: "pend" }), "pending: written down, no values yet"),
    ...Object.entries(KINDS).map(([kind, label]) => h("span", {}, h("i", { style: `background: var(--k-${kind}-soft); border-color: var(--k-${kind})` }), label)));

  function advance() {
    const program = PROGRAMS[key];
    if (count < program.steps.length) { count += 1; render(); }
    return count < program.steps.length;
  }
  function start() {
    if (count >= PROGRAMS[key].steps.length) count = 0;
    play.textContent = "Pause";
    advance();
    timer = setInterval(() => { if (!advance()) stop(); }, reducedMotion ? PLAY_MS * 1.5 : PLAY_MS);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    play.textContent = "Play";
  }

  function render() {
    const program = PROGRAMS[key];
    const all = program.steps.flatMap((step) => step.add || []);
    const place = layout(all);
    const state = stateAfter(program, count, lazy);
    const step = count ? program.steps[count - 1] : null;

    const ran = new Set(program.steps.slice(0, count).map((st) => st.line));
    codeBox.replaceChildren(...program.code.map((text, i) => h("div", {
      class: `lt-line ${step && step.line === i ? "now" : ran.has(i) ? "done" : "todo"}`,
    }, h("span", { class: "ln" }, i + 1), text,
      step && step.line === i && step.turn !== undefined ? h("span", { class: "lt-turn" }, `turn ${step.turn + 1} of 3`) : null)));

    const lines = [
      ...state.out.map((o) => ({ kind: "out", text: o.text, line: o.line })),
    ];
    consoleBox.replaceChildren(h("p", { class: "lt-console-title" }, lazy ? "Output and the layer's log" : "Output"));
    if (!lines.length && !state.logs.length) consoleBox.append(h("div", { class: "lt-empty" }, "Nothing printed yet."));
    // Output and log lines in the order the program produced them.
    for (let i = 0; i < count; i++) {
      const st = program.steps[i];
      if (lazy && st.log) consoleBox.append(h("div", { class: "lt-log" }, st.log));
      if (st.out) consoleBox.append(h("div", { class: "lt-out" }, st.out[lazy ? "lazy" : "eager"]));
    }

    graphBox.replaceChildren(h("p", { class: "lt-graph-title" }, lazy ? "What the layer has written down" : "What NumPy has computed"), draw(program, state, place), legend);

    const nodes = [...state.nodes.values()].filter((n) => n.kind !== "input");
    written.set(String(nodes.length));
    pending.set(String(nodes.filter((n) => !n.computed).length));
    computed.set(String(nodes.filter((n) => n.computed).length));
    graphs.set(String(state.graphs));
    graphs.querySelector(".w-stat-l").textContent = lazy ? "graphs computed" : "separate computations";

    story.innerHTML = "";
    if (!step) story.append(lazy
      ? "Press Step to run the first line. With np = hotkernels.get_lazy(), each line writes its operations down instead of computing them."
      : "Press Step to run the first line. This is the same program with import numpy as np: every line computes at once.");
    else {
      const text = step[lazy ? "lazy" : "eager"];
      if (lazy && step.compute) story.append(h("b", {}, step.compute.trigger === "gather" ? "Gathered. " : step.compute.trigger === "needed" ? "A value was needed. " : "Made at once. "));
      story.append(text);
    }
    back.disabled = count === 0;
    next.disabled = count >= program.steps.length;
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Program"),
      seg(Object.entries(PROGRAMS).map(([value, p]) => ({ value, label: p.label })), key, (v) => { stop(); key = v; count = 0; render(); }),
      h("span", { class: "w-label" }, "Run with"),
      seg([{ value: "lazy", label: "get_lazy()" }, { value: "eager", label: "NumPy" }], "lazy", (v) => { lazy = v === "lazy"; render(); })),
    h("div", { class: "lt-body" },
      h("div", {}, h("div", { class: "lt-buttons" }, back, next, play, reset), codeBox, story, consoleBox),
      graphBox),
    h("div", { class: "w-readout" }, written, pending, computed, graphs),
  );
  render();
}
