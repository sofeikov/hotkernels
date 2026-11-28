// Where a training loop's graph is cut.
//
// A linear regression by gradient descent, 30 steps on 200 examples of 8
// features. The reader chooses how often the loop looks at its loss and how it
// updates its weights. The strip shows the graphs the layer computed, in the
// order it computed them, each as wide as its number of operations, coloured
// by what made the layer compute it.
//
// The graphs are the prototype's: the loop was run on prototypes/lazy under
// record_graphs() for each of the twelve settings, and the number of
// operations in each graph it recorded is below, with how many operations
// were written and dropped without ever being computed (NumPy 2.5.5, Apple
// M1 Pro). Each entry is [trigger, operations] or [trigger, operations, times].
//
// Attributes:
//   data-look   how often the loss is looked at: 0 (never), 10, 5 or 1 (default 5)
//   data-write  inplace, rebind or own (default inplace)

import { h, seg, stat } from "./util.js";

// What made the layer compute a graph, by the letter used in GRAPHS.
const TRIGGERS = {
  g: { label: "gather()", tone: "var(--np)", soft: "var(--np-soft)" },
  n: { label: "a value needed: the loss formatted", tone: "var(--hot)", soft: "var(--hot-soft)" },
  w: { label: "a write made at once", tone: "var(--k-atonce)", soft: "var(--k-atonce-soft)" },
  p: { label: "still pending at the end", tone: "var(--muted)", soft: "var(--panel)" },
};

// The recorded graphs, by "look/write", and the operations dropped unseen.
const LAZY_GRAPHS = {
  0: { graphs: [["g", 184], ["p", 2]], dropped: 58 },
  10: { graphs: [["n", 8], ["n", 62, 2], ["g", 58], ["p", 2]], dropped: 52 },
  5: { graphs: [["n", 8], ["n", 32, 5], ["g", 28], ["p", 2]], dropped: 46 },
  1: { graphs: [["n", 8, 30], ["g", 4]], dropped: 0 },
};
// With the weights an ndarray of the user's, every write is made at once, and
// that computes everything pending, the loss included, whatever is looked at.
const OWN_GRAPHS = { graphs: [["w", 11], ["w", 7, 29]], dropped: 0 };

const STEPS = 30;

export const css = `
.lc-body { display: grid; grid-template-columns: minmax(0, 1fr); gap: .9rem; }
.lc-code { margin: 0; padding: .55rem 0; background: var(--code-bg); border-radius: 10px; font-family: var(--mono); font-size: .78rem; line-height: 1.7; overflow-x: auto; }
.lc-code div { display: flex; gap: .7rem; padding: 0 .8rem; white-space: pre; }
.lc-code .ln { color: var(--muted); min-width: 1.1rem; text-align: right; user-select: none; }
.lc-code .changed { background: var(--hot-soft); }
.lc-code .off { color: var(--muted); text-decoration: line-through; text-decoration-color: var(--rule-strong); }
.lc-mark { margin-left: auto; padding-left: 1rem; font-family: var(--sans); font-size: .72rem; font-weight: 650; white-space: nowrap; }
.lc-strip-title { font-size: .7rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .35rem; }
.lc-strip { display: flex; gap: 2px; height: 3.1rem; align-items: stretch; }
.lc-block { border-radius: 4px; border: 1.5px solid; min-width: 3px; display: grid; place-items: center; font-size: .7rem; font-weight: 700; overflow: hidden; transition: flex-grow .4s; }
.lc-block.drop { border-style: dashed; background: repeating-linear-gradient(135deg, transparent 0 5px, var(--rule) 5px 6px); border-color: var(--rule-strong); color: var(--muted); }
.lc-scale { display: flex; justify-content: space-between; font-size: .7rem; color: var(--muted); margin-top: .25rem; font-variant-numeric: tabular-nums; }
.lc-legend { display: flex; flex-wrap: wrap; gap: .3rem .9rem; margin-top: .5rem; font-size: .74rem; color: var(--muted); }
.lc-legend span { display: inline-flex; align-items: center; gap: .3rem; }
.lc-legend i { width: .8rem; height: .8rem; border-radius: 3px; border: 1.5px solid; display: inline-block; }
`;

// Expands [trigger, operations, times] into one entry per graph.
const expand = (graphs) => graphs.flatMap(([t, ops, times = 1]) => Array.from({ length: times }, () => [t, ops]));

function listing(look, write) {
  const lines = [
    ["X = (X - X.mean(axis=0)) / X.std(axis=0)", ""],
    [write === "own" ? "w = np.asarray(my_weights)        # your own ndarray" : "w = np.zeros(8)", write === "own" ? "changed" : ""],
    [`for step in range(${STEPS}):`, ""],
    ["    residual = X @ w - y", ""],
    ["    loss = (residual ** 2).mean()", ""],
    [write === "rebind" ? "    w = w - 0.05 * (2 / 200) * (X.T @ residual)" : "    w -= 0.05 * (2 / 200) * (X.T @ residual)", write === "rebind" ? "changed" : ""],
    [`    if step % ${look || 5} == 0:`, look ? "" : "off"],
    ["        print(f\"{loss:.4f}\")", look ? "" : "off"],
    ["w.gather()", ""],
  ];
  return lines;
}

export default async function mount(el) {
  let look = [0, 10, 5, 1].includes(Number(el.dataset.look)) ? Number(el.dataset.look) : 5;
  let write = ["inplace", "rebind", "own"].includes(el.dataset.write) ? el.dataset.write : "inplace";

  const code = h("pre", { class: "lc-code", "aria-label": "The loop" });
  const strip = h("div", { class: "lc-strip", role: "img" });
  const scale = h("div", { class: "lc-scale" });
  const graphs = stat("graphs computed", "hot");
  const largest = stat("operations in the largest");
  const computed = stat("operations computed");
  const dropped = stat("written, never computed");
  const note = h("p", { class: "w-note", "aria-live": "polite" });

  function render() {
    const data = write === "own" ? OWN_GRAPHS : LAZY_GRAPHS[look];
    const list = expand(data.graphs);
    const total = list.reduce((sum, [, ops]) => sum + ops, 0);
    const count = (t) => list.filter(([k]) => k === t).length;

    // The listing, with what each line made the layer compute.
    const marks = new Map();
    if (count("n")) marks.set(7, [`needed → ${count("n")} graph${count("n") === 1 ? "" : "s"}`, TRIGGERS.n.tone]);
    if (count("w")) marks.set(5, [`write → ${count("w")} graphs`, TRIGGERS.w.tone]);
    marks.set(8, [`gather → ${count("g")} graph${count("g") === 1 ? "" : "s"}`, TRIGGERS.g.tone]);
    code.replaceChildren(...listing(look, write).map(([text, cls], i) => {
      const mark = marks.get(i);
      return h("div", { class: cls }, h("span", { class: "ln" }, i + 1), text,
        mark ? h("span", { class: "lc-mark", style: `color: ${mark[1]}` }, mark[0]) : null);
    }));

    strip.replaceChildren(...list.map(([t, ops], i) => h("div", {
      class: "lc-block", title: `Graph ${i + 1}: ${ops} operations, ${TRIGGERS[t].label}`,
      style: `flex: ${ops} 1 0; background: ${TRIGGERS[t].soft}; border-color: ${TRIGGERS[t].tone}; color: ${TRIGGERS[t].tone}`,
    }, ops >= 20 ? String(ops) : "")));
    if (data.dropped) strip.append(h("div", { class: "lc-block drop", title: `${data.dropped} operations written and never computed`, style: `flex: ${data.dropped} 1 0` }, String(data.dropped)));
    strip.setAttribute("aria-label", `${list.length} graphs, the largest of ${Math.max(...list.map((g) => g[1]))} operations`);
    scale.replaceChildren(h("span", {}, "first operation"), h("span", {}, `${total + data.dropped} operations written`));

    graphs.set(String(list.length));
    largest.set(String(Math.max(...list.map((g) => g[1]))));
    computed.set(String(total));
    dropped.set(String(data.dropped));

    if (write === "own") {
      note.textContent = "w is your own ndarray, taken in by reference. You hold its memory, so NumPy's meaning of a write, that the memory changes there and then, has to be kept: each w -= … computes everything pending and writes into your array. The loop is cut at every step, into graphs of seven operations, and looking at the loss costs nothing more, since it is always computed already.";
    } else if (look === 0) {
      note.textContent = `Nothing asks for a value until the last line, so the 30 steps are one graph of 184 operations. ${write === "rebind" ? "w = w - … makes a new array each step" : "w -= … is a node like any other"}: the two spellings give the same graphs. The ${data.dropped} operations dropped are the 29 losses nobody looked at. They were written down and never computed.`;
    } else if (look === 1) {
      note.textContent = "Formatting the loss needs its value at every step, so every step is a graph of its own, of eight operations: the loss and the update of w that it reads. That is the program asking, and no rule of the layer can remove it.";
    } else {
      note.textContent = `Each print needs a value, and computes what the loss reads: the weights' updates since the last print and the loss itself. So the loop is cut every ${look} steps, into graphs of ${look === 5 ? 32 : 62} operations. The losses of the other steps are never computed.`;
    }
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Print the loss"),
      seg([{ value: 0, label: "never" }, { value: 10, label: "every 10 steps" }, { value: 5, label: "every 5" }, { value: 1, label: "every step" }], look, (v) => { look = Number(v); render(); }),
      h("span", { class: "w-label" }, "Update w"),
      seg([{ value: "inplace", label: "w -= …" }, { value: "rebind", label: "w = w - …" }, { value: "own", label: "w -= … on your ndarray" }], write, (v) => { write = v; render(); })),
    h("div", { class: "lc-body" },
      code,
      h("div", {},
        h("p", { class: "lc-strip-title" }, "The graphs the layer computed, in order, each as wide as its operations"),
        strip, scale,
        h("div", { class: "lc-legend" },
          ...Object.values(TRIGGERS).map((t) => h("span", {}, h("i", { style: `background: ${t.soft}; border-color: ${t.tone}` }), t.label)),
          h("span", {}, h("i", { style: "background: repeating-linear-gradient(135deg, transparent 0 3px, var(--rule) 3px 4px); border-color: var(--rule-strong); border-style: dashed" }), "written, never computed")))),
    note,
    h("div", { class: "w-readout" }, graphs, largest, computed, dropped),
  );
  render();
}
