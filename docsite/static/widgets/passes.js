// What computing an expression one call at a time costs in memory traffic.
//
// NumPy computes each operation of an expression on its own: it reads the
// operands from memory, writes a whole temporary array, and the next
// operation reads that back. A loop that computed the whole expression for
// one element at a time would read each input once and write the result
// once. The figure counts the bytes each way, for three expressions and an
// array size the reader chooses.
//
// This is arithmetic, not a measurement: bytes read and written by each call,
// with float64 elements (8 bytes) and booleans (1 byte). Nothing in the lazy
// layer's first build computes this way; the figure shows what a graph would
// make possible. The cache sizes are the Apple M1 Pro's, as in the chapter on
// the machine.
//
// Attributes:
//   data-expr  the expression shown first: exp, gelu or leaky (default gelu)

import { h, range, seg, stat, fmtN } from "./util.js";

// Bytes of one element.
const F64 = 8, BOOL = 1;
// Cache sizes of an Apple M1 Pro performance core, in bytes.
const L1 = 128 * 1024, L2 = 12 * 1024 * 1024;

// Each operation: what it is, the bytes per element it reads, and writes.
const EXPRESSIONS = {
  exp: {
    label: "np.exp(a * b + c).sum()",
    inputs: ["a", "b", "c"], inBytes: 3 * F64, outBytes: 0,
    ops: [
      ["a * b", 2 * F64, F64],
      ["… + c", 2 * F64, F64],
      ["np.exp(…)", F64, F64],
      ["….sum()", F64, 0],
    ],
  },
  gelu: {
    label: "0.5 * x * (1 + np.tanh(k * (x + 0.044715 * x ** 3)))",
    inputs: ["x"], inBytes: F64, outBytes: F64,
    ops: [
      ["0.5 * x", F64, F64],
      ["x ** 3", F64, F64],
      ["0.044715 * …", F64, F64],
      ["x + …", 2 * F64, F64],
      ["k * …", F64, F64],
      ["np.tanh(…)", F64, F64],
      ["1 + …", F64, F64],
      ["… * …", 2 * F64, F64],
    ],
  },
  leaky: {
    label: "np.where(x > 0, x, 0.01 * x)",
    inputs: ["x"], inBytes: F64, outBytes: F64,
    ops: [
      ["x > 0", F64, BOOL],
      ["0.01 * x", F64, F64],
      ["np.where(…)", BOOL + 2 * F64, F64],
    ],
  },
};

export const css = `
.ps-expr { font-family: var(--mono); font-size: .86rem; background: var(--code-bg); padding: .45rem .75rem; border-radius: 8px; overflow-x: auto; white-space: pre; margin-bottom: .9rem; }
.ps-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem 1.4rem; }
@media (max-width: 640px) { .ps-cols { grid-template-columns: 1fr; } }
.ps-title { font-size: .72rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 .4rem; }
.ps-row { display: grid; grid-template-columns: 8.4rem 1fr; gap: .5rem; align-items: center; margin-bottom: 4px; font-size: .76rem; }
.ps-row code { font-family: var(--mono); font-size: .72rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border: 0; background: none; padding: 0; }
.ps-bar { display: flex; height: 1.05rem; border-radius: 4px; overflow: hidden; background: var(--panel); }
.ps-r { background: var(--np); }
.ps-w { background: var(--hot); }
.ps-sum { display: flex; gap: .5rem; align-items: center; margin-top: .55rem; font-size: .78rem; color: var(--muted); }
.ps-sum b { color: var(--ink); font-variant-numeric: tabular-nums; }
.ps-key { display: flex; gap: 1rem; font-size: .74rem; color: var(--muted); margin-top: .7rem; }
.ps-key i { display: inline-block; width: .75rem; height: .75rem; border-radius: 2px; margin-right: .3rem; vertical-align: -1px; }
.ps-where { display: inline-block; padding: .1rem .45rem; border-radius: 999px; font-size: .72rem; font-weight: 650; }
`;

// A size in bytes, in decimal units: a million float64 values are 8 MB.
function bytes(n) {
  if (n < 1e3) return `${n} B`;
  if (n < 1e6) return `${Number((n / 1e3).toPrecision(3))} KB`;
  if (n < 1e9) return `${Number((n / 1e6).toPrecision(3))} MB`;
  return `${Number((n / 1e9).toPrecision(3))} GB`;
}

// Where an array of this many bytes lives while it is read again.
function home(size) {
  if (size <= L1) return ["first-level cache", "var(--good-soft)", "var(--good)"];
  if (size <= L2) return ["second-level cache", "var(--np-soft)", "var(--np)"];
  return ["main memory", "var(--hot-soft)", "var(--hot)"];
}

function bars(rows, widest) {
  return rows.map(([name, read, written]) => h("div", { class: "ps-row" },
    h("code", { title: name }, name),
    h("div", { class: "ps-bar", title: `reads ${bytes(read)}, writes ${bytes(written)}` },
      h("span", { class: "ps-r", style: `width: ${(100 * read) / widest}%` }),
      h("span", { class: "ps-w", style: `width: ${(100 * written) / widest}%` }))));
}

export default async function mount(el) {
  let key = EXPRESSIONS[el.dataset.expr] ? el.dataset.expr : "gelu";
  let exponent = 6;

  const expr = h("div", { class: "ps-expr" });
  const eager = h("div", {});
  const fused = h("div", {});
  const ratio = stat("times the traffic, one call at a time", "hot");
  const temps = stat("temporary arrays NumPy makes");
  const moved = stat("moved one call at a time");
  const once = stat("moved in one pass", "good");
  const note = h("p", { class: "w-note", "aria-live": "polite" });

  function render() {
    const e = EXPRESSIONS[key];
    const n = Math.round(10 ** exponent);
    const rows = e.ops.map(([name, r, w]) => [name, r * n, w * n]);
    const total = rows.reduce((sum, [, r, w]) => sum + r + w, 0);
    const one = (e.inBytes + e.outBytes) * n;
    const widest = Math.max(...rows.map(([, r, w]) => r + w), one);
    const temp = F64 * n;
    const [where, soft, tone] = home(temp);

    expr.textContent = e.label;
    eager.replaceChildren(h("p", { class: "ps-title" }, "One call at a time, as NumPy does"), ...bars(rows, widest),
      h("div", { class: "ps-sum" }, "In all:", h("b", {}, bytes(total))));
    fused.replaceChildren(h("p", { class: "ps-title" }, "The whole expression in one pass"),
      ...bars([[`${e.inputs.join(", ")} → ${e.outBytes ? "result" : "a number"}`, e.inBytes * n, e.outBytes * n]], widest),
      h("div", { class: "ps-sum" }, "In all:", h("b", {}, bytes(one))),
      h("div", { class: "ps-sum" }, "Each temporary is", h("b", {}, bytes(temp)), "and lives in",
        h("span", { class: "ps-where", style: `background: ${soft}; color: ${tone}` }, where)));

    ratio.set(`${(total / one).toFixed(1)}x`);
    // Every call but the last makes an array that only the next call reads.
    temps.set(String(e.ops.length - 1));
    moved.set(bytes(total));
    once.set(bytes(one));
    note.innerHTML = temp > L2
      ? `At ${fmtN(n)} elements a temporary does not fit in any cache, so every pass goes to main memory, and memory is the limit. There, time follows bytes: computing the expression in one pass could take up to <b>${(total / one).toFixed(1)}x</b> less time. That is an upper bound from counting, not a measurement.`
      : `At ${fmtN(n)} elements a temporary stays in the ${where}, so the extra passes are cheap and the instructions are the limit. Fusing would save the passes and NumPy's fixed cost per call, which at this size matters more than the bytes.`;
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Expression"),
      seg([{ value: "exp", label: "exp of a sum" }, { value: "gelu", label: "GELU" }, { value: "leaky", label: "leaky ReLU" }], key, (v) => { key = v; render(); }),
      range({ label: "Elements", min: 3, max: 8, step: 0.25, value: exponent, format: (v) => fmtN(Math.round(10 ** v)), onInput: (v) => { exponent = v; render(); } })),
    expr,
    h("div", { class: "ps-cols" }, eager, fused),
    h("div", { class: "ps-key" }, h("span", {}, h("i", { style: "background: var(--np)" }), "bytes read"), h("span", {}, h("i", { style: "background: var(--hot)" }), "bytes written")),
    note,
    h("div", { class: "w-readout" }, ratio, temps, moved, once),
  );
  render();
}
