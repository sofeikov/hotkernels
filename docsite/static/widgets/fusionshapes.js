// What mining the corpus found: where NumPy's time goes, and the shapes worth a kernel.
//
// Two views, switchable. "Where the time goes" is one bar per program, the
// time NumPy spent computing its nodes split by what the operations were:
// element by element in an expression of two or more operations, the same
// operations standing alone, other element-by-element calls (a mathematics
// library, another element type), matrix products, reductions, writes,
// indexing, and the rest. "The library, shape by shape" is the library chosen
// one shape at a time: each bar is what one more shape adds, and the line is
// the share of NumPy's time in expressions that the library saves so far. The
// miner (mine.py, `choose`) counts an expression of k operations as k passes
// of equal cost and a covering as one pass per shape or operation left over,
// so a share is of all the time in expressions, not of what fusing each
// expression whole would save; the design document calls it "the share of the
// removable passes". Picking a shape shows its
// name, the NumPy expression it stands for, and the programs that have it; a
// program can be picked to light up the shapes it has.
//
// Data: docs/lazy-fusion-design.md, "Mining" (the table of shares, large size,
// for the seventeen programs whose NumPy code is their own) and "The shapes"
// (the first twenty-four shapes, and "sixty remove 61%"), from
//   uv run --group apps python -m prototypes.lazy.programs.mine --size large
// on an Apple M1 Pro, NumPy 2.5.5. The column meanings are those of the tool's
// template, prototypes/lazy/templates/mine.txt.j2. Shapes 25 to 59 are not
// listed in the document, so the figure draws the sixtieth as a point by
// itself and nothing between. The numbers are copied by a script that reads
// the document; nothing was mined or timed for this figure.
//
// Each SHARES row: deferred, graphs, largest graph, ms, then the eight shares in %.
// Each SHAPES row: rank, share removed so far in %, operations, name, programs.
//
// Attributes:
//   data-view   shares or library (default library)
//   data-shape  the rank of the shape picked first (default 1)

import { h, s, seg } from "./util.js";

const SHARES = {
  "regression":[384,22,261,46.6,29.9,0.3,0,46.8,22.8,0,0,0.2],
  "attention":[220,1,220,29.1,4.7,11.7,57.8,21.2,4.1,0,0,0.5],
  "perceptron":[8854,565,33,38.8,26.4,3.6,6.1,22.8,10.5,26.6,2.5,1.6],
  "char_rnn":[32423,207,25387,60.7,39.4,6,7.7,30.9,2.9,5.6,0,7.5],
  "particle_filter":[2537,43,280,321.5,38.2,14.4,20.5,4.4,1.3,1.5,5.3,14.3],
  "veros_eos":[366,1,366,175.7,85.5,11,3.6,0,0,0,0,0],
  "veros_isoneutral":[718,7,255,227.4,77,0,22.7,0,0,0.1,0,0.2],
  "veros_tke":[382,7,272,85.3,86.1,3.2,4.5,0,0,6,0,0.3],
  "cavity_flow":[6920,2,6856,165.2,94.7,0,2.9,0,0,1.3,0,1],
  "hdiff":[40,1,40,109.1,93.5,6.4,0,0,0,0,0,0],
  "black_scholes":[28,2,25,104.1,20.9,7.5,71.7,0,0,0,0,0],
  "example":[13,9,4,20.9,0,2.3,11.5,0,17.7,1.7,43.6,23.2],
  "flame":[7418,416,53,117.5,7.4,4.3,17.5,0,0.1,35.4,35.2,0],
  "azimint_naive":[10001,1,10001,183,0,2.8,78.2,0,1.3,0.8,16.8,0],
  "mandelbrot1":[1205,2,1053,154.3,0,44.4,16.6,0,0,17.9,21.1,0],
  "mandelbrot2":[727,322,9,138.2,0,33,6.7,0,0,32.1,28.1,0],
  "make_decision":[491520,5,100000,545,0,13.7,29.5,0,20.1,21.8,14.9,0]};

const SHAPES = [[1,6.4,2,"add(V0,mul(V1,V2))",["flame","particle_filter","veros_eos","veros_isoneutral","veros_tke"]],[2,12.6,6,"sub(div(add(V0,mul(S0,add(V1,V2))),S1),mul(S2,V3))",["cavity_flow"]],[3,17.3,2,"add(V0,mul(S0,V1))",["attention","cavity_flow","flame","particle_filter","perceptron","regression","veros_eos","veros_isoneutral","veros_tke"]],[4,21,5,"sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))",["hdiff"]],[5,24.5,2,"mul(V0,sub(V1,V2))",["black_scholes","cavity_flow","flame","hdiff","veros_eos","veros_isoneutral","veros_tke"]],[6,27.6,6,"div(neg(add(V0,mul(V1,V2))),sub(min(S0,V3),S1))",["veros_isoneutral"]],[7,30.5,2,"mul(V0,mul(V1,V2))",["veros_eos","veros_isoneutral","veros_tke"]],[8,32.7,4,"where(gt(mul(V0,sub(V1,V2)),S0),S1,V3)",["hdiff"]],[9,34.4,2,"mul(S0,add(V0,V1))",["attention","cavity_flow","flame","particle_filter","regression","veros_isoneutral"]],[10,36.1,2,"mul(V0,add(S0,V1))",["attention","black_scholes","veros_eos"]],[11,37.8,5,"sub(V0,mul(V1,sub(add(V2,sub(V3,V4)),V5)))",["hdiff"]],[12,39.2,6,"add(V0,mul(V1,mul(mul(V2,mul(S0,V3)),sq(V4))))",["veros_isoneutral"]],[13,40.6,5,"add(V0,mul(max(S0,mul(V1,V2)),mul(V3,V4)))",["veros_isoneutral"]],[14,42,5,"sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))",["cavity_flow"]],[15,43.4,3,"add(V0,add(V1,add(V2,V3)))",["char_rnn","hdiff","veros_eos","veros_isoneutral"]],[16,44.6,2,"mul(V0,mul(S0,V1))",["attention","flame","veros_eos","veros_isoneutral","veros_tke"]],[17,45.8,2,"sub(V0,mul(S0,V1))",["cavity_flow","perceptron","regression","veros_tke"]],[18,47,4,"div(add(S0,neg(abs(V0))),S1)",["veros_isoneutral"]],[19,47.9,3,"add(sq(V0),sq(V1))",["flame","particle_filter"]],[20,48.7,2,"add(V0,add(S0,V1))",["veros_eos","veros_tke"]],[21,49.5,2,"div(sub(V0,V1),S0)",["cavity_flow","veros_tke"]],[22,50.2,4,"sub(sub(V0,mul(V1,V2)),mul(V3,V4))",["cavity_flow","veros_eos"]],[23,50.9,6,"add(mul(S0,add(V0,V1)),sub(V2,mul(S1,sub(V3,V4))))",["cavity_flow"]],[24,51.6,3,"mul(V0,add(V1,sub(V2,V3)))",["black_scholes","veros_eos"]]];

// What the design document says beyond the first twenty-four shapes.
const SIXTIETH = { rank: 60, share: 61 };
const HALF = { rank: 22, share: 50 };

// The eight classes of the shares table, in its order.
const CLASSES = [
  { label: "in expressions", color: "var(--hot)", note: "operations a fused kernel could make, in a tree of two or more" },
  { label: "simple, alone", color: "var(--k-elementwise)", note: "the same operations with nothing to fuse with" },
  { label: "other element by element", color: "var(--cc)", note: "a mathematics library, element types other than float64" },
  { label: "matrix products", color: "var(--k-matmul)" },
  { label: "reductions", color: "var(--k-reduction)" },
  { label: "writes", color: "var(--k-write)" },
  { label: "indexing", color: "var(--k-view)" },
  { label: "the rest", color: "var(--rule-strong)" },
];

// The colour of a shape by its number of operations.
const OPS_COLOR = { 2: "var(--np)", 3: "var(--k-view)", 4: "var(--k-reduction)", 5: "var(--k-matmul)", 6: "var(--hot)" };

// The library chart, in viewBox units; the axis is broken between shape 24 and shape 60.
const W = 640, H = 290, M = { l: 40, r: 14, t: 16, b: 34 };
const GAP_AT = 25.2, LAST_X = 27;

export const css = `
.fs-rows { display: grid; grid-template-columns: minmax(6.5rem, 9.5rem) minmax(0, 1fr) 3.6rem; gap: .28rem .6rem; align-items: center; }
.fs-name { font-family: var(--mono); font-size: .74rem; background: none; border: 0; padding: 0; text-align: left; color: var(--ink); cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fs-name.on, .fs-name:hover { color: var(--hot); }
.fs-name.on { font-weight: 700; }
.fs-bar { display: flex; height: 1.05rem; border-radius: 3px; overflow: hidden; cursor: pointer; }
.fs-bar i { display: block; height: 100%; }
.fs-bar.on { outline: 2px solid var(--ink); outline-offset: 1px; }
.fs-ms { font-size: .74rem; text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
.fs-legend { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font-size: .76rem; color: var(--muted); margin-top: .8rem; }
.fs-legend span { display: inline-flex; align-items: center; gap: .35rem; }
.fs-legend i { width: .8rem; height: .8rem; border-radius: 3px; display: inline-block; }
.fs-detail { margin-top: .9rem; }
.fs-detail table { border-collapse: collapse; font-size: .8rem; width: 100%; }
.fs-detail td { padding: .15rem .4rem; border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums; }
.fs-detail td:last-child { text-align: right; }
.fs-shape { font-family: var(--mono); font-size: .84rem; background: var(--code-bg); padding: .45rem .7rem; border-radius: 8px; overflow-x: auto; white-space: nowrap; margin-top: .3rem; }
.fs-shape b { color: var(--muted); font-family: var(--sans); font-size: .7rem; font-weight: 650; letter-spacing: .05em; text-transform: uppercase; margin-right: .6rem; }
.fs-progs { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .6rem; }
.fs-progs span { font-family: var(--mono); font-size: .7rem; padding: .12rem .4rem; border-radius: 5px; background: var(--panel); }
.fs-progs span.on { background: var(--hot-soft); color: var(--hot); }
.fs-col { cursor: pointer; }
.fs-col rect.hit { fill: transparent; }
.fs-col:hover rect.hit { fill: var(--panel); }
.fs-col.on rect.hit { fill: var(--hot-soft); }
.fs-col.dim rect.bar { opacity: .2; }
@media (max-width: 40rem) {
  .fs-rows { grid-template-columns: minmax(5rem, 7rem) minmax(0, 1fr) 3rem; }
  .fs-name { font-size: .66rem; }
}
`;

// Parses a pattern's name, e.g. add(V0,mul(S0,V1)), into a tree.
function parse(name) {
  let i = 0;
  function node() {
    const m = /^[A-Za-z]+\d*/.exec(name.slice(i));
    i += m[0].length;
    if (name[i] !== "(") return { leaf: m[0] };
    i += 1;
    const args = [node()];
    while (name[i] === ",") { i += 1; args.push(node()); }
    i += 1;
    return { op: m[0], args };
  }
  return node();
}

// Binary operators with their precedence, as Python groups them.
const BINARY = { add: ["+", 1], sub: ["-", 1], mul: ["*", 2], div: ["/", 2], gt: [">", 0] };
const CALLS = { abs: "abs", min: "np.minimum", max: "np.maximum", where: "np.where" };

// The NumPy expression a tree stands for, parenthesised exactly where Python needs it
// to keep the tree's order: [text, precedence].
function infix(t) {
  if (t.leaf) return [t.leaf, 9];
  const parts = t.args.map(infix);
  const wrap = ([text, p], need) => (p < need ? `(${text})` : text);
  if (BINARY[t.op]) {
    const [sym, p] = BINARY[t.op];
    return [`${wrap(parts[0], p)} ${sym} ${wrap(parts[1], p + 1)}`, p];
  }
  if (t.op === "neg") return [`-${wrap(parts[0], 4)}`, 3];
  if (t.op === "sq") return [`${wrap(parts[0], 9)}**2`, 4];
  return [`${CALLS[t.op] || t.op}(${parts.map((q) => q[0]).join(", ")})`, 9];
}

export default async function mount(el) {
  let view = el.dataset.view === "shares" ? "shares" : "library";
  let shape = Math.min(Math.max(Number(el.dataset.shape) || 1, 1), SHAPES.length);
  let program = "";
  let picked = "cavity_flow";

  const stage = h("div", { class: "w-stage" });
  const detail = h("div", { class: "fs-detail", "aria-live": "polite" });
  const programs = [...new Set(SHAPES.flatMap((r) => r[4]))].sort();
  const select = h("select", { class: "w-select", "aria-label": "Light up the shapes of one program", onchange: () => { program = select.value; render(); } },
    h("option", { value: "" }, "every program"), ...programs.map((p) => h("option", { value: p }, p)));
  const programControl = h("span", { style: "display: inline-flex; gap: .5rem; align-items: center" }, h("span", { class: "w-label" }, "Shapes of"), select);

  function renderShares() {
    const names = Object.keys(SHARES).sort((a, b) => SHARES[b][4] - SHARES[a][4]);
    const rows = names.flatMap((name) => {
      const r = SHARES[name];
      const on = name === picked;
      const pick = () => { picked = name; render(); };
      const title = `${name}: ${CLASSES.map((c, i) => `${c.label} ${r[4 + i]}%`).join(", ")}`;
      return [
        h("button", { type: "button", class: `fs-name${on ? " on" : ""}`, onclick: pick, title }, name),
        h("div", { class: `fs-bar${on ? " on" : ""}`, onclick: pick, title, role: "img", "aria-label": title },
          ...CLASSES.map((c, i) => h("i", { style: `width: ${r[4 + i]}%; background: ${c.color}` }))),
        h("span", { class: "fs-ms" }, `${r[3]} ms`),
      ];
    });
    stage.replaceChildren(
      h("div", { class: "fs-rows" }, ...rows),
      h("div", { class: "fs-legend" }, ...CLASSES.map((c) => h("span", { title: c.note || "" }, h("i", { style: `background: ${c.color}` }), c.label))));
    const r = SHARES[picked];
    detail.replaceChildren(
      h("p", { class: "w-note" }, h("b", {}, picked), `: ${r[0].toLocaleString("en-US")} operations deferred in ${r[1]} graphs, the largest of ${r[2].toLocaleString("en-US")}; NumPy spent ${r[3]} ms computing its nodes.`),
      h("table", {}, ...CLASSES.map((c, i) => h("tr", {},
        h("td", {}, h("i", { style: `display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .4rem; background: ${c.color}` }), c.label, c.note ? h("span", { style: "color: var(--muted)" }, ` — ${c.note}`) : null),
        h("td", {}, `${r[4 + i]}%`)))));
  }

  function renderLibrary() {
    const xs = (rank) => M.l + ((rank > SHAPES.length ? LAST_X : rank) - 0.5) / LAST_X * (W - M.l - M.r);
    const ys = (pct) => H - M.b - (pct / 70) * (H - M.t - M.b);
    const colW = ((W - M.l - M.r) / LAST_X) * 0.8;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "The share of NumPy's time in expressions saved by the library, shape by shape" });
    const axis = s("g", { class: "axis" });
    for (const t of [0, 10, 20, 30, 40, 50, 60, 70]) {
      axis.append(s("line", { x1: M.l, x2: W - M.r, y1: ys(t), y2: ys(t), "stroke-dasharray": t ? "2 3" : "" }),
        s("text", { x: M.l - 6, y: ys(t) + 4, "text-anchor": "end", "font-size": 11 }, `${t}%`));
    }
    for (const rank of [1, 5, 10, 15, 20, 24]) axis.append(s("text", { x: xs(rank), y: H - M.b + 15, "text-anchor": "middle", "font-size": 11 }, rank));
    axis.append(s("text", { x: xs(60), y: H - M.b + 15, "text-anchor": "middle", "font-size": 11 }, 60));
    const gx = M.l + (GAP_AT / LAST_X) * (W - M.l - M.r);
    axis.append(s("text", { x: gx, y: H - M.b + 15, "text-anchor": "middle", "font-size": 11, class: "muted" }, "⋯"),
      s("text", { x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle", "font-size": 11, class: "muted" }, "shapes in the library, in the order they were chosen"));
    svg.append(axis);

    // A column for each shape: the bar is what it adds, the dot the total so far.
    let prev = 0;
    SHAPES.forEach(([rank, share, ops, , progs]) => {
      const add = share - prev;
      const has = !program || progs.includes(program);
      const col = s("g", { class: `fs-col${rank === shape ? " on" : ""}${has ? "" : " dim"}`, onclick: () => { shape = rank; render(); } },
        s("title", {}, `Shape ${rank}: +${add.toFixed(1)} points, ${share}% saved so far, ${ops} operations`),
        s("rect", { class: "hit", x: xs(rank) - colW / 2 - 1, y: M.t, width: colW + 2, height: H - M.t - M.b }),
        s("rect", { class: "bar", x: xs(rank) - colW / 2, y: ys(add), width: colW, height: ys(0) - ys(add), fill: OPS_COLOR[ops], rx: 2 }));
      svg.append(col);
      prev = share;
    });
    // Half of the time in expressions, saved from the twenty-second shape on.
    svg.append(s("line", { x1: M.l, x2: W - M.r, y1: ys(HALF.share), y2: ys(HALF.share), stroke: "var(--good)", "stroke-dasharray": "5 4", "stroke-width": 1.2 }),
      s("text", { x: M.l + 6, y: ys(HALF.share) - 6, "font-size": 11, fill: "var(--good)" }, `half, at shape ${HALF.rank}`));

    const path = SHAPES.map(([rank, share], i) => `${i ? "L" : "M"}${xs(rank).toFixed(1)},${ys(share).toFixed(1)}`).join(" ");
    svg.append(s("path", { d: `M${xs(0.5).toFixed(1)},${ys(0)} L${path.slice(1)}`, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6 }));
    for (const [rank, share] of SHAPES) svg.append(s("circle", { cx: xs(rank), cy: ys(share), r: rank === shape ? 4.5 : 2.5, fill: rank === shape ? "var(--hot)" : "var(--ink)" }));
    svg.append(s("circle", { cx: xs(60), cy: ys(SIXTIETH.share), r: 4, fill: "var(--bg-raised)", stroke: "var(--ink)", "stroke-width": 1.6 }),
      s("text", { x: xs(60) - 8, y: ys(SIXTIETH.share) - 9, "text-anchor": "end", "font-size": 11 }, `${SIXTIETH.share}% at shape ${SIXTIETH.rank}`));

    stage.replaceChildren(svg,
      h("div", { class: "fs-legend" },
        ...Object.entries(OPS_COLOR).map(([n, c]) => h("span", {}, h("i", { style: `background: ${c}` }), `${n} operations`)),
        h("span", {}, h("i", { style: "background: var(--ink); height: 2px" }), "saved so far")));

    const [rank, share, ops, name, progs] = SHAPES[shape - 1];
    const before = shape > 1 ? SHAPES[shape - 2][1] : 0;
    detail.replaceChildren(
      h("div", { class: "fs-shape" }, h("b", {}, "pattern"), name),
      h("div", { class: "fs-shape" }, h("b", {}, "NumPy"), infix(parse(name))[0]),
      h("p", { class: "w-note" }, `Shape ${rank}, ${ops} operations. With it the library saves ${share}% of NumPy's time in expressions, each operation counted as one pass${rank === 1 ? "" : `: ${(share - before).toFixed(1)} points more than the ${rank - 1} shape${rank === 2 ? "" : "s"} before it`}. V is an array as long as the result, S a number for the whole run.`),
      h("div", { class: "fs-progs" }, ...progs.map((p) => h("span", { class: p === program ? "on" : "" }, p))));
  }

  function render() {
    programControl.style.display = view === "library" ? "inline-flex" : "none";
    if (view === "shares") renderShares(); else renderLibrary();
  }

  el.append(
    h("div", { class: "w-controls" },
      seg([{ value: "shares", label: "Where NumPy's time goes" }, { value: "library", label: "The library, shape by shape" }], view, (v) => { view = v; render(); }),
      programControl),
    stage, detail);
  render();
}
