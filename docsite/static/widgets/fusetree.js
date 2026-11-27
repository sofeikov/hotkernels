// What the planner sees in an expression: its tree, what is inside it and what
// is an operand, the name of its pattern, and the kernel calls that cover it.
//
// The reader picks one of twelve expressions, most of them lines of the
// corpus's programs (the Veros equation of state, the horizontal diffusion of
// npbench's hdiff, the pressure iteration and the velocity update of npbench's
// cavity flow) and the rest the small cases of docs/lazy-fusion-design.md,
// "What is fused". The tree is drawn with its root on the left. Operations
// inside the expression are circles; an array it reads is a V leaf, a number
// an S leaf, and an operation outside it, which NumPy computes first, a dashed
// leaf. "Kernel calls" colours each operation by the kernel call that makes it.
//
// Everything below is the prototype's own answer: each expression was run on
// lazy arrays of prototypes/lazy (NumPy 2.5.5, CPython 3.12, with the
// layer in Python: the C core, whose reader a test holds to the same
// expressions, was not in use), and a planner hooked into evaluate.planner
// recorded what
// fuse.find read off the graph, fuse.canonical's name for it, and fuse.tile's
// cover with the library of the eighty kernels in prototypes/lazy/kernels/,
// whose shapes were read off their oracles by fuse.load_library. The module
// _fused was not built for this: the library's table of names was given to
// load_library directly, which is all planning reads of it. Only the labels
// of the numbers are the source's spellings (dt / dx**2, v15) where Python
// had already folded them into one float. Arrays were small; the names and
// covers do not depend on the size.
//
// Attributes:
//   data-example  the expression shown first (default scaled): one of the keys of EXAMPLES
//   data-view     inside (default) or kernels

import { h, s, seg, stat } from "./util.js";

// Per expression the planner found: the pattern's name, the tree as the
// program wrote it, and the tiles. In the tree, an operation is
// {op, tile, c: children}; a leaf {k, t} with k = V (an array), S (a number)
// or O (an operation outside the expression, computed first by NumPy). tile is
// an index into tiles: [kernel, the shape it computes].
const FOUND = {
  scaled: [{"name":"add(V0,mul(S0,V1))","tree":{"op":"add","tile":0,"c":[{"k":"V","t":"a"},{"op":"mul","tile":0,"c":[{"k":"S","t":"s"},{"k":"V","t":"b"}]}]},"tiles":[["fuse_add_scaled_f64","add(V0,mul(S0,V1))"]]}],
  same: [{"name":"add(V0,mul(S0,V1))","tree":{"op":"add","tile":0,"c":[{"k":"V","t":"a"},{"op":"mul","tile":0,"c":[{"k":"S","t":"s"},{"k":"V","t":"a"}]}]},"tiles":[["fuse_add_scaled_f64","add(V0,mul(S0,V1))"]]}],
  turned: [{"name":"add(V0,mul(V1,V2))","tree":{"op":"add","tile":0,"c":[{"op":"mul","tile":0,"c":[{"k":"V","t":"b"},{"k":"V","t":"a"}]},{"k":"V","t":"c"}]},"tiles":[["fuse_add_product_f64","add(V0,mul(V1,V2))"]]}],
  gradient: [{"name":"sub(V0,mul(S0,V1))","tree":{"op":"sub","tile":0,"c":[{"k":"V","t":"w"},{"op":"mul","tile":0,"c":[{"k":"S","t":"rate"},{"k":"V","t":"gradient"}]}]},"tiles":[["fuse_sub_scaled_f64","sub(V0,mul(S0,V1))"]]}],
  laplacian: [{"name":"sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))","tree":{"op":"sub","tile":0,"c":[{"op":"mul","tile":0,"c":[{"k":"S","t":"4.0"},{"k":"V","t":"u[1:-1, 1:-1, :]"}]},{"op":"add","tile":0,"c":[{"op":"add","tile":0,"c":[{"op":"add","tile":0,"c":[{"k":"V","t":"u[2:, 1:-1, :]"},{"k":"V","t":"u[:-2, 1:-1, :]"}]},{"k":"V","t":"u[1:-1, 2:, :]"}]},{"k":"V","t":"u[1:-1, :-2, :]"}]}]},"tiles":[["fuse_laplacian5_f64","sub(mul(S0,V0),add(V1,add(V2,add(V3,V4))))"]]}],
  poisson: [{"name":"sub(div(add(mul(S0,add(V0,V1)),mul(S1,add(V2,V3))),S2),mul(S3,V4))","tree":{"op":"sub","tile":0,"c":[{"op":"div","tile":0,"c":[{"op":"add","tile":0,"c":[{"op":"mul","tile":0,"c":[{"op":"add","tile":0,"c":[{"k":"V","t":"pn[1:-1, 2:]"},{"k":"V","t":"pn[1:-1, 0:-2]"}]},{"k":"S","t":"dy**2"}]},{"op":"mul","tile":0,"c":[{"op":"add","tile":0,"c":[{"k":"V","t":"pn[2:, 1:-1]"},{"k":"V","t":"pn[0:-2, 1:-1]"}]},{"k":"S","t":"dx**2"}]}]},{"k":"S","t":"2 * (dx**2 + dy**2)"}]},{"op":"mul","tile":0,"c":[{"k":"S","t":"dx**2 * dy**2 / (2 * (dx**2 + dy**2))"},{"k":"V","t":"b[1:-1, 1:-1]"}]}]},"tiles":[["fuse_poisson_step_f64","sub(div(add(mul(S0,add(V0,V1)),mul(S1,add(V2,V3))),S2),mul(S3,V4))"]]}],
  eos: [{"name":"mul(V0,add(S0,mul(S1,V1)))","tree":{"op":"mul","tile":0,"c":[{"k":"V","t":"sa"},{"op":"add","tile":0,"c":[{"k":"S","t":"v15"},{"op":"mul","tile":0,"c":[{"k":"S","t":"v16"},{"k":"V","t":"ct"}]}]}]},"tiles":[["fuse_mul_v_add_s_mul_s_v_f64","mul(V0,add(S0,mul(S1,V1)))"]]}],
  exp: [{"name":"add(V0,mul(S0,V1))","tree":{"op":"add","tile":0,"c":[{"op":"mul","tile":0,"c":[{"k":"O","t":"np.exp(a * b + c)"},{"k":"S","t":"2.0"}]},{"k":"V","t":"a"}]},"tiles":[["fuse_add_scaled_f64","add(V0,mul(S0,V1))"]]},{"name":"add(V0,mul(V1,V2))","tree":{"op":"add","tile":0,"c":[{"op":"mul","tile":0,"c":[{"k":"V","t":"a"},{"k":"V","t":"b"}]},{"k":"V","t":"c"}]},"tiles":[["fuse_add_product_f64","add(V0,mul(V1,V2))"]]}],
  twice: [{"name":"add(V0,mul(V1,V2))","tree":{"op":"add","tile":0,"c":[{"k":"O","t":"t = a * b"},{"op":"mul","tile":0,"c":[{"k":"O","t":"t = a * b"},{"k":"V","t":"c"}]}]},"tiles":[["fuse_add_product_f64","add(V0,mul(V1,V2))"]]}],
  sumsq: [{"name":"sum(sq(V0))","tree":{"op":"sum","tile":0,"c":[{"op":"sq","tile":0,"c":[{"k":"V","t":"a"}]}]},"tiles":[["fuse_sum_squares_f64","sum(sq(V0))"]]}],
  cavity: [{"name":"add(mul(S0,add(mul(S1,add(V0,sub(V1,mul(S2,V2)))),mul(S3,add(V3,sub(V4,mul(S4,V5)))))),sub(sub(sub(V6,mul(div(mul(S5,V7),S6),sub(V8,V9))),mul(div(mul(S7,V10),S8),sub(V11,V12))),mul(S9,sub(V13,V14))))","tree":{"op":"add","tile":0,"c":[{"op":"sub","tile":0,"c":[{"op":"sub","tile":1,"c":[{"op":"sub","tile":2,"c":[{"k":"V","t":"un[1:-1, 1:-1]"},{"op":"mul","tile":2,"c":[{"op":"div","tile":2,"c":[{"op":"mul","tile":2,"c":[{"k":"V","t":"un[1:-1, 1:-1]"},{"k":"S","t":"dt"}]},{"k":"S","t":"dx"}]},{"op":"sub","tile":2,"c":[{"k":"V","t":"un[1:-1, 1:-1]"},{"k":"V","t":"un[1:-1, 0:-2]"}]}]}]},{"op":"mul","tile":1,"c":[{"op":"div","tile":1,"c":[{"op":"mul","tile":1,"c":[{"k":"V","t":"vn[1:-1, 1:-1]"},{"k":"S","t":"dt"}]},{"k":"S","t":"dy"}]},{"op":"sub","tile":1,"c":[{"k":"V","t":"un[1:-1, 1:-1]"},{"k":"V","t":"un[0:-2, 1:-1]"}]}]}]},{"op":"mul","tile":0,"c":[{"k":"S","t":"dt / (2 * rho * dx)"},{"op":"sub","tile":0,"c":[{"k":"V","t":"p[1:-1, 2:]"},{"k":"V","t":"p[1:-1, 0:-2]"}]}]}]},{"op":"mul","tile":0,"c":[{"k":"S","t":"nu"},{"op":"add","tile":0,"c":[{"op":"mul","tile":3,"c":[{"k":"S","t":"dt / dx**2"},{"op":"add","tile":3,"c":[{"op":"sub","tile":3,"c":[{"k":"V","t":"un[1:-1, 2:]"},{"op":"mul","tile":3,"c":[{"k":"S","t":"2"},{"k":"V","t":"un[1:-1, 1:-1]"}]}]},{"k":"V","t":"un[1:-1, 0:-2]"}]}]},{"op":"mul","tile":4,"c":[{"k":"S","t":"dt / dy**2"},{"op":"add","tile":4,"c":[{"op":"sub","tile":4,"c":[{"k":"V","t":"un[2:, 1:-1]"},{"op":"mul","tile":4,"c":[{"k":"S","t":"2"},{"k":"V","t":"un[1:-1, 1:-1]"}]}]},{"k":"V","t":"un[0:-2, 1:-1]"}]}]}]}]}]},"tiles":[["fuse_add_mul_s_add_v_v_sub_v_mul_s_sub_v_v_f64","add(mul(S0,add(V0,V1)),sub(V2,mul(S1,sub(V3,V4))))"],["fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64","sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))"],["fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64","sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))"],["fuse_mul_s_add_v_sub_v_mul_s_v_f64","mul(S0,add(V0,sub(V1,mul(S1,V2))))"],["fuse_mul_s_add_v_sub_v_mul_s_v_f64","mul(S0,add(V0,sub(V1,mul(S1,V2))))"]]}],
  shape: [{"name":"sub(div(sqrt(add(V0,sq(V1))),S0),V2)","tree":{"op":"sub","tile":0,"c":[{"op":"div","tile":1,"c":[{"op":"sqrt","tile":2,"c":[{"op":"add","tile":3,"c":[{"op":"sq","tile":4,"c":[{"k":"V","t":"w"}]},{"k":"O","t":"g * g"}]}]},{"k":"S","t":"2.0"}]},{"k":"V","t":"w"}]},"tiles":[["fuse_sub_v_v_f64","sub(V0,V1)"],["fuse_div_v_s_f64","div(V0,S0)"],["fuse_sqrt_v_f64","sqrt(V0)"],["fuse_add_v_v_f64","add(V0,V1)"],["fuse_sq_v_f64","sq(V0)"]]}],
};

// The expressions offered, in the order a reader meets the ideas.
const EXAMPLES = [
  {
    key: "scaled", label: "a + s * b", code: "y = a + s * b",
    note: "Two operations and one intermediate, <code>s * b</code>, that only the addition reads. <code>a</code> and <code>b</code> are arrays as long as the result, the <b>V</b> leaves; <code>s</code> is one number for every element, an <b>S</b> leaf. NumPy makes two passes and one temporary; <code>fuse_add_scaled_f64</code>, written by hand, makes one pass. Eight programs of the corpus have this shape.",
  },
  {
    key: "same", label: "a + s * a", code: "y = a + s * a",
    note: "<code>a</code> is read in two places. Each place an array is read is a leaf of its own, so the name is the name of <code>a + s * b</code>, and the same kernel is called with one pointer given twice. A pattern's name does not depend on which of its arrays happen to be the same one.",
  },
  {
    key: "turned", label: "b * a + c", code: "y = b * a + c",
    note: "Written with the product first, named with it second: the two operands of a sum, a product, a maximum or a minimum are put in a fixed order, so <code>c + a * b</code> and <code>b * a + c</code> have one name and one kernel. Only the two sides of one operation are swapped. Which additions are made first is never changed.",
  },
  {
    key: "gradient", label: "w -= rate * gradient", code: "w -= rate * gradient      # w an array the layer owns",
    note: "A write is an operation of the expression. <code>w -= rate * gradient</code> is <code>sub(V0,mul(S0,V1))</code>, one kernel call, with no temporary for <code>rate * gradient</code>. Five programs of the corpus take a gradient step this way.",
  },
  {
    key: "eos", label: "Ocean model: one line of the equation of state",
    code: "t11 = sa * (v15 + v16 * ct)      # Veros, gsw_dHdT; v15 and v16 are Python numbers",
    note: "One line of the Veros equation of state. The coefficients are Python numbers, so they are S leaves, and the line is a shape of three operations that <code>make_kernel.py</code> wrote a kernel for. Two later lines of the routine read <code>t11</code>, so its value is wanted twice: it is an expression of its own, and an operand of both. Elsewhere the routine has an expression of 153 operations.",
  },
  {
    key: "laplacian", label: "Weather model: the five-point Laplacian",
    code: "lap = 4.0 * u[1:-1, 1:-1, :] - (u[2:, 1:-1, :] + u[:-2, 1:-1, :]\n                               + u[1:-1, 2:, :] + u[1:-1, :-2, :])",
    note: "The first line of npbench's <code>hdiff</code>, with its <code>in_field</code> written <code>u</code> and its bounds as they fall for a field two wider than the result on each side. Five slices of one grid, each with the result's shape, and four intermediates. The program adds left to right, and the name keeps that grouping: in <code>add(V1,add(V2,add(V3,V4)))</code> the innermost addition is of the two slices written first. NumPy makes five passes; <code>fuse_laplacian5_f64</code> makes one.",
  },
  {
    key: "poisson", label: "Cavity flow: the pressure iteration",
    code: "p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +\n                  (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /\n                 (2 * (dx**2 + dy**2)) - dx**2 * dy**2 /\n                 (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])",
    note: "Eight operations, the largest kernel written by hand. <code>dx</code> and <code>dy</code> are Python numbers, so <code>dy**2</code>, <code>2 * (dx**2 + dy**2)</code> and the long coefficient of <code>b</code> are numbers before the layer sees anything: the graph is the program after Python has run, and the layer never sees how a number was written.",
  },
  {
    key: "exp", label: "np.exp ends an expression", code: "y = np.exp(a * b + c) * 2.0 + a",
    note: "<code>np.exp</code> calls the platform's mathematics library, which a kernel cannot do and could not match bit for bit. So it ends the expression below it and starts one above it: <code>a * b + c</code> is computed first, by one kernel, NumPy computes the exponential, and <code>(…) * 2.0 + a</code> reads it as an array.",
  },
  {
    key: "twice", label: "An intermediate read twice", code: "t = a * b\ny = t + t * c",
    note: "<code>t</code> is read by the addition and by the product. Its value is wanted twice, so it has to exist: computing past it would compute it twice. NumPy's own call makes it, and it is an operand of the expression that reads it. The same holds for an intermediate the program still has a name for when the computation runs.",
  },
  {
    key: "shape", label: "Another shape, and the long tail", code: "y = np.sqrt(w * w + g * g) / 2.0 - w     # w a large grid, g as long as one of its rows",
    note: "<code>g * g</code> has the shape of the row <code>g</code>, not the result's: it belongs to another loop and is computed first. <code>w * w</code> is named <code>sq</code>, since <code>x * x</code>, <code>x ** 2</code> and <code>np.square(x)</code> are one multiplication. No shape of the library has this tree, so it is covered with five kernels of one operation each. Switch to <b>Kernel calls</b>: they still run block by block, one pass over memory.",
  },
  {
    key: "cavity", label: "Cavity flow: the update of u",
    code: "u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])\n    - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])\n    - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])\n    + nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])\n          + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))",
    note: "Twenty-four operations on fifteen slices of three grids: NumPy makes 24 passes, each into a temporary, and then copies the last into <code>u</code>. No kernel has this whole tree. Switch to <b>Kernel calls</b> to see how the planner covers it with five shapes of the library, by the fewest passes. The next figure runs those five calls.",
  },
  {
    key: "sumsq", label: "A sum at the top", code: "loss = (a ** 2).sum()",
    note: "A sum or a mean of every element may stand at the top of an expression. <code>fuse_sum_squares_f64</code> squares each element as it meets it and adds in NumPy's own pairwise order, so there is no array of squares and the sum has NumPy's bits.",
  },
];

// How an operation is drawn inside its circle.
const SYMBOL = { add: "+", sub: "−", mul: "×", div: "÷", max: "max", min: "min", neg: "−x", abs: "|x|", sqrt: "√", sq: "x²", sum: "Σ", mean: "μ" };
// One colour per kernel call, in the order the tiles are met from the top.
const TILE_COLOURS = ["--g1", "--g2", "--g3", "--k-matmul", "--k-reduction", "--k-view"];
// Spacing of the drawing, in SVG units: a column per depth, a row per leaf.
const COLUMN = 56, ROW = 30, RADIUS = 12, PAD = 14;
// The width of a character of the leaves' monospaced labels, and a leaf's padding.
const CHAR = 7.1, LEAF_PAD = 14;

export const css = `
.ft-code { font-family: var(--mono); font-size: .8rem; background: var(--code-bg); padding: .5rem .75rem; border-radius: 8px; overflow-x: auto; white-space: pre; margin: 0 0 .8rem; line-height: 1.55; }
.ft-stage { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--bg); padding: .4rem 0; }
.ft-stage svg { display: block; max-width: none; height: auto; }
.ft-stage text { font-size: 12px; }
.ft-stage .ft-op text { font-weight: 700; font-size: 12px; }
.ft-stage .ft-leaf text { font-family: var(--mono); font-size: 11.5px; }
.ft-names { margin-top: .8rem; display: grid; gap: .45rem; }
.ft-name { font-family: var(--mono); font-size: .8rem; word-break: break-all; line-height: 1.5; }
.ft-name .v { color: var(--k-elementwise); font-weight: 700; }
.ft-name .s { color: var(--k-reduction); font-weight: 700; }
.ft-name-l { font-family: var(--sans); font-size: .72rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-right: .5rem; }
.ft-tiles { list-style: none; margin: .7rem 0 0; padding: 0; display: grid; gap: .3rem; font-size: .78rem; }
.ft-tiles li { display: flex; gap: .55rem; align-items: baseline; flex-wrap: wrap; }
.ft-tiles i { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; flex: none; align-self: center; }
.ft-tiles code { font-family: var(--mono); font-size: .74rem; border: 0; background: none; padding: 0; word-break: break-all; }
.ft-tiles .shape { color: var(--muted); }
.ft-key { display: flex; flex-wrap: wrap; gap: .35rem 1rem; font-size: .76rem; color: var(--muted); margin-top: .6rem; }
.ft-key span { display: inline-flex; align-items: center; gap: .35rem; }
.ft-key i { display: inline-block; width: .85rem; height: .85rem; border: 1.5px solid; flex: none; }
`;

// Lays a tree out with its root on the left: leaves one to a row, in the
// order they are met, and every operation in the middle of its children.
function layout(tree) {
  const nodes = [], edges = [];
  let row = 0, deepest = 0;
  function place(part, depth) {
    deepest = Math.max(deepest, depth);
    if (!part.op) {
      const item = { part, depth, y: row++ };
      nodes.push(item);
      return item;
    }
    const children = part.c.map((child) => place(child, depth + 1));
    const item = { part, depth, y: (children[0].y + children[children.length - 1].y) / 2 };
    nodes.push(item);
    for (const child of children) edges.push([item, child]);
    return item;
  }
  place(tree, 0);
  return { nodes, edges, rows: row, deepest };
}

// The pattern's name with its V and S leaves picked out.
function nameMarkup(name) {
  return name.replace(/([VS])(\d+)/g, (_, kind, n) => `<span class="${kind.toLowerCase()}">${kind}${n}</span>`);
}

// Draws one expression's tree.
function drawTree(found, view, base = 0) {
  const { nodes, edges, rows, deepest } = layout(found.tree);
  const x = (depth) => PAD + RADIUS + depth * COLUMN;
  const y = (r) => PAD + r * ROW + ROW / 2;
  const longest = Math.max(...nodes.filter((n) => !n.part.op).map((n) => n.part.t.length));
  const width = x(deepest) + longest * CHAR + LEAF_PAD * 2 + PAD;
  const height = rows * ROW + PAD * 2;
  const tone = (part) => view === "kernels" ? `var(${TILE_COLOURS[(base + part.tile) % TILE_COLOURS.length]})` : "var(--k-elementwise)";
  const soft = (part) => view === "kernels" ? `color-mix(in srgb, var(${TILE_COLOURS[(base + part.tile) % TILE_COLOURS.length]}) 18%, var(--bg-raised))` : "var(--k-elementwise-soft)";

  const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `The tree of ${found.name}` });
  for (const [parent, child] of edges) {
    const midX = x(parent.depth) + COLUMN / 2;
    svg.append(s("path", {
      d: `M${x(parent.depth)},${y(parent.y)} H${midX} V${y(child.y)} H${x(child.depth)}`,
      fill: "none", stroke: "var(--rule-strong)", "stroke-width": 1.4,
    }));
  }
  for (const item of nodes) {
    const { part } = item;
    const cx = x(item.depth), cy = y(item.y);
    if (part.op) {
      const label = SYMBOL[part.op] || part.op;
      svg.append(s("g", { class: "ft-op" },
        s("title", {}, view === "kernels" ? `${part.op}: made by ${found.tiles[part.tile][0]}` : `${part.op}: inside the expression`),
        s("circle", { cx, cy, r: RADIUS, fill: soft(part), stroke: tone(part), "stroke-width": 1.8 }),
        s("text", { x: cx, y: cy + 4, "text-anchor": "middle", style: `fill: ${tone(part)}; font-size: ${label.length > 2 ? 9 : 12}px` }, label)));
    } else {
      const w = part.t.length * CHAR + LEAF_PAD;
      const kind = part.k;
      const stroke = kind === "V" ? "var(--k-input)" : kind === "S" ? "var(--rule-strong)" : "var(--np)";
      const fill = kind === "V" ? "var(--k-input-soft)" : kind === "S" ? "var(--panel)" : "var(--np-soft)";
      const what = kind === "V" ? "an array the expression reads (V)" : kind === "S" ? "a number (S)" : "an operation outside the expression: NumPy computes it first, and the expression reads it as an array";
      svg.append(s("g", { class: "ft-leaf" },
        s("title", {}, `${part.t}: ${what}`),
        s("rect", { x: cx - RADIUS + 2, y: cy - 10, width: w, height: 20, rx: kind === "S" ? 10 : 4, fill, stroke, "stroke-width": 1.4, "stroke-dasharray": kind === "O" ? "4 3" : null }),
        s("text", { x: cx - RADIUS + 2 + LEAF_PAD / 2, y: cy + 4, style: kind === "S" ? "fill: var(--muted)" : kind === "O" ? "fill: var(--np)" : "" }, part.t)));
    }
  }
  return svg;
}

// Counts what an expression's tree holds: operations, and leaves of each kind.
function count(part, totals = { ops: 0, V: 0, S: 0, O: 0 }) {
  if (part.op) {
    totals.ops += 1;
    for (const child of part.c) count(child, totals);
  } else {
    totals[part.k] += 1;
  }
  return totals;
}

export default async function mount(el) {
  let key = EXAMPLES.some((e) => e.key === el.dataset.example) ? el.dataset.example : "scaled";
  let view = el.dataset.view === "kernels" ? "kernels" : "inside";

  const select = h("select", { class: "w-select", "aria-label": "Expression", onchange: () => { key = select.value; render(); } },
    ...EXAMPLES.map((e) => h("option", { value: e.key }, e.label)));
  select.value = key;
  const code = h("pre", { class: "ft-code" });
  const stage = h("div", { class: "ft-stage" });
  const names = h("div", { class: "ft-names" });
  const tiles = h("ul", { class: "ft-tiles" });
  const key_ = h("div", { class: "ft-key" });
  const note = h("p", { class: "w-note", "aria-live": "polite" });
  const operations = stat("operations fused");
  const passes = stat("passes NumPy makes for them", "np");
  const calls = stat("kernel calls, one pass", "hot");
  const operands = stat("leaves: arrays and numbers read");

  function render() {
    const example = EXAMPLES.find((e) => e.key === key);
    const found = FOUND[key];
    code.textContent = example.code;
    stage.replaceChildren(...found.map((f, i) => h("div", {},
      found.length > 1 ? h("p", { class: "w-label", style: "margin: .3rem .8rem 0" }, i === 0 ? "The expression above np.exp" : "The expression below it") : null,
      drawTree(f, view, found.slice(0, i).reduce((sum, g) => sum + g.tiles.length, 0)))));
    names.replaceChildren(...found.map((f) => h("div", { class: "ft-name" }, h("span", { class: "ft-name-l" }, "Pattern"), h("span", { html: nameMarkup(f.name) }))));
    if (view === "kernels") {
      tiles.replaceChildren(...found.flatMap((f) => f.tiles).map(([kernel, shape], i) => h("li", {},
        h("i", { style: `background: var(${TILE_COLOURS[i % TILE_COLOURS.length]})` }),
        h("code", {}, kernel), h("code", { class: "shape" }, shape))));
      tiles.hidden = false;
    } else {
      tiles.hidden = true;
    }
    key_.replaceChildren(...[
      view === "inside" ? h("span", {}, h("i", { style: "border-radius: 50%; background: var(--k-elementwise-soft); border-color: var(--k-elementwise)" }), "an operation inside the expression") : null,
      h("span", {}, h("i", { style: "border-radius: 3px; background: var(--k-input-soft); border-color: var(--k-input)" }), "V: an array as long as the result"),
      h("span", {}, h("i", { style: "border-radius: 50%; background: var(--panel); border-color: var(--rule-strong)" }), "S: one number"),
      found.some((f) => count(f.tree).O) ? h("span", {}, h("i", { style: "border-radius: 3px; background: var(--np-soft); border-color: var(--np); border-style: dashed" }), "computed first, by NumPy") : null].filter(Boolean));
    note.innerHTML = example.note;
    const totals = found.map((f) => count(f.tree));
    const ops = totals.reduce((sum, t) => sum + t.ops, 0);
    operations.set(String(ops));
    passes.set(String(ops));
    calls.set(String(found.reduce((sum, f) => sum + f.tiles.length, 0)));
    operands.set(String(totals.reduce((sum, t) => sum + t.V + t.S + t.O, 0)));
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Expression"), select,
      seg([{ value: "inside", label: "Inside and outside" }, { value: "kernels", label: "Kernel calls" }], view, (v) => { view = v; render(); })),
    code, stage, key_, names, tiles, note,
    h("div", { class: "w-readout" }, operations, passes, calls, operands),
  );
  render();
}
