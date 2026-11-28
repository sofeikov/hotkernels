// The fourteen fused kernels written by hand, alone, by size.
//
// One line per kernel: how many times faster than NumPy's passes it computes
// its expression, at 4,096, 65,536, 1,048,576 and 16,777,216 elements,
// coloured by how many operations the expression has. Picking a kernel shows
// what it computes, its four figures, and how it compares with the compiled C
// reference over all of the judge's sizes.
//
// Data: the table in docs/lazy-fusion-design.md, "The kernels written by hand,
// alone", from `uv run judge run <kernel> --unlocked` on an Apple M1 Pro,
// NumPy 2.5.5. The NumPy side is the spec's oracle called from Python, the
// expression as a program writes it. What each kernel computes is the table in
// "Fourteen written by hand". The numbers are copied by a script that reads the
// document; nothing was timed for this figure.
//
// Each KERNELS row: name, operations, speedup at the four sizes, over naive C.
//
// Attributes:
//   data-kernel  the kernel picked first (default fuse_add_scaled_f64)

import { h, s, fmtX } from "./util.js";

const KERNELS = [["fuse_add_scaled_f64",2,4.37,1.86,1.66,1.54,1.05],["fuse_sub_scaled_f64",2,4.34,1.75,2.03,1.65,1.04],["fuse_add_product_f64",2,3.21,1.54,1.6,1.36,1.03],["fuse_sub_product_f64",2,3.24,1.53,1.7,1.43,1.03],["fuse_product3_f64",2,3.25,1.53,1.58,1.36,1.03],["fuse_scaled_sum_f64",2,4.04,1.89,1.56,1.56,1.04],["fuse_mul_shifted_f64",2,4.35,1.92,1.69,1.54,1.05],["fuse_mul_difference_f64",2,3.26,1.52,1.58,1.36,1.02],["fuse_sum3_f64",2,3.22,1.53,1.56,1.36,1.02],["fuse_sum_squares_f64",2,7.71,3.16,2.86,2.76,1.49],["fuse_flux_limit_f64",4,6.98,3.16,3,2.47,6.52],["fuse_laplacian5_f64",5,3.5,2.19,1.79,1.91,1],["fuse_flux_update_f64",5,2.7,1.81,1.71,1.74,0.99],["fuse_poisson_step_f64",8,5.05,3.13,2.72,2.72,1]];

// What each kernel computes and where the corpus has it.
const COMPUTES = {
  fuse_add_scaled_f64: ["a + s * b", "Eight programs"],
  fuse_sub_scaled_f64: ["a - s * b", "A gradient step: five programs"],
  fuse_add_product_f64: ["a + b * c", "Four programs"],
  fuse_sub_product_f64: ["a - b * c", "Four programs"],
  fuse_product3_f64: ["a * (b * c)", "The three Veros routines"],
  fuse_scaled_sum_f64: ["s * (a + b)", "Five programs"],
  fuse_mul_shifted_f64: ["a * (s + b)", "Horner's rule, GELU: three programs"],
  fuse_mul_difference_f64: ["a * (b - c)", "A coefficient times a finite difference: six programs"],
  fuse_sum3_f64: ["a + (b + c)", "Four programs"],
  fuse_sum_squares_f64: ["np.sum(x * x), and the mean", "An L2 penalty, a loss: four programs"],
  fuse_laplacian5_f64: ["s * a - (b + c + d + e)", "The horizontal diffusion"],
  fuse_flux_limit_f64: ["np.where(a * (b - c) > lo, fill, d)", "The horizontal diffusion, twice"],
  fuse_flux_update_f64: ["a - b * (c - d + e - f)", "The horizontal diffusion, its last line"],
  fuse_poisson_step_f64: ["((a + b) * s0 + (c + d) * s1) / s2 - s3 * e", "The cavity flow's pressure iteration"],
};

const SIZES = ["4,096", "65,536", "1,048,576", "16,777,216"];
const SIZE_NOTES = ["mostly calls", "", "", "fits no cache"];
// The colour of a kernel by its number of operations.
const OPS_COLOR = { 2: "var(--np)", 4: "var(--k-reduction)", 5: "var(--k-matmul)", 8: "var(--hot)" };

// The drawing, in viewBox units.
const W = 620, H = 300, M = { l: 44, r: 150, t: 18, b: 40 };
const Y = [1, 8];
const Y_TICKS = [1, 1.5, 2, 3, 4, 6, 8];

export const css = `
.fk-body { display: grid; grid-template-columns: minmax(0, 1fr); gap: .8rem; }
.fk-chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.fk-chip { font: inherit; font-family: var(--mono); font-size: .7rem; padding: .2rem .45rem; border-radius: 6px; border: 1px solid var(--rule-strong); background: var(--bg-raised); color: var(--ink-soft); cursor: pointer; display: inline-flex; align-items: center; gap: .35rem; }
.fk-chip i { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }
.fk-chip.on { border-color: var(--ink); color: var(--ink); font-weight: 700; }
.fk-chip:hover { border-color: var(--hot); }
.fk-expr { font-family: var(--mono); font-size: .88rem; background: var(--code-bg); padding: .45rem .7rem; border-radius: 8px; overflow-x: auto; white-space: nowrap; }
.fk-line { fill: none; stroke-width: 1.6; opacity: .35; transition: opacity .2s, stroke-width .2s; }
.fk-line.on { opacity: 1; stroke-width: 3; }
.fk-pt { opacity: 0; }
.fk-pt.on { opacity: 1; }
`;

const x = (i) => M.l + (i / (SIZES.length - 1)) * (W - M.l - M.r);
const y = (v) => H - M.b - ((Math.log(v) - Math.log(Y[0])) / (Math.log(Y[1]) - Math.log(Y[0]))) * (H - M.t - M.b);

export default async function mount(el) {
  let picked = COMPUTES[el.dataset.kernel] ? el.dataset.kernel : "fuse_add_scaled_f64";

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Speedup of each hand-written fused kernel over NumPy's passes, by number of elements" });
  const chips = h("div", { class: "fk-chips" });
  const info = h("div", {});
  const lines = new Map();
  const points = new Map();

  // Axes and grid.
  const axis = s("g", { class: "axis" });
  for (const t of Y_TICKS) {
    axis.append(s("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), "stroke-dasharray": t === 1 ? "" : "2 3" }),
      s("text", { x: M.l - 6, y: y(t) + 4, "text-anchor": "end", "font-size": 11 }, `${t}x`));
  }
  SIZES.forEach((label, i) => {
    axis.append(s("text", { x: x(i), y: H - M.b + 16, "text-anchor": "middle", "font-size": 11 }, label));
    if (SIZE_NOTES[i]) axis.append(s("text", { x: x(i), y: H - M.b + 30, "text-anchor": "middle", "font-size": 10, class: "muted" }, SIZE_NOTES[i]));
  });
  axis.append(s("text", { x: W - M.r + 8, y: y(1) + 4, "font-size": 10, class: "muted" }, "NumPy's passes"));
  svg.append(axis);

  // A line and its points for each kernel; a label at the right for the picked one.
  const label = s("text", { "font-size": 11, "font-weight": 700 });
  for (const [name, ops, ...v] of KERNELS) {
    const color = OPS_COLOR[ops];
    const d = v.slice(0, 4).map((val, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(val).toFixed(1)}`).join(" ");
    const line = s("path", { d, class: "fk-line", stroke: color, onclick: () => pick(name) },
      s("title", {}, `${name}: ${v.slice(0, 4).map(fmtX).join(", ")}`));
    const pts = s("g", { class: "fk-pt" }, ...v.slice(0, 4).map((val, i) => [
      s("circle", { cx: x(i), cy: y(val), r: 4, fill: color }),
      s("text", { x: x(i), y: y(val) - 9, "text-anchor": "middle", "font-size": 11, "font-weight": 650 }, fmtX(val)),
    ]));
    lines.set(name, line);
    points.set(name, pts);
    svg.append(line, pts);
  }
  svg.append(label);

  for (const [name, ops] of KERNELS) {
    chips.append(h("button", { type: "button", class: "fk-chip", "data-name": name, onclick: () => pick(name) },
      h("i", { style: `background: ${OPS_COLOR[ops]}` }), name.replace(/^fuse_|_f64$/g, "")));
  }

  function pick(name) {
    picked = name;
    const [, ops, ...v] = KERNELS.find((k) => k[0] === name);
    for (const [n, line] of lines) {
      line.classList.toggle("on", n === name);
      points.get(n).classList.toggle("on", n === name);
    }
    // Bring the picked line to the front.
    svg.insertBefore(lines.get(name), label);
    svg.insertBefore(points.get(name), label);
    label.setAttribute("x", W - M.r + 8);
    label.setAttribute("y", y(v[3]) + 4);
    label.setAttribute("fill", OPS_COLOR[ops]);
    label.textContent = `${ops} operations`;
    for (const chip of chips.children) chip.classList.toggle("on", chip.dataset.name === name);
    const [expr, where] = COMPUTES[name];
    info.replaceChildren(
      h("div", { class: "fk-expr" }, expr),
      h("p", { class: "w-note" },
        h("b", {}, name), ` — ${ops} operations, in the corpus: ${where}. `,
        `${fmtX(v[0])} at 4,096 elements, ${fmtX(v[3])} at 16,777,216. Against the compiled C reference over all of the judge's sizes: ${fmtX(v[4])}.`));
  }

  el.append(
    h("div", { class: "fk-body" },
      h("div", { class: "w-stage" }, svg),
      h("div", { class: "w-legend" },
        h("span", {}, h("i", { style: `background: ${OPS_COLOR[2]}` }), "2 operations"),
        h("span", {}, h("i", { style: `background: ${OPS_COLOR[4]}` }), "4"),
        h("span", {}, h("i", { style: `background: ${OPS_COLOR[5]}` }), "5"),
        h("span", {}, h("i", { style: `background: ${OPS_COLOR[8]}` }), "8")),
      chips,
      info));
  pick(picked);
}
