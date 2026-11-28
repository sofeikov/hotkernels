// How a program of kernel calls is run on arrays of any shape.
//
// A kernel is a leaf function over contiguous runs of float64. _fused.c stands
// between it and arrays of any layout. The reader picks a layout and sees the
// memory it occupies, a row of squares for each row of the array, coloured by
// the run each element is computed in: what is read where it lies, what is
// copied into a buffer first, what is computed and thrown away, and what is
// left to NumPy.
//
// The drawings are small schematic grids, not the arrays' real sizes, which
// the text names. The rules, the examples and every figure quoted come from
// docs/lazy-fusion-design.md, "Computing a group" and "An expression block by
// block", measured on an Apple M1 Pro with NumPy 2.5.5.
//
// Attributes:
//   data-layout  the layout shown first (default contiguous): a key of LAYOUTS

import { h } from "./util.js";

// Colours of the runs a drawing shows, in turn.
const RUNS = ["--g1", "--g2", "--g3", "--k-matmul"];

// A panel's cell function returns one of these for each element of memory:
// a run number, or one of the letters below.
const KIND = {
  out: "not part of the array the expression reads",
  w: "computed and thrown away: an element between the rows",
  b: "copied into a buffer first, by a loop of its own",
  c: "the same values for every row, held in a buffer",
  x: "left to NumPy",
  n: "one number for each row, which the row's arithmetic takes as its number",
};

// row r of `rows`, column c of `pitch`: each panel says what lies there.
const LAYOUTS = {
  contiguous: {
    label: "A C-contiguous array",
    code: "y = a * (b - c)            # a, b, c of shape (204, 204, 26), all C-contiguous",
    panels: [{ title: "a, b, c and y: every element, in memory order", rows: 4, pitch: 13, cell: () => 0 }],
    how: "Every outer axis that every operand walks straight through joins the run, so a C-contiguous expression of any number of dimensions is <b>one run</b>, and the program walks it a block of 4,096 elements at a time.",
    when: "Whenever the expression is fused at all: no layout is simpler.",
  },
  slice: {
    label: "A slice of a grid",
    code: "lap = 4.0 * u[1:-1, 1:-1] - (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])",
    panels: [{ title: "u[1:-1, 1:-1] inside u", rows: 6, pitch: 13, cell: (r, c) => (r > 0 && r < 5 && c > 0 && c < 12 ? (r - 1) % RUNS.length : "out") }],
    how: "A row of the slice does not end where the next begins, so <b>each row is a run</b>, and the program runs once for each, block by block. The other slices read are rows of the same length at the same pitch.",
    when: "Rows of 32 elements or more, and rows of 16 to 31 with more than a quarter of the walk between them. Shorter rows have rules of their own, further down this list.",
  },
  broadcast: {
    label: "An operand the same for every row",
    code: "y = (grid - mean) * dzt[None, None, :]      # grid of 204 × 204 × 26, mean a number, dzt the 26 depths",
    panels: [
      { title: "grid and the result: one run", rows: 4, pitch: 13, cell: () => 0 },
      { title: "dzt[None, None, :]: one row of values, the same for every row", rows: 1, pitch: 13, cell: () => "c" },
      { title: "its buffer: that row over and over, the length of a block", rows: 4, pitch: 13, cell: () => "c" },
    ],
    how: "<code>dzt[None, None, :]</code> beside a grid of 204 × 204 × 26 would make every run 26 elements long. Instead the run is lengthened all the same, and that operand is read from a buffer that holds its 26 values over and over for the length of a block.",
    when: "An operand that is the same along an outer axis; its innermost axis is as long as the result's.",
  },
  gathered: {
    label: "A column of a table",
    code: "y = (particles[:, 0] - x0) * scale + offset     # particles of shape (n, 4); offset contiguous",
    panels: [
      { title: "particles: the column read is every fourth element", rows: 6, pitch: 4, cell: (r, c) => (c === 0 ? "b" : "out") },
      { title: "copied into a buffer of its own, a block at a time", rows: 1, pitch: 6, cell: () => "b" },
    ],
    how: "A column, <code>particles[:, 0]</code>, or a grid at one time level, <code>tke[:, :, :, tau]</code>, steps over elements, and a kernel reads only runs. The shape is walked in the order of its rows a block at a time: an array that lies in that order is read where it is, and any other is <b>copied into a buffer</b> by a loop written for it. An array read in two places is copied once, and the result's block is copied out if the result steps too.",
    when: "Only when the program saves more passes than it makes copies. A copy costs about what a pass of NumPy's does: 458 µs for a million elements three apart, since reading one element in three pulls in every cache line of an array three times the size. So the program needs more operations than arrays to copy. One kernel of two operations on two such arrays was 0.6 ms a million elements slower than NumPy's two calls; the advection flux, seventeen operations on three such arrays and one value per row, took 3.94 ms and 5.36 ms in its two directions where NumPy took 6.9 ms in each.",
  },
  through: {
    label: "Short rows at one pitch",
    code: "y = a[2:-2, 2:-2, :-1] * (b[2:-2, 2:-2, :-1] - c[2:-2, 2:-2, :-1])     # grids 26 deep: rows of 25 every 26",
    panels: [{ title: "rows of 6 every 8, walked from the first to the last", rows: 5, pitch: 8, cell: (r, c) => {
      if (c >= 1 && c <= 6) return 0;
      // Between the first element of the first row and the last of the last: computed, then thrown away.
      if ((r === 0 && c === 0) || (r === 4 && c === 7)) return "out";
      return "w";
    } }],
    how: "Rows that cannot be joined, but where every array of the expression has the same elements between its rows: the steps <b>walk each array from its first element to its last</b>, the elements between the rows included, and only the rows are copied out of the last step's buffer. Nothing is copied in, at the price of arithmetic on elements nobody asked for. A floating-point flag that only those elements raised drops the result like any other.",
    when: "Rows of 16 to 31 elements when a quarter of the walk at most lies between them; rows shorter than 16 up to eight times the row's length apart. For a row of a few elements, NumPy calls its loop once per row and the call costs more than the row: <code>particles[:, :2] += particles[:, 2:] * dt</code> on 100,000 rows takes NumPy 1,256 µs, the same arithmetic on 200,000 elements in one run 82 µs. So a walk that computes eight times what was asked for still wins.",
  },
  pitches: {
    label: "Short rows at two pitches",
    code: "y = (a[:, :-1] + b) * c - d      # a has rows of 26, b, c and d rows of 25",
    panels: [
      { title: "a[:, :-1]: rows of 6 every 7", rows: 4, pitch: 7, cell: (r, c) => (c < 6 ? r % RUNS.length : "out") },
      { title: "b, c and d: rows of 6 every 6", rows: 4, pitch: 6, cell: (r) => r % RUNS.length },
    ],
    how: "A slice of a grid beside an array computed from one has rows at two pitches. Nothing can be walked through, so <b>each row is a program</b>: every step is a call for that row, by a walk that has worked out once everything that is the same for every row.",
    when: "From rows of 16 elements. On <code>(a + b) * c - d</code> over four million elements that was 1.06 to 1.21 times as fast as NumPy's three passes for rows of 16 to 28, and 0.70 to 0.88 for rows of 4 to 12, which are left to NumPy. Rows of one element, <code>x[:, :1]</code>, are never a call each: that ran 3 to 9 times slower than NumPy.",
  },
  rows: {
    label: "Rows of a few elements",
    code: "y = table * weights[:, None]      # 100,000 rows of 4, one weight per row\ntable[:, :2] += made",
    panels: [
      { title: "table: rows of 4, one after another", rows: 6, pitch: 4, cell: (r) => r % RUNS.length },
      { title: "weights[:, None]", rows: 6, pitch: 1, cell: () => "n" },
    ],
    how: "One operation of <code>+</code>, <code>-</code>, <code>*</code> or <code>/</code> on rows shorter than 16 is worth neither a call for each row nor a walk through what lies between them. The module makes it itself, <code>program_rows</code>: <b>two loops with nothing in them but the rows' arithmetic</b>, the processor's one instruction for the operation, as in NumPy's loop and in the kernel. The arrays may lie at any pitch, the number may be another for each row (<code>weights[:, None]</code>), and the result may be one of the arrays.",
    when: "Rows shorter than 16. Written out by themselves such loops took 105 µs for <code>table * weights[:, None]</code> on 100,000 rows of four and 90 µs for <code>table[:, :2] += made</code>, where NumPy takes 294 µs and 861 µs.",
  },
  declined: {
    label: "What is declined",
    code: "y = a[::-1] * 2.0 + b           # reversed\ny = a.T * 2.0 + b               # two orders",
    panels: [{ title: "a[::-1]: strides that run backwards", rows: 4, pitch: 13, cell: () => "x" }],
    how: "The module reads float64 arrays whose strides are positive and do not grow from one axis to the next: then NumPy's result is laid out in C order, as the output is, and a later sum adds in NumPy's order. Arrays that all lie in one other order (<code>a.T * 2.0 + b.T</code>) are walked in that order, as NumPy walks them, and the result is laid out as NumPy's. It <b>declines</b> a reversed array, arrays that lie in two orders, an array out of order beside a row or a column or with an axis of length one, an output that overlaps an operand without being it, rows shorter than 16 at more than one pitch or more than eight times their length apart, an array that steps with fewer than 32 elements at one step (save the one stored product isoneutral mixing makes), and a program with too few operations for the arrays it would copy.",
    when: "Declining is decided before anything is computed or copied. The group's nodes are then computed by their own NumPy calls, in order, and the result is the same.",
  },
};

export const css = `
.fl-code { font-family: var(--mono); font-size: .78rem; background: var(--code-bg); padding: .5rem .75rem; border-radius: 8px; overflow-x: auto; white-space: pre; margin: 0 0 .9rem; line-height: 1.55; }
.fl-panels { display: flex; flex-wrap: wrap; gap: 1rem 1.6rem; align-items: flex-start; }
.fl-panel h4 { margin: 0 0 .35rem; font-size: .72rem; font-weight: 650; color: var(--muted); max-width: 17rem; }
.fl-grid { display: grid; gap: 2px; }
.fl-c { width: .95rem; height: .95rem; border-radius: 3px; border: 1.5px solid var(--rule); background: transparent; }
.fl-c.w { background: repeating-linear-gradient(135deg, transparent 0 2px, var(--rule-strong) 2px 3px); border-color: var(--rule-strong); }
.fl-c.b { background: var(--np-soft); border-color: var(--np); }
.fl-c.c { background: var(--k-reduction-soft); border-color: var(--k-reduction); }
.fl-c.n { background: var(--k-reduction-soft); border-color: var(--k-reduction); border-radius: 50%; }
.fl-c.x { background: var(--panel); border-color: var(--muted); border-style: dashed; }
.fl-key { display: flex; flex-wrap: wrap; gap: .35rem 1rem; font-size: .74rem; color: var(--muted); margin-top: .8rem; }
.fl-key span { display: inline-flex; align-items: center; gap: .35rem; }
.fl-key .fl-c { width: .8rem; height: .8rem; }
.fl-text { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: .8rem 1.4rem; margin-top: .9rem; font-size: .84rem; line-height: 1.5; }
@media (max-width: 640px) { .fl-text { grid-template-columns: minmax(0, 1fr); } }
.fl-text h5 { margin: 0 0 .25rem; font-size: .7rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.fl-text p { margin: 0; }
.fl-text code { font-size: .76rem; }
`;

// One panel: a grid of squares, a row of memory for each row of the array.
function panel({ title, rows, pitch, cell }) {
  const grid = h("div", { class: "fl-grid", style: `grid-template-columns: repeat(${pitch}, .95rem)` });
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < pitch; c++) {
      const kind = cell(r, c);
      if (typeof kind === "number") {
        const tone = `var(${RUNS[kind % RUNS.length]})`;
        grid.append(h("span", { class: "fl-c", title: `run ${kind + 1}`, style: `background: color-mix(in srgb, ${tone} 30%, transparent); border-color: ${tone}` }));
      } else {
        grid.append(h("span", { class: `fl-c ${kind === "out" ? "" : kind}`.trim(), title: KIND[kind] }));
      }
    }
  }
  return h("div", { class: "fl-panel" }, h("h4", {}, title), grid);
}

export default async function mount(el) {
  let key = LAYOUTS[el.dataset.layout] ? el.dataset.layout : "contiguous";
  const select = h("select", { class: "w-select", "aria-label": "Layout", onchange: () => { key = select.value; render(); } },
    ...Object.entries(LAYOUTS).map(([value, layout]) => h("option", { value }, layout.label)));
  select.value = key;
  const code = h("pre", { class: "fl-code" });
  const panels = h("div", { class: "fl-panels" });
  const legend = h("div", { class: "fl-key" });
  const how = h("p", {});
  const when = h("p", {});

  function render() {
    const layout = LAYOUTS[key];
    code.textContent = layout.code;
    panels.replaceChildren(...layout.panels.map(panel));
    // The key names only what the drawing has in it.
    const seen = new Set();
    for (const p of layout.panels) for (let r = 0; r < p.rows; r++) for (let c = 0; c < p.pitch; c++) {
      const kind = p.cell(r, c);
      seen.add(typeof kind === "number" ? "run" : kind);
    }
    legend.replaceChildren(...[
      seen.has("run") ? h("span", {}, h("span", { class: "fl-c", style: "background: color-mix(in srgb, var(--g1) 30%, transparent); border-color: var(--g1)" }),
        h("span", { class: "fl-c", style: "background: color-mix(in srgb, var(--g2) 30%, transparent); border-color: var(--g2)" }), "a contiguous run a kernel reads where it lies, one colour each") : null,
      ...["w", "b", "c", "n", "x", "out"].filter((k) => seen.has(k)).map((k) => h("span", {}, h("span", { class: `fl-c ${k === "out" ? "" : k}`.trim() }), KIND[k]))].filter(Boolean));
    how.innerHTML = layout.how;
    when.innerHTML = layout.when;
  }

  el.append(
    h("div", { class: "w-controls" }, h("span", { class: "w-label" }, "Layout"), select),
    code, panels, legend,
    h("div", { class: "fl-text" }, h("div", {}, h("h5", {}, "How it is computed"), how), h("div", {}, h("h5", {}, "When"), when)),
  );
  render();
}
