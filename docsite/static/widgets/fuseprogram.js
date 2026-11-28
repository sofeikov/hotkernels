// One program of kernel calls, run a block at a time.
//
// An expression the library has no single kernel for is covered with several,
// and their calls are a program: for each block of up to 4,096 elements, every
// step runs on that block, and what one step gives the next is a buffer of one
// block, 32 KB, that stays in the first-level cache. The reader steps through
// one block of a program: which kernel runs, what it reads from memory and from
// the buffers, where it writes, and which buffers are free again.
//
// The programs are the prototype's own. Each expression was run on lazy arrays
// of prototypes/lazy (NumPy 2.5.5, CPython 3.12, the layer in Python without
// its C core), and a planner hooked into
// evaluate.planner recorded the steps fuse._Program wrote for fuse.tile's cover
// with the library of the eighty kernels in prototypes/lazy/kernels/ (read off
// their oracles by fuse.load_library; the module _fused itself was not built,
// since planning reads only its table of names). A step is the kernel, then
// where it writes, then the places of the arrays and numbers it reads, as
// _fused.program takes it. The cavity flow's update of u is npbench's line as
// the corpus runs it; the chain and the expression no shape fits are written
// for the figure.
//
// The counts of arrays read and written are counting, not timing: each array
// an operation reads or writes once over its elements. How many runs and
// blocks a program walks follows from the rules of docs/lazy-fusion-design.md,
// "Computing a group", and the sizes in prototypes/lazy/programs/corpus.py.
//
// Attributes:
//   data-program  cavity (default), chain or root (the expression no shape fits)

import { h, seg, stat, button } from "./util.js";

// Elements in a block, and a buffer's size in kilobytes: BLOCK_ELEMENTS of fuse.py, eight bytes each.
const BLOCK = 4096, BUFFER_KB = 32;

const PROGRAMS = {
  cavity: {
    label: "Cavity flow",
    code: "u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])\n    - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])\n    - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])\n    + nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])\n          + dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))",
    operations: 24,
    // What NumPy's calls read and write, counted from the tree: 38 arrays read, 24 written.
    numpyReads: 38, numpyWrites: 24,
    arrays: ["un[1:-1, 0:-2]", "un[1:-1, 2:]", "un[1:-1, 1:-1]", "un[0:-2, 1:-1]", "un[2:, 1:-1]", "un[1:-1, 1:-1]",
      "un[1:-1, 1:-1]", "un[1:-1, 1:-1]", "un[1:-1, 1:-1]", "un[1:-1, 0:-2]", "vn[1:-1, 1:-1]", "un[1:-1, 1:-1]",
      "un[0:-2, 1:-1]", "p[1:-1, 2:]", "p[1:-1, 0:-2]"],
    numbers: ["dt / dx**2", "2", "dt / dy**2", "dt", "dx (= dy)", "nu", "dt / (2 * rho * dx)"],
    steps: [
      [45, 15, 0, 1, 2, 0, 1],
      [45, 16, 3, 4, 5, 2, 1],
      [73, 17, 6, 7, 8, 9, 3, 4],
      [73, 18, 17, 10, 11, 12, 3, 4],
      [6, -1, 15, 16, 18, 13, 14, 5, 6],
    ],
    kernels: ["fuse_mul_s_add_v_sub_v_mul_s_v_f64", "fuse_mul_s_add_v_sub_v_mul_s_v_f64",
      "fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64", "fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64",
      "fuse_add_mul_s_add_v_v_sub_v_mul_s_sub_v_v_f64"],
    shapes: ["mul(S0,add(V0,sub(V1,mul(S1,V2))))", "mul(S0,add(V0,sub(V1,mul(S1,V2))))",
      "sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))", "sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))",
      "add(mul(S0,add(V0,V1)),sub(V2,mul(S1,sub(V3,V4))))"],
    vectors: [3, 3, 4, 4, 5],
    fragments: [
      "dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])",
      "dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])",
      "un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])",
      "(buffer 2) - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])",
      "(buffer 3) - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * ((buffer 0) + (buffer 1))",
    ],
    // The corpus's large cavity is a grid of 401 × 401 (CAVITY in corpus.py); u[1:-1, 1:-1] is 399 rows of 399.
    runs: 399, runLength: 399,
    walk: "On the corpus's grid of 401 × 401, <code>u[1:-1, 1:-1]</code> is 399 rows of 399 elements, 401 apart. Each row is a contiguous run, and a run of 399 is one block: the program runs once for each row.",
  },
  chain: {
    label: "A chain",
    code: "x = a\nfor _ in range(6):\n    x = x * 1.01 + b",
    operations: 12,
    numpyReads: 18, numpyWrites: 12,
    arrays: ["b", "a"],
    numbers: ["1.01"],
    steps: [[10, 2, 0, 1, 0], [10, 3, 0, 2, 0], [10, 2, 0, 3, 0], [10, 3, 0, 2, 0], [10, 2, 0, 3, 0], [10, -1, 0, 2, 0]],
    kernels: Array(6).fill("fuse_add_scaled_f64"),
    shapes: Array(6).fill("add(V0,mul(S0,V1))"),
    vectors: [2, 2, 2, 2, 2, 2],
    fragments: ["b + 1.01 * a", "b + 1.01 * (buffer 0)", "b + 1.01 * (buffer 1)", "b + 1.01 * (buffer 0)", "b + 1.01 * (buffer 1)", "b + 1.01 * (buffer 0)"],
    runs: 1, runLength: 1_000_000,
    walk: "Twelve operations, one expression: nothing asks for <code>x</code> between the turns of the loop. On contiguous arrays of 1,000,000 elements the program is one run of 245 blocks, the last of 576 elements. Each step reads the one before, so the program needs two buffers however long the chain is.",
  },
  root: {
    label: "No shape fits",
    code: "y = np.sqrt(w * w + g * g) / 2.0 - w     # w is a grid, g one row",
    operations: 5,
    numpyReads: 7, numpyWrites: 5,
    arrays: ["w", "g * g"],
    numbers: ["2.0"],
    steps: [[61, 2, 0], [20, 3, 2, 1], [63, 2, 3], [32, 3, 2, 0], [77, -1, 3, 0]],
    kernels: ["fuse_sq_v_f64", "fuse_add_v_v_f64", "fuse_sqrt_v_f64", "fuse_div_v_s_f64", "fuse_sub_v_v_f64"],
    shapes: ["sq(V0)", "add(V0,V1)", "sqrt(V0)", "div(V0,S0)", "sub(V0,V1)"],
    vectors: [1, 2, 1, 1, 2],
    fragments: ["w * w", "(buffer 0) + g * g", "np.sqrt(buffer 1)", "(buffer 0) / 2.0", "(buffer 1) - w"],
    runs: 1, runLength: 1_000_000,
    walk: "No shape of the library has this tree, so every step is a kernel of one operation. They fuse nothing by themselves, and still the expression is one pass over memory. <code>g * g</code> has another shape and is computed first, by NumPy; the same for every row, it is read from a buffer that holds it over and over for the length of a block. On a contiguous grid of 1,000 × 1,000 that is one run of 245 blocks.",
  },
};

export const css = `
.fp-code { font-family: var(--mono); font-size: .78rem; background: var(--code-bg); padding: .5rem .75rem; border-radius: 8px; overflow-x: auto; white-space: pre; margin: 0 0 .9rem; line-height: 1.55; }
.fp-body { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: .9rem; }
.fp-body > .fp-wide { grid-column: 1 / -1; }
@media (max-width: 560px) { .fp-body { grid-template-columns: minmax(0, 1fr); } }
.fp-col { border: 1px solid var(--rule); border-radius: 10px; padding: .6rem .65rem; background: var(--bg); min-width: 0; }
.fp-col h4 { margin: 0 0 .5rem; font-size: .7rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.fp-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.fp-chip { font-family: var(--mono); font-size: .7rem; padding: .12rem .4rem; border-radius: 5px; border: 1.5px solid var(--rule-strong); background: var(--bg-raised); color: var(--muted); transition: background .2s, border-color .2s, color .2s; white-space: nowrap; }
.fp-chip.read { border-color: var(--np); background: var(--np-soft); color: var(--np); }
.fp-chip.num { border-radius: 999px; }
.fp-chip.num.read { border-color: var(--k-reduction); background: var(--k-reduction-soft); color: var(--k-reduction); }
.fp-chip.result { display: block; margin-top: .6rem; text-align: center; }
.fp-chip.result.wrote { border-color: var(--hot); background: var(--hot-soft); color: var(--hot); }
.fp-steps { list-style: none; margin: 0; padding: 0 !important; display: grid; gap: 4px; }
.fp-steps li { border: 1.5px solid var(--rule); border-radius: 7px; padding: .3rem .45rem; font-size: .74rem; transition: border-color .2s, background .2s, opacity .2s; }
.fp-steps li.done { opacity: .55; }
.fp-steps li.now { border-color: var(--hot); background: var(--hot-soft); }
.fp-steps li { min-width: 0; }
.fp-steps code { font-family: var(--mono); font-size: .7rem; border: 0; background: none; padding: 0; white-space: normal; overflow-wrap: anywhere; }
.fp-steps .k { display: block; font-weight: 650; color: var(--ink); }
.fp-steps .arrow { color: var(--muted); }
.fp-buf { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: .4rem; align-items: center; border: 1.5px dashed var(--rule-strong); border-radius: 7px; padding: .3rem .45rem; margin-bottom: 4px; font-size: .72rem; color: var(--muted); transition: border-color .2s, background .2s; }
.fp-buf b { font-family: var(--mono); font-size: .7rem; color: var(--ink); }
.fp-buf.full { border-style: solid; border-color: var(--good); background: var(--good-soft); color: var(--good); }
.fp-buf.read { border-color: var(--np); background: var(--np-soft); color: var(--np); }
.fp-buf.wrote { border-style: solid; border-color: var(--hot); background: var(--hot-soft); color: var(--hot); }
.fp-where { font-size: .7rem; color: var(--muted); margin-top: .5rem; }
.fp-block { font-variant-numeric: tabular-nums; font-size: .82rem; color: var(--muted); }
.fp-block b { color: var(--ink); }
`;

// The arrays a program reads, each slice once, in the order first met.
function distinct(program) {
  return [...new Set(program.arrays)];
}

// Says what a place of a step names: an array, a buffer, or the result.
function placeOf(program, place) {
  if (place === -1) return { kind: "result" };
  if (place < program.arrays.length) return { kind: "array", index: place };
  return { kind: "buffer", index: place - program.arrays.length };
}

// Splits a step into what it writes, the arrays or buffers it reads, and the numbers.
function decode(program, i) {
  const step = program.steps[i];
  const vectors = program.vectors[i];
  return {
    writes: placeOf(program, step[1]),
    reads: step.slice(2, 2 + vectors).map((place) => placeOf(program, place)),
    numbers: step.slice(2 + vectors),
  };
}

// How many buffers a program uses: the highest buffer any step writes, plus one.
function buffersOf(program) {
  let most = 0;
  program.steps.forEach((_, i) => {
    const { writes } = decode(program, i);
    if (writes.kind === "buffer") most = Math.max(most, writes.index + 1);
  });
  return most;
}

// What each buffer holds after the first `done` steps of a block: the step that wrote it, or null when free.
function buffersAfter(program, done) {
  const held = Array(buffersOf(program)).fill(null);
  for (let i = 0; i < done; i++) {
    const { writes, reads } = decode(program, i);
    // A buffer is free again as soon as the step that read it is written.
    for (const r of reads) if (r.kind === "buffer") held[r.index] = null;
    if (writes.kind === "buffer") held[writes.index] = i;
  }
  return held;
}

export default async function mount(el) {
  let key = PROGRAMS[el.dataset.program] ? el.dataset.program : "cavity";
  // Steps made in the current block (0 to steps.length), and the block's number from 1.
  let done = 0, block = 1;

  const code = h("pre", { class: "fp-code" });
  const memory = h("div", { class: "fp-col" });
  const steps = h("div", { class: "fp-col fp-wide" });
  const cache = h("div", { class: "fp-col" });
  const where = h("p", { class: "w-note", html: "" });
  const note = h("p", { class: "w-verdict", "aria-live": "polite" });
  const blockLine = h("span", { class: "fp-block" });
  const next = button("Next step", () => advance(), "primary");
  const reset = button("Start again", () => { done = 0; block = 1; render(); });
  const numpy = stat("NumPy: calls, each a pass over memory", "np");
  const calls = stat("kernel calls per block, in the cache", "hot");
  const traffic = stat("arrays read and written: NumPy");
  const ours = stat("the same, by the program: each slice once", "good");

  function advance() {
    const program = PROGRAMS[key];
    if (done < program.steps.length) {
      done += 1;
    } else {
      done = 1;
      block = block >= program.runs * Math.ceil(program.runLength / BLOCK) ? 1 : block + 1;
    }
    render();
  }

  function render() {
    const program = PROGRAMS[key];
    const blocks = program.runs * Math.ceil(program.runLength / BLOCK);
    const current = done > 0 ? decode(program, done - 1) : null;
    const held = buffersAfter(program, done);

    code.textContent = program.code;

    // Main memory: the arrays the program reads, without repeats, and the result.
    // A slice a step reads, by name: one read twice is the same memory, and comes from memory once.
    const readNow = new Set(current ? current.reads.filter((r) => r.kind === "array").map((r) => program.arrays[r.index]) : []);
    const numbersNow = new Set(current ? current.numbers : []);
    memory.replaceChildren(
      h("h4", {}, "Main memory"),
      h("div", { class: "fp-chips" }, ...distinct(program).map((name) => h("span", { class: `fp-chip${readNow.has(name) ? " read" : ""}` }, name))),
      h("h4", { style: "margin-top: .7rem" }, "Numbers, in registers"),
      h("div", { class: "fp-chips" }, ...program.numbers.map((name, i) => h("span", { class: `fp-chip num${numbersNow.has(i) ? " read" : ""}` }, name))),
      h("span", { class: `fp-chip result${current && current.writes.kind === "result" ? " wrote" : ""}` }, "the result's block"));

    // The program: one row per step.
    steps.replaceChildren(h("h4", {}, `The program: ${program.steps.length} steps for every block`),
      h("ol", { class: "fp-steps" }, ...program.steps.map((_, i) => {
        const { writes } = decode(program, i);
        const target = writes.kind === "result" ? "the result" : `buffer ${writes.index}`;
        return h("li", { class: i === done - 1 ? "now" : i < done - 1 ? "done" : "" },
          h("code", { class: "k" }, `${i + 1}. ${program.kernels[i]}`),
          h("code", {}, program.fragments[i]), " ", h("span", { class: "arrow" }, `→ ${target}`));
      })));

    // The buffers: what each holds after the step just made.
    const bufferReads = new Set(current ? current.reads.filter((r) => r.kind === "buffer").map((r) => r.index) : []);
    cache.replaceChildren(h("h4", {}, `Buffers in the cache, ${BUFFER_KB} KB each`),
      ...held.map((step, i) => {
        const wrote = current && current.writes.kind === "buffer" && current.writes.index === i;
        const read = bufferReads.has(i);
        const cls = wrote ? "wrote" : read ? "read" : step !== null ? "full" : "";
        const text = wrote ? `step ${done} wrote it` : read ? `read by step ${done}, free again` : step !== null ? `holds step ${step + 1}` : "free";
        return h("div", { class: `fp-buf ${cls}`.trim() }, h("b", {}, `buffer ${i}`), h("span", {}, text));
      }),
      h("p", { class: "fp-where" }, `${buffersOf(program)} buffer${buffersOf(program) === 1 ? "" : "s"} for ${program.steps.length} steps, reused for every block. Nothing written into them reaches main memory.`));

    blockLine.innerHTML = `Block <b>${block}</b> of ${blocks.toLocaleString("en-US")}, step <b>${done}</b> of ${program.steps.length}`;
    next.textContent = done < program.steps.length ? "Next step" : "Next block";

    if (!current) {
      note.className = "w-verdict";
      note.textContent = `Press Next step to run the first step of block ${block}. Every buffer is free at the start of a block.`;
    } else {
      const fromMemory = [...new Set(current.reads.filter((r) => r.kind === "array").map((r) => program.arrays[r.index]))];
      const fromBuffers = current.reads.filter((r) => r.kind === "buffer").map((r) => `buffer ${r.index}`);
      const parts = [];
      if (fromMemory.length) parts.push(`${fromMemory.length} slice${fromMemory.length === 1 ? "" : "s"} of memory`);
      if (fromBuffers.length) parts.push(fromBuffers.join(" and "));
      const target = current.writes.kind === "result" ? "the result's block, which goes to memory" : `buffer ${current.writes.index}, in the cache`;
      note.className = `w-verdict ${current.writes.kind === "result" ? "hot" : ""}`.trim();
      note.textContent = `Step ${done}: ${program.kernels[done - 1]}, the shape ${program.shapes[done - 1]}, reads ${parts.join(" and ")} and writes ${target}.`;
    }
    where.innerHTML = program.walk;

    numpy.set(String(program.operations));
    calls.set(String(program.steps.length));
    traffic.set(String(program.numpyReads + program.numpyWrites));
    // Each slice once from memory (a block read again by a later step is still in the cache), and the result once.
    ours.set(String(distinct(program).length + 1));
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Program"),
      seg(Object.entries(PROGRAMS).map(([value, p]) => ({ value, label: p.label })), key, (v) => { key = v; done = 0; block = 1; render(); })),
    code,
    h("div", { class: "fp-body" }, steps, memory, cache),
    note,
    h("div", { class: "w-controls bottom" }, next, reset, h("span", { class: "w-spacer" }), blockLine),
    where,
    h("div", { class: "w-readout" }, numpy, calls, traffic, ours),
  );
  render();
}
