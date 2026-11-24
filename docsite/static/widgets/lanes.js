// What a vector register buys, and what to do with the elements left over.
//
// A 128-bit vector register holds four float32, two float64 or sixteen uint8,
// and one instruction handles all of them. An array whose length is not a
// multiple of that leaves a remainder, and there are three things a loop can
// do with it: finish one element at a time, run one last vector that ends
// exactly at the end of the array and redoes a few elements, or load a full
// vector anyway and read past the end. The third is the shortest code and the
// one the judge forbids: its buffers end at an inaccessible page.
//
// Attributes:
//   data-n     initial element count, 1 to 48 (default 37)
//   data-mode  "simd" (default), or "tail": opens on the leftover strategies,
//              with the inaccessible page drawn and the read past the end shown

import { h, seg, range, button, stat } from "./util.js";

// Elements per 128-bit vector and bytes per element, by element type.
const TYPES = {
  float32: { per: 4, bytes: 4 },
  float64: { per: 2, bytes: 8 },
  uint8: { per: 16, bytes: 1 },
};
const SINGLE = "single", OVERLAP = "overlap", PAST = "past";
const STRATEGIES = [
  { value: SINGLE, label: "One at a time" },
  { value: OVERLAP, label: "Overlapping last vector" },
  { value: PAST, label: "Read past the end" },
];
const MIN_N = 1, MAX_N = 48, DEFAULT_N = 37;
// Cells drawn for the inaccessible page: enough for the widest overhang, never fewer than this.
const MIN_GUARD_CELLS = 3;
// Milliseconds between steps while playing.
const STEP_MS = 420;

export const css = `
.lanes-cells { align-items: center; row-gap: 5px; }
.lanes-cells .cell.guard {
  border: 1px dashed var(--bad); color: var(--bad);
  background: repeating-linear-gradient(135deg, var(--bad-soft) 0 4px, transparent 4px 8px);
}
.lanes-cells .cell.guard.bad { border-style: solid; background: var(--bad); color: var(--bg-raised); }
.lanes-guard-label { font-size: .74rem; color: var(--bad); margin-left: .4rem; white-space: nowrap; }
.lanes-status { min-height: 2.9em; }
.lanes-group { display: inline-flex; flex-wrap: wrap; align-items: center; gap: .5rem .6rem; }
`;

// The steps a loop takes over n elements with w elements per vector.
// Each step is {start, len, kind}; kind is "vector", "single", "overlap" or "past".
function plan(n, w, strategy) {
  const full = Math.floor(n / w), rem = n % w;
  const steps = [];
  for (let i = 0; i < full; i++) steps.push({ start: i * w, len: w, kind: "vector" });
  const result = { steps, full, rem, extra: 0, twice: 0, outside: 0, short: false };
  if (rem === 0) return result;
  if (strategy === PAST) {
    steps.push({ start: full * w, len: w, kind: PAST });
    result.extra = 1;
    result.outside = w - rem;
  } else if (strategy === OVERLAP && full > 0) {
    steps.push({ start: n - w, len: w, kind: OVERLAP });
    result.extra = 1;
    result.twice = w - rem;
  } else {
    for (let i = full * w; i < n; i++) steps.push({ start: i, len: 1, kind: SINGLE });
    result.extra = rem;
    result.short = strategy === OVERLAP;
  }
  return result;
}

function describe(step, index, total, n) {
  const last = step.start + step.len - 1;
  const where = `Step ${index + 1} of ${total}: `;
  if (step.kind === "vector") return `${where}one vector load covers elements ${step.start} to ${last}.`;
  if (step.kind === SINGLE) return `${where}element ${step.start}, on its own.`;
  if (step.kind === OVERLAP) {
    return `${where}one more vector, placed to end exactly at element ${n - 1}. It covers ${step.start} to ${last}, so some elements are handled a second time.`;
  }
  return `${where}a full vector load starting at element ${step.start}. The array ends at ${n - 1}; the load carries on to ${last}.`;
}

export default async function mount(el) {
  const tailMode = el.dataset.mode === "tail";
  const state = {
    n: Math.min(MAX_N, Math.max(MIN_N, Number(el.dataset.n) || DEFAULT_N)),
    type: "float32",
    strategy: tailMode ? PAST : SINGLE,
    pos: 0, // steps taken so far
  };
  let timer = null;

  const cells = h("div", { class: "cells lanes-cells" });
  const status = h("div", { class: "w-note lanes-status", "aria-live": "polite" });
  const verdict = h("div", { class: "w-verdict", hidden: true });
  const scalarSteps = stat("steps, one element at a time", "np");
  const vectorSteps = stat("steps, vector loop", "hot");
  const twice = stat("elements handled twice");
  const outside = stat("bytes read outside the buffer");

  const current = () => plan(state.n, TYPES[state.type].per, state.strategy);

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
    playButton.textContent = "Play";
  };

  function render() {
    const { per, bytes } = TYPES[state.type];
    const p = current();
    const total = p.steps.length;
    state.pos = Math.min(state.pos, total);
    const now = state.pos > 0 ? p.steps[state.pos - 1] : null;
    const covered = new Array(state.n).fill(false);
    for (const step of p.steps.slice(0, Math.max(0, state.pos - 1))) {
      for (let i = step.start; i < step.start + step.len && i < state.n; i++) covered[i] = true;
    }
    const inNow = (i) => now && i >= now.start && i < now.start + now.len;

    cells.replaceChildren();
    for (let i = 0; i < state.n; i++) {
      const cls = inNow(i) ? (covered[i] ? "np" : "hot") : covered[i] ? "done" : "";
      cells.appendChild(h("span", { class: `cell small ${cls}`.trim() }, i));
    }
    if (tailMode || state.strategy === PAST) {
      const guardCells = Math.max(MIN_GUARD_CELLS, per - 1);
      for (let g = 0; g < guardCells; g++) {
        const touched = now && now.kind === PAST && inNow(state.n + g);
        cells.appendChild(h("span", { class: `cell small guard ${touched ? "bad" : ""}`.trim(), "aria-label": "inaccessible" }, "×"));
      }
      cells.appendChild(h("span", { class: "lanes-guard-label" }, "inaccessible page"));
    }

    if (!now) {
      status.textContent = `${state.n} ${state.type} elements, ${per} per vector: ${p.full} full vector${p.full === 1 ? "" : "s"}${p.rem ? ` and ${p.rem} left over` : ", nothing left over"}.`;
    } else {
      status.textContent = describe(now, state.pos - 1, total, state.n);
    }

    const done = state.pos === total;
    verdict.hidden = true;
    verdict.className = "w-verdict";
    if (now && now.kind === PAST) {
      verdict.hidden = false;
      verdict.classList.add("bad");
      verdict.innerHTML = `The load touches a page the program does not own: SIGBUS. The judge reports <code>memory safety</code>.`;
    } else if (done && p.short) {
      verdict.hidden = false;
      verdict.textContent = `An overlapping vector needs at least one full vector to overlap with. Arrays shorter than one vector (${per} elements here) take a scalar path.`;
    } else if (done && now && now.kind === OVERLAP) {
      verdict.hidden = false;
      verdict.classList.add("good");
      verdict.textContent = `Every load stayed inside the buffer. ${p.twice} element${p.twice === 1 ? " was" : "s were"} handled twice, which is harmless whenever doing an element again gives the same answer.`;
    } else if (done && p.rem === 0 && state.strategy !== SINGLE) {
      verdict.hidden = false;
      verdict.textContent = `${state.n} is a multiple of ${per}: nothing is left over, so the strategies do not differ.`;
    }

    scalarSteps.set(String(state.n));
    vectorSteps.set(p.extra ? `${p.full} + ${p.extra}` : String(p.full));
    twice.set(String(p.twice));
    outside.set(String(p.outside * bytes));
    outside.className = `w-stat ${p.outside ? "bad" : ""}`.trim();
    stepButton.disabled = done;
    if (done) stop();
  }

  const restart = (pos = 0) => { stop(); state.pos = pos; render(); };
  const step = () => { state.pos += 1; render(); };

  const stepButton = button("Step", () => { stop(); step(); });
  // Play steps through on a timer whatever the reduced-motion setting: each
  // step is a redraw, not a movement, and the reader asked for it. From the
  // last step, where the tail view opens, it starts again from the first.
  const playButton = button("Play", () => {
    if (timer) return stop();
    if (state.pos === current().steps.length) restart();
    playButton.textContent = "Pause";
    timer = setInterval(step, STEP_MS);
  }, "primary");
  const resetButton = button("Reset", () => restart());

  const elements = range({
    label: "Elements", min: MIN_N, max: MAX_N, value: state.n,
    onInput: (v) => { state.n = v; restart(); },
  });
  const typeControl = seg(
    Object.keys(TYPES).map((name) => ({ value: name, label: `${name} · ${TYPES[name].per}` })),
    state.type, (v) => { state.type = v; restart(); },
  );
  const strategyControl = seg(STRATEGIES, state.strategy, (v) => { state.strategy = v; restart(); });

  // A label and its control wrap onto a new line together.
  const group = (label, control) => h("span", { class: "lanes-group" }, h("span", { class: "w-label" }, label), control);
  const leftover = [group("Leftover", strategyControl)];
  const shape = [elements, group("Per vector", typeControl)];
  el.append(
    h("div", { class: "w-controls" }, tailMode ? [leftover, shape] : [shape, leftover]),
    h("div", { class: "w-stage" }, cells),
    h("div", { class: "w-legend" },
      h("span", {}, h("i", { class: "hot" }), "in this step"),
      h("span", {}, h("i", { style: "background: var(--panel); border: 1px solid var(--rule-strong)" }), "handled"),
      h("span", {}, h("i", { class: "np" }), "handled a second time"),
      h("span", {}, h("i", { class: "bad" }), "outside the buffer")),
    status,
    verdict,
    h("div", { class: "w-controls bottom" }, stepButton, playButton, resetButton),
    h("div", { class: "w-readout" }, scalarSteps, vectorSteps, twice, outside),
  );

  // The tail view opens on the problem it is there to show: the final load already made.
  restart(tailMode ? current().steps.length : 0);
}
