// Block maxima instead of tracking an index.
//
// The obvious argmax carries "best value so far" and "its index" through the
// loop, and branches on every element. The kernel does neither. It reduces
// each block of elements to one number, its maximum, and remembers only which
// block held the best maximum so far; that marker moves when a block's
// maximum is strictly greater, so of two tied blocks the first one stays.
// When the array is finished, only the winning block is looked at again, for
// the first element equal to the maximum.
//
// A maximum taken with NaN in the block is NaN, and a compare with NaN comes
// out "unordered". That one outcome stops the loop, and a scalar scan finds
// the first NaN, which by NumPy's rule wins over everything.
//
// The figure uses six blocks of eight elements. The float32 kernel uses
// blocks of 64.
//
// Attributes:
//   data-seed  seed of the first data set (default 3)

import { h, button, stat, rng } from "./util.js";

const BLOCKS = 6, BLOCK = 8;
const N = BLOCKS * BLOCK;
// Ordinary values lie below this; the maximum is placed above it, so it is unique until a tie is added.
const ORDINARY_BELOW = 90, VALUE_BELOW = 100;
// Milliseconds between steps while playing.
const STEP_MS = 800;
const DEFAULT_SEED = 3;

export const css = `
.amx-blocks { display: flex; flex-wrap: wrap; gap: .9rem 1.1rem; }
.amx-block { padding: .35rem .4rem .3rem; border: 1px solid transparent; border-radius: 8px; }
.amx-block.now { border-color: var(--hot); background: var(--hot-soft); }
.amx-block.best { border-color: var(--good); }
.amx-block.now.best { border-color: var(--hot); }
.amx-cap { display: flex; justify-content: space-between; gap: .5rem; font-size: .74rem; color: var(--muted); margin-top: .3rem; min-height: 1.2rem; font-variant-numeric: tabular-nums; }
.amx-cap b { color: var(--ink); font-family: var(--mono); }
.amx-cap .tag { color: var(--good); font-weight: 650; }
.amx-block .cell.nan { font-size: .5rem; }
.amx-status { min-height: 4.4em; }
.w-btn[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
`;

// One data set: ordinary values, the position of the maximum, and where a
// tie and a NaN go when the reader asks for them.
function makeData(seed) {
  const random = rng(seed);
  const pick = (lo, hi) => lo + Math.floor(random() * (hi - lo));
  const base = Array.from({ length: N }, () => pick(0, ORDINARY_BELOW));
  // The maximum sits in the first four blocks, so a later block is free for the tie.
  const maxAt = pick(0, 4 * BLOCK);
  base[maxAt] = pick(ORDINARY_BELOW, VALUE_BELOW);
  const maxBlock = Math.floor(maxAt / BLOCK);
  const tieAt = pick((maxBlock + 1) * BLOCK, N);
  // The NaN goes in the third block or later, so the loop has something to do before it stops.
  let nanAt = pick(2 * BLOCK, N);
  while (nanAt === maxAt || nanAt === tieAt) nanAt = pick(2 * BLOCK, N);
  return { base, maxAt, tieAt, nanAt };
}

function values(data, tie, nan) {
  const x = [...data.base];
  if (tie) x[data.tieAt] = x[data.maxAt];
  if (nan) x[data.nanAt] = NaN;
  return x;
}

// np.argmax: the first NaN wins over everything, otherwise the first maximum.
function numpyArgmax(x) {
  let best = 0;
  for (let i = 0; i < x.length; i++) {
    if (Number.isNaN(x[i])) return i;
    if (x[i] > x[best]) best = i;
  }
  return best;
}

// Maximum of one block the way the kernel's reduction gives it: NaN if the block holds one.
function blockMax(x, b) {
  const part = x.slice(b * BLOCK, (b + 1) * BLOCK);
  return part.some(Number.isNaN) ? NaN : Math.max(...part);
}

// The kernel's steps over x. Each is {kind, block, max, best, bestMax, moved} for a block
// reduction, then one final step: {kind: "rescan" | "nan", block, index, scanned}.
function plan(x) {
  const steps = [];
  let best = -1, bestMax = -Infinity;
  for (let b = 0; b < BLOCKS; b++) {
    const max = blockMax(x, b);
    if (Number.isNaN(max)) {
      steps.push({ kind: "reduce", block: b, max, best, bestMax, moved: false, unordered: true });
      const index = x.findIndex(Number.isNaN);
      steps.push({ kind: "nan", block: b, index, scanned: index - b * BLOCK + 1 });
      return steps;
    }
    const moved = max > bestMax;
    const previous = bestMax;
    if (moved) { best = b; bestMax = max; }
    steps.push({ kind: "reduce", block: b, max, best, bestMax, previous, moved });
  }
  const index = best * BLOCK + x.slice(best * BLOCK, (best + 1) * BLOCK).indexOf(bestMax);
  steps.push({ kind: "rescan", block: best, index, scanned: BLOCK, bestMax });
  return steps;
}

function describe(step, first) {
  if (step.kind === "rescan") {
    return `The array is finished. Only block ${step.block} is looked at again: the first of its ${BLOCK} elements equal to ${step.bestMax} is at index <b>${step.index}</b>.`;
  }
  if (step.kind === "nan") {
    return `A scalar scan of block ${step.block} finds the first NaN at index <b>${step.index}</b>. NaN wins over everything, as in NumPy.`;
  }
  const head = `Block ${step.block} reduces to ${Number.isNaN(step.max) ? "NaN" : step.max}. `;
  if (step.unordered) return `${head}A compare with NaN is neither greater nor not greater: it comes out <i>unordered</i>, and that stops the block loop.`;
  if (first) return `${head}It is the first block, so it is the best so far.`;
  if (step.moved) return `${head}${step.max} > ${step.previous}, so the best-block marker moves here.`;
  if (step.max === step.previous) return `${head}Equal to the best so far, not strictly greater: the marker stays on block ${step.best}, so the first of the tied maxima wins.`;
  return `${head}Not greater than ${step.previous}: the marker stays on block ${step.best}.`;
}

export default async function mount(el) {
  const state = { seed: Number(el.dataset.seed) || DEFAULT_SEED, tie: false, nan: false, pos: 0 };
  let data = makeData(state.seed);
  let timer = null;

  const blocks = h("div", { class: "amx-blocks" });
  const status = h("div", { class: "w-note amx-status", "aria-live": "polite" });
  const result = stat("index the kernel returns", "hot");
  const reference = stat("np.argmax on the same data", "np");
  const rescanned = stat("elements looked at twice");
  const branches = stat("branches taken because of the data");

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
    playButton.textContent = "Play";
  };

  function render() {
    const x = values(data, state.tie, state.nan);
    const steps = plan(x);
    state.pos = Math.min(state.pos, steps.length);
    const now = state.pos > 0 ? steps[state.pos - 1] : null;
    const done = state.pos === steps.length;
    const reduced = steps.slice(0, state.pos).filter((step) => step.kind === "reduce");
    const lastReduce = reduced[reduced.length - 1];
    const best = lastReduce ? lastReduce.best : -1;
    const final = now && now.kind !== "reduce" ? now : null;

    blocks.replaceChildren();
    for (let b = 0; b < BLOCKS; b++) {
      const seen = reduced.find((step) => step.block === b);
      const isNow = now && now.block === b && (now.kind === "reduce" || final);
      const cells = h("div", { class: "cells nowrap" });
      for (let j = 0; j < BLOCK; j++) {
        const i = b * BLOCK + j;
        let cls = seen ? "done" : "";
        if (final && final.block === b) {
          // The rescan compares the whole block at once; the scalar scan for a NaN stops where it finds one.
          cls = i === final.index ? "good" : final.kind === "rescan" || i < final.index ? "hot" : cls;
        }
        const nan = Number.isNaN(x[i]);
        cells.appendChild(h("span", { class: `cell small ${cls} ${nan ? "nan" : ""}`.trim(), title: `index ${i}` }, nan ? "NaN" : x[i]));
      }
      const max = seen ? (Number.isNaN(seen.max) ? "NaN" : seen.max) : "–";
      blocks.appendChild(h("div", { class: `amx-block ${isNow ? "now" : ""} ${best === b ? "best" : ""}`.trim() },
        cells,
        h("div", { class: "amx-cap" },
          h("span", {}, `block ${b} · max `, h("b", {}, max)),
          best === b ? h("span", { class: "tag" }, "best so far") : null)));
    }

    status.innerHTML = now
      ? describe(now, state.pos === 1)
      : `${N} elements in ${BLOCKS} blocks of ${BLOCK}. Each step reduces one block to its maximum and compares that with the best so far. No element is compared with an index, and nothing branches on a value.`;

    const expected = numpyArgmax(x);
    reference.set(String(expected));
    result.set(final ? `${final.index}${final.index === expected ? " ✓" : " ✗"}` : "–");
    rescanned.set(final ? String(final.scanned) : "–");
    branches.set(reduced.some((step) => step.unordered) ? "1" : "0");
    stepButton.disabled = done;
    tieButton.setAttribute("aria-pressed", String(state.tie));
    nanButton.setAttribute("aria-pressed", String(state.nan));
    if (done) stop();
  }

  const restart = () => { stop(); state.pos = 0; render(); };
  const step = () => { state.pos += 1; render(); };

  const stepButton = button("Step", () => { stop(); step(); });
  // Play steps through on a timer whatever the reduced-motion setting: each
  // step is a redraw, not a movement, and the reader asked for it. From the
  // last step it starts again from the first.
  const playButton = button("Play", () => {
    if (timer) return stop();
    if (stepButton.disabled) { state.pos = 0; render(); }
    playButton.textContent = "Pause";
    timer = setInterval(step, STEP_MS);
  }, "primary");
  const resetButton = button("Reset", restart);
  const newButton = button("New data", () => { state.seed += 1; data = makeData(state.seed); restart(); });
  const tieButton = button("Add a tie", () => { state.tie = !state.tie; restart(); });
  const nanButton = button("Add a NaN", () => { state.nan = !state.nan; restart(); });

  el.append(
    h("div", { class: "w-controls" }, stepButton, playButton, resetButton,
      h("span", { class: "w-spacer" }), tieButton, nanButton, newButton),
    h("div", { class: "w-stage" }, blocks),
    status,
    h("div", { class: "w-readout" }, result, reference, rescanned, branches),
    h("div", { class: "w-note" }, `Six blocks of eight keep the figure readable. The float32 kernel works in blocks of 64 elements and the float64 kernel in blocks of 32; arrays too short for that take straight-line code with the same idea.`),
  );
  render();
}
