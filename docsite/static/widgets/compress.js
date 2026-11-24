// How `x[mask]` is done without a branch.
//
// The obvious loop asks "is this element selected?" before each store, and on
// an unpredictable mask the processor guesses that answer wrong about half
// the time. The figure shows the two ways round it.
//
//   Branchless loop  Store every element at the current output position and
//                    advance the position only when the element is selected.
//                    An unselected element is overwritten by the next store;
//                    the last one, if unselected, stays behind just past the
//                    returned count. That is the naive C reference.
//   Table lookup     Pack the eight mask bytes of a block into one number,
//                    use it to pick a row of a 256-row table of shuffle
//                    patterns, permute the block and store it whole. The write
//                    pointer advances by the number selected.
//
// Attributes:
//   data-view  "loop" (default) or "table": the view the figure opens on

import { h, s, seg, button, stat, rng } from "./util.js";

const VALUES = [3.1, 0.7, 5.2, 1.4, 9.0, 2.6, 7.3, 4.8];
const N = VALUES.length;
// The mask the figure opens with: its last element is unselected, so the stray store shows.
const START_MASK = [1, 0, 1, 1, 0, 1, 1, 0];
const LOOP = "loop", TABLE = "table";
// Milliseconds between steps while playing.
const STEP_MS = 650;
// Geometry of a .cell, to place the arrows: width in rem, gap in px (see widgets.css).
const CELL_REM = 2.1, GAP_PX = 3;
// Height of the strip the arrows are drawn in, in px.
const ARROW_HEIGHT = 44;

export const css = `
.cmp { --lab: 4.6rem; --pitch: calc(2.1rem + 3px); min-width: calc(var(--lab) + 8 * var(--pitch)); }
.cmp-row { display: flex; align-items: center; margin-bottom: 6px; }
.cmp-lab { width: var(--lab); flex: none; font-family: var(--mono); font-size: .76rem; color: var(--muted); }
.cmp-idx .cell { border-color: transparent; background: transparent; color: var(--muted); height: 1.2rem; }
.cmp .cell.store { box-shadow: 0 0 0 2px var(--hot); }
.cmp .cell.stray { border-style: dashed; border-color: var(--bad); color: var(--bad); background: var(--bad-soft); }
.cmp .cell.empty { border-style: dashed; color: var(--muted); background: transparent; }
.cmp-pointer { font-family: var(--mono); font-size: .72rem; color: var(--hot); height: 1.3rem; white-space: nowrap; }
.cmp-arrows { display: block; margin-left: var(--lab); }
.cmp-arrows path { fill: none; stroke: var(--hot); stroke-width: 1.6; }
.cmp-arrows marker path { fill: var(--hot); stroke: none; }
.cmp-number { font-family: var(--mono); font-size: .82rem; margin: .5rem 0 .7rem; }
.cmp-number b { color: var(--hot); }
.cmp-status { min-height: 2.9em; }
`;

// State of the branchless loop after `steps` iterations: which source index
// each output slot holds, the output position k, and where the last store went.
function runLoop(mask, steps) {
  const out = new Array(N).fill(null);
  let k = 0, last = -1;
  for (let i = 0; i < steps; i++) {
    out[k] = i;
    last = k;
    k += mask[i];
  }
  return { out, k, last };
}

function row(label, cells, cls = "") {
  return h("div", { class: `cmp-row ${cls}`.trim() }, h("span", { class: "cmp-lab" }, label), h("div", { class: "cells nowrap" }, cells));
}

// Where the write pointer sits after the store: a caret under output slot k.
function pointer(k, text) {
  return h("div", { class: "cmp-pointer", style: `margin-left: calc(var(--lab) + ${k} * var(--pitch))` }, `▲ ${text}`);
}

// Curved arrows from x[source] down to output slot j, for the selected elements.
function arrows(sources) {
  const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
  const cell = CELL_REM * rem, pitch = cell + GAP_PX;
  const width = N * pitch, centre = (i) => i * pitch + cell / 2;
  const svg = s("svg", { class: "cmp-arrows", width, height: ARROW_HEIGHT, viewBox: `0 0 ${width} ${ARROW_HEIGHT}`, "aria-hidden": "true" },
    // The head points along +x; `orient: auto` turns it to follow the end of the curve.
    s("defs", {}, s("marker", { id: "cmp-head", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto" },
      s("path", { d: "M0 0 L8 4 L0 8 Z" }))));
  const top = 2, bottom = ARROW_HEIGHT - 3, mid = ARROW_HEIGHT / 2;
  sources.forEach((source, j) => {
    const from = centre(source), to = centre(j);
    svg.append(s("path", { d: `M${from} ${top} C${from} ${mid} ${to} ${mid} ${to} ${bottom}`, "marker-end": "url(#cmp-head)" }));
  });
  return svg;
}

export default async function mount(el) {
  const random = rng(7);
  const state = {
    mask: [...START_MASK],
    view: el.dataset.view === TABLE ? TABLE : LOOP,
    pos: 0, // iterations of the loop done so far
  };
  let timer = null;

  const stage = h("div", { class: "w-stage" });
  const status = h("div", { class: "w-note cmp-status", "aria-live": "polite" });
  const first = stat("stores made");
  const kept = stat("elements returned", "good");
  const branches = stat("branches that depend on the data", "hot");

  const stop = () => {
    if (timer) clearInterval(timer);
    timer = null;
    playButton.textContent = "Play";
  };

  function maskRow() {
    return row("mask", state.mask.map((bit, i) => h("button", {
      type: "button", class: `cell ${bit ? "good" : ""}`.trim(), "data-mask-index": i,
      "aria-pressed": bit ? "true" : "false", "aria-label": `mask[${i}]`,
      onclick: () => setMask(state.mask.map((b, j) => (j === i ? 1 - b : b))),
    }, bit)));
  }

  function renderLoop(body) {
    const { out, k, last } = runLoop(state.mask, state.pos);
    const i = state.pos - 1; // the element handled by the latest step
    const done = state.pos === N;
    body.append(
      row("x", VALUES.map((v, j) => h("span", { class: `cell ${j === i ? "hot" : j < i ? "done" : ""}`.trim() }, v.toFixed(1)))),
      maskRow(),
      row("out", out.map((source, j) => {
        const cls = j < k ? "good" : source !== null ? "stray" : "empty";
        return h("span", { class: `cell ${cls} ${j === last && !done ? "store" : ""}`.trim() }, source === null ? "" : VALUES[source].toFixed(1));
      })),
      pointer(k, done ? `returned count = ${k}` : `k = ${k}`),
    );
    if (state.pos === 0) {
      status.innerHTML = "Each step runs the same two lines whatever the mask says: <code>out[k] = x[i]; k += mask[i];</code>";
    } else if (!done || state.mask[N - 1]) {
      const moved = state.mask[i];
      status.innerHTML = `<code>i = ${i}</code>: store <code>x[${i}] = ${VALUES[i].toFixed(1)}</code> at <code>out[${last}]</code>. `
        + (moved ? `<code>mask[${i}]</code> is 1, so <code>k</code> moves on to ${k}.`
          : `<code>mask[${i}]</code> is 0, so <code>k</code> stays at ${k}${done ? "." : ": the next store overwrites it."}`)
        + (done ? ` Done: ${k} element${k === 1 ? "" : "s"} returned.` : "");
    } else {
      status.innerHTML = `Done: ${k} element${k === 1 ? "" : "s"} returned. The last store put <code>${VALUES[N - 1].toFixed(1)}</code> at <code>out[${k}]</code> and nothing came after to overwrite it. It is scratch: stored, not returned.`;
    }
    first.set(`${state.pos} of ${N}`);
    kept.set(done ? String(k) : "–");
    stepButton.disabled = done;
    if (done) stop();
  }

  function renderTable(body) {
    const sources = state.mask.flatMap((bit, i) => (bit ? [i] : []));
    const count = sources.length;
    const number = state.mask.reduce((sum, bit, i) => sum + (bit << i), 0);
    const bits = [...state.mask].reverse().join("");
    body.append(
      maskRow(),
      h("div", { class: "cmp-number" }, "packed into one number: ", h("b", {}, `0b${bits} = ${number}`), "  (element 0 is the lowest bit)"),
      row("x", VALUES.map((v, j) => h("span", { class: `cell ${state.mask[j] ? "hot" : "dim"}`.trim() }, v.toFixed(1)))),
      arrows(sources),
      row(`row ${number}`, Array.from({ length: N }, (_, j) => h("span", { class: `cell ${j < count ? "hot" : "empty"}`.trim() }, j < count ? sources[j] : "·"))),
      row("out", Array.from({ length: N }, (_, j) => h("span", { class: `cell ${j < count ? "good" : "stray"}`.trim() }, j < count ? VALUES[sources[j]].toFixed(1) : "?"))),
      pointer(count, `write pointer advances by ${count}`),
    );
    status.innerHTML = `Row ${number} of the 256-row table says which source element goes in each output slot: the selected ones first, in order. `
      + `The whole block is permuted and stored in one go. ${count} slot${count === 1 ? " is" : "s are"} kept; the other ${N - count} hold rubbish that the next block's store overwrites.`;
    first.set("1 block");
    kept.set(String(count));
  }

  function render() {
    const focused = document.activeElement?.dataset?.maskIndex;
    const body = h("div", { class: "cmp" },
      row("", Array.from({ length: N }, (_, i) => h("span", { class: "cell" }, i)), "cmp-idx"));
    if (state.view === LOOP) renderLoop(body); else renderTable(body);
    stage.replaceChildren(body);
    // Not the `hidden` attribute: .w-controls sets `display`, which would override it.
    playback.style.display = state.view === LOOP ? "" : "none";
    branches.set("0");
    if (focused !== undefined) stage.querySelector(`[data-mask-index="${focused}"]`)?.focus();
  }

  // A new mask keeps the loop at its end if it had finished, so toggling a bit shows the new result at once.
  function setMask(mask) {
    stop();
    state.mask = mask;
    state.pos = state.pos === N ? N : 0;
    render();
  }

  const step = () => { state.pos += 1; render(); };
  const stepButton = button("Step", () => { stop(); step(); });
  // Play steps through on a timer whatever the reduced-motion setting: each
  // step is a redraw, not a movement, and the reader asked for it. From the
  // last step it starts again from the first.
  const playButton = button("Play", () => {
    if (timer) return stop();
    if (state.pos === N) { state.pos = 0; render(); }
    playButton.textContent = "Pause";
    timer = setInterval(step, STEP_MS);
  }, "primary");
  const resetButton = button("Reset", () => { stop(); state.pos = 0; render(); });
  const playback = h("div", { class: "w-controls bottom" }, stepButton, playButton, resetButton);

  el.append(
    h("div", { class: "w-controls" },
      seg([{ value: LOOP, label: "Branchless loop" }, { value: TABLE, label: "Table lookup" }], state.view,
        (v) => { stop(); state.view = v; state.pos = 0; render(); }),
      h("span", { class: "w-spacer" }),
      h("span", { class: "w-label" }, "Mask"),
      button("Random mask", () => setMask(state.mask.map(() => (random() < 0.5 ? 1 : 0)))),
      button("All", () => setMask(new Array(N).fill(1))),
      button("None", () => setMask(new Array(N).fill(0)))),
    stage,
    status,
    playback,
    h("div", { class: "w-readout" }, first, kept, branches),
    h("div", { class: "w-note" }, "Click a mask cell to flip it. compress_f32 does this for eight elements per step with a table of 256 patterns; the leftover slots are overwritten by the next block's store."),
  );
  render();
}
