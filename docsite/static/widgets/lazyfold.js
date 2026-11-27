// How a recording finds the steps of a loop and folds them.
//
// A graph of a loop is too large to read node by node, so a recording keeps
// each graph's operations a second time, as a tree of groups
// (docs/lazy-recording-design.md, "Groups"). The figure takes one loop through
// the three moves that build the tree: consecutive operations of one line
// become one member; members whose signatures come in the same order three
// times or more in a row become a repeat; a long repeat lists its ends, and a
// long call is cut into stretches. Then it shows the tree as the graph page
// opens it: a group of more than five operations folded.
//
// The program is the prototype's regression (prototypes/lazy/programs/
// regression.py) written out as one function, with one line added that can
// make one step differ from the others:
//
//    1  def fit(np, X, y, steps, odd=-1):
//    2      X, y = np.asarray(X), np.asarray(y)
//    3      X = (X - X.mean(axis=0)) / X.std(axis=0)
//    4      w, b = np.zeros(X.shape[1]), 0.0
//    5      losses = []
//    6      for i in range(steps):
//    7          residual = X @ w + b - y
//    8          losses.append((residual ** 2).mean() + 1e-3 * (w ** 2).sum())
//    9          gradient = 2.0 / len(y) * (X.T @ residual) + 2e-3 * w
//   10          if i == odd:
//   11              gradient = np.minimum(gradient, 1.0)
//   12          w = w - 0.05 * gradient
//   13          b = b - 0.1 * residual.mean()
//   14      return w, b, losses
//
// It was run on prototypes/lazy under record_graphs(), on 200 x 8 float64
// data, with every setting the figure offers (3, 12, 13, 50 and 200 steps;
// and 50 steps with odd = 1, 24 and 47), and gathered at the end: one graph
// each (NumPy 2.5.5, Apple M1 Pro, the prototype as of the fusion merge). The
// operations of each line below are the recording's. The tree is built here by
// the recording's own rules (prototypes/lazy/record.py, class _Tree), ported
// to JavaScript, and for every setting it was checked against the tree the
// recording gave: each group's kind, label and number of operations agree.
// A signature is the digest of an operation's name, element type, shape and
// line; the shapes of each line are the same on every step, so here an
// operation's signature is written as its name and its line.
//
// Attributes:
//   data-steps  the number of steps shown first: 3, 12, 13, 50 or 200 (default 50)
//   data-stage  the move shown first: ops, runs or repeats (default repeats)

import { h, seg, stat, button, fmtN } from "./util.js";

// The operations each line of the program wrote on one run, as the recording
// names them: [name, kind].
const LINES = {
  3: { code: "X = (X - X.mean(axis=0)) / X.std(axis=0)", ops: [["mean", "reduction"], ["subtract", "elementwise"], ["std", "reduction"], ["divide", "elementwise"]], tone: "var(--k-input)" },
  7: { code: "residual = X @ w + b - y", ops: [["matmul", "matmul"], ["add", "elementwise"], ["subtract", "elementwise"]], tone: "var(--g2)" },
  8: { code: "losses.append((residual ** 2).mean() + 1e-3 * (w ** 2).sum())", ops: [["power", "elementwise"], ["mean", "reduction"], ["power", "elementwise"], ["sum", "reduction"], ["multiply", "elementwise"], ["add", "elementwise"]], tone: "var(--k-reduction)" },
  9: { code: "gradient = 2.0 / len(y) * (X.T @ residual) + 2e-3 * w", ops: [["transpose", "view"], ["matmul", "matmul"], ["multiply", "elementwise"], ["multiply", "elementwise"], ["add", "elementwise"]], tone: "var(--k-matmul)" },
  11: { code: "gradient = np.minimum(gradient, 1.0)", ops: [["minimum", "elementwise"]], tone: "var(--k-atonce)" },
  12: { code: "w = w - 0.05 * gradient", ops: [["multiply", "elementwise"], ["subtract", "elementwise"]], tone: "var(--k-write)" },
  13: { code: "b = b - 0.1 * residual.mean()", ops: [["mean", "reduction"], ["multiply", "elementwise"], ["subtract", "elementwise"]], tone: "var(--k-view)" },
};

// The listing shown, with each line's number.
const LISTING = [
  [1, "def fit(np, X, y, steps, odd=-1):"],
  [2, "    X, y = np.asarray(X), np.asarray(y)"],
  [3, "    X = (X - X.mean(axis=0)) / X.std(axis=0)"],
  [4, "    w, b = np.zeros(X.shape[1]), 0.0"],
  [5, "    losses = []"],
  [6, "    for i in range(STEPS):"],
  [7, "        residual = X @ w + b - y"],
  [8, "        losses.append((residual ** 2).mean() + 1e-3 * (w ** 2).sum())"],
  [9, "        gradient = 2.0 / len(y) * (X.T @ residual) + 2e-3 * w"],
  [10, "        if i == ODD:"],
  [11, "            gradient = np.minimum(gradient, 1.0)"],
  [12, "        w = w - 0.05 * gradient"],
  [13, "        b = b - 0.1 * residual.mean()"],
  [14, "    return w, b, losses"],
];

// The recording's constants (prototypes/lazy/record.py).
const FOLD_ABOVE = 5;            // a group of more operations opens folded
const MIN_REPEATS = 3;           // a run must come this many times in a row
const MAX_PERIOD = 64;           // the longest run looked for, in members
const SHOWN_STEPS = 12, HEAD_STEPS = 2, TAIL_STEPS = 1;
const REST_BLOCK = 50;           // most members of a group of left-out steps
const MOST_MEMBERS = 16, BLOCK_MEMBERS = 8;

// The settings that were run on the prototype: steps, and which step differs.
const STEP_CHOICES = [3, 12, 13, 50, 200];
const ODD_CHOICES = [{ value: -1, label: "none" }, { value: 1, label: "the 2nd" }, { value: 24, label: "the 25th" }, { value: 47, label: "the 48th" }];
// Only the 50-step loop was run with a step that differs.
const ODD_STEPS = 50;

// What each setting shows, in words.
const NOTES = {
  "3/-1": "Three steps are the fewest that make a repeat: the search asks for a run of members that comes at least three times in a row. Each step is five members, one per line of the loop's body, and 19 operations. With twelve steps or fewer every step is listed.",
  "12/-1": "Twelve steps, all listed. Each is 19 operations, more than five, so each opens folded, and so does the repeat.",
  "13/-1": "From thirteen steps on, a repeat lists its first two steps and its last, and holds the others in one group, here steps 3 to 12. Unfolding the loop shows four boxes, whatever its length.",
  "50/-1": "The regression of this chapter: 954 operations. The page opens on the four operations of line 3 and one box, × 50. Unfolding the box gives step 1, step 2, steps 3 to 49 and step 50; unfolding a step gives its five lines; unfolding a line gives its operations.",
  "200/-1": "A long group of left-out steps is itself split, into groups of at most fifty, so that unfolding one never puts thousands of boxes on the page: steps 3 to 52, 53 to 102, 103 to 152 and 153 to 199.",
  "50/1": "One extra operation in the second step. The search is greedy from the left and only asks that runs be alike, so the repeat it finds begins in the middle of the second step, at line 12, and each of its 48 steps is lines 12, 13, 7, 8 and 9. A step of the tree is not always a turn of the loop. Nothing is lost by that: every operation is still in the graph, and the tree only says how to fold it.",
  "50/24": "The 25th step has one operation more. The loop is cut where it differs: 24 steps alike, then the first lines of the 25th by themselves (7, 8, 9 and the extra 11), and a second repeat of 25 that, like the 2nd-step case, begins at line 12 and ends with the 50th step's line 9.",
  "50/47": "The 48th step differs, and what follows it is two steps: too few for a repeat. fit() is left with 18 members, more than 16, so it is cut into stretches of eight members that follow each other, named by the lines they cover. This is the rule that keeps a long function of lines that are not alike from opening as a hundred boxes.",
};

export const css = `
.lf-code { margin: 0 0 .9rem; padding: .5rem 0; background: var(--code-bg); border-radius: 10px; font-family: var(--mono); font-size: .74rem; line-height: 1.65; overflow-x: auto; }
.lf-code div { display: flex; gap: .6rem; padding: 0 .8rem; white-space: pre; }
.lf-code .ln { min-width: 1.4rem; text-align: center; border-radius: 4px; color: var(--muted); user-select: none; font-weight: 650; }
.lf-code .ln.has { color: var(--bg-raised); }
.lf-code .off { color: var(--muted); text-decoration: line-through; text-decoration-color: var(--rule-strong); }
.lf-title { font-size: .7rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: .2rem 0 .4rem; }
.lf-strip { display: flex; flex-wrap: wrap; gap: 2px; align-items: flex-end; max-height: 17rem; overflow-y: auto; padding: 2px; }
.lf-op { width: 6px; height: 10px; border-radius: 1.5px; }
.lf-m { min-width: 1.15rem; height: 1.05rem; padding: 0 2px; border-radius: 3px; font: 650 .62rem/1.05rem var(--mono); text-align: center; color: var(--bg-raised); }
.lf-m.small { min-width: 7px; width: 7px; padding: 0; font-size: 0; }
.lf-rep { display: inline-flex; flex-wrap: wrap; gap: 2px 5px; align-items: center; padding: 3px 4px; border: 1.5px solid var(--hot); border-radius: 6px; background: var(--hot-soft); }
.lf-rep > b { font: 700 .7rem var(--sans); color: var(--hot); margin-right: 2px; }
.lf-step { display: inline-flex; gap: 2px; padding: 2px; border-radius: 4px; }
.lf-step:nth-of-type(even) { background: var(--bg-raised); }
.lf-tree { margin-top: .9rem; border: 1px solid var(--rule); border-radius: 10px; padding: .5rem .6rem; font-size: .8rem; max-height: 22rem; overflow: auto; }
.lf-row { display: flex; align-items: baseline; gap: .45rem; padding: .12rem .2rem; border-radius: 5px; white-space: nowrap; }
.lf-row.grp { cursor: pointer; }
.lf-row.grp:hover { background: var(--panel); }
.lf-row .tw { width: .8rem; color: var(--muted); font-size: .7rem; }
.lf-row .kind { font: 650 .62rem var(--sans); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); min-width: 3.1rem; }
.lf-row .lab { font-family: var(--mono); font-size: .74rem; overflow: hidden; text-overflow: ellipsis; }
.lf-row .n { margin-left: auto; padding-left: .8rem; color: var(--muted); font-variant-numeric: tabular-nums; font-size: .74rem; }
.lf-row.folded .lab { font-weight: 650; }
.lf-row.folded .n { color: var(--hot); font-weight: 650; }
.lf-kid { padding-left: 1.05rem; border-left: 1px solid var(--rule); margin-left: .45rem; }
.lf-opchip { display: inline-block; padding: 0 .35rem; border-radius: 4px; border: 1.5px solid; font: 600 .68rem/1.25rem var(--mono); margin: .1rem .2rem .1rem 0; }
.lf-ops { padding: .1rem 0 .15rem 1.3rem; }
.lf-legend { display: flex; flex-wrap: wrap; gap: .25rem .9rem; margin-top: .45rem; font-size: .72rem; color: var(--muted); }
.lf-legend i { display: inline-block; width: .75rem; height: .75rem; border-radius: 3px; vertical-align: -1px; margin-right: .3rem; }
`;

// ------------------------------------------------------------ the rules, ported

// An operation: its name, kind and line, and its signature.
const op = (name, kind, line) => ({ op: true, name, kind, line, sig: `${name}@${line}`, n: 1 });

// A group, with its operations counted and its signature worked out as
// record.py's _Tree.group does: kind, a call's name, its lines, its number of
// steps and its members' signatures; a step's label does not count.
function group(k, l, c, lines = [0, 0], repeats = 0, line = 0) {
  const n = c.reduce((sum, m) => sum + m.n, 0);
  const sig = JSON.stringify([k, k === "call" ? l : "", lines, repeats, c.map((m) => m.sig)]);
  return { k, l, c, n, lines, repeats, sig, line };
}

// The first and last line some members cover.
function span(members) {
  const first = Math.min(...members.map((m) => (m.op ? m.line : m.lines[0])));
  const last = Math.max(...members.map((m) => (m.op ? m.line : m.lines[1])));
  return [first, last];
}

// Whether two runs of `period` members are alike, member by member.
function sameRun(sigs, first, second, period) {
  if (second + period > sigs.length) return false;
  for (let i = 0; i < period; i++) if (sigs[first + i] !== sigs[second + i]) return false;
  return true;
}

// The repeat starting at `start` that covers the most members: [times, period].
function longestRepeat(sigs, start) {
  let best = [1, 1];
  const longest = Math.min(MAX_PERIOD, Math.floor((sigs.length - start) / MIN_REPEATS));
  for (let period = 1; period <= longest; period++) {
    let repeats = 1;
    while (sameRun(sigs, start, start + repeats * period, period)) repeats++;
    if (repeats >= MIN_REPEATS && repeats * period > best[0] * best[1]) best = [repeats, period];
  }
  return best;
}

// The group of the steps a long repeat does not list singly.
function rest(steps, first) {
  const label = `steps ${first} to ${first + steps.length - 1}`;
  if (steps.length <= REST_BLOCK) return group("rest", label, steps, span(steps));
  let size = REST_BLOCK;
  while (steps.length > size * REST_BLOCK) size *= REST_BLOCK;
  const blocks = [];
  for (let start = 0; start < steps.length; start += size) blocks.push(rest(steps.slice(start, start + size), first + start));
  return group("rest", label, blocks, span(steps));
}

// The repeat group of some steps that are alike.
function repeat(steps) {
  const count = steps.length;
  let listed = steps.map((step, place) => (step.length === 1 ? step[0] : group("step", `step ${place + 1} of ${count}`, step, span(step))));
  if (count > SHOWN_STEPS) {
    listed = [...listed.slice(0, HEAD_STEPS), rest(listed.slice(HEAD_STEPS, count - TAIL_STEPS), HEAD_STEPS + 1), ...listed.slice(count - TAIL_STEPS)];
  }
  return group("repeat", `× ${count}`, listed, span(steps[0]), count);
}

// Members, with each run of members that are alike replaced by a repeat. Greedy from the left.
function repeats(members) {
  const sigs = members.map((m) => m.sig);
  const found = [];
  let start = 0;
  while (start < members.length) {
    const [times, period] = longestRepeat(sigs, start);
    if (times < MIN_REPEATS) { found.push(members[start]); start++; continue; }
    const steps = [];
    for (let step = 0; step < times; step++) steps.push(members.slice(start + step * period, start + (step + 1) * period));
    found.push(repeat(steps));
    start += times * period;
  }
  return found;
}

// A long list of members, cut into stretches that follow each other.
function stretches(members) {
  if (members.length <= MOST_MEMBERS) return members;
  let size = BLOCK_MEMBERS;
  while (members.length > size * MOST_MEMBERS) size *= BLOCK_MEMBERS;
  const out = [];
  for (let start = 0; start < members.length; start += size) {
    const part = members.slice(start, start + size);
    if (part.length === 1) { out.push(part[0]); continue; }
    const lines = span(part);
    out.push(group("lines", `lines ${lines[0]} to ${lines[1]}`, stretches(part), lines));
  }
  return out;
}

// One run of one line: its operations, or a line group of them.
function lineRun(line) {
  const ops = LINES[line].ops.map(([name, kind]) => op(name, kind, line));
  if (ops.length === 1) return ops[0];
  const inner = repeats(ops);
  return group("line", LINES[line].code, inner, [line, line], 0, line);
}

// The operations of fit(), in the order written, as runs of a line.
export function runsOf(steps, odd) {
  const members = [lineRun(3)];
  for (let i = 0; i < steps; i++) {
    for (const line of [7, 8, 9]) members.push(lineRun(line));
    if (i === odd) members.push(lineRun(11));
    for (const line of [12, 13]) members.push(lineRun(line));
  }
  return members;
}

// The whole tree of the graph, as the recording builds it.
export function treeOf(steps, odd) {
  const runs = runsOf(steps, odd);
  const call = group("call", "fit()", stretches(repeats(runs)), span(runs));
  return group("block", "the recorded block", [call]);
}

// ------------------------------------------------------------ the figure

// The groups that are open when the page first shows a tree: none of more
// than FOLD_ABOVE operations, except that while the top holds one folded
// group and nothing else, that group is opened.
function initialOpen(tree) {
  const open = new Set();
  const walk = (g) => {
    if (g.op) return;
    if (g.n <= FOLD_ABOVE) open.add(g);
    g.c.forEach(walk);
  };
  walk(tree);
  let top = tree;
  open.add(top);
  while (top.c.length === 1 && !top.c[0].op && !open.has(top.c[0])) { top = top.c[0]; open.add(top); }
  return open;
}

// What the page draws at first: open frames, operations and folded boxes.
function drawn(tree, open) {
  const count = { frames: 0, ops: 0, folded: 0 };
  const walk = (g, root) => {
    if (g.op) { count.ops++; return; }
    if (!open.has(g)) { count.folded++; return; }
    if (!root) count.frames++;
    g.c.forEach((m) => walk(m, false));
  };
  walk(tree, true);
  return count;
}

function chip(member, small) {
  const line = member.op ? member.line : member.line || member.lines[0];
  return h("span", {
    class: `lf-m${small ? " small" : ""}`, style: `background: ${LINES[line].tone}`,
    title: `line ${line}: ${LINES[line].code}, ${member.n} operation${member.n === 1 ? "" : "s"}`,
  }, small ? "" : String(line));
}

export default async function mount(el) {
  let steps = STEP_CHOICES.includes(Number(el.dataset.steps)) ? Number(el.dataset.steps) : 50;
  let odd = -1;
  let stage = ["ops", "runs", "repeats"].includes(el.dataset.stage) ? el.dataset.stage : "repeats";
  let tree, open;

  const code = h("pre", { class: "lf-code", "aria-label": "The program" });
  const title = h("p", { class: "lf-title" });
  const strip = h("div", { class: "lf-strip" });
  const legend = h("div", { class: "lf-legend" });
  const treeBox = h("div", { class: "lf-tree", role: "tree", "aria-label": "The tree of groups, as the page opens it" });
  const note = h("p", { class: "w-note", "aria-live": "polite" });
  const sOps = stat("operations in the graph", "hot");
  const sRuns = stat("runs of a line in fit()");
  const sFound = stat("what the search found");
  const sDrawn = stat("drawn as the tree is now");

  const oddSeg = seg(ODD_CHOICES, odd, (v) => { odd = Number(v); rebuild(); });

  function renderCode() {
    code.replaceChildren(...LISTING.map(([n, text]) => {
      let shown = text.replace("STEPS", String(steps)).replace("ODD", odd < 0 ? "-1" : String(odd));
      const off = odd < 0 && (n === 10 || n === 11);
      const has = LINES[n] && !off;
      return h("div", {},
        h("span", { class: `ln${has ? " has" : ""}`, style: has ? `background: ${LINES[n].tone}` : null }, n),
        h("span", { class: off ? "off" : "" }, shown));
    }));
  }

  function renderStrip() {
    const runs = runsOf(steps, odd);
    const small = runs.length > 300;
    if (stage === "ops") {
      const ops = runs.flatMap((m) => (m.op ? [m] : flatten(m)));
      title.textContent = `${fmtN(ops.length)} operations, in the order they were written, coloured by kind`;
      strip.replaceChildren(...ops.map((o) => h("span", { class: "lf-op", style: `background: var(--k-${o.kind})`, title: `${o.name}, line ${o.line}` })));
      legend.replaceChildren(...[["elementwise", "element by element"], ["reduction", "reduction"], ["matmul", "matrix product"], ["view", "view"]]
        .map(([k, label]) => h("span", {}, h("i", { style: `background: var(--k-${k})` }), label)));
    } else if (stage === "runs") {
      title.textContent = `${fmtN(runs.length)} runs: consecutive operations of one line are one member of fit(), labelled by its line`;
      strip.replaceChildren(...runs.map((m) => chip(m, small)));
      legend.replaceChildren(...lineLegend());
    } else {
      const found = repeats(runs);
      title.textContent = "The members after the search: each run that comes three times or more in a row is one repeat";
      strip.replaceChildren(...found.map((m) => {
        if (m.k !== "repeat") return chip(m, small);
        const stepsOf = expandSteps(m);
        return h("span", { class: "lf-rep", title: `${m.l}: ${m.n} operations` }, h("b", {}, m.l),
          ...stepsOf.map((step) => h("span", { class: "lf-step" }, ...step.map((x) => chip(x, small)))));
      }));
      legend.replaceChildren(...lineLegend());
    }
  }

  function renderTree() {
    const row = (g) => {
      if (g.op) return h("span", { class: "lf-opchip", style: `border-color: var(--k-${g.kind}); background: var(--k-${g.kind}-soft); color: var(--k-${g.kind})` }, g.name);
      const isOpen = open.has(g);
      const head = h("div", {
        class: `lf-row grp${isOpen ? "" : " folded"}`, role: "treeitem", "aria-expanded": String(isOpen), tabindex: "0",
        onclick: () => { if (isOpen) open.delete(g); else open.add(g); renderTree(); },
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.currentTarget.click(); } },
      }, h("span", { class: "tw" }, isOpen ? "▾" : "▸"), h("span", { class: "kind" }, g.k), h("span", { class: "lab", title: g.l }, g.l),
        h("span", { class: "n" }, `${fmtN(g.n)} op${g.n === 1 ? "" : "s"}`));
      if (!isOpen) return head;
      // A group holds operations or groups, or both, in the order written.
      const kids = g.c.every((m) => m.op)
        ? [h("div", { class: "lf-ops" }, ...g.c.map(row))]
        : g.c.map((m) => (m.op ? h("div", { class: "lf-ops" }, row(m)) : row(m)));
      return h("div", {}, head, h("div", { class: "lf-kid", role: "group" }, ...kids));
    };
    treeBox.replaceChildren(row(tree));
    const d = drawn(tree, open);
    sDrawn.set(`${d.ops} ops, ${d.folded} box${d.folded === 1 ? "" : "es"}`);
  }

  function rebuild() {
    if (steps !== ODD_STEPS && odd !== -1) { odd = -1; oddSeg.set(-1); }
    oddSeg.querySelectorAll("button").forEach((b) => { b.disabled = steps !== ODD_STEPS && b.dataset.value !== "-1"; });
    tree = treeOf(steps, odd);
    open = initialOpen(tree);
    const found = repeats(runsOf(steps, odd)).filter((m) => m.k === "repeat");
    sOps.set(fmtN(tree.n));
    sRuns.set(fmtN(runsOf(steps, odd).length));
    sFound.set(found.length ? found.map((r) => r.l).join(", ") : "no repeat");
    note.textContent = NOTES[`${steps}/${odd}`];
    renderCode();
    renderStrip();
    renderTree();
  }

  el.append(
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Steps"),
      seg(STEP_CHOICES.map((v) => ({ value: v, label: String(v) })), steps, (v) => { steps = Number(v); rebuild(); }),
      h("span", { class: "w-label" }, "A step with one more operation"),
      oddSeg),
    code,
    h("div", { class: "w-controls" },
      h("span", { class: "w-label" }, "Show"),
      seg([{ value: "ops", label: "1 · operations" }, { value: "runs", label: "2 · runs of a line" }, { value: "repeats", label: "3 · repeats found" }], stage, (v) => { stage = v; renderStrip(); })),
    title, strip, legend,
    h("div", { class: "w-controls bottom" },
      h("span", { class: "w-label" }, "The tree, as the page opens it. Click a group to unfold it"),
      button("Fold", () => { open = initialOpen(tree); renderTree(); })),
    treeBox,
    note,
    h("div", { class: "w-readout" }, sOps, sRuns, sFound, sDrawn),
  );
  rebuild();
}

// The operations under a group, in order.
function flatten(g) {
  return g.c.flatMap((m) => (m.op ? [m] : flatten(m)));
}

// The steps of a repeat, each as its list of members, with left-out steps put back.
function expandSteps(r) {
  const out = [];
  const walk = (m) => {
    if (m.k === "rest") { m.c.forEach(walk); return; }
    out.push(m.k === "step" ? m.c : [m]);
  };
  r.c.forEach(walk);
  return out;
}

// The legend of line colours.
function lineLegend() {
  return Object.entries(LINES).map(([n, line]) => h("span", {}, h("i", { style: `background: ${line.tone}` }), `line ${n}`));
}
