// The workloads over time: how much faster they run with hotkernels, run by
// run, with the merges that changed the package marked on the time axis.
//
// Data: data/history.json, which the build reads from the perf-history branch
// (delivery/history.py). Runs are placed at the date and time they started,
// a backfill run at the date of the commit it timed,
// and the line goes through every run. Runs of one commit on several nights
// are enclosed in a band that spans their dates and their values: how far a
// figure moves by chance. A change of Python, NumPy or a library breaks a
// line, as does a change of the set of workloads in the headline.
// Attributes: data-repo  the repository's URL, to link a change to its pull request.
// Pointing at a dot, or tabbing to it, fills the table under the chart with that run.

import { h, s, loadData, seg, fmtX, fmtNs, fmtN, scale } from "./util.js";

// Drawing areas, in viewBox units: the headline chart and one small chart per workload.
const BIG = { W: 680, H: 320, LEFT: 52, RIGHT: 20, TOP: 34, BOTTOM: 40, dot: 4.5 };
const SMALL = { W: 340, H: 190, LEFT: 44, RIGHT: 12, TOP: 14, BOTTOM: 30, dot: 3.2 };
// One colour per machine, in the order they first recorded a run.
const MACHINE_COLORS = ["var(--hot)", "var(--np)", "var(--cc)", "var(--good)"];
// One colour per size in the small charts.
const SIZE_COLORS = { S: "var(--cc)", M: "var(--hot)", L: "var(--np)" };
const DAY_MS = 86400000;
// Steps of the date axis, in days; the first that keeps the ticks at MAX_DATE_TICKS or fewer is used.
const DATE_STEPS = [1, 2, 7, 14, 28, 56, 91, 182, 364];
const MAX_DATE_TICKS = 7;
// Rows of change labels above the headline, so that labels of close merges do not overlap.
const LABEL_ROWS = 2;
// Merges closer than this on the time axis, in viewBox units, share one marker.
const MERGE_GAP = 14;

export const css = `
.hs-line { fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round; }
.hs-band { opacity: .2; }
.hs-dot { stroke-width: 1.6; cursor: pointer; outline: none; }
.hs-dot:hover, .hs-dot:focus { stroke: var(--ink); stroke-width: 2.2; }
.hs-dot.on { stroke: var(--ink); stroke-width: 2.4; }
.hs-axis text { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.hs-axis line { stroke: var(--rule); }
.hs-one { stroke: var(--rule-strong); stroke-width: 1.5; }
.hs-change line { stroke: var(--muted); stroke-dasharray: 3 4; }
.hs-change text { font-size: 10.5px; fill: var(--muted); cursor: pointer; }
.hs-change:hover line, .hs-change:focus line { stroke: var(--ink); }
.hs-change:hover text, .hs-change:focus text { fill: var(--ink); }
.hs-change { outline: none; }
.hs-read { min-height: 1.3em; }
.hs-notes { margin: .3rem 0 0; padding-left: 1.2rem; }
.hs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr)); gap: 1.2rem 1.4rem; margin-top: 1.4rem; }
.hs-grid h4 { font-size: .84rem; margin: 0 0 .3rem; font-weight: 650; }
.hs-grid h4 small { color: var(--muted); font-weight: 500; }
.hs-table { overflow-x: auto; margin-top: .9rem; }
.hs-table th button { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer; text-transform: inherit; letter-spacing: inherit; }
.hs-table th button:hover { color: var(--ink); }
.hs-table td.differ { color: var(--bad); }
.w-stage > svg.hs-big { min-width: 27rem; }
`;

// "19 Nov" in UTC, with the year when the axis spans more than one.
function fmtDate(ms, withYear = false) {
  return new Date(ms).toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC", ...(withYear ? { year: "numeric" } : {}) });
}

// "19 Nov 2026, 02:00 UTC"
function fmtWhen(iso) {
  const d = new Date(iso);
  const time = d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
  return `${fmtDate(d.getTime(), true)}, ${time} UTC`;
}

// Round steps for a linear axis from lo to hi, neither of them zero necessarily.
function ticks(lo, hi) {
  const raw = (hi - lo) / 5;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((m) => m >= raw);
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(Number(t.toPrecision(12)));
  return out;
}

// Midnights (UTC) between lo and hi at a step that leaves at most MAX_DATE_TICKS of them.
function dateTicks(lo, hi) {
  const step = DATE_STEPS.find((d) => (hi - lo) / (d * DAY_MS) <= MAX_DATE_TICKS) || DATE_STEPS[DATE_STEPS.length - 1];
  const out = [];
  for (let t = Math.ceil(lo / DAY_MS) * DAY_MS; t <= hi; t += step * DAY_MS) out.push(t);
  return out;
}

// Where a run's dot sits on the time axis, in milliseconds: when it started, or for a
// backfill run the date of the commit it timed (Run.placed_at in delivery/history.py).
function runTime(sr, id) {
  return Date.parse(sr.runs[id].placed_at);
}

// The time span shown: every run, with a margin; at least a day on each side so that one run sits in the middle.
function dateDomain(series) {
  const dates = series.flatMap((sr) => sr.groups.flatMap((g) => g.runs.map(([id]) => runTime(sr, id))));
  let lo = Math.min(...dates), hi = Math.max(...dates);
  const pad = Math.max(DAY_MS, (hi - lo) * 0.04);
  return [lo - pad, hi + pad];
}

// Text of a change's note without its markup, for a tooltip.
function plain(html) {
  const div = document.createElement("div");
  div.innerHTML = html;
  return div.textContent;
}

// Merges whose markers would overlap, as lists: [[change, ...], ...], in date order.
function clusters(changes, x) {
  const out = [];
  for (const c of changes.slice().sort((a, b) => Date.parse(a.date) - Date.parse(b.date))) {
    const last = out[out.length - 1];
    if (last && x(Date.parse(c.date)) - x(Date.parse(last[last.length - 1].date)) < MERGE_GAP) last.push(c);
    else out.push([c]);
  }
  return out;
}

// Draws one chart. series: [{label, color, groups}]; changes are drawn as dashed lines
// when given; onRun(id) and onChange([change, ...]) are called when a dot or a marker is chosen.
function chart(box, series, { changes = [], labels = false, selected = null, onRun, onChange, aria }) {
  const { W, H, LEFT, RIGHT, TOP, BOTTOM, dot } = box;
  const [d0, d1] = dateDomain(series);
  const x = scale([d0, d1], [LEFT, W - RIGHT]);
  const values = series.flatMap((sr) => sr.groups.flatMap((g) => g.runs.map((r) => r[1])));
  const lo = Math.min(1, ...values), hi = Math.max(1, ...values);
  const span = Math.max(hi - lo, 0.1);
  const yLo = Math.max(0, lo - span * 0.12), yHi = hi + span * 0.12;
  const y = scale([yLo, yHi], [H - BOTTOM, TOP]);

  const svg = s("svg", { class: box === BIG ? "hs-chart hs-big" : "hs-chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": aria });
  const axis = s("g", { class: "hs-axis" });
  for (const t of ticks(yLo, yHi)) {
    axis.append(s("line", { x1: LEFT, x2: W - RIGHT, y1: y(t), y2: y(t) }),
      s("text", { x: LEFT - 7, y: y(t) + 4, "text-anchor": "end" }, `${t}x`));
  }
  const withYear = new Date(d0).getUTCFullYear() !== new Date(d1).getUTCFullYear();
  for (const t of dateTicks(d0, d1)) {
    axis.append(s("line", { x1: x(t), x2: x(t), y1: H - BOTTOM, y2: H - BOTTOM + 5 }),
      s("text", { x: x(t), y: H - BOTTOM + 18, "text-anchor": "middle" }, fmtDate(t, withYear)));
  }
  if (box === BIG) {
    axis.append(s("text", { x: LEFT, y: TOP - 14, "text-anchor": "start" }, "faster than NumPy, by"),
      s("text", { x: (LEFT + W - RIGHT) / 2, y: H - 4, "text-anchor": "middle" }, "date of the run"));
  }
  svg.append(axis, s("line", { class: "hs-one", x1: LEFT, x2: W - RIGHT, y1: y(1), y2: y(1) }));

  const shown = changes.filter((c) => Date.parse(c.date) > d0 && Date.parse(c.date) < d1);
  clusters(shown, x).forEach((group, i) => {
    const cx = x(Date.parse(group[0].date));
    const mark = s("g", { class: "hs-change", tabindex: labels ? 0 : null, onclick: () => onChange(group), onfocus: () => onChange(group) },
      s("title", {}, group.map((c) => `#${c.pull_request} (${c.kind}): ${plain(c.note)}`).join("\n")),
      s("line", { x1: cx, x2: cx, y1: TOP, y2: H - BOTTOM }));
    const label = group.length === 1 ? `#${group[0].pull_request}` : `${group.length} merges`;
    if (labels) mark.append(s("text", { x: cx + 4, y: TOP + 10 + (i % LABEL_ROWS) * 13 }, label));
    svg.append(mark);
  });

  for (const sr of series) {
    // Bands first, so that the lines are drawn over them. A band encloses the dots of one commit's runs.
    for (const g of sr.groups.filter((g) => g.runs.length > 1)) {
      const ts = g.runs.map(([id]) => x(runTime(sr, id))), vs = g.runs.map((r) => y(r[1]));
      const x0 = Math.min(...ts) - dot, y0 = Math.min(...vs) - dot;
      svg.append(s("rect", { class: "hs-band", x: x0, y: y0, width: Math.max(...ts) + dot - x0, height: Math.max(...vs) + dot - y0,
        rx: dot, style: `fill:${sr.color}` }));
    }
    // A break starts a new piece of line, so that two runs that cannot be compared are not joined.
    const d = sr.groups.flatMap((g, i) => g.runs.map(([id, value], j) =>
      `${(i === 0 || g.break) && j === 0 ? "M" : "L"}${x(runTime(sr, id)).toFixed(1)} ${y(value).toFixed(1)}`)).join(" ");
    svg.append(s("path", { class: "hs-line", d, style: `stroke:${sr.color}` }));
  }
  // Dots go on top of every line. A busy run is drawn hollow.
  for (const sr of series) {
    for (const g of sr.groups) {
      g.runs.forEach(([id, value]) => {
        const run = sr.runs[id];
        const cx = x(runTime(sr, id));
        svg.append(s("circle", {
          class: `hs-dot${id === selected ? " on" : ""}`, cx, cy: y(value), r: dot, tabindex: box === BIG ? 0 : null,
          style: `fill:${run.busy ? "var(--bg-raised)" : sr.color};stroke:${run.busy ? sr.color : "var(--bg-raised)"}`,
          onclick: () => onRun(id), onfocus: () => onRun(id), onmouseenter: () => onRun(id, true),
        }, s("title", {}, `${sr.label}, commit ${g.commit}, run of ${fmtWhen(run.recorded_at)}: ${fmtX(value)}${run.busy ? ", machine busy" : ""}`)));
      });
    }
  }
  return svg;
}

// The night's table: one row per workload and size of a run, sortable by any column.
function runTable(run, sortState, onSort) {
  const columns = [
    { key: "workload", label: "Workload" },
    { key: "size", label: "Size" },
    { key: "numpy_ns", label: "NumPy", num: true, fmt: (r) => (r.identical ? fmtNs(r.numpy_ns) : "–") },
    { key: "hotkernels_ns", label: "hotkernels", num: true, fmt: (r) => (r.identical ? fmtNs(r.hotkernels_ns) : "–") },
    { key: "speedup", label: "Faster by", num: true, fmt: (r) => (r.identical ? fmtX(r.speedup) : "results differ") },
    { key: "calls", label: "Kernel calls", num: true, fmt: (r) => fmtN(r.calls) },
    { key: "left", label: "Left to NumPy", num: true, fmt: (r) => fmtN(r.left) },
  ];
  const rows = run.rows.slice();
  if (sortState.key) {
    const sign = sortState.dir === "ascending" ? 1 : -1;
    rows.sort((a, b) => (a[sortState.key] < b[sortState.key] ? -sign : a[sortState.key] > b[sortState.key] ? sign : 0));
  }
  return h("div", { class: "hs-table" }, h("table", {},
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", {
      class: c.num ? "num" : null, "aria-sort": sortState.key === c.key ? sortState.dir : null, scope: "col",
    }, h("button", { type: "button", onclick: () => onSort(c.key) }, c.label))))),
    h("tbody", {}, rows.map((r) => h("tr", {}, columns.map((c) => h("td", {
      class: [c.num ? "num" : "", c.key === "speedup" && !r.identical ? "differ" : ""].join(" ").trim() || null,
    }, c.fmt ? c.fmt(r) : r[c.key])))))));
}

export default async function mount(el) {
  const data = await loadData("history");
  const repo = el.dataset.repo || "";
  if (!data.runs.length) {
    el.append(h("p", { class: "w-note" }, "No runs have been recorded yet. The chart fills in once the nightly runs start."));
    return;
  }
  const runs = data.runs;
  const cpu = Object.fromEntries(data.machines.map((m) => [m.slug, m.cpu]));
  const color = Object.fromEntries(data.machines.map((m, i) => [m.slug, MACHINE_COLORS[i % MACHINE_COLORS.length]]));
  const withRuns = (sr, label, c) => ({ ...sr, label, color: c, runs });
  const headline = data.series.filter((sr) => sr.workload === null)
    .map((sr) => withRuns(sr, cpu[sr.machine], color[sr.machine]));

  let selected = runs[runs.length - 1].id;
  let machine = runs[selected].machine;
  let marks = "on";
  let sortState = { key: null, dir: "ascending" };

  const stage = h("div", { class: "w-stage" });
  const read = h("p", { class: "w-note hs-read", "aria-live": "polite" });
  const table = h("div", {});
  const grid = h("div", { class: "hs-grid" });
  const machineSeg = data.machines.length > 1
    ? seg(data.machines.map((m) => ({ value: m.slug, label: m.cpu })), machine, (v) => { machine = v; drawSmall(); })
    : null;

  function showRun(id, hoverOnly = false) {
    const run = runs[id];
    const head = headline.find((sr) => sr.machine === run.machine);
    const value = head && head.groups.flatMap((g) => g.runs).find((r) => r[0] === id);
    read.replaceChildren(h("b", {}, cpu[run.machine]), `, commit ${run.commit}, run of ${fmtWhen(run.recorded_at)}: `,
      value ? h("b", {}, `${fmtX(value[1])} `) : "", value ? `geometric mean at size ${data.headline_size}. ` : "",
      `NumPy ${run.numpy}, Python ${run.python}, hotkernels ${run.hotkernels}. Load average ${run.load_average} at the start, ${run.load_average_end} at the end${run.busy ? ": the machine was in use" : ""}.`
      + (run.source !== "nightly" ? ` A ${run.source} run.` : ""));
    if (hoverOnly) return;
    selected = id;
    if (run.machine !== machine) { machine = run.machine; if (machineSeg) machineSeg.set(machine); }
    drawAll();
  }

  function showChange(group) {
    const link = (c) => (repo ? h("a", { href: `${repo}/pull/${c.pull_request}` }, `#${c.pull_request}`) : `#${c.pull_request}`);
    const line = (c) => [link(c), ` (${c.kind}), merged ${fmtWhen(c.date)}: `, h("span", { html: c.note })];
    read.replaceChildren(...(group.length === 1 ? line(group[0])
      : [h("b", {}, `${group.length} merges with a release note:`), h("ul", { class: "hs-notes" }, group.map((c) => h("li", {}, line(c))))]));
  }

  function drawBig() {
    stage.replaceChildren(chart(BIG, headline, {
      changes: marks === "on" ? data.changes : [], labels: true, selected, onRun: showRun, onChange: showChange,
      aria: `Geometric mean of the workloads' speedups at size ${data.headline_size} against date, per machine`,
    }));
  }

  function drawSmall() {
    grid.replaceChildren(...data.workloads.map((w) => {
      const series = data.series.filter((sr) => sr.workload === w.name && sr.machine === machine)
        .map((sr) => withRuns(sr, `${w.name} at ${sr.size}`, SIZE_COLORS[sr.size] || "var(--ink)"));
      if (!series.length) return null;
      return h("div", {}, h("h4", {}, w.name, w.ours ? h("small", {}, " (ours)") : ""),
        chart(SMALL, series, { changes: marks === "on" ? data.changes : [], selected, onRun: showRun, onChange: showChange,
          aria: `${w.name}: speedup against date at sizes ${series.map((sr) => sr.size).join(" and ")}` }));
    }).filter(Boolean));
  }

  function drawTable() {
    table.replaceChildren(runTable(runs[selected], sortState, (key) => {
      sortState = { key, dir: sortState.key === key && sortState.dir === "ascending" ? "descending" : "ascending" };
      drawTable();
    }));
  }

  function drawAll() { drawBig(); drawSmall(); drawTable(); }

  const sizes = [...new Set(data.series.filter((sr) => sr.workload).map((sr) => sr.size))];
  // The DOM's own append writes a null child as the text "null", so absent parts are left out first.
  el.append(...[
    h("div", { class: "w-controls" },
      seg([{ value: "on", label: "Mark merges" }, { value: "off", label: "No marks" }], marks, (v) => { marks = v; drawBig(); drawSmall(); })),
    stage,
    h("div", { class: "w-legend" },
      headline.map((sr) => h("span", {}, h("i", { style: `background:${sr.color}` }), sr.label)),
      h("span", {}, "hollow dot: machine in use"), h("span", {}, "band: runs of one commit"), h("span", {}, "dashed: a merge with a release note")),
    read,
    table,
    h("h4", { class: "w-label", style: "margin-top:1.6rem" }, "Each workload"),
    machineSeg ? h("div", { class: "w-controls" }, machineSeg) : null,
    h("div", { class: "w-legend" }, sizes.map((size) => h("span", {}, h("i", { style: `background:${SIZE_COLORS[size]}` }), `size ${size}`))),
    grid,
    h("p", { class: "w-note" }, `${fmtN(runs.length)} runs, read from ${data.ref} when this site was built.`),
  ].filter(Boolean));
  drawAll();
  showRun(selected, true);
}
