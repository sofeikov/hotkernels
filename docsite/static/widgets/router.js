// Which hook catches a NumPy call, and whether a kernel takes it.
//
// The reader picks a statement and describes the array; the figure draws the
// path the call takes: the statement, NumPy's Python wrapper where there is
// one, the hook with the checks it makes, and where the call ends: in a
// kernel, or in NumPy's own code exactly as before.
//
// The rules are those of the binding, hotkernels/src/hotkernels/_binding/,
// and of docs/delivery-design.md. Each entry of STATEMENTS names where its
// conditions come from.

import { h, seg } from "./util.js";

// Below this many elements the type-slot hook leaves the call to NumPy's own
// loop (HK_SLOT_MIN_N in hook_slot.c; "What changed in the building").
const SLOT_MIN_N = 128;
// The two array sizes the reader can choose between.
const SIZES = [16, 1048576];

// The five kinds of hook ("The five hooks"), and what each one changed.
const HOOKS = {
  function: { name: "Function hook", pointer: "The call pointer inside the NumPy function object was changed. The function is still the same object." },
  method: { name: "Method hook", pointer: "The C function pointer in ndarray's method table was changed." },
  slot: { name: "Type-slot hook", pointer: "The entry in the element type's own table of functions was changed." },
  loop: { name: "Inner-loop hook", pointer: "The ufunc's loop for this element type was replaced, through the interface NumPy provides for it." },
  index: { name: "Indexing hook", pointer: "One of the two pointers behind a[i] and a[i] = v in ndarray's type was changed." },
};

const OTHER = { label: "another type", kernel: null };

// The layout nonzero takes (hook_method.c hk_try_nonzero): one dimension, or on
// ARM64 a C-ordered matrix, which goes to nonzero_2d_bool.
const NONZERO_LAYOUT = "mask is contiguous and one-dimensional, or on Apple Silicon a C-ordered matrix (served by nonzero_2d_bool)";

// One entry per statement.
//   code      the statement as the reader writes it
//   array     the array the controls describe
//   via       the hook the statement meets first; `below` names the hook that sits under it,
//             "slot" (the type slot) or "loop" (the clip loop), which a call passed on reaches
//   fallback  what NumPy does with a call the first hook passes on, on its way to the hook below
//   types     element types offered, with the kernel that serves each (null: none)
//   typeText  how the element-type check reads
//   wrapper   what NumPy's Python wrapper does on this path, when it runs before the hook
//   oneD      the array must be one-dimensional
//   layout    how the layout check reads, where the usual wording does not fit
//   given     conditions this statement meets as written, listed so the reader sees them
//   loopGiven the same, for the clip loop below the first hook
//   extra     one more choice that decides a check
const STATEMENTS = [
  { // hook_function.c call_argext; "Which hook serves which kernel": type slot, plus the function and the method.
    code: "np.argmax(x)", array: "x", via: "function", below: "slot",
    types: [{ label: "float32", kernel: "argmax_f32" }, { label: "float64", kernel: "argmax_f64" }, OTHER],
    typeText: "elements are float32 or float64",
    given: ["only the array is passed: no axis, out or keepdims"],
    fallback: "The hook passes the call on. NumPy's own np.argmax runs, and ends in the element type's argmax function.",
  },
  { // hook_method.c method_argext; the type slot reaches x.argmax(), strided input and axis=.
    code: "x.argmax()", array: "x", via: "method", below: "slot",
    types: [{ label: "float32", kernel: "argmax_f32" }, { label: "float64", kernel: "argmax_f64" }, OTHER],
    typeText: "elements are float32 or float64",
    given: ["no axis, out or keepdims, or each given as None"],
    fallback: "The hook passes the call on. NumPy's own method runs, and ends in the element type's argmax function.",
  },
  { // hook_function.c try_clip and hk_try_clip (and hook_method.c try_clip for x.clip); below them hook_loop.c HK_CLIP_LOOP.
    code: "np.clip(x, lo, hi)", array: "x", via: "function", below: "loop",
    types: [{ label: "float32", kernel: "clip_f32" }, { label: "float64", kernel: "clip_f64" }, OTHER],
    typeText: "elements are float32 or float64",
    given: ["lo and hi are Python numbers, or NumPy scalars of x's own type, so the result keeps x's element type", "no out, dtype, casting or where"],
    fallback: "The hook passes the call on. np.clip calls x.clip, which runs the clip ufunc. NumPy settles the element type and describes the memory layout before the loop is called.",
    loopGiven: ["both bounds are scalars, not arrays", "the output is not the input (not in place)"],
    extra: { label: "Bounds", options: [{ label: "lo <= hi", ok: true }, { label: "lo > hi, or NaN", ok: false }], check: "lo <= hi, which a NaN bound also fails" },
  },
  { // hook_index.c try_compress.
    code: "x[mask]", array: "x", via: "index", oneD: true,
    types: [{ label: "float32", kernel: "compress_f32" }, { label: "float64", kernel: "compress_f64" }, OTHER],
    typeText: "elements are float32 or float64",
    given: ["the index is a plain boolean ndarray", "mask is one-dimensional, contiguous and as long as x", "every mask byte is 0 or 1 (the mask is read once to check)"],
  },
  { // hook_index.c try_maskfill.
    code: "x[mask] = 0.0", array: "x", via: "index", oneD: true,
    types: [{ label: "float64", kernel: "maskfill_f64" }, OTHER],
    typeText: "elements are float64",
    given: ["the index is a plain boolean ndarray", "mask is one-dimensional, contiguous and as long as x", "the value is a Python float or int",
      "x is writable and does not share memory with mask", "every mask byte is 0 or 1 (the mask is read once to check)"],
  },
  { // hook_function.c try_compress, sharing hook_index.c hk_try_compress; without an axis NumPy flattens x.
    code: "np.compress(mask, x)", array: "x", via: "function",
    types: [{ label: "float32", kernel: "compress_f32" }, { label: "float64", kernel: "compress_f64" }, OTHER],
    typeText: "elements are float32 or float64",
    given: ["no axis and no out: the kernel's scratch output never goes into the caller's buffer", "mask is a plain one-dimensional boolean ndarray, contiguous, with as many elements as x",
      "every mask byte is 0 or 1 (the mask is read once to check)"],
  },
  { // hook_function.c try_putmask, sharing hook_index.c hk_try_maskfill; NumPy needs only the sizes to match.
    code: "np.putmask(x, mask, 0.0)", array: "x", via: "function",
    types: [{ label: "float64", kernel: "maskfill_f64" }, OTHER],
    typeText: "elements are float64",
    given: ["mask is a plain contiguous boolean ndarray with as many elements as x", "the value is a Python float or int",
      "x is writable and does not share memory with mask", "every mask byte is 0 or 1 (the mask is read once to check)"],
  },
  { // hook_function.c call_where; np.where calls NumPy's C nonzero directly, not the method.
    code: "np.where(mask)", array: "mask", via: "function", oneD: true, layout: NONZERO_LAYOUT,
    types: [{ label: "bool", kernel: "nonzero_bool" }, OTHER],
    typeText: "elements are bool",
    given: ["mask is the only argument", "every mask byte is 0 or 1 (the mask is read once to check)"],
  },
  { // hook_method.c method_nonzero; np.nonzero calls the method.
    code: "np.nonzero(mask)", array: "mask", via: "method", oneD: true, layout: NONZERO_LAYOUT,
    types: [{ label: "bool", kernel: "nonzero_bool" }, OTHER],
    typeText: "elements are bool",
    wrapper: "np.nonzero(mask) is a Python function that calls mask.nonzero().",
    given: ["no arguments reach the method", "every mask byte is 0 or 1 (the mask is read once to check)"],
  },
  { // hook_method.c method_cumsum; "Decisions made in review": float kernels only with inexact=True.
    code: "np.cumsum(x)", array: "x", via: "method",
    types: [{ label: "int64", kernel: "cumsum_i64" }, { label: "float64", kernel: "cumsum_f64", inexact: true }, OTHER],
    typeText: "a kernel is installed for the element type: int64, or float32 and float64 after install(inexact=True)",
    wrapper: "np.cumsum(x) is a Python function that calls x.cumsum(axis=None, dtype=None, out=None).",
    given: ["axis, dtype and out are all None"],
    extra: { label: "Installed with", options: [{ label: "install()", ok: true }, { label: "install(inexact=True)", ok: true }] },
  },
  { // hook_method.c method_astype.
    code: "x.astype(np.float32)", array: "x", via: "method",
    types: [{ label: "uint8", kernel: "cast_u8_f32", c: true }, OTHER],
    typeText: "elements are uint8",
    given: ["float32 is the only argument, as np.float32, its dtype, \"float32\" or \"f4\""],
  },
  { // hook_function.c call_sum.
    code: "np.sum(x)", array: "x", via: "function",
    types: [{ label: "uint8", kernel: "sum_u8" }, OTHER],
    typeText: "elements are uint8",
    given: ["only the array is passed: no axis, dtype or out"],
  },
  { // hook_function.c call_count_nonzero.
    code: "np.count_nonzero(x)", array: "x", via: "function",
    types: [{ label: "int64", kernel: "count_nonzero_i64" }, { label: "float64", kernel: "count_nonzero_f64" }, OTHER],
    typeText: "elements are int64 or float64",
    given: ["only the array is passed: no axis"],
  },
  { // hook_function.c try_histogram, histogram_range, histogram_edges.
    code: "np.histogram(x, 50, (lo, hi))", array: "x", via: "function",
    types: [{ label: "float64", kernel: "histogram_f64" }, OTHER],
    typeText: "elements are float64",
    given: ["bins is a Python int of at least 1; no density or weights",
      "the range is a pair of finite numbers with lo < hi, and the bin edges, built as NumPy builds them, are strictly increasing"],
  },
  { // hook_function.c try_bincount.
    code: "np.bincount(x)", array: "x", via: "function", oneD: true,
    types: [{ label: "uint8", kernel: "bincount_u8" }, OTHER],
    typeText: "elements are uint8",
    given: ["no weights"],
  },
  { // hook_function.c try_isin, isin_test_values; "What changed in the building": a list of Python ints is accepted too.
    code: "np.isin(x, values)", array: "x", via: "function",
    types: [{ label: "int64", kernel: "isin_i64" }, OTHER],
    typeText: "elements are int64",
    given: ["only the two arguments are passed"],
    extra: { label: "Test values", options: [{ label: "up to 8", ok: true }, { label: "more than 8", ok: false }],
      check: "values is an int64 array, or a list or tuple of Python ints, with 1 to 8 entries" },
  },
];

export const css = `
.rt-controls { display: grid; gap: .55rem; margin-bottom: 1.2rem; }
.rt-controls .w-controls { margin: 0; gap: .4rem .7rem; }
.rt-controls .w-label { min-width: 6.3rem; }
.rt-path { display: flex; flex-direction: column; max-width: 36rem; margin: 0 auto; }
.rt-box { border: 1px solid var(--rule-strong); border-radius: 10px; padding: .65rem .85rem; background: var(--bg-raised); }
.w .rt-box p { margin: .2rem 0 0; font-size: .84rem; color: var(--muted); line-height: 1.45; }
.rt-box .tag { font-size: .7rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); }
.rt-box .title { font-weight: 650; }
.rt-stmt { background: var(--panel); font-family: var(--mono); font-size: .92rem; font-weight: 600; }
.rt-wrap.skipped { border-style: dashed; background: transparent; }
.rt-wrap.skipped .title { color: var(--muted); text-decoration: line-through; text-decoration-color: var(--rule-strong); }
.rt-hook { border-left: 3px solid var(--ink); }
.w .rt-checks { list-style: none; margin: .55rem 0 0; padding: 0; font-size: .84rem; line-height: 1.45; }
.w .rt-checks li { display: grid; grid-template-columns: 1.3rem 1fr; padding: .12rem 0; margin: 0; }
.rt-checks .mark { font-weight: 700; color: var(--good); }
.rt-checks li.fail { color: var(--bad); font-weight: 600; }
.rt-checks li.fail .mark { color: var(--bad); }
.rt-out.kernel { background: var(--hot-soft); border-color: var(--hot); }
.rt-out.kernel .title { color: var(--hot); }
.rt-out.numpy { background: var(--np-soft); border-color: var(--np); }
.rt-out.numpy .title { color: var(--np); }
.rt-out .title code { font-size: .95em; }
.rt-arrow { align-self: center; color: var(--muted); font-size: 1rem; line-height: 1; padding: .3rem 0; }
`;

function check(label, ok) {
  return { label, ok };
}

// The kernel's display name, what it is written in, and how its result compares with NumPy's.
function kernelLabel(type) {
  return { name: type.kernel, written: type.c ? "C" : "assembly", exact: !type.inexact };
}

// Works out the path a call takes. Returns the boxes of the figure in order.
function route(statement, state) {
  const type = statement.types[state.type];
  const plain = state.kind === "plain";
  const contiguous = state.layout === "contiguous";
  const big = SIZES[state.size] >= SLOT_MIN_N;
  const extra = statement.extra ? statement.extra.options[state.extra] : null;
  const name = statement.array;
  const steps = [{ box: "statement", code: statement.code }];

  const kernel = () => ({ box: "kernel", ...kernelLabel(type) });
  const numpy = (reached) => ({ box: "numpy", reached });
  const slot = () => {
    steps.push({
      box: "hook", hook: "slot",
      note: "NumPy calls this entry with one contiguous run of elements, having copied the data first if the array was strided. The array's class no longer matters here.",
      checks: [check(`at least ${SLOT_MIN_N} elements: the hook has to take the interpreter lock, which costs more than the kernel saves on a short run`, big)],
    });
    steps.push(big ? kernel() : numpy(true));
  };
  const loop = () => {
    const checks = [
      check("x and the output are contiguous: the step between elements is the element size", contiguous),
      ...statement.loopGiven.map((label) => check(label, true)),
      check(statement.extra.check, extra.ok),
    ];
    steps.push({ box: "hook", hook: "loop", checks,
      note: "The loop runs after NumPy's own checking, so the array's class does not matter here." });
    steps.push(checks.every((c) => c.ok) ? kernel() : numpy(true));
  };

  // The checks the function, method and indexing hooks share ("When a call takes a kernel").
  const shared = () => {
    const typeOk = type.kernel !== null && (!type.inexact || state.extra === 1);
    const layoutText = statement.layout
      || (statement.oneD ? `${name} is one-dimensional and contiguous` : `${name} is contiguous (any shape)`);
    const list = [
      check(`${name} is exactly an ndarray, not a subclass`, plain),
      check(statement.typeText, typeOk),
      check(layoutText, contiguous),
      ...(statement.given || []).map((label) => check(label, true)),
    ];
    if (extra && statement.extra.check) list.push(check(statement.extra.check, extra.ok));
    if (type.inexact && typeOk) list.push(check("the kernel raised no floating-point flag (if it does, its output is dropped and NumPy computes the result)", true));
    return list;
  };

  if (statement.via === "method" && statement.wrapper) {
    steps.push({ box: "wrapper", title: "NumPy's Python wrapper", text: statement.wrapper });
  }
  const checks = shared();
  const taken = checks.every((c) => c.ok);
  steps.push({ box: "hook", hook: statement.via, checks });
  if (statement.via === "function" && taken) {
    steps.push({ box: "wrapper", skipped: true, title: "NumPy's Python wrapper",
      text: "Skipped: the hook answered before it ran. On a small array the wrapper is most of the cost of the call." });
  }
  if (taken) {
    steps.push(kernel());
  } else if (statement.below && type.kernel) {
    const title = statement.via === "method" ? "NumPy's own method" : "NumPy's Python wrapper";
    steps.push({ box: "wrapper", title, text: statement.fallback });
    if (statement.below === "slot") slot();
    else loop();
  } else {
    steps.push(numpy(true));
  }
  return steps;
}

function drawStep(step) {
  if (step.box === "statement") return h("div", { class: "rt-box rt-stmt" }, step.code);
  if (step.box === "wrapper") {
    return h("div", { class: `rt-box rt-wrap${step.skipped ? " skipped" : ""}` },
      h("div", { class: "title" }, step.title), h("p", {}, step.text));
  }
  if (step.box === "hook") {
    const hook = HOOKS[step.hook];
    return h("div", { class: "rt-box rt-hook" },
      h("div", { class: "tag" }, "Hook"),
      h("div", { class: "title" }, hook.name),
      h("p", {}, hook.pointer),
      step.note && h("p", {}, step.note),
      h("ul", { class: "rt-checks" }, step.checks.map((c) =>
        h("li", { class: c.ok ? "" : "fail" }, h("span", { class: "mark", "aria-label": c.ok ? "met" : "not met" }, c.ok ? "✓" : "✗"), h("span", {}, c.label)))));
  }
  if (step.box === "kernel") {
    return h("div", { class: "rt-box rt-out kernel" },
      h("div", { class: "title" }, "Kernel: ", h("code", {}, step.name), ` (${step.written})`),
      h("p", {}, step.exact
        ? "The result is the one NumPy returns: the same type, shape and bits."
        : "The result has NumPy's type and shape and is within the kernel's declared tolerance of NumPy's values, not bit-identical."));
  }
  return h("div", { class: "rt-box rt-out numpy" },
    h("div", { class: "title" }, "NumPy's own code, as before"),
    h("p", {}, step.reached
      ? "Cost of the detour: about 1 ns."
      : "No hook sits on this path for this element type, so nothing is added."));
}

export default function mount(el) {
  const state = { statement: 0, type: 0, layout: "contiguous", kind: "plain", size: 1, extra: 0 };

  const select = h("select", { class: "w-select", "aria-label": "Statement",
    onchange: () => { state.statement = select.selectedIndex; state.type = 0; state.extra = 0; controls(); draw(); } },
  STATEMENTS.map((statement) => h("option", {}, statement.code)));
  const choices = h("div", { class: "rt-controls" });
  const path = h("div", { class: "rt-path", "aria-live": "polite" });

  function row(label, options, value, set) {
    return h("div", { class: "w-controls" }, h("span", { class: "w-label" }, label),
      seg(options, value, (v) => { set(v); draw(); }));
  }

  // The choices depend on the statement: its element types, and its extra question if it has one.
  function controls() {
    const statement = STATEMENTS[state.statement];
    const name = statement.array;
    choices.replaceChildren(
      row("Element type", statement.types.map((t, i) => ({ value: i, label: t.label })), state.type, (v) => { state.type = Number(v); }),
      row("Layout", [{ value: "contiguous", label: "contiguous" }, { value: "strided", label: `strided, ${name}[::2]` }], state.layout, (v) => { state.layout = v; }),
      row("Kind", [{ value: "plain", label: "plain ndarray" }, { value: "subclass", label: "a subclass" }], state.kind, (v) => { state.kind = v; }),
      row("Size", SIZES.map((n, i) => ({ value: i, label: n.toLocaleString("en-US") })), state.size, (v) => { state.size = Number(v); }),
    );
    if (statement.extra) {
      choices.append(row(statement.extra.label, statement.extra.options.map((o, i) => ({ value: i, label: o.label })), state.extra, (v) => { state.extra = Number(v); }));
    }
  }

  function draw() {
    const steps = route(STATEMENTS[state.statement], state);
    path.replaceChildren();
    steps.forEach((step, i) => {
      if (i > 0) path.append(h("div", { class: "rt-arrow", "aria-hidden": "true" }, "↓"));
      path.append(drawStep(step));
    });
  }

  el.append(
    h("div", { class: "w-controls" }, h("span", { class: "w-label" }, "Statement"), select),
    choices,
    h("div", { class: "w-stage" }, path),
    h("p", { class: "w-note" }, "A call a kernel does not take is passed to NumPy with the caller's own arguments; nothing is converted or copied on the way."),
  );
  controls();
  draw();
}
