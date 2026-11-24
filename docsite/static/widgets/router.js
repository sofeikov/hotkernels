// Which hook catches a NumPy call, and what serves it.
//
// The reader picks a statement and describes the array; the figure draws the
// path the call takes: the statement, each hook it meets with the checks that
// hook makes, the NumPy code that runs between two hooks, and where the call
// ends: in a kernel, in a loop built into hotkernels, in NumPy's own inner
// loop called directly, or in NumPy's own code exactly as before.
//
// The rules are those of the binding, hotkernels/src/hotkernels/_binding/,
// and of the table "Which hook serves which kernel" in docs/delivery-design.md.
// Each entry of STATEMENTS names the code its conditions come from. When the
// binding changes what a call needs or where it goes, this file changes with
// it (AGENTS.md, "Documentation moves with the change").

import { h, seg } from "./util.js";

// Below this many elements the type-slot hook leaves the call to NumPy's own
// loop (HK_SLOT_MIN_N in hook_slot.c).
const SLOT_MIN_N = 128;
// Most elements the ufunc lane takes (LANE_MAX_ELEMENTS in hook_ufunc.c).
const LANE_MAX_N = 16384;
// Most elements the binding's int64 clip and float64 round loops take
// (CLIP_I64_MAX_N in hook_function.c, ROUND_F64_MAX_N in hook_method.c).
const SHORT_LOOP_MAX_N = 4096;
// Fewest elements the stable argsort kernels take (ARGSORT_MIN_N in hook_method.c).
const ARGSORT_MIN_N = 256;
// Fewest elements the in-place add and subtract loops take (INPLACE_MIN_N in hook_loop.c).
const INPLACE_MIN_N = 131072;
// The two array sizes the reader can choose between.
const SIZES = [16, 1048576];

// The kinds of hook, as report() and hooks.py name them, and what each one changed.
const HOOKS = {
  function: { name: "Function hook", pointer: "The call pointer inside the NumPy function object, the one CPython calls it through, was changed. The function is still the same object, so names imported before install() reach the hook too." },
  method: { name: "Method hook", pointer: "The C function pointer in the method's entry of its type's method table was changed. Every way of calling the method, a bound method saved earlier included, reads it at the time of the call." },
  index: { name: "Indexing hook", pointer: "One of the two pointers behind a[i] and a[i] = v in ndarray's type was changed. Every index operation passes it; one whose index is not a boolean array leaves after one comparison." },
  slot: { name: "Type-slot hook", pointer: "The argmax or argmin entry in the element type's own table of functions was changed." },
  loop: { name: "Inner-loop hook", pointer: "The ufunc's loop for this element type was replaced, through the interface NumPy provides for it." },
  ufunc: { name: "Ufunc call hook", pointer: "The call pointer inside each of NumPy's ufunc objects was changed. ndarray's operators, a + b and a > 0 among them, call through it too." },
  bottleneck: { name: "bottleneck hook", pointer: "The call pointer inside bottleneck's function object was set. bottleneck leaves it empty, and CPython then calls the function another way, which is where a call the hook does not take goes." },
};

// Kernels written in C rather than assembly, and on which architecture.
const C_KERNELS = {
  cast_u8_f32: "ARM64", compress_i16: "ARM64", compress_u8: "ARM64", isin_i32: "ARM64", maskfill_f32: "ARM64",
  argsort_f64: "both", argsort_i64: "both", argsort_u64: "both",
  nanmean_cols_f64: "both", nanstd_cols_f64: "both", nanmin_cols_f64: "both", nanmax_cols_f64: "both",
};

// What can serve a call.
const kernel = (name, how = "") => ({ kind: "kernel", name, how });
const inexactKernel = (name) => ({ kind: "kernel", name, how: "", inexact: true });
const builtin = (name, how = "") => ({ kind: "builtin", name, how });
const LANE = { kind: "lane", name: "ufunc_lane" };

// An element type the reader can pick: its label, what serves it, and what more it needs.
const T = (label, serve, more = {}) => ({ label, serve, ...more });
const OTHER = T("another type", null);

// Wording shared by several statements.
const MASK_CLEAN = "every mask byte is 0 or 1 (the mask is read once to check)";
const NONZERO_LAYOUT = "mask is contiguous and one-dimensional, or a C-ordered matrix (served by nonzero_2d_bool)";
const SKIPPED = "Skipped: the hook answered before it ran. On a small array the wrapper is most of the cost of the call.";
const SLOT_NOTE = "NumPy calls this entry with one contiguous run of elements, having copied the data first if the array was strided. The array's class no longer matters here.";
const LOOP_NOTE = "The loop runs after NumPy's own checking, so the array's class does not matter here.";
const MASK_TYPES = [
  T("float64 or int64", kernel("compress_f64")), T("float32 or int32", kernel("compress_f32")),
  T("int16 or uint16", kernel("compress_i16")), T("uint8, int8 or bool", kernel("compress_u8")), OTHER,
];
const MASK_TYPE_TEXT = "elements are float64, int64, float32, int32, int16, uint16, uint8, int8 or bool: the compress kernels move them as 8-, 4-, 2- and 1-byte patterns";
const FILL_TYPES = [T("float64 or int64", kernel("maskfill_f64")), T("float32 or int32", kernel("maskfill_f32")), OTHER];
const FILL_TYPE_TEXT = "elements are float64, int64, float32 or int32: the maskfill kernels move them as 8- and 4-byte patterns";
const FILL_VALUE = "v is a value it takes for x's type: a Python float or int for float64; for int64 and int32 a Python int in range or the array's own scalar type (see Get started)";
const LANE_GIVEN = ["no keyword but out, which is the first input itself or an array clear of the inputs"];

function check(label, ok) {
  return { label, ok };
}

// The checks most hooks make, in the order they make them. `c` is the call
// described by the controls (see route); `layout` says how the layout check reads.
function common(c, { layout, given = [], size = null }) {
  const s = c.statement;
  const list = [
    check(s.plainText || `${s.array} is exactly an ndarray, not a subclass`, c.plain),
    check(s.typeText, c.typeOk),
    check(c.type.layout || layout, c.layoutOk),
  ];
  if (size) list.push(check(size.text, size.ok(c.n)));
  if (c.type.limit) list.push(check(c.type.limit.text, c.type.limit.ok(c.n)));
  list.push(...given.map((label) => check(label, true)));
  list.push(...(c.type.given || []).map((label) => check(label, true)));
  if (c.extra && s.extra.check) list.push(check(s.extra.check, c.extra.ok));
  if (c.type.serve && c.type.serve.inexact && c.typeOk) {
    list.push(check("the kernel raised no floating-point flag (if it does, its output is dropped and NumPy computes the result)", true));
  }
  return list;
}

// The type slot under argmax and argmin, which NumPy reaches whatever spelling passed the call on.
function slotStage(via, rows = false) {
  return {
    hook: "slot", via, note: SLOT_NOTE, applies: (c) => c.type.k !== undefined,
    checks: (c) => [rows
      ? check(`each row has at least ${SLOT_MIN_N} elements: the hook has to take the interpreter lock, which costs more than the kernel saves on a short row`, !c.extra.narrow)
      : check(`at least ${SLOT_MIN_N} elements: the hook has to take the interpreter lock, which costs more than the kernel saves on a short run`, c.n >= SLOT_MIN_N)],
    serve: (c) => kernel(c.type.k, rows ? "once per row" : ""),
  };
}

// The families, as groups of the statement list.
const FIND = "Finding", CLIP = "Clipping, rounding, median", MASKS = "Masks", COUNT = "Sums and counts",
  SEARCH = "Searching and sorting", ACC = "Accumulating", ELEM = "Elementwise", BN = "bottleneck";

// One entry per statement.
//   family     the group it is listed under
//   code       the statement as the reader writes it
//   array      the array the controls describe
//   types      element types offered, with what serves each (null: nothing)
//   typeText   how the element-type check reads
//   path       the hooks the call meets, in order; each is
//                hook     a key of HOOKS
//                via      what NumPy does between the previous hook and this one
//                applies  whether this hook is there for the chosen type (default: always)
//                checks   the checks it makes, from the call
//                serve    what serves the call when every check holds (default: the type's)
//                note     what the hook can take for granted
//   extra      one more choice that decides a check: {label, options: [{label, ok}], check}
//   layouts    the layouts offered, when not "contiguous" and "strided"
//   sizes      false when the array's size decides nothing here
//   fallback   what runs when no hook takes the call
const STATEMENTS = [
  { // hook_function.c call_argext and hk_try_argext; below them hook_slot.c.
    family: FIND, code: "np.argmax(x)", array: "x",
    types: [T("float32", kernel("argmax_f32"), { k: "argmax_f32" }), T("float64", kernel("argmax_f64"), { k: "argmax_f64" }), OTHER],
    typeText: "elements are float32 or float64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["only the array is passed: no axis, out or keepdims"] }) },
      slotStage("The hook passes the call on. NumPy's own np.argmax runs, and ends in the element type's argmax function."),
    ],
  },
  { // hook_method.c method_argext; below it hook_slot.c.
    family: FIND, code: "x.argmax()", array: "x",
    types: [T("float32", kernel("argmax_f32"), { k: "argmax_f32" }), T("float64", kernel("argmax_f64"), { k: "argmax_f64" }), OTHER],
    typeText: "elements are float32 or float64",
    path: [
      { hook: "method", checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["no axis, out or keepdims, or each given as None"] }) },
      slotStage("The hook passes the call on. NumPy's own method runs, and ends in the element type's argmax function."),
    ],
  },
  { // hook_function.c hk_try_argext_rows (ARGEXT_ROWS_KERNEL_COLS_F64 and _F32); below it hook_slot.c, once per row.
    family: FIND, code: "np.argmax(m, axis=1)", array: "m", sizes: false,
    types: [T("float32", null, { k: "argmax_f32", rows: "argmax_rows_f32" }), T("float64", null, { k: "argmax_f64", rows: "argmax_rows_f64" }), OTHER],
    typeText: "elements are float32 or float64",
    extra: { label: "Rows", options: [{ label: "narrow: 4 elements", ok: true, narrow: true }, { label: "wide: 1,000 elements", ok: true }] },
    path: [
      { hook: "function", skips: true,
        checks: (c) => common(c, { layout: "m is two-dimensional and in C order, with at least one column", given: ["axis is the number 1 or -1, by position or by name; no out or keepdims"] }),
        serve: (c) => (c.extra.narrow ? builtin(c.type.rows, "a row per vector lane") : kernel(c.type.k, "once per row")) },
      slotStage("The hook passes the call on. NumPy's own np.argmax calls the element type's argmax function once per row.", true),
    ],
  },
  { // hook_function.c try_clip and hk_try_clip; hook_method.c method_clip; below them hook_loop.c HK_CLIP_LOOP.
    family: CLIP, code: "np.clip(x, lo, hi)", array: "x",
    types: [
      T("float32", kernel("clip_f32"), { loop: true }), T("float64", kernel("clip_f64"), { loop: true }),
      T("int64", builtin("clip_i64"), { limit: { text: `int64: at most ${SHORT_LOOP_MAX_N.toLocaleString("en-US")} elements; above that NumPy's own loop is as fast`, ok: (n) => n <= SHORT_LOOP_MAX_N } }),
      OTHER,
    ],
    typeText: "elements are float32 or float64, or int64 for the binding's own loop",
    extra: { label: "Bounds", options: [{ label: "lo <= hi", ok: true }, { label: "lo > hi, or NaN", ok: false }], check: "lo <= hi, which a NaN bound also fails" },
    path: [
      { hook: "function", skips: true,
        checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["lo and hi are Python numbers, or NumPy scalars of x's own type, so the result keeps x's element type", "no dtype, casting or where; no out, or out is x itself (float32 and float64)"] }) },
      { hook: "loop", note: LOOP_NOTE, applies: (c) => c.type.loop === true,
        via: "The hook passes the call on. np.clip calls x.clip, whose method hook makes the same checks and passes it on too. NumPy then runs the clip ufunc: it settles the element type and describes the memory layout before it calls the loop.",
        checks: (c) => [
          check("x and the output are contiguous: the step between elements is the element size", c.layoutOk),
          check("both bounds are scalars, not arrays", true),
          check(c.statement.extra.check, c.extra.ok),
          check("the output is clear of x, or is x itself, which is clipped through a buffer of the binding's", true),
        ] },
    ],
  },
  { // hook_function.c call_round and call_around, hook_method.c hk_try_round.
    family: CLIP, code: "np.round(x, 2)", array: "x",
    types: [T("float64", builtin("round_f64"), { limit: { text: `at most ${SHORT_LOOP_MAX_N.toLocaleString("en-US")} elements; above that NumPy's own loops are as fast`, ok: (n) => n <= SHORT_LOOP_MAX_N } }), OTHER],
    typeText: "elements are float64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["decimals is left out or a Python int from 0 to 15; no out", "no multiplication overflows and x holds no signalling NaN: otherwise NumPy computes the call and warns"] }) },
    ],
  },
  { // hook_function.c call_median.
    family: CLIP, code: "np.median(x)", array: "x",
    types: [T("float64", builtin("median_f64")), OTHER],
    typeText: "elements are float64",
    extra: { label: "Values", options: [{ label: "no NaN", ok: true }, { label: "a NaN", ok: false }], check: "x holds no NaN (it is read once to check)" },
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["the array alone, or with axis=None", "no zero median beside a negative zero, and no overflow in the mean of the two middle elements"] }) },
    ],
  },
  { // hook_index.c get_by_mask and hk_try_compress.
    family: MASKS, code: "x[mask]", array: "x", types: MASK_TYPES, typeText: MASK_TYPE_TEXT,
    path: [
      { hook: "index", checks: (c) => common(c, { layout: "x and mask are contiguous and of the same shape, of any number of dimensions", given: ["the index is a plain boolean ndarray", MASK_CLEAN] }) },
    ],
  },
  { // hook_index.c set_by_mask and hk_try_maskfill.
    family: MASKS, code: "x[mask] = v", array: "x", types: FILL_TYPES, typeText: FILL_TYPE_TEXT,
    path: [
      { hook: "index", checks: (c) => common(c, { layout: "x and mask are contiguous and of the same shape, of any number of dimensions", given: ["the index is a plain boolean ndarray", FILL_VALUE, "x is writable and does not share memory with mask", MASK_CLEAN] }) },
    ],
  },
  { // hook_function.c try_compress, sharing hook_index.c hk_try_compress; without an axis NumPy flattens x.
    family: MASKS, code: "np.compress(mask, x)", array: "x", types: MASK_TYPES, typeText: MASK_TYPE_TEXT,
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["mask is a plain one-dimensional boolean ndarray, contiguous, with as many elements as x", "no axis and no out: the kernel's scratch output never goes into the caller's buffer", MASK_CLEAN] }) },
    ],
  },
  { // hook_function.c try_putmask, sharing hook_index.c hk_try_maskfill; NumPy needs only the sizes to match.
    family: MASKS, code: "np.putmask(x, mask, v)", array: "x", types: FILL_TYPES, typeText: FILL_TYPE_TEXT,
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["mask is a plain contiguous boolean ndarray with as many elements as x", FILL_VALUE, "x is writable and does not share memory with mask", MASK_CLEAN] }) },
    ],
  },
  { // hook_function.c call_where; np.where calls NumPy's C nonzero directly, not the method.
    family: MASKS, code: "np.where(mask)", array: "mask",
    types: [T("bool", kernel("nonzero_bool")), OTHER], typeText: "elements are bool",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: NONZERO_LAYOUT, given: ["mask is the only argument", MASK_CLEAN] }) },
    ],
  },
  { // hook_method.c method_nonzero; np.nonzero calls the method.
    family: MASKS, code: "np.nonzero(mask)", array: "mask",
    types: [T("bool", kernel("nonzero_bool")), OTHER], typeText: "elements are bool",
    path: [
      { hook: "method", via: "np.nonzero(mask) is a Python function that calls mask.nonzero(); np.argwhere calls np.nonzero.",
        checks: (c) => common(c, { layout: NONZERO_LAYOUT, given: ["no arguments reach the method", MASK_CLEAN] }) },
    ],
  },
  { // hook_function.c call_where, try_where; int64 moves through the float64 kernel as bit patterns.
    family: MASKS, code: "np.where(mask, a, b)", array: "a",
    types: [T("float64", kernel("where_f64")), T("int64", kernel("where_f64", "as bit patterns")), OTHER],
    typeText: "a and b are both float64 or both int64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "mask, a and b are contiguous and of one shape, of any number of dimensions", given: ["mask is a plain boolean ndarray; for float64 one of a and b may be a single number instead", MASK_CLEAN] }) },
    ],
  },
  { // hook_method.c method_any and hk_try_any_rows; np.any(m, axis=1) has a function hook of its own.
    family: MASKS, code: "m.any(axis=1)", array: "m",
    types: [T("bool", kernel("any_rows_bool")), OTHER], typeText: "elements are bool",
    path: [
      { hook: "method", checks: (c) => common(c, { layout: "m is two-dimensional and in C order, with at least one column", given: ["axis is the number 1 or -1, by position or by name, and no other argument", MASK_CLEAN] }) },
    ],
  },
  { // hook_method.c method_any and hk_try_any_cols; the kernel takes any non-zero byte as True, as NumPy does.
    family: MASKS, code: "m.any(axis=0)", array: "m",
    types: [T("bool", kernel("any_cols_bool")), OTHER], typeText: "elements are bool",
    path: [
      { hook: "method", checks: (c) => common(c, { layout: "m is two-dimensional and in C order, with at least one row and one column", given: ["axis is the number 0 or -2, by position or by name, and no other argument"] }) },
    ],
  },
  { // hook_function.c call_sum and hk_try_sum.
    family: COUNT, code: "np.sum(x)", array: "x",
    types: [
      T("uint8", kernel("sum_u8")),
      T("bool", kernel("sum_u8"), { given: ["every byte is 0 or 1 (x is read once to check)"] }),
      T("float32", kernel("sum_f32"), { layout: "x is one-dimensional, contiguous, and aligned to 4 bytes", given: ["the sum is finite: otherwise the kernel's result is dropped and NumPy computes it and warns"] }),
      T("float64", kernel("sum_f64"), { layout: "x is one-dimensional, contiguous, and aligned to 8 bytes", given: ["the sum is finite: otherwise the kernel's result is dropped and NumPy computes it and warns"] }),
      OTHER,
    ],
    typeText: "elements are uint8, bool, float32 or float64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["only the array is passed: no axis, dtype, out, keepdims, initial or where"] }) },
    ],
  },
  { // hook_function.c call_count_nonzero.
    family: COUNT, code: "np.count_nonzero(x)", array: "x",
    types: [T("int64", kernel("count_nonzero_i64")), T("float64", kernel("count_nonzero_f64")), OTHER],
    typeText: "elements are int64 or float64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["only the array is passed: no axis"] }) },
    ],
  },
  { // hook_method.c method_cumsum; the float kernels only under install(inexact=True).
    family: COUNT, code: "np.cumsum(x)", array: "x",
    types: [T("int64", kernel("cumsum_i64")), T("bool", kernel("cumsum_bool"), { given: ["every byte is 0 or 1 (x is read once to check)"] }),
      T("float32", inexactKernel("cumsum_f32")), T("float64", inexactKernel("cumsum_f64")), OTHER],
    typeText: "a kernel is installed for the element type: int64 and bool, or float32 and float64 after install(inexact=True)",
    extra: { label: "Installed with", options: [{ label: "install()", ok: true }, { label: "install(inexact=True)", ok: true, inexact: true }] },
    path: [
      { hook: "method", via: "np.cumsum(x) is a Python function that calls x.cumsum(axis=None, dtype=None, out=None).",
        checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["axis, dtype and out are all None"] }) },
    ],
  },
  { // hook_function.c try_bincount and bincount_codes.
    family: COUNT, code: "np.bincount(x)", array: "x",
    types: [T("uint8", kernel("bincount_u8")),
      T("int16", kernel("bincount_i16"), { given: ["no element is negative, and the result is no longer than 65,536 counts or than x (x is read once for its largest element)"] }),
      T("int64", kernel("bincount_i64"), { given: ["no element is negative, and the result is no longer than 65,536 counts or than x (x is read once for its largest element)"] }),
      OTHER],
    typeText: "elements are uint8, int16 or int64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is one-dimensional and contiguous", given: ["no weights; minlength, if given, is a Python int"] }) },
    ],
  },
  { // hook_function.c try_histogram, histogram_range, histogram_edges.
    family: COUNT, code: "np.histogram(x, 50, (lo, hi))", array: "x",
    types: [T("float64", kernel("histogram_f64")), OTHER], typeText: "elements are float64",
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["bins is a Python int of at least 1; no density or weights",
        "the range is a pair of finite numbers with lo < hi, and the bin edges, built as NumPy builds them, are strictly increasing"] }) },
    ],
  },
  { // hook_function.c call_unique; a layout other than C order is copied first, as NumPy's flatten does.
    family: COUNT, code: "np.unique(x)", array: "x", anyLayout: true,
    types: [T("int64", kernel("unique_i64")), OTHER], typeText: "elements are int64",
    extra: { label: "Values", options: [{ label: "labels, close together", ok: true }, { label: "far apart", ok: false }], check: "the largest element minus the smallest is under 256 or under the element count (x is read once to check)" },
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x has at least one element, of any shape or layout: one that is not contiguous is copied first", given: ["no return_index, return_inverse or return_counts; no axis; sorted left at True"] }) },
    ],
  },
  { // hook_method.c method_astype.
    family: COUNT, code: "x.astype(np.float32)", array: "x",
    types: [T("uint8", kernel("cast_u8_f32")), OTHER], typeText: "elements are uint8",
    path: [
      { hook: "method", checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["float32 is the only argument, as np.float32, its dtype, \"float32\" or \"f4\""] }) },
    ],
  },
  { // hook_function.c try_isin, isin_test_values.
    family: SEARCH, code: "np.isin(x, values)", array: "x",
    types: [T("int64", kernel("isin_i64")), T("int32", kernel("isin_i32")), OTHER], typeText: "elements are int64 or int32",
    extra: { label: "Test values", options: [{ label: "up to 8", ok: true }, { label: "more than 8", ok: false }], check: "values holds 1 to 8 entries" },
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x is contiguous (any shape)", given: ["only the two arguments are passed", "values is an array of x's element type, or a list or tuple of Python ints that fit it"] }) },
    ],
  },
  { // hook_method.c method_searchsorted; each side has a kernel of its own.
    family: SEARCH, code: "np.searchsorted(a, v)", array: "a",
    types: [T("float64", null, { k: "searchsorted_f64", right: "searchsorted_right_f64" }), T("int64", null, { k: "searchsorted_i64", right: "searchsorted_right_i64" }), OTHER],
    typeText: "elements are float64 or int64",
    extra: { label: "Side", options: [{ label: "left", ok: true }, { label: "right", ok: true, right: true }] },
    path: [
      { hook: "method", via: "np.searchsorted(a, v) is a Python function that calls a.searchsorted(v). np.digitize(x, bins) calls np.searchsorted when its bins increase.",
        checks: (c) => common(c, { layout: "a is one-dimensional and contiguous", given: ["v is an array of a's element type (a float64 a also takes float32), or a single number; no sorter"] }),
        serve: (c) => kernel(c.extra.right ? c.type.right : c.type.k) },
    ],
  },
  { // hook_function.c try_interp.
    family: SEARCH, code: "np.interp(x, xp, fp)", array: "x",
    types: [T("float64", kernel("interp_f64")), OTHER], typeText: "x, xp and fp are float64",
    extra: { label: "Sample points", options: [{ label: "up to 64 per query", ok: true }, { label: "more than 64 per query", ok: false }], check: "at most 64 sample points per query, and none of them NaN (xp is read once to check)" },
    path: [
      { hook: "function", skips: true, checks: (c) => common(c, { layout: "x, xp and fp are contiguous; xp and fp are one-dimensional and the same length", given: ["no left, right or period"] }) },
    ],
  },
  { // hook_method.c method_argsort; a strided view is copied element by element.
    family: SEARCH, code: "np.argsort(x, kind=\"stable\")", array: "x", anyLayout: true,
    types: [T("float64", kernel("argsort_f64")), T("int64", kernel("argsort_i64")), T("uint64", kernel("argsort_u64")), OTHER],
    typeText: "elements are float64, int64 or uint64",
    extra: { label: "Sort kind", options: [{ label: "kind=\"stable\"", ok: true }, { label: "the default kind", ok: false }], check: "the sort is stable: kind is \"stable\" or \"mergesort\", or stable=True. The default kind's order of equal elements is NumPy's quicksort's" },
    path: [
      { hook: "method", via: "np.argsort(x, kind=...) is a Python function that calls x.argsort(kind=...).",
        checks: (c) => common(c, { layout: "x is one-dimensional; it need not be contiguous: a view with a step is copied first",
          size: { text: `at least ${ARGSORT_MIN_N} elements`, ok: (n) => n >= ARGSORT_MIN_N },
          given: ["no order or descending; axis left out, 0 or -1", "the memory it borrows, 32 bytes an element, can be had; an array already sorted is answered without sorting"] }) },
    ],
  },
  { // hook_method.c method_accumulate, the ufunc type's method.
    family: ACC, code: "np.fmax.accumulate(x)", array: "x",
    types: [T("int64", kernel("cummax_i64")),
      T("float64", kernel("cumfmax_f64"), { given: ["x holds no signalling NaN, no negative zero, and NaNs of one bit pattern only (it is read once to check)"] }), OTHER],
    typeText: "elements are int64 or float64",
    path: [
      { hook: "method", note: "accumulate is a method of the ufunc type, which every ufunc shares: the hook tells np.fmax from the others by the object it is called on.",
        checks: (c) => common(c, { layout: "x is one-dimensional and contiguous, with at least one element", given: ["no dtype or out; axis left out, 0 or -1"] }) },
    ],
  },
  { // hook_method.c method_accumulate; float64 maximum carries a NaN forward, which no kernel does.
    family: ACC, code: "np.maximum.accumulate(x)", array: "x",
    types: [T("int64", kernel("cummax_i64")), OTHER], typeText: "elements are int64 (on float64 np.maximum carries a NaN forward, which no kernel does)",
    path: [
      { hook: "method", note: "accumulate is a method of the ufunc type, which every ufunc shares: the hook tells np.maximum from the others by the object it is called on.",
        checks: (c) => common(c, { layout: "x is one-dimensional and contiguous, with at least one element", given: ["no dtype or out; axis left out, 0 or -1"] }) },
    ],
  },
  { // hook_method.c method_cumsum, and the rows loops of docs/accumulate-rows-design.md.
    family: ACC, code: "a.cumsum(axis=0)", array: "a",
    types: [T("float64", builtin("cumsum_rows_f64", "a row at a time")), OTHER], typeText: "elements are float64",
    path: [
      { hook: "method", checks: (c) => common(c, { layout: "a has two or more dimensions, is in C order, and has at least one row and one column",
        given: ["axis is 0 or -a.ndim; no dtype; out left out, or a C-ordered float64 array of a's shape that is a itself or does not overlap it",
          "a sum that meets a NaN or a floating-point flag is finished by NumPy from that row, so the warnings are NumPy's"] }) },
    ],
  },
  { // hook_method.c method_reduceat, the ufunc type's method.
    family: ACC, code: "np.fmax.reduceat(x, starts)", array: "x",
    types: [T("float64", kernel("fmax_reduce_f64", "once per run")), OTHER], typeText: "elements are float64",
    path: [
      { hook: "method", note: "reduceat is a method of the ufunc type, which every ufunc shares: the hook tells np.fmax from the others by the object it is called on.",
        checks: (c) => common(c, { layout: "x is one-dimensional and contiguous, with at least one element",
          given: ["starts is a one-dimensional int64 array, not a list, with every start inside x; no dtype or out", "x holds no signalling NaN (it is read once to check)"] }) },
    ],
  },
  { // hook_ufunc.c call_lane.
    family: ELEM, code: "a + b", array: "a", plainText: "a and b are exactly ndarrays, not subclasses", detour: "3 to 4 ns",
    types: [T("float64", LANE), T("float32", LANE), T("int64", LANE), T("float16", null), T("int64 + float64", null)],
    typeText: "a and b share an element type among bool, the integer types, float32 and float64: NumPy's loop needs no conversion",
    path: [
      { hook: "ufunc", checks: (c) => common(c, { layout: "a and b are contiguous and of one shape, or b is a row of a",
        size: { text: `at most ${LANE_MAX_N.toLocaleString("en-US")} elements: above that NumPy's call path is under 5% of the call`, ok: (n) => n <= LANE_MAX_N },
        given: LANE_GIVEN }) },
    ],
  },
  { // hook_ufunc.c call_lane, with a number as the second operand.
    family: ELEM, code: "a * 2.0", array: "a", detour: "3 to 4 ns",
    types: [T("float64", LANE), T("float32", LANE), T("int64", null)],
    typeText: "NumPy's loop needs no conversion of a: float64 and float32 multiply by a Python float as they are, int64 would be converted to float64",
    path: [
      { hook: "ufunc", checks: (c) => common(c, { layout: "a is contiguous (any shape)",
        size: { text: `at most ${LANE_MAX_N.toLocaleString("en-US")} elements: above that NumPy's call path is under 5% of the call`, ok: (n) => n <= LANE_MAX_N },
        given: ["the number is a Python float or int, or a NumPy number of a's type, and converts to a's type exactly", ...LANE_GIVEN] }) },
    ],
  },
  { // hook_ufunc.c call_lane; below it hook_loop.c, the in-place subtract loop, ARM64 only.
    family: ELEM, code: "w -= b", array: "w", plainText: "w and b are exactly ndarrays, not subclasses", detour: "3 to 4 ns",
    types: [T("float64", LANE, { inplace: "subtract_inplace_f64" }), T("float32", LANE), OTHER],
    typeText: "w and b share an element type the lane takes: here float64 or float32",
    extra: { label: "Machine", options: [{ label: "Apple Silicon", ok: true, arm: true }, { label: "x86-64", ok: true }] },
    path: [
      { hook: "ufunc", checks: (c) => common(c, { layout: "w and b are contiguous and of one shape",
        size: { text: `at most ${LANE_MAX_N.toLocaleString("en-US")} elements: above that NumPy's call path is under 5% of the call`, ok: (n) => n <= LANE_MAX_N },
        given: ["the output is w itself, the first input"] }) },
      { hook: "loop", note: LOOP_NOTE, applies: (c) => Boolean(c.type.inplace),
        via: "The lane passes the call on. NumPy runs its subtract ufunc, which calls its loop for float64.",
        checks: (c) => [
          check("w, b and the output step by 8 bytes: contiguous", c.layoutOk),
          check("the output is w itself, the first input", true),
          check(`at least ${INPLACE_MIN_N.toLocaleString("en-US")} elements (1 MB)`, c.n >= INPLACE_MIN_N),
          check("an Apple Silicon machine: on x86-64 the loop is not built", Boolean(c.extra.arm)),
        ],
        serve: (c) => builtin(c.type.inplace, "prefetching the output for a store") },
    ],
  },
  { // hook_bottleneck.c call_nanmean and the column kernels.
    family: BN, code: "bn.nanmean(a, axis=0)", array: "a", sizes: false, fallback: "bottleneck",
    types: [T("float64", kernel("nanmean_cols_f64")), OTHER], typeText: "elements are float64",
    layouts: [{ label: "C order", ok: true }, { label: "rows skipped, a[::2]", ok: true }, { label: "columns skipped, a[:, ::2]", ok: false }],
    path: [
      { hook: "bottleneck", checks: (c) => common(c, { layout: "a is two-dimensional, and the axis that is kept has its elements next to each other: axis=0 of a C-ordered array or of a rolling-window view",
        given: ["axis is a Python int, by position or by name, and there is no other argument", "bottleneck was imported, before install() or after it"] }) },
    ],
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
.rt-select { max-width: 100%; }
`;

// The layouts a statement offers.
function layoutsOf(statement) {
  return statement.layouts || [{ label: "contiguous", ok: true }, { label: `strided, ${statement.array}[::2]`, ok: Boolean(statement.anyLayout) }];
}

// What a kernel is written in, on each architecture.
function written(name) {
  const where = C_KERNELS[name];
  if (!where) return "assembly";
  return where === "both" ? "C" : "C on ARM64, assembly on x86-64";
}

// Works out the path a call takes. Returns the boxes of the figure in order.
function route(statement, state) {
  const type = statement.types[state.type];
  const extra = statement.extra ? statement.extra.options[state.extra] : null;
  const c = {
    statement, type, extra,
    plain: state.kind === "plain",
    layoutOk: layoutsOf(statement)[state.layout].ok,
    n: SIZES[state.size],
    typeOk: (type.serve !== null || type.k !== undefined) && (!type.serve || !type.serve.inexact || Boolean(extra && extra.inexact)),
  };
  const steps = [{ box: "statement", code: statement.code }];
  for (const [i, stage] of statement.path.entries()) {
    if (stage.applies && !stage.applies(c)) continue;
    if (stage.via) steps.push({ box: "wrapper", title: i === 0 ? "NumPy's Python wrapper" : "NumPy's own code", text: stage.via });
    const checks = stage.checks(c);
    steps.push({ box: "hook", hook: stage.hook, checks, note: stage.note });
    if (checks.every((x) => x.ok)) {
      if (stage.skips) steps.push({ box: "wrapper", skipped: true, title: "NumPy's Python wrapper", text: SKIPPED });
      steps.push({ box: "serve", ...(stage.serve ? stage.serve(c) : type.serve), bottleneck: statement.fallback === "bottleneck" });
      return steps;
    }
  }
  steps.push({ box: "fallback", bottleneck: statement.fallback === "bottleneck", detour: statement.detour || "about 1 ns" });
  return steps;
}

function drawServe(step) {
  const how = step.how ? `, ${step.how}` : "";
  if (step.kind === "lane") {
    return h("div", { class: "rt-box rt-out kernel" },
      h("div", { class: "title" }, "NumPy's own inner loop, called directly"),
      h("p", {}, "The loop NumPy would have run, asked for once for this ufunc and these operand types and kept. The binding allocates the output and calls the loop with NumPy's counts and strides, so the result is NumPy's by construction. Floating-point errors go to NumPy's own handler, so warnings and np.errstate are unchanged."));
  }
  if (step.kind === "builtin") {
    return h("div", { class: "rt-box rt-out kernel" },
      h("div", { class: "title" }, "Loop built into hotkernels: ", h("code", {}, step.name), ` (C${how})`),
      h("p", {}, "Plain C inside the binding, not one of the judged kernels. delivery check compares it with NumPy like the rest of the binding: the result is NumPy's, bit for bit."));
  }
  const exact = step.bottleneck
    ? "The result is bottleneck's own, bit for bit."
    : step.inexact
      ? "The result has NumPy's type and shape and is within the kernel's declared tolerance of NumPy's values, not bit-identical."
      : "The result is the one NumPy returns: the same type, shape and bits.";
  return h("div", { class: "rt-box rt-out kernel" },
    h("div", { class: "title" }, "Kernel: ", h("code", {}, step.name), ` (${written(step.name)}${how})`),
    h("p", {}, exact));
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
  if (step.box === "serve") return drawServe(step);
  return h("div", { class: "rt-box rt-out numpy" },
    h("div", { class: "title" }, step.bottleneck ? "bottleneck's own code, as before" : "NumPy's own code, as before"),
    h("p", {}, `The call goes on with the caller's own arguments. Cost of the detour: ${step.detour}.`));
}

export default function mount(el) {
  const state = { statement: 0, type: 0, layout: 0, kind: "plain", size: 1, extra: 0 };

  // The statements, grouped by family, each option's value its index in STATEMENTS.
  const families = [...new Set(STATEMENTS.map((s) => s.family))];
  const select = h("select", { class: "w-select rt-select", "aria-label": "Statement",
    onchange: () => { state.statement = Number(select.value); state.type = 0; state.layout = 0; state.extra = 0; controls(); draw(); } },
  families.map((family) => h("optgroup", { label: family },
    STATEMENTS.map((s, i) => (s.family === family ? h("option", { value: i }, s.code) : null)))));
  const choices = h("div", { class: "rt-controls" });
  const path = h("div", { class: "rt-path", "aria-live": "polite" });

  function row(label, options, value, set) {
    return h("div", { class: "w-controls" }, h("span", { class: "w-label" }, label),
      seg(options, value, (v) => { set(v); draw(); }));
  }

  // The choices depend on the statement: its element types, its layouts, and its extra question if it has one.
  function controls() {
    const statement = STATEMENTS[state.statement];
    choices.replaceChildren(
      row("Element type", statement.types.map((t, i) => ({ value: i, label: t.label })), state.type, (v) => { state.type = Number(v); }),
      row("Layout", layoutsOf(statement).map((l, i) => ({ value: i, label: l.label })), state.layout, (v) => { state.layout = Number(v); }),
      row("Kind", [{ value: "plain", label: "plain ndarray" }, { value: "subclass", label: "a subclass" }], state.kind, (v) => { state.kind = v; }),
    );
    if (statement.sizes !== false) {
      choices.append(row("Size", SIZES.map((n, i) => ({ value: i, label: n.toLocaleString("en-US") })), state.size, (v) => { state.size = Number(v); }));
    }
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
    h("p", { class: "w-note" }, "A call no hook takes is passed on with the caller's own arguments; nothing is converted or copied on the way. Checks that only read data, such as a mask's bytes, are shown as met: the figure does not describe the values."),
  );
  controls();
  draw();
}
