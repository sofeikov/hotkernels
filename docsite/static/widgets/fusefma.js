// Why a fused kernel never uses a multiply-add.
//
// NumPy computes c + a * b as two calls: the product, rounded to float64, then
// the sum, rounded again. A processor's fused multiply-add (fmla on ARM64)
// computes a * b + c exactly and rounds once. The one rounding is closer to
// the exact value, and it is not NumPy's: on ordinary inputs the two results
// often differ in the last bit, and where the sum cancels they can differ
// entirely. The figure computes both for numbers
// the reader picks or draws, and shows their bits.
//
// Nothing here is recorded: both results are computed in the page. The two
// roundings are JavaScript's own a * b then + c, which are IEEE float64
// operations rounded to nearest, as NumPy's are. The single rounding is exact
// arithmetic on the numbers' binary values with BigInt, rounded once to
// nearest, ties to even: what fma(a, b, c) gives. The three presets were
// found with Python, comparing a * b + c with float(Fraction(a) * Fraction(b)
// + Fraction(c)). The running count of draws that differ is the page's own.
//
// Attributes:
//   data-preset  scaled (default), gradient or cancel

import { h, seg, stat, button, rng } from "./util.js";

const PRESETS = {
  scaled: { label: "a + s * b", names: ["s", "b", "a"], sign: 1, values: ["0.1", "0.1", "0.1"],
    note: "The multiply-add's answer is the one that prints as 0.11, and it is closer to the exact value. It is still not NumPy's: <code>a + s * b</code> in NumPy is two calls, and the program that wrote it gets 0.11000000000000001. A kernel that gave the other would be a different program." },
  gradient: { label: "w - rate * g", names: ["rate", "g", "w"], sign: -1, values: ["0.1", "0.3", "0.3"],
    note: "A gradient step. The multiply-add, <code>fmls</code> here, gives 0.26999999999999996 where NumPy's two calls give 0.27. One step is one bit; a training loop repeats it on every weight at every step, and its result would drift away from NumPy's." },
  cancel: { label: "a * b - 1", names: ["a", "b", "c"], sign: 1, values: ["1.0000000009313226", "0.9999999990686774", "-1"],
    note: "a = 1 + 2⁻³⁰ and b = 1 − 2⁻³⁰, so a × b = 1 − 2⁻⁶⁰ exactly. Rounded to float64 that is 1.0, and NumPy's sum is 0. The multiply-add keeps the 2⁻⁶⁰ and gives −8.67 × 10⁻¹⁹. Where a sum cancels, the two are not one bit apart: one is zero and the other is not." },
};

export const css = `
.ff-inputs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .6rem; margin-bottom: .9rem; }
@media (max-width: 560px) { .ff-inputs { grid-template-columns: minmax(0, 1fr); } }
.ff-inputs label { display: grid; gap: .2rem; font-size: .74rem; color: var(--muted); font-weight: 650; }
.ff-inputs input { font-family: var(--mono); font-size: .82rem; padding: .35rem .5rem; border-radius: 7px; border: 1px solid var(--rule-strong); background: var(--bg-raised); color: var(--ink); min-width: 0; }
.ff-expr { font-family: var(--mono); font-size: .86rem; margin: 0 0 .7rem; }
.ff-rows { display: grid; gap: .7rem; }
.ff-row { border: 1px solid var(--rule); border-radius: 10px; padding: .55rem .7rem; background: var(--bg); }
.ff-row.np { border-color: var(--np); }
.ff-row.fma { border-color: var(--bad); }
.ff-head { display: flex; flex-wrap: wrap; gap: .2rem .8rem; align-items: baseline; margin-bottom: .35rem; }
.ff-head b { font-size: .8rem; }
.ff-row.np .ff-head b { color: var(--np); }
.ff-row.fma .ff-head b { color: var(--bad); }
.ff-head .ins { font-family: var(--mono); font-size: .72rem; color: var(--muted); }
.ff-val { font-family: var(--mono); font-size: .9rem; font-weight: 650; word-break: break-all; }
.ff-bits { display: flex; flex-wrap: wrap; gap: 2px 8px; margin-top: .4rem; font-family: var(--mono); font-size: .7rem; line-height: 1.5; }
.ff-bits span.g { word-break: break-all; }
.ff-bits i { font-style: normal; color: var(--muted); }
.ff-bits i.diff { color: var(--bad); background: var(--bad-soft); font-weight: 700; border-radius: 2px; }
.ff-bits .lab { color: var(--muted); font-family: var(--sans); font-size: .64rem; letter-spacing: .05em; text-transform: uppercase; align-self: center; }
.ff-exact { font-family: var(--mono); font-size: .78rem; color: var(--muted); margin-top: .7rem; word-break: break-all; }
`;

// ---- float64 values as integers, exactly. ----

const view = new DataView(new ArrayBuffer(8));
const MANTISSA_BITS = 52n, EXPONENT_BIAS = 1075, SUBNORMAL_EXP = -1074;

// The 64 bits of a float64, as a BigInt.
function bitsOf(x) {
  view.setFloat64(0, x);
  return view.getBigUint64(0);
}

// The float64 whose bits these are.
function fromBits(bits) {
  view.setBigUint64(0, bits);
  return view.getFloat64(0);
}

// 2 ** e as a float64, built from its bits, for -1074 <= e <= 1023; Infinity above.
function pow2(e) {
  if (e > 1023) return Infinity;
  if (e >= -1022) return fromBits(BigInt(e + 1023) << MANTISSA_BITS);
  return fromBits(1n << BigInt(e - SUBNORMAL_EXP));
}

// A finite float64 as sign * m * 2 ** e, with m a non-negative BigInt.
function split(x) {
  const bits = bitsOf(x);
  const negative = bits >> 63n === 1n;
  const exponent = Number((bits >> MANTISSA_BITS) & 0x7ffn);
  const fraction = bits & ((1n << MANTISSA_BITS) - 1n);
  const m = exponent === 0 ? fraction : fraction | (1n << MANTISSA_BITS);
  const e = exponent === 0 ? SUBNORMAL_EXP : exponent - EXPONENT_BIAS;
  return { m: negative ? -m : m, e };
}

// The exact value of sign * a * b + c, as an integer S and exponent E: S * 2 ** E.
function exactSum(a, b, c, sign) {
  const A = split(a), B = split(b), C = split(c);
  const pe = A.e + B.e;
  const low = Math.min(pe, C.e);
  const product = BigInt(sign) * A.m * B.m * (1n << BigInt(pe - low));
  return { S: product + C.m * (1n << BigInt(C.e - low)), E: low };
}

// The number of binary digits of a positive BigInt.
const bitLength = (n) => n.toString(2).length;

// fma(sign * a, b, c): the exact value rounded once to the nearest float64, ties to even.
function fused(a, b, c, sign) {
  if (![a, b, c].every(Number.isFinite)) return c + sign * a * b;
  const { S, E } = exactSum(a, b, c, sign);
  // An exact zero: a * b is then representable, so the two roundings give the same zero, with its sign.
  if (S === 0n) return c + sign * a * b;
  const negative = S < 0n;
  const M = negative ? -S : S;
  const lowest = Math.max(E + bitLength(M) - 53, SUBNORMAL_EXP);
  let value;
  if (lowest <= E) {
    value = Number(M) * pow2(E);
  } else {
    const k = BigInt(lowest - E);
    let q = M >> k;
    const rest = M - (q << k);
    const half = 1n << (k - 1n);
    if (rest > half || (rest === half && (q & 1n) === 1n)) q += 1n;
    value = Number(q) * pow2(lowest);
  }
  return negative ? -value : value;
}

// The exact value of S * 2 ** E to `digits` significant decimal digits, cut, not rounded.
function exactDecimal(S, E, digits = 25) {
  if (S === 0n) return "0";
  const negative = S < 0n;
  let num = negative ? -S : S, den = 1n;
  if (E >= 0) num <<= BigInt(E); else den <<= BigInt(-E);
  // The power of ten of the leading digit.
  let e10 = num.toString().length - den.toString().length;
  const atLeast = (p) => (p >= 0 ? num >= den * 10n ** BigInt(p) : num * 10n ** BigInt(-p) >= den);
  while (!atLeast(e10)) e10 -= 1;
  while (atLeast(e10 + 1)) e10 += 1;
  const p = digits - 1 - e10;
  const q = p >= 0 ? (num * 10n ** BigInt(p)) / den : num / (den * 10n ** BigInt(-p));
  const text = q.toString();
  return `${negative ? "−" : ""}${text[0]}.${text.slice(1)}…e${e10 >= 0 ? "+" : ""}${e10}`;
}

// The bits of x as sign, exponent and fraction, with the bits that differ from `other` marked.
function bitsView(x, other) {
  const mine = bitsOf(x).toString(2).padStart(64, "0");
  const theirs = bitsOf(other).toString(2).padStart(64, "0");
  const group = (from, to) => h("span", { class: "g" }, ...[...mine.slice(from, to)].map((bit, i) =>
    h("i", { class: bit !== theirs[from + i] ? "diff" : "" }, bit)));
  return h("div", { class: "ff-bits" },
    h("span", { class: "lab" }, "sign"), group(0, 1),
    h("span", { class: "lab" }, "exponent"), group(1, 12),
    h("span", { class: "lab" }, "fraction"), group(12, 64));
}

// How many float64 values lie between two of one sign: the distance of their bits.
function ulpsApart(x, y) {
  if (Object.is(x, y)) return 0n;
  if (Math.sign(x) !== Math.sign(y) || x === 0 || y === 0) return null;
  const d = bitsOf(x) - bitsOf(y);
  return d < 0n ? -d : d;
}

// The shortest decimal that reads back as the same float64, as Python's repr gives it.
const show = (x) => (Object.is(x, -0) ? "-0.0" : Number.isInteger(x) && Math.abs(x) < 1e16 ? x.toFixed(1) : String(x));

export default async function mount(el) {
  let key = PRESETS[el.dataset.preset] ? el.dataset.preset : "scaled";
  let values = [...PRESETS[key].values];
  const draw = rng(298);
  let draws = 0, differed = 0;

  const inputs = [0, 1, 2].map((i) => h("input", { type: "text", inputmode: "decimal", spellcheck: "false", oninput: () => { values[i] = inputs[i].value; render(); } }));
  const labels = inputs.map((input) => h("label", {}, h("span", {}), input));
  const expr = h("p", { class: "ff-expr" });
  const rows = h("div", { class: "ff-rows" });
  const exact = h("p", { class: "ff-exact" });
  const note = h("p", { class: "w-note", "aria-live": "polite" });
  const verdict = h("p", { class: "w-verdict" });
  const apart = stat("float64 values apart", "bad");
  const drawn = stat("draws from [0, 1) so far");
  const share = stat("of them differ", "hot");
  const segment = seg(Object.entries(PRESETS).map(([value, p]) => ({ value, label: p.label })), key, (v) => choose(v));

  function choose(v) {
    key = v;
    values = [...PRESETS[key].values];
    render();
  }

  function randomDraw() {
    // Three numbers from [0, 1) with every bit of the fraction random: rng gives 32 bits, so two draws make one.
    values = [0, 1, 2].map(() => String(draw() + draw() * 2 ** -32));
    const [a, b, c] = values.map(Number);
    const p = PRESETS[key];
    draws += 1;
    if (!Object.is(c + p.sign * a * b, fused(a, b, c, p.sign))) differed += 1;
    render();
  }

  function render() {
    const p = PRESETS[key];
    labels.forEach((label, i) => {
      label.firstChild.textContent = p.names[i];
      if (document.activeElement !== inputs[i]) inputs[i].value = values[i];
    });
    const [a, b, c] = values.map((v) => Number(String(v).trim()));
    const [na, nb, nc] = p.names;
    expr.textContent = `${nc} ${p.sign < 0 ? "-" : "+"} ${na} * ${nb}`;
    drawn.set(String(draws));
    share.set(draws ? `${Math.round((100 * differed) / draws)}%` : "–");
    if (![a, b, c].every(Number.isFinite) || values.some((v) => String(v).trim() === "")) {
      rows.replaceChildren();
      exact.textContent = "";
      verdict.className = "w-verdict";
      verdict.textContent = "Type three finite numbers.";
      apart.set("–");
      note.innerHTML = p.note;
      return;
    }
    const twice = c + p.sign * a * b;
    const once = fused(a, b, c, p.sign);
    const product = a * b;
    rows.replaceChildren(
      h("div", { class: "ff-row np" },
        h("div", { class: "ff-head" }, h("b", {}, "NumPy, and every fused kernel: two roundings"),
          h("span", { class: "ins" }, `fmul, then ${p.sign < 0 ? "fsub" : "fadd"}`)),
        h("div", { class: "ff-val" }, show(twice)),
        h("div", { class: "ff-head", style: "margin: .25rem 0 0" }, h("span", { class: "ins" }, `${na} * ${nb} rounded first: ${show(product)}`)),
        bitsView(twice, once)),
      h("div", { class: "ff-row fma" },
        h("div", { class: "ff-head" }, h("b", {}, "A fused multiply-add: one rounding"),
          h("span", { class: "ins" }, p.sign < 0 ? "fmls (fmsub)" : "fmla (fmadd)")),
        h("div", { class: "ff-val" }, show(once)),
        bitsView(once, twice)));
    const { S, E } = exactSum(a, b, c, p.sign);
    exact.textContent = `exact ${nc} ${p.sign < 0 ? "-" : "+"} ${na} × ${nb} = ${exactDecimal(S, E)}`;
    const same = Object.is(twice, once);
    verdict.className = `w-verdict ${same ? "good" : "bad"}`;
    const distance = ulpsApart(twice, once);
    verdict.textContent = same
      ? "The same bits this time. A kernel still may not use a multiply-add: it has to be NumPy's on every input, not on most."
      : distance === null ? "Different numbers altogether, not one bit apart." : `Different bits: ${distance} float64 value${distance === 1n ? "" : "s"} apart.`;
    apart.set(same ? "0" : distance === null ? "∞" : distance.toString());
    note.innerHTML = p.note;
  }

  el.append(
    h("div", { class: "w-controls" }, h("span", { class: "w-label" }, "Example"), segment,
      button("Draw three numbers", () => randomDraw())),
    // In the order the expression is written: the term added first, then the two factors.
    h("div", { class: "ff-inputs" }, labels[2], labels[0], labels[1]),
    expr, rows, exact, verdict, note,
    h("div", { class: "w-readout" }, apart, drawn, share),
  );
  render();
}
