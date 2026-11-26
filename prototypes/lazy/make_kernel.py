"""Writes a fused kernel from the name of its shape: the spec, the C reference, and the NEON assembly.

    uv run python -m prototypes.lazy.make_kernel "sub(V0,mul(S0,V1))"
    uv run python -m prototypes.lazy.make_kernel --all

A shape's name says everything a kernel for it has to do: which operations,
on which operands, in which order. `sub(V0,mul(S0,V1))` is `a - s0 * b`
for two arrays and a number. The loop for it is always the same loop: load
eight elements of each array, make the operations in the order the tree
gives, each one instruction that rounds once, and store. So it is written
by a program, here, and not by hand; what a person would add afterwards is
tuning, which goes in the kernel's `attempts.md` like any other attempt.

A kernel written here is a kernel like the others. It gets a directory
under `kernels/` with a `spec.py` whose oracle is the shape as a NumPy
expression, a `naive.c`, and `arm64/kernel.s`; `uv run judge check <name>`
holds it to NumPy's bits; and `fuse.py` reads the pattern it serves off its
oracle, as it does for every kernel. Nothing here is trusted: the judge is.

`--all` writes every shape of SHAPES that has no directory yet. The first
of them are the single operations, `add(V0,V1)` and the like. They fuse
nothing by themselves; they are what lets a whole expression be computed
block by block (`_fused.program`), with a kernel for every step.

The loop:

- **Eight elements a step**, four vectors of two, as the kernels written by
  hand. Each array is loaded into four registers, each operation is four
  instructions, and the result is stored from four.
- **Every load of a step comes before its store.** So the result may be
  written where one of the arrays is, for a length that is a multiple of
  eight or below eight; `_fused.c` makes its calls so.
- **A remainder of 1 to 7 elements** is one more step that ends with the
  arrays and overlaps the step before. Fewer than 8 are done one at a time.
- **Registers** are given to the tree by the usual rule: the operand that
  needs more of them is computed first. A number is held in every lane of
  one register (`v31` down), and read from `d0` up in the one-at-a-time loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import fuse

log = logging.getLogger(__name__)

# The layer's directory, where its templates and its kernels are.
LAYER = Path(__file__).resolve().parent
KERNELS = LAYER / "kernels"
TEMPLATES = LAYER / "templates"
# The templates of the four files of a kernel.
SPEC_TEMPLATE, NAIVE_TEMPLATE = "kernel_spec.py.j2", "kernel_naive.c.j2"
ASSEMBLY_TEMPLATE, LOG_TEMPLATE = "kernel_arm64.s.j2", "kernel_attempts.md.j2"
# The architecture the assembly is written for.
ARCH = "arm64"
# How a kernel's name starts and ends: what `build_fused` and the judge find it by, and its element type.
PREFIX, SUFFIX = "fuse_", "_f64"

@dataclass(frozen=True)
class Operation:
    """One operation a kernel can make."""

    # The instruction, for vectors of two float64 and for one alone.
    instruction: str
    # How many operands it reads.
    operands: int
    # The operation in a NumPy expression and in C, with `{0}` and `{1}` for its operands.
    python: str
    c: str
    # What it is called in a sentence.
    word: str


# The operations a kernel can be written for, by their names in a shape. A square is a product with itself.
OPERATIONS = {
    "add": Operation("fadd", 2, "({0} + {1})", "({0} + {1})", "sum"),
    "sub": Operation("fsub", 2, "({0} - {1})", "({0} - {1})", "difference"),
    "mul": Operation("fmul", 2, "({0} * {1})", "({0} * {1})", "product"),
    "div": Operation("fdiv", 2, "({0} / {1})", "({0} / {1})", "quotient"),
    "neg": Operation("fneg", 1, "(-{0})", "(-{0})", "negation"),
    "abs": Operation("fabs", 1, "np.abs({0})", "__builtin_fabs({0})", "absolute value"),
    "sqrt": Operation("fsqrt", 1, "np.sqrt({0})", "__builtin_sqrt({0})", "square root"),
    "sq": Operation("fmul", 1, "np.square({0})", "({0} * {0})", "square"),
    # `np.maximum` and `np.minimum` are the processor's `fmax` and `fmin` on ARM64, at every length and position:
    # a NaN wins over a number, and of two zeros the maximum is the positive one and the minimum the negative. C's
    # `fmax` is another function (a number wins over a NaN), so the reference spells the rule out; see the template.
    "max": Operation("fmax", 2, "np.maximum({0}, {1})", "larger({0}, {1})", "maximum"),
    "min": Operation("fmin", 2, "np.minimum({0}, {1})", "smaller({0}, {1})", "minimum"),
}
# The names of a kernel's arrays, in the order of the shape's V0, V1, ...; and how its numbers are named.
ARRAY_NAMES = "abcdef"
NUMBER_NAME = "s{}"
# The default value of each number, none a power of two, so that a product or a quotient with one is rounded.
NUMBER_DEFAULTS = (0.37, -1.9, 2.3, 0.61, 1.7, -0.43, 3.1, 0.29)
# The values every number is also tried at, by the name of the set: the ones where an operation's result changes
# in kind (a zero of either sign, an overflow, a denormal, an infinity, a NaN).
NUMBER_TRIALS = (("zero", "0.0"), ("negative_zero", "-0.0"), ("huge", "HUGE"), ("tiny", "TINY"),
                 ("infinite", 'float("inf")'), ("nan", 'float("nan")'))
# Elements a step of the loop computes, and vectors of two that is.
BLOCK, LANES = 8, 4
# The first vector register that holds a number in every lane; the next number takes the one below.
TOP_REGISTER = 31
# The vector registers a kernel may use without saving: v0 to v7 and v16 to v31. v8 to v15 are the caller's.
FREE_LOW, FREE_HIGH = range(0, 8), range(16, 32)
# The most arrays a kernel reads: `_fused.c` passes eight addresses and lengths, two of them the result and the length.
MOST_ARRAYS = 6

# The shapes `--all` writes. First the single operations, which every expression is made of; then the shapes of
# two to six operations that the corpus's expressions are covered best by, as `programs/mine.py` chose them one at
# a time, less those that have a kernel written by hand and those with an operation this tool has no rule for.
# One more is left out, `div(add(S0,neg(abs(V0))),S1)`: the compiler turns `s0 + (-|a|)` of its C reference into
# `s0 - |a|`, which gives a NaN the other sign, so the reference is not NumPy's and the judge refuses the kernel.
SHAPES = (
    "add(V0,V1)", "add(S0,V0)", "sub(V0,V1)", "sub(S0,V0)", "sub(V0,S0)", "mul(V0,V1)", "mul(S0,V0)",
    "div(V0,V1)", "div(S0,V0)", "div(V0,S0)", "neg(V0)", "abs(V0)", "sqrt(V0)", "sq(V0)",
    "add(V0,mul(V1,mul(mul(V2,mul(S0,V3)),sq(V4))))", "sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3)))",
    "mul(V0,mul(S0,V1))", "add(sq(V0),sq(V1))", "add(V0,add(S0,V1))",
    "div(sub(V0,V1),S0)", "sub(sub(V0,mul(V1,V2)),mul(V3,V4))", "add(mul(S0,add(V0,V1)),sub(V2,mul(S1,sub(V3,V4))))",
    "mul(V0,add(V1,sub(V2,V3)))", "add(add(S0,V0),mul(V1,add(V2,mul(V3,add(S1,V4)))))", "add(mul(V0,V1),mul(V2,V3))",
    "sub(V0,sq(div(sub(V1,V2),S0)))", "mul(S0,add(V0,sub(V1,mul(S1,V2))))", "sub(add(V0,mul(V1,V2)),mul(V3,mul(V4,V5)))",
    "sub(mul(V0,V1),mul(V2,V3))", "mul(V0,add(S0,mul(S1,V1)))", "div(mul(V0,sub(V1,V2)),S0)", "sub(add(S0,V0),V1)",
    "div(sub(V0,S0),S1)", "mul(S0,add(S1,V0))", "mul(V0,add(V1,V2))", "div(sub(V0,V1),V2)", "div(mul(V0,sub(V1,V2)),V3)",
    "add(V0,add(V1,add(V2,add(V3,add(V4,V5)))))", "div(mul(S0,V0),V1)", "div(S0,sqrt(V0))", "mul(V0,div(mul(S0,V1),S1))",
    "sqrt(add(sq(V0),sq(V1)))", "div(neg(V0),V1)", "add(add(S0,V0),mul(S1,V1))", "add(V0,sub(V1,V2))",
    "add(V0,mul(S0,add(sq(V1),sq(V2))))", "sub(mul(S0,V0),mul(S1,V1))", "add(V0,mul(V1,mul(S0,V2)))",
    "add(add(S0,V0),mul(V1,add(V2,add(S1,V3))))", "add(div(V0,S0),div(V1,S1))", "sub(V0,mul(V1,mul(S0,V2)))",
    "mul(V0,mul(V1,add(V2,mul(V3,V4))))", "mul(add(V0,V1),mul(S0,V2))", "add(add(S0,V0),mul(V1,add(S1,mul(S2,V2))))",
    "sub(mul(S0,add(V0,div(sub(V1,V2),S1))),sq(V3))", "div(mul(S0,V0),S1)", "add(V0,add(V1,add(V2,V3)))",
    "max(V0,V1)", "max(S0,V0)", "min(V0,V1)", "min(S0,V0)", "div(neg(add(V0,mul(V1,V2))),sub(min(S0,V3),S1))",
    "add(V0,mul(max(S0,mul(V1,V2)),mul(V3,V4)))", "max(S0,max(V0,min(S1,V1)))",
    # The two halves of an advection flux, four operations and eight: written whole because a program of six
    # kernel calls for its thirteen operations took twice what one loop takes (docs/lazy-fusion-design.md).
    "mul(S0,mul(add(V0,V1),mul(S1,V2)))", "mul(S0,mul(V0,mul(abs(mul(S1,V1)),add(mul(V2,V3),sub(S2,V4)))))",
)

app = typer.Typer(add_completion=False)


class CannotWrite(ValueError):
    """A shape this tool writes no kernel for: an operation without a rule here, or too many arrays or registers."""


@dataclass
class _Registers:
    """The vector registers of one loop, handed out in groups: four for a step of eight elements, one for one element."""

    # Registers not in use, in the order they are handed out.
    free: list[int]
    # How many registers hold one value: LANES in the loop, 1 one element at a time.
    width: int

    def take(self) -> tuple[int, ...]:
        """Hand out the registers for one value.

        Raises:
            CannotWrite: When the shape needs more values at once than there are registers for.
        """
        if len(self.free) < self.width:
            raise CannotWrite("the shape needs more registers at once than a kernel may use")
        taken, self.free = tuple(self.free[:self.width]), self.free[self.width:]
        return taken

    def give_back(self, registers: tuple[int, ...]) -> None:
        """Take back the registers of a value that has been read for the last time."""
        self.free = list(registers) + self.free


@dataclass
class _Loop:
    """The instructions of one loop's body, as they are written: `(mnemonic, operands)`."""

    # 1 for the loop of eight elements, 0 for the loop of one: which form each instruction has.
    wide: bool
    # The addresses of the arrays, by the array's number, and of the result: register numbers.
    arrays: list[int]
    out: int
    registers: _Registers
    lines: list[tuple[str, str]] = field(default_factory=list)

    def name(self, register: int) -> str:
        """Write a register as an operand of an arithmetic instruction."""
        return f"v{register}.2d" if self.wide else f"d{register}"

    def load(self, array: int) -> tuple[int, ...]:
        """Load the next elements of an array into new registers.

        Args:
            array: The array's number in the shape.

        Returns:
            The registers that hold them.
        """
        held = self.registers.take()
        address = self.arrays[array]
        if self.wide:
            self.lines.append(("ldp", f"q{held[0]}, q{held[1]}, [x{address}]"))
            self.lines.append(("ldp", f"q{held[2]}, q{held[3]}, [x{address}, #32]"))
        else:
            self.lines.append(("ldr", f"d{held[0]}, [x{address}], #8"))
        return held

    def store(self, held: tuple[int, ...]) -> None:
        """Store a value as the result."""
        if self.wide:
            self.lines.append(("stp", f"q{held[0]}, q{held[1]}, [x{self.out}]"))
            self.lines.append(("stp", f"q{held[2]}, q{held[3]}, [x{self.out}, #32]"))
        else:
            self.lines.append(("str", f"d{held[0]}, [x{self.out}], #8"))

    def number(self, place: int) -> tuple[int, ...]:
        """Return the registers a number is read from: one register in every position."""
        register = TOP_REGISTER - place if self.wide else place
        return (register,) * self.registers.width

    def operate(self, instruction: str, target: tuple[int, ...], operands: list[tuple[int, ...]]) -> None:
        """Write one operation: an instruction for each register of the value.

        Args:
            instruction: The mnemonic.
            target: The registers the result goes to.
            operands: The registers of each operand, in the order the operation reads them.
        """
        for lane, register in enumerate(target):
            read = ", ".join(self.name(operand[lane]) for operand in operands)
            self.lines.append((instruction, f"{self.name(register)}, {read}"))


def _needs(pattern: fuse.Pattern) -> int:
    """Count the values that have to be held at once to compute a tree.

    Args:
        pattern: A shape, or part of one.

    Returns:
        0 for a number, which has a register of its own; 1 for an array;
        for an operation, what its operands need when the one that needs
        more is computed first.
    """
    if not pattern.op:
        return 1 if pattern.kind == fuse.VECTOR else 0
    needs = sorted((_needs(part) for part in pattern.parts), reverse=True)
    return max(max(needs), 1) if len(needs) == 1 or needs[0] != needs[1] else needs[0] + 1


def _compute(pattern: fuse.Pattern, loop: _Loop) -> tuple[tuple[int, ...], bool]:
    """Write the instructions that compute a tree.

    Args:
        pattern: A shape, or part of one.
        loop: The loop the instructions are written into.

    Returns:
        (the registers that hold the value, whether they are the loop's to
        write over: not for a number, whose register is read by every step).

    Raises:
        CannotWrite: For an operation without a rule here.
    """
    if not pattern.op:
        if pattern.kind == fuse.VECTOR:
            return loop.load(pattern.place), True
        return loop.number(pattern.place), False
    operation = OPERATIONS.get(pattern.op)
    if operation is None or operation.operands != len(pattern.parts):
        raise CannotWrite(f"no rule for the operation '{pattern.op}'")
    # The operand that needs more registers first; the instruction still reads them in the order written.
    order = sorted(range(len(pattern.parts)), key=lambda place: _needs(pattern.parts[place]), reverse=True)
    values: dict[int, tuple[tuple[int, ...], bool]] = {}
    for place in order:
        values[place] = _compute(pattern.parts[place], loop)
    held = [values[place] for place in range(len(pattern.parts))]
    own = [registers for registers, writable in held if writable]
    target = own[0] if own else loop.registers.take()
    read = [registers for registers, _ in held]
    loop.operate(operation.instruction, target, read * 2 if pattern.op == fuse.SQUARE else read)
    for registers in own[1:]:
        loop.registers.give_back(registers)
    return target, True


def _written(pattern: fuse.Pattern, form: str, names: list[str], numbers: list[str]) -> str:
    """Write a tree as an expression.

    Args:
        pattern: A shape, or part of one.
        form: "python" or "c": which of an operation's two forms to use.
        names: What each array is written as, by its number.
        numbers: The same for the numbers.

    Returns:
        The expression, in brackets unless it is one operand.
    """
    if not pattern.op:
        return names[pattern.place] if pattern.kind == fuse.VECTOR else numbers[pattern.place]
    operation = OPERATIONS[pattern.op]
    return getattr(operation, form).format(*(_written(part, form, names, numbers) for part in pattern.parts))


def _bare(expression: str) -> str:
    """Take the brackets off an expression that is one operation in brackets."""
    return expression[1:-1] if expression.startswith("(") and expression.endswith(")") else expression


def name_of(shape: str) -> str:
    """Give the kernel of a shape its name.

    Args:
        shape: A shape's name, such as "sub(V0,mul(S0,V1))".

    Returns:
        "fuse_", the shape read from left to right with `v` for an array
        and `s` for a number, and "_f64": `fuse_sub_v_mul_s_v_f64`. Two
        shapes never share one, since each operation has a fixed number of
        operands.
    """
    words = []
    for word in shape.replace("(", ",").replace(")", "").split(","):
        words.append(word[0].lower() if word[0] in (fuse.VECTOR, fuse.SCALAR) and word[1:].isdigit() else word)
    return PREFIX + "_".join(words) + SUFFIX


def _count(pattern: fuse.Pattern, kind: str) -> int:
    """Count the leaves of one kind in a tree, or its operations for an empty kind."""
    if not pattern.op:
        return int(pattern.kind == kind)
    return int(not kind) + sum(_count(part, kind) for part in pattern.parts)


def write(shape: str, force: bool = False) -> Path | None:
    """Write the four files of the kernel for one shape.

    Args:
        shape: The shape's name, as `fuse.whole_shape` writes one.
        force: Whether to write over a directory that is there.

    Returns:
        The kernel's directory, or None when it was there and `force` is not set.

    Raises:
        CannotWrite: For a shape this tool writes no kernel for.
    """
    pattern = fuse.parse(shape)
    arrays, numbers, operations = _count(pattern, fuse.VECTOR), _count(pattern, fuse.SCALAR), _count(pattern, "")
    if not operations or not arrays:
        raise CannotWrite("a shape has an operation and reads an array")
    if arrays > MOST_ARRAYS or numbers > len(NUMBER_DEFAULTS):
        raise CannotWrite(f"a kernel reads at most {MOST_ARRAYS} arrays and {len(NUMBER_DEFAULTS)} numbers")
    name = name_of(shape)
    directory = KERNELS / name
    if directory.exists() and not force:
        return None
    array_names = list(ARRAY_NAMES[:arrays])
    number_names = [NUMBER_NAME.format(place) for place in range(numbers)]
    out, length = arrays, arrays + 1  # The registers of the result's address and of the length: after the arrays'.
    held_by_numbers = set(range(numbers)) | {TOP_REGISTER - place for place in range(numbers)}
    free = [register for register in (*FREE_HIGH, *FREE_LOW) if register not in held_by_numbers]
    loops = {}
    for wide in (True, False):
        loop = _Loop(wide, list(range(arrays)), out, _Registers(list(free), LANES if wide else 1))
        value, _ = _compute(pattern, loop)
        loop.store(value)
        loops[wide] = loop.lines
    python = _bare(_written(pattern, "python", array_names, number_names))
    context = {
        "name": name, "shape": shape, "operations": operations, "arrays": array_names, "numbers": number_names,
        "python": python, "c": _bare(_written(pattern, "c", [f"{each}[i]" for each in array_names], number_names)),
        "defaults": NUMBER_DEFAULTS[:numbers], "trials": NUMBER_TRIALS, "uses_numpy": "np." in python,
        "out": out, "length": length, "block": BLOCK, "top": TOP_REGISTER, "wide": loops[True], "narrow": loops[False],
        "words": sorted({OPERATIONS[word].word for word in OPERATIONS if f"{word}(" in shape}),
        "larger": "max(" in shape, "smaller": "min(" in shape,
    }
    environment = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                              lstrip_blocks=True, keep_trailing_newline=True)
    (directory / ARCH).mkdir(parents=True, exist_ok=True)
    for template, path in ((SPEC_TEMPLATE, directory / "spec.py"), (NAIVE_TEMPLATE, directory / "naive.c"),
                           (ASSEMBLY_TEMPLATE, directory / ARCH / "kernel.s"),
                           (LOG_TEMPLATE, directory / ARCH / "attempts.md")):
        path.write_text(environment.get_template(template).render(**context))
    return directory


@app.command()
def main(shape: list[str] = typer.Argument(None, help="A shape to write the kernel of, such as 'sub(V0,mul(S0,V1))'."),
         every: bool = typer.Option(False, "--all", help="Write every shape of this tool's list that has no kernel."),
         force: bool = typer.Option(False, help="Write over a kernel's directory that is there.")) -> None:
    """Write the kernels of some shapes, each into its directory under `kernels/`.

    Args:
        shape: The shapes named on the command line.
        every: Whether to write the tool's own list as well.
        force: Whether to write over directories that are there.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    wanted = list(shape or ()) + (list(SHAPES) if every else [])
    if not wanted:
        log.error("name a shape, or give --all")
        raise typer.Exit(1)
    for each in wanted:
        try:
            directory = write(each, force)
        except CannotWrite as error:
            log.error("%s: %s", each, error)
            raise typer.Exit(1) from error
        if directory is None:
            log.info("%s is there: %s", each, name_of(each))
        else:
            log.info("wrote %s for %s; check it with: uv run judge check %s --unlocked", directory.name, each, directory.name)


if __name__ == "__main__":
    app()
