"""Writes a fused kernel from the name of its shape: the spec, the C reference, and the NEON and AVX2 assembly.

    uv run python -m prototypes.lazy.make_kernel "sub(V0,mul(S0,V1))"
    uv run python -m prototypes.lazy.make_kernel --all
    uv run python -m prototypes.lazy.make_kernel --arch x86_64 --all

A shape's name says everything a kernel for it has to do: which operations,
on which operands, in which order. `sub(V0,mul(S0,V1))` is `a - s0 * b`
for two arrays and a number. The loop for it is always the same loop: load
eight elements of each array, make the operations in the order the tree
gives, each one instruction that rounds once, and store. So it is written
by a program, here, and not by hand; what a person would add afterwards is
tuning, which goes in the kernel's `attempts.md` like any other attempt.

A kernel written here is a kernel like the others. It gets a directory
under `kernels/` with a `spec.py` whose oracle is the shape as a NumPy
expression, a `naive.c`, `arm64/kernel.s` and `x86_64/kernel.s`; `uv run judge check <name>`
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

On x86-64 the loop is the same, eight elements a step in two ymm registers
of four, with every address an array's base plus one index in `rax`; a
number is held in `ymm15` down. `--arch x86_64` writes that loop alone for
kernels that are there, reading each one's shape off its oracle as `fuse.py`
does, so the kernels written by hand for ARM64 get one too.
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
# The architecture the assembly above is written for.
ARCH = "arm64"
# The name of an architecture's log beside its assembly.
LOG_NAME = "attempts.md"
# The same for x86-64: its architecture's directory and its two templates.
X86_ARCH = "x86_64"
X86_ASSEMBLY_TEMPLATE, X86_LOG_TEMPLATE = "kernel_x86_64.s.j2", "kernel_attempts_x86_64.md.j2"
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
    # `fmax` is another function (a number wins over a NaN), so the reference spells the rule out; see the template,
    # which has x86-64's rule as well (X86_OPERATIONS).
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
                 ("infinite", 'float("inf")'), ("nan", "DEFAULT_NAN"))
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

    def arity(self, op: str) -> int | None:
        """Return how many operands an operation reads, or None for one this loop has no rule for."""
        operation = OPERATIONS.get(op)
        return None if operation is None else operation.operands

    def operate(self, op: str, target: tuple[int, ...], operands: list[tuple[int, ...]]) -> None:
        """Write one operation: an instruction for each register of the value.

        Args:
            op: The operation's name in the shape, a key of OPERATIONS.
            target: The registers the result goes to.
            operands: The registers of each operand, in the order the operation reads them.
        """
        instruction = OPERATIONS[op].instruction
        for lane, register in enumerate(target):
            read = ", ".join(self.name(operand[lane]) for operand in operands)
            self.lines.append((instruction, f"{self.name(register)}, {read}"))


@dataclass(frozen=True)
class X86Operation:
    """How the x86-64 loops make one operation: the forms of its instruction and the way its operands are laid out."""

    # The instruction on four float64 in a ymm register, and on one in the low lane of an xmm register.
    wide: str
    narrow: str
    # How many operands it reads.
    operands: int
    # The layout of the instruction (see `_X86Loop.operate`): "plain" is `op t, a, b` or `op t, a`; "masked"
    # reads a constant mask as its last operand; "extreme" is `np.maximum` or `np.minimum`; "compare" takes
    # a predicate; "select" is `np.where`.
    form: str = "plain"
    # For "masked", which constant: a key of X86_CONSTANTS.
    constant: str = ""


# The operations an x86-64 kernel can be written for. Each is the AVX instruction NumPy's own loop executes for
# it on this architecture, and rounds once. A negation and an absolute value change the sign bit alone, which is
# what NumPy does, a NaN included. `vsqrtsd` reads two sources and takes the upper lane from the first.
X86_OPERATIONS = {
    "add": X86Operation("vaddpd", "vaddsd", 2),
    "sub": X86Operation("vsubpd", "vsubsd", 2),
    "mul": X86Operation("vmulpd", "vmulsd", 2),
    "div": X86Operation("vdivpd", "vdivsd", 2),
    "sq": X86Operation("vmulpd", "vmulsd", 1),
    "sqrt": X86Operation("vsqrtpd", "vsqrtsd", 1),
    "neg": X86Operation("vxorpd", "vxorpd", 1, "masked", "sign"),
    "abs": X86Operation("vandpd", "vandpd", 1, "masked", "magnitude"),
    # `np.maximum` on x86-64 is `maxpd` with a NaN of the first operand carried through: a NaN if either operand
    # is one, and of two zeros the second operand, whichever its sign. `maxpd` alone gives the second operand
    # whenever either is a NaN, so the first is put in the second's place where it is a NaN (see `operate`).
    "max": X86Operation("vmaxpd", "vmaxsd", 2, "extreme"),
    "min": X86Operation("vminpd", "vminsd", 2, "extreme"),
    # A comparison is ordered and quiet, as NumPy's `>` is: false where either operand is a NaN, and no flag.
    "gt": X86Operation("vcmppd", "vcmpsd", 2, "compare"),
    "le": X86Operation("vcmppd", "vcmpsd", 2, "compare"),
    "where": X86Operation("vblendvpd", "vblendvpd", 3, "select"),
}
# The operations only the x86-64 loops have a rule for, as they are written in a kernel's header comment.
X86_ONLY_FORMS = {"gt": "({0} > {1})", "le": "({0} <= {1})", "where": "({0} ? {1} : {2})"}
# The predicates of the comparisons, as `vcmppd` takes them, each ordered and quiet: greater than (_CMP_GT_OQ), and
# less than or equal (_CMP_LE_OQ).
X86_PREDICATES = {"gt": "0x1e", "le": "0x12"}
# The constants an operation reads, by name: the label each has in the kernel's read-only data, and its bits.
X86_CONSTANTS = {"sign": (".Lsign", "0x8000000000000000"), "magnitude": (".Lmagnitude", "0x7fffffffffffffff")}
# The registers the integer and pointer arguments arrive in, in order; the ones after them are on the stack, the
# seventh at [rsp + 8] and each next 8 bytes above (the System V calling convention).
X86_ARGUMENTS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
X86_STACK_FIRST, X86_STACK_STEP = 8, 8
# Where an argument that came on the stack is held: the result's address and the length, the only arguments
# after the sixth a kernel has (six arrays at most).
X86_SPARE = ("r10", "r11")
# The vector registers: ymm0 to ymm15, all free to use. A number is held in every lane of one, from the top down,
# and the constants and the scratch register of `np.maximum` below the numbers.
X86_REGISTERS = 16
# Elements a step computes on x86-64, and ymm registers of four that is: a block is eight elements on both
# architectures, which `_fused.c` relies on (see the README: every load of a block comes before its store).
X86_LANES = 2


@dataclass
class _X86Loop:
    """The instructions of one x86-64 loop's body, as they are written: `(mnemonic, operands)`.

    Every address is the array's base register plus the element index in `rax`, scaled by 8: the loop moves one
    index, not a pointer per array.
    """

    # True for the loop of eight elements (two ymm registers a value), False for the loop of one (one xmm).
    wide: bool
    # The base register of each array, by the array's number in the shape, and of the result.
    arrays: list[str]
    out: str
    # The xmm register each number arrived in, by its number in the shape; in the wide loop it is held, in every
    # lane, in the ymm register X86_REGISTERS - 1 - its number instead.
    numbers: list[int]
    # The register of each constant the shape reads, by name, and the scratch register of `np.maximum`, if any.
    constants: dict[str, int]
    scratch: int | None
    registers: _Registers
    lines: list[tuple[str, str]] = field(default_factory=list)

    def name(self, register: int) -> str:
        """Write a register as an operand of an arithmetic instruction."""
        return f"ymm{register}" if self.wide else f"xmm{register}"

    def _address(self, base: str, block: int) -> str:
        """Write the address of the elements one register of a value holds.

        Args:
            base: The array's base register.
            block: Which register of the value: the elements 4 * block on from the index.

        Returns:
            The memory operand.
        """
        if self.wide:
            size, offset = "ymmword", f" + {block * 32}" if block else ""
        else:
            size, offset = "qword", ""
        return f"{size} ptr [{base} + rax*8{offset}]"

    def load(self, array: int) -> tuple[int, ...]:
        """Load the elements of an array this step computes into new registers.

        Args:
            array: The array's number in the shape.

        Returns:
            The registers that hold them.
        """
        held = self.registers.take()
        mnemonic = "vmovupd" if self.wide else "vmovsd"
        for block, register in enumerate(held):
            self.lines.append((mnemonic, f"{self.name(register)}, {self._address(self.arrays[array], block)}"))
        return held

    def store(self, held: tuple[int, ...]) -> None:
        """Store a value as the result."""
        mnemonic = "vmovupd" if self.wide else "vmovsd"
        for block, register in enumerate(held):
            self.lines.append((mnemonic, f"{self._address(self.out, block)}, {self.name(register)}"))

    def number(self, place: int) -> tuple[int, ...]:
        """Return the registers a number is read from: one register in every position."""
        register = X86_REGISTERS - 1 - place if self.wide else self.numbers[place]
        return (register,) * self.registers.width

    def arity(self, op: str) -> int | None:
        """Return how many operands an operation reads, or None for one this loop has no rule for."""
        operation = X86_OPERATIONS.get(op)
        return None if operation is None else operation.operands

    def operate(self, op: str, target: tuple[int, ...], operands: list[tuple[int, ...]]) -> None:
        """Write one operation: its instructions for each register of the value.

        Args:
            op: The operation's name in the shape, a key of X86_OPERATIONS.
            target: The registers the result goes to. It may be one of an operand's own: every form reads all
                its operands before it writes the target.
            operands: The registers of each operand, in the order the operation reads them.
        """
        operation = X86_OPERATIONS[op]
        mnemonic = operation.wide if self.wide else operation.narrow
        for lane, register in enumerate(target):
            t = self.name(register)
            read = [self.name(operand[lane]) for operand in operands]
            if operation.form == "masked":
                self.lines.append((mnemonic, f"{t}, {read[0]}, {self.name(self.constants[operation.constant])}"))
            elif operation.form == "extreme":
                # x = a where a is a NaN, else b; then maxpd(a, x). Where a is a NaN that is maxpd(a, a), a NaN;
                # elsewhere it is maxpd(a, b), which gives b where b is a NaN.
                x = self.name(self.scratch)
                compare = "vcmpunordpd" if self.wide else "vcmpunordsd"
                self.lines.append((compare, f"{x}, {read[0]}, {read[0]}"))
                self.lines.append(("vblendvpd", f"{x}, {read[1]}, {read[0]}, {x}"))
                self.lines.append((mnemonic, f"{t}, {read[0]}, {x}"))
            elif operation.form == "compare":
                self.lines.append((mnemonic, f"{t}, {read[0]}, {read[1]}, {X86_PREDICATES[op]}"))
            elif operation.form == "select":
                # `vblendvpd t, b, a, mask` takes a where the mask's sign bit is set: np.where(mask, a, b).
                self.lines.append((mnemonic, f"{t}, {read[2]}, {read[1]}, {read[0]}"))
            elif operation.operands == 1 and not self.wide:
                self.lines.append((mnemonic, f"{t}, {read[0]}, {read[0]}"))
            else:
                self.lines.append((mnemonic, f"{t}, {', '.join(read)}"))


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
    if loop.arity(pattern.op) != len(pattern.parts):
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
    loop.operate(pattern.op, target, read * 2 if pattern.op == fuse.SQUARE else read)
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
    parts = (_written(part, form, names, numbers) for part in pattern.parts)
    if pattern.op not in OPERATIONS:
        return X86_ONLY_FORMS[pattern.op].format(*parts)
    return getattr(OPERATIONS[pattern.op], form).format(*parts)


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
        "defaults": NUMBER_DEFAULTS[:numbers], "trials": NUMBER_TRIALS,
        "out": out, "length": length, "block": BLOCK, "top": TOP_REGISTER, "wide": loops[True], "narrow": loops[False],
        "words": sorted({OPERATIONS[word].word for word in OPERATIONS if f"{word}(" in shape}),
        "larger": "max(" in shape, "smaller": "min(" in shape,
    }
    environment = _environment()
    (directory / ARCH).mkdir(parents=True, exist_ok=True)
    for template, path in ((SPEC_TEMPLATE, directory / "spec.py"), (NAIVE_TEMPLATE, directory / "naive.c"),
                           (ASSEMBLY_TEMPLATE, directory / ARCH / "kernel.s"),
                           (LOG_TEMPLATE, directory / ARCH / "attempts.md")):
        path.write_text(environment.get_template(template).render(**context))
    identity = tuple(range(arrays)), tuple(range(numbers))
    _write_x86(directory, name, shape, pattern, array_names, number_names, *identity, environment)
    return directory


def _environment() -> Environment:
    """Return the Jinja environment the kernels' files are written with."""
    return Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True,
                       lstrip_blocks=True, keep_trailing_newline=True)


def _x86_argument(position: int) -> tuple[str, tuple[str, int] | None]:
    """Find where an integer or pointer argument of an x86-64 kernel is, and where it is held.

    Args:
        position: The argument's place among the integer and pointer arguments, from 0.

    Returns:
        (the register it is read from, and for one that came on the stack (the register, its offset from rsp),
        which the kernel loads first; else None).
    """
    if position < len(X86_ARGUMENTS):
        return X86_ARGUMENTS[position], None
    spare = X86_SPARE[position - len(X86_ARGUMENTS)]
    return spare, (spare, X86_STACK_FIRST + X86_STACK_STEP * (position - len(X86_ARGUMENTS)))


def _write_x86(directory: Path, name: str, shape: str, pattern: fuse.Pattern, array_names: list[str],
               number_names: list[str], vector_order: tuple[int, ...], scalar_order: tuple[int, ...],
               environment: Environment) -> None:
    """Write the x86-64 assembly of one kernel, and the first entry of its log.

    Args:
        directory: The kernel's directory.
        name: The kernel's name: its C function's.
        shape: The shape's name.
        pattern: The same as a tree.
        array_names: The array arguments' names, in the order the kernel takes them.
        number_names: The same for the number arguments, which come after the result and the length.
        vector_order: For each array argument, the number of the shape's vector leaf it is.
        scalar_order: For each number argument, the number of the scalar leaf it is.
        environment: The templates.

    Raises:
        CannotWrite: For a shape the x86-64 loops have no rule for, or that needs more registers than there are.
    """
    arrays, numbers = len(array_names), len(number_names)
    places = [_x86_argument(position) for position in range(arrays + 2)]
    registers = [register for register, _ in places]
    by_leaf = [registers[vector_order.index(leaf)] for leaf in range(arrays)]
    number_registers = [scalar_order.index(leaf) for leaf in range(numbers)]
    # The constants and the scratch register take the ymm registers below the numbers'.
    below = X86_REGISTERS - 1 - numbers
    wanted = [key for key, operation in ((each, X86_OPERATIONS[each]) for each in X86_OPERATIONS)
              if operation.form == "masked" and f"{key}(" in shape]
    constants = {X86_OPERATIONS[key].constant: below - place for place, key in enumerate(wanted)}
    scratch = below - len(constants) if "max(" in shape or "min(" in shape else None
    reserved = set(constants.values()) | ({scratch} if scratch is not None else set())
    if reserved and min(reserved) < numbers:
        raise CannotWrite("the shape has too many numbers for its constants to have registers of their own")
    loops = {}
    for wide in (True, False):
        held = {X86_REGISTERS - 1 - place for place in range(numbers)} if wide else set(range(numbers))
        free = [register for register in range(X86_REGISTERS) if register not in held | reserved]
        loop = _X86Loop(wide, by_leaf, registers[arrays], number_registers, constants, scratch,
                        _Registers(free, X86_LANES if wide else 1))
        value, _ = _compute(pattern, loop)
        loop.store(value)
        loops[wide] = loop.lines
    leaf_names = [array_names[vector_order.index(leaf)] for leaf in range(arrays)]
    leaf_numbers = [number_names[scalar_order.index(leaf)] for leaf in range(numbers)]
    labels = {key: X86_CONSTANTS[key] for key in constants}
    context = {
        "name": name, "shape": shape, "operations": _count(pattern, ""), "arguments": array_names,
        "numbers": number_names, "registers": registers, "out_argument": registers[arrays],
        "length_argument": registers[arrays + 1], "length": registers[arrays + 1],
        "c": _bare(_written(pattern, "c", [f"{each}[i]" for each in leaf_names], leaf_numbers)),
        "stack": [place for _, place in places if place is not None], "block": BLOCK, "top": X86_REGISTERS - 1,
        "constants": [(labels[key][0], register) for key, register in constants.items()],
        "data": list(labels.values()), "extremes": scratch is not None,
        "wide": loops[True], "narrow": loops[False],
    }
    (directory / X86_ARCH).mkdir(parents=True, exist_ok=True)
    for template, path in ((X86_ASSEMBLY_TEMPLATE, directory / X86_ARCH / "kernel.s"),
                           (X86_LOG_TEMPLATE, directory / X86_ARCH / "attempts.md")):
        path.write_text(environment.get_template(template).render(**context))


def write_x86(directory: Path, force: bool = False) -> bool:
    """Write the x86-64 assembly of a kernel that has a directory, from the pattern its oracle computes.

    This is how a kernel written by hand for ARM64, or one written by this tool before it wrote x86-64, gets its
    x86-64 loop: the shape is read off the spec's oracle, as `fuse.py` reads it, and the arguments are taken in
    the spec's order, whichever leaf of the shape each one is.

    Args:
        directory: The kernel's directory, with its `spec.py`.
        force: Whether to write over an x86-64 kernel that is there.

    Returns:
        True when it was written; False when the kernel has an x86-64 log and `force` is not set: its loop is
        there, or was timed against the compiled C and not kept, which the log says.

    Raises:
        CannotWrite: For a kernel that sums, or a shape the x86-64 loops have no rule for.
    """
    from . import build_fused
    if (directory / X86_ARCH / LOG_NAME).exists() and not force:
        return False  # Written already: kept, or timed and left to the compiled C, as its log says.
    fused = build_fused.read(directory)
    if fused.reduces:
        raise CannotWrite("a kernel that sums is written by hand")
    shape, vector_order, scalar_order = fuse.pattern_of(fused.spec.oracle, fused.vectors, fused.scalars)
    names = [arg.name for arg in fused.spec.args]
    array_names, number_names = names[:fused.vectors], names[fused.vectors + 2:]
    _write_x86(directory, fused.name, shape, fuse.parse(shape), array_names, number_names, vector_order,
               scalar_order, _environment())
    return True


@app.command()
def main(shape: list[str] = typer.Argument(None, help="A shape to write the kernel of, such as 'sub(V0,mul(S0,V1))'; "
                                                       "with --arch x86_64, a kernel's name."),
         every: bool = typer.Option(False, "--all", help="Write every shape of this tool's list that has no kernel; "
                                                         "with --arch x86_64, every kernel that has no x86-64 loop."),
         arch: str = typer.Option("", help="x86_64: write only the x86-64 assembly of kernels that are there."),
         force: bool = typer.Option(False, help="Write over a kernel's directory that is there.")) -> None:
    """Write the kernels of some shapes, each into its directory under `kernels/`.

    Args:
        shape: The shapes named on the command line, or with `arch` the kernels.
        every: Whether to write the tool's own list as well, or with `arch` every kernel.
        arch: Empty to write whole kernels, or X86_ARCH to write the x86-64 assembly of kernels that are there.
        force: Whether to write over directories that are there.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if arch:
        _main_x86(list(shape or ()), every, arch, force)
        return
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


def _main_x86(names: list[str], every: bool, arch: str, force: bool) -> None:
    """Write the x86-64 assembly of the kernels named, or of every kernel.

    Args:
        names: Kernels' names.
        every: Whether to take every kernel's directory; one this tool cannot write is then passed over.
        arch: The architecture asked for, which must be X86_ARCH.
        force: Whether to write over an x86-64 kernel that is there.
    """
    if arch != X86_ARCH:
        log.error("--arch takes %s: the ARM64 assembly is written with the whole kernel", X86_ARCH)
        raise typer.Exit(1)
    directories = [KERNELS / name for name in names] + (sorted(path.parent for path in KERNELS.glob("*/spec.py")) if every else [])
    if not directories:
        log.error("name a kernel, or give --all")
        raise typer.Exit(1)
    for directory in directories:
        try:
            written = write_x86(directory, force)
        except CannotWrite as error:
            log.log(logging.INFO if every else logging.ERROR, "%s: %s", directory.name, error)
            if not every:
                raise typer.Exit(1) from error
            continue
        log.info("%s %s/kernel.s", "wrote" if written else "there is", f"{directory.name}/{X86_ARCH}")


if __name__ == "__main__":
    app()
