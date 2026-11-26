"""Fusion by pattern: finding expressions in the pending graph, and the kernels that compute one in a single pass.

An expression is a tree of element-by-element operations on float64 arrays
whose intermediate results nobody else reads: `a * b + c` is two
operations and one intermediate. NumPy computes it in two passes over
memory and allocates the intermediate; a kernel written for the shape of
that tree computes it in one pass and allocates nothing but the result.

This module has the three parts of that, each usable alone:

- `find` reads the expressions off the nodes one computation is about to
  take. It is also what `programs/mine.py` counts patterns with.
- `shapes_at` lists the shapes of tree a kernel could be written for, cut
  out of an expression, and `canonical` names one so that two trees of one
  shape have one name.
- `tile` covers an expression with the shapes of a library, each tile one
  kernel call, by the fewest calls, and `plan` does that for every
  expression of a computation: it is what `enable` gives `evaluate` as its
  planner.

What a fused kernel may contain is decided by one rule: every operation in
it is a single IEEE operation that NumPy makes with the same instruction,
in the order the program wrote them. Then each element goes through the
same roundings as in NumPy's separate passes, and the bits are NumPy's. The
operations of `ARITHMETIC`, `UNARY`, `COMPARISONS` and `np.where` are such
operations; `np.exp`, `np.tanh` and a power other than 2 are calls into a
mathematics library, which a kernel cannot make, and end an expression.
"""
from __future__ import annotations

import operator
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from . import array, evaluate, graph, rules
from .graph import Node

# The element type every operand and result of a fused expression has.
FLOAT64 = np.dtype(np.float64)
# The element type a comparison inside an expression has.
BOOLEAN = np.dtype(np.bool_)

# Element-by-element operations of two operands that are one IEEE operation each, by the name a pattern gives them.
ARITHMETIC = {np.add: "add", np.subtract: "sub", np.multiply: "mul", np.true_divide: "div",
              np.maximum: "max", np.minimum: "min"}
# The same of one operand. `np.square` is one multiplication.
UNARY = {np.negative: "neg", np.absolute: "abs", np.sqrt: "sqrt", np.square: "sq"}
# Comparisons. Their result is a mask, which a pattern reads only as the condition of a `where`.
COMPARISONS = {np.greater: "gt", np.greater_equal: "ge", np.less: "lt", np.less_equal: "le",
               np.equal: "eq", np.not_equal: "ne"}
# The names of the comparisons.
MASKS = frozenset(COMPARISONS.values())
# The operations whose two operands may be written in either order.
COMMUTATIVE = frozenset({"add", "mul", "max", "min"})
# Every such operation by the callable a node holds for it.
OPERATIONS = {**ARITHMETIC, **UNARY, **COMPARISONS, np.where: "where"}
# The ufunc an in-place operator applies, by the name `array.py` gives its write.
IN_PLACE = {"x += y": np.add, "x -= y": np.subtract, "x *= y": np.multiply, "x /= y": np.true_divide}
# The largest whole number a float64 holds exactly: a Python integer up to it is the same number as a float.
EXACT_INTEGER = 2 ** 53
# The exponents of `np.power` that NumPy computes as one multiplication, `x * x`.
SQUARE_EXPONENTS = (2, 2.0)
# The name of the operation that is one multiplication of a value by itself.
SQUARE = "sq"
# The name of `np.where` inside a pattern.
WHERE = "where"
# The reductions over every element that a pattern may end in, by the node's name.
REDUCTIONS = frozenset({"sum", "mean"})

# Leaves of a pattern: an array read along the innermost axis, and a number that is the same along it.
VECTOR, SCALAR = "V", "S"
# What an expression gives: an array of its own, the sum or the mean of one, or an array stored into part of another.
ARRAY, STORE = "array", "store"

# The most operations a shape cut out of an expression may have. A kernel of more would need more registers than
# hold its operands and intermediates, and no shape of more was common in the programs mined.
MOST_OPERATIONS = 6


@dataclass(frozen=True, slots=True)
class Leaf:
    """One operand of an expression that is not computed inside it."""

    # VECTOR or SCALAR.
    kind: str
    # Its place among the expression's operands.
    place: int


@dataclass(frozen=True, slots=True)
class Term:
    """One operation of an expression."""

    # The operation's name in a pattern: a value of ARITHMETIC, UNARY or COMPARISONS, WHERE, or a name of REDUCTIONS.
    op: str
    # Its operands: terms and leaves.
    operands: tuple
    # The node that wrote it.
    node: Node = field(compare=False, hash=False)


@dataclass(slots=True)
class ViewOf:
    """An operand that is part of an array a write is made into: what the array held there before the write."""

    # The node of the array's contents before the write. None in a record kept only for its steps.
    old: Node | None
    # The view operations that lead from the array to the part.
    steps: tuple

    def of(self, value: np.ndarray) -> np.ndarray:
        """Take the part of an array.

        Args:
            value: The array's contents, or a copy of them.

        Returns:
            NumPy's view of the part.
        """
        for step in self.steps:
            value = step.of(value)
        return value


@dataclass(slots=True)
class Expression:
    """A tree of operations one computation is about to take, with what it reads."""

    # The node whose value the expression gives.
    root: Node
    # The tree. Its top is the root's operation.
    term: Term
    # What it reads, in the order the leaves count: nodes, `ndarray`s, numbers, and `ViewOf` for a part written into.
    operands: list
    # ARRAY, STORE, or a name of REDUCTIONS.
    result: str
    # The shape the element-by-element operations run over.
    shape: tuple[int, ...]
    # For STORE: where the result goes, as the part of a copy of an array. None otherwise.
    target: ViewOf | None = None

    @property
    def elements(self) -> int:
        """How many elements each operation of the expression computes."""
        count = 1
        for length in self.shape:
            count *= length
        return count


def count_uses(todo: list[Node], unread: dict[int, int] | None = None) -> dict[int, int]:
    """Count how often each node is read by the nodes of one computation.

    Args:
        todo: The pending nodes about to be computed.
        unread: When given, filled with node number -> how many of those nodes keep it as the old contents of
            an array they write the whole of without reading it: `np.add(a, b, out=x)` once `rewrite_write`
            has made it one call. Such a node is referred to, which `held_among` has to know, and is not read.

    Returns:
        Node number -> how many operands of those nodes it is. An operand written twice counts twice.
    """
    uses: dict[int, int] = {}
    for node in todo:
        args = node.args
        if type(node.operation) is _WholeCall and not node.operation.reads_old:
            old, args = args[0], args[1:]
            if unread is not None and type(old) is Node:
                unread[old.number] = unread.get(old.number, 0) + 1
        for arg in args:
            if type(arg) is Node:
                uses[arg.number] = uses.get(arg.number, 0) + 1
    return uses


def _spare_references(nodes: list) -> list[int]:
    """Count the references to each object of a list, this function's own among them.

    Args:
        nodes: Any objects.

    Returns:
        `sys.getrefcount` of each, in order.
    """
    return [sys.getrefcount(node) for node in nodes]


# What `_spare_references` reports for an object that its list alone refers to: the list's reference and the ones
# the count itself makes. Measured, not assumed, since an interpreter may count its own references differently.
_ALONE = _spare_references([object()])[0]


def held_among(todo: list[Node], uses: dict[int, int], unread: dict[int, int] | None = None) -> set[int]:
    """Find the nodes of a computation whose value something outside it will want.

    Such a node is one the program holds a lazy array of, or one that a
    pending node outside this computation reads. Either way its value may
    be asked for later, so computing past it would compute it twice. Both
    show as a reference: CPython counts references exactly, and a node of
    this computation is referred to by the nodes of it that read it, by the
    list it is in here, and by nothing else unless one of the two holds.

    The count decides only what is worth fusing. A node that a fused
    expression skips stays pending and is computed when it is asked for,
    whatever this says.

    Args:
        todo: The pending nodes about to be computed. The caller's one list of them: a second list would count.
        uses: How often each node is read by the nodes of the computation, from `count_uses`.
        unread: How often each is referred to by them without being read, from `count_uses`; None for never.

    Returns:
        The numbers of the nodes with a reference the computation and the list do not account for.
    """
    spare = _spare_references(todo)
    if not unread:
        return {node.number for node, count in zip(todo, spare) if count - _ALONE > uses.get(node.number, 0)}
    return {node.number for node, count in zip(todo, spare)
            if count - _ALONE > uses.get(node.number, 0) + unread.get(node.number, 0)}


def _is_number(arg: Any) -> bool:
    """Say whether an operand is a number a kernel can take in a register: a Python float or integer, or a float64."""
    kind = type(arg)
    return kind is float or kind is np.float64 or (kind is int and -EXACT_INTEGER <= arg <= EXACT_INTEGER)


def _leaf_kind(arg: Any, shape: tuple[int, ...]) -> str | None:
    """Classify an operand that is read, not computed, by an expression.

    Args:
        arg: A node, an `ndarray` or a number.
        shape: The shape the expression's operations run over; at least one dimension.

    Returns:
        VECTOR for a float64 array as long as the shape along the innermost
        axis, SCALAR for a number or a float64 array of length one along it,
        and None for anything a fused kernel does not read: another element
        type, a length that is not known.
    """
    if _is_number(arg):
        return SCALAR
    if type(arg) is not Node and type(arg) is not np.ndarray:
        return None
    if arg.dtype != FLOAT64 or not arg.dtype.isnative:
        return None
    if not arg.shape or arg.shape[-1] == 1 != shape[-1]:
        return SCALAR
    return VECTOR if arg.shape[-1] == shape[-1] and graph.is_known(arg.shape) else None


def call_of(node: Node) -> tuple[Any, tuple] | None:
    """Read a node as one call of an element-by-element function, if it is one.

    Args:
        node: A pending node.

    Returns:
        (the ufunc or `np.where`, its operands as written), or None. A write
        into the whole of an array by an in-place operator is the operator's
        ufunc on what the array held and the operand, and one by a ufunc with
        `out=` is the ufunc on its own operands; `rewrite_write` says why
        those have the bits of the write.
    """
    operation, args = node.operation, node.args
    if node.kind == rules.ELEMENTWISE:
        return (operation, args) if not node.kwargs else None
    if node.kind != rules.WRITE:
        return None
    if type(operation) is array._Write:
        if operation.steps:
            return None
        if type(operation.perform) is array._Into:
            # `np.add(a, b, out=x)` leaves nothing of what `x` held: the operands after it are the call's.
            return (operation.perform.function, args[1:]) if isinstance(operation.perform.function, np.ufunc) else None
        ufunc = IN_PLACE.get(node.name)
        return (ufunc, args) if ufunc is not None else None
    if type(operation) is _WholeCall:
        # A write `rewrite_write` has already turned into its ufunc.
        return operation.ufunc, (args if operation.reads_old else args[1:])
    return None


def operation_of(node: Node) -> tuple[str, tuple] | None:
    """Read a node as one operation of an expression, if it is one a kernel can make.

    Args:
        node: A pending node.

    Returns:
        (the operation's name in a pattern, its operands as written), or None.
    """
    call = call_of(node)
    if call is None:
        return None
    operation, args = call
    name = OPERATIONS.get(operation)
    if name is not None:
        return name, args
    if (operation is np.power or operation is operator.pow) and type(args[1]) in (int, float) \
            and args[1] in SQUARE_EXPONENTS:
        return SQUARE, args[:1]
    return None


class _WholeCall:
    """What a write into the whole of an array calls once it is its one ufunc call.

    `x += y` is, as written, a copy of `x` and the sum made into the copy.
    This makes the sum of what `x` held and `y` into new memory laid out as
    the copy would have been, or, when nothing reads the old contents again,
    into the memory `x` has: one pass either way, and the second is what
    NumPy does.
    """

    __slots__ = ("ufunc", "reads_old")

    def __init__(self, ufunc: np.ufunc, reads_old: bool) -> None:
        """Remember the call.

        Args:
            ufunc: The operation.
            reads_old: True for an in-place operator, whose first operand is what the array held. False for a
                ufunc with `out=`, which reads only its own operands; the node keeps the old contents as its
                first operand all the same, since their memory is where the result may go.
        """
        self.ufunc = ufunc
        self.reads_old = reads_old

    def __call__(self, old: np.ndarray, *values: Any) -> np.ndarray:
        """Make the call into new memory.

        Args:
            old: The array's contents before the write.
            *values: The operands after it.

        Returns:
            The array's contents after the write, laid out as a copy of `old` would be.
        """
        new = np.empty_like(old)
        return self.ufunc(old, *values, out=new) if self.reads_old else self.ufunc(*values, out=new)

    def in_place(self, old: np.ndarray, *values: Any) -> np.ndarray:
        """Make the call into the memory the array has.

        Args:
            old: The array's contents before the write, in an array nothing else will read.
            *values: The operands after it.

        Returns:
            The same array, holding the result.
        """
        return self.ufunc(old, *values, out=old) if self.reads_old else self.ufunc(*values, out=old)


def rewrite_write(node: Node) -> bool:
    """Turn a write into the whole of a float64 array into the one call that gives the same array.

    As written, the node copies what the array held and applies the
    operator to the copy in place: two passes over memory. `old + y` is one
    pass and has the same bits when everything is float64: the same loop
    adds the same operands, and nothing is cast on the way into the array.
    The result is laid out as the copy would have been. The node keeps its
    kind, its name, its line and the old contents as its first operand.

    Args:
        node: A pending node.

    Returns:
        True when the node was rewritten.
    """
    if node.kind != rules.WRITE or type(node.operation) is not array._Write or node.dtype != FLOAT64 or not node.shape:
        return False
    call = call_of(node)
    if call is None or not isinstance(call[0], np.ufunc) or not graph.is_known(node.shape):
        return False
    if any(_leaf_kind(arg, node.shape) is None for arg in call[1]):
        return False
    reads_old = type(node.operation.perform) is not array._Into
    node.operation = _WholeCall(call[0], reads_old)
    return True


class _Finder:
    """Reads the expressions off the nodes of one computation."""

    def __init__(self, todo: list[Node], fewest: int) -> None:
        """Start on one computation.

        Args:
            todo: The pending nodes about to be computed.
            fewest: The fewest elements an expression's operations run over for it to be read.
        """
        # The numbers of the nodes about to be computed.
        self.taking = {node.number for node in todo}
        # How often each node is read by the nodes of the computation.
        # How often each is kept, unread, as the old contents of an array a rewritten write gives the whole of.
        self.unread: dict[int, int] = {}
        self.uses = count_uses(todo, self.unread)
        # The numbers of the nodes whose value something outside the computation will want.
        self.held = held_among(todo, self.uses, self.unread)
        self.fewest = fewest
        # The numbers of the nodes that are inside an expression found so far, and not its root.
        self.inside: set[int] = set()
        # The same in the order they were put there, so that a tree given up can take its nodes out again.
        self.journal: list[int] = []

    def give_up(self, mark: int) -> None:
        """Take out the nodes put inside since a point: the tree they were read for is not an expression.

        Args:
            mark: The length the journal had at that point.
        """
        for number in self.journal[mark:]:
            self.inside.discard(number)
        del self.journal[mark:]

    def _put_inside(self, node: Node) -> None:
        """Note that a node is computed inside an expression.

        Args:
            node: The node.
        """
        self.inside.add(node.number)
        self.journal.append(node.number)

    def expression(self, root: Node) -> Expression | None:
        """Read the largest expression that ends in a node.

        Args:
            root: A node about to be computed that is inside no expression.

        Returns:
            The expression, or None when the node is not the top of one.
        """
        if not graph.is_known(root.shape):
            return None
        operands: list = []
        if root.kind == rules.REDUCTION:
            return self._reduction(root, operands)
        if root.kind == rules.WRITE and type(root.operation) is array._Write and root.operation.steps:
            return self._store(root, operands)
        if root.kind == rules.WRITE and type(root.operation) is array._Write and type(root.operation.perform) is array._Assign:
            return self._store(root, operands)
        if not root.shape or root.dtype != FLOAT64 or _elements(root.shape) < self.fewest:
            return None
        term = self._term(root, root.shape, operands, inner=False)
        return Expression(root, term, operands, ARRAY, root.shape) if term is not None else None

    def _reduction(self, root: Node, operands: list) -> Expression | None:
        """Read a sum or a mean over every element of an expression.

        Args:
            root: A reduction about to be computed.
            operands: The list the expression's operands are added to.

        Returns:
            The expression, or None unless the reduction is of every element
            of an array that an expression gives and nothing else reads.
        """
        if root.name not in REDUCTIONS or len(root.args) != 1 or root.kwargs or root.dtype != FLOAT64:
            return None
        reduced = root.args[0]
        if type(reduced) is not Node or not reduced.shape or not self._is_inner(reduced, reduced.shape):
            return None
        if _elements(reduced.shape) < self.fewest:
            return None
        inner = self._term(reduced, reduced.shape, operands, inner=True)
        if inner is None:
            return None
        return Expression(root, Term(root.name, (inner,), root), operands, root.name, reduced.shape)

    def _store(self, root: Node, operands: list) -> Expression | None:
        """Read a write into part of an array as an expression stored there.

        `x[index] = e` stores `e`; `x[index] += e` and a ufunc with
        `out=x[index]` store the operation, which reads what the part held.

        Args:
            root: A deferred write about to be computed.
            operands: The list the expression's operands are added to.

        Returns:
            The expression, or None when what is stored is not one.
        """
        write = root.operation
        old = root.args[0]
        if type(old) is not Node or old.dtype != FLOAT64 or not graph.is_known(old.shape):
            return None
        steps, perform = write.steps, write.perform
        if _elements(old.shape) < self.fewest:
            return None  # The part is no larger than the array.
        if type(perform) is array._Assign:
            if not rules.is_basic_index(perform.index):
                return None
            part = rules.basic_shape(steps[-1].shape if steps else old.shape, perform.index)
            stored = root.args[1]
            if part is None or not part or type(stored) is not Node or stored.shape != part or not self._is_inner(stored, part):
                return None
            term = self._term(stored, part, operands, inner=True)
            target = ViewOf(old, steps + (array._Step(operator.getitem, "getitem", (perform.index,), {}, part, FLOAT64,
                                                      root.origin),))
            return Expression(root, term, operands, STORE, part, target) if term is not None else None
        part = steps[-1].shape
        if not part or not graph.is_known(part) or steps[-1].dtype != FLOAT64:
            return None
        target = ViewOf(old, steps)
        if type(perform) is array._Into:
            name, written = ARITHMETIC.get(perform.function) or UNARY.get(perform.function), root.args[1:]
        else:
            name, written = ARITHMETIC.get(IN_PLACE.get(root.name)), (target,) + root.args[1:]
        if name is None:
            return None
        term = self._operation(root, name, written, part, operands)
        return Expression(root, term, operands, STORE, part, target) if term is not None else None

    def _is_inner(self, node: Node, shape: tuple[int, ...]) -> bool:
        """Say whether a node may be computed inside an expression instead of by itself.

        Args:
            node: An operand of a node of the expression.
            shape: The shape the expression's operations run over.

        Returns:
            True for a pending node of that shape that this computation
            takes, that one pending node reads, once, and that the program
            holds no lazy array of.
        """
        number = node.number
        return (node.operation is not None and number in self.taking and self.uses.get(number) == 1
                and node.shape == shape and number not in self.held)

    def _term(self, node: Node, shape: tuple[int, ...], operands: list, inner: bool) -> Term | None:
        """Read a node and what it is computed from as a tree.

        Args:
            node: A pending node of the expression's shape.
            shape: The shape the expression's operations run over.
            operands: The list the expression's operands are added to.
            inner: Whether the node is inside the expression, and not its root.

        Returns:
            The tree, or None when the node is not an operation a kernel can make on these operands.
        """
        read = operation_of(node)
        if read is None:
            return None
        term = self._operation(node, read[0], read[1], shape, operands)
        if term is not None and inner:
            self._put_inside(node)
        return term

    def _operation(self, node: Node, name: str, written: tuple, shape: tuple[int, ...], operands: list) -> Term | None:
        """Read one operation and its operands as a tree.

        Args:
            node: The node that wrote the operation.
            name: The operation's name in a pattern.
            written: Its operands as written.
            shape: The shape the expression's operations run over.
            operands: The list the expression's operands are added to.

        Returns:
            The tree, or None when an operand is one a kernel does not read.
            Nothing is added to `operands` then.
        """
        mark, inside = len(operands), len(self.journal)
        parts = []
        for place, arg in enumerate(written):
            part = self._operand(arg, shape, operands, mask=name == WHERE and place == 0)
            if part is None:
                del operands[mark:]
                self.give_up(inside)
                return None
            parts.append(part)
        if name == "mul" and parts[0] == parts[1] and type(parts[0]) is Leaf:
            return Term(SQUARE, (parts[0],), node)
        return Term(name, tuple(parts), node)

    def _operand(self, arg: Any, shape: tuple[int, ...], operands: list, mask: bool) -> Term | Leaf | None:
        """Read one operand of an operation: a tree if it is computed inside the expression, a leaf if it is read.

        Args:
            arg: The operand as written.
            shape: The shape the expression's operations run over.
            operands: The list the expression's operands are added to.
            mask: Whether the operand is the condition of a `where`, which has to be a comparison made inside.

        Returns:
            The tree or the leaf, or None for an operand a kernel does not read.
        """
        if type(arg) is Node and self._is_inner(arg, shape):
            read = operation_of(arg)
            if read is not None and (read[0] in MASKS) == mask and arg.dtype == (BOOLEAN if mask else FLOAT64):
                term = self._operation(arg, read[0], read[1], shape, operands)
                if term is not None:
                    self._put_inside(arg)
                    return term
        if mask:
            return None
        if type(arg) is ViewOf:
            kind = VECTOR
        else:
            kind = _leaf_kind(arg, shape)
            if kind is None:
                return None
        for place, known in enumerate(operands):
            if known is arg and kind == VECTOR:
                return Leaf(kind, place)
        operands.append(arg)
        return Leaf(kind, len(operands) - 1)


def find(todo: list[Node], fewest: int = 1) -> tuple[list[Expression], set[int]]:
    """Read the expressions off the nodes one computation is about to take.

    Args:
        todo: The pending nodes about to be computed, newest first, as `evaluate` lists them: a node is then met
            before the nodes it reads, so the largest tree it is the top of is read before any part of it. The
            caller's one list of them, since references are counted (`held_among`).
        fewest: The fewest elements an expression's operations run over for it to be read.

    Returns:
        (the expressions of two operations or more, the numbers of the
        nodes inside them). Each node is in one expression at most; a node
        inside one is not the root of another.
    """
    finder = _Finder(todo, fewest)
    found = []
    for node in todo:
        if node.number in finder.inside:
            continue
        mark = len(finder.journal)
        expression = finder.expression(node)
        if expression is not None and (count_operations(expression.term) > 1 or expression.result == STORE):
            found.append(expression)
        else:
            finder.give_up(mark)
    return found, finder.inside


def _number(node: Node) -> int:
    """Return a node's number, to sort by."""
    return node.number


def _elements(shape: tuple[int, ...]) -> int:
    """Count the elements of an array of a shape.

    Args:
        shape: A shape whose lengths are all known.

    Returns:
        Their product.
    """
    count = 1
    for length in shape:
        count *= length
    return count


def count_operations(part: Term | Leaf) -> int:
    """Count the operations of a tree.

    Args:
        part: A tree or a leaf.

    Returns:
        How many terms it has.
    """
    if type(part) is Leaf:
        return 0
    return 1 + sum(count_operations(operand) for operand in part.operands)


@dataclass(frozen=True, slots=True)
class Shape:
    """A tree cut out of an expression: the operations one kernel call would make."""

    # Its name: the operations as written, with numbered leaves, such as "add(mul(S0,V0),V1)".
    name: str
    # The term at its top.
    top: Term
    # What each leaf of the name is, in the order of their numbers within each kind: a term whose value is read as
    # an array, or a leaf of the expression.
    vectors: tuple
    scalars: tuple
    # The terms whose operations it makes, the top first.
    terms: tuple

    @property
    def operations(self) -> int:
        """How many operations it has."""
        return len(self.terms)


def _cuts(part: Term | Leaf, budget: int) -> Iterator[tuple[Any, int]]:
    """List the ways to cut a tree off below a term, within a number of operations.

    Args:
        part: A tree or a leaf of an expression.
        budget: The most operations the cut tree may have.

    Yields:
        (the cut tree, its operations). The cut tree is a nested tuple
        `(op, parts, term)` where it keeps an operation, and the leaf or the
        term itself where it reads a value: a term kept whole stands for
        the array it gives.
    """
    if type(part) is Leaf:
        yield part, 0
        return
    if part.op not in MASKS:
        yield part, 0  # A comparison gives a mask, which no kernel reads as an array: it is kept or nothing is.
    if budget < 1:
        return
    for parts, used in _combinations(part.operands, budget - 1):
        yield (part.op, parts, part), used + 1


def _combinations(operands: tuple, budget: int) -> Iterator[tuple[tuple, int]]:
    """List the ways to cut each of several operands, within one budget for all.

    Args:
        operands: The operands of one term.
        budget: The most operations the cut operands may have together.

    Yields:
        (one cut per operand, the operations they have together).
    """
    if not operands:
        yield (), 0
        return
    for first, used in _cuts(operands[0], budget):
        for rest, more in _combinations(operands[1:], budget - used):
            yield (first,) + rest, used + more


def _structure(cut: Any) -> str:
    """Write a cut tree without numbering its leaves, to order the operands of a commutative operation by.

    Args:
        cut: A cut tree from `_cuts`.

    Returns:
        Text such as "mul(S,V)".
    """
    if type(cut) is Leaf:
        return cut.kind
    if type(cut) is Term:
        return VECTOR
    op, parts, _ = cut
    texts = [_structure(part) for part in parts]
    if op in COMMUTATIVE:
        texts.sort()
    return f"{op}({','.join(texts)})"


def _named(cut: Any, vectors: list, scalars: list, terms: list) -> str:
    """Write a cut tree with its leaves numbered in the order they are met, collecting what each leaf is.

    Args:
        cut: A cut tree from `_cuts`.
        vectors: The values read as arrays so far; grows.
        scalars: The numbers read so far; grows.
        terms: The terms whose operations the tree keeps, so far; grows.

    Returns:
        The tree's name. The operands of a commutative operation are
        written in the order of their structure, so `c + a * b` and
        `a * b + c` have one name.
    """
    if type(cut) is Leaf and cut.kind == SCALAR:
        scalars.append(cut)
        return f"{SCALAR}{len(scalars) - 1}"
    if type(cut) is Leaf or type(cut) is Term:
        # Each place an array is read is a leaf of its own, even where two read one array: a kernel takes a
        # pointer per leaf, and may be given the same one twice. So `a + s * a` is served by the kernel of
        # `a + s * b`, and a shape's name does not depend on which of its arrays happen to be the same.
        vectors.append(cut)
        return f"{VECTOR}{len(vectors) - 1}"
    op, parts, term = cut
    terms.append(term)
    if op in COMMUTATIVE:
        parts = sorted(parts, key=_structure)
    return f"{op}({','.join(_named(part, vectors, scalars, terms) for part in parts)})"


def shapes_at(term: Term, most: int = MOST_OPERATIONS) -> list[Shape]:
    """List the trees of two operations or more that can be cut out of an expression with a term at their top.

    Args:
        term: A term of an expression.
        most: The most operations a tree may have.

    Returns:
        One `Shape` per tree.
    """
    found = []
    for cut, used in _cuts(term, most):
        if used < 2:
            continue
        found.append(_shape(cut))
    return found


def _shape(cut: Any) -> Shape:
    """Name a cut tree and collect what it reads and what it keeps.

    Args:
        cut: A cut tree from `_cuts` that keeps its top term.

    Returns:
        The shape.
    """
    vectors: list = []
    scalars: list = []
    terms: list = []
    name = _named(cut, vectors, scalars, terms)
    return Shape(name, cut[2], tuple(vectors), tuple(scalars), tuple(terms))


def whole_shape(expression: Expression) -> Shape:
    """Return the shape that keeps every operation of an expression.

    Args:
        expression: An expression.

    Returns:
        The shape, whose name is the expression's.
    """
    return _shape(_whole(expression.term))


def terms_of(part: Term | Leaf) -> Iterator[Term]:
    """List the terms of a tree, the top first.

    Args:
        part: A tree or a leaf.

    Yields:
        Each term.
    """
    if type(part) is Term:
        yield part
        for operand in part.operands:
            yield from terms_of(operand)


def canonical(expression: Expression) -> str:
    """Name a whole expression as a pattern would be named.

    Args:
        expression: An expression.

    Returns:
        The name of the tree that keeps every operation.
    """
    return whole_shape(expression).name


def _whole(part: Term | Leaf) -> Any:
    """Cut nothing off a tree: the cut tree that keeps every operation.

    Args:
        part: A tree or a leaf.

    Returns:
        The tree in the form `_cuts` yields.
    """
    if type(part) is Leaf:
        return part
    return part.op, tuple(_whole(operand) for operand in part.operands), part


@dataclass(frozen=True, slots=True)
class Pattern:
    """A shape of the library as a tree, to hold against the terms of an expression."""

    # The operation's name, or empty for a leaf.
    op: str
    # The operation's operands, as patterns.
    parts: tuple = ()
    # For a leaf: VECTOR or SCALAR, and its number among the shape's leaves of that kind.
    kind: str = ""
    place: int = 0


def parse(name: str) -> Pattern:
    """Read a shape's name as a tree.

    Args:
        name: A name as `whole_shape` writes one, such as "add(V0,mul(S0,V1))".

    Returns:
        The tree.
    """
    return _parse(name, 0)[0]


def _parse(text: str, start: int) -> tuple[Pattern, int]:
    """Read one operation or leaf of a shape's name.

    Args:
        text: The name.
        start: Where the operation or leaf begins.

    Returns:
        (the tree, the place in the text after it).
    """
    end = start
    while end < len(text) and text[end] not in "(,)":
        end += 1
    word = text[start:end]
    if end == len(text) or text[end] != "(":
        return Pattern("", (), word[0], int(word[1:])), end
    parts = []
    at = end + 1
    while True:
        part, at = _parse(text, at)
        parts.append(part)
        at += 1  # Past the comma after the operand, or the bracket that closes the operation.
        if text[at - 1] == ")":
            return Pattern(word, tuple(parts)), at


def _fits(pattern: Pattern, part: Term | Leaf, vectors: list, scalars: list, terms: list) -> bool:
    """Hold a pattern against part of an expression.

    Args:
        pattern: A pattern, or part of one.
        part: A term or a leaf of an expression.
        vectors: What the pattern's vector leaves read, by their numbers; filled in where the pattern fits.
        scalars: The same for its scalar leaves.
        terms: The terms whose operations the pattern makes; grows where it fits.

    Returns:
        True when the pattern fits. A vector leaf fits an array the
        expression reads and any term that gives one, whose value is then
        computed first; a scalar leaf fits a number alone. The operands of
        a commutative operation are tried in both orders. Where it does
        not fit, `terms` is as it was; leaves filled in on the way are
        left, and are written again by whatever fits later.
    """
    if not pattern.op:
        if pattern.kind == SCALAR:
            if type(part) is not Leaf or part.kind != SCALAR:
                return False
            scalars[pattern.place] = part
            return True
        if type(part) is Leaf:
            if part.kind != VECTOR:
                return False
        elif part.op in MASKS:
            return False  # A comparison gives a mask, which no kernel reads as an array.
        vectors[pattern.place] = part
        return True
    if type(part) is not Term or part.op != pattern.op:
        return False
    parts, operands = pattern.parts, part.operands
    if len(parts) != len(operands):
        return False
    mark = len(terms)
    terms.append(part)
    for sub, operand in zip(parts, operands):
        if not _fits(sub, operand, vectors, scalars, terms):
            break
    else:
        return True
    del terms[mark + 1:]
    if (pattern.op in COMMUTATIVE and _fits(parts[0], operands[1], vectors, scalars, terms)
            and _fits(parts[1], operands[0], vectors, scalars, terms)):
        return True
    del terms[mark:]
    return False


@dataclass(frozen=True, slots=True)
class Entry:
    """One shape of the library, ready to be held against expressions."""

    # The shape's name and the same as a tree.
    name: str
    pattern: Pattern
    # What computes it.
    kernel: Any
    # How many vector leaves and scalar leaves the shape has.
    vectors: int
    scalars: int


def index(library: dict[str, Any]) -> dict[str, list[Entry]]:
    """Arrange a library by the operation at the top of each shape, which is what a term is looked up by.

    Args:
        library: Shape name -> what computes it.

    Returns:
        Operation name -> the entries whose shape has it at the top, the largest shapes first.
    """
    by_top: dict[str, list[Entry]] = {}
    for name, kernel in library.items():
        pattern = parse(name)
        entry = Entry(name, pattern, kernel, name.count(VECTOR), name.count(SCALAR))
        by_top.setdefault(pattern.op, []).append(entry)
    for entries in by_top.values():
        entries.sort(key=lambda entry: entry.name.count("("), reverse=True)
    return by_top


@dataclass(frozen=True, slots=True)
class Tile:
    """One kernel call of a tiling: a shape of the library, placed on an expression."""

    # The shape as placed: its top term and what its leaves read.
    shape: Shape
    # The library's entry for the shape's name.
    kernel: Any


def tile(term: Term, library: dict[str, list[Entry]]) -> tuple[int, dict[int, Tile]]:
    """Cover a tree with shapes of a library, by the fewest passes over memory.

    Every operation is computed by one pass: a tile's, or NumPy's own call
    for an operation no tile covers. The fewest passes are found from the
    leaves up: at each term, every shape of the library with that term's
    operation at its top is held against the tree.

    Args:
        term: The top of an expression, or of part of one.
        library: The library, from `index`.

    Returns:
        (the number of passes, the tiles by the number of the node at each
        tile's top). A term that is the top of no tile is computed by its
        node's own call.
    """
    best: dict[int, tuple[int, Tile | None]] = {}
    total = _passes(term, library, best)
    tiles: dict[int, Tile] = {}
    _collect(term, best, tiles)
    return total, tiles


def _passes(part: Term, library: dict[str, list[Entry]], best: dict[int, tuple[int, Tile | None]]) -> int:
    """Find the fewest passes for the tree below a term, remembering how.

    A function of the module and not one made inside `tile`: a function
    that calls itself through a name of the function it was made in is part
    of a cycle of references, which Python frees only when its collector
    next runs. Everything a tiling touches would stay alive until then, the
    nodes of the expression and the arrays they read among them.

    Args:
        part: A term of the expression.
        library: The library, from `index`.
        best: The answers so far, by the identity of the term: (passes, the tile at its top or None).

    Returns:
        The number of passes.
    """
    known = best.get(id(part))
    if known is not None:
        return known[0]
    cost = 1
    for operand in part.operands:
        if type(operand) is Term:
            cost += _passes(operand, library, best)
    choice: tuple[int, Tile | None] = (cost, None)
    for entry in library.get(part.op, ()):
        vectors: list = [None] * entry.vectors
        scalars: list = [None] * entry.scalars
        terms: list = []
        if not _fits(entry.pattern, part, vectors, scalars, terms):
            continue
        cost = 1
        for vector in vectors:
            if type(vector) is Term:
                cost += _passes(vector, library, best)
        if cost < choice[0]:
            choice = (cost, Tile(Shape(entry.name, part, tuple(vectors), tuple(scalars), tuple(terms)), entry.kernel))
    best[id(part)] = choice
    return choice[0]


def _collect(part: Term, best: dict[int, tuple[int, Tile | None]], tiles: dict[int, Tile]) -> None:
    """Walk the chosen tiling from a term down, collecting its tiles.

    Args:
        part: A term of the expression.
        best: What `_passes` found.
        tiles: The tiles by the number of the node at each one's top; grows.
    """
    chosen = best[id(part)][1]
    if chosen is None:
        below = [operand for operand in part.operands if type(operand) is Term]
    else:
        tiles[part.node.number] = chosen
        below = [vector for vector in chosen.shape.vectors if type(vector) is Term]
    for operand in below:
        _collect(operand, best, tiles)


# ---- The library: one kernel per shape, each shape read off its kernel's own oracle. ----

# The arrays and numbers a kernel's oracle is run on to read its shape: their values are never computed. The
# numbers are no whole numbers, since a whole exponent or factor can be a different operation.
PROBE_LENGTH, PROBE_NUMBER, PROBE_STEP = 4, 1.3, 0.25
# How the name of a shape that ends in a sum, and in a mean, starts.
SUM_PREFIX, MEAN_PREFIX = "sum(", "mean("


class NoPattern(ValueError):
    """Raised for a kernel whose oracle is not one expression of operations a kernel can make."""


@dataclass(frozen=True, slots=True)
class Kernel:
    """One fused kernel, as the entry of a shape in the library."""

    # The kernel's name.
    name: str
    # Its place in the table of `_fused`.
    index: int
    # For each array the kernel takes, in the order of its arguments, the number of the shape's vector leaf it reads.
    vectors: tuple[int, ...]
    # The same for its numbers and the shape's scalar leaves.
    scalars: tuple[int, ...]
    # True for a kernel that returns one number.
    reduces: bool
    # True for the entry of a mean, which is the kernel's sum divided by the number of elements.
    mean: bool = False


def pattern_of(oracle: Any, vectors: int, scalars: int) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """Read the shape a kernel computes off its oracle, by running the oracle on lazy arrays.

    The oracle is the NumPy expression the judge holds the kernel to. Run
    on lazy arrays it writes the graph of that expression, and the shape of
    that graph is the pattern the kernel serves: the specification is its
    own pattern, and the two cannot disagree.

    Args:
        oracle: The spec's oracle: `oracle(*arrays, *numbers)`.
        vectors: How many arrays it takes.
        scalars: How many numbers it takes after them.

    Returns:
        (the shape's name, for each array argument the number of the vector
        leaf it is, for each number argument the number of the scalar leaf).

    Raises:
        NoPattern: When the oracle's graph is not one expression that keeps every operation.
    """
    arrays = [array.wrap(np.full(PROBE_LENGTH, 1.0 + place)) for place in range(vectors)]
    numbers = [PROBE_NUMBER + PROBE_STEP * place for place in range(scalars)]
    result = oracle(*arrays, *numbers)
    if not isinstance(result, array.LazyArray) or not result._node.pending:
        raise NoPattern("the oracle computed its result: an operation in it is not deferred")
    root = result._node
    todo = evaluate._needed([root])
    expressions, _ = find(todo)
    if len(expressions) != 1 or expressions[0].root is not root or count_operations(expressions[0].term) != len(todo):
        raise NoPattern("the oracle is not one expression of operations a kernel can make")
    expression = expressions[0]
    shape = whole_shape(expression)
    nodes = [lazy._node for lazy in arrays]
    read_vectors = [expression.operands[leaf.place] for leaf in shape.vectors]
    read_scalars = [expression.operands[leaf.place] for leaf in shape.scalars]
    # By identity for the arrays: two lazy arrays are never equal to Python, they give a lazy array of booleans.
    vector_order = tuple(_place_of(node, read_vectors, by_identity=True) for node in nodes)
    scalar_order = tuple(_place_of(number, read_scalars, by_identity=False) for number in numbers)
    if len(read_vectors) != vectors or len(read_scalars) != scalars:
        raise NoPattern("the oracle reads an argument twice, or a number that is not an argument")
    return shape.name, vector_order, scalar_order


def _place_of(argument: Any, read: list, by_identity: bool) -> int:
    """Find which leaf of a shape reads one argument of an oracle.

    Args:
        argument: The node of an array the oracle was given, or a number it was given.
        read: What the shape's leaves of that kind read, in the order of their numbers.
        by_identity: Whether a leaf reads the argument when it is the same object, and not merely equal to it.

    Returns:
        The leaf's number.

    Raises:
        NoPattern: When no leaf reads the argument.
    """
    for place, operand in enumerate(read):
        if operand is argument or (not by_identity and operand == argument):
            return place
    raise NoPattern("the oracle does not read every one of its arguments")


# The module that calls the kernels, once `enable` has imported it.
_fused: Any = None
# The library in use: shape name -> kernel. Empty until `enable` has loaded it.
LIBRARY: dict[str, Kernel] = {}
# The same arranged for `tile`.
_BY_TOP: dict[str, list[Entry]] = {}


def load_library(module: Any) -> dict[str, Kernel]:
    """Read the shape of every kernel the built module has.

    Args:
        module: `_fused`, built from the kernels' directories.

    Returns:
        Shape name -> kernel. A kernel that sums has a second entry for the
        mean of the same expression.
    """
    from . import build_fused
    table = {row[0]: place for place, row in enumerate(module.kernels())}
    found: dict[str, Kernel] = {}
    for fused in build_fused.kernels():
        if fused.name not in table:
            continue  # Added since the module was built.
        name, vectors, scalars = pattern_of(fused.spec.oracle, fused.vectors, fused.scalars)
        kernel = Kernel(fused.name, table[fused.name], vectors, scalars, fused.reduces)
        found[name] = kernel
        if name.startswith(SUM_PREFIX):
            found[MEAN_PREFIX + name[len(SUM_PREFIX):]] = Kernel(fused.name, kernel.index, vectors, scalars, True, mean=True)
    return found


# ---- Planning one computation, and computing its groups. ----

# The fewest elements an expression has for its operations to be fused. Below it the calls are left to NumPy:
# reading an expression off the graph costs about two microseconds a node, and a pass of NumPy's over fewer
# elements than this costs no more than that, so there is nothing to win back.
MIN_ELEMENTS = 16_384
# With fusion on, a call that gives fewer elements than this is made as the graph is built, not written as a
# node (`array.at_once_below`). The same count as MIN_ELEMENTS: below it nothing is planned, so a node would only
# cost its writing and its computing. Read by `enable`.
AT_ONCE_BELOW = MIN_ELEMENTS
# Lines of the program remembered at most as ones whose kernel raised an exception NumPy reports.
FLAGGED_LIMIT = 4096
# The floating-point exceptions `_fused` reports, by the name `np.errstate` has for each. Filled by `enable`.
_EXCEPTIONS: dict[str, int] = {}
# What `np.errstate` calls leaving an exception unreported.
IGNORED = "ignore"


@dataclass
class Stats:
    """What fusion did, for the tools that measure it."""

    # Computations planned, and nanoseconds spent planning them.
    plans: int = 0
    planning_ns: int = 0
    # Expressions found, and those of them at least one kernel call was planned for.
    expressions: int = 0
    tiled: int = 0
    # Kernel calls made, by kernel name.
    calls: dict[str, int] = field(default_factory=dict)
    # Nodes computed inside a kernel call instead of by a call of their own.
    fused_nodes: int = 0
    # Kernel calls given up: the arrays were laid out in a way the kernels do not read, or an exception was
    # raised that NumPy would have reported.
    declined: int = 0
    flagged: int = 0
    # Writes turned into one NumPy call: into a whole array, and into part of one.
    whole_writes: int = 0
    part_writes: int = 0

    def reset(self) -> None:
        """Start counting from nothing."""
        self.__init__()


# The one set of counters of the process.
STATS = Stats()
# The lines of the program whose kernel call raised a floating-point exception that NumPy reports. Such a call's
# result is dropped and NumPy's calls are made, so the expression is computed twice; a line that did it once is
# left to NumPy from then on, which is what a loop over data with an infinity or a division by zero needs.
_flagged: set[graph.Origin] = set()


class _StoreCall:
    """What a write into part of an array calls once its one operation is made straight into the part.

    `x[i] = a + b` is, as written, the sum into a temporary and the
    temporary assigned into a copy of `x`. This makes the sum with the part
    of the copy as its `out=`: the same loop on the same operands, and no temporary.
    """

    __slots__ = ("target", "ufunc")

    def __init__(self, target: ViewOf, ufunc: np.ufunc) -> None:
        """Remember where the result goes and what makes it.

        Args:
            target: The part written into, as steps from the array.
            ufunc: The operation whose result is stored.
        """
        # The steps alone, under no node: a reference to the node of the old contents kept here would be
        # counted as a reader of them, and the write would never be made in place (`evaluate.reusable`).
        self.target = ViewOf(None, target.steps)
        self.ufunc = ufunc

    def __call__(self, old: np.ndarray, *values: Any) -> np.ndarray:
        """Make the operation into the part of a copy of what the array held.

        Args:
            old: The array's contents before the write.
            *values: The operation's operands.

        Returns:
            The array's contents after the write.
        """
        new, part = _copy_for(self.target, old)
        self.ufunc(*values, out=part)
        return new

    def in_place(self, old: np.ndarray, *values: Any) -> np.ndarray:
        """Make the operation into the part of the memory the array has.

        Args:
            old: The array's contents before the write, in an array nothing else will read.
            *values: The operation's operands. One that is a view of `old` may overlap the part; NumPy then
                computes as if it did not, which is what the assignment of a temporary gives.

        Returns:
            The same array, written into.
        """
        self.ufunc(*values, out=self.target.of(old))
        return old


def _copy_for(target: ViewOf, old: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Make the array a write gives, ready for the part to be written.

    Args:
        target: The part written into, as steps from the array.
        old: The array's contents before the write.

    Returns:
        (the new array, the part of it to write). The new array is a copy
        of the old one, as the write is defined. Where the part is all of
        it, every element is about to be written and nothing is copied: the
        new array is empty memory of the copy's layout.
    """
    new = np.empty_like(old) if target.of(old).size == old.size else old.copy(order=array.KEPT_ORDER)
    return new, target.of(new)


class _Group:
    """The nodes one kernel call computes: a tile of an expression, with what it reads."""

    __slots__ = ("kernel", "sources", "members", "watched", "result", "target", "shape")

    def __init__(self, kernel: Kernel, sources: list, members: list[Node], watched: int, result: str,
                 target: ViewOf | None, shape: tuple[int, ...]) -> None:
        """Remember one planned kernel call.

        Args:
            kernel: The kernel.
            sources: What it reads, in the order of its arguments: nodes, `ndarray`s, numbers, a `ViewOf`.
            members: The nodes inside the tile, oldest first, without the node the tile gives the value of.
            watched: The floating-point exceptions NumPy would report for these nodes, as `_fused`'s bits.
            result: ARRAY, STORE, or a name of REDUCTIONS.
            target: For STORE, the part of the array written into.
            shape: The shape the operations run over.
        """
        self.kernel, self.sources, self.members, self.watched = kernel, sources, members, watched
        self.result, self.target, self.shape = result, target, shape

    def operands(self) -> tuple | None:
        """Collect the values the kernel reads.

        Returns:
            The operands as `_fused` takes them, or None when one is not
            there: a node that failed, or was never computed.
        """
        values = []
        for source in self.sources:
            kind = type(source)
            if kind is Node:
                if source.operation is not None or source.failure is not None:
                    return None
                source = source.value
            elif kind is ViewOf:
                old = source.old
                if old.operation is not None or old.failure is not None:
                    return None
                source = source.of(old.value)
            if type(source) is not np.ndarray or not source.ndim:
                source = float(source)
            values.append(source)
        return tuple(values)

    def _written(self, node: Node) -> Node:
        """Find the write whose array the group's result belongs in.

        Args:
            node: The write the group gives the value of.

        Returns:
            `node`, unless what its array held is itself a write of the
            whole array that the group computes inside: `np.add(a, b,
            out=x); np.multiply(x, 0.5, out=x)` is one kernel call, and its
            result belongs where `x` was before the first of the two. Then
            the earliest such write.
        """
        write = node
        while True:
            old = write.args[0]
            if type(old) is Node and type(old.operation) is _WholeCall and any(old is member for member in self.members):
                write = old
            else:
                return write

    def _held(self, write: Node) -> int:
        """Count the references this group and its members have to the node a write's array held.

        Args:
            write: The write whose array the group's result belongs in, from `_written`.

        Returns:
            How many references to the write's first operand are the
            plan's own, which `evaluate.reusable` then does not take for
            readers: the other nodes inside the group that read it, each
            place it is among what the kernel reads, and the record of the
            part written into.
        """
        old = write.args[0]
        held = sum(1 for source in self.sources if source is old)
        held += sum(1 for member in self.members if member is not write for arg in member.args if arg is old)
        parts = {id(source) for source in self.sources if type(source) is ViewOf and source.old is old}
        if self.target is not None and self.target.old is old:
            parts.add(id(self.target))
        return held + len(parts)

    def _may_overwrite(self, write: Node) -> bool:
        """Say whether the kernel may write its result into the memory the written array has.

        A kernel's result is dropped when it raised a floating-point
        exception that NumPy reports, and NumPy's own calls are then made
        on the same operands. That is only possible while the operands are
        what they were. So the kernel writes where the array is only when
        it reads nothing of what the array held (`x[i] = a * (b - c)`),
        or when no exception is reported under the settings the lines were
        written under. Otherwise its result goes to a copy, or to new
        memory, as before.

        Args:
            write: The write whose array the result belongs in, from `_written`.

        Returns:
            True when both that holds and nothing else reads what the array held (`evaluate.reusable`).
        """
        held = self._held(write)
        # The record of the part written into refers to the old node and reads nothing of it. No name here may
        # refer to that node while its references are counted.
        reading = held - (1 if self.target is not None and self.target.old is write.args[0]
                          and not any(source is self.target for source in self.sources) else 0)
        if reading and self.watched:
            return False
        return evaluate.reusable(write, held)

    def _elementwise(self, write: Node | None, operands: tuple, reuse: bool) -> tuple[int, Any]:
        """Make the group's kernel call into the memory its result belongs in.

        Args:
            write: The write whose array the result belongs in, from `_written`; None when the group's node is
                no write.
            operands: What the kernel reads.
            reuse: Whether that array may be written where it is.

        Returns:
            (what `_fused.elementwise` returned, the node's value). For a
            write the value is the array written into: the one the node's
            first operand holds when `reuse` says so, where the kernel's
            call is then the only pass over memory, and otherwise a copy of
            it, or new memory where every element is written.
        """
        if write is None:
            value = out = np.empty(self.shape)
            return _fused.elementwise(self.kernel.index, operands, out), value
        old = write.args[0]
        if type(old) is not Node or old.operation is not None or old.failure is not None:
            return _fused.DECLINED, None
        memory = old.value
        if self.result == STORE:
            value, out = (memory, self.target.of(memory)) if reuse else _copy_for(self.target, memory)
        else:
            value = out = memory if reuse else np.empty_like(memory)
        raised = _fused.elementwise(self.kernel.index, operands, out)
        if reuse and raised != _fused.DECLINED and not raised & self.watched:
            # The old node's memory is this node's now.
            old.value = None
            evaluate.WRITES["in_place"] += 1
        return raised, value

    def run(self, node: Node) -> bool:
        """Compute the group's node with one kernel call.

        Args:
            node: The pending node the group gives the value of.

        Returns:
            True when the node has its value. False when the call was not
            made or its result was dropped: an operand is missing, the
            arrays are laid out in a way the kernels do not read, or the
            kernel raised an exception NumPy would have reported, in which
            case NumPy's own calls are what report it.
        """
        # Asked before the operands are collected: a view of the array among them would count as a reader of it.
        write = self._written(node) if node.kind == rules.WRITE else None
        reuse = write is not None and self._may_overwrite(write)
        operands = self.operands()
        if operands is None:
            return False
        kernel = self.kernel
        if kernel.reduces:
            answer = _fused.reduce(kernel.index, operands)
            if answer == _fused.DECLINED:
                STATS.declined += 1
                return False
            total, raised = answer
            count = 1
            for length in self.shape:
                count *= length
            value: Any = np.float64(total / count) if kernel.mean else np.float64(total)
        else:
            raised, value = self._elementwise(write, operands, reuse)
            if raised == _fused.DECLINED:
                STATS.declined += 1
                return False
        if raised & self.watched:
            STATS.flagged += 1
            if len(_flagged) >= FLAGGED_LIMIT:
                _flagged.clear()
            if node.origin is not None:
                _flagged.add(node.origin)
            return False
        node.value = value
        graph.settle(node)
        STATS.calls[kernel.name] = STATS.calls.get(kernel.name, 0) + 1
        STATS.fused_nodes += len(self.members)
        # The nodes inside are done with. Nothing here may keep them alive past the call.
        self.sources, self.members = (), ()
        return True


class Plan:
    """Which nodes of one computation are computed together: what `evaluate` asks a planner for."""

    __slots__ = ("inner", "groups")

    def __init__(self) -> None:
        """Start with nothing planned."""
        # The numbers of the nodes computed inside a group.
        self.inner: set[int] = set()
        # The groups, by the number of the node each gives the value of.
        self.groups: dict[int, _Group] = {}

    def run(self, group: _Group, node: Node) -> bool:
        """Compute a group's node with its kernel. See `_Group.run`."""
        return group.run(node)

    def members(self, group: _Group) -> list[Node]:
        """Return the nodes inside a group whose kernel call was not made, oldest first, and let go of them.

        Args:
            group: A group whose `run` returned False.

        Returns:
            The nodes, for `evaluate` to compute one by one. The group
            forgets them and what it reads: it will not run again, and a
            reference kept here would keep arrays in memory until the whole
            computation is over, where NumPy's own calls free each as soon
            as its last reader has run.
        """
        nodes = list(group.members)
        group.sources, group.members, group.target = (), (), None
        return nodes


# The exceptions NumPy would report under each error setting met so far, by the identity of the setting's
# dictionary. An entry holds the dictionary, so that the identity stays its own.
_watched: dict[int, tuple[dict, int]] = {}


def _watched_under(errstate: dict | None) -> int:
    """Return the floating-point exceptions NumPy reports under an error setting.

    Args:
        errstate: The `np.errstate` a node was written under.

    Returns:
        `_fused`'s bits for the exceptions that are warned of, raised or
        otherwise acted on; an exception that is ignored changes nothing.
    """
    if errstate is None:
        return 0
    known = _watched.get(id(errstate))
    if known is not None and known[0] is errstate:
        return known[1]
    bits = 0
    for name, bit in _EXCEPTIONS.items():
        if errstate.get(name, IGNORED) != IGNORED:
            bits |= bit
    _watched[id(errstate)] = (errstate, bits)
    return bits


def plan(todo: list[Node]) -> Plan | None:
    """Plan one computation: which of its nodes a kernel call computes together.

    Args:
        todo: The pending nodes about to be computed, newest first. The caller's one list of them.

    Returns:
        The plan, or None when nothing in the computation is fused.
    """
    started = time.perf_counter_ns()
    STATS.plans += 1
    rewritten, large = _rewrite_writes(todo, MIN_ELEMENTS)
    STATS.whole_writes += rewritten
    if not large:
        # Nothing here is large enough to be worth reading expressions off: the computation is NumPy's calls.
        STATS.planning_ns += time.perf_counter_ns() - started
        return None
    expressions, _ = find(todo, MIN_ELEMENTS)
    planned = Plan()
    for expression in expressions:
        STATS.expressions += 1
        _plan_expression(expression, planned)
    STATS.planning_ns += time.perf_counter_ns() - started
    return planned if planned.groups or planned.inner else None


def _rewrite_writes(todo: list[Node], fewest: int) -> tuple[int, bool]:
    """Turn every write into a whole float64 array among some nodes into its one call, and look at their sizes.

    A function of its own so that no name of this module still refers to a
    node when references are counted: `held_among` would take the node for
    one the program holds.

    Args:
        todo: The pending nodes about to be computed.
        fewest: The fewest elements an expression's operations run over for it to be read.

    Returns:
        (how many writes were rewritten, whether any node gives an array of
        `fewest` elements or more). An expression's operations all give
        arrays of its shape, so where no node does there is no expression
        worth reading. A length that is not known counts as large.
    """
    rewritten, large = 0, fewest <= 1
    for node in todo:
        if node.kind == rules.WRITE and rewrite_write(node):
            rewritten += 1
        if not large and _elements(node.shape) >= fewest:
            large = True
    return rewritten, large


def _plan_expression(expression: Expression, planned: Plan) -> None:
    """Plan the kernel calls of one expression.

    Args:
        expression: The expression.
        planned: The computation's plan, which gains the expression's groups.
    """
    _, tiles = tile(expression.term, _BY_TOP)
    top, root = expression.term, expression.root
    stores = expression.result == STORE
    if tiles:
        STATS.tiled += 1
    for placed in tiles.values():
        shape, kernel = placed.shape, placed.kernel
        read = [_source(expression, part) for part in shape.vectors]
        numbers = [_source(expression, part) for part in shape.scalars]
        sources = [read[place] for place in kernel.vectors] + [numbers[place] for place in kernel.scalars]
        at_top = shape.top is top
        covered = [term.node for term in shape.terms]
        # A stored expression's value is the write's: the node at the top of the tree is inside the group too.
        own = root if at_top and stores else shape.top.node
        if own.origin in _flagged:
            continue  # This line's kernel raised an exception NumPy reports: its operations are NumPy's now.
        members = sorted((node for node in covered if node is not own), key=_number)
        watched = _watched_under(own.errstate)
        for node in members:
            watched |= _watched_under(node.errstate)
        result = STORE if at_top and stores else ARRAY
        if kernel.reduces:
            result = shape.top.op
        planned.groups[own.number] = _Group(kernel, sources, members, watched, result,
                                            expression.target if result == STORE else None, expression.shape)
        planned.inner.update(node.number for node in members)
    if stores and top.node.number not in tiles and top.node is not root and _store_directly(expression):
        planned.inner.add(top.node.number)
        STATS.part_writes += 1


def _source(expression: Expression, part: Term | Leaf) -> Any:
    """Return what a tile reads for one of its leaves.

    Args:
        expression: The expression the tile is placed on.
        part: A leaf of the expression, or a term whose value the tile reads as an array.

    Returns:
        The expression's operand for a leaf, and the node that gives the term's value for a term.
    """
    return expression.operands[part.place] if type(part) is Leaf else part.node


def _store_directly(expression: Expression) -> bool:
    """Turn `x[index] = op(...)` into the operation made straight into the part, when no kernel covers its top.

    The write's node takes the operation's operands in place of the
    operation's result, and the operation's node is then not computed. That
    is done only when the two were written on one line under one error
    setting, so that a warning is reported where it would have been.

    Args:
        expression: A stored expression whose top term no tile covers.

    Returns:
        True when the write's node was rewritten.
    """
    root, stored = expression.root, expression.term.node
    call = call_of(stored)
    # Equal, not the same object: two places on one line each have a record of their own.
    if call is None or not isinstance(call[0], np.ufunc) or stored.origin != root.origin or stored.errstate != root.errstate:
        return False
    root.operation, root.args = _StoreCall(expression.target, call[0]), (root.args[0],) + tuple(call[1])
    return True


def enable() -> bool:
    """Have computations planned for fusion from now on.

    Returns:
        True when the fused kernels are built and in use. False when the
        module that calls them is not built (`python -m
        prototypes.lazy.build_fused`): nothing changes then.
    """
    global _fused
    if _fused is None:
        try:
            from . import _fused as module
        except ImportError:
            return False
        _fused = module
        _EXCEPTIONS.update(invalid=module.INVALID, divide=module.DIVIDE, over=module.OVER, under=module.UNDER)
        LIBRARY.update(load_library(module))
        _BY_TOP.update(index(LIBRARY))
    evaluate.planner = plan
    array.at_once_below(AT_ONCE_BELOW)
    return True


def disable() -> None:
    """Have every node computed by its own call again, and every operation written as a node."""
    evaluate.planner = None
    array.at_once_below(0)


def is_enabled() -> bool:
    """Say whether computations are planned for fusion."""
    return evaluate.planner is plan
