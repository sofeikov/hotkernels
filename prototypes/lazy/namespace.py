"""The stand-in for the `numpy` module: the same names, and every array it makes is lazy.

    np = get_lazy()             # where `import numpy as np` stood
    x = np.zeros(1000)          # a lazy array
    y = np.exp(x) + 1           # deferred

A lazy array already makes `np.exp(x)` lazy when `np` is the real module,
because NumPy passes the call to it. What the real module cannot do is make
an array lazy when no lazy array is among the arguments: `np.zeros(3)`,
`np.arange(10)`, `np.array(data)`. That is all the stand-in adds.

An `ndarray` that exists already enters through it too, held by reference:

    x = np.asarray(existing_ndarray)

`get_lazy(strict=True)` gives a stand-in whose arrays, and everything computed
from them, raise `NotGathered` where a value is needed and `gather()` was not
called, instead of computing.

Nothing is patched: `get_lazy()` returns an object, the line above binds a
name in the module that wrote it, and `sys.modules["numpy"]` and NumPy's own
functions are as they were. Another module's `import numpy as np` is the real
module, and its calls still reach a lazy array that is handed to them.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import numpy as np

from . import array
from .array import LazyArray, _at_once, wrap

# Functions that turn their first argument into an array and have no protocol
# through which a lazy array could answer: NumPy would ask the lazy array for
# its values. The stand-in answers them itself.
CONVERTERS = frozenset({"array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray"})
# The converters that return their argument unchanged when it is an array already.
PASS_THROUGH = frozenset({"asarray", "asanyarray"})
# NumPy's random generators, whose methods make arrays out of the module's sight.
RANDOM_GENERATORS = (np.random.Generator, np.random.RandomState)
# Types of an argument that is no array and holds none: a call's arguments are mostly these and arrays.
NO_ARRAY = frozenset({int, float, bool, complex, str, type(None)})


def get_lazy(strict: bool = False) -> "LazyNumPy":
    """Return the stand-in for the `numpy` module. NumPy itself is not changed.

    Args:
        strict: Whether the arrays it makes refuse to give their values
            without `gather()`. By default a value that is needed is
            computed, and the log says so.

    Returns:
        The stand-in.
    """
    return LazyNumPy(np, strict)


def _lifted(arg: Any, strict: bool) -> Any:
    """Make the `ndarray`s of an argument lazy.

    Args:
        arg: One argument of a call made through the stand-in.
        strict: Whether an `ndarray` made lazy here is strict.

    Returns:
        A lazy array for an exact `ndarray`; a list or tuple with its
        `ndarray`s made lazy; anything else unchanged. A subclass of `ndarray`
        is left alone: what NumPy does with it is NumPy's business.
    """
    if type(arg) is np.ndarray:
        return wrap(arg, strict=strict)
    if isinstance(arg, (list, tuple)) and any(type(item) is np.ndarray for item in arg):
        return type(arg)(_lifted(item, strict) for item in arg)
    return arg


def _holds_arrays(arguments: Any) -> bool:
    """Say whether a call's arguments hold an array, at any depth of lists, tuples and dicts.

    Args:
        arguments: Arguments of a call made through the stand-in.

    Returns:
        True for an `ndarray` or a lazy array among them. A function called
        with none has just made whatever array it returns, so nothing
        outside the layer knows that array's memory.
    """
    if isinstance(arguments, (np.ndarray, LazyArray)):
        return True
    if isinstance(arguments, dict):
        arguments = tuple(arguments.values())
    return isinstance(arguments, (list, tuple)) and any(_holds_arrays(item) for item in arguments)


def _made_lazy(result: Any, source: str, strict: bool, owned: bool = False) -> Any:
    """Make the arrays a function returned lazy.

    Args:
        result: What the NumPy function returned.
        source: The function as a reader would name it, such as "np.zeros",
            so a recording can say where an array entered.
        strict: Whether an `ndarray` made lazy here is strict.
        owned: Whether the function has just made the array, from no array:
            a write into it can then be deferred. See `array.wrap`.

    Returns:
        A lazy array for an exact `ndarray`; a tuple or list with its
        `ndarray`s made lazy; anything else unchanged, a NumPy scalar
        included: `np.sqrt(2.0)` stays an `np.float64`.
    """
    if type(result) is np.ndarray:
        return wrap(result, source, strict, owned)
    if type(result) in (list, tuple) and any(type(item) is np.ndarray for item in result):
        return type(result)(_made_lazy(item, source, strict, owned) for item in result)
    if isinstance(result, RANDOM_GENERATORS):
        return LazyRandom(result, strict)
    return result


class LazyCallable:
    """A NumPy function or ufunc whose `ndarray` arguments and results are made lazy.

    Calling it makes the `ndarray` arguments lazy and calls the real one.
    NumPy then passes the call to the lazy arrays, which defer it or make it
    at once; with no array among the arguments (`np.zeros(3)`) NumPy just
    makes the array, which is made lazy. A ufunc keeps its methods:
    `np.add.reduce` is wrapped the same way.
    """

    # Slots, so that the C core can read them where it stands in for `__call__`.
    __slots__ = ("_function", "_strict", "_source")

    def __init__(self, function: Callable[..., Any], strict: bool) -> None:
        """Wrap one callable.

        Args:
            function: The function or ufunc of the real module.
            strict: Whether the arrays it makes lazy are strict.
        """
        self._function = function
        self._strict = strict
        # The function as a reader would name it, for a recording.
        self._source = f"np.{getattr(function, '__name__', 'function')}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the NumPy function with lazy arrays in place of `ndarray`s.

        Args:
            *args: Positional arguments as written.
            **kwargs: Keyword arguments as written.

        Returns:
            The result, with any `ndarray` in it made lazy.
        """
        strict = self._strict
        # The commonest call: lazy arrays and plain values by position, plain values by keyword. Nothing
        # has to be made lazy, so the arguments go to NumPy as they came.
        owned = True
        for arg in args:
            kind = type(arg)
            if kind is LazyArray:
                owned = False
            elif kind not in NO_ARRAY:
                break
        else:
            for arg in kwargs.values():
                if type(arg) not in NO_ARRAY:
                    break
            else:
                result = self._function(*args, **kwargs)
                if type(result) is LazyArray:
                    return result
                return _made_lazy(result, self._source, strict, owned)
        # Whether no array is among the arguments (`_holds_arrays`), found in the pass that makes the
        # `ndarray`s lazy (`_lifted`). The commonest arguments are told by their type, without a call.
        owned = True
        lifted = []
        for arg in args:
            kind = type(arg)
            if kind is LazyArray:
                owned = False
            elif kind is np.ndarray:
                owned = False
                arg = wrap(arg, strict=strict)
            elif kind not in NO_ARRAY:
                owned = owned and not _holds_arrays(arg)
                arg = _lifted(arg, strict)
            lifted.append(arg)
        for key, arg in kwargs.items():
            kind = type(arg)
            if kind not in NO_ARRAY:
                owned = owned and not _holds_arrays(arg)
                lifted_one = _lifted(arg, strict)
                if lifted_one is not arg:
                    kwargs = {**kwargs, key: lifted_one}  # A dictionary of the same order; `kwargs.items()` goes on over the old one.
        result = self._function(*lifted, **kwargs)
        if type(result) is LazyArray:
            return result  # Deferred: the commonest answer, and already what `_made_lazy` would return.
        return _made_lazy(result, self._source, strict, owned)

    def __repr__(self) -> str:
        """Say which function this wraps."""
        return f"<lazy {self._function!r}>"

    def __getattr__(self, name: str) -> Any:
        """Answer an attribute of the function: a ufunc's method, wrapped, or anything else as it is.

        Args:
            name: The attribute asked for.

        Returns:
            `np.add.reduce` as another wrapped callable; `np.add.nin` as the number it is.
        """
        attribute = getattr(self._function, name)
        return LazyCallable(attribute, self._strict) if callable(attribute) else attribute


class LazyRandom:
    """Stands where one of NumPy's random generators stands, and makes the arrays it draws lazy.

    `np.random.default_rng()` returns an object of NumPy's, whose methods the
    stand-in for the module never sees. Without this, `rng.standard_normal(n)`
    is a plain `ndarray`, and a program that takes its data and its weights
    from a generator computes most of its arithmetic outside any graph.
    """

    def __init__(self, generator: Any, strict: bool) -> None:
        """Stand in for one generator.

        Args:
            generator: A `numpy.random.Generator` or `numpy.random.RandomState`.
            strict: Whether the arrays it draws are strict.
        """
        self._generator = generator
        self._strict = strict

    def __repr__(self) -> str:
        """Say which generator this stands in for."""
        return f"<lazy stand-in for {self._generator!r}>"

    def __getattr__(self, name: str) -> Any:
        """Answer an attribute of the generator.

        Args:
            name: The attribute asked for.

        Returns:
            A method wrapped so that the arrays it returns are lazy;
            anything else (`bit_generator`) as it is.
        """
        attribute = getattr(self._generator, name)
        return LazyCallable(attribute, self._strict) if callable(attribute) else attribute


class LazyNumPy:
    """Stands where a NumPy module stands, and answers every name the module has."""

    # Slots, so that the C core can read them where it stands in for `__getattr__`.
    __slots__ = ("_module", "_strict", "_answers")

    def __init__(self, module: ModuleType, strict: bool) -> None:
        """Stand in for one module.

        Args:
            module: `numpy`, or one of its submodules such as `numpy.linalg`.
            strict: Whether the arrays it makes refuse to give their values without `gather()`.
        """
        self._module = module
        self._strict = strict
        # What `__getattr__` answered for each name, with the module's attribute it answered for.
        self._answers: dict[str, tuple[Any, Any]] = {}

    def __repr__(self) -> str:
        """Say which module this stands in for."""
        return f"<lazy stand-in for {self._module.__name__}>"

    def __dir__(self) -> list[str]:
        """List the module's names, so completion in a prompt works."""
        return dir(self._module)

    def __getattr__(self, name: str) -> Any:
        """Look a name up in the module.

        Args:
            name: The attribute asked for.

        Returns:
            Another stand-in for a submodule (`np.linalg`); the object itself
            for a type or a constant (`np.float32`, `np.pi`, `np.newaxis`,
            `np.errstate`); a wrapped function for a function.

        Raises:
            AttributeError: For a name the module lacks.
        """
        attribute = getattr(self._module, name)
        answered = self._answers.get(name)
        if answered is not None and answered[0] is attribute:
            return answered[1]  # The module still has what it had: the same answer, not another wrapper.
        answer = self._answer(attribute, name)
        self._answers[name] = (attribute, answer)
        return answer

    def _answer(self, attribute: Any, name: str) -> Any:
        """Work out what stands in for one attribute of the module.

        Args:
            attribute: What the module has under the name.
            name: The name.

        Returns:
            What `__getattr__` returns for it.
        """
        if isinstance(attribute, ModuleType):
            return LazyNumPy(attribute, self._strict)
        if isinstance(attribute, type) or not callable(attribute):
            return attribute
        if self._module is np and name in CONVERTERS:
            return self._converter(attribute, name)
        return LazyCallable(attribute, self._strict)

    def _converter(self, function: Callable[..., Any], name: str) -> Callable[..., Any]:
        """Wrap `np.array`, `np.asarray` and their like.

        Args:
            function: The function of the real module.
            name: Its name.

        Returns:
            A function that returns a lazy array unchanged for `np.asarray(x)`,
            makes the conversion at once on a lazy array's values otherwise
            (`np.array(x)` copies, as in NumPy), and wraps the result for
            anything that is not lazy.
        """
        def convert(*args, **kwargs):
            """Convert to a lazy array."""
            if name in PASS_THROUGH and len(args) == 1 and not kwargs and isinstance(args[0], LazyArray):
                return args[0]
            if args and isinstance(args[0], LazyArray):
                return _at_once(function, f"np.{name}", args, kwargs)
            # `np.array` copies unless told not to; the others return an array they are given as it is.
            owned = not _holds_arrays((args, kwargs)) or (name == "array" and kwargs.get("copy", True) is True)
            return _made_lazy(function(*args, **kwargs), f"np.{name}", self._strict, owned)
        return convert


# The roads the C core can stand on: see `array.use_core`.
LazyCallable.__call__.road = ("standin_call", _made_lazy, NO_ARRAY)
LazyNumPy.__getattr__.road = ("standin_name",)

# Every class a road can stand on exists now: take the C core if it is built.
array.use_core(array._built_core)
