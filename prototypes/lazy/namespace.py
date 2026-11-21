"""The stand-in for the `numpy` module: the same names, and every array it makes is lazy.

    np = get_lazy()             # where `import numpy as np` stood
    x = np.zeros(1000)          # a lazy array
    y = np.exp(x) + 1           # deferred

A lazy array already makes `np.exp(x)` lazy when `np` is the real module,
because NumPy passes the call to it. What the real module cannot do is make
an array lazy when no lazy array is among the arguments: `np.zeros(3)`,
`np.arange(10)`, `np.array(data)`. That is all the stand-in adds.

It can also be called, to wrap one array and keep `import numpy as np`:

    lazy = get_lazy()
    x = lazy(existing_ndarray)
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import numpy as np

from .array import LazyArray, _at_once, wrap

# Functions that turn their first argument into an array and have no protocol
# through which a lazy array could answer: NumPy would ask the lazy array for
# its values. The stand-in answers them itself.
CONVERTERS = frozenset({"array", "asarray", "asanyarray", "ascontiguousarray", "asfortranarray"})
# The converters that return their argument unchanged when it is an array already.
PASS_THROUGH = frozenset({"asarray", "asanyarray"})


def get_lazy() -> "LazyNumPy":
    """Return the stand-in for the `numpy` module. NumPy itself is not changed."""
    return LazyNumPy(np)


def _lifted(arg: Any) -> Any:
    """Make the `ndarray`s of an argument lazy.

    Args:
        arg: One argument of a call made through the stand-in.

    Returns:
        A lazy array for an exact `ndarray`; a list or tuple with its
        `ndarray`s made lazy; anything else unchanged. A subclass of `ndarray`
        is left alone: what NumPy does with it is NumPy's business.
    """
    if type(arg) is np.ndarray:
        return wrap(arg)
    if isinstance(arg, (list, tuple)) and any(type(item) is np.ndarray for item in arg):
        return type(arg)(_lifted(item) for item in arg)
    return arg


def _made_lazy(result: Any) -> Any:
    """Make the arrays a function returned lazy.

    Args:
        result: What the NumPy function returned.

    Returns:
        A lazy array for an exact `ndarray`; a tuple or list with its
        `ndarray`s made lazy; anything else unchanged, a NumPy scalar
        included: `np.sqrt(2.0)` stays an `np.float64`.
    """
    if type(result) is np.ndarray:
        return wrap(result)
    if type(result) in (list, tuple) and any(type(item) is np.ndarray for item in result):
        return type(result)(_made_lazy(item) for item in result)
    return result


class LazyCallable:
    """A NumPy function or ufunc whose `ndarray` arguments and results are made lazy.

    Calling it makes the `ndarray` arguments lazy and calls the real one.
    NumPy then passes the call to the lazy arrays, which defer it or make it
    at once; with no array among the arguments (`np.zeros(3)`) NumPy just
    makes the array, which is made lazy. A ufunc keeps its methods:
    `np.add.reduce` is wrapped the same way.
    """

    def __init__(self, function: Callable[..., Any]) -> None:
        """Wrap one callable.

        Args:
            function: The function or ufunc of the real module.
        """
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the NumPy function with lazy arrays in place of `ndarray`s.

        Args:
            *args: Positional arguments as written.
            **kwargs: Keyword arguments as written.

        Returns:
            The result, with any `ndarray` in it made lazy.
        """
        result = self._function(*(_lifted(arg) for arg in args), **{key: _lifted(arg) for key, arg in kwargs.items()})
        return _made_lazy(result)

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
        return LazyCallable(attribute) if callable(attribute) else attribute


class LazyNumPy:
    """Stands where a NumPy module stands, and answers every name the module has."""

    def __init__(self, module: ModuleType) -> None:
        """Stand in for one module.

        Args:
            module: `numpy`, or one of its submodules such as `numpy.linalg`.
        """
        self._module = module

    def __call__(self, array: Any) -> LazyArray:
        """Wrap one array.

        Args:
            array: An `ndarray`, held by reference, or anything `np.asarray` accepts.

        Returns:
            A lazy array with those values. A lazy array is returned as it is.
        """
        return wrap(array)

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
        if isinstance(attribute, ModuleType):
            return LazyNumPy(attribute)
        if isinstance(attribute, type) or not callable(attribute):
            return attribute
        if self._module is np and name in CONVERTERS:
            return self._converter(attribute, name)
        return LazyCallable(attribute)

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
            return _made_lazy(function(*args, **kwargs))
        return convert
