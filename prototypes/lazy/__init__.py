"""A prototype of the lazy NumPy layer designed in docs/lazy-design.md.

    from prototypes.lazy import get_lazy
    np = get_lazy()                 # where `import numpy as np` stood
    x = np.arange(6.0) * 2 + 1      # written down, not computed
    x.gather()                      # a numpy.ndarray

It exists to be read and argued with. It defers a narrow set of NumPy
operations into a graph, makes every other NumPy call at once, and computes
the graph one node at a time with the NumPy call each node stands for, so it
makes nothing faster. Values leave only through `gather()`: anything else
that would need them raises `NotGathered`.

The modules, in the order to read them:

    graph.py      the node, and the list of nodes not computed yet
    rules.py      which calls are deferred, and their shape and element type
    evaluate.py   computing nodes
    array.py      LazyArray: what the user holds
    namespace.py  the stand-in for the numpy module
    explain.py    the text of a pending graph
"""
from .array import LazyArray, gather
from .graph import NotGathered
from .namespace import get_lazy

__all__ = ["get_lazy", "gather", "LazyArray", "NotGathered"]
