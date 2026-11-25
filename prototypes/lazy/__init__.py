"""A prototype of the lazy NumPy layer designed in docs/lazy-design.md.

    from prototypes.lazy import get_lazy
    np = get_lazy()                 # where `import numpy as np` stood
    x = np.arange(6.0) * 2 + 1      # written down, not computed
    x.gather()                      # a numpy.ndarray

It exists to be read and argued with. It defers a narrow set of NumPy
operations into a graph, makes every other NumPy call at once, and computes
the graph one node at a time with the NumPy call each node stands for, so
left alone it makes nothing faster. `fuse.enable()` has expressions the
graph holds computed by one kernel call each (docs/lazy-fusion-design.md).
`gather()` is the way to ask for values; anything else
that needs them (`float(x)`, `if x > 0:`, `np.asarray(x)`) gets them computed
on the spot and logged, and raises `NotGathered` for an array made by
`get_lazy(strict=True)`.

The modules, in the order to read them:

    graph.py      the node, and the list of nodes not computed yet
    rules.py      which calls are deferred, and their shape and element type
    evaluate.py   computing nodes
    array.py      LazyArray: what the user holds
    _core.c       the quick road for writing a node, in C; optional, built by build_core.py
    namespace.py  the stand-in for the numpy module
    explain.py    the text of a pending graph
    record.py     the graph as an object, and the mode that records a block's graphs
    report.py     what was computed without being asked, ranked
    view.py       the page that draws recorded graphs
    fuse.py       fusion: expressions read off the graph, and the kernels that compute one in a single pass
    _fused.c      the call into a fused kernel, in C; optional, built by build_fused.py from kernels/
"""
from .array import LazyArray, gather
from .graph import NotGathered
from .namespace import get_lazy
from .record import Graph, Recording, record_graphs
from .report import lazy_report

__all__ = ["get_lazy", "gather", "record_graphs", "lazy_report", "LazyArray", "NotGathered", "Graph", "Recording"]
