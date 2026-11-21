# Recording the lazy layer's graphs

This is a proposal for review, and a part of [lazy-design.md](lazy-design.md): the lazy layer writes NumPy operations down as a graph, and this document is about getting that graph out as an object. Drawing it is the other half, in [lazy-graph-page-design.md](lazy-graph-page-design.md).

A prototype is in [`prototypes/lazy/record.py`](../prototypes/lazy/record.py), with tests in [`tests/test_lazy_graphs.py`](../tests/test_lazy_graphs.py). Nothing here is in the package.

## Summary

- **`x.graph()` returns what is pending under one array as plain data,** and `graph.to_json()` writes it out. Nothing is computed.
- **`with hotkernels.record_graphs() as recording:` keeps every graph a block of code produced,** the computed ones included. It is a mode because a node forgets its operands once it is computed, which is what frees memory; outside the block nothing is recorded.
- **One graph per computation.** Each time the layer computes, the nodes it computed together are one graph, with what made it compute and on which line. What is still pending when the block ends is a last graph.
- **A graph is plain data.** It holds no array and no live node, so keeping one keeps no memory alive.
- **Each graph carries its operations as a tree of groups:** the operations one line wrote, the calls of a function, and the steps of a loop. The page folds by this tree; a reader of the JSON can use it too.
- **Graphs that are alike carry the same signature,** so a loop that gathers at every step shows as one graph, fifty times.
- **A recording also keeps what was computed without being asked,** by line: the calls NumPy made at once, the writes, the values needed. `recording.report()` ranks them.
- **A page carries as much of the program's code as it is told to:** whole files, only the lines that did something, or none.
- **Recording changes no result.** The prototype's tests hold a recorded program to the bits of the unrecorded one.

## What the user writes

```python
import hotkernels
np = hotkernels.get_lazy()

with hotkernels.record_graphs() as recording:
    weights, losses = fit(np, features, targets)
    hotkernels.gather(weights, *losses)

recording.graphs        # a list of graphs, in the order they were computed
recording.to_json()     # all of them as text
recording.show()        # the page, in the browser
```

Given a path, the block writes the page itself when it ends, which suits a script run from a terminal:

```python
with hotkernels.record_graphs("graphs.html"):
    main()
```

Without a recording, one array can still be asked what is pending under it:

```python
loss.graph()            # a graph whose trigger is "pending"
loss.graph().show()
```

| Name | What it is |
|---|---|
| `hotkernels.record_graphs(path=None, source="files")` | A `with` block. While it runs, the graphs it produces are kept in `recording.graphs`. `source` says how much code its pages carry |
| `recording.graphs` | The graphs, one per computation, in order; and a last one for what is still pending when the block ends |
| `recording.dropped` | How many operations the block wrote and then dropped without their ever being computed |
| `recording.report()` | What the layer computed without being asked during the block, ranked: calls made at once, writes, values needed, each with its lines. Printing it gives the table of `hotkernels.lazy_report()` |
| `recording.to_dict()`, `to_json()` | Everything as built-in types, and as JSON text, the counts of the report included as `events` |
| `recording.show()`, `save(path)`, `page()` | The page of [lazy-graph-page-design.md](lazy-graph-page-design.md): opened in the browser, written to a file, or returned as HTML. Each takes `source=` to carry more or less code than the recording was made with |
| `x.graph()` | The pending graph of one array. It has the same methods |

The names are provisional. One recording runs at a time; entering a second raises.

## What one graph is

**The nodes one computation computed.** The layer computes when something makes it, and each such moment is one graph:

| Trigger | When | Example of `call` |
|---|---|---|
| `gather` | `x.gather()` or `hotkernels.gather(...)` | `gather` |
| `at once` | A call outside the deferred set needed its operands | `np.argmax` |
| `write` | A write into a lazy array had to be made at once, in memory known outside the layer, and computed everything pending first. A deferred write triggers nothing: it is a node | `x[index] = v`, `x += y` |
| `needed` | Something needed values and `gather()` was not called | `float()`, `Conversion to an ndarray`, `np.add` |
| `limit` | Too many nodes were pending | `the limit of pending nodes` |
| `pending` | Nothing did: the nodes were not computed. The last graph of a recording, and every `x.graph()` | `still pending when the block ended` |

Each graph also has the file and line of the trigger and the text of that line.

This unit is chosen for two reasons.

- **It is what an evaluator is handed.** Whatever computes a graph differently later, by fusing or otherwise, receives exactly these nodes at once and no others. A graph of a recording is the input such an evaluator would have had.
- **Every trigger is a cut.** A program's operations form one large graph only on paper; in fact the graph is cut wherever a value is needed. Seeing the cuts, with the line that made each, is what tells a user why their graph is in forty pieces. The `needed` ones matter most: nobody wrote `gather()` there, so the code does not show them ([lazy-design.md](lazy-design.md#what-is-computed-without-gather)).

The graphs are disjoint, and they say how they connect: a node one graph computed and a later graph reads appears in the later one as an input of kind `earlier`, with the index of the graph it came from.

The other reading of "the graphs a block produced" is the connected pieces of everything the block wrote, whenever each part was computed. It shows the program's structure and hides where it was cut. It is under [Open questions](#open-questions).

### An example

The prototype's three example programs, run in one recorded block by `uv run python -m prototypes.lazy.programs.pages`:

| Program | Graphs | What they are |
|---|---|---|
| The regression that never needs a value, 50 steps | 1 | 954 operations, computed by the one `gather` at the end |
| The transformer block | 1 | 55 operations |
| A regression that gathers its loss every fifth step and ends with `np.argmax` ([`programs/monitor.py`](../prototypes/lazy/programs/monitor.py)) | 13 | The first loss with the standardising of the features, 8 operations. The second, 32. Eight more of 32 that are alike. One of 25 computed because `np.argmax` was made at once. One of 4 for the final weights. And 2 operations still pending when the block ended |

15 graphs and 1,336 operations in all. A further 78 operations were written and dropped: the losses of the steps that were not reported.

## How much code a page carries

The page opens on the program's code, so it has to carry it. How much is a choice, since a page may be kept, shown or sent on:

| `source=` | The page carries | For |
|---|---|---|
| `"files"`, the default | Every file that wrote an operation or made the layer compute, in full | Reading one's own program with its graphs beside it |
| `"lines"` | Only the lines that wrote an operation or made the layer compute | A page of a program that should not travel whole |
| `"none"` | No code: file names and line numbers only. A group that would be called by its line is called by its place | A page for a demonstration, or one passed on |

A file is named by its name alone at every level, never by its path. `x.graph()` on its own has no recording behind it and carries the lines of its nodes.

## What a node is

| Field | Meaning |
|---|---|
| `id` | The node's number. Negative for an `ndarray` that is not a node of the layer |
| `kind` | `operation`; `input` for an array from outside this graph that was never an operation of the recording; `earlier` for an operation an earlier graph computed |
| `name` | The operation's name (`add`, `matmul`, `getitem`); for an input, what made it |
| `category` | For an operation, its kind in the layer's rules: `elementwise`, `matmul`, `reduction`, `view`, `indexing` or `write`. A `write` is a write into a lazy array that was deferred: its first operand is what the array held For an input, `array`, or `at once` when it is the result of a call made at once |
| `dtype`, `shape`, `nbytes` | The element type, shape and size of its result |
| `where`, `code` | File name and line of the user's code that wrote it, and the text of that line |
| `operands` | The ids of the nodes it reads |
| `arguments` | Every argument in the order written: an id for an operand, text for anything else (`2.0`, `axis=0`, `1:3, None`) |
| `state` | `pending`, `computed` or `failed` |
| `graph` | For `earlier`, the index of the graph that computed it |

Three things about it:

- **`category` is what a fusible group is read from.** Connected `elementwise` operations ending in a `reduction` are the first thing a fusing evaluator would look for ([lazy-design.md](lazy-design.md#what-the-graph-is-for)), and the page colours by it.
- **Arguments are kept as written,** as the node keeps them ([lazy-design.md](lazy-design.md#with-hotkernelsinstall)): the record of `x.sum(0)` says `0`, not `axis=0`.
- **`where` is a file's name and a line, never a path.** A page may be passed on, and a path says where on a machine the code lives.

### Inputs

An input says where it entered the lazy world, as well as the layer knows:

| How the array entered | `name` | `where` |
|---|---|---|
| A function of the stand-in made it or took it in | `np.zeros`, `np.asarray` | The line of that call |
| A plain `ndarray` met a lazy array in an operation | `ndarray` | The line of the first operation that read it |
| A call made at once returned it | The call: `np.argmax` | The line of that call |
| It was made before the recording began | `array`, or the operation's name | The line it remembers, if it was an operation |

How an `ndarray` was made before it entered is outside the layer and is not recorded.

### Outputs

A graph's `outputs` are the nodes the trigger asked for. Where nothing in particular was asked (a write, the limit, what is left pending), they are the operations nothing else in the graph reads.

## Groups

A graph of a loop is too large to read node by node: 954 operations for 50 steps of the regression. Each graph therefore carries its operations a second time, as a tree.

| Kind | What it holds | Label |
|---|---|---|
| `block` | The whole graph. The root | `the recorded block` |
| `call` | The operations one call of a function wrote | `softmax()` |
| `line` | The operations one run of one line wrote, when there are several | The line's text |
| `lines` | A stretch of a long call: members that follow each other | `lines 71 to 78` |
| `repeat` | Runs that are alike and follow each other: the steps of a loop | `× 50` |
| `step` | One of those runs, when it has more than one member | `step 3 of 50` |
| `rest` | The steps of a long repeat that are not listed singly | `steps 3 to 49` |

For the regression's graph the tree is:

```text
block  the recorded block                              954 operations
  call  fit()                                            954
    line  X = (X - X.mean(axis=0)) / X.std(axis=0)         4
    repeat  × 50   regression.py:32-36                   950
      step  step 1 of 50                                  19
      step  step 2 of 50                                  19
      rest  steps 3 to 49                                893
      step  step 50 of 50                                 19
```

and for the transformer block, the calls of its functions: `block()` holding `layer_norm()` twice, `attention()` holding `softmax()`, `gelu()`, and the lines between them.

How the tree is found:

- **Calls come from the stack.** When a node is made, the recording walks the stack from the node's line up to the `with` block and keeps one entry per function of the user's code. Frames of the layer and of NumPy are left out. A generator expression runs in a frame of its own; it is counted as a line of the function it is written in.
- **One call is told from the next by the frame object.** A loop that calls `step(w)` fifty times makes fifty frames of the same function at the same line. The recording keeps a reference to the frames of the last node made, so the next call cannot be handed the same object, and compares by identity.
- **A line is a run.** Consecutive operations written on the same line under the same call are one `line` group. A line that calls a function in its middle is two runs with the call between them, in the order things happened.
- **A repeat is found by comparing signatures.** Each operation has a signature of its name, type, shape and line, and each group one of its members' signatures. Where the same run of members comes three times or more in a row, it is a `repeat`. The longest run looked for is 64 members.
- **A long call is cut into stretches.** A function of a hundred lines that are not alike has a hundred members and no group large enough to fold. Above 16 members a call's members are put in `lines` groups of 8 that follow each other, and in groups of those when there are still too many.
- **A long repeat lists its ends.** Above 12 steps it lists the first two and the last, and holds the others in a `rest` group, itself split into groups of 50 when there are more, so that unfolding one never puts thousands of boxes on the page.

The tree is best effort, and it cannot lose anything: the nodes are all in `nodes` whatever the tree says. A wrong grouping costs a reader a click.

## How it works

- **The layer has one hook,** an observer that is told three things: a node was made; these nodes are about to be computed; this computation is over. The recording is the only observer, and outside a `with` block there is none: the layer then does no more than test for one when a node is made and note a number when a node is computed.
- **A record is written when the node is made,** not when it is computed, because by then the operands are gone. It holds the fields of the table above and the calls the node was made under.
- **No value is held.** A record keeps a weak reference to its node, to tell at the end whether pending work is still pending or was dropped, and nothing else that could keep an array alive. What a recording costs is per operation and does not depend on the size of an array.
- **The frames of the last node are held until the next one,** for the comparison above. A finished function's local variables can therefore outlive it by one operation while a block is recorded. They are let go when the block ends.
- **A node made before the block and computed inside it is written down when it is computed.** No call is known for it, only the line it remembers.
- **A fourth thing is told through the same hook:** that the layer computed without being asked, with the call and the line. The recording counts it by line and remembers the file, so that a page can show the file.
- **Everything is done under the layer's lock.**

## What was verified

On the prototype, Apple M1 Pro, CPython 3.12.7, NumPy 2.5.5. `uv run --group dev pytest tests/test_lazy_graphs.py` runs 16 tests:

- **One graph per computation:** a gather, a call made at once, a write and what is left pending give four graphs in order, each with its trigger, its line and the text of the line.
- **What was computed unasked:** a recording's own counts by call and line, its report, and the same as `events` in its data.
- **How much code:** a page with whole files has every line of the file that was recorded; with lines only, the three that did something; with none, no text of any line anywhere in the page.
- **A value needed without `gather()`:** a `float()` is a graph of the two operations it needed; an array handed to `np.asarray` is a graph of everything that was pending.
- **Nodes:** kind, category, type, shape, size, line, text, operands and arguments as written, for operations; what made it and where, for an input from the stand-in, a plain `ndarray` and the result of a call made at once; and `earlier` with the graph's index.
- **Groups:** a call, a line, a loop of four steps as a `repeat` of `step`s whose signatures are one; a loop of 30 steps as two steps, a `rest` and a step; a call of twenty lines as three stretches; a generator expression as a line of its function.
- **Alike:** a loop that gathers at each step gives one graph per step with one signature from the second on.
- **A loop that writes in place:** fifty steps with two in-place updates and an assignment each are one graph of 300 operations, 150 of them writes, grouped as one `repeat`, with no write made at once.
- **No result changes,** and after the block the recording holds no frame.
- **A failure** is in its graph with the state `failed`.
- **Dropped work is counted,** and a second recording inside the first raises.
- **`x.graph()`** gives the pending nodes without computing, outside a recording too, and for a computed array one input.

Not measured: what recording costs in time. No speed is claimed anywhere in this design.

## Considered and not proposed

| Option | Why not |
|---|---|
| Keep every node's operation and operands after it is computed, so that a graph can always be looked at | Operands are what keep intermediates alive. A record without values, kept only while asked for, gives the same picture |
| Find loops and functions by reading the source (`ast`) or by tracing (`sys.settrace`) | The source is not always there, and tracing slows every line of the block. The stack at the moment a node is made is already known, and repeats can be seen in what was recorded |
| Group by depth below an output alone | Simple, and wrong for a loop: step 50 is fifty levels from step 1, so depth folds by distance from the end and not by step |
| Give every graph of a recording its inputs' values, for replaying | It would keep every input alive, and inputs are the large arrays. Replaying a graph is a different tool |

## Open questions

- **What one recorded graph is.** Proposed: one per computation. The other reading is the connected pieces of everything the block wrote. Both can be had from the same records; the question is which the page opens on.
- **A whole script without editing it.** An environment variable that records from start to end and writes the page on exit would need no `with`. It can be added beside the block.
- **The constants.** Fold above 5 operations, a repeat from 3 runs, a run of at most 64 members, 12 steps listed, a call cut into stretches above 16 members. They were chosen by looking at three programs.
- **The names** `x.graph()` and `hotkernels.record_graphs()`.
- **Whether `explain()` should be the text form of this object.** Today it walks the live nodes on its own.

## Not examined

- **Two files of one name.** A file is known to a page by its name without its directory, so `a/model.py` and `b/model.py` would be shown as one. Not handled.
- **Threads.** A node made on a thread other than the block's is recorded with its innermost line only. Not tried.
- **Generators and coroutines** that make operations across several resumptions. Their frames are reused, and what the grouping does with that was not looked at.
- **The cost of finding repeats in a very long group.** The search is bounded per member, not tested beyond a thousand operations.
- **A program of real size.** The three programs are a few dozen lines each.
