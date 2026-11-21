<p class="lede">The kernels in this chapter do no arithmetic to speak of. They find an element, pick elements out, or look elements up. In each case the obvious loop is slow for a reason from chapter 2, a branch that cannot be predicted or a chain that cannot be overlapped, and the kernel is a way of organising the same work so that the reason goes away.</p>

Every kernel here returns NumPy's result bit for bit.

## `argmax`: stop tracking the index

**The problem.** The naive loop keeps two things as it walks the array: the best value so far and where it was. Each element is compared with the best, and a branch decides whether to update both. The compiler leaves this loop scalar, one element at a time, at about 1.3 ns per element.

The textbook vector version carries an index vector alongside the maximum vector and updates both lane by lane. That works, but it does several operations per vector on a quantity, the index, that is only needed once, at the very end.

**The idea.** Split the question in two. First find *what* the maximum is, which is a plain reduction with no index at all. Then find *where* it is, by looking only in the neighbourhood that produced it.

- Walk the array in blocks of 64 elements (32 for float64). Reduce each block to a single number, its maximum. That is fifteen `fmax` operations on sixteen vectors and one final fold: pure, branch-free, independent work.
- Compare the block's maximum with the best so far. If it is *strictly* greater, remember this block's address using a conditional select. No branch.
- At the end, go back to the one winning block and scan its 64 elements for the first one equal to the maximum.

Strictly greater is what makes ties come out right: of two blocks with the same maximum the earlier one is kept, and within the block the scan takes the first match. That is NumPy's rule.

**NaN for free.** NumPy says the first NaN wins over everything. Checking each element for NaN would double the work. But the `fmax` instruction returns NaN if either input is NaN, so a block that contains a NaN reduces to NaN, and a comparison involving NaN comes out "unordered", which the same compare instruction reports as a separate flag. One compare per block therefore answers both "is this the new best block?" and "is there a NaN in here?". Only in the second case does the kernel branch, to a simple loop that finds the first NaN.

<figure class="widget">
<div data-widget="argmax"></div>
<figcaption>Block maxima. The first pass reduces each block to one number and remembers which block is best; it never looks at an index. Only the winning block is scanned again. Add a tie to see that the earlier block is kept, or a NaN to see the one data-dependent branch the kernel has.</figcaption>
</figure>

**Short arrays get their own code.** A loop has overhead that 16 elements cannot repay. For 8 to 64 float32 elements the kernel runs as a straight line with no loop: take the maximum of everything using a front half and a back half that overlap in the middle, compare every element with it, and turn the comparison results into a bit mask whose lowest set bit is the answer.

**What it achieves.**

| | float32 | float64 |
|---|---|---|
| Over naive C, all sizes | 23.4x [23.39 – 23.41] | 12.5x [12.53 – 12.54] |
| Over naive C at 4,096 elements | 45.9x | 22.9x |
| Over NumPy at 1,048,576 elements | 8.82x | 6.86x |
| Time per element in the first-level cache | 0.028 ns | about twice that |

In cache the float32 kernel runs within a few percent of the rate at which the core can load data at all. Beyond the caches it runs at the speed of a plain maximum with no index, which is to say the index costs nothing.

**What did not help:** software prefetch (5% slower), aligning the loop, a vector search for the first NaN (invisible to the judge, so not kept), and several cheaper-looking ways to fold a block down to one number.

::: note The cost of the rare case
On data without a NaN the kernel's speed does not depend on the data at all: ascending, descending and constant inputs take exactly as long as random ones. A NaN changes that. In the block loop it ends the search early. In the straight-line code for short arrays it sends the whole array to a scalar scan: 256 float32 elements take 12 ns without a NaN and 112 ns with one in the last position. That is still far faster than NumPy, and it is the price of making the common case branch-free.
:::

## Boolean masks without a branch

`x[mask]` is the largest gap the survey found that a compiler does not close. Three kernels share the technique: `compress` for float32 and float64, and `nonzero`, which emits the indices instead of the elements. Two siblings for 8- and 16-bit elements, `compress_u8` and `compress_i16`, were added later for x86-64, where a byte shuffle compacts eight elements ([kernel-findings](https://github.com/sofeikov/hotkernels/blob/main/docs/kernel-findings.md#xmask-on-8--and-16-bit-elements)); on ARM64 they are the compiled C.

**Step one: no branch.** As [chapter 1]([[ root ]]report/numpy/#the-mask-surprise) showed, most of the cost of the obvious loop is a mispredicted branch per element. The fix is to store every element at the current output position and advance the position by the mask value. An unselected element is simply overwritten by the next store. That loop is the *baseline* here: it is the naive C these kernels have to beat.

**Step two: eight at a time, with a table.** The vector version does the same thing for a block of eight elements per step, and the trick is in how it compacts them.

- The eight mask bytes are loaded as one 64-bit number, and one multiplication squeezes them into an 8-bit number between 0 and 255. That number *is* the mask, as an integer.
- It indexes a precomputed table with 256 rows. Each row is a shuffle pattern: "output slot 0 takes input 2, slot 1 takes input 5, slot 2 takes input 6, the rest do not matter".
- A table-lookup instruction applies the pattern, moving the selected elements to the front of the block. The whole block is stored at the write position.
- The write position advances by the number of selected elements, which a second multiplication has already computed from the same 64-bit number.

There is no branch anywhere in that, and no loop over the eight elements. The elements stored beyond the selected ones are rubbish, and the next block's store lands on top of them.

<figure class="widget">
<div data-widget="compress"></div>
<figcaption>Two ways to do <code>x[mask]</code> with no data-dependent branch. Toggle mask bits, then compare the scalar loop, which is the baseline these kernels are measured against, with the table lookup that handles eight elements in one step. Both store more than they keep.</figcaption>
</figure>

**Sparse masks need a different plan.** If one element in twenty is selected, compacting every block of eight is mostly wasted motion. For sparse stretches the kernels reduce 64 mask bytes to a single 64-bit word with one bit per element, and then pull out the set bits one after another with a count-leading-zeros instruction: a fixed six extractions per step, whether or not there are six bits set (surplus extractions copy rubbish that is overwritten). The kernel chooses between the two plans by measuring: after each stretch it looks at the fraction it just selected.

```mermaid How the mask kernels move between their two plans. The decision is made per stretch of the array, from the density the kernel has just seen, so a mask that is dense in one region and sparse in another gets both.
flowchart TD
  dense["<b>Dense plan</b><br/>8 mask bytes → table row → shuffle<br/>→ store the whole block"]
  sparse["<b>Sparse plan</b><br/>64 mask bytes → one 64-bit word<br/>→ extract 6 set bits with clz"]
  d1{"at most 1 in 8<br/>selected in the<br/>last stretch?"}
  d2{"more than 12 of<br/>these 64 selected?"}
  dense --> d1
  d1 -->|"yes"| sparse
  d1 -->|"no: longer dense stretch"| dense
  sparse --> d2
  d2 -->|"no"| sparse
  d2 -->|"yes"| dense
  class dense,sparse hot
```

**The float64 kernel has no table.** With only two doubles per vector, a block of eight is four pairs, and there is a neater trick. A pair `(x[p], x[p+1])` needs one 16-byte store at its output position. If `mask[p]` is true, store the pair as it is. If not, store the pair shifted by one, `(x[p+1], x[p+2])`. Either way it is one load and one store, and the *address of the load* is `x + p + 1 - mask[p]`. The compaction is done by arithmetic on an address, with no shuffle at all.

**This depends on scratch output.** Every one of these stores writes past the elements that are kept. The spec permits that inside the buffer's capacity, and the stores can never run past the capacity, because the write position never gets ahead of the read position. What it means for the caller is the subject of [a section in chapter 7]([[ root ]]report/delivery/#scratch-output-and-the-cost-of-allocation).

**What it achieves**, at 1,048,576 elements:

| Kernel | Half selected: over the branch-free C / over NumPy | One in twenty selected |
|---|---|---|
| `compress_f32` | 2.70x / 41.7x | 5.70x / 17.3x |
| `compress_f64` | 1.67x / 22.5x | 4.10x / 11.0x |
| `nonzero_bool` | 2.47x / 5.15x | 8.08x / 19.2x |

The NumPy column is so large because it combines both steps: removing the branch (worth 14x on its own at half density) and then the vector compaction. With half the elements selected, `compress_f64` is limited by how fast memory can be written from 16,384 elements up; there is nothing left to tune.

::: warning A weak spot the judge does not time
The switch between the two plans is itself a branch, and near 10% density it is mispredicted. Measured outside the judge, `nonzero_bool` costs 0.127 ns per element at 10% density, against 0.045 at 5% and 0.110 at 20%. The judge times 5% and 50%, so this dip is real but does not appear in the tables.
:::

**On a matrix,** `np.nonzero(m)` needs a row and a column for each True element, and NumPy returns them as two views of one array of (row, column) pairs. `nonzero_2d_bool`, on ARM64, writes those pairs. Its sparse plan is `nonzero_bool`'s, with each position turned into a row by a multiplication instead of a division; its dense plan works a row at a time, in chunks that never cross the end of a row, and rows narrower than 8 are taken several at once with a small table of the row and column of each byte. At a million elements in rows of 22 it is 14.0x faster than the branch-free C with one in twenty True and 3.7x with half, and 36x and 22x faster than NumPy.

## Masked assignment: write what you did not change

`x[mask] = v` looks like the mirror image of selection, and the naive loop has the same problem: `if (mask[i]) x[i] = v;` is a branch per element that a random mask makes unpredictable. Both NumPy and naive C cost 3.4 to 3.9 ns per element at half density.

**The idea.** Load sixteen elements. Build, from the mask, a vector that is all ones where the mask is true and all zeros elsewhere. Blend: take `v` where the vector is ones and the original element where it is zeros. Store all sixteen. No branch, and every step independent of the last.

**The catch is in the last sentence.** The kernel stores elements the mask did not select. It stores them back with exactly the bits they had, so the result is identical. But it *wrote* to them, and the source code said to leave them alone. This is precisely why a compiler does not produce this loop: it is not allowed to introduce a write to memory the program did not write. If another thread modified an unselected element between the load and the store, the kernel would undo that modification. NumPy's own loop would not.

The project decided that this is acceptable for an array the caller owns, which is the normal case, and made the binding hold the interpreter lock while this kernel runs, so no other Python thread can write the array in between.

**Sparse masks avoid it entirely.** When a block of sixteen has at most one or two selected elements in each half, the kernel loads nothing: it computes the addresses of the selected elements from the mask word and stores `v` there directly. Unselected elements are not touched. A dense mask pays for one density test per 64 blocks; a sparse mask never blends.

**What it achieves.** 7.19x over naive C overall with half selected, and 19.0x at a million elements, where it is 21.4x faster than NumPy.

::: note Reading the jump at 65,536 elements
The speedup over naive C leaps from about 3.5x to 19x between 16,384 and 65,536 elements. The kernel did not get faster there. Naive C got slower. From 4,096 elements up the judge repeats one input per size, and below about 16,384 elements the branch predictor *memorises* that input, so the branchy reference looks far better than it would on fresh data. The large-size figure is the honest one, and the small-size figures understate the kernel.
:::

## `isin`: membership as a table lookup

`np.isin(x, test)` with a handful of test values is slow in NumPy for reasons that are mostly fixed cost (it is a Python-level function that sorts and searches). The kernel handles the case of up to eight test values over int64 data. A sibling for int32 data, `isin_i32`, was added later for x86-64, where it narrows with a saturating pack in the same spirit ([kernel-findings](https://github.com/sofeikov/hotkernels/blob/main/docs/kernel-findings.md#npisin-on-int32)); on ARM64 it is the compiled C.

The straightforward vector method compares each element with all eight test values. At 64 bits there are two elements per vector, so that is eight compares per two elements. The compiler already emits exactly this, and it cannot be improved: there is no wider compare on this hardware.

**The idea: make the elements smaller.** If all eight test values lie in a narrow range, an element outside that range cannot be a member, and the elements inside it can be represented in 16 bits. So:

1. Subtract a base and narrow each element from 64 bits to 16 with *saturation*: anything out of range becomes 65,535, a value no test key has. Now there are eight elements per vector, not two.
2. Split each 16-bit key into four 4-bit nibbles. For each nibble position there is a 16-entry table whose entries are bytes with one bit per test value: bit $j$ is set if test value $j$ has that nibble in that position.
3. Look up all four nibbles and AND the four bytes together. Bit $j$ survives only if test value $j$ matches the key in all four nibble positions, which is to say equals it. The element is a member exactly when the result is non-zero.

Four table lookups replace eight compares, on vectors holding four times as many elements. It is an exact method, not a probabilistic filter: the tables encode the test set completely.

**The catch: it depends on the test values.** The narrow path applies when all eight values lie between 0 and 65,534, or all lie within 32,767 of the first one. Then the kernel is 4.5x to 5.6x faster than the compiled C. Otherwise it falls back to 64-bit compares and ties the C.

| Test values | Over naive C at 1M | Over NumPy at 1M |
|---|---|---|
| Close together | 5.34x | 75.6x |
| Arbitrary 64-bit values | 1.00x, not significant | 1.83x |

Because the second window is centred on the *first* test value, the same eight values can be fast in one order and slow in another: eight values spanning 60,000 ran at 0.17 ns per element with a middle value first and 0.62 with the lowest first. The judge's original input only ever took the fast path. A second input distribution was added specifically so that the slow path would show in the results.

## `searchsorted` and `interp`: search many at once

Binary search is the standard example of code a processor hates. Each probe depends on the comparison before it, so the probes of one search form a dependency chain, and each probe is a load from an unpredictable address, so on a large array each is a cache miss. Worse, the usual implementation branches on each comparison, and on random needles that branch is a coin flip.

**Step one: no branch.** A binary search over $n$ elements can be written with a `base` index and a step that halves:

```text
base = 0
for h = n/2, n/4, ..., 1:
    if a[base + h - 1] < key:  base += h
```

The `if` becomes arithmetic: the comparison produces a mask of all ones or all zeros, and `base` is advanced by `h` AND that mask. No branch, so no misprediction.

**Step two: the step size does not depend on the data.** In that formulation the sequence of step sizes $h$ depends only on $n$. Every search over the same array takes exactly the same steps, whatever its needle. So many searches can be run *in step*: at each level, load one probe for each needle, compare them all, advance all the bases.

That converts the problem. One search is a chain of $\log_2 n$ dependent loads and cannot be sped up. But sixty-four searches are sixty-four *independent* chains, and the processor can have all sixty-four loads in flight at once. On a large array, where each load is a cache miss taking a hundred cycles or more, the misses overlap instead of queueing.

<figure class="widget">
<div data-widget="search"></div>
<figcaption>A branch-free binary search for several needles at once. At every step all needles probe at the same distance from their own base, so their memory loads are independent of each other and can be outstanding together. The timelines underneath compare that with searching one needle at a time.</figcaption>
</figure>

The kernel picks its batch by the size of the array: sixteen needles at a time below 64 elements, sixty-four up to a million, and blocks of 128 beyond. For the last few levels of every search, where the remaining candidates sit in eight adjacent elements, it stops probing and simply counts how many of the eight are below the key: one step in the chain instead of four.

**`interp`** runs the same search and then the interpolation arithmetic for two queries per vector, with a real division and a fused multiply-add, which is what NumPy's own build uses. Bit-identical interpolation has many special cases (a query exactly on a sample point, a NaN, infinities). The kernel does not handle them in the fast path: it computes the general formula for everyone, and any group whose result contains a NaN is redone by a line-by-line copy of the reference.

**What it achieves.**

| Kernel | Over naive C, all sizes | at 1,048,576 | at 16,777,216 | Over NumPy at 1M |
|---|---|---|---|---|
| `searchsorted_f64` | 9.08x [8.98 – 9.14] | 8.83x | 15.5x | 6.40x |
| `interp_f64` | 5.00x [4.93 – 5.03] | 6.40x | 10.2x | 16.8x |

Timed with 1,000 needles per call; the sizes are lengths of the sorted array. Each figure is the mean of ten runs on a machine that was not fully quiet, with the lowest and highest of the ten in brackets.

These are the only kernels whose advantage *grows* at the largest sizes, because overlapping cache misses matters most when every probe is a miss.

**What limits it.** In cache, the instruction count is not the limit; *which elements get probed* is. A variant in which every search probed the same elements ran 1.25x to 1.33x faster with identical instructions. Beyond the cache, the limit is the number of cache and address-translation misses, which the kernel overlaps but cannot reduce.

::: note Two lengths
These kernels were first written with one length for the sorted array and the needles. That was enough to develop and judge them and not enough to serve a real call, where the two lengths differ, so for a while they were complete and not installed. They now take the two lengths separately. The length of the sorted array chooses the path; a call with fewer needles than a path handles at once takes a narrower one, down to one needle at a time. `interp_f64` is only checked by the judge up to 65,537 sample points, because NumPy's `interp` is too slow to act as an oracle beyond that; its large-array path was compared with the reference separately.
:::

::: note The right side
`np.searchsorted(a, v, side='right')`, which `np.digitize` asks for by default, is a second kernel, `searchsorted_right_f64`. It is the same search with the comparison turned round, `a[i] <= key` in place of `a[i] < key`, so the two differ only for a needle that equals an element, and for a NaN needle, whose answer is the length of the array. The judge checks it like any other kernel, with needles drawn from the array at every size, so that some equal its elements.
:::

::: note Any along the other axis
`any_cols_bool` serves `m.any(axis=0)`, the call beside `m.any(axis=1)`: whether each column of a boolean matrix holds a True. It keeps the results in vector registers for the whole call and makes them 0 or 1 once at the end, where a compiled loop or's each row into memory. It was written for x86-64 and ships as its C on ARM64 for now. On an Intel Core i5-14600KF a mask of 30 rows by 31 columns takes about 95 ns from Python against NumPy's 840 ns.
:::

::: note A float sum in NumPy's order
`sum_f32` serves `np.sum(x)` and `x.sum()` on float32 with NumPy's bits. NumPy adds in blocks of up to 128 elements, each with eight running sums, and combines the blocks as a tree; its loop waits on each addition. The kernel makes the same additions but steps up to eight blocks side by side, each in a register of its own, so it waits on loads instead. It has assembly for both architectures, written separately. On an Intel Core i5-14600KF it is 5.9x over that C at 4,096 elements, and `x.sum()` from Python is about 8x faster than NumPy at 4,096 elements and 2.6x at a million. `sum_f64` does the same for float64, a block's eight running sums in two registers: on the same machine 3.8x over its C at 4,096 elements, and `x.sum()` from Python 5.7x faster than NumPy there and 1.6x at a million.
:::

::: note `nonzero` on a matrix
`np.nonzero(m)` on a boolean matrix needs a row and a column for each True element, and NumPy returns them as two views of one array of (row, column) pairs. `nonzero_2d_bool` writes those pairs. It carries the row and column of each block of 32 elements from block to block, so no position is divided by the width; a sparse block takes its few positions out of a mask word, and a dense one turns eight mask bytes at a time into pairs through tables, with a table of the row and column of each byte for rows of up to 64. It has assembly for both architectures, written separately. On an Intel Core i5-14600KF it is 3.95x over that C across the four widths the judge times, 7.4x to 8.5x with one in twenty True and about 2x with half, and from Python `np.nonzero(m)` on 40 rows of 22 is about 5x faster than NumPy, and 24x at a million elements.
:::

::: note The search on int64
`searchsorted_i64` and `searchsorted_right_i64` are the same two searches for a sorted int64 array and int64 needles, with no NaN to order. They were written for x86-64 first, where a pass takes two steps of each search and the first steps need no gather. On ARM64 sixteen searches run in step in integer registers, some compared with the flags and some in vector registers because this core runs few flag instructions per cycle, and arrays beyond the first-level cache are swept one step at a time over blocks of needles, as NumPy's own int64 search does. NumPy's int64 search is about three times faster than its float64 search, so the gain with many needles is smaller: 4x to 5x over NumPy up to 4,096 elements on an Intel Core i5-14600KF, and 1.2x beyond the caches; on an Apple M1 Pro 2.7x at 4,096 elements and 1.7x at a million. The call they were added for passes one `np.int64` at a time, and there the gain is NumPy's call path: about 2x from Python at any length.
:::
