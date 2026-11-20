# searchsorted_i64 attempts (arm64)

**Spec.** The attempts below were judged against a spec of this run's own
(branch `claude/gaps160-02-searchsorted-i64`), which had the same signature,
naive C, value classes and timed sets (64 needles and `many`, 1,000) as the
spec of PR 245 that this kernel is built on, and differed only in the untimed
boundary sets: it had 7, 8, 9, 31, 32 and 33 needles where that spec has 2, 3,
4 and 5. The final kernel was checked and recorded against that spec (63,444
cases).

All judge commands in this log were run with `--unlocked` (the spec is new on
this branch and not on `main` yet), on an Apple M1 Pro, while other agents were
timing other kernels on the same machine: the one-minute load average is given
with each figure, and figures are indicative.

The float64 kernel ([its log](../../searchsorted_f64/arm64/attempts.md)) is the
idea: many searches kept in step, without a branch. Its instructions were
chosen around `fcmp`, which runs at one per cycle on this core; an integer
`cmp` + `csel` has no such limit, so the forms here are chosen afresh.

## 1. Baseline: branchless lower bound, one needle at a time

- **Idea:** `half = len / 2; if (a[base + half - 1] < key) base += half;
  len -= half`, with `cmp` + `csel`, then one last probe. A call with an empty
  array writes zeros.
- **check:** pass, 68,654 cases
- **compare:** not applicable (first version)
- **Kept:** yes
- **Notes:** `judge run` at a load average of about 19: 1.47x over the naive C
  overall, 1.40x over NumPy; behind NumPy from 262,144 elements with 64
  needles and from 65,536 with 1,000. One needle at a time leaves the core
  waiting on each load in turn: NumPy's own loop overlaps consecutive
  needles out of order and wins where the probes miss the first-level cache.

## 2. Eight needles in step, keys and bases in integer registers

- **Idea:** the float64 kernel's lower bound with power-of-two steps. Let p
  be the largest power of two <= n; `base = (a[p - 1] < key) ? n - p : 0`,
  then for h = p/2 .. 1 `if (a[base + h - 1] < key) base += h`, then one last
  probe. The step size depends only on n, so eight searches run in step. One
  probe is `ldr`, `add`, `cmp`, `csel`; h is a register, so one loop serves
  every level. Keys and bases of eight needles fill x0-x15 and x19-x20
  (saved on the stack), and the call's state lives on the stack between
  groups. Leftover needles are covered by one more group ending at the last
  needle; fewer than eight needles go one at a time through the same search.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.774x [1.770 – 1.778] (load average 8.3)
- **Kept:** yes
- **Notes:** 1.3x at 16 elements rising to 2.3x at 16,777,216. Against NumPy
  it is still behind from 4,194,304 elements: NumPy 2.5's int64 search is not
  the naive loop. Disassembled, it advances every needle by one level per
  sweep over all the needles, the base of each kept in the output, with a
  `csel`: nothing waits on a load except the store of that needle, so every
  probe of a sweep can be in flight at once. Eight searches in step cannot
  match that once the probes leave the cache.

## 3. Twelve needles in step

- **Idea:** more searches in flight: keys and bases of twelve needles in
  x0-x15 and x19-x28.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.080x [1.068 – 1.093] (load average 9.8)
- **Kept:** yes
- **Notes:** 1.03x to 1.09x with 1,000 needles in the cache, 1.24x to 1.33x
  from 65,536 elements. With 64 needles it is 3% to 8% slower up to 16,384:
  64 is five groups of twelve and a sixth that redoes eight, 72 searches for
  64 answers.

## 4. From 65,536 elements: NumPy's order, one level per sweep over a block

- **Idea:** NumPy's own method, done in fewer instructions. From 65,536
  elements, take the needles in blocks of up to 256, keep their bases in
  `out`, and sweep the block once per level: load base and key, probe,
  `cmp`, `csel`, store the base, four needles per iteration (`ldp`/`stp`)
  and singly for the rest of a block. No probe waits on another, so the
  out-of-order window holds the probes of dozens of needles. The last probe
  (`if (a[base] < key) base += 1`) is the level h = 1 run twice. Any number
  of needles from one up takes this path.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.151x [1.143 – 1.159] (load average 6.7)
- **Kept:** yes
- **Notes:** 1.14x at 65,536 rising to 1.69x (64 needles) and 2.02x (1,000)
  at 16,777,216; unchanged below 65,536. In a local harness the sweep is as
  fast as the twelve-needle groups at 4,096 elements and slower below, which
  is why it starts at 65,536.

## 5. The wide path from 16,384 elements

- **Idea:** start the sweeps of attempt 4 at 16,384 instead of 65,536.
- **check:** pass, 68,654 cases
- **compare:** significantly slower, 0.991x [0.989 – 0.993] (load average 7.2)
- **Kept:** no
- **Notes:** 0.98x at 16,384 with 64 needles and 0.83x with 1,000.

## 6. Sixteen needles, half of them compared in a vector register

- **Idea:** a micro-benchmark on this core runs only about three `cmp` (or
  `csel`) per cycle, so the two flag instructions of a probe, not the loads,
  bound the groups. The float64 kernel's form uses no flags: `ldr d`,
  `cmgt d`, `fmov` to an integer register, then `and` with h and `add` to
  the base (five instructions; h is a register, so the masked add cannot be
  one shifted `sub`). That form also needs no integer register for the key,
  which stays in a vector register. Alternate the two forms: eight needles
  with `cmp`/`csel` (two registers each) and eight with the vector compare
  (one register each), sixteen needles in the same registers as twelve.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.128x [1.127 – 1.129] (load average 4.0)
- **Kept:** yes
- **Notes:** 1.40x at 16 falling to 1.17x at 16,384 with 64 needles (which
  are now four whole groups), 1.26x to 1.05x with 1,000; the wide path from
  65,536 is unchanged. Locally, other mixes: all sixteen in the vector form
  was no faster than attempt 3, one vector needle in three (twelve needles)
  about the same as this, eighteen needles with six flag compares and twelve
  vector ones 5% slower than this.

## 7. The vector needles' mask and-ed with h in a vector register

- **Idea:** `and v27.8b, v27.8b, v28.8b` with h moved to v28 once per level,
  in place of the integer `and` after `fmov`: one instruction fewer on the
  integer ports per vector needle.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.004x [1.003 – 1.005] (load average 3.4)
- **Kept:** yes
- **Notes:** 1% to 2% below 4,096 elements, nothing elsewhere. A
  micro-benchmark of one probe repeated in a chain explains why so little:
  the flag form's chain (`ldr`, `cmp`, `csel`) is about 6.3 cycles and the
  vector form's (`ldr d`, `cmgt`, `fmov`, `and`, `add`) about 13.5, whether
  the mask is and-ed or shifted. A group lasts as long as its slowest
  chain, so the vector needles set the pace and the instruction count
  hardly matters.

## 8. From 16 elements, the last four probes replaced by a count of eight

- **Idea:** the float64 kernel's ending. Once a search's window is eight
  elements long, load a[base .. base + 7] (one 64-byte `ld1`), compare all
  eight with the key in four vector compares, add the masks and subtract the
  sum from the base. One chain of about 19 cycles in place of four probes,
  which cost about 25 cycles with the flags and 54 in the vector form. The
  keys of all sixteen needles are now loaded into both lanes of a vector
  register (`ld1r`). Below 16 elements the group ends as before. The kernel
  is now written by a generator kept outside the repository; its header
  describes the paths.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.017x [1.015 – 1.020] (load average 3.4)
- **Kept:** yes
- **Notes:** 1.08x at 16 elements falling to 1.02x at 4,096; nothing from
  16,384, where the wide path starts soon after and the chain matters less
  than the misses. Without the count, the same code with `ld1r` keys was
  about 5% slower than attempt 7 at 16 elements: each group loads the keys
  of its flag needles twice.

## 9. The wide path ends with the count over eight elements too

- **Idea:** in the wide path, sweep while h >= 8 and then make one sweep
  that counts over a[base .. base + 7] (attempt 8's ending), in place of four
  sweeps of probes. Locally, blocks of 128, 512 and 1,024 needles in place
  of 256 were within the noise of each other, so the block stays at 256.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.017x [1.013 – 1.021] (load average
  4.7); again 1.020x [1.018 – 1.022] (load average 5.1) after removing the
  two setup instructions the sweep count no longer needed
- **Kept:** yes
- **Notes:** 1.12x at 65,536 with 64 needles, 1.10x at 262,144, 1.06x at
  1,048,576, nothing beyond: there the sweeps wait on misses, not on
  instructions.

## 10. Ten flag needles and six vector needles

- **Idea:** with the count ending, a vector needle's chain is shorter than
  before, so the balance between the two forms may have moved. Locally, of
  the mixes of sixteen that fit the registers (two flag needles and one
  vector needle repeated, ten and six; four and twelve; eight and eight in
  pairs) and of fourteen and twelve, ten flag needles and six vector ones was
  the only one ahead, by 2% to 3%.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.015x [1.014 – 1.016] (load average 4.8)
- **Kept:** yes
- **Notes:** 1% to 3% below 16,384 elements. All 28 integer registers other
  than x16-x18 are now in use in the group path.

## 11. In the groups, count over four elements instead of eight

- **Idea:** the count of attempt 8 costs nine vector instructions per needle,
  and with sixteen needles a group's ending is limited by the four vector
  ports (about 36 cycles) more than by its chain. Stop one step later and
  count over a window of four: one 32-byte load, two compares, five vector
  instructions. Locally a window of sixteen (two 64-byte loads, eight
  compares) was 10% to 60% slower; the wide path keeps eight.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.010x [1.008 – 1.011] (load average 4.9)
- **Kept:** yes
- **Notes:** 2% at 16 elements, 1% at 4,096.

## 12. Every needle of a wide sweep compared in vector registers (rejected on local timing)

- **Idea:** take the flags out of the wide path too: four bases and four keys
  as two vector registers each, the probes loaded into lanes (`ldr d` and
  `ld1 {v.d}[1]`), two `cmgt`, two `and` with h, two vector adds, one `stp q`.
  About 19 instructions for four needles in place of 26, none of them
  reading or setting the flags.
- **check:** not run through the judge (local harness: correct)
- **compare:** not run through the judge; locally 0.95x to 1.01x from 65,536
  to 16,777,216
- **Kept:** no
- **Notes:** the sweeps from 65,536 elements are limited by the probes'
  misses, not by the flags.

## 13. The wide path from 16,384 elements again

- **Idea:** attempt 5 again, now that both paths end with a count.
- **check:** pass, 68,654 cases
- **compare:** significantly slower, 0.987x [0.985 – 0.989] (load average 4.4)
- **Kept:** no
- **Notes:** 0.87x at 16,384 with 64 needles, 0.83x with 1,000.

## 14. The wide path from 262,144 elements

- **Idea:** the other direction: groups at 65,536 as well.
- **check:** pass, 68,654 cases
- **compare:** significantly slower, 0.996x [0.994 – 0.998] (load average 5.9)
- **Kept:** no
- **Notes:** 0.98x at 65,536 with 64 needles, 0.95x with 1,000.

## 15. When n is a power of two, the first step is h = p/2 with its probe preloaded

- **Idea:** when n is a power of two, n - p is 0 and the first step gives
  every needle base 0, which is a step wasted. Make the step h = p/2 the
  first step instead: with base 0 its probe is always a[p/2 - 1], loaded
  once per call, so the first step is a compare with a register, and each
  search is one probe shorter. The change is five instructions at entry,
  which set the first step's pivot and size and halve the starting h; the
  paths themselves do not change.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.076x [1.070 – 1.082] (load average 5.1)
- **Kept:** yes
- **Notes:** 1.21x and 1.24x at 16 elements, 1.08x to 1.14x up to 16,384, 1%
  to 5% from 65,536. The judge times only powers of two, so this is a gain
  for the sizes timed; an array of any other length is searched as before.

## 16. For a power-of-two n of 32 or more, the first two steps as a count of three pivots (rejected on local timing)

- **Idea:** attempt 15 one step further. With base 0, the first two steps
  probe a[p/2 - 1] and then a[p/4 - 1] or a[3p/4 - 1]; the three are fixed
  per call, and how many of them are below the key, times p/4, is the base
  after both steps. Each needle: two vector compares against the pivots
  held in registers, an add, `addp`, `fmov`, `mneg`, and no load.
- **check:** not run through the judge (local harness: correct, after
  fixing a scalar `add d` that cleared the lane holding the third pivot's
  mask)
- **compare:** not run through the judge; locally 0.95x at 64 elements,
  0.97x at 1,024, level at 4,096 and beyond
- **Kept:** no
- **Notes:** six vector instructions per needle at the start of a group,
  sixteen needles at a time, cost more on the four vector ports than the
  shorter chain saves.

## 17. Eight needles per iteration of a wide sweep

- **Idea:** unroll the sweep to eight needles, each key loaded just before
  its compare into a shared register, so that eight bases fit the saved
  registers.
- **check:** pass, 68,654 cases
- **compare:** significantly slower, 0.996x [0.995 – 0.998] (load average 3.2)
- **Kept:** no
- **Notes:** 0.97x from 65,536 to 1,048,576 with 64 needles.

## 18. Blocks of 128 needles in the wide path

- **Idea:** half the block, so a block's bases, keys and probed lines stay
  nearer the core between sweeps.
- **check:** pass, 68,654 cases
- **compare:** no significant difference, 0.999x [0.995 – 1.003] (load average 4.4)
- **Kept:** no

## 19. Other mixes of the two compare forms, again (rejected on local timing)

- **Idea:** attempt 10 again now that each search is a step shorter: eight
  and eight alternating, eight and eight in pairs.
- **check:** not run through the judge
- **compare:** not run through the judge; locally 1% to 5% slower than ten
  and six at every size up to 4,096
- **Kept:** no

## 20. The flag needles' keys loaded in pairs

- **Idea:** ten flag needles sit in pairs in `v` (needles 0 and 1, 3 and 4,
  ...), so their integer copies of the keys can be loaded with five `ldp`
  in place of ten `ldr` at the start of each group.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.011x [1.008 – 1.014] (load average 3.9)
- **Kept:** yes
- **Notes:** 2% at 16 elements, 1% at 4,096: the start of a group is a
  larger part of a short search.

## 21. Prefetch the next sweep's probe in the wide path (rejected on local timing)

- **Idea:** once a needle's base is updated, the address of its probe in
  the next sweep is known exactly (`&a[base + h/2 - 1]`), a whole sweep
  ahead; issue a `prfm` for it.
- **check:** not run through the judge (local harness: correct)
- **compare:** not run through the judge; locally 0.71x to 0.73x with 1,000
  needles and 0.69x to 0.87x with 64, from 65,536 to 16,777,216
- **Kept:** no
- **Notes:** the float64 kernel's log found the same (its attempt 5). The
  sweep already has as many misses in flight as the core will hold; the
  prefetches only take load slots.

## Where this run stopped

Twenty-one attempts, the last six of which gave one kept change (attempt
20, 1.1%). The sizes from 16 to 16,384 are limited by the dependency chain
of a group of sixteen searches (the window cannot reach the next group),
those from 65,536 by the misses of the sweeps.

Not tried, or tried only in part, and why:

- **The path for one needle.** `Lscalar` still runs every step as a probe;
  the count ending of attempt 11 and a preloaded first step would shorten
  it by about a third. The binding sends `np.searchsorted(a, x)` with one
  needle here, so it is the path most calls from Python take. But no timed
  scalar set has fewer needles than a group, so `compare` reports exactly
  1.000x for any change to it and no change can satisfy the one rule. The
  float64 kernels hit the same gap (their log, issue 39 and issue 65); a
  timed scalar set of one needle in `spec.py` would close it.
- **Software pipelining of groups:** starting the next group's first steps
  while the current one ends, so the chain of one group overlaps the next.
  The group code uses every integer register, so this needs a different
  allocation; not attempted in this run.
