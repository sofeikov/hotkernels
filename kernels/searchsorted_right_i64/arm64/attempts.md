# searchsorted_right_i64 attempts (arm64)

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

This kernel is `searchsorted_i64` ([its log](../../searchsorted_i64/arm64/attempts.md))
with the comparison turned round: every `a[i] < key` becomes `a[i] <= key`
(`lt` becomes `le` after `cmp`, and `cmgt` with the key first becomes
`cmge`). It was developed after the left side, carrying over each idea that
was kept there in the same order and judging it here on its own. The
attempts are numbered for this kernel; each names the left side's attempt
it carries over, where the idea and the local measurements are described.

## 1. Baseline: branchless upper bound, one needle at a time (left attempt 1)

- **Idea:** as on the left, with `le` in place of `lt`.
- **check:** pass, 68,654 cases
- **compare:** not applicable (first version)
- **Kept:** yes

## 2. Eight needles in step, keys and bases in integer registers (left attempt 2)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.783x [1.766 – 1.800] (load average 3.22)
- **Kept:** yes

## 3. Twelve needles in step (left attempt 3)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.093x [1.088 – 1.097] (load average 3.29)
- **Kept:** yes

## 4. From 65,536 elements one level per sweep over a block (left attempt 4)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.140x [1.128 – 1.151] (load average 5.52)
- **Kept:** yes

## 5. Sixteen needles, half compared in vector registers (left attempt 6)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.127x [1.124 – 1.131] (load average 7.96)
- **Kept:** yes

## 6. The vector mask and-ed with h in a vector register (left attempt 7)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.005x [1.003 – 1.008] (load average 6.65)
- **Kept:** yes

## 7. From 16 elements, the last four probes replaced by a count of eight (left attempt 8)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.021x [1.017 – 1.024] (load average 4.96)
- **Kept:** yes

## 8. The wide path ends with the count over eight elements too (left attempt 9)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.016x [1.014 – 1.018] (load average 6.55)
- **Kept:** yes

## 9. Ten flag needles and six vector needles (left attempt 10)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.019x [1.018 – 1.021] (load average 5.29)
- **Kept:** yes

## 10. In the groups, count over four elements instead of eight (left attempt 11)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.008x [1.003 – 1.013] (load average 3.92)
- **Kept:** yes

## 11. When n is a power of two, the first step is h = p/2 with its probe preloaded (left attempt 15)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.077x [1.075 – 1.080] (load average 4.86)
- **Kept:** yes

## 12. The flag needles' keys loaded in pairs (left attempt 20)

- **Idea:** as on the left.
- **check:** pass, 68,654 cases
- **compare:** significantly faster, 1.016x [1.013 – 1.018] (load average 4.55)
- **Kept:** yes

## The recorded run

`uv run judge run searchsorted_right_i64 --unlocked --record` on the final kernel, against
the spec of PR 245 (check: pass, 63,444 cases), Apple M1 Pro, NumPy 2.5.5,
on 20 November 2026 on a quiet machine (load average 2.9, no CI run, no other kernel
being timed). An earlier run at a load average of 8.8 read 4.51x over the C and
3.38x over NumPy. Report:
`results/searchsorted_right_i64/arm64-apple-m1-pro.json`.

- **Overall:** 4.42x [4.42 – 4.42] over the naive C, 3.36x over NumPy, across both timed
  needle counts. Faster than the C and than NumPy at every size.
- **By size:** 64 needles: 3.13x over the C at 16 elements, 3.93x at 4,096, 4.96x at 1,048,576, 6.69x at 16,777,216; over NumPy 4.70x overall and 2.44x at 1,048,576. 1,000 needles: 3.51x, 3.96x, 5.06x and 9.42x over the C; over NumPy 3.82x at 16, 2.70x at 4,096, 1.76x at 1,048,576 and 1.45x at 16,777,216.
- **Output 16 bytes behind the input:** level (1.00x to 1.07x). The busy run had
  0.98x at 16,384 and 0.96x at 4,194,304, which this run does not repeat.
