# searchsorted_right_i64 attempts (x86_64)

All runs on an Intel Core i5-14600KF, Linux, pinned to one performance core by the judge. Every run is `--unlocked`: the spec and the reference came in the same pull request as the kernel and were not on `main` to lock against. This kernel and `searchsorted_i64` are generated from one source and differ in the compare and the conditional move; each attempt was made on both and judged on each. The reasoning and the measurements behind an attempt are in [that kernel's log](../../searchsorted_i64/x86_64/attempts.md), under the same number.

## 1. Baseline: the left kernel with `<=` for `<`

- **Idea:** side='right' advances when `a[lo + half - 1] <= needle`, which is when the probe is not greater than the needle: `vpcmpgtq` with the needles as its second operand, straight from memory, and `vpandn` to take `half` where the compare is false. `cmovge` in the path for one to three needles.
- **check:** pass
- **compare:** not run (first version). `run`: 4.31x over naive C overall (64 needles 2.48x, 1,000 needles 7.47x), 3.22x over NumPy (4.55x and 2.27x).
- **Kept:** yes
- **Notes:** one instruction fewer per four searches than the left kernel, and the same time: 131.9 ns for 64 needles at 1,024 elements against 132.6.

## 2. Two steps of each search in one pass over the chunk

- **Idea:** as attempt 2 of the left kernel. The needles are loaded into a register once per pass, where the baseline compared against memory at each step.
- **check:** pass
- **compare:** significantly faster, 1.090x [1.088 – 1.093]
- **Kept:** yes
- **Notes:** the same shape as the left kernel: 1.15x to 1.17x up to 4,096 elements, 1.05x to 1.07x to 262,144, nothing above, where a pass takes one step. With two steps at every size it was 1.071x [1.064 – 1.078] overall and 0.93x, 0.91x and 0.88x at the three largest sizes with 1,000 needles.

## 3. The first pass takes two or three steps without a gather

- **Idea:** as attempt 3 of the left kernel. The masks here say that a search did not advance, so the selections are `vpandn` where the left kernel has `vpand`.
- **check:** pass
- **compare:** significantly faster, 1.061x [1.059 – 1.064]
- **Kept:** yes
- **Notes:** 1.22x at 16 elements, 1.15x at 64, 1.06x at 1,024, 1.04x at 65,536, 1.01x to 1.02x from 1,048,576.

## 4. Chunks of 256 needles while `a` fits the second-level cache

- **Idea:** as attempt 4 of the left kernel.
- **check:** pass
- **compare:** significantly faster, 1.007x [1.005 – 1.009]
- **Kept:** yes
- **Notes:** nothing with 64 needles; with 1,000 needles 1.04x at 16 elements, 1.03x at 1,024, 1.02x at 16,384, nothing from 65,536.

## 5. Two steps in a pass for the early steps of a large array

- **Idea:** as attempt 5 of the left kernel.
- **check:** pass
- **compare:** significantly slower, 0.980x [0.977 – 0.982]
- **Kept:** no
- **Notes:** unchanged below 524,288 elements; 0.97x, 0.96x and 0.95x at the three largest sizes with 64 needles, 0.89x, 0.89x and 0.93x with 1,000.

## 6. A pass unrolled over two vectors of searches

- **Idea:** as attempt 6 of the left kernel.
- **check:** pass
- **compare:** not significantly faster, 1.002x [0.999 – 1.004]
- **Kept:** no
- **Notes:** 0.99x at 16 elements for both needle counts and 0.98x at 1,024 with 1,000 needles, 1.01x to 1.02x from 65,536. The left kernel's own figure was 1.006x [1.005 – 1.008], and it was not kept there either; see its log.

## Where it stands

Final `judge run`, recorded, at a load average of 1.0:

`random`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 42.0 ns | 400.0 ns | 9.52x [9.42 – 9.58] | 690.7 ns | 16.4x [16.43 – 16.44] |
| 64 | 64.5 ns | 784.1 ns | 12.2x [12.11 – 12.20] | 772.9 ns | 12.0x [11.97 – 11.98] |
| 256 | 90.9 ns | 328.9 ns | 3.62x [3.61 – 3.62] | 845.2 ns | 9.29x [9.29 – 9.30] |
| 1,024 | 107.3 ns | 383.4 ns | 3.57x [3.57 – 3.58] | 925.6 ns | 8.63x [8.62 – 8.63] |
| 4,096 | 149.1 ns | 423.2 ns | 2.84x [2.84 – 2.84] | 996.8 ns | 6.69x [6.68 – 6.69] |
| 16,384 | 177.8 ns | 488.9 ns | 2.75x [2.75 – 2.75] | 1.1 us | 6.08x [6.07 – 6.08] |
| 65,536 | 238.9 ns | 566.9 ns | 2.37x [2.37 – 2.37] | 1.2 us | 4.96x [4.96 – 4.97] |
| 262,144 | 308.7 ns | 635.4 ns | 2.06x [2.05 – 2.06] | 1.3 us | 4.11x [4.10 – 4.11] |
| 1,048,576 | 776.0 ns | 796.5 ns | 1.03x [1.02 – 1.03] | 1.7 us | 2.20x [2.19 – 2.21] |
| 4,194,304 | 1.5 us | 1.9 us | 1.31x [1.31 – 1.31] | 2.4 us | 1.67x [1.67 – 1.67] |
| 16,777,216 | 2.8 us | 3.6 us | 1.30x [1.27 – 1.33] | 4.1 us | 1.47x [1.45 – 1.50] |
| overall |  |  | 2.85x [2.84 – 2.85] |  | 5.17x [5.16 – 5.18] |

`random, scalars=many`:

| n | kernel | naive C | speedup [95%] | NumPy | speedup [95%] |
|---|---|---|---|---|---|
| 16 | 607.1 ns | 5.3 us | 8.65x [8.61 – 8.70] | 3.1 us | 5.13x [5.12 – 5.13] |
| 64 | 942.9 ns | 11.1 us | 11.7x [11.66 – 11.79] | 4.2 us | 4.50x [4.50 – 4.50] |
| 256 | 1.3 us | 20.1 us | 15.4x [15.26 – 15.44] | 5.4 us | 4.14x [4.13 – 4.14] |
| 1,024 | 1.7 us | 28.1 us | 16.7x [16.66 – 16.73] | 6.5 us | 3.89x [3.88 – 3.89] |
| 4,096 | 1.9 us | 7.8 us | 4.02x [4.00 – 4.04] | 7.7 us | 3.98x [3.98 – 3.98] |
| 16,384 | 3.1 us | 21.0 us | 6.85x [6.80 – 6.90] | 9.0 us | 2.94x [2.93 – 2.94] |
| 65,536 | 4.2 us | 31.1 us | 7.31x [7.25 – 7.36] | 10.7 us | 2.52x [2.51 – 2.52] |
| 262,144 | 6.5 us | 48.2 us | 7.40x [7.37 – 7.44] | 13.5 us | 2.07x [2.06 – 2.07] |
| 1,048,576 | 12.3 us | 124.0 us | 10.1x [10.03 – 10.14] | 18.1 us | 1.48x [1.47 – 1.48] |
| 4,194,304 | 31.4 us | 232.7 us | 7.40x [7.23 – 7.55] | 38.5 us | 1.22x [1.21 – 1.25] |
| 16,777,216 | 56.4 us | 456.1 us | 8.08x [8.05 – 8.11] | 65.5 us | 1.16x [1.15 – 1.17] |
| overall |  |  | 8.76x [8.74 – 8.78] |  | 2.66x [2.66 – 2.67] |

All distributions: 5.00x [4.99 – 5.00] over naive C, 3.71x [3.71 – 3.71] over NumPy

With the output buffer 16 bytes behind its input (64 needles; "apart" is the same kernel with the output clear of the input):

`random`:

| n | kernel | apart | speedup [95%] | naive C | speedup [95%] |
|---|---|---|---|---|---|
| 4,096 | 182.3 ns | 130.9 ns | 0.72x [0.72 – 0.72] | 417.3 ns | 2.29x [2.29 – 2.29] |
| 16,384 | 178.9 ns | 178.9 ns | 1.00x [1.00 – 1.00] n.s. | 488.2 ns | 2.73x [2.72 – 2.73] |
| 65,536 | 243.9 ns | 237.4 ns | 0.97x [0.97 – 0.97] | 560.6 ns | 2.30x [2.30 – 2.30] |
| 262,144 | 359.8 ns | 331.5 ns | 0.92x [0.92 – 0.92] | 626.0 ns | 1.74x [1.74 – 1.74] |
| 1,048,576 | 664.3 ns | 656.8 ns | 0.99x [0.99 – 0.99] | 718.8 ns | 1.08x [1.08 – 1.08] |
| 4,194,304 | 1.4 us | 1.6 us | 1.12x [1.11 – 1.12] | 1.9 us | 1.31x [1.31 – 1.31] |
| 16,777,216 | 3.1 us | 3.0 us | 0.95x [0.93 – 0.97] | 3.4 us | 1.10x [1.09 – 1.14] |
| overall |  |  | 0.95x [0.94 – 0.95] |  | 1.69x [1.68 – 1.70] |

The figure at 4,096 elements in the first table, 149.1 ns, is 132.8 in the `compare` of attempt 2 and 130.9 in the second table here; the 0.72x of the second table did not repeat in the left kernel's runs. Both are discussed in the left kernel's log, with what holds the kernel, the candidates of the design and what was measured outside the judge.
