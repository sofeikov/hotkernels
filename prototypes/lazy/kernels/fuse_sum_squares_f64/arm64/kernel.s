// double fuse_sum_squares_f64(const double *x, size_t n);
//
// x0 = x, x1 = n. Returns np.sum(x * x) with NumPy's bits.
//
// A fused kernel: NumPy's two passes, x * x into a temporary and the sum of
// that, in one. Each element is squared (fmul) and the rounded square is
// added (fadd); no fmla, which rounds once. The additions are made in the
// order NumPy makes them, its pairwise summation (spec.py), or the sum
// would be another number:
//
//   - fewer than 8 squares: one after another, from +0.0;
//   - a leaf of 8 to 128: eight running sums in four registers, square i
//     into sum i mod 8, combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) +
//     (r6 + r7)), then the last n mod 8 squares one after another;
//   - more than 128: split at n / 2 rounded down to a multiple of 8, and the
//     sums of the two halves added, the first half's as the first operand.
//
// The split is a recursion in NumPy. Here it is a walk with two stacks in a
// frame below sp, since a kernel calls nothing:
//
//   - a stack of work: ranges to sum, as (pointer, length), and markers
//     (length 0) that say "add the two sums on top of the other stack";
//   - a stack of sums.
//
// A range of more than 128 elements is replaced by a marker, its second half
// and its first half, the first half on top, so that it is summed first and
// its sum lies under the second half's when the marker is reached. A range
// in the walk is never shorter than 64 elements, so length 0 is free to be
// the marker. An array of 2^61 elements is 54 splits deep: at most 109
// ranges and markers and 55 sums are stacked at once, which the frame holds.

// Elements NumPy adds as one leaf of eight running sums, at most.
.set LEAF, 128
// Bytes of the frame: room for 112 ranges and markers of 16 bytes, then for 64 sums.
.set FRAME, 2304
// Where the stack of sums starts in the frame.
.set SUMS, 1792

.text
.globl _fuse_sum_squares_f64
.p2align 2
_fuse_sum_squares_f64:
    cmp     x1, #8
    b.lo    Ltiny
    sub     sp, sp, #FRAME
    mov     x9, sp                  // x9 = top of the stack of work: the next free slot
    add     x10, sp, #SUMS          // x10 = top of the stack of sums
    stp     x0, x1, [x9], #16       // the whole array
Lnext:
    cmp     sp, x9                  // sp can only be the first operand of a compare
    b.eq    Lfinish                 // no work left
    ldp     x2, x3, [x9, #-16]!     // x2 = pointer, x3 = length
    cbz     x3, Ladd
    cmp     x3, #LEAF
    b.ls    Lleaf
    lsr     x4, x3, #1
    and     x4, x4, #0xfffffffffffffff8     // x4 = length of the first half
    stp     xzr, xzr, [x9], #16     // the marker, reached when both halves are summed
    add     x5, x2, x4, lsl #3
    sub     x6, x3, x4
    stp     x5, x6, [x9], #16       // the second half
    stp     x2, x4, [x9], #16       // the first half, on top
    b       Lnext
Ladd:
    ldp     d1, d2, [x10, #-16]!    // d1 = the first half's sum, d2 = the second's
    fadd    d1, d1, d2
    str     d1, [x10], #8
    b       Lnext
Lleaf:
    // 8 to 128 elements at x2. v16-v19 are the eight running sums.
    ldp     q16, q17, [x2]
    ldp     q18, q19, [x2, #32]
    add     x2, x2, #64
    and     x4, x3, #7              // x4 = squares added one by one at the end
    lsr     x5, x3, #3
    fmul    v16.2d, v16.2d, v16.2d
    fmul    v17.2d, v17.2d, v17.2d
    fmul    v18.2d, v18.2d, v18.2d
    fmul    v19.2d, v19.2d, v19.2d
    subs    x5, x5, #1              // x5 = groups of eight after the first
    b.eq    Lcombine
Lgroup:
    ldp     q20, q21, [x2]
    ldp     q22, q23, [x2, #32]
    add     x2, x2, #64
    fmul    v20.2d, v20.2d, v20.2d
    fmul    v21.2d, v21.2d, v21.2d
    fmul    v22.2d, v22.2d, v22.2d
    fmul    v23.2d, v23.2d, v23.2d
    fadd    v16.2d, v16.2d, v20.2d
    fadd    v17.2d, v17.2d, v21.2d
    fadd    v18.2d, v18.2d, v22.2d
    fadd    v19.2d, v19.2d, v23.2d
    subs    x5, x5, #1
    b.ne    Lgroup
Lcombine:
    faddp   d16, v16.2d             // r0 + r1
    faddp   d17, v17.2d             // r2 + r3
    faddp   d18, v18.2d             // r4 + r5
    faddp   d19, v19.2d             // r6 + r7
    fadd    d16, d16, d17
    fadd    d18, d18, d19
    fadd    d0, d16, d18
    cbz     x4, Lleafdone
Ltail:
    ldr     d20, [x2], #8
    fmul    d20, d20, d20
    fadd    d0, d0, d20
    subs    x4, x4, #1
    b.ne    Ltail
Lleafdone:
    str     d0, [x10], #8
    b       Lnext
Lfinish:
    ldr     d0, [x10, #-8]          // the one sum left
    add     sp, sp, #FRAME
    movi    d1, #0
    fadd    d0, d1, d0              // the reduction starts from +0.0
    ret
Ltiny:
    movi    d0, #0
    cbz     x1, Ltinydone
Ltinyone:
    ldr     d1, [x0], #8
    fmul    d1, d1, d1
    fadd    d0, d0, d1
    subs    x1, x1, #1
    b.ne    Ltinyone
Ltinydone:
    movi    d1, #0
    fadd    d0, d1, d0
    ret
