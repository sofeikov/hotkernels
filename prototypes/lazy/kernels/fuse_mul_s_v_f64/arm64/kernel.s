// void fuse_mul_s_v_f64(const double *a, double *out, size_t n, double s0);
// out[i] = s0 * a[i]
//
// x0 = a, x1 = out, x2 = n, d0 = s0.
//
// A fused kernel for the shape mul(S0,V0), written by
// prototypes/lazy/make_kernel.py from that name. Every operation is one
// instruction that rounds once, in the order NumPy's passes make them: no
// multiply-add, and a division is a division.
//
// Eight elements per iteration, four vectors of two. Every load of an
// iteration comes before its store, so out may be one of the arrays itself
// for a length that is a multiple of eight or below eight. A remainder of 1
// to 7 elements is one more block that ends exactly at the end of the arrays
// and overlaps the block before it, computed from the arrays again: there
// out must not overlap an array. Fewer than 8 elements are done one at a time.

.text
.globl _fuse_mul_s_v_f64
.p2align 2
_fuse_mul_s_v_f64:
    cmp     x2, #8
    b.lo    Lsmall
    dup     v31.2d, v0.d[0]         // s0 in every lane
    sub     x8, x2, #8              // x8 = elements left after this block
Lblock:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    fmul    v16.2d, v31.2d, v16.2d
    fmul    v17.2d, v31.2d, v17.2d
    fmul    v18.2d, v31.2d, v18.2d
    fmul    v19.2d, v31.2d, v19.2d
    stp     q16, q17, [x1]
    stp     q18, q19, [x1, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    subs    x8, x8, #8
    b.hs    Lblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Ldone
    sub     x9, x8, #8              // back by what is missing from a whole block
    add     x0, x0, x9, lsl #3
    add     x1, x1, x9, lsl #3
    mov     x8, #0
    b       Lblock
Lsmall:
    cbz     x2, Ldone
Lone:
    ldr     d16, [x0], #8
    fmul    d16, d0, d16
    str     d16, [x1], #8
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
