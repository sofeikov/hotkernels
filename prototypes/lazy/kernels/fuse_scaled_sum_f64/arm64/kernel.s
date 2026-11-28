// void fuse_scaled_sum_f64(const double *a, const double *b, double *out, size_t n, double s);
// out[i] = s * (a[i] + b[i])
//
// x0 = a, x1 = b, x2 = out, x3 = n, d0 = s.
//
// A fused kernel: NumPy's two passes, a + b and then s * that, in one. The
// sum is rounded (fadd) and then multiplied (fmul), as NumPy's loops do it.
//
// Eight elements per iteration, four vectors of two. A remainder of 1 to 7
// elements is one more block that ends exactly at the end of the arrays and
// overlaps the block before it, computed from the inputs again: out must
// not overlap an input. Fewer than 8 elements are done one at a time.

.text
.globl _fuse_scaled_sum_f64
.p2align 2
_fuse_scaled_sum_f64:
    cmp     x3, #8
    b.lo    Lsmall
    dup     v28.2d, v0.d[0]         // s in every lane
    sub     x8, x3, #8              // x8 = elements left after this block
Lblock:
    ldnp    q1, q2, [x0]
    ldnp    q3, q4, [x0, #32]
    ldnp    q16, q17, [x1]
    ldnp    q18, q19, [x1, #32]
    fadd    v1.2d, v1.2d, v16.2d
    fadd    v2.2d, v2.2d, v17.2d
    fadd    v3.2d, v3.2d, v18.2d
    fadd    v4.2d, v4.2d, v19.2d
    fmul    v1.2d, v28.2d, v1.2d
    fmul    v2.2d, v28.2d, v2.2d
    fmul    v3.2d, v28.2d, v3.2d
    fmul    v4.2d, v28.2d, v4.2d
    stp     q1, q2, [x2]
    stp     q3, q4, [x2, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    add     x2, x2, #64
    subs    x8, x8, #8
    b.hs    Lblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Ldone
    // Step back so that one more block ends exactly at the end.
    sub     x9, x8, #8
    add     x0, x0, x9, lsl #3
    add     x1, x1, x9, lsl #3
    add     x2, x2, x9, lsl #3
    mov     x8, #0
    b       Lblock
Lsmall:
    cbz     x3, Ldone
Lone:
    ldr     d1, [x0], #8
    ldr     d2, [x1], #8
    fadd    d1, d1, d2
    fmul    d1, d0, d1
    str     d1, [x2], #8
    subs    x3, x3, #1
    b.ne    Lone
Ldone:
    ret
