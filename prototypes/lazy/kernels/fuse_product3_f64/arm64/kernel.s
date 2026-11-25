// void fuse_product3_f64(const double *a, const double *b, const double *c, double *out, size_t n);
// out[i] = a[i] * (b[i] * c[i])
//
// x0 = a, x1 = b, x2 = c, x3 = out, x4 = n.
//
// A fused kernel: NumPy's two passes, b * c and then a * that, in one. Two
// fmul per element, the inner product rounded before the outer one, as in
// NumPy's loops.
//
// Eight elements per iteration, four vectors of two. A remainder of 1 to 7
// elements is one more block that ends exactly at the end of the arrays and
// overlaps the block before it, computed from the inputs again: out must
// not overlap an input. Fewer than 8 elements are done one at a time.

.text
.globl _fuse_product3_f64
.p2align 2
_fuse_product3_f64:
    cmp     x4, #8
    b.lo    Lsmall
    sub     x8, x4, #8              // x8 = elements left after this block
Lblock:
    ldnp    q0, q1, [x1]
    ldnp    q2, q3, [x1, #32]
    ldnp    q4, q5, [x2]
    ldnp    q6, q7, [x2, #32]
    ldnp    q16, q17, [x0]
    ldnp    q18, q19, [x0, #32]
    fmul    v0.2d, v0.2d, v4.2d
    fmul    v1.2d, v1.2d, v5.2d
    fmul    v2.2d, v2.2d, v6.2d
    fmul    v3.2d, v3.2d, v7.2d
    fmul    v0.2d, v16.2d, v0.2d
    fmul    v1.2d, v17.2d, v1.2d
    fmul    v2.2d, v18.2d, v2.2d
    fmul    v3.2d, v19.2d, v3.2d
    stp     q0, q1, [x3]
    stp     q2, q3, [x3, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    subs    x8, x8, #8
    b.hs    Lblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Ldone
    // Step back so that one more block ends exactly at the end.
    sub     x9, x8, #8
    add     x0, x0, x9, lsl #3
    add     x1, x1, x9, lsl #3
    add     x2, x2, x9, lsl #3
    add     x3, x3, x9, lsl #3
    mov     x8, #0
    b       Lblock
Lsmall:
    cbz     x4, Ldone
Lone:
    ldr     d0, [x1], #8
    ldr     d1, [x2], #8
    ldr     d2, [x0], #8
    fmul    d0, d0, d1
    fmul    d0, d2, d0
    str     d0, [x3], #8
    subs    x4, x4, #1
    b.ne    Lone
Ldone:
    ret
