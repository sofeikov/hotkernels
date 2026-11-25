// void fuse_flux_update_f64(const double *a, const double *b, const double *c, const double *d,
//                           const double *e, const double *f, double *out, size_t n);
// out[i] = a[i] - b[i] * (((c[i] - d[i]) + e[i]) - f[i])
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = e, x5 = f, x6 = out, x7 = n.
//
// A fused kernel: the five passes NumPy makes for the expression (a
// difference, a sum, a difference, a product and a difference, each into a
// temporary) in one. Every operation is the instruction NumPy's loop for it
// executes, in the order the expression is written. No fmls, which would
// round the product and the last difference as one.
//
// Eight elements per iteration, four vectors of two; six loads and a store
// per vector. A remainder of 1 to 7 elements is one more block that ends
// exactly at the end of the arrays and overlaps the block before it,
// computed from the inputs again: out must not overlap an input. Fewer than
// 8 elements are done one at a time.

.text
.globl _fuse_flux_update_f64
.p2align 2
_fuse_flux_update_f64:
    cmp     x7, #8
    b.lo    Lsmall
    sub     x8, x7, #8              // x8 = elements left after this block
Lblock:
    ldp     q0, q1, [x2]
    ldp     q2, q3, [x2, #32]
    ldp     q4, q5, [x3]
    ldp     q6, q7, [x3, #32]
    ldp     q16, q17, [x4]
    ldp     q18, q19, [x4, #32]
    ldp     q20, q21, [x5]
    ldp     q22, q23, [x5, #32]
    fsub    v0.2d, v0.2d, v4.2d     // c - d
    fsub    v1.2d, v1.2d, v5.2d
    fsub    v2.2d, v2.2d, v6.2d
    fsub    v3.2d, v3.2d, v7.2d
    ldp     q4, q5, [x1]
    ldp     q6, q7, [x1, #32]
    fadd    v0.2d, v0.2d, v16.2d    // ... + e
    fadd    v1.2d, v1.2d, v17.2d
    fadd    v2.2d, v2.2d, v18.2d
    fadd    v3.2d, v3.2d, v19.2d
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    fsub    v0.2d, v0.2d, v20.2d    // ... - f
    fsub    v1.2d, v1.2d, v21.2d
    fsub    v2.2d, v2.2d, v22.2d
    fsub    v3.2d, v3.2d, v23.2d
    fmul    v0.2d, v4.2d, v0.2d     // b * that
    fmul    v1.2d, v5.2d, v1.2d
    fmul    v2.2d, v6.2d, v2.2d
    fmul    v3.2d, v7.2d, v3.2d
    fsub    v0.2d, v16.2d, v0.2d    // a - the product
    fsub    v1.2d, v17.2d, v1.2d
    fsub    v2.2d, v18.2d, v2.2d
    fsub    v3.2d, v19.2d, v3.2d
    stp     q0, q1, [x6]
    stp     q2, q3, [x6, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    add     x4, x4, #64
    add     x5, x5, #64
    add     x6, x6, #64
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
    add     x4, x4, x9, lsl #3
    add     x5, x5, x9, lsl #3
    add     x6, x6, x9, lsl #3
    mov     x8, #0
    b       Lblock
Lsmall:
    cbz     x7, Ldone
Lone:
    ldr     d0, [x2], #8
    ldr     d1, [x3], #8
    ldr     d2, [x4], #8
    ldr     d3, [x5], #8
    ldr     d4, [x1], #8
    ldr     d5, [x0], #8
    fsub    d0, d0, d1
    fadd    d0, d0, d2
    fsub    d0, d0, d3
    fmul    d0, d4, d0
    fsub    d0, d5, d0
    str     d0, [x6], #8
    subs    x7, x7, #1
    b.ne    Lone
Ldone:
    ret
