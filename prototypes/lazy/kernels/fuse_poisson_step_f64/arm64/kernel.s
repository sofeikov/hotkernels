// void fuse_poisson_step_f64(const double *a, const double *b, const double *c, const double *d,
//                            const double *e, double *out, size_t n,
//                            double s0, double s1, double s2, double s3);
// out[i] = ((a[i] + b[i]) * s0 + (c[i] + d[i]) * s1) / s2 - s3 * e[i]
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = e, x5 = out, x6 = n, d0-d3 = s0-s3.
//
// A fused kernel: the eight passes NumPy makes for a Jacobi step of the
// pressure equation (two sums, two products, a sum, a quotient, a product
// and a difference, each into a temporary) in one. Every operation is the
// instruction NumPy's loop for it executes, in the order the expression is
// written, with the operands of each in the order written: the sum is the
// first operand of `(a + b) * s0`, s3 the first of `s3 * e`. No fmla and no
// fmls, and fdiv by s2, not a product with its reciprocal: each of those
// rounds differently from NumPy's passes.
//
// Eight elements per iteration, four vectors of two; five loads and a store
// per vector, and one fdiv, which is what the loop waits for while the
// arrays fit the cache. A remainder of 1 to 7 elements is one more block
// that ends exactly at the end of the arrays and overlaps the block before
// it, computed from the inputs again: out must not overlap an input. Fewer
// than 8 elements are done one at a time.

.text
.globl _fuse_poisson_step_f64
.p2align 2
_fuse_poisson_step_f64:
    cmp     x6, #8
    b.lo    Lsmall
    dup     v28.2d, v0.d[0]         // s0 in every lane
    dup     v29.2d, v1.d[0]         // s1
    dup     v30.2d, v2.d[0]         // s2
    dup     v31.2d, v3.d[0]         // s3
    sub     x8, x6, #8              // x8 = elements left after this block
Lblock:
    ldnp    q0, q1, [x0]
    ldnp    q2, q3, [x0, #32]
    ldnp    q4, q5, [x1]
    ldnp    q6, q7, [x1, #32]
    ldnp    q16, q17, [x2]
    ldnp    q18, q19, [x2, #32]
    ldnp    q20, q21, [x3]
    ldnp    q22, q23, [x3, #32]
    fadd    v0.2d, v0.2d, v4.2d     // a + b
    fadd    v1.2d, v1.2d, v5.2d
    fadd    v2.2d, v2.2d, v6.2d
    fadd    v3.2d, v3.2d, v7.2d
    fadd    v16.2d, v16.2d, v20.2d  // c + d
    fadd    v17.2d, v17.2d, v21.2d
    fadd    v18.2d, v18.2d, v22.2d
    fadd    v19.2d, v19.2d, v23.2d
    ldnp    q4, q5, [x4]
    ldnp    q6, q7, [x4, #32]
    fmul    v0.2d, v0.2d, v28.2d    // (a + b) * s0
    fmul    v1.2d, v1.2d, v28.2d
    fmul    v2.2d, v2.2d, v28.2d
    fmul    v3.2d, v3.2d, v28.2d
    fmul    v16.2d, v16.2d, v29.2d  // (c + d) * s1
    fmul    v17.2d, v17.2d, v29.2d
    fmul    v18.2d, v18.2d, v29.2d
    fmul    v19.2d, v19.2d, v29.2d
    fmul    v4.2d, v31.2d, v4.2d    // s3 * e
    fmul    v5.2d, v31.2d, v5.2d
    fmul    v6.2d, v31.2d, v6.2d
    fmul    v7.2d, v31.2d, v7.2d
    fadd    v0.2d, v0.2d, v16.2d    // the two weighted sums together
    fadd    v1.2d, v1.2d, v17.2d
    fadd    v2.2d, v2.2d, v18.2d
    fadd    v3.2d, v3.2d, v19.2d
    fdiv    v0.2d, v0.2d, v30.2d    // ... / s2
    fdiv    v1.2d, v1.2d, v30.2d
    fdiv    v2.2d, v2.2d, v30.2d
    fdiv    v3.2d, v3.2d, v30.2d
    fsub    v0.2d, v0.2d, v4.2d     // ... - s3 * e
    fsub    v1.2d, v1.2d, v5.2d
    fsub    v2.2d, v2.2d, v6.2d
    fsub    v3.2d, v3.2d, v7.2d
    stp     q0, q1, [x5]
    stp     q2, q3, [x5, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    add     x4, x4, #64
    add     x5, x5, #64
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
    mov     x8, #0
    b       Lblock
Lsmall:
    cbz     x6, Ldone
Lone:
    ldr     d4, [x0], #8
    ldr     d5, [x1], #8
    ldr     d6, [x2], #8
    ldr     d7, [x3], #8
    ldr     d16, [x4], #8
    fadd    d4, d4, d5
    fadd    d6, d6, d7
    fmul    d4, d4, d0
    fmul    d6, d6, d1
    fmul    d16, d3, d16
    fadd    d4, d4, d6
    fdiv    d4, d4, d2
    fsub    d4, d4, d16
    str     d4, [x5], #8
    subs    x6, x6, #1
    b.ne    Lone
Ldone:
    ret
