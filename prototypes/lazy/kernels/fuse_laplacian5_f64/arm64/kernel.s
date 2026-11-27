// void fuse_laplacian5_f64(const double *a, const double *b, const double *c, const double *d,
//                          const double *e, double *out, size_t n, double s);
// out[i] = s * a[i] - (((b[i] + c[i]) + d[i]) + e[i])
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = e, x5 = out, x6 = n, d0 = s.
//
// A fused kernel: the five passes NumPy makes for a five-point Laplacian
// (one product, three sums, one difference, each into a temporary) in one.
// Every operation is the instruction NumPy's loop for it executes, in the
// order the expression is written: the neighbours are added left to right,
// the centre is multiplied, and the sum is subtracted from the product. No
// fmls, which would round the product and the difference as one.
//
// Eight elements per iteration, four vectors of two; five loads and a store
// per vector. A remainder of 1 to 7 elements is one more block that ends
// exactly at the end of the arrays and overlaps the block before it,
// computed from the inputs again: out must not overlap an input. Fewer than
// 8 elements are done one at a time.

.text
.globl _fuse_laplacian5_f64
.p2align 2
_fuse_laplacian5_f64:
    cmp     x6, #8
    b.lo    Lsmall
    dup     v28.2d, v0.d[0]         // s in every lane
    sub     x8, x6, #8              // x8 = elements left after this block
Lblock:
    ldnp    q0, q1, [x1]
    ldnp    q2, q3, [x1, #32]
    ldnp    q4, q5, [x2]
    ldnp    q6, q7, [x2, #32]
    ldnp    q16, q17, [x3]
    ldnp    q18, q19, [x3, #32]
    ldnp    q20, q21, [x4]
    ldnp    q22, q23, [x4, #32]
    fadd    v0.2d, v0.2d, v4.2d     // b + c
    fadd    v1.2d, v1.2d, v5.2d
    fadd    v2.2d, v2.2d, v6.2d
    fadd    v3.2d, v3.2d, v7.2d
    ldnp    q4, q5, [x0]
    ldnp    q6, q7, [x0, #32]
    fadd    v0.2d, v0.2d, v16.2d    // ... + d
    fadd    v1.2d, v1.2d, v17.2d
    fadd    v2.2d, v2.2d, v18.2d
    fadd    v3.2d, v3.2d, v19.2d
    fmul    v4.2d, v28.2d, v4.2d    // s * a
    fmul    v5.2d, v28.2d, v5.2d
    fmul    v6.2d, v28.2d, v6.2d
    fmul    v7.2d, v28.2d, v7.2d
    fadd    v0.2d, v0.2d, v20.2d    // ... + e
    fadd    v1.2d, v1.2d, v21.2d
    fadd    v2.2d, v2.2d, v22.2d
    fadd    v3.2d, v3.2d, v23.2d
    fsub    v0.2d, v4.2d, v0.2d     // s * a - the sum
    fsub    v1.2d, v5.2d, v1.2d
    fsub    v2.2d, v6.2d, v2.2d
    fsub    v3.2d, v7.2d, v3.2d
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
    ldr     d1, [x1], #8
    ldr     d2, [x2], #8
    ldr     d3, [x3], #8
    ldr     d4, [x4], #8
    ldr     d5, [x0], #8
    fadd    d1, d1, d2
    fadd    d1, d1, d3
    fadd    d1, d1, d4
    fmul    d5, d0, d5
    fsub    d1, d5, d1
    str     d1, [x5], #8
    subs    x6, x6, #1
    b.ne    Lone
Ldone:
    ret
