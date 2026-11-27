// void fuse_flux_limit_f64(const double *a, const double *b, const double *c, const double *d,
//                          double *out, size_t n, double lo, double fill);
// out[i] = a[i] * (b[i] - c[i]) > lo ? fill : d[i]
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = out, x5 = n, d0 = lo, d1 = fill.
//
// A fused kernel: the four passes NumPy makes for
// np.where(a * (b - c) > lo, fill, d), a difference, a product, a comparison
// into a mask of bytes and a selection, in one, with the mask in a register.
// fsub and fmul are the instructions NumPy's loops execute. fcmgt is false
// for a NaN product, as NumPy's comparison is, and bsl copies bits, so a
// NaN, an infinity or a signed zero in d or in fill comes out as it went in.
//
// fcmgt raises the invalid-operation flag for a NaN operand. The layer
// reads the flags after the call (see spec.py).
//
// Eight elements per iteration, four vectors of two. A remainder of 1 to 7
// elements is one more block that ends exactly at the end of the arrays and
// overlaps the block before it, computed from the inputs again: out must
// not overlap an input. Fewer than 8 elements are done one at a time.

.text
.globl _fuse_flux_limit_f64
.p2align 2
_fuse_flux_limit_f64:
    cmp     x5, #8
    b.lo    Lsmall
    dup     v28.2d, v0.d[0]         // lo in every lane
    dup     v29.2d, v1.d[0]         // fill in every lane
    sub     x8, x5, #8              // x8 = elements left after this block
Lblock:
    ldnp    q0, q1, [x1]
    ldnp    q2, q3, [x1, #32]
    ldnp    q4, q5, [x2]
    ldnp    q6, q7, [x2, #32]
    ldnp    q16, q17, [x0]
    ldnp    q18, q19, [x0, #32]
    ldnp    q20, q21, [x3]
    ldnp    q22, q23, [x3, #32]
    fsub    v0.2d, v0.2d, v4.2d     // b - c
    fsub    v1.2d, v1.2d, v5.2d
    fsub    v2.2d, v2.2d, v6.2d
    fsub    v3.2d, v3.2d, v7.2d
    fmul    v0.2d, v16.2d, v0.2d    // a * that
    fmul    v1.2d, v17.2d, v1.2d
    fmul    v2.2d, v18.2d, v2.2d
    fmul    v3.2d, v19.2d, v3.2d
    fcmgt   v0.2d, v0.2d, v28.2d    // all ones where the product is above lo
    fcmgt   v1.2d, v1.2d, v28.2d
    fcmgt   v2.2d, v2.2d, v28.2d
    fcmgt   v3.2d, v3.2d, v28.2d
    bsl     v0.16b, v29.16b, v20.16b    // fill there, d elsewhere
    bsl     v1.16b, v29.16b, v21.16b
    bsl     v2.16b, v29.16b, v22.16b
    bsl     v3.16b, v29.16b, v23.16b
    stp     q0, q1, [x4]
    stp     q2, q3, [x4, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    add     x4, x4, #64
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
    mov     x8, #0
    b       Lblock
Lsmall:
    cbz     x5, Ldone
Lone:
    ldr     d2, [x1], #8
    ldr     d3, [x2], #8
    ldr     d4, [x0], #8
    ldr     d5, [x3], #8
    fsub    d2, d2, d3
    fmul    d2, d4, d2
    fcmp    d2, d0
    fcsel   d2, d1, d5, gt          // fill where the product is above lo; d where it is not, or is NaN
    str     d2, [x4], #8
    subs    x5, x5, #1
    b.ne    Lone
Ldone:
    ret
