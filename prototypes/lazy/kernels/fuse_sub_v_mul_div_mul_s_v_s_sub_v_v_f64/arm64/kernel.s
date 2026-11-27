// void fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64(const double *a, const double *b, const double *c, const double *d, double *out, size_t n, double s0, double s1);
// out[i] = a[i] - (((s0 * b[i]) / s1) * (c[i] - d[i]))
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = out, x5 = n, d0 = s0, d1 = s1.
//
// A fused kernel for the shape sub(V0,mul(div(mul(S0,V1),S1),sub(V2,V3))), written by
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
.globl _fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64
.p2align 2
_fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64:
    cmp     x5, #8
    b.lo    Lsmall
    dup     v31.2d, v0.d[0]         // s0 in every lane
    dup     v30.2d, v1.d[0]         // s1 in every lane
    sub     x8, x5, #8              // x8 = elements left after this block
Lblock:
    ldp     q16, q17, [x2]
    ldp     q18, q19, [x2, #32]
    ldp     q20, q21, [x3]
    ldp     q22, q23, [x3, #32]
    fsub    v16.2d, v16.2d, v20.2d
    fsub    v17.2d, v17.2d, v21.2d
    fsub    v18.2d, v18.2d, v22.2d
    fsub    v19.2d, v19.2d, v23.2d
    ldp     q20, q21, [x1]
    ldp     q22, q23, [x1, #32]
    fmul    v20.2d, v31.2d, v20.2d
    fmul    v21.2d, v31.2d, v21.2d
    fmul    v22.2d, v31.2d, v22.2d
    fmul    v23.2d, v31.2d, v23.2d
    fdiv    v20.2d, v20.2d, v30.2d
    fdiv    v21.2d, v21.2d, v30.2d
    fdiv    v22.2d, v22.2d, v30.2d
    fdiv    v23.2d, v23.2d, v30.2d
    fmul    v20.2d, v20.2d, v16.2d
    fmul    v21.2d, v21.2d, v17.2d
    fmul    v22.2d, v22.2d, v18.2d
    fmul    v23.2d, v23.2d, v19.2d
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    fsub    v16.2d, v16.2d, v20.2d
    fsub    v17.2d, v17.2d, v21.2d
    fsub    v18.2d, v18.2d, v22.2d
    fsub    v19.2d, v19.2d, v23.2d
    stp     q16, q17, [x4]
    stp     q18, q19, [x4, #32]
    add     x0, x0, #64
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    add     x4, x4, #64
    subs    x8, x8, #8
    b.hs    Lblock
    adds    x8, x8, #8              // elements left: 0 to 7
    b.eq    Ldone
    sub     x9, x8, #8              // back by what is missing from a whole block
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
    ldr     d16, [x2], #8
    ldr     d17, [x3], #8
    fsub    d16, d16, d17
    ldr     d17, [x1], #8
    fmul    d17, d0, d17
    fdiv    d17, d17, d1
    fmul    d17, d17, d16
    ldr     d16, [x0], #8
    fsub    d16, d16, d17
    str     d16, [x4], #8
    subs    x5, x5, #1
    b.ne    Lone
Ldone:
    ret
