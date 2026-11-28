// void fuse_mul_s_mul_v_mul_abs_mul_s_v_add_mul_v_v_sub_s_v_f64(const double *a, const double *b, const double *c, const double *d, const double *e, double *out, size_t n, double s0, double s1, double s2);
// out[i] = s0 * (a[i] * (__builtin_fabs((s1 * b[i])) * ((c[i] * d[i]) + (s2 - e[i]))))
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = e, x5 = out, x6 = n, d0 = s0, d1 = s1, d2 = s2.
//
// A fused kernel for the shape mul(S0,mul(V0,mul(abs(mul(S1,V1)),add(mul(V2,V3),sub(S2,V4))))), written by
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
.globl _fuse_mul_s_mul_v_mul_abs_mul_s_v_add_mul_v_v_sub_s_v_f64
.p2align 2
_fuse_mul_s_mul_v_mul_abs_mul_s_v_add_mul_v_v_sub_s_v_f64:
    cmp     x6, #8
    b.lo    Lsmall
    dup     v31.2d, v0.d[0]         // s0 in every lane
    dup     v30.2d, v1.d[0]         // s1 in every lane
    dup     v29.2d, v2.d[0]         // s2 in every lane
    sub     x8, x6, #8              // x8 = elements left after this block
Lblock:
    ldp     q16, q17, [x2]
    ldp     q18, q19, [x2, #32]
    ldp     q20, q21, [x3]
    ldp     q22, q23, [x3, #32]
    fmul    v16.2d, v16.2d, v20.2d
    fmul    v17.2d, v17.2d, v21.2d
    fmul    v18.2d, v18.2d, v22.2d
    fmul    v19.2d, v19.2d, v23.2d
    ldp     q20, q21, [x4]
    ldp     q22, q23, [x4, #32]
    fsub    v20.2d, v29.2d, v20.2d
    fsub    v21.2d, v29.2d, v21.2d
    fsub    v22.2d, v29.2d, v22.2d
    fsub    v23.2d, v29.2d, v23.2d
    fadd    v16.2d, v16.2d, v20.2d
    fadd    v17.2d, v17.2d, v21.2d
    fadd    v18.2d, v18.2d, v22.2d
    fadd    v19.2d, v19.2d, v23.2d
    ldp     q20, q21, [x1]
    ldp     q22, q23, [x1, #32]
    fmul    v20.2d, v30.2d, v20.2d
    fmul    v21.2d, v30.2d, v21.2d
    fmul    v22.2d, v30.2d, v22.2d
    fmul    v23.2d, v30.2d, v23.2d
    fabs    v20.2d, v20.2d
    fabs    v21.2d, v21.2d
    fabs    v22.2d, v22.2d
    fabs    v23.2d, v23.2d
    fmul    v20.2d, v20.2d, v16.2d
    fmul    v21.2d, v21.2d, v17.2d
    fmul    v22.2d, v22.2d, v18.2d
    fmul    v23.2d, v23.2d, v19.2d
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    fmul    v16.2d, v16.2d, v20.2d
    fmul    v17.2d, v17.2d, v21.2d
    fmul    v18.2d, v18.2d, v22.2d
    fmul    v19.2d, v19.2d, v23.2d
    fmul    v16.2d, v31.2d, v16.2d
    fmul    v17.2d, v31.2d, v17.2d
    fmul    v18.2d, v31.2d, v18.2d
    fmul    v19.2d, v31.2d, v19.2d
    stp     q16, q17, [x5]
    stp     q18, q19, [x5, #32]
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
    sub     x9, x8, #8              // back by what is missing from a whole block
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
    ldr     d16, [x2], #8
    ldr     d17, [x3], #8
    fmul    d16, d16, d17
    ldr     d17, [x4], #8
    fsub    d17, d2, d17
    fadd    d16, d16, d17
    ldr     d17, [x1], #8
    fmul    d17, d1, d17
    fabs    d17, d17
    fmul    d17, d17, d16
    ldr     d16, [x0], #8
    fmul    d16, d16, d17
    fmul    d16, d0, d16
    str     d16, [x5], #8
    subs    x6, x6, #1
    b.ne    Lone
Ldone:
    ret
