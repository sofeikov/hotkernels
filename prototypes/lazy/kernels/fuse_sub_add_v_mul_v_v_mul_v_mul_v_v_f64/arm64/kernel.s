// void fuse_sub_add_v_mul_v_v_mul_v_mul_v_v_f64(const double *a, const double *b, const double *c, const double *d, const double *e, const double *f, double *out, size_t n);
// out[i] = (a[i] + (b[i] * c[i])) - (d[i] * (e[i] * f[i]))
//
// x0 = a, x1 = b, x2 = c, x3 = d, x4 = e, x5 = f, x6 = out, x7 = n.
//
// A fused kernel for the shape sub(add(V0,mul(V1,V2)),mul(V3,mul(V4,V5))), written by
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
.globl _fuse_sub_add_v_mul_v_v_mul_v_mul_v_v_f64
.p2align 2
_fuse_sub_add_v_mul_v_v_mul_v_mul_v_v_f64:
    cmp     x7, #8
    b.lo    Lsmall
    sub     x8, x7, #8              // x8 = elements left after this block
Lblock:
    ldp     q16, q17, [x1]
    ldp     q18, q19, [x1, #32]
    ldp     q20, q21, [x2]
    ldp     q22, q23, [x2, #32]
    fmul    v16.2d, v16.2d, v20.2d
    fmul    v17.2d, v17.2d, v21.2d
    fmul    v18.2d, v18.2d, v22.2d
    fmul    v19.2d, v19.2d, v23.2d
    ldp     q20, q21, [x0]
    ldp     q22, q23, [x0, #32]
    fadd    v20.2d, v20.2d, v16.2d
    fadd    v21.2d, v21.2d, v17.2d
    fadd    v22.2d, v22.2d, v18.2d
    fadd    v23.2d, v23.2d, v19.2d
    ldp     q16, q17, [x4]
    ldp     q18, q19, [x4, #32]
    ldp     q24, q25, [x5]
    ldp     q26, q27, [x5, #32]
    fmul    v16.2d, v16.2d, v24.2d
    fmul    v17.2d, v17.2d, v25.2d
    fmul    v18.2d, v18.2d, v26.2d
    fmul    v19.2d, v19.2d, v27.2d
    ldp     q24, q25, [x3]
    ldp     q26, q27, [x3, #32]
    fmul    v24.2d, v24.2d, v16.2d
    fmul    v25.2d, v25.2d, v17.2d
    fmul    v26.2d, v26.2d, v18.2d
    fmul    v27.2d, v27.2d, v19.2d
    fsub    v20.2d, v20.2d, v24.2d
    fsub    v21.2d, v21.2d, v25.2d
    fsub    v22.2d, v22.2d, v26.2d
    fsub    v23.2d, v23.2d, v27.2d
    stp     q20, q21, [x6]
    stp     q22, q23, [x6, #32]
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
    sub     x9, x8, #8              // back by what is missing from a whole block
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
    ldr     d16, [x1], #8
    ldr     d17, [x2], #8
    fmul    d16, d16, d17
    ldr     d17, [x0], #8
    fadd    d17, d17, d16
    ldr     d16, [x4], #8
    ldr     d18, [x5], #8
    fmul    d16, d16, d18
    ldr     d18, [x3], #8
    fmul    d18, d18, d16
    fsub    d17, d17, d18
    str     d17, [x6], #8
    subs    x7, x7, #1
    b.ne    Lone
Ldone:
    ret
