// void fuse_div_sub_v_s_s_f64(const double *a, double *out, size_t n, double s0, double s1);
// out[i] = (a[i] - s0) / s1
//
// x0 = a, x1 = out, x2 = n, d0 = s0, d1 = s1.
//
// A fused kernel for the shape div(sub(V0,S0),S1), written by
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
.globl _fuse_div_sub_v_s_s_f64
.p2align 2
_fuse_div_sub_v_s_s_f64:
    cmp     x2, #8
    b.lo    Lsmall
    dup     v31.2d, v0.d[0]         // s0 in every lane
    dup     v30.2d, v1.d[0]         // s1 in every lane
    sub     x8, x2, #8              // x8 = elements left after this block
Lblock:
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    fsub    v16.2d, v16.2d, v31.2d
    fsub    v17.2d, v17.2d, v31.2d
    fsub    v18.2d, v18.2d, v31.2d
    fsub    v19.2d, v19.2d, v31.2d
    fdiv    v16.2d, v16.2d, v30.2d
    fdiv    v17.2d, v17.2d, v30.2d
    fdiv    v18.2d, v18.2d, v30.2d
    fdiv    v19.2d, v19.2d, v30.2d
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
    fsub    d16, d16, d0
    fdiv    d16, d16, d1
    str     d16, [x1], #8
    subs    x2, x2, #1
    b.ne    Lone
Ldone:
    ret
