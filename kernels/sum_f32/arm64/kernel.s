// float sum_f32(const float *x, size_t n);
//
// x0 = x, x1 = n. Returns np.sum(x), NumPy's bits: the same additions in
// NumPy's pairwise order (see spec.py). The tree is walked with a stack
// below sp; a node whose leaves all lie at one depth below it is summed as a
// group, its 2, 4 or 8 leaves side by side.

// Bytes of one vector of eight elements, the unit the tree is cut in.
.set VEC, 32
// Layout of the bytes below sp the walk uses, as offsets from the moved sp.
.set EXTRAS, 0                // 8 pointers: each leaf's last vector, or ZERO
.set ZERO, 64                   // 32 bytes of +0.0
.set SUMS, 96                   // stack of sums, 4 bytes each
.set NODES, 352                 // node stack, 16 bytes a node
.set FRAME, 2400

// Sum one leaf of x7 >= 1 vectors at x6 into s0: eight running sums, in
// v16 (lanes 0..3) and v17 (lanes 4..7), combined as NumPy combines them.
// Leaves x6 after the vectors.
.macro LEAF
    ldp     q16, q17, [x6], #VEC
    subs    x8, x7, #1
    b.eq    2f
1:
    ldp     q0, q1, [x6], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    subs    x8, x8, #1
    b.ne    1b
2:
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v16.4s, v16.4s, v16.4s
    faddp   s0, v16.2s
.endm

// Add the x2 elements at x6 to s0, one after another.
.macro TAIL
    cbz     x2, 4f
3:
    ldr     s1, [x6], #4
    fadd    s0, s0, s1
    subs    x2, x2, #1
    b.ne    3b
4:
.endm


// For a group of leaves run side by side: leaf j runs from pointer \pj to
// \pnext and has either q vectors or q + 1, with x1 = q * VEC. Stores in
// EXTRAS + 8 * \slot the address of its last vector when it has q + 1, and
// of ZERO, which adds nothing, when it has q. Uses x0.
.macro EXTRA slot, pj, pnext
    sub     x0, \pnext, \pj
    cmp     x0, x1
    sub     x0, \pnext, #VEC
    csel    x0, x0, x7, hi
    str     x0, [sp, #EXTRAS + 8 * \slot]
.endm

.text
.globl _sum_f32
.p2align 2
_sum_f32:
    cmp     x1, #8
    b.lo    Lsmall
    cmp     x1, #128
    b.hi    Lbig
    and     x2, x1, #7
    lsr     x7, x1, #3
    mov     x6, x0
    LEAF
    TAIL
    movi    d1, #0
    fadd    s0, s0, s1
    ret

Lsmall:
    movi    d0, #0
    cbz     x1, 6f
5:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    subs    x1, x1, #1
    b.ne    5b
6:
    ret

Lbig:
    and     x2, x1, #7                  // tail elements of the last leaf
    lsr     x7, x1, #3                  // m: vectors of the node
    mov     x6, x0                      // p: start of the node
    add     x3, x0, x7, lsl #5          // end of the whole vectors
    sub     sp, sp, #FRAME
    add     x5, sp, #SUMS               // stack of sums, next free entry
    add     x4, sp, #NODES
    mov     x9, x4
    stp     xzr, xzr, [sp, #ZERO]
    stp     xzr, xzr, [sp, #ZERO + 16]
Lnode:
    // A node on the right edge holds the tail: a leaf while 8m + t <= 128.
    cbz     x2, Lplain
    add     x8, x6, x7, lsl #5
    cmp     x8, x3
    b.ne    Lplain
    cmp     x7, #15
    b.hi    Lsplit
    LEAF
    TAIL
    b       Lpush
Lplain:
    // A node of up to 128 vectors whose leaves are all at one depth below
    // it is summed as a group: its 2, 4 or 8 leaves side by side.
    cmp     x7, #16
    b.ls    Lleaf
    cmp     x7, #32
    b.ls    Lpair
    cmp     x7, #33
    b.eq    Lsplit
    cmp     x7, #64
    b.ls    Lquad
    cmp     x7, #67
    b.ls    Lsplit
    cmp     x7, #128
    b.ls    Locto
    b       Lsplit
Lleaf:
    LEAF
Lpush:
    str     s0, [x5], #4
Lnext:
    cmp     x4, x9
    b.eq    Ldone
    ldp     x6, x7, [x4, #-16]!
    cbnz    x7, Lnode
    // A combine marker: both children are on the stack of sums.
    ldp     s0, s1, [x5, #-8]!
    fadd    s0, s0, s1
    str     s0, [x5], #4
    b       Lnext
Lsplit:
    lsr     x8, x7, #1                  // left child: m / 2 vectors
    sub     x10, x7, x8                 // right child: the rest
    add     x11, x6, x8, lsl #5
    stp     xzr, xzr, [x4], #16         // combine marker (m = 0)
    stp     x11, x10, [x4], #16         // right child
    mov     x7, x8
    b       Lnode

// ---- Groups. x6 = p, x7 = m. Each pushes its sum and goes to Lnext.

Lpair:      // 17..32 vectors: two leaves, m/2 and the rest
    lsr     x8, x7, #1                  // q
    add     x10, x6, x8, lsl #5         // right leaf
    add     x11, x6, x7, lsl #5         // end
    lsl     x1, x8, #5
    add     x7, sp, #ZERO
    EXTRA   1, x10, x11
    ldp     q16, q17, [x6], #VEC
    ldp     q18, q19, [x10], #VEC
    sub     x8, x8, #1
1:
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x10], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    subs    x8, x8, #1
    b.ne    1b
    ldr     x0, [sp, #EXTRAS + 8]
    ldp     q2, q3, [x0]
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v18.4s, v18.4s, v19.4s
    faddp   v16.4s, v16.4s, v18.4s      // halves of leaf 0, of leaf 1
    faddp   v16.4s, v16.4s, v16.4s      // leaf 0, leaf 1
    faddp   s0, v16.2s
    b       Lpush

Lquad:      // 34..64 vectors: four leaves
    lsr     x8, x7, #2                  // q
    lsr     x11, x7, #1                 // b2
    lsr     x10, x11, #1                // b1
    sub     x0, x7, x11
    add     x12, x11, x0, lsr #1        // b3
    add     x10, x6, x10, lsl #5
    add     x11, x6, x11, lsl #5
    add     x12, x6, x12, lsl #5
    add     x13, x6, x7, lsl #5
    lsl     x1, x8, #5
    add     x7, sp, #ZERO
    EXTRA   0, x6, x10
    EXTRA   1, x10, x11
    EXTRA   2, x11, x12
    EXTRA   3, x12, x13
    ldp     q16, q17, [x6], #VEC
    ldp     q18, q19, [x10], #VEC
    ldp     q20, q21, [x11], #VEC
    ldp     q22, q23, [x12], #VEC
    sub     x8, x8, #1
1:
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x10], #VEC
    ldp     q4, q5, [x11], #VEC
    ldp     q6, q7, [x12], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    fadd    v20.4s, v20.4s, v4.4s
    fadd    v21.4s, v21.4s, v5.4s
    fadd    v22.4s, v22.4s, v6.4s
    fadd    v23.4s, v23.4s, v7.4s
    subs    x8, x8, #1
    b.ne    1b
    ldp     x0, x1, [sp, #EXTRAS]
    ldp     x10, x11, [sp, #EXTRAS + 16]
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x1]
    ldp     q4, q5, [x10]
    ldp     q6, q7, [x11]
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    fadd    v20.4s, v20.4s, v4.4s
    fadd    v21.4s, v21.4s, v5.4s
    fadd    v22.4s, v22.4s, v6.4s
    fadd    v23.4s, v23.4s, v7.4s
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v18.4s, v18.4s, v19.4s
    faddp   v20.4s, v20.4s, v21.4s
    faddp   v22.4s, v22.4s, v23.4s
    faddp   v16.4s, v16.4s, v18.4s
    faddp   v20.4s, v20.4s, v22.4s
    faddp   v16.4s, v16.4s, v20.4s      // leaves 0..3
    faddp   v16.4s, v16.4s, v16.4s      // 0 + 1, 2 + 3
    faddp   s0, v16.2s
    b       Lpush

Locto:      // 68..128 vectors: eight leaves
    lsr     x8, x7, #3                  // q
    lsr     x13, x7, #1                 // b4
    lsr     x11, x13, #1                // b2
    sub     x0, x7, x13
    add     x15, x13, x0, lsr #1        // b6
    lsr     x10, x11, #1                // b1
    sub     x0, x13, x11
    add     x12, x11, x0, lsr #1        // b3
    sub     x0, x15, x13
    add     x14, x13, x0, lsr #1        // b5
    sub     x0, x7, x15
    add     x16, x15, x0, lsr #1        // b7
    add     x10, x6, x10, lsl #5
    add     x11, x6, x11, lsl #5
    add     x12, x6, x12, lsl #5
    add     x13, x6, x13, lsl #5
    add     x14, x6, x14, lsl #5
    add     x15, x6, x15, lsl #5
    add     x16, x6, x16, lsl #5
    add     x17, x6, x7, lsl #5
    lsl     x1, x8, #5
    add     x7, sp, #ZERO
    EXTRA   0, x6, x10
    EXTRA   1, x10, x11
    EXTRA   2, x11, x12
    EXTRA   3, x12, x13
    EXTRA   4, x13, x14
    EXTRA   5, x14, x15
    EXTRA   6, x15, x16
    EXTRA   7, x16, x17
    ldp     q16, q17, [x6], #VEC
    ldp     q18, q19, [x10], #VEC
    ldp     q20, q21, [x11], #VEC
    ldp     q22, q23, [x12], #VEC
    ldp     q24, q25, [x13], #VEC
    ldp     q26, q27, [x14], #VEC
    ldp     q28, q29, [x15], #VEC
    ldp     q30, q31, [x16], #VEC
    sub     x8, x8, #1
1:
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x10], #VEC
    ldp     q4, q5, [x11], #VEC
    ldp     q6, q7, [x12], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    fadd    v20.4s, v20.4s, v4.4s
    fadd    v21.4s, v21.4s, v5.4s
    fadd    v22.4s, v22.4s, v6.4s
    fadd    v23.4s, v23.4s, v7.4s
    ldp     q0, q1, [x13], #VEC
    ldp     q2, q3, [x14], #VEC
    ldp     q4, q5, [x15], #VEC
    ldp     q6, q7, [x16], #VEC
    fadd    v24.4s, v24.4s, v0.4s
    fadd    v25.4s, v25.4s, v1.4s
    fadd    v26.4s, v26.4s, v2.4s
    fadd    v27.4s, v27.4s, v3.4s
    fadd    v28.4s, v28.4s, v4.4s
    fadd    v29.4s, v29.4s, v5.4s
    fadd    v30.4s, v30.4s, v6.4s
    fadd    v31.4s, v31.4s, v7.4s
    subs    x8, x8, #1
    b.ne    1b
    ldp     x0, x1, [sp, #EXTRAS]
    ldp     x10, x11, [sp, #EXTRAS + 16]
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x1]
    ldp     q4, q5, [x10]
    ldp     q6, q7, [x11]
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    fadd    v20.4s, v20.4s, v4.4s
    fadd    v21.4s, v21.4s, v5.4s
    fadd    v22.4s, v22.4s, v6.4s
    fadd    v23.4s, v23.4s, v7.4s
    ldp     x0, x1, [sp, #EXTRAS + 32]
    ldp     x10, x11, [sp, #EXTRAS + 48]
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x1]
    ldp     q4, q5, [x10]
    ldp     q6, q7, [x11]
    fadd    v24.4s, v24.4s, v0.4s
    fadd    v25.4s, v25.4s, v1.4s
    fadd    v26.4s, v26.4s, v2.4s
    fadd    v27.4s, v27.4s, v3.4s
    fadd    v28.4s, v28.4s, v4.4s
    fadd    v29.4s, v29.4s, v5.4s
    fadd    v30.4s, v30.4s, v6.4s
    fadd    v31.4s, v31.4s, v7.4s
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v18.4s, v18.4s, v19.4s
    faddp   v20.4s, v20.4s, v21.4s
    faddp   v22.4s, v22.4s, v23.4s
    faddp   v24.4s, v24.4s, v25.4s
    faddp   v26.4s, v26.4s, v27.4s
    faddp   v28.4s, v28.4s, v29.4s
    faddp   v30.4s, v30.4s, v31.4s
    faddp   v16.4s, v16.4s, v18.4s
    faddp   v20.4s, v20.4s, v22.4s
    faddp   v24.4s, v24.4s, v26.4s
    faddp   v28.4s, v28.4s, v30.4s
    faddp   v16.4s, v16.4s, v20.4s      // leaves 0..3
    faddp   v24.4s, v24.4s, v28.4s      // leaves 4..7
    faddp   v16.4s, v16.4s, v24.4s      // 0 + 1, 2 + 3, 4 + 5, 6 + 7
    faddp   v16.4s, v16.4s, v16.4s      // (0 + 1) + (2 + 3), (4 + 5) + (6 + 7)
    faddp   s0, v16.2s
    b       Lpush

Ldone:
    ldr     s0, [x5, #-4]
    movi    d1, #0
    fadd    s0, s0, s1
    add     sp, sp, #FRAME
    ret
