// double sum_f64(const double *x, size_t n);
//
// x0 = x, x1 = n. Returns np.sum(x), NumPy's bits: the same additions in
// NumPy's pairwise order (see spec.py). The tree is walked with a stack
// below sp; a node whose leaves all lie at one depth below it is summed as a
// group, its 2 or 4 leaves side by side.

// Bytes of one vector of eight elements, the unit the tree is cut in.
.set VEC, 64
// Layout of the bytes below sp the walk uses, as offsets from the moved sp.
.set EXTRAS, 0                  // 4 pointers: each leaf's last vector, or ZERO
.set ZERO, 32                   // 64 bytes of +0.0
.set SUMS, 96                   // stack of sums, 8 bytes each
.set NODES, 608                 // node stack, 16 bytes a node
.set FRAME, 2656

// Sum one leaf of x7 >= 1 vectors at x6 into d0: eight running sums, in
// v16..v19 (lanes 0..1, 2..3, 4..5, 6..7), combined as NumPy combines them.
// Leaves x6 after the vectors.
.macro LEAF
    ldp     q16, q17, [x6]
    ldp     q18, q19, [x6, #32]
    add     x6, x6, #VEC
    subs    x8, x7, #1
    b.eq    2f
1:
    ldp     q0, q1, [x6]
    ldp     q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    subs    x8, x8, #1
    b.ne    1b
2:
    faddp   v16.2d, v16.2d, v17.2d
    faddp   v18.2d, v18.2d, v19.2d
    faddp   v16.2d, v16.2d, v18.2d
    faddp   d0, v16.2d
.endm

// Add the x2 elements at x6 to d0, one after another.
.macro TAIL
    cbz     x2, 4f
3:
    ldr     d1, [x6], #8
    fadd    d0, d0, d1
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

// One step of one leaf of a group: the vector at \p, after which \p moves on,
// into its running sums v\a, v\b, v\c, v\d.
.macro STEP p, a, b, c, d
    ldp     q0, q1, [\p], #VEC
    ldp     q2, q3, [\p, #-32]
    fadd    v\a\().2d, v\a\().2d, v0.2d
    fadd    v\b\().2d, v\b\().2d, v1.2d
    fadd    v\c\().2d, v\c\().2d, v2.2d
    fadd    v\d\().2d, v\d\().2d, v3.2d
.endm

// The vector at \p into the running sums v\a, v\b, v\c, v\d; \p stays.
.macro ADDVEC p, a, b, c, d
    ldp     q0, q1, [\p]
    ldp     q2, q3, [\p, #32]
    fadd    v\a\().2d, v\a\().2d, v0.2d
    fadd    v\b\().2d, v\b\().2d, v1.2d
    fadd    v\c\().2d, v\c\().2d, v2.2d
    fadd    v\d\().2d, v\d\().2d, v3.2d
.endm

// Combine one leaf's eight running sums v\a..v\d as NumPy does, as far as
// its two halves: lane 0 of v\a becomes (r0 + r1) + (r2 + r3), lane 1
// (r4 + r5) + (r6 + r7).
.macro HALVES a, b, c, d
    faddp   v\a\().2d, v\a\().2d, v\b\().2d
    faddp   v\c\().2d, v\c\().2d, v\d\().2d
    faddp   v\a\().2d, v\a\().2d, v\c\().2d
.endm

.text
.globl _sum_f64
.p2align 2
_sum_f64:
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
    fadd    d0, d0, d1
    ret

Lsmall:
    movi    d0, #0
    cbz     x1, 6f
5:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1
    subs    x1, x1, #1
    b.ne    5b
6:
    ret

Lbig:
    and     x2, x1, #7                  // tail elements of the last leaf
    lsr     x7, x1, #3                  // m: vectors of the node
    mov     x6, x0                      // p: start of the node
    add     x3, x0, x7, lsl #6          // end of the whole vectors
    sub     sp, sp, #FRAME
    add     x5, sp, #SUMS               // stack of sums, next free entry
    add     x4, sp, #NODES              // node stack, next free entry
    mov     x9, x4                      // node stack, bottom
    stp     xzr, xzr, [sp, #ZERO]
    stp     xzr, xzr, [sp, #ZERO + 16]
    stp     xzr, xzr, [sp, #ZERO + 32]
    stp     xzr, xzr, [sp, #ZERO + 48]
Lnode:
    // A node on the right edge holds the tail: a leaf while 8m + t <= 128.
    cbz     x2, Lplain
    add     x8, x6, x7, lsl #6
    cmp     x8, x3
    b.ne    Lplain
    cmp     x7, #15
    b.hi    Lsplit
    LEAF
    TAIL
    b       Lpush
Lplain:
    // A node of up to 64 vectors whose leaves are all at one depth below it
    // is summed as a group: its 2 or 4 leaves side by side.
    cmp     x7, #16
    b.ls    Lleaf
    cmp     x7, #32
    b.ls    Lpair
    cmp     x7, #33
    b.eq    Lsplit
    cmp     x7, #64
    b.ls    Lquad
    b       Lsplit
Lleaf:
    LEAF
Lpush:
    str     d0, [x5], #8
Lnext:
    cmp     x4, x9
    b.eq    Ldone
    ldp     x6, x7, [x4, #-16]!
    cbnz    x7, Lnode
    // A combine marker: both children are on the stack of sums.
    ldp     d0, d1, [x5, #-16]!
    fadd    d0, d0, d1
    str     d0, [x5], #8
    b       Lnext
Lsplit:
    lsr     x8, x7, #1                  // left child: m / 2 vectors
    sub     x10, x7, x8                 // right child: the rest
    add     x11, x6, x8, lsl #6
    stp     xzr, xzr, [x4], #16         // combine marker (m = 0)
    stp     x11, x10, [x4], #16         // right child
    mov     x7, x8
    b       Lnode

// ---- Groups. x6 = p, x7 = m. Each pushes its sum and goes to Lnext.

Lpair:      // 17..32 vectors: two leaves, m/2 and the rest
    lsr     x8, x7, #1                  // q
    add     x10, x6, x8, lsl #6         // right leaf
    add     x11, x6, x7, lsl #6         // end
    ldp     q16, q17, [x6]
    ldp     q18, q19, [x6, #32]
    add     x6, x6, #VEC
    ldp     q20, q21, [x10]
    ldp     q22, q23, [x10, #32]
    add     x10, x10, #VEC
    sub     x8, x8, #1
1:
    STEP    x6, 16, 17, 18, 19
    STEP    x10, 20, 21, 22, 23
    subs    x8, x8, #1
    b.ne    1b
    tbz     x7, #0, Lp_sum              // m odd: the right leaf has q + 1
    ADDVEC  x10, 20, 21, 22, 23
Lp_sum:
    HALVES  16, 17, 18, 19
    HALVES  20, 21, 22, 23
    faddp   v16.2d, v16.2d, v20.2d      // leaf 0, leaf 1
    faddp   d0, v16.2d
    b       Lpush

Lquad:      // 34..64 vectors: four leaves
    lsr     x8, x7, #2                  // q
    lsr     x11, x7, #1                 // b2
    lsr     x10, x11, #1                // b1
    sub     x0, x7, x11
    add     x12, x11, x0, lsr #1        // b3
    add     x10, x6, x10, lsl #6
    add     x11, x6, x11, lsl #6
    add     x12, x6, x12, lsl #6
    add     x13, x6, x7, lsl #6
    lsl     x1, x8, #6
    ands    x7, x7, #3                  // 0: every leaf has q vectors
    b.eq    Lq_setup
    add     x7, sp, #ZERO
    EXTRA   0, x6, x10
    EXTRA   1, x10, x11
    EXTRA   2, x11, x12
    EXTRA   3, x12, x13
Lq_setup:
    // The running sums start at +0.0, which changes no sum but the sign of
    // a zero, so every step is alike.
    movi    v16.2d, #0
    movi    v17.2d, #0
    movi    v18.2d, #0
    movi    v19.2d, #0
    movi    v20.2d, #0
    movi    v21.2d, #0
    movi    v22.2d, #0
    movi    v23.2d, #0
    movi    v24.2d, #0
    movi    v25.2d, #0
    movi    v26.2d, #0
    movi    v27.2d, #0
    movi    v28.2d, #0
    movi    v29.2d, #0
    movi    v30.2d, #0
    movi    v31.2d, #0
Lq_step:
    STEP    x6, 16, 17, 18, 19
    STEP    x10, 20, 21, 22, 23
    STEP    x11, 24, 25, 26, 27
    STEP    x12, 28, 29, 30, 31
    subs    x8, x8, #1
    b.ne    Lq_step
    cbz     x7, Lq_sum
    ldp     x0, x1, [sp, #EXTRAS]
    ldp     x10, x11, [sp, #EXTRAS + 16]
    ADDVEC  x0, 16, 17, 18, 19
    ADDVEC  x1, 20, 21, 22, 23
    ADDVEC  x10, 24, 25, 26, 27
    ADDVEC  x11, 28, 29, 30, 31
Lq_sum:
    HALVES  16, 17, 18, 19
    HALVES  20, 21, 22, 23
    HALVES  24, 25, 26, 27
    HALVES  28, 29, 30, 31
    faddp   v16.2d, v16.2d, v20.2d      // leaf 0, leaf 1
    faddp   v24.2d, v24.2d, v28.2d      // leaf 2, leaf 3
    faddp   v16.2d, v16.2d, v24.2d      // 0 + 1, 2 + 3
    faddp   d0, v16.2d
    b       Lpush

Ldone:
    ldr     d0, [x5, #-8]
    movi    d1, #0
    fadd    d0, d0, d1
    add     sp, sp, #FRAME
    ret
