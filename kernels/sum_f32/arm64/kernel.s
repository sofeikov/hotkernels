// float sum_f32(const float *x, size_t n);
//
// x0 = x, x1 = n. Returns np.sum(x), NumPy's bits: the same additions in
// NumPy's pairwise order (see spec.py), each one rounded once, with only the
// operands of an addition swapped.
//
// NumPy's tree in vectors of 8 elements. n = 8m + t. A leaf is at most 128
// elements: eight running sums, element i into sum i mod 8, combined as
// ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then its last t
// elements one by one. A larger node splits at 8 * floor(m / 2) elements, so
// only the nodes on the right edge carry the t elements, and every other
// node is a whole number of vectors, split into floor(m/2) and ceil(m/2).
//
// Signs of zeros. A zero operand changes no sum that is not zero, so the
// sign of an intermediate zero only reaches a result that is zero, and NumPy
// returns 0 + pairwise(x), never -0.0. So running sums may start at +0.0
// instead of the first vector, and the result only needs a final + 0.0.
//
// Paths, by n:
//   0..7       one after another from +0.0
//   8..128     one leaf, 15 steps written out and entered at its length
//   136..256   a multiple of 8: one group of two leaves, straight in
//   544..1024  a multiple of 64: one group of eight equal leaves, straight in
//   others     the walk: a stack of nodes below sp, split until a node is a
//              group or a leaf, and a stack of sums. A marker on the node
//              stack says when the sums of two children are to be added.
//
// Groups. A node of 17..32 vectors is two leaves, of 34..64 four, of
// 68..128 eight (33 and 65..67 have leaves at two depths and are split
// again). A group's leaves are summed in one loop, each leaf in its own two
// registers, so eight leaves are sixteen independent chains of additions and
// the loop runs at the core's three loads per cycle instead of one leaf's
// latency of three cycles a step. The leaves have q or q + 1 vectors; after
// the loop each longer leaf adds its last vector and each shorter one a
// vector of +0.0 from the frame. Leaf sums are combined across leaves by
// faddp in the node's own order.
//
// Groups of eight, q even: leaves whose starts are a multiple of 8 vectors
// apart (every power-of-two size) are taken in a diagonal order, leaf j's
// step k beside leaf 0's step k + j, because the L1 is slow to serve loads of
// the same 32 bytes of different 256-byte blocks together. From LONG_N
// elements, beyond the L1, they read 64 bytes a visit with ldnp, and each
// group prefetches the first line of each leaf of the next.

// Bytes of one vector of eight elements, the unit the tree is cut in.
.set VEC, 32
// Layout of the bytes below sp the walk uses, as offsets from the moved sp.
.set EXTRAS, 0                // 8 pointers: each leaf's last vector, or ZERO
.set ZERO, 64                   // 32 bytes of +0.0
.set SUMS, 96                   // stack of sums, 4 bytes each
.set NODES, 352                 // node stack, 16 bytes a node
.set LONG, 2400                 // non-zero when the array is beyond the L1
.set FRAME, 2416
// From this many elements (128 KB, the first-level cache of a performance
// core) the groups of eight read their leaves 64 bytes at a time.
.set LONG_N, 32768

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

// Two steps of a leaf of a group of eight: its vectors 2k and 2k + 1 after
// the pointer \p, 64 bytes, into its running sums v\lo and v\hi.
.macro DSTEP2 lo, hi, p, k
    ldnp    q0, q1, [\p, #\k * 2 * VEC]
    ldnp    q2, q3, [\p, #\k * 2 * VEC + VEC]
    fadd    v\lo\().4s, v\lo\().4s, v0.4s
    fadd    v\hi\().4s, v\hi\().4s, v1.4s
    fadd    v\lo\().4s, v\lo\().4s, v2.4s
    fadd    v\hi\().4s, v\hi\().4s, v3.4s
.endm

// One step of a leaf of a group of eight: its vector \k after the pointer
// \p into its running sums v\lo (lanes 0..3) and v\hi (lanes 4..7).
.macro DSTEP lo, hi, p, k
    ldp     q0, q1, [\p, #\k * VEC]
    fadd    v\lo\().4s, v\lo\().4s, v0.4s
    fadd    v\hi\().4s, v\hi\().4s, v1.4s
.endm

// Sum a group of two leaves, x7 = m of 17..32 vectors at x6, into s0.
.macro PAIR
    lsr     x8, x7, #1                  // q
    add     x10, x6, x8, lsl #5         // right leaf
    add     x11, x6, x7, lsl #5         // end
    ldp     q16, q17, [x6], #VEC
    ldp     q18, q19, [x10], #VEC
    sub     x8, x8, #1
7:
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x10], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
    subs    x8, x8, #1
    b.ne    7b
    tbz     x7, #0, 8f                  // m odd: the right leaf has q + 1
    ldp     q2, q3, [x11, #-VEC]
    fadd    v18.4s, v18.4s, v2.4s
    fadd    v19.4s, v19.4s, v3.4s
8:
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v18.4s, v18.4s, v19.4s
    faddp   v16.4s, v16.4s, v18.4s      // halves of leaf 0, of leaf 1
    faddp   v16.4s, v16.4s, v16.4s      // leaf 0, leaf 1
    faddp   s0, v16.2s
.endm

.text
.globl _sum_f32
.p2align 2
_sum_f32:
    // 8..128 elements are one leaf, the commonest calls: straight in.
    sub     x2, x1, #8
    cmp     x2, #120
    b.hi    Lnot_leaf
    ldp     q16, q17, [x0], #VEC
    lsr     x8, x2, #3                  // vectors after the first, 0..15
    // Into a straight run of 15 steps of 12 bytes, at the step that leaves
    // exactly x8 of them.
    adr     x9, Lf_sum
    sub     x9, x9, x8, lsl #3
    sub     x9, x9, x8, lsl #2
    br      x9
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
    ldp     q0, q1, [x0], #VEC
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v17.4s, v17.4s, v1.4s
Lf_sum:
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v16.4s, v16.4s, v16.4s
    faddp   s0, v16.2s
    ands    x2, x1, #7
    b.ne    Lf_tail
    movi    d1, #0
    fadd    s0, s0, s1
    ret
Lf_tail:
    ldr     s1, [x0], #4
    fadd    s0, s0, s1
    subs    x2, x2, #1
    b.ne    Lf_tail
    movi    d1, #0
    fadd    s0, s0, s1
    ret

Lnot_leaf:
    cmp     x1, #8
    b.lo    Lsmall
    // 544..1024 elements, a multiple of 64, are one group of eight leaves
    // of equal length, which needs of the frame only the flag LONG.
    sub     x2, x1, #544
    cmp     x2, #1024 - 544
    b.hi    Lnot_eight
    tst     x1, #63
    b.ne    Lbig
    sub     sp, sp, #FRAME
    str     xzr, [sp, #LONG]
    mov     x4, #0                      // no node stack: Lpush returns
    mov     x6, x0
    lsr     x7, x1, #3
    b       Locto
Lnot_eight:
    // 136..256 elements, a multiple of 8, are one group of two leaves.
    sub     x2, x1, #136
    cmp     x2, #256 - 136
    b.hi    Lbig
    tst     x1, #7
    b.ne    Lbig
    mov     x6, x0
    lsr     x7, x1, #3
    PAIR
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
    cmp     x1, #LONG_N
    cset    x0, hs
    str     x0, [sp, #LONG]
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
    // Eight first: nearly all of a long array is groups of eight.
    sub     x0, x7, #68
    cmp     x0, #128 - 68
    b.ls    Locto
    cmp     x7, #128
    b.hi    Lsplit
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
    cbz     x4, Ldone_alone
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
    PAIR
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
    ands    x7, x7, #3                  // 0: every leaf has q vectors
    b.eq    Lq_setup
    add     x7, sp, #ZERO
    EXTRA   0, x6, x10
    EXTRA   1, x10, x11
    EXTRA   2, x11, x12
    EXTRA   3, x12, x13
Lq_setup:
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
    cbz     x7, Lq_sum
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
Lq_sum:
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
    // Beyond the first-level cache, ask for the first line of each leaf of
    // the next group, which is as long as this one and starts where it ends.
    ldr     x0, [sp, #LONG]
    cbz     x0, Lo_noprefetch
    sub     x0, x17, x6
    prfm    pldl1keep, [x6, x0]
    prfm    pldl1keep, [x10, x0]
    prfm    pldl1keep, [x11, x0]
    prfm    pldl1keep, [x12, x0]
    prfm    pldl1keep, [x13, x0]
    prfm    pldl1keep, [x14, x0]
    prfm    pldl1keep, [x15, x0]
    prfm    pldl1keep, [x16, x0]
Lo_noprefetch:
    lsl     x1, x8, #5
    ands    x7, x7, #7                  // 0: every leaf has q vectors
    b.eq    Lo_setup
    add     x7, sp, #ZERO
    EXTRA   0, x6, x10
    EXTRA   1, x10, x11
    EXTRA   2, x11, x12
    EXTRA   3, x12, x13
    EXTRA   4, x13, x14
    EXTRA   5, x14, x15
    EXTRA   6, x15, x16
    EXTRA   7, x16, x17
Lo_setup:
    // The running sums start at +0.0, which changes no sum but the sign of
    // a zero (see the header), so every step is alike.
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
    // When q is even, leaves whose starts are a multiple of 8 vectors apart
    // are common, and the L1 serves loads of the same 32 bytes of different
    // 256-byte blocks at once slowly. So 8 steps are taken in a diagonal
    // order: leaf j's step k beside leaf 0's step k + j.
    tbnz    x8, #0, Lo_plain
    cmp     x8, #8
    b.lo    Lo_plain
    ldr     x0, [sp, #LONG]
    cbz     x0, Lo_diag
    cmp     x8, #16
    b.lo    Lo_diag
Lo_diag16:
    DSTEP2  16, 17, x6, 0
    DSTEP2  16, 17, x6, 1
    DSTEP2  18, 19, x10, 0
    DSTEP2  16, 17, x6, 2
    DSTEP2  18, 19, x10, 1
    DSTEP2  20, 21, x11, 0
    DSTEP2  16, 17, x6, 3
    DSTEP2  18, 19, x10, 2
    DSTEP2  20, 21, x11, 1
    DSTEP2  22, 23, x12, 0
    DSTEP2  16, 17, x6, 4
    DSTEP2  18, 19, x10, 3
    DSTEP2  20, 21, x11, 2
    DSTEP2  22, 23, x12, 1
    DSTEP2  24, 25, x13, 0
    DSTEP2  16, 17, x6, 5
    DSTEP2  18, 19, x10, 4
    DSTEP2  20, 21, x11, 3
    DSTEP2  22, 23, x12, 2
    DSTEP2  24, 25, x13, 1
    DSTEP2  26, 27, x14, 0
    DSTEP2  16, 17, x6, 6
    DSTEP2  18, 19, x10, 5
    DSTEP2  20, 21, x11, 4
    DSTEP2  22, 23, x12, 3
    DSTEP2  24, 25, x13, 2
    DSTEP2  26, 27, x14, 1
    DSTEP2  28, 29, x15, 0
    DSTEP2  16, 17, x6, 7
    DSTEP2  18, 19, x10, 6
    DSTEP2  20, 21, x11, 5
    DSTEP2  22, 23, x12, 4
    DSTEP2  24, 25, x13, 3
    DSTEP2  26, 27, x14, 2
    DSTEP2  28, 29, x15, 1
    DSTEP2  30, 31, x16, 0
    DSTEP2  18, 19, x10, 7
    DSTEP2  20, 21, x11, 6
    DSTEP2  22, 23, x12, 5
    DSTEP2  24, 25, x13, 4
    DSTEP2  26, 27, x14, 3
    DSTEP2  28, 29, x15, 2
    DSTEP2  30, 31, x16, 1
    DSTEP2  20, 21, x11, 7
    DSTEP2  22, 23, x12, 6
    DSTEP2  24, 25, x13, 5
    DSTEP2  26, 27, x14, 4
    DSTEP2  28, 29, x15, 3
    DSTEP2  30, 31, x16, 2
    DSTEP2  22, 23, x12, 7
    DSTEP2  24, 25, x13, 6
    DSTEP2  26, 27, x14, 5
    DSTEP2  28, 29, x15, 4
    DSTEP2  30, 31, x16, 3
    DSTEP2  24, 25, x13, 7
    DSTEP2  26, 27, x14, 6
    DSTEP2  28, 29, x15, 5
    DSTEP2  30, 31, x16, 4
    DSTEP2  26, 27, x14, 7
    DSTEP2  28, 29, x15, 6
    DSTEP2  30, 31, x16, 5
    DSTEP2  28, 29, x15, 7
    DSTEP2  30, 31, x16, 6
    DSTEP2  30, 31, x16, 7
    add     x6, x6, #16 * VEC
    add     x10, x10, #16 * VEC
    add     x11, x11, #16 * VEC
    add     x12, x12, #16 * VEC
    add     x13, x13, #16 * VEC
    add     x14, x14, #16 * VEC
    add     x15, x15, #16 * VEC
    add     x16, x16, #16 * VEC
    sub     x8, x8, #16
    cbz     x8, Lo_extra
    cmp     x8, #8
    b.lo    Lo_plain
Lo_diag:
    DSTEP   16, 17, x6, 0
    DSTEP   16, 17, x6, 1
    DSTEP   18, 19, x10, 0
    DSTEP   16, 17, x6, 2
    DSTEP   18, 19, x10, 1
    DSTEP   20, 21, x11, 0
    DSTEP   16, 17, x6, 3
    DSTEP   18, 19, x10, 2
    DSTEP   20, 21, x11, 1
    DSTEP   22, 23, x12, 0
    DSTEP   16, 17, x6, 4
    DSTEP   18, 19, x10, 3
    DSTEP   20, 21, x11, 2
    DSTEP   22, 23, x12, 1
    DSTEP   24, 25, x13, 0
    DSTEP   16, 17, x6, 5
    DSTEP   18, 19, x10, 4
    DSTEP   20, 21, x11, 3
    DSTEP   22, 23, x12, 2
    DSTEP   24, 25, x13, 1
    DSTEP   26, 27, x14, 0
    DSTEP   16, 17, x6, 6
    DSTEP   18, 19, x10, 5
    DSTEP   20, 21, x11, 4
    DSTEP   22, 23, x12, 3
    DSTEP   24, 25, x13, 2
    DSTEP   26, 27, x14, 1
    DSTEP   28, 29, x15, 0
    DSTEP   16, 17, x6, 7
    DSTEP   18, 19, x10, 6
    DSTEP   20, 21, x11, 5
    DSTEP   22, 23, x12, 4
    DSTEP   24, 25, x13, 3
    DSTEP   26, 27, x14, 2
    DSTEP   28, 29, x15, 1
    DSTEP   30, 31, x16, 0
    DSTEP   18, 19, x10, 7
    DSTEP   20, 21, x11, 6
    DSTEP   22, 23, x12, 5
    DSTEP   24, 25, x13, 4
    DSTEP   26, 27, x14, 3
    DSTEP   28, 29, x15, 2
    DSTEP   30, 31, x16, 1
    DSTEP   20, 21, x11, 7
    DSTEP   22, 23, x12, 6
    DSTEP   24, 25, x13, 5
    DSTEP   26, 27, x14, 4
    DSTEP   28, 29, x15, 3
    DSTEP   30, 31, x16, 2
    DSTEP   22, 23, x12, 7
    DSTEP   24, 25, x13, 6
    DSTEP   26, 27, x14, 5
    DSTEP   28, 29, x15, 4
    DSTEP   30, 31, x16, 3
    DSTEP   24, 25, x13, 7
    DSTEP   26, 27, x14, 6
    DSTEP   28, 29, x15, 5
    DSTEP   30, 31, x16, 4
    DSTEP   26, 27, x14, 7
    DSTEP   28, 29, x15, 6
    DSTEP   30, 31, x16, 5
    DSTEP   28, 29, x15, 7
    DSTEP   30, 31, x16, 6
    DSTEP   30, 31, x16, 7
    add     x6, x6, #8 * VEC
    add     x10, x10, #8 * VEC
    add     x11, x11, #8 * VEC
    add     x12, x12, #8 * VEC
    add     x13, x13, #8 * VEC
    add     x14, x14, #8 * VEC
    add     x15, x15, #8 * VEC
    add     x16, x16, #8 * VEC
    sub     x8, x8, #8
    cmp     x8, #8
    b.hs    Lo_diag
Lo_plain:
    cbz     x8, Lo_extra
Lo_step:
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
    b.ne    Lo_step
Lo_extra:
    cbz     x7, Lo_sum
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
Lo_sum:
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

Ldone_alone:
    movi    d1, #0
    fadd    s0, s0, s1
    add     sp, sp, #FRAME
    ret
Ldone:
    ldr     s0, [x5, #-4]
    movi    d1, #0
    fadd    s0, s0, s1
    add     sp, sp, #FRAME
    ret
