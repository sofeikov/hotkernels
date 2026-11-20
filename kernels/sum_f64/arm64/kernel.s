// double sum_f64(const double *x, size_t n);
//
// x0 = x, x1 = n. Returns np.sum(x), NumPy's bits: the same additions in
// NumPy's pairwise order (see spec.py), each one rounded once, with only the
// operands of an addition swapped. sum_f32's kernel is the same design at
// half the width, and its header has the proofs.
//
// NumPy's tree in vectors of 8 elements, here 64 bytes, four registers.
// n = 8m + t. A leaf is at most 128 elements: eight running sums, element i
// into sum i mod 8, combined as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) +
// (r6 + r7)), then its last t elements one by one. A larger node splits at
// 8 * floor(m / 2) elements: only the nodes on the right edge carry the t
// elements, and every other node is floor(m/2) and ceil(m/2) vectors.
//
// Signs of zeros. A zero operand changes no sum that is not zero, and NumPy
// returns 0 + pairwise(x), never -0.0, so running sums may start at +0.0 and
// the result needs a final + 0.0.
//
// Paths, by n:
//   0..7       one after another from +0.0
//   8..128     one leaf, 15 steps written out and entered at its length
//   others     the walk: a stack of nodes below sp, split until a node is a
//              group or a leaf, and a stack of sums. A marker on the node
//              stack says when the sums of two children are to be added.
//
// Groups, while the array fits the first-level cache (up to LONG_N
// elements). A node of 17..32 vectors is two leaves, of 34..64 four (33 has
// leaves at two depths and is split again); eight leaves would need 32
// registers of running sums. A group's leaves are summed in one loop, four
// leaves being sixteen independent chains of additions. Leaves have q or
// q + 1 vectors; after the loop each longer leaf adds its last vector and
// each shorter one 64 bytes of +0.0 from the frame. When q is even, eight
// steps are taken in a diagonal order, leaf j's step k beside leaf 0's step
// k + j, because the L1 is slow to serve loads of one offset in different
// 256-byte blocks together.
//
// Beyond the first-level cache there are no groups: the leaves are summed
// one after another, one stream through memory, and the out-of-order core
// overlaps neighbouring leaves. Above HUGE_N elements their loads are ldnp.

// Bytes of one vector of eight elements, the unit the tree is cut in.
.set VEC, 64
// Layout of the bytes below sp the walk uses, as offsets from the moved sp.
.set EXTRAS, 0                  // 4 pointers: each leaf's last vector, or ZERO
.set ZERO, 32                   // 64 bytes of +0.0
.set SUMS, 96                   // stack of sums, 8 bytes each
.set NODES, 608                 // node stack, 16 bytes a node
.set SAVED, 2656                // two words kept across a QUAD4
.set FRAME, 2672
// Above this many elements (128 KB, the first-level cache of a performance
// core) the leaves are summed one at a time.
.set LONG_N, 16384
// Above this many elements (4 MB) their loads are non-temporal, ldnp; x14 is
// 2 then.
.set HUGE_N, 2097152

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

// LEAF with non-temporal loads, for arrays beyond the second-level cache.
.macro LEAF_NT
    ldp     q16, q17, [x6]
    ldp     q18, q19, [x6, #32]
    add     x6, x6, #VEC
    subs    x8, x7, #1
    b.eq    2f
1:
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
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

// One step of one leaf of a group: its vector \k after the pointer \p into
// its running sums v\a, v\b, v\c, v\d.
.macro DSTEP p, k, a, b, c, d
    ldp     q0, q1, [\p, #\k * VEC]
    ldp     q2, q3, [\p, #\k * VEC + 32]
    fadd    v\a\().2d, v\a\().2d, v0.2d
    fadd    v\b\().2d, v\b\().2d, v1.2d
    fadd    v\c\().2d, v\c\().2d, v2.2d
    fadd    v\d\().2d, v\d\().2d, v3.2d
.endm

.macro LEAF_RUN
    ldp     q16, q17, [x6]
    ldp     q18, q19, [x6, #32]
    add     x6, x6, #VEC
    sub     x8, x7, #1
    adr     x0, 9f
    sub     x0, x0, x8, lsl #4
    sub     x0, x0, x8, lsl #3
    br      x0
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x6], #VEC
    ldp     q2, q3, [x6, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
9:
    faddp   v16.2d, v16.2d, v17.2d
    faddp   v18.2d, v18.2d, v19.2d
    faddp   v16.2d, v16.2d, v18.2d
    faddp   d0, v16.2d
.endm

// Sum a group of four leaves, x7 = m of 34..64 vectors at x6, into d0.
// Uses x0, x1, x6..x8, x10..x13, v0..v3 and v16..v31.
.macro QUAD
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
    b.eq    1f
    add     x7, sp, #ZERO
    EXTRA   0, x6, x10
    EXTRA   1, x10, x11
    EXTRA   2, x11, x12
    EXTRA   3, x12, x13
1:
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
    // When q is even, leaves whose starts are a multiple of 4 vectors apart
    // are common, and the L1 serves loads of the same offset in different
    // 256-byte blocks at once slowly (sum_f32's log). So 8 steps are taken in
    // a diagonal order: leaf j's step k beside leaf 0's step k + j.
    tbnz    x8, #0, 3f
    cmp     x8, #8
    b.lo    3f
2:
    DSTEP   x6, 0, 16, 17, 18, 19
    DSTEP   x6, 1, 16, 17, 18, 19
    DSTEP   x10, 0, 20, 21, 22, 23
    DSTEP   x6, 2, 16, 17, 18, 19
    DSTEP   x10, 1, 20, 21, 22, 23
    DSTEP   x11, 0, 24, 25, 26, 27
    DSTEP   x6, 3, 16, 17, 18, 19
    DSTEP   x10, 2, 20, 21, 22, 23
    DSTEP   x11, 1, 24, 25, 26, 27
    DSTEP   x12, 0, 28, 29, 30, 31
    DSTEP   x6, 4, 16, 17, 18, 19
    DSTEP   x10, 3, 20, 21, 22, 23
    DSTEP   x11, 2, 24, 25, 26, 27
    DSTEP   x12, 1, 28, 29, 30, 31
    DSTEP   x6, 5, 16, 17, 18, 19
    DSTEP   x10, 4, 20, 21, 22, 23
    DSTEP   x11, 3, 24, 25, 26, 27
    DSTEP   x12, 2, 28, 29, 30, 31
    DSTEP   x6, 6, 16, 17, 18, 19
    DSTEP   x10, 5, 20, 21, 22, 23
    DSTEP   x11, 4, 24, 25, 26, 27
    DSTEP   x12, 3, 28, 29, 30, 31
    DSTEP   x6, 7, 16, 17, 18, 19
    DSTEP   x10, 6, 20, 21, 22, 23
    DSTEP   x11, 5, 24, 25, 26, 27
    DSTEP   x12, 4, 28, 29, 30, 31
    DSTEP   x10, 7, 20, 21, 22, 23
    DSTEP   x11, 6, 24, 25, 26, 27
    DSTEP   x12, 5, 28, 29, 30, 31
    DSTEP   x11, 7, 24, 25, 26, 27
    DSTEP   x12, 6, 28, 29, 30, 31
    DSTEP   x12, 7, 28, 29, 30, 31
    add     x6, x6, #8 * VEC
    add     x10, x10, #8 * VEC
    add     x11, x11, #8 * VEC
    add     x12, x12, #8 * VEC
    sub     x8, x8, #8
    cmp     x8, #8
    b.hs    2b
    cbz     x8, 4f
3:
    STEP    x6, 16, 17, 18, 19
    STEP    x10, 20, 21, 22, 23
    STEP    x11, 24, 25, 26, 27
    STEP    x12, 28, 29, 30, 31
    subs    x8, x8, #1
    b.ne    3b
4:
    cbz     x7, 5f
    ldp     x0, x1, [sp, #EXTRAS]
    ldp     x10, x11, [sp, #EXTRAS + 16]
    ADDVEC  x0, 16, 17, 18, 19
    ADDVEC  x1, 20, 21, 22, 23
    ADDVEC  x10, 24, 25, 26, 27
    ADDVEC  x11, 28, 29, 30, 31
5:
    HALVES  16, 17, 18, 19
    HALVES  20, 21, 22, 23
    HALVES  24, 25, 26, 27
    HALVES  28, 29, 30, 31
    faddp   v16.2d, v16.2d, v20.2d      // leaf 0, leaf 1
    faddp   v24.2d, v24.2d, v28.2d      // leaf 2, leaf 3
    faddp   v16.2d, v16.2d, v24.2d      // 0 + 1, 2 + 3
    faddp   d0, v16.2d
.endm

// Sum the eight leaves of a node of x7 = 68..128 vectors at x6 one after
// another into d0; x6 ends after the node. Uses x0, x1, x6..x8, x10..x17,
// v0..v3, v16..v27.
.macro OCTO_SEQ
    lsr     x13, x7, #1                 // b4
    lsr     x11, x13, #1                // b2
    sub     x0, x7, x13
    add     x15, x13, x0, lsr #1        // b6
    lsr     x10, x11, #1                // b1
    sub     x0, x13, x11
    add     x12, x11, x0, lsr #1        // b3
    sub     x0, x15, x13
    add     x1, x13, x0, lsr #1         // b5
    sub     x0, x7, x15
    add     x16, x15, x0, lsr #1        // b7
    add     x10, x6, x10, lsl #6
    add     x11, x6, x11, lsl #6
    add     x12, x6, x12, lsl #6
    add     x13, x6, x13, lsl #6
    add     x1, x6, x1, lsl #6
    add     x15, x6, x15, lsl #6
    add     x16, x6, x16, lsl #6
    add     x17, x6, x7, lsl #6
    LEAFAT  x10, 20
    LEAFAT  x11, 21
    LEAFAT  x12, 22
    LEAFAT  x13, 23
    LEAFAT  x1, 24
    LEAFAT  x15, 25
    LEAFAT  x16, 26
    LEAFAT  x17, 27
    fadd    d20, d20, d21
    fadd    d22, d22, d23
    fadd    d24, d24, d25
    fadd    d26, d26, d27
    fadd    d20, d20, d22
    fadd    d24, d24, d26
    fadd    d0, d20, d24
.endm

// Sum the four groups of four of a node of x7 = 136..256 vectors at x6 into
// d0, one after another. Uses what QUAD does and x8, x9, x15..x17, v5, v6;
// restores x9.
.macro QUAD4
    lsr     x8, x7, #1                  // left half
    sub     x9, x7, x8                  // right half
    add     x17, x6, x8, lsl #6         // its start
    lsr     x16, x8, #1                 // the left half's left group
    add     x15, x6, x16, lsl #6
    sub     x16, x8, x16                // the left half's right group
    lsr     x7, x8, #1
    QUAD
    fmov    d5, d0
    mov     x6, x15
    mov     x7, x16
    QUAD
    fadd    d5, d5, d0                  // left half
    lsr     x16, x9, #1
    add     x15, x17, x16, lsl #6
    sub     x9, x9, x16
    mov     x6, x17
    mov     x7, x16
    QUAD
    fmov    d6, d0
    mov     x6, x15
    mov     x7, x9
    QUAD
    fadd    d6, d6, d0                  // right half
    fadd    d0, d5, d6
    add     x9, sp, #NODES              // the bottom of the node stack, borrowed
.endm

.text
.globl _sum_f64
.p2align 2
_sum_f64:
    // 8..128 elements are one leaf, the commonest calls: straight in.
    sub     x2, x1, #8
    cmp     x2, #120
    b.hi    Lnot_leaf
    ldp     q16, q17, [x0]
    ldp     q18, q19, [x0, #32]
    add     x0, x0, #VEC
    lsr     x8, x2, #3                  // vectors after the first, 0..15
    // Into a straight run of 15 steps of 24 bytes, at the step that leaves
    // exactly x8 of them.
    adr     x9, Lf_sum
    sub     x9, x9, x8, lsl #4
    sub     x9, x9, x8, lsl #3
    br      x9
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldp     q0, q1, [x0], #VEC
    ldp     q2, q3, [x0, #-32]
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
Lf_sum:
    faddp   v16.2d, v16.2d, v17.2d
    faddp   v18.2d, v18.2d, v19.2d
    faddp   v16.2d, v16.2d, v18.2d
    faddp   d0, v16.2d
    ands    x2, x1, #7
    b.ne    Lf_tail
    movi    d1, #0
    fadd    d0, d0, d1
    ret
Lf_tail:
    ldr     d1, [x0], #8
    fadd    d0, d0, d1
    subs    x2, x2, #1
    b.ne    Lf_tail
    movi    d1, #0
    fadd    d0, d0, d1
    ret

Lnot_leaf:
    cmp     x1, #8
    b.hs    Lbig

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
    // x14: 0 for an array that fits the first-level cache, 1 above LONG_N
    // elements, 2 above HUGE_N.
    cmp     x1, #LONG_N
    cset    x14, hi
    mov     x8, #HUGE_N
    cmp     x1, x8
    cinc    x14, x14, hi
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
    // Beyond the first-level cache, leaves one at a time, in the order of
    // memory: there the groups' several streams cost more than they gain.
    cbnz    x14, Lplain_long
    cmp     x7, #32
    b.ls    Lpair
    cmp     x7, #33
    b.eq    Lsplit
    cmp     x7, #64
    b.ls    Lquad
    cmp     x7, #67
    b.ls    Lsplit
    cmp     x7, #128
    b.ls    Lquad2
    cmp     x7, #135
    b.ls    Lsplit
    cmp     x7, #256
    b.ls    Lquad4
    cmp     x7, #271
    b.ls    Lsplit
    cmp     x7, #512
    b.ls    Lquad8
    b       Lsplit
Lplain_long:
    cmp     x7, #32
    b.ls    Lpair_seq
    cmp     x7, #33
    b.eq    Lsplit
    cmp     x7, #64
    b.ls    Lquad_seq
    cmp     x7, #67
    b.ls    Lsplit
    cmp     x7, #128
    b.ls    Locto_seq
    cmp     x7, #135
    b.ls    Lsplit
    cmp     x7, #256
    b.ls    Locto2_seq
    b       Lsplit
Lleaf:
    cmp     x14, #2
    b.eq    Lleaf_nt
    LEAF_RUN
    b       Lpush
Lleaf_nt:
    LEAF_NT
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
    QUAD
    b       Lpush

// ---- Beyond the first-level cache: a group's leaves one after another, in
// the order of memory, each a straight run, combined in registers.
// LEAFAT \dst: the leaf from x6 to \end into d\dst; x6 ends at \end.
.macro LEAF_RUN_NT
    ldnp    q16, q17, [x6]
    ldnp    q18, q19, [x6, #32]
    add     x6, x6, #VEC
    sub     x8, x7, #1
    adr     x0, 9f
    sub     x0, x0, x8, lsl #5
    add     x0, x0, x8, lsl #2
    br      x0
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
    ldnp    q0, q1, [x6]
    ldnp    q2, q3, [x6, #32]
    add     x6, x6, #VEC
    fadd    v16.2d, v16.2d, v0.2d
    fadd    v17.2d, v17.2d, v1.2d
    fadd    v18.2d, v18.2d, v2.2d
    fadd    v19.2d, v19.2d, v3.2d
9:
    faddp   v16.2d, v16.2d, v17.2d
    faddp   v18.2d, v18.2d, v19.2d
    faddp   v16.2d, v16.2d, v18.2d
    faddp   d0, v16.2d
.endm

.macro LEAFAT end, dst
    sub     x7, \end, x6
    lsr     x7, x7, #6
    cmp     x14, #2
    b.eq    5f
    LEAF_RUN
    b       6f
5:
    LEAF_RUN_NT
6:
    fmov    d\dst, d0
.endm

Lpair_seq:  // 17..32 vectors: two leaves
    lsr     x8, x7, #1
    add     x10, x6, x8, lsl #6
    add     x11, x6, x7, lsl #6
    LEAFAT  x10, 20
    LEAFAT  x11, 21
    fadd    d0, d20, d21
    b       Lpush

Lquad_seq:  // 34..64 vectors: four leaves
    lsr     x11, x7, #1                 // b2
    lsr     x10, x11, #1                // b1
    sub     x0, x7, x11
    add     x12, x11, x0, lsr #1        // b3
    add     x10, x6, x10, lsl #6
    add     x11, x6, x11, lsl #6
    add     x12, x6, x12, lsl #6
    add     x13, x6, x7, lsl #6
    LEAFAT  x10, 20
    LEAFAT  x11, 21
    LEAFAT  x12, 22
    LEAFAT  x13, 23
    fadd    d20, d20, d21
    fadd    d22, d22, d23
    fadd    d0, d20, d22
    b       Lpush

Locto_seq:  // 68..128 vectors: eight leaves
    OCTO_SEQ
    b       Lpush

Lquad8:     // 272..512 vectors: eight groups of four, one after the other
    lsr     x8, x7, #1
    sub     x9, x7, x8
    add     x17, x6, x8, lsl #6
    str     x17, [sp, #SAVED]           // the right half, for after the left
    str     x9, [sp, #SAVED + 8]
    mov     x7, x8
    QUAD4
    fmov    d7, d0
    ldr     x6, [sp, #SAVED]
    ldr     x7, [sp, #SAVED + 8]
    QUAD4
    fadd    d0, d7, d0
    b       Lpush

Lquad4:     // 136..256 vectors: four groups of four, one after the other
    QUAD4
    b       Lpush

Lquad2:     // 68..128 vectors: two groups of four, one after the other, so
            // that the walk does not pass between them
    lsr     x8, x7, #1
    add     x15, x6, x8, lsl #6         // the right group
    sub     x16, x7, x8
    mov     x7, x8
    QUAD
    fmov    d5, d0
    mov     x6, x15
    mov     x7, x16
    QUAD
    fadd    d0, d5, d0
    b       Lpush

Locto2_seq: // 136..256 vectors: two nodes of eight leaves, one after the other
    lsr     x8, x7, #1
    sub     x9, x7, x8                  // the right node, which starts where
    mov     x7, x8                      // the left one ends
    OCTO_SEQ
    fmov    d5, d0
    mov     x7, x9
    OCTO_SEQ
    fadd    d0, d5, d0
    add     x9, sp, #NODES              // the bottom of the node stack, borrowed
    b       Lpush

Ldone:
    ldr     d0, [x5, #-8]
    movi    d1, #0
    fadd    d0, d0, d1
    add     sp, sp, #FRAME
    ret
