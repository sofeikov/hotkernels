// float sum_f32(const float *x, size_t n);
//
// x0 = x, x1 = n. Returns np.sum(x), NumPy's bits: the same additions in
// NumPy's pairwise order (see spec.py). Baseline: the tree walked one leaf
// at a time.

// Bytes of one vector of eight elements, the unit the tree is cut in.
.set VEC, 32
// Bytes below sp for the walk: the node stack (16 bytes a node) in the
// first NODE_BYTES, the stack of sums (4 bytes each) after it.
.set NODE_BYTES, 2048
.set FRAME, 2560

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
    mov     x4, sp                      // node stack, next free entry
    mov     x9, sp                      // node stack, bottom
    add     x5, sp, #NODE_BYTES         // stack of sums, next free entry
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
    cmp     x7, #16
    b.hi    Lsplit
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
Ldone:
    ldr     s0, [x5, #-4]
    movi    d1, #0
    fadd    s0, s0, s1
    add     sp, sp, #FRAME
    ret
