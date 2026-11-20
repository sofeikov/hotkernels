// void any_cols_bool(const bool *m, bool *out, size_t n, size_t cols);
//
// x0 = m, x1 = out, x2 = n, x3 = cols (called "width" below). m holds
// n / width whole rows of width bytes; a partial row after them is ignored.
// out[c] = 1 if any whole row has a non-zero byte in column c, else 0, for
// every c < width, so a matrix with no whole row gives width zeros. Writes
// exactly width bytes. width >= 1. Any byte value is allowed in m: the columns
// are or'ed as raw bytes and every result goes through `umin` with 1 before it
// is stored.
//
// Paths, by width, chosen once at entry:
//   1..16    the whole rows as one stream of bytes, 16 per load. A load
//            starts at a multiple of width from m, so lane i of every load
//            is column i mod width: the loads step by the largest multiple
//            of width that is at most 16 and are or'ed lane by lane into
//            eight accumulators. The last 16 bytes of the rows are loaded
//            and moved down by `tbl` to the same lanes. At the end the lanes
//            of a column are folded onto the first width lanes by `tbl`.
//            Under 16 bytes of whole rows, the first and last power of two
//            bytes are loaded and placed by `tbl`.
//   17..32   every row as its first 16 bytes and its last 16, which overlap,
//            or'ed into two accumulators per row of four: four rows a step.
//   33..48   every row as its first 32 bytes and its last 16: two rows a step.
//   49..64   every row as its first 32 bytes and its last 32: two rows a step.
//   65..112  every row as its first 64 bytes and its last 48: one row a step.
//   113..128 every row as its first 64 bytes and its last 64: one row a step.
//   129..256 every row as its first 128 bytes and its last 128: one row a step.
//   257..    tiles of TILE rows; for each block of 128 columns of a tile, the
//            columns' results so far are loaded from out, the rows or'ed in,
//            and stored back; the first tile starts from zero and the last
//            stores through `umin` with 1. The last block ends at the last
//            column.
// From HUGE_N bytes (8 MB, more than the second-level cache) the loops whose
// loads take an immediate offset (widths dividing 16, 33..48, 65..112 and
// the wide path) use `ldnp` for their pairs, and the wide path's tiles are
// TILE / 2 rows. Those loops sit after the return. Widths of 17 to 32 and
// the other widths under 16 load with register offsets and keep `ldr`.

// Rows per tile of the wide path: a tile of rows of 4,096 is 64 KB, half the
// first-level data cache.
.set TILE, 16

// Rows of more than 128: one row at x13, 128 bytes of it, or'ed into v16 to
// v23; x13 moves one row. \ld is `ldp`, or `ldnp` for a matrix larger than
// the second-level cache.
.macro WIDE1 ld
    \ld     q0, q1, [x13]
    \ld     q2, q3, [x13, #32]
    \ld     q4, q5, [x13, #64]
    \ld     q6, q7, [x13, #96]
    add     x13, x13, x3
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
.endm

// From this many bytes (8 MB), more than the second-level cache holds, the
// wide path loads with `ldnp`.
.set HUGE_N, 8388608

// Eight contiguous vectors at x10 or'ed into v16 to v23; x10 moves past them.
// \ld is `ldp`, or `ldnp` above HUGE_N.
.macro NARROW16 ld
    \ld     q0, q1, [x10]
    \ld     q2, q3, [x10, #32]
    \ld     q4, q5, [x10, #64]
    \ld     q6, q7, [x10, #96]
    add     x10, x10, #128
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
.endm

// Two rows of 33 to 48 at x10 and x12: the first 32 bytes by \ld (`ldp`,
// or `ldnp` above HUGE_N) into v16, v17 and v20, v21, the last 16 at x6 into
// v18 and v22. Both move two rows.
.macro MID48 ld
    \ld     q0, q1, [x10]
    ldr     q2, [x10, x6]
    \ld     q4, q5, [x12]
    ldr     q6, [x12, x6]
    add     x10, x10, x14
    add     x12, x12, x14
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
.endm

// One row of 65 to 112 at x10: its first 64 bytes into v16 to v19, its last
// 48, at x11, into v20 to v22, the pairs by \ld (`ldp`, or `ldnp` above
// HUGE_N). Both move one row.
.macro MID112 ld
    \ld     q0, q1, [x10]
    \ld     q2, q3, [x10, #32]
    \ld     q4, q5, [x11]
    ldr     q6, [x11, #32]
    add     x10, x10, x3
    add     x11, x11, x3
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
.endm

// One row of 129 to 256: its first 128 bytes at x10 into v16 to v23, its
// last 128 at x11 into v24 to v31, by \ld (`ldp`, or `ldnp` above HUGE_N).
// Both move one row.
.macro MID256 ld
    \ld     q0, q1, [x10]
    \ld     q2, q3, [x10, #32]
    \ld     q4, q5, [x10, #64]
    \ld     q6, q7, [x10, #96]
    add     x10, x10, x3
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
    \ld     q0, q1, [x11]
    \ld     q2, q3, [x11, #32]
    \ld     q4, q5, [x11, #64]
    \ld     q6, q7, [x11, #96]
    add     x11, x11, x3
    orr     v24.16b, v24.16b, v0.16b
    orr     v25.16b, v25.16b, v1.16b
    orr     v26.16b, v26.16b, v2.16b
    orr     v27.16b, v27.16b, v3.16b
    orr     v28.16b, v28.16b, v4.16b
    orr     v29.16b, v29.16b, v5.16b
    orr     v30.16b, v30.16b, v6.16b
    orr     v31.16b, v31.16b, v7.16b
.endm

.text
.globl _any_cols_bool
.p2align 2
_any_cols_bool:
    udiv    x4, x2, x3                  // rows
    cbz     x4, Lzero
    mul     x5, x4, x3                  // bytes of whole rows
    cmp     x3, #16
    b.ls    Lnarrow
    cmp     x3, #64
    b.ls    Lmid
    cmp     x3, #112
    b.ls    Lmid112
    cmp     x3, #128
    b.ls    Lmid128
    cmp     x3, #256
    b.ls    Lmid256
    b       Lwide

// No whole row: width zeros. Over 16, 32-byte pairs, the last pair ending at
// width; over 64, 64 bytes a step and the last 64 ending at width.
Lzero:
    movi    v0.16b, #0
    cmp     x3, #16
    b.ls    Lstore16
    add     x7, x1, x3                  // the end of out
    cmp     x3, #32
    b.hi    1f
    str     q0, [x1]
    stur    q0, [x7, #-16]
    b       Ldone
1:  cmp     x3, #64
    b.hi    2f
    stp     q0, q0, [x1]
    stp     q0, q0, [x7, #-32]
    b       Ldone
2:  sub     x7, x7, #64                 // the last 64 bytes of out
    mov     x8, x1
3:  stp     q0, q0, [x8]
    stp     q0, q0, [x8, #32]
    add     x8, x8, #64
    cmp     x8, x7
    b.lo    3b
    stp     q0, q0, [x7]
    stp     q0, q0, [x7, #32]
    b       Ldone

// Stores the first width (1 to 16) bytes of v0: the largest power of two K
// at most width from the start, and K bytes ending at width from v0 moved
// down by width - K lanes.
Lstore16:
    adrp    x9, Liota@PAGE
    ldr     q31, [x9, Liota@PAGEOFF]
Lstore16_iota:                          // v31 holds Liota already
    clz     x6, x3
    mov     x7, #63
    sub     x6, x7, x6                  // log2 K
    mov     x7, #1
    lsl     x7, x7, x6                  // K
    sub     x8, x3, x7                  // width - K
    dup     v2.16b, w8
    add     v2.16b, v31.16b, v2.16b
    tbl     v1.16b, {v0.16b}, v2.16b
    add     x9, x1, x8
    cmp     x6, #2
    b.hi    1f
    b.eq    2f
    cbnz    x6, 3f
    str     b0, [x1]                    // width 1
    b       Ldone
3:  str     h0, [x1]                    // width 2, 3
    str     h1, [x9]
    b       Ldone
2:  str     s0, [x1]                    // width 4 to 7
    str     s1, [x9]
    b       Ldone
1:  cmp     x6, #4
    b.eq    4f
    str     d0, [x1]                    // width 8 to 15
    str     d1, [x9]
    b       Ldone
4:  str     q0, [x1]                    // width 16
    b       Ldone

// Rows of 1 to 16 bytes, as a stream.
Lnarrow:
    adrp    x9, Liota@PAGE
    ldr     q31, [x9, Liota@PAGEOFF]
    cmp     x5, #16
    b.lo    Ltiny
    adrp    x6, Lstep@PAGE
    add     x6, x6, Lstep@PAGEOFF
    ldrb    w7, [x6, x3]                // L: step of one load
    add     x8, x0, x5                  // end of the whole rows
    sub     x9, x8, #16                 // the last load that fits
    movi    v16.16b, #0
    lsl     x16, x7, #3
    sub     x16, x16, x7                // 7 L
    sub     x17, x9, x16                // the last x10 whose eight loads fit
    cmp     x0, x17
    b.hi    Lnarrow_short               // fewer than eight loads in all
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    mov     x10, x0                     // loads at x10 + 0, 1, 2, 3 L
    lsl     x11, x7, #1                 // 2 L
    add     x12, x11, x7                // 3 L
    lsl     x13, x7, #2                 // 4 L
    add     x14, x10, x13               // loads at x14 + 0, 1, 2, 3 L
    lsl     x15, x7, #3                 // 8 L: one step
    cmp     x7, #16
    b.eq    Lnarrow16
Lnarrow_loop:
    ldr     q0, [x10]
    ldr     q1, [x10, x7]
    ldr     q2, [x10, x11]
    ldr     q3, [x10, x12]
    ldr     q4, [x14]
    ldr     q5, [x14, x7]
    ldr     q6, [x14, x11]
    ldr     q7, [x14, x12]
    add     x10, x10, x15
    add     x14, x14, x15
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
    cmp     x10, x17
    b.ls    Lnarrow_loop
Lnarrow_one:
    cmp     x10, x9
    b.hi    Lnarrow_last
1:  ldr     q0, [x10]
    add     x10, x10, x7
    orr     v16.16b, v16.16b, v0.16b
    cmp     x10, x9
    b.ls    1b
Lnarrow_last:
    // The last 16 bytes: lane i is column (i - (16 - L)) mod width, so lane
    // j + 16 - L moves to lane j. The lanes it drops were loaded before.
    ldr     q0, [x9]
    mov     x6, #16
    sub     x6, x6, x7
    dup     v1.16b, w6
    add     v1.16b, v31.16b, v1.16b
    tbl     v0.16b, {v0.16b}, v1.16b
    orr     v16.16b, v16.16b, v17.16b
    orr     v18.16b, v18.16b, v19.16b
    orr     v20.16b, v20.16b, v21.16b
    orr     v22.16b, v22.16b, v23.16b
    orr     v16.16b, v16.16b, v18.16b
    orr     v20.16b, v20.16b, v22.16b
    orr     v16.16b, v16.16b, v0.16b
    orr     v16.16b, v16.16b, v20.16b
// v16: lane i is column i mod width. Lanes i + width, i + 2 width, ... are
// or'ed onto lane i by shifts of width, 2 width, 4 width, 8 width.
Lnarrow_fold:
    mov     x6, x3
1:  dup     v1.16b, w6
    add     v1.16b, v31.16b, v1.16b
    tbl     v2.16b, {v16.16b}, v1.16b
    orr     v16.16b, v16.16b, v2.16b
    lsl     x6, x6, #1
    cmp     x6, #16
    b.lo    1b
    movi    v1.16b, #1
    umin    v0.16b, v16.16b, v1.16b
    b       Lstore16_iota

// Under 16 bytes of whole rows (width under 16). K, the largest power of
// two at most that count N, bytes from the start go to lanes 0 to K - 1 and
// K bytes ending at N to lanes 8 to 8 + K - 1; `tbl` puts byte j in lane j
// for j < N and clears the lanes from N up.
Ltiny:
    clz     x6, x5
    mov     x7, #63
    sub     x6, x7, x6                  // log2 K
    mov     x7, #1
    lsl     x7, x7, x6                  // K
    sub     x8, x5, x7                  // N - K
    cmp     x6, #2
    b.hi    3f
    b.eq    2f
    cbnz    x6, 1f
    ldrb    w9, [x0]
    ldrb    w10, [x0, x8]
    b       4f
1:  ldrh    w9, [x0]
    ldrh    w10, [x0, x8]
    b       4f
2:  ldr     w9, [x0]
    ldr     w10, [x0, x8]
    b       4f
3:  ldr     x9, [x0]
    ldr     x10, [x0, x8]
4:  fmov    d0, x9
    mov     v0.d[1], x10
    dup     v1.16b, w7
    cmhs    v2.16b, v31.16b, v1.16b     // lanes from K
    mov     x11, #8
    add     x11, x11, x7
    sub     x11, x11, x5                // 8 + K - N
    dup     v3.16b, w11
    and     v3.16b, v3.16b, v2.16b
    add     v1.16b, v31.16b, v3.16b
    dup     v4.16b, w5
    cmhs    v4.16b, v31.16b, v4.16b     // lanes from N: index out of range
    orr     v1.16b, v1.16b, v4.16b
    tbl     v16.16b, {v0.16b}, v1.16b
    b       Lnarrow_fold

// Rows of 17 to 64.
Lmid:
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    cmp     x3, #32
    b.hi    Lmid_over32
    // Rows of 17 to 32: the first 16 bytes into v16, v18, v20, v22, the last
    // 16 into v17, v19, v21, v23, by the row's place in a step of four.
    sub     x6, x3, #16                 // where a row's last 16 bytes start
    mov     x10, x0
    add     x11, x3, x3                 // 2 width
    add     x12, x11, x3                // 3 width
    lsl     x13, x3, #2                 // 4 width: one step
    add     x14, x6, x3
    add     x15, x6, x11
    add     x16, x6, x12
    subs    x9, x4, #4
    b.lo    Lmid32_rest
Lmid32_loop:
    ldr     q0, [x10]
    ldr     q1, [x10, x6]
    ldr     q2, [x10, x3]
    ldr     q3, [x10, x14]
    ldr     q4, [x10, x11]
    ldr     q5, [x10, x15]
    ldr     q6, [x10, x12]
    ldr     q7, [x10, x16]
    add     x10, x10, x13
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
    subs    x9, x9, #4
    b.hs    Lmid32_loop
Lmid32_rest:
    adds    x9, x9, #4
    b.eq    Lmid32_end
1:  ldr     q0, [x10]
    ldr     q1, [x10, x6]
    add     x10, x10, x3
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    subs    x9, x9, #1
    b.ne    1b
Lmid32_end:
    orr     v16.16b, v16.16b, v18.16b
    orr     v17.16b, v17.16b, v19.16b
    orr     v20.16b, v20.16b, v22.16b
    orr     v21.16b, v21.16b, v23.16b
    orr     v16.16b, v16.16b, v20.16b
    orr     v17.16b, v17.16b, v21.16b
    movi    v1.16b, #1
    umin    v16.16b, v16.16b, v1.16b
    umin    v17.16b, v17.16b, v1.16b
    str     q17, [x1, x6]
    str     q16, [x1]
    b       Ldone

// Rows of 33 to 48: the first 32 bytes into v16, v17 (v20, v21 for the second
// row of a step), the last 16 into v18 (v22).
Lmid_over32:
    cmp     x3, #48
    b.hi    Lmid64
Lmid48:
    sub     x6, x3, #16                 // where a row's last 16 bytes start
    mov     x10, x0
    add     x12, x0, x3
    lsl     x14, x3, #1                 // 2 width: one step
    subs    x9, x4, #2
    b.lo    Lmid48_rest
    mov     x7, #HUGE_N
    cmp     x2, x7
    b.hs    Lmid48_nt
1:  MID48   ldp
    subs    x9, x9, #2
    b.hs    1b
Lmid48_rest:
    adds    x9, x9, #2
    b.eq    3f
    ldp     q0, q1, [x10]
    ldr     q2, [x10, x6]
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
3:  orr     v16.16b, v16.16b, v20.16b
    orr     v17.16b, v17.16b, v21.16b
    orr     v18.16b, v18.16b, v22.16b
    movi    v1.16b, #1
    umin    v16.16b, v16.16b, v1.16b
    umin    v17.16b, v17.16b, v1.16b
    umin    v18.16b, v18.16b, v1.16b
    str     q18, [x1, x6]
    stp     q16, q17, [x1]
    b       Ldone

// Rows of 49 to 64: the first 32 bytes into v16, v17 (v20, v21 for the second
// row of a step), the last 32 into v18, v19 (v22, v23).
Lmid64:
    sub     x6, x3, #32                 // where a row's last 32 bytes start
    mov     x10, x0
    add     x11, x0, x6
    add     x12, x10, x3
    add     x13, x11, x3
    lsl     x14, x3, #1                 // 2 width: one step
    subs    x9, x4, #2
    b.lo    Lmid64_rest
Lmid64_loop:
    ldp     q0, q1, [x10]
    ldp     q2, q3, [x11]
    ldp     q4, q5, [x12]
    ldp     q6, q7, [x13]
    add     x10, x10, x14
    add     x11, x11, x14
    add     x12, x12, x14
    add     x13, x13, x14
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
    subs    x9, x9, #2
    b.hs    Lmid64_loop
Lmid64_rest:
    adds    x9, x9, #2
    b.eq    Lmid64_end
    ldp     q0, q1, [x10]
    ldp     q2, q3, [x11]
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
Lmid64_end:
    orr     v16.16b, v16.16b, v20.16b
    orr     v17.16b, v17.16b, v21.16b
    orr     v18.16b, v18.16b, v22.16b
    orr     v19.16b, v19.16b, v23.16b
    movi    v1.16b, #1
    umin    v16.16b, v16.16b, v1.16b
    umin    v17.16b, v17.16b, v1.16b
    umin    v18.16b, v18.16b, v1.16b
    umin    v19.16b, v19.16b, v1.16b
    add     x7, x1, x6
    stp     q18, q19, [x7]
    stp     q16, q17, [x1]
    b       Ldone

// Rows of 65 to 112: every row as its first 64 bytes, into v16 to v19, and
// its last 48, into v20 to v22, one row a step.
Lmid112:
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    sub     x6, x3, #48                 // where a row's last 48 bytes start
    mov     x10, x0
    add     x11, x0, x6
    mov     x7, #HUGE_N
    cmp     x2, x7
    b.hs    Lmid112_nt
1:  MID112  ldp
    subs    x4, x4, #1
    b.ne    1b
Lmid112_end:
    movi    v1.16b, #1
    umin    v16.16b, v16.16b, v1.16b
    umin    v17.16b, v17.16b, v1.16b
    umin    v18.16b, v18.16b, v1.16b
    umin    v19.16b, v19.16b, v1.16b
    umin    v20.16b, v20.16b, v1.16b
    umin    v21.16b, v21.16b, v1.16b
    umin    v22.16b, v22.16b, v1.16b
    add     x7, x1, x6
    stp     q20, q21, [x7]
    str     q22, [x7, #32]
    stp     q16, q17, [x1]
    stp     q18, q19, [x1, #32]
    b       Ldone

// Rows of 113 to 128: every row as its first 64 bytes, into v16 to v19, and
// its last 64, into v20 to v23, one row a step; nothing is stored until the
// end.
Lmid128:
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    sub     x6, x3, #64                 // where a row's last 64 bytes start
    mov     x10, x0
    add     x11, x0, x6
1:  ldp     q0, q1, [x10]
    ldp     q2, q3, [x10, #32]
    ldp     q4, q5, [x11]
    ldp     q6, q7, [x11, #32]
    add     x10, x10, x3
    add     x11, x11, x3
    orr     v16.16b, v16.16b, v0.16b
    orr     v17.16b, v17.16b, v1.16b
    orr     v18.16b, v18.16b, v2.16b
    orr     v19.16b, v19.16b, v3.16b
    orr     v20.16b, v20.16b, v4.16b
    orr     v21.16b, v21.16b, v5.16b
    orr     v22.16b, v22.16b, v6.16b
    orr     v23.16b, v23.16b, v7.16b
    subs    x4, x4, #1
    b.ne    1b
    movi    v1.16b, #1
    umin    v16.16b, v16.16b, v1.16b
    umin    v17.16b, v17.16b, v1.16b
    umin    v18.16b, v18.16b, v1.16b
    umin    v19.16b, v19.16b, v1.16b
    umin    v20.16b, v20.16b, v1.16b
    umin    v21.16b, v21.16b, v1.16b
    umin    v22.16b, v22.16b, v1.16b
    umin    v23.16b, v23.16b, v1.16b
    add     x7, x1, x6
    stp     q20, q21, [x7]
    stp     q22, q23, [x7, #32]
    stp     q16, q17, [x1]
    stp     q18, q19, [x1, #32]
    b       Ldone

// Rows of more than 256, in blocks of 128 columns. The first tile starts
// from zero and the last one stores its blocks through `umin` with 1: no pass
// over out of their own.
Lwide:
    sub     x6, x3, #128                // the last block's first column
    mov     x7, x0                      // the tile's first row
    mov     x8, x4                      // rows left
    movi    v24.16b, #1
    mov     x17, #HUGE_N
    cmp     x2, x17
    cset    x17, hs                     // non-temporal loads
Ltile:
    mov     x9, #TILE
    lsr     x9, x9, x17                 // half a tile above HUGE_N
    cmp     x8, x9
    csel    x9, x8, x9, lo              // rows in this tile
    mov     x10, #0                     // the block's first column, unclamped
Lblock:
    cmp     x10, x6
    csel    x11, x10, x6, lo            // the block's first column
    add     x12, x1, x11
    cmp     x7, x0
    b.eq    0f                          // the first tile
    ldp     q16, q17, [x12]
    ldp     q18, q19, [x12, #32]
    ldp     q20, q21, [x12, #64]
    ldp     q22, q23, [x12, #96]
    b       1f
0:  movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
1:  add     x13, x7, x11                // the tile's first row at the block
    mov     x14, x9
    cbnz    x17, Lwide_nt
2:  WIDE1   ldp
    subs    x14, x14, #1
    b.ne    2b
Lwide_rest:
    cmp     x8, x9
    b.ne    4f                          // not the last tile
    umin    v16.16b, v16.16b, v24.16b
    umin    v17.16b, v17.16b, v24.16b
    umin    v18.16b, v18.16b, v24.16b
    umin    v19.16b, v19.16b, v24.16b
    umin    v20.16b, v20.16b, v24.16b
    umin    v21.16b, v21.16b, v24.16b
    umin    v22.16b, v22.16b, v24.16b
    umin    v23.16b, v23.16b, v24.16b
4:  stp     q16, q17, [x12]
    stp     q18, q19, [x12, #32]
    stp     q20, q21, [x12, #64]
    stp     q22, q23, [x12, #96]
    add     x10, x10, #128
    add     x11, x11, #128
    cmp     x11, x3
    b.lo    Lblock
    madd    x7, x9, x3, x7              // the next tile
    subs    x8, x8, x9
    b.ne    Ltile

Ldone:
    ret

// The row pairs of the wide path with non-temporal loads, out of line.
Lwide_nt:
    WIDE1   ldnp
    subs    x14, x14, #1
    b.ne    Lwide_nt
    b       Lwide_rest

// The loop of rows of 65 to 112 with non-temporal loads, above HUGE_N.
Lmid112_nt:
    MID112  ldnp
    subs    x4, x4, #1
    b.ne    Lmid112_nt
    b       Lmid112_end

// The loop of rows of 33 to 48 with non-temporal loads, above HUGE_N.
Lmid48_nt:
    MID48   ldnp
    subs    x9, x9, #2
    b.hs    Lmid48_nt
    b       Lmid48_rest

// The loop of the narrow path when the step is 16 (widths 1, 2, 4, 8, 16):
// the loads are contiguous, four pairs a step.
Lnarrow16:
    mov     x6, #HUGE_N
    cmp     x2, x6
    b.hs    Lnarrow16_nt
Lnarrow16_loop:
    NARROW16 ldp
    cmp     x10, x17
    b.ls    Lnarrow16_loop
    b       Lnarrow_one
Lnarrow16_nt:
    NARROW16 ldnp
    cmp     x10, x17
    b.ls    Lnarrow16_nt
    b       Lnarrow_one

// Rows of 129 to 256: every row as its first 128 bytes, into v16 to v23, and
// its last 128, into v24 to v31, one row a step; nothing is stored until the
// end. Out of line, after the other paths.
Lmid256:
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0
    movi    v28.16b, #0
    movi    v29.16b, #0
    movi    v30.16b, #0
    movi    v31.16b, #0
    sub     x6, x3, #128                // where a row's last 128 bytes start
    mov     x10, x0
    add     x11, x0, x6
    mov     x7, #HUGE_N
    cmp     x2, x7
    b.hs    Lmid256_nt
1:  MID256  ldp
    subs    x4, x4, #1
    b.ne    1b
Lmid256_end:
    movi    v0.16b, #1
    umin    v16.16b, v16.16b, v0.16b
    umin    v17.16b, v17.16b, v0.16b
    umin    v18.16b, v18.16b, v0.16b
    umin    v19.16b, v19.16b, v0.16b
    umin    v20.16b, v20.16b, v0.16b
    umin    v21.16b, v21.16b, v0.16b
    umin    v22.16b, v22.16b, v0.16b
    umin    v23.16b, v23.16b, v0.16b
    umin    v24.16b, v24.16b, v0.16b
    umin    v25.16b, v25.16b, v0.16b
    umin    v26.16b, v26.16b, v0.16b
    umin    v27.16b, v27.16b, v0.16b
    umin    v28.16b, v28.16b, v0.16b
    umin    v29.16b, v29.16b, v0.16b
    umin    v30.16b, v30.16b, v0.16b
    umin    v31.16b, v31.16b, v0.16b
    add     x7, x1, x6
    stp     q24, q25, [x7]
    stp     q26, q27, [x7, #32]
    stp     q28, q29, [x7, #64]
    stp     q30, q31, [x7, #96]
    stp     q16, q17, [x1]
    stp     q18, q19, [x1, #32]
    stp     q20, q21, [x1, #64]
    stp     q22, q23, [x1, #96]
    b       Ldone

// The loop of rows of 129 to 256 with non-temporal loads, above HUGE_N.
Lmid256_nt:
    MID256  ldnp
    subs    x4, x4, #1
    b.ne    Lmid256_nt
    b       Lmid256_end

// Rows of 1 to 16 when the whole rows take fewer than eight loads: one
// accumulator, v16, a load at a time, then the last 16 bytes as in
// Lnarrow_last.
Lnarrow_short:
    mov     x10, x0
1:  ldr     q0, [x10]
    add     x10, x10, x7
    orr     v16.16b, v16.16b, v0.16b
    cmp     x10, x9
    b.ls    1b
    ldr     q0, [x9]
    mov     x6, #16
    sub     x6, x6, x7
    dup     v1.16b, w6
    add     v1.16b, v31.16b, v1.16b
    tbl     v0.16b, {v0.16b}, v1.16b
    orr     v16.16b, v16.16b, v0.16b
    b       Lnarrow_fold

.section __TEXT,__const
.p2align 4
// 0 to 15: the lane numbers, from which `tbl` indices are made.
Liota:
    .byte 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
// By width from 0 to 16: the largest multiple of the width that is at most
// 16, the step of the narrow path's loads. Width 0 is never used.
Lstep:
    .byte 0, 16, 16, 15, 16, 15, 12, 14, 16, 9, 10, 11, 12, 13, 14, 15, 16
