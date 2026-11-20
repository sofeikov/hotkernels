// size_t nonzero_2d_bool(const bool *m, int64_t *out, size_t n, size_t width);
// Writes the row and column of every true byte of m (rows of width) to out as
// int64 pairs, in C order. Returns how many int64 it wrote.
//
// Two ways of handling a stretch of the mask, chosen by how many bytes are true.
//
// Sparse (64 bytes per step, any width): the 64 bytes are packed into a 64-bit
// word, first byte in the top bit, and the true ones are counted. If at most 6
// are true, the positions of the first 6 set bits are extracted one after
// another with `clz`, with no branches; if 7 to 12, a second copy extracts 12. Each position f becomes a row and a
// column without a division: row = high half of (2f * M), M = ceil(2^63 / w),
// exact while f * w < 2^63, and column = f - row * w. Positions past the true
// count are rubbish that the next store overwrites (`out` is scratch). More
// than 12 true bytes send those 64 bytes to the dense code.
//
// Dense (a row at a time): chunks of 8 bytes and a last chunk of what is left
// of the row, so that no chunk crosses a row; in the last chunk the bytes past
// the row are masked off. One multiplication packs a chunk into an 8-bit
// number, another gives the true count (times 8). The 8-bit number indexes a
// table whose entry lists the positions of the true bytes, in order, as eight
// bytes; those are widened to 64 bits, added to the chunk's first column,
// paired with the row by `zip`, and stored: eight (row, column) pairs at the
// write pointer, of which the first `count` are kept.
//
// Narrow rows (dense, width under 8): chunks of as many whole rows as fit in
// 8 bytes, so a chunk always starts a row. The positions of the true bytes
// index two small tables, built for the width, of the row within the chunk
// and the column; the two are interleaved byte by byte, widened to 64 bits
// with the chunk's first row added, and stored as eight pairs.
//
// After a dense stretch, which ends at the end of a row, the share of true
// bytes in it decides what handles the next bytes: sparse if at most one in
// eight was true, else a longer dense stretch.
//
// Stores stay inside the capacity: a step starting at element f writes at
// most 8 pairs and is taken only when f + 8 <= n, and at most f pairs have
// been kept before it. The last 0 to 7 bytes of the mask, when a chunk cannot
// be read there, are handled one at a time. The sparse code runs only when
// n < 2^31, which keeps f * w below 2^63 with w taken as at most n.

// Widen the 8 positions in d0, add the column in v17, pair each with the row
// in v20 and store the 8 pairs at x5. Uses v0 to v6 and v21 to v27.
.macro PAIRS
    ushll   v0.8h, v0.8b, #0
    ushll   v1.4s, v0.4h, #0
    ushll2  v2.4s, v0.8h, #0
    uaddw   v21.2d, v17.2d, v1.2s
    uaddw2  v23.2d, v17.2d, v1.4s
    uaddw   v25.2d, v17.2d, v2.2s
    uaddw2  v27.2d, v17.2d, v2.4s
    zip1    v3.2d, v20.2d, v21.2d
    zip2    v4.2d, v20.2d, v21.2d
    zip1    v5.2d, v20.2d, v23.2d
    zip2    v6.2d, v20.2d, v23.2d
    stp     q3, q4, [x5]
    stp     q5, q6, [x5, #32]
    zip1    v3.2d, v20.2d, v25.2d
    zip2    v4.2d, v20.2d, v25.2d
    zip1    v5.2d, v20.2d, v27.2d
    zip2    v6.2d, v20.2d, v27.2d
    stp     q3, q4, [x5, #64]
    stp     q5, q6, [x5, #96]
.endm

// Sparse step: take the first set bit of x14 (bit 63 - j for byte j), clear
// it, and store its (row, column) pair at [x5, #off]. x7 = 2 * the position
// of the block's first byte, x6 = M, x15 = 1 << 63, x3 = width. \a and \b are
// scratch registers.
.macro EXTRACT a, b, off
    clz     \a, x14
    lsr     \b, x15, \a
    bic     x14, x14, \b
    add     \a, x7, \a, lsl #1          // 2f
    umulh   \b, \a, x6                  // row
    lsr     \a, \a, #1                  // f
    msub    \a, \b, x3, \a              // column
    stp     \b, \a, [x5, #\off]
.endm

.text
.globl _nonzero_2d_bool
.p2align 2
_nonzero_2d_bool:
    mov     x5, x1                      // x5 = write pointer
    mov     x6, #0                      // x6 = row
    mov     x17, x0                     // x17 = start of the row
    add     x4, x0, x2                  // x4 = end of the mask
    cmp     x2, #8
    b.lo    Ltail
    adrp    x9, Lpositions@PAGE
    add     x9, x9, Lpositions@PAGEOFF
    mov     x10, #0x4080                // 0x0102040810204080: gathers the low
    movk    x10, #0x1020, lsl #16       // bit of each byte into the top byte
    movk    x10, #0x0408, lsl #32
    movk    x10, #0x0102, lsl #48
    mov     x11, #0x0808080808080808    // sums the bytes, times 8, into the top byte
    mov     w8, #8
    dup     v18.2d, x8                  // 8, twice: the column step of a chunk
    cmp     x3, x2
    csel    x3, x3, x2, lo              // a width above n is as good as n
    cmp     x3, #8
    b.hs    Lwide_setup
    adrp    x8, Lnarrow_tables@PAGE     // rows and columns of 8 positions
    add     x8, x8, Lnarrow_tables@PAGEOFF
    add     x8, x8, x3, lsl #4
    ldp     d29, d30, [x8, #-16]        // v29 = rows, v30 = columns
    adrp    x8, Lnarrow_rows@PAGE
    add     x8, x8, Lnarrow_rows@PAGEOFF
    ldrb    w8, [x8, x3]                // whole rows per chunk
    movi    v31.2d, #0
    mov     v31.d[0], x8                // v31 = (rows per chunk, 0)
Lwide_setup:
    mov     x7, x4                      // dense to the end, unless sparse is used
    sub     x2, x4, #8                  // x2 = last place a chunk can start
    sub     x8, x4, x0
    cmp     x8, #64
    b.lo    Lrow
    lsr     x8, x8, #31
    cbnz    x8, Lrow                    // n >= 2^31: dense only
    mov     x8, #0x7fffffffffffffff
    udiv    x8, x8, x3
    add     x8, x8, #1                  // M = ceil(2^63 / width)
    fmov    d19, x8
    mov     v19.d[1], x0                // v19 = (M, start of the mask)
    adrp    x8, Lbit_shifts@PAGE
    ldr     q28, [x8, Lbit_shifts@PAGEOFF]
    b       Lsparse_enter
Lsparse_enter:                          // x0 = the next byte
    fmov    x6, d19
    mov     x15, #0x8000000000000000
    mov     x7, v19.d[1]
    sub     x7, x0, x7
    lsl     x7, x7, #1                  // x7 = 2 * its position
Lsparse:
    sub     x8, x4, x0
    cmp     x8, #64
    b.lo    Lsparse_exit
    ldp     q0, q1, [x0]
    ldp     q2, q3, [x0, #32]
    ushl    v0.16b, v0.16b, v28.16b     // byte j becomes bit (j mod 8)
    ushl    v1.16b, v1.16b, v28.16b
    ushl    v2.16b, v2.16b, v28.16b
    ushl    v3.16b, v3.16b, v28.16b
    addp    v0.16b, v0.16b, v1.16b      // three rounds of pairwise sums
    addp    v2.16b, v2.16b, v3.16b      // leave one byte per 8 mask bytes
    addp    v0.16b, v0.16b, v2.16b
    addp    v0.16b, v0.16b, v0.16b
    cnt     v1.8b, v0.8b
    fmov    x14, d0                     // bit j = mask byte j
    addv    b1, v1.8b
    rbit    x14, x14                    // bit 63-j = mask byte j
    fmov    w13, s1                     // number of true bytes
    cmp     w13, #6
    b.hi    Lsparse_tier
    EXTRACT x8, x12, 0
    EXTRACT x16, x17, 16
    EXTRACT x8, x12, 32
    EXTRACT x16, x17, 48
    EXTRACT x8, x12, 64
    EXTRACT x16, x17, 80
Lsparse_next:
    add     x5, x5, x13, lsl #4
    add     x0, x0, #64
    add     x7, x7, #128
    b       Lsparse
Lsparse_more:                           // 7 to 12 true bytes
    EXTRACT x8, x12, 0
    EXTRACT x16, x17, 16
    EXTRACT x8, x12, 32
    EXTRACT x16, x17, 48
    EXTRACT x8, x12, 64
    EXTRACT x16, x17, 80
    EXTRACT x8, x12, 96
    EXTRACT x16, x17, 112
    EXTRACT x8, x12, 128
    EXTRACT x16, x17, 144
    EXTRACT x8, x12, 160
    EXTRACT x16, x17, 176
    b       Lsparse_next
Lsparse_tier:
    cmp     w13, #12
    b.ls    Lsparse_more
Lsparse_fail:                           // a dense stretch of at least 64 bytes
    add     x13, x0, #64
    b       Lsparse_leave
Lsparse_exit:                           // fewer than 64 bytes left: dense to the end
    mov     x13, x4
Lsparse_leave:                          // row and start of row of x0, from 2f in x7
    umulh   x8, x7, x6
    lsr     x12, x7, #1
    msub    x12, x8, x3, x12            // column
    mov     x6, x8
    sub     x17, x0, x12
    mov     x7, x13                     // x7 = end of the dense stretch
Lstretch:
    sub     x16, x5, x0, lsl #1         // x16 = 16 * true - 2 * bytes, offset
Lrow:                                   // x0 in row x6, which starts at x17
    cmp     x3, #8
    b.hs    Lrow_wide
    cmp     x0, x17
    b.eq    Lnarrow
Lrow_wide:
    add     x12, x17, x3
    cmp     x12, x4
    csel    x12, x12, x4, lo            // x12 = end of the row, or of the mask
    sub     x13, x12, #8                // x13 = last place a whole chunk can start
    dup     v20.2d, x6                  // the row, twice
    sub     x14, x0, x17
    dup     v17.2d, x14                 // column of the chunk
    cmp     x0, x13
    b.hi    Lpart
Lfull:                                  // 8 bytes, all in this row
    ldr     x14, [x0], #8
    mul     x15, x14, x10
    mul     x8, x14, x11
    lsr     x15, x15, #56               // 8-bit mask
    lsr     x8, x8, #56                 // 8 * (number of true bytes)
    ldr     d0, [x9, x15, lsl #3]       // positions of the true bytes
    PAIRS
    add     x5, x5, x8, lsl #1
    add     v17.2d, v17.2d, v18.2d
    cmp     x0, x13
    b.ls    Lfull
Lpart:                                  // 0 to 7 bytes left in the row
    cmp     x0, x12
    b.eq    Lrow_end
    cmp     x0, x2
    b.hi    Ltail                       // a chunk cannot be read here
    sub     x12, x12, x0                // length, 1 to 7
    ldr     x14, [x0]
    add     x0, x0, x12
    neg     x15, x12, lsl #3            // 64 - 8 * length
    mov     x8, #-1
    lsr     x15, x8, x15                // the row's bytes of the word
    and     x14, x14, x15
    mul     x15, x14, x10
    mul     x8, x14, x11
    lsr     x15, x15, #56
    lsr     x8, x8, #56
    ldr     d0, [x9, x15, lsl #3]
    PAIRS
    add     x5, x5, x8, lsl #1
Lrow_end:
    add     x6, x6, #1
    mov     x17, x0
    cmp     x0, x7
    b.lo    Lrow
Lstretch_end:                           // x0 >= x7, at the start of a row
    cmp     x0, x4
    b.hs    Ldone
    sub     x8, x5, x0, lsl #1          // was at most one byte in eight true?
    subs    x8, x8, x16
    b.le    Lsparse_enter
    add     x7, x0, #512                // else a longer dense stretch
    cmp     x7, x4
    csel    x7, x7, x4, lo
    b       Lstretch
Lnarrow:                                // x0 = x17 starts row x6; width under 8
    umov    w12, v31.b[0]
    mul     x12, x12, x3                // x12 = bytes per chunk
    neg     x17, x12, lsl #3
    mov     x13, #-1
    lsr     x17, x13, x17               // x17 = those bytes of a word
    add     x13, x2, #1
    cmp     x13, x7
    csel    x13, x13, x7, lo            // x13 = stop before here
    cmp     x0, x13
    b.hs    Lnarrow_done
    movi    v16.2d, #0
    mov     v16.d[0], x6                // v16 = (row, 0)
Lnarrow_loop:
    ldr     x14, [x0]
    add     x0, x0, x12
    and     x14, x14, x17
    mul     x15, x14, x10
    mul     x8, x14, x11
    lsr     x15, x15, #56               // 8-bit mask
    lsr     x8, x8, #56                 // 8 * (number of true bytes)
    ldr     d0, [x9, x15, lsl #3]       // positions of the true bytes
    tbl     v1.8b, {v29.16b}, v0.8b     // their rows within the chunk
    tbl     v2.8b, {v30.16b}, v0.8b     // and columns
    zip1    v1.16b, v1.16b, v2.16b      // row, column, row, column, ...
    ushll   v2.8h, v1.8b, #0
    ushll2  v3.8h, v1.16b, #0
    ushll   v4.4s, v2.4h, #0
    ushll2  v5.4s, v2.8h, #0
    ushll   v6.4s, v3.4h, #0
    ushll2  v7.4s, v3.8h, #0
    uaddw   v21.2d, v16.2d, v4.2s
    uaddw2  v22.2d, v16.2d, v4.4s
    uaddw   v23.2d, v16.2d, v5.2s
    uaddw2  v24.2d, v16.2d, v5.4s
    stp     q21, q22, [x5]
    stp     q23, q24, [x5, #32]
    uaddw   v25.2d, v16.2d, v6.2s
    uaddw2  v26.2d, v16.2d, v6.4s
    uaddw   v27.2d, v16.2d, v7.2s
    uaddw2  v0.2d, v16.2d, v7.4s
    stp     q25, q26, [x5, #64]
    stp     q27, q0, [x5, #96]
    add     x5, x5, x8, lsl #1
    add     v16.2d, v16.2d, v31.2d
    cmp     x0, x13
    b.lo    Lnarrow_loop
    fmov    x6, d16
Lnarrow_done:
    mov     x17, x0
    cmp     x0, x7
    b.hs    Lstretch_end
    b       Lrow_wide                   // fewer than 8 bytes left
Ltail:                                  // 1 to 7 bytes left in the mask
    sub     x7, x0, x17                 // column
    cmp     x0, x4
    b.hs    Ldone
Ltail_loop:
    ldrb    w8, [x0], #1
    stp     x6, x7, [x5]
    add     x5, x5, x8, lsl #4
    add     x7, x7, #1
    cmp     x7, x3
    csel    x7, x7, xzr, ne
    cinc    x6, x6, eq
    cmp     x0, x4
    b.lo    Ltail_loop
Ldone:
    sub     x0, x5, x1
    lsr     x0, x0, #3
    ret

.section __TEXT,__const
.p2align 4
// For each width w from 1 to 7: the row and the column, within a chunk of
// whole rows, of each of the chunk's 8 bytes. Bytes past the chunk are 0.
Lnarrow_tables:
    .byte 0,1,2,3,4,5,6,7  // width 1: rows
    .byte 0,0,0,0,0,0,0,0  //          columns
    .byte 0,0,1,1,2,2,3,3  // width 2: rows
    .byte 0,1,0,1,0,1,0,1  //          columns
    .byte 0,0,0,1,1,1,0,0  // width 3: rows
    .byte 0,1,2,0,1,2,0,0  //          columns
    .byte 0,0,0,0,1,1,1,1  // width 4: rows
    .byte 0,1,2,3,0,1,2,3  //          columns
    .byte 0,0,0,0,0,0,0,0  // width 5: rows
    .byte 0,1,2,3,4,0,0,0  //          columns
    .byte 0,0,0,0,0,0,0,0  // width 6: rows
    .byte 0,1,2,3,4,5,0,0  //          columns
    .byte 0,0,0,0,0,0,0,0  // width 7: rows
    .byte 0,1,2,3,4,5,6,0  //          columns
// Whole rows of width w in 8 bytes, for w from 0 (unused) to 7.
Lnarrow_rows:
    .byte 0,8,4,2,2,1,1,1
.p2align 4
// Left-shift amounts that give each of 8 consecutive bytes its own bit.
Lbit_shifts:
    .byte 0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7
// For each 8-bit mask (bit j is mask byte j): the positions of its set bits,
// in ascending order, one byte each. Unused bytes are zero. The comment shows
// mask bytes 0 to 7.
Lpositions:
    .byte 0,0,0,0,0,0,0,0  // 00000000
    .byte 0,0,0,0,0,0,0,0  // 10000000
    .byte 1,0,0,0,0,0,0,0  // 01000000
    .byte 0,1,0,0,0,0,0,0  // 11000000
    .byte 2,0,0,0,0,0,0,0  // 00100000
    .byte 0,2,0,0,0,0,0,0  // 10100000
    .byte 1,2,0,0,0,0,0,0  // 01100000
    .byte 0,1,2,0,0,0,0,0  // 11100000
    .byte 3,0,0,0,0,0,0,0  // 00010000
    .byte 0,3,0,0,0,0,0,0  // 10010000
    .byte 1,3,0,0,0,0,0,0  // 01010000
    .byte 0,1,3,0,0,0,0,0  // 11010000
    .byte 2,3,0,0,0,0,0,0  // 00110000
    .byte 0,2,3,0,0,0,0,0  // 10110000
    .byte 1,2,3,0,0,0,0,0  // 01110000
    .byte 0,1,2,3,0,0,0,0  // 11110000
    .byte 4,0,0,0,0,0,0,0  // 00001000
    .byte 0,4,0,0,0,0,0,0  // 10001000
    .byte 1,4,0,0,0,0,0,0  // 01001000
    .byte 0,1,4,0,0,0,0,0  // 11001000
    .byte 2,4,0,0,0,0,0,0  // 00101000
    .byte 0,2,4,0,0,0,0,0  // 10101000
    .byte 1,2,4,0,0,0,0,0  // 01101000
    .byte 0,1,2,4,0,0,0,0  // 11101000
    .byte 3,4,0,0,0,0,0,0  // 00011000
    .byte 0,3,4,0,0,0,0,0  // 10011000
    .byte 1,3,4,0,0,0,0,0  // 01011000
    .byte 0,1,3,4,0,0,0,0  // 11011000
    .byte 2,3,4,0,0,0,0,0  // 00111000
    .byte 0,2,3,4,0,0,0,0  // 10111000
    .byte 1,2,3,4,0,0,0,0  // 01111000
    .byte 0,1,2,3,4,0,0,0  // 11111000
    .byte 5,0,0,0,0,0,0,0  // 00000100
    .byte 0,5,0,0,0,0,0,0  // 10000100
    .byte 1,5,0,0,0,0,0,0  // 01000100
    .byte 0,1,5,0,0,0,0,0  // 11000100
    .byte 2,5,0,0,0,0,0,0  // 00100100
    .byte 0,2,5,0,0,0,0,0  // 10100100
    .byte 1,2,5,0,0,0,0,0  // 01100100
    .byte 0,1,2,5,0,0,0,0  // 11100100
    .byte 3,5,0,0,0,0,0,0  // 00010100
    .byte 0,3,5,0,0,0,0,0  // 10010100
    .byte 1,3,5,0,0,0,0,0  // 01010100
    .byte 0,1,3,5,0,0,0,0  // 11010100
    .byte 2,3,5,0,0,0,0,0  // 00110100
    .byte 0,2,3,5,0,0,0,0  // 10110100
    .byte 1,2,3,5,0,0,0,0  // 01110100
    .byte 0,1,2,3,5,0,0,0  // 11110100
    .byte 4,5,0,0,0,0,0,0  // 00001100
    .byte 0,4,5,0,0,0,0,0  // 10001100
    .byte 1,4,5,0,0,0,0,0  // 01001100
    .byte 0,1,4,5,0,0,0,0  // 11001100
    .byte 2,4,5,0,0,0,0,0  // 00101100
    .byte 0,2,4,5,0,0,0,0  // 10101100
    .byte 1,2,4,5,0,0,0,0  // 01101100
    .byte 0,1,2,4,5,0,0,0  // 11101100
    .byte 3,4,5,0,0,0,0,0  // 00011100
    .byte 0,3,4,5,0,0,0,0  // 10011100
    .byte 1,3,4,5,0,0,0,0  // 01011100
    .byte 0,1,3,4,5,0,0,0  // 11011100
    .byte 2,3,4,5,0,0,0,0  // 00111100
    .byte 0,2,3,4,5,0,0,0  // 10111100
    .byte 1,2,3,4,5,0,0,0  // 01111100
    .byte 0,1,2,3,4,5,0,0  // 11111100
    .byte 6,0,0,0,0,0,0,0  // 00000010
    .byte 0,6,0,0,0,0,0,0  // 10000010
    .byte 1,6,0,0,0,0,0,0  // 01000010
    .byte 0,1,6,0,0,0,0,0  // 11000010
    .byte 2,6,0,0,0,0,0,0  // 00100010
    .byte 0,2,6,0,0,0,0,0  // 10100010
    .byte 1,2,6,0,0,0,0,0  // 01100010
    .byte 0,1,2,6,0,0,0,0  // 11100010
    .byte 3,6,0,0,0,0,0,0  // 00010010
    .byte 0,3,6,0,0,0,0,0  // 10010010
    .byte 1,3,6,0,0,0,0,0  // 01010010
    .byte 0,1,3,6,0,0,0,0  // 11010010
    .byte 2,3,6,0,0,0,0,0  // 00110010
    .byte 0,2,3,6,0,0,0,0  // 10110010
    .byte 1,2,3,6,0,0,0,0  // 01110010
    .byte 0,1,2,3,6,0,0,0  // 11110010
    .byte 4,6,0,0,0,0,0,0  // 00001010
    .byte 0,4,6,0,0,0,0,0  // 10001010
    .byte 1,4,6,0,0,0,0,0  // 01001010
    .byte 0,1,4,6,0,0,0,0  // 11001010
    .byte 2,4,6,0,0,0,0,0  // 00101010
    .byte 0,2,4,6,0,0,0,0  // 10101010
    .byte 1,2,4,6,0,0,0,0  // 01101010
    .byte 0,1,2,4,6,0,0,0  // 11101010
    .byte 3,4,6,0,0,0,0,0  // 00011010
    .byte 0,3,4,6,0,0,0,0  // 10011010
    .byte 1,3,4,6,0,0,0,0  // 01011010
    .byte 0,1,3,4,6,0,0,0  // 11011010
    .byte 2,3,4,6,0,0,0,0  // 00111010
    .byte 0,2,3,4,6,0,0,0  // 10111010
    .byte 1,2,3,4,6,0,0,0  // 01111010
    .byte 0,1,2,3,4,6,0,0  // 11111010
    .byte 5,6,0,0,0,0,0,0  // 00000110
    .byte 0,5,6,0,0,0,0,0  // 10000110
    .byte 1,5,6,0,0,0,0,0  // 01000110
    .byte 0,1,5,6,0,0,0,0  // 11000110
    .byte 2,5,6,0,0,0,0,0  // 00100110
    .byte 0,2,5,6,0,0,0,0  // 10100110
    .byte 1,2,5,6,0,0,0,0  // 01100110
    .byte 0,1,2,5,6,0,0,0  // 11100110
    .byte 3,5,6,0,0,0,0,0  // 00010110
    .byte 0,3,5,6,0,0,0,0  // 10010110
    .byte 1,3,5,6,0,0,0,0  // 01010110
    .byte 0,1,3,5,6,0,0,0  // 11010110
    .byte 2,3,5,6,0,0,0,0  // 00110110
    .byte 0,2,3,5,6,0,0,0  // 10110110
    .byte 1,2,3,5,6,0,0,0  // 01110110
    .byte 0,1,2,3,5,6,0,0  // 11110110
    .byte 4,5,6,0,0,0,0,0  // 00001110
    .byte 0,4,5,6,0,0,0,0  // 10001110
    .byte 1,4,5,6,0,0,0,0  // 01001110
    .byte 0,1,4,5,6,0,0,0  // 11001110
    .byte 2,4,5,6,0,0,0,0  // 00101110
    .byte 0,2,4,5,6,0,0,0  // 10101110
    .byte 1,2,4,5,6,0,0,0  // 01101110
    .byte 0,1,2,4,5,6,0,0  // 11101110
    .byte 3,4,5,6,0,0,0,0  // 00011110
    .byte 0,3,4,5,6,0,0,0  // 10011110
    .byte 1,3,4,5,6,0,0,0  // 01011110
    .byte 0,1,3,4,5,6,0,0  // 11011110
    .byte 2,3,4,5,6,0,0,0  // 00111110
    .byte 0,2,3,4,5,6,0,0  // 10111110
    .byte 1,2,3,4,5,6,0,0  // 01111110
    .byte 0,1,2,3,4,5,6,0  // 11111110
    .byte 7,0,0,0,0,0,0,0  // 00000001
    .byte 0,7,0,0,0,0,0,0  // 10000001
    .byte 1,7,0,0,0,0,0,0  // 01000001
    .byte 0,1,7,0,0,0,0,0  // 11000001
    .byte 2,7,0,0,0,0,0,0  // 00100001
    .byte 0,2,7,0,0,0,0,0  // 10100001
    .byte 1,2,7,0,0,0,0,0  // 01100001
    .byte 0,1,2,7,0,0,0,0  // 11100001
    .byte 3,7,0,0,0,0,0,0  // 00010001
    .byte 0,3,7,0,0,0,0,0  // 10010001
    .byte 1,3,7,0,0,0,0,0  // 01010001
    .byte 0,1,3,7,0,0,0,0  // 11010001
    .byte 2,3,7,0,0,0,0,0  // 00110001
    .byte 0,2,3,7,0,0,0,0  // 10110001
    .byte 1,2,3,7,0,0,0,0  // 01110001
    .byte 0,1,2,3,7,0,0,0  // 11110001
    .byte 4,7,0,0,0,0,0,0  // 00001001
    .byte 0,4,7,0,0,0,0,0  // 10001001
    .byte 1,4,7,0,0,0,0,0  // 01001001
    .byte 0,1,4,7,0,0,0,0  // 11001001
    .byte 2,4,7,0,0,0,0,0  // 00101001
    .byte 0,2,4,7,0,0,0,0  // 10101001
    .byte 1,2,4,7,0,0,0,0  // 01101001
    .byte 0,1,2,4,7,0,0,0  // 11101001
    .byte 3,4,7,0,0,0,0,0  // 00011001
    .byte 0,3,4,7,0,0,0,0  // 10011001
    .byte 1,3,4,7,0,0,0,0  // 01011001
    .byte 0,1,3,4,7,0,0,0  // 11011001
    .byte 2,3,4,7,0,0,0,0  // 00111001
    .byte 0,2,3,4,7,0,0,0  // 10111001
    .byte 1,2,3,4,7,0,0,0  // 01111001
    .byte 0,1,2,3,4,7,0,0  // 11111001
    .byte 5,7,0,0,0,0,0,0  // 00000101
    .byte 0,5,7,0,0,0,0,0  // 10000101
    .byte 1,5,7,0,0,0,0,0  // 01000101
    .byte 0,1,5,7,0,0,0,0  // 11000101
    .byte 2,5,7,0,0,0,0,0  // 00100101
    .byte 0,2,5,7,0,0,0,0  // 10100101
    .byte 1,2,5,7,0,0,0,0  // 01100101
    .byte 0,1,2,5,7,0,0,0  // 11100101
    .byte 3,5,7,0,0,0,0,0  // 00010101
    .byte 0,3,5,7,0,0,0,0  // 10010101
    .byte 1,3,5,7,0,0,0,0  // 01010101
    .byte 0,1,3,5,7,0,0,0  // 11010101
    .byte 2,3,5,7,0,0,0,0  // 00110101
    .byte 0,2,3,5,7,0,0,0  // 10110101
    .byte 1,2,3,5,7,0,0,0  // 01110101
    .byte 0,1,2,3,5,7,0,0  // 11110101
    .byte 4,5,7,0,0,0,0,0  // 00001101
    .byte 0,4,5,7,0,0,0,0  // 10001101
    .byte 1,4,5,7,0,0,0,0  // 01001101
    .byte 0,1,4,5,7,0,0,0  // 11001101
    .byte 2,4,5,7,0,0,0,0  // 00101101
    .byte 0,2,4,5,7,0,0,0  // 10101101
    .byte 1,2,4,5,7,0,0,0  // 01101101
    .byte 0,1,2,4,5,7,0,0  // 11101101
    .byte 3,4,5,7,0,0,0,0  // 00011101
    .byte 0,3,4,5,7,0,0,0  // 10011101
    .byte 1,3,4,5,7,0,0,0  // 01011101
    .byte 0,1,3,4,5,7,0,0  // 11011101
    .byte 2,3,4,5,7,0,0,0  // 00111101
    .byte 0,2,3,4,5,7,0,0  // 10111101
    .byte 1,2,3,4,5,7,0,0  // 01111101
    .byte 0,1,2,3,4,5,7,0  // 11111101
    .byte 6,7,0,0,0,0,0,0  // 00000011
    .byte 0,6,7,0,0,0,0,0  // 10000011
    .byte 1,6,7,0,0,0,0,0  // 01000011
    .byte 0,1,6,7,0,0,0,0  // 11000011
    .byte 2,6,7,0,0,0,0,0  // 00100011
    .byte 0,2,6,7,0,0,0,0  // 10100011
    .byte 1,2,6,7,0,0,0,0  // 01100011
    .byte 0,1,2,6,7,0,0,0  // 11100011
    .byte 3,6,7,0,0,0,0,0  // 00010011
    .byte 0,3,6,7,0,0,0,0  // 10010011
    .byte 1,3,6,7,0,0,0,0  // 01010011
    .byte 0,1,3,6,7,0,0,0  // 11010011
    .byte 2,3,6,7,0,0,0,0  // 00110011
    .byte 0,2,3,6,7,0,0,0  // 10110011
    .byte 1,2,3,6,7,0,0,0  // 01110011
    .byte 0,1,2,3,6,7,0,0  // 11110011
    .byte 4,6,7,0,0,0,0,0  // 00001011
    .byte 0,4,6,7,0,0,0,0  // 10001011
    .byte 1,4,6,7,0,0,0,0  // 01001011
    .byte 0,1,4,6,7,0,0,0  // 11001011
    .byte 2,4,6,7,0,0,0,0  // 00101011
    .byte 0,2,4,6,7,0,0,0  // 10101011
    .byte 1,2,4,6,7,0,0,0  // 01101011
    .byte 0,1,2,4,6,7,0,0  // 11101011
    .byte 3,4,6,7,0,0,0,0  // 00011011
    .byte 0,3,4,6,7,0,0,0  // 10011011
    .byte 1,3,4,6,7,0,0,0  // 01011011
    .byte 0,1,3,4,6,7,0,0  // 11011011
    .byte 2,3,4,6,7,0,0,0  // 00111011
    .byte 0,2,3,4,6,7,0,0  // 10111011
    .byte 1,2,3,4,6,7,0,0  // 01111011
    .byte 0,1,2,3,4,6,7,0  // 11111011
    .byte 5,6,7,0,0,0,0,0  // 00000111
    .byte 0,5,6,7,0,0,0,0  // 10000111
    .byte 1,5,6,7,0,0,0,0  // 01000111
    .byte 0,1,5,6,7,0,0,0  // 11000111
    .byte 2,5,6,7,0,0,0,0  // 00100111
    .byte 0,2,5,6,7,0,0,0  // 10100111
    .byte 1,2,5,6,7,0,0,0  // 01100111
    .byte 0,1,2,5,6,7,0,0  // 11100111
    .byte 3,5,6,7,0,0,0,0  // 00010111
    .byte 0,3,5,6,7,0,0,0  // 10010111
    .byte 1,3,5,6,7,0,0,0  // 01010111
    .byte 0,1,3,5,6,7,0,0  // 11010111
    .byte 2,3,5,6,7,0,0,0  // 00110111
    .byte 0,2,3,5,6,7,0,0  // 10110111
    .byte 1,2,3,5,6,7,0,0  // 01110111
    .byte 0,1,2,3,5,6,7,0  // 11110111
    .byte 4,5,6,7,0,0,0,0  // 00001111
    .byte 0,4,5,6,7,0,0,0  // 10001111
    .byte 1,4,5,6,7,0,0,0  // 01001111
    .byte 0,1,4,5,6,7,0,0  // 11001111
    .byte 2,4,5,6,7,0,0,0  // 00101111
    .byte 0,2,4,5,6,7,0,0  // 10101111
    .byte 1,2,4,5,6,7,0,0  // 01101111
    .byte 0,1,2,4,5,6,7,0  // 11101111
    .byte 3,4,5,6,7,0,0,0  // 00011111
    .byte 0,3,4,5,6,7,0,0  // 10011111
    .byte 1,3,4,5,6,7,0,0  // 01011111
    .byte 0,1,3,4,5,6,7,0  // 11011111
    .byte 2,3,4,5,6,7,0,0  // 00111111
    .byte 0,2,3,4,5,6,7,0  // 10111111
    .byte 1,2,3,4,5,6,7,0  // 01111111
    .byte 0,1,2,3,4,5,6,7  // 11111111
