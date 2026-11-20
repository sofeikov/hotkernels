// size_t nonzero_2d_bool(const bool *m, int64_t *out, size_t n, size_t width);
// Writes the row and column of every true byte of m (rows of width) to out as
// int64 pairs, in C order. Returns how many int64 it wrote.
//
// The mask is read a row at a time, in chunks of 8 bytes and a last chunk of
// what is left of the row, so that no chunk crosses a row. A chunk is loaded
// as one 64-bit word; in the last chunk of a row the bytes past its end are
// masked off. One multiplication packs it into an 8-bit number, another gives
// the true count (times 8). The 8-bit number indexes a table whose entry lists
// the positions of the true bytes, in order, as eight bytes; those are widened
// to 64 bits and added to the chunk's first column, and they are
// interleaved with the row and stored: eight (row, column) pairs at the write pointer, of
// which the first `count` are kept.
//
// Stores stay inside the capacity: a chunk starting at element f is read only
// when f + 8 <= n, at most f pairs have been kept before it, and it writes 8
// pairs. The last 0 to 7 bytes of the mask, when a chunk cannot be read there,
// are handled one at a time.

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
    sub     x2, x4, #8                  // x2 = last place a chunk can start
    adrp    x9, Lpositions@PAGE
    add     x9, x9, Lpositions@PAGEOFF
    mov     x10, #0x4080                // 0x0102040810204080: gathers the low
    movk    x10, #0x1020, lsl #16       // bit of each byte into the top byte
    movk    x10, #0x0408, lsl #32
    movk    x10, #0x0102, lsl #48
    mov     x11, #0x0808080808080808    // sums the bytes, times 8, into the top byte
    mov     x16, #-1
    mov     w8, #8
    dup     v18.2d, x8                  // 8, twice: the column step of a chunk
Lrow:                                   // x0 = x17 = start of row x6, below the end
    sub     x12, x4, x0
    cmp     x12, x3
    csel    x12, x12, x3, lo
    add     x12, x0, x12                // x12 = end of the row, or of the mask
    sub     x13, x12, #8                // x13 = last place a whole chunk can start
    dup     v20.2d, x6                  // the row, twice
    movi    v17.2d, #0                  // column of the chunk
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
    lsr     x15, x16, x15               // the row's bytes of the word
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
    cmp     x0, x4
    b.lo    Lrow
    b       Ldone
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
