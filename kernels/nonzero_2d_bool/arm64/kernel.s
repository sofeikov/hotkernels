// size_t nonzero_2d_bool(const bool *m, int64_t *out, size_t n, size_t width);
// Writes the row and column of every true byte of m (rows of width) to out as
// int64 pairs, in C order. Returns how many int64 it wrote.
//
// Scalar: every element's pair is stored, and the write position advances by
// two only when the byte is true (`out` is a scratch buffer of 2n).
.text
.globl _nonzero_2d_bool
.p2align 2
_nonzero_2d_bool:
    mov     x5, x1                      // x5 = write pointer
    mov     x6, #0                      // x6 = row
    mov     x7, #0                      // x7 = column
    cbz     x2, Ldone
Lloop:
    ldrb    w8, [x0], #1
    stp     x6, x7, [x5]
    add     x5, x5, x8, lsl #4
    add     x7, x7, #1
    cmp     x7, x3
    csel    x7, x7, xzr, ne
    cinc    x6, x6, eq
    subs    x2, x2, #1
    b.ne    Lloop
Ldone:
    sub     x0, x5, x1
    lsr     x0, x0, #3
    ret
