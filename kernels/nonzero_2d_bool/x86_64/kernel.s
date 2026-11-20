.intel_syntax noprefix
# size_t nonzero_2d_bool(const bool *m, int64_t *out, size_t n, size_t width);
#
# rdi = m, rsi = out, rdx = n, rcx = width. Writes the row and the column of
# every true element of m, a matrix of n elements in rows of width, to out as
# pairs of int64 in C order, and returns how many int64 it wrote: two per true
# element. `out` is a scratch buffer of 2n int64: stores may land anywhere
# inside it, and only the returned count has to be right. Every mask byte is
# 0 or 1. Element i is at row i / width and column i % width wherever it is,
# so a partial last row is a row like the others.
#
# The width is first taken as min(width, n), which changes no element's row or
# column. The matrix is read in blocks of 32 mask bytes; one compare and one
# `vpmovmskb` give a word of 32 bits, one per element.
#
# A block with at most SPARSE_MAX true elements takes a fixed SPARSE_MAX
# positions out of the word with `tzcnt` and `blsr`, and stores each pair with
# two 8-byte stores. Past the last set bit `tzcnt` returns 32: a pair the
# write position does not advance over, and the next store overwrites it.
#
# A block with more goes through tables, eight elements (one group) at a
# time. The group's byte of the word indexes a table of its positions, and
# `vpshufb` spreads them into four registers of two (row, column) pairs each,
# stored at the write position, which then advances by twice the group's
# count. Every group stores 128 bytes for at most eight pairs.
#
# Wide rows (more than 8 elements after the clamp). A sparse position f gets
# its row without a division: row = high half of 2f * M with
# M = ceil(2^63 / width), exact while f * width < 2^63, which n < 2^31 ensures;
# the column is f - row * width. A dense block computes its first element's
# row and column the same way, and then each group adds its positions to
# (row, column). A group that runs past the end of a row (column + 8 > width,
# at most one end since width > 8) adds (1, -width) to the pairs of the
# positions past the end, chosen by a byte compare of the positions.
#
# Narrow rows (width 1 to 8). A block holds a whole number of rows, the most
# that fit in 32 bytes (30 for width 3), and the word's bits past them are
# cleared, so every block starts a row. The row within the block and the
# column of each of its 32 positions are then fixed for the width and come
# from tables: `vpshufb` looks them up for a group, and the block's first row
# is added. A sparse block reads them one byte at a time.
#
# Stores stay inside the capacity: a block starting at element f is read only
# when f + 32 <= n, at most 2f int64 are kept before it, and a block writes at
# most 2 * 32 int64 past that. The last 0 to 31 elements, matrices under 32
# elements and matrices of 2^31 elements or more are done one element at a
# time, as the reference does it.

# A block with more true elements than this goes through the tables.
.set SPARSE_MAX, 4
# Elements per block, and the largest width the narrow path takes.
.set BLOCK, 32
.set NARROW_MAX, 8
# Above this the multiplication is not exact; such a matrix runs scalar.
.set MAX_N, 0x7fffffff

# One entry of the position table: for byte \m, the positions of its set bits
# in ascending order, one byte each, zero past the count.
.macro POS8_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .if ((\m) >> \bit) & 1
    .byte \bit
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 8 - .Lcount, 1, 0
.endm

# The same positions for the narrow tables: each position o twice, as o and
# o + 8, to look up its row (table bytes 0 to 7) and its column (bytes 8 to
# 15) with one `vpshufb`.
.macro POSI_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .if ((\m) >> \bit) & 1
    .byte \bit, \bit + 8
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 16 - 2 * .Lcount, 1, 0
.endm

# The tables of one narrow width \w, 128 bytes: for each group k of a block,
# the row within the block of its eight positions 8k + o, then their columns;
# then the row of each of the 32 positions, then their columns.
.macro NARROW_TABLES w
    .irp k, 0, 1, 2, 3
    .irp o, 0, 1, 2, 3, 4, 5, 6, 7
    .byte (8 * \k + \o) / \w
    .endr
    .irp o, 0, 1, 2, 3, 4, 5, 6, 7
    .byte (8 * \k + \o) - ((8 * \k + \o) / \w) * \w
    .endr
    .endr
    .set .Lp, 0
    .rept BLOCK
    .byte .Lp / \w
    .set .Lp, .Lp + 1
    .endr
    .set .Lp, 0
    .rept BLOCK
    .byte .Lp - (.Lp / \w) * \w
    .set .Lp, .Lp + 1
    .endr
.endm

.section .rodata
.p2align 6
.Ltables:
.Lpos8:
    .set .Lbyte, 0
    .rept 256
    POS8_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr
.Lposi:
    .set .Lbyte, 0
    .rept 256
    POSI_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr
.Lnarrow:
    .irp w, 1, 2, 3, 4, 5, 6, 7, 8
    NARROW_TABLES \w
    .endr
    # A sparse block's positions past its last set bit are 32, one past the
    # last width's column table.
    .fill 64, 1, 0
# For each narrow width: elements in a block (whole rows), rows in a block.
.Lrows:
    .irp w, 1, 2, 3, 4, 5, 6, 7, 8
    .byte (BLOCK / \w) * \w, BLOCK / \w
    .endr

.set POSI_AT, .Lposi - .Ltables
.set NARROW_AT, .Lnarrow - .Ltables
.set ROWS_AT, .Lrows - .Ltables
# Offsets within one width's narrow tables.
.set NARROW_SIZE, 128
.set ROW_OF, 64
.set COL_OF, 96

# Spread controls, one 32-byte vector for each pair of pairs k = 0..3.
# Wide: from a group's eight positions (in every quadword), position 2k to the
# column of the low lane's pair and 2k + 1 to the high lane's; zero elsewhere.
.macro SPREAD_WIDE k
    .fill 8, 1, 0x80
    .byte 2 * \k
    .fill 7, 1, 0x80
    .fill 8, 1, 0x80
    .byte 2 * \k + 1
    .fill 7, 1, 0x80
.endm
# Narrow: from (row, column) byte pairs, pair 2k to the low lane and 2k + 1 to
# the high lane, row in the low quadword and column in the high one.
.macro SPREAD_NARROW k
    .byte 4 * \k
    .fill 7, 1, 0x80
    .byte 4 * \k + 1
    .fill 7, 1, 0x80
    .byte 4 * \k + 2
    .fill 7, 1, 0x80
    .byte 4 * \k + 3
    .fill 7, 1, 0x80
.endm
# Row ends: the compare byte of position 2k across the low lane and of 2k + 1
# across the high lane.
.macro SPREAD_MASK k
    .fill 16, 1, 2 * \k
    .fill 16, 1, 2 * \k + 1
.endm

.p2align 5
.Lspread_wide:
    .irp k, 0, 1, 2, 3
    SPREAD_WIDE \k
    .endr
.Lspread_narrow:
    .irp k, 0, 1, 2, 3
    SPREAD_NARROW \k
    .endr
.Lspread_mask:
    .irp k, 0, 1, 2, 3
    SPREAD_MASK \k
    .endr
# Added to (row, column, row, column) from one group to the next.
.Leight:
    .quad 0, 8, 0, 8

# Wide, sparse: the j-th position of eax, its pair at out pair j.
# r11 = block's first element, rdx = M, r10 = width, r9 = int64 written.
.macro WIDE_SPARSE j
    tzcnt   r8d, eax
    blsr    eax, eax
    add     r8, r11                     # f
    lea     r13, [r8 + r8]
    mulx    r13, r14, r13               # row: high half of 2f * M
    mov     [rsi + r9 * 8 + 16 * \j], r13
    imul    r13, r10
    sub     r8, r13                     # column
    mov     [rsi + r9 * 8 + 16 * \j + 8], r8
.endm

# Wide, dense: group j of the block, from the low byte of eax.
# ymm1 = (row, column, row, column) of the group's first element, r8 = its
# column. A group that crosses a row's end goes to WIDE_CROSS j.
.macro WIDE_GROUP j
    movzx   ecx, al
    shr     eax, 8
    vpbroadcastq ymm2, qword ptr [r12 + rcx * 8]
    lea     r14, [r8 + 8]
    cmp     r14, r10
    ja      .Lwide_cross\j
    vpshufb ymm3, ymm2, ymm8
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8], ymm3
    vpshufb ymm3, ymm2, ymm9
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8 + 32], ymm3
    vpshufb ymm3, ymm2, ymm10
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8 + 64], ymm3
    vpshufb ymm3, ymm2, ymm11
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8 + 96], ymm3
    popcnt  ecx, ecx
    add     r9, rcx
    add     r9, rcx
    vpaddq  ymm1, ymm1, ymm13
    mov     r8, r14
.Lwide_next\j:
.endm

# One pair of pairs of a crossing group: ymm4 holds 0xff for the positions
# past the row's end, ymm12 = (1, -width, 1, -width).
.macro CROSS_PAIRS k, spread
    vpshufb ymm3, ymm2, \spread
    vpaddq  ymm3, ymm3, ymm1
    vpshufb ymm5, ymm4, [rip + .Lspread_mask + 32 * \k]
    vpand   ymm5, ymm5, ymm12
    vpaddq  ymm3, ymm3, ymm5
    vmovdqu [rsi + r9 * 8 + 32 * \k], ymm3
.endm

# The group of WIDE_GROUP j when the row ends at position d = width - column.
.macro WIDE_CROSS j
.Lwide_cross\j:
    mov     r15, r10
    sub     r15, r8
    dec     r15                         # d - 1, 0 to 6
    vmovd   xmm4, r15d
    vpbroadcastb ymm4, xmm4
    vpcmpgtb ymm4, ymm2, ymm4           # 0xff where the position is >= d
    CROSS_PAIRS 0, ymm8
    CROSS_PAIRS 1, ymm9
    CROSS_PAIRS 2, ymm10
    CROSS_PAIRS 3, ymm11
    popcnt  ecx, ecx
    add     r9, rcx
    add     r9, rcx
    vpaddq  ymm1, ymm1, ymm14           # next row, column + 8 - width
    sub     r14, r10
    mov     r8, r14
    jmp     .Lwide_next\j
.endm

# Narrow, sparse: the j-th position of eax. r15 = the width's tables,
# r13 = the block's first row.
.macro NARROW_SPARSE j
    tzcnt   r8d, eax
    blsr    eax, eax
    movzx   ebp, byte ptr [r15 + r8 + ROW_OF]
    add     rbp, r13
    mov     [rsi + r9 * 8 + 16 * \j], rbp
    movzx   ebp, byte ptr [r15 + r8 + COL_OF]
    mov     [rsi + r9 * 8 + 16 * \j + 8], rbp
.endm

# Narrow, dense: group j, from the low byte of eax. \table holds the group's
# rows and columns; ymm1 = (first row, 0, first row, 0).
.macro NARROW_GROUP table
    movzx   ebp, al
    shr     eax, 8
    lea     r8d, [rbp + rbp]
    vbroadcasti128 ymm2, xmmword ptr [r12 + r8 * 8 + POSI_AT]
    vpshufb ymm2, \table, ymm2          # (row, column) bytes of each position
    vpshufb ymm3, ymm2, ymm8
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8], ymm3
    vpshufb ymm3, ymm2, ymm9
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8 + 32], ymm3
    vpshufb ymm3, ymm2, ymm10
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8 + 64], ymm3
    vpshufb ymm3, ymm2, ymm11
    vpaddq  ymm3, ymm3, ymm1
    vmovdqu [rsi + r9 * 8 + 96], ymm3
    popcnt  ebp, ebp
    add     r9, rbp
    add     r9, rbp
.endm

.text
.globl nonzero_2d_bool
.type nonzero_2d_bool, @function
.p2align 4
nonzero_2d_bool:
    xor     r9d, r9d                    # int64 written
    xor     r11d, r11d                  # next element
    mov     r10, rcx
    cmp     r10, rdx
    cmova   r10, rdx                    # width = min(width, n)
    xor     ecx, ecx                    # row
    xor     r8d, r8d                    # column
    cmp     rdx, BLOCK
    jb      .Lscalar
    cmp     rdx, MAX_N
    ja      .Lscalar
    test    r10, r10
    jz      .Lscalar
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    lea     rbx, [rdx - BLOCK]          # last block start
    lea     r12, [rip + .Ltables]
    vpxor   xmm0, xmm0, xmm0
    vmovdqa ymm8, [rip + .Lspread_wide]
    vmovdqa ymm9, [rip + .Lspread_wide + 32]
    vmovdqa ymm10, [rip + .Lspread_wide + 64]
    vmovdqa ymm11, [rip + .Lspread_wide + 96]
    cmp     r10, NARROW_MAX
    jbe     .Lnarrow_rows

    # Wide rows.
    mov     rax, 0x7fffffffffffffff
    xor     edx, edx
    div     r10
    lea     rdx, [rax + 1]              # M = ceil(2^63 / width)
    mov     eax, 1
    vmovq   xmm12, rax
    mov     rax, r10
    neg     rax
    vpinsrq xmm12, xmm12, rax, 1
    vinserti128 ymm12, ymm12, xmm12, 1  # (1, -width, 1, -width)
    vmovdqa ymm13, [rip + .Leight]
    vpaddq  ymm14, ymm12, ymm13
.p2align 5
.Lwide_block:
    vpcmpeqb ymm3, ymm0, [rdi + r11]    # 0xff where the mask byte is zero
    vpmovmskb eax, ymm3
    not     eax                         # one bit per true element
    popcnt  ecx, eax
    cmp     ecx, SPARSE_MAX
    ja      .Lwide_dense
    WIDE_SPARSE 0
    WIDE_SPARSE 1
    WIDE_SPARSE 2
    WIDE_SPARSE 3
    add     r9, rcx
    add     r9, rcx
.Lwide_advance:
    add     r11, BLOCK
    cmp     r11, rbx
    jbe     .Lwide_block
    # Row and column of the first element left.
    lea     r8, [r11 + r11]
    mulx    rcx, r8, r8
    mov     r8, rcx
    imul    r8, r10
    neg     r8
    add     r8, r11
    jmp     .Ltail
.Lwide_dense:
    lea     r8, [r11 + r11]
    mulx    r13, r8, r8                 # row of the block's first element
    mov     r8, r13
    imul    r8, r10
    neg     r8
    add     r8, r11                     # its column
    vmovq   xmm1, r13
    vpinsrq xmm1, xmm1, r8, 1
    vinserti128 ymm1, ymm1, xmm1, 1
    WIDE_GROUP 0
    WIDE_GROUP 1
    WIDE_GROUP 2
    WIDE_GROUP 3
    jmp     .Lwide_advance
    WIDE_CROSS 0
    WIDE_CROSS 1
    WIDE_CROSS 2
    WIDE_CROSS 3

    # Narrow rows: blocks of r14 elements, rdx rows each.
.Lnarrow_rows:
    lea     r15, [r10 - 1]
    shl     r15, 7                      # NARROW_SIZE
    lea     r15, [r12 + r15 + NARROW_AT]
    movzx   r14d, byte ptr [r12 + r10 * 2 + ROWS_AT - 2]
    movzx   edx, byte ptr [r12 + r10 * 2 + ROWS_AT - 1]
    xor     r13d, r13d                  # the block's first row
    vbroadcasti128 ymm4, xmmword ptr [r15]
    vbroadcasti128 ymm5, xmmword ptr [r15 + 16]
    vbroadcasti128 ymm6, xmmword ptr [r15 + 32]
    vbroadcasti128 ymm7, xmmword ptr [r15 + 48]
    vmovdqa ymm8, [rip + .Lspread_narrow]
    vmovdqa ymm9, [rip + .Lspread_narrow + 32]
    vmovdqa ymm10, [rip + .Lspread_narrow + 64]
    vmovdqa ymm11, [rip + .Lspread_narrow + 96]
    vpxor   xmm1, xmm1, xmm1
    vmovq   xmm13, rdx
    vinserti128 ymm13, ymm13, xmm13, 1  # (rows, 0, rows, 0)
.p2align 5
.Lnarrow_block:
    vpcmpeqb ymm3, ymm0, [rdi + r11]
    vpmovmskb eax, ymm3
    not     eax
    bzhi    eax, eax, r14d              # whole rows only
    popcnt  ecx, eax
    cmp     ecx, SPARSE_MAX
    ja      .Lnarrow_dense
    NARROW_SPARSE 0
    NARROW_SPARSE 1
    NARROW_SPARSE 2
    NARROW_SPARSE 3
    add     r9, rcx
    add     r9, rcx
.Lnarrow_advance:
    vpaddq  ymm1, ymm1, ymm13
    add     r13, rdx
    add     r11, r14
    cmp     r11, rbx
    jbe     .Lnarrow_block
    mov     rcx, r13
    xor     r8d, r8d
    jmp     .Ltail
.Lnarrow_dense:
    NARROW_GROUP ymm4
    NARROW_GROUP ymm5
    NARROW_GROUP ymm6
    NARROW_GROUP ymm7
    jmp     .Lnarrow_advance

    # rcx = row and r8 = column of element r11, the first not done.
.Ltail:
    lea     rdx, [rbx + BLOCK]          # n
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx

    # One element at a time, as the reference: rdx = n, r11 = next element,
    # rcx = its row, r8 = its column, r10 = width, r9 = int64 written.
.Lscalar:
    cmp     r11, rdx
    jae     .Ldone
.Lone:
    mov     [rsi + r9 * 8], rcx
    mov     [rsi + r9 * 8 + 8], r8
    movzx   eax, byte ptr [rdi + r11]
    add     eax, eax
    add     r9, rax
    inc     r8
    cmp     r8, r10
    jne     .Lsame_row
    xor     r8d, r8d
    inc     rcx
.Lsame_row:
    inc     r11
    cmp     r11, rdx
    jb      .Lone
.Ldone:
    mov     rax, r9
    vzeroupper
    ret
.size nonzero_2d_bool, .-nonzero_2d_bool

.section .note.GNU-stack,"",@progbits
