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
# Wide rows (more than 8 elements after the clamp). The row and the column of
# the block's first element are carried from block to block: the column
# advances by 32 mod width and the row by 32 / width, plus one where the
# column passes the width (one `div` on entry, then a compare and a `cmov`
# per block). A sparse position p of the block is at column c + p of the
# block's row r. Up to 32 elements a row, a table for the width gives the
# row within the block and the column of c + p. Wider, a block either lies in
# one row (c + 32 <= width: the pair is r and c + p, stored as they are) or
# crosses one row end, where a compare of c + p with the width picks the next
# row. A dense block adds each group's positions to (r, c), and a group that
# runs past the end of a row (column + 8 > width, at most one end since
# width > 8) adds (1, -width) to the pairs of the positions past it, chosen by
# a byte compare of the positions.
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
# most 2 * 32 int64 past that. The last 0 to 31 elements and matrices under
# 32 elements are done one element at a time, as the reference does it.

# A block with more true elements than this goes through the tables.
.set SPARSE_MAX, 4
# Elements per block, and the largest width the narrow path takes.
.set BLOCK, 32
.set NARROW_MAX, 8
# The largest width with sparse tables: a block's c + p is below 2 * BLOCK.
.set TABLE_MAX, 32

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

# The tables of one width \w, WIDTH_SIZE bytes. For narrow widths: for each
# group k of a block, the row within the block of its eight positions 8k + o,
# then their columns. For every width up to TABLE_MAX: j / w and j % w for
# j = 0 to 63, the row and the column of the block's c + p.
.macro WIDTH_TABLES w
    .irp k, 0, 1, 2, 3
    .irp o, 0, 1, 2, 3, 4, 5, 6, 7
    .byte (8 * \k + \o) / \w
    .endr
    .irp o, 0, 1, 2, 3, 4, 5, 6, 7
    .byte (8 * \k + \o) - ((8 * \k + \o) / \w) * \w
    .endr
    .endr
    .set .Lj, 0
    .rept 2 * BLOCK
    .byte .Lj / \w
    .set .Lj, .Lj + 1
    .endr
    .set .Lj, 0
    .rept 2 * BLOCK
    .byte .Lj - (.Lj / \w) * \w
    .set .Lj, .Lj + 1
    .endr
    .fill 64, 1, 0
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
.Lwidths:
    .set .Lw, 1
    .rept TABLE_MAX
    WIDTH_TABLES .Lw
    .set .Lw, .Lw + 1
    .endr
# For each narrow width: elements in a block (whole rows), rows in a block.
.Lrows:
    .irp w, 1, 2, 3, 4, 5, 6, 7, 8
    .byte (BLOCK / \w) * \w, BLOCK / \w
    .endr

.set POSI_AT, .Lposi - .Ltables
.set WIDTHS_AT, .Lwidths - .Ltables
.set ROWS_AT, .Lrows - .Ltables
# Size of one width's tables (a power of two), and offsets within them.
.set WIDTH_SHIFT, 8
.set ROW_OF, 64
.set COL_OF, 128

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

# Red-zone slots of the wide path, below the six saved registers.
.set STEP_COL, -8                       # 32 mod width
.set STEP_ROW, -16                      # -1 - 32 / width, for `sbb`

# Sparse, with tables: the j-th position of eax. rbp = the width's tables
# plus the block's first column, r13 = the block's first row.
.macro TABLE_SPARSE j
    tzcnt   r8d, eax
    blsr    eax, eax
    movzx   edx, byte ptr [rbp + r8 + ROW_OF]
    add     rdx, r13
    mov     [rsi + r9 * 8 + 16 * \j], rdx
    movzx   edx, byte ptr [rbp + r8 + COL_OF]
    mov     [rsi + r9 * 8 + 16 * \j + 8], rdx
.endm

# Sparse, wide, a block inside one row: row r13, column r15 + p.
.macro ROW_SPARSE j
    tzcnt   r8d, eax
    blsr    eax, eax
    add     r8, r15
    mov     [rsi + r9 * 8 + 16 * \j], r13
    mov     [rsi + r9 * 8 + 16 * \j + 8], r8
.endm

# Sparse, wide, a block that crosses a row's end: c + p past the width is
# column c + p - width of the next row.
.macro CROSS_SPARSE j
    tzcnt   r8d, eax
    blsr    eax, eax
    add     r8, r15                     # c + p
    mov     r14, r8
    sub     r14, r10                    # carry: still in the block's row
    cmovae  r8, r14
    mov     rdx, r13
    sbb     rdx, -1                     # row + 1 - carry
    mov     [rsi + r9 * 8 + 16 * \j], rdx
    mov     [rsi + r9 * 8 + 16 * \j + 8], r8
.endm

# The byte of eax for group j of a block in ecx (no REX prefix, so that `ah`
# can be named), and eax shifted on to the next two groups after an odd one.
.macro GROUP_BYTE j, reg
    .if \j & 1
    movzx   \reg, ah
    shr     eax, 16
    .else
    movzx   \reg, al
    .endif
.endm

# Wide, dense: four pairs of pairs from the positions in ymm2 and the
# (row, column, row, column) of the group's first element in ymm1, stored.
.macro WIDE_PAIRS
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
.endm

# Wide, dense, a block inside one row: group j.
.macro ROW_GROUP j
    GROUP_BYTE \j, ecx
    vpbroadcastq ymm2, qword ptr [r12 + rcx * 8]
    WIDE_PAIRS
    popcnt  ecx, ecx
    lea     r9, [r9 + rcx * 2]
    vpaddq  ymm1, ymm1, ymm13
.endm

# Wide, dense: group j of the block. r8 = the group's first column. A group
# that crosses a row's end goes to WIDE_CROSS j.
.macro WIDE_GROUP j
    GROUP_BYTE \j, ecx
    vpbroadcastq ymm2, qword ptr [r12 + rcx * 8]
    lea     r14, [r8 + 8]
    cmp     r14, r10
    ja      .Lwide_cross\j
    WIDE_PAIRS
    popcnt  ecx, ecx
    lea     r9, [r9 + rcx * 2]
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
    mov     rdx, r10
    sub     rdx, r8
    dec     edx                         # d - 1, 0 to 6
    vmovd   xmm4, edx
    vpbroadcastb ymm4, xmm4
    vpcmpgtb ymm4, ymm2, ymm4           # 0xff where the position is >= d
    CROSS_PAIRS 0, ymm8
    CROSS_PAIRS 1, ymm9
    CROSS_PAIRS 2, ymm10
    CROSS_PAIRS 3, ymm11
    popcnt  ecx, ecx
    lea     r9, [r9 + rcx * 2]
    vpaddq  ymm1, ymm1, ymm14           # next row, column + 8 - width
    sub     r14, r10
    mov     r8, r14
    jmp     .Lwide_next\j
.endm

# Narrow, dense: group j. \table holds the group's rows and columns;
# ymm1 = (first row, 0, first row, 0).
.macro NARROW_GROUP j, table
    GROUP_BYTE \j, edx
    lea     r8d, [rdx + rdx]
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
    popcnt  edx, edx
    lea     r9, [r9 + rdx * 2]
.endm

# The mask word of the block at r11 in eax, its count in ecx.
.macro BLOCK_WORD
    vpcmpeqb ymm3, ymm0, [rdi + r11]    # 0xff where the mask byte is zero
    vpmovmskb eax, ymm3
    not     eax                         # one bit per true element
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
    xor     r13d, r13d                  # the block's first row
    xor     r15d, r15d                  # and column
    lea     rbp, [r10 - 1]
    shl     rbp, WIDTH_SHIFT
    lea     rbp, [r12 + rbp + WIDTHS_AT] # the width's tables, if it has them
    vmovdqa ymm8, [rip + .Lspread_wide]
    vmovdqa ymm9, [rip + .Lspread_wide + 32]
    vmovdqa ymm10, [rip + .Lspread_wide + 64]
    vmovdqa ymm11, [rip + .Lspread_wide + 96]
    cmp     r10, NARROW_MAX
    jbe     .Lnarrow_rows

    # Wide rows: the block's first element at row r13, column r15.
    mov     eax, BLOCK                  # over 32 elements a row: 32 columns, no row
    xor     edx, edx
    cmp     r10, BLOCK
    ja      .Lsteps
    div     r10d
    xchg    eax, edx
.Lsteps:
    mov     [rsp + STEP_COL], rax
    not     rdx
    mov     [rsp + STEP_ROW], rdx
    mov     eax, 1
    vmovq   xmm12, rax
    mov     rax, r10
    neg     rax
    vpinsrq xmm12, xmm12, rax, 1
    vinserti128 ymm12, ymm12, xmm12, 1  # (1, -width, 1, -width)
    vmovdqa ymm13, [rip + .Leight]
    vpaddq  ymm14, ymm12, ymm13
    cmp     r10, TABLE_MAX
    ja      .Lwide_block
.p2align 5
.Lmid_block:
    BLOCK_WORD
    popcnt  ecx, eax
    cmp     ecx, SPARSE_MAX
    ja      .Lwide_dense
    mov     r14, rbp
    add     rbp, r15
    TABLE_SPARSE 0
    TABLE_SPARSE 1
    TABLE_SPARSE 2
    TABLE_SPARSE 3
    mov     rbp, r14
    add     r9, rcx
    add     r9, rcx
.Lmid_advance:
    add     r15, [rsp + STEP_COL]
    mov     r14, r15
    sub     r14, r10
    cmovae  r15, r14
    sbb     r13, [rsp + STEP_ROW]       # row + 32 / width + 1 - carry
    add     r11, BLOCK
    cmp     r11, rbx
    jbe     .Lmid_block
    jmp     .Lwide_tail
.p2align 5
.Lwide_block:
    BLOCK_WORD
    popcnt  ecx, eax
    cmp     ecx, SPARSE_MAX
    ja      .Lwide_dense
    lea     r8, [r15 + BLOCK]
    cmp     r8, r10
    ja      .Lwide_sparse_cross
    ROW_SPARSE 0
    ROW_SPARSE 1
    ROW_SPARSE 2
    ROW_SPARSE 3
    add     r9, rcx
    add     r9, rcx
.Lwide_advance:
    add     r15, [rsp + STEP_COL]
    mov     r14, r15
    sub     r14, r10
    cmovae  r15, r14
    sbb     r13, [rsp + STEP_ROW]
    add     r11, BLOCK
    cmp     r11, rbx
    jbe     .Lwide_block
    jmp     .Lwide_tail
.Lwide_sparse_cross:
    CROSS_SPARSE 0
    CROSS_SPARSE 1
    CROSS_SPARSE 2
    CROSS_SPARSE 3
    add     r9, rcx
    add     r9, rcx
    jmp     .Lwide_advance

    # A dense block of either wide loop.
.Lwide_dense:
    vmovq   xmm1, r13
    vpinsrq xmm1, xmm1, r15, 1
    vinserti128 ymm1, ymm1, xmm1, 1
    lea     r8, [r15 + BLOCK]
    cmp     r8, r10
    ja      .Lwide_dense_cross
    ROW_GROUP 0
    ROW_GROUP 1
    ROW_GROUP 2
    ROW_GROUP 3
    cmp     r10, TABLE_MAX
    jbe     .Lmid_advance
    jmp     .Lwide_advance
.Lwide_dense_cross:
    mov     r8, r15
    WIDE_GROUP 0
    WIDE_GROUP 1
    WIDE_GROUP 2
    WIDE_GROUP 3
    cmp     r10, TABLE_MAX
    jbe     .Lmid_advance
    jmp     .Lwide_advance
    WIDE_CROSS 0
    WIDE_CROSS 1
    WIDE_CROSS 2
    WIDE_CROSS 3

.Lwide_tail:
    mov     rcx, r13
    mov     r8, r15
    jmp     .Ltail

    # Narrow rows: blocks of r14 elements, rdx rows each.
.Lnarrow_rows:
    movzx   r14d, byte ptr [r12 + r10 * 2 + ROWS_AT - 2]
    movzx   r15d, byte ptr [r12 + r10 * 2 + ROWS_AT - 1]
    vbroadcasti128 ymm4, xmmword ptr [rbp]
    vbroadcasti128 ymm5, xmmword ptr [rbp + 16]
    vbroadcasti128 ymm6, xmmword ptr [rbp + 32]
    vbroadcasti128 ymm7, xmmword ptr [rbp + 48]
    vmovdqa ymm8, [rip + .Lspread_narrow]
    vmovdqa ymm9, [rip + .Lspread_narrow + 32]
    vmovdqa ymm10, [rip + .Lspread_narrow + 64]
    vmovdqa ymm11, [rip + .Lspread_narrow + 96]
    vpxor   xmm1, xmm1, xmm1
    vmovq   xmm13, r15
    vinserti128 ymm13, ymm13, xmm13, 1  # (rows, 0, rows, 0)
.p2align 5
.Lnarrow_block:
    BLOCK_WORD
    bzhi    eax, eax, r14d              # whole rows only
    popcnt  ecx, eax
    cmp     ecx, SPARSE_MAX
    ja      .Lnarrow_dense
    TABLE_SPARSE 0
    TABLE_SPARSE 1
    TABLE_SPARSE 2
    TABLE_SPARSE 3
    add     r9, rcx
    add     r9, rcx
.Lnarrow_advance:
    vpaddq  ymm1, ymm1, ymm13
    add     r13, r15
    add     r11, r14
    cmp     r11, rbx
    jbe     .Lnarrow_block
    mov     rcx, r13
    xor     r8d, r8d
    jmp     .Ltail
.Lnarrow_dense:
    NARROW_GROUP 0, ymm4
    NARROW_GROUP 1, ymm5
    NARROW_GROUP 2, ymm6
    NARROW_GROUP 3, ymm7
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
    inc     r8
    cmp     r8, r10
    jne     .Lsame_row
    xor     r8d, r8d
    inc     rcx
.Lsame_row:
    lea     r9, [r9 + rax * 2]
    inc     r11
    cmp     r11, rdx
    jae     .Ldone
    mov     [rsi + r9 * 8], rcx
    mov     [rsi + r9 * 8 + 8], r8
    movzx   eax, byte ptr [rdi + r11]
    inc     r8
    cmp     r8, r10
    jne     .Lsame_row2
    xor     r8d, r8d
    inc     rcx
.Lsame_row2:
    lea     r9, [r9 + rax * 2]
    inc     r11
    cmp     r11, rdx
    jb      .Lone
.Ldone:
    mov     rax, r9
    vzeroupper
    ret
.size nonzero_2d_bool, .-nonzero_2d_bool

.section .note.GNU-stack,"",@progbits
