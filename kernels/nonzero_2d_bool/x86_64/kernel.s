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
# positions out of the word with `tzcnt` and `blsr`, and stores each pair
# (one 16-byte store or two of 8 bytes, by the path below). Past the last set bit `tzcnt` returns 32: a pair the
# write position does not advance over, and the next store overwrites it.
# After a sparse block the kernel looks at a window of two blocks, a 64-bit
# word, and takes PAIR_MAX positions out of it if it has no more; a window
# with more is taken one block at a time, and so is everything after a dense
# block until the next sparse one. A dense stretch never pays for the window.
#
# A block with more goes eight elements (one group) at a time, and stores
# four registers of two (row, column) pairs each at the write position, which
# then advances by twice the group's count: 128 bytes for at most eight pairs.
#
# The row r and the column c of each block's first element are carried from
# block to block: the column advances by 32 mod width and the row by
# 32 / width, plus one where the column passes the width (one `div` on entry,
# then a compare, `cmov` and `sbb` per block). Position p of a block is at
# column c + p of row r, until the row ends.
#
# Rows of up to 64 elements. c + p is below 128, and a table of 256 bytes for
# the width holds the row (c + p) / width and the column (c + p) % width of
# each, a byte each. A sparse position widens its two bytes with `vpmovzxbq`,
# adds the block's row and stores the pair with one store. A dense group
# loads the 16 bytes of the table at its first column, the (row, column)
# bytes of its eight positions; the group's byte of the word indexes a table
# of its positions, and one `vpshufb` picks the bytes of the positions present
# out of the window, four more spread them into the four registers, and the
# block's row is added.
#
# Wider rows. A sparse block or window either lies in one row (c + 32 or
# c + 64 <= width: the pair is r and c + p, stored as they are) or crosses one
# row end, where a
# compare of c + p with the width picks the next row. A dense group's byte
# indexes a table of its positions, `vpbroadcastq` puts them in every
# quadword and four `vpshufb` spread them into four registers of two
# (0, position) pairs, to which (r, column of the group's first element) is
# added. A group that runs past a row's end (column + 8 > width; only in a
# block that does) adds (1, -width) to the pairs of the positions past it,
# chosen by a byte compare of the positions.
#
# Stores stay inside the capacity: a block starting at element f is read only
# when f + 32 <= n (a window when f + 64 <= n), at most 2f int64 are kept
# before it, and a block writes at most 2 * 32 int64 past that (a window
# 2 * PAIR_MAX). The last 0 to 31 elements and matrices under
# 32 elements are done one element at a time, as the reference does it.

# A block with more true elements than this goes through the tables.
.set SPARSE_MAX, 4
# A window of two blocks with more than this is taken one block at a time.
.set PAIR_MAX, 6
# Elements per block.
.set BLOCK, 32
# The largest width with a table: a window's c + p is below 4 * BLOCK.
.set TABLE_MAX, 64

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

# The same positions for the table path: each position o as the two bytes
# 2o and 2o + 1, to pick its (row, column) bytes out of a table window with
# one `vpshufb`.
.macro POSI_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .if ((\m) >> \bit) & 1
    .byte 2 * \bit, 2 * \bit + 1
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 16 - 2 * .Lcount, 1, 0
.endm

# The table of one width \w, 256 bytes: for j = 0 to 127, the row j / w and
# the column j % w, a byte each. A block whose first element is at column c
# finds the pair of its position p at j = c + p.
.macro WIDTH_TABLE w
    .set .Lj, 0
    .rept 4 * BLOCK
    .byte .Lj / \w, .Lj - (.Lj / \w) * \w
    .set .Lj, .Lj + 1
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
.Lwidths:
    .set .Lw, 1
    .rept TABLE_MAX
    WIDTH_TABLE .Lw
    .set .Lw, .Lw + 1
    .endr

.set POSI_AT, .Lposi - .Ltables
.set WIDTHS_AT, .Lwidths - .Ltables
# Size of one width's table, as a power of two.
.set WIDTH_SHIFT, 8

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
# Table: from (row, column) byte pairs, pair 2k to the low lane and 2k + 1 to
# the high lane, row in the low quadword and column in the high one.
.macro SPREAD_TABLE k
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
.Lspread_table:
    .irp k, 0, 1, 2, 3
    SPREAD_TABLE \k
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
.set PAIR_COL, -24                      # 64 mod width
.set PAIR_ROW, -32                      # -1 - 64 / width

# The row r13 and column r15 of the next block's first element, from the
# block's or window's step: column + step mod width, row + step / width, plus
# one where the column passes the width.
.macro CARRY col, row
    add     r15, [rsp + \col]
    mov     r14, r15
    sub     r14, r10
    cmovae  r15, r14
    sbb     r13, [rsp + \row]          # row + step / width + 1 - carry
.endm

# Sparse, with a table: the j-th position of the word. r14 = the width's
# table at the block's first column, xmm1 = (the block's first row, 0). The
# pair's two bytes are widened to two int64 and stored with one store.
.macro TABLE_SPARSE j, word=eax, pos=r8d
    tzcnt   \pos, \word
    blsr    \word, \word
    vpmovzxbq xmm2, word ptr [r14 + r8 * 2]
    vpaddq  xmm2, xmm2, xmm1
    vmovdqu [rsi + r9 * 8 + 16 * \j], xmm2
.endm

# Sparse, wide, a block inside one row: row r13, column r15 + p.
.macro ROW_SPARSE j, word=eax, pos=r8d
    tzcnt   \pos, \word
    blsr    \word, \word
    add     r8, r15
    mov     [rsi + r9 * 8 + 16 * \j], r13
    mov     [rsi + r9 * 8 + 16 * \j + 8], r8
.endm

# Sparse, wide, a block that crosses a row's end: c + p past the width is
# column c + p - width of the next row.
.macro CROSS_SPARSE j, word=eax, pos=r8d
    tzcnt   \pos, \word
    blsr    \word, \word
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

# Dense, with a table: group j. The 16 bytes of the table from the group's
# first column are the (row, column) bytes of its eight positions; the
# positions present pick theirs out. ymm1 = (first row, 0, first row, 0).
.macro TABLE_GROUP j
    GROUP_BYTE \j, edx
    lea     r8d, [rdx + rdx]
    vbroadcasti128 ymm2, xmmword ptr [r12 + r8 * 8 + POSI_AT]
    vbroadcasti128 ymm3, xmmword ptr [r14 + 16 * \j]
    vpshufb ymm2, ymm3, ymm2
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

# The mask word of the block at r11 in eax.
.macro BLOCK_WORD
    vpcmpeqb ymm3, ymm0, [rdi + r11]    # 0xff where the mask byte is zero
    vpmovmskb eax, ymm3
    not     eax                         # one bit per true element
.endm

# The mask word of the two blocks at r11 in rax, the first in the low half.
.macro PAIR_WORD
    vpcmpeqb ymm3, ymm0, [rdi + r11]
    vpcmpeqb ymm4, ymm0, [rdi + r11 + BLOCK]
    vpmovmskb eax, ymm3
    vpmovmskb ecx, ymm4
    shl     rcx, 32
    or      rax, rcx
    not     rax
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
    lea     rbp, [r12 + rbp + WIDTHS_AT] # the width's table, if it has one

    # The block's first element is at row r13, column r15.
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
    mov     eax, 2 * BLOCK
    xor     edx, edx
    cmp     r10, 2 * BLOCK
    ja      .Lpair_steps
    div     r10d
    xchg    eax, edx
.Lpair_steps:
    mov     [rsp + PAIR_COL], rax
    not     rdx
    mov     [rsp + PAIR_ROW], rdx
    cmp     r10, TABLE_MAX
    ja      .Lwide_rows

    # Rows of up to 64 elements: tables. Two loop heads: a window of two
    # blocks while they are sparse, one block while they are dense.
    vmovdqa ymm8, [rip + .Lspread_table]
    vmovdqa ymm9, [rip + .Lspread_table + 32]
    vmovdqa ymm10, [rip + .Lspread_table + 64]
    vmovdqa ymm11, [rip + .Lspread_table + 96]
.p2align 5
.Ltable_pair:
    lea     rax, [r11 + BLOCK]
    cmp     rax, rbx
    ja      .Ltable_block               # no room for two blocks
    PAIR_WORD
    popcnt  rcx, rax
    lea     r14, [rbp + r15 * 2]        # the table at the window's column
    vmovq   xmm1, r13
    cmp     ecx, PAIR_MAX
    ja      .Ltable_split
    TABLE_SPARSE 0, rax, r8
    TABLE_SPARSE 1, rax, r8
    TABLE_SPARSE 2, rax, r8
    TABLE_SPARSE 3, rax, r8
    TABLE_SPARSE 4, rax, r8
    TABLE_SPARSE 5, rax, r8
    lea     r9, [r9 + rcx * 2]
    CARRY   PAIR_COL, PAIR_ROW
    add     r11, 2 * BLOCK
    jmp     .Ltable_pair
.Ltable_split:
    popcnt  ecx, eax                    # the first block alone
    jmp     .Ltable_word
.Ltable_block:
    cmp     r11, rbx
    ja      .Lwide_tail
.p2align 5
.Ltable_next:
    BLOCK_WORD
    popcnt  ecx, eax
    lea     r14, [rbp + r15 * 2]        # the table at the block's column
    vmovq   xmm1, r13
.Ltable_word:
    cmp     ecx, SPARSE_MAX
    ja      .Ltable_dense
    TABLE_SPARSE 0
    TABLE_SPARSE 1
    TABLE_SPARSE 2
    TABLE_SPARSE 3
    lea     r9, [r9 + rcx * 2]
    CARRY   STEP_COL, STEP_ROW
    add     r11, BLOCK
    jmp     .Ltable_pair
.Ltable_dense:
    vmovq   xmm1, r13
    vinserti128 ymm1, ymm1, xmm1, 1
    TABLE_GROUP 0
    TABLE_GROUP 1
    TABLE_GROUP 2
    TABLE_GROUP 3
    CARRY   STEP_COL, STEP_ROW
    add     r11, BLOCK
    cmp     r11, rbx
    jbe     .Ltable_next
    jmp     .Lwide_tail

    # Rows of more than 64 elements: a window crosses at most one row end.
.Lwide_rows:
    vmovdqa ymm8, [rip + .Lspread_wide]
    vmovdqa ymm9, [rip + .Lspread_wide + 32]
    vmovdqa ymm10, [rip + .Lspread_wide + 64]
    vmovdqa ymm11, [rip + .Lspread_wide + 96]
    mov     eax, 1
    vmovq   xmm12, rax
    mov     rax, r10
    neg     rax
    vpinsrq xmm12, xmm12, rax, 1
    vinserti128 ymm12, ymm12, xmm12, 1  # (1, -width, 1, -width)
    vmovdqa ymm13, [rip + .Leight]
    vpaddq  ymm14, ymm12, ymm13
.p2align 5
.Lwide_pair:
    lea     rax, [r11 + BLOCK]
    cmp     rax, rbx
    ja      .Lwide_block
    PAIR_WORD
    popcnt  rcx, rax
    cmp     ecx, PAIR_MAX
    ja      .Lwide_split
    lea     r8, [r15 + 2 * BLOCK]
    cmp     r8, r10
    ja      .Lwide_pair_cross
    ROW_SPARSE 0, rax, r8
    ROW_SPARSE 1, rax, r8
    ROW_SPARSE 2, rax, r8
    ROW_SPARSE 3, rax, r8
    ROW_SPARSE 4, rax, r8
    ROW_SPARSE 5, rax, r8
.Lwide_pair_advance:
    lea     r9, [r9 + rcx * 2]
    CARRY   PAIR_COL, PAIR_ROW
    add     r11, 2 * BLOCK
    jmp     .Lwide_pair
.Lwide_pair_cross:
    CROSS_SPARSE 0, rax, r8
    CROSS_SPARSE 1, rax, r8
    CROSS_SPARSE 2, rax, r8
    CROSS_SPARSE 3, rax, r8
    CROSS_SPARSE 4, rax, r8
    CROSS_SPARSE 5, rax, r8
    jmp     .Lwide_pair_advance
.Lwide_split:
    popcnt  ecx, eax
    jmp     .Lwide_word
.Lwide_block:
    cmp     r11, rbx
    ja      .Lwide_tail
.p2align 5
.Lwide_next:
    BLOCK_WORD
    popcnt  ecx, eax
.Lwide_word:
    cmp     ecx, SPARSE_MAX
    ja      .Lwide_dense
    lea     r8, [r15 + BLOCK]
    cmp     r8, r10
    ja      .Lwide_sparse_cross
    ROW_SPARSE 0
    ROW_SPARSE 1
    ROW_SPARSE 2
    ROW_SPARSE 3
.Lwide_advance:
    lea     r9, [r9 + rcx * 2]
    CARRY   STEP_COL, STEP_ROW
    add     r11, BLOCK
    jmp     .Lwide_pair
.Lwide_sparse_cross:
    CROSS_SPARSE 0
    CROSS_SPARSE 1
    CROSS_SPARSE 2
    CROSS_SPARSE 3
    jmp     .Lwide_advance

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
.Lwide_dense_advance:
    CARRY   STEP_COL, STEP_ROW
    add     r11, BLOCK
    cmp     r11, rbx
    jbe     .Lwide_next
    jmp     .Lwide_tail
.Lwide_dense_cross:
    mov     r8, r15
    WIDE_GROUP 0
    WIDE_GROUP 1
    WIDE_GROUP 2
    WIDE_GROUP 3
    jmp     .Lwide_dense_advance
    WIDE_CROSS 0
    WIDE_CROSS 1
    WIDE_CROSS 2
    WIDE_CROSS 3

.Lwide_tail:
    mov     rcx, r13
    mov     r8, r15
    jmp     .Ltail

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
