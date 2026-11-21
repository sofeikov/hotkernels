.intel_syntax noprefix
# void any_cols_bool(const bool *m, bool *out, size_t n, size_t cols);
#
# rdi = m, rsi = out, rdx = n, rcx = cols. m holds n / cols whole rows of
# cols elements; out[c] = 1 if any row holds a non-zero byte in column c, else
# 0. Writes exactly cols bytes, all 0 when there is no whole row. cols >= 1.
# A byte of m other than 0 and 1 counts as true on every path.
#
# The results are or'ed together in registers and made 0 or 1 once, by a
# minimum with 1, when they are stored. The rows are counted by address: a
# pass ends at the last address a whole row starts at. Only the narrow path
# divides, for the bytes in whole rows, which is where its loads end.
#
# Paths, by cols:
#   32..     strips of 32 columns, each with a register of its own for the
#            whole pass over the rows: eight strips in a pass while 256
#            columns are left, then four, two and one. The columns left under
#            32 are done by one more strip that ends at the row's end and
#            overlaps the one before. A pass of one strip takes two rows per
#            step into two registers.
#   1..31    one register for the whole call. A 32-byte load at a row's start
#            holds `step` bytes of whole rows, as many as fit in 32, and the
#            loads are `step` apart; the bytes above `step` belong to rows
#            that have a load of their own and are never looked at. The rows
#            that start in the last 31 bytes of m cannot be loaded that way:
#            the last 32 bytes of m are loaded once and moved down to where a
#            load at the first of those rows would have put them, with zeros
#            above. Then the rows inside the register are folded onto the
#            first, and the first cols bytes are the results. A matrix under
#            32 bytes is put in the register from two loads that overlap.
#
# Moving the bytes of a register down by a count is two `vpshufb`, one for
# the bytes that stay in their half and one for those that cross from the
# upper half to the lower, with index vectors read from .Lshift at the count.
# The results of a narrow matrix are stored the same way: the first 16, 8 or
# 4 and the last 16, 8 or 4, which overlap.

.section .rodata
.p2align 6
# The 16 bytes at .Lshift + 16 + k are the `vpshufb` indices that move the
# bytes of a 16-byte half down by k, with zeros moved in (0x80 gives zero);
# the 16 at .Lshift + k move the bytes of the other half that cross into it.
# k is 0 to 32.
.Lshift:
    .fill 16, 1, 0x80
    .byte 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    .fill 48, 1, 0x80
# For cols of 1 to 31, the bytes of a 32-byte vector that are whole rows:
# cols * (32 / cols). The entry for 0 is not used.
.Lstep:
    .byte 0, 32, 32, 30, 32, 30, 30, 28, 32, 27, 30, 22, 24, 26, 28, 30
    .byte 32, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
# Where .Lstep lies from .Lshift, so that a register holding the address of
# .Lshift reaches both tables. It is a constant of its own because clang's
# assembler, which builds the wheels, takes one symbol in a memory operand and
# refuses the difference of two written there; GNU as, the judge's, takes both.
.set STEP_AT, .Lstep - .Lshift

# Bytes in one vector, which is the columns in one strip.
.set VEC, 32

# One pass over the rows for the strips listed, which are numbered from 0 and
# kept in the ymm registers of the same numbers: strip i is the 32 columns
# from r9 + 32 * i. rdi = m, rsi = out, rax = where the last whole row starts,
# and there is one, rcx = cols, ymm15 = 1 in every byte. Uses rdx and r11.
.macro PASS label, strips:vararg
\label:
    lea     rdx, [rdi + r9]
    lea     r11, [rax + r9]             # the strip in the last row
  .irp i, \strips
    vpxor   xmm\i, xmm\i, xmm\i
  .endr
\label\()_loop:
  .irp i, \strips
    vpor    ymm\i, ymm\i, [rdx + \i * VEC]
  .endr
    add     rdx, rcx
    cmp     rdx, r11
    jbe     \label\()_loop
  .irp i, \strips
    vpminub ymm\i, ymm\i, ymm15
    vmovdqu [rsi + r9 + \i * VEC], ymm\i
  .endr
.endm

# ymm0 |= \src moved down by \count bytes, with zeros moved in. \crossed is
# the upper half of \src in its lower half, with zeros above. r9 = .Lshift.
# Uses ymm4 and ymm5.
.macro OR_MOVED_DOWN src, crossed, count
    vbroadcasti128 ymm4, [r9 + \count + 16]
    vpshufb ymm4, \src, ymm4
    vmovdqu xmm5, [r9 + \count]
    vpshufb ymm5, \crossed, ymm5
    vpor    ymm4, ymm4, ymm5
    vpor    ymm0, ymm0, ymm4
.endm

.text
.globl any_cols_bool
.type any_cols_bool, @function
.p2align 4
any_cols_bool:
    vpcmpeqd ymm15, ymm15, ymm15
    vpabsb  ymm15, ymm15                # 1 in every byte
    cmp     rcx, VEC
    jb      .Lnarrow
    cmp     rdx, rcx
    jb      .Lno_rows
    lea     rax, [rdi + rdx]
    sub     rax, rcx                    # where the last whole row starts
    xor     r9d, r9d                    # columns done
    mov     r10, rcx                    # columns left
    lea     r8, [rcx + rcx]             # two rows
.Lnext:
    cmp     r10, 8 * VEC
    jae     .Lpass8
    cmp     r10, 4 * VEC
    jae     .Lpass4
    cmp     r10, 2 * VEC
    jae     .Lpass2
    test    r10, r10
    jz      .Ldone
    cmp     r10, VEC
    jae     .Lpass1
    lea     r9, [rcx - VEC]             # the strip that ends at the row's end
    mov     r10d, VEC
.Lpass1:
    lea     rdx, [rdi + r9]
    lea     r11, [rax + r9]
    sub     r11, rcx                    # the strip in the last row but one; below m when there is one row, so the compares are signed
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    cmp     rdx, r11
    jg      .Lpass1_one
.Lpass1_loop:
    vpor    ymm0, ymm0, [rdx]
    vpor    ymm1, ymm1, [rdx + rcx]
    add     rdx, r8
    cmp     rdx, r11
    jle     .Lpass1_loop
    vpor    ymm0, ymm0, ymm1
.Lpass1_one:
    sub     rdx, rcx
    cmp     rdx, r11
    jg      .Lpass1_store
    vpor    ymm0, ymm0, [rdx + rcx]     # a row without a pair
.Lpass1_store:
    vpminub ymm0, ymm0, ymm15
    vmovdqu [rsi + r9], ymm0
    add     r9, VEC
    sub     r10, VEC
    jmp     .Lnext

    PASS .Lpass2, 0, 1
    add     r9, 2 * VEC
    sub     r10, 2 * VEC
    jmp     .Lnext

    PASS .Lpass4, 0, 1, 2, 3
    add     r9, 4 * VEC
    sub     r10, 4 * VEC
    jmp     .Lnext

    PASS .Lpass8, 0, 1, 2, 3, 4, 5, 6, 7
    add     r9, 8 * VEC
    sub     r10, 8 * VEC
    jmp     .Lnext

# No rows and at least 32 columns: zeros, the last 32 ending at the end.
.Lno_rows:
    vpxor   xmm0, xmm0, xmm0
    lea     rdx, [rsi + rcx - VEC]
.Lno_rows_loop:
    vmovdqu [rsi], ymm0
    add     rsi, VEC
    cmp     rsi, rdx
    jb      .Lno_rows_loop
    vmovdqu [rdx], ymm0
    jmp     .Ldone

# Fewer than 32 columns.
.Lnarrow:
    lea     r9, [rip + .Lshift]
    vpxor   xmm0, xmm0, xmm0
    cmp     rdx, rcx
    jb      .Lnarrow_store              # no rows: zeros
    mov     r8, rdx
    mov     rax, rdx
    xor     edx, edx
    div     rcx
    sub     r8, rdx                     # the bytes in whole rows
    movzx   r10d, byte ptr [r9 + rcx + STEP_AT]     # step
    cmp     r8, VEC
    jb      .Ltiny
    lea     r11, [rdi + r8 - VEC]       # the last address a vector can be loaded from
    mov     rdx, rdi
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    lea     r8, [r10 + r10]             # two steps
    lea     rdi, [r8 + r10]             # three steps
    mov     rax, r11
    sub     rax, rdi                    # the last address four vectors can be loaded from; may be below m, so the compares are signed
    cmp     rdx, rax
    jg      .Lnarrow_ones
.Lnarrow_loop:
    vpor    ymm0, ymm0, [rdx]
    vpor    ymm1, ymm1, [rdx + r10]
    vpor    ymm2, ymm2, [rdx + r8]
    vpor    ymm3, ymm3, [rdx + rdi]
    lea     rdx, [rdx + 4 * r10]
    cmp     rdx, rax
    jle     .Lnarrow_loop
    vpor    ymm0, ymm0, ymm1
    vpor    ymm2, ymm2, ymm3
    vpor    ymm0, ymm0, ymm2
.Lnarrow_ones:
    cmp     rdx, r11
    ja      .Lnarrow_last
.Lnarrow_one:
    vpor    ymm0, ymm0, [rdx]
    add     rdx, r10
    cmp     rdx, r11
    jbe     .Lnarrow_one
.Lnarrow_last:
    # The rows from rdx on, fewer than 32 bytes and maybe none, are the top
    # of the last 32 bytes of m: move them down by rdx - r11, 1 to 32.
    sub     rdx, r11
    vmovdqu ymm2, [r11]
    vperm2i128 ymm3, ymm2, ymm2, 0x81
    OR_MOVED_DOWN ymm2, ymm3, rdx
.Lfold:
    # ymm0 holds step / cols rows, and nothing else below step that is not
    # zero. Or the rows after the first onto it.
    mov     rdx, rcx
    cmp     rdx, r10
    jae     .Lnarrow_store
    vmovdqa ymm2, ymm0
    vperm2i128 ymm3, ymm0, ymm0, 0x81
.Lfold_loop:
    OR_MOVED_DOWN ymm2, ymm3, rdx
    add     rdx, rcx
    cmp     rdx, r10
    jb      .Lfold_loop
.Lnarrow_store:
    vpminub ymm0, ymm0, ymm15
    cmp     rcx, 16
    jae     .Lstore_16
    cmp     rcx, 8
    jae     .Lstore_8
    cmp     rcx, 4
    jae     .Lstore_4
    vmovd   eax, xmm0                   # one to three results
    mov     [rsi], al
    cmp     rcx, 2
    jb      .Ldone
    mov     [rsi + 1], ah
    je      .Ldone
    shr     eax, 16
    mov     [rsi + 2], al
    jmp     .Ldone
.Lstore_4:
    vmovd   [rsi], xmm0
    vpshufb xmm1, xmm0, [r9 + rcx + 12]         # down by cols - 4
    vmovd   [rsi + rcx - 4], xmm1
    jmp     .Ldone
.Lstore_8:
    vmovq   [rsi], xmm0
    vpshufb xmm1, xmm0, [r9 + rcx + 8]          # down by cols - 8
    vmovq   [rsi + rcx - 8], xmm1
    jmp     .Ldone
.Lstore_16:
    vmovdqu [rsi], xmm0
    vextracti128 xmm1, ymm0, 1
    vpshufb xmm2, xmm0, [r9 + rcx]              # down by cols - 16
    vpshufb xmm1, xmm1, [r9 + rcx - 16]
    vpor    xmm1, xmm1, xmm2
    vmovdqu [rsi + rcx - 16], xmm1
.Ldone:
    vzeroupper
    ret

# Fewer than 32 bytes in the matrix, r8 of them, at least one: its first 16,
# 8 or 4 bytes and its last 16, 8 or 4, which overlap, the last moved down to
# follow the first with zeros above. ymm0 = the matrix, then zeros.
.Ltiny:
    cmp     r8, 16
    jae     .Ltiny_16
    cmp     r8, 8
    jae     .Ltiny_8
    cmp     r8, 4
    jae     .Ltiny_4
    movzx   eax, byte ptr [rdi]         # one to three bytes
    cmp     r8, 2
    jb      .Ltiny_small
    movzx   eax, word ptr [rdi]
    je      .Ltiny_small
    movzx   edx, byte ptr [rdi + 2]
    shl     edx, 16
    or      eax, edx
.Ltiny_small:
    vmovd   xmm0, eax
    jmp     .Lfold
.Ltiny_4:
    vmovd   xmm0, dword ptr [rdi]
    vmovd   xmm1, dword ptr [rdi + r8 - 4]
    neg     r8
    vpshufb xmm1, xmm1, [r9 + r8 + 24]          # down by 8 - r8
    vpunpckldq xmm0, xmm0, xmm1
    jmp     .Lfold
.Ltiny_8:
    vmovq   xmm0, qword ptr [rdi]
    vmovq   xmm1, qword ptr [rdi + r8 - 8]
    neg     r8
    vpshufb xmm1, xmm1, [r9 + r8 + 32]          # down by 16 - r8
    vpunpcklqdq xmm0, xmm0, xmm1
    jmp     .Lfold
.Ltiny_16:
    vmovdqu xmm0, [rdi]
    vmovdqu xmm1, [rdi + r8 - 16]
    neg     r8
    vpshufb xmm1, xmm1, [r9 + r8 + 48]          # down by 32 - r8
    vinserti128 ymm0, ymm0, xmm1, 1
    jmp     .Lfold

.section .note.GNU-stack,"",@progbits
