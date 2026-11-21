.intel_syntax noprefix
# size_t compress_u8(const uint8_t *x, const bool *mask, uint8_t *out, size_t n);
#
# rdi = x, rsi = mask, rdx = out, rcx = n. Copies x[i] to out wherever mask[i]
# is not zero, in order, and returns how many were copied. `out` is a scratch
# buffer of n elements: stores may land anywhere inside it, and only the
# returned count of elements has to be right. Serves uint8, int8 and bool by
# their bits.
#
# Eight elements at a time. The eight mask bytes become eight bits, the bits
# index a table of 256 byte shuffles, and `vpshufb` moves the selected elements
# to the front of the register in one instruction. All eight bytes are stored
# at the write position, which then advances by the number selected, so the
# next store overwrites the bytes that were not. A store of eight bytes at
# position k covers out[k .. k+8), and k is never ahead of the read position,
# so it stays inside the n elements as long as the eight elements read do.
#
# The mask bits for 32 elements come from one compare and one `vpmovmskb`. A
# mask byte counts as true when it is not zero, whatever its value.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, four table steps
#   8..31       one table step per 8, the mask bits from an 8-byte load
#   1..7        one element at a time, as the reference does it

# Offset of a table entry is 8 times the mask byte. The mask word is shifted
# left by this once per block, so that a group's offset is one `and` away.
.set ENTRY_SHIFT, 3
# The bits of the shifted mask word that hold one group's offset.
.set ENTRY_BITS, 0xff << ENTRY_SHIFT

# One entry of the shuffle table: for mask byte \m, the indices of its set bits
# in ascending order, one byte each. The bytes past the count are zero: they
# select element 0 of the group, which lands beyond the write position's
# advance and is overwritten or ignored.
.macro PERM_ENTRY m
    .set .Lcount, 0
    .irp bit, 0, 1, 2, 3, 4, 5, 6, 7
    .if ((\m) >> \bit) & 1
    .byte \bit
    .set .Lcount, .Lcount + 1
    .endif
    .endr
    .fill 8 - .Lcount, 1, 0
.endm

.section .rodata
.p2align 6
.Lperm:
    .set .Lbyte, 0
    .rept 256
    PERM_ENTRY .Lbyte
    .set .Lbyte, .Lbyte + 1
    .endr
    # `vpshufb` reads 16 bytes of its control, eight more than an entry; the
    # last entry's are this padding. Their bytes only shape lanes not stored.
    .fill 8, 1, 0

# One table step: compact the eight elements at \src by the group whose table
# offset is in r8, store them at byte r9 of the output and advance r9 by the
# number selected.
.macro STEP src
    vmovq   xmm2, \src
    vpshufb xmm2, xmm2, [r10 + r8]
    vmovq   [rdx + r9], xmm2
    popcnt  r8d, r8d
    add     r9, r8
.endm

.text
.globl compress_u8
.type compress_u8, @function
.p2align 4
compress_u8:
    xor     r9d, r9d                    # elements written
    lea     r10, [rip + .Lperm]
    vpxor   xmm0, xmm0, xmm0
    mov     r11, rcx
    shr     r11, 5                      # blocks of 32
    jz      .Lgroups
.p2align 6
.Lblock:
    vpcmpeqb ymm1, ymm0, [rsi]          # 0xff where the mask byte is zero
    vpmovmskb eax, ymm1
    not     eax                         # one bit per selected element
    shl     rax, ENTRY_SHIFT
    mov     r8d, eax
    and     r8d, ENTRY_BITS
    STEP    [rdi]
    rorx    r8, rax, 8
    and     r8d, ENTRY_BITS
    STEP    [rdi + 8]
    rorx    r8, rax, 16
    and     r8d, ENTRY_BITS
    STEP    [rdi + 16]
    shr     rax, 24
    and     eax, ENTRY_BITS
    vmovq   xmm2, [rdi + 24]
    vpshufb xmm2, xmm2, [r10 + rax]
    vmovq   [rdx + r9], xmm2
    popcnt  eax, eax
    add     r9, rax
    add     rsi, 32
    add     rdi, 32
    dec     r11
    jnz     .Lblock

.Lgroups:
    and     ecx, 31
    cmp     ecx, 8
    jb      .Ltail
.Lgroup:
    vmovq   xmm1, [rsi]
    vpcmpeqb xmm1, xmm1, xmm0
    vpmovmskb eax, xmm1
    not     eax
    movzx   r8d, al
    shl     r8d, ENTRY_SHIFT
    STEP    [rdi]
    add     rsi, 8
    add     rdi, 8
    sub     ecx, 8
    cmp     ecx, 8
    jae     .Lgroup

.Ltail:
    test    ecx, ecx
    jz      .Ldone
.Lone:
    movzx   eax, byte ptr [rdi]
    mov     [rdx + r9], al
    xor     eax, eax
    cmp     byte ptr [rsi], 0
    setne   al
    add     r9, rax
    inc     rdi
    inc     rsi
    dec     ecx
    jnz     .Lone
.Ldone:
    mov     rax, r9
    vzeroupper
    ret
.size compress_u8, .-compress_u8

.section .note.GNU-stack,"",@progbits
