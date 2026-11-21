.intel_syntax noprefix
# void maskfill_f32(float *x, const bool *mask, size_t n, float value);
#
# rdi = x, rsi = mask, rdx = n, xmm0 = value. Writes value to x[i] wherever
# mask[i] is not zero. x is read and modified in place.
#
# Eight elements at a time: `vpmovsxbd` widens eight mask bytes to eight
# 32-bit lanes, a compare with zero turns them into a lane mask, and the eight
# elements are loaded, blended with the value and stored back. An element
# that is not selected is rewritten with its own bits, which the spec allows
# for a buffer that is read and written in place. The value's bits are copied
# and never computed with, so a NaN, a signed zero or a denormal arrives
# unchanged. A mask byte is true when it is not zero, whatever its value.
#
# A block of 32 with at most SPARSE_MAX selected elements is not blended.
# `tzcnt` and `blsr` take the positions out of its mask word, and the value
# is stored to each: no element is loaded, and a cache line with no selected
# element is not touched at all.
#
# Paths, by what is left:
#   32 or more  blocks of 32: one mask word, then four single stores or four
#               blends
#   8..31       one blend per 8
#   1..7        one element at a time, with a branch

# A block with more selected elements than this is blended.
.set SPARSE_MAX, 4

# Blend the eight elements at x + \xoff by the eight mask bytes at
# mask + \moff. ymm0 = the value in every lane, ymm1 = zeros.
.macro BLEND moff, xoff
    vpmovsxbd ymm2, qword ptr [rsi + \moff]
    vpcmpeqd ymm2, ymm2, ymm1           # all ones where the element stays
    vpand   ymm3, ymm2, [rdi + \xoff]
    vpandn  ymm2, ymm2, ymm0
    vpor    ymm3, ymm3, ymm2
    vmovdqu [rdi + \xoff], ymm3
.endm

.text
.globl maskfill_f32
.type maskfill_f32, @function
.p2align 4
maskfill_f32:
    cmp     rdx, 8
    jb      .Ltail
    vbroadcastss ymm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    mov     rcx, rdx
    shr     rcx, 5                      # blocks of 32
    jz      .Lgroups
    lea     r9, [rsp - 16]              # four bytes of our own, below the stack pointer
.p2align 6
.Lblock:
    vpcmpeqb ymm2, ymm1, [rsi]          # 0xff where the mask byte is zero
    vpmovmskb eax, ymm2
    not     eax                         # one bit per selected element
    popcnt  r8d, eax
    cmp     r8d, SPARSE_MAX
    ja      .Ldense
    # Sparse block: a fixed SPARSE_MAX stores and no load. `tzcnt` sets the
    # carry flag when no bit is left. The store then goes where the one
    # before it went, writing the value to a selected element a second time,
    # or, in a block with no bit at all, to the bytes below the stack pointer.
    mov     r10, r9
    tzcnt   r8d, eax
    lea     r11, [rdi + r8 * 4]
    cmovnc  r10, r11
    vmovss  [r10], xmm0
    blsr    eax, eax
    tzcnt   r8d, eax
    lea     r11, [rdi + r8 * 4]
    cmovnc  r10, r11
    vmovss  [r10], xmm0
    blsr    eax, eax
    tzcnt   r8d, eax
    lea     r11, [rdi + r8 * 4]
    cmovnc  r10, r11
    vmovss  [r10], xmm0
    blsr    eax, eax
    tzcnt   r8d, eax
    lea     r11, [rdi + r8 * 4]
    cmovnc  r10, r11
    vmovss  [r10], xmm0
    add     rsi, 32
    sub     rdi, -128
    dec     rcx
    jnz     .Lblock
    jmp     .Lgroups
.p2align 4
.Ldense:
    BLEND   0, 0
    BLEND   8, 32
    BLEND   16, 64
    BLEND   24, 96
    add     rsi, 32
    sub     rdi, -128
    dec     rcx
    jnz     .Lblock

.Lgroups:
    and     edx, 31
    cmp     edx, 8
    jb      .Ltailv
.Lgroup:
    BLEND   0, 0
    add     rsi, 8
    add     rdi, 32
    sub     edx, 8
    cmp     edx, 8
    jae     .Lgroup
.Ltailv:
    vmovd   eax, xmm0                   # the value's bits
    vzeroupper
    jmp     .Ltailbits

.Ltail:
    vmovd   eax, xmm0
.Ltailbits:
    test    edx, edx
    jz      .Ldone
.Lone:
    cmp     byte ptr [rsi], 0
    je      .Lskip
    mov     [rdi], eax
.Lskip:
    add     rdi, 4
    inc     rsi
    dec     edx
    jnz     .Lone
.Ldone:
    ret
.size maskfill_f32, .-maskfill_f32

.section .note.GNU-stack,"",@progbits
