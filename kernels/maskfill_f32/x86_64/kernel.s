.intel_syntax noprefix
# void maskfill_f32(float *x, const bool *mask, size_t n, float value);
#
# rdi = x, rsi = mask, rdx = n, xmm0 = value. Writes value to x[i] wherever
# mask[i] is not zero. x is read and modified in place.
#
# Eight elements at a time: `vpmovzxbd` widens eight mask bytes to eight
# 32-bit lanes, a signed compare with zero sets every bit of the lanes whose
# byte is not zero, and `vmaskmovps` stores the value to those lanes only.
# No element of x is loaded, and an element the mask does not select is not
# written. The value's bits are copied and never computed with, so a NaN, a
# signed zero or a denormal arrives unchanged. A mask byte is true when it is
# not zero, whatever its value.
#
# Paths, by n:
#   8 or more   blocks of 32, then groups of 8, then one group of 8 that ends
#               at the end of the array and overlaps the one before it
#   1..7        one element at a time, with a branch

# Store the value to the eight elements at x + \xoff selected by the eight
# mask bytes at mask + \moff. ymm0 = the value in every lane, ymm1 = zeros.
.macro FILL moff, xoff, reg
    vpmovzxbd \reg, qword ptr [rsi + \moff]
    vpcmpgtd \reg, \reg, ymm1           # all ones where the element is selected
    vmaskmovps [rdi + \xoff], \reg, ymm0
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
    lea     r8, [rsi + rdx - 8]         # the last eight mask bytes
    lea     r9, [rdi + rdx * 4 - 32]    # the last eight elements
    mov     rcx, rdx
    shr     rcx, 5                      # blocks of 32
    jz      .Lgroups
.p2align 5
.Lblock:
    FILL    0, 0, ymm2
    FILL    8, 32, ymm3
    FILL    16, 64, ymm4
    FILL    24, 96, ymm5
    add     rsi, 32
    sub     rdi, -128
    dec     rcx
    jnz     .Lblock
.Lgroups:
    and     edx, 31
    cmp     edx, 8
    jb      .Llast
.Lgroup:
    FILL    0, 0, ymm2
    add     rsi, 8
    add     rdi, 32
    sub     edx, 8
    cmp     edx, 8
    jae     .Lgroup
.Llast:
    # The last eight elements, some of them already done: storing the value
    # to a selected element a second time changes nothing.
    vpmovzxbd ymm2, qword ptr [r8]
    vpcmpgtd ymm2, ymm2, ymm1
    vmaskmovps [r9], ymm2, ymm0
    vzeroupper
    ret

.Ltail:
    vmovd   eax, xmm0                   # the value's bits
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
