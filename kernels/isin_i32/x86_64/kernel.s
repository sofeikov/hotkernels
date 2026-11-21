.intel_syntax noprefix
# void isin_i32(const int32_t *x, const int32_t *test, bool *out, size_t n);
#
# rdi = x (n values), rsi = test (exactly 8 values), rdx = out (n bytes),
# rcx = n. out[i] = 1 when x[i] equals any test value, else 0.
#
# Thirty-two elements per block. How a block is done is chosen once per call
# from the test values.
#
# Narrow keys, when the largest test value exceeds the smallest by at most
# 65,533. Every element x becomes the key c - x, with c = smallest + 32,766
# and the subtraction wrapping at 32 bits; the test values become keys from
# -32,767 to 32,766. Subtraction from c is one-to-one on 32-bit integers, so
# an element equals a test value exactly when their keys are equal. The keys
# are narrowed to 16 bits with `vpackssdw`, which saturates: a key in the
# test values' range is unchanged, and any other becomes -32,768 or 32,767,
# which is no test key. Sixteen 16-bit keys in a register are compared with
# the eight test keys in eight `vpcmpeqw`, where 32-bit compares need eight
# for eight elements.
#
# Wide, for any other test values: each vector of eight elements is compared
# with the eight test values at 32 bits, the values kept broadcast in
# registers.
#
# Both leave the 32 results packed to bytes by `vpacksswb` in an order the
# packing instructions produce (they work inside each 128-bit half): groups
# of four elements starting at 0 8 16 24 in the low half and at 4 12 20 28
# in the high half. One `vpermd` puts them in element order and
# `vpabsb` turns 0xff into 1.
#
# The leftover (n mod 32) is one more block that ends exactly at the end of
# the arrays and overlaps the one before it. From 8 to 31 elements are done
# eight at a time at 32 bits, the last eight overlapping, and fewer than 8
# one at a time, each against the eight test values in one compare.
#
# Registers: r9 index of the block's first element, r10 index of the last
# block. ymm8-ymm15 the eight test values, as 16-bit keys broadcast to every
# lane on the narrow path and 32-bit values on the wide one; ymm6 c in every
# 32-bit lane; ymm7 the order for `vpermd`.

# Largest spread of the test values for which their keys fit -32,767 .. 32,766.
.set NARROW_SPREAD, 65533
# c - smallest test value: the largest test key.
.set NARROW_TOP, 32766
# Block sizes, in elements.
.set BLOCK, 32
.set SMALL_BLOCK, 8

# One block of 32 elements from r9 on, narrow keys.
.macro narrow_block
    vpsubd  ymm0, ymm6, [rdi + r9 * 4]
    vpsubd  ymm1, ymm6, [rdi + r9 * 4 + 32]
    vpsubd  ymm2, ymm6, [rdi + r9 * 4 + 64]
    vpsubd  ymm3, ymm6, [rdi + r9 * 4 + 96]
    vpackssdw ymm0, ymm0, ymm1          # keys of elements 0-15
    vpackssdw ymm2, ymm2, ymm3          # keys of elements 16-31
    vpcmpeqw ymm1, ymm0, ymm8
    vpcmpeqw ymm3, ymm2, ymm8
    vpcmpeqw ymm4, ymm0, ymm9
    vpcmpeqw ymm5, ymm2, ymm9
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpcmpeqw ymm4, ymm0, ymm10
    vpcmpeqw ymm5, ymm2, ymm10
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpcmpeqw ymm4, ymm0, ymm11
    vpcmpeqw ymm5, ymm2, ymm11
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpcmpeqw ymm4, ymm0, ymm12
    vpcmpeqw ymm5, ymm2, ymm12
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpcmpeqw ymm4, ymm0, ymm13
    vpcmpeqw ymm5, ymm2, ymm13
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpcmpeqw ymm4, ymm0, ymm14
    vpcmpeqw ymm5, ymm2, ymm14
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpcmpeqw ymm4, ymm0, ymm15
    vpcmpeqw ymm5, ymm2, ymm15
    vpor    ymm1, ymm1, ymm4
    vpor    ymm3, ymm3, ymm5
    vpacksswb ymm1, ymm1, ymm3
    vpermd  ymm1, ymm7, ymm1            # element order
    vpabsb  ymm1, ymm1                  # 0xff to 1
    vmovdqu [rdx + r9], ymm1
.endm

# One block of 32 elements from r9 on, 32-bit compares.
.macro wide_block
    vmovdqu ymm4, [rdi + r9 * 4]
    vpcmpeqd ymm0, ymm4, ymm8
    vpcmpeqd ymm6, ymm4, ymm9
    vpor    ymm0, ymm0, ymm6
    vpcmpeqd ymm5, ymm4, ymm10
    vpor    ymm0, ymm0, ymm5
    vpcmpeqd ymm6, ymm4, ymm11
    vpor    ymm0, ymm0, ymm6
    vpcmpeqd ymm5, ymm4, ymm12
    vpor    ymm0, ymm0, ymm5
    vpcmpeqd ymm6, ymm4, ymm13
    vpor    ymm0, ymm0, ymm6
    vpcmpeqd ymm5, ymm4, ymm14
    vpor    ymm0, ymm0, ymm5
    vpcmpeqd ymm6, ymm4, ymm15
    vpor    ymm0, ymm0, ymm6
    vmovdqu ymm4, [rdi + r9 * 4 + 32]
    vpcmpeqd ymm1, ymm4, ymm8
    vpcmpeqd ymm6, ymm4, ymm9
    vpor    ymm1, ymm1, ymm6
    vpcmpeqd ymm5, ymm4, ymm10
    vpor    ymm1, ymm1, ymm5
    vpcmpeqd ymm6, ymm4, ymm11
    vpor    ymm1, ymm1, ymm6
    vpcmpeqd ymm5, ymm4, ymm12
    vpor    ymm1, ymm1, ymm5
    vpcmpeqd ymm6, ymm4, ymm13
    vpor    ymm1, ymm1, ymm6
    vpcmpeqd ymm5, ymm4, ymm14
    vpor    ymm1, ymm1, ymm5
    vpcmpeqd ymm6, ymm4, ymm15
    vpor    ymm1, ymm1, ymm6
    vmovdqu ymm4, [rdi + r9 * 4 + 64]
    vpcmpeqd ymm2, ymm4, ymm8
    vpcmpeqd ymm6, ymm4, ymm9
    vpor    ymm2, ymm2, ymm6
    vpcmpeqd ymm5, ymm4, ymm10
    vpor    ymm2, ymm2, ymm5
    vpcmpeqd ymm6, ymm4, ymm11
    vpor    ymm2, ymm2, ymm6
    vpcmpeqd ymm5, ymm4, ymm12
    vpor    ymm2, ymm2, ymm5
    vpcmpeqd ymm6, ymm4, ymm13
    vpor    ymm2, ymm2, ymm6
    vpcmpeqd ymm5, ymm4, ymm14
    vpor    ymm2, ymm2, ymm5
    vpcmpeqd ymm6, ymm4, ymm15
    vpor    ymm2, ymm2, ymm6
    vmovdqu ymm4, [rdi + r9 * 4 + 96]
    vpcmpeqd ymm3, ymm4, ymm8
    vpcmpeqd ymm6, ymm4, ymm9
    vpor    ymm3, ymm3, ymm6
    vpcmpeqd ymm5, ymm4, ymm10
    vpor    ymm3, ymm3, ymm5
    vpcmpeqd ymm6, ymm4, ymm11
    vpor    ymm3, ymm3, ymm6
    vpcmpeqd ymm5, ymm4, ymm12
    vpor    ymm3, ymm3, ymm5
    vpcmpeqd ymm6, ymm4, ymm13
    vpor    ymm3, ymm3, ymm6
    vpcmpeqd ymm5, ymm4, ymm14
    vpor    ymm3, ymm3, ymm5
    vpcmpeqd ymm6, ymm4, ymm15
    vpor    ymm3, ymm3, ymm6
    vpackssdw ymm0, ymm0, ymm1
    vpackssdw ymm2, ymm2, ymm3
    vpacksswb ymm0, ymm0, ymm2
    vpermd  ymm0, ymm7, ymm0            # element order
    vpabsb  ymm0, ymm0                  # 0xff to 1
    vmovdqu [rdx + r9], ymm0
.endm

.section .rodata
.p2align 5
.Lorder:
    .long 0, 4, 1, 5, 2, 6, 3, 7

.text
.globl isin_i32
.type isin_i32, @function
.p2align 4
isin_i32:
    cmp     rcx, BLOCK
    jb      .Lsmall
    xor     r9d, r9d
    lea     r10, [rcx - BLOCK]
    vmovdqa ymm7, [rip + .Lorder]
    # Smallest and largest test value, in every lane of xmm1 and xmm2.
    vmovdqu ymm0, [rsi]
    vextracti128 xmm3, ymm0, 1
    vpminsd xmm1, xmm0, xmm3
    vpmaxsd xmm2, xmm0, xmm3
    vpshufd xmm3, xmm1, 0x4e
    vpminsd xmm1, xmm1, xmm3
    vpshufd xmm3, xmm1, 0xb1
    vpminsd xmm1, xmm1, xmm3
    vpshufd xmm3, xmm2, 0x4e
    vpmaxsd xmm2, xmm2, xmm3
    vpshufd xmm3, xmm2, 0xb1
    vpmaxsd xmm2, xmm2, xmm3
    vmovd   eax, xmm1
    vmovd   r8d, xmm2
    movsxd  rax, eax
    movsxd  r8, r8d
    sub     r8, rax                     # spread, at most 2^32 - 1
    cmp     r8, NARROW_SPREAD
    ja      .Lwidesetup
    add     eax, NARROW_TOP             # c, wrapping at 32 bits
    vmovd   xmm6, eax
    vpbroadcastd ymm6, xmm6
    vpsubd  ymm0, ymm6, ymm0            # test keys, -32,767 .. 32,766
    vmovdqu [rsp - 32], ymm0            # red zone, to broadcast each key
    vpbroadcastw ymm8, word ptr [rsp - 32]
    vpbroadcastw ymm9, word ptr [rsp - 28]
    vpbroadcastw ymm10, word ptr [rsp - 24]
    vpbroadcastw ymm11, word ptr [rsp - 20]
    vpbroadcastw ymm12, word ptr [rsp - 16]
    vpbroadcastw ymm13, word ptr [rsp - 12]
    vpbroadcastw ymm14, word ptr [rsp - 8]
    vpbroadcastw ymm15, word ptr [rsp - 4]

.p2align 6
.Lnarrow:
    narrow_block
    add     r9, BLOCK
    cmp     r9, r10
    jbe     .Lnarrow
    sub     r9, BLOCK
    cmp     r9, r10
    je      .Ldone                      # the blocks ended with the last element
    mov     r9, r10                     # one more block, ending with the last element
    narrow_block

.Ldone:
    vzeroupper
    ret

    # ---- A wide call.
.Lwidesetup:
    vpbroadcastd ymm8, dword ptr [rsi + 0]
    vpbroadcastd ymm9, dword ptr [rsi + 4]
    vpbroadcastd ymm10, dword ptr [rsi + 8]
    vpbroadcastd ymm11, dword ptr [rsi + 12]
    vpbroadcastd ymm12, dword ptr [rsi + 16]
    vpbroadcastd ymm13, dword ptr [rsi + 20]
    vpbroadcastd ymm14, dword ptr [rsi + 24]
    vpbroadcastd ymm15, dword ptr [rsi + 28]
.p2align 6
.Lwide:
    wide_block
    add     r9, BLOCK
    cmp     r9, r10
    jbe     .Lwide
    sub     r9, BLOCK
    cmp     r9, r10
    je      .Ldone                      # the blocks ended with the last element
    mov     r9, r10                     # one more block, ending with the last element
    wide_block
    jmp     .Ldone

    # ---- Fewer than 32 elements.
.Lsmall:
    cmp     rcx, SMALL_BLOCK
    jb      .Lfew
    xor     r9d, r9d
    lea     r10, [rcx - SMALL_BLOCK]
    vpbroadcastd ymm8, dword ptr [rsi + 0]
    vpbroadcastd ymm9, dword ptr [rsi + 4]
    vpbroadcastd ymm10, dword ptr [rsi + 8]
    vpbroadcastd ymm11, dword ptr [rsi + 12]
    vpbroadcastd ymm12, dword ptr [rsi + 16]
    vpbroadcastd ymm13, dword ptr [rsi + 20]
    vpbroadcastd ymm14, dword ptr [rsi + 24]
    vpbroadcastd ymm15, dword ptr [rsi + 28]
.Leight:
    vmovdqu ymm4, [rdi + r9 * 4]
    vpcmpeqd ymm0, ymm4, ymm8
    vpcmpeqd ymm1, ymm4, ymm9
    vpcmpeqd ymm2, ymm4, ymm10
    vpcmpeqd ymm3, ymm4, ymm11
    vpor    ymm0, ymm0, ymm1
    vpor    ymm2, ymm2, ymm3
    vpcmpeqd ymm1, ymm4, ymm12
    vpcmpeqd ymm3, ymm4, ymm13
    vpor    ymm0, ymm0, ymm1
    vpor    ymm2, ymm2, ymm3
    vpcmpeqd ymm1, ymm4, ymm14
    vpcmpeqd ymm3, ymm4, ymm15
    vpor    ymm0, ymm0, ymm1
    vpor    ymm2, ymm2, ymm3
    vpor    ymm0, ymm0, ymm2
    vextracti128 xmm1, ymm0, 1
    vpackssdw xmm0, xmm0, xmm1
    vpacksswb xmm0, xmm0, xmm0
    vpabsb  xmm0, xmm0
    vmovq   [rdx + r9], xmm0
    cmp     r9, r10
    jae     .Ldone
    add     r9, SMALL_BLOCK
    cmp     r9, r10
    cmova   r9, r10                     # the last eight end with the last element
    jmp     .Leight

.Lfew:
    test    rcx, rcx
    jz      .Lreturn
    xor     r9d, r9d
    vmovdqu ymm1, [rsi]
.Lone:
    vpbroadcastd ymm0, dword ptr [rdi + r9 * 4]
    vpcmpeqd ymm0, ymm0, ymm1
    vptest  ymm0, ymm0
    setnz   byte ptr [rdx + r9]
    inc     r9
    cmp     r9, rcx
    jb      .Lone
    vzeroupper
.Lreturn:
    ret
.size isin_i32, .-isin_i32

.section .note.GNU-stack,"",@progbits
