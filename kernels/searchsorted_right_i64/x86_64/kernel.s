.intel_syntax noprefix
# void searchsorted_right_i64(const int64_t *a, const int64_t *v, int64_t *out, size_t n, size_t needles);
#
# rdi = a, rsi = v, rdx = out, rcx = n, r8 = needles. For each needle in v,
# writes the first index i such that the needle is less than a[i], or n if there is none
# (side='right'). a holds n values sorted ascending.
#
# One search is a fixed number of steps that depends only on n, with no
# branch on the data:
#
#     lo = 0, len = n
#     while len > 0:
#         half = (len + 1) / 2
#         if a[lo + half - 1] <= needle: lo += half
#         len -= half
#
# The answer is always in lo .. lo + len, and lo + len never grows, so every
# probe is inside a whatever the data, sorted or not. The comparison is the
# signed one, `vpcmpgtq`; there is no NaN to order, so nothing is fixed
# afterwards.
#
# Up to 64 searches run in step, and up to 512 when a does not fit the
# second-level cache. Their positions live in out, which is exactly one
# 64-bit slot per needle, and their needles stay in v. A step is, per four
# searches, a load of the positions, one `vpgatherqq` for the four probes,
# one compare with the needles, `half` under the compare's mask, an add and a
# store. Nothing in a step waits for another search.
#
# The first step needs no gather: every search probes the same element.
#
# Paths, by the number of needles:
#   0        nothing
#   1..3     one search at a time, in general registers
#   4..      chunks of up to 64 or 512 in step, four per vector; then, if the
#            count is not a multiple of four, one more vector that ends with
#            the last needle and overlaps the one before it
#
# `vpgatherqq` is fast on this processor and slow on some others (older AMD,
# and Intel processors that carry the gather mitigation).
#
# Registers: r9 len, rax the address of a[half - 1], r10 and r11 the byte
# offsets where the chunk starts and ends, rbx the offset inside it (saved
# below the stack pointer). ymm8 half in every lane, ymm9 the gather's mask.

# Needles per chunk, in bytes of v: 64 while a fits the second-level cache.
.set CHUNK_BYTES, 512
# From this many elements a no longer does (2 MB), and the chunk is larger.
.set LARGE_N, 262144
# Needles per chunk for such an a: 512.
.set LARGE_CHUNK_BYTES, 4096

# The start of a step: half into rax and ymm8, len reduced, and rax turned
# into the address of a[half - 1]. Leaves the zero flag set when this is the
# last step; nothing that follows in a step changes the flags.
.macro HALVE
    lea     rax, [r9 + 1]
    shr     rax, 1
    vmovq   xmm8, rax
    vpbroadcastq ymm8, xmm8
    sub     r9, rax
    lea     rax, [rdi + rax * 8 - 8]
.endm

.text
.globl searchsorted_right_i64
.type searchsorted_right_i64, @function
.p2align 4
searchsorted_right_i64:
    test    r8, r8
    jz      .Lreturn
    cmp     r8, 4
    jb      .Lfew
    mov     [rsp - 8], rbx
    mov     eax, CHUNK_BYTES
    mov     r9d, LARGE_CHUNK_BYTES
    cmp     rcx, LARGE_N
    cmovae  eax, r9d
    mov     [rsp - 16], rax             # bytes of v per chunk
    xor     r10d, r10d
    test    rcx, rcx
    jz      .Lempty

.Lchunk:
    mov     rax, r8
    shr     rax, 2
    shl     rax, 5                      # where the whole vectors end
    cmp     r10, rax
    jae     .Lremainder
    mov     r11, r10
    add     r11, [rsp - 16]
    cmp     r11, rax
    cmova   r11, rax                    # where this chunk ends
    # The first step: every search probes a[half - 1].
    mov     r9, rcx
    HALVE
    vpbroadcastq ymm10, qword ptr [rax]
    mov     rbx, r10
.Lfirst:
    vpcmpgtq ymm11, ymm10, [rsi + rbx]  # the probe is greater than the needle
    vpandn  ymm11, ymm11, ymm8          # half where it is not
    vmovdqu [rdx + rbx], ymm11
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lfirst
    test    r9, r9
    jz      .Lchunkdone
.Lstep:
    HALVE
    mov     rbx, r10
.p2align 5
.Lprobe:
    vmovdqu ymm0, [rdx + rbx]
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm10, xmm10, xmm10         # a gather merges into its destination: start it afresh
    vpgatherqq ymm10, [rax + ymm0 * 8], ymm9
    vpcmpgtq ymm10, ymm10, [rsi + rbx]
    vpandn  ymm10, ymm10, ymm8
    vpaddq  ymm0, ymm0, ymm10
    vmovdqu [rdx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lprobe
    test    r9, r9
    jnz     .Lstep
.Lchunkdone:
    mov     r10, r11
    jmp     .Lchunk

    # ---- The last one to three needles of a count that is not a multiple
    #      of four: the last four needles as one vector, in registers.
.Lremainder:
    mov     rbx, [rsp - 8]
    test    r8b, 3
    jz      .Ldone
    lea     r10, [r8 * 8 - 32]
    vmovdqu ymm4, [rsi + r10]
    vpxor   xmm0, xmm0, xmm0
    mov     r9, rcx
.Lstep4:
    HALVE
    vpcmpeqd ymm9, ymm9, ymm9
    vpgatherqq ymm10, [rax + ymm0 * 8], ymm9
    vpcmpgtq ymm10, ymm10, ymm4
    vpandn  ymm10, ymm10, ymm8
    vpaddq  ymm0, ymm0, ymm10
    test    r9, r9
    jnz     .Lstep4
    vmovdqu [rdx + r10], ymm0
.Ldone:
    vzeroupper
.Lreturn:
    ret

    # ---- a is empty: every answer is 0. No 256-bit register was written.
.Lempty:
    xor     eax, eax
.Lzero:
    mov     qword ptr [rdx + r10 * 8], rax
    inc     r10
    cmp     r10, r8
    jb      .Lzero
    ret

    # ---- One to three needles, one search at a time, in general registers:
    #      rsi and rdx walk v and out, r8 counts down. No 256-bit register is
    #      written here.
.Lfew:
    mov     r11, [rsi]                  # the needle
    xor     eax, eax                    # lo
    mov     r9, rcx
    test    r9, r9
    jz      .Lonestore
.Lonestep:
    lea     r10, [r9 + 1]
    shr     r10, 1                      # half
    sub     r9, r10
    add     r10, rax                    # lo + half
    cmp     r11, [rdi + r10 * 8 - 8]
    cmovge  rax, r10
    test    r9, r9
    jnz     .Lonestep
.Lonestore:
    mov     [rdx], rax
    add     rsi, 8
    add     rdx, 8
    dec     r8
    jnz     .Lfew
    ret
.size searchsorted_right_i64, .-searchsorted_right_i64

.section .note.GNU-stack,"",@progbits
