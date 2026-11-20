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
# 64-bit slot per needle, and their needles stay in v. A pass over the chunk
# loads four positions and their needles, takes two steps of those four
# searches, and stores the positions: per step one `vpgatherqq` for the four
# probes, one compare with the needles, `half` under the compare's mask and
# an add. Nothing in a pass waits for another vector of searches.
#
# Two steps in a pass because the load of the positions, their store and
# the loop are then paid once for two steps. Not three: the second and third
# gathers wait for the one before, and the processor runs out of room for
# instructions that wait. And one step in a pass from DEEP_N elements, where
# the probes miss the second-level cache and a probe that waits for another
# is a miss that is not in flight.
#
# The first pass over a chunk needs no gather. Every search probes the same
# element first, then one of two, then one of four, and which one is chosen
# with the masks of the compares before it: an AND and an XOR against the
# difference of the candidates. The pass takes two steps, or three when the
# number of steps (the number of bits of n) is odd, so that what is left for
# the gathers is always pairs.
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
# Registers: r9 len; rax and rcx the addresses of a[half - 1] for the two
# steps of a pass, ymm8 and ymm7 their halves in every lane (ymm6 the third
# of the first pass); r10 and r11 the byte offsets where the chunk starts and
# ends, rbx the offset inside it (saved below the stack pointer, where n, the
# number of needles and the chunk's size are kept too). ymm0 positions, ymm4
# needles, ymm9 the gather's mask; ymm10 to ymm15 and ymm5 the candidates of
# the first pass.

# Needles per chunk, in bytes of v: 64 while a fits the second-level cache.
.set CHUNK_BYTES, 512
# From this many elements a no longer does (2 MB), and the chunk is larger.
.set LARGE_N, 262144
# Needles per chunk for such an a: 512.
.set LARGE_CHUNK_BYTES, 4096
# From this many elements (4 MB, twice the second-level cache) a pass over
# the chunk takes one step, not two.
.set DEEP_N, 524288

# The next half: into \addr as the address of a[half - 1] and into every
# lane of \half, and len (r9) reduced by it. Leaves the zero flag set when
# this was the last step.
.macro HALVE addr, half, xhalf
    lea     \addr, [r9 + 1]
    shr     \addr, 1
    vmovq   \xhalf, \addr
    vpbroadcastq \half, \xhalf
    sub     r9, \addr
    lea     \addr, [rdi + \addr * 8 - 8]
.endm

# One step for the four searches whose positions are in ymm0 and whose
# needles are in ymm4. A gather clears its mask register, and it merges into
# its destination, which is therefore one more input: it is zeroed first, so
# that no probe waits for an earlier one.
.macro PROBE addr, half, n
    vpcmpeqd ymm9, ymm9, ymm9
    vpxor   xmm\n, xmm\n, xmm\n
    vpgatherqq ymm\n, [\addr + ymm0 * 8], ymm9
    vpcmpgtq ymm\n, ymm\n, ymm4
    vpandn  ymm\n, ymm\n, \half
    vpaddq  ymm0, ymm0, ymm\n
.endm

# One pass over the chunk: \steps steps of each of its searches.
.macro PASS steps
    mov     rbx, r10
.p2align 5
1:
    vmovdqu ymm0, [rdx + rbx]
    vmovdqu ymm4, [rsi + rbx]
    PROBE   rax, ymm8, 10
.if \steps > 1
    PROBE   rcx, ymm7, 11
.endif
    vmovdqu [rdx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      1b
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
    mov     [rsp - 24], rcx             # n
    mov     [rsp - 32], r8              # needles
    xor     r10d, r10d
    test    rcx, rcx
    jz      .Lempty

.Lchunk:
    mov     r8, [rsp - 32]
    mov     rax, r8
    shr     rax, 2
    shl     rax, 5                      # where the whole vectors end
    cmp     r10, rax
    jae     .Lremainder
    mov     r11, r10
    add     r11, [rsp - 16]
    cmp     r11, rax
    cmova   r11, rax                    # where this chunk ends
    # The first two steps need no gather. Every search probes a[half - 1]
    # first, and then one of two elements: ymm11 is the one a search probes
    # when it did not advance, ymm12 what differs in the other.
    mov     r9, [rsp - 24]
    HALVE   rax, ymm8, xmm8
    vpbroadcastq ymm10, qword ptr [rax]
    jz      .Lfirst1
    HALVE   rcx, ymm7, xmm7
    vpbroadcastq ymm11, qword ptr [rcx]
    lea     rbx, [rax + rcx + 8]
    sub     rbx, rdi                    # the address of a[half1 + half2 - 1]
    vpbroadcastq ymm12, qword ptr [rbx]
    vpxor   ymm12, ymm12, ymm11
    # The number of steps is the number of bits of n. When it is odd the
    # first pass takes three steps, so that what is left is always pairs.
    bsr     r8, qword ptr [rsp - 24]
    test    r8b, 1
    jz      .Lthree
    mov     rbx, r10
.p2align 4
.Lfirst2:
    vmovdqu ymm4, [rsi + rbx]
    vpcmpgtq ymm0, ymm10, ymm4          # the first probe is greater than the needle
    vpandn  ymm1, ymm0, ymm12
    vpxor   ymm1, ymm1, ymm11           # the second probe
    vpcmpgtq ymm1, ymm1, ymm4
    vpandn  ymm0, ymm0, ymm8
    vpandn  ymm1, ymm1, ymm7
    vpaddq  ymm0, ymm0, ymm1
    vmovdqu [rdx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lfirst2
    test    r9, r9
    jnz     .Lstep
    jmp     .Lchunkdone
    # Three steps. The third probe is one of four elements: ymm13 is the
    # one for a search that advanced at neither step, ymm14 and ymm15 what
    # differs in the one that advanced at the first only and at the second
    # only, and ymm5 what the two differences leave of the fourth.
.Lthree:
    HALVE   r8, ymm6, xmm6
    vpbroadcastq ymm13, qword ptr [r8]
    sub     r8, rdi                     # (half3 - 1) * 8
    vpbroadcastq ymm14, qword ptr [rax + r8 + 8]
    vpbroadcastq ymm15, qword ptr [rcx + r8 + 8]
    vpbroadcastq ymm5, qword ptr [rbx + r8 + 8]
    vpxor   ymm14, ymm14, ymm13
    vpxor   ymm15, ymm15, ymm13
    vpxor   ymm5, ymm5, ymm13
    vpxor   ymm5, ymm5, ymm14
    vpxor   ymm5, ymm5, ymm15
    mov     rbx, r10
.p2align 4
.Lfirst3:
    vmovdqu ymm4, [rsi + rbx]
    vpcmpgtq ymm0, ymm10, ymm4          # the first probe is greater than the needle
    vpandn  ymm1, ymm0, ymm12
    vpxor   ymm1, ymm1, ymm11           # the second probe
    vpandn  ymm2, ymm0, ymm14
    vpandn  ymm3, ymm0, ymm5
    vpcmpgtq ymm1, ymm1, ymm4
    vpxor   ymm2, ymm2, ymm13
    vpxor   ymm3, ymm3, ymm15
    vpandn  ymm3, ymm1, ymm3
    vpxor   ymm2, ymm2, ymm3            # the third probe
    vpcmpgtq ymm2, ymm2, ymm4
    vpandn  ymm0, ymm0, ymm8
    vpandn  ymm1, ymm1, ymm7
    vpandn  ymm2, ymm2, ymm6
    vpaddq  ymm0, ymm0, ymm1
    vpaddq  ymm0, ymm0, ymm2
    vmovdqu [rdx + rbx], ymm0
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lfirst3
    test    r9, r9
    jnz     .Lstep
    jmp     .Lchunkdone
    # One element: one step.
.Lfirst1:
    mov     rbx, r10
.Lfirst:
    vpcmpgtq ymm11, ymm10, [rsi + rbx]
    vpandn  ymm11, ymm11, ymm8
    vmovdqu [rdx + rbx], ymm11
    add     rbx, 32
    cmp     rbx, r11
    jb      .Lfirst
    jmp     .Lchunkdone
.Lstep:
    HALVE   rax, ymm8, xmm8
    jz      .Lpass1
    cmp     qword ptr [rsp - 24], DEEP_N
    jae     .Lpass1
    HALVE   rcx, ymm7, xmm7
    PASS    2
    test    r9, r9
    jnz     .Lstep
    jmp     .Lchunkdone
.Lpass1:
    PASS    1
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
    mov     r9, [rsp - 24]
.Lstep4:
    HALVE   rax, ymm8, xmm8
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
