.intel_syntax noprefix
# size_t unique_i64(const int64_t *x, int64_t *out, size_t n, int64_t lo, size_t span);
#
# rdi = x, rsi = out, rdx = n, rcx = lo, r8 = span. Writes the distinct
# values of the n elements at x that lie in [lo, lo + span) to out, in
# ascending order, and returns how many. An element is inside the range when
# d = x - lo, taken as an unsigned 64-bit number, is below span; every other
# element is ignored, so nothing outside out[0..span) is ever written,
# whatever the data. lo + span - 1 is at most the largest int64.
#
# Two paths, chosen by span:
#
#   bits    span <= 64. Value lo + d is bit d of a 64-bit set; nothing is
#           written to out until the values are read out of the set. Four
#           elements per vector: d = x - lo, then 1 << d by vpsllvq, which
#           gives zero for a count of 64 or more, so an element far outside
#           the range sets nothing and needs no test. A d from span to 63
#           sets a bit at or above span, which bzhi drops at the end. The
#           last vector ends at n and overlaps the one before it, which a
#           set does not mind. Fewer than four elements are read one at a
#           time.
#   table   span > 64. The last span bytes of out are a table of one byte
#           per value: cleared, 0x80 stored at byte d for each element
#           inside the range, then read out in order into the front of out,
#           32 bytes per vpmovmskb. The values written never reach a byte
#           of the table not yet read (see .Lread_start).
#
# No stack, no constants.

# Largest span of the bits path: one bit per value of a 64-bit register.
.set BITS_SPAN, 64
# From this many elements the bits path aligns its loads to 32 bytes, so that
# none straddles two cache lines.
.set ALIGN_MIN_N, 32
# What the table path stores for a value that is present: the top bit of
# the byte, which is the bit vpmovmskb reads.
.set PRESENT, 0x80

.text
.globl unique_i64
.type unique_i64, @function
.p2align 4
unique_i64:
    cmp     r8, BITS_SPAN
    ja      .Ltable

    # ---- bits path: rax = the set of values found.
    xor     eax, eax
    cmp     rdx, 4
    jb      .Lbits_few

    vpcmpeqd ymm15, ymm15, ymm15
    vpsrlq  ymm15, ymm15, 63            # 1 in every lane
    vpxor   xmm0, xmm0, xmm0            # four sets
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    lea     r10, [rdi + rdx*8]          # end of x
    lea     r11, [r10 - 128]            # the last place a step of 16 may start

    # Labels from zero up: when 0 <= lo and lo + span <= 64, an element is
    # its own bit number, and the shift takes its count from memory. The
    # set is shifted down by lo at the end (r9; 0 on the general path).
    test    rcx, rcx
    js      .Lbits_general
    lea     r9, [rcx + r8]
    cmp     r9, BITS_SPAN
    ja      .Lbits_general
    mov     r9, rcx
    # From ALIGN_MIN_N elements the loads are aligned: the first vector is
    # taken where x starts, and the steps go on from the next multiple of
    # 32 bytes, over some of the same elements again. Only for an x that
    # is a multiple of 8 itself, so that a step still starts at an element.
    cmp     rdx, ALIGN_MIN_N
    jb      .Lraw_start
    test    dil, 7
    jnz     .Lraw_start
    vpsllvq ymm0, ymm15, [rdi]
    add     rdi, 32
    and     rdi, -32
.Lraw_start:
    cmp     rdi, r11
    ja      .Lraw_singles
.Lraw16:
    vpsllvq ymm4, ymm15, [rdi]
    vpsllvq ymm5, ymm15, [rdi + 32]
    vpsllvq ymm6, ymm15, [rdi + 64]
    vpsllvq ymm7, ymm15, [rdi + 96]
    add     rdi, 128
    vpor    ymm0, ymm0, ymm4
    vpor    ymm1, ymm1, ymm5
    vpor    ymm2, ymm2, ymm6
    vpor    ymm3, ymm3, ymm7
    cmp     rdi, r11
    jbe     .Lraw16
.Lraw_singles:
    lea     r11, [r10 - 32]             # the last vector, which ends at n
    cmp     rdi, r11
    jae     .Lraw_last
.Lraw4:
    vpsllvq ymm4, ymm15, [rdi]
    add     rdi, 32
    vpor    ymm0, ymm0, ymm4
    cmp     rdi, r11
    jb      .Lraw4
.Lraw_last:
    vpsllvq ymm4, ymm15, [r11]
    vpor    ymm1, ymm1, ymm4
    jmp     .Lbits_reduce

.Lbits_general:
    mov     r9, rcx
    neg     r9
    vmovq   xmm14, r9
    vpbroadcastq ymm14, xmm14           # -lo in every lane
    xor     r9d, r9d
    cmp     rdx, ALIGN_MIN_N            # aligned loads, as above
    jb      .Lbits_start
    test    dil, 7
    jnz     .Lbits_start
    vpaddq  ymm4, ymm14, [rdi]
    vpsllvq ymm0, ymm15, ymm4
    add     rdi, 32
    and     rdi, -32
.Lbits_start:
    cmp     rdi, r11
    ja      .Lbits_singles
.Lbits16:
    vpaddq  ymm4, ymm14, [rdi]
    vpaddq  ymm5, ymm14, [rdi + 32]
    vpaddq  ymm6, ymm14, [rdi + 64]
    vpaddq  ymm7, ymm14, [rdi + 96]
    add     rdi, 128
    vpsllvq ymm4, ymm15, ymm4
    vpsllvq ymm5, ymm15, ymm5
    vpsllvq ymm6, ymm15, ymm6
    vpsllvq ymm7, ymm15, ymm7
    vpor    ymm0, ymm0, ymm4
    vpor    ymm1, ymm1, ymm5
    vpor    ymm2, ymm2, ymm6
    vpor    ymm3, ymm3, ymm7
    cmp     rdi, r11
    jbe     .Lbits16
.Lbits_singles:
    lea     r11, [r10 - 32]             # the last vector, which ends at n
    cmp     rdi, r11
    jae     .Lbits_last
.Lbits4:
    vpaddq  ymm4, ymm14, [rdi]
    add     rdi, 32
    vpsllvq ymm4, ymm15, ymm4
    vpor    ymm0, ymm0, ymm4
    cmp     rdi, r11
    jb      .Lbits4
.Lbits_last:
    vpaddq  ymm4, ymm14, [r11]
    vpsllvq ymm4, ymm15, ymm4
    vpor    ymm1, ymm1, ymm4
    # The four sets and their four lanes into one register.
.Lbits_reduce:
    vpor    ymm0, ymm0, ymm1
    vpor    ymm2, ymm2, ymm3
    vpor    ymm0, ymm0, ymm2
    vextracti128 xmm1, ymm0, 1
    vpor    xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0x4e
    vpor    xmm0, xmm0, xmm1
    vmovq   rax, xmm0
    shrx    rax, rax, r9                # bit lo + d down to bit d
    vzeroupper
    jmp     .Lbits_done

    # Fewer than four elements: one at a time, (d < 64) << d.
.Lbits_few:
    test    rdx, rdx
    jz      .Lbits_done
    lea     rdx, [rdi + rdx*8]
.Lfew_step:
    mov     r10, [rdi]
    add     rdi, 8
    sub     r10, rcx
    xor     r11d, r11d
    cmp     r10, BITS_SPAN
    setb    r11b
    shlx    r11, r11, r10
    or      rax, r11
    cmp     rdi, rdx
    jne     .Lfew_step

    # Drop the bits at or above span (bzhi keeps all 64 for a span of 64),
    # then read the set out, lowest bit first: value lo + (number of the bit).
.Lbits_done:
    bzhi    r9, rax, r8
    xor     eax, eax
    test    r9, r9
    jz      .Lbits_ret
.Lbits_read:
    tzcnt   r10, r9
    add     r10, rcx
    mov     [rsi + rax*8], r10
    inc     rax
    blsr    r9, r9
    jnz     .Lbits_read
.Lbits_ret:
    ret

    # ---- table path: span > 64, so out has at least 65 elements, 520 bytes.
    # The table is one byte per value, the last span bytes of out: byte d
    # at r10 + d, where r10 = out + 7 * span bytes.
.Ltable:
    lea     r9, [rsi + r8*8]            # end of out
    mov     r10, r9
    sub     r10, r8                     # the table
    # Clear the table, 32 bytes per step; the last step ends at its end.
    vpxor   xmm0, xmm0, xmm0
    lea     r11, [r9 - 32]
    mov     rax, r10
.Lclear:
    vmovdqu [rax], ymm0
    add     rax, 32
    cmp     rax, r11
    jb      .Lclear
    vmovdqu [r11], ymm0

    # Mark: byte d = PRESENT for each element inside the range. Four
    # elements per step, each with its own test and branch, which the data
    # the binding passes (every element inside) makes predictable.
    mov     r11, rdx
    shr     r11, 2                      # steps of four
    and     edx, 3                      # the rest, one at a time
    test    r11, r11
    jz      .Lmark_rest
.Lmark4:
    mov     rax, [rdi]
    sub     rax, rcx
    cmp     rax, r8
    jae     1f
    mov     byte ptr [r10 + rax], PRESENT
1:  mov     rax, [rdi + 8]
    sub     rax, rcx
    cmp     rax, r8
    jae     2f
    mov     byte ptr [r10 + rax], PRESENT
2:  mov     rax, [rdi + 16]
    sub     rax, rcx
    cmp     rax, r8
    jae     3f
    mov     byte ptr [r10 + rax], PRESENT
3:  mov     rax, [rdi + 24]
    sub     rax, rcx
    cmp     rax, r8
    jae     4f
    mov     byte ptr [r10 + rax], PRESENT
4:  add     rdi, 32
    dec     r11
    jnz     .Lmark4
.Lmark_rest:
    test    edx, edx
    jz      .Lread_start
.Lmark1:
    mov     rax, [rdi]
    add     rdi, 8
    sub     rax, rcx
    cmp     rax, r8
    jae     5f
    mov     byte ptr [r10 + rax], PRESENT
5:  dec     edx
    jnz     .Lmark1

    # Read the table out: value lo + j goes to out[count] for each byte j
    # that is set. 32 bytes are taken per step, as a mask of their top
    # bits, before any value of the step is written. A store to out[count]
    # covers bytes 8 * count to 8 * count + 7 of out, and count is at most
    # j + 32 after a step that started at byte j, so the step writes below
    # byte 8 * j + 256 of out. The next byte of the table still to be read
    # is byte 7 * span + j + 32 of out, which is at or above that for
    # j <= span - 32, the last place a step starts. The last span % 32
    # bytes cannot be read that way, so the mask of the table's last 32
    # bytes is taken before the first value is written, and the bits of the
    # bytes the steps did not reach are read from it at the end.
    #
    # rdx = count, rdi = j, rcx = lo + j, r9d = the mask of the last 32 bytes.
.Lread_start:
    vmovdqu ymm0, [r9 - 32]
    vpmovmskb r9d, ymm0
    xor     edx, edx
    xor     edi, edi
    lea     r11, [r8 - 32]              # the last j a step may start at
.Lread32:
    vmovdqu ymm0, [r10 + rdi]
    vpmovmskb eax, ymm0
    test    eax, eax
    jz      .Lread32_next
.Lread32_bit:
    tzcnt   r11d, eax
    add     r11, rcx
    mov     [rsi + rdx*8], r11
    inc     rdx
    blsr    eax, eax
    jnz     .Lread32_bit
    lea     r11, [r8 - 32]
.Lread32_next:
    add     rdi, 32
    add     rcx, 32
    cmp     rdi, r11
    jbe     .Lread32
    # The bytes from j to span - 1: bits 32 - (span - j) and up of the mask
    # of the last 32 bytes. A shift by 32, when none are left, gives zero.
    mov     rax, r8
    sub     rax, rdi                    # span - j, 0 to 31
    neg     eax
    add     eax, 32
    shrx    r9, r9, rax
    test    r9d, r9d
    jz      .Ltable_ret
.Lread_last:
    tzcnt   r11d, r9d
    add     r11, rcx
    mov     [rsi + rdx*8], r11
    inc     rdx
    blsr    r9d, r9d
    jnz     .Lread_last
.Ltable_ret:
    mov     rax, rdx
    vzeroupper
    ret

.size unique_i64, .-unique_i64
.section .note.GNU-stack,"",@progbits
