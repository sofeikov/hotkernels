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
#           sets a bit at or above span, which bzhi drops at the end. When
#           0 <= lo and lo + span <= 64, an element is its own bit number:
#           the shift takes its counts from memory and the set is shifted
#           down by lo at the end. The last vector ends at n and overlaps
#           the one before it, which a set does not mind; from ALIGN_MIN_N
#           elements the first vector is taken where x starts and the rest
#           from a multiple of 32 bytes, for the same reason. Fewer than
#           four elements are read one at a time.
#   table   span > 64. The last span bytes of out are a table of one byte
#           per value: cleared, 0x80 stored at byte d for each element
#           inside the range, then read out in order into the front of out,
#           32 bytes per vpmovmskb. The values written never reach a byte
#           of the table not yet read (see .Lread_start).
#
#           A store per element is the slow part, so the table is the way
#           out, not the way in. A window of WINDOW values, a 64-bit set as
#           on the bits path, takes the elements first, sixteen per chunk:
#           t = x - (the window's first value), 1 << t into the set, and
#           the or of the sixteen t tested for a bit from 64 up. A chunk
#           without one is done. A chunk with one is marked in the table,
#           element by element, and the window is put around that chunk's
#           values if they are less than WINDOW apart, after the set has
#           been stored in the table. A window that starts at the value 0
#           needs no subtraction, and is the one chosen where it fits.
#           Fewer than sixteen elements all go to the table. Labels that
#           keep to 64 neighbouring values never reach it: if no chunk ever
#           failed there is no table at all, and the values are read
#           straight out of the set. Sorted labels reach it only where the
#           window moves. Where the first chunk is spread wider
#           than a window, or chunks keep failing, the window is left out
#           for a stretch that grows each time (see .Lwin_fail), so that
#           labels spread over a wider range pay one store per element and
#           a failed chunk or two.
#
# The table path keeps six values in the 128 bytes below the stack
# pointer, which is not moved. No constants.
#
# From FAR_MIN_BYTES of x, every vector loop asks for the cache lines AHEAD
# bytes on with prefetcht0, which reads nothing: an address past the end of
# x is ignored.

# Largest span of the bits path: one bit per value of a 64-bit register.
.set BITS_SPAN, 64
# From this many elements the bits path aligns its loads to 32 bytes, so that
# none straddles two cache lines.
.set ALIGN_MIN_N, 32
# From this many bytes of x, more than the first-level cache holds, the
# vector loops ask for the lines AHEAD bytes on.
.set FAR_MIN_BYTES, 65536
.set AHEAD, 2048
# What the table path stores for a value that is present: the top bit of
# the byte, which is the bit vpmovmskb reads.
.set PRESENT, 0x80

# Elements of a chunk of the table path's window, in bytes of x.
.set CHUNK_BYTES, 128
# Values in the window: one bit each of a 64-bit set.
.set WINDOW, 64
# A chunk that fails within this many bytes of x of the last one that
# failed is one of a row, and a window that lasted no longer was not worth
# moving: sixteen chunks. A failed chunk costs about what the window saves
# over that many.
.set FREQUENT_BYTES, 16 * CHUNK_BYTES
# The first stretch without the window, in bytes of x, and the longest.
.set SKIP_FIRST, 64 * CHUNK_BYTES
.set SKIP_MAX, 1 << 19
# Places below the stack pointer, in the 128 bytes a leaf function may use:
# out, the chunk that failed last, the chunk the window was moved at, the
# stretch, the window's new first value while the sets are stored, and the
# bytes of x.
.set OUT_SLOT, 8
.set LAST_FAIL_SLOT, 16
.set LAST_MOVE_SLOT, 24
.set SKIP_SLOT, 32
.set WINDOW_SLOT, 40
.set SIZE_SLOT, 48

# Marks the elements from rdi up to rsi in the table, byte d = PRESENT for
# each inside the range, each with its own test and branch, and leaves rdi
# at rsi. Makes the table first if there is none: the last span bytes of
# out, cleared 32 bytes per step, the last step ending at its end. Changes
# rax and r11.
.macro MARK_STORES
    test    r10, r10
    jnz     20f
    mov     r10, [rsp - OUT_SLOT]
    lea     r11, [r10 + r8*8]           # end of out
    mov     r10, r11
    sub     r10, r8                     # the table
    vpxor   xmm10, xmm10, xmm10
    sub     r11, 32
    mov     rax, r10
10: vmovdqu [rax], ymm10
    add     rax, 32
    cmp     rax, r11
    jb      10b
    vmovdqu [r11], ymm10
20: lea     r11, [rsi - 32]
    cmp     rdi, r11
    ja      40f
30: .irp k, 0, 8, 16, 24
    mov     rax, [rdi + \k]
    sub     rax, rcx
    cmp     rax, r8
    jae     31f
    mov     byte ptr [r10 + rax], PRESENT
31:
    .endr
    add     rdi, 32
    cmp     rdi, r11
    jbe     30b
40: cmp     rdi, rsi
    jae     60f
50: mov     rax, [rdi]
    add     rdi, 8
    sub     rax, rcx
    cmp     rax, r8
    jae     51f
    mov     byte ptr [r10 + rax], PRESENT
51: cmp     rdi, rsi
    jb      50b
60:
.endm

# The four sets and their lanes into rax, without the bits of values at or
# above lo + span. Leaves the window's first d in rsi. Changes r11.
.macro REDUCE_SET
    vpor    ymm0, ymm0, ymm1
    vpor    ymm2, ymm2, ymm3
    vpor    ymm0, ymm0, ymm2
    vextracti128 xmm1, ymm0, 1
    vpor    xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0x4e
    vpor    xmm0, xmm0, xmm1
    vmovq   rax, xmm0
    mov     rsi, r9
    sub     rsi, rcx                    # the window's first d
    mov     r11, r8
    sub     r11, rsi                    # values of the range from there on
    cmp     r11, WINDOW
    jae     70f
    bzhi    rax, rax, r11
70:
.endm

# Marks the values of the set in rax in the table; rsi = the window's
# first d. Changes rax, rsi and r11.
.macro STORE_SET
    add     rsi, r10
    test    rax, rax
    jz      81f
80: tzcnt   r11, rax
    mov     byte ptr [rsi + r11], PRESENT
    blsr    rax, rax
    jnz     80b
81:
.endm

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
    cmp     rdx, FAR_MIN_BYTES / 8
    jae     .Lbits_raw16_far
.Lbits_raw16:
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
    jbe     .Lbits_raw16
    jmp     .Lraw_singles
    # The same with the lines AHEAD bytes on asked for, for an x beyond
    # the first-level cache.
    .p2align 5
.Lbits_raw16_far:
    prefetcht0 [rdi + AHEAD]
    prefetcht0 [rdi + AHEAD + 64]
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
    jbe     .Lbits_raw16_far
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
    cmp     rdx, FAR_MIN_BYTES / 8
    jae     .Lbits16_far
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
    jmp     .Lbits_singles
    .p2align 5
.Lbits16_far:
    prefetcht0 [rdi + AHEAD]
    prefetcht0 [rdi + AHEAD + 64]
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
    jbe     .Lbits16_far
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
    #
    # rdi = the next element, rdx = the end of x, rcx = lo, r8 = span,
    # r9 = the window's first value, r10 = the table or 0 while there is
    # none, r11 = the last place a chunk may start. rax and rsi are free
    # while elements are marked; out waits below the stack pointer.
.Ltable:
    mov     [rsp - OUT_SLOT], rsi
    lea     rdx, [rdi + rdx*8]          # end of x
    xor     r10d, r10d                  # no table yet
    vpcmpeqd ymm13, ymm13, ymm13
    vpsrlq  ymm15, ymm13, 63            # 1 in every lane
    vpsllq  ymm13, ymm13, 6             # every bit from 64 up
    vpxor   xmm0, xmm0, xmm0            # four sets of the window
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
    mov     r9, rcx                     # the window starts at lo
    lea     r11, [rdx - CHUNK_BYTES]
    mov     rsi, rdx
    mov     rax, rdx
    sub     rax, rdi
    mov     [rsp - SIZE_SLOT], rax      # bytes of x
    cmp     rax, CHUNK_BYTES
    jb      .Lmark_plain                # fewer than a chunk: all through the table
    # The first window. If the first element is from 0 to 63 and 0 is in
    # the range, the window starts at the value 0, where an element is its
    # own bit number (see .Lwin_raw16). Otherwise it starts WINDOW / 2 below
    # the first element, if that is inside the range, and not below lo.
    mov     rax, [rdi]
    cmp     rax, WINDOW
    jae     .Lwin_around
    test    rcx, rcx
    jg      .Lwin_around
    mov     rsi, rcx
    neg     rsi
    cmp     rsi, r8
    jae     .Lwin_around
    xor     r9d, r9d
    jmp     .Lwin_go
.Lwin_around:
    sub     rax, rcx
    cmp     rax, r8
    jae     .Lwin_set
    sub     rax, WINDOW / 2
    jb      .Lwin_set
    lea     r9, [rcx + rax]
.Lwin_set:
    mov     rax, r9
    neg     rax
    vmovq   xmm14, rax
    vpbroadcastq ymm14, xmm14           # minus the window's first value

    # A chunk: t = x - (the window's first value) for sixteen elements,
    # 1 << t into the sets, and the or of every t tested for a bit from 64
    # up. Without one, every element of the chunk is a bit of the sets.
    # Four loops: for a window that starts at the value 0, t is the element
    # itself and the four subtractions are not needed; and for an x beyond
    # the first-level cache the lines AHEAD bytes on are asked for.
    jmp     .Lwin_go
    .p2align 5
.Lwin16:
    vpaddq  ymm4, ymm14, [rdi]
    vpaddq  ymm5, ymm14, [rdi + 32]
    vpaddq  ymm6, ymm14, [rdi + 64]
    vpaddq  ymm7, ymm14, [rdi + 96]
    vpor    ymm8, ymm4, ymm5
    vpor    ymm9, ymm6, ymm7
    vpsllvq ymm4, ymm15, ymm4
    vpsllvq ymm5, ymm15, ymm5
    vpsllvq ymm6, ymm15, ymm6
    vpsllvq ymm7, ymm15, ymm7
    vpor    ymm8, ymm8, ymm9
    vpor    ymm0, ymm0, ymm4
    vpor    ymm1, ymm1, ymm5
    vpor    ymm2, ymm2, ymm6
    vpor    ymm3, ymm3, ymm7
    vptest  ymm8, ymm13
    jnz     .Lwin_fail
    add     rdi, CHUNK_BYTES
    cmp     rdi, r11
    jbe     .Lwin16
    jmp     .Lwin_head
    .p2align 5
.Lwin_raw16:
    vmovdqu ymm4, [rdi]
    vmovdqu ymm5, [rdi + 32]
    vmovdqu ymm6, [rdi + 64]
    vmovdqu ymm7, [rdi + 96]
    vpor    ymm8, ymm4, ymm5
    vpor    ymm9, ymm6, ymm7
    vpsllvq ymm4, ymm15, ymm4
    vpsllvq ymm5, ymm15, ymm5
    vpsllvq ymm6, ymm15, ymm6
    vpsllvq ymm7, ymm15, ymm7
    vpor    ymm8, ymm8, ymm9
    vpor    ymm0, ymm0, ymm4
    vpor    ymm1, ymm1, ymm5
    vpor    ymm2, ymm2, ymm6
    vpor    ymm3, ymm3, ymm7
    vptest  ymm8, ymm13
    jnz     .Lwin_fail
    add     rdi, CHUNK_BYTES
    cmp     rdi, r11
    jbe     .Lwin_raw16

    jmp     .Lwin_head
    .p2align 5
.Lwin16_far:
    prefetcht0 [rdi + AHEAD]
    prefetcht0 [rdi + AHEAD + 64]
    vpaddq  ymm4, ymm14, [rdi]
    vpaddq  ymm5, ymm14, [rdi + 32]
    vpaddq  ymm6, ymm14, [rdi + 64]
    vpaddq  ymm7, ymm14, [rdi + 96]
    vpor    ymm8, ymm4, ymm5
    vpor    ymm9, ymm6, ymm7
    vpsllvq ymm4, ymm15, ymm4
    vpsllvq ymm5, ymm15, ymm5
    vpsllvq ymm6, ymm15, ymm6
    vpsllvq ymm7, ymm15, ymm7
    vpor    ymm8, ymm8, ymm9
    vpor    ymm0, ymm0, ymm4
    vpor    ymm1, ymm1, ymm5
    vpor    ymm2, ymm2, ymm6
    vpor    ymm3, ymm3, ymm7
    vptest  ymm8, ymm13
    jnz     .Lwin_fail
    add     rdi, CHUNK_BYTES
    cmp     rdi, r11
    jbe     .Lwin16_far
    jmp     .Lwin_head
    .p2align 5
.Lwin_raw16_far:
    prefetcht0 [rdi + AHEAD]
    prefetcht0 [rdi + AHEAD + 64]
    vmovdqu ymm4, [rdi]
    vmovdqu ymm5, [rdi + 32]
    vmovdqu ymm6, [rdi + 64]
    vmovdqu ymm7, [rdi + 96]
    vpor    ymm8, ymm4, ymm5
    vpor    ymm9, ymm6, ymm7
    vpsllvq ymm4, ymm15, ymm4
    vpsllvq ymm5, ymm15, ymm5
    vpsllvq ymm6, ymm15, ymm6
    vpsllvq ymm7, ymm15, ymm7
    vpor    ymm8, ymm8, ymm9
    vpor    ymm0, ymm0, ymm4
    vpor    ymm1, ymm1, ymm5
    vpor    ymm2, ymm2, ymm6
    vpor    ymm3, ymm3, ymm7
    vptest  ymm8, ymm13
    jnz     .Lwin_fail
    add     rdi, CHUNK_BYTES
    cmp     rdi, r11
    jbe     .Lwin_raw16_far


    # The next chunk, or the last one, which ends at n.
.Lwin_head:
    cmp     rdi, r11
    jbe     .Lwin_go
    cmp     rdi, rdx
    je      .Lmark_done
    mov     rdi, r11
.Lwin_go:
    cmp     qword ptr [rsp - SIZE_SLOT], FAR_MIN_BYTES
    jae     1f
    test    r9, r9
    jz      .Lwin_raw16
    jmp     .Lwin16
1:  test    r9, r9
    jz      .Lwin_raw16_far
    jmp     .Lwin16_far

    # A chunk with an element outside the window. Its elements are marked
    # in the table, one at a time; the bits it set stay, since each is a
    # value that is there. What else happens depends on the chunk's values
    # and on how long ago the last chunk failed.
    #
    # The chunk is read for the smallest and the largest d of its elements
    # inside the range (rsi, r11). If they are less than WINDOW apart, the
    # chunk is of one cluster, and the window is put at the value 0 or
    # around their middle, and not below lo. If the window it replaces was
    # itself put there no more than FREQUENT_BYTES back, the clusters do
    # not last, and the data is not for a window.
    #
    # Otherwise the chunk is spread wider than a window, or holds an
    # element from elsewhere. If the last failure was more than
    # FREQUENT_BYTES back, that is all; if not, and so too when the first
    # chunk of all fails this way, the data is not for a window.
    #
    # Data that is not for a window, either way, is marked in the table
    # for a stretch without trying the window: SKIP_FIRST bytes of x, and
    # four times as much each time the chunk after a stretch fails wide
    # again, up to SKIP_MAX. After a stretch the window may move once
    # without that counting.
.Lwin_fail:
    lea     rax, [rdi + CHUNK_BYTES]
    cmp     rax, rdx
    je      .Lfail_last
    # The first failure, which is the one before there is a table, starts
    # the bookkeeping: no window moved yet, the first stretch, and a last
    # failure far back, or one chunk back if this is the first chunk of x,
    # so that it counts as one in a row.
    test    r10, r10
    jnz     5f
    mov     qword ptr [rsp - LAST_MOVE_SLOT], 0
    mov     qword ptr [rsp - SKIP_SLOT], SKIP_FIRST
    mov     rax, rdx
    sub     rax, rdi
    cmp     rax, [rsp - SIZE_SLOT]
    lea     rax, [rdi - CHUNK_BYTES]
    mov     esi, 0
    cmovne  rax, rsi
    mov     [rsp - LAST_FAIL_SLOT], rax
5:
    mov     rsi, -1
    xor     r11d, r11d
    .irp k, 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120
    mov     rax, [rdi + \k]
    sub     rax, rcx
    cmp     rax, r8
    jae     1f
    cmp     rax, rsi
    cmovb   rsi, rax
    cmp     rax, r11
    cmova   r11, rax
1:
    .endr
    mov     [rsp - WINDOW_SLOT], r9     # the window stays, unless moved below
    cmp     rsi, r11
    ja      .Lfail_wide                 # none inside the range
    mov     rax, r11
    sub     rax, rsi
    cmp     rax, WINDOW
    jae     .Lfail_wide
    # One cluster. From the value 0, if lo is not above it and the chunk's
    # values are all from 0 to 63; else around their middle.
    test    rcx, rcx
    jg      3f
    lea     rax, [rcx + rsi]
    test    rax, rax
    js      3f
    lea     rax, [rcx + r11]
    cmp     rax, WINDOW
    jae     3f
    xor     eax, eax
    jmp     4f
3:  lea     rax, [rsi + r11]
    shr     rax, 1
    sub     rax, WINDOW / 2
    jae     2f
    xor     eax, eax
2:  add     rax, rcx                    # the new window's first value
4:  mov     [rsp - WINDOW_SLOT], rax
    mov     [rsp - LAST_FAIL_SLOT], rdi
    lea     rsi, [rdi + CHUNK_BYTES]    # mark up to here
    mov     rax, [rsp - LAST_MOVE_SLOT]
    mov     [rsp - LAST_MOVE_SLOT], rdi
    test    rax, rax
    jz      .Lfail_mark                 # the first move, or the first after a stretch
    sub     rax, rdi
    neg     rax                         # bytes of x the window lasted
    cmp     rax, FREQUENT_BYTES
    ja      .Lfail_calm
    jmp     .Lfail_stretch
.Lfail_wide:
    mov     rax, rdi
    sub     rax, [rsp - LAST_FAIL_SLOT]
    mov     [rsp - LAST_FAIL_SLOT], rdi
    lea     rsi, [rdi + CHUNK_BYTES]
    cmp     rax, FREQUENT_BYTES
    ja      .Lfail_calm
.Lfail_stretch:
    mov     qword ptr [rsp - LAST_MOVE_SLOT], 0
    mov     rax, [rsp - SKIP_SLOT]
    lea     rsi, [rdi + rax]
    cmp     rsi, rdx
    cmova   rsi, rdx
    cmp     rax, SKIP_MAX
    jae     1f
    shl     rax, 2
    mov     [rsp - SKIP_SLOT], rax
1:  lea     rax, [rsi - CHUNK_BYTES]
    mov     [rsp - LAST_FAIL_SLOT], rax # a failure right after the stretch is one in a row
    jmp     .Lfail_mark
.Lfail_calm:
    mov     qword ptr [rsp - SKIP_SLOT], SKIP_FIRST
.Lfail_mark:
    MARK_STORES
    cmp     rdi, rdx
    je      .Lmark_done                 # nothing left for a window to take
    mov     rax, [rsp - WINDOW_SLOT]
    cmp     rax, r9
    je      .Lwin_keep
    # The sets go to the table before the window moves.
    REDUCE_SET
    STORE_SET
    mov     r9, [rsp - WINDOW_SLOT]
    mov     rax, r9
    neg     rax
    vmovq   xmm14, rax
    vpbroadcastq ymm14, xmm14
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm1, xmm1, xmm1
    vpxor   xmm2, xmm2, xmm2
    vpxor   xmm3, xmm3, xmm3
.Lwin_keep:
    lea     r11, [rdx - CHUNK_BYTES]
    jmp     .Lwin_head

    # The chunk that failed is the last: nothing is left for a window to
    # take, so its elements are marked and that is all.
.Lfail_last:
    mov     rsi, rdx
    MARK_STORES
    jmp     .Lmark_done

    # Fewer than a chunk of elements: all of them through the table.
.Lmark_plain:
    MARK_STORES

    # Every element is in the sets or in the table. Without a table the
    # values are read straight out of the sets, as on the bits path.
.Lmark_done:
    REDUCE_SET
    test    r10, r10
    jnz     .Lread_table
    mov     rsi, [rsp - OUT_SLOT]
    mov     rcx, r9                     # value of bit 0
    mov     r9, rax
    xor     eax, eax
    vzeroupper
    test    r9, r9
    jnz     .Lbits_read
    ret
.Lread_table:
    STORE_SET
    mov     rsi, [rsp - OUT_SLOT]

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
    vmovdqu ymm0, [r10 + r8 - 32]
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
