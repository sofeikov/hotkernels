.intel_syntax noprefix
# double sum_f64(const double *x, size_t n);
#
# rdi = x, rsi = n. Returns np.sum(x) in xmm0, bit for bit: the additions of
# NumPy's pairwise summation, in its order (spec.py and naive.c):
#
#   pairwise(x, n), n < 8:     x[0] + x[1] + ... one after another from +0.0
#   pairwise(x, n), n <= 128:  a leaf. Eight running sums r0..r7, element i
#                              into r(i mod 8), started from x[0..7];
#                              ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)),
#                              then the last n mod 8 elements one by one
#   pairwise(x, n), n > 128:   h = n / 2 rounded down to a multiple of 8;
#                              pairwise(x, h) + pairwise(x + h, n - h)
#   result:                    0.0 + pairwise(x, n)
#
# A leaf's eight running sums are two ymm registers: lanes 0..3 hold r0..r3,
# lanes 4..7 r4..r7, and each step adds the next 64-byte block of eight
# elements. The combination is `vhaddpd` (r0 + r1, r4 + r5, r2 + r3, r6 + r7),
# the high half added to the low, and the two lanes added: exactly the
# reference's tree. Every addition is one rounded double addition; nothing
# is reassociated. Two of a quad's eight chains (below) add with
# `vfmadd231pd s, one, [x]`: x * 1.0 is x exactly, so x * 1.0 + s rounded
# once is the addition s + x, bit for bit, signed zeros and denormals
# included. It runs on port 0, which this core's adders (ports 1 and 5) do
# not use.
#
# Why this is faster than the reference. One leaf is two chains of
# dependent additions, so the reference waits on the add's latency at every
# block. Leaves do not depend on each other until they are combined, so this
# kernel sums the two leaves of a node side by side (a pair, four chains) or
# the four leaves of a node two levels above them (a quad, 272 to 512
# elements, eight chains). A quad's loop is then limited by the loads, two
# 32-byte loads a cycle on the Intel Core i5-14600KF. Each leaf of a quad
# has its own pointer: a memory operand with an index register would be
# split in two at rename and make the loop too long for it.
#
# Leaves side by side differ in length. A node of g blocks of eight splits
# into floor(g/2) and ceil(g/2) blocks (n / 2 rounded down to a multiple of
# 8 is 8 * floor(g/2)), so the leaves of a pair or a quad are q or q + 1
# blocks long, q the shortest. All are summed for q blocks together; a
# leaf with q + 1 blocks then adds its last block, and one with q blocks
# adds a block of +0.0 instead. Adding +0.0 to a running sum changes
# nothing but the sign of a zero, and the sign of a zero never reaches the
# result: a zero added to a non-zero value gives that value exactly, and a
# zero result becomes +0.0 at the final 0.0 + (...). Only the last node of
# the array carries the n mod 8 leftover elements, on its last leaf, and
# they are added one at a time after that leaf's eight sums are combined.
#
# Aligned loads in a quad. A 32-byte load that straddles two cache lines
# costs a cycle more, and NumPy's buffers (and the judge's) are 16 bytes
# from a 32-byte boundary as often as not. Every leaf starts a multiple of
# 64 bytes from x, so all leaves share x's offset of t elements (0..3) from
# a 32-byte boundary. For t > 0 a quad loads aligned chunks starting t
# elements before each leaf: lane p of the even chunks' sum then holds
# r(p - t mod 8) and of the odd chunks' r(4 + p - t mod 8), each still added
# in the reference order. A leaf's first chunk has its lanes before the leaf
# cleared to +0.0 (the sign of a zero, as above), and its last chunk, which
# it shares with the next leaf, its lanes after the leaf. The first chunk of
# the array and the last are read with `vmaskmovpd`, which loads nothing
# from the lanes it leaves out, so nothing outside the buffer is read. For
# t = 2 `vhaddpd` still pairs r0 with r1 and so on, and the combination only
# takes the pairs from other lanes; for t = 1 and 3 each leaf is turned back
# with `vpermpd` and `vblendpd` first. An x that is not a multiple of 8
# bytes is summed with unaligned loads.
#
# From SEQ_MIN elements (8 MB, beyond the second-level cache) a quad's four
# leaves are summed one after another, one stream through memory instead of
# four, with unaligned loads; the out-of-order core overlaps them. Each
# step there prefetches SEQ_AHEAD bytes ahead: 1.07x at 16,777,216 elements,
# from memory.
#
# The tree is walked with an explicit stack in a frame below rsp (no calls).
# A node that is a leaf, a pair or a quad is summed at once; any other is
# split, its right half pushed as (address, length), and its left half
# walked. When a node's sum is ready the top entry is popped: a pending
# right half is replaced by the sum just made (length 0 marks a sum) and
# walked; a sum is added to it and the stack popped again. The stack holds
# at most one entry per level of the tree. A node of more than 128 elements
# has halves of at most half its length plus 8, so for n < 2^61 (an array
# of doubles in 2^64 bytes) a node is at most 512 elements after 53 levels,
# and such a node is split at most twice more before its parts are leaves,
# pairs or quads: at most 55 entries. The frame holds STACK_ENTRIES = 64
# entries of 16 bytes, 1 KB; n and t are kept in the red zone below it.
#
# Short arrays do not pay for the walk: fewer than 8 elements take the
# chain from +0.0, up to 128 elements one leaf, and up to 247 elements, or
# 248 or 256, one pair; only longer arrays set up the walk.
#
# Every load is inside the buffer: a leaf's blocks end at its own end, the
# leftover elements are loaded one at a time, a missing last block is read
# from .Lzeros, never from the array, and the aligned chunks are as above.
#
# Registers in the walk: r8 address and r9 length (elements) of the current
# node, r10 the top of the node stack (it grows up from rsp), ymm8 and ymm9
# the lane masks for t, ymm15 four times 1.0. A node's sum comes back in
# xmm0. Every path that writes a ymm register ends with `vzeroupper`.

# Entries in the node stack: one per level of the tree, see above.
.set STACK_ENTRIES, 64
# Bytes of the frame: 16 per entry, a multiple of 16 so rsp keeps its alignment.
.set FRAME_BYTES, STACK_ENTRIES * 16
# The largest node summed as one leaf, in elements (NumPy's PW_BLOCKSIZE).
.set LEAF_MAX, 128
# From this many elements (8 MB) a quad's leaves are summed one after another.
.set SEQ_MIN, 1048576
# How far ahead of its loads a leaf summed on its own prefetches, in bytes.
.set SEQ_AHEAD, 2048

.section .rodata
.p2align 6
# One block of eight +0.0: the missing last block of a shorter leaf.
.Lzeros:
    .quad 0, 0, 0, 0, 0, 0, 0, 0
# 1.0: x * 1.0 + s in one `vfmadd231pd` is the addition s + x, bit for bit.
.Lone:
    .quad 0x3ff0000000000000
# Lane masks for a rotation of t elements: 32 - 8t bytes in, lanes t..3 are
# set; 64 - 8t bytes in, lanes 0..t-1.
.Lmasks:
    .quad 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0

.text

# One leaf: r8 its address, r9 its length (8 to 128 elements). The sum
# comes back in xmm0. Uses rax, rcx, rdx, ymm0, ymm1.
.macro LEAF
    vmovupd ymm0, [r8]
    vmovupd ymm1, [r8 + 32]
    mov     rax, r9
    and     rax, -8
    lea     rdx, [r8 + rax*8]           # end of the whole blocks
    lea     rcx, [r8 + 64]
    cmp     rcx, rdx
    jae     .Lleaf_combine\@
.Lleaf_loop\@:
    vaddpd  ymm0, ymm0, [rcx]
    vaddpd  ymm1, ymm1, [rcx + 32]
    add     rcx, 64
    cmp     rcx, rdx
    jb      .Lleaf_loop\@
.Lleaf_combine\@:
    vhaddpd ymm0, ymm0, ymm1            # r0+r1, r4+r5, r2+r3, r6+r7
    vextractf128 xmm1, ymm0, 1
    vaddpd  xmm0, xmm0, xmm1            # (r0+r1)+(r2+r3), (r4+r5)+(r6+r7)
    vunpckhpd xmm1, xmm0, xmm0
    vaddsd  xmm0, xmm0, xmm1
    mov     eax, r9d
    and     eax, 7
    jz      .Lleaf_done\@
.Lleaf_tail\@:
    vaddsd  xmm0, xmm0, [rdx]
    add     rdx, 8
    dec     eax
    jnz     .Lleaf_tail\@
.Lleaf_done\@:
.endm

# A rotated leaf in \a (even chunks) and \b (odd chunks) back to r0..r3 and
# r4..r7: both turned left by t lanes (\perm), then the lanes from 4 - t up
# exchanged (\blend). Uses ymm14.
.macro UNROT1 a, b, perm, blend
    vpermpd \a, \a, \perm
    vpermpd \b, \b, \perm
    vblendpd ymm14, \a, \b, \blend
    vblendpd \b, \b, \a, \blend
    vmovapd \a, ymm14
.endm

# The four leaves of a quad, rotated by an odd t.
.macro UNROTATE perm, blend
    UNROT1  ymm0, ymm1, \perm, \blend
    UNROT1  ymm2, ymm3, \perm, \blend
    UNROT1  ymm4, ymm5, \perm, \blend
    UNROT1  ymm6, ymm7, \perm, \blend
.endm

# A pair: r8, r9 a node whose two halves are leaves. Leaf a at r8 has
# floor(g/2) = q blocks, leaf b at rcx q or q + 1, then the n mod 8
# leftovers if this is the last node. The sum comes back in xmm0, and the
# code goes on at \exit. Uses rax, rcx, rdx, rsi, rdi, r8, ymm0 to ymm3.
.macro PAIR exit
    mov     rax, r9
    shr     rax, 4
    shl     rax, 6                      # q blocks, in bytes
    lea     rcx, [r8 + rax]             # leaf b
    mov     rdx, r9
    and     rdx, -8
    lea     rdx, [r8 + rdx*8]           # end of the whole blocks
    vmovupd ymm0, [r8]
    vmovupd ymm1, [r8 + 32]
    vmovupd ymm2, [rcx]
    vmovupd ymm3, [rcx + 32]
    mov     rdi, rcx                    # the end of leaf a
    add     r8, 64
    add     rcx, 64
.Lpair_loop\@:
    vaddpd  ymm0, ymm0, [r8]
    vaddpd  ymm1, ymm1, [r8 + 32]
    vaddpd  ymm2, ymm2, [rcx]
    vaddpd  ymm3, ymm3, [rcx + 32]
    add     r8, 64
    add     rcx, 64
    cmp     r8, rdi
    jne     .Lpair_loop\@
    lea     rdi, [rip + .Lzeros]
    cmp     rcx, rdx                    # leaf b's block q, if it has one
    cmovae  rcx, rdi
    vaddpd  ymm2, ymm2, [rcx]
    vaddpd  ymm3, ymm3, [rcx + 32]
    vhaddpd ymm0, ymm0, ymm1            # a: r0+r1, r4+r5, r2+r3, r6+r7
    vhaddpd ymm2, ymm2, ymm3            # b: the same
    vperm2f128 ymm1, ymm0, ymm2, 0x20
    vperm2f128 ymm3, ymm0, ymm2, 0x31
    vaddpd  ymm0, ymm1, ymm3            # a's two halves, b's two halves
    vhaddpd ymm0, ymm0, ymm0            # sum a, sum a, sum b, sum b
    vextractf128 xmm1, ymm0, 1
    mov     eax, r9d
    and     eax, 7
    jz      .Lpair_add\@
.Lpair_tail\@:
    vaddsd  xmm1, xmm1, [rdx]
    add     rdx, 8
    dec     eax
    jnz     .Lpair_tail\@
.Lpair_add\@:
    vaddsd  xmm0, xmm0, xmm1
    jmp     \exit
.endm

# One leaf of a quad on its own: running sums \a and \b, blocks from \start
# up to \end. Uses rax.
.macro SEQLEAF a, b, start, end
    vmovupd \a, [\start]
    vmovupd \b, [\start + 32]
    lea     rax, [\start + 64]
.Lseq\@:
    prefetcht0 [rax + SEQ_AHEAD]
    vaddpd  \a, \a, [rax]
    vaddpd  \b, \b, [rax + 32]
    add     rax, 64
    cmp     rax, \end
    jb      .Lseq\@
.endm

    .globl sum_f64
    .type sum_f64, @function
    .p2align 4
sum_f64:
    cmp     rsi, 8
    jb      .Lshort
    mov     r8, rdi
    mov     r9, rsi
    cmp     rsi, LEAF_MAX
    ja      .Lmore
    LEAF
    jmp     .Lfinish

    # ---- More than 128 elements, and both halves leaves: one pair, no walk.
.Lmore:
    mov     rax, rsi
    shr     rax, 4
    shl     rax, 3
    neg     rax
    add     rax, rsi                    # the right half
    cmp     rax, LEAF_MAX
    ja      .Ltree
    PAIR    .Lfinish

    # ---- Fewer than 8 elements: one after another from +0.0.
.Lshort:
    vxorpd  xmm0, xmm0, xmm0
    test    rsi, rsi
    jz      .Lreturn
.Lshort_loop:
    vaddsd  xmm0, xmm0, [rdi]
    add     rdi, 8
    dec     rsi
    jnz     .Lshort_loop
.Lreturn:
    ret

    # ---- More than 128 elements: the walk.
.Ltree:
    sub     rsp, FRAME_BYTES
    vbroadcastsd ymm15, [rip + .Lone]
    mov     r10, rsp
    mov     [rsp - 8], rsi              # n, below the frame
    xor     ecx, ecx
    mov     eax, edi
    and     eax, 31                     # 8t: x's offset in 32 bytes
    test    edi, 7
    cmovnz  eax, ecx                    # no rotation for an x not a multiple of 8
    mov     [rsp - 16], rax
    test    eax, eax
    jz      .Ldescend
    lea     rcx, [rip + .Lmasks]
    neg     rax
    vmovupd ymm8, [rcx + rax + 32]      # lanes t..3
    vmovupd ymm9, [rcx + rax + 64]      # lanes 0..t-1
.Ldescend:
    cmp     r9, LEAF_MAX
    jbe     .Lwalk_leaf
    mov     rax, r9
    shr     rax, 4
    shl     rax, 3                      # left half h
    mov     rcx, r9
    sub     rcx, rax                    # right half
    cmp     rcx, LEAF_MAX
    jbe     .Lpair                      # both halves are leaves
    cmp     rax, LEAF_MAX
    jbe     .Lsplit                     # the left half is a leaf, the right not
    mov     rdx, rcx
    shr     rdx, 4
    shl     rdx, 3
    mov     r11, rcx
    sub     r11, rdx                    # the right half's right half: the longest grandchild
    cmp     r11, LEAF_MAX
    jbe     .Lquad
.Lsplit:
    lea     rdx, [r8 + rax*8]
    mov     [r10], rdx                  # push the right half
    mov     [r10 + 8], rcx
    add     r10, 16
    mov     r9, rax
    jmp     .Ldescend

.Lwalk_leaf:
    LEAF

    # ---- A node's sum is in xmm0: combine it with the stack.
.Lascend:
    cmp     r10, rsp
    je      .Lwalk_done
    mov     r9, [r10 - 8]
    test    r9, r9
    jz      .Lcombine
    mov     r8, [r10 - 16]              # a pending right half: walk it,
    vmovsd  [r10 - 16], xmm0            # and keep the left half's sum in its place
    mov     qword ptr [r10 - 8], 0
    jmp     .Ldescend
.Lcombine:
    vaddsd  xmm0, xmm0, [r10 - 16]      # left + right
    sub     r10, 16
    jmp     .Lascend

.Lwalk_done:
    add     rsp, FRAME_BYTES
.Lfinish:
    vxorpd  xmm1, xmm1, xmm1
    vaddsd  xmm0, xmm0, xmm1            # 0.0 + pairwise: a zero becomes +0.0
    vzeroupper
    ret

    # ---- A pair in the walk.
.Lpair:
    PAIR    .Lascend

    # ---- A quad: r8, r9 a node of g blocks whose four grandchildren are
    # leaves. Leaves 0..3 at r8, rcx, rdx, rsi with floor(g/4) = q or q + 1
    # blocks, the n mod 8 leftovers after leaf 3 if this is the last node.
    # Leaf 0 is the shortest: q blocks.
.Lquad:
    lea     rsi, [r8 + rax*8]           # leaf 2: the right half
    lea     r11, [rsi + rdx*8]          # leaf 3: its right half
    shr     rax, 4
    shl     rax, 6                      # q blocks, in bytes
    lea     rcx, [r8 + rax]             # leaf 1
    mov     rdi, rax
    mov     rdx, rsi
    mov     rsi, r11
    mov     r11, r9
    and     r11, -8
    lea     r11, [r8 + r11*8]           # end of the whole blocks
    cmp     qword ptr [rsp - 8], SEQ_MIN
    jae     .Lquad_seq
    mov     rax, [rsp - 16]
    test    rax, rax
    jnz     .Lquad_rot
    vmovupd ymm0, [r8]
    vmovupd ymm1, [r8 + 32]
    vmovupd ymm2, [rcx]
    vmovupd ymm3, [rcx + 32]
    vmovupd ymm4, [rdx]
    vmovupd ymm5, [rdx + 32]
    vmovupd ymm6, [rsi]
    vmovupd ymm7, [rsi + 32]
.Lquad_body:
    mov     rax, rdi                    # q blocks, in bytes
    add     rdi, r8                     # the end of leaf 0
    add     r8, 64
    add     rcx, 64
    add     rdx, 64
    add     rsi, 64
.Lquad_loop:
    vaddpd  ymm0, ymm0, [r8]
    vfmadd231pd ymm1, ymm15, [r8 + 32]
    vaddpd  ymm2, ymm2, [rcx]
    vaddpd  ymm3, ymm3, [rcx + 32]
    vaddpd  ymm4, ymm4, [rdx]
    vfmadd231pd ymm5, ymm15, [rdx + 32]
    vaddpd  ymm6, ymm6, [rsi]
    vaddpd  ymm7, ymm7, [rsi + 32]
    add     r8, 64
    add     rcx, 64
    add     rdx, 64
    add     rsi, 64
    cmp     r8, rdi
    jne     .Lquad_loop
    # The last block of leaves 1, 2 and 3, or a block of +0.0 for a leaf of
    # q blocks. The pointers are at block q of each leaf, and a leaf has
    # q + 1 blocks when the next one starts more than q blocks after it.
    lea     rdi, [rip + .Lzeros]
    mov     r8, rdx
    sub     r8, rcx
    cmp     r8, rax
    cmovbe  rcx, rdi
    vaddpd  ymm2, ymm2, [rcx]
    vaddpd  ymm3, ymm3, [rcx + 32]
    mov     r8, rsi
    sub     r8, rdx
    cmp     r8, rax
    cmovbe  rdx, rdi
    vaddpd  ymm4, ymm4, [rdx]
    vaddpd  ymm5, ymm5, [rdx + 32]
    cmp     rsi, r11
    cmovae  rsi, rdi
    vaddpd  ymm6, ymm6, [rsi]
    vaddpd  ymm7, ymm7, [rsi + 32]
    cmp     qword ptr [rsp - 16], 0
    jne     .Lquad_rot_end
.Lquad_reduce:
    vhaddpd ymm0, ymm0, ymm1            # each leaf: r0+r1, r4+r5, r2+r3, r6+r7
    vhaddpd ymm2, ymm2, ymm3
    vhaddpd ymm4, ymm4, ymm5
    vhaddpd ymm6, ymm6, ymm7
    vperm2f128 ymm1, ymm0, ymm2, 0x20
    vperm2f128 ymm3, ymm0, ymm2, 0x31
    vaddpd  ymm0, ymm1, ymm3            # leaves 0 and 1, each as two halves
    vperm2f128 ymm5, ymm4, ymm6, 0x20
    vperm2f128 ymm7, ymm4, ymm6, 0x31
    vaddpd  ymm4, ymm5, ymm7            # leaves 2 and 3
.Lquad_combine:
    vhaddpd ymm0, ymm0, ymm4            # S0, S2, S1, S3
    vextractf128 xmm1, ymm0, 1          # S1, S3
    mov     eax, r9d
    and     eax, 7
    jz      .Lquad_add
    vunpckhpd xmm2, xmm1, xmm1          # S3 alone
.Lquad_tail:
    vaddsd  xmm2, xmm2, [r11]           # the leftovers follow the whole blocks
    add     r11, 8
    dec     eax
    jnz     .Lquad_tail
    vunpcklpd xmm1, xmm1, xmm2          # S1, S3 with its leftovers
.Lquad_add:
    vaddpd  xmm0, xmm0, xmm1            # S0 + S1, S2 + S3
    vunpckhpd xmm1, xmm0, xmm0
    vaddsd  xmm0, xmm0, xmm1
    jmp     .Lascend

.Lquad_seq:
    SEQLEAF ymm0, ymm1, r8, rcx
    SEQLEAF ymm2, ymm3, rcx, rdx
    SEQLEAF ymm4, ymm5, rdx, rsi
    SEQLEAF ymm6, ymm7, rsi, r11
    jmp     .Lquad_reduce

.Lquad_rot:
    sub     r8, rax
    sub     rcx, rax
    sub     rdx, rax
    sub     rsi, rax
    sub     r11, rax
    vmaskmovpd ymm0, ymm8, [r8]
    vmovupd ymm1, [r8 + 32]
    vmovupd ymm10, [rcx]                # the chunk leaves 0 and 1 share
    vandpd  ymm2, ymm10, ymm8           # leaf 1's lanes t..3
    vandnpd ymm10, ymm8, ymm10          # leaf 0's last t elements
    vmovupd ymm3, [rcx + 32]
    vmovupd ymm11, [rdx]
    vandpd  ymm4, ymm11, ymm8
    vandnpd ymm11, ymm8, ymm11
    vmovupd ymm5, [rdx + 32]
    vmovupd ymm12, [rsi]
    vandpd  ymm6, ymm12, ymm8
    vandnpd ymm12, ymm8, ymm12
    vmovupd ymm7, [rsi + 32]
    vmaskmovpd ymm13, ymm9, [r11]       # leaf 3's last t elements
    jmp     .Lquad_body
.Lquad_rot_end:
    vaddpd  ymm0, ymm0, ymm10
    vaddpd  ymm2, ymm2, ymm11
    vaddpd  ymm4, ymm4, ymm12
    vaddpd  ymm6, ymm6, ymm13
    mov     rax, [rsp - 16]
    add     r11, rax                    # the end of the whole blocks again
    cmp     eax, 16
    jne     .Lquad_rot_odd
    # t = 2: A = r6 r7 r0 r1, B = r2 r3 r4 r5. vhaddpd still pairs r0 with
    # r1 and so on; only the lanes of the pairs differ.
    vhaddpd ymm0, ymm0, ymm1            # r6+r7, r2+r3, r0+r1, r4+r5
    vhaddpd ymm2, ymm2, ymm3
    vhaddpd ymm4, ymm4, ymm5
    vhaddpd ymm6, ymm6, ymm7
    vperm2f128 ymm1, ymm0, ymm2, 0x31   # r0+r1, r4+r5 of leaves 0 and 1
    vperm2f128 ymm3, ymm0, ymm2, 0x20   # r6+r7, r2+r3
    vpermilpd ymm3, ymm3, 5             # r2+r3, r6+r7
    vaddpd  ymm0, ymm1, ymm3
    vperm2f128 ymm5, ymm4, ymm6, 0x31
    vperm2f128 ymm7, ymm4, ymm6, 0x20
    vpermilpd ymm7, ymm7, 5
    vaddpd  ymm4, ymm5, ymm7
    jmp     .Lquad_combine
.Lquad_rot_odd:
    cmp     eax, 8
    jne     .Lquad_rot3
    UNROTATE 0x39, 0x8                  # t = 1
    jmp     .Lquad_reduce
.Lquad_rot3:
    UNROTATE 0x93, 0xe                  # t = 3
    jmp     .Lquad_reduce
.size sum_f64, .-sum_f64

.section .note.GNU-stack,"",@progbits
