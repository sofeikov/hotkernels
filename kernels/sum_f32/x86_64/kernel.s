.intel_syntax noprefix
# float sum_f32(const float *x, size_t n);
#
# rdi = x, rsi = n. Returns np.sum(x) in xmm0, NumPy's bits: the same
# additions in NumPy's pairwise order (see spec.py), each one rounded once,
# with only the operands of an addition swapped.
#
# NumPy's tree in vectors of 8 elements. n = 8m + t. A leaf is at most 128
# elements: eight running sums, element i into sum i mod 8, combined as
# ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then its last t
# elements one by one. A larger node splits at 8 * floor(m / 2) elements, so
# only the nodes on the right edge carry the t elements, and every other node
# is a whole number of vectors, split into floor(m/2) and ceil(m/2). A node
# on the right edge is a leaf when 8m + t <= 128, any other when m <= 16.
#
# Signs of zeros. A zero operand changes no sum that is not zero, so the sign
# of an intermediate zero only reaches a result that is zero, and NumPy
# returns 0 + pairwise(x), never -0.0. So a running sum may have +0.0 added
# to it where NumPy adds nothing, and the result only needs a final + 0.0.
#
# Paths, by n:
#   0..7       one after another from +0.0
#   8..128     one leaf: one ymm of eight running sums
#   129..      the walk, below
#
# The walk. The node in hand is split, its right half pushed on a stack of
# 16-byte frames below rsp, until it is a leaf or a group of eight (below);
# its sum is then taken. Coming back up, a frame whose right half is still
# to do keeps the left sum in its place and the walk goes on with the right
# half; a frame whose right half is done adds the two sums. A frame is
# pushed for each halving of m, so the stack holds at most 64 frames (1 KB)
# for any n that fits in memory.
#
# Groups of eight. A node of 68 to 128 vectors not on the right edge has its
# eight leaves three levels down, each of q = floor(m/8) or q + 1 vectors.
# Their start offsets are worked out as NumPy's halvings make them, and the
# eight leaves are summed in one loop, each in its own ymm, so the loop has
# eight independent chains of additions and is held by the loads rather
# than by the latency of one leaf's chain. After q steps a leaf of q + 1
# vectors adds its last one and a leaf of q adds a vector of +0.0. The eight
# leaf sums, and then the node's sum, are combined with vhaddps in the
# order NumPy combines them.

.section .rodata
.p2align 5
# A vector of +0.0, added to a leaf of a group that has no last vector.
.Lzero:
    .zero 32

.text

# Combine the eight running sums in ymm\r into the scalar in xmm\r as NumPy
# does: ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)). ymm\t is clobbered.
# vmovshdup puts r1 beside r0 (and r3, r5, r7 beside r2, r4, r6), vpermilps
# puts r2 + r3 beside r0 + r1 in each half: six instructions, where two
# vhaddps and the rest take eight and a longer chain.
.macro COMBINE r, t
    vmovshdup    ymm\t, ymm\r
    vaddps       ymm\r, ymm\r, ymm\t
    vpermilps    ymm\t, ymm\r, 0x02
    vaddps       ymm\r, ymm\r, ymm\t
    vextractf128 xmm\t, ymm\r, 1
    vaddss       xmm\r, xmm\r, xmm\t
.endm

# Sum one leaf of \m >= 1 vectors at \p into xmm0. \e and \i are clobbered.
.macro LEAF p, m, e, i
    vmovups ymm0, [\p]
    mov     \i, \m
    shl     \i, 5
    lea     \e, [\p + \i]
    neg     \i
    add     \i, 32
    jz      2f
1:
    vaddps  ymm0, ymm0, [\e + \i]
    add     \i, 32
    jnz     1b
2:
    COMBINE 0, 1
.endm

# Add the \t elements at \p to xmm0 one after another. \t is clobbered.
.macro TAIL p, t
    test    \t, \t
    jz      2f
1:
    vaddss  xmm0, xmm0, [\p]
    add     \p, 4
    dec     \t
    jnz     1b
2:
.endm

# After a group's loop, add leaf \j's last vector at \pj when its length
# (\pk - \pj, from the next leaf's pointer) is above r14 = 32q, else +0.0.
.macro EXTRA j, pj, pk
    mov     rdi, \pk
    sub     rdi, \pj
    cmp     rdi, r14
    lea     rdi, [rip + .Lzero]
    cmova   rdi, \pj
    vaddps  ymm\j, ymm\j, [rdi]
.endm

.globl sum_f32
.type sum_f32, @function
.p2align 4
sum_f32:
    cmp     rsi, 8
    jb      .Lsmall
    cmp     rsi, 128
    ja      .Lwalk

    # One leaf: m = n / 8 vectors, then n mod 8 elements.
    mov     rax, rsi
    shr     rax, 3
    LEAF    rdi, rax, rdx, rcx
    and     esi, 7
    TAIL    rdx, rsi
    vxorps  xmm1, xmm1, xmm1
    vaddss  xmm0, xmm0, xmm1
    vzeroupper
    ret

.Lsmall:
    vxorps  xmm0, xmm0, xmm0
    TAIL    rdi, rsi
    ret

.Lwalk:
    # rbx = node start, rbp = node length in vectors, r12 = t, r13 = the end
    # of the whole vectors (where the t elements start), r15 = rsp with no
    # frames on it.
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8              # rsp was 8 mod 16 at entry: now 0 mod 16
    mov     r15, rsp
    mov     rbx, rdi
    mov     rbp, rsi
    shr     rbp, 3
    mov     r12, rsi
    and     r12, 7
    mov     r13, rbp
    shl     r13, 5
    add     r13, rdi

.Lnode:
    test    r12, r12
    jz      .Lplain
    mov     rax, rbp
    shl     rax, 5
    add     rax, rbx
    cmp     rax, r13
    jne     .Lplain
    # On the right edge, with t elements after it.
    cmp     rbp, 15
    ja      .Lsplit
    LEAF    rbx, rbp, rdx, rcx
    mov     rax, r12
    mov     rdx, r13
    TAIL    rdx, rax
    jmp     .Lup

.Lplain:
    cmp     rbp, 16
    ja      .Lnot_leaf
    LEAF    rbx, rbp, rdx, rcx
    jmp     .Lup
.Lnot_leaf:
    lea     rax, [rbp - 68]
    cmp     rax, 128 - 68
    jbe     .Lgroup

.Lsplit:
    mov     rax, rbp
    shr     rax, 1
    sub     rsp, 16
    mov     rcx, rbp
    sub     rcx, rax
    mov     [rsp + 8], rcx
    mov     rcx, rax
    shl     rcx, 5
    add     rcx, rbx
    mov     [rsp], rcx
    mov     rbp, rax
    jmp     .Lnode

    # The sum of the node just done is in xmm0.
.Lup:
    cmp     rsp, r15
    je      .Ldone
    mov     rax, [rsp]
    test    rax, rax
    jz      1f
    # Left half done: keep its sum, go on with the right half.
    mov     rbp, [rsp + 8]
    mov     qword ptr [rsp], 0
    vmovss  [rsp + 8], xmm0
    mov     rbx, rax
    jmp     .Lnode
1:
    vaddss  xmm0, xmm0, [rsp + 8]
    add     rsp, 16
    jmp     .Lup

.Ldone:
    vxorps  xmm1, xmm1, xmm1
    vaddss  xmm0, xmm0, xmm1
    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    vzeroupper
    ret

.Lgroup:
    # Leaf starts in vectors, as NumPy's halvings make them: h = m/2, a = h,
    # b = m - h; s1 = a/4, s2 = a/2, s3 = s2 + (a - s2)/2, s4 = h,
    # s5 = h + b/4, s6 = h + b/2, s7 = s6 + (b - b/2)/2.
    # Pointers r8, r9, r10, r11, rax, rcx, rsi, rdx are each leaf's start
    # plus 32q; rdi runs from -32q to 0.
    mov     rdi, rbp
    and     rdi, -8
    shl     rdi, 2
    lea     r8, [rbx + rdi]
    neg     rdi
    mov     rax, rbp
    shr     rax, 1
    mov     rdx, rbp
    sub     rdx, rax
    mov     r10, rax
    shr     r10, 1
    mov     r9, rax
    shr     r9, 2
    mov     r11, rax
    sub     r11, r10
    shr     r11, 1
    add     r11, r10
    mov     rcx, rdx
    shr     rcx, 2
    add     rcx, rax
    mov     r14, rdx
    shr     r14, 1
    sub     rdx, r14
    shr     rdx, 1
    lea     rsi, [rax + r14]
    add     rdx, rsi
    shl     r9, 5
    add     r9, r8
    shl     r10, 5
    add     r10, r8
    shl     r11, 5
    add     r11, r8
    shl     rax, 5
    add     rax, r8
    shl     rcx, 5
    add     rcx, r8
    shl     rsi, 5
    add     rsi, r8
    shl     rdx, 5
    add     rdx, r8

    vmovups ymm0, [r8 + rdi]
    vmovups ymm1, [r9 + rdi]
    vmovups ymm2, [r10 + rdi]
    vmovups ymm3, [r11 + rdi]
    vmovups ymm4, [rax + rdi]
    vmovups ymm5, [rcx + rdi]
    vmovups ymm6, [rsi + rdi]
    vmovups ymm7, [rdx + rdi]
    add     rdi, 32
.Lgroup_loop:
    vaddps  ymm0, ymm0, [r8 + rdi]
    vaddps  ymm1, ymm1, [r9 + rdi]
    vaddps  ymm2, ymm2, [r10 + rdi]
    vaddps  ymm3, ymm3, [r11 + rdi]
    vaddps  ymm4, ymm4, [rax + rdi]
    vaddps  ymm5, ymm5, [rcx + rdi]
    vaddps  ymm6, ymm6, [rsi + rdi]
    vaddps  ymm7, ymm7, [rdx + rdi]
    add     rdi, 32
    jnz     .Lgroup_loop

    test    rbp, 7
    jz      .Lgroup_sum
    # m is not a multiple of 8: some leaves have q + 1 vectors. The last
    # leaf always has (it is ceil(m/8) long).
    mov     r14, rbp
    and     r14, -8
    shl     r14, 2
    EXTRA   0, r8, r9
    EXTRA   1, r9, r10
    EXTRA   2, r10, r11
    EXTRA   3, r11, rax
    EXTRA   4, rax, rcx
    EXTRA   5, rcx, rsi
    EXTRA   6, rsi, rdx
    vaddps  ymm7, ymm7, [rdx]

.Lgroup_sum:
    # Leaf sums: vhaddps of two leaves gives each one's pairs per 128-bit
    # half, a second vhaddps their quads, and the two halves added give
    # [L0 .. L7] in ymm0. Then the node's sum from those, as in a leaf.
    vhaddps ymm0, ymm0, ymm1
    vhaddps ymm2, ymm2, ymm3
    vhaddps ymm4, ymm4, ymm5
    vhaddps ymm6, ymm6, ymm7
    vhaddps ymm0, ymm0, ymm2
    vhaddps ymm4, ymm4, ymm6
    vperm2f128 ymm1, ymm0, ymm4, 0x20
    vperm2f128 ymm2, ymm0, ymm4, 0x31
    vaddps  ymm0, ymm1, ymm2
    COMBINE 0, 1
    jmp     .Lup
.size sum_f32, .-sum_f32

.section .note.GNU-stack,"",@progbits
