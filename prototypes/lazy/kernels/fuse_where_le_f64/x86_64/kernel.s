.intel_syntax noprefix
# void fuse_where_le_f64(const double *a, const double *b, const double *c, double *out, size_t n, double s);
# out[i] = (a[i] <= s) ? b[i] : c[i]
#
# a = rdi, b = rsi, c = rdx, out = rcx, n = r8, s = xmm0.
#
# A fused kernel for the shape where(le(V0,S0),V1,V2), written by
# prototypes/lazy/make_kernel.py from that name. Every operation is one AVX
# instruction that rounds once, in the order NumPy's passes make them: no
# multiply-add, and a division is a division.
#
# Eight elements per iteration, two ymm registers of four, every address
# the array's base plus the index in rax. Every load of an iteration comes
# before its store, so out may be one of the arrays itself for a length that
# is a multiple of eight or below eight. A remainder of 1 to 7 elements is
# one more block that ends exactly at the end of the arrays and overlaps the
# block before it, computed from the arrays again: there out must not overlap
# an array. Fewer than 8 elements are done one at a time.

.text
.globl fuse_where_le_f64
.type fuse_where_le_f64, @function
.p2align 4
fuse_where_le_f64:
    xor     eax, eax
    cmp     r8, 8
    jb      .Lsmall
    vbroadcastsd ymm15, xmm0     # s in every lane
    sub     r8, 8                # where the last block starts
.Lblock:
    vmovupd     ymm0, ymmword ptr [rdi + rax*8]
    vmovupd     ymm1, ymmword ptr [rdi + rax*8 + 32]
    vcmppd      ymm0, ymm0, ymm15, 0x12
    vcmppd      ymm1, ymm1, ymm15, 0x12
    vmovupd     ymm2, ymmword ptr [rsi + rax*8]
    vmovupd     ymm3, ymmword ptr [rsi + rax*8 + 32]
    vmovupd     ymm4, ymmword ptr [rdx + rax*8]
    vmovupd     ymm5, ymmword ptr [rdx + rax*8 + 32]
    vblendvpd   ymm0, ymm4, ymm2, ymm0
    vblendvpd   ymm1, ymm5, ymm3, ymm1
    vmovupd     ymmword ptr [rcx + rax*8], ymm0
    vmovupd     ymmword ptr [rcx + rax*8 + 32], ymm1
    add     rax, 8
    cmp     rax, r8
    jbe     .Lblock
    sub     rax, 8                 # where the block just done started
    cmp     rax, r8
    je      .Ldone
    mov     rax, r8                # 1 to 7 left: the block that ends with the arrays
    jmp     .Lblock
.Lsmall:
    test    r8, r8
    jz      .Ldone
.Lone:
    vmovsd      xmm1, qword ptr [rdi + rax*8]
    vcmpsd      xmm1, xmm1, xmm0, 0x12
    vmovsd      xmm2, qword ptr [rsi + rax*8]
    vmovsd      xmm3, qword ptr [rdx + rax*8]
    vblendvpd   xmm1, xmm3, xmm2, xmm1
    vmovsd      qword ptr [rcx + rax*8], xmm1
    inc     rax
    cmp     rax, r8
    jb      .Lone
.Ldone:
    vzeroupper
    ret
.size fuse_where_le_f64, .-fuse_where_le_f64

.section .note.GNU-stack,"",@progbits
