.intel_syntax noprefix
# void fuse_sqrt_v_f64(const double *a, double *out, size_t n);
# out[i] = __builtin_sqrt(a[i])
#
# a = rdi, out = rsi, n = rdx.
#
# A fused kernel for the shape sqrt(V0), written by
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
.globl fuse_sqrt_v_f64
.type fuse_sqrt_v_f64, @function
.p2align 4
fuse_sqrt_v_f64:
    xor     eax, eax
    cmp     rdx, 8
    jb      .Lsmall
    sub     rdx, 8                # where the last block starts
.Lblock:
    vmovupd     ymm0, ymmword ptr [rdi + rax*8]
    vmovupd     ymm1, ymmword ptr [rdi + rax*8 + 32]
    vsqrtpd     ymm0, ymm0
    vsqrtpd     ymm1, ymm1
    vmovupd     ymmword ptr [rsi + rax*8], ymm0
    vmovupd     ymmword ptr [rsi + rax*8 + 32], ymm1
    add     rax, 8
    cmp     rax, rdx
    jbe     .Lblock
    sub     rax, 8                 # where the block just done started
    cmp     rax, rdx
    je      .Ldone
    mov     rax, rdx                # 1 to 7 left: the block that ends with the arrays
    jmp     .Lblock
.Lsmall:
    test    rdx, rdx
    jz      .Ldone
.Lone:
    vmovsd      xmm0, qword ptr [rdi + rax*8]
    vsqrtsd     xmm0, xmm0, xmm0
    vmovsd      qword ptr [rsi + rax*8], xmm0
    inc     rax
    cmp     rax, rdx
    jb      .Lone
.Ldone:
    vzeroupper
    ret
.size fuse_sqrt_v_f64, .-fuse_sqrt_v_f64

.section .note.GNU-stack,"",@progbits
