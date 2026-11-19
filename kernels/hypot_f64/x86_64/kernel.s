.intel_syntax noprefix
# void hypot_f64(const double *a, const double *b, double *out, size_t n);
#
# rdi = a, rsi = b, rdx = out, rcx = n. out[i] = hypot(a[i], b[i]).
#
# EXACT ONLY AGAINST ONE C LIBRARY. np.hypot calls the C library's hypot for
# every element, and hypot is not correctly rounded: libraries differ in which
# of two neighbouring doubles they return for a fraction of a percent to about
# ten percent of pairs. This kernel computes the corrected algorithm of
# glibc's e_hypot.c (glibc from about 2.35), without fused multiply-add, and
# matches NumPy bit for bit only where the C library is that algorithm. A
# binding that ships it must check it against the C library where it runs.
#
# The algorithm, for x = |a[i]|, y = |b[i]|, ax = max(x, y), ay = min(x, y):
#
#   either input infinite, and neither a signalling NaN   +inf
#   either input NaN                                      a[i] + b[i]
#   ax > 2^511:   ay <= ax * 2^-54 ? ax + ay : core(ax * 2^-600, ay * 2^-600) * 2^600
#   ay < 2^-459:  ax >= ay * 2^54  ? ax + ay : core(ax * 2^600, ay * 2^600) * 2^-600,
#                 and the underflow flag raised when that result is below 2^-1022
#   otherwise:    ax >= ay * 2^54  ? ax + ay : core(ax, ay)
#
#   core(ax, ay):
#     h = sqrt(ax * ax + ay * ay)
#     if h <= 2 * ay: d = h - ay; t1 = ax * (2d - ax);      t2 = (d - 2(ax - ay)) * d
#     else:           d = h - ax; t1 = 2d * (ax - 2 ay);    t2 = (4d - ay) * ay + d * d
#     h - (t1 + t2) / (2h)
#
# Every operation in that order, each rounded on its own: a fused
# multiply-add changes results.
#
# out may be exactly a or exactly b; it overlaps neither in any other way.
# Every group of four loads its inputs before it stores its output, and the
# elements are written once, in order.
#
# Four elements per step. Both inputs of a group are tested first, on their
# bits, before any floating-point instruction touches them: when every
# element lies in [2^-459, 2^511] the group takes the common case, both
# branches of core computed and one taken by a blend, then ax + ay taken
# where ax >= ay * 2^54. A group with another element goes to a second test:
# when every element is finite and at most 2^511, and every pair with ay
# below 2^-459 has ax >= ay * 2^54 (a zero beside anything finite, for
# example), the group is computed as above with 1.0 put in place of the
# inputs of those pairs, whose result is ax + ay. Any other group goes to the
# scalar routine at the end of the file, which computes the whole algorithm
# one element at a time, as do the last one to three elements.
#
# Floating-point flags: on a group of the common case the kernel raises the
# flags the C library raises for the same elements, apart from one case: a
# pair with ax >= ay * 2^54 where ay * ay / ax is below 2^-1022, for which
# the unused core raises underflow and the library raises nothing. No
# arithmetic is done on a NaN or an infinity before the scalar routine has
# it, so a quiet NaN raises nothing.

.section .rodata
.p2align 5
# Clears the sign bit.
.Labs:      .quad 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff, 0x7fffffffffffffff
# The range test of the common case, on the bits of |x| as a signed integer:
# |x| is in [2^-459, 2^511] when bits + BIAS is not greater than LIMIT.
# BIAS = 2^63 - bits(2^-459); LIMIT = (bits(2^511) - bits(2^-459)) ^ 2^63.
.Lbias:     .quad 0x5cc0000000000000, 0x5cc0000000000000, 0x5cc0000000000000, 0x5cc0000000000000
.Llimit:    .quad 0xbca0000000000000, 0xbca0000000000000, 0xbca0000000000000, 0xbca0000000000000
# 2^54: ay below ax * 2^-54 is negligible beside it.
.Ltwo54:    .quad 0x4350000000000000, 0x4350000000000000, 0x4350000000000000, 0x4350000000000000
# 2^511, the largest ax that needs no scaling, as bits and as a double.
.Llarge:    .quad 0x5fe0000000000000, 0x5fe0000000000000, 0x5fe0000000000000, 0x5fe0000000000000
# 2^-459, the smallest ay that needs no scaling.
.Ltiny:     .quad 0x2340000000000000, 0x2340000000000000, 0x2340000000000000, 0x2340000000000000
.Lone:      .quad 0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000, 0x3ff0000000000000
# Scalar constants.
# 2^-54.
.Leps:      .quad 0x3c90000000000000
# 2^-600 and 2^600, the scale factors.
.Lsmall:    .quad 0x1a70000000000000
.Lbig:      .quad 0x6570000000000000
# 2^-1022, the smallest normal double.
.Lmin:      .quad 0x0010000000000000
.Linf:      .quad 0x7ff0000000000000
# The bits of an infinity shifted left by one, sign dropped: a value whose
# shifted bits are at least this is not finite, above it a NaN.
.Linf2:     .quad 0xffe0000000000000

# core(ax, ay) on four lanes, result in ymm4. Uses ymm5 to ymm11.
.macro CORE x, y
    vmulpd      ymm4, \x, \x
    vmulpd      ymm5, \y, \y
    vaddpd      ymm4, ymm4, ymm5
    vsqrtpd     ymm4, ymm4                  # h
    vaddpd      ymm5, \y, \y                # 2 ay
    vcmppd      ymm6, ymm4, ymm5, 0x12      # h <= 2 ay (LE_OQ)
    # h <= 2 ay
    vsubpd      ymm7, ymm4, \y              # d = h - ay
    vaddpd      ymm8, ymm7, ymm7
    vsubpd      ymm8, ymm8, \x
    vmulpd      ymm8, ymm8, \x              # t1 = ax * (2d - ax)
    vsubpd      ymm9, \x, \y
    vaddpd      ymm9, ymm9, ymm9
    vsubpd      ymm9, ymm7, ymm9
    vmulpd      ymm9, ymm9, ymm7            # t2 = (d - 2 (ax - ay)) * d
    # h > 2 ay
    vsubpd      ymm7, ymm4, \x              # d = h - ax
    vaddpd      ymm10, ymm7, ymm7           # 2d
    vsubpd      ymm11, \x, ymm5
    vmulpd      ymm11, ymm10, ymm11         # t1 = 2d * (ax - 2 ay)
    vaddpd      ymm10, ymm10, ymm10         # 4d
    vsubpd      ymm10, ymm10, \y
    vmulpd      ymm10, ymm10, \y
    vmulpd      ymm7, ymm7, ymm7
    vaddpd      ymm10, ymm10, ymm7          # t2 = (4d - ay) * ay + d * d
    vblendvpd   ymm8, ymm11, ymm8, ymm6
    vblendvpd   ymm9, ymm10, ymm9, ymm6
    vaddpd      ymm8, ymm8, ymm9
    vaddpd      ymm9, ymm4, ymm4
    vdivpd      ymm8, ymm8, ymm9
    vsubpd      ymm4, ymm4, ymm8            # h - (t1 + t2) / (2h)
.endm

# ax + ay in ymm4 where ax >= ay * 2^54. ymm12 holds 2^54. Uses ymm5, ymm6.
.macro NEGLIGIBLE x, y
    vmulpd      ymm5, \y, ymm12
    vcmppd      ymm5, \x, ymm5, 0x1d        # ax >= ay * 2^54 (GE_OQ)
    vaddpd      ymm6, \x, \y
    vblendvpd   ymm4, ymm4, ymm6, ymm5
.endm

.text
.globl hypot_f64
.type hypot_f64, @function
.p2align 4
hypot_f64:
    vmovapd     ymm15, [rip + .Labs]
    vmovapd     ymm14, [rip + .Lbias]
    vmovapd     ymm13, [rip + .Llimit]
    vmovapd     ymm12, [rip + .Ltwo54]
    mov         r10, rcx
    and         r10, -4                     # end of the whole groups
    lea         r11, [rcx - 7]              # a pair of groups starts below this,
                                            # compared as signed: n - 7 may be negative
    xor         eax, eax

# Two groups at a time while eight elements are left, one while four are.
.Ldispatch:
    cmp         rax, r11
    jl          .Lpair
    cmp         rax, r10
    jb          .Lloop
    jmp         .Ltail

# Two groups, tested together. When either has an element outside
# [2^-459, 2^511] the first is done alone, and the second after it.
.p2align 4
.Lpair:
    vandpd      ymm0, ymm15, [rdi + rax*8]
    vandpd      ymm1, ymm15, [rsi + rax*8]
    vandpd      ymm2, ymm15, [rdi + rax*8 + 32]
    vandpd      ymm3, ymm15, [rsi + rax*8 + 32]
    vpaddq      ymm4, ymm0, ymm14
    vpaddq      ymm5, ymm1, ymm14
    vpaddq      ymm6, ymm2, ymm14
    vpaddq      ymm7, ymm3, ymm14
    vpcmpgtq    ymm4, ymm4, ymm13
    vpcmpgtq    ymm5, ymm5, ymm13
    vpcmpgtq    ymm6, ymm6, ymm13
    vpcmpgtq    ymm7, ymm7, ymm13
    vpor        ymm4, ymm4, ymm5
    vpor        ymm6, ymm6, ymm7
    vpor        ymm4, ymm4, ymm6
    vptest      ymm4, ymm4
    jnz         .Lloop
    vminpd      ymm5, ymm0, ymm1
    vmaxpd      ymm0, ymm0, ymm1            # ax of the first group
    vmovapd     ymm1, ymm5                  # ay
    CORE        ymm0, ymm1
    NEGLIGIBLE  ymm0, ymm1
    vmovupd     [rdx + rax*8], ymm4
    vminpd      ymm5, ymm2, ymm3
    vmaxpd      ymm2, ymm2, ymm3            # ax of the second group
    vmovapd     ymm3, ymm5                  # ay
    CORE        ymm2, ymm3
    NEGLIGIBLE  ymm2, ymm3
    vmovupd     [rdx + rax*8 + 32], ymm4
    add         rax, 8
    cmp         rax, r11
    jl          .Lpair
    jmp         .Ldispatch

# One group.
.p2align 4
.Lloop:
    vandpd      ymm0, ymm15, [rdi + rax*8]  # |a|
    vandpd      ymm1, ymm15, [rsi + rax*8]  # |b|
    vpaddq      ymm2, ymm0, ymm14
    vpaddq      ymm3, ymm1, ymm14
    vpcmpgtq    ymm2, ymm2, ymm13
    vpcmpgtq    ymm3, ymm3, ymm13
    vpor        ymm2, ymm2, ymm3
    vptest      ymm2, ymm2
    jnz         .Lother                     # an element outside [2^-459, 2^511]
    vmaxpd      ymm2, ymm0, ymm1            # ax
    vminpd      ymm3, ymm0, ymm1            # ay
    CORE        ymm2, ymm3
    NEGLIGIBLE  ymm2, ymm3
    vmovupd     [rdx + rax*8], ymm4
    add         rax, 4
    jmp         .Ldispatch

# A group with an element outside [2^-459, 2^511]. ymm0 = |a|, ymm1 = |b|.
.Lother:
    vmovapd     ymm4, [rip + .Llarge]
    vpcmpgtq    ymm2, ymm0, ymm4
    vpcmpgtq    ymm3, ymm1, ymm4
    vpor        ymm2, ymm2, ymm3
    vptest      ymm2, ymm2
    jnz         .Lgroup_scalar              # above 2^511, infinite or NaN
    vmaxpd      ymm2, ymm0, ymm1
    vminpd      ymm1, ymm0, ymm1            # ay
    vmovapd     ymm0, ymm2                  # ax
    vcmppd      ymm6, ymm1, [rip + .Ltiny], 0x11   # ay < 2^-459 (LT_OQ)
    vmulpd      ymm5, ymm1, ymm12
    vcmppd      ymm5, ymm0, ymm5, 0x1d      # ax >= ay * 2^54
    vandnpd     ymm5, ymm5, ymm6            # tiny ay and not negligible
    vptest      ymm5, ymm5
    jnz         .Lgroup_scalar
    vmovapd     ymm7, [rip + .Lone]
    vblendvpd   ymm2, ymm0, ymm7, ymm6      # 1.0 in place of those pairs
    vblendvpd   ymm3, ymm1, ymm7, ymm6
    CORE        ymm2, ymm3
    NEGLIGIBLE  ymm0, ymm1
    vmovupd     [rdx + rax*8], ymm4
    add         rax, 4
    jmp         .Ldispatch

.Lgroup_scalar:
    lea         r8, [rax + 4]
    jmp         .Lscalar

.Ltail:
    cmp         rax, rcx
    jae         .Ldone
    mov         r8, rcx

# Elements rax to r8 - 1 one at a time, then back to the groups or the tail.
.Lscalar:
    mov         r9, [rdi + rax*8]
    add         r9, r9                      # bits of |a| shifted left by one
    cmp         r9, [rip + .Linf2]
    jae         .Lnot_finite
    mov         r9, [rsi + rax*8]
    add         r9, r9
    cmp         r9, [rip + .Linf2]
    jae         .Lnot_finite
    vmovsd      xmm0, [rdi + rax*8]         # one element: a vandpd from memory
    vmovsd      xmm1, [rsi + rax*8]         # would read sixteen bytes
    vandpd      xmm0, xmm0, xmm15
    vandpd      xmm1, xmm1, xmm15
    vmaxsd      xmm2, xmm0, xmm1            # ax
    vminsd      xmm3, xmm0, xmm1            # ay
    vucomisd    xmm2, [rip + .Llarge]
    ja          .Lscalar_large
    vucomisd    xmm3, [rip + .Ltiny]
    jb          .Lscalar_tiny
    vmulsd      xmm5, xmm3, xmm12
    vucomisd    xmm2, xmm5
    jae         .Lscalar_sum
    mov         r9, 0x3ff0000000000000      # 1.0: no scaling
    jmp         .Lscalar_core
.Lscalar_large:
    vmulsd      xmm5, xmm2, [rip + .Leps]
    vucomisd    xmm3, xmm5
    jbe         .Lscalar_sum
    vmulsd      xmm2, xmm2, [rip + .Lsmall]
    vmulsd      xmm3, xmm3, [rip + .Lsmall]
    mov         r9, [rip + .Lbig]
    jmp         .Lscalar_core
.Lscalar_tiny:
    vmulsd      xmm5, xmm3, xmm12
    vucomisd    xmm2, xmm5
    jae         .Lscalar_sum
    vmulsd      xmm2, xmm2, [rip + .Lbig]
    vmulsd      xmm3, xmm3, [rip + .Lbig]
    mov         r9, [rip + .Lsmall]
# core(xmm2, xmm3), then times the double whose bits are in r9.
.Lscalar_core:
    vmulsd      xmm4, xmm2, xmm2
    vmulsd      xmm5, xmm3, xmm3
    vaddsd      xmm4, xmm4, xmm5
    vsqrtsd     xmm4, xmm4, xmm4            # h
    vaddsd      xmm5, xmm3, xmm3            # 2 ay
    vucomisd    xmm4, xmm5
    ja          .Lscalar_wide
    vsubsd      xmm7, xmm4, xmm3            # d = h - ay
    vaddsd      xmm8, xmm7, xmm7
    vsubsd      xmm8, xmm8, xmm2
    vmulsd      xmm8, xmm8, xmm2            # t1
    vsubsd      xmm9, xmm2, xmm3
    vaddsd      xmm9, xmm9, xmm9
    vsubsd      xmm9, xmm7, xmm9
    vmulsd      xmm9, xmm9, xmm7            # t2
    jmp         .Lscalar_correct
.Lscalar_wide:
    vsubsd      xmm7, xmm4, xmm2            # d = h - ax
    vaddsd      xmm10, xmm7, xmm7           # 2d
    vsubsd      xmm8, xmm2, xmm5
    vmulsd      xmm8, xmm10, xmm8           # t1
    vaddsd      xmm10, xmm10, xmm10         # 4d
    vsubsd      xmm10, xmm10, xmm3
    vmulsd      xmm10, xmm10, xmm3
    vmulsd      xmm7, xmm7, xmm7
    vaddsd      xmm9, xmm10, xmm7           # t2
.Lscalar_correct:
    vaddsd      xmm8, xmm8, xmm9
    vaddsd      xmm9, xmm4, xmm4
    vdivsd      xmm8, xmm8, xmm9
    vsubsd      xmm4, xmm4, xmm8
    vmovq       xmm5, r9
    vmulsd      xmm4, xmm4, xmm5
    vucomisd    xmm4, [rip + .Lmin]
    jae         .Lscalar_store
    vmovsd      xmm5, [rip + .Lmin]         # below 2^-1022: raise underflow, as
    vmulsd      xmm5, xmm5, xmm5            # the C library does
    jmp         .Lscalar_store
.Lscalar_sum:
    vaddsd      xmm4, xmm2, xmm3
    jmp         .Lscalar_store

# An infinity or a NaN among the two. +inf when either is infinite and
# neither is a signalling NaN; otherwise a + b, which is a NaN.
.Lnot_finite:
    vmovsd      xmm0, [rdi + rax*8]
    vmovsd      xmm1, [rsi + rax*8]
    mov         r9, [rdi + rax*8]
    add         r9, r9
    cmp         r9, [rip + .Linf2]
    jbe         1f                          # a is not a NaN
    bt          r9, 52                      # the quiet bit, shifted
    jnc         .Lscalar_nan                # a is a signalling NaN
1:
    mov         r9, [rsi + rax*8]
    add         r9, r9
    cmp         r9, [rip + .Linf2]
    jbe         2f
    bt          r9, 52
    jnc         .Lscalar_nan
2:
    mov         r9, [rdi + rax*8]
    add         r9, r9
    cmp         r9, [rip + .Linf2]
    je          .Lscalar_inf
    mov         r9, [rsi + rax*8]
    add         r9, r9
    cmp         r9, [rip + .Linf2]
    je          .Lscalar_inf
.Lscalar_nan:
    vaddsd      xmm4, xmm0, xmm1
    jmp         .Lscalar_store
.Lscalar_inf:
    vmovsd      xmm4, [rip + .Linf]

.Lscalar_store:
    vmovsd      [rdx + rax*8], xmm4
    inc         rax
    cmp         rax, r8
    jb          .Lscalar
    cmp         rax, r10
    jb          .Ldispatch
    cmp         rax, rcx
    jb          .Ltail

.Ldone:
    vzeroupper
    ret
.size hypot_f64, .-hypot_f64

.section .note.GNU-stack,"",@progbits
