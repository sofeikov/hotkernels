// void hypot_f64(const double *a, const double *b, double *out, size_t n);
//
// x0 = a, x1 = b, x2 = out, x3 = n. out[i] = hypot(a[i], b[i]).
//
// EXACT ONLY AGAINST ONE C LIBRARY. np.hypot calls the C library's hypot for
// every element, and hypot is not correctly rounded: libraries differ in which
// of two neighbouring doubles they return. This kernel computes the hypot of
// macOS's libsystem_m on arm64, as read from its instructions on macOS 27.2.1
// (attempts.md says how, and how it was checked), and matches NumPy bit for
// bit only where the C library is that algorithm. A binding that ships it
// must check it against the C library where it runs.
//
// The algorithm, for x = |a[i]|, y = |b[i]|, big = fmaxnm(x, y),
// small = fmin(x, y) (fmaxnm returns the number beside a quiet NaN, fmin the
// NaN), and eb, es the exponent fields of big and small:
//
//   eb = 0x7ff (infinity, or both NaN)     big + big
//   small = 0                              big
//   small NaN                              small + small
//   eb - es > 27                           small = 0, then on as below
//   eb >= 0x5fe (big >= 2^511)             core(big * 2^-768, small * 2^-768) * 2^768
//   eb < 0x21b (big < 2^-484)              core(big * 2^768, small * 2^768) * 2^-768
//   otherwise                              core(big, small)
//
//   core(big, small) = sqrt(fma(small, small, big * big))
//
// with big * big rounded on its own and then one rounding for the fused
// multiply-add. About 16% of pairs of standard normal values differ from the
// correctly rounded result by one unit in the last place.
//
// The vector path. Two elements per vector. big and small are computed with
// fmax and fmin, which agree with fmaxnm and fmin whenever neither input is a
// NaN. Then big's bits are tested as an unsigned integer for
// [2^-484, 2^511): fmax returns a NaN when either input is one, and a NaN's
// bits lie above that range, as do an infinity's. A vector of which every
// element passes computes sqrt(fma(small, small, big * big)), skipping two
// steps of the algorithm, each of which provably changes nothing there:
//
//   - The scaling: eb is in [0x21b, 0x5fe), where it multiplies by 1.0.
//   - Setting small to 0 when eb - es > 27, and returning big when small is
//     0. With eb - es >= 28, small < 2^(es - 1022) and big >= 2^(eb - 1023),
//     so small * small < 2^(2 eb - 2100), while B = big * big rounded is at
//     least 2^(2 eb - 2046) and its unit in the last place at least
//     2^(2 eb - 2098): small * small is under a quarter of that unit, so the
//     fused multiply-add rounds to B, which is what it gives with small = 0.
//     This needs round to nearest, which is why the vector path is taken only
//     when FPCR selects it (below). And with small = 0 the library returns big
//     where this path returns sqrt(B): equal, since sqrt(RN(big^2)) = big for
//     every double without overflow or underflow in the square (checked on
//     every pair of the test program with a zero, attempts.md).
//
// Anything else, a vector with an element out of that range or a NaN, goes to
// the scalar routine at the end of the file, which follows the library
// instruction for instruction; so do the last element of an odd n, and every
// element when FPCR is not in its default rounding mode or flushes denormals
// to zero.
//
// Floating-point flags. On the vector path the kernel raises inexact, and
// nothing else: the square of a big in [2^-484, 2^511) neither overflows nor
// underflows, and a fused multiply-add whose result is at least 2^-968 does
// not underflow, whatever small is. The library raises the same, apart from
// inexact, which it does not raise where it sets small to 0 and the square
// root is exact; NumPy does not report inexact. fmax and fmin raise invalid
// for a signalling NaN only, as the library's fmaxnm and fmin do on the same
// element. The scalar routine raises what the library raises.
//
// out may be exactly a or exactly b; it overlaps neither in any other way.
// Each vector's inputs are loaded before its output is stored, and the scalar
// routine loads an element before it stores it.
//
// Registers: x5 byte offset of the next element, x6 byte offset of the end,
// x7 end of the scalar routine's range, x16 where the scalar routine returns.
// v30 = 2^-484 as bits, v31 = 2^511 - 2^-484 as bits.

.text
.globl _hypot_f64
.p2align 2
_hypot_f64:
    mov     x5, #0
    lsl     x6, x3, #3
    // FPCR: RMode (bits 22-23) and FZ (bit 24) must be clear for the vector
    // path; otherwise everything goes to the scalar routine.
    mrs     x9, fpcr
    tst     x9, #0x01c00000
    b.ne    Lall_scalar
    movz    x9, #0x21b0, lsl #48
    dup     v30.2d, x9
    movz    x9, #0x3e30, lsl #48
    dup     v31.2d, x9

Lloop:
    add     x9, x5, #16
    cmp     x9, x6
    b.hi    Ltail
    ldr     q0, [x0, x5]
    ldr     q1, [x1, x5]
    fabs    v0.2d, v0.2d
    fabs    v1.2d, v1.2d
    fmax    v2.2d, v0.2d, v1.2d
    fmin    v3.2d, v0.2d, v1.2d
    // big's bits minus 2^-484's, unsigned at or above the width of the range:
    // out of [2^-484, 2^511), or a NaN
    sub     v4.2d, v2.2d, v30.2d
    cmhs    v4.2d, v4.2d, v31.2d
    addp    d4, v4.2d
    fmov    x10, d4
    cbnz    x10, Lslow
    fmul    v2.2d, v2.2d, v2.2d
    fmla    v2.2d, v3.2d, v3.2d
    fsqrt   v2.2d, v2.2d
    str     q2, [x2, x5]
    mov     x5, x9
    b       Lloop

Lslow:
    // This vector's two elements by the scalar routine, then back to the loop.
    mov     x7, x9
    adr     x16, Lloop
    b       Lscalar

Lall_scalar:
Ltail:
    mov     x7, x6
    adr     x16, Ldone
    b       Lscalar

Ldone:
    ret

// The scalar routine: elements from byte offset x5 up to x7, then br x16.
// The library's hypot, instruction for instruction.
Lscalar:
    cmp     x5, x7
    b.hs    Lscalar_out
    ldr     d0, [x0, x5]
    ldr     d1, [x1, x5]
    fabs    d2, d0
    fabs    d3, d1
    fmaxnm  d0, d2, d3
    fmov    x10, d0
    ubfx    x10, x10, #52, #11
    cmp     x10, #0x7ff
    b.eq    Ls_inf
    fmin    d1, d2, d3
    fmov    x11, d1
    ubfx    x11, x11, #52, #11
    fcmp    d1, #0.0
    b.eq    Ls_store
    fcmp    d1, d1
    b.ne    Ls_nan
    fmov    d2, xzr
    sub     x11, x10, x11
    cmp     x11, #27
    fcsel   d1, d2, d1, hi
    cmp     x10, #0x5fe
    b.hs    Ls_huge
    cmp     x10, #0x21b
    b.lo    Ls_tiny
    fmov    d2, #1.0
Ls_core:
    fmul    d0, d0, d0
    fmadd   d0, d1, d1, d0
    fsqrt   d0, d0
    fmul    d0, d0, d2
Ls_store:
    str     d0, [x2, x5]
    add     x5, x5, #8
    b       Lscalar
Ls_nan:
    fmov    d0, d1
Ls_inf:
    fadd    d0, d0, d0
    b       Ls_store
Ls_huge:
    // inputs by 2^-768, result by 2^768
    movz    x12, #0x0ff0, lsl #48
    fmov    d3, x12
    movz    x12, #0x6ff0, lsl #48
    fmov    d2, x12
    fmul    d0, d0, d3
    fmul    d1, d1, d3
    b       Ls_core
Ls_tiny:
    // inputs by 2^768, result by 2^-768
    movz    x12, #0x6ff0, lsl #48
    fmov    d3, x12
    movz    x12, #0x0ff0, lsl #48
    fmov    d2, x12
    fmul    d0, d0, d3
    fmul    d1, d1, d3
    b       Ls_core
Lscalar_out:
    br      x16
