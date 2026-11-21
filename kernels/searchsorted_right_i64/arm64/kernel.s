// void searchsorted_right_i64(const int64_t *a, const int64_t *v, int64_t *out, size_t n,
//                                size_t needles);
//
// a holds n int64 values sorted ascending; v holds `needles` values to search
// for. For each needle, out[j] = the first index i with !(a[i] <= v[j]), or n
// if there is none: np.searchsorted(a, v, side='right').
//
// The search is a branchless lower bound with power-of-two steps. Let p be
// the largest power of two <= n. Each needle keeps an index `base`, with the
// answer always in [base, base + len] and base + len <= n:
//     first:  base = (a[p - 1] <= key) ? n - p : 0          len = p
//     then for h = p/2, p/4, ..., 1:
//             if (a[base + h - 1] <= key) base += h          len = h
//     last:   if (a[base] <= key) base += 1
// The step size depends only on n, so many searches run in step and their
// loads overlap. When n is a power of two the first step always gives 0, so
// the step h = p/2 takes its place: its probe is always a[p/2 - 1], loaded
// once per call, and every search is one load shorter.
//
// Four code paths:
//   n == 0                  every answer is 0
//   n >= 65,536            the wide path (Lwide): the needles in blocks of up
//                           to 256, each block swept once per step with its bases
//                           kept in out, as NumPy does. No probe waits on
//                           another, so the probes of dozens of needles are in
//                           flight at once while the array is out of the cache.
//   fewer than 16 needles  one needle at a time (Lscalar)
//   otherwise               16 needles at once (Lgroup), the rest of this note
//
// ---- The group path
//
// A group holds 16 needles whose bases are all in integer registers. They
// are compared in two ways, alternately, because a probe of either kind alone
// leaves part of the core idle:
//   10 with the flags:  ldr t, [&a[h - 1], base, lsl #3]; add u, base, h
//                   cmp t, key; csel base, u, base, le
//               a chain of about 6 cycles per step, but this core runs only
//               about three flag-setting or flag-reading instructions per
//               cycle, and each needle needs a second register for its key.
//   6 in a vector register:  ldr d, [&a[h - 1], base, lsl #3]
//                   cmge d, key, d; and with h; fmov to an integer; add
//               no flags and no integer register for the key, but a chain
//               of about 13 cycles per step.
// The keys sit in v0-v7 and v16-v23, in both lanes; the flag needles also
// have them in integer registers. x19-x30 are saved on the stack, and
// so is the state of the call between groups. Leftover needles are covered
// by one more group that ends exactly at the last needle and recomputes a
// few answers.
//
// ---- The endings
//
// A search does not run its last steps one probe at a time. Once its window
// is short, it loads the whole window and counts how many elements are
// <= the key with vector compares; base minus the sum of the masks is the
// answer. In the group path (from 16 elements) the window is four
// elements, after the step h = 4: one 32-byte load and two compares, one
// chain of about 16 cycles in place of three probes. In the wide path it is
// eight, after the step h = 8: one 64-byte load and four compares in one
// sweep, in place of four sweeps.
.text
.globl _searchsorted_right_i64
.p2align 2
_searchsorted_right_i64:
    cbz     x4, Lret
    cbz     x3, Lzero
    clz     x5, x3
    mov     x6, #0x8000000000000000
    lsr     x6, x6, x5              // p
    sub     x7, x3, x6              // n - p
    add     x8, x0, x6, lsl #3
    ldur    x8, [x8, #-8]           // a[p - 1]
    lsr     x6, x6, #1              // h0 = p / 2
    cbnz    x7, Lsteps_set
    cbz     x6, Lsteps_set
    mov     x7, x6                  // n = p: the first step is h = p / 2,
    add     x8, x0, x6, lsl #3      // whose probe is always a[p/2 - 1]
    ldur    x8, [x8, #-8]
    lsr     x6, x6, #1
Lsteps_set:
    mov     x9, #65536
    cmp     x3, x9
    b.hs    Lwide
    cmp     x4, #16
    b.lo    Lscalar
    sub     sp, sp, #192
    stp     x19, x20, [sp, #0]
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    stp     x27, x28, [sp, #64]
    stp     x29, x30, [sp, #80]
    stp     x0, x1, [sp, #96]
    stp     x2, x4, [sp, #112]
    stp     x7, x8, [sp, #128]
    stp     x6, xzr, [sp, #144]
    cmp     x3, #16
    b.hs    Lt_group
Lg_group:
    ldr     x16, [sp, #152]
    ldr     x17, [sp, #120]
    sub     x17, x17, #16
    cmp     x16, x17
    csel    x16, x16, x17, lo        // j = min(i, needles - G)
    add     x17, x16, #16
    stp     x17, x16, [sp, #152]
    ldr     x17, [sp, #104]
    add     x16, x17, x16, lsl #3      // &v[j]
    mov     x17, x16
    ldp     x21, x22, [x16, #0]
    ldp     x23, x24, [x16, #24]
    ldp     x25, x26, [x16, #48]
    ldp     x27, x28, [x16, #72]
    ldp     x29, x30, [x16, #96]
    ld1r    {v0.2d}, [x17], #8
    ld1r    {v1.2d}, [x17], #8
    ld1r    {v2.2d}, [x17], #8
    ld1r    {v3.2d}, [x17], #8
    ld1r    {v4.2d}, [x17], #8
    ld1r    {v5.2d}, [x17], #8
    ld1r    {v6.2d}, [x17], #8
    ld1r    {v7.2d}, [x17], #8
    ld1r    {v16.2d}, [x17], #8
    ld1r    {v17.2d}, [x17], #8
    ld1r    {v18.2d}, [x17], #8
    ld1r    {v19.2d}, [x17], #8
    ld1r    {v20.2d}, [x17], #8
    ld1r    {v21.2d}, [x17], #8
    ld1r    {v22.2d}, [x17], #8
    ld1r    {v23.2d}, [x17], #8
    ldp     x1, x0, [sp, #128]
    fmov    d26, x0                // a[p - 1]
    cmp     x0, x21
    csel    x2, x1, xzr, le
    cmp     x0, x22
    csel    x3, x1, xzr, le
    cmge    d27, d2, d26
    fmov    x4, d27
    and     x4, x4, x1
    cmp     x0, x23
    csel    x5, x1, xzr, le
    cmp     x0, x24
    csel    x6, x1, xzr, le
    cmge    d27, d5, d26
    fmov    x7, d27
    and     x7, x7, x1
    cmp     x0, x25
    csel    x8, x1, xzr, le
    cmp     x0, x26
    csel    x9, x1, xzr, le
    cmge    d27, d16, d26
    fmov    x10, d27
    and     x10, x10, x1
    cmp     x0, x27
    csel    x11, x1, xzr, le
    cmp     x0, x28
    csel    x12, x1, xzr, le
    cmge    d27, d19, d26
    fmov    x13, d27
    and     x13, x13, x1
    cmp     x0, x29
    csel    x14, x1, xzr, le
    cmp     x0, x30
    csel    x15, x1, xzr, le
    cmge    d27, d22, d26
    fmov    x19, d27
    and     x19, x19, x1
    cmge    d27, d23, d26
    fmov    x20, d27
    and     x20, x20, x1
    ldr     x17, [sp, #144]
    ldr     x16, [sp, #96]
    cbz     x17, Lg_final
    add     x16, x16, x17, lsl #3
    sub     x16, x16, #8              // &a[h - 1]
Lg_level:
    fmov    d28, x17
    ldr     x0, [x16, x2, lsl #3]
    add     x1, x2, x17
    cmp     x0, x21
    csel    x2, x1, x2, le
    ldr     x0, [x16, x3, lsl #3]
    add     x1, x3, x17
    cmp     x0, x22
    csel    x3, x1, x3, le
    ldr     d27, [x16, x4, lsl #3]
    cmge    d27, d2, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x4, x4, x1
    ldr     x0, [x16, x5, lsl #3]
    add     x1, x5, x17
    cmp     x0, x23
    csel    x5, x1, x5, le
    ldr     x0, [x16, x6, lsl #3]
    add     x1, x6, x17
    cmp     x0, x24
    csel    x6, x1, x6, le
    ldr     d27, [x16, x7, lsl #3]
    cmge    d27, d5, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x7, x7, x1
    ldr     x0, [x16, x8, lsl #3]
    add     x1, x8, x17
    cmp     x0, x25
    csel    x8, x1, x8, le
    ldr     x0, [x16, x9, lsl #3]
    add     x1, x9, x17
    cmp     x0, x26
    csel    x9, x1, x9, le
    ldr     d27, [x16, x10, lsl #3]
    cmge    d27, d16, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x10, x10, x1
    ldr     x0, [x16, x11, lsl #3]
    add     x1, x11, x17
    cmp     x0, x27
    csel    x11, x1, x11, le
    ldr     x0, [x16, x12, lsl #3]
    add     x1, x12, x17
    cmp     x0, x28
    csel    x12, x1, x12, le
    ldr     d27, [x16, x13, lsl #3]
    cmge    d27, d19, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x13, x13, x1
    ldr     x0, [x16, x14, lsl #3]
    add     x1, x14, x17
    cmp     x0, x29
    csel    x14, x1, x14, le
    ldr     x0, [x16, x15, lsl #3]
    add     x1, x15, x17
    cmp     x0, x30
    csel    x15, x1, x15, le
    ldr     d27, [x16, x19, lsl #3]
    cmge    d27, d22, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x19, x19, x1
    ldr     d27, [x16, x20, lsl #3]
    cmge    d27, d23, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x20, x20, x1
    lsr     x17, x17, #1
    sub     x16, x16, x17, lsl #3
    cbnz    x17, Lg_level
Lg_final:
    ldr     x0, [x16, x2, lsl #3]
    cmp     x0, x21
    cinc    x2, x2, le
    ldr     x0, [x16, x3, lsl #3]
    cmp     x0, x22
    cinc    x3, x3, le
    ldr     d27, [x16, x4, lsl #3]
    cmge    d27, d2, d27
    fmov    x1, d27
    sub     x4, x4, x1
    ldr     x0, [x16, x5, lsl #3]
    cmp     x0, x23
    cinc    x5, x5, le
    ldr     x0, [x16, x6, lsl #3]
    cmp     x0, x24
    cinc    x6, x6, le
    ldr     d27, [x16, x7, lsl #3]
    cmge    d27, d5, d27
    fmov    x1, d27
    sub     x7, x7, x1
    ldr     x0, [x16, x8, lsl #3]
    cmp     x0, x25
    cinc    x8, x8, le
    ldr     x0, [x16, x9, lsl #3]
    cmp     x0, x26
    cinc    x9, x9, le
    ldr     d27, [x16, x10, lsl #3]
    cmge    d27, d16, d27
    fmov    x1, d27
    sub     x10, x10, x1
    ldr     x0, [x16, x11, lsl #3]
    cmp     x0, x27
    cinc    x11, x11, le
    ldr     x0, [x16, x12, lsl #3]
    cmp     x0, x28
    cinc    x12, x12, le
    ldr     d27, [x16, x13, lsl #3]
    cmge    d27, d19, d27
    fmov    x1, d27
    sub     x13, x13, x1
    ldr     x0, [x16, x14, lsl #3]
    cmp     x0, x29
    cinc    x14, x14, le
    ldr     x0, [x16, x15, lsl #3]
    cmp     x0, x30
    cinc    x15, x15, le
    ldr     d27, [x16, x19, lsl #3]
    cmge    d27, d22, d27
    fmov    x1, d27
    sub     x19, x19, x1
    ldr     d27, [x16, x20, lsl #3]
    cmge    d27, d23, d27
    fmov    x1, d27
    sub     x20, x20, x1
    ldr     x16, [sp, #112]
    ldr     x17, [sp, #160]
    add     x16, x16, x17, lsl #3
    stp     x2, x3, [x16, #0]
    stp     x4, x5, [x16, #16]
    stp     x6, x7, [x16, #32]
    stp     x8, x9, [x16, #48]
    stp     x10, x11, [x16, #64]
    stp     x12, x13, [x16, #80]
    stp     x14, x15, [x16, #96]
    stp     x19, x20, [x16, #112]
    ldr     x16, [sp, #152]
    ldr     x17, [sp, #120]
    cmp     x16, x17
    b.lo    Lg_group
    b       Lgroups_done
Lt_group:
    ldr     x16, [sp, #152]
    ldr     x17, [sp, #120]
    sub     x17, x17, #16
    cmp     x16, x17
    csel    x16, x16, x17, lo        // j = min(i, needles - G)
    add     x17, x16, #16
    stp     x17, x16, [sp, #152]
    ldr     x17, [sp, #104]
    add     x16, x17, x16, lsl #3      // &v[j]
    mov     x17, x16
    ldp     x21, x22, [x16, #0]
    ldp     x23, x24, [x16, #24]
    ldp     x25, x26, [x16, #48]
    ldp     x27, x28, [x16, #72]
    ldp     x29, x30, [x16, #96]
    ld1r    {v0.2d}, [x17], #8
    ld1r    {v1.2d}, [x17], #8
    ld1r    {v2.2d}, [x17], #8
    ld1r    {v3.2d}, [x17], #8
    ld1r    {v4.2d}, [x17], #8
    ld1r    {v5.2d}, [x17], #8
    ld1r    {v6.2d}, [x17], #8
    ld1r    {v7.2d}, [x17], #8
    ld1r    {v16.2d}, [x17], #8
    ld1r    {v17.2d}, [x17], #8
    ld1r    {v18.2d}, [x17], #8
    ld1r    {v19.2d}, [x17], #8
    ld1r    {v20.2d}, [x17], #8
    ld1r    {v21.2d}, [x17], #8
    ld1r    {v22.2d}, [x17], #8
    ld1r    {v23.2d}, [x17], #8
    ldp     x1, x0, [sp, #128]
    fmov    d26, x0                // a[p - 1]
    cmp     x0, x21
    csel    x2, x1, xzr, le
    cmp     x0, x22
    csel    x3, x1, xzr, le
    cmge    d27, d2, d26
    fmov    x4, d27
    and     x4, x4, x1
    cmp     x0, x23
    csel    x5, x1, xzr, le
    cmp     x0, x24
    csel    x6, x1, xzr, le
    cmge    d27, d5, d26
    fmov    x7, d27
    and     x7, x7, x1
    cmp     x0, x25
    csel    x8, x1, xzr, le
    cmp     x0, x26
    csel    x9, x1, xzr, le
    cmge    d27, d16, d26
    fmov    x10, d27
    and     x10, x10, x1
    cmp     x0, x27
    csel    x11, x1, xzr, le
    cmp     x0, x28
    csel    x12, x1, xzr, le
    cmge    d27, d19, d26
    fmov    x13, d27
    and     x13, x13, x1
    cmp     x0, x29
    csel    x14, x1, xzr, le
    cmp     x0, x30
    csel    x15, x1, xzr, le
    cmge    d27, d22, d26
    fmov    x19, d27
    and     x19, x19, x1
    cmge    d27, d23, d26
    fmov    x20, d27
    and     x20, x20, x1
    ldr     x17, [sp, #144]
    ldr     x16, [sp, #96]
    add     x16, x16, x17, lsl #3
    sub     x16, x16, #8              // &a[h - 1]
Lt_level:
    fmov    d28, x17
    ldr     x0, [x16, x2, lsl #3]
    add     x1, x2, x17
    cmp     x0, x21
    csel    x2, x1, x2, le
    ldr     x0, [x16, x3, lsl #3]
    add     x1, x3, x17
    cmp     x0, x22
    csel    x3, x1, x3, le
    ldr     d27, [x16, x4, lsl #3]
    cmge    d27, d2, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x4, x4, x1
    ldr     x0, [x16, x5, lsl #3]
    add     x1, x5, x17
    cmp     x0, x23
    csel    x5, x1, x5, le
    ldr     x0, [x16, x6, lsl #3]
    add     x1, x6, x17
    cmp     x0, x24
    csel    x6, x1, x6, le
    ldr     d27, [x16, x7, lsl #3]
    cmge    d27, d5, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x7, x7, x1
    ldr     x0, [x16, x8, lsl #3]
    add     x1, x8, x17
    cmp     x0, x25
    csel    x8, x1, x8, le
    ldr     x0, [x16, x9, lsl #3]
    add     x1, x9, x17
    cmp     x0, x26
    csel    x9, x1, x9, le
    ldr     d27, [x16, x10, lsl #3]
    cmge    d27, d16, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x10, x10, x1
    ldr     x0, [x16, x11, lsl #3]
    add     x1, x11, x17
    cmp     x0, x27
    csel    x11, x1, x11, le
    ldr     x0, [x16, x12, lsl #3]
    add     x1, x12, x17
    cmp     x0, x28
    csel    x12, x1, x12, le
    ldr     d27, [x16, x13, lsl #3]
    cmge    d27, d19, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x13, x13, x1
    ldr     x0, [x16, x14, lsl #3]
    add     x1, x14, x17
    cmp     x0, x29
    csel    x14, x1, x14, le
    ldr     x0, [x16, x15, lsl #3]
    add     x1, x15, x17
    cmp     x0, x30
    csel    x15, x1, x15, le
    ldr     d27, [x16, x19, lsl #3]
    cmge    d27, d22, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x19, x19, x1
    ldr     d27, [x16, x20, lsl #3]
    cmge    d27, d23, d27
    and     v27.8b, v27.8b, v28.8b
    fmov    x1, d27
    add     x20, x20, x1
    lsr     x17, x17, #1
    sub     x16, x16, x17, lsl #3
    cmp     x17, #2
    b.hi    Lt_level
    ldr     x17, [sp, #96]
    add     x0, x17, x2, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v0.2d, v24.2d
    cmge    v25.2d, v0.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x2, x2, x1
    add     x0, x17, x3, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v1.2d, v24.2d
    cmge    v25.2d, v1.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x3, x3, x1
    add     x0, x17, x4, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v2.2d, v24.2d
    cmge    v25.2d, v2.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x4, x4, x1
    add     x0, x17, x5, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v3.2d, v24.2d
    cmge    v25.2d, v3.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x5, x5, x1
    add     x0, x17, x6, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v4.2d, v24.2d
    cmge    v25.2d, v4.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x6, x6, x1
    add     x0, x17, x7, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v5.2d, v24.2d
    cmge    v25.2d, v5.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x7, x7, x1
    add     x0, x17, x8, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v6.2d, v24.2d
    cmge    v25.2d, v6.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x8, x8, x1
    add     x0, x17, x9, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v7.2d, v24.2d
    cmge    v25.2d, v7.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x9, x9, x1
    add     x0, x17, x10, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v16.2d, v24.2d
    cmge    v25.2d, v16.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x10, x10, x1
    add     x0, x17, x11, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v17.2d, v24.2d
    cmge    v25.2d, v17.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x11, x11, x1
    add     x0, x17, x12, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v18.2d, v24.2d
    cmge    v25.2d, v18.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x12, x12, x1
    add     x0, x17, x13, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v19.2d, v24.2d
    cmge    v25.2d, v19.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x13, x13, x1
    add     x0, x17, x14, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v20.2d, v24.2d
    cmge    v25.2d, v20.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x14, x14, x1
    add     x0, x17, x15, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v21.2d, v24.2d
    cmge    v25.2d, v21.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x15, x15, x1
    add     x0, x17, x19, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v22.2d, v24.2d
    cmge    v25.2d, v22.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x19, x19, x1
    add     x0, x17, x20, lsl #3
    ld1     {v24.2d, v25.2d}, [x0]
    cmge    v24.2d, v23.2d, v24.2d
    cmge    v25.2d, v23.2d, v25.2d
    add     v24.2d, v24.2d, v25.2d
    addp    d24, v24.2d
    fmov    x1, d24
    sub     x20, x20, x1
    ldr     x16, [sp, #112]
    ldr     x17, [sp, #160]
    add     x16, x16, x17, lsl #3
    stp     x2, x3, [x16, #0]
    stp     x4, x5, [x16, #16]
    stp     x6, x7, [x16, #32]
    stp     x8, x9, [x16, #48]
    stp     x10, x11, [x16, #64]
    stp     x12, x13, [x16, #80]
    stp     x14, x15, [x16, #96]
    stp     x19, x20, [x16, #112]
    ldr     x16, [sp, #152]
    ldr     x17, [sp, #120]
    cmp     x16, x17
    b.lo    Lt_group
    b       Lgroups_done
Lgroups_done:
    ldp     x19, x20, [sp, #0]
    ldp     x21, x22, [sp, #16]
    ldp     x23, x24, [sp, #32]
    ldp     x25, x26, [sp, #48]
    ldp     x27, x28, [sp, #64]
    ldp     x29, x30, [sp, #80]
    add     sp, sp, #192
Lret:
    ret

// The wide path. x11 = needles in this block, x17 = h, x9 = &a[h - 1].
// Sweeps run while h >= 8, then one sweep counts over eight elements.
Lwide:
    sub     sp, sp, #64
    stp     x19, x20, [sp, #0]
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
Lw_block:
    mov     x11, #256
    cmp     x4, x11
    csel    x11, x4, x11, lo        // needles in this block
    mov     x12, x1
    mov     x13, x2
    mov     x14, x11
Lw_init:
    ldr     x15, [x12], #8
    cmp     x8, x15
    csel    x16, x7, xzr, le
    str     x16, [x13], #8
    subs    x14, x14, #1
    b.ne    Lw_init
    cmp     x6, #1
    csinc   x17, x6, xzr, hs        // h = max(h0, 1)
Lw_level:
    add     x9, x0, x17, lsl #3
    sub     x9, x9, #8              // &a[h - 1]
    mov     x12, x1
    mov     x13, x2
    lsr     x14, x11, #2
    cbz     x14, Lw_rest
Lw_quad:
    ldp     x19, x20, [x13]
    ldp     x21, x22, [x13, #16]
    ldp     x23, x24, [x12], #16
    ldp     x25, x26, [x12], #16
    ldr     x15, [x9, x19, lsl #3]
    add     x16, x19, x17
    cmp     x15, x23
    csel    x19, x16, x19, le
    ldr     x15, [x9, x20, lsl #3]
    add     x16, x20, x17
    cmp     x15, x24
    csel    x20, x16, x20, le
    ldr     x15, [x9, x21, lsl #3]
    add     x16, x21, x17
    cmp     x15, x25
    csel    x21, x16, x21, le
    ldr     x15, [x9, x22, lsl #3]
    add     x16, x22, x17
    cmp     x15, x26
    csel    x22, x16, x22, le
    stp     x19, x20, [x13], #16
    stp     x21, x22, [x13], #16
    subs    x14, x14, #1
    b.ne    Lw_quad
Lw_rest:
    ands    x14, x11, #3
    b.eq    Lw_next
Lw_one:
    ldr     x19, [x13]
    ldr     x23, [x12], #8
    ldr     x15, [x9, x19, lsl #3]
    add     x16, x19, x17
    cmp     x15, x23
    csel    x19, x16, x19, le
    str     x19, [x13], #8
    subs    x14, x14, #1
    b.ne    Lw_one
Lw_next:
    lsr     x17, x17, #1
    cmp     x17, #4
    b.hi    Lw_level
    mov     x12, x1
    mov     x13, x2
    mov     x14, x11
Lw_count:
    ldr     x16, [x13]
    ld1r    {v0.2d}, [x12], #8
    add     x15, x0, x16, lsl #3
    ld1     {v24.2d, v25.2d, v26.2d, v27.2d}, [x15]
    cmge    v24.2d, v0.2d, v24.2d
    cmge    v25.2d, v0.2d, v25.2d
    cmge    v26.2d, v0.2d, v26.2d
    cmge    v27.2d, v0.2d, v27.2d
    add     v24.2d, v24.2d, v25.2d
    add     v26.2d, v26.2d, v27.2d
    add     v24.2d, v24.2d, v26.2d
    addp    d24, v24.2d
    fmov    x15, d24
    sub     x16, x16, x15
    str     x16, [x13], #8
    subs    x14, x14, #1
    b.ne    Lw_count
    add     x1, x1, x11, lsl #3
    add     x2, x2, x11, lsl #3
    subs    x4, x4, x11
    b.ne    Lw_block
    ldp     x19, x20, [sp, #0]
    ldp     x21, x22, [sp, #16]
    ldp     x23, x24, [sp, #32]
    ldp     x25, x26, [sp, #48]
    add     sp, sp, #64
    ret

// One needle at a time, the same search with the flags.
Lscalar:
    ldr     x9, [x1], #8
    cmp     x8, x9
    csel    x10, x7, xzr, le
    mov     x11, x6
    mov     x12, x0
    cbz     x11, Ls_final
    add     x12, x0, x11, lsl #3
    sub     x12, x12, #8
Ls_level:
    ldr     x13, [x12, x10, lsl #3]
    add     x14, x10, x11
    cmp     x13, x9
    csel    x10, x14, x10, le
    lsr     x11, x11, #1
    sub     x12, x12, x11, lsl #3
    cbnz    x11, Ls_level
Ls_final:
    ldr     x13, [x12, x10, lsl #3]
    cmp     x13, x9
    cinc    x10, x10, le
    str     x10, [x2], #8
    subs    x4, x4, #1
    b.ne    Lscalar
    ret
Lzero:
    str     xzr, [x2], #8
    subs    x4, x4, #1
    b.ne    Lzero
    ret
