// out[i] = larger(s0, larger(a[i], smaller(s1, b[i]))), every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

#if defined(__x86_64__)
// `np.maximum` on x86-64: the processor's `maxpd` with a NaN of the first number carried through. A NaN if
// either is one, the first's if both are; of two zeros, the second, whichever its sign.
static inline double larger(double x, double y) {
    return x != x ? x : (x > y ? x : y);
}
#else
// `np.maximum`: the larger of two numbers; of two zeros the positive one; a NaN if either is one, the first's
// if both are. C's `fmax` gives the number where one operand is a NaN, which is another function.
static inline double larger(double x, double y) {
    if (x > y) return x;
    if (y > x) return y;
    if (x == y) return x != 0.0 ? x : x + y;
    return x != x ? x : y;
}
#endif

#if defined(__x86_64__)
// `np.minimum` on x86-64: the processor's `minpd` with a NaN of the first number carried through. A NaN if
// either is one, the first's if both are; of two zeros, the second, whichever its sign.
static inline double smaller(double x, double y) {
    return x != x ? x : (x < y ? x : y);
}
#else
// `np.minimum`: the smaller of two numbers; of two zeros the negative one; a NaN if either is one, the first's
// if both are.
static inline double smaller(double x, double y) {
    if (x < y) return x;
    if (y < x) return y;
    if (x == y) return x != 0.0 ? x : -((-x) + (-y));
    return x != x ? x : y;
}
#endif

void fuse_max_s_max_v_min_s_v_f64(
    const double *a,
    const double *b,
    double *out,
    size_t n,
    double s0,
    double s1
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = larger(s0, larger(a[i], smaller(s1, b[i])));
    }
}
