// out[i] = s0 * (a[i] * (__builtin_fabs((s1 * b[i])) * ((c[i] * d[i]) + (s2 - e[i])))), every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_mul_s_mul_v_mul_abs_mul_s_v_add_mul_v_v_sub_s_v_f64(
    const double *a,
    const double *b,
    const double *c,
    const double *d,
    const double *e,
    double *out,
    size_t n,
    double s0,
    double s1,
    double s2
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = s0 * (a[i] * (__builtin_fabs((s1 * b[i])) * ((c[i] * d[i]) + (s2 - e[i]))));
    }
}
