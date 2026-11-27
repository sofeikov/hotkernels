// out[i] = a[i] - (((s0 * b[i]) / s1) * (c[i] - d[i])), every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_sub_v_mul_div_mul_s_v_s_sub_v_v_f64(
    const double *a,
    const double *b,
    const double *c,
    const double *d,
    double *out,
    size_t n,
    double s0,
    double s1
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] - (((s0 * b[i]) / s1) * (c[i] - d[i]));
    }
}
