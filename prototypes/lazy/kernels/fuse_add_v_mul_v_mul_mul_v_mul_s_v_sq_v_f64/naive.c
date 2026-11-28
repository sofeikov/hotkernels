// out[i] = a[i] + (b[i] * ((c[i] * (s0 * d[i])) * (e[i] * e[i]))), every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_add_v_mul_v_mul_mul_v_mul_s_v_sq_v_f64(
    const double *a,
    const double *b,
    const double *c,
    const double *d,
    const double *e,
    double *out,
    size_t n,
    double s0
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + (b[i] * ((c[i] * (s0 * d[i])) * (e[i] * e[i])));
    }
}
