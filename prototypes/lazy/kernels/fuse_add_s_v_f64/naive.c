// out[i] = s0 + a[i], every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_add_s_v_f64(
    const double *a,
    double *out,
    size_t n,
    double s0
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = s0 + a[i];
    }
}
