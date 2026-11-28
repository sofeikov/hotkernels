// out[i] = __builtin_sqrt(((a[i] * a[i]) + (b[i] * b[i]))), every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_sqrt_add_sq_v_sq_v_f64(
    const double *a,
    const double *b,
    double *out,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        out[i] = __builtin_sqrt(((a[i] * a[i]) + (b[i] * b[i])));
    }
}
