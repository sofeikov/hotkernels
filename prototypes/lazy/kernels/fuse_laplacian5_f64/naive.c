// out[i] = s * a[i] - (((b[i] + c[i]) + d[i]) + e[i]), every operation rounded, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_laplacian5_f64(const double *a, const double *b, const double *c, const double *d,
                         const double *e, double *out, size_t n, double s) {
    for (size_t i = 0; i < n; i++) {
        out[i] = s * a[i] - (((b[i] + c[i]) + d[i]) + e[i]);
    }
}
