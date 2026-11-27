// out[i] = ((a[i] + b[i]) * s0 + (c[i] + d[i]) * s1) / s2 - s3 * e[i], every operation rounded, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
// The division stays a division: a product with 1 / s2 rounds differently.
#pragma STDC FP_CONTRACT OFF

void fuse_poisson_step_f64(const double *a, const double *b, const double *c, const double *d, const double *e,
                           double *out, size_t n, double s0, double s1, double s2, double s3) {
    for (size_t i = 0; i < n; i++) {
        out[i] = ((a[i] + b[i]) * s0 + (c[i] + d[i]) * s1) / s2 - s3 * e[i];
    }
}
