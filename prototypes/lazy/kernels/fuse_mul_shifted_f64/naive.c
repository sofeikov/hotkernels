// out[i] = a[i] * (s + b[i]), the sum rounded before the product.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_mul_shifted_f64(const double *a, const double *b, double *out, size_t n, double s) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * (s + b[i]);
    }
}
