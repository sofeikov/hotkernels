// out[i] = a[i] + b[i] * c[i], the product rounded before the sum.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

void fuse_add_product_f64(const double *a, const double *b, const double *c, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] + b[i] * c[i];
    }
}
