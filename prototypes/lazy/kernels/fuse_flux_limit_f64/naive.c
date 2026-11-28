// out[i] = a[i] * (b[i] - c[i]) > lo ? fill : d[i], the difference rounded before the product.
//
// A product that is NaN is not above anything, so d[i] is taken, as NumPy's
// comparison and np.where give. The pragma is every fused kernel's: nothing
// here could be fused into a multiply-add, and it stays so if that changes.
#pragma STDC FP_CONTRACT OFF

void fuse_flux_limit_f64(const double *a, const double *b, const double *c, const double *d, double *out, size_t n,
                         double lo, double fill) {
    for (size_t i = 0; i < n; i++) {
        out[i] = a[i] * (b[i] - c[i]) > lo ? fill : d[i];
    }
}
