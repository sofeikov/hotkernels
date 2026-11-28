// out[i] = a[i] <= s ? b[i] : c[i], with a comparison that raises no flag.
//
// A NaN in a[i] is not below anything, so c[i] is taken, as NumPy's
// comparison and np.where give. `__builtin_islessequal` is C's
// `islessequal`: the quiet comparison, where `<=` raises the
// invalid-operation flag for a NaN. NumPy's comparison reports nothing for
// a NaN, so neither does this. The pragma is every fused kernel's: nothing
// here could be fused into a multiply-add, and it stays so if that changes.
#pragma STDC FP_CONTRACT OFF

void fuse_where_le_f64(const double *a, const double *b, const double *c, double *out, size_t n, double s) {
    for (size_t i = 0; i < n; i++) {
        out[i] = __builtin_islessequal(a[i], s) ? b[i] : c[i];
    }
}
