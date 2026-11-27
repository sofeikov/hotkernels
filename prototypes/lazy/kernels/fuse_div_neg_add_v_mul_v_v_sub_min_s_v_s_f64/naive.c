// out[i] = (-(a[i] + (b[i] * c[i]))) / (smaller(s0, d[i]) - s1), every operation rounded by itself, in that order.
//
// The pragma keeps the compiler from fusing a product and a sum into one
// multiply-add, which rounds once and gives other bits than NumPy's passes.
#pragma STDC FP_CONTRACT OFF

// `np.minimum`: the smaller of two numbers; of two zeros the negative one; a NaN if either is one, the first's
// if both are.
static inline double smaller(double x, double y) {
    if (x < y) return x;
    if (y < x) return y;
    if (x == y) return x != 0.0 ? x : -((-x) + (-y));
    return x != x ? x : y;
}

void fuse_div_neg_add_v_mul_v_v_sub_min_s_v_s_f64(
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
        out[i] = (-(a[i] + (b[i] * c[i]))) / (smaller(s0, d[i]) - s1);
    }
}
