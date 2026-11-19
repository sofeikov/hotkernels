// The C library's hypot, element by element: what np.hypot on float64 calls.
//
// The package compiles this file without <math.h>, so the function is
// declared here. It is the C library's own, which makes this reference
// NumPy's computation on every platform, whatever algorithm that library uses.
double hypot(double, double);

void hypot_f64(const double *a, const double *b, double *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = hypot(a[i], b[i]);
    }
}
