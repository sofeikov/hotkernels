// Sum of the squares of a float64 array, np.sum(x * x), with NumPy's result bit for bit.
//
// The squares are added in the order NumPy adds the elements of a contiguous
// array, its pairwise summation; kernels/sum_f64/naive.c is the same walk
// over the elements themselves, and its comment has the rules. Every square
// is a single rounded multiplication and every addition a single rounded
// addition: the pragma keeps the compiler from fusing the two into one
// multiply-add, and the file must never be built with -ffast-math.
#pragma STDC FP_CONTRACT OFF

// Largest number of elements NumPy adds as one block of eight running sums.
#define BLOCK 128
// Number of running sums in a block.
#define LANES 8

static double pairwise(const double *x, size_t n) {
    if (n < LANES) {
        double s = 0.0;
        for (size_t i = 0; i < n; i++) s += x[i] * x[i];
        return s;
    }
    if (n <= BLOCK) {
        double r[LANES];
        for (size_t k = 0; k < LANES; k++) r[k] = x[k] * x[k];
        size_t i = LANES;
        for (; i < n - n % LANES; i += LANES) {
            for (size_t k = 0; k < LANES; k++) r[k] += x[i + k] * x[i + k];
        }
        double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += x[i] * x[i];
        return s;
    }
    size_t half = n / 2;
    half -= half % LANES;
    return pairwise(x, half) + pairwise(x + half, n - half);
}

double fuse_sum_squares_f64(const double *x, size_t n) {
    return 0.0 + pairwise(x, n);
}
