// Sum of a float32 array, np.sum(x), with NumPy's result bit for bit.
//
// NumPy adds the elements of a contiguous array in a fixed order, its pairwise
// summation (pairwise_sum in numpy/_core/src/umath/loops_utils.h.src), and this
// makes the same additions in the same order:
//
// - fewer than 8 elements are added one after another into a sum that starts
//   at +0.0;
// - up to BLOCK elements are added into eight running sums, element i into
//   sum i mod 8, starting from the first eight elements; the sums are combined
//   as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), and the last n mod 8
//   elements are added to that one after another;
// - more than BLOCK elements are split at n / 2 rounded down to a multiple of
//   8, and the sums of the two halves are added.
//
// The reduction itself starts from +0.0, so the result is 0 + pairwise(x, n):
// a sum of negative zeros is +0.0. Every addition is a single rounded float
// addition; the compiler may not reassociate them, so this must never be
// built with -ffast-math or -fassociative-math.

// Largest number of elements NumPy adds as one block of eight running sums.
#define BLOCK 128
// Number of running sums in a block.
#define LANES 8

static float pairwise(const float *x, size_t n) {
    if (n < LANES) {
        float s = 0.0f;
        for (size_t i = 0; i < n; i++) s += x[i];
        return s;
    }
    if (n <= BLOCK) {
        float r[LANES];
        for (size_t k = 0; k < LANES; k++) r[k] = x[k];
        size_t i = LANES;
        for (; i < n - n % LANES; i += LANES) {
            for (size_t k = 0; k < LANES; k++) r[k] += x[i + k];
        }
        float s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) s += x[i];
        return s;
    }
    size_t half = n / 2;
    half -= half % LANES;
    return pairwise(x, half) + pairwise(x + half, n - half);
}

float sum_f32(const float *x, size_t n) {
    return 0.0f + pairwise(x, n);
}
