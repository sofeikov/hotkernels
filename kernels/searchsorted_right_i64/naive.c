// For each of the `needles` values in v, writes the first index i such that
// the needle is less than a[i], or n if there is none (side='right'): the
// index after the last element equal to the needle. a holds n values sorted
// ascending.
void searchsorted_right_i64(const int64_t *a, const int64_t *v, int64_t *out, size_t n, size_t needles) {
    for (size_t j = 0; j < needles; j++) {
        int64_t key = v[j];
        size_t lo = 0, hi = n;
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (key < a[mid]) hi = mid;
            else lo = mid + 1;
        }
        out[j] = (int64_t)lo;
    }
}
