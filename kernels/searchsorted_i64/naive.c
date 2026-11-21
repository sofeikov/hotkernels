// For each of the `needles` values in v, writes the first index i such that
// a[i] is not less than the needle (side='left'). a holds n values sorted
// ascending.
void searchsorted_i64(const int64_t *a, const int64_t *v, int64_t *out, size_t n, size_t needles) {
    for (size_t j = 0; j < needles; j++) {
        int64_t key = v[j];
        size_t lo = 0, hi = n;
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            if (a[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        out[j] = (int64_t)lo;
    }
}
