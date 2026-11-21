// Copies x[i] to out wherever mask[i] is true. Returns how many were copied.
//
// Branchless: every element is stored at the current output position, and the
// position advances only when the element is selected. An unselected element is
// overwritten by the next store. The last store, if unselected, stays behind
// just past the returned count, which is why `out` is a scratch buffer. The
// store never passes the capacity, because k never runs ahead of i.
size_t compress_i16(const int16_t *x, const bool *mask, int16_t *out, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        out[k] = x[i];
        k += mask[i];
    }
    return k;
}
