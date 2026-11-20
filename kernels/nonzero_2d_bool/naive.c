// Writes the row and the column of every true element of m, a matrix of n
// elements in rows of width, to out as pairs of int64, in C order. Returns how
// many int64 it wrote: two per true element.
//
// Branchless, like nonzero_bool: every element's pair is stored, and the
// output position advances only when the element is true. `out` is a scratch
// buffer of 2 * n elements.
size_t nonzero_2d_bool(const bool *m, int64_t *out, size_t n, size_t width) {
    size_t k = 0;
    int64_t row = 0, col = 0;
    for (size_t i = 0; i < n; i++) {
        out[k] = row;
        out[k + 1] = col;
        k += 2 * (size_t)m[i];
        col++;
        if (col == (int64_t)width) {
            col = 0;
            row++;
        }
    }
    return k;
}
