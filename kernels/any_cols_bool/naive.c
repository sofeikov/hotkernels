// Writes to out[c] whether column c of the n / cols whole rows of cols elements at m holds a byte that is not zero, for each of the cols columns.
// The matrix is read as bytes, not as bool: a byte other than 0 and 1 counts as True, as it does for NumPy.
void any_cols_bool(const bool *m, bool *out, size_t n, size_t cols) {
    const unsigned char *bytes = (const unsigned char *)m;
    unsigned char *seen = (unsigned char *)out;
    size_t rows = n / cols;
    for (size_t c = 0; c < cols; c++) {
        seen[c] = 0;
    }
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            seen[c] |= bytes[r * cols + c];
        }
    }
    for (size_t c = 0; c < cols; c++) {
        seen[c] = seen[c] != 0;
    }
}
