// Writes value to x[i] wherever mask[i] is true. Other elements are left as they are.
void maskfill_f32(float *x, const bool *mask, size_t n, float value) {
    for (size_t i = 0; i < n; i++) {
        if (mask[i]) x[i] = value;
    }
}
