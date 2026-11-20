# nonzero_2d_bool attempts (arm64)

All judge commands in this run were passed `--unlocked`, because the spec is not on `main` yet. Other agents were timing kernels on the same machine (load averages of 5 to 10), so intervals are wider than usual and figures are indicative.

## 1. Scalar baseline

- **Idea:** the naive loop in assembly: store every element's (row, column) pair with one `stp`, advance the write pointer by 16 bytes times the mask byte, step the column and wrap it with `csel` / `cinc`.
- **check:** pass
- **compare:** not run (first version, nothing to compare against)
- **Kept:** yes
- **Notes:** `judge run`: 1.14x the naive C at every width and density, about 1 ns per element.
