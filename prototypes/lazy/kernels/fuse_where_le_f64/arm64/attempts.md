# fuse_where_le_f64 attempts (arm64)

There is no `kernel.s` beside this log: on ARM64 the kernel is built from its `naive.c`. It was written on an Intel Core i5-14600KF for [issue 317](https://github.com/sofeikov/hotkernels/issues/317), with no ARM64 machine at hand, so nothing was tried and nothing was timed here.

What an attempt would start from: `make_kernel.py` writes no NEON loop for a comparison or for `np.where`, so the loop is one to write by hand, as `fuse_flux_limit_f64`'s was: `fcmge` of the threshold against the array and `bsl` between the two others, four vectors of two an iteration. Whether the compiled C raises the invalid-operation flag for a NaN on ARM64 was not looked at; if it does, the layer computes the expression with NumPy's calls for data that has one, as it does for the flux limiter, and the result is NumPy's either way.
