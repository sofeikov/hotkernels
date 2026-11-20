// Shared declarations of the hotkernels binding: the code between NumPy and
// the kernels. See docs/delivery-design.md.
//
// The binding is one extension module built from several files. NumPy's C API
// is a table of pointers that each file reaches through a symbol, so every
// file must name the same symbol and only module.c may define and fill it.
#ifndef HOTKERNELS_BINDING_H
#define HOTKERNELS_BINDING_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL hotkernels_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL hotkernels_UFUNC_API
#ifndef HK_DEFINE_NUMPY_API
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "kernels.h"

// ---------------------------------------------------------------- counters

// One entry per kernel the binding can call. The order is the order of
// HK_KERNEL_NAMES in module.c.
//
// The entries after the kernels are loops built into the binding: for calls
// on short arrays, where NumPy's own call path is nearly all of a call's
// cost and the loop hardly matters; for accumulations along the first axis
// of a matrix, where the gain is the order in which memory is walked; and,
// on ARM64, for float64 adds and subtracts in place, where what decides the
// speed is which core's cache holds the output (hook_loop.c). They are plain
// C in the hook files, not
// kernels from `kernels/`, and the judge does not see them; `delivery check`
// compares them with NumPy as it does the rest of the binding. They are
// counted like kernels so that report() and the self-test of install() can
// show and prove them.
typedef enum {
    HK_ARGMAX_F32, HK_ARGMAX_F64, HK_ARGMIN_F32, HK_ARGMIN_F64,
    HK_CLIP_F32, HK_CLIP_F64,
    HK_COMPRESS_F32, HK_COMPRESS_F64, HK_MASKFILL_F64, HK_NONZERO_BOOL,
    HK_CUMSUM_I64, HK_CUMSUM_F32, HK_CUMSUM_F64, HK_CUMSUM_BOOL,
    HK_CAST_U8_F32, HK_SUM_U8, HK_COUNT_NONZERO_I64, HK_COUNT_NONZERO_F64,
    HK_HISTOGRAM_F64, HK_BINCOUNT_U8, HK_ISIN_I64,
    HK_SEARCHSORTED_F64, HK_SEARCHSORTED_RIGHT_F64, HK_INTERP_F64,
    HK_WHERE_F64, HK_CUMMAX_I64, HK_CUMFMAX_F64, HK_FMAX_REDUCE_F64, HK_ANY_ROWS_BOOL,
    HK_BINCOUNT_I16, HK_BINCOUNT_I64, HK_UNIQUE_I64,
    HK_ARGSORT_F64, HK_ARGSORT_I64, HK_ARGSORT_U64,
    HK_SEARCHSORTED_I64, HK_SEARCHSORTED_RIGHT_I64, HK_ANY_COLS_BOOL, HK_SUM_F32,
    // Served through bottleneck's functions, not NumPy's (hook_bottleneck.c).
    HK_NANMEAN_COLS_F64, HK_NANSTD_COLS_F64, HK_NANMIN_COLS_F64, HK_NANMAX_COLS_F64,
    // Built into the binding.
    HK_CLIP_I64, HK_ROUND_F64, HK_MEDIAN_F64,
    HK_CUMSUM_ROWS_F64, HK_CUMPROD_ROWS_F64, HK_CUMMAX_ROWS_F64, HK_CUMFMAX_ROWS_F64,
    HK_ARGMAX_ROWS_F32, HK_ARGMAX_ROWS_F64, HK_ARGMIN_ROWS_F32, HK_ARGMIN_ROWS_F64,
    HK_ADD_INPLACE_F64, HK_SUBTRACT_INPLACE_F64,
    HK_KERNEL_COUNT
} hk_kernel;

// One entry per NumPy operation that has a hook. A call that reaches a hook
// and does not meet a kernel's conditions is counted here. The order is the
// order of HK_OPERATION_NAMES in module.c.
typedef enum {
    HK_OP_ARGMAX, HK_OP_ARGMIN, HK_OP_CLIP, HK_OP_COMPRESS, HK_OP_MASKFILL,
    HK_OP_NONZERO, HK_OP_CUMSUM, HK_OP_ASTYPE, HK_OP_SUM, HK_OP_COUNT_NONZERO,
    HK_OP_HISTOGRAM, HK_OP_BINCOUNT, HK_OP_ISIN, HK_OP_SEARCHSORTED, HK_OP_INTERP,
    HK_OP_WHERE, HK_OP_ACCUMULATE, HK_OP_REDUCEAT, HK_OP_ANY, HK_OP_ROUND, HK_OP_MEDIAN,
    HK_OP_UNIQUE, HK_OP_ARGSORT,
    // bottleneck's functions: a call counted here was left to bottleneck.
    HK_OP_BN_NANMEAN, HK_OP_BN_NANSTD, HK_OP_BN_NANMIN, HK_OP_BN_NANMAX,
    HK_OPERATION_COUNT
} hk_operation;

// Calls a kernel took, and calls passed on to NumPy.
//
// NumPy runs inner loops with the interpreter lock released, so two threads
// can be inside a hook at once and the counters must be atomic objects. They
// are updated by a separate load and store, not by an atomic add: an atomic
// add costs several nanoseconds on every call that passes through a hook,
// which is more than the rest of a fallback. The price is that two threads
// counting the same thing at the same instant can lose one of the two
// counts. These are statistics for report(), and install()'s self-test,
// which does depend on them, runs on one thread.
extern _Atomic uint64_t hk_taken[HK_KERNEL_COUNT];
extern _Atomic uint64_t hk_passed[HK_OPERATION_COUNT];

// Add one to a counter. See above for why this is not an atomic add.
static inline void hk_count(_Atomic uint64_t *counter) {
    atomic_store_explicit(counter, atomic_load_explicit(counter, memory_order_relaxed) + 1, memory_order_relaxed);
}

// Record that kernel `k` handled a call.
static inline void hk_take(hk_kernel k) {
    hk_count(&hk_taken[k]);
}

// Record that a call to operation `op` reached a hook and went to NumPy.
static inline void hk_pass(hk_operation op) {
    hk_count(&hk_passed[op]);
}

// One Python call of argmax or argmin that NumPy may split into many calls of
// the type slot, one per row. The method hook sets `active` before it hands
// a float32 or float64 array on to NumPy; while it is set the slot hook
// records here what it did with each row instead of counting, and the method
// hook counts the whole call once when NumPy returns. One per thread: NumPy
// runs the slot on the calling thread with the interpreter lock released,
// while other threads may be in calls of their own. See
// docs/argmax-rows-design.md.
typedef struct {
    // True while a method hook waits for NumPy to return.
    bool active;
    // True when the slot took a kernel for at least one row.
    bool taken;
    // The kernel the slot took. Every row of one call has the same length,
    // so it is the same kernel for all of them.
    hk_kernel slot_kernel;
} hk_argext_call;

extern _Thread_local hk_argext_call hk_argext_mark;

// True when install(inexact=True) was given: kernels that are not
// bit-identical to NumPy may then take calls.
extern bool hk_inexact;

// -------------------------------------------------------------- the checks

// Keeps a function out of line. A hook's first lines run on every call NumPy
// receives, most of which no kernel takes. If the compiler folds the work of
// taking a kernel into the hook, every call pays for that work's stack frame
// before the first test: saving registers and reserving the scratch buffer
// cost an index operation more than the hook's own test does. The work is
// therefore kept in functions of its own, marked with this.
#define HK_NOINLINE __attribute__((noinline))

// The largest magnitude up to which a double represents every integer, 2^53.
#define HK_EXACT_INT_LIMIT 9007199254740992LL

// Element count from which a hook releases the interpreter lock around a
// kernel. Below it, taking the lock back costs more than other threads gain.
#define HK_NOGIL_MIN_N 8192

// Largest scratch buffer, in bytes, a hook keeps on the stack. Above it the
// buffer comes from malloc.
#define HK_STACK_SCRATCH_BYTES 2048

// The ndarray type. NumPy's headers reach it through a table of pointers; it
// is kept here so that the test made on every hooked call is one comparison.
extern PyTypeObject *hk_ndarray_type;

// True when `obj` is exactly an ndarray, not an instance of a subclass.
static inline bool hk_is_plain_array(PyObject *obj) {
    return Py_TYPE(obj) == hk_ndarray_type;
}

// True when `obj` is exactly an ndarray, not a subclass, holding elements of
// NumPy type `type_num` in native byte order as one C-contiguous run with at
// least one dimension. A kernel can then treat its data as a flat buffer of
// PyArray_SIZE elements. Alignment is not required: the kernels accept
// unaligned buffers.
static inline bool hk_is_flat(PyObject *obj, int type_num) {
    if (!hk_is_plain_array(obj)) return false;
    PyArrayObject *a = (PyArrayObject *)obj;
    return PyArray_TYPE(a) == type_num && PyArray_ISNOTSWAPPED(a) &&
           PyArray_NDIM(a) >= 1 && PyArray_IS_C_CONTIGUOUS(a);
}

// As hk_is_flat, and one-dimensional.
static inline bool hk_is_flat_1d(PyObject *obj, int type_num) {
    return hk_is_flat(obj, type_num) && PyArray_NDIM((PyArrayObject *)obj) == 1;
}

// True when `obj` is exactly an ndarray, not a subclass, of booleans with two
// dimensions: the arrays whose any(axis=1) and any(axis=0) have a kernel.
// Byte order and layout are left to the hook to test.
static inline bool hk_is_bool_matrix(PyObject *obj) {
    return hk_is_plain_array(obj) && PyArray_TYPE((PyArrayObject *)obj) == NPY_BOOL &&
           PyArray_NDIM((PyArrayObject *)obj) == 2;
}

// True when `axis` names the last axis of a two-dimensional array as a
// Python int: 1 or -1. None, a tuple, a bool and NumPy integers are NumPy's
// to interpret.
static inline bool hk_is_last_axis(PyObject *axis) {
    if (!PyLong_CheckExact(axis)) return false;
    int overflow = 0;
    const long value = PyLong_AsLongAndOverflow(axis, &overflow);
    return overflow == 0 && (value == 1 || value == -1);
}

// True when `axis` names the first axis of a two-dimensional array as a
// Python int: 0 or -2. Anything else, a bool among them, is NumPy's to
// interpret.
static inline bool hk_is_first_axis(PyObject *axis) {
    if (!PyLong_CheckExact(axis)) return false;
    int overflow = 0;
    const long value = PyLong_AsLongAndOverflow(axis, &overflow);
    return overflow == 0 && (value == 0 || value == -2);
}

// True when two arrays have the same number of dimensions and the same
// length along each.
static inline bool hk_same_shape(PyArrayObject *a, PyArrayObject *b) {
    const int dimensions = PyArray_NDIM(a);
    if (dimensions != PyArray_NDIM(b)) return false;
    const npy_intp *a_shape = PyArray_DIMS(a), *b_shape = PyArray_DIMS(b);
    for (int i = 0; i < dimensions; i++) {
        if (a_shape[i] != b_shape[i]) return false;
    }
    return true;
}

// True when the `a_bytes` bytes at `a` and the `b_bytes` bytes at `b` share
// no byte.
static inline bool hk_disjoint(const char *a, size_t a_bytes, const char *b, size_t b_bytes) {
    return a + a_bytes <= b || b + b_bytes <= a;
}

// True when every one of the `n` bytes at `mask` is 0 or 1. NumPy treats any
// non-zero byte of a boolean array as True; the mask kernels use the byte's
// value as a count and write outside their buffers when it is above 1, so a
// mask is read once here before a kernel sees it. Bytes of 0 and 1 OR
// together to at most 1.
static inline bool hk_mask_is_clean(const uint8_t *mask, size_t n) {
    uint8_t seen = 0;
    for (size_t i = 0; i < n; i++) seen |= mask[i];
    return seen <= 1;
}

// True when any of the `n` doubles at `values` is a NaN. A NaN is the only
// value that differs from itself, and OR-ing the comparisons leaves a loop the
// compiler turns into vector code: about 0.13 ns per element.
static inline bool hk_has_nan(const double *values, size_t n) {
    bool seen = false;
    for (size_t i = 0; i < n; i++) seen |= values[i] != values[i];
    return seen;
}

// Code that runs only for a kernel may use the instructions the kernels need.
// The x86-64 kernels are written for the x86-64-v3 level, and install()
// refuses a processor without it, so such a function may be compiled for
// AVX2, which the rest of the binding, compiled for any x86-64, is not. Such
// a function is not inlined into a caller compiled for any x86-64, so it is
// one that loops over an array. ARM64 has one level and needs no attribute.
#if defined(__x86_64__)
#define HK_TARGET_KERNEL_LEVEL __attribute__((target("avx2")))
#else
#define HK_TARGET_KERNEL_LEVEL
#endif

// True when any of the `n` doubles at `values` is a signalling NaN: a NaN
// whose quiet bit, the top bit of the fraction, is clear. NumPy answers a
// signalling NaN in np.fmax with a quiet NaN, where a kernel that ignores NaN
// skips it.
//
// A NaN is the one value that differs from itself. `nan` is all ones for a
// NaN and zero otherwise, and `clear` gathers the bits that are clear in some
// NaN: there is a signalling NaN exactly when the quiet bit is among them.
// The loop has no branch, and with AVX2 the compiler makes it a compare and
// two logical instructions per four elements: about 0.05 ns per element on
// the Intel Core i5-14600KF in the first-level cache. A first version found
// a NaN by comparing magnitudes as integers, which SSE2 cannot do in one
// instruction, at 0.17 ns per element, and 0.08 with AVX2. The compare raises
// the invalid-operation flag for a signalling NaN, never for a quiet one, and
// a call that holds a signalling NaN goes to NumPy. The bits are copied out
// with memcpy, which the compiler turns into a plain load.
//
// On ARM64 the same loop takes about 0.16 ns per element on the Apple M1 Pro,
// three times what the kernels it guards take to read the elements, so there
// a pre-test reads the array first (below) and the loop runs only on the
// blocks the pre-test cannot clear.
#if defined(__aarch64__)

// Doubles per block of the ARM64 pre-tests: 256 bytes. A block the pre-test
// cannot clear is read again by the test per element.
#define HK_GUARD_BLOCK 32

// The top 16 bits of each of the 32 doubles at `x` into `top`: the sign, the
// exponent, and the top four bits of the fraction, the quiet bit of a NaN
// among them. Element 8k + j of the block is lane j of top[k]. Two unzips
// gather them: of the high 32-bit words of four elements, then of the high
// halves of those. Three instructions per eight elements, where a test of the
// whole 64 bits takes at least two per two.
static inline void hk_top_halves(const double *x, uint16x8_t top[4]) {
    for (int k = 0; k < 4; k++) {
        const uint32x4_t a = vreinterpretq_u32_f64(vld1q_f64(x + 8 * k));
        const uint32x4_t b = vreinterpretq_u32_f64(vld1q_f64(x + 8 * k + 2));
        const uint32x4_t c = vreinterpretq_u32_f64(vld1q_f64(x + 8 * k + 4));
        const uint32x4_t d = vreinterpretq_u32_f64(vld1q_f64(x + 8 * k + 6));
        top[k] = vuzp2q_u16(vreinterpretq_u16_u32(vuzp2q_u32(a, b)), vreinterpretq_u16_u32(vuzp2q_u32(c, d)));
    }
}

// The test of every element, the loop of the x86-64 version below: true when
// one of the `n` doubles at `values` is a signalling NaN.
static inline bool hk_signalling_nan_scan(const double *values, size_t n) {
    // The quiet bit of a NaN.
    const uint64_t quiet = 0x0008000000000000ULL;
    uint64_t clear = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        const uint64_t nan = values[i] != values[i] ? ~0ULL : 0;
        clear |= ~bits & nan;
    }
    return (clear & quiet) != 0;
}

// False when none of the 32 doubles at `x` can be a signalling NaN. A
// signalling NaN, and an infinity, has a top halfword whose low 15 bits are
// 0x7ff0 to 0x7ff7: all ones in the exponent and the quiet bit clear. Shifted
// left by one, which drops the sign, that is 0xffe0 to 0xffee, and adding 0x20
// takes exactly those, wrapping, below 0x10. A quiet NaN gives 0x10 to 0x1e,
// and every number but an infinity more. Integer instructions only: no
// floating-point flag is raised here.
static inline bool hk_block_may_signal(const double *x) {
    // Added to the shifted top halfword: wraps the range of a signalling NaN
    // and an infinity to the bottom.
    const uint16x8_t bias = vdupq_n_u16(0x20);
    // The shifted top halfword of a quiet NaN, plus the bias, is at least this.
    const uint16_t quiet_floor = 0x10;
    uint16x8_t top[4];
    hk_top_halves(x, top);
    const uint16x8_t low01 = vminq_u16(vaddq_u16(vshlq_n_u16(top[0], 1), bias), vaddq_u16(vshlq_n_u16(top[1], 1), bias));
    const uint16x8_t low23 = vminq_u16(vaddq_u16(vshlq_n_u16(top[2], 1), bias), vaddq_u16(vshlq_n_u16(top[3], 1), bias));
    return vminvq_u16(vminq_u16(low01, low23)) < quiet_floor;
}

// ARM64: a block of 32 elements at a time, pre-tested on its top halfwords
// (hk_block_may_signal) and tested element by element only when that finds an
// infinity or a signalling NaN; it returns at the first block that holds a
// signalling NaN. The last block ends at n and overlaps the one before it;
// fewer than 32 elements are tested element by element. The answer is the
// x86-64 version's for every input, and the floating-point flags are too: the
// test per element raises the invalid-operation flag for a signalling NaN,
// and every signalling NaN is in a block it tests. About 0.062 ns per element
// in the first-level cache, where the test of every element took 0.156, and
// 0.097 from the second-level cache, where it took 0.156 (Apple M1 Pro).
static inline bool hk_has_signalling_nan(const double *values, size_t n) {
    if (n < HK_GUARD_BLOCK) return hk_signalling_nan_scan(values, n);
    for (size_t i = 0;; i += HK_GUARD_BLOCK) {
        if (i > n - HK_GUARD_BLOCK) i = n - HK_GUARD_BLOCK;
        if (hk_block_may_signal(values + i) && hk_signalling_nan_scan(values + i, HK_GUARD_BLOCK)) return true;
        if (i == n - HK_GUARD_BLOCK) return false;
    }
}

#else
static inline HK_TARGET_KERNEL_LEVEL bool hk_has_signalling_nan(const double *values, size_t n) {
    // The quiet bit of a NaN.
    const uint64_t quiet = 0x0008000000000000ULL;
    uint64_t clear = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t bits;
        memcpy(&bits, &values[i], sizeof bits);
        const uint64_t nan = values[i] != values[i] ? ~0ULL : 0;
        clear |= ~bits & nan;
    }
    return (clear & quiet) != 0;
}
#endif

// Clear the floating-point status flags a kernel may have set. NumPy's own
// loops leave none behind, and a kernel's vmaxpd sets the invalid-operation
// flag for every NaN operand, quiet ones included. On x86-64 the kernels use
// only SSE and AVX instructions, whose flags are the low six bits of MXCSR,
// and the register is written only when one is set: about 1 ns, where the C
// library's feclearexcept, which also saves and reloads the x87 state, takes
// about 46 ns (Intel Core i5-14600KF), a fifth of a call on 16 elements.
static inline void hk_clear_float_flags(void) {
#if defined(__x86_64__)
    // The six exception flags of MXCSR: invalid, denormal, divide by zero,
    // overflow, underflow and precision.
    const unsigned int flags = 0x3f;
    const unsigned int csr = _mm_getcsr();
    if (csr & flags) _mm_setcsr(csr & ~flags);
#else
    feclearexcept(FE_ALL_EXCEPT);
#endif
}

// Clear the four floating-point status flags NumPy reports (invalid
// operation, divide by zero, overflow, underflow) and leave the others.
// After np.sum, NumPy leaves these four clear and the inexact flag as its
// additions left it, set or not; a kernel that makes the same additions sets
// the same inexact flag. The status register is written only when one of the
// four is set.
static inline void hk_clear_reported_float_flags(void) {
#if defined(__x86_64__)
    // MXCSR's flags for invalid (0x01), divide by zero (0x04), overflow
    // (0x08) and underflow (0x10). The denormal flag (0x02) is left, as
    // NumPy's feclearexcept leaves it: it is not one of fenv.h's flags.
    const unsigned int reported = 0x1d;
    const unsigned int csr = _mm_getcsr();
    if (csr & reported) _mm_setcsr(csr & ~reported);
#elif defined(__aarch64__)
    // IOC, DZC, OFC and UFC: bits 0 to 3 of FPSR.
    const uint64_t reported = 0xf;
    uint64_t fpsr;
    __asm__ volatile("mrs %0, fpsr" : "=r"(fpsr) : : "memory");
    if (fpsr & reported) __asm__ volatile("msr fpsr, %0" : : "r"(fpsr & ~reported) : "memory");
#else
    feclearexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
#endif
}

// True when a floating-point exception that NumPy reports, as a warning or an
// error by np.errstate, has been raised since the flags were last cleared:
// invalid operation, divide by zero, overflow or underflow. NumPy ignores the
// precision flag and, on x86-64, the denormal flag. On x86-64 only MXCSR is
// read, which holds the flags of every SSE and AVX instruction, the only kind
// the binding's own float code uses: one instruction, where the C library's
// fetestexcept also reads the x87 state.
static inline bool hk_float_flags_reported(void) {
#if defined(__x86_64__)
    // MXCSR's flags for invalid (0x01), divide by zero (0x04), overflow
    // (0x08) and underflow (0x10).
    const unsigned int reported = 0x1d;
    return (_mm_getcsr() & reported) != 0;
#else
    return fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW) != 0;
#endif
}

// Convert a Python float or a Python int to a double, as NumPy does when such
// a value is stored into a float64 array: an int is rounded to the nearest
// double. Returns false, with no exception set, for anything else, including
// bool, NumPy scalars, and an int too large for a double.
static inline bool hk_as_double(PyObject *obj, double *out) {
    if (PyFloat_CheckExact(obj)) {
        *out = PyFloat_AS_DOUBLE(obj);
        return true;
    }
    if (PyLong_CheckExact(obj)) {
        const double v = PyLong_AsDouble(obj);
        if (v == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        *out = v;
        return true;
    }
    return false;
}

// As hk_as_double, but an int is accepted only when a double holds it
// exactly, so that arithmetic on the double and on the int cannot differ.
static inline bool hk_as_exact_double(PyObject *obj, double *out) {
    if (PyLong_CheckExact(obj)) {
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
        if (overflow != 0 || v < -HK_EXACT_INT_LIMIT || v > HK_EXACT_INT_LIMIT) return false;
        *out = (double)v;
        return true;
    }
    return hk_as_double(obj, out);
}

// Convert an np.int64 or a Python int that fits int64 to an int64. Returns
// false, with no exception set, for anything else, including bool, the other
// NumPy integers and a Python int outside int64: NumPy converts those by its
// own rules.
static inline bool hk_as_int64(PyObject *obj, int64_t *out) {
    if (Py_IS_TYPE(obj, &PyInt64ArrType_Type)) {
        *out = PyArrayScalar_VAL(obj, Int64);
        return true;
    }
    if (!PyLong_CheckExact(obj)) return false;
    int overflow = 0;
    const long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
    if (overflow != 0 || (v == -1 && PyErr_Occurred())) {
        PyErr_Clear();
        return false;
    }
    *out = (int64_t)v;
    return true;
}

// Release the interpreter lock for the block that follows when `n` elements
// are enough to be worth it. HK_END_NOGIL takes it back.
#define HK_BEGIN_NOGIL(n) \
    PyThreadState *hk_saved_thread = (n) >= HK_NOGIL_MIN_N ? PyEval_SaveThread() : NULL;
#define HK_END_NOGIL \
    if (hk_saved_thread) PyEval_RestoreThread(hk_saved_thread);

// ----------------------------------------------- operations shared by hooks

// Each returns a new reference when a kernel took the call, and NULL with no
// exception set when the call does not meet the kernel's conditions and
// belongs to NumPy. None of them raises.
PyObject *hk_try_sum(PyObject *array);
PyObject *hk_try_argext(PyObject *array, bool want_max);

// argmax or argmin along the last axis of a matrix, as the new
// one-dimensional np.intp array NumPy's `m.argmax(axis=1)` returns: the
// operation's kernel once per row for rows of at least ARGEXT_ROWS_KERNEL_COLS
// elements, and a loop of the binding's own (argmax_rows_f32 and the rest)
// for narrower ones. The caller has tested that the axis is 1 or -1 and that
// there is no other argument but an `out` of None; the array's type, shape and
// layout are tested here. In hook_function.c.
PyObject *hk_try_argext_rows(PyObject *array, bool want_max);

// The array NumPy's clip returns for `array` clamped to the bounds `lo` and
// `hi`, as given by the caller: the clip kernels, or for int64 the binding's
// own loop, on a new array. `array` is contiguous float32 or float64 with at
// least one element, or contiguous int64 with 1 to CLIP_I64_MAX_N elements;
// each bound is an object that leaves the result in the array's element type
// and converts to it exactly as NumPy converts it, with lo <= hi after the
// conversion. `out` is the caller's `out` argument, NULL when it was not
// given: absent or None gives a new array, the array itself (float32 and
// float64, writeable) clips it in place and returns it, and anything else is
// NumPy's. A caller whose NumPy operation needs more tests that first. In
// hook_function.c.
PyObject *hk_try_clip(PyObject *array, PyObject *lo, PyObject *hi, PyObject *out);

// Clip the `n` elements at `data` to [lo, hi] in place with the clip kernel
// `kernel` (HK_CLIP_F32 or HK_CLIP_F64; for float32 the bounds are float32
// values held as doubles). The kernels are wrong when their output is their
// input (issue 7), so each run of the array is copied to a buffer on the
// stack and the kernel clips the copy back over the run: every kernel call
// sees two buffers that do not overlap. Leaves the floating-point flags as the kernels leave them. Needs
// no interpreter lock. In hook_function.c.
void hk_clip_in_place(hk_kernel kernel, void *data, size_t n, double lo, double hi);

// The elements of `x` where `mask` is True, as a new one-dimensional array:
// the compress kernels. `x` is contiguous float32, float64 or int64 (which
// the float64 kernel selects, as bit patterns) and `mask` a
// contiguous boolean array of the same size, each of any shape; both are
// read in C order. A caller whose NumPy operation needs more, such as both
// being one-dimensional, tests that first. In hook_index.c.
PyObject *hk_try_compress(PyObject *x, PyObject *mask);

// The positions where a one-dimensional contiguous boolean array is True, as
// the one-element tuple NumPy's nonzero returns: the nonzero_bool kernel. In
// hook_method.c.
PyObject *hk_try_nonzero(PyObject *mask);

// Whether each row of a boolean matrix holds a True, as the new
// one-dimensional boolean array NumPy's `m.any(axis=1)` returns: the
// any_rows_bool kernel. The caller has tested hk_is_bool_matrix and the
// arguments; the layout and the mask's bytes are tested here. In
// hook_method.c.
PyObject *hk_try_any_rows(PyObject *mask);

// Whether each column of a boolean matrix holds a True, as the new
// one-dimensional boolean array NumPy's `m.any(axis=0)` returns: the
// any_cols_bool kernel. The caller has tested hk_is_bool_matrix and the
// arguments; the layout and the shape are tested here. In hook_method.c.
PyObject *hk_try_any_cols(PyObject *mask);

// np.round(x, decimals) for a contiguous float64 `array` of 1 to
// ROUND_F64_MAX_N elements, as the new array NumPy returns: the binding's own
// loop, round_f64. `decimals` is the caller's argument, or NULL when it was
// not given; it must be a Python int from 0 to 15. The caller has tested the
// array's type, and that there is no `out`. In hook_method.c.
PyObject *hk_try_round(PyObject *array, PyObject *decimals);

// Set the elements of `x` where `mask` is True to `value`, in place: the
// maskfill kernel. `x` is contiguous, writable float64 and `mask` a
// contiguous boolean array of the same size that does not share memory with
// it, each of any shape; `value` is a Python float, or a Python int, which
// is rounded to a double as NumPy rounds it (see hk_as_double). Returns true when the kernel did it, and false, having
// changed nothing and set no exception, when the call belongs to NumPy. A
// caller whose NumPy operation needs more, such as matching shapes, tests
// that first. In hook_index.c.
bool hk_try_maskfill(PyObject *x, PyObject *mask, PyObject *value);

// Build a NumPy scalar of type `type_num` from the value at `value`. Returns
// NULL with no exception set if it cannot be built.
PyObject *hk_scalar(int type_num, void *value);

// Match the arguments of a call to parameter names.
//
// `names` lists the function's parameters in order; a NULL entry is a
// parameter that can only be given by position. At most `max_positional`
// arguments may be positional. On success `bound[i]` is the argument given
// for parameter i, as a borrowed reference, or NULL when it was not given.
// Returns false for a call NumPy would reject or that names a parameter not
// listed: too many positional arguments, an unknown keyword, or a parameter
// given twice. No exception is set.
bool hk_bind_arguments(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                       const char *const *names, Py_ssize_t count,
                       Py_ssize_t max_positional, PyObject **bound);

// True when the Python string `text` is exactly the ASCII string `ascii`.
// Keyword names and short option strings are compact ASCII objects, whose
// characters can be compared in place: this runs on every call that reaches a
// hook with keywords, where a library call per comparison is most of what a
// call left to NumPy costs. The bytes are compared in a loop of its own for
// the same reason: a strlen and a memcmp per parameter name cost about 20 ns
// on a call with one keyword, measured on `x.clip(lo, None, out=None)`.
static inline bool hk_text_is(PyObject *text, const char *ascii) {
    if (!PyUnicode_IS_COMPACT_ASCII(text)) return false;
    const char *data = (const char *)PyUnicode_DATA(text);
    const Py_ssize_t length = PyUnicode_GET_LENGTH(text);
    for (Py_ssize_t i = 0; i < length; i++) {
        // `ascii` ends at its NUL, which no character of `text` may match.
        if (ascii[i] == '\0' || ascii[i] != data[i]) return false;
    }
    return ascii[length] == '\0';
}

// True when an argument was not given or is None.
static inline bool hk_is_absent(PyObject *value) {
    return value == NULL || value == Py_None;
}

// ------------------------------------------------------------------- hooks

// One hook: one pointer inside NumPy that can be pointed at the binding.
typedef struct {
    // Name used from Python, for example "slot.argmax.f32".
    const char *name;
    // Point NumPy at the binding. Takes `index`. Returns 0, or -1 with an
    // exception set, in which case nothing was changed.
    int (*apply)(int index);
    // Put the original pointer back. Takes `index`. Returns 1 when it did, and
    // 0 when the pointer no longer held the binding's function and was left
    // alone.
    int (*remove)(int index);
    // Which entry of its kind's own table this hook is.
    int index;
    // Whether the hook is in place now.
    bool applied;
} hk_hook;

// The hooks of each kind, each array ended by an entry with a NULL name.
extern hk_hook hk_slot_hooks[];
extern hk_hook hk_loop_hooks[];
extern hk_hook hk_function_hooks[];
extern hk_hook hk_method_hooks[];
extern hk_hook hk_index_hooks[];
extern hk_hook hk_bottleneck_hooks[];

#endif
