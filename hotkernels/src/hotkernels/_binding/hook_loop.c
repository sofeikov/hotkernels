// Inner-loop hooks: a ufunc's loop for one element type, replaced through
// PyUFunc_ReplaceLoopBySignature, the interface NumPy provides for this.
//
// `np.clip(x, lo, hi)` and `x.clip(lo, hi)` both run the `clip` ufunc, whose
// loop receives four operands: x, lo, hi and the output. `np.hypot(a, b)`
// runs the `hypot` ufunc, whose loop receives three: a, b and the output.
// NumPy has settled the element type before the loop is called and describes
// the memory layout through `steps`, the distance in bytes between
// consecutive elements of each operand. A scalar operand has a step of zero.
#include "binding.h"

// Operands of the clip loop, in the order NumPy passes them.
enum { CLIP_X, CLIP_LO, CLIP_HI, CLIP_OUT, CLIP_OPERANDS };

// Operands of the hypot loop, in the order NumPy passes them.
enum { HYPOT_A, HYPOT_B, HYPOT_OUT, HYPOT_OPERANDS };

// The most operands any hooked loop has.
#define LOOP_MAX_OPERANDS CLIP_OPERANDS

// One hooked loop.
typedef struct {
    // Name of the ufunc in numpy._core.umath.
    const char *ufunc_name;
    // Number of operands of its loop, inputs and output; all have the same
    // NumPy type.
    int operands;
    // NumPy type number of every operand.
    int type_num;
    // The binding's loop.
    PyUFuncGenericFunction wrapper;
    // NumPy's own loop, kept while the hook is applied.
    PyUFuncGenericFunction original;
    // The ufunc, fetched on first use and kept.
    PyObject *ufunc;
} hk_loop;

// Indices of the hooked loops in `loops`.
enum { LOOP_CLIP_F32, LOOP_CLIP_F64, LOOP_HYPOT_F64, LOOP_COUNT };

static hk_loop loops[LOOP_COUNT];

// The kernel is taken when x and the output are contiguous, both bounds are
// scalars with lo <= hi (which a NaN bound fails), and the output is not the
// input. Everything else is NumPy's: strided operands, array bounds (NumPy
// uses different arithmetic for them, issue 9), reversed or NaN bounds, and
// in-place calls (the kernels are wrong in place on some data, issue 7).
//
// The kernels leave the invalid-operation flag set on NaN data. NumPy turns
// status flags into warnings after a loop returns, and its own clip loop ends
// by clearing them, so the hook does the same.
#define HK_CLIP_LOOP(function, index, kernel, kernel_id, c_type)                              \
    static void function(char **args, npy_intp const *dimensions, npy_intp const *steps,      \
                         void *data) {                                                        \
        const npy_intp n = dimensions[0];                                                     \
        const npy_intp size = (npy_intp)sizeof(c_type);                                       \
        if (n > 0 && steps[CLIP_X] == size && steps[CLIP_OUT] == size &&                      \
            steps[CLIP_LO] == 0 && steps[CLIP_HI] == 0) {                                     \
            c_type lo, hi;                                                                    \
            memcpy(&lo, args[CLIP_LO], sizeof lo);                                            \
            memcpy(&hi, args[CLIP_HI], sizeof hi);                                            \
            const size_t bytes = (size_t)n * sizeof(c_type);                                  \
            if (lo <= hi && hk_disjoint(args[CLIP_X], bytes, args[CLIP_OUT], bytes)) {        \
                kernel((const c_type *)args[CLIP_X], (c_type *)args[CLIP_OUT], (size_t)n,     \
                       lo, hi);                                                               \
                feclearexcept(FE_ALL_EXCEPT);                                                 \
                hk_take(kernel_id);                                                           \
                return;                                                                       \
            }                                                                                 \
        }                                                                                     \
        hk_pass(HK_OP_CLIP);                                                                  \
        loops[index].original(args, dimensions, steps, data);                                 \
    }

HK_CLIP_LOOP(loop_clip_f32, LOOP_CLIP_F32, clip_f32, HK_CLIP_F32, float)
HK_CLIP_LOOP(loop_clip_f64, LOOP_CLIP_F64, clip_f64, HK_CLIP_F64, double)

// ------------------------------------------------------------------- hypot
//
// The kernel computes the C library's hypot bit for bit only where that
// library is the algorithm it was written to (glibc's e_hypot.c from about
// 2.35); install() compares the two on the machine before it keeps this hook
// (hooks.py). It is taken when all three operands are contiguous doubles and
// the output either shares no byte with an input or starts exactly where it
// does, as `np.hypot(a, b, out=a)` gives. Everything else is NumPy's: a
// scalar operand (a step of zero), strided operands, and a partial overlap,
// which is how `np.hypot.accumulate` calls the loop.
//
// Floating-point flags. NumPy clears the status flags once before a ufunc
// call's loops and turns those set after them into warnings, or errors under
// np.errstate. The kernel raises the flags the C library raises for the same
// elements, and on some data one more (underflow, kernel.s). So the flags
// that were set before the kernel are saved, the kernel runs with them
// clear, and when it raised overflow, underflow, invalid or divide by zero
// its result is dropped and NumPy's own loop computes the same elements
// again, raising NumPy's own flags. The saved flags are put back either way:
// a call whose loop runs several times, once per row of a two-dimensional
// operand for example, keeps what the earlier runs raised.

// Elements of a call whose output is one of its inputs that the kernel
// computes at a time, into a buffer on the stack. NumPy's loop must still
// find the input as it was when a block raises a flag, so a block is copied
// to the output only once it is known to be clean.
#define HYPOT_BLOCK (HK_STACK_SCRATCH_BYTES / sizeof(double))

#if defined(__x86_64__)
// The floating-point status as saved around a kernel: MXCSR, whose low six
// bits are the flags of every SSE and AVX instruction, the only kind the
// kernel uses.
typedef unsigned int hypot_status;
// MXCSR's six flags: invalid, denormal, divide by zero, overflow, underflow
// and precision.
#define MXCSR_FLAGS 0x3fu
#else
// The status as saved around a kernel, through the C library.
typedef fexcept_t hypot_status;
#endif

// Save the floating-point flags and clear them.
static inline void hypot_save_flags(hypot_status *saved) {
#if defined(__x86_64__)
    *saved = _mm_getcsr();
    // Writing MXCSR costs more than reading it; skip it when nothing is set.
    if (*saved & MXCSR_FLAGS) _mm_setcsr(*saved & ~MXCSR_FLAGS);
#else
    fegetexceptflag(saved, FE_ALL_EXCEPT);
    feclearexcept(FE_ALL_EXCEPT);
#endif
}

// Put back the flags saved by hypot_save_flags, dropping those set since.
static inline void hypot_restore_flags(const hypot_status *saved) {
#if defined(__x86_64__)
    if (_mm_getcsr() != *saved) _mm_setcsr(*saved);
#else
    fesetexceptflag(saved, FE_ALL_EXCEPT);
#endif
}

// Run the kernel over `n` elements and say whether it raised none of the
// flags NumPy reports. The flags are as they were before, whatever it raised.
static bool hypot_clean(const double *a, const double *b, double *out, size_t n) {
    hypot_status saved;
    hypot_save_flags(&saved);
    hypot_f64(a, b, out, n);
    const bool clean = !hk_float_flags_reported();
    hypot_restore_flags(&saved);
    return clean;
}

// Compute as much of a contiguous call as the kernel may. Returns how many
// leading elements of the output hold their final value: all `n` when the
// kernel did the call, 0 when it did not take it (overlap, or a flag on an
// output clear of the inputs), and a whole number of blocks when the output
// is an input and a later block raised a flag. The inputs past that count
// are as the caller passed them.
static HK_NOINLINE size_t hypot_take(char **args, size_t n) {
    const double *a = (const double *)args[HYPOT_A];
    const double *b = (const double *)args[HYPOT_B];
    double *out = (double *)args[HYPOT_OUT];
    const size_t bytes = n * sizeof(double);
    const bool on_a = (const double *)out == a, on_b = (const double *)out == b;
    if (!on_a && !hk_disjoint((const char *)a, bytes, (const char *)out, bytes)) return 0;
    if (!on_b && !hk_disjoint((const char *)b, bytes, (const char *)out, bytes)) return 0;
    if (!on_a && !on_b) {
        // NumPy's loop overwrites every element the kernel wrote, from inputs
        // the kernel did not touch.
        return hypot_clean(a, b, out, n) ? n : 0;
    }
    double block[HYPOT_BLOCK];
    size_t done = 0;
    while (done < n) {
        const size_t m = n - done < HYPOT_BLOCK ? n - done : HYPOT_BLOCK;
        if (!hypot_clean(a + done, b + done, block, m)) break;
        memcpy(out + done, block, m * sizeof(double));
        done += m;
    }
    return done;
}

// The hypot loop for float64: the kernel where it is taken, NumPy's loop for
// the rest of the call.
static void loop_hypot_f64(char **args, npy_intp const *dimensions, npy_intp const *steps, void *data) {
    const npy_intp n = dimensions[0];
    const npy_intp size = (npy_intp)sizeof(double);
    if (n > 0 && steps[HYPOT_A] == size && steps[HYPOT_B] == size && steps[HYPOT_OUT] == size) {
        const npy_intp done = (npy_intp)hypot_take(args, (size_t)n);
        if (done == n) {
            hk_take(HK_HYPOT_F64);
            return;
        }
        if (done > 0) {
            // NumPy's loop does the elements after the clean blocks.
            const npy_intp rest = n - done;
            char *moved[HYPOT_OPERANDS];
            for (int i = 0; i < HYPOT_OPERANDS; i++) moved[i] = args[i] + done * size;
            hk_pass(HK_OP_HYPOT);
            loops[LOOP_HYPOT_F64].original(moved, &rest, steps, data);
            return;
        }
    }
    hk_pass(HK_OP_HYPOT);
    loops[LOOP_HYPOT_F64].original(args, dimensions, steps, data);
}

// --------------------------------------------------------------- the table

static hk_loop loops[LOOP_COUNT] = {
    [LOOP_CLIP_F32] = {"clip", CLIP_OPERANDS, NPY_FLOAT, loop_clip_f32, NULL, NULL},
    [LOOP_CLIP_F64] = {"clip", CLIP_OPERANDS, NPY_DOUBLE, loop_clip_f64, NULL, NULL},
    [LOOP_HYPOT_F64] = {"hypot", HYPOT_OPERANDS, NPY_DOUBLE, loop_hypot_f64, NULL, NULL},
};

// Return the ufunc of loop `index` as a borrowed reference, or NULL with an
// exception set.
static PyObject *get_ufunc(int index) {
    if (loops[index].ufunc == NULL) {
        PyObject *umath = PyImport_ImportModule("numpy._core.umath");
        if (umath == NULL) return NULL;
        loops[index].ufunc = PyObject_GetAttrString(umath, loops[index].ufunc_name);
        Py_DECREF(umath);
    }
    return loops[index].ufunc;
}

// Swap the loop of entry `index` to `replacement`, storing the loop that was
// there in `previous`. Returns 0, or -1 with an exception set.
static int swap_loop(int index, PyUFuncGenericFunction replacement, PyUFuncGenericFunction *previous) {
    PyObject *ufunc = get_ufunc(index);
    if (ufunc == NULL) return -1;
    int signature[LOOP_MAX_OPERANDS];
    for (int i = 0; i < loops[index].operands; i++) signature[i] = loops[index].type_num;
    if (PyUFunc_ReplaceLoopBySignature((PyUFuncObject *)ufunc, replacement, signature, previous) != 0) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_RuntimeError, "the %s ufunc has no loop for this type", loops[index].ufunc_name);
        }
        return -1;
    }
    return 0;
}

// Point the loop at the binding. See hk_hook.apply.
static int loop_apply(int index) {
    return swap_loop(index, loops[index].wrapper, &loops[index].original);
}

// Put NumPy's loop back. See hk_hook.remove. If something else replaced the
// loop after the binding did, its loop is put back and left in place.
static int loop_remove(int index) {
    PyUFuncGenericFunction found = NULL, ignored = NULL;
    if (swap_loop(index, loops[index].original, &found) != 0) {
        PyErr_Clear();
        return 0;
    }
    if (found == loops[index].wrapper) return 1;
    if (swap_loop(index, found, &ignored) != 0) PyErr_Clear();
    return 0;
}

hk_hook hk_loop_hooks[] = {
    {"loop.clip.f32", loop_apply, loop_remove, LOOP_CLIP_F32, false},
    {"loop.clip.f64", loop_apply, loop_remove, LOOP_CLIP_F64, false},
    {"loop.hypot.f64", loop_apply, loop_remove, LOOP_HYPOT_F64, false},
    {NULL, NULL, NULL, 0, false},
};
