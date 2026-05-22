#include "squaring.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>

/* Helper: max of n doubles via vDSP. */
static inline double vec_max(const double *x, size_t n) {
    double m = 0.0;
    vDSP_maxvD(x, 1, &m, n);
    return m;
}

/* aligned_alloc (C11) requires size to be a multiple of alignment. Round up so
 * tiny matrices (e.g. test_basic's ndi=6) don't return NULL. */
static inline void *xaligned_alloc(size_t alignment, size_t size) {
    size_t rounded = (size + alignment - 1) & ~(alignment - 1);
    return aligned_alloc(alignment, rounded);
}

int ppv_c_squaring(
    const double *Q,
    const double *R,
    int ndi,
    int num_delegates,
    int num_policies,
    double tol,
    int max_iter,
    double *consensus_out,
    double *influence_out
) {
    const size_t n  = (size_t)ndi;
    const size_t n2 = n * n;
    const size_t bytes = n2 * sizeof(double);

    /* 4 ndi*ndi buffers: t (accumulator), p (current power), and two distinct
     * GEMM-output scratches because the two dgemms below run concurrently on
     * separate threads — they need independent output buffers. */
    double *t       = xaligned_alloc(64, bytes);
    double *p       = xaligned_alloc(64, bytes);
    double *scratch = xaligned_alloc(64, bytes);
    double *p_new   = xaligned_alloc(64, bytes);
    if (!t || !p || !scratch || !p_new) {
        free(t); free(p); free(scratch); free(p_new);
        return -1;
    }

    /* t = I */
    memset(t, 0, bytes);
    for (size_t i = 0; i < n; i++) {
        t[i * n + i] = 1.0;
    }
    /* p = Q (copy in; we mutate p inside the loop) */
    memcpy(p, Q, bytes);

    /* The two dgemms in each squaring step (scratch = t@p, p_new = p@p) share
     * no data hazards. Dispatch them on a concurrent GCD queue so two AMX
     * units (one per P-cluster on M1/M2/M3 Pro/Max/Ultra) can in principle
     * run them in parallel. In practice on M1 Max the wall-time gain is
     * modest (~4% at n=5000) — Accelerate's single-call dgemm appears to
     * already engage AMX heavily, and memory bandwidth is shared. */
    dispatch_queue_t gemm_queue = dispatch_get_global_queue(
        QOS_CLASS_USER_INITIATED, 0
    );

    int iters = 0;
    for (iters = 0; iters < max_iter; iters++) {
        if (vec_max(p, n2) < tol) break;

        dispatch_group_t group = dispatch_group_create();
        dispatch_group_async(group, gemm_queue, ^{
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ndi, ndi, ndi,
                1.0, t, ndi, p, ndi,
                0.0, scratch, ndi
            );
        });
        dispatch_group_async(group, gemm_queue, ^{
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ndi, ndi, ndi,
                1.0, p, ndi, p, ndi,
                0.0, p_new, ndi
            );
        });
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        dispatch_release(group);

        cblas_daxpy((int)n2, 1.0, scratch, 1, t, 1);
        double *tmp = p; p = p_new; p_new = tmp;
    }

    /* Outputs derived from t (= T_∞ ≈ (I - Q)^-1 on the transient block).
     *
     * consensus = R · t · e_d
     *   where e_d = [1]*num_delegates + [0]*num_intermediates.
     * Compute as two matvecs to avoid a temporary num_policies×ndi product.
     *
     * influence_i = row_sum(t)_i / diag(t)_i
     *   row_sum = t @ 1_vec  (one matvec)
     *   diag    = t[i*n + i] (strided pick) */
    double *vec_a = xaligned_alloc(64, n * sizeof(double));
    double *vec_b = xaligned_alloc(64, n * sizeof(double));
    if (!vec_a || !vec_b) {
        free(t); free(p); free(scratch); free(p_new);
        free(vec_a); free(vec_b);
        return -1;
    }

    /* vec_a = e_d */
    for (size_t i = 0; i < (size_t)num_delegates; i++) vec_a[i] = 1.0;
    for (size_t i = num_delegates; i < n; i++) vec_a[i] = 0.0;

    /* vec_b = t · e_d  (length ndi) */
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans,
        ndi, ndi,
        1.0, t, ndi, vec_a, 1,
        0.0, vec_b, 1
    );
    /* consensus_out = R · vec_b  (length num_policies) */
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans,
        num_policies, ndi,
        1.0, R, ndi, vec_b, 1,
        0.0, consensus_out, 1
    );

    /* vec_a = 1; vec_b = t · 1  (row sums of t) */
    for (size_t i = 0; i < n; i++) vec_a[i] = 1.0;
    cblas_dgemv(
        CblasRowMajor, CblasNoTrans,
        ndi, ndi,
        1.0, t, ndi, vec_a, 1,
        0.0, vec_b, 1
    );
    /* influence = row_sum / diag */
    for (size_t i = 0; i < n; i++) {
        influence_out[i] = vec_b[i] / t[i * n + i];
    }

    free(t);
    free(p);
    free(scratch);
    free(p_new);
    free(vec_a);
    free(vec_b);

    return iters;
}
