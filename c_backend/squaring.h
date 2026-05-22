#ifndef PPV_SQUARING_H
#define PPV_SQUARING_H

#ifdef __cplusplus
extern "C" {
#endif

/* Joint squaring recurrence on (I - Q)^-1 via dense GEMM on Accelerate (AMX).
 *
 * Inputs:
 *   Q              row-major ndi x ndi, column-stochastic transient block
 *                  (entries are 0..1, no self-loops on the diagonal).
 *   R              row-major num_policies x ndi, policy-from-transient block.
 *   ndi            number of transient states (delegates + intermediates).
 *   num_delegates  number of delegate columns (the first num_delegates of Q).
 *   num_policies   row count of R.
 *   tol            convergence threshold; loop exits when max(Q^(2^m)) < tol.
 *   max_iter       hard cap on squarings (each iter doubles the term count).
 *
 * Outputs (caller-allocated):
 *   consensus_out  length num_policies, R · (I-Q)^-1 · e_delegates.
 *   influence_out  length ndi, row_sum(N) / diag(N) where N = (I-Q)^-1.
 *
 * Returns: number of squarings performed (>=0), or -1 on allocation failure.
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* PPV_SQUARING_H */
