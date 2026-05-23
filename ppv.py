import logging
import warnings
from typing import Literal, NamedTuple

import cupy as cp
import numpy as np
from cupy.cuda import cublas, runtime

log = logging.getLogger(__name__)

# TF32 on the default device's cuBLAS handle. Used as the fallback (high-
# precision) path inside the squaring loop once P.max() drops below
# _SWITCH_THRESH and fp16 inputs would start shedding meaningful bits.
cublas.setMathMode(
    cp.cuda.Device().cublas_handle,
    cublas.CUBLAS_TENSOR_OP_MATH,
)

# Below this P.max(), switch from fp16 GEMM to fp32 (TF32) GEMM. 0 means
# always run fp16 — that's what check_ordering.py validated against the
# fp64 reference: the only inversions are pairs whose reference values are
# within atol=rtol=1e-3 (tie swaps), identical to what the cupy branch
# already produces with TF32. Bump up if a future input class needs an
# fp32 tail for ordering parity.
_SWITCH_THRESH = 0.0
_ALPHA_FP32 = np.array(1.0, dtype=np.float32)
_BETA0_FP32 = np.array(0.0, dtype=np.float32)
_BETA1_FP32 = np.array(1.0, dtype=np.float32)


def _gemm_acc(
    a_h: cp.ndarray, b_h: cp.ndarray, c: cp.ndarray, *, accumulate: bool
) -> None:
    """C = A @ B + (accumulate ? C : 0), with fp16 inputs and fp32 accumulator.

    Square (n×n) only — that's all the squaring kernel needs. Inputs `a_h`,
    `b_h` are already fp16; `c` is a pre-allocated fp32 buffer that's either
    overwritten or accumulated into. Dispatches to cuBLAS HMMA tensor cores
    via gemmEx. cuBLAS is column-major; swapping operand order is the standard
    row-major adapter.
    """
    n = c.shape[0]
    beta = _BETA1_FP32 if accumulate else _BETA0_FP32
    handle = cp.cuda.Device().cublas_handle
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    cublas.gemmEx(
        handle,
        cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N,
        n, n, n,
        _ALPHA_FP32.ctypes.data,
        b_h.data.ptr, runtime.CUDA_R_16F, n,
        a_h.data.ptr, runtime.CUDA_R_16F, n,
        beta.ctypes.data,
        c.data.ptr, runtime.CUDA_R_32F, n,
        cublas.CUBLAS_COMPUTE_32F,
        cublas.CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )

Role = Literal["delegate", "intermediate"]


class Consensus(NamedTuple):
    label: str
    value: float


class Influence(NamedTuple):
    label: str
    role: Role
    value: float


def compute(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[list[Consensus], list[Influence]]:
    """High level Propagational Proxy Voting from a human-friendly sparse description.

    Each delegate and intermediate is a dict mapping target-name → weight. Weights
    in a single voter's dict must sum to 1.0 (they describe how that voter splits
    their unit of voting power). Policies are absorbing states; you only pass
    their names — the function inserts the identity block for you.

    Ordering inside the matrix is fixed: delegates first, then intermediates,
    then policies. Labels must be unique across all three groups.

    Args:
        delegates: Mapping of delegate name to their outgoing votes,
            e.g. ``{'Alice': {'Bob': 0.2, 'FAR2': 0.8}}``. Targets may be other
            delegates, intermediates, or policies. A voter cannot vote for
            themselves (no self-key).
        intermediates: Same shape as ``delegates``. Intermediates re-distribute
            mass they receive but are not themselves a final destination.
        policies: Ordered list of policy names (absorbing states). The order here
            controls the policy row/column order in the underlying matrix.
        tol: Convergence threshold on the max remaining transient mass. Stops
            iterating once ``A_k[:ndi, :].max() < tol``.
        max_iter: Hard cap on iterations. If exceeded without convergence, emits
            a ``UserWarning`` but still returns the partial result.

    Returns:
        ``(consensus, influences)`` where:
          - ``consensus``: list of :class:`Consensus`, sorted descending by value,
            one entry per policy.
          - ``influences``: list of :class:`Influence`, sorted descending by value,
            one entry per delegate + intermediate.

    Example:
        >>> delegates = {
        ...     'Alice': {'RedFruit': 0.3, 'apple': 0.7},
        ...     'Bob':   {'Alice': 0.2, 'banana': 0.8},
        ... }
        >>> intermediates = {'RedFruit': {'apple': 1.0}}
        >>> policies = ['apple', 'banana']
        >>> consensus, influences = compute(delegates, intermediates, policies)
        >>> consensus[0].label  # winning policy
        'apple'

    Raises:
        AssertionError: If labels are not unique, columns don't sum to 1, a voter
            votes for themselves, or policies aren't strictly absorbing.

    Debug:
        The assembled voting matrix is emitted at ``logging.DEBUG`` on the
        ``"ppv"`` logger. Enable with::

            import logging
            logging.basicConfig(level=logging.DEBUG)

    See Also:
        :func:`build_matrix` for the dict→matrix step on its own.
        :func:`compute_matrix` for the propagation kernel.
    """
    v, labels, num_delegates, num_intermediates = build_matrix(
        delegates, intermediates, policies
    )

    if log.isEnabledFor(logging.DEBUG):
        with np.printoptions(precision=3, suppress=True, linewidth=120):
            log.debug("voting matrix (labels=%s):\n%s", labels, v)

    return compute_matrix(
        v, labels, num_delegates, num_intermediates, tol=tol, max_iter=max_iter
    )


def build_matrix(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
) -> tuple[np.ndarray, list[str], int, int]:
    """Assemble the column-stochastic voting matrix from sparse inputs.

    This is the matrix-construction half of :func:`compute`, exposed so callers
    can inspect or modify ``v`` before handing it to :func:`compute_matrix`
    (debugging, perturbation analysis, caching, etc.).

    Args:
        delegates: See :func:`compute`.
        intermediates: See :func:`compute`.
        policies: See :func:`compute`.

    Returns:
        ``(v, labels, num_delegates, num_intermediates)`` — exactly the four
        positional arguments :func:`compute_matrix` expects. Labels are ordered
        ``list(delegates) + list(intermediates) + list(policies)``.

    Example:
        >>> v, labels, nd, ni = build_matrix(
        ...     delegates={'Alice': {'apple': 1.0}, 'Bob': {'banana': 1.0}},
        ...     intermediates={'RedFruit': {'apple': 1.0}},
        ...     policies=['apple', 'banana'],
        ... )
        >>> labels
        ['Alice', 'Bob', 'RedFruit', 'apple', 'banana']
        >>> nd, ni
        (2, 1)
    """
    labels = list(delegates) + list(intermediates) + list(policies)
    assert len(labels) == len(set(labels)), (
        "labels must be unique across delegates, intermediates, policies"
    )

    index_of = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    num_delegates = len(delegates)
    num_intermediates = len(intermediates)

    v = np.zeros((n, n), dtype=np.float32)
    for voter, edges in (delegates | intermediates).items():
        j = index_of[voter]
        for target, weight in edges.items():
            v[index_of[target], j] = weight
    for p in policies:
        i = index_of[p]
        v[i, i] = 1.0

    return v, labels, num_delegates, num_intermediates


def compute_matrix(
    v: np.ndarray,
    labels: list[str],
    num_delegates: int,
    num_intermediates: int,
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[list[Consensus], list[Influence]]:
    """Run Propagational Proxy Voting from a column-stochastic matrix.

    This is the low-level kernel. Use it when you already have the voting
    matrix as a numpy array. The matrix layout is positional and strict:

    - Column ``j`` represents voter ``j``'s outgoing vote distribution and
      must sum to 1 (column-stochastic).
    - Rows/columns ``[0, num_delegates)`` are delegates.
    - Rows/columns ``[num_delegates, num_delegates + num_intermediates)`` are
      intermediates.
    - The remaining rows/columns are policies. The policy × policy block must
      be the identity matrix (absorbing states), and policies must have zero
      outgoing votes to non-policy rows.

    Args:
        v: Square ``(n, n)`` column-stochastic ndarray. ``v[i, j]`` is the
            fraction of voter ``j``'s voting power that flows to entity ``i``.
        labels: Length-``n`` list of names, in the same order as the matrix
            rows/columns. Used to label the returned tuples.
        num_delegates: Number of delegate rows/columns at the top of ``v``.
        num_intermediates: Number of intermediate rows/columns immediately
            after the delegates.
        tol: Convergence threshold on the max remaining transient mass. Stops
            iterating once ``A_k[:ndi, :].max() < tol``.
        max_iter: Hard cap on iterations. If exceeded without convergence,
            emits a ``UserWarning`` but still returns the partial result.

    Returns:
        See :func:`compute`

    Example:

        >>> import numpy as np
        >>> v = np.array([
        ...     #  Alice  Bob  RedFruit  apple  banana
        ...     [   0.0,  0.2,    0.0,    0.0,   0.0 ],  # → Alice
        ...     [   0.0,  0.0,    0.0,    0.0,   0.0 ],  # → Bob
        ...     [   0.3,  0.0,    0.0,    0.0,   0.0 ],  # → RedFruit
        ...     [   0.7,  0.0,    1.0,    1.0,   0.0 ],  # → apple (absorbing)
        ...     [   0.0,  0.8,    0.0,    0.0,   1.0 ],  # → banana (absorbing)
        ... ])
        >>> consensus, influences = compute_matrix(
        ...     v,
        ...     labels=['Alice', 'Bob', 'RedFruit', 'apple', 'banana'],
        ...     num_delegates=2,
        ...     num_intermediates=1,
        ... )
        >>> consensus[0].label
        'apple'

    Raises:
        AssertionError: If ``v`` isn't square, labels don't match its size,
            columns don't sum to 1, voters self-vote, or the policy block
            isn't a clean absorbing identity.
    """
    n = v.shape[0]
    ndi = num_delegates + num_intermediates
    num_policies = n - ndi

    assert v.shape == (n, n), "v must be square"
    assert len(labels) == n, "labels must match matrix size"
    assert np.allclose(v.sum(axis=0), 1.0), "columns must sum to 1 (column-stochastic)"
    assert np.allclose(np.diag(v[:ndi, :ndi]), 0), "voters cannot vote for themselves"
    assert np.allclose(v[ndi:, ndi:], np.eye(num_policies)), (
        "policies must be absorbing (identity block)"
    )
    assert np.allclose(v[:ndi, ndi:], 0), "policies must not vote outward"

    # Joint squaring recurrence on the transient block Q = v[:ndi, :ndi]:
    #   P_m = Q^(2^m)           T_m = I + Q + Q^2 + ... + Q^(2^m - 1)
    #   T_{m+1} = T_m + T_m·P_m   P_{m+1} = P_m·P_m
    # Reaches V^k after ~log2(k) squarings instead of k sequential GEMMs.
    # T_inf equals the transient block of the original `influence` matrix.
    # GEMMs run on the GPU via cuBLAS sgemm with TF32 enabled.
    q = cp.asarray(v[:ndi, :ndi], dtype=cp.float32)
    r = cp.asarray(v[ndi:, :ndi], dtype=cp.float32)

    p = q.copy()
    t = cp.eye(ndi, dtype=cp.float32)

    # Adaptive precision: fp16 tensor-core GEMM while P entries are still
    # large enough to survive the fp16 cast; TF32 below the threshold so the
    # tail iterations (which set the final ordering of low-mass policies)
    # don't lose precision.
    #
    # fp16 path is tightened vs the naive version: P is cast once per iter and
    # reused across both GEMMs, and T += T·P is fused into a single gemmEx
    # call with beta=1 (saving the separate elementwise add and one fp32
    # buffer pass over T).
    p_new = cp.empty((ndi, ndi), dtype=cp.float32)
    for _ in range(max_iter):
        p_max = float(p.max())
        if p_max < tol:
            break
        if p_max > _SWITCH_THRESH:
            p_h = p.astype(cp.float16)
            t_h = t.astype(cp.float16)
            _gemm_acc(t_h, p_h, t, accumulate=True)
            _gemm_acc(p_h, p_h, p_new, accumulate=False)
            p, p_new = p_new, p
        else:
            t = t + t @ p
            p = p @ p
    else:
        warnings.warn(
            f"did not converge within {max_iter} squarings "
            f"(max transient mass = {float(p.max()):.2e})",
            stacklevel=2,
        )

    e_d = cp.zeros(ndi, dtype=cp.float32)
    e_d[:num_delegates] = 1.0
    # Reduce right-to-left so the intermediate is a vector (ndi,) rather than
    # the (num_policies, ndi) matrix r·t — saves an n×n GEMM at the end.
    policy_totals = cp.asnumpy(r @ (t @ e_d)).tolist()
    consensus = sorted(
        (
            Consensus(label, value)
            for label, value in zip(labels[ndi:], policy_totals, strict=True)
        ),
        key=lambda c: c.value,
        reverse=True,
    )

    inf_values = cp.asnumpy(t.sum(axis=1) / t.diagonal()).tolist()
    roles: list[Role] = ["delegate"] * num_delegates + [
        "intermediate"
    ] * num_intermediates
    influences = sorted(
        (
            Influence(label, role, value)
            for label, role, value in zip(labels[:ndi], roles, inf_values, strict=True)
        ),
        key=lambda i: i.value,
        reverse=True,
    )

    return (consensus, influences)
