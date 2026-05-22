import logging
import warnings
from typing import Literal, NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

log = logging.getLogger(__name__)

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
    their names — no identity block is materialized internally.

    Ordering inside outputs is fixed: delegates first, then intermediates,
    then policies. Labels must be unique across all three groups.

    Args:
        delegates: Mapping of delegate name to their outgoing votes,
            e.g. ``{'Alice': {'Bob': 0.2, 'FAR2': 0.8}}``. Targets may be other
            delegates, intermediates, or policies. A voter cannot vote for
            themselves (no self-key).
        intermediates: Same shape as ``delegates``. Intermediates re-distribute
            mass they receive but are not themselves a final destination.
        policies: Ordered list of policy names (absorbing states).
        tol: Unused by the direct solver. Retained for API compatibility with
            the iterative baseline; ignored here.
        max_iter: Unused by the direct solver. Retained for API compatibility.

    Returns:
        ``(consensus, influences)`` where:
          - ``consensus``: list of :class:`Consensus`, sorted descending by value,
            one entry per policy.
          - ``influences``: list of :class:`Influence`, sorted descending by value,
            one entry per delegate + intermediate.

    Implementation:
        This branch (`sparse-solve`) uses a direct sparse LU factorization of
        ``(I − Q)`` (where ``Q`` is the transient × transient block of the
        column-stochastic voting matrix). All quantities are then computed by
        triangular solves against the single factor:

          - consensus = R · (I − Q)⁻¹ · e_d
          - row sums of (I − Q)⁻¹ via one solve with RHS = 1
          - diagonal of (I − Q)⁻¹ via block solves with unit-vector RHSs

        This converges to machine precision (no truncation), avoids the
        O(n²) dense matrix the iterative baseline allocates, and turns the
        O(k·n³) iteration into one O(nnz) factor plus a handful of solves.
    """
    del tol, max_iter  # not used by the direct solver

    q, r, labels, num_delegates, num_intermediates = _build_sparse(
        delegates, intermediates, policies
    )

    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "sparse system: Q.nnz=%d R.nnz=%d ndi=%d num_policies=%d",
            q.nnz, r.nnz, q.shape[0], r.shape[0],
        )

    return _solve_sparse(q, r, labels, num_delegates, num_intermediates)


def _build_sparse(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
) -> tuple[sp.csc_matrix, sp.csc_matrix, list[str], int, int]:
    """Assemble sparse Q (transient × transient) and R (policy × transient) from dicts.

    Q[i, j] = weight that transient voter j sends to transient target i.
    R[p, j] = weight that transient voter j sends to policy p.
    Each column j of [Q; R] sums to 1 (column-stochastic).
    """
    d_labels = list(delegates)
    i_labels = list(intermediates)
    p_labels = list(policies)
    labels = d_labels + i_labels + p_labels

    assert len(labels) == len(set(labels)), (
        "labels must be unique across delegates, intermediates, policies"
    )

    num_delegates = len(d_labels)
    num_intermediates = len(i_labels)
    ndi = num_delegates + num_intermediates
    num_policies = len(p_labels)

    index_of = {label: i for i, label in enumerate(labels)}

    q_rows: list[int] = []
    q_cols: list[int] = []
    q_data: list[float] = []
    r_rows: list[int] = []
    r_cols: list[int] = []
    r_data: list[float] = []

    voter_dicts = list(delegates.items()) + list(intermediates.items())
    for j, (voter, edges) in enumerate(voter_dicts):
        assert voter not in edges, f"voter {voter!r} cannot vote for themselves"
        col_total = 0.0
        for target, weight in edges.items():
            i = index_of[target]
            col_total += weight
            if i < ndi:
                q_rows.append(i)
                q_cols.append(j)
                q_data.append(weight)
            else:
                r_rows.append(i - ndi)
                r_cols.append(j)
                r_data.append(weight)
        assert abs(col_total - 1.0) < 1e-9, (
            f"voter {voter!r} edge weights sum to {col_total}, must sum to 1"
        )

    q = sp.csc_matrix(
        (q_data, (q_rows, q_cols)),
        shape=(ndi, ndi),
        dtype=np.float64,
    )
    r = sp.csc_matrix(
        (r_data, (r_rows, r_cols)),
        shape=(num_policies, ndi),
        dtype=np.float64,
    )
    return q, r, labels, num_delegates, num_intermediates


def _solve_sparse(
    q: sp.csc_matrix,
    r: sp.csc_matrix,
    labels: list[str],
    num_delegates: int,
    num_intermediates: int,
) -> tuple[list[Consensus], list[Influence]]:
    """Direct sparse LU on (I − Q); derive consensus + influence from the factor."""
    ndi = num_delegates + num_intermediates

    # One factorization, used for every solve below.
    lu = spla.splu((sp.eye(ndi, format="csc") - q).tocsc())

    # consensus = R · N · e_d, where N = (I − Q)^-1 and e_d picks delegate columns.
    e_d = np.zeros(ndi)
    e_d[:num_delegates] = 1.0
    consensus_vec = (r @ lu.solve(e_d))

    # Row sums of N: solve (I − Q) x = 1 ⇒ x = N · 1.
    row_sums = lu.solve(np.ones(ndi))

    # Diagonal of N: solve in column blocks against unit-vector RHSs.
    # Block size trades Python overhead vs. RHS memory footprint
    # (each block is an ndi × block_size dense matrix).
    diag = _diag_inverse(lu, ndi)
    inf_values = (row_sums / diag).tolist()

    if (diag <= 0).any():
        warnings.warn(
            "non-positive diagonal in (I − Q)^-1; system may be near-singular",
            stacklevel=2,
        )

    policy_totals = consensus_vec.tolist()
    consensus = sorted(
        (
            Consensus(label, value)
            for label, value in zip(labels[ndi:], policy_totals, strict=True)
        ),
        key=lambda c: c.value,
        reverse=True,
    )

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


def _diag_inverse(lu: spla.SuperLU, ndi: int, block_size: int = 512) -> np.ndarray:
    """Return diag((I − Q)^-1) by solving against unit-vector RHSs in blocks.

    Each block builds a dense ndi × block_size RHS containing a slice of the
    identity matrix, calls one batched triangular solve, then extracts the
    relevant diagonal entries from the result.
    """
    diag = np.empty(ndi, dtype=np.float64)
    for start in range(0, ndi, block_size):
        end = min(start + block_size, ndi)
        width = end - start
        rhs = np.zeros((ndi, width), dtype=np.float64)
        for col, j in enumerate(range(start, end)):
            rhs[j, col] = 1.0
        sol = lu.solve(rhs)
        for col, j in enumerate(range(start, end)):
            diag[j] = sol[j, col]
    return diag
