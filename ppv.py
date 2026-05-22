"""MLX-backed squaring implementation.

Uses Apple's MLX library — runs the dense GEMM on Apple GPU using MLX's
internal matmul kernels (separate from MPSMatrixMultiplication, which the
swift_metal_squaring branch hit known precision issues with at n>=15k).

MLX uses lazy evaluation: operations are queued and only executed when you
read a value (``.item()``, ``mx.eval(arr)``, conversion to numpy). The
convergence check forces a sync each iteration.

Build/install: `pip install mlx`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, NamedTuple

import mlx.core as mx
import numpy as np

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
    """Propagational Proxy Voting via MLX squaring kernel on Apple GPU."""
    q_np, r_np, labels, num_delegates, num_intermediates = _build_blocks(
        delegates, intermediates, policies
    )
    ndi = q_np.shape[0]
    num_policies = r_np.shape[0]

    # Hand off to MLX. After this point we stay on the GPU until the convergence
    # check pulls scalars back across the boundary.
    q = mx.array(q_np)
    r = mx.array(r_np)
    t = mx.eye(ndi, dtype=mx.float32)
    p = q  # mx arrays are immutable; aliasing is safe

    iters = 0
    for m in range(max_iter):
        # `.item()` forces eval up to and including p's current value.
        p_max = p.max().item()
        if p_max < tol:
            iters = m
            break
        t = t + t @ p
        p = p @ p
        iters = m + 1
    else:
        warnings.warn(
            f"MLX squaring did not converge within {max_iter} iterations",
            stacklevel=2,
        )

    # Final outputs.
    e_d_np = np.zeros(ndi, dtype=np.float32)
    e_d_np[:num_delegates] = 1.0
    e_d = mx.array(e_d_np)

    consensus_vec = np.asarray(r @ t @ e_d)
    row_sums = np.asarray(t.sum(axis=1))
    diag = np.asarray(mx.diagonal(t))
    inf_values = row_sums / diag

    log.debug("mlx_squaring: ndi=%d iters=%d", ndi, iters)

    consensus = sorted(
        (
            Consensus(label, float(value))
            for label, value in zip(labels[ndi:], consensus_vec, strict=True)
        ),
        key=lambda c: c.value,
        reverse=True,
    )
    roles: list[Role] = ["delegate"] * num_delegates + [
        "intermediate"
    ] * num_intermediates
    influences = sorted(
        (
            Influence(label, role, float(value))
            for label, role, value in zip(labels[:ndi], roles, inf_values,
                                          strict=True)
        ),
        key=lambda i: i.value,
        reverse=True,
    )
    return consensus, influences


def _build_blocks(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    """Build dense row-major fp32 Q (ndi x ndi) and R (num_policies x ndi)."""
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
    index_of = {label: idx for idx, label in enumerate(labels)}

    q = np.zeros((ndi, ndi), dtype=np.float32)
    r = np.zeros((num_policies, ndi), dtype=np.float32)

    voter_dicts = list(delegates.items()) + list(intermediates.items())
    for j, (voter, edges) in enumerate(voter_dicts):
        assert voter not in edges, f"voter {voter!r} cannot vote for themselves"
        col_total = 0.0
        for target, weight in edges.items():
            i = index_of[target]
            col_total += weight
            if i < ndi:
                q[i, j] = weight
            else:
                r[i - ndi, j] = weight
        assert abs(col_total - 1.0) < 1e-6, (
            f"voter {voter!r} edge weights sum to {col_total}, must sum to 1"
        )

    return q, r, labels, num_delegates, num_intermediates
