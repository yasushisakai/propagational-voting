"""Python wrapper around the C+Accelerate squaring kernel.

The heavy lifting (the squaring loop, dense GEMM, vector reductions) runs in
``c_backend/libppv_squaring.dylib``. This module is a thin ctypes shim that
keeps the public ``compute(...)`` signature unchanged from earlier branches —
the gatekeeper test imports ``compute`` and never sees the C call.

Build the dylib once with ``make -C c_backend`` before running any tests.
"""

from __future__ import annotations

import ctypes
import logging
import os
import warnings
from typing import Literal, NamedTuple

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


_DLL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "c_backend",
    "libppv_squaring.dylib",
)

if not os.path.exists(_DLL_PATH):
    raise ImportError(
        f"C+Accelerate kernel not found at {_DLL_PATH}. "
        f"Build it with `make -C c_backend` (requires clang + Accelerate)."
    )

_lib = ctypes.cdll.LoadLibrary(_DLL_PATH)
_lib.ppv_c_squaring.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # Q
    ctypes.POINTER(ctypes.c_double),  # R
    ctypes.c_int,                      # ndi
    ctypes.c_int,                      # num_delegates
    ctypes.c_int,                      # num_policies
    ctypes.c_double,                   # tol
    ctypes.c_int,                      # max_iter
    ctypes.POINTER(ctypes.c_double),  # consensus_out
    ctypes.POINTER(ctypes.c_double),  # influence_out
]
_lib.ppv_c_squaring.restype = ctypes.c_int


def compute(
    delegates: dict[str, dict[str, float]],
    intermediates: dict[str, dict[str, float]],
    policies: list[str],
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> tuple[list[Consensus], list[Influence]]:
    """Propagational Proxy Voting via C+Accelerate squaring kernel.

    Same API as earlier Python implementations. See `help(ppv.compute)` on the
    `baseline` or `squaring-v1` tags for full docstring; behavior is identical
    up to floating-point noise.

    Args mirror the Python signature; `tol` and `max_iter` pass through to the
    C kernel's convergence loop.
    """
    q, r, labels, num_delegates, num_intermediates = _build_blocks(
        delegates, intermediates, policies
    )
    ndi = q.shape[0]
    num_policies = r.shape[0]

    consensus_arr = np.zeros(num_policies, dtype=np.float64)
    influence_arr = np.zeros(ndi, dtype=np.float64)

    iters = _lib.ppv_c_squaring(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ndi, num_delegates, num_policies,
        tol, max_iter,
        consensus_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        influence_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if iters < 0:
        raise MemoryError("C kernel allocation failed")
    if iters >= max_iter:
        warnings.warn(
            f"C kernel did not converge within {max_iter} squarings",
            stacklevel=2,
        )

    log.debug("c_squaring: ndi=%d iters=%d num_policies=%d",
              ndi, iters, num_policies)

    consensus = sorted(
        (
            Consensus(label, float(value))
            for label, value in zip(labels[ndi:], consensus_arr, strict=True)
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
            for label, role, value in zip(labels[:ndi], roles, influence_arr,
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
    """Build dense row-major Q (ndi x ndi) and R (num_policies x ndi) directly from dicts.

    Skips the full n x n matrix that earlier branches built — at n=20000 that
    intermediate alone was 3.2 GB. Now we go dict → Q + R, ~2.4 GB total.
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
    index_of = {label: idx for idx, label in enumerate(labels)}

    q = np.zeros((ndi, ndi), dtype=np.float64)
    r = np.zeros((num_policies, ndi), dtype=np.float64)

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
        assert abs(col_total - 1.0) < 1e-9, (
            f"voter {voter!r} edge weights sum to {col_total}, must sum to 1"
        )

    return q, r, labels, num_delegates, num_intermediates
