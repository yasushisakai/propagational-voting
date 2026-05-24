"""Step-1 validation for the cupy_multi branch (MULTI_GPU_PLAN.md).

Exercises the NCCL init, memory-gate, padding, and Q-sharding helpers in
isolation. compute_matrix is NOT yet using them — that comes in step 2.
This script is the gate that says "the foundations are sound before we
start the algorithm rewrite."

Run:    CUDA_VISIBLE_DEVICES=0,1,2 .venv/bin/python validate_multi_gpu.py
Exit 0 = all checks pass; exit 1 = something failed.
"""

from __future__ import annotations

import os
import sys

import cupy as cp
import numpy as np


def main() -> int:
    print("=== cupy_multi step 1 validation ===\n", flush=True)

    # --- 1. NCCL communicator setup ----------------------------------------
    print("1. NCCL communicator init...", flush=True)
    from ppv import _NUM_GPUS, _get_nccl_comms

    comms = _get_nccl_comms()
    assert len(comms) == _NUM_GPUS, f"expected {_NUM_GPUS} comms, got {len(comms)}"
    # Calling again should return the same cached list (no re-init).
    assert _get_nccl_comms() is comms, "lazy init not cached"
    print(f"   OK: {_NUM_GPUS} ranks initialized, cache works", flush=True)

    # --- 2. NCCL ring round-trip: send a scalar around all 3 GPUs ----------
    print("\n2. NCCL all-reduce smoke test...", flush=True)
    from cupy.cuda import nccl

    # Each rank starts with its own rank id; max-all-reduce should give p-1.
    values = []
    streams = []
    for rank in range(_NUM_GPUS):
        with cp.cuda.Device(rank):
            v = cp.array([float(rank)], dtype=cp.float32)
            values.append(v)
            streams.append(cp.cuda.Stream(non_blocking=True))
    nccl.groupStart()
    for rank in range(_NUM_GPUS):
        with cp.cuda.Device(rank):
            comms[rank].allReduce(
                values[rank].data.ptr, values[rank].data.ptr,
                1, nccl.NCCL_FLOAT32, nccl.NCCL_MAX, streams[rank].ptr,
            )
    nccl.groupEnd()
    for rank in range(_NUM_GPUS):
        with cp.cuda.Device(rank):
            streams[rank].synchronize()
            got = float(values[rank].get()[0])
            assert got == float(_NUM_GPUS - 1), (
                f"rank {rank}: expected {_NUM_GPUS - 1}, got {got}"
            )
    print(f"   OK: all-reduce(MAX) returned {_NUM_GPUS - 1} on every rank", flush=True)

    # --- 3. Memory gate: should PASS for tiny workload ---------------------
    print("\n3. Memory gate (small workload)...", flush=True)
    from ppv import _check_gpu_memory, _estimate_per_gpu_bytes

    est_small = _estimate_per_gpu_bytes(ndi=1000)
    print(f"   est for ndi=1000: {est_small / 1e9:.2f} GB", flush=True)
    _check_gpu_memory(ndi=1000)
    print("   OK: passes for ndi=1000", flush=True)

    # --- 4. Memory gate: should FAIL for absurd workload -------------------
    print("\n4. Memory gate (impossible workload)...", flush=True)
    huge_ndi = 200_000
    est_huge = _estimate_per_gpu_bytes(ndi=huge_ndi)
    print(f"   est for ndi={huge_ndi}: {est_huge / 1e9:.1f} GB", flush=True)
    try:
        _check_gpu_memory(ndi=huge_ndi)
    except RuntimeError as exc:
        first_line = str(exc).splitlines()[0]
        print(f"   OK: bailed cleanly — {first_line!r}", flush=True)
    else:
        print("   FAIL: should have raised", flush=True)
        return 1

    # --- 5. Memory gate: PPV_SKIP_MEMORY_CHECK bypass ----------------------
    print("\n5. Memory gate (PPV_SKIP_MEMORY_CHECK=1 bypass)...", flush=True)
    os.environ["PPV_SKIP_MEMORY_CHECK"] = "1"
    try:
        _check_gpu_memory(ndi=huge_ndi)
        print("   OK: bypass honored — no raise", flush=True)
    finally:
        del os.environ["PPV_SKIP_MEMORY_CHECK"]

    # --- 6. ndi padding ----------------------------------------------------
    print("\n6. ndi padding to multiples of NUM_GPUS...", flush=True)
    from ppv import _pad_ndi

    cases = [
        (99, 99),     # already multiple of 3
        (100, 102),   # round up by 2
        (101, 102),   # round up by 1
        (2420, 2421),
        (19600, 19602),
    ]
    for ndi, expected in cases:
        got = _pad_ndi(ndi)
        assert got == expected, f"_pad_ndi({ndi}) = {got}, want {expected}"
    print(f"   OK: {len(cases)} cases matched", flush=True)

    # --- 7. Q sharding -----------------------------------------------------
    print("\n7. Q sharding across 3 GPUs...", flush=True)
    from ppv import _shard_q

    ndi = 10
    rng = np.random.default_rng(0)
    # v is (n_total, n_total) where n_total = ndi + num_policies; we only
    # care about the [:ndi, :ndi] block.
    v = rng.random((15, 15), dtype=np.float32)
    shards = _shard_q(v, ndi=ndi)
    assert len(shards) == _NUM_GPUS, f"expected {_NUM_GPUS} shards"

    ndi_padded = _pad_ndi(ndi)              # 12
    rows_per_shard = ndi_padded // _NUM_GPUS  # 4
    expected_shape = (rows_per_shard, ndi_padded)

    # Each shard lives on its own device with the right shape and content.
    reassembled = np.zeros((ndi_padded, ndi_padded), dtype=np.float32)
    for i, shard in enumerate(shards):
        assert shard.shape == expected_shape, f"GPU {i}: shape {shard.shape}"
        with cp.cuda.Device(i):
            reassembled[i * rows_per_shard:(i + 1) * rows_per_shard] = cp.asnumpy(shard)
    # The unpadded block should match v[:ndi, :ndi] exactly; the padded
    # region (rows ≥ ndi or cols ≥ ndi) should be zero.
    assert np.array_equal(reassembled[:ndi, :ndi], v[:ndi, :ndi]), "unpadded mismatch"
    assert np.all(reassembled[ndi:, :] == 0), "padded rows nonzero"
    assert np.all(reassembled[:, ndi:] == 0), "padded cols nonzero"
    print(f"   OK: 3 shards of {expected_shape}, content + padding correct", flush=True)

    # --- 8. Memory estimate matches single-GPU baseline order of magnitude -
    print("\n8. Memory estimate sanity at current bench sizes...", flush=True)
    for ndi in (1000, 5000, 19600):
        est_bytes = _estimate_per_gpu_bytes(ndi)
        print(f"   ndi={ndi:>5}: est per-GPU = {est_bytes / 1e9:>5.2f} GB", flush=True)
    # The single-GPU cupy_fp16 row at ndi=19600 was 8.8 GB total; sharded
    # across 3 GPUs we expect ~3-4 GB worst-case shard plus 1.5 GB overhead.
    # Generous range: bigger means we're being conservative, smaller would
    # mean the budget is unrealistic.
    est_20k = _estimate_per_gpu_bytes(19600)
    assert 4e9 < est_20k < 9e9, (
        f"est for ndi=19600 = {est_20k / 1e9:.2f} GB outside expected 4-9 GB range"
    )
    print(f"   OK: {est_20k / 1e9:.2f} GB per GPU at ndi=19600 "
          f"(single-GPU cupy_fp16 used 8.8 GB total)", flush=True)

    print("\n=== all step-1 validations passed ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
