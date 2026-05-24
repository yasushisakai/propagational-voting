# `cupy_multi` design doc

Sharded multi-GPU implementation of the squaring kernel. Built on top of
`cupy_fp16` (inherits the fp16 tensor-core GEMM strategy). Intended to (a)
shrink wall-time on the existing biggest cells by ~1.6× and (b) raise the
n cap from ~45k single-GPU to ~65k three-GPU.

## Decisions (settled)

| | choice |
|---|---|
| Process model | Single-process, multi-device (cupy `Device(i)` contexts) |
| Comm library | `cupy.cuda.nccl` — single communicator, one rank per GPU |
| Sharding | 1D row-shard of P and T; R, e_d replicated |
| Hardware tag in CSV | `3x RTX 6000 Ada` |
| `peak_rss_mb` semantics | `max()` across GPUs (the bottleneck card) |
| GPU set | All three: `CUDA_VISIBLE_DEVICES=0,1,2` |
| Shared-box safety | Pre-flight memory check; bail out clearly if any GPU lacks headroom |

## Sharding layout

```
  ndi rows (padded up to multiple of 3)
  ┌─────────────────────────────────┐
  │ GPU 0: rows [0 .. ndi/3)        │  ← P_0, T_0
  ├─────────────────────────────────┤
  │ GPU 1: rows [ndi/3 .. 2·ndi/3)  │  ← P_1, T_1
  ├─────────────────────────────────┤
  │ GPU 2: rows [2·ndi/3 .. ndi)    │  ← P_2, T_2
  └─────────────────────────────────┘
```

- `Q` is sharded on construction; never reconstituted whole.
- `R` (num_policies × ndi, small) and `e_d` are replicated on every GPU.
- `ndi` padded up to nearest multiple of 3 (at most 2 zero rows). Padded
  rows are trimmed off the final influence / consensus output.

## Per-iter algorithm (pipelined ring all-gather GEMM)

For `C = A @ B` with `A`, `B` both row-sharded, every GPU computes its
row-shard of `C` via 3 rounds, each consuming one P shard from the ring:

```
# rank = this GPU's index in [0, p)
C_shard = zeros((ndi/p, ndi))     # output shard accumulator
P_shard = P_local                 # round 0: own shard
for k in range(p):
    src_shard = (rank + k) % p    # which row-shard of B we're using
    col_lo, col_hi = src_shard * ndi/p, (src_shard + 1) * ndi/p
    C_shard[:, col_lo:col_hi] += A_local @ P_shard
    if k < p - 1:
        P_shard = nccl_send_recv_ring_rotate(P_shard)
```

- A single rotation = O(ndi²/p) bytes per GPU = ~510 MB at n=20000, p=3.
- 2 rotations per ring (covers p-1 hops, since round 0 uses own shard).
- Each per-round GEMM is `(ndi/p) × (ndi/p)` square — 1/9 the work of the
  monolithic single-GPU GEMM. With p GPUs running in parallel, theoretical
  compute speedup = p (3× here), modulo comm overhead.

Both GEMMs in the iter (`T @ P` and `P @ P`) share the same P-shard ring
schedule — we cast each rotated P shard to fp16 once and reuse it for both
GEMMs. Same for T_local (cast once per iter).

For the fused `T += T @ P`: each round's GEMM uses `beta=1, C=T_local` so
the partial products accumulate into T_local directly. Round 0 starts from
T_local's existing value; rounds 1..p-1 keep accumulating. Matches the
fused-accumulate trick from single-GPU.

## Convergence check

```
local_max = float(P_local.max())               # device-side reduction + sync
global_max = nccl_all_reduce(local_max, op=MAX)  # 4 bytes, microseconds
if global_max < tol: break
```

## Final reductions

1. `T @ e_d`: each GPU does local `T_local @ e_d` → vector of length ndi/p.
   `ncclAllGather` → full vector on every GPU.
2. `R @ (T @ e_d)`: local on every GPU (R is replicated). Identical result
   everywhere; take rank 0's copy.
3. `T.sum(axis=1)`: each GPU sums its own row shard → length ndi/p.
   `ncclAllGather` → full vector.
4. `T.diagonal()`: diagonal entries within rows `[i·ndi/p, (i+1)·ndi/p)`
   live on GPU i — extract local diagonal slice. `ncclAllGather` → full
   vector.
5. Local divide on rank 0; build `Influence` / `Consensus` on host.

## Pre-flight memory check (shared-box safety)

Multi-GPU box is communal — we cannot assume any GPU has its full 48 GB
free. Before any allocation we:

1. Query free memory per visible GPU via `cp.cuda.runtime.memGetInfo()`.
2. Compute estimated per-GPU peak (formula in [Memory budget](#memory-budget)).
3. Multiply by `1 + headroom` (default `headroom = 0.30` → require 30%
   slack). Bail out if any GPU's free memory falls short.

Bail-out is a clear `RuntimeError`:

```
cupy_multi requires ~4.6 GB headroom on each visible GPU; gpu 0 has
2.1 GB free. Free up GPU 0 (another process is using ~46 GB) or run
on cupy_fp16 single-GPU.
```

No automatic fallback to single-GPU — that's the caller's decision.

### Tuning knobs (env vars, no API change)

| var | default | meaning |
|---|---|---|
| `PPV_HEADROOM_FRACTION` | `0.30` | Demand `(1 + this) * estimated` to pass the gate |
| `PPV_SKIP_MEMORY_CHECK` | unset | If set to `1`, skip the gate entirely (let cupy OOM if it OOMs) |

## Memory budget

Estimated per-GPU peak at ndi (p=3):

| buffer | bytes per GPU |
|---|---|
| P_local fp32 | 4·(ndi/p)·ndi |
| T_local fp32 | 4·(ndi/p)·ndi |
| P_ring fp32 (incoming shard during rotation) | 4·(ndi/p)·ndi |
| p_new fp32 (ping-pong target for P=P·P) | 4·(ndi/p)·ndi |
| ring accumulate scratch fp32 (T += partials) | 4·(ndi/p)·ndi |
| fp16 cast of P_local | 2·(ndi/p)·ndi |
| fp16 cast of P_ring | 2·(ndi/p)·ndi |
| fp16 cast of T_local | 2·(ndi/p)·ndi |
| cuBLAS workspace, framework | ~1.5 GB constant |

Total formula: `est_bytes ≈ (26 · ndi²)/p + 1.5e9`. Some of those buffers can
share storage in a tighter implementation, but we size for the worst case so
the memory gate doesn't surprise the user with an OOM.

At ndi=20000, p=3: ~3.4 GB compute + 1.5 GB overhead = **~5 GB per GPU**.
At ndi=44000 (single-GPU cap): ~16.8 GB + 1.5 GB = **~18 GB per GPU**.
At ndi=66000 (theoretical 3-GPU cap): ~37.8 GB + 1.5 GB = **~39 GB per GPU**.

## Numerical impact

Pipelined ring sums partial products in a different order than monolithic
GEMM. With fp32 accumulator the difference is bounded by a few ULPs per
accumulation. Expected: consensus values differ by ~1e-3 from cupy_fp16,
same character as the cupy_fp16 → fp64 gap. **`check_ordering.py` should
pass** with tie-only inversions; verify before benching.

## Failure modes

1. **NCCL hang** if GPUs disagree on the collective sequence. Mitigation:
   strict same-order issuance in code; assert collective count at end of
   each iter.
2. **PCIe P2P misconfig.** Probe `cudaCanAccessPeer` at init; assert; on
   this box all GPUs share a PCIe root complex so P2P should work.
3. **Stream sync.** Comm stream must signal an event the compute stream
   waits on before the consuming GEMM reads the rotated shard. Explicit
   `event.record()` → `wait_event()`.
4. **ndi not divisible by p.** Pad Q with zero rows at construction; trim
   final outputs.
5. **GPU 0 contended.** Pre-flight memory check catches this.

## Bench logistics

- `VERSION = "cupy_multi"`
- `HARDWARE = "3x RTX 6000 Ada"`
- `peak_rss_mb = max(cp.get_default_memory_pool().total_bytes()) / MB`
  taken across the 3 device contexts after each cell, then `free_all_blocks`
  on each before the next cell.
- All 21 existing snapshots; cherry-pick CSV rows to main as before.

## Implementation order

1. **Skeleton & memory gate** (~½ day): branch off cupy_fp16; create NCCL
   communicator; write `_check_gpu_memory(ndi)`; pad ndi; shard Q.
2. **Naive non-pipelined version** (~1 day): all-gather full P onto every
   GPU each iter, do local GEMM, get correctness working. Verify ordering
   gate passes. Memory-inefficient but a reference for step 3.
3. **Pipelined ring** (~2 days): replace bulk all-gather with overlapping
   ring rotation. Confirm same numerical output as step 2 within tolerance.
4. **Stream overlap** (~1 day): comm and compute on separate streams;
   measure actual overlap with Nsight.
5. **Bench + cherry-pick** (~½ day): run all 21 cells, cherry-pick CSV
   rows to main.
6. **README caveat** (~½ day): document `3x RTX 6000 Ada` semantics,
   `peak_rss_mb = max(per_gpu)`, env-var knobs.

Estimate: ~1 week focused, 2 weeks at normal pace.

## Open items

- **Initial input transfer.** `v` arrives as a host numpy array. We can
  copy whole-`v` to all 3 GPUs (small for typical cell sizes), or shard at
  copy time (saves device-to-device shard later). Probably whole-`v` to
  device 0, then NCCL broadcast — simplest. Decide during implementation.
- **GPU 0 contention.** Document `nvidia-smi` check in the bench README
  section so future contributors don't get spurious OOMs.
- **Pad/unpad bookkeeping.** Final consensus and influence outputs must
  drop the padded rows. Trivial slice but easy to forget.
