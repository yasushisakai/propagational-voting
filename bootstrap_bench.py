"""Generate benchmark snapshots and record baseline timings.

Reusable: edit CELLS and re-run to add new sizes later. Idempotent — cells whose
snapshot already exists locally are skipped (their CSV row is preserved).

Outputs (committed):
  - tests/snapshots.manifest.json   — filenames + sha256s + release URL
  - benchmark.csv                   — one row per (version, cell, seed)

Outputs (NOT committed; gitignored, hosted on GitHub Releases):
  - tests/snapshots/*.npz           — self-contained input+output bundles

Flags:
  --manifest-only   regenerate the manifest from existing snapshots without
                    running any compute (useful after manually adding/removing
                    .npz files).

Workflow for adding a new cell size:
  1. Add a new tuple to CELLS below.
  2. Run `python bootstrap_bench.py` (cached cells skip, new ones compute).
  3. Upload the new snapshots:
       gh release upload snapshots-v1 tests/snapshots/n<N>_seed*.npz
  4. Commit the updated tests/snapshots.manifest.json and benchmark.csv.

Workflow for benchmarking an alternative implementation:
  1. Copy this file (e.g. `bench_squaring.py`).
  2. Change VERSION below and the `compute` import to point at the new impl.
  3. Run; rows for the new VERSION append next to baseline rows in benchmark.csv.
     Rows from other VERSIONs pass through untouched (no clobber).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

from ppv import compute

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(REPO_ROOT, "tests", "snapshots")
MANIFEST_PATH = os.path.join(REPO_ROOT, "tests", "snapshots.manifest.json")
CSV_PATH = os.path.join(REPO_ROOT, "benchmark.csv")

RELEASE_TAG = "snapshots-v2"
RELEASE_BASE_URL = (
    "https://github.com/yasushisakai/propagational-voting"
    f"/releases/download/{RELEASE_TAG}"
)

# Label for the rows this script writes to benchmark.csv. Future implementations
# (squaring, scipy.sparse, C+Accelerate, Swift+Metal) should use their own
# label and append to the same CSV so versions can be compared side-by-side.
VERSION = "squaring_fp32"

# (n_delegates, n_intermediates, n_policies, seeds)
# All cells produce snapshots. "slow" marker on the gatekeeper test is decided
# by cell size, not by this list — see tests/test_gatekeeper.py.
CELLS: list[tuple[int, int, int, list[int]]] = [
    (100,   20,   5,   [0, 1, 2]),  # n_tot=  125, ~0.01s/seed
    (500,   100,  10,  [0, 1, 2]),  # n_tot=  610, ~1.0s/seed
    (1000,  200,  10,  [0, 1, 2]),  # n_tot= 1210, ~12s/seed
    (2000,  400,  20,  [0, 1, 2]),  # n_tot= 2420, ~100s/seed
    (4000,  900,  100, [0, 1, 2]),  # n_tot= 5000, ~5.5min/seed (observed)
    (8000,  1800, 200, [0, 1, 2]),  # n_tot=10000, ~40min/seed (observed)
    (16000, 3600, 400, [0, 1, 2]),  # n_tot=20000, ~3.8h/seed (estimated)
]
DELEGATE_OUT = 100
INTERMEDIATE_OUT = 10


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_inputs(
    n_d: int, n_i: int, n_p: int, seed: int,
    delegate_out: int = DELEGATE_OUT,
    intermediate_out: int = INTERMEDIATE_OUT,
) -> tuple[dict, dict, list[str]]:
    rng = np.random.default_rng(seed)
    d_labels = [f"D{i:05d}" for i in range(n_d)]
    i_labels = [f"I{i:05d}" for i in range(n_i)]
    p_labels = [f"P{i:05d}" for i in range(n_p)]
    all_labels = d_labels + i_labels + p_labels
    n_total = len(all_labels)

    def voter_edges(global_idx: int, target_out: int) -> dict[str, float]:
        out = min(target_out, n_total - 1)
        size = min(out + 1, n_total)
        picks = rng.choice(n_total, size=size, replace=False)
        picks = picks[picks != global_idx][:out]
        weights = rng.dirichlet(np.ones(out))
        return {all_labels[p]: float(w) for p, w in zip(picks, weights, strict=True)}

    delegates = {d_labels[i]: voter_edges(i, delegate_out) for i in range(n_d)}
    intermediates = {
        i_labels[i]: voter_edges(n_d + i, intermediate_out) for i in range(n_i)
    }
    return delegates, intermediates, p_labels


def save_snapshot(
    path: str, delegates: dict, intermediates: dict, policies: list[str],
    consensus, influences, seed: int, sha: str,
) -> None:
    d_labels = list(delegates)
    i_labels = list(intermediates)
    all_labels = d_labels + i_labels + list(policies)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}

    row_ptr = [0]
    col_idx: list[int] = []
    weights: list[float] = []
    for edges in list(delegates.values()) + list(intermediates.values()):
        for target, w in edges.items():
            col_idx.append(label_to_idx[target])
            weights.append(w)
        row_ptr.append(len(col_idx))

    np.savez_compressed(
        path,
        delegate_labels=np.array(d_labels),
        intermediate_labels=np.array(i_labels),
        policy_labels=np.array(list(policies)),
        row_ptr=np.array(row_ptr, dtype=np.int32),
        col_idx=np.array(col_idx, dtype=np.int32),
        weights=np.array(weights, dtype=np.float64),
        consensus_labels=np.array([c.label for c in consensus]),
        consensus_values=np.array([c.value for c in consensus], dtype=np.float64),
        influence_labels=np.array([i.label for i in influences]),
        influence_roles=np.array([i.role for i in influences]),
        influence_values=np.array([i.value for i in influences], dtype=np.float64),
        seed=np.array(seed),
        ppv_git_sha=np.array(sha),
    )


def peak_rss_mb() -> float:
    # macOS: ru_maxrss is in bytes. Linux: KB. We target Darwin.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def load_existing_csv() -> tuple[dict[tuple, dict], list[dict]]:
    """Return ({(n_d, n_i, n_p, seed): row_for_our_VERSION}, all_rows_passthrough).

    Rows from other VERSIONs are passed through unchanged so this script doesn't
    clobber timings recorded by future implementations. Pre-versioning rows (no
    'version' column) are treated as belonging to VERSION."""
    by_key: dict[tuple, dict] = {}
    all_rows: list[dict] = []
    if not os.path.exists(CSV_PATH):
        return by_key, all_rows
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            row.setdefault("version", VERSION)  # migrate pre-versioning rows
            all_rows.append(row)
            if row["version"] == VERSION:
                key = (int(row["n_d"]), int(row["n_i"]),
                       int(row["n_p"]), int(row["seed"]))
                by_key[key] = row
    return by_key, all_rows


def write_manifest(sha: str) -> None:
    files = []
    for fname in sorted(os.listdir(SNAPSHOT_DIR)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(SNAPSHOT_DIR, fname)
        files.append({
            "name": fname,
            "sha256": sha256(fpath),
            "size": os.path.getsize(fpath),
        })
    manifest = {
        "release_tag": RELEASE_TAG,
        "base_url": RELEASE_BASE_URL,
        "ppv_git_sha": sha,
        "files": files,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    total_mb = sum(f["size"] for f in files) / (1024 * 1024)
    print(f"wrote {MANIFEST_PATH} ({len(files)} files, {total_mb:.1f} MB total)",
          flush=True)


def main() -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    sha = git_sha()

    if "--manifest-only" in sys.argv:
        write_manifest(sha)
        return

    existing_rows, passthrough = load_existing_csv()

    # Warm-up: prime numpy / dyld so the first real cell isn't penalized.
    _ = compute(*make_inputs(5, 2, 2, seed=999))

    # Carry forward rows belonging to other VERSIONs unchanged.
    rows: list[dict] = [r for r in passthrough if r["version"] != VERSION]

    print(f"{'n_d':>6} {'n_i':>6} {'n_p':>6} {'n_tot':>7} {'seed':>4} "
          f"{'wall_s':>10} {'rss_mb':>8}  status", flush=True)
    for n_d, n_i, n_p, seeds in CELLS:
        n_total = n_d + n_i + n_p
        for seed in seeds:
            key = (n_d, n_i, n_p, seed)
            snap_name = f"n{n_total}_seed{seed}.npz"
            snap_path = os.path.join(SNAPSHOT_DIR, snap_name)

            if os.path.exists(snap_path) and key in existing_rows:
                rows.append(existing_rows[key])
                print(f"{n_d:>6} {n_i:>6} {n_p:>6} {n_total:>7} {seed:>4} "
                      f"{existing_rows[key]['wall_seconds']:>10} "
                      f"{existing_rows[key]['peak_rss_mb']:>8}  cached",
                      flush=True)
                continue

            delegates, intermediates, policies = make_inputs(n_d, n_i, n_p, seed)
            t0 = time.perf_counter()
            consensus, influences = compute(delegates, intermediates, policies)
            wall = time.perf_counter() - t0
            rss = peak_rss_mb()
            # Only the canonical (baseline) snapshot is saved. Other versions
            # benchmark against the same inputs but must not overwrite reference
            # outputs — the gatekeeper test relies on baseline's snapshot bytes.
            if not os.path.exists(snap_path):
                save_snapshot(
                    snap_path, delegates, intermediates, policies,
                    consensus, influences, seed, sha,
                )
            print(f"{n_d:>6} {n_i:>6} {n_p:>6} {n_total:>7} {seed:>4} "
                  f"{wall:>10.3f} {rss:>8.1f}  computed",
                  flush=True)
            rows.append({
                "version": VERSION,
                "n_d": n_d, "n_i": n_i, "n_p": n_p, "n_total": n_total,
                "seed": seed, "wall_seconds": f"{wall:.4f}",
                "peak_rss_mb": f"{rss:.2f}", "ppv_git_sha": sha,
            })

    # Write CSV
    rows.sort(key=lambda r: (
        r["version"], int(r["n_total"]), int(r["seed"])
    ))
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "version", "n_d", "n_i", "n_p", "n_total", "seed",
            "wall_seconds", "peak_rss_mb", "ppv_git_sha",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {CSV_PATH} ({len(rows)} rows)", flush=True)

    write_manifest(sha)
    print(
        f"\nNext: gh release create {RELEASE_TAG} "
        f"{SNAPSHOT_DIR}/*.npz \\\n"
        f"        --title 'ppv benchmark snapshots {RELEASE_TAG}' \\\n"
        f"        --notes 'Generated from ppv git sha {sha}.'\n"
        f"Then commit tests/snapshots.manifest.json and benchmark.csv.",
        flush=True,
    )


if __name__ == "__main__":
    main()
