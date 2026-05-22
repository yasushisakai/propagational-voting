"""Run the current ppv implementation against each snapshot and report error
magnitudes vs the fp64 reference outputs.

Designed for the squaring_fp32 branch: the gatekeeper will fail strict tolerance,
so this script measures HOW MUCH it diverges and whether rankings are preserved.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ppv import compute
from tests.test_gatekeeper import _ensure_snapshot, _load_manifest, _reconstruct_inputs

HERE = os.path.dirname(os.path.abspath(__file__))


def kendall_swaps(a: list, b: list) -> int:
    """Count adjacent swaps to turn list a's order into list b's order (inversions
    in the permutation mapping a -> b). Bigger = ranking changed more."""
    pos = {x: i for i, x in enumerate(b)}
    perm = [pos[x] for x in a]
    swaps = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                swaps += 1
    return swaps


def main():
    manifest = _load_manifest()
    base_url = manifest["base_url"]

    header = (
        f"{'snapshot':>22} | {'c_atol':>9} {'c_rtol':>9} | "
        f"{'i_atol':>9} {'i_rtol':>9} | {'c_swaps':>7} {'i_swaps':>7}"
    )
    print(header)
    print('-' * len(header))

    for entry in manifest["files"]:
        path = _ensure_snapshot(entry, base_url)
        snap = np.load(path, allow_pickle=False)
        delegates, intermediates, policies = _reconstruct_inputs(snap)

        # Reference (fp64 from squaring snapshots-v2)
        ref_c = dict(zip(
            (str(x) for x in snap["consensus_labels"]),
            snap["consensus_values"], strict=True,
        ))
        ref_i = dict(zip(
            (str(x) for x in snap["influence_labels"]),
            snap["influence_values"], strict=True,
        ))

        # fp32 outputs
        consensus, influences = compute(delegates, intermediates, policies)
        got_c = {c.label: c.value for c in consensus}
        got_i = {i.label: i.value for i in influences}

        c_keys = list(ref_c.keys())
        c_diff = np.array([got_c[k] - ref_c[k] for k in c_keys])
        c_ref = np.array([ref_c[k] for k in c_keys])
        c_atol = float(np.abs(c_diff).max())
        c_rtol = float((np.abs(c_diff) / np.maximum(np.abs(c_ref), 1e-30)).max())

        i_keys = list(ref_i.keys())
        i_diff = np.array([got_i[k] - ref_i[k] for k in i_keys])
        i_ref = np.array([ref_i[k] for k in i_keys])
        i_atol = float(np.abs(i_diff).max())
        i_rtol = float((np.abs(i_diff) / np.maximum(np.abs(i_ref), 1e-30)).max())

        # Ranking divergence: how many adjacent swaps to align our ranking
        # with the snapshot's? Capped on big lists to avoid O(n²) blowup.
        c_ranked_ref = [k for k, _ in sorted(ref_c.items(),
                                             key=lambda kv: -kv[1])]
        c_ranked_got = [c.label for c in consensus]
        c_swaps = kendall_swaps(c_ranked_got, c_ranked_ref)

        if len(i_keys) <= 5000:
            i_ranked_ref = [k for k, _ in sorted(ref_i.items(),
                                                 key=lambda kv: -kv[1])]
            i_ranked_got = [i.label for i in influences]
            i_swaps = kendall_swaps(i_ranked_got, i_ranked_ref)
            i_swaps_str = f"{i_swaps:>7}"
        else:
            i_swaps_str = f"{'skip':>7}"  # O(n²) too slow on big lists

        print(
            f"{entry['name']:>22} | "
            f"{c_atol:>9.2e} {c_rtol:>9.2e} | "
            f"{i_atol:>9.2e} {i_rtol:>9.2e} | "
            f"{c_swaps:>7} {i_swaps_str}"
        )


if __name__ == "__main__":
    main()
