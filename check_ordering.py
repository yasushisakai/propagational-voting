"""Hard ordering gate for the cupy_fp16 branch.

For each fp64 snapshot, compute the consensus with the current ppv.compute
and verify that the consensus ordering matches the reference ordering — with
near-ties allowed to swap. A "near-tie" is when two adjacent consensus values
in the reference are within `atol + rtol * max(|values|)` of each other.

Exits non-zero if any snapshot has a non-tie inversion. Used as a precondition
before running bootstrap_bench on this branch.
"""

from __future__ import annotations

import sys

import numpy as np

from ppv import compute
from tests.test_gatekeeper import _ensure_snapshot, _load_manifest, _reconstruct_inputs

ATOL = 1e-3
RTOL = 1e-3


def main() -> int:
    manifest = _load_manifest()
    base_url = manifest["base_url"]

    header = f"{'snapshot':>22} | {'inversions':>10} {'non-tie':>8} | first non-tie"
    print(header)
    print("-" * len(header))

    failed = []
    for entry in manifest["files"]:
        path = _ensure_snapshot(entry, base_url)
        snap = np.load(path, allow_pickle=False)
        delegates, intermediates, policies = _reconstruct_inputs(snap)

        ref_labels = [str(x) for x in snap["consensus_labels"]]
        ref_values = np.asarray(snap["consensus_values"], dtype=np.float64)
        ref_index = {label: idx for idx, label in enumerate(ref_labels)}

        consensus, _ = compute(delegates, intermediates, policies)
        got_labels = [c.label for c in consensus]

        # Walk our ordering. For each adjacent pair (a, b), look up their
        # positions in the reference ordering. If b is ranked higher than a
        # in the reference (an inversion), demand the two ref values be
        # within tie tolerance — otherwise this is a real, non-tie swap.
        inversions = 0
        non_tie = 0
        first_non_tie = None
        for a, b in zip(got_labels, got_labels[1:]):
            ra, rb = ref_index[a], ref_index[b]
            if ra <= rb:  # consistent with reference order
                continue
            inversions += 1
            va, vb = ref_values[ra], ref_values[rb]
            tie = abs(va - vb) <= ATOL + RTOL * max(abs(va), abs(vb))
            if not tie:
                non_tie += 1
                if first_non_tie is None:
                    first_non_tie = (a, b, float(va), float(vb))

        marker = "" if non_tie == 0 else "  FAIL"
        ft = ""
        if first_non_tie is not None:
            a, b, va, vb = first_non_tie
            ft = f"{a}({va:.3f}) > {b}({vb:.3f})"
        print(f"{entry['name']:>22} | {inversions:>10} {non_tie:>8} | {ft}{marker}")

        if non_tie > 0:
            failed.append(entry["name"])

    if failed:
        print(f"\nFAIL: {len(failed)} snapshot(s) had non-tie inversions: {failed}",
              file=sys.stderr)
        return 1
    print("\nOK: all snapshots passed the hard ordering rule "
          f"(atol={ATOL}, rtol={RTOL}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
