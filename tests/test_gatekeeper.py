"""Gatekeeper: every snapshot listed in the manifest must reproduce its outputs.

Snapshots themselves are not committed — they're downloaded on demand from the
URL in tests/snapshots.manifest.json (hosted on GitHub Releases), cached in
tests/snapshots/, and verified by sha256 before use.

Cells where n_total >= SLOW_THRESHOLD are marked @pytest.mark.slow. Run with
`pytest --runslow` to include them; default `pytest` skips them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request

import numpy as np
import pytest

from ppv import compute

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "snapshots.manifest.json")
SNAPSHOT_DIR = os.path.join(HERE, "snapshots")

SLOW_THRESHOLD = 5000  # n_total >= this is marked slow


def _load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_snapshot(entry: dict, base_url: str) -> str:
    """Return local path, downloading and verifying if needed."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, entry["name"])
    if os.path.exists(path) and _sha256(path) == entry["sha256"]:
        return path
    url = f"{base_url}/{entry['name']}"
    tmp = path + ".part"
    print(f"\ndownloading {entry['name']} ({entry['size'] / 1024:.0f} KB) "
          f"from {url}")
    urllib.request.urlretrieve(url, tmp)
    actual = _sha256(tmp)
    if actual != entry["sha256"]:
        os.remove(tmp)
        raise RuntimeError(
            f"sha256 mismatch for {entry['name']}: "
            f"got {actual}, expected {entry['sha256']}"
        )
    os.replace(tmp, path)
    return path


def _n_total_from_name(name: str) -> int:
    m = re.match(r"n(\d+)_seed\d+\.npz$", name)
    if not m:
        raise ValueError(f"unrecognized snapshot filename: {name}")
    return int(m.group(1))


def _params():
    manifest = _load_manifest()
    base_url = manifest["base_url"]
    out = []
    for entry in manifest["files"]:
        n_total = _n_total_from_name(entry["name"])
        marks = [pytest.mark.slow] if n_total >= SLOW_THRESHOLD else []
        out.append(pytest.param((entry, base_url), marks=marks, id=entry["name"]))
    return out


def _reconstruct_inputs(snap) -> tuple[dict, dict, list[str]]:
    d_labels = [str(x) for x in snap["delegate_labels"]]
    i_labels = [str(x) for x in snap["intermediate_labels"]]
    p_labels = [str(x) for x in snap["policy_labels"]]
    all_labels = d_labels + i_labels + p_labels

    row_ptr = snap["row_ptr"]
    col_idx = snap["col_idx"]
    weights = snap["weights"]

    voter_labels = d_labels + i_labels
    delegates: dict[str, dict[str, float]] = {}
    intermediates: dict[str, dict[str, float]] = {}
    for v_idx, voter in enumerate(voter_labels):
        edges = {
            all_labels[col_idx[k]]: float(weights[k])
            for k in range(row_ptr[v_idx], row_ptr[v_idx + 1])
        }
        if v_idx < len(d_labels):
            delegates[voter] = edges
        else:
            intermediates[voter] = edges
    return delegates, intermediates, p_labels


# Gatekeeper tolerance is a per-implementation choice. fp32 squaring lands at
# ~1.5e-6 rtol; sparse-solve and fp64 squaring are essentially machine-precision.
# Loose enough to accept any reasonable-precision impl, strict enough to catch
# real algorithmic regressions. Per discussion: ranking preservation is the
# primary contract; value agreement within these tolerances is the secondary.
ATOL = 1e-3
RTOL = 1e-5


def _check_ranking_preserved(got: dict, ref: dict, atol: float, what: str) -> None:
    """Walk got's sorted-descending order. For each adjacent pair whose got values
    differ by more than 2*atol, the ref values must agree on the same ordering.
    Pairs whose values are within tolerance are treated as a tie cluster and may
    be in any order — fp32 commonly swaps these without changing meaning."""
    sorted_got = sorted(got.items(), key=lambda kv: -kv[1])
    for i in range(len(sorted_got) - 1):
        la, va = sorted_got[i]
        lb, vb = sorted_got[i + 1]
        if va - vb > 2 * atol:
            # got ranks la above lb with a meaningful gap. ref must agree.
            assert ref[la] + atol >= ref[lb], (
                f"{what} ranking flipped: got {la}({va:.6g}) > {lb}({vb:.6g}) "
                f"but ref {la}({ref[la]:.6g}) < {lb}({ref[lb]:.6g})"
            )


@pytest.mark.parametrize("entry_and_url", _params())
def test_gatekeeper(entry_and_url) -> None:
    entry, base_url = entry_and_url
    path = _ensure_snapshot(entry, base_url)
    snap = np.load(path, allow_pickle=False)
    delegates, intermediates, policies = _reconstruct_inputs(snap)
    consensus, influences = compute(delegates, intermediates, policies)

    # CONSENSUS: same label set, values within tolerance, ranking preserved on
    # meaningfully-different pairs.
    got_c = {c.label: c.value for c in consensus}
    want_c = dict(zip(
        (str(x) for x in snap["consensus_labels"]),
        snap["consensus_values"], strict=True,
    ))
    assert got_c.keys() == want_c.keys()
    np.testing.assert_allclose(
        [got_c[k] for k in want_c], list(want_c.values()),
        atol=ATOL, rtol=RTOL,
    )
    _check_ranking_preserved(got_c, want_c, atol=ATOL, what="consensus")

    # INFLUENCE: same label set + roles, values within tolerance, ranking
    # preserved on meaningfully-different pairs.
    got_i_full = {i.label: (i.role, i.value) for i in influences}
    want_i_full = dict(zip(
        (str(x) for x in snap["influence_labels"]),
        zip((str(r) for r in snap["influence_roles"]),
            snap["influence_values"], strict=True),
        strict=True,
    ))
    assert got_i_full.keys() == want_i_full.keys()
    for k in want_i_full:
        assert got_i_full[k][0] == want_i_full[k][0], f"role mismatch for {k}"
    got_i = {k: v[1] for k, v in got_i_full.items()}
    want_i = {k: v[1] for k, v in want_i_full.items()}
    np.testing.assert_allclose(
        [got_i[k] for k in want_i], list(want_i.values()),
        atol=ATOL, rtol=RTOL,
    )
    _check_ranking_preserved(got_i, want_i, atol=ATOL, what="influence")


def test_manifest_present() -> None:
    """Guardrail: the manifest must exist and list at least one snapshot."""
    manifest = _load_manifest()
    assert manifest["files"], "manifest has no files"
    assert manifest["base_url"], "manifest missing base_url"
