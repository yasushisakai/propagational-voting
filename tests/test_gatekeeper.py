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


@pytest.mark.parametrize("entry_and_url", _params())
def test_gatekeeper(entry_and_url) -> None:
    entry, base_url = entry_and_url
    path = _ensure_snapshot(entry, base_url)
    snap = np.load(path, allow_pickle=False)
    delegates, intermediates, policies = _reconstruct_inputs(snap)
    consensus, influences = compute(delegates, intermediates, policies)

    assert [c.label for c in consensus] == [str(x) for x in snap["consensus_labels"]]
    np.testing.assert_allclose(
        [c.value for c in consensus], snap["consensus_values"],
        atol=1e-7, rtol=1e-6,
    )

    assert [i.label for i in influences] == [str(x) for x in snap["influence_labels"]]
    assert [i.role for i in influences] == [str(x) for x in snap["influence_roles"]]
    np.testing.assert_allclose(
        [i.value for i in influences], snap["influence_values"],
        atol=1e-7, rtol=1e-6,
    )


def test_manifest_present() -> None:
    """Guardrail: the manifest must exist and list at least one snapshot."""
    manifest = _load_manifest()
    assert manifest["files"], "manifest has no files"
    assert manifest["base_url"], "manifest missing base_url"
