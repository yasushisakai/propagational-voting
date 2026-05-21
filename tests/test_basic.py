import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ppv

delegates = {
    "Alice": {"Bob": 0.2, "ProD": 0.1, "FAR2": 0.7},
    "Bob": {"Alice": 0.2, "Daniel": 0.1, "FAR4": 0.7},
    "Charlie": {"NA": 0.7, "ProD": 0.3},
    "Daniel": {"FAR6": 1.0},
}
intermediates = {
    "NA": {"Alice": 0.5, "Bob": 0.5},
    "ProD": {"FAR6": 0.5, "FAR4": 0.5},
}
policies = ["FAR6", "FAR4", "FAR2"]


def test_small():
    consensus, influences = ppv.compute(delegates, intermediates, policies)

    assert len(consensus) == len(policies)
    assert len(influences) == len(delegates) + len(intermediates)

    assert {c.label for c in consensus} == {"FAR6", "FAR4", "FAR2"}
    assert {i.label for i in influences} == {
        "Alice",
        "Bob",
        "Charlie",
        "Daniel",
        "NA",
        "ProD",
    }

    assert [c.value for c in consensus] == sorted(
        (c.value for c in consensus), reverse=True
    )
    assert [i.value for i in influences] == sorted(
        (i.value for i in influences), reverse=True
    )
    assert all(c.value >= 0 for c in consensus)

    role_by_label = {i.label: i.role for i in influences}
    assert role_by_label["Alice"] == "delegate"
    assert role_by_label["Daniel"] == "delegate"
    assert role_by_label["NA"] == "intermediate"
    assert role_by_label["ProD"] == "intermediate"
