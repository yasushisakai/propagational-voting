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


if __name__ == "__main__":
    consensus, influences = ppv.compute(delegates, intermediates, policies)

    print("Consensus (policies, descending):")
    for c in consensus:
        print(f"  {c.label:8s} {c.value:.4f}")

    print("\nInfluence (delegates + intermediates, descending):")
    for i in influences:
        print(f"  {i.label:8s} {i.role:12s} {i.value:.4f}")
