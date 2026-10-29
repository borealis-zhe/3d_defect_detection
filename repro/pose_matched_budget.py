"""Airplane reference coverage after pose alignment at the original point budget."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from .bundle_ablate import ROOT, records


AUDIT = ROOT / "outputs/pose_registration/front_view_audit.csv"
OUT = ROOT / "outputs/pose_registration/airplane_matched_budget.csv"


def main() -> None:
    with AUDIT.open(encoding="utf-8", newline="") as handle:
        audit = [r for r in csv.DictReader(handle)
                 if r["category"] == "airplane" and r["split"] == "test" and
                 r["is_anomaly"] == "1"]
    paths = records("airplane")
    index = {p.parent.name: i for i, p in enumerate(paths)}
    labels = {}
    for row in audit:
        path = ROOT / "outputs/full_surface/airplane" / f"sample_{index[row['sample']]:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            labels[row["sample"]] = data["labels"].astype(np.int8)
    selected = {row["sample"]: np.load(ROOT / "outputs/pose_registration/airplane/test" /
                                        row["sample"] / "aligned_front_ids.npy") for row in audit}
    if any(len(selected[r["sample"]]) < int(r["old_reference_points"]) for r in audit):
        raise ValueError("Matched point budget impossible for some airplane objects")
    results = []
    for trial in range(100):
        positive = 0
        no_positive = 0
        for row in audit:
            name = row["sample"]
            seed = int.from_bytes(hashlib.sha256(f"{trial}/{name}".encode()).digest()[:8], "little")
            chosen = np.random.default_rng(seed).choice(selected[name],
                                                    int(row["old_reference_points"]), replace=False)
            count = int(np.sum(labels[name][chosen] == 1))
            positive += count
            no_positive += count == 0
        results.append({"trial": trial, "positive_references": positive,
                        "zero_positive_anomaly_objects": no_positive})
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(f"Airplane 100 repeats: positives median={np.median([r['positive_references'] for r in results]):.0f}, "
          f"zero-positive objects median={np.median([r['zero_positive_anomaly_objects'] for r in results]):.0f}; "
          f"saved {OUT}")


if __name__ == "__main__":
    main()
