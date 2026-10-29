"""Audit processed data counts and the restricted reference domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preprocess import CATEGORIES


def audit(processed: Path, categories: list[str]) -> dict:
    per_category = []
    for category in categories:
        path = processed / category / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        tests = manifest["test"]
        anomalies = [item for item in tests if item["object_label"] == 1]
        per_category.append({
            "category": category,
            "train_objects": len(manifest["train"]),
            "test_objects": len(tests),
            "anomalous_test_objects": len(anomalies),
            "raw_test_points": sum(item["raw_points"] for item in tests),
            "reference_test_points": sum(item["reference_points"] for item in tests),
            "anomalies_without_positive_reference": sum(
                item["positive_reference_points"] == 0 for item in anomalies),
        })
    totals = {key: sum(item[key] for item in per_category)
              for key in per_category[0] if key != "category"}
    return {"per_category": per_category, "total": totals,
            "note": "Reference-domain counts depend on this independent renderer and 64-pixel front grid."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.processed, args.categories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["total"], indent=2))


if __name__ == "__main__":
    main()
