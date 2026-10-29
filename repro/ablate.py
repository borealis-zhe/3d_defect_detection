"""Independent post-training ablations for multi-view evidence.

Primary comparison uses a common Dinomaly checkpoint, point domain, labels,
and object aggregation. Unlike the companion's learned cross-region and local
geometry heads, these ablations test cross-view consensus and disagreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .geometry import inference_view_indices
from .preprocess import CATEGORIES


def fused_scores(view_scores: np.ndarray, visible: np.ndarray) -> dict[str, np.ndarray]:
    if view_scores.ndim != 2 or visible.shape != view_scores.shape:
        raise ValueError("Expected matching [views, reference points] arrays")
    uniform = view_scores.mean(axis=0)
    count = visible.sum(axis=0)
    safe_count = np.maximum(count, 1)
    visible_mean = np.where(count > 0, (view_scores * visible).sum(axis=0) /
                            safe_count, uniform)
    # Explicit fallback avoids treating an unseen point as a normal point.
    ordered = np.sort(np.where(visible, view_scores, np.inf), axis=0)
    middle_low = np.maximum((count - 1) // 2, 0)
    middle_high = count // 2
    columns = np.arange(len(count))
    median = 0.5 * (ordered[middle_low, columns] +
                    ordered[middle_high, columns])
    median = np.where(count > 0, median, uniform)
    residual = np.where(visible, view_scores - visible_mean, 0.0)
    disagreement = np.sqrt((residual ** 2).sum(axis=0) / safe_count)
    disagreement = np.where(count >= 2, disagreement, 0.0)
    out = {"baseline_uniform": uniform,
           "visible_mean_control": visible_mean,
           "view_consensus": median,
           "view_disagreement": visible_mean + 0.5 * disagreement,
           "consensus_plus_disagreement": median + 0.5 * disagreement}
    return {name: values.astype(np.float32) for name, values in out.items()}


def budget_scores(view_scores: np.ndarray, view_ids: np.ndarray) -> dict[str, np.ndarray]:
    ordered_ids = inference_view_indices()
    id_to_row = {int(view_id): row for row, view_id in enumerate(view_ids)}
    out = {}
    for budget in (3, 5, 9, 15):
        selected = [id_to_row[view_id] for view_id in ordered_ids[:budget]]
        out[f"views_{budget}"] = view_scores[selected].mean(axis=0).astype(np.float32)
    return out


def evaluate_category(prediction_dir: Path, category: str) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    files = sorted((prediction_dir / category / "predictions").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No predictions for {category}")
    records: dict[str, dict[str, list]] = {}
    for file in files:
        data = np.load(file)
        scores = fused_scores(data["view_scores"], data["visible"])
        scores.update(budget_scores(data["view_scores"], data["view_ids"]))
        labels = data["labels"]
        known = labels >= 0
        object_label = int(data["object_label"])
        for name, values in scores.items():
            rec = records.setdefault(name, {"object_y": [], "object_s": [],
                                            "point_y": [], "point_s": []})
            rec["object_y"].append(object_label)
            rec["object_s"].append(float(np.max(values)))
            rec["point_y"].append((labels[known] == 1).astype(np.uint8))
            rec["point_s"].append(values[known])
    rows = []
    for name, rec in records.items():
        point_y = np.concatenate(rec["point_y"])
        point_s = np.concatenate(rec["point_s"])
        if len(set(rec["object_y"])) != 2 or len(np.unique(point_y)) != 2:
            raise ValueError(f"Missing positive or negative labels in {category}")
        rows.append({"category": category, "method": name,
                     "object_auroc": 100 * roc_auc_score(rec["object_y"], rec["object_s"]),
                     "point_auroc": 100 * roc_auc_score(point_y, point_s),
                     "point_ap": 100 * average_precision_score(point_y, point_s),
                     "objects": len(files), "known_points": len(point_y),
                     "positive_points": int(point_y.sum())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for category in args.categories
            for row in evaluate_category(args.processed, category)]
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        rows.append({"category": "macro", "method": method,
                     "object_auroc": np.mean([r["object_auroc"] for r in subset]),
                     "point_auroc": np.mean([r["point_auroc"] for r in subset]),
                     "point_ap": np.mean([r["point_ap"] for r in subset]),
                     "objects": sum(r["objects"] for r in subset),
                     "known_points": sum(r["known_points"] for r in subset),
                     "positive_points": sum(r["positive_points"] for r in subset)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([r for r in rows if r["category"] == "macro"], indent=2),
          flush=True)


if __name__ == "__main__":
    main()
