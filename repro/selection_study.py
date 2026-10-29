"""Label-blind point-domain selection ablation on frozen full-surface scores.

All budget-matched methods select exactly as many points per object as the
legacy front-112 domain. Labels never enter selection. The all-visible method
is included as the no-subsampling reference and has a much larger point count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, records
from .full_surface import normalized_raw
from .raw_coverage import RAW_ZIP, read_xyz


FULL = ROOT / "outputs" / "full_surface"
OUT = ROOT / "outputs" / "selection_study"
METHODS = ("front112", "random_visible", "voxel3d_visible",
           "six_side_visible", "all_visible")


def stable_rng(sample_id: str, salt: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{sample_id}|{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def exact_budget(candidates: np.ndarray, eligible: np.ndarray, budget: int,
                 rng: np.random.Generator) -> np.ndarray:
    candidates = np.unique(candidates)
    if len(candidates) >= budget:
        return np.sort(rng.choice(candidates, budget, replace=False))
    chosen = candidates
    remaining = np.setdiff1d(eligible, chosen, assume_unique=False)
    fill = rng.choice(remaining, budget-len(chosen), replace=False)
    return np.sort(np.r_[chosen, fill])


def random_visible(eligible: np.ndarray, budget: int, sample_id: str):
    return np.sort(stable_rng(sample_id, "random").choice(
        eligible, budget, replace=False))


def voxel_visible(xyz: np.ndarray, eligible: np.ndarray, budget: int,
                  sample_id: str):
    points = xyz[eligible]
    lo, span = points.min(0), np.maximum(np.ptp(points, axis=0), 1e-9)
    grid = max(4, int(np.ceil(np.sqrt(budget) * 1.25)))
    candidates = np.array([], dtype=np.int64)
    for _ in range(6):
        cell = np.floor((points-lo)/span*grid).astype(np.int64)
        cell = np.clip(cell, 0, grid-1)
        key = cell[:, 0] + grid*(cell[:, 1] + grid*cell[:, 2])
        _, first = np.unique(key, return_index=True)
        candidates = eligible[first]
        if len(candidates) >= budget:
            break
        grid = int(np.ceil(grid*1.35))
    return exact_budget(candidates, eligible, budget,
                        stable_rng(sample_id, "voxel"))


def six_side_visible(xyz: np.ndarray, eligible: np.ndarray, budget: int,
                     sample_id: str):
    points = xyz[eligible]
    lo, span = points.min(0), np.maximum(np.ptp(points, axis=0), 1e-9)
    grid = max(4, int(np.ceil(np.sqrt(budget/4))))
    candidates = np.array([], dtype=np.int64)
    for _ in range(7):
        all_candidates = []
        scaled = np.floor((points-lo)/span*grid).astype(np.int64)
        scaled = np.clip(scaled, 0, grid-1)
        for axis in range(3):
            other = [d for d in range(3) if d != axis]
            key = scaled[:, other[0]] + grid*scaled[:, other[1]]
            mins = np.full(grid*grid, np.inf)
            maxs = np.full(grid*grid, -np.inf)
            np.minimum.at(mins, key, points[:, axis])
            np.maximum.at(maxs, key, points[:, axis])
            extreme = ((points[:, axis] == mins[key]) |
                       (points[:, axis] == maxs[key]))
            all_candidates.append(eligible[np.flatnonzero(extreme)])
        candidates = np.unique(np.concatenate(all_candidates))
        if len(candidates) >= budget:
            break
        grid = int(np.ceil(grid*1.3))
    return exact_budget(candidates, eligible, budget,
                        stable_rng(sample_id, "six-side"))


def load_one(meta_path: Path, index: int, archive: ZipFile):
    category = meta_path.parts[-6]
    with np.load(FULL/category/f"sample_{index:04d}.npz",
                 allow_pickle=False) as d:
        full = {k: d[k].copy() for k in d.files}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = read_xyz(archive.read(meta["raw_member"]))
    xyz = normalized_raw(raw, meta)
    if len(xyz) != len(full["labels"]):
        raise ValueError(f"Raw/cache mismatch: {meta_path}")
    return meta, xyz, full


def infer(output_root: Path, archive_path: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        for category in CATEGORIES:
            paths = records(category)
            category_dir = output_root/category
            category_dir.mkdir(parents=True, exist_ok=True)
            for index, meta_path in enumerate(paths):
                destination = category_dir/f"sample_{index:04d}.npz"
                if destination.exists():
                    continue
                meta, xyz, full = load_one(meta_path, index, archive)
                labels = full["labels"]
                known = labels >= 0
                eligible = np.flatnonzero(full["seen"] & known)
                front = full["reference_ids"]
                front = front[known[front]]
                budget = len(front)
                if len(eligible) < budget:
                    raise ValueError(f"Visible domain below budget: {meta_path}")
                sample_id = str(full["sample_id"])
                selections = {
                    "front112": front,
                    "random_visible": random_visible(eligible, budget, sample_id),
                    "voxel3d_visible": voxel_visible(xyz, eligible, budget, sample_id),
                    "six_side_visible": six_side_visible(xyz, eligible, budget, sample_id),
                    "all_visible": eligible,
                }
                temp = destination.with_name(destination.stem+".partial.npz")
                np.savez_compressed(temp, sample_id=sample_id, budget=budget,
                                    **selections)
                temp.replace(destination)
                if (index+1) % 10 == 0 or index+1 == len(paths):
                    print(f"{category}: selected {index+1}/{len(paths)}", flush=True)


def evaluate(selection_root: Path, result_csv: Path, audit_csv: Path):
    metric_rows, audit_rows = [], []
    for category in CATEGORIES:
        buckets = {m: {"y": [], "s": [], "objects": 0,
                       "positive_objects": 0, "zero_positive": 0,
                       "selected_positive": 0, "raw_positive": 0,
                       "selected_points": 0} for m in METHODS}
        for index, meta_path in enumerate(records(category)):
            with np.load(FULL/category/f"sample_{index:04d}.npz",
                         allow_pickle=False) as full, np.load(
                             selection_root/category/f"sample_{index:04d}.npz",
                             allow_pickle=False) as selection:
                labels = full["labels"]
                scores = full["uniform_scores"]
                positive_object = "/good/" not in str(full["sample_id"])
                raw_positive = int((labels == 1).sum())
                row = {"sample_id": str(full["sample_id"]),
                       "category": category, "raw_points": len(labels),
                       "raw_positive_points": raw_positive,
                       "budget": int(selection["budget"])}
                for method in METHODS:
                    ids = selection[method]
                    y, s = labels[ids], scores[ids]
                    if np.any(y < 0):
                        raise ValueError("Unknown point entered selection")
                    positive = int((y == 1).sum())
                    b = buckets[method]
                    b["y"].append(y); b["s"].append(s)
                    b["objects"] += 1
                    b["positive_objects"] += int(positive_object)
                    b["zero_positive"] += int(positive_object and positive == 0)
                    b["selected_positive"] += positive
                    b["raw_positive"] += raw_positive
                    b["selected_points"] += len(ids)
                    row[f"{method}_points"] = len(ids)
                    row[f"{method}_positive"] = positive
                audit_rows.append(row)
        for method, b in buckets.items():
            y, s = np.concatenate(b["y"]), np.concatenate(b["s"])
            metric_rows.append({
                "category": category, "method": method,
                "point_auroc": 100*roc_auc_score(y, s),
                "point_ap": 100*average_precision_score(y, s),
                "points": len(y), "positive_points": int((y == 1).sum()),
                "positive_rate": 100*float(y.mean()),
                "positive_objects_zero_selected": b["zero_positive"],
                "positive_objects": b["positive_objects"],
                "raw_positive_coverage": (100*b["selected_positive"] /
                                           max(b["raw_positive"], 1)),
            })
        print(f"evaluated selections: {category}", flush=True)
    for method in METHODS:
        subset = [r for r in metric_rows if r["method"] == method]
        metric_rows.append({
            "category": "macro", "method": method,
            **{key: float(np.mean([r[key] for r in subset])) for key in
               ("point_auroc", "point_ap", "positive_rate",
                "raw_positive_coverage")},
            **{key: int(sum(r[key] for r in subset)) for key in
               ("points", "positive_points", "positive_objects_zero_selected",
                "positive_objects")},
        })
    for path, rows in ((result_csv, metric_rows), (audit_csv, audit_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps([r for r in metric_rows if r["category"] == "macro"],
                     indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("infer", "evaluate"))
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--selection-root", type=Path, default=OUT/"selections")
    parser.add_argument("--result-csv", type=Path,
                        default=OUT/"selection_metrics.csv")
    parser.add_argument("--audit-csv", type=Path,
                        default=OUT/"selection_audit.csv")
    args = parser.parse_args()
    if args.mode == "infer":
        infer(args.selection_root, args.archive)
    else:
        evaluate(args.selection_root, args.result_csv, args.audit_csv)


if __name__ == "__main__":
    main()
