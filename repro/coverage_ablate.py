"""Coverage-based view selection with a frozen Dinomaly-MV checkpoint.

Coverage here means the fraction of the *existing front reference points*
visible from at least one selected view. It is not full raw-cloud coverage.
View selection uses visibility only; test anomaly labels are read later for
evaluation and never influence the chosen views.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, VIEW_IDS, load_runtime, records


ALL_VIEWS = tuple(range(27))
BUDGETS = (3, 5, 9, 15)
METHODS = ("fixed_prefix", "fixed_spread", "normal_greedy_15",
           "normal_greedy_27", "object_greedy_27")
# The source renderer enumerates xyz Euler angles in this product order.
# Only x/y change the viewing direction; z rotates the image in plane.
VIEW_DIRECTIONS = np.asarray(list(product((0, -15, 15), repeat=3)),
                             dtype=np.float32)[:, :2]


def greedy_coverage(visibility: np.ndarray, budget: int,
                    candidates: tuple[int, ...] = ALL_VIEWS) -> tuple[int, ...]:
    """Start with the front view, then maximize newly visible reference IDs."""
    if visibility.ndim != 2 or visibility.shape[0] != 27:
        raise ValueError("Expected 27 by N visibility matrix")
    if 0 not in candidates or len(set(candidates)) != len(candidates):
        raise ValueError("Candidate views must include the unique front view")
    if not 1 <= budget <= len(candidates):
        raise ValueError("Invalid view budget")
    chosen = [0]
    covered = visibility[0].copy()
    while len(chosen) < budget:
        gains = np.sum(visibility & ~covered[None, :], axis=1)
        novelty = np.min(np.sum((VIEW_DIRECTIONS[:, None, :] -
                                 VIEW_DIRECTIONS[np.asarray(chosen)][None, :, :]) ** 2,
                                axis=2), axis=1)
        options = [v for v in candidates if v not in chosen]
        # The first criterion is coverage. Angular novelty prevents arbitrary
        # near-duplicate choices once all reference points are covered.
        nxt = max(options, key=lambda v: (int(gains[v]), float(novelty[v]), -v))
        chosen.append(nxt)
        covered |= visibility[nxt]
    return tuple(chosen)


def normal_training_selection(category: str,
                              candidates: tuple[int, ...] = ALL_VIEWS) -> tuple[int, ...]:
    base = HANDOFF / "data" / "pcd_self_v1" / category / "train"
    files = sorted(base.glob("good/xyz/*/points.npz"))
    if len(files) != 4:
        raise ValueError(f"Expected four normal prototypes: {category}")
    arrays = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            arrays.append(data["visibility"])
    return greedy_coverage(np.concatenate(arrays, axis=1), 15, candidates)


def selected(visibility: np.ndarray, category_order_15: tuple[int, ...],
             category_order_27: tuple[int, ...],
             budget: int) -> dict[str, tuple[int, ...]]:
    spread = tuple(VIEW_IDS[int(i)] for i in
                   np.rint(np.linspace(0, len(VIEW_IDS) - 1, budget)).astype(int))
    return {
        "fixed_prefix": tuple(VIEW_IDS[:budget]),
        "fixed_spread": spread,
        "normal_greedy_15": category_order_15[:budget],
        "normal_greedy_27": category_order_27[:budget],
        "object_greedy_27": greedy_coverage(visibility, budget),
    }


def coverage_stats() -> list[dict]:
    rows = []
    for category in CATEGORIES:
        order_15 = normal_training_selection(category, VIEW_IDS)
        order_27 = normal_training_selection(category)
        for meta_path in records(category):
            folder = meta_path.parent
            with np.load(folder / "points.npz", allow_pickle=False) as point:
                visible = point["visibility"]
                labels = point["labels"].astype(np.int8)
                if "labels_valid" in point:
                    labels[~point["labels_valid"]] = -1
            for budget in BUDGETS:
                for method, ids in selected(visible, order_15, order_27,
                                            budget).items():
                    union = visible[list(ids)].any(axis=0)
                    positive = labels == 1
                    rows.append({"category": category, "sample": folder.name,
                                 "defect": folder.parents[1].name,
                                 "method": method, "budget": budget,
                                 "view_ids": ",".join(map(str, ids)),
                                 "reference_points": len(labels),
                                 "covered_points": int(union.sum()),
                                 "positive_points": int(positive.sum()),
                                 "covered_positive_points": int((union & positive).sum()),
                                 "mean_visible_views": float(visible[list(ids)].sum(axis=0).mean())})
    return rows


def write_coverage(path: Path) -> None:
    rows = coverage_stats()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} label-blind selections to {path}", flush=True)


def infer_all(category: str, output_root: Path, batch_size: int = 2) -> None:
    import torch
    import torch.nn.functional as F

    rt = load_runtime()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = rt.build_model(device=device)
    ckpt = torch.load(HANDOFF / "results" / "dinomaly_mv_v3" / "train" /
                      category / "final.pt", map_location="cpu", weights_only=True)
    if ckpt["category"] != category or ckpt["step"] != 5000:
        raise ValueError("Unexpected checkpoint")
    model.bottleneck.load_state_dict(ckpt["bottleneck"], strict=True)
    model.decoder.load_state_dict(ckpt["decoder"], strict=True)
    model.eval()
    out = output_root / category
    out.mkdir(parents=True, exist_ok=True)
    paths = records(category)
    for index, meta_path in enumerate(paths):
        output = out / f"sample_{index:04d}.npz"
        if output.exists():
            with np.load(output, allow_pickle=False) as existing:
                if existing["view_scores"].shape[0] != 27:
                    raise ValueError(f"Incomplete cache: {output}")
            continue
        folder = meta_path.parent
        with np.load(folder / "points.npz", allow_pickle=False) as point:
            visible = point["visibility"]
            labels = point["labels"].astype(np.int8)
            if "labels_valid" in point:
                labels[~point["labels_valid"]] = -1
            raw_ids = point["raw_point_ids"]
        n_points = len(labels)
        all_scores = []
        with torch.inference_mode():
            for first in range(0, 27, batch_size):
                chunk = ALL_VIEWS[first:first + batch_size]
                images = torch.stack([rt.image_tensor(folder / f"view_{v:03d}.png")
                                      for v in chunk]).to(device)
                targets, predictions = model(images)
                maps = rt.score_maps(targets, predictions)
                positions = torch.as_tensor(np.stack([
                    np.load(folder / f"view_{v:03d}.npy", allow_pickle=False)
                    for v in chunk]), device=device, dtype=torch.float32)
                if positions.shape != (len(chunk), 2, n_points):
                    raise ValueError(f"Unexpected projection shape: {folder}")
                pixels = (positions + 0.5) * 2 - 0.5 - 28
                if bool(((pixels < 0) | (pixels > 391)).any()):
                    raise ValueError(f"Projection outside crop: {folder}")
                grid = ((pixels + 0.5) / 392 * 2 - 1).permute(0, 2, 1)[:, None]
                dense = F.interpolate(maps[:, None], size=(392, 392),
                                      mode="bilinear", align_corners=False)
                values = F.grid_sample(dense, grid, mode="bilinear",
                                       padding_mode="border",
                                       align_corners=False)[:, 0, 0]
                all_scores.append(values.cpu().numpy())
        scores = np.concatenate(all_scores).astype(np.float32)
        saved_path = (HANDOFF / "results" / "dinomaly_mv_v3" / "eval" /
                      category / f"sample_{index:04d}.npz")
        with np.load(saved_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["raw_point_ids"], raw_ids):
                raise ValueError(f"Raw IDs differ: {folder}")
            if not np.array_equal(saved["labels"], labels):
                raise ValueError(f"Labels differ: {folder}")
            error = float(np.max(np.abs(scores[list(VIEW_IDS)].mean(0) -
                                        saved["scores"])))
            if error > 2e-6:
                raise ValueError(f"Fixed-15 baseline mismatch {error}: {folder}")
            sid = str(saved["sample_id"])
        temp = output.with_name(output.stem + ".partial.npz")
        np.savez_compressed(temp, sample_id=sid, view_scores=scores,
                            visibility=visible, labels=labels, raw_point_ids=raw_ids,
                            baseline_max_abs_error=error)
        temp.replace(output)
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"{category}: {index+1}/{len(paths)} all-view scores; "
                  f"baseline error {error:.2g}", flush=True)


def evaluate(output_root: Path, coverage_path: Path, metrics_path: Path) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    coverage = {}
    with coverage_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            coverage[(row["category"], row["sample"], row["method"],
                      int(row["budget"]))] = row
    rows = []
    for category in CATEGORIES:
        files = sorted((output_root / category).glob("sample_*.npz"))
        if len(files) != (53 if category == "chicken" else 50):
            raise ValueError(f"Missing all-view predictions: {category}")
        bucket = {}
        for file in files:
            with np.load(file, allow_pickle=False) as data:
                sid = str(data["sample_id"])
                name = sid.split("/")[-1]
                labels = data["labels"]
                known = labels >= 0
                oy = int(sid.split("/")[2] != "good")
                for budget in BUDGETS:
                    for method in METHODS:
                        cr = coverage[(category, name, method, budget)]
                        ids = [int(v) for v in cr["view_ids"].split(",")]
                        scores = data["view_scores"][ids].mean(axis=0)
                        key = (method, budget)
                        record = bucket.setdefault(key, {"oy": [], "os": [],
                                                         "py": [], "ps": [],
                                                         "covered": 0, "total": 0,
                                                         "covered_positive": 0,
                                                         "positive": 0})
                        record["oy"].append(oy)
                        record["os"].append(float(scores.max()))
                        record["py"].append(labels[known])
                        record["ps"].append(scores[known])
                        record["covered"] += int(cr["covered_points"])
                        record["total"] += int(cr["reference_points"])
                        record["covered_positive"] += int(cr["covered_positive_points"])
                        record["positive"] += int(cr["positive_points"])
        for (method, budget), rec in sorted(bucket.items()):
            py = np.concatenate(rec["py"])
            ps = np.concatenate(rec["ps"])
            rows.append({"category": category, "method": method,
                         "budget": budget,
                         "object_auroc": 100 * roc_auc_score(rec["oy"], rec["os"]),
                         "point_auroc": 100 * roc_auc_score(py, ps),
                         "point_ap": 100 * average_precision_score(py, ps),
                         "coverage_percent": 100 * rec["covered"] / rec["total"],
                         "positive_coverage_percent": 100 * rec["covered_positive"] /
                         rec["positive"] if rec["positive"] else float("nan"),
                         "objects": len(files), "points": len(py)})
        print(f"evaluated {category}", flush=True)
    for method in METHODS:
        for budget in BUDGETS:
            subset = [r for r in rows if r["method"] == method and r["budget"] == budget]
            rows.append({"category": "macro", "method": method,
                         "budget": budget,
                         **{key: float(np.mean([r[key] for r in subset])) for key in
                            ("object_auroc", "point_auroc", "point_ap",
                             "coverage_percent", "positive_coverage_percent")},
                         "objects": sum(r["objects"] for r in subset),
                         "points": sum(r["points"] for r in subset)})
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([r for r in rows if r["category"] == "macro"],
                     indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("coverage", "infer", "evaluate"))
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / "coverage_all_views")
    parser.add_argument("--coverage-csv", type=Path,
                        default=ROOT / "outputs" / "coverage_selection.csv")
    parser.add_argument("--metrics-csv", type=Path,
                        default=ROOT / "outputs" / "coverage_metrics.csv")
    args = parser.parse_args()
    if args.mode == "coverage":
        write_coverage(args.coverage_csv)
    elif args.mode == "infer":
        for category in args.categories:
            infer_all(category, args.output_root, args.batch_size)
    else:
        evaluate(args.output_root, args.coverage_csv, args.metrics_csv)


if __name__ == "__main__":
    main()
