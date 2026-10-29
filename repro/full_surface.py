"""Expand Dinomaly-MV scoring from front references to visible raw PCD points.

The score map is the frozen baseline's original 15-view cosine residual.
Raw-point visibility is reconstructed geometrically from the saved cameras;
it approximates, rather than exactly reproduces, Open3D's point-sprite depth.
Only visible, annotated raw points enter point-level metrics. Unseen points
remain explicitly unscored. No anomaly labels are used during inference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, VIEW_IDS, load_runtime, records
from .raw_coverage import IMAGE_SIZE, RAW_ZIP, read_xyz


def matched_labels(archive: ZipFile, member: str, raw: np.ndarray,
                   defect: str) -> np.ndarray:
    if defect == "good":
        return np.zeros(len(raw), dtype=np.int8)
    gt_name = member.replace("/test/", "/gt/")[:-4] + ".txt"
    gt = np.loadtxt(io.BytesIO(archive.read(gt_name)), dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] != 4 or not np.isin(gt[:, 3], [0, 1]).all():
        raise ValueError(f"Invalid GT: {gt_name}")
    # Coordinate order differs from PCD order. Require a unique match;
    # never silently transfer labels by row position.
    unique, inverse = np.unique(gt[:, :3], axis=0, return_inverse=True)
    lower = np.ones(len(unique), dtype=np.int8)
    upper = np.zeros(len(unique), dtype=np.int8)
    np.minimum.at(lower, inverse, gt[:, 3].astype(np.int8))
    np.maximum.at(upper, inverse, gt[:, 3].astype(np.int8))
    if np.any(lower != upper):
        raise ValueError(f"Conflicting duplicate labels: {gt_name}")
    tree = cKDTree(unique)
    distances, indices = tree.query(raw, k=2, workers=-1)
    if np.any(distances[:, 1] <= 1e-7):
        raise ValueError(f"Ambiguous GT coordinate in {gt_name}")
    known = distances[:, 0] <= 1e-7
    labels = np.full(len(raw), -1, dtype=np.int8)
    labels[known] = lower[indices[known, 0]]
    return labels


def normalized_raw(raw: np.ndarray, sample: dict) -> np.ndarray:
    return ((raw.astype(np.float64) @ np.asarray(sample["basis"], np.float64) -
             np.asarray(sample["oriented_center"], np.float64)) /
            float(sample["normalization_radius"]))


def project_visible(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(camera["rotation"], np.float64)
    extrinsic = np.asarray(camera["extrinsic"], np.float64)
    intrinsic = np.asarray(camera["intrinsic_672"], np.float64)
    cam = (points @ rotation.T) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = cam[:, 2]
    projected = cam @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    pixel = np.rint(uv).astype(np.int32)
    in_camera = ((depth > 0) & (pixel[:, 0] >= 0) &
                 (pixel[:, 0] < IMAGE_SIZE) & (pixel[:, 1] >= 0) &
                 (pixel[:, 1] < IMAGE_SIZE))
    ids = np.flatnonzero(in_camera)
    flat = pixel[ids, 1] * IMAGE_SIZE + pixel[ids, 0]
    depth_buffer = np.full(IMAGE_SIZE * IMAGE_SIZE, np.inf, dtype=np.float32)
    np.minimum.at(depth_buffer, flat, depth[ids].astype(np.float32))
    front = minimum_filter(depth_buffer.reshape(IMAGE_SIZE, IMAGE_SIZE),
                           size=3, mode="constant", cval=np.inf).ravel()
    visible = np.zeros(len(points), dtype=bool)
    visible[ids] = np.isfinite(front[flat]) & (
        np.abs(depth[ids] - front[flat]) <= 0.025)
    position = np.rint((uv + 0.5) / 3 - 0.5).astype(np.float32)
    # The image backbone sees only the 392-pixel center crop after 2x resize.
    crop = ((position[:, 0] >= 14) & (position[:, 0] <= 209) &
            (position[:, 1] >= 14) & (position[:, 1] <= 209))
    visible &= crop
    return position, visible


def score_at_raw_positions(score_map, positions, torch, F):
    pixels = (torch.as_tensor(positions, device=score_map.device,
                              dtype=torch.float32) + 0.5) * 2 - 0.5 - 28
    grid = ((pixels + 0.5) / 392 * 2 - 1)[None, None]
    dense = F.interpolate(score_map[None, None], size=(392, 392),
                          mode="bilinear", align_corners=False)
    return F.grid_sample(dense, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)[0, 0, 0]


def infer_category(category: str, output_root: Path, archive: ZipFile,
                   limit: int = 0) -> None:
    import torch
    import torch.nn.functional as F

    rt = load_runtime()
    model = rt.build_model(device="cuda:0" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    ckpt = torch.load(HANDOFF / "results" / "dinomaly_mv_v3" / "train" /
                      category / "final.pt", map_location="cpu", weights_only=True)
    if ckpt["category"] != category or ckpt["step"] != 5000:
        raise ValueError("Wrong baseline checkpoint")
    model.bottleneck.load_state_dict(ckpt["bottleneck"], strict=True)
    model.decoder.load_state_dict(ckpt["decoder"], strict=True)
    model.eval()
    out = output_root / category
    out.mkdir(parents=True, exist_ok=True)
    paths = records(category)
    if limit:
        paths = paths[:limit]
    for index, meta_path in enumerate(paths):
        destination = out / f"sample_{index:04d}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                if "uniform_scores" not in cached.files:
                    raise ValueError(f"Incomplete cached sample: {destination}")
            continue
        begin = time.perf_counter()
        folder = meta_path.parent
        sample = json.loads(meta_path.read_text(encoding="utf-8"))
        cameras = json.loads((folder / "cameras.json").read_text(
            encoding="utf-8"))["cameras"]
        raw = read_xyz(archive.read(sample["raw_member"]))
        if len(raw) != sample["raw_points"] or len(cameras) != 27:
            raise ValueError(f"Raw source mismatch: {folder}")
        labels = matched_labels(archive, sample["raw_member"], raw,
                                sample["defect"])
        if int((labels == 1).sum()) != sample["raw_positive_points"]:
            raise ValueError(f"Positive label count mismatch: {folder}")
        points = normalized_raw(raw, sample)
        with np.load(folder / "points.npz", allow_pickle=False) as point:
            ref_ids = point["raw_point_ids"]
            expected_labels = point["labels"].astype(np.int8)
            if "labels_valid" in point:
                expected_labels[~point["labels_valid"]] = -1
            if not np.array_equal(labels[ref_ids], expected_labels):
                raise ValueError(f"Reference label mapping mismatch: {folder}")
            if np.max(np.abs(points[ref_ids] - point["xyz"])) > 1e-6:
                raise ValueError(f"Reference point coordinates mismatch: {folder}")
        score_sum = np.zeros(len(raw), dtype=np.float64)
        visible_sum = np.zeros(len(raw), dtype=np.float64)
        visible_count = np.zeros(len(raw), dtype=np.uint8)
        with torch.inference_mode():
            for view_id in VIEW_IDS:
                position, visible = project_visible(points, cameras[view_id])
                image = rt.image_tensor(folder / f"view_{view_id:03d}.png")
                targets, predictions = model(image[None].to(device))
                residual = rt.score_maps(targets, predictions)[0]
                values = score_at_raw_positions(residual, position, torch, F)
                values = values.cpu().numpy().astype(np.float32)
                score_sum += values
                visible_sum[visible] += values[visible]
                visible_count += visible.astype(np.uint8)
        uniform = (score_sum / len(VIEW_IDS)).astype(np.float32)
        seen = visible_count > 0
        visible_mean = np.where(seen, visible_sum / np.maximum(visible_count, 1),
                                uniform).astype(np.float32)
        saved_path = (HANDOFF / "results" / "dinomaly_mv_v3" / "eval" /
                      category / f"sample_{index:04d}.npz")
        with np.load(saved_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["raw_point_ids"], ref_ids):
                raise ValueError(f"Saved raw IDs mismatch: {folder}")
            baseline_error = float(np.max(np.abs(uniform[ref_ids] - saved["scores"])))
            if baseline_error > 2e-6:
                raise ValueError(f"Baseline score mismatch {baseline_error}: {folder}")
            sample_id = str(saved["sample_id"])
        temp = destination.with_name(destination.stem + ".partial.npz")
        np.savez_compressed(temp, sample_id=sample_id, labels=labels,
                            uniform_scores=uniform, visible_scores=visible_mean,
                            seen=seen, visible_count=visible_count,
                            reference_ids=ref_ids,
                            baseline_max_abs_error=baseline_error)
        temp.replace(destination)
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"{category}: {index+1}/{len(paths)} full-domain maps; "
                  f"last {time.perf_counter()-begin:.1f}s, "
                  f"ref error {baseline_error:.2g}", flush=True)


def evaluate(output_root: Path, result_csv: Path, audit_csv: Path) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    metric_rows = []
    audit_rows = []
    for category in CATEGORIES:
        files = sorted((output_root / category).glob("sample_*.npz"))
        if len(files) != (53 if category == "chicken" else 50):
            raise ValueError(f"Missing full-domain samples in {category}")
        buckets = {m: {"point_y": [], "point_s": [], "obj_y": [], "obj_s": []}
                   for m in ("uniform15", "visible15")}
        for file in files:
            with np.load(file, allow_pickle=False) as data:
                sid = str(data["sample_id"])
                labels = data["labels"]
                seen = data["seen"]
                known = labels >= 0
                evaluated = seen & known
                if not evaluated.any():
                    raise ValueError(f"No evaluated raw points: {sid}")
                oy = int(sid.split("/")[2] != "good")
                audit_rows.append({"sample_id": sid, "category": category,
                                   "object_label": oy, "raw_points": len(labels),
                                   "reference_points": len(data["reference_ids"]),
                                   "annotated_points": int(known.sum()),
                                   "visible_points": int(seen.sum()),
                                   "evaluated_points": int(evaluated.sum()),
                                   "raw_positive_points": int((labels == 1).sum()),
                                   "visible_positive_points": int(((labels == 1) & seen).sum()),
                                   "reference_positive_points": int((labels[data["reference_ids"]] == 1).sum()),
                                   "baseline_max_abs_error":
                                   float(data["baseline_max_abs_error"])})
                for method, score_key in (("uniform15", "uniform_scores"),
                                          ("visible15", "visible_scores")):
                    scores = data[score_key]
                    bucket = buckets[method]
                    bucket["point_y"].append(labels[evaluated])
                    bucket["point_s"].append(scores[evaluated])
                    bucket["obj_y"].append(oy)
                    bucket["obj_s"].append(float(scores[seen].max()))
        for method, bucket in buckets.items():
            y = np.concatenate(bucket["point_y"])
            s = np.concatenate(bucket["point_s"])
            metric_rows.append({"category": category, "method": method,
                                "object_auroc": 100 * roc_auc_score(bucket["obj_y"],
                                                                  bucket["obj_s"]),
                                "point_auroc": 100 * roc_auc_score(y, s),
                                "point_ap": 100 * average_precision_score(y, s),
                                "objects": len(files), "evaluated_points": len(y),
                                "positive_points": int((y == 1).sum())})
        print(f"evaluated full visible raw domain: {category}", flush=True)
    for method in ("uniform15", "visible15"):
        rows = [r for r in metric_rows if r["method"] == method]
        metric_rows.append({"category": "macro", "method": method,
                            **{key: float(np.mean([r[key] for r in rows])) for key in
                               ("object_auroc", "point_auroc", "point_ap")},
                            **{key: int(sum(r[key] for r in rows)) for key in
                               ("objects", "evaluated_points", "positive_points")}})
    for path, rows in ((result_csv, metric_rows), (audit_csv, audit_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps([r for r in metric_rows if r["category"] == "macro"],
                     ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("infer", "evaluate"))
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--limit-per-category", type=int, default=0,
                        help="0 processes all samples; positive values are smoke tests")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "outputs" / "full_surface")
    parser.add_argument("--result-csv", type=Path,
                        default=ROOT / "outputs" / "full_surface_metrics.csv")
    parser.add_argument("--audit-csv", type=Path,
                        default=ROOT / "outputs" / "full_surface_audit.csv")
    args = parser.parse_args()
    if args.mode == "infer":
        with ZipFile(args.archive) as archive:
            for category in args.categories:
                infer_category(category, args.output_root, archive,
                               args.limit_per_category)
    else:
        evaluate(args.output_root, args.result_csv, args.audit_csv)


if __name__ == "__main__":
    main()
