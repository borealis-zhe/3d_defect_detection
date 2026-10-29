"""Normal-only 3D geometry memory versus frozen Dinomaly-MV scores.

This is a small, non-neural research control, not a new architecture. The
primary hypothesis is that multiscale local shape novelty complements the
2D reconstruction residual. All memory items come from the four normal
training PCDs per category. No test labels enter feature extraction.

Score fusion uses category-wide label-blind percentile ranks of test scores;
this is transductive and must not be presented as an online calibration or
independent confirmation on the repeatedly inspected test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, records
from .raw_coverage import RAW_ZIP, read_xyz
from .full_surface import normalized_raw


K_NEIGHBORS = (32, 128)
POSITION_STRIDE = 8


def local_shape(raw_xyz: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Scale-free covariance eigenvalue ratios at two neighborhood sizes."""
    if len(raw_xyz) < max(K_NEIGHBORS):
        raise ValueError("Too few raw points for local geometry")
    neighbors = cKDTree(raw_xyz).query(references,
                                        k=max(K_NEIGHBORS), workers=-1)[1]
    nearby = raw_xyz[neighbors]
    parts = []
    for k in K_NEIGHBORS:
        group = nearby[:, :k]
        centered = group - group.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / k
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        ratios = eigenvalues / np.maximum(eigenvalues.sum(axis=1, keepdims=True),
                                          1e-12)
        # Largest ratio is determined by the two smaller ones.
        parts.append(ratios[:, :2])
    return np.concatenate(parts, axis=1).astype(np.float32)


def local_shape_radius(raw_xyz: np.ndarray, references: np.ndarray,
                       radius: float = 0.04) -> np.ndarray:
    """Covariance shape at a fixed normalized spatial support radius.

    All neighbors inside the radius contribute equally. A 12-nearest-neighbor
    fallback is used only where the raw scan has too few points locally.
    Chunking bounds temporary memory for multi-million-point clouds.
    """
    tree = cKDTree(raw_xyz)
    parts = []
    for start in range(0, len(references), 256):
        chunk = references[start:start + 256]
        groups = tree.query_ball_point(chunk, r=radius, workers=-1)
        sparse = [i for i, group in enumerate(groups) if len(group) < 12]
        if sparse:
            neighbors = tree.query(chunk[sparse], k=12, workers=-1)[1]
            for i, ids in zip(sparse, neighbors):
                groups[i] = np.atleast_1d(ids).tolist()
        lengths = np.asarray([len(group) for group in groups], dtype=np.int32)
        owners = np.repeat(np.arange(len(chunk)), lengths)
        all_ids = np.concatenate(groups).astype(np.int64)
        xyz = raw_xyz[all_ids]
        count = lengths.astype(np.float64)
        mean = np.column_stack([
            np.bincount(owners, weights=xyz[:, i], minlength=len(chunk)) / count
            for i in range(3)])
        covariance = np.empty((len(chunk), 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(i, 3):
                moment = (np.bincount(owners, weights=xyz[:, i] * xyz[:, j],
                                      minlength=len(chunk)) / count)
                covariance[:, i, j] = moment - mean[:, i] * mean[:, j]
                covariance[:, j, i] = covariance[:, i, j]
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        ratios = eigenvalues / np.maximum(eigenvalues.sum(axis=1, keepdims=True),
                                          1e-12)
        parts.append(ratios[:, :2].astype(np.float32))
    return np.concatenate(parts)


def training_memory(category: str, archive: ZipFile,
                    descriptor: str = "knn"):
    extractor = local_shape if descriptor == "knn" else local_shape_radius
    base = HANDOFF / "data" / "pcd_self_v1" / category
    meta_paths = sorted(base.glob("train/good/xyz/*/sample.json"))
    if len(meta_paths) != 4:
        raise ValueError(f"Need four normal prototypes: {category}")
    positions = []
    shapes = []
    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = read_xyz(archive.read(meta["raw_member"]))
        xyz = normalized_raw(raw, meta)
        with np.load(meta_path.parent / "points.npz", allow_pickle=False) as point:
            refs = point["xyz"]
            if np.max(np.abs(xyz[point["raw_point_ids"]] - refs)) > 1e-6:
                raise ValueError(f"Normal raw/reference mismatch: {meta_path}")
        positions.append(xyz[::POSITION_STRIDE].astype(np.float32))
        shapes.append(extractor(xyz, refs))
    position_memory = np.concatenate(positions)
    shape_memory = np.concatenate(shapes)
    median = np.median(shape_memory, axis=0)
    lower, upper = np.quantile(shape_memory, [0.25, 0.75], axis=0)
    scale = np.maximum(upper - lower, 0.002)
    return cKDTree(position_memory), cKDTree((shape_memory - median) / scale), {
        "normal_prototypes": len(meta_paths),
        "descriptor": descriptor,
        "support_radius": 0.04 if descriptor == "radius" else None,
        "position_memory_points": len(position_memory),
        "shape_memory_points": len(shape_memory),
        "shape_median": median.tolist(),
        "shape_iqr_scale": scale.tolist(),
        "k_neighbors": list(K_NEIGHBORS),
        "position_stride": POSITION_STRIDE,
    }


def extract_category(category: str, output_root: Path, archive: ZipFile,
                     limit: int = 0, descriptor: str = "knn") -> None:
    position_tree, shape_tree, info = training_memory(category, archive,
                                                     descriptor)
    extractor = local_shape if descriptor == "knn" else local_shape_radius
    median = np.asarray(info["shape_median"])
    scale = np.asarray(info["shape_iqr_scale"])
    out = output_root / category
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    paths = records(category)
    if limit:
        paths = paths[:limit]
    for index, meta_path in enumerate(paths):
        destination = out / f"sample_{index:04d}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as cached:
                if "shape_score" not in cached.files:
                    raise ValueError(f"Incomplete geometry cache: {destination}")
            continue
        start = time.perf_counter()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = read_xyz(archive.read(meta["raw_member"]))
        xyz = normalized_raw(raw, meta)
        with np.load(meta_path.parent / "points.npz", allow_pickle=False) as point:
            refs = point["xyz"]
            labels = point["labels"].astype(np.int8)
            if "labels_valid" in point:
                labels[~point["labels_valid"]] = -1
            raw_ids = point["raw_point_ids"]
            if np.max(np.abs(xyz[raw_ids] - refs)) > 1e-6:
                raise ValueError(f"Test raw/reference mismatch: {meta_path}")
        descriptors = extractor(xyz, refs)
        position_score = position_tree.query(refs, workers=-1)[0].astype(np.float32)
        shape_score = shape_tree.query((descriptors - median) / scale,
                                       workers=-1)[0].astype(np.float32)
        saved = (HANDOFF / "results" / "dinomaly_mv_v3" / "eval" / category /
                 f"sample_{index:04d}.npz")
        with np.load(saved, allow_pickle=False) as baseline:
            if not np.array_equal(baseline["raw_point_ids"], raw_ids):
                raise ValueError(f"Baseline point IDs differ: {meta_path}")
            if not np.array_equal(baseline["labels"], labels):
                raise ValueError(f"Baseline labels differ: {meta_path}")
            dino_score = baseline["scores"].astype(np.float32)
            sid = str(baseline["sample_id"])
        temp = destination.with_name(destination.stem + ".partial.npz")
        np.savez_compressed(temp, sample_id=sid, labels=labels,
                            raw_point_ids=raw_ids, dino_score=dino_score,
                            position_score=position_score,
                            shape_score=shape_score)
        temp.replace(destination)
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"{category}: geometry {index+1}/{len(paths)}, "
                  f"last {time.perf_counter()-start:.1f}s", flush=True)


def methods_for_category(files: list[Path]):
    data = []
    for file in files:
        with np.load(file, allow_pickle=False) as item:
            data.append({key: item[key].copy() for key in item.files})
    lengths = [len(item["labels"]) for item in data]
    name_to_values = {}
    for name in ("dino_score", "position_score", "shape_score"):
        values = np.concatenate([item[name] for item in data])
        rank = (rankdata(values, method="average") - 0.5) / len(values)
        name_to_values[name] = rank.astype(np.float32)
    dino = name_to_values["dino_score"]
    position = name_to_values["position_score"]
    shape = name_to_values["shape_score"]
    methods = {
        "dinomaly_baseline": dino,
        "geometry_position": position,
        "geometry_shape": shape,
        "dinomaly_plus_position": 0.5 * (dino + position),
        "dinomaly_plus_shape": 0.5 * (dino + shape),
        "dinomaly_plus_both": 0.5 * dino + 0.25 * position + 0.25 * shape,
    }
    return data, lengths, methods


def evaluate(output_root: Path, result_csv: Path) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for category in CATEGORIES:
        files = sorted((output_root / category).glob("sample_*.npz"))
        if len(files) != (53 if category == "chicken" else 50):
            raise ValueError(f"Missing geometry predictions: {category}")
        data, lengths, methods = methods_for_category(files)
        labels = np.concatenate([item["labels"] for item in data])
        known = labels >= 0
        object_labels = np.asarray([int(str(item["sample_id"]).split("/")[2] != "good")
                                    for item in data])
        ends = np.cumsum(lengths)
        starts = np.r_[0, ends[:-1]]
        for name, scores in methods.items():
            object_scores = np.asarray([scores[a:b].max() for a, b in
                                        zip(starts, ends)])
            rows.append({"category": category, "method": name,
                         "object_auroc": 100 * roc_auc_score(object_labels,
                                                              object_scores),
                         "point_auroc": 100 * roc_auc_score(labels[known],
                                                             scores[known]),
                         "point_ap": 100 * average_precision_score(labels[known],
                                                                     scores[known]),
                         "objects": len(files), "points": int(known.sum()),
                         "positive_points": int((labels == 1).sum())})
        print(f"evaluated geometry memory: {category}", flush=True)
    for name in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == name]
        rows.append({"category": "macro", "method": name,
                     **{metric: float(np.mean([r[metric] for r in subset]))
                        for metric in ("object_auroc", "point_auroc", "point_ap")},
                     **{metric: int(sum(r[metric] for r in subset))
                        for metric in ("objects", "points", "positive_points")}})
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([r for r in rows if r["category"] == "macro"],
                     ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("extract", "evaluate"))
    parser.add_argument("--descriptor", choices=("knn", "radius"), default="knn")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--result-csv", type=Path)
    parser.add_argument("--limit-per-category", type=int, default=0,
                        help="0 extracts all objects; positive values are smoke tests")
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = ROOT / "outputs" / (
            "geometry_radius_memory" if args.descriptor == "radius" else
            "geometry_memory")
    if args.result_csv is None:
        args.result_csv = ROOT / "outputs" / (
            "geometry_radius_metrics.csv" if args.descriptor == "radius" else
            "geometry_memory_metrics.csv")
    if args.mode == "extract":
        with ZipFile(args.archive) as archive:
            for category in args.categories:
                extract_category(category, args.output_root, archive,
                                 args.limit_per_category, args.descriptor)
    else:
        evaluate(args.output_root, args.result_csv)


if __name__ == "__main__":
    main()
