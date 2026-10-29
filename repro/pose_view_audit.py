"""Compare original and registered front-view references on all objects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .bundle_ablate import CATEGORIES, ROOT, records
from .pose_registration import DATA, DEFAULT_OUTPUT, normalized_raw, sample_paths
from .raw_coverage import RAW_ZIP


def front_ids(points: np.ndarray, K: np.ndarray, E: np.ndarray) -> np.ndarray:
    cam = points @ E[:3, :3].T + E[:3, 3]
    uvw = cam @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    cells = np.floor((uv + .5) / 6).astype(np.int32)
    valid = (cam[:, 2] > 0) & (cells >= 0).all(1) & (cells < 112).all(1)
    ids = np.flatnonzero(valid)
    flat = cells[ids, 1] * 112 + cells[ids, 0]
    order = np.lexsort((ids, cam[ids, 2], flat))
    sorted_flat = flat[order]
    return ids[order[np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pose_registration/front_view_audit.csv")
    args = parser.parse_args()
    rows = []
    with ZipFile(args.archive) as archive:
        for category in args.categories:
            test_paths = records(category)
            test_index = {p.parent.name: i for i, p in enumerate(test_paths)}
            paths = sample_paths(category)
            for index, path in enumerate(paths, 1):
                meta = json.loads(path.read_text(encoding="utf-8"))
                name = Path(meta["raw_member"]).stem
                pose_path = args.pose_root / category / meta["split"] / name / "pose.json"
                pose = json.loads(pose_path.read_text(encoding="utf-8"))
                camera = json.loads((path.parent / "cameras.json").read_text(encoding="utf-8"))["cameras"][0]
                K = np.asarray(camera["intrinsic_672"], dtype=np.float64)
                E = np.asarray(camera["extrinsic"], dtype=np.float64)
                raw = normalized_raw(archive, meta)
                aligned = raw @ np.asarray(pose["rotation_row"]) + np.asarray(pose["translation_row"])
                selected = front_ids(aligned, K, E)
                np.save(args.pose_root / category / meta["split"] / name / "aligned_front_ids.npy",
                        np.sort(selected))
                if meta["split"] == "train":
                    labels = np.zeros(len(raw), dtype=np.int8)
                else:
                    cache = ROOT / "outputs/full_surface" / category / f"sample_{test_index[name]:04d}.npz"
                    with np.load(cache, allow_pickle=False) as data:
                        labels = data["labels"].astype(np.int8)
                if len(labels) != len(raw):
                    raise ValueError(f"Label count mismatch for {category}/{name}")
                old_ids = np.load(path.parent / "points.npz")["raw_point_ids"]
                old_positive = int(np.sum(labels[old_ids] == 1))
                if old_positive != meta["reference_positive_points"]:
                    raise ValueError(f"Original label mismatch for {category}/{name}")
                rows.append({
                    "category": category, "split": meta["split"], "sample": name,
                    "is_anomaly": int(meta["defect"] != "good"),
                    "old_reference_points": len(old_ids), "aligned_reference_points": len(selected),
                    "old_positive_references": old_positive,
                    "aligned_positive_references": int(np.sum(labels[selected] == 1)),
                    "raw_positive_points": int(np.sum(labels == 1)),
                    "pose_rotation_deg": pose["rotation_deg"],
                    "chamfer_before": pose["chamfer_before"],
                    "chamfer_after": pose["chamfer_after"],
                    "opposite_inplane_chamfer": pose["opposite_inplane_chamfer"],
                })
            print(f"{category}: {len(paths)} front-view audits", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} paired front-view records to {args.output}")


if __name__ == "__main__":
    main()
