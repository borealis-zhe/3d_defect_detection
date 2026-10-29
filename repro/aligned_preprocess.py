"""Render the audited per-object rigid poses into an isolated Real3D-AD tree.

Run on the remote host under Xvfb with its Open3D environment. The existing
PCD-self-v1 tree is read only; labels are loaded only after the pose is fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


CATEGORIES = ("airplane", "candybar", "car", "chicken", "diamond", "duck",
              "fish", "gemstone", "seahorse", "shell", "starfish", "toffees")
EVAL_VIEWS = (0, 1, 2, 3, 6, 9, 13, 14, 16, 17, 18, 22, 23, 25, 26)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_preprocessor(source: Path):
    path = source / "scripts" / "preprocess_pcd.py"
    spec = importlib.util.spec_from_file_location("audited_original_preprocessor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_sample(processor, archive: ZipFile, sample_meta_path: Path,
                  pose_root: Path, output: Path, renderer) -> dict:
    original = json.loads(sample_meta_path.read_text())
    category, split, defect = (original[k] for k in ("category", "split", "defect"))
    name = sample_meta_path.parent.name
    pose_path = pose_root / category / split / name / "pose.json"
    pose = json.loads(pose_path.read_text())
    if pose["raw_member"] != original["raw_member"]:
        raise ValueError(f"Pose/source mismatch: {name}")
    destination = output / category / split / defect / "xyz" / name
    done = destination / "sample.json"
    expected = {"source_sha256": sha(sample_meta_path), "pose_sha256": sha(pose_path)}
    views = tuple(range(27)) if split == "train" else EVAL_VIEWS
    if done.exists():
        previous = json.loads(done.read_text())
        files = [destination / "points.npz"]
        files += [destination / f"view_{v:03d}.{ext}" for v in views for ext in ("png", "npy")]
        if all(previous.get(k) == v for k, v in expected.items()) and all(p.exists() for p in files):
            return previous
        raise RuntimeError(f"Incomplete or changed output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    raw = processor.read_pcd(archive.read(original["raw_member"]))
    if split == "train" or defect == "good":
        labels = np.zeros(len(raw), dtype=np.int8)
    else:
        gt_member = original["raw_member"].replace("/test/", "/gt/")[:-4] + ".txt"
        labels, _ = processor.match_labels(raw, np.loadtxt(io.BytesIO(archive.read(gt_member))))
    basis = np.asarray(original["basis"], dtype=np.float64)
    center = np.asarray(original["oriented_center"], dtype=np.float64)
    scale = float(original["normalization_radius"])
    points = (raw.astype(np.float64) @ basis - center) / scale
    rotation = np.asarray(pose["rotation_row"], dtype=np.float64)
    translation = np.asarray(pose["translation_row"], dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .99999:
        raise ValueError(f"Invalid rigid pose: {name}")
    points = points @ rotation + translation
    cloud = renderer.cloud(points)
    intrinsic, extrinsic = renderer.render(cloud)
    raw_ids, point_ids = processor.front_representatives(points, intrinsic, extrinsic)
    refs = points[raw_ids]
    visibility = []
    for view in views:
        image, positions, visible, _ = renderer.render(cloud, refs, processor.ANGLES[view])
        image.save(destination / f"view_{view:03d}.png")
        np.save(destination / f"view_{view:03d}.npy", positions)
        visibility.append(visible)
    payload = dict(raw_point_ids=raw_ids, point_ids=point_ids,
                   xyz=refs.astype(np.float32), raw_xyz=raw[raw_ids],
                   labels=np.maximum(labels[raw_ids], 0).astype(np.uint8),
                   visibility=np.stack(visibility), view_ids=np.asarray(views, dtype=np.int8))
    if np.any(labels < 0):
        payload["labels_valid"] = labels[raw_ids] >= 0
    np.savez_compressed(destination / "points.npz", **payload)
    record = dict(category=category, split=split, defect=defect, sample=name,
                  raw_member=original["raw_member"], source_sha256=expected["source_sha256"],
                  pose_sha256=expected["pose_sha256"], raw_points=len(raw),
                  reference_points=len(raw_ids), positive_references=int((labels[raw_ids] == 1).sum()),
                  unknown_references=int((labels[raw_ids] < 0).sum()), views=list(views),
                  orientation="PCD-self-v1 normalization followed by label-blind per-object rigid pose",
                  renderer="audited original Open3D 672-to-224 renderer")
    temporary = done.with_suffix(".json.partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(done)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=["train", "test"])
    args = parser.parse_args()
    processor = original_preprocessor(args.original)
    renderer = processor.Renderer()
    try:
        with ZipFile(args.archive) as archive:
            for category in args.categories:
                for split in args.splits:
                    files = sorted((args.original / "data/pcd_self_v1" / category / split).glob("*/xyz/*/sample.json"))
                    if len(files) != (4 if split == "train" else (53 if category == "chicken" else 50)):
                        raise ValueError(f"Unexpected sample count: {category}/{split}: {len(files)}")
                    for index, path in enumerate(files, 1):
                        result = export_sample(processor, archive, path, args.pose_root, args.output, renderer)
                        if index % 10 == 0 or index == len(files):
                            print(f"{category}/{split} {index}/{len(files)}: {result['sample']}", flush=True)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
