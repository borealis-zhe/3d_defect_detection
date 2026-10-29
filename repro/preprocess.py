"""Prepare independent Real3D-AD gray renders and point correspondences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import (category_basis, front_reference_indices,
                       inference_view_indices, local_views, normalize_points,
                       render)


CATEGORIES = ("airplane", "candybar", "car", "chicken", "diamond", "duck",
              "fish", "gemstone", "seahorse", "shell", "starfish", "toffees")


def read_pcd(path: Path) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Invalid or empty PCD: {path}")
    return points


def load_labels(pcd_path: Path, raw_points: np.ndarray, gt_dir: Path) -> np.ndarray:
    if "good" in pcd_path.stem.lower():
        return np.zeros(len(raw_points), dtype=np.int8)
    gt_path = gt_dir / (pcd_path.stem + ".txt")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing anomaly labels: {gt_path}")
    gt = np.loadtxt(gt_path, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] < 4:
        raise ValueError(f"Expected xyz plus point label in {gt_path}")
    if len(gt) == len(raw_points) and np.allclose(gt[:, :3], raw_points,
                                                  rtol=0, atol=1e-5):
        return gt[:, 3].astype(np.int8)
    # Never silently assume that row order equals point identity.
    from scipy.spatial import cKDTree

    distances, indices = cKDTree(gt[:, :3]).query(raw_points, workers=-1)
    if np.max(distances) > 1e-4:
        raise ValueError(f"GT/PCD coordinates do not match: {gt_path}; "
                         f"max nearest distance={np.max(distances):.6g}")
    return gt[indices, 3].astype(np.int8)


def process_category(dataset_root: Path, output_root: Path, category: str,
                     render_size: int = 224, reference_grid: int = 64) -> dict:
    source = dataset_root / category
    normal_paths = sorted((source / "train").glob("*.pcd"))
    test_paths = sorted(p for p in (source / "test").glob("*.pcd")
                        if "cut" not in p.stem.lower() and "copy" not in p.stem.lower()
                        and "temp" not in p.stem.lower())
    if not normal_paths or not test_paths:
        raise FileNotFoundError(f"Missing train or test PCD files for {category}")
    basis = category_basis(read_pcd(normal_paths[0]))
    views = local_views()
    infer_indices = inference_view_indices()
    out = output_root / category
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "category_basis.npy", basis)
    manifest = {"category": category, "renderer": "independent_orthographic_depth_gray_v1",
                "render_size": render_size, "reference_grid": reference_grid,
                "normal_prototype": normal_paths[0].name,
                "train_angles": [v.angles for v in views],
                "inference_view_indices": infer_indices,
                "train": [], "test": []}

    for split, paths in (("train", normal_paths), ("test", test_paths)):
        for number, path in enumerate(paths, 1):
            raw = read_pcd(path)
            points = normalize_points(raw, basis)
            labels = (np.zeros(len(raw), dtype=np.int8) if split == "train" else
                      load_labels(path, raw, source / "gt"))
            reference_ids = front_reference_indices(points, views[0].rotation,
                                                    grid_size=reference_grid)
            selected_views = list(enumerate(views)) if split == "train" else [
                (i, views[i]) for i in infer_indices]
            item_dir = out / split / path.stem
            item_dir.mkdir(parents=True, exist_ok=True)
            projections = []
            visibility = []
            for view_id, view in selected_views:
                rgb, uv, visible, _ = render(points, view.rotation, render_size)
                Image.fromarray(rgb, mode="RGB").save(item_dir / f"{view_id:02d}.png")
                projections.append(uv[reference_ids])
                visibility.append(visible[reference_ids])
            np.savez_compressed(item_dir / "points.npz",
                                raw_ids=reference_ids.astype(np.int32),
                                xyz=points[reference_ids],
                                labels=labels[reference_ids],
                                projection=np.stack(projections),
                                visible=np.stack(visibility),
                                view_ids=np.array([i for i, _ in selected_views], dtype=np.int8),
                                object_label=np.int8(int(np.any(labels == 1))))
            manifest[split].append({"source": str(path), "name": path.stem,
                                    "raw_points": len(raw),
                                    "reference_points": len(reference_ids),
                                    "positive_reference_points": int(np.sum(labels[reference_ids] == 1)),
                                    "object_label": int(np.any(labels == 1))})
            print(f"{category} {split} {number}/{len(paths)} {path.stem}: "
                  f"{len(raw)} raw, {len(reference_ids)} reference", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--render-size", type=int, default=224)
    parser.add_argument("--reference-grid", type=int, default=64)
    args = parser.parse_args()
    for category in args.categories:
        process_category(args.dataset_root, args.output_root, category,
                         args.render_size, args.reference_grid)


if __name__ == "__main__":
    main()
