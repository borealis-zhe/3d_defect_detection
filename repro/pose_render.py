"""Re-render a pose-registered sample with the original Open3D camera contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, VIEW_IDS, records
from .pose_registration import DATA, DEFAULT_OUTPUT, normalized_raw
from .raw_coverage import RAW_ZIP


def original_preprocessor():
    source = HANDOFF / "scripts" / "preprocess_pcd.py"
    spec = importlib.util.spec_from_file_location("original_real3d_preprocessor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_sample(category: str, split: str, sample: str) -> Path:
    paths = sorted((DATA / category / split).glob(f"*/xyz/{sample}/sample.json"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {category}/{split}/{sample}, got {len(paths)}")
    return paths[0]


def raw_labels(category: str, split: str, sample: str, n: int) -> np.ndarray:
    if split == "train":
        return np.zeros(n, dtype=np.int8)
    paths = records(category)
    matches = [i for i, p in enumerate(paths) if p.parent.name == sample]
    if len(matches) != 1:
        raise ValueError("Unable to associate sample with original full-surface label cache")
    path = ROOT / "outputs" / "full_surface" / category / f"sample_{matches[0]:04d}.npz"
    with np.load(path, allow_pickle=False) as old:
        labels = old["labels"].astype(np.int8)
    if len(labels) != n:
        raise ValueError("Raw label count differs from point count")
    return labels


def render_one(category: str, split: str, sample: str, views: tuple[int, ...],
               output_root: Path, archive: ZipFile, processor,
               inplane_flip: bool = False,
               pose_root: Path = DEFAULT_OUTPUT) -> Path:
    sample_path = find_sample(category, split, sample)
    meta = json.loads(sample_path.read_text(encoding="utf-8"))
    pose_path = pose_root / category / split / sample / "pose.json"
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    rotation = np.asarray(pose["rotation_row"], dtype=np.float64)
    translation = np.asarray(pose["translation_row"], dtype=np.float64)
    normalized = normalized_raw(archive, meta)
    aligned = normalized @ rotation + translation
    if inplane_flip:
        aligned = aligned @ np.diag([-1., -1., 1.])
    labels = raw_labels(category, split, sample, len(aligned))
    target = output_root / category / split / sample
    target.mkdir(parents=True, exist_ok=True)
    renderer = processor.Renderer()
    try:
        cloud = renderer.cloud(aligned)
        K, E = renderer.render(cloud)
        ref_ids, _ = processor.front_representatives(aligned, K, E)
        refs = aligned[ref_ids]
        positions = []
        visibility = []
        for view in views:
            image, pos, visible, _ = renderer.render(cloud, refs, processor.ANGLES[view])
            image.save(target / f"view_{view:03d}.png")
            np.save(target / f"view_{view:03d}.npy", pos)
            positions.append(pos)
            visibility.append(visible)
    finally:
        renderer.close()
    np.savez_compressed(target / "points.npz", raw_point_ids=ref_ids, xyz=refs.astype(np.float32),
                        labels=labels[ref_ids], visibility=np.stack(visibility),
                        view_ids=np.asarray(views, dtype=np.int8))
    record = {"category": category, "split": split, "sample": sample,
              "raw_points": len(aligned), "render_points": len(cloud.points),
              "reference_points": len(ref_ids), "positive_reference_points": int(np.sum(labels[ref_ids] == 1)),
              "views": list(views), "source_meta": str(sample_path), "pose": str(pose_path),
              "inplane_flip": inplane_flip,
              "renderer": "original_open3d_pcd_self_v1"}
    (target / "render.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{category}/{split}/{sample}: {len(ref_ids)} references, {record['positive_reference_points']} positive", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--all27", action="store_true")
    parser.add_argument("--views", nargs="+", type=int,
                        help="Explicit view IDs; default is the 15-view inference set")
    parser.add_argument("--inplane-flip", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pose_render")
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    args = parser.parse_args()
    processor = original_preprocessor()
    views = tuple(args.views) if args.views else (tuple(range(27)) if args.all27 else tuple(VIEW_IDS))
    if not views or any(v < 0 or v >= 27 for v in views):
        raise ValueError("View IDs must be within 0..26")
    with ZipFile(args.archive) as archive:
        render_one(args.category, args.split, args.sample, views, args.output, archive,
                   processor, args.inplane_flip, args.pose_root)


if __name__ == "__main__":
    main()
