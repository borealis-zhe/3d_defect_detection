"""Normal-only stress test of local-shape features under point subsampling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT
from .geometry_memory import local_shape, local_shape_radius
from .raw_coverage import RAW_ZIP, read_xyz
from .full_surface import normalized_raw


def one_category(category: str, archive: ZipFile,
                 descriptor: str = "knn") -> list[dict]:
    extractor = local_shape if descriptor == "knn" else local_shape_radius
    base = HANDOFF / "data" / "pcd_self_v1" / category
    paths = sorted(base.glob("train/good/xyz/*/sample.json"))
    if len(paths) != 4:
        raise ValueError(f"Need four normal prototypes: {category}")
    raw_clouds = []
    refs = []
    descriptors = []
    for path in paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        raw = normalized_raw(read_xyz(archive.read(meta["raw_member"])), meta)
        with np.load(path.parent / "points.npz", allow_pickle=False) as data:
            reference = data["xyz"]
        raw_clouds.append(raw)
        refs.append(reference)
        descriptors.append(extractor(raw, reference))
    normal = np.concatenate(descriptors)
    median = np.median(normal, axis=0)
    q25, q75 = np.quantile(normal, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 0.002)
    train_other = np.concatenate(descriptors[1:])
    tree = cKDTree((train_other - median) / scale)
    rng = np.random.default_rng(20261028)
    output = []
    for fraction in (1.0, 0.5, 0.25):
        raw = raw_clouds[0]
        if fraction < 1:
            ids = np.sort(rng.choice(len(raw),
                                     size=int(len(raw) * fraction), replace=False))
            raw = raw[ids]
        features = extractor(raw, refs[0])
        scores = tree.query((features - median) / scale,
                            workers=-1)[0]
        output.append({"category": category, "descriptor": descriptor,
                       "retained_fraction": fraction,
                       "reference_points": len(refs[0]),
                       "median_shape_novelty": float(np.median(scores)),
                       "p95_shape_novelty": float(np.quantile(scores, 0.95))})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--descriptor", choices=("knn", "radius"), default="knn")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "geometry_density_audit.csv")
    args = parser.parse_args()
    rows = []
    with ZipFile(args.archive) as archive:
        for category in CATEGORIES:
            rows.extend(one_category(category, archive, args.descriptor))
            print(f"audited normal subsampling: {category}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
