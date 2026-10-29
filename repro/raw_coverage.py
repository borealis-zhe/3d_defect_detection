"""Approximate raw-cloud visibility audit for selected camera sets.

This reconstructs a geometric z-buffer from *raw* PCD points and the saved
camera matrices. It does not reproduce Open3D's voxelized point-sprite depth
buffer, so these are geometric proxies, separate from the exact reference
visibility used to choose views and compute detection metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import minimum_filter

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, records
from .coverage_ablate import METHODS


RAW_ZIP = Path("F:/data/Real3D-AD/real3d-ad-pcd.zip")
EXPECTED_SHA = "921de624168bf329a668954ccbc26f8c169cbaabc9123176361c98d8b13e033d"
IMAGE_SIZE = 672


def read_xyz(blob: bytes) -> np.ndarray:
    marker = b"DATA binary\n"
    if marker not in blob:
        raise ValueError("Expected binary XYZ PCD")
    header, body = blob.split(marker, 1)
    fields = {line.split()[0]: line.split()[1:]
              for line in header.decode().splitlines()
              if line and not line.startswith("#")}
    if (fields["FIELDS"] != ["x", "y", "z"] or
            fields["SIZE"] != ["4", "4", "4"] or
            fields["TYPE"] != ["F", "F", "F"]):
        raise ValueError("Unexpected PCD schema")
    raw = np.frombuffer(body, dtype="<f4").reshape(-1, 3)
    if len(raw) != int(fields["POINTS"][0]):
        raise ValueError("Invalid PCD point count")
    return raw


def visibility_for_sample(raw: np.ndarray, sample: dict,
                          cameras: list[dict]) -> np.ndarray:
    basis = np.asarray(sample["basis"], dtype=np.float64)
    center = np.asarray(sample["oriented_center"], dtype=np.float64)
    scale = float(sample["normalization_radius"])
    points = (raw.astype(np.float64) @ basis - center) / scale
    out = np.zeros((len(cameras), len(raw)), dtype=bool)
    for view_id, camera in enumerate(cameras):
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        extrinsic = np.asarray(camera["extrinsic"], dtype=np.float64)
        intrinsic = np.asarray(camera["intrinsic_672"], dtype=np.float64)
        cam = (points @ rotation.T) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        depth = cam[:, 2]
        projected = cam @ intrinsic.T
        uv = projected[:, :2] / projected[:, 2:3]
        pixel = np.rint(uv).astype(np.int32)
        valid = ((depth > 0) & (pixel[:, 0] >= 0) &
                 (pixel[:, 0] < IMAGE_SIZE) & (pixel[:, 1] >= 0) &
                 (pixel[:, 1] < IMAGE_SIZE))
        ids = np.flatnonzero(valid)
        flat = pixel[ids, 1] * IMAGE_SIZE + pixel[ids, 0]
        z_buffer = np.full(IMAGE_SIZE * IMAGE_SIZE, np.inf, dtype=np.float32)
        np.minimum.at(z_buffer, flat, depth[ids].astype(np.float32))
        front = minimum_filter(z_buffer.reshape(IMAGE_SIZE, IMAGE_SIZE),
                               size=3, mode="constant", cval=np.inf).ravel()
        out[view_id, ids] = np.isfinite(front[flat]) & (np.abs(
            depth[ids] - front[flat]) <= 0.025)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--selection-csv", type=Path,
                        default=ROOT / "outputs" / "coverage_selection.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "raw_coverage_proxy.csv")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--limit-per-category", type=int, default=0,
                        help="0 audits every object; positive values are smoke tests")
    args = parser.parse_args()
    selection = {}
    with args.selection_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selection[(row["category"], row["sample"], row["method"],
                       int(row["budget"]))] = [int(x) for x in row["view_ids"].split(",")]
    rows = []
    with ZipFile(args.archive) as archive:
        for category in args.categories:
            paths = records(category)
            if args.limit_per_category:
                paths = paths[:args.limit_per_category]
            for index, meta_path in enumerate(paths, 1):
                t0 = time.perf_counter()
                sample = json.loads(meta_path.read_text(encoding="utf-8"))
                cameras = json.loads((meta_path.parent / "cameras.json").read_text(
                    encoding="utf-8"))["cameras"]
                raw = read_xyz(archive.read(sample["raw_member"]))
                if len(raw) != sample["raw_points"] or len(cameras) != 27:
                    raise ValueError(f"Metadata mismatch: {meta_path}")
                visible = visibility_for_sample(raw, sample, cameras)
                reachable = visible.any(axis=0)
                reachable_count = int(reachable.sum())
                for budget in (3, 5, 9, 15):
                    for method in METHODS:
                        ids = selection[(category, meta_path.parent.name,
                                         method, budget)]
                        covered = int(visible[ids].any(axis=0).sum())
                        rows.append({"category": category,
                                     "sample": meta_path.parent.name,
                                     "method": method, "budget": budget,
                                     "raw_points": len(raw),
                                     "covered_raw_points": covered,
                                     "reachable_by_27": reachable_count,
                                     "raw_coverage_percent": 100 * covered / len(raw),
                                     "reachable_fraction_percent":
                                     100 * covered / reachable_count
                                     if reachable_count else float("nan")})
                if index % 10 == 0 or index == len(paths):
                    print(f"{category}: raw visibility {index}/{len(paths)}, "
                          f"last {time.perf_counter()-t0:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} raw geometric proxy rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
