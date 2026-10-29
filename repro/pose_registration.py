"""Label-blind per-object rigid registration to one normal prototype per class.

The existing category PCA transform and radius normalization are read from the
audited PCD-self-v1 metadata. Twenty-four proper PCA-axis mappings are scored
against the first normal training object, then the best candidates are refined
with trimmed point-to-point ICP. No anomaly labels enter pose estimation.
Outputs are additive and never overwrite the original preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import permutations, product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT
from .raw_coverage import RAW_ZIP, read_xyz


DATA = HANDOFF / "data" / "pcd_self_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "pose_registration"


def sample_paths(category: str) -> list[Path]:
    return sorted((DATA / category).glob("train/good/xyz/*/sample.json")) + \
        sorted((DATA / category).glob("test/*/xyz/*/sample.json"))


def normalized_raw(archive: ZipFile, meta: dict) -> np.ndarray:
    raw = read_xyz(archive.read(meta["raw_member"]))
    return ((raw.astype(np.float64) @ np.asarray(meta["basis"], np.float64) -
             np.asarray(meta["oriented_center"], np.float64)) /
            float(meta["normalization_radius"]))


def sample_indices(n: int, budget: int, key: str) -> np.ndarray:
    if n <= budget:
        return np.arange(n)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).choice(n, budget, replace=False)


def pca_frame(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(0)
    values, vectors = np.linalg.eigh(centered.T @ centered / (len(x) - 1))
    return vectors[:, ::-1], values[::-1]


def axis_maps():
    for perm in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            matrix = np.zeros((3, 3))
            matrix[np.arange(3), perm] = signs
            yield matrix


AXIS_MAPS = tuple(axis_maps())
assert len(AXIS_MAPS) == 48


def chamfer(source: np.ndarray, target: np.ndarray, target_tree: cKDTree,
            keep: float = .85) -> float:
    dist_forward = target_tree.query(source, k=1, workers=1)[0]
    source_tree = cKDTree(source)
    dist_backward = source_tree.query(target, k=1, workers=1)[0]
    kf = max(1, int(len(dist_forward) * keep))
    kb = max(1, int(len(dist_backward) * keep))
    return float((np.partition(dist_forward, kf - 1)[:kf].mean() +
                  np.partition(dist_backward, kb - 1)[:kb].mean()) / 2)


def icp(source: np.ndarray, target: np.ndarray, tree: cKDTree,
        initial: np.ndarray, steps: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rotation = initial.copy()
    translation = np.zeros(3)
    for _ in range(steps):
        moved = source @ rotation + translation
        distances, nearest = tree.query(moved, k=1, workers=1)
        cutoff = np.quantile(distances, .8)
        good = distances <= cutoff
        a = moved[good]
        b = target[nearest[good]]
        a0, b0 = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - a0).T @ (b - b0))
        delta = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        rotation = rotation @ delta
        translation = (translation - a0) @ delta + b0
        if np.linalg.norm(translation) > .35:
            raise ValueError("ICP translation exceeds safe alignment range")
    return rotation, translation


def register(query: np.ndarray, template: np.ndarray, *, refine_top: int = 4) -> dict:
    query_frame, query_eig = pca_frame(query)
    template_frame, template_eig = pca_frame(template)
    target_tree = cKDTree(template)
    candidates = []
    for perm in AXIS_MAPS:
        rotation = query_frame @ perm @ template_frame.T
        if np.linalg.det(rotation) < 0:
            continue
        score = chamfer(query @ rotation, template, target_tree)
        candidates.append((score, rotation))
    candidates.sort(key=lambda row: row[0])
    refined = []
    for start_score, initial in candidates[:refine_top]:
        try:
            rotation, translation = icp(query, template, target_tree, initial)
        except ValueError:
            continue
        score = chamfer(query @ rotation + translation, template, target_tree)
        refined.append((score, rotation, translation, start_score))
    if not refined:
        refined = [(candidates[0][0], candidates[0][1], np.zeros(3), candidates[0][0])]
    refined.sort(key=lambda row: row[0])
    best = refined[0]
    original_score = chamfer(query, template, target_tree)
    if original_score <= best[0]:
        best = (original_score, np.eye(3), np.zeros(3), original_score)
    flip_z = np.diag([-1., -1., 1.])
    opposite_score = chamfer((query @ best[1] + best[2]) @ flip_z,
                             template, target_tree)
    return {
        "rotation": best[1], "translation": best[2], "chamfer_before": original_score,
        "chamfer_after": best[0], "candidate_gap": candidates[1][0] - candidates[0][0],
        "opposite_inplane_chamfer": opposite_score,
        "query_eigenvalues": query_eig, "template_eigenvalues": template_eig,
        "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(best[1]) - 1) / 2, -1, 1)))),
    }


def align_category(category: str, archive: ZipFile, output: Path,
                   sample_budget: int, limit: int = 0) -> list[dict]:
    paths = sample_paths(category)
    normal = [p for p in paths if "/train/" in p.as_posix()]
    if len(normal) != 4:
        raise ValueError(f"Expected four normal training clouds for {category}")
    template_path = normal[0]
    template_meta = json.loads(template_path.read_text(encoding="utf-8"))
    template_raw = normalized_raw(archive, template_meta)
    template = template_raw[sample_indices(len(template_raw), sample_budget,
                                           template_meta["raw_member"])]
    tree = cKDTree(template)
    rows = []
    if limit:
        paths = paths[:limit]
    for index, path in enumerate(paths, 1):
        meta = json.loads(path.read_text(encoding="utf-8"))
        sample_key = meta["raw_member"]
        raw = normalized_raw(archive, meta)
        query = raw[sample_indices(len(raw), sample_budget, sample_key)]
        if path == template_path:
            result = {"rotation": np.eye(3), "translation": np.zeros(3),
                      "chamfer_before": 0., "chamfer_after": 0., "candidate_gap": None,
                      "opposite_inplane_chamfer": None,
                      "query_eigenvalues": pca_frame(query)[1],
                      "template_eigenvalues": pca_frame(template)[1], "rotation_deg": 0.}
        else:
            result = register(query, template)
        rotation, translation = result["rotation"], result["translation"]
        if abs(np.linalg.det(rotation) - 1) > 1e-5:
            raise ValueError("Registration produced reflection")
        destination = output / category / meta["split"] / Path(sample_key).stem
        destination.mkdir(parents=True, exist_ok=True)
        record = {
            "category": category, "split": meta["split"], "sample": Path(sample_key).stem,
            "raw_member": sample_key, "normal_template": template_meta["raw_member"],
            "rotation_row": rotation.tolist(), "translation_row": translation.tolist(),
            "chamfer_before": result["chamfer_before"], "chamfer_after": result["chamfer_after"],
            "candidate_gap": result["candidate_gap"], "rotation_deg": result["rotation_deg"],
            "opposite_inplane_chamfer": result["opposite_inplane_chamfer"],
            "query_eigenvalues": result["query_eigenvalues"].tolist(),
            "template_eigenvalues": result["template_eigenvalues"].tolist(),
            "sample_budget": sample_budget, "algorithm": "PCA24_trimmed_chamfer_ICP7_v1",
        }
        (destination / "pose.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({k: record[k] for k in ("category", "split", "sample", "chamfer_before",
                                          "chamfer_after", "candidate_gap", "rotation_deg",
                                          "opposite_inplane_chamfer")})
        if index % 10 == 0 or index == len(paths):
            print(f"{category}: {index}/{len(paths)} pose registrations", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-budget", type=int, default=1200)
    parser.add_argument("--limit-per-category", type=int, default=0)
    args = parser.parse_args()
    if args.sample_budget < 200:
        raise ValueError("At least 200 points are needed for pose fitting")
    rows = []
    with ZipFile(args.archive) as archive:
        for category in args.categories:
            rows.extend(align_category(category, archive, args.output,
                                       args.sample_budget, args.limit_per_category))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "pose_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} pose records to {args.output}")


if __name__ == "__main__":
    main()
