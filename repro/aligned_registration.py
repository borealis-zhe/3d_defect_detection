"""Recompute label-blind per-object Real3D-AD poses on the remote raw archive.

This is the standalone server form of repro/pose_registration.py. It reads the
existing PCD-self-v1 normalization metadata and never reads anomaly labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from itertools import permutations, product
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from .aligned_preprocess import CATEGORIES


def processor_from(source: Path):
    path = source / "scripts/preprocess_pcd.py"
    spec = importlib.util.spec_from_file_location("audited_original_preprocessor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_indices(n: int, budget: int, key: str) -> np.ndarray:
    if n <= budget:
        return np.arange(n)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).choice(n, budget, replace=False)


def pca_frame(x):
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


def chamfer(source, target, target_tree, keep=.85):
    forward = target_tree.query(source, k=1, workers=1)[0]
    backward = cKDTree(source).query(target, k=1, workers=1)[0]
    nf = max(1, int(len(forward) * keep))
    nb = max(1, int(len(backward) * keep))
    return float((np.partition(forward, nf - 1)[:nf].mean() +
                  np.partition(backward, nb - 1)[:nb].mean()) / 2)


def icp(source, target, tree, initial, steps=7):
    rotation = initial.copy()
    translation = np.zeros(3)
    for _ in range(steps):
        moved = source @ rotation + translation
        distances, nearest = tree.query(moved, k=1, workers=1)
        good = distances <= np.quantile(distances, .8)
        a, b = moved[good], target[nearest[good]]
        a0, b0 = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((a - a0).T @ (b - b0))
        delta = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        rotation = rotation @ delta
        translation = (translation - a0) @ delta + b0
        if np.linalg.norm(translation) > .35:
            raise ValueError("ICP translation exceeds safe range")
    return rotation, translation


def register(query, template):
    query_frame, query_eig = pca_frame(query)
    template_frame, template_eig = pca_frame(template)
    tree = cKDTree(template)
    candidates = []
    for axis in AXIS_MAPS:
        rotation = query_frame @ axis @ template_frame.T
        if np.linalg.det(rotation) > 0:
            candidates.append((chamfer(query @ rotation, template, tree), rotation))
    candidates.sort(key=lambda item: item[0])
    refined = []
    for start_score, initial in candidates[:4]:
        try:
            rotation, translation = icp(query, template, tree, initial)
        except ValueError:
            continue
        score = chamfer(query @ rotation + translation, template, tree)
        refined.append((score, rotation, translation, start_score))
    if not refined:
        refined = [(candidates[0][0], candidates[0][1], np.zeros(3), candidates[0][0])]
    refined.sort(key=lambda item: item[0])
    best = refined[0]
    original_score = chamfer(query, template, tree)
    if original_score <= best[0]:
        best = (original_score, np.eye(3), np.zeros(3), original_score)
    flip = np.diag([-1., -1., 1.])
    opposite = chamfer((query @ best[1] + best[2]) @ flip, template, tree)
    rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(best[1]) - 1) / 2, -1, 1))))
    return dict(rotation_row=best[1].tolist(), translation_row=best[2].tolist(),
                chamfer_before=original_score, chamfer_after=best[0],
                candidate_gap=candidates[1][0] - candidates[0][0],
                opposite_inplane_chamfer=opposite, rotation_deg=rotation_deg,
                query_eigenvalues=query_eig.tolist(), template_eigenvalues=template_eig.tolist())


def normalized_raw(processor, archive, meta):
    raw = processor.read_pcd(archive.read(meta["raw_member"]))
    return ((raw.astype(np.float64) @ np.asarray(meta["basis"], np.float64) -
             np.asarray(meta["oriented_center"], np.float64)) /
            float(meta["normalization_radius"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    args = parser.parse_args()
    processor = processor_from(args.original)
    with ZipFile(args.archive) as archive:
        for category in args.categories:
            base = args.original / "data/pcd_self_v1" / category
            train = sorted(base.glob("train/good/xyz/*/sample.json"))
            test = sorted(base.glob("test/*/xyz/*/sample.json"))
            if len(train) != 4 or len(test) != (53 if category == "chicken" else 50):
                raise ValueError(f"Unexpected sample count: {category}")
            template_meta = json.loads(train[0].read_text())
            template_all = normalized_raw(processor, archive, template_meta)
            template = template_all[sample_indices(len(template_all), 1200, template_meta["raw_member"])]
            for index, path in enumerate(train + test, 1):
                meta = json.loads(path.read_text())
                sample = path.parent.name
                output = args.output / category / meta["split"] / sample / "pose.json"
                if output.exists():
                    old = json.loads(output.read_text())
                    if old["raw_member"] != meta["raw_member"] or old["normal_template"] != template_meta["raw_member"]:
                        raise RuntimeError(f"Existing pose provenance mismatch: {output}")
                    continue
                query_all = normalized_raw(processor, archive, meta)
                query = query_all[sample_indices(len(query_all), 1200, meta["raw_member"])]
                if path == train[0]:
                    result = dict(rotation_row=np.eye(3).tolist(), translation_row=np.zeros(3).tolist(),
                                  chamfer_before=0., chamfer_after=0., candidate_gap=None,
                                  opposite_inplane_chamfer=None, rotation_deg=0.)
                else:
                    result = register(query, template)
                result.update(category=category, split=meta["split"], sample=sample,
                              raw_member=meta["raw_member"], normal_template=template_meta["raw_member"],
                              sample_budget=1200, algorithm="PCA24_trimmed_chamfer_ICP7_v1")
                output.parent.mkdir(parents=True, exist_ok=True)
                temp = output.with_suffix(".json.partial")
                temp.write_text(json.dumps(result, indent=2) + "\n")
                temp.replace(output)
                if index % 10 == 0 or index == len(train) + len(test):
                    print(f"{category}: {index}/{len(train) + len(test)} poses", flush=True)


if __name__ == "__main__":
    main()
