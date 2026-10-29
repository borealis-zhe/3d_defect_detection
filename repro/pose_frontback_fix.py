"""Resolve ambiguous front/back poses against normal training orientations.

This is a label-blind correction of saved v1 poses. It writes a separate pose
tree, retaining the original outputs for comparison. Only categories explicitly
requested by the caller are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from .pose_registration import (
    AXIS_MAPS, DATA, DEFAULT_OUTPUT, chamfer, icp, normalized_raw,
    pca_frame, sample_indices,
)
from .raw_coverage import RAW_ZIP


DEFAULT_FIXED = DEFAULT_OUTPUT.parent / "pose_registration_frontback_v2"


def side_angle(rotation: np.ndarray) -> float:
    """Camera-side change in degrees, independent of in-plane rotation."""
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))


def front_candidate(query: np.ndarray, template: np.ndarray,
                    max_side_deg: float = 30.0, refine_top: int = 4) -> dict:
    """Best near-side rigid pose, including the identity alternative."""
    qframe, _ = pca_frame(query)
    tframe, _ = pca_frame(template)
    tree = cKDTree(template)
    starts = []
    for axis in AXIS_MAPS:
        rotation = qframe @ axis @ tframe.T
        if np.linalg.det(rotation) > 0 and side_angle(rotation) <= max_side_deg:
            starts.append((chamfer(query @ rotation, template, tree), rotation))
    starts.sort(key=lambda item: item[0])
    best = (chamfer(query, template, tree), np.eye(3), np.zeros(3))
    for _, initial in starts[:refine_top]:
        try:
            rotation, translation = icp(query, template, tree, initial)
        except ValueError:
            continue
        if side_angle(rotation) > max_side_deg:
            continue
        score = chamfer(query @ rotation + translation, template, tree)
        if score < best[0]:
            best = (score, rotation, translation)
    return {"chamfer": best[0], "rotation": best[1],
            "translation": best[2], "starts_considered": len(starts)}


def correct_category(category: str, archive: ZipFile, source: Path,
                     output: Path, *, max_train_side_deg: float = 15.0,
                     max_front_side_deg: float = 30.0,
                     max_relative_cost: float = 1.10) -> list[dict]:
    training = sorted((source / category / "train").glob("*/pose.json"))
    if len(training) != 4:
        raise ValueError(f"Expected four normal training poses for {category}")
    training_sides = [side_angle(np.asarray(json.loads(p.read_text())["rotation_row"]))
                      for p in training]
    if max(training_sides) > max_train_side_deg:
        raise ValueError(f"Normal training sides disagree for {category}: {training_sides}")
    first = json.loads(training[0].read_text())
    template_meta = json.loads(next((DATA / category / "train").glob(
        f"*/xyz/{Path(first['normal_template']).stem}/sample.json")).read_text())
    template_all = normalized_raw(archive, template_meta)
    template = template_all[sample_indices(len(template_all), 1200,
                                          template_meta["raw_member"])]
    rows = []
    for path in sorted((source / category / "test").glob("*/pose.json")):
        old = json.loads(path.read_text(encoding="utf-8"))
        destination = output / category / "test" / path.parent.name / "pose.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        new = dict(old)
        old_side = side_angle(np.asarray(old["rotation_row"]))
        new_side = old_side
        front_cost = None
        changed = False
        if old_side > 90.0:
            sample_meta = json.loads(next((DATA / category / "test").glob(
                f"*/xyz/{path.parent.name}/sample.json")).read_text())
            query_all = normalized_raw(archive, sample_meta)
            query = query_all[sample_indices(len(query_all), 1200,
                                             sample_meta["raw_member"])]
            front = front_candidate(query, template, max_front_side_deg)
            front_cost = front["chamfer"]
            if front_cost <= old["chamfer_after"] * max_relative_cost:
                rotation, translation = front["rotation"], front["translation"]
                new.update(rotation_row=rotation.tolist(),
                           translation_row=translation.tolist(),
                           chamfer_after=front_cost,
                           rotation_deg=float(np.degrees(np.arccos(np.clip(
                               (np.trace(rotation) - 1) / 2, -1, 1)))),
                           algorithm="PCA24_trimmed_chamfer_ICP7_front_side_prior_v2",
                           frontback_fix={
                               "old_pose": str(path), "old_side_deg": old_side,
                               "front_side_deg": side_angle(rotation),
                               "old_chamfer": old["chamfer_after"],
                               "front_chamfer": front_cost,
                               "max_relative_cost": max_relative_cost,
                               "selection_basis": "normal training side consistency; no labels"})
                new_side = side_angle(rotation)
                changed = True
        destination.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        rows.append(dict(category=category, sample=path.parent.name,
                         old_side_deg=old_side, new_side_deg=new_side,
                         old_chamfer=old["chamfer_after"], front_chamfer=front_cost,
                         changed=int(changed)))
    # The normal training poses are unchanged and remain available to renderers.
    for path in training:
        destination = output / category / "train" / path.parent.name / "pose.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["fish"])
    parser.add_argument("--archive", type=Path, default=RAW_ZIP)
    parser.add_argument("--source", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXED)
    args = parser.parse_args()
    rows = []
    with ZipFile(args.archive) as archive:
        for category in args.categories:
            rows.extend(correct_category(category, archive, args.source, args.output))
    with (args.output / "frontback_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(r['changed'] for r in rows)}/{len(rows)} test poses corrected; "
          f"saved in {args.output}")


if __name__ == "__main__":
    main()
