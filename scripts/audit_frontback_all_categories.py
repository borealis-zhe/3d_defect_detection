"""Screen all categories for train/test camera-side mismatch using rendered views."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from repro.bundle_ablate import CATEGORIES, ROOT
from repro.pose_registration import DATA, DEFAULT_OUTPUT
from repro.pose_render import original_preprocessor, render_one
from repro.raw_coverage import RAW_ZIP


OUT = ROOT / "outputs" / "frontback_allclass_audit"


def side_angle(pose: dict) -> float:
    rotation = np.asarray(pose["rotation_row"])
    return float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1, 1))))


def image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("L").resize((224, 224))).copy()


def best_similarity(source: np.ndarray, templates: list[np.ndarray]) -> float:
    # Center crop reduces dominance of the common white canvas.
    crop = slice(25, 199)
    return max(structural_similarity(source[crop, crop], target[crop, crop],
                                     data_range=255) for target in templates)


def main() -> None:
    processor = original_preprocessor()
    rows = []
    with ZipFile(RAW_ZIP) as archive:
        for category in CATEGORIES:
            pose_dir = DEFAULT_OUTPUT / category
            train_paths = sorted((pose_dir / "train").glob("*/pose.json"))
            test_paths = sorted((pose_dir / "test").glob("*/pose.json"))
            train_sides = [side_angle(json.loads(p.read_text())) for p in train_paths]
            suspects = []
            for pose_path in test_paths:
                pose = json.loads(pose_path.read_text())
                side = side_angle(pose)
                gap = min(abs(side - value) for value in train_sides)
                if gap > 60:
                    suspects.append((pose_path, pose, side, gap))
            templates = []
            if suspects:
                for path in train_paths:
                    name = path.parent.name
                    target = OUT / "v1" / category / "train" / name / "view_000.png"
                    if not target.exists():
                        render_one(category, "train", name, (0,), OUT / "v1",
                                   archive, processor)
                    templates.append(image(target))
            for path, pose, side, gap in suspects:
                name = path.parent.name
                target = OUT / "v1" / category / "test" / name / "view_000.png"
                if not target.exists():
                    render_one(category, "test", name, (0,), OUT / "v1",
                               archive, processor)
                original = list((DATA / category / "test").glob(
                    f"*/xyz/{name}/view_000.png"))
                if len(original) != 1:
                    raise FileNotFoundError(f"Expected one original image for {category}/{name}")
                before = best_similarity(image(original[0]), templates)
                after = best_similarity(image(target), templates)
                rows.append(dict(category=category, sample=name,
                                 train_side_degrees=json.dumps([round(x, 2) for x in train_sides]),
                                 test_side_degrees=round(side, 3),
                                 nearest_train_side_gap=round(gap, 3),
                                 original_best_ssim=round(before, 5),
                                 pose_v1_best_ssim=round(after, 5),
                                 ssim_change=round(after - before, 5),
                                 chamfer_before=pose["chamfer_before"],
                                 chamfer_after=pose["chamfer_after"]))
            print(f"{category}: {len(suspects)} side outliers", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "all_category_side_outliers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for category in CATEGORIES:
        subset = [r for r in rows if r["category"] == category]
        if subset:
            print(category, len(subset), "appearance drops >.05:",
                  sum(r["ssim_change"] < -.05 for r in subset),
                  "median change:", round(float(np.median([r["ssim_change"]
                                                             for r in subset])), 3))


if __name__ == "__main__":
    main()
