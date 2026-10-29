"""Compare fixed camera views before and after per-object pose registration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.bundle_ablate import CATEGORIES
from repro.pose_registration import DATA, DEFAULT_OUTPUT
from repro.pose_render import original_preprocessor, render_one
from repro.raw_coverage import RAW_ZIP


def main() -> None:
    font_names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [next((f for f in ("Microsoft YaHei", "SimHei") if f in font_names), "DejaVu Sans")]
    render_root = ROOT / "outputs/pose_render"
    processor = original_preprocessor()
    cases = []
    with ZipFile(RAW_ZIP) as archive:
        for category in CATEGORIES:
            paths = sorted((DATA / category).glob("train/good/xyz/*/sample.json"))
            if len(paths) != 4:
                raise ValueError(f"Expected four normal prototypes for {category}")
            reference, heldout = paths[0], paths[1]
            name = heldout.parent.name
            destination = render_root / category / "train" / name / "view_000.png"
            if not destination.exists():
                render_one(category, "train", name, (0,), render_root, archive, processor)
            pose = json.loads((DEFAULT_OUTPUT / category / "train" / name / "pose.json").read_text())
            cases.append((category, reference.parent / "view_000.png",
                          heldout.parent / "view_000.png", destination, pose["rotation_deg"]))

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    outer = fig.add_gridspec(4, 3)
    for i, (category, template, original, aligned, degrees) in enumerate(cases):
        inner = outer[i // 3, i % 3].subgridspec(1, 3, wspace=.015)
        for j, (path, label) in enumerate(((template, "类别模板"),
                                          (original, "原主视角"),
                                          (aligned, "配准后主视角"))):
            ax = fig.add_subplot(inner[0, j])
            ax.imshow(Image.open(path))
            ax.set_title(label, fontsize=8, color="#0072B2" if j == 2 else "#444444")
            ax.axis("off")
        ax.text(-1.48, 1.12, f"{category}｜配准旋转 {degrees:.0f}°", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.suptitle("12类正常训练物体：同一台0号相机，逐物体配准前后", fontsize=17, fontweight="bold")
    output = ROOT / "outputs/pose_registration/normal_view_comparison.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
