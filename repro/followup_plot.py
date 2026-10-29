"""Plot point-domain sensitivity and geometry memory ablations."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .bundle_ablate import ROOT


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    old = read_rows(ROOT / "outputs" / "geometry_memory_metrics.csv")
    new = read_rows(ROOT / "outputs" / "full_surface_metrics.csv")
    radius = read_rows(ROOT / "outputs" / "geometry_radius_metrics.csv")
    categories = sorted({r["category"] for r in old} - {"macro"})
    old_ap = np.array([float(next(r for r in old if r["category"] == c and
                                  r["method"] == "dinomaly_baseline")["point_ap"])
                       for c in categories])
    new_ap = np.array([float(next(r for r in new if r["category"] == c and
                                  r["method"] == "uniform15")["point_ap"])
                       for c in categories])
    methods = [
        ("Dinomaly", old, "dinomaly_baseline"),
        ("Position memory", old, "geometry_position"),
        ("kNN shape memory", old, "geometry_shape"),
        ("Radius shape memory", radius, "geometry_shape"),
        ("Dinomaly + kNN shape", old, "dinomaly_plus_shape"),
        ("Dinomaly + radius shape", radius, "dinomaly_plus_shape"),
    ]
    values = [float(next(r for r in rows if r["category"] == "macro" and
                         r["method"] == key)["point_ap"]) for _, rows, key in methods]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2),
                             constrained_layout=True, width_ratios=(1.2, 1))
    y = np.arange(len(categories))
    axes[0].hlines(y, old_ap, new_ap, color="0.72", linewidth=1.4)
    axes[0].scatter(old_ap, y, color="#2b6cb0", s=30, label="Front references")
    axes[0].scatter(new_ap, y, color="#dd6b20", s=30,
                    label="Visible raw points")
    axes[0].set_yticks(y, categories)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Point AP (%)")
    axes[0].set_title("Same model, different point domains")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="x", alpha=.25)
    colors = ["#2b6cb0", "#a0aec0", "#a0aec0", "#a0aec0",
              "#dd6b20", "#dd6b20"]
    bars = axes[1].barh(np.arange(len(methods)), values, color=colors)
    axes[1].set_yticks(np.arange(len(methods)), [x[0] for x in methods])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 50)
    axes[1].set_xlabel("Macro Point AP on front references (%)")
    axes[1].set_title("Normal-only geometry memory")
    axes[1].grid(axis="x", alpha=.25)
    for bar, value in zip(bars, values):
        axes[1].text(value + .6, bar.get_y() + bar.get_height() / 2,
                     f"{value:.1f}", va="center", fontsize=8)
    fig.suptitle("Real3D-AD: evaluation domain and geometry ablations", fontsize=12)
    output = ROOT / "outputs" / "evaluation_geometry.png"
    fig.savefig(output, dpi=180)
    print(output, flush=True)


if __name__ == "__main__":
    main()
