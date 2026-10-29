"""Plot the controlled point-selection ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from repro.bundle_ablate import ROOT, records
from repro.full_surface import normalized_raw
from repro.raw_coverage import RAW_ZIP, read_xyz


OUT = ROOT/"outputs"/"selection_study"
LABELS = {
    "front112": "Single-front 112 grid",
    "random_visible": "Random visible",
    "voxel3d_visible": "3D voxel-stratified",
    "six_side_visible": "Six-side surface",
    "all_visible": "All visible raw points",
}


def read_metrics():
    with (OUT/"selection_metrics.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_plot(rows):
    macro = {r["method"]: r for r in rows if r["category"] == "macro"}
    methods = list(LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=180,
                             constrained_layout=True)
    x = np.arange(len(methods))
    names = [LABELS[m] for m in methods]
    ax = axes[0]
    width = .35
    ax.bar(x-width/2, [float(macro[m]["point_auroc"]) for m in methods],
           width, color="#4d91a6", label="Point AUROC")
    ax.bar(x+width/2, [float(macro[m]["point_ap"]) for m in methods],
           width, color="#e58355", label="Point AP")
    ax.set_ylim(0, 100); ax.set_ylabel("Macro metric (%)")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_title("A  Same frozen scores, different point domains", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    zeros = [int(macro[m]["positive_objects_zero_selected"]) for m in methods]
    bars = ax.bar(x, zeros, color=["#c94f45", "#55a777", "#d7a13e",
                                  "#55a777", "#55a777"])
    ax.bar_label(bars, fontsize=11)
    ax.set_ylim(0, max(21, max(zeros)*1.18))
    ax.set_ylabel("Anomalous objects with zero selected defect points")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_title("B  The old domain entirely loses 19 defects", loc="left",
                 fontweight="bold")

    ax = axes[2]
    counts = [int(macro[m]["points"]) for m in methods]
    bars = ax.bar(x, np.asarray(counts)/1e6, color="#657c9a")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                f"{count/1e6:.2f}M", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Evaluated points (millions)")
    ax.set_xticks(x, names, rotation=30, ha="right")
    ax.set_title("C  Four methods share the old point budget", loc="left",
                 fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Point-selection ablation on Dinomaly-MV",
                 fontsize=19, fontweight="bold", color="#172a3a")
    path = OUT/"selection_metrics.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def sample_plot():
    category = "airplane"
    sid = "airplane/test/sink/635_sink"
    paths = records(category)
    index = next(i for i, p in enumerate(paths) if p.parent.name == "635_sink")
    meta_path = paths[index]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with ZipFile(RAW_ZIP) as archive:
        xyz = normalized_raw(read_xyz(archive.read(meta["raw_member"])), meta)
    with np.load(ROOT/"outputs"/"full_surface"/category/f"sample_{index:04d}.npz",
                 allow_pickle=False) as full:
        labels = full["labels"].copy(); seen = full["seen"].copy()
    with np.load(OUT/"selections"/category/f"sample_{index:04d}.npz",
                 allow_pickle=False) as selection:
        selected = {k: selection[k].copy() for k in selection.files}
    axes_id = np.argsort(np.ptp(xyz, axis=0))[-2:]
    px, py = xyz[:, axes_id[0]], xyz[:, axes_id[1]]
    rng = np.random.default_rng(20261028)
    background = np.flatnonzero(seen & (labels == 0))
    background = rng.choice(background, min(24000, len(background)), replace=False)
    positive = np.flatnonzero(seen & (labels == 1))
    methods = ["all_visible", "front112", "random_visible",
               "voxel3d_visible", "six_side_visible"]
    titles = ["Visible raw GT (no sampling)", *[LABELS[m] for m in methods[1:]]]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.8), dpi=180,
                             constrained_layout=True)
    for ax, method, title in zip(axes, methods, titles):
        ax.scatter(px[background], py[background], s=.35, color="#c0cad3",
                   alpha=.25, rasterized=True)
        if method == "all_visible":
            ids = np.flatnonzero(seen)
        else:
            ids = selected[method]
            ax.scatter(px[ids[labels[ids] == 0]], py[ids[labels[ids] == 0]],
                       s=3, color="#426b91", alpha=.72, rasterized=True)
        pos = ids[labels[ids] == 1]
        ax.scatter(px[pos], py[pos], s=12 if method != "all_visible" else 1.8,
                   color="#e54343", alpha=.9, rasterized=True)
        ax.set_title(f"{title}\n{len(ids):,} points; {len(pos):,} defect",
                     fontsize=10.5, fontweight="bold")
        ax.set_aspect("equal"); ax.set_axis_off()
    fig.suptitle("airplane/635_sink: label-blind selections at the same 1,211-point budget",
                 fontsize=18, fontweight="bold", color="#172a3a")
    fig.text(.5, -.015,
             "Light gray = visible raw surface; blue = selected normal points; red = selected defect points. "
             "Selection never uses the red labels.",
             ha="center", fontsize=10.5, color="#526575")
    path = OUT/"airplane_selection_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    rows = read_metrics()
    print(metric_plot(rows))
    print(sample_plot())


if __name__ == "__main__":
    main()
