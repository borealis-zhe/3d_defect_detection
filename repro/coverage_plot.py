"""Create a standalone view-budget tradeoff figure from completed CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .coverage_ablate import METHODS
from .bundle_ablate import ROOT


LABELS = {
    "fixed_prefix": "Fixed prefix",
    "fixed_spread": "Fixed spread",
    "normal_greedy_15": "Normal greedy, 15 candidates",
    "normal_greedy_27": "Normal greedy, 27 candidates",
    "object_greedy_27": "Object greedy, 27 candidates",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path,
                        default=ROOT / "outputs" / "coverage_metrics.csv")
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "outputs" / "raw_coverage_proxy.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "coverage_tradeoff.png")
    args = parser.parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as handle:
        metric = {(r["method"], int(r["budget"])): r for r in csv.DictReader(handle)
                  if r["category"] == "macro"}
    raw_aggregate = defaultdict(lambda: [0, 0])
    with args.raw.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            key = (r["method"], int(r["budget"]))
            raw_aggregate[key][0] += int(r["covered_raw_points"])
            raw_aggregate[key][1] += int(r["raw_points"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    budgets = [3, 5, 9, 15]
    for method in METHODS:
        ap = [float(metric[method, k]["point_ap"]) for k in budgets]
        reference = [float(metric[method, k]["coverage_percent"]) for k in budgets]
        raw = [100 * raw_aggregate[method, k][0] /
               raw_aggregate[method, k][1] for k in budgets]
        for ax, values in zip(axes, [ap, reference, raw]):
            ax.plot(budgets, values, marker="o", linewidth=1.8,
                    markersize=4.5, label=LABELS[method])
    for ax, title, ylabel in zip(
            axes,
            ["Localization", "Front reference coverage", "Raw geometry proxy"],
            ["Macro Point AP (%)", "Covered reference points (%)",
             "Geometrically visible raw points (%)"]):
        ax.set_title(title)
        ax.set_xlabel("Network views per object")
        ax.set_ylabel(ylabel)
        ax.set_xticks(budgets)
        ax.grid(alpha=.25)
    axes[0].legend(loc="best", fontsize=7.5)
    fig.suptitle("Frozen Dinomaly-MV: coverage versus detection", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
