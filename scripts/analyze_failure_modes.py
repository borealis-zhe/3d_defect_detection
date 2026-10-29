"""Visual diagnosis of Dinomaly-MV metric gaps and failure modes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from repro.bundle_ablate import CATEGORIES, HANDOFF, ROOT, records


EVAL = HANDOFF / "results" / "dinomaly_mv_v3" / "eval"
FULL = ROOT / "outputs" / "full_surface"
OUT = ROOT / "outputs" / "failure_analysis"


def load_category(category: str):
    items = []
    for path in sorted((EVAL / category).glob("sample_*.npz")):
        with np.load(path, allow_pickle=False) as d:
            items.append({k: d[k].copy() for k in d.files})
    return items


def point_stats(y, score):
    known = y >= 0
    y, score = y[known], score[known]
    precision, recall, thresholds = precision_recall_curve(y, score)
    def precision_at(target):
        eligible = np.flatnonzero(recall >= target)
        return float(precision[eligible].max()) if len(eligible) else 0.0
    top_n = int((y == 1).sum())
    top_ids = np.argpartition(score, -top_n)[-top_n:] if top_n else np.array([], int)
    return dict(point_auroc=roc_auc_score(y, score),
                point_ap=average_precision_score(y, score),
                prevalence=float(y.mean()),
                precision_r50=precision_at(0.5),
                precision_r80=precision_at(0.8),
                precision_top_equal=float(y[top_ids].mean()) if top_n else 0.0,
                points=len(y), positives=top_n)


def collect():
    rows, data = [], {}
    all_y, all_s = [], []
    for category in CATEGORIES:
        items = load_category(category)
        y = np.concatenate([d["labels"] for d in items])
        score = np.concatenate([d["scores"] for d in items])
        stats = point_stats(y, score)
        obj_y = np.asarray([int("/good/" not in str(d["sample_id"])) for d in items])
        obj_s = np.asarray([float(d["scores"].max()) for d in items])
        stats["object_auroc"] = float(roc_auc_score(obj_y, obj_s))
        rows.append(dict(category=category, **stats))
        data[category] = dict(items=items, y=y, score=score,
                              obj_y=obj_y, obj_s=obj_s)
        known = y >= 0
        all_y.append(y[known]); all_s.append(score[known])
    y = np.concatenate(all_y); score = np.concatenate(all_s)
    micro = point_stats(y, score)
    macro = {key: float(np.mean([r[key] for r in rows])) for key in
             ("object_auroc", "point_auroc", "point_ap", "prevalence",
              "precision_r50", "precision_r80", "precision_top_equal")}
    macro.update(points=int(sum(r["points"] for r in rows)),
                 positives=int(sum(r["positives"] for r in rows)))
    return rows, data, micro, macro


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18, linewidth=.7)


def diagnostic_figure(rows, data, micro, macro):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), dpi=180,
                             constrained_layout=True)
    fig.suptitle("Why Point AP is much lower than point AUROC",
                 fontsize=22, fontweight="bold", color="#172a3a")
    ax = axes[0, 0]
    order = np.arange(len(rows))
    width = .25
    ax.bar(order-width, [100*r["object_auroc"] for r in rows], width,
           label="Object AUROC", color="#4169a1")
    ax.bar(order, [100*r["point_auroc"] for r in rows], width,
           label="Point AUROC", color="#47a3a2")
    ax.bar(order+width, [100*r["point_ap"] for r in rows], width,
           label="Point AP", color="#e67e52")
    ax.set_xticks(order, [r["category"] for r in rows], rotation=42, ha="right")
    ax.set_ylim(0, 105); ax.set_ylabel("Metric (%)")
    ax.set_title("A  Per-category performance is highly uneven", loc="left",
                 fontsize=14, fontweight="bold")
    ax.legend(ncols=3, fontsize=9, loc="upper left")
    style_ax(ax)

    ax = axes[0, 1]
    prevalence = np.asarray([100*r["prevalence"] for r in rows])
    ap = np.asarray([100*r["point_ap"] for r in rows])
    auroc = np.asarray([100*r["point_auroc"] for r in rows])
    sc = ax.scatter(prevalence, ap, c=auroc, cmap="viridis", s=95,
                    edgecolor="white", linewidth=.8)
    for r, x, yv in zip(rows, prevalence, ap):
        ax.annotate(r["category"], (x, yv), xytext=(4, 4),
                    textcoords="offset points", fontsize=8.5)
    lim = max(prevalence.max()*1.12, 4)
    ax.plot([0, lim], [0, lim], "--", color="#8795a3", lw=1,
            label="random ranking AP = positive rate")
    ax.set(xlim=(0, lim), ylim=(0, 84), xlabel="Defect-point rate (%)",
           ylabel="Point AP (%)")
    ax.set_title("B  Defect points are rare; high-score false positives dominate AP",
                 loc="left", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.colorbar(sc, ax=ax, label="Point AUROC (%)", fraction=.05, pad=.02)
    style_ax(ax)

    ax = axes[1, 0]
    for category, color in [("airplane", "#ba4a4a"), ("duck", "#dd8b36"),
                            ("starfish", "#8c62aa"), ("candybar", "#3987b8"),
                            ("fish", "#32956b")]:
        y, score = data[category]["y"], data[category]["score"]
        known = y >= 0
        p, r, _ = precision_recall_curve(y[known], score[known])
        apv = average_precision_score(y[known], score[known])
        ax.plot(r, p, lw=2, color=color, label=f"{category}  AP {100*apv:.1f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall: fraction of defects found",
           ylabel="Precision: fraction of alarms that are true")
    ax.set_title("C  Weak classes cannot keep precision when recall increases",
                 loc="left", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    style_ax(ax)

    ax = axes[1, 1]
    categories = list(CATEGORIES)
    pos = np.arange(len(categories))
    normal = [data[c]["obj_s"][data[c]["obj_y"] == 0] for c in categories]
    anomaly = [data[c]["obj_s"][data[c]["obj_y"] == 1] for c in categories]
    bp1 = ax.boxplot(normal, positions=pos-.17, widths=.28, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    bp2 = ax.boxplot(anomaly, positions=pos+.17, widths=.28, patch_artist=True,
                     showfliers=False, manage_ticks=False)
    for box in bp1["boxes"]: box.set(facecolor="#8fb2cf", edgecolor="#466b87")
    for box in bp2["boxes"]: box.set(facecolor="#ef9b76", edgecolor="#9e5134")
    for key in ("whiskers", "caps", "medians"):
        for item in bp1[key]: item.set(color="#466b87", linewidth=1)
        for item in bp2[key]: item.set(color="#9e5134", linewidth=1)
    ax.set_xticks(pos, categories, rotation=42, ha="right")
    ax.set_ylabel("Maximum point score per object")
    ax.set_title("D  Max-point object scores overlap in weak classes",
                 loc="left", fontsize=14, fontweight="bold")
    ax.plot([], [], color="#8fb2cf", lw=7, label="normal objects")
    ax.plot([], [], color="#ef9b76", lw=7, label="anomalous objects")
    ax.legend(fontsize=9, ncols=2, loc="upper left")
    style_ax(ax)

    fig.text(.5, -.012,
             f"Old reference-point domain: {macro['points']:,} labeled points, "
             f"{macro['positives']:,} positive ({100*micro['prevalence']:.2f}%).  "
             f"Macro: object AUROC {100*macro['object_auroc']:.2f}, point AUROC {100*macro['point_auroc']:.2f}, Point AP {100*macro['point_ap']:.2f}.",
             ha="center", fontsize=11, color="#526575")
    path = OUT / "metric_diagnosis.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def projection_axes(xyz):
    spread = np.ptp(xyz, axis=0)
    return np.argsort(spread)[-2:]


def sample_index(category, sample_id):
    paths = records(category)
    parts = str(sample_id).split("/")
    defect, name = parts[-2], parts[-1]
    return next(i for i, path in enumerate(paths)
                if path.parent.name == name and path.parent.parent.parent.name == defect)


def load_full(category, sample_id):
    index = sample_index(category, sample_id)
    with np.load(FULL/category/f"sample_{index:04d}.npz", allow_pickle=False) as d:
        return {k: d[k].copy() for k in d.files}


def choose_examples(data):
    # Strong localization, weak localization, and a high-scoring normal false alarm.
    strong = "candybar/test/bulge/431_bulge"
    weak = "airplane/test/sink/635_sink"
    category = "airplane"
    items = data[category]["items"]
    normals = [d for d in items if "/good/" in str(d["sample_id"])]
    false_alarm = max(normals, key=lambda d: float(d["scores"].max()))
    return [("Successful localization", "candybar", strong),
            ("Missed / poorly ranked defect", "airplane", weak),
            ("Normal object with false hot spots", category,
             str(false_alarm["sample_id"]))]


def examples_figure(data):
    examples = choose_examples(data)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), dpi=180,
                             constrained_layout=True)
    fig.suptitle("Concrete localization outcomes on the original reference-point domain",
                 fontsize=21, fontweight="bold", color="#172a3a")
    details = []
    for row, (label, category, sid) in enumerate(examples):
        full = load_full(category, sid)
        xyz = None
        index = sample_index(category, sid)
        meta = records(category)[index].parent
        # Use the exact reference-point coordinates on which the baseline metrics
        # were computed; full-surface counts are retained only for the audit table.
        with np.load(EVAL/category/f"sample_{index:04d}.npz", allow_pickle=False) as pred:
            xyz = pred["xyz"].copy(); y = pred["labels"].copy(); s = pred["scores"].copy()
        known = y >= 0
        axes_id = projection_axes(xyz)
        x, z = xyz[:, axes_id[0]], xyz[:, axes_id[1]]
        ax = axes[row, 0]
        ax.scatter(x[y == 0], z[y == 0], s=4, color="#aab6c1", alpha=.48,
                   rasterized=True)
        if np.any(y == 1):
            ax.scatter(x[y == 1], z[y == 1], s=14, color="#e64b4b", alpha=.95,
                       label="GT defect", rasterized=True)
            ax.legend(fontsize=8, loc="best")
        ax.set_title(f"{label}\nGT — {sid}", fontsize=11.5, fontweight="bold")
        ax.set_aspect("equal"); ax.set_axis_off()
        ax = axes[row, 1]
        vmax = float(np.quantile(s, .99))
        sc = ax.scatter(x, z, s=6, c=s, cmap="viridis", vmin=0, vmax=vmax,
                        rasterized=True)
        if np.any(y == 1):
            ax.scatter(x[y == 1], z[y == 1], s=34, facecolors="none",
                       edgecolors="#ff3b30", linewidths=.8)
        sample_ap = (average_precision_score(y[known], s[known])
                     if np.any(y[known] == 1) else np.nan)
        ax.set_title("Model point scores" +
                     (f" — sample AP {100*sample_ap:.2f}%" if np.isfinite(sample_ap)
                      else f" — max score {s.max():.3f}"),
                     fontsize=11.5, fontweight="bold")
        ax.set_aspect("equal"); ax.set_axis_off()
        fig.colorbar(sc, ax=ax, fraction=.045, pad=.02, label="anomaly score")
        details.append(dict(case=label, category=category, sample_id=sid,
                            reference_points=len(y), positives=int((y == 1).sum()),
                            sample_ap=None if not np.isfinite(sample_ap) else float(sample_ap),
                            max_score=float(s.max()), visible_raw_points=int(full["seen"].sum()),
                            raw_positive_points=int((full["labels"] == 1).sum())))
    fig.text(.5, -.01,
             "Red filled points are ground truth. Red outlines on the score map mark the same defect points. "
             "The false-alarm row is a normal object, so every hot spot is erroneous.",
             ha="center", fontsize=10.5, color="#526575")
    path = OUT / "failure_examples.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, details


def write_outputs(rows, micro, macro, details):
    with (OUT/"per_category_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(micro=micro, macro=macro, examples=details)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, data, micro, macro = collect()
    metric_path = diagnostic_figure(rows, data, micro, macro)
    examples_path, details = examples_figure(data)
    write_outputs(rows, micro, macro, details)
    print(json.dumps(dict(metric_figure=str(metric_path), examples=str(examples_path),
                          micro=micro, macro=macro, cases=details), indent=2))


if __name__ == "__main__":
    main()
