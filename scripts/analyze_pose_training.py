"""Paired local audit of the original and pose-registered Real3D-AD predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/handoff/results/dinomaly_mv_v3/eval"
NEW = ROOT / "data/pose_mv_v1/eval"
POSE = ROOT / "outputs/pose_registration"
OUT = ROOT / "outputs/pose_training_analysis"
CATEGORIES = ("airplane", "candybar", "car", "chicken", "diamond", "duck",
              "fish", "gemstone", "seahorse", "shell", "starfish", "toffees")


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    valid = y >= 0
    y, s = y[valid], s[valid]
    positives = int((y == 1).sum())
    if not positives or positives == len(y):
        return dict(n=len(y), positive=positives, auroc=float("nan"), ap=float("nan"),
                    precision_at_p=float("nan"))
    k = positives
    top = np.argpartition(s, -k)[-k:]
    return dict(n=len(y), positive=positives,
                auroc=float(roc_auc_score(y, s)),
                ap=float(average_precision_score(y, s)),
                precision_at_p=float((y[top] == 1).mean()))


def load_predictions(category: str, root: Path) -> dict:
    result = {}
    for path in sorted((root / category).glob("sample_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sample_id = str(data["sample_id"])
            if sample_id in result:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            result[sample_id] = {
                "scores": data["scores"].astype(np.float32),
                "labels": data["labels"].astype(np.int8),
                "raw_ids": data["raw_point_ids"].astype(np.int64),
                "xyz": data["xyz"].astype(np.float32),
            }
    return result


def common_points(old: dict, new: dict) -> tuple[np.ndarray, np.ndarray]:
    ids, oi, ni = np.intersect1d(old["raw_ids"], new["raw_ids"], return_indices=True)
    if not np.array_equal(old["labels"][oi], new["labels"][ni]):
        raise ValueError("GT labels differ on identical raw points")
    return oi, ni


def rank01(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = (np.arange(len(scores)) + .5) / len(scores)
    return ranks


def score_panel(ax, xyz: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                title: str, class_scores: np.ndarray, top_threshold: float) -> None:
    rank = np.searchsorted(class_scores, scores, side="right") / len(class_scores)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rank, s=3, cmap="viridis", vmin=0, vmax=1,
               linewidths=0, rasterized=True)
    positive = labels == 1
    ax.scatter(xyz[positive, 0], xyz[positive, 1], facecolors="none",
               edgecolors="#e31a1c", s=35, linewidths=.8)
    if np.isfinite(top_threshold):
        false = np.flatnonzero((labels == 0) & (scores >= top_threshold))
        ax.scatter(xyz[false, 0], xyz[false, 1], marker="x", color="#ff7f00",
                   s=20, linewidths=.7, label=f"{len(false)} high-score false positives")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def make_sample_figure(category: str, sample_id: str, old: dict, new: dict, path: Path) -> None:
    sample = sample_id.split("/")[-1]
    pose = json.loads((POSE / category / "test" / sample / "pose.json").read_text())
    rotation = np.asarray(pose["rotation_row"])
    translation = np.asarray(pose["translation_row"])
    old_xyz = old["xyz"].astype(np.float64) @ rotation + translation
    new_xyz = new["xyz"]
    old_m = metrics(old["labels"], old["scores"])
    new_m = metrics(new["labels"], new["scores"])
    class_context = []
    for root in (OLD, NEW):
        predictions = load_predictions(category, root)
        all_labels = np.concatenate([item["labels"] for item in predictions.values()])
        all_scores = np.concatenate([item["scores"] for item in predictions.values()])
        valid = all_labels >= 0
        all_labels, all_scores = all_labels[valid], all_scores[valid]
        k = int((all_labels == 1).sum())
        threshold = float(np.partition(all_scores, -k)[-k])
        class_context.append((np.sort(all_scores), threshold))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(left=.02, right=.98, bottom=.11, top=.84, wspace=.14)
    axes[0].scatter(new_xyz[:, 0], new_xyz[:, 1], color="#c9c9c9", s=3,
                    linewidths=0, rasterized=True)
    positive = new["labels"] == 1
    axes[0].scatter(new_xyz[positive, 0], new_xyz[positive, 1], color="#e31a1c", s=14)
    axes[0].set_title(f"GT on aligned front points\n{positive.sum()} anomaly points", fontsize=10)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    old_label = f"AP {old_m['ap']:.3f}" if np.isfinite(old_m["ap"]) else "normal object"
    new_label = f"AP {new_m['ap']:.3f}" if np.isfinite(new_m["ap"]) else "normal object"
    score_panel(axes[1], old_xyz, old["labels"], old["scores"],
                f"Original model/view | {old_label}\n{old_m['n']} points, {old_m['positive']} positive",
                *class_context[0])
    score_panel(axes[2], new_xyz, new["labels"], new["scores"],
                f"Pose-trained model/view | {new_label}\n{new_m['n']} points, {new_m['positive']} positive",
                *class_context[1])
    fig.suptitle(f"{sample_id}: score ranks in the same aligned coordinates", fontsize=12)
    fig.text(.5, .025, "Red = labeled anomalies; orange crosses = false positives in the class-wide top K (K = number of positives); color = class-wide score percentile",
             ha="center", fontsize=8.5)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sample_rows, caches = [], [], {}
    for category in CATEGORIES:
        old = load_predictions(category, OLD)
        new = load_predictions(category, NEW)
        if old.keys() != new.keys():
            raise ValueError(f"Sample ID mismatch in {category}")
        old_y, old_s, new_y, new_s = [], [], [], []
        common_y, common_old_s, common_new_s = [], [], []
        object_y, object_old_s, object_new_s = [], [], []
        common_obj_y, common_obj_old, common_obj_new = [], [], []
        for sample_id in sorted(old):
            a, b = old[sample_id], new[sample_id]
            oi, ni = common_points(a, b)
            old_y.append(a["labels"]); old_s.append(a["scores"])
            new_y.append(b["labels"]); new_s.append(b["scores"])
            common_y.append(a["labels"][oi])
            common_old_s.append(a["scores"][oi])
            common_new_s.append(b["scores"][ni])
            object_y.append(int("/good/" not in sample_id))
            object_old_s.append(float(a["scores"].max()))
            object_new_s.append(float(b["scores"].max()))
            if len(oi):
                common_obj_y.append(object_y[-1])
                common_obj_old.append(float(a["scores"][oi].max()))
                common_obj_new.append(float(b["scores"][ni].max()))
            am, bm = metrics(a["labels"], a["scores"]), metrics(b["labels"], b["scores"])
            acm = metrics(a["labels"][oi], a["scores"][oi])
            bcm = metrics(b["labels"][ni], b["scores"][ni])
            sample_rows.append(dict(category=category, sample_id=sample_id,
                                    old_points=am["n"], new_points=bm["n"], common_points=len(oi),
                                    old_positive=am["positive"], new_positive=bm["positive"],
                                    common_positive=acm["positive"],
                                    old_ap=am["ap"], new_ap=bm["ap"],
                                    old_common_ap=acm["ap"], new_common_ap=bcm["ap"]))
        old_m = metrics(np.concatenate(old_y), np.concatenate(old_s))
        new_m = metrics(np.concatenate(new_y), np.concatenate(new_s))
        old_common = metrics(np.concatenate(common_y), np.concatenate(common_old_s))
        new_common = metrics(np.concatenate(common_y), np.concatenate(common_new_s))
        old_report = json.loads((OLD / category / "metrics.json").read_text())
        new_report = json.loads((NEW / category / "metrics.json").read_text())
        for metric, key in (("auroc", "P_AUROC"), ("ap", "Point_AP")):
            if not np.isclose(old_m[metric], old_report[key], atol=1e-7):
                raise ValueError(f"Old {category} {metric} verification failed")
            if not np.isclose(new_m[metric], new_report[key], atol=1e-7):
                raise ValueError(f"New {category} {metric} verification failed")
        rows.append(dict(category=category, objects=len(old),
                         old_object_auc=roc_auc_score(object_y, object_old_s),
                         new_object_auc=roc_auc_score(object_y, object_new_s),
                         old_common_object_auc=roc_auc_score(common_obj_y, common_obj_old),
                         new_common_object_auc=roc_auc_score(common_obj_y, common_obj_new),
                         common_objects=len(common_obj_y),
                         old_point_auc=old_m["auroc"], new_point_auc=new_m["auroc"],
                         old_ap=old_m["ap"], new_ap=new_m["ap"],
                         old_points=old_m["n"], new_points=new_m["n"],
                         old_positive=old_m["positive"], new_positive=new_m["positive"],
                         common_points=old_common["n"], common_positive=old_common["positive"],
                         old_common_auc=old_common["auroc"], new_common_auc=new_common["auroc"],
                         old_common_ap=old_common["ap"], new_common_ap=new_common["ap"],
                         old_precision_at_p=old_m["precision_at_p"],
                         new_precision_at_p=new_m["precision_at_p"],
                         old_common_precision_at_p=old_common["precision_at_p"],
                         new_common_precision_at_p=new_common["precision_at_p"]))
        caches[category] = dict(old=old, new=new,
                                full=(np.concatenate(old_y), np.concatenate(old_s),
                                      np.concatenate(new_y), np.concatenate(new_s)),
                                common=(np.concatenate(common_y), np.concatenate(common_old_s),
                                        np.concatenate(common_new_s)))
        print(category, f"AP {old_m['ap']:.3f}->{new_m['ap']:.3f}",
              f"common AP {old_common['ap']:.3f}->{new_common['ap']:.3f}", flush=True)
    with (OUT / "category_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader(); writer.writerows(sample_rows)

    x = np.arange(len(rows))
    old_ap = np.array([row["old_ap"] for row in rows]) * 100
    new_ap = np.array([row["new_ap"] for row in rows]) * 100
    fig, ax = plt.subplots(figsize=(10.5, 6.1), constrained_layout=True)
    ax.barh(x + .19, old_ap, height=.36, color="#a6a6a6", label="Original baseline")
    ax.barh(x - .19, new_ap, height=.36, color="#1f78b4", label="Pose-aligned retraining")
    ax.set_yticks(x, [row["category"] for row in rows]); ax.invert_yaxis()
    ax.set_xlabel("Point AP (%) on each method's own front-point domain")
    ax.set_title("Category-wise Point AP: large gains and regressions coexist")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(OUT / "figures" / "category_point_ap.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    delta_auc = 100 * np.array([row["new_point_auc"] - row["old_point_auc"] for row in rows])
    delta_ap = 100 * np.array([row["new_ap"] - row["old_ap"] for row in rows])
    ax.axhline(0, color="#777", linewidth=.8); ax.axvline(0, color="#777", linewidth=.8)
    ax.scatter(delta_auc, delta_ap, color="#1f78b4", s=45)
    for i, row in enumerate(rows):
        ax.annotate(row["category"], (delta_auc[i], delta_ap[i]), xytext=(5, 4),
                    textcoords="offset points", fontsize=9)
    ax.set(xlabel="Change in point AUROC (percentage points)",
           ylabel="Change in Point AP (percentage points)",
           title="AUROC and AP respond differently across categories")
    ax.grid(alpha=.2)
    fig.savefig(OUT / "figures" / "auc_ap_changes.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for category, ax in zip(("fish", "chicken"), axes):
        common_y, old_common_s, new_common_s = caches[category]["common"]
        known = common_y >= 0
        for label, scores, color in (("Old on common points", old_common_s, "#777777"),
                                     ("New on common points", new_common_s, "#1f78b4")):
            precision, recall, _ = precision_recall_curve(common_y[known], scores[known])
            ax.plot(recall, precision, label=label, color=color, linewidth=1.5)
        ax.set(xlabel="Recall", ylabel="Precision", title=f"{category}: same raw point IDs")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(OUT / "figures" / "common_point_pr.png", dpi=220)
    plt.close(fig)

    fish_cases = [row for row in sample_rows if row["category"] == "fish" and
                  row["old_positive"] >= 20 and row["new_positive"] >= 20 and
                  np.isfinite(row["old_ap"]) and np.isfinite(row["new_ap"])]
    fish_cases.sort(key=lambda row: row["new_ap"] - row["old_ap"])
    if fish_cases:
        sample_id = fish_cases[0]["sample_id"]
        make_sample_figure("fish", sample_id, caches["fish"]["old"][sample_id],
                           caches["fish"]["new"][sample_id], OUT / "figures" / "fish_failure_sample.png")
        print("FISH_CASE", sample_id, fish_cases[0]["old_ap"],
              fish_cases[0]["new_ap"], flush=True)
    normal_fish_id = "fish/test/good/213_good"
    make_sample_figure("fish", normal_fish_id, caches["fish"]["old"][normal_fish_id],
                       caches["fish"]["new"][normal_fish_id], OUT / "figures" / "fish_normal_false_positive.png")
    airplane_id = "airplane/test/sink/635_sink"
    make_sample_figure("airplane", airplane_id, caches["airplane"]["old"][airplane_id],
                       caches["airplane"]["new"][airplane_id], OUT / "figures" / "airplane_635_sample.png")
    print("OUTPUT", OUT, flush=True)


if __name__ == "__main__":
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    main()
