"""Visual audit of Dinomaly-MV scores for the Real3D-AD airplane class."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro.raw_coverage import RAW_ZIP, read_xyz


OLD = ROOT / "data/handoff/results/dinomaly_mv_v3/eval/airplane"
FULL = ROOT / "outputs/full_surface/airplane"
OUT = ROOT / "outputs/airplane_analysis"


def set_style() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [next((x for x in candidates if x in available), "DejaVu Sans")]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})


def load_rows() -> list[dict]:
    rows = []
    for old_path in sorted(OLD.glob("sample_*.npz")):
        full_path = FULL / old_path.name
        old = np.load(old_path)
        full = np.load(full_path)
        sid = str(old["sample_id"])
        is_anomaly = "/good/" not in sid
        labels = old["labels"].astype(np.int8)
        scores = old["scores"].astype(np.float32)
        raw_labels = full["labels"].astype(np.int8)
        seen = full["seen"].astype(bool) & (raw_labels >= 0)
        raw_scores = full["uniform_scores"].astype(np.float32)
        vis_scores = full["visible_scores"].astype(np.float32)
        rows.append({
            "path": old_path,
            "full_path": full_path,
            "sid": sid,
            "name": sid.rsplit("/", 1)[-1],
            "kind": sid.split("/")[-2],
            "is_anomaly": is_anomaly,
            "labels": labels,
            "scores": scores,
            "object_score": float(scores.max()),
            "front_pos": int(labels.sum()),
            "front_ap": float(average_precision_score(labels, scores)) if labels.sum() else np.nan,
            "raw_labels": raw_labels,
            "seen": seen,
            "raw_scores": raw_scores,
            "vis_scores": vis_scores,
            "raw_pos": int(np.sum(raw_labels[seen] == 1)),
            "raw_ap": float(average_precision_score(raw_labels[seen], raw_scores[seen])) if np.any(raw_labels[seen] == 1) else np.nan,
            "vis_ap": float(average_precision_score(raw_labels[seen], vis_scores[seen])) if np.any(raw_labels[seen] == 1) else np.nan,
        })
    return rows


def overview(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    normal = [r for r in rows if not r["is_anomaly"]]
    anomaly = [r for r in rows if r["is_anomaly"]]
    y_obj = np.array([r["is_anomaly"] for r in rows], dtype=int)
    s_obj = np.array([r["object_score"] for r in rows])
    front_y = np.concatenate([r["labels"] for r in rows])
    front_s = np.concatenate([r["scores"] for r in rows])
    raw_y = np.concatenate([r["raw_labels"][r["seen"]] for r in rows])
    raw_s = np.concatenate([r["raw_scores"][r["seen"]] for r in rows])
    vis_s = np.concatenate([r["vis_scores"][r["seen"]] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

    ax = axes[0, 0]
    order = np.argsort(s_obj)[::-1]
    for rank, idx in enumerate(order):
        r = rows[idx]
        color = "#D55E00" if r["is_anomaly"] else "#0072B2"
        marker = "X" if r["is_anomaly"] else "o"
        ax.scatter(rank, r["object_score"], c=color, marker=marker, s=48, zorder=3)
    ax.axhline(np.median([r["object_score"] for r in normal]), color="#0072B2", ls="--", lw=1)
    ax.axhline(np.median([r["object_score"] for r in anomaly]), color="#D55E00", ls="--", lw=1)
    ax.set(xlabel="按物体异常分数从高到低排序（50 个测试物体）", ylabel="物体分数 = 前表面参考点最大分数",
           title=f"A. 物体级排序反转：AUROC = {roc_auc_score(y_obj, s_obj)*100:.2f}%")
    ax.scatter([], [], c="#0072B2", marker="o", label="正常 25 个")
    ax.scatter([], [], c="#D55E00", marker="X", label="缺陷 25 个")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    for r in sorted(normal, key=lambda x: x["object_score"], reverse=True)[:1] + sorted(anomaly, key=lambda x: x["object_score"])[:1]:
        rank = int(np.where(order == rows.index(r))[0][0])
        ax.annotate(r["name"], (rank, r["object_score"]), xytext=(5, 8), textcoords="offset points", fontsize=8)

    ax = axes[0, 1]
    vals = [[r["object_score"] for r in normal], [r["object_score"] for r in anomaly]]
    bp = ax.boxplot(vals, tick_labels=["正常物体", "缺陷物体"], widths=.5, patch_artist=True,
                    medianprops={"color": "black", "lw": 2})
    for patch, color in zip(bp["boxes"], ["#56B4E9", "#E69F00"]):
        patch.set_facecolor(color); patch.set_alpha(.7)
    rng = np.random.default_rng(7)
    for x, values, color in zip([1, 2], vals, ["#0072B2", "#D55E00"]):
        ax.scatter(x + rng.uniform(-.08, .08, len(values)), values, s=18, c=color, alpha=.7)
    ax.set(ylabel="物体异常分数", title="B. 正常物体的分数反而更高")
    ax.text(.04, .95, f"中位数\n正常 {np.median(vals[0]):.3f}\n缺陷 {np.median(vals[1]):.3f}",
            transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "fc": "white", "alpha": .85})
    ax.grid(axis="y", alpha=.2)

    ax = axes[1, 0]
    bins = np.linspace(np.quantile(raw_s, .001), np.quantile(raw_s, .999), 100)
    for lab, color, name in [(0, "#777777", "真实正常点"), (1, "#D55E00", "真实缺陷点")]:
        h, edges = np.histogram(raw_s[raw_y == lab], bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, h, color=color, lw=2, label=name)
        ax.fill_between(centers, h, alpha=.12, color=color)
    ax.set(xlabel="点异常分数（15 视角等权融合）", ylabel="概率密度",
           title=f"C. 点分数高度重叠：完整可见点 AUROC = {roc_auc_score(raw_y, raw_s)*100:.2f}%")
    ax.legend(frameon=False); ax.grid(alpha=.2)

    ax = axes[1, 1]
    names = [r["name"].replace("_bulge", " B").replace("_sink", " S") for r in anomaly]
    x = np.arange(len(anomaly))
    raw_ap = np.array([r["raw_ap"] for r in anomaly]) * 100
    vis_ap = np.array([r["vis_ap"] for r in anomaly]) * 100
    front_ap = np.array([r["front_ap"] for r in anomaly]) * 100
    ax.bar(x-.2, raw_ap, width=.4, color="#009E73", label="完整可见点：15视角等权")
    ax.bar(x+.2, vis_ap, width=.4, color="#CC79A7", label="完整可见点：仅可见视角")
    valid = np.isfinite(front_ap)
    ax.scatter(x[valid], front_ap[valid], c="black", marker="_", s=70, label="旧前表面参考点 AP")
    zero = np.array([r["front_pos"] == 0 for r in anomaly])
    ax.scatter(x[zero], np.zeros(zero.sum()), c="#D55E00", marker="x", s=35, label="旧点域无缺陷点")
    ax.set(xticks=x, xticklabels=names, ylabel="单物体 Point AP (%)",
           title="D. 25 个缺陷飞机逐个定位；红叉表示旧评估根本没选到缺陷点")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.2)

    fig.suptitle("Real3D-AD 飞机类别：Dinomaly-MV baseline 打分审计", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "airplane_category_scores.png", dpi=220, facecolor="white")
    plt.close(fig)

    return {
        "objects": len(rows), "normal_objects": len(normal), "anomaly_objects": len(anomaly),
        "object_auroc": float(roc_auc_score(y_obj, s_obj)),
        "normal_object_score_median": float(np.median(vals[0])),
        "anomaly_object_score_median": float(np.median(vals[1])),
        "front_points": int(len(front_y)), "front_positive_points": int(front_y.sum()),
        "front_point_auroc": float(roc_auc_score(front_y, front_s)),
        "front_point_ap": float(average_precision_score(front_y, front_s)),
        "zero_positive_front_anomaly_objects": int(sum(r["front_pos"] == 0 for r in anomaly)),
        "visible_raw_points": int(len(raw_y)), "visible_raw_positive_points": int(raw_y.sum()),
        "visible_raw_point_auroc_uniform": float(roc_auc_score(raw_y, raw_s)),
        "visible_raw_point_ap_uniform": float(average_precision_score(raw_y, raw_s)),
        "visible_raw_point_auroc_visible_only": float(roc_auc_score(raw_y, vis_s)),
        "visible_raw_point_ap_visible_only": float(average_precision_score(raw_y, vis_s)),
    }


def pca_xy(xyz: np.ndarray) -> np.ndarray:
    centered = xyz - np.median(xyz, axis=0)
    _, _, vt = np.linalg.svd(centered[::max(1, len(centered)//20000)], full_matrices=False)
    return centered @ vt[:2].T


def gallery(rows: list[dict]) -> None:
    normal = [r for r in rows if not r["is_anomaly"]]
    anomaly = [r for r in rows if r["is_anomaly"]]
    chosen = [
        max(normal, key=lambda r: r["object_score"]),
        min(anomaly, key=lambda r: r["object_score"]),
        next(r for r in anomaly if r["name"] == "635_sink"),
        max(anomaly, key=lambda r: r["raw_ap"]),
    ]
    pooled = np.concatenate([r["raw_scores"][r["seen"]] for r in chosen])
    common_vmin, common_vmax = np.quantile(pooled, [.02, .98])
    labels = ["最高分正常物体（误报）", "最低分缺陷物体（漏报）", "典型 sink 漏定位", "完整点域 AP 相对最好"]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 15), constrained_layout=True)
    rng = np.random.default_rng(11)
    with ZipFile(RAW_ZIP) as archive:
        for row_i, (r, case_label) in enumerate(zip(chosen, labels)):
            member = f"Real3D-AD-PCD/airplane/test/{r['name']}.pcd"
            xyz = read_xyz(archive.read(member))
            xy = pca_xy(xyz)
            y = r["raw_labels"]
            seen = r["seen"]
            pos = (y == 1) & seen
            normal_ids = np.flatnonzero((y == 0) & seen)
            if len(normal_ids) > 30000:
                normal_ids = rng.choice(normal_ids, 30000, replace=False)
            ids = np.concatenate([normal_ids, np.flatnonzero(pos)])
            ids = ids[rng.permutation(len(ids))]

            ax = axes[row_i, 0]
            ax.scatter(xy[normal_ids, 0], xy[normal_ids, 1], s=.25, c="#BDBDBD", rasterized=True)
            if pos.any():
                ax.scatter(xy[pos, 0], xy[pos, 1], s=2.5, c="#D55E00", label=f"缺陷点 {pos.sum():,}", rasterized=True)
                ax.legend(frameon=False, loc="upper right", fontsize=8)
            ax.set_title(f"{case_label}｜{r['name']}\n真实标签（灰=正常，红=缺陷）")
            ax.set_aspect("equal"); ax.axis("off")

            ax = axes[row_i, 1]
            sc = ax.scatter(xy[ids, 0], xy[ids, 1], s=.35, c=r["raw_scores"][ids], cmap="viridis",
                            vmin=common_vmin, vmax=common_vmax,
                            rasterized=True)
            if pos.any():
                # A thin outline makes the actual defect location visible without changing score colors.
                ax.scatter(xy[pos, 0], xy[pos, 1], s=3.0, facecolors="none", edgecolors="#FF2D2D", linewidths=.25, rasterized=True)
            fig.colorbar(sc, ax=ax, fraction=.035, pad=.01, label="异常分数")
            ap_text = f"{r['raw_ap']*100:.2f}%" if np.isfinite(r["raw_ap"]) else "不适用"
            ax.set_title(f"模型打分（红圈=真实缺陷位置；四行共用色标范围）\n物体分数 {r['object_score']:.3f}｜完整点 AP {ap_text}｜旧参考点缺陷数 {r['front_pos']}")
            ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle("飞机实例：真实缺陷位置与模型热图对照", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "airplane_score_gallery.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    set_style()
    rows = load_rows()
    summary = overview(rows)
    gallery(rows)
    (OUT / "airplane_score_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
