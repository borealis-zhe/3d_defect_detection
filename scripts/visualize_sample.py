"""Visualize one verified Dinomaly-MV Real3D-AD sample end to end.

Uses the supplied checkpoint and preprocessed views. The 15-view point scores
are checked against the original saved prediction; no model training occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.gridspec import GridSpec
from PIL import Image

from repro.bundle_ablate import HANDOFF, ROOT, VIEW_IDS, load_runtime, records
from repro.full_surface import normalized_raw
from repro.raw_coverage import RAW_ZIP, read_xyz


CATEGORY = "candybar"
SAMPLE = "347_bulge"
VIEW = 0
OUT = ROOT / "outputs" / "visual_walkthrough"


def load_sample():
    paths = records(CATEGORY)
    meta_path = next(path for path in paths if SAMPLE in str(path))
    index = paths.index(meta_path)
    folder = meta_path.parent
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with ZipFile(RAW_ZIP) as archive:
        raw = read_xyz(archive.read(meta["raw_member"]))
    xyz = normalized_raw(raw, meta)
    with np.load(folder / "points.npz", allow_pickle=False) as point:
        refs = {key: point[key].copy() for key in point.files}
    with np.load(OUT.parent / "full_surface" / CATEGORY /
                 f"sample_{index:04d}.npz", allow_pickle=False) as full:
        full = {key: full[key].copy() for key in full.files}
    with np.load(OUT.parent / "bundle_views" / CATEGORY /
                 f"sample_{index:04d}.npz", allow_pickle=False) as views:
        views = {key: views[key].copy() for key in views.files}
    with np.load(HANDOFF / "results" / "dinomaly_mv_v3" / "eval" /
                 CATEGORY / f"sample_{index:04d}.npz", allow_pickle=False) as saved:
        saved_scores = saved["scores"].copy()
    assert np.array_equal(full["reference_ids"], refs["raw_point_ids"])
    assert np.array_equal(views["raw_point_ids"], refs["raw_point_ids"])
    err = float(np.max(np.abs(views["view_scores"].mean(0) - saved_scores)))
    assert err < 2e-6, err
    return folder, xyz, refs, full, views, meta, err


def selected_view_map(folder):
    import torch
    import torch.nn.functional as F

    rt = load_runtime()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = rt.build_model(device=device)
    ckpt = torch.load(HANDOFF / "results" / "dinomaly_mv_v3" / "train" /
                      CATEGORY / "final.pt", map_location="cpu", weights_only=True)
    assert ckpt["category"] == CATEGORY and ckpt["step"] == 5000
    model.bottleneck.load_state_dict(ckpt["bottleneck"], strict=True)
    model.decoder.load_state_dict(ckpt["decoder"], strict=True)
    model.eval()
    with torch.inference_mode():
        tensor = rt.image_tensor(folder / f"view_{VIEW:03d}.png").unsqueeze(0).to(device)
        targets, reconstructed = model(tensor)
        patch_score = rt.score_maps(targets, reconstructed)[0]
        dense = F.interpolate(patch_score[None, None], size=(392, 392),
                              mode="bilinear", align_corners=False)[0, 0]
    return dense.cpu().numpy(), patch_score.cpu().numpy()


def equal_3d(ax, xyz):
    lo, hi = np.min(xyz, axis=0), np.max(xyz, axis=0)
    center = (lo + hi) / 2
    half = max((hi - lo).max() / 2, 0.1)
    ax.set(xlim=(center[0]-half, center[0]+half),
           ylim=(center[1]-half, center[1]+half),
           zlim=(center[2]-half, center[2]+half))
    ax.set_box_aspect((1, 1, 1), zoom=0.93)
    # View 00 looks approximately along negative Z, so the top view exposes
    # the candybar silhouette while preserving the actual 3D point positions.
    ax.view_init(elev=78, azim=-90)
    ax.set_axis_off()


def cloud_panel(ax, xyz, labels, seen, scores=None, chosen=None):
    rng = np.random.default_rng(20261028)
    ids = np.sort(rng.choice(len(xyz), min(22000, len(xyz)), replace=False))
    if scores is None:
        ax.scatter(*xyz[ids].T, s=0.45, color="#586778", alpha=0.43,
                   rasterized=True)
    else:
        ids = ids[seen[ids]]
        vmax = float(np.quantile(scores[seen], 0.99))
        ax.scatter(*xyz[ids].T, s=0.8, c=scores[ids], cmap="turbo",
                   vmin=0, vmax=vmax, alpha=0.85, rasterized=True)
    if chosen is not None:
        ax.scatter(*xyz[chosen], s=120, color="#f9d547", edgecolors="black",
                   linewidths=1.3, depthshade=False)
    equal_3d(ax, xyz)


def draw_views(folder):
    fig, axes = plt.subplots(3, 5, figsize=(14.5, 9.0), dpi=170)
    fig.patch.set_facecolor("white")
    for ax, view_id in zip(axes.flat, VIEW_IDS):
        image = Image.open(folder / f"view_{view_id:03d}.png").convert("RGB")
        ax.imshow(image)
        ax.set_title(f"View {view_id:02d}" + ("  ← illustrated below" if view_id == VIEW else ""),
                     fontsize=11, color="#b54725" if view_id == VIEW else "#263548")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2 if view_id == VIEW else 0.6)
            spine.set_edgecolor("#d5663e" if view_id == VIEW else "#d5dce4")
    fig.suptitle(f"Real3D-AD {CATEGORY}/{SAMPLE}: the 15 actual model views",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.text(0.5, 0.018,
             "Each image is a preprocessed point-cloud rendering. These are the fixed 15 test views, not 15 different objects.",
             ha="center", fontsize=11, color="#405064")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96), h_pad=1.3, w_pad=1.0)
    path = OUT / "15_views.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def draw_flow(folder, xyz, refs, full, views, dense, err):
    labels = full["labels"]
    scores = full["uniform_scores"]
    seen = full["seen"]
    ref_ids = refs["raw_point_ids"]
    ref_labels = refs["labels"]
    eligible = np.flatnonzero((ref_labels == 1) & refs["visibility"][VIEW])
    chosen_ref = int(eligible[len(eligible)//2])
    chosen_raw = int(ref_ids[chosen_ref])
    position = np.load(folder / f"view_{VIEW:03d}.npy", allow_pickle=False)[:, chosen_ref]
    crop_xy = (position + 0.5) * 2 - 0.5 - 28
    one = float(views["view_scores"][0, chosen_ref])
    mean15 = float(views["view_scores"][:, chosen_ref].mean())
    assert abs(mean15 - float(scores[chosen_raw])) < 2e-6

    fig = plt.figure(figsize=(20, 11.5), dpi=170, facecolor="white")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1],
                  left=0.035, right=0.985, top=0.87, bottom=0.075,
                  wspace=0.18, hspace=0.32)
    ax_cloud = fig.add_subplot(gs[0, 0], projection="3d")
    ax_img = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[0, 2])
    ax_score = fig.add_subplot(gs[0, 3], projection="3d")
    ax_gt = fig.add_subplot(gs[1, 0], projection="3d")
    ax_detail = fig.add_subplot(gs[1, 1:3])
    ax_stats = fig.add_subplot(gs[1, 3])

    cloud_panel(ax_cloud, xyz, labels, seen, chosen=chosen_raw)
    ax_cloud.set_title("1  Original 3D point cloud", fontsize=14, fontweight="bold")

    image = np.asarray(Image.open(folder / f"view_{VIEW:03d}.png").convert("RGB"))
    ax_img.imshow(image)
    ax_img.scatter(position[0], position[1], s=210, facecolors="none",
                   edgecolors="#ffe353", linewidths=2.5)
    ax_img.set_title("2  One of 15 rendered views (view 00)", fontsize=14,
                     fontweight="bold")
    ax_img.set_xticks([]); ax_img.set_yticks([])

    im = ax_map.imshow(dense, cmap="turbo", vmin=0,
                       vmax=float(np.quantile(dense, 0.995)))
    ax_map.scatter(crop_xy[0], crop_xy[1], s=210, facecolors="none",
                   edgecolors="#ffe353", linewidths=2.5)
    ax_map.set_title("3  Feature-reconstruction error map", fontsize=14,
                     fontweight="bold")
    ax_map.set_xticks([]); ax_map.set_yticks([])
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.025, label="cosine error")

    cloud_panel(ax_score, xyz, labels, seen, scores=scores, chosen=chosen_raw)
    ax_score.set_title("4  Mean of 15 scores → raw 3D points", fontsize=14,
                       fontweight="bold")
    score_vmax = float(np.quantile(scores[seen], 0.99))
    score_sm = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=score_vmax),
                                 cmap="turbo")
    fig.colorbar(score_sm, ax=ax_score, fraction=0.046, pad=0.025,
                 label="point score")

    rng = np.random.default_rng(20261028)
    background = np.flatnonzero(labels == 0)
    background = rng.choice(background, min(17000, len(background)), replace=False)
    positive = np.flatnonzero(labels == 1)
    ax_gt.scatter(*xyz[background].T, s=0.45, color="#9ca9b7", alpha=0.33,
                  rasterized=True)
    ax_gt.scatter(*xyz[positive].T, s=2.6, color="#e94747", alpha=0.9,
                  rasterized=True)
    ax_gt.scatter(*xyz[chosen_raw], s=120, color="#f9d547",
                  edgecolors="black", depthshade=False)
    equal_3d(ax_gt, xyz)
    ax_gt.set_title("5  Ground truth (red, evaluation only)", fontsize=14,
                    fontweight="bold")

    ax_detail.set_axis_off()
    ax_detail.text(0.02, 0.94, "How one yellow point gets its score", fontsize=17,
                   fontweight="bold", color="#1c3044", va="top")
    steps = [
        f"Raw point ID {chosen_raw:,}  →  view 00 pixel ({int(position[0])}, {int(position[1])})",
        f"Pixel samples a cosine-error map  →  view 00 score = {one:.4f}",
        f"Repeat for 15 views  →  mean point score = {mean15:.4f}",
        "Repeat for all points  →  3D score cloud; object score = maximum point score",
    ]
    for i, line in enumerate(steps):
        ax_detail.text(0.035, 0.77 - i*0.15, line, transform=ax_detail.transAxes,
                       fontsize=13.2, color="#253c52", va="center",
                       bbox=dict(boxstyle="round,pad=0.37", fc="#eef4f8",
                                 ec="#cfdee7", lw=0.8))
    ax_detail.text(0.02, 0.1,
                   "Yellow = one matched 3D point / image pixel. Ground truth never enters the model.\n"
                   "Only points visible in at least one view are colored in the 3D score panel.",
                   transform=ax_detail.transAxes, fontsize=11.5,
                   color="#526478", va="top")

    ax_stats.set_axis_off()
    ax_stats.text(0.02, 0.94, "This sample", fontsize=17, fontweight="bold",
                  color="#1c3044", va="top")
    items = [
        ("Raw PCD points", f"{len(xyz):,}"),
        ("GT defect points", f"{int((labels==1).sum()):,}"),
        ("Visible raw points", f"{int(seen.sum()):,}"),
        ("Old evaluation references", f"{len(ref_ids):,}"),
        ("Positive references", f"{int((ref_labels==1).sum()):,}"),
        ("Score check max error", f"{err:.1e}"),
    ]
    for i, (key, value) in enumerate(items):
        y = 0.78 - i*0.115
        ax_stats.text(0.02, y, key, transform=ax_stats.transAxes,
                      fontsize=12.2, color="#526478")
        ax_stats.text(0.98, y, value, transform=ax_stats.transAxes,
                      fontsize=12.6, color="#182e42", fontweight="bold", ha="right")
        ax_stats.plot([0.02, 0.98], [y-0.025, y-0.025], transform=ax_stats.transAxes,
                      color="#e2e9ee", lw=0.8)
    fig.suptitle(f"Real3D-AD | {CATEGORY}/{SAMPLE} | frozen Dinomaly-MV baseline",
                 x=0.035, y=0.98, ha="left", fontsize=22,
                 color="#1a2b3d", fontweight="bold")
    fig.text(0.035, 0.925,
             "Actual PCD and 15-view model outputs; one view shown in detail. Yellow circle tracks the same point through projection and back-projection.",
             fontsize=13.2, color="#526478")
    fig.text(0.035, 0.03,
             "Point scores here use equal 15-view averaging, including views where a point is occluded; the scored 3D panel displays only points visible in ≥1 view."
             "  Model score ≠ ground-truth label.",
             fontsize=10.8, color="#526478")
    path = OUT / "end_to_end.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path, chosen_raw, one, mean15


def draw_interactive(xyz, full, chosen_raw):
    import plotly.graph_objects as go

    labels = full["labels"]
    scores = full["uniform_scores"]
    seen = full["seen"]
    rng = np.random.default_rng(20261028)
    ids = np.flatnonzero(seen)
    ids = rng.choice(ids, min(28000, len(ids)), replace=False)
    ids = np.unique(np.r_[ids, chosen_raw])
    show = xyz[ids]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=show[:, 0], y=show[:, 1], z=show[:, 2],
                               mode="markers", name="Model point score",
                               marker=dict(size=2.2, color=scores[ids],
                                           colorscale="Turbo", opacity=0.77,
                                           colorbar=dict(title="Score")),
                               customdata=np.stack([ids, labels[ids]], axis=1),
                               hovertemplate="raw point %{customdata[0]}<br>score %{marker.color:.3f}<br>GT %{customdata[1]}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[xyz[chosen_raw, 0]], y=[xyz[chosen_raw, 1]],
                               z=[xyz[chosen_raw, 2]], mode="markers",
                               name="Tracked yellow point",
                               marker=dict(size=8, color="#ffe353",
                                           line=dict(color="black", width=3))))
    fig.update_layout(title=f"Rotate and zoom: {CATEGORY}/{SAMPLE} (visible raw points)",
                      width=1100, height=760, margin=dict(l=0, r=0, b=0, t=70),
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                                 aspectmode="data"))
    path = OUT / "interactive_point_cloud.html"
    fig.write_html(path, include_plotlyjs=True)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    folder, xyz, refs, full, views, meta, err = load_sample()
    dense, patch = selected_view_map(folder)
    view_path = draw_views(folder)
    flow_path, chosen_raw, one, mean15 = draw_flow(folder, xyz, refs, full,
                                                   views, dense, err)
    interactive_path = draw_interactive(xyz, full, chosen_raw)
    result = dict(sample=f"{CATEGORY}/test/bulge/{SAMPLE}", raw_points=len(xyz),
                  positive_raw_points=int((full["labels"] == 1).sum()),
                  visible_raw_points=int(full["seen"].sum()),
                  reference_points=len(refs["raw_point_ids"]),
                  tracked_raw_point_id=chosen_raw, view0_score=one,
                  mean15_score=mean15, baseline_max_abs_error=err,
                  full_image=str(flow_path), views_image=str(view_path),
                  interactive_html=str(interactive_path))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
