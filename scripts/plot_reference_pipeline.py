"""Show the exact front-reference selection stages for airplane/635_sink."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.raw_coverage import RAW_ZIP, read_xyz

SAMPLE = ROOT / "data/handoff/data/pcd_self_v1/airplane/test/sink/xyz/635_sink"
OUT = ROOT / "outputs/airplane_analysis/airplane_reference_pipeline.png"


def equal_axes(ax, xyz):
    center = np.mean([xyz.min(axis=0), xyz.max(axis=0)], axis=0)
    radius = np.ptp(xyz, axis=0).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=26, azim=-75)


def main():
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [next((s for s in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if s in fonts), "DejaVu Sans")]
    plt.rcParams["axes.unicode_minus"] = False
    meta = json.loads((SAMPLE / "sample.json").read_text(encoding="utf-8"))
    cam = json.loads((SAMPLE / "cameras.json").read_text(encoding="utf-8"))["cameras"][0]
    saved = np.load(SAMPLE / "points.npz")
    with ZipFile(RAW_ZIP) as archive:
        raw = read_xyz(archive.read(meta["raw_member"]))
    normalized = (raw.astype(np.float64) @ np.asarray(meta["basis"]) - np.asarray(meta["oriented_center"])) / meta["normalization_radius"]
    rotation = np.asarray(cam["rotation"])
    extrinsic = np.asarray(cam["extrinsic"])
    intrinsic = np.asarray(cam["intrinsic_672"])
    camera = (normalized @ rotation.T) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    depth = camera[:, 2]
    cell = np.floor((uv + .5) / 6).astype(np.int32)
    valid = (depth > 0) & (cell >= 0).all(1) & (cell < 112).all(1)
    valid_ids = np.flatnonzero(valid)
    flat = cell[valid_ids, 1] * 112 + cell[valid_ids, 0]
    order = np.lexsort((valid_ids, depth[valid_ids], flat))
    group = flat[order]
    winners = valid_ids[order[np.r_[True, group[1:] != group[:-1]]]]
    ref_ids = saved["raw_point_ids"]
    if set(winners) != set(ref_ids):
        raise AssertionError("Reference selection did not reproduce stored raw IDs")
    ref_uv = np.load(SAMPLE / "view_000.npy").T
    pos = saved["labels"] == 1
    example_ref = int(ref_ids[np.flatnonzero(pos)[0]])
    example_cell = cell[example_ref]
    same_cell = valid_ids[flat == example_cell[1] * 112 + example_cell[0]]
    raw_show = np.random.default_rng(2026).choice(len(raw), min(25000, len(raw)), replace=False)
    ref_xyz = saved["xyz"]

    fig = plt.figure(figsize=(20, 17), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(grid[0, 0], projection="3d")
    ax.scatter(*raw[raw_show].T, s=.12, c="#505050", alpha=.25, rasterized=True)
    equal_axes(ax, raw[raw_show])
    ax.set(title="① 原始PCD：700,353个真实三维点\n仅抽样显示25,000点；选点仍用全部点", xlabel="x", ylabel="y", zlabel="z")

    ax = fig.add_subplot(grid[0, 1], projection="3d")
    ax.scatter(*normalized[raw_show].T, s=.12, c="#0072B2", alpha=.25, rasterized=True)
    equal_axes(ax, normalized[raw_show])
    ax.set(title="② 统一方向、中心和尺度\n渲染另用体素降采样后的57,202点", xlabel="x", ylabel="y", zlabel="z")

    ax = fig.add_subplot(grid[0, 2])
    image = np.asarray(Image.open(SAMPLE / "view_000.png"))
    ax.imshow(image)
    ax.set(title="③ 主视角 v00：实际保存的224×224图\nOpen3D先以672×672透视渲染，再缩小")
    ax.axis("off")

    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(image, extent=(0, 224, 224, 0))
    for v in np.arange(0, 225, 2):
        ax.axvline(v, color="#777777", alpha=.07, lw=.3)
        ax.axhline(v, color="#777777", alpha=.07, lw=.3)
    for v in np.arange(0, 225, 20):
        ax.axvline(v, color="black", alpha=.45, lw=.6)
        ax.axhline(v, color="black", alpha=.45, lw=.6)
    ax.add_patch(plt.Rectangle((96, 148), 26, 24, fill=False, ec="#D55E00", lw=2))
    ax.set(xlim=(58, 158), ylim=(187, 30), title="④ 把相机画面切成112×112格\n每格=672图的6×6像素=此图的2×2像素")
    ax.set_aspect("equal")

    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(image, extent=(0, 224, 224, 0))
    for v in np.arange(96, 123, 2):
        ax.axvline(v, color="#222222", lw=.65, alpha=.7)
    for v in np.arange(148, 173, 2):
        ax.axhline(v, color="#222222", lw=.65, alpha=.7)
    ax.scatter(ref_uv[:, 0], ref_uv[:, 1], s=25, facecolors="none", edgecolors="#0072B2", lw=.8)
    ax.scatter(ref_uv[pos, 0], ref_uv[pos, 1], s=90, c="#D55E00", marker="x", lw=2)
    ax.add_patch(plt.Rectangle((example_cell[0] * 2, example_cell[1] * 2), 2, 2,
                               fill=False, ec="#D55E00", lw=2.5))
    ax.set(xlim=(96, 122), ylim=(172, 148), title="⑤ 缺陷附近放大：蓝圈为每格留下的点\n橙叉为留下的两个缺陷点")
    ax.set_aspect("equal")

    ax = fig.add_subplot(grid[1, 2])
    near = np.sort(depth[same_cell])
    ax.scatter(np.arange(len(near)), near, s=9, c="#888888", alpha=.6)
    ax.scatter(0, near[0], s=100, c="#D55E00", marker="*", zorder=4, label="保留：深度最小")
    ax.set(xlabel="同一格内按深度从近到远的点", ylabel="相机深度（越小越靠前）",
           title=f"⑥ 一个具体格子：{len(same_cell):,}个原始点竞争\n格坐标({example_cell[0]}, {example_cell[1]})，只留最近1点")
    ax.legend(frameon=False); ax.grid(alpha=.2)

    ax = fig.add_subplot(grid[2, 0])
    ax.imshow(image)
    ax.scatter(ref_uv[:, 0], ref_uv[:, 1], s=9, facecolors="none", edgecolors="#0072B2", lw=.5)
    ax.scatter(ref_uv[pos, 0], ref_uv[pos, 1], s=70, c="#D55E00", marker="x", lw=2)
    ax.set(title=f"⑦ 共1,211个有点的格子 → 1,211个参考点\n每个保存原始点ID；其中缺陷点仅{pos.sum()}个")
    ax.axis("off")

    ax = fig.add_subplot(grid[2, 1], projection="3d")
    ax.scatter(*normalized[raw_show].T, s=.08, c="#BBBBBB", alpha=.08, rasterized=True)
    ax.scatter(*ref_xyz.T, s=4, c="#0072B2", alpha=.85, rasterized=True)
    ax.scatter(*ref_xyz[pos].T, s=45, c="#D55E00", marker="x", lw=2)
    equal_axes(ax, normalized[raw_show])
    ax.set(title="⑧ 回到三维：覆盖的是主视角前壳\n二维格子均匀，三维表面并不均匀", xlabel="x", ylabel="y", zlabel="z")

    ax = fig.add_subplot(grid[2, 2])
    ax.axis("off")
    for view, left in zip((0, 13, 26), (0, .34, .68)):
        sub = ax.inset_axes((left, .26, .31, .49))
        sub.imshow(Image.open(SAMPLE / f"view_{view:03d}.png"))
        coords = np.load(SAMPLE / f"view_{view:03d}.npy").T
        sub.scatter(coords[:, 0], coords[:, 1], s=2, facecolors="none", edgecolors="#0072B2", lw=.25)
        sub.set_title(f"v{view:02d}", fontsize=10)
        sub.axis("off")
    ax.text(.5, .95, "⑨ 后续15视角：同一批参考点", transform=ax.transAxes, ha="center", va="top", fontsize=13)
    ax.text(.5, .12, "每个参考点投到各图取分，最后融合。\n其他视角新看见的点不会自动加入。",
            transform=ax.transAxes, ha="center", va="center", fontsize=10)

    fig.suptitle("真实数据流程｜飞机 635_sink 的前表面参考点如何产生", fontsize=19, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"{OUT}\nraw={len(raw)} render={meta['render_points']} valid={len(valid_ids)} refs={len(ref_ids)} example_cell_candidates={len(same_cell)}")


if __name__ == "__main__":
    main()
