"""Explain the 112x112 front-reference selection on airplane/635_sink."""

from pathlib import Path
from zipfile import ZipFile
import json
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.raw_coverage import RAW_ZIP, read_xyz

SAMPLE = ROOT / "data/handoff/data/pcd_self_v1/airplane/test/sink/xyz/635_sink"
OUTPUT = ROOT / "outputs/airplane_analysis/airplane_reference_selection.png"


def main() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in available else "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    meta = json.loads((SAMPLE / "sample.json").read_text(encoding="utf-8"))
    cameras = json.loads((SAMPLE / "cameras.json").read_text(encoding="utf-8"))["cameras"]
    cam = cameras[0]
    with ZipFile(RAW_ZIP) as archive:
        raw = read_xyz(archive.read(meta["raw_member"]))
    basis = np.asarray(meta["basis"], dtype=float)
    center = np.asarray(meta["oriented_center"], dtype=float)
    points = (raw.astype(float) @ basis - center) / float(meta["normalization_radius"])
    rot = np.asarray(cam["rotation"], dtype=float)
    ext = np.asarray(cam["extrinsic"], dtype=float)
    intr = np.asarray(cam["intrinsic_672"], dtype=float)
    camera_xyz = (points @ rot.T) @ ext[:3, :3].T + ext[:3, 3]
    projected = camera_xyz @ intr.T
    uv672 = projected[:, :2] / projected[:, 2:3]
    depth = camera_xyz[:, 2]
    cell = np.floor((uv672 + .5) * 112 / 672).astype(int)
    valid = (depth > 0) & (cell >= 0).all(1) & (cell < 112).all(1)
    ids = np.flatnonzero(valid)
    flat = cell[ids, 1] * 112 + cell[ids, 0]
    order = np.lexsort((ids, depth[ids], flat))
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    selected = ids[order[first]]

    stored = np.load(SAMPLE / "points.npz")
    ref_ids = stored["raw_point_ids"]
    if set(selected) != set(ref_ids):
        raise RuntimeError("Recomputed reference IDs do not match stored preprocessing")
    labels = stored["labels"]
    ref_uv224 = np.load(SAMPLE / "view_000.npy").T
    uv224 = (uv672 + .5) / 3 - .5

    counts = np.bincount(flat, minlength=112 * 112)
    occupied = counts[counts > 0]

    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(Image.open(SAMPLE / "view_000.png"))
    ax.scatter(ref_uv224[:, 0], ref_uv224[:, 1], s=4, facecolors="none", edgecolors="#0072B2", linewidths=.35,
               label=f"参考点 {len(ref_ids):,}")
    positive = labels == 1
    ax.scatter(ref_uv224[positive, 0], ref_uv224[positive, 1], s=45, c="#D55E00", marker="x",
               linewidths=1.6, label=f"其中缺陷点 {positive.sum()}")
    for pos in np.arange(0, 225, 20):
        ax.axvline(pos, color="black", lw=.25, alpha=.22); ax.axhline(pos, color="black", lw=.25, alpha=.22)
    ax.add_patch(plt.Rectangle((98, 148), 22, 22, fill=False, edgecolor="#D55E00", lw=1.5))
    ax.set(xlim=(55, 165), ylim=(195, 25), title="A. 完整224×224渲染图与已选参考点\n细网格实际每2×2像素一格；图中只画每10格的辅助线")
    ax.legend(frameon=True, fontsize=9); ax.set_aspect("equal")

    ax = fig.add_subplot(gs[0, 1])
    x0, x1, y0, y1 = 98, 120, 148, 170
    in_zoom = valid & (uv224[:, 0] >= x0) & (uv224[:, 0] < x1) & (uv224[:, 1] >= y0) & (uv224[:, 1] < y1)
    zids = np.flatnonzero(in_zoom)
    # Thin raw points only for readability; selections are always shown in full.
    if len(zids) > 20000:
        zids = np.random.default_rng(1).choice(zids, 20000, replace=False)
    sc = ax.scatter(uv224[zids, 0], uv224[zids, 1], c=depth[zids], cmap="cividis_r", s=2, alpha=.35,
                    rasterized=True, label="投影进格子的原始点")
    ref_zoom = (ref_uv224[:, 0] >= x0) & (ref_uv224[:, 0] < x1) & (ref_uv224[:, 1] >= y0) & (ref_uv224[:, 1] < y1)
    ax.scatter(ref_uv224[ref_zoom, 0], ref_uv224[ref_zoom, 1], s=42, facecolors="none", edgecolors="#0072B2",
               linewidths=1.2, label="每格最终保留的最近点")
    pos_zoom = ref_zoom & positive
    ax.scatter(ref_uv224[pos_zoom, 0], ref_uv224[pos_zoom, 1], s=80, c="#D55E00", marker="x", linewidths=2,
               label="保留下来的缺陷点")
    for pos in np.arange(x0, x1 + .1, 2): ax.axvline(pos, color="black", lw=.5, alpha=.45)
    for pos in np.arange(y0, y1 + .1, 2): ax.axhline(pos, color="black", lw=.5, alpha=.45)
    ax.set(xlim=(x0, x1), ylim=(y1, y0), title="B. 红框局部放大：每个小格是224图上的2×2像素\n等价于672渲染画面上的6×6像素")
    ax.set_xlabel("224图像横坐标"); ax.set_ylabel("224图像纵坐标")
    ax.legend(frameon=True, fontsize=8, loc="lower left")
    fig.colorbar(sc, ax=ax, fraction=.045, pad=.02, label="相机深度（越小越靠近相机）")

    ax = fig.add_subplot(gs[1, 0])
    bins = np.logspace(0, np.log10(max(occupied.max(), 2)), 45)
    ax.hist(occupied, bins=bins, color="#56B4E9", edgecolor="white")
    ax.set_xscale("log")
    ax.set(xlabel="一个112×112格子中投影进来的原始点数（对数轴）", ylabel="格子数量",
           title=f"C. 每格代表的原始点数差异巨大\n占用格 {len(occupied):,}；中位数 {np.median(occupied):.0f} 点；最大 {occupied.max():,} 点")
    ax.grid(alpha=.2)

    ax = fig.add_subplot(gs[1, 1], projection="3d")
    rng = np.random.default_rng(2)
    raw_show = rng.choice(len(points), min(25000, len(points)), replace=False)
    ax.scatter(points[raw_show, 0], points[raw_show, 1], points[raw_show, 2], s=.15, c="#BDBDBD", alpha=.15,
               rasterized=True, label="完整原始点云（抽样显示）")
    ref_xyz = stored["xyz"]
    ax.scatter(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], s=3, c="#0072B2", alpha=.7,
               rasterized=True, label="正面参考点")
    ax.scatter(ref_xyz[positive, 0], ref_xyz[positive, 1], ref_xyz[positive, 2], s=40, c="#D55E00", marker="x",
               label="缺陷参考点")
    ax.view_init(elev=18, azim=-55)
    ax.set_title("D. 映回三维后并不均匀：只留下朝向初始相机的可见外壳")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.set_box_aspect((1, 1, .7))

    fig.suptitle("飞机 635_sink：旧前表面参考点究竟怎么选", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
