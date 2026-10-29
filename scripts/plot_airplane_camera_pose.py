"""Plot the saved main-view camera pose and image for airplane/635_sink."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.raw_coverage import RAW_ZIP, read_xyz

SAMPLE = ROOT / "data/handoff/data/pcd_self_v1/airplane/test/sink/xyz/635_sink"
OUTPUT = ROOT / "outputs/airplane_analysis/airplane_635_main_camera_pose.png"


def main() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = [next((x for x in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if x in available), "DejaVu Sans")]
    plt.rcParams["axes.unicode_minus"] = False

    meta = json.loads((SAMPLE / "sample.json").read_text(encoding="utf-8"))
    camera = json.loads((SAMPLE / "cameras.json").read_text(encoding="utf-8"))["cameras"][0]
    rotation = np.asarray(camera["rotation"], dtype=float)
    extrinsic = np.asarray(camera["extrinsic"], dtype=float)
    intrinsic = np.asarray(camera["intrinsic_672"], dtype=float)
    world_to_camera = extrinsic[:3, :3] @ rotation
    camera_center = -world_to_camera.T @ extrinsic[:3, 3]
    optical = world_to_camera.T @ np.array([0., 0., 1.])
    image_right = world_to_camera.T @ np.array([1., 0., 0.])
    image_up = world_to_camera.T @ np.array([0., -1., 0.])

    with ZipFile(RAW_ZIP) as archive:
        raw = read_xyz(archive.read(meta["raw_member"]))
    points = (raw.astype(np.float64) @ np.asarray(meta["basis"]) -
              np.asarray(meta["oriented_center"])) / float(meta["normalization_radius"])
    sampled = points[np.random.default_rng(635).choice(len(points), 35000, replace=False)]
    saved = np.load(SAMPLE / "points.npz")
    refs = saved["xyz"]
    positive = saved["labels"] == 1
    projections = np.load(SAMPLE / "view_000.npy").T
    image = Image.open(SAMPLE / "view_000.png")

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.17, 1])

    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], c="#657786", s=.12, alpha=.22,
               rasterized=True)
    ax.scatter(*camera_center, color="#D55E00", s=120, marker="D", depthshade=False,
               label="主视角相机中心 C")
    ax.quiver(*camera_center, *(optical * 1.28), color="#D55E00", linewidth=3.2,
              arrow_length_ratio=.12, label="镜头朝向")
    # A wireframe of the actual camera rays, scaled to a plane through the object centre.
    pixels = [(0, 0), (671, 0), (671, 671), (0, 671)]
    plane_depth = float((world_to_camera @ (np.zeros(3) - camera_center))[2])
    corners = []
    for u, v in pixels:
        ray_camera = np.linalg.solve(intrinsic, np.array([u, v, 1.]))
        target = camera_center + world_to_camera.T @ (ray_camera * plane_depth)
        corners.append(target)
        ax.plot([camera_center[0], target[0]], [camera_center[1], target[1]],
                [camera_center[2], target[2]], color="#D55E00", alpha=.36, lw=1.2)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="#D55E00", alpha=.36, lw=1.2)
    for basis, color, label in [(image_right, "#0072B2", "画面右"),
                                (image_up, "#009E73", "画面上")]:
        start = camera_center + optical * .2
        ax.quiver(*start, *(basis * .4), color=color, linewidth=2, arrow_length_ratio=.18)
        end = start + basis * .45
        ax.text(*end, label, color=color, fontsize=10)
    for axis, color, name in [(np.array([.55, 0, 0]), "#B50020", "x"),
                               (np.array([0, .55, 0]), "#009E73", "y"),
                               (np.array([0, 0, .55]), "#0072B2", "z")]:
        ax.quiver(0, 0, 0, *axis, color=color, linewidth=2, arrow_length_ratio=.16)
        ax.text(*(axis * 1.15), name, color=color, fontsize=11)
    ax.scatter(*np.zeros(3), c="black", s=20)
    ax.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), zlim=(-1.15, 2.8),
           xlabel="归一化 x", ylabel="归一化 y", zlabel="归一化 z",
           title="A. 相机与完整点云在同一个三维坐标系中")
    ax.set_box_aspect((3.1, 3.1, 3.95))
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", frameon=False)
    ax.text2D(.02, .03, "C = (0, 0, 2.5)\n镜头方向 = (0, 0, −1)\n红色四棱锥 = 相机视锥示意",
              transform=ax.transAxes, fontsize=11,
              bbox={"boxstyle": "round", "fc": "white", "ec": "#D0D0D0", "alpha": .95})

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(image)
    ax.set_title("B. 从这台相机拍到的真实主视角图（v00）")
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate("画面右 = +x", xy=(.93, .07), xytext=(.65, .07), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 2}, color="#0072B2")
    ax.annotate("画面上 = +y", xy=(.07, .93), xytext=(.07, .70), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#009E73", "lw": 2}, color="#009E73")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(image)
    ax.scatter(projections[:, 0], projections[:, 1], s=9, facecolors="none",
               edgecolors="#0072B2", linewidths=.5, label="前表面参考点 1,211个")
    ax.scatter(projections[positive, 0], projections[positive, 1], s=100,
               c="#D55E00", marker="x", linewidths=2.5, label="其中真实缺陷点 2个")
    ax.set_title("C. 同一画面上，主视角网格选出的参考点")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.suptitle("飞机 635_sink｜主视角相机的真实位姿与所见画面", fontsize=18, fontweight="bold")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"{OUTPUT}\ncamera_center={camera_center.tolist()} optical={optical.tolist()} refs={len(refs)}")


if __name__ == "__main__":
    main()
