"""Show original and registered virtual cameras around airplane/635_sink."""

from pathlib import Path
from zipfile import ZipFile
import json
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.pose_registration import DATA, DEFAULT_OUTPUT, normalized_raw
from repro.raw_coverage import RAW_ZIP


def main() -> None:
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in fonts else "SimHei"]
    meta = json.loads((DATA / "airplane/test/sink/xyz/635_sink/sample.json").read_text())
    pose = json.loads((DEFAULT_OUTPUT / "airplane/test/635_sink/pose.json").read_text())
    Q = np.asarray(pose["rotation_row"])
    shift = np.asarray(pose["translation_row"])
    old_cam = np.array([0., 0., 2.5])
    old_look = np.array([0., 0., -1.])
    new_cam = (old_cam - shift) @ Q.T
    new_look = old_look @ Q.T
    with ZipFile(RAW_ZIP) as archive:
        cloud = normalized_raw(archive, meta)
    show = cloud[np.random.default_rng(12).choice(len(cloud), 25000, replace=False)]
    angle = np.degrees(np.arccos(np.clip(old_look @ new_look, -1, 1)))
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(show[:, 0], show[:, 1], show[:, 2], s=.2, c="#666666", alpha=.27, rasterized=True)
    for center, direction, color, name in ((old_cam, old_look, "#0072B2", "原主视角"),
                                           (new_cam, new_look, "#D55E00", "配准后的等效视角")):
        ax.scatter(*center, s=120, c=color, marker="D", depthshade=False, label=name)
        ax.quiver(*center, *(direction * 1.25), color=color, linewidth=3, arrow_length_ratio=.15)
    ax.scatter(0, 0, 0, s=20, c="black")
    ax.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), zlim=(-1.4, 2.8),
           xlabel="归一化 x", ylabel="归一化 y", zlabel="归一化 z")
    ax.set_box_aspect((5.6, 5.6, 4.2))
    ax.view_init(elev=23, azim=-55)
    ax.legend(loc="upper left")
    ax.set_title(f"飞机 635_sink：两种观察方向相差约 {angle:.1f}°\n"
                 "画面中旋转点云，等效于相机绕原始点云移动", fontsize=15)
    output = ROOT / "outputs/pose_registration/airplane_635_virtual_cameras.png"
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
