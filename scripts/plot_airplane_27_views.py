"""Montage of the actual 27 airplane training renders."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/handoff/data/pcd_self_v1/airplane/train/good/xyz/128_template"
OUTPUT = ROOT / "outputs/airplane_analysis/airplane_27_training_views.png"
ANGLES = list(product((0, -15, 15), repeat=3))
EVAL_VIEWS = {0, 1, 2, 3, 6, 9, 13, 14, 16, 17, 18, 22, 23, 25, 26}


def main() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in available else "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 9, figsize=(19, 7.4), constrained_layout=True)
    for idx, (ax, angle) in enumerate(zip(axes.flat, ANGLES)):
        ax.imshow(Image.open(SOURCE / f"view_{idx:03d}.png"))
        ax.set_title(f"v{idx:02d}  x/y/z={angle}", fontsize=8,
                     color="#0072B2" if idx in EVAL_VIEWS else "#444444")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5 if idx in EVAL_VIEWS else .6)
            spine.set_edgecolor("#0072B2" if idx in EVAL_VIEWS else "#B0B0B0")
    fig.suptitle("Real3D-AD airplane / 128_template：实际的 27 张正常训练渲染图", fontsize=16, fontweight="bold")
    fig.text(.5, .005, "蓝色边框 = 测试时固定使用的 15 个视角；其余视角只参与 baseline 训练。z 旋转主要是画面内旋转。",
             ha="center", fontsize=11)
    fig.savefig(OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
