"""Large before/after view comparison for airplane/635_sink."""

from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repro.pose_registration import DATA, DEFAULT_OUTPUT


def main() -> None:
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in fonts else "SimHei"]
    template = DATA / "airplane/train/good/xyz/128_template/view_000.png"
    original = DATA / "airplane/test/sink/xyz/635_sink/view_000.png"
    aligned = ROOT / "outputs/pose_render/airplane/test/635_sink/view_000.png"
    pose = json.loads((DEFAULT_OUTPUT / "airplane/test/635_sink/pose.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), constrained_layout=True)
    for ax, path, title, subtitle in zip(axes, (template, original, aligned),
                                        ("正常模板 128_template", "缺陷飞机 635_sink：原视角", "635_sink：配准后视角"),
                                        ("固定0号相机；参考点3,040", "同一相机；参考点1,211，缺陷点2",
                                         "同一相机；参考点3,117，缺陷点14")):
        ax.imshow(Image.open(path))
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"相机没有移动：刚体旋转物体 {pose['rotation_deg']:.1f}° 后，主视角外观接近正常模板",
                 fontsize=16, fontweight="bold")
    output = ROOT / "outputs/pose_registration/airplane_635_view_before_after.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
