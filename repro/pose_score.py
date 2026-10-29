"""Diagnostic frozen-Dinomaly scores on newly pose-registered renders."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .bundle_ablate import CATEGORIES, HANDOFF, ROOT, VIEW_IDS, load_runtime


def score_one(category: str, folder: Path, output: Path, batch_size: int = 2) -> dict:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    rt = load_runtime()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = rt.build_model(device=device)
    checkpoint = torch.load(HANDOFF / "results/dinomaly_mv_v3/train" / category / "final.pt",
                            map_location="cpu", weights_only=True)
    model.bottleneck.load_state_dict(checkpoint["bottleneck"], strict=True)
    model.decoder.load_state_dict(checkpoint["decoder"], strict=True)
    model.eval()
    with np.load(folder / "points.npz", allow_pickle=False) as refs:
        ids = refs["raw_point_ids"].copy()
        labels = refs["labels"].astype(np.int8)
    scores = []
    with torch.inference_mode():
        for first in range(0, len(VIEW_IDS), batch_size):
            views = VIEW_IDS[first:first + batch_size]
            images = torch.stack([rt.image_tensor(folder / f"view_{v:03d}.png")
                                  for v in views]).to(device)
            target, prediction = model(images)
            maps = rt.score_maps(target, prediction)
            positions = torch.as_tensor(np.stack([
                np.load(folder / f"view_{v:03d}.npy", allow_pickle=False)
                for v in views]), device=device, dtype=torch.float32)
            pixels = (positions + .5) * 2 - .5 - 28
            if bool(((pixels < 0) | (pixels > 391)).any()):
                raise ValueError("Reference outside model centre crop")
            grid = ((pixels + .5) / 392 * 2 - 1).permute(0, 2, 1)[:, None]
            dense = F.interpolate(maps[:, None], size=(392, 392), mode="bilinear",
                                  align_corners=False)
            sampled = F.grid_sample(dense, grid, mode="bilinear", padding_mode="border",
                                    align_corners=False)[:, 0, 0]
            scores.append(sampled.cpu().numpy())
    per_view = np.concatenate(scores)
    mean = per_view.mean(axis=0)
    valid = labels >= 0
    summary = {"category": category, "sample": folder.name, "points": len(ids),
               "positive_points": int(np.sum(labels == 1)),
               "object_score": float(mean.max()),
               "point_auroc": float(roc_auc_score(labels[valid], mean[valid]))
               if len(np.unique(labels[valid])) == 2 else None,
               "point_ap": float(average_precision_score(labels[valid], mean[valid]))
               if np.sum(labels[valid] == 1) else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, raw_point_ids=ids, labels=labels,
                        view_scores=per_view.astype(np.float32), scores=mean.astype(np.float32))
    print(summary, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--sample", required=True)
    parser.add_argument("--render-root", type=Path, default=ROOT / "outputs/pose_render")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/pose_scores")
    args = parser.parse_args()
    folder = args.render_root / args.category / args.split / args.sample
    score_one(args.category, folder, args.output_root / args.category / f"{args.sample}.npz")


if __name__ == "__main__":
    main()
