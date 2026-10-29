"""Measure preloaded-image network latency for different view budgets."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .bundle_ablate import HANDOFF, ROOT, load_runtime, records
from .coverage_ablate import selected, normal_training_selection
from .bundle_ablate import VIEW_IDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="airplane")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "view_latency.csv")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this latency benchmark")
    rt = load_runtime()
    model = rt.build_model(device="cuda:0")
    ckpt = torch.load(HANDOFF / "results" / "dinomaly_mv_v3" / "train" /
                      args.category / "final.pt", map_location="cpu", weights_only=True)
    model.bottleneck.load_state_dict(ckpt["bottleneck"], strict=True)
    model.decoder.load_state_dict(ckpt["decoder"], strict=True)
    model.eval()
    folder = records(args.category)[0].parent
    with np.load(folder / "points.npz", allow_pickle=False) as point:
        visible = point["visibility"]
    order_15 = normal_training_selection(args.category, VIEW_IDS)
    order_27 = normal_training_selection(args.category)
    images = {v: rt.image_tensor(folder / f"view_{v:03d}.png")
              for v in range(27)}
    rows = []
    with torch.inference_mode():
        for budget in (3, 5, 9, 15):
            ids = selected(visible, order_15, order_27,
                           budget)["fixed_spread"]
            batch = torch.stack([images[v] for v in ids])
            times = []
            for iteration in range(args.repeats + 1):
                torch.cuda.synchronize()
                begin = time.perf_counter()
                for first in range(0, len(batch), 2):
                    model(batch[first:first + 2].cuda())
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begin
                if iteration:
                    times.append(elapsed)
            rows.append({"category": args.category, "method": "fixed_spread",
                         "budget": budget, "median_network_seconds":
                         float(np.median(times)),
                         "min_network_seconds": float(np.min(times)),
                         "repeats": args.repeats})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
