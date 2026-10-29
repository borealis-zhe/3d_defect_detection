"""New cross-view ablations on the supplied, frozen Dinomaly-MV baseline.

The handoff saves the *mean* point score, so per-view scores are recomputed
from its original checkpoint and images. Each recomputed mean must match the
saved baseline before new methods are evaluated.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from .ablate import fused_scores


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data" / "handoff"
VIEW_IDS = (0, 1, 2, 3, 6, 9, 13, 14, 16, 17, 18, 22, 23, 25, 26)
CATEGORIES = ("airplane", "candybar", "car", "chicken", "diamond", "duck",
              "fish", "gemstone", "seahorse", "shell", "starfish", "toffees")


def load_runtime():
    path = HANDOFF / "experiments" / "dinomaly_mv" / "runtime.py"
    spec = importlib.util.spec_from_file_location("handoff_dinomaly_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def records(category: str):
    base = HANDOFF / "data" / "pcd_self_v1" / category / "test"
    return sorted(base.glob("*/xyz/*/sample.json"))


def infer(category: str, output_root: Path, batch_size: int = 2) -> None:
    import torch
    import torch.nn.functional as F

    rt = load_runtime()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = rt.build_model(device=device)
    ckpt = torch.load(HANDOFF / "results" / "dinomaly_mv_v3" / "train" /
                      category / "final.pt", map_location="cpu", weights_only=True)
    if ckpt["step"] != 5000 or ckpt["category"] != category:
        raise ValueError("Unexpected checkpoint identity")
    model.bottleneck.load_state_dict(ckpt["bottleneck"], strict=True)
    model.decoder.load_state_dict(ckpt["decoder"], strict=True)
    model.eval()
    out = output_root / category
    out.mkdir(parents=True, exist_ok=True)
    paths = records(category)
    for index, meta_path in enumerate(paths):
        output = out / f"sample_{index:04d}.npz"
        if output.exists():
            # A previous interrupted run must not leave an unreadable result
            # that silently counts as finished.
            with np.load(output, allow_pickle=False) as existing:
                if existing["view_scores"].shape[0] != len(VIEW_IDS):
                    raise ValueError(f"Incomplete cached prediction: {output}")
            continue
        folder = meta_path.parent
        with np.load(folder / "points.npz", allow_pickle=False) as point:
            labels = point["labels"].astype(np.int8)
            if "labels_valid" in point:
                labels[~point["labels_valid"]] = -1
            visible = point["visibility"][list(VIEW_IDS)]
            raw_ids = point["raw_point_ids"]
        n_points = len(labels)
        arrays = []
        with torch.inference_mode():
            for start in range(0, len(VIEW_IDS), batch_size):
                chunk = VIEW_IDS[start:start + batch_size]
                images = torch.stack([rt.image_tensor(folder / f"view_{v:03d}.png")
                                      for v in chunk]).to(device)
                targets, predictions = model(images)
                maps = rt.score_maps(targets, predictions)
                positions = torch.as_tensor(np.stack([
                    np.load(folder / f"view_{v:03d}.npy", allow_pickle=False)
                    for v in chunk]), device=device, dtype=torch.float32)
                if positions.shape != (len(chunk), 2, n_points):
                    raise ValueError(f"Bad point positions in {folder}")
                pixels = (positions + 0.5) * 2 - 0.5 - 28
                if bool(((pixels < 0) | (pixels > 391)).any()):
                    raise ValueError(f"Point projects outside crop: {folder}")
                grid = ((pixels + 0.5) / 392 * 2 - 1).permute(0, 2, 1)[:, None]
                dense = F.interpolate(maps[:, None], size=(392, 392),
                                      mode="bilinear", align_corners=False)
                point_scores = F.grid_sample(dense, grid, mode="bilinear",
                                             padding_mode="border",
                                             align_corners=False)[:, 0, 0]
                arrays.append(point_scores.cpu().numpy())
        view_scores = np.concatenate(arrays).astype(np.float32)
        saved_path = (HANDOFF / "results" / "dinomaly_mv_v3" / "eval" /
                      category / f"sample_{index:04d}.npz")
        with np.load(saved_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["raw_point_ids"], raw_ids):
                raise ValueError(f"Raw point IDs differ: {folder}")
            if not np.array_equal(saved["labels"], labels):
                raise ValueError(f"Point labels differ: {folder}")
            mean_error = float(np.max(np.abs(view_scores.mean(0) - saved["scores"])))
            if mean_error > 2e-6:
                raise ValueError(f"Recomputed baseline differs by {mean_error}: {folder}")
            sample_id = str(saved["sample_id"])
        temporary = output.with_name(output.stem + ".partial.npz")
        np.savez_compressed(temporary, sample_id=sample_id, view_ids=VIEW_IDS,
                            view_scores=view_scores, visible=visible,
                            labels=labels, raw_point_ids=raw_ids,
                            baseline_max_abs_error=mean_error)
        temporary.replace(output)
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f"{category}: {index+1}/{len(paths)} per-view scores, "
                  f"last baseline error {mean_error:.2g}", flush=True)


def methods(view_scores, visible):
    out = fused_scores(view_scores, visible)
    for count in (3, 5, 9, 15):
        out[f"views_{count}"] = view_scores[:count].mean(0)
    return out


def evaluate(output_root: Path, csv_path: Path) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for category in CATEGORIES:
        files = sorted((output_root / category).glob("sample_*.npz"))
        if len(files) != (53 if category == "chicken" else 50):
            raise ValueError(f"Missing per-view predictions for {category}")
        bucket = {}
        for file in files:
            with np.load(file, allow_pickle=False) as data:
                sid = str(data["sample_id"])
                y = data["labels"]
                known = y >= 0
                oy = int(sid.split("/")[2] != "good")
                for name, scores in methods(data["view_scores"],
                                            data["visible"]).items():
                    record = bucket.setdefault(name, {"oy": [], "os": [],
                                                       "py": [], "ps": []})
                    record["oy"].append(oy)
                    record["os"].append(float(scores.max()))
                    record["py"].append(y[known])
                    record["ps"].append(scores[known])
        for name, record in bucket.items():
            py = np.concatenate(record["py"])
            ps = np.concatenate(record["ps"])
            rows.append({"category": category, "method": name,
                         "object_auroc": 100 * roc_auc_score(record["oy"], record["os"]),
                         "point_auroc": 100 * roc_auc_score(py, ps),
                         "point_ap": 100 * average_precision_score(py, ps),
                         "objects": len(files), "points": len(py),
                         "positive_points": int((py == 1).sum())})
        print(f"evaluated {category}", flush=True)
    for name in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == name]
        rows.append({"category": "macro", "method": name,
                     **{metric: float(np.mean([r[metric] for r in subset]))
                        for metric in ("object_auroc", "point_auroc", "point_ap")},
                     **{metric: int(sum(r[metric] for r in subset))
                        for metric in ("objects", "points", "positive_points")}})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([r for r in rows if r["category"] == "macro"],
                     ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("infer", "evaluate"))
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "bundle_views")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs" / "bundle_ablation.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "infer":
        for category in args.categories:
            infer(category, args.output_root, args.batch_size)
    else:
        evaluate(args.output_root, args.results)


if __name__ == "__main__":
    main()
