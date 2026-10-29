"""Shared B/C training and evaluation methods.

Algorithm extracted unchanged from the verified B/C runners.
External Dinomaly runtime is still required; see docs/methods.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("airplane", "candybar", "car", "chicken", "diamond", "duck",
              "fish", "gemstone", "seahorse", "shell", "starfish", "toffees")
STEPS = 5000
BATCH = 16
SCHEMES = {
    "B": ("bilateral26_v1", "bilateral26_cache_v1", "bilateral 26 training views v1"),
    "C": ("global26_v2", "global26_cache_v2", "global 26 training views v2"),
}
r = None


def configure(scheme: str) -> None:
    """Select one experiment per process without changing its model or scoring."""
    global r
    import os
    import importlib

    dataset, cache, label = SCHEMES[scheme]
    source = Path(os.environ.get("REAL3D_RUNTIME_ROOT", str(ROOT.parent / "real3d-dinomaly-20261026")))
    sys.path.insert(0, str(source / "experiments/dinomaly_mv"))
    r = importlib.import_module("runtime")
    r.ROOT = ROOT
    r.DATA = ROOT / "data" / dataset
    r.CACHE = ROOT / "data" / cache
    r.RESULT = ROOT / "results" / dataset
    r.WEIGHTS = ROOT / "checkpoints/dinov2_vitb14_reg4_pretrain.pth"
    r.CONFIG.update(name="Dinomaly-MV / " + label,
                    point_domain="aligned_front_112_raw_representatives",
                    model_selection="final step 5000; no test-based selection",
                    train_views=list(range(26)))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def records(category: str) -> list[Path]:
    paths = sorted((r.DATA / category / "train/good/xyz").glob("*/sample.json"))
    if len(paths) != 4:
        raise ValueError(f"Need four aligned normal training objects: {category}")
    for path in paths:
        meta = json.loads(path.read_text())
        if meta["split"] != "train" or meta["defect"] != "good":
            raise ValueError(f"Training split violation: {path}")
        if len(meta["views"]) != 26:
            raise ValueError(f"Need 26 train views: {path}")
    return paths


def cache_category(category: str, model) -> tuple[torch.Tensor, torch.Tensor, str]:
    paths = records(category)
    image_paths = [path.parent / f"view_{view:03d}.png" for path in paths for view in range(26)]
    fingerprint = hashlib.sha256()
    for path in paths + image_paths:
        fingerprint.update(str(path.relative_to(ROOT)).encode())
        fingerprint.update(digest(path).encode())
    key = fingerprint.hexdigest()
    folder = r.CACHE / category
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "teacher.pt"
    manifest = folder / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["training_inputs_sha256"] != key or info["cache_sha256"] != digest(cached):
            raise RuntimeError(f"Changed or corrupt frozen teacher cache: {category}")
        tensors = torch.load(cached, map_location="cpu", weights_only=True)
        return tensors["z"].cuda(), tensors["targets"].cuda(), key
    start = time.perf_counter()
    z_parts, target_parts = [], []
    with torch.no_grad():
        for index in range(0, len(image_paths), 2):
            batch = torch.stack([r.image_tensor(path) for path in image_paths[index:index + 2]]).cuda()
            z, targets = r.encode(model, batch)
            z_parts.append(z.cpu())
            target_parts.append(torch.stack(targets, dim=1).cpu())
    payload = {"z": torch.cat(z_parts), "targets": torch.cat(target_parts)}
    if payload["z"].shape != (104, 789, 768) or payload["targets"].shape != (104, 2, 768, 28, 28):
        raise RuntimeError("Unexpected DINO teacher feature shape")
    temp = cached.with_suffix(".pt.partial")
    torch.save(payload, temp)
    temp.replace(cached)
    atomic_json(manifest, dict(category=category, normal_only=True,
                               training_inputs_sha256=key, cache_sha256=digest(cached),
                               seconds=time.perf_counter() - start))
    return payload["z"].cuda(), payload["targets"].cuda(), key


class Sampler:
    def __init__(self, n: int, seed: int):
        self.n = n
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = 0

    def next(self):
        if self.cursor + BATCH > len(self.order):
            self.order = torch.randperm(self.n, generator=self.generator)
            self.cursor = 0
        ids = self.order[self.cursor:self.cursor + BATCH]
        self.cursor += BATCH
        return ids

    def state_dict(self):
        return dict(order=self.order, cursor=self.cursor, generator=self.generator.get_state())

    def load_state_dict(self, value):
        self.order = value["order"]
        self.cursor = value["cursor"]
        self.generator.set_state(value["generator"])


def save_checkpoint(path: Path, model, optimizer, scheduler, sampler, step, key, elapsed):
    state = dict(category=path.parent.name, step=step, training_inputs_sha256=key,
                 source_sha256=digest(Path(__file__)), elapsed_seconds=elapsed,
                 bottleneck=model.bottleneck.state_dict(), decoder=model.decoder.state_dict(),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 sampler=sampler.state_dict(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate())
    temp = path.with_suffix(".pt.partial")
    torch.save(state, temp)
    temp.replace(path)


def train(category: str) -> None:
    folder = r.RESULT / "train" / category
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "training_report.json"
    if done.exists():
        print(f"{category}: completed already", flush=True)
        return
    r.seed(1)
    model = r.build_model()
    encoder_hash = r.module_hash(model.encoder)
    z, targets, key = cache_category(category, model)
    model.encoder.cpu()
    torch.cuda.empty_cache()
    optimizer, scheduler = r.optimizer_schedule(model)
    sampler = Sampler(len(z), 1)
    resume = folder / "resume.pt"
    start_step = 0
    elapsed_before = 0.
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["training_inputs_sha256"] != key or state["source_sha256"] != digest(Path(__file__)):
            raise RuntimeError("Resume provenance mismatch")
        model.bottleneck.load_state_dict(state["bottleneck"])
        model.decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        start_step, elapsed_before = state["step"], state["elapsed_seconds"]
        del state
    model.bottleneck.train()
    model.decoder.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with (folder / "updates.jsonl").open("a", buffering=1) as log:
        for step in range(start_step, STEPS):
            ids = sampler.next().to(z.device)
            loss, norm = r.update(model, optimizer, scheduler, z[ids],
                                  [targets[ids, 0], targets[ids, 1]], step)
            log.write(json.dumps(dict(step=step + 1, loss=loss, grad_norm=norm)) + "\n")
            if (step + 1) % 1000 == 0:
                torch.cuda.synchronize()
                elapsed = elapsed_before + time.perf_counter() - start
                save_checkpoint(resume, model, optimizer, scheduler, sampler, step + 1, key, elapsed)
                print(f"{category}: {step + 1}/{STEPS} updates; {elapsed / 60:.1f} min", flush=True)
    torch.cuda.synchronize()
    elapsed = elapsed_before + time.perf_counter() - start
    if r.module_hash(model.encoder) != encoder_hash:
        raise RuntimeError("Frozen DINO encoder changed")
    final = folder / "final.pt"
    # Keep final prediction checkpoint small while retaining resumable state.
    torch.save(dict(category=category, step=STEPS, training_inputs_sha256=key,
                    bottleneck=model.bottleneck.state_dict(), decoder=model.decoder.state_dict()),
               final.with_suffix(".pt.partial"))
    final.with_suffix(".pt.partial").replace(final)
    atomic_json(done, dict(category=category, steps=STEPS, seed=1, train_views=26,
                           normal_prototypes=4, training_inputs_sha256=key,
                           checkpoint_sha256=digest(final), seconds=elapsed,
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                           encoder_unchanged=True))
    resume.unlink(missing_ok=True)
    print(f"{category}: training complete in {elapsed / 60:.1f} min", flush=True)


def evaluate(category: str) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    final = r.RESULT / "train" / category / "final.pt"
    if not final.exists():
        raise FileNotFoundError(final)
    output = r.RESULT / "eval" / category
    output.mkdir(parents=True, exist_ok=True)
    r.seed(1)
    model = r.build_model()
    checkpoint = torch.load(final, map_location="cpu", weights_only=True)
    model.bottleneck.load_state_dict(checkpoint["bottleneck"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    paths = sorted((ROOT / "data/exterior_camera_all_v1" / category / "test").glob("*/xyz/*/sample.json"))
    expected = 53 if category == "chicken" else 50
    if len(paths) != expected:
        raise ValueError(f"Incomplete aligned test data: {category}: {len(paths)}")
    object_y, object_s, point_y, point_s = [], [], [], []
    for index, path in enumerate(paths):
        meta = json.loads(path.read_text())
        folder = path.parent
        with np.load(folder / "points.npz", allow_pickle=False) as data:
            labels = data["labels"].astype(np.int8)
            if "labels_valid" in data:
                labels[~data["labels_valid"]] = -1
            raw_ids, xyz = data["raw_point_ids"], data["xyz"]
        views = r.VIEWS
        maps = []
        with torch.inference_mode():
            for first in range(0, len(views), 3):
                images = torch.stack([r.image_tensor(folder / f"view_{view:03d}.png")
                                      for view in views[first:first + 3]]).cuda()
                teacher, prediction = model(images)
                maps.append(r.score_maps(teacher, prediction))
            positions = [np.load(folder / f"view_{view:03d}.npy", allow_pickle=False) for view in views]
            scores = r.point_scores_from_maps(torch.cat(maps), positions).cpu().numpy()
        if len(scores) != len(labels) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid scores: {folder}")
        np.savez_compressed(output / f"sample_{index:04d}.npz", sample_id=f"{category}/test/{meta['defect']}/{folder.name}",
                            scores=scores, labels=labels, raw_point_ids=raw_ids, xyz=xyz)
        object_y.append(int(meta["defect"] != "good"))
        object_s.append(float(scores.max()))
        point_y.append(labels)
        point_s.append(scores)
    point_y = np.concatenate(point_y)
    point_s = np.concatenate(point_s)
    known = point_y >= 0
    report = dict(category=category, objects=len(paths), points=len(point_y),
                  positive_points=int((point_y == 1).sum()), unknown_points=int((~known).sum()),
                  O_AUROC=float(roc_auc_score(object_y, object_s)),
                  P_AUROC=float(roc_auc_score(point_y[known], point_s[known])),
                  Point_AP=float(average_precision_score(point_y[known], point_s[known])),
                  checkpoint_sha256=digest(final), point_domain="exterior_camera_all_v1_front_112")
    atomic_json(output / "metrics.json", report)
    print(f"{category}: {report['O_AUROC']:.4f} / {report['P_AUROC']:.4f} / {report['Point_AP']:.4f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=tuple(SCHEMES), required=True)
    parser.add_argument("stage", choices=("train", "evaluate", "all"))
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    args = parser.parse_args()
    configure(args.scheme)
    if args.stage in ("train", "all"):
        for category in args.categories:
            train(category)
    if args.stage in ("evaluate", "all"):
        if set(args.categories) == set(CATEGORIES) and not all((r.RESULT / "train" / c / "training_report.json").exists() for c in CATEGORIES):
            raise RuntimeError("All 12 categories must finish training before full evaluation")
        reports = [evaluate(category) for category in args.categories]
        if set(args.categories) == set(CATEGORIES):
            atomic_json(r.RESULT / "macro_metrics.json",
                        {key: float(np.mean([report[key] for report in reports]))
                         for key in ("O_AUROC", "P_AUROC", "Point_AP")})


if __name__ == "__main__":
    main()

