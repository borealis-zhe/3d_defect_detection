"""Dinomaly-MV baseline using the public CVPR architecture and our own 3D adapter.

Commands, in order, for each category:
  python -m repro.baseline cache   --processed ... --category airplane
  python -m repro.baseline train   --processed ... --category airplane
  python -m repro.baseline predict --processed ... --category airplane

The 6 GiB laptop GPU requires a smaller physical batch and gradient
accumulation. This preserves an effective batch of 16, but it is not a
bit-for-bit reproduction of the companion's RTX 4090 run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from .geometry import inference_view_indices


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "third_party" / "Dinomaly"
sys.path.insert(0, str(OFFICIAL))


def imports():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    from functools import partial
    from models import vit_encoder
    from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
    from dinov1.utils import trunc_normal_
    from optimizers import StableAdamW
    from utils import WarmCosineScheduler, global_cosine_hm_percent
    return locals()


def seed_all(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_transform(lib):
    transforms = lib["transforms"]
    return transforms.Compose([
        transforms.Resize((448, 448)), transforms.ToTensor(),
        transforms.CenterCrop(392),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_encoder(lib, device):
    encoder = lib["vit_encoder"].load("dinov2reg_vit_base_14")
    encoder.eval().requires_grad_(False)
    return encoder.to(device)


def build_trainable(lib, device):
    nn = lib["nn"]
    partial = lib["partial"]
    bottleneck = nn.ModuleList([lib["bMlp"](768, 3072, 768, drop=0.2)])
    decoder = nn.ModuleList([
        lib["VitBlock"](dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                        attn=lib["LinearAttention2"])
        for _ in range(8)
    ])
    trainable = nn.ModuleList([bottleneck, decoder])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            lib["trunc_normal_"](module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    return bottleneck.to(device), decoder.to(device)


def encode_teacher(lib, encoder, image_batch):
    """Cache exactly the decoder input and two grouped teacher targets."""
    torch = lib["torch"]
    with torch.no_grad():
        hidden = encoder.prepare_tokens(image_batch)
        layers = []
        for i, block in enumerate(encoder.blocks):
            if i > 9:
                break
            hidden = block(hidden)
            if 2 <= i <= 9:
                layers.append(hidden)
        source = torch.stack(layers, 0).mean(0)
        offset = 1 + encoder.num_register_tokens
        target_a = torch.stack(layers[:4], 0).mean(0)[:, offset:, :]
        target_b = torch.stack(layers[4:], 0).mean(0)[:, offset:, :]
    return source, target_a, target_b


def decode(lib, source, bottleneck, decoder, checkpoint_blocks=False):
    torch = lib["torch"]
    hidden = source
    for block in bottleneck:
        hidden = block(hidden)
    layers = []
    for block in decoder:
        if checkpoint_blocks and block.training:
            from torch.utils.checkpoint import checkpoint
            hidden = checkpoint(block, hidden, use_reentrant=False)
        else:
            hidden = block(hidden)
        layers.append(hidden)
    layers.reverse()
    groups = [torch.stack(layers[:4], 0).mean(0),
              torch.stack(layers[4:], 0).mean(0)]
    out = []
    for group in groups:
        patches = group[:, 5:, :]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError(f"Non-square patch count: {patches.shape[1]}")
        out.append(patches.transpose(1, 2).reshape(len(source), 768, side, side))
    return out


def teacher_maps(target_a, target_b):
    side = math.isqrt(target_a.shape[1])
    if side * side != target_a.shape[1]:
        raise ValueError("Teacher patch grid is not square")
    return [target.transpose(1, 2).reshape(len(target), 768, side, side)
            for target in (target_a, target_b)]


def cache_category(processed: Path, category: str, device: str) -> None:
    lib = imports()
    torch = lib["torch"]
    encoder = build_encoder(lib, device)
    transform = image_transform(lib)
    files = sorted((processed / category / "train").glob("*/*.png"))
    if not files:
        raise FileNotFoundError("Run repro.preprocess first")
    cache_dir = processed / category / "teacher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for number, file in enumerate(files, 1):
        out = cache_dir / file.parent.name / (file.stem + ".pt")
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        image = transform(lib["Image"].open(file).convert("RGB"))[None].to(device)
        source, target_a, target_b = encode_teacher(lib, encoder, image)
        # The 6 GiB laptop setup also has finite disk space. Keeping the
        # frozen teacher cache in FP16 cuts its footprint in half; restore
        # FP32 before the trainable network and loss below.
        torch.save({"source": source[0].half().cpu(),
                    "target_a": target_a[0].half().cpu(),
                    "target_b": target_b[0].half().cpu()}, out)
        if number % 10 == 0 or number == len(files):
            print(f"{category}: cached {number}/{len(files)} normal views", flush=True)


def train_category(processed: Path, category: str, device: str,
                   steps: int, micro_batch: int, effective_batch: int,
                   checkpoint_blocks: bool = True,
                   seed: int = 1, resume: bool = False) -> None:
    lib = imports()
    torch, nn = lib["torch"], lib["nn"]
    seed_all(seed, torch)
    if effective_batch % micro_batch:
        raise ValueError("effective_batch must be divisible by micro_batch")
    accumulation = effective_batch // micro_batch
    bottleneck, decoder = build_trainable(lib, device)
    trainable = nn.ModuleList([bottleneck, decoder])
    optimizer = lib["StableAdamW"]([{"params": trainable.parameters()}],
                                   lr=2e-3, betas=(0.9, 0.999),
                                   weight_decay=1e-4, amsgrad=True, eps=1e-8)
    scheduler = lib["WarmCosineScheduler"](optimizer, base_value=2e-3,
                                           final_value=2e-4, total_iters=steps,
                                           warmup_iters=min(100, steps))
    cache_files = sorted((processed / category / "teacher_cache").glob("*/*.pt"))
    if not cache_files:
        raise FileNotFoundError("Run the cache command first")
    rng = np.random.default_rng(seed)
    model_dir = processed / category / "baseline_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    resume_file = model_dir / "resume.pt"
    start_step = 0
    if resume:
        saved = torch.load(resume_file, map_location="cpu", weights_only=False)
        if (saved["steps"] != steps or
                saved["effective_batch"] != effective_batch or
                saved["micro_batch"] != micro_batch):
            raise ValueError("Resume run configuration differs from checkpoint")
        bottleneck.load_state_dict(saved["bottleneck"])
        decoder.load_state_dict(saved["decoder"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device=device)
        start_step = saved["step"]
    trainable.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        p = min(0.9 * step / 1000, 0.9)
        for _ in range(accumulation):
            choices = rng.integers(0, len(cache_files), size=micro_batch)
            batch = [torch.load(cache_files[i], map_location="cpu",
                                weights_only=True) for i in choices]
            source = torch.stack([item["source"] for item in batch]).to(device=device, dtype=torch.float32)
            target_a = torch.stack([item["target_a"] for item in batch]).to(device=device, dtype=torch.float32)
            target_b = torch.stack([item["target_b"] for item in batch]).to(device=device, dtype=torch.float32)
            reconstructed = decode(lib, source, bottleneck, decoder,
                                   checkpoint_blocks=checkpoint_blocks)
            target = teacher_maps(target_a, target_b)
            loss = lib["global_cosine_hm_percent"](target, reconstructed,
                                                    p=p, factor=0.1)
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
        nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 100 == 0 or step == 0 or step + 1 == steps:
            print(f"{category} update {step + 1}/{steps}: "
                  f"normal loss={np.mean(losses):.6f}", flush=True)
        if step + 1 == steps:
            torch.save({"bottleneck": bottleneck.state_dict(),
                        "decoder": decoder.state_dict(), "step": step + 1,
                        "seed": seed, "micro_batch": micro_batch,
                        "effective_batch": effective_batch},
                       model_dir / "final.pt")
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            torch.save({"bottleneck": bottleneck.state_dict(),
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "numpy_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device=device),
                        "step": step + 1, "steps": steps,
                        "micro_batch": micro_batch,
                        "effective_batch": effective_batch}, resume_file)


def sample_score_map(lib, score_map, original_uv):
    """Match 224-pixel projections to the 448-resize/392-crop score image."""
    torch, F = lib["torch"], lib["F"]
    q = 2 * (original_uv + 0.5) - 0.5 - 28
    grid = torch.as_tensor(q, dtype=torch.float32, device=score_map.device)
    grid = 2 * grid / 391 - 1
    grid = grid.reshape(1, 1, -1, 2)
    return F.grid_sample(score_map, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)[0, 0, 0]


def predict_category(processed: Path, category: str, device: str) -> None:
    lib = imports()
    torch, F = lib["torch"], lib["F"]
    seed_all(1, torch)
    encoder = build_encoder(lib, device)
    bottleneck, decoder = build_trainable(lib, device)
    checkpoint = torch.load(processed / category / "baseline_model" / "final.pt",
                            map_location="cpu", weights_only=True)
    bottleneck.load_state_dict(checkpoint["bottleneck"])
    decoder.load_state_dict(checkpoint["decoder"])
    bottleneck.eval()
    decoder.eval()
    transform = image_transform(lib)
    with torch.no_grad():
        for split, output_name in (("train", "normal_predictions"),
                                   ("test", "predictions")):
            items = sorted((processed / category / split).glob("*/points.npz"))
            output = processed / category / output_name
            output.mkdir(parents=True, exist_ok=True)
            for number, points_file in enumerate(items, 1):
                point_data = np.load(points_file)
                all_ids = point_data["view_ids"]
                selected = [j for j, view_id in enumerate(all_ids)
                            if int(view_id) in inference_view_indices()]
                view_scores = []
                for j in selected:
                    view_id = all_ids[j]
                    image_file = points_file.parent / f"{int(view_id):02d}.png"
                    image = transform(lib["Image"].open(image_file).convert("RGB"))
                    source, target_a, target_b = encode_teacher(lib, encoder,
                                                                 image[None].to(device))
                    reconstructed = decode(lib, source, bottleneck, decoder)
                    targets = teacher_maps(target_a, target_b)
                    residual = sum(1 - F.cosine_similarity(a, b, dim=1)
                                   for a, b in zip(targets, reconstructed)) / 2
                    enlarged = F.interpolate(residual[:, None], size=(392, 392),
                                             mode="bilinear", align_corners=True)
                    values = sample_score_map(lib, enlarged,
                                              point_data["projection"][j])
                    view_scores.append(values.cpu().numpy())
                view_scores = np.stack(view_scores).astype(np.float32)
                np.savez_compressed(output / f"{points_file.parent.name}.npz",
                                    view_scores=view_scores,
                                    baseline_scores=view_scores.mean(axis=0),
                                    view_ids=all_ids[selected],
                                    visible=point_data["visible"][selected],
                                    xyz=point_data["xyz"],
                                    labels=point_data["labels"],
                                    raw_ids=point_data["raw_ids"],
                                    object_label=point_data["object_label"])
                if number % 10 == 0 or number == len(items):
                    print(f"{category} {split}: predicted "
                          f"{number}/{len(items)} objects", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["cache", "train", "predict"])
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=16)
    parser.add_argument("--no-activation-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "cache":
        cache_category(args.processed, args.category, args.device)
    elif args.command == "train":
        train_category(args.processed, args.category, args.device,
                       args.steps, args.micro_batch, args.effective_batch,
                       checkpoint_blocks=not args.no_activation_checkpoint,
                       resume=args.resume)
    else:
        predict_category(args.processed, args.category, args.device)


if __name__ == "__main__":
    main()
