"""Check public Dinomaly decoder forward/backward and laptop GPU memory."""

from __future__ import annotations

import argparse
import time

from .baseline import build_trainable, decode, imports, seed_all, teacher_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch", type=int, default=1)
    args = parser.parse_args()
    lib = imports()
    torch = lib["torch"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is not installed")
    seed_all(1, torch)
    bottleneck, decoder = build_trainable(lib, args.device)
    optimizer = lib["StableAdamW"]([{"params": list(bottleneck.parameters()) +
                                      list(decoder.parameters())}],
                                   lr=2e-3, betas=(0.9, 0.999),
                                   weight_decay=1e-4, amsgrad=True, eps=1e-8)
    torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
    source = torch.randn(args.micro_batch, 789, 768, device=args.device)
    target_a = torch.randn(args.micro_batch, 784, 768, device=args.device)
    target_b = torch.randn(args.micro_batch, 784, 768, device=args.device)
    start = time.perf_counter()
    reconstructed = decode(lib, source, bottleneck, decoder,
                           checkpoint_blocks=True)
    target = teacher_maps(target_a, target_b)
    loss = lib["global_cosine_hm_percent"](target, reconstructed,
                                            p=0.9, factor=0.1)
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - start
    print(f"forward/backward okay, loss={float(loss):.6f}, seconds={elapsed:.2f}")
    if args.device == "cuda":
        print(f"peak allocated GiB={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
