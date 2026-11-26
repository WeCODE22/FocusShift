"""Batch-1 latency / throughput benchmark for LSNet variants.

Closes the LSNet paper gap of reporting only batch throughput on RTX 3090.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from timm.models import create_model

import model.build  # noqa: F401


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.inference_mode()
def measure(model, device, batch_size=1, img_size=224, warmup=30, iters=100):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = 1000.0 * sum(times) / len(times)
    ips = batch_size / (avg_ms / 1000.0)
    return avg_ms, ips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lsnet_t", "focusshift_t", "lsnet_micro", "focusshift_micro"],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 128])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", type=str, default="custom_data/latency_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    print(f"Device: {device}")
    print(f"{'model':<14} {'params(M)':>10} {'bs':>4} {'latency_ms':>12} {'img/s':>10}")

    for name in args.models:
        model = create_model(name, num_classes=args.num_classes, distillation=False)
        model = model.to(device)
        params = count_params(model)
        for bs in args.batch_sizes:
            try:
                ms, ips = measure(
                    model, device, batch_size=bs, img_size=args.img_size,
                    warmup=args.warmup, iters=args.iters,
                )
            except RuntimeError as exc:
                print(f"{name:<14} {params:10.2f} {bs:4d}  OOM/ERR: {exc}")
                rows.append({
                    "model": name, "params_m": round(params, 3),
                    "batch_size": bs, "latency_ms": None, "images_per_sec": None,
                    "error": str(exc),
                })
                continue
            print(f"{name:<14} {params:10.2f} {bs:4d} {ms:12.2f} {ips:10.1f}")
            rows.append({
                "model": name,
                "params_m": round(params, 3),
                "batch_size": bs,
                "latency_ms": round(ms, 3),
                "images_per_sec": round(ips, 2),
                "device": str(device),
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
