"""
GPU Throughput and Batch Scaling Benchmark for AlphaZero UTTT
Hardware: NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)
Benchmarks:
1. Pure Neural Network Forward Pass Throughput across Batch Sizes (32 .. 8192) in FP32 and FP16 (AMP)
2. Batched MCTS Simulation Scaling
3. Double-Buffered Pipelined Evaluation Architecture
"""

import time
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import AlphaZeroResNet


def benchmark_nn_throughput():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    model = AlphaZeroResNet(num_channels=64, num_res_blocks=4).to(device)
    model.eval()

    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    print("\n" + "=" * 80)
    print(f"{'Batch Size':>10} | {'Precision':>9} | {'Latency (ms)':>12} | {'Throughput (evals/s)':>20} | {'VRAM (MB)':>10}")
    print("=" * 80)

    results = []

    for bs in batch_sizes:
        x = torch.randn(bs, 5, 9, 9, device=device)
        mask = torch.ones(bs, 81, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x, mask)
        torch.cuda.synchronize()

        # FP32 Benchmark
        iters = max(10, min(200, 100000 // bs))
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(x, mask)
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_fp32 = (t1 - t0) / iters
        fps_fp32 = bs / elapsed_fp32
        vram_fp32 = torch.cuda.memory_allocated() / (1024 * 1024)

        print(f"{bs:>10} | {'FP32':>9} | {elapsed_fp32 * 1000:>12.3f} | {fps_fp32:>20,.0f} | {vram_fp32:>10.1f}")
        results.append({"batch": bs, "precision": "FP32", "latency_ms": elapsed_fp32 * 1000, "throughput": fps_fp32, "vram_mb": vram_fp32})

        # FP16 (AMP) Benchmark
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for _ in range(iters):
                    _ = model(x, mask)
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_fp16 = (t1 - t0) / iters
        fps_fp16 = bs / elapsed_fp16
        vram_fp16 = torch.cuda.memory_allocated() / (1024 * 1024)

        print(f"{bs:>10} | {'FP16 (AMP)':>9} | {elapsed_fp16 * 1000:>12.3f} | {fps_fp16:>20,.0f} | {vram_fp16:>10.1f}")
        results.append({"batch": bs, "precision": "FP16", "latency_ms": elapsed_fp16 * 1000, "throughput": fps_fp16, "vram_mb": vram_fp16})

    print("=" * 80)
    return results


if __name__ == "__main__":
    benchmark_nn_throughput()
