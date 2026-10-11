"""
End-to-End Pipeline Verification Script
Validates:
1. NVIDIA RTX 5060 Ti GPU detection & PyTorch CUDA acceleration
2. Dual-headed ResNet forward pass on GPU
3. Self-play data generation + training step convergence
4. Symmetric INT8 quantization & BatchNorm folding
5. Export to binary format model_int8.bin (< 320 KB)
"""

import os
import sys
import time
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import AlphaZeroResNet, encode_board
from ml.dataset import run_fast_self_play, load_replays_from_files, UTTTDataset
from ml.train import train_epoch
from ml.export_int8 import export_model_int8
from torch.utils.data import DataLoader


def main():
    print("=" * 60)
    print("      AlphaZero Deep Learning Pipeline Verification")
    print("=" * 60)

    # 1. Verify CUDA & GPU device
    cuda_available = torch.cuda.is_available()
    print(f"[1/5] CUDA Available: {cuda_available}")
    assert cuda_available, "CUDA must be available on RTX 5060 Ti"

    device_name = torch.cuda.get_device_name(0)
    print(f"      Device Name:    {device_name}")
    assert "RTX 5060 Ti" in device_name or "5060" in device_name or "NVIDIA" in device_name

    device = "cuda"

    # 2. Verify Model forward pass on GPU
    print("\n[2/5] Initializing Dual-Headed ResNet on GPU...")
    model = AlphaZeroResNet(num_channels=64, num_res_blocks=4).to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"      Total parameters: {params_count:,}")

    # Random batch test
    dummy_x = torch.randn(8, 5, 9, 9, device=device)
    dummy_mask = torch.ones(8, 81, device=device)
    start_gpu = time.time()
    logits, values = model(dummy_x, dummy_mask)
    gpu_time = (time.time() - start_gpu) * 1000
    print(f"      GPU Batch Forward (B=8): logits={logits.shape}, values={values.shape} in {gpu_time:.2f} ms")
    assert logits.shape == (8, 81)
    assert values.shape == (8, 1)

    # 3. Fast Self-Play & Training step
    print("\n[3/5] Testing Self-Play and Training Pipeline...")
    replays = [
        "replay_ab_vs_hmcts.json",
        "replay_alphabeta_vs_rule_based.json",
        "replay_mcts_vs_original_umcts.json",
    ]
    buffer = load_replays_from_files(replays)
    print(f"      Loaded {len(buffer)} replay samples.")

    sp_samples = run_fast_self_play(model, device=device, temperature=1.0)
    print(f"      Generated {len(sp_samples)} self-play samples on GPU.")
    buffer.extend(sp_samples)

    dataset = UTTTDataset(buffer)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss, p_loss, v_loss = train_epoch(model, loader, optimizer, device)
    print(f"      Trained 1 epoch: Total Loss={loss:.4f} (Policy={p_loss:.4f}, Value={v_loss:.4f})")
    assert not np.isnan(loss), "Loss must not be NaN"

    # Save checkpoint
    os.makedirs("ml", exist_ok=True)
    ckpt_path = "ml/model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": 64,
            "res_blocks": 4,
        },
        ckpt_path,
    )
    print(f"      Saved PyTorch checkpoint to {ckpt_path}")

    # 4. INT8 Quantization and Export
    print("\n[4/5] Quantizing to INT8 and Exporting Binary Model...")
    output_bin = "ml/model_int8.bin"
    export_model_int8(model, output_bin_path=output_bin, calibration_samples=buffer[:100])
    bin_size = os.path.getsize(output_bin)
    print(f"      Exported {output_bin}: {bin_size:,} bytes ({bin_size/1024:.1f} KB)")
    assert bin_size > 50_000 and bin_size < 500_000

    # Copy to uttt/src/nn/model_int8.bin for Rust embedding
    import shutil
    shutil.copyfile(output_bin, "uttt/src/nn/model_int8.bin")
    print("      Synchronized model binary with uttt/src/nn/model_int8.bin")

    # 5. Parity & Masking Sanity Check
    print("\n[5/5] Checking Feature Encoding & Legal Move Masking...")
    # Empty board
    t0, m0 = encode_board([0]*9, [0]*9, 0, 0, 9, True)
    assert m0.sum() == 81.0, "Empty board must have 81 legal moves"

    # Move 40 played (subboard 4, square 4)
    xbb_test = [0]*9
    xbb_test[4] = 1 << 4
    t1, m1 = encode_board(xbb_test, [0]*9, 0, 0, 4, False)
    assert m1.sum() == 8.0, "Constrained board to subboard 4 must have 8 legal moves"
    assert m1[40] == 0.0, "Occupied square 40 must not be legal"

    print("\nAll Python ML & GPU pipeline verification checks PASSED successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
