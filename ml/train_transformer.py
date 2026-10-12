"""
Train UTTTTransformer on 1,000,000 packed states from ml/replay_buffer_checkpoint.npz
and benchmark performance vs AlphaZeroResNet Generation 7 Champion.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import AlphaZeroResNet
from ml.transformer_model import UTTTTransformer


class ReplayDataset(Dataset):
    def __init__(self, obs: np.ndarray, masks: np.ndarray, policies: np.ndarray, values: np.ndarray, is_train: bool = True):
        self.obs = obs
        self.masks = masks
        self.policies = policies
        self.values = values
        self.is_train = is_train

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # Convert uint8 to float32
        x = self.obs[idx].astype(np.float32)
        m = self.masks[idx].astype(np.float32)
        p = self.policies[idx]
        v = self.values[idx]

        # Optional D4 dihedral augmentation during training
        if self.is_train and np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            flip = np.random.rand() > 0.5
            # Rotations on 9x9 grid
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2)).copy()
            if flip:
                x = np.flip(x, axis=2).copy()

        return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(p), torch.from_numpy(v)


def evaluate_model(model: nn.Module, loader: DataLoader, device: str = "cuda"):
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1_correct = 0
    total_samples = 0

    with torch.no_grad():
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
            for x, m, p_target, v_target in loader:
                x = x.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)
                p_target = p_target.to(device, non_blocking=True)
                v_target = v_target.to(device, non_blocking=True)

                logits, value = model(x, m)

                log_probs = F.log_softmax(logits, dim=-1)
                p_loss = -torch.sum(p_target * log_probs, dim=-1).mean()
                v_loss = F.mse_loss(value, v_target)
                loss = p_loss + v_loss

                # Top-1 accuracy
                pred_moves = torch.argmax(logits, dim=-1)
                target_moves = torch.argmax(p_target, dim=-1)
                top1_correct = (pred_moves == target_moves).sum().item()

                bs = x.size(0)
                total_loss += loss.item() * bs
                total_policy_loss += p_loss.item() * bs
                total_value_loss += v_loss.item() * bs
                total_top1_correct += top1_correct
                total_samples += bs

    return (
        total_loss / total_samples,
        total_policy_loss / total_samples,
        total_value_loss / total_samples,
        total_top1_correct / total_samples,
    )


def train_transformer(
    data_path: str = "ml/replay_buffer_checkpoint.npz",
    champion_path: str = "ml/champion.pt",
    output_path: str = "ml/transformer_champion.pt",
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 5e-4,
    device: str = "cuda",
):
    print("=" * 65)
    print("  UTTTTransformer High-Throughput Replay Buffer Training")
    print(f"  Device: {device.upper()} ({torch.cuda.get_device_name(0)})")
    print(f"  Dataset: {data_path}")
    print(f"  Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr}")
    print("=" * 65)

    # 1. Load Replay Buffer
    t0 = time.time()
    print("Loading 1,000,000 states from replay buffer...")
    npz_data = np.load(data_path)
    obs = npz_data["observations"]
    masks = npz_data["legal_masks"]
    policies = npz_data["policies"]
    values = npz_data["values"]
    total_n = len(obs)
    print(f"Loaded {total_n:,} states in {time.time() - t0:.2f}s.")

    # 2. Train / Val Split (90% / 10%)
    np.random.seed(42)
    indices = np.random.permutation(total_n)
    val_size = 100_000
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = ReplayDataset(
        obs[train_indices], masks[train_indices], policies[train_indices], values[train_indices], is_train=True
    )
    val_dataset = ReplayDataset(
        obs[val_indices], masks[val_indices], policies[val_indices], values[val_indices], is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )

    print(f"Train samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}")

    # 3. Evaluate Baseline Champion ResNet (Gen-7) on Validation Set
    if os.path.exists(champion_path):
        print(f"\nEvaluating Baseline Champion ResNet Gen-7 ({champion_path})...")
        resnet = AlphaZeroResNet().to(device)
        ckpt = torch.load(champion_path, map_location=device)
        resnet.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
        res_loss, res_ploss, res_vloss, res_top1 = evaluate_model(resnet, val_loader, device=device)
        print(f"  [ResNet Gen-7 Baseline]")
        print(f"  Val Loss: {res_loss:.4f} (Policy: {res_ploss:.4f}, Value: {res_vloss:.4f}) | Top-1 Accuracy: {res_top1 * 100:.2f}%")
        del resnet
        torch.cuda.empty_cache()

    # 4. Initialize UTTTTransformer
    transformer = UTTTTransformer(
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.0,
    ).to(device)
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"\nInitialized UTTTTransformer with {num_params:,} parameters.")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler("cuda" if device == "cuda" else "cpu")

    best_val_loss = float("inf")

    # 5. Training Loop
    print("\nStarting Training Epochs...")
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        transformer.train()
        train_loss = 0.0
        train_p_loss = 0.0
        train_v_loss = 0.0
        train_top1 = 0
        samples_processed = 0

        for batch_idx, (x, m, p_target, v_target) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            p_target = p_target.to(device, non_blocking=True)
            v_target = v_target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                logits, value = transformer(x, m)
                log_probs = F.log_softmax(logits, dim=-1)
                p_loss = -torch.sum(p_target * log_probs, dim=-1).mean()
                v_loss = F.mse_loss(value, v_target)
                loss = p_loss + v_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_p_loss += p_loss.item() * bs
            train_v_loss += v_loss.item() * bs
            pred_m = torch.argmax(logits, dim=-1)
            tgt_m = torch.argmax(p_target, dim=-1)
            train_top1 += (pred_m == tgt_m).sum().item()
            samples_processed += bs

        epoch_time = time.time() - t_epoch
        throughput = samples_processed / epoch_time

        # Validation
        val_loss, val_p_loss, val_v_loss, val_top1 = evaluate_model(transformer, val_loader, device=device)

        print(
            f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s, {throughput:,.0f} samples/s) | "
            f"Train Loss: {train_loss / samples_processed:.4f} (P: {train_p_loss / samples_processed:.4f}, V: {train_v_loss / samples_processed:.4f}, Top-1: {train_top1 / samples_processed * 100:.2f}%) | "
            f"Val Loss: {val_loss:.4f} (P: {val_p_loss:.4f}, V: {val_v_loss:.4f}, Top-1: {val_top1 * 100:.2f}%)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": transformer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_p_loss": val_p_loss,
                    "val_v_loss": val_v_loss,
                    "val_top1": val_top1,
                    "architecture": "UTTTTransformer",
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 4,
                    "dim_feedforward": 512,
                },
                output_path,
            )
            print(f"  ⭐ New Best Model saved to {output_path} (Val Loss: {val_loss:.4f})")

    print(f"\nTraining completed! Best Validation Loss: {best_val_loss:.4f}")
    return transformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UTTTTransformer on Replay Buffer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--data", type=str, default="ml/replay_buffer_checkpoint.npz", help="Dataset path")
    parser.add_argument("--champion", type=str, default="ml/champion.pt", help="Baseline Champion path")
    parser.add_argument("--output", type=str, default="ml/transformer_champion.pt", help="Output model path")
    args = parser.parse_args()

    train_transformer(
        data_path=args.data,
        champion_path=args.champion,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
