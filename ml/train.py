"""
AlphaZero Training Pipeline for Ultimate Tic-Tac-Toe
Uses:
- Dual-headed ResNet (ml/model.py)
- AdamW optimizer
- Cross-entropy loss for policy, MSE loss for value
- Self-play data generation + replay buffer + D4 augmentation
- GPU acceleration on NVIDIA RTX 5060 Ti
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.model import AlphaZeroResNet
from ml.dataset import (
    UTTTDataset,
    load_replays_from_files,
    run_fast_self_play,
)


def train_epoch(
    model: AlphaZeroResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    count = 0

    for tensors, legal_masks, target_policies, target_values in loader:
        tensors = tensors.to(device)
        legal_masks = legal_masks.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)

        optimizer.zero_grad()
        logits, values = model(tensors, legal_masks)

        # Policy cross-entropy loss: - sum(target_p * log_softmax(logits))
        log_probs = F.log_softmax(logits, dim=-1)
        # Avoid NaN on illegal moves (where logits are -1e9, target_p is 0)
        policy_loss = -torch.sum(target_policies * log_probs, dim=-1).mean()

        # Value MSE loss
        value_loss = F.mse_loss(values, target_values)

        loss = policy_loss + value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * tensors.size(0)
        total_policy_loss += policy_loss.item() * tensors.size(0)
        total_value_loss += value_loss.item() * tensors.size(0)
        count += tensors.size(0)

    return total_loss / count, total_policy_loss / count, total_value_loss / count


def run_training(
    num_iterations: int = 5,
    games_per_iteration: int = 10,
    epochs_per_iteration: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    checkpoint_path: str = "ml/model_checkpoint.pt",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting AlphaZero Training Pipeline on {device.upper()} ===")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = AlphaZeroResNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initial replay buffer loaded from existing games
    replay_files = [
        "replay_ab_vs_hmcts.json",
        "replay_alphabeta_vs_rule_based.json",
        "replay_mcts_vs_original_umcts.json",
        "replay_v2_vs_mcts.json",
    ]
    buffer = load_replays_from_files(replay_files)
    print(f"Initialized replay buffer with {len(buffer)} samples from replay files.")

    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration}/{num_iterations} ---")

        # 1. Self-play data collection
        new_samples = []
        for g in range(games_per_iteration):
            temp = 1.0 if g < games_per_iteration // 2 else 0.5
            game_samples = run_fast_self_play(model, device=device, temperature=temp)
            new_samples.extend(game_samples)

        print(f"Collected {len(new_samples)} new augmented samples from {games_per_iteration} self-play games.")
        buffer.extend(new_samples)

        # Cap buffer size to recent 50,000 samples
        if len(buffer) > 50000:
            buffer = buffer[-50000:]

        dataset = UTTTDataset(buffer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # 2. Train network
        for epoch in range(1, epochs_per_iteration + 1):
            loss, p_loss, v_loss = train_epoch(model, loader, optimizer, device)
            print(f"Epoch {epoch:2d}/{epochs_per_iteration} | Loss: {loss:.4f} (Policy: {p_loss:.4f}, Value: {v_loss:.4f})")

        iter_time = time.time() - iter_start
        print(f"Iteration {iteration} finished in {iter_time:.2f}s")

    # Save final checkpoint
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": 64,
            "res_blocks": 4,
        },
        checkpoint_path,
    )
    print(f"\nModel checkpoint successfully saved to {checkpoint_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero UTTT Training")
    parser.add_argument("--iterations", type=int, default=3, help="Training iterations")
    parser.add_argument("--games", type=int, default=10, help="Self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="ml/model_checkpoint.pt", help="Checkpoint output path")
    args = parser.parse_args()

    run_training(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        epochs_per_iteration=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint,
    )
