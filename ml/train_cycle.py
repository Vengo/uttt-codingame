"""
GPU-Accelerated AlphaZero Training Cycle for Ultimate Tic-Tac-Toe
Features:
- Batched parallel self-play on NVIDIA GeForce RTX 5060 Ti GPU
- Tactical 1-ply winning move enforcement during self-play
- Full D4 dihedral group data augmentation (8x sample multiplier)
- Dual-loss training (Policy Cross-Entropy + Value MSE)
- Automatic INT8 quantization and binary synchronization to Rust engine
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.model import AlphaZeroResNet
from ml.dataset import UTTTEnv, UTTTDataset, augment_d4, load_replays_from_files
from ml.export_int8 import export_model_int8
from ml.batched_mcts import run_batched_mcts_self_play
from ml.replay_buffer import StratifiedReplayBuffer


def run_batched_self_play(
    model: AlphaZeroResNet,
    num_games: int = 32,
    device: str = "cuda",
    temperature: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Run `num_games` simultaneously on the GPU using batched tensor evaluation.
    Returns augmented training samples (tensor, legal_mask, policy, value).
    """
    model.eval()
    envs = [UTTTEnv() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]
    active_indices = list(range(num_games))

    while active_indices:
        batch_tensors = []
        batch_legal_masks = []
        batch_legal_indices = []

        for idx in active_indices:
            env = envs[idx]
            t, lm = env.get_observation()
            batch_tensors.append(t)
            batch_legal_masks.append(lm)
            batch_legal_indices.append([i for i in range(81) if lm[i] > 0])

        x = torch.from_numpy(np.array(batch_tensors, dtype=np.float32)).to(device)
        masks = torch.from_numpy(np.array(batch_legal_masks, dtype=np.float32)).to(device)

        with torch.no_grad():
            logits, values = model(x, masks)
            probs = torch.softmax(logits / max(temperature, 0.1), dim=-1).cpu().numpy()

        finished_this_step = []

        for b_i, g_idx in enumerate(active_indices):
            env = envs[g_idx]
            legal = batch_legal_indices[b_i]
            if not legal:
                finished_this_step.append(g_idx)
                continue

            # Tactical check: does any legal move immediately win the macro game?
            immediate_win_move = None
            for m in legal:
                # Test move in cloned environment state
                sb, sq = m // 9, m % 9
                test_x_supers = env.x_supers | (1 << sb) if env.x_to_move else env.x_supers
                test_o_supers = env.o_supers | (1 << sb) if not env.x_to_move else env.o_supers
                from ml.dataset import check_subboard_win
                if env.x_to_move and check_subboard_win(env.xbb[sb] | (1 << sq)):
                    if check_subboard_win(test_x_supers):
                        immediate_win_move = m
                        break
                elif not env.x_to_move and check_subboard_win(env.obb[sb] | (1 << sq)):
                    if check_subboard_win(test_o_supers):
                        immediate_win_move = m
                        break

            t = batch_tensors[b_i]
            lm = batch_legal_masks[b_i]

            if immediate_win_move is not None:
                policy = np.zeros(81, dtype=np.float32)
                policy[immediate_win_move] = 1.0
                chosen = immediate_win_move
            else:
                p_dist = np.zeros(81, dtype=np.float32)
                my_supers = env.x_supers if env.x_to_move else env.o_supers
                opp_supers = env.o_supers if env.x_to_move else env.x_supers
                complete_bbs = my_supers | opp_supers
                for sb in range(9):
                    if (complete_bbs & (1 << sb)) == 0 and ((env.xbb[sb] | env.obb[sb]) == 0x1FF):
                        complete_bbs |= (1 << sb)

                for m in legal:
                    sb, sq = m // 9, m % 9
                    w = probs[b_i, m]
                    # Check subboard win
                    bb = (env.xbb[sb] if env.x_to_move else env.obb[sb]) | (1 << sq)
                    if check_subboard_win(bb):
                        w *= 3.0
                    # Check sending opponent to completed board (free choice)
                    if (complete_bbs & (1 << sq)) != 0:
                        w *= 0.35
                    p_dist[m] = w

                s = p_dist.sum()
                if s > 0:
                    p_dist /= s
                else:
                    p_dist[legal] = 1.0 / len(legal)
                policy = p_dist
                chosen = np.random.choice(81, p=policy)

            histories[g_idx].append((t, lm, policy, env.x_to_move))
            game_over, _ = env.step(chosen)
            if game_over:
                finished_this_step.append(g_idx)

        for f_idx in finished_this_step:
            active_indices.remove(f_idx)

    # Collect and augment training samples
    all_samples = []
    for g_idx in range(num_games):
        env = envs[g_idx]
        winner = env.winner  # 1 for X, -1 for O, 0 for tie
        for t, lm, policy, x_to_move in histories[g_idx]:
            val = float(winner) if x_to_move else float(-winner)
            all_samples.extend(augment_d4(t, lm, policy, val))

    return all_samples


def train_epoch(
    model: AlphaZeroResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float, float]:
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

        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -torch.sum(target_policies * log_probs, dim=-1).mean()
        value_loss = F.mse_loss(values, target_values)

        loss = policy_loss + value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_sz = tensors.size(0)
        total_loss += loss.item() * batch_sz
        total_policy_loss += policy_loss.item() * batch_sz
        total_value_loss += value_loss.item() * batch_sz
        count += batch_sz

    return total_loss / count, total_policy_loss / count, total_value_loss / count


def run_cycle(
    iterations: int = 3,
    games_per_iter: int = 32,
    epochs_per_iter: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_mcts: bool = False,
    mcts_sims: int = 32,
    buffer_capacity: int = 500_000,
    buffer_checkpoint: str = "ml/replay_buffer_checkpoint.npz",
    checkpoint_path: str = "ml/model_checkpoint.pt",
    output_bin: str = "ml/model_int8.bin",
    sync_to_rust: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f"  AlphaZero Continual Improvement Loop on {device.upper()}")
    print(f"============================================================")
    if device == "cuda":
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  Batch Games per Step: {games_per_iter}")
        print(f"  MCTS Mode: {'Batched GPU MCTS (' + str(mcts_sims) + ' sims)' if use_mcts else '1-Ply Fast Self-Play'}")
        print(f"  Buffer Capacity: {buffer_capacity:,} states")

    model = AlphaZeroResNet().to(device)
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded existing model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"  Starting fresh model (load failed: {e})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Initialize Stratified Replay Buffer
    replay_buffer = StratifiedReplayBuffer(capacity=buffer_capacity)
    loaded_states = 0
    if os.path.exists(buffer_checkpoint):
        loaded_states = replay_buffer.load(buffer_checkpoint)

    if loaded_states == 0:
        # Seed buffer with historical and newly generated expert replays
        replay_files = [
            "ml_replays_ab_mcts.json",
            "ml_replays_ab_rb.json",
            "ml_replays_v2_ab.json",
            "replay_ab_vs_hmcts.json",
            "replay_alphabeta_vs_rule_based.json",
            "replay_mcts_vs_original_umcts.json",
            "replay_v2_vs_mcts.json",
            "replay_az_rb.json",
        ]
        raw_samples = load_replays_from_files(replay_files)
        for obs, mask, pol, val in raw_samples:
            replay_buffer.add_sample(obs, mask, pol, val, game_length=35, tier="grandmaster")
        print(f"  Seeded stratified buffer with {len(replay_buffer):,} expert samples.")

    for it in range(1, iterations + 1):
        t0 = time.time()
        print(f"\n--- [Cycle {it}/{iterations}] Generating GPU Self-Play Games ---")
        temp = 1.0 if it == 1 else 0.7

        if use_mcts:
            mcts_samples, mcts_stats = run_batched_mcts_self_play(
                model=model,
                num_games=games_per_iter,
                num_simulations=mcts_sims,
                device=device,
                temperature=temp,
            )
            for obs, mask, pol, val in mcts_samples:
                replay_buffer.add_sample(
                    obs, mask, pol, val,
                    game_length=int(mcts_stats.get("avg_game_length", 30)),
                    tier="deep_mcts" if mcts_sims >= 32 else "self_play",
                )
            print(f"  Batched GPU MCTS: {len(mcts_samples)} samples (X:{mcts_stats['x_wins']}, O:{mcts_stats['o_wins']}, Ties:{mcts_stats['draws']})")
        else:
            new_samples = run_batched_self_play(
                model,
                num_games=games_per_iter,
                device=device,
                temperature=temp,
            )
            for obs, mask, pol, val in new_samples:
                replay_buffer.add_sample(obs, mask, pol, val, game_length=30, tier="self_play")
            print(f"  Fast Self-Play: {len(new_samples)} samples.")

        print(f"  Replay buffer total size: {len(replay_buffer):,} states.")

        steps_per_epoch = max(20, min(150, len(replay_buffer) // batch_size))
        print(f"--- [Cycle {it}/{iterations}] Training Dual-Headed ResNet ({epochs_per_iter} epochs, {steps_per_epoch} steps/epoch) ---")

        model.train()
        for ep in range(1, epochs_per_iter + 1):
            epoch_loss = 0.0
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0

            for _ in range(steps_per_epoch):
                x_b, m_b, p_b, v_b = replay_buffer.sample_batch(batch_size=batch_size, augment_d4_prob=1.0)
                x_b, m_b, p_b, v_b = x_b.to(device), m_b.to(device), p_b.to(device), v_b.to(device)

                optimizer.zero_grad()
                logits, value = model(x_b, m_b)

                # Policy loss (cross-entropy with target visit distribution)
                log_probs = F.log_softmax(logits, dim=-1)
                p_loss = -torch.sum(p_b * log_probs) / x_b.size(0)

                # Value loss (mean squared error)
                v_loss = F.mse_loss(value, v_b)

                loss = p_loss + v_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_p_loss += p_loss.item()
                epoch_v_loss += v_loss.item()

            print(f"  Epoch {ep:2d}/{epochs_per_iter} | Loss: {epoch_loss/steps_per_epoch:.4f} (Policy: {epoch_p_loss/steps_per_epoch:.4f}, Value: {epoch_v_loss/steps_per_epoch:.4f})")

        print(f"  Cycle {it} elapsed: {time.time() - t0:.2f}s")

    # Save replay buffer checkpoint
    replay_buffer.save(buffer_checkpoint)

    # Save PyTorch checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": 64,
            "res_blocks": 4,
        },
        checkpoint_path,
    )
    print(f"\nSaved updated checkpoint to {checkpoint_path}")

    # Export INT8 model
    print(f"Quantizing and exporting INT8 model to {output_bin}...")
    calib_x, calib_m, calib_p, calib_v = replay_buffer.sample_batch(batch_size=100)
    calib_samples = [
        (calib_x[i].numpy(), calib_m[i].numpy(), calib_p[i].numpy(), calib_v[i].item())
        for i in range(100)
    ]
    export_model_int8(model, output_bin_path=output_bin, calibration_samples=calib_samples)

    if sync_to_rust:
        rust_dest = "uttt/src/nn/model_int8.bin"
        shutil.copyfile(output_bin, rust_dest)
        print(f"Synchronized updated weights to {rust_dest} ({os.path.getsize(rust_dest)} bytes)")

    print(f"============================================================")
    print(f"  AlphaZero Training Cycle Complete!")
    print(f"============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-mcts", action="store_true", help="Enable batched GPU MCTS self-play")
    parser.add_argument("--mcts-sims", type=int, default=32, help="Number of MCTS simulations per move")
    parser.add_argument("--buffer-capacity", type=int, default=500_000, help="Replay buffer capacity")
    args = parser.parse_args()

    run_cycle(
        iterations=args.iterations,
        games_per_iter=args.games,
        epochs_per_iter=args.epochs,
        lr=args.lr,
        use_mcts=args.use_mcts,
        mcts_sims=args.mcts_sims,
        buffer_capacity=args.buffer_capacity,
    )
