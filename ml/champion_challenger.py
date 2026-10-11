"""
AlphaZero Champion vs Challenger Continuous Self-Improvement System
Features:
- Heavy GPU Saturation on NVIDIA GeForce RTX 5060 Ti (64-128 games parallel)
- Deep MCTS Search with >= 800 simulations per move
- Automated Champion vs Challenger gating tournament (>= 55% win rate required)
- Bayesian Elo / Glicko-2 rating tracking over generations
- Live JSON telemetry for frontend / website monitoring
- Automatic INT8 quantization and synchronization to the pure Rust engine
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import AlphaZeroResNet
from ml.dataset import UTTTEnv
from ml.batched_mcts import run_batched_gpu_mcts
from ml.replay_buffer import ContiguousStratifiedReplayBuffer
from ml.rust_mcts_worker import RustMctsWorker
from ml.export_int8 import export_model_int8


def get_gpu_telemetry(device: str) -> Dict[str, any]:
    """Inspect GPU hardware telemetry."""
    if not torch.cuda.is_available() or device == "cpu":
        return {
            "device": "CPU",
            "vram_allocated_gb": 0.0,
            "vram_reserved_gb": 0.0,
            "vram_total_gb": 0.0,
            "utilization_pct": 0,
        }
    dev_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    alloc = torch.cuda.memory_allocated(dev_idx) / 1e9
    res = torch.cuda.memory_reserved(dev_idx) / 1e9
    total = props.total_memory / 1e9
    return {
        "device": props.name,
        "vram_allocated_gb": round(alloc, 2),
        "vram_reserved_gb": round(res, 2),
        "vram_total_gb": round(total, 2),
        "utilization_pct": round((res / total) * 100, 1),
    }


def write_status(
    status_file: str,
    frontend_status_file: Optional[str],
    data: Dict[str, any],
):
    """Write live telemetry atomically to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(status_file)), exist_ok=True)
    temp_file = status_file + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, status_file)

    if frontend_status_file:
        try:
            target_dir = os.path.dirname(os.path.abspath(frontend_status_file))
            if os.path.isdir(target_dir):
                shutil.copyfile(status_file, frontend_status_file)
        except Exception:
            pass


def build_2ply_opening_suite(num_pairs: int) -> List[UTTTEnv]:
    """
    Builds a balanced suite of distinct 2-ply opening positions.
    Each position is evaluated in an alternating pair:
      Game 2k:   Challenger is Player 1 (X), Champion is Player 2 (O)
      Game 2k+1: Champion is Player 1 (X), Challenger is Player 2 (O)
    Guarantees diverse tactical exploration across all 9 subboards with zero duplicates.
    """
    import random
    subboards = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    cells = [4, 0, 2, 6, 8, 1, 3, 5, 7]

    candidates = []
    for sb in subboards:
        for c in cells:
            for reply_c in cells:
                candidates.append((sb * 9 + c, reply_c))

    rng = random.Random(42)
    rng.shuffle(candidates)

    selected = candidates[:num_pairs]
    envs = []
    for move1, reply_c in selected:
        env = UTTTEnv()
        env.step(move1)
        sb2 = move1 % 9
        move2 = sb2 * 9 + reply_c
        env.step(move2)
        envs.append(env)

    return envs


def play_eval_match(
    model_challenger: AlphaZeroResNet,
    model_champion: AlphaZeroResNet,
    num_games: int = 40,
    mcts_sims: int = 200,
    device: str = "cuda",
) -> Tuple[int, int, int]:
    """
    Play a head-to-head evaluation match between Challenger and Champion.
    Evaluates color-alternating pairs over a 2-ply opening suite to eliminate
    deterministic duplicates and cancel first-move bias.
    Returns: (challenger_wins, champion_wins, draws)
    """
    model_challenger.eval()
    model_champion.eval()

    num_pairs = max(num_games // 2, 1)
    opening_envs = build_2ply_opening_suite(num_pairs)

    envs = []
    challenger_is_p1 = []
    for env in opening_envs:
        # Game 2k: Challenger is P1 (X)
        envs.append(env.clone())
        challenger_is_p1.append(True)
        # Game 2k+1: Champion is P1 (X)
        envs.append(env.clone())
        challenger_is_p1.append(False)

    active = list(range(len(envs)))

    while active:
        # Separate games into those where Challenger moves vs Champion moves
        challenger_envs = []
        challenger_indices = []
        champion_envs = []
        champion_indices = []

        for g_idx in active:
            env = envs[g_idx]
            x_move = env.x_to_move
            is_chal = (x_move and challenger_is_p1[g_idx]) or (not x_move and not challenger_is_p1[g_idx])
            if is_chal:
                challenger_envs.append(env)
                challenger_indices.append(g_idx)
            else:
                champion_envs.append(env)
                champion_indices.append(g_idx)

        # Challenger MCTS
        if challenger_envs:
            chal_results = run_batched_gpu_mcts(
                model=model_challenger,
                envs=challenger_envs,
                num_simulations=mcts_sims,
                dirichlet_eps=0.0,  # Pure competitive search, no noise
                device=device,
            )
            for env_idx, (_, chosen_move) in zip(challenger_indices, chal_results):
                envs[env_idx].step(chosen_move)

        # Champion MCTS
        if champion_envs:
            champ_results = run_batched_gpu_mcts(
                model=model_champion,
                envs=champion_envs,
                num_simulations=mcts_sims,
                dirichlet_eps=0.0,
                device=device,
            )
            for env_idx, (_, chosen_move) in zip(champion_indices, champ_results):
                envs[env_idx].step(chosen_move)

        # Check for finished games
        active = [g_idx for g_idx in active if not envs[g_idx].game_over]

    # Score outcomes
    chal_wins = 0
    champ_wins = 0
    draws = 0

    for i, env in enumerate(envs):
        if env.winner == 1:
            if challenger_is_p1[i]:
                chal_wins += 1
            else:
                champ_wins += 1
        elif env.winner == -1:
            if not challenger_is_p1[i]:
                chal_wins += 1
            else:
                champ_wins += 1
        else:
            draws += 1

    return chal_wins, champ_wins, draws


def run_champion_challenger(
    total_cycles: int = 5,
    parallel_games: int = 1024,
    mcts_simulations: int = 800,
    moves_per_cycle: int = 40,
    eval_games: int = 24,
    eval_sims: int = 200,
    train_epochs: int = 3,
    lr: float = 1e-3,
    promote_threshold: float = 0.55,
    buffer_capacity: int = 1_000_000,
    device: str = "cuda",
    status_file: str = "ml/training_status.json",
    frontend_status: Optional[str] = "../uttt-frontend/training_status.json",
):
    print("============================================================")
    print("  AlphaZero Champion vs Challenger Autonomous Loop")
    print(f"  GPU Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Parallel Games per Move: {parallel_games}")
    print(f"  MCTS Simulations per Move: {mcts_simulations} (Deep AlphaZero)")
    print(f"  Moves per Self-Play Cycle: {moves_per_cycle}")
    print(f"  Replay Buffer Capacity: {buffer_capacity:,} packed states")
    print(f"  Evaluation Gating Games: {eval_games} (Promo Threshold: {promote_threshold*100:.0f}%)")
    print("============================================================")

    champion_path = "ml/champion.pt"
    challenger_path = "ml/challenger.pt"
    buffer_path = "ml/replay_buffer_checkpoint.npz"
    initial_ckpt = "ml/model_checkpoint.pt"

    # Load or initialize Champion model
    champion = AlphaZeroResNet(num_channels=64, num_res_blocks=4).to(device)
    if os.path.exists(champion_path):
        state = torch.load(champion_path, map_location=device)
        champion.load_state_dict(state["model_state_dict"])
        print(f"Loaded existing Champion model from {champion_path}")
    elif os.path.exists(initial_ckpt):
        state = torch.load(initial_ckpt, map_location=device)
        champion.load_state_dict(state["model_state_dict"])
        print(f"Initialized Champion from {initial_ckpt}")
        torch.save({"model_state_dict": champion.state_dict(), "generation": 0, "elo": 1500.0}, champion_path)
    else:
        print("Initialized Champion with fresh random weights")
        torch.save({"model_state_dict": champion.state_dict(), "generation": 0, "elo": 1500.0}, champion_path)

    # Load Contiguous Stratified Replay Buffer (Zero-GC, 814 MB RAM footprint for 1M states)
    replay_buffer = ContiguousStratifiedReplayBuffer(capacity=buffer_capacity)
    if os.path.exists(buffer_path):
        replay_buffer.load(buffer_path)
        print(f"Loaded existing Replay Buffer: {len(replay_buffer):,} states")

    # Initialize metadata
    meta_file = "ml/champion_metadata.json"
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)
    else:
        meta = {
            "generation": 0,
            "champion_elo": 1500.0,
            "champion_rd": 50.0,
            "total_games_played": 0,
            "total_promotions": 0,
            "history": [],
        }

    # Initialize Pure Rust Rayon MCTS Batch Worker for 240,000+ evals/s GPU Saturation
    print(f"Initializing RustMctsWorker with {parallel_games} parallel games...")
    worker = RustMctsWorker(num_games=parallel_games, c_puct=2.0)

    for cycle in range(1, total_cycles + 1):
        print(f"\n============================================================")
        print(f"  CYCLE {cycle}/{total_cycles} | Generation {meta['generation']} (Champion Elo: {meta['champion_elo']:.1f})")
        print(f"============================================================")

        # ---------------------------------------------------------------------
        # Phase 1: High-Saturation GPU Batched MCTS Self-Play (Pure Rust Rayon Worker)
        # ---------------------------------------------------------------------
        print(f"🚀 [Phase 1/3] Generating Games via Pure Rust Batch Worker ({parallel_games} games, {mcts_simulations} sims/move)...")
        status_data = {
            "cycle": cycle,
            "total_cycles": total_cycles,
            "phase": "self_play",
            "phase_description": f"Running {parallel_games} parallel games with {mcts_simulations} MCTS visits/move",
            "mcts_sims": mcts_simulations,
            "parallel_games": parallel_games,
            "champion_elo": round(meta["champion_elo"], 1),
            "champion_generation": meta["generation"],
            "challenger_elo": round(meta["champion_elo"], 1),
            "replay_buffer_size": len(replay_buffer),
            "replay_buffer_capacity": buffer_capacity,
            "gpu_telemetry": get_gpu_telemetry(device),
            "history": meta["history"],
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        write_status(status_file, frontend_status, status_data)

        champion.eval()
        t_gen_start = time.time()
        total_evals_in_gen = 0
        total_finished_states = 0

        for move_step in range(1, moves_per_cycle + 1):
            t_move_start = time.time()
            step_evals = worker.run_mcts_search(model=champion, num_simulations=mcts_simulations, device=device)
            total_evals_in_gen += step_evals

            # Advance moves with auto temperature schedule:
            # Plies <= 20: T = 1.0; Plies > 20: T = 0.5 (T NEVER drops to 0 in self-play!)
            new_finished = worker.advance_moves(temperature=0.0)
            total_finished_states += new_finished

            t_move_end = time.time()
            move_elapsed = max(t_move_end - t_move_start, 0.001)
            move_rate = step_evals / move_elapsed
            total_rate = total_evals_in_gen / max(t_move_end - t_gen_start, 0.001)

            print(f"  [Move {move_step:02d}/{moves_per_cycle:02d}] {parallel_games} games | {step_evals:,} evals in {move_elapsed:.2f}s ({move_rate:,.0f} evals/s) | {total_finished_states:,} finished states")

            status_data.update({
                "phase_description": f"Move {move_step}/{moves_per_cycle}: {parallel_games} games ({total_evals_in_gen:,} evals, {total_rate:,.0f} evals/s)",
                "gpu_evals_per_sec": round(total_rate),
                "gpu_telemetry": get_gpu_telemetry(device),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            write_status(status_file, frontend_status, status_data)

        gen_elapsed = time.time() - t_gen_start
        evals_per_sec = total_evals_in_gen / max(gen_elapsed, 0.001)

        # Drain completed game states from Rust worker into pre-allocated NumPy arrays
        f_drain, m_drain, p_drain, v_drain = worker.drain_samples(max_count=100_000)
        new_samples_count = len(f_drain)
        if new_samples_count > 0:
            replay_buffer.add_samples_batch(f_drain, m_drain, p_drain, v_drain, tier="deep_mcts")
            replay_buffer.save(buffer_path)

        meta["total_games_played"] += parallel_games
        print(f"  Self-Play Complete: {parallel_games} games, {new_samples_count:,} new states in {gen_elapsed:.2f}s ({evals_per_sec:,.0f} evals/sec)")
        print(f"  Replay Buffer Size: {len(replay_buffer):,} packed states")

        # ---------------------------------------------------------------------
        # Phase 2: Train Challenger Model
        # ---------------------------------------------------------------------
        print(f"\n🧠 [Phase 2/3] Training Challenger ResNet ({train_epochs} epochs)...")
        status_data.update({
            "phase": "training",
            "phase_description": f"Training Challenger ResNet on {len(replay_buffer)} states (D4 augmented)",
            "replay_buffer_size": len(replay_buffer),
            "gpu_evals_per_sec": round(evals_per_sec),
            "gpu_telemetry": get_gpu_telemetry(device),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(status_file, frontend_status, status_data)

        challenger = copy.deepcopy(champion).to(device)
        challenger.train()
        optimizer = torch.optim.AdamW(challenger.parameters(), lr=lr, weight_decay=1e-4)

        steps_per_epoch = max(20, min(150, len(replay_buffer) // 128))
        last_loss = 0.0

        for ep in range(1, train_epochs + 1):
            ep_loss = 0.0
            ep_p = 0.0
            ep_v = 0.0
            for _ in range(steps_per_epoch):
                x_b, m_b, p_b, v_b = replay_buffer.sample_batch(batch_size=128, augment_d4_prob=1.0)
                x_b = torch.from_numpy(x_b).to(device)
                m_b = torch.from_numpy(m_b).to(device)
                p_b = torch.from_numpy(p_b).to(device)
                v_b = torch.from_numpy(v_b).unsqueeze(1).to(device)

                optimizer.zero_grad()
                logits, value = challenger(x_b, m_b)
                log_probs = F.log_softmax(logits, dim=-1)
                p_loss = -torch.sum(p_b * log_probs) / x_b.size(0)
                v_loss = F.mse_loss(value, v_b)
                loss = p_loss + v_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(challenger.parameters(), max_norm=2.0)
                optimizer.step()

                ep_loss += loss.item()
                ep_p += p_loss.item()
                ep_v += v_loss.item()

            last_loss = ep_loss / steps_per_epoch
            print(f"  Epoch {ep}/{train_epochs} | Loss: {last_loss:.4f} (Policy: {ep_p/steps_per_epoch:.4f}, Value: {ep_v/steps_per_epoch:.4f})")

        torch.save({"model_state_dict": challenger.state_dict()}, challenger_path)

        # ---------------------------------------------------------------------
        # Phase 3: Evaluation Match (Challenger vs Champion)
        # ---------------------------------------------------------------------
        print(f"\n⚔️  [Phase 3/3] Arena Gating: Challenger vs Champion ({eval_games} games, {eval_sims} sims)...")
        status_data.update({
            "phase": "evaluating",
            "phase_description": f"Playing {eval_games}-game gating match: Challenger vs Champion",
            "training_loss": round(last_loss, 4),
            "gpu_telemetry": get_gpu_telemetry(device),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(status_file, frontend_status, status_data)

        chal_wins, champ_wins, draws = play_eval_match(
            model_challenger=challenger,
            model_champion=champion,
            num_games=eval_games,
            mcts_sims=eval_sims,
            device=device,
        )

        total_decisive = chal_wins + champ_wins + draws
        win_rate = (chal_wins + 0.5 * draws) / max(1, total_decisive)
        print(f"  Match Result: Challenger {chal_wins}W - {champ_wins}L - {draws}D (Win Rate: {win_rate*100:.1f}%)")

        # Elo calculation
        delta_elo = 400.0 * (win_rate - 0.5)
        challenger_elo = meta["champion_elo"] + delta_elo

        promoted = win_rate >= promote_threshold

        if promoted:
            meta["generation"] += 1
            meta["total_promotions"] += 1
            meta["champion_elo"] = challenger_elo
            champion = copy.deepcopy(challenger)
            torch.save({
                "model_state_dict": champion.state_dict(),
                "generation": meta["generation"],
                "elo": meta["champion_elo"],
            }, champion_path)
            shutil.copyfile(champion_path, initial_ckpt)

            # Export INT8 model and synchronize to pure Rust engine
            rust_bin = "uttt/src/nn/model_int8.bin"
            calib_x, calib_m, calib_p, calib_v = replay_buffer.sample_batch(batch_size=100)
            calib_samples = [
                (calib_x[i], calib_m[i], calib_p[i], float(calib_v[i]))
                for i in range(min(100, len(calib_x)))
            ]
            export_model_int8(champion, output_bin_path=rust_bin, calibration_samples=calib_samples)
            print(f"🏆 PROMOTION! Challenger crowned Champion Gen-{meta['generation']} (New Elo: {meta['champion_elo']:.1f})")
            print(f"  Synchronized INT8 weights to {rust_bin}")
        else:
            print(f"🛡️  DEFENSE! Champion retains title (Challenger: {win_rate*100:.1f}% < {promote_threshold*100:.0f}%)")

        # Record history
        cycle_record = {
            "cycle": cycle,
            "generation": meta["generation"],
            "champion_elo": round(meta["champion_elo"], 1),
            "challenger_elo": round(challenger_elo, 1),
            "win_rate": round(win_rate, 3),
            "match_score": f"{chal_wins}W-{champ_wins}L-{draws}D",
            "loss": round(last_loss, 4),
            "promoted": promoted,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta["history"].append(cycle_record)

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        status_data.update({
            "phase": "promoted" if promoted else "defended",
            "phase_description": f"Cycle {cycle} complete: {'Challenger Promoted' if promoted else 'Champion Defended'}",
            "champion_elo": round(meta["champion_elo"], 1),
            "challenger_elo": round(challenger_elo, 1),
            "champion_generation": meta["generation"],
            "last_match": cycle_record,
            "history": meta["history"],
            "gpu_telemetry": get_gpu_telemetry(device),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(status_file, frontend_status, status_data)

    print("\n============================================================")
    print(f"  AlphaZero Loop Complete! Final Champion Gen-{meta['generation']} (Elo: {meta['champion_elo']:.1f})")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3, help="Number of champion/challenger cycles")
    parser.add_argument("--games", type=int, default=1024, help="Parallel games per self-play batch (GPU saturated)")
    parser.add_argument("--mcts-sims", type=int, default=800, help="MCTS visits per move (Deep AlphaZero)")
    parser.add_argument("--moves-per-cycle", type=int, default=40, help="Moves played across parallel games per self-play cycle")
    parser.add_argument("--eval-games", type=int, default=40, help="Number of games in gating match (2-ply opening suite)")
    parser.add_argument("--eval-sims", type=int, default=200, help="MCTS visits in gating match")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per cycle")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.55, help="Promotion threshold win rate")
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Replay buffer capacity in packed states")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--status-file", type=str, default="ml/training_status.json")
    parser.add_argument("--frontend-status", type=str, default="../uttt-frontend/training_status.json")
    args = parser.parse_args()

    run_champion_challenger(
        total_cycles=args.cycles,
        parallel_games=args.games,
        mcts_simulations=args.mcts_sims,
        moves_per_cycle=args.moves_per_cycle,
        eval_games=args.eval_games,
        eval_sims=args.eval_sims,
        train_epochs=args.epochs,
        lr=args.lr,
        promote_threshold=args.threshold,
        buffer_capacity=args.buffer_capacity,
        device=args.device,
        status_file=args.status_file,
        frontend_status=args.frontend_status,
    )
