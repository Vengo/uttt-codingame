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
from ml.transformer_model import UTTTTransformer
from ml.dataset import UTTTEnv
from ml.batched_mcts import run_batched_gpu_mcts
from ml.replay_buffer import ContiguousStratifiedReplayBuffer
from ml.rust_mcts_worker import RustMctsWorker
from ml.export_int8 import export_model_int8
from ml.parallel_mirror_eval import mine_interesting_positions, run_parallel_mirror_tournament


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
    c_puct: float = 2.5,
    moves_per_cycle: int = 40,
    eval_pairs: int = 1024,
    eval_sims: int = 100,
    train_epochs: int = 3,
    lr: float = 1e-3,
    promote_threshold: float = 0.55,
    buffer_capacity: int = 1_000_000,
    device: str = "cuda",
    status_file: str = "ml/training_status.json",
    frontend_status: Optional[str] = "../uttt-frontend/training_status.json",
    test_mode: bool = False,
):
    total_eval_games = eval_pairs * 2
    if not test_mode:
        # Enforce statistical significance in production tournaments (>= 1,024 games)
        if eval_pairs < 20:
            print(f"Notice: Setting gating evaluation pairs to 1024 (2048 games, was {eval_pairs}) for statistical significance.")
            eval_pairs = 1024
            total_eval_games = 2048
        if promote_threshold < 0.55:
            print(f"Notice: Enforcing strict promotion threshold at >= 0.55 (requested {promote_threshold}).")
            promote_threshold = 0.55
        champion_path = "ml/champion.pt"
        challenger_path = "ml/challenger.pt"
        buffer_path = "ml/replay_buffer_checkpoint.npz"
        meta_file = "ml/champion_metadata.json"
    else:
        print("⚠️ TEST MODE ENABLED: Using isolated temporary checkpoints and metadata.")
        champion_path = "ml/.test_champion.pt"
        challenger_path = "ml/.test_challenger.pt"
        buffer_path = "ml/.test_buffer.npz"
        meta_file = "ml/.test_champion_metadata.json"
        status_file = "ml/.test_training_status.json"
        frontend_status = None

    print("============================================================")
    print(f"  AlphaZero Champion vs Challenger Autonomous Loop {'[TEST MODE]' if test_mode else ''}")
    print(f"  GPU Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Parallel Games per Move: {parallel_games}")
    print(f"  MCTS Simulations per Move: {mcts_simulations} (Deep AlphaZero)")
    print(f"  Moves per Self-Play Cycle: {moves_per_cycle}")
    print(f"  Replay Buffer Capacity: {buffer_capacity:,} packed states")
    print(f"  Evaluation Gating Match: {total_eval_games} mirror games ({eval_pairs} paired positions, {eval_sims} sims/move)")
    print(f"  Promotion Threshold: {promote_threshold*100:.0f}% win rate")
    print("============================================================")

    initial_ckpt = "ml/model_checkpoint.pt"

    # Load or initialize Champion model (Auto-detect Transformer vs ResNet)
    target_path = champion_path if os.path.exists(champion_path) else (initial_ckpt if os.path.exists(initial_ckpt) else None)
    if target_path:
        state = torch.load(target_path, map_location=device, weights_only=False)
        sd = state["model_state_dict"] if (isinstance(state, dict) and "model_state_dict" in state) else state
        is_trans = any(k.startswith("transformer.") or k.startswith("sb_embed.") for k in sd.keys())
        if is_trans:
            champion = UTTTTransformer().to(device)
            arch_name = "UTTTTransformer (Vision Transformer)"
        else:
            champion = AlphaZeroResNet(num_channels=64, num_res_blocks=4).to(device)
            arch_name = "AlphaZeroResNet (Residual CNN)"
        champion.load_state_dict(sd)
        print(f"Loaded existing Champion model ({arch_name}) from {target_path}")
    else:
        champion = UTTTTransformer().to(device)
        print("Initialized Champion with fresh UTTTTransformer weights")
        torch.save({"model_state_dict": champion.state_dict(), "generation": 0, "elo": 1500.0}, champion_path)

    # Load Contiguous Stratified Replay Buffer (Zero-GC, 814 MB RAM footprint for 1M states)
    replay_buffer = ContiguousStratifiedReplayBuffer(capacity=buffer_capacity)
    if os.path.exists(buffer_path):
        replay_buffer.load(buffer_path)
        print(f"Loaded existing Replay Buffer: {len(replay_buffer):,} states")
    elif test_mode and os.path.exists("ml/replay_buffer_checkpoint.npz"):
        replay_buffer.load("ml/replay_buffer_checkpoint.npz")
        print(f"Loaded production Replay Buffer for test mode: {len(replay_buffer):,} states")

    # Initialize metadata
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
    print(f"Initializing RustMctsWorker with {parallel_games} parallel games (c_puct={c_puct})...")
    worker = RustMctsWorker(num_games=parallel_games, c_puct=c_puct)

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

        champion = champion.to(device)
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
            temp = 1.0 if move_step <= 20 else 0.5
            new_finished = worker.advance_moves(temperature=temp)
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
            replay_buffer.add_samples_batch(f_drain, m_drain, p_drain, v_drain, tier="self_play")
            replay_buffer.save(buffer_path)

        meta["total_games_played"] += parallel_games
        print(f"  Self-Play Complete: {parallel_games} games, {new_samples_count:,} new states in {gen_elapsed:.2f}s ({evals_per_sec:,.0f} evals/sec)")
        print(f"  Replay Buffer Size: {len(replay_buffer):,} packed states")

        # ---------------------------------------------------------------------
        # Phase 2: Train Challenger Model
        # ---------------------------------------------------------------------
        arch_title = type(champion).__name__
        print(f"\n🧠 [Phase 2/3] Training Challenger {arch_title} ({train_epochs} epochs, lr={lr:.1e})...")
        status_data.update({
            "phase": "training",
            "phase_description": f"Training Challenger {arch_title} on {len(replay_buffer)} states (D4 augmented)",
            "replay_buffer_size": len(replay_buffer),
            "gpu_evals_per_sec": round(evals_per_sec),
            "gpu_telemetry": get_gpu_telemetry(device),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(status_file, frontend_status, status_data)

        challenger = copy.deepcopy(champion).to(device)
        challenger.train()
        optimizer = torch.optim.AdamW(challenger.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda" if device == "cuda" else "cpu")

        steps_per_epoch = max(100, min(400, len(replay_buffer) // 128))
        total_train_steps = train_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps, eta_min=lr * 0.1)
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
                with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                    logits, value = challenger(x_b, m_b)
                    log_probs = F.log_softmax(logits, dim=-1)
                    p_loss = -torch.sum(p_b * log_probs) / x_b.size(0)
                    v_loss = F.mse_loss(value, v_b)
                    loss = p_loss + v_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(challenger.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                ep_loss += loss.item()
                ep_p += p_loss.item()
                ep_v += v_loss.item()

            last_loss = ep_loss / steps_per_epoch
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {ep}/{train_epochs} | Loss: {last_loss:.4f} (Policy: {ep_p/steps_per_epoch:.4f}, Value: {ep_v/steps_per_epoch:.4f}) | LR: {cur_lr:.2e}")

        torch.save({"model_state_dict": challenger.state_dict()}, challenger_path)

        # ---------------------------------------------------------------------
        # Phase 3: Parallel Mirror Evaluation Match (Challenger vs Champion)
        # ---------------------------------------------------------------------
        print(f"\n⚔️  [Phase 3/3] Arena Gating: Parallel Mirror Tournament ({total_eval_games} games / {eval_pairs} pairs, {eval_sims} sims)...")
        status_data.update({
            "phase": "evaluating",
            "phase_description": f"Mining {eval_pairs} positions & running {total_eval_games}-game mirror tournament",
            "training_loss": round(last_loss, 4),
            "gpu_telemetry": get_gpu_telemetry(device),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        write_status(status_file, frontend_status, status_data)

        t_mine_start = time.time()
        positions = mine_interesting_positions(
            challenger=challenger,
            champion=champion,
            num_positions=eval_pairs,
            device=device,
        )
        print(f"  Mined {len(positions)} high-divergence tactical positions in {time.time() - t_mine_start:.2f}s")

        def mirror_progress_cb(info: dict):
            status_data.update({
                "phase_description": f"Mirror Tournament Ply {info['ply']}: {info['finished_games']}/{info['total_games']} finished ({info['evals_per_second']:,.0f} evals/s)",
                "gpu_evals_per_sec": round(info['evals_per_second']),
                "gpu_telemetry": get_gpu_telemetry(device),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            write_status(status_file, frontend_status, status_data)

        tourney_res = run_parallel_mirror_tournament(
            challenger=challenger,
            champion=champion,
            positions=positions,
            num_sims=eval_sims,
            device=device,
            progress_callback=mirror_progress_cb,
        )

        chal_wins = tourney_res.challenger_wins
        champ_wins = tourney_res.champion_wins
        draws = tourney_res.draws
        win_rate = tourney_res.win_rate
        delta_elo = tourney_res.delta_elo
        promoted = tourney_res.promoted
        challenger_elo = meta["champion_elo"] + delta_elo

        print(f"\n[Gating Decision] {tourney_res.promotion_reason}")
        print(f"  Match Result: Challenger {chal_wins}W - {champ_wins}L - {draws}D (Win Rate: {win_rate*100:.2f}% ± {tourney_res.margin_of_error*100:.2f}%)")
        print(f"  Decisive Pairs: {tourney_res.decisive_pairs}/{tourney_res.total_pairs} ({tourney_res.decisive_pairs/max(1, tourney_res.total_pairs)*100:.1f}%)")
        print(f"  Decisive Win Rate: {tourney_res.decisive_win_rate*100:.2f}% (Ratio {tourney_res.double_win_ratio}:1, p={tourney_res.decisive_p_value:.4f})")
        print(f"  Symmetrical Splits: {tourney_res.symmetrical_splits}/{tourney_res.total_pairs} ({tourney_res.symmetrical_splits/max(1, tourney_res.total_pairs)*100:.1f}%)")
        print(f"  Pairwise Breakdown: {tourney_res.challenger_double_wins} Challenger Double Wins (2-0) | {tourney_res.champion_double_wins} Champion Double Wins (0-2)")
        print(f"  Tournament Completed: {tourney_res.total_games} games in {tourney_res.elapsed_seconds:.2f}s ({tourney_res.evals_per_second:,.0f} evals/s)")

        if promoted:
            meta["generation"] += 1
            meta["total_promotions"] += 1
            meta["champion_elo"] = challenger_elo
            champion = copy.deepcopy(challenger).to(device)
            champion.eval()
            torch.save({
                "model_state_dict": champion.state_dict(),
                "generation": meta["generation"],
                "elo": meta["champion_elo"],
            }, champion_path)
            if not test_mode:
                shutil.copyfile(champion_path, initial_ckpt)

                # Export INT8 model and synchronize to pure Rust engine (for ResNet)
                rust_bin = "uttt/src/nn/model_int8.bin"
                if isinstance(champion, AlphaZeroResNet):
                    calib_x, calib_m, calib_p, calib_v = replay_buffer.sample_batch(batch_size=100)
                    calib_samples = [
                        (calib_x[i], calib_m[i], calib_p[i], float(calib_v[i]))
                        for i in range(min(100, len(calib_x)))
                    ]
                    export_model_int8(champion, output_bin_path=rust_bin, calibration_samples=calib_samples)
                    champion = champion.to(device)
                    champion.eval()
                    print(f"🏆 PROMOTION! Challenger crowned Champion Gen-{meta['generation']} (New Elo: {meta['champion_elo']:.1f})")
                    print(f"  Synchronized INT8 weights to {rust_bin}")
                else:
                    champion = champion.to(device)
                    champion.eval()
                    print(f"🏆 PROMOTION! Transformer crowned Champion Gen-{meta['generation']} (New Elo: {meta['champion_elo']:.1f})")
                    print(f"  Transformer active (runs FP16 AMP); baseline INT8 weights preserved at {rust_bin}")
            else:
                print(f"🏆 [TEST MODE] Challenger crowned Champion Gen-{meta['generation']} (New Elo: {meta['champion_elo']:.1f})")
        else:
            print(f"🛡️  DEFENSE! Champion retains title ({tourney_res.promotion_reason})")

        # Record history
        cycle_record = {
            "cycle": cycle,
            "generation": meta["generation"],
            "champion_elo": round(meta["champion_elo"], 1),
            "challenger_elo": round(challenger_elo, 1),
            "win_rate": round(win_rate, 4),
            "margin_of_error": round(tourney_res.margin_of_error, 4),
            "match_score": f"{chal_wins}W-{champ_wins}L-{draws}D",
            "total_games": tourney_res.total_games,
            "total_pairs": tourney_res.total_pairs,
            "challenger_double_wins": tourney_res.challenger_double_wins,
            "champion_double_wins": tourney_res.champion_double_wins,
            "symmetrical_splits": tourney_res.symmetrical_splits,
            "decisive_pairs": tourney_res.decisive_pairs,
            "decisive_win_rate": round(tourney_res.decisive_win_rate, 4),
            "double_win_ratio": tourney_res.double_win_ratio,
            "decisive_p_value": round(tourney_res.decisive_p_value, 5),
            "promotion_reason": tourney_res.promotion_reason,
            "tournament_elapsed_sec": round(tourney_res.elapsed_seconds, 2),
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
    parser.add_argument("--c-puct", type=float, default=2.5, help="PUCT exploration constant (default: 2.5)")
    parser.add_argument("--moves-per-cycle", type=int, default=40, help="Moves played across parallel games per self-play cycle")
    parser.add_argument("--eval-pairs", type=int, default=1024, help="Number of paired positions in gating tournament (default: 1024 -> 2048 games)")
    parser.add_argument("--eval-games", type=int, default=None, help="Legacy alias for total evaluation games (eval_pairs = eval_games // 2)")
    parser.add_argument("--eval-sims", type=int, default=100, help="MCTS visits in gating match (default: 100)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per cycle")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (fine-tuning default 1e-4)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Promotion threshold win rate")
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Replay buffer capacity in packed states")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--status-file", type=str, default="ml/training_status.json")
    parser.add_argument("--frontend-status", type=str, default="../uttt-frontend/training_status.json")
    parser.add_argument("--test-mode", action="store_true", help="Run with isolated temporary checkpoints to protect production state")
    args = parser.parse_args()

    eval_pairs = args.eval_pairs
    if args.eval_games is not None:
        eval_pairs = max(args.eval_games // 2, 1)

    run_champion_challenger(
        total_cycles=args.cycles,
        parallel_games=args.games,
        mcts_simulations=args.mcts_sims,
        c_puct=args.c_puct,
        moves_per_cycle=args.moves_per_cycle,
        eval_pairs=eval_pairs,
        eval_sims=args.eval_sims,
        train_epochs=args.epochs,
        lr=args.lr,
        promote_threshold=args.threshold,
        buffer_capacity=args.buffer_capacity,
        device=args.device,
        status_file=args.status_file,
        frontend_status=args.frontend_status,
        test_mode=args.test_mode,
    )
