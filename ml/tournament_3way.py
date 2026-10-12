"""
3-Way Round-Robin Tournament: Transformer vs ResNet vs AlphaBeta
Evaluates the three primary architectures on 512 balanced mirrored games per pairing:
1. UTTTTransformer (Generation 8 Champion, Elo 2031.1)
2. AlphaZeroResNet (Generation 7 Champion, Elo 1906.4)
3. AlphaBetaV2Bot (Rust Minimax Engine with TT, Killer Moves, History Heuristics, DF-PN)

Guarantees:
- Balanced mirrored pairs: for each position P_k, both 2k and 2k+1 are played from opposite sides.
- Exact cancellation of intrinsic position bias V(P_k) and color bias.
- Full GPU saturation for neural evaluations and Rayon thread pool for AlphaBeta.
- Joint Bradley-Terry maximum likelihood Elo ratings anchored at ResNet = 1906.4.
- Double-win (2-0 vs 0-2) and symmetrical split (1-1) tracking.
"""

from dataclasses import dataclass
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import AlphaZeroResNet
from ml.transformer_model import UTTTTransformer
from ml.rust_mcts_worker import RustMctsWorker
from ml.parallel_mirror_eval import (
    PairedPosition,
    generate_candidate_positions,
    mine_interesting_positions,
)


@dataclass
class MatchResult:
    competitor_a: str
    competitor_b: str
    total_games: int
    total_pairs: int
    a_wins: int
    b_wins: int
    draws: int
    a_double_wins: int
    b_double_wins: int
    symmetrical_splits: int
    a_win_rate: float
    margin_of_error: float
    delta_elo: float
    elapsed_seconds: float


def solve_bradley_terry(
    competitors: List[str],
    match_results: List[MatchResult],
    anchor_name: str = "AlphaZeroResNet",
    anchor_elo: float = 1906.4,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes joint maximum-likelihood Bradley-Terry ratings anchored at anchor_elo.
    Uses iterative Minorization-Maximization (MM) algorithm.
    """
    n = len(competitors)
    name_to_idx = {name: i for i, name in enumerate(competitors)}

    # W[i, j] = wins for i against j (draws count as 0.5)
    W = np.zeros((n, n), dtype=np.float64)
    for m in match_results:
        i = name_to_idx[m.competitor_a]
        j = name_to_idx[m.competitor_b]
        W[i, j] = m.a_wins + 0.5 * m.draws
        W[j, i] = m.b_wins + 0.5 * m.draws

    # Initialize skill parameters p_i
    p = np.ones(n, dtype=np.float64)
    total_wins = np.sum(W, axis=1)

    for _ in range(max_iter):
        p_prev = p.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i != j and (W[i, j] + W[j, i]) > 0:
                    denom += (W[i, j] + W[j, i]) / (p_prev[i] + p_prev[j])
            if denom > 0:
                p[i] = total_wins[i] / denom
            else:
                p[i] = 1.0

        # Normalize to geometric mean 1.0
        p = p / np.exp(np.mean(np.log(p)))
        if np.max(np.abs(p - p_prev)) < tol:
            break

    # Convert to Elo scale anchored at anchor_name
    anchor_idx = name_to_idx[anchor_name]
    anchor_p = p[anchor_idx]

    elo_ratings = {}
    for i, name in enumerate(competitors):
        elo = anchor_elo + 400.0 * math.log10(p[i] / anchor_p)
        elo_ratings[name] = round(elo, 1)

    return elo_ratings


def run_neural_vs_neural_match(
    model_a: torch.nn.Module,
    name_a: str,
    model_b: torch.nn.Module,
    name_b: str,
    positions: List[PairedPosition],
    num_sims: int = 100,
    device: str = "cuda",
) -> MatchResult:
    """
    Executes a balanced mirrored match between two neural models (e.g. Transformer vs ResNet).
    Both models evaluate leaves simultaneously via dual-model GPU batching.
    """
    num_pairs = len(positions)
    num_games = num_pairs * 2

    model_a.eval()
    model_b.eval()

    worker = RustMctsWorker(num_games=num_games, dirichlet_eps=0.0)

    pos_x_to_move = np.zeros(num_pairs, dtype=bool)
    for i, pos in enumerate(positions):
        pos_x_to_move[i] = pos.x_to_move
        worker.set_game_moves(2 * i, pos.moves)
        worker.set_game_moves(2 * i + 1, pos.moves)

    game_to_model = np.zeros(num_games, dtype=bool)
    finished_count = 0
    ply = 0
    max_plies = 81
    t_start = time.time()

    while ply < max_plies and finished_count < num_games:
        # Determine active player model for each game:
        # True = Model A evaluates; False = Model B evaluates
        for i in range(num_pairs):
            pos_x = pos_x_to_move[i]
            g0 = 2 * i
            g1 = 2 * i + 1

            cur_x0 = worker.get_game_x_to_move(g0)
            game_to_model[g0] = (cur_x0 == pos_x)

            cur_x1 = worker.get_game_x_to_move(g1)
            game_to_model[g1] = (cur_x1 != pos_x)

        # Run MCTS simulations across all games in parallel
        for _ in range(num_sims):
            worker.step_simulation_dual(model_a, model_b, game_to_model, device=device)

        # Greedy argmax advance
        finished_count = worker.advance_moves_tournament(temperature=0.0)
        ply += 1

    elapsed = max(time.time() - t_start, 0.001)

    # Score compilation
    a_wins = 0
    b_wins = 0
    draws = 0
    a_double_wins = 0
    b_double_wins = 0
    symmetrical_splits = 0

    for i in range(num_pairs):
        pos_x = pos_x_to_move[i]
        a_side_g0 = 1 if pos_x else -1
        a_side_g1 = -1 if pos_x else 1

        st0 = worker.get_game_status(2 * i)
        st1 = worker.get_game_status(2 * i + 1)

        # Rescale: 1=P1 win, -1=P2 win, 2=draw
        r0 = 1.0 if st0 == a_side_g0 else (-1.0 if st0 == -a_side_g0 else 0.0)
        r1 = 1.0 if st1 == a_side_g1 else (-1.0 if st1 == -a_side_g1 else 0.0)

        if r0 > 0: a_wins += 1
        elif r0 < 0: b_wins += 1
        else: draws += 1

        if r1 > 0: a_wins += 1
        elif r1 < 0: b_wins += 1
        else: draws += 1

        if r0 > 0 and r1 > 0:
            a_double_wins += 1
        elif r0 < 0 and r1 < 0:
            b_double_wins += 1
        elif (r0 > 0 and r1 < 0) or (r0 < 0 and r1 > 0):
            symmetrical_splits += 1

    total_pts = a_wins + 0.5 * draws
    win_rate = total_pts / num_games
    moe = 1.96 * math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / num_games)
    delta_elo = 400.0 * math.log10((total_pts + 1.0) / (num_games - total_pts + 1.0))

    return MatchResult(
        competitor_a=name_a,
        competitor_b=name_b,
        total_games=num_games,
        total_pairs=num_pairs,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_double_wins=a_double_wins,
        b_double_wins=b_double_wins,
        symmetrical_splits=symmetrical_splits,
        a_win_rate=win_rate,
        margin_of_error=moe,
        delta_elo=delta_elo,
        elapsed_seconds=elapsed,
    )


def run_neural_vs_alphabeta_match(
    neural_model: torch.nn.Module,
    neural_name: str,
    positions: List[PairedPosition],
    num_sims: int = 100,
    alphabeta_ms: int = 10,
    device: str = "cuda",
) -> MatchResult:
    """
    Executes a balanced mirrored match between a Neural Model and AlphaBetaV2.
    - Game 2i: Neural plays first from P_k, AlphaBeta plays second.
    - Game 2i+1: AlphaBeta plays first from P_k, Neural plays second.
    - Alternates full-batch GPU MCTS for Neural with 16-thread Rayon Minimax for AlphaBeta.
    """
    num_pairs = len(positions)
    num_games = num_pairs * 2

    neural_model.eval()
    worker = RustMctsWorker(num_games=num_games, dirichlet_eps=0.0)

    pos_x_to_move = np.zeros(num_pairs, dtype=bool)
    game_histories: List[List[int]] = []
    for i, pos in enumerate(positions):
        pos_x_to_move[i] = pos.x_to_move
        worker.set_game_moves(2 * i, pos.moves)
        worker.set_game_moves(2 * i + 1, pos.moves)
        game_histories.append(list(pos.moves))
        game_histories.append(list(pos.moves))

    # Phase 0: In odd games (2i+1), AlphaBeta plays the first move from P_k.
    # This immediately aligns all 512 games to Neural's turn!
    odd_indices = [2 * i + 1 for i in range(num_pairs)]
    odd_histories = [game_histories[g] for g in odd_indices]
    first_ab_moves = worker.choose_alphabeta_moves(odd_histories, time_limit_ms=alphabeta_ms)
    for idx, g in enumerate(odd_indices):
        game_histories[g].append(first_ab_moves[idx])
    worker.apply_moves_batch(odd_indices, first_ab_moves)

    finished_count = 0
    ply = 0
    max_plies = 81
    t_start = time.time()

    while ply < max_plies and finished_count < num_games:
        # Phase 1: Neural Move Phase (All unfinished games are Neural's turn!)
        active_games = [g for g in range(num_games) if worker.get_game_status(g) == 0]
        if not active_games:
            break

        # Run MCTS simulations for Neural model on GPU
        for _ in range(num_sims):
            worker.step_simulation(neural_model, device=device)

        # Advance Neural moves and record chosen actions
        finished_count, chosen_neural_moves = worker.advance_moves_tournament_recorded(temperature=0.0)

        for g in active_games:
            mv = chosen_neural_moves[g]
            if mv != 255:
                game_histories[g].append(int(mv))

        # Check if all games finished after Neural moves
        active_games = [g for g in range(num_games) if worker.get_game_status(g) == 0]
        if not active_games or finished_count >= num_games:
            break

        # Phase 2: AlphaBeta Move Phase (All active games are AlphaBeta's turn!)
        active_histories = [game_histories[g] for g in active_games]
        ab_moves = worker.choose_alphabeta_moves(active_histories, time_limit_ms=alphabeta_ms)

        for idx, g in enumerate(active_games):
            game_histories[g].append(ab_moves[idx])

        worker.apply_moves_batch(active_games, ab_moves)

        # Check finished count after AlphaBeta moves
        finished_count = sum(1 for g in range(num_games) if worker.get_game_status(g) != 0)
        ply += 1

    elapsed = max(time.time() - t_start, 0.001)

    # Score compilation
    neural_wins = 0
    ab_wins = 0
    draws = 0
    neural_double_wins = 0
    ab_double_wins = 0
    symmetrical_splits = 0

    for i in range(num_pairs):
        pos_x = pos_x_to_move[i]
        neural_side_g0 = 1 if pos_x else -1
        neural_side_g1 = -1 if pos_x else 1

        st0 = worker.get_game_status(2 * i)
        st1 = worker.get_game_status(2 * i + 1)

        r0 = 1.0 if st0 == neural_side_g0 else (-1.0 if st0 == -neural_side_g0 else 0.0)
        r1 = 1.0 if st1 == neural_side_g1 else (-1.0 if st1 == -neural_side_g1 else 0.0)

        if r0 > 0: neural_wins += 1
        elif r0 < 0: ab_wins += 1
        else: draws += 1

        if r1 > 0: neural_wins += 1
        elif r1 < 0: ab_wins += 1
        else: draws += 1

        if r0 > 0 and r1 > 0:
            neural_double_wins += 1
        elif r0 < 0 and r1 < 0:
            ab_double_wins += 1
        elif (r0 > 0 and r1 < 0) or (r0 < 0 and r1 > 0):
            symmetrical_splits += 1

    total_pts = neural_wins + 0.5 * draws
    win_rate = total_pts / num_games
    moe = 1.96 * math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / num_games)
    delta_elo = 400.0 * math.log10((total_pts + 1.0) / (num_games - total_pts + 1.0))

    return MatchResult(
        competitor_a=neural_name,
        competitor_b="AlphaBetaV2",
        total_games=num_games,
        total_pairs=num_pairs,
        a_wins=neural_wins,
        b_wins=ab_wins,
        draws=draws,
        a_double_wins=neural_double_wins,
        b_double_wins=ab_double_wins,
        symmetrical_splits=symmetrical_splits,
        a_win_rate=win_rate,
        margin_of_error=moe,
        delta_elo=delta_elo,
        elapsed_seconds=elapsed,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"================================================================================")
    print(f"      3-WAY ROUND-ROBIN TOURNAMENT: TRANSFORMER vs RESNET vs ALPHABETA          ")
    print(f"================================================================================")
    print(f"Device: {device} | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transformer_path = os.path.join(base_dir, "ml", "champion.pt")
    resnet_path = os.path.join(base_dir, "ml", "champion_gen7_resnet.pt")

    print(f"\n[1/4] Loading competitor models...")
    print(f"  - Loading UTTTTransformer (Gen 8 Champion) from {transformer_path}...")
    transformer = UTTTTransformer().to(device)
    transformer_ckpt = torch.load(transformer_path, map_location=device, weights_only=False)
    if isinstance(transformer_ckpt, dict) and "model_state_dict" in transformer_ckpt:
        transformer.load_state_dict(transformer_ckpt["model_state_dict"])
    elif isinstance(transformer_ckpt, dict):
        transformer.load_state_dict(transformer_ckpt)
    else:
        transformer = transformer_ckpt.to(device)
    transformer.eval()
    t_params = sum(p.numel() for p in transformer.parameters())
    print(f"    UTTTTransformer loaded successfully ({t_params:,} parameters).")

    print(f"  - Loading AlphaZeroResNet (Gen 7 Champion) from {resnet_path}...")
    resnet = AlphaZeroResNet().to(device)
    resnet_ckpt = torch.load(resnet_path, map_location=device, weights_only=False)
    if isinstance(resnet_ckpt, dict) and "model_state_dict" in resnet_ckpt:
        resnet.load_state_dict(resnet_ckpt["model_state_dict"])
    elif isinstance(resnet_ckpt, dict):
        resnet.load_state_dict(resnet_ckpt)
    else:
        resnet = resnet_ckpt.to(device)
    resnet.eval()
    r_params = sum(p.numel() for p in resnet.parameters())
    print(f"    AlphaZeroResNet loaded successfully ({r_params:,} parameters).")

    print(f"  - AlphaBetaV2: Native Rust Minimax loaded via libuttt.so C-ABI.")

    # 2. Generate balanced opening suite
    print(f"\n[2/4] Generating balanced opening suite (K=256 paired positions, N=512 games/match)...")
    paired_positions = mine_interesting_positions(
        challenger=transformer,
        champion=resnet,
        num_positions=256,
        device=device,
        seed=42,
    )
    print(f"    Generated {len(paired_positions)} contested paired positions across all 9 subboards.")

    # 3. Execute Round-Robin Matches
    print(f"\n[3/4] Executing 3-Way Round-Robin Matches...")
    matches: List[MatchResult] = []

    # Match 1: UTTTTransformer vs AlphaZeroResNet
    print(f"\n--- Match 1: UTTTTransformer vs AlphaZeroResNet (512 games) ---")
    m1 = run_neural_vs_neural_match(
        model_a=transformer,
        name_a="UTTTTransformer",
        model_b=resnet,
        name_b="AlphaZeroResNet",
        positions=paired_positions,
        num_sims=100,
        device=device,
    )
    matches.append(m1)
    print(f"    Result: {m1.a_wins}W - {m1.b_wins}L - {m1.draws}D (Win Rate: {m1.a_win_rate*100:.2f}% ± {m1.margin_of_error*100:.2f}%)")
    print(f"    Double Wins: Transformer {m1.a_double_wins} | ResNet {m1.b_double_wins} | Splits: {m1.symmetrical_splits}")
    print(f"    Delta Elo: {m1.delta_elo:+.1f} | Elapsed: {m1.elapsed_seconds:.1f}s")

    # Match 2: UTTTTransformer vs AlphaBetaV2
    print(f"\n--- Match 2: UTTTTransformer vs AlphaBetaV2 (512 games) ---")
    m2 = run_neural_vs_alphabeta_match(
        neural_model=transformer,
        neural_name="UTTTTransformer",
        positions=paired_positions,
        num_sims=100,
        alphabeta_ms=10,
        device=device,
    )
    matches.append(m2)
    print(f"    Result: {m2.a_wins}W - {m2.b_wins}L - {m2.draws}D (Win Rate: {m2.a_win_rate*100:.2f}% ± {m2.margin_of_error*100:.2f}%)")
    print(f"    Double Wins: Transformer {m2.a_double_wins} | AlphaBeta {m2.b_double_wins} | Splits: {m2.symmetrical_splits}")
    print(f"    Delta Elo: {m2.delta_elo:+.1f} | Elapsed: {m2.elapsed_seconds:.1f}s")

    # Match 3: AlphaZeroResNet vs AlphaBetaV2
    print(f"\n--- Match 3: AlphaZeroResNet vs AlphaBetaV2 (512 games) ---")
    m3 = run_neural_vs_alphabeta_match(
        neural_model=resnet,
        neural_name="AlphaZeroResNet",
        positions=paired_positions,
        num_sims=100,
        alphabeta_ms=10,
        device=device,
    )
    matches.append(m3)
    print(f"    Result: {m3.a_wins}W - {m3.b_wins}L - {m3.draws}D (Win Rate: {m3.a_win_rate*100:.2f}% ± {m3.margin_of_error*100:.2f}%)")
    print(f"    Double Wins: ResNet {m3.a_double_wins} | AlphaBeta {m3.b_double_wins} | Splits: {m3.symmetrical_splits}")
    print(f"    Delta Elo: {m3.delta_elo:+.1f} | Elapsed: {m3.elapsed_seconds:.1f}s")

    # 4. Joint Bradley-Terry Elo Estimation
    print(f"\n[4/4] Computing Joint Bradley-Terry Maximum Likelihood Elo Ratings...")
    competitors = ["UTTTTransformer", "AlphaZeroResNet", "AlphaBetaV2"]
    elo_ratings = solve_bradley_terry(
        competitors=competitors,
        match_results=matches,
        anchor_name="AlphaZeroResNet",
        anchor_elo=1906.4,
    )

    print(f"\n================================================================================")
    print(f"                         FINAL TOURNAMENT LEADERBOARD                           ")
    print(f"================================================================================")
    print(f"{'Rank':<5} | {'Competitor':<18} | {'Elo Rating':<11} | {'Type / Description':<30}")
    print(f"------+--------------------+-------------+-------------------------------")
    sorted_ranks = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    descriptions = {
        "UTTTTransformer": "Gen-8 Champion (Vision Transformer)",
        "AlphaZeroResNet": "Gen-7 Champion (Residual CNN)",
        "AlphaBetaV2": "Pure Minimax (C-ABI / TT / DF-PN)",
    }
    for rank, (name, elo) in enumerate(sorted_ranks, start=1):
        print(f"#{rank:<4} | {name:<18} | {elo:>9.1f}   | {descriptions[name]:<30}")

    # Output detailed match breakdown
    print(f"\nDetailed Head-to-Head Cross-Table:")
    print(f"{'Match':<35} | {'Score':<14} | {'Win Rate':<14} | {'2-0':<5} | {'0-2':<5} | {'1-1':<5}")
    print(f"------------------------------------+----------------+----------------+-------+-------+-------")
    for m in matches:
        match_label = f"{m.competitor_a} vs {m.competitor_b}"
        score_str = f"{m.a_wins}W-{m.b_wins}L-{m.draws}D"
        wr_str = f"{m.a_win_rate*100:5.1f}% ±{m.margin_of_error*100:4.1f}%"
        print(f"{match_label:<35} | {score_str:<14} | {wr_str:<14} | {m.a_double_wins:<5} | {m.b_double_wins:<5} | {m.symmetrical_splits:<5}")

    # Save results to JSON
    output_path = os.path.join(base_dir, "ml", "tournament_3way_results.json")
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "leaderboard": [
            {"rank": rank, "competitor": name, "elo": elo, "description": descriptions[name]}
            for rank, (name, elo) in enumerate(sorted_ranks, start=1)
        ],
        "matches": [
            {
                "competitor_a": m.competitor_a,
                "competitor_b": m.competitor_b,
                "total_games": m.total_games,
                "total_pairs": m.total_pairs,
                "a_wins": m.a_wins,
                "b_wins": m.b_wins,
                "draws": m.draws,
                "a_double_wins": m.a_double_wins,
                "b_double_wins": m.b_double_wins,
                "symmetrical_splits": m.symmetrical_splits,
                "a_win_rate": round(m.a_win_rate, 4),
                "margin_of_error": round(m.margin_of_error, 4),
                "delta_elo": round(m.delta_elo, 1),
                "elapsed_seconds": round(m.elapsed_seconds, 2),
            }
            for m in matches
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nTournament results saved to: {output_path}")


if __name__ == "__main__":
    main()
