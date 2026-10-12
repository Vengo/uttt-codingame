"""
Benchmarking UTTTTransformer and AlphaZeroResNet against umcts_v2
Also calibrates against AlphaBeta (V1) to bridge the original 1713 vs 1601 rating gap.

Matches:
1. UTTTTransformer vs umcts_v2 (512 mirrored games)
2. AlphaZeroResNet vs umcts_v2 (512 mirrored games)
3. AlphaBeta (V1) vs umcts_v2 (512 mirrored games)

Solves joint 5-Way Bradley-Terry MLE Leaderboard anchored at AlphaZeroResNet = 1906.4.
"""

from dataclasses import dataclass
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import AlphaZeroResNet
from ml.transformer_model import UTTTTransformer
from ml.rust_mcts_worker import RustMctsWorker
from ml.parallel_mirror_eval import (
    PairedPosition,
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
    n = len(competitors)
    name_to_idx = {name: i for i, name in enumerate(competitors)}

    W = np.zeros((n, n), dtype=np.float64)
    for m in match_results:
        if m.competitor_a in name_to_idx and m.competitor_b in name_to_idx:
            i = name_to_idx[m.competitor_a]
            j = name_to_idx[m.competitor_b]
            W[i, j] += m.a_wins + 0.5 * m.draws
            W[j, i] += m.b_wins + 0.5 * m.draws

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

        p = p / np.exp(np.mean(np.log(p)))
        if np.max(np.abs(p - p_prev)) < tol:
            break

    anchor_idx = name_to_idx[anchor_name]
    anchor_p = p[anchor_idx]

    elo_ratings = {}
    for i, name in enumerate(competitors):
        elo = anchor_elo + 400.0 * math.log10(p[i] / anchor_p)
        elo_ratings[name] = round(elo, 1)

    return elo_ratings


def run_neural_vs_bot_match(
    neural_model: torch.nn.Module,
    neural_name: str,
    bot_name: str,
    bot_version: str,
    positions: List[PairedPosition],
    num_sims: int = 100,
    bot_ms: int = 10,
    device: str = "cuda",
) -> MatchResult:
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

    # Phase 0: In odd games (2i+1), Bot plays the first move from P_k
    odd_indices = [2 * i + 1 for i in range(num_pairs)]
    odd_histories = [game_histories[g] for g in odd_indices]
    first_bot_moves = worker.choose_alphabeta_moves(
        odd_histories, time_limit_ms=bot_ms, bot_version=bot_version
    )
    for idx, g in enumerate(odd_indices):
        game_histories[g].append(first_bot_moves[idx])
    worker.apply_moves_batch(odd_indices, first_bot_moves)

    finished_count = 0
    ply = 0
    max_plies = 81
    t_start = time.time()

    while ply < max_plies and finished_count < num_games:
        active_games = [g for g in range(num_games) if worker.get_game_status(g) == 0]
        if not active_games:
            break

        # Phase 1: Neural Move Phase
        for _ in range(num_sims):
            worker.step_simulation(neural_model, device=device)

        finished_count, chosen_neural_moves = worker.advance_moves_tournament_recorded(temperature=0.0)

        for g in active_games:
            mv = chosen_neural_moves[g]
            if mv != 255:
                game_histories[g].append(int(mv))

        active_games = [g for g in range(num_games) if worker.get_game_status(g) == 0]
        if not active_games or finished_count >= num_games:
            break

        # Phase 2: Bot Move Phase
        active_histories = [game_histories[g] for g in active_games]
        bot_moves = worker.choose_alphabeta_moves(
            active_histories, time_limit_ms=bot_ms, bot_version=bot_version
        )

        for idx, g in enumerate(active_games):
            game_histories[g].append(bot_moves[idx])

        worker.apply_moves_batch(active_games, bot_moves)
        finished_count = sum(1 for g in range(num_games) if worker.get_game_status(g) != 0)
        ply += 1

    elapsed = max(time.time() - t_start, 0.001)

    neural_wins = 0
    bot_wins = 0
    draws = 0
    neural_double_wins = 0
    bot_double_wins = 0
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
        elif r0 < 0: bot_wins += 1
        else: draws += 1

        if r1 > 0: neural_wins += 1
        elif r1 < 0: bot_wins += 1
        else: draws += 1

        if r0 > 0 and r1 > 0:
            neural_double_wins += 1
        elif r0 < 0 and r1 < 0:
            bot_double_wins += 1
        elif (r0 > 0 and r1 < 0) or (r0 < 0 and r1 > 0):
            symmetrical_splits += 1

    total_pts = neural_wins + 0.5 * draws
    win_rate = total_pts / num_games
    moe = 1.96 * math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / num_games)
    delta_elo = 400.0 * math.log10((total_pts + 1.0) / (num_games - total_pts + 1.0))

    return MatchResult(
        competitor_a=neural_name,
        competitor_b=bot_name,
        total_games=num_games,
        total_pairs=num_pairs,
        a_wins=neural_wins,
        b_wins=bot_wins,
        draws=draws,
        a_double_wins=neural_double_wins,
        b_double_wins=bot_double_wins,
        symmetrical_splits=symmetrical_splits,
        a_win_rate=win_rate,
        margin_of_error=moe,
        delta_elo=delta_elo,
        elapsed_seconds=elapsed,
    )


def run_bot_vs_bot_match(
    name_a: str,
    version_a: str,
    name_b: str,
    version_b: str,
    positions: List[PairedPosition],
    bot_ms: int = 10,
) -> MatchResult:
    num_pairs = len(positions)
    num_games = num_pairs * 2

    worker = RustMctsWorker(num_games=num_games, dirichlet_eps=0.0)

    pos_x_to_move = np.zeros(num_pairs, dtype=bool)
    game_histories: List[List[int]] = []
    for i, pos in enumerate(positions):
        pos_x_to_move[i] = pos.x_to_move
        worker.set_game_moves(2 * i, pos.moves)
        worker.set_game_moves(2 * i + 1, pos.moves)
        game_histories.append(list(pos.moves))
        game_histories.append(list(pos.moves))

    # Phase 0: In odd games (2i+1), Bot B plays the first move from P_k
    odd_indices = [2 * i + 1 for i in range(num_pairs)]
    odd_histories = [game_histories[g] for g in odd_indices]
    first_b_moves = worker.choose_alphabeta_moves(
        odd_histories, time_limit_ms=bot_ms, bot_version=version_b
    )
    for idx, g in enumerate(odd_indices):
        game_histories[g].append(first_b_moves[idx])
    worker.apply_moves_batch(odd_indices, first_b_moves)

    finished_count = 0
    ply = 0
    max_plies = 81
    t_start = time.time()

    while ply < max_plies and finished_count < num_games:
        active_games = [g for g in range(num_games) if worker.get_game_status(g) == 0]
        if not active_games:
            break

        active_histories = [game_histories[g] for g in active_games]
        a_moves = worker.choose_alphabeta_moves(
            active_histories, time_limit_ms=bot_ms, bot_version=version_a
        )
        for idx, g in enumerate(active_games):
            game_histories[g].append(a_moves[idx])
        worker.apply_moves_batch(active_games, a_moves)

        active_games = [g for g in range(num_games) if worker.get_game_status(g) == 0]
        if not active_games:
            break

        active_histories = [game_histories[g] for g in active_games]
        b_moves = worker.choose_alphabeta_moves(
            active_histories, time_limit_ms=bot_ms, bot_version=version_b
        )
        for idx, g in enumerate(active_games):
            game_histories[g].append(b_moves[idx])
        worker.apply_moves_batch(active_games, b_moves)

        finished_count = sum(1 for g in range(num_games) if worker.get_game_status(g) != 0)
        ply += 1

    elapsed = max(time.time() - t_start, 0.001)

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


def main():
    print("================================================================================")
    print("   Benchmarking UTTTTransformer and AlphaZeroResNet against umcts_v2            ")
    print("================================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    transformer_path = os.path.join(base_dir, "ml", "champion.pt")
    resnet_path = os.path.join(base_dir, "ml", "champion_gen7_resnet.pt")

    transformer = UTTTTransformer().to(device)
    t_ckpt = torch.load(transformer_path, map_location=device, weights_only=False)
    if isinstance(t_ckpt, dict) and "model_state_dict" in t_ckpt:
        transformer.load_state_dict(t_ckpt["model_state_dict"])
    elif isinstance(t_ckpt, dict):
        transformer.load_state_dict(t_ckpt)
    else:
        transformer = t_ckpt.to(device)
    transformer.eval()
    print(f"  ✓ UTTTTransformer loaded from {transformer_path}")

    resnet = AlphaZeroResNet().to(device)
    r_ckpt = torch.load(resnet_path, map_location=device, weights_only=False)
    if isinstance(r_ckpt, dict) and "model_state_dict" in r_ckpt:
        resnet.load_state_dict(r_ckpt["model_state_dict"])
    elif isinstance(r_ckpt, dict):
        resnet.load_state_dict(r_ckpt)
    else:
        resnet = r_ckpt.to(device)
    resnet.eval()
    print(f"  ✓ AlphaZeroResNet loaded from {resnet_path}")

    print(f"\n[1/4] Mining 256 Balanced Tactical Positions...")
    paired_positions = mine_interesting_positions(
        challenger=transformer,
        champion=resnet,
        num_positions=256,
        device=device,
        seed=42,
    )
    print(f"  ✓ Selected 256 positions (512 mirrored games per pairing)")

    # Load 4-way tournament results
    matches: List[MatchResult] = []
    results_4way_path = os.path.join(base_dir, "ml", "tournament_4way_results.json")
    if os.path.exists(results_4way_path):
        with open(results_4way_path, "r") as f:
            data = json.load(f)
        for m in data.get("matches", []):
            matches.append(MatchResult(**m))
        print(f"  ✓ Reused {len(matches)} previously computed matches from tournament_4way_results.json")

    # Match: UTTTTransformer vs umcts_v2
    print(f"\n--- Match: UTTTTransformer vs umcts_v2 (512 games) ---")
    m_trans_umcts = run_neural_vs_bot_match(
        transformer, "UTTTTransformer", "umcts_v2", "umcts_v2", paired_positions, num_sims=100, bot_ms=10, device=device
    )
    matches.append(m_trans_umcts)
    print(f"    Result: {m_trans_umcts.a_wins}W - {m_trans_umcts.b_wins}L - {m_trans_umcts.draws}D (Win Rate: {m_trans_umcts.a_win_rate*100:.2f}% ± {m_trans_umcts.margin_of_error*100:.2f}%)")
    print(f"    Double Wins: Transformer {m_trans_umcts.a_double_wins} | umcts_v2 {m_trans_umcts.b_double_wins} | Splits: {m_trans_umcts.symmetrical_splits}")
    print(f"    Delta Elo: {m_trans_umcts.delta_elo:+.1f} | Elapsed: {m_trans_umcts.elapsed_seconds:.1f}s")

    # Match: AlphaZeroResNet vs umcts_v2
    print(f"\n--- Match: AlphaZeroResNet vs umcts_v2 (512 games) ---")
    m_res_umcts = run_neural_vs_bot_match(
        resnet, "AlphaZeroResNet", "umcts_v2", "umcts_v2", paired_positions, num_sims=100, bot_ms=10, device=device
    )
    matches.append(m_res_umcts)
    print(f"    Result: {m_res_umcts.a_wins}W - {m_res_umcts.b_wins}L - {m_res_umcts.draws}D (Win Rate: {m_res_umcts.a_win_rate*100:.2f}% ± {m_res_umcts.margin_of_error*100:.2f}%)")
    print(f"    Double Wins: ResNet {m_res_umcts.a_double_wins} | umcts_v2 {m_res_umcts.b_double_wins} | Splits: {m_res_umcts.symmetrical_splits}")
    print(f"    Delta Elo: {m_res_umcts.delta_elo:+.1f} | Elapsed: {m_res_umcts.elapsed_seconds:.1f}s")

    # Match: AlphaBeta (V1) vs umcts_v2
    print(f"\n--- Match: AlphaBeta (V1) vs umcts_v2 (512 games) ---")
    m_ab_umcts = run_bot_vs_bot_match(
        "AlphaBeta", "v1", "umcts_v2", "umcts_v2", paired_positions, bot_ms=10
    )
    matches.append(m_ab_umcts)
    print(f"    Result: {m_ab_umcts.a_wins}W - {m_ab_umcts.b_wins}L - {m_ab_umcts.draws}D (Win Rate: {m_ab_umcts.a_win_rate*100:.2f}% ± {m_ab_umcts.margin_of_error*100:.2f}%)")
    print(f"    Double Wins: AlphaBeta (V1) {m_ab_umcts.a_double_wins} | umcts_v2 {m_ab_umcts.b_double_wins} | Splits: {m_ab_umcts.symmetrical_splits}")
    print(f"    Delta Elo: {m_ab_umcts.delta_elo:+.1f} | Elapsed: {m_ab_umcts.elapsed_seconds:.1f}s")

    # Joint 5-Way Bradley-Terry Elo estimation
    print(f"\n[4/4] Computing Joint Bradley-Terry Maximum Likelihood Elo Ratings across all 5 engines...")
    competitors = ["AlphaBeta", "AlphaBetaV2", "UTTTTransformer", "AlphaZeroResNet", "umcts_v2"]
    elo_ratings = solve_bradley_terry(
        competitors=competitors,
        match_results=matches,
        anchor_name="AlphaZeroResNet",
        anchor_elo=1906.4,
    )

    print(f"\n================================================================================")
    print(f"                         5-WAY GRAND TOURNAMENT LEADERBOARD                     ")
    print(f"================================================================================")
    print(f"{'Rank':<5} | {'Competitor':<18} | {'Elo Rating':<11} | {'Type / Description':<35}")
    print(f"------+--------------------+-------------+------------------------------------")
    sorted_ranks = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    descriptions = {
        "AlphaBeta": "AlphaBeta V1 (Original Leaderboard 1713)",
        "AlphaBetaV2": "Minimax V2 (TT + History + DF-PN)",
        "UTTTTransformer": "Gen-8 Champion (Vision Transformer)",
        "AlphaZeroResNet": "Gen-7 Champion (Residual CNN)",
        "umcts_v2": "UMCTS-V2 (Original Leaderboard 1601)",
    }
    for rank, (name, elo) in enumerate(sorted_ranks, start=1):
        print(f"#{rank:<4} | {name:<18} | {elo:>9.1f}   | {descriptions[name]:<35}")

    print(f"\nDetailed Head-to-Head Cross-Table:")
    print(f"{'Match':<35} | {'Score':<14} | {'Win Rate':<14} | {'2-0':<5} | {'0-2':<5} | {'1-1':<5}")
    print(f"------------------------------------+----------------+----------------+-------+-------+-------")
    for m in matches:
        match_label = f"{m.competitor_a} vs {m.competitor_b}"
        score_str = f"{m.a_wins}W-{m.b_wins}L-{m.draws}D"
        wr_str = f"{m.a_win_rate*100:5.1f}% ±{m.margin_of_error*100:4.1f}%"
        print(f"{match_label:<35} | {score_str:<14} | {wr_str:<14} | {m.a_double_wins:<5} | {m.b_double_wins:<5} | {m.symmetrical_splits:<5}")

    output_path = os.path.join(base_dir, "ml", "tournament_5way_results.json")
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
