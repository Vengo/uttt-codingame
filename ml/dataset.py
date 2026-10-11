"""
Self-Play Dataset and Game Simulation for Ultimate Tic-Tac-Toe
Provides:
- UTTTEnv: Fast Python environment for UTTT
- Symmetries / D4 data augmentation (rotations & flips)
- Replay file loader
- Self-play data generator (AlphaZero MCTS / fast self-play)
- PyTorch Dataset and DataLoader
"""

import json
import os
import random
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.model import (
    encode_board,
    move_index_to_row_col,
    row_col_to_move_index,
    AlphaZeroResNet,
)

WIN_MASKS = [7, 56, 448, 73, 146, 292, 273, 84]


def check_subboard_win(bb: int) -> bool:
    for m in WIN_MASKS:
        if (bb & m) == m:
            return True
    return False


class UTTTEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.xbb = [0] * 9
        self.obb = [0] * 9
        self.x_supers = 0
        self.o_supers = 0
        self.last_super_index = 9
        self.x_to_move = True
        self.game_over = False
        self.winner = 0  # 1 for X, -1 for O, 0 for tie / unresolved
        return self.get_observation()

    def clone(self) -> 'UTTTEnv':
        e = UTTTEnv.__new__(UTTTEnv)
        e.xbb = list(self.xbb)
        e.obb = list(self.obb)
        e.x_supers = self.x_supers
        e.o_supers = self.o_supers
        e.last_super_index = self.last_super_index
        e.x_to_move = self.x_to_move
        e.game_over = self.game_over
        e.winner = self.winner
        return e

    def get_observation(self) -> Tuple[np.ndarray, np.ndarray]:
        return encode_board(
            self.xbb,
            self.obb,
            self.x_supers,
            self.o_supers,
            self.last_super_index,
            self.x_to_move,
        )

    def legal_moves(self) -> List[int]:
        if self.game_over:
            return []

        won_mask = self.x_supers | self.o_supers
        tied_mask = 0
        for sb in range(9):
            if (won_mask & (1 << sb)) == 0 and ((self.xbb[sb] | self.obb[sb]) == 0x1FF):
                tied_mask |= (1 << sb)
        complete_bbs = won_mask | tied_mask

        if self.last_super_index >= 9 or (complete_bbs & (1 << self.last_super_index)) != 0:
            target_mask = (~complete_bbs) & 0x1FF
        else:
            target_mask = 1 << self.last_super_index

        moves = []
        for sb in range(9):
            if (target_mask & (1 << sb)) != 0:
                occ = self.xbb[sb] | self.obb[sb]
                for sq in range(9):
                    if (occ & (1 << sq)) == 0:
                        moves.append(9 * sb + sq)
        return moves

    def step(self, move_idx: int) -> Tuple[bool, int]:
        """
        Execute move_idx.
        Returns: (game_over, winner)
        """
        assert not self.game_over, "Game is already over"
        sb = move_idx // 9
        sq = move_idx % 9

        if self.x_to_move:
            self.xbb[sb] |= (1 << sq)
            if (self.x_supers & (1 << sb)) == 0 and check_subboard_win(self.xbb[sb]):
                self.x_supers |= (1 << sb)
        else:
            self.obb[sb] |= (1 << sq)
            if (self.o_supers & (1 << sb)) == 0 and check_subboard_win(self.obb[sb]):
                self.o_supers |= (1 << sb)

        # Check macro game win
        if check_subboard_win(self.x_supers):
            self.game_over = True
            self.winner = 1
        elif check_subboard_win(self.o_supers):
            self.game_over = True
            self.winner = -1
        else:
            self.last_super_index = sq
            # Fast bitboard tie check: game can only end if all 9 subboards are completed
            won_mask = self.x_supers | self.o_supers
            tied_mask = 0
            for s in range(9):
                if (won_mask & (1 << s)) == 0 and ((self.xbb[s] | self.obb[s]) == 0x1FF):
                    tied_mask |= (1 << s)
            if (won_mask | tied_mask) == 0x1FF:
                self.game_over = True
                x_won = bin(self.x_supers).count('1')
                o_won = bin(self.o_supers).count('1')
                if x_won > o_won:
                    self.winner = 1
                elif o_won > x_won:
                    self.winner = -1
                else:
                    self.winner = 0

        self.x_to_move = not self.x_to_move
        return self.game_over, self.winner


def augment_d4(
    tensor: np.ndarray,
    legal_mask: np.ndarray,
    policy: np.ndarray,
    value: float,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Generate all 8 D4 dihedral symmetries (4 rotations + reflections)
    for a single training state.
    """
    results = []

    # Map policy and legal_mask to 9x9 grids
    policy_grid = np.zeros((9, 9), dtype=np.float32)
    legal_grid = np.zeros((9, 9), dtype=np.float32)
    for m in range(81):
        r, c = move_index_to_row_col(m)
        policy_grid[r, c] = policy[m]
        legal_grid[r, c] = legal_mask[m]

    for rot in range(4):
        for flip in [False, True]:
            t = np.rot90(tensor, rot, axes=(1, 2))
            pg = np.rot90(policy_grid, rot)
            lg = np.rot90(legal_grid, rot)

            if flip:
                t = np.flip(t, axis=2)
                pg = np.flip(pg, axis=1)
                lg = np.flip(lg, axis=1)

            t = np.ascontiguousarray(t)
            new_policy = np.zeros(81, dtype=np.float32)
            new_legal = np.zeros(81, dtype=np.float32)
            for m in range(81):
                r, c = move_index_to_row_col(m)
                new_policy[m] = pg[r, c]
                new_legal[m] = lg[r, c]

            results.append((t, new_legal, new_policy, value))

    return results


def load_replays_from_files(file_paths: List[str]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Load game replays from JSON files and extract augmented training samples.
    """
    samples = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        try:
            with open(fp, "r") as f:
                raw_data = json.load(f)
        except Exception:
            continue

        if isinstance(raw_data, list):
            games_list = raw_data
        elif isinstance(raw_data, dict):
            games_list = [raw_data]
        else:
            games_list = []

        for data in games_list:
            moves = data.get("moves", [])
            if not moves:
                continue

            result_str = data.get("result", "")
            player_x = data.get("player_x", "")
            player_o = data.get("player_o", "")
            if "Winner: " in result_str:
                w_name = result_str.split("Winner: ")[1].strip()
                if w_name == player_x:
                    final_winner = 1
                elif w_name == player_o:
                    final_winner = -1
                else:
                    final_winner = 0
            elif "X Win" in result_str:
                final_winner = 1
            elif "O Win" in result_str:
                final_winner = -1
            else:
                final_winner = 0

            env = UTTTEnv()
            game_states = []
            for m in moves:
                move_idx = m["move_index"]
                tensor, legal_mask = env.get_observation()
                current_x_to_move = env.x_to_move

                policy = np.zeros(81, dtype=np.float32)
                policy[move_idx] = 1.0

                game_states.append((tensor, legal_mask, policy, current_x_to_move))
                game_over, _ = env.step(move_idx)
                if game_over:
                    break

            # Calculate values from active player's perspective
            for tensor, legal_mask, policy, x_to_move in game_states:
                val = float(final_winner) if x_to_move else float(-final_winner)
                augmented = augment_d4(tensor, legal_mask, policy, val)
                samples.extend(augmented)

    return samples


def run_fast_self_play(
    model: Optional[AlphaZeroResNet] = None,
    device: str = "cpu",
    temperature: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Generate a self-play game using the model policy (or random if model is None).
    """
    env = UTTTEnv()
    game_history = []  # (tensor, legal_mask, policy, x_to_move)

    if model is not None:
        model.eval()

    while not env.game_over:
        tensor, legal_mask = env.get_observation()
        legal_indices = [i for i in range(81) if legal_mask[i] > 0]
        if not legal_indices:
            break

        if model is None:
            # Uniform policy among legal moves
            policy = np.zeros(81, dtype=np.float32)
            for m in legal_indices:
                policy[m] = 1.0 / len(legal_indices)
            chosen_move = random.choice(legal_indices)
        else:
            with torch.no_grad():
                x = torch.from_numpy(tensor).unsqueeze(0).to(device)
                lm = torch.from_numpy(legal_mask).unsqueeze(0).to(device)
                logits, _ = model(x, lm)
                probs = torch.softmax(logits / max(temperature, 0.1), dim=-1)[0].cpu().numpy()

            # Ensure only legal moves have probability
            legal_probs = np.zeros(81, dtype=np.float32)
            legal_probs[legal_indices] = probs[legal_indices]
            s = legal_probs.sum()
            if s > 0:
                legal_probs /= s
            else:
                legal_probs[legal_indices] = 1.0 / len(legal_indices)

            policy = legal_probs
            chosen_move = np.random.choice(81, p=policy)

        game_history.append((tensor, legal_mask, policy, env.x_to_move))
        env.step(chosen_move)

    final_winner = env.winner
    samples = []
    for tensor, legal_mask, policy, x_to_move in game_history:
        val = float(final_winner) if x_to_move else float(-final_winner)
        samples.extend(augment_d4(tensor, legal_mask, policy, val))

    return samples


class UTTTDataset(Dataset):
    def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor, legal_mask, policy, value = self.samples[idx]
        return (
            torch.from_numpy(tensor),
            torch.from_numpy(legal_mask),
            torch.from_numpy(policy),
            torch.tensor(value, dtype=torch.float32).unsqueeze(0),
        )
