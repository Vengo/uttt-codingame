"""
Intelligent, High-Capacity Replay Buffer for Ultimate Tic-Tac-Toe
Capacity: 500,000 - 1,000,000 states (~150-300 MB RAM)
Features:
- Elo / Quality tier stratification (Grandmaster, Deep MCTS, AlphaZero self-play)
- Outcome & length stratification (preserves tactical draws and deep 40+ ply battles)
- Reservoir sampling for historical stability (prevents catastrophic forgetting)
- Memory-efficient uint8 / bit-packed observation storage
- On-the-fly D4 dihedral group data augmentation (8x multiplier during sampling)
- Fast disk checkpointing and incremental loading
"""

import os
import random
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.model import (
    move_index_to_row_col,
    row_col_to_move_index,
)
from ml.dataset import augment_d4


class ReplaySample:
    __slots__ = ("observation", "legal_mask", "policy", "value", "game_length", "tier")

    def __init__(
        self,
        observation: np.ndarray,  # shape (5, 9, 9) float32 or uint8
        legal_mask: np.ndarray,   # shape (81,) float32 or bool
        policy: np.ndarray,       # shape (81,) float32
        value: float,             # float in [-1.0, 1.0]
        game_length: int = 30,
        tier: str = "self_play",  # "grandmaster", "deep_mcts", "self_play"
    ):
        # Pack observation as uint8 to save 4x memory (5 * 81 = 405 bytes)
        self.observation = (observation > 0.5).astype(np.uint8)
        self.legal_mask = (legal_mask > 0.5).astype(np.uint8)
        self.policy = policy.astype(np.float32)
        self.value = float(value)
        self.game_length = int(game_length)
        self.tier = str(tier)

    def unpack(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            self.observation.astype(np.float32),
            self.legal_mask.astype(np.float32),
            self.policy,
            self.value,
        )


class StratifiedReplayBuffer:
    def __init__(
        self,
        capacity: int = 600_000,
        gm_ratio: float = 0.30,       # 30% Grandmaster / Expert replays
        deep_draw_ratio: float = 0.35, # 35% Long games (>= 38 plies) & hard draws
        self_play_ratio: float = 0.35, # 35% Active self-play generations
    ):
        self.capacity = capacity
        self.gm_capacity = int(capacity * gm_ratio)
        self.deep_draw_capacity = int(capacity * deep_draw_ratio)
        self.self_play_capacity = int(capacity * self_play_ratio)

        self.gm_buffer: List[ReplaySample] = []
        self.deep_draw_buffer: List[ReplaySample] = []
        self.self_play_buffer: List[ReplaySample] = []
        self.total_inserted = 0

    def __len__(self) -> int:
        return len(self.gm_buffer) + len(self.deep_draw_buffer) + len(self.self_play_buffer)

    def add_sample(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        policy: np.ndarray,
        value: float,
        game_length: int = 30,
        tier: str = "self_play",
    ):
        sample = ReplaySample(observation, legal_mask, policy, value, game_length, tier)
        self.total_inserted += 1

        is_deep_or_draw = (game_length >= 38) or (abs(value) < 0.05)

        if tier == "grandmaster":
            self._insert_with_eviction(self.gm_buffer, sample, self.gm_capacity)
        elif is_deep_or_draw:
            self._insert_with_eviction(self.deep_draw_buffer, sample, self.deep_draw_capacity)
        else:
            self._insert_with_eviction(self.self_play_buffer, sample, self.self_play_capacity)

    def _insert_with_eviction(self, target_list: List[ReplaySample], sample: ReplaySample, max_len: int):
        if len(target_list) < max_len:
            target_list.append(sample)
        else:
            # Reservoir replacement to ensure long-term uniform coverage across generations
            idx = random.randint(0, len(target_list) - 1)
            target_list[idx] = sample

    def add_game(
        self,
        game_history: List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]],
        final_winner: int,
        tier: str = "self_play",
    ):
        """
        Ingest a full game history: [(obs, mask, policy, x_to_move), ...]
        final_winner: +1 (X won), -1 (O won), 0 (tie)
        """
        game_len = len(game_history)
        for obs, mask, policy, x_to_move in game_history:
            val = float(final_winner) if x_to_move else float(-final_winner)
            self.add_sample(obs, mask, policy, val, game_length=game_len, tier=tier)

    def sample_batch(
        self,
        batch_size: int = 256,
        augment_d4_prob: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a balanced batch according to tier ratios with on-the-fly D4 dihedral augmentation.
        Returns PyTorch tensors on CPU: (tensors, masks, policies, values)
        """
        total = len(self)
        if total == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        # Determine counts from each sub-buffer
        available_pools = []
        weights = []
        if self.gm_buffer:
            available_pools.append(self.gm_buffer)
            weights.append(0.30)
        if self.deep_draw_buffer:
            available_pools.append(self.deep_draw_buffer)
            weights.append(0.35)
        if self.self_play_buffer:
            available_pools.append(self.self_play_buffer)
            weights.append(0.35)

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        selected_samples: List[ReplaySample] = []
        for _ in range(batch_size):
            pool = random.choices(available_pools, weights=norm_weights, k=1)[0]
            selected_samples.append(random.choice(pool))

        batch_tensors = []
        batch_masks = []
        batch_policies = []
        batch_values = []

        for sample in selected_samples:
            obs, mask, pol, val = sample.unpack()

            if augment_d4_prob > 0.0 and random.random() < augment_d4_prob:
                # Random D4 transform (4 rotations x 2 flips)
                rot = random.randint(0, 3)
                flip = random.random() < 0.5

                policy_grid = np.zeros((9, 9), dtype=np.float32)
                legal_grid = np.zeros((9, 9), dtype=np.float32)
                for m in range(81):
                    r, c = move_index_to_row_col(m)
                    policy_grid[r, c] = pol[m]
                    legal_grid[r, c] = mask[m]

                t = np.rot90(obs, rot, axes=(1, 2))
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

                batch_tensors.append(t)
                batch_masks.append(new_legal)
                batch_policies.append(new_policy)
                batch_values.append(val)
            else:
                batch_tensors.append(obs)
                batch_masks.append(mask)
                batch_policies.append(pol)
                batch_values.append(val)

        x = torch.from_numpy(np.array(batch_tensors, dtype=np.float32))
        m = torch.from_numpy(np.array(batch_masks, dtype=np.float32))
        p = torch.from_numpy(np.array(batch_policies, dtype=np.float32))
        v = torch.from_numpy(np.array(batch_values, dtype=np.float32)).unsqueeze(1)

        return x, m, p, v

    def save(self, filepath: str):
        """Save buffer states to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        all_samples = self.gm_buffer + self.deep_draw_buffer + self.self_play_buffer
        if not all_samples:
            return

        obs_arr = np.array([s.observation for s in all_samples], dtype=np.uint8)
        mask_arr = np.array([s.legal_mask for s in all_samples], dtype=np.uint8)
        pol_arr = np.array([s.policy for s in all_samples], dtype=np.float32)
        val_arr = np.array([s.value for s in all_samples], dtype=np.float32)
        len_arr = np.array([s.game_length for s in all_samples], dtype=np.int16)
        tier_arr = np.array([s.tier for s in all_samples])

        np.savez_compressed(
            filepath,
            obs=obs_arr,
            mask=mask_arr,
            pol=pol_arr,
            val=val_arr,
            lengths=len_arr,
            tiers=tier_arr,
        )
        print(f"📦 Successfully saved {len(all_samples)} states to replay buffer: {filepath}")

    def load(self, filepath: str) -> int:
        """Load buffer states from disk."""
        if not os.path.exists(filepath):
            return 0
        data = np.load(filepath, allow_pickle=True)
        obs_arr = data["obs"]
        mask_arr = data["mask"]
        pol_arr = data["pol"]
        val_arr = data["val"]
        len_arr = data["lengths"]
        tier_arr = data["tiers"]

        count = len(obs_arr)
        for i in range(count):
            self.add_sample(
                obs_arr[i],
                mask_arr[i],
                pol_arr[i],
                float(val_arr[i]),
                int(len_arr[i]),
                str(tier_arr[i]),
            )
        print(f"📥 Successfully loaded {count} states from {filepath}. Buffer size now: {len(self)}")
        return count
