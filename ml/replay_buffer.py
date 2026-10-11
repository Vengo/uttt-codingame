"""
Contiguous, Zero-GC Stratified Replay Buffer for AlphaZero UTTT
Capacity: 1,000,000 states (~817 MB total RAM)

Features:
- Flat, contiguous pre-allocated NumPy storage (Zero Python object allocations per sample)
- Ring-buffer FIFO eviction per tier with lock-free slice access
- Stratified sampling:
  * 30% Grandmaster / Expert replays
  * 35% Deep tactical battles (>= 38 plies) & hard draws
  * 35% Active self-play generations
- On-the-fly D4 dihedral symmetry augmentation during sampling (8x effective data expansion)
- Fast memory-mapped disk checkpointing (.npz)
"""

import os
import random
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ml.model import move_index_to_row_col


def _build_d4_permutations() -> np.ndarray:
    """Precompute flat index permutation tables for all 8 D4 dihedral symmetries."""
    perms = np.zeros((8, 81), dtype=np.int64)
    idx = 0
    for rot in range(4):
        for flip in [False, True]:
            grid = np.zeros((9, 9), dtype=np.int64)
            for m in range(81):
                r, c = move_index_to_row_col(m)
                grid[r, c] = m
            t_grid = np.rot90(grid, rot)
            if flip:
                t_grid = np.flip(t_grid, axis=1)
            for m in range(81):
                r, c = move_index_to_row_col(m)
                perms[idx, m] = t_grid[r, c]
            idx += 1
    return perms


D4_PERMUTATIONS = _build_d4_permutations()


class ContiguousStratifiedReplayBuffer:
    def __init__(
        self,
        capacity: int = 1_000_000,
        gm_ratio: float = 0.30,
        deep_draw_ratio: float = 0.35,
        self_play_ratio: float = 0.35,
    ):
        self.capacity = int(capacity)
        self.gm_capacity = int(capacity * gm_ratio)
        self.deep_draw_capacity = int(capacity * deep_draw_ratio)
        self.self_play_capacity = self.capacity - self.gm_capacity - self.deep_draw_capacity

        # Offsets for each tier slice in the flat array
        self.gm_start = 0
        self.deep_draw_start = self.gm_capacity
        self.self_play_start = self.gm_capacity + self.deep_draw_capacity

        # Pre-allocate contiguous flat arrays
        # Total footprint for 1M: 405 MB + 81 MB + 324 MB + 4 MB = 814 MB RAM
        self.observations = np.zeros((self.capacity, 5, 9, 9), dtype=np.uint8)
        self.legal_masks = np.zeros((self.capacity, 81), dtype=np.uint8)
        self.policies = np.zeros((self.capacity, 81), dtype=np.float32)
        self.values = np.zeros((self.capacity, 1), dtype=np.float32)

        # Ring buffer pointers and item counts
        self.gm_ptr = 0
        self.gm_count = 0

        self.deep_draw_ptr = 0
        self.deep_draw_count = 0

        self.self_play_ptr = 0
        self.self_play_count = 0

        self.total_inserted = 0

    def __len__(self) -> int:
        return self.gm_count + self.deep_draw_count + self.self_play_count

    @property
    def memory_usage_bytes(self) -> int:
        return (
            self.observations.nbytes
            + self.legal_masks.nbytes
            + self.policies.nbytes
            + self.values.nbytes
        )

    def add_sample(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        policy: np.ndarray,
        value: float,
        game_length: int = 30,
        tier: str = "self_play",
    ):
        """Adds a single sample into the contiguous buffer with ring eviction."""
        obs_u8 = (observation > 0.5).astype(np.uint8) if observation.dtype != np.uint8 else observation
        mask_u8 = (legal_mask > 0.5).astype(np.uint8) if legal_mask.dtype != np.uint8 else legal_mask
        pol_f32 = policy.astype(np.float32) if policy.dtype != np.float32 else policy
        val_f32 = float(value)

        is_deep_or_draw = (game_length >= 38) or (abs(val_f32) < 0.05) or (tier in ("deep_draw", "deep_mcts"))

        # Route to appropriate tier, falling back to any tier with available capacity
        if tier == "grandmaster" and self.gm_capacity > 0:
            slot = self.gm_start + self.gm_ptr
            self.gm_ptr = (self.gm_ptr + 1) % self.gm_capacity
            self.gm_count = min(self.gm_count + 1, self.gm_capacity)
        elif is_deep_or_draw and self.deep_draw_capacity > 0:
            slot = self.deep_draw_start + self.deep_draw_ptr
            self.deep_draw_ptr = (self.deep_draw_ptr + 1) % self.deep_draw_capacity
            self.deep_draw_count = min(self.deep_draw_count + 1, self.deep_draw_capacity)
        elif self.self_play_capacity > 0:
            slot = self.self_play_start + self.self_play_ptr
            self.self_play_ptr = (self.self_play_ptr + 1) % self.self_play_capacity
            self.self_play_count = min(self.self_play_count + 1, self.self_play_capacity)
        elif self.deep_draw_capacity > 0:
            slot = self.deep_draw_start + self.deep_draw_ptr
            self.deep_draw_ptr = (self.deep_draw_ptr + 1) % self.deep_draw_capacity
            self.deep_draw_count = min(self.deep_draw_count + 1, self.deep_draw_capacity)
        elif self.gm_capacity > 0:
            slot = self.gm_start + self.gm_ptr
            self.gm_ptr = (self.gm_ptr + 1) % self.gm_capacity
            self.gm_count = min(self.gm_count + 1, self.gm_capacity)
        else:
            raise RuntimeError("ReplayBuffer has zero capacity across all tiers!")

        self.observations[slot] = obs_u8
        self.legal_masks[slot] = mask_u8
        self.policies[slot] = pol_f32
        self.values[slot, 0] = val_f32
        self.total_inserted += 1

    def sample_batch(
        self,
        batch_size: int,
        apply_augmentation: bool = True,
        augment_d4_prob: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a stratified batch directly from contiguous memory.
        Returns: (observations_float32, masks_float32, policies_float32, values_float32)
        """
        total = len(self)
        if total == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")

        # Determine allocation per tier based on available count
        # Target proportions: 30% GM, 35% Deep/Draw, 35% Self-Play
        p_gm = self.gm_count / total
        p_deep = self.deep_draw_count / total
        p_sp = self.self_play_count / total

        # If a tier has items, give it a proportionate slice
        target_gm = int(batch_size * 0.30) if self.gm_count > 0 else 0
        target_deep = int(batch_size * 0.35) if self.deep_draw_count > 0 else 0
        target_sp = batch_size - target_gm - target_deep

        # Clamp to available counts
        n_gm = min(target_gm, self.gm_count)
        n_deep = min(target_deep, self.deep_draw_count)
        n_sp = min(target_sp, self.self_play_count)

        # Distribute any remainder
        remainder = batch_size - (n_gm + n_deep + n_sp)
        if remainder > 0 and self.self_play_count > n_sp:
            add_sp = min(remainder, self.self_play_count - n_sp)
            n_sp += add_sp
            remainder -= add_sp
        if remainder > 0 and self.deep_draw_count > n_deep:
            add_deep = min(remainder, self.deep_draw_count - n_deep)
            n_deep += add_deep
            remainder -= add_deep
        if remainder > 0 and self.gm_count > n_gm:
            add_gm = min(remainder, self.gm_count - n_gm)
            n_gm += add_gm

        indices = []
        if n_gm > 0:
            gm_indices = np.random.randint(self.gm_start, self.gm_start + self.gm_count, size=n_gm)
            indices.append(gm_indices)
        if n_deep > 0:
            deep_indices = np.random.randint(
                self.deep_draw_start, self.deep_draw_start + self.deep_draw_count, size=n_deep
            )
            indices.append(deep_indices)
        if n_sp > 0:
            sp_indices = np.random.randint(
                self.self_play_start, self.self_play_start + self.self_play_count, size=n_sp
            )
            indices.append(sp_indices)

        if not indices:
            # Fallback to random choice from all valid slots
            all_valid = []
            if self.gm_count > 0:
                all_valid.extend(range(self.gm_start, self.gm_start + self.gm_count))
            if self.deep_draw_count > 0:
                all_valid.extend(range(self.deep_draw_start, self.deep_draw_start + self.deep_draw_count))
            if self.self_play_count > 0:
                all_valid.extend(range(self.self_play_start, self.self_play_start + self.self_play_count))
            idx_array = np.random.choice(all_valid, size=batch_size, replace=True)
        else:
            idx_array = np.concatenate(indices)
            if len(idx_array) < batch_size:
                extra = np.random.choice(idx_array, size=batch_size - len(idx_array), replace=True)
                idx_array = np.concatenate([idx_array, extra])

        # Slice contiguous memory
        b_obs = self.observations[idx_array].astype(np.float32)
        b_mask = self.legal_masks[idx_array].astype(np.float32)
        b_pol = self.policies[idx_array].copy()
        b_val = self.values[idx_array].squeeze(1).copy()

        if apply_augmentation and augment_d4_prob > 0.0:
            # Apply D4 symmetry dihedral transformations on-the-fly using precomputed permutations
            for i in range(batch_size):
                if augment_d4_prob < 1.0 and random.random() >= augment_d4_prob:
                    continue
                rot = random.randint(0, 3)
                flip = random.choice([True, False])
                if rot != 0 or flip:
                    sym_idx = rot * 2 + (1 if flip else 0)
                    perm = D4_PERMUTATIONS[sym_idx]
                    t = np.rot90(b_obs[i], rot, axes=(1, 2))
                    if flip:
                        t = np.flip(t, axis=2)
                    b_obs[i] = np.ascontiguousarray(t)
                    b_mask[i] = b_mask[i, perm]
                    b_pol[i] = b_pol[i, perm]

        return b_obs, b_mask, b_pol, b_val

    def save(self, filepath: str):
        """Atomically saves the buffer to compressed .npz archive."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        temp_path = filepath + ".tmp.npz"

        np.savez_compressed(
            temp_path,
            observations=self.observations,
            legal_masks=self.legal_masks,
            policies=self.policies,
            values=self.values,
            meta=np.array([
                self.capacity,
                self.gm_capacity,
                self.deep_draw_capacity,
                self.self_play_capacity,
                self.gm_ptr,
                self.gm_count,
                self.deep_draw_ptr,
                self.deep_draw_count,
                self.self_play_ptr,
                self.self_play_count,
                self.total_inserted,
            ], dtype=np.int64),
        )
        os.replace(temp_path, filepath)

    def load(self, filepath: str) -> bool:
        """Loads buffer from disk checkpoint if available."""
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath, mmap_mode="r")
            meta = data["meta"]
            loaded_cap = int(meta[0])

            if loaded_cap != self.capacity:
                print(f"Notice: Checkpoint capacity ({loaded_cap:,}) differs from current ({self.capacity:,}). Re-ingesting valid items.")
                # Copy matching counts
                copy_gm = min(int(meta[5]), self.gm_capacity)
                copy_deep = min(int(meta[7]), self.deep_draw_capacity)
                copy_sp = min(int(meta[9]), self.self_play_capacity)

                old_gm_start = 0
                old_deep_start = int(meta[1])
                old_sp_start = int(meta[1]) + int(meta[2])

                if copy_gm > 0:
                    self.observations[:copy_gm] = data["observations"][old_gm_start : old_gm_start + copy_gm]
                    self.legal_masks[:copy_gm] = data["legal_masks"][old_gm_start : old_gm_start + copy_gm]
                    self.policies[:copy_gm] = data["policies"][old_gm_start : old_gm_start + copy_gm]
                    self.values[:copy_gm] = data["values"][old_gm_start : old_gm_start + copy_gm]
                    self.gm_count = copy_gm
                    self.gm_ptr = copy_gm % self.gm_capacity

                if copy_deep > 0:
                    self.observations[self.deep_draw_start : self.deep_draw_start + copy_deep] = data["observations"][old_deep_start : old_deep_start + copy_deep]
                    self.legal_masks[self.deep_draw_start : self.deep_draw_start + copy_deep] = data["legal_masks"][old_deep_start : old_deep_start + copy_deep]
                    self.policies[self.deep_draw_start : self.deep_draw_start + copy_deep] = data["policies"][old_deep_start : old_deep_start + copy_deep]
                    self.values[self.deep_draw_start : self.deep_draw_start + copy_deep] = data["values"][old_deep_start : old_deep_start + copy_deep]
                    self.deep_draw_count = copy_deep
                    self.deep_draw_ptr = copy_deep % self.deep_draw_capacity

                if copy_sp > 0:
                    self.observations[self.self_play_start : self.self_play_start + copy_sp] = data["observations"][old_sp_start : old_sp_start + copy_sp]
                    self.legal_masks[self.self_play_start : self.self_play_start + copy_sp] = data["legal_masks"][old_sp_start : old_sp_start + copy_sp]
                    self.policies[self.self_play_start : self.self_play_start + copy_sp] = data["policies"][old_sp_start : old_sp_start + copy_sp]
                    self.values[self.self_play_start : self.self_play_start + copy_sp] = data["values"][old_sp_start : old_sp_start + copy_sp]
                    self.self_play_count = copy_sp
                    self.self_play_ptr = copy_sp % self.self_play_capacity
            else:
                self.observations[:] = data["observations"]
                self.legal_masks[:] = data["legal_masks"]
                self.policies[:] = data["policies"]
                self.values[:] = data["values"]
                self.gm_ptr = int(meta[4])
                self.gm_count = int(meta[5])
                self.deep_draw_ptr = int(meta[6])
                self.deep_draw_count = int(meta[7])
                self.self_play_ptr = int(meta[8])
                self.self_play_count = int(meta[9])
                self.total_inserted = int(meta[10])

            print(f"Loaded {len(self):,} states from {filepath} (GM: {self.gm_count:,}, Deep/Draw: {self.deep_draw_count:,}, SP: {self.self_play_count:,})")
            return True
        except Exception as e:
            print(f"Warning: Failed to load replay buffer from {filepath}: {e}")
            return False


# Backward-compatible alias
StratifiedReplayBuffer = ContiguousStratifiedReplayBuffer


class ReplayBufferDataset(Dataset):
    def __init__(self, buffer: ContiguousStratifiedReplayBuffer, epoch_steps: int = 1000, batch_size: int = 256):
        self.buffer = buffer
        self.epoch_steps = epoch_steps
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.epoch_steps * self.batch_size

    def __getitem__(self, idx: int):
        # Fallback single-item getter
        b_obs, b_mask, b_pol, b_val = self.buffer.sample_batch(1, apply_augmentation=True)
        return (
            torch.from_numpy(b_obs[0]),
            torch.from_numpy(b_mask[0]),
            torch.from_numpy(b_pol[0]),
            torch.tensor(b_val[0], dtype=torch.float32),
        )
