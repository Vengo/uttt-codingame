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
        gm_ratio: float = 0.10,
        deep_draw_ratio: float = 0.45,
        self_play_ratio: float = 0.45,
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
        game_length: Optional[int] = None,
        tier: str = "self_play",
    ):
        """Adds a single sample into the contiguous buffer with ring eviction."""
        obs_u8 = (observation > 0.5).astype(np.uint8) if observation.dtype != np.uint8 else observation
        mask_u8 = (legal_mask > 0.5).astype(np.uint8) if legal_mask.dtype != np.uint8 else legal_mask
        pol_f32 = policy.astype(np.float32) if policy.dtype != np.float32 else policy
        val_f32 = float(value)

        # Compute board ply: use explicit game_length if provided, otherwise derive from marks
        if game_length is not None:
            ply = game_length
        else:
            ply = int(obs_u8[0].sum() + obs_u8[1].sum())
        is_deep_or_draw = (ply >= 38) or (abs(val_f32) < 0.05) or (tier == "deep_draw")

        # Route to appropriate tier strictly within dedicated bounds
        if tier == "grandmaster" and self.gm_capacity > 0:
            slot = self.gm_start + self.gm_ptr
            self.gm_ptr = (self.gm_ptr + 1) % self.gm_capacity
            self.gm_count = min(self.gm_count + 1, self.gm_capacity)
        elif is_deep_or_draw:
            slot = self.deep_draw_start + self.deep_draw_ptr
            self.deep_draw_ptr = (self.deep_draw_ptr + 1) % self.deep_draw_capacity
            self.deep_draw_count = min(self.deep_draw_count + 1, self.deep_draw_capacity)
        else:
            slot = self.self_play_start + self.self_play_ptr
            self.self_play_ptr = (self.self_play_ptr + 1) % self.self_play_capacity
            self.self_play_count = min(self.self_play_count + 1, self.self_play_capacity)

        self.observations[slot] = obs_u8
        self.legal_masks[slot] = mask_u8
        self.policies[slot] = pol_f32
        self.values[slot, 0] = val_f32
        self.total_inserted += 1

    def add_samples_batch(
        self,
        observations: np.ndarray,
        legal_masks: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        tier: str = "self_play",
    ):
        """Adds a batch of samples into the contiguous buffer with ring eviction."""
        n = len(observations)
        for i in range(n):
            self.add_sample(
                observations[i],
                legal_masks[i],
                policies[i],
                float(values[i]),
                tier=tier,
            )

    @staticmethod
    def _sample_ring_recency(
        start: int,
        count: int,
        ptr: int,
        capacity: int,
        n: int,
        recency_window: int = 120_000,
        recent_share: float = 0.70,
    ) -> np.ndarray:
        """
        Samples `n` indices from a ring-buffer slice `[start, start + count)`.
        When `count > recency_window`, `recent_share` (default 70%) of samples are drawn
        from the `recency_window` most recently inserted slots backward from `ptr`,
        and the remaining `1 - recent_share` (30%) are drawn uniformly across all `count` slots.
        """
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if count > recency_window and recency_window > 0:
            n_recent = int(round(n * recent_share))
            n_uniform = n - n_recent
            parts = []
            if n_recent > 0:
                offsets = np.random.randint(0, recency_window, size=n_recent)
                recent_idx = start + ((ptr - 1 - offsets) % count)
                parts.append(recent_idx)
            if n_uniform > 0:
                uniform_idx = np.random.randint(start, start + count, size=n_uniform)
                parts.append(uniform_idx)
            return np.concatenate(parts) if len(parts) > 1 else parts[0]
        else:
            return np.random.randint(start, start + count, size=n)

    def sample_batch(
        self,
        batch_size: int,
        apply_augmentation: bool = True,
        augment_d4_prob: float = 1.0,
        recency_window: int = 120_000,
        recent_share: float = 0.70,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a stratified, recency-weighted batch directly from contiguous memory.
        Returns: (observations_float32, masks_float32, policies_float32, values_float32)
        """
        total = len(self)
        if total == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")

        # Allocate batch proportionally to actual populated counts, capping frozen GM bootstrap tier at <= 2% once buffer >= 200k
        non_gm = self.deep_draw_count + self.self_play_count
        if total >= 200_000 and non_gm > 0:
            p_gm = min(self.gm_count / total, 0.02)
            rem_p = 1.0 - p_gm
            p_deep = rem_p * (self.deep_draw_count / non_gm)
            p_sp = rem_p * (self.self_play_count / non_gm)
        else:
            p_gm = self.gm_count / total
            p_deep = self.deep_draw_count / total
            p_sp = self.self_play_count / total

        target_gm = int(round(batch_size * p_gm))
        target_deep = int(round(batch_size * p_deep))
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
            deep_indices = self._sample_ring_recency(
                self.deep_draw_start,
                self.deep_draw_count,
                self.deep_draw_ptr,
                self.deep_draw_capacity,
                n_deep,
                recency_window=recency_window,
                recent_share=recent_share,
            )
            indices.append(deep_indices)
        if n_sp > 0:
            sp_indices = self._sample_ring_recency(
                self.self_play_start,
                self.self_play_count,
                self.self_play_ptr,
                self.self_play_capacity,
                n_sp,
                recency_window=recency_window,
                recent_share=recent_share,
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
            old_gm_cap = int(meta[1])
            old_deep_cap = int(meta[2])
            old_sp_cap = int(meta[3])

            if (loaded_cap != self.capacity or
                old_gm_cap != self.gm_capacity or
                old_deep_cap != self.deep_draw_capacity or
                old_sp_cap != self.self_play_capacity):
                print(f"Notice: Checkpoint capacities (gm={old_gm_cap:,}, deep={old_deep_cap:,}, sp={old_sp_cap:,}) "
                      f"differ from current (gm={self.gm_capacity:,}, deep={self.deep_draw_capacity:,}, sp={self.self_play_capacity:,}). "
                      f"Re-ingesting and re-aligning valid slices.")
                # Copy matching counts
                copy_gm = min(int(meta[5]), self.gm_capacity)
                copy_deep = min(int(meta[7]), self.deep_draw_capacity)
                copy_sp = min(int(meta[9]), self.self_play_capacity)

                old_gm_start = 0
                old_deep_start = old_gm_cap
                old_sp_start = old_gm_cap + old_deep_cap

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


def test_recency_weighted_sampling():
    """
    Unit test for recency-weighted replay buffer sampling and 2% GM tier cap:
    - Populates a buffer with >= 200,000 total states so the 2% GM cap is active.
    - Populates self_play beyond recency_window, tagging old slots with -1.0 and recent slots with +1.0.
    - Verifies that recent slots are sampled at the expected rate (~70% + 30% * (recency_window / count))
      and GM slots are capped at <= 2%.
    """
    np.random.seed(42)
    buf = ContiguousStratifiedReplayBuffer(
        capacity=300_000,
        gm_ratio=0.10,
        deep_draw_ratio=0.45,
        self_play_ratio=0.45,
    )
    # Directly populate counts and values to test sampling distribution without loop overhead
    buf.gm_count = 30_000
    buf.gm_ptr = 0
    buf.values[buf.gm_start : buf.gm_start + buf.gm_count, 0] = 0.0

    buf.deep_draw_count = 70_000
    buf.deep_draw_ptr = 70_000
    buf.values[buf.deep_draw_start : buf.deep_draw_start + 50_000, 0] = -1.0
    buf.values[buf.deep_draw_start + 50_000 : buf.deep_draw_start + 70_000, 0] = 1.0

    # Test wraparound on self_play ring buffer: count = 120_000, capacity = 135_000
    # Suppose ptr wrapped around to 10_000 with count = 120_000 and recency_window = 20_000:
    # Recent 20,000 slots are [0..10_000) and [110_000..120_000) within the self_play slice.
    buf.self_play_count = 120_000
    buf.self_play_ptr = 10_000
    sp_s = buf.self_play_start
    buf.values[sp_s : sp_s + buf.self_play_count, 0] = -1.0
    buf.values[sp_s : sp_s + 10_000, 0] = 1.0
    buf.values[sp_s + 110_000 : sp_s + 120_000, 0] = 1.0

    _, _, _, vals = buf.sample_batch(
        batch_size=10_000,
        apply_augmentation=False,
        recency_window=20_000,
        recent_share=0.70,
    )

    gm_share = float(np.mean(vals == 0.0))
    recent_non_gm_share = float(np.mean(vals[vals != 0.0] == 1.0))

    assert gm_share <= 0.025, f"Expected GM tier share <= 2.5%, got {gm_share:.4f}"
    # Expected recent share in non-GM tiers: 0.70 + 0.30 * (20k/70k or 20k/120k) ~ 0.75 - 0.78
    assert 0.72 <= recent_non_gm_share <= 0.82, (
        f"Expected recent non-GM share in [0.72, 0.82], got {recent_non_gm_share:.4f}"
    )


if __name__ == "__main__":
    test_recency_weighted_sampling()
    print("test_recency_weighted_sampling PASSED")

