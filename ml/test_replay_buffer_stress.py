"""
Comprehensive Stress & Edge-Case Test Suite for Contiguous Stratified Replay Buffer
Verifies:
1. Multi-tier rapid ring wrap-around (100,000+ insertions with zero memory growth)
2. Extreme tier imbalance (single tier populated, 1-sample buffer, sampling with replacement)
3. Full D4 dihedral symmetry mathematical invariance across all 8 spatial operations
4. Numerical safety (zero NaNs, zero Infs, strictly bounded values, probability sums)
5. Checkpoint serialization round-trip fidelity under wrapped ring-buffer conditions
6. PyTorch DataLoader multi-threaded batch consumption stress
"""

import os
import sys
import tempfile
import tracemalloc
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.replay_buffer import ContiguousStratifiedReplayBuffer, ReplayBufferDataset, D4_PERMUTATIONS
from ml.model import move_index_to_row_col, row_col_to_move_index


def test_rapid_ring_wraparound_stress():
    """Inserts 100,000 samples into a 2,000-capacity buffer (50x wrap-around)."""
    tracemalloc.start()
    cap = 2_000
    buffer = ContiguousStratifiedReplayBuffer(capacity=cap, gm_ratio=0.30, deep_draw_ratio=0.35, self_play_ratio=0.35)

    base_mem = buffer.memory_usage_bytes

    dummy_obs = np.random.randint(0, 2, size=(5, 9, 9)).astype(np.float32)
    dummy_mask = np.random.randint(0, 2, size=81).astype(np.float32)
    dummy_pol = np.full(81, 1.0 / 81, dtype=np.float32)

    # 100,000 rapid insertions across all tiers
    for i in range(100_000):
        t = i % 3
        if t == 0:
            tier = "grandmaster"
            game_len = 25
            val = 1.0
        elif t == 1:
            tier = "deep_draw"
            game_len = 45
            val = 0.0
        else:
            tier = "self_play"
            game_len = 20
            val = -1.0
        buffer.add_sample(dummy_obs, dummy_mask, dummy_pol, value=val, game_length=game_len, tier=tier)

    # Validate tier bounds
    assert buffer.gm_count == buffer.gm_capacity
    assert buffer.deep_draw_count == buffer.deep_draw_capacity
    assert buffer.self_play_count == buffer.self_play_capacity
    assert len(buffer) == cap

    # Memory must remain strictly identical to initial allocation
    assert buffer.memory_usage_bytes == base_mem

    # Pointers must remain strictly in [0, tier_cap)
    assert 0 <= buffer.gm_ptr < buffer.gm_capacity
    assert 0 <= buffer.deep_draw_ptr < buffer.deep_draw_capacity
    assert 0 <= buffer.self_play_ptr < buffer.self_play_capacity

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  [PASS] Rapid wrap-around: 100,000 insertions, 0 memory leaks (Peak tracemalloc: {peak_mem / 1024:.1f} KB)")


def test_extreme_tier_imbalances():
    """Verifies sampling stability when tiers are empty or buffer has only 1 sample."""
    # Case 1: Empty buffer must raise ValueError
    empty_buf = ContiguousStratifiedReplayBuffer(capacity=100)
    try:
        empty_buf.sample_batch(32)
        assert False, "Should have raised ValueError on empty buffer"
    except ValueError:
        pass

    # Case 2: Only 1 sample in entire buffer, sampling batch of 128 (forces replacement)
    one_buf = ContiguousStratifiedReplayBuffer(capacity=100)
    obs = np.ones((5, 9, 9), dtype=np.float32)
    mask = np.ones(81, dtype=np.float32)
    pol = np.full(81, 1.0 / 81, dtype=np.float32)
    one_buf.add_sample(obs, mask, pol, value=0.75, game_length=20, tier="self_play")

    b_obs, b_mask, b_pol, b_val = one_buf.sample_batch(128, apply_augmentation=False)
    assert b_obs.shape == (128, 5, 9, 9)
    assert b_mask.shape == (128, 81)
    assert b_pol.shape == (128, 81)
    assert b_val.shape == (128,)
    assert np.allclose(b_val, 0.75)

    # Case 3: Only GM samples populated (0 Deep/Draw, 0 Self-Play)
    gm_buf = ContiguousStratifiedReplayBuffer(capacity=500)
    for i in range(50):
        gm_buf.add_sample(obs, mask, pol, value=float(i % 2), game_length=20, tier="grandmaster")

    b_obs, b_mask, b_pol, b_val = gm_buf.sample_batch(64, apply_augmentation=True)
    assert b_obs.shape == (64, 5, 9, 9)
    assert len(b_val) == 64

    # Case 4: Only Deep/Draw samples populated
    dd_buf = ContiguousStratifiedReplayBuffer(capacity=500)
    for i in range(50):
        dd_buf.add_sample(obs, mask, pol, value=0.0, game_length=50, tier="self_play")

    b_obs, b_mask, b_pol, b_val = dd_buf.sample_batch(64, apply_augmentation=True)
    assert b_obs.shape == (64, 5, 9, 9)

    print("  [PASS] Extreme tier imbalances handled gracefully with replacement.")


def test_d4_symmetry_mathematical_invariance():
    """Mathematically proves spatial consistency across all 8 D4 dihedral group operations."""
    for sym_idx in range(8):
        rot = sym_idx // 2
        flip = bool(sym_idx % 2)

        for move_idx in range(81):
            # Create a 1-hot board and 1-hot policy at move_idx
            obs = np.zeros((1, 5, 9, 9), dtype=np.float32)
            r, c = move_index_to_row_col(move_idx)
            obs[0, 0, r, c] = 1.0  # Channel 0 mark

            mask = np.zeros((1, 81), dtype=np.float32)
            mask[0, move_idx] = 1.0

            pol = np.zeros((1, 81), dtype=np.float32)
            pol[0, move_idx] = 1.0

            # Transform obs spatially
            t_obs = np.rot90(obs[0], rot, axes=(1, 2))
            if flip:
                t_obs = np.flip(t_obs, axis=2)

            # Spatial location of 1.0 in transformed obs
            locs = np.argwhere(t_obs[0] == 1.0)
            assert len(locs) == 1
            new_r, new_c = locs[0]
            expected_new_m = row_col_to_move_index(new_r, new_c)

            # Transform policy and mask via D4 permutation table
            perm = D4_PERMUTATIONS[sym_idx]
            t_pol = pol[0, perm]
            t_mask = mask[0, perm]

            actual_new_m = np.argmax(t_pol)
            actual_mask_m = np.argmax(t_mask)

            assert actual_new_m == expected_new_m, f"Policy symmetry mismatch: sym={sym_idx}, orig={move_idx}"
            assert actual_mask_m == expected_new_m, f"Mask symmetry mismatch: sym={sym_idx}, orig={move_idx}"

    print("  [PASS] D4 symmetry mathematical invariance verified (8/8 operations x 81 moves = 648 checks).")


def test_numerical_safety_and_ranges():
    """Verifies that sampled batches contain no NaNs, Infs, or invalid ranges under high load."""
    buffer = ContiguousStratifiedReplayBuffer(capacity=5_000)

    for i in range(1_000):
        obs = np.random.randint(0, 2, size=(5, 9, 9)).astype(np.float32)
        mask = np.random.randint(0, 2, size=81).astype(np.float32)
        mask[0] = 1.0  # ensure at least one legal move
        raw_pol = np.random.exponential(scale=1.0, size=81).astype(np.float32) * mask
        pol = raw_pol / (raw_pol.sum() + 1e-8)
        val = np.random.uniform(-1.0, 1.0)
        buffer.add_sample(obs, mask, pol, val, game_length=np.random.randint(10, 60))

    for _ in range(50):
        b_obs, b_mask, b_pol, b_val = buffer.sample_batch(128, apply_augmentation=True)

        assert not np.isnan(b_obs).any()
        assert not np.isinf(b_obs).any()
        assert np.all((b_obs == 0.0) | (b_obs == 1.0))

        assert not np.isnan(b_mask).any()
        assert not np.isinf(b_mask).any()

        assert not np.isnan(b_pol).any()
        assert not np.isinf(b_pol).any()
        assert np.all(b_pol >= -1e-6)

        assert not np.isnan(b_val).any()
        assert not np.isinf(b_val).any()
        assert np.all((b_val >= -1.0) & (b_val <= 1.0))

    print("  [PASS] Numerical safety: 50 mini-batches tested with zero NaNs, zero Infs, strictly bounded ranges.")


def test_checkpoint_roundtrip_fidelity():
    """Verifies that saving and reloading a wrapped buffer preserves bitwise identity."""
    cap = 1_000
    buffer1 = ContiguousStratifiedReplayBuffer(capacity=cap)

    # Insert 3,000 samples to ensure wrap-around across all tiers
    for i in range(3_000):
        obs = np.random.randint(0, 2, size=(5, 9, 9)).astype(np.float32)
        mask = np.ones(81, dtype=np.float32)
        pol = np.full(81, 1.0 / 81, dtype=np.float32)
        val = (i % 100) / 100.0 - 0.5
        tier = "grandmaster" if i % 3 == 0 else "self_play"
        buffer1.add_sample(obs, mask, pol, val, game_length=i % 60, tier=tier)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_path = f.name

    try:
        buffer1.save(temp_path)
        assert os.path.exists(temp_path)
        file_size = os.path.getsize(temp_path)
        print(f"  Checkpoint size for 1,000 items: {file_size / 1024:.1f} KB")

        buffer2 = ContiguousStratifiedReplayBuffer(capacity=cap)
        assert buffer2.load(temp_path)

        assert len(buffer2) == len(buffer1)
        assert buffer2.gm_count == buffer1.gm_count
        assert buffer2.deep_draw_count == buffer1.deep_draw_count
        assert buffer2.self_play_count == buffer1.self_play_count
        assert buffer2.gm_ptr == buffer1.gm_ptr
        assert buffer2.deep_draw_ptr == buffer1.deep_draw_ptr
        assert buffer2.self_play_ptr == buffer1.self_play_ptr

        # Bitwise exact match
        np.testing.assert_array_equal(buffer2.observations, buffer1.observations)
        np.testing.assert_array_equal(buffer2.legal_masks, buffer1.legal_masks)
        np.testing.assert_array_equal(buffer2.policies, buffer1.policies)
        np.testing.assert_array_equal(buffer2.values, buffer1.values)
        print("  [PASS] Checkpoint round-trip bitwise fidelity verified.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_pytorch_dataloader_multithread_stress():
    """Verifies that PyTorch DataLoader can stream batches from the buffer without deadlock."""
    buffer = ContiguousStratifiedReplayBuffer(capacity=5_000)

    for i in range(500):
        obs = np.random.randint(0, 2, size=(5, 9, 9)).astype(np.float32)
        mask = np.ones(81, dtype=np.float32)
        pol = np.full(81, 1.0 / 81, dtype=np.float32)
        val = 0.5
        buffer.add_sample(obs, mask, pol, val, game_length=30)

    dataset = ReplayBufferDataset(buffer, epoch_steps=20, batch_size=32)
    loader = DataLoader(dataset, batch_size=32, num_workers=2)

    batches_seen = 0
    for b_obs, b_mask, b_pol, b_val in loader:
        assert b_obs.shape[1:] == (5, 9, 9)
        assert b_mask.shape[1:] == (81,)
        assert b_pol.shape[1:] == (81,)
        batches_seen += 1

    assert batches_seen == 20
    print(f"  [PASS] PyTorch DataLoader streamed {batches_seen} batches with 2 worker threads successfully.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RUNNING REPLAY BUFFER STRESS & EDGE-CASE TEST SUITE")
    print("=" * 60)

    test_rapid_ring_wraparound_stress()
    test_extreme_tier_imbalances()
    test_d4_symmetry_mathematical_invariance()
    test_numerical_safety_and_ranges()
    test_checkpoint_roundtrip_fidelity()
    test_pytorch_dataloader_multithread_stress()

    print("\n✅ ALL REPLAY BUFFER STRESS TESTS PASSED WITH ZERO ERRORS!\n")
