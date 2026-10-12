# Replay Buffer Technical Challenges & Problem Domain

The [`replay_buffer.py`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py) module addresses several memory efficiency, data distribution, and data integrity problems that arise when training AlphaZero agents at large scales.

---

## 1. Zero-GC Memory Management at Million-State Scale

### Problem
Storing $1,000,000$ game transitions using standard Python lists of dictionaries or tuples requires creating over $4,000,000$ individual heap objects. This results in:
- High memory overhead (> 4–6 GB) due to Python object header bloat.
- Constant Python Garbage Collector (GC) sweeps during self-play, introducing multi-second pauses in the MCTS pipeline.
- Severe virtual memory fragmentation.

### Implementation & Solution
- **Pre-Allocated Contiguous Storage**: Memory is allocated once at startup as four contiguous NumPy arrays.
- **Type Compaction**:
  - `observations` ($5 \times 9 \times 9$) are stored as `np.uint8` instead of `np.float32`, cutting RAM from 1.62 GB to 405 MB.
  - `legal_masks` ($81$) are stored as `np.uint8`, cutting RAM from 324 MB to 81 MB.
- Total memory for 1,000,000 transitions is bounded at exactly $814$ MB.
- Writes and evictions are performed via pointer arithmetic on flat pre-allocated memory without allocating or freeing a single Python object.

---

## 2. Preventing Catastrophic Forgetting & Tier Starvation

### Problem
A standard FIFO replay buffer evicts oldest experiences indiscriminately. When generating thousands of self-play games quickly, high-value expert games (Grandmaster replays) and rare drawn endgames are flushed out within a few cycles. Training on purely recent self-play leads to policy collapse, cyclic strategies, and catastrophic forgetting of complex endgame tactics.

### Implementation & Solution
- **Stratified Partitioning**: The buffer is physically segregated into three dedicated ring buffers:
  1. `grandmaster` ($30\%$): Curated human and strong engine replays.
  2. `deep_draw` ($35\%$): Tactically complex games ($\ge 38$ plies) or hard-fought draws ($|v| < 0.05$).
  3. `self_play` ($35\%$): Active self-play generations.
- **Flexible Overflow Allocation**: To prevent wasting memory when one tier is empty (e.g., prior to grandmaster data ingestion), [`add_sample`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L97-L159) dynamically utilizes spare capacity in other tiers before enforcing ring-buffer FIFO eviction.
- **Dynamic Sampling Proportions**: When sampling batches, proportions dynamically adjust to the *actual populated counts* of each tier rather than nominal capacities. This prevents zero-sample crashes or massive oversampling when a tier has only a few samples.

---

## 3. On-The-Fly D4 Dihedral Symmetry Permutations

### Problem
Ultimate Tic-Tac-Toe boards possess $D_4$ dihedral group symmetry (4 rotations $\times$ 2 reflections = 8 isomorphic states). While spatial input channels can be rotated with standard matrix operations, policy vectors and legal masks are flat 1D arrays of size 81. Mapping a 1D move index under a 2D spatial rotation is non-trivial because subboard index $\lfloor m / 9 \rfloor$ and subsquare index $(m \pmod 9)$ transform simultaneously in 2D space.

### Implementation & Solution
- In [`_build_d4_permutations`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L25-L42), all 8 permutations are precomputed into an `(8, 81)` index table:
  ```python
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
  ```
- During [`sample_batch`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L179-L276), applying symmetry requires only a vectorized 1D index lookup:
  ```python
  b_mask[i] = b_mask[i, perm]
  b_pol[i] = b_pol[i, perm]
  b_obs[i] = np.ascontiguousarray(np.rot90(b_obs[i], rot, axes=(1, 2)))
  ```
- This yields an effective $8\times$ data expansion during training without increasing memory consumption.

---

## 4. Invariants & Edge Cases

1. **Non-Atomic Disk Corruption Prevention**: [`save`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L277-L303) writes to `<path>.tmp.npz` before executing `os.replace`. If the process is halted mid-save, the existing checkpoint remains uncorrupted.
2. **Checkpoint Capacity Resizing**: When [`load`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L304-L364) encounters a checkpoint saved with a different capacity, it reads the data using `mmap_mode="r"` and slices the maximum valid count per tier into the current buffer layout, preventing index errors.
3. **Empty Buffer Protection**: [`sample_batch`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L179-L276) verifies `len(self) > 0` and raises an explicit `ValueError`, preventing undefined behavior or NaN returns during training initialization.
