# Plans & Progress: `uttt/src/mcts_worker.rs`

This document details the high-throughput parallel MCTS batch worker, GPU-CPU zero-copy integration, and C-ABI export interface in [`uttt/src/mcts_worker.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs).

---

## 1. Architectural Design & Plan

- **High-Throughput Batch Worker Architecture**:
  - Bridges native Rust bitboard speed (12.4M plies/s) with GPU Tensor Core neural network inference (368k evals/s).
  - Executes $B$ parallel games concurrently across a Rayon worker pool (e.g. 16 threads).
  - Employs a synchronized lock-step execution pipeline:
    1. [`prepare_step`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs#L233): Parallel descent to tree leaves. Terminal leaves are backpropagated immediately on the CPU; non-terminal leaves are packed into contiguous shared memory buffers for GPU evaluation.
    2. GPU Forward Pass: Evaluates policy logits and scalar values in PyTorch.
    3. [`finish_step`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs#L398): Expands leaves with softmax priors and backpropagates neural values in parallel.
    4. [`advance_moves`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs#L523): Samples actions according to temperature $T$, updates board states, handles game conclusions, drains completed training samples, and resets slots with fresh openings.

- **Flat Arena Memory Layout**:
  - Eliminates pointer chasing, reference counting, and dynamic heap allocation during tree traversal.
  - Nodes and children are stored in flat contiguous vectors per game:
    * [`MctsNode`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs#L54): 32-byte struct containing `parent_idx: u32`, `parent_action: u8`, `visits: u32`, `total_val: f32`, `is_terminal: bool`, `terminal_val: f32`, `is_expanded: bool`, `first_child: u32`, `num_children: u16`.
    * [`MctsChild`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs#L34): 20-byte struct containing `action: u8`, `prior: f32`, `visits: u32`, `total_val: f32`, `node_idx: u32` (pointing to child `MctsNode`).

- **NCHW 5x9x9 Feature Plane Representation**:
  - `extract_nchw_features(board, out_feat, out_mask)` writes directly into host arrays:
    * Channel 0: Active player marks (81 floats).
    * Channel 1: Opponent marks (81 floats).
    * Channel 2: Active player won subboards (81 floats).
    * Channel 3: Opponent won subboards (81 floats).
    * Channel 4: Legal target subboard mask (81 floats).
    * Legal Mask: 81 floats indicating legal move indices.

- **PUCT Search & Exploration Noise**:
  - Action selection inside trees follows PUCT:
    $$Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
  - Root exploration noise is sampled via Dirichlet distribution $(\alpha, \epsilon)$ over legal child priors.
  - Softmax sampling over visit counts with two-stage temperature schedule ($T = 1.0$ for moves $\le 20$, $T = 0.5$ thereafter; $T < 0.05$ for greedy tournament evaluation).

- **C-ABI Export Interface for Python ctypes**:
  - Provides `extern "C"` functions with raw pointer APIs for zero-copy memory exchange:
    * Lifecycle: `mcts_worker_create`, `mcts_worker_destroy`, `mcts_worker_reset`.
    * Step execution: `mcts_worker_prepare_step`, `mcts_worker_finish_step`, `mcts_worker_advance_moves`.
    * Data extraction: `mcts_worker_drain_samples`, `mcts_worker_get_active_indices`.
    * Competitive tournament play: `mcts_worker_advance_moves_tournament`, `mcts_worker_advance_moves_tournament_recorded`, `mcts_worker_apply_moves_batch`.
    * Baseline Bot Batch APIs: `alphabeta_batch_choose_moves` (AlphaBetaV2Bot), `alphabeta_v1_batch_choose_moves` (AlphaBetaBot), `umcts_v2_batch_choose_moves` (UmctsV2Bot).

---

## 2. Types & Public APIs

- `MctsChild`, `MctsNode`: Contiguous flat tree elements.
- `GameState`: Per-game search context with arena buffers, board history, and local `SmallRng`.
- `CompletedSample`: Self-play training sample containing features ($5 \times 9 \times 9$), legal mask ($81$), policy ($81$), value ($f32$), and game length.
- `MctsWorker`: Core batch manager with `new()`, `reset_all()`, `prepare_step()`, `finish_step()`, `advance_moves()`, `advance_moves_tournament()`, and `advance_moves_tournament_recorded()`.
- C-ABI FFI exports: Complete suite of `extern "C"` functions for Python zero-copy training and evaluation.

---

## 3. Status & Verification

- Unit test `test_mcts_worker_terminal_win_preferred` verifies that parallel simulations with uniform priors correctly converge on immediate terminal winning lines at root.
- C-ABI exports verified for thread-safety and shared-memory tensor alignment.
