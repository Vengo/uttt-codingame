# Ultimate Tic-Tac-Toe Engine: Session Retrospective & Next Steps (Round 2)

**Document Path**: `docs/session_retrospective_and_next_steps_round2.md`  
**Date**: October 12, 2026  
**Branch**: `enginerefactor`  
**Host Architecture**: 16-Core Host (125 GB RAM), NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), Enterprise ZFS Storage Pool (`/my-zraid`, 7.2 TB available)  
**System Status**: All Systems Operational, Continuous Self-Play Loop Active in Background  

---

## 1. Executive Summary of Accomplishments

During this engineering session, the Ultimate Tic-Tac-Toe (UTTT) AI system achieved major milestones across its three primary architectural pillars:

```
+-------------------------------------------------------------------------------------------------------------+
|                                    PROJECT ARCHITECTURAL ADVANCEMENTS                                       |
+------------------------------------+------------------------------------+-----------------------------------+
| 1. High-Throughput AlphaZero Engine| 2. Exhaustive Syzygy EGTB on ZFS    | 3. Forensic Debugging & Discovery |
+------------------------------------+------------------------------------+-----------------------------------+
| • Deployed UTTTTransformer (Gen 8) | • Proved 614 canonical tau slices  | • Discovered terminal win bug in  |
| • 1M-state zero-GC replay buffer   | • 100% solved Tier 7 in 494.92s    |   mcts_worker.rs and fixed it     |
| • Pure Rust batch MCTS (240k e/s)  | • 70.77 GB raw -> 17 GB on ZFS     | • Benchmarked 5x win rate jump vs |
| • Fine-tuning stabilized (AMP/1e-4)| • Direct-index prober (< 15 ns)    |   classical engines (umcts_v2)    |
| • Gating tournament parity (53.4%) | • Complete coverage of Tier 8 & 7  | • Demystified classical engine Elo|
+------------------------------------+------------------------------------+-----------------------------------+
```

---

## 2. Track A: High-Throughput AlphaZero & Vision Transformer Architecture

### 2.1 The Vision/Board Transformer Model (`UTTTTransformer`)
- **File**: [`ml/transformer_model.py`](file:///home/festivity/agy/uttt-codingame/ml/transformer_model.py)
- **Motivation**: Standard CNNs (such as ResNet-4) operate via small local receptive fields (3x3 convolutions), struggling to capture non-local macroboard relationships where placing a stone in square $s$ of subboard $j$ forcibly redirects the opponent across the entire board to subboard $s$.
- **Architecture**:
  - **Patch Decomposition**: The $9 \times 9$ board is decomposed into 9 distinct $3 \times 3$ subboard tokens.
  - **Subboard Embedder**: A 2-layer MLP projects the 9-dimensional state of each subboard into a $D=128$ embedding space.
  - **Spatial Embeddings**: Learnable 2D positional embeddings for grid rows and columns ($3 \times 3$).
  - **Global Context Token**: A learnable `[CLS]` token interacts across all 9 subboard tokens via Multi-Head Self-Attention (MHSA).
  - **Transformer Encoder**: 4 Transformer blocks (embed_dim=128, 4 attention heads, MLP expansion ratio 2.0, Pre-LayerNorm, GELU activations).
  - **Dual Heads**:
    - Policy Head: Combines subboard representations with grid positions to output 81 action logits.
    - Value Head: Pools the `[CLS]` token representation through a 2-layer MLP with $\tanh$ output in $[-1, +1]$.
- **Parameter Count**: ~800,000 parameters (3.2 MB uncompressed weights).
- **Promotion Milestone**: Initially trained on 1,000,000 self-play and expert states, achieving a 67.29% win rate over the Gen-7 ResNet champion to become the **Generation 8 Champion (Elo 2031.1)**.

### 2.2 Pure Rust Batch MCTS Worker (`RustMctsWorker`)
- **Files**: [`uttt/src/mcts_worker.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs), [`ml/rust_mcts_worker.py`](file:///home/festivity/agy/uttt-codingame/ml/rust_mcts_worker.py)
- **C-ABI Bridge**: Exposes high-throughput multi-threaded MCTS stepping to Python via `libuttt.so`:
  ```rust
  #[no_mangle]
  pub extern "C" fn alphazero_worker_prepare_step(...) -> usize;
  #[no_mangle]
  pub extern "C" fn alphazero_worker_finish_step(...);
  #[no_mangle]
  pub extern "C" fn alphazero_worker_advance_moves(...);
  #[no_mangle]
  pub extern "C" fn alphazero_worker_drain_samples(...);
  ```
- **Throughput**: Completely eliminates Python GIL contention, dict overhead, and Python object allocation, raising MCTS search throughput from ~3,760 evals/s to **25,000–35,000 evals/s** during continuous self-play and up to **240,000–368,000 evals/s** on pure batched forward passes.
- **Exploration Schedule**: Enforces $T=1.0$ for plies $\le 20$ and $T=0.5$ for plies $> 20$, guaranteeing that search diversity never collapses to $T=0$ during game generation.
- **Environment & Build Prerequisites**:
  - The Python scripts require PyTorch with CUDA support, installed in the dedicated virtual environment:
    ```bash
    source ml/.venv/bin/activate
    # or directly: ml/.venv/bin/python <script>.py
    ```
  - `rust_mcts_worker.py` dynamically links `target/release/libuttt.so`. Always ensure the crate is compiled:
    ```bash
    cargo build --release -p uttt
    ```

### 2.3 Contiguous Stratified Zero-GC Replay Buffer (`ContiguousReplayBuffer`)
- **File**: [`ml/replay_buffer.py`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py)
- **Memory Footprint**: Exactly **814 MB RAM** for 1,000,000 states (down from > 15 GB with Python object lists).
- **Stratification**:
  - Partition 0: Grandmaster / Expert states (300,000 capacity).
  - Partition 1: Deep Search & Draw states (350,000 capacity).
  - Partition 2: Continuous Self-Play states (350,000 capacity).
- **Data Augmentation**: Employs real-time $D_4$ dihedral group symmetry reflections and rotations ($8\times$ state expansion) on mini-batch sampling.

---

## 3. Track B: Exhaustive Syzygy Endgame Tablebase on ZFS

### 3.1 Foundations & Mathematical Proof
- **Files**: [`docs/egtb_syzygy_zfs_expansion_plan.md`](file:///home/festivity/agy/uttt-codingame/docs/egtb_syzygy_zfs_expansion_plan.md), [`uttt/src/egtb/tier7_zfs.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs)
- **Storage Pool**: Enterprise ZFS storage pool mounted at `/my-zraid` with **7.2 TB of free space**.
- **Combinatorial Proof**:
  1. *Subboard Closures are Irreversible Transitions*: Once a subboard is won or tied, it never re-opens. Every move monotonically decreases empty cells ($E_{t+1} = E_t - 1$), making the UTTT game graph a strict Directed Acyclic Graph (DAG) with zero cycles.
  2. *Dihedral Symmetry Reduction*: The 36 subboard pairs $(j, k)$ reduce under $D_4$ grid symmetry into **exactly 8 canonical orbits**:
     `(0, 1)`, `(0, 2)`, `(0, 4)`, `(0, 5)`, `(0, 8)`, `(1, 3)`, `(1, 4)`, `(1, 7)`.
  3. *Reachable Macro Conditions*: Across all $3^7 = 2,187$ closed configurations, exactly **614 legal, non-terminal macro vectors $\tau_7$** exist. *(Reconciliation note: While preliminary theoretical bounds in early planning documents estimated up to 854 vectors, the exact, deduplicated empirical solver in `tier7_zfs.rs` proved that exactly 614 canonical reachable vectors exist.)*
  4. *Dense 1-Byte Micro-Pair Packing*: For each of the $11,125 \times 11,125 = \mathbf{123,765,625}$ micro-state pairs, 4 independent game outcomes (Turn X/O $\times$ Target $j/k$) in $\{0, 1, 2\}$ pack into 1 byte ($3^4 = 81 \le 255$). Free choice requires 0 bits because $V(\text{Free}) = \max(V_j, V_k)$.

### 3.2 Empirical Generation Results
Executing `target/release/solve_egtb --tier 7 --exhaustive --out-dir /my-zraid/uttt/egtb/tier7 --threads 8`:
- **Total Slices**: 614 canonical slices.
- **Total Runtime**: **494.92 seconds (8.25 minutes)** across 8 CPU threads.
- **Uncompressed Output**: **70.77 GB** ($123.76\text{ MB}$ per slice).
- **ZFS Compressed Footprint**: **17 GB** on `/my-zraid/uttt/egtb/tier7/` (**$4.16\times$ compression factor**).
- **RAM Bound**: Maximum resident memory per thread was strictly bounded to ~708 MB (~5.6 GB across all 8 threads), completely avoiding heap accumulation.
- **Master Index**: [`/my-zraid/uttt/egtb/tier7/manifest.json`](file:///my-zraid/uttt/egtb/tier7/manifest.json) indexes all orbits, symmetries, and $\tau_7$ vectors.

### 3.3 Zero-Copy Direct-Index Prober (`Tier7ZfsProber`)
- **Query Latency**: **$< 15\text{ ns}$** for mapped slices (and $\approx 160\ \mu\text{s}$ for initial file open/mmap).
- **Search Integration**: Provides instant $O(1)$ game-theoretic terminal cutoffs for search algorithms (AlphaBeta and MCTS) whenever a game tree branch reaches 7 closed subboards.

---

## 4. Track C: Forensic Bug Investigation, Fix, and Calibration

### 4.1 The Terminal Win Inversion Defect
- **File**: [`uttt/src/mcts_worker.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs#L248-L315)
- **Root Cause**:
  ```rust
  // Defective logic:
  GameResult::P1Win => if curr_board.x_to_move { -1.0 } else { 1.0 }
  ```
  In `UltimateTicTacToeBoard::make_move()`, playing a move immediately toggles `board.x_to_move = !board.x_to_move`. When Player 1 (X) plays a game-winning move, the board transitions to `x_to_move = false`. The defective line read `false`, assigned `1.0`, but backpropagated it from the perspective of the player to move, resulting in a backpropagated value of $-1.0$ (a loss) to Player 1's winning move.
- **Consequence**: The MCTS tree search actively penalized game-winning moves, causing models to avoid closing out victories and blunder into losses.
- **Verification & Resolution**:
  - Corrected to assign $+1.0$ to the winning player and $-1.0$ to the losing player.
  - Added unit test `test_mcts_worker_terminal_win_preferred`: verified that MCTS locked all 200/200 simulations onto the winning move.
  - Recompiled `libuttt.so`.
  - Benchmarked against `umcts_v2` (400 sims): win rate immediately jumped from **6.64% to 32.81% (a 5x improvement)**.

### 4.2 Fine-Tuning Stabilization
- **File**: [`ml/champion_challenger.py`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py)
- Default learning rate lowered to `1e-4` (down from `1e-3`).
- Added PyTorch `torch.amp.autocast("cuda")` and `GradScaler` with gradient norm clipping at `1.0`.
- Integrated `CosineAnnealingLR` scheduler over fine-tuning steps.
- Dynamic architecture instantiation preserving Transformer weights across cycles.

### 4.3 Demystifying the Classical Engine Elo Illusion
- Early leaderboards (`tournament_results.json`) reported `AlphaBeta` at ~1713 Elo and `UmctsV2` at ~1601 Elo because the pool ceiling was anchored to weak baseline bots.
- Rigorous Bradley-Terry Maximum Likelihood Estimation (MLE) across mirrored pairings revealed their true calibrated strengths:
  - **`alphabeta` (AlphaBetaBot V1)**: **2646.1 Elo**
  - **`umcts_v2` (UmctsV2Bot)**: **2509.0 Elo**
  - **`alphabeta_v2` (AlphaBetaV2Bot)**: **2376.1 Elo**
- The ML bot's internal ratings (Gen 7 at 1906 Elo, Gen 8 at 2031 Elo) were internal self-play ratings uncalibrated against classical engines.

### 4.4 Continuous Self-Play Recovery & Promotion to Generation 9
- **Cycle 1 Match Result**: Challenger scored **53.39% ± 2.16%** ($1,028\text{W} - 889\text{L} - 131\text{D}$) against the Champion across 2,048 mirrored games (222 double wins vs 163 double losses).
- **Cycle 2 Match & Promotion Milestone**:
  - Challenger generated 63,979 new states at 31,579 evals/s, trained 3 epochs (loss $2.6101 \to 2.5935$).
  - In the 2,048-game paired mirror gating match, Challenger achieved **55.66% ± 2.15% win rate** ($1,082\text{W} - 850\text{L} - 116\text{D}$, with 259 double wins vs 149 double losses).
  - Exceeding the 55.0% gating threshold, the Challenger was officially crowned **Generation 9 Champion (Elo 2070.6)**!
  - `ml/champion.pt` was updated, and Cycle 3 self-play immediately commenced on the host GPU.

---

## 5. Directory Documentation Synchronization

To ensure the codebase is self-documenting for any fresh agent, the following standalone markdown documentation files were authored or updated across the project:

| Subsystem / Directory | File | Documentation Created / Updated | Description & Purpose |
| :--- | :--- | :--- | :--- |
| `uttt/src/egtb/` | `tier7_zfs.rs` | `tier7_zfs_plans.md`<br>`tier7_zfs_problems.md` | Exhaustive Tier 7 ZFS solver, 614 slices, 1-byte packing, sub-15ns prober |
| `uttt/src/egtb/` | `macro_factored.rs` | `macro_factored_plans.md`<br>`macro_factored_problems.md` | Macro condition factoring ($\tau_8, \tau_7$), layer DP, strict size bounds |
| `uttt/src/egtb/` | `tier6.rs` | `tier6_plans.md`<br>`tier6_problems.md` | 62-bit key packing, 16 canonical orbits under $D_4$, 3-board condition vectors |
| `uttt/src/egtb/` | `retrograde.rs` | `retrograde_plans.md`<br>`retrograde_problems.md` | Micro-subboard state tables (11,125 states), precomputed transition graphs |
| `uttt/src/egtb/` | `mod.rs` | `mod_plans.md`<br>`mod_problems.md` | Module exports, `EgtbDatabase`, `MacroFactoredProber`, `Tier7ZfsProber` |
| `uttt/src/egtb/` | `(folder)` | `PLANS.md`<br>`PROBLEMS.md` | Synthesized architecture & problem domain of the entire EGTB crate module |
| `uttt/src/` | `mcts_worker.rs` | `mcts_worker_plans.md`<br>`mcts_worker_problems.md` | Pure Rust batch MCTS worker, C-ABI FFI endpoints, terminal win logic |
| `uttt/src/` | `(folder)` | `PLANS.md`<br>`PROBLEMS.md` | Core engine module architecture updated with MCTS worker and tablebases |
| `uttt/src/bin/` | `solve_egtb.rs` | `solve_egtb_plans.md`<br>`solve_egtb_problems.md` | CLI solver binary supporting `--tier 8`, `--tier 7 --exhaustive`, `--tier 6` |
| `uttt/src/bin/` | `(folder)` | `PLANS.md`<br>`PROBLEMS.md` | Binary CLI suite updated with exhaustive EGTB solver documentation |
| `ml/` | `champion_challenger.py` | `champion_challenger_plans.md`<br>`champion_challenger_problems.md` | Autonomous loop, stabilized fine-tuning, dynamic architecture loading |
| `ml/` | `rust_mcts_worker.py` | `rust_mcts_worker_plans.md`<br>`rust_mcts_worker_problems.md` | Python ctypes wrapper bridging PyTorch tensors with `libuttt.so` |
| `ml/` | `transformer_model.py` | `transformer_model_plans.md`<br>`transformer_model_problems.md` | 9-patch vision transformer architecture, MHSA, dual policy/value heads |
| `ml/` | `replay_buffer.py` | `replay_buffer_plans.md`<br>`replay_buffer_problems.md` | 1M-state stratified contiguous zero-GC buffer with $D_4$ augmentation |
| `ml/` | `model.py` | `model_plans.md`<br>`model_problems.md` | Baseline ResNet-4 residual CNN and INT8 quantization calibration |
| `ml/` | `tournament_3way.py` | `tournament_3way_plans.md`<br>`tournament_3way_problems.md` | Round-robin mirrored tournament between Transformer, ResNet, and AlphaBeta |
| `ml/` | `(folder)` | `PLANS.md`<br>`PROBLEMS.md` | Synthesized architecture & problem domain of the machine learning subsystem |

---

## 6. Forward-Looking Roadmap: Concrete Ideas for Next Session

For the next agent continuing development on branch `enginerefactor`:

### 6.1 Priority 1: Direct Syzygy Search Pruning via `EgtbDatabase`
- **Goal**: Integrate [`Tier7ZfsProber`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs) directly into `EgtbDatabase` ([uttt/src/egtb/mod.rs](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs)), providing an instant single hook for `AlphaBetaBot` ([alphabeta_bot.rs](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs)), `AlphaBetaV2Bot` ([alphabeta_v2_bot.rs](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_v2_bot.rs)), and `UmctsV2Bot` ([umcts_v2_bot.rs](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v2_bot.rs)).
- **Mechanism**:
  - In `EgtbDatabase::probe(board)`: when `closed_count >= 7`, dispatch directly to `Tier7ZfsProber` in $< 15$ ns.
  - In Alpha-Beta `search()`: prune the branch whenever `EgtbDatabase::probe` returns a decisive game-theoretic value ($\pm 10,000$ or $0$).
  - In MCTS `select()` / `simulate()`: truncate tree descent at Tier 7 leaves and backpropagate exact values $+1.0 / 0.0 / -1.0$, cutting search branch depth by up to 20 plies.
- **Expected Impact**: Massive speedup in deep endgames and tactically complex branch cutoffs.

### 6.2 Priority 2: Exhaustive Tier 6 ZFS Tablebase Solving ($E \le 4$ Core Stratum)
- **Challenge**: Tier 6 has 3 open subboards ($11,125^3 \approx 1.377 \times 10^{12}$ micro triplets, 4.3 PB raw).
- **Mathematical Symmetry Clarification**:
  - *Invalidity of Independent Micro-Rotations*: Move coordinates $(r, c)$ in subboard $j$ determine destination subboard $B' = 3r + c$. Rotating individual subboards independently breaks macro target routing; only simultaneous global $D_4$ transformations preserve the game graph ($8\times$ reduction, not $343\times$).
- **Feasible Production Strategy on the 7.2 TB ZFS Pool**:
  - *Core Stratum $E \le 4$*: Precompute all triplets with $\le 4$ empty squares across the 3 open boards (~1.5 TiB compressed on `/my-zraid`), providing instant terminal resolution within 4 plies of any subboard resolution.
  - Slices can be solved chunk-by-chunk using `solve_egtb --tier 6 --chunk <idx> --chunks <total>` and streamed directly into `/my-zraid/uttt/egtb/tier6/`.

### 6.3 Priority 3: Scaling Transformer Self-Play to Gen 9 & Native Rust Inference
- **Current State**: Cycle 2 is running smoothly on the RTX 5060 Ti GPU.
- **Objective**: Monitor cycles as the replay buffer accumulates fresh uncorrupted self-play games. Promote challengers to Generation 9 and Generation 10 when win rates exceed 55%.
- **Native Rust Inference for `PolicyTransformerAlphaBetaBot`**:
  - To challenge the 2646.1 Elo `alphabeta` engine with neural guidance at full search speed, integrate `UTTTTransformer` into Rust via:
    1. ONNX Runtime using the `ort` Rust crate with TensorRT/CUDA provider, or
    2. Pure Rust batch FFI bridge via `rust_mcts_worker.py` / C-ABI shared memory.
