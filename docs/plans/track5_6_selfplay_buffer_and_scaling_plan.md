# Tracks 5 & 6: Breaking the Gen-42 Self-Play Plateau + Function-Preserving 2x Transformer Scaling

**Status**: Approved by isolated `plan-reviewer` subagent (`62dbf5f6-9232-4b3e-b32b-130223f1f3ce`).

---

## 1. Problem Description

After deploying Track 4 (disjoint MCTS trees + Challenger persistence), Champion promoted 13 times in 45 cycles from **Gen-29 (2583.2 Elo)** to **Gen-42 (2903.8 Elo)** (achieving a **94.63% win rate / 229:1 double-win ratio** against upgraded `AlphaBetaV2Bot` in the 3-way round-robin benchmark). However, across Cycles 46–99 at Gen-42, promotions stalled completely, and in 35+ cycles where Challenger was freshly initialized from `copy.deepcopy(Gen-42)`, training for 3 epochs on the replay buffer caused Challenger to **regress significantly against Gen-42** (`Win Rate 46.1%–48.1%`, `reg_p < 0.05`).

Four concrete issues in the self-play and training pipeline explain this regression:

### Problem 5A: Outer-Loop `move_step` Overrides Per-Game Temperature Schedule (`ml/champion_challenger.py` & `uttt/src/mcts_worker.rs`)
In `ml/champion_challenger.py` (lines 368–381):
```python
for move_step in range(1, moves_per_cycle + 1):
    ...
    temp = 1.0 if move_step <= 20 else 0.5
    new_finished = worker.advance_moves(temperature=temp)
```
Because `moves_per_cycle = 500` and each of the 128 parallel games finishes and resets asynchronously to `game.moves_played = 0`:
1. For **96% of every cycle (`move_step = 21..500`)**, `champion_challenger.py` passes `temperature = 0.5`, overriding the per-game check (`temperature <= 0.0`) so newly reset games play their opening moves (`game.moves_played <= 20`) at `T = 0.5` instead of `T = 1.0`.
2. Every game plays its late midgame and endgame (`game.moves_played > 28`) at `T = 0.5` all the way to move 81. Sampling moves proportionally to $N^{1/0.5} = N^2$ in tactical endgames introduces random blunders that corrupt the terminal game outcome $z \in \{-1, 0, +1\}$ assigned to all earlier states in `game.history`.

### Problem 5B: Pure Terminal Game Outcome $z$ Without MCTS Root Q-Value ($Q_{\text{root}}$) Target Blending (`uttt/src/mcts_worker.rs`)
Currently, `CompletedSample.value` is 100% the terminal outcome $z \in \{-1, 0, +1\}$, discarding the 800-simulation MCTS root value evaluation $Q_{\text{root}}$ (which incorporates MCTS-Solver and exact Tier 6/7 EGTB proofs). Blending $v_{\text{target}} = 0.5 z + 0.5 Q_{\text{root}}$ reduces value variance and prevents late-game exploratory moves from inverting early-game value targets.

### Problem 6A: Replay Buffer Dilution ("Regression to the Buffer Mean") (`ml/replay_buffer.py`)
Because the buffer holds `1,000,000` states (`100,000` frozen early-gen `GM` states, `450,000` `deep_draw` states, and `450,000` `self_play` states) and each cycle adds ~64,000 states, uniform sampling draws >85% of every training batch from 15–20 generations ago (plus 10% from the frozen bootstrap `GM` tier). At Gen-42 (2903.8 Elo), training on uniformly sampled old buffer states pulls Challenger's weights backward toward the weaker historical buffer average.

### Problem 6B: Fresh `AdamW` Optimizer Shock Every Cycle (`ml/champion_challenger.py`)
Re-creating `AdamW` from scratch every cycle resets second-moment estimates ($v_t = 0$) with zero warmup at peak `lr = 1e-4`, causing an initial step shock on the Transformer weights.

### Problem 6C: Function-Preserving 2x Transformer Scaling (`ml/transformer_model.py`)
`UTTTTransformer` currently hardcodes `d_model=128, nhead=4, num_layers=4, dim_feedforward=512` when loaded in `champion_challenger.py`. When a model reaches true capacity saturation (diagnosed when policy KL divergence $\text{KL}(\pi_{\text{MCTS}} \parallel p_\theta)$ and Value MSE on recent self-play states plateau for $\ge 10$ persisted cycles), we need a way to scale `UTTTTransformer` to **2x depth (`num_layers=8`, ~1.6M params)** via **function-preserving Net2Net depth expansion** (`expand_transformer_depth`) so the 8-layer network starts at the exact same output logits/values (and 2903.8+ Elo) as the 4-layer Champion on Step 0 rather than restarting from 1500 Elo.

---

## 2. Implementation Plan

### Step 1: Per-Game Temperature Schedule & Root Q-Value ($Q_{\text{root}}$) Blending (`uttt/src/mcts_worker.rs` & `ml/champion_challenger.py`)
1. In `ml/champion_challenger.py`:
   - Pass `temperature=0.0` (which activates the per-game `game.moves_played` schedule in Rust) to `worker.advance_moves(temperature=0.0)` instead of `temp = 1.0 if move_step <= 20 else 0.5`.
2. In `uttt/src/mcts_worker.rs`:
   - Update the auto per-game temperature schedule in `MctsWorker::advance_moves` (when `temperature <= 0.0`):
     ```rust
     let effective_temp = if temperature <= 0.0 {
         if game.moves_played <= 16 {
             1.0f32
         } else if game.moves_played <= 28 {
             0.5f32
         } else {
             0.15f32
         }
     } else if temperature < 0.05 {
         0.0f32
     } else {
         temperature
     };
     ```
   - Change `GameState.history` from `Vec<(UltimateTicTacToeBoard, [f32; 81], bool)>` to `Vec<(UltimateTicTacToeBoard, [f32; 81], bool, f32)>` (`(board, policy, x_to_move, root_q)`).
   - In `MctsWorker::advance_moves`, compute `root_q: f32` (from the perspective of `game.board.x_to_move`, the player to move at `root`) right after `policy` is constructed:
     - If `best_winning_action.is_some()` or `root.proof == ProofStatus::ProvenWin`: `root_q = 1.0`.
     - Else if `root.proof == ProofStatus::ProvenLoss`: `root_q = -1.0`.
     - Else if `root.proof == ProofStatus::ProvenDraw`: `root_q = 0.0`.
     - Else: compute the policy-weighted root evaluation over `root`'s children:
       ```rust
       let mut q_acc = 0.0f32;
       for c_off in 0..root.num_children {
           let ch = &game.children[(root.first_child + c_off as u32) as usize];
           let p = policy[ch.action as usize];
           if p > 0.0 {
               let child_v_for_root = match ch.proof {
                   ProofStatus::ProvenLoss => 1.0,
                   ProofStatus::ProvenWin => -1.0,
                   ProofStatus::ProvenDraw => 0.0,
                   ProofStatus::Unknown => -ch.q_value(),
               };
               q_acc += p * child_v_for_root;
           }
       }
       let root_q = q_acc.clamp(-1.0, 1.0);
       ```
   - When a game terminates in `MctsWorker::advance_moves`:
     ```rust
     for (b, pi, x_to_move, root_q) in game.history.drain(..) {
         let z = if x_to_move { p1_outcome } else { -p1_outcome };
         let v = (0.5 * z + 0.5 * root_q).clamp(-1.0, 1.0);
         ...
     ```

### Step 2: Recency-Weighted Sampling in `ContiguousStratifiedReplayBuffer` (`ml/replay_buffer.py`)
1. Add a helper `_sample_ring_recency(start: int, count: int, ptr: int, capacity: int, n: int, recency_window: int = 120_000, recent_share: float = 0.70) -> np.ndarray` to `ContiguousStratifiedReplayBuffer`:
   - When `count > recency_window`:
     - Draw `n_recent = int(round(n * recent_share))` samples uniformly from the most recently inserted `recency_window` slots (`start + ((ptr - 1 - offset) % count)` for `offset in [0, recency_window)`).
     - Draw the remaining `n - n_recent` samples uniformly across `[start, start + count)`.
   - When `count <= recency_window`, sample uniformly across `[start, start + count)`.
2. In `sample_batch`:
   - Cap the sampling proportion of the frozen `gm` bootstrap tier to at most `2%` (`p_gm = min(self.gm_count / total, 0.02)`) when `total >= 200_000`, splitting the remaining `98%` proportionally between `deep_draw` and `self_play`.
   - Use `_sample_ring_recency` for both `deep_draw` and `self_play` tiers so **70% of every training batch comes from the most recent ~240,000 states (~3–4 recent cycles)** while 30% regularizes across the full ring buffer.

### Step 3: Warmup + Cosine Scheduler & Optimizer Persistence (`ml/champion_challenger.py`)
1. In `ml/champion_challenger.py`:
   - Persist `optimizer: Optional[torch.optim.AdamW]` in memory across cycles whenever `challenger` is persisted (`not promoted and not tourney_res.regressed`), and reset `optimizer = None` whenever `challenger` is promoted or reset after regression.
   - At the start of Phase 2, if `optimizer` is reused, reset `pg["lr"] = eff_lr` and `pg["initial_lr"] = eff_lr` across `optimizer.param_groups` before attaching the new cycle's `LambdaLR` warmup+cosine scheduler (`warmup_steps = max(10, total_train_steps // 10)`, decaying to `0.1 * eff_lr`).
   - Use `eff_lr = min(lr, 5e-5)` when `meta["generation"] >= 25` (unless a lower `lr` was explicitly specified).
   - Log **Policy Entropy $H(\pi)$** and **Policy KL Divergence $\text{KL}(\pi \parallel p_\theta) = \text{CE}(\pi, p_\theta) - H(\pi)$** alongside Policy CE and Value MSE in Phase 2, and record `policy_kl` and `value_loss` in `cycle_record` so capacity saturation can be tracked automatically.

### Step 4: Auto-Detect Transformer Depth & Function-Preserving 2x Depth Scaling (`ml/transformer_model.py` & `ml/champion_challenger.py`)
1. In `ml/transformer_model.py`:
   - Add `infer_transformer_kwargs_from_state_dict(sd: dict) -> dict` that inspects `sd` keys (`transformer.layers.{i}.*`, `cls_token.shape[-1]`, `transformer.layers.0.linear1.weight.shape[0]`) to automatically infer `d_model`, `num_layers`, and `dim_feedforward` when loading any checkpoint.
   - Add `expand_transformer_depth(model: UTTTTransformer, new_num_layers: int) -> UTTTTransformer`:
     - Creates a new `UTTTTransformer(d_model=model.d_model, nhead=..., num_layers=new_num_layers, dim_feedforward=...)`.
     - Copies `input_proj`, `sb_embed`, `sq_embed`, `cls_token`, `norm`, `policy_head`, `value_head`, and `transformer.layers[0..old_num_layers]` directly from `model`.
     - For each newly added layer `idx in old_num_layers..new_num_layers`:
       - Copies `norm1`, `norm2`, `self_attn.in_proj_weight`, `self_attn.in_proj_bias`, `linear1.weight`, and `linear1.bias` from `model.transformer.layers[idx % old_num_layers]`.
       - Zero-initializes the residual output projections `self_attn.out_proj.weight`, `self_attn.out_proj.bias`, `linear2.weight`, and `linear2.bias` (`nn.init.zeros_`), guaranteeing exact functional equivalence (`max_abs_diff == 0.0`) at Step 0 in Pre-LN architecture.
2. In `ml/champion_challenger.py` and `ml/tournament_3way.py`:
   - Use `infer_transformer_kwargs_from_state_dict(sd)` when instantiating `UTTTTransformer` from a checkpoint, and add an optional `--expand-layers <N>` CLI flag to `ml/champion_challenger.py` so expanding Champion/Challenger to 8 layers is a single flag away whenever capacity saturation is reached.

---

## 3. Testing Plan
1. **Rust unit test (`test_per_game_temp_and_q_root_blending` in `uttt/src/mcts_worker.rs`)**:
   - Run self-play simulations and `advance_moves(0.0)` until a game finishes, and verify that `CompletedSample.value` reflects the $0.5 z + 0.5 Q_{\text{root}}$ blend and remains in `[-1.0, 1.0]`.
2. **Python unit test for Recency-Weighted Replay Buffer (`test_recency_weighted_sampling` in `ml/replay_buffer.py`)**:
   - Populate a `ContiguousStratifiedReplayBuffer` beyond `recency_window`, tag recent slots with `values = +1.0` and older slots with `values = -1.0`, sample a large batch, and assert that recent slots are sampled at the expected ~70%+ rate and `gm` slots are capped at $\le 2\%$.
3. **Python unit test for Function-Preserving 2x Transformer Expansion (`test_expand_transformer_depth_exact_equivalence`)**:
   - Create a 4-layer `UTTTTransformer`, expand it to 8 layers via `expand_transformer_depth(model, 8)`, verify parameter count doubles, verify forward-pass logits and values match the 4-layer model to `< 1e-6` across random boards, and verify 2 optimizer steps produce non-zero parameter updates across all 8 layers (`layers[0..8]`).
