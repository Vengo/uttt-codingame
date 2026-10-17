# Track 4 Plan: Isolated Per-Player Search Trees in Tournament Mode & Challenger Weight Persistence Across Non-Regressing Cycles

**Status**: Reviewed and Approved by Isolated Plan Reviewer (`62dbf5f6-9232-4b3e-b32b-130223f1f3ce`)

---

## 1. Problem Description

### Problem 4A: Champion and Challenger Share the Same MCTS Search Tree Memory in Head-to-Head Tournaments
In `uttt/src/mcts_worker.rs`, each `GameState` currently has a single MCTS arena (`nodes: Vec<MctsNode>`, `children: Vec<MctsChild>`).
In head-to-head tournament matches (`run_parallel_mirror_tournament` in `ml/parallel_mirror_eval.py` and `run_neural_vs_neural_match` / `run_neural_vs_alphabeta_match` in `ml/tournament_3way.py`):
- On Ply `t`, Player 1 (e.g., Challenger) runs MCTS simulations that populate `game.nodes` and `game.children` using Challenger's policy priors and value evaluations.
- Then `advance_moves_tournament_recorded` chooses `chosen_action`, applies it to `game.board`, and calls `game.reroot_at_action(chosen_action, 0.0, 0.0)`.
- On Ply `t + 1`, it is Player 2's turn (e.g., Champion). Because `game.nodes` and `game.children` were re-rooted in place, Champion starts its search inside the subtree that Challenger built and evaluated on Ply `t` (and on Ply `t + 2`, Challenger inherits Champion's evaluations).

### Problem 4B: `ml/champion_challenger.py` Resets `challenger = copy.deepcopy(champion)` After Every Non-Promoted Cycle
In `ml/champion_challenger.py`, `challenger` is overwritten with `copy.deepcopy(champion)` at the start of every cycle and trained for only 3 epochs (`1,200` batches). At Generation 29 (Elo 2583.2), a single 3-epoch pass moves the challenger enough to achieve statistically significant decisive-pair wins (`55.0%–56.16%` decisive WR, `p = 0.020–0.047`, `51.6%–52.0%` overall WR), which falls slightly below the Gate B promotion threshold (`win_rate >= 0.525 and decisive_wr >= 0.565 and p_val < 0.05`). Resetting `challenger` back to `champion` discards those weights instead of persisting `challenger` across non-regressing cycles.

---

## 2. Implementation Plan

### Step 1: Give Each Player Its Own Isolated Search Tree in `GameState` (`uttt/src/mcts_worker.rs`)
1. Add a second, completely disjoint tree arena to `GameState`:
   ```rust
   pub struct GameState {
       pub board: UltimateTicTacToeBoard,
       pub nodes: Vec<MctsNode>,
       pub children: Vec<MctsChild>,
       pub alt_nodes: Vec<MctsNode>,
       pub alt_children: Vec<MctsChild>,
       ...
   }
   ```
   - `nodes` / `children` is always the MCTS search tree belonging to the player currently to move (`board.x_to_move`).
   - `alt_nodes` / `alt_children` is the MCTS search tree belonging to the opposing player (`!board.x_to_move`).
2. Implement `pub fn advance_both_trees_and_swap(&mut self, action: u8)` on `GameState`:
   ```rust
   pub fn advance_both_trees_and_swap(&mut self, action: u8) {
       self.reroot_at_action(action, 0.0, 0.0);
       std::mem::swap(&mut self.nodes, &mut self.alt_nodes);
       std::mem::swap(&mut self.children, &mut self.alt_children);
       self.reroot_at_action(action, 0.0, 0.0);
   }
   ```
   - Call `game.advance_both_trees_and_swap(chosen_action)` on non-terminal moves in `advance_moves_tournament_recorded` and `mcts_worker_apply_moves_batch`.
   - Keep `game.reroot_at_action(chosen_action, dirichlet_alpha, dirichlet_eps)` in `advance_moves` (self-play, where a single model plays both sides).
   - Ensure `game.alt_nodes.clear()` and `game.alt_children.clear()` are called whenever a game terminates or resets (`advance_moves`, `advance_moves_tournament_recorded`, `mcts_worker_apply_moves_batch`, `reset_all`, and `mcts_worker_set_game_moves`).

### Step 2: Persist Challenger Across Cycles Unless It Suffers a Statistically Significant Regression (`ml/parallel_mirror_eval.py` & `ml/champion_challenger.py`)
1. In `ml/parallel_mirror_eval.py`:
   - Add `regression_p_value: float` and `regressed: bool` to `MirrorTournamentResult`:
     - `regression_p_value = binomial_p_value(champ_double_wins, decisive_pairs, p=0.5)`
     - `regressed = (win_rate < 0.50 and regression_p_value < 0.05) or ((win_rate + margin) < 0.50)`
2. In `ml/champion_challenger.py`:
   - Initialize `challenger: Optional[torch.nn.Module] = None` before the `for cycle in range(1, total_cycles + 1):` loop (loading from `challenger_path` if it exists and matches `type(champion)`, else `copy.deepcopy(champion).to(device)`).
   - In Phase 2 (Training):
     - If `challenger is None`, set `challenger = copy.deepcopy(champion).to(device)`.
     - Train `challenger` from its persisted weights for `train_epochs` epochs.
   - In Phase 3 (Arena Gating):
     - If `tourney_res.promoted`:
       - Crown `challenger` as the new `champion` (`champion = copy.deepcopy(challenger).to(device)`), save `champion.pt`, and keep `challenger` for the next cycle.
     - Else if `tourney_res.regressed`:
       - Log regression warning (`reg_p = tourney_res.regression_p_value`) and reset `challenger = copy.deepcopy(champion).to(device)` (and overwrite `challenger_path`).
     - Else (non-regressing defense):
       - Log that `challenger` is persisted across cycles (`reg_p = tourney_res.regression_p_value >= 0.05`) to accumulate training in the next cycle.

---

## 3. Testing Plan
1. **`test_dual_trees_never_share_memory_in_tournament_mode` (`uttt/src/mcts_worker.rs`)**:
   - Verify that on Ply 0 (Player X), simulations populate `game.nodes` while `game.alt_nodes` stays empty.
   - After `advance_moves_tournament(0.0)` makes Player X's move `m_0`, verify that on Ply 1 (Player O's turn), `game.nodes` is empty (Player O has not searched yet) while `game.alt_nodes` holds Player X's re-rooted subtree under `m_0`.
   - After Player O runs simulations on Ply 1 (expanding `m_1` with distinct values) and `advance_moves_tournament(0.0)` makes `m_1`, verify that on Ply 2 (Player X's turn), `game.nodes` holds Player X's 2-ply re-rooted subtree under `(m_0, m_1)` with Player X's own values, and `game.alt_nodes` holds Player O's subtree under `m_1`.
2. **Python unit test for `regressed` detection (`ml/parallel_mirror_eval.py`)**:
   - Verify `regressed` is `False` on parity / near-miss challenger leads (e.g. `184 vs 153` or `140 vs 160` where `reg_p = 0.136 >= 0.05`) and `True` on statistically significant regressions (e.g. `110 vs 170` where `reg_p < 0.001`).
