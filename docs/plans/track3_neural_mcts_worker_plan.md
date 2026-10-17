# Track 3 Plan: Neural MCTS & Self-Play Engine (`RustMctsWorker`) Upgrades

**Status**: Reviewed and Approved by Isolated Plan Reviewer (`62dbf5f6-9232-4b3e-b32b-130223f1f3ce`)

---

## 1. Problem Description

### Problem 3A: No Exact Win/Loss/Draw Proof Propagation (MCTS-Solver) in `RustMctsWorker`
In `uttt/src/mcts_worker.rs`, `MctsNode` and `MctsChild` currently average terminal outcomes (`-1.0` or `0.0`) into `total_val / visits` without tracking game-theoretic proof status (`ProvenWin`, `ProvenLoss`, `ProvenDraw`). Even after discovering an immediate 1-ply winning child, PUCT continues exploring alternative moves or averaging noisy neural evaluations from subtrees with proven losses.

### Problem 3B: `RustMctsWorker` Never Probes the Endgame Tablebase (`EgtbDatabase`)
During self-play (`ml/champion_challenger.py`) and tournament evaluation (`ml/tournament_3way.py`), `MctsWorker::prepare_step` never queries `EgtbDatabase` when `curr_board.get_resolved_count() >= 6`. Probing the EGTB at leaves (`curr_idx > 0`) immediately resolves endgame leaves with their exact game-theoretic value (`+1.0`, `0.0`, or `-1.0`) and `ProofStatus` without requiring a GPU forward pass.

### Problem 3C: MCTS Tree Is Completely Wiped After Every Move (`nodes.clear()`)
In `advance_moves`, `advance_moves_tournament_recorded`, and `mcts_worker_apply_moves_batch`, `game.nodes.clear()` and `game.children.clear()` discard the entire search tree after every ply, throwing away 100–400+ simulations already invested in the chosen action's subtree.

### Problem 3D: Small GPU Batch Size (`B = 128`) in Self-Play Due to 1 Leaf Per Game Per Step
With `num_games = 128` in self-play, selecting 1 leaf per game per step limits GPU batch sizes to `<= 128`. Adding multi-leaf virtual loss (`k` leaves per game per step) enables 128 games to emit `128 * k` leaves per GPU batch while avoiding duplicate paths.

---

## 2. Implementation Plan

### Step 1: MCTS-Solver Exact `ProofStatus` & Backpropagation (`uttt/src/mcts_worker.rs`)
1. Define `ProofStatus` in the **node/child's current-mover perspective** (matching `MctsNode::total_val`, `MctsNode::terminal_val`, and `MctsChild::q_value()`, which are all from the perspective of `curr_board.x_to_move` at that node):
   ```rust
   #[derive(Clone, Copy, PartialEq, Eq, Debug)]
   #[repr(u8)]
   pub enum ProofStatus {
       Unknown = 0,
       ProvenWin = 1,   // Proven win (+1.0) for the player TO MOVE at this node
       ProvenLoss = 2,  // Proven loss (-1.0) for the player TO MOVE at this node (win for parent's mover!)
       ProvenDraw = 3,  // Proven draw (0.0) under optimal play from this node
   }
   ```
   Add `pub proof: ProofStatus` to both `MctsNode` and `MctsChild`.
2. **Terminal & EGTB leaf proof initialization**:
   - When a move from `parent` wins the game (`curr_board.get_win_state()` is `P1Win` or `P2Win`), the player to move at `curr_board` has already lost: `term_val = -1.0`, `proof = ProofStatus::ProvenLoss`, `is_terminal = true`.
   - When a move ties the game (`GameResult::Tie` or 81 moves with equal subboards): `term_val = 0.0`, `proof = ProofStatus::ProvenDraw`, `is_terminal = true`.
3. **Parent proof backup from children (`update_parent_proof`)**:
   - When backing up from a child `ch` to `parent`:
     - Copy `node.proof` to `ch.proof`.
     - If `parent.is_expanded && parent.num_children > 0`:
       - If **any** child `c` of `parent` has `c.proof == ProofStatus::ProvenLoss` (meaning choosing `c` leaves the opponent in a losing state, so `parent`'s mover wins!): set `parent.proof = ProofStatus::ProvenWin`, `parent.terminal_val = 1.0`.
       - Else if **all** children `c` of `parent` have `c.proof == ProofStatus::ProvenWin` (every move leaves the opponent in a winning state): set `parent.proof = ProofStatus::ProvenLoss`, `parent.terminal_val = -1.0`.
       - Else if **all** children `c` of `parent` are proven (`c.proof == ProofStatus::ProvenWin || c.proof == ProofStatus::ProvenDraw`) and at least one has `ProvenDraw`: set `parent.proof = ProofStatus::ProvenDraw`, `parent.terminal_val = 0.0`.
     - Note: Do **not** set `parent.is_terminal = true` on expanded interior nodes when `parent.proof` becomes proven, so subsequent simulations can continue descending into the winning child's subtree to build out the tree for the next turn's `reroot_at_action`.
4. **PUCT Selection (`prepare_step`)**:
   - At an expanded node:
     - If any child `c` has `c.proof == ProofStatus::ProvenLoss` (winning move for current mover), restrict selection to children with `c.proof == ProofStatus::ProvenLoss` (picking the one with highest PUCT / prior).
     - Otherwise, skip any child with `c.proof == ProofStatus::ProvenWin` (proven loss for current mover) unless all children have `c.proof == ProofStatus::ProvenWin`.
     - For a child with `c.proof == ProofStatus::ProvenDraw`, use exact `q = 0.0`.
5. **Root Move Selection (`advance_moves` and `advance_moves_tournament_recorded`)**:
   - If any root child `ch` has `ch.proof == ProofStatus::ProvenLoss` (proven win for root's mover):
     - Set `policy` to distribute probability mass exclusively over `ProvenLoss` children (one-hot on the highest-visit `ProvenLoss` child) and choose the highest-visit `ProvenLoss` child.
   - Otherwise, if some root children have `ch.proof == ProofStatus::ProvenWin` (proven loss for root's mover) and at least one child is `Unknown` or `ProvenDraw`, zero out the selection weights of the `ProvenWin` (losing) children so the bot never blunders into a proven loss when a draw or unknown continuation exists.

### Step 2: EGTB Leaf Probing in `MctsWorker` (`uttt/src/mcts_worker.rs`)
1. Add `pub egtb: std::sync::Arc<Option<crate::egtb::EgtbDatabase>>` to `MctsWorker` (shared via `OnceLock` across workers so `MctsWorker::new` loads it once without disk overhead).
2. In `prepare_step`, when lazily creating a child node (`curr_idx > 0`, while `win == GameResult::Unresolved` and `curr_board.get_resolved_count() >= 6`):
   - Query `egtb.probe_with_constants(&curr_board, z_consts)`.
   - If `Some(res)` is returned:
     - Determine outcome from `curr_board.x_to_move`'s perspective:
       - Win for `curr_board.x_to_move`: `is_term = true`, `term_val = 1.0`, `proof = ProofStatus::ProvenWin`.
       - Loss for `curr_board.x_to_move`: `is_term = true`, `term_val = -1.0`, `proof = ProofStatus::ProvenLoss`.
       - Tie: `is_term = true`, `term_val = 0.0`, `proof = ProofStatus::ProvenDraw`.
     - This immediately resolves the leaf and triggers exact backpropagation without needing a GPU forward evaluation.

### Step 3: MCTS Subtree Reuse Across Moves (`reroot_at_action` in `uttt/src/mcts_worker.rs`)
1. Implement `pub fn reroot_at_action(&mut self, action: u8, dirichlet_alpha: f32, dirichlet_eps: f32)` on `GameState`:
   - If `self.nodes.is_empty() || !self.nodes[0].is_expanded || self.nodes[0].num_children == 0`, clear `nodes` and `children` and return.
   - Find the root child `ch` with `ch.action == action`.
   - If `ch.node_idx == u32::MAX`, or if the child node `new_root = &self.nodes[ch.node_idx as usize]` has `!new_root.is_expanded || new_root.num_children == 0 || new_root.is_terminal`, clear `nodes` and `children` and return.
   - Otherwise, compact the subtree rooted at `ch.node_idx` into fresh `new_nodes` and `new_children` vectors via BFS/DFS:
     - New root at `new_nodes[0]` has `parent_idx = u32::MAX`, `parent_action = 255`.
     - Remap `first_child` on every copied `MctsNode`, `parent_idx` on every child `MctsNode`, and `node_idx` on every copied `MctsChild` where `node_idx != u32::MAX`.
   - If `dirichlet_eps > 0.0 && new_nodes[0].num_children > 1`, mix fresh root Dirichlet exploration noise into `new_children[0..new_nodes[0].num_children]`.
   - Swap `self.nodes = new_nodes; self.children = new_children;`.
2. Call `game.reroot_at_action(...)` in `advance_moves`, `advance_moves_tournament_recorded`, and `mcts_worker_apply_moves_batch` instead of wiping `game.nodes.clear(); game.children.clear();` (except when a game finishes and resets to a fresh opening).

### Step 4: Multi-Leaf Virtual Loss Batching (`uttt/src/mcts_worker.rs` & `ml/rust_mcts_worker.py`)
1. Support selecting up to `k` leaves per game in a single batch (`prepare_step_vloss(k, out_features, out_masks)` / `finish_step_vloss(in_logits, in_values)`), while keeping `prepare_step` / `finish_step` (`k = 1`) 100% compatible:
   - Store `pending_leaves: Vec<(u32, UltimateTicTacToeBoard, Vec<(u32, u32)>)>` on `GameState` (recording `(leaf_node_idx, leaf_board, path_of_(node_idx, child_idx))`).
   - During `prepare_step_vloss(k)`:
     - For `sub_sim in 0..k`:
       - Walk from root (`curr_idx = 0`) down the tree, recording each traversed `(curr_idx, chosen_child_idx)` in `path`.
       - Apply virtual loss along each traversed edge:
         - `game.nodes[curr_idx as usize].visits += 1;`
         - `game.nodes[curr_idx as usize].total_val -= 1.0;`
         - `game.children[chosen_child_idx].visits += 1;`
         - `game.children[chosen_child_idx].total_val += 1.0;` (since `q = -child.q_value()`, increasing `child.total_val` toward `+1.0` decreases `q` toward `-1.0`!).
       - If the reached leaf is `is_terminal`:
         - Revert the virtual loss along `path`, perform real terminal backpropagation + `update_parent_proof`, and continue to next `sub_sim`.
       - If the reached non-terminal leaf `curr_idx` is already in `game.pending_leaves` (duplicate leaf in narrow subtree):
         - Revert the virtual loss along `path` and break out of the `0..k` loop for this game.
       - Otherwise, push `(curr_idx, curr_board, path)` to `game.pending_leaves`.
   - In `finish_step` / `finish_step_vloss`:
     - First revert the virtual loss along each `path` in `game.pending_leaves`, then expand each leaf (if not yet expanded) and backpropagate its real neural `value` and `update_parent_proof`.
2. Expose `mcts_worker_prepare_step_vloss` and `step_simulation(model, device, virtual_loss_k=1)` in `ml/rust_mcts_worker.py`.

---

## 3. Testing Plan
1. **`test_mcts_solver_proves_win_and_loss` (`uttt/src/mcts_worker.rs`)**:
   - Construct a board with a forced win, run MCTS steps with uniform logits and 0.0 neural values, and verify that `ProofStatus::ProvenLoss` on the winning child propagates `ProofStatus::ProvenWin` to the parent and that `advance_moves_tournament` deterministically selects the winning move.
2. **`test_mcts_subtree_reuse_preserves_visits` (`uttt/src/mcts_worker.rs`)**:
   - Run 30 MCTS simulations from the opening position, inspect the chosen root child's `visits` count `V > 1`, advance the move, and assert that `game.nodes[0].visits == V` and all child indices are valid.
3. **`test_virtual_loss_explores_distinct_leaves` (`uttt/src/mcts_worker.rs`)**:
   - Expand the root node, run 1 step of `prepare_step_vloss(4)` + `finish_step`, and verify that 4 distinct children of the root are evaluated in a single batch and that all virtual losses are cleanly reverted (`nodes[0].visits == 5` and exact `total_val` sum).
