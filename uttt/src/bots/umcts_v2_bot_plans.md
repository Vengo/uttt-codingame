# Plans & Progress: `uttt/src/bots/umcts_v2_bot.rs`

This document details MCTS-Solver and tree harvesting implementation in `uttt/src/bots/umcts_v2_bot.rs`.

---

## 1. Plan & Architecture
- **MCTS-Solver with Bound Proofs**:
  - `ProofStatus`: `Unproven`, `ProvenWin`, `ProvenLoss`, `ProvenDraw`.
  - When a terminal state is encountered during selection/expansion, its proof status is assigned directly.
  - Proofs propagate up the tree during the backpropagation step (`update_proof_status`). If root is proven win, search terminates early.
- **Move Selection Priority**:
  1. Plays any child that is a `ProvenLoss` from child's perspective (i.e. `ProvenWin` for us).
  2. Otherwise picks child with highest visit count among legal non-losing children.
- **`harvest_tree` with UCT Dominance**:
  ```rust
  let is_dominant = best_child_visits >= (second_child_visits as f64 * 1.35) as u32
      || (curr_node.visits > 0 && (best_child_visits as f64 / curr_node.visits as f64) >= 0.40);
  ```
  Only moves satisfying dominance and visit floors ($N_{\text{node}} \ge 100$, child $N \ge 50$) along PV paths (depth $\le 8$) are extracted.

---

## 2. Status
- The flagship MCTS search engine of the repository.
- Used by the continuous study daemon to deepen the master opening book.
