# Problems & Challenges: `uttt/src/bots/umcts_v3_bot.rs`

This document details the search, proof backpropagation, heuristic move priors, and endgame pruning challenges for `UmctsV3Bot` in `uttt/src/bots/umcts_v3_bot.rs`.

---

## 1. Context & Baseline Constraints
- **Preservation of Baseline**: `UmctsV2Bot` in [`umcts_v2_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v2_bot.rs) is strictly preserved "as is" as the reference MCTS-Solver baseline.
- **Goal of `umcts_v3`**: Significantly outperform `umcts_v2` and `alphabeta` in tournament play through in-tree endgame tablebase pruning, static move priors during node expansion, and opening book interception.

---

## 2. Core Technical Challenges

### 2.1 In-Tree Tablebase Pruning & Proof Backpropagation via `EgtbDatabase`
In `umcts_v2_bot.rs`, bound proofs (`ProofStatus::ProvenWin`, `ProofStatus::ProvenLoss`, `ProofStatus::ProvenDraw`) are derived purely from terminal game states reached at leaf nodes.
- **Challenge**: When expanding a node where $\ge 4$ subboards are resolved, immediately probe `EgtbDatabase::probe(&self, board, zobrist_constants)`. If the state is solved:
  - Assign the node exact game-theoretic bounds (`ProofStatus::ProvenWin`, `ProofStatus::ProvenLoss`, or `ProofStatus::ProvenDraw`).
  - Backpropagate the proof up the parent tree immediately without launching wasteful Monte Carlo rollouts.
  - Mark the node proven so tree traversal terminates further expansion beneath it.

### 2.2 Heuristic Move Priors During Node Expansion
In `umcts_v2_bot.rs`, unexpanded moves are generated without domain prioritization. When a node has 81 legal moves, poor moves may be expanded before obvious tactical winning moves.
- **Challenge**: Pass `&UltimateTicTacToeBoard` into node construction and score candidate moves with a fast, zero-allocation static evaluator (`score_move_quick`). Sorting unexpanded moves guarantees that high-priority tactical moves (subboard win: +100, block: +60, center: +15) are expanded and explored first.

### 2.3 Opening Book Integration at Root
- **Challenge**: Directly query `OpeningBook::lookup(board, zobrist_constants)` at root without requiring non-existent board turn fields. If a master move exists, return it immediately in $< 1$ $\mu$s.
