# Problems & Challenges: `uttt/src/bots/alphabeta_v2_bot.rs`

This document details the search, pruning, caching, and endgame evaluation challenges for `AlphaBetaV2Bot` in `uttt/src/bots/alphabeta_v2_bot.rs`.

---

## 1. Context & Baseline Constraints
- **Preservation of Baseline**: The original `AlphaBetaBot` in [`alphabeta_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs) is strictly preserved "as is" as the fixed reference baseline.
- **Goal of `alphabeta_v2`**: Significantly surpass the baseline in playing strength (targeting $> +70$ Elo against `AlphaBetaBot` with $95\%$ credible lower bound $> 1500.0$) within identical per-move time budgets (20 ms – 100 ms).

---

## 2. Core Technical Challenges

### 2.1 Depth-Preferred Transposition Table Replacement
In `alphabeta_bot.rs`, `HashMap<u64, TTEntry>` unconditionally overwrites entries on every insertion. During iterative deepening, shallower branches evaluated in later stages could overwrite deep PV evaluations from earlier iterations.
- **Challenge**: Implement a replacement scheme requiring `depth >= existing.depth`. A shallow exact evaluation (e.g. `depth = 1`) must never destroy a deep bound (e.g. `depth = 8`).
- **Memory Bounding**: Transposition cache should be cleared on `bot.reset()` or capped at 1,000,000 entries to prevent memory exhaustion over prolonged tournament play.
- **PV Move Retention**: When overwriting an existing entry with an evaluation that lacks a `best_move`, preserve `existing.best_move` to maintain move ordering quality.

### 2.2 In-Tree Endgame Tablebase (EGTB) Probing via `EgtbDatabase`
Positions with $\ge 4$ resolved subboards are often game-theoretically decided or near-terminal.
- **Challenge**: Seamlessly integrate `EgtbDatabase::probe(&self, board: &UltimateTicTacToeBoard, zobrist_constants: &ZobristConstants)` inside `negamax`.
- **Minimizing Probe Overhead**: Probing involves 8-fold dihedral symmetry canonicalization and memory-mapped binary search. To avoid redundant probes, check the transposition table first; only query `EgtbDatabase` if the state is not already cached, and cache the probe outcome with `depth = 255`.
- **Distance-to-Mate Scoring**:
  - Proven Win: Return $+100,000 - \text{ply}$ to incentivize the fastest forced win.
  - Proven Loss: Return $-100,000 + \text{ply}$ to delay defeat.
  - Proven Draw: Return $0$.

### 2.3 Search Ply Tracking & Killer Move Heuristics
- **Challenge**: `negamax` in `alphabeta_bot.rs` did not track distance from search root. `AlphaBetaV2Bot` must track `ply: usize` (bounded by `MAX_PLY = 81`).
- **Killer Move Storage**: Maintain `killers: [[Option<GameMove>; 2]; MAX_PLY]`. When a quiet move causes an alpha-beta cutoff (`score >= beta`), record it in `killers[ply]`.
- **Subboard Legality Guard**: In Ultimate Tic-Tac-Toe, candidate killer moves recorded at the same ply in another subtree may direct into a different subboard. Move ordering must verify that killer moves are strictly members of the currently legal move list before prioritizing them.

### 2.4 Opening Book Interception
- **Challenge**: Query `OpeningBook::lookup(board, zobrist_constants)` directly at root without relying on nonexistent board turn fields. If a master line move is returned, play immediately in $< 1$ $\mu$s.
