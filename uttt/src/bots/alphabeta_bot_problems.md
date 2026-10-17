# Problems & Challenges: `uttt/src/bots/alphabeta_bot.rs`

This document details minimax search, transposition caching, and move ordering challenges in `uttt/src/bots/alphabeta_bot.rs`.

---

## 1. Challenges

### 1.1 Transposition Table Replacement Policy
The transposition table is backed by `HashMap<u64, TTEntry>`. In the current implementation, insertions unconditionally overwrite previous entries (`self.tt.insert(hash, ...)`). During iterative deepening, later iterations search deeper, but shallower branches can still overwrite deep PV evaluations if encountered later in the search tree. A depth-preferred replacement policy (preserving entries where `existing.depth > new_depth`) is an active planned refinement.

### 1.2 Iterative Deepening Time Budgeting
`AlphaBetaBot` evaluates depths progressively ($1, 2, 3 \dots$) until the per-move deadline (`Instant::now() >= deadline`). A key challenge is returning immediately when a deadline expires (`*time_exceeded = true`) without accepting a corrupted partially evaluated depth score.

### 1.3 Move Ordering Overhead vs. Pruning Efficiency
Alpha-beta pruning efficiency depends heavily on exploring the principal variation (PV) move first. Move ordering is performed via internal `order_moves(&mut moves, tt_best, board)`, sorting candidate moves by PV move priority, cell positional values (center = 15, corners = 8, edges = 4), and opponent direction penalties (+30). The challenge is balancing the sorting overhead against the branch reduction achieved.
