# Plans & Progress: `uttt/src/bots/alphabeta_bot.rs`

This document details minimax search, evaluation weights, and pruning implementations in `uttt/src/bots/alphabeta_bot.rs`.

---

## 1. Plan & Search Architecture

### 1.1 Iterative Deepening Framework
- Dynamically increases search depth ($1, 2, 3 \dots$) while `Instant::now() < deadline`.
- If time expires during a search depth, the partially searched iteration is discarded and the best move from the last fully completed depth is returned.

### 1.2 Move Ordering Pipeline
Rather than relying on generic engine tables, `AlphaBetaBot` implements custom move ordering via `order_moves`:
1. **Transposition Table Best Move**: Assigned top priority (0).
2. **Cell Positional Values**: Center cell (10), Corner cells (20), Edge cells (30).
3. **Opponent Direction Penalty**: Moves that send the opponent to an already resolved subboard (granting free choice) receive a $+30$ penalty.

### 1.3 Static Evaluation Function
- **Terminal States**: Win = $\pm 100,000$, Tie = 0.
- **Superboard Ownership**: `SUBBOARD_VALUES = [200, 150, 200, 150, 300, 150, 200, 150, 200]` (Center subboard = 300, corners = 200, edges = 150).
- **Superboard Winning Threats**:
  - 1-in-a-row: $+40$ / $-40$.
  - 2-in-a-row (open line threat): $+500$ / $-500$.
  - 3-in-a-row: $+10,000$ / $-10,000$.
- **Local Subboard Control**:
  - Center cell: $+15$ / $-15$.
  - Corner cells: $+8$ / $-8$.
  - Edge cells: $+4$ / $-4$.
  - Local 2-in-a-row threats: $+20$ / $-20$.

### 1.4 Transposition Table
- Backed by `HashMap<u64, TTEntry>` initialized with 1,000,000 capacity.
- Records `{ depth, score, flag, best_move }` with `TTFlag` (`Exact`, `LowerBound`, `UpperBound`).
- Currently unconditionally replaces entries; depth-preferred replacement is planned.

---

## 2. Status
- Operational. Reaches search depths 4–8 in practical play within 40 ms budgets.
