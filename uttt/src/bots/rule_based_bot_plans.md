# Plans & Progress: `uttt/src/bots/rule_based_bot.rs`

This document details expert heuristic policy implementation and evaluation weights in `uttt/src/bots/rule_based_bot.rs`.

---

## 1. Plan & Heuristic Priority Pipeline
For each candidate legal move $m$, `evaluate_move` performs a single make/unmake cycle and computes:
1. **Immediate Game Win (+1,000,000)**: If `get_win_state()` produces a game win for the active player.
2. **Subboard Win Bonus (+500 to +1,000)**: `SUBBOARD_WEIGHTS[subboard] * 10`, where `SUBBOARD_WEIGHTS = [75, 50, 75, 50, 100, 50, 75, 50, 75]` (center = 100, corners = 75, edges = 50).
3. **Square Positional Value (+4 to +15)**: `SQUARE_WEIGHTS = [8, 4, 8, 4, 15, 4, 8, 4, 8]` (center = 15, corners = 8, edges = 4).
4. **Target Subboard Steering**:
   - Sending opponent to a completed board (granting them free choice): **-60 penalty**.
   - Sending opponent to a board where they have an immediate win: **-80 penalty**.
   - Sending opponent to a safe board: **+15 bonus**.
5. **Local Threat Interactions**:
   - Blocking opponent 2-in-a-row threat: **+45 bonus**.
   - Creating an open 2-in-a-row for ourselves: **+30 bonus**.

---

## 2. Status
- Consistently beats `random` with a 100% win rate in tournament benchmarks.
- Re-used in simplified zero-allocation form (`score_move_quick`) inside `heuristic_mcts`.
