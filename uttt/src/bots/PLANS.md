# Plans & Progress: `uttt/src/bots` Bot Ecosystem

This document outlines the architecture, search designs, heuristic scoring tables, and verification status of the 9 bot implementations in `uttt/src/bots`.

---

## 1. Bot Roster & Architecture Map

1. **`random`** ([`random_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/random_bot.rs)):
   - Baseline bot choosing uniformly among legal moves via `SmallRng`. Serves as the ground baseline in tournaments.

2. **`rule_based`** ([`rule_based_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/rule_based_bot.rs)):
   - Expert static heuristic policy. Evaluates candidate moves using hardcoded weights:
     - `SUBBOARD_WEIGHTS = [3, 2, 3, 2, 4, 2, 3, 2, 3]` (center = 4, corners = 3, edges = 2).
     - `SQUARE_WEIGHTS = [3, 2, 3, 2, 4, 2, 3, 2, 3]`.
     - Game win: 1,000,000.
     - Subboard win: 10,000.
     - Threat block: 5,000.
     - Avoid giving opponent free choice: -3,000 penalty.

3. **`alphabeta`** ([`alphabeta_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs)):
   - Minimax with alpha-beta pruning and fixed-duration iterative deepening.
   - Evaluates subboard control (center = 40, corners = 30, edges = 20) and line threats (1-in-row = 40, 2-in-row = 500, win = 10,000).
   - Move ordering via internal `order_moves()` (TT PV move first, subboard center/corners, opponent direction check).
   - Caching via `HashMap<u64, TTEntry>` (currently unconditional overwrite; depth-preferred replacement planned).

4. **`mcts`** ([`mcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/mcts_bot.rs)):
   - Classic Monte Carlo Tree Search using UCT ($C = \sqrt{2}$).
   - Allocates tree on heap per turn; rollouts run uniform random moves capped at 60 plies.
   - Unfinished rollouts evaluate as $0.5$ (draw) to penalize non-decisive play.

5. **`heuristic_mcts`** ([`heuristic_mcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/heuristic_mcts_bot.rs)):
   - MCTS combining UCT tree selection with an $\epsilon$-greedy rollout loop ($70\%$ best local score via zero-allocation `score_move_quick`, $30\%$ uniform random) capped at 50 plies.
   - Bypasses full `RuleBasedBot` struct instantiations during rollouts to eliminate heap allocation overhead.

6. **`original_umcts`** ([`original_umcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/original_umcts_bot.rs)):
   - Legacy UMCTS search engine preserved from the original codebase. Maintains a persistent `HashMap<u64, UmctsNode>` across turns without proof-number bounds.

7. **`umcts_v2`** ([`umcts_v2_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v2_bot.rs)):
   - Refactored MCTS-Solver featuring single-node expansion per iteration, backward bound propagation (`ProvenWin`, `ProvenLoss`, `ProvenDraw`), and high-density cache-local node stores.

8. **`book_umcts_v2`** ([`book_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/book_bot.rs)):
   - Wraps `umcts_v2` with instant D4 Opening Book lookups (< 1 $\mu$s) from `OpeningBook`. Probes the book at ply 0; falls back to `umcts_v2` search when out of book.

9. **`book_alphabeta`** ([`book_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/book_bot.rs)):
   - Wraps `alphabeta` with instant D4 Opening Book lookups (< 1 $\mu$s) from `OpeningBook`. Falls back to iterative deepening minimax when out of book.

---

## 2. Planned Bot Enhancements
- **Direct In-Tree EGTB Probing**: Integrate `EgtbDatabase::probe()` into `UmctsV2Bot` node expansion and `AlphaBetaBot` leaf evaluation to instantly prune subtrees with $\ge 4$ resolved subboards.
- **Depth-Preferred TT Replacement**: Update `AlphaBetaBot` transposition table writes to preserve deeper search results when overwritten by shallower iterations.

---

## 3. Status & Verification
- All 9 bots instantiate cleanly and are selectable via CLI (`arena match`, `arena tournament`, `arena play`).
- Unit and integration tests verify baseline performance: `test_all_bots_instantiate`, `test_rule_based_beats_random`, and `test_umcts_v2_beats_random`.
