# Plans & Progress: `uttt/src/bots` Bot Ecosystem

This document outlines the architecture, search designs, heuristic scoring tables, and verification status of the bot implementations in `uttt/src/bots`.

---

## 1. Bot Roster & Architecture Map

1. **`random`** ([`random_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/random_bot.rs)):
   - Baseline bot choosing uniformly among legal moves via `SmallRng`. Serves as the ground baseline in tournaments.

2. **`rule_based`** ([`rule_based_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/rule_based_bot.rs)):
   - Expert static heuristic policy. Evaluates candidate moves using hardcoded weights:
     - `SUBBOARD_WEIGHTS = [75, 50, 75, 50, 100, 50, 75, 50, 75]` (center = 100, corners = 75, edges = 50).
     - `SQUARE_WEIGHTS = [8, 4, 8, 4, 15, 4, 8, 4, 8]`.
     - Game win: 1,000,000.
     - Subboard win: `weight * 10`.
     - Threat block: 45.
     - Avoid giving opponent free choice: -60 penalty.

3. **`alphabeta`** ([`alphabeta_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs)):
   - Preserved reference baseline bot. Minimax with alpha-beta pruning and fixed-duration iterative deepening.
   - Evaluates subboard control (center = 300, corners = 200, edges = 150) and line threats (1-in-row = 40, 2-in-row = 500, win = 10,000).
   - Caching via `HashMap<u64, TTEntry>` (unconditional overwrite).

4. **`alphabeta_v2`** ([`alphabeta_v2_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_v2_bot.rs)):
   - High-performance minimax evolution:
     - In-tree EGTB probing at $\ge 4$ resolved subboards with instant forced win/loss pruning.
     - Depth-preferred transposition table replacement preserving deep evaluations.
     - Killer move heuristics (2 killer moves per ply) for faster beta cutoffs.
     - Root opening book lookup in $< 1$ $\mu$s.

5. **`mcts`** ([`mcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/mcts_bot.rs)):
   - Classic Monte Carlo Tree Search using UCT ($C = \sqrt{2}$).
   - Allocates tree on heap per turn; rollouts run uniform random moves capped at 60 plies.
   - Unfinished rollouts evaluate as $0.5$ (draw) to penalize non-decisive play.

6. **`heuristic_mcts`** ([`heuristic_mcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/heuristic_mcts_bot.rs)):
   - MCTS combining UCT tree selection with an $\epsilon$-greedy rollout loop ($70\%$ best local score via zero-allocation `score_move_quick`, $30\%$ uniform random) capped at 50 plies.
   - Bypasses full `RuleBasedBot` struct instantiations during rollouts to eliminate heap allocation overhead.

7. **`original_umcts`** ([`original_umcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/original_umcts_bot.rs)):
   - Legacy UMCTS search engine preserved from the original codebase. Maintains a persistent `HashMap<u64, UmctsNode>` across turns without proof-number bounds.

8. **`umcts_v2`** ([`umcts_v2_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v2_bot.rs)):
   - Preserved reference MCTS-Solver baseline featuring single-node expansion per iteration, backward bound propagation (`ProvenWin`, `ProvenLoss`, `ProvenDraw`), and high-density cache-local node stores.

9. **`umcts_v3`** ([`umcts_v3_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v3_bot.rs)):
   - Advanced MCTS-Solver featuring:
     - In-tree EGTB probing at $\ge 4$ resolved subboards with direct proof backpropagation (`ProvenWin/Loss/Draw`), pruning subtrees without rollouts.
     - Static heuristic move expansion priors (subboard captures, threat blocks, center squares).
     - Root opening book lookup in $< 1$ $\mu$s.

10. **`book_umcts_v2`** ([`book_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/book_bot.rs)):
    - Wraps `umcts_v2` with instant D4 Opening Book lookups (< 1 $\mu$s) from `OpeningBook`.

11. **`book_alphabeta`** ([`book_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/book_bot.rs)):
    - Wraps `alphabeta` with instant D4 Opening Book lookups (< 1 $\mu$s) from `OpeningBook`.

---

## 2. Experimental Verification Plan
- **Baseline Anchoring**: `alphabeta` and `umcts_v2` are kept intact as benchmark references.
- **Statistical Significance**: All bot evaluations are executed with Glicko-2 Bayesian rating tracking until Rating Deviation $RD < 35$ against the `alphabeta` baseline.
