# Problems & Challenges: `uttt/src/bots` Bot Ecosystem

This document details the search, heuristic, and decision-making challenges across the bot ecosystem in `uttt/src/bots`.

---

## 1. Challenges & Algorithmic Complexities

### 1.1 Strict Competitive Time Constraints
Competitive environments (e.g. CodinGame) grant 1,000 ms on move 1, but only 100 ms per subsequent move (with 50 ms recommended safety margin). Locally and in tournament play, bots are invoked with explicit `Duration` budgets (typically 20 ms – 100 ms). Every bot must implement responsive polling of elapsed time or time-bounded search loops to return a legal move before timing out, resulting in immediate forfeit.

### 1.2 Search Paradigm Diversity & Memory Lifecycles
The bot suite provides a wide spectrum of computational complexities and memory lifecycles:
- **Baseline Heuristics**: Uniform random move selection ([`random_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/random_bot.rs)) and single-pass static greedy evaluation ([`rule_based_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/rule_based_bot.rs)).
- **Classic MCTS (`mcts_bot.rs`)**: Dynamically allocates node trees on the heap per turn, running uniform random rollouts capped at 60 plies, clearing the entire tree after every move.
- **Original UMCTS (`original_umcts_bot.rs`)**: Preserves a persistent `HashMap<u64, UmctsNode>` across turns, running legacy rollout loops without proof-number bounds.
- **Modern MCTS-Solver (`umcts_v2_bot.rs`)**: Single-node expansion per iteration with backward bound propagation (`ProvenWin`, `ProvenLoss`, `ProvenDraw`) and high-density cache-local node arenas.
- **Iterative Deepening Minimax (`alphabeta_bot.rs`)**: Fixed-time iterative deepening minimax using alpha-beta pruning, quiescent evaluations, and transposition tables.

### 1.3 Transposition Table Replacement & Hash Collisions
In [`alphabeta_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs), transposition caching is maintained via a `HashMap<u64, TTEntry>`. In the current implementation, entries are unconditionally updated upon new evaluation. Because shallower searches can overwrite entries from deeper iterations, implementing a strict depth-preferred replacement scheme is an ongoing challenge to preserve deep variation quality.

### 1.4 Rollout Horizon Truncation & Non-Decisive Evaluation
In MCTS rollouts, games can drag on for dozens of moves without a resolution.
- [`mcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/mcts_bot.rs) caps rollouts at 60 plies.
- [`heuristic_mcts_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/heuristic_mcts_bot.rs) caps rollouts at 50 plies.
When a rollout reaches its ply cutoff without a winner, the result is scored as `0.5` (draw). This deliberate penalty discourages the search engine from steering into protracted, non-decisive positions where winning advantages cannot be forced.

### 1.5 Move Ordering vs. Engine Lookup Independence
In [`alphabeta_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs), move sorting relies on a dedicated `order_moves()` routine rather than the precomputed `_legal_moves_lookup` slice. It prioritizes the PV move retrieved from the transposition table, followed by subboard center and corners, while penalizing moves that direct the opponent into open free-choice boards.

### 1.6 Knowledge Store Integration: Book & EGTB Scope
[`BookBot`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/book_bot.rs) wraps search engines ([`UmctsV2Bot`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v2_bot.rs) or [`AlphaBetaBot`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/alphabeta_bot.rs)) with opening book lookups at ply 0. While the endgame tablebase ([`EndgameTablebase`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs)) is fully functional and probed via `EgtbDatabase::probe()`, direct in-tree tablebase probing within the inner search loops of MCTS and Minimax remains a planned integration task.
