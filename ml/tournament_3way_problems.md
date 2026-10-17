# 3-Way Tournament Technical Challenges & Problem Domain

The [`tournament_3way.py`](file:///home/festivity/agy/uttt-codingame/ml/tournament_3way.py) module addresses several complex synchronization, statistical rating, and heterogeneous engine execution challenges.

---

## 1. Heterogeneous Search Synchronization (MCTS vs. AlphaBeta)

### Problem
Neural network models evaluate positions efficiently in large parallel batches on GPUs via batched MCTS. Conversely, AlphaBeta minimax engines search sequential move sequences on CPUs via recursive alpha-beta pruning with transposition tables.
In a paired tournament:
- Game $2i$ starts with the Neural model to move.
- Game $2i+1$ starts with the AlphaBeta bot to move.
If games progress naively, half the games require GPU MCTS while the other half require CPU minimax at any given ply. Interleaving tiny CPU and GPU operations creates severe pipeline serialization stalls and low hardware utilization.

### Implementation & Solution
- **Phase-0 Phase Alignment**:
  Before entering the game loop, [`run_neural_vs_alphabeta_match`](file:///home/festivity/agy/uttt-codingame/ml/tournament_3way.py#L229-L364) isolates all odd games ($2i+1$) and runs an initial AlphaBeta move on them:
  ```python
  odd_indices = [2 * i + 1 for i in range(num_pairs)]
  odd_histories = [game_histories[g] for g in odd_indices]
  first_ab_moves = worker.choose_alphabeta_moves(odd_histories, time_limit_ms=alphabeta_ms)
  worker.apply_moves_batch(odd_indices, first_ab_moves)
  ```
- **Synchronized Lockstep Batching**:
  Because odd games have advanced by 1 ply, *every active game on the board is now the Neural model's turn*.
  The search cleanly bifurcates into two distinct, fully saturated phases per ply:
  1. **Phase 1 (All Neural)**: All active games run batched GPU MCTS simultaneously. Moves are recorded and applied.
  2. **Phase 2 (All AlphaBeta)**: All active games are now AlphaBeta's turn. All active histories are dispatched to the Rust Rayon minimax thread pool in a single batch.
- This phase alignment maximizes GPU saturation during Phase 1 and utilizes all available CPU cores during Phase 2.

---

## 2. Global Joint Rating Estimation via Bradley-Terry MM

### Problem
Independent pairwise Elo calculations can lead to cyclical inconsistencies (e.g. $A > B$, $B > C$, $C > A$) and fail to incorporate multi-competitor network effects. Moreover, rating shifts must be calibrated against an established anchor to avoid unbounded score inflation or compression across generations.

### Implementation & Solution
- [`solve_bradley_terry`](file:///home/festivity/agy/uttt-codingame/ml/tournament_3way.py#L58-L111) solves the joint maximum likelihood Bradley-Terry model across the entire round-robin tournament using the Minorization-Maximization (MM) algorithm.
- Draws are handled symmetrically by crediting each competitor with 0.5 wins ($W_{ij} = \text{wins}_{ij} + 0.5 \cdot \text{draws}_{ij}$).
- The algorithm anchors the resulting latent skill distribution to `AlphaZeroResNet = 1906.4`, scaling all competitors relative to a fixed empirical standard.

---

## 3. Position Bias Elimination & Paired Mirror Tracking

### Problem
Ultimate Tic-Tac-Toe openings can possess hidden forced winning or drawing lines. Evaluating bots on arbitrary positions might reward a weaker engine simply because it received more favorable opening positions.

### Implementation & Solution
- **Exact Positional Cancellation**: For every paired position $P_k$, two mirror games are executed ($2k$ and $2k+1$).
- **Symmetric Metric Decomposition**:
  Outcomes are decomposed into:
  - **Double Wins (2-0 / 0-2)**: Games where one engine won from *both* sides of position $P_k$, proving decisive tactical superiority irrespective of position evaluation.
  - **Symmetrical Splits (1-1)**: Games where the advantage belonged strictly to the first or second player from $P_k$, indicating opening bias rather than skill disparity.
- This decomposition provides statistical proof that rating advantages stem from engine strength rather than opening anomalies.

---

## 4. Invariants & Edge Cases

1. **Move History Tracking**: In neural-vs-alphabeta matches, `game_histories` tracks complete move paths in Python for every game slot, ensuring that AlphaBeta transposition table hashing and DF-PN move generation receive full, uncorrupted trajectories.
2. **Dynamic Game Termination**: Unfinished games are tracked per ply (`worker.get_game_status(g) == 0`). As games conclude, active batch sizes shrink dynamically, avoiding redundant compute on finished games.
3. **Argmax Action Selection**: All tournament games operate under `temperature=0.0` and `dirichlet_eps=0.0`, eliminating random noise to measure true competitive peak strength.
