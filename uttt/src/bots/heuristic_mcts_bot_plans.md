# Plans & Progress: `uttt/src/bots/heuristic_mcts_bot.rs`

This document details hybrid heuristic MCTS implementation, rollout policy parameters, and scoring logic in `uttt/src/bots/heuristic_mcts_bot.rs`.

---

## 1. Plan & Architecture

### 1.1 In-Tree UCT Selection & Prioritized Expansion
- **Tree Policy**: Standard UCT formula ($C = \sqrt{2} \approx 1.41421356$) balancing exploitation (`wins / visits`) and exploration.
- **Sorted Expansion**: When expanding a node, `unexpanded_moves` are sorted in ascending order by `score_move_quick` so that highest-scoring tactical moves pop first.

### 1.2 Zero-Allocation $\epsilon$-Greedy Rollout Policy
- Rather than instantiating the full `RuleBasedBot` struct, rollouts use a fast inlined simulation loop:
  - **70% probability**: Selects the candidate move with the highest `score_move_quick` score:
    - Subboard win: $+100$
    - Subboard block (blocking opponent 2-in-a-row): $+60$
    - Center square bonus: $+15$
    - Completed board avoidance (avoiding free choice to opponent): $-20$
  - **30% probability**: Selects uniformly at random via `moves.choose(rng)`.
- **Simulation Limit**: Maximum 50 plies (`depth < 50`).
- **Evaluation Reward**:
  - Attacker win: `1.0`
  - Attacker loss: `0.0`
  - Board tie: `0.5`
  - Unresolved cutoff: `0.5` (penalizes non-decisive play).

---

## 2. Status
- Operational. Outperforms classical `mcts` and `rule_based` bots in tournament matchups.
