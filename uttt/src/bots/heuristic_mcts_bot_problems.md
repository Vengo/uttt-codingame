# Problems & Challenges: `uttt/src/bots/heuristic_mcts_bot.rs`

This document details hybrid heuristic MCTS challenges, rollout throughput, and horizon limits in `uttt/src/bots/heuristic_mcts_bot.rs`.

---

## 1. Challenges

### 1.1 Heuristic Rollout Overhead vs. Simulation Throughput
Pure random MCTS achieves hundreds of thousands of rollouts per second, but uniform random play often blunders obvious wins and tactical blocks. Injecting full `RuleBasedBot` decisions into rollouts would cause significant performance bottlenecks due to heap allocations and full move generation. The challenge is providing an $\epsilon$-greedy rollout using a lightweight, zero-allocation evaluator (`score_move_quick`) that maintains high throughput.

### 1.2 Rollout Depth Horizon and Unresolved Penalization
To prevent protracted rollouts in closed or blocked mid-game configurations, simulations are capped at 50 plies (`depth < 50`). If a simulation terminates with `GameResult::Unresolved`, it receives a reward of `0.5` (identical to a tie). A key challenge is ensuring that this draw evaluation properly penalizes non-decisive moves without disincentivizing defensive play in balanced endgames.
