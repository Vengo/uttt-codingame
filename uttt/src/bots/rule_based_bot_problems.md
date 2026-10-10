# Problems & Challenges: `uttt/src/bots/rule_based_bot.rs`

This document details expert heuristic policy challenges in `uttt/src/bots/rule_based_bot.rs`.

---

## 1. Challenges
- **Tactical Priority Inversion**: Heuristics must strictly prioritize macro game wins over subboard wins, and subboard wins over blocking individual cell threats.
- **Avoiding Dangerous Free-Choice Gifts**: Moving into a square that directs the opponent to an already resolved subboard grants the opponent free choice across all active subboards—a catastrophic strategic blunder in many midgames.
- **Zero-Allocation Execution**: Scoring every legal move in < 10 microseconds to serve as an ultra-fast simulation rollout policy.
