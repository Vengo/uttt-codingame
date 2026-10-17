# Problems & Challenges: `uttt/src/arena/tournament.rs`

This document details round-robin tournament execution challenges in `uttt/src/arena/tournament.rs`.

---

## 1. Challenges
- **Combinatorial Match Pairing**: Scheduling round-robin matches across $B$ bots with $G$ games per pair such that each pairing plays equal games as X and O.
- **Leaderboard Ranking Consistency**: Sorting bots by points (1 for win, 0.5 for draw), win percentage, and dynamic Elo updates without rating drift.
- **Result Serialization**: Exporting complete tournament cross-tables, head-to-head match matrices, and individual game records to `tournament_results.json`.
