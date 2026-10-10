# Plans & Progress: `uttt/src/arena/tournament.rs`

This document details tournament execution implementation in `uttt/src/arena/tournament.rs`.

---

## 1. Plan & Data Models
- **`Tournament` Struct**:
  - `run(bot_names, legal_moves, zobrist) -> TournamentResult`:
    Executes all $\binom{B}{2}$ pairings. For each pair, runs $G$ games, alternating X/O turns on each game.
  - Updates Elo ratings for both players after each game using `EloCalculator`.
- **`LeaderboardEntry`**:
  Stores `bot_name`, `elo`, `points`, `wins`, `losses`, `draws`, `win_percentage`, and `avg_move_ms`.
- **Serialization**: Writes results to `tournament_results.json`.

---

## 2. Status
- Operational via `arena tournament [--games N] [--time-ms T] [--bots b1,b2,...]`.
