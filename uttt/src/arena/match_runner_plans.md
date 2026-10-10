# Plans & Progress: `uttt/src/arena/match_runner.rs`

This document details two-bot match implementation in `uttt/src/arena/match_runner.rs`.

---

## 1. Plan & Runner Design
- **`MatchRunner`**:
  - `run(bot1_name, bot2_name, games, time_limit, ...) -> MatchResult`:
    Runs $N$ games, alternating X/O assignments on every game.
  - Aggregates `bot1_wins`, `bot2_wins`, `draws`, `bot1_avg_ms`, `bot2_avg_ms`.
  - Exports replay files to disk when configured.
- **CLI Command**: Exposed via `arena match --bot1 <b1> --bot2 <b2> --games <N> --time-ms <T>`.

---

## 2. Status
- Operational. Used for rapid regression testing between bot versions.
