# Problems & Challenges: `uttt/src/arena/game.rs`

This document details game execution and move validation challenges in `uttt/src/arena/game.rs`.

---

## 1. Challenges
- **Strict Turn Timeout Enforcement**: Enforcing time limits per move (e.g. 40 ms) and recording duration with microsecond precision.
- **Illegal Move Forfeit Handling**: Detecting when a bot returns an illegal move index or fails to respond, immediately forfeiting the game to the opponent.
- **Replay Serialization Coupling**: Recording per-move board states without slowing down fast headless tournament simulations.
