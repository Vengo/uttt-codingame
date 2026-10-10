# Plans & Progress: `uttt/src/arena/game.rs`

This document details game orchestration implementation in `uttt/src/arena/game.rs`.

---

## 1. Plan & Game Loop
- **`Game` Struct**:
  - `play(bot_x, bot_o, legal_moves, zobrist) -> (GameOutcome, GameStats)`: Executes turn-by-turn game loop until `board.get_win_state() != Unresolved` or max plies (81) reached.
  - Optional replay capture (`with_replay(bool)`): Stores `GameReplay` records with move indices, times, and ASCII board displays.
- **Forfeit Logic**: Validates move against `board.get_legal_moves()`. If invalid, immediately declares `Forfeit` for the offending player.

---

## 2. Status
- Verified by unit test `test_game_execution_no_forfeits`.
