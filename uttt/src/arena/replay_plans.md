# Plans & Progress: `uttt/src/arena/replay.rs`

This document details game replay serialization and visualization implementation in `uttt/src/arena/replay.rs`.

---

## 1. Plan & Data Schema
- **`GameReplay`**:
  ```rust
  pub struct GameReplay {
      pub game_id: String,
      pub player_x: String,
      pub player_o: String,
      pub result: String,
      pub total_turns: usize,
      pub total_time_ms: f64,
      pub moves: Vec<MoveRecord>,
  }
  ```
- **`MoveRecord`**: Stores turn, player, move index, row/col coordinates, subboard index, sub-square index, duration in ms, and ASCII board representation.
- **`ReplayExporter`**: Exports formatted JSON replays and generates `replay_viewer.html` containing an embedded interactive HTML5 canvas player with playback controls (play, pause, step forward/backward).

---

## 2. Status
- Verified by `test_replay_serialization`.
- Integrated with the web studio server.
