# Plans & Progress: `uttt/src/game_solving.rs`

This document details the universal game abstractions implemented in `uttt/src/game_solving.rs`.

---

## 1. Design & Plan
- **`GameMove`**:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
  pub struct GameMove {
      pub x_to_move: bool,
      pub move_index: u8,
  }
  ```
- **`GameResult`**: Standardized outcomes: `P1Win`, `P2Win`, `Tie`, and `Unresolved`.
- **`AlphaBetaCompatible` Trait**: Exposes `get_legal_moves`, `make_move`, `unmake_move`, `get_win_state`, and evaluation primitives required by minimax and heuristic bots.

---

## 2. Status
- Shared across all bot implementations, game runners, and test suites.
