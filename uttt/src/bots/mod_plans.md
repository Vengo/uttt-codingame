# Plans & Progress: `uttt/src/bots/mod.rs`

This document details the bot trait and factory implementation in `uttt/src/bots/mod.rs`.

---

## 1. Plan & Architecture
- **`Bot` Trait**:
  ```rust
  pub trait Bot: Send + Sync {
      fn name(&self) -> &str;
      fn reset(&mut self) {}
      fn choose_move(
          &mut self,
          board: &UltimateTicTacToeBoard,
          legal_moves_lookup: &[Vec<GameMove>],
          zobrist_constants: &ZobristConstants,
          time_limit: Duration,
      ) -> GameMove;
  }
  ```
- **Factory Functions**: `available_bots() -> Vec<&'static str>` and `create_bot(name: &str) -> Option<Box<dyn Bot>>`.

---

## 2. Status
- Supports all 9 bot variants with normalized names and aliases.
