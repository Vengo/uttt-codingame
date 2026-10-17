# Plans & Progress: `uttt/src/bots/book_bot.rs`

This document details opening book wrapper implementation in `uttt/src/bots/book_bot.rs`.

---

## 1. Plan & Architecture
- **`BookBot` Wrapper**:
  ```rust
  pub struct BookBot {
      pub inner: Box<dyn Bot>,
      pub book: OpeningBook,
      pub name: String,
  }
  ```
- **Move Selection Pipeline**:
  1. Checks `self.book.lookup(board, zobrist_constants)` using $D_4$ symmetry canonical matching.
  2. If present and legal, returns the inverted actual board move immediately (< 1 $\mu$s).
  3. Otherwise, delegates to `self.inner.choose_move(...)`.
- **Variants**:
  - `book_umcts_v2`: Wraps `UmctsV2Bot`.
  - `book_alphabeta`: Wraps `AlphaBetaBot`.
- **Planned Enhancements**:
  - Direct integration of `EgtbDatabase::probe()` into the wrapper to provide instant tablebase endgame moves before invoking the inner search bot.

---

## 2. Status
- Operational. Top contenders in the arena leaderboard with instant opening moves.
