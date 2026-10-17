# Plans & Progress: `uttt/src/canonical.rs`

This document details the canonical representation solutions in `uttt/src/canonical.rs`.

---

## 1. Design & Plan
- **Canonical Bitboard Constants (Row-Major $0 \dots 8$)**:
  - `CANONICAL_X_WIN = 0x1FF`: All 9 cells marked as X, loser's bitboard zeroed out (`obb = 0`).
  - `CANONICAL_O_WIN = 0x1FF`: All 9 cells marked as O, loser's bitboard zeroed out (`xbb = 0`).
  - `CANONICAL_TIE_X = 0x0E3` (`0b011100011`, 5 cells) and `CANONICAL_TIE_O = 0x11C` (`0b100011100`, 4 cells): Disjoint valid tie pattern filling all 9 cells with zero 3-in-a-row lines in row-major indexing (cell $i = \text{row } i/3, \text{col } i\%3$).
- **Lifecycle & Execution**:
  - `normalize_directed_subboard(&mut UltimateTicTacToeBoard)` normalizes `last_super_index` to `FREE_CHOICE_INDEX = 9` whenever pointing to a resolved subboard.
  - Executed automatically inside `canonicalize_board(&mut board)`, which is called before state packing, EGTB probing, and transposition table insertion.
- **Piece Parity Safety**:
  - `board.x_to_move` strictly stores the turn flag, ensuring popcount increases never skew turn order. Move undoing uses `UndoState`, never unpacked canonical history.

---

## 2. Status
- Fully implemented, unit tested (`test_canonical_tie_pattern_validity`), and actively utilized across the engine.
