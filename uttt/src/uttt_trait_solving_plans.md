# Plans & Progress: `uttt/src/uttt_trait_solving.rs`

This document details the 9x9 Ultimate Tic-Tac-Toe engine implementation in `uttt/src/uttt_trait_solving.rs`.

---

## 1. Architectural Design & Plan
- **`UltimateTicTacToeBoard`**:
  - `xbb: [u16; 9]`: Bitboards for X marks in each of the 9 subboards.
  - `obb: [u16; 9]`: Bitboards for O marks in each of the 9 subboards.
  - `x_supers: u16`: 9-bit bitmask of subboards won by X.
  - `o_supers: u16`: 9-bit bitmask of subboards won by O.
  - `last_super_index: u8`: Subboard where active player must move ($0 \dots 8$), or 9 for free choice.
  - `x_to_move: bool`: Turn flag.
  - `hash_state: u64`: Incremental Zobrist hash.
- **Move Generation**:
  - If `last_super_index < 9` and subboard is not resolved: generates legal moves in that subboard.
  - Else: generates legal moves across all open squares in all unresolved subboards.
- **`make_move` & `unmake_move`**:
  - Fast bitwise mark placement, superboard line updates, and reversible undo via `UndoState`.
- **Guarded `get_tied_mask()`**:
  - Strictly excludes subboards present in `x_supers | o_supers`.

---

## 2. Status & Verification
- Passes 100% of move generation, perft, and playout tests.
- Core foundation for all search bots and tablebase generators.
