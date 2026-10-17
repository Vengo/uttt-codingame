# Plans & Progress: `uttt/src/bots/random_bot.rs`

This document details baseline random move implementation in `uttt/src/bots/random_bot.rs`.

---

## 1. Plan & Architecture
- Uses `rand::rngs::SmallRng` with `SliceRandom::choose` to select uniformly from `board.get_legal_moves(legal_moves_lookup)`.
- Fallbacks to the first legal move if legal moves list is unexpectedly singleton.

---

## 2. Status
- Baseline bot against which smarter bots must achieve 100% win rates.
