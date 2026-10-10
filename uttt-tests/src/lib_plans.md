# Plans & Progress: `uttt-tests/src/lib.rs`

This document details legacy test cases and bot validation implementations in `uttt-tests/src/lib.rs`.

---

## 1. Test Suite Layout
- `perft_positions`, `perft_paths`, `perft_games`: Ground-truth perft counts for 3x3 standard boards.
- `perft_positions` (UTTT): Traverses 7 plies of Ultimate Tic-Tac-Toe, asserting 4,116,682 positions visited.
- `test_all_bots_instantiate`: Verifies instantiation of all 9 bot architectures.
- `test_rule_based_beats_random` & `test_umcts_v2_beats_random`: Baseline win-rate assertions.

---

## 2. Status
- 46 integration test cases passing in ~2.7s.
