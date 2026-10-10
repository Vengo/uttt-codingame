# Plans & Progress: `uttt-tests` Test Suite Crate

This document outlines the testing strategy, test harness structure, and status for `uttt-tests`.

---

## 1. Test Harness Strategy & Suite Layout
- **Core Invariant Tests** ([`uttt-tests/src/canonical_state_tests.rs`](file:///home/festivity/agy/uttt-codingame/uttt-tests/src/canonical_state_tests.rs)):
  - Verify 167-bit packing fidelity and 24-byte alignment (`test_packed_state_size_and_alignment`).
  - Roundtrip deserialization across 100 randomized games (`test_pack_unpack_roundtrip_random_playouts`).
  - Guarded tie detection (`test_tied_mask_guard_with_canonical_won_subboards`).
  - Transposition collapse: distinct move sequences resolving a subboard collapse into identical `PackedBoardState` values (`test_transposition_collapse_different_win_lines`).
- **Perft & Combinatorial Ground Truth** ([`uttt-tests/src/lib.rs`](file:///home/festivity/agy/uttt-codingame/uttt-tests/src/lib.rs)):
  - 3x3 Perft: **5,478** positions, **549,946** paths, **255,168** terminal games verified.
  - 9x9 UTTT Perft: Depth 7 visits **4,116,682** states in ~950 ms.
- **Bot Regression & Ecosystem Validation**:
  - `test_all_bots_instantiate`: Verifies all 9 bot architectures (`random`, `rule_based`, `alphabeta`, `mcts`, `heuristic_mcts`, `original_umcts`, `umcts_v2`, `book_umcts_v2`, `book_alphabeta`) construct and choose legal moves without panic.
  - `test_rule_based_beats_random`: 100% win rate baseline.
  - `test_umcts_v2_beats_random`: 100% win rate baseline.
  - Head-to-head match runner with zero forfeits verified by `test_game_execution_no_forfeits`.

---

## 2. Status & Execution
- 55 integration and regression tests running in ~2.8 seconds.
- 100% passing with zero regressions.

---

## 3. Cross-References
- Root Architecture: [`/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md) and [`/PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- In-Depth Technical Specification: [`/documentation/plans_and_progress.md`](file:///home/festivity/agy/uttt-codingame/documentation/plans_and_progress.md)
