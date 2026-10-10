# Plans & Progress: `uttt-tests/src` Test Harness

This document details the test modules, invariant verifications, and progress in `uttt-tests/src`.

---

## 1. Test Modules & Suite Layout

### 1.1 Invariant Verification Suite ([`canonical_state_tests.rs`](file:///home/festivity/agy/uttt-codingame/uttt-tests/src/canonical_state_tests.rs))
- `test_packed_state_size_and_alignment`: Confirms `size_of::<PackedBoardState>() == 24` and `align_of == 8`.
- `test_tied_mask_guard_with_canonical_won_subboards`: Guarantees `0x1FF` won subboards are never classified as tied.
- `test_pack_unpack_roundtrip_random_playouts`: 100 randomized games verified for zero-loss roundtrip conversion.
- `test_canonical_invariance_across_random_playouts`: Verifies move legality, win state, and score invariance between raw and canonical boards.
- `test_transposition_collapse_different_win_lines`: Verifies distinct winning lines collapse to the exact same `PackedBoardState`.
- `test_directed_subboard_normalization_collapse`: Verifies moves directed to resolved subboards normalize `last_super_index` to 9.
- `test_dfpn_solver_with_packed_state_and_grandfather`: Tests DF-PN proof solving with 2-ply Grandfather refutation lookahead.
- `test_systematic_egtb_tier9_and_tier8`: Proves Tier 9 terminal generation (19,891 canonical states) and Tier 8 solving (+200 states).
- `test_opening_book_harvesting_and_growth`: Verifies UCT dominance filtering ($N_1 \ge 1.35 \times N_2$, depth $\le 8$).

### 1.2 Engine & Perft Verification Suite ([`lib.rs`](file:///home/festivity/agy/uttt-codingame/uttt-tests/src/lib.rs))
- **3x3 Standard Tic-Tac-Toe Perft**: 5,478 states, 549,946 paths, 255,168 complete games.
- **9x9 Ultimate Tic-Tac-Toe Perft**:
  - Depth 1: 81
  - Depth 2: 720
  - Depth 3: 5,616
  - Depth 4: 43,184
  - Depth 5: 331,344
  - Depth 6: 2,544,144
  - Depth 7: 4,116,682 unique states.
- **Bot Behavior Tests**: `test_all_bots_instantiate`, `test_rule_based_beats_random`, and `test_umcts_v2_beats_random`.

---

## 2. Test Execution & Status
- **Crate Breakdown**: 55 tests in `uttt-tests` (46 in `lib.rs` + 9 in `canonical_state_tests.rs`) + 4 unit tests in `uttt/src/lib.rs` = 59 total tests across workspace.
- **Execution Speed**: All 59 tests execute and pass in ~2.8 seconds (`cargo test --workspace`).
