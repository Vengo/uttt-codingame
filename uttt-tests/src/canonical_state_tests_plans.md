# Plans & Progress: `uttt-tests/src/canonical_state_tests.rs`

This document details canonical test implementations in `uttt-tests/src/canonical_state_tests.rs`.

---

## 1. Test Suite Implementations
- `test_packed_state_size_and_alignment`: Confirms `std::mem::size_of::<PackedBoardState>() == 24` and `align_of == 8`.
- `test_tied_mask_guard_with_canonical_won_subboards`: Confirms `get_tied_mask()` strictly ignores `CANONICAL_X_WIN = 0x1FF`.
- `test_transposition_collapse_different_win_lines`: Asserts `board_a.to_packed_canonical() == board_b.to_packed_canonical()`.
- `test_dfpn_solver_with_packed_state_and_grandfather`: Confirms solver proves mate-in-1 and mate-in-2 refutations in microseconds.
- `test_systematic_egtb_tier9_and_tier8`: Confirms Tier 9 produces all 19,891 canonical terminal states in < 15 ms.

---

## 2. Status
- All 9 canonical test cases passing with 100% reliability.
