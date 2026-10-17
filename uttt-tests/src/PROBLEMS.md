# Problems & Challenges: `uttt-tests/src` Test Harness

This document details integration testing challenges, combinatorial perft verification, and invariant testing in `uttt-tests/src`.

---

## 1. Challenges & Invariant Verification Goals

### 1.1 Combinatorial Explosion in Perft Assertions
Perft (performance test) recursively walks the complete move tree from the root board to verify that move generation, execution, and undoing do not lose or fabricate states.
- 3x3 standard Tic-Tac-Toe perft visits 5,478 distinct states and 549,946 total leaf paths (255,168 complete games).
- 9x9 Ultimate Tic-Tac-Toe branches exponentially:
  - Depth 0: 1
  - Depth 1: 81
  - Depth 2: 720
  - Depth 3: 5,616
  - Depth 4: 43,184
  - Depth 5: 331,344
  - Depth 6: 2,544,144
  - Depth 7: 4,116,682 unique states.
Testing higher depths quickly exceeds CI execution time limits unless algorithms operate with nanosecond bitboard efficiency.

### 1.2 Invariant Verification across Bit-Packed States
Proving the mathematical soundness of 167-bit packed state representations requires rigorous property-based testing:
1. **Zero Information Loss**: Unpacking a packed board must yield identical bitboards (`xbb`, `obb`), turn flags, and macro win status across 100+ random playout games.
2. **Canonical Transposition Collapse**: Verifying that distinct paths to winning a subboard (e.g. top row `0b000_000_111` vs. diagonal `0b100_010_001`) collapse to the exact same canonical bit pattern (`0x1FF`).
3. **Directed Normalization Collapse**: Verifying that directing moves to won or tied subboards collapses `last_super_index` to `FREE_CHOICE_INDEX = 9`.
4. **Tie Mask Distinction**: Proving that canonical won boards (`0x1FF`) are never misclassified as subboard ties by `get_tied_mask()`.

### 1.3 Scope and Crate Organization
The test suite is partitioned between crate-level unit tests and integration tests:
- `uttt-tests` crate contains 55 integration tests (46 in `lib.rs` + 9 in `canonical_state_tests.rs`).
- `uttt` crate contains 4 internal unit tests in `uttt/src/lib.rs`.
- Total workspace test suite encompasses 59 tests passing in ~2.8 seconds.
