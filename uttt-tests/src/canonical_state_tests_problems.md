# Problems & Challenges: `uttt-tests/src/canonical_state_tests.rs`

This document details canonical state invariant testing challenges in `uttt-tests/src/canonical_state_tests.rs`.

---

## 1. Challenges
- Formulating bit-exact unit tests that prove disparate winning lines (e.g. subboard won via top row vs. diagonal) collapse to identical `PackedBoardState` values.
- Validating that Grandfather 2-ply refutation lookahead correctly refutes candidate blunders without recursive thrashing.
- Ensuring zero regressions in systematic Tier 9 and Tier 8 tablebase generation.
