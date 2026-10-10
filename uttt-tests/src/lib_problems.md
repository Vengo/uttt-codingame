# Problems & Challenges: `uttt-tests/src/lib.rs`

This document details legacy test harness and perft verification challenges in `uttt-tests/src/lib.rs`.

---

## 1. Challenges
- **Combinatorial Perft Ground Truth**: Verifying that move generation matches standard perft counts (5,478 standard positions, 4,116,682 UTTT depth-7 positions) without memory leaks.
- **Bot Regression Validation**: Ensuring changes to move generators or heuristics do not cause bots to make illegal moves or forfeit matches.
