# Problems & Challenges: `uttt-tests` Test Suite Crate

This document outlines the testing challenges, invariant verification needs, and coverage goals for `uttt-tests`.

---

## 1. Challenges in Verifying Complex Game Engines
- **Non-Deterministic Heuristics & Search Trees**: Verifying MCTS and minimax engines across deep randomized rollouts without introducing flaky tests.
- **Transposition & Symmetry Edge Cases**: Proving that dihedral symmetry operations and canonical collapses do not drop winning paths or corrupt legal moves.
- **Perft & Move-Generator Correctness**:
  - For standard 3x3 Tic-Tac-Toe, reference perft values must match exact combinatorial ground truths: **5,478** reachable positions, **549,946** move paths, and **255,168** terminal games.
  - For 9x9 Ultimate Tic-Tac-Toe, perft traversal across 7 plies must match the reference count of **4,116,682** visited positions in < 1 second.
- **Solver Soundness**: Verifying that proof numbers and Grandfather lookahead never disprove a winning line or prove an invalid win.

---

## 2. Cross-References
- Root Architecture: [`/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md) and [`/PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- Test Suite Plans: [`/uttt-tests/PLANS.md`](file:///home/festivity/agy/uttt-codingame/uttt-tests/PLANS.md)
- In-Depth Technical Specification: [`/documentation/problems_and_challenges.md`](file:///home/festivity/agy/uttt-codingame/documentation/problems_and_challenges.md)
