# Problems & Challenges: `uttt/src/dfpn` Depth-First Proof-Number Solver

This document outlines the theoretical hurdles and algorithmic constraints in `uttt/src/dfpn`.

---

## 1. Challenges
- **AND/OR Game-Tree Proof Search**: Solving whether a player can force a win in complex tactical positions with wide branching.
- **Three-Outcome Handling in Classical DF-PN**: DF-PN was formulated for two-outcome games. UTTT has Win, Loss, and Draw. A sound solver must treat draws as non-wins for the attacker ($pn = \infty, dn = 0$).
- **Transposition Table Hash Collisions**: False transposition hits corrupt proof numbers. The transposition table must use exact zero-collision keys (`PackedBoardState`).
- **Deep Recursive Thrashing without Immediate Refutation**: Candidate moves that allow an immediate winning counter-move by the defender must be refuted in 2 plies (Grandfather lookahead) across all candidate moves prior to deep descent.
- **Path-Cycle Oscillations**: Cyclic movement permutations in repeated free-choice positions must be detected and assigned non-win values ($pn = \infty, dn = 0$).
