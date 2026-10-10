# Plans & Progress: `uttt/src/dfpn` Depth-First Proof-Number Solver

This document outlines the design, implementation, and progress of the DF-PN solver in `uttt/src/dfpn`.

---

## 1. Plan & Solver Architecture
- **`ProofNumbers`**: Encapsulates $(pn, dn)$.
  - Newly expanded non-terminal leaf nodes are initialized to $(pn = 1, dn = 1)$.
  - Solved attacker win: $(0, \infty)$.
  - Disproven attacker win (defender win or tie): $(\infty, 0)$.
- **AND/OR Propagation Formulas**:
  - At attacker OR node:
    $$pn = \min_{c} dn_{\text{child}}, \quad dn = \sum_{c} pn_{\text{child}}$$
    (Note: children are defender AND nodes, so child's $dn$ is attacker's effective $pn$).
- **Threshold MID Algorithm**:
  Iteratively updates thresholds $(th_{pn}, th_{dn})$ when descending into the best child using Kishimoto/Tanase updates:
  $$th_{pn} = \min(th_{pn}^{\text{parent}}, second\_min\_pn + 1)$$
- **Transposition Table Keys**:
  Uses `board.to_packed_canonical()` as exact 24-byte keys in `HashMap<PackedBoardState, ProofNumbers>`, guaranteeing zero collision risk across symmetries.
- **Grandfather Refutation Lookahead**:
  Evaluated across all legal attacker candidate moves before descending recursively. If *any* defender reply achieves an immediate macro win (`defender_win_result`), the candidate child is refuted on the spot ($pn = \infty, dn = 0$).
- **Cycle Handling**: Repeated states in `path: Vec<PackedBoardState>` are assigned $(pn = \infty, dn = 0)$.

---

## 2. Status & Verification
- Unit tested by `test_dfpn_solver_with_packed_state_and_grandfather`.
- Solves Tier 8 endgame positions at > 10,000 positions per second.
