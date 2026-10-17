# Problems & Challenges: `uttt/src/symmetry.rs`

This document details the geometric symmetry challenges in `uttt/src/symmetry.rs`.

---

## 1. Challenges
- **Two-Level Spatial Transformation**: In Ultimate Tic-Tac-Toe, a symmetry operation in the dihedral group $D_4$ acts simultaneously on the macro $3 \times 3$ grid of subboards AND the micro $3 \times 3$ grid of squares within each subboard.
- **Move Inverse Mapping**: When a canonical move is retrieved from an opening book or tablebase, it must be inverted back to the actual game board's current orientation.
- **Target Subboard Transformation**: Transforming directed subboard indices correctly, while preserving free-choice invariants (`index = 9`).
- **Deterministic Canonical Representative**: Defining a strict, unambiguous ordering rule across all 8 symmetries so that symmetric boards always map to the exact same canonical state.
