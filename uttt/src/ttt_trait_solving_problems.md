# Problems & Challenges: `uttt/src/ttt_trait_solving.rs`

This document details the single 3x3 Tic-Tac-Toe subboard solving challenges in `uttt/src/ttt_trait_solving.rs`.

---

## 1. Challenges
- **Precomputed Move Ordering & Successors**: Generating all legal successor states and optimal move orders for standard 3x3 boards at startup to eliminate runtime allocations.
- **Lookup Table Generation**: Efficiently caching the 5,478 reachable single-board Tic-Tac-Toe positions and mapping them to compact IDs (`TicTacToeBoardId`).
- **Bitboard Masks**: Rapidly evaluating 3-in-a-row wins using 8 precomputed bitmasks (`RESULT_MASKS = [7, 56, 448, 73, 146, 292, 84, 273]`).
