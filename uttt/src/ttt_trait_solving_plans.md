# Plans & Progress: `uttt/src/ttt_trait_solving.rs`

This document details the single 3x3 Tic-Tac-Toe engine implementation in `uttt/src/ttt_trait_solving.rs`.

---

## 1. Plan & Architecture
- **`TicTacToeEngine`**: Precomputes all reachable standard Tic-Tac-Toe boards, assigning each a compact `u16` ID.
- **Move Ordering Array**: Generates a static lookup table mapping each board ID to a list of legal moves ordered by minimax game-theoretic strength (winning moves first, drawing moves second).
- **Subboard Win Detection**: Fast branchless masking:
  ```rust
  RESULT_MASKS.iter().any(|&m| (bb & m) == m)
  ```

---

## 2. Status
- Table initialization executes in < 5 ms at engine startup.
- Powers rapid legal move ordering for both AlphaBeta and MCTS bots.
