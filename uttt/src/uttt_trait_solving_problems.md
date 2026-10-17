# Problems & Challenges: `uttt/src/uttt_trait_solving.rs`

This document details the core 9x9 Ultimate Tic-Tac-Toe engine challenges in `uttt/src/uttt_trait_solving.rs`.

---

## 1. Challenges
- **Guarded Tie Detection**: Preventing canonical won subboards (`0x1FF`) from being falsely diagnosed as ties by requiring both full occupancy AND the absence of a win:
  ```rust
  let full = (self.xbb[i] | self.obb[i]) == 0x1FF;
  let not_won = ((self.x_supers | self.o_supers) & (1 << i)) == 0;
  ```
- **Fast Legal Move Generation**: Supporting fast move generation under both forced subboard constraints and free-choice turns across 81 cells.
- **Incremental Zobrist Hashing**: Maintaining a 64-bit Zobrist hash updated via XOR on `make_move` and restored on `unmake_move`.
- **Termination & CodinGame Rules Compliance**: Evaluating immediate 3-in-a-row superboard wins, as well as subboard majority tie-breakers when all 9 subboards are closed without 3-in-a-row.
