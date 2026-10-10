# Plans & Progress: `uttt/src/bin/cg.rs`

This document details the CodinGame platform entry point implementation in `uttt/src/bin/cg.rs`.

---

## 1. Plan & Architecture
- Standard I/O loop: Reads `opponent_row opponent_col` from `stdin`.
- Updates board state via `make_move`.
- Selects move using `create_bot("book_umcts_v2")` with a conservative 45 ms safety budget per turn.
- Prints `chosen_row chosen_col` to `stdout`.

---

## 2. Status
- Operational and ready for single-file submission.
