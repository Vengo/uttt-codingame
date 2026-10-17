# Problems & Challenges: `uttt/benches/uttt_benches.rs`

This document details 9x9 Ultimate Tic-Tac-Toe micro-benchmark challenges in `uttt/benches/uttt_benches.rs`.

---

## 1. Challenges
- Benchmarking full 9x9 move generation and `make_move` / `unmake_move` cycles under both forced subboard constraints and free-choice turns.
- Profiling rollout throughput across thousands of playouts to measure nanoseconds per simulated turn.
