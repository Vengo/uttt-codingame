# Problems & Challenges: `uttt/src/bin/cg.rs`

This document details CodinGame referee I/O and protocol challenges in `uttt/src/bin/cg.rs`.

---

## 1. Challenges
- **Strict Turn Limits**: 1,000 ms on move 1, 100 ms on subsequent turns. Exceeding 100 ms results in an immediate timeout loss on the referee server.
- **Parsing Referee Grid**: Parsing opponent move inputs (`opponent_row opponent_col`) and legal move lists provided on standard input, reconstructing the internal `UltimateTicTacToeBoard` representation in < 0.1 ms.
