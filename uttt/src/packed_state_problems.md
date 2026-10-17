# Problems & Challenges: `uttt/src/packed_state.rs`

This document details state compression and zero-collision requirements in `uttt/src/packed_state.rs`.

---

## 1. Challenges
- **Size & Alignment**: Full board structs consume 64+ bytes, causing CPU cache lines (64 bytes) to spill over. A compact, 8-byte aligned representation is required.
- **Hash Collisions in Exact Solvers**: Traditional 64-bit Zobrist hashing has a collision risk that grows with the square root of evaluations (birthday paradox). A sound endgame solver (DF-PN) cannot tolerate false positive transposition collisions.
- **Dual-Layer Hashing Architecture**: Distinguishing the role of fast incremental Zobrist hashes (used by soft-time search bots where occasional collisions are harmless) from exact 167-bit bit-packed states (used by DF-PN and EGTB).
- **Exact State Completeness**: The state representation must encode all 81 X squares, 81 O squares, target subboard ($0 \dots 9$), and the player to move without losing data.
