# Plans & Progress: `uttt/src/egtb/mod.rs`

This document details implementation details, persistence, and probing logic in `uttt/src/egtb/mod.rs`.

---

## 1. Implementation Architecture
- **`EgtbDatabase`**:
  - `load_mmap()`: Maps tablebase file into memory using `memmap2::Mmap`.
  - `probe(board, ...) -> Option<GameResult>`: Canonicalizes board, extracts `canonical_hash`, and executes binary search over `mmap`.
  - `generate_tier_9`: Generates all 19,891 terminal canonical states in < 15 ms.
  - `generate_tier_8`: Solves 8-resolved configurations via `DfPnSolver` in batches of 100–250.
  - `generate_tier_7`: Solves 7-resolved configurations via `DfPnSolver`.
  - `persist_entries`: Atomically sorts entries by hash and writes back to disk.

---

## 2. Status
- Tablebase contains 29,611 positions and integrates into bot evaluation routines.
