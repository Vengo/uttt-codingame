# Problems & Challenges: `uttt/src/egtb/mod.rs`

This document details implementation-level challenges in `uttt/src/egtb/mod.rs`.

---

## 1. Challenges
- **Incremental Entry Merging & Deduplication**: Merging newly solved endgame states into an existing tablebase file without duplicating hashes.
- **Binary Search on Mapped Byte Slices**: Parsing 10-byte unaligned records from memory-mapped memory safely and fast.
- **Handling Contradictory Macro States**: Skipping mathematically impossible macro states (e.g. both players having won a 3-in-a-row simultaneously) during systematic tier enumeration.
