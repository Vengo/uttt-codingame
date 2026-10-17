# Plans & Progress: `uttt/src/opening_book.rs`

This document details opening book implementation, binary schemas, and status in `uttt/src/opening_book.rs`.

---

## 1. Plan & Architecture
- **Frontier Deepening (`find_frontier`)**: Traverses existing book lines along Principal Variations up to depth 8 to find unexpanded leaves.
- **UCT Dominance Harvesting**: Only captures moves satisfying $N_1 \ge 1.35 \times N_2$ or $\ge 40\%$ plurality with visit floors ($N \ge 100$, child $N \ge 50$).
- **Binary File Format (`book.bin`)**:
  - Header (16 bytes): Magic `UTTTOB01`, count `u32`, 4-byte padding.
  - Entries (16 bytes per entry): `canonical_hash (u64)`, `move_index (u8)`, 3-byte pad, `visits (u32)`.
- **Runtime Lookup**: $O(1)$ lookup via memory-mapped table or `HashMap`, with move index inverted via `sym.inverse().transform_move_index(c_move)`.

---

## 2. Status
- 108,578 canonical positions loaded at `/my-zraid/uttt/book.bin`.
- Deepens steadily at 5–10 master moves per cycle during study daemon runs.
