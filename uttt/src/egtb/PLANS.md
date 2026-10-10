# Plans & Progress: `uttt/src/egtb` Endgame Tablebase Crate Module

This document details the systematic tier generation and probing architecture in `uttt/src/egtb`.

---

## 1. Architectural Design & Plan
- **Tier Decomposition**:
  - **Tier 9 (9 resolved subboards)**: $3^9 = 19,683$ macro configurations $\times 2$ turns $= 39,366$ raw states. Evaluates superboard win lines in $O(1)$. Due to asymmetric micro-tie patterns, macro symmetry degeneracy is partially preserved, producing **19,891 canonical terminal states** generated in < 15 ms.
  - **Tier 8 (8 resolved subboards, 1 active subboard)**: Iterates over the 9 active subboards $k \in \{0, 1, 4, 2, 3, 5, 6, 7, 8\}$ (corner, edge, center symmetry representatives) and all $3^8 = 6,561$ macro combinations of the other 8 subboards. Solves curated micro mark patterns in $k$ (empty, 1 mark, 2 marks, 3 marks) via Grandfather DF-PN at > 10,000 pos/sec.
  - **Tier 7 (7 resolved subboards, 2 active subboards)**: Iterates over distinct pairs of open subboards $(j, k)$ and all $3^7 = 2,187$ macro combinations. Implemented in `generate_tier_7` (`egtb/mod.rs:422`) and actively solved during extended daemon cycles.
- **Tablebase Binary Schema (`egtb_4plus.bin`)**:
  - Header (16 bytes, Little-Endian): 8-byte ASCII magic `UTTTEG01` + 8-byte `u64` entry count.
  - Records (10 bytes each, sorted by `canonical_hash`):
    - `canonical_hash: u64` (8 bytes, Little-Endian).
    - `outcome: u8` (1 byte: `0=Draw`, `1=P1Win` (Player X), `2=P2Win` (Player O)).
    - `plies_to_mate: u8` (1 byte, `0` for already-terminal Tier 9 states).
- **Zero-Copy Probing (`EgtbDatabase::probe`)**:
  Binary search directly against the memory-mapped byte slice `&mmap[16..16 + count * 10]` taking $\approx 15$ comparisons with zero heap allocations.
- **Search Integration**:
  - Alpha-Beta: Returns exact terminal cutoff score ($\pm 10,000$ or $0$).
  - MCTS: Truncates rollout simulation, backpropagates $1.0/0.0/0.5$, and marks node `ProofStatus`.

---

## 2. Status & Verification
- 29,611 verified solved positions stored in `/my-zraid/uttt/egtb/egtb_4plus.bin` (fallback to `egtb_4plus.bin` locally).
- Probing verified by unit test `test_systematic_egtb_tier9_and_tier8`.
