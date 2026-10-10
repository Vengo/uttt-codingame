# Plans & Progress: `uttt` Core Engine Crate

This document outlines the architectural solutions, module structures, and implementation progress for the `uttt` crate.

---

## 1. Module Structure and Implementation Map
The `uttt` crate is organized into modular subsystems:
- `canonical`: Constant bitboards for resolved subboards (`0x1FF`, `0x0E3/0x11C`) and directed index normalization to index 9.
- `packed_state`: `PackedBoardState` (24 bytes, `[u64; 3]`), exact equality, zero hash collisions.
- `symmetry`: $D_4$ dihedral group transformations, canonical orientation selection via lexicographical `is_less`.
- `game_solving`: Universal traits for moves and game results.
- `ttt_trait_solving` & `uttt_trait_solving`: Bitboard operations, legal move generators, Zobrist hash tables, guarded tie masks (`get_tied_mask()`).
- `opening_book`: Memory-mapped opening book with UCT dominance filtering.
- `dfpn`: Depth-First Proof-Number solver with Grandfather 2-ply refutation lookahead.
- `egtb`: Systematic tiered endgame tablebases (Tiers 9, 8, 7).
- `bots`: 9 distinct bot architectures (`random`, `rule_based`, `alphabeta`, `mcts`, `heuristic_mcts`, `original_umcts`, `umcts_v2`, `book_umcts_v2`, `book_alphabeta`).
- `arena`: Tournament runner, Elo calculator, replay system, flamegraph profiler, and HTTP web studio.
- `bin`: CLI targets: `arena`, `cg`, `runlocal`.

---

## 2. Implementation Progress & Status
- **Bitboard Engine**: 100% operational with guarded `get_tied_mask()`.
- **Packed Representation**: Verified exact 167-bit bit-packing in `[u64; 3]`.
- **Symmetry & Canonicalization**: Fully integrated into MCTS, AlphaBeta, DF-PN, and EGTB.
- **Tablebases & Book**: 29,611 EGTB positions and 108,578 book entries active on NVMe ZFS storage.
- **Binary Targets**: Fully compiling in release mode with flamegraph profiling and tournament support.

---

## 3. Cross-References
- Root Architecture: [`/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md) and [`/PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- In-Depth Technical Specification: [`/documentation/plans_and_progress.md`](file:///home/festivity/agy/uttt-codingame/documentation/plans_and_progress.md)
