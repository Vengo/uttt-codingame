# Problems & Challenges: `uttt/src/egtb` Endgame Tablebase Crate Module

This document details the endgame solving, storage, and retrieval challenges in `uttt/src/egtb`.

---

## 1. Challenges
- **Random Playout Discovery Inefficiency**: Simulating random games from move 0 hoping to hit late-game positions ($\ge 7$ resolved subboards) is mathematically intractable; random playouts terminate early or drift into chaotic non-terminal states.
- **Micro-Tie Asymmetry vs. Macro Symmetries**: While won subboards (`0x1FF`) possess full $D_4$ symmetry, valid 9-cell subboard ties (`0x0E3 / 0x11C`) have asymmetric micro configurations, preventing multi-tie macro boards from collapsing to pure macro equivalence classes under bit-exact minimization.
- **Zero-Copy Memory-Mapped Retrieval**: Tablebase queries during MCTS rollouts or AlphaBeta searches must execute in nanoseconds without heap allocation or disk I/O latency.
- **Dual-Layer Tablebase Indexing**: Using 64-bit canonical Zobrist hashes in sorted binary search slices for instant $O(1)$ disk/mmap table lookups while ensuring exact `PackedBoardState` fidelity during forward search expansion.
- **Atomic Persistence on ZFS Storage**: Safe, crash-resilient tablebase writing to `/my-zraid/uttt/egtb/egtb_4plus.bin` during continuous daemon runs, with automatic fallback to local storage if `/my-zraid/uttt` is unavailable.
