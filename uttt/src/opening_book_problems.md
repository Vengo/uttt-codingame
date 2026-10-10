# Problems & Challenges: `uttt/src/opening_book.rs`

This document details opening book generation and retrieval challenges in `uttt/src/opening_book.rs`.

---

## 1. Challenges
- **Stalling vs. Exponential Bloat**: Avoiding shallow stalls while preventing runaway growth from noisy MCTS nodes.
- **Statistical Move Dominance**: Distinguishing genuine master moves from exploratory MCTS noise.
- **Fast Zero-Allocation Lookup**: Ensuring opening moves are returned in $< 1\,\mu\text{s}$ during competitive play without allocating memory.
- **Symmetry Normalization**: Canonicalizing queries under $D_4$ dihedral symmetry and inverting the recommended move index back to actual board orientation.
