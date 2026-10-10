# Problems & Challenges: `uttt/src/game_solving.rs`

This document details universal game abstractions and move traits in `uttt/src/game_solving.rs`.

---

## 1. Challenges
- **Generic Trait Compatibility**: Designing a unified `AlphaBetaCompatible` trait that works seamlessly across 3x3 standard boards, 9x9 Ultimate boards, and abstract game models.
- **Compact Move Representation**: Packaging move indices ($0 \dots 80$) and turn booleans into a lightweight `GameMove` struct (2 bytes) that fits into registers without heap allocation.
