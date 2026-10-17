# Problems & Challenges: `uttt/src/bots/original_umcts_bot.rs`

This document details legacy MCTS implementation challenges in `uttt/src/bots/original_umcts_bot.rs`.

---

## 1. Challenges
- **Legacy Codebase Preservation**: Retaining the original repository's MCTS search implementation for historical benchmarking, regression tracking, and profiling comparisons against modern engine refactors.
- **Transposition Cache Sizing**: Managing hash table capacity (`UmctsNode`) to avoid memory pressure during extended search runs.
