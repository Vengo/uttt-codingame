# Problems & Challenges: `uttt/src/bots/umcts_v2_bot.rs`

This document details MCTS-Solver and UCT tree harvesting challenges in `uttt/src/bots/umcts_v2_bot.rs`.

---

## 1. Challenges
- **MCTS Blind Spots in Solved Positions**: Standard MCTS continues to simulate rollouts on nodes that are already provably won or lost, wasting critical compute budget.
- **Proof Propagation in MCTS Trees**: Propagating game-theoretic proofs (`ProvenWin`, `ProvenLoss`, `ProvenDraw`) up the search tree: an OR node is proven if *any* child is a proven win; an AND node is proven if *all* children are proven wins.
- **Controlled Opening Book Harvesting**: Extracting master moves from the search tree without capturing noisy, unverified exploration branches.
- **UCT Dominance Margin**: Defining a statistical criterion ($N_1 \ge 1.35 \times N_2$ or $\ge 40\%$ plurality) to ensure only decisive candidate moves are harvested into opening books.
