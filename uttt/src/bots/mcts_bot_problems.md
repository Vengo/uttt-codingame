# Problems & Challenges: `uttt/src/bots/mcts_bot.rs`

This document details classic Monte Carlo Tree Search challenges in `uttt/src/bots/mcts_bot.rs`.

---

## 1. Challenges
- **Shallow Simulation Noise**: In early and mid-game states, uniform random rollouts to terminal states produce noisy outcomes that fail to detect immediate 1-ply and 2-ply tactical blunders.
- **Tree Node Allocation Pressure**: Dynamically allocating child vectors during UCT expansion can strain the memory allocator during 10,000+ iteration bursts.
- **Root Move Selection**: Selecting the child move with the highest visit count ($N$) rather than highest average value ($Q$), ensuring statistical robustness against outlier rollout wins.
