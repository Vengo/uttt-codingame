# Plans & Progress: `uttt/src/bots/mcts_bot.rs`

This document details classic MCTS implementation in `uttt/src/bots/mcts_bot.rs`.

---

## 1. Plan & Search Loop
- **Four Phases of MCTS**:
  1. **Selection**: Recursively traverses children using the UCB1 formula:
     $$\text{UCT}(c) = \frac{W_c}{N_c} + C \sqrt{\frac{\ln N_p}{N_c}}, \quad C = \sqrt{2}$$
  2. **Expansion**: Adds one unvisited child to the tree.
  3. **Simulation**: Plays uniform random moves using `SmallRng` until terminal game result.
  4. **Backpropagation**: Propagates outcome (+1 for win, 0 for loss, 0.5 for tie) up the ancestral path.
- **Root Action**: Chooses $\arg\max_c N_c$.

---

## 2. Status
- Operational baseline MCTS engine for tournament benchmarking.
