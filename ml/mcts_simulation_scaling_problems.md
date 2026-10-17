# MCTS Simulation Scaling & Prior Bias Technical Challenges

The [`rust_mcts_worker.py`](file:///home/festivity/agy/uttt-codingame/ml/rust_mcts_worker.py), [`batched_mcts.py`](file:///home/festivity/agy/uttt-codingame/ml/batched_mcts.py), and [`champion_challenger.py`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py) modules coordinate Monte Carlo Tree Search (MCTS) for both self-play policy target generation and tournament gating. This document outlines the mathematical, algorithmic, and computational challenges of MCTS simulation scaling in Ultimate Tic-Tac-Toe.

---

## 1. The Shallow Simulation Trap & Prior Confirmation Bias

### Problem
In AlphaZero-style reinforcement learning, MCTS serves as a policy improvement operator: the visited distribution $\pi(a|s) \propto N(s, a)^{1/T}$ should be strictly stronger than the raw neural network prior $P(a|s)$. The network is trained by minimizing the cross-entropy loss:
$$\mathcal{L}_{\text{policy}} = -\sum_{a} \pi(a|s) \log P(a|s)$$

In standard PUCT tree descent, action selection is governed by:
$$a^* = \arg\max_a \left[ Q(s, a) + U(s, a) \right], \quad \text{where } U(s, a) = c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

When the simulation budget $N$ is too small (e.g., $N = 400$):
1. **Prior Confirmation Loop**: When a node is newly expanded or unvisited ($N(s, a) = 0$), $Q(s, a) = 0$. Consequently, $U(s, a) = c_{\text{puct}} P(s, a) \sqrt{\sum N}$ dictates initial branch selection entirely by prior $P(s, a)$.
   - For frequently visited actions ($N(s, a) \approx \kappa N$), the exploration term $U(s, a)$ decays as $O(1/\sqrt{N})$.
   - If an inferior move has a high prior $P(s, a)$ and receives a moderately positive value from early evaluations, $Q(s, a)$ dominates before deep refutations can be explored, freezing visits on the suboptimal branch.
2. **Empirical Diagnostic Proof**: In our automated diagnostic study ([`diagnostics.py`](file:///home/festivity/agy/uttt-codingame/ml/diagnostics.py)), evaluating diverse UTTT positions across simulation counts revealed:
   - At $N = 400$, the top-1 move selected by MCTS matches deep ground truth ($N = 1600$) **only 41.7% of the time**.
   - At $N = 1600$, MCTS overturns the neural network prior in **83.3% of positions**, driving policy-prior divergence $D_{\text{KL}}(\pi \parallel P)$ up from $0.088$ to $0.1368$ and sharpening entropy from $1.95$ to $1.89$.
3. **Training Stagnation**: If $\pi \approx P$, the policy gradient $-\nabla_\theta \sum \pi \log P_\theta$ approaches zero, causing the network's self-improvement cycle to stall.

---

## 2. Spatial Constraint Divergence in UTTT Search Trees

### Problem
Unlike standard Chess or Go where moves have local geometric consequences, Ultimate Tic-Tac-Toe possesses a global routing constraint: a move placed in cell $(r, c)$ of subboard $B$ forces the opponent's next move to occur in subboard $B' = 3r + c$.

1. **Receptive Field Non-Locality**: A tactical blunder in subboard $B'$ is only discovered if the MCTS search expands a branch at least 2 plies deep (one move in $B$, followed by the forced reply in $B'$).
2. **Visit Dilution**: With up to 81 legal moves on free-choice turns and an average branching factor of $\sim 9$ on forced turns, a shallow budget of $N = 400$ simulations allocates an average of only $400 / 9 \approx 44$ visits per root child, and $< 5$ visits at depth 2.
3. **Tactical Blunder Blindness**: 2-ply forced refutations (such as giving the opponent an immediate macro-board win via free choice) remain unvisited and unpruned at $N = 400$, injecting noisy, blunder-prone policy targets into the replay buffer.

---

## 3. Hardware Saturation vs. Search Latency Trade-Off

### Problem
Scaling MCTS simulations increases compute per move. If scaling simulations degrades evaluation throughput, the self-play loop generates fewer games per hour, reducing data diversity.

1. **Hardware Context**: The host environment features an NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) paired with 16 CPU cores and the pure Rust Rayon worker ([`libuttt.so`](file:///home/festivity/agy/uttt-codingame/target/release/libuttt.so)).
2. **Evaluation Throughput & Batch Sizing**:
   - At $B = 128$ parallel games, GPU Tensor Core batch occupancy is moderate, yielding $\sim 35,000$ to $50,000$ leaf evals/s ($N = 800 \implies 102,400$ evals/ply $\approx 2.2\text{s}$ per step).
   - At $B = 1,024$ parallel games, GPU Tensor Cores are fully saturated, delivering $> 240,000$ to $325,000$ leaf evals/s ($N = 800 \implies 819,200$ evals/ply $\approx 3.4\text{s}$ per step).
3. **Latency Profile**: Whether operating at 128 games (lightweight iteration) or 1024 games (production saturation), scaling from $N = 400$ to $N = 800$ simulations adds only $\approx 1.1\text{s}$ to $1.7\text{s}$ per move step while providing more than double the depth-2 tactical resolution.


---

## 4. Invariants & Guardrails

1. **Exploration Noise Invariance**: Root Dirichlet noise ($\alpha = 0.3, \epsilon = 0.25$) must remain applied exclusively at the root node during self-play, and must remain completely disabled ($\epsilon = 0.0$) during tournament evaluation matches.
2. **Temperature Schedule Preservation**: Temperature must maintain $T = 1.0$ for plies $\le 20$ and $T = 0.5$ for plies $> 20$ in self-play. Increasing simulation count must not be accompanied by lowering temperature to $0.0$, which would destroy endgame diversity.
3. **Memory Bound**: Rayon thread-local arenas in Rust must not leak tree nodes across games; `worker.advance_moves` must recycle node memory for finished games.
