# Problems & Challenges: `uttt/src/mcts_worker.rs`

This document details the technical challenges, concurrency synchronization, memory layout optimizations, and self-play algorithms solved in [`uttt/src/mcts_worker.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/mcts_worker.rs).

---

## 1. Technical Challenges

- **Heterogeneous CPU-GPU Throughput Imbalance**:
  CPU bitboard move generation operates at over 12 million plies per second, while neural network inference requires batched tensor passes (64–512 states) to achieve peak GPU efficiency (368k evals/s). Running independent un-batched searches leads to severe GPU under-utilization.
  - *Solution*: `MctsWorker` synchronizes $B$ concurrent game simulations into a coordinated multi-phase pipeline (`prepare_step` $\to$ GPU evaluation $\to$ `finish_step`). Features are written directly into pre-allocated NumPy/CUDA pinned host buffers via zero-copy C-ABI pointers.

- **Pointer Indirection & Memory Fragmentation in Tree Nodes**:
  Traditional tree data structures allocate nodes dynamically on the heap (`Box`, `Rc`, or `Arc<Mutex<T>>`). Under multi-threaded parallel descent across 16 threads, heap allocation contention and CPU cache misses severely throttle simulation speed.
  - *Solution*: A flat arena layout is used per game (`Vec<MctsNode>` and `Vec<MctsChild>`), indexing children by contiguous 32-bit offsets (`first_child + offset`). Node descent and backpropagation operate entirely on contiguous arrays with zero allocations during search.

- **Exploration Collapse and Temperature Scheduling**:
  If the sampling temperature $T$ drops to 0 too early during training, MCTS greedily focuses on narrow lines, starving the neural network of diverse positions and causing policy collapse. Conversely, excessive randomness weakens endgame play.
  - *Solution*: The self-play generator enforces an automatic two-phase temperature schedule: $T = 1.0$ for the opening (moves $\le 20$) and $T = 0.5$ thereafter. $T$ strictly remains $> 0$ during self-play data generation, while greedy argmax ($T < 0.05$) is reserved for competitive tournament evaluation.

- **Root Policy Novelty via Dirichlet Noise Injection**:
  During MCTS self-play, sharp priors from previous training checkpoints can dominate search visits, preventing exploration of promising alternative candidate moves.
  - *Solution*: Parametric Dirichlet noise $(\alpha, \epsilon)$ is injected into the root node priors over legal moves:
    $$P'(s_{\text{root}}, a) = (1 - \epsilon) P(s_{\text{root}}, a) + \epsilon \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$
    ensuring continuous exploration of novel game branches.

- **CodinGame Saturated Mini-Board Win Tiebreaker Fidelity**:
  In competitive Ultimate Tic-Tac-Toe on CodinGame, drawn games with 9 resolved subboards are scored by subboard popcount ($|M_X|$ vs $|M_O|$). Training value heads on flat draw values ($0.0$) produces models that neglect subboard maximization in the endgame.
  - *Solution*: Terminal game evaluations evaluate subboard popcounts upon reaching full boards or 81 plies, awarding $+1.0$ to the subboard leader and $-1.0$ to the opponent.

- **Zero-Copy FFI Marshalling for Python Training Loops**:
  Marshaling game states, policies, legal masks, and value targets across Python and Rust via JSON or socket IPC introduces prohibitive serialization overhead.
  - *Solution*: High-performance C-ABI functions (`mcts_worker_prepare_step`, `mcts_worker_finish_step`, `mcts_worker_drain_samples`) take raw raw memory pointers directly from NumPy arrays (`ctypes.c_void_p`), enabling zero-copy tensor exchange at memory bus bandwidth.
