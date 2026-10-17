# Problems & Challenges: `uttt/benches` Performance Micro-Benchmarks

This document details benchmarking challenges, Criterion measurement durations, and micro-performance optimization in `uttt/benches`.

---

## 1. Challenges & Profiling Bottlenecks

### 1.1 Criterion Execution Time vs. Comprehensive Coverage
A full Criterion benchmark suite encompassing move generation, state updates, Zobrist hashing, and deep rollouts requires 100 sample iterations per function with statistical warmups. In Ultimate Tic-Tac-Toe:
- Running all 14 defined benchmarks in [`uttt_benches.rs`](file:///home/festivity/agy/uttt-codingame/uttt/benches/uttt_benches.rs) requires 15–20 minutes.
- To balance CI responsiveness with performance tracking, the default active Criterion group (`uttt_benches`) is scoped to core minimax searches (`uttt_alpha_beta_big_window`, `uttt_alpha_beta_midgame`), while high-speed rollout profiling is delegated to [`arena profile`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/arena.rs) or manually enabled.

### 1.2 Nanosecond Move Generation & Dead-Code Elimination
Move generation in bitboard engines operates in the nanosecond domain (< 50 ns per board). Without defensive benchmarking:
- LLVM aggressively optimizes out pure bitwise expressions when the resulting moves are unread.
- All benchmark loops must wrap board states and returned moves in `criterion::black_box` to force real memory and register computation without measuring allocator noise.

### 1.3 High-Throughput Rollout Target Verification
MCTS bots require massive simulation volume to evaluate complex branching positions. A key performance requirement is achieving $\ge 500,000$ pure random rollouts per second on modern x86_64 CPUs. Measuring this reliably requires amortizing RNG state initialization (`SmallRng`) across tens of thousands of continuous game simulations.
