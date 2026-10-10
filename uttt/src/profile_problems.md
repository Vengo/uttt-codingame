# Problems & Challenges: `uttt/src/profile.rs`

This document details hardware benchmarking and latency profiling challenges in `uttt/src/profile.rs`.

---

## 1. Challenges
- **CPU Cache Spilling & Memory Bottlenecks**: In MCTS and minimax search, hash table sizing critically determines performance. A table that fits within CPU L3 cache (e.g. 50,000 entries) runs with nanosecond latency, whereas a table that spills into main DRAM (e.g. 2,000,000 entries) suffers a 3x–10x latency penalty due to cache misses.
- **Dynamic Allocation Overhead**: Frequent small heap allocations during tree node expansion can stall the search engine. Measuring allocation throughput ($ns$ per allocation) is essential for tuning tree node allocators.
- **Bot Search Throughput**: Benchmarking rollouts per second across bot architectures under standardized time budgets.
