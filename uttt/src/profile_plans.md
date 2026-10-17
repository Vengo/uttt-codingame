# Plans & Progress: `uttt/src/profile.rs`

This document details hardware profiling implementation and benchmark metrics in `uttt/src/profile.rs`.

---

## 1. Plan & Profiling Harness
- **`profile_cpu_cache()`**: Measures 1,000,000 lookups against an L3-resident 50,000-entry table vs. a DRAM-spilling 2,000,000-entry table using `IdentityHasherBuilder`. Calculates the cache slowdown factor.
- **`profile_allocator()`**: Benchmarks 1,000,000 small node allocations to compute nanoseconds per allocation.
- **`profile_search_throughput()`**: Measures raw rollouts/sec and average search depth for bot architectures (`random`, `rule_based`, `alphabeta`, `mcts`, `umcts_v2`).
- **`generate_hardware_profile_report()`**: Assembles comprehensive system report exported to `hardware_profile.json`.

---

## 2. Status
- Executed via `arena profile`. Results recorded in `hardware_profile.json`.
