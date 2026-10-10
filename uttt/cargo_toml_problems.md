# Problems & Challenges: `uttt/Cargo.toml`

This document details dependency management, feature flags, and binary definitions in `uttt/Cargo.toml`.

---

## 1. Challenges & Constraints
- **Zero Heavy External Dependencies for Core Engine**: To remain competitive on CodinGame, the core engine must not depend on complex C-libraries or massive runtimes.
- **Fast Random Number Generation**: MCTS rollouts require billions of fast pseudorandom draws per second; standard `ThreadRng` introduces thread-local synchronization overhead.
- **Profiling & Flamegraphs**: Enabling CPU profiling via `pprof` without introducing unwanted overhead in standard production/release builds.
- **Memory Mapping**: Zero-copy lookup into multi-gigabyte tablebases requires safe, low-overhead memory-mapping (`memmap2`).
