# Plans & Progress: `uttt/Cargo.toml`

This document details the configuration choices, crate dependencies, and status of `uttt/Cargo.toml`.

---

## 1. Plan & Dependency Selections
- `rand = { version = "0.8.5", features = ["small_rng"] }`: Provides `SmallRng` (Xoshiro256++ / SplitMix64), achieving > 100M random numbers/sec for MCTS rollouts.
- `memmap2 = "0.9"`: Provides zero-copy memory-mapping for instant tablebase probing directly against OS page caches.
- `pprof = { version = "0.13", features = ["flamegraph"] }`: Native Linux perf-based sampling producing standalone interactive SVGs.
- `serde` & `serde_json`: High-throughput serialization for replay files and daemon telemetry.

---

## 2. Current Implementation & Status
- All dependencies are pinned, audited, and compile cleanly in release mode.
- Binary targets `cg`, `runlocal`, and `arena` are properly defined with dedicated entry points.
