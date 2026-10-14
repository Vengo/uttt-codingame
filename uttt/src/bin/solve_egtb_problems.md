# Problems & Challenges: `uttt/src/bin/solve_egtb.rs`

This document details the operational constraints, storage safety invariants, and backend routing challenges in [`uttt/src/bin/solve_egtb.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/solve_egtb.rs).

---

## 1. Technical Challenges

- **Zero-Dependency CLI Ergonomics**:
  To maintain instant build times and zero third-party crate overhead, the binary implements CLI parsing manually using standard library primitives (`std::env::args`).
  - *Solution*: A robust token scan loops through CLI arguments, handling flags with positional parameters, default fallbacks, numeric conversion, and automated `--help` output with zero external dependencies.

- **Unified Dispatch Across Divergent Tablebase Backends**:
  The engine supports three fundamentally different tablebase algorithms:
  1. *Macro-Factored Solving* (`MacroFactoredSolver`): generates strictly bounded compact binary files ($\le 50$ MB).
  2. *Exhaustive ZFS Direct Indexing* (`Tier7ZfsSolver`): generates 614 uncompressed/ZFS-compressed binary slices (~76 GB).
  3. *Streaming Retrograde Inducting* (`RetrogradeSolver`): generates independent chunk files for distributed cluster solving and subsequent file-level merging.
  - *Solution*: `solve_egtb.rs` provides a unified CLI front-end that normalizes tier numbers, parameter limits, thread counts, and output directories across all three engines.

- **Preventing Accidental Disk-Exhaustion on Local Drives**:
  Exhaustive Tier 7 generation creates over 76 GB of slice data. Accidental invocation on a small root drive could crash the host system.
  - *Solution*: `solve_egtb.rs` requires an explicit `--exhaustive` flag or path sniffing (`tier7` or `zraid` in the output path) before triggering `Tier7ZfsSolver`. Otherwise, it safely defaults to the resource-bounded `MacroFactoredSolver`.

- **Orchestrating Distributed Chunked Solving**:
  In cluster or distributed environments (such as cloud worker spot instances), workers must solve small slices of Tier 6 independently without cross-worker communication.
  - *Solution*: Command-line flags `--chunk <idx>` and `--chunks <total>` partition the 84 triplets among worker processes. Each chunk outputs its own `.bin` and `.done` files, allowing an idempotent `merge6` subcommand to consolidate all results into `egtb_tier6.bin` once all workers finish.
