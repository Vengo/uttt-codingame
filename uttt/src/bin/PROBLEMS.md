# Problems & Challenges: `uttt/src/bin` Executable Binaries

This document details executable entry point requirements, CLI subcommand ergonomics, and execution environments in `uttt/src/bin`.

---

## 1. Challenges & Entry Point Constraints

### 1.1 Zero-Dependency CLI Subcommand Suite
The CLI (`arena`) provides 9 distinct operational commands (`tournament`, `match`, `play`, `replay`, `server`, `profile`, `flamegraph`, `study`, `list`). To maintain minimal build dependencies and instant compile times, argument parsing is implemented without heavy third-party crates (like `clap`). The challenge is providing clear usage help, robust flag extraction, and descriptive error messages for optional arguments.

### 1.2 CodinGame Single-File Submission Constraints (`cg.rs`)
The CodinGame platform referee requires competitors to submit a single, monolithic Rust source file without access to workspace crates or Cargo package managers.
- `cg.rs` maintains an self-contained MCTS search loop and board definition optimized for standard input/output (`stdin`/`stdout`).
- Modern engine improvements (`PackedBoardState`, DF-PN EGTB, UCT opening book) must eventually be concatenated into `cg.rs` via an automated bundler without exceeding CodinGame source size limits (typically 100 KB).

### 1.3 Hardware Profiling Metrics (`arena profile`)
Hardware profiling in [`profile.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/profile.rs) provides micro-architectural diagnostics before running compute-intensive searches. It benchmarks:
- Dynamic heap allocator throughput (`Vec` allocations vs. stack arrays).
- Cache line memory latency across varying stride sizes.
- Board hashing and clone speed.
- Pure move generation and rollout iteration limits.

### 1.4 Continuous Study Daemon Resilience & Storage Fallback (`arena study`)
The autonomous study daemon continuously expands the opening book, generates systematic endgame tablebases, and logs telemetry to `study_status.json`.
- **Storage Target**: Primary storage is directed to `/my-zraid/uttt/` (7.2 TB NVMe ZFS array).
- **Graceful Fallback**: If `/my-zraid/uttt/` is unmounted or missing, the daemon automatically falls back to local workspace paths (`book.bin` and `egtb_4plus.bin`) to ensure uninterrupted background execution.

### 1.5 Scalable Multi-Backend Tablebase Solving (`solve_egtb.rs`)
The dedicated endgame tablebase solver [`solve_egtb.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/solve_egtb.rs) requires routing across fundamentally divergent solving paradigms:
- **Multiplexing Engines**: Seamlessly dispatches between resource-capped `MacroFactoredSolver` ($\le 50$ MB), exhaustive NVMe ZFS direct indexing `Tier7ZfsSolver` (~76 GB), and chunk-streaming `RetrogradeSolver`.
- **Accidental Disk Exhaustion Prevention**: Requires explicit confirmation (`--exhaustive` or path matching `tier7`/`zraid`) before initiating full 614-slice generation to prevent overflowing root partitions.
- **Distributed Chunk Coordination**: Enables independent worker execution across spot instances using `--chunk` / `--chunks` flags, followed by an idempotent offline chunk consolidation step (`merge6`).
