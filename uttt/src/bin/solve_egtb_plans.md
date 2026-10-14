# Plans & Progress: `uttt/src/bin/solve_egtb.rs`

This document details the command-line interface, backend dispatch logic, and execution options in [`uttt/src/bin/solve_egtb.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/solve_egtb.rs).

---

## 1. Architectural Design & Plan

- **Dedicated Tablebase Solving Executable**:
  - Independent binary entry point decoupled from the match runner (`arena`) and submission wrapper (`cg`), optimized for long-running batch solving, HPC cluster tasks, and local tablebase generation.
  - Implements lightweight, zero-dependency argument parsing (`std::env::args()`).

- **Command-Line Interface & Subcommands**:
  - Flags & Options:
    * `--tier <8|7|7-zfs|6|all6|merge6|all>`: Specifies the target endgame tier or merging operation.
    * `--out-dir <path>`: Target directory for `.bin` files and chunk subdirectories (default: `egtb`).
    * `--threads <N>`: Worker thread pool size (default: 0, utilizing all logical CPU cores).
    * `--max-positions <N>`: Maximum position budget for bounded retrograde runs (default: 500,000).
    * `--chunk <idx>`: Chunk index for Tier 6 distributed streaming (default: 0).
    * `--chunks <total>`: Total number of chunks for distributed streaming (default: 8).
    * `--exhaustive`: Directs Tier 7 generation to the exhaustive 614-slice ZFS solver.
    * `--help` / `-h`: Displays formatted usage and argument descriptions.

- **Backend Dispatch Logic**:
  - **Tier 8**: Dispatches to `MacroFactoredSolver::solve_tier_8()` to generate the compact `egtb_tier8.bin` ($\le 500\text{k}$ entries, $\le 5$ MB).
  - **Tier 7**:
    * If `--exhaustive` is specified or `--out-dir` points to a ZFS array (`tier7` or `zraid`), invokes `Tier7ZfsSolver::solve_all()`, generating all 614 canonical slices (~76 GB total).
    * Otherwise, invokes `MacroFactoredSolver::solve_tier_7(empty_limit)` producing a compact table ($\le 3.5\text{M}$ entries, $\le 50$ MB).
  - **Tier 6**:
    * If `--chunk <idx>` is passed: dispatches to `RetrogradeSolver::solve_tier_6_chunk(chunk_idx, total_chunks, max_positions)`.
    * If `--tier all6` is passed: executes sequential streaming of all 8 chunks and automatically triggers `merge_tier_6_chunks()`.
    * Otherwise: invokes `MacroFactoredSolver::solve_tier_6(empty_limit)` for fast depth-stratified retrograde DP.
  - **Merge Tier 6**: Invokes `RetrogradeSolver::merge_tier_6_chunks()` to aggregate all binary chunk files in `chunks/` into the consolidated `egtb_tier6.bin`.
  - **All**: Runs a sequential full generation sequence (Tier 8, Tier 7, and Tier 6 Chunk 0) via `RetrogradeSolver`.

---

## 2. Types & Public APIs

- Entry point: `fn main()` with `print_usage()`.
- Dependencies: `uttt::egtb::macro_factored::MacroFactoredSolver`, `uttt::egtb::tier7_zfs::Tier7ZfsSolver`, and `uttt::egtb::retrograde::RetrogradeSolver`.

---

## 3. Status & Verification

- Compiles cleanly as a first-class binary target (`cargo build --bin solve_egtb`).
- Verified for correct argument routing across Tier 8, Tier 7 (compact and ZFS), Tier 6, and chunk merging modes.
