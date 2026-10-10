# Plans & Progress: `uttt/src/bin` Executable Binaries

This document details executable binary implementations, CLI subcommands, and submission pipelines in `uttt/src/bin`.

---

## 1. Binary Roster & Architectures

1. **`arena`** ([`arena.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/arena.rs)):
   - Unified CLI suite supporting:
     - `arena tournament`: Multi-game round-robin competitions with Elo calculations.
     - `arena match`: Head-to-head bot showdown with optional replay logging.
     - `arena play`: Terminal-based interactive human vs. bot game.
     - `arena replay`: Reads recorded JSON match files and renders turn-by-turn board positions and timing to stdout.
     - `arena server`: Launches the multi-threaded HTTP/1.1 web studio on port 8080.
     - `arena profile`: Executes cache latency, allocator speed, and movegen throughput benchmarks.
     - `arena flamegraph`: Samples bot searches via `pprof` (100 Hz, 30 ms/move) and outputs `.svg` profiles.
     - `arena study`: Autonomous daemon executing cycles of opening book UCT dominance deepening and EGTB Tier solving.
     - `arena list`: Prints registered bots and capability summaries.

2. **`cg`** ([`cg.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/cg.rs)):
   - Monolithic single-file submission entry point tailored for the CodinGame online referee.
   - Reads opponent moves from `stdin` and writes move coordinates (`row col`) to `stdout` within 100 ms deadlines.
   - Planned for transition to an automated bundle of `book_umcts_v2`.

3. **`runlocal`** ([`runlocal.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bin/runlocal.rs)):
   - Standalone offline test driver for validating CodinGame protocol compatibility locally.

---

## 2. Status & Verification
- All 3 binaries compile cleanly in both debug and release configurations (`cargo build --bins`).
- `arena` tested and verified across all subcommands (`tournament`, `match`, `play`, `replay`, `server`, `profile`, `flamegraph`, `study`).
