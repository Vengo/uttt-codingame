# Plans & Progress: `uttt/src/bin/arena.rs`

This document details the CLI suite and study daemon implementation in `uttt/src/bin/arena.rs`.

---

## 1. Plan & Command Map
- `arena tournament [--games N] [--time-ms T] [--bots b1,b2,...]`: Runs round-robin tournament, writes `tournament_results.json`.
- `arena match --bot1 <name> --bot2 <name> [--games N] [--time-ms T] [--replay <path>]`: Head-to-head match runner with replay export.
- `arena play --bot <name> [--time-ms T] [--play-as x|o]`: Interactive terminal play.
- `arena replay <file.json>`: Parses recorded JSON match file and outputs step-by-step ASCII terminal visualizations of board state progression and move timings.
- `arena server [--port P] [--host H]`: Launches HTTP web studio on port 8080.
- `arena profile`: Executes cache, allocator, and rollout throughput benchmarks, saving `hardware_profile.json`.
- `arena flamegraph --bot <name> [--time-ms T] [--out <file.svg>]`: Generates interactive SVG flamegraph via `pprof` (100 Hz, 30 ms/move).
- `arena study [--cycles N] [--time-limit-mins M] [--target-egtb T] [--book-depth D]`: Continuous learning daemon with ZFS storage fallback and JSON telemetry.

---

## 2. Status
- Fully implemented and compiled in release mode (`target/release/arena`).
