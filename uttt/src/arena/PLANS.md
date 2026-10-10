# Plans & Progress: `uttt/src/arena` Tournament Arena & Web Server

This document details tournament execution, web server architecture, API specifications, and profiling status in `uttt/src/arena`.

---

## 1. Architectural Design & Subsystems

- **`Game` & `MatchRunner`** ([`game.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/game.rs), [`match_runner.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/match_runner.rs)):
  - Orchestrates two-player games with strict wall-clock timeout and illegal-move forfeit enforcement.
  - Records per-turn durations in microseconds and outputs replay histories.

- **`Tournament` & `EloCalculator`** ([`tournament.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/tournament.rs), [`elo.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/elo.rs)):
  - Executes round-robin pairings with symmetrical color assignment ($X$ and $O$ swaps).
  - Scores games: Win = $1.0$, Draw = $0.5$, Loss = $0.0$.
  - Updates Elo ratings using standard logistic curves ($K = 32$, initial Elo $1500.0$).

- **`ReplayExporter`** ([`replay.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/replay.rs)):
  - Serializes match records to JSON format (`ReplayData`) and renders interactive turn-by-turn game playbacks.

- **`ArenaServer`** ([`server.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/server.rs)):
  - Standard library multi-threaded HTTP/1.1 server (`std::net::TcpListener`) on port 8080.
  - Delivers embedded HTML5/Canvas UI and handles REST endpoints:
    - `/api/bots`: Lists available bot roster.
    - `/api/play`: Evaluates human vs bot moves on demand.
    - `/api/match`: Triggers live bot vs bot exhibition matches.
    - `/api/tournament`: Runs background round-robin competitions.
    - `/study_status.json`: Exposes real-time study daemon progress.

- **`FlamegraphSampler`** ([`flamegraph.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/flamegraph.rs)):
  - Uses `pprof` (100 Hz frequency) to profile bot searches over configurable durations.
  - Enforces a 30 ms search deadline per move to capture realistic, balanced execution stacks.

---

## 2. Status
- Server operational on port 8080.
- All tournament, replay export, and flamegraph profiling features verified.
