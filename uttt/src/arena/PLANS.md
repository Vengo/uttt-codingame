# Plans & Progress: `uttt/src/arena` Tournament Arena & Web Server

This document details tournament execution, Bayesian rating tracking, web server architecture, API specifications, and profiling status in `uttt/src/arena`.

---

## 1. Architectural Design & Subsystems

- **`Game` & `MatchRunner`** ([`game.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/game.rs), [`match_runner.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/match_runner.rs)):
  - Orchestrates two-player games with strict wall-clock timeout and illegal-move forfeit enforcement.
  - Records per-turn durations in microseconds and outputs replay histories.

- **`Tournament` & `EloCalculator`** ([`tournament.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/tournament.rs), [`elo.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/elo.rs)):
  - Executes round-robin pairings with symmetrical color assignment ($X$ and $O$ swaps).
  - Scores games: Win = $1.0$, Draw = $0.5$, Loss = $0.0$.
  - Updates Elo ratings using standard logistic curves ($K = 32$, initial Elo $1500.0$).

- **`GlickoCalculator` & Bayesian Benchmark Runner** ([`glicko.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/glicko.rs)):
  - Implements the complete Glicko-2 rating system tracking rating $r$, Rating Deviation $RD$, and volatility $\sigma$.
  - Executes paired head-to-head matches against the baseline `AlphaBetaBot` until $RD < 35$, computing 95% Bayesian credible intervals for sound experimental validation.

- **`ReplayExporter`** ([`replay.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/replay.rs)):
  - Serializes match records to JSON format (`ReplayData`) and renders interactive turn-by-turn game playbacks.

- **`ArenaServer`** ([`server.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/server.rs)):
  - Standard library multi-threaded HTTP/1.1 server (`std::net::TcpListener`) on port 8080.
  - Delivers embedded HTML5/Canvas UI and handles REST endpoints (`/api/bots`, `/api/play`, `/api/match`, `/api/tournament`, `/study_status.json`).

- **`FlamegraphSampler`** ([`flamegraph.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/flamegraph.rs)):
  - Uses `pprof` (100 Hz frequency) to profile bot searches with a 30 ms search deadline per move.

---

## 2. Status
- Server operational on port 8080.
- Bayesian benchmark runner with Glicko-2 convergence planned for validating `alphabeta_v2` and `umcts_v3`.
