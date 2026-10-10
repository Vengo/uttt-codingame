# Problems & Challenges: `uttt/src/arena` Tournament Arena & Web Server

This document details tournament management, Elo rating calculation, flamegraph profiling, timeout enforcement, and web server challenges in `uttt/src/arena`.

---

## 1. Challenges & Implementation Details

### 1.1 Fair Multi-Bot Tournament Execution & Scoring
- **Color Inversion**: To neutralize first-player advantage in Ultimate Tic-Tac-Toe, tournaments enforce even-numbered game counts between any bot pair $(A, B)$, alternating $X$ (first) and $O$ (second).
- **Match Scoring**: Standard tournament scoring assigns:
  - Win: $1.0$ point
  - Draw: $0.5$ points
  - Loss: $0.0$ points
- **Elo Calibration**: Ratings start at a baseline of $1500.0$ and update after each game using standard logistic expected score formulas with $K = 32$.

### 1.2 Strict Wall-Clock Timeout & Forfeit Enforcement
Each bot move is monitored using `Instant::now()`. If a bot fails to return within its allotted time limit (`time_limit + margin`), or if it returns an illegal move index, the runner immediately terminates the match and awards an automatic forfeit victory to the opposing bot.

### 1.3 Flamegraph CPU Profiling without Skew
Generating execution flamegraphs via `pprof` requires sampling call stacks at a steady frequency (100 Hz) without interfering with search thread scheduling. In [`flamegraph.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/flamegraph.rs), simulated moves run with a fixed 30 ms decision deadline so that realistic game tree searches are sampled repeatedly across root, midgame, and endgame phases.

### 1.4 Zero-Dependency HTTP Web Server & Schema Definitions
The web server runs directly on `std::net::TcpListener` without external web framework dependencies (e.g. Actix or Axum).
- **Embedded UI**: The HTML5/Canvas visualizer is embedded directly in binary source strings, guaranteeing zero runtime asset lookup failures.
- **REST API Schemas**:
  - `/api/play`: Request: `{ "bot": "<name>", "moves": [u8], "time_ms": Option<u64> }`. Response: `{ "bot_move_index": u8, "bot_row": usize, "bot_col": usize, "bot_subboard": usize, "bot_subsquare": usize, "game_over": bool, "winner": String }`.
  - `/api/match`: Request: `{ "bot1": "<name>", "bot2": "<name>", "time_ms": Option<u64> }`.
  - `/study_status.json`: Serves continuous study telemetry including cycle count, book entry size, and tablebase entry size.
