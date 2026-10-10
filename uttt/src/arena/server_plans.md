# Plans & Progress: `uttt/src/arena/server.rs`

This document details HTTP server and API implementation in `uttt/src/arena/server.rs`.

---

## 1. Plan & REST API Specification
- **`ArenaServer` Architecture**:
  - Multi-threaded TCP listener on port 8080 (default host `0.0.0.0`).
  - Shares read-only `Arc<Vec<Vec<GameMove>>>` and `Arc<ZobristConstants>` across threads.
- **REST Endpoints**:
  - `POST /api/match`: Body `{ "bot1": "alphabeta", "bot2": "umcts_v2", "time_ms": 40 }`. Returns complete match results and move lists.
  - `POST /api/tournament`: Body `{ "games": 10, "time_ms": 40, "bots": ["mcts", "alphabeta"] }`. Returns full leaderboard and pairwise win-loss matrices.
  - `POST /api/play`: Interactive move endpoint for human vs. bot canvas play.
  - `GET /api/bots`: Returns JSON array of available bot names.
  - `GET /study_status.json`: Serves live training daemon telemetry.
  - `GET /`: Serves the embedded HTML5/JS interactive replay and arena studio interface.

---

## 2. Status
- Running live on port 8080.
