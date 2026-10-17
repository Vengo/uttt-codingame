# Opening Suite Implementation Plan: Book-Diversified Benchmark Tournament

## 1. Architectural Overview
This plan defines the integration of Opening Book suites into the arena evaluation harness. This allows deterministic bots (such as `AlphaBetaBot` and `AlphaBetaV2Bot`) to be benchmarked across diverse tactical game-tree branches using paired color games, resolving the empty-board deterministic stagnation trap.

```
       +-----------------------------------------------+
       |             OpeningBook (book.bin)            |
       +-----------------------------------------------+
                              |
                     BFS Tree Walk (even depth K=2 or 4)
                              |
                              v
       +-----------------------------------------------+
       |        OpeningSuite (Vec<UltimateTicTacToe>)  |
       +-----------------------------------------------+
                              |
                 Paired Color Game Schedule
                              |
               +--------------+--------------+
               |                             |
               v                             v
        [Game 1 from P_i]             [Game 2 from P_i]
         Bot 1 (X) vs Bot 2 (O)        Bot 2 (X) vs Bot 1 (O)
               |                             |
               +--------------+--------------+
                              |
                              v
       +-----------------------------------------------+
       |       Glicko-2 Single-Period Convergence      |
       |             Target RD <= 35.0                 |
       +-----------------------------------------------+
```

## 2. Refined Implementation Steps

### Step 1: Position-Based Game Execution in `uttt/src/arena/game.rs` & `replay.rs`
1. Extend `GameReplay` in `uttt/src/arena/replay.rs`:
   - Add `#[serde(default)] pub initial_board_display: Option<String>`.
2. Add `pub fn play_from_position(&self, bot_x: &mut dyn Bot, bot_o: &mut dyn Bot, initial_board: &UltimateTicTacToeBoard, legal_moves_lookup: &[Vec<GameMove>], zobrist_constants: &ZobristConstants) -> (GameStats, Option<GameReplay>)`:
   - Reset both bots (`bot_x.reset()`, `bot_o.reset()`).
   - Clone `initial_board`. If `self.record_replay`, set `replay.initial_board_display = Some(initial_board.display())`.
   - `turn` in `GameStats` counts moves executed *from the starting position*.
   - Bot roles are stationary (`bot_x` always plays 'X', `bot_o` always plays 'O').
   - Game outcomes (`XWin` vs `OWin`) map directly to piece ownership.
3. Refactor `play(&self, ...)`:
   - Delegates directly to `self.play_from_position(bot_x, bot_o, &UltimateTicTacToeBoard::default(), ...)`.

### Step 2: Opening Suite Extraction in `uttt/src/opening_book.rs`
1. Implement `pub fn extract_opening_suite(&self, target_count: usize, target_ply: usize, legal_moves_lookup: &[Vec<GameMove>], zobrist_constants: &ZobristConstants) -> Vec<UltimateTicTacToeBoard>`:
   - Enforce even ply depth: `let target_ply = (target_ply / 2).max(1) * 2;` (guarantees `board.x_to_move == true`).
   - Use BFS exploration starting from `c_root = SymmetryEngine::canonical_form(&root, zobrist_constants)`:
     - For depth $0 \le d < \text{target\_ply}$:
       - If position in `self.entries`, follow the 1 canonical book move.
       - Also follow top 2 legal moves from `c_board.get_legal_moves(legal_moves_lookup)` (branching factor 3).
       - For each child, compute canonical form `(canon_child, _)` and deduplicate with `visited_hashes.insert(canon_child.full_hash())`.
     - When reaching `target_ply`:
       - If `c_board.get_win_state() == GameResult::Unresolved`, add to `suite`.
       - Stop when `suite.len() >= target_count`.
2. Implement fallback supplementation:
   - If `suite.len() < target_count`:
     - Run a loop up to `max_attempts = target_count * 20` using random legal plies from root to `target_ply` with RNG.
     - Add unique unresolved positions until `suite.len() == target_count` or attempts exhausted.
   - If `suite.is_empty()`, push `UltimateTicTacToeBoard::default()`.

### Step 3: MatchRunner & Arena CLI Integration
1. Extend `MatchRunner` (`uttt/src/arena/match_runner.rs`):
   - Add `pub fn with_openings(mut self, openings: Vec<UltimateTicTacToeBoard>) -> Self`.
   - In `run_match`: for `game_idx in 0..games`:
     - Select opening: `let initial_board = &self.openings[(game_idx / 2) % self.openings.len()];`
     - Alternating games $2k$ (Bot 1 as X, Bot 2 as O) and $2k+1$ (Bot 2 as X, Bot 1 as O) both start from the exact same `initial_board`.
2. Update `run_benchmark_cmd` in `uttt/src/bin/arena.rs`:
   - Parse `--openings` (flag) and `--opening-depth <usize>` (default: 2 or 4).
   - If `--openings` is enabled:
     - Extract `opening_suite` via `book.extract_opening_suite(max_games / 2, opening_depth, &legal_moves, &zobrist_constants)`.
     - Print suite stats: `println!("Generated opening suite: {} unique positions (depth {} plies)", suite.len(), opening_depth);`
     - In the pair loop: for `pair`:
       - Select `pos = &suite[(pair - 1) % suite.len()]`.
       - Game 1: `game_runner.play_from_position(candidate_bot, baseline_bot, pos, ...)`
       - Game 2: `game_runner.play_from_position(baseline_bot, candidate_bot, pos, ...)`
3. Update `run_match_cmd`:
   - Support `--openings` using `MatchRunner::new(...).with_openings(suite)`.

### Step 4: Verification & Benchmarking
1. Run `cargo test --workspace` ensuring all tests pass cleanly.
2. Run `arena benchmark --bot alphabeta_v2 --baseline alphabeta --openings --opening-depth 2 --target-rd 35 --time-ms 30 --max-games 100`:
   - Confirm that every pair starts from a distinct opening position.
   - Analyze Glicko-2 ratings under diversified opening play.
