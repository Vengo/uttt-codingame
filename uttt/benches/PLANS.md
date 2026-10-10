# Plans & Progress: `uttt/benches` Performance Micro-Benchmarks

This document details Criterion micro-benchmark suites, active test groups, and throughput targets in `uttt/benches`.

---

## 1. Benchmark Suites & Group Configurations

### 1.1 Standard 3x3 Benchmarks ([`ttt_benches.rs`](file:///home/festivity/agy/uttt-codingame/uttt/benches/ttt_benches.rs))
- **Active Group**: `ttt_benches`
  - `ttt_display_test`: String formatting and rendering latency.
  - `ttt_movegen_empty`: Move generation on an empty 3x3 board (< 15 ns).
  - `ttt_movegen_full`: Move generation on a partially filled board (< 10 ns).

### 1.2 9x9 Ultimate Tic-Tac-Toe Benchmarks ([`uttt_benches.rs`](file:///home/festivity/agy/uttt-codingame/uttt/benches/uttt_benches.rs))
- **Active Group** (Fast Alpha-Beta Regression):
  - `uttt_alpha_beta_big_window`: Evaluates wide alpha-beta window exploration on root board.
  - `uttt_alpha_beta_midgame`: Evaluates search throughput on complex mid-game tactical boards.
- **Standby Group** (Commented out to prevent 20-minute CI runs; runnable on demand or via `arena profile`):
  - `uttt_display_test`, `uttt_movegen_empty`, `uttt_movegen_full`.
  - `uttt_make_move`, `uttt_unmake_move`, `uttt_result_check`.
  - `uttt_playout`: Measures pure random rollout simulation throughput (Target: > 500,000 rollouts/sec).
  - `uttt_alpha_beta_zero_window`, `uttt_get_tied_mask`, `uttt_get_subboard_moves`, `uttt_hash_test`.

---

## 2. Performance Targets & Empirical Status
- **Move Generation**: < 45 ns per board.
- **Alpha-Beta Search**: > 1,200,000 nodes/sec with move ordering and transposition table hits.
- **Rollout Simulation**: ~550,000 rollouts/sec with `SmallRng`.
- **Status**: Verifiable via `cargo bench --bench ttt_benches` and `cargo bench --bench uttt_benches`.
