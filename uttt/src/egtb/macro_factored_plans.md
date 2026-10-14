# Plans & Progress: `uttt/src/egtb/macro_factored.rs`

This document details the macro-factored endgame tablebase architecture, retrograde induction, and zero-copy prober in [`uttt/src/egtb/macro_factored.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs).

---

## 1. Architectural Design & Plan

- **Macro-Factoring Principle**:
  - Decouples macro board configurations from micro subboard states by parameterizing the macro board as a compact boundary vector:
    $$\tau = (xw_j, xw_k, ow_j, ow_k, x_{\text{shared}}, o_{\text{shared}})$$
    combined with a clamped mini-board win differential $\Delta \in [-3..+3]$.
  - Solves the game via exact backward induction from Tier 8 down through Tier 7 and Tier 6 without combinatorial explosion.

- **Strict Resource & Size Ceilings**:
  - Enforces strict upper bounds on disk file sizes and entry counts:
    * Tier 8: $\le 500,000$ entries, $\le 5$ MB file size (`TIER_8_MAX_ENTRIES`, `TIER_8_MAX_FILE_BYTES`).
    * Tier 7: $\le 3,500,000$ entries, $\le 50$ MB file size (`TIER_7_MAX_ENTRIES`, `TIER_7_MAX_FILE_BYTES`).
    * Tier 6: $\le 3,500,000$ entries, $\le 50$ MB file size (`TIER_6_MAX_ENTRIES`, `TIER_6_MAX_FILE_BYTES`).
  - Strict resident memory budget guard via [`MemoryBudgetGuard`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L64) monitoring `/proc/self/statm` against a 512 MB heap ceiling.

- **In-Memory Precomputed Lookup Tables**:
  - [`MicroTransitionTable`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L325): Precomputes legal moves, outcomes (`MicroMoveOutcome::Win`, `Tie`, `Next`), and target state indices for all 11,125 legal micro configurations across 9 cells.
  - [`Tier8Lookup`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L218): Compact in-memory DP table (445 KB resident) providing instantaneous $O(1)$ retrograde query when an open subboard closes during Tier 7 DP.
  - [`PairLayerTable`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L243): 2D compact layer structure storing game outcomes for pair partition $(e_j, e_k)$ across 2 turns and 3 target conditions (6 outcomes per state).

- **Backward Induction Engine ([`MacroFactoredSolver`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L397))**:
  - `solve_tier_8()`: Solves Tier 8 across 20 macro configurations in parallel using Rayon, persisting to `egtb_tier8.bin`.
  - `solve_tier_7(max_empty_cells)`: Iterates over the 36 open subboard pairs $(j, k)$, backward inducting by remaining empty cells $E = 1..\text{max\_empty}$, linking closed moves into `Tier8Lookup`.
  - `solve_tier_6(max_empty_cells)`: Iterates over 84 subboard triplets $(j, k, l)$, linking resolved moves into Tier 7 prober queries.
  - `persist_records()`: Writes sorted records prefixed with 8-byte `UTTTEG01` magic and entry count into 10-byte binary records (`hash: u64`, `outcome: u8`, `dtm: u8`).

- **Zero-Copy Memory-Mapped Runtime Prober ([`MacroFactoredProber`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L1755))**:
  - Loads tablebase files via `memmap2::Mmap`.
  - Provides `< 30` ns probing latency with zero heap allocations during search.
  - Automatically routes probes to macro win detection, `probe_tier_8`, `probe_tier_7`, `probe_tier_6`, or 9-resolved CodinGame tiebreaker calculation.

---

## 2. Types & Public APIs

- `check_subboard_win(bb: u16) -> bool`: Evaluates winning 3-in-a-row masks.
- `pair_to_index(j: usize, k: usize) -> usize`: $O(1)$ mapping from $(j, k)$ to $0..36$.
- `extract_augmented_tau(x_supers, o_supers, j, k) -> (bool, bool, bool, bool, bool, bool, i8)`: Computes 6-bit boundary vector and clamped differential $\Delta$.
- `encode_tier8_key`, `encode_tier7_key`: 64-bit packed keys for binary search records.
- `MicroMoveOutcome`, `MicroMove`, `MicroMovesForState`, `MicroTransitionTable`: Move transition structures.
- `Tier8Lookup`: Precomputed resident Tier 8 table with `new()` and `get()`.
- `PairLayerTable`: 2D retrograde partition table with `get()` and `set()`.
- `MacroFactoredSolver`: Multi-threaded solver with `new()`, `solve_tier_8()`, `solve_tier_7()`, `solve_tier_6()`, and `persist_records()`.
- `MacroFactoredProber`: Memory-mapped prober with `new()`, `with_dir()`, `probe()`, `probe_tier_8()`, `probe_tier_7()`, `probe_tier_6()`, and `probe_game_result()`.

---

## 3. Status & Verification

- `test_tier8_and_tier7_size_bounds`: Verifies that Tier 8 ($\le 500\text{k}$ entries, $\le 5$ MB) and Tier 7 ($\le 3.5\text{M}$ entries, $\le 50$ MB) strictly respect entry count and file size bounds.
- `test_collinear_shared_line_unlocks_tier8_win`: Validates that collinear shared lines correctly upgrade to single-board wins upon subboard resolution.
- `test_codingame_miniboard_tiebreaker`: Verifies mini-board count tiebreak evaluation at $E=0$.
- `test_macro_factored_prober_latency_and_accuracy`: Benchmarks probing throughput at $< 100$ ns per query with zero memory allocations.
