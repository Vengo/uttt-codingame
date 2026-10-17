# Track 2 Plan: Classical AlphaBeta Bots (Zobrist Full-Hash Fix, Fixed-Size TT, Mate-Distance Scoring, Root PV Preservation & PVS/LMR)

**Status**: Reviewed and Approved by Isolated Plan Reviewer (`7adcfed4-e72c-4c42-828b-e7e384bad7b9`)

---

## 1. Problem Description

### Problem 2A: Transposition Table Zobrist Key Collisions on `last_super_index`
In `UltimateTicTacToeBoard::make_move` (`uttt/src/uttt_trait_solving.rs`), `self.hash_state` (`board.hash_val()`) only XORs stone placements (`x_constants` / `o_constants`) and `x_to_move`. It does **not** include `board.last_super_index` (the forced target subboard `0..=8`, or `>= 9` for Free Choice).
All three classical AlphaBeta bots (`AlphaBetaV2Bot`, `AlphaBetaBot`, and `PolicyAlphaBetaBot`) currently index their Transposition Table using raw `board.hash_val()`.
Whenever two move sequences place the same X and O stones on the board in a different order ending on different sub-squares, they share the exact same `board.hash_val()` and `x_to_move` but have different `last_super_index` constraints, causing false TT cutoffs and wrong `tt_best` moves.

### Problem 2B: `HashMap` Overhead and Sudden-Death `self.tt.clear()` at 1,000,000 Entries
In `AlphaBetaV2Bot` and `PolicyAlphaBetaBot`, `store_tt` calls `if self.tt.len() >= 1_000_000 { self.tt.clear(); }`. Wiping the entire TT mid-search destroys principal variation and move ordering across iterative deepening depths, and `std::collections::HashMap` incurs pointer-chasing and hashing overhead.

### Problem 2C: Flat Win Scores (`100_000`) Without Ply-Distance Adjustment or Terminal vs. EGTB Separation
Returning a flat `100_000` for both immediate 1-ply terminal wins, 9-ply terminal wins, and non-terminal EGTB wins means the bots cannot distinguish shorter mates from longer mates (or an immediate terminal win from an `InProgress` EGTB win), leading to delayed wins or cycling in winning endgames.

### Problem 2D: Root PV Move Preservation & Fail-Soft PVS / LMR
At the root (`ply == 0`), `best_move` from iterative deepening depth `d - 1` is only recovered via a TT lookup at the end of the iteration and is not guaranteed to be searched first if the root TT entry was overwritten. In addition, `negamax` searches every child with a full `(-beta, -alpha)` window at full `depth - 1` rather than using Principal Variation Search (PVS) and Late Move Reductions (LMR).

---

## 2. Implementation Plan

### Step 1: Unified `full_hash` Including `last_super_index` Across All Three AlphaBeta Bots
In `AlphaBetaV2Bot` (`uttt/src/bots/alphabeta_v2_bot.rs`), `AlphaBetaBot` (`uttt/src/bots/alphabeta_bot.rs`), and `PolicyAlphaBetaBot` (`uttt/src/bots/policy_alphabeta_bot.rs`), define:
```rust
#[inline(always)]
pub fn full_hash(board: &UltimateTicTacToeBoard) -> u64 {
    let target_idx = (board.last_super_index as usize).min(9) as u64;
    board.hash_val() ^ (target_idx + 1).wrapping_mul(0x9E3779B97F4A7C15).rotate_left(17)
}
```
- Clamping `board.last_super_index` at `9` ensures that `9` and `255` (both representing Free Choice) map to the exact same hash while `0..=8` and `9` all map to distinct 64-bit masks.
- Replace all `board.hash_val()` calls in `AlphaBetaV2Bot`, `AlphaBetaBot`, and `PolicyAlphaBetaBot` with `Self::full_hash(board)`.

### Step 2: Fixed-Size Power-of-Two Transposition Table (`Vec<TTEntry>`) in `AlphaBetaV2Bot` and `PolicyAlphaBetaBot`
Replace `HashMap<u64, TTEntry>` with a flat `Vec<TTEntry>` of size `TT_SIZE = 1 << 20` (1,048,576 entries, 16 MB):
```rust
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum TTFlag {
    Empty = 0,
    Exact = 1,
    LowerBound = 2,
    UpperBound = 3,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct TTEntry {
    key: u64,
    score: i32,
    depth: u8,
    flag: TTFlag,
    best_move: Option<GameMove>,
    generation: u8,
}
```
- Increment `self.generation = self.generation.wrapping_add(1)` at the start of each `choose_move`.
- Lookup: `let idx = (hash as usize) & (TT_SIZE - 1);`, match if `entry.flag != TTFlag::Empty && entry.key == hash`.
- Store: replace if `entry.flag == TTFlag::Empty || entry.key == hash || entry.generation != self.generation || depth >= entry.depth`. Preserve `existing.best_move` if `best_move.is_none() && entry.key == hash`. Never wipe the whole table mid-search.

### Step 3: Separated Mate-Distance (`MATE_SCORE`) and EGTB-Distance (`EGTB_WIN_SCORE`) Scoring with TT Ply Normalization
1. Define constants:
   - `pub const MATE_SCORE: i32 = 100_000;`
   - `pub const EGTB_WIN_SCORE: i32 = 95_000;`
   - `pub const MATE_THRESHOLD: i32 = 90_000;`
2. Score assignments:
   - Actual terminal game win/loss (`board.get_win_state() != GameResult::Unresolved` or 81-ply tiebreak):
     - Win for side to move: `MATE_SCORE - ply as i32`
     - Loss for side to move: `-MATE_SCORE + ply as i32`
   - 1-ply depth-0 macro-win threat detection:
     - Win in 1 ply for side to move: `MATE_SCORE - (ply as i32 + 1)`
   - Non-terminal EGTB hit (`ply > 0 && resolved_count >= 6`):
     - EGTB win for side to move: `EGTB_WIN_SCORE - ply as i32`
     - EGTB loss for side to move: `-EGTB_WIN_SCORE + ply as i32`
     - EGTB draw: `0`
3. TT Score Ply Normalization:
   - `#[inline(always)] fn score_to_tt(score: i32, ply: usize) -> i32`:
     if `score >= MATE_THRESHOLD` -> `score + ply as i32`, else if `score <= -MATE_THRESHOLD` -> `score - ply as i32`, else `score`.
   - `#[inline(always)] fn score_from_tt(score: i32, ply: usize) -> i32`:
     if `score >= MATE_THRESHOLD` -> `score - ply as i32`, else if `score <= -MATE_THRESHOLD` -> `score + ply as i32`, else `score`.

### Step 4: Root PV Preservation, Fail-Soft PVS, and LMR in `AlphaBetaV2Bot` and `PolicyAlphaBetaBot`
1. At the root (`ply == 0`), pass `Some(best_move)` from the previous completed depth iteration into `order_moves` (falling back to `tt_best` at `ply > 0`), and stop iterative deepening early if `score >= MATE_SCORE - 100`.
2. In `negamax` move loop (`for (i, m) in moves.into_iter().enumerate()`):
   - For `i == 0`: search full window `-self.negamax(board, depth - 1, ply + 1, -beta, -alpha, ...)`.
   - For `i > 0`:
     - If `depth >= 3 && i >= 4 && !is_tactical`: run LMR search at `depth - 2` with null window `(-alpha - 1, -alpha)`.
     - If LMR was not run or returned `score > alpha`: run unreduced null-window search at `depth - 1` with window `(-alpha - 1, -alpha)`.
     - If the `depth - 1` null-window search returned `score > alpha && beta > alpha + 1`: re-search at `depth - 1` with full window `(-beta, -alpha)`.

---

## 3. Testing Plan
1. **`test_full_hash_distinguishes_target_subboard`**:
   - Construct two move sequences reaching identical `xbb`, `obb`, and `x_to_move` with different `last_super_index`; verify `board_a.hash_val() == board_b.hash_val()` and `AlphaBetaV2Bot::full_hash(&board_a) != AlphaBetaV2Bot::full_hash(&board_b)`.
   - Verify `last_super_index = 9` and `last_super_index = 255` yield identical `full_hash`.
2. **`test_mate_in_1_preferred_over_longer_win_and_egtb`**:
   - Verify that `AlphaBetaV2Bot` strictly prefers an immediate 1-ply terminal macro win (`100_000 - 1`) over a 3-ply terminal win (`100_000 - 3`) or a non-terminal EGTB win (`95_000 - 1`).
3. **Full Test Suite Verification**:
   - Run `cargo test --release` across all unit and integration tests.
