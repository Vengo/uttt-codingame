# Plans & Progress: `uttt/src/bots/alphabeta_v2_bot.rs`

This document details the architecture, search enhancements, and implementation plan for `AlphaBetaV2Bot` in `uttt/src/bots/alphabeta_v2_bot.rs`.

---

## 1. Architectural Plan & Search Enhancements

### 1.1 Structural Definition & Constants
```rust
use crate::bots::Bot;
use crate::egtb::EgtbDatabase;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::opening_book::OpeningBook;
use crate::uttt_trait_solving::{parse_index, UltimateTicTacToeBoard, ZobristConstants};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const MAX_PLY: usize = 81;

pub struct AlphaBetaV2Bot {
    name: String,
    tt: HashMap<u64, TTEntry>,
    killers: [[Option<GameMove>; 2]; MAX_PLY],
    book: OpeningBook,
    egtb: Option<EgtbDatabase>,
}
```

### 1.2 Transposition Table Depth-Preferred Replacement
```rust
fn store_tt(&mut self, hash: u64, depth: u8, score: i32, flag: TTFlag, best_move: Option<GameMove>) {
    if let Some(existing) = self.tt.get_mut(&hash) {
        if depth >= existing.depth {
            let preserved_best = best_move.or(existing.best_move);
            *existing = TTEntry { depth, score, flag, best_move: preserved_best };
        }
    } else {
        if self.tt.len() >= 1_000_000 {
            self.tt.clear();
        }
        self.tt.insert(hash, TTEntry { depth, score, flag, best_move });
    }
}
```

### 1.3 In-Tree EGTB Probing
Placed directly after transposition table lookup in `negamax`:
```rust
let resolved_count = (board.x_supers | board.o_supers | board.get_tied_mask()).count_ones();
if resolved_count >= 4 {
    if let Some(ref egtb) = self.egtb {
        if let Some(res) = egtb.probe(board, zobrist_constants) {
            let score = match res {
                GameResult::P1Win => if board.x_to_move { 100_000 - ply as i32 } else { -100_000 + ply as i32 },
                GameResult::P2Win => if !board.x_to_move { 100_000 - ply as i32 } else { -100_000 + ply as i32 },
                GameResult::Tie => 0,
                GameResult::Unresolved => unreachable!(),
            };
            self.store_tt(hash, 255, score, TTFlag::Exact, None);
            return score;
        }
    }
}
```

### 1.4 Search Ply Tracking, Killer Moves, History Table & Countermove Table
- `negamax` signature:
  ```rust
  fn negamax(
      &mut self,
      board: &mut UltimateTicTacToeBoard,
      depth: u8,
      ply: usize,
      mut alpha: i32,
      mut beta: i32,
      prev_move: Option<GameMove>,
      zobrist_constants: &ZobristConstants,
      deadline: Instant,
      time_exceeded: &mut bool,
  ) -> i32
  ```
- **Killer Moves**: Recorded on quiet beta cutoff (`score >= beta`):
  `killers[ply][0]` and `killers[ply][1]`.
- **Countermove Table**: `countermoves: [Option<GameMove>; 81]` storing refutations to previous moves:
  `self.countermoves[prev_move] = Some(curr_move)`.
- **History Table**: Flat 26 KB table `history: [[u32; 81]; 81]` storing cutoff frequency:
  `self.history[prev_move][curr_move] += (depth * depth) as u32`.
- **Aging / Decay**: When history values exceed `MAX_HISTORY = 100_000`, all history entries are divided by 2.

### 1.5 Enhanced Move Ordering Pipeline & Macro-Board Routing Heuristics
Moves are scored and ordered hierarchically:
1. **TT Best Move**: Transposition table PV move (Score `10_000_000`).
2. **Immediate Game Win**: Move immediately completing a winning superboard line (Score `9_000_000`).
3. **Primary Killer Move**: `killers[ply][0]` (Score `8_000_000`).
4. **Secondary Killer Move**: `killers[ply][1]` (Score `7_000_000`).
5. **Countermove**: `countermoves[prev_move]` (Score `6_000_000`).
6. **History Score + Tactical / Macro Routing Heuristics**:
   - `history[prev_move][curr_move]` score.
   - **Tactical subboard captures**: `+600` bonus; local threat creation `+60`.
   - **Macro-Board Routing Heuristics**:
     - *Free Choice Gifting Penalty*: Moves directing the opponent to a resolved/completed subboard (gifting free choice across any open subboard) penalized by `-300` in move scoring and `-250` in static evaluation.
     - *Opponent Threat Penalty*: Moves directing the opponent into a subboard where the opponent has an active, unblocked 2-in-a-row threat penalized by `-400` in move scoring and `-350` in static evaluation.
     - *Neutralized Subboard Reward*: Directing opponent to a dead subboard where they have 0 winning lines rewarded with `+150`.
     - *Defensive Control Reward*: Directing opponent where active player holds defensive threat control rewarded with `+100`.
   - **Positional Values**: Center cell `+20`, corners `+10`, center subboard `+15`.

### 1.6 Ply-0 Opening Book
- Checks `self.book.lookup_with_min_visits(board, zobrist_constants, 5_000)` at ply 0.
- If present and legal, returns immediately in $< 1$ $\mu$s.

---

## 2. Experimental Verification Plan
- **Baseline Showdown**: Evaluated against preserved baseline `AlphaBetaBot` in the tournament arena using Glicko-2 Bayesian ratings.
- **Stopping Criterion**: Executed in single accumulated rating period until Rating Deviation $RD \le 35.0$ (or safety cutoff of 500 games).
- **Hypothesis Decision**: Experiment declared successful if the 95% Bayesian credible lower bound satisfies:
  $$r' - 1.96 \times RD' > 1500.0$$
