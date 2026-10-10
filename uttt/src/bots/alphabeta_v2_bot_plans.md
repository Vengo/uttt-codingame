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

### 1.4 Ply Tracking & Killer Move Heuristics
- `negamax` signature:
  ```rust
  fn negamax(
      &mut self,
      board: &mut UltimateTicTacToeBoard,
      depth: u8,
      ply: usize,
      mut alpha: i32,
      mut beta: i32,
      zobrist_constants: &ZobristConstants,
      deadline: Instant,
      time_exceeded: &mut bool,
  ) -> i32
  ```
- Killer moves recorded on beta cutoff (`score >= beta`) if the move was non-tactical:
  ```rust
  if ply < MAX_PLY && unmove.a_super_unmask == 0 {
      if self.killers[ply][0] != Some(m) {
          self.killers[ply][1] = self.killers[ply][0];
          self.killers[ply][0] = Some(m);
      }
  }
  ```

### 1.5 Move Ordering Pipeline
1. Transposition table PV move (Priority 0).
2. Subboard captures & immediate line threats (Priority 10).
3. Primary killer move `killers[ply][0]` if legal (Priority 20).
4. Secondary killer move `killers[ply][1]` if legal (Priority 25).
5. Cell positional values & opponent direction penalties (Priority 30+).

### 1.6 Ply-0 Opening Book
- Checks `self.book.lookup(board, zobrist_constants)` at ply 0.
- If present and legal, returns immediately in $< 1$ $\mu$s.

---

## 2. Experimental Verification Plan
- **Baseline Showdown**: Evaluated against preserved baseline `AlphaBetaBot` in the tournament arena using Glicko-2 Bayesian ratings.
- **Stopping Criterion**: Executed in single accumulated rating period until Rating Deviation $RD \le 35.0$ (or safety cutoff of 500 games).
- **Hypothesis Decision**: Experiment declared successful if the 95% Bayesian credible lower bound satisfies:
  $$r' - 1.96 \times RD' > 1500.0$$
