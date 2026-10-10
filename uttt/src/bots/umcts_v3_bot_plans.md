# Plans & Progress: `uttt/src/bots/umcts_v3_bot.rs`

This document details the architecture, proof propagation, heuristic move ordering, and experimental plan for `UmctsV3Bot` in `uttt/src/bots/umcts_v3_bot.rs`.

---

## 1. Architectural Plan & Search Enhancements

### 1.1 Structural Definition
```rust
use crate::bots::Bot;
use crate::egtb::EgtbDatabase;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::opening_book::OpeningBook;
use crate::uttt_trait_solving::{parse_index, UltimateTicTacToeBoard, ZobristConstants};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProofStatus {
    Unproven,
    ProvenWin,
    ProvenLoss,
    ProvenDraw,
}

pub struct UmctsV3Bot {
    name: String,
    book: OpeningBook,
    egtb: Option<EgtbDatabase>,
    exploration_constant: f64,
}
```

### 1.2 In-Tree EGTB Probing & Proof Pruning
When creating or expanding a new node for `sim_board`:
```rust
let resolved = (sim_board.x_supers | sim_board.o_supers | sim_board.get_tied_mask()).count_ones();
let mut proof = ProofStatus::Unproven;
let mut egtb_result = None;

if resolved >= 4 {
    if let Some(ref egtb) = self.egtb {
        if let Some(res) = egtb.probe(&sim_board, zobrist_constants) {
            egtb_result = Some(res);
            let active_is_p1 = sim_board.x_to_move;
            proof = match res {
                GameResult::P1Win => if active_is_p1 { ProofStatus::ProvenWin } else { ProofStatus::ProvenLoss },
                GameResult::P2Win => if !active_is_p1 { ProofStatus::ProvenWin } else { ProofStatus::ProvenLoss },
                GameResult::Tie => ProofStatus::ProvenDraw,
                GameResult::Unresolved => ProofStatus::Unproven,
            };
        }
    }
}
```
- If `proof != ProofStatus::Unproven`:
  - Skip Monte Carlo rollout simulations entirely.
  - Determine reward:
    - P1Win: `1.0` if `action.x_to_move`, else `0.0`.
    - P2Win: `1.0` if `!action.x_to_move`, else `0.0`.
    - Tie: `0.5`.
  - Increment visits, record reward, set node proof status, and call `update_proof_status` along the traversal parent path.

### 1.3 Prioritized Move Expansion
- `UmctsV3Node::new(board, legal_moves)` accepts `&UltimateTicTacToeBoard`.
- Candidate legal moves are sorted in ascending order of `score_move_quick(board, m)` so that top-tier moves pop first:
  - Subboard capture: $+100$
  - Threat block: $+60$
  - Center square: $+15$
  - Sending to completed board: $-20$

### 1.4 Ply-0 Opening Book
- Direct check: `self.book.lookup(board, zobrist_constants)`.
- Returns immediately in $< 1$ $\mu$s if found and legal.

---

## 2. Experimental Verification Plan
- **Baseline Matchup**: Paired tournament matches against baseline `AlphaBetaBot` and `UmctsV2Bot` evaluated via Glicko-2 Bayesian ratings.
- **Stopping Criterion**: Executed in single accumulated rating period until Rating Deviation $RD \le 35.0$ (or safety cutoff of 500 games).
- **Hypothesis Decision**: Declared successful if $r' - 1.96 \times RD' > 1500.0$.
