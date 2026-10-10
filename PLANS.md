# Project-Wide Plans & Progress: Ultimate Tic-Tac-Toe Engine & Solver

This document outlines the overarching architectural plans, system decomposition, implementation status, and verification milestones across the entire Ultimate Tic-Tac-Toe workspace.

---

## 1. System Decomposition & Module Map

The workspace is structured into two primary Cargo crates:
1. [`uttt`](file:///home/festivity/agy/uttt-codingame/uttt): The core engine, bitboard operations, bot ecosystem, DF-PN solver, EGTB generator, and arena runner.
2. [`uttt-tests`](file:///home/festivity/agy/uttt-codingame/uttt-tests): Comprehensive unit, property, and integration test suite (55 tests) verifying engine invariants and tournament stability.

### Directory Hierarchy:
```
uttt-codingame/
├── Cargo.toml / Cargo.lock   # Workspace configuration
├── PROBLEMS.md / PLANS.md     # Root architecture & progress specs
├── documentation/             # In-depth technical specifications
│   ├── problems_and_challenges.md
│   ├── plans_and_progress.md
│   └── canonical_states_and_packed_representation.md
├── uttt/                      # Core engine crate
│   ├── Cargo.toml / build.rs  # Dependencies & native build script
│   ├── benches/               # Criterion micro-benchmarks
│   └── src/
│       ├── lib.rs             # Public crate exports
│       ├── canonical.rs       # Canonical subboard constants & normalization
│       ├── packed_state.rs    # Exact 167-bit bit-packed board state ([u64; 3])
│       ├── symmetry.rs        # D4 dihedral group transformations & canonical form
│       ├── game_solving.rs    # Universal GameMove & GameResult interfaces
│       ├── ttt_trait_solving.rs  # 3x3 single-board engine & move tables
│       ├── uttt_trait_solving.rs # 9x9 ultimate engine, bitboards, and rules
│       ├── opening_book.rs    # Frontier-deepening book with UCT dominance
│       ├── profile.rs         # Execution timing & benchmark utilities
│       ├── dfpn/              # Depth-First Proof-Number solver with Grandfather lookahead
│       ├── egtb/              # Systematic endgame tablebases (Tiers 9, 8, 7)
│       ├── bots/              # 9 diverse bot architectures (heuristic, AB, MCTS, solvers)
│       ├── arena/             # Game, match runner, tournaments, Elo, replay, web server
│       └── bin/               # Executables: arena CLI, CodinGame generator, local runner
└── uttt-tests/                # Comprehensive test crate
    ├── Cargo.toml
    └── src/
        ├── lib.rs             # Integration & regression tests
        └── canonical_state_tests.rs # Canonical collapse, packing & solver tests
```

---

## 2. Global Architectural Solutions & Status

### 2.1 167-Bit Packed State (`PackedBoardState`)
- **Structure**: Packed into exactly 24 bytes (`[u64; 3]`, 8-byte aligned) in [`uttt/src/packed_state.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/packed_state.rs).
- **Exact Bit Allocation**:
  - `words[0]` (64 bits): bits $0 \dots 63$ of 81-bit X mask.
  - `words[1]` (64 bits): bits $64 \dots 80$ of X mask (17 bits) + bits $0 \dots 46$ of O mask (47 bits).
  - `words[2]` (64 bits): bits $47 \dots 80$ of O mask (34 bits) + 4 bits target subboard ($0 \dots 9$) + 1 bit turn ($1 = \text{X}, 0 = \text{O}$).
  - Upper 25 bits of `words[2]` (bits 39..63) are **strictly masked to zero**, guaranteeing that derived `Eq`, `Ord`, and `Hash` operate without false mismatches.
- **Status**: **Completed & Verified** across 100 randomized games and transposition collapse unit tests.

### 2.2 Canonical Subboards & Directed Normalization
- **Won Subboard Overwrite**: When subboard $s$ is won by X, `xbb[s] = CANONICAL_X_WIN = 0x1FF` and `obb[s] = 0x000`. Legal move generation checks empty cells via `(xbb | obb) & (1 << cell) == 0`. Because all 9 bits are set, exactly 0 legal moves are generated inside won subboards. Superboard win lines check macro masks `x_supers` and `o_supers`, remaining fully decoupled from internal popcounts.
- **Canonical Tie Masks**: Row-major coordinates ($0 \dots 8$, where bit $i$ corresponds to row $i/3$, col $i\%3$) define `CANONICAL_TIE_X = 0x0E3 = 0b011100011` (5 cells) and `CANONICAL_TIE_O = 0x11C = 0b100011100` (4 cells). This pattern is disjoint, fills all 9 cells, and has zero 3-in-a-row lines.
- **Directed Normalization**: Any move directing to a resolved subboard normalizes to `FREE_CHOICE_INDEX = 9`.
- **Status**: **Completed & Verified**.

### 2.3 Principled Opening Book via UCT Dominance
- **UCT Dominance Formula**: Captures moves satisfying $N_1 \ge 1.35 \times N_2$ (where $N_1, N_2$ are visits of the top 2 candidate child moves) OR $N_1 / \sum_i N_i \ge 0.40$ (plurality across wide action spaces).
- **Visit Floor**: Enforces $N_{\text{node}} \ge 100$ and $N_1 \ge 50$ to eliminate low-sample noise.
- **Storage Format (`book.bin`)**: 16-byte header (`UTTTOB01` magic + `u32` count), followed by 16-byte records (`[u64 hash, u8 move_index, 3-byte pad, u32 visits]`), loaded into a flat array or `HashMap` for $< 1\,\mu\text{s}$ zero-allocation lookup.
- **Legacy Retention vs. Curation**: The on-disk file `/my-zraid/uttt/book.bin` contains 108,578 positions (originating from legacy runs and preserved for broad coverage). The new dominance filter restricts subsequent growth to ~5–10 master moves per cycle.
- **Status**: **Operational & Curated**.

### 2.4 Systematic EGTB Tiers (Tiers 9, 8, and 7)
- **Tier Definition**: Tier $k$ designates configurations where exactly **$k$ of the 9 subboards are fully resolved** (won or tied).
- **File Name (`egtb_4plus.bin`)**: Denotes the late-game tablebase storing all solved states with $\ge 4$ resolved subboards.
- **Tier 9**: All 9 subboards resolved (terminal). Evaluates $3^9 \times 2 = 39,366$ raw turn combinations, collapsing under $D_4$ dihedral symmetry into **19,891 canonical terminal states** solved in < 15 ms.
- **Tier 8**: 8 subboards resolved, 1 active. Solved using Grandfather DF-PN at > 10,000 pos/sec. Currently accounts for 9,720 solved states ($19,891 + 9,720 = 29,611$ total in `egtb_4plus.bin`).
- **Tier 7**: 7 subboards resolved, 2 active. Generator implemented in [`uttt/src/egtb/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs#L422); actively solved during extended daemon cycles.
- **Status**: **Operational** (29,611 positions verified).

### 2.5 Grandfather 2-Ply Refutation Lookahead
- At an attacker OR node, for each candidate move $m$ leading to defender AND child $C$, the solver checks all defender counter-replies leading to grandchildren $G$.
- If *any* grandchild $G$ is an immediate macro game win for the defender, candidate move $m$ is refuted on the spot. Child $C$ is assigned $(pn = \infty, dn = 0)$, pruning the branch in 2 plies without deep recursion.
- Draws/ties are soundly treated as non-wins for the attacker ($pn = \infty, dn = 0$).

### 2.6 Guarded Subboard Tie Detection (`get_tied_mask()`)
- In [`uttt/src/uttt_trait_solving.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/uttt_trait_solving.rs#L628):
  ```rust
  let full = (self.xbb[i] | self.obb[i]) == 0x1FF;
  let not_won = ((self.x_supers | self.o_supers) & (1 << i)) == 0;
  if full && not_won { tied_mask |= 1 << i; }
  ```
  Strictly prevents canonical won subboards (`0x1FF`) or subboards won on the 9th move from being misclassified as ties.

### 2.7 Complete Roster of 9 Bot Architectures
Defined in [`uttt/src/bots/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/mod.rs#L35-L47):
1. `random`: Uniformly random legal move baseline.
2. `rule_based`: Fast expert heuristic bot (subboard capture, threat blocking, opponent steering).
3. `alphabeta`: Minimax with Alpha-Beta pruning, iterative deepening, and subboard/superboard evaluation.
4. `mcts`: Classic Monte Carlo Tree Search with UCT selection and random rollouts.
5. `heuristic_mcts`: MCTS combining UCT tree search with heuristic rollout policies.
6. `original_umcts`: Legacy baseline MCTS search engine.
7. `umcts_v2`: Modernized MCTS-Solver with 1-node expansion, bound proofs, and cache-local memory stores.
8. `book_umcts_v2`: Wraps `umcts_v2` with instant (< 1 $\mu$s) opening book lookups and EGTB probing.
9. `book_alphabeta`: Wraps `alphabeta` with instant opening book lookups and EGTB probing.

### 2.8 Arena, Study Daemon, and Web Dashboard
- Multi-threaded tournament runner, Elo tracking, flamegraph CPU sampling via `pprof`.
- Background study daemon with live JSON telemetry (`study_status.json`).
- Interactive HTML5 web studio running on port 8080.
- Status: **Operational**.

---

## 3. Cross-References & Detailed Documentation
- Architectural Problems & Challenges: [`PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md)
- In-Depth Technical Specification: [`documentation/problems_and_challenges.md`](file:///home/festivity/agy/uttt-codingame/documentation/problems_and_challenges.md)
- Complete Implementation Analysis: [`documentation/plans_and_progress.md`](file:///home/festivity/agy/uttt-codingame/documentation/plans_and_progress.md)
- Core Engine Crate Specs: [`uttt/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/uttt/PROBLEMS.md) and [`uttt/PLANS.md`](file:///home/festivity/agy/uttt-codingame/uttt/PLANS.md)
- Test Suite Crate Specs: [`uttt-tests/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/uttt-tests/PROBLEMS.md) and [`uttt-tests/PLANS.md`](file:///home/festivity/agy/uttt-codingame/uttt-tests/PLANS.md)
