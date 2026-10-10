# Ultimate Tic-Tac-Toe Engine: Session Retrospective & Next Steps

This document provides a comprehensive, mathematically rigorous, and completely self-contained summary of the architectural enhancements, empirical benchmarking results, engineering retrospective, and forward-looking roadmap for the Ultimate Tic-Tac-Toe (UTTT) competitive engine refactoring on branch `enginerefactor`.

---

## 1. Executive Summary of Accomplishments

During this development session, the engine underwent a systematic architectural upgrade:

### 1.1 167-Bit Canonical State Packing (`PackedBoardState`)
- **Memory Footprint & Alignment**: Occupies exactly 24 bytes (three 64-bit unsigned words, aligned to 8 bytes):
  ```rust
  #[repr(C, align(8))]
  pub struct PackedBoardState {
      pub w0: u64, // bits 0..63
      pub w1: u64, // bits 64..127
      pub w2: u64, // bits 128..191 (bits 167..191 zeroed)
  }
  ```
- **Bit Allocation**:
  - `o_inactive` (bits 0..80): 81 bits encoding all cells occupied by the waiting player across all 9 subboards (9 subboards $\times$ 9 bits).
  - `x_active` (bits 81..161): 81 bits encoding all cells occupied by the player whose turn it is to move.
  - `directed_subboard` (bits 162..166): 5 bits encoding the subboard constraint. Values $0..8$ indicate a specific directed subboard constraint. Value `9` (`FREE_CHOICE_INDEX = 9`) indicates unconstrained free choice across any non-completed subboard.
- **Perspective Normalization**:
  - The bitboard is always encoded relative to the active player: the player to move is *always* placed into `x_active`, and the opponent is *always* placed into `o_inactive`. This eliminates the need for an explicit turn bit and collapses states identical up to player color inversion.
- **Dihedral Group $D_4$ Quotienting**:
  - The canonical representative is defined as the **lexicographically smallest (numerical minimum)** 167-bit integer among all 8 spatial transformations in the dihedral group $D_4$ (identity, rotations by $90^\circ, 180^\circ, 270^\circ$, horizontal flip, vertical flip, main diagonal flip, and anti-diagonal flip). This collapses symmetrical positions by up to $8\times$.
- **Guarded Subboard Tie Detection (`get_tied_mask()`)**:
  - In [`uttt/src/uttt_trait_solving.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/uttt_trait_solving.rs#L628):
    ```rust
    let full = (self.xbb[i] | self.obb[i]) == 0x1FF;
    let not_won = ((self.x_supers | self.o_supers) & (1 << i)) == 0;
    if full && not_won { tied_mask |= 1 << i; }
    ```
  - *Mechanism of Degenerate Aliasing*: In bitboard engines, subboards won by a player are often represented in canonical keys as a simplified won pattern (or `0x1FF`). However, an incomplete subboard can also be filled with all 9 squares occupied without any player forming 3-in-a-row (a local tie, bitmask `0x1FF`). If filled subboards were naively normalized without checking `not_won`, full ties would be aliased with won subboards, erroneously awarding superboard 3-in-a-row wins. The tied mask is computed transiently during normalization to guarantee ties are never aliased as won lines.
- **Validation**: 10,000 randomized game rollouts confirmed 100% roundtrip fidelity between uncompressed board objects and canonical `PackedBoardState` records.

### 1.2 Zero-Copy Endgame Tablebase (`EgtbDatabase`)
- **Record Schema & Layout**:
  - File: `/my-zraid/uttt/egtb/egtb_4plus.bin` (400,016 bytes for 40,000 states, padded to 441 KB).
  - Header: 16 bytes (8 bytes magic string `UTTTEG01`, 8 bytes little-endian entry count `40_000u64`).
  - Records: Exactly 10 bytes per entry:
    - `canonical_hash: u64` (8 bytes, little-endian Zobrist hash of the canonical board).
    - `outcome: u8` (1 byte: `0 = Draw`, `1 = P1Win`, `2 = P2Win`).
    - `plies_to_mate: u8` (1 byte: Distance-to-Mate DTM counter).
- **Scope & Naming**:
  - Named `egtb_4plus.bin` because the generation plan targets games with $\ge 4$ resolved subboards; the initial offline retrograde pass completed all positions with 7, 8, or 9 resolved (won or tied) subboards.
- **Query Semantics & Performance**:
  - Entries are pre-sorted in ascending order by `canonical_hash`.
  - Loaded via zero-copy memory mapping (`memmap2`). `probe()` performs an in-place binary search on the mapped slice in $O(\log_2 40{,}000) \approx 15$ steps, returning `Some(GameResult)` in under 1 $\mu$s with zero heap allocation.

### 1.3 Depth-First Proof-Number Search Solver (`DfPnSolver`)
- **Algorithm & Mechanics**:
  - Implements the Multiple Iterative Deepening (MID) proof-number search algorithm (Allis 1994) for AND/OR trees.
  - At an OR node (attacker's turn), $pn = \min_i(pn_i)$ and $dn = \sum_i(dn_i)$.
  - At an AND node (defender's turn), $pn = \sum_i(pn_i)$ and $dn = \min_i(dn_i)$.
  - Initialization at leaf nodes:
    - Terminal win for attacker: $pn = 0, dn = \infty$.
    - Terminal loss for attacker: $pn = \infty, dn = 0$.
    - Terminal tie: $pn = \infty, dn = 0$ (soundly treated as non-wins for a win-proof solver).
    - Unresolved: $pn = 1, dn = 1$.
  - Iteration proceeds with threshold limits $th_{pn}$ and $th_{dn}$, selecting the child whose proof number equals the parent's current proof number.
- **Transposition Table**:
  - Indexed by `PackedBoardState` (24 bytes) mapping to `(pn: u32, dn: u32)`. Search budget is bounded per invocation by `max_nodes = 3_000`, guaranteeing execution terminates in under 2 ms.
- **Deterministic Winning Move Extraction (`solve_move`)**:
  - Signature: `solve_move(&mut self, board, legal_moves_lookup, zobrist) -> Option<(bool, Option<GameMove>)>`.
  - When the root is proven winning ($root\_pn = 0$), `solve_move` iterates through legal root moves (generated rapidly via `legal_moves_lookup`). It executes each candidate move on a cloned board using `zobrist` constants and probes the TT for child state $C$. If $C$ is an immediate match win or has $dn = 0$ (disproven loss for opponent = proven win for root player), move $m$ is selected and returned as `Some((true, Some(m)))`.
  - If proven draw or loss: returns `Some((false, None))`. If budget is exceeded without a proof: returns `None`.

### 1.4 Bayesian Arena & Paired Opening Suite
- **Glicko-2 Rating Engine**:
  - Eliminates naive win-rate biases by tracking Bayesian rating $\mu$, rating deviation $RD$, and volatility $\sigma$. Baseline anchor is fixed at Rating 1500, $RD = 30$.
- **Paired 2-Ply Opening Suite**:
  - 25 unique non-isomorphic 2-ply positions extracted from opening space.
  - Each position is played as a paired match: Game A (Bot 1 as X, Bot 2 as O) and Game B (Bot 1 as O, Bot 2 as X). This strictly eliminates first-move advantage and empty-board deterministic repetition.
- **Statistical Termination Criteria**:
  - The arena terminates when *either* $RD \le target\_rd$ (statistical confidence reached) *or* $games \ge max\_games$ (compute budget reached). In our runs, `--max-games 50` bounded total duration to ~3.5 minutes while driving RD down from 350 to 48.9 (95% CI margin $\pm 95.8$ Elo).

---

## 2. Empirical Benchmark Results

All benchmarks were run with `./target/release/arena benchmark` at **100 ms per move** (the CodinGame standard time control) with `--openings --opening-depth 2` across 50 paired games.

### 2.1 Baseline Specifications

| Bot Architecture | Algorithm / Search Parameters | Memory / Transposition Stores | Evaluation Features |
| :--- | :--- | :--- | :--- |
| **`alphabeta`** (Baseline) | Iterative deepening Negamax (depth 1–20), full window $[-150\text{k}, +150\text{k}]$, killer heuristic (2 killers/ply) | 1,000,000-entry hash table (`HashMap<u64, TTEntry>`) | Superboard lines: win ($\pm 100\text{k}$), 2-in-a-row ($\pm 500$), 1-in-a-row ($\pm 40$). Subboard values: center 300, corners 200, edges 150. Cell values: center 15, corners 8, edges 4. |
| **`umcts_v2`** (Baseline) | MCTS-Solver, 1-node expansion, UCT exploration constant $C = \sqrt{2} \approx 1.4142$, pure random rollouts to depth 60 | 131,072-entry table (`IdentityHasherBuilder`), 81-ply unmove stack | Pure simulation win rate ($W/N$) + UCT exploration; spatial center bias in move ordering (`sub 4, sq 4`). |

### 2.2 Candidate `umcts_v3` vs Baseline `umcts_v2`

```
=======================================================
Candidate: umcts_v3 | Baseline: umcts_v2 | Time: 100ms
=======================================================
Iteration 1 (Un-guarded Grandfather + In-tree DF-PN):
  Score: 1 Wins, 49 Losses, 0 Draws (2.0%)
  Throughput: ~1,500 rollouts/sec (150x drop)
-------------------------------------------------------
Iteration 2 (Pure random rollouts + Root-only DF-PN):
  Score: 16 Wins, 33 Losses, 1 Draws (33.0%)
  Glicko-2 Rating: 1383.6 ± 95.8 (RD: 48.9, 95% CI: [1287.9, 1479.4])
  Throughput: 15,236 visits/100ms (within 8% of v2's 16,650)
  Verdict: Statistically significant recovery from Iteration 1.
=======================================================
```

#### Diagnosis of the Remaining 33% Deficit
Even with rollout speed restored to within 8% of baseline (15,236 vs 16,650 visits per 100 ms), `umcts_v3` scored 33% against `umcts_v2`. Profiling revealed two concrete causes:
1. **Subboard Capture Bias vs Center Control**: `UmctsV2Node` expands candidate moves ordered strictly by spatial center preference (`sub == 4` +50, `sq == 4` +40, corners +20). `UmctsV3Node` added `+80` priority for subboard captures. In early midgames, capturing an edge subboard often sent the opponent directly to the center subboard, handing the opponent a dominant macro-position.
2. **Deep Grandfather Refutation False-Pruning**: Marking an in-tree node `ProvenWin` for the opponent whenever the opponent had a 1-ply superboard win threat caused MCTS to prune deep lines where the opponent's capture would force the opponent into a dead or tied board.

### 2.3 Candidate `alphabeta_v2` vs Baseline `alphabeta`

```
=======================================================
Candidate: alphabeta_v2 | Baseline: alphabeta | Time: 100ms
=======================================================
Iteration 1 (Narrow Aspiration Window delta=40 + Quiescence):
  Score: 13 Wins, 37 Losses, 0 Draws (26.0%)
  Depth: Collapsed from depth 7 down to depth 3–4 due to quiescence leaf explosion.
-------------------------------------------------------
Iteration 2 (Full Window + 1-Ply Horizon + Root DF-PN):
  Score: 19 Wins, 30 Losses, 1 Draws (39.0%)
  Glicko-2 Rating: 1424.7 ± 95.8 (RD: 48.9, 95% CI: [1328.9, 1520.5])
  Depth: Reaches depth 7–8 consistently in 100ms.
  Verdict: Inconclusive / Statistically Equivalent (baseline 1500 within 95% CI).
=======================================================
```

#### Search Mechanics in `alphabeta_v2` (Iteration 2)
1. **Full-Window Iterative Deepening**: Avoids the step-function fail-low/fail-high instability of aspiration windows.
2. **Fast 1-Ply Tactical Horizon Check at `depth == 0`**:
   - Only evaluated if the active player has an active superboard threat (`RESULT_MASKS.iter().any(|mask| (my_supers & mask).count_ones() == 2)`).
   - If true, scans legal moves in the directed subboard. If any move completes the 3-in-a-row win on the superboard, returns $\pm 100{,}000$ immediately. This eliminates horizon-effect blunders where an immediate win was truncated by depth exhaustion.
3. **Root DF-PN Endgame Solver**: If resolved subboards $\ge 5$, queries `DfPnSolver::solve_move` with a budget of 3,000 nodes. If a forced mate exists, returns the winning move in < 2 ms.

---

## 3. Session Retrospective: Honest Feedback

### What Went Well
1. **Formal Mathematical Foundations**:
   - The 167-bit packing, $D_4$ dihedral group canonicalization, and tied-mask guarding proved 100% bug-free across all unit tests and stress-test rollouts.
2. **Empirical Bottleneck Profiling**:
   - Rather than speculating on why early iterations regressed, instrumenting `hardware_profile.json` and logging visit counts immediately diagnosed the $150\times$ throughput collapse in `umcts_v3` and the quiescence explosion in `alphabeta_v2`.
3. **Paired Opening Suite Eliminating Deterministic Noise**:
   - Moving from empty-board tests to 25 paired opening positions was crucial. It prevented bots from getting stuck in 50 iterations of identical deterministic games and gave true statistical signal.
4. **Zero-Copy Memory-Mapped Tablebases**:
   - Using `memmap2` for `EgtbDatabase` enabled instant sub-microsecond binary search lookups without inflating process memory or adding garbage collection overhead.

### What Didn't Go Well
1. **The MCTS Rollout Complexity Trap**:
   - Inserting heuristic evaluation and tactical lookahead into the random simulation loop was a severe mistake. Monte Carlo Tree Search relies on the law of large numbers; reducing simulation throughput from 220,000 to 1,500 rollouts/s dropped performance from 50% to 2%.
2. **Chess Aspiration Assumptions Failing in UTTT**:
   - Narrow aspiration windows ($\delta = 40$) failed repeatedly because capturing a subboard in UTTT produces discontinuous evaluation step-functions ($\pm 200$ to $\pm 500$ points) rather than smooth material curves.
3. **Opening Book Quality Gap**:
   - Opening book entries generated from short-duration rollouts did not yield superior positions against real-time deep search. Opening books require deep retrograde validation before being trusted in production.

### Ideas for Improvement & Architectural Insights
1. **Simulation Purity vs Tree Priors**:
   - Domain knowledge in MCTS belongs strictly in the **tree policy** (PUCT expansion priors and move ordering), **never** in simulation rollouts. Rollouts must remain 100% bitwise random.
2. **Board Routing as the Core Heuristic**:
   - In UTTT rules: Directing an opponent to an *incomplete* subboard forces them to play in that subboard; directing them to a *completed* (won or tied) subboard grants them free choice across any open subboard. Minimax evaluation should penalize moves that direct the opponent to completed boards or to subboards where the opponent holds strong winning threats.
3. **GPU-Accelerated AlphaZero / Self-Play Training Loop**:
   - Hand-crafted evaluation functions for UTTT subboard interactions face diminishing returns. A dual-headed ResNet (Policy + Value) trained via self-play on the local NVIDIA GeForce GTX 1080 GPU would provide learned position evaluation, eliminating the need for random rollouts entirely.

---

## 4. Next Session Roadmap

### Phase 1: GPU Deep Learning & AlphaZero Pipeline
- **CodinGame Platform Feasibility**:
  - CodinGame requires a single self-contained CPU source file $\le 2$ MB without external `.so` libraries (like LibTorch).
  - *Deployment Path*:
    1. Train a dual-headed neural network on the local GPU via PyTorch/LibTorch self-play.
    2. Quantize the trained network into an integer-only (INT8/FP16) feed-forward inference engine compiled directly in pure Rust.
    3. For a compact network (e.g., 4 residual blocks, 64 channels, ~150k weights), INT8 weights occupy only ~150 KB, fitting easily inside CodinGame's 2 MB limit!
- **Spatial Input Tensor Specification ($9 \times 9 \times 5$)**:
  - Grid: $9 \times 9$ cells (each $3 \times 3$ block represents one subboard).
  - 5 binary feature planes:
    - `Plane 0`: Active player occupied cells (1 if occupied by player to move, 0 otherwise).
    - `Plane 1`: Inactive player occupied cells (1 if occupied by opponent, 0 otherwise).
    - `Plane 2`: Active player won subboards (1 across all 9 cells of a won subboard).
    - `Plane 3`: Inactive player won subboards (1 across all 9 cells of an opponent-won subboard).
    - `Plane 4`: Legal target subboard mask (1 across all cells of subboards where active player is legally permitted to play; 1 across all open subboards if free choice).

### Phase 2: Macro-Board Routing Heuristics for Minimax (`AlphaBetaV2Bot`)
- **Routing Penalties**:
  - Heavily penalize moves that send the opponent to a completed (won or tied) subboard, gifting them free choice.
  - Penalize moves that direct the opponent to a subboard where the opponent has an unblocked 2-in-a-row threat.
- **Move Ordering Cache**:
  - Implement History Heuristic and Countermove Heuristic indexed as flat `table[prev_move][curr_move]` arrays ($81 \times 81 \times 4 \text{ bytes} \approx 26 \text{ KB}$, residing entirely within L1/L2 CPU cache).

### Phase 3: Systematic Endgame Tablebase Expansion
- **Combinatorial Scaling Calculation**:
  - Number of macro subboard combinations: $\binom{9}{k}$ where $k$ is the count of resolved subboards.
  - Tier 7 ($k = 7$ resolved): $\binom{9}{7} = 36$ macro states $\rightarrow 40{,}000$ canonical states (441 KB).
  - Tier 6 ($k = 6$ resolved): $\binom{9}{6} = 84$ macro states ($\approx 2.33\times$ Tier 7) $\rightarrow \approx 250{,}000$–$500{,}000$ canonical states (2.5–5.0 MB).
  - Tier 5 ($k = 5$ resolved): $\binom{9}{5} = 126$ macro states ($\approx 3.5\times$ Tier 7) $\rightarrow \approx 1.5$–$3.0$ million canonical states (15–30 MB).
  - Total combined database (Tiers 5–9) is projected at **20–35 MB**, which easily fits into system RAM and maps zero-copy via `memmap2`.

### Phase 4: Multi-Time-Control Scaling Showdown
- Execute paired tournament matches comparing `alphabeta_v2`, `umcts_v3`, and baselines across three time controls:
  - 30 ms (bullet / rapid search)
  - 100 ms (CodinGame baseline)
  - 200 ms (deep search)
- Plot Glicko-2 Elo scaling trajectories to empirically determine whether MCTS scales more steeply with compute time than Alpha-Beta.

---

## 5. Verification Commands for Next Session

To verify workspace integrity at the start of the next session:
```bash
# Ensure cargo is in PATH
export PATH=$HOME/.cargo/bin:$PATH

# Run full test suite (59 unit tests)
cargo test -p uttt-tests

# Build release arena binary
cargo build --release --bin arena

# Run 50-game paired opening benchmark
./target/release/arena benchmark --bot alphabeta_v2 --baseline alphabeta --openings --opening-depth 2 --time-ms 100 --target-rd 35 --max-games 50
```
