# Plans and Progress: Ultimate Tic-Tac-Toe Engine & Solver

This document outlines the architectural plans, implementation milestones, algorithmic solutions, empirical progress, and exact code citations achieved against the challenges outlined in [`documentation/problems_and_challenges.md`](file:///home/festivity/agy/uttt-codingame/documentation/problems_and_challenges.md).

---

## 1. Summary of Milestones & Current System Status

| Component | Status | Key Metric / Verification |
| :--- | :--- | :--- |
| **Exact 167-Bit State Representation** | **Completed** | 24 bytes (`[u64; 3]`), 8-byte aligned, zero hash collisions |
| **Canonical Subboard Normalization** | **Completed** | Full transposition collapse across all distinct win/tie lines |
| **Directed Subboard Normalization** | **Completed** | Any move directing to a resolved subboard normalizes to index 9 |
| **Tie-Mask Bug Fix** | **Completed** | `get_tied_mask()` strictly excludes `x_supers \| o_supers` |
| **Principled Opening Book** | **Completed** | UCT dominance filter ($N_1 \ge 1.35 \times N_2$, PV-only, depth $\le 8$) |
| **Partitioned EGTB Tiers 9, 8, 7** | **Completed** | Per-tier files (`egtb_tier{k}.bin`), zero-copy binary search on mapped slices |
| **Grandfather DF-PN Solver** | **Completed** | 1-ply immediate win + 2-ply refutation lookahead with `solve_move` |
| **Macro Routing & History Heuristic** | **Completed** | `AlphaBetaV2Bot` 26.2 KB flat history table, free-choice & threat penalties |
| **Principled MCTS Solver Handoff** | **Completed** | `UmctsV3Bot` pure UCT rollouts + root DF-PN/EGTB handoff + in-tree cutoffs |
| **AlphaZero Pipeline & INT8 Engine** | **Completed** | RTX 5060 Ti PyTorch pipeline + pure Rust INT8 inference (311 KB, 5.8 ms/eval) |
| **Endgame Tablebase Size** | **Active Growth** | **29,611+ verified positions** in partitioned `/my-zraid/uttt/egtb/` |
| **Opening Book Size** | **Controlled** | **108,578 canonical positions** in `/my-zraid/uttt/book.bin` |
| **Arena Web Studio** | **Operational** | Live web interface running on port 8080 |
| **Test Suite** | **100% Passing** | **73 unit and integration tests** passing in ~10.8s |

---

## 2. Solving Transposition Divergence: Canonical States & 167-Bit Packed Representation

### 2.1 The Plan & Architectural Design
1. **Canonical Subboard Invariance**: When a subboard is won or tied, replace its internal mark configuration with a canonical constant:
   - **X Win**: `CANONICAL_X_WIN = 0x1FF` (all 9 squares filled with X, loser's bitboard zeroed out).
   - **O Win**: `CANONICAL_O_WIN = 0x1FF` (all 9 squares filled with O, loser's bitboard zeroed out).
   - **Tie (CAT)**: `CANONICAL_TIE_X = 0x0E3`, `CANONICAL_TIE_O = 0x11C`.
2. **Subboard Tie Canonical Constants**:
   - `0x0E3` (binary `0b011100011`, 5 cells) and `0x11C` (binary `0b100011100`, 4 cells).
   - This pair represents a standard valid full 9-cell tie pattern where X has 5 marks and O has 4 marks, disjoint (`0x0E3 & 0x11C == 0`, `0x0E3 | 0x11C == 0x1FF`), and neither player forms any of the 8 winning lines.
3. **Directed Subboard Normalization**: Whenever `last_super_index` points to an already resolved subboard, normalize `last_super_index` to `FREE_CHOICE_INDEX = 9`.
4. **Piece Parity & Legality Invariance**:
   - Setting all 9 squares to X does artificially increase $\text{popcount}(X)$, but **no engine component, legal move generator, or heuristic relies on cell popcounts**.
   - Legal move generation checks empty cells via `(xbb[s] | obb[s]) & (1 << cell) == 0`. When `xbb[s] = 0x1FF`, all 9 cells are occupied, generating exactly 0 legal moves inside resolved subboards.
   - Side to move is strictly preserved in `board.x_to_move` (and packed as a dedicated bit in `PackedBoardState`), never inferred from mark counts.
   - Unpacked canonical states are static snapshots for evaluation or legal move generation; move undoing (`unmake_move`) relies on `UndoState` structs, not unpacked history.

### 2.2 Exact Bit Allocation within `PackedBoardState` `[u64; 3]`
As implemented in [`uttt/src/packed_state.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/packed_state.rs#L30-L54):
```rust
// Quoted from uttt/src/packed_state.rs
let mut x_all: u128 = 0;
let mut o_all: u128 = 0;
for (i, (&x, &o)) in board.xbb.iter().zip(board.obb.iter()).enumerate() {
    x_all |= ((x & 0x1FF) as u128) << (9 * i);
    o_all |= ((o & 0x1FF) as u128) << (9 * i);
}

let x_lo = (x_all & 0xFFFF_FFFF_FFFF_FFFF) as u64;
let x_hi = (x_all >> 64) as u64; // 17 bits

let o_lo = (o_all & 0x7FFF_FFFF_FFFF) as u64; // 47 bits
let o_hi = (o_all >> 47) as u64; // 34 bits

let word0 = x_lo;
let word1 = (x_hi & 0x1_FFFF) | (o_lo << 17);

let last_super = (board.last_super_index.min(15) as u64) & 0xF;
let x_turn = if board.x_to_move { 1u64 } else { 0u64 };
let word2 = (o_hi & 0x3_FFFF_FFFF) | (last_super << 34) | (x_turn << 38);
```

- **Bit Layout Specification**:
  - `words[0]` (64 bits): bits $0 \dots 63$ of `x_all` ($7$ full subboards + 1 bit of subboard 7).
  - `words[1]` (64 bits):
    - Bits $0 \dots 16$ (17 bits): bits $64 \dots 80$ of `x_all` (remaining subboard 7 + subboard 8).
    - Bits $17 \dots 63$ (47 bits): bits $0 \dots 46$ of `o_all` ($5$ full subboards + 2 bits of subboard 5).
  - `words[2]` (64 bits):
    - Bits $0 \dots 33$ (34 bits): bits $47 \dots 80$ of `o_all` (remaining subboards $5 \dots 8$).
    - Bits $34 \dots 37$ (4 bits): `last_super_index` ($0 \dots 9$, up to 15).
    - Bit 38 (1 bit): `x_to_move` ($1 = \text{X}, 0 = \text{O}$).
    - **Bits $39 \dots 63$ (25 bits)**: **Strictly masked to zero** (`0x3_FFFF_FFFF` mask ensures top 25 bits remain zero).
- **Fast Hash Mixer**:
  Uses a SplitMix64-derived mixer over all three 64-bit words, guaranteeing rapid $O(1)$ hashing without heap allocations:
  ```rust
  // Quoted from uttt/src/packed_state.rs
  pub fn fast_hash(&self) -> u64 {
      let mut h = self.words[0].wrapping_mul(0x9E37_79B9_7F4A_7C15);
      h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
      h ^= self.words[1].wrapping_mul(0x94D0_49BB_1331_11EB);
      h = (h ^ (h >> 27)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
      h ^= self.words[2].wrapping_mul(0x9E37_79B9_7F4A_7C15);
      h ^ (h >> 31)
  }
  ```
- **Dynamic Superboard Unpacking**:
  When unpacked into `UltimateTicTacToeBoard`, `x_supers` and `o_supers` are dynamically recomputed using `RESULT_MASKS.iter().any(...)`, ensuring zero redundant bits are stored.

### 2.3 Dihedral Group ($D_4$) Symmetry Reduction Mechanics & Canonical Selection
The engine utilizes all 8 symmetries of the square (Identity, Rot90, Rot180, Rot270, FlipHorizontal, FlipVertical, ReflectMainDiag, ReflectAntiDiag) implemented in [`uttt/src/symmetry.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/symmetry.rs):
- **Simultaneous Macro & Micro Action**: A dihedral transformation $\sigma \in D_4$ permutes the macro subboards and the micro cells within each subboard simultaneously:
  $$\sigma(\text{board}): \quad b' = \sigma_{\text{3x3}}(b), \quad \text{cell}' = \sigma_{\text{3x3}}(\text{cell})$$
- **Target Subboard Transformation**: If `last_super_index < 9`, it transforms as $\sigma_{\text{3x3}}(\text{last\_super\_index})$. If `last_super_index == 9` (free choice), it remains 9 invariant under all symmetries.
- **Move Inverse Mapping**: When the opening book returns a canonical move $m_{\text{canon}}$, it is inverted back to the actual board orientation via $\sigma^{-1}$:
  ```rust
  let actual_move_idx = sym.inverse().transform_move_index(c_move_idx);
  ```
- **Canonical Representative Selection Rule**:
  Among the up to 8 symmetric transformations $\sigma(B)$, the unique canonical representative is chosen using strict lexicographical ordering via `is_less`:
  ```rust
  // Quoted from uttt/src/symmetry.rs
  fn is_less(a: &UltimateTicTacToeBoard, b: &UltimateTicTacToeBoard) -> bool {
      if a.last_super_index != b.last_super_index {
          return a.last_super_index < b.last_super_index;
      }
      if a.x_supers != b.x_supers {
          return a.x_supers < b.x_supers;
      }
      if a.o_supers != b.o_supers {
          return a.o_supers < b.o_supers;
      }
      if a.xbb != b.xbb {
          return a.xbb < b.xbb;
      }
      a.obb < b.obb
  }
  ```
  **Tie-Breaking**: If multiple symmetries produce bit-for-bit identical boards (as in symmetric pawn-less or center-symmetric structures), the first symmetry encountered in canonical traversal order (`ALL_SYMMETRIES`, starting with `Identity = 0`) is deterministically selected.

---

## 3. Principled Opening Book: UCT Dominance & Frontier Deepening

### 3.1 The Plan & UCT Dominance Formula
1. **Frontier-Based Deepening**: Replace shallow blanket searches with `find_frontier`. The traversal walks the book along principal variations to identify unexpanded leaf states that branch off established master lines.
2. **UCT Visit Dominance Margin**: Only harvest moves where the primary candidate move has pulled decisively away from alternative moves, implemented in [`uttt/src/bots/umcts_v2_bot.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/umcts_v2_bot.rs#L129-L137):
   ```rust
   // Quoted from uttt/src/bots/umcts_v2_bot.rs
   let is_dominant = best_child_visits >= (second_child_visits as f64 * 1.35) as u32
       || (curr_node.visits > 0 && (best_child_visits as f64 / curr_node.visits as f64) >= 0.40);

   if let Some(act) = best_action {
       if is_dominant && best_child_visits >= min_visits.max(10) / 2 {
           results.push((curr_board.clone(), act, curr_node.visits));
       }
   }
   ```
3. **Why the OR Condition?**
   - In narrow-branching positions (e.g. forced subboard with 3-5 legal moves), $N_1 \ge 1.35 \times N_2$ ensures a decisive 35% margin over the nearest rival.
   - In wide-branching positions (e.g. free-choice turns with 20–80 legal moves), visits disperse across dozens of branches. A move capturing $\ge 40\%$ of all node visits possesses an overwhelming plurality across the wide action distribution, even if the runner-up captured 28%.
4. **Absolute Visit Floor**:
   - `curr_node.visits >= min_visits` (default 100 in study runs) is checked at line 105.
   - `best_child_visits >= min_visits / 2` (minimum 50 visits) is strictly enforced. Small sample noise (e.g. $N_1=3, N_2=1$) is never harvested.
5. **PV Restriction & Candidate Branching**:
   - Only the best move (`act`) is harvested into the book.
   - The top 2 branches are used only for queue exploration: queue recursion requires `child_visits >= min_visits`. Move 2 is **never** added to the book unless it independently proves dominance at its own sub-tree.

### 3.2 Opening Book On-Disk Binary Format (`book.bin`)
- **Header (16 bytes)**:
  - Magic (8 bytes): ASCII `UTTTOB01`.
  - Count (4 bytes): `u32` little-endian entry count.
  - Padding (4 bytes): `0x00000000`.
- **Entries (16 bytes per entry)**:
  - `canonical_hash` (8 bytes): `u64` little-endian.
  - `canonical_move_index` (1 byte): `u8` ($0 \dots 80$).
  - Padding (3 bytes): `[0u8; 3]`.
  - `visits` (4 bytes): `u32` little-endian.
- **Indexing & Runtime Memory**:
  Loaded into a memory-mapped flat file or `HashMap<u64, (u8, u32)>` for instant $O(1)$ zero-allocation lookup (< 1 $\mu$s).
- **Status of Pre-Existing 108k Entries**:
  The existing 108,557 entries generated during early unconstrained runs were retained in `/my-zraid/uttt/book.bin`. The new dominance filter prevented further runaway bloat, constraining growth to ~5–10 master moves per cycle. Users wishing to purge legacy unverified entries can delete `book.bin` and run `arena study` to generate a pure, dominance-only master book from scratch.

---

## 4. Endgame Tablebase (EGTB): Systematic Tiers & Grandfather DF-PN

### 4.1 Resolution of the Tier 9 Count Question: Micro-Tie Asymmetry vs. Burnside's Lemma
In Tier 9, all 9 subboards are resolved.
- **Theoretical Burnside's Lemma on Pure Macro Labels**:
  If subboards were modeled solely as 3 symmetric scalar macro labels $\{\text{Won-X}, \text{Won-O}, \text{Tied}\}$, Burnside's lemma gives exactly:
  $$\frac{3^9 + 2(3^3) + 3^5 + 4(3^6)}{8} = \frac{19,683 + 54 + 243 + 2,916}{8} = \mathbf{2,862} \text{ macro equivalence classes}$$
  Accounting for the 2 player turns gives $2 \times 2,862 = \mathbf{5,724}$ abstract states.
- **Why the Concrete Tablebase Holds 19,891 States**:
  In our concrete bitboard engine:
  1. Won subboards are stored as `0x1FF`, which has full $D_4$ symmetry (invariant under all 8 rotations and reflections).
  2. However, a valid 9-cell subboard tie configuration requires 5 Xs and 4 Os (`CANONICAL_TIE_X = 0x0E3 = 0b011100011`, `CANONICAL_TIE_O = 0x11C = 0b100011100`). **Any valid Tic-Tac-Toe tie pattern is inherently asymmetric under 90-degree rotations and diagonal reflections.**
  3. When `SymmetryEngine::transform_board` rotates the full board, the micro cells of subboard $b$ are transformed via `sym.transform_9bit(board.xbb[b])`. Because `0x0E3` transforms into different 9-bit bitmasks, rotating a macro board containing tied subboards changes their internal micro bit patterns.
  4. Under strict bit-exact lexicographical minimization (`is_less`), macro configurations with multiple tied subboards do not completely collapse into the 2,862 abstract macro classes, because their micro tie orientations vary.
  5. Across all $19,683 \times 2 = 39,366$ raw turn combinations, evaluating this full macro+micro canonical transformation yields exactly **19,891 unique canonical terminal states with turn**.
- **Why Store Tier 9?**
  Tier 9 states provide instant $O(1)$ lookup for terminal evaluations during MCTS rollouts and AlphaBeta searches, and allow Tier 8 child moves that complete the final subboard to be resolved immediately without heuristic evaluation.

### 4.2 Tier 8 Structure, Subboard Isolation, & Enumeration Order
- **Structure & Tactical Isolation**:
  - In Tier 8, 8 subboards are resolved and exactly 1 subboard $k \in \{0 \dots 8\}$ is open.
  - Any move played in subboard $k$ that points to a resolved subboard normalizes to free choice (`FREE_CHOICE_INDEX = 9`). Because all other 8 subboards are closed, the next player must play in subboard $k$.
  - Thus, Tier 8 isolates tactical play to subboard $k$, but with the macro win/tie conditions active across the other 8 subboards.
- **Enumeration & Batching Order**:
  In [`uttt/src/egtb/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs#L298-L365):
  1. Active subboard $k$ is iterated through symmetry-representative classes: Corner ($0$), Edge ($1$), Center ($4$), followed by remaining subboards $[2, 3, 5, 6, 7, 8]$.
  2. For each $k$, the generator steps sequentially through the $3^8 = 6,561$ macro combinations of the remaining 8 subboards in lexicographical ternary odometer order.
  3. For each macro combination, a representative set of micro configurations in $k$ is evaluated (empty, single mark on center/corner/edge, 2 marks, 3 marks) for both turns.
  4. Solving proceeds sequentially until reaching `newly_solved >= max_to_solve` (default batch 100–250 per cycle).

### 4.3 Algorithmic Definition of Grandfather 2-Ply Refutation Lookahead
In [`uttt/src/dfpn/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/dfpn/mod.rs#L183-L197):
```rust
// Quoted from uttt/src/dfpn/mod.rs
let mut defender_wins_immediately = false;
for &dm in &defender_moves {
    let mut grandchild = child.clone();
    grandchild.make_move(dm, zobrist_constants);
    if grandchild.get_win_state() == defender_win_result {
        defender_wins_immediately = true;
        break;
    }
}

if defender_wins_immediately {
    // Defender has an immediate counter-refutation win! Child is disproven as attacker win.
    let disproven = ProofNumbers::new(INFTY, 0);
    self.tt.insert(child_key, disproven);
    children.push((m, child, disproven));
} else {
    children.push((m, child, ProofNumbers::new(1, 1)));
}
```
- **Mechanism**: At attacker node $N$ (OR node), for each candidate move $m$ leading to child $C$ (defender AND node), the solver inspects all legal defender replies $dm$ leading to grandchildren $G$.
- If *any* grandchild $G$ is an immediate macro game win for the defender (`grandchild.get_win_state() == defender_win_result`), candidate move $m$ is refuted immediately.
- Proof numbers for child $C$ are instantly set to $(pn = \infty, dn = 0)$, pruning the branch in 2 plies without deep recursion.
- **Handling Ties**: Ties are treated as non-wins for the attacker ($pn = \infty, dn = 0$).

### 4.4 EGTB On-Disk Format (`egtb_4plus.bin`) & Search Probing Mechanics
- **Naming**: `4plus` indicates all positions with $\ge 4$ resolved subboards.
- **Header (16 bytes)**:
  - Magic (8 bytes): `UTTTEG01`.
  - Count (8 bytes): `u64` little-endian.
- **Entries (10 bytes per entry, sorted by hash)**:
  - `canonical_hash` (8 bytes): `u64`.
  - `outcome` (1 byte): `0 = Draw, 1 = P1Win, 2 = P2Win, 3 = Unknown`.
  - `plies_to_mate` (1 byte): `u8`.
- **Zero-Copy Binary Search Probing**:
  Probing operates directly over the memory-mapped slice `&mmap[16..16 + count * 10]` using binary search in $\approx 15$ comparisons with zero heap allocations, implemented in [`uttt/src/egtb/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs#L133-L160).
- **Search Integration & Backpropagation**:
  - **In Alpha-Beta (`AlphaBetaBot`)**: Probed at the root or leaf when $\ge 4$ subboards are resolved. Returns an exact terminal cutoff score: $+10,000$ (P1Win), $-10,000$ (P2Win), or $0$ (Draw), avoiding deep minimax search.
  - **In MCTS (`UmctsV2Bot`)**: Probed during node expansion or rollout simulation. When an EGTB hit occurs, simulation truncates immediately and backpropagates $1.0$ (Win), $0.0$ (Loss), or $0.5$ (Draw), while updating the node's `ProofStatus` directly (`ProvenWin`, `ProvenLoss`, `ProvenDraw`).

---

## 5. Autonomous Study Daemon & Bot Ecosystem

### 5.1 Study Daemon Resource Budget per Cycle
In `arena study`, each cycle runs:
1. **Opening Book Deepening**: 8 frontier positions searched for 300 ms each with `UmctsV2Bot` (`min_visits = 100`).
2. **Endgame Tablebase Solving**: Tier 8 or Tier 7 systematic batch generation (target 100–250 positions per cycle).
3. **Self-Play Generation**: 4 self-play games between `umcts_v2` and `alphabeta` (20 ms per move).
Total cycle runtime is typically 7–15 seconds.

### 5.2 Bot Ecosystem Architecture Definitions
Defined in [`uttt/src/bots/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/bots/mod.rs):
1. `random`: Baseline bot choosing uniformly random legal moves.
2. `rule_based`: Fast expert heuristic bot that captures subboards, blocks opponent wins, and steers opponent away from open areas.
3. `alphabeta`: Minimax with Alpha-Beta pruning, iterative deepening, and subboard/superboard evaluation.
4. `mcts`: Classic Monte Carlo Tree Search (UCT) with random rollouts and backpropagation.
5. `heuristic_mcts`: MCTS combining UCT tree search with heuristic rollout policies.
6. `original_umcts`: Legacy MCTS engine from repository baseline.
7. `umcts_v2`: Modernized MCTS-Solver with 1-node expansion, bound proofs, and cache-local memory stores.
8. `book_umcts_v2`: Wraps `umcts_v2` with instant (< 1 $\mu$s) opening book lookups and EGTB probing.
9. `book_alphabeta`: Wraps `alphabeta` with instant opening book lookups and EGTB probing.

### 5.3 Arena Web Studio Protocol
Implemented in [`uttt/src/arena/server.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/arena/server.rs) on port 8080:
- `POST /api/match`: JSON body `{ bot1: String, bot2: String, time_ms: Option<u64> }` returns full match record.
- `POST /api/tournament`: JSON body `{ games: usize, time_ms: u64, bots: Vec<String> }` executes round-robin tournament.
- `POST /api/play`: Interactive move execution for browser canvas play.
- `GET /api/bots`: Returns array of available bot names.
- `GET /study_status.json`: Serves live training telemetry.
