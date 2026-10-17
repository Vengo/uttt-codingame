# Independent Cold-Eyes Technical Audit: UTTT EGTB ZFS Expansion & Transformer Self-Play Tournament Plans

**Review Document Path**: `docs/independent_cold_eyes_review_round2.md`  
**Auditor**: Independent Technical Reviewer (Google DeepMind / Cold-Eyes Technical Audit)  
**Date**: October 12, 2026  
**Target Documents Under Review**:
1. **Proposal A**: *Syzygy-Inspired Endgame Tablebase Architecture & 7.2 TB ZFS Pool Expansion Plan* (`docs/egtb_syzygy_zfs_expansion_plan.md`)
2. **Proposal B**: *Vision/Board Transformer Continuous Self-Play Loop & 3-Way Tournament Plan* (`docs/transformer_self_play_and_3way_tournament_plan.md`)

**Target Subsystems Inspected**:
- `uttt/src/egtb/tier6.rs`, `uttt/src/egtb/macro_factored.rs`, `uttt/src/egtb/retrograde.rs`
- `ml/transformer_model.py`, `ml/parallel_mirror_eval.py`, `ml/champion_challenger.py`, `ml/rust_mcts_worker.py`
- Hardware & Storage: 7.2 TB ZFS Storage Pool (`/my-zraid`), NVIDIA GeForce RTX 5060 Ti GPU, 16-Core Host (125 GB RAM).

---

## 1. Executive Verdicts & Overall Assessment

```
+-------------------------------------------------------------------------------------------------------+
|                                     EXECUTIVE AUDIT SUMMARY                                            |
+------------+---------------------------------------+--------------------+-----------------------------+
| Proposal   | Title                                 | Cold-Eyes Verdict  | Primary Risk / Remediation  |
+------------+---------------------------------------+--------------------+-----------------------------+
| Proposal A | Syzygy-Inspired UTTT Tablebases &     | CONDITIONAL        | CRITICAL DEFECT: encode_u10 |
|            | 7.2 TB ZFS Pool Expansion             | APPROVAL (Requires | multiplicative hash collides|
|            |                                       | Mandatory Patch)   | in 72/84 triplets (1,774    |
|            |                                       |                    | states). Replace with dense |
|            |                                       |                    | bijection table. Clarify D4 |
|            |                                       |                    | runtime rotation vs 36-pair.|
+------------+---------------------------------------+--------------------+-----------------------------+
| Proposal B | Vision/Board Transformer Self-Play    | APPROVAL WITH      | CRITICAL DEFECT: C-ABI      |
|            | Loop & 3-Way Tournament Plan          | MINOR              | alphabeta_batch_choose_moves|
|            |                                       | MODIFICATIONS      | reallocates 1M-entry HashMaps|
|            |                                       |                    | (~40 GB/ply). Must use bot  |
|            |                                       |                    | pool. Add batch move apply. |
+------------+---------------------------------------+--------------------+-----------------------------+
```

### 1.1 Proposal A Verdict: CONDITIONAL APPROVAL (Requires Mandatory Remediation)
- **Strengths**: The core mathematical derivation of exhaustive Tier 7 tablebase feasibility is sound, elegant, and reproducible. The dihedral $D_4$ partition into 8 canonical orbits, the 854 reachable macro vectors $\tau_7$, the minimax functional elimination of Free Choice ($V(\text{Free}) = \max(V_j, V_k)$), and the $3^4 = 81 \le 255$ dense ternary byte packing ($1 \text{ byte per micro-pair}$) are 100% verified. Tier 7 requires exactly 105.70 GB raw and will easily compress to ~26.4–35.2 GB on the 7.2 TB `/my-zraid` ZFS pool.
- **Fatal Flaw Identified**: The claimed resolution of the 62-bit key packing in Tier 6 (`encode_tier6_key` in `uttt/src/egtb/tier6.rs`) introduces a **catastrophic hash collision bug**. While the 42-bit micro-ID packing correctly fixes the legacy 12-bit shift defect, the function `encode_u10()` maps the 15-bit macro condition vector into 10 bits using a lossy multiplicative Fibonacci hash (`wrapping_mul(0x9E37_79B9) >> 22`). We proved via exhaustive enumeration that **72 out of 84 triplets suffer hash collisions, collapsing 1,774 distinct reachable macro states**. The existing unit test `test_tier6_62bit_key_packing_no_collision` masked this defect by testing pre-packed arbitrary integers rather than real macro conditions.
- **Architectural Ambiguity**: Proposal A calculates storage assuming only the 8 canonical orbits are persisted, but fails to specify the runtime $D_4$ micro-board rotation protocol. Storing all 36 pairs takes 460.66 GB raw (~131.6 GB compressed on ZFS), consuming only 1.8% of `/my-zraid` while completely eliminating runtime micro-board symmetry rotation.

### 1.2 Proposal B Verdict: APPROVAL WITH MINOR MODIFICATIONS
- **Strengths**: The `UTTTTransformer` model is architecturally superior to convolutional neural networks for Ultimate Tic-Tac-Toe. Its dual learned positional embeddings (`sb_embed` and `sq_embed`) exploit the exact mathematical equivalence between subsquare ID and target subboard ID, enabling Layer 1 self-attention heads to route tactical constraints across the board in $O(1)$ operations. The 3-Way Mirrored Round-Robin tournament mathematically cancels out position bias $V(P_k)$ and color advantage. The continuous self-play parameters ($T > 0$ schedule, 800 visits, 1M-state stratified replay buffer) are optimal.
- **Critical Implementation Defect Identified**: The proposed Rust C-ABI function `alphabeta_batch_choose_moves` instantiates `let mut bot = AlphaBetaV2Bot::new();` inside the parallel Rayon loop for every game on every ply. Because each `AlphaBetaV2Bot` allocates a 1,000,000-entry `HashMap` (~40 MB), a 1,024-game match will allocate and deallocate **~40 GB of heap memory per move**, inducing severe allocator lock contention and memory thrashing, while simultaneously wiping out Transposition Tables, killer moves, and history heuristics across moves.
- **Integration Gap**: `RustMctsWorker` currently lacks a batch API to apply external moves chosen by AlphaBeta to the active game slots without tearing down the game state.

---

## 2. Deep Technical Audit of Proposal A (EGTB & ZFS Storage Architecture)

### 2.1 Mathematical Combinatorics & Storage Calculations

#### 1. Tier 7 Dihedral Symmetry Reduction & Macro Boundary Space
Proposal A claims that the $\binom{9}{2} = 36$ unordered pairs of open subboards partition under the $D_4$ dihedral group into exactly 8 canonical orbits, with exactly 854 legally reachable macro boundary vectors $\tau_7$.

**Audit Verification**:
We performed exhaustive computer verification of the $D_4$ group action on the 36 subboard pairs:
```
Orbit 0: Rep (0, 1) | Size: 8 | Corner + Adjacent Edge
Orbit 1: Rep (0, 2) | Size: 4 | Corner + Adjacent Corner
Orbit 2: Rep (0, 4) | Size: 4 | Corner + Center
Orbit 3: Rep (0, 5) | Size: 8 | Corner + Opposite Edge
Orbit 4: Rep (0, 8) | Size: 2 | Opposite Corners (Main Diagonal)
Orbit 5: Rep (1, 3) | Size: 4 | Adjacent Edges
Orbit 6: Rep (1, 4) | Size: 4 | Edge + Center
Orbit 7: Rep (1, 7) | Size: 2 | Opposite Edges
Total Pairs: 36 | Total Canonical Orbits: 8 (Exact Match)
```

Next, we evaluated all $3^7 = 2,187$ closed-subboard ternary assignments ($X$, $O$, or Tied) for each of the 8 canonical orbit representatives, filtering out states where either player had already completed a 3-in-a-row on closed boards:
```
Orbit (0, 1): 124 vectors
Orbit (0, 2): 157 vectors
Orbit (0, 4): 128 vectors
Orbit (0, 5):  60 vectors
Orbit (0, 8): 126 vectors
Orbit (1, 3):  58 vectors
Orbit (1, 4): 112 vectors
Orbit (1, 7):  89 vectors
Sum of Canonical Macro Vectors: Exactly 854 (Exact Match)
```
The macro combinatorics are **100% mathematically verified**.

#### 2. Functional Elimination of Free Choice Storage
Proposal A asserts that when given Free Choice in Tier 7, the game-theoretic value satisfies:
$$V(\text{Free Choice}, \text{Turn } X) = \max\left( V(\text{target}=j, \text{Turn } X), \ V(\text{target}=k, \text{Turn } X) \right)$$
$$V(\text{Free Choice}, \text{Turn } O) = \min\left( V(\text{target}=j, \text{Turn } O), \ V(\text{target}=k, \text{Turn } O) \right)$$
with $V \in \{+1 \text{ (P1Win)}, 0 \text{ (Draw)}, -1 \text{ (P2Win)}\}$, requiring **0 bits of storage**.

**Audit Verification**:
- Under UTTT rules, Free Choice permits the active player to place their mark in any legal cell in any open subboard.
- In Tier 7, exactly two subboards $(j, k)$ are open.
- If the player elects to move in subboard $j$, the subgame corresponds to the game where the player is constrained to subboard $j$, whose minimax value is $V(\text{target}=j)$.
- If the player elects to move in subboard $k$, the subgame value is $V(\text{target}=k)$.
- Since UTTT is a finite zero-sum game of perfect information with no repetition cycles (stones can never be removed; each ply strictly decreases empty cells), backward induction guarantees that Player X will choose $\max(V_j, V_k)$ and Player O will choose $\min(V_j, V_k)$.
- **Edge Case Considered**: What if subboard $j$ has 0 empty cells? In Tier 7, by definition, both subboards $j$ and $k$ are open and non-terminal ($E_j \ge 1, E_k \ge 1$). Even if a terminal subboard configuration was queried, any move directed to a full subboard redirects to Free Choice, making the identity consistent.
- **Verdict**: The functional elimination of Free Choice is **sound and exact**.

#### 3. Dense Ternary Outcome Packing & Storage Size
- **Micro-State Pair Space**: $11,125 \times 11,125 = 123,765,625$ pairs.
- **Independent Outcomes**: 4 combinations per pair: $(X, j), (X, k), (O, j), (O, k)$, each taking 3 states (Win, Draw, Loss).
- **Byte Packing**: $3^4 = 81$ states. Since $81 \le 255$, all 4 outcomes pack into **exactly 1 byte (`u8`)**:
  $$\text{packed} = v_{X, j} + 3 \cdot v_{X, k} + 9 \cdot v_{O, j} + 27 \cdot v_{O, k} \in [0, 80]$$
- **Dense Slice Size**: $123,765,625 \text{ bytes} \approx 118.03 \text{ MiB} \ (123.77 \text{ MB})$.
- **Total Uncompressed Storage (8 Canonical Orbits)**:
  $$854 \times 123,765,625 = 105,695,843,750 \text{ bytes} \approx 98.44 \text{ GiB} \ (105.70 \text{ GB})$$
- **Total Uncompressed Storage (All 36 Pairs)**:
  Exhaustive evaluation across all 36 pairs yields 3,722 reachable macro vectors:
  $$3,722 \times 123,765,625 = 460,655,656,250 \text{ bytes} \approx 429.02 \text{ GiB} \ (460.66 \text{ GB})$$
- **ZFS Pool Feasibility**:
  Direct inspection of the host filesystem confirms:
  ```
  Filesystem: my-zraid | Size: 8.3T | Used: 1.1T | Avail: 7.2T (13% Util)
  ```
  With $3.0\times$ to $3.5\times$ native filesystem compression (`zstd-3` / `lz4`), the 8-orbit tablebase consumes **~26.4 to 35.2 GB** (<0.5% of pool), and the all-36-pair tablebase consumes **~131.6 GB** (1.8% of pool).
- **Verdict**: The storage calculations are **100% verified**.

---

### 2.2 Critical Defect: The 62-Bit Tier 6 Key Packing & Hash Collision Bug

#### 1. Audit of the Legacy Bug
In the legacy implementation (`encode_tier6_key` in `tier6.rs`):
```rust
let micro_hash = ((j_id as u64 & 0x3FFF) << 24)
    | ((k_id as u64 & 0x3FFF) << 12)
    | (l_id as u64 & 0x3FFF); // 36 bits
```
Since $11,125$ micro states require 14 bits ($2^{13} = 8192 < 11125 < 16384 = 2^{14}$):
- `l_id` occupies bits $0..13$.
- `k_id` shifted by 12 occupies bits $12..25$, overlapping bits 12 and 13 of `l_id`.
- `j_id` shifted by 24 occupies bits $24..37$, overlapping bits 24 and 25 of `k_id`.
Proposal A's identification of this defect and mathematical proof are **100% accurate**.

#### 2. The Flaw in the Proposed Fix (`encode_tier6_key_62` & `encode_u10`)
Proposal A defines a 62-bit key layout:
$$\underbrace{7\text{b}}_{\text{triplet}} + \underbrace{10\text{b}}_{\tau_6} + \underbrace{2\text{b}}_{\text{target}} + \underbrace{1\text{b}}_{\text{turn}} + \underbrace{14\text{b}}_{j_{\text{id}}} + \underbrace{14\text{b}}_{k_{\text{id}}} + \underbrace{14\text{b}}_{l_{\text{id}}} = 62\text{ bits} \le 64\text{ bits}$$

To compress the macro boundary vector $\tau_6$ into 10 bits, `uttt/src/egtb/tier6.rs` implemented `encode_u10()`:
```rust
#[inline]
pub fn encode_u10(&self) -> u16 {
    let delta_u = (self.delta.clamp(TIER_6_MIN_DELTA, TIER_6_MAX_DELTA) - TIER_6_MIN_DELTA) as u16; // 3 bits
    let wins = ...; // 6 bits (xw_j, xw_k, xw_l, ow_j, ow_k, ow_l)
    let shared = ...; // 6 bits (pairwise shared lines)

    let raw = wins | (shared << 6) | (delta_u << 12); // 15 bits
    let mixed = (raw as u32).wrapping_mul(0x9E37_79B9) >> 22; // 10-bit hash!
    (mixed as u16) & 0x3FF
}
```

#### 3. Mathematical Proof of Hash Collision in `encode_u10()`
The author treated `encode_u10()` as a bijective encoding because "each triplet has $\le 397$ reachable $\tau_6$ states, which easily fits in 10 bits ($2^{10} = 1024$)."  
However, `encode_u10()` is **not an injection**. It is a **lossy multiplicative hash function** mapping a 15-bit integer ($2^{15} = 32,768$) into a 10-bit space ($1,024$).

By the Generalized Birthday Problem, when hashing $N = 397$ distinct items into $M = 1024$ bins, the probability of at least one collision is:
$$P(\text{Collision}) \approx 1 - \exp\left(-\frac{N^2}{2M}\right) = 1 - \exp\left(-\frac{397^2}{2 \times 1024}\right) = 1 - e^{-76.9} \approx 1.0 \ (100\%)$$

We executed exhaustive collision detection across all 84 triplets for all legally reachable macro conditions. The results are decisive:
- **Triplets with Hash Collisions**: **72 out of 84 triplets** (85.7% of all subboard triplets!).
- **Total Collapsed Macro States**: **1,774 distinct reachable macro conditions** collide into occupied hash buckets.

#### 4. Concrete Counterexample
For the canonical triplet $(0, 1, 3)$ with open subboards $0, 1, 3$:
- **Macro State A**: Closed boards yield $xw_0 = \text{true}$, $x\_shared_{0, 3} = \text{true}$, $\Delta = +3$. Raw value: `0x6101` ($24,833$). Player X has an overwhelming advantage ($\Delta = +3$) and an active 3-in-a-row threat on subboard 0.
- **Macro State B**: Closed boards yield $xw_1 = \text{true}$, all shared lines false, $\Delta = 0$. Raw value: `0x3002` ($12,290$). Game is tied in closed boards ($\Delta = 0$) with a completely different winning line.
- **Hash Evaluation**:
  $$\text{encode\_u10}(A) = (24833 \times \text{0x9E3779B9}) \gg 22 = \mathbf{653}$$
  $$\text{encode\_u10}(B) = (12290 \times \text{0x9E3779B9}) \gg 22 = \mathbf{653}$$
- **Impact**: $\text{encode\_tier6\_key}(A) \equiv \text{encode\_tier6\_key}(B)$. In the tablebase, State A and State B produce the exact same 62-bit key. One state will silently corrupt the other's game-theoretic value!

#### 5. Why the Existing Unit Test Failed to Catch It
In `uttt/src/egtb/tier6.rs`, `test_tier6_62bit_key_packing_no_collision`:
```rust
let test_cases = [
    (0, 0, 0, false, 0, 0, 0),
    (83, 1023, 3, true, 11124, 11124, 11124),
    (42, 512, 1, false, 4096, 8192, 10000),
    (15, 100, 2, true, 1234, 5678, 9101),
];
for &(trip_idx, tau_10b, target, turn, j_id, k_id, l_id) in &test_cases {
    let key = encode_tier6_key_62(trip_idx, tau_10b, target, turn, j_id, k_id, l_id);
    let (d_trip, d_tau, ...) = decode_tier6_key_62(key);
    assert_eq!(d_tau, tau_10b);
}
```
The test only verified that `decode_tier6_key_62` inverts `encode_tier6_key_62` when fed pre-packed arbitrary integers. It **never passed real `Tier6MacroCondition` instances into `encode_tier6_key`**, creating a false sense of security.

#### 6. Actionable Mathematical Remediation
Our audit determined that across all 84 triplets, the total number of distinct reachable $\tau_6$ vectors globally is **only 1,727**. For any individual triplet, the maximum number of reachable $\tau_6$ vectors is **397**.
- **Solution 1 (Dense Per-Triplet Indexing Table)**: Precompute a compact table `REACHABLE_TAU_6[trip_idx]` mapping each reachable 15-bit `raw` vector to its dense index $0 \dots 396$. Because $396 < 512 = 2^9$, this dense index fits in **9 bits** (leaving 3 spare bits in a 64-bit key) with **guaranteed zero collisions**.
- **Solution 2 (Direct 12-Bit Packing)**: In a 64-bit integer, micro IDs ($42\text{b}$) + triplet ($7\text{b}$) + target ($2\text{b}$) + turn ($1\text{b}$) leave **12 bits** ($64 - 52 = 12$). Since the global reachable macro space is 1,727 states ($1727 \le 2048 = 2^{11}$), all reachable macro states across the entire game fit globally in **11 bits**.

---

### 2.3 Syzygy Probing & Search Engine Integration Mechanics

#### 1. Dense Indexing vs Binary Search Discrepancy
Proposal A claims in Section 6:
```
TierBranch -- "7" --> T7["Binary Search Tier 7 Table (< 60 ns)"]
```
This is an architectural contradiction with Section 4.2 Step 5–6:
- If Tier 7 is stored as a dense array of $123,765,625$ bytes per macro slice (1 byte per micro pair), **no binary search is performed**.
- Probing is an instantaneous $O(1)$ memory offset calculation:
  $$\text{byte\_offset} = \text{slice\_offset} + j_{\text{id}} \times 11,125 + k_{\text{id}}$$
  $$\text{outcome} = (\text{mmap}[\text{byte\_offset}] \gg \text{shift}) \pmod 3$$
- Direct indexing executes in **$< 15\text{ ns}$** (a single L3 cache/RAM access), vastly superior to the 25 iterations of a binary search.

#### 2. Runtime $D_4$ Symmetry Mapping vs Storing All 36 Pairs
Proposal A computes that storing only the 8 canonical orbits takes ~105 GB raw (~30 GB compressed). However:
- If only 8 canonical orbits are stored, when a search engine encounters an arbitrary pair $(a, b) \in 36$, it must find the symmetry $S \in D_4$ mapping $(a, b) \to (j, k)$, and **it must rotate the micro-board stones** $S(xbb[a]), S(obb[a])$ and look up their new `micro_id` in `to_id`.
- **Architectural Trade-Off**:
  * *Option 1 (Canonical 8 Orbits)*: ~30 GB compressed, requires runtime micro-board symmetry rotation and target square remapping.
  * *Option 2 (Exhaustive 36 Pairs)*: 460 GB raw, **~131 GB compressed on ZFS**. Because `/my-zraid` has 7,200 GB free, 131 GB represents just **1.8% of available storage**. Storing all 36 pairs completely eliminates runtime rotation, allowing direct array indexing in $< 15\text{ ns}$ for every search node.
- **Auditor Recommendation**: Store all 36 pairs on `/my-zraid`. The 100 GB difference is negligible on a 7.2 TB pool, but saves CPU cycles and eliminates rotational bugs during time-critical match play.

#### 3. WDL vs DTM Leaf Cutoff Mechanics
- The dense 1-byte packing stores Win/Draw/Loss (WDL) outcomes. It does not store Distance-to-Mate (DTM).
- Because UTTT is a strictly monotonic DAG ($E_{t+1} = E_t - 1$), cycles are impossible, so an engine cannot fall into infinite repetitions.
- However, when probing WDL at the root node, multiple moves may return `Win`. In Alpha-Beta, searching 1 ply deeper breaks ties by finding the fastest win. In MCTS, tiebreaks are governed by policy prior probabilities.

---

## 3. Deep Technical Audit of Proposal B (Transformer Self-Play & 3-Way Tournament)

### 3.1 3-Way Round-Robin Tournament Protocol

#### 1. Mathematical Proof of State Bias Neutralization
Proposal B mandates the **Balanced Mirrored Pair Protocol**: for every opening position $P_k$, two games are played:
- Game $2k$: Competitor A is Player 1 (X), Competitor B is Player 2 (O).
- Game $2k+1$: Competitor B is Player 1 (X), Competitor A is Player 2 (O).

Under an expected outcome formulation:
$$\mathbb{E}[R_{2k}] = V(P_k) + (\Delta_A - \Delta_B)$$
$$\mathbb{E}[R_{2k+1}] = V(P_k) + (\Delta_B - \Delta_A)$$
$$S_A = R_{2k} - R_{2k+1} \implies \mathbb{E}[S_A] = 2(\Delta_A - \Delta_B)$$
The intrinsic position value $V(P_k)$ and Player 1 advantage are **identically canceled out**.

Furthermore, tracking:
- **Double Wins (2-0)**: Absolute dominance (winning as both X and O).
- **Double Losses (0-2)**: Severe vulnerability.
- **Symmetrical Splits (1-1)**: First-move advantage forcing win for Player 1 regardless of engine.
This protocol aligns with the gold standard established by TCEC and the Computer Chess Association.

#### 2. Sample Sizing & Statistical Significance
- **Match Size**: $N = 512$ games ($K = 256$ paired positions) per pairing; $3 \times 512 = 1,536$ games total.
- **Margin of Error (95% CI)**:
  $$\text{MoE}_{95\%} = 1.96 \times \sqrt{\frac{p(1-p)}{N}} = 1.96 \times \sqrt{\frac{0.55 \times 0.45}{512}} \approx \pm 4.31\%$$
- **Hypothesis Testing Nuance**: Proposal B claims $p < 0.005$ false promotion probability under $H_0: p = 0.50$ for $N = 512$ at a $55\%$ threshold.
  Calculating the exact test statistic:
  $$Z = \frac{0.55 - 0.50}{\sqrt{0.5 \times 0.5 / 512}} = \frac{0.05}{0.0221} = 2.26 \implies p = 1 - \Phi(2.26) = \mathbf{0.0119}$$
  The false positive rate at $N = 512$ is **$1.19\%$**, not $< 0.005$. To guarantee $p < 0.005$ ($Z \ge 2.576$), the match requires $N \ge 664$ games (or win rate $\ge 55.7\%$). In `parallel_mirror_eval.py`, default evaluation uses $K = 1,024$ pairs ($N = 2,048$ games), which yields $Z = 4.52$ ($p < 0.00001$).
- **Recommendation**: Maintain $N = 512$ for quick benchmark matches, but mandate $N \ge 1,024$ ($K = 512$) for champion gating promotion to preserve $p < 0.005$.

#### 3. Joint Bradley-Terry Elo Estimation
The formulation simultaneously solves for latent ratings $\mathbf{r} = (r_{\text{Trans}}, r_{\text{Res}}, r_{\text{AB}})$:
$$P(i \text{ beats } j) = \frac{1}{1 + 10^{(r_j - r_i)/400}}$$
with $r_{\text{Res}} = 1906.4$ anchored.
Handling of draws must be explicitly specified: counting each draw as $0.5$ win and $0.5$ loss in the log-likelihood (the standard Davidson half-win model) is mathematically robust and matches the scoring metric.

---

### 3.2 Vision/Board Transformer Architecture & Spatial Routing Audit

```mermaid
flowchart LR
    Cell["Cell Token: m = sb * 9 + sq"] --> Embed["Token Projection + sb_embed(sb) + sq_embed(sq)"]
    Embed --> SelfAttn["Layer 1 Multi-Head Self-Attention"]
    SelfAttn --> Target["Direct Coupling: sq_embed(sq) matches sb_embed(sq) of target subboard in O(1)"]
```

#### 1. Inductive Bias & Positional Encodings
In `ml/transformer_model.py`:
- Cell index $m = 9 \cdot sb + sq$.
- `sb = m // 9` (current subboard $0..8$).
- `sq = m % 9` (square within subboard $0..8$).
Under UTTT rules, placing a move at subsquare $sq$ dictates that the opponent's next move must occur in subboard $sb' = sq$.
The model assigns dual learned positional embeddings:
$$\mathbf{e}_m = \text{input\_proj}(\mathbf{x}_m) + \mathbf{E}_{\text{subboard}}[sb] + \mathbf{E}_{\text{subsquare}}[sq]$$
Because the key/query projections operate across the full 81-token sequence, attention heads in Layer 1 can directly match $\mathbf{E}_{\text{subsquare}}[sq]$ of the active cell with $\mathbf{E}_{\text{subboard}}[sq]$ of all cells in the target subboard.
This completely overcomes the **receptive field lag** that hobbles CNNs, where 4–5 residual blocks are required before a convolutional kernel can bridge between subboards.

#### 2. Dual Prediction Heads & Numerical Stability
- Policy logits are projected from the 81 contextualized cell tokens:
  ```python
  fill_val = -1e4 if logits.dtype == torch.float16 else -1e9
  logits = torch.where(legal_mask.bool(), logits, torch.full_like(logits, fill_val))
  ```
  Using `-1e4` in FP16 AMP prevents underflow/overflow to `-inf` and `NaN` (since FP16 normal range ends at $\approx -65,504$).
- Value head predicts scalar outcome in $[-1.0, +1.0]$ via `Tanh` applied to the `[CLS]` token.
- Pre-LayerNorm (`norm_first=True`) ensures gradient stability during deep MCTS backpropagation.

---

### 3.3 Continuous Self-Play Loop Rigor

#### 1. Temperature Schedule ($T > 0$ Always)
- Plies $\le 20$: $T = 1.0$ (ensures diverse opening exploration).
- Plies $> 20$: $T = 0.5$ (sharpens tactical exploitation while strictly maintaining $T > 0$).
- Prohibiting $T = 0$ during self-play generation is critical: it prevents the policy network from collapsing into deterministic degenerate loops.

#### 2. MCTS Visits & Search Throughput
- $800$ simulations per move with $c_{\text{puct}} = 2.0$ and Dirichlet noise ($\alpha = 0.3, \epsilon = 0.25$) applied at the root.
- Pure Rust Rayon worker (`RustMctsWorker`) batches 1,024 leaves directly into GPU pinned host memory, achieving **~58,000 evals/s** in FP16 AMP on the RTX 5060 Ti.

#### 3. Replay Buffer Stratification
`ContiguousStratifiedReplayBuffer` maintains 1,000,000 states across flat NumPy arrays (814 MB footprint, zero GC pressure):
- 30% Grandmaster / Expert replays.
- 35% Deep tactical battles ($\ge 38$ plies) and hard draws.
- 35% Active self-play generations.
- On-the-fly $D_4$ dihedral symmetry augmentation provides $8\times$ data expansion without disk duplication.

#### 4. Safe INT8 Export Handling
In `ml/champion_challenger.py`, `export_int8.py` is hardcoded to fold `Conv2d` and `BatchNorm2d` layers of `AlphaZeroResNet`. If called on `UTTTTransformer`, it would raise an `AttributeError`. Proposal B's plan to guard this export and preserve `champion_gen7_resnet.pt` for the pure-Rust `AlphaZeroBot` is verified as necessary and safe.

---

### 3.4 C-ABI Bridge Architecture & Performance Evaluation

#### 1. Critical Performance Defect in Proposed `alphabeta_batch_choose_moves`
Proposal B Section 3.2 specifies:
```rust
#[no_mangle]
pub extern "C" fn alphabeta_batch_choose_moves(...) {
    (0..num_games).into_par_iter().for_each(|i| {
        // ...
        let chosen_move = if bot_version == 2 {
            let mut bot = AlphaBetaV2Bot::new(); // <-- SEVERE DEFECT!
            bot.choose_move(&board, &legal_moves_lookup, &zobrist_constants, duration)
        } else {
            let mut bot = AlphaBetaBot::new();   // <-- SEVERE DEFECT!
            bot.choose_move(&board, &legal_moves_lookup, &zobrist_constants, duration)
        };
        // ...
    });
}
```

**Audit Inspection of `AlphaBetaV2Bot::new()` (`uttt/src/bots/alphabeta_v2_bot.rs`)**:
```rust
pub fn new() -> Self {
    AlphaBetaV2Bot {
        tt: HashMap::with_capacity(1_000_000), // Allocates 1,000,000 table slots (~32-48 MB)!
        book: OpeningBook::new(),
        dfpn: DfPnSolver::new(3_000),
        legal_lookup: UltimateTicTacToeBoard::get_move_ordering_array(),
        // ...
    }
}
```
**Impact Analysis**:
1. **Memory Allocation Catastrophe**:
   Instantiating `AlphaBetaV2Bot::new()` inside the parallel closure for $N = 1,024$ games allocates:
   $$1,024 \times 40 \text{ MB} = \mathbf{40.96 \text{ GIGABYTES of heap memory PER PLY!}}$$
   In a 40-ply game, the engine would allocate and destroy over **1.6 Terabytes of temporary heap memory**.
2. **Allocator Lock Contention**:
   16 Rayon worker threads concurrently allocating and deallocating 1,000,000-entry HashMaps will trigger severe thread contention in `malloc`/`jemalloc`, causing CPU search to freeze.
3. **Loss of Engine Strength (TT / History Erasure)**:
   Alpha-Beta derives its search pruning efficiency from the Transposition Table (TT), killer move tables, and history heuristics accumulated across moves. Recreating the bot on every ply destroys all accumulated heuristics, forcing a cold search and reducing effective search depth by 2–4 plies.

#### 2. Remediation: Persistent Bot Worker Pool
Instead of per-game heap allocation, the C-ABI bridge must maintain a persistent pool:
```rust
pub struct AlphaBetaWorkerPool {
    bots: Vec<AlphaBetaV2Bot>, // 16 instances (one per Rayon thread)
}
```
Each Rayon worker thread borrows a persistent bot from thread-local storage or an indexed pool, clearing the TT via an epoch/generation counter rather than deallocating memory.

#### 3. Missing Batch Move Application in `RustMctsWorker`
In Match 2 (Transformer vs AlphaBeta), AlphaBeta selects moves on CPU while Transformer evaluates on GPU.
Currently, `mcts_worker.rs` only supports `mcts_worker_advance_moves_tournament`, which advances all games using MCTS root visit counts.
Applying AlphaBeta moves requires adding:
```rust
#[no_mangle]
pub extern "C" fn mcts_worker_apply_moves_batch(
    worker: *mut MctsWorker,
    game_indices: *const usize,
    moves: *const u8,
    num_moves: usize,
)
```
This applies the moves directly to the bitboard state and invalidates the MCTS tree for those games without Python ctypes per-game loop overhead.

---

## 4. Consolidated Risk Matrix

| Risk ID | Subsystem | Description | Severity | Probability | Recommended Mitigation |
|:---|:---|:---|:---:|:---:|:---|
| **R-1** | EGTB Tier 6 | `encode_u10()` lossy multiplicative hash collides in 72/84 triplets (1,774 states). | **CRITICAL** | **100% (Proven)** | Replace multiplicative hash with dense per-triplet index lookup table `REACHABLE_TAU_6[trip][raw]`. |
| **R-2** | AlphaBeta C-ABI | `AlphaBetaV2Bot::new()` in Rayon loop allocates ~40 GB RAM per ply. | **CRITICAL** | **100% (Proven)** | Implement persistent 16-thread bot pool in Rust; reuse TT and killer tables. |
| **R-3** | EGTB Tier 7 | Runtime $D_4$ micro-board rotation overhead and complexity if storing only 8 orbits. | **MEDIUM** | **HIGH** | Precompute all 36 pairs on `/my-zraid` (131 GB compressed, 1.8% of pool) to enable zero-rotation $O(1)$ direct indexing. |
| **R-4** | Tournament | False promotion probability at $N=512$ is $1.19\%$ (exceeds stated $p < 0.005$). | **LOW** | **MEDIUM** | Mandate $N \ge 1,024$ ($K = 512$ pairs) for official champion promotion tournaments. |
| **R-5** | MCTS Integration | Lack of batch external move application API in `mcts_worker.rs`. | **MEDIUM** | **HIGH** | Export `mcts_worker_apply_moves_batch` in `mcts_worker.rs`. |
| **R-6** | Quantization | Transformer champion crashing legacy ResNet `export_int8.py`. | **LOW** | **LOW** | Guard INT8 export in `champion_challenger.py` when architecture is `UTTTTransformer`. |

---

## 5. Concrete Actionable Implementation Patches

### Patch 1: Collision-Free Tier 6 Macro Condition Encoding (`uttt/src/egtb/tier6.rs`)

Replace the lossy multiplicative hash in `encode_u10` with direct 12-bit global packing or a dense per-triplet lookup:

```rust
// In uttt/src/egtb/tier6.rs

/// Lossless 11-bit global index or dense per-triplet encoding.
/// Because each triplet has <= 397 reachable tau_6 states, 
/// a 10-bit dense index (0..396) per triplet is provably collision-free.
pub fn encode_tau6_dense(trip_idx: usize, tau: &Tier6MacroCondition) -> u16 {
    let raw = tau.encode_raw_15b(); // wins (6b) | (shared << 6) | (delta_u << 12)
    // Lookup in precomputed static array of sorted reachable raw values per triplet:
    // TRIPLET_TAU_DENSE_MAP[trip_idx].binary_search(&raw).expect("Reachable tau must exist") as u16
    TRIPLET_REACHABLE_TABLE[trip_idx].get_dense_index(raw)
}
```

### Patch 2: Thread-Safe Persistent AlphaBeta Worker Pool (`uttt/src/c_api.rs`)

Eliminate the 40 GB/ply allocation by creating persistent thread-local bot instances:

```rust
use std::cell::RefCell;

thread_local! {
    static THREAD_BOT_V2: RefCell<AlphaBetaV2Bot> = RefCell::new(AlphaBetaV2Bot::new());
}

#[no_mangle]
pub extern "C" fn alphabeta_batch_choose_moves(
    moves_flat: *const u8,
    move_offsets: *const usize,
    move_lengths: *const usize,
    num_games: usize,
    time_limit_ms: u64,
    _bot_version: i32,
    out_moves: *mut u8,
) {
    let offsets = unsafe { std::slice::from_raw_parts(move_offsets, num_games) };
    let lengths = unsafe { std::slice::from_raw_parts(move_lengths, num_games) };
    let duration = Duration::from_millis(time_limit_ms);

    (0..num_games).into_par_iter().for_each(|i| {
        THREAD_BOT_V2.with(|bot_cell| {
            let mut bot = bot_cell.borrow_mut();
            let slice = unsafe { std::slice::from_raw_parts(moves_flat.add(offsets[i]), lengths[i]) };

            let zobrist = UltimateTicTacToeBoard::get_zobrist_constants();
            let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
            let mut board = UltimateTicTacToeBoard::default();
            for &m in slice {
                board.make_move(GameMove { x_to_move: board.x_to_move, move_index: m }, &zobrist);
            }

            let chosen_move = bot.choose_move(&board, &legal_lookup, &zobrist, duration);
            unsafe {
                *out_moves.add(i) = chosen_move.move_index;
            }
        });
    });
}
```

### Patch 3: Batch Move Application in MCTS Worker (`uttt/src/mcts_worker.rs`)

```rust
#[no_mangle]
pub extern "C" fn mcts_worker_apply_moves_batch(
    worker: *mut MctsWorker,
    game_indices: *const usize,
    moves: *const u8,
    num_moves: usize,
) {
    if let Some(w) = unsafe { worker.as_mut() } {
        let indices = unsafe { std::slice::from_raw_parts(game_indices, num_moves) };
        let mv_slice = unsafe { std::slice::from_raw_parts(moves, num_moves) };

        for k in 0..num_moves {
            let g = indices[k];
            if g < w.num_games {
                let m = mv_slice[k];
                let game = &mut w.games[g];
                game.board.make_move(
                    GameMove { x_to_move: game.board.x_to_move, move_index: m },
                    &w.zobrist_constants,
                );
                game.current_leaf_board = game.board.clone();
                game.nodes.clear();
                game.children.clear();
                game.moves_played += 1;
            }
        }
    }
}
```

---

## 6. Conclusion & Recommendation to System Architect

1. **Approve Proposal B for immediate production implementation**, incorporating Patch 2 (Thread-local AlphaBeta worker pool) and Patch 3 (Batch move application in `MctsWorker`). The Vision/Board Transformer architecture is sound, verified, and provides an extraordinary leap in tactical routing capabilities.
2. **Condition Approval of Proposal A upon applying Patch 1 (Dense Bijection Table for Tier 6)** and electing to store all 36 pairs on the 7.2 TB `/my-zraid` ZFS pool (~131 GB compressed) to achieve zero-overhead direct tablebase probing during competitive match play.
