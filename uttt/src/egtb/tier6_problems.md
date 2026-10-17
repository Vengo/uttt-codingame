# Problems & Challenges: `uttt/src/egtb/tier6.rs`

This document details the technical problems, symmetry reductions, and invariant handling solved in [`uttt/src/egtb/tier6.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs).

---

## 1. Technical Challenges

- **Tri-Board Combinatorial Explosion**:
  With 3 open subboards, there are $\binom{9}{3} = 84$ possible triplets. Each triplet has up to $11,125^3 \approx 1.37 \times 10^{12}$ micro configurations across $3^6 = 729$ macro assignments.
  - *Solution*: $D_4$ dihedral symmetry reduces the 84 triplets into exactly 16 canonical orbits (10 orbits when center 4 is closed, 6 orbits when center 4 is open). Backward induction is depth-stratified to shallow empty cell counts ($E \le 2$ or $E \le 6$), with deeper states solved on demand via [`HybridDfPnSolver`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs#L690).

- **Multi-Board Collinear Shared Line Cascades**:
  With 3 open boards $(j, k, l)$, there are 3 distinct pairs ($(j, k)$, $(j, l)$, $(k, l)$) that may each share collinear lines with closed subboards, or all three may form a single collinear winning line (`line_jkl`).
  - *Solution*: `Tier6MacroCondition` explicitly tracks 6 pairwise shared line flags (`x_shared_jk`, etc.) plus `line_jkl`. When subboard $j$ is won by X in `transition_tier6_to_tier7`, any collinear line through $j$ and $k$ activates `xw_k'`, and the 3-board line `line_jkl` converts into an active Tier 7 pairwise shared line between the remaining open boards $(k, l)$.

- **Elimination of Hash Collisions in 64-Bit Keys**:
  Earlier 12-bit shift allocations overflowed when micro IDs exceeded 4096 ($11,125 > 4,096$), causing cross-subboard collisions (e.g., $l_{\text{id}} = 4096$ colliding with $k_{\text{id}} = 1$). Furthermore, pseudo-random hashing of macro conditions produced birthday paradox transpositions.
  - *Solution*: Micro IDs are allocated 14 bits each ($2^{14} = 16,384 > 11,125$), and macro conditions are mapped into a dense 9-bit index via `TRIPLET_TAU_DENSE_TABLE`. The entire key packs into 61 bits ($< 2^{61} \le 64$), providing a mathematically guaranteed injection with zero transpositions.

- **Reachable Macro State Filtering**:
  Naively treating all 13 macro boundary bits as independent yields $2^{13} = 8,192$ states, most of which are mathematically unreachable or contradictory on a 3x3 macro board.
  - *Solution*: `TRIPLET_TAU_DENSE_TABLE` exhaustively simulates the $3^6 = 729$ closed subboard configurations, discards games already won on closed boards, and dedups the resulting vectors. This guarantees that no triplet ever has more than 397 reachable conditions, easily fitting within 9 bits.

- **Low-Latency Transition into Tier 7**:
  Retrograde DP requires evaluating thousands of subboard-closing moves per second. Re-solving the remaining 2 open subboards from scratch is computationally infeasible.
  - *Solution*: `transition_tier6_to_tier7` evaluates immediate macro-board wins in $O(1)$ and queries `MacroFactoredProber` to retrieve precomputed Tier 7 outcomes in $< 100$ ns.
