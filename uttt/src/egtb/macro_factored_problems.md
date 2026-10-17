# Problems & Challenges: `uttt/src/egtb/macro_factored.rs`

This document details the technical challenges, edge cases, invariants, and algorithms solved in [`uttt/src/egtb/macro_factored.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs).

---

## 1. Technical Challenges

- **Combinatorial State Space Explosion Across Tiers**:
  Solving Ultimate Tic-Tac-Toe endgames directly by full-board state enumeration quickly becomes impossible because of the sheer number of closed subboard permutations ($3^8 = 6,561$ for Tier 8; $3^7 = 2,187$ for Tier 7).
  - *Solution*: Macro-factoring replaces raw macro configurations with a 6-bit boundary condition vector $\tau$ and clamped win differential $\Delta$. This reduces thousands of macro permutations to at most 20 distinct boundary conditions in Tier 8 and a small set of legal configurations in Tier 7.

- **Collinear Shared Lines Between Multiple Open Subboards**:
  When two subboards $(j, k)$ are collinear with a closed subboard $m$ (forming a winning line $\{j, k, m\}$), winning subboard $j$ creates an immediate 3-in-a-row threat on subboard $k$ if $m$ was already won by that player.
  - *Solution*: The augmented macro vector tracks `x_shared` and `o_shared` explicitly. In the retrograde step, resolving subboard $j$ with an X win upgrades $xw_k'$ to `xw_k || x_shared`, ensuring multi-board win tactics are rigorously propagated.

- **CodinGame Saturated Mini-Board Win Count Tiebreaker**:
  Under CodinGame rules, if neither player forms a 3-in-a-row on the macro-board, the match is decided by the popcount of won subboards ($|M_X|$ vs $|M_O|$).
  - *Solution*: The differential $\Delta = |M_X| - |M_O|$ on closed boards is clamped to $[-3..+3]$ (or $[-2..+2]$) and tracked as an invariant. When a subboard is won during retrograde DP, the differential transitions to $(\Delta \pm 1)$, guaranteeing exact game outcomes when all subboards reach full/tied terminal status ($E = 0$).

- **Runtime Memory Exhaustion Guarding**:
  Multi-threaded retrograde DP over 36 pairs can easily trigger out-of-memory errors on memory-constrained execution environments.
  - *Solution*: [`MemoryBudgetGuard`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L64) samples `/proc/self/statm` on Linux before processing chunks. Any resident memory delta exceeding the 512 MB ceiling triggers an immediate descriptive panic, preventing uncontrollable OS swapping or OOM killer events.

- **Micro-State Move Generation Bottlenecks**:
  Iterating over 9 squares, calculating bitwise occupations, checking subboard win lines, and looking up state indices for millions of micro-state combinations creates a severe CPU hotspot.
  - *Solution*: [`MicroTransitionTable`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L325) precomputes all legal transitions, successor IDs, and subboard outcomes once at initialization. During retrograde DP, move generation is replaced by a flat array slice iteration.

- **Zero-Allocation Probing in Tight Search Loops**:
  MCTS rollouts and Alpha-Beta search routines cannot afford heap allocation or slow disk reads when querying endgame results.
  - *Solution*: [`MacroFactoredProber`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L1755) maps tablebase files directly into virtual memory via `memmap2::Mmap` and performs in-place binary searches on sorted 10-byte records, returning exact `EgtbOutcome` in $< 30$ ns with zero heap allocation.
