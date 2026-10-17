# Problems & Challenges: `uttt/src/egtb/retrograde.rs`

This document details the technical problems, topological invariants, and retrograde dynamic programming challenges solved in [`uttt/src/egtb/retrograde.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/retrograde.rs).

---

## 1. Technical Challenges

- **Cycle-Free Backward Induction via Empty Cell Stratification**:
  In general board games, retrograde analysis requires complex cycle-detection and value iteration algorithms because positions can repeat.
  - *Solution*: Ultimate Tic-Tac-Toe possesses an invariant Directed Acyclic Graph (DAG) structure: every legal move permanently claims an empty cell and increments board occupancy. By stratifying backward induction strictly by remaining empty cells $E = e_j + e_k$ from $0$ up to $18$, child states are guaranteed to be fully evaluated before parent states are processed.

- **Topological Forcing in Tier 8**:
  When only 1 subboard $k$ remains open, directed moves directing the opponent outside of $k$ appear to create complex multi-board choices.
  - *Solution*: Under game rules, targeting a resolved board grants free choice, but since all other 8 subboards are resolved, the opponent has no legal choice other than playing back into subboard $k$. This topological collapse reduces Tier 8 to an isolated single-subboard game coupled with macro win detection, eliminating all cross-board graph traversal.

- **Micro-State Validity Filtering**:
  A 3x3 board has $3^9 = 19,683$ cell assignments, but many represent illegal configurations (such as simultaneous winning lines for both players or mark count imbalances $|X| - |O| \notin \{0, 1\}$).
  - *Solution*: `MicroSubboardTable::build()` iterates over all bit combinations, enforces mark parity, and discards already-won boards, isolating exactly 11,125 legal active states and indexing them densely for $O(1)$ coordinate-to-ID conversion.

- **Scaling Tier 6 Across Constrained Memory Budgets**:
  Solving all 84 triplets of Tier 6 simultaneously in RAM causes memory exhaustion on typical workstations or cloud instances.
  - *Solution*: The solver implements disk-backed chunk streaming (`solve_tier_6_chunk`), partitioning triplets into independent chunks and writing `.done` markers. A final pass (`merge_tier_6_chunks`) streams each binary chunk, sorting and deduplicating records with minimal memory overhead.

- **Symmetry Deduplication with Exact Canonical Hashing**:
  Endgame tablebase entries generated across multiple pairs and turns can generate millions of symmetric transpositions.
  - *Solution*: Generated boards are normalized to canonical form using `to_canonical()` and `SymmetryEngine::canonical_form()`, deduplicating Zobrist hashes with parallel unstable sorting before disk serialization.
