# Plans & Progress: `uttt/src/egtb/tier6.rs`

This document details the Tier 6 endgame tablebase specification, symmetry reduction, retrograde linkage into Tier 7, and hybrid solver implementation in [`uttt/src/egtb/tier6.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs).

---

## 1. Architectural Design & Plan

- **Tier 6 Boundary Vector Specification (`Tier6MacroCondition`)**:
  - Encodes macro boundary conditions across the 6 closed subboards for a triplet of open subboards $(j, k, l)$ with $0 \le j < k < l < 9$:
    * Single-board wins: `xw_j`, `xw_k`, `xw_l`, `ow_j`, `ow_k`, `ow_l` (6 bits).
    * Pairwise collinear shared lines: `x_shared_jk`, `x_shared_jl`, `x_shared_kl`, `o_shared_jk`, `o_shared_jl`, `o_shared_kl` (6 bits).
    * Triplet collinear winning line: `line_jkl` (1 bit, whether $\{j, k, l\}$ forms a 3-in-a-row).
    * CodinGame saturated tiebreaker: `delta` ($|M_X| - |M_O|$ on closed boards), clamped to $[-3..+3]$ (3 bits).
  - Encodes to a raw 16-bit representation (`encode_u16` / `decode_u16`).

- **$D_4$ Dihedral Symmetry Reduction of the 84 Triplets**:
  - The $\binom{9}{3} = 84$ subboard triplets partition under the $D_4$ dihedral group into exactly **16 canonical orbits**:
    * **56 triplets with Center 4 closed** partition into **exactly 10 canonical classes** (`class_10_id` $\in 0..9$):
      `(0, 1, 2)`, `(0, 1, 3)`, `(0, 1, 5)`, `(0, 1, 6)`, `(0, 1, 7)`, `(0, 1, 8)`, `(0, 2, 6)`, `(0, 2, 7)`, `(0, 5, 7)`, `(1, 3, 5)`.
    * **28 triplets with Center 4 open** partition into **6 canonical classes**:
      `(0, 1, 4)`, `(0, 2, 4)`, `(0, 4, 5)`, `(0, 4, 8)`, `(1, 3, 4)`, `(1, 4, 7)`.
  - Computed via [`canonical_triplet(j, k, l) -> CanonicalTriplet`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs#L311).

- **Dense Bijection Mapping (`TRIPLET_TAU_DENSE_TABLE`)**:
  - Precomputes all legally reachable macro conditions from $3^6 = 729$ closed configurations for each of the 84 triplets.
  - Provably bounded to $\le 397$ reachable conditions per triplet ($< 512$).
  - `encode_dense(trip_idx) -> u16` performs a binary search over the precomputed list, providing a 9-bit collision-free dense index.

- **Collision-Free 61-Bit Key Packing**:
  - Positions are encoded into 61 bits ($< 2^{61} \le 64$ bits) via [`encode_tier6_key_dense`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs#L565):
    * Bits 54..60 (7 bits): triplet index ($0..83 < 128$)
    * Bits 45..53 (9 bits): tau_6 dense macro index ($0..396 < 512$)
    * Bits 43..44 (2 bits): target open subboard ($0=j, 1=k, 2=l, 3=\text{Free Choice}$)
    * Bit 42 (1 bit): turn ($1=X, 0=O$)
    * Bits 28..41 (14 bits): subboard $j$ micro ID ($0..11,124 < 16,384$)
    * Bits 14..27 (14 bits): subboard $k$ micro ID ($0..11,124 < 16,384$)
    * Bits 0..13 (14 bits): subboard $l$ micro ID ($0..11,124 < 16,384$)

- **Retrograde Transition into Tier 7 (`transition_tier6_to_tier7`)**:
  - Any move completing an open subboard (by win or tie) transitions immediately into Tier 7.
  - Queries [`MacroFactoredProber`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs#L1755) in $< 100$ ns with updated boundary vector $\tau'$, active shared lines, and updated delta.

- **Hybrid Solver Architecture ([`HybridDfPnSolver`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs#L690))**:
  - Combines fast tablebase leaf cutoffs with Grandfather DF-PN search.
  - Evaluates root and leaves against `MacroFactoredProber` ($< 100$ ns) when resolved subboards $\ge 6$, falling back to DF-PN proof numbers for unresolved positions.

---

## 2. Types & Public APIs

- `Tier6MacroCondition`: 13-bit condition struct with `encode_u16`, `decode_u16`, `encode_dense`, `decode_dense`.
- `CanonicalTriplet`: Contains `rep_j`, `rep_k`, `rep_l`, `sym`, `orbit_id`, `is_center_closed`, `class_10_id`.
- `SubboardResolution`: Enum (`XWin`, `OWin`, `Tie`).
- `Tier6LayerTable`: 3D layer table for $(e_j, e_k, e_l)$ with 8 outcomes per state.
- `HybridDfPnSolver`: Hybrid solver combining DF-PN with instant EGTB probing.
- `extract_tier6_tau(x_supers, o_supers, j, k, l) -> Tier6MacroCondition`.
- `triplet_to_index(j, k, l) -> usize` & `index_to_triplet(idx) -> (usize, usize, usize)`.
- `canonical_triplet(j, k, l) -> CanonicalTriplet`.
- `transition_tier6_to_tier7(...) -> EgtbOutcome`.
- `encode_tier6_key`, `encode_tier6_key_dense`, `decode_tier6_key_dense`.

---

## 3. Status & Verification

- `test_d4_symmetry_canonical_reduction_orbits`: Verifies 56 center-closed triplets reduce to 10 classes and total 84 triplets reduce to 16 orbits.
- `test_triplet_indexing_bijection`: Confirms perfect bidirectional mapping between $(j, k, l)$ and $0..84$.
- `test_tier6_macro_condition_bit_packing`: Confirms lossless roundtrip of boundary conditions.
- `test_extract_tier6_tau_invariants`: Verifies collinear triplet identification and shared line detection.
- `test_retrograde_transition_soundness`: Verifies immediate terminal detection and Tier 7 probe queries.
- `test_tier6_62bit_key_packing_no_collision`: Proves collision-free encoding for micro IDs $\ge 4096$.
- `test_tier6_dense_macro_bijection_no_collisions_exhaustive`: Exhaustively validates that all 84 triplets have $\le 397$ reachable states, strict sorting, and zero collisions.
