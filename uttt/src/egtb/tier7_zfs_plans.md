# Plans & Progress: `uttt/src/egtb/tier7_zfs.rs`

This document details the exhaustive ZFS-backed Tier 7 endgame tablebase solver and zero-copy direct-index prober in [`uttt/src/egtb/tier7_zfs.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs).

---

## 1. Architectural Design & Plan

- **Exhaustive Dense Micro-Pair Matrix Architecture**:
  - Instead of hash-based sparse tables, Tier 7 is decomposed into dense 2D matrices across all $11,125 \times 11,125 = 123,765,625$ micro-board state pairs.
  - Each micro pair stores 4 ternary outcomes (Turn X/O $\times$ Target $j$/$k$) densely packed into a single byte ($3^4 = 81 \le 255$):
    $$\text{byte} = v(X, j) + 3 \cdot v(X, k) + 9 \cdot v(O, j) + 27 \cdot v(O, k)$$
  - Each canonical slice file (`tier7_orbit_{}_tau_{:03}.bin`) occupies exactly **123,765,625 bytes** (~123.8 MB).

- **$D_4$ Dihedral Symmetry Orbit Reduction**:
  - The 36 possible pairs of open subboards $(j, k)$ ($0 \le j < k < 9$) partition under the $D_4$ dihedral group into exactly **8 canonical orbits**:
    `CANONICAL_ORBITS = [(0, 1), (0, 2), (0, 4), (0, 5), (0, 8), (1, 3), (1, 4), (1, 7)]`.
  - Orbit counts cover the full set of 36 pairs: $[8, 4, 4, 8, 2, 4, 4, 2]$.
  - `map_to_canonical_pair(j, k) -> CanonicalPairMap` identifies the canonical representative, orbit ID, symmetry transform, and axis reflection flag (`flipped`).

- **Macro Tau Boundary Vector (`Tier7MacroTau`)**:
  - Represents the macro state over the 7 closed subboards:
    * `xw_j`, `xw_k`: whether X completing subboard $j$ or $k$ completes macro 3-in-a-row.
    * `ow_j`, `ow_k`: whether O completing subboard $j$ or $k$ completes macro 3-in-a-row.
    * `x_shared`, `o_shared`: whether open subboards $j$ and $k$ share a collinear line with a closed subboard won by X/O.
    * `delta`: closed subboard win differential ($|M_X| - |M_O|$), clamped to $[-2..=2]$.
  - `enumerate_orbit_taus(rep_j, rep_k)` filters out non-terminal legal configurations from $3^7 = 2,187$ closed assignments, yielding **exactly 614 unique canonical tau slices** across all 8 orbits.

- **Backward Induction Engine (`solve_single_canonical_slice`)**:
  - Solves via retrograde dynamic programming layered by total empty cells $E = e_j + e_k$ from $0$ up to $18$.
  - Layer memory is allocated via `allocate_pair_layer_tables` (~708 MB resident per thread).
  - Transitions resolving a subboard instantly query `Tier8Lookup::get` with updated $\tau'$ and differential $\Delta'$.

- **Storage & Probing Pipeline**:
  - [`Tier7ZfsSolver`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs#L624): Rayon-parallel execution writing atomic temporary files (`.tmp.<pid>`), renaming to final binary files, and exporting `manifest.json`. Supports idempotent restarts by checking file existence and length.
  - [`Tier7ZfsProber`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs#L768): Zero-copy prober holding a mutex-guarded map of lazy `Mmap` slices. Looks up byte directly at `j_id * 11125 + k_id` in $< 15$ ns.

---

## 2. Types & Public APIs

- `CANONICAL_ORBITS: [(usize, usize); 8]`: Canonical orbit representatives.
- `SLICE_BYTE_SIZE: usize = 123_765_625`: Exact byte size of each slice file.
- `CanonicalPairMap`: Contains `orbit_id`, `rep_j`, `rep_k`, `sym`, `flipped`.
- `Tier7MacroTau`: Packed macro boundary struct with `encode_u16` and `decode_u16`.
- `map_to_canonical_pair(j: usize, k: usize) -> CanonicalPairMap`: Computes $D_4$ canonical pair reduction.
- `enumerate_orbit_taus(rep_j: usize, rep_k: usize) -> Vec<Tier7MacroTau>`: Generates all legal non-terminal macro vectors for an orbit.
- `solve_single_canonical_slice(...)`: Solves a full 123.76M-cell retrograde DP slice.
- `Tier7ZfsSolver::new(output_dir, num_threads) -> Self` & `solve_all() -> Result<(usize, u64), String>`.
- `Tier7ZfsProber::new(dir) -> Result<Self, String>` & `probe(&UltimateTicTacToeBoard) -> Option<EgtbOutcome>`.

---

## 3. Status & Verification

- Unit test `test_canonical_pair_reduction_all_36_pairs` verifies all 36 pairs map correctly to the 8 orbits.
- Unit test `test_enumerate_orbit_taus_count` confirms exactly 614 canonical tau slices across all orbits.
- Unit test `test_single_canonical_slice_solve_accuracy` validates retrograde DP accuracy and packed byte bounds ($\le 80$).
- Unit test `test_tier7_zfs_prober_live_queries` verifies zero-copy querying against live ZFS datasets when mounted.
