# Plans & Progress: `uttt/src/egtb` Endgame Tablebase Module

This document synthesizes the architectural plans, submodules, data structures, and verification across the endgame tablebase module based on the folder's documentation.

---

## 1. Architectural Architecture & Submodule Map

- [`mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod_plans.md):
  - **`EgtbDatabase`**: Central tablebase management engine coordinating partitioned tier files (`tier_mmaps: [Option<Mmap>; 10]`), monolithic legacy files (`monolithic_mmap`), and the ultra-compact `MacroFactoredProber`.
  - **Multi-Level Probing Priority**: Probes `MacroFactoredProber` first ($< 30$ ns), falls back to direct tier mmap binary search by resolved subboard count $k$, then falls back to monolithic mmap.
  - **Systematic Solvers**: Implements `generate_tier_9` ($O(1)$ superboard line evaluation in $< 15$ ms), `generate_tier_8` and `generate_tier_7` via `DfPnSolver`, and `generate_egtb` dispatcher.

- [`retrograde.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/retrograde_plans.md):
  - **DAG Backward Induction**: Leverages the strictly acyclic nature of Ultimate Tic-Tac-Toe (every move decrements empty cells $E$ by 1) for cycle-free retrograde induction.
  - **Topological Isolation for Tier 8**: Proves that with 8 closed subboards, directed moves targeting closed subboards force the opponent back to open subboard $k$, generating Tier 8 in $< 2$ seconds (~83 MB).
  - **Tier 7 Layered Induction**: Evaluates 36 pairs across 2,187 macro states, grouped by boundary condition $\tau$.
  - **Disk-Backed Chunk Streaming for Tier 6**: Partitions 84 triplets into $N$ standalone chunk files with `.done` markers and merges them into `egtb_tier6.bin` with bounded resident memory.

- [`macro_factored.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored_plans.md):
  - **Macro-Factoring Principle**: Parameterizes macro boards via boundary vector $\tau = (xw_j, xw_k, ow_j, ow_k, x_{\text{shared}}, o_{\text{shared}})$ and clamped CodinGame win differential $\Delta \in [-3..+3]$.
  - **Strict Size & Memory Ceilings**: Enforces hard caps on entry counts ($\le 500\text{k}$ for Tier 8, $\le 3.5\text{M}$ for Tier 7/6) and file sizes ($\le 5$ MB for Tier 8, $\le 50$ MB for Tier 7/6) guarded by `MemoryBudgetGuard` (512 MB heap ceiling via `/proc/self/statm`).
  - **Precomputed Transition & Lookup Tables**: Employs `MicroTransitionTable` (11,125 micro states) and `Tier8Lookup` (resident 445 KB table for $O(1)$ Tier 8 backward linkage).
  - **`MacroFactoredProber`**: Delivers $< 30$ ns zero-copy probing with zero allocations.

- [`tier6.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6_plans.md):
  - **13-Bit $\tau_6$ Specification**: Encodes single-board wins, pairwise collinear shared lines, 3-board collinear lines (`line_jkl`), and clamped $\Delta$.
  - **$D_4$ Triplet Symmetry Reduction**: Reduces 84 triplets to 16 canonical orbits (56 center-closed triplets partition into 10 canonical classes; 28 center-open into 6 classes).
  - **Dense Bijection Mapping**: `TRIPLET_TAU_DENSE_TABLE` bounds reachable conditions to $\le 397$ per triplet ($< 512$, 9 bits).
  - **Collision-Free 61-Bit Key**: Packs triplet index, 9-bit dense tau, target, turn, and three 14-bit micro IDs into 61 bits ($< 2^{61}$).
  - **Closed-Form Tier 7 Transition & Hybrid Solver**: `transition_tier6_to_tier7` hooks into `MacroFactoredProber` in $< 100$ ns, integrated with `HybridDfPnSolver`.

- [`tier7_zfs.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs_plans.md):
  - **Exhaustive Dense Matrix Architecture**: Represents all $11,125 \times 11,125 = 123,765,625$ micro-board pairs in contiguous 2D binary matrices (~123.8 MB per slice, ~76 GB total).
  - **Base-3 Outcome Compression**: Densely packs 4 ternary game outcomes (Turn X/O $\times$ Target $j/k$) into a single byte ($\le 80$).
  - **8 Canonical Orbits & 614 Slices**: Partitions 36 open pairs under $D_4$ into 8 canonical orbits and exactly 614 canonical $\tau$ slices.
  - **Parallel ZFS Solver & Zero-Copy Prober**: Implements Rayon-parallel `Tier7ZfsSolver` with atomic writes/manifest export and `Tier7ZfsProber` delivering direct-offset $O(1)$ lookups in $< 15$ ns.

---

## 2. Status & Verification Summary

- **Tier 9**: 19,891 canonical terminal states generated in $< 15$ ms.
- **Tier 8**: Fully solved via topological isolation ($< 2$s) and macro-factoring ($\le 500\text{k}$ entries, $\le 5$ MB).
- **Tier 7**: Dual architecture complete — bounded macro-factored table ($\le 3.5\text{M}$ entries, $\le 50$ MB) and exhaustive ZFS dataset (614 canonical slices, 123.76 MB each).
- **Tier 6**: Depth-stratified retrograde induction, 16-orbit $D_4$ symmetry reduction, collision-free 61-bit key packing, and chunk streaming verified.
- **Probing Latency**: Verified $< 15$ ns for Tier 7 ZFS direct indexing and $< 30$ ns for `MacroFactoredProber`.
