# Plans & Progress: `uttt/src/egtb/mod.rs`

This document details the endgame tablebase management interface, persistence, multi-tier memory-mapped probing, and systematic generation in [`uttt/src/egtb/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs).

---

## 1. Architectural Architecture & Submodules

- **Submodule Decomposition**:
  - [`retrograde`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/retrograde.rs): Bottom-up retrograde dynamic programming, topological isolation for Tier 8, and disk chunk streaming for Tier 6.
  - [`macro_factored`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored.rs): Boundary-factored solver, saturated mini-board win tiebreaker, memory budget guard, and ultra-fast zero-copy prober.
  - [`tier6`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier6.rs): 13-bit $\tau_6$ condition specification, 16-orbit $D_4$ symmetry reduction, collision-free 61-bit key packing, and hybrid DF-PN solver.
  - [`tier7_zfs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs): Exhaustive direct-indexed $11,125 \times 11,125$ matrix solver and prober on ZFS storage.

- **Unified Probing Architecture ([`EgtbDatabase`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs#L46))**:
  - **Tier-Specific Memory Mapping**: Holds partitioned mmap handles `tier_mmaps: [Option<Mmap>; 10]` and entry counts `tier_counts: [usize; 10]`, eliminating cross-tier search overhead.
  - **Monolithic Fallback Memory Mapping**: Maintains fallback handle `monolithic_mmap` for unified legacy files (`egtb_4plus.bin`).
  - **Ultra-Compact Macro Prober**: Integrates `MacroFactoredProber` covering Tiers 9, 8, 7, and 6 with $< 30$ ns query times.
  - **Exhaustive Tier 7 ZFS Prober (`Tier7ZfsProber`)**: Direct-index memory-mapped access over the 614 canonical slices on `/my-zraid/uttt/egtb/tier7/` in $< 15$ ns.
  - **Multi-Level Probing Priority**:
    1. Direct query via `macro_prober.probe_game_result(board)`.
    2. Direct query via `zfs_tier7_prober.probe(board)` whenever resolved subboard count $\ge 7$.
    3. Direct binary search in `tier_mmaps[k]` where $k = \text{board.get\_resolved\_count()}$.
    4. Fallback binary search in `monolithic_mmap`.

- **ZFS & Local Storage Hierarchy**:
  - Primary path defaults to `/my-zraid/uttt/egtb/` (7.2 TB NVMe ZFS array) if available.
  - Automatic fallback to local workspace paths (`egtb/` or `.`) if ZFS is unmounted.
  - Constructors support explicit directory (`with_dir`) and single file (`with_path`) customization.

- **Systematic Tablebase Generation**:
  - `generate_tier_9`: Generates all 19,891 canonical terminal states in $< 15$ ms in $O(1)$ by superboard lines.
  - `generate_tier_8`: Solves 8-resolved configurations via `DfPnSolver` with Grandfather lookahead across the 9 subboards and $3^8 = 6,561$ macro combinations.
  - `generate_tier_7`: Solves 7-resolved configurations via `DfPnSolver` across all 36 active subboard pairs.
  - `generate_egtb`: Unified high-level entry point routing generation according to `min_resolved`.
  - `persist_tier_entries`: Atomically sorts, dedups, and writes 10-byte binary records to `egtb_tier{k}.bin` and synchronizes the monolithic fallback file.

---

## 2. Types & Public APIs

- `EgtbOutcome`: Enum (`Draw = 0`, `P1Win = 1`, `P2Win = 2`, `Unknown = 3`) with `From<u8>`.
- `EgtbDatabase`: Unified tablebase management and query interface.
  * `new() -> Self`, `with_dir(dir: &Path) -> Self`, `with_path(path: &Path) -> Self`.
  * `load_mmap() -> Result<usize, String>`.
  * `count_resolved_subboards(board: &UltimateTicTacToeBoard) -> usize`.
  * `probe(board: &UltimateTicTacToeBoard) -> Option<GameResult>`.
  * `probe_with_constants(board, zobrist_constants) -> Option<GameResult>`.
  * `generate_tier_9(zobrist_constants) -> Result<usize, String>`.
  * `generate_tier_8(max_to_solve, legal_lookup, zobrist_constants) -> Result<usize, String>`.
  * `generate_tier_7(max_to_solve, legal_lookup, zobrist_constants) -> Result<usize, String>`.
  * `generate_egtb(new_target_positions, min_resolved, legal_lookup, zobrist_constants) -> Result<usize, String>`.

---

## 3. Status & Verification

- Tiered tablebase loading and zero-copy binary search verified.
- Seamless coordination between `MacroFactoredProber`, partitioned tier files, and monolithic fallbacks verified.
