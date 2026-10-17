# Track 1 Plan: Tier 6 & Tier 7 ZFS EGTB Solver Speedup & Prober Completion

**Status**: Reviewed and Approved by Isolated Plan Reviewer (`e024ce35-51ac-41a5-ad93-83f73cc1c9b6`)

---

## 1. Problem Description

### Problem 1A: Inner-Loop Global Mutex, Heap Allocation & Redundant $D_4$ Symmetry Lookups in `Tier6ZfsSolver` $\to$ `Tier7ZfsProber::probe_raw`
During Tier 6 ZFS tablebase generation (`Tier6ZfsSolver::solve_single_tau_slice` in `uttt/src/egtb/tier6_zfs.rs`), backward induction solves layers `total_e = 1..=max_e` (where `max_e = 4`, totaling `1,172,707,629` micro-state triplets per slice across Rayon threads).
Whenever a micro-move in one of the 3 open subboards `(j, k, l)` results in `MicroMoveOutcome::Win` or `MicroMoveOutcome::Tie`, it calls `transition_to_tier7_zfs`, which calls `Tier7ZfsProber::probe_raw`.

Because `total_e = 3` (`(1,1,1)`) and `total_e = 4` (`(2,1,1), (1,2,1), (1,1,2)`) consist almost entirely of subboards with `e = 1` empty square (where every move resolves the subboard into `Win` or `Tie`), `probe_raw` is executed billions of times per slice across all Rayon worker threads.
Every call currently:
1. Loops over all 8 $D_4$ symmetries in `map_to_canonical_pair(rem_a, rem_b)`.
2. Binary-searches `self.orbit_taus[pair_map.orbit_id].binary_search(&target_tau)`.
3. Performs 4 bitboard rotations (`pair_map.sym.transform_9bit(...)`) and 2 `micro.to_id` lookups.
4. Allocates a `String` on the heap (`format!("tier7_orbit_{}_tau_{:03}.bin", ...)`), locks a global `Mutex<HashMap<String, Mmap>>`, and hashes the filename string.

### Problem 1B: Unreachable `e_j == 0 || e_k == 0 || e_l == 0` Layers Computed During Tier 6 Backward Induction
In Tier 6, all 3 open subboards `(j, k, l)` must have at least 1 empty square (`e_j >= 1, e_k >= 1, e_l >= 1`). Moreover, when `e = 1`, playing the last empty square always fills the subboard (`(next_x | o) == 0x1FF`), producing `MicroMoveOutcome::Win` or `MicroMoveOutcome::Tie` and NEVER `MicroMoveOutcome::Next`.
Therefore:
- Layers with `e_j == 0 || e_k == 0 || e_l == 0` are NEVER read by any valid Tier 6 state (`e_j, e_k, e_l >= 1`) during backward induction (`total_e = 3` only transitions to Tier 7; `total_e = 4` only transitions to `(1,1,1)` or Tier 7).
- To preserve 100% binary file format compatibility with already-generated `tier6_orbit_6_tau_*.bin` files on disk (`2,345,415,274` bytes per `.bin` slice), the on-disk layer layout must remain identical while skipping the inner-loop computation for `e_j == 0 || e_k == 0 || e_l == 0`.

### Problem 1C: `Tier6ZfsProber::probe` is an Unimplemented Stub
In `uttt/src/egtb/tier6_zfs.rs`, `Tier6ZfsProber::probe` currently ignores `_mmap` and returns a static `target_tau.delta` tiebreak instead of canonicalizing the 3 open subboards under $D_4$, computing the induced $S_3$ permutation of `(j, k, l)`, indexing into the memory-mapped `.bin` slice, and unpacking the solved `u16` outcome.

### Problem 1D: `EgtbDatabase::probe_with_constants` Queries Sparse `macro_prober` Before Exhaustive `zfs_tier7_prober` and Does Not Wire `Tier6ZfsProber`
In `uttt/src/egtb/mod.rs`, `EgtbDatabase` queries the downsampled 50 MB `macro_prober` (via binary search) before the exhaustive `<15 ns` direct-indexed `zfs_tier7_prober`, and does not yet include `zfs_tier6_prober: Option<Arc<Tier6ZfsProber>>`.

---

## 2. Implementation Plan

### Step 1: Lock-Free Pre-Mapped `Tier7ZfsProber` & Precomputed `sym_micro_id` Table (`uttt/src/egtb/tier7_zfs.rs`)
1. Add `pub sym_micro_id: Box<[[u16; TOTAL_MICRO_STATES]; 8]>` to `Tier7ZfsProber` (computed once in `Tier7ZfsProber::new` for all 8 symmetries in `D4Symmetry::ALL` and all `id in 0..TOTAL_MICRO_STATES` by transforming `(x, o) = micro.states[id]` with `sym.transform_9bit` and looking up `micro.to_id`).
2. Replace `open_mmaps: Mutex<HashMap<String, Mmap>>` with a pre-indexed lock-free table `pub mmaps: [Vec<std::sync::OnceLock<Option<Mmap>>>; 6]` sized to `orbit_taus[orbit_id].len()` for each of the 6 orbits.
3. Add `pub fn get_slice(&self, orbit_id: usize, tau_idx: usize) -> Option<&[u8]>` which lazily opens and maps `tier7_orbit_{orbit_id}_tau_{tau_idx:03}.bin` inside `self.mmaps[orbit_id][tau_idx].get_or_init(...)` (verifying `mmap.len() == SLICE_BYTE_SIZE`) and returns `mmap.as_deref()`.
4. Update `Tier7ZfsProber::probe` and `Tier7ZfsProber::probe_raw` to use `self.sym_micro_id` (after checking `raw_id < TOTAL_MICRO_STATES`) and `self.get_slice(pair_map.orbit_id, tau_idx)`.

### Step 2: Pre-Resolved Tier 7 Transition Table & Skipping `e == 0` Compute in `Tier6ZfsSolver` (`uttt/src/egtb/tier6_zfs.rs`)
1. Define a pre-resolved lookup structure `PreboundTier6To7<'a>` created once at the start of `solve_single_tau_slice` for a given `(j, k, l)` and `tau6`:
   - Indexed by `[sb_idx][res_idx]` where `sb_idx in 0..3` corresponds to `j (0), k (1), l (2)` and `res_idx in 0..3` corresponds to `XWin (0), OWin (1), Tie (2)`.
   - Each entry is an enum:
     - `Immediate(EgtbOutcome)` if resolving `sb_idx` immediately completes a macro 3-in-a-row (`tau6.xw_j`, `tau6.ow_j`, etc.).
     - `Tier7Slice { slice: Option<&'a [u8]>, sym_idx: usize, flipped: bool, rem_a: usize, rem_b: usize, fallback: EgtbOutcome }` which precomputes `rem_a`, `rem_b`, `pair_map = map_to_canonical_pair(rem_a, rem_b)`, `target_tau`, `tau_idx`, and `slice = zfs_prober.get_slice(pair_map.orbit_id, tau_idx)` once per slice!
   - Provide an `#[inline(always)]` method `eval(&self, sb_idx: usize, res_idx: usize, last_sq: usize, turn_was_x: bool, micro_ids: (usize, usize, usize), sym_micro_id: &[[u16; TOTAL_MICRO_STATES]; 8]) -> EgtbOutcome` that performs the exact lookup into `slice` in ~5 instructions with zero locks, zero allocations, and zero bitboard transforms.
2. In `solve_single_tau_slice`:
   - Keep `tables` allocated for `0..=max_e` (preserving the exact `2,345,415,274`-byte `.bin` file layout).
   - In the backward induction loop (`for total_e in 1..=max_e`), skip any layer where `e_j == 0 || e_k == 0 || e_l == 0` (`if e_j == 0 || e_k == 0 || e_l == 0 { continue; }`).

### Step 3: Complete Implementation of `Tier6ZfsProber::probe` (`uttt/src/egtb/tier6_zfs.rs`)
1. Update `Tier6ZfsProber` to hold:
   - `pub mmaps: [Vec<std::sync::OnceLock<Option<Mmap>>>; 16]`
   - `pub sym_micro_id: Box<[[u16; TOTAL_MICRO_STATES]; 8]>`
   - `pub layer_byte_offsets: [[[usize; 5]; 5]; 5]` (precomputed in `Tier6ZfsProber::new` starting at byte offset `16` in the exact loop order `for total_e in 0..=4 { for e_j in 0..=total_e { for e_k in 0..=(total_e - e_j) { let e_l = total_e - e_j - e_k; ... } } }`, where each layer adds `n_j * n_k * n_l * 2` bytes).
2. In `Tier6ZfsProber::probe(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome>`:
   - Check immediate macro win on `board.x_supers` (`P1Win`) and `board.o_supers` (`P2Win`).
   - Identify the 3 open subboards `open = [s0, s1, s2]` (`s0 < s1 < s2`). If `open_cnt != 3`, return `None`.
   - Compute `let canon = canonical_triplet(s0, s1, s2);`.
   - Find the permutation of `(s0, s1, s2)` that maps to `(canon.rep_j, canon.rep_k, canon.rep_l)` under `canon.sym.transform_3x3(s)`:
     - Specifically, for each `s in open`, compute `t = canon.sym.transform_3x3(s)` and assign `orig_j = s` if `t == canon.rep_j`, `orig_k = s` if `t == canon.rep_k`, `orig_l = s` if `t == canon.rep_l`.
   - Compute empty counts in the canonical frame:
     - `e_j = 9 - (board.xbb[orig_j] | board.obb[orig_j]).count_ones() as usize`
     - `e_k = 9 - (board.xbb[orig_k] | board.obb[orig_k]).count_ones() as usize`
     - `e_l = 9 - (board.xbb[orig_l] | board.obb[orig_l]).count_ones() as usize`
     - Require `e_j >= 1 && e_k >= 1 && e_l >= 1 && e_j + e_k + e_l <= DEFAULT_MAX_EMPTY_CELLS`.
   - Transform macro bitboards:
     - `canon_x = canon.sym.transform_9bit(board.x_supers);`
     - `canon_o = canon.sym.transform_9bit(board.o_supers);`
     - `target_tau = extract_tier6_tau(canon_x, canon_o, canon.rep_j, canon.rep_k, canon.rep_l);`
     - Binary-search `self.orbit_taus[canon.orbit_id]` by `encode_u16()` to find `tau_idx`.
   - Retrieve `slice = self.get_slice(canon.orbit_id, tau_idx)?`.
   - Verify `slice.len() >= 16 && &slice[0..8] == b"UTTT_T6S"`. Read `slice_max_e = u32::from_le_bytes(slice[8..12].try_into().ok()?) as usize` and verify `e_j + e_k + e_l <= slice_max_e`.
   - Look up raw micro IDs `raw_j`, `raw_k`, `raw_l` for `orig_j`, `orig_k`, `orig_l` in `self.micro.to_id`; check `< TOTAL_MICRO_STATES`.
   - Map through `sym_idx = canon.sym as usize`:
     - `j_id = self.sym_micro_id[sym_idx][raw_j] as usize`
     - `k_id = self.sym_micro_id[sym_idx][raw_k] as usize`
     - `l_id = self.sym_micro_id[sym_idx][raw_l] as usize`
   - Look up dense indices within each empty-count stratum:
     - `idx_j = self.micro.dense_idx[j_id] as usize`
     - `idx_k = self.micro.dense_idx[k_id] as usize`
     - `idx_l = self.micro.dense_idx[l_id] as usize`
   - Compute flat index and byte offset:
     - `n_k = self.micro.states_by_empty[e_k].len();`
     - `n_l = self.micro.states_by_empty[e_l].len();`
     - `elem_idx = (idx_j * n_k + idx_k) * n_l + idx_l;`
     - `byte_offset = self.layer_byte_offsets[e_j][e_k][e_l] + elem_idx * 2;`
     - Check `slice.len() >= byte_offset + 2`, read `packed = u16::from_le_bytes([slice[byte_offset], slice[byte_offset + 1]])`.
   - Map `board.last_super_index` to canonical `target`:
     - `0` if `board.last_super_index == orig_j as u8`
     - `1` if `board.last_super_index == orig_k as u8`
     - `2` if `board.last_super_index == orig_l as u8`
     - `3` otherwise (Free Choice)
   - Return `Some(unpack_outcome(packed, target, board.x_to_move, e_j, e_k, e_l))`.

### Step 4: Update `EgtbDatabase` Probing Order (`uttt/src/egtb/mod.rs`)
1. Add `pub zfs_tier6_prober: Option<Arc<tier6_zfs::Tier6ZfsProber>>` to `EgtbDatabase` and initialize it from candidate directories (`self.base_dir.join("tier6")`, `/my-zraid/uttt/egtb/tier6`, `egtb/tier6`).
2. In `EgtbDatabase::probe_with_constants`:
   - Check immediate macro wins first (`check_subboard_win`).
   - Dispatch by `resolved = board.get_resolved_count()`:
     - When `resolved == 7`, query `self.zfs_tier7_prober` FIRST before falling back to `self.macro_prober`.
     - When `resolved == 6`, query `self.zfs_tier6_prober` FIRST before falling back to `self.macro_prober`.
     - For `resolved == 8` or `9` (or fallback), query `self.macro_prober`.

---

## 3. Testing Plan
1. **`test_sym_micro_id_table_consistency` (`tier7_zfs.rs`)**:
   - Verify for all 8 `D4Symmetry` elements and all 5,478 valid micro states that `sym_micro_id` matches bitboard transformation + `to_id` lookup and preserves empty square counts.
2. **`test_prebound_tier6_to_tier7_equivalence` (`tier6_zfs.rs`)**:
   - Verify across sample triplets, `tau6` configurations, and micro-state triplets that `PreboundTier6To7::eval` produces identical results to `transition_to_tier7_zfs`.
3. **`test_tier6_zfs_solver_and_prober_d4_invariance_roundtrip` (`tier6_zfs.rs`)**:
   - Solve a Tier 6 slice with `max_e = 3` in a temporary test directory using a mock/temp Tier 7 slice, construct `UltimateTicTacToeBoard` states for `(1,1,1)` across all 8 $D_4$ symmetries (covering all $S_3$ permutations of the 3 open subboards and all 4 target modes), and assert that `Tier6ZfsProber::probe` returns the exact minimax outcome and is 100% invariant across all 8 $D_4$ board transformations.
