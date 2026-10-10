# Plans & Progress: `uttt/src` Core Engine Module

This document outlines the solutions, module decomposition, and current progress in `uttt/src`.

---

## 1. Architectural Architecture & Module Mapping
- [`canonical.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/canonical.rs): In-place canonicalization replacing won subboards with `0x1FF` (loser zeroed) and ties with `0x0E3 / 0x11C`. Normalizes `last_super_index` to 9 for resolved targets.
- [`packed_state.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/packed_state.rs): `PackedBoardState` bit-packing 167 bits into `[u64; 3]` (24 bytes, 8-byte aligned) with SplitMix64 `fast_hash`.
- [`symmetry.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/symmetry.rs): Dihedral group $D_4$ transform (`transform_3x3` and `transform_9bit`), with canonical form chosen via lexicographical `is_less`.
- [`uttt_trait_solving.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/uttt_trait_solving.rs): Core 9x9 board representation, move generator, and guarded tie detection:
  ```rust
  let full = (self.xbb[i] | self.obb[i]) == 0x1FF;
  let not_won = ((self.x_supers | self.o_supers) & (1 << i)) == 0;
  if full && not_won { tied_mask |= 1 << i; }
  ```
- [`game_solving.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/game_solving.rs): Provides `GameMove` (2 bytes: turn + move index $0 \dots 80$), `GameResult`, and the universal `AlphaBetaCompatible` trait.
- [`profile.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/profile.rs): Benchmarks L3 cache latency vs. DRAM table spilling, allocation latency, and bot rollouts/sec.
- [`opening_book.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/opening_book.rs): Memory-mapped binary opening book with UCT dominance filtering ($N_1 \ge 1.35 \times N_2$ or $\ge 40\%$ plurality, $N_{\text{node}} \ge 100, N_1 \ge 50$).

---

## 2. Implementation Progress & Status
- **Zero Hash Collisions**: Verified exact equality and ordering over millions of states in `PackedBoardState`.
- **Transposition Collapses**: Unit tests prove identical packed canonical states for different winning lines.
- **All Modules Integrated**: Fully utilized by bots, DF-PN solver, and endgame tablebases.

---

## 3. Cross-References
- Root Architecture: [`/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md) and [`/PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- In-Depth Technical Specification: [`/documentation/plans_and_progress.md`](file:///home/festivity/agy/uttt-codingame/documentation/plans_and_progress.md)
