# Problems & Challenges: `uttt/src` Core Engine Module

This document details the architectural problems, bitboard mechanics, and memory representation challenges in `uttt/src`.

---

## 1. Responsibilities of `uttt/src`
`uttt/src` is the central core of the Ultimate Tic-Tac-Toe engine. It houses:
- Representation primitives: `canonical.rs`, `packed_state.rs`, `symmetry.rs`.
- Rule engines & move generators: `ttt_trait_solving.rs`, `uttt_trait_solving.rs`.
- Universal abstractions & interfaces: `game_solving.rs` (`GameMove`, `GameResult`, `AlphaBetaCompatible`).
- Profilers & hardware benchmarking: `profile.rs` (cache latency, allocator speed, search throughput).
- Knowledge stores: `opening_book.rs`.

---

## 2. Core Technical Problems
1. **Transposition Divergence in Subboards**: Distinct mark configurations yielding the same macro result must collapse to identical states.
2. **Directed Subboard Target Inconsistency**: Moves directing into resolved subboards must normalize to free choice (`FREE_CHOICE_INDEX = 9`).
3. **Subboard Tie Detection Inversion**: Guarding `get_tied_mask()` so canonical won boards (`CANONICAL_X_WIN = 0x1FF`) are never misclassified as draws.
4. **Zero-Collision 167-Bit State Packing**: Storing 81 X cells, 81 O cells, 4-bit target, and 1-bit turn in 24 bytes (`[u64; 3]`).
5. **High-Speed D4 Dihedral Symmetry**: Applying rotations and reflections simultaneously to macro subboards and micro cells in a single pass.
6. **Low-Latency Opening Book Lookups**: Fast zero-allocation queries (< 1 $\mu$s) in master opening lines.

---

## 3. Cross-References
- Root Architecture: [`/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md) and [`/PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- In-Depth Technical Specification: [`/documentation/problems_and_challenges.md`](file:///home/festivity/agy/uttt-codingame/documentation/problems_and_challenges.md)
