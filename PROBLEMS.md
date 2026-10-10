# Project-Wide Problems & Challenges: Ultimate Tic-Tac-Toe Engine & Solver

This document outlines the overarching architectural problems, theoretical bottlenecks, and computational challenges of the Ultimate Tic-Tac-Toe (UTTT) project across all modules and crates.

---

## 1. High-Level Project Mission & Scope
The mission of this project is to build an unbeatable, mathematically rigorous Ultimate Tic-Tac-Toe AI engine, tournament arena, opening book generator, and endgame tablebase (EGTB) capable of:
1. Solving endgames in real time and tabulating millions of terminal/near-terminal states.
2. Generating a deep opening book filtered by statistical dominance.
3. Competing at peak superhuman strength in tournaments and on competitive platforms (e.g. CodinGame) within strict per-move time budgets (20 ms – 100 ms).
4. Running continuous self-play, tree deepening, and profiling daemons on high-capacity ZFS storage (7.2 TB NVMe array).

---

## 2. Core Architectural Challenges

### 2.1 Combinatorial State Explosion ($10^{27} \dots 10^{30}$ States)
- Ultimate Tic-Tac-Toe has an unconstrained state space of $3^{81} \approx 4.43 \times 10^{38}$.
- Constrained by alternating turns ($\approx 10^{37}$), subboard completions, and directed movement rules, the reachable state space is estimated between $10^{27}$ and $10^{30}$.
- Brute-force search and naive transposition caching fail completely without deep symmetry reduction and canonical subboard representations.

### 2.2 Transposition Divergence in Subboards
- A subboard can be won via 8 distinct lines (rows, columns, diagonals) with arbitrary irrelevant opponent marks.
- Naive bitboards and 64-bit Zobrist hashes treat every permutation of internal marks as distinct, causing identical strategic macro states to diverge in memory and tablebases.

### 2.3 Directed Movement Normalization Anomaly
- When a move directs the opponent to an already resolved subboard, the opponent receives free choice across all open subboards.
- Recording distinct target indices (`last_super_index`) for moves directing into different won subboards causes identical game situations to produce distinct hashes. Normalizing `last_super_index` to `FREE_CHOICE_INDEX = 9` is mathematically mandatory.

### 2.4 Hash Collisions & Cache Footprint: Dual-Layer Hashing
- 64-bit Zobrist hashes face catastrophic birthday-paradox collision risk after $\approx 4 \times 10^9$ evaluations.
- In soft-time search bots (AlphaBeta, MCTS), Zobrist hashes are retained for fast incremental $O(1)$ XOR updates where occasional benign transpositions are tolerable.
- In exact solvers (DF-PN, EGTB), 64-bit Zobrist hashing cannot be trusted. An exact 167-bit bit-packed representation ([`PackedBoardState`](file:///home/festivity/agy/uttt-codingame/uttt/src/packed_state.rs)) fitting in 24 bytes (`[u64; 3]`) is required to guarantee zero hash collisions.

### 2.5 Opening Book: Legacy Bloat vs. Principled Dominance
- **Legacy Bloat (The 108k Issue)**: Early unconstrained MCTS harvesting with a low visit threshold ($N \ge 25$) scraped low-probability exploration noise, ballooning `/my-zraid/uttt/book.bin` to 108,557 entries.
- **The Retention vs. Purification Decision**: While the legacy database was retained on disk, a strict UCT dominance filter ($N_1 \ge 1.35 \times N_2$ or $\ge 40\%$ plurality) was introduced to freeze the addition of unverified moves. New study runs expand the book by only 5–10 verified master lines per cycle. Users desiring a pure book can delete `book.bin` and regenerate from scratch.

### 2.6 Endgame Tablebase Discovery Bottleneck & Solver Thrashing
- Random playouts from ply 0 rarely reach $\ge 7$ resolved subboards.
- Retrograde analysis from terminal states requires computing and holding billions of inverse directed movement transitions in RAM. Forward DF-PN proof search evaluates only reachable on-demand states, but thrashes without 2-ply Grandfather refutation lookahead.

### 2.7 Multi-Platform Compatibility & Timing Constraints
- Code must execute locally with high-performance 64-bit integer bit-parallel logic (bitboards) and multi-threading, while remaining compatible with CodinGame's single-file compilation constraint and 100 ms turn limit. Note: Bit-parallel integer operations on `u128` / `u64` are utilized rather than architecture-specific AVX2 vector intrinsics to maintain 100% portability.

---

## 3. Cross-References & Detailed Documentation
- Architectural Plans & Progress: [`PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- In-Depth Technical Specification: [`documentation/problems_and_challenges.md`](file:///home/festivity/agy/uttt-codingame/documentation/problems_and_challenges.md)
- Complete Implementation Analysis: [`documentation/plans_and_progress.md`](file:///home/festivity/agy/uttt-codingame/documentation/plans_and_progress.md)
- Core Engine Crate Specs: [`uttt/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/uttt/PROBLEMS.md) and [`uttt/PLANS.md`](file:///home/festivity/agy/uttt-codingame/uttt/PLANS.md)
- Test Suite Crate Specs: [`uttt-tests/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/uttt-tests/PROBLEMS.md) and [`uttt-tests/PLANS.md`](file:///home/festivity/agy/uttt-codingame/uttt-tests/PLANS.md)
