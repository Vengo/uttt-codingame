# Problems & Challenges: `uttt` Core Engine Crate

This document outlines the technical challenges, architecture constraints, and computational goals specific to the `uttt` crate (`/uttt`).

---

## 1. Role and Responsibilities of `uttt`
The `uttt` crate contains the fundamental bitboard representations, symmetry pipelines, depth-first proof number (DF-PN) solvers, endgame tablebases (EGTB), Monte Carlo and Minimax search bots, and the arena tournament server.

---

## 2. Key Technical Problems in `uttt`

### 2.1 Low-Latency Move Generation & Bit-Parallel Integer Logic
- Move generation must execute in nanoseconds to enable millions of rollout simulations per second.
- Uses bitboards (`u16` per subboard for 9 subboards, plus superboard win tracking) with 64-bit integer bit-parallel arithmetic. Architecture-independent bit-parallel operations are used in place of CPU-specific AVX2 vector intrinsics to ensure compatibility with CodinGame.

### 2.2 Memory Alignment & Cache Locality in Search
- During recursive tree search (MCTS, AlphaBeta, DF-PN), node representations must be compact to fit inside L1/L2 CPU caches without frequent heap allocations.

### 2.3 Exact State Packing without Loss of Information
- Compressing an 81-square board with side-to-move and target index into 167 bits (24 bytes, `[u64; 3]`) without losing fast bitboard unpack performance.

### 2.4 Transposition Collapse via Symmetry
- Applying dihedral group ($D_4$) operations efficiently to both macro subboards and micro cells in a unified transform, with deterministic canonical representation selection.

### 2.5 Multi-Target Binary Compilation
- Crate must simultaneously support the competitive single-file bundler binary (`cg.rs`), local benchmarks (`runlocal.rs`), and the high-throughput multi-threaded studio CLI (`arena.rs`).

---

## 3. Cross-References
- Root Architecture: [`/PROBLEMS.md`](file:///home/festivity/agy/uttt-codingame/PROBLEMS.md) and [`/PLANS.md`](file:///home/festivity/agy/uttt-codingame/PLANS.md)
- Crate Plans: [`/uttt/PLANS.md`](file:///home/festivity/agy/uttt-codingame/uttt/PLANS.md)
- In-Depth Technical Specification: [`/documentation/problems_and_challenges.md`](file:///home/festivity/agy/uttt-codingame/documentation/problems_and_challenges.md)
