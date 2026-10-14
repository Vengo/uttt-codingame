# Problems & Challenges: `uttt/src/egtb/tier7_zfs.rs`

This document details the technical problems, invariant handling, and algorithmic challenges solved in [`uttt/src/egtb/tier7_zfs.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/tier7_zfs.rs).

---

## 1. Technical Challenges

- **Combinatorial Explosion of Tier 7 States**:
  With 2 open subboards, there are 36 open board pairs and $11,125 \times 11,125 \approx 123.77 \times 10^6$ micro configurations per pair. Unreduced enumeration would yield over $4.45 \times 10^9$ state pairs across $3^7 = 2,187$ macro conditions, demanding petabytes of storage.
  - *Solution*: By reducing pairs under $D_4$ dihedral symmetry into 8 canonical orbits and factoring macro boundary invariants into `Tier7MacroTau`, the problem collapses into exactly 614 canonical slices, fitting within ~76 GB uncompressed storage.

- **Dense Direct Indexing vs. Hash Probing Overhead**:
  Searching 64-bit Zobrist hashes via binary search over tens of millions of records incurs $O(\log N)$ branch penalties and memory latency during deep tree rollouts.
  - *Solution*: Each slice is stored as a contiguous, row-major $11,125 \times 11,125$ matrix where byte offset $j_{\text{id}} \times 11,125 + k_{\text{id}}$ directly returns the outcome in $O(1)$ time ($< 15$ ns) with zero hash searching.

- **Ternary Base-3 Outcome Compression**:
  Each micro pair must store the game result for both players' turns ($X$ or $O$) and both target subboard directives (forced to $j$, or forced to $k$). Free choice can be computed on the fly as $\max(v_j, v_k)$.
  - *Solution*: Storing 4 ternary values ($3^4 = 81$) fits inside a single byte ($\le 80$):
    $$\text{byte} = v_1 + 3 \cdot v_2 + 9 \cdot v_3 + 27 \cdot v_4$$
    Decoding takes simple integer arithmetic (`byte % 3`, `(byte / 3) % 3`, `(byte / 9) % 3`, `(byte / 27) % 3`).

- **Retrograde Induction Linkage into Tier 8**:
  When a move completes subboard $j$ or $k$ (by win or tie), the game transitions from Tier 7 to Tier 8.
  - *Solution*: The backward induction step hooks directly into the 445 KB resident `Tier8Lookup` table. Winning a subboard updates macro collinear flags (activating `x_shared` or `o_shared` into single-board wins $xw_k'$ or $ow_k'$) and updates CodinGame win differentials $\Delta \pm 1$.

- **Memory Management Across Concurrent Threads**:
  Allocating dynamic DP layers across multiple threads can cause severe resident memory spikes (each worker thread requires ~708 MB for `PairLayerTable` structures plus a 124 MB output buffer).
  - *Solution*: Memory is pre-allocated and reused per thread within a bounded Rayon pool, keeping total process RSS stable and preventing Out-Of-Memory termination.

- **Atomic Persistence and Idempotency**:
  Long-running tablebase solving over network-attached or ZFS pools risks corrupted partially written files if interrupted.
  - *Solution*: Each worker writes to a temporary file (`.tmp.<pid>`), flushes buffers, and renames atomically to the target file. The solver checks existing files for exact length (`123,765,625` bytes) to skip already solved slices cleanly upon restart.
