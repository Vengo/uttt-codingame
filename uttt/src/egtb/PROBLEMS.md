# Problems & Challenges: `uttt/src/egtb` Endgame Tablebase Module

This document synthesizes the problem domain, technical bottlenecks, edge cases, and mathematical solutions across the endgame tablebase module based on the folder's documentation.

---

## 1. Core Technical Problem Domain

- **Combinatorial State Space Explosion**:
  - Full board enumeration is impossible due to the explosion of closed subboard permutations ($3^8 = 6,561$ for Tier 8; $3^7 = 2,187$ for Tier 7; $3^6 = 729$ for Tier 6) crossed with 11,125 legal micro configurations per open subboard.
  - *Solutions*:
    1. *Topological Isolation* in Tier 8: Proves that directed moves targeting closed boards force play strictly back into the single open subboard, reducing Tier 8 to an isolated single-subboard game ($< 2$s solve time).
    2. *Macro-Factoring*: Replaces thousands of raw macro states with compact boundary condition vectors ($\tau$) and win count differentials ($\Delta$).
    3. *$D_4$ Dihedral Symmetry Reduction*: Collapses 36 open pairs into 8 canonical orbits (Tier 7) and 84 triplets into 16 canonical orbits (Tier 6).

- **Multi-Board Collinear Shared Lines**:
  - Open subboards that are collinear with each other and a closed subboard won by a player create latent multi-board winning threats. Winning one open board instantly activates a winning threat on the other.
  - *Solutions*:
    1. Tier 7 tracks pairwise collinear lines via `x_shared` and `o_shared` in `Tier7MacroTau`.
    2. Tier 6 tracks 6 pairwise shared line flags plus `line_jkl` (3-board collinear line) in `Tier6MacroCondition`, dynamically cascading them into Tier 7 parameters upon subboard completion via `transition_tier6_to_tier7`.

- **CodinGame Saturated Mini-Board Tiebreaker**:
  - CodinGame contest rules resolve 9-subboard ties by comparing won subboard counts ($|M_X|$ vs $|M_O|$).
  - *Solutions*: Clamped integer differentials ($\Delta \in [-3..+3]$ or $[-2..+2]$) are tracked as macro invariants and updated dynamically ($\Delta \pm 1$) whenever a subboard resolves, guaranteeing exact terminal evaluation when remaining empty squares reach $E = 0$.

- **Zero-Collision 64-Bit Key Packing**:
  - Naive bit shifting with insufficient bit allocation causes micro-ID overflows (e.g., $l_{\text{id}} \ge 4096$ colliding with $k_{\text{id}} = 1$) and pseudo-random hash transpositions.
  - *Solutions*:
    1. 14 bits allocated for each micro ID ($11,125 < 16,384$).
    2. Reachable macro conditions precomputed into monotonic lists (`TRIPLET_TAU_DENSE_TABLE`), mapping conditions into a dense 9-bit index ($\le 397 < 512$).
    3. The complete state packs into 61 bits ($< 2^{61} \le 64$), providing a provably collision-free bijection.

- **Storage & Probing Latency Trade-Offs**:
  - Forward search engines (MCTS, Alpha-Beta) require tens of thousands of evaluations per second without GC pauses or heap allocations.
  - *Solutions*:
    1. *Direct Offset Probing* (`tier7_zfs.rs`): Contiguous $11,125 \times 11,125$ matrix slices indexed directly in $O(1)$ at $j_{\text{id}} \times 11,125 + k_{\text{id}}$ with base-3 byte outcome packing, executing in $< 15$ ns.
    2. *Memory-Mapped Probing* (`macro_factored.rs` & `mod.rs`): Direct 10-byte binary search records loaded via `memmap2::Mmap`, delivering $< 30$ ns query times.
    3. *Partitioned Tier MMaps*: Slices segmented by resolved subboard count $k$, avoiding binary searches over irrelevant game phases.

- **System Memory Safety & Scalable Chunking**:
  - Running multi-threaded retrograde DP over large state spaces can trigger severe memory spikes and process aborts.
  - *Solutions*:
    1. [`MemoryBudgetGuard`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/macro_factored_problems.md) polls `/proc/self/statm` on Linux and enforces a strict 512 MB heap ceiling.
    2. Disk-backed chunk streaming (`retrograde.rs`) partitions Tier 6 solving across independent chunk files with `.done` markers and merges them with bounded memory.
    3. Idempotent atomic writing (`.tmp.<pid>` rename) ensures data integrity across restarts on ZFS storage.
