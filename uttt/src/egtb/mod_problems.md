# Problems & Challenges: `uttt/src/egtb/mod.rs`

This document details implementation-level challenges, multi-tier indexing, and probing synchronization in [`uttt/src/egtb/mod.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/mod.rs).

---

## 1. Technical Challenges

- **Multi-Tier Tablebase Partitioning vs. Monolithic Scaling**:
  Storing all endgame positions in a single monolithic tablebase file (`egtb_4plus.bin`) causes file sizes to grow beyond practical single-mmap limits, increasing binary search depth ($O(\log N)$) across unrelated game phases.
  - *Solution*: `EgtbDatabase` partitions tablebases by resolved subboard count $k \in 0..=9$ into `egtb_tier{k}.bin` files while retaining a backward-compatible monolithic fallback. Table queries index directly into `tier_mmaps[k]`, eliminating search comparisons over unrelated tiers.

- **Harmonizing Ultra-Fast Macro Probing with Hash Table Lookups**:
  The introduction of `MacroFactoredProber` provides instant $< 30$ ns queries for structured endgames, but legacy or sparse solved positions reside in Zobrist hash binary files.
  - *Solution*: A strict three-tier priority hierarchy is implemented: (1) `MacroFactoredProber` is queried first for zero-allocation instant cutoffs, (2) partitioned tier mmaps are searched by resolved count $k$, and (3) monolithic mmap is queried as a last resort.

- **Concurrent Memory-Mapped File Updates**:
  Regenerating or extending a tier tablebase file while an existing `Mmap` handle is active can cause segmentation faults or operating system lock contention.
  - *Solution*: `persist_tier_entries` explicitly sets `self.tier_mmaps[tier] = None` to unmap the active file before truncating and rewriting. Once the new records are flushed to disk, `load_mmap()` re-establishes fresh memory-mapped views.

- **Contradictory and Impossible Macro Configurations**:
  During systematic generation over ternary combinations ($3^9$ for Tier 9, $3^8$ for Tier 8, $3^7$ for Tier 7), combinations can arise where both players simultaneously hold 3-in-a-row on the macro-board.
  - *Solution*: `EgtbDatabase` explicitly tests `x_won_super && o_won_super` and skips contradictory configurations, preventing corrupt or invalid records from entering the database.

- **Dual-Layer Primary Storage with Automatic Fallback**:
  Daemon runs target the primary NVMe ZFS dataset at `/my-zraid/uttt/egtb/`, but local compilation or offline CI environments may not have the mount point available.
  - *Solution*: Path resolution checks for directory existence at initialization, falling back gracefully to relative paths (`egtb/` or `.`) without runtime crashes.
