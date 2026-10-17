# Problems & Challenges: `uttt/src/bin/arena.rs`

This document details CLI architecture and continuous daemon challenges in `uttt/src/bin/arena.rs`.

---

## 1. Challenges
- **Continuous Study Daemon Resource Orchestration**: Orchestrating three diverse learning stages per cycle:
  1. Opening book deepening with `UmctsV2Bot` along active master frontiers.
  2. Systematic endgame solving across progressive tiers (Tier 9 instant, Tier 8 high-speed DF-PN, Tier 7).
  3. Self-play game generation for deep learning training export.
- **Crash-Safe Live Telemetry**: Writing live JSON status updates atomically to both local directory (`study_status.json`) and NVMe ZFS storage (`/my-zraid/uttt/study_status.json`) so remote monitoring dashboards never read corrupted half-written files.
