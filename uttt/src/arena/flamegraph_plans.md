# Plans & Progress: `uttt/src/arena/flamegraph.rs`

This document details flamegraph generation implementation in `uttt/src/arena/flamegraph.rs`.

---

## 1. Plan & Sampling Architecture
- **`FlamegraphSampler`**:
  - `profile_bot(bot_name, sample_duration) -> Result<Vec<u8>, String>`:
    Uses `pprof::ProfilerGuardBuilder::default().frequency(999).blocklist(&["libc", "libgcc", "pthread", "ld"]).build()` to sample the call stack during live `choose_move` executions.
  - Generates standalone interactive SVG bytes using `report.flamegraph(&mut buffer)`.
- **CLI Integration**:
  Exposed via `arena flamegraph --bot <name> --time-ms <T> --out <file.svg>`.

---

## 2. Status
- Operational on Linux. Produces clean, readable SVGs revealing search hotspots.
