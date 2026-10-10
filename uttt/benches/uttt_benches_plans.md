# Plans & Progress: `uttt/benches/uttt_benches.rs`

This document details 9x9 Criterion benchmark implementations in `uttt/benches/uttt_benches.rs`.

---

## 1. Benchmarks
- `uttt_movegen_forced`: Benchmarks move generation when confined to a specific subboard.
- `uttt_movegen_free_choice`: Benchmarks move generation across all remaining open subboards.
- `uttt_rollout_benchmark`: Benchmarks raw simulation speed from root to terminal outcome.

---

## 2. Status
- Operational Criterion harness for engine regression detection.
