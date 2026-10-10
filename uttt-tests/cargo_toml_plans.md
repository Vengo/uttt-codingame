# Plans & Progress: `uttt-tests/Cargo.toml`

This document details the configuration and status for `uttt-tests/Cargo.toml`.

---

## 1. Plan & Dependencies
- Path dependency `uttt = { path = "../uttt" }` directly links the core engine without version skew.
- `serde_json` for verifying replay and tournament JSON serialization.
- `rand` for property-based randomized game playout testing.

---

## 2. Status
- Operational, fast, and builds without warnings or outdated crates.
