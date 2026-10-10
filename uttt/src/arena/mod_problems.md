# Problems & Challenges: `uttt/src/arena/mod.rs`

This document details module exports and arena subsystem aggregation in `uttt/src/arena/mod.rs`.

---

## 1. Challenges
- Exposing clean, unified interfaces for games, tournaments, Elo calculations, server daemon, and profiling without circular dependency loops across bot and engine crates.
