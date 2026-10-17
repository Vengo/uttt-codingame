# Problems & Challenges: `uttt/src/arena/replay.rs`

This document details game replay serialization and browser visualization challenges in `uttt/src/arena/replay.rs`.

---

## 1. Challenges
- **JSON Serialization Overhead**: Storing rich per-move debug information (timestamps, move indices, subboard coordinates, and ASCII boards) while maintaining compact JSON payloads.
- **Embedded Web Player**: Providing an interactive HTML5/Canvas visualization player directly from the binary without relying on node/npm or external CDNs.
