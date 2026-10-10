# Problems & Challenges: `uttt/build.rs`

This document details the build-time requirements and linking constraints of `uttt/build.rs`.

---

## 1. Responsibilities & Challenges
- **Stack Size Overflows in Deep Recursion**: Deep minimax search or DF-PN traversal can exhaust standard thread stack sizes (default 2 MB on Linux). Setting native linker arguments (e.g. `/stack:8388608` or equivalent `-C link-args`) must be handled cross-platform without breaking standard Cargo builds.
- **Cross-Platform Compatibility**: Ensuring build scripts do not break on Linux or Windows when passing platform-specific linker flags.
