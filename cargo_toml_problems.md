# Problems & Challenges: Root `Cargo.toml` Workspace Configuration

This document details workspace partitioning, profile optimization, and build configurations in root `Cargo.toml`.

---

## 1. Challenges
- **Workspace Member Orchestration**: Coordinating multi-crate compilation between the core engine library/binaries (`uttt`) and the integration test harness (`uttt-tests`) using Cargo resolver version 2.
- **Compiler Optimization vs. Debuggability**: High-performance bitboard logic and game tree rollouts require `opt-level = 3` even in development and test profiles to run fast enough for interactive testing, while retaining `debug = true` for line diagnostics and flamegraph generation.
- **Debug Assertion Overhead**: In `profile.dev`, disabling debug assertions (`debug-assertions = false`) is necessary to prevent severe slowdowns during recursive tree traversals.
