# Plans & Progress: `uttt/build.rs`

This document details the configuration and progress of `uttt/build.rs`.

---

## 1. Design & Plan
- Keep `build.rs` lightweight to ensure fast build times.
- Allow optional stack growth directives when compiling memory-intensive deep-search binaries.
- Avoid platform-specific linker flags unless explicitly gated behind `#[cfg(target_os = "...")]`.

---

## 2. Current Implementation & Status
- Implemented as a clean, minimal build script.
- All recursive search algorithms (DF-PN and MCTS) use heap-allocated paths or iterative queue traversals (`path: Vec<PackedBoardState>`, `VecDeque`), rendering artificial stack expansions unnecessary under standard Linux 8 MB stack configurations.
