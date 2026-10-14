# Plans & Progress: `uttt/src/egtb/retrograde.rs`

This document details the bottom-up retrograde dynamic programming solver and chunk streaming architecture in [`uttt/src/egtb/retrograde.rs`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/retrograde.rs).

---

## 1. Architectural Design & Plan

- **Bottom-Up Retrograde Dynamic Programming**:
  - Solves endgame tablebases backward from terminal configurations by exploiting the strictly acyclic Directed Acyclic Graph (DAG) property of Ultimate Tic-Tac-Toe: every move fills an empty cell, decrementing total remaining empty cells $E$ by 1.
  - Parallelized over subboards and pairs using Rayon.

- **Topological Isolation Solver for Tier 8 (`solve_tier_8`)**:
  - In Tier 8, exactly 8 subboards are closed.
  - Any move inside the single open subboard $k$ pointing to an external cell directs the opponent to a closed subboard. Under standard rules, this grants free choice, but because all other 8 subboards are closed, the opponent is topologically forced to play back into subboard $k$.
  - Play on subboard $k$ is therefore strictly isomorphic to an isolated single-subboard game combined with macro win conditions.
  - Generates the entire Tier 8 tablebase in $< 2$ seconds (~83 MB uncompressed).

- **Layered Backward Induction for Tier 7 (`solve_tier_7`)**:
  - Solves the 36 open subboard pairs $(j, k)$ across $3^7 = 2,187$ macro combinations.
  - Groups configurations by macro condition $\tau = (xw_j, xw_k, ow_j, ow_k)$.
  - Inducts layer by layer from $E = 1$ up to `max_empty` (e.g. 4 plies) using [`PairLayerTable`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/retrograde.rs#L991).

- **Disk-Backed Chunk Streaming for Tier 6 (`solve_tier_6_chunk` & `merge_tier_6_chunks`)**:
  - Splits the 84 subboard triplets into $N$ standalone chunks (default: 8 chunks).
  - Each chunk runs independently and persists records to `chunks/tier6_chunk_{:03}.bin` alongside a `.done` marker.
  - `merge_tier_6_chunks` combines, sorts, and deduplicates all chunks into the final `egtb_tier6.bin` tablebase file with bounded memory usage.

- **Storage Format & Record Serialization**:
  - Magic header: 8-byte ASCII `UTTTEG01`.
  - Entry count: 8-byte little-endian `u64`.
  - Records: 10 bytes each ([`RetrogradeRecord`](file:///home/festivity/agy/uttt-codingame/uttt/src/egtb/retrograde.rs#L39)), comprising `hash: u64`, `outcome: u8`, `dtm: u8`.

---

## 2. Types & Public APIs

- `RetrogradeRecord`: 10-byte packed struct (`hash: u64`, `outcome: u8`, `dtm: u8`).
- `RetrogradeSolver`: Multi-threaded DP engine.
  * `new(output_dir: &Path, num_threads: usize) -> Self`.
  * `solve_tier_8() -> Result<(usize, PathBuf), String>`.
  * `solve_tier_7(max_positions: usize) -> Result<(usize, PathBuf), String>`.
  * `solve_tier_6_chunk(chunk_idx, total_chunks, max_positions) -> Result<(usize, PathBuf), String>`.
  * `merge_tier_6_chunks(chunks_dir, out_file) -> Result<usize, String>`.
  * `persist_records(path, records) -> Result<(), String>`.
- `MicroSubboardTable`: Dense representation of all 11,125 legal non-won 3x3 configurations.
  * `build() -> Self`: Generates states, lookup indices, states partitioned by empty cells, and retrograde outcomes (`MicroOutcome`).
- `PairLayerTable`: 2D table holding layer outcomes for pair partition $(e_j, e_k)$ across 2 turns and 3 targets.
- `GameStateOutcome`: Outcome enum (`P1Win = 1`, `P2Win = 2`, `Draw = 3`).
- `MicroOutcome`: Micro outcome enum (`XWin`, `OWin`, `Tie`, `ActiveWins`, `ActiveLoses`, `Draw`).

---

## 3. Status & Verification

- Tier 8 solver verified: generates all 8-resolved canonical configurations in $< 2$ seconds.
- Tier 7 layered DP verified: solves across 36 pairs and groups by macro condition $\tau$.
- Tier 6 streaming verified: chunk execution and multi-chunk merge validated with `.done` markers for idempotent checkpointing.
