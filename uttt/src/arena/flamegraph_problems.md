# Problems & Challenges: `uttt/src/arena/flamegraph.rs`

This document details CPU sampling and flamegraph profiling challenges in `uttt/src/arena/flamegraph.rs`.

---

## 1. Challenges
- **Linux Perf Guard Constraints**: CPU sampling using `pprof` requires Linux `perf_event_open` kernel permissions. In environments where `perf_event_paranoid` > 2 or inside restricted containers, CPU frequency sampling may fail.
- **Flamegraph Serialization**: Generating self-contained SVG graphics with embedded CSS/JS without requiring external perl scripts (`flamegraph.pl`).
- **Profiling Realistic Bot Decisions**: Sampling bots during actual search under match conditions rather than synthetic micro-benchmarks.
