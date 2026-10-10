# Plans & Progress: Root `Cargo.toml` Workspace Configuration

This document details workspace setup, optimization profiles, and crate member declarations in root `Cargo.toml`.

---

## 1. Plan & Workspace Configuration

### 1.1 Workspace Members & Resolver
- **Resolver**: `resolver = "2"`
- **Members**:
  - `./uttt`: Core engine crate (`lib`), CLI suite (`arena`), CodinGame submission entry point (`cg`), and local test runner (`runlocal`).
  - `./uttt-tests`: Dedicated integration test harness.

### 1.2 Performance Compilation Profiles
- **`[profile.release]`**:
  - `opt-level = 3` (full LLVM optimization)
  - `debug = true` (DWARF symbol preservation for profiling and flamegraphs)
- **`[profile.dev]`**:
  - `opt-level = 3`
  - `debug = true`
  - `debug-assertions = false` (removes assertion penalties during local dev cycles)
- **`[profile.test]`**:
  - `opt-level = 3`
  - `debug = true` (ensures integration tests execute in < 3s while preserving backtraces)

---

## 2. Status
- Operational. Powers fast compilation and uniform optimization across the entire workspace.
