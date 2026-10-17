# Distributed Google Colab Worker for UTTT Tier 6 Tablebase

This directory provides turnkey scripts and a Google Colab notebook to distribute the computation of the **Ultimate Tic-Tac-Toe (UTTT) Tier 6 Endgame Tablebase (EGTB)** across cloud nodes.

---

## 1. Background & Architecture

- **Tier 6 Structure**: Exactly 3 subboards open, 6 subboards decided/closed.
- **Independence of Orbits**: The problem decomposes into **16 canonical board orbits** comprising **3,584 canonical macro-conditions ($\tau$)**. Every orbit and every $\tau$ slice is mathematically independent.
- **Retrograde Dependencies**: Whenever a move in a Tier 6 state resolves an open subboard, the game transitions to Tier 7 (2 open subboards). The solver queries the exhaustive Tier 7 ZFS tablebase directly in `< 20 ns` via memory-mapped slices.
- **Zero Loss / Resumable**: Every $\tau$ slice writes to `.tmp.<pid>` and atomically renames to `tier6_orbit_<O>_tau_<T>.bin`. If a job is stopped, interrupted, or restarted, existing `.bin` files are automatically skipped.

---

## 2. Canonical Orbit Catalog & Tau Counts

| Orbit | Canonical Triplet $(j, k, l)$ | Open Center (4)? | Reachable $\tau$ Count | Status / Recommended Worker |
|:---:|:---:|:---:|:---:|:---|
| **0** | `(0, 1, 2)` | No | 107 | ✅ **Completed** (Local Workstation) |
| **1** | `(0, 1, 3)` | No | 263 | ⏳ **In Progress** (Local Workstation) |
| **2** | `(0, 1, 5)` | No | 186 | Local Queue |
| **3** | `(0, 1, 6)` | No | 271 | Local Queue |
| **4** | `(0, 1, 7)` | No | 207 | Colab / Cloud |
| **5** | `(0, 1, 8)` | No | 255 | Colab / Cloud |
| **6** | `(0, 2, 6)` | No | 389 | 🚀 **Colab Worker 1 (High Priority)** |
| **7** | `(0, 2, 7)` | No | 203 | Colab / Cloud |
| **8** | `(0, 5, 7)` | No | 99 | Colab / Cloud |
| **9** | `(1, 3, 5)` | No | 159 | Colab / Cloud |
| **10** | `(0, 1, 4)` | Yes | 343 | 🚀 **Colab Worker 3 (High Priority)** |
| **11** | `(0, 2, 4)` | Yes | 397 | 🚀 **Colab Worker 2 (High Priority)** |
| **12** | `(0, 4, 5)` | Yes | 262 | Colab / Cloud |
| **13** | `(0, 4, 8)` | Yes | 99 | Colab / Cloud |
| **14** | `(1, 3, 4)` | Yes | 266 | Colab / Cloud |
| **15** | `(1, 4, 7)` | Yes | 78 | Colab / Cloud |
| **Total** | | | **3,584** | |

---

## 3. Getting Started on Google Colab

### Prerequisites
1. Upload the compressed Tier 7 package (`tier7_egtb.tar.zst`, 1.1 GB, compressed 15.5x from 17 GB) to your Google Drive under:
   `MyDrive/uttt_egtb/tier7_egtb.tar.zst`
2. Open `colab/tier6_worker.ipynb` in [Google Colab](https://colab.research.google.com).

### Worker Execution Flow
1. **Mount Google Drive**: The notebook mounts `/content/drive`.
2. **Fast Local SSD Extraction**: `tier7_egtb.tar.zst` is unpacked into `/content/tier7` on the Colab instance's fast local NVMe SSD.
   > **Note**: Tier 7 lookups perform millions of random reads. Running directly off Google Drive FUSE would introduce 5–50 ms latency per probe. Unpacking onto local SSD provides instant `< 20 ns` memory-mapped lookups.
3. **Build Binary**: Installs Rust and builds `./target/release/solve_egtb`.
4. **Solve Orbit**: Runs `solve_egtb` with:
   ```bash
   ./target/release/solve_egtb \
       --tier 6-zfs \
       --orbit <ORBIT_INDEX> \
       --threads $(nproc) \
       --tier7-dir /content/tier7 \
       --out-dir /content/drive/MyDrive/uttt_egtb/tier6 \
       --max-empty 4
   ```
5. **Persistence**: Finished `tier6_orbit_<O>_tau_<T>.bin` slices are saved directly into your Google Drive at `MyDrive/uttt_egtb/tier6/`. Even if the Colab instance terminates unexpectedly, all completed slices are preserved.

### Merging Completed Slices Locally
Once an orbit completes on Colab, copy the `.bin` slices from Google Drive into your local tablebase directory:
```bash
rsync -avP /path/to/GoogleDrive/uttt_egtb/tier6/*.bin /my-zraid/uttt/egtb/tier6/
```
The local engine and retrograde probers will automatically discover and load the newly merged slices.
