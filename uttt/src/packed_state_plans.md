# Plans & Progress: `uttt/src/packed_state.rs`

This document details the exact 167-bit bit-packed state implementation in `uttt/src/packed_state.rs`.

---

## 1. Architectural Design & Plan
- **`PackedBoardState` Structure**:
  ```rust
  #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
  pub struct PackedBoardState {
      pub words: [u64; 3], // exactly 24 bytes, 8-byte aligned
  }
  ```
- **Bit Allocation & Coordinate Ordering**:
  - Micro cells are ordered by subboard first ($s \times 9 + c$, where $s \in 0..9$ and $c \in 0..9$).
  - `words[0]` (64 bits): bits $0 \dots 63$ of 81-bit X mask.
  - `words[1]` (64 bits): remaining 17 bits of X mask (bits $0 \dots 16$) + 47 bits of O mask (bits $17 \dots 63$).
  - `words[2]` (64 bits): remaining 34 bits of O mask (bits $0 \dots 33$) + 4 bits `last_super_index` (bits $34 \dots 37$) + 1 bit `x_to_move` (bit 38).
  - Bits $39 \dots 63$ of `words[2]` (25 padding bits) are strictly zero-masked.
- **SplitMix64 Hash Mixer Implementation**:
  ```rust
  pub fn fast_hash(&self) -> u64 {
      let mut h = self.words[0].wrapping_mul(0x9E37_79B9_7F4A_7C15);
      h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
      h ^= self.words[1].wrapping_mul(0x94D0_49BB_1331_11EB);
      h = (h ^ (h >> 27)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
      h ^= self.words[2].wrapping_mul(0x9E37_79B9_7F4A_7C15);
      h ^ (h >> 31)
  }
  ```
- **Dynamic Unpacking (`to_board()`)**:
  Recomputes `x_supers` and `o_supers` on the fly using `RESULT_MASKS`, restoring full board structs with zero redundant stored bits.

---

## 2. Status & Verification
- 100% roundtrip fidelity verified across randomized games.
- Used as primary transposition key in DF-PN solver.
