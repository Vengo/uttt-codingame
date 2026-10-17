# Plans & Progress: `uttt/src/symmetry.rs`

This document details the dihedral symmetry implementation in `uttt/src/symmetry.rs`.

---

## 1. Design & Plan
- **Dihedral Group Enum & Operations**:
  - `Rot90`: Clockwise rotation $(r, c) \mapsto (c, 2 - r)$.
  - `Rot180`: $(r, c) \mapsto (2 - r, 2 - c)$.
  - `Rot270`: $(r, c) \mapsto (2 - c, r)$.
  - `FlipHorizontal`: Reflection across vertical centerline $(r, c) \mapsto (r, 2 - c)$.
  - `FlipVertical`: Reflection across horizontal centerline $(r, c) \mapsto (2 - r, c)$.
  - `ReflectMainDiag`: $(r, c) \mapsto (c, r)$.
  - `ReflectAntiDiag`: $(r, c) \mapsto (2 - c, 2 - r)$.
- **Traversal Sequence (`ALL_SYMMETRIES`)**:
  `[Identity, Rot90, Rot180, Rot270, FlipHorizontal, FlipVertical, ReflectMainDiag, ReflectAntiDiag]`.
  `Identity` is checked first and prioritized in tie-breaking.
- **Canonical Selection (`is_less`)**:
  Compares boards lexicographically:
  ```rust
  if a.last_super_index != b.last_super_index { return a.last_super_index < b.last_super_index; }
  if a.x_supers != b.x_supers { return a.x_supers < b.x_supers; }
  if a.o_supers != b.o_supers { return a.o_supers < b.o_supers; }
  if a.xbb != b.xbb { return a.xbb < b.xbb; } // array comparison [u16; 9]
  a.obb < b.obb
  ```
- **Move Inverse Mapping**:
  `transform_move_index` decomposes move index $m \in 0..80$ into $(s = m/9, c = m\%9)$, applies `transform_3x3` to both, and recomposes as $s' \times 9 + c'$. The inverse mapping uses `sym.inverse().transform_move_index(c_move)`.

---

## 2. Status
- Verified by unit tests and integrated into all tablebase and opening book operations.
