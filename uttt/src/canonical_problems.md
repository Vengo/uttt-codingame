# Problems & Challenges: `uttt/src/canonical.rs`

This document details the subboard canonicalization challenges in `uttt/src/canonical.rs`.

---

## 1. Challenges
- **Distinct Subboard Winning Configurations**: Subboards won via different rows, columns, or diagonals diverge in raw representation despite being game-theoretically identical.
- **Irrelevant Opponent Marks**: Opponent marks placed before a subboard was won clutter the bitboards and prevent transposition table hits.
- **Subboard Ties & Asymmetric Micro Patterns**: Finding a canonical full 9-cell pattern that is guaranteed disjoint and contains no 3-in-a-row for either player. Because any valid 9-cell tie pattern is asymmetric under $90^\circ$ rotations, micro patterns transform when the macro board is rotated.
- **Piece Parity Decoupling**: Setting all 9 squares to X (`0x1FF`) artificially alters local cell counts, but move generation only checks empty cells `(xbb | obb) & (1 << cell) == 0`. It must be guaranteed that no engine component infers the player to move or score from cell popcounts.
- **Directed Target Normalization**: Any move pointing to an already resolved subboard must be recognized as free choice (`FREE_CHOICE_INDEX = 9`) to collapse equivalent states.
