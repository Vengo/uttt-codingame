# Problems & Challenges: `uttt/src/bots/book_bot.rs`

This document details opening book integration and fallback search challenges in `uttt/src/bots/book_bot.rs`.

---

## 1. Challenges
- **Zero-Latency Opening Probing**: Querying the opening book in < 1 $\mu$s at ply 0, while ensuring the returned move is strictly verified against current legal moves before playing.
- **Graceful Fallback to Search**: Seamlessly delegating to an underlying search bot (`umcts_v2` or `alphabeta`) when the position is out of book.
- **Symmetry Preservation**: Ensuring canonical book moves are inverted back into actual board coordinates without orientation errors.
