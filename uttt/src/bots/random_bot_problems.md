# Problems & Challenges: `uttt/src/bots/random_bot.rs`

This document details baseline random move generation challenges in `uttt/src/bots/random_bot.rs`.

---

## 1. Challenges
- **Uniformity & Zero Bias**: Selecting uniformly at random from all legal moves without skewing toward low or high cell indices.
- **RNG Performance**: Using a fast pseudo-random generator (`SmallRng`) seeded deterministically or from OS entropy.
