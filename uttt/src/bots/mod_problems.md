# Problems & Challenges: `uttt/src/bots/mod.rs`

This document details bot interface traits and dynamic factory challenges in `uttt/src/bots/mod.rs`.

---

## 1. Challenges
- **Unified Trait Abstraction**: Defining a `Bot: Send + Sync` trait with a uniform `choose_move` method taking a `Duration` time limit, supporting both stateful (MCTS node stores) and stateless (random) searchers.
- **Factory Registration**: Instantiating bots dynamically by string identifier (`create_bot`) without compile-time coupling.
