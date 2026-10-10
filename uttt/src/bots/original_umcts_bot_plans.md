# Plans & Progress: `uttt/src/bots/original_umcts_bot.rs`

This document details legacy MCTS implementation and role in `uttt/src/bots/original_umcts_bot.rs`.

---

## 1. Plan & Role
- Wraps the original repository's `advance_rollout` and node store architecture into the standard `Bot` trait interface.
- Preserved verbatim to serve as the baseline for `test_profile_original_mcts_behavior`.

---

## 2. Status
- Operational and integrated into the tournament leaderboard.
