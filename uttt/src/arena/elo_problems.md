# Problems & Challenges: `uttt/src/arena/elo.rs`

This document details Elo rating calculation challenges in `uttt/src/arena/elo.rs`.

---

## 1. Challenges
- **Dynamic Rating Adjustments for Draws**: Accurately reflecting 0.5-point draw allocations in expected score calculations without causing rating inflation or deflation.
- **Initial Rating Calibration**: Standardizing baseline ratings ($1500.0$) and selecting a robust $K$-factor ($K = 32.0$) suitable for short tournament series.
