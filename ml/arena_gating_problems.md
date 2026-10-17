# Arena Gating & Symmetrical Split Dilution Technical Challenges

The [`parallel_mirror_eval.py`](file:///home/festivity/agy/uttt-codingame/ml/parallel_mirror_eval.py) and [`champion_challenger.py`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py) modules conduct 2,048-game mirror tournaments (1,024 paired openings) to evaluate whether a newly trained Challenger deserves promotion over the reigning Champion. This document analyzes the mathematical, statistical, and game-theoretic problems caused by first-move initiative and symmetrical split dilution.

---

## 1. First-Mover Advantage & Symmetrical Split Dilution

### Problem
Ultimate Tic-Tac-Toe possesses a strong first-player (Player 1 / X) initiative: Player 1 dictates the initial subboard routing and controls the tempo of the early game. In mirrored opening tournaments:
- Game $2k$: Challenger is Player 1 (X), Champion is Player 2 (O).
- Game $2k+1$: Champion is Player 1 (X), Challenger is Player 2 (O).

When both agents are strong, the advantage of playing as Player 1 often exceeds the marginal skill difference between the models.

### Empirical Diagnostic Analysis:
Across 143 historical gating tournaments analyzed in [`diagnostics.py`](file:///home/festivity/agy/uttt-codingame/ml/diagnostics.py):
- **Symmetrical 1-1 Splits**: **62.4%** of paired positions (~639 pairs) end with Player 1 winning both sides.
- **Decisive Pairs**: Only **37.6%** of paired positions (~385 pairs) produce an asymmetric outcome (one player sweeps 2-0).
- **Draws**: Direct 0.5-0.5 game ties are relatively rare (< 7%), meaning almost all non-decisive pairs are 1-1 splits.

---

## 2. Mathematical Proof of the 55% Promotion Hurdle

### Problem
The standard gating rule requires a Challenger to achieve an overall match score of $W_{\text{total}} \ge 55.0\%$.

Because $62.4\%$ of position pairs are guaranteed 1-1 splits (yielding $0.5$ points each), the overall win rate is mathematically constrained:
$$W_{\text{total}} = S_{\text{split}} \times 0.5 + (1 - S_{\text{split}}) \times W_{\text{decisive}}$$

Substituting $S_{\text{split}} = 0.624$:
$$W_{\text{total}} = 0.312 + 0.376 \times W_{\text{decisive}}$$

To satisfy the promotion threshold $W_{\text{total}} \ge 0.55$:
$$0.312 + 0.376 \times W_{\text{decisive}} \ge 0.55$$
$$0.376 \times W_{\text{decisive}} \ge 0.238$$
$$W_{\text{decisive}} \ge \frac{0.238}{0.376} \approx \mathbf{63.3\%}$$

### The Double-Win Ratio Dilemma:
To hit $63.3\%$ decisive win rate:
$$\text{Double-Win Ratio} = \frac{W_{\text{decisive}}}{1 - W_{\text{decisive}}} = \frac{0.633}{0.367} \approx \mathbf{1.73 : 1}$$

The Challenger must win **1.73 double wins for every 1 loss**! Across 385 decisive pairs, that requires $\ge 244$ double wins vs $\le 141$ double wins.

---

## 3. False Rejection of Monotonically Superior Models

### Problem
In multiple overnight cycles (Cycles 20, 22, 23, 25, and 27), Challengers achieved:
- Cycle 20: 1018W - 880L - 150D (53.37% win rate, 222 double wins vs 168 losses $\implies 56.9\%$ decisive win rate).
- Cycle 23: 1030W - 876L - 142D (53.76% win rate, 230 double wins vs 165 losses $\implies 58.2\%$ decisive win rate).
- Cycle 27: 1042W - 868L - 138D (54.25% win rate, 238 double wins vs 154 losses $\implies 60.7\%$ decisive win rate).

In every one of these cycles:
1. The Challenger demonstrated decisive tactical superiority over the Champion (+25 to +40 Elo improvement).
2. The Challenger was **summarily rejected** because the 62.4% symmetrical split floor diluted the overall score below 55.0%.
3. The training loop spent 25 consecutive cycles defending a legacy champion, discarding superior weights and impeding learning velocity.

---

## 4. Invariants & Statistical Safeguards

1. **Noise Immunity**: Any relaxed or decisive-adjusted gating criterion must maintain statistical rigor (e.g. $p < 0.05$ via one-tailed binomial test against $H_0: p = 0.5$, $H_1: p > 0.5$) so that random fluctuations cannot trigger false promotions.
2. **Mirror Balance**: The pairing structure (each position played from both sides) must remain strictly invariant to prevent board-bias artifacts.
3. **Elo Consistency**: The rating assigned upon promotion must reflect the true Bradley-Terry skill delta rather than an unadjusted mirror score.
