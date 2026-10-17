# Problems & Challenges: `uttt/src/arena/glicko.rs` Bayesian Rating Engine

This document details the statistical challenges of quantifying bot skill, rating uncertainty, and experiment convergence in `uttt/src/arena/glicko.rs`.

---

## 1. Challenges

### 1.1 Rating Uncertainty in Standard Elo
Standard Elo maintains a single point estimate ($R$) without variance or uncertainty intervals. A bot could score 6/10 in a small 10-game tournament and gain $+70$ Elo purely by variance, leading to false positives when evaluating experimental bot modifications.
- **Challenge**: Quantify the statistical uncertainty of each bot's rating so experiments are neither prematurely accepted nor abandoned without statistical significance.

### 1.2 The Mathematical Convergence Trap (Rating Period Sizing)
In Glicko-2, prior uncertainty is expanded at the start of every rating period by adding volatility variance:
$$\phi^* = \sqrt{\phi^2 + {\sigma'}^2}$$
On the Glicko-2 scale ($173.7178$), default volatility $\sigma = 0.06$ adds ${\sigma'}^2 \approx 0.0036$ to $\phi^2$ every rating period.
- **The Equilibrium Trap**: If rating updates occur per game or per 2-game pair, the variance gathered per pair ($v \approx 2.0$) reaches equilibrium with volatility inflation at:
  $$0.5 = \frac{1}{\phi^2} - \frac{1}{\phi^2 + 0.0036} \implies \phi \approx 0.288 \implies RD \approx 50.1$$
  Under 2-game rating periods, **$RD$ mathematically asymptotically stalls at $\approx 50.1$ and can never reach $RD \le 35$!**
- **Resolution**: To evaluate an experimental bot against a stationary baseline, all match results in a benchmark session must be accumulated together into a **single comprehensive rating period** (or large evaluation batches of $\ge 50$ games). Under a single rating period of $N \approx 100$ games, information variance $v \approx 4/N \approx 0.04$, allowing $RD$ to smoothly decline from $350 \to 30$ without recurring volatility inflation.

### 1.3 Hypothesis Decision & Credible Intervals
- **Challenge**: At $RD = 35$, the margin of error is $1.96 \times 35 = \pm 68.6$ rating points. The evaluation harness must establish unambiguous gating criteria:
  - **Statistically Significant Improvement**: The 95% Bayesian credible lower bound exceeds baseline ($r' - 1.96 \times RD' > 1500.0$).
  - **Inconclusive / Neutral**: Baseline falls within the credible interval.
  - **Safety Abort**: If `games >= max_games` (e.g. 500 games) or win rate falls significantly below 40%, abort the run early to avoid wasted compute on regressed bots.

### 1.4 CLI Dispatch Disambiguation
- In `arena.rs`, the string `"benchmark"` was previously an alias for `profile` (hardware profiler).
- **Challenge**: Disambiguate CLI routing so that `arena benchmark` exclusively triggers the Bayesian baseline bot tournament suite.
