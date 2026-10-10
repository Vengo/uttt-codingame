# Plans & Progress: `uttt/src/arena/glicko.rs` Bayesian Rating Engine

This document details the mathematical implementation, root-finding algorithms, and benchmark harness for Glicko-2 in `uttt/src/arena/glicko.rs`.

---

## 1. Architectural Plan & Mathematical Formulation

### 1.1 Data Structures
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlickoPlayer {
    pub rating: f64,        // Standard scale (default 1500.0)
    pub rd: f64,            // Rating Deviation (default 350.0)
    pub volatility: f64,    // Default 0.06
}

pub struct GlickoCalculator {
    pub tau: f64,           // System constant (default 0.5)
}
```

### 1.2 Mathematical Update Equations
1. **Scale Conversion**:
   $$\mu = \frac{r - 1500}{173.7178}, \quad \phi = \frac{RD}{173.7178}$$
2. **Estimated Variance $v$ & Intermediate $\Delta$**:
   $$g(\phi_j) = \frac{1}{\sqrt{1 + 3 \phi_j^2 / \pi^2}}$$
   $$E(\mu, \mu_j, \phi_j) = \frac{1}{1 + \exp(-g(\phi_j)(\mu - \mu_j))}$$
   $$v = \left[ \sum_{j=1}^m g(\phi_j)^2 E(1 - E) \right]^{-1}$$
   $$\Delta = v \sum_{j=1}^m g(\phi_j)(s_j - E)$$
   where $s_j \in \{1.0, 0.5, 0.0\}$.
3. **Volatility Update $\sigma'$ (Glickman's Objective Function)**:
   $$f(x) = \frac{e^x (\Delta^2 - \phi^2 - v - e^x)}{2 (\phi^2 + v + e^x)^2} - \frac{x - \ln(\sigma^2)}{\tau^2}$$
   - Solved via the Illinois algorithm (regula falsi with halved bracket values on same-side steps) initialized with $A = \ln(\sigma^2)$ and step-bracket $B = a - k\tau$, terminating when $|B - A| < 10^{-6}$.
   - Updated volatility: $\sigma' = \exp(A / 2)$.
4. **Updated Rating & Deviation**:
   $$\phi^* = \sqrt{\phi^2 + {\sigma'}^2}$$
   $$\phi' = \frac{1}{\sqrt{1 / {\phi^*}^2 + 1 / v}}$$
   $$\mu' = \mu + {\phi'}^2 \sum_{j=1}^m g(\phi_j)(s_j - E)$$
   $$r' = 173.7178 \mu' + 1500, \quad RD' = 173.7178 \phi'$$

### 1.3 Benchmark Protocol against Stationary Baseline
- To avoid the stationary equilibrium trap ($RD \approx 50.1$), the benchmark accumulates all $m$ games against baseline `AlphaBetaBot` ($\mu_{\text{base}} = 0, \phi_{\text{base}} \approx 0.01$) into a **single rating period**.
- The runner plays games in balanced pairs ($X$ and $O$ alternating), accumulating $(s_j, \mu_{\text{base}}, \phi_{\text{base}})$.
- After every pair, it evaluates the single-period Glicko-2 update for the candidate bot:
  - When $m \approx 80$–$100$ games, $RD'$ drops below $35.0$.
  - When $RD' \le 35.0$ (or safety cutoff $m \ge 500$), the run concludes.
- **Hypothesis Decision**:
  $$\text{Accepted if: } r' - 1.96 \times RD' > 1500.0$$

### 1.4 CLI Integration
- Subcommand: `arena benchmark --bot <name> --baseline alphabeta [--target-rd 35] [--time-ms 40]`.
- Strips `"benchmark"` alias from `arena profile` to ensure clean routing.

---

## 2. Status & Verification Plan
- Unit test suite verifying Glickman's published test vectors (rating 1500, RD 200, opponents 1400/1550/1700 $\to$ rating 1464.05, RD 151.52).
- Benchmark runner execution to prove $> 95\%$ confidence superiority for `alphabeta_v2` and `umcts_v3`.
