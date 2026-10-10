# Plans & Progress: `uttt/src/arena/elo.rs`

This document details Elo rating calculation algorithms and data structures in `uttt/src/arena/elo.rs`.

---

## 1. Plan & Mathematical Design

### 1.1 Rating Data Structure
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloCalculator {
    pub k_factor: f64,
    pub ratings: HashMap<String, f64>,
}
```
- Initial baseline rating: $1500.0$.
- Default $K$-factor: $K = 32.0$.

### 1.2 Mathematical Formulas
- **Expected Score**:
  $$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$
  $$E_B = 1 - E_A$$
- **Rating Update**:
  $$R_A' = R_A + K \times (S_A - E_A)$$
  $$R_B' = R_B + K \times (S_B - E_B)$$
  where $S_A \in \{1.0, 0.5, 0.0\}$ and $S_B = 1.0 - S_A$.

---

## 2. Status
- Fully implemented, serialized with `serde`, and tested via `test_elo_calculator`.
