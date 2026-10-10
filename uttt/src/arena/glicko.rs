use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

pub const GLICKO_SCALE: f64 = 173.7178;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlickoPlayer {
    pub rating: f64,
    pub rd: f64,
    pub volatility: f64,
}

impl Default for GlickoPlayer {
    fn default() -> Self {
        GlickoPlayer {
            rating: 1500.0,
            rd: 350.0,
            volatility: 0.06,
        }
    }
}

impl GlickoPlayer {
    pub fn new(rating: f64, rd: f64, volatility: f64) -> Self {
        GlickoPlayer {
            rating,
            rd,
            volatility,
        }
    }

    pub fn credible_interval_95(&self) -> (f64, f64) {
        (self.rating - 1.96 * self.rd, self.rating + 1.96 * self.rd)
    }
}

pub struct MatchOutcome {
    pub opponent_rating: f64,
    pub opponent_rd: f64,
    pub score: f64, // 1.0 = win, 0.5 = draw, 0.0 = loss
}

pub struct GlickoCalculator {
    pub tau: f64,
}

impl Default for GlickoCalculator {
    fn default() -> Self {
        GlickoCalculator { tau: 0.5 }
    }
}

impl GlickoCalculator {
    pub fn new(tau: f64) -> Self {
        GlickoCalculator { tau }
    }

    fn g(phi: f64) -> f64 {
        1.0 / (1.0 + 3.0 * phi * phi / (PI * PI)).sqrt()
    }

    fn e(mu: f64, mu_j: f64, phi_j: f64) -> f64 {
        1.0 / (1.0 + (-Self::g(phi_j) * (mu - mu_j)).exp())
    }

    /// Updates a player based on a collection of match outcomes evaluated in a single rating period.
    pub fn update(&self, player: &GlickoPlayer, matches: &[MatchOutcome]) -> GlickoPlayer {
        if matches.is_empty() {
            // No matches: only increase RD due to volatility
            let phi = player.rd / GLICKO_SCALE;
            let phi_star = (phi * phi + player.volatility * player.volatility).sqrt();
            return GlickoPlayer {
                rating: player.rating,
                rd: phi_star * GLICKO_SCALE,
                volatility: player.volatility,
            };
        }

        // Step 1: Scale conversion
        let mu = (player.rating - 1500.0) / GLICKO_SCALE;
        let phi = player.rd / GLICKO_SCALE;
        let sigma = player.volatility;

        // Step 2: Estimated variance v and delta
        let mut v_inv = 0.0;
        let mut delta_sum = 0.0;

        for m in matches {
            let mu_j = (m.opponent_rating - 1500.0) / GLICKO_SCALE;
            let phi_j = m.opponent_rd / GLICKO_SCALE;
            let g_val = Self::g(phi_j);
            let e_val = Self::e(mu, mu_j, phi_j);

            v_inv += g_val * g_val * e_val * (1.0 - e_val);
            delta_sum += g_val * (m.score - e_val);
        }

        let v = 1.0 / v_inv;
        let delta = v * delta_sum;

        // Step 3: Volatility update via Illinois root-finding on f(x)
        let a = (sigma * sigma).ln();
        let tau2 = self.tau * self.tau;
        let phi2 = phi * phi;

        let f = |x: f64| -> f64 {
            let ex = x.exp();
            let numerator = ex * (delta * delta - phi2 - v - ex);
            let denominator = 2.0 * (phi2 + v + ex) * (phi2 + v + ex);
            (numerator / denominator) - (x - a) / tau2
        };

        let mut bracket_a = a;
        let mut bracket_b = if delta * delta > phi2 + v {
            (delta * delta - phi2 - v).ln()
        } else {
            let mut k = 1.0;
            while f(a - k * self.tau) < 0.0 {
                k += 1.0;
            }
            a - k * self.tau
        };

        let mut fa = f(bracket_a);
        let mut fb = f(bracket_b);

        while (bracket_b - bracket_a).abs() > 1e-6 {
            let c = bracket_a + (bracket_a - bracket_b) * fa / (fb - fa);
            let fc = f(c);

            if fc * fb <= 0.0 {
                bracket_a = bracket_b;
                fa = fb;
            } else {
                fa /= 2.0;
            }
            bracket_b = c;
            fb = fc;
        }

        let sigma_prime = (bracket_a / 2.0).exp();

        // Step 4: Update phi to phi_star
        let phi_star = (phi2 + sigma_prime * sigma_prime).sqrt();

        // Step 5: Updated rating and deviation
        let phi_prime = 1.0 / (1.0 / (phi_star * phi_star) + 1.0 / v).sqrt();
        let mu_prime = mu + phi_prime * phi_prime * delta_sum;

        // Step 6: Convert back to Glicko scale
        GlickoPlayer {
            rating: 1500.0 + GLICKO_SCALE * mu_prime,
            rd: GLICKO_SCALE * phi_prime,
            volatility: sigma_prime,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glickman_canonical_vector() {
        // Player: rating = 1500, rd = 200, volatility = 0.06
        let player = GlickoPlayer::new(1500.0, 200.0, 0.06);
        let calc = GlickoCalculator::new(0.5);

        let matches = vec![
            MatchOutcome {
                opponent_rating: 1400.0,
                opponent_rd: 30.0,
                score: 1.0,
            },
            MatchOutcome {
                opponent_rating: 1550.0,
                opponent_rd: 100.0,
                score: 0.0,
            },
            MatchOutcome {
                opponent_rating: 1700.0,
                opponent_rd: 300.0,
                score: 0.0,
            },
        ];

        let updated = calc.update(&player, &matches);

        // Expected results from Glickman's published example:
        // Rating: ~1464.06 (or 1464.05)
        // RD: ~151.52
        // Volatility: ~0.05999
        assert!((updated.rating - 1464.06).abs() < 0.1, "Rating was {}", updated.rating);
        assert!((updated.rd - 151.52).abs() < 0.1, "RD was {}", updated.rd);
        assert!((updated.volatility - 0.05999).abs() < 0.001, "Volatility was {}", updated.volatility);
    }
}
