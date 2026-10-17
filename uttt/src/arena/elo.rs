use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloCalculator {
    pub k_factor: f64,
    pub ratings: HashMap<String, f64>,
}

impl EloCalculator {
    pub fn new(k_factor: f64) -> Self {
        EloCalculator {
            k_factor,
            ratings: HashMap::new(),
        }
    }

    pub fn get_rating(&self, player: &str) -> f64 {
        *self.ratings.get(player).unwrap_or(&1500.0)
    }

    pub fn set_rating(&mut self, player: &str, rating: f64) {
        self.ratings.insert(player.to_string(), rating);
    }

    pub fn expected_score(ra: f64, rb: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rb - ra) / 400.0))
    }

    pub fn update(&mut self, player_a: &str, player_b: &str, score_a: f64) -> (f64, f64) {
        let ra = self.get_rating(player_a);
        let rb = self.get_rating(player_b);

        let ea = Self::expected_score(ra, rb);
        let eb = 1.0 - ea;

        let score_b = 1.0 - score_a;

        let new_ra = ra + self.k_factor * (score_a - ea);
        let new_rb = rb + self.k_factor * (score_b - eb);

        self.ratings.insert(player_a.to_string(), new_ra);
        self.ratings.insert(player_b.to_string(), new_rb);

        (new_ra, new_rb)
    }
}

impl Default for EloCalculator {
    fn default() -> Self {
        Self::new(32.0)
    }
}
