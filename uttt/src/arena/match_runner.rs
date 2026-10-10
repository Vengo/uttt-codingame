use crate::arena::elo::EloCalculator;
use crate::arena::game::{Game, GameOutcome, GameStats};
use crate::bots::Bot;
use crate::game_solving::GameMove;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub bot1_name: String,
    pub bot2_name: String,
    pub total_games: usize,
    pub bot1_wins: usize,
    pub bot2_wins: usize,
    pub draws: usize,
    pub bot1_winrate: f64,
    pub bot2_winrate: f64,
    pub game_history: Vec<GameStats>,
}

pub struct MatchRunner {
    pub time_limit_per_move: Duration,
    pub record_replays: bool,
    pub openings: Vec<UltimateTicTacToeBoard>,
}

impl MatchRunner {
    pub fn new(time_limit_per_move: Duration) -> Self {
        MatchRunner {
            time_limit_per_move,
            record_replays: false,
            openings: Vec::new(),
        }
    }

    pub fn with_replays(mut self, record: bool) -> Self {
        self.record_replays = record;
        self
    }

    pub fn with_openings(mut self, openings: Vec<UltimateTicTacToeBoard>) -> Self {
        self.openings = openings;
        self
    }

    pub fn run_match(
        &self,
        bot1: &mut dyn Bot,
        bot2: &mut dyn Bot,
        num_games: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        elo: Option<&mut EloCalculator>,
    ) -> MatchResult {
        let game_runner = Game::new(self.time_limit_per_move).with_replay(self.record_replays);
        let mut bot1_wins = 0;
        let mut bot2_wins = 0;
        let mut draws = 0;
        let mut history = Vec::with_capacity(num_games);

        let bot1_name = bot1.name().to_string();
        let bot2_name = bot2.name().to_string();

        let mut elo_calc = elo;
        let default_board = UltimateTicTacToeBoard::default();

        for game_idx in 0..num_games {
            let initial_board = if !self.openings.is_empty() {
                &self.openings[(game_idx / 2) % self.openings.len()]
            } else {
                &default_board
            };

            let (stats, _replay) = if game_idx % 2 == 0 {
                // Bot 1 plays X, Bot 2 plays O
                let (s, r) = game_runner.play_from_position(bot1, bot2, initial_board, legal_moves_lookup, zobrist_constants);
                match &s.outcome {
                    GameOutcome::XWin | GameOutcome::OForfeit(_) => {
                        bot1_wins += 1;
                        if let Some(ec) = elo_calc.as_deref_mut() {
                            ec.update(&bot1_name, &bot2_name, 1.0);
                        }
                    }
                    GameOutcome::OWin | GameOutcome::XForfeit(_) => {
                        bot2_wins += 1;
                        if let Some(ec) = elo_calc.as_deref_mut() {
                            ec.update(&bot1_name, &bot2_name, 0.0);
                        }
                    }
                    GameOutcome::Draw => {
                        draws += 1;
                        if let Some(ec) = elo_calc.as_deref_mut() {
                            ec.update(&bot1_name, &bot2_name, 0.5);
                        }
                    }
                }
                (s, r)
            } else {
                // Bot 2 plays X, Bot 1 plays O
                let (s, r) = game_runner.play_from_position(bot2, bot1, initial_board, legal_moves_lookup, zobrist_constants);
                match &s.outcome {
                    GameOutcome::XWin | GameOutcome::OForfeit(_) => {
                        bot2_wins += 1;
                        if let Some(ec) = elo_calc.as_deref_mut() {
                            ec.update(&bot2_name, &bot1_name, 1.0);
                        }
                    }
                    GameOutcome::OWin | GameOutcome::XForfeit(_) => {
                        bot1_wins += 1;
                        if let Some(ec) = elo_calc.as_deref_mut() {
                            ec.update(&bot2_name, &bot1_name, 0.0);
                        }
                    }
                    GameOutcome::Draw => {
                        draws += 1;
                        if let Some(ec) = elo_calc.as_deref_mut() {
                            ec.update(&bot2_name, &bot1_name, 0.5);
                        }
                    }
                }
                (s, r)
            };

            history.push(stats);
        }

        let bot1_winrate = if num_games > 0 {
            (bot1_wins as f64 + 0.5 * draws as f64) / (num_games as f64) * 100.0
        } else {
            0.0
        };

        let bot2_winrate = if num_games > 0 {
            (bot2_wins as f64 + 0.5 * draws as f64) / (num_games as f64) * 100.0
        } else {
            0.0
        };

        MatchResult {
            bot1_name,
            bot2_name,
            total_games: num_games,
            bot1_wins,
            bot2_wins,
            draws,
            bot1_winrate,
            bot2_winrate,
            game_history: history,
        }
    }
}
