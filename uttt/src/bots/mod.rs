use crate::game_solving::GameMove;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use std::time::Duration;

pub mod random_bot;
pub mod rule_based_bot;
pub mod alphabeta_bot;
pub mod mcts_bot;
pub mod heuristic_mcts_bot;
pub mod original_umcts_bot;

pub use random_bot::RandomBot;
pub use rule_based_bot::RuleBasedBot;
pub use alphabeta_bot::AlphaBetaBot;
pub use mcts_bot::MctsBot;
pub use heuristic_mcts_bot::HeuristicMctsBot;
pub use original_umcts_bot::OriginalUmctsBot;

pub trait Bot: Send + Sync {
    fn name(&self) -> &str;
    fn reset(&mut self) {}
    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove;
}

pub fn available_bots() -> Vec<&'static str> {
    vec![
        "random",
        "rule_based",
        "alphabeta",
        "mcts",
        "heuristic_mcts",
        "original_umcts",
    ]
}

pub fn create_bot(name: &str) -> Option<Box<dyn Bot>> {
    match name.to_lowercase().replace('-', "_").as_str() {
        "random" => Some(Box::new(RandomBot::new())),
        "rule_based" | "rule" | "heuristic" => Some(Box::new(RuleBasedBot::new())),
        "alphabeta" | "alpha_beta" | "minimax" => Some(Box::new(AlphaBetaBot::new())),
        "mcts" | "uct" => Some(Box::new(MctsBot::new())),
        "heuristic_mcts" | "hmcts" => Some(Box::new(HeuristicMctsBot::new())),
        "original_umcts" | "umcts" | "original" => Some(Box::new(OriginalUmctsBot::new())),
        _ => None,
    }
}
