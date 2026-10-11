use crate::game_solving::GameMove;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use std::time::Duration;

pub mod random_bot;
pub mod rule_based_bot;
pub mod alphabeta_bot;
pub mod alphabeta_v2_bot;
pub mod mcts_bot;
pub mod heuristic_mcts_bot;
pub mod original_umcts_bot;
pub mod umcts_v2_bot;
pub mod umcts_v3_bot;
pub mod book_bot;
pub mod alphazero_bot;

pub use random_bot::RandomBot;
pub use rule_based_bot::RuleBasedBot;
pub use alphabeta_bot::AlphaBetaBot;
pub use alphabeta_v2_bot::AlphaBetaV2Bot;
pub use mcts_bot::MctsBot;
pub use heuristic_mcts_bot::HeuristicMctsBot;
pub use original_umcts_bot::OriginalUmctsBot;
pub use umcts_v2_bot::UmctsV2Bot;
pub use umcts_v3_bot::UmctsV3Bot;
pub use book_bot::BookBot;
pub use alphazero_bot::AlphaZeroBot;

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
        "alphabeta_v2",
        "mcts",
        "heuristic_mcts",
        "original_umcts",
        "umcts_v2",
        "umcts_v3",
        "book_umcts_v2",
        "book_alphabeta",
        "alphazero",
    ]
}

pub fn create_bot(name: &str) -> Option<Box<dyn Bot>> {
    match name.to_lowercase().replace('-', "_").as_str() {
        "random" => Some(Box::new(RandomBot::new())),
        "rule_based" | "rule" | "heuristic" => Some(Box::new(RuleBasedBot::new())),
        "alphabeta" | "alpha_beta" | "minimax" => Some(Box::new(AlphaBetaBot::new())),
        "alphabeta_v2" | "ab_v2" | "ab2" => Some(Box::new(AlphaBetaV2Bot::new())),
        "mcts" | "uct" => Some(Box::new(MctsBot::new())),
        "heuristic_mcts" | "hmcts" => Some(Box::new(HeuristicMctsBot::new())),
        "original_umcts" | "umcts" | "original" => Some(Box::new(OriginalUmctsBot::new())),
        "umcts_v2" | "v2" | "umcts2" => Some(Box::new(UmctsV2Bot::new())),
        "umcts_v3" | "v3" | "umcts3" => Some(Box::new(UmctsV3Bot::new())),
        "book_umcts_v2" | "book_v2" => Some(Box::new(BookBot::new("umcts_v2"))),
        "book_alphabeta" | "book_ab" => Some(Box::new(BookBot::new("alphabeta"))),
        "alphazero" | "alphazero_bot" | "az" | "nn" | "nn_bot" => Some(Box::new(AlphaZeroBot::new())),
        _ => None,
    }
}
