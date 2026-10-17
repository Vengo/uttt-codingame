use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove};
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use rand::seq::SliceRandom;
use std::time::Duration;

pub struct RandomBot {
    name: String,
}

impl RandomBot {
    pub fn new() -> Self {
        RandomBot {
            name: "RandomBot".to_string(),
        }
    }
}

impl Default for RandomBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for RandomBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        _zobrist_constants: &ZobristConstants,
        _time_limit: Duration,
    ) -> GameMove {
        let moves = if !legal_moves_lookup.is_empty() {
            board.get_legal_moves(legal_moves_lookup)
        } else {
            board.get_legal_moves_fast()
        };

        let mut rng = rand::thread_rng();
        *moves.choose(&mut rng).expect("Legal moves should not be empty")
    }
}
