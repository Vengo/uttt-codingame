use std::time::Duration;
use crate::bots::{create_bot, Bot};
use crate::game_solving::{AlphaBetaCompatible, GameMove};
use crate::opening_book::OpeningBook;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};

pub struct BookBot {
    pub inner: Box<dyn Bot>,
    pub book: OpeningBook,
    pub name: String,
}

impl BookBot {
    pub fn new(fallback_bot_name: &str) -> Self {
        let inner = create_bot(fallback_bot_name).unwrap_or_else(|| create_bot("umcts_v2").unwrap());
        let book = OpeningBook::new();
        BookBot {
            inner,
            book,
            name: format!("book_{}", fallback_bot_name),
        }
    }
}

impl Bot for BookBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        // Step 1: Probe opening book
        if let Some(book_move) = self.book.lookup(board, zobrist_constants) {
            // Verify move is legal
            let legal = board.get_legal_moves(legal_moves_lookup);
            if legal.iter().any(|m| m.move_index == book_move.move_index) {
                return book_move;
            }
        }

        // Step 2: Fall back to underlying bot
        self.inner.choose_move(board, legal_moves_lookup, zobrist_constants, time_limit)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}
