use crate::bots::Bot;
use crate::game_solving::{
    advance_rollout, reverse_rollout, AlphaBetaCompatible, GameMove, GameResult,
    IdentityHasherBuilder, UmctsNode, UmctsQueueElement,
};
use crate::uttt_trait_solving::{
    UltimateTicTacToeBoard, UltimateTicTacToeUnmove, ZobristConstants,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct OriginalUmctsBot {
    name: String,
    node_store: HashMap<u64, UmctsNode, IdentityHasherBuilder>,
    unmove_queue: Vec<UmctsQueueElement<u64, UltimateTicTacToeUnmove>>,
}

impl OriginalUmctsBot {
    pub fn new() -> Self {
        OriginalUmctsBot {
            name: "OriginalUmctsBot".to_string(),
            node_store: HashMap::with_capacity_and_hasher(1_000_000, IdentityHasherBuilder {}),
            unmove_queue: Vec::with_capacity(81),
        }
    }
}

impl Default for OriginalUmctsBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for OriginalUmctsBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.node_store.clear();
        self.unmove_queue.clear();
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        let moves = if !legal_moves_lookup.is_empty() {
            board.get_legal_moves(legal_moves_lookup)
        } else {
            board.get_legal_moves_fast()
        };

        if moves.len() <= 1 {
            return moves.first().copied().unwrap_or_default();
        }

        let start_time = Instant::now();
        let time_limit_micros = time_limit.as_micros();
        let mut mut_board = board.clone();

        while start_time.elapsed().as_micros() < time_limit_micros {
            let mut search_alpha = i8::MIN;
            let mut search_beta = i8::MAX;

            while let Some(new_alpha_beta) = advance_rollout(
                &mut self.unmove_queue,
                &mut mut_board,
                legal_moves_lookup,
                search_alpha,
                search_beta,
                &mut self.node_store,
                zobrist_constants,
            ) {
                let (new_alpha, new_beta) = new_alpha_beta;
                search_alpha = new_alpha;
                search_beta = new_beta;
            }

            let final_hash = mut_board.hash_val();
            let terminal_node = self.node_store.get_mut(&final_hash).unwrap();

            let rollout_result = match mut_board.get_win_state() {
                GameResult::P1Win => {
                    let res = i8::MAX;
                    terminal_node.value_lower = res;
                    terminal_node.value_upper = res;
                    res
                }
                GameResult::P2Win => {
                    let res = i8::MIN;
                    terminal_node.value_lower = res;
                    terminal_node.value_upper = res;
                    res
                }
                GameResult::Tie => {
                    let res = 0;
                    terminal_node.value_lower = res;
                    terminal_node.value_upper = res;
                    res
                }
                _ => search_alpha,
            };

            reverse_rollout(
                rollout_result,
                &mut self.unmove_queue,
                &mut mut_board,
                &mut self.node_store,
            );
        }

        let mut probe_board = board.clone();
        let best_move = moves
            .iter()
            .max_by_key(|a_move| {
                let undo = probe_board.make_move(**a_move, zobrist_constants);
                let this_hash = probe_board.hash_val();
                probe_board.unmake_move(undo);
                match self.node_store.get(&this_hash) {
                    Some(node) => node.visits,
                    None => 0,
                }
            })
            .copied()
            .unwrap_or(moves[0]);

        best_move
    }
}
