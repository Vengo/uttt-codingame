use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::uttt_trait_solving::{
    parse_index, UltimateTicTacToeBoard, ZobristConstants,
};
use std::time::Duration;

const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
const SUBBOARD_WEIGHTS: [i32; 9] = [
    75, 50, 75,
    50, 100, 50,
    75, 50, 75,
];
const SQUARE_WEIGHTS: [i32; 9] = [
    8, 4, 8,
    4, 15, 4,
    8, 4, 8,
];

pub struct RuleBasedBot {
    name: String,
}

impl RuleBasedBot {
    pub fn new() -> Self {
        RuleBasedBot {
            name: "RuleBasedBot".to_string(),
        }
    }

    fn evaluate_move(
        &self,
        board: &mut UltimateTicTacToeBoard,
        m: GameMove,
        zobrist_constants: &ZobristConstants,
    ) -> i32 {
        let is_x = m.x_to_move;
        let parsed = parse_index(m.move_index);
        let subboard = parsed.sub_board_index;
        let subsquare = parsed.sub_square_index;

        let prev_x_supers = board.x_supers;
        let prev_o_supers = board.o_supers;
        let unmove = board.make_move(m, zobrist_constants);

        // 1. Immediate full-game win: highest priority
        match board.get_win_state() {
            GameResult::P1Win if is_x => {
                board.unmake_move(unmove);
                return 1_000_000;
            }
            GameResult::P2Win if !is_x => {
                board.unmake_move(unmove);
                return 1_000_000;
            }
            _ => {}
        }

        let mut score = 0i32;

        // 2. Subboard win bonus
        let won_subboard = if is_x {
            (board.x_supers & (1 << subboard)) != 0 && (prev_x_supers & (1 << subboard)) == 0
        } else {
            (board.o_supers & (1 << subboard)) != 0 && (prev_o_supers & (1 << subboard)) == 0
        };

        if won_subboard {
            score += SUBBOARD_WEIGHTS[subboard] * 10;
        }

        // 3. Positional square placement bonus
        score += SQUARE_WEIGHTS[subsquare];

        // 4. Analysis of where we send the opponent
        let target_subboard = subsquare;
        let complete_bbs = board.x_supers | board.o_supers | board.get_tied_mask();
        let target_is_complete = (complete_bbs & (1 << target_subboard)) != 0;

        if target_is_complete {
            // Sending opponent to a completed board gives them a FREE MOVE anywhere!
            score -= 60;
        } else {
            // Check if opponent can immediately win the target subboard
            let opp_bb = if is_x { board.obb[target_subboard] } else { board.xbb[target_subboard] };
            let my_bb = if is_x { board.xbb[target_subboard] } else { board.obb[target_subboard] };
            let occupied = opp_bb | my_bb;

            let mut opp_can_win = false;
            for mask in RESULT_MASKS.iter() {
                // If opponent has 2 out of 3 and the 3rd is open
                let opp_matching = opp_bb & mask;
                if opp_matching.count_ones() == 2 {
                    let missing = mask ^ opp_matching;
                    if (occupied & missing) == 0 {
                        opp_can_win = true;
                        break;
                    }
                }
            }

            if opp_can_win {
                // Dangerous: opponent can claim this board next turn
                score -= 80;
            } else {
                score += 15;
            }
        }

        // 5. Check if we blocked opponent's 2-in-a-row in the current subboard
        let opp_bb_current = if is_x { board.obb[subboard] } else { board.xbb[subboard] };
        let sq_mask = 1u16 << subsquare;
        for mask in RESULT_MASKS.iter() {
            if (mask & sq_mask) != 0 && (opp_bb_current & mask).count_ones() == 2 {
                score += 45; // blocked opponent threat
            }
        }

        // 6. Check if we created 2-in-a-row for ourselves
        let my_bb_current = if is_x { board.xbb[subboard] } else { board.obb[subboard] };
        let occupied_current = my_bb_current | opp_bb_current;
        for mask in RESULT_MASKS.iter() {
            if (mask & sq_mask) != 0 {
                let my_in_mask = my_bb_current & mask;
                if my_in_mask.count_ones() == 2 && (occupied_current & mask) == my_in_mask {
                    score += 30; // created open 2-in-a-row
                }
            }
        }

        board.unmake_move(unmove);
        score
    }
}

impl Default for RuleBasedBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for RuleBasedBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        _time_limit: Duration,
    ) -> GameMove {
        let moves = if !legal_moves_lookup.is_empty() {
            board.get_legal_moves(legal_moves_lookup)
        } else {
            board.get_legal_moves_fast()
        };

        if moves.len() == 1 {
            return moves[0];
        }

        let mut mut_board = board.clone();
        let mut best_move = moves[0];
        let mut best_score = i32::MIN;

        for m in moves {
            let score = self.evaluate_move(&mut mut_board, m, zobrist_constants);
            if score > best_score {
                best_score = score;
                best_move = m;
            }
        }

        best_move
    }
}
