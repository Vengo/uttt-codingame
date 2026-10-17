use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::uttt_trait_solving::{
    parse_index, UltimateTicTacToeBoard, ZobristConstants,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
const SUBBOARD_VALUES: [i32; 9] = [200, 150, 200, 150, 300, 150, 200, 150, 200];
const CELL_VALUES: [i32; 9] = [8, 4, 8, 4, 15, 4, 8, 4, 8];

pub const MATE_SCORE: i32 = 100_000;
pub const EGTB_WIN_SCORE: i32 = 95_000;
pub const MATE_THRESHOLD: i32 = 90_000;

#[derive(Clone, Copy)]
enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy)]
struct TTEntry {
    depth: u8,
    score: i32,
    flag: TTFlag,
    best_move: Option<GameMove>,
}

pub struct AlphaBetaBot {
    name: String,
    tt: HashMap<u64, TTEntry>,
}

impl AlphaBetaBot {
    pub fn new() -> Self {
        AlphaBetaBot {
            name: "AlphaBetaBot".to_string(),
            tt: HashMap::with_capacity(1_000_000),
        }
    }

    #[inline(always)]
    pub fn full_hash(board: &UltimateTicTacToeBoard) -> u64 {
        let target_idx = (board.last_super_index as usize).min(9) as u64;
        board.hash_val() ^ (target_idx + 1).wrapping_mul(0x9E3779B97F4A7C15).rotate_left(17)
    }

    #[inline(always)]
    fn score_to_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE_THRESHOLD {
            score + ply as i32
        } else if score <= -MATE_THRESHOLD {
            score - ply as i32
        } else {
            score
        }
    }

    #[inline(always)]
    fn score_from_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE_THRESHOLD {
            score - ply as i32
        } else if score <= -MATE_THRESHOLD {
            score + ply as i32
        } else {
            score
        }
    }

    fn evaluate_board(board: &UltimateTicTacToeBoard) -> i32 {
        match board.get_win_state() {
            GameResult::P1Win => return MATE_SCORE,
            GameResult::P2Win => return -MATE_SCORE,
            GameResult::Tie => return 0,
            GameResult::Unresolved => {}
        }

        let mut score = 0i32;

        // 1. Superboard subboard ownership
        for sub in 0..9 {
            let mask = 1u16 << sub;
            if (board.x_supers & mask) != 0 {
                score += SUBBOARD_VALUES[sub];
            } else if (board.o_supers & mask) != 0 {
                score -= SUBBOARD_VALUES[sub];
            }
        }

        // 2. Superboard potential lines (threats to win game)
        for mask in RESULT_MASKS.iter() {
            let x_count = (board.x_supers & mask).count_ones();
            let o_count = (board.o_supers & mask).count_ones();
            if x_count > 0 && o_count == 0 {
                score += match x_count {
                    1 => 40,
                    2 => 500,
                    _ => 10_000,
                };
            } else if o_count > 0 && x_count == 0 {
                score -= match o_count {
                    1 => 40,
                    2 => 500,
                    _ => 10_000,
                };
            }
        }

        // 3. Local subboard threats and positional control (for unfinished subboards)
        let complete = board.x_supers | board.o_supers | board.get_tied_mask();
        for sub in 0..9 {
            if (complete & (1 << sub)) != 0 {
                continue;
            }
            let xbb = board.xbb[sub];
            let obb = board.obb[sub];

            // Center and corner presence in local board
            for sq in 0..9 {
                let bit = 1u16 << sq;
                if (xbb & bit) != 0 {
                    score += CELL_VALUES[sq];
                } else if (obb & bit) != 0 {
                    score -= CELL_VALUES[sq];
                }
            }

            // Local 2-in-a-row threats
            for mask in RESULT_MASKS.iter() {
                let x_m = xbb & mask;
                let o_m = obb & mask;
                if x_m.count_ones() == 2 && o_m == 0 {
                    score += 40;
                }
                if o_m.count_ones() == 2 && x_m == 0 {
                    score -= 40;
                }
            }
        }

        score
    }

    fn order_moves(
        moves: &mut [GameMove],
        tt_best: Option<GameMove>,
        board: &UltimateTicTacToeBoard,
    ) {
        moves.sort_by_cached_key(|m| {
            if Some(*m) == tt_best {
                return i32::MIN; // Search TT move first
            }
            let parsed = parse_index(m.move_index);
            let sub = parsed.sub_board_index;
            let sq = parsed.sub_square_index;

            // Prioritize center square and center subboard
            let mut priority = 0i32;
            if sq == 4 {
                priority -= 20;
            }
            if sub == 4 {
                priority -= 15;
            }

            // Penalize moves directing opponent to completed board (free move)
            let complete = board.x_supers | board.o_supers | board.get_tied_mask();
            if (complete & (1 << sq)) != 0 {
                priority += 30; // later
            }

            priority
        });
    }

    fn negamax(
        &mut self,
        board: &mut UltimateTicTacToeBoard,
        depth: u8,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        zobrist_constants: &ZobristConstants,
        deadline: Instant,
        time_exceeded: &mut bool,
    ) -> i32 {
        if *time_exceeded || Instant::now() >= deadline {
            *time_exceeded = true;
            return 0;
        }

        let hash = Self::full_hash(board);
        let original_alpha = alpha;

        if let Some(entry) = self.tt.get(&hash) {
            if entry.depth >= depth {
                let tt_score = Self::score_from_tt(entry.score, ply);
                match entry.flag {
                    TTFlag::Exact => return tt_score,
                    TTFlag::LowerBound => alpha = alpha.max(tt_score),
                    TTFlag::UpperBound => beta = beta.min(tt_score),
                }
                if alpha >= beta {
                    return tt_score;
                }
            }
        }

        let win_state = board.get_win_state();
        if win_state != GameResult::Unresolved {
            return match win_state {
                GameResult::P1Win => {
                    if board.x_to_move {
                        MATE_SCORE - ply as i32
                    } else {
                        -MATE_SCORE + ply as i32
                    }
                }
                GameResult::P2Win => {
                    if !board.x_to_move {
                        MATE_SCORE - ply as i32
                    } else {
                        -MATE_SCORE + ply as i32
                    }
                }
                GameResult::Tie => 0,
                GameResult::Unresolved => unreachable!(),
            };
        }

        if depth == 0 {
            let static_eval = Self::evaluate_board(board);
            return if board.x_to_move { static_eval } else { -static_eval };
        }

        let mut moves = board.get_legal_moves_fast();
        if moves.is_empty() {
            let static_eval = Self::evaluate_board(board);
            return if board.x_to_move { static_eval } else { -static_eval };
        }

        let tt_best = self.tt.get(&hash).and_then(|e| e.best_move);
        Self::order_moves(&mut moves, tt_best, board);

        let mut best_score = i32::MIN + 1;
        let mut best_move = moves[0];

        for m in moves {
            let unmove = board.make_move(m, zobrist_constants);
            let score = -self.negamax(
                board,
                depth - 1,
                ply + 1,
                -beta,
                -alpha,
                zobrist_constants,
                deadline,
                time_exceeded,
            );
            board.unmake_move(unmove);

            if *time_exceeded {
                return 0;
            }

            if score > best_score {
                best_score = score;
                best_move = m;
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                break; // Alpha-beta cutoff
            }
        }

        let flag = if best_score <= original_alpha {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };

        self.tt.insert(
            hash,
            TTEntry {
                depth,
                score: Self::score_to_tt(best_score, ply),
                flag,
                best_move: Some(best_move),
            },
        );

        best_score
    }
}

impl Default for AlphaBetaBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for AlphaBetaBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.tt.clear();
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        _legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        let moves = board.get_legal_moves_fast();
        if moves.len() <= 1 {
            return moves.first().copied().unwrap_or_default();
        }

        let deadline = Instant::now() + time_limit;
        let mut mut_board = board.clone();
        let mut best_move = moves[0];
        let hash = Self::full_hash(board);

        // Iterative deepening from depth 1 up to depth 12
        for current_depth in 1..=12 {
            let mut time_exceeded = false;
            let score = self.negamax(
                &mut mut_board,
                current_depth,
                0,
                -150_000,
                150_000,
                zobrist_constants,
                deadline,
                &mut time_exceeded,
            );

            if time_exceeded {
                break;
            }

            if let Some(entry) = self.tt.get(&hash) {
                if let Some(m) = entry.best_move {
                    best_move = m;
                }
            }

            if score >= MATE_SCORE - 100 {
                break;
            }
        }

        best_move
    }
}
