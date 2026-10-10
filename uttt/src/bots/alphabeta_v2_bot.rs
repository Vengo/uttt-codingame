use crate::bots::Bot;
use crate::egtb::EgtbDatabase;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::opening_book::OpeningBook;
use crate::uttt_trait_solving::{
    parse_index, UltimateTicTacToeBoard, ZobristConstants,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
const SUBBOARD_VALUES: [i32; 9] = [200, 150, 200, 150, 300, 150, 200, 150, 200];
const CELL_VALUES: [i32; 9] = [8, 4, 8, 4, 15, 4, 8, 4, 8];

pub const MAX_PLY: usize = 81;

#[derive(Clone, Copy, PartialEq, Eq)]
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

pub struct AlphaBetaV2Bot {
    name: String,
    tt: HashMap<u64, TTEntry>,
    killers: [[Option<GameMove>; 2]; MAX_PLY],
    book: OpeningBook,
    egtb: Option<EgtbDatabase>,
}

impl AlphaBetaV2Bot {
    pub fn new() -> Self {
        let egtb = Some(EgtbDatabase::new());
        AlphaBetaV2Bot {
            name: "AlphaBetaV2Bot".to_string(),
            tt: HashMap::with_capacity(1_000_000),
            killers: [[None; 2]; MAX_PLY],
            book: OpeningBook::new(),
            egtb,
        }
    }

    fn store_tt(
        &mut self,
        hash: u64,
        depth: u8,
        score: i32,
        flag: TTFlag,
        best_move: Option<GameMove>,
    ) {
        if let Some(existing) = self.tt.get_mut(&hash) {
            if depth >= existing.depth {
                let preserved_best = best_move.or(existing.best_move);
                *existing = TTEntry {
                    depth,
                    score,
                    flag,
                    best_move: preserved_best,
                };
            }
        } else {
            if self.tt.len() >= 1_000_000 {
                self.tt.clear();
            }
            self.tt.insert(
                hash,
                TTEntry {
                    depth,
                    score,
                    flag,
                    best_move,
                },
            );
        }
    }

    fn evaluate_board(board: &UltimateTicTacToeBoard) -> i32 {
        match board.get_win_state() {
            GameResult::P1Win => return 100_000,
            GameResult::P2Win => return -100_000,
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

        // 3. Local subboard threats and positional control
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
                } else if o_m.count_ones() == 2 && x_m == 0 {
                    score -= 40;
                }
            }
        }

        score
    }

    fn order_moves(
        moves: &mut [GameMove],
        tt_best: Option<GameMove>,
        killer1: Option<GameMove>,
        killer2: Option<GameMove>,
        board: &UltimateTicTacToeBoard,
    ) {
        let complete = board.x_supers | board.o_supers | board.get_tied_mask();

        moves.sort_by_cached_key(|&m| {
            // 1. TT Best move has absolute top priority
            if let Some(best) = tt_best {
                if m.move_index == best.move_index {
                    return -1000;
                }
            }

            let parsed = parse_index(m.move_index);
            let sub = parsed.sub_board_index;
            let sq = parsed.sub_square_index;

            let mut priority = 0i32;

            // 2. Killer move bonus
            if let Some(k1) = killer1 {
                if m.move_index == k1.move_index {
                    priority -= 25;
                }
            } else if let Some(k2) = killer2 {
                if m.move_index == k2.move_index {
                    priority -= 15;
                }
            }

            // 3. Positional square preferences
            if sq == 4 {
                priority -= 20; // Center square
            } else if matches!(sq, 0 | 2 | 6 | 8) {
                priority -= 10; // Corners
            }

            // 4. Center subboard bonus
            if sub == 4 {
                priority -= 15;
            }

            // 5. Penalty for sending opponent to completed board (free choice)
            if (complete & (1 << sq)) != 0 {
                priority += 30;
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

        let hash = board.hash_val();
        let original_alpha = alpha;

        // 1. Transposition table lookup
        if let Some(entry) = self.tt.get(&hash) {
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => alpha = alpha.max(entry.score),
                    TTFlag::UpperBound => beta = beta.min(entry.score),
                }
                if alpha >= beta {
                    return entry.score;
                }
            }
        }

        // 2. Terminal game state evaluation
        let win_state = board.get_win_state();
        if win_state != GameResult::Unresolved {
            let static_eval = Self::evaluate_board(board);
            return if board.x_to_move { static_eval } else { -static_eval };
        }

        // 3. In-tree Endgame Tablebase (EGTB) Probing (for ply > 0 child evaluations)
        let resolved_count = (board.x_supers | board.o_supers | board.get_tied_mask()).count_ones();
        if ply > 0 && resolved_count >= 7 {
            if let Some(ref egtb) = self.egtb {
                if let Some(res) = egtb.probe(board, zobrist_constants) {
                    let score = match res {
                        GameResult::P1Win => {
                            if board.x_to_move { 100_000 } else { -100_000 }
                        }
                        GameResult::P2Win => {
                            if !board.x_to_move { 100_000 } else { -100_000 }
                        }
                        GameResult::Tie => 0,
                        GameResult::Unresolved => unreachable!(),
                    };
                    return score;
                }
            }
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
        let k1 = if ply < MAX_PLY { self.killers[ply][0] } else { None };
        let k2 = if ply < MAX_PLY { self.killers[ply][1] } else { None };

        Self::order_moves(&mut moves, tt_best, k1, k2, board);

        let mut best_score = -200_000;
        let mut best_move = moves[0];

        for m in moves {
            // Check if move is tactical (captures a subboard) for killer tracking
            let parsed = parse_index(m.move_index);
            let my_bb = if m.x_to_move { board.xbb[parsed.sub_board_index] } else { board.obb[parsed.sub_board_index] };
            let is_tactical = RESULT_MASKS.iter().any(|mask| ((my_bb | (1 << parsed.sub_square_index)) & mask) == *mask);

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
                // Beta cutoff: record killer move if non-tactical
                if ply < MAX_PLY && !is_tactical {
                    if self.killers[ply][0] != Some(m) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(m);
                    }
                }
                break;
            }
        }

        let flag = if best_score <= original_alpha {
            TTFlag::UpperBound
        } else if best_score >= beta {
            TTFlag::LowerBound
        } else {
            TTFlag::Exact
        };

        self.store_tt(hash, depth, best_score, flag, Some(best_move));

        best_score
    }
}

impl Default for AlphaBetaV2Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for AlphaBetaV2Bot {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.tt.clear();
        self.killers = [[None; 2]; MAX_PLY];
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        _legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        let legal_moves = board.get_legal_moves_fast();
        if legal_moves.is_empty() {
            panic!("No legal moves available!");
        }
        if legal_moves.len() == 1 {
            return legal_moves[0];
        }

        // 1. Check Opening Book at root (only trusted entries with >= 5,000 visits)
        if let Some(book_move) = self.book.lookup_with_min_visits(board, zobrist_constants, 5_000) {
            if legal_moves.iter().any(|m| m.move_index == book_move.move_index) {
                return book_move;
            }
        }

        // 2. Reset killer table for the new search
        self.killers = [[None; 2]; MAX_PLY];

        // 3. Iterative deepening minimax
        let start = Instant::now();
        let deadline = start + time_limit;
        let mut sim_board = board.clone();
        let mut best_move = legal_moves[0];

        let mut depth = 1u8;
        while depth <= 20 {
            if Instant::now() >= deadline {
                break;
            }

            let mut time_exceeded = false;
            let _score = self.negamax(
                &mut sim_board,
                depth,
                0, // Root ply = 0
                -150_000,
                150_000,
                zobrist_constants,
                deadline,
                &mut time_exceeded,
            );

            if time_exceeded {
                break;
            }

            if let Some(entry) = self.tt.get(&board.hash_val()) {
                if let Some(bm) = entry.best_move {
                    best_move = bm;
                }
            }

            depth += 1;
        }

        best_move
    }
}
