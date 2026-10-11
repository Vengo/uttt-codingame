use crate::bots::Bot;
use crate::dfpn::DfPnSolver;
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
    pub killers: [[Option<GameMove>; 2]; MAX_PLY],
    pub history: [[u32; 81]; 81],
    pub countermoves: [Option<GameMove>; 81],
    book: OpeningBook,
    egtb: Option<EgtbDatabase>,
    dfpn: DfPnSolver,
    legal_lookup: Vec<Vec<GameMove>>,
}

impl AlphaBetaV2Bot {
    pub fn new() -> Self {
        let egtb = Some(EgtbDatabase::new());
        let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
        AlphaBetaV2Bot {
            name: "AlphaBetaV2Bot".to_string(),
            tt: HashMap::with_capacity(1_000_000),
            killers: [[None; 2]; MAX_PLY],
            history: [[0u32; 81]; 81],
            countermoves: [None; 81],
            book: OpeningBook::new(),
            egtb,
            dfpn: DfPnSolver::new(3_000),
            legal_lookup,
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

    pub fn evaluate_board(board: &UltimateTicTacToeBoard) -> i32 {
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

        // 4. Macro-Board Routing Heuristic
        let board_has_marks = board.x_supers != 0
            || board.o_supers != 0
            || board.xbb.iter().any(|&b| b != 0)
            || board.obb.iter().any(|&b| b != 0);

        if board_has_marks {
            let is_free_choice =
                board.last_super_index >= 9 || (complete & (1 << board.last_super_index)) != 0;
            if is_free_choice {
                // Opponent gifted the active player free choice across any open subboard.
                let free_choice_val = 250;
                if board.x_to_move {
                    score += free_choice_val;
                } else {
                    score -= free_choice_val;
                }
            } else {
                let c = board.last_super_index as usize;
                let (active_bb, defending_bb) = if board.x_to_move {
                    (board.xbb[c], board.obb[c])
                } else {
                    (board.obb[c], board.xbb[c])
                };

                // Check if active player has an active 2-in-a-row threat in subboard c
                let active_threat = RESULT_MASKS.iter().any(|mask| {
                    (active_bb & mask).count_ones() == 2 && (defending_bb & mask) == 0
                });

                if active_threat {
                    // Directing active player into an unblocked threat grants them an immediate subboard capture!
                    let threat_val = 350;
                    if board.x_to_move {
                        score += threat_val;
                    } else {
                        score -= threat_val;
                    }
                } else {
                    // Check if active player has NO winning lines in c (neutralized subboard)
                    let active_has_lines = RESULT_MASKS.iter().any(|mask| {
                        (defending_bb & mask) == 0
                    });
                    if !active_has_lines {
                        let dead_board_val = 150;
                        if board.x_to_move {
                            score -= dead_board_val;
                        } else {
                            score += dead_board_val;
                        }
                    } else {
                        // Check if defending player holds defensive control (threat while active has none)
                        let defending_threat = RESULT_MASKS.iter().any(|mask| {
                            (defending_bb & mask).count_ones() == 2 && (active_bb & mask) == 0
                        });
                        if defending_threat {
                            let control_val = 100;
                            if board.x_to_move {
                                score -= control_val;
                            } else {
                                score += control_val;
                            }
                        }
                    }
                }
            }
        }

        score
    }

    pub fn score_move(
        &self,
        m: GameMove,
        tt_best: Option<GameMove>,
        ply: usize,
        prev_move: Option<GameMove>,
        board: &UltimateTicTacToeBoard,
    ) -> i32 {
        let complete = board.x_supers | board.o_supers | board.get_tied_mask();
        self.score_move_internal(m, tt_best, ply, prev_move, board, complete)
    }

    fn score_move_internal(
        &self,
        m: GameMove,
        tt_best: Option<GameMove>,
        ply: usize,
        prev_move: Option<GameMove>,
        board: &UltimateTicTacToeBoard,
        complete: u16,
    ) -> i32 {
        // 1. TT Best move has absolute top priority
        if let Some(best) = tt_best {
            if m.move_index == best.move_index {
                return 10_000_000;
            }
        }

        let parsed = parse_index(m.move_index);
        let sub = parsed.sub_board_index;
        let sq = parsed.sub_square_index;

        let my_bb = if board.x_to_move { board.xbb[sub] } else { board.obb[sub] };
        let new_my_bb = my_bb | (1 << sq);
        let wins_sub = RESULT_MASKS.iter().any(|mask| (new_my_bb & mask) == *mask);

        let my_supers = if board.x_to_move { board.x_supers } else { board.o_supers };
        let wins_game = wins_sub && RESULT_MASKS.iter().any(|mask| ((my_supers | (1 << sub)) & mask) == *mask);

        if wins_game {
            return 9_000_000;
        }

        // 2. Killer moves (2 slots per ply)
        if ply < MAX_PLY {
            if let Some(k1) = self.killers[ply][0] {
                if m.move_index == k1.move_index {
                    return 8_000_000;
                }
            }
            if let Some(k2) = self.killers[ply][1] {
                if m.move_index == k2.move_index {
                    return 7_000_000;
                }
            }
        }

        // 3. Countermove from countermoves[prev_move]
        if let Some(prev) = prev_move {
            if let Some(cm) = self.countermoves[prev.move_index as usize] {
                if m.move_index == cm.move_index {
                    return 6_000_000;
                }
            }
        }

        // 4. History score
        let history_score = if let Some(prev) = prev_move {
            self.history[prev.move_index as usize][m.move_index as usize]
        } else {
            self.history.iter().map(|row| row[m.move_index as usize]).max().unwrap_or(0)
        };

        // 5. Macro routing and tactical heuristic features
        let mut heuristic = 0i32;

        if wins_sub {
            heuristic += 600;
        } else {
            let opp_bb_in_sub = if board.x_to_move { board.obb[sub] } else { board.xbb[sub] };
            let creates_threat = RESULT_MASKS.iter().any(|mask| {
                (new_my_bb & mask).count_ones() == 2 && (opp_bb_in_sub & mask) == 0
            });
            if creates_threat {
                heuristic += 60;
            }
        }

        let opp_bb_in_sub = if board.x_to_move { board.obb[sub] } else { board.xbb[sub] };
        let ties_sub = !wins_sub && (new_my_bb | opp_bb_in_sub) == 511;
        let is_completed_target = (complete & (1 << sq)) != 0 || (sq == sub && (wins_sub || ties_sub));

        if is_completed_target {
            heuristic -= 300;
        } else {
            let opp_bb_target = if board.x_to_move { board.obb[sq] } else { board.xbb[sq] };
            let my_bb_target = if sq == sub {
                new_my_bb
            } else {
                if board.x_to_move { board.xbb[sq] } else { board.obb[sq] }
            };

            let opp_has_threat = RESULT_MASKS.iter().any(|mask| {
                (opp_bb_target & mask).count_ones() == 2 && (my_bb_target & mask) == 0
            });

            if opp_has_threat {
                heuristic -= 400;
            } else {
                let opp_has_lines = RESULT_MASKS.iter().any(|mask| {
                    (my_bb_target & mask) == 0
                });
                if !opp_has_lines {
                    heuristic += 150;
                } else {
                    let my_threat = RESULT_MASKS.iter().any(|mask| {
                        (my_bb_target & mask).count_ones() == 2 && (opp_bb_target & mask) == 0
                    });
                    if my_threat {
                        heuristic += 100;
                    }
                }
            }
        }

        if sq == 4 {
            heuristic += 20;
        } else if matches!(sq, 0 | 2 | 6 | 8) {
            heuristic += 10;
        }

        if sub == 4 {
            heuristic += 15;
        }

        (history_score.min(100_000) as i32) + heuristic
    }

    pub fn order_moves(
        &self,
        moves: &mut [GameMove],
        tt_best: Option<GameMove>,
        ply: usize,
        prev_move: Option<GameMove>,
        board: &UltimateTicTacToeBoard,
    ) {
        let complete = board.x_supers | board.o_supers | board.get_tied_mask();
        moves.sort_by_cached_key(|&m| {
            std::cmp::Reverse(self.score_move_internal(m, tt_best, ply, prev_move, board, complete))
        });
    }

    pub fn decay_history(&mut self, factor: u32) {
        if factor <= 1 {
            return;
        }
        for row in self.history.iter_mut() {
            for val in row.iter_mut() {
                *val /= factor;
            }
        }
    }

    fn negamax(
        &mut self,
        board: &mut UltimateTicTacToeBoard,
        depth: u8,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        prev_move: Option<GameMove>,
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
        let resolved_count = board.get_resolved_count();
        if ply > 0 && resolved_count >= 7 {
            if let Some(ref egtb) = self.egtb {
                if let Some(res) = egtb.probe(board) {
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
            let my_supers = if board.x_to_move { board.x_supers } else { board.o_supers };
            let my_has_threat = RESULT_MASKS.iter().any(|mask| (my_supers & mask).count_ones() == 2);
            if my_has_threat {
                let moves = board.get_legal_moves_fast();
                for m in &moves {
                    let parsed = parse_index(m.move_index);
                    let sub = parsed.sub_board_index;
                    let sq = parsed.sub_square_index;
                    let my_bb = if board.x_to_move { board.xbb[sub] } else { board.obb[sub] };
                    let new_my_bb = my_bb | (1 << sq);
                    if RESULT_MASKS.iter().any(|mask| (new_my_bb & mask) == *mask) {
                        let new_supers = my_supers | (1 << sub);
                        if RESULT_MASKS.iter().any(|mask| (new_supers & mask) == *mask) {
                            return 100_000;
                        }
                    }
                }
            }
            let static_eval = Self::evaluate_board(board);
            return if board.x_to_move { static_eval } else { -static_eval };
        }

        let mut moves = board.get_legal_moves_fast();
        if moves.is_empty() {
            let static_eval = Self::evaluate_board(board);
            return if board.x_to_move { static_eval } else { -static_eval };
        }

        let tt_best = self.tt.get(&hash).and_then(|e| e.best_move);

        self.order_moves(&mut moves, tt_best, ply, prev_move, board);

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
                Some(m),
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
                // Beta cutoff
                if !is_tactical {
                    if ply < MAX_PLY {
                        if self.killers[ply][0] != Some(m) {
                            self.killers[ply][1] = self.killers[ply][0];
                            self.killers[ply][0] = Some(m);
                        }
                    }
                    if let Some(prev) = prev_move {
                        let p_idx = prev.move_index as usize;
                        let c_idx = m.move_index as usize;
                        self.countermoves[p_idx] = Some(m);
                        self.history[p_idx][c_idx] = self.history[p_idx][c_idx].saturating_add((depth as u32) * (depth as u32));

                        const MAX_HISTORY: u32 = 100_000;
                        if self.history[p_idx][c_idx] > MAX_HISTORY {
                            self.decay_history(2);
                        }
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
        self.history = [[0u32; 81]; 81];
        self.countermoves = [None; 81];
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

        // 2. Query DF-PN Solver at root in late game (>= 5 resolved subboards)
        let resolved = (board.x_supers | board.o_supers | board.get_tied_mask()).count_ones();
        if resolved >= 5 {
            self.dfpn.max_nodes = 3_000;
            if let Some((true, Some(winning_move))) = self.dfpn.solve_move(board, &self.legal_lookup, zobrist_constants) {
                if legal_moves.iter().any(|m| m.move_index == winning_move.move_index) {
                    return winning_move;
                }
            }
        }

        // 3. Reset killer table for the new search
        self.killers = [[None; 2]; MAX_PLY];

        // 4. Iterative deepening minimax
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
                None, // Root has no previous move in search tree
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
