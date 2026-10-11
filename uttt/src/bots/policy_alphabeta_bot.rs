use crate::bots::Bot;
use crate::dfpn::DfPnSolver;
use crate::egtb::EgtbDatabase;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::nn::{extract_features, InferenceWorkspace, ModelInt8, DEFAULT_MODEL_BYTES};
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

/// Hybrid Policy-Guided Minimax Bot (`PolicyAlphaBetaBot` / `NnueBot`)
/// Combines pure Rust INT8 neural policy prior guidance with Alpha-Beta minimax,
/// macro-board routing heuristics, and DF-PN/EGTB endgame solving.
pub struct PolicyAlphaBetaBot {
    name: String,
    tt: HashMap<u64, TTEntry>,
    pub killers: [[Option<GameMove>; 2]; MAX_PLY],
    pub history: [[u32; 81]; 81],
    pub countermoves: [Option<GameMove>; 81],
    book: OpeningBook,
    egtb: Option<EgtbDatabase>,
    dfpn: DfPnSolver,
    nn_model: Option<ModelInt8>,
    nn_workspace: InferenceWorkspace,
    legal_lookup: Vec<Vec<GameMove>>,
    cached_root_priors: Option<[f32; 81]>,
}

impl PolicyAlphaBetaBot {
    pub fn new() -> Self {
        let egtb = Some(EgtbDatabase::new());
        let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
        let nn_model = ModelInt8::from_bytes(DEFAULT_MODEL_BYTES).ok();
        let nn_workspace = InferenceWorkspace::new();

        PolicyAlphaBetaBot {
            name: "PolicyAlphaBetaBot".to_string(),
            tt: HashMap::with_capacity(1_000_000),
            killers: [[None; 2]; MAX_PLY],
            history: [[0u32; 81]; 81],
            countermoves: [None; 81],
            book: OpeningBook::new(),
            egtb,
            dfpn: DfPnSolver::new(3_000),
            nn_model,
            nn_workspace,
            legal_lookup,
            cached_root_priors: None,
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

        // 2. Superboard potential lines
        for mask in RESULT_MASKS.iter() {
            let x_count = (board.x_supers & mask).count_ones();
            let o_count = (board.o_supers & mask).count_ones();
            if x_count > 0 && o_count == 0 {
                score += match x_count {
                    2 => 500,
                    1 => 40,
                    _ => 0,
                };
            } else if o_count > 0 && x_count == 0 {
                score -= match o_count {
                    2 => 500,
                    1 => 40,
                    _ => 0,
                };
            }
        }

        // 3. Subboard local micro control
        for sub in 0..9 {
            let mask = 1u16 << sub;
            if ((board.x_supers | board.o_supers) & mask) != 0 {
                continue;
            }
            let x_board = board.xbb[sub];
            let o_board = board.obb[sub];

            for cell in 0..9 {
                let cell_mask = 1u16 << cell;
                if (x_board & cell_mask) != 0 {
                    score += CELL_VALUES[cell];
                } else if (o_board & cell_mask) != 0 {
                    score -= CELL_VALUES[cell];
                }
            }

            for lmask in RESULT_MASKS.iter() {
                let x_c = (x_board & lmask).count_ones();
                let o_c = (o_board & lmask).count_ones();
                if x_c == 2 && o_c == 0 {
                    score += 15;
                } else if o_c == 2 && x_c == 0 {
                    score -= 15;
                }
            }
        }

        score
    }

    /// Macro-board routing penalty / reward calculation
    #[inline(always)]
    fn get_routing_bonus(
        board: &UltimateTicTacToeBoard,
        m: GameMove,
        opp_supers: u16,
        opp_bb: &[u16; 9],
        complete_bbs: u16,
    ) -> i32 {
        let parsed = parse_index(m.move_index);
        let sq = parsed.sub_square_index as usize;
        let mut bonus = 0i32;

        if (complete_bbs & (1 << sq)) != 0 {
            // Directs opponent to completed board -> grants Free Choice!
            bonus -= 300;
        } else {
            let opp_sub = opp_bb[sq];
            let has_threat = RESULT_MASKS.iter().any(|&mask| {
                (opp_sub & mask).count_ones() == 2 && ((board.xbb[sq] | board.obb[sq]) & mask) == 2
            });

            if has_threat {
                let threatens_game_win = RESULT_MASKS.iter().any(|&mask| {
                    ((opp_supers | (1 << sq)) & mask) == mask
                });
                if threatens_game_win {
                    bonus -= 700;
                } else {
                    bonus -= 400;
                }
            } else {
                let open_lines = RESULT_MASKS.iter().any(|&mask| {
                    (board.xbb[sq] & mask) == 0 && (board.obb[sq] & mask) == 0
                });
                if !open_lines {
                    bonus += 150;
                }
            }
        }

        bonus
    }

    /// Compute neural policy priors for moves at the root position
    fn compute_root_priors(&mut self, board: &UltimateTicTacToeBoard) -> [f32; 81] {
        if let Some(ref model) = self.nn_model {
            let (tensor, legal_mask) = extract_features(board);
            let (logits, _val) = model.forward(&tensor, &legal_mask, &mut self.nn_workspace);

            // Compute softmax on legal logits
            let mut max_logit = -1e9f32;
            for i in 0..81 {
                if legal_mask[i] && logits[i] > max_logit {
                    max_logit = logits[i];
                }
            }

            let mut sum_exp = 0.0f32;
            let mut probs = [0.0f32; 81];
            for i in 0..81 {
                if legal_mask[i] {
                    let e = (logits[i] - max_logit).exp();
                    probs[i] = e;
                    sum_exp += e;
                }
            }

            if sum_exp > 0.0 {
                for i in 0..81 {
                    probs[i] /= sum_exp;
                }
            }
            probs
        } else {
            [0.0f32; 81]
        }
    }

    /// Orders candidate moves with neural policy priors at root,
    /// TT best move, killer moves, countermoves, and macro routing.
    fn order_moves(
        &self,
        moves: &mut [GameMove],
        tt_best: Option<GameMove>,
        ply: usize,
        prev_move: Option<GameMove>,
        board: &UltimateTicTacToeBoard,
        root_priors: Option<&[f32; 81]>,
    ) {
        let (my_supers, opp_supers, opp_bb) = if board.x_to_move {
            (board.x_supers, board.o_supers, &board.obb)
        } else {
            (board.o_supers, board.x_supers, &board.xbb)
        };
        let tied_mask = board.get_tied_mask();
        let complete_bbs = board.x_supers | board.o_supers | tied_mask;

        let killer1 = if ply < MAX_PLY { self.killers[ply][0] } else { None };
        let killer2 = if ply < MAX_PLY { self.killers[ply][1] } else { None };
        let countermove = prev_move.and_then(|pm| self.countermoves[pm.move_index as usize]);

        let mut scores: [i32; 81] = [0; 81];

        for (i, &m) in moves.iter().enumerate() {
            let m_idx = m.move_index as usize;

            if Some(m) == tt_best {
                scores[i] = 1_000_000;
                continue;
            }

            // Neural Policy Guidance (if at root)
            let mut neural_bonus = 0i32;
            if let Some(priors) = root_priors {
                neural_bonus = (priors[m_idx] * 50_000.0) as i32;
            }

            // Check if move immediately wins the macro game
            let parsed = parse_index(m.move_index);
            let sb = parsed.sub_board_index as usize;
            let sq = parsed.sub_square_index as usize;
            let sub_after = if board.x_to_move {
                board.xbb[sb] | (1 << sq)
            } else {
                board.obb[sb] | (1 << sq)
            };

            let wins_subboard = RESULT_MASKS.iter().any(|&mask| (sub_after & mask) == mask);
            if wins_subboard {
                let new_supers = my_supers | (1 << sb);
                let wins_game = RESULT_MASKS.iter().any(|&mask| (new_supers & mask) == mask);
                if wins_game {
                    scores[i] = 500_000;
                    continue;
                }
            }

            if Some(m) == killer1 {
                scores[i] = 100_000 + neural_bonus;
                continue;
            }
            if Some(m) == killer2 {
                scores[i] = 90_000 + neural_bonus;
                continue;
            }
            if Some(m) == countermove {
                scores[i] = 80_000 + neural_bonus;
                continue;
            }

            let mut score = neural_bonus;

            if wins_subboard {
                score += 300 + SUBBOARD_VALUES[sb];
            }

            // Macro-board routing bonus/penalty
            score += Self::get_routing_bonus(board, m, opp_supers, opp_bb, complete_bbs);

            // History heuristic
            if let Some(pm) = prev_move {
                score += (self.history[pm.move_index as usize][m_idx] / 10).min(5_000) as i32;
            }

            scores[i] = score;
        }

        // Sort moves descending by score
        let len = moves.len();
        for i in 1..len {
            let key_move = moves[i];
            let key_score = scores[i];
            let mut j = i;
            while j > 0 && scores[j - 1] < key_score {
                moves[j] = moves[j - 1];
                scores[j] = scores[j - 1];
                j -= 1;
            }
            moves[j] = key_move;
            scores[j] = key_score;
        }
    }

    fn decay_history(&mut self, factor: u32) {
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
        beta: i32,
        prev_move: Option<GameMove>,
        zobrist_constants: &ZobristConstants,
        deadline: Instant,
        time_exceeded: &mut bool,
    ) -> i32 {
        if ply & 15 == 0 && Instant::now() >= deadline {
            *time_exceeded = true;
            return 0;
        }

        let hash = board.hash_val();
        let original_alpha = alpha;

        if let Some(entry) = self.tt.get(&hash) {
            if entry.depth >= depth {
                match entry.flag {
                    TTFlag::Exact => return entry.score,
                    TTFlag::LowerBound => {
                        if entry.score >= beta {
                            return entry.score;
                        }
                    }
                    TTFlag::UpperBound => {
                        if entry.score <= alpha {
                            return entry.score;
                        }
                    }
                }
            }
        }

        let resolved_count = board.get_resolved_count();
        if resolved_count >= 7 {
            if let Some(ref mut egtb) = self.egtb {
                if let Some(outcome) = egtb.probe(board) {
                    let score = match outcome {
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
                    let sub = parsed.sub_board_index as usize;
                    let sq = parsed.sub_square_index as usize;
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
        let root_priors = if ply == 0 { self.cached_root_priors.as_ref() } else { None };

        self.order_moves(&mut moves, tt_best, ply, prev_move, board, root_priors);

        let mut best_score = -200_000;
        let mut best_move = moves[0];

        for m in moves {
            let parsed = parse_index(m.move_index);
            let sub = parsed.sub_board_index as usize;
            let sq = parsed.sub_square_index as usize;
            let my_bb = if m.x_to_move { board.xbb[sub] } else { board.obb[sub] };
            let is_tactical = RESULT_MASKS.iter().any(|mask| ((my_bb | (1 << sq)) & mask) == *mask);

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

impl Bot for PolicyAlphaBetaBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {
        self.tt.clear();
        self.killers = [[None; 2]; MAX_PLY];
        self.history = [[0u32; 81]; 81];
        self.countermoves = [None; 81];
        self.cached_root_priors = None;
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

        // 1. Opening Book Probe (trusted entries >= 5,000 visits)
        if let Some(book_move) = self.book.lookup_with_min_visits(board, zobrist_constants, 5_000) {
            if legal_moves.iter().any(|m| m.move_index == book_move.move_index) {
                return book_move;
            }
        }

        // 2. Endgame Tablebase & DF-PN Solver Handoff
        let resolved = (board.x_supers | board.o_supers | board.get_tied_mask()).count_ones();
        if resolved >= 5 {
            self.dfpn.max_nodes = 3_000;
            if let Some((true, Some(winning_move))) = self.dfpn.solve_move(board, &self.legal_lookup, zobrist_constants) {
                if legal_moves.iter().any(|m| m.move_index == winning_move.move_index) {
                    return winning_move;
                }
            }
        }

        // 3. Compute Neural Policy Priors at Root (takes ~1.7ms)
        let priors = self.compute_root_priors(board);
        self.cached_root_priors = Some(priors);

        // 4. Reset killer table for the new search
        self.killers = [[None; 2]; MAX_PLY];

        // 5. Iterative deepening minimax with policy guidance
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
                0,
                -150_000,
                150_000,
                None,
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
