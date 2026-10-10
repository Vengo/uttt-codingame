use crate::bots::Bot;
use crate::dfpn::DfPnSolver;
use crate::egtb::EgtbDatabase;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult, IdentityHasherBuilder};
use crate::opening_book::OpeningBook;
use crate::uttt_trait_solving::{
    parse_index, UltimateTicTacToeBoard, UltimateTicTacToeUnmove, ZobristConstants,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const EXPLORATION_CONSTANT: f64 = 1.41421356;
const DEFAULT_TABLE_CAPACITY: usize = 131_072;
const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProofStatus {
    Unproven,
    ProvenWin,  // Proven win for player whose turn it is at this node
    ProvenLoss, // Proven loss for player whose turn it is at this node
    ProvenDraw, // Proven draw
}

#[derive(Clone, Debug)]
pub struct UmctsV3Node {
    pub visits: u32,
    pub wins: f64,
    pub proof: ProofStatus,
    pub unexpanded_moves: Vec<GameMove>,
    pub children: Vec<(GameMove, u64)>, // (Action, child hash)
}

impl UmctsV3Node {
    pub fn new(board: &UltimateTicTacToeBoard, mut legal_moves: Vec<GameMove>) -> Self {
        // Sort candidate moves ascending by heuristic score so pop() extracts the highest-scoring move first
        legal_moves.sort_by_cached_key(|&m| Self::score_move_quick(board, m));

        UmctsV3Node {
            visits: 0,
            wins: 0.0,
            proof: ProofStatus::Unproven,
            unexpanded_moves: legal_moves,
            children: Vec::new(),
        }
    }

    pub fn score_move_quick(board: &UltimateTicTacToeBoard, m: GameMove) -> i32 {
        let parsed = parse_index(m.move_index);
        let sub = parsed.sub_board_index;
        let sq = parsed.sub_square_index;

        let mut priority = 0i32;

        if sub == 4 {
            priority += 50;
        }
        if sq == 4 {
            priority += 40;
        } else if matches!(sq, 0 | 2 | 6 | 8) {
            priority += 20;
        }

        let is_x = m.x_to_move;
        let my_bb = if is_x { board.xbb[sub] } else { board.obb[sub] };
        let new_my_bb = my_bb | (1 << sq);
        if RESULT_MASKS.iter().any(|mask| (new_my_bb & mask) == *mask) {
            priority += 80;
            let my_supers = if is_x { board.x_supers } else { board.o_supers };
            let new_supers = my_supers | (1 << sub);
            if RESULT_MASKS.iter().any(|mask| (new_supers & mask) == *mask) {
                priority += 10_000;
            }
        }

        priority
    }
}

pub struct UmctsV3Bot {
    name: String,
    node_store: HashMap<u64, UmctsV3Node, IdentityHasherBuilder>,
    unmove_queue: Vec<(u64, UltimateTicTacToeUnmove, GameMove)>,
    rng: SmallRng,
    book: OpeningBook,
    egtb: Option<EgtbDatabase>,
    dfpn: DfPnSolver,
    legal_lookup: Vec<Vec<GameMove>>,
}

impl UmctsV3Bot {
    pub fn new() -> Self {
        let egtb = Some(EgtbDatabase::new());
        let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
        UmctsV3Bot {
            name: "UmctsV3Bot".to_string(),
            node_store: HashMap::with_capacity_and_hasher(
                DEFAULT_TABLE_CAPACITY,
                IdentityHasherBuilder {},
            ),
            unmove_queue: Vec::with_capacity(81),
            rng: SmallRng::from_entropy(),
            book: OpeningBook::new(),
            egtb,
            dfpn: DfPnSolver::new(3_000),
            legal_lookup,
        }
    }

    pub fn node_store(&self) -> &HashMap<u64, UmctsV3Node, IdentityHasherBuilder> {
        &self.node_store
    }

    #[inline(always)]
    fn uct_value(&self, parent_visits: u32, child: &UmctsV3Node) -> f64 {
        if child.visits == 0 {
            return 10_000.0;
        }
        let q = child.wins / (child.visits as f64);
        let u = EXPLORATION_CONSTANT * ((parent_visits as f64).ln() / (child.visits as f64)).sqrt();
        q + u
    }

    fn select_child(&self, parent_hash: u64) -> Option<(GameMove, u64)> {
        let parent = self.node_store.get(&parent_hash)?;
        if parent.children.is_empty() {
            return None;
        }

        // 1. If any child is a ProvenLoss (for that child, meaning a win for parent), select it immediately
        for &(action, child_hash) in &parent.children {
            if let Some(child_node) = self.node_store.get(&child_hash) {
                if child_node.proof == ProofStatus::ProvenLoss {
                    return Some((action, child_hash));
                }
            }
        }

        // 2. Select child with highest UCT score among unproven/drawn children
        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = None;

        for &(action, child_hash) in &parent.children {
            if let Some(child_node) = self.node_store.get(&child_hash) {
                if child_node.proof == ProofStatus::ProvenWin {
                    continue; // Skip lines where the opponent already has a proven win
                }

                let score = self.uct_value(parent.visits, child_node);
                if score > best_score {
                    best_score = score;
                    best_child = Some((action, child_hash));
                }
            }
        }

        best_child.or_else(|| {
            parent
                .children
                .iter()
                .max_by_key(|&(_, h)| self.node_store.get(h).map_or(0, |c| c.visits))
                .copied()
                .or_else(|| parent.children.first().copied())
        })
    }

    fn simulate(
        mut board: UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
        rng: &mut SmallRng,
    ) -> GameResult {
        let mut depth = 0;
        while board.get_win_state() == GameResult::Unresolved && depth < 60 {
            let moves = board.get_legal_moves_fast();
            if moves.is_empty() {
                break;
            }
            let chosen = *moves.choose(rng).unwrap();
            board.make_move(chosen, zobrist_constants);
            depth += 1;
        }
        board.get_win_state()
    }

    fn update_proof_status(&mut self, node_hash: u64, child_proof: ProofStatus) -> ProofStatus {
        if child_proof == ProofStatus::ProvenLoss {
            if let Some(n) = self.node_store.get_mut(&node_hash) {
                n.proof = ProofStatus::ProvenWin;
            }
            return ProofStatus::ProvenWin;
        }

        let node = match self.node_store.get(&node_hash) {
            Some(n) => n,
            None => return ProofStatus::Unproven,
        };

        if node.proof != ProofStatus::Unproven || !node.unexpanded_moves.is_empty() {
            return node.proof;
        }

        // If the updated child is unproven, all children cannot possibly be opponent wins
        if child_proof == ProofStatus::Unproven {
            return ProofStatus::Unproven;
        }

        let mut all_children_opponent_win = true;
        let mut any_child_draw = false;

        for &(_action, child_hash) in &node.children {
            if let Some(child_node) = self.node_store.get(&child_hash) {
                match child_node.proof {
                    ProofStatus::ProvenLoss => {
                        if let Some(n) = self.node_store.get_mut(&node_hash) {
                            n.proof = ProofStatus::ProvenWin;
                        }
                        return ProofStatus::ProvenWin;
                    }
                    ProofStatus::ProvenDraw => {
                        any_child_draw = true;
                    }
                    ProofStatus::ProvenWin => {}
                    ProofStatus::Unproven => {
                        all_children_opponent_win = false;
                        break;
                    }
                }
            } else {
                all_children_opponent_win = false;
                break;
            }
        }

        if all_children_opponent_win {
            let new_proof = if any_child_draw {
                ProofStatus::ProvenDraw
            } else {
                ProofStatus::ProvenLoss
            };
            if let Some(n) = self.node_store.get_mut(&node_hash) {
                n.proof = new_proof;
            }
            new_proof
        } else {
            ProofStatus::Unproven
        }
    }
}

#[inline(always)]
fn compute_full_hash(board: &UltimateTicTacToeBoard) -> u64 {
    board.hash_val() ^ ((board.last_super_index as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

impl Default for UmctsV3Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for UmctsV3Bot {
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
        _legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        let root_moves = board.get_legal_moves_fast();
        if root_moves.len() <= 1 {
            return root_moves.first().copied().unwrap_or_default();
        }

        // 1. Query Opening Book at root (only trusted entries with >= 5,000 visits)
        if let Some(book_move) = self.book.lookup_with_min_visits(board, zobrist_constants, 5_000) {
            if root_moves.iter().any(|m| m.move_index == book_move.move_index) {
                return book_move;
            }
        }

        // 2. Query DF-PN Solver at root in late game (>= 5 resolved subboards)
        let resolved = (board.x_supers | board.o_supers | board.get_tied_mask()).count_ones();
        if resolved >= 5 {
            self.dfpn.max_nodes = 3_000;
            if let Some((true, Some(winning_move))) = self.dfpn.solve_move(board, &self.legal_lookup, zobrist_constants) {
                if root_moves.iter().any(|m| m.move_index == winning_move.move_index) {
                    return winning_move;
                }
            }
        }

        self.node_store.clear();
        self.unmove_queue.clear();

        let root_hash = compute_full_hash(board);
        self.node_store
            .insert(root_hash, UmctsV3Node::new(board, root_moves.clone()));

        let start_time = Instant::now();
        let deadline = start_time + time_limit;
        let mut sim_board = board.clone();

        while Instant::now() < deadline {
            let mut curr_hash = root_hash;
            self.unmove_queue.clear();

            // 1. Selection
            loop {
                let win_state = sim_board.get_win_state();
                if win_state != GameResult::Unresolved {
                    break;
                }

                let curr_node = match self.node_store.get(&curr_hash) {
                    Some(n) => n,
                    None => break,
                };

                if !curr_node.unexpanded_moves.is_empty() {
                    break;
                }

                if curr_node.proof != ProofStatus::Unproven {
                    break;
                }

                if let Some((action, next_hash)) = self.select_child(curr_hash) {
                    let undo = sim_board.make_move(action, zobrist_constants);
                    self.unmove_queue.push((curr_hash, undo, action));
                    curr_hash = next_hash;
                } else {
                    break;
                }
            }

            // 2. Expansion & Tablebase Probing
            let win_state = sim_board.get_win_state();
            let mut egtb_result = None;

            if win_state == GameResult::Unresolved {
                if let Some(curr_node) = self.node_store.get_mut(&curr_hash) {
                    if let Some(m) = curr_node.unexpanded_moves.pop() {
                        let undo = sim_board.make_move(m, zobrist_constants);
                        let child_hash = compute_full_hash(&sim_board);
                        curr_node.children.push((m, child_hash));

                        let child_win_state = sim_board.get_win_state();
                        let child_moves = if child_win_state == GameResult::Unresolved {
                            sim_board.get_legal_moves_fast()
                        } else {
                            Vec::new()
                        };

                        let mut child_node = UmctsV3Node::new(&sim_board, child_moves.clone());

                        if child_win_state != GameResult::Unresolved {
                            child_node.proof = match child_win_state {
                                GameResult::Tie => ProofStatus::ProvenDraw,
                                GameResult::P1Win => {
                                    if m.x_to_move {
                                        ProofStatus::ProvenLoss
                                    } else {
                                        ProofStatus::ProvenWin
                                    }
                                }
                                GameResult::P2Win => {
                                    if !m.x_to_move {
                                        ProofStatus::ProvenLoss
                                    } else {
                                        ProofStatus::ProvenWin
                                    }
                                }
                                _ => ProofStatus::Unproven,
                            };
                        } else {
                            // 1. 2-Ply Grandfather refutation check:
                            // Can the opponent (whose turn it now is from sim_board) immediately win the game?
                            let opp_supers = if m.x_to_move { sim_board.o_supers } else { sim_board.x_supers };
                            let opp_has_threat = RESULT_MASKS.iter().any(|mask| (opp_supers & mask).count_ones() == 2);
                            let opp_can_win_immediately = opp_has_threat && child_moves.iter().any(|&opp_m| {
                                UmctsV3Node::score_move_quick(&sim_board, opp_m) >= 10_000
                            });

                            if opp_can_win_immediately {
                                // Opponent has an immediate winning response from sim_board!
                                // From child's perspective (opponent to move), this is a ProvenWin.
                                child_node.proof = ProofStatus::ProvenWin;
                            } else {
                                // 2. In-tree EGTB Probing (resolved >= 7)
                                let resolved = (sim_board.x_supers
                                    | sim_board.o_supers
                                    | sim_board.get_tied_mask())
                                .count_ones();

                                if resolved >= 7 {
                                    if let Some(ref egtb) = self.egtb {
                                        if let Some(res) = egtb.probe(&sim_board, zobrist_constants) {
                                            egtb_result = Some(res);
                                            let active_is_p1 = sim_board.x_to_move;
                                            child_node.proof = match res {
                                                GameResult::P1Win => {
                                                    if active_is_p1 {
                                                        ProofStatus::ProvenWin
                                                    } else {
                                                        ProofStatus::ProvenLoss
                                                    }
                                                }
                                                GameResult::P2Win => {
                                                    if !active_is_p1 {
                                                        ProofStatus::ProvenWin
                                                    } else {
                                                        ProofStatus::ProvenLoss
                                                    }
                                                }
                                                GameResult::Tie => ProofStatus::ProvenDraw,
                                                GameResult::Unresolved => ProofStatus::Unproven,
                                            };
                                        }
                                    }
                                }
                            }
                        }

                        self.node_store.insert(child_hash, child_node);
                        self.unmove_queue.push((curr_hash, undo, m));
                        curr_hash = child_hash;
                    }
                }
            }

            // 3. Simulation (skipped if terminal, solved by EGTB, or proven by tactical/DF-PN check)
            let curr_proof = self.node_store.get(&curr_hash).map_or(ProofStatus::Unproven, |n| n.proof);
            let result = if curr_proof != ProofStatus::Unproven {
                let active_is_p1 = sim_board.x_to_move;
                match curr_proof {
                    ProofStatus::ProvenWin => if active_is_p1 { GameResult::P1Win } else { GameResult::P2Win },
                    ProofStatus::ProvenLoss => if active_is_p1 { GameResult::P2Win } else { GameResult::P1Win },
                    ProofStatus::ProvenDraw => GameResult::Tie,
                    ProofStatus::Unproven => unreachable!(),
                }
            } else if let Some(res) = egtb_result {
                res
            } else if sim_board.get_win_state() != GameResult::Unresolved {
                sim_board.get_win_state()
            } else {
                Self::simulate(sim_board.clone(), zobrist_constants, &mut self.rng)
            };

            // 4. Backpropagation & Proof Propagation (Unmaking moves up the queue)
            if let Some(leaf_node) = self.node_store.get_mut(&curr_hash) {
                leaf_node.visits += 1;
            }

            while let Some((parent_hash, undo, action)) = self.unmove_queue.pop() {
                let player_just_moved = action.x_to_move;
                let reward = match result {
                    GameResult::P1Win => {
                        if player_just_moved {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    GameResult::P2Win => {
                        if !player_just_moved {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    GameResult::Tie => 0.5,
                    GameResult::Unresolved => 0.5,
                };

                let child_proof = if let Some(child_node) = self.node_store.get_mut(&curr_hash) {
                    child_node.wins += reward;
                    child_node.proof
                } else {
                    ProofStatus::Unproven
                };

                if let Some(parent_node) = self.node_store.get_mut(&parent_hash) {
                    parent_node.visits += 1;
                }

                self.update_proof_status(parent_hash, child_proof);
                sim_board.unmake_move(undo);
                curr_hash = parent_hash;
            }

            if let Some(root_node) = self.node_store.get(&root_hash) {
                if root_node.proof == ProofStatus::ProvenWin {
                    break;
                }
            }
        }

        // Return child with highest visit count (or proven win)
        let root_node = match self.node_store.get(&root_hash) {
            Some(n) => n,
            None => return root_moves[0],
        };

        for &(action, child_hash) in &root_node.children {
            if root_moves.iter().any(|m| m.move_index == action.move_index) {
                if let Some(child_node) = self.node_store.get(&child_hash) {
                    if child_node.proof == ProofStatus::ProvenLoss {
                        return action;
                    }
                }
            }
        }

        let unrefuted_children: Vec<_> = root_node
            .children
            .iter()
            .filter(|&(action, child_hash)| {
                root_moves.iter().any(|m| m.move_index == action.move_index)
                    && self
                        .node_store
                        .get(child_hash)
                        .map_or(true, |c| c.proof != ProofStatus::ProvenWin)
            })
            .collect();

        if let Some(&&(action, _)) = unrefuted_children.iter().max_by_key(|&&(_, child_hash)| {
            self.node_store.get(child_hash).map_or(0, |c| c.visits)
        }) {
            return action;
        }

        // Fallback if all children were refuted: pick child with highest visits
        root_node
            .children
            .iter()
            .filter(|&(action, _)| root_moves.iter().any(|m| m.move_index == action.move_index))
            .max_by_key(|&(_, child_hash)| {
                self.node_store.get(child_hash).map_or(0, |c| c.visits)
            })
            .map(|&(action, _)| action)
            .unwrap_or(root_moves[0])
    }
}
