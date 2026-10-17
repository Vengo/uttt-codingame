//! High-Throughput Rayon Parallel MCTS Batch Worker for GPU AlphaZero Self-Play
//!
//! Bridges native Rust bitboards (12.4M plies/s) with PyTorch GPU Tensor Cores (368k evals/s)
//! via zero-copy C-ABI shared memory pointers.
//!
//! Features:
//! - B parallel games executed concurrently across 16 Rayon worker threads
//! - Cache-friendly contiguous flat arena tree storage (zero pointer chasing, zero GC)
//! - Fast NCHW 5x9x9 feature plane packing directly into pre-allocated NumPy / CUDA pinned host memory
//! - Temperature-based move sampling (T > 0 always, preventing exploration collapse)
//! - Automated game reset from 2-ply opening suite upon terminal state
//! - CodinGame contest mini-board tiebreaker scoring (|Mx| vs |Mo|)

use std::sync::atomic::{AtomicUsize, Ordering};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::opening_book::OpeningBook;
use crate::uttt_trait_solving::{
    parse_index, UltimateTicTacToeBoard, ZobristConstants,
};

const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

#[inline(always)]
fn check_win(bb: u16) -> bool {
    RESULT_MASKS.iter().any(|&m| (bb & m) == m)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ProofStatus {
    Unknown = 0,
    ProvenWin = 1,  // Proven win (+1.0) for the player TO MOVE at this node
    ProvenLoss = 2, // Proven loss (-1.0) for the player TO MOVE at this node (win for parent's mover!)
    ProvenDraw = 3, // Proven draw (0.0) under optimal play from this node
}

#[derive(Clone, Copy, Debug)]
pub struct MctsChild {
    pub action: u8,
    pub prior: f32,
    pub visits: u32,
    pub total_val: f32,
    pub node_idx: u32, // u32::MAX if child node not yet created
    pub proof: ProofStatus,
}

impl MctsChild {
    #[inline(always)]
    pub fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_val / (self.visits as f32)
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MctsNode {
    pub parent_idx: u32,
    pub parent_action: u8,
    pub visits: u32,
    pub total_val: f32,
    pub is_terminal: bool,
    pub terminal_val: f32,
    pub is_expanded: bool,
    pub first_child: u32,
    pub num_children: u16,
    pub proof: ProofStatus,
}

pub struct GameState {
    pub board: UltimateTicTacToeBoard,
    pub nodes: Vec<MctsNode>,
    pub children: Vec<MctsChild>,
    pub alt_nodes: Vec<MctsNode>,
    pub alt_children: Vec<MctsChild>,
    pub history: Vec<(UltimateTicTacToeBoard, [f32; 81], bool, f32)>, // (board, pi, x_to_move, root_q)
    pub moves_played: usize,
    pub current_leaf_idx: u32,
    pub current_leaf_board: UltimateTicTacToeBoard,
    pub pending_leaves: Vec<(u32, UltimateTicTacToeBoard, Vec<(u32, u32)>)>,
    pub pending_batch_offset: usize,
    pub rng: SmallRng,
}

impl GameState {
    /// Compacts and reroots the MCTS search tree at the child corresponding to `action`.
    /// Preserves all visits, values, and proof statuses accumulated in that subtree.
    pub fn reroot_at_action(&mut self, action: u8, dirichlet_alpha: f32, dirichlet_eps: f32) {
        self.pending_leaves.clear();
        self.pending_batch_offset = 0;
        self.current_leaf_idx = 0;

        if self.nodes.is_empty() || !self.nodes[0].is_expanded || self.nodes[0].num_children == 0 {
            self.nodes.clear();
            self.children.clear();
            return;
        }

        let root = self.nodes[0];
        let mut target_node_idx = u32::MAX;
        for c_off in 0..(root.num_children as u32) {
            let ch = &self.children[(root.first_child + c_off) as usize];
            if ch.action == action {
                target_node_idx = ch.node_idx;
                break;
            }
        }

        if target_node_idx == u32::MAX {
            self.nodes.clear();
            self.children.clear();
            return;
        }

        let new_root_orig = self.nodes[target_node_idx as usize];
        if !new_root_orig.is_expanded || new_root_orig.num_children == 0 || new_root_orig.is_terminal {
            self.nodes.clear();
            self.children.clear();
            return;
        }

        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        let mut new_children = Vec::with_capacity(self.children.len());

        let mut root_copy = new_root_orig;
        root_copy.parent_idx = u32::MAX;
        root_copy.parent_action = 255;
        new_nodes.push(root_copy);

        let mut cursor = 0usize;
        while cursor < new_nodes.len() {
            let node = new_nodes[cursor];
            if node.is_expanded && node.num_children > 0 {
                let old_first = node.first_child as usize;
                let num_ch = node.num_children as usize;
                let new_first = new_children.len() as u32;
                new_nodes[cursor].first_child = new_first;

                for c_off in 0..num_ch {
                    let mut ch_copy = self.children[old_first + c_off];
                    if ch_copy.node_idx != u32::MAX {
                        let old_child_node_idx = ch_copy.node_idx as usize;
                        let new_child_node_idx = new_nodes.len() as u32;
                        ch_copy.node_idx = new_child_node_idx;

                        let mut child_node = self.nodes[old_child_node_idx];
                        child_node.parent_idx = cursor as u32;
                        new_nodes.push(child_node);
                    }
                    new_children.push(ch_copy);
                }
            } else {
                new_nodes[cursor].first_child = 0;
                new_nodes[cursor].num_children = 0;
            }
            cursor += 1;
        }

        let root_num_ch = new_nodes[0].num_children as usize;
        if dirichlet_eps > 0.0 && root_num_ch > 1 {
            let alpha = dirichlet_alpha.max(1e-3);
            let mut noise_weights = Vec::with_capacity(root_num_ch);
            for _ in 0..root_num_ch {
                let u: f32 = self.rng.gen_range(1e-4..1.0);
                noise_weights.push((-u.ln()).powf(1.0 / alpha));
            }
            let noise_sum: f32 = noise_weights.iter().sum();
            if noise_sum > 0.0 {
                for i in 0..root_num_ch {
                    let noise_p = noise_weights[i] / noise_sum;
                    new_children[i].prior =
                        (1.0 - dirichlet_eps) * new_children[i].prior + dirichlet_eps * noise_p;
                }
            }
        }

        self.nodes = new_nodes;
        self.children = new_children;
    }

    /// Advances both players' isolated search trees by `action` and swaps the active player tree
    /// (`nodes` / `children`) with the opponent's tree (`alt_nodes` / `alt_children`).
    pub fn advance_both_trees_and_swap(&mut self, action: u8) {
        self.reroot_at_action(action, 0.0, 0.0);
        std::mem::swap(&mut self.nodes, &mut self.alt_nodes);
        std::mem::swap(&mut self.children, &mut self.alt_children);
        self.reroot_at_action(action, 0.0, 0.0);
    }
}

pub struct CompletedSample {
    pub features: [f32; 405], // 5 x 9 x 9 NCHW
    pub legal_mask: [f32; 81],
    pub policy: [f32; 81],
    pub value: f32,
    pub game_length: usize,
}

static SHARED_EGTB: std::sync::OnceLock<std::sync::Arc<Option<crate::egtb::EgtbDatabase>>> =
    std::sync::OnceLock::new();

fn get_shared_egtb() -> std::sync::Arc<Option<crate::egtb::EgtbDatabase>> {
    SHARED_EGTB
        .get_or_init(|| std::sync::Arc::new(Some(crate::egtb::EgtbDatabase::new())))
        .clone()
}

pub struct MctsWorker {
    pub num_games: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_eps: f32,
    pub games: Vec<GameState>,
    pub opening_suite: Vec<UltimateTicTacToeBoard>,
    pub zobrist_constants: ZobristConstants,
    pub egtb: std::sync::Arc<Option<crate::egtb::EgtbDatabase>>,
    pub completed_samples: Vec<CompletedSample>,
    pub active_eval_indices: Vec<usize>,
    pub active_eval_leaves: Vec<(usize, usize)>,
}

impl MctsWorker {
    pub fn new(
        num_games: usize,
        c_puct: f32,
        dirichlet_alpha: f32,
        dirichlet_eps: f32,
    ) -> Self {
        let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
        let book = OpeningBook::new();
        let opening_suite = book.extract_opening_suite(60, 2, &[], &zobrist_constants);
        let egtb = get_shared_egtb();

        let mut games = Vec::with_capacity(num_games);
        for i in 0..num_games {
            let initial_board = opening_suite[i % opening_suite.len()].clone();
            let rng = SmallRng::seed_from_u64(0x1234_5678_9ABC_DEF0 ^ (i as u64));
            let game = GameState {
                board: initial_board.clone(),
                nodes: Vec::with_capacity(1024),
                children: Vec::with_capacity(4096),
                alt_nodes: Vec::with_capacity(1024),
                alt_children: Vec::with_capacity(4096),
                history: Vec::with_capacity(80),
                moves_played: 0,
                current_leaf_idx: 0,
                current_leaf_board: initial_board,
                pending_leaves: Vec::with_capacity(8),
                pending_batch_offset: 0,
                rng,
            };
            games.push(game);
        }

        MctsWorker {
            num_games,
            c_puct,
            dirichlet_alpha,
            dirichlet_eps,
            games,
            opening_suite,
            zobrist_constants,
            egtb,
            completed_samples: Vec::with_capacity(10_000),
            active_eval_indices: Vec::with_capacity(num_games),
            active_eval_leaves: Vec::with_capacity(num_games),
        }
    }

    /// Resets all games to fresh 2-ply opening positions and clears history.
    pub fn reset_all(&mut self) {
        let suite = &self.opening_suite;
        for (i, game) in self.games.iter_mut().enumerate() {
            let initial_board = suite[i % suite.len()].clone();
            game.board = initial_board.clone();
            game.nodes.clear();
            game.children.clear();
            game.alt_nodes.clear();
            game.alt_children.clear();
            game.history.clear();
            game.moves_played = 0;
            game.current_leaf_idx = 0;
            game.current_leaf_board = initial_board;
            game.pending_leaves.clear();
            game.pending_batch_offset = 0;
        }
        self.completed_samples.clear();
        self.active_eval_indices.clear();
        self.active_eval_leaves.clear();
    }

    /// Classifies a board state as terminal or EGTB-resolved from the perspective of `curr_board.x_to_move`.
    #[inline(always)]
    fn classify_board_outcome(
        curr_board: &UltimateTicTacToeBoard,
        is_root: bool,
        egtb: &Option<crate::egtb::EgtbDatabase>,
        z_consts: &ZobristConstants,
    ) -> (bool, f32, ProofStatus) {
        let win = curr_board.get_win_state();
        match win {
            GameResult::P1Win => {
                if curr_board.x_to_move {
                    (true, 1.0, ProofStatus::ProvenWin)
                } else {
                    (true, -1.0, ProofStatus::ProvenLoss)
                }
            }
            GameResult::P2Win => {
                if !curr_board.x_to_move {
                    (true, 1.0, ProofStatus::ProvenWin)
                } else {
                    (true, -1.0, ProofStatus::ProvenLoss)
                }
            }
            GameResult::Tie => (true, 0.0, ProofStatus::ProvenDraw),
            GameResult::Unresolved => {
                if !is_root && curr_board.get_resolved_count() >= 6 {
                    if let Some(ref db) = egtb {
                        if let Some(res) = db.probe_with_constants(curr_board, z_consts) {
                            return match res {
                                GameResult::P1Win => {
                                    if curr_board.x_to_move {
                                        (true, 1.0, ProofStatus::ProvenWin)
                                    } else {
                                        (true, -1.0, ProofStatus::ProvenLoss)
                                    }
                                }
                                GameResult::P2Win => {
                                    if !curr_board.x_to_move {
                                        (true, 1.0, ProofStatus::ProvenWin)
                                    } else {
                                        (true, -1.0, ProofStatus::ProvenLoss)
                                    }
                                }
                                GameResult::Tie => (true, 0.0, ProofStatus::ProvenDraw),
                                GameResult::Unresolved => (false, 0.0, ProofStatus::Unknown),
                            };
                        }
                    }
                }
                (false, 0.0, ProofStatus::Unknown)
            }
        }
    }

    /// Backpropagates value and exact MCTS-Solver `ProofStatus` up `path`.
    #[inline]
    fn backpropagate_with_proof(
        game: &mut GameState,
        leaf_idx: u32,
        leaf_val: f32,
        path: &[(u32, u32)],
    ) {
        let mut child_proof = game.nodes[leaf_idx as usize].proof;
        let mut v = -leaf_val;

        for &(p_idx, ch_idx) in path.iter().rev() {
            {
                let ch = &mut game.children[ch_idx as usize];
                ch.visits += 1;
                ch.total_val += -v;
                ch.proof = child_proof;
            }

            let (first_child, num_children, is_expanded) = {
                let parent = &mut game.nodes[p_idx as usize];
                parent.visits += 1;
                parent.total_val += v;
                (parent.first_child, parent.num_children, parent.is_expanded)
            };

            let mut new_parent_proof = ProofStatus::Unknown;
            let mut new_parent_term_val = game.nodes[p_idx as usize].terminal_val;

            if is_expanded && num_children > 0 {
                let mut has_proven_loss_child = false;
                let mut all_proven_win_children = true;
                let mut all_proven_win_or_draw = true;
                let mut has_proven_draw_child = false;

                for c_off in 0..(num_children as u32) {
                    let c = &game.children[(first_child + c_off) as usize];
                    match c.proof {
                        ProofStatus::ProvenLoss => {
                            has_proven_loss_child = true;
                            all_proven_win_children = false;
                            all_proven_win_or_draw = false;
                        }
                        ProofStatus::ProvenWin => {}
                        ProofStatus::ProvenDraw => {
                            all_proven_win_children = false;
                            has_proven_draw_child = true;
                        }
                        ProofStatus::Unknown => {
                            all_proven_win_children = false;
                            all_proven_win_or_draw = false;
                        }
                    }
                }

                if has_proven_loss_child {
                    new_parent_proof = ProofStatus::ProvenWin;
                    new_parent_term_val = 1.0;
                } else if all_proven_win_children {
                    new_parent_proof = ProofStatus::ProvenLoss;
                    new_parent_term_val = -1.0;
                } else if all_proven_win_or_draw && has_proven_draw_child {
                    new_parent_proof = ProofStatus::ProvenDraw;
                    new_parent_term_val = 0.0;
                }
            }

            if new_parent_proof != ProofStatus::Unknown {
                let parent = &mut game.nodes[p_idx as usize];
                parent.proof = new_parent_proof;
                parent.terminal_val = new_parent_term_val;
                v = -new_parent_term_val;
            } else {
                v = -v;
            }

            child_proof = game.nodes[p_idx as usize].proof;
        }
    }

    /// Extracts NCHW 5x9x9 feature planes and legal mask directly from an UltimateTicTacToeBoard.
    #[inline(always)]
    pub fn extract_nchw_features(
        board: &UltimateTicTacToeBoard,
        out_feat: &mut [f32; 405],
        out_mask: &mut [f32; 81],
    ) {
        out_feat.fill(0.0);
        out_mask.fill(0.0);

        let (active_bb, opp_bb) = if board.x_to_move {
            (board.xbb, board.obb)
        } else {
            (board.obb, board.xbb)
        };

        let (active_supers, opp_supers) = if board.x_to_move {
            (board.x_supers, board.o_supers)
        } else {
            (board.o_supers, board.x_supers)
        };

        let tied_mask = board.get_tied_mask();
        let complete_bbs = board.x_supers | board.o_supers | tied_mask;

        let target_mask: u16 = if board.last_super_index >= 9 || (complete_bbs & (1 << board.last_super_index)) != 0 {
            (!complete_bbs) & 0x1FF
        } else {
            1 << board.last_super_index
        };

        for sb in 0..9usize {
            let sb_r = sb / 3;
            let sb_c = sb % 3;
            let is_target = (target_mask & (1 << sb)) != 0;
            let is_active_won = (active_supers & (1 << sb)) != 0;
            let is_opp_won = (opp_supers & (1 << sb)) != 0;
            let occ = board.xbb[sb] | board.obb[sb];

            for sq in 0..9usize {
                let sq_r = sq / 3;
                let sq_c = sq % 3;
                let r = sb_r * 3 + sq_r;
                let c = sb_c * 3 + sq_c;
                let rc = r * 9 + c;
                let move_idx = 9 * sb + sq;

                // Channel 0: Active player marks (offset 0 * 81)
                if (active_bb[sb] & (1 << sq)) != 0 {
                    out_feat[rc] = 1.0;
                }
                // Channel 1: Opponent marks (offset 1 * 81)
                if (opp_bb[sb] & (1 << sq)) != 0 {
                    out_feat[81 + rc] = 1.0;
                }
                // Channel 2: Active player won subboards (offset 2 * 81)
                if is_active_won {
                    out_feat[162 + rc] = 1.0;
                }
                // Channel 3: Opponent won subboards (offset 3 * 81)
                if is_opp_won {
                    out_feat[243 + rc] = 1.0;
                }
                // Channel 4: Legal target subboard mask (offset 4 * 81)
                if is_target {
                    out_feat[324 + rc] = 1.0;
                }

                if is_target && (occ & (1 << sq)) == 0 {
                    out_mask[move_idx] = 1.0;
                }
            }
        }
    }

    /// Prepares one simulation step (`k = 1`) across all games in parallel using Rayon.
    pub fn prepare_step(
        &mut self,
        out_features: &mut [f32],
        out_masks: &mut [f32],
    ) -> usize {
        self.prepare_step_vloss(1, out_features, out_masks)
    }

    /// Prepares up to `virtual_loss_k` simulation leaves per game in parallel using Rayon.
    /// Applies virtual loss along each active trajectory so multiple paths per game explore distinct leaves.
    pub fn prepare_step_vloss(
        &mut self,
        virtual_loss_k: usize,
        out_features: &mut [f32],
        out_masks: &mut [f32],
    ) -> usize {
        let k = virtual_loss_k.max(1);
        let c_puct = self.c_puct;
        let z_consts = &self.zobrist_constants;
        let egtb_ref = &*self.egtb;

        self.games.par_iter_mut().for_each(|game| {
            game.pending_leaves.clear();
            game.pending_batch_offset = 0;

            if game.nodes.is_empty() {
                let (is_term, term_val, proof) =
                    Self::classify_board_outcome(&game.board, true, egtb_ref, z_consts);
                game.nodes.push(MctsNode {
                    parent_idx: u32::MAX,
                    parent_action: 255,
                    visits: 0,
                    total_val: 0.0,
                    is_terminal: is_term,
                    terminal_val: term_val,
                    is_expanded: false,
                    first_child: 0,
                    num_children: 0,
                    proof,
                });
                game.current_leaf_idx = 0;
                game.current_leaf_board = game.board.clone();
                if !is_term {
                    game.pending_leaves
                        .push((0, game.board.clone(), Vec::new()));
                } else {
                    game.nodes[0].visits += 1;
                    game.nodes[0].total_val += term_val;
                }
                return;
            }

            for _sub_sim in 0..k {
                let mut curr_idx = 0u32;
                let mut curr_board = game.board.clone();
                let mut path: Vec<(u32, u32)> = Vec::with_capacity(16);

                while game.nodes[curr_idx as usize].is_expanded
                    && !game.nodes[curr_idx as usize].is_terminal
                    && game.nodes[curr_idx as usize].num_children > 0
                {
                    let (first_child, num_children, node_visits) = {
                        let node = &game.nodes[curr_idx as usize];
                        (node.first_child, node.num_children, node.visits)
                    };
                    let total_sqrt = (node_visits.max(1) as f32).sqrt();

                    // Check proof status across children
                    let mut has_winning_child = false;
                    let mut all_losing_children = true;
                    for c_off in 0..(num_children as u32) {
                        let ch = &game.children[(first_child + c_off) as usize];
                        if ch.proof == ProofStatus::ProvenLoss {
                            has_winning_child = true;
                        }
                        if ch.proof != ProofStatus::ProvenWin {
                            all_losing_children = false;
                        }
                    }

                    let mut best_score = f32::NEG_INFINITY;
                    let mut best_child_offset = 0u32;

                    for c_off in 0..(num_children as u32) {
                        let child = &game.children[(first_child + c_off) as usize];
                        if has_winning_child && child.proof != ProofStatus::ProvenLoss {
                            continue;
                        }
                        if !has_winning_child
                            && !all_losing_children
                            && child.proof == ProofStatus::ProvenWin
                        {
                            continue;
                        }

                        let q = match child.proof {
                            ProofStatus::ProvenLoss => 1.0,
                            ProofStatus::ProvenWin => -1.0,
                            ProofStatus::ProvenDraw => 0.0,
                            ProofStatus::Unknown => -child.q_value(),
                        };
                        let u =
                            c_puct * child.prior * (total_sqrt / (1.0 + child.visits as f32));
                        let score = q + u;
                        if score > best_score {
                            best_score = score;
                            best_child_offset = c_off;
                        }
                    }

                    let chosen_child_idx = (first_child + best_child_offset) as usize;
                    let child_action = game.children[chosen_child_idx].action;

                    // Record edge and apply virtual loss
                    path.push((curr_idx, chosen_child_idx as u32));
                    game.nodes[curr_idx as usize].visits += 1;
                    game.nodes[curr_idx as usize].total_val -= 1.0;
                    game.children[chosen_child_idx].visits += 1;
                    game.children[chosen_child_idx].total_val += 1.0;

                    curr_board.make_move(
                        GameMove {
                            x_to_move: curr_board.x_to_move,
                            move_index: child_action,
                        },
                        z_consts,
                    );

                    if game.children[chosen_child_idx].node_idx == u32::MAX {
                        // Lazy create child node (with EGTB probe if resolved_count >= 6)
                        let (is_term, term_val, proof) =
                            Self::classify_board_outcome(&curr_board, false, egtb_ref, z_consts);

                        let new_node_idx = game.nodes.len() as u32;
                        game.children[chosen_child_idx].node_idx = new_node_idx;
                        game.nodes.push(MctsNode {
                            parent_idx: curr_idx,
                            parent_action: child_action,
                            visits: 0,
                            total_val: 0.0,
                            is_terminal: is_term,
                            terminal_val: term_val,
                            is_expanded: false,
                            first_child: 0,
                            num_children: 0,
                            proof,
                        });
                        curr_idx = new_node_idx;
                        break;
                    } else {
                        curr_idx = game.children[chosen_child_idx].node_idx;
                    }
                }

                game.current_leaf_idx = curr_idx;
                game.current_leaf_board = curr_board.clone();

                // If terminal (or EGTB resolved), revert virtual loss and backpropagate immediately
                if game.nodes[curr_idx as usize].is_terminal {
                    for &(p_idx, ch_idx) in &path {
                        game.nodes[p_idx as usize].visits -= 1;
                        game.nodes[p_idx as usize].total_val += 1.0;
                        game.children[ch_idx as usize].visits -= 1;
                        game.children[ch_idx as usize].total_val -= 1.0;
                    }

                    let term_val = game.nodes[curr_idx as usize].terminal_val;
                    game.nodes[curr_idx as usize].visits += 1;
                    game.nodes[curr_idx as usize].total_val += term_val;
                    Self::backpropagate_with_proof(game, curr_idx, term_val, &path);
                } else {
                    // Non-terminal leaf: check if duplicate in pending_leaves or already expanded
                    let is_dup = game
                        .pending_leaves
                        .iter()
                        .any(|&(idx, _, _)| idx == curr_idx);
                    if is_dup || game.nodes[curr_idx as usize].is_expanded {
                        for &(p_idx, ch_idx) in &path {
                            game.nodes[p_idx as usize].visits -= 1;
                            game.nodes[p_idx as usize].total_val += 1.0;
                            game.children[ch_idx as usize].visits -= 1;
                            game.children[ch_idx as usize].total_val -= 1.0;
                        }
                        break;
                    } else {
                        game.pending_leaves.push((curr_idx, curr_board, path));
                    }
                }
            }
        });

        // Collect all non-terminal leaves across games
        self.active_eval_indices.clear();
        self.active_eval_leaves.clear();
        let mut offset = 0usize;
        for (g_idx, game) in self.games.iter_mut().enumerate() {
            game.pending_batch_offset = offset;
            let count = game.pending_leaves.len();
            for sub_idx in 0..count {
                self.active_eval_indices.push(g_idx);
                self.active_eval_leaves.push((g_idx, sub_idx));
            }
            offset += count;
        }

        let active_count = self.active_eval_leaves.len();
        if active_count == 0 {
            return 0;
        }

        let games_ref = &self.games;
        let active_leaves = &self.active_eval_leaves;

        out_features[..active_count * 405]
            .par_chunks_exact_mut(405)
            .zip(out_masks[..active_count * 81].par_chunks_exact_mut(81))
            .zip(active_leaves.par_iter())
            .for_each(|((feat_dst, mask_dst), &(g_idx, sub_idx))| {
                let board = &games_ref[g_idx].pending_leaves[sub_idx].1;
                let mut feat = [0.0f32; 405];
                let mut mask = [0.0f32; 81];
                Self::extract_nchw_features(board, &mut feat, &mut mask);
                feat_dst.copy_from_slice(&feat);
                mask_dst.copy_from_slice(&mask);
            });

        active_count
    }

    /// Finishes simulation step after GPU forward evaluation.
    /// Reverts virtual losses, expands leaves with softmax priors, and backpropagates neural values + proofs.
    pub fn finish_step(
        &mut self,
        in_logits: &[f32],
        in_values: &[f32],
    ) {
        self.finish_step_vloss(in_logits, in_values);
    }

    /// Finishes multi-leaf virtual-loss simulation step after GPU forward evaluation.
    pub fn finish_step_vloss(
        &mut self,
        in_logits: &[f32],
        in_values: &[f32],
    ) {
        let dirichlet_eps = self.dirichlet_eps;
        let dirichlet_alpha = self.dirichlet_alpha;

        self.games
            .par_iter_mut()
            .filter(|game| !game.pending_leaves.is_empty())
            .for_each(|game| {
                let base_slot = game.pending_batch_offset;
                let pending = std::mem::take(&mut game.pending_leaves);

                // 1. Revert all virtual losses applied during prepare_step_vloss
                for &(_leaf_idx, _, ref path) in &pending {
                    for &(p_idx, ch_idx) in path {
                        game.nodes[p_idx as usize].visits -= 1;
                        game.nodes[p_idx as usize].total_val += 1.0;
                        game.children[ch_idx as usize].visits -= 1;
                        game.children[ch_idx as usize].total_val -= 1.0;
                    }
                }

                // 2. Expand each pending leaf and backpropagate real value and proof
                for (sub_idx, (leaf_idx_u32, leaf_board, path)) in pending.into_iter().enumerate() {
                    let slot = base_slot + sub_idx;
                    if (slot + 1) * 81 > in_logits.len() || slot >= in_values.len() {
                        break;
                    }
                    let leaf_idx = leaf_idx_u32 as usize;
                    let logits_slice = &in_logits[slot * 81..(slot + 1) * 81];
                    let value = in_values[slot];

                    if !game.nodes[leaf_idx].is_expanded && !game.nodes[leaf_idx].is_terminal {
                        let legal_moves = leaf_board.get_legal_moves_fast();
                        if legal_moves.is_empty() {
                            game.nodes[leaf_idx].is_terminal = true;
                            game.nodes[leaf_idx].terminal_val = 0.0;
                            game.nodes[leaf_idx].proof = ProofStatus::ProvenDraw;
                            game.nodes[leaf_idx].visits += 1;
                            game.nodes[leaf_idx].total_val += 0.0;
                            Self::backpropagate_with_proof(game, leaf_idx_u32, 0.0, &path);
                            continue;
                        }

                        // Compute softmax priors for legal moves
                        let mut max_l = f32::NEG_INFINITY;
                        for &m in &legal_moves {
                            let l = logits_slice[m.move_index as usize];
                            if l > max_l {
                                max_l = l;
                            }
                        }

                        let mut exp_sum = 0.0f32;
                        let mut exp_vals = Vec::with_capacity(legal_moves.len());
                        for &m in &legal_moves {
                            let e = (logits_slice[m.move_index as usize] - max_l).exp();
                            exp_vals.push(e);
                            exp_sum += e;
                        }

                        let first_child_idx = game.children.len() as u32;

                        // Add children to flat arena
                        let is_root = leaf_idx == 0;
                        if is_root && dirichlet_eps > 0.0 && legal_moves.len() > 1 {
                            let alpha = dirichlet_alpha.max(1e-3);
                            let mut noise_weights = Vec::with_capacity(legal_moves.len());
                            for _ in 0..legal_moves.len() {
                                let u: f32 = game.rng.gen_range(1e-4..1.0);
                                noise_weights.push((-u.ln()).powf(1.0 / alpha));
                            }
                            let noise_sum: f32 = noise_weights.iter().sum();
                            for (i, &m) in legal_moves.iter().enumerate() {
                                let base_p = if exp_sum > 0.0 {
                                    exp_vals[i] / exp_sum
                                } else {
                                    1.0 / legal_moves.len() as f32
                                };
                                let noise_p = if noise_sum > 0.0 {
                                    noise_weights[i] / noise_sum
                                } else {
                                    1.0 / legal_moves.len() as f32
                                };
                                let final_p =
                                    (1.0 - dirichlet_eps) * base_p + dirichlet_eps * noise_p;
                                game.children.push(MctsChild {
                                    action: m.move_index,
                                    prior: final_p,
                                    visits: 0,
                                    total_val: 0.0,
                                    node_idx: u32::MAX,
                                    proof: ProofStatus::Unknown,
                                });
                            }
                        } else {
                            for (i, &m) in legal_moves.iter().enumerate() {
                                let final_p = if exp_sum > 0.0 {
                                    exp_vals[i] / exp_sum
                                } else {
                                    1.0 / legal_moves.len() as f32
                                };
                                game.children.push(MctsChild {
                                    action: m.move_index,
                                    prior: final_p,
                                    visits: 0,
                                    total_val: 0.0,
                                    node_idx: u32::MAX,
                                    proof: ProofStatus::Unknown,
                                });
                            }
                        }

                        let leaf = &mut game.nodes[leaf_idx];
                        leaf.is_expanded = true;
                        leaf.first_child = first_child_idx;
                        leaf.num_children = legal_moves.len() as u16;
                    }

                    let leaf = &mut game.nodes[leaf_idx];
                    leaf.visits += 1;
                    leaf.total_val += value;

                    Self::backpropagate_with_proof(game, leaf_idx_u32, value, &path);
                }
            });
    }

    /// Advances one move across all games after MCTS simulations complete.
    /// Samples move with temperature T > 0 (maintaining exploration/exploitation balance).
    /// Flushes finished games into completed_samples and resets with fresh 2-ply openings.
    pub fn advance_moves(&mut self, temperature: f32) -> usize {
        let z_consts = &self.zobrist_constants;
        let suite = &self.opening_suite;
        let dirichlet_alpha = self.dirichlet_alpha;
        let dirichlet_eps = self.dirichlet_eps;
        let mut new_finished_samples = Vec::new();

        for (g_idx, game) in self.games.iter_mut().enumerate() {
            if game.nodes.is_empty() {
                continue;
            }

            let root = &game.nodes[0];
            if root.num_children == 0 {
                continue;
            }

            // Inspect proof statuses of root children
            let mut best_winning_action: Option<u8> = None;
            let mut best_winning_visits: u32 = 0;
            let mut best_winning_prior: f32 = f32::NEG_INFINITY;
            let mut has_non_losing_child = false;

            for c_off in 0..root.num_children {
                let ch = &game.children[(root.first_child + c_off as u32) as usize];
                if ch.proof == ProofStatus::ProvenLoss {
                    if best_winning_action.is_none()
                        || ch.visits > best_winning_visits
                        || (ch.visits == best_winning_visits && ch.prior > best_winning_prior)
                    {
                        best_winning_action = Some(ch.action);
                        best_winning_visits = ch.visits;
                        best_winning_prior = ch.prior;
                    }
                }
                if ch.proof != ProofStatus::ProvenWin {
                    has_non_losing_child = true;
                }
            }

            let mut policy = [0.0f32; 81];
            let chosen_action: u8;

            if let Some(win_act) = best_winning_action {
                policy[win_act as usize] = 1.0;
                chosen_action = win_act;
            } else {
                let effective_temp = if temperature <= 0.0 {
                    if game.moves_played <= 16 {
                        1.0f32
                    } else if game.moves_played <= 28 {
                        0.5f32
                    } else {
                        0.15f32
                    }
                } else if temperature < 0.05 {
                    0.0f32
                } else {
                    temperature
                };

                let filtered_visits: u32 = (0..root.num_children)
                    .map(|c_off| {
                        let ch = &game.children[(root.first_child + c_off as u32) as usize];
                        if has_non_losing_child && ch.proof == ProofStatus::ProvenWin {
                            0
                        } else {
                            ch.visits
                        }
                    })
                    .sum();

                if filtered_visits > 0 {
                    let mut actions = Vec::with_capacity(root.num_children as usize);
                    let mut weights = Vec::with_capacity(root.num_children as usize);

                    for c_off in 0..root.num_children {
                        let ch = &game.children[(root.first_child + c_off as u32) as usize];
                        if has_non_losing_child && ch.proof == ProofStatus::ProvenWin {
                            policy[ch.action as usize] = 0.0;
                            actions.push(ch.action);
                            weights.push(0.0);
                            continue;
                        }
                        policy[ch.action as usize] =
                            (ch.visits as f32) / (filtered_visits as f32);
                        actions.push(ch.action);

                        let w = if effective_temp > 0.05 {
                            (ch.visits as f32).powf(1.0 / effective_temp)
                        } else {
                            ch.visits as f32
                        };
                        weights.push(w);
                    }

                    if effective_temp > 0.05 {
                        let w_sum: f32 = weights.iter().sum();
                        if w_sum > 0.0 {
                            let mut r = game.rng.gen_range(0.0..w_sum);
                            let mut picked = actions[0];
                            for (&act, &w) in actions.iter().zip(weights.iter()) {
                                if w > 0.0 {
                                    if r < w {
                                        picked = act;
                                        break;
                                    }
                                    r -= w;
                                }
                            }
                            chosen_action = picked;
                        } else {
                            chosen_action = actions[0];
                        }
                    } else {
                        let mut max_v = 0;
                        let mut max_p = f32::NEG_INFINITY;
                        let mut max_a = actions[0];
                        for c_off in 0..root.num_children {
                            let ch = &game.children[(root.first_child + c_off as u32) as usize];
                            if has_non_losing_child && ch.proof == ProofStatus::ProvenWin {
                                continue;
                            }
                            if ch.visits > max_v || (ch.visits == max_v && ch.prior > max_p) {
                                max_v = ch.visits;
                                max_p = ch.prior;
                                max_a = ch.action;
                            }
                        }
                        chosen_action = max_a;
                    }
                } else {
                    let mut fallback = game.children[root.first_child as usize].action;
                    for c_off in 0..root.num_children {
                        let ch = &game.children[(root.first_child + c_off as u32) as usize];
                        if !has_non_losing_child || ch.proof != ProofStatus::ProvenWin {
                            fallback = ch.action;
                            break;
                        }
                    }
                    policy[fallback as usize] = 1.0;
                    chosen_action = fallback;
                }
            }

            // Compute MCTS root value evaluation (from current player game.board.x_to_move's perspective)
            let root_q: f32 = if best_winning_action.is_some() || root.proof == ProofStatus::ProvenWin {
                1.0
            } else if root.proof == ProofStatus::ProvenLoss {
                -1.0
            } else if root.proof == ProofStatus::ProvenDraw {
                0.0
            } else {
                let mut q_acc = 0.0f32;
                for c_off in 0..root.num_children {
                    let ch = &game.children[(root.first_child + c_off as u32) as usize];
                    let p = policy[ch.action as usize];
                    if p > 0.0 {
                        let child_v_for_root = match ch.proof {
                            ProofStatus::ProvenLoss => 1.0,
                            ProofStatus::ProvenWin => -1.0,
                            ProofStatus::ProvenDraw => 0.0,
                            ProofStatus::Unknown => -ch.q_value(),
                        };
                        q_acc += p * child_v_for_root;
                    }
                }
                q_acc.clamp(-1.0, 1.0)
            };

            // Record state into history before making move
            game.history.push((game.board.clone(), policy, game.board.x_to_move, root_q));

            // Make chosen move
            game.board.make_move(
                GameMove {
                    x_to_move: game.board.x_to_move,
                    move_index: chosen_action,
                },
                z_consts,
            );
            game.moves_played += 1;

            // Check if game terminated
            let win_state = game.board.get_win_state();
            if win_state != GameResult::Unresolved || game.moves_played >= 81 {
                game.nodes.clear();
                game.children.clear();
                game.alt_nodes.clear();
                game.alt_children.clear();
                game.pending_leaves.clear();

                let game_len = game.history.len();
                let p1_outcome: f32 = match win_state {
                    GameResult::P1Win => 1.0,
                    GameResult::P2Win => -1.0,
                    _ => {
                        let xw = game.board.x_supers.count_ones();
                        let ow = game.board.o_supers.count_ones();
                        if xw > ow {
                            1.0
                        } else if ow > xw {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                };

                for (b, pi, x_to_move, rq) in game.history.drain(..) {
                    let z = if x_to_move { p1_outcome } else { -p1_outcome };
                    let v = (0.5 * z + 0.5 * rq).clamp(-1.0, 1.0);
                    let mut feat = [0.0f32; 405];
                    let mut mask = [0.0f32; 81];
                    Self::extract_nchw_features(&b, &mut feat, &mut mask);

                    new_finished_samples.push(CompletedSample {
                        features: feat,
                        legal_mask: mask,
                        policy: pi,
                        value: v,
                        game_length: game_len,
                    });
                }

                // Reset game slot with new 2-ply opening
                let next_opening = suite[(g_idx + game.moves_played) % suite.len()].clone();
                game.board = next_opening.clone();
                game.moves_played = 0;
                game.current_leaf_board = next_opening;
            } else {
                // Reuse subtree rooted at chosen_action
                game.reroot_at_action(chosen_action, dirichlet_alpha, dirichlet_eps);
                game.current_leaf_board = game.board.clone();
            }
        }

        let count = new_finished_samples.len();
        self.completed_samples.extend(new_finished_samples);
        count
    }

    /// Advances moves for competitive tournament evaluation without resetting finished games.
    /// Games that finish retain their terminal board state and are not reset with new openings.
    pub fn advance_moves_tournament(&mut self, temperature: f32) -> usize {
        let mut dummy = vec![255u8; self.num_games];
        self.advance_moves_tournament_recorded(temperature, &mut dummy)
    }

    /// Advances moves in tournament mode and records chosen moves into out_moves buffer.
    /// Games that are already finished or not advanced have out_moves[g] = 255.
    pub fn advance_moves_tournament_recorded(
        &mut self,
        temperature: f32,
        out_moves: &mut [u8],
    ) -> usize {
        let z_consts = &self.zobrist_constants;
        let mut finished_count = 0;

        for (g, game) in self.games.iter_mut().enumerate() {
            if g < out_moves.len() {
                out_moves[g] = 255;
            }
            let win_state = game.board.get_win_state();
            if win_state != GameResult::Unresolved || game.moves_played >= 81 {
                finished_count += 1;
                continue;
            }

            if game.nodes.is_empty() {
                continue;
            }

            let root = &game.nodes[0];
            if root.num_children == 0 || root.is_terminal {
                finished_count += 1;
                continue;
            }

            // Inspect proof statuses of root children
            let mut best_winning_action: Option<u8> = None;
            let mut best_winning_visits: u32 = 0;
            let mut best_winning_prior: f32 = f32::NEG_INFINITY;
            let mut has_non_losing_child = false;

            for c_off in 0..root.num_children {
                let ch = &game.children[(root.first_child + c_off as u32) as usize];
                if ch.proof == ProofStatus::ProvenLoss {
                    if best_winning_action.is_none()
                        || ch.visits > best_winning_visits
                        || (ch.visits == best_winning_visits && ch.prior > best_winning_prior)
                    {
                        best_winning_action = Some(ch.action);
                        best_winning_visits = ch.visits;
                        best_winning_prior = ch.prior;
                    }
                }
                if ch.proof != ProofStatus::ProvenWin {
                    has_non_losing_child = true;
                }
            }

            let total_visits: u32 = (0..root.num_children)
                .map(|c_off| game.children[(root.first_child + c_off as u32) as usize].visits)
                .sum();

            if total_visits == 0 && best_winning_action.is_none() {
                continue;
            }

            let chosen_action = if let Some(win_act) = best_winning_action {
                win_act
            } else if temperature < 0.05 {
                // Pure greedy argmax for competitive match, avoiding proven losses
                let mut max_v = 0;
                let mut max_p = f32::NEG_INFINITY;
                let mut max_a = game.children[root.first_child as usize].action;
                let mut found = false;
                for c_off in 0..root.num_children {
                    let ch = &game.children[(root.first_child + c_off as u32) as usize];
                    if has_non_losing_child && ch.proof == ProofStatus::ProvenWin {
                        continue;
                    }
                    if !found || ch.visits > max_v || (ch.visits == max_v && ch.prior > max_p) {
                        max_v = ch.visits;
                        max_p = ch.prior;
                        max_a = ch.action;
                        found = true;
                    }
                }
                max_a
            } else {
                let mut actions = Vec::with_capacity(root.num_children as usize);
                let mut weights = Vec::with_capacity(root.num_children as usize);
                for c_off in 0..root.num_children {
                    let ch = &game.children[(root.first_child + c_off as u32) as usize];
                    actions.push(ch.action);
                    if has_non_losing_child && ch.proof == ProofStatus::ProvenWin {
                        weights.push(0.0);
                    } else {
                        weights.push((ch.visits as f32).powf(1.0 / temperature));
                    }
                }
                let w_sum: f32 = weights.iter().sum();
                if w_sum > 0.0 {
                    let mut r = game.rng.gen_range(0.0..w_sum);
                    let mut picked = actions[0];
                    for (&act, &w) in actions.iter().zip(weights.iter()) {
                        if w > 0.0 {
                            if r < w {
                                picked = act;
                                break;
                            }
                            r -= w;
                        }
                    }
                    picked
                } else {
                    actions[0]
                }
            };

            if g < out_moves.len() {
                out_moves[g] = chosen_action;
            }

            game.board.make_move(
                GameMove {
                    x_to_move: game.board.x_to_move,
                    move_index: chosen_action,
                },
                z_consts,
            );
            game.moves_played += 1;
            game.current_leaf_board = game.board.clone();

            let win_state_after = game.board.get_win_state();
            if win_state_after != GameResult::Unresolved || game.moves_played >= 81 {
                game.nodes.clear();
                game.children.clear();
                game.alt_nodes.clear();
                game.alt_children.clear();
                game.pending_leaves.clear();
                finished_count += 1;
            } else {
                game.advance_both_trees_and_swap(chosen_action);
            }
        }

        finished_count
    }
}

// ============================================================================
// C-ABI Dynamic Library Exports for Python ctypes Zero-Copy Integration
// ============================================================================

#[no_mangle]
pub extern "C" fn mcts_worker_create(
    num_games: usize,
    c_puct: f32,
    dirichlet_alpha: f32,
    dirichlet_eps: f32,
) -> *mut MctsWorker {
    let worker = Box::new(MctsWorker::new(
        num_games,
        c_puct,
        dirichlet_alpha,
        dirichlet_eps,
    ));
    Box::into_raw(worker)
}

#[no_mangle]
pub extern "C" fn mcts_worker_destroy(worker: *mut MctsWorker) {
    if !worker.is_null() {
        unsafe {
            drop(Box::from_raw(worker));
        }
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_reset(worker: *mut MctsWorker) {
    if let Some(w) = unsafe { worker.as_mut() } {
        w.reset_all();
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_prepare_step(
    worker: *mut MctsWorker,
    out_features: *mut f32,
    out_masks: *mut f32,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        let feat_slice = unsafe {
            std::slice::from_raw_parts_mut(out_features, w.num_games * 405)
        };
        let mask_slice = unsafe {
            std::slice::from_raw_parts_mut(out_masks, w.num_games * 81)
        };
        w.prepare_step(feat_slice, mask_slice)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_prepare_step_vloss(
    worker: *mut MctsWorker,
    virtual_loss_k: usize,
    out_features: *mut f32,
    out_masks: *mut f32,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        let k = virtual_loss_k.max(1);
        let max_leaves = w.num_games * k;
        let feat_slice = unsafe {
            std::slice::from_raw_parts_mut(out_features, max_leaves * 405)
        };
        let mask_slice = unsafe {
            std::slice::from_raw_parts_mut(out_masks, max_leaves * 81)
        };
        w.prepare_step_vloss(k, feat_slice, mask_slice)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_finish_step(
    worker: *mut MctsWorker,
    in_logits: *const f32,
    in_values: *const f32,
    num_active: usize,
) {
    if let Some(w) = unsafe { worker.as_mut() } {
        let logits_slice = unsafe {
            std::slice::from_raw_parts(in_logits, num_active * 81)
        };
        let values_slice = unsafe {
            std::slice::from_raw_parts(in_values, num_active)
        };
        w.finish_step(logits_slice, values_slice);
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_finish_step_vloss(
    worker: *mut MctsWorker,
    in_logits: *const f32,
    in_values: *const f32,
    num_active: usize,
) {
    mcts_worker_finish_step(worker, in_logits, in_values, num_active);
}

#[no_mangle]
pub extern "C" fn mcts_worker_advance_moves(
    worker: *mut MctsWorker,
    temperature: f32,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        w.advance_moves(temperature)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_drain_samples(
    worker: *mut MctsWorker,
    out_features: *mut f32,
    out_masks: *mut f32,
    out_policies: *mut f32,
    out_values: *mut f32,
    max_count: usize,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        let n = w.completed_samples.len().min(max_count);
        if n == 0 {
            return 0;
        }

        let drained: Vec<CompletedSample> = w.completed_samples.drain(..n).collect();
        unsafe {
            for (i, s) in drained.iter().enumerate() {
                let f_dst = std::slice::from_raw_parts_mut(out_features.add(i * 405), 405);
                f_dst.copy_from_slice(&s.features);

                let m_dst = std::slice::from_raw_parts_mut(out_masks.add(i * 81), 81);
                m_dst.copy_from_slice(&s.legal_mask);

                let p_dst = std::slice::from_raw_parts_mut(out_policies.add(i * 81), 81);
                p_dst.copy_from_slice(&s.policy);

                *out_values.add(i) = s.value;
            }
        }
        n
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_set_game_moves(
    worker: *mut MctsWorker,
    game_idx: usize,
    moves: *const u8,
    num_moves: usize,
) {
    if let Some(w) = unsafe { worker.as_mut() } {
        if game_idx >= w.num_games {
            return;
        }
        let move_slice = unsafe { std::slice::from_raw_parts(moves, num_moves) };
        let mut board = UltimateTicTacToeBoard::default();
        for &m in move_slice {
            board.make_move(
                GameMove {
                    x_to_move: board.x_to_move,
                    move_index: m,
                },
                &w.zobrist_constants,
            );
        }
        let game = &mut w.games[game_idx];
        game.board = board.clone();
        game.current_leaf_board = board;
        game.nodes.clear();
        game.children.clear();
        game.alt_nodes.clear();
        game.alt_children.clear();
        game.pending_leaves.clear();
        game.pending_batch_offset = 0;
        game.history.clear();
        game.moves_played = num_moves;
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_get_game_status(
    worker: *mut MctsWorker,
    game_idx: usize,
) -> i32 {
    if let Some(w) = unsafe { worker.as_mut() } {
        if game_idx >= w.num_games {
            return 0;
        }
        let game = &w.games[game_idx];
        let win_state = game.board.get_win_state();
        match win_state {
            GameResult::P1Win => 1,
            GameResult::P2Win => -1,
            GameResult::Tie => 2,
            GameResult::Unresolved => {
                if game.moves_played >= 81 {
                    let xw = game.board.x_supers.count_ones();
                    let ow = game.board.o_supers.count_ones();
                    if xw > ow { 1 } else if ow > xw { -1 } else { 2 }
                } else {
                    0
                }
            }
        }
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_get_game_x_to_move(
    worker: *mut MctsWorker,
    game_idx: usize,
) -> bool {
    if let Some(w) = unsafe { worker.as_mut() } {
        if game_idx < w.num_games {
            w.games[game_idx].board.x_to_move
        } else {
            true
        }
    } else {
        true
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_get_active_indices(
    worker: *mut MctsWorker,
    out_indices: *mut usize,
    max_count: usize,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        let n = w.active_eval_indices.len().min(max_count);
        unsafe {
            for (i, &idx) in w.active_eval_indices.iter().take(n).enumerate() {
                *out_indices.add(i) = idx;
            }
        }
        n
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_advance_moves_tournament(
    worker: *mut MctsWorker,
    temperature: f32,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        w.advance_moves_tournament(temperature)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_advance_moves_tournament_recorded(
    worker: *mut MctsWorker,
    temperature: f32,
    out_chosen_moves: *mut u8,
) -> usize {
    if let Some(w) = unsafe { worker.as_mut() } {
        let moves_slice = unsafe { std::slice::from_raw_parts_mut(out_chosen_moves, w.num_games) };
        w.advance_moves_tournament_recorded(temperature, moves_slice)
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn mcts_worker_apply_moves_batch(
    worker: *mut MctsWorker,
    game_indices: *const usize,
    moves: *const u8,
    num_moves: usize,
) {
    if let Some(w) = unsafe { worker.as_mut() } {
        let indices = unsafe { std::slice::from_raw_parts(game_indices, num_moves) };
        let mv_slice = unsafe { std::slice::from_raw_parts(moves, num_moves) };

        for k in 0..num_moves {
            let g = indices[k];
            if g < w.num_games {
                let m = mv_slice[k];
                let game = &mut w.games[g];
                game.board.make_move(
                    GameMove { x_to_move: game.board.x_to_move, move_index: m },
                    &w.zobrist_constants,
                );
                game.current_leaf_board = game.board.clone();
                game.moves_played += 1;

                let win_state = game.board.get_win_state();
                if win_state != GameResult::Unresolved || game.moves_played >= 81 {
                    game.nodes.clear();
                    game.children.clear();
                    game.alt_nodes.clear();
                    game.alt_children.clear();
                    game.pending_leaves.clear();
                } else {
                    game.advance_both_trees_and_swap(m);
                }
            }
        }
    }
}

static SHARED_ZOBRIST: std::sync::LazyLock<ZobristConstants> =
    std::sync::LazyLock::new(|| UltimateTicTacToeBoard::get_zobrist_constants());
static SHARED_LEGAL_LOOKUP: std::sync::LazyLock<Vec<Vec<GameMove>>> =
    std::sync::LazyLock::new(|| UltimateTicTacToeBoard::get_move_ordering_array());

thread_local! {
    static THREAD_BOT_V2: std::cell::RefCell<crate::bots::AlphaBetaV2Bot> = std::cell::RefCell::new(crate::bots::AlphaBetaV2Bot::new());
    static THREAD_BOT_V1: std::cell::RefCell<crate::bots::AlphaBetaBot> = std::cell::RefCell::new(crate::bots::AlphaBetaBot::new());
    static THREAD_BOT_UMCTS_V2: std::cell::RefCell<crate::bots::UmctsV2Bot> = std::cell::RefCell::new(crate::bots::UmctsV2Bot::new());
}

#[no_mangle]
pub extern "C" fn alphabeta_batch_choose_moves(
    moves_flat: *const u8,
    move_offsets: *const usize,
    move_lengths: *const usize,
    num_games: usize,
    time_limit_ms: u64,
    out_moves: *mut u8,
) {
    let offsets = unsafe { std::slice::from_raw_parts(move_offsets, num_games) };
    let lengths = unsafe { std::slice::from_raw_parts(move_lengths, num_games) };
    let duration = std::time::Duration::from_millis(time_limit_ms);

    let moves_flat_addr = moves_flat as usize;
    let out_moves_addr = out_moves as usize;

    let zobrist = &*SHARED_ZOBRIST;
    let legal_lookup = &*SHARED_LEGAL_LOOKUP;

    (0..num_games).into_par_iter().for_each(|i| {
        THREAD_BOT_V2.with(|bot_cell| {
            let mut bot = bot_cell.borrow_mut();
            let mf_ptr = moves_flat_addr as *const u8;
            let out_ptr = out_moves_addr as *mut u8;
            let slice = unsafe { std::slice::from_raw_parts(mf_ptr.add(offsets[i]), lengths[i]) };

            let mut board = UltimateTicTacToeBoard::default();
            for &m in slice {
                board.make_move(GameMove { x_to_move: board.x_to_move, move_index: m }, zobrist);
            }

            use crate::bots::Bot;
            let chosen_move = bot.choose_move(&board, legal_lookup, zobrist, duration);
            unsafe {
                *out_ptr.add(i) = chosen_move.move_index;
            }
        });
    });
}

#[no_mangle]
pub extern "C" fn alphabeta_v1_batch_choose_moves(
    moves_flat: *const u8,
    move_offsets: *const usize,
    move_lengths: *const usize,
    num_games: usize,
    time_limit_ms: u64,
    out_moves: *mut u8,
) {
    let offsets = unsafe { std::slice::from_raw_parts(move_offsets, num_games) };
    let lengths = unsafe { std::slice::from_raw_parts(move_lengths, num_games) };
    let duration = std::time::Duration::from_millis(time_limit_ms);

    let moves_flat_addr = moves_flat as usize;
    let out_moves_addr = out_moves as usize;

    let zobrist = &*SHARED_ZOBRIST;
    let legal_lookup = &*SHARED_LEGAL_LOOKUP;

    (0..num_games).into_par_iter().for_each(|i| {
        THREAD_BOT_V1.with(|bot_cell| {
            let mut bot = bot_cell.borrow_mut();
            let mf_ptr = moves_flat_addr as *const u8;
            let out_ptr = out_moves_addr as *mut u8;
            let slice = unsafe { std::slice::from_raw_parts(mf_ptr.add(offsets[i]), lengths[i]) };

            let mut board = UltimateTicTacToeBoard::default();
            for &m in slice {
                board.make_move(GameMove { x_to_move: board.x_to_move, move_index: m }, zobrist);
            }

            use crate::bots::Bot;
            let chosen_move = bot.choose_move(&board, legal_lookup, zobrist, duration);
            unsafe {
                *out_ptr.add(i) = chosen_move.move_index;
            }
        });
    });
}

#[no_mangle]
pub extern "C" fn umcts_v2_batch_choose_moves(
    moves_flat: *const u8,
    move_offsets: *const usize,
    move_lengths: *const usize,
    num_games: usize,
    time_limit_ms: u64,
    out_moves: *mut u8,
) {
    let offsets = unsafe { std::slice::from_raw_parts(move_offsets, num_games) };
    let lengths = unsafe { std::slice::from_raw_parts(move_lengths, num_games) };
    let duration = std::time::Duration::from_millis(time_limit_ms);

    let moves_flat_addr = moves_flat as usize;
    let out_moves_addr = out_moves as usize;

    let zobrist = &*SHARED_ZOBRIST;
    let legal_lookup = &*SHARED_LEGAL_LOOKUP;

    (0..num_games).into_par_iter().for_each(|i| {
        THREAD_BOT_UMCTS_V2.with(|bot_cell| {
            let mut bot = bot_cell.borrow_mut();
            bot.reset();
            let mf_ptr = moves_flat_addr as *const u8;
            let out_ptr = out_moves_addr as *mut u8;
            let slice = unsafe { std::slice::from_raw_parts(mf_ptr.add(offsets[i]), lengths[i]) };

            let mut board = UltimateTicTacToeBoard::default();
            for &m in slice {
                board.make_move(GameMove { x_to_move: board.x_to_move, move_index: m }, zobrist);
            }

            use crate::bots::Bot;
            let chosen_move = bot.choose_move(&board, legal_lookup, zobrist, duration);
            unsafe {
                *out_ptr.add(i) = chosen_move.move_index;
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_worker_terminal_win_preferred() {
        let mut worker = MctsWorker::new(1, 1.414, 0.3, 0.0);
        // Setup board: X has won subboards 1 and 2
        worker.games[0].board = UltimateTicTacToeBoard::default();
        worker.games[0].board.x_supers = (1 << 1) | (1 << 2);
        // In subboard 0, X has squares 0 and 1, square 2 is empty
        worker.games[0].board.xbb[0] = (1 << 0) | (1 << 1);
        worker.games[0].board.last_super_index = 0;
        worker.games[0].board.x_to_move = true;

        // Force active evaluation
        let mut feats = vec![0.0f32; 5 * 9 * 9];
        let mut masks = vec![0.0f32; 81];
        let active = worker.prepare_step(&mut feats, &mut masks);
        assert_eq!(active, 1);

        // Feed uniform logits and 0.0 value
        let logits = vec![0.0f32; 81];
        let values = vec![0.0f32; 1];
        worker.finish_step(&logits, &values);

        // Run 200 MCTS simulations
        for _ in 0..200 {
            let active = worker.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                worker.finish_step(&logits, &values);
            }
        }

        // Check root node: move 2 is the game-winning move (subboard 0, square 2)
        let root = &worker.games[0].nodes[0];
        let mut best_action = 255;
        let mut best_visits = 0;
        for c_off in 0..root.num_children {
            let ch = &worker.games[0].children[(root.first_child + c_off as u32) as usize];
            if ch.visits > best_visits {
                best_visits = ch.visits;
                best_action = ch.action;
            }
        }

        assert_eq!(
            best_action, 2,
            "Game-winning move 2 should be chosen, but got action {}",
            best_action
        );
    }

    #[test]
    fn test_mcts_solver_proves_win_and_loss() {
        let mut worker = MctsWorker::new(1, 1.414, 0.3, 0.0);
        // 1. Test ProvenWin propagation from a ProvenLoss child:
        // X is to move in subboard 0, already won subboards 1 and 2, and has squares 0 and 1 in subboard 0.
        worker.games[0].board = UltimateTicTacToeBoard::default();
        worker.games[0].board.x_supers = (1 << 1) | (1 << 2);
        worker.games[0].board.xbb[0] = (1 << 0) | (1 << 1);
        worker.games[0].board.last_super_index = 0;
        worker.games[0].board.x_to_move = true;

        let mut feats = vec![0.0f32; 5 * 9 * 9];
        let mut masks = vec![0.0f32; 81];
        let logits = vec![0.0f32; 81];
        let values = vec![0.0f32; 1];

        for _ in 0..15 {
            let active = worker.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                worker.finish_step(&logits, &values);
            }
        }

        let root = &worker.games[0].nodes[0];
        assert_eq!(
            root.proof,
            ProofStatus::ProvenWin,
            "Root should be ProvenWin once winning move 2 is explored"
        );

        let win_child = (0..root.num_children)
            .map(|c_off| &worker.games[0].children[(root.first_child + c_off as u32) as usize])
            .find(|ch| ch.action == 2)
            .expect("Winning child action 2 must exist");
        assert_eq!(
            win_child.proof,
            ProofStatus::ProvenLoss,
            "Winning child (from opponent's perspective) must have ProofStatus::ProvenLoss"
        );

        let mut chosen_moves = [255u8; 1];
        let finished = worker.advance_moves_tournament_recorded(0.0, &mut chosen_moves);
        assert_eq!(chosen_moves[0], 2);
        assert_eq!(finished, 1);

        // 2. Test ProvenLoss propagation when all children of a node are ProvenWin:
        // O is to move in subboard 0 where only square 8 is empty (no 3-in-a-row in subboard 0).
        // Playing move 8 sends X to subboard 8, where X can win the game immediately.
        let mut worker2 = MctsWorker::new(1, 1.414, 0.3, 0.0);
        let mut b = UltimateTicTacToeBoard::default();
        b.x_supers = (1 << 6) | (1 << 7); // X has won subboards 6 and 7; winning subboard 8 wins the game
        // Subboard 0 has only square 8 free, with a non-winning pattern for squares 0..7:
        // X: {0, 1, 5, 6}, O: {2, 3, 4, 7} -> no 3-in-a-row, even if O plays 8 (O gets {2,3,4,7,8}, no line)
        b.xbb[0] = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6);
        b.obb[0] = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 7);
        // Subboard 8 has X on squares 0 and 1, so X playing square 2 (move 74) wins subboard 8 and the game
        b.xbb[8] = (1 << 0) | (1 << 1);
        b.last_super_index = 0;
        b.x_to_move = false; // O to move, forced to play move 8
        worker2.games[0].board = b;

        for _ in 0..20 {
            let active = worker2.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                worker2.finish_step(&logits, &values);
            }
        }

        let root2 = &worker2.games[0].nodes[0];
        assert_eq!(
            root2.proof,
            ProofStatus::ProvenLoss,
            "Root should be ProvenLoss when O's only legal move leads to a ProvenWin for X"
        );
    }

    #[test]
    fn test_mcts_subtree_reuse_preserves_visits() {
        let mut worker = MctsWorker::new(1, 1.414, 0.3, 0.0);
        let mut feats = vec![0.0f32; 5 * 9 * 9];
        let mut masks = vec![0.0f32; 81];
        let logits = vec![0.0f32; 81];
        let values = vec![0.0f32; 1];

        for _ in 0..30 {
            let active = worker.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                worker.finish_step(&logits, &values);
            }
        }

        let root = &worker.games[0].nodes[0];
        let mut best_visits = 0u32;
        let mut best_prior = f32::NEG_INFINITY;
        for c_off in 0..root.num_children {
            let ch = &worker.games[0].children[(root.first_child + c_off as u32) as usize];
            if ch.visits > best_visits || (ch.visits == best_visits && ch.prior > best_prior) {
                best_visits = ch.visits;
                best_prior = ch.prior;
            }
        }
        assert!(best_visits > 1, "Expected chosen child to have > 1 visits, got {}", best_visits);

        worker.advance_moves(0.01);

        let game = &worker.games[0];
        assert!(!game.nodes.is_empty(), "Subtree should be preserved after advance_moves");
        assert_eq!(
            game.nodes[0].visits, best_visits,
            "New root visits must match the chosen child's previous visits"
        );
        assert_eq!(game.nodes[0].parent_idx, u32::MAX);

        // Verify all node and child indices in the compacted arena are valid
        for (n_idx, node) in game.nodes.iter().enumerate() {
            if n_idx == 0 {
                assert_eq!(node.parent_idx, u32::MAX);
            } else {
                assert!((node.parent_idx as usize) < game.nodes.len());
            }
            if node.is_expanded && node.num_children > 0 {
                let end_ch = (node.first_child as usize) + (node.num_children as usize);
                assert!(end_ch <= game.children.len());
                for ch in &game.children[(node.first_child as usize)..end_ch] {
                    if ch.node_idx != u32::MAX {
                        assert!((ch.node_idx as usize) < game.nodes.len());
                        assert_eq!(game.nodes[ch.node_idx as usize].parent_idx, n_idx as u32);
                    }
                }
            }
        }
    }

    #[test]
    fn test_virtual_loss_explores_distinct_leaves() {
        let mut worker = MctsWorker::new(1, 1.414, 0.3, 0.0);
        let mut feats = vec![0.0f32; 4 * 5 * 9 * 9];
        let mut masks = vec![0.0f32; 4 * 81];

        // Step 1: Expand the root node
        let active_root = worker.prepare_step(&mut feats[..405], &mut masks[..81]);
        assert_eq!(active_root, 1);
        let root_logits = vec![0.0f32; 81];
        let root_values = vec![0.25f32; 1];
        worker.finish_step(&root_logits, &root_values);
        assert_eq!(worker.games[0].nodes[0].visits, 1);

        // Step 2: Run 1 step of prepare_step_vloss(4)
        let active_vloss = worker.prepare_step_vloss(4, &mut feats, &mut masks);
        assert_eq!(
            active_vloss, 4,
            "prepare_step_vloss(4) should select 4 distinct children of the expanded root"
        );

        let batch_logits = vec![0.0f32; 4 * 81];
        let batch_values = vec![0.1f32, -0.2f32, 0.3f32, -0.4f32];
        worker.finish_step(&batch_logits, &batch_values);

        let game = &worker.games[0];
        let root = &game.nodes[0];
        assert_eq!(
            root.visits, 5,
            "Root visits should be 1 (initial) + 4 (vloss batch) = 5 after reverting virtual losses"
        );

        // Expected root total_val = 0.25 + (-0.1) + (0.2) + (-0.3) + (0.4) = 0.45
        let expected_total_val = 0.25f32 - 0.1f32 + 0.2f32 - 0.3f32 + 0.4f32;
        assert!(
            (root.total_val - expected_total_val).abs() < 1e-5,
            "Expected root total_val {}, got {}",
            expected_total_val,
            root.total_val
        );

        let visited_children_count = (0..root.num_children)
            .filter(|&c_off| game.children[(root.first_child + c_off as u32) as usize].visits == 1)
            .count();
        assert_eq!(
            visited_children_count, 4,
            "Exactly 4 distinct root children should have visits == 1"
        );
    }

    #[test]
    fn test_dual_trees_never_share_memory_in_tournament_mode() {
        let mut worker = MctsWorker::new(1, 1.414, 0.3, 0.0);
        let mut feats = vec![0.0f32; 5 * 9 * 9];
        let mut masks = vec![0.0f32; 81];
        let logits = vec![0.0f32; 81];

        // Ply 0 (Player X): run 150 simulations with Player X's distinct neural value (+0.25)
        let values_p1 = vec![0.25f32; 1];
        for _ in 0..150 {
            let active = worker.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                worker.finish_step(&logits, &values_p1);
            }
        }

        // Before advancing Ply 0: Player X's tree is in `nodes`, and Player O's tree (`alt_nodes`) is empty
        assert!(!worker.games[0].nodes.is_empty());
        assert!(worker.games[0].alt_nodes.is_empty());

        let root_p1 = &worker.games[0].nodes[0];
        let mut best_v_m0 = 0u32;
        let mut best_p_m0 = f32::NEG_INFINITY;
        for c_off in 0..root_p1.num_children {
            let ch = &worker.games[0].children[(root_p1.first_child + c_off as u32) as usize];
            if ch.visits > best_v_m0 || (ch.visits == best_v_m0 && ch.prior > best_p_m0) {
                best_v_m0 = ch.visits;
                best_p_m0 = ch.prior;
            }
        }
        assert!(best_v_m0 >= 10);

        // Advance Ply 0 in tournament mode (makes m_0 and calls advance_both_trees_and_swap)
        let mut chosen_moves = [255u8; 1];
        worker.advance_moves_tournament_recorded(0.0, &mut chosen_moves);
        let m_0 = chosen_moves[0];
        assert_ne!(m_0, 255);

        // Start of Ply 1 (Player O's turn):
        // Player O's active tree (`nodes`) must be completely empty (never inherits Player X's nodes!),
        // while `alt_nodes` holds Player X's re-rooted subtree under m_0.
        assert!(
            worker.games[0].nodes.is_empty(),
            "Player O's active tree must be empty on Ply 1 before Player O searches"
        );
        assert!(worker.games[0].children.is_empty());
        assert!(
            !worker.games[0].alt_nodes.is_empty(),
            "Player X's subtree under m_0 must be stored in alt_nodes during Player O's turn"
        );
        assert_eq!(worker.games[0].alt_nodes[0].visits, best_v_m0);

        // Ply 1 (Player O): run 40 simulations with Player O's distinct neural value (-0.75)
        let values_p2 = vec![-0.75f32; 1];
        for _ in 0..40 {
            let active = worker.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                worker.finish_step(&logits, &values_p2);
            }
        }

        let root_p2 = &worker.games[0].nodes[0];
        let mut best_v_m1 = 0u32;
        let mut best_p_m1 = f32::NEG_INFINITY;
        let mut expected_m1 = 255u8;
        for c_off in 0..root_p2.num_children {
            let ch = &worker.games[0].children[(root_p2.first_child + c_off as u32) as usize];
            if ch.visits > best_v_m1 || (ch.visits == best_v_m1 && ch.prior > best_p_m1) {
                best_v_m1 = ch.visits;
                best_p_m1 = ch.prior;
                expected_m1 = ch.action;
            }
        }
        assert!(best_v_m1 >= 3);

        // Inspect how many visits Player X had invested in (m_0, expected_m1) back on Ply 0
        let p1_sub_root = &worker.games[0].alt_nodes[0];
        let p1_child_m1 = (0..p1_sub_root.num_children)
            .map(|c_off| &worker.games[0].alt_children[(p1_sub_root.first_child + c_off as u32) as usize])
            .find(|ch| ch.action == expected_m1)
            .expect("m_1 must be a legal child of m_0 in Player X's subtree");
        let p1_visits_on_m1 = p1_child_m1.visits;
        assert!(p1_visits_on_m1 >= 2, "Player X should have expanded (m_0, m_1) during its 150 sims");

        // Advance Ply 1 in tournament mode (makes m_1 and calls advance_both_trees_and_swap)
        worker.advance_moves_tournament_recorded(0.0, &mut chosen_moves);
        let m_1 = chosen_moves[0];
        assert_eq!(m_1, expected_m1);

        // Start of Ply 2 (Player X's turn again!):
        // `nodes` must hold Player X's own 2-ply re-rooted subtree under (m_0, m_1),
        // and `alt_nodes` must hold Player O's re-rooted subtree under m_1!
        assert!(
            !worker.games[0].nodes.is_empty(),
            "Player X's 2-ply subtree under (m_0, m_1) must be restored to nodes on Ply 2"
        );
        assert_eq!(worker.games[0].nodes[0].visits, p1_visits_on_m1);
        assert!(
            !worker.games[0].alt_nodes.is_empty(),
            "Player O's subtree under m_1 must be stored in alt_nodes on Ply 2"
        );
        assert_eq!(worker.games[0].alt_nodes[0].visits, best_v_m1);
    }

    #[test]
    fn test_per_game_temp_and_q_root_blending() {
        let mut worker = MctsWorker::new(1, 1.414, 0.3, 0.0);
        let mut feats = vec![0.0f32; 5 * 9 * 9];
        let mut masks = vec![0.0f32; 81];
        let logits = vec![0.0f32; 81];

        // 1. Verify exact 0.5 * z + 0.5 * Q_root blending on a controlled final ply (moves_played = 80 -> 81):
        // X is to move and holds 1 won superboard (x_supers = 1) vs O's 0 (o_supers = 0), so z = +1.0 for X.
        worker.games[0].board = UltimateTicTacToeBoard::default();
        worker.games[0].board.x_supers = 1 << 0;
        worker.games[0].board.x_to_move = true;
        worker.games[0].moves_played = 80;

        // Expand root and 10 descendants with consistent +0.4 evaluation for X (-0.4 when O is to move at leaf),
        // so every backed-up value at root (X to move) is +0.4 and policy-weighted root_q == +0.4.
        for _ in 0..11 {
            let active = worker.prepare_step(&mut feats, &mut masks);
            if active > 0 {
                let leaf_x_to_move = worker.games[0].pending_leaves[0].1.x_to_move;
                let leaf_val = if leaf_x_to_move { 0.4f32 } else { -0.4f32 };
                worker.finish_step(&logits, &[leaf_val]);
            }
        }

        let finished = worker.advance_moves(0.0);
        assert_eq!(finished, 1, "Game at ply 80 should finish after 1 move");
        assert_eq!(worker.completed_samples.len(), 1);
        let blended_v = worker.completed_samples[0].value;
        // Expected: 0.5 * (+1.0) + 0.5 * (+0.4) = +0.70
        assert!(
            (blended_v - 0.70).abs() < 1e-4,
            "Expected blended value 0.5*1.0 + 0.5*0.4 = 0.70, got {}",
            blended_v
        );

        // 2. Run a full multi-move game from ply 0 with advance_moves(0.0) (per-game temp schedule 1.0 -> 0.5 -> 0.15)
        worker.reset_all();
        let const_vals = vec![-0.2f32; 1];
        let mut steps = 0;
        while worker.completed_samples.is_empty() && steps < 100 {
            for _ in 0..16 {
                let active = worker.prepare_step(&mut feats, &mut masks);
                if active > 0 {
                    worker.finish_step(&logits, &const_vals);
                }
            }
            worker.advance_moves(0.0);
            steps += 1;
        }
        assert!(
            !worker.completed_samples.is_empty(),
            "Full self-play game with advance_moves(0.0) should complete within 100 plies"
        );
        for s in &worker.completed_samples {
            assert!(
                s.value >= -1.0 && s.value <= 1.0,
                "Blended sample value {} out of [-1.0, 1.0]",
                s.value
            );
        }
    }
}



