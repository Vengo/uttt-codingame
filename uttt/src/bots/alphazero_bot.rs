//! AlphaZero Bot powered by pure Rust INT8 neural network inference.
//!
//! Evaluates positions using a dual-headed ResNet trained via self-play:
//! - Policy head prior P(s, a)
//! - Value head evaluation V(s) in [-1.0, 1.0]
//! - 1-ply Negamax / PUCT search evaluating candidate positions.

use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::nn::{InferenceWorkspace, ModelInt8};
use crate::packed_state::PackedBoardState;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
struct PuctChild {
    m: GameMove,
    prior: f32,
    visits: u32,
    total_val: f32,
}

#[derive(Clone, Debug)]
struct PuctNode {
    children: Vec<PuctChild>,
    visits: u32,
    is_terminal: bool,
    terminal_val: f32,
}

pub struct AlphaZeroBot {
    name: String,
    model: ModelInt8,
    workspace: InferenceWorkspace,
}

impl AlphaZeroBot {
    pub fn new() -> Self {
        AlphaZeroBot {
            name: "AlphaZeroBot".to_string(),
            model: ModelInt8::default_model(),
            workspace: InferenceWorkspace::new(),
        }
    }

    pub fn with_model(model: ModelInt8) -> Self {
        AlphaZeroBot {
            name: "AlphaZeroBot".to_string(),
            model,
            workspace: InferenceWorkspace::new(),
        }
    }
}

impl Default for AlphaZeroBot {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_softmax_priors(moves: &[GameMove], logits: &[f32; 81]) -> Vec<f32> {
    if moves.is_empty() {
        return Vec::new();
    }
    let mut max_l = f32::NEG_INFINITY;
    for &m in moves {
        let l = logits[m.move_index as usize];
        if l > max_l {
            max_l = l;
        }
    }
    let mut sum = 0.0f32;
    let mut exp_vals = Vec::with_capacity(moves.len());
    for &m in moves {
        let ep = (logits[m.move_index as usize] - max_l).exp();
        exp_vals.push(ep);
        sum += ep;
    }
    if sum > 0.0 {
        exp_vals.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / (moves.len() as f32); moves.len()]
    }
}

impl Bot for AlphaZeroBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn reset(&mut self) {}

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        let start = Instant::now();
        let root_moves = if !legal_moves_lookup.is_empty() {
            board.get_legal_moves(legal_moves_lookup)
        } else {
            board.get_legal_moves_fast()
        };

        if root_moves.is_empty() {
            return GameMove::default();
        }
        if root_moves.len() == 1 {
            return root_moves[0];
        }

        // 1. Tactical Immediate Win Check:
        for &m in &root_moves {
            let mut clone = board.clone();
            clone.make_move(m, zobrist_constants);
            let res = clone.get_win_state();
            if (res == GameResult::P1Win && board.x_to_move)
                || (res == GameResult::P2Win && !board.x_to_move)
            {
                return m;
            }
        }

        // 2. Tactical 1-ply Defense: Exclude moves that immediately give the opponent a game win
        let is_move_safe = |m: GameMove| -> bool {
            let mut clone = board.clone();
            clone.make_move(m, zobrist_constants);
            let win = clone.get_win_state();
            if win != GameResult::Unresolved {
                return true; // We won or tied
            }
            let opp_moves = clone.get_legal_moves_fast();
            for &om in &opp_moves {
                let mut opp_clone = clone.clone();
                opp_clone.make_move(om, zobrist_constants);
                let opp_res = opp_clone.get_win_state();
                if (opp_res == GameResult::P1Win && clone.x_to_move)
                    || (opp_res == GameResult::P2Win && !clone.x_to_move)
                {
                    return false;
                }
            }
            true
        };

        let safe_moves: Vec<GameMove> = root_moves
            .iter()
            .copied()
            .filter(|&m| is_move_safe(m))
            .collect();
        let candidate_moves = if !safe_moves.is_empty() {
            safe_moves
        } else {
            root_moves.clone()
        };

        if candidate_moves.len() == 1 {
            return candidate_moves[0];
        }

        // 3. Tactical prior modulation and PUCT search
        let complete_bbs = board.x_supers | board.o_supers | board.get_tied_mask();
        let (my_bb, opp_bb) = if board.x_to_move {
            (board.xbb, board.obb)
        } else {
            (board.obb, board.xbb)
        };

        const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

        let tactical_weight = |m: GameMove| -> f32 {
            let parsed = crate::uttt_trait_solving::parse_index(m.move_index);
            let sb = parsed.sub_board_index;
            let sq = parsed.sub_square_index;
            let mut factor = 1.0f32;

            // Subboard capture bonus
            let sq_mask = 1u16 << sq;
            let my_new = my_bb[sb] | sq_mask;
            for &mask in &RESULT_MASKS {
                if (my_new & mask) == mask {
                    factor *= 3.5;
                    break;
                }
            }

            // Block opponent threat in current subboard
            for &mask in &RESULT_MASKS {
                if (mask & sq_mask) != 0 && (opp_bb[sb] & mask).count_ones() == 2 {
                    factor *= 2.2;
                    break;
                }
            }

            // Free choice penalty: sending opponent to already completed subboard
            let target_complete = (complete_bbs & (1 << sq)) != 0;
            if target_complete {
                factor *= 0.35;
            } else {
                // Check if opponent can immediately win target subboard
                let opp_target_bb = opp_bb[sq];
                let my_target_bb = my_bb[sq];
                let occ = opp_target_bb | my_target_bb;
                let mut opp_can_win = false;
                for &mask in &RESULT_MASKS {
                    let matching = opp_target_bb & mask;
                    if matching.count_ones() == 2 {
                        let missing = mask ^ matching;
                        if (occ & missing) == 0 {
                            opp_can_win = true;
                            break;
                        }
                    }
                }
                if opp_can_win {
                    factor *= 0.25;
                }
            }

            factor
        };

        let mut tree: HashMap<PackedBoardState, PuctNode> = HashMap::with_capacity(256);

        // Root node expansion
        let (root_logits, _root_val) = self.model.evaluate(board, &mut self.workspace);
        let raw_root_priors = compute_softmax_priors(&root_moves, &root_logits);

        // Apply tactical weights to priors
        let mut weighted_priors: Vec<f32> = root_moves
            .iter()
            .zip(raw_root_priors.iter())
            .map(|(&m, &prior)| {
                let tw = tactical_weight(m);
                let base = prior * tw;
                if candidate_moves.contains(&m) {
                    base
                } else {
                    base * 0.001
                }
            })
            .collect();
        let p_sum: f32 = weighted_priors.iter().sum();
        if p_sum > 0.0 {
            for p in &mut weighted_priors {
                *p /= p_sum;
            }
        }

        let root_children: Vec<PuctChild> = root_moves
            .iter()
            .zip(weighted_priors.iter())
            .map(|(&m, &prior)| PuctChild {
                m,
                prior,
                visits: 0,
                total_val: 0.0,
            })
            .collect();

        let root_packed = PackedBoardState::from_board(board);
        tree.insert(
            root_packed,
            PuctNode {
                children: root_children,
                visits: 0,
                is_terminal: false,
                terminal_val: 0.0,
            },
        );

        let budget = time_limit.saturating_sub(Duration::from_millis(8));
        let c_puct = 1.414f32;

        while start.elapsed() < budget {
            let mut curr_board = board.clone();
            let mut path: Vec<(PackedBoardState, usize)> = Vec::with_capacity(32);
            let mut backup_val: f32;

            loop {
                let curr_packed = PackedBoardState::from_board(&curr_board);
                let node_opt = tree.get(&curr_packed);

                match node_opt {
                    Some(node) => {
                        if node.is_terminal {
                            backup_val = node.terminal_val;
                            break;
                        }
                        if node.children.is_empty() {
                            backup_val = 0.0;
                            break;
                        }

                        // Select child maximizing PUCT: Q(s, a) + c_puct * P(s, a) * sqrt(N + 1) / (1 + n)
                        let sqrt_parent = ((node.visits as f32) + 1.0).sqrt();
                        let mut best_score = f32::NEG_INFINITY;
                        let mut best_child_idx = 0;

                        for (idx, ch) in node.children.iter().enumerate() {
                            let q = if ch.visits > 0 {
                                ch.total_val / (ch.visits as f32)
                            } else {
                                0.0
                            };
                            let u = c_puct * ch.prior * sqrt_parent / (1.0 + ch.visits as f32);
                            let score = q + u;
                            if score > best_score {
                                best_score = score;
                                best_child_idx = idx;
                            }
                        }

                        let chosen_move = node.children[best_child_idx].m;
                        path.push((curr_packed, best_child_idx));
                        curr_board.make_move(chosen_move, zobrist_constants);
                    }
                    None => {
                        // Leaf reached: check terminal state first
                        let win_state = curr_board.get_win_state();
                        if win_state != GameResult::Unresolved {
                            // In UTTT, whoever is to move at a decided terminal state has lost (-1.0).
                            let t_val = match win_state {
                                GameResult::P1Win | GameResult::P2Win => -1.0,
                                GameResult::Tie => 0.0,
                                GameResult::Unresolved => 0.0,
                            };
                            tree.insert(
                                curr_packed,
                                PuctNode {
                                    children: Vec::new(),
                                    visits: 0,
                                    is_terminal: true,
                                    terminal_val: t_val,
                                },
                            );
                            backup_val = t_val;
                            break;
                        }

                        // Evaluate leaf with NN
                        let (leaf_logits, leaf_val) = self.model.evaluate(&curr_board, &mut self.workspace);
                        let leaf_moves = if !legal_moves_lookup.is_empty() {
                            curr_board.get_legal_moves(legal_moves_lookup)
                        } else {
                            curr_board.get_legal_moves_fast()
                        };

                        let leaf_priors = compute_softmax_priors(&leaf_moves, &leaf_logits);
                        let leaf_children: Vec<PuctChild> = leaf_moves
                            .iter()
                            .zip(leaf_priors.iter())
                            .map(|(&m, &prior)| PuctChild {
                                m,
                                prior,
                                visits: 0,
                                total_val: 0.0,
                            })
                            .collect();

                        tree.insert(
                            curr_packed,
                            PuctNode {
                                children: leaf_children,
                                visits: 0,
                                is_terminal: false,
                                terminal_val: 0.0,
                            },
                        );
                        backup_val = leaf_val;
                        break;
                    }
                }
            }

            // Backpropagate up the path (Negamax alternation)
            for &(parent_packed, child_idx) in path.iter().rev() {
                backup_val = -backup_val;
                if let Some(parent_node) = tree.get_mut(&parent_packed) {
                    parent_node.visits += 1;
                    parent_node.children[child_idx].visits += 1;
                    parent_node.children[child_idx].total_val += backup_val;
                }
            }
        }

        // Select move from root: highest visits among candidate moves, fallback to highest prior
        if let Some(root_node) = tree.get(&root_packed) {
            let safe_children: Vec<&PuctChild> = root_node
                .children
                .iter()
                .filter(|c| candidate_moves.contains(&c.m))
                .collect();
            if let Some(best_child) = safe_children.iter().filter(|c| c.visits > 0).max_by_key(|c| c.visits) {
                return best_child.m;
            }
            if let Some(best_by_prior) = safe_children.iter().max_by(|a, b| a.prior.partial_cmp(&b.prior).unwrap_or(std::cmp::Ordering::Equal)) {
                return best_by_prior.m;
            }
            if let Some(best_child) = root_node.children.iter().max_by_key(|c| c.visits) {
                if best_child.visits > 0 {
                    return best_child.m;
                }
            }
        }

        candidate_moves[0]
    }
}
