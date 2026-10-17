use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::uttt_trait_solving::{parse_index, UltimateTicTacToeBoard, ZobristConstants};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const EXPLORATION_CONSTANT: f64 = 1.41421356;
const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

struct HeuristicMctsNode {
    parent: Option<usize>,
    children: Vec<usize>,
    action: GameMove,
    visits: u32,
    wins: f64,
    unexpanded_moves: Vec<GameMove>,
}

pub struct HeuristicMctsBot {
    name: String,
}

impl HeuristicMctsBot {
    pub fn new() -> Self {
        HeuristicMctsBot {
            name: "HeuristicMctsBot".to_string(),
        }
    }

    fn uct_select(node_idx: usize, nodes: &[HeuristicMctsNode]) -> usize {
        let parent_visits = (nodes[node_idx].visits as f64).ln();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = nodes[node_idx].children[0];

        for &child_idx in &nodes[node_idx].children {
            let child = &nodes[child_idx];
            let exploit = child.wins / (child.visits as f64);
            let explore = EXPLORATION_CONSTANT * (parent_visits / (child.visits as f64)).sqrt();
            let score = exploit + explore;

            if score > best_score {
                best_score = score;
                best_child = child_idx;
            }
        }

        best_child
    }

    fn score_move_quick(board: &UltimateTicTacToeBoard, m: GameMove) -> i32 {
        let is_x = m.x_to_move;
        let parsed = parse_index(m.move_index);
        let sub = parsed.sub_board_index;
        let sq = parsed.sub_square_index;

        let my_bb = if is_x { board.xbb[sub] } else { board.obb[sub] };
        let opp_bb = if is_x { board.obb[sub] } else { board.xbb[sub] };
        let new_my_bb = my_bb | (1 << sq);

        // Subboard win
        if RESULT_MASKS.iter().any(|mask| (new_my_bb & mask) == *mask) {
            return 100;
        }

        // Subboard block
        let sq_bit = 1u16 << sq;
        for mask in RESULT_MASKS.iter() {
            if (mask & sq_bit) != 0 && (opp_bb & mask).count_ones() == 2 {
                return 60;
            }
        }

        // Avoid sending to completed board (free move)
        let complete = board.x_supers | board.o_supers | board.get_tied_mask();
        if (complete & (1 << sq)) != 0 {
            return -20;
        }

        // Center square bonus
        if sq == 4 {
            return 15;
        }

        0
    }

    fn simulate_heuristic(
        mut board: UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
        rng: &mut SmallRng,
    ) -> GameResult {
        let mut depth = 0;
        while board.get_win_state() == GameResult::Unresolved && depth < 50 {
            let moves = board.get_legal_moves_fast();
            if moves.is_empty() {
                break;
            }

            // 70% chance to pick best heuristic move, 30% random
            let chosen = if rng.gen_bool(0.7) {
                let mut best_m = moves[0];
                let mut best_s = i32::MIN;
                for &m in &moves {
                    let s = Self::score_move_quick(&board, m);
                    if s > best_s {
                        best_s = s;
                        best_m = m;
                    }
                }
                best_m
            } else {
                *moves.choose(rng).unwrap()
            };

            board.make_move(chosen, zobrist_constants);
            depth += 1;
        }
        board.get_win_state()
    }
}

impl Default for HeuristicMctsBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for HeuristicMctsBot {
    fn name(&self) -> &str {
        &self.name
    }

    fn choose_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        _legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
        time_limit: Duration,
    ) -> GameMove {
        let mut root_moves = board.get_legal_moves_fast();
        if root_moves.len() <= 1 {
            return root_moves.first().copied().unwrap_or_default();
        }

        // Order unexpanded moves by quick heuristic
        root_moves.sort_by_cached_key(|&m| Self::score_move_quick(board, m));

        let mut nodes: Vec<HeuristicMctsNode> = Vec::with_capacity(100_000);
        nodes.push(HeuristicMctsNode {
            parent: None,
            children: Vec::new(),
            action: GameMove::default(),
            visits: 0,
            wins: 0.0,
            unexpanded_moves: root_moves,
        });

        let mut rng = SmallRng::from_entropy();
        let deadline = Instant::now() + time_limit;

        while Instant::now() < deadline {
            let mut curr_idx = 0;
            let mut sim_board = board.clone();

            // 1. Selection
            while nodes[curr_idx].unexpanded_moves.is_empty() && !nodes[curr_idx].children.is_empty() {
                curr_idx = Self::uct_select(curr_idx, &nodes);
                sim_board.make_move(nodes[curr_idx].action, zobrist_constants);
            }

            // 2. Expansion
            if !nodes[curr_idx].unexpanded_moves.is_empty() {
                let m = nodes[curr_idx].unexpanded_moves.pop().unwrap();
                sim_board.make_move(m, zobrist_constants);

                let mut next_moves = if sim_board.get_win_state() == GameResult::Unresolved {
                    sim_board.get_legal_moves_fast()
                } else {
                    Vec::new()
                };

                // Order candidate moves by quick score so best moves are expanded first
                next_moves.sort_by_cached_key(|&nm| Self::score_move_quick(&sim_board, nm));

                let new_node_idx = nodes.len();
                nodes.push(HeuristicMctsNode {
                    parent: Some(curr_idx),
                    children: Vec::new(),
                    action: m,
                    visits: 0,
                    wins: 0.0,
                    unexpanded_moves: next_moves,
                });
                nodes[curr_idx].children.push(new_node_idx);
                curr_idx = new_node_idx;
            }

            // 3. Simulation with heuristic policy
            let result = Self::simulate_heuristic(sim_board, zobrist_constants, &mut rng);

            // 4. Backpropagation
            let mut node_opt = Some(curr_idx);
            while let Some(idx) = node_opt {
                nodes[idx].visits += 1;
                if let Some(_parent_idx) = nodes[idx].parent {
                    let player_just_moved = nodes[idx].action.x_to_move;
                    let reward = match result {
                        GameResult::P1Win => if player_just_moved { 1.0 } else { 0.0 },
                        GameResult::P2Win => if !player_just_moved { 1.0 } else { 0.0 },
                        GameResult::Tie => 0.5,
                        GameResult::Unresolved => 0.5,
                    };
                    nodes[idx].wins += reward;
                }
                node_opt = nodes[idx].parent;
            }
        }

        let best_child = nodes[0]
            .children
            .iter()
            .max_by_key(|&&idx| nodes[idx].visits)
            .copied()
            .unwrap_or(0);

        if best_child != 0 {
            nodes[best_child].action
        } else {
            *board.get_legal_moves_fast().first().unwrap()
        }
    }
}
