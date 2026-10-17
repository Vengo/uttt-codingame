use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult, IdentityHasherBuilder};
use crate::uttt_trait_solving::{
    UltimateTicTacToeBoard, UltimateTicTacToeUnmove, ZobristConstants,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::time::{Duration, Instant};

const EXPLORATION_CONSTANT: f64 = 1.41421356;
const DEFAULT_TABLE_CAPACITY: usize = 131_072; // 128K entries (~4MB), stays resident in L3 cache

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProofStatus {
    Unproven,
    ProvenWin,  // Proven win for the player whose turn it is at this node
    ProvenLoss, // Proven loss for the player whose turn it is at this node
    ProvenDraw, // Proven draw
}

#[derive(Clone, Debug)]
pub struct UmctsV2Node {
    pub visits: u32,
    pub wins: f64, // Wins from perspective of player who moved INTO this node
    pub proof: ProofStatus,
    pub unexpanded_moves: Vec<GameMove>,
    pub children: Vec<(GameMove, u64)>, // (Action taken, child node hash)
}

impl UmctsV2Node {
    pub fn new(mut legal_moves: Vec<GameMove>) -> Self {
        // Prioritize center board and center squares so pop() extracts high-leverage tactical cells first
        legal_moves.sort_by_key(|m| {
            let parsed = crate::uttt_trait_solving::parse_index(m.move_index);
            let mut score: i32 = 0;
            if parsed.sub_board_index == 4 {
                score += 50;
            }
            if parsed.sub_square_index == 4 {
                score += 40;
            } else if [0, 2, 6, 8].contains(&parsed.sub_square_index) {
                score += 20;
            }
            score
        });

        UmctsV2Node {
            visits: 0,
            wins: 0.0,
            proof: ProofStatus::Unproven,
            unexpanded_moves: legal_moves,
            children: Vec::new(),
        }
    }
}

pub struct UmctsV2Bot {
    name: String,
    node_store: HashMap<u64, UmctsV2Node, IdentityHasherBuilder>,
    unmove_queue: Vec<(u64, UltimateTicTacToeUnmove, GameMove)>,
    rng: SmallRng,
}

impl UmctsV2Bot {
    pub fn new() -> Self {
        UmctsV2Bot {
            name: "UmctsV2Bot".to_string(),
            node_store: HashMap::with_capacity_and_hasher(
                DEFAULT_TABLE_CAPACITY,
                IdentityHasherBuilder {},
            ),
            unmove_queue: Vec::with_capacity(81),
            rng: SmallRng::from_entropy(),
        }
    }

    pub fn node_store(&self) -> &HashMap<u64, UmctsV2Node, IdentityHasherBuilder> {
        &self.node_store
    }

    /// Extracts all positions and their best moves from the search tree that received >= min_visits.
    pub fn harvest_tree(
        &self,
        root_board: &UltimateTicTacToeBoard,
        min_visits: u32,
        zobrist_constants: &ZobristConstants,
    ) -> Vec<(UltimateTicTacToeBoard, GameMove, u32)> {
        let mut results = Vec::new();
        let root_hash = compute_full_hash(root_board);

        let mut queue = std::collections::VecDeque::new();
        let mut visited_hashes = std::collections::HashSet::new();

        queue.push_back((root_board.clone(), root_hash, 0usize));
        visited_hashes.insert(root_hash);

        while let Some((curr_board, curr_hash, depth)) = queue.pop_front() {
            let curr_node = match self.node_store.get(&curr_hash) {
                Some(n) => n,
                None => continue,
            };

            if curr_node.visits < min_visits {
                continue;
            }

            // Find best child action (highest visit count) and second best for dominance check
            let mut best_action = None;
            let mut best_child_visits = 0;
            let mut second_child_visits = 0;

            for &(action, child_hash) in &curr_node.children {
                let child_visits = self
                    .node_store
                    .get(&child_hash)
                    .map(|n| n.visits)
                    .unwrap_or(0);
                if child_visits > best_child_visits {
                    second_child_visits = best_child_visits;
                    best_child_visits = child_visits;
                    best_action = Some(action);
                } else if child_visits > second_child_visits {
                    second_child_visits = child_visits;
                }
            }

            // Principled UCT dominance condition:
            // Top move must have a clear margin over the runner-up (or capture a dominant plurality of visits)
            let is_dominant = best_child_visits >= (second_child_visits as f64 * 1.35) as u32
                || (curr_node.visits > 0 && (best_child_visits as f64 / curr_node.visits as f64) >= 0.40);

            if let Some(act) = best_action {
                if is_dominant && best_child_visits >= min_visits.max(10) / 2 {
                    results.push((curr_board.clone(), act, curr_node.visits));
                }
            }

            // Recurse along Principal Variation and top alternative counter-reply
            if depth < 8 {
                let mut sorted_children = curr_node.children.clone();
                sorted_children.sort_by_key(|&(_, h)| {
                    std::cmp::Reverse(self.node_store.get(&h).map(|n| n.visits).unwrap_or(0))
                });

                for &(action, child_hash) in sorted_children.iter().take(2) {
                    let child_visits = self
                        .node_store
                        .get(&child_hash)
                        .map(|n| n.visits)
                        .unwrap_or(0);

                    if child_visits >= min_visits && visited_hashes.insert(child_hash) {
                        let mut next_board = curr_board.clone();
                        next_board.make_move(action, zobrist_constants);
                        queue.push_back((next_board, child_hash, depth + 1));
                    }
                }
            }

        }

        results
    }


    /// Fast simulation to end of game without tree allocations.
    /// Operates directly on board state in-place or via clone.
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
            // Fast random move selection
            let chosen = *moves.choose(rng).unwrap();
            board.make_move(chosen, zobrist_constants);
            depth += 1;
        }
        board.get_win_state()
    }

    /// Selects best child among expanded children using UCT + Proof filtering (Single pass)
    fn select_child(&self, node_hash: u64) -> Option<(GameMove, u64)> {
        let node = self.node_store.get(&node_hash)?;
        if node.children.is_empty() {
            return None;
        }

        let parent_ln = (node.visits.max(1) as f64).ln();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_candidate = None;

        for &(action, child_hash) in &node.children {
            if let Some(child_node) = self.node_store.get(&child_hash) {
                // 1. Instant win shortcut!
                if child_node.proof == ProofStatus::ProvenLoss {
                    return Some((action, child_hash));
                }
                // 2. Prune proven loss for us
                if child_node.proof == ProofStatus::ProvenWin {
                    continue;
                }

                let score = if child_node.visits == 0 {
                    10_000.0 // Prioritize unvisited child
                } else {
                    let exploit = child_node.wins / (child_node.visits as f64);
                    let explore =
                        EXPLORATION_CONSTANT * (parent_ln / (child_node.visits as f64)).sqrt();
                    exploit + explore
                };

                if score > best_score {
                    best_score = score;
                    best_candidate = Some((action, child_hash));
                }
            }
        }

        best_candidate.or_else(|| node.children.first().copied())
    }

    /// Fast proof propagation: only triggers full sibling scan when a child is proven
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

impl Default for UmctsV2Bot {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn compute_full_hash(board: &UltimateTicTacToeBoard) -> u64 {
    board.hash_val() ^ ((board.last_super_index as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

impl Bot for UmctsV2Bot {
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

        // Fresh search tree for each turn to avoid stale hash collisions across moves
        self.node_store.clear();
        self.unmove_queue.clear();

        let root_hash = compute_full_hash(board);
        self.node_store
            .insert(root_hash, UmctsV2Node::new(root_moves.clone()));

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

                // If node has unexpanded moves, stop selection and expand
                if !curr_node.unexpanded_moves.is_empty() {
                    break;
                }

                // If node is already proven, no need to delve further
                if curr_node.proof != ProofStatus::Unproven {
                    break;
                }

                // Choose child using UCT
                if let Some((action, next_hash)) = self.select_child(curr_hash) {
                    let undo = sim_board.make_move(action, zobrist_constants);
                    self.unmove_queue.push((curr_hash, undo, action));
                    curr_hash = next_hash;
                } else {
                    break;
                }
            }

            // 2. Expansion (1 node expansion)
            let win_state = sim_board.get_win_state();
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

                        let mut child_node = UmctsV2Node::new(child_moves);
                        if child_win_state != GameResult::Unresolved {
                            child_node.proof = match child_win_state {
                                GameResult::Tie => ProofStatus::ProvenDraw,
                                GameResult::P1Win => {
                                    if m.x_to_move {
                                        ProofStatus::ProvenLoss // For next player (O)
                                    } else {
                                        ProofStatus::ProvenWin
                                    }
                                }
                                GameResult::P2Win => {
                                    if !m.x_to_move {
                                        ProofStatus::ProvenLoss // For next player (X)
                                    } else {
                                        ProofStatus::ProvenWin
                                    }
                                }
                                _ => ProofStatus::Unproven,
                            };
                        }

                        self.node_store.insert(child_hash, child_node);
                        self.unmove_queue.push((curr_hash, undo, m));
                        curr_hash = child_hash;
                    }
                }
            }

            // 3. Simulation (Pure lightweight rollout to terminal)
            let result = if sim_board.get_win_state() != GameResult::Unresolved {
                sim_board.get_win_state()
            } else {
                Self::simulate(sim_board.clone(), zobrist_constants, &mut self.rng)
            };

            // 4. Backpropagation & Proof Propagation (Unmaking moves up the queue)
            // Record reward for terminal leaf
            if let Some(leaf_node) = self.node_store.get_mut(&curr_hash) {
                leaf_node.visits += 1;
            }

            while let Some((parent_hash, undo, action)) = self.unmove_queue.pop() {
                // Determine reward from perspective of player who took `action`
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

                // Update child stats & extract proof status
                let child_proof = if let Some(child_node) = self.node_store.get_mut(&curr_hash) {
                    child_node.wins += reward;
                    child_node.proof
                } else {
                    ProofStatus::Unproven
                };

                // Update parent stats
                if let Some(parent_node) = self.node_store.get_mut(&parent_hash) {
                    parent_node.visits += 1;
                }

                // Update sound MCTS-Solver proofs
                self.update_proof_status(parent_hash, child_proof);

                // Unmake move on the persistent board
                sim_board.unmake_move(undo);
                curr_hash = parent_hash;
            }

            // Early exit if root is proven
            if let Some(root_node) = self.node_store.get(&root_hash) {
                if root_node.proof == ProofStatus::ProvenWin {
                    break;
                }
            }
        }

        // Final Move Selection:
        let root_node = match self.node_store.get(&root_hash) {
            Some(n) => n,
            None => return root_moves[0],
        };

        // 1. If any child is a ProvenLoss from child's perspective (ProvenWin for us), play it!
        for &(action, child_hash) in &root_node.children {
            if root_moves.iter().any(|m| m.move_index == action.move_index) {
                if let Some(child_node) = self.node_store.get(&child_hash) {
                    if child_node.proof == ProofStatus::ProvenLoss {
                        return action;
                    }
                }
            }
        }

        // 2. Otherwise pick child with highest visit count among legal non-losing children
        let best_action = root_node
            .children
            .iter()
            .filter(|&(action, child_hash)| {
                root_moves.iter().any(|m| m.move_index == action.move_index)
                    && self
                        .node_store
                        .get(child_hash)
                        .map_or(true, |c| c.proof != ProofStatus::ProvenWin)
            })
            .max_by_key(|&(_, child_hash)| {
                self.node_store
                    .get(child_hash)
                    .map_or(0, |c| c.visits)
            })
            .map(|&(action, _)| action);

        best_action
            .or_else(|| {
                // Fallback to highest visit child among legal root moves
                root_node
                    .children
                    .iter()
                    .filter(|&(action, _)| {
                        root_moves.iter().any(|m| m.move_index == action.move_index)
                    })
                    .max_by_key(|&(_, child_hash)| {
                        self.node_store
                            .get(child_hash)
                            .map_or(0, |c| c.visits)
                    })
                    .map(|&(action, _)| action)
            })
            .unwrap_or(root_moves[0])
    }
}
