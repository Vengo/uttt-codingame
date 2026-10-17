use std::cmp::{max, min, Ordering};
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::{BuildHasher, Hash, Hasher};

#[derive(Debug)]
pub struct HashState {
    value_lower: i8,
    value_upper: i8,
    depth: u8,
    move_index: u8,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum GameResult {
    P1Win,
    P2Win,
    Tie,
    Unresolved,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GameMove {
    pub x_to_move: bool,
    pub move_index: u8,
}

impl Default for GameMove {
    fn default() -> GameMove {
        GameMove {
            x_to_move: true,
            move_index: 255,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct UmctsNode {
    pub value_lower: i8,
    pub value_upper: i8,
    pub best_known: u8,
    pub visits: u32,
    pub value: i32,
}

impl Default for UmctsNode {
    fn default() -> UmctsNode {
        UmctsNode {
            value_lower: i8::MIN,
            value_upper: i8::MAX,
            best_known: u8::MAX,
            visits: 0,
            value: 0,
        }
    }
}
impl UmctsNode {
    fn new_windowed(lower: i8, upper: i8) -> UmctsNode {
        UmctsNode {
            value_lower: lower,
            value_upper: upper,
            best_known: u8::MAX,
            visits: 0,
            value: 0,
        }
    }
}

pub trait AlphaBetaCompatible<T: Hash + Eq + Copy, Unmove, HashConstants> {
    fn hash_val(&self) -> T;
    fn get_win_state(&self) -> GameResult;
    fn x_to_move(&self) -> bool;
    fn make_move(&mut self, a_move: GameMove, zobrist_constants: &HashConstants) -> Unmove;
    fn unmake_move(&mut self, a_move: Unmove);
    fn score(&self) -> i8;
    fn get_legal_moves(&self, legal_lookup: &[Vec<GameMove>]) -> Vec<GameMove>;
    fn display(&self) -> String;
    fn is_valid(&self) -> bool;
}

const MCTS_LAMBDA: f32 = 1.414 * (i8::MAX as f32);

pub fn uct_value(
    value: i32,
    node_visits: u32,
    action_visits: u32,
    x_to_move: bool,
) -> OrderedFloat {
    let adjusted_visit_count = (action_visits as f32) + 0.01;
    let average_val = (value as f32) / (adjusted_visit_count);
    let explore_contribution =
        MCTS_LAMBDA * f32::sqrt(f32::ln(node_visits as f32) / (adjusted_visit_count));

    if x_to_move {
        OrderedFloat(average_val + explore_contribution)
    } else {
        OrderedFloat(average_val - explore_contribution)
    }
}

const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);
const MAX: fn(i8, i8) -> i8 = |a, b| a.max(b);

// #[derive(PartialOrd, PartialEq, Debug)]
#[derive(PartialEq, Debug)]
pub struct OrderedFloat(pub f32);

impl Eq for OrderedFloat {}
impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &OrderedFloat) -> Option<Ordering> {
        self.0.partial_cmp(&(other.0))
    }
}
impl Ord for OrderedFloat {
    fn cmp(&self, other: &OrderedFloat) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct IdentityHasher {
    hash_val: u64,
}
impl IdentityHasher {
    fn new() -> IdentityHasher {
        IdentityHasher { hash_val: 0 }
    }
}
impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.hash_val
    }

    fn write(&mut self, bytes: &[u8]) {
        let start_val = self.hash_val;
        let result = bytes.iter().fold(start_val, |state, next| {
            (state >> 8) | (u64::from(*next) << 56)
        });
        self.hash_val = result;
        panic!("please use write_u64 exclusively for IdentityHasher.");
    }

    fn write_u64(&mut self, val: u64) {
        self.hash_val = val;
    }
}

pub struct IdentityHasherBuilder {}
impl BuildHasher for IdentityHasherBuilder {
    type Hasher = IdentityHasher;
    fn build_hasher(&self) -> IdentityHasher {
        IdentityHasher::new()
    }
}

pub struct UmctsQueueElement<T, Unmove> {
    undo: Unmove,
    board_hash: T,
    child_hashes: Vec<T>,
    x_to_move: bool,
}

//    #[inline(always)]
fn get_optimal<'a>(
    candidate_moves: &'a [(&GameMove, UmctsNode)],
    x_to_move: bool,
    board_visits: u32,
) -> Option<&'a (&'a GameMove, UmctsNode)> {
    if x_to_move {
        candidate_moves.iter().max_by_key(|(_a_move, a_node)| {
            let UmctsNode {
                value_upper: move_beta,
                visits: move_visits,
                value: move_value,
                ..
            } = a_node;
            let uct_val = uct_value(*move_value, board_visits, *move_visits, x_to_move);
            //                            eprintln!("successor UCT value: {:?}", uct_val);
            (move_beta, uct_val)
        })
    } else {
        candidate_moves.iter().min_by_key(|(_a_move, a_node)| {
            let UmctsNode {
                value_lower: move_alpha,
                visits: move_visits,
                value: move_value,
                ..
            } = a_node;
            let uct_val = uct_value(*move_value, board_visits, *move_visits, x_to_move);
            (move_alpha, uct_val)
        })
    }
}

//    #[inline(never)]
//    #[inline(always)]
pub fn get_successors<
    'a,
    T: Hash + Eq + Copy + Display,
    Unmove,
    HashConstants,
    BoardState: AlphaBetaCompatible<T, Unmove, HashConstants>,
    S: BuildHasher,
>(
    all_moves: &'a [GameMove],
    board: &mut BoardState,
    node_store: &mut HashMap<T, UmctsNode, S>,
    zobrist_constants: &HashConstants,
    alpha: i8,
    beta: i8,
    parent_visits: u32,
) -> (Vec<(&'a GameMove, UmctsNode)>, Vec<T>) {
    let legal_move_count = all_moves.len();

    let mut candidate_moves = Vec::with_capacity(legal_move_count);
    let mut child_hashes = Vec::with_capacity(legal_move_count);

    for each_move in all_moves.iter() {
        let undo = board.make_move(*each_move, zobrist_constants);
        let successor_hash = board.hash_val();
        child_hashes.push(successor_hash);
        board.unmake_move(undo);

        let retrieved_node = if parent_visits > 1 {
            match node_store.get(&successor_hash) {
                Some(val) => UmctsNode {
                    value_lower: alpha.max(val.value_lower),
                    value_upper: beta.min(val.value_upper),
                    best_known: u8::MAX,
                    visits: val.visits,
                    value: val.value,
                },
                None => UmctsNode::new_windowed(alpha, beta),
            }
        } else {
            UmctsNode::new_windowed(alpha, beta)
        };

        if retrieved_node.value_lower < retrieved_node.value_upper {
            candidate_moves.push((each_move, retrieved_node));
        };
    }

    (candidate_moves, child_hashes)
}

pub fn advance_rollout<
    T: Hash + Eq + Copy + Display,
    Unmove,
    HashConstants,
    BoardState: AlphaBetaCompatible<T, Unmove, HashConstants> + Clone,
    S: BuildHasher,
>(
    board_queue: &mut Vec<UmctsQueueElement<T, Unmove>>,
    board: &mut BoardState,
    legal_moves: &[Vec<GameMove>],
    alpha: i8,
    beta: i8,
    node_store: &mut HashMap<T, UmctsNode, S>,
    zobrist_constants: &HashConstants,
) -> Option<(i8, i8)> {
    let board_hash = board.hash_val();

    let a_node = node_store
        .entry(board_hash)
        .or_insert_with(UmctsNode::default);
    a_node.visits += 1;

    let board_visits = a_node.visits;
    let board_lower = a_node.value_lower;
    let board_upper = a_node.value_upper;

    let alpha = max(board_lower, alpha);
    let beta = min(board_upper, beta);

    let x_to_move = board.x_to_move();
    let all_moves = board.get_legal_moves(legal_moves);

    // eprintln!("all_moves in order are:\n{:?}\n", all_moves);
    let (candidate_moves, child_hashes) = get_successors(
        &all_moves,
        board,
        node_store,
        zobrist_constants,
        alpha,
        beta,
        board_visits,
    );

    // eprintln!("candidate moves in some order are:\n{:?}\n",candidate_moves);
    let opt_tuple = get_optimal(&candidate_moves, x_to_move, board_visits);
    // eprintln!("{:?}", opt_tuple);

    match opt_tuple {
        Some((move_ref, opt_node)) => {
            let UmctsNode {
                value_lower: best_alpha,
                value_upper: best_beta,
                ..
            } = opt_node;
            let undo = board.make_move(**move_ref, zobrist_constants);
            let new_queue_element = UmctsQueueElement {
                undo,
                board_hash,
                child_hashes,
                x_to_move,
            };
            board_queue.push(new_queue_element);
            Some((*best_alpha, *best_beta))
        }
        None => None,
    }
}

pub fn reverse_rollout<
    T: Hash + Eq + Copy + Display,
    Unmove,
    HashConstants,
    BoardState: AlphaBetaCompatible<T, Unmove, HashConstants> + Clone,
    S: BuildHasher,
>(
    rollout_result: i8,
    board_queue: &mut Vec<UmctsQueueElement<T, Unmove>>,
    board: &mut BoardState,
    node_store: &mut HashMap<T, UmctsNode, S>,
) {
    let mut update_bounds = true;

    while let Some(queue_element) = board_queue.pop() {
        let UmctsQueueElement {
            undo,
            board_hash,
            child_hashes,
            x_to_move,
        } = queue_element;

        let bounds = if update_bounds & x_to_move {
            let mut lower_bound = i8::MIN;
            let mut upper_bound = i8::MIN;
            for child_hash in child_hashes.iter() {
                let (child_lower, child_upper) = match node_store.get(child_hash) {
                    Some(stored_node) => (stored_node.value_lower, stored_node.value_upper),
                    None => (i8::MIN, i8::MAX),
                };
                lower_bound = max(lower_bound, child_lower);
                upper_bound = max(upper_bound, child_upper);

                if lower_bound == i8::MAX {
                    break;
                };
            }
            (lower_bound, upper_bound)
        } else if update_bounds & !x_to_move {
            let mut lower_bound = i8::MAX;
            let mut upper_bound = i8::MAX;
            for child_hash in child_hashes.iter() {
                let (child_lower, child_upper) = match node_store.get(child_hash) {
                    Some(stored_node) => (stored_node.value_lower, stored_node.value_upper),
                    None => (i8::MIN, i8::MAX),
                };
                lower_bound = min(lower_bound, child_lower);
                upper_bound = min(upper_bound, child_upper);

                if upper_bound == i8::MIN {
                    break;
                };
            }
            (lower_bound, upper_bound)
        } else {
            (i8::MIN, i8::MAX)
        };

        if bounds == (i8::MIN, i8::MAX) {
            update_bounds = false;
        }

        let cached_node = node_store.get_mut(&board_hash).unwrap();
        if update_bounds {
            let (lower_bound, upper_bound) = bounds;
            cached_node.value_upper = upper_bound;
            cached_node.value_lower = lower_bound;
        }
        //            eprint!("cache value with result {:?}\n pre \t post \n{}",rollout_result,cached_node.value);
        cached_node.value += i32::from(rollout_result);
        //            eprintln!("\t{}",cached_node.value);
        board.unmake_move(undo);
    }
}

pub fn alpha_beta_search<T: Hash + Eq + Copy, Unmove, HashConstants, S: BuildHasher>(
    board: &mut impl AlphaBetaCompatible<T, Unmove, HashConstants>,
    depth: u8,
    alpha: i8,
    beta: i8,
    transpositions: &mut HashMap<T, HashState, S>,
    legal_moves: &[Vec<GameMove>],
    zobrist_constants: &HashConstants,
) -> (i8, i8, u8, bool) {
    // implemented based on pseudo-code from from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    //        eprintln!("starting an alpha beta call");
    let node_hash = board.hash_val();
    let mut terminal_found = true;

    let HashState {
        value_lower: stored_lower,
        value_upper: stored_upper,
        depth: _stored_depth,
        move_index: stored_move_index,
    } = match transpositions.get(&node_hash) {
        Some(hashed_node) if hashed_node.depth >= depth => {
            //                eprintln!("restoring transposition eval:\n{:?}",hashed_node);
            terminal_found &= hashed_node.depth == u8::MAX;
            hashed_node
        }
        _ => &HashState {
            value_lower: i8::MIN,
            value_upper: i8::MAX,
            depth: 0,
            move_index: u8::MAX,
        },
    };

    let mut move_index = min(u8::MAX, *stored_move_index);
    let mut node_alpha = max(alpha, *stored_lower);
    let mut node_beta = min(beta, *stored_upper);

    let (new_lower, new_upper) = match board.get_win_state() {
        GameResult::P1Win => (i8::MAX, i8::MAX),
        GameResult::P2Win => (i8::MIN, i8::MIN),
        GameResult::Tie => (0, 0),
        GameResult::Unresolved if depth == 0 => {
            terminal_found = false;
            (board.score(), board.score())
        }
        GameResult::Unresolved => {
            //                eprintln!("board unresolved:\n{}", board.display());
            let (selector, opt_val, start_val): (fn(i8, i8) -> i8, i8, i8) = if board.x_to_move() {
                (MAX, i8::MAX, i8::MIN)
            } else {
                (MIN, i8::MIN, i8::MAX)
            };
            let mut new_lower = start_val;
            let mut new_upper = start_val;

            let moves = board.get_legal_moves(legal_moves);

            if move_index == 255 {
                move_index = moves[0].move_index;
            }

            if node_alpha >= node_beta {
                //                    terminal_found = true;
                return (*stored_lower, *stored_upper, move_index, terminal_found);
            } else {
                for each_move in moves.iter() {
                    assert_eq!(
                        board.x_to_move(),
                        each_move.x_to_move,
                        "side to move mismatch in alpha beta search"
                    );
                    let unmake_instructions = board.make_move(*each_move, zobrist_constants);
                    let (candidate_lower, candidate_upper, _move_index, _terminal_search) = {
                        //                        eprintln!("looking at board after move at depth {}. x to play? {}:\n{}",
                        //                                  depth, board.x_to_move(), board.display());
                        alpha_beta_search(
                            board,
                            depth - 1,
                            node_alpha,
                            node_beta,
                            transpositions,
                            legal_moves,
                            zobrist_constants,
                        )
                    };

                    //                    terminal_found = terminal_search;
                    board.unmake_move(unmake_instructions);

                    new_lower = selector(new_lower, candidate_lower);
                    new_upper = selector(new_upper, candidate_upper);

                    if board.x_to_move() {
                        if new_lower > node_alpha {
                            node_alpha = new_lower;
                            move_index = each_move.move_index;
                        }
                    } else if new_upper < node_beta {
                        node_beta = new_upper;
                        move_index = each_move.move_index;
                    }

                    if (new_lower == opt_val) & (new_upper == opt_val) {
                        move_index = each_move.move_index;
                        break;
                    }
                }
            }

            (new_lower, new_upper)
        }
    };

    let insert_depth = if terminal_found { u8::MAX } else { depth };
    let inserted_state = HashState {
        value_lower: new_lower,
        value_upper: new_upper,
        depth: insert_depth,
        move_index,
    };

    transpositions.insert(node_hash, inserted_state);
    //        eprintln!("returning from alpha beta");
    (new_lower, new_upper, move_index, terminal_found)
}
