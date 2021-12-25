use crate::ttt_trait_solving::TicTacToeEngine;

pub mod game_solving {
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
                let (selector, opt_val, start_val): (fn(i8, i8) -> i8, i8, i8) =
                    if board.x_to_move() {
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
}

pub mod ttt_trait_solving {
    use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult, OrderedFloat};
    use itertools::Itertools;
    use std::collections::{HashMap, HashSet};

    // tictactoe constants
    const TTT_BOARD_WIDTH: usize = 3;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const CAT: u16 = (1 << 9) - 1;

    const REACHABLE_BOARD_COUNT: usize = 18_753; // At most 3**9=19683 boards should exist
                                                 // 18_753 is the number of tic tac toe boards reachable from the initial position without
                                                 //  requiring that players alternate moves (so x might move twice, then O 3 times.

    #[derive(Debug)]
    pub struct TicTacToeEngine {
        /// On approaching this project again, it feels like it would be useful to have some sort of
        /// an object to store state related to game search, especially when it comes to handling
        /// stuff like converting to and from IDs, storing computed search results (which will be
        /// more relevant for ultimate tic tac toe since it's actually mildly complicated), and
        /// maybe other little bits of state.
        ///
        boards: [TicTacToeBoard; REACHABLE_BOARD_COUNT],
        board_to_index: HashMap<TicTacToeBoard, u16>,
        board_transitions_x_to_move: [[TicTacToeBoardId; 9]; REACHABLE_BOARD_COUNT],
        board_transitions_o_to_move: [[TicTacToeBoardId; 9]; REACHABLE_BOARD_COUNT],

        board_distance_to_x_win: [u8; REACHABLE_BOARD_COUNT],
        board_distance_to_o_win: [u8; REACHABLE_BOARD_COUNT],
        board_minimax_value_depth_o_to_move: [(i8, u8); REACHABLE_BOARD_COUNT],
        board_minimax_value_depth_x_to_move: [(i8, u8); REACHABLE_BOARD_COUNT],
    }

    impl Default for TicTacToeEngine {
        fn default() -> Self {
            TicTacToeEngine::new()
        }
    }

    impl TicTacToeEngine {
        pub fn get_board(&self, board_id: &TicTacToeBoardId) -> &TicTacToeBoard {
            self.boards.get(board_id.board_id as usize).unwrap()
        }

        pub fn new() -> Self {
            // Might be better to use the builder pattern, but I'm going to do it this way!
            eprintln!("generating boards!");
            let mut board_to_index = HashMap::with_capacity(2 * REACHABLE_BOARD_COUNT);
            // 2*REACHABLE_BOARD_COUNT might stop a resize or two, but seems like a good bet.
            let mut numbered_boards: [TicTacToeBoard; REACHABLE_BOARD_COUNT] =
                [TicTacToeBoard::default(); REACHABLE_BOARD_COUNT];
            let mut x_transitions = [[TicTacToeBoardId::default(); 9]; REACHABLE_BOARD_COUNT];
            let mut o_transitions = [[TicTacToeBoardId::default(); 9]; REACHABLE_BOARD_COUNT];
            eprintln!("allocated boards!");

            let init_board = TicTacToeBoard { xbb: 0, obb: 0 };
            let mut board_index_to_expand = 0usize;
            let mut board_index_to_write = 1usize;
            numbered_boards[board_index_to_expand] = init_board;
            for board_index_to_expand in 0..REACHABLE_BOARD_COUNT {
                let board: TicTacToeBoard = numbered_boards[board_index_to_expand];
                if board.get_win_state() == GameResult::Unresolved {
                    for x_to_move in &[true, false] {
                        let legal_moves = board.get_legal_moves_no_lookup(*x_to_move);
                        for move_index in 0..legal_moves.move_count {
                            let a_move: GameMove = legal_moves.moves[move_index as usize];
                            let (_unmove, new_board) =
                                board.make_move_const(a_move, &TicTacToeZobrist {});
                            // let board_int = new_board.to_int();
                            let board_index =
                                board_to_index.entry(new_board).or_insert_with(|| {
                                    let old_index = board_index_to_write;
                                    numbered_boards[board_index_to_write] = new_board;
                                    board_index_to_write += 1;
                                    old_index as u16
                                });

                            if *x_to_move {
                                x_transitions[board_index_to_expand][a_move.move_index as usize] =
                                    TicTacToeBoardId {
                                        board_id: *board_index,
                                    };
                            } else {
                                o_transitions[board_index_to_expand][a_move.move_index as usize] =
                                    TicTacToeBoardId {
                                        board_id: *board_index,
                                    };
                            }
                        }
                    }
                }
            }

            let mut board_distance_to_x_win = [u8::MAX; REACHABLE_BOARD_COUNT];
            let mut board_distance_to_o_win = [u8::MAX; REACHABLE_BOARD_COUNT];
            let mut board_minimax_value_depth_x_to_move =
                [(i8::MIN, u8::MAX); REACHABLE_BOARD_COUNT];
            let mut board_minimax_value_depth_o_to_move =
                [(i8::MAX, u8::MAX); REACHABLE_BOARD_COUNT];

            for board_index in (0..REACHABLE_BOARD_COUNT).rev() {
                let board_id = TicTacToeBoardId {
                    board_id: board_index as u16,
                };
                let board: TicTacToeBoard = numbered_boards[board_index];
                let board_win_state = board.get_win_state();
                if board_win_state == GameResult::P1Win {
                    board_distance_to_x_win[board_index] = 0;
                    board_minimax_value_depth_o_to_move[board_index] = (i8::MAX, 0);
                    board_minimax_value_depth_x_to_move[board_index] = (i8::MAX, 0);
                } else if board_win_state == GameResult::P2Win {
                    board_distance_to_o_win[board_index] = 0;
                    board_minimax_value_depth_o_to_move[board_index] = (i8::MIN, 0);
                    board_minimax_value_depth_x_to_move[board_index] = (i8::MIN, 0);
                } else if board_win_state == GameResult::Tie {
                    board_minimax_value_depth_o_to_move[board_index] = (0, 0);
                    board_minimax_value_depth_x_to_move[board_index] = (0, 0);
                } else {
                    // use pre-computed transitions to look over successors
                    let successors_x_to_move = x_transitions[board_index]
                        .iter()
                        .filter(|each_successor| (each_successor.board_id != 0));

                    let this_distance_to_x_win: u8 = 1 + successors_x_to_move
                        .clone()
                        .map(|each_successor| {
                            board_distance_to_x_win[each_successor.board_id as usize]
                        })
                        .min()
                        .unwrap()
                        .min(254);
                    board_distance_to_x_win[board_index] = this_distance_to_x_win;

                    let (minimax_val, minimax_depth): (i8, u8) = successors_x_to_move
                        .clone()
                        .map(|each_successor| {
                            board_minimax_value_depth_o_to_move[each_successor.board_id as usize]
                        })
                        .max_by_key(|(value, depth)| (*value, 9 - depth))
                        .unwrap();
                    board_minimax_value_depth_x_to_move[board_index] =
                        (minimax_val, 1 + minimax_depth);

                    let successors_o_to_move = o_transitions[board_index]
                        .iter()
                        .filter(|each_successor| (each_successor.board_id != 0));

                    let this_distance_to_o_win: u8 = 1 + successors_o_to_move
                        .clone()
                        .map(|each_successor| {
                            board_distance_to_o_win[each_successor.board_id as usize]
                        })
                        .min()
                        .unwrap()
                        .min(254);
                    board_distance_to_o_win[board_index] = this_distance_to_o_win;

                    let (minimax_val, minimax_depth): (i8, u8) = successors_o_to_move
                        .clone()
                        .map(|each_successor| {
                            board_minimax_value_depth_x_to_move[each_successor.board_id as usize]
                        })
                        .min_by_key(|(value, depth)| (*value, 9 - *depth))
                        .unwrap();
                    board_minimax_value_depth_o_to_move[board_index] =
                        (minimax_val, 1 + minimax_depth);
                }
            }

            TicTacToeEngine {
                boards: numbered_boards,
                board_to_index,
                board_transitions_x_to_move: x_transitions,
                board_transitions_o_to_move: o_transitions,
                board_distance_to_x_win,
                board_distance_to_o_win,
                board_minimax_value_depth_o_to_move,
                board_minimax_value_depth_x_to_move,
            }
        }

        pub fn get_move_ordering_array(&self) -> Vec<Vec<GameMove>> {
            let mut move_vec: Vec<Vec<GameMove>> = vec![Vec::with_capacity(0); 524_288];
            for (number, each_board) in self.boards.iter().enumerate() {
                let TicTacToeBoard { xbb, obb } = each_board;
                for x_to_move in &[true, false] {
                    let board = TicTacToeState {
                        xbb: *xbb,
                        obb: *obb,
                        x_to_move: *x_to_move,
                    };
                    let big_index = board.to_index();

                    let successor_ids = if *x_to_move {
                        self.board_transitions_x_to_move[number]
                    } else {
                        self.board_transitions_o_to_move[number]
                    };

                    let mut valid_successors: Vec<(usize, &TicTacToeBoardId)> = successor_ids
                        .iter()
                        .enumerate()
                        .filter(|each_successor| {
                            let (_index, id) = each_successor;
                            TicTacToeBoardId { board_id: 0 }.ne(id)
                        })
                        .collect();

                    if *x_to_move {
                        valid_successors.sort_by_key(|(move_number, board_id)| {
                            let successor_id = board_id.board_id as usize;
                            let (val, val_depth) =
                                self.board_minimax_value_depth_x_to_move[successor_id];
                            let threat_depth = self.board_distance_to_x_win[successor_id];
                            let defense_depth = self.board_distance_to_o_win[successor_id];
                            (-(val as i16), threat_depth, -(defense_depth as i16))
                        });
                    } else {
                        valid_successors.sort_by_key(|(move_number, board_id)| {
                            let successor_id = board_id.board_id as usize;
                            let (val, val_depth) =
                                self.board_minimax_value_depth_o_to_move[successor_id];
                            let threat_depth = self.board_distance_to_o_win[successor_id];
                            let defense_depth = self.board_distance_to_x_win[successor_id];
                            (-(val as i16), threat_depth, -(defense_depth as i16))
                        });
                    }

                    move_vec[big_index as usize] = valid_successors
                        .iter()
                        .map(|(move_number, _board_id)| GameMove {
                            move_index: *move_number as u8,
                            x_to_move: *x_to_move,
                        })
                        .collect();
                }
            }
            move_vec
        }
    }

    pub const BB_STATE_CHECK: [GameResult; 512] = TicTacToeBoard::make_bb_lookup();

    #[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
    pub struct TicTacToeTerminal {
        pub terminal_value: i8,
        pub depth_to_terminal: u8,
    }

    impl Default for TicTacToeTerminal {
        fn default() -> TicTacToeTerminal {
            TicTacToeTerminal {
                terminal_value: 0,
                depth_to_terminal: 9,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct TicTacToeState {
        pub xbb: u16,
        pub obb: u16,
        pub x_to_move: bool,
    }

    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct TicTacToeBoard {
        pub xbb: u16,
        pub obb: u16,
    }
    #[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct TicTacToeBoardId {
        pub board_id: u16,
    }
    impl TicTacToeBoardId {
        pub const fn default() -> TicTacToeBoardId {
            TicTacToeBoardId { board_id: 0 }
        }
    }

    #[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
    pub struct OrderedMoves {
        move_count: u8,
        moves: [GameMove; 9],
    }
    impl OrderedMoves {
        pub const fn default() -> OrderedMoves {
            let default_move = GameMove {
                x_to_move: true,
                move_index: 0,
            };
            OrderedMoves {
                move_count: 0,
                moves: [default_move; 9],
            }
        }
    }

    pub struct TicTacToeUnmove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    pub struct TicTacToeZobrist {
        //I'm not ACTUALLY implementing Zobrist constants for normal tictactoe...
    // I have to create this to match the make_move API though, so here we are!
    }

    impl TicTacToeState {
        pub fn to_index(&self) -> u32 {
            (u32::from(self.xbb) << 10)
                | (u32::from(self.obb) << 1)
                | (if self.x_to_move { 1u32 } else { 0u32 })
        }

        fn get_legal_moves_no_lookup(&self) -> Vec<GameMove> {
            let occupied_bb = self.xbb | self.obb;

            MOVE_ORDERING
                .iter()
                .filter_map(|a_move_index| match occupied_bb & (1u16 << a_move_index) {
                    0 => Some(GameMove {
                        x_to_move: self.x_to_move,
                        move_index: *a_move_index,
                    }),
                    _nonzero => None,
                })
                .collect()
        }
    }

    impl TicTacToeBoard {
        const fn default() -> TicTacToeBoard {
            TicTacToeBoard { xbb: 0, obb: 0 }
        }

        fn display(&self) -> String {
            let mut rows = Vec::new();
            for row in 0..TTT_BOARD_WIDTH {
                let mut row_rep = String::new();
                for col in 0..TTT_BOARD_WIDTH {
                    let mask = 1 << (col + TTT_BOARD_WIDTH * row);
                    if self.xbb & mask != 0 {
                        row_rep.push('X');
                    } else if self.obb & mask != 0 {
                        row_rep.push('O');
                    } else {
                        row_rep.push('.');
                    }
                }
                rows.push(row_rep);
            }
            rows.join("\n")
        }

        pub const fn get_win_state(&self) -> GameResult {
            // Forgive the weird casting here, it's due to const fn. See:
            //   https://stackoverflow.com/questions/60125657/rust-cant-use-enum-in-const-fn

            if (BB_STATE_CHECK[self.xbb as usize] as isize) == (GameResult::P1Win as isize) {
                GameResult::P1Win
            } else if (BB_STATE_CHECK[self.obb as usize] as isize) == (GameResult::P1Win as isize) {
                GameResult::P2Win
            } else if (self.xbb | self.obb) == CAT {
                GameResult::Tie
            } else {
                GameResult::Unresolved
            }
        }

        pub const fn get_win_state_alternate(&self) -> GameResult {
            // check for X or O win patterns, check for a filled board, or return unresolved
            let mut mask_index = 0;
            while mask_index < RESULT_MASKS.len() {
                if (self.xbb & RESULT_MASKS[mask_index]) == RESULT_MASKS[mask_index] {
                    return GameResult::P1Win;
                } else if (self.obb & RESULT_MASKS[mask_index]) == RESULT_MASKS[mask_index] {
                    return GameResult::P2Win;
                }
                mask_index += 1;
            }
            if (self.xbb | self.obb) == CAT {
                GameResult::Tie
            } else {
                GameResult::Unresolved
            }
        }

        pub const fn get_legal_moves_no_lookup(&self, x_to_move: bool) -> OrderedMoves {
            let occupied_bb = self.xbb | self.obb;

            // let mut move_order_copy = MOVE_ORDERING;
            let mut ordered_moves = OrderedMoves::default();
            // let mut moves: Vec<GameMove> = Vec::new();

            let mut move_index = 0u8;
            let mut assigned_moves = 0;
            while (move_index as usize) < MOVE_ORDERING.len() {
                if occupied_bb & (1u16 << move_index) == 0 {
                    let new_move = GameMove {
                        x_to_move,
                        move_index,
                    };
                    ordered_moves.moves[assigned_moves] = new_move;
                    ordered_moves.move_count += 1;
                    assigned_moves += 1;
                }
                move_index += 1;
            }

            ordered_moves
        }

        pub const fn to_int(&self) -> usize {
            // xbb and obb are bitboards with 9 bits
            ((self.xbb as usize) << 9) | (self.obb as usize)
        }

        pub const fn make_move_const(
            &self,
            a_move: GameMove,
            _zobrist_constants: &TicTacToeZobrist,
        ) -> (TicTacToeUnmove, TicTacToeBoard) {
            let mut board_copy = *self;
            let unmove = TicTacToeUnmove {
                x_to_move: a_move.x_to_move,
                move_index: a_move.move_index,
            };
            match a_move {
                GameMove {
                    x_to_move: true,
                    move_index,
                } => {
                    board_copy.xbb ^= 1u16 << move_index;
                }
                GameMove {
                    x_to_move: false,
                    move_index,
                } => {
                    board_copy.obb ^= 1u16 << move_index;
                }
            }
            (unmove, board_copy)
        }

        pub const fn make_bb_lookup() -> [GameResult; 512] {
            let mut bb = 0u16;
            let mut results = [GameResult::Unresolved; 512];
            while bb < 512 {
                let mut mask_index = 0;
                while mask_index < RESULT_MASKS.len() {
                    if bb & RESULT_MASKS[mask_index] == RESULT_MASKS[mask_index] {
                        results[bb as usize] = GameResult::P1Win;
                    }
                    mask_index += 1;
                }
                bb += 1;
            }

            results
        }
    }

    impl AlphaBetaCompatible<u32, TicTacToeUnmove, TicTacToeZobrist> for TicTacToeState {
        fn hash_val(&self) -> u32 {
            let base_hash = (u32::from(self.xbb)) << 16 | u32::from(self.obb);

            if self.x_to_move {
                base_hash
            } else {
                !base_hash
            }
        }

        fn get_win_state(&self) -> GameResult {
            // check for X or O win patterns, check for a filled board, or return unresolved
            if RESULT_MASKS.iter().any(|x| (x & self.xbb).eq(x)) {
                GameResult::P1Win
            } else if RESULT_MASKS.iter().any(|x| (x & self.obb).eq(x)) {
                GameResult::P2Win
            } else if (self.xbb | self.obb) == CAT {
                GameResult::Tie
            } else {
                GameResult::Unresolved
            }
        }

        fn x_to_move(&self) -> bool {
            self.x_to_move
        }

        fn make_move(
            &mut self,
            a_move: GameMove,
            _zobrist_constants: &TicTacToeZobrist,
        ) -> TicTacToeUnmove {
            //            pub x_to_move: bool,
            //            pub move_index: u8,
            let unmove = TicTacToeUnmove {
                x_to_move: self.x_to_move,
                move_index: a_move.move_index,
            };

            match a_move {
                GameMove {
                    x_to_move: true,
                    move_index,
                } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = false;
                }
                GameMove {
                    x_to_move: false,
                    move_index,
                } => {
                    self.obb ^= 1u16 << move_index;
                    self.x_to_move = true;
                }
            }

            unmove
        }

        fn unmake_move(&mut self, unmove: TicTacToeUnmove) {
            match unmove {
                TicTacToeUnmove {
                    x_to_move: true,
                    move_index,
                } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = true;
                }
                TicTacToeUnmove {
                    x_to_move: false,
                    move_index,
                } => {
                    self.obb ^= 1u16 << move_index;
                    self.x_to_move = false;
                }
            }
        }

        fn score(&self) -> i8 {
            match self.get_win_state() {
                GameResult::P1Win => i8::MAX,
                GameResult::P2Win => i8::MIN,
                GameResult::Tie => 0,
                GameResult::Unresolved if self.x_to_move => 1,
                GameResult::Unresolved => -1,
            }
        }

        fn get_legal_moves(&self, _legal_lookup: &[Vec<GameMove>]) -> Vec<GameMove> {
            //TODO: This should probably use the legal lookup directly
            let board_state = TicTacToeState {
                xbb: self.xbb,
                obb: self.obb,
                x_to_move: self.x_to_move,
            };
            let board_index = board_state.to_index();
            _legal_lookup[board_index as usize].clone()
        }

        fn display(&self) -> String {
            //        let mut board_rep = String::new();
            let board = TicTacToeBoard {
                xbb: self.xbb,
                obb: self.obb,
            };
            board.display()
        }

        fn is_valid(&self) -> bool {
            // there are more ways to invalidate a board than are checked here
            (self.obb & self.xbb) == 0
        }
    }
}

pub mod uttt_trait_solving {
    use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
    use crate::ttt_trait_solving::{TicTacToeEngine, TicTacToeState};
    use rand::{rngs, Rng, SeedableRng};
    use std::iter::FromIterator;

    // tictactoe constants
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const CAT: u16 = (1 << 9) - 1;
    const ZOBRIST_SEED: u64 = 0x95_62_9A_0C_B8_00_E9_41u64; //generated from random.org
    const HASH_DEFAULT: u64 = 0x8e_6c_65_b8_2e_fd_36_70u64; //generated from random.org

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct UltimateTicTacToeBoard {
        pub xbb: [u16; 9],
        pub x_supers: u16,
        pub obb: [u16; 9],
        pub o_supers: u16,
        //        cbb: u128,
        //        c_supers: u16,
        pub last_super_index: u8,
        pub x_to_move: bool,
        pub hash_state: u64,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub struct SubBoardAndSquare {
        pub sub_board_index: usize,
        pub sub_square_index: usize,
    }

    pub fn parse_index(an_index: u8) -> SubBoardAndSquare {
        // Returns the sub board [0,8] and sub square [0-8]
        let sub_square = (an_index % 9) as usize;
        let sub_board = (an_index / 9) as usize;

        SubBoardAndSquare {
            sub_square_index: sub_square,
            sub_board_index: sub_board,
        }
    }

    pub fn parse_index_from_super_coords(row_index: u8, col_index: u8) -> u8 {
        let board_row = row_index / 3;
        let subboard_row = row_index % 3;

        let board_col = col_index / 3;
        let subboard_col = col_index % 3;
        // Returns the sub board [0,8] and sub square [0-8]

        let subboard_index = 3 * board_row + board_col;
        let subsquare_index = 3 * subboard_row + subboard_col;

        9 * subboard_index + subsquare_index
    }

    #[derive(Clone, Copy)]
    pub struct UltimateTicTacToeUnmove {
        x_to_move: bool,
        move_index: u8,
        super_unmask: u16,
        last_super: u8,
        hash_reset: u64,
    }

    pub struct ZobristConstants {
        pub x_constants: [u64; 81],
        pub o_constants: [u64; 81],
        pub x_to_move: u64,
    }

    impl Default for UltimateTicTacToeBoard {
        fn default() -> UltimateTicTacToeBoard {
            UltimateTicTacToeBoard {
                xbb: [0u16; 9],
                obb: [0u16; 9],
                last_super_index: 9,
                o_supers: 0,
                x_supers: 0,
                x_to_move: true,
                hash_state: HASH_DEFAULT,
            }
        }
    }

    impl UltimateTicTacToeBoard {
        pub fn get_tied_mask(&self) -> u16 {
            self.xbb
                .iter()
                .zip(self.obb.iter())
                .enumerate()
                .filter_map(|(index, (xbb, obb))| match xbb | obb {
                    CAT => Some(1u16 << index),
                    _ => None,
                })
                .fold(0u16, |current, next| current | next)
        }

        pub fn get_subboard_moves(
            &self,
            subboard_index: usize,
            legality_lookup: &[Vec<GameMove>],
            side_to_move: bool,
        ) -> Vec<GameMove> {
            let subboard = TicTacToeState {
                xbb: self.xbb[subboard_index],
                obb: self.obb[subboard_index],
                x_to_move: side_to_move,
            };

            let lookup_index = subboard.to_index();
            match legality_lookup.get(lookup_index as usize) {
                Some(move_vec) => move_vec
                    .iter()
                    .map(
                        |GameMove {
                             x_to_move: _side,
                             move_index: a_move_index,
                         }| GameMove {
                            x_to_move: side_to_move,
                            move_index: a_move_index + 9 * subboard_index as u8,
                        },
                    )
                    .collect(),
                None => {
                    panic!("It shouldn't be possible for subboard moves not to find a move vector")
                }
            }
        }

        pub fn get_move_ordering_array() -> Vec<Vec<GameMove>> {
            eprintln!("making an engine");
            let engine = TicTacToeEngine::new();
            eprintln!("engine made");
            engine.get_move_ordering_array()
        }

        pub fn get_zobrist_constants() -> ZobristConstants {
            // zobrist hashing using a seedable RNG, and a seed from random.org
            let mut seeded = rngs::StdRng::seed_from_u64(ZOBRIST_SEED);
            let x_consts = UltimateTicTacToeBoard::generate_zobrist_random(&mut seeded);
            let o_consts = UltimateTicTacToeBoard::generate_zobrist_random(&mut seeded);
            let x_to_move_constant = seeded.gen::<u64>();

            ZobristConstants {
                x_constants: x_consts,
                o_constants: o_consts,
                x_to_move: x_to_move_constant,
            }
        }

        fn generate_zobrist_random(seeded: &mut rngs::StdRng) -> [u64; 81] {
            let x_consts = {
                let mut empty = [0u64; 81];
                for element in empty.iter_mut() {
                    *element = seeded.gen::<u64>();
                }
                empty
            };
            x_consts
        }
    }

    impl AlphaBetaCompatible<u64, UltimateTicTacToeUnmove, ZobristConstants>
        for UltimateTicTacToeBoard
    {
        fn hash_val(&self) -> u64 {
            //TODO: Try to enforce a consistent hash value. As of now, hash_state is only correct
            //  for the case where to positions were make_move'd from the start position.
            self.hash_state
        }

        fn get_win_state(&self) -> GameResult {
            // check for X or O win patterns, check for a filled board, or return unresolved
            if RESULT_MASKS.iter().any(|x| (x & self.x_supers).eq(x)) {
                GameResult::P1Win
            } else if RESULT_MASKS.iter().any(|x| (x & self.o_supers).eq(x)) {
                GameResult::P2Win
            } else {
                let new_mask = self
                    .xbb
                    .iter()
                    .zip(self.obb.iter())
                    .enumerate()
                    .filter_map(|(index, (xbb, obb))| match xbb | obb {
                        CAT => Some(1u16 << index),
                        _ => None,
                    })
                    .fold(0u16, |current, next| current | next);
                if new_mask | self.o_supers | self.x_supers == CAT {
                    let x_won = self.x_supers.count_ones() as i8;
                    let o_won = self.o_supers.count_ones() as i8;
                    let heuristic_score = x_won - o_won;
                    match heuristic_score {
                        0 => GameResult::Tie,
                        value if value > 0 => GameResult::P1Win,
                        _ => GameResult::P2Win,
                    }
                } else {
                    GameResult::Unresolved
                }
            }
        }

        fn x_to_move(&self) -> bool {
            self.x_to_move
        }

        fn make_move(
            &mut self,
            a_move: GameMove,
            zobrist_constants: &ZobristConstants,
        ) -> UltimateTicTacToeUnmove {
            let GameMove {
                x_to_move,
                move_index: a_move_index,
            } = a_move;

            let original_hash_state = self.hash_state;

            let hash_update: u64 = if x_to_move {
                zobrist_constants.x_to_move ^ zobrist_constants.x_constants[a_move_index as usize]
            } else {
                zobrist_constants.x_to_move ^ zobrist_constants.o_constants[a_move_index as usize]
            };
            self.hash_state ^= hash_update;

            let SubBoardAndSquare {
                sub_board_index,
                sub_square_index,
            } = parse_index(a_move_index);
            //            eprintln!("{} {}", sub_board_index, sub_square_index);

            let to_move_reset = x_to_move;
            let move_to_unmask = a_move_index;
            let mut a_super_unmask = 0;
            let sub_board_reset = self.last_super_index;

            match a_move {
                GameMove {
                    x_to_move: true, ..
                } => {
                    self.xbb[sub_board_index] ^= 1 << sub_square_index;
                    let result_board = self.xbb[sub_board_index];

                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                    if is_sub_win {
                        let super_mask = 1u16 << sub_board_index;
                        self.x_supers |= super_mask;
                        a_super_unmask = super_mask;
                    }
                    self.last_super_index = sub_square_index as u8;
                    self.x_to_move = false;
                }
                GameMove {
                    x_to_move: false, ..
                } => {
                    self.obb[sub_board_index] ^= 1 << sub_square_index;
                    let result_board = self.obb[sub_board_index];

                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                    if is_sub_win {
                        let super_mask = 1u16 << sub_board_index;
                        self.o_supers |= 1 << sub_board_index;
                        a_super_unmask = super_mask;
                    }
                    self.last_super_index = sub_square_index as u8;
                    self.x_to_move = true;
                }
            }

            UltimateTicTacToeUnmove {
                x_to_move: to_move_reset,
                move_index: move_to_unmask,
                super_unmask: a_super_unmask,
                last_super: sub_board_reset,
                hash_reset: original_hash_state,
            }
        }

        fn unmake_move(&mut self, unmove: UltimateTicTacToeUnmove) {
            let UltimateTicTacToeUnmove {
                x_to_move,
                move_index,
                super_unmask,
                last_super,
                hash_reset,
            } = unmove;

            let SubBoardAndSquare {
                sub_board_index,
                sub_square_index,
            } = parse_index(move_index);

            self.x_to_move = x_to_move;
            self.last_super_index = last_super;
            self.hash_state = hash_reset;

            if x_to_move {
                self.xbb[sub_board_index] ^= 1 << sub_square_index;
                self.x_supers ^= super_unmask;
            } else {
                self.obb[sub_board_index] ^= 1 << sub_square_index;
                self.o_supers ^= super_unmask;
            };
        }

        fn score(&self) -> i8 {
            let x_won = self.x_supers.count_ones() as i8;
            let o_won = self.o_supers.count_ones() as i8;

            match self.get_win_state() {
                GameResult::P1Win => i8::MAX,
                GameResult::P2Win => i8::MIN,
                GameResult::Tie => 0i8,
                GameResult::Unresolved => x_won - o_won,
            }
        }

        fn get_legal_moves(&self, legality_lookup: &[Vec<GameMove>]) -> Vec<GameMove> {
            // TODO: P2: Modify this code to generate legal moves by an array lookup
            // TODO: P1: Consider generating those legal moves ordered by 'next best (small) TTT move'
            // gather won bitboards, the x/o occupation bitboards, and the total occupation
            // of the bitboard directed by the last move
            let complete_bbs = self.x_supers | self.o_supers | self.get_tied_mask();

            //            eprintln!("generating moves for x? {}",self.x_to_move);
            // arm 1 9 indicates 'play anywhere', as does directing to a full
            // arm 2 checks to see if the directed subboard has been won
            // arm 3 checks if the directed subboard is full
            // arm 4 creates a bitmask of 1s aligned to the directed board
            if (self.last_super_index >= 9) | ((complete_bbs & (1 << self.last_super_index)) != 0) {
                let mut moves = Vec::with_capacity(81);
                for subboard_index in 0..9usize {
                    if (complete_bbs & (1 << subboard_index)) == 0 {
                        moves.append(&mut self.get_subboard_moves(
                            subboard_index,
                            legality_lookup,
                            self.x_to_move,
                        ))
                    }
                }
                moves
            } else {
                self.get_subboard_moves(
                    self.last_super_index as usize,
                    legality_lookup,
                    self.x_to_move,
                )
            }
        }

        fn display(&self) -> String {
            let mut rows = [['+'; 11]; 11];
            for row_index in 0..9 {
                let the_row = rows.get_mut(row_index + row_index / 3).unwrap();
                for col_index in 0..9 {
                    let subboard_row = row_index % 3;
                    let subboard_col = col_index % 3;

                    let superboard_row = row_index / 3;
                    let superboard_col = col_index / 3;

                    let subboard = 3 * superboard_row + superboard_col;
                    let subboard_square_mask = 1u16 << (3 * subboard_row + subboard_col);

                    let an_item = the_row.get_mut(col_index + col_index / 3).unwrap();
                    if self.xbb[subboard] & subboard_square_mask != 0 {
                        *an_item = 'X';
                    } else if self.obb[subboard] & subboard_square_mask != 0 {
                        *an_item = 'O';
                    } else {
                        *an_item = '.';
                    }
                }
            }

            rows.iter()
                .map(|a_row| {
                    let mut row = String::from_iter(a_row.iter());
                    row.push('\n');
                    row
                })
                .collect()
        }

        fn is_valid(&self) -> bool {
            self.obb
                .iter()
                .zip(self.xbb.iter())
                .all(|(o_subboard, x_subboard)| (o_subboard & x_subboard) == 0)
        }
    }
}

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

pub fn main() {
    // game loop for codingame.com competition
    use crate::game_solving::{
        advance_rollout, reverse_rollout, uct_value, AlphaBetaCompatible, GameMove, GameResult,
        IdentityHasherBuilder,
    };
    use crate::uttt_trait_solving::{parse_index_from_super_coords, UltimateTicTacToeBoard};
    use rand::distributions::{Standard, Uniform};
    use rand::{rngs, Rng, SeedableRng};

    use std::collections::HashMap;
    use std::time;

    let mut uttt_board = UltimateTicTacToeBoard {
        xbb: [0u16; 9],
        obb: [0u16; 9],
        last_super_index: 9,
        o_supers: 0,
        x_supers: 0,
        x_to_move: true,
        hash_state: 0,
    };

    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let mut turn_count = 0;

    let mut unmove_queue = Vec::with_capacity(81);
    let mut node_store = HashMap::with_capacity_and_hasher(2_000_000, IdentityHasherBuilder {});
    let mut legal_moves = Vec::new();
    let mut engine;

    let rand_generation = time::Instant::now();
    let random_indices = {
        let mut seeded = rngs::StdRng::seed_from_u64(0);
        let mut temporary_buffer: Vec<u16> =
            seeded.sample_iter(&Standard).take(1_600_000).collect();
        // seeded.fill(&mut temporary_buffer);2

        temporary_buffer
    };
    // println!(
    //     "generated values in {} microseconds",
    //     rand_generation.elapsed().as_micros()
    // );

    loop {
        // **************************************************************
        // begin supplied boiler plate
        // reads game input, and parses the result
        let (opponent_row, opponent_col) = parse_codingame_inputs();

        // end supplied boiler plate
        // ***************************************************************
        // setting start of turn timer

        // **************************************************************
        let turn_start = time::Instant::now();
        let time_limit = if turn_count == 0 {
            engine = TicTacToeEngine::new();
            legal_moves = engine.get_move_ordering_array();
            990_000
        } else {
            90_000
        };

        let mut rollout_count = 0;

        let best_move_index = match (opponent_row, opponent_col) {
            (-1, -1) => {
                let default_first = GameMove {
                    move_index: 40,
                    x_to_move: uttt_board.x_to_move(),
                };
                uttt_board.make_move(default_first, &zobrist_constants);

                while turn_start.elapsed().as_micros() < time_limit {
                    let mut search_alpha = i8::MIN;
                    let mut search_beta = i8::MAX;
                    //                    let mut rollout_result = 0;

                    while let Some(new_alpha_beta) = advance_rollout(
                        &mut unmove_queue,
                        &mut uttt_board,
                        &legal_moves,
                        search_alpha,
                        search_beta,
                        &mut node_store,
                        &zobrist_constants,
                    ) {
                        let (new_alpha, new_beta) = new_alpha_beta;
                        search_alpha = new_alpha;
                        search_beta = new_beta;
                    }

                    let final_hash = uttt_board.hash_val();
                    let terminal_node = node_store.get_mut(&final_hash).unwrap();

                    let rollout_result = match uttt_board.get_win_state() {
                        GameResult::P1Win => {
                            let rollout_result = i8::MAX;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        }
                        GameResult::P2Win => {
                            let rollout_result = i8::MIN;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        }
                        GameResult::Tie => {
                            let rollout_result = 0;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        }
                        _ => {
                            //TODO: P3: Figure out what on earth this means!
                            search_alpha
                        }
                    };
                    reverse_rollout(
                        rollout_result,
                        &mut unmove_queue,
                        &mut uttt_board,
                        &mut node_store,
                    );
                    rollout_count += 1;
                }
                let moves = uttt_board.get_legal_moves(&legal_moves);

                for each_move in moves.iter() {
                    eprintln!("========================");
                    //                eprintln!("x to move? {}",each_move.x_to_move);
                    let undo_move = uttt_board.make_move(*each_move, &zobrist_constants);
                    let this_hash = uttt_board.hash_val();
                    uttt_board.unmake_move(undo_move);
                    let child_stats = node_store.get(&this_hash);

                    match child_stats {
                        Some(node) => {
                            eprintln!(
                                "node avg value: {}",
                                (node.value as f32) / (node.visits as f32)
                            );
                            eprintln!("node visits: {}", node.visits);
                            eprintln!("move index: {}", each_move.move_index);
                        }
                        None => {
                            eprintln!("no stored statistics found for hash value: {}", this_hash);
                        }
                    }
                }

                40
            }
            (row, col) => {
                let opp_move_index = parse_index_from_super_coords(row as u8, col as u8);
                let opp_move = GameMove {
                    move_index: opp_move_index,
                    x_to_move: uttt_board.x_to_move(),
                };
                uttt_board.make_move(opp_move, &zobrist_constants);
                let x_to_move = uttt_board.x_to_move();
                eprintln!("board representation pre-move:\n{:?}", uttt_board);

                //                let mut rollout_result = 0;
                //                let board_hash = uttt_board.hash_val();

                while turn_start.elapsed().as_micros() < time_limit {
                    let mut search_alpha = i8::MIN;
                    let mut search_beta = i8::MAX;

                    while let Some(new_alpha_beta) = advance_rollout(
                        &mut unmove_queue,
                        &mut uttt_board,
                        &legal_moves,
                        search_alpha,
                        search_beta,
                        &mut node_store,
                        &zobrist_constants,
                    ) {
                        let (new_alpha, new_beta) = new_alpha_beta;
                        search_alpha = new_alpha;
                        search_beta = new_beta;
                    }

                    let final_hash = uttt_board.hash_val();
                    let terminal_node = node_store.get_mut(&final_hash).unwrap();

                    let rollout_result = match uttt_board.get_win_state() {
                        GameResult::P1Win => {
                            let rollout_result = i8::MAX;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        }
                        GameResult::P2Win => {
                            let rollout_result = i8::MIN;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        }
                        GameResult::Tie => {
                            let rollout_result = 0;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        }
                        _ => {
                            //TODO: P3: Figure out what on earth this means!
                            search_alpha
                        }
                    };
                    reverse_rollout(
                        rollout_result,
                        &mut unmove_queue,
                        &mut uttt_board,
                        &mut node_store,
                    );
                    rollout_count += 1;
                }

                let moves = uttt_board.get_legal_moves(&legal_moves);

                let mut mut_board = uttt_board.clone();

                let best_move = moves
                    .iter()
                    .max_by_key(|a_move| {
                        let undo = mut_board.make_move(**a_move, &zobrist_constants);
                        let this_hash = mut_board.hash_val();
                        mut_board.unmake_move(undo);
                        let child_stats = node_store.get(&this_hash);
                        match child_stats {
                            Some(node) => node.visits,
                            None => 0,
                        }
                    })
                    .unwrap();

                let moves = uttt_board.get_legal_moves(&legal_moves);

                for each_move in moves.iter() {
                    eprintln!("========================");
                    //                eprintln!("x to move? {}",each_move.x_to_move);
                    let undo_move = uttt_board.make_move(*each_move, &zobrist_constants);
                    let this_hash = uttt_board.hash_val();
                    uttt_board.unmake_move(undo_move);
                    let child_stats = node_store.get(&this_hash);

                    match child_stats {
                        Some(node) => {
                            eprintln!(
                                "node avg value: {}",
                                (node.value as f32) / (node.visits as f32)
                            );
                            eprintln!("node visits: {}", node.visits);
                            eprintln!("move index: {}", each_move.move_index);
                            eprintln!(
                                "uct value: {:?}",
                                uct_value(node.value, rollout_count, node.visits, x_to_move)
                            );
                            //                            value node visits action visits side to move
                        }
                        None => {
                            eprintln!("no stored statistics found for hash value: {}", this_hash);
                        }
                    }
                    eprintln!();
                }
                uttt_board.make_move(*best_move, &zobrist_constants);
                best_move.move_index
            }
        };
        let best_subboard_index = best_move_index / 9;
        let best_sub_square_index = best_move_index % 9;

        let base_row = best_subboard_index / 3;
        let base_col = best_subboard_index % 3;

        let final_row = base_row * 3 + best_sub_square_index / 3;
        let final_column = base_col * 3 + best_sub_square_index % 3;

        turn_count += 1;
        eprintln!("elapsed microseconds: {}", turn_start.elapsed().as_micros());
        eprintln!("completed rollouts: {}", rollout_count);
        eprintln!("board representation post-move:\n{:?}", uttt_board);
        eprintln!("table size: {}", node_store.len());
        println!("{} {}", final_row, final_column);
    }
}

fn parse_codingame_inputs() -> (i32, i32) {
    let mut input_line = String::new();
    std::io::stdin().read_line(&mut input_line).unwrap();

    let inputs = input_line.split(' ').collect::<Vec<_>>();
    let opponent_row = parse_input!(inputs[0], i32);
    let opponent_col = parse_input!(inputs[1], i32);
    let mut input_line = String::new();
    std::io::stdin().read_line(&mut input_line).unwrap();
    let valid_action_count = parse_input!(input_line, i32);
    for _i in 0..valid_action_count as usize {
        let mut input_line = String::new();
        std::io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(' ').collect::<Vec<_>>();
        let _row = parse_input!(inputs[0], i32);
        let _col = parse_input!(inputs[1], i32);
    }
    (opponent_row, opponent_col)
}
