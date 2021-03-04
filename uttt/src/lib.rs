//TODO: Generally increase code quality by reducing the number of clippy lints and compiler warnings
pub mod game_solving {
    use std::collections::HashMap;
    use std::hash::{Hash, BuildHasher, Hasher};
    use std::cmp::{max,min,Ordering};
    use std::fmt::{Display};

    #[derive(Debug)]
    pub struct HashState {
        value_lower: i8,
        value_upper: i8,
        depth: u8,
        move_index: u8,
    }

    #[derive(Debug, Eq, PartialEq)]
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
            GameMove { x_to_move: true, move_index: 255 }
        }
    }

    #[derive(Clone,Copy,Debug)]
    pub struct UMCTSNode {
        pub value_lower: i8,
        pub value_upper: i8,
        pub best_known: u8,
        pub visits: u32,
        pub value: i32,
    }

    impl Default for UMCTSNode {
        fn default() -> UMCTSNode {
            UMCTSNode {value_lower: std::i8::MIN, value_upper: std::i8::MAX, best_known: std::u8::MAX, visits: 0, value: 0}
        }
    }
    impl UMCTSNode{
        fn new_windowed(lower:i8, upper:i8) -> UMCTSNode {
            UMCTSNode{value_lower: lower, value_upper: upper, best_known: std::u8::MAX, visits: 0, value: 0}
        }
    }

    pub trait AlphaBetaTTCompatible<T: Hash + Eq + Copy, Unmove, HashConstants> {
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


    const MCTS_LAMBDA : f32 = 1.414*(std::i8::MAX as f32);

    pub fn uct_value(value: i32, node_visits: u32, action_visits: u32, x_to_move: bool) -> OrderedFloat {
        let adjusted_visit_count = (action_visits as f32) + 0.01;
        let average_val = (value as f32)/(adjusted_visit_count);
        let explore_contribution = MCTS_LAMBDA * f32::sqrt(
            f32::ln(node_visits as f32)/(adjusted_visit_count));

        if x_to_move {
            OrderedFloat(average_val + explore_contribution)
        } else {
            OrderedFloat(average_val - explore_contribution)
        }
    }

    const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);
    const MAX: fn(i8, i8) -> i8 = |a, b| a.max(b);

    #[derive(PartialOrd, PartialEq, Debug)]
    pub struct OrderedFloat(pub f32);

    impl Eq for OrderedFloat {}
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
            IdentityHasher{hash_val: 0}
        }
    }
    impl Hasher for IdentityHasher {
        fn finish(&self) -> u64 {self.hash_val}

        fn write(&mut self, bytes: &[u8]) {
            let start_val = self.hash_val;
            let result = bytes.iter().fold(
                start_val,
                |state, next| (state>>8)|(u64::from(*next )<<56)
            );
            self.hash_val = result;
            panic!("please use write_u64 exclusively for IdentityHasher.");
        }

        fn write_u64(&mut self, val: u64) {self.hash_val = val;}
    }

    pub struct IdentityHasherBuilder {
    }
    impl BuildHasher for IdentityHasherBuilder {
        type Hasher = IdentityHasher;
        fn build_hasher(&self) -> IdentityHasher {
            IdentityHasher::new()
        }
    }

    pub struct UMCTSQueueElement<T, Unmove> {
        undo: Unmove,
        board_hash: T,
        child_hashes: Vec<T>,
        x_to_move: bool,
    }

//    #[inline(always)]
    fn get_optimal<'a>(
        candidate_moves: &'a [(&GameMove, UMCTSNode)],
        x_to_move:bool,
        board_visits: u32
    ) -> Option<&'a (&'a GameMove, UMCTSNode)> {


        if x_to_move {
            candidate_moves.iter().max_by_key(
                |(_a_move, a_node)|
                    {
                        let UMCTSNode{value_upper: move_beta, visits: move_visits, value: move_value,..}=  a_node;
                        let uct_val = uct_value(
                            *move_value, board_visits, *move_visits, x_to_move);
//                            eprintln!("successor UCT value: {:?}", uct_val);
                        (move_beta, uct_val)
                    })
        } else {
            candidate_moves.iter().min_by_key(
                |(_a_move, a_node)|
                    {
                        let UMCTSNode{value_lower: move_alpha, visits: move_visits, value: move_value,..} =  a_node;
                        let uct_val = uct_value(
                            *move_value, board_visits, *move_visits, x_to_move);
                        (move_alpha, uct_val)
                    })
        }
    }

//    #[inline(never)]
//    #[inline(always)]
    pub fn get_successors<'a, T: Hash + Eq + Copy + Display, Unmove, HashConstants,
        BoardState: AlphaBetaTTCompatible<T,Unmove,HashConstants>, S: BuildHasher> (
        all_moves: &'a [GameMove],
        board: &mut BoardState,
        node_store: &mut HashMap<T,UMCTSNode,S>,
        zobrist_constants: &HashConstants,
        alpha: i8, beta: i8, parent_visits: u32
    ) -> (Vec<(&'a GameMove,UMCTSNode)>, Vec<T>) {
        let legal_move_count = all_moves.len();

        let mut candidate_moves = Vec::with_capacity(legal_move_count);
        let mut child_hashes = Vec::with_capacity(legal_move_count);

        for each_move in all_moves.iter() {
            let undo = board.make_move(*each_move, zobrist_constants);
            let successor_hash= board.hash_val();
            child_hashes.push( successor_hash);
            board.unmake_move(undo);

            let retrieved_node = if parent_visits > 1 {
                match node_store.get(&successor_hash) {
                    Some(val) => {
                        UMCTSNode{
                            value_lower: alpha.max(val.value_lower),
                            value_upper: beta.min(val.value_upper),
                            best_known: std::u8::MAX,
                            visits: val.visits,
                            value: val.value,
                        }
                    },
                    None => {UMCTSNode::new_windowed(alpha,beta)},

                }
            } else {
                UMCTSNode::new_windowed(alpha,beta)
            };


            if retrieved_node.value_lower < retrieved_node.value_upper {
                candidate_moves.push((each_move, retrieved_node));
            };
        }

        (candidate_moves, child_hashes)
    }

    pub fn advance_rollout<T: Hash + Eq + Copy + Display, Unmove, HashConstants,
        BoardState: AlphaBetaTTCompatible<T,Unmove,HashConstants> + Clone, S: BuildHasher> (
        board_queue: &mut Vec<UMCTSQueueElement<T,Unmove>>,
        board: &mut BoardState, legal_moves: &[Vec<GameMove>],
        alpha: i8, beta: i8,
        node_store: &mut HashMap<T,UMCTSNode,S>,
        zobrist_constants: &HashConstants,
    ) -> Option<(i8, i8)> {

        let board_hash = board.hash_val();

        let a_node = node_store.entry(board_hash).or_insert_with(UMCTSNode::default);
        a_node.visits += 1;

        let board_visits = a_node.visits;
        let board_lower = a_node.value_lower;
        let board_upper = a_node.value_upper;

        let alpha = max(board_lower,alpha);
        let beta = min(board_upper,beta);

        let x_to_move = board.x_to_move();
        let all_moves = board.get_legal_moves(legal_moves);

        eprintln!("all_moves in order are:\n{:?}\n", all_moves);
        let (candidate_moves, child_hashes) = get_successors(
            &all_moves, board, node_store, zobrist_constants,
            alpha, beta, board_visits
        );

        eprintln!("candidate moves in some order are:\n{:?}\n",candidate_moves);
        let opt_tuple = get_optimal(
            &candidate_moves, x_to_move, board_visits
        );
        eprintln!("{:?}", opt_tuple);

        match opt_tuple {
            Some((move_ref, opt_node)) => {
                let UMCTSNode{value_lower: best_alpha, value_upper: best_beta, ..}= opt_node;
                let undo = board.make_move(**move_ref, zobrist_constants);
                let new_queue_element = UMCTSQueueElement {
                    undo,
                    board_hash,
                    child_hashes,
                    x_to_move,
                };
                board_queue.push(new_queue_element);
                Some((*best_alpha, *best_beta))
            },
            None => {
                None
            },
        }

    }

    pub fn reverse_rollout<T: Hash + Eq + Copy + Display, Unmove, HashConstants,
        BoardState: AlphaBetaTTCompatible<T,Unmove,HashConstants> + Clone, S: BuildHasher> (
        rollout_result: i8,
        board_queue: &mut Vec<UMCTSQueueElement<T,Unmove>>,
        board: &mut BoardState,
        node_store: &mut HashMap<T,UMCTSNode, S>
    ) {

        let mut update_bounds = true;

        while let Some(queue_element) = board_queue.pop(){

            let UMCTSQueueElement {
                undo, board_hash, child_hashes, x_to_move
            } = queue_element;

            let bounds = if update_bounds & x_to_move {
                let mut lower_bound = std::i8::MIN;
                let mut upper_bound = std::i8::MIN;
                for child_hash in child_hashes.iter() {
                    let (child_lower,child_upper) = match node_store.get(child_hash) {
                        Some(stored_node) => (stored_node.value_lower, stored_node.value_upper),
                        None => (std::i8::MIN, std::i8::MAX),
                    };
                    lower_bound = max(lower_bound, child_lower);
                    upper_bound = max(upper_bound, child_upper);

                    if lower_bound==std::i8::MAX {break;};
                }
                (lower_bound, upper_bound)

            } else if update_bounds & !x_to_move {
                let mut lower_bound = std::i8::MAX;
                let mut upper_bound = std::i8::MAX;
                for child_hash in child_hashes.iter() {
                    let (child_lower,child_upper) = match node_store.get(child_hash) {
                        Some(stored_node) => (stored_node.value_lower, stored_node.value_upper),
                        None => (std::i8::MIN, std::i8::MAX),
                    };
                    lower_bound = min(lower_bound, child_lower);
                    upper_bound = min(upper_bound, child_upper);

                    if upper_bound == std::i8::MIN {break;};
                }
                (lower_bound, upper_bound)
            } else {
                (std::i8::MIN, std::i8::MAX)
            };

            if bounds == (std::i8::MIN, std::i8::MAX) {update_bounds = false;}

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

    pub fn alpha_beta_search<T: Hash + Eq + Copy, Unmove, HashConstants, S: BuildHasher> (
        board: &mut impl AlphaBetaTTCompatible<T, Unmove, HashConstants>, depth: u8, alpha: i8, beta: i8,
        transpositions: &mut HashMap<T, HashState, S>,
        legal_moves: &[Vec<GameMove>],
        zobrist_constants: &HashConstants
        ) -> (i8, i8, u8, bool) {
        // implemented based on pseudo-code from from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
//        eprintln!("starting an alpha beta call");
        let node_hash = board.hash_val();
        let mut terminal_found = true;


        let HashState{ value_lower: stored_lower, value_upper: stored_upper,
                    depth: _stored_depth, move_index: stored_move_index
        } = match transpositions.get(&node_hash) {
            Some(hashed_node) if hashed_node.depth >= depth => {
//                eprintln!("restoring transposition eval:\n{:?}",hashed_node);
                terminal_found &= hashed_node.depth==std::u8::MAX;
                hashed_node
            },
            _ => {
                &HashState{
                    value_lower: std::i8::MIN, value_upper: std::i8::MAX,
                    depth: 0, move_index: std::u8::MAX,
                }
            },
        };

        let mut move_index = min(std::u8::MAX, *stored_move_index);
        let mut node_alpha = max(alpha,*stored_lower);
        let mut node_beta  = min(beta, *stored_upper);


        let (new_lower, new_upper) = match board.get_win_state() {
            GameResult::P1Win => {
                (std::i8::MAX, std::i8::MAX)
            },
            GameResult::P2Win => {
                (std::i8::MIN, std::i8::MIN)
            },
            GameResult::Tie => {
                (0,0)
            },
            GameResult::Unresolved if depth == 0 => {
                terminal_found = false;
                (board.score(),board.score())
            },
            GameResult::Unresolved => {
//                eprintln!("board unresolved:\n{}", board.display());
                let (selector, opt_val, start_val): (fn(i8, i8) -> i8, i8, i8) = if board.x_to_move() {
                    (MAX, std::i8::MAX, std::i8::MIN)
                } else{
                    (MIN, std::i8::MIN, std::i8::MAX)
                };
                let mut new_lower = start_val;
                let mut new_upper = start_val;

                let moves = board.get_legal_moves(legal_moves);

                if move_index == 255 {
                    move_index = moves[0].move_index;
                }

                if node_alpha >= node_beta {
//                    terminal_found = true;
                    return(*stored_lower, *stored_upper, move_index, terminal_found);
                } else {
                    for each_move in moves.iter() {
                        assert_eq!(board.x_to_move(),each_move.x_to_move, "side to move mismatch in alpha beta search");
                        let unmake_instructions = board.make_move(*each_move, zobrist_constants);
                        let (candidate_lower, candidate_upper, _move_index, _terminal_search) = {
//                        eprintln!("looking at board after move at depth {}. x to play? {}:\n{}",
//                                  depth, board.x_to_move(), board.display());
                            alpha_beta_search(
                                board, depth - 1, node_alpha, node_beta, transpositions, legal_moves,
                                zobrist_constants
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


                        if (new_lower==opt_val)&(new_upper==opt_val) {
                            move_index = each_move.move_index;
                            break;
                        }
                    };
                }

                (new_lower, new_upper)
            },
        };

        let insert_depth = if terminal_found {std::u8::MAX} else {depth};
        let inserted_state = HashState{
            value_lower: new_lower, value_upper: new_upper,
            depth: insert_depth, move_index
        };

        transpositions.insert(node_hash, inserted_state);
//        eprintln!("returning from alpha beta");
        (new_lower, new_upper, move_index, terminal_found)
    }
}

pub mod ttt_trait_solving {
    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult, OrderedFloat};
    use itertools::Itertools;
    use std::collections::{VecDeque, HashMap};

    // tictactoe constants
    const TTT_BOARD_WIDTH: usize = 3;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const CAT: u16 = (1 << 9) - 1;
    const REACHABLE_BOARD_COUNT: usize = 19_683; // At most 2**3 boards should exist

    #[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
    pub struct TTTTerminal {
        pub terminal_value: i8,
        pub depth_to_terminal: u8,
    }

    impl Default for TTTTerminal {
        fn default() -> TTTTerminal {
            TTTTerminal{
                terminal_value: 0,
                depth_to_terminal: 9,
            }
        }
    }

    #[derive(Debug,Clone)]
    pub struct TTTState {
        pub xbb: u16,
        pub obb: u16,
        pub x_to_move: bool,
    }

    #[derive(Debug,Clone,Ord, PartialOrd, Eq, PartialEq, Hash)]
    pub struct TTTBoard {
        pub xbb: u16,
        pub obb: u16
    }


    pub struct TTTUnmove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    pub struct TTTZobrist {
        //I'm not ACTUALLY implementing Zobrist constants for normal tictactoe...
    }

    impl TTTState {
        pub fn to_index(&self) -> u32 {
            (u32::from(self.xbb) << 10) | (u32::from(self.obb) << 1) | (if self.x_to_move {1u32} else {0u32})
        }

        fn get_legal_moves_no_lookup(&self) -> Vec<GameMove> {
            let occupied_bb = self.xbb | self.obb;

            MOVE_ORDERING.iter().filter_map(|a_move_index| {
                match occupied_bb & (1u16 << a_move_index) {
                    0 => Some(GameMove { x_to_move: self.x_to_move, move_index: *a_move_index }),
                    _nonzero => {
                        None
                    },
                }
            }).collect()
        }

        pub fn get_move_ordering_array() -> Vec<Vec<GameMove>> {

            let mut move_vec: Vec<Vec<GameMove>> = vec![Vec::with_capacity(0); 524_288];
            let mut terminal_conditions = vec![TTTTerminal::default(); 524_288];

            let zobrist_constants = TTTZobrist{};
//            let move_lookup = Vec::new();

            let xbb_iter = (0..512u16).rev();
            let obb_iter = (0..512u16).rev();
            let side_iter = [true,false].iter();

            let bb_iter = xbb_iter.cartesian_product(obb_iter);
            let sorted_bbs = bb_iter.sorted_by_key(|(a,b)| a.count_ones()+b.count_ones());

            for ((xbb,obb),x_to_move) in sorted_bbs.rev().cartesian_product(side_iter) {
                let mut init_board = TTTState {
                    xbb,
                    obb,
                    x_to_move: *x_to_move,
                };
                if init_board.is_valid() {
                    let board_index = init_board.to_index();

                    let (ttt_terminal, ordered_moves) = if init_board.get_win_state() != GameResult::Unresolved {
                        (TTTTerminal {
                            depth_to_terminal: 0,
                            terminal_value: init_board.score(),
                        },
                         Vec::with_capacity(0))
                    } else {
                        let legal_moves = init_board.get_legal_moves_no_lookup();

                        let mut mapped_moves: Vec<(OrderedFloat, u32, &TTTTerminal, &GameMove)> = legal_moves.iter().map(|a_move| {
                            let undo = init_board.make_move(*a_move,&zobrist_constants);
                            let new_index = init_board.to_index();
                            let restored_terminal = terminal_conditions.get(new_index as usize).unwrap();
                            let point_rate = OrderedFloat(
                                (restored_terminal.terminal_value as f32)/((restored_terminal.depth_to_terminal+1) as f32));
                            let result_tuple = (point_rate, new_index, restored_terminal,a_move);
                            init_board.unmake_move(undo);
                            result_tuple
                        }).collect();
                        mapped_moves.sort();

                        if *x_to_move {mapped_moves.reverse()};

                        let ordered_moves = mapped_moves.iter().map(
                            |(_,_,_,&a_move)| (a_move)).collect();

                        let optimal_terminal = match mapped_moves.first() {
                            Some((_,_,terminal, _a_move)) => {*terminal},
                            None => {panic!("mapped_moves should never be non-empty")},
                        };

                        (TTTTerminal {
                            depth_to_terminal: optimal_terminal.depth_to_terminal+1,
                            terminal_value: optimal_terminal.terminal_value,
                        }, ordered_moves)
                    };

                    match terminal_conditions.get_mut(board_index as usize) {
                        Some(terminal_value) => {
                            *terminal_value = ttt_terminal;
                        },
                        None => {
                            eprintln!("Error finding terminal condition");
                            panic!("attempted to get a ");
                        }
                    };

                    match move_vec.get_mut(board_index as usize) {
                        Some(move_list) => {
                            *move_list = ordered_moves;
                        },
                        None => {
                            eprintln!("Error finding terminal condition");
                            panic!("attempted to get a ");
                        }
                    };

                } else {
//                    invalid_count += 1;
                }

            }

            move_vec
        }

    }

    impl TTTBoard {
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

        pub fn get_legal_moves_no_lookup(&self, x_to_move:bool) -> Vec<GameMove> {
            let occupied_bb = self.xbb | self.obb;

            MOVE_ORDERING.iter().filter_map(|a_move_index| {
                match occupied_bb & (1u16 << a_move_index) {
                    0 => Some(GameMove { x_to_move: x_to_move, move_index: *a_move_index }),
                    _nonzero => {
                        None
                    },
                }
            }).collect()
        }

        pub fn make_move(&mut self, a_move: GameMove, _zobrist_constants: &TTTZobrist) -> TTTUnmove {
            let unmove = TTTUnmove { x_to_move: a_move.x_to_move, move_index: a_move.move_index };
            match a_move {
                GameMove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                },
                GameMove { x_to_move: false, move_index } => {
                    self.obb ^= 1u16 << move_index;
                },
            }
            unmove
        }

        pub fn make_boards() -> Vec<TTTBoard> {
            let mut expandable_boards: VecDeque<_> = VecDeque::with_capacity(REACHABLE_BOARD_COUNT);
            let mut board_index = 0usize;
            let mut numbered_boards = Vec::with_capacity(REACHABLE_BOARD_COUNT);
            let mut number_lookup = HashMap::with_capacity(REACHABLE_BOARD_COUNT*2);

            let init_board = TTTBoard{xbb:0, obb:0};
            expandable_boards.push_back(init_board);

            while let Some(board) = expandable_boards.pop_front() {
                let legal_x_moves = board.get_legal_moves_no_lookup(true);
                let legal_o_moves = board.get_legal_moves_no_lookup(false);
                let all_moves = legal_x_moves.iter().chain(legal_o_moves.iter());
                if GameResult::Unresolved == board.get_win_state() {
                    for each_move in all_moves {
                        let mut new_board: TTTBoard = board.clone();
                        new_board.make_move(each_move.clone(), &TTTZobrist{});
                        if !number_lookup.contains_key(&new_board) {
                            number_lookup.insert(new_board.clone(), board_index);
                            numbered_boards.push(new_board.clone());
                            expandable_boards.push_back(new_board.clone());
                            board_index+=1;
                        }
                    }
                }
            }
            numbered_boards
        }
    }

    impl AlphaBetaTTCompatible<u32, TTTUnmove, TTTZobrist> for TTTState {
        fn hash_val(&self) -> u32 {
            let base_hash = (u32::from(self.xbb)) << 16 | u32::from(self.obb);

            if self.x_to_move {base_hash} else {!base_hash}
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

        fn make_move(&mut self, a_move: GameMove, _zobrist_constants: &TTTZobrist) -> TTTUnmove {
//            pub x_to_move: bool,
//            pub move_index: u8,
            let unmove = TTTUnmove { x_to_move: self.x_to_move, move_index: a_move.move_index };

            match a_move {
                GameMove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = false;
                },
                GameMove { x_to_move: false, move_index } => {
                    self.obb ^= 1u16 << move_index;
                    self.x_to_move = true;
                },
            }

            unmove
        }

        fn unmake_move(&mut self, unmove: TTTUnmove) {
            match unmove {
                TTTUnmove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = true;
                },
                TTTUnmove { x_to_move: false, move_index } => {
                    self.obb ^= 1u16 << move_index;
                    self.x_to_move = false;
                },
            }
        }

        fn score(&self) -> i8 {
            match self.get_win_state() {
                GameResult::P1Win => { std::i8::MAX },
                GameResult::P2Win => { std::i8::MIN },
                GameResult::Tie => { 0 },
                GameResult::Unresolved if self.x_to_move => { 1 },
                GameResult::Unresolved => { -1 },
            }
        }

        fn get_legal_moves(&self, _legal_lookup: &[Vec<GameMove>]) -> Vec<GameMove> {
            //TODO: This should probably use the legal lookup directly
            let occupied_bb = self.xbb | self.obb;

            MOVE_ORDERING.iter().filter_map(|a_move_index| {
                match occupied_bb & (1u16 << a_move_index) {
                    0 => Some(GameMove { x_to_move: self.x_to_move, move_index: *a_move_index }),
                    _nonzero => {
                        None
                    },
                }
            }).collect()
        }

        fn display(&self) -> String {
            //        let mut board_rep = String::new();
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

        fn is_valid(&self) -> bool {
            // there are more ways to invalidate a board than are checked here
            (self.obb & self.xbb) == 0
        }
    }
}

pub mod uttt_trait_solving {
    use std::iter::FromIterator;
    use rand::{rngs, SeedableRng, Rng};
    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult};
    use crate::ttt_trait_solving::TTTState;

    // tictactoe constants
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const CAT: u16 = (1 << 9) - 1;
    const ZOBRIST_SEED: u64 = 0x95_62_9A_0C_B8_00_E9_41u64; //generated from random.org
    const HASH_DEFAULT: u64 = 0x8e_6c_65_b8_2e_fd_36_70u64; //generated from random.org

    #[derive(Debug, Hash, Clone, PartialEq, Eq)]
    pub struct UTTTBoard {
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

        SubBoardAndSquare { sub_square_index: sub_square, sub_board_index: sub_board }
    }

    pub fn parse_index_from_super_coords(row_index: u8, col_index: u8) -> u8 {
        let board_row = row_index/3;
        let subboard_row = row_index % 3;

        let board_col = col_index/3;
        let subboard_col = col_index % 3;
        // Returns the sub board [0,8] and sub square [0-8]

        let subboard_index = 3* board_row + board_col;
        let subsquare_index = 3* subboard_row + subboard_col;

        9*subboard_index+subsquare_index
    }

    #[derive(Clone, Copy)]
    pub struct UTTTUnmove {
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

    impl Default for UTTTBoard {
        fn default() -> UTTTBoard {
            UTTTBoard {
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

    impl UTTTBoard {
        pub fn get_tied_mask(&self) -> u16 {
            self.xbb.iter().zip(
                self.obb.iter()).enumerate().filter_map(
                |(index, (xbb, obb))| match xbb|obb {
                    CAT => Some(1u16<<index),
                    _ => None,
                }).fold(0u16, |current,next| current|next)
        }

        pub fn get_subboard_moves(&self, subboard_index: usize,
              legality_lookup: &[Vec<GameMove>], side_to_move: bool) -> Vec<GameMove> {

            let subboard = TTTState{
                xbb: self.xbb[subboard_index],
                obb: self.obb[subboard_index],
                x_to_move: side_to_move,
            };

            let lookup_index = subboard.to_index();
            match legality_lookup.get(lookup_index as usize) {
                Some(move_vec) => {
                    move_vec.iter(
                    ).map(|GameMove{x_to_move: _side, move_index: a_move_index}|
                        GameMove{x_to_move:side_to_move, move_index: a_move_index+ 9*subboard_index as u8}
                    ).collect()
                },
                None => panic!("It shouldn't be possible for subboard moves not to find a move vector"),
            }
        }


        pub fn get_move_ordering_array() -> Vec<Vec<GameMove>> {
            TTTState::get_move_ordering_array()
        }

        pub fn get_zobrist_constants() -> ZobristConstants {
            // zobrist hashing using a seedable RNG, and a seed from random.org
            let mut seeded = rngs::StdRng::seed_from_u64(ZOBRIST_SEED);

            let x_consts = {
                let mut empty = [0u64; 81];
                for element in empty.iter_mut() {
                    *element = seeded.gen::<u64>();
                };
                empty
            };

            let o_consts = {
                let mut empty = [0u64; 81];
                for element in empty.iter_mut() {
                    *element = seeded.gen::<u64>();
                };
                empty
            };

            let x_to_move_constant = seeded.gen::<u64>();

            ZobristConstants{
                x_constants: x_consts,
                o_constants: o_consts,
                x_to_move: x_to_move_constant,
            }

        }
    }

    impl AlphaBetaTTCompatible<u64, UTTTUnmove, ZobristConstants> for UTTTBoard {
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
                let new_mask = self.xbb.iter().zip(
                    self.obb.iter()).enumerate().filter_map(
                    |(index, (xbb, obb))| match xbb|obb {
                        CAT => Some(1u16<<index),
                        _ => None,
                    }).fold(0u16, |current,next| current|next);
                if new_mask | self.o_supers | self.x_supers == CAT {
                    let x_won = self.x_supers.count_ones() as i8;
                    let o_won = self.o_supers.count_ones() as i8;
                    let heuristic_score = x_won-o_won;
                    match heuristic_score {
                        0 => GameResult::Tie,
                        value if value>0 => GameResult::P1Win,
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

        fn make_move(&mut self, a_move: GameMove, zobrist_constants: &ZobristConstants) -> UTTTUnmove {
            let GameMove { x_to_move, move_index: a_move_index } = a_move;

            let original_hash_state = self.hash_state;

            let hash_update: u64 = if x_to_move {
                zobrist_constants.x_to_move ^ zobrist_constants.x_constants[a_move_index as usize]
            } else {
                zobrist_constants.x_to_move ^ zobrist_constants.o_constants[a_move_index as usize]
            };
            self.hash_state ^= hash_update;

            let SubBoardAndSquare { sub_board_index, sub_square_index } = parse_index(a_move_index);
//            eprintln!("{} {}", sub_board_index, sub_square_index);

            let to_move_reset = x_to_move;
            let move_to_unmask = a_move_index;
            let mut a_super_unmask = 0;
            let sub_board_reset = self.last_super_index;

            match a_move {
                GameMove { x_to_move: true, .. } => {
                    self.xbb[sub_board_index] ^= 1<<sub_square_index;
                    let result_board = self.xbb[sub_board_index];

                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                    if is_sub_win {
                        let super_mask = 1u16 << sub_board_index;
                        self.x_supers |= super_mask;
                        a_super_unmask = super_mask;
                    }
                    self.last_super_index = sub_square_index as u8;
                    self.x_to_move = false;
                },
                GameMove { x_to_move: false, .. } => {
                    self.obb[sub_board_index] ^= 1<<sub_square_index;
                    let result_board = self.obb[sub_board_index];

                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                    if is_sub_win {
                        let super_mask = 1u16 << sub_board_index;
                        self.o_supers |= 1 << sub_board_index;
                        a_super_unmask = super_mask;
                    }
                    self.last_super_index = sub_square_index as u8;
                    self.x_to_move = true;
                },
            }

            UTTTUnmove {
                x_to_move: to_move_reset,
                move_index: move_to_unmask,
                super_unmask: a_super_unmask,
                last_super: sub_board_reset,
                hash_reset: original_hash_state,
            }

        }

        fn unmake_move(&mut self, unmove: UTTTUnmove) {
            let UTTTUnmove {
                x_to_move, move_index,
                super_unmask, last_super,
                hash_reset
            } = unmove;

            let SubBoardAndSquare{sub_board_index, sub_square_index} = parse_index(move_index);

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
                GameResult::P1Win => { std::i8::MAX },
                GameResult::P2Win => { std::i8::MIN },
                GameResult::Tie => { 0i8 },
                GameResult::Unresolved => {x_won-o_won},
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
            if (self.last_super_index>=9) | ((complete_bbs & (1 << self.last_super_index)) != 0) {
                let mut moves = Vec::with_capacity(81);
                for subboard_index in 0..9usize {
                    if (complete_bbs & (1 << subboard_index)) == 0 {
                        moves.append(&mut self.get_subboard_moves(
                            subboard_index, legality_lookup, self.x_to_move))
                    }
                }
                moves
            } else {
                self.get_subboard_moves(
                    self.last_super_index as usize, legality_lookup, self.x_to_move)
            }

        }

        fn display(&self) -> String {
            let mut rows = [['.';9]; 9];

            for (row_index, row) in rows.iter_mut().enumerate().take(9usize) {
                for (col_index, an_item) in row.iter_mut().enumerate().take(9usize) {
                    let subboard_row = row_index % 3;
                    let subboard_col = col_index % 3;

                    let superboard_row = row_index/3;
                    let superboard_col = col_index/3;

                    let subboard = 3*superboard_row + superboard_col;
                    let subboard_square_mask = 1u16 <<(3*subboard_row+subboard_col);
                    if self.xbb[subboard] & subboard_square_mask != 0 {
                        *an_item = 'X';
                    } else if self.obb[subboard] & subboard_square_mask != 0 {
                        *an_item = 'O';
                    }
                }
            }

            rows.iter().map(
                |a_row| {
                    let mut row = String::from_iter(a_row.iter());
                    row.push('\n');
                    row
                }).collect()
        }

        fn is_valid(&self) -> bool {
            self.obb.iter().zip(self.xbb.iter()).all(|(o_subboard, x_subboard)| {
                (o_subboard & x_subboard) == 0
            })
        }
    }
}


macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}
pub fn main() {
    // game loop
    use crate::uttt_trait_solving::{UTTTBoard,parse_index_from_super_coords};
    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult, advance_rollout,
                              reverse_rollout, uct_value, IdentityHasherBuilder
    };
    use std::time;
    use std::collections::HashMap;

    let mut uttt_board = UTTTBoard {
        xbb: [0u16 ;9],
        obb: [0u16 ;9],
        last_super_index: 9,
        o_supers: 0,
        x_supers: 0,
        x_to_move: true,
        hash_state: 0,
    };

    let zobrist_constants = UTTTBoard::get_zobrist_constants();
    let mut turn_count = 0;

    let mut unmove_queue = Vec::with_capacity(81);
    let mut node_store= HashMap::with_capacity_and_hasher(2_000_000, IdentityHasherBuilder{});
    let mut legal_moves = Vec::new();

    loop {
        // **************************************************************
        // begin supplied boiler plate
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
        // end supplied boiler plate
        // ***************************************************************
        // setting start of turn timer

        // **************************************************************
        let turn_start = time::Instant::now();
        let time_limit = if turn_count == 0 {
            legal_moves = UTTTBoard::get_move_ordering_array();
            990_000
        } else {
            90_000
        };

        let mut rollout_count = 0;

        let best_move_index = match (opponent_row, opponent_col) {
            (-1, -1) => {

                let default_first = GameMove { move_index: 40, x_to_move: uttt_board.x_to_move() };
                uttt_board.make_move(default_first, &zobrist_constants);

                while turn_start.elapsed().as_micros() < time_limit {
                    let mut search_alpha = std::i8::MIN;
                    let mut search_beta = std::i8::MAX;
//                    let mut rollout_result = 0;

                    while let Some(new_alpha_beta) = advance_rollout(&mut unmove_queue,
                            &mut uttt_board, &legal_moves, search_alpha, search_beta, &mut node_store, &zobrist_constants){
                        let (new_alpha, new_beta) = new_alpha_beta;
                        search_alpha = new_alpha;
                        search_beta = new_beta;
                    };

                    let final_hash = uttt_board.hash_val();
                    let terminal_node = node_store.get_mut(&final_hash).unwrap();

                    let rollout_result = match uttt_board.get_win_state() {
                        GameResult::P1Win => {
                            let rollout_result = std::i8::MAX;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        },
                        GameResult::P2Win => {
                            let rollout_result = std::i8::MIN;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        },
                        GameResult::Tie   => {
                            let rollout_result = 0;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        },
                        _ => {
                            //TODO: P3: Figure out what on earth this means!
                            search_alpha
                        },
                    };
                    reverse_rollout(rollout_result, &mut unmove_queue, &mut uttt_board,
                                    &mut node_store
                    );
                    rollout_count +=1;
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
                            eprintln!("node avg value: {}",(node.value as f32)/(node.visits as f32));
                            eprintln!("node visits: {}",node.visits);
                            eprintln!("move index: {}", each_move.move_index);
                        }
                        None => {
                            eprintln!("no stored statistics found for hash value: {}",this_hash);
                        }
                    }
                }

                40
            },
            (row, col) => {
                let opp_move_index = parse_index_from_super_coords(row as u8,col as u8);
                let opp_move = GameMove { move_index: opp_move_index, x_to_move: uttt_board.x_to_move() };
                uttt_board.make_move(opp_move, &zobrist_constants);
                let x_to_move = uttt_board.x_to_move();
                eprintln!("board representation pre-move:\n{:?}", uttt_board);

//                let mut rollout_result = 0;
//                let board_hash = uttt_board.hash_val();

                while turn_start.elapsed().as_micros() < time_limit {
                    let mut search_alpha = std::i8::MIN;
                    let mut search_beta = std::i8::MAX;

                    while let Some(new_alpha_beta) = advance_rollout(&mut unmove_queue,
                            &mut uttt_board, &legal_moves, search_alpha, search_beta, &mut node_store, &zobrist_constants){
                        let (new_alpha, new_beta) = new_alpha_beta;
                        search_alpha = new_alpha;
                        search_beta = new_beta;
                    };

                    let final_hash = uttt_board.hash_val();
                    let terminal_node = node_store.get_mut(&final_hash).unwrap();

                    let rollout_result = match uttt_board.get_win_state() {
                        GameResult::P1Win => {
                            let rollout_result = std::i8::MAX;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        },
                        GameResult::P2Win => {
                            let rollout_result = std::i8::MIN;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        },
                        GameResult::Tie   => {
                            let rollout_result = 0;
                            terminal_node.value_lower = rollout_result;
                            terminal_node.value_upper = rollout_result;
                            rollout_result
                        },
                        _ => {
                            //TODO: P3: Figure out what on earth this means!
                            search_alpha
                        },
                    };
                    reverse_rollout(rollout_result, &mut unmove_queue, &mut uttt_board,
                                     &mut node_store
                    );
                    rollout_count +=1;
                }

                let moves = uttt_board.get_legal_moves(&legal_moves);

                let mut mut_board = uttt_board.clone();

                let best_move = moves.iter().max_by_key(|a_move| {
                    let undo = mut_board.make_move(**a_move, &zobrist_constants);
                    let this_hash = mut_board.hash_val();
                    mut_board.unmake_move(undo);
                    let child_stats = node_store.get(&this_hash);
                    match child_stats {
                        Some(node) => node.visits,
                        None => 0,
                    }
                }).unwrap();

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
                            eprintln!("node avg value: {}",(node.value as f32)/(node.visits as f32));
                            eprintln!("node visits: {}",node.visits);
                            eprintln!("move index: {}", each_move.move_index);
                            eprintln!("uct value: {:?}",
                                      uct_value(
                                          node.value, rollout_count,
                                          node.visits, x_to_move
                                      )
                            );
//                            value node visits action visits side to move
                        }
                        None => {
                            eprintln!("no stored statistics found for hash value: {}",this_hash);
                        }
                    }
                    eprintln!();
                }
                uttt_board.make_move(*best_move, &zobrist_constants);
                best_move.move_index
            },
        };
        let best_subboard_index = best_move_index/9;
        let best_sub_square_index = best_move_index%9;

        let base_row = best_subboard_index/3;
        let base_col = best_subboard_index%3;

        let final_row = base_row*3 + best_sub_square_index/3;
        let final_column = base_col*3 + best_sub_square_index%3;

        turn_count +=1;
        eprintln!("elapsed microseconds: {}",turn_start.elapsed().as_micros());
        eprintln!("completed rollouts: {}", rollout_count);
        eprintln!("board representation post-move:\n{:?}", uttt_board);
        eprintln!("table size: {}", node_store.len());
        println!("{} {}",final_row, final_column);
    }
}