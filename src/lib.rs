//TODO: Generally increase code quality by reducing the number of clippy lints and compiler warnings
pub mod game_solving {
    use std::collections::HashMap;
    use std::hash::{Hash, BuildHasher, Hasher};
    use std::cmp::{max,min,Ordering};
    use std::fmt::Display;

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

    #[derive(Debug, Copy, Clone)]
    pub struct GameMove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    #[derive(Clone, Debug)]
    pub struct MCTSAction {
        pub game_move: GameMove,
        pub visits: u32,
        pub value: i32,
    }

    #[derive(Clone)]
    pub struct UMCTSNode {
        pub value_lower: i8,
        pub value_upper: i8,
        pub visits: u32,
        pub value: i32,
    }

    impl Default for UMCTSNode {
        fn default() -> UMCTSNode {
            UMCTSNode {
                value_lower: std::i8::MIN,
                value_upper: std::i8::MAX,
                visits: 0,
                value: 0,
            }
        }
    }

    pub trait AlphaBetaTTCompatible<T: Hash + Eq + Copy, Unmove, HashConstants> {
        fn hash_val(&self) -> T;
        fn get_win_state(&self) -> GameResult;
        fn x_to_move(&self) -> bool;
        fn make_move(&mut self, a_move: GameMove, zobrist_constants: &HashConstants) -> Unmove;
        fn unmake_move(&mut self, a_move: Unmove);
        fn score(&self) -> i8;
        fn get_legal_moves(&self, legal_lookup: &[Vec<Vec<Vec<GameMove>>>]) -> Vec<GameMove>;
        fn display(&self) -> String;
        fn is_valid(&self) -> bool;
    }


    const MCTS_LAMBDA : f32 = 141.4;

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
        fn finish(&self) -> u64 {
            self.hash_val
        }

        fn write(&mut self, bytes: &[u8]) {
            let start_val = self.hash_val;
            let result = bytes.iter().fold(
                start_val,
                |state, next| (state>>8)|(u64::from(*next )<<56)
            );
            self.hash_val = result;
            panic!("please use write_u64 exclusively for IdentityHasher.");
        }

        fn write_u64(&mut self, val: u64) {
            self.hash_val = val;
        }
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

//    pub fn retrieve_successor_statistics(node_store: &mut HashMap<T,UMCTSNode,S>,) -> (i8,i8, u32, i32) {
//
//    }

//    #[inline(never)]
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
                            visits: val.visits,
                            value: val.value,
                        }
                    },
                    None => {
                        UMCTSNode{
                            value_lower: alpha,
                            value_upper: beta,
                            visits: 0,
                            value: 0,
                        }
                    },
                }
            } else {
                UMCTSNode{
                    value_lower: alpha,
                    value_upper: beta,
                    visits: 0,
                    value: 0,
                }
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
        board: &mut BoardState, legal_moves: &[Vec<Vec<Vec<GameMove>>>],
        alpha: i8, beta: i8,
        node_store: &mut HashMap<T,UMCTSNode,S>,
        zobrist_constants: &HashConstants,
    ) -> Option<(i8, i8)> {

        let board_hash = board.hash_val();

        let a_node = node_store.entry(board_hash).or_insert(
            {
                UMCTSNode{
                    value_lower: std::i8::MIN,
                    value_upper: std::i8::MAX,
                    visits: 0,
                    value: 0,
                }
            });
        a_node.visits += 1;

        let board_visits = a_node.visits;
        let board_lower = a_node.value_lower;
        let board_upper = a_node.value_upper;

        let alpha = max(board_lower,alpha);
        let beta = min(board_upper,beta);

        let x_to_move = board.x_to_move();
        let all_moves = board.get_legal_moves(legal_moves);


        let (candidate_moves, child_hashes) = get_successors(
            &all_moves, board, node_store, zobrist_constants,
            alpha, beta, board_visits
        );

        let opt_tuple = get_optimal(
            &candidate_moves, x_to_move, board_visits
        );


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
            cached_node.value += i32::from(rollout_result);
            board.unmake_move(undo);
        }
    }

    pub fn alpha_beta_search<T: Hash + Eq + Copy, Unmove, HashConstants, S: BuildHasher> (
        board: &mut impl AlphaBetaTTCompatible<T, Unmove, HashConstants>, depth: u8, alpha: i8, beta: i8,
        transpositions: &mut HashMap<T, HashState, S>,
        legal_moves: &[Vec<Vec<Vec<GameMove>>>],
        zobrist_constants: &HashConstants
        ) -> (i8, i8, u8, bool) {
        // implemented based on pseudo-code from from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
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

        (new_lower, new_upper, move_index, terminal_found)
    }
}

pub mod ttt_trait_solving {
    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult};

    // tictactoe constants
    const TTT_BOARD_WIDTH: usize = 3;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    //    const MOVE_ORDERING: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const CAT: u16 = (1 << 9) - 1;

    #[derive(Debug,Clone)]
    pub struct TTTBoard {
        pub xbb: u16,
        pub obb: u16,
        pub x_to_move: bool,
    }


    pub struct TTTUnmove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    struct TTTZobrist {
        //I'm not ACTUALLY implementing Zobrist constants for normal tictactoe...
    }

//    impl TTTBoard {
//        fn get_zobrist_constants() -> TTTZobrist {
//            TTTZobrist{}
//        }
//    }

    impl AlphaBetaTTCompatible<u32, TTTUnmove, TTTZobrist> for TTTBoard {
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

        fn get_legal_moves(&self, _legal_lookup: &[Vec<Vec<Vec<GameMove>>>]) -> Vec<GameMove> {
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
            rows.join("\n").to_string()
        }

        fn is_valid(&self) -> bool {
            // there are more ways to invalidate a board than are checked here
            (self.obb & self.xbb) == 0
        }
    }

    #[cfg(test)]
    mod test {
        use std::collections::{HashMap, HashSet};
        use std::time;

        use crate::game_solving::{
            alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult};

        use super::{RESULT_MASKS, TTTBoard, TTTZobrist};
        use crate::uttt_trait_solving::UTTTBoard;

        #[test]
        fn display_test1() {
            let x = TTTBoard {
                xbb: 1 << 8,
                obb: 0 << 8,
                x_to_move: false,
            };
            let board_string = x.display();
            assert_eq!(board_string, "...\n...\n..X");
        }

        #[test]
        fn display_test2() {
            let x = TTTBoard {
                xbb: 1 << 8,
                obb: 1 << 2,
                x_to_move: false,
            };
            let board_string = x.display();
            eprintln!("{}", board_string);
            assert_eq!(board_string, "..O\n...\n..X");
        }


        #[test]
        fn make_and_unmake_move_test_1() {
            let mut x = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };
            let zobrist_constants = TTTZobrist{};

            let board_string = x.display();
            assert_eq!(board_string, "...\n...\n...");
            assert_eq!(x.x_to_move, true);


            //testing an x move
            let ttt_move = GameMove { move_index: 0, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "X..\n...\n...");
            assert_eq!(x.x_to_move, false);

            // testing an x unmake move
            x.unmake_move(unmake_move);
            let board_string = x.display();
            assert_eq!(board_string, "...\n...\n...");
            assert_eq!(x.x_to_move, true);

            //readding an x move
            let ttt_move = GameMove { move_index: 4, x_to_move: true };
            x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "...\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = GameMove { move_index: 1, x_to_move: false };
            let unmake = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, ".O.\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //unmaking an o move
            x.unmake_move(unmake);
            let board_string = x.display();
            assert_eq!(board_string, "...\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = GameMove { move_index: 0, x_to_move: false };
            x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "O..\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //testing an x move
            let ttt_move = GameMove { move_index: 1, x_to_move: true };
            x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "OX.\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = GameMove { move_index: 2, x_to_move: false };
            x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "OXO\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //testing an x move
            let ttt_move = GameMove { move_index: 7, x_to_move: true };
            x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            RESULT_MASKS.iter().for_each(|mask| println! {"mask:\t{}\nbitw:\t{}\nxbb:\t{}\n", mask, x.xbb & mask, x.xbb});

            assert_eq!(board_string, "OXO\n.X.\n.X.");
            assert_eq!(x.x_to_move, false);
            assert_eq!(x.get_win_state(), GameResult::P1Win);
        }

        #[test]
        fn x_win_tests() {
            let test_board = TTTBoard { xbb: 7, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "XXX\n...\n...");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 7 << 3, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "...\nXXX\n...");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 7 << 6, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "...\n...\nXXX");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 73, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "X..\nX..\nX..");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 73 << 1, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), ".X.\n.X.\n.X.");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 73 << 2, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "..X\n..X\n..X");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 273, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "X..\n.X.\n..X");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);

            let test_board = TTTBoard { xbb: 84, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "..X\n.X.\nX..");
            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
        }

        #[test]
        fn o_win_tests() {
            let test_board = TTTBoard { xbb: 0, obb: 7, x_to_move: true };
            assert_eq!(test_board.display(), "OOO\n...\n...");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 7 << 3, x_to_move: true };
            assert_eq!(test_board.display(), "...\nOOO\n...");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 7 << 6, x_to_move: true };
            assert_eq!(test_board.display(), "...\n...\nOOO");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 73, x_to_move: true };
            assert_eq!(test_board.display(), "O..\nO..\nO..");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 73 << 1, x_to_move: true };
            assert_eq!(test_board.display(), ".O.\n.O.\n.O.");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 73 << 2, x_to_move: true };
            assert_eq!(test_board.display(), "..O\n..O\n..O");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 273, x_to_move: true };
            assert_eq!(test_board.display(), "O..\n.O.\n..O");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);

            let test_board = TTTBoard { xbb: 0, obb: 84, x_to_move: true };
            assert_eq!(test_board.display(), "..O\n.O.\nO..");
            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
        }

        #[test]
        fn movegen_test_1() {
            let mut x = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };

            let legal_moves = Vec::new();
            let zobrist_constants = TTTZobrist{};

            while x.get_legal_moves(&legal_moves).len() > 0 {
                let moves = x.get_legal_moves(&legal_moves);
                x.make_move(moves[0], &zobrist_constants);
                eprintln!("{}", x.display());
                assert_eq!(true, x.is_valid());
            }
        }

        #[test]
        fn alpha_beta_hash_test_1a() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 0,
                obb: 44,
                x_to_move: true,
            };
            let legal_moves = Vec::new();
            let zobrist_constants = TTTZobrist{};

            let mut ttable = HashMap::new();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );


//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(a_move, &zobrist_constants);
                eprintln!("{}", ttt_board.display());
            }
            assert_eq!(-128, board_val);

            assert_eq!(ttt_board.display(), "O.O\nOXO\nX.O");
        }

        #[test]
        fn alpha_beta_hash_test_1b() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 1,
                obb: 8,
                x_to_move: true,
            };
            let legal_moves = Vec::new();
            let zobrist_constants = TTTZobrist{};



            let mut ttable = HashMap::new();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );

//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                eprintln!("\n====================\n");
                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}\nevaluation {}", ttt_board.display(), a_move, board_val);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(a_move, &zobrist_constants);
                eprintln!("{}", ttt_board.display());
                assert_eq!(127, board_val);
            }
            eprintln!("game complete:\n{}", ttt_board.display());
            assert_eq!(127, board_val);

            assert_eq!(ttt_board.display(), "X.O\nOX.\n..X");
        }

        #[test]
        fn alpha_beta_hash_test_1c() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 81,
                obb: 12,
                x_to_move: false,
            };
            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = TTTZobrist{};

            let mut ttable = HashMap::new();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );

            let legal_moves = Vec::new();

//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(a_move, &zobrist_constants);
//                eprintln!("{}", ttt_board.display());
                assert_eq!(0, board_val);
            }
            assert_eq!(0, board_val);

            assert_eq!(ttt_board.display(), "XOO\nOXX\nXXO");
        }

        #[test]
        fn alpha_beta_hash_test_1d() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 81,
                obb: 268,
                x_to_move: true,
            };

            let legal_moves = Vec::new();
            let zobrist_constants = TTTZobrist{};

            let mut ttable = HashMap::new();
            let (board_val, board_upper,move_index, found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );



//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(a_move, &zobrist_constants);
                eprintln!("{}", ttt_board.display());
                assert_eq!(0, board_val);
            }
            assert_eq!(0, board_val);

            assert_eq!(ttt_board.display(), "XOO\nOXX\nXXO");
        }


        #[test]
        fn alpha_beta_hash_test_2() {
            let mut ttt_board = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };
            let zobrist_constants = TTTZobrist{};

            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable
//            );
            let legal_moves = Vec::new();

            assert_eq!(
                (0, 0, 4, true),
                alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN,
                    std::i8::MAX, &mut ttable, &legal_moves, &zobrist_constants)
            );


            let now_instant = time::Instant::now();
            while ttt_board.get_legal_moves(&legal_moves).len() > 0 {
                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(a_move, &zobrist_constants);
            }

            assert_eq!(ttt_board.display(), "OOX\nXXO\nOXX");
        }

        #[test]
        fn alpha_beta_hash_test_3() {
            let mut invalid_count = 0;
            let mut valid_count = 0;
            let mut ttable = HashMap::new();
            let legal_moves = Vec::new();
            let zobrist_constants = TTTZobrist{};

            for side in &[true, false] {
                for xbb in (0..512).rev() {
                    for obb in (0..512).rev() {
                        let mut init_board = TTTBoard {
                            xbb,
                            obb,
                            x_to_move: *side,
                        };
                        if init_board.is_valid() {
                            valid_count += 1;

                            while (init_board.get_legal_moves(&legal_moves).len() > 0) &&
                                (init_board.get_win_state() == GameResult::Unresolved) {
                                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                                    &mut init_board, 10, std::i8::MIN, std::i8::MAX,
                                    &mut ttable, &legal_moves, &zobrist_constants
                                );
                                let a_move = GameMove { move_index, x_to_move: init_board.x_to_move };
                                //            eprintln!("board:\n{}\nnext move:{:?}",ttt_board.display(),a_move);
                                assert_eq!(true, init_board.is_valid());

                                init_board.make_move(a_move, &zobrist_constants);
                            }
                        } else {
                            invalid_count += 1;
                        }
                    }
                }
            };
            assert_eq!(28036, ttable.len());
            assert_eq!(484922, invalid_count);
            assert_eq!(39366, valid_count);
        }

        #[test]
        fn perft_positions() {
            let mut visited = HashSet::new();
            let mut init_board = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };
            let zobrist_constants = TTTZobrist{};



            fn hash_sucessors(mut a_board: &mut TTTBoard,
                              mut visited_set: &mut HashSet<u32>, zobrist_constants: &TTTZobrist
            ) {
                let this_hash = a_board.hash_val();
                let legal_moves: Vec<Vec<Vec<Vec<GameMove>>>>= Vec::new();
                visited_set.insert(this_hash);

                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves(&legal_moves);
                    for each_move in &successors {
                        let an_unmake = a_board.make_move(*each_move, &zobrist_constants);
                        hash_sucessors(&mut a_board, &mut visited_set, zobrist_constants);
                        a_board.unmake_move(an_unmake);
                    }
                }
            }
            hash_sucessors(&mut init_board, &mut visited, &zobrist_constants);
            assert_eq!(5478, visited.len());
        }

        #[test]
        fn perft_paths() {
            let mut paths = HashSet::new();
            let mut init_board = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };
            let zobrist_constants = TTTZobrist{};

//            let legal_moves = UTTTBoard::get_move_orderings();

            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard,
                          mut visited_set: &mut HashSet<u128>, zobrist_constants: &TTTZobrist
            ) {
                //            eprintln!("hash base: {}", hash_base);
                let legal_moves: Vec<Vec<Vec<Vec<GameMove>>>>= Vec::new();
                visited_set.insert(hash_base);
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves(&legal_moves);
                    for each_move in &successors {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        let unmake_move = a_board.make_move(*each_move, &zobrist_constants);
                        hash_paths(new_base, &mut a_board, &mut visited_set, zobrist_constants);
                        a_board.unmake_move(unmake_move);
                    }
                }
            }
            hash_paths(1, &mut init_board, &mut paths, &zobrist_constants);
            assert_eq!(549946, paths.len());
        }

        #[test]
        fn perft_games() {
            let mut paths = HashSet::new();
            let mut init_board = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };

            let zobrist_constants = TTTZobrist{};

            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard,
                          mut visited_set: &mut HashSet<u128>, zobrist_constants: &TTTZobrist
            ) {
                //            eprintln!("hash base: {}", hash_base);
                let legal_moves: Vec<Vec<Vec<Vec<GameMove>>>> = Vec::new();
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves(&legal_moves);
                    for each_move in successors.iter() {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        let unmake_move = a_board.make_move(*each_move, &zobrist_constants);
                        hash_paths(new_base, &mut a_board, &mut visited_set, zobrist_constants);
                        a_board.unmake_move(unmake_move);
                    }
                } else {
                    visited_set.insert(hash_base);
                }
            }
            hash_paths(1, &mut init_board, &mut paths, &zobrist_constants);
            assert_eq!(255168, paths.len());
        }
    }
}

pub mod uttt_trait_solving {

    use std::iter::FromIterator;
    use std::u16;
    use std;

    use rand;
    use rand::{rngs, SeedableRng, Rng};
//    use rand;

    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult};
//    use std::cmp::Ordering;

    // tictactoe constants
//    const UTTT_BOARD_WIDTH: usize = 9;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
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
    struct SubBoardAndSquare {
        sub_board_index: usize,
        sub_square_index: usize,
    }

    fn parse_index(an_index: u8) -> SubBoardAndSquare {
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
        x_constants: [u64; 81],
        o_constants: [u64; 81],
        x_to_move: u64,
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
              legality_lookup: &[Vec<Vec<Vec<GameMove>>>], side_to_move: bool) -> Vec<GameMove> {

            let xbb = self.xbb[subboard_index];
            let obb = self.obb[subboard_index];
            let side_int = if side_to_move { 1usize} else { 0usize};
            let lookup_moves = &legality_lookup[xbb as usize][obb as usize][side_int];

            let adjusted_moves = lookup_moves.iter(
            ).map(|GameMove{x_to_move: _side, move_index: a_move_index}|
                GameMove{x_to_move:side_to_move, move_index: a_move_index+ 9*subboard_index as u8});

            adjusted_moves.collect()
        }

        pub fn get_move_orderings() -> Vec<Vec<Vec<Vec<GameMove>>>> {
            let mut x_move_vecs = Vec::with_capacity(512);
            for xbb in 0..512u16 {
                let mut xbb_moves = Vec::with_capacity(512);
                for obb in 0..512u16 {
                    let mut side_to_move_moves = Vec::with_capacity(2);
                    for x_to_move in &[true, false] {
                        let mut legal_moves = Vec::with_capacity(9);
                        if xbb & obb == 0 {
                            let occupation_mask = xbb|obb;
                            for i in MOVE_ORDERING.iter() {
                                if (occupation_mask & (1u16<<i))==0 {
                                    legal_moves.push(GameMove{
                                        move_index: *i,
                                        x_to_move: *x_to_move}
                                    )
                                }
                            }
                        }
                        side_to_move_moves.push(legal_moves);
                    }
                    xbb_moves.push(side_to_move_moves);
                }
                x_move_vecs.push(xbb_moves);
            }

            x_move_vecs
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
                    let heur_score = x_won-o_won;
                    match heur_score {
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

        fn get_legal_moves(&self, legality_lookup: &[Vec<Vec<Vec<GameMove>>>]) -> Vec<GameMove> {
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

    #[cfg(test)]
    mod test {
        use std::collections::{HashMap, HashSet};
        use std::time;

        use crate::game_solving::{
            alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult, UMCTSNode, uct_value,
            advance_rollout, reverse_rollout, IdentityHasherBuilder
        };
        use crate::uttt_trait_solving::{parse_index, SubBoardAndSquare};
        use super::{UTTTBoard};

        #[test]
        fn zobrist_constants_test() {
            let constants = UTTTBoard::get_zobrist_constants();

            let mut start_val = constants.x_to_move;

            let x_iter = constants.x_constants.iter();
            let o_iter = constants.o_constants.iter();

            let all_hashed = x_iter.chain(o_iter).fold(start_val, |a,b| a^b);

            let expected_allhash = 10891453110652296350u64;

            eprintln!("{} {}", all_hashed, expected_allhash);
            assert_eq!(expected_allhash, all_hashed);
        }

        #[test]
        fn advance_rollout_test() {
            let mut unmove_queue = Vec::with_capacity(81);

            let mut node_store= HashMap::with_capacity_and_hasher(2_000_000, IdentityHasherBuilder{});
            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            let mut search_alpha = std::i8::MIN;
            let mut search_beta  = std::i8::MAX;

            let time_limit = 990_000;
            let mut rollout_count = 0;


            let mut uttt_board = UTTTBoard::default();
            let first_move = GameMove{x_to_move:true, move_index: 40};
            uttt_board.make_move(first_move, &zobrist_constants);

            let orig_board = uttt_board.clone();
            let now_instant = time::Instant::now();
            while now_instant.elapsed().as_micros()<time_limit {
                while let Some(new_alpha_beta) = advance_rollout(&mut unmove_queue,
                        &mut uttt_board, &legal_moves, search_alpha, search_beta, &mut node_store, &zobrist_constants){
                    let (new_alpha, new_beta) = new_alpha_beta;
                    search_alpha = new_alpha;
                    search_beta = new_beta;
                };
                let rollout_result = search_alpha;

                assert_eq!(uttt_board.get_win_state()!= GameResult::Unresolved, true);

                reverse_rollout(rollout_result, &mut unmove_queue, &mut uttt_board,
                                &mut node_store
                );
                assert_eq!(orig_board, uttt_board);
                rollout_count +=1;
            }

            eprintln!("completed {} rollouts", rollout_count);
            assert_eq!(true, rollout_count>4000);
            assert_eq!(rollout_count, 50000);
        }

        #[test]
        fn square_ordering_test() {
            const SUPER_MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
            const SUB_MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
            let mut ordered = Vec::new();
            for board_index in SUPER_MOVE_ORDERING.iter() {
                for square_index in SUB_MOVE_ORDERING.iter() {
                    ordered.push(9 * board_index + square_index);
                }
            }

            eprintln!("{:?}", ordered);
            assert_eq!(ordered.len(), 81);
        }

        #[test]
        fn display_test1() {
            let x = UTTTBoard {
                xbb: [0u16,0u16,4u16,0u16,0u16,0u16,0u16,0u16,0u16],
                obb: [0u16;9],
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: false,
                hash_state: 0,
            };
            let board_string = x.display();
            eprintln!("{}",board_string);
            eprintln!("........X\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........");
            assert_eq!(board_string, "........X\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
        }

        #[test]
        fn display_test2() {
            let x = UTTTBoard {
                xbb: [0u16,0u16,4u16,0u16,0u16,0u16,0u16,0u16,0u16,],
                obb: [4u16,0u16,0u16,0u16,0u16,0u16,0u16,0u16,0u16,],
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: false,
                hash_state: 0,
            };
            let board_string = x.display();
            eprintln!("{}", board_string);
            assert_eq!(board_string, "..O.....X\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
        }

        #[test]
        fn make_and_get_moves_test_1() {
            let mut observed_hashes = HashSet::with_capacity(81);
            let mut x = UTTTBoard::default();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(9, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);
            assert_eq!(false, observed_hashes.contains(&x.hash_state));
            observed_hashes.insert(x.hash_state);

            //testing an x move
            let ttt_move = GameMove { move_index: 40, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            eprintln!("{}\n", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);
            assert_eq!(false, observed_hashes.contains(&x.hash_state));
            observed_hashes.insert(x.hash_state);

            //testing an x move
            let ttt_move = GameMove { move_index: 36, x_to_move: false };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            eprintln!("{}\n", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(0, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 0, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(0, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 4, x_to_move: false };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O.......\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 37, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O.......\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(1, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 13, x_to_move: false };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            let ttt_move = GameMove { move_index: 43, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 71, x_to_move: false };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O...\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(8, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 79, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O.X.\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 69, x_to_move: false };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n...O.O.X.\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(6, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 61, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.X.O.O.X.\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 70, x_to_move: false };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.X.OOO.X.\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(7, x.last_super_index);
            assert_eq!(1 << 7, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            eprintln!("{}\n", board_string);
//            assert_eq!(1,2);
        }

        #[test]
        fn get_index_tests() {
            let mut sub_col = 0u8;
            let mut super_col = 0u8;
            let mut sub_row = 0u8;
            let mut super_row = 0u8;

            let sub_board = 5;
            let sub_square = 6;

            let expected_tuple = SubBoardAndSquare { sub_board_index: sub_board, sub_square_index: sub_square };
            let actual_tuple = parse_index(51);

            assert_eq!(expected_tuple, actual_tuple);
        }

        #[test]
        fn playout_test() {

            let mut move_count = 0;
            let base_board = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();


            let expt_start = time::Instant::now();

            for i in 0..1200u64 {
                let mut uttt_board = base_board.clone();
                while uttt_board.get_win_state() == GameResult::Unresolved {
                    let moves = uttt_board.get_legal_moves(&legal_moves);
                    let first_move = moves.get(0);
                    match first_move {
                        Some(a_move) => {
                            uttt_board.make_move(*a_move, &zobrist_constants);
                            move_count += 1;
                        },
                        None => break,
                    }
//                    eprintln!("{}\n", uttt_board.display());
//                    assert_eq!(true, uttt_board.is_valid());
                }
            }

            let elapsed_time = expt_start.elapsed().as_micros();
            eprintln!("elapsed time: {}", elapsed_time);
            assert_eq!(true, elapsed_time<50000, "time was not less than 50k microseconds");
        }


        #[test]
        fn make_and_unmake_move_test_1() {
            let mut x = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };
            let zobrist_constants = UTTTBoard::get_zobrist_constants();
            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);


            //testing an x move
            let ttt_move = GameMove { move_index: 40, x_to_move: true };
            let unmake_move = x.make_move(ttt_move, &zobrist_constants);
            let board_string = x.display();
            eprintln!("{}", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);

            // testing an x unmake move
            x.unmake_move(unmake_move);
            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
        }

        #[test]
        fn movegen_test_1() {
            let mut x = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                o_supers: 0,
                x_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            while x.get_legal_moves(&legal_moves).len() > 0 {
                let moves = x.get_legal_moves(&legal_moves);
                x.make_move(moves[0], &zobrist_constants);
                eprintln!("{}", x.display());
                assert_eq!(true, x.is_valid());
            }
        }


        #[test]
        fn alpha_beta_playout_test() {
            let mut uttt_board = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let mut move_count = 0;
            let mut ttable = HashMap::new();

            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            while uttt_board.get_win_state() == GameResult::Unresolved {
                let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                    &mut uttt_board, 1, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );

                let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                uttt_board.make_move(ab_move, &zobrist_constants);
            }
        }

        #[test]
        fn alpha_beta_playout_fast_d2_test() {

            let mut move_count = 0;
            let mut ttable = HashMap::new();

            let legal_moves = UTTTBoard::get_move_orderings();

            let now = time::Instant::now();
            for i in 0..5 {
                let mut uttt_board = UTTTBoard {
                    xbb: [0u16 ;9],
                    obb: [0u16 ;9],
                    last_super_index: 9,
                    x_supers: 0,
                    o_supers: 0,
                    x_to_move: true,
                    hash_state: 0,
                };
                let zobrist_constants = UTTTBoard::get_zobrist_constants();
                while uttt_board.get_win_state() == GameResult::Unresolved {
//                    eprintln!("===========\n{}",uttt_board.display());
                    let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                        &mut uttt_board, 2, std::i8::MIN, std::i8::MAX,
                        &mut ttable, &legal_moves, &zobrist_constants
                    );

                    let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                    if move_index < 100 {
                        uttt_board.make_move(ab_move, &zobrist_constants);
                        move_count += 1;
                    } else {
                        break
                    }
                }
            }

            let time_check = now.elapsed().as_micros() < 50000;
            eprintln!("fast playout time was: {}",now.elapsed().as_micros());
            assert_eq!(true,time_check);
        }

        #[test]
        fn alpha_beta_playout_fast_d4_test() {

            let mut move_count = 0;
            let mut ttable = HashMap::new();

            let legal_moves = UTTTBoard::get_move_orderings();

            let master_board = UTTTBoard {
                xbb: [0, 0, 0, 0, 4, 0, 0, 0, 0],
                obb: [0u16 ;9],
                last_super_index: 4,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            let now = time::Instant::now();
            for i in 0..3000 {
                let mut uttt_board = master_board.clone();
                while uttt_board.get_win_state() == GameResult::Unresolved {
                    let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                        &mut uttt_board, 4, std::i8::MIN, std::i8::MAX,
                        &mut ttable,  &legal_moves, &zobrist_constants
                    );

                    let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                    uttt_board.make_move(ab_move, &zobrist_constants);
                }
            }

            let time_check = now.elapsed().as_micros() < 500000;
            eprintln!("fast playout time was: {}",now.elapsed().as_micros());
            assert_eq!(true,time_check);
        }



        #[test]
        fn alpha_beta_zero_window() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut uttt_board = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                o_supers: 0,
                x_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
            let search_start = time::Instant::now();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut uttt_board, 8, std::i8::MAX-1, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );

            eprintln!("board lower: {}\nboard upper: {}",board_val, board_upper);
            assert_eq!(board_val, -128);
            assert_eq!(board_upper, 1);

            eprintln!("ttable size: {}", ttable.len());
            eprintln!("elapsed search time: {}", search_start.elapsed().as_micros());
            assert_eq!(true, uttt_board.is_valid(), "expected a valid bitboard");
//            assert_eq!(true,false);
        }



        #[test]
        fn alpha_beta_hash_test_1a() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut uttt_board = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                o_supers: 0,
                x_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
            let search_start = time::Instant::now();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut uttt_board, 5, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );

            eprintln!("ttable size: {}", ttable.len());
            eprintln!("elapsed search time: {}", search_start.elapsed().as_millis());
            assert_eq!(true, uttt_board.is_valid(), "expected a valid bitboard");
        }



        #[test]
        fn alpha_beta_hash_test_1b() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut uttt_board = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                o_supers: 0,
                x_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let legal_moves = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();

            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
            let search_start = time::Instant::now();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut uttt_board, 6, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );

            eprintln!("ttable size: {}", ttable.len());
            eprintln!("elapsed search time: {}", search_start.elapsed().as_micros());
            assert_eq!(true, uttt_board.is_valid(), "expected a valid bitboard");
        }



        #[test]
        fn constant_move_order_test() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut uttt_board = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                o_supers: 0,
                x_supers: 0,
                x_to_move: true,
                hash_state: 0,
            };

            let default_ordering = UTTTBoard::get_move_orderings();
            let zobrist_constants = UTTTBoard::get_zobrist_constants();


            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
            let search_start = time::Instant::now();
            let (board_val, board_upper, move_index, found_terminal) = alpha_beta_search(
                &mut uttt_board, 5, std::i8::MIN, std::i8::MAX,
                &mut ttable, &default_ordering, &zobrist_constants
            );

            eprintln!("ttable size: {}", ttable.len());
            eprintln!("elapsed search time: {}", search_start.elapsed().as_micros());
            assert_eq!(true, uttt_board.is_valid(), "expected a valid bitboard");
        }
    }
}


macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}
pub fn main() {
    // game loop
    use crate::uttt_trait_solving::{UTTTBoard,parse_index_from_super_coords};
//    use crate::game_solving::{umcts, AlphaBetaTTCompatible, GameMove, uct_value,
//                              MCTSAction, UMCTSNode};
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


//    let mut search_alpha = std::i8::MIN;
//    let mut search_beta  = std::i8::MAX;

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
            legal_moves = UTTTBoard::get_move_orderings();
            990_000
        } else {
            90_000
        };

//        eprintln!("opponent row was: {} {}", opponent_row, opponent_col);
//        eprintln!("elapsed so far: {}", turn_start.elapsed().as_micros());
//        eprintln!("start of turn board state was:\n{}", uttt_board.display());
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
//                            value node vists action vists side to move
                        }
                        None => {
                            eprintln!("no stored statistics found for hash value: {}",this_hash);
                        }
                    }
                    eprintln!("");
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