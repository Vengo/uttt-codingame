//use std::collections::HashMap;
//use std::io;

pub mod game_solving {
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::time::Instant;

    pub struct HashState {
        alpha: i8,
        beta: i8,
        depth: u8,
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

    pub struct MCTSAction {
        game_move: GameMove,
        visits: u32,
        value: u32,
    }

    pub struct MCTSNode<BoardState> {
        game_state: BoardState,
        actions: Vec<MCTSAction>,
    }

    pub trait AlphaBetaTTCompatible<T: Hash + Eq + Copy, Unmove> {
        fn hash_val(&self) -> T;
        fn get_win_state(&self) -> GameResult;
        fn x_to_move(&self) -> bool;
        fn make_move(&mut self, a_move: &GameMove) -> Unmove;
        fn unmake_move(&mut self, a_move: Unmove);
        fn score(&self) -> i8;
        fn get_legal_moves(&self, legal_lookup: &Vec<Vec<Vec<Vec<GameMove>>>>) -> Vec<GameMove>;
        fn display(&self) -> String;
        fn is_valid(&self) -> bool;
//        fn compatible_transposition_table() -> HashMap<Self::HashKeyType,HashState>;
    }

    const LT: fn(i8, i8) -> bool = |a, b| a.lt(&b);
    const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);

    const GT: fn(i8, i8) -> bool = |a, b| a.gt(&b);
    const MAX: fn(i8, i8) -> i8 = |a, b| a.max(b);

    pub fn mcts_playout<T: Hash + Eq + Copy, Unmove>(
        board: &mut impl AlphaBetaTTCompatible<T, Unmove>, playout_store: &mut HashMap<T,i32>) -> i32 {
        unimplemented!("MCTS playout not implemented");
        return 0;
    }

    pub fn mcts_search<T: Hash + Eq + Copy, Unmove, BoardState: AlphaBetaTTCompatible<T,Unmove> + Clone> (
        board: &mut BoardState, legal_moves: &Vec<Vec<Vec<Vec<GameMove>>>>,
        timing_object: &Instant, time_limit_micros: &u32,
        mcts_store: &mut HashMap<T, MCTSNode<BoardState>>,
        playout_store: &mut HashMap<T,i32>) -> (i8,u8) {

//        let example = MCTSNode{
//            game_state:board.clone(),
//            actions: Vec::new(),
//        };
//        let example_ref = mcts_store.insert(board.hash_val(), example).unwrap();

        let root_node = match mcts_store.get(&board.hash_val()) {
            Some(board_state) => board_state,
            None => {
                let board_actions = board.get_legal_moves(legal_moves);
                let mcts_actions = board_actions.iter(
                ).map(|an_action| MCTSAction{
                    game_move: an_action.clone(),
                    visits: 0,
                    value: 0,
                }).collect();
                let this_node = MCTSNode{
                    game_state: board.clone(),
                    actions: mcts_actions,
                };

                mcts_store.insert(board.hash_val(), this_node).unwrap();
                let root_ref2 = mcts_store.get(&board.hash_val()).unwrap();
                root_ref2
            }
        };

        //TODO: evaluate best action
        //TODO: call mcts_search on the board resulting from that move
        //TODO: update the chosen action with action statistics
        //TODO: insert the board with updated action statistics back into the hashmap

        unimplemented!("MCTS not yet implemented");
    }

    pub fn alpha_beta_search<T: Hash + Eq + Copy, Unmove> (
        board: &mut impl AlphaBetaTTCompatible<T, Unmove>, depth: u8, alpha: i8, beta: i8,
        transpositions: &mut HashMap<T, HashState>, legal_moves: &Vec<Vec<Vec<Vec<GameMove>>>>
        ) -> (i8, u8) {
        // implemented based on pseudo-code from from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning


        let mut node_alpha = alpha;
        let mut node_beta = beta;
        let node_hash = board.hash_val();
        let mut early_return = false;
        let mut new_score = std::i8::MAX;
        let mut move_index = std::u8::MAX;

        match board.get_win_state() {
            GameResult::Unresolved => {
                if depth > 0 {
                    match transpositions.get(&node_hash) {
                        Some(hashed_node) => {
                            if hashed_node.depth >= depth {
                                node_alpha = hashed_node.alpha;
                                node_beta = hashed_node.beta;
                            }
                        },
                        None => {},
                    }
                } else {
                    new_score = if board.x_to_move() { 1 } else { -1 };
                    early_return = true;
                }
            },
            GameResult::P1Win => {
                new_score = std::i8::MAX;
                move_index = std::u8::MAX;
                let inserted_state = HashState { alpha: new_score, beta: new_score, depth: std::u8::MAX };
                transpositions.insert(node_hash, inserted_state);
                early_return = true;
            },
            GameResult::P2Win => {
                new_score = std::i8::MIN;
                move_index = std::u8::MAX;
                let inserted_state = HashState { alpha: new_score, beta: new_score, depth: std::u8::MAX };
                transpositions.insert(node_hash, inserted_state);
                early_return = true;
            },
            GameResult::Tie => {
                new_score = 0i8;
                move_index = std::u8::MAX;
                let inserted_state = HashState { alpha: new_score, beta: new_score, depth: std::u8::MAX };
                transpositions.insert(node_hash, inserted_state);
                early_return = true;
            },
        }
        if early_return {
            return (new_score, move_index)
        };

        let (comp, selector, opt_val, start_val): (fn(i8, i8) -> bool, fn(i8, i8) -> i8, i8, i8) = match board.x_to_move() {
            true => (GT, MAX, std::i8::MAX, std::i8::MIN),
            false => (LT, MIN, std::i8::MIN, std::i8::MAX)
        };

        let mut new_score = start_val;
        let moves = board.get_legal_moves(legal_moves);


        if moves.len() == 0 {
            return (0, 255)
        }

        let mut move_index= moves[0].move_index;

        for each_move in moves.iter() {
            let unmake_instructions = board.make_move(&each_move);
            let (move_score, _move_index) = {
                alpha_beta_search(
                    board, depth - 1, node_alpha, node_beta, transpositions, legal_moves
                )
            };

            match move_score {
                move_score if comp(move_score, new_score) => {
                    new_score = move_score;
                    move_index = each_move.move_index;
                    board.unmake_move(unmake_instructions);

                    if board.x_to_move() {
                        node_alpha = selector(node_alpha, new_score);
                    } else {
                        node_beta = selector(node_beta, new_score);
                    }
                    if move_score == opt_val {
                        break;
                    }
                },
                _ => {
                    if move_index == std::u8::MAX { move_index = each_move.move_index; };
                    board.unmake_move(unmake_instructions);
                },
            };
        };

        let inserted_state = HashState { alpha: node_alpha, beta: node_beta, depth };
        transpositions.insert(node_hash, inserted_state);

//        let side_moved = if board.x_to_move() { "X" } else { "O" };
        return (new_score, move_index);
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

    impl AlphaBetaTTCompatible<u32, TTTUnmove> for TTTBoard {
        fn hash_val(&self) -> u32 {
            let base_hash = (self.xbb as u32) << 16 | (self.obb as u32);
            return match self.x_to_move {
                true => { base_hash },
                false => { !base_hash },
            };
        }

        fn get_win_state(&self) -> GameResult {
            // check for X or O win patterns, check for a filled board, or return unresolved
            if RESULT_MASKS.iter().any(|x| (x & self.xbb).eq(x)) {
                return GameResult::P1Win;
            } else if RESULT_MASKS.iter().any(|x| (x & self.obb).eq(x)) {
                return GameResult::P2Win;
            } else if (self.xbb | self.obb) == CAT {
                return GameResult::Tie;
            } else {
                return GameResult::Unresolved;
            };
        }

        fn x_to_move(&self) -> bool {
            return self.x_to_move;
        }

        fn make_move(&mut self, a_move: &GameMove) -> TTTUnmove {
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
            return unmove;
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

        fn get_legal_moves(&self, legal_lookup: &Vec<Vec<Vec<Vec<GameMove>>>>) -> Vec<GameMove> {
            let occupied_bb = self.xbb | self.obb;

            let valid_moves = MOVE_ORDERING.iter().filter_map(|a_move_index| {
                match occupied_bb & (1u16 << a_move_index) {
                    0 => Some(GameMove { x_to_move: self.x_to_move, move_index: a_move_index.clone() }),
                    _nonzero => {
                        None
                    },
                }
            }).collect();

            return valid_moves;
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
            return rows.join("\n").to_string();
        }

        fn is_valid(&self) -> bool {
            // there are more ways to invalidate a board than are checked here
            return (self.obb & self.xbb) == 0;
        }
    }

    #[cfg(test)]
    mod test {
        use std::collections::{HashMap, HashSet};
        use std::time;

        use crate::game_solving::{
            alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult, HashState};

        use super::{RESULT_MASKS, TTTBoard, TTTUnmove};
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
            let board_string = x.display();
            assert_eq!(board_string, "...\n...\n...");
            assert_eq!(x.x_to_move, true);


            //testing an x move
            let ttt_move = GameMove { move_index: 0, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
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
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "...\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = GameMove { move_index: 1, x_to_move: false };
            let unmake = x.make_move(&ttt_move);
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
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "O..\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //testing an x move
            let ttt_move = GameMove { move_index: 1, x_to_move: true };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "OX.\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = GameMove { move_index: 2, x_to_move: false };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "OXO\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //testing an x move
            let ttt_move = GameMove { move_index: 7, x_to_move: true };
            x.make_move(&ttt_move);
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

            while x.get_legal_moves(&legal_moves).len() > 0 {
                let moves = x.get_legal_moves(&legal_moves);
                x.make_move(&moves[0]);
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

            let mut ttable = HashMap::new();
            let (board_val, move_index) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves
            );


//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
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



            let mut ttable = HashMap::new();
            let (board_val, move_index) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves
            );

//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
                eprintln!("{}", ttt_board.display());
                assert_eq!(127, board_val);
            }
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

            let mut ttable = HashMap::new();
            let (board_val, move_index) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves
            );

            let legal_moves = Vec::new();

//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
                eprintln!("{}", ttt_board.display());
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

            let mut ttable = HashMap::new();
            let (board_val, move_index) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves
            );



//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
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

            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable
//            );
            let legal_moves = Vec::new();

            assert_eq!(
                (0, 4),
                alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN,
                    std::i8::MAX, &mut ttable, &legal_moves)
            );


            let now_instant = time::Instant::now();
            while ttt_board.get_legal_moves(&legal_moves).len() > 0 {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
            }

            assert_eq!(ttt_board.display(), "OOX\nXXO\nOXX");
        }

        #[test]
        fn alpha_beta_hash_test_3() {
            let mut invalid_count = 0;
            let mut valid_count = 0;
            let mut ttable = HashMap::new();
            let legal_moves = Vec::new();

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
                                let (board_val, move_index) = alpha_beta_search(
                                    &mut init_board, 10, std::i8::MIN, std::i8::MAX,
                                    &mut ttable, &legal_moves
                                );
                                let a_move = GameMove { move_index, x_to_move: init_board.x_to_move };
                                //            eprintln!("board:\n{}\nnext move:{:?}",ttt_board.display(),a_move);
                                assert_eq!(true, init_board.is_valid());

                                init_board.make_move(&a_move);
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



            fn hash_sucessors(mut a_board: &mut TTTBoard, mut visited_set: &mut HashSet<u32>) {
                let this_hash = a_board.hash_val();
                let legal_moves: Vec<Vec<Vec<Vec<GameMove>>>>= Vec::new();
                visited_set.insert(this_hash);

                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves(&legal_moves);
                    for each_move in &successors {
                        let an_unmake = a_board.make_move(&each_move);
                        hash_sucessors(&mut a_board, &mut visited_set);
                        a_board.unmake_move(an_unmake);
                    }
                }
            }
            hash_sucessors(&mut init_board, &mut visited);
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

//            let legal_moves = UTTTBoard::get_move_orderings();

            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard, mut visited_set: &mut HashSet<u128>) {
                //            eprintln!("hash base: {}", hash_base);
                let legal_moves: Vec<Vec<Vec<Vec<GameMove>>>>= Vec::new();
                visited_set.insert(hash_base);
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves(&legal_moves);
                    for each_move in &successors {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        let unmake_move = a_board.make_move(&each_move);
                        hash_paths(new_base, &mut a_board, &mut visited_set);
                        a_board.unmake_move(unmake_move);
                    }
                }
            }
            hash_paths(1, &mut init_board, &mut paths);
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



            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard, mut visited_set: &mut HashSet<u128>) {
                //            eprintln!("hash base: {}", hash_base);
                let legal_moves: Vec<Vec<Vec<Vec<GameMove>>>> = Vec::new();
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves(&legal_moves);
                    for each_move in successors.iter() {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        let unmake_move = a_board.make_move(each_move);
                        hash_paths(new_base, &mut a_board, &mut visited_set);
                        a_board.unmake_move(unmake_move);
                    }
                } else {
                    visited_set.insert(hash_base);
                }
            }
            hash_paths(1, &mut init_board, &mut paths);
            assert_eq!(255168, paths.len());
        }
    }
}

pub mod uttt_trait_solving {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::iter::FromIterator;
    use std::u16;
    use std;

    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult};

    // tictactoe constants
//    const UTTT_BOARD_WIDTH: usize = 9;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const SQUARE_ORDERING: [u8; 81] = [40, 36, 38, 42, 39, 41, 44, 37, 43, 4, 0, 2, 6, 3, 5, 8, 1,
        7, 22, 18, 20, 24, 21, 23, 26, 19, 25, 58, 54, 56, 60, 57, 59, 62, 55, 61, 31, 27, 29, 33,
        30, 32, 35, 28, 34, 49, 45, 47, 51, 48, 50, 53, 46, 52, 76, 72, 74, 78, 75, 77, 80, 73, 79,
        13, 9, 11, 15, 12, 14, 17, 10, 16, 67, 63, 65, 69, 66, 68, 71, 64, 70];

//    const MOVE_ORDERING: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const CAT: u16 = (1 << 9) - 1;

    #[derive(Debug, Hash, Clone, Copy)]
    pub struct UTTTBoard {
        pub xbb: [u16; 9],
        pub x_supers: u16,
        pub obb: [u16; 9],
        pub o_supers: u16,
        //        cbb: u128,
//        c_supers: u16,
        pub last_super_index: u8,
        pub x_to_move: bool,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    struct SubBoardAndSquare {
        sub_board_index: usize,
        sub_square_index: usize,
    }

    fn parse_index(an_index: &u8) -> SubBoardAndSquare {
        // Returns the sub board [0,8] and sub square [0-8]
        let sub_square = (an_index % 9) as usize;
        let sub_board = (an_index / 9) as usize;

        return SubBoardAndSquare { sub_square_index: sub_square, sub_board_index: sub_board };
    }

    pub fn parse_index_from_super_coords(row_index: &u8, col_index: &u8) -> u8 {
        let board_row = row_index/3;
        let subboard_row = row_index % 3;

        let board_col = col_index/3;
        let subboard_col = col_index % 3;
        // Returns the sub board [0,8] and sub square [0-8]

        let subboard_index = 3* board_row + board_col;
        let subsquare_index = 3* subboard_row + subboard_col;
        return 9*subboard_index+subsquare_index;
    }

    #[derive(Clone, Copy)]
    pub struct UTTTUnmove {
        x_to_move: bool,
        move_index: u8,
        super_unmask: u16,
        last_super: u8,
    }

    impl UTTTBoard {
        pub fn get_tied_mask(&self) -> u16 {
            let new_mask = self.xbb.iter().zip(
                self.obb.iter()).enumerate().filter_map(
                |(index, (xbb, obb))| match xbb|obb {
                    CAT => Some(1u16<<index),
                    _ => None,
                }).fold(0u16, |current,next| current|next);

            return new_mask;
        }

        pub fn get_subboard_moves(&self, subboard_index: usize,
              legality_lookup: &Vec<Vec<Vec<Vec<GameMove>>>>, side_to_move: &bool) -> Vec<GameMove> {
            let xbb = self.xbb[subboard_index];
            let obb = self.obb[subboard_index];
            let side_int = if *side_to_move { 1usize} else { 0usize};
            let lookup_moves = &legality_lookup[xbb as usize][obb as usize][side_int];

//            let mut moves = Vec::with_capacity(9);
//            let occupied = xbb|obb;
//
//            for i in 0..9 {
//                if occupied & 1<<i == 0 {
//                    let move_index = 9*subboard_index + i;
//                    moves.push(GameMove{
//                        x_to_move: self.x_to_move,
//                        move_index: move_index as u8
//                    });
//                }
//            }
            let adjusted_moves = lookup_moves.iter().map(|GameMove{x_to_move:side, move_index: a_move_index}|
                GameMove{x_to_move:*side, move_index: a_move_index+ 9*subboard_index as u8});

            return adjusted_moves.collect();
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
            return x_move_vecs;
        }

    }

    impl AlphaBetaTTCompatible<u64, UTTTUnmove> for UTTTBoard {
        fn hash_val(&self) -> u64 {
            //TODO: Replace with a Zobrist hashing scheme
            let mut hasher = DefaultHasher::new();
            self.hash(&mut hasher);
            return hasher.finish();
        }

        fn get_win_state(&self) -> GameResult {
            // check for X or O win patterns, check for a filled board, or return unresolved
            if RESULT_MASKS.iter().any(|x| (x & self.x_supers).eq(x)) {
                return GameResult::P1Win;
            } else if RESULT_MASKS.iter().any(|x| (x & self.o_supers).eq(x)) {
                return GameResult::P2Win;
            } else {
                let new_mask = self.xbb.iter().zip(
                    self.obb.iter()).enumerate().filter_map(
                    |(index, (xbb, obb))| match xbb|obb {
                        CAT => Some(1u16<<index),
                        _ => None,
                    }).fold(0u16, |current,next| current|next);
                if new_mask | self.o_supers | self.x_supers == CAT {
                    return GameResult::Tie;
                } else {
                    return GameResult::Unresolved;
                }
            }
        }

        fn x_to_move(&self) -> bool {
            return self.x_to_move;
        }

        fn make_move(&mut self, a_move: &GameMove) -> UTTTUnmove {
            let GameMove { x_to_move, move_index: a_move_index } = a_move;
            let SubBoardAndSquare { sub_board_index, sub_square_index } = parse_index(a_move_index);
//            eprintln!("{} {}", sub_board_index, sub_square_index);

            let to_move_reset = x_to_move.clone();
            let move_to_unmask = a_move_index.clone();
            let mut super_unmask = 0;
            let sub_board_reset = self.last_super_index.clone();

            match a_move {
                GameMove { x_to_move: true, move_index } => {
                    self.xbb[sub_board_index] = self.xbb[sub_board_index] ^ (1<<sub_square_index);
                    let result_board = self.xbb[sub_board_index];

                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                    if is_sub_win {
                        let super_mask = 1u16 << sub_board_index;
                        self.x_supers |= super_mask;
                        super_unmask = super_mask;
                    }
                    self.last_super_index = sub_square_index as u8;
                    self.x_to_move = false;
                },
                GameMove { x_to_move: false, move_index } => {
                    self.obb[sub_board_index] = self.obb[sub_board_index] ^ (1<<sub_square_index);
                    let result_board = self.obb[sub_board_index];

                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                    if is_sub_win {
                        let super_mask = 1u16 << sub_board_index;
                        self.o_supers |= 1 << sub_board_index;
                        super_unmask = super_mask;
                    }
                    self.last_super_index = sub_square_index as u8;
                    self.x_to_move = true;
                },
            }

            let unmove = UTTTUnmove {
                x_to_move: to_move_reset,
                move_index: move_to_unmask,
                super_unmask: super_unmask,
                last_super: sub_board_reset,
            };
            return unmove;
        }

        fn unmake_move(&mut self, unmove: UTTTUnmove) {
            let UTTTUnmove {
                x_to_move, move_index,
                super_unmask, last_super
            } = unmove;

            let SubBoardAndSquare{sub_board_index, sub_square_index} = parse_index(&move_index);

            self.x_to_move = x_to_move;
            self.last_super_index = last_super;

            match x_to_move {
                true => {
                    self.xbb[sub_board_index] ^= 1 << sub_square_index;
                    self.x_supers ^= super_unmask;
                },
                false => {
                    self.obb[sub_board_index] ^= 1 << sub_square_index;
                    self.o_supers ^= super_unmask;
                }
            };
        }

        fn score(&self) -> i8 {

            let x_won = self.x_supers.count_ones() as i8;
            let o_won = self.o_supers.count_ones() as i8;
            let val = match self.get_win_state() {
                GameResult::P1Win => { std::i8::MAX },
                GameResult::P2Win => { std::i8::MIN },
                GameResult::Tie => { 0i8 },
                GameResult::Unresolved => {x_won-o_won},
            };
            return val;
        }

        fn get_legal_moves(&self, legality_lookup: &Vec<Vec<Vec<Vec<GameMove>>>>) -> Vec<GameMove> {
            // TODO: Modify this code to generate legal moves by an array lookup
            // TODO: Consider generating those legal moves ordered by 'next best (small) TTT move'
            // gather won bitboards, the x/o occupation bitboards, and the total occupation
            // of the bitboard directed by the last move
            let complete_bbs = self.x_supers | self.o_supers | self.get_tied_mask();

            // arm 1 9 indicates 'play anywhere', as does directing to a full
            // arm 2 checks to see if the directed subboard has been won
            // arm 3 checks if the directed subboard is full
            // arm 4 creates a bitmask of 1s aligned to the directed board
            let moves_to_return = match self.last_super_index as usize {
                value if (value >= 9) | ((complete_bbs & (1 << value)) != 0) => {
                    let mut moves = Vec::with_capacity(81);
                    for subboard_index in 0..9usize {
                        if (complete_bbs & (1 << subboard_index)) == 0 {
                            moves.append(&mut self.get_subboard_moves(
                                subboard_index, legality_lookup, &self.x_to_move))
                        }
                    }
                    moves
                },
                value => self.get_subboard_moves(
                    value, legality_lookup, &self.x_to_move),
            };

            return moves_to_return;
        }

        fn display(&self) -> String {
            let mut rows = [['.';9]; 9];

            for row_index in 0..9usize {
                for col_index in 0..9usize {
                    let subboard_row = row_index % 3;
                    let subboard_col = col_index % 3;

                    let superboard_row = row_index/3;
                    let superboard_col = col_index/3;

                    let subboard = 3*superboard_row + superboard_col;
                    let subboard_square_mask = 1u16 <<(3*subboard_row+subboard_col);
                    if self.xbb[subboard] & subboard_square_mask != 0 {
                        rows[row_index][col_index] = 'X';
                    } else if self.obb[subboard] & subboard_square_mask != 0 {
                        rows[row_index][col_index] = 'O';
                    }
                }
            }

            let output = rows.iter().map(
                |a_row| {
                    let mut row = String::from_iter(a_row.iter());
                    row.push('\n');
                    return row;
                }).collect();
//            .join("\n").to_string();
//            return rows.join("\n").to_string();
            return output;
        }

        fn is_valid(&self) -> bool {
            return self.obb.iter().zip(self.xbb.iter()).all(|(o_subboard, x_subboard)| {
                (o_subboard & x_subboard) == 0
            });

        }
    }

    #[cfg(test)]
    mod test {
        use std::collections::{HashMap, HashSet};
        use std::time;

        use crate::game_solving::{
            alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult, HashState};
        use crate::uttt_trait_solving::{parse_index, SubBoardAndSquare};

        use super::{RESULT_MASKS, UTTTBoard};

        #[test]
        fn square_ordering_test() {
            const SUPER_MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
            const SUB_MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
            let mut ordered = Vec::new();
            for board_index in SUPER_MOVE_ORDERING.iter() {
                for square_index in SUB_MOVE_ORDERING.iter() {
                    ordered.push((9 * board_index + square_index));
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
            };
            let board_string = x.display();
            eprintln!("{}", board_string);
            assert_eq!(board_string, "..O.....X\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
        }

        #[test]
        fn make_and_get_moves_test_1() {
            let mut x = UTTTBoard {
                xbb: [0u16 ;9],
                obb: [0u16 ;9],
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
            };
            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(9, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 40, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            eprintln!("{}\n", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 36, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            eprintln!("{}\n", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(0, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 0, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(0, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 4, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O.......\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 37, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O.......\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(1, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 13, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            let ttt_move = GameMove { move_index: 43, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 71, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O...\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(8, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 79, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O.X.\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 69, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n...O.O.X.\n");
            assert_eq!(x.x_to_move, true);
            assert_eq!(6, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 61, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.X.O.O.X.\n");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 70, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
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
            let actual_tuple = parse_index(&51);

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
            };

            let legal_moves = UTTTBoard::get_move_orderings();


            let expt_start = time::Instant::now();
            for i in 0..3000u64 {
                let mut uttt_board = base_board;
                while uttt_board.get_win_state() == GameResult::Unresolved {
                    let moves = uttt_board.get_legal_moves(&legal_moves);
                    let first_move = moves.get(0);
                    match first_move {
                        Some(a_move) => {
                            uttt_board.make_move(a_move);
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
            };
            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);


            //testing an x move
            let ttt_move = GameMove { move_index: 40, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            eprintln!("{}", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, false);

            // testing an x unmake move
            x.unmake_move(unmake_move);
            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n");
            assert_eq!(x.x_to_move, true);

//            //re-adding an x move
//            let ttt_move = GameMove { move_index: 4, x_to_move: true };
//            x.make_move(&ttt_move);
//            let board_string = x.display();
//            assert_eq!(board_string, "...\n.X.\n...");
//            assert_eq!(x.x_to_move, false);
//
//            //testing an o move
//            let ttt_move = GameMove { move_index: 1, x_to_move: false };
//            let unmake = x.make_move(&ttt_move);
//            let board_string = x.display();
//            assert_eq!(board_string, ".O.\n.X.\n...");
//            assert_eq!(x.x_to_move, true);
//
//            //unmaking an o move
//            x.unmake_move(unmake);
//            let board_string = x.display();
//            assert_eq!(board_string, "...\n.X.\n...");
//            assert_eq!(x.x_to_move, false);
//
//            //testing an o move
//            let ttt_move = GameMove { move_index: 0, x_to_move: false };
//            x.make_move(&ttt_move);
//            let board_string = x.display();
//            assert_eq!(board_string, "O..\n.X.\n...");
//            assert_eq!(x.x_to_move, true);
//
//            //testing an x move
//            let ttt_move = GameMove { move_index: 1, x_to_move: true };
//            x.make_move(&ttt_move);
//            let board_string = x.display();
//            assert_eq!(board_string, "OX.\n.X.\n...");
//            assert_eq!(x.x_to_move, false);
//
//            //testing an o move
//            let ttt_move = GameMove { move_index: 2, x_to_move: false };
//            x.make_move(&ttt_move);
//            let board_string = x.display();
//            assert_eq!(board_string, "OXO\n.X.\n...");
//            assert_eq!(x.x_to_move, true);
//
//            //testing an x move
//            let ttt_move = GameMove { move_index: 7, x_to_move: true };
//            x.make_move(&ttt_move);
//            let board_string = x.display();
////            RESULT_MASKS.iter().for_each(|mask| println! {"mask:\t{}\nbitw:\t{}\nxbb:\t{}\n", mask, x.xbb & mask, x.xbb});
//
//            assert_eq!(board_string, "OXO\n.X.\n.X.");
//            assert_eq!(x.x_to_move, false);
//            assert_eq!(x.get_win_state(), GameResult::P1Win);
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
            };

            let legal_moves = UTTTBoard::get_move_orderings();

            while x.get_legal_moves(&legal_moves).len() > 0 {
                let moves = x.get_legal_moves(&legal_moves);
                x.make_move(&moves[0]);
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
            };

            let mut move_count = 0;
            let mut ttable = HashMap::new();

            let legal_moves = UTTTBoard::get_move_orderings();

            while uttt_board.get_win_state() == GameResult::Unresolved {
                let (board_val, move_index) = alpha_beta_search(
                    &mut uttt_board, 1, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves
                );

                let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                if move_index < 100 {
                    uttt_board.make_move(&ab_move);
                    move_count += 1;
                } else {
                    break
                }
            }
        }

        #[test]
        fn alpha_beta_playout_fast_d2_test() {

            let mut move_count = 0;
            let mut ttable = HashMap::new();

            let legal_moves = UTTTBoard::get_move_orderings();

            let now = time::Instant::now();
            for i in 0..10 {
                let mut uttt_board = UTTTBoard {
                    xbb: [0u16 ;9],
                    obb: [0u16 ;9],
                    last_super_index: 9,
                    x_supers: 0,
                    o_supers: 0,
                    x_to_move: true,
                };
                while uttt_board.get_win_state() == GameResult::Unresolved {
                    let (board_val, move_index) = alpha_beta_search(
                        &mut uttt_board, 2, std::i8::MIN, std::i8::MAX,
                        &mut ttable, &legal_moves
                    );

                    let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                    if move_index < 100 {
                        uttt_board.make_move(&ab_move);
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

            let now = time::Instant::now();
            for i in 0..1 {
                let mut uttt_board = UTTTBoard {
                    xbb: [0u16 ;9],
                    obb: [0u16 ;9],
                    last_super_index: 9,
                    x_supers: 0,
                    o_supers: 0,
                    x_to_move: true,
                };
                while uttt_board.get_win_state() == GameResult::Unresolved {
                    let (board_val, move_index) = alpha_beta_search(
                        &mut uttt_board, 4, std::i8::MIN, std::i8::MAX,
                        &mut ttable,  &legal_moves
                    );

                    let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                    if move_index < 100 {
                        uttt_board.make_move(&ab_move);
                        move_count += 1;
                    } else {
                        break
                    }
                }
            }

            let time_check = now.elapsed().as_micros() < 500000;
            eprintln!("fast playout time was: {}",now.elapsed().as_micros());
            assert_eq!(true,time_check);
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
            };

            let legal_moves = UTTTBoard::get_move_orderings();

            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
            let search_start = time::Instant::now();
            let (board_val, move_index) = alpha_beta_search(
                &mut uttt_board, 5, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves
            );

            eprintln!("ttable size: {}", ttable.len());
            eprintln!("elapsed search time: {}", search_start.elapsed().as_millis());
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
            };

            let default_ordering = UTTTBoard::get_move_orderings();


            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
            let search_start = time::Instant::now();
            let (board_val, move_index) = alpha_beta_search(
                &mut uttt_board, 5, std::i8::MIN, std::i8::MAX,
                &mut ttable, &default_ordering
            );

            eprintln!("ttable size: {}", ttable.len());
            eprintln!("elapsed search time: {}", search_start.elapsed().as_millis());
            assert_eq!(true, uttt_board.is_valid(), "expected a valid bitboard");
        }
    }
}


macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}
/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/


pub fn main() {
    // game loop

    //    use crate::ttt_trait_solving::{TTTBoard};
    use crate::uttt_trait_solving::{UTTTBoard,parse_index_from_super_coords};
    use crate::game_solving::{alpha_beta_search, AlphaBetaTTCompatible, GameMove};

    use std::collections::HashMap;

    let mut uttt_board = UTTTBoard {
        xbb: [0u16 ;9],
        obb: [0u16 ;9],
        last_super_index: 9,
        o_supers: 0,
        x_supers: 0,
        x_to_move: true,
    };

    let mut transposition_table = HashMap::new();
    const COMPETITIVE_AB_DEPTH: u8 = 5;

    let legal_moves = UTTTBoard::get_move_orderings();

    loop {
        let mut input_line = String::new();
        std::io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(" ").collect::<Vec<_>>();
        let opponent_row = parse_input!(inputs[0], i32);
        let opponent_col = parse_input!(inputs[1], i32);



        eprintln!("opponent row was: {} {}", opponent_row, opponent_col);
        eprintln!("start of turn board state was:\n{}", uttt_board.display());

        let best_move_index = match (opponent_row, opponent_col) {
            (-1, -1) => {
                let (_best_value, _best_action) = alpha_beta_search(
                    &mut uttt_board, COMPETITIVE_AB_DEPTH/2, std::i8::MIN,
                    std::i8::MAX, &mut transposition_table, &legal_moves);

                let (best_value, best_action) = alpha_beta_search(
                    &mut uttt_board, COMPETITIVE_AB_DEPTH, std::i8::MIN, std::i8::MAX,
                    &mut transposition_table, &legal_moves);
                eprintln!("best value: {}\nbest action: {}", best_value, best_action);
                let my_move = GameMove { move_index: best_action, x_to_move: uttt_board.x_to_move };
                uttt_board.make_move(&my_move);
//                println!("{} {}", best_action / 3, best_action % 3);
                best_action
            },
            (row, col) => {
                let opp_move_index = parse_index_from_super_coords(&(row as u8),&(col as u8));
//                    (9 * row + col) as u8;

                let opp_move = GameMove { move_index: opp_move_index, x_to_move: uttt_board.x_to_move };
                uttt_board.make_move(&opp_move);
                eprintln!("board after opponent move was:\n{}", uttt_board.display());
                eprintln!("\ndebug information is:\n{:?}\n",uttt_board);
                eprintln!("subboard identified as: {}", uttt_board.last_super_index);


                let (best_value, best_action) = alpha_beta_search(
                    &mut uttt_board, COMPETITIVE_AB_DEPTH/2, std::i8::MIN,
                    std::i8::MAX, &mut transposition_table, &legal_moves);

                let (best_value, best_action) = alpha_beta_search(
                    &mut uttt_board, COMPETITIVE_AB_DEPTH, std::i8::MIN, std::i8::MAX,
                    &mut transposition_table, &legal_moves);
                let my_move = GameMove { move_index: best_action, x_to_move: uttt_board.x_to_move };
                uttt_board.make_move(&my_move);

                assert_eq!(true, uttt_board.is_valid());

                eprintln!("best value: {}\nbest action: {}", best_value, best_action);
//                println!("{} {}", best_action / 3, best_action % 3);
                best_action
            },
        };

        let mut input_line = String::new();
        std::io::stdin().read_line(&mut input_line).unwrap();
        let valid_action_count = parse_input!(input_line, i32);
        eprintln!("Valid moves:");

        let mut actions = Vec::new();

        for _i in 0..valid_action_count as usize {
            let mut input_line = String::new();
            std::io::stdin().read_line(&mut input_line).unwrap();
            let inputs = input_line.split(" ").collect::<Vec<_>>();
            let row = parse_input!(inputs[0], i32);
            let col = parse_input!(inputs[1], i32);
            actions.push((row, col));
            eprintln!("\t{} {}", row, col);
        }

        let best_subboard_index = best_move_index/9;
        let best_sub_square_index = best_move_index%9;

        let base_row = best_subboard_index/3;
        let base_col = best_subboard_index%3;

        let final_row = base_row*3 + best_sub_square_index/3;
        let final_column = base_col*3 + best_sub_square_index%3;
        println!("{} {}",final_row, final_column);
//        println!("{} {}",actions[0].0, actions[0].1);
    }
}


