use std::collections::HashMap;
use std::io;

mod game_solving {
    use std::collections::HashMap;

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

    #[derive(Debug)]
    pub struct GameMove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    pub struct GameUnmove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    pub trait AlphaBetaTTCompatible {
        fn hash_val(&self) -> u32;
        fn get_win_state(&self) -> GameResult;
        fn x_to_move(&self) -> bool;
        fn make_move(&mut self, a_move: &GameMove);
        fn unmake_move(&mut self, a_move: &GameMove);
        fn score(&self) -> i8;
        fn get_legal_moves(&self) -> Vec<GameMove>;
    }

    const LT: fn(i8, i8) -> bool = |a, b| a.lt(&b);
    const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);

    const GT: fn(i8, i8) -> bool = |a, b| a.gt(&b);
    const MAX: fn(i8, i8) -> i8 = |a, b| a.min(b);

    pub fn alpha_beta_search(mut board: &mut impl AlphaBetaTTCompatible, depth: u8, alpha: i8, beta: i8, mut transpositions: &mut HashMap<u32, HashState>) -> (i8, u8) {
        // implemented based on pseudo-code from from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning

        let mut node_alpha = alpha;
        let mut node_beta = beta;
        let node_hash = board.hash_val();
        let mut early_return = false;
        let mut new_score = std::i8::MAX;
        let mut move_index = std::u8::MAX;

        match board.get_win_state() {
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
                    eprintln!("breaking early from unresolved");
                    early_return = true;
                }
            },
        }
        if early_return {
            return (new_score, move_index)
        }
        let (comp, selector, opt_val): (fn(i8, i8) -> bool, fn(i8, i8) -> i8, i8) = match board.x_to_move() {
            true => (GT, MAX, std::i8::MAX),
            false => (LT, MIN, std::i8::MIN)
        };

        match (depth, board.score()) {
            (_a_depth, _a_score) => {
                let mut new_score = std::i8::MIN;
                let mut move_index = std::u8::MAX;
                for each_move in board.get_legal_moves() {
                    board.make_move(&each_move);
                    let (move_score, _move_index) = {
                        alpha_beta_search(board, depth - 1, node_alpha, node_beta, &mut transpositions)
                    };

                    match move_score {
                        opt_val => {
                            new_score = move_score;
                            move_index = each_move.move_index;
                            node_alpha = move_score;
                            board.unmake_move(&each_move);
                            break;
                        },
                        move_score if comp(move_score, new_score) => {
                            new_score = move_score;
                            move_index = each_move.move_index;
                            let new_val = selector(node_alpha, move_score);

                            if board.x_to_move() {
                                node_alpha = selector(node_alpha, move_score);
                            } else {
                                node_beta = selector(node_beta, move_score);
                            }

                            board.unmake_move(&each_move);
                            if node_alpha >= node_beta { break; }
                        },
                        _move_score => {
                            if move_index == std::u8::MAX { move_index = each_move.move_index; };
                            board.unmake_move(&each_move);
                        },
                    };
                };

                let inserted_state = HashState { alpha: node_alpha, beta: node_beta, depth };
                transpositions.insert(node_hash, inserted_state);
                return (new_score, move_index);
            },
            other_stuff => panic!(
                "encountered unhandled stuff in alpha_beta search: {:?}", other_stuff),
        }
    }
}

mod ttt_trait_solving {
    use std::collections::HashMap;

    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult, GameUnmove, HashState};

    const LT: fn(i8, i8) -> bool = |a, b| a.lt(&b);
    const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);

    const GT: fn(i8, i8) -> bool = |a, b| a.gt(&b);
    const MAX: fn(i8, i8) -> i8 = |a, b| a.min(b);

    // tictactoe constants
    const TTT_BOARD_WIDTH: usize = 3;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const CAT: u16 = (1 << 9) - 1;

    #[derive(Debug)]
    pub struct TTTBoard {
        pub xbb: u16,
        pub obb: u16,
        pub x_to_move: bool,
    }

//    #[derive(Debug)]
//    pub struct TTTMove {
//        pub x_to_move: bool,
//        pub move_index: u8,
//    }

    //impl Default for HashState {
//    fn default() -> Self {HashState{alpha: std::i8::MIN, beta: std::i8::MAX, depth: 0}}
//}
    impl TTTBoard {
        pub fn is_valid(&self) -> bool {
            // there are more ways to invalidate a board than are checked here
            return (self.obb & self.xbb) == 0;
        }

        pub fn display(&self) -> String {
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
    }


    //    fn hash_val(&self)-> u32;
//    fn get_win_state(&self) -> GameResult;
//    fn x_to_move(&self) -> bool;
//    fn make_move(&mut self, a_move: &GameMove);
//    fn unmake_move(&mut self, a_move: &GameMove);
//    fn score(&self) -> i8;
//    fn get_legal_moves(&self) -> Vec<GameMove>;
    impl AlphaBetaTTCompatible for TTTBoard {
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

        fn make_move(&mut self, a_move: &GameMove) {
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
        }

        fn unmake_move(&mut self, a_move: &GameMove) {
            match a_move {
                GameMove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = true;
                },
                GameMove { x_to_move: false, move_index } => {
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

        fn get_legal_moves(&self) -> Vec<GameMove> {
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
    }

    #[cfg(test)]
    mod test {
        use std::collections::{HashMap, HashSet};

        use crate::game_solving::{
            alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult, GameUnmove, HashState};

        use super::{RESULT_MASKS, TTTBoard};

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
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X..\n...\n...");
            assert_eq!(x.x_to_move, false);

            // testing an x unmake move
            x.unmake_move(&ttt_move);
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
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, ".O.\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //unmaking an o move
            x.unmake_move(&ttt_move);
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

            while x.get_legal_moves().len() > 0 {
                let moves = x.get_legal_moves();
                x.make_move(&moves[0]);
                eprintln!("{}", x.display());
                assert_eq!(true, x.is_valid());
            }
        }

        #[test]
        fn alpha_beta_hash_test_1() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 0,
                obb: 44,
                x_to_move: true,
            };

            let mut ttable = HashMap::new();
            let (board_val, move_index) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
            );


//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves().len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
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
        fn alpha_beta_hash_test_2() {
            let mut ttt_board = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };

            let mut ttable = HashMap::new();
            let (board_val, move_index) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
            );
            assert_eq!((0, 4), (board_val, move_index));

//        let now_instant = time::Instant::now();
            assert_eq!((0, 4), alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
            ));
//        assert_eq!(now_instant.elapsed().as_micros(),10);
//            assert_eq!(10, ttable.len());


//        let now_instant = time::Instant::now();
            while ttt_board.get_legal_moves().len() > 0 {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
                );
                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//            eprintln!("board:\n{}\nnext move:{:?}",ttt_board.display(),a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
            }
//        assert_eq!(now_instant.elapsed().as_micros(),10);

            assert_eq!(ttt_board.display(), "OOX\nXXO\nOXX");
        }

        #[test]
        fn alpha_beta_hash_test_3() {
            let mut invalid_count = 0;
            let mut valid_count = 0;
            let mut ttable = HashMap::new();
            for side in &[true, false] {
                for xbb in 0..512 {
                    for obb in 0..512 {
                        let mut init_board = TTTBoard {
                            xbb,
                            obb,
                            x_to_move: *side,
                        };

                        if init_board.is_valid() {
                            valid_count += 1;

                            while (init_board.get_legal_moves().len() > 0) &&
                                (init_board.get_win_state() == GameResult::Unresolved) {
                                let (board_val, move_index) = alpha_beta_search(
                                    &mut init_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
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
//            assert_eq!(28090,ttable.len());
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
                visited_set.insert(this_hash);

                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves();
                    for each_move in &successors {
                        a_board.make_move(&each_move);
                        hash_sucessors(&mut a_board, &mut visited_set);
                        a_board.unmake_move(&each_move);
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

            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard, mut visited_set: &mut HashSet<u128>) {
                //            eprintln!("hash base: {}", hash_base);
                visited_set.insert(hash_base);
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves();
                    for each_move in &successors {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        a_board.make_move(&each_move);
                        hash_paths(new_base, &mut a_board, &mut visited_set);
                        a_board.unmake_move(&each_move);
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
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves();
                    for each_move in &successors {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        a_board.make_move(&each_move);
                        hash_paths(new_base, &mut a_board, &mut visited_set);
                        a_board.unmake_move(&each_move);
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

mod ttt_solving {
    use std::collections::HashMap;

    const LT: fn(i8, i8) -> bool = |a, b| a.lt(&b);
    const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);

    const GT: fn(i8, i8) -> bool = |a, b| a.gt(&b);
    const MAX: fn(i8, i8) -> i8 = |a, b| a.min(b);

    // tictactoe constants
    const TTT_BOARD_WIDTH: usize = 3;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const CAT: u16 = (1 << 9) - 1;

    #[derive(Debug)]
    pub struct TTTBoard {
        pub xbb: u16,
        pub obb: u16,
        pub x_to_move: bool,
    }

    #[derive(Debug)]
    pub struct TTTMove {
        pub x_to_move: bool,
        pub move_index: u8,
    }

    pub struct HashState {
        alpha: i8,
        beta: i8,
        depth: u8,
    }

//impl Default for HashState {
//    fn default() -> Self {HashState{alpha: std::i8::MIN, beta: std::i8::MAX, depth: 0}}
//}

    impl TTTBoard {
        pub fn alpha_beta_search(&mut self, depth: u8, alpha: i8, beta: i8, mut transpositions: &mut HashMap<u32, HashState>) -> (i8, u8) {
            // implemented based on pseudo-code from from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning

            let mut node_alpha = alpha;
            let mut node_beta = beta;
            let node_hash = self.hash_val();
            let mut early_return = false;
            let mut new_score = std::i8::MAX;
            let mut move_index = std::u8::MAX;

            match self.get_win_state() {
                GameState::XWin => {
                    new_score = std::i8::MAX;
                    move_index = std::u8::MAX;
                    let inserted_state = HashState { alpha: new_score, beta: new_score, depth: std::u8::MAX };
                    transpositions.insert(node_hash, inserted_state);
                    early_return = true;
                },
                GameState::OWin => {
                    new_score = std::i8::MIN;
                    move_index = std::u8::MAX;
                    let inserted_state = HashState { alpha: new_score, beta: new_score, depth: std::u8::MAX };
                    transpositions.insert(node_hash, inserted_state);
                    early_return = true;
                },
                GameState::Cat => {
                    new_score = 0i8;
                    move_index = std::u8::MAX;
                    let inserted_state = HashState { alpha: new_score, beta: new_score, depth: std::u8::MAX };
                    transpositions.insert(node_hash, inserted_state);
                    early_return = true;
                },
                GameState::Unresolved => {
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
                        new_score = if self.x_to_move { 1 } else { -1 };
                        eprintln!("breaking early from unresolved");
                        early_return = true;
                    }
                },
            }
            if early_return {
                return (new_score, move_index)
            }
            let (comp, selector, opt_val): (fn(i8, i8) -> bool, fn(i8, i8) -> i8, i8) = match self.x_to_move {
                true => (GT, MAX, std::i8::MAX),
                false => (LT, MIN, std::i8::MIN)
            };

            match (depth, self.score()) {
                (_a_depth, _a_score) => {
                    let mut new_score = std::i8::MIN;
                    let mut move_index = std::u8::MAX;
                    for each_move in self.get_legal_moves() {
                        self.make_move(&each_move);
                        let (move_score, _move_index) = {
                            self.alpha_beta_search(depth - 1, node_alpha, node_beta, &mut transpositions)
                        };

                        match move_score {
                            opt_val => {
                                new_score = move_score;
                                move_index = each_move.move_index;
                                node_alpha = move_score;
                                self.unmake_move(&each_move);
                                break;
                            },
                            move_score if comp(move_score, new_score) => {
                                new_score = move_score;
                                move_index = each_move.move_index;
                                let new_val = selector(node_alpha, move_score);

                                if self.x_to_move {
                                    node_alpha = selector(node_alpha, move_score);
                                } else {
                                    node_beta = selector(node_beta, move_score);
                                }

                                self.unmake_move(&each_move);
                                if node_alpha >= node_beta { break; }
                            },
                            _move_score => {
                                if move_index == std::u8::MAX { move_index = each_move.move_index; };
                                self.unmake_move(&each_move);
                            },
                        };
                    };

                    let inserted_state = HashState { alpha: node_alpha, beta: node_beta, depth };
                    transpositions.insert(node_hash, inserted_state);
                    return (new_score, move_index);
                },
                other_stuff => panic!(
                    "encountered unhandled stuff in alpha_beta search: {:?}", other_stuff),
            }
        }

        fn hash_val(&self) -> u32 {
            let base_hash = (self.xbb as u32) << 16 | (self.obb as u32);
            return match self.x_to_move {
                true => { base_hash },
                false => { !base_hash },
            };
        }

        pub fn is_valid(&self) -> bool {
            // there are more ways to invalidate a board than are checked here
            return (self.obb & self.xbb) == 0;
        }

        pub fn make_move(&mut self, a_move: &TTTMove) {
            match a_move {
                TTTMove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = false;
                },
                TTTMove { x_to_move: false, move_index } => {
                    self.obb ^= 1u16 << move_index;
                    self.x_to_move = true;
                },
            }
        }

        fn unmake_move(&mut self, a_move: &TTTMove) {
            match a_move {
                TTTMove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = true;
                },
                TTTMove { x_to_move: false, move_index } => {
                    self.obb ^= 1u16 << move_index;
                    self.x_to_move = false;
                },
            }
        }

        fn get_win_state(&self) -> GameState {
            // check for X or O win patterns, check for a filled board, or return unresolved
            if RESULT_MASKS.iter().any(|x| (x & self.xbb).eq(x)) {
                return GameState::XWin;
            } else if RESULT_MASKS.iter().any(|x| (x & self.obb).eq(x)) {
                return GameState::OWin;
            } else if (self.xbb | self.obb) == CAT {
                return GameState::Cat;
            } else {
                return GameState::Unresolved;
            };
        }

        fn get_legal_moves(&self) -> Vec<TTTMove> {
            let occupied_bb = self.xbb | self.obb;

            let valid_moves = MOVE_ORDERING.iter().filter_map(|a_move_index| {
                match occupied_bb & (1u16 << a_move_index) {
                    0 => Some(TTTMove { x_to_move: self.x_to_move, move_index: a_move_index.clone() }),
                    _nonzero => {
                        None
                    },
                }
            }).collect();

            return valid_moves;
        }

        fn score(&self) -> i8 {
            match self.get_win_state() {
                GameState::XWin => { 127 },
                GameState::OWin => { -128 },
                GameState::Cat => { 0 },
                GameState::Unresolved if self.x_to_move => { 1 },
                GameState::Unresolved => { -1 },
            }
        }

        pub fn display(&self) -> String {
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
    }

    #[derive(Debug, Eq, PartialEq)]
    enum GameState {
        XWin,
        OWin,
        Cat,
        Unresolved,
    }

    #[cfg(test)]
    mod test {
        use std::collections::{HashMap, HashSet};

        use super::{GameState, RESULT_MASKS, TTTBoard, TTTMove};

//        use std::time;

        #[test]
        fn testprint() {
            eprintln!("printing to stderr");
            println!("printing to stdout");
            let x = 2;
            assert_eq!(x, 2);
        }

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
            let ttt_move = TTTMove { move_index: 0, x_to_move: true };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X..\n...\n...");
            assert_eq!(x.x_to_move, false);

            // testing an x unmake move
            x.unmake_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "...\n...\n...");
            assert_eq!(x.x_to_move, true);

            //readding an x move
            let ttt_move = TTTMove { move_index: 4, x_to_move: true };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "...\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = TTTMove { move_index: 1, x_to_move: false };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, ".O.\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //unmaking an o move
            x.unmake_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "...\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = TTTMove { move_index: 0, x_to_move: false };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "O..\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //testing an x move
            let ttt_move = TTTMove { move_index: 1, x_to_move: true };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "OX.\n.X.\n...");
            assert_eq!(x.x_to_move, false);

            //testing an o move
            let ttt_move = TTTMove { move_index: 2, x_to_move: false };
            x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "OXO\n.X.\n...");
            assert_eq!(x.x_to_move, true);

            //testing an x move
            let ttt_move = TTTMove { move_index: 7, x_to_move: true };
            x.make_move(&ttt_move);
            let board_string = x.display();
            RESULT_MASKS.iter().for_each(|mask| println! {"mask:\t{}\nbitw:\t{}\nxbb:\t{}\n", mask, x.xbb & mask, x.xbb});

            assert_eq!(board_string, "OXO\n.X.\n.X.");
            assert_eq!(x.x_to_move, false);
            assert_eq!(x.get_win_state(), GameState::XWin);
        }

        #[test]
        fn x_win_tests() {
            let test_board = TTTBoard { xbb: 7, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "XXX\n...\n...");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 7 << 3, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "...\nXXX\n...");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 7 << 6, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "...\n...\nXXX");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 73, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "X..\nX..\nX..");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 73 << 1, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), ".X.\n.X.\n.X.");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 73 << 2, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "..X\n..X\n..X");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 273, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "X..\n.X.\n..X");
            assert_eq!(test_board.get_win_state(), GameState::XWin);

            let test_board = TTTBoard { xbb: 84, obb: 0, x_to_move: true };
            assert_eq!(test_board.display(), "..X\n.X.\nX..");
            assert_eq!(test_board.get_win_state(), GameState::XWin);
        }

        #[test]
        fn o_win_tests() {
            let test_board = TTTBoard { xbb: 0, obb: 7, x_to_move: true };
            assert_eq!(test_board.display(), "OOO\n...\n...");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 7 << 3, x_to_move: true };
            assert_eq!(test_board.display(), "...\nOOO\n...");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 7 << 6, x_to_move: true };
            assert_eq!(test_board.display(), "...\n...\nOOO");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 73, x_to_move: true };
            assert_eq!(test_board.display(), "O..\nO..\nO..");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 73 << 1, x_to_move: true };
            assert_eq!(test_board.display(), ".O.\n.O.\n.O.");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 73 << 2, x_to_move: true };
            assert_eq!(test_board.display(), "..O\n..O\n..O");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 273, x_to_move: true };
            assert_eq!(test_board.display(), "O..\n.O.\n..O");
            assert_eq!(test_board.get_win_state(), GameState::OWin);

            let test_board = TTTBoard { xbb: 0, obb: 84, x_to_move: true };
            assert_eq!(test_board.display(), "..O\n.O.\nO..");
            assert_eq!(test_board.get_win_state(), GameState::OWin);
        }

        #[test]
        fn movegen_test_1() {
            let mut x = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };

            while x.get_legal_moves().len() > 0 {
                let moves = x.get_legal_moves();
                x.make_move(&moves[0]);
                eprintln!("{}", x.display());
                assert_eq!(true, x.is_valid());
            }
        }

        #[test]
        fn alpha_beta_hash_test_1() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 0,
                obb: 44,
                x_to_move: true,
            };

            let mut ttable = HashMap::new();
            let (board_val, move_index) = ttt_board.alpha_beta_search(
                10, std::i8::MIN, std::i8::MAX, &mut ttable,
            );


//        let table_entry = ttable.entry(129);
            while (ttt_board.get_legal_moves().len() > 0) & (ttt_board.get_win_state() == GameState::Unresolved) {
                let (board_val, move_index) = ttt_board.alpha_beta_search(
                    10, std::i8::MIN, std::i8::MAX, &mut ttable,
                );
                let a_move = TTTMove { move_index, x_to_move: ttt_board.x_to_move };
                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
                eprintln!("{}", ttt_board.display());
            }
            assert_eq!(-128, board_val);

            assert_eq!(ttt_board.display(), "O.O\nOXO\nX.O");
        }

        #[test]
        fn alpha_beta_hash_test_2() {
            let mut ttt_board = TTTBoard {
                xbb: 0,
                obb: 0,
                x_to_move: true,
            };

            let mut ttable = HashMap::new();
            let (board_val, move_index) = ttt_board.alpha_beta_search(
                10, std::i8::MIN, std::i8::MAX, &mut ttable,
            );
            assert_eq!((0, 4), (board_val, move_index));

//        let now_instant = time::Instant::now();
            assert_eq!((0, 4), ttt_board.alpha_beta_search(
                10, std::i8::MIN, std::i8::MAX, &mut ttable,
            ));
//        assert_eq!(now_instant.elapsed().as_micros(),10);
//            assert_eq!(10, ttable.len());


//        let now_instant = time::Instant::now();
            while ttt_board.get_legal_moves().len() > 0 {
                let (board_val, move_index) = ttt_board.alpha_beta_search(
                    10, std::i8::MIN, std::i8::MAX, &mut ttable,
                );
                let a_move = TTTMove { move_index, x_to_move: ttt_board.x_to_move };
//            eprintln!("board:\n{}\nnext move:{:?}",ttt_board.display(),a_move);
                assert_eq!(true, ttt_board.is_valid());

                ttt_board.make_move(&a_move);
            }
//        assert_eq!(now_instant.elapsed().as_micros(),10);

            assert_eq!(ttt_board.display(), "OOX\nXXO\nOXX");
        }

        #[test]
        fn alpha_beta_hash_test_3() {
            let mut invalid_count = 0;
            let mut valid_count = 0;
            let mut ttable = HashMap::new();
            for side in &[true, false] {
                for xbb in 0..512 {
                    for obb in 0..512 {
                        let mut init_board = TTTBoard {
                            xbb,
                            obb,
                            x_to_move: *side,
                        };

                        if init_board.is_valid() {
                            valid_count += 1;

                            while (init_board.get_legal_moves().len() > 0) &&
                                (init_board.get_win_state() == GameState::Unresolved) {
                                let (board_val, move_index) = init_board.alpha_beta_search(
                                    10, std::i8::MIN, std::i8::MAX, &mut ttable,
                                );
                                let a_move = TTTMove { move_index, x_to_move: init_board.x_to_move };
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
//            assert_eq!(28090,ttable.len());
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
                visited_set.insert(this_hash);

                if a_board.get_win_state() == GameState::Unresolved {
                    let successors = a_board.get_legal_moves();
                    for each_move in &successors {
                        a_board.make_move(&each_move);
                        hash_sucessors(&mut a_board, &mut visited_set);
                        a_board.unmake_move(&each_move);
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

            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard, mut visited_set: &mut HashSet<u128>) {
                //            eprintln!("hash base: {}", hash_base);
                visited_set.insert(hash_base);
                if a_board.get_win_state() == GameState::Unresolved {
                    let successors = a_board.get_legal_moves();
                    for each_move in &successors {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        a_board.make_move(&each_move);
                        hash_paths(new_base, &mut a_board, &mut visited_set);
                        a_board.unmake_move(&each_move);
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
                if a_board.get_win_state() == GameState::Unresolved {
                    let successors = a_board.get_legal_moves();
                    for each_move in &successors {
                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
                        a_board.make_move(&each_move);
                        hash_paths(new_base, &mut a_board, &mut visited_set);
                        a_board.unmake_move(&each_move);
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


macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}
/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/


fn main() {
    // game loop
    use crate::ttt_solving::{TTTBoard, TTTMove};

    let mut ttt_board = TTTBoard {
        xbb: 0,
        obb: 0,
        x_to_move: true,
    };

    let mut transposition_table = HashMap::new();

    loop {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(" ").collect::<Vec<_>>();
        let opponent_row = parse_input!(inputs[0], i32);
        let opponent_col = parse_input!(inputs[1], i32);

        eprintln!("opponent row was: {} {}", opponent_row, opponent_col);
        eprintln!("start of turn board state was:\n{}", ttt_board.display());
        match (opponent_row, opponent_col) {
            (-1, -1) => {
                let (best_value, best_action) = ttt_board.alpha_beta_search(
                    10, std::i8::MIN, std::i8::MAX, &mut transposition_table);
                eprintln!("best value: {}\nbest action: {}", best_value, best_action);
                let my_move = TTTMove { move_index: best_action, x_to_move: ttt_board.x_to_move };
                ttt_board.make_move(&my_move);
                println!("{} {}", best_action / 3, best_action % 3);
            },
            (row, col) => {
                let opp_move_index = (3 * row + col) as u8;

                let opp_move = TTTMove { move_index: opp_move_index, x_to_move: ttt_board.x_to_move };
                ttt_board.make_move(&opp_move);
                eprintln!("board after opponent move was:\n{}", ttt_board.display());

                let (best_value, best_action) = ttt_board.alpha_beta_search(
                    10, std::i8::MIN, std::i8::MAX, &mut transposition_table);
                let my_move = TTTMove { move_index: best_action, x_to_move: ttt_board.x_to_move };
                ttt_board.make_move(&my_move);

                assert_eq!(true, ttt_board.is_valid());

                eprintln!("best value: {}\nbest action: {}", best_value, best_action);
                println!("{} {}", best_action / 3, best_action % 3);
            },
        }
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let valid_action_count = parse_input!(input_line, i32);
        eprintln!("Valid moves:");

        let mut actions = Vec::new();

        for _i in 0..valid_action_count as usize {
            let mut input_line = String::new();
            io::stdin().read_line(&mut input_line).unwrap();
            let inputs = input_line.split(" ").collect::<Vec<_>>();
            let row = parse_input!(inputs[0], i32);
            let col = parse_input!(inputs[1], i32);
            actions.push((row, col));
            eprintln!("\t{} {}", row, col);
        }
//        println!("{} {}",actions[0].0, actions[0].1);
    }
}


