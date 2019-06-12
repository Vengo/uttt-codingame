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
        fn make_move(&mut self, a_move: &GameMove) -> GameUnmove;
        fn unmake_move(&mut self, a_move: GameUnmove);
        fn score(&self) -> i8;
        fn get_legal_moves(&self) -> Vec<GameMove>;
        fn display(&self) -> String;
        fn is_valid(&self) -> bool;
    }

    const LT: fn(i8, i8) -> bool = |a, b| a.lt(&b);
    const MIN: fn(i8, i8) -> i8 = |a, b| a.min(b);

    const GT: fn(i8, i8) -> bool = |a, b| a.gt(&b);
    const MAX: fn(i8, i8) -> i8 = |a, b| a.max(b);

    pub fn alpha_beta_search(board: &mut impl AlphaBetaTTCompatible, depth: u8, alpha: i8, beta: i8, transpositions: &mut HashMap<u32, HashState>) -> (i8, u8) {
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
        let moves = board.get_legal_moves();
        let mut move_index = moves[0].move_index;

        for each_move in moves.iter() {
            let unmake_instructions = board.make_move(&each_move);
            let (move_score, _move_index) = {
                alpha_beta_search(
                    board, depth - 1, node_alpha, node_beta, transpositions,
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

        let side_moved = if board.x_to_move() { "X" } else { "O" };
        return (new_score, move_index);
    }
}

mod ttt_trait_solving {
    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult, GameUnmove};

    // tictactoe constants
    const TTT_BOARD_WIDTH: usize = 3;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    //    const MOVE_ORDERING: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const CAT: u16 = (1 << 9) - 1;

    #[derive(Debug)]
    pub struct TTTBoard {
        pub xbb: u16,
        pub obb: u16,
        pub x_to_move: bool,
    }

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

        fn make_move(&mut self, a_move: &GameMove) -> GameUnmove {
//            pub x_to_move: bool,
//            pub move_index: u8,
            let unmove = GameUnmove { x_to_move: self.x_to_move, move_index: a_move.move_index };

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
//            assert_eq!(true, self.is_valid());
            return unmove;
        }

        fn unmake_move(&mut self, unmove: GameUnmove) {
            match unmove {
                GameUnmove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u16 << move_index;
                    self.x_to_move = true;
                },
                GameUnmove { x_to_move: false, move_index } => {
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

            while x.get_legal_moves().len() > 0 {
                let moves = x.get_legal_moves();
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
        fn alpha_beta_hash_test_1b() {
            // 256 128 64 32 16 8 4 2 1
            // 1 2 4 8 16 32 64 128 256
            let mut ttt_board = TTTBoard {
                xbb: 1,
                obb: 8,
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

            assert_eq!(
                (0, 4),
                alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN,
                    std::i8::MAX, &mut ttable)
            );


            let now_instant = time::Instant::now();
            while ttt_board.get_legal_moves().len() > 0 {
                let (board_val, move_index) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
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
                visited_set.insert(this_hash);

                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves();
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

            fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard, mut visited_set: &mut HashSet<u128>) {
                //            eprintln!("hash base: {}", hash_base);
                visited_set.insert(hash_base);
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves();
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
                if a_board.get_win_state() == GameResult::Unresolved {
                    let successors = a_board.get_legal_moves();
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

mod uttt_trait_solving {
    use crate::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult, GameUnmove};

    // tictactoe constants
    const UTTT_BOARD_WIDTH: usize = 9;
    const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
    const SUPER_MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const SUB_MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
    const SQUARE_ORDERING: [u8; 81] = [40, 36, 38, 42, 39, 41, 44, 37, 43, 4, 0, 2, 6, 3, 5, 8, 1,
        7, 22, 18, 20, 24, 21, 23, 26, 19, 25, 58, 54, 56, 60, 57, 59, 62, 55, 61, 31, 27, 29, 33,
        30, 32, 35, 28, 34, 49, 45, 47, 51, 48, 50, 53, 46, 52, 76, 72, 74, 78, 75, 77, 80, 73, 79,
        13, 9, 11, 15, 12, 14, 17, 10, 16, 67, 63, 65, 69, 66, 68, 71, 64, 70];
    //    const SQUARE_ORDERING : [u8; 81] = SUPER_MOVE_ORDERING.iter().flat_map(
//        |board_index| SUB_MOVE_ORDERING.iter().map(
//            |square_index| board_index*9+square_index)).collect();
    //    const MOVE_ORDERING: [u8; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const CAT: u128 = (1 << 81) - 1;

    #[derive(Debug)]
    pub struct UTTTBoard {
        pub xbb: u128,
        pub x_supers: u16,
        pub obb: u128,
        pub o_supers: u16,
        pub last_super_index: u8,
        pub x_to_move: bool,
    }

    #[derive(PartialEq, Eq, Debug)]
    struct SubBoardAndSquare {
        sub_board_index: u8,
        sub_square_index: u8,
    }

    fn parse_index(an_index: &u8) -> SubBoardAndSquare {
        // Returns the sub board [0,8] and sub square [0-8]
        let sub_square = an_index % 9;
        let sub_board = an_index / 9;

        return SubBoardAndSquare { sub_square_index: sub_square, sub_board_index: sub_board };
    }

    impl AlphaBetaTTCompatible for UTTTBoard {
        fn hash_val(&self) -> u32 {
//            let base_hash = (self.xbb as u32) << 16 | (self.obb as u32);
//            return match self.x_to_move {
//                true => { base_hash },
//                false => { !base_hash },
//            };
            return 0;
        }

        fn get_win_state(&self) -> GameResult {
            // check for X or O win patterns, check for a filled board, or return unresolved
            if RESULT_MASKS.iter().any(|x| (x & self.x_supers).eq(x)) {
                return GameResult::P1Win;
            } else if RESULT_MASKS.iter().any(|x| (x & self.o_supers).eq(x)) {
                return GameResult::P2Win;
            } else if (self.xbb | self.obb) == CAT {
                return GameResult::Tie;
            } else {
                return GameResult::Unresolved;
            };
            return GameResult::Unresolved;
        }

        fn x_to_move(&self) -> bool {
            return self.x_to_move;
        }

        fn make_move(&mut self, a_move: &GameMove) -> GameUnmove {
            let GameMove { x_to_move, move_index: a_move_index } = a_move;
            let unmove = GameUnmove { x_to_move: x_to_move.clone(), move_index: a_move_index.clone() };
            let SubBoardAndSquare { sub_board_index, sub_square_index } = parse_index(a_move_index);
//            eprintln!("{} {}", sub_board_index, sub_square_index);

            // TODO: Remember to only unmake sub-board wins for the first
            // move that wins that square
            match a_move {
                GameMove { x_to_move: true, move_index } => {
                    self.xbb ^= 1u128 << move_index;
                    let sub_board = ((self.xbb >> (9 * sub_board_index)) as u16) & ((1 << 9) - 1 as u16);
                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & sub_board).eq(x));

                    if is_sub_win {
                        self.x_supers |= (1 << sub_board_index);
                    }
                    self.last_super_index = sub_square_index;
                    self.x_to_move = false;
                },
                GameMove { x_to_move: false, move_index } => {
                    self.obb ^= 1u128 << move_index;
                    let sub_board = ((self.obb >> (9 * sub_board_index)) as u16) & ((1 << 9) - 1 as u16);
                    let is_sub_win = RESULT_MASKS.iter().any(|x| (x & sub_board).eq(x));

                    if is_sub_win {
                        self.o_supers |= (1 << sub_board_index);
                    }
                    self.last_super_index = sub_square_index;
                    self.x_to_move = true;
                },
            }


            return unmove;
        }

        fn unmake_move(&mut self, unmove: GameUnmove) {
//            match unmove {
//                GameUnmove { x_to_move: true, move_index } => {
//                    self.xbb ^= 1u16 << move_index;
//                    self.x_to_move = true;
//                },
//                GameUnmove { x_to_move: false, move_index } => {
//                    self.obb ^= 1u16 << move_index;
//                    self.x_to_move = false;
//                },
//            }
        }

        fn score(&self) -> i8 {
            let val = match self.get_win_state() {
                GameResult::P1Win => { std::i8::MAX },
                GameResult::P2Win => { std::i8::MIN },
                GameResult::Tie => { 0i8 },
                GameResult::Unresolved if self.x_to_move => { 1i8 },
                GameResult::Unresolved => { -1i8 },
            };
            return val;
        }

        fn get_legal_moves(&self) -> Vec<GameMove> {
            let won_bbs = self.x_supers | self.o_supers;

            let directed_xbb = ((self.xbb >> (9 * self.last_super_index)) as u16) &
                ((1 << 9) - 1);
            let directed_obb = ((self.obb >> (9 * self.last_super_index)) as u16) & ((1 << 9) - 1);
            let directed_occ = directed_obb | directed_xbb;

            let directed_sub_board_index = match self.last_super_index {
                value if value >= 9 => { (1u128 << 81) - 1 },
                value if (won_bbs & (1 << value)) != 0 => { (1u128 << 81) - 1 },
                value if directed_occ == ((1 << 9) - 1) => { (1u128 << 81) - 1 },
                value => { ((1u128 << 9) - 1) << 9 * value },
            };
            let occupied_bb = self.xbb | self.obb;
            let open_and_legal = !occupied_bb & directed_sub_board_index;

//            eprintln!("directed board: {}",self.last_super_index);
//            eprintln!("directed xbb: {:b}",directed_xbb);
//            eprintln!("directed obb: {:b}",directed_obb);
//            eprintln!("directed occ: {:b}",directed_occ);
//            eprintln!("won bbs: {:b}",won_bbs);
//            eprintln!("occ: {:b}",occupied_bb);
//            eprintln!("dir: {:b}",directed_sub_board_index);
//            eprintln!("oal: {:b}",open_and_legal);


            let valid_moves: Vec<GameMove> = SQUARE_ORDERING.iter().filter_map(|a_move_index| {
                match open_and_legal & (1u128 << a_move_index) {
                    0 => None,
//                    nonzero if => Some(GameMove { x_to_move: self.x_to_move, move_index: a_move_index.clone() }),
                    nonzero => Some(GameMove { x_to_move: self.x_to_move, move_index: a_move_index.clone() }),
                }
            }).collect();

//            eprintln!("{}",valid_moves.len());
            return valid_moves;
        }

        fn display(&self) -> String {
            //        let mut board_rep = String::new();
            let mut rows = Vec::new();
            //  0  1  2  9 10 11 18 19 20
            //  3  4  5 12 13 14 21 22 23
            //  6  7  8 15 16 17 24 25 26
            // 27 28 29 36 37 38 45 46 47
            // 30 31 32 39 40 41 48 49 50
            // 33 34 35 42 43 44 51 52 53
            //

            for super_row_index in 0..3 {
                for sub_row_index in 0..3 {
                    let mut row_rep = String::new();
                    let indices: [u128; 9] = [ //
                        (27 * super_row_index + sub_row_index * 3),
                        (27 * super_row_index + sub_row_index * 3 + 1),
                        (27 * super_row_index + sub_row_index * 3 + 2),
                        (27 * super_row_index + sub_row_index * 3 + 9),
                        (27 * super_row_index + sub_row_index * 3 + 10),
                        (27 * super_row_index + sub_row_index * 3 + 11),
                        (27 * super_row_index + sub_row_index * 3 + 18),
                        (27 * super_row_index + sub_row_index * 3 + 19),
                        (27 * super_row_index + sub_row_index * 3 + 20)
                    ];

                    for an_index in indices.iter() {
                        let mask = 1u128 << an_index;
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
            alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult, GameUnmove, HashState};
        use crate::uttt_trait_solving::{parse_index, SubBoardAndSquare};

        use super::{RESULT_MASKS, UTTTBoard};

        #[test]
        fn square_ordering_test() {
            use super::{SUPER_MOVE_ORDERING, SUB_MOVE_ORDERING};
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
                xbb: 1 << 20,
                obb: 0,
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: false,
            };
            let board_string = x.display();
            assert_eq!(board_string, "........X\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........");
        }

        #[test]
        fn display_test2() {
            let x = UTTTBoard {
                xbb: 1 << 20,
                obb: 1 << 2,
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: false,
            };
            let board_string = x.display();
            eprintln!("{}", board_string);
            assert_eq!(board_string, "..O.....X\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........");
        }

        #[test]
        fn make_and_get_moves_test_1() {
            let mut x = UTTTBoard {
                xbb: 0,
                obb: 0,
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
            };
            let board_string = x.display();
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, true);
            assert_eq!(9, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 40, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            eprintln!("{}\n", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, false);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 36, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            eprintln!("{}\n", board_string);
            assert_eq!(board_string, ".........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, true);
            assert_eq!(0, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 0, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, false);
            assert_eq!(0, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 4, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O.......\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, true);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 37, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O.......\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, false);
            assert_eq!(1, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            //testing an x move
            let ttt_move = GameMove { move_index: 13, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, true);
            assert_eq!(4, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(0, x.x_supers);

            let ttt_move = GameMove { move_index: 43, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.........");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 71, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
//            eprintln!("{}\n", board_string);
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O...");
            assert_eq!(x.x_to_move, true);
            assert_eq!(8, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 79, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O.X.");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 69, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n...O.O.X.");
            assert_eq!(x.x_to_move, true);
            assert_eq!(6, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 61, x_to_move: true };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.X.O.O.X.");
            assert_eq!(x.x_to_move, false);
            assert_eq!(7, x.last_super_index);
            assert_eq!(0, x.o_supers);
            assert_eq!(1 << 4, x.x_supers);

            let ttt_move = GameMove { move_index: 70, x_to_move: false };
            let unmake_move = x.make_move(&ttt_move);
            let board_string = x.display();
            assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.X.OOO.X.");
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
            let mut x = UTTTBoard {
                xbb: 0,
                obb: 0,
                last_super_index: 9,
                x_supers: 0,
                o_supers: 0,
                x_to_move: true,
            };

            let mut move_count = 0;
            for i in 0..2500000u64 {
                while x.get_win_state() == GameResult::Unresolved {
//                eprintln!("{}", x.display());
                    let moves = x.get_legal_moves();
                    let first_move = moves.get(0);
                    match first_move {
                        Some(a_move) => {
                            x.make_move(a_move);
                            move_count += 1;
                        },
                        None => break,
                    }
                }
                assert_eq!(81, move_count);
            }

//            assert_eq!(x.display(),"I have no idea what I'll get here");
        }
//
//        #[test]
//        fn make_and_unmake_move_test_1() {
//            let mut x = UTTTBoard {
//                xbb: 0,
//                obb: 0,
//                last_super_index: 9,
//                won_supers: 0,
//                x_to_move: true,
//            };
//            let board_string = x.display();
//            assert_eq!(board_string, ".........\n.........\n.........\n.........\n.........\n.........\n.........\n.........\n.........");
//            assert_eq!(x.x_to_move, true);
//
//
//            //testing an x move
//            let ttt_move = GameMove { move_index: 40, x_to_move: true };
//            let unmake_move = x.make_move(&ttt_move);
//            let board_string = x.display();
//            eprintln!("{}",board_string);
//            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........");
//            assert_eq!(x.x_to_move, false);
//
//            // testing an x unmake move
//            x.unmake_move(unmake_move);
//            let board_string = x.display();
//            assert_eq!(board_string, ".........\n.........\n.........\n.........\n....X....\n.........\n.........\n.........\n.........");
//            assert_eq!(x.x_to_move, true);
//
//            //readding an x move
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
//        }

//        #[test]
//        fn x_win_tests() {
//            let test_board = UTTTBoard { xbb: 7, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "XXX\n...\n...");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 7 << 3, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "...\nXXX\n...");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 7 << 6, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "...\n...\nXXX");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 73, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "X..\nX..\nX..");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 73 << 1, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), ".X.\n.X.\n.X.");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 73 << 2, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "..X\n..X\n..X");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 273, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "X..\n.X.\n..X");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//
//            let test_board = UTTTBoard { xbb: 84, obb: 0, x_to_move: true };
//            assert_eq!(test_board.display(), "..X\n.X.\nX..");
//            assert_eq!(test_board.get_win_state(), GameResult::P1Win);
//        }
//
//        #[test]
//        fn o_win_tests() {
//            let test_board = UTTTBoard { xbb: 0, obb: 7, x_to_move: true };
//            assert_eq!(test_board.display(), "OOO\n...\n...");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 7 << 3, x_to_move: true };
//            assert_eq!(test_board.display(), "...\nOOO\n...");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 7 << 6, x_to_move: true };
//            assert_eq!(test_board.display(), "...\n...\nOOO");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 73, x_to_move: true };
//            assert_eq!(test_board.display(), "O..\nO..\nO..");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 73 << 1, x_to_move: true };
//            assert_eq!(test_board.display(), ".O.\n.O.\n.O.");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 73 << 2, x_to_move: true };
//            assert_eq!(test_board.display(), "..O\n..O\n..O");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 273, x_to_move: true };
//            assert_eq!(test_board.display(), "O..\n.O.\n..O");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//
//            let test_board = UTTTBoard { xbb: 0, obb: 84, x_to_move: true };
//            assert_eq!(test_board.display(), "..O\n.O.\nO..");
//            assert_eq!(test_board.get_win_state(), GameResult::P2Win);
//        }
//
//        #[test]
//        fn movegen_test_1() {
//            let mut x = UTTTBoard {
//                xbb: 0,
//                obb: 0,
//                x_to_move: true,
//            };
//
//            while x.get_legal_moves().len() > 0 {
//                let moves = x.get_legal_moves();
//                x.make_move(&moves[0]);
//                eprintln!("{}", x.display());
//                assert_eq!(true, x.is_valid());
//            }
//        }
//
//        #[test]
//        fn alpha_beta_hash_test_1a() {
//            // 256 128 64 32 16 8 4 2 1
//            // 1 2 4 8 16 32 64 128 256
//            let mut ttt_board = UTTTBoard {
//                xbb: 0,
//                obb: 44,
//                x_to_move: true,
//            };
//
//            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
//
//
////        let table_entry = ttable.entry(129);
//            while (ttt_board.get_legal_moves().len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
//                let (board_val, move_index) = alpha_beta_search(
//                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//                );
//                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
//                assert_eq!(true, ttt_board.is_valid());
//
//                ttt_board.make_move(&a_move);
//                eprintln!("{}", ttt_board.display());
//            }
//            assert_eq!(-128, board_val);
//
//            assert_eq!(ttt_board.display(), "O.O\nOXO\nX.O");
//        }
//
//        #[test]
//        fn alpha_beta_hash_test_1b() {
//            // 256 128 64 32 16 8 4 2 1
//            // 1 2 4 8 16 32 64 128 256
//            let mut ttt_board = UTTTBoard {
//                xbb: 1,
//                obb: 8,
//                x_to_move: true,
//            };
//
//            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
//
//
////        let table_entry = ttable.entry(129);
//            while (ttt_board.get_legal_moves().len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
//                let (board_val, move_index) = alpha_beta_search(
//                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//                );
//                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
//                assert_eq!(true, ttt_board.is_valid());
//
//                ttt_board.make_move(&a_move);
//                eprintln!("{}", ttt_board.display());
//                assert_eq!(127, board_val);
//            }
//            assert_eq!(127, board_val);
//
//            assert_eq!(ttt_board.display(), "X.O\nOX.\n..X");
//        }
//
//        #[test]
//        fn alpha_beta_hash_test_1c() {
//            // 256 128 64 32 16 8 4 2 1
//            // 1 2 4 8 16 32 64 128 256
//            let mut ttt_board = UTTTBoard {
//                xbb: 81,
//                obb: 12,
//                x_to_move: false,
//            };
//
//            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
//
//
////        let table_entry = ttable.entry(129);
//            while (ttt_board.get_legal_moves().len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
//                let (board_val, move_index) = alpha_beta_search(
//                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//                );
//                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
//                assert_eq!(true, ttt_board.is_valid());
//
//                ttt_board.make_move(&a_move);
//                eprintln!("{}", ttt_board.display());
//                assert_eq!(0, board_val);
//            }
//            assert_eq!(0, board_val);
//
//            assert_eq!(ttt_board.display(), "XOO\nOXX\nXXO");
//        }
//
//        #[test]
//        fn alpha_beta_hash_test_1d() {
//            // 256 128 64 32 16 8 4 2 1
//            // 1 2 4 8 16 32 64 128 256
//            let mut ttt_board = UTTTBoard {
//                xbb: 81,
//                obb: 268,
//                x_to_move: true,
//            };
//
//            let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
//
//
////        let table_entry = ttable.entry(129);
//            while (ttt_board.get_legal_moves().len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
//                let (board_val, move_index) = alpha_beta_search(
//                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//                );
//                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
//                assert_eq!(true, ttt_board.is_valid());
//
//                ttt_board.make_move(&a_move);
//                eprintln!("{}", ttt_board.display());
//                assert_eq!(0, board_val);
//            }
//            assert_eq!(0, board_val);
//
//            assert_eq!(ttt_board.display(), "XOO\nOXX\nXXO");
//        }
//
//
//        #[test]
//        fn alpha_beta_hash_test_2() {
//            let mut ttt_board = UTTTBoard {
//                xbb: 0,
//                obb: 0,
//                x_to_move: true,
//            };
//
//            let mut ttable = HashMap::new();
////            let (board_val, move_index) = alpha_beta_search(
////                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable
////            );
//
//            assert_eq!(
//                (0, 4),
//                alpha_beta_search(
//                    &mut ttt_board, 10, std::i8::MIN,
//                    std::i8::MAX, &mut ttable )
//            );
//
//
//            let now_instant = time::Instant::now();
//            while ttt_board.get_legal_moves().len() > 0 {
//                let (board_val, move_index) = alpha_beta_search(
//                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//                );
//                let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                assert_eq!(true, ttt_board.is_valid());
//
//                ttt_board.make_move(&a_move);
//            }
//
//            assert_eq!(ttt_board.display(), "OOX\nXXO\nOXX");
//        }
//
//        #[test]
//        fn alpha_beta_hash_test_3() {
//            let mut invalid_count = 0;
//            let mut valid_count = 0;
//            let mut ttable = HashMap::new();
//            for side in &[true, false] {
//                for xbb in (0..512).rev() {
//                    for obb in (0..512).rev() {
//                        let mut init_board = UTTTBoard {
//                            xbb,
//                            obb,
//                            x_to_move: *side,
//                        };
//                        if init_board.is_valid() {
//                            valid_count += 1;
//
//                            while (init_board.get_legal_moves().len() > 0) &&
//                                (init_board.get_win_state() == GameResult::Unresolved) {
//                                let (board_val, move_index) = alpha_beta_search(
//                                    &mut init_board, 10, std::i8::MIN, std::i8::MAX, &mut ttable,
//                                );
//                                let a_move = GameMove { move_index, x_to_move: init_board.x_to_move };
//                                //            eprintln!("board:\n{}\nnext move:{:?}",ttt_board.display(),a_move);
//                                assert_eq!(true, init_board.is_valid());
//
//                                init_board.make_move(&a_move);
//                            }
//                        } else {
//                            invalid_count += 1;
//                        }
//                    }
//                }
//            };
//            assert_eq!(28036,ttable.len());
//            assert_eq!(484922, invalid_count);
//            assert_eq!(39366, valid_count);
//        }
//
//        #[test]
//        fn perft_positions() {
//            let mut visited = HashSet::new();
//            let mut init_board = UTTTBoard {
//                xbb: 0,
//                obb: 0,
//                x_to_move: true,
//            };
//
//            fn hash_sucessors(mut a_board: &mut UTTTBoard, mut visited_set: &mut HashSet<u32>) {
//                let this_hash = a_board.hash_val();
//                visited_set.insert(this_hash);
//
//                if a_board.get_win_state() == GameResult::Unresolved {
//                    let successors = a_board.get_legal_moves();
//                    for each_move in &successors {
//                        let an_unmake = a_board.make_move(&each_move);
//                        hash_sucessors(&mut a_board, &mut visited_set);
//                        a_board.unmake_move(an_unmake);
//                    }
//                }
//            }
//            hash_sucessors(&mut init_board, &mut visited);
//            assert_eq!(5478, visited.len());
//        }
//
//        #[test]
//        fn perft_paths() {
//            let mut paths = HashSet::new();
//            let mut init_board = UTTTBoard {
//                xbb: 0,
//                obb: 0,
//                x_to_move: true,
//            };
//
//            fn hash_paths(hash_base: u128, mut a_board: &mut UTTTBoard, mut visited_set: &mut HashSet<u128>) {
//                //            eprintln!("hash base: {}", hash_base);
//                visited_set.insert(hash_base);
//                if a_board.get_win_state() == GameResult::Unresolved {
//                    let successors = a_board.get_legal_moves();
//                    for each_move in &successors {
//                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
//                        let unmake_move = a_board.make_move(&each_move);
//                        hash_paths(new_base, &mut a_board, &mut visited_set);
//                        a_board.unmake_move(unmake_move);
//                    }
//                }
//            }
//            hash_paths(1, &mut init_board, &mut paths);
//            assert_eq!(549946, paths.len());
//        }
//
//        #[test]
//        fn perft_games() {
//            let mut paths = HashSet::new();
//            let mut init_board = UTTTBoard {
//                xbb: 0,
//                obb: 0,
//                x_to_move: true,
//            };
//
//            fn hash_paths(hash_base: u128, mut a_board: &mut UTTTBoard, mut visited_set: &mut HashSet<u128>) {
//                //            eprintln!("hash base: {}", hash_base);
//                if a_board.get_win_state() == GameResult::Unresolved {
//                    let successors = a_board.get_legal_moves();
//                    for each_move in successors.iter() {
//                        let new_base = (hash_base << 8) + (each_move.move_index as u128);
//                        let unmake_move = a_board.make_move(each_move);
//                        hash_paths(new_base, &mut a_board, &mut visited_set);
//                        a_board.unmake_move(unmake_move);
//                    }
//                } else {
//                    visited_set.insert(hash_base);
//                }
//            }
//            hash_paths(1, &mut init_board, &mut paths);
//            assert_eq!(255168, paths.len());
//        }
//    }
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

    use crate::ttt_trait_solving::{TTTBoard};
    use crate::game_solving::{alpha_beta_search, AlphaBetaTTCompatible, GameMove};
    use std::collections::HashMap;

    let mut ttt_board = TTTBoard {
        xbb: 0,
        obb: 0,
        x_to_move: true,
    };

    let mut transposition_table = HashMap::new();

    loop {
        let mut input_line = String::new();
        std::io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(" ").collect::<Vec<_>>();
        let opponent_row = parse_input!(inputs[0], i32);
        let opponent_col = parse_input!(inputs[1], i32);

        eprintln!("opponent row was: {} {}", opponent_row, opponent_col);
        eprintln!("start of turn board state was:\n{}", ttt_board.display());
        match (opponent_row, opponent_col) {
            (-1, -1) => {
                let (best_value, best_action) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut transposition_table);
                eprintln!("best value: {}\nbest action: {}", best_value, best_action);
                let my_move = GameMove { move_index: best_action, x_to_move: ttt_board.x_to_move };
                ttt_board.make_move(&my_move);
                println!("{} {}", best_action / 3, best_action % 3);
            },
            (row, col) => {
                let opp_move_index = (3 * row + col) as u8;

                let opp_move = GameMove { move_index: opp_move_index, x_to_move: ttt_board.x_to_move };
                ttt_board.make_move(&opp_move);
                eprintln!("board after opponent move was:\n{}", ttt_board.display());

                let (best_value, best_action) = alpha_beta_search(
                    &mut ttt_board, 10, std::i8::MIN, std::i8::MAX, &mut transposition_table);
                let my_move = GameMove { move_index: best_action, x_to_move: ttt_board.x_to_move };
                ttt_board.make_move(&my_move);

                assert_eq!(true, ttt_board.is_valid());

                eprintln!("best value: {}\nbest action: {}", best_value, best_action);
                println!("{} {}", best_action / 3, best_action % 3);
            },
        }
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
//        println!("{} {}",actions[0].0, actions[0].1);
    }
}


