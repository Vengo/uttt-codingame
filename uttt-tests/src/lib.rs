#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

#[cfg(test)]
mod tests_ttt {
    use std::collections::{HashMap, HashSet};

    use uttt::ttt_trait_solving::{TTTBoard, TTTZobrist};
    use uttt::game_solving::{AlphaBetaTTCompatible,alpha_beta_search, GameMove, GameResult};

    #[test]
    fn test_move_ordering_array() {
        let array = TTTBoard::get_move_ordering_array();

        let board = TTTBoard{x_to_move:true, xbb:2, obb:1};
        println!("board is:\n{}",board.display());

        let legal = board.get_legal_moves(&array);
        println!("legal moves: {:?}",legal);

        let board = TTTBoard{x_to_move:true, xbb:128, obb:256};
        println!("board is:\n{}",board.display());

        let legal = board.get_legal_moves(&array);
        println!("legal moves: {:?}",legal);

        let board = TTTBoard{x_to_move:true, xbb:128+16, obb:256};
        println!("board is:\n{}",board.display());

        let legal = board.get_legal_moves(&array);
        println!("legal moves: {:?}",legal);

        assert!(false);
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
//            RESULT_MASKS.iter().for_each(|mask| println! {"mask:\t{}\nbitw:\t{}\nxbb:\t{}\n", mask, x.xbb & mask, x.xbb});

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
//                eprintln!("{}", x.display());
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
        let (board_val, _board_upper, _move_index, _found_terminal) = alpha_beta_search(
            &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
            &mut ttable, &legal_moves, &zobrist_constants
        );


//        let table_entry = ttable.entry(129);
        while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
            let (_board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );
            let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}", ttt_board.display(), a_move);
//                assert_eq!(true, ttt_board.is_valid());

            ttt_board.make_move(a_move, &zobrist_constants);
//                eprintln!("{}", ttt_board.display());
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
        let (board_val, _board_upper, _move_index, _found_terminal) = alpha_beta_search(
            &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
            &mut ttable, &legal_moves, &zobrist_constants
        );

//        let table_entry = ttable.entry(129);
        while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
//                eprintln!("\n====================\n");
            let (board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );
            let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
//                eprintln!("board:\n{}\nnext move:{:?}\nevaluation {}", ttt_board.display(), a_move, board_val);
            assert_eq!(true, ttt_board.is_valid());

            ttt_board.make_move(a_move, &zobrist_constants);
//                eprintln!("{}", ttt_board.display());
            assert_eq!(127, board_val);
        }
//            eprintln!("game complete:\n{}", ttt_board.display());
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
        let legal_moves = TTTBoard::get_move_ordering_array();
        let zobrist_constants = TTTZobrist{};

        let mut ttable = HashMap::new();
        let (board_val, _board_upper, _move_index, _found_terminal) = alpha_beta_search(
            &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
            &mut ttable, &legal_moves, &zobrist_constants
        );

        let legal_moves = Vec::new();

//        let table_entry = ttable.entry(129);
        while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
            let (board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );
            let a_move = GameMove { move_index, x_to_move: ttt_board.x_to_move };
            assert_eq!(true, ttt_board.is_valid());

            ttt_board.make_move(a_move, &zobrist_constants);
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
        let (board_val, _board_upper, _move_index, _found_terminal) = alpha_beta_search(
            &mut ttt_board, 10, std::i8::MIN, std::i8::MAX,
            &mut ttable, &legal_moves, &zobrist_constants
        );

        while (ttt_board.get_legal_moves(&legal_moves).len() > 0) & (ttt_board.get_win_state() == GameResult::Unresolved) {
            let (board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
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
    fn alpha_beta_hash_test_2() {
        let mut ttt_board = TTTBoard {
            xbb: 0,
            obb: 0,
            x_to_move: true,
        };
        let zobrist_constants = TTTZobrist{};

        let mut ttable = HashMap::new();
        let legal_moves = Vec::new();
        assert_eq!(
            (0, 0, 4, true),
            alpha_beta_search(
                &mut ttt_board, 10, std::i8::MIN,
                std::i8::MAX, &mut ttable, &legal_moves, &zobrist_constants)
        );

        while ttt_board.get_legal_moves(&legal_moves).len() > 0 {
            let (_board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
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
                            let (_board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
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

        fn hash_successors(a_board: &mut TTTBoard,
                           visited_set: &mut HashSet<u32>, zobrist_constants: &TTTZobrist
        ) {
            let this_hash = a_board.hash_val();
            let legal_moves = Vec::new();
            visited_set.insert(this_hash);

            if a_board.get_win_state() == GameResult::Unresolved {
                let successors = a_board.get_legal_moves(&legal_moves);
                for each_move in &successors {
                    let an_unmake = a_board.make_move(*each_move, &zobrist_constants);
                    hash_successors(a_board, visited_set, zobrist_constants);
                    a_board.unmake_move(an_unmake);
                }
            }
        }
        hash_successors(&mut init_board, &mut visited, &zobrist_constants);
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

        fn hash_paths(hash_base: u128, mut a_board: &mut TTTBoard,
                      mut visited_set: &mut HashSet<u128>, zobrist_constants: &TTTZobrist
        ) {
            //            eprintln!("hash base: {}", hash_base);
            let legal_moves = Vec::new();
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
            let legal_moves = Vec::new();
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

#[cfg(test)]
mod tests_uttt {
    use std::collections::{HashMap, HashSet};
    use std::time;

    use uttt::game_solving::{
        alpha_beta_search, AlphaBetaTTCompatible, GameMove, GameResult, advance_rollout,
        reverse_rollout, IdentityHasherBuilder
    };
    use uttt::uttt_trait_solving::{UTTTBoard, parse_index, SubBoardAndSquare, ZobristConstants};

    #[test]
    fn zobrist_constants_test() {
        let constants = UTTTBoard::get_zobrist_constants();

        let start_val = constants.x_to_move;

        let x_iter = constants.x_constants.iter();
        let o_iter = constants.o_constants.iter();

        let all_hashed = x_iter.chain(o_iter).fold(start_val, |a,b| a^b);

        let expected_allhash = 10891453110652296350u64;

        eprintln!("{} {}", all_hashed, expected_allhash);
        assert_eq!(expected_allhash, all_hashed);
    }

    #[test]
    fn advance_rollout_test_nobackup() {
        let mut unmove_queue = Vec::with_capacity(81);

        let mut node_store= HashMap::with_capacity_and_hasher(2_000_000, IdentityHasherBuilder{});
        let legal_moves = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();

        let search_alpha = std::i8::MIN;
        let search_beta  = std::i8::MAX;

        let time_limit = 990_000;
        let mut rollout_count = 0;


        let mut uttt_board = UTTTBoard::default();
        let first_move = GameMove{x_to_move:true, move_index: 40};
        uttt_board.make_move(first_move, &zobrist_constants);

        let orig_board = uttt_board.clone();
        let now_instant = time::Instant::now();
        while now_instant.elapsed().as_micros()<time_limit {
            uttt_board = orig_board.clone();
            while let Some(_new_alpha_beta) = advance_rollout(&mut unmove_queue,
                                                             &mut uttt_board, &legal_moves, search_alpha, search_beta, &mut node_store, &zobrist_constants){
//                    let (new_alpha, new_beta) = new_alpha_beta;
//                    search_alpha = new_alpha;
//                    search_beta = new_beta;
            };
//            let rollout_result = search_alpha;

            assert_eq!(uttt_board.get_win_state()!= GameResult::Unresolved, true);
            rollout_count +=1;
        }

        eprintln!("completed {} rollouts no backup", rollout_count);
        assert_eq!(true, rollout_count>4000);
    }

    #[test]
    fn perft_positions() {
        let mut visited = HashSet::new();
        let mut init_board = UTTTBoard::default();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();
        let legal_moves = UTTTBoard::get_move_ordering_array();


        fn hash_successors(depth: u8, a_board: &mut UTTTBoard,
                           visited_set: &mut HashSet<u64>, legal_moves: &Vec<Vec<GameMove>>,
                           zobrist_constants: &ZobristConstants, leaf_count: &mut usize
        ) {
            let this_hash = a_board.hash_val();
            visited_set.insert(this_hash);

            if depth == 0 {

            } else if a_board.get_win_state() == GameResult::Unresolved {
                let successors = a_board.get_legal_moves(&legal_moves);
                if depth == 1 {
                    *leaf_count += successors.len();
                } else {
                    for each_move in &successors {
//                        let an_unmake = a_board.make_move(*each_move, &zobrist_constants);
//                        hash_successors(depth-1,&mut a_board, &mut visited_set, legal_moves, zobrist_constants);
//                        a_board.unmake_move(an_unmake);
                        let mut new_board = a_board.clone();
                        new_board.make_move(*each_move, &zobrist_constants);
                        hash_successors(depth-1,&mut new_board, visited_set, legal_moves, zobrist_constants, leaf_count);
                    }
                }
            }
        }
        for i in 0..=5 {
            let expt_start = time::Instant::now();
            let mut leaf_count = 0;
            hash_successors(i,&mut init_board, &mut visited, &legal_moves, &zobrist_constants, &mut leaf_count);

            eprintln!("found {} positions at depth {} in {} ms", visited.len()+leaf_count, i, expt_start.elapsed().as_millis())
        }
        assert_eq!(59122, visited.len()); // Value isn't crucial
    }

    #[test]
    fn advance_rollout_test() {
        let mut unmove_queue = Vec::with_capacity(81);

        let mut node_store= HashMap::with_capacity_and_hasher(2_000_000, IdentityHasherBuilder{});
//        let mut node_store= HashMap::with_capacity(2_000_000, );
        let legal_moves = UTTTBoard::get_move_ordering_array();
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

        eprintln!("completed {} rollouts with backup", rollout_count);
        assert_eq!(true, rollout_count>12000, "expected more than 12000 rollouts from starting position");
//            assert_eq!(rollout_count, 50000);
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
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
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
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
        eprintln!("{}\n", board_string);
        assert_eq!(board_string, ".........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
        assert_eq!(x.x_to_move, true);
        assert_eq!(0, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(0, x.x_supers);

        //testing an x move
        let ttt_move = GameMove { move_index: 0, x_to_move: true };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
//            eprintln!("{}\n", board_string);
        assert_eq!(board_string, "X........\n.........\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
        assert_eq!(x.x_to_move, false);
        assert_eq!(0, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(0, x.x_supers);

        //testing an x move
        let ttt_move = GameMove { move_index: 4, x_to_move: false };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
//            eprintln!("{}\n", board_string);
        assert_eq!(board_string, "X........\n.O.......\n.........\n...O.....\n....X....\n.........\n.........\n.........\n.........\n");
        assert_eq!(x.x_to_move, true);
        assert_eq!(4, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(0, x.x_supers);

        //testing an x move
        let ttt_move = GameMove { move_index: 37, x_to_move: true };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
//            eprintln!("{}\n", board_string);
        assert_eq!(board_string, "X........\n.O.......\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........\n");
        assert_eq!(x.x_to_move, false);
        assert_eq!(1, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(0, x.x_supers);

        //testing an x move
        let ttt_move = GameMove { move_index: 13, x_to_move: false };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
//            eprintln!("{}\n", board_string);
        assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n.........\n.........\n.........\n.........\n");
        assert_eq!(x.x_to_move, true);
        assert_eq!(4, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(0, x.x_supers);

        let ttt_move = GameMove { move_index: 43, x_to_move: true };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
//            eprintln!("{}\n", board_string);
        assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.........\n");
        assert_eq!(x.x_to_move, false);
        assert_eq!(7, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(1 << 4, x.x_supers);

        let ttt_move = GameMove { move_index: 71, x_to_move: false };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
//            eprintln!("{}\n", board_string);
        assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O...\n");
        assert_eq!(x.x_to_move, true);
        assert_eq!(8, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(1 << 4, x.x_supers);

        let ttt_move = GameMove { move_index: 79, x_to_move: true };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
        assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.....O.X.\n");
        assert_eq!(x.x_to_move, false);
        assert_eq!(7, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(1 << 4, x.x_supers);

        let ttt_move = GameMove { move_index: 69, x_to_move: false };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
        assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n...O.O.X.\n");
        assert_eq!(x.x_to_move, true);
        assert_eq!(6, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(1 << 4, x.x_supers);

        let ttt_move = GameMove { move_index: 61, x_to_move: true };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
        let board_string = x.display();
        assert_eq!(board_string, "X........\n.O..O....\n.........\n...OX....\n....X....\n....X....\n.........\n.........\n.X.O.O.X.\n");
        assert_eq!(x.x_to_move, false);
        assert_eq!(7, x.last_super_index);
        assert_eq!(0, x.o_supers);
        assert_eq!(1 << 4, x.x_supers);

        let ttt_move = GameMove { move_index: 70, x_to_move: false };
        let _unmake_move = x.make_move(ttt_move, &zobrist_constants);
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
//            let mut sub_col = 0u8;
//            let mut super_col = 0u8;
//            let mut sub_row = 0u8;
//            let mut super_row = 0u8;

        let sub_board = 5;
        let sub_square = 6;

        let expected_tuple = SubBoardAndSquare { sub_board_index: sub_board, sub_square_index: sub_square };
        let actual_tuple = parse_index(51);

        assert_eq!(expected_tuple, actual_tuple);
    }

    #[test]
    fn playout_test() {

//            let mut move_count = 0;
        let base_board = UTTTBoard {
            xbb: [0u16 ;9],
            obb: [0u16 ;9],
            last_super_index: 9,
            x_supers: 0,
            o_supers: 0,
            x_to_move: true,
            hash_state: 0,
        };

        let legal_moves = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();


        let expt_start = time::Instant::now();

        for _ in 0..1200u64 {
            let mut uttt_board = base_board.clone();
            while uttt_board.get_win_state() == GameResult::Unresolved {
                let moves = uttt_board.get_legal_moves(&legal_moves);
                let first_move = moves.get(0);
                match first_move {
                    Some(a_move) => {
                        uttt_board.make_move(*a_move, &zobrist_constants);

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

        let legal_moves = UTTTBoard::get_move_ordering_array();
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
        let mut ttable = HashMap::new();

        let legal_moves = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();

        while uttt_board.get_win_state() == GameResult::Unresolved {
            let (_board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
                &mut uttt_board, 1, std::i8::MIN, std::i8::MAX,
                &mut ttable, &legal_moves, &zobrist_constants
            );

            let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
            uttt_board.make_move(ab_move, &zobrist_constants);
        }
    }

    #[test]
    fn alpha_beta_playout_fast_d2_test() {

//            let mut move_count = 0;
        let mut ttable = HashMap::new();

        let legal_moves = UTTTBoard::get_move_ordering_array();

        let now = time::Instant::now();
        for _ in 0..5 {
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
                let (_board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
                    &mut uttt_board, 2, std::i8::MIN, std::i8::MAX,
                    &mut ttable, &legal_moves, &zobrist_constants
                );

                let ab_move = GameMove { x_to_move: uttt_board.x_to_move(), move_index };
                if move_index < 100 {
                    uttt_board.make_move(ab_move, &zobrist_constants);
//                        move_count += 1;
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

//            let mut move_count = 0;
        let mut ttable = HashMap::new();

        let legal_moves = UTTTBoard::get_move_ordering_array();

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
        for _ in 0..12000 {
            let mut uttt_board = master_board.clone();
            while uttt_board.get_win_state() == GameResult::Unresolved {
                let (_board_val, _board_upper, move_index, _found_terminal) = alpha_beta_search(
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
    fn array_legal_lookup() {
        use uttt::ttt_trait_solving::TTTBoard;
//            let mut uttt_board = UTTTBoard::default();
        let legal_moves = UTTTBoard::get_move_ordering_array();


        let test_pos_x_1 = 256+16;
        let test_pos_o_1 = 4+2;
        let x_to_move = true;

        let mut init_board = TTTBoard {
            xbb: test_pos_x_1,
            obb: test_pos_o_1,
            x_to_move,
        };
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 0);
//            eprintln!("For board:\n{}\n{:?}\n",init_board.display(),lookup_result);
        init_board.x_to_move=false;
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 0);
//            eprintln!("For board:\n{}\n{:?}\n",init_board.display(),lookup_result);

//            init_board.get_legal_moves()
        let test_pos_x_2 = 256+128;
        let test_pos_o_2 = 4+2;
        let x_to_move = true;

        let mut init_board = TTTBoard {
            xbb: test_pos_x_2,
            obb: test_pos_o_2,
            x_to_move,
        };
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 6);
        init_board.x_to_move=false;
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 0);

//            init_board.get_legal_moves()
        let test_pos_x_3 = 1+16;
        let test_pos_o_3 = 64+128;
        let x_to_move = true;

        let mut init_board = TTTBoard {
            xbb: test_pos_x_3,
            obb: test_pos_o_3,
            x_to_move,
        };
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 8);
        init_board.x_to_move=false;
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 8);

//            init_board.get_legal_moves()
        let test_pos_x_3 = 1+2;
        let test_pos_o_3 = 64+128;
        let x_to_move = true;

        let mut init_board = TTTBoard {
            xbb: test_pos_x_3,
            obb: test_pos_o_3,
            x_to_move,
        };
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 2);
        init_board.x_to_move=false;
        let lookup_result = legal_moves.get(init_board.to_index() as usize);
        assert_eq!(lookup_result.unwrap().get(0).unwrap().move_index, 8);

        assert_eq!(524288, legal_moves.len());
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

        let legal_moves = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();

        let mut ttable = HashMap::new();



        let search_start = time::Instant::now();
        let (board_val, board_upper, _move_index, _found_terminal) = alpha_beta_search(
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

        let legal_moves = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();

        let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
        let search_start = time::Instant::now();

        alpha_beta_search(
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

        let legal_moves = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();

        let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
        let search_start = time::Instant::now();
        alpha_beta_search(
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

        let default_ordering = UTTTBoard::get_move_ordering_array();
        let zobrist_constants = UTTTBoard::get_zobrist_constants();


        let mut ttable = HashMap::new();
//            let (board_val, move_index) = alpha_beta_search(
//                &mut uttt_board, 3, std::i8::MIN, std::i8::MAX, &mut ttable,
//            );
        let search_start = time::Instant::now();
        alpha_beta_search(
            &mut uttt_board, 5, std::i8::MIN, std::i8::MAX,
            &mut ttable, &default_ordering, &zobrist_constants
        );

        eprintln!("ttable size: {}", ttable.len());
        eprintln!("elapsed search time: {}", search_start.elapsed().as_micros());
        assert_eq!(true, uttt_board.is_valid(), "expected a valid bitboard");
    }
}