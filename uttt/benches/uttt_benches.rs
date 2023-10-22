#[macro_use]
extern crate criterion;
extern crate uttt;

use criterion::black_box;
use criterion::Criterion;

use std::collections::HashMap;
use uttt::game_solving::{alpha_beta_search, AlphaBetaCompatible, GameMove, GameResult};
use uttt::uttt_trait_solving::{UltimateTicTacToeBoard, HASH_DEFAULT};

fn uttt_display_test(c: &mut Criterion) {
    let x = UltimateTicTacToeBoard {
        xbb: [0; 9],
        obb: [0; 9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };

    c.bench_function("basic uttt display", move |b| {
        b.iter(|| black_box(x.clone()).display())
    });
}

fn uttt_movegen_empty(c: &mut Criterion) {
    let x = UltimateTicTacToeBoard {
        xbb: [0; 9],
        obb: [0; 9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    c.bench_function("uttt movegen empty", move |b| {
        b.iter(|| black_box(x.clone()).get_legal_moves(&legal_moves))
    });
}

fn uttt_movegen_full(c: &mut Criterion) {
    let x = UltimateTicTacToeBoard {
        xbb: [0, 0, 4, 0, 0, 0, 0, 0, 0],
        obb: [0; 9],
        last_super_index: 2,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    c.bench_function("uttt movegen full board", move |b| {
        b.iter(|| black_box(x.clone()).get_legal_moves(&legal_moves))
    });
}

fn uttt_make_move(c: &mut Criterion) {
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 2, 0, 0, 0, 0, 0, 0],
        obb: [0; 9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };

    let uttt_move = GameMove {
        x_to_move: false,
        move_index: 40,
    };

    c.bench_function("uttt make move", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            let a_move = black_box(uttt_move.clone());
            let _unmove = a_board.make_move(a_move, &zobrist_constants);
        })
    });
}

fn uttt_unmake_move(c: &mut Criterion) {
    let mut uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 2, 0, 0, 0, 0, 0, 0],
        obb: [0; 9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let uttt_move = GameMove {
        x_to_move: false,
        move_index: 40,
    };

    let unmake_move = uttt_board.make_move(uttt_move, &zobrist_constants);

    c.bench_function("uttt unmake move", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            let a_move = black_box(unmake_move.clone());
            a_board.unmake_move(a_move);
        })
    });
}

fn uttt_result_check(c: &mut Criterion) {
    let mut uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 2, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };

    c.bench_function("uttt winstate check: unresolved", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            assert_eq!(GameResult::Unresolved, a_board.get_win_state());
        })
    });
}

fn uttt_playout(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    c.bench_function("uttt playout", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            while a_board.get_win_state() == GameResult::Unresolved {
                let moves = a_board.get_legal_moves(&legal_moves);

                let first_move: Option<&GameMove> = moves.get(0);
                match first_move {
                    Some(a_move) => {
                        a_board.make_move(*a_move, &zobrist_constants);
                    }
                    None => break,
                }
            }
        })
    });
}

fn uttt_playout_turn_2(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 16, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    c.bench_function("uttt playout from turn 2", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            while a_board.get_win_state() == GameResult::Unresolved {
                let moves = a_board.get_legal_moves(&legal_moves);

                let first_move = moves.get(0);
                match first_move {
                    Some(a_move) => {
                        a_board.make_move(*a_move, &zobrist_constants);
                    }
                    None => break,
                }
            }
        })
    });
}

fn uttt_playout_alpha_beta(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 4, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    const COMPETITIVE_AB_DEPTH: u8 = 2;

    c.bench_function("uttt playout alpha beta", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            let mut transposition_table = HashMap::new();
            while a_board.get_win_state() == GameResult::Unresolved {
                let (best_value, val_upper, best_action, found_terminal) = alpha_beta_search(
                    &mut a_board,
                    COMPETITIVE_AB_DEPTH,
                    std::i8::MIN,
                    std::i8::MAX,
                    &mut transposition_table,
                    &legal_moves,
                    &zobrist_constants,
                );

                let my_move = GameMove {
                    move_index: best_action,
                    x_to_move: a_board.x_to_move(),
                };
                a_board.make_move(my_move, &zobrist_constants);
            }
        })
    });
}

fn uttt_alpha_beta_zero_window(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 4, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();
    let mut transposition_table = HashMap::new();

    const COMPETITIVE_AB_DEPTH: u8 = 12;

    c.bench_function("uttt alpha beta zero window", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());

            let (best_value, val_upper, best_action, found_terminal) = alpha_beta_search(
                &mut a_board,
                COMPETITIVE_AB_DEPTH,
                std::i8::MIN,
                std::i8::MIN + 1,
                &mut transposition_table,
                &legal_moves,
                &zobrist_constants,
            );
        })
    });
}

fn uttt_alpha_beta_big_window(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 4, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    const COMPETITIVE_AB_DEPTH: u8 = 11;

    c.bench_function("uttt alpha beta big window", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            let mut transposition_table = HashMap::new();

            let (best_value, val_upper, best_action, found_terminal) = alpha_beta_search(
                &mut a_board,
                COMPETITIVE_AB_DEPTH,
                i8::MIN,
                i8::MAX,
                &mut transposition_table,
                &legal_moves,
                &zobrist_constants,
            );
        })
    });
}

// TODO: I think I want to have a UMCTS function again?
// fn mcts_bench(c: &mut Criterion) {
//     let uttt_board = UltimateTicTacToeBoard {
//         xbb: [0u16, 0u16, 0u16, 4u16, 0u16, 0u16, 0u16, 0u16, 0u16],
//         obb: [0u16; 9],
//         last_super_index: 4,
//         x_supers: 0,
//         o_supers: 0,
//         x_to_move: false,
//         hash_state: HASH_DEFAULT,
//     };
//     let board_string = uttt_board.display();
//     let mut move_count = 0;
//
//     let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();
//
//     c.bench_function("mcts search benchmarking", move |b| {
//         b.iter(|| {
//             let mut a_board = black_box(uttt_board.clone());
//             let mut mcts_table = HashMap::new();
//
//             umcts(
//                 &mut a_board,
//                 &legal_moves,
//                 std::i8::MIN,
//                 std::i8::MAX,
//                 &mut mcts_table,
//             );
//         })
//     });
// }

fn uttt_playout_alpha_beta_depths(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
        hash_state: HASH_DEFAULT,
    };
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    const COMPETITIVE_AB_DEPTH: u8 = 4;

    let mut transposition_table = HashMap::new();

    c.bench_function_over_inputs(
        "uttt playout alpha beta at depths",
        move |b, &&input_depth| {
            b.iter(|| {
                let mut a_board = black_box(uttt_board.clone());
                while a_board.get_win_state() == GameResult::Unresolved {
                    let (best_value, val_upper, best_action, found_terminal) = alpha_beta_search(
                        &mut a_board,
                        input_depth,
                        std::i8::MIN,
                        std::i8::MAX,
                        &mut transposition_table,
                        &legal_moves,
                        &zobrist_constants,
                    );

                    let my_move = GameMove {
                        move_index: best_action,
                        x_to_move: uttt_board.x_to_move,
                    };
                    a_board.make_move(my_move, &zobrist_constants);
                }
            })
        },
        &[8],
    );
}

fn uttt_get_tied_mask(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
        hash_state: HASH_DEFAULT,
    };

    c.bench_function("uttt getting tied board bitmask", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            a_board.get_tied_mask();
        })
    });
}

fn uttt_get_subboard_moves(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
        hash_state: HASH_DEFAULT,
    };

    let legal_moves = UltimateTicTacToeBoard::get_move_ordering_array();

    c.bench_function("uttt get subboard moves", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            a_board.get_subboard_moves(1, &legal_moves, a_board.x_to_move);
        })
    });
}

fn uttt_hash_test(c: &mut Criterion) {
    let uttt_board = UltimateTicTacToeBoard {
        xbb: [0, 0, 0, 0, 4, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
        hash_state: HASH_DEFAULT,
    };

    //    let legal_moves = UTTTBoard::get_move_orderings();

    c.bench_function("uttt hash test", move |b| {
        b.iter(|| {
            let mut a_board = black_box(uttt_board.clone());
            a_board.hash_val();
            //            a_board.get_subboard_moves(1, &legal_moves,&a_board.x_to_move);
        })
    });
}

// criterion_group!(
//     uttt_benches,
//     uttt_display_test,
//     uttt_movegen_empty,
//     uttt_movegen_full,
//     uttt_make_move,
//     uttt_unmake_move,
//     uttt_result_check,
//     uttt_playout,
//     uttt_playout_turn_2,
//     uttt_playout_alpha_beta,
//     uttt_alpha_beta_zero_window,
//     uttt_alpha_beta_big_window,
//     uttt_playout_alpha_beta_depths,
//     uttt_get_tied_mask,
//     uttt_get_subboard_moves,
//     uttt_hash_test
// );
criterion_group!(uttt_benches, uttt_alpha_beta_big_window,);

criterion_main!(uttt_benches);
