#[macro_use]
extern crate criterion;

use criterion::Criterion;
use criterion::black_box;

//use utt::game_solving;

use uttt::game_solving::{AlphaBetaTTCompatible, GameMove, GameResult, alpha_beta_search};
use uttt::uttt_trait_solving::{UTTTBoard};
use std::collections::HashMap;

fn uttt_display_test(c: &mut Criterion) {
    let x = UTTTBoard {
        xbb: [0 ;9],
        obb: [0 ;9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    c.bench_function(
        "basic uttt display",
        move |b| b.iter( || black_box(x.clone()).display()));
}

fn uttt_movegen_empty(c: &mut Criterion) {
    let x = UTTTBoard {
        xbb: [0 ;9],
        obb: [0 ;9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    c.bench_function(
        "uttt movegen empty",
        move |b| b.iter( || black_box(x.clone()).get_legal_moves()));
}

fn uttt_movegen_full(c: &mut Criterion) {
    let x = UTTTBoard {
        xbb: [0, 0, 4, 0, 0, 0, 0, 0, 0],
        obb: [0 ;9],
        last_super_index: 2,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    c.bench_function(
        "uttt movegen full board",
        move |b| b.iter( || black_box(x.clone()).get_legal_moves()));
}

fn uttt_make_move(c: &mut Criterion) {
    let uttt_board = UTTTBoard {
        xbb: [0, 0, 2, 0, 0, 0, 0, 0, 0],
        obb: [0 ;9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    let uttt_move = GameMove{x_to_move: false, move_index: 40};

    c.bench_function(
        "uttt make move",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            let a_move = black_box(uttt_move.clone());
            let _unmove = a_board.make_move(&a_move);
        }));
}

fn uttt_unmake_move(c: &mut Criterion) {
    let mut uttt_board = UTTTBoard {
        xbb: [0, 0, 2, 0, 0, 0, 0, 0, 0],
        obb: [0 ;9],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    let uttt_move = GameMove{x_to_move: false, move_index: 40};

    let unmake_move = uttt_board.make_move(&uttt_move);


    c.bench_function(
        "uttt unmake move",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            let a_move = black_box(unmake_move.clone());
            a_board.unmake_move(a_move);
        }));
}

fn uttt_result_check(c: &mut Criterion) {
    let mut uttt_board = UTTTBoard {
        xbb: [0, 0, 2, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    c.bench_function(
        "uttt winstate check: unresolved",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            assert_eq!(GameResult::Unresolved, a_board.get_win_state());
        }));
}

fn uttt_playout(c: &mut Criterion) {
    let uttt_board = UTTTBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
    };

    c.bench_function(
        "uttt playout",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            while a_board.get_win_state() == GameResult::Unresolved {
                let moves = a_board.get_legal_moves();

                let first_move = moves.get(0);
                match first_move {
                    Some(a_move) => {
                        a_board.make_move(a_move);
                    },
                    None => break,
                }
            }
        }));
}

fn uttt_playout_turn_2(c: &mut Criterion) {
    let uttt_board = UTTTBoard {
        xbb: [0, 0, 0, 0, 16, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    c.bench_function(
        "uttt playout from turn 2",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            while a_board.get_win_state() == GameResult::Unresolved {
                let moves = a_board.get_legal_moves();

                let first_move = moves.get(0);
                match first_move {
                    Some(a_move) => {
                        a_board.make_move(a_move);
                    },
                    None => break,
                }
            }
        }));
}

fn uttt_playout_alpha_beta(c: &mut Criterion) {
    let uttt_board = UTTTBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
    };

    const COMPETITIVE_AB_DEPTH: u8 = 2;

    let mut transposition_table = HashMap::new();

    c.bench_function(
        "uttt playout alpha beta",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            while a_board.get_win_state() == GameResult::Unresolved {

                let (best_value, best_action) = alpha_beta_search(
                    &mut a_board, COMPETITIVE_AB_DEPTH, std::i8::MIN, std::i8::MAX, &mut transposition_table);

                let my_move = GameMove { move_index: best_action, x_to_move: uttt_board.x_to_move };
                a_board.make_move(&my_move);
            }
        }));
}

fn uttt_get_tied_mask(c: &mut Criterion) {
    let uttt_board = UTTTBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
    };

    c.bench_function(
        "uttt getting tied board bitmask",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            a_board.get_tied_mask();
        }));
}

fn uttt_get_subboard_moves(c: &mut Criterion) {
    let uttt_board = UTTTBoard {
        xbb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        obb: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        last_super_index: 9,
        x_supers: 0,
        o_supers: 0,
        x_to_move: true,
    };

    c.bench_function(
        "uttt get subboard moves",
        move |b| b.iter( || {
            let mut a_board = black_box(uttt_board.clone());
            a_board.get_subboard_moves(1);
        }));
}

//criterion_group!(
//    uttt_benches, uttt_playout_alpha_beta, uttt_playout, uttt_display_test, uttt_movegen_empty,
//    uttt_movegen_full, uttt_make_move, uttt_unmake_move, uttt_result_check, );
//criterion_group!(uttt_benches, uttt_playout_turn_2, uttt_get_subboard_moves, uttt_get_tied_mask,
//uttt_playout, uttt_movegen_empty, uttt_movegen_full, uttt_make_move, uttt_unmake_move, uttt_result_check);
criterion_group!(uttt_benches, uttt_playout_turn_2, uttt_get_subboard_moves);

criterion_main!(uttt_benches);