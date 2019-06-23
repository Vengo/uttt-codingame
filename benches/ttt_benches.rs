#[macro_use]
extern crate criterion;

use criterion::Criterion;
use criterion::black_box;

//use utt::game_solving;

use uttt::game_solving::AlphaBetaTTCompatible;
use uttt::ttt_trait_solving::TTTBoard;

fn ttt_display_test(c: &mut Criterion) {
    let x = TTTBoard {
        xbb: 1 << 8,
        obb: 0 << 8,
        x_to_move: false,
    };

    c.bench_function(
        "basic ttt display",
        move |b| b.iter( || black_box(x.clone()).display()));
}

fn ttt_movegen_empty(c: &mut Criterion) {
    let x = TTTBoard {
        xbb: 0 << 8,
        obb: 0 << 8,
        x_to_move: true,
    };

    c.bench_function(
        "ttt movegen empty",
        move |b| b.iter( || black_box(x.clone()).get_legal_moves()));
}

fn ttt_movegen_full(c: &mut Criterion) {
    let x = TTTBoard {
        xbb: 73 << 1,
        obb: 40,
        x_to_move: true,
    };

    c.bench_function(
        "ttt movegen full board",
        move |b| b.iter( || black_box(x.clone()).get_legal_moves()));
}

//criterion_group!(ttt_benches, ttt_display_test, ttt_movegen_empty, ttt_movegen_full);
criterion_group!(ttt_benches, ttt_display_test);
criterion_main!(ttt_benches);