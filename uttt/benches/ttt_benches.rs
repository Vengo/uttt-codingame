#[macro_use]
extern crate criterion;
extern crate uttt;

use criterion::black_box;
use criterion::Criterion;

//use utt::game_solving;

use uttt::game_solving::AlphaBetaCompatible;
use uttt::ttt_trait_solving::{TicTacToeBoard, TicTacToeState};

fn ttt_display_test(c: &mut Criterion) {
    let x = TicTacToeBoard {
        xbb: 1 << 8,
        obb: 0 << 8,
    };

    c.bench_function("basic ttt display", move |b| {
        b.iter(|| black_box(x.clone()).display())
    });
}

fn ttt_movegen_empty(c: &mut Criterion) {
    let x = TicTacToeBoard {
        xbb: 0 << 8,
        obb: 0 << 8,
    };

    c.bench_function("ttt movegen empty", move |b| {
        b.iter(|| black_box(x.clone()).get_legal_moves_no_lookup(false))
    });
}

fn ttt_movegen_full(c: &mut Criterion) {
    let x = TicTacToeBoard {
        xbb: 73 << 1,
        obb: 40,
    };

    c.bench_function("ttt movegen full board", move |b| {
        b.iter(|| black_box(x.clone()).get_legal_moves_no_lookup(false))
    });
}

criterion_group!(
    ttt_benches,
    ttt_display_test,
    ttt_movegen_empty,
    ttt_movegen_full
);
// criterion_group!(ttt_benches, ttt_display_test);
criterion_main!(ttt_benches);
