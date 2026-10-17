//! Unit tests for pure Rust INT8 inference and AlphaZeroBot

use std::time::{Duration, Instant};
use uttt::bots::create_bot;
use uttt::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use uttt::nn::{extract_features, InferenceWorkspace, ModelInt8};
use uttt::uttt_trait_solving::UltimateTicTacToeBoard;

#[test]
fn test_feature_extraction_initial_board() {
    let board = UltimateTicTacToeBoard::default();
    let (tensor, legal_mask) = extract_features(&board);

    // Initial board: all cells are empty
    for r in 0..9 {
        for c in 0..9 {
            assert_eq!(tensor[r][c][0], 0, "No active player marks initially");
            assert_eq!(tensor[r][c][1], 0, "No opponent marks initially");
            assert_eq!(tensor[r][c][2], 0, "No active won subboards initially");
            assert_eq!(tensor[r][c][3], 0, "No opponent won subboards initially");
            assert_eq!(tensor[r][c][4], 1, "All subboards are legal target initially");
        }
    }

    // All 81 moves are legal initially
    assert_eq!(legal_mask.iter().filter(|&&m| m).count(), 81);
}

#[test]
fn test_feature_extraction_constrained_subboard() {
    let mut board = UltimateTicTacToeBoard::default();
    let z_consts = UltimateTicTacToeBoard::get_zobrist_constants();

    // Play move 40: subboard 4, square 4
    let m = GameMove {
        x_to_move: true,
        move_index: 40,
    };
    board.make_move(m, &z_consts);

    let (tensor, legal_mask) = extract_features(&board);

    // Now opponent (O) is active, directed to subboard 4
    // Subboard 4 corresponds to rows 3..6, cols 3..6
    for r in 0..9 {
        for c in 0..9 {
            let sb = 3 * (r / 3) + (c / 3);
            if sb == 4 {
                assert_eq!(tensor[r][c][4], 1, "Subboard 4 must be legal target");
            } else {
                assert_eq!(tensor[r][c][4], 0, "Subboards != 4 must not be legal target");
            }
        }
    }

    // In subboard 4, cell 4 (r=4, c=4) has X's mark (opponent to O)
    assert_eq!(tensor[4][4][1], 1, "Opponent (X) mark should be at center cell");

    // Exactly 8 legal moves remaining in subboard 4
    let legal_count = legal_mask.iter().filter(|&&m| m).count();
    assert_eq!(legal_count, 8, "Expected exactly 8 legal moves in subboard 4");
    assert!(!legal_mask[40], "Move 40 is already occupied and illegal");
}

#[test]
fn test_model_int8_loading() {
    let model = ModelInt8::default_model();
    assert_eq!(model.num_channels, 64);
    assert_eq!(model.num_res_blocks, 4);
    assert_eq!(model.conv0.in_channels, 5);
    assert_eq!(model.conv0.out_channels, 64);
    assert_eq!(model.res_blocks.len(), 4);
    assert_eq!(model.fc_p.out_features, 81);
    assert_eq!(model.fc_v2.out_features, 1);
}

#[test]
fn test_model_int8_forward_inference() {
    let model = ModelInt8::default_model();
    let mut ws = InferenceWorkspace::new();
    let board = UltimateTicTacToeBoard::default();

    let (logits, value) = model.evaluate(&board, &mut ws);

    // Initial position: all 81 moves are legal
    for &logit in &logits {
        assert!(logit > -1e8, "Legal move logits must not be masked");
        assert!(!logit.is_nan(), "Logit must not be NaN");
    }

    // Value must be bounded in [-1.0, 1.0]
    assert!(value >= -1.0 && value <= 1.0, "Value {} out of bounds", value);

    // Constrained board test
    let mut constrained_board = board.clone();
    let z_consts = UltimateTicTacToeBoard::get_zobrist_constants();
    constrained_board.make_move(
        GameMove {
            x_to_move: true,
            move_index: 40,
        },
        &z_consts,
    );

    let (c_logits, _c_value) = model.evaluate(&constrained_board, &mut ws);
    let mut legal_count = 0;
    let mut masked_count = 0;
    for (m, &logit) in c_logits.iter().enumerate() {
        if logit <= -1e8 {
            masked_count += 1;
        } else {
            legal_count += 1;
            assert!(m >= 36 && m < 45 && m != 40, "Legal move {} outside subboard 4", m);
        }
    }
    assert_eq!(legal_count, 8);
    assert_eq!(masked_count, 73);
}

#[test]
fn test_alphazero_bot_creation_and_move() {
    let mut bot = create_bot("alphazero").expect("AlphaZero bot should be registered");
    assert_eq!(bot.name(), "AlphaZeroBot");

    let board = UltimateTicTacToeBoard::default();
    let z_consts = UltimateTicTacToeBoard::get_zobrist_constants();
    let chosen = bot.choose_move(&board, &[], &z_consts, Duration::from_millis(50));

    assert!(chosen.move_index < 81, "Move index must be valid [0..80]");
    let legal = board.get_legal_moves_fast();
    assert!(
        legal.iter().any(|&m| m.move_index == chosen.move_index),
        "Chosen move {} must be legal",
        chosen.move_index
    );
}

#[test]
fn test_alphazero_bot_latency() {
    let model = ModelInt8::default_model();
    let mut ws = InferenceWorkspace::new();
    let board = UltimateTicTacToeBoard::default();

    // Warm-up
    for _ in 0..5 {
        let _ = model.evaluate(&board, &mut ws);
    }

    let iterations = 20;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.evaluate(&board, &mut ws);
    }
    let total_micros = start.elapsed().as_micros();
    let avg_micros = total_micros / iterations;
    println!(
        "\nPure Rust INT8 forward inference: {} iterations in {} µs (avg: {} µs/eval, {:.1} evals/sec)",
        iterations,
        total_micros,
        avg_micros,
        1_000_000.0 / (avg_micros as f64)
    );

    // Sanity check: each forward pass should complete well within 50 ms even in unoptimized debug mode
    assert!(avg_micros < 50_000, "Inference too slow: {} µs", avg_micros);
}

#[test]
fn test_policy_alphabeta_bot_creation_and_search() {
    let mut bot = create_bot("policy_alphabeta").expect("PolicyAlphaBetaBot must be registered");
    assert_eq!(bot.name(), "PolicyAlphaBetaBot");

    let board = UltimateTicTacToeBoard::default();
    let z_consts = UltimateTicTacToeBoard::get_zobrist_constants();
    let chosen = bot.choose_move(&board, &[], &z_consts, Duration::from_millis(100));

    assert!(chosen.move_index < 81, "Chosen move index must be valid [0..80]");
    let legal = board.get_legal_moves_fast();
    assert!(
        legal.iter().any(|&m| m.move_index == chosen.move_index),
        "Chosen move {} must be legal",
        chosen.move_index
    );
}

#[test]
fn test_debug_game_policy_vs_alphabeta() {
    let mut bot1 = create_bot("policy_alphabeta").unwrap();
    let mut bot2 = create_bot("alphabeta").unwrap();
    let mut board = UltimateTicTacToeBoard::default();
    let z_consts = UltimateTicTacToeBoard::get_zobrist_constants();

    for ply in 0..40 {
        let is_p1 = board.x_to_move;
        let m = if is_p1 {
            bot1.choose_move(&board, &[], &z_consts, Duration::from_millis(50))
        } else {
            bot2.choose_move(&board, &[], &z_consts, Duration::from_millis(50))
        };
        let legal = board.get_legal_moves_fast();
        assert!(legal.iter().any(|&lm| lm.move_index == m.move_index), "Illegal move by {}", if is_p1 { "bot1" } else { "bot2" });
        board.make_move(m, &z_consts);
        let win = board.get_win_state();
        println!("Ply {:2}: {} played move {} (sb {}, sq {}) -> win_state: {:?}", ply, if is_p1 { "PolicyAlphaBeta (X)" } else { "AlphaBeta (O)" }, m.move_index, m.move_index / 9, m.move_index % 9, win);
        if win != GameResult::Unresolved {
            println!("🏆 Game finished at ply {} with result: {:?}", ply, win);
            break;
        }
    }
}


