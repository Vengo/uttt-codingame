extern crate uttt;
use uttt::game_solving;
use uttt::ttt_trait_solving;
use uttt::uttt_trait_solving;

use std::time::Instant;

use rand::distributions::{Distribution, Uniform};
use rand::{rngs, Rng, SeedableRng};
use std::collections::HashMap;
use std::time;
use uttt::game_solving::{
    advance_rollout, reverse_rollout, AlphaBetaCompatible, GameResult, IdentityHasherBuilder,
    UmctsNode, GameMove, UmctsQueueElement, uct_value
};
use uttt::uttt_trait_solving::{UltimateTicTacToeBoard, UltimateTicTacToeUnmove, ZobristConstants};
use uttt::ttt_trait_solving::TicTacToeEngine;

fn main() {
    // game loop

    let mut turn_count = 0;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut uttt_board = UltimateTicTacToeBoard::default();

    let mut unmove_queue = Vec::with_capacity(81);
    let mut node_store: HashMap<u64, UmctsNode, IdentityHasherBuilder> =
        HashMap::with_capacity_and_hasher(10_000_000, IdentityHasherBuilder {});
    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();

    let rand_generation = time::Instant::now();
    let random_indices = {
        let seeded = rngs::StdRng::seed_from_u64(0);
        let uniform_range = Uniform::from(0..u16::MAX);
        let temporary_buffer: Vec<u16> =
            uniform_range.sample_iter(seeded).take(15_000_000).collect();
        // seeded.fill(&mut temporary_buffer);

        temporary_buffer
    };
    eprintln!(
        "generated values in {} microseconds",
        rand_generation.elapsed().as_micros()
    );

    while uttt_board.get_win_state() ==GameResult::Unresolved {
        // ***************************************************************
        // setting start of turn timer

        // **************************************************************
        let turn_start = time::Instant::now();
        eprintln!("\n====");
        eprintln!("started turn {}", turn_count);
        let time_limit = if (turn_count == 0) || (turn_count == 1) {
            998_000
        } else {
            98_000
        };

        let best_move = calculate_turn(
            &mut uttt_board,
            &zobrist_constants,
            &mut unmove_queue,
            &mut node_store,
            &legal_moves,
            turn_start,
            time_limit
        );

        let best_move_index = best_move.move_index;

        let best_subboard_index = best_move_index / 9;
        let best_sub_square_index = best_move_index % 9;

        let base_row = best_subboard_index / 3;
        let base_col = best_subboard_index % 3;

        let final_row = base_row * 3 + best_sub_square_index / 3;
        let final_column = base_col * 3 + best_sub_square_index % 3;

        eprintln!("elapsed microseconds: {}", turn_start.elapsed().as_micros());
        eprintln!("board representation post-move:\n{:?}", uttt_board);
        eprintln!("table size: {}", node_store.len());


        let moves = uttt_board.get_legal_moves(&legal_moves);

        eprintln!("==== move stat table ====");
        eprintln!("index\tavg\tvisits\tlower\tupper\tvalue");
        for each_move in moves.iter() {
            //                eprintln!("x to move? {}",each_move.x_to_move);
            let undo_move = uttt_board.make_move(*each_move, &zobrist_constants);
            let this_hash = uttt_board.hash_val();
            uttt_board.unmake_move(undo_move);
            let child_stats = node_store.get(&this_hash);

            match child_stats {
                Some(node) => {
                    eprintln!(
                        "{}\t{:3.2}\t{}\t{}\t{}\t{}",
                        each_move.move_index,
                        (node.value as f32) / (node.visits as f32),
                        node.visits,
                        node.value_lower,
                        node.value_upper,
                        node.value
                    );
                }
                None => {
                    eprintln!("no stored statistics found for hash value: {}", this_hash);
                }
            }
        }

        uttt_board.make_move(best_move, &zobrist_constants);

        turn_count += 1;
        eprintln!(
            "board after move {:?}:\n{}",
            best_move,
            uttt_board.display()
        );
        eprintln!("elapsed microseconds: {}", turn_start.elapsed().as_micros());

        eprintln!("board representation post-move:\n{:?}", uttt_board);
        eprintln!("table size: {}", node_store.len());
    }

    eprintln!("Ended in state: {:?}", uttt_board.get_win_state());
}

fn calculate_turn(
    uttt_board: &mut UltimateTicTacToeBoard,
    zobrist_constants: &ZobristConstants,
    mut unmove_queue: &mut Vec<UmctsQueueElement<u64, UltimateTicTacToeUnmove>>,
    mut node_store: &mut HashMap<u64, UmctsNode, IdentityHasherBuilder>,
    legal_moves: &[Vec<GameMove>],
    turn_start: Instant, time_limit: u128) -> GameMove {
    use uttt::uttt_trait_solving::parse_index_from_super_coords;
    let mut rollout_count = 0;
    let mut move_count = 0;

    let best_move = {
        let x_to_move = uttt_board.x_to_move();
        eprintln!("board representation pre-move:\n{:?}", uttt_board);

        while turn_start.elapsed().as_micros() < time_limit {
            let mut search_alpha = i8::MIN;
            let mut search_beta = i8::MAX;

            while let Some(new_alpha_beta) = advance_rollout(
                &mut unmove_queue,
                uttt_board,
                &legal_moves,
                search_alpha,
                search_beta,
                &mut node_store,
                &zobrist_constants,
            ) {
                let (new_alpha, new_beta) = new_alpha_beta;
                search_alpha = new_alpha;
                search_beta = new_beta;
                move_count += 1;
            }

            let final_hash = uttt_board.hash_val();
            let terminal_node = node_store.get_mut(&final_hash).unwrap();

            let rollout_result = match uttt_board.get_win_state() {
                GameResult::P1Win => {
                    let rollout_result = i8::MAX;
                    terminal_node.value_lower = rollout_result;
                    terminal_node.value_upper = rollout_result;
                    rollout_result
                }
                GameResult::P2Win => {
                    let rollout_result = i8::MIN;
                    terminal_node.value_lower = rollout_result;
                    terminal_node.value_upper = rollout_result;
                    rollout_result
                }
                GameResult::Tie => {
                    let rollout_result = 0;
                    terminal_node.value_lower = rollout_result;
                    terminal_node.value_upper = rollout_result;
                    rollout_result
                }
                _ => {
                    //TODO: P3: Figure out what on earth this means!
                    search_alpha
                }
            };
            reverse_rollout(
                rollout_result,
                &mut unmove_queue,
                uttt_board,
                &mut node_store,
            );
            rollout_count += 1;
        }

        let moves = uttt_board.get_legal_moves(&legal_moves);

        let mut mut_board = uttt_board.clone();

        let best_move = moves
            .iter()
            .max_by_key(|a_move| {
                let undo = mut_board.make_move(**a_move, &zobrist_constants);
                let this_hash = mut_board.hash_val();
                mut_board.unmake_move(undo);
                let child_stats = node_store.get(&this_hash);
                match child_stats {
                    Some(node) => node.visits,
                    None => 0,
                }
            })
            .unwrap();

        let moves = uttt_board.get_legal_moves(&legal_moves);

        // inspect_moves(uttt_board, &zobrist_constants, node_store, rollout_count, x_to_move, moves);
        *best_move

    };
    eprintln!("calculate using rollouts: {}", rollout_count);
    eprintln!(
        "rollout stats.\nmoves\trollouts\n{}\t{}",
        move_count, rollout_count
    );
    best_move
}

fn inspect_moves(uttt_board: &mut UltimateTicTacToeBoard, zobrist_constants: &&ZobristConstants,
                 node_store: &mut HashMap<u64, UmctsNode, IdentityHasherBuilder>, rollout_count: u32, x_to_move: bool, moves: Vec<GameMove>) {
    eprintln!("debugging moves available");
    for each_move in moves.iter() {
        eprintln!("========================");
        //                eprintln!("x to move? {}",each_move.x_to_move);
        let undo_move = uttt_board.make_move(*each_move, &zobrist_constants);
        let this_hash = uttt_board.hash_val();
        uttt_board.unmake_move(undo_move);
        let child_stats = node_store.get(&this_hash);

        match child_stats {
            Some(node) => {
                eprintln!(
                    "node avg value: {}",
                    (node.value as f32) / (node.visits as f32)
                );
                eprintln!("node visits: {}", node.visits);
                eprintln!("move index: {}", each_move.move_index);
                eprintln!(
                    "uct value: {:?}",
                    uct_value(node.value, rollout_count, node.visits, x_to_move)
                );
                //                            value node visits action visits side to move
            }
            None => {
                eprintln!("no stored statistics found for hash value: {}", this_hash);
            }
        }
        eprintln!();
    }
}