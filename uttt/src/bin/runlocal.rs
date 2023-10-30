extern crate uttt;

use rand::distributions::{Distribution, Uniform};
use rand::{rngs, Rng, SeedableRng};
use std::collections::HashMap;
use std::time;
use uttt::game_solving::{
    advance_rollout, reverse_rollout, AlphaBetaCompatible, GameResult, IdentityHasherBuilder,
    UmctsNode,
};
use uttt::ttt_trait_solving::TicTacToeEngine;
use uttt::uttt_trait_solving::UltimateTicTacToeBoard;

fn main() {
    // game loop
    // use std::io::stdin;

    eprintln!("entering main");

    let mut turn_count = 0;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut uttt_board = UltimateTicTacToeBoard::default();

    let mut unmove_queue = Vec::with_capacity(81);
    let mut node_store: HashMap<u64, UmctsNode, IdentityHasherBuilder> =
        HashMap::with_capacity_and_hasher(10_000_000, IdentityHasherBuilder {});
    eprintln!("entering main2");
    let engine = TicTacToeEngine::new();
    eprintln!("entering main3");
    let legal_moves = engine.get_move_ordering_array();
    eprintln!("entering main4");

    let rand_generation = time::Instant::now();
    let random_indices = {
        let seeded = rngs::StdRng::seed_from_u64(0);
        let uniform_range = Uniform::from(0..u16::MAX);
        let temporary_buffer: Vec<u16> =
            uniform_range.sample_iter(seeded).take(15_000_000).collect();
        // seeded.fill(&mut temporary_buffer);

        temporary_buffer
    };
    println!(
        "generated values in {} microseconds",
        rand_generation.elapsed().as_micros()
    );

    loop {
        // ***************************************************************
        // setting start of turn timer

        // **************************************************************
        let turn_start = time::Instant::now();
        eprintln!("started turn {}", turn_count);
        let time_limit = if (turn_count == 0) || (turn_count == 1) {
            998_000
        } else {
            98_000
        };
        eprintln!("generated legal moves");

        let mut rollout_count = 0;
        let mut move_count = 0;

        while turn_start.elapsed().as_micros() < time_limit {
            let mut search_alpha = i8::MIN;
            let mut search_beta = i8::MAX;
            //            let mut rollout_result = 0;

            while let Some(new_alpha_beta) = advance_rollout(
                &mut unmove_queue,
                &mut uttt_board,
                &legal_moves,
                search_alpha,
                search_beta,
                &mut node_store,
                &zobrist_constants,
            ) {
                let (new_alpha, new_beta) = new_alpha_beta;
                search_alpha = new_alpha;
                search_beta = new_beta;

                // println!("{}",uttt_board.display());
                let mut _read_target = String::new();
                // stdin().read_line(&mut _read_target);

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
                    //TODO: Figure out what on earth this means!
                    search_alpha
                }
            };
            //            eprintln!("processing rollout for board hash: {}",uttt_board.hash_val());
            reverse_rollout(
                rollout_result,
                &mut unmove_queue,
                &mut uttt_board,
                &mut node_store,
            );
            rollout_count += 1;
        }

        let moves = uttt_board.get_legal_moves(&legal_moves);
        let mut mut_board = uttt_board.clone();

        let best_move_option = moves.iter().max_by_key(|a_move| {
            let undo = mut_board.make_move(**a_move, &zobrist_constants);
            let this_hash = mut_board.hash_val();
            mut_board.unmake_move(undo);
            let child_stats = node_store.get(&this_hash);

            match child_stats {
                Some(node) => node.visits,
                None => 0,
            }
        });

        let best_move = match best_move_option {
            Some(game_move) => game_move,
            None => {
                eprintln!("finished playing");
                return;
            }
        };

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

        uttt_board.make_move(*best_move, &zobrist_constants);

        turn_count += 1;
        eprintln!(
            "board after move {:?}:\n{}",
            *best_move,
            uttt_board.display()
        );
        eprintln!("elapsed microseconds: {}", turn_start.elapsed().as_micros());
        eprintln!(
            "rollout stats.\nmoves\trollouts\n{}\t{}",
            move_count, rollout_count
        );
        eprintln!("board representation post-move:\n{:?}", uttt_board);
        eprintln!("table size: {}", node_store.len());
    }
}
