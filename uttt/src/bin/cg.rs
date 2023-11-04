// game loop for codingame.com competition
extern crate uttt;
use uttt::game_solving;
use uttt::ttt_trait_solving;
use uttt::uttt_trait_solving;

use std::collections::HashMap;
use std::time;
use std::time::Instant;
use uttt::game_solving::{GameMove, IdentityHasherBuilder, UmctsNode, UmctsQueueElement};
use uttt::uttt_trait_solving::{UltimateTicTacToeBoard, UltimateTicTacToeUnmove, ZobristConstants};

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

fn parse_codingame_inputs() -> (i32, i32) {
    let mut input_line = String::new();
    std::io::stdin().read_line(&mut input_line).unwrap();

    let inputs = input_line.split(' ').collect::<Vec<_>>();
    let opponent_row = parse_input!(inputs[0], i32);
    let opponent_col = parse_input!(inputs[1], i32);
    let mut input_line = String::new();
    std::io::stdin().read_line(&mut input_line).unwrap();
    let valid_action_count = parse_input!(input_line, i32);
    for _i in 0..valid_action_count as usize {
        let mut input_line = String::new();
        std::io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(' ').collect::<Vec<_>>();
        let _row = parse_input!(inputs[0], i32);
        let _col = parse_input!(inputs[1], i32);
    }
    (opponent_row, opponent_col)
}

pub fn main() {
    use game_solving::{
        IdentityHasherBuilder,
    };
    use rand::distributions::{Distribution, Uniform};
    use rand::{rngs, SeedableRng};
    use ttt_trait_solving::TicTacToeEngine;
    use uttt_trait_solving::UltimateTicTacToeBoard;

    let mut uttt_board = UltimateTicTacToeBoard {
        xbb: [0u16; 9],
        obb: [0u16; 9],
        last_super_index: 9,
        o_supers: 0,
        x_supers: 0,
        x_to_move: true,
        hash_state: 0,
    };

    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let mut turn_count = 0;

    let mut unmove_queue = Vec::with_capacity(81);
    let mut node_store = HashMap::with_capacity_and_hasher(2_000_000, IdentityHasherBuilder {});
    let mut legal_moves = Vec::new();
    let mut engine;

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

    loop {
        // **************************************************************
        // begin supplied boiler plate
        // reads game input, and parses the result
        let (opponent_row, opponent_col) = parse_codingame_inputs();

        // end supplied boiler plate
        // ***************************************************************
        // overview of the first turn
        // 1. Start a timer
        // 2. Set up (including reading input)
        // 3. Evaluate the position until time runs out
        // 4. Return chosen move to the referee over standard out
        // setting start of turn timer

        // overview of later turns
        // 1. Start a timer
        // 3. Evaluate the position until time runs out
        // 4. Return chosen move to the referee over standard out
        // setting start of turn timer

        // **************************************************************
        let turn_start = time::Instant::now();
        let time_limit = if turn_count == 0 {
            engine = TicTacToeEngine::new();
            legal_moves = engine.get_move_ordering_array();
            990_000
        } else {
            90_000
        };

        let (final_row, final_column) = calculate_turn(
            &mut uttt_board,
            &zobrist_constants,
            &mut unmove_queue,
            &mut node_store,
            &mut legal_moves,
            opponent_row,
            opponent_col,
            turn_start,
            time_limit
        );
        turn_count += 1;
        println!("{} {}", final_row, final_column);
    }
}

fn calculate_turn(
    uttt_board: &mut UltimateTicTacToeBoard,
    zobrist_constants: &ZobristConstants,
    mut unmove_queue: &mut Vec<UmctsQueueElement<u64, UltimateTicTacToeUnmove>>,
    mut node_store: &mut HashMap<u64, UmctsNode, IdentityHasherBuilder>,
    legal_moves: &mut Vec<Vec<GameMove>>,
    opponent_row: i32, opponent_col: i32,
    turn_start: Instant, time_limit: u128) -> (u8, u8) {
    use game_solving::{
        advance_rollout, reverse_rollout, uct_value, AlphaBetaCompatible, GameResult
    };
    use uttt::uttt_trait_solving::parse_index_from_super_coords;
    let mut rollout_count = 0;

    let best_move = match (opponent_row, opponent_col) {
        (-1, -1) => {
            let default_first = GameMove {
                move_index: 40,
                x_to_move: uttt_board.x_to_move(),
            };
            uttt_board.make_move(default_first, &zobrist_constants);

            while turn_start.elapsed().as_micros() < time_limit {
                let mut search_alpha = i8::MIN;
                let mut search_beta = i8::MAX;
                //                    let mut rollout_result = 0;

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
                    }
                    None => {
                        eprintln!("no stored statistics found for hash value: {}", this_hash);
                    }
                }
            }

            default_first
        }
        (row, col) => {
            let opp_move_index = parse_index_from_super_coords(row as u8, col as u8);
            let opp_move = GameMove {
                move_index: opp_move_index,
                x_to_move: uttt_board.x_to_move(),
            };
            uttt_board.make_move(opp_move, &zobrist_constants);
            let x_to_move = uttt_board.x_to_move();
            eprintln!("board representation pre-move:\n{:?}", uttt_board);

            //                let mut rollout_result = 0;
            //                let board_hash = uttt_board.hash_val();

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
            *best_move
        }
    };

    let best_move_index = best_move.move_index;

    let best_subboard_index = best_move_index / 9;
    let best_sub_square_index = best_move_index % 9;

    let base_row = best_subboard_index / 3;
    let base_col = best_subboard_index % 3;

    let final_row = base_row * 3 + best_sub_square_index / 3;
    let final_column = base_col * 3 + best_sub_square_index % 3;

    eprintln!("elapsed microseconds: {}", turn_start.elapsed().as_micros());
    eprintln!("completed rollouts: {}", rollout_count);
    eprintln!("board representation post-move:\n{:?}", uttt_board);
    eprintln!("table size: {}", node_store.len());
    (final_row, final_column)
}
