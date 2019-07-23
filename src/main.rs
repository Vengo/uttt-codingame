use uttt;

fn main() {
    // game loop
    use uttt::uttt_trait_solving::{UTTTBoard,parse_index_from_super_coords};
    use uttt::game_solving::{umcts, AlphaBetaTTCompatible, GameMove, MCTSAction, UMCTSNode,
                             GameResult, advance_rollout, reverse_rollout,
        IdentityHasher, IdentityHasherBuilder
    };
    use std::time;
    use std::collections::HashMap;

    let mut uttt_board = UTTTBoard {
        xbb: [0u16 ;9],
        obb: [0u16 ;9],
        last_super_index: 9,
        o_supers: 0,
        x_supers: 0,
        x_to_move: true,
    };

    let mut turn_count = 0;
    let mut node_store: HashMap<u64, UMCTSNode> = HashMap::with_capacity(40_000_000);

    let mut legal_moves: Vec<Vec<Vec<Vec<GameMove>>>> = Vec::new();


    let mut uttt_board = UTTTBoard {
        xbb: [0u16,0u16,0u16,0u16,16u16,0u16,0u16,0u16,0u16],
        obb: [0u16,0u16,0u16,0u16, 0u16,0u16,0u16,0u16,0u16],
        last_super_index: 4,
        x_supers: 0,
        o_supers: 0,
        x_to_move: false,
    };

    let orig_board = uttt_board.clone();
    let mut unmove_queue = Vec::with_capacity(81);
//    let mut node_store = HashMap::new();
    let mut node_store= HashMap::with_hasher(IdentityHasherBuilder{});
    let mut legal_moves = UTTTBoard::get_move_orderings();

    loop {
        // ***************************************************************
        // setting start of turn timer

        // **************************************************************
        let turn_start = time::Instant::now();
        let time_limit = if turn_count == 0 {
            legal_moves = UTTTBoard::get_move_orderings();
            9_998_000
        } else {
            9_998_000
        };


        let mut rollout_count = 0;

        let mut search_alpha = std::i8::MIN;
        let mut search_beta  = std::i8::MAX;

        let board_hash = uttt_board.hash_val();

        while turn_start.elapsed().as_micros() < time_limit {
            let mut rollout_result = 0;
            while let Some(new_alpha_beta) = advance_rollout(&mut unmove_queue,
                                                             &mut uttt_board, &legal_moves, search_alpha, search_beta, &mut node_store){
                let (new_alpha, new_beta) = new_alpha_beta;
                search_alpha = new_alpha;
                search_beta = new_beta;
            };

            let rollout_result = match uttt_board.get_win_state() {
                    GameResult::P1Win => std::i8::MAX,
                    GameResult::P2Win => std::i8::MIN,
                    GameResult::Tie   => 0,
                    _ => search_alpha,
                };
            //            rollout_result: i8,
            //            board_queue: &mut Vec<umcts_queue_element<T,Unmove>>,
            //            board: &mut BoardState, legal_moves: &Vec<Vec<Vec<Vec<GameMove>>>>,
            //            node_store: &mut HashMap<T,UMCTSNode>
            reverse_rollout(rollout_result, &mut unmove_queue, &mut uttt_board,
                            &legal_moves, &mut node_store
            );
//            assert_eq!(orig_board, uttt_board);
            rollout_count +=1;
        }

        let moves = uttt_board.get_legal_moves(&legal_moves);
        let mut mut_board = uttt_board.clone();

        let best_move_option = moves.iter().max_by_key(|a_move| {
            let undo = mut_board.make_move(a_move);
            let this_hash = mut_board.hash_val();
            mut_board.unmake_move(undo);
            let child_stats = node_store.get(&this_hash);
            let child_visit_count = match child_stats {
                Some(node) => node.visits,
                None => 0,
            };
            child_visit_count
        });

        let best_move = match best_move_option {
            Some(game_move) => game_move,
            None => {
                eprintln!("finished playing");
                return;
            }
        };

        let moves = uttt_board.get_legal_moves(&legal_moves);

        for each_move in moves.iter() {
            eprintln!("========================");
//                eprintln!("x to move? {}",each_move.x_to_move);
            let undo_move = uttt_board.make_move(each_move);
            let this_hash = uttt_board.hash_val();
            uttt_board.unmake_move(undo_move);
            let child_stats = node_store.get(&this_hash);

            match child_stats {
                Some(node) => {
                    eprintln!("node avg value: {}",(node.value as f32)/(node.visits as f32));
                    eprintln!("node visits: {}",node.visits);
                    eprintln!("move index: {}", each_move.move_index);
                }
                None => {
                    eprintln!("no stored statistics found for hash value: {}",this_hash);
                }
            }
            eprintln!("");

        }

        uttt_board.make_move(best_move);

        turn_count +=1;
        eprintln!("board after move:\n{}", uttt_board.display());
        eprintln!("elapsed microseconds: {}",turn_start.elapsed().as_micros());
        eprintln!("completed rollouts: {}", rollout_count);
        eprintln!("board representation post-move:\n{:?}", uttt_board);
        eprintln!("table size: {}", node_store.len());
    }
}









