use std::env;
use std::io::{self, Write};
use std::time::Duration;
use uttt::arena::{Game, MatchRunner, Tournament};
use uttt::bots::{available_bots, create_bot};
use uttt::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use uttt::ttt_trait_solving::TicTacToeEngine;
use uttt::uttt_trait_solving::{parse_index_from_super_coords, UltimateTicTacToeBoard};

fn print_usage() {
    println!("Ultimate Tic-Tac-Toe Arena & Bot Tournament Runner");
    println!("Usage:");
    println!("  arena tournament [--games <N>] [--time-ms <T>] [--bots <b1,b2,...>]");
    println!("  arena match --bot1 <name> --bot2 <name> [--games <N>] [--time-ms <T>] [--replay <path>]");
    println!("  arena play --bot <name> [--time-ms <T>] [--play-as <x|o>]");
    println!("  arena server [--port <P>] [--host <H>]");
    println!("  arena list");
    println!("\nAvailable bots: {:?}", available_bots());
}

fn print_bot_descriptions() {
    println!("\n=== Available Ultimate Tic-Tac-Toe Bots ===");
    println!("  1. random         - Baseline choosing uniformly random legal moves.");
    println!("  2. rule_based     - Fast expert heuristic bot: captures subboards, blocks wins, steers opponent.");
    println!("  3. alphabeta      - Minimax with Alpha-Beta pruning, iterative deepening, and superboard evaluation.");
    println!("  4. mcts           - Classic Monte Carlo Tree Search (UCT) with random rollouts and backprop.");
    println!("  5. heuristic_mcts - Advanced MCTS combining UCT tree search with heuristic rollout policies.");
    println!("  6. original_umcts - Original UMCTS search engine from the repository.");
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "list" | "bots" => {
            print_bot_descriptions();
        }
        "tournament" => {
            run_tournament_cmd(&args[2..]);
        }
        "match" => {
            run_match_cmd(&args[2..]);
        }
        "play" => {
            run_play_cmd(&args[2..]);
        }
        "replay" => {
            run_replay_cmd(&args[2..]);
        }
        "server" | "serve" => {
            run_server_cmd(&args[2..]);
        }
        "--help" | "-h" | "help" => {
            print_usage();
        }
        unknown => {
            eprintln!("Unknown command: {}", unknown);
            print_usage();
        }
    }
}

fn parse_arg_usize(args: &[String], flag: &str, default: usize) -> usize {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().unwrap_or(default);
        }
    }
    default
}

fn parse_arg_u64(args: &[String], flag: &str, default: u64) -> u64 {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().unwrap_or(default);
        }
    }
    default
}

fn parse_arg_string(args: &[String], flag: &str, default: &str) -> String {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

fn run_tournament_cmd(args: &[String]) {
    let games = parse_arg_usize(args, "--games", 10);
    let time_ms = parse_arg_u64(args, "--time-ms", 40);
    let bots_arg = parse_arg_string(args, "--bots", "");

    let bot_names: Vec<&str> = if bots_arg.is_empty() {
        available_bots()
    } else {
        bots_arg.split(',').map(|s| s.trim()).collect()
    };

    println!("Initializing TicTacToeEngine lookup tables...");
    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let tournament = Tournament::new(Duration::from_millis(time_ms), games);
    let result = tournament.run(&bot_names, &legal_moves, &zobrist_constants);

    // Write result to JSON
    let json = serde_json::to_string_pretty(&result).unwrap();
    if let Ok(mut f) = std::fs::File::create("tournament_results.json") {
        let _ = f.write_all(json.as_bytes());
        println!("Full tournament results written to tournament_results.json");
    }
}

fn run_match_cmd(args: &[String]) {
    let bot1_name = parse_arg_string(args, "--bot1", "alphabeta");
    let bot2_name = parse_arg_string(args, "--bot2", "rule_based");
    let games = parse_arg_usize(args, "--games", 20);
    let time_ms = parse_arg_u64(args, "--time-ms", 40);
    let replay_path = parse_arg_string(args, "--replay", "");

    println!("Setting up match: {} vs {} ({} games, {}ms/move)", bot1_name, bot2_name, games, time_ms);
    let mut bot1 = create_bot(&bot1_name).expect("Failed to create bot1");
    let mut bot2 = create_bot(&bot2_name).expect("Failed to create bot2");

    println!("Initializing engine...");
    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let runner = MatchRunner::new(Duration::from_millis(time_ms)).with_replays(!replay_path.is_empty());
    let res = runner.run_match(
        bot1.as_mut(),
        bot2.as_mut(),
        games,
        &legal_moves,
        &zobrist_constants,
        None,
    );

    println!("\n=======================================================");
    println!("                     MATCH SUMMARY                     ");
    println!("=======================================================");
    println!("  {:<15} Wins: {:<4} ({:.1}%)", res.bot1_name, res.bot1_wins, res.bot1_winrate);
    println!("  {:<15} Wins: {:<4} ({:.1}%)", res.bot2_name, res.bot2_wins, res.bot2_winrate);
    println!("  Draws:           {:<4}", res.draws);
    println!("=======================================================\n");

    if !replay_path.is_empty() {
        println!("Match replay saved to {}", replay_path);
    }
}

fn run_replay_cmd(args: &[String]) {
    let bot1_name = parse_arg_string(args, "--bot1", "alphabeta");
    let bot2_name = parse_arg_string(args, "--bot2", "rule_based");
    let out_path = parse_arg_string(args, "--out", "replay.json");
    let time_ms = parse_arg_u64(args, "--time-ms", 50);

    let mut bot1 = create_bot(&bot1_name).expect("Failed to create bot1");
    let mut bot2 = create_bot(&bot2_name).expect("Failed to create bot2");

    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    println!("Playing 1 game between {} (X) and {} (O)...", bot1_name, bot2_name);
    let game = Game::new(Duration::from_millis(time_ms)).with_replay(true);
    let (stats, replay) = game.play(bot1.as_mut(), bot2.as_mut(), &legal_moves, &zobrist_constants);

    println!("Outcome: {:?}", stats.outcome);
    println!("Total turns: {}, Total time: {:.1}ms", stats.total_turns, stats.total_duration_ms);

    if let Some(r) = replay {
        r.save_to_file(&out_path).expect("Failed to write replay file");
        println!("Replay saved successfully to: {}", out_path);
    }
}

fn run_play_cmd(args: &[String]) {
    let bot_name = parse_arg_string(args, "--bot", "alphabeta");
    let time_ms = parse_arg_u64(args, "--time-ms", 100);
    let play_as = parse_arg_string(args, "--play-as", "x").to_lowercase();

    let human_is_x = play_as != "o";
    let mut bot = create_bot(&bot_name).expect("Failed to create bot");

    println!("Initializing engine...");
    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut board = UltimateTicTacToeBoard::default();

    println!("\n=======================================================");
    println!("        ULTIMATE TIC-TAC-TOE: YOU vs {}", bot.name().to_uppercase());
    println!("=======================================================");
    println!("You are playing as: {}", if human_is_x { "X (First)" } else { "O (Second)" });
    println!("Enter moves as: <row> <col> (0 to 8)\n");

    loop {
        println!("{}", board.display());

        let win_state = board.get_win_state();
        if win_state != GameResult::Unresolved {
            match win_state {
                GameResult::P1Win => println!("🎉 GAME OVER: Player 1 (X) WINS!"),
                GameResult::P2Win => println!("🎉 GAME OVER: Player 2 (O) WINS!"),
                GameResult::Tie => println!("🤝 GAME OVER: IT'S A TIE!"),
                _ => {}
            }
            break;
        }

        let is_x = board.x_to_move;
        let human_turn = (is_x && human_is_x) || (!is_x && !human_is_x);

        let legal_moves_list = board.get_legal_moves(&legal_moves);
        if legal_moves_list.is_empty() {
            println!("No legal moves left!");
            break;
        }

        if human_turn {
            println!("Your turn ({}): Enter row and col (0-8):", if is_x { "X" } else { "O" });
            if board.last_super_index < 9 {
                let complete = board.x_supers | board.o_supers | board.get_tied_mask();
                if (complete & (1 << board.last_super_index)) == 0 {
                    println!("Must play in subboard index: {}", board.last_super_index);
                } else {
                    println!("Subboard {} is complete! Free move anywhere.", board.last_super_index);
                }
            } else {
                println!("Free move anywhere!");
            }

            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("Failed to read line");
            let parts: Vec<&str> = input.trim().split_whitespace().collect();
            if parts.len() < 2 {
                println!("Invalid input! Please enter two numbers: row col (e.g. 4 4)");
                continue;
            }

            let r: u8 = match parts[0].parse() {
                Ok(val) if val < 9 => val,
                _ => { println!("Row must be between 0 and 8"); continue; }
            };
            let c: u8 = match parts[1].parse() {
                Ok(val) if val < 9 => val,
                _ => { println!("Col must be between 0 and 8"); continue; }
            };

            let move_idx = parse_index_from_super_coords(r, c);
            let chosen = GameMove {
                x_to_move: is_x,
                move_index: move_idx,
            };

            if !legal_moves_list.iter().any(|m| m.move_index == chosen.move_index) {
                println!("⚠️ Move ({}, {}) is ILLEGAL! Choose from active subboard.", r, c);
                continue;
            }

            board.make_move(chosen, &zobrist_constants);
        } else {
            println!("Bot ({}) is thinking...", bot.name());
            let bot_move = bot.choose_move(
                &board,
                &legal_moves,
                &zobrist_constants,
                Duration::from_millis(time_ms),
            );
            let parsed = uttt::uttt_trait_solving::parse_index(bot_move.move_index);
            let r = (parsed.sub_board_index / 3) * 3 + (parsed.sub_square_index / 3);
            let c = (parsed.sub_board_index % 3) * 3 + (parsed.sub_square_index % 3);
            println!("Bot played: row {} col {} (index {})", r, c, bot_move.move_index);
            board.make_move(bot_move, &zobrist_constants);
        }
    }
}

fn run_server_cmd(args: &[String]) {
    let port = parse_arg_u64(args, "--port", 8080) as u16;
    let host = parse_arg_string(args, "--host", "0.0.0.0");
    let server = uttt::arena::ArenaServer::new(&host, port);
    server.start();
}

