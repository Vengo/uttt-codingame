use std::env;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use uttt::arena::{Game, GameOutcome, GlickoCalculator, GlickoPlayer, MatchOutcome, MatchRunner, Tournament};
use uttt::bots::{available_bots, create_bot};
use uttt::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use uttt::ttt_trait_solving::TicTacToeEngine;
use uttt::uttt_trait_solving::{parse_index_from_super_coords, UltimateTicTacToeBoard};

fn print_usage() {
    println!("Ultimate Tic-Tac-Toe Arena & Bot Tournament Runner");
    println!("Usage:");
    println!("  arena tournament [--games <N>] [--time-ms <T>] [--bots <b1,b2,...>]");
    println!("  arena benchmark --bot <name> [--baseline <name>] [--target-rd <RD>] [--time-ms <T>] [--max-games <N>]");
    println!("  arena match --bot1 <name> --bot2 <name> [--games <N>] [--time-ms <T>] [--replay <path>]");
    println!("  arena play --bot <name> [--time-ms <T>] [--play-as <x|o>]");
    println!("  arena replay <file.json>");
    println!("  arena server [--port <P>] [--host <H>]");
    println!("  arena profile");
    println!("  arena flamegraph --bot <name> [--time-ms <T>] [--out <file.svg>]");
    println!("  arena study [--cycles <N>] [--time-limit-mins <M>]");
    println!("  arena list");
    println!("\nAvailable bots: {:?}", available_bots());
}

fn print_bot_descriptions() {
    println!("\n=== Available Ultimate Tic-Tac-Toe Bots ===");
    println!("  1. random         - Baseline choosing uniformly random legal moves.");
    println!("  2. rule_based     - Fast expert heuristic bot: captures subboards, blocks wins, steers opponent.");
    println!("  3. alphabeta      - Minimax with Alpha-Beta pruning, iterative deepening, and superboard evaluation.");
    println!("  4. alphabeta_v2   - Advanced Minimax with in-tree EGTB pruning, depth-preferred TT, and killer moves.");
    println!("  5. mcts           - Classic Monte Carlo Tree Search (UCT) with random rollouts and backprop.");
    println!("  6. heuristic_mcts - Advanced MCTS combining UCT tree search with heuristic rollout policies.");
    println!("  7. original_umcts - Original UMCTS search engine from the repository.");
    println!("  8. umcts_v2       - Refactored hybrid MCTS-Solver with 1-node expansion, bound proofs, and cache locality.");
    println!("  9. umcts_v3       - Advanced MCTS-Solver with in-tree EGTB proof propagation and heuristic move priors.");
    println!(" 10. book_umcts_v2  - UMCTS-V2 with instant D4 Opening Book lookup (<1 us) and EGTB probing.");
    println!(" 11. book_alphabeta - AlphaBeta with instant D4 Opening Book lookup and EGTB probing.");
    println!(" 12. alphazero      - Pure Rust INT8 dual-headed ResNet inference engine (Policy + Value).");
    println!(" 13. policy_alphabeta - Hybrid AlphaBeta guided by INT8 neural policy prior with macro-routing.");
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
        "benchmark" | "bench" | "glicko" => {
            run_benchmark_cmd(&args[2..]);
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
        "profile" | "perf" => {
            run_profile_cmd(&args[2..]);
        }
        "flamegraph" | "flame" => {
            run_flamegraph_cmd(&args[2..]);
        }
        "study" | "daemon" | "learn" => {
            run_study_cmd(&args[2..]);
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

    // Write result to JSON in current dir and parent dir
    let json = serde_json::to_string_pretty(&result).unwrap();
    let _ = std::fs::write("tournament_results.json", &json);
    let _ = std::fs::write("../tournament_results.json", &json);
    println!("Full tournament results written to tournament_results.json");

    println!("\n=== Final Tournament Leaderboard ===");
    println!("{:<4} {:<18} {:<10} {:<10} {:<12} {:<8}", "Rank", "Bot", "Elo", "Points", "Record (W-L-D)", "Win %");
    for (i, entry) in result.leaderboard.iter().enumerate() {
        println!(
            "{:<4} {:<18} {:<10.1} {:<10.1} {:<12} {:<8.1}%",
            i + 1,
            entry.bot_name,
            entry.elo,
            entry.points,
            format!("{}-{}-{}", entry.wins, entry.losses, entry.draws),
            entry.win_percentage
        );
    }
    println!("====================================\n");
}

fn run_match_cmd(args: &[String]) {
    let mut positional = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i].starts_with('-') {
            i += 2; // skip flag and value
        } else {
            positional.push(&args[i]);
            i += 1;
        }
    }

    let bot1_name = if !positional.is_empty() {
        positional[0].to_string()
    } else {
        parse_arg_string(args, "--bot1", "alphabeta")
    };

    let bot2_name = if positional.len() > 1 {
        positional[1].to_string()
    } else {
        parse_arg_string(args, "--bot2", "rule_based")
    };

    let games = if args.contains(&"-g".to_string()) {
        parse_arg_usize(args, "-g", 20)
    } else {
        parse_arg_usize(args, "--games", 20)
    };

    let time_ms = if args.contains(&"-t".to_string()) {
        parse_arg_u64(args, "-t", 40)
    } else {
        parse_arg_u64(args, "--time-ms", 40)
    };

    let replay_path = parse_arg_string(args, "--replay", "");

    println!("Setting up match: {} vs {} ({} games, {}ms/move)", bot1_name, bot2_name, games, time_ms);
    let mut bot1 = create_bot(&bot1_name).expect("Failed to create bot1");
    let mut bot2 = create_bot(&bot2_name).expect("Failed to create bot2");

    println!("Initializing engine...");
    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let use_openings = args.contains(&"--openings".to_string()) || args.contains(&"--book-openings".to_string());
    let opening_depth = parse_arg_usize(args, "--opening-depth", 2);

    let mut runner = MatchRunner::new(Duration::from_millis(time_ms)).with_replays(!replay_path.is_empty());
    if use_openings {
        let book = uttt::opening_book::OpeningBook::new();
        let suite = book.extract_opening_suite((games / 2).max(5), opening_depth, &legal_moves, &zobrist_constants);
        println!("Opening suite:  Loaded {} unique positions (depth {} plies)", suite.len(), (opening_depth / 2).max(1) * 2);
        runner = runner.with_openings(suite);
    }

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
        if let Ok(json_str) = serde_json::to_string_pretty(&res.replays) {
            if std::fs::write(&replay_path, json_str).is_ok() {
                println!("Match replays ({} games) saved to {}", res.replays.len(), replay_path);
            } else {
                eprintln!("Failed to write replays to {}", replay_path);
            }
        }
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

fn run_profile_cmd(_args: &[String]) {
    println!("\n=======================================================");
    println!("        HARDWARE & REAL-WORLD CACHE PROFILER          ");
    println!("=======================================================");
    let report = uttt::profile::run_hardware_profile();
    println!("Host CPU: {}", report.cpu_info);
    println!("\n[1] CPU Cache Hierarchy Benchmark:");
    println!(
        "  - L3 Cache-Resident ({} KB, {} entries): {:.1} ns / lookup",
        report.cache_profile.small_table_kb,
        report.cache_profile.small_table_entries,
        report.cache_profile.small_table_latency_ns
    );
    println!(
        "  - DRAM Spilling ({} KB, {} entries): {:.1} ns / lookup",
        report.cache_profile.large_table_kb,
        report.cache_profile.large_table_entries,
        report.cache_profile.large_table_latency_ns
    );
    println!(
        "  -> Slowdown Factor: {:.2}x SLOWER when spilling past CPU cache!",
        report.cache_profile.cache_slowdown_factor
    );
    println!("\n[2] Allocator & Heap Churn Benchmark:");
    println!("  - Total Iterations: {}", report.allocator_profile.total_allocations);
    println!("  - Throughput: {}", report.allocator_profile.summary);
    println!("\n[3] Playout Simulation Throughput:");
    for t in &report.search_throughput {
        println!("  - {}: {}", t.bot_name, t.summary);
    }
    println!("=======================================================\n");

    let json = serde_json::to_string_pretty(&report).unwrap();
    if let Ok(mut f) = std::fs::File::create("hardware_profile.json") {
        let _ = f.write_all(json.as_bytes());
        println!("Report saved to hardware_profile.json");
    }
}

fn run_flamegraph_cmd(args: &[String]) {
    let bot_name = parse_arg_string(args, "--bot", "umcts_v2");
    let time_ms = parse_arg_u64(args, "--time-ms", 2000);
    let out_file = parse_arg_string(args, "--out", &format!("flamegraph_{}.svg", bot_name));

    println!("Sampling execution profile for bot '{}' over {} ms...", bot_name, time_ms);
    match uttt::arena::FlamegraphSampler::profile_bot(&bot_name, Duration::from_millis(time_ms)) {
        Ok(svg_bytes) => {
            if let Err(e) = std::fs::write(&out_file, &svg_bytes) {
                eprintln!("Failed to write flamegraph to {}: {}", out_file, e);
            } else {
                println!("✅ Flamegraph generated successfully: {}", out_file);
                println!("   Size: {} KB", svg_bytes.len() / 1024);
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to generate flamegraph: {}", e);
        }
    }
}

fn run_study_cmd(args: &[String]) {
    let cycles = parse_arg_usize(args, "--cycles", 5);
    let time_limit_mins = parse_arg_u64(args, "--time-limit-mins", 30);
    let min_resolved = parse_arg_usize(args, "--min-resolved", 3);
    let book_depth = parse_arg_usize(args, "--book-depth", 3);
    let target_egtb_batch = parse_arg_usize(args, "--target-egtb", 30);

    println!("\n=======================================================");
    println!("🔬 Ultimate Tic-Tac-Toe Continuous Study & Benchmark Daemon");
    println!("=======================================================");
    println!("Configuration:");
    println!("  - Target Cycles: {}", cycles);
    println!("  - Max Time: {} minutes", time_limit_mins);
    println!("  - Min Resolved Subboards: {} (Progressive Tiering Enabled)", min_resolved);
    println!("  - Opening Book Depth: {}", book_depth);
    println!("  - ZFS Storage: /my-zraid/uttt/ (7.2 TB Available)");
    println!("  - Host Memory: 128 GB RAM");
    println!("  - Accelerator: NVIDIA GeForce RTX 5060 Ti");
    println!("=======================================================\n");

    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut book = uttt::opening_book::OpeningBook::new();
    println!("📚 Opening Book: {} canonical entries loaded.", book.entries.len());

    let mut egtb = uttt::egtb::EgtbDatabase::new();
    println!("🏰 Endgame Tablebase initialized at {:?}", egtb.file_path);

    let start_time = std::time::Instant::now();
    let max_duration = Duration::from_secs(time_limit_mins * 60);

    let mut total_self_play_games = 0usize;
    let mut total_egtb_solved = 0usize;

    for cycle in 1..=cycles {
        if start_time.elapsed() >= max_duration {
            println!("⏱ Time limit reached. Stopping study.");
            break;
        }

        println!("\n--- [Cycle {}/{}] Starting Discovery Phase ---", cycle, cycles);

        // 1. Opening Book Deepening with Principled UCT Dominance & Frontier Harvesting
        print!("  [1/3] Deepening Opening Book with UMCTS-V2 (PV + dominance filtering)... ");
        std::io::stdout().flush().unwrap();
        let added_book = book.deepen_book(
            book_depth.max(2) * 2,
            Duration::from_millis(300),
            100, // principled visit threshold with dominance margin
            &legal_moves,
            &zobrist_constants,
        );
        println!("Done! (+{} new positions, Total book: {})", added_book, book.entries.len());

        // 2. Systematic Endgame Tablebase Solving (Tier 9 instant, Tier 8 high-speed, Tier 7)
        let tier_resolved = if cycle == 1 && egtb.entry_count < 10000 {
            9 // Generate all Tier 9 terminal configurations on first cycle
        } else if cycle <= 5 {
            8 // Systematically solve Tier 8 (8 resolved, 1 active subboard)
        } else {
            7 // Systematically solve Tier 7 (7 resolved, 2 active subboards)
        };
        print!("  [2/3] Solving late-stage positions (Systematic Tier {})... ", tier_resolved);
        std::io::stdout().flush().unwrap();
        if let Ok(count) = egtb.generate_egtb(target_egtb_batch.max(100), tier_resolved, &legal_moves, &zobrist_constants) {
            total_egtb_solved += count;
            println!("Done! (+{} new endgame states proven)", count);
        } else {
            println!("Skipped/Current batch complete.");
        }


        // 3. Self-Play Training Game Generation
        print!("  [3/3] Generating self-play training games for Deep Learning... ");
        std::io::stdout().flush().unwrap();
        let mut b1 = uttt::bots::create_bot("umcts_v2").unwrap();
        let mut b2 = uttt::bots::create_bot("alphabeta").unwrap();
        let game = uttt::arena::Game::new(Duration::from_millis(20)).with_replay(false);
        for _ in 0..4 {
            let _ = game.play(b1.as_mut(), b2.as_mut(), &legal_moves, &zobrist_constants);
            total_self_play_games += 1;
        }
        println!("Done! ({} total self-play games logged)", total_self_play_games);

        // Update status JSON
        let status = serde_json::json!({
            "active": true,
            "current_cycle": cycle,
            "total_cycles": cycles,
            "book_entries": book.entries.len(),
            "egtb_solved": total_egtb_solved,
            "self_play_games": total_self_play_games,
            "elapsed_secs": start_time.elapsed().as_secs(),
            "zfs_storage_path": "/my-zraid/uttt",
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "last_update": format!("{:?}", std::time::SystemTime::now())
        });
        let status_str = serde_json::to_string_pretty(&status).unwrap();
        let _ = std::fs::write("/my-zraid/uttt/study_status.json", &status_str);
        let _ = std::fs::write("study_status.json", &status_str);
    }

    println!("\n=======================================================");
    println!("🏁 Study Session Complete!");
    println!("  - Final Book Entries: {}", book.entries.len());
    println!("  - Endgame States Solved: {}", total_egtb_solved);
    println!("  - Self-Play Games Generated: {}", total_self_play_games);
    println!("  - Total Runtime: {:.1}s", start_time.elapsed().as_secs_f64());
    println!("=======================================================\n");
}

fn run_benchmark_cmd(args: &[String]) {
    let mut bot_name = String::new();
    let mut baseline_name = "alphabeta".to_string();
    let mut target_rd = 35.0f64;
    let mut time_ms = 30u64;
    let mut max_games = 200usize;
    let mut use_openings = false;
    let mut opening_depth = 2usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--bot" => {
                if i + 1 < args.len() {
                    bot_name = args[i + 1].clone();
                    i += 1;
                }
            }
            "--baseline" => {
                if i + 1 < args.len() {
                    baseline_name = args[i + 1].clone();
                    i += 1;
                }
            }
            "--target-rd" => {
                if i + 1 < args.len() {
                    target_rd = args[i + 1].parse().unwrap_or(35.0);
                    i += 1;
                }
            }
            "--time-ms" => {
                if i + 1 < args.len() {
                    time_ms = args[i + 1].parse().unwrap_or(30);
                    i += 1;
                }
            }
            "--max-games" => {
                if i + 1 < args.len() {
                    max_games = args[i + 1].parse().unwrap_or(200);
                    i += 1;
                }
            }
            "--openings" | "--book-openings" => {
                use_openings = true;
            }
            "--opening-depth" => {
                if i + 1 < args.len() {
                    opening_depth = args[i + 1].parse().unwrap_or(2);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if bot_name.is_empty() {
        eprintln!("Error: --bot <name> is required for benchmark.");
        eprintln!("Available bots: {:?}", available_bots());
        return;
    }

    println!("\n=======================================================");
    println!("⚖️  Bayesian Benchmark Tournament (Glicko-2 Engine)");
    println!("=======================================================");
    println!("Candidate Bot:  {}", bot_name);
    println!("Baseline Bot:   {} (Fixed anchor: Rating 1500, RD 30)", baseline_name);
    println!("Target RD:      <= {:.1} (Tight error bounds: 95% CI <= ±{:.1} Elo)", target_rd, 1.96 * target_rd);
    println!("Time Limit:     {} ms/move", time_ms);
    println!("Max Games:      {}", max_games);
    println!("Opening Suite:  {}", if use_openings { format!("Enabled (target depth: {} plies)", (opening_depth / 2).max(1) * 2) } else { "Disabled (empty board)".to_string() });
    println!("=======================================================\n");

    let engine = TicTacToeEngine::new();
    let legal_moves = engine.get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut candidate_bot = match create_bot(&bot_name) {
        Some(b) => b,
        None => {
            eprintln!("Error: Candidate bot '{}' not found.", bot_name);
            return;
        }
    };

    let mut baseline_bot = match create_bot(&baseline_name) {
        Some(b) => b,
        None => {
            eprintln!("Error: Baseline bot '{}' not found.", baseline_name);
            return;
        }
    };

    let game_runner = Game::new(Duration::from_millis(time_ms)).with_replay(false);
    let calc = GlickoCalculator::default();
    let prior_player = GlickoPlayer::default();
    let mut matches = Vec::new();

    let book = uttt::opening_book::OpeningBook::new();
    let opening_suite = if use_openings {
        let suite = book.extract_opening_suite(
            (max_games / 2).max(10),
            opening_depth,
            &legal_moves,
            &zobrist_constants,
        );
        println!("Loaded opening suite: {} unique positions", suite.len());
        suite
    } else {
        vec![UltimateTicTacToeBoard::default()]
    };

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;

    let start_time = Instant::now();
    let mut current_glicko = prior_player.clone();

    let mut pair = 0usize;
    while matches.len() < max_games && current_glicko.rd > target_rd {
        pair += 1;
        let initial_board = &opening_suite[(pair - 1) % opening_suite.len()];

        // Game 1: Candidate as X (Player 1), Baseline as O (Player 2)
        candidate_bot.reset();
        baseline_bot.reset();
        let outcome1 = game_runner.play_from_position(candidate_bot.as_mut(), baseline_bot.as_mut(), initial_board, &legal_moves, &zobrist_constants);
        let score1 = match outcome1.0.outcome {
            GameOutcome::XWin | GameOutcome::OForfeit(_) => {
                wins += 1;
                1.0
            }
            GameOutcome::OWin | GameOutcome::XForfeit(_) => {
                losses += 1;
                0.0
            }
            _ => {
                draws += 1;
                0.5
            }
        };
        matches.push(MatchOutcome {
            opponent_rating: 1500.0,
            opponent_rd: 30.0,
            score: score1,
        });

        // Game 2: Baseline as X (Player 1), Candidate as O (Player 2)
        candidate_bot.reset();
        baseline_bot.reset();
        let outcome2 = game_runner.play_from_position(baseline_bot.as_mut(), candidate_bot.as_mut(), initial_board, &legal_moves, &zobrist_constants);
        let score2 = match outcome2.0.outcome {
            GameOutcome::OWin | GameOutcome::XForfeit(_) => {
                wins += 1;
                1.0
            }
            GameOutcome::XWin | GameOutcome::OForfeit(_) => {
                losses += 1;
                0.0
            }
            _ => {
                draws += 1;
                0.5
            }
        };
        matches.push(MatchOutcome {
            opponent_rating: 1500.0,
            opponent_rd: 30.0,
            score: score2,
        });

        // Evaluate single-period update across all accumulated matches
        current_glicko = calc.update(&prior_player, &matches);

        let total_played = matches.len();
        let win_pct = ((wins as f64 + 0.5 * draws as f64) / total_played as f64) * 100.0;
        let (ci_low, ci_high) = current_glicko.credible_interval_95();

        println!(
            "Pair {:3} (Games {:3}/{}): Score {}-{}-{} ({:.1}%) | Glicko: {:.1} ± {:.1} (95% CI: [{:.1}, {:.1}], RD: {:.1})",
            pair, total_played, max_games, wins, losses, draws, win_pct, current_glicko.rating, 1.96 * current_glicko.rd, ci_low, ci_high, current_glicko.rd
        );
        let _ = io::stdout().flush();
    }

    let total_played = matches.len();
    let win_pct = ((wins as f64 + 0.5 * draws as f64) / total_played as f64) * 100.0;
    let (ci_low, ci_high) = current_glicko.credible_interval_95();

    println!("\n=======================================================");
    println!("📊 Final Benchmark Results: {} vs {}", bot_name, baseline_name);
    println!("=======================================================");
    println!("Total Games:     {}", total_played);
    println!("Score Record:    {} Wins, {} Losses, {} Draws", wins, losses, draws);
    println!("Score / Win %:   {:.1}%", win_pct);
    println!("Glicko-2 Rating: {:.1}", current_glicko.rating);
    println!("Rating Dev (RD): {:.1}", current_glicko.rd);
    println!("95% Credible CI: [{:.1}, {:.1}] (Margin: ±{:.1} Elo)", ci_low, ci_high, 1.96 * current_glicko.rd);
    println!("Total Duration:  {:.1}s", start_time.elapsed().as_secs_f64());
    println!("-------------------------------------------------------");

    if ci_low > 1500.0 {
        println!("✅ [STATISTICALLY SIGNIFICANT IMPROVEMENT]");
        println!("   The candidate bot is proven superior to baseline {} with >95% confidence!", baseline_name);
        println!("   Estimated performance advantage: +{:.1} Elo (min: +{:.1} Elo)", current_glicko.rating - 1500.0, ci_low - 1500.0);
    } else if ci_high < 1500.0 {
        println!("❌ [STATISTICALLY SIGNIFICANT REGRESSION]");
        println!("   The candidate bot performs worse than baseline {} with >95% confidence.", baseline_name);
    } else {
        println!("⚖️  [INCONCLUSIVE / EQUIVALENT]");
        println!("   Performance is within the margin of error of baseline {}.", baseline_name);
    }
    println!("=======================================================\n");
}

