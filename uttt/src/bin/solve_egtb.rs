use std::env;
use std::path::PathBuf;
use std::time::Instant;

use uttt::egtb::retrograde::RetrogradeSolver;

fn print_usage() {
    println!("============================================================");
    println!("  UTTT Scalable Exhaustive Tablebase Solver (solve_egtb)");
    println!("============================================================");
    println!("Usage:");
    println!("  solve_egtb --tier <8|7|6|merge6|all> [options]");
    println!();
    println!("Options:");
    println!("  --tier <N>            Endgame tier to solve: 8, 7, 6, merge6, or 'all'");
    println!("  --out-dir <path>       Output directory for generated .bin tablebases (default: 'egtb')");
    println!("  --threads <N>          Number of worker threads (default: all available cores)");
    println!("  --max-positions <N>    Maximum position budget for Tier 7/6 (default: 500,000)");
    println!("  --chunk <idx>          Chunk index for Tier 6 streaming (default: 0)");
    println!("  --chunks <total>       Total chunks for Tier 6 streaming (default: 8)");
    println!("  --help                 Show this help message");
    println!("============================================================");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.contains(&"--help".to_string()) || args.contains(&"-h".to_string()) || args.len() < 2 {
        print_usage();
        return;
    }

    let mut tier = "8".to_string();
    let mut out_dir = PathBuf::from("egtb");
    let mut threads = 0usize;
    let mut max_positions = 500_000usize;
    let mut chunk_idx = 0usize;
    let mut total_chunks = 8usize;
    let mut chunk_specified = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tier" => {
                if i + 1 < args.len() {
                    tier = args[i + 1].clone();
                    i += 1;
                }
            }
            "--out-dir" => {
                if i + 1 < args.len() {
                    out_dir = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--threads" => {
                if i + 1 < args.len() {
                    threads = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--max-positions" => {
                if i + 1 < args.len() {
                    max_positions = args[i + 1].parse().unwrap_or(500_000);
                    i += 1;
                }
            }
            "--chunk" => {
                if i + 1 < args.len() {
                    chunk_idx = args[i + 1].parse().unwrap_or(0);
                    chunk_specified = true;
                    i += 1;
                }
            }
            "--chunks" => {
                if i + 1 < args.len() {
                    total_chunks = args[i + 1].parse().unwrap_or(8);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let solver = RetrogradeSolver::new(&out_dir, threads);
    let total_start = Instant::now();

    match tier.as_str() {
        "8" => {
            let mf_solver = uttt::egtb::macro_factored::MacroFactoredSolver::new(&out_dir, threads);
            match mf_solver.solve_tier_8() {
                Ok((count, path)) => {
                    println!("\n✅ Successfully solved Macro-Factored Tier 8: {} canonical entries in {:?}", count, path);
                }
                Err(err) => eprintln!("❌ Tier 8 solver error: {}", err),
            }
        }
        "7" => {
            let mf_solver = uttt::egtb::macro_factored::MacroFactoredSolver::new(&out_dir, threads);
            let empty_limit = if max_positions > 0 && max_positions <= 10 { max_positions } else { 4 };
            match mf_solver.solve_tier_7(empty_limit) {
                Ok((count, path)) => {
                    println!("\n✅ Successfully solved Macro-Factored Tier 7: {} canonical entries in {:?}", count, path);
                }
                Err(err) => eprintln!("❌ Tier 7 solver error: {}", err),
            }
        }
        "6" | "all6" => {
            let mf_solver = uttt::egtb::macro_factored::MacroFactoredSolver::new(&out_dir, threads);
            let empty_limit = if max_positions > 0 && max_positions <= 4 { max_positions } else { 1 };
            match mf_solver.solve_tier_6(empty_limit) {
                Ok((count, path)) => {
                    println!("\n✅ Successfully solved Macro-Factored Tier 6: {} canonical entries in {:?}", count, path);
                }
                Err(err) => eprintln!("❌ Tier 6 solver error: {}", err),
            }
        }
        "merge6" | "merge-tier6" => {
            let chunks_dir = out_dir.join("chunks");
            let out_file = out_dir.join("egtb_tier6.bin");
            match solver.merge_tier_6_chunks(&chunks_dir, &out_file) {
                Ok(count) => {
                    println!("\n✅ Successfully merged Tier 6 chunks: {} canonical entries in {:?}", count, out_file);
                }
                Err(err) => eprintln!("❌ Tier 6 merge error: {}", err),
            }
        }
        "all" => {
            println!("🚀 Executing Full Systematic Solving: Tier 8, Tier 7, and Tier 6 Chunk 0...");
            if let Ok((count8, path8)) = solver.solve_tier_8() {
                println!("✅ Tier 8: {} entries in {:?}", count8, path8);
            }
            if let Ok((count7, path7)) = solver.solve_tier_7(max_positions) {
                println!("✅ Tier 7: {} entries in {:?}", count7, path7);
            }
            if let Ok((count6, path6)) = solver.solve_tier_6_chunk(0, total_chunks, max_positions) {
                println!("✅ Tier 6 Chunk 0: {} entries in {:?}", count6, path6);
            }
        }
        other => {
            eprintln!("Unknown tier '{}'. Supported: 8, 7, 6, merge6, all", other);
        }
    }

    println!("Total solve execution time: {:.2}s", total_start.elapsed().as_secs_f64());
}
