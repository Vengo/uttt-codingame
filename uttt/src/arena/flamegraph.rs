use std::time::Duration;
use pprof::ProfilerGuardBuilder;
use crate::bots::create_bot;
use crate::game_solving::AlphaBetaCompatible;
use crate::ttt_trait_solving::TicTacToeEngine;
use crate::uttt_trait_solving::UltimateTicTacToeBoard;

pub struct FlamegraphSampler;

impl FlamegraphSampler {
    /// Profiles a bot's search over `duration`, generating an interactive SVG flamegraph
    pub fn profile_bot(bot_name: &str, duration: Duration) -> Result<Vec<u8>, String> {
        let guard = ProfilerGuardBuilder::default()
            .frequency(100)
            .blocklist(&["libc", "libgcc", "pthread", "libm"])
            .build()
            .map_err(|e| format!("Failed to initialize profiler: {}", e))?;

        let mut bot = create_bot(bot_name).ok_or_else(|| format!("Bot '{}' not found", bot_name))?;
        let engine = TicTacToeEngine::new();
        let legal_moves = engine.get_move_ordering_array();
        let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
        let root_board = UltimateTicTacToeBoard::default();

        let start = std::time::Instant::now();
        let mut sim_board = root_board.clone();

        while start.elapsed() < duration {
            let m = bot.choose_move(&sim_board, &legal_moves, &zobrist_constants, Duration::from_millis(30));
            sim_board.make_move(m, &zobrist_constants);
            if sim_board.get_win_state() != crate::game_solving::GameResult::Unresolved {
                sim_board = root_board.clone();
                bot.reset();
            }
        }

        let report = guard
            .report()
            .build()
            .map_err(|e| format!("Failed to build profile report: {}", e))?;

        let mut svg_bytes = Vec::new();
        report
            .flamegraph(&mut svg_bytes)
            .map_err(|e| format!("Failed to generate flamegraph SVG: {}", e))?;

        Ok(svg_bytes)
    }
}
