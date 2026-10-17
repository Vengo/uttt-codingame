use crate::arena::replay::GameReplay;
use crate::bots::Bot;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GameOutcome {
    XWin,
    OWin,
    Draw,
    XForfeit(String),
    OForfeit(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStats {
    pub player_x: String,
    pub player_o: String,
    pub outcome: GameOutcome,
    pub total_turns: usize,
    pub avg_time_x_ms: f64,
    pub avg_time_o_ms: f64,
    pub total_duration_ms: f64,
}

pub struct Game {
    pub time_limit_per_move: Duration,
    pub record_replay: bool,
}

impl Game {
    pub fn new(time_limit_per_move: Duration) -> Self {
        Game {
            time_limit_per_move,
            record_replay: false,
        }
    }

    pub fn with_replay(mut self, record: bool) -> Self {
        self.record_replay = record;
        self
    }

    pub fn play(
        &self,
        bot_x: &mut dyn Bot,
        bot_o: &mut dyn Bot,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> (GameStats, Option<GameReplay>) {
        self.play_from_position(
            bot_x,
            bot_o,
            &UltimateTicTacToeBoard::default(),
            legal_moves_lookup,
            zobrist_constants,
        )
    }

    pub fn play_from_position(
        &self,
        bot_x: &mut dyn Bot,
        bot_o: &mut dyn Bot,
        initial_board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> (GameStats, Option<GameReplay>) {
        bot_x.reset();
        bot_o.reset();

        let mut board = initial_board.clone();
        let mut turn = 0;
        let mut total_time_x = 0.0;
        let mut turns_x = 0;
        let mut total_time_o = 0.0;
        let mut turns_o = 0;

        let game_id = format!("{}_vs_{}_{}", bot_x.name(), bot_o.name(), Instant::now().elapsed().as_nanos());
        let mut replay = if self.record_replay {
            let mut r = GameReplay::new(&game_id, bot_x.name(), bot_o.name());
            r.initial_board_display = Some(initial_board.display());
            Some(r)
        } else {
            None
        };

        let game_start = Instant::now();
        let outcome;

        loop {
            let win_state = board.get_win_state();
            if win_state != GameResult::Unresolved {
                outcome = match win_state {
                    GameResult::P1Win => GameOutcome::XWin,
                    GameResult::P2Win => GameOutcome::OWin,
                    GameResult::Tie => GameOutcome::Draw,
                    GameResult::Unresolved => unreachable!(),
                };
                break;
            }

            let legal_moves = if !legal_moves_lookup.is_empty() {
                board.get_legal_moves(legal_moves_lookup)
            } else {
                board.get_legal_moves_fast()
            };

            if legal_moves.is_empty() {
                // No legal moves left, check final state
                outcome = match board.get_win_state() {
                    GameResult::P1Win => GameOutcome::XWin,
                    GameResult::P2Win => GameOutcome::OWin,
                    _ => GameOutcome::Draw,
                };
                break;
            }

            let is_x = board.x_to_move;
            let current_bot: &mut dyn Bot = if is_x { bot_x } else { bot_o };

            let move_start = Instant::now();
            let chosen_move = current_bot.choose_move(
                &board,
                legal_moves_lookup,
                zobrist_constants,
                self.time_limit_per_move,
            );
            let move_duration = move_start.elapsed().as_secs_f64() * 1000.0;

            if is_x {
                total_time_x += move_duration;
                turns_x += 1;
            } else {
                total_time_o += move_duration;
                turns_o += 1;
            }

            // Verify legality
            let is_legal = legal_moves.iter().any(|m| m.move_index == chosen_move.move_index);
            if !is_legal {
                let err = format!("Illegal move chosen: index {}", chosen_move.move_index);
                outcome = if is_x {
                    GameOutcome::XForfeit(err)
                } else {
                    GameOutcome::OForfeit(err)
                };
                break;
            }

            let _unmove = board.make_move(chosen_move, zobrist_constants);
            turn += 1;

            if let Some(r) = replay.as_mut() {
                r.record_move(turn, chosen_move, move_duration, board.display());
            }

            if turn >= 150 {
                // Safeguard against infinite loops
                outcome = GameOutcome::Draw;
                break;
            }
        }

        let total_duration_ms = game_start.elapsed().as_secs_f64() * 1000.0;
        let avg_time_x_ms = if turns_x > 0 { total_time_x / (turns_x as f64) } else { 0.0 };
        let avg_time_o_ms = if turns_o > 0 { total_time_o / (turns_o as f64) } else { 0.0 };

        if let Some(r) = replay.as_mut() {
            r.total_turns = turn;
            r.total_time_ms = total_duration_ms;
            r.result = match &outcome {
                GameOutcome::XWin => format!("Winner: {}", bot_x.name()),
                GameOutcome::OWin => format!("Winner: {}", bot_o.name()),
                GameOutcome::Draw => "Draw".to_string(),
                GameOutcome::XForfeit(reason) => format!("X Forfeit: {}", reason),
                GameOutcome::OForfeit(reason) => format!("O Forfeit: {}", reason),
            };
        }

        let stats = GameStats {
            player_x: bot_x.name().to_string(),
            player_o: bot_o.name().to_string(),
            outcome,
            total_turns: turn,
            avg_time_x_ms,
            avg_time_o_ms,
            total_duration_ms,
        };

        (stats, replay)
    }
}
