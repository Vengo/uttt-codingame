use crate::game_solving::GameMove;
use crate::uttt_trait_solving::parse_index;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveRecord {
    pub turn: usize,
    pub player: String,
    pub move_index: u8,
    pub row: usize,
    pub col: usize,
    pub subboard: usize,
    pub subsquare: usize,
    pub duration_ms: f64,
    pub board_display: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameReplay {
    pub game_id: String,
    pub player_x: String,
    pub player_o: String,
    pub result: String,
    pub total_turns: usize,
    pub total_time_ms: f64,
    pub moves: Vec<MoveRecord>,
}

impl GameReplay {
    pub fn new(game_id: &str, player_x: &str, player_o: &str) -> Self {
        GameReplay {
            game_id: game_id.to_string(),
            player_x: player_x.to_string(),
            player_o: player_o.to_string(),
            result: "In Progress".to_string(),
            total_turns: 0,
            total_time_ms: 0.0,
            moves: Vec::new(),
        }
    }

    pub fn record_move(
        &mut self,
        turn: usize,
        m: GameMove,
        duration_ms: f64,
        board_display: String,
    ) {
        let parsed = parse_index(m.move_index);
        let subboard = parsed.sub_board_index;
        let subsquare = parsed.sub_square_index;

        let row = (subboard / 3) * 3 + (subsquare / 3);
        let col = (subboard % 3) * 3 + (subsquare % 3);

        self.moves.push(MoveRecord {
            turn,
            player: if m.x_to_move { "X".to_string() } else { "O".to_string() },
            move_index: m.move_index,
            row,
            col,
            subboard,
            subsquare,
            duration_ms,
            board_display,
        });
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

pub struct ReplayExporter;
