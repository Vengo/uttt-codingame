use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use crate::arena::game::Game;
use crate::arena::tournament::Tournament;
use crate::bots::{available_bots, create_bot};
use crate::game_solving::{AlphaBetaCompatible, GameMove};
use crate::ttt_trait_solving::TicTacToeEngine;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct MatchRequest {
    bot1: String,
    bot2: String,
    time_ms: Option<u64>,
}

#[derive(Deserialize)]
struct TournamentRequest {
    games: Option<usize>,
    time_ms: Option<u64>,
    bots: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct PlayMoveRequest {
    bot: String,
    moves: Vec<u8>,
    time_ms: Option<u64>,
}

#[derive(Serialize)]
struct PlayMoveResponse {
    bot_move_index: u8,
    bot_row: usize,
    bot_col: usize,
    bot_subboard: usize,
    bot_subsquare: usize,
    game_over: bool,
    winner: String,
}

pub struct ArenaServer {
    pub host: String,
    pub port: u16,
    pub legal_moves: Arc<Vec<Vec<GameMove>>>,
    pub zobrist_constants: Arc<ZobristConstants>,
}

impl ArenaServer {
    pub fn new(host: &str, port: u16) -> Self {
        println!("Initializing TicTacToeEngine for web server...");
        let engine = TicTacToeEngine::new();
        let legal_moves = engine.get_move_ordering_array();
        let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

        ArenaServer {
            host: host.to_string(),
            port,
            legal_moves: Arc::new(legal_moves),
            zobrist_constants: Arc::new(zobrist_constants),
        }
    }

    pub fn start(&self) {
        let addr = format!("{}:{}", self.host, self.port);
        let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
            panic!("Failed to bind to {}: {}", addr, e);
        });

        println!("\n=======================================================");
        println!("🚀 Ultimate Tic-Tac-Toe Web Server Running!");
        println!("=======================================================");
        println!("Local Access:    http://localhost:{}", self.port);
        println!("LAN Access:      http://192.168.50.34:{}", self.port);
        println!("Serving Replay Viewer, Live Battles & Tournament API");
        println!("Press Ctrl+C to stop.\n");

        for stream in listener.incoming() {
            if let Ok(stream) = stream {
                let legal_moves = Arc::clone(&self.legal_moves);
                let zobrist_constants = Arc::clone(&self.zobrist_constants);
                std::thread::spawn(move || {
                    Self::handle_connection(stream, &legal_moves, &zobrist_constants);
                });
            }
        }
    }

    fn handle_connection(
        mut stream: TcpStream,
        legal_moves: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) {
        let mut reader = BufReader::new(&stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
            return;
        }

        let parts: Vec<&str> = request_line.split_whitespace().collect();
        if parts.len() < 2 {
            return;
        }

        let method = parts[0];
        let path_and_query = parts[1];
        let path = path_and_query.split('?').next().unwrap_or("/");

        // Read headers to get content length if POST
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                break;
            }
            let lower = line.to_lowercase();
            if lower.starts_with("content-length:") {
                if let Some(val) = line.split(':').nth(1) {
                    content_length = val.trim().parse().unwrap_or(0);
                }
            }
        }

        if method == "OPTIONS" {
            let response = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(response.as_bytes());
            return;
        }

        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            let _ = reader.read_exact(&mut body);
        }

        // Routing
        match (method, path) {
            ("GET", "/") | ("GET", "/index.html") | ("GET", "/viewer") | ("HEAD", "/") => {
                Self::serve_html(&mut stream);
            }
            ("GET", "/api/bots") => {
                let bots = available_bots();
                let json = serde_json::to_string(&bots).unwrap();
                Self::send_json(&mut stream, 200, &json);
            }
            ("GET", "/api/tournament") => {
                let json = fs::read_to_string("tournament_results.json")
                    .or_else(|_| fs::read_to_string("uttt-codingame/tournament_results.json"))
                    .unwrap_or_else(|_| "{}".to_string());
                Self::send_json(&mut stream, 200, &json);
            }
            ("GET", "/api/profile") => {
                let json = fs::read_to_string("hardware_profile.json")
                    .or_else(|_| fs::read_to_string("uttt-codingame/hardware_profile.json"))
                    .unwrap_or_else(|_| {
                        let report = crate::profile::run_hardware_profile();
                        serde_json::to_string_pretty(&report).unwrap_or_default()
                    });
                Self::send_json(&mut stream, 200, &json);
            }
            ("GET", "/api/flamegraph") => {
                let mut bot_name = "umcts_v2".to_string();
                let mut time_ms = 1000u64;
                if let Some((_, query)) = path_and_query.split_once('?') {
                    for param in query.split('&') {
                        if let Some((k, v)) = param.split_once('=') {
                            if k == "bot" {
                                bot_name = v.to_string();
                            } else if k == "time_ms" {
                                if let Ok(ms) = v.parse::<u64>() {
                                    time_ms = ms.clamp(100, 10000);
                                }
                            }
                        }
                    }
                }
                match crate::arena::flamegraph::FlamegraphSampler::profile_bot(&bot_name, Duration::from_millis(time_ms)) {
                    Ok(svg) => Self::send_svg(&mut stream, 200, &svg),
                    Err(e) => Self::send_error(&mut stream, 500, &e),
                }
            }
            ("GET", "/api/study/status") => {
                let json = fs::read_to_string("/my-zraid/uttt/study_status.json")
                    .or_else(|_| fs::read_to_string("study_status.json"))
                    .unwrap_or_else(|_| {
                        let book = crate::opening_book::OpeningBook::new();
                        let default_json = serde_json::json!({
                            "active": false,
                            "book_entries": book.entries.len(),
                            "egtb_solved": 0,
                            "zfs_storage_avail_tb": 7.2,
                            "ram_gb": 128,
                            "gpu": "NVIDIA GeForce RTX 5060 Ti"
                        });
                        serde_json::to_string_pretty(&default_json).unwrap_or_default()
                    });
                Self::send_json(&mut stream, 200, &json);
            }
            ("GET", "/api/training/status") => {
                let json = fs::read_to_string("ml/training_status.json")
                    .or_else(|_| fs::read_to_string("uttt-codingame/ml/training_status.json"))
                    .or_else(|_| fs::read_to_string("../uttt-codingame/ml/training_status.json"))
                    .or_else(|_| fs::read_to_string("training_status.json"))
                    .unwrap_or_else(|_| {
                        let default_json = serde_json::json!({
                            "cycle": 0,
                            "total_cycles": 0,
                            "phase": "idle",
                            "phase_description": "AlphaZero Self-Play Daemon Idle",
                            "mcts_sims": 800,
                            "parallel_games": 64,
                            "champion_elo": 1500.0,
                            "champion_generation": 0,
                            "challenger_elo": 1500.0,
                            "replay_buffer_size": 0,
                            "gpu_telemetry": {
                                "device": "NVIDIA GeForce RTX 5060 Ti",
                                "vram_allocated_gb": 0.0,
                                "vram_reserved_gb": 0.0,
                                "vram_total_gb": 16.62,
                                "utilization_pct": 0.0
                            },
                            "history": [],
                            "updated_at": ""
                        });
                        serde_json::to_string_pretty(&default_json).unwrap_or_default()
                    });
                Self::send_json(&mut stream, 200, &json);
            }
            ("GET", "/api/replays") => {
                let mut replays = Vec::new();
                for entry in fs::read_dir(".").into_iter().flatten().flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                            if name.starts_with("replay") {
                                replays.push(name.to_string());
                            }
                        }
                    }
                }
                let json = serde_json::to_string(&replays).unwrap();
                Self::send_json(&mut stream, 200, &json);
            }
            ("GET", p) if p.starts_with("/api/replays/") => {
                let filename = &p["/api/replays/".len()..];
                // Sanitize filename
                if filename.contains('/') || filename.contains('\\') || filename.contains("..") {
                    Self::send_error(&mut stream, 400, "Invalid filename");
                    return;
                }
                let path1 = filename.to_string();
                let path2 = format!("uttt-codingame/{}", filename);
                if let Ok(content) = fs::read_to_string(&path1).or_else(|_| fs::read_to_string(&path2)) {
                    Self::send_json(&mut stream, 200, &content);
                } else {
                    Self::send_error(&mut stream, 404, "Replay not found");
                }
            }
            ("POST", "/api/match") => {
                if let Ok(req) = serde_json::from_slice::<MatchRequest>(&body) {
                    let time_ms = req.time_ms.unwrap_or(30);
                    let mut b1 = create_bot(&req.bot1).unwrap_or_else(|| create_bot("alphabeta").unwrap());
                    let mut b2 = create_bot(&req.bot2).unwrap_or_else(|| create_bot("heuristic_mcts").unwrap());

                    let game = Game::new(Duration::from_millis(time_ms)).with_replay(true);
                    let (_stats, replay) = game.play(b1.as_mut(), b2.as_mut(), legal_moves, zobrist_constants);

                    if let Some(r) = replay {
                        let json = serde_json::to_string(&r).unwrap();
                        let filename = format!("replay_{}_vs_{}.json", req.bot1, req.bot2);
                        let _ = fs::write(&filename, &json);
                        Self::send_json(&mut stream, 200, &json);
                    } else {
                        Self::send_error(&mut stream, 500, "Failed to record replay");
                    }
                } else {
                    Self::send_error(&mut stream, 400, "Invalid match request payload");
                }
            }
            ("POST", "/api/tournament") => {
                let req: TournamentRequest = serde_json::from_slice(&body).unwrap_or(TournamentRequest {
                    games: Some(4),
                    time_ms: Some(15),
                    bots: None,
                });
                let games = req.games.unwrap_or(4);
                let time_ms = req.time_ms.unwrap_or(15);
                let default_bots = available_bots();
                let bot_names: Vec<&str> = req
                    .bots
                    .as_ref()
                    .map(|v| v.iter().map(|s| s.as_str()).collect())
                    .unwrap_or(default_bots);

                let tournament = Tournament::new(Duration::from_millis(time_ms), games);
                let result = tournament.run(&bot_names, legal_moves, zobrist_constants);
                let json = serde_json::to_string_pretty(&result).unwrap();
                let _ = fs::write("tournament_results.json", &json);
                Self::send_json(&mut stream, 200, &json);
            }
            ("POST", "/api/play-move") => {
                if let Ok(req) = serde_json::from_slice::<PlayMoveRequest>(&body) {
                    let mut board = UltimateTicTacToeBoard::default();
                    for &m_idx in &req.moves {
                        let is_x = board.x_to_move;
                        let gm = GameMove { x_to_move: is_x, move_index: m_idx };
                        board.make_move(gm, zobrist_constants);
                    }

                    let win_state = board.get_win_state();
                    if win_state != crate::game_solving::GameResult::Unresolved {
                        let winner = match win_state {
                            crate::game_solving::GameResult::P1Win => "Player 1 (X) Wins",
                            crate::game_solving::GameResult::P2Win => "Player 2 (O) Wins",
                            _ => "Tie Game",
                        };
                        let resp = PlayMoveResponse {
                            bot_move_index: 255,
                            bot_row: 0,
                            bot_col: 0,
                            bot_subboard: 0,
                            bot_subsquare: 0,
                            game_over: true,
                            winner: winner.to_string(),
                        };
                        Self::send_json(&mut stream, 200, &serde_json::to_string(&resp).unwrap());
                        return;
                    }

                    let mut bot = create_bot(&req.bot).unwrap_or_else(|| create_bot("alphabeta").unwrap());
                    let time_ms = req.time_ms.unwrap_or(50);
                    let chosen = bot.choose_move(&board, legal_moves, zobrist_constants, Duration::from_millis(time_ms));

                    board.make_move(chosen, zobrist_constants);
                    let final_state = board.get_win_state();
                    let (game_over, winner) = match final_state {
                        crate::game_solving::GameResult::P1Win => (true, "Player 1 (X) Wins".to_string()),
                        crate::game_solving::GameResult::P2Win => (true, "Player 2 (O) Wins".to_string()),
                        crate::game_solving::GameResult::Tie => (true, "Tie Game".to_string()),
                        _ => (false, "".to_string()),
                    };

                    let parsed = crate::uttt_trait_solving::parse_index(chosen.move_index);
                    let row = (parsed.sub_board_index / 3) * 3 + (parsed.sub_square_index / 3);
                    let col = (parsed.sub_board_index % 3) * 3 + (parsed.sub_square_index % 3);

                    let resp = PlayMoveResponse {
                        bot_move_index: chosen.move_index,
                        bot_row: row,
                        bot_col: col,
                        bot_subboard: parsed.sub_board_index,
                        bot_subsquare: parsed.sub_square_index,
                        game_over,
                        winner,
                    };
                    Self::send_json(&mut stream, 200, &serde_json::to_string(&resp).unwrap());
                } else {
                    Self::send_error(&mut stream, 400, "Invalid play-move request");
                }
            }
            _ => {
                Self::send_error(&mut stream, 404, "Not Found");
            }
        }
    }

    fn serve_html(stream: &mut TcpStream) {
        let content = fs::read_to_string("replay_viewer.html")
            .or_else(|_| fs::read_to_string("../replay_viewer.html"))
            .unwrap_or_else(|_| "<h1>Replay Viewer HTML not found</h1>".to_string());

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=UTF-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
            content.len(),
            content
        );
        let _ = stream.write_all(response.as_bytes());
    }

    fn send_json(stream: &mut TcpStream, status: u16, json: &str) {
        let status_text = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            _ => "Internal Server Error",
        };
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: application/json; charset=UTF-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
            status, status_text, json.len(), json
        );
        let _ = stream.write_all(response.as_bytes());
    }

    fn send_error(stream: &mut TcpStream, status: u16, message: &str) {
        let err_json = format!("{{\"error\":\"{}\"}}", message);
        Self::send_json(stream, status, &err_json);
    }

    fn send_svg(stream: &mut TcpStream, status: u16, svg: &[u8]) {
        let status_text = match status {
            200 => "OK",
            _ => "Internal Server Error",
        };
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: image/svg+xml; charset=UTF-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
            status, status_text, svg.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(svg);
    }
}
