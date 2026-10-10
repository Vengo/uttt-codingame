use crate::arena::elo::EloCalculator;
use crate::arena::match_runner::{MatchResult, MatchRunner};
use crate::bots::create_bot;
use crate::game_solving::GameMove;
use crate::uttt_trait_solving::ZobristConstants;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub rank: usize,
    pub bot_name: String,
    pub elo: f64,
    pub points: f64,
    pub games_played: usize,
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    pub win_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentResult {
    pub winner: String,
    pub leaderboard: Vec<LeaderboardEntry>,
    pub matches: Vec<MatchResult>,
}

pub struct Tournament {
    pub time_limit_per_move: Duration,
    pub games_per_pairing: usize,
}

impl Tournament {
    pub fn new(time_limit_per_move: Duration, games_per_pairing: usize) -> Self {
        Tournament {
            time_limit_per_move,
            games_per_pairing,
        }
    }

    pub fn run(
        &self,
        bot_names: &[&str],
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> TournamentResult {
        let mut elo_calc = EloCalculator::new(32.0);
        let mut points: HashMap<String, f64> = HashMap::new();
        let mut wins: HashMap<String, usize> = HashMap::new();
        let mut losses: HashMap<String, usize> = HashMap::new();
        let mut draws: HashMap<String, usize> = HashMap::new();
        let mut games_played: HashMap<String, usize> = HashMap::new();

        for name in bot_names {
            elo_calc.set_rating(name, 1500.0);
            points.insert(name.to_string(), 0.0);
            wins.insert(name.to_string(), 0);
            losses.insert(name.to_string(), 0);
            draws.insert(name.to_string(), 0);
            games_played.insert(name.to_string(), 0);
        }

        let match_runner = MatchRunner::new(self.time_limit_per_move);
        let mut all_match_results = Vec::new();

        println!("\n=======================================================");
        println!("           STARTING ROUND-ROBIN TOURNAMENT             ");
        println!("=======================================================");
        println!("Participants: {:?}", bot_names);
        println!("Games per pairing: {}", self.games_per_pairing);
        println!("Time limit per move: {:?}\n", self.time_limit_per_move);

        for i in 0..bot_names.len() {
            for j in (i + 1)..bot_names.len() {
                let name1 = bot_names[i];
                let name2 = bot_names[j];

                print!("Match: {:<16} vs {:<16} ... ", name1, name2);
                let mut bot1 = create_bot(name1).unwrap_or_else(|| panic!("Bot {} not found", name1));
                let mut bot2 = create_bot(name2).unwrap_or_else(|| panic!("Bot {} not found", name2));

                let match_res = match_runner.run_match(
                    bot1.as_mut(),
                    bot2.as_mut(),
                    self.games_per_pairing,
                    legal_moves_lookup,
                    zobrist_constants,
                    Some(&mut elo_calc),
                );

                println!(
                    "Result: {}-{}-{} ({}: {:.1}% / {}: {:.1}%)",
                    match_res.bot1_wins,
                    match_res.bot2_wins,
                    match_res.draws,
                    name1,
                    match_res.bot1_winrate,
                    name2,
                    match_res.bot2_winrate,
                );

                // Update point tallies
                *points.get_mut(name1).unwrap() += match_res.bot1_wins as f64 + 0.5 * match_res.draws as f64;
                *points.get_mut(name2).unwrap() += match_res.bot2_wins as f64 + 0.5 * match_res.draws as f64;

                *wins.get_mut(name1).unwrap() += match_res.bot1_wins;
                *losses.get_mut(name1).unwrap() += match_res.bot2_wins;
                *draws.get_mut(name1).unwrap() += match_res.draws;
                *games_played.get_mut(name1).unwrap() += self.games_per_pairing;

                *wins.get_mut(name2).unwrap() += match_res.bot2_wins;
                *losses.get_mut(name2).unwrap() += match_res.bot1_wins;
                *draws.get_mut(name2).unwrap() += match_res.draws;
                *games_played.get_mut(name2).unwrap() += self.games_per_pairing;

                all_match_results.push(match_res);
            }
        }

        // Build leaderboard sorted by points descending, then Elo descending
        let mut entries: Vec<LeaderboardEntry> = bot_names
            .iter()
            .map(|&name| {
                let bot = create_bot(name).unwrap();
                let display_name = bot.name();
                let gp = *games_played.get(name).unwrap();
                let w = *wins.get(name).unwrap();
                let l = *losses.get(name).unwrap();
                let d = *draws.get(name).unwrap();
                let p = *points.get(name).unwrap();
                let elo = elo_calc.get_rating(display_name);
                let win_pct = if gp > 0 { (p / gp as f64) * 100.0 } else { 0.0 };

                LeaderboardEntry {
                    rank: 0,
                    bot_name: format!("{} ({})", name, display_name),
                    elo,
                    points: p,
                    games_played: gp,
                    wins: w,
                    losses: l,
                    draws: d,
                    win_percentage: win_pct,
                }
            })
            .collect();

        entries.sort_by(|a, b| {
            b.points
                .partial_cmp(&a.points)
                .unwrap()
                .then(b.elo.partial_cmp(&a.elo).unwrap())
        });

        for (idx, entry) in entries.iter_mut().enumerate() {
            entry.rank = idx + 1;
        }

        let winner = entries
            .first()
            .map(|e| e.bot_name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        Self::print_scoreboard(&entries, &winner);

        TournamentResult {
            winner,
            leaderboard: entries,
            matches: all_match_results,
        }
    }

    pub fn print_scoreboard(entries: &[LeaderboardEntry], winner: &str) {
        println!("\n=========================================================================================");
        println!("                                FINAL TOURNAMENT STANDINGS                               ");
        println!("=========================================================================================");
        println!(
            "{:<5} | {:<18} | {:<7} | {:<6} | {:<5} | {:<5} | {:<5} | {:<8}",
            "Rank", "Bot Name", "Elo", "Points", "Wins", "Losses", "Draws", "Score %"
        );
        println!("------+--------------------+---------+--------+-------+-------+-------+----------");
        for e in entries {
            println!(
                "{:<5} | {:<18} | {:<7.1} | {:<6.1} | {:<5} | {:<5} | {:<5} | {:<7.1}%",
                e.rank, e.bot_name, e.elo, e.points, e.wins, e.losses, e.draws, e.win_percentage
            );
        }
        println!("=========================================================================================");
        println!("🏆 TOURNAMENT WINNER: {} 🏆", winner);
        println!("=========================================================================================\n");
    }
}
