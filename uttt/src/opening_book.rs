use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::bots::{Bot, UmctsV2Bot};

use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::symmetry::SymmetryEngine;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};

const BOOK_MAGIC: &[u8; 8] = b"UTTTOB01";
pub const DEFAULT_BOOK_PATH_ZFS: &str = "/my-zraid/uttt/book.bin";
pub const DEFAULT_BOOK_PATH_LOCAL: &str = "book.bin";

#[derive(Clone, Debug)]
pub struct BookEntry {
    pub canonical_hash: u64,
    pub canonical_move_index: u8,
    pub visits: u32,
}

pub struct OpeningBook {
    pub entries: HashMap<u64, (u8, u32)>,
    pub file_path: PathBuf,
}

impl Default for OpeningBook {
    fn default() -> Self {
        Self::new()
    }
}

impl OpeningBook {
    pub fn new() -> Self {
        let path = if Path::new("/my-zraid/uttt").exists() {
            PathBuf::from(DEFAULT_BOOK_PATH_ZFS)
        } else {
            PathBuf::from(DEFAULT_BOOK_PATH_LOCAL)
        };

        let mut book = OpeningBook {
            entries: HashMap::new(),
            file_path: path,
        };

        // Attempt to auto-load existing book if present
        let _ = book.load();
        book
    }

    pub fn with_path(path: &Path) -> Self {
        let mut book = OpeningBook {
            entries: HashMap::new(),
            file_path: path.to_path_buf(),
        };
        let _ = book.load();
        book
    }

    /// Looks up the best opening move for the board, using D4 symmetry reduction.
    /// Returns Some(GameMove) in the actual board orientation if present.
    pub fn lookup(
        &self,
        board: &UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
    ) -> Option<GameMove> {
        if self.entries.is_empty() {
            return None;
        }

        let (c_hash, sym) = SymmetryEngine::canonical_hash(board, zobrist_constants);
        if let Some(&(c_move_idx, _visits)) = self.entries.get(&c_hash) {
            let actual_move_idx = sym.inverse().transform_move_index(c_move_idx);
            Some(GameMove {
                x_to_move: board.x_to_move,
                move_index: actual_move_idx,
            })
        } else {
            None
        }
    }

    /// Records or updates a canonical entry
    pub fn record(
        &mut self,
        board: &UltimateTicTacToeBoard,
        best_move: GameMove,
        visits: u32,
        zobrist_constants: &ZobristConstants,
    ) {
        let (canonical_board, sym) = SymmetryEngine::canonical_form(board, zobrist_constants);
        let canonical_hash = canonical_board.full_hash();
        let canonical_move_idx = sym.transform_move_index(best_move.move_index);

        self.entries
            .entry(canonical_hash)
            .and_modify(|(m, v)| {
                if visits > *v {
                    *m = canonical_move_idx;
                    *v = visits;
                }
            })
            .or_insert((canonical_move_idx, visits));
    }

    /// Loads binary book from self.file_path
    pub fn load(&mut self) -> Result<usize, String> {
        if !self.file_path.exists() {
            return Ok(0);
        }

        let mut file = File::open(&self.file_path).map_err(|e| e.to_string())?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != BOOK_MAGIC {
            return Err("Invalid book magic header".to_string());
        }

        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf).map_err(|e| e.to_string())?;
        let count = u32::from_le_bytes(count_buf) as usize;

        let mut pad = [0u8; 4];
        file.read_exact(&mut pad).map_err(|e| e.to_string())?;

        self.entries.clear();
        for _ in 0..count {
            let mut h_buf = [0u8; 8];
            file.read_exact(&mut h_buf).map_err(|e| e.to_string())?;
            let hash = u64::from_le_bytes(h_buf);

            let mut m_buf = [0u8; 1];
            file.read_exact(&mut m_buf).map_err(|e| e.to_string())?;
            let move_idx = m_buf[0];

            let mut p3 = [0u8; 3];
            file.read_exact(&mut p3).map_err(|e| e.to_string())?;

            let mut v_buf = [0u8; 4];
            file.read_exact(&mut v_buf).map_err(|e| e.to_string())?;
            let visits = u32::from_le_bytes(v_buf);

            self.entries.insert(hash, (move_idx, visits));
        }

        Ok(self.entries.len())
    }

    /// Saves binary book to self.file_path
    pub fn save(&self) -> Result<(), String> {
        if let Some(parent) = self.file_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let mut file = File::create(&self.file_path).map_err(|e| e.to_string())?;
        file.write_all(BOOK_MAGIC).map_err(|e| e.to_string())?;

        let count = self.entries.len() as u32;
        file.write_all(&count.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&[0u8; 4]).map_err(|e| e.to_string())?;

        for (&hash, &(move_idx, visits)) in &self.entries {
            file.write_all(&hash.to_le_bytes()).map_err(|e| e.to_string())?;
            file.write_all(&[move_idx]).map_err(|e| e.to_string())?;
            file.write_all(&[0u8; 3]).map_err(|e| e.to_string())?;
            file.write_all(&visits.to_le_bytes()).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Discovers unexpanded frontier positions branching off current book entries.
    pub fn find_frontier(
        &self,
        max_frontier: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Vec<UltimateTicTacToeBoard> {
        let mut frontier = Vec::new();
        let root = UltimateTicTacToeBoard::default();

        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();

        let (c_root, _) = SymmetryEngine::canonical_form(&root, zobrist_constants);
        queue.push_back((c_root.clone(), 0usize));
        visited.insert(c_root.full_hash());


        while let Some((board, depth)) = queue.pop_front() {
            if board.get_win_state() != GameResult::Unresolved {
                continue;
            }

            let (c_board, _) = SymmetryEngine::canonical_form(&board, zobrist_constants);
            let c_hash = c_board.full_hash();

            if let Some(&(c_move_idx, _)) = self.entries.get(&c_hash) {
                // In book: follow principal variation and opponent alternative replies
                if depth < 8 {
                    // 1. Follow book move
                    let mut child = c_board.clone();
                    let book_move = GameMove {
                        x_to_move: c_board.x_to_move,
                        move_index: c_move_idx,
                    };
                    child.make_move(book_move, zobrist_constants);
                    let (canon_child, _) = SymmetryEngine::canonical_form(&child, zobrist_constants);
                    if visited.insert(canon_child.full_hash()) {
                        queue.push_back((canon_child, depth + 1));
                    }

                    // 2. Also follow top 2 opponent alternative replies
                    let moves = c_board.get_legal_moves(legal_moves_lookup);
                    for m in moves.into_iter().take(3) {
                        if m.move_index != c_move_idx {
                            let mut alt_child = c_board.clone();
                            alt_child.make_move(m, zobrist_constants);
                            let (canon_alt, _) = SymmetryEngine::canonical_form(&alt_child, zobrist_constants);
                            if visited.insert(canon_alt.full_hash()) {
                                queue.push_back((canon_alt, depth + 1));
                            }
                        }
                    }
                }
            } else {
                // Not in book: target frontier position found!
                frontier.push(c_board);
                if frontier.len() >= max_frontier {
                    break;
                }
            }
        }

        if frontier.is_empty() {
            frontier.push(UltimateTicTacToeBoard::default());
        }

        frontier
    }

    /// Deepens and expands the opening book by searching frontier positions with UMCTS-V2
    /// and harvesting all well-visited positions (visits >= min_visits) across the search tree.
    pub fn deepen_book(
        &mut self,
        max_frontier_searches: usize,
        search_time_per_pos: Duration,
        min_visits: u32,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> usize {
        let initial_count = self.entries.len();
        let frontier = self.find_frontier(max_frontier_searches, legal_moves_lookup, zobrist_constants);
        let mut bot = UmctsV2Bot::new();

        println!(
            "📚 Deepening Opening Book ({} frontier positions queued, min_visits={}, time_per_pos={:?})...",
            frontier.len(),
            min_visits,
            search_time_per_pos
        );

        for (idx, target_board) in frontier.into_iter().enumerate() {
            bot.reset();
            let best_move = bot.choose_move(
                &target_board,
                legal_moves_lookup,
                zobrist_constants,
                search_time_per_pos,
            );

            // Always record the explicitly searched frontier node
            let (c_target, sym_target) = SymmetryEngine::canonical_form(&target_board, zobrist_constants);
            let c_hash_target = c_target.full_hash();
            let c_move_target = sym_target.transform_move_index(best_move.move_index);
            let root_visits = 100u32;
            let mut added_this_pos = 0;
            if self.entries.insert(c_hash_target, (c_move_target, root_visits)).is_none() {
                added_this_pos += 1;
            }

            // Harvest descendant positions in the MCTS tree that received >= min_visits and show dominance
            let harvested = bot.harvest_tree(&target_board, min_visits, zobrist_constants);


            for (pos, best_m, visits) in harvested {
                let (c_board, sym) = SymmetryEngine::canonical_form(&pos, zobrist_constants);
                let c_hash = c_board.full_hash();
                let c_move = sym.transform_move_index(best_m.move_index);

                use std::collections::hash_map::Entry;
                match self.entries.entry(c_hash) {
                    Entry::Vacant(e) => {
                        e.insert((c_move, visits));
                        added_this_pos += 1;
                    }
                    Entry::Occupied(mut e) => {
                        if visits > e.get().1 {
                            e.insert((c_move, visits));
                        }
                    }
                }
            }

            println!(
                "  [Frontier {}] Harvested +{} new positions (Total in book: {})",
                idx + 1,
                added_this_pos,
                self.entries.len()
            );
        }

        let new_total = self.entries.len();
        let added = new_total.saturating_sub(initial_count);
        let _ = self.save();
        println!(
            "📚 Opening Book updated: +{} new entries (Total: {}) saved to {:?}",
            added, new_total, self.file_path
        );
        added
    }

    /// Generates opening book entries using frontier deepening with tree harvesting.
    pub fn generate_book(
        &mut self,
        max_depth: usize,
        search_time_per_pos: Duration,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) {
        let frontier_budget = max_depth.max(2) * 2;
        let min_visits = 25u32;
        self.deepen_book(
            frontier_budget,
            search_time_per_pos,
            min_visits,
            legal_moves_lookup,
            zobrist_constants,
        );
    }
}
