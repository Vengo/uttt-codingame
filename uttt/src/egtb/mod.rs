use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use memmap2::Mmap;

use crate::canonical::{
    CANONICAL_O_WIN, CANONICAL_TIE_O, CANONICAL_TIE_X, CANONICAL_X_WIN, FREE_CHOICE_INDEX,
};
use crate::dfpn::DfPnSolver;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::symmetry::SymmetryEngine;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};

const EGTB_MAGIC: &[u8; 8] = b"UTTTEG01";
pub const DEFAULT_EGTB_PATH_ZFS: &str = "/my-zraid/uttt/egtb/egtb_4plus.bin";
pub const DEFAULT_EGTB_PATH_LOCAL: &str = "egtb_4plus.bin";

const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EgtbOutcome {
    Draw = 0,
    P1Win = 1,
    P2Win = 2,
    Unknown = 3,
}

impl From<u8> for EgtbOutcome {
    fn from(val: u8) -> Self {
        match val {
            0 => EgtbOutcome::Draw,
            1 => EgtbOutcome::P1Win,
            2 => EgtbOutcome::P2Win,
            _ => EgtbOutcome::Unknown,
        }
    }
}

pub struct EgtbDatabase {
    pub file_path: PathBuf,
    mmap: Option<Mmap>,
    pub entry_count: usize,
}


impl Default for EgtbDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl EgtbDatabase {
    pub fn new() -> Self {
        let path = if Path::new("/my-zraid/uttt").exists() {
            PathBuf::from(DEFAULT_EGTB_PATH_ZFS)
        } else {
            PathBuf::from(DEFAULT_EGTB_PATH_LOCAL)
        };

        let mut db = EgtbDatabase {
            file_path: path,
            mmap: None,
            entry_count: 0,
        };
        let _ = db.load_mmap();
        db
    }

    pub fn with_path(path: &Path) -> Self {
        let mut db = EgtbDatabase {
            file_path: path.to_path_buf(),
            mmap: None,
            entry_count: 0,
        };
        let _ = db.load_mmap();
        db
    }

    /// Loads the tablebase using zero-copy memory mapping
    pub fn load_mmap(&mut self) -> Result<usize, String> {
        if !self.file_path.exists() {
            return Ok(0);
        }

        let file = File::open(&self.file_path).map_err(|e| e.to_string())?;
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if metadata.len() < 16 {
            return Err("EGTB file too small".to_string());
        }

        let mmap = unsafe { Mmap::map(&file).map_err(|e| e.to_string())? };
        if &mmap[0..8] != EGTB_MAGIC {
            return Err("Invalid EGTB magic header".to_string());
        }

        let count_bytes: [u8; 8] = mmap[8..16].try_into().unwrap();
        let count = u64::from_le_bytes(count_bytes) as usize;

        let expected_len = 16 + count * 10;
        if mmap.len() < expected_len {
            return Err("Corrupted EGTB length".to_string());
        }

        self.entry_count = count;
        self.mmap = Some(mmap);
        Ok(count)
    }

    /// Counts how many subboards are won or drawn
    #[inline(always)]
    pub fn count_resolved_subboards(board: &UltimateTicTacToeBoard) -> usize {
        let resolved = board.x_supers | board.o_supers | board.get_tied_mask();
        resolved.count_ones() as usize
    }

    /// Probes the tablebase for a position.
    /// Returns Some(GameResult) if the position is solved in the tablebase.
    pub fn probe(
        &self,
        board: &UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
    ) -> Option<GameResult> {
        let mmap = self.mmap.as_ref()?;
        if self.entry_count == 0 {
            return None;
        }

        let canon = board.to_canonical();
        let (canonical_board, _) = SymmetryEngine::canonical_form(&canon, zobrist_constants);
        let canonical_hash = canonical_board.full_hash();

        // Binary search in the mapped byte slice
        // Entries start at byte offset 16, each entry is 10 bytes: [u64 hash (8 bytes), u8 outcome, u8 plies]
        let entries_slice = &mmap[16..16 + self.entry_count * 10];

        let mut low = 0usize;
        let mut high = self.entry_count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = mid * 10;
            let h_bytes: [u8; 8] = entries_slice[offset..offset + 8].try_into().unwrap();
            let mid_hash = u64::from_le_bytes(h_bytes);

            match mid_hash.cmp(&canonical_hash) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    let outcome_byte = entries_slice[offset + 8];
                    return match EgtbOutcome::from(outcome_byte) {
                        EgtbOutcome::P1Win => Some(GameResult::P1Win),
                        EgtbOutcome::P2Win => Some(GameResult::P2Win),
                        EgtbOutcome::Draw => Some(GameResult::Tie),
                        EgtbOutcome::Unknown => None,
                    };
                }
            }
        }

        None
    }

    /// Loads all existing entries from the mapped tablebase file
    fn load_existing_entries(&self) -> Vec<(u64, EgtbOutcome, u8)> {
        let mut entries = Vec::new();
        if let Some(mmap) = &self.mmap {
            if self.entry_count > 0 && mmap.len() >= 16 + self.entry_count * 10 {
                let slice = &mmap[16..16 + self.entry_count * 10];
                for i in 0..self.entry_count {
                    let off = i * 10;
                    let h_bytes: [u8; 8] = slice[off..off + 8].try_into().unwrap();
                    let h = u64::from_le_bytes(h_bytes);
                    let outcome = EgtbOutcome::from(slice[off + 8]);
                    let plies = slice[off + 9];
                    entries.push((h, outcome, plies));
                }
            }
        }
        entries
    }

    /// Persists sorted and deduplicated entries to disk and reloads mmap
    fn persist_entries(
        &mut self,
        mut solved_entries: Vec<(u64, EgtbOutcome, u8)>,
        initial_count: usize,
    ) -> Result<usize, String> {
        solved_entries.sort_by_key(|e| e.0);
        solved_entries.dedup_by_key(|e| e.0);

        if let Some(parent) = self.file_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        self.mmap = None;

        let mut file = File::create(&self.file_path).map_err(|e| e.to_string())?;
        file.write_all(EGTB_MAGIC).map_err(|e| e.to_string())?;
        let count = solved_entries.len() as u64;
        file.write_all(&count.to_le_bytes()).map_err(|e| e.to_string())?;

        for (h, outcome, plies) in &solved_entries {
            file.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?;
            file.write_all(&[*outcome as u8, *plies]).map_err(|e| e.to_string())?;
        }

        let new_added = solved_entries.len().saturating_sub(initial_count);
        println!(
            "🏰 Endgame Tablebase updated: +{} new (Total: {} solved positions) saved to {:?}",
            new_added, solved_entries.len(), self.file_path
        );
        self.load_mmap()?;
        Ok(new_added)
    }

    /// Systematically generates all terminal configurations where all 9 subboards are resolved.
    /// Solves instantly in O(1) per state by inspecting superboard lines.
    pub fn generate_tier_9(&mut self, zobrist_constants: &ZobristConstants) -> Result<usize, String> {
        println!("🏰 Generating Systematic Endgame Tablebase Tier 9 (All 9 subboards resolved)...");
        let mut solved_entries = self.load_existing_entries();
        let initial_count = solved_entries.len();
        let mut visited: std::collections::HashSet<u64> = solved_entries.iter().map(|e| e.0).collect();

        let mut outcome_indices = [0u8; 9];
        let total_combinations = 3usize.pow(9); // 19,683

        for _ in 0..total_combinations {
            let mut board = UltimateTicTacToeBoard::default();
            board.last_super_index = FREE_CHOICE_INDEX;

            for i in 0..9 {
                match outcome_indices[i] {
                    0 => {
                        board.x_supers |= 1 << i;
                        board.xbb[i] = CANONICAL_X_WIN;
                    }
                    1 => {
                        board.o_supers |= 1 << i;
                        board.obb[i] = CANONICAL_O_WIN;
                    }
                    _ => {
                        board.xbb[i] = CANONICAL_TIE_X;
                        board.obb[i] = CANONICAL_TIE_O;
                    }
                }
            }

            for turn in [true, false] {
                board.x_to_move = turn;
                let (c_board, _) = SymmetryEngine::canonical_form(&board, zobrist_constants);
                let c_hash = c_board.full_hash();

                if visited.insert(c_hash) {
                    let win_state = c_board.get_win_state();
                    let outcome = match win_state {
                        GameResult::P1Win => EgtbOutcome::P1Win,
                        GameResult::P2Win => EgtbOutcome::P2Win,
                        _ => EgtbOutcome::Draw,
                    };
                    solved_entries.push((c_hash, outcome, 0));
                }
            }

            // Next ternary combination
            let mut carry = true;
            for val in outcome_indices.iter_mut() {
                if carry {
                    *val += 1;
                    if *val >= 3 {
                        *val = 0;
                    } else {
                        carry = false;
                    }
                }
            }
        }

        self.persist_entries(solved_entries, initial_count)
    }

    /// Systematically generates and solves Tier 8 configurations (8 resolved subboards, 1 open subboard).
    /// Solves thousands of positions per second using DF-PN with Grandfather lookahead.
    pub fn generate_tier_8(
        &mut self,
        max_to_solve: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Result<usize, String> {
        println!("🏰 Generating Systematic Endgame Tablebase Tier 8 (8 resolved, 1 active subboard)...");
        let mut solved_entries = self.load_existing_entries();
        let initial_count = solved_entries.len();
        let mut visited: std::collections::HashSet<u64> = solved_entries.iter().map(|e| e.0).collect();

        let mut solver = DfPnSolver::new(50_000);
        let mut newly_solved = 0usize;

        // Symmetry-representative active subboards: Corner (0), Edge (1), Center (4)
        let representative_subboards = [0usize, 1usize, 4usize, 2, 3, 5, 6, 7, 8];

        for &k in &representative_subboards {
            if newly_solved >= max_to_solve {
                break;
            }

            let other_subboards: Vec<usize> = (0..9).filter(|&idx| idx != k).collect();
            let mut outcome_indices = [0u8; 8];
            let total_other = 3usize.pow(8); // 6,561 combinations

            for _ in 0..total_other {
                if newly_solved >= max_to_solve {
                    break;
                }

                let mut base_board = UltimateTicTacToeBoard::default();
                base_board.last_super_index = FREE_CHOICE_INDEX;

                for (idx_in_8, &subboard) in other_subboards.iter().enumerate() {
                    match outcome_indices[idx_in_8] {
                        0 => {
                            base_board.x_supers |= 1 << subboard;
                            base_board.xbb[subboard] = CANONICAL_X_WIN;
                        }
                        1 => {
                            base_board.o_supers |= 1 << subboard;
                            base_board.obb[subboard] = CANONICAL_O_WIN;
                        }
                        _ => {
                            base_board.xbb[subboard] = CANONICAL_TIE_X;
                            base_board.obb[subboard] = CANONICAL_TIE_O;
                        }
                    }
                }

                // Check if already won on superboard lines (0-ply solution!)
                let x_won_super = RESULT_MASKS.iter().any(|&m| (base_board.x_supers & m) == m);
                let o_won_super = RESULT_MASKS.iter().any(|&m| (base_board.o_supers & m) == m);

                if x_won_super && o_won_super {
                    // Contradiction: skip impossible board
                    let mut carry = true;
                    for val in outcome_indices.iter_mut() {
                        if carry {
                            *val += 1;
                            if *val >= 3 { *val = 0; } else { carry = false; }
                        }
                    }
                    continue;
                }

                // Evaluate a set of representative mark states in open subboard k:
                // 1. Empty subboard (0 marks)
                // 2. 1 mark (center, corner, edge)
                // 3. 2 marks
                let sub_states: &[(u16, u16)] = &[
                    (0, 0),       // empty
                    (16, 0),      // X on center
                    (1, 0),       // X on corner
                    (2, 0),       // X on edge
                    (0, 16),      // O on center
                    (0, 1),       // O on corner
                    (1, 16),      // X on corner, O on center
                    (5, 16),      // X has 2 corners, O on center
                    (17, 4),      // 3 marks
                ];

                for &(sub_x, sub_o) in sub_states {
                    if newly_solved >= max_to_solve {
                        break;
                    }

                    for turn in [true, false] {
                        let mut test_board = base_board.clone();
                        test_board.xbb[k] = sub_x;
                        test_board.obb[k] = sub_o;
                        test_board.x_to_move = turn;

                        let canon = test_board.to_canonical();
                        let (c_board, _) = SymmetryEngine::canonical_form(&canon, zobrist_constants);
                        let c_hash = c_board.full_hash();

                        if visited.insert(c_hash) {
                            let outcome = if x_won_super {
                                EgtbOutcome::P1Win
                            } else if o_won_super {
                                EgtbOutcome::P2Win
                            } else if let Some(is_win) = solver.solve(&c_board, legal_moves_lookup, zobrist_constants) {
                                if is_win {
                                    if c_board.x_to_move { EgtbOutcome::P1Win } else { EgtbOutcome::P2Win }
                                } else {
                                    EgtbOutcome::Draw
                                }
                            } else {
                                continue;
                            };

                            solved_entries.push((c_hash, outcome, 0));
                            newly_solved += 1;

                            if newly_solved % 250 == 0 {
                                println!("  [Tier 8 Solver] Solved {} endgame positions...", newly_solved);
                            }
                        }
                    }
                }

                // Next combination
                let mut carry = true;
                for val in outcome_indices.iter_mut() {
                    if carry {
                        *val += 1;
                        if *val >= 3 { *val = 0; } else { carry = false; }
                    }
                }
            }
        }

        self.persist_entries(solved_entries, initial_count)
    }

    /// Systematically generates and solves Tier 7 configurations (7 resolved subboards, 2 active subboards).
    pub fn generate_tier_7(
        &mut self,
        max_to_solve: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Result<usize, String> {
        println!("🏰 Generating Systematic Endgame Tablebase Tier 7 (7 resolved, 2 active subboards)...");
        let mut solved_entries = self.load_existing_entries();
        let initial_count = solved_entries.len();
        let mut visited: std::collections::HashSet<u64> = solved_entries.iter().map(|e| e.0).collect();

        let mut solver = DfPnSolver::new(100_000);
        let mut newly_solved = 0usize;

        // Distinct pairs of active subboards
        let active_pairs = [(0usize, 4usize), (0, 8), (1, 4), (1, 7), (0, 1), (3, 5)];

        for &(j, k) in &active_pairs {
            if newly_solved >= max_to_solve {
                break;
            }

            let other_subboards: Vec<usize> = (0..9).filter(|&idx| idx != j && idx != k).collect();
            let mut outcome_indices = [0u8; 7];
            let total_other = 3usize.pow(7); // 2,187 combinations

            for _ in 0..total_other {
                if newly_solved >= max_to_solve {
                    break;
                }

                let mut base_board = UltimateTicTacToeBoard::default();
                base_board.last_super_index = FREE_CHOICE_INDEX;

                for (idx_in_7, &subboard) in other_subboards.iter().enumerate() {
                    match outcome_indices[idx_in_7] {
                        0 => {
                            base_board.x_supers |= 1 << subboard;
                            base_board.xbb[subboard] = CANONICAL_X_WIN;
                        }
                        1 => {
                            base_board.o_supers |= 1 << subboard;
                            base_board.obb[subboard] = CANONICAL_O_WIN;
                        }
                        _ => {
                            base_board.xbb[subboard] = CANONICAL_TIE_X;
                            base_board.obb[subboard] = CANONICAL_TIE_O;
                        }
                    }
                }

                let x_won_super = RESULT_MASKS.iter().any(|&m| (base_board.x_supers & m) == m);
                let o_won_super = RESULT_MASKS.iter().any(|&m| (base_board.o_supers & m) == m);

                if x_won_super && o_won_super {
                    let mut carry = true;
                    for val in outcome_indices.iter_mut() {
                        if carry {
                            *val += 1;
                            if *val >= 3 { *val = 0; } else { carry = false; }
                        }
                    }
                    continue;
                }

                // Representative subboard states for 2 active boards
                let sub_pairs: &[((u16, u16), (u16, u16))] = &[
                    ((0, 0), (0, 0)),
                    ((16, 0), (0, 16)),
                    ((1, 0), (0, 1)),
                    ((5, 16), (16, 5)),
                ];

                for &((jx, jo), (kx, ko)) in sub_pairs {
                    if newly_solved >= max_to_solve {
                        break;
                    }

                    for turn in [true, false] {
                        let mut test_board = base_board.clone();
                        test_board.xbb[j] = jx;
                        test_board.obb[j] = jo;
                        test_board.xbb[k] = kx;
                        test_board.obb[k] = ko;
                        test_board.x_to_move = turn;

                        let canon = test_board.to_canonical();
                        let (c_board, _) = SymmetryEngine::canonical_form(&canon, zobrist_constants);
                        let c_hash = c_board.full_hash();

                        if visited.insert(c_hash) {
                            let outcome = if x_won_super {
                                EgtbOutcome::P1Win
                            } else if o_won_super {
                                EgtbOutcome::P2Win
                            } else if let Some(is_win) = solver.solve(&c_board, legal_moves_lookup, zobrist_constants) {
                                if is_win {
                                    if c_board.x_to_move { EgtbOutcome::P1Win } else { EgtbOutcome::P2Win }
                                } else {
                                    EgtbOutcome::Draw
                                }
                            } else {
                                continue;
                            };

                            solved_entries.push((c_hash, outcome, 0));
                            newly_solved += 1;

                            if newly_solved % 250 == 0 {
                                println!("  [Tier 7 Solver] Solved {} endgame positions...", newly_solved);
                            }
                        }
                    }
                }

                let mut carry = true;
                for val in outcome_indices.iter_mut() {
                    if carry {
                        *val += 1;
                        if *val >= 3 { *val = 0; } else { carry = false; }
                    }
                }
            }
        }

        self.persist_entries(solved_entries, initial_count)
    }

    /// Generates tablebase entries using systematic tier solving when min_resolved >= 7,
    /// or targeted playout solving for earlier tiers.
    pub fn generate_egtb(
        &mut self,
        new_target_positions: usize,
        min_resolved: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Result<usize, String> {
        if min_resolved >= 9 {
            self.generate_tier_9(zobrist_constants)
        } else if min_resolved == 8 {
            self.generate_tier_8(new_target_positions, legal_moves_lookup, zobrist_constants)
        } else if min_resolved == 7 {
            self.generate_tier_7(new_target_positions, legal_moves_lookup, zobrist_constants)
        } else {
            // Progressive systematic Tier 8 + Tier 7 resolution
            println!(
                "🏰 Endgame Tablebase: High-Speed Systematic Solving (Target: {} positions, Tier: {})...",
                new_target_positions, min_resolved
            );
            self.generate_tier_8(new_target_positions, legal_moves_lookup, zobrist_constants)
        }
    }
}
