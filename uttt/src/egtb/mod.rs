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
pub const DEFAULT_EGTB_DIR_ZFS: &str = "/my-zraid/uttt/egtb";
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
    pub base_dir: PathBuf,
    pub file_path: PathBuf,
    pub is_custom: bool,
    pub entry_count: usize,
    pub monolithic_mmap: Option<Mmap>,
    pub monolithic_count: usize,
    pub tier_mmaps: [Option<Mmap>; 10],
    pub tier_counts: [usize; 10],
    pub tier_paths: [PathBuf; 10],
    pub zobrist_constants: ZobristConstants,
}

impl Default for EgtbDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl EgtbDatabase {
    pub fn new() -> Self {
        let base_dir = if Path::new(DEFAULT_EGTB_DIR_ZFS).exists() || Path::new("/my-zraid/uttt").exists() {
            PathBuf::from(DEFAULT_EGTB_DIR_ZFS)
        } else if Path::new("egtb").is_dir() {
            PathBuf::from("egtb")
        } else {
            PathBuf::from(".")
        };

        let file_path = base_dir.join(DEFAULT_EGTB_PATH_LOCAL);
        let tier_paths = std::array::from_fn(|k| base_dir.join(format!("egtb_tier{}.bin", k)));

        let mut db = EgtbDatabase {
            base_dir,
            file_path,
            is_custom: false,
            entry_count: 0,
            monolithic_mmap: None,
            monolithic_count: 0,
            tier_mmaps: Default::default(),
            tier_counts: [0; 10],
            tier_paths,
            zobrist_constants: UltimateTicTacToeBoard::get_zobrist_constants(),
        };
        let _ = db.load_mmap();
        db
    }

    pub fn with_path(path: &Path) -> Self {
        if path.is_dir() {
            return Self::with_dir(path);
        }

        let mono = path.to_path_buf();
        let base = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("egtb");
        let tiers = if stem == "egtb_4plus" || stem == "egtb" {
            std::array::from_fn(|k| base.join(format!("egtb_tier{}.bin", k)))
        } else {
            std::array::from_fn(|k| base.join(format!("{}_tier{}.bin", stem, k)))
        };

        let mut db = EgtbDatabase {
            base_dir: base,
            file_path: mono,
            is_custom: true,
            entry_count: 0,
            monolithic_mmap: None,
            monolithic_count: 0,
            tier_mmaps: Default::default(),
            tier_counts: [0; 10],
            tier_paths: tiers,
            zobrist_constants: UltimateTicTacToeBoard::get_zobrist_constants(),
        };
        let _ = db.load_mmap();
        db
    }

    pub fn with_dir(dir: &Path) -> Self {
        let _ = std::fs::create_dir_all(dir);
        let base = dir.to_path_buf();
        let mono = base.join(DEFAULT_EGTB_PATH_LOCAL);
        let tiers = std::array::from_fn(|k| base.join(format!("egtb_tier{}.bin", k)));

        let mut db = EgtbDatabase {
            base_dir: base,
            file_path: mono,
            is_custom: true,
            entry_count: 0,
            monolithic_mmap: None,
            monolithic_count: 0,
            tier_mmaps: Default::default(),
            tier_counts: [0; 10],
            tier_paths: tiers,
            zobrist_constants: UltimateTicTacToeBoard::get_zobrist_constants(),
        };
        let _ = db.load_mmap();
        db
    }

    pub fn tier_path(&self, tier: usize) -> PathBuf {
        if tier < self.tier_paths.len() && !self.tier_paths[tier].as_os_str().is_empty() {
            self.tier_paths[tier].clone()
        } else {
            self.base_dir.join(format!("egtb_tier{}.bin", tier))
        }
    }

    fn find_tier_path(&self, k: usize) -> Option<PathBuf> {
        let mut candidates = vec![
            self.tier_path(k),
            self.base_dir.join(format!("egtb_tier{}.bin", k)),
        ];
        if !self.is_custom {
            candidates.push(PathBuf::from(format!("{}/egtb_tier{}.bin", DEFAULT_EGTB_DIR_ZFS, k)));
            candidates.push(PathBuf::from(format!("egtb/egtb_tier{}.bin", k)));
            candidates.push(PathBuf::from(format!("egtb_tier{}.bin", k)));
        }
        candidates.into_iter().find(|p| p.exists())
    }

    fn find_monolithic_path(&self) -> Option<PathBuf> {
        let mut candidates = vec![
            self.file_path.clone(),
            self.base_dir.join(DEFAULT_EGTB_PATH_LOCAL),
        ];
        if !self.is_custom {
            candidates.push(PathBuf::from(DEFAULT_EGTB_PATH_ZFS));
            candidates.push(PathBuf::from(format!("egtb/{}", DEFAULT_EGTB_PATH_LOCAL)));
            candidates.push(PathBuf::from(DEFAULT_EGTB_PATH_LOCAL));
        }
        candidates.into_iter().find(|p| p.exists())
    }

    /// Loads the partitioned tier tablebases and fallback monolithic tablebase using zero-copy memory mapping
    pub fn load_mmap(&mut self) -> Result<usize, String> {
        let mut total_count = 0;

        // 1. Load Tier files (0..=9, particularly 4..=9)
        for k in 0..=9 {
            self.tier_mmaps[k] = None;
            self.tier_counts[k] = 0;

            if let Some(p) = self.find_tier_path(k) {
                if let Ok(file) = File::open(&p) {
                    if let Ok(metadata) = file.metadata() {
                        if metadata.len() >= 16 {
                            if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                                if mmap.len() >= 16 && &mmap[0..8] == EGTB_MAGIC {
                                    let count_bytes: [u8; 8] = mmap[8..16].try_into().unwrap();
                                    let count = u64::from_le_bytes(count_bytes) as usize;
                                    if mmap.len() >= 16 + count * 10 {
                                        total_count += count;
                                        self.tier_counts[k] = count;
                                        self.tier_mmaps[k] = Some(mmap);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Load monolithic fallback tablebase
        self.monolithic_mmap = None;
        self.monolithic_count = 0;

        if let Some(p) = self.find_monolithic_path() {
            if let Ok(file) = File::open(&p) {
                if let Ok(metadata) = file.metadata() {
                    if metadata.len() >= 16 {
                        if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                            if mmap.len() >= 16 && &mmap[0..8] == EGTB_MAGIC {
                                let count_bytes: [u8; 8] = mmap[8..16].try_into().unwrap();
                                let count = u64::from_le_bytes(count_bytes) as usize;
                                if mmap.len() >= 16 + count * 10 {
                                    self.monolithic_count = count;
                                    self.monolithic_mmap = Some(mmap);
                                    if total_count == 0 {
                                        total_count = count;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.entry_count = total_count;
        Ok(total_count)
    }

    /// Counts how many subboards are won or drawn
    #[inline(always)]
    pub fn count_resolved_subboards(board: &UltimateTicTacToeBoard) -> usize {
        board.get_resolved_count()
    }

    /// Probes the tablebase for a position using the cached Zobrist constants.
    /// Selects the partitioned tier tablebase directly by resolved subboard count k.
    #[inline]
    pub fn probe(&self, board: &UltimateTicTacToeBoard) -> Option<GameResult> {
        self.probe_with_constants(board, &self.zobrist_constants)
    }

    /// Probes the tablebase with explicit Zobrist constants.
    /// Probes directly by resolved subboard count k = board.get_resolved_count(),
    /// eliminating probes into unrelated tiers and supporting scaling past 10 GB.
    pub fn probe_with_constants(
        &self,
        board: &UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
    ) -> Option<GameResult> {
        if self.entry_count == 0 {
            return None;
        }

        let k = board.get_resolved_count();
        let canon = board.to_canonical();
        let (canonical_board, _) = SymmetryEngine::canonical_form(&canon, zobrist_constants);
        let canonical_hash = canonical_board.full_hash();

        // 1. Probe the specific tier table directly by resolved subboard count k
        if k < self.tier_mmaps.len() {
            if let Some(ref mmap) = self.tier_mmaps[k] {
                let count = self.tier_counts[k];
                if count > 0 && mmap.len() >= 16 + count * 10 {
                    if let Some(res) = Self::binary_search_mmap(mmap, count, canonical_hash) {
                        return Some(res);
                    }
                }
            }
        }

        // 2. Fallback to monolithic tablebase if tier table is not yet partitioned or missed
        if let Some(ref mmap) = self.monolithic_mmap {
            let count = self.monolithic_count;
            if count > 0 && mmap.len() >= 16 + count * 10 {
                if let Some(res) = Self::binary_search_mmap(mmap, count, canonical_hash) {
                    return Some(res);
                }
            }
        }

        None
    }

    fn binary_search_mmap(mmap: &Mmap, count: usize, target_hash: u64) -> Option<GameResult> {
        let entries_slice = &mmap[16..16 + count * 10];
        let mut low = 0usize;
        let mut high = count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = mid * 10;
            let h_bytes: [u8; 8] = entries_slice[offset..offset + 8].try_into().unwrap();
            let mid_hash = u64::from_le_bytes(h_bytes);

            match mid_hash.cmp(&target_hash) {
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

    fn load_entries_from_mmap(&self, mmap_opt: Option<&Mmap>, count: usize) -> Vec<(u64, EgtbOutcome, u8)> {
        let mut entries = Vec::new();
        if let Some(mmap) = mmap_opt {
            if count > 0 && mmap.len() >= 16 + count * 10 {
                let slice = &mmap[16..16 + count * 10];
                for i in 0..count {
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

    fn load_existing_tier_entries(&self, tier: usize) -> Vec<(u64, EgtbOutcome, u8)> {
        if tier < self.tier_mmaps.len() {
            if let Some(ref mmap) = self.tier_mmaps[tier] {
                return self.load_entries_from_mmap(Some(mmap), self.tier_counts[tier]);
            }
        }
        let tier_file = self.tier_path(tier);
        if tier_file.exists() {
            if let Ok(file) = File::open(&tier_file) {
                if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                    if mmap.len() >= 16 && &mmap[0..8] == EGTB_MAGIC {
                        let count_bytes: [u8; 8] = mmap[8..16].try_into().unwrap();
                        let count = u64::from_le_bytes(count_bytes) as usize;
                        return self.load_entries_from_mmap(Some(&mmap), count);
                    }
                }
            }
        }
        Vec::new()
    }

    /// Persists sorted and deduplicated entries to tier file and monolithic fallback file, then reloads mmaps
    fn persist_tier_entries(
        &mut self,
        tier: usize,
        mut solved_entries: Vec<(u64, EgtbOutcome, u8)>,
        initial_count: usize,
    ) -> Result<usize, String> {
        solved_entries.sort_by_key(|e| e.0);
        solved_entries.dedup_by_key(|e| e.0);

        let tier_file = self.tier_path(tier);
        if let Some(parent) = tier_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Drop current mmap for this tier before writing
        self.tier_mmaps[tier] = None;

        let mut file = File::create(&tier_file).map_err(|e| e.to_string())?;
        file.write_all(EGTB_MAGIC).map_err(|e| e.to_string())?;
        let count = solved_entries.len() as u64;
        file.write_all(&count.to_le_bytes()).map_err(|e| e.to_string())?;

        for (h, outcome, plies) in &solved_entries {
            file.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?;
            file.write_all(&[*outcome as u8, *plies]).map_err(|e| e.to_string())?;
        }

        let new_added = solved_entries.len().saturating_sub(initial_count);
        println!(
            "🏰 Endgame Tablebase Tier {} updated: +{} new (Total: {} solved positions) saved to {:?}",
            tier, new_added, solved_entries.len(), tier_file
        );

        // Also update monolithic fallback file if configured and not identical to tier_file
        if self.file_path != tier_file {
            let _ = self.update_monolithic_with_entries(&solved_entries);
        }

        let _ = self.load_mmap();
        Ok(new_added)
    }

    fn update_monolithic_with_entries(&mut self, new_entries: &[(u64, EgtbOutcome, u8)]) -> Result<(), String> {
        let mut all_entries = self.load_entries_from_mmap(self.monolithic_mmap.as_ref(), self.monolithic_count);
        all_entries.extend_from_slice(new_entries);
        all_entries.sort_by_key(|e| e.0);
        all_entries.dedup_by_key(|e| e.0);

        if let Some(parent) = self.file_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        self.monolithic_mmap = None;

        let mut file = File::create(&self.file_path).map_err(|e| e.to_string())?;
        file.write_all(EGTB_MAGIC).map_err(|e| e.to_string())?;
        let count = all_entries.len() as u64;
        file.write_all(&count.to_le_bytes()).map_err(|e| e.to_string())?;

        for (h, outcome, plies) in &all_entries {
            file.write_all(&h.to_le_bytes()).map_err(|e| e.to_string())?;
            file.write_all(&[*outcome as u8, *plies]).map_err(|e| e.to_string())?;
        }

        Ok(())
    }

    /// Systematically generates all terminal configurations where all 9 subboards are resolved.
    /// Solves instantly in O(1) per state by inspecting superboard lines and persists to egtb_tier9.bin.
    pub fn generate_tier_9(&mut self, zobrist_constants: &ZobristConstants) -> Result<usize, String> {
        println!("🏰 Generating Systematic Endgame Tablebase Tier 9 (All 9 subboards resolved)...");
        let mut solved_entries = self.load_existing_tier_entries(9);
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

        self.persist_tier_entries(9, solved_entries, initial_count)
    }

    /// Systematically generates and solves Tier 8 configurations (8 resolved subboards, 1 active subboard).
    /// Solves positions using DF-PN with Grandfather lookahead and persists to egtb_tier8.bin.
    pub fn generate_tier_8(
        &mut self,
        max_to_solve: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Result<usize, String> {
        println!("🏰 Generating Systematic Endgame Tablebase Tier 8 (8 resolved, 1 active subboard)...");
        let mut solved_entries = self.load_existing_tier_entries(8);
        let initial_count = solved_entries.len();
        let mut visited: std::collections::HashSet<u64> = solved_entries.iter().map(|e| e.0).collect();

        let mut solver = DfPnSolver::new(50_000);
        let mut newly_solved = 0usize;

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
                    let mut carry = true;
                    for val in outcome_indices.iter_mut() {
                        if carry {
                            *val += 1;
                            if *val >= 3 { *val = 0; } else { carry = false; }
                        }
                    }
                    continue;
                }

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

                let mut carry = true;
                for val in outcome_indices.iter_mut() {
                    if carry {
                        *val += 1;
                        if *val >= 3 { *val = 0; } else { carry = false; }
                    }
                }
            }
        }

        self.persist_tier_entries(8, solved_entries, initial_count)
    }

    /// Systematically generates and solves Tier 7 configurations (7 resolved subboards, 2 active subboards).
    /// Iterates across all 36 active subboard pairs and representative mark assignments,
    /// solving positions with DF-PN and persisting to egtb_tier7.bin.
    pub fn generate_tier_7(
        &mut self,
        max_to_solve: usize,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Result<usize, String> {
        println!("🏰 Generating Systematic Endgame Tablebase Tier 7 (7 resolved, 2 active subboards)...");
        let mut solved_entries = self.load_existing_tier_entries(7);
        let initial_count = solved_entries.len();
        let mut visited: std::collections::HashSet<u64> = solved_entries.iter().map(|e| e.0).collect();

        let mut solver = DfPnSolver::new(50_000);
        let mut newly_solved = 0usize;

        // All 36 pairs of active subboards
        let mut active_pairs = Vec::with_capacity(36);
        for j in 0..8 {
            for k in (j + 1)..9 {
                active_pairs.push((j, k));
            }
        }

        for (j, k) in active_pairs {
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

                let sub_pairs: &[((u16, u16), (u16, u16))] = &[
                    ((0, 0), (0, 0)),
                    ((16, 0), (0, 16)),
                    ((1, 0), (0, 1)),
                    ((2, 0), (0, 2)),
                    ((5, 16), (16, 5)),
                    ((1, 16), (0, 0)),
                    ((0, 0), (1, 16)),
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

        self.persist_tier_entries(7, solved_entries, initial_count)
    }

    /// Generates tablebase entries writing to tier-specific files alongside monolithic fallback
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
            println!(
                "🏰 Endgame Tablebase: High-Speed Systematic Solving (Target: {} positions, Tier: {})...",
                new_target_positions, min_resolved
            );
            self.generate_tier_8(new_target_positions, legal_moves_lookup, zobrist_constants)
        }
    }
}
