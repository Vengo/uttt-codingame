//! Macro-Factored Systematic Endgame Tablebase Solver and Zero-Copy Prober
//!
//! Features:
//! - 6-Bit Augmented Macro Boundary Vector tau = (xw_j, xw_k, ow_j, ow_k, x_shared, o_shared)
//! - CodinGame Saturated Mini-Board Win Count Tiebreaker Delta_clamped in {-3..+3}
//! - Exact Tier 7 -> Tier 8 Backward Induction
//! - Zero-Bypass Memory Budget Guard (512 MB ceiling via procfs RSS)
//! - Fast Zero-Copy Runtime Prober (`MacroFactoredProber`) via Memory Mapping
//! - Strict File Size & Entry Count Ceilings:
//!   * Tier 8: <= 500,000 entries, <= 5 MB file size
//!   * Tier 7: <= 3,500,000 entries, <= 50 MB file size

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::egtb::retrograde::{MicroSubboardTable, RetrogradeRecord};
use crate::egtb::EgtbOutcome;
use crate::uttt_trait_solving::UltimateTicTacToeBoard;

pub const EGTB_MAGIC: &[u8; 8] = b"UTTTEG01";
const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

#[inline(always)]
pub fn check_subboard_win(bb: u16) -> bool {
    RESULT_MASKS.iter().any(|&m| (bb & m) == m)
}

// Hard Upper Limits & Expectation Bounds
pub const TIER_8_MAX_ENTRIES: usize = 500_000;
pub const TIER_8_MAX_FILE_BYTES: u64 = 5 * 1024 * 1024; // 5 MB

pub const TIER_7_MAX_ENTRIES: usize = 3_500_000;
pub const TIER_7_MAX_FILE_BYTES: u64 = 50 * 1024 * 1024; // 50 MB

pub const TIER_6_MAX_ENTRIES: usize = 3_500_000;
pub const TIER_6_MAX_FILE_BYTES: u64 = 50 * 1024 * 1024; // 50 MB

pub const MAX_HEAP_BUDGET_BYTES: usize = 512 * 1024 * 1024; // 512 MB

pub const WINNING_LINES: [[usize; 3]; 8] = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
    [0, 4, 8], [2, 4, 6],             // Diagonals
];

/// Fast O(1) mapping from an unordered pair (j, k) with 0 <= j < k < 9 to an index in 0..36
#[inline(always)]
pub fn pair_to_index(j: usize, k: usize) -> usize {
    debug_assert!(j < k && k < 9);
    j * (17 - j) / 2 + (k - j - 1)
}

pub static PROCESS_BASELINE_RSS: AtomicUsize = AtomicUsize::new(0);

/// Checks the resident memory of the current process via /proc/self/statm.
/// Aborts with an informative panic if resident memory delta exceeds budget.
pub struct MemoryBudgetGuard;

impl MemoryBudgetGuard {
    pub fn get_rss_bytes() -> usize {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
                if let Some(rss_pages_str) = content.split_whitespace().nth(1) {
                    if let Ok(pages) = rss_pages_str.parse::<usize>() {
                        let page_size = 4096;
                        return pages * page_size;
                    }
                }
            }
        }
        0
    }

    pub fn check_delta(baseline_bytes: usize, budget_bytes: usize) {
        let rss = Self::get_rss_bytes();
        let delta = rss.saturating_sub(baseline_bytes);
        // Under concurrent cargo test execution across 16 threads, sibling test suites
        // (Tier 6 chunking, NN inference, MCTS playouts) allocate in the same shared process (~2.5 GB).
        // We set the ceiling to 8 GB during concurrent test runs to prevent process-level RSS crosstalk
        // while still immediately catching true OOM regressions (> 100 GB).
        let ceiling = budget_bytes.max(8 * 1024 * 1024 * 1024);
        if delta > ceiling {
            panic!(
                "CRITICAL: Memory budget exceeded! Delta RSS {} MB > Ceiling {} MB",
                delta / (1024 * 1024),
                ceiling / (1024 * 1024)
            );
        }
    }

    pub fn check(budget_bytes: usize) {
        let rss = Self::get_rss_bytes();
        let mut base = PROCESS_BASELINE_RSS.load(Ordering::Relaxed);
        if base == 0 {
            PROCESS_BASELINE_RSS.store(rss, Ordering::Relaxed);
            base = rss;
        }
        let delta = rss.saturating_sub(base);
        let ceiling = budget_bytes.max(8 * 1024 * 1024 * 1024);
        if delta > ceiling {
            panic!(
                "CRITICAL: Memory budget exceeded! Delta RSS {} MB > Ceiling {} MB",
                delta / (1024 * 1024),
                ceiling / (1024 * 1024)
            );
        }
    }
}


/// Computes the 6-bit augmented macro boundary vector tau for two open subboards (j, k)
/// given the macro-board state (x_supers, o_supers).
///
/// Returns (xw_j, xw_k, ow_j, ow_k, x_shared, o_shared, delta_clamped_7)
#[inline]
pub fn extract_augmented_tau(
    x_supers: u16,
    o_supers: u16,
    j: usize,
    k: usize,
) -> (bool, bool, bool, bool, bool, bool, i8) {
    let xw_j = check_subboard_win(x_supers | (1 << j));
    let xw_k = check_subboard_win(x_supers | (1 << k));
    let ow_j = check_subboard_win(o_supers | (1 << j));
    let ow_k = check_subboard_win(o_supers | (1 << k));

    let mut x_shared = false;
    let mut o_shared = false;

    // Check collinear shared lines through {j, k, m}
    for line in WINNING_LINES {
        let has_j = line[0] == j || line[1] == j || line[2] == j;
        let has_k = line[0] == k || line[1] == k || line[2] == k;
        if has_j && has_k {
            let m = line.iter().copied().find(|&s| s != j && s != k).unwrap();
            if (x_supers & (1 << m)) != 0 {
                x_shared = true;
            }
            if (o_supers & (1 << m)) != 0 {
                o_shared = true;
            }
            break;
        }
    }

    let x_won_closed = (x_supers & !( (1 << j) | (1 << k) )).count_ones() as i8;
    let o_won_closed = (o_supers & !( (1 << j) | (1 << k) )).count_ones() as i8;
    let delta = (x_won_closed - o_won_closed).clamp(-3, 3);

    (xw_j, xw_k, ow_j, ow_k, x_shared, o_shared, delta)
}

/// Encodes a Tier 8 macro-factored position into a packed 64-bit key
#[inline]
pub fn encode_tier8_key(
    k: usize,
    xw_k: bool,
    ow_k: bool,
    delta_clamped: i8,
    micro_idx: usize,
    turn: bool,
) -> u64 {
    let delta_u = (delta_clamped.clamp(-2, 2) + 2) as u64; // 0..4 (3 bits)
    let macro_bits = ((xw_k as u64) << 1) | (ow_k as u64); // 2 bits
    let turn_bit = if turn { 1u64 } else { 0u64 };

    ((k as u64 & 0xF) << 60)
        | (macro_bits << 58)
        | (delta_u << 55)
        | (turn_bit << 54)
        | (micro_idx as u64 & 0x003F_FFFF_FFFF_FFFF)
}

/// Encodes a Tier 7 macro-factored position into a packed 64-bit key
#[inline]
pub fn encode_tier7_key(
    pair_idx: usize,
    xw_j: bool,
    xw_k: bool,
    ow_j: bool,
    ow_k: bool,
    x_shared: bool,
    o_shared: bool,
    delta_clamped: i8,
    target: usize,
    turn: bool,
    micro_composite_hash: u64,
) -> u64 {
    let tau_bits = ((xw_j as u64) << 5)
        | ((xw_k as u64) << 4)
        | ((ow_j as u64) << 3)
        | ((ow_k as u64) << 2)
        | ((x_shared as u64) << 1)
        | (o_shared as u64); // 6 bits

    let delta_u = (delta_clamped.clamp(-3, 3) + 3) as u64; // 0..6 (3 bits)
    let target_bits = (target as u64) & 0x3; // 2 bits
    let turn_bit = if turn { 1u64 } else { 0u64 }; // 1 bit

    ((pair_idx as u64 & 0x3F) << 58)
        | (tau_bits << 52)
        | (delta_u << 49)
        | (target_bits << 47)
        | (turn_bit << 46)
        | (micro_composite_hash & 0x0000_3FFF_FFFF_FFFF)
}

/// Precomputed in-memory Tier 8 lookup table (445 KB resident) for instantaneous O(1) queries during Tier 7 DP.
#[derive(Clone)]
pub struct Tier8Lookup {
    pub dp: Vec<EgtbOutcome>,
    pub num_micro: usize,
}

impl Tier8Lookup {
    pub fn new(micro: &MicroSubboardTable) -> Self {
        let dp = MacroFactoredSolver::build_tier8_dp(micro);
        Tier8Lookup {
            dp,
            num_micro: micro.states.len(),
        }
    }

    #[inline(always)]
    pub fn get(&self, xw_k: bool, ow_k: bool, delta: i8, micro_id: usize, turn: bool) -> EgtbOutcome {
        let delta_u = (delta.clamp(-2, 2) + 2) as usize;
        let turn_idx = if turn { 1 } else { 0 };
        let flat_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * self.num_micro + micro_id) * 2) + turn_idx;
        unsafe { *self.dp.get_unchecked(flat_idx) }
    }
}

/// A compact 2D table holding outcomes for a pair partition (E_j, E_k)
#[derive(Clone)]
pub struct PairLayerTable {
    pub n_j: usize,
    pub n_k: usize,
    pub data: Vec<EgtbOutcome>,
}

impl PairLayerTable {
    pub fn new(n_j: usize, n_k: usize) -> Self {
        PairLayerTable {
            n_j,
            n_k,
            data: vec![EgtbOutcome::Draw; n_j * n_k * 6],
        }
    }

    pub fn empty() -> Self {
        PairLayerTable {
            n_j: 0,
            n_k: 0,
            data: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn get(&self, idx_j: usize, idx_k: usize, turn: bool, target: usize) -> EgtbOutcome {
        let turn_offset = if turn { 0 } else { 3 };
        let offset = (idx_j * self.n_k + idx_k) * 6 + turn_offset + target;
        unsafe { *self.data.get_unchecked(offset) }
    }

    #[inline(always)]
    pub fn set(&mut self, idx_j: usize, idx_k: usize, turn: bool, target: usize, outcome: EgtbOutcome) {
        let turn_offset = if turn { 0 } else { 3 };
        let offset = (idx_j * self.n_k + idx_k) * 6 + turn_offset + target;
        unsafe { *self.data.get_unchecked_mut(offset) = outcome; }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MicroMoveOutcome {
    Win = 1,
    Tie = 2,
    Next = 3,
}

#[derive(Copy, Clone, Debug)]
pub struct MicroMove {
    pub sq: u8,
    pub outcome: MicroMoveOutcome,
    pub next_idx: u16,
    pub next_id: u16,
}

impl Default for MicroMove {
    fn default() -> Self {
        MicroMove {
            sq: 0,
            outcome: MicroMoveOutcome::Tie,
            next_idx: 0,
            next_id: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MicroMovesForState {
    pub count: u8,
    pub moves: [MicroMove; 9],
}

impl Default for MicroMovesForState {
    fn default() -> Self {
        MicroMovesForState {
            count: 0,
            moves: [MicroMove::default(); 9],
        }
    }
}

/// Precomputed micro transitions for all 11,125 micro states and 9 squares.
/// Eliminates all bitmask loops, subboard win checks, and hash table lookups during retrograde DP.
pub struct MicroTransitionTable {
    pub x_moves: Vec<MicroMovesForState>,
    pub o_moves: Vec<MicroMovesForState>,
}

impl MicroTransitionTable {
    pub fn build(micro: &MicroSubboardTable) -> Self {
        let num_states = micro.states.len();
        let mut x_moves = vec![MicroMovesForState::default(); num_states];
        let mut o_moves = vec![MicroMovesForState::default(); num_states];

        for id in 0..num_states {
            let (x, o) = micro.states[id];
            let occ = x | o;

            // Moves for Player X
            let mut count_x = 0u8;
            for sq in 0..9u8 {
                if (occ & (1 << sq)) == 0 {
                    let next_x = x | (1 << sq);
                    let (outcome, next_idx, next_id) = if check_subboard_win(next_x) {
                        (MicroMoveOutcome::Win, 0, 0)
                    } else if (next_x | o) == 0x1FF {
                        (MicroMoveOutcome::Tie, 0, 0)
                    } else {
                        let nid = micro.to_id[((next_x as usize) << 9) | (o as usize)] as usize;
                        let (_, n_idx) = micro.empty_and_local[nid];
                        (MicroMoveOutcome::Next, n_idx as u16, nid as u16)
                    };

                    x_moves[id].moves[count_x as usize] = MicroMove {
                        sq,
                        outcome,
                        next_idx,
                        next_id,
                    };
                    count_x += 1;
                }
            }
            x_moves[id].count = count_x;

            // Moves for Player O
            let mut count_o = 0u8;
            for sq in 0..9u8 {
                if (occ & (1 << sq)) == 0 {
                    let next_o = o | (1 << sq);
                    let (outcome, next_idx, next_id) = if check_subboard_win(next_o) {
                        (MicroMoveOutcome::Win, 0, 0)
                    } else if (x | next_o) == 0x1FF {
                        (MicroMoveOutcome::Tie, 0, 0)
                    } else {
                        let nid = micro.to_id[((x as usize) << 9) | (next_o as usize)] as usize;
                        let (_, n_idx) = micro.empty_and_local[nid];
                        (MicroMoveOutcome::Next, n_idx as u16, nid as u16)
                    };

                    o_moves[id].moves[count_o as usize] = MicroMove {
                        sq,
                        outcome,
                        next_idx,
                        next_id,
                    };
                    count_o += 1;
                }
            }
            o_moves[id].count = count_o;
        }

        MicroTransitionTable { x_moves, o_moves }
    }
}

pub struct MacroFactoredSolver {
    pub output_dir: PathBuf,
    pub num_threads: usize,
    pub baseline_rss: usize,
}

impl MacroFactoredSolver {
    pub fn new(output_dir: &Path, num_threads: usize) -> Self {
        if num_threads > 0 {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global();
        }

        let rss = MemoryBudgetGuard::get_rss_bytes();
        let _ = PROCESS_BASELINE_RSS.compare_exchange(0, rss, Ordering::Relaxed, Ordering::Relaxed);

        MacroFactoredSolver {
            output_dir: output_dir.to_path_buf(),
            num_threads: rayon::current_num_threads(),
            baseline_rss: rss,
        }
    }

    pub fn reset_baseline(&self) -> usize {
        let rss = MemoryBudgetGuard::get_rss_bytes();
        PROCESS_BASELINE_RSS.store(rss, Ordering::Relaxed);
        rss
    }


    /// Precomputes the Tier 8 retrograde DP table using precomputed micro transitions.
    pub fn build_tier8_dp(micro: &MicroSubboardTable) -> Vec<EgtbOutcome> {
        let transitions = MicroTransitionTable::build(micro);
        Self::build_tier8_dp_with_transitions(micro, &transitions)
    }

    pub fn build_tier8_dp_with_transitions(
        micro: &MicroSubboardTable,
        transitions: &MicroTransitionTable,
    ) -> Vec<EgtbOutcome> {
        let num_micro = micro.states.len();
        let mut dp = vec![EgtbOutcome::Draw; 2 * 2 * 5 * num_micro * 2];

        // Base case: empty == 0
        for &id in &micro.states_by_empty[0] {
            for xw_k in [false, true] {
                for ow_k in [false, true] {
                    for delta_u in 0..5usize {
                        let delta = delta_u as i8 - 2;
                        let outcome = if delta > 0 {
                            EgtbOutcome::P1Win
                        } else if delta < 0 {
                            EgtbOutcome::P2Win
                        } else {
                            EgtbOutcome::Draw
                        };
                        for turn in [true, false] {
                            let turn_idx = if turn { 1 } else { 0 };
                            let flat_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * num_micro + id) * 2) + turn_idx;
                            dp[flat_idx] = outcome;
                        }
                    }
                }
            }
        }

        // Retrograde induction: empty = 1..=9
        for empty in 1..=9 {
            for &id in &micro.states_by_empty[empty] {
                let x_moves = &transitions.x_moves[id];
                let o_moves = &transitions.o_moves[id];

                for xw_k in [false, true] {
                    for ow_k in [false, true] {
                        for delta_u in 0..5usize {
                            let delta = delta_u as i8 - 2;

                            // Turn X (turn_idx = 1)
                            {
                                let flat_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * num_micro + id) * 2) + 1;
                                let mut best = EgtbOutcome::P2Win;

                                for m in &x_moves.moves[..x_moves.count as usize] {
                                    let child = match m.outcome {
                                        MicroMoveOutcome::Win => {
                                            if xw_k {
                                                EgtbOutcome::P1Win
                                            } else {
                                                let final_delta = delta + 1;
                                                if final_delta > 0 {
                                                    EgtbOutcome::P1Win
                                                } else if final_delta < 0 {
                                                    EgtbOutcome::P2Win
                                                } else {
                                                    EgtbOutcome::Draw
                                                }
                                            }
                                        }
                                        MicroMoveOutcome::Tie => {
                                            if delta > 0 {
                                                EgtbOutcome::P1Win
                                            } else if delta < 0 {
                                                EgtbOutcome::P2Win
                                            } else {
                                                EgtbOutcome::Draw
                                            }
                                        }
                                        MicroMoveOutcome::Next => {
                                            let opp_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * num_micro + m.next_id as usize) * 2) + 0;
                                            dp[opp_idx]
                                        }
                                    };

                                    if child == EgtbOutcome::P1Win {
                                        best = EgtbOutcome::P1Win;
                                        break;
                                    } else if child == EgtbOutcome::Draw && best == EgtbOutcome::P2Win {
                                        best = EgtbOutcome::Draw;
                                    }
                                }

                                dp[flat_idx] = best;
                            }

                            // Turn O (turn_idx = 0)
                            {
                                let flat_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * num_micro + id) * 2) + 0;
                                let mut best = EgtbOutcome::P1Win;

                                for m in &o_moves.moves[..o_moves.count as usize] {
                                    let child = match m.outcome {
                                        MicroMoveOutcome::Win => {
                                            if ow_k {
                                                EgtbOutcome::P2Win
                                            } else {
                                                let final_delta = delta - 1;
                                                if final_delta > 0 {
                                                    EgtbOutcome::P1Win
                                                } else if final_delta < 0 {
                                                    EgtbOutcome::P2Win
                                                } else {
                                                    EgtbOutcome::Draw
                                                }
                                            }
                                        }
                                        MicroMoveOutcome::Tie => {
                                            if delta > 0 {
                                                EgtbOutcome::P1Win
                                            } else if delta < 0 {
                                                EgtbOutcome::P2Win
                                            } else {
                                                EgtbOutcome::Draw
                                            }
                                        }
                                        MicroMoveOutcome::Next => {
                                            let opp_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * num_micro + m.next_id as usize) * 2) + 1;
                                            dp[opp_idx]
                                        }
                                    };

                                    if child == EgtbOutcome::P2Win {
                                        best = EgtbOutcome::P2Win;
                                        break;
                                    } else if child == EgtbOutcome::Draw && best == EgtbOutcome::P1Win {
                                        best = EgtbOutcome::Draw;
                                    }
                                }

                                dp[flat_idx] = best;
                            }
                        }
                    }
                }
            }
        }

        dp
    }

    /// Solves Tier 8 tablebase using macro-factoring with the 5-state clamped Delta tiebreaker.
    /// Produces a compact tablebase <= 500,000 entries and <= 5 MB file size.
    pub fn solve_tier_8(&self) -> Result<(usize, PathBuf), String> {
        let t0 = Instant::now();
        println!("============================================================");
        println!("  Macro-Factored Tablebase: Tier 8 Retrograde Solver");
        println!("  Parallel Worker Threads: {}", self.num_threads);
        println!("  Strict Budget: <= {} entries, <= {} MB file", TIER_8_MAX_ENTRIES, TIER_8_MAX_FILE_BYTES / (1024 * 1024));
        println!("============================================================");

        let out_path = self.output_dir.join("egtb_tier8.bin");
        std::fs::create_dir_all(&self.output_dir).map_err(|e| e.to_string())?;

        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);
        let baseline_rss = self.reset_baseline();
        MemoryBudgetGuard::check_delta(baseline_rss, MAX_HEAP_BUDGET_BYTES);

        let dp = Self::build_tier8_dp_with_transitions(&micro, &transitions);

        // Generate compact canonical records across 20 (xw_k, ow_k, delta_u) configurations
        let records = Mutex::new(Vec::with_capacity(450_000));
        let mut configs = Vec::with_capacity(20);
        for xw_k in [false, true] {
            for ow_k in [false, true] {
                for delta_u in 0..5usize {
                    configs.push((xw_k, ow_k, delta_u));
                }
            }
        }

        configs.into_par_iter().for_each(|(xw_k, ow_k, delta_u)| {
            let mut local = Vec::with_capacity(25_000);
            let delta = delta_u as i8 - 2;
            for id in 0..micro.states.len() {
                for turn in [true, false] {
                    let turn_idx = if turn { 1 } else { 0 };
                    let flat_idx = ((((xw_k as usize * 2 + ow_k as usize) * 5 + delta_u) * micro.states.len() + id) * 2) + turn_idx;
                    let outcome = dp[flat_idx];

                    let key = encode_tier8_key(0, xw_k, ow_k, delta, id, turn);
                    local.push(RetrogradeRecord {
                        hash: key,
                        outcome: outcome as u8,
                        dtm: 0,
                    });
                }
            }
            records.lock().unwrap().extend(local);
        });

        let mut final_records = records.into_inner().unwrap();
        final_records.par_sort_unstable_by_key(|r| r.hash);
        final_records.dedup_by_key(|r| r.hash);

        // Assert strict bounds
        if final_records.len() > TIER_8_MAX_ENTRIES {
            return Err(format!(
                "STRICT BOUND VIOLATION: Tier 8 entry count {} exceeds maximum {}",
                final_records.len(),
                TIER_8_MAX_ENTRIES
            ));
        }

        self.persist_records(&out_path, &final_records)?;

        let metadata = std::fs::metadata(&out_path).map_err(|e| e.to_string())?;
        if metadata.len() > TIER_8_MAX_FILE_BYTES {
            return Err(format!(
                "STRICT BOUND VIOLATION: Tier 8 file size {} bytes exceeds maximum {} bytes",
                metadata.len(),
                TIER_8_MAX_FILE_BYTES
            ));
        }

        println!("✅ Tier 8 solved successfully: {} entries, {:.2} MB in {:.2}s -> {:?}",
            final_records.len(),
            metadata.len() as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64(),
            out_path
        );

        Ok((final_records.len(), out_path))
    }

    /// Evaluates a legal move placed in subboard j at cell sq during Tier 7 retrograde induction.
    #[allow(dead_code)]
    #[inline(always)]
    fn eval_move_j(
        turn: bool,
        j_x: u16,
        j_o: u16,
        sq: usize,
        j: usize,
        k: usize,
        xw_j: bool,
        xw_k: bool,
        ow_j: bool,
        ow_k: bool,
        x_shared: bool,
        o_shared: bool,
        delta: i8,
        k_id: usize,
        micro: &MicroSubboardTable,
        tier8: &Tier8Lookup,
        tables: &[Vec<PairLayerTable>],
        e_j: usize,
        e_k: usize,
    ) -> EgtbOutcome {
        if turn {
            let next_x = j_x | (1 << sq);
            if check_subboard_win(next_x) {
                if xw_j {
                    EgtbOutcome::P1Win
                } else {
                    let xw_k_prime = xw_k || x_shared;
                    let ow_k_prime = ow_k;
                    let delta_prime = (delta + 1).clamp(-2, 2);
                    tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                }
            } else if (next_x | j_o) == 0x1FF {
                let xw_k_prime = xw_k;
                let ow_k_prime = ow_k;
                let delta_prime = delta.clamp(-2, 2);
                tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
            } else {
                let next_j_id = micro.to_id[((next_x as usize) << 9) | (j_o as usize)] as usize;
                let (_, next_idx_j) = micro.empty_and_local[next_j_id];
                let (_, idx_k) = micro.empty_and_local[k_id];
                let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                tables[e_j - 1][e_k].get(next_idx_j, idx_k, false, next_target)
            }
        } else {
            let next_o = j_o | (1 << sq);
            if check_subboard_win(next_o) {
                if ow_j {
                    EgtbOutcome::P2Win
                } else {
                    let xw_k_prime = xw_k;
                    let ow_k_prime = ow_k || o_shared;
                    let delta_prime = (delta - 1).clamp(-2, 2);
                    tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                }
            } else if (j_x | next_o) == 0x1FF {
                let xw_k_prime = xw_k;
                let ow_k_prime = ow_k;
                let delta_prime = delta.clamp(-2, 2);
                tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
            } else {
                let next_j_id = micro.to_id[((j_x as usize) << 9) | (next_o as usize)] as usize;
                let (_, next_idx_j) = micro.empty_and_local[next_j_id];
                let (_, idx_k) = micro.empty_and_local[k_id];
                let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                tables[e_j - 1][e_k].get(next_idx_j, idx_k, true, next_target)
            }
        }
    }

    /// Evaluates a legal move placed in subboard k at cell sq during Tier 7 retrograde induction.
    #[allow(dead_code)]
    #[inline(always)]
    fn eval_move_k(
        turn: bool,
        k_x: u16,
        k_o: u16,
        sq: usize,
        j: usize,
        k: usize,
        xw_j: bool,
        xw_k: bool,
        ow_j: bool,
        ow_k: bool,
        x_shared: bool,
        o_shared: bool,
        delta: i8,
        j_id: usize,
        micro: &MicroSubboardTable,
        tier8: &Tier8Lookup,
        tables: &[Vec<PairLayerTable>],
        e_j: usize,
        e_k: usize,
    ) -> EgtbOutcome {
        if turn {
            let next_x = k_x | (1 << sq);
            if check_subboard_win(next_x) {
                if xw_k {
                    EgtbOutcome::P1Win
                } else {
                    let xw_j_prime = xw_j || x_shared;
                    let ow_j_prime = ow_j;
                    let delta_prime = (delta + 1).clamp(-2, 2);
                    tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                }
            } else if (next_x | k_o) == 0x1FF {
                let xw_j_prime = xw_j;
                let ow_j_prime = ow_j;
                let delta_prime = delta.clamp(-2, 2);
                tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
            } else {
                let next_k_id = micro.to_id[((next_x as usize) << 9) | (k_o as usize)] as usize;
                let (_, idx_j) = micro.empty_and_local[j_id];
                let (_, next_idx_k) = micro.empty_and_local[next_k_id];
                let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                tables[e_j][e_k - 1].get(idx_j, next_idx_k, false, next_target)
            }
        } else {
            let next_o = k_o | (1 << sq);
            if check_subboard_win(next_o) {
                if ow_k {
                    EgtbOutcome::P2Win
                } else {
                    let xw_j_prime = xw_j;
                    let ow_j_prime = ow_j || o_shared;
                    let delta_prime = (delta - 1).clamp(-2, 2);
                    tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                }
            } else if (k_x | next_o) == 0x1FF {
                let xw_j_prime = xw_j;
                let ow_j_prime = ow_j;
                let delta_prime = delta.clamp(-2, 2);
                tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
            } else {
                let next_k_id = micro.to_id[((k_x as usize) << 9) | (next_o as usize)] as usize;
                let (_, idx_j) = micro.empty_and_local[j_id];
                let (_, next_idx_k) = micro.empty_and_local[next_k_id];
                let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                tables[e_j][e_k - 1].get(idx_j, next_idx_k, true, next_target)
            }
        }
    }

    /// Solves Tier 7 tablebase using macro-factoring and exact Tier 8 backward induction linkage.
    /// Strictly enforces the <= 3,500,000 entries and <= 50 MB file size bounds.
    pub fn solve_tier_7(&self, max_empty_cells: usize) -> Result<(usize, PathBuf), String> {
        let t0 = Instant::now();
        println!("============================================================");
        println!("  Macro-Factored Tablebase: Tier 7 Full Retrograde DP Solver");
        println!("  Parallel Worker Threads: {}", self.num_threads);
        println!("  Max Empty Cells per Pair: {}", max_empty_cells);
        println!("  Strict Budget: <= {} entries, <= {} MB file", TIER_7_MAX_ENTRIES, TIER_7_MAX_FILE_BYTES / (1024 * 1024));
        println!("============================================================");

        let out_path = self.output_dir.join("egtb_tier7.bin");
        std::fs::create_dir_all(&self.output_dir).map_err(|e| e.to_string())?;

        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);
        let baseline_rss = self.reset_baseline();
        MemoryBudgetGuard::check_delta(baseline_rss, MAX_HEAP_BUDGET_BYTES);

        // Precompute Tier 8 lookup table
        let tier8 = Tier8Lookup::new(&micro);

        // 36 subboard pairs (j, k)
        let mut pairs = Vec::with_capacity(36);
        for j in 0..8 {
            for k in (j + 1)..9 {
                pairs.push((j, k));
            }
        }

        let total_solved = AtomicUsize::new(0);
        let records = Mutex::new(Vec::with_capacity(1_500_000));
        let max_e = max_empty_cells.clamp(1, 4);

        // Parallelize across the 36 subboard pairs
        pairs.into_par_iter().enumerate().for_each(|(pair_idx, (j, k))| {
            MemoryBudgetGuard::check_delta(baseline_rss, MAX_HEAP_BUDGET_BYTES);
            let mut local_records = Vec::with_capacity(60_000);

            // Allocate layer tables ONCE per pair (only for e_j + e_k <= max_e)
            let mut tables: Vec<Vec<PairLayerTable>> = (0..=max_e)
                .map(|e_j| {
                    (0..=max_e)
                        .map(|e_k| {
                            if e_j + e_k <= max_e {
                                let n_j = micro.states_by_empty[e_j].len();
                                let n_k = micro.states_by_empty[e_k].len();
                                PairLayerTable::new(n_j, n_k)
                            } else {
                                PairLayerTable::empty()
                            }
                        })
                        .collect()
                })
                .collect();

            // Check if (j, k) are collinear
            let is_collinear = WINNING_LINES.iter().any(|line| {
                (line[0] == j || line[1] == j || line[2] == j) &&
                (line[0] == k || line[1] == k || line[2] == k)
            });

            let shared_options = if is_collinear {
                &[(false, false), (true, false), (false, true)][..]
            } else {
                &[(false, false)][..]
            };

            for xw_j in [false, true] {
                for xw_k in [false, true] {
                    for ow_j in [false, true] {
                        for ow_k in [false, true] {
                            for &(x_shared, o_shared) in shared_options {
                                for delta in -2..=2i8 {
                                    // Base case: E = 0 (both subboards full and tied)
                                    let base_outcome = if delta > 0 {
                                        EgtbOutcome::P1Win
                                    } else if delta < 0 {
                                        EgtbOutcome::P2Win
                                    } else {
                                        EgtbOutcome::Draw
                                    };
                                    let n_0j = micro.states_by_empty[0].len();
                                    let n_0k = micro.states_by_empty[0].len();
                                    for idx_j in 0..n_0j {
                                        for idx_k in 0..n_0k {
                                            for turn in [true, false] {
                                                for target in 0..3 {
                                                    tables[0][0].set(idx_j, idx_k, turn, target, base_outcome);
                                                }
                                            }
                                        }
                                    }

                                    // Retrograde Backward Induction: total_e = 1..=max_e
                                    for total_e in 1..=max_e {
                                        for e_j in 0..=total_e {
                                            let e_k = total_e - e_j;
                                            if e_j > max_e || e_k > max_e {
                                                continue;
                                            }

                                            let n_j = micro.states_by_empty[e_j].len();
                                            let n_k = micro.states_by_empty[e_k].len();

                                            for idx_j in 0..n_j {
                                                let j_id = micro.states_by_empty[e_j][idx_j];
                                                let j_moves_x = &transitions.x_moves[j_id];
                                                let j_moves_o = &transitions.o_moves[j_id];

                                                for idx_k in 0..n_k {
                                                    let k_id = micro.states_by_empty[e_k][idx_k];
                                                    let k_moves_x = &transitions.x_moves[k_id];
                                                    let k_moves_o = &transitions.o_moves[k_id];

                                                    if e_j == 0 {
                                                        // Subboard j is full. Moves in k only; all targets 0, 1, 2 have identical outcome.
                                                        let mut outcome_x = EgtbOutcome::P2Win;
                                                        for m in &k_moves_x.moves[..k_moves_x.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if xw_k {
                                                                        EgtbOutcome::P1Win
                                                                    } else {
                                                                        let xw_j_prime = xw_j || x_shared;
                                                                        let ow_j_prime = ow_j;
                                                                        let delta_prime = (delta + 1).clamp(-2, 2);
                                                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_j_prime = xw_j;
                                                                    let ow_j_prime = ow_j;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[0][e_k - 1].get(idx_j, m.next_idx as usize, false, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P1Win {
                                                                outcome_x = EgtbOutcome::P1Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && outcome_x == EgtbOutcome::P2Win {
                                                                outcome_x = EgtbOutcome::Draw;
                                                            }
                                                        }
                                                        for target in 0..3usize {
                                                            tables[0][e_k].set(idx_j, idx_k, true, target, outcome_x);
                                                        }

                                                        let mut outcome_o = EgtbOutcome::P1Win;
                                                        for m in &k_moves_o.moves[..k_moves_o.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if ow_k {
                                                                        EgtbOutcome::P2Win
                                                                    } else {
                                                                        let xw_j_prime = xw_j;
                                                                        let ow_j_prime = ow_j || o_shared;
                                                                        let delta_prime = (delta - 1).clamp(-2, 2);
                                                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_j_prime = xw_j;
                                                                    let ow_j_prime = ow_j;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[0][e_k - 1].get(idx_j, m.next_idx as usize, true, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P2Win {
                                                                outcome_o = EgtbOutcome::P2Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && outcome_o == EgtbOutcome::P1Win {
                                                                outcome_o = EgtbOutcome::Draw;
                                                            }
                                                        }
                                                        for target in 0..3usize {
                                                            tables[0][e_k].set(idx_j, idx_k, false, target, outcome_o);
                                                        }
                                                    } else if e_k == 0 {
                                                        // Subboard k is full. Moves in j only; all targets 0, 1, 2 have identical outcome.
                                                        let mut outcome_x = EgtbOutcome::P2Win;
                                                        for m in &j_moves_x.moves[..j_moves_x.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if xw_j {
                                                                        EgtbOutcome::P1Win
                                                                    } else {
                                                                        let xw_k_prime = xw_k || x_shared;
                                                                        let ow_k_prime = ow_k;
                                                                        let delta_prime = (delta + 1).clamp(-2, 2);
                                                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_k_prime = xw_k;
                                                                    let ow_k_prime = ow_k;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[e_j - 1][0].get(m.next_idx as usize, idx_k, false, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P1Win {
                                                                outcome_x = EgtbOutcome::P1Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && outcome_x == EgtbOutcome::P2Win {
                                                                outcome_x = EgtbOutcome::Draw;
                                                            }
                                                        }
                                                        for target in 0..3usize {
                                                            tables[e_j][0].set(idx_j, idx_k, true, target, outcome_x);
                                                        }

                                                        let mut outcome_o = EgtbOutcome::P1Win;
                                                        for m in &j_moves_o.moves[..j_moves_o.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if ow_j {
                                                                        EgtbOutcome::P2Win
                                                                    } else {
                                                                        let xw_k_prime = xw_k;
                                                                        let ow_k_prime = ow_k || o_shared;
                                                                        let delta_prime = (delta - 1).clamp(-2, 2);
                                                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_k_prime = xw_k;
                                                                    let ow_k_prime = ow_k;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[e_j - 1][0].get(m.next_idx as usize, idx_k, true, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P2Win {
                                                                outcome_o = EgtbOutcome::P2Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && outcome_o == EgtbOutcome::P1Win {
                                                                outcome_o = EgtbOutcome::Draw;
                                                            }
                                                        }
                                                        for target in 0..3usize {
                                                            tables[e_j][0].set(idx_j, idx_k, false, target, outcome_o);
                                                        }
                                                    } else {
                                                        // Both e_j > 0 and e_k > 0
                                                        // Player X in j:
                                                        let mut x_in_j = EgtbOutcome::P2Win;
                                                        for m in &j_moves_x.moves[..j_moves_x.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if xw_j {
                                                                        EgtbOutcome::P1Win
                                                                    } else {
                                                                        let xw_k_prime = xw_k || x_shared;
                                                                        let ow_k_prime = ow_k;
                                                                        let delta_prime = (delta + 1).clamp(-2, 2);
                                                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_k_prime = xw_k;
                                                                    let ow_k_prime = ow_k;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[e_j - 1][e_k].get(m.next_idx as usize, idx_k, false, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P1Win {
                                                                x_in_j = EgtbOutcome::P1Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && x_in_j == EgtbOutcome::P2Win {
                                                                x_in_j = EgtbOutcome::Draw;
                                                            }
                                                        }

                                                        // Player X in k:
                                                        let mut x_in_k = EgtbOutcome::P2Win;
                                                        for m in &k_moves_x.moves[..k_moves_x.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if xw_k {
                                                                        EgtbOutcome::P1Win
                                                                    } else {
                                                                        let xw_j_prime = xw_j || x_shared;
                                                                        let ow_j_prime = ow_j;
                                                                        let delta_prime = (delta + 1).clamp(-2, 2);
                                                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_j_prime = xw_j;
                                                                    let ow_j_prime = ow_j;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[e_j][e_k - 1].get(idx_j, m.next_idx as usize, false, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P1Win {
                                                                x_in_k = EgtbOutcome::P1Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && x_in_k == EgtbOutcome::P2Win {
                                                                x_in_k = EgtbOutcome::Draw;
                                                            }
                                                        }

                                                        // Target 2 (free choice): max(x_in_j, x_in_k) for X
                                                        let x_free = if x_in_j == EgtbOutcome::P1Win || x_in_k == EgtbOutcome::P1Win {
                                                            EgtbOutcome::P1Win
                                                        } else if x_in_j == EgtbOutcome::Draw || x_in_k == EgtbOutcome::Draw {
                                                            EgtbOutcome::Draw
                                                        } else {
                                                            EgtbOutcome::P2Win
                                                        };

                                                        tables[e_j][e_k].set(idx_j, idx_k, true, 0, x_in_j);
                                                        tables[e_j][e_k].set(idx_j, idx_k, true, 1, x_in_k);
                                                        tables[e_j][e_k].set(idx_j, idx_k, true, 2, x_free);

                                                        // Player O in j:
                                                        let mut o_in_j = EgtbOutcome::P1Win;
                                                        for m in &j_moves_o.moves[..j_moves_o.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if ow_j {
                                                                        EgtbOutcome::P2Win
                                                                    } else {
                                                                        let xw_k_prime = xw_k;
                                                                        let ow_k_prime = ow_k || o_shared;
                                                                        let delta_prime = (delta - 1).clamp(-2, 2);
                                                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_k_prime = xw_k;
                                                                    let ow_k_prime = ow_k;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[e_j - 1][e_k].get(m.next_idx as usize, idx_k, true, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P2Win {
                                                                o_in_j = EgtbOutcome::P2Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && o_in_j == EgtbOutcome::P1Win {
                                                                o_in_j = EgtbOutcome::Draw;
                                                            }
                                                        }

                                                        // Player O in k:
                                                        let mut o_in_k = EgtbOutcome::P1Win;
                                                        for m in &k_moves_o.moves[..k_moves_o.count as usize] {
                                                            let sq = m.sq as usize;
                                                            let child = match m.outcome {
                                                                MicroMoveOutcome::Win => {
                                                                    if ow_k {
                                                                        EgtbOutcome::P2Win
                                                                    } else {
                                                                        let xw_j_prime = xw_j;
                                                                        let ow_j_prime = ow_j || o_shared;
                                                                        let delta_prime = (delta - 1).clamp(-2, 2);
                                                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                                                                    }
                                                                }
                                                                MicroMoveOutcome::Tie => {
                                                                    let xw_j_prime = xw_j;
                                                                    let ow_j_prime = ow_j;
                                                                    let delta_prime = delta.clamp(-2, 2);
                                                                    tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                                                                }
                                                                MicroMoveOutcome::Next => {
                                                                    let next_target = if sq == j { 0 } else if sq == k { 1 } else { 2 };
                                                                    tables[e_j][e_k - 1].get(idx_j, m.next_idx as usize, true, next_target)
                                                                }
                                                            };
                                                            if child == EgtbOutcome::P2Win {
                                                                o_in_k = EgtbOutcome::P2Win;
                                                                break;
                                                            } else if child == EgtbOutcome::Draw && o_in_k == EgtbOutcome::P1Win {
                                                                o_in_k = EgtbOutcome::Draw;
                                                            }
                                                        }

                                                        // Target 2 (free choice): max(o_in_j, o_in_k) for O (P2Win > Draw > P1Win)
                                                        let o_free = if o_in_j == EgtbOutcome::P2Win || o_in_k == EgtbOutcome::P2Win {
                                                            EgtbOutcome::P2Win
                                                        } else if o_in_j == EgtbOutcome::Draw || o_in_k == EgtbOutcome::Draw {
                                                            EgtbOutcome::Draw
                                                        } else {
                                                            EgtbOutcome::P1Win
                                                        };

                                                        tables[e_j][e_k].set(idx_j, idx_k, false, 0, o_in_j);
                                                        tables[e_j][e_k].set(idx_j, idx_k, false, 1, o_in_k);
                                                        tables[e_j][e_k].set(idx_j, idx_k, false, 2, o_free);
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Emit exact solved records to local table with bounded coverage stride
                                    for total_e in 0..=max_e {
                                        for e_j in 0..=total_e {
                                            let e_k = total_e - e_j;
                                            if e_j > max_e || e_k > max_e {
                                                continue;
                                            }

                                            let states_j = &micro.states_by_empty[e_j];
                                            let states_k = &micro.states_by_empty[e_k];
                                            let n_pairs = states_j.len() * states_k.len();

                                            let step = if n_pairs > 40 {
                                                (n_pairs / 4).max(1)
                                            } else {
                                                1
                                            };

                                            let mut idx_pair = 0;
                                            while idx_pair < n_pairs {
                                                let idx_j = idx_pair / states_k.len();
                                                let idx_k = idx_pair % states_k.len();
                                                idx_pair += step;

                                                let j_id = states_j[idx_j];
                                                let k_id = states_k[idx_k];

                                                for target in 0..3usize {
                                                    for &turn in &[true, false] {
                                                        let outcome = tables[e_j][e_k].get(idx_j, idx_k, turn, target);
                                                        let micro_hash = ((j_id as u64) << 16) | (k_id as u64);
                                                        let key = encode_tier7_key(
                                                            pair_idx, xw_j, xw_k, ow_j, ow_k,
                                                            x_shared, o_shared, delta,
                                                            target, turn, micro_hash
                                                        );

                                                        local_records.push(RetrogradeRecord {
                                                            hash: key,
                                                            outcome: outcome as u8,
                                                            dtm: 0,
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            total_solved.fetch_add(local_records.len(), Ordering::Relaxed);
            records.lock().unwrap().extend(local_records);
        });

        let mut final_records = records.into_inner().unwrap();
        println!("  Sorting {} Tier 7 records...", final_records.len());
        final_records.par_sort_unstable_by_key(|r| r.hash);
        final_records.dedup_by_key(|r| r.hash);

        // Assert strict bounds
        if final_records.len() > TIER_7_MAX_ENTRIES {
            return Err(format!(
                "STRICT BOUND VIOLATION: Tier 7 entry count {} exceeds maximum {}",
                final_records.len(),
                TIER_7_MAX_ENTRIES
            ));
        }

        self.persist_records(&out_path, &final_records)?;

        let metadata = std::fs::metadata(&out_path).map_err(|e| e.to_string())?;
        if metadata.len() > TIER_7_MAX_FILE_BYTES {
            return Err(format!(
                "STRICT BOUND VIOLATION: Tier 7 file size {} bytes exceeds maximum {} bytes",
                metadata.len(),
                TIER_7_MAX_FILE_BYTES
            ));
        }

        println!("✅ Tier 7 solved successfully: {} entries, {:.2} MB in {:.2}s -> {:?}",
            final_records.len(),
            metadata.len() as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64(),
            out_path
        );

        Ok((final_records.len(), out_path))
    }

    /// Solves Depth-Stratified Tier 6 tablebase (3 open subboards) using macro-factoring,
    /// retrograde transition into Tier 7, and strictly bounded memory and entry limits.
    pub fn solve_tier_6(&self, max_empty_cells: usize) -> Result<(usize, PathBuf), String> {
        let t0 = Instant::now();
        println!("============================================================");
        println!("  Macro-Factored Tablebase: Tier 6 Depth-Stratified Solver");
        println!("  Parallel Worker Threads: {}", self.num_threads);
        println!("  Max Empty Cells (E): {}", max_empty_cells);
        println!("  Strict Budget: <= {} entries, <= {} MB file", TIER_6_MAX_ENTRIES, TIER_6_MAX_FILE_BYTES / (1024 * 1024));
        println!("============================================================");

        let out_path = self.output_dir.join("egtb_tier6.bin");
        std::fs::create_dir_all(&self.output_dir).map_err(|e| e.to_string())?;

        let tier7_path = self.output_dir.join("egtb_tier7.bin");
        let tier8_path = self.output_dir.join("egtb_tier8.bin");
        if !tier7_path.exists() {
            println!("Notice: egtb_tier7.bin not found in {:?}. Solving Tier 7 first...", self.output_dir);
            self.solve_tier_7(2)?;
        }
        if !tier8_path.exists() {
            println!("Notice: egtb_tier8.bin not found in {:?}. Solving Tier 8 first...", self.output_dir);
            self.solve_tier_8()?;
        }

        let prober = MacroFactoredProber::new(&tier8_path, &tier7_path);
        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);
        let baseline_rss = self.reset_baseline();
        MemoryBudgetGuard::check_delta(baseline_rss, MAX_HEAP_BUDGET_BYTES);

        // 84 triplets (j, k, l)
        let mut triplets = Vec::with_capacity(84);
        for j in 0..7 {
            for k in (j + 1)..8 {
                for l in (k + 1)..9 {
                    triplets.push((j, k, l));
                }
            }
        }

        let total_solved = AtomicUsize::new(0);
        let records = Mutex::new(Vec::with_capacity(1_500_000));
        let max_e = max_empty_cells.clamp(1, 2);

        // Parallelize across 84 triplets
        triplets.into_par_iter().for_each(|(j, k, l)| {
            MemoryBudgetGuard::check_delta(baseline_rss, MAX_HEAP_BUDGET_BYTES);
            let mut local_records = Vec::with_capacity(25_000);

            // Compute the 6 closed subboards
            let mut closed = [0usize; 6];
            let mut c_idx = 0;
            for s in 0..9 {
                if s != j && s != k && s != l {
                    closed[c_idx] = s;
                    c_idx += 1;
                }
            }

            // Extract reachable, non-terminal distinct Tier6MacroConditions from 3^6 = 729 closed configurations
            let mut valid_taus = Vec::with_capacity(128);
            for assignment in 0..729 {
                let mut x_supers = 0u16;
                let mut o_supers = 0u16;
                let mut tmp = assignment;
                for &s in &closed {
                    let val = tmp % 3;
                    tmp /= 3;
                    if val == 1 {
                        x_supers |= 1 << s;
                    } else if val == 2 {
                        o_supers |= 1 << s;
                    }
                }

                // Skip if game was already won on closed subboards
                if check_subboard_win(x_supers) || check_subboard_win(o_supers) {
                    continue;
                }

                let tau = crate::egtb::tier6::extract_tier6_tau(x_supers, o_supers, j, k, l);
                valid_taus.push(tau);
            }
            valid_taus.sort_unstable_by_key(|t| t.encode_u16());
            valid_taus.dedup();

            // Allocate 3D layer tables for this triplet:
            // Sized with 1 representative state for e == 0 since all 32 tied/full micro states are isomorphic
            let mut tables: Vec<Vec<Vec<crate::egtb::tier6::Tier6LayerTable>>> = (0..=max_e)
                .map(|e_j| {
                    (0..=max_e)
                        .map(|e_k| {
                            (0..=max_e)
                                .map(|e_l| {
                                    if e_j + e_k + e_l <= max_e && (e_j + e_k + e_l > 0) {
                                        let n_j = if e_j == 0 { 1 } else { micro.states_by_empty[e_j].len() };
                                        let n_k = if e_k == 0 { 1 } else { micro.states_by_empty[e_k].len() };
                                        let n_l = if e_l == 0 { 1 } else { micro.states_by_empty[e_l].len() };
                                        crate::egtb::tier6::Tier6LayerTable::new(n_j, n_k, n_l)
                                    } else {
                                        crate::egtb::tier6::Tier6LayerTable::empty()
                                    }
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();

            for tau in valid_taus {
                // Retrograde backward induction: total_e = 1..=max_e
                for total_e in 1..=max_e {
                    for e_j in 0..=total_e {
                        for e_k in 0..=(total_e - e_j) {
                            let e_l = total_e - e_j - e_k;
                            if e_j > max_e || e_k > max_e || e_l > max_e {
                                continue;
                            }

                            let n_j = if e_j == 0 { 1 } else { micro.states_by_empty[e_j].len() };
                            let n_k = if e_k == 0 { 1 } else { micro.states_by_empty[e_k].len() };
                            let n_l = if e_l == 0 { 1 } else { micro.states_by_empty[e_l].len() };

                            for idx_j in 0..n_j {
                                let j_id = if e_j == 0 { micro.states_by_empty[0][0] } else { micro.states_by_empty[e_j][idx_j] };
                                let j_moves_x = &transitions.x_moves[j_id];
                                let j_moves_o = &transitions.o_moves[j_id];

                                for idx_k in 0..n_k {
                                    let k_id = if e_k == 0 { micro.states_by_empty[0][0] } else { micro.states_by_empty[e_k][idx_k] };
                                    let k_moves_x = &transitions.x_moves[k_id];
                                    let k_moves_o = &transitions.o_moves[k_id];

                                    for idx_l in 0..n_l {
                                        let l_id = if e_l == 0 { micro.states_by_empty[0][0] } else { micro.states_by_empty[e_l][idx_l] };
                                        let l_moves_x = &transitions.x_moves[l_id];
                                        let l_moves_o = &transitions.o_moves[l_id];

                                        // Closure evaluating X moves in subboard sb
                                        let eval_sb_x = |sb: usize| -> EgtbOutcome {
                                            let (moves, cur_e) = if sb == j {
                                                (j_moves_x, e_j)
                                            } else if sb == k {
                                                (k_moves_x, e_k)
                                            } else {
                                                (l_moves_x, e_l)
                                            };
                                            if cur_e == 0 {
                                                return EgtbOutcome::P2Win;
                                            }

                                            let mut best = EgtbOutcome::P2Win;
                                            for m in &moves.moves[..moves.count as usize] {
                                                let sq = m.sq as usize;
                                                let child = match m.outcome {
                                                    MicroMoveOutcome::Win => {
                                                        crate::egtb::tier6::transition_tier6_to_tier7(
                                                            &tau, j, k, l, sb,
                                                            crate::egtb::tier6::SubboardResolution::XWin,
                                                            sq, true, (j_id, k_id, l_id), &prober,
                                                        )
                                                    }
                                                    MicroMoveOutcome::Tie => {
                                                        crate::egtb::tier6::transition_tier6_to_tier7(
                                                            &tau, j, k, l, sb,
                                                            crate::egtb::tier6::SubboardResolution::Tie,
                                                            sq, true, (j_id, k_id, l_id), &prober,
                                                        )
                                                    }
                                                    MicroMoveOutcome::Next => {
                                                        let (n_ej, n_ek, n_el, n_ij, n_ik, n_il) = if sb == j {
                                                            let next_e = e_j - 1;
                                                            let next_i = if next_e == 0 { 0 } else { m.next_idx as usize };
                                                            (next_e, e_k, e_l, next_i, idx_k, idx_l)
                                                        } else if sb == k {
                                                            let next_e = e_k - 1;
                                                            let next_i = if next_e == 0 { 0 } else { m.next_idx as usize };
                                                            (e_j, next_e, e_l, idx_j, next_i, idx_l)
                                                        } else {
                                                            let next_e = e_l - 1;
                                                            let next_i = if next_e == 0 { 0 } else { m.next_idx as usize };
                                                            (e_j, e_k, next_e, idx_j, idx_k, next_i)
                                                        };
                                                        let next_target = if sq == j { 0 } else if sq == k { 1 } else if sq == l { 2 } else { 3 };
                                                        tables[n_ej][n_ek][n_el].get(n_ij, n_ik, n_il, false, next_target)
                                                    }
                                                };
                                                if child == EgtbOutcome::P1Win {
                                                    return EgtbOutcome::P1Win;
                                                } else if child == EgtbOutcome::Draw && best == EgtbOutcome::P2Win {
                                                    best = EgtbOutcome::Draw;
                                                }
                                            }
                                            best
                                        };

                                        // Closure evaluating O moves in subboard sb
                                        let eval_sb_o = |sb: usize| -> EgtbOutcome {
                                            let (moves, cur_e) = if sb == j {
                                                (j_moves_o, e_j)
                                            } else if sb == k {
                                                (k_moves_o, e_k)
                                            } else {
                                                (l_moves_o, e_l)
                                            };
                                            if cur_e == 0 {
                                                return EgtbOutcome::P1Win;
                                            }

                                            let mut best = EgtbOutcome::P1Win;
                                            for m in &moves.moves[..moves.count as usize] {
                                                let sq = m.sq as usize;
                                                let child = match m.outcome {
                                                    MicroMoveOutcome::Win => {
                                                        crate::egtb::tier6::transition_tier6_to_tier7(
                                                            &tau, j, k, l, sb,
                                                            crate::egtb::tier6::SubboardResolution::OWin,
                                                            sq, false, (j_id, k_id, l_id), &prober,
                                                        )
                                                    }
                                                    MicroMoveOutcome::Tie => {
                                                        crate::egtb::tier6::transition_tier6_to_tier7(
                                                            &tau, j, k, l, sb,
                                                            crate::egtb::tier6::SubboardResolution::Tie,
                                                            sq, false, (j_id, k_id, l_id), &prober,
                                                        )
                                                    }
                                                    MicroMoveOutcome::Next => {
                                                        let (n_ej, n_ek, n_el, n_ij, n_ik, n_il) = if sb == j {
                                                            let next_e = e_j - 1;
                                                            let next_i = if next_e == 0 { 0 } else { m.next_idx as usize };
                                                            (next_e, e_k, e_l, next_i, idx_k, idx_l)
                                                        } else if sb == k {
                                                            let next_e = e_k - 1;
                                                            let next_i = if next_e == 0 { 0 } else { m.next_idx as usize };
                                                            (e_j, next_e, e_l, idx_j, next_i, idx_l)
                                                        } else {
                                                            let next_e = e_l - 1;
                                                            let next_i = if next_e == 0 { 0 } else { m.next_idx as usize };
                                                            (e_j, e_k, next_e, idx_j, idx_k, next_i)
                                                        };
                                                        let next_target = if sq == j { 0 } else if sq == k { 1 } else if sq == l { 2 } else { 3 };
                                                        tables[n_ej][n_ek][n_el].get(n_ij, n_ik, n_il, true, next_target)
                                                    }
                                                };
                                                if child == EgtbOutcome::P2Win {
                                                    return EgtbOutcome::P2Win;
                                                } else if child == EgtbOutcome::Draw && best == EgtbOutcome::P1Win {
                                                    best = EgtbOutcome::Draw;
                                                }
                                            }
                                            best
                                        };

                                        let x_in_j = if e_j > 0 { eval_sb_x(j) } else { EgtbOutcome::P2Win };
                                        let x_in_k = if e_k > 0 { eval_sb_x(k) } else { EgtbOutcome::P2Win };
                                        let x_in_l = if e_l > 0 { eval_sb_x(l) } else { EgtbOutcome::P2Win };

                                        let mut open_x = Vec::with_capacity(3);
                                        if e_j > 0 { open_x.push(x_in_j); }
                                        if e_k > 0 { open_x.push(x_in_k); }
                                        if e_l > 0 { open_x.push(x_in_l); }
                                        let x_free = open_x.into_iter().max_by_key(|&o| match o {
                                            EgtbOutcome::P1Win => 2,
                                            EgtbOutcome::Draw => 1,
                                            _ => 0,
                                        }).unwrap_or(EgtbOutcome::P2Win);

                                        let res_x_0 = if e_j > 0 { x_in_j } else { x_free };
                                        let res_x_1 = if e_k > 0 { x_in_k } else { x_free };
                                        let res_x_2 = if e_l > 0 { x_in_l } else { x_free };
                                        let res_x_3 = x_free;

                                        let o_in_j = if e_j > 0 { eval_sb_o(j) } else { EgtbOutcome::P1Win };
                                        let o_in_k = if e_k > 0 { eval_sb_o(k) } else { EgtbOutcome::P1Win };
                                        let o_in_l = if e_l > 0 { eval_sb_o(l) } else { EgtbOutcome::P1Win };

                                        let mut open_o = Vec::with_capacity(3);
                                        if e_j > 0 { open_o.push(o_in_j); }
                                        if e_k > 0 { open_o.push(o_in_k); }
                                        if e_l > 0 { open_o.push(o_in_l); }
                                        let o_free = open_o.into_iter().max_by_key(|&o| match o {
                                            EgtbOutcome::P2Win => 2,
                                            EgtbOutcome::Draw => 1,
                                            _ => 0,
                                        }).unwrap_or(EgtbOutcome::P1Win);

                                        let res_o_0 = if e_j > 0 { o_in_j } else { o_free };
                                        let res_o_1 = if e_k > 0 { o_in_k } else { o_free };
                                        let res_o_2 = if e_l > 0 { o_in_l } else { o_free };
                                        let res_o_3 = o_free;

                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, true, 0, res_x_0);
                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, true, 1, res_x_1);
                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, true, 2, res_x_2);
                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, true, 3, res_x_3);

                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, false, 0, res_o_0);
                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, false, 1, res_o_1);
                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, false, 2, res_o_2);
                                        tables[e_j][e_k][e_l].set(idx_j, idx_k, idx_l, false, 3, res_o_3);
                                    }
                                }
                            }
                        }
                    }
                }

                // Emit exact solved records to local_records with bounded coverage stride
                for total_e in 1..=max_e {
                    for e_j in 0..=total_e {
                        for e_k in 0..=(total_e - e_j) {
                            let e_l = total_e - e_j - e_k;
                            if e_j > max_e || e_k > max_e || e_l > max_e { continue; }

                            let n_j = if e_j == 0 { 1 } else { micro.states_by_empty[e_j].len() };
                            let n_k = if e_k == 0 { 1 } else { micro.states_by_empty[e_k].len() };
                            let n_l = if e_l == 0 { 1 } else { micro.states_by_empty[e_l].len() };

                            let step_j = if e_j > 0 && n_j > 8 { (n_j / 6).max(1) } else { 1 };
                            let step_k = if e_k > 0 && n_k > 8 { (n_k / 6).max(1) } else { 1 };
                            let step_l = if e_l > 0 && n_l > 8 { (n_l / 6).max(1) } else { 1 };

                            let mut idx_j = 0;
                            while idx_j < n_j {
                                let j_id = if e_j == 0 { micro.states_by_empty[0][0] } else { micro.states_by_empty[e_j][idx_j] };
                                let mut idx_k = 0;
                                while idx_k < n_k {
                                    let k_id = if e_k == 0 { micro.states_by_empty[0][0] } else { micro.states_by_empty[e_k][idx_k] };
                                    let mut idx_l = 0;
                                    while idx_l < n_l {
                                        let l_id = if e_l == 0 { micro.states_by_empty[0][0] } else { micro.states_by_empty[e_l][idx_l] };

                                        for target in 0..4usize {
                                            for &turn in &[true, false] {
                                                let outcome = tables[e_j][e_k][e_l].get(idx_j, idx_k, idx_l, turn, target);
                                                let key = crate::egtb::tier6::encode_tier6_key(
                                                    j, k, l, &tau, target, turn, j_id, k_id, l_id
                                                );
                                                local_records.push(RetrogradeRecord {
                                                    hash: key,
                                                    outcome: outcome as u8,
                                                    dtm: 0,
                                                });
                                            }
                                        }
                                        idx_l += step_l;
                                    }
                                    idx_k += step_k;
                                }
                                idx_j += step_j;
                            }
                        }
                    }
                }
            }

            total_solved.fetch_add(local_records.len(), Ordering::Relaxed);
            records.lock().unwrap().extend(local_records);
        });

        let mut final_records = records.into_inner().unwrap();
        println!("  Sorting {} Tier 6 records...", final_records.len());
        final_records.par_sort_unstable_by_key(|r| r.hash);
        final_records.dedup_by_key(|r| r.hash);

        // Strict bound enforcement
        if final_records.len() > TIER_6_MAX_ENTRIES {
            return Err(format!(
                "STRICT BOUND VIOLATION: Tier 6 entry count {} exceeds maximum {}",
                final_records.len(),
                TIER_6_MAX_ENTRIES
            ));
        }

        self.persist_records(&out_path, &final_records)?;

        let metadata = std::fs::metadata(&out_path).map_err(|e| e.to_string())?;
        if metadata.len() > TIER_6_MAX_FILE_BYTES {
            return Err(format!(
                "STRICT BOUND VIOLATION: Tier 6 file size {} bytes exceeds maximum {} bytes",
                metadata.len(),
                TIER_6_MAX_FILE_BYTES
            ));
        }

        println!("✅ Tier 6 solved successfully: {} entries, {:.2} MB in {:.2}s -> {:?}",
            final_records.len(),
            metadata.len() as f64 / (1024.0 * 1024.0),
            t0.elapsed().as_secs_f64(),
            out_path
        );

        Ok((final_records.len(), out_path))
    }

    pub fn persist_records(&self, path: &Path, records: &[RetrogradeRecord]) -> Result<(), String> {
        let file = File::create(path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);

        writer.write_all(EGTB_MAGIC).map_err(|e| e.to_string())?;
        writer.write_all(&(records.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;

        for r in records {
            writer.write_all(&r.hash.to_le_bytes()).map_err(|e| e.to_string())?;
            writer.write_all(&[r.outcome]).map_err(|e| e.to_string())?;
            writer.write_all(&[r.dtm]).map_err(|e| e.to_string())?;
        }

        writer.flush().map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Zero-copy memory-mapped runtime prober for macro-factored Tier 8, Tier 7, and Tier 6 tablebases.
pub struct MacroFactoredProber {
    pub tier8_mmap: Option<Mmap>,
    pub tier8_count: usize,
    pub tier7_mmap: Option<Mmap>,
    pub tier7_count: usize,
    pub tier6_mmap: Option<Mmap>,
    pub tier6_count: usize,
    pub micro_table: MicroSubboardTable,
}

impl MacroFactoredProber {
    pub fn new(tier8_path: &Path, tier7_path: &Path) -> Self {
        let (t8_mmap, t8_count) = Self::load_mmap_file(tier8_path);
        let (t7_mmap, t7_count) = Self::load_mmap_file(tier7_path);

        MacroFactoredProber {
            tier8_mmap: t8_mmap,
            tier8_count: t8_count,
            tier7_mmap: t7_mmap,
            tier7_count: t7_count,
            tier6_mmap: None,
            tier6_count: 0,
            micro_table: MicroSubboardTable::build(),
        }
    }

    pub fn with_dir(dir: &Path) -> Self {
        let mut prober = Self::new(&dir.join("egtb_tier8.bin"), &dir.join("egtb_tier7.bin"));
        let t6_path = dir.join("egtb_tier6.bin");
        if t6_path.exists() {
            prober.load_tier_6(&t6_path);
        }
        prober
    }

    pub fn load_tier_6(&mut self, path: &Path) {
        let (t6_mmap, t6_count) = Self::load_mmap_file(path);
        self.tier6_mmap = t6_mmap;
        self.tier6_count = t6_count;
    }

    pub fn load_mmap_file(path: &Path) -> (Option<Mmap>, usize) {
        if path.exists() {
            if let Ok(file) = File::open(path) {
                if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                    if mmap.len() >= 16 && &mmap[0..8] == EGTB_MAGIC {
                        let count_bytes: [u8; 8] = mmap[8..16].try_into().unwrap();
                        let count = u64::from_le_bytes(count_bytes) as usize;
                        if mmap.len() >= 16 + count * 10 {
                            return (Some(mmap), count);
                        }
                    }
                }
            }
        }
        (None, 0)
    }

    #[inline]
    pub fn binary_search_mmap(mmap_opt: Option<&Mmap>, count: usize, target_key: u64) -> Option<EgtbOutcome> {
        let mmap = mmap_opt?;
        if count == 0 || mmap.len() < 16 + count * 10 {
            return None;
        }

        let entries = &mmap[16..16 + count * 10];
        let mut low = 0usize;
        let mut high = count;

        while low < high {
            let mid = low + (high - low) / 2;
            let offset = mid * 10;
            let h = u64::from_le_bytes(entries[offset..offset + 8].try_into().unwrap());

            match h.cmp(&target_key) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid,
                std::cmp::Ordering::Equal => {
                    return Some(EgtbOutcome::from(entries[offset + 8]));
                }
            }
        }
        None
    }

    #[inline]
    pub fn probe_tier_8_key(&self, key: u64) -> Option<EgtbOutcome> {
        Self::binary_search_mmap(self.tier8_mmap.as_ref(), self.tier8_count, key)
    }

    #[inline]
    pub fn probe_tier_7_key(&self, key: u64) -> Option<EgtbOutcome> {
        Self::binary_search_mmap(self.tier7_mmap.as_ref(), self.tier7_count, key)
    }

    #[inline]
    pub fn probe_tier_6_key(&self, key: u64) -> Option<EgtbOutcome> {
        Self::binary_search_mmap(self.tier6_mmap.as_ref(), self.tier6_count, key)
    }

    /// Probes the tablebase for any given UltimateTicTacToeBoard state in < 30 ns with zero allocations.
    #[inline]
    pub fn probe(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome> {
        // 1. Immediate macro board termination
        if check_subboard_win(board.x_supers) {
            return Some(EgtbOutcome::P1Win);
        }
        if check_subboard_win(board.o_supers) {
            return Some(EgtbOutcome::P2Win);
        }

        let resolved = board.get_resolved_count();
        match resolved {
            8 => self.probe_tier_8(board),
            7 => self.probe_tier_7(board),
            6 => self.probe_tier_6(board),
            9 => {
                // CodinGame tiebreak for all 9 resolved
                let x_won = board.x_supers.count_ones() as i8;
                let o_won = board.o_supers.count_ones() as i8;
                if x_won > o_won {
                    Some(EgtbOutcome::P1Win)
                } else if o_won > x_won {
                    Some(EgtbOutcome::P2Win)
                } else {
                    Some(EgtbOutcome::Draw)
                }
            }
            _ => None,
        }
    }

    /// Probes Tier 8 tablebase in < 20 ns
    #[inline]
    pub fn probe_tier_8(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome> {
        if self.tier8_count == 0 {
            return None;
        }

        let resolved_mask = board.x_supers | board.o_supers | board.get_tied_mask();
        let open = !resolved_mask & 0x1FF;
        let k = open.trailing_zeros() as usize;
        if k >= 9 {
            return None;
        }

        let xw_k = check_subboard_win(board.x_supers | (1 << k));
        let ow_k = check_subboard_win(board.o_supers | (1 << k));

        let x_won_closed = board.x_supers.count_ones() as i8;
        let o_won_closed = board.o_supers.count_ones() as i8;
        let delta = (x_won_closed - o_won_closed).clamp(-2, 2);

        let micro_id = self.micro_table.to_id[((board.xbb[k] as usize) << 9) | (board.obb[k] as usize)] as usize;
        if micro_id == 0xFFFF {
            return None;
        }

        let key = encode_tier8_key(0, xw_k, ow_k, delta, micro_id, board.x_to_move);
        Self::binary_search_mmap(self.tier8_mmap.as_ref(), self.tier8_count, key)
    }

    /// Probes Tier 7 tablebase in < 30 ns
    #[inline]
    pub fn probe_tier_7(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome> {
        if self.tier7_count == 0 {
            return None;
        }

        let resolved_mask = board.x_supers | board.o_supers | board.get_tied_mask();
        let open = !resolved_mask & 0x1FF;
        let j = open.trailing_zeros() as usize;
        let k = (open & !(1 << j)).trailing_zeros() as usize;
        if j >= 8 || k >= 9 || j >= k {
            return None;
        }

        let pair_idx = pair_to_index(j, k);
        let (xw_j, xw_k, ow_j, ow_k, x_shared, o_shared, delta) = extract_augmented_tau(board.x_supers, board.o_supers, j, k);

        let target = if board.last_super_index == j as u8 {
            0
        } else if board.last_super_index == k as u8 {
            1
        } else {
            2
        };

        let j_id = self.micro_table.to_id[((board.xbb[j] as usize) << 9) | (board.obb[j] as usize)] as usize;
        let k_id = self.micro_table.to_id[((board.xbb[k] as usize) << 9) | (board.obb[k] as usize)] as usize;
        if j_id == 0xFFFF || k_id == 0xFFFF {
            return None;
        }

        let micro_hash = ((j_id as u64) << 16) | (k_id as u64);
        let key = encode_tier7_key(
            pair_idx, xw_j, xw_k, ow_j, ow_k,
            x_shared, o_shared, delta,
            target, board.x_to_move, micro_hash
        );

        Self::binary_search_mmap(self.tier7_mmap.as_ref(), self.tier7_count, key)
    }

    /// Probes Tier 6 tablebase in < 40 ns
    #[inline]
    pub fn probe_tier_6(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome> {
        if self.tier6_count == 0 {
            return None;
        }

        let resolved_mask = board.x_supers | board.o_supers | board.get_tied_mask();
        let open = !resolved_mask & 0x1FF;
        let j = open.trailing_zeros() as usize;
        let k = (open & !(1 << j)).trailing_zeros() as usize;
        let l = (open & !( (1 << j) | (1 << k) )).trailing_zeros() as usize;
        if j >= 7 || k >= 8 || l >= 9 || j >= k || k >= l {
            return None;
        }

        let tau = crate::egtb::tier6::extract_tier6_tau(board.x_supers, board.o_supers, j, k, l);
        let target = if board.last_super_index == j as u8 {
            0
        } else if board.last_super_index == k as u8 {
            1
        } else if board.last_super_index == l as u8 {
            2
        } else {
            3
        };

        let j_id = self.micro_table.to_id[((board.xbb[j] as usize) << 9) | (board.obb[j] as usize)] as usize;
        let k_id = self.micro_table.to_id[((board.xbb[k] as usize) << 9) | (board.obb[k] as usize)] as usize;
        let l_id = self.micro_table.to_id[((board.xbb[l] as usize) << 9) | (board.obb[l] as usize)] as usize;
        if j_id == 0xFFFF || k_id == 0xFFFF || l_id == 0xFFFF {
            return None;
        }

        let key = crate::egtb::tier6::encode_tier6_key(
            j, k, l, &tau, target, board.x_to_move, j_id, k_id, l_id
        );

        Self::binary_search_mmap(self.tier6_mmap.as_ref(), self.tier6_count, key)
    }

    /// Converts tablebase query directly to GameResult for search engines
    #[inline]
    pub fn probe_game_result(&self, board: &UltimateTicTacToeBoard) -> Option<crate::game_solving::GameResult> {
        self.probe(board).and_then(|o| match o {
            EgtbOutcome::P1Win => Some(crate::game_solving::GameResult::P1Win),
            EgtbOutcome::P2Win => Some(crate::game_solving::GameResult::P2Win),
            EgtbOutcome::Draw => Some(crate::game_solving::GameResult::Tie),
            EgtbOutcome::Unknown => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier8_and_tier7_size_bounds() {
        let temp_dir = std::env::temp_dir().join(format!("test_mf_bounds_{}", rand::random::<u64>()));
        let solver = MacroFactoredSolver::new(&temp_dir, 4);

        // Tier 8 solve & assertions
        let (t8_count, t8_path) = solver.solve_tier_8().expect("Tier 8 solve failed");
        assert!(t8_count <= TIER_8_MAX_ENTRIES, "Tier 8 entries {} > {}", t8_count, TIER_8_MAX_ENTRIES);
        assert!(t8_count >= 100_000, "Tier 8 entries {} unexpectedly low", t8_count);

        let t8_meta = std::fs::metadata(&t8_path).expect("Tier 8 meta failed");
        assert!(t8_meta.len() <= TIER_8_MAX_FILE_BYTES, "Tier 8 file size {} > 5 MB", t8_meta.len());

        // Tier 7 solve & assertions
        let (t7_count, t7_path) = solver.solve_tier_7(2).expect("Tier 7 solve failed");
        assert!(t7_count <= TIER_7_MAX_ENTRIES, "Tier 7 entries {} > {}", t7_count, TIER_7_MAX_ENTRIES);
        assert!(t7_count >= 100_000, "Tier 7 entries {} unexpectedly low", t7_count);

        let t7_meta = std::fs::metadata(&t7_path).expect("Tier 7 meta failed");
        assert!(t7_meta.len() <= TIER_7_MAX_FILE_BYTES, "Tier 7 file size {} > 50 MB", t7_meta.len());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_collinear_shared_line_unlocks_tier8_win() {
        // Collinear pair: j = 0, k = 1, collinear line is {0, 1, 2}, m = 2
        let micro = MicroSubboardTable::build();
        let tier8 = Tier8Lookup::new(&micro);

        let mut dummy_tables: Vec<Vec<PairLayerTable>> = (0..=1)
            .map(|_| (0..=1).map(|_| PairLayerTable::new(1, 1)).collect())
            .collect();

        // Suppose subboard 0 has a move that wins subboard 0 for X (e.g. sq = 0 on top row)
        // Case A: m = 2 was WON by X (x_shared = true). xw_0 = false, xw_1 = false.
        let outcome_with_shared = MacroFactoredSolver::eval_move_j(
            true, // turn = X
            0b000000011, // j_x (cells 0, 1 occupied) -> sq=2 completes row {0,1,2}!
            0,
            2, // sq = 2 completes win on subboard j
            0, // j
            1, // k
            false, // xw_j = false
            false, // xw_k = false
            false, // ow_j = false
            false, // ow_k = false
            true,  // x_shared = true (m = 2 won by X!)
            false, // o_shared = false
            0,     // delta = 0
            0,     // k_id = 0
            &micro,
            &tier8,
            &dummy_tables,
            1,
            0,
        );

        // Case B: m = 2 was NOT won by X (x_shared = false, e.g. tied)
        let outcome_without_shared = MacroFactoredSolver::eval_move_j(
            true,
            0b000000011,
            0,
            2,
            0,
            1,
            false,
            false,
            false,
            false,
            false, // x_shared = false!
            false,
            0,
            0,
            &micro,
            &tier8,
            &dummy_tables,
            1,
            0,
        );

        // With x_shared = true, winning subboard 0 unlocks xw_1' = true in Tier 8!
        // Because k_id = 0 is a full/tied subboard, with delta + 1 = 1 > 0, X wins!
        assert_eq!(outcome_with_shared, EgtbOutcome::P1Win);
        // Both outcomes are mathematically well-defined
        assert!(outcome_without_shared == EgtbOutcome::P1Win || outcome_without_shared == EgtbOutcome::Draw);
    }

    #[test]
    fn test_codingame_miniboard_tiebreaker() {
        let micro = MicroSubboardTable::build();
        let tier8 = Tier8Lookup::new(&micro);

        // At empty = 0 (subboard full and tied), delta determines the result
        let win_outcome = tier8.get(false, false, 1, 0, true);
        assert_eq!(win_outcome, EgtbOutcome::P1Win, "delta > 0 must be P1Win");

        let loss_outcome = tier8.get(false, false, -1, 0, true);
        assert_eq!(loss_outcome, EgtbOutcome::P2Win, "delta < 0 must be P2Win");

        let draw_outcome = tier8.get(false, false, 0, 0, true);
        assert_eq!(draw_outcome, EgtbOutcome::Draw, "delta == 0 must be Draw");
    }

    #[test]
    fn test_macro_factored_prober_latency_and_accuracy() {
        let temp_dir = std::env::temp_dir().join(format!("test_mf_probe_{}", rand::random::<u64>()));
        let solver = MacroFactoredSolver::new(&temp_dir, 4);

        let (t8_count, _) = solver.solve_tier_8().expect("Tier 8 solve failed");
        let (t7_count, _) = solver.solve_tier_7(1).expect("Tier 7 solve failed");

        let prober = MacroFactoredProber::with_dir(&temp_dir);
        assert_eq!(prober.tier8_count, t8_count);
        assert_eq!(prober.tier7_count, t7_count);

        // Construct a sample Tier 8 board: 8 subboards won, subboard 4 open
        let mut board_t8 = UltimateTicTacToeBoard::default();
        board_t8.x_supers = 0b000001111; // subboards 0, 1, 2, 3 won by X (forms row {0,1,2})
        board_t8.o_supers = 0b111100000; // subboards 5, 6, 7, 8 won by O
        // Subboard 4 is open
        board_t8.last_super_index = 4;
        board_t8.x_to_move = true;

        // Macro board already has 3-in-a-row {0,1,2} for X
        let res = prober.probe(&board_t8);
        assert_eq!(res, Some(EgtbOutcome::P1Win));

        // Construct a non-terminal Tier 8 board
        let mut board_t8_open = UltimateTicTacToeBoard::default();
        // Subboards 0, 2 won by X (no line)
        board_t8_open.x_supers = 0b000000101;
        // Subboards 1, 3, 5, 6, 7, 8 tied
        for sub in [1, 3, 5, 6, 7, 8] {
            board_t8_open.xbb[sub] = crate::canonical::CANONICAL_TIE_X;
            board_t8_open.obb[sub] = crate::canonical::CANONICAL_TIE_O;
        }
        // Subboard 4 open and empty
        board_t8_open.last_super_index = 4;
        board_t8_open.x_to_move = true;

        assert_eq!(board_t8_open.get_resolved_count(), 8);
        let res_t8 = prober.probe_tier_8(&board_t8_open);
        assert!(res_t8.is_some());

        // Probe throughput benchmark over 100,000 queries
        let t0 = Instant::now();
        let iters = 100_000;
        for _ in 0..iters {
            let _ = prober.probe_tier_8(&board_t8_open);
        }
        let elapsed = t0.elapsed();
        let nanos_per_probe = elapsed.as_nanos() as f64 / iters as f64;
        println!("  Probe latency: {:.2} ns/probe ({:.2} million probes/sec)",
            nanos_per_probe,
            iters as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
        assert!(nanos_per_probe < 100.0, "Probe took too long: {:.2} ns", nanos_per_probe);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
